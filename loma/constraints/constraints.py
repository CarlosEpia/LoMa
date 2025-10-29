#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:47:50 2025

@author: student
"""

    
def load_reduction_constraint_14a(n, snapshots):
    # relevant generators for load reduction
    gens_14a = n.generators.index[n.generators.carrier == "14a"]

    # for each generator create a binary variable which defines if 
    # generator is on (1) or off(0) ---> sum of this variable for 
    # each day is maxium 2 (2h usage of load reduction per day)
    for gen in gens_14a:
        p = n.model.variables["Generator-p"].loc[:, gen]  
        
        # Binary Variable per snapshot and Generator
        on = n.model.add_variables(
            name=f"on_{gen}",    
            binary=True,
            coords={"snapshot": snapshots},
            dims=("snapshot",)
        )

        p_max = n.generators.p_nom[gen]

        # Couple binary variable with power of generator

        for i, t in enumerate(snapshots):
            n.model.add_constraints(
                lhs=p[i] - p_max * on[i],
                sign="<=",
                rhs=0,
                name=f"link_on_{gen}_{t}"
            )
        
        # Sliding Window: max. 2h on per day
        for day in snapshots.to_series().dt.floor("D").unique():
            # Liste der Indizes (Positionen) für diesen Tag
            window_idx = [i for i, t in enumerate(snapshots) if t.floor("D") == day]
            
            n.model.add_constraints(
                lhs=sum(on[i] for i in window_idx),
                sign="<=",
                rhs=2,
                name=f"max2h_per_day_{gen}_{day}"
            )

    """
    Adds constraints to ensure:
    - SOC is cyclic every 24h and at the end.
    - At least 4.2 kW of EV load is directly covered by the grid.
    """
    m = n.model
    soc = m.variables["Store-e"]

    # --- 1️ SOC cyclic constraint (same as before)
    ev_stores = [s for s in n.stores.index if "Par14_Store" in s]
    for store in ev_stores:
        t0, t1 = snapshots[0], snapshots[-1]
        soc_start = soc.loc[t0, store]
        soc_end = soc.loc[t1, store]

        # Every 24h SoC reset
        for t in snapshots[::24]:
            m.add_constraints(
                soc.loc[t, store] == soc_start,
                name=f"soc_cyclic_true_at_{store}_{t}"
            )

        # Cyclic over full horizon
        m.add_constraints(soc_start == soc_end,
                          name=f"soc_cyclic_true_at_end_{store}")

    # --- 2️ Minimum grid contribution constraint (>= 4.2 kW)
    link_power = m.variables["Link-p"]

    minimum_load_par14a = 0.0042  # MW = 4.2 kW

    for link_name in n.links.index:
        if "Par14_StoreDischarge_Link" in link_name:
            # Store -> EV bus link
            ev_bus = n.links.at[link_name, "bus1"]

            # find the EV load at that EV bus
            ev_load = n.loads[n.loads.bus == ev_bus].index
            if len(ev_load) == 0:
                continue
            ev_loads = ev_load[0]

            # get EV power demand for all snapshots
            p_ev_series = n.loads_t.p_set[ev_loads].reindex(snapshots).fillna(0.0)

            # Define link variable for store discharge
            p_store_discharge = link_power.loc[snapshots, link_name]

            # Add constraint: store contribution ≤ load - 4.2 kW
            for t in snapshots:
                p_ev = p_ev_series.loc[t]
                if p_ev > minimum_load_par14a:
                    m.add_constraints(
                        p_store_discharge.loc[t] <= p_ev - minimum_load_par14a,
                        name=f"minimum_load_par14a_{link_name}_{t}"
                    )