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


#def load_shedding_constraints_ev(n, snapshots):
    """
    Each Par14-Store has own SoC-cycles,
    SOC is set to be equal to starting value each 24h (intervall can be changed) and at the last snapshot.
    Lade-/Entladeleistung pro EV auf Anteil von bev_charger_rate begrenzt.
    """
    m = n.model
    soc = m.variables["Store-e"]      # SoC-Variablen aller Stores

    # Filter all Par14-Stores
    ev_stores = [s for s in n.stores.index if "Par14_Store" in s]

    for store in ev_stores:
        # Start- und End-SoC
        t0, t1 = snapshots[0], snapshots[-1]
        soc_start = soc.loc[t0, store]
        soc_end = soc.loc[t1, store]

        # alle 24h: SoC = Startwert(100%)
        for t in snapshots[::24]:
            m.add_constraints(soc.loc[t, store] == soc_start,
                              name=f"soc_cyclic_true_at_{store}_{t}")

        # zyklisch für letzten Snapshot
        m.add_constraints(soc_start == soc_end,
                          name=f"soc_cyclic_true_at_end_{store}")
        
