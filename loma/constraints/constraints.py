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


