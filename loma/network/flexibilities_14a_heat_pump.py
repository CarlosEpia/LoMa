#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:31:02 2025

@author: student
"""
import numpy as np


def insert_heat_pump_flexibilities_14a(n):
    
    heat_loads = n.loads[n.loads.index.str.contains("heat")] 

    for idx, row in heat_loads.iterrows():
        bus = row.bus
 
        max_load_allowed = 0.0042  # 4.2kW maximum reduction of the laod
    
        # max capacity of generator: load_timeseries - 4.2 kW (not < 0)
        p_gen_14a = np.maximum(n.loads_t.p_set[idx] - max_load_allowed, 0)
        p_max = p_gen_14a.max()
        if p_max > 0:       
            n.add("Generator",
                  name=f"Gen_14a_{idx}",
                  bus=bus,  
                  carrier = "14a",
                  p_nom=p_max,
                  p_max_pu=p_gen_14a / p_gen_14a.max(),
                  marginal_cost=10000,   # high value, that generator is used only if line capacity is not enough
                  overwrite=True
                  )
            
    return n
    
    