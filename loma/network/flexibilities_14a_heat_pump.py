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
                  marginal_cost=200,   # high value, that generator is used only if line capacity is not enough
                  overwrite=True
                  )
            
    return n
    
    
'''
def insert_heat_pump_flexibilities_14a(n):
   

    # Alle steuerbaren Lasten identifizieren (Heat-Pumps, ggf. EV-Ladungen)
    flex_loads = n.loads[n.loads.index.str.contains("heat|EV")]

    # Anzahl steuerbarer Verbraucher pro Bus ermitteln
    n_steuVE_per_bus = flex_loads.groupby("bus").size()

    # Gleichzeitigkeitsfaktor definieren
    

    for idx, row in flex_loads.iterrows():
        bus = row.bus
        n_steuVE = n_steuVE_per_bus[bus]

        # Basis-Minimalbelastung
        P_min_base = 0.0042  # 4.2 kW
        # 14a-Mindestlast inkl. Gleichzeitigkeitsfaktor
        P_min_14a = P_min_base + (n_steuVE - 1) * gzf(n_steuVE) * P_min_base

        # maximale Generatorleistung = Last - P_min_14a (nicht <0)
        p_gen_14a = np.maximum(n.loads_t.p_set[idx] - P_min_14a, 0)
        p_max = p_gen_14a.max()

        if p_max > 0:
            n.add("Generator",
                  name=f"Gen_14a_{idx}",
                  bus=bus,
                  carrier="14a",
                  p_nom=p_max,
                  p_max_pu=p_gen_14a / p_gen_14a.max(),  # normierte Zeitreihe
                  marginal_cost=10000,  # nur bei Engpässen genutzt
                  overwrite=True
                  )

    return n


def gzf(n_steuVE):
    if n_steuVE == 2: return 0.8
    elif n_steuVE == 3: return 0.75
    elif n_steuVE == 4: return 0.7
    elif n_steuVE == 5: return 0.65
    elif n_steuVE == 6: return 0.6
    elif n_steuVE == 7: return 0.55
    elif n_steuVE == 8: return 0.5
    else: return 0.45
'''