#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 07:43:07 2025

@author: student
"""

import networkx as nx
import pandas as pd 
##### network connected consistently???
# Erzeuge den Graph des Stromnetzes
G = nx.Graph()

print("Ist Netzwerk zusammenhängend?", nx.is_connected(G))
  
#Falls nicht: wie viele separate Teilgraphen?
if not nx.is_connected(G):
    components = list(nx.connected_components(G))
    print(f"Anzahl Teilnetze: {len(components)}")
    print("Größen der Teilnetze:", [len(c) for c in components]) 
    
    for i, comp in enumerate(components):
       comp_sorted = sorted(comp)
       print(f"\nTeilnetz {i+1}: ({len(comp)} Busse)")
       print("Beispiel-Busse:", comp_sorted[:3])
        
        
        # Kleine Teilnetze identifizieren
    small_components = [comp for comp in components if len(comp) < 10]
    small_buses = set().union(*small_components)

    print(f"\nEntferne {len(small_components)} kleine Teilnetze mit insgesamt {len(small_buses)} Bussen.")

    # Verbundene Leitungen löschen
    lines_to_remove = n.lines[
        n.lines.bus0.isin(small_buses) | n.lines.bus1.isin(small_buses)
    ].index
    n.remove("Line", list(lines_to_remove))

    # Andere Komponenten löschen, die an diesen Bussen hängen
    for comp in ["Load", "Generator", "StorageUnit", "Store", "Link"]:
        if comp.lower() + "s" in n.components.keys():
            df = getattr(n, comp.lower() + "s")
            if "bus" in df.columns:
                print('Yeah')
                print(df[df.bus.isin(small_buses)])
                idx = df[df.bus.isin(small_buses)].index
                if len(idx) > 0:
                    print(f"  → Entferne {len(idx)} {comp}s")
                    n.remove(comp, list(idx))
            elif {"bus0", "bus1"} <= set(df.columns):  # für Links oder ähnliche
                idx = df[df.bus0.isin(small_buses) | df.bus1.isin(small_buses)].index
                if len(idx) > 0:
                    print(f"  → Entferne {len(idx)} {comp}s (bus0/bus1) ")
                    n.remove(comp, list(idx))

    # Jetzt die Busse selbst löschen
    n.remove("Bus", list(small_buses))
    print("Busse gelöscht.")

    
     
     
     
#### unconnected busses????   
source_buses = n.generators.bus.unique()
# Alle erreichbaren Busse über den Netzgraphen
reachable = set()
for b in source_buses:
     reachable.update(nx.node_connected_component(G, b))
                      
# Unerreichbare Busse
unreachable = set(n.buses.index) - reachable
print("Unerreichbare Busse:", unreachable)


##if network just feasible with line extension = True 
def check_line_overloading(n):
    s_nom = n.lines.s_nom
    s_nom_opt = n.lines.s_nom_opt
    
    diff = pd.DataFrame({
        's_nom': s_nom,
        's_nom_opt': s_nom_opt
        })


    diff['dif'] = (diff['s_nom_opt'] - diff['s_nom']).round(6)
    
    print(diff['dif'].sort_values(ascending=False).head(5))
