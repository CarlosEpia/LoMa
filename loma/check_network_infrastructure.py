#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 07:43:07 2025

@author: student
"""

import networkx as nx

##### network connected consistently???
# Erzeuge den Graph des Stromnetzes
G = n.graph()

# Ist das Netzwerk vollständig verbunden?
print("Ist Netzwerk zusammenhängend?", nx.is_connected(G))
  
#Falls nicht: wie viele separate Teilgraphen?
if not nx.is_connected(G):
     components = list(nx.connected_components(G))
     print(f"Anzahl Teilnetze: {len(components)}")
     print("Größen der Teilnetze:", [len(c) for c in components]) 
     
     
     
     
     
#### unconnected busses????   
source_buses = n.generators.bus.unique()
# Alle erreichbaren Busse über den Netzgraphen
reachable = set()
for b in source_buses:
     reachable.update(nx.node_connected_component(G, b))
                      
# Unerreichbare Busse
unreachable = set(n.buses.index) - reachable
print("Unerreichbare Busse:", unreachable)