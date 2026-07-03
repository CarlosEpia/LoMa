#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detects and resolves meshed (looped) sections of the LV/MV grid so it stays radial for PyPSA's power flow."""
import networkx as nx
import pandas as pd
from collections import deque



def demesh_lv_network_filtered(n):
    # --- NEU: Valide MV-verbundene Trafos finden ---
    # 1. Busse finden, an denen MV-Leitungen hängen
    mv_lines_mask = n.lines.comp_type == 'mv_line'
    mv_connected_buses = set(n.lines.loc[mv_lines_mask, 'bus0']).union(
                         set(n.lines.loc[mv_lines_mask, 'bus1']))

    # 2. Trafos filtern: bus0 muss an einem MV-Bus hängen
    valid_trafos = n.transformers[n.transformers.bus0.isin(mv_connected_buses)]
    trafo_lv_buses = list(valid_trafos.bus1.unique())
    
    print(f"Valid Trafos (with MV connections): {len(trafo_lv_buses)}")

    g = nx.Graph()
    lv_mask = n.lines.comp_type.isin(['lv_line', 'hc_line'])
    for _, row in n.lines[lv_mask].iterrows():
        g.add_edge(row.bus0, row.bus1, key=row.name, length=row.length)
    
    # Nur Trafos nehmen, die auch im LV-Graph existieren
    sources = [b for b in trafo_lv_buses if b in g]

    if not sources:
        print("No Trafo found in LV-grid!")
        return []

    # Multi-Source Dijkstra
    distances, paths = nx.multi_source_dijkstra(g, sources, weight='length')
    bus_to_trafo = {bus: path[0] for bus, path in paths.items()}

    lines_to_remove = []
    for _, row in n.lines[lv_mask].iterrows():
        b0, b1 = row.bus0, row.bus1
        if b0 in bus_to_trafo and b1 in bus_to_trafo:
            if bus_to_trafo[b0] != bus_to_trafo[b1]:
                lines_to_remove.append(row.name)

    n.remove("Line", lines_to_remove)
    print(f"Grid meshes are deleted: {len(lines_to_remove)} Leitungen entfernt.")
    


def analyze_lv_feeding_with_boundaries(n):
    # 1. Graph bauen (nur LV-Ebene)
    g = nx.Graph()
    g.add_nodes_from(n.buses.index)
    lv_mask = n.lines.comp_type.isin(['lv_line', 'hc_line'])
    edges = list(zip(n.lines.loc[lv_mask, 'bus0'], n.lines.loc[lv_mask, 'bus1']))
    g.add_edges_from(edges)
    
    # 2. Trafo-Busse als Startpunkte und Barrieren
    lvmv_trafos= n.transformers[n.transformers.comp_type!='trafo_HV']
    trafo_lv_buses = set(lvmv_trafos.bus1.unique())
    
    # Dictionary: Bus -> Liste der erreichten Trafos
    bus_feeding_map = {bus: set() for bus in n.buses.index}
    
    # 3. Für jeden Trafo eine eigene Suche starten
    for root_trafo in trafo_lv_buses:
        if root_trafo not in g:
            continue
            
        # Standard BFS
        queue = deque([root_trafo])
        visited = {root_trafo}
        
        while queue:
            current_bus = queue.popleft()
            bus_feeding_map[current_bus].add(root_trafo)
            
            for neighbor in g.neighbors(current_bus):
                if neighbor not in visited:
                    # Der entscheidende Check:
                    # Wenn der Nachbar ein Trafo ist, markieren wir ihn als besucht (Endpunkt),
                    # fügen ihn aber NICHT der Queue hinzu, um nicht dahinter weiterzusuchen.
                    if neighbor in trafo_lv_buses:
                        bus_feeding_map[neighbor].add(root_trafo)
                        visited.add(neighbor)
                    else:
                        visited.add(neighbor)
                        queue.append(neighbor)

    # In Listen umwandeln für die Übersichtlichkeit
    result = {k: list(v) for k, v in bus_feeding_map.items() if v}
    
    con = pd.Series(result, name="connected_trafo_lv_buses")
    
    #check bus connections
    meshed_buses = con[con.apply(len) > 1]
    print(f"\nAnzahl vermaschter Busse: {len(meshed_buses)}")

    if not meshed_buses.empty:
        print("Beispiel für einen vermaschten Bus:")
        print(meshed_buses.iloc[0])

    # 2. Finde Busse, die GAR KEINEN Trafo haben (Isolierte Inseln)
    dead_buses = con[con.apply(len) == 0]
    print(f"Anzahl unversorgter Busse: {len(dead_buses)}")
    
    return con

def assign_lv_grid_ids(n, con):
    """
    Gruppiert Busse nach ihren Trafo-Anschlüssen und weist 
    eindeutige IDs zu. MV-Busse bleiben leer (NA).
    """
    # 1. Trafo-Kombinationen als sortierte Tupel (für Eindeutigkeit)
    groups = con.apply(lambda x: tuple(sorted(x)))
    
    # 2. Eindeutige IDs für jede Trafo-Kombination vergeben
    unique_groups = groups.unique()
    group_to_id = {group: i + 1 for i, group in enumerate(unique_groups)}
    
    # 3. Zuweisung zu n.buses
    # .map() lässt Busse, die nicht in 'groups' sind, automatisch als NaN stehen
    n.buses["lv_grid_id"] = groups.map(group_to_id)
    
    # 4. Datentyp-Korrektur: Nutze "Int64" (Großes I!)
    # Dieser Pandas-Typ erlaubt Ganzzahlen UND 'NA' (leere Felder)
    n.buses["lv_grid_id"] = n.buses["lv_grid_id"].astype("Int64")
    
    print(f"Zuweisung abgeschlossen: {len(unique_groups)} LV-Netze identifiziert.")
    
    return n

def avoid_meshes_in_network(n):
      
      demesh_lv_network_filtered(n)
      con = analyze_lv_feeding_with_boundaries(n)
      n = assign_lv_grid_ids(n, con)
      
      return n
      



