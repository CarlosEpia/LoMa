#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detects and resolves meshed (looped) sections of the LV/MV grid so it stays radial for PyPSA's power flow."""
import networkx as nx
import pandas as pd
from collections import deque



def demesh_lv_network_filtered(n):
    """Remove LV lines that create meshes by keeping, for each LV bus, only
    the shortest path to its nearest MV-connected transformer (multi-source
    Dijkstra) and dropping any line that would connect two different
    transformer "territories"."""
    # 1. Find buses that MV lines are connected to
    mv_lines_mask = n.lines.comp_type == 'mv_line'
    mv_connected_buses = set(n.lines.loc[mv_lines_mask, 'bus0']).union(
                         set(n.lines.loc[mv_lines_mask, 'bus1']))

    # 2. Filter transformers: bus0 must be connected to an MV bus
    valid_trafos = n.transformers[n.transformers.bus0.isin(mv_connected_buses)]
    trafo_lv_buses = list(valid_trafos.bus1.unique())

    print(f"Valid Trafos (with MV connections): {len(trafo_lv_buses)}")

    g = nx.Graph()
    lv_mask = n.lines.comp_type.isin(['lv_line', 'hc_line'])
    for _, row in n.lines[lv_mask].iterrows():
        g.add_edge(row.bus0, row.bus1, key=row.name, length=row.length)

    # Only use transformers that also exist in the LV graph
    sources = [b for b in trafo_lv_buses if b in g]

    if not sources:
        print("No Trafo found in LV-grid!")
        return []

    # Multi-source Dijkstra
    distances, paths = nx.multi_source_dijkstra(g, sources, weight='length')
    bus_to_trafo = {bus: path[0] for bus, path in paths.items()}

    lines_to_remove = []
    for _, row in n.lines[lv_mask].iterrows():
        b0, b1 = row.bus0, row.bus1
        if b0 in bus_to_trafo and b1 in bus_to_trafo:
            if bus_to_trafo[b0] != bus_to_trafo[b1]:
                lines_to_remove.append(row.name)

    n.remove("Line", lines_to_remove)
    print(f"Grid meshes are deleted: {len(lines_to_remove)} lines removed.")



def analyze_lv_feeding_with_boundaries(n):
    """For each LV bus, find which LV/MV transformer(s) it is reachable from
    via a breadth-first search that treats other transformers as boundaries.
    Returns a Series mapping bus -> list of feeding transformer buses (more
    than one entry means the bus is still meshed)."""
    # 1. Build graph (LV level only)
    g = nx.Graph()
    g.add_nodes_from(n.buses.index)
    lv_mask = n.lines.comp_type.isin(['lv_line', 'hc_line'])
    edges = list(zip(n.lines.loc[lv_mask, 'bus0'], n.lines.loc[lv_mask, 'bus1']))
    g.add_edges_from(edges)

    # 2. Transformer buses as starting points and boundaries
    lvmv_trafos = n.transformers[n.transformers.comp_type != 'trafo_HV']
    trafo_lv_buses = set(lvmv_trafos.bus1.unique())

    # Dictionary: bus -> set of transformers it is reachable from
    bus_feeding_map = {bus: set() for bus in n.buses.index}

    # 3. Start a separate search for each transformer
    for root_trafo in trafo_lv_buses:
        if root_trafo not in g:
            continue

        # standard BFS
        queue = deque([root_trafo])
        visited = {root_trafo}

        while queue:
            current_bus = queue.popleft()
            bus_feeding_map[current_bus].add(root_trafo)

            for neighbor in g.neighbors(current_bus):
                if neighbor not in visited:
                    # Key check: if the neighbor is itself a transformer, mark
                    # it as visited (boundary) but don't add it to the queue,
                    # so the search doesn't continue past it.
                    if neighbor in trafo_lv_buses:
                        bus_feeding_map[neighbor].add(root_trafo)
                        visited.add(neighbor)
                    else:
                        visited.add(neighbor)
                        queue.append(neighbor)

    # convert to lists for readability
    result = {k: list(v) for k, v in bus_feeding_map.items() if v}

    con = pd.Series(result, name="connected_trafo_lv_buses")

    # check bus connections
    meshed_buses = con[con.apply(len) > 1]
    print(f"\nNumber of meshed buses: {len(meshed_buses)}")

    if not meshed_buses.empty:
        print("Example of a meshed bus:")
        print(meshed_buses.iloc[0])

    # find buses with no transformer at all (isolated islands)
    dead_buses = con[con.apply(len) == 0]
    print(f"Number of unsupplied buses: {len(dead_buses)}")

    return con

def assign_lv_grid_ids(n, con):
    """
    Groups buses by their transformer connections and assigns a unique
    lv_grid_id to each group. MV buses are left empty (NA).
    """
    # 1. Transformer combinations as sorted tuples (for uniqueness)
    groups = con.apply(lambda x: tuple(sorted(x)))

    # 2. Assign a unique ID to each transformer combination
    unique_groups = groups.unique()
    group_to_id = {group: i + 1 for i, group in enumerate(unique_groups)}

    # 3. Assign to n.buses
    # .map() automatically leaves buses that aren't in 'groups' as NaN
    n.buses["lv_grid_id"] = groups.map(group_to_id)

    # 4. Fix dtype: use "Int64" (capital I!), which allows integers AND
    # 'NA' (empty fields)
    n.buses["lv_grid_id"] = n.buses["lv_grid_id"].astype("Int64")

    print(f"Assignment complete: {len(unique_groups)} LV grids identified.")

    return n

def avoid_meshes_in_network(n):
      """Remove LV meshes from the network and (re-)assign lv_grid_id per bus."""
      demesh_lv_network_filtered(n)
      con = analyze_lv_feeding_with_boundaries(n)
      n = assign_lv_grid_ids(n, con)

      return n
      



