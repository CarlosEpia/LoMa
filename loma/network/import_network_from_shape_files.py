#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 09:49:16 2025

@author: student
"""

#ToDo: Create function to add manual buses for certain lines and split lines there



import geopandas as gpd
import os
from shapely.ops import split
from shapely.geometry import MultiPoint, Point, LineString, GeometryCollection
from shapely.ops import split, linemerge, unary_union, snap
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import pypsa
from collections import defaultdict
from shapely.strtree import STRtree

from demands.household_count import count_households_per_bus





def create_gdf_from_shape(input_folder):
    """
    Creates GeoDataFrame for pypsa network components based on shapefiles

    Returns
    -------
    buses : GeoDataFrame
        contains all potential bus-components
    lines : GeoDataFrame
        contains all potential line-components

    """
    
    LV_lines = gpd.read_file(os.path.join(input_folder, "Gis NSP Kabelabschnitt Verlauf.shp"))
    HA_lines = gpd.read_file(os.path.join(input_folder, "Gis NSP HA Abschnitt Verlauf.shp"))
    HA_Bus = gpd.read_file(os.path.join(input_folder, "Gis NSP HA Kasten Position.shp"))
    distributors = gpd.read_file(os.path.join(input_folder, "old_Kabelverteiler", "Gis ST Kabelverteiler Position.shp"))
    joints = gpd.read_file(os.path.join(input_folder, "Gis NSP Muffe Position.shp"))
    MVLV_trafos = gpd.read_file(os.path.join(input_folder, "Gis ST Station Position.shp"))

    # component-type-column for distinguish the components
    LV_lines["comp_type"] = "lv_line"
    HA_lines["comp_type"] = "hc_line"
    joints["comp_type"] = joints.ART
    distributors["comp_type"] = "distributor"
    MVLV_trafos["comp_type"] = "trafo"
    HA_Bus["comp_type"] = "house_connection"
    
    ##buses
    bus_columns= ['comp_type', 'LOKATION_S', 'HAUSNUMMER', 'geometry']
    def ensure_columns(df, columns):
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan  # oder np.nan
        return df[columns]
    joints_clean = ensure_columns(joints, bus_columns)
    distributors_clean = ensure_columns(distributors, bus_columns)
    MVLV_trafos_clean = ensure_columns(MVLV_trafos, bus_columns)
    HA_Bus_clean = ensure_columns(HA_Bus, bus_columns)
    
    #secure same crs
    target_crs = "EPSG:32632"
    joints_clean = joints_clean.to_crs(target_crs)
    distributors_clean = distributors_clean.to_crs(target_crs)
    HA_Bus_clean = HA_Bus_clean.to_crs(target_crs)
    MVLV_trafos_clean = MVLV_trafos_clean.to_crs(target_crs)
    LV_lines = LV_lines.to_crs(target_crs)
    HA_lines = HA_lines.to_crs(target_crs)
    
    # combine all bus-datfarmes
    buses = pd.concat([joints_clean, distributors_clean, MVLV_trafos_clean, HA_Bus_clean], ignore_index=True)
    buses["bus_id"] = [f"bus_{i}" for i in range(len(buses))]
    buses = buses.reset_index(drop=True)
    
    
    ##lines
    line_columns = ['comp_type', 'KABELTYP', 'geometry' ]
    #combine all line-dataframes
    lines = pd.concat([LV_lines[line_columns], HA_lines[line_columns]], ignore_index=True)
    lines["line_id"] = [f"line_{i}" for i in range(len(lines))]
    lines = lines.reset_index(drop=True)
    
    return buses, lines

###LV-lines
def cut_line_at_points(line, cutting_points):
    """
    Split line at each given point (points have to be on the line)
    """
    # Calculate position along the line
    distances = [line.project(point) for point in cutting_points]
    distances = sorted(set(distances))  # sort points/delete duplicated points

    # Add starting and endpoint to the split-line
    if 0.0 not in distances:
        distances = [0.0] + distances
    if line.length not in distances:
        distances.append(line.length)
 
    # interpolate points
    points = [line.interpolate(distance) for distance in distances]

    # create segments of each line
    segments = []
    for i in range(len(points) - 1):
        segment = cut_line_between_distances(line, distances[i], distances[i + 1])
        if segment is not None:
            segments.append(segment)

    return segments

def cut_line_between_distances(line, start_distance, end_distance, tolerance=0.01):
    """
    Extracts geometry of one segment of splitted line from given end and start-distance
    """
    if end_distance - start_distance < tolerance:
        return None

    coords = [line.interpolate(start_distance).coords[0]]

    for point in line.coords:
        point_distance = line.project(Point(point))
        if start_distance < point_distance < end_distance:
            coords.append(point)

    coords.append(line.interpolate(end_distance).coords[0])

    return LineString(coords)


def split_lines_on_joints(lines, buses, tolerance=0.1):
    """
    Splits LV_lines at joint buses (snapped to the line) and returns a new GeoDataFrame with the splitted lines.
    """
    LV_lines = lines[lines.comp_type=='lv_line']
    HA_lines = lines[lines.comp_type=='hc_line']
    split_lines = []

    for idx, row in lines.iterrows():
        #print(f"Spliting line {idx}")
        line_geom = row.geometry

        # filter relevant buses (just split at the joint-buses)
        joint_buses = buses[buses.comp_type.isin(['Hausanschlußmuffe', 'Verbindungsmuffe', 'Endmuffe',
                                              'Übergangsmuffe', 'Reparaturmuffe', 'vorverlegtes Ende'])].copy()

        joint_buses['distance'] = joint_buses.geometry.apply(lambda p: line_geom.distance(p))
        near_joints = joint_buses[joint_buses['distance'] < tolerance]

        if near_joints.empty:
            split_lines.append({
                'geometry': line_geom,
                'comp_type': row.comp_type,
            })
        else:
            snapped_points = list(near_joints.geometry)
            split_segments = cut_line_at_points(line_geom, snapped_points)

            for segment in split_segments:
                split_lines.append({
                    'geometry': segment,
                    'comp_type': row.comp_type,
                })

    split_lines_df = pd.DataFrame(split_lines)
    split_lines_gdf = gpd.GeoDataFrame(split_lines_df, geometry='geometry', crs=LV_lines.crs)
    lines = split_lines_gdf.reset_index(drop=True)
    lines['line_id'] = ['line_' + str(i) for i in range(len(lines))]

    return lines

 

def snap_joint_buses_to_lines(lines, buses, tolerance=0.1):
    """
    Snaps all 'joint'-buses directly on the closest line, if distance < tolerance.
    Secures that coords from buses are directly on the line to avoid skipping inaccurate data.
    """
    joint_buses = buses[buses.comp_type.isin(['Hausanschlußmuffe', 'Verbindungsmuffe', 'Endmuffe',
                                          'Übergangsmuffe', 'Reparaturmuffe', 'vorverlegtes Ende'])].copy()

    other_buses = buses[~buses.comp_type.isin(['Hausanschlußmuffe', 'Verbindungsmuffe', 'Endmuffe',
                                           'Übergangsmuffe', 'Reparaturmuffe', 'vorverlegtes Ende'])].copy()

    
    LV_lines = lines[lines.comp_type=='lv_line']
    line_geometries = list(LV_lines.geometry)
    str_tree = STRtree(line_geometries)
    
    # Optional: falls du Referenz zu originalem DataFrame brauchst
  #  geom_to_line = {geom: line for geom, line in zip(line_geometries, lines.itertuples())}
    
    snapped_indices = []
    
    for idx, bus in joint_buses.iterrows():
        nearest_idx = str_tree.nearest(bus.geometry)
        nearest_geom = line_geometries[nearest_idx] 
        min_dist = bus.geometry.distance(nearest_geom)
        
        if 0 < min_dist <= tolerance :
            projected_distance = nearest_geom.project(bus.geometry)
            snapped_point = nearest_geom.interpolate(projected_distance)
            print(f"{bus.bus_id} is projected to the corresponding line")
            joint_buses.at[idx, 'geometry'] = snapped_point
            snapped_indices.append(idx)
            
    buses_updated = pd.concat([joint_buses, other_buses]).sort_index()   
    
    return buses_updated


#####house connection buses
def find_connected_HA_line(current_point, HA_lines, visited_lines, tolerance=0.1):
    """
    Searches for another HA line connected to the current point 
    that has not yet been visited.
    
    Parameters:
    - current_point: Shapely Point to check connections from.
    - HA_lines: GeoDataFrame of house connection lines.
    - visited_lines: Set of already visited line indices.
    - tolerance: Maximum distance to consider points as connected.

    Returns:
    - Index of the next HA line and the opposite endpoint as the new reference point.
    """
    for idx, line in HA_lines.iterrows():
        if idx in visited_lines:
            continue

        start_point = Point(line.geometry.coords[0])
        end_point = Point(line.geometry.coords[-1])

        if current_point.distance(start_point) < tolerance:
            return idx, end_point
        elif current_point.distance(end_point) < tolerance:
            return idx, start_point

    return None, None  # No further connection found


def assign_house_connections_to_joints(lines, buses, tolerance=0.1):
    """
    Assigns each house connection bus to the nearest joint bus, 
    following the connected HA lines starting from the end opposite the house connection.

    Parameters:
    - HA_lines: GeoDataFrame of house connection lines.
    - buses: GeoDataFrame of all buses with 'comp_type' attribute.
    - tolerance: Maximum distance to consider connections between lines.

    Returns:
    - mapping: Dictionary mapping house connection bus indices to joint bus indices.
    """
    HA_lines = lines[lines.comp_type=='HA_Line']
    joint_buses = buses[buses.comp_type == 'joint'].copy()
    house_buses = buses[buses.comp_type == 'house_connection'].copy()

    mapping = {}

    for idx, house_bus in house_buses.iterrows():
        # Find the closest HA line to the house connection bus
        min_dist = float('inf')
        closest_line_idx = None

        for line_idx, line in HA_lines.iterrows():
            dist = line.geometry.distance(house_bus.geometry)
            if dist < min_dist:
                min_dist = dist
                closest_line_idx = line_idx

        if closest_line_idx is None:
            print(f"No HA line found for house connection {idx}")
            continue

        # Determine the end of the line opposite to the house connection bus
        line = HA_lines.loc[closest_line_idx]
        start_point = Point(line.geometry.coords[0])
        end_point = Point(line.geometry.coords[-1])

        if house_bus.geometry.distance(start_point) < house_bus.geometry.distance(end_point):
            current_point = end_point
        else:
            current_point = start_point

        visited_lines = {closest_line_idx}

        # Follow connected HA lines until no further connection is found
        while True:
            next_line_idx, next_point = find_connected_HA_line(current_point, HA_lines, visited_lines, tolerance)
            if next_line_idx is None:
                break
            visited_lines.add(next_line_idx)
            current_point = next_point

        # Find the nearest joint bus to the final reference point
        joint_buses['distance'] = joint_buses.geometry.apply(lambda p: p.distance(current_point))
        closest_joint_idx = joint_buses['distance'].idxmin()

        # Map the house connection bus to the closest joint bus
        mapping[idx] = closest_joint_idx

    return mapping

def map_load_bus_to_network_bus(buses, lines):
    """
    Adds house connection assignments and filters out buses that do not lie on any line.
    Buses with comp_type 'house connection' are filtered out beforehand.

    Parameters:
    - buses: GeoDataFrame containing all buses (with 'bus_id', 'geometry', 'comp_type')
    - lines: GeoDataFrame of lines

    Returns:
    - Filtered GeoDataFrame containing buses that have house connections
      or lie on any line, excluding buses with comp_type 'house connection' from input
    """
    # First filter out buses with comp_type 'house connection'
    buses_filtered = buses[buses['comp_type'] != 'house_connection']

    # Get assignments from house connections to network buses (based on original buses)
    mapping = assign_house_connections_to_joints(lines, buses)

    # Group house connections per network bus
    grouped_mapping = defaultdict(lambda: {'house_connection_bus_ids': [], 'house_connection_geoms': []})
    for house_bus_id, network_bus_id in mapping.items():
        house_bus = buses.loc[buses.index == house_bus_id].iloc[0]
        grouped_mapping[network_bus_id]['house_connection_bus_ids'].append(house_bus.bus_id)
        grouped_mapping[network_bus_id]['house_connection_geoms'].append(house_bus.geometry)

    # Add house connection info (empty lists if no assignment)
    buses_filtered = buses_filtered.copy()
    buses_filtered['house_connection_bus_ids'] = buses_filtered.index.map(
        lambda idx: grouped_mapping[idx]['house_connection_bus_ids'] if idx in grouped_mapping else []
    )
    buses_filtered['house_connection_geoms'] = buses_filtered.index.map(
        lambda idx: grouped_mapping[idx]['house_connection_geoms'] if idx in grouped_mapping else []
    )

    # Combine all line geometries for fast distance queries
    all_lines_geom = lines.unary_union
    def on_line(geom, tolerance=1e-6):
        return all_lines_geom.distance(geom) < tolerance
    
    # Keep only buses that lie on any line
    filtered_buses = buses_filtered[
        buses_filtered.geometry.apply(on_line)
    ]

    return filtered_buses



## fix line infrastructure
def merge_unconnected_lines(lines, buses, tolerance=0.1):
    """
    Merge lines where line endpoints have no nearby bus but are close to endpoints of other lines.
    
    Parameters:
    - lines: GeoDataFrame with columns ['geometry', 'comp_type', 'index', 'line_id']
    - buses: GeoDataFrame with bus geometries
    - distance_threshold: max distance to consider points connected
    
    Returns:
    - GeoDataFrame with merged lines
    """
    bus_coords = [(bus.geometry.x, bus.geometry.y) for _, bus in buses.iterrows()]
    bus_tree = cKDTree(bus_coords) if bus_coords else None

    lines = lines.copy()
    used_lines = set()
    merged_lines = []

    for idx, line in lines.iterrows():
        if idx in used_lines:
            continue

        current_geom = line.geometry
        coords = list(current_geom.coords)
        merged_geom = current_geom
        merged_with_any = False

        # Check both endpoints: start and end
        for endpoint_index in [0, -1]:
            point = Point(coords[endpoint_index])
            bus_nearby = False
            if bus_tree:
                dist_to_bus, _ = bus_tree.query([point.x, point.y])
                if dist_to_bus < tolerance:
                    bus_nearby = True

            if bus_nearby:
                # if bus is nearby dont merge the lines
                continue

            # if no bus nearby search for other line nearby
            for other_idx, other_line in lines.iterrows():
                if other_idx == idx or other_idx in used_lines:
                    continue

                other_geom = other_line.geometry
                other_coords = list(other_geom.coords)

                # check start and end-point for interesection with current line
                for other_point in [Point(other_coords[0]), Point(other_coords[-1])]:
                    if point.distance(other_point) < tolerance:
                        # Linien zusammenführen
                        # Je nach Position der Punkte muss Reihenfolge der Koordinaten beachtet werden
                        if endpoint_index == 0 and other_point.equals(Point(other_coords[-1])):
                            # current start == other end → other + current
                            merged_geom = LineString(other_coords + coords)
                        elif endpoint_index == 0 and other_point.equals(Point(other_coords[0])):
                            # current start == other start → andere Linie umdrehen + current
                            merged_geom = LineString(other_coords[::-1] + coords)
                        elif endpoint_index == -1 and other_point.equals(Point(other_coords[0])):
                            # current end == other start → current + other
                            merged_geom = LineString(coords + other_coords)
                        elif endpoint_index == -1 and other_point.equals(Point(other_coords[-1])):
                            # current end == other end → current + umgekehrte andere
                            merged_geom = LineString(coords + other_coords[::-1])
                        else:
                            print(f'#### no connection line found for merging line {idx}')
                            continue  #no connection found

                        used_lines.add(other_idx)
                        merged_with_any = True
                        break

                if merged_with_any:
                    break
            if merged_with_any:
                coords = list(merged_geom.coords)

        used_lines.add(idx)
        merged_lines.append({'geometry': merged_geom, 'comp_type': line.comp_type, 'index': line.index, 'line_id': line.line_id})

    merged_lines_gdf = gpd.GeoDataFrame(merged_lines, geometry='geometry', crs=lines.crs)
    return merged_lines_gdf



def get_nearest_bus(point, bus_tree, buses_df):
    """
    Returns closest bus to given start-/end-point of a line
    """
    dist, idx = bus_tree.query([point.x, point.y])
    return buses_df.loc[idx, "bus_id"], dist



###creating pypsa grid 
def import_grid_infrastructure(n, buses, lines):
    """    
    Based on exported shapefiles recreate the grid infrastructure as pysa_network

    Parameters
    ----------
    n : Pypsa_network

    buses : GeoDataFrame
        contains all potential bus-components
    lines : GeoDataFrame
        contains all potential line-components

    Returns
    -------
    None.

    """

    
    # import buses
    for _, row in buses.iterrows():
        n.add("Bus",
          row["bus_id"],
          x=row.geometry.x,
          y=row.geometry.y)
        n.buses.at[row["bus_id"], 'comp_type'] = row['comp_type']
        n.buses.at[row["bus_id"], 'house_count'] = row['house_count']
        n.buses.at[row["bus_id"], 'geom'] = row['geometry']
        
    # import LV lines
    for idx, row in lines.iterrows():
        line_geom = row.geometry
        start_point = Point(line_geom.coords[0])
        end_point = Point(line_geom.coords[-1])

        #relevant_buses = buses[buses.comp_type.isin(['joint', 'distributor', 'trafo'])]
        #relevant_buses = relevant_buses.reset_index(drop=True)
        relevant_buses = buses
    
        bus_coords = np.array([[geom.x, geom.y] for geom in relevant_buses.geometry])
        bus_tree = cKDTree(bus_coords)
    
        # Starte mit allen relevanten Bussen
        bus0, dist0 = get_nearest_bus(start_point, bus_tree, relevant_buses)
        bus1, dist1 = get_nearest_bus(end_point, bus_tree, relevant_buses)

        # if closest bus not directly next to the start/endpoint connect trafo/distributor bus
        if 10 > dist0 > 0.1 and row['comp_type']=='lv_line':
            #print('##### line connecting to trafo/dist ############', 'Line:', idx)
            traf_dist_buses = buses[buses.comp_type.isin(['distributor', 'trafo'])].reset_index(drop=True)
            traf_dist_coords = np.array([[geom.x, geom.y] for geom in traf_dist_buses.geometry])
            traf_dist_tree = cKDTree(traf_dist_coords)
            bus0, _ = get_nearest_bus(start_point, traf_dist_tree, traf_dist_buses)
    
        if 10 > dist1 > 0.1 and row['comp_type']=='lv_line':
            #print('##### line connecting to trafo/dist ############', 'Line:', idx)
            traf_dist_buses = buses[buses.comp_type.isin(['distributor', 'trafo'])].reset_index(drop=True)
            traf_dist_coords = np.array([[geom.x, geom.y] for geom in traf_dist_buses.geometry])
            traf_dist_tree = cKDTree(traf_dist_coords)
            bus1, _ = get_nearest_bus(end_point, traf_dist_tree, traf_dist_buses)
    
        length_km = line_geom.length / 1000
    
        n.add("Line", row['line_id'], bus0=bus0, bus1=bus1, carrier='AC',
              length=length_km, r=0.3, x=0.05, s_nom=100e6)
        n.lines.at[row["line_id"], 'comp_type'] = row['comp_type']
        n.lines.at[row["line_id"], 'geom'] = row['geometry']
        lines.at[idx, 'bus_0'] = bus0
        lines.at[idx, 'bus_1'] = bus1
        
        ##for validating
        lines.at[idx, 'bus_0'] = bus0
        lines.at[idx, 'bus_1'] = bus1
        lines.at[idx, 'line_id'] = row['line_id']
    
    
    #add generator at trafo
    trafo_buses = n.buses[n.buses.comp_type=='trafo'] 
    for idx, bus in trafo_buses.iterrows():
        n.add('Generator', 
              f'gen_{idx}',
              bus=bus.name,
              p_nom=1e6, 
              marginal_cost=100)   
    
    
    #add carriers
    carriers = ["AC", "CTS", "industrial", "household"]
    for c in carriers:
        n.add("Carrier", c)
    
    

def open_LV_circle(n, lv_line_idx):
    if lv_line_idx in n.lines.index:
        n.lines.drop(lv_line_idx, inplace=True)
        print(f"Leitung '{lv_line_idx}' wurde entfernt.")
    else:
        print(f"Leitung '{lv_line_idx}' nicht gefunden im Netzwerk.")

    return n


             
def fix_grid_infrastructure(n):
    #delete loop lines
    loop_lines = n.lines[n.lines.bus0==n.lines.bus1]
    print(f" ⚠️ Warning: Following lines have same bus0 and bus1: {loop_lines[['bus0', 'bus1']]}")
    print("⚠️ Warning: This lines will be deleted from the network")
    n.mremove("Line", loop_lines.index.tolist())
    
    #delete unconnected buses
    con_buses = pd.concat([n.lines.bus0, n.lines.bus1]).unique().tolist()
    uncon_buses = n.buses[~n.buses.index.isin(con_buses)]
    print(f" ⚠️ Warning: Following buses are not connected to the network: {uncon_buses}")
    print("⚠️ Warning: This buses and connected components will be deleted from the network")
    n.mremove("Bus", uncon_buses.index.tolist())
    
    components_to_clean = ["Generator"]#, "Load"]  # list extendable

    for comp in components_to_clean:

        df = n.df(comp)
        to_remove = df[df.bus.isin(uncon_buses.index)].index.tolist()
        if to_remove:
            print(f"⚠️ Removing {len(to_remove)} {comp}(s) connected to unconnected buses")
            n.mremove(comp, to_remove)
    
    
    
def import_ev_chargers(n):
    
    #use shapefile for ladesäulen
    return
    



def create_pypsa_network(shape_files_folder, q_households_folder):
    n = pypsa.Network()
    time_index = pd.date_range('2023-01-01', periods=8760, freq='h')
    n.snapshots = time_index
    
    buses, lines = create_gdf_from_shape(shape_files_folder)
    buses = snap_joint_buses_to_lines(lines, buses)
    buses = count_households_per_bus(buses, q_households_folder)
   
    #final_load_buses = map_load_bus_to_network_bus(buses, lines)
    #network_buses = buses[buses.comp_type.isin(['trafo', 'distributor'])]
    #all_network_buses = pd.concat([final_load_buses, network_buses])
    split_lines = split_lines_on_joints(lines, buses)
    #merged_lines = merge_unconnected_lines(split_lines, buses)
    import_grid_infrastructure(n, buses, split_lines) 
    fix_grid_infrastructure(n)
    n = open_LV_circle(n, 'line_187')
    
    return n
    
    
    

    
    