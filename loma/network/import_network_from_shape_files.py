#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 09:49:16 2025

@author: student
"""

#ToDo: Create function to add manual buses for certain lines and split lines there 

import os
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from shapely.ops import linemerge, unary_union

from loma.demands.import_hp_demand import check_heat_pumps
from loma.demands.household_count import count_households_per_bus_input_file
from loma.demands.household_count import count_households_per_bus_census_data



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
    MV_lines = gpd.read_file(os.path.join(input_folder, "Gis MSP Kabelabschnitt Verlauf.shp"))
    HA_lines = gpd.read_file(os.path.join(input_folder, "Gis NSP HA Abschnitt Verlauf.shp"))
    HA_Bus = gpd.read_file(os.path.join(input_folder, "Gis NSP HA Kasten Position.shp"))
    distributors = gpd.read_file(os.path.join(input_folder, "Gis ST Kabelverteiler Position.shp"))
    joints = gpd.read_file(os.path.join(input_folder, "Gis NSP Muffe Position.shp"))
    MVLV_trafos = gpd.read_file(os.path.join(input_folder, "Gis ST Station Position.shp"))
    
    #rename columns to generalize the names
    for df in [HA_Bus, distributors, joints]:
        df.rename(columns= {'LOKATION_S': 'Straße', 'HAUSNUMMER': 'Hausnummer'}, inplace =True)
    for df in [MVLV_trafos]:
        df.rename(columns= {'LOKATION_S': 'Straße', 'HAUSNUMMER': 'Hausnummer', 'TRAFOBELAS': 's_nom'}, inplace =True)
    
    
    # component-type-column for distinguish the components
    LV_lines["comp_type"] = "lv_line"
    MV_lines["comp_type"] = "mv_line"
    HA_lines["comp_type"] = "hc_line"
    joints["comp_type"] = joints.ART
    distributors["comp_type"] = "distributor"
    MVLV_trafos["comp_type"] = "trafo"
    HA_Bus["comp_type"] = "house_connection"
    
    ##buses
    bus_columns = ['comp_type', 'Straße', 'Hausnummer', 'geometry']
    trafo_columns = ['comp_type', 'Straße', 'Hausnummer', 's_nom', 'geometry']
    def ensure_columns(df, columns):
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan  # oder np.nan
        return df[columns]
    joints_clean = ensure_columns(joints, bus_columns)
    distributors_clean = ensure_columns(distributors, bus_columns)
    MVLV_trafos_clean = ensure_columns(MVLV_trafos, trafo_columns)
    HA_Bus_clean = ensure_columns(HA_Bus, bus_columns)
    
    #secure same crs
    target_crs = "EPSG:32632"
    joints_clean = joints_clean.to_crs(target_crs)
    distributors_clean = distributors_clean.to_crs(target_crs)
    HA_Bus_clean = HA_Bus_clean.to_crs(target_crs)
    MVLV_trafos_clean = MVLV_trafos_clean.to_crs(target_crs)
    LV_lines = LV_lines.to_crs(target_crs)
    MV_lines = MV_lines.to_crs(target_crs)
    HA_lines = HA_lines.to_crs(target_crs)
    
    # combine all bus-datfarmes
    buses = pd.concat([joints_clean, distributors_clean, MVLV_trafos_clean, HA_Bus_clean], ignore_index=True)
    buses["bus_id"] = [f"bus_{i}" for i in range(len(buses))]
    buses = buses.reset_index(drop=True)   
    
    ##lines
    MV_lines['KABELTYP'] = None    ###Delete when KABElTYP is part of shape file attributes
    line_columns = ['comp_type', 'KABELTYP', 'geometry']  
    #combine all line-dataframes
    lines = pd.concat([LV_lines[line_columns], MV_lines[line_columns], HA_lines[line_columns]], ignore_index=True)
    lines["line_id"] = [f"line_{i}" for i in range(len(lines))]
    lines = lines.reset_index(drop=True)
    
    return buses, lines


def merge_connected_mv_lines(lines, tolerance=0.001):
    """
    If MV-Lines are just splitted by "Muffe" the lines will be merged together and treated as one line 
    """

    lines = lines.copy()
    lines_mv = lines[lines.comp_type=='mv_line'].reset_index(drop=True)
    merged_lines = []

    while len(lines_mv) > 0:
        # Nimm die erste Linie als Basis
        base_line = lines_mv.iloc[0]
        base_geom = base_line.geometry
        base_attrs = base_line.drop(labels='geometry').to_dict()
        lines_mv = lines_mv.drop(0).reset_index(drop=True)

        changed = True
        while changed:
            changed = False
            for idx, other_line in lines_mv.iterrows():
                other_geom = other_line.geometry

                # Check line end and beginning
                base_start = Point(base_geom.coords[0])
                base_end = Point(base_geom.coords[-1])
                other_start = Point(other_geom.coords[0])
                other_end = Point(other_geom.coords[-1])

                if (base_start.distance(other_start) < tolerance or
                    base_start.distance(other_end) < tolerance or
                    base_end.distance(other_start) < tolerance or
                    base_end.distance(other_end) < tolerance):

                    # Merge Linien korrekt orientiert
                    merged_geom = linemerge(unary_union([base_geom, other_geom]))
                    # Falls MultiLineString, längste Linie wählen
                    if merged_geom.geom_type == "MultiLineString":
                        merged_geom = max(merged_geom.geoms, key=lambda x: x.length)

                    base_geom = merged_geom
                    # Die verbundene Linie aus dem DataFrame entfernen
                    lines_mv = lines_mv.drop(idx).reset_index(drop=True)
                    changed = True
                    break  # for-Schleife neu starten

        # Fertige gemergte Linie speichern
        merged_lines.append({**base_attrs, 'geometry': base_geom})

    merged_gdf = gpd.GeoDataFrame(merged_lines, geometry='geometry', crs=lines.crs)
    
    #add adjusted mv_lines in original dataframe
    lines = lines[lines.comp_type!='mv_line']
    lines = pd.concat([lines, merged_gdf], ignore_index=True)
    return lines



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
    Splits LV- and HC-lines at joint buses (snapped to the line) and returns a new GeoDataFrame
    with all lines (LV, HC, MV), where only LV and HC lines are split.
    """

    lines_to_split = lines[lines.comp_type.isin(['lv_line', 'hc_line'])].copy()
    other_lines = lines[~lines.comp_type.isin(['lv_line', 'hc_line'])].copy()  # z.B. mv_line

    split_lines = []

    for idx, row in lines_to_split.iterrows():
        line_geom = row.geometry

        # Filter relevant Muffen/Buses for splitting
        joint_buses = buses[buses.comp_type.isin([
            'Hausanschlußmuffe', 'Verbindungsmuffe', 'Endmuffe',
            'Übergangsmuffe', 'Reparaturmuffe', 'vorverlegtes Ende',
            'HA-Kombimuffe', 'distributor'
        ])].copy()

        joint_buses['distance'] = joint_buses.geometry.apply(lambda p: line_geom.distance(p))
        near_joints = joint_buses[joint_buses['distance'] < tolerance]

        if near_joints.empty:
            split_lines.append({
                'geometry': line_geom,
                'comp_type': row.comp_type,
                'KABELTYP': row.KABELTYP
            })
        else:
            snapped_points = list(near_joints.geometry)
            split_segments = cut_line_at_points(line_geom, snapped_points)  # Annahme: existierende Hilfsfunktion

            for segment in split_segments:
                split_lines.append({
                    'geometry': segment,
                    'comp_type': row.comp_type,
                    'KABELTYP': row.KABELTYP
                })

    split_lines_df = pd.DataFrame(split_lines)
    split_lines_gdf = gpd.GeoDataFrame(split_lines_df, geometry='geometry', crs=lines.crs)
    final_gdf = pd.concat([split_lines_gdf, other_lines], ignore_index=True)

    # new line_ids
    final_gdf = final_gdf.reset_index(drop=True)
    final_gdf['line_id'] = ['line_' + str(i) for i in range(len(final_gdf))]

    return final_gdf

 

def snap_joint_buses_to_lines(lines, buses, tolerance=0.1):
    """
    Snaps all 'joint'-buses directly on the closest line, if distance < tolerance.
    Secures that coords from buses are directly on the line to avoid skipping inaccurate data.
    """
    joint_buses = buses[buses.comp_type.isin(['Hausanschlußmuffe', 'Verbindungsmuffe', 'Endmuffe',
                                          'Übergangsmuffe', 'Reparaturmuffe', 'vorverlegtes Ende', 'HA-Kombimuffe'])].copy()

    other_buses = buses[~buses.comp_type.isin(['Hausanschlußmuffe', 'Verbindungsmuffe', 'Endmuffe',
                                           'Übergangsmuffe', 'Reparaturmuffe', 'vorverlegtes Ende', 'HA-Kombimuffe'])].copy()

    
    lines = lines[lines.comp_type.isin(['lv_line', 'hc_line'])]
    line_geometries = list(lines.geometry)
    str_tree = STRtree(line_geometries)
    
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
    
    #buses_updated.to_file('/home/student/Documents/LoMa/Code/test_grid_buses_before_network.shp')
    
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





def get_nearest_bus(point, bus_tree, buses_df):
    """
    Returns closest bus to given start-/end-point of a line
    """
    dist, idx = bus_tree.query([point.x, point.y])
    return buses_df.loc[idx, "bus_id"], dist



###creating pypsa grid 
def import_grid_infrastructure(n, buses, lines, cable_types, household_count):
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
        n.buses.at[row["bus_id"], 'household_count'] = row['household_count']
        n.buses.at[row["bus_id"], 'HP'] = row['HP']
        n.buses.at[row["bus_id"], 'trafo_cap'] = row['s_nom']
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
        if 10 > dist0 > 0.4 :#and row['comp_type']=='lv_line':
            #print('##### line connecting to trafo/dist ############', 'Line:', idx)
            traf_dist_buses = buses[buses.comp_type.isin(['distributor', 'trafo'])].reset_index(drop=True)
            traf_dist_coords = np.array([[geom.x, geom.y] for geom in traf_dist_buses.geometry])
            traf_dist_tree = cKDTree(traf_dist_coords)
            bus0, dist0 = get_nearest_bus(start_point, traf_dist_tree, traf_dist_buses)
            if dist0 >= 10:
                print(f"line {row['line_id']} skipped cause no bus nearby at beginning of the line")
                continue          
        elif dist0 >= 10:
            print(f"line {row['line_id']} skipped cause no bus nearby at beginning of the line")
            continue
    
        if 10 > dist1 > 0.4 :#and row['comp_type']=='lv_line':
            #print('##### line connecting to trafo/dist ############', 'Line:', idx)
            traf_dist_buses = buses[buses.comp_type.isin(['distributor', 'trafo'])].reset_index(drop=True)
            traf_dist_coords = np.array([[geom.x, geom.y] for geom in traf_dist_buses.geometry])
            traf_dist_tree = cKDTree(traf_dist_coords)
            bus1, dist1 = get_nearest_bus(end_point, traf_dist_tree, traf_dist_buses)
            if dist1 >= 10:
                print(f"line {row['line_id']} skipped cause no bus nearby at beginning of the line")
                continue            
        elif dist1 >= 10:
            print(f"line {row['line_id']} skipped cause no bus nearby at end of the line")
            continue
        
        if row['line_id'] == 'line_5057':
            import pdb; pdb.set_trace()
    
        length_km = line_geom.length / 1000
        cable_type = row['KABELTYP']
        if cable_type in cable_types:
           r = cable_types[cable_type]["R"] * length_km      # Ohm
           x = cable_types[cable_type]["L"]/1000 * 50 * 2 * np.pi * length_km  # 2pi*frequenz #Ohm
           s_nom  = cable_types[cable_type]["U"] * cable_types[cable_type]["I_max"] * np.sqrt(3)/1e6    # MW 
           
        else:
           # ToDo: define reasonable default-values
           r = 0.3
           x = 0.05
           s_nom = 1
         
        capital_costs = 100_000*length_km/s_nom
           
        n.add("Line", row['line_id'], bus0=bus0, bus1=bus1, carrier='AC',
              length=length_km, r=r, x=x, s_nom=s_nom, s_nom_min = s_nom, capital_cost = capital_costs)
        n.lines.at[row["line_id"], 'comp_type'] = row['comp_type']
        n.lines.at[row["line_id"], 'geom'] = row['geometry']
        n.lines.at[row["line_id"], 'cable_type'] = row['KABELTYP']
        
        ##for validating
        lines.at[idx, 'bus_0'] = bus0
        lines.at[idx, 'bus_1'] = bus1
        lines.at[idx, 'line_id'] = row['line_id']
    
   
    
    #add generator at trafo
    trafo_buses = n.buses[n.buses.comp_type=='trafo'] 
    n.buses = n.buses.drop('trafo_cap', axis='columns')    # trafo_cap column isn't used anymore
    for idx, bus in trafo_buses.iterrows():
        lv_bus = bus.name               
        ms_bus = f"{lv_bus}_MS"         #dummy bus for now
        s_nom = bus.trafo_cap / 1e3 if bus.trafo_cap != 0 else 0.63
        
        n.add("Bus",
              name=ms_bus,
              v_nom=20,  
              carrier="AC",
              HP = bus.HP,
              household_count = bus.household_count,
              geom = bus.geom)
        
        n.add("Transformer",
              name=f"trafo_{lv_bus}",
              bus0=ms_bus,  
              bus1=lv_bus,   
              x=0.03864647477581,        #example vlaues from dingo
              r=0.0103174603174603,
              s_nom=s_nom)
        
        # 3. Generator am MS-Bus anschließen
        n.add("Generator", 
              name=f"gen_{idx}",
              bus=ms_bus,
              carrier="AC",
              p_nom=1e6, 
              marginal_cost=100)
    
    #add carriers
    carriers = ["AC", "land_transport_EV", "14a", "home_battery", "solar_rooftop"]
    for c in carriers:
        n.add("Carrier", c)

    return buses, lines

    

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
    line_buses = pd.concat([n.lines.bus0, n.lines.bus1]).unique()
    connected_transformers = n.transformers[
        n.transformers.bus0.isin(line_buses) | n.transformers.bus1.isin(line_buses)
    ]
    con_buses = pd.concat([n.lines.bus0, n.lines.bus1, connected_transformers.bus1, connected_transformers.bus0]).unique().tolist()
    uncon_buses = n.buses[~n.buses.index.isin(con_buses)]
    #filter out connected buses via transformator 
    
    print(f" ⚠️ Warning: Following buses are not connected to the network: {uncon_buses}")
    print("⚠️ Warning: This buses and connected components will be deleted from the network")
    n.mremove("Bus", uncon_buses.index.tolist())
    
    components_to_clean = ["Generator", "Transformer"]#, "Load"]  # list extendable

    for comp in components_to_clean:

        df = n.df(comp)
        # Check if the component is a 'Transformer'
        if comp == "Transformer":
            # For Transformers, check both bus0 and bus1
            to_remove = df[df.bus0.isin(uncon_buses.index) | df.bus1.isin(uncon_buses.index)].index.tolist()
        else:
            # For other components like 'Generator', use the 'bus' column
            to_remove = df[df.bus.isin(uncon_buses.index)].index.tolist()
        
        if to_remove:
            print(f"⚠️ Removing {len(to_remove)} {comp}(s) connected to unconnected buses")
            n.mremove(comp, to_remove)
    
    
def import_ev_chargers(n):
    
    #use shapefile for ladesäulen
    return
    

    

def create_pypsa_network(shape_files_folder, q_households_folder, heat_pump_folder, cable_types, household_count, export_shape_files, census_data):
    n = pypsa.Network()
    time_index = pd.date_range('2023-01-01', periods=8760, freq='h')
    n.snapshots = time_index
    
    buses, lines = create_gdf_from_shape(shape_files_folder)
    buses = snap_joint_buses_to_lines(lines, buses)
    if household_count: #check if household_data taken from cencus or own input_file
        buses = count_households_per_bus_census_data(buses, census_data)
    else:
        buses = count_households_per_bus_input_file(buses, q_households_folder)  
    buses = check_heat_pumps(buses, heat_pump_folder)
    lines = merge_connected_mv_lines(lines)
   
    #final_load_buses = map_load_bus_to_network_bus(buses, lines)
    #network_buses = buses[buses.comp_type.isin(['trafo', 'distributor'])]
    #all_network_buses = pd.concat([final_load_buses, network_buses])
    split_lines = split_lines_on_joints(lines, buses)
    #merged_lines = merge_unconnected_lines(split_lines, buses)
    buses, lines = import_grid_infrastructure(n, buses, split_lines, cable_types, household_count) 
    if export_shape_files:
        os.makedirs('results', exist_ok=True)
        buses.to_file('results/grid_buses_test.shp')
        lines.to_file('results/grid_lines_test.shp')
    fix_grid_infrastructure(n)
    #n = open_LV_circle(n, 'line_163')
    
    return n
    
    
    

    
    
