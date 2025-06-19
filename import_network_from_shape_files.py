#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 09:49:16 2025

@author: student
"""

#ToDo: Create function to add manual buses for certain lines and split lines there



import geopandas as gpd
import os
from shapely.geometry import Point
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import pypsa

input_folder = '/home/student/Documents/LoMa/Code/data/Filtered data Kronenburg/'   # input folder for shapefiles  #ToDo: set the input folder in the args of appl.py

def create_gdf_from_shape():
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
    distributors = gpd.read_file(os.path.join(input_folder, "Gis ST Kabelverteiler Position.shp"))
    joints = gpd.read_file(os.path.join(input_folder, "Gis NSP Muffe Position.shp"))
    MVLV_trafos = gpd.read_file(os.path.join(input_folder, "Gis ST Station Position.shp"))

    # component-type-column for distinguish the buses
    LV_lines["comp_type"] = "LV_Line"
    HA_lines["comp_type"] = "HA_Line"
    joints["comp_type"] = "joint"
    distributors["comp_type"] = "distributor"
    MVLV_trafos["comp_type"] = "trafo"
    HA_Bus["comp_type"] = "house_connection"
    
    
    # combine all bus-datfarmes
    buses = pd.concat([joints, distributors, MVLV_trafos, HA_Bus], ignore_index=True)
    buses["bus_id"] = [f"bus_{i}" for i in range(len(buses))]
    buses = buses.reset_index(drop=True)
    
    #combine all line-dataframes
    lines = pd.concat([LV_lines, HA_lines], ignore_index=True)
    lines["line_id"] = [f"line_{i}" for i in range(len(lines))]
    lines = lines.reset_index(drop=True)
    
    return buses, lines



def get_nearest_bus(point, bus_tree, buses_df):
    """
    Returns closest bus to given start-/end-point of a line
    """
    dist, idx = bus_tree.query([point.x, point.y])
    if dist>1:
        print(dist, buses_df.loc[idx, "bus_id"])
    return buses_df.loc[idx, "bus_id"]



def import_grid_structure(n, buses, lines):
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

    
    LV_lines = lines[lines.comp_type == 'LV_Line']
    HA_lines = lines[lines.comp_type == 'HA_Line']

    # import buses
    for _, row in buses.iterrows():
        n.add("Bus", row["bus_id"], x=row.geometry.x, y=row.geometry.y)
    
    # import LV lines
    for idx, row in LV_lines.iterrows():
        line_geom = row.geometry
        start_point = Point(line_geom.coords[0])
        end_point = Point(line_geom.coords[-1])
        
        relevant_buses = buses[buses.comp_type.isin(['joint', 'distributor', 'trafo'])]
        relevant_buses = relevant_buses.reset_index(drop=True)
        bus_coords = np.array([[geom.x, geom.y] for geom in relevant_buses.geometry])
        bus_tree = cKDTree(bus_coords)
        bus0 = get_nearest_bus(start_point, bus_tree, relevant_buses)
        bus1 = get_nearest_bus(end_point, bus_tree,relevant_buses)
    
        length_km = line_geom.length / 1000
    
        n.add("Line", f"lv_line_{idx}", bus0=bus0, bus1=bus1,
                    length=length_km, r=0.3, x=0.05, s_nom=100)   #ToDo: look for valid x,r and s_nom values
        
        LV_lines.at[idx, 'bus_0'] = bus0
        LV_lines.at[idx, 'bus_1'] = bus1
     
        
    # import HA_lines
    for idx, row in HA_lines.iterrows():
        line_geom = row.geometry
        start_point = Point(line_geom.coords[0])
        end_point = Point(line_geom.coords[-1])
        
        relevant_buses = buses[buses.comp_type.isin(['joint', 'house_connection'])]
        relevant_buses = relevant_buses.reset_index(drop=True)
        bus_coords = np.array([[geom.x, geom.y] for geom in relevant_buses.geometry])
        bus_tree = cKDTree(bus_coords)
        bus0 = get_nearest_bus(start_point, bus_tree, relevant_buses)
        bus1 = get_nearest_bus(end_point, bus_tree,relevant_buses)
    
        length_km = line_geom.length / 1000
    
        n.add("Line", f"ha_line_{idx}", bus0=bus0, bus1=bus1,
                    length=length_km, r=0.3, x=0.05, s_nom=100)   #ToDo: look for valid x,r and s_nom values
    
            
        HA_lines.at[idx, 'bus_0'] = bus0
        HA_lines.at[idx, 'bus_1'] = bus1
    

    
    
    
    #just for validating infrasturcture
    HA_lines.to_file('/home/student/Documents/LoMa/Code/test_grid_HA.shp')
    LV_lines.to_file('/home/student/Documents/LoMa/Code/test_grid_LV.shp')
    buses.to_file('/home/student/Documents/LoMa/Code/test_grid_buses.shp')
    
    
def create_pypsa_network():
    n = pypsa.Network()
    
    buses, lines = create_gdf_from_shape()
    import_grid_structure(n, buses, lines)    
    
    return n
    
    
    

    
    