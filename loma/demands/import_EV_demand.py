#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 08:34:28 2025

@author: student
"""


import geopandas as gpd
import os
from collections import defaultdict
from shapely.strtree import STRtree


#input_folder = 'data/Input_files/Filtered_data_Kronenburg_V3'

def import_EV_loads(n, input_folder):
    EV_locations = gpd.read_file(os.path.join(input_folder, "Gis ST Ladesäule Position.shp"))
    EV_locations = EV_locations.to_crs('EPSG:32632')
    con_buses = n.buses[n.buses.comp_type == 'house_connection'].copy()
    
    # spartial queries with STRtree
    bus_ids = con_buses.index.values
    tree = STRtree(con_buses.geom)

    # Counter in case multiple EV-chargers for one bus
    bus_load_counter = defaultdict(int)

    for ev_geom in EV_locations.geometry:
        nearest_geom = tree.nearest(ev_geom)
        bus_name = bus_ids[nearest_geom]

        # in case two EV-chargers are connected to the same bus 
        bus_load_counter[bus_name] += 1
        counter = bus_load_counter[bus_name]
        load_name = f"EV_load_{bus_name}_{counter}"
        n.add("Load",
              load_name,
              bus=bus_name,
              p_set=0,       
              carrier="land_transport_EV")
        
    return n



#ToDo: add EV_load_profile and EV_flexible model from paul