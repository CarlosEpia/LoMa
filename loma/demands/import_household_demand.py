#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 11:02:25 2025

@author: student

"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# ToDo: change path to data_bundle, if the file is part of it 
folderpath = '/home/student/airflow_exe2/data_bundle_egon_data/household_electricity_demand_profiles/hh_el_load_profiles_100k.hdf'


def create_profile_pool_from_df(profil_type, profiles_df, limit=1):
    """
    
    """
    
    cols = [col for col in profiles_df.columns if col.startswith(profil_type)]    
    if not cols:
        raise ValueError(f"Kein Profil vom Typ {profil_type} gefunden.")
    
    # Choose random profil
    chosen_col = np.random.choice(cols)
    profile = profiles_df[chosen_col]
    
    return profile

def distribute_household_demand(n, profile_dist):
    
    """
    
    """
    #create profile pool from data_bundle
    pool = pd.read_hdf(folderpath)

    if not isinstance(profile_dist, gpd.GeoDataFrame):
        profile_dist = gpd.GeoDataFrame(profile_dist, geometry=gpd.GeoSeries.from_wkt(profile_dist['geometry']))
    
    
    for bus_name, bus in n.buses[n.buses.comp_type == 'house_connection'].iterrows():
        x = bus.x
        y = bus.y
        point = Point(x, y)

        profile_dist['dist'] = profile_dist.geometry.distance(point)
        closest_row = profile_dist.loc[profile_dist['dist'].idxmin()]

        # extract profile_types with properbilities 
        profil_types = ['SR', 'SO', 'SK', 'PR', 'PO', 'OR', 'OO', 'P1', 'P2', 'P3']
        percentage = closest_row[profil_types].astype(float).values
        profil_type = np.random.choice(profil_types, p=percentage)

        profile = create_profile_pool_from_df(profil_type, pool, limit=1)
        time_index = pd.date_range('2023-01-01', periods=8760, freq='h')
        profile_series = pd.Series(profile.values, index=time_index)
        profile_series_mwh = profile_series / 1e6
        
        load_name = 'Load_'+bus_name.split("_")[1]
        n.add("Load", load_name, bus=bus_name, carrier='household', p_set=0)
        
        if n.loads_t.p_set.empty:
            n.loads_t.p_set = pd.DataFrame(index=time_index)
    
        n.loads_t.p_set[load_name] = profile_series_mwh
        
    return n


#toDo: use excel file from husum for household_data for defining the amount of households
    





