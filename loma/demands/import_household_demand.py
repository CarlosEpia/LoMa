#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distributes household electricity demand profiles across buses and assigns standard load profiles (SLP)."""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


def create_profile_pool_from_df(profil_type, profiles_df, limit=1):
    """Pick a random household load profile of the given type from the profile pool."""

    cols = [col for col in profiles_df.columns if col.startswith(profil_type)]
    if not cols:
        raise ValueError(f"No profile of type {profil_type} found.")

    # choose random profile
    chosen_col = np.random.choice(cols)
    profile = profiles_df[chosen_col]

    return profile

def distribute_household_demand(n, profile_dist, random_seed=42):
    """Assign each household at a house-connection bus a random load
    profile, drawn according to the household-type probabilities of the
    nearest profile-distribution cell."""
    #define seed
    np.random.seed(random_seed)
    
    # Load profile pool from HDF file
    pool = pd.read_hdf('data/data_bundle/hh_el_load_profiles_100k.hdf')

    # Ensure profile_dist is a GeoDataFrame and change crs system
    if not isinstance(profile_dist, gpd.GeoDataFrame):
        profile_dist = gpd.GeoDataFrame(
            profile_dist,
            geometry=gpd.GeoSeries.from_wkt(profile_dist['geometry'])
        )
    profile_dist = profile_dist.set_crs("EPSG:4326") 
    profile_dist = profile_dist.to_crs("EPSG:32632")
        
        
    # Define available profile types
    profil_types = ['SR', 'SO', 'SK', 'PR', 'PO', 'OR', 'OO', 'P1', 'P2', 'P3']
    # Define time index for one year with hourly resolution
    time_index = n.snapshots

    # Dictionary to collect all load time series before concatenation
    new_profiles = {}
    
    # Iterate over all house connection buses
    for bus_name, bus in n.buses[n.buses.comp_type == 'house_connection'].iterrows():
        x, y = bus.x, bus.y
        point = Point(x, y)

        # Find the closest profile distribution row
        profile_dist['dist'] = profile_dist.geometry.distance(point)
        closest_row = profile_dist.loc[profile_dist['dist'].idxmin()]

        # Extract probabilities for the profile types
        percentage = closest_row[profil_types].astype(float).values

        # Create one load per household at this bus
        for i in range(1, int(bus.household_count) + 1):
            # Randomly select a profile type based on probabilities
            profil_type = np.random.choice(profil_types, p=percentage)
            profile = create_profile_pool_from_df(profil_type, pool, limit=1)

            # Create a time series with hourly resolution in MWh
            profile_series = pd.Series(profile.values, index=time_index)
            profile_series_mwh = profile_series / 1e6

            # Define unique load name
            load_name = f'HH_Load_{bus_name}_{i}'
            n.add("Load", load_name, bus=bus_name, carrier='conventional_load', p_set=0)

            # Collect profile series for later batch insertion
            new_profiles[load_name] = profile_series_mwh

    # Convert collected profiles to DataFrame
    profiles_df = pd.DataFrame(new_profiles, index=time_index)

    # Merge with existing time series DataFrame
    if n.loads_t.p_set.empty:
        n.loads_t.p_set = profiles_df
    else:
        n.loads_t.p_set = pd.concat([n.loads_t.p_set, profiles_df], axis=1)

    return n

def define_slp_as_load_profile(n):
    """Replace each load's individual time series with the standard load
    profile (SLP), shaped to preserve each load's original annual sum."""
    hourly_profile = pd.read_csv("data/data_bundle/SLP_hourly.csv", index_col=0)
    slp_normed = hourly_profile/hourly_profile.sum()
    slp = slp_normed["load"]  # Series, not DataFrame
    slp.index = n.loads_t.p_set.index
    slp_shifted = pd.Series(np.roll(slp.values, 24), index=slp.index)

    annual_sums = n.loads_t.p_set.sum()
    new_loads = slp_shifted.values.reshape(-1, 1) * annual_sums.values.reshape(1, -1)

    n.loads_t.p_set = pd.DataFrame(new_loads, index=slp.index, columns=n.loads_t.p_set.columns)

    return n






