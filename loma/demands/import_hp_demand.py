#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 10:27:02 2025

@author: student
"""
import pandas as pd
import geopandas as gpd
import re
import numpy as np

from loma.demands.household_count import parse_bus_numbers, hausnummer_split


def check_heat_pumps(buses, input_path):

    hp_df = pd.read_csv("data/Input_files/heat_pumps.csv")
    buses = buses.rename(
        columns={"HAUSNUMMER": "Hausnummer", "Strasse": "Straße"}
    )
    con_buses = buses[buses.comp_type == "house_connection"].copy()
    con_buses["Straße"] = con_buses["Straße"].str.replace("Strasse", "Straße")
    con_buses["parsed_numbers"] = con_buses["Hausnummer"].apply(
        parse_bus_numbers
    )
    hp_df["parsed_numbers"] = hp_df["Hnr."].apply(parse_bus_numbers)

    # create Heat_pump column
    con_buses["HP"] = 0
    hp_df = hp_df[hp_df.WP == "ja"]

    for idx, row in hp_df.iterrows():
        street = row["Straße"]
        num_list = row["parsed_numbers"]
        street_buses = con_buses[con_buses.Straße == street]

        if len(num_list) == 0:
            continue
        if street_buses.empty:
            continue

        # 1) try to find exact match (Number, letter))
        found = False
        for num in num_list:
            exact_matches = street_buses[
                street_buses.parsed_numbers.apply(lambda nums: num in nums)
            ]

            if not exact_matches.empty:
                con_buses.loc[exact_matches.index, "HP"] = 1
                found = True
                break

        if not found:
            print(
                f"Warning: No Match found for Heat_pump in {street} {num_list}"
            )

        ########## check if an fallback option is neccessary for other LV-Region

        # Fallback: just compare number without letter
        # number_matches = street_buses[
        # street_buses['parsed_numbers'].apply(lambda nums: any(n == num for n, _ in nums))
        # ]
        # if not number_matches.empty:
        #  buses.loc[number_matches.index, 'house_count'] += 1
        # continue
    hp_bus_ids = con_buses[con_buses.HP == 1].bus_id.to_list()
    buses["HP"] = buses["bus_id"].isin(hp_bus_ids).astype(int)

    return buses


def add_heat_loads_to_network(n):
    """
    Adds heat loads to a PyPSA network if bus.HP == 1.
    Load profiles are calculated based on census cells, daily profiles,
    IDP profiles, and yearly climate scaling.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network with buses (must contain the column 'HP').

    Returns
    -------
    pypsa.Network
        Network with additional loads and corresponding p_set time series.
    """

    # load input-data
    census_cells = gpd.read_file("data/data_bundle/Census_cells_SH.shp")
    daily_profiles = pd.read_hdf("data/data_bundle/heat_daily_profiles.hdf")
    yearly_profiles = pd.read_hdf(
        "data/data_bundle/heat_yearly_profile.hdf", key="yearly_profile"
    )
    idp_pool = pd.read_hdf("data/data_bundle/idp_pool.hdf", key="idp_pool")
    peta_heat = pd.read_csv("data/data_bundle/Peta_heat_demand.csv")

    heat_demand_cells = peta_heat.merge(
        census_cells,
        left_on="zensus_population_id",
        right_on="zensus_pop",
        how="left",
    )
    heat_demand_cells = heat_demand_cells[
        heat_demand_cells.scenario == "status2019"
    ]
    heat_demand_cells = gpd.GeoDataFrame(
        heat_demand_cells,
        geometry=heat_demand_cells["geometry"],
        crs=census_cells.crs,
    )

    # Relevant buses with Heat_pump
    buses_with_hp = n.buses[n.buses.HP == 1].copy()
    bus_gdf = gpd.GeoDataFrame(
        buses_with_hp,
        geometry=gpd.points_from_xy(buses_with_hp.x, buses_with_hp.y),
        crs=census_cells.crs,
    )

    # Spatial mapping: Bus → Census cell
    bus_with_cell = gpd.sjoin_nearest(bus_gdf, heat_demand_cells, how="left")

    # Initialize load time series
    snapshots = n.snapshots
    n_hours = len(snapshots)
    load_profiles = pd.DataFrame(
        0.0, index=snapshots, columns=bus_with_cell.index
    )
    
    #avg heat-demand for calculating a scaling_factor for areas with really low heat_demand due to old census-data
    avg_hp_capcity = 0.0122 #acccording to "Technology Assessment Report - e-HIGHWAY 2050 , Technofi, 2015"

    
    # Create a profile for each bus
    used_profiles = {}
    for bus_idx, row in bus_with_cell.iterrows():
        zensus_id = row["zensus_pop"]
        annual_demand = row["demand"]
        if pd.isna(annual_demand):
            continue

        # Daily profiles for this cell
        daily_candidates = daily_profiles.loc[
            daily_profiles["zensus_population_id"] == zensus_id
        ]
        if daily_candidates.empty:
            continue

        # Number of profiles (original buildings) in the cell
        n_profiles = len(daily_candidates)

        # If cell not used yet → start at 0
        if zensus_id not in used_profiles:
            used_profiles[zensus_id] = 0

        # Determine profile index (modulo for looping through)
        profile_idx = used_profiles[zensus_id] % n_profiles
        daily = daily_candidates.iloc[profile_idx]
        used_profiles[zensus_id] += 1

        # Yearly scaling factor (same for all if only one column)
        climate_factor = yearly_profiles["daily_demand_share"].values

        # IDPs: selected daily IDs for this year (365 IDs)
        idp_ids = daily["selected_idp_profiles"]

        # Create 8760h profile
        hourly_profile = []
        for day, idp_id in enumerate(idp_ids):
            idp = np.array(idp_pool.loc[idp_id, "idp"])  # 24 values
            day_share = climate_factor[day]  # Daily factor from yearly profile
            # Normalize by the number of profiles per cell
            hourly_profile.extend(
                idp * day_share * (annual_demand / n_profiles)
            )

        hourly_profile = np.array(hourly_profile)
        # Normalize to annual demand (optional, in case of rounding errors)
        # hourly_profile *= annual_demand / hourly_profile.sum()

        # Transform heat load into electrical load
        temp_air = pd.read_csv(
            "data/data_bundle/wetterdaten_2011_Luft.csv"
        ).set_index("MESS_DATUM")
        cop_air = calculate_cop_air(temp_air["TT_TU"])
        elec_profile = hourly_profile / cop_air
        
        if elec_profile.max() < 0.0005:   #ToDo: Discuss if adjustemnt of value is necessary
            scaling_factor = avg_hp_capcity/cop_air/elec_profile.max()
            elec_profile = elec_profile * scaling_factor
        # Write into matrix
        elec_profile.index = load_profiles.index
        load_profiles.loc[:, bus_idx] = elec_profile[:n_hours]
        
        

        # Add load to the network
        n.add(
            "Load",
            name=f"heat_load_{bus_idx}",
            bus=bus_idx,
            carrier="AC",
            p_set=load_profiles[bus_idx],
        )

    return n


def calculate_cop_air(t_source, t_sink=55):
    delta_t = t_sink - t_source
    return (
        6.81 - 0.121 * delta_t + 0.000630 * delta_t**2
    )  # according to Brown et. al: Synergies of sector coupling and transmission reinforcement in a cost-optimised, highlyrenewable European energy system", 2018, p. 8
