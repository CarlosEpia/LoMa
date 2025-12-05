#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 10:27:02 2025

@author: student
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np

from loma.demands.household_count import parse_bus_numbers


def check_heat_pumps(n):
    """
    Assigns Heat Pumps to buses based on input CSV or fallback shapefile.

    Priority:
    1. If 'heat_pumps.csv' exists, use it.
    2. Otherwise, use 'hp_2035.shp' fallback.
    """
    con_buses = n.buses[n.buses.comp_type == "house_connection"].copy()
    con_buses["HP"] = 0  # Initialize HP column
    con_buses["hp_capacity"] = 0.0

    print("Heat_pumps are distributes according to shapefile 'hp_2035.shp'.")
    shp_path = "data/Input_files/hp_husum_2035/hp_2035.shp"

    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"No file found like:{shp_path}")

    fallback_hp = gpd.read_file(shp_path)
    fallback_hp = fallback_hp.rename(
        columns={"building_i": "building_id", "hp_capacit": "hp_capacity"}
    )

    # Convert buses to GeoDataFrame
    bus_gdf = gpd.GeoDataFrame(
        con_buses,
        geometry="geom",
        crs=fallback_hp.crs,
    )

    if "bus_id" in bus_gdf.columns:
        bus_gdf["_bus_id_map"] = bus_gdf["bus_id"]
    else:
        bus_gdf["_bus_id_map"] = bus_gdf.index.astype(str)

    max_dist = 25

    closest = gpd.sjoin_nearest(
        fallback_hp,
        bus_gdf,
        how="left",
        distance_col="distance",
    )

    closest = closest[closest["distance"] < max_dist]

    n.buses["HP"] = n.buses.apply(
        lambda x: 1 if x.name in closest["_bus_id_map"].values else 0, axis=1
    )

    return n


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
    n = check_heat_pumps(n)
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
    bus_with_cell = bus_with_cell[
        ~bus_with_cell.index.duplicated(keep="first")
    ]

    # Save original left-index (these are the bus IDs/labels from the network)
    left_bus_ids = list(bus_with_cell.index)

    # Reset index to get a simple RangeIndex for iteration and for load_profiles columns
    bus_with_cell = bus_with_cell.reset_index(drop=True)

    # Initialize load time series
    snapshots = n.snapshots
    n_hours = len(snapshots)
    n_buses = len(bus_with_cell)
    load_profiles = pd.DataFrame(
        0.0, index=snapshots, columns=bus_with_cell.index
    )

    # avg heat-demand for calculating a scaling_factor for areas with really low heat_demand due to old census-data
    avg_hp_capcity = 0.0122  # acccording to "Technology Assessment Report - e-HIGHWAY 2050 , Technofi, 2015"

    # Create a profile for each bus
    used_profiles = {}
    for bus_idx, row in bus_with_cell.iterrows():
        try:
            bus_id = left_bus_ids[bus_idx]
        except IndexError:
            # Sicherheitsnetz: falls Mapping aus irgendeinem Grund nicht passt
            print(
                f"IndexError: Kein bus_id für Spalte {bus_idx}; überspringe."
            )
            continue

        zensus_id = row["zensus_pop"]
        annual_demand = row["demand"]
        if pd.isna(annual_demand) or pd.isna(zensus_id):
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

        if (
            elec_profile.max() < 0.0005
        ):  # ToDo: Discuss if adjustemnt of value is necessary
            scaling_factor = avg_hp_capcity / cop_air / elec_profile.max()
            elec_profile = elec_profile * scaling_factor
        # Write into matrix
        elec_profile.index = load_profiles.index
        load_profiles.loc[:, bus_idx] = elec_profile[:n_hours]

        # Add load to the network
        n.add(
            "Load",
            name=f"heat_load_{bus_id}",
            bus=bus_id,
            carrier="AC",
            p_set=load_profiles[bus_idx],
        )

    return n


def calculate_cop_air(t_source, t_sink=55):
    delta_t = t_sink - t_source
    return (
        6.81 - 0.121 * delta_t + 0.000630 * delta_t**2
    )  # according to Brown et. al: Synergies of sector coupling and transmission reinforcement in a cost-optimised, highlyrenewable European energy system", 2018, p. 8
