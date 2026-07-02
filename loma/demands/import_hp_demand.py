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
import logging

from loma.demands.household_count import parse_bus_numbers


def check_heat_pumps(n, hp_shapefile_path):
    """
    Assigns Heat Pumps to buses based on a heat pump location shapefile.
    """
    con_buses = n.buses[n.buses.comp_type == "house_connection"].copy()
    con_buses["HP"] = 0  # Initialize HP column
    con_buses["hp_capacity"] = 0.0

    print(f"Heat_pumps are distributed according to shapefile '{hp_shapefile_path}'.")
    shp_path = hp_shapefile_path

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


def add_heat_loads_to_network(n, project_config):
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
    n = check_heat_pumps(n, project_config["paths"]["heat_pump_shapefile"])
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
    potential_buses = n.buses[
          n.buses.comp_type == 'house_connection'
    ].copy()

    potential_gdf = gpd.GeoDataFrame(
        potential_buses,
        geometry=gpd.points_from_xy(potential_buses.x, potential_buses.y),
        crs=census_cells.crs,
    )
    # spartial joint for potential buses and heat_demand cells
    mapped_buses = gpd.sjoin_nearest(potential_gdf, heat_demand_cells, how="left")
    mapped_buses = mapped_buses[~mapped_buses.index.duplicated(keep="first")]

    # just keep buses with a zensus_poplulation_id for avoiding errors
    valid_zensus_ids = daily_profiles["zensus_population_id"].unique()
    mapped_buses = mapped_buses[
        (mapped_buses["zensus_pop"].isin(valid_zensus_ids)) & 
        (mapped_buses["demand"].notna())
    ]

    # Filter buses with Heatpump 
    bus_with_hp = mapped_buses[mapped_buses.HP == 1].copy()
    
    # scenario target for the configured project/scenario
    target_count = project_config["scenario_targets"]["heat_pumps"]
    # scaling factor for models covering only part of the reference grid
    reference_bus_count = project_config["scenario_targets"]["reference_house_connection_bus_count"]
    scaling_factor = len(potential_buses) / reference_bus_count
    target_count = int(target_count * scaling_factor)
    current_count = len(bus_with_hp)

    if target_count < current_count:
        # Fall 1: Weniger HPs -> Zufällig aus dem HP=1 Set wählen
        logging.info(f"Reduce HPs from {current_count} to {target_count}")
        bus_with_cell = bus_with_hp.sample(n=target_count, random_state=42)
        
    elif target_count > current_count:
        # Fall 2: Mehr HPs -> HP=1 behalten + Rest aus anderen house_connections auffüllen
        logging.info(f"Increase HPs from {current_count} to {target_count}")
        n_extra = target_count - current_count
        
        # Verfügbare Busse sind alle house_connection, die NOCH KEINE HP haben
        available_buses = mapped_buses.loc[~mapped_buses.index.isin(bus_with_hp.index)]
        extra_samples = available_buses.sample(n=n_extra, replace=False, random_state=42)
            
        bus_with_cell = pd.concat([bus_with_hp, extra_samples])
    
    # Initialize load time series
    snapshots = n.snapshots
    n_hours = len(snapshots)
    n_buses = len(bus_with_cell)
    load_profiles = pd.DataFrame(
        0.0, index=snapshots, columns=bus_with_cell.index
    )

    #source : "Branchenstudie 2023: Marktentwicklung – Prognose – Handlungsempfehlungen" - Bundesverband Wärmepumpe (BWP) e. V, 2023 
    # prognosed avg. heatpump capacity 10 kW for 2030
    HP_CAPACITY_MIN_MW = 0.010   # thermische Lesitung
    HP_CAPACITY_MAX_MW = 0.015   # thermisceh Leistung
      
    # Reproduzierbarer RNG – Seed einmal pro Simulation setzen
    rng = np.random.default_rng(seed=42)
      
    # --- Profil-Schleife ---
    used_profiles = {}
    scale_up_count = 0
    scale_down_count = 0
    for bus_idx, row in bus_with_cell.iterrows():
        try:
            bus_id = bus_idx
        except IndexError:
            print(f"IndexError: Kein bus_id für Spalte {bus_idx}; überspringe.")
            continue
      
        zensus_id   = row["zensus_pop"]
        annual_demand = row["demand"]
        if pd.isna(annual_demand) or pd.isna(zensus_id):
              continue
      
        daily_candidates = daily_profiles.loc[
            daily_profiles["zensus_population_id"] == zensus_id
        ]
        if daily_candidates.empty:
            continue
      
        n_profiles = len(daily_candidates)
      
        if zensus_id not in used_profiles:
            used_profiles[zensus_id] = 0
      
        profile_idx = used_profiles[zensus_id] % n_profiles
        daily = daily_candidates.iloc[profile_idx]
        used_profiles[zensus_id] += 1
      
        climate_factor = yearly_profiles["daily_demand_share"].values
        idp_ids = daily["selected_idp_profiles"]
      
        hourly_profile = []
        for day, idp_id in enumerate(idp_ids):
            idp       = np.array(idp_pool.loc[idp_id, "idp"])
            day_share = climate_factor[day]
            hourly_profile.extend(idp * day_share * (annual_demand / n_profiles))
        hourly_profile = np.array(hourly_profile)
      
        # Thermisch → elektrisch via COP
        temp_air = pd.read_csv(
              "data/data_bundle/wetterdaten_2011_Luft.csv"
        ).set_index("MESS_DATUM")
        cop_air = calculate_cop_air(temp_air["TT_TU"])
        elec_profile = hourly_profile / cop_air
      
        thermal_peak = hourly_profile.max()  # Thermischer Spitzenwert in MW
      
        if thermal_peak < 0.005 or thermal_peak > 0.05:   #to avoid extrem high and low profiles due to 
            hp_capacity_target = rng.uniform(HP_CAPACITY_MIN_MW, HP_CAPACITY_MAX_MW)
            scaling_factor = hp_capacity_target / cop_air / elec_profile.max()
            elec_profile = elec_profile * scaling_factor
      
            if thermal_peak < 0.001:
                scale_up_count += 1
            else:
                scale_down_count += 1 
      
        # In Lastmatrix schreiben
        elec_profile.index = load_profiles.index
        load_profiles.loc[:, bus_idx] = elec_profile[:n_hours]
      
        n.add(
            "Load",
            name=f"heat_load_{bus_id}",
            bus=bus_id,
            carrier="heat_pump",
            p_set=load_profiles[bus_idx],
        )
        
    print(
        f"In total {scale_up_count + scale_down_count} profiles were scaled "
        f"({scale_up_count} scaled up due to too low peak demand, "
        f"{scale_down_count} scaled down due to too high peak demand)"
    )
    
    return n


def calculate_cop_air(t_source, t_sink=55):
    delta_t = t_sink - t_source
    return (
        6.81 - 0.121 * delta_t + 0.000630 * delta_t**2
    )  # according to Brown et. al: Synergies of sector coupling and transmission reinforcement in a cost-optimised, highlyrenewable European energy system", 2018, p. 8
