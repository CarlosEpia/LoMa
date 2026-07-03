#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inserts solar rooftop generators and home battery storage units onto house-connection buses, scaled to project config targets."""

import ast
import geopandas as gpd
import pandas as pd
import logging


def insert_pv_rooftop_and_battery(
    network, shape_path, pv_rooftop_path, pv_feedin_path, batteries_path, project_config
):
    """Insert solar rooftop generators and home battery storage units at house-connection buses within `shape_path`."""
    crs = project_config["project"]["crs"]
    shape = gpd.read_file(shape_path).to_crs(crs)
    buses = network.buses.copy()
    buses = buses[buses["comp_type"] == "house_connection"]
    buses = gpd.GeoDataFrame(buses, geometry="geom", crs=crs)
    network = insert_pv_rooftop(network, shape, buses, pv_rooftop_path, pv_feedin_path, project_config)
    network = insert_home_battery(network, shape, buses, batteries_path, project_config)

    return network


def insert_pv_rooftop(network, shape, buses, pv_rooftop_path, pv_feedin_path, project_config):
    """Insert solar rooftop generators at the nearest house-connection bus,
    scaled down to the project's pv_rooftop_mwp target if needed, with
    feed-in profiles from the weather-cell-based feed-in data."""
    crs = project_config["project"]["crs"]
    solar = gpd.read_file(pv_rooftop_path).to_crs(crs)
    solar = gpd.clip(solar, shape)

    buses.index.rename("bus", inplace=True)
    solar = gpd.sjoin_nearest(solar, buses, "left", distance_col="distance")

    logging.warning(
        f"""
                    {len(solar[solar["distance"] > 50])} pv_rooftop generators
                    discarded because of distance to the closest bus
                    """
    )
    solar = solar[solar["distance"] <= 50]
    
    #scale down amount of pvs according to scenario target
    target_p_nom = project_config["scenario_targets"]["pv_rooftop_mwp"]

    current_p_nom = solar["capacity"].sum()
    
    if current_p_nom > target_p_nom:
        logging.info(f"Reducing solar rooftop capacity from {current_p_nom:.2f} MWp to {target_p_nom:.2f} MWp")

        # shuffle the dataset randomly (reproducible via random_state)
        solar = solar.sample(frac=1, random_state=42)

        # build the cumulative sum and filter by it
        solar["cumsum_p_nom"] = solar["capacity"].cumsum()
        solar = solar[solar["cumsum_p_nom"] <= target_p_nom].copy()
        
        logging.info(f"PV capacity reduction finished. Remaining pv: {len(solar)}")
       
    
    # insert data into network tables
    solar.rename(columns={"capacity": "p_nom"}, inplace=True)
    solar["Generator"] = solar.apply(lambda b: f"pv_roof_{b.name}_{b.bus}", axis=1)
    solar.set_index("Generator", drop=True, inplace=True)
    solar["carrier"] = "solar_rooftop"
    solar["efficiency"] = 0.9

    for name, row in solar.iterrows():
        network.add(
            "Generator",
            name=name,
            bus=row["bus"],
            carrier=row["carrier"],
            p_nom=float(row["p_nom"]),
            control="PQ",
            efficiency=float(row["efficiency"]),
        )

    solar_t = pd.read_csv(
        pv_feedin_path,
        index_col="w_id",
        usecols=["w_id", "weather_year", "feedin"],
    )
    solar_t = solar_t[solar_t.index.isin(solar["weather_cell_id"])]
    solar_t["feedin"] = solar_t["feedin"].apply(ast.literal_eval)
    network.generators_t["p_max_pu"].loc[:, solar.index] = 0

    for gen, data in solar.iterrows():
        network.generators_t["p_max_pu"][gen] = solar_t.at[
            data["weather_cell_id"], "feedin"
        ]

    return network


def insert_home_battery(network, shape, buses, batteries_path, project_config):
    """Insert home battery storage units at buses that already have a solar
    rooftop generator, scaled to the project's home_battery_mw target."""
    crs = project_config["project"]["crs"]
    bat = gpd.read_file(batteries_path).to_crs(crs)
    bat = gpd.clip(bat, shape)
    bat = gpd.sjoin_nearest(bat, buses, "left", distance_col="distance")

    logging.warning(
        f"""
                    {len(bat[bat["distance"] > 50])} home_batteries
                    discarded because of distance to the closest bus
                    """
    )
    bat = bat[bat["distance"] <= 50]

    # Filter batteries: only keep batteries at buses that have PV generators ()
    pv_buses = set(network.generators[network.generators["carrier"] == "solar_rooftop"]["bus"].unique())
    bat_before_filter = len(bat)
    bat = bat[bat["bus"].isin(pv_buses)].copy()
    logging.info(
        f"Filtered batteries to only buses with PV generators: "
        f"{bat_before_filter} batteries -> {len(bat)} batteries"
    )

    #scale down amount of batteries according to scenario target
    target_p_nom = project_config["scenario_targets"]["home_battery_mw"]

    current_p_nom = bat["p_nom"].sum()
    
    if current_p_nom > target_p_nom:
        logging.info(f"Reducing home-batteries capacity from {current_p_nom:.2f} MW to {target_p_nom:.2f} MW")

        # shuffle the dataset randomly (reproducible via random_state)
        bat = bat.sample(frac=1, random_state=42)

        # build the cumulative sum and filter by it
        bat["cumsum_p_nom"] = bat["p_nom"].cumsum()
        bat = bat[bat["cumsum_p_nom"] <= target_p_nom].copy()
        
        logging.info(f"Home-batteries capacity reduction finished. Remaining batteries: {len(bat)}")
        
    if current_p_nom < target_p_nom: 
        bat['p_nom'] *= target_p_nom/current_p_nom

    # insert data into network tables
    bat.rename(columns={"Bus": "bus"}, inplace=True)
    bat["StorageUnit"] = bat.apply(lambda b: f"sto_unit_{b.name}_{b.bus}", axis=1)
    bat.set_index("StorageUnit", drop=True, inplace=True)

    bat["carrier"] = "home_battery"
    bat["sign"] = 1
    bat["max_hours"] = 6

    for name, row in bat.iterrows():
        network.add(
            "StorageUnit",
            name=name,
            bus=row["bus"],
            carrier=row["carrier"],
            p_nom=float(row["p_nom"]),
            max_hours=float(row["max_hours"]),
            control="PQ",
            p_nom_extendable = False
        )

    return network
