#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assigns commercial/trade/services (CTS) electricity demand to buses based on building floor area shares."""

import ast

import geopandas as gpd
import pandas as pd
import numpy as np


def load_cts_demand_per_building(shape):
    shape.to_crs(3035, inplace=True)
    building_share = gpd.read_file(
        "data/data_bundle/building_share", mask=shape
    )
    building_share.rename(
        columns={
            "zensus_id": "zensus_id",
            "bus_id": "bus_id",
            "profile_sh": "profile_share",
            "geometry": "geom",
        },
        inplace=True,
    )
    building_share["p_set"] = [[0] * 8760 for _ in range(len(building_share))]
    cts_bus = pd.read_csv(
        "data/data_bundle/cts_bus.csv",
        index_col="bus_id",
    )
    cts_bus = cts_bus[cts_bus.index.isin(building_share["bus_id"])].squeeze()
    cts_bus = cts_bus.apply(ast.literal_eval)
    building_share["p_set"] = building_share.apply(
        lambda x: [x.profile_share * n for n in cts_bus[x.bus_id]], axis=1
    )

    return building_share[["p_set", "geom"]]


def assign_cts_demand_to_buses(
    network,
    cts_demands,
    target_demand,
    mv_threshold_mw=0.1,      # 100 kW
    max_dist_lv=150.0,        # m
    max_dist_mv=None,         # None = all CTS loads >100 kW are connected to closest trafo
):
    """
    Assigns CTS loads to buses.

    Connection logic:
      - peak load after scaling <= 100 kW -> nearest LV house_connection bus
      - peak load after scaling > 100 kW  -> nearest MV-side transformer bus

    CTS profiles are assumed to be in MW and are scaled by 0.2 before insertion.
    """

    CTS_SCALING_FACTOR = 0.2

    # Prepare CTS demand GeoDataFrame
    cts_demands = cts_demands.copy()
    cts_demands = gpd.GeoDataFrame(cts_demands, geometry="geom")
    cts_demands.to_crs(crs=32632, inplace=True)

    # Determine peak load per CTS demand before and after scaling
    cts_demands["peak_load_mw_original"] = cts_demands["p_set"].apply(np.max)
    cts_demands["peak_load_mw_after_scaling"] = (
        cts_demands["peak_load_mw_original"] * CTS_SCALING_FACTOR
    )

    # Prepare LV candidate buses: house connections
    lv_buses = network.buses.copy()
    lv_buses = lv_buses[lv_buses["comp_type"] == "house_connection"].copy()
    lv_buses = lv_buses.reset_index(names="bus")
    lv_buses["geometry"] = lv_buses["geom"]

    lv_buses = gpd.GeoDataFrame(
        lv_buses,
        geometry="geometry",
        crs=32632,
    )

    # Prepare MV candidate buses: MV side of transformers, i.e. bus0
    if (
        hasattr(network, "transformers")
        and not network.transformers.empty
        and "bus0" in network.transformers.columns
    ):
        mv_bus_names = network.transformers["bus0"].dropna().unique()

        mv_buses = network.buses.loc[
            network.buses.index.intersection(mv_bus_names)
        ].copy()

        mv_buses = mv_buses.reset_index(names="bus")
        mv_buses["geometry"] = mv_buses["geom"]

        mv_buses = gpd.GeoDataFrame(
            mv_buses,
            geometry="geometry",
            crs=32632,
        )
    else:
        mv_buses = gpd.GeoDataFrame(
            columns=list(network.buses.columns) + ["bus", "geometry"],
            geometry="geometry",
            crs=32632,
        )

    # Fallback: use buses marked as trafo if transformer bus0 is not available
    if mv_buses.empty and "comp_type" in network.buses.columns:
        mv_buses = network.buses[
            network.buses["comp_type"] == "trafo"
        ].copy()

        mv_buses = mv_buses.reset_index(names="bus")
        mv_buses["geometry"] = mv_buses["geom"]

        mv_buses = gpd.GeoDataFrame(
            mv_buses,
            geometry="geometry",
            crs=32632,
        )

    # Split CTS loads by connection level
    # Threshold is applied after scaling
    cts_lv = cts_demands[
        cts_demands["peak_load_mw_after_scaling"] <= mv_threshold_mw
    ].copy()

    cts_mv = cts_demands[
        cts_demands["peak_load_mw_after_scaling"] > mv_threshold_mw
    ].copy()

    assigned_parts = []

    # Assign LV CTS loads to nearest house_connection bus
    if not cts_lv.empty and not lv_buses.empty:
        cts_lv = gpd.sjoin_nearest(
            cts_lv,
            lv_buses,
            how="left",
            distance_col="distance",
        )

        cts_lv = cts_lv[cts_lv["distance"] <= max_dist_lv].copy()
        cts_lv["voltage_level"] = "LV"

        assigned_parts.append(cts_lv)

    # Assign MV CTS loads to nearest MV-side transformer bus
    if not cts_mv.empty and not mv_buses.empty:
        cts_mv = gpd.sjoin_nearest(
            cts_mv,
            mv_buses,
            how="left",
            distance_col="distance",
        )

        # If max_dist_mv is None, all CTS loads >100 kW are connected
        # to the closest trafo bus, independent of distance.
        if max_dist_mv is not None:
            cts_mv = cts_mv[cts_mv["distance"] <= max_dist_mv].copy()

        cts_mv["voltage_level"] = "MV"

        assigned_parts.append(cts_mv)

    if not assigned_parts:
        print(
            """
            No CTS loads found close enough to valid LV or MV buses.
            Skipping CTS import.
            """
        )
        return network

    cts_demands = pd.concat(assigned_parts, axis=0)

    if cts_demands.empty:
        print(
            """
            No CTS loads found close enough to valid LV or MV buses.
            Skipping CTS import.
            """
        )
        return network

    # Create load names
    cts_demands["Load"] = cts_demands.apply(
        lambda b: f"CTS_Load_{b.name}_{b.bus}",
        axis=1,
    )
    cts_demands.set_index("Load", drop=True, inplace=True)

    cts_demands_t = cts_demands["p_set"].copy()

    # Scale demand down according to expected target peak demand
    peak_loads = cts_demands_t.apply(np.max)
    delete_count = 0

    while peak_loads.sum() > target_demand:
        worst = peak_loads.idxmax()

        peak_loads.drop(worst, inplace=True)
        cts_demands_t.drop(worst, inplace=True)
        cts_demands.drop(worst, inplace=True)

        delete_count += 1

    if cts_demands_t.empty:
        print("All CTS loads removed to meet target demand. Skipping CTS import.")
        return network
    else:
        print(f"{delete_count} CTS loads removed to meet target demand.")

    # Insert CTS loads into PyPSA network
    cts_demands["carrier"] = "conventional_load"
    cts_demands["sign"] = -1
    cts_demands["q_set"] = 0
    cts_demands["p_set"] = 0
    cts_demands["active"] = True

    network.loads = pd.concat(
        [
            network.loads,
            cts_demands[
                ["bus", "carrier", "type", "p_set", "q_set", "sign", "active"]
            ],
        ]
    )
    
    # Prepare CTS time series
    ts_df = pd.DataFrame(
        cts_demands_t.tolist(),
        index=cts_demands_t.index,
    ).T

    # Adjustment for scaling CTS demand down due to given values
    # in "Regional-Szenarien"
    ts_df = ts_df * CTS_SCALING_FACTOR
    ts_df.index = network.loads_t.p_set.index

    network.loads_t.p_set = pd.concat(
        [network.loads_t.p_set, ts_df],
        axis=1,
    )
        
    # Remove household demand at same buses
    if "household_count" in network.buses.columns:
        affected_buses = cts_demands["bus"].unique()
        network.buses.loc[affected_buses, "household_count"] = 0
        
    print(
        """
        CTS loads are successfully imported.
        """
    )

    return network


def inser_cts_demand_per_building(network, shape_path, target_demand=31):
    shape = gpd.read_file(shape_path)
    cts_demands = load_cts_demand_per_building(shape)
    network = assign_cts_demand_to_buses(network, cts_demands, target_demand)

    return network
