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


def assign_cts_demand_to_buses(network, cts_demands, target_demand):
    buses = network.buses.copy()
    buses = buses[buses["comp_type"] == "house_connection"]
    buses = gpd.GeoDataFrame(buses, geometry="geom", crs=32632)
    
    cts_demands = gpd.GeoDataFrame(cts_demands, geometry="geom")
    cts_demands.to_crs(crs=32632, inplace=True)
    cts_demands = gpd.sjoin_nearest(
        cts_demands, buses, "left", distance_col="distance"
    )
    cts_demands = cts_demands[cts_demands["distance"] < 150]
    
    if cts_demands.empty:
          print("""
                ⚠️ No CTS loads found within 150m of any bus. Skipping CTS import.
                """)
          return network

    # ######## ONLY FOR VALIDATION PURPOSES ###############

    #cts_demands[["geom", "Bus", "distance"]].to_file(
    #     "/home/student/Documents/LoMa/Validation/cts_demands.shp"
    #)
    #buses.to_file("/home/student/Documents/LoMa/Validation/buses.shp")

    # #####################################################

    # insert data into network tables
    
    cts_demands.rename(columns={"name": "bus"}, inplace=True)
    cts_demands["Load"] = cts_demands.apply(
        lambda b: f"CTS_Load_{b.name}_{b.bus}", axis=1
    )
    cts_demands.set_index("Load", drop=True, inplace=True)

    cts_demands_t = cts_demands["p_set"].copy()

    
    #scale demand down according to expected value
    peak_loads = cts_demands_t.apply(np.max)
    delete_count=0
    while peak_loads.sum() > target_demand:
        worst = peak_loads.idxmax()
        peak_loads.drop(worst, inplace=True)
        cts_demands_t.drop(worst, inplace=True)
        cts_demands.drop(worst, inplace=True)
        delete_count += 1
      
    if cts_demands_t.empty:
        print("⚠️ All CTS loads removed to meet target demand. Skipping CTS import.")
        return network
    else:
        print(f"⚠️ {delete_count} CTS loads removed to meet target demand.")

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
    #implement timeseries für cts loads
    ts_df = pd.DataFrame(cts_demands_t.tolist(), index=cts_demands_t.index).T
    #adjustment for saling cts demand down due to given values in "Regional-Szenarien"
    ts_df = ts_df * 0.2
    ts_df.index = network.loads_t.p_set.index
    network.loads_t.p_set = pd.concat([network.loads_t.p_set, ts_df], axis=1)

    # Remove household demand at same buses
    if "household_count" in network.buses.columns:
        affected_buses = cts_demands["bus"].unique()
        network.buses.loc[affected_buses, "household_count"] = 0


    print("""
          ✅ CTS loads are succesfully imported.
          """)
    return network


def inser_cts_demand_per_building(network, shape_path, target_demand=31):
    # breakpoint()
    shape = gpd.read_file(shape_path)
    cts_demands = load_cts_demand_per_building(shape)
    network = assign_cts_demand_to_buses(network, cts_demands, target_demand)

    return network
