import pandas as pd
import geopandas as gpd
import ast
import pypsa
import math
import numpy as np
from shapely.geometry import Point


def load_cts_demand_per_building(shape):
    shape.to_crs(3035, inplace=True)
    building_share = gpd.read_file("data/data_bundle/building_share", mask=shape)
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


def assign_cts_demand_to_buses(network, cts_demands):
    buses = network.buses.copy()
    buses = buses[buses["comp_type"] == "house_connection"]

    ########## Temporal ########
    buses["x"] = buses["x"].apply(float)
    buses["y"] = buses["y"].apply(float)
    buses["geom"] = buses.apply(lambda x: Point(x.x, x.y), axis=1)
    buses = gpd.GeoDataFrame(buses, geometry="geom", crs="EPSG:25832")
    buses.to_crs(3035, inplace=True)
    ############################

    cts_demands = gpd.GeoDataFrame(cts_demands, geometry="geom")
    cts_demands = gpd.sjoin_nearest(
        cts_demands, buses, "left", distance_col="distance"
    )

    ######## ONLY FOR VALIDATION PURPOSES ###############
    '''
    cts_demands[["geom", "Bus", "distance"]].to_file(
        "validation/cts_demands.shp"
    )
    buses.to_file("validation/buses.shp")
    '''
    #####################################################

    # insert data into network tables
    cts_demands = cts_demands[["Bus", "p_set"]]    
    cts_demands.rename(columns={"Bus": "bus"}, inplace=True)
    import pdb; pdb.set_trace()
    cts_demands = cts_demands.groupby("bus").agg({
        "p_set": lambda series: list(np.sum(series.to_list(), axis=0))
    }).reset_index()
    cts_demands["Load"] = cts_demands["bus"].apply(lambda b: f"CTS_Load_{b}")
    cts_demands["carrier"] = "CTS"
    cts_demands.set_index("Load", drop=True, inplace=True)

    network.loads = pd.concat([network.loads, cts_demands[["bus", "carrier"]]])
    for l in cts_demands.index:
        network.loads_t.p_set[l] = cts_demands.at[l, "p_set"]

    return network


def inser_cts_demand_per_building(network, shape_path):
    ######## DELETE AFTER INTEGRACION ##########################
    # network = pypsa.Network(
    #     import_name="/home/carlos/Documents/LoMa/databundle/test_network"
    # )
    ############################################################
    shape = gpd.read_file(shape_path)
    cts_demands = load_cts_demand_per_building(shape)
    network = assign_cts_demand_to_buses(network, cts_demands)

    return network
