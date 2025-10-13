import ast
import geopandas as gpd
import pandas as pd
import logging


def insert_pv_rooftop_and_battery(
    network, shape_path, pv_rooftop_path, pv_feedin_path, batteries_path
):
    shape = gpd.read_file(shape_path).to_crs(32632)
    buses = network.buses.copy()
    buses = gpd.GeoDataFrame(buses, geometry="geom", crs=32632)
    network = insert_pv_rooftop(
        network, shape, buses, pv_rooftop_path, pv_feedin_path
    )
    network = insert_home_battery(network, shape, buses, batteries_path)

    return network


def insert_pv_rooftop(network, shape, buses, pv_rooftop_path, pv_feedin_path):
    solar = gpd.read_file(pv_rooftop_path).to_crs(32632)
    solar = gpd.clip(solar, shape)
    solar = gpd.sjoin_nearest(solar, buses, "left", distance_col="distance")

    logging.warning(
        f"""
                    {len(solar[solar["distance"] > 15])} pv_rooftop generators
                    discarded because of distance to the closest bus
                    """
    )
    solar = solar[solar["distance"] <= 15]

    # insert data into network tables
    solar.rename(columns={"Bus": "bus", "capacity": "p_nom"}, inplace=True)
    solar["Generator"] = solar.apply(
        lambda b: f"pv_roof_{b.name}_{b.bus}", axis=1
    )
    solar.set_index("Generator", drop=True, inplace=True)
    solar["carrier"] = "solar_rooftop"
    solar["efficiency_dispatch"] = 0.9

    for name, row in solar.iterrows():  # Später eingefügt,weiß nicht ob das richtig ist
        network.add(
        "Generator",
        name=name,
        bus=row["bus"],
        carrier=row["carrier"],
        p_nom=float(row["p_nom"]),
        efficiency_dispatch=float(row["efficiency_dispatch"]),
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


def insert_home_battery(network, shape, buses, batteries_path):
    bat = gpd.read_file(batteries_path).to_crs(32632)
    bat = gpd.clip(bat, shape)
    bat = gpd.sjoin_nearest(bat, buses, "left", distance_col="distance")

    logging.warning(
        f"""
                    {len(bat[bat["distance"] > 15])} home_batteries
                    discarded because of distance to the closest bus
                    """
    )
    bat = bat[bat["distance"] <= 15]

    # insert data into network tables
    bat.rename(columns={"Bus": "bus"}, inplace=True)
    bat["StorageUnit"] = bat.apply(
        lambda b: f"sto_unit_{b.name}_{b.bus}", axis=1
    )
    bat.set_index("StorageUnit", drop=True, inplace=True)

    bat["carrier"] = "home_battery"
    bat["sign"] = 1
    bat["max_hours"] = bat["capacity"] / bat["p_nom"]
    bat["control"] = "PQ"
    bat["p_nom_extendable"] = False

    for name, row in bat.iterrows():
        network.add(
        "StorageUnit",
        name=name,
        bus=row["bus"],
        carrier=row["carrier"],
        p_nom=float(row["p_nom"]),
        max_hours=float(row["max_hours"]),
    )

    return network
