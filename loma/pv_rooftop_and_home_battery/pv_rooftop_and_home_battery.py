import ast
import geopandas as gpd
import pandas as pd
import logging


def insert_pv_rooftop_and_battery(
    network, shape_path, pv_rooftop_path, pv_feedin_path, batteries_path, scenario
):
    shape = gpd.read_file(shape_path).to_crs(32632)
    buses = network.buses.copy()
    buses = buses[buses["comp_type"] == "house_connection"]
    buses = gpd.GeoDataFrame(buses, geometry="geom", crs=32632)
    network = insert_pv_rooftop(network, shape, buses, pv_rooftop_path, pv_feedin_path, scenario)
    network = insert_home_battery(network, shape, buses, batteries_path, scenario)

    return network


def insert_pv_rooftop(network, shape, buses, pv_rooftop_path, pv_feedin_path, scenario):
    solar = gpd.read_file(pv_rooftop_path).to_crs(32632)
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
    
    #scale down amount of pvs according to scenario values
    target_capacities = {
        "Husum_statusQuo": 26.1,  # MWp
        "Husum_2035": 32.0,
    } 
    if scenario not in target_capacities:
        logging.warning(
            f"Szenario '{scenario}' not found! "
            f"Use whole pv-capacity of egon2035 data ({solar['capacity'].sum():.2f} MWp)."
        )
        target_p_nom = solar["capacity"].sum()
    else:
        target_p_nom = target_capacities[scenario]
        
    current_p_nom = solar["capacity"].sum()
    
    if current_p_nom > target_p_nom:
        logging.info(f"Reducing solar rooftop capacity from {current_p_nom:.2f} MWp to {target_p_nom:.2f} MWp")
        
        # Datensatz zufällig mischen (reproduzierbar durch random_state)
        solar = solar.sample(frac=1, random_state=42) 
        
        # Kumulierte Summe bilden und filtern
        solar["cumsum_p_nom"] = solar["capacity"].cumsum()
        solar = solar[solar["cumsum_p_nom"] <= target_p_nom].copy()
        
        logging.info(f"PV capacity reduction finished. Remaining pv: {len(solar)}")
       
    
    # insert data into network tables
    solar.rename(columns={"capacity": "p_nom"}, inplace=True)
    solar["Generator"] = solar.apply(lambda b: f"pv_roof_{b.name}_{b.bus}", axis=1)
    solar.set_index("Generator", drop=True, inplace=True)
    solar["carrier"] = "solar_rooftop"
    solar["efficiency"] = 0.9

    for name, row in solar.iterrows():  # Später eingefügt,weiß nicht ob das richtig ist
        network.add(
            "Generator",
            name=name,
            bus=row["bus"],
            carrier=row["carrier"],
            p_nom=float(row["p_nom"]),
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


def insert_home_battery(network, shape, buses, batteries_path, scenario):
    bat = gpd.read_file(batteries_path).to_crs(32632)
    bat = gpd.clip(bat, shape)
    bat = gpd.sjoin_nearest(bat, buses, "left", distance_col="distance")

    logging.warning(
        f"""
                    {len(bat[bat["distance"] > 50])} home_batteries
                    discarded because of distance to the closest bus
                    """
    )
    bat = bat[bat["distance"] <= 50]

    #scale down amount of pvs according to scenario values
    target_capacities = {
        "Husum_statusQuo": 2.6,  # MW
        "Husum_2035": 7.5,
    } 
    if scenario not in target_capacities:
        logging.warning(
            f"Szenario '{scenario}' not found! "
            f"Use whole home-batteries capacity of egon2035 data ({bat['p_nom'].sum():.2f} MWp)."
        )
        target_p_nom = bat["capacity"].sum()
    else:
        target_p_nom = target_capacities[scenario]
     
    current_p_nom = bat["p_nom"].sum()
    
    if current_p_nom > target_p_nom:
        logging.info(f"Reducing home-batteries capacity from {current_p_nom:.2f} MW to {target_p_nom:.2f} MW")
        
        # Datensatz zufällig mischen (reproduzierbar durch random_state)
        bat = bat.sample(frac=1, random_state=42) 
        
        # Kumulierte Summe bilden und filtern
        bat["cumsum_p_nom"] = bat["p_nom"].cumsum()
        bat = bat[bat["cumsum_p_nom"] <= target_p_nom].copy()
        
        logging.info(f"Home-batteries capacity reduction finished. Remaining batteries: {len(bat)}")

    # insert data into network tables
    bat.rename(columns={"Bus": "bus"}, inplace=True)
    bat["StorageUnit"] = bat.apply(lambda b: f"sto_unit_{b.name}_{b.bus}", axis=1)
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
