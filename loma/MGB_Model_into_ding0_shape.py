#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 09:10:41 2025

@author: student
"""
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
import pypsa
import os
import numpy as np

###translate pypsa network into ding0 shape


def add_dummy_mv_grid(n):
    """
    Add PyPSA-Network a dummy MV-grid:
        - new MV_bus with connected Generator
        - new HV-bus and HV/MV Transformer
    """

    mv_candidates = [bus for bus in n.buses.index if "MS" in bus]
    if not mv_candidates:
        raise ValueError("Kein MV-Bus mit 'MS' im Namen gefunden!")
    existing_mv_bus = mv_candidates[0]

    # Optional: Use coordinates from existing Mv_bus
    x_existing = (
        n.buses.at[existing_mv_bus, "x"] if "x" in n.buses.columns else 0
    )
    y_existing = (
        n.buses.at[existing_mv_bus, "y"] if "y" in n.buses.columns else 0
    )

    # New MV-bus
    mv_bus_name = "Busbar_mvgd_2095_MV"
    n.add(
        "Bus",
        name=mv_bus_name,
        v_nom=20,  #
        x=x_existing + 0.1,  # etwas versetzt
        y=y_existing + 0.1,
        carrier="AC",
        overwrite=True,
    )

    # New MV-line connection
    n.add(
        "Line",
        name="line_522",
        bus0=mv_bus_name,
        bus1=existing_mv_bus,
        x=0.004,
        r=0.005,
        s_nom=6.18342138302089,
        length=0.1,
        cable_type="NA2XS2Y 3x1x185 RM/25",
        overwrite=True,
    )

    hv_bus_name = "HV_dummy_bus"
    """
    # new trafo connection to HV-level
    hv_bus_name = "HV_dummy_bus"
    n.add("Bus",
          name=hv_bus_name,
          v_nom=110,
          x=x_existing + 0.1, 
          y=y_existing + 0.1,
          carrier="AC",
          overwrite=True)
    
    # New Genertaor 
    n.add("Generator",
          name="HV_dummy_gen_slack",
          bus=hv_bus_name,
          p_nom=1,            # 1 MW
          carrier="AC",
          control='Slack',
          marginal_cost=50,
          efficiency=0.9,
          overwrite=True)
    """
    n.add(
        "Transformer",
        name="MV_to_HV_dummy_trafo",
        bus0=hv_bus_name,
        bus1=mv_bus_name,
        s_nom=63,  # 2 MVA
        x=0.1,
        r=0.01,
        overwrite=True,
    )

    return n


def adjust_network_shape(n, export_path, mv_grid_id=35725, lv_grid_id=1):

    ##buses
    buses = pd.DataFrame(index=n.buses.index)
    buses["name"] = buses.index
    transformer = Transformer.from_crs(
        "EPSG:32632", "EPSG:4326", always_xy=True
    )
    buses["v_nom"] = buses["name"].apply(
        lambda x: 10 if "MS" in x else (10 if "MV" in x else 0.4)
    )
    buses["x"] = n.buses.x
    buses["y"] = n.buses.y
    buses["x"], buses["y"] = transformer.transform(
        buses["x"].values, buses["y"].values
    )
    buses["mv_grid_id"] = 35725  # mv_grid_id from ding0 Husum grid
    buses["lv_grid_id"] = buses["name"].apply(
        lambda x: np.nan if "MS" in x else (np.nan if "MV" in x else 1)
    )  # toDo: check how to define this value the right way
    buses["in_building"] = False

    ## generators
    generators = pd.DataFrame(index=n.generators.index)
    generators["name"] = generators.index
    generators["bus"] = n.generators.bus
    generators["p_nom"] = n.generators.p_nom
    generators["type"] = n.generators.carrier.apply(
        lambda x: (
            "solar"
            if "solar_rooftop" in x
            else ("station" if x == "AC" else "conventional")
        )
    )
    generators["control"] = n.generators.control
    generators["weather_cell_id"] = None
    generators["subtype"] = n.generators.carrier.apply(
        lambda x: (
            "pv_rooftop"
            if "solar_rooftop" in x
            else ("mv_substation" if x == "AC" else "unkown")
        )
    )
    generators["source_id"] = None
    generators["voltage_level"] = generators["name"].apply(
        lambda x: "lv" if "MS" not in x else "mv"
    )

    ### lines
    lines = pd.DataFrame(index=n.lines.index)
    lines["name"] = lines.index
    lines["bus0"] = n.lines.bus0
    lines["bus1"] = n.lines.bus1
    lines["length"] = n.lines.length
    lines["x"] = n.lines.x
    lines["r"] = n.lines.r
    lines["b"] = n.lines.b
    lines["s_nom"] = n.lines.s_nom
    lines["num_parallel"] = n.lines.num_parallel
    lines["type_info"] = (
        n.lines.cable_type
    )  ###ToDo check, if husum cable type match with edisgo requirements
    lines["kind"] = "cable"

    ### loads
    loads = pd.DataFrame(index=n.loads.index)
    loads["name"] = loads.index
    loads["bus"] = n.loads.bus
    loads["p_set"] = n.loads_t.p_set.max()
    loads["building_id"] = None
    loads["type"] = n.loads.carrier
    loads["annual_consumption"] = None
    loads["sector"] = loads[
        "name"
    ].apply(  ####toDo: distinguish heat pump loads
        lambda x: (
            "cts"
            if "cts" in x.lower()
            else "industrial" if "ind" in x.lower() else "residential"
        )
    )
    loads["number_households"] = 1
    loads["voltage_level"] = "lv "  ###adjust if there are also mv_loads

    ### network
    mv_grid_geom = gpd.read_file(
        "data/Input_files/MV_grid_district/husum_district.shp"
    )
    mv_grid_geom = mv_grid_geom.to_crs(4326)
    network = pd.DataFrame(
        {
            "name": mv_grid_id,
            "mv_grid_district_population": [22227],
            "mv_grid_district_geom": [mv_grid_geom.geometry.iloc[0]],
            "srid": [4326],
        }
    )

    ### transformers lv/mv     ###todo anpassen der trafo definition und exkludieren des HVMV trafos dabei
    trafos = n.transformers
    transformers = pd.DataFrame(index=trafos.index)
    transformers["name"] = trafos.index
    transformers["bus0"] = trafos.bus0
    transformers["bus1"] = trafos.bus1
    transformers["x"] = trafos.x
    transformers["r"] = trafos.r
    transformers["s_nom"] = trafos.s_nom
    transformers["type"] = (transformers["s_nom"] * 1e3).astype(int).astype(
        str
    ) + " kVA"
    transformers["type_info"] = (transformers["s_nom"] * 1e3).astype(
        int
    ).astype(
        str
    ) + " kVA"  ##ToDo: check if this format fits to edigo requirements

    ##transformer mv/hv 
    
    trafo_hv = n.transformers[
        n.transformers.comp_type =='trafo_HV'
    ]
    transformers_hv = pd.DataFrame(index=trafo_hv.index)
    transformers_hv["name"] = trafo_hv.index
    transformers_hv["bus0"] = trafo_hv.bus1
    transformers_hv["bus1"] = trafo_hv.bus0
    transformers_hv["x"] = np.nan
    transformers_hv["r"] = np.nan
    transformers_hv["s_nom"] = trafo_hv.s_nom
    transformers_hv["type"] = transformers_hv["s_nom"].astype(
        int
    ).astype(
        str
    ) + " MVA"  ##ToDo: check if this format fits to edigo requirements
    transformers_hv["type_info"] = transformers_hv["s_nom"].astype(
        int
    ).astype(str) + " MVA"

    ### links (not part of ding0-Output so no template )
    links = pd.DataFrame(index=n.links.index)
    links["name"] = links.index
    links["bus0"] = n.links.bus0
    links["bus1"] = n.links.bus1
    links["p_nom"] = n.links.p_nom
    links["efficiency"] = n.links.efficiency
    links["p_max_pu"] = n.links.p_max_pu

    #### export grid datframes to csv
    os.makedirs(export_path, exist_ok=True)
    buses.to_csv(os.path.join(export_path, "buses.csv"), index=False)
    generators.to_csv(os.path.join(export_path, "generators.csv"), index=False)
    loads.to_csv(os.path.join(export_path, "loads.csv"), index=False)
    transformers.to_csv(
        os.path.join(export_path, "transformers.csv"), index=False
    )
    lines.to_csv(os.path.join(export_path, "lines.csv"), index=False)
    network.to_csv(os.path.join(export_path, "network.csv"), index=False)
    transformers.to_csv(
        os.path.join(export_path, "transformers.csv"), index=False
    )
    transformers_hv.to_csv(
        os.path.join(export_path, "transformers_hvmv.csv"), index=False
    )
    links.to_csv(os.path.join(export_path, "links.csv"), index=False)
    # extra folder for storage_units.csv
    storage_folder = os.path.join(export_path, "storage_units_folder")
    os.makedirs(storage_folder, exist_ok=True)
    n.storage_units.to_csv(
        os.path.join(storage_folder, "storage_units.csv"), index=False
    )


###export load_timeseries
def export_timeseries(n, export_path):

    export_path = os.path.join(export_path, "timeseries")
    os.makedirs(export_path, exist_ok=True)

    # export load_timeseries
    loads_ts = n.loads_t.p_set
    loads_ts.to_csv(
        os.path.join(export_path, "load_timeseries.csv"), index=True
    )

    # export p_max_pu for load_shedding_gens
    gens_ts = n.generators_t.p_max_pu
    gens_ts.to_csv(
        os.path.join(export_path, "gen_p_max_pu_timeseries.csv"), index=True
    )


def prepare_ding0_shape_export(n, export_path):
    #n = add_dummy_mv_grid(n)
    adjust_network_shape(n, export_path)
    export_timeseries(n, export_path)
