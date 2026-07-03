#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:11:55 2025

@author: student
"""

from datetime import datetime

from loma.constraints.constraints import load_reduction_constraint_14a
from loma.demands.create_household_distribution import create_household_dist
from loma.demands.create_industrial_demand import (
    insert_ind_demand_per_building,
)
from loma.demands.cts_demands import inser_cts_demand_per_building
from loma.demands.import_household_demand import (distribute_household_demand, define_slp_as_load_profile)
from loma.demands.import_EV_demand import import_charging_points
from loma.demands.import_hp_demand import add_heat_loads_to_network

from loma.network.correct_meshed_grid import avoid_meshes_in_network
from loma.network.flexibilities_14a_heat_pump import (
    insert_heat_pump_flexibilities_14a,
)
from loma.network.load_cable_types import load_kabeltypen
from loma.network.load_project_config import load_project_config
from loma.network.import_network_from_shape_files import create_pypsa_network
from loma.plot_results import plot_results
from loma.pv_rooftop_and_home_battery.pv_rooftop_and_home_battery import (
    insert_pv_rooftop_and_battery,
)
from loma.pypsa_model_into_ding0_shape import (
    prepare_ding0_shape_export,
)
import pypsa
from shapely import wkt

from loma.pypsa_model_into_ding0_shape import (
    add_dummy_mv_grid,
    prepare_ding0_shape_export,
)


PROJECT_CONFIG_PATH = "data/Input_files/project_config_husum.yaml"
project_config = load_project_config(PROJECT_CONFIG_PATH)

args = {
    "import_network_structure": False,  # "/home/carlos/LoMa/network_structures/MGB",
    "path_to_shapefiles_grid": project_config["paths"]["shapefiles_grid"],
    "path_to_shapefile_MV_grid": project_config["paths"]["mv_grid_boundary"],
    "nuts3_focus_region": project_config["nuts3_focus_region"],
    "path_to_household_data": project_config["paths"]["household_data"],
    "batteries_path": project_config["paths"]["batteries"],
    "pv_rooftop_path": project_config["paths"]["pv_rooftop"],
    "pv_feedin_path": project_config["paths"]["pv_feedin"],
    "switches_path": project_config["paths"]["switches"],
    "use_census_household_data": True,
    "export_shape_files_grid": True,
    "Kabeltypen": { # U[V], I[A], R[Ohm/km], L[mH/km]
        "Default_LV": {
            "U": 400,"I_max":500, "R": 0.3, "L": 0.25,
        },
        "Default_MV": {
            "U": 20000, "I_max":500, "R": 0.2, "L": 0.3,
        },
    },
    "path_to_cable_types": project_config["paths"]["cable_types"],
}

#add all cable_types to args
args["Kabeltypen"].update(load_kabeltypen(args["path_to_cable_types"]))

# household-type distribution on 100x100m             ###ToDo: combine create_household_dist and distribute_household_demand
household_dist_df = create_household_dist(args["path_to_shapefile_MV_grid"])


if args["import_network_structure"]:
    n = pypsa.Network()
    epsg_code = int(str(project_config["project"]["crs"]).split(":")[-1])
    if n.c.shapes.static.crs is not None:
        n.c.shapes.static.set_crs(epsg_code, allow_override=True, inplace=True)
    n.import_from_csv_folder(args["import_network_structure"])
    for df in [n.buses, n.lines, n.transformers]:
        if "geom" in df.columns and isinstance(df["geom"].iloc[0], str):
            df["geom"] = df["geom"].apply(wkt.loads)

else:
    # create pypsa network with grid topology shapefiles
    n = create_pypsa_network(
        args["path_to_shapefiles_grid"],
        args["path_to_household_data"],
        args["Kabeltypen"],
        args["use_census_household_data"],
        args["export_shape_files_grid"],
        args["switches_path"],
        household_dist_df,
        project_config,
    )
    
    # avoid meshes in the grid
    #n = avoid_meshes_in_network(n)

    # insert solar_rooftop and home_batteries
    n = insert_pv_rooftop_and_battery(
        n,
        args["path_to_shapefile_MV_grid"],
        args["pv_rooftop_path"],
        args["pv_feedin_path"],
        args["batteries_path"],
        project_config,
    )

    # allocate profiles to buses
    n = distribute_household_demand(n, household_dist_df)

    #change profiles to SLP 
    n = define_slp_as_load_profile(n)
    
    # insert cts demand
    n = inser_cts_demand_per_building(n, args["path_to_shapefile_MV_grid"])

    # insert industrial demands
    n = insert_ind_demand_per_building(
        n, args["path_to_shapefile_MV_grid"], args["nuts3_focus_region"]
    )

    # insert_heat_loas_for_heat_pump_location
    n = add_heat_loads_to_network(n, project_config)

# insert EV_loads
n = import_charging_points(n, args["path_to_shapefiles_grid"], project_config)


n.lines.s_nom_extendable = False
n.transformers.s_nom_extendable = False

if project_config["project"]["is_test_model"]:  ##delete cts loads for test models
      loads_to_remove = n.loads.index[n.loads.index.str.contains("CTS")]  #
      n.remove("Load", loads_to_remove)


n.consistency_check()

print(f"Start Optimierung: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Optimize
# n.optimize(
#     snapshots=n.snapshots[0:1],
#     solver_name="highs",
#     solver_options={
#         "threads": 4,
#     },
# )
print(f"Ende Optimierung:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#n.export_to_csv_folder("results/MGB_model_pypsa")
n.export_to_csv_folder("results/Whole_Husum_final_statusQuo_LV_ids")

