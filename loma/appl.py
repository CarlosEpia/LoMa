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
from loma.demands.import_household_demand import distribute_household_demand
from loma.demands.import_EV_demand import import_charging_points
from loma.demands.import_hp_demand import add_heat_loads_to_network

from loma.network.correct_meshed_grid import avoid_meshes_in_network
from loma.network.flexibilities_14a_heat_pump import (
    insert_heat_pump_flexibilities_14a,
)
from loma.network.import_network_from_shape_files import create_pypsa_network
from loma.plot_results import plot_results
from loma.pv_rooftop_and_home_battery.pv_rooftop_and_home_battery import (
    insert_pv_rooftop_and_battery,
)
from loma.pypsa_model_into_ding0_shape import (
    add_dummy_mv_grid,
    prepare_ding0_shape_export,
)
import pypsa
from shapely import wkt

from loma.pypsa_model_into_ding0_shape import (
    add_dummy_mv_grid,
    prepare_ding0_shape_export,
)


args = {
    "import_network_structure": False,  # "/home/carlos/LoMa/network_structures/MGB",
    "path_to_shapefiles_grid": "data/Input_files/shape_files_grid",  # define path of shapefiles for grid infrastructure (related to execution folder)
    "path_to_shapefile_MV_grid": "data/Input_files/MV_grid_district/husum_district.shp",  # define path of shapefiles for boundaries of husum_district
    "nuts3_focus_region": "Nordfriesland, Schleswig-Holstein, Germany",
    "path_to_household_data": "data/Input_files/all_streets_household_count.csv",
    "path_to_heat_pump_data": "data/Input_files/heat_pumps.csv",
    "batteries_path": "data/data_bundle/generators_and_batteries/batt_SH.geojson",
    "pv_rooftop_path": "data/data_bundle/generators_and_batteries/rooftop_SH.geojson",
    "pv_feedin_path": "data/data_bundle/generators_and_batteries/pv_feedin.csv",
    "switches_path": "data/Input_files/switches_Husum.shp",
    "use_census_household_data": True,
    "export_shape_files_grid": True,
    "Kabeltypen": {
        "NAYY 4x240": {
            "U": 400,
            "I_max": 364,
            "R": 0.125,
            "L": 0.254,
        },  # values based on FaberKabel Starkstromkabel NAYY-J/-O nach VDE 0276-603 (same as dingO-grid-values)
        "NAYY 4x150": {
            "U": 400,
            "I_max": 275,
            "R": 0.206,
            "L": 0.256,
        },  # U[v], I[A], R[Ohm/km], L[mH/km]
        "NAYY 4x95": {"U": 400, "I_max": 215, "R": 0.206, "L": 0.261},
        "NAYY 4x35": {"U": 400, "I_max": 123, "R": 0.868, "L": 0.271},
        "NA2XS(F)2Y 3x150": {
            "U": 20000,
            "I_max": 319,
            "R": 0.206,
            "L": 0.4011,
        },
    },
}

# household-type distribution on 100x100m             ###ToDo: combine create_household_dist and distribute_household_demand
household_dist_df = create_household_dist(args["path_to_shapefile_MV_grid"])

if args["import_network_structure"]:
    n = pypsa.Network()
    if n.c.shapes.static.crs is not None:
        n.c.shapes.static.set_crs(32632, allow_override=True, inplace=True)
    n.import_from_csv_folder(args["import_network_structure"])
    for df in [n.buses, n.lines, n.transformers]:
        if "geom" in df.columns and isinstance(df["geom"].iloc[0], str):
            df["geom"] = df["geom"].apply(wkt.loads)

else:
    # create pypsa network with grid topology shapefiles
    n = create_pypsa_network(
        args["path_to_shapefiles_grid"],
        args["path_to_household_data"],
        args["path_to_heat_pump_data"],
        args["Kabeltypen"],
        args["use_census_household_data"],
        args["export_shape_files_grid"],
        args["switches_path"],
        household_dist_df,
    )
    # avoid meshes in the grid
    # n = avoid_meshes_in_network(n)

    # insert solar_rooftop and home_batteries
    n = insert_pv_rooftop_and_battery(
        n,
        args["path_to_shapefile_MV_grid"],
        args["pv_rooftop_path"],
        args["pv_feedin_path"],
        args["batteries_path"],
    )

    # allocate profiles to buses
    n = distribute_household_demand(n, household_dist_df)

    # insert cts demand
    n = inser_cts_demand_per_building(n, args["path_to_shapefile_MV_grid"])

    # insert industrial demands
    n = insert_ind_demand_per_building(
        n, args["path_to_shapefile_MV_grid"], args["nuts3_focus_region"]
    )

    # insert_heat_loas_for_heat_pump_location
    n = add_heat_loads_to_network(n)

# insert EV_loads
n = import_charging_points(n, args["path_to_shapefiles_grid"])

# Manual fixes: To Do
n.lines.s_nom_extendable = False
n.transformers.s_nom_extendable = False
n.generators.control = "PQ"

if len(n.buses) < 1000:
    n = add_dummy_mv_grid(n)
else:
    n.add("Generator",
          name="HV_dummy_gen_slack",
          bus="bus_20111_HV",
          p_nom=10000,
          carrier="AC",
          control='Slack',
          marginal_cost=50)
    n.buses.at["bus_20111_HV", "control"] = "Slack"

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

n.export_to_csv_folder("results/MGB_model_pypsa")
