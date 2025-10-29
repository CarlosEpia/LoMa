#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:11:55 2025

@author: student
"""

from loma.demands.create_household_distribution import create_household_dist
from loma.demands.create_industrial_demand import (
    insert_ind_demand_per_building,
)
from loma.demands.cts_demands import inser_cts_demand_per_building
from loma.demands.import_household_demand import distribute_household_demand
from loma.network.import_network_from_shape_files import create_pypsa_network
from loma.demands.import_EV_demand import import_EV_loads
from loma.demands.import_EV_demand import import_EV_demands
from loma.demands.import_hp_demand import add_heat_loads_to_network
from loma.network.flexibilities_14a_heat_pump import (
    insert_heat_pump_flexibilities_14a,
)
from loma.constraints.constraints import load_reduction_constraint_14a
from loma.pv_rooftop_and_home_battery.pv_rooftop_and_home_battery import (
    insert_pv_rooftop_and_battery,
)
from loma.MGB_Model_into_ding0_shape import prepare_ding0_shape_export
from loma.plot_results import plot_results

args = {
    "path_to_shapefiles_grid": "data/Input_files/shape_files_grid",  # define path of shapefiles for grid infrastructure (related to execution folder)
    "path_to_shapefile_MV_grid": "data/Input_files/MV_grid_district/husum_district.shp",  # define path of shapefiles for boundaries of husum_district
    "nuts3_focus_region": "Nordfriesland, Schleswig-Holstein, Germany",
    "path_to_household_data": "data/Input_files/all_streets_household_count.csv",
    "path_to_heat_pump_data": "data/Input_files/heat_pumps.csv",
    "batteries_path": "data/data_bundle/generators_and_batteries/batt_SH.geojson",
    "pv_rooftop_path": "data/data_bundle/generators_and_batteries/rooftop_SH.geojson",
    "pv_feedin_path": "data/data_bundle/generators_and_batteries/pv_feedin.csv",
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
    },
}


#household-type distribution on 100x100m             ###ToDo: combine create_household_dist and distribute_household_demand
household_dist_df = create_household_dist(args['path_to_shapefile_MV_grid'])

#create pypsa network with grid topology shapefilees
n = create_pypsa_network(
    args['path_to_shapefiles_grid'], 
    args['path_to_household_data'], 
    args['path_to_heat_pump_data'], 
    args['Kabeltypen'], 
    args['use_census_household_data'], 
    args['export_shape_files_grid'], 
    household_dist_df)

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
#n = inser_cts_demand_per_building(n, args["path_to_shapefile_MV_grid"])

# insert industrial demands
#n = insert_ind_demand_per_building(
#    n, args["path_to_shapefile_MV_grid"], args["nuts3_focus_region"]
#)

# insert_heat_loas_for_heat_pump_location
n = add_heat_loads_to_network(n)

# insert EV_loads
n = import_EV_loads(n, args["path_to_shapefiles_grid"])
n = import_EV_demands(n)

# # insert heat pump flexibilities
#heat_loads = n.loads[n.loads.index.str.contains("heat")]
#n.loads_t["p_set"].loc[:,heat_loads.index] = n.loads_t["p_set"].loc[:,heat_loads.index] * 50
n = insert_heat_pump_flexibilities_14a(n)

snapshots = 24
# Optimize
n.optimize(
    snapshots=n.snapshots[:snapshots],
    solver_name="glpk",
    extra_functionality=load_reduction_constraint_14a,
)

n = plot_results(n)

#export model into ding0_shape ####
#### define own export_folder in arguments of the functions
#prepare_ding0_shape_export(n, '/home/student/Execution/LoMa_exe/results/MGB_model')


