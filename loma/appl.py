#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:11:55 2025

@author: student
"""

from demands.create_household_distribution import create_household_dist
from demands.create_industrial_demand import insert_ind_demand_per_building
from demands.cts_demands import inser_cts_demand_per_building
from demands.import_household_demand import distribute_household_demand
from network.import_network_from_shape_files import create_pypsa_network
from demands.import_EV_demand import import_EV_loads
from demands.import_hp_demand import add_heat_loads_to_network

args = {
        "path_to_shapefiles_grid": 'data/Input_files/Filtered_data_Kronenburg_V3',  # define path of shapefiles for grid infrastructure (related to execution folder)
        "path_to_shapefile_MV_grid": 'data/Input_files/MV_grid_district/husum_district.shp',  #define path of shapefiles for boundaries of husum_district 
        "nuts3_focus_region": "Nordfriesland, Schleswig-Holstein, Germany",  
        "path_to_household_data": "data/Input_files/quantity_household_fixed.csv",
        "path_to_heat_pump_data": "data/Input_files/heat_pumps.csv",
        "Kabeltypen": {
            "NAYY 4x240": {"U": 400, "I_max": 364, "R": 0.125, "L": 0.254},     #values based on FaberKabel Starkstromkabel NAYY-J/-O nach VDE 0276-603 (same as dingO-grid-values)
            "NAYY 4x150": {"U": 400, "I_max": 275, "R": 0.206, "L": 0.256},     # U[v], I[A], R[Ohm/km], L[mH/km]
            "NAYY 4x95": {"U": 400, "I_max": 215, "R": 0.206, "L": 0.261},
            "NAYY 4x35": {"U": 400, "I_max": 123, "R": 0.868, "L": 0.271}
                    }
        }

#create pypsa network with grid topology shapefilees
n = create_pypsa_network(args['path_to_shapefiles_grid'], args['path_to_household_data'], args['path_to_heat_pump_data'], args['Kabeltypen'])

#insert cts demand
n = inser_cts_demand_per_building(n, args['path_to_shapefile_MV_grid'])

#insert industrial demands
n = insert_ind_demand_per_building(n,  args['path_to_shapefile_MV_grid'], args['nuts3_focus_region'])

#household-type distribution on 100x100m             ###ToDo: combine create_household_dist and distribute_household_demand
household_dist_df = create_household_dist(args['path_to_shapefile_MV_grid'])

#allocate profiles to buses
n = distribute_household_demand(n, household_dist_df)

#insert_heat_loas_for_heat_pump_location
n = add_heat_loads_to_network(n)

#insert EV_loads
n = import_EV_loads(n, args['path_to_shapefiles_grid'])

