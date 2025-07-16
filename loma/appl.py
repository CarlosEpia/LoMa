#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:11:55 2025

@author: student
"""

from network.import_network_from_shape_files import create_pypsa_network
from demands.create_household_distribution import create_household_dist
from demands.import_household_demand import distribute_household_demand
from demands.cts_demands import inser_cts_demand_per_building

args = {
        "path_to_shapefiles_grid": 'data/Input_files/Filtered_data_Kronenburg_V2',  # define path of shapefiles for grid infrastructure (related to execution folder)
        "path_to_shapefile_MV_grid": 'data/Input_files/MV_grid_district/husum_district.shp',  #define path of shapefiles for boundaries of husum_district 
        "nuts3_focus_region": "Nordfriesland, Schleswig-Holstein, Germany"     
        }


#create pypsa network with grid topology shapefilees
n = create_pypsa_network(args['path_to_shapefiles_grid'])

#insert cts demand
n=inser_cts_demand_per_building(n, args['path_to_shapefile_MV_grid'])

#household-type distribution on 100x100m             ###ToDo: combine create_household_dist and distribute_household_demand
household_dist_df = create_household_dist(args['path_to_shapefile_MV_grid'])

#allocate profiles to buses
n = distribute_household_demand(n, household_dist_df)

