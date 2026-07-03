#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 07:53:13 2025

@author: student
"""

import os
import zipfile
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import pandas as pd

#define folder for downloading census_data
target_file_household_type = "data/data_bundle/census_22_householdtyp_100m.zip"   
target_file_population = "data/data_bundle/census_22_population.zip"
target_file_hh_size = "data/data_bundle/census_22_hh_size.zip"

#define folder where excel_file with household-distibution is stored (more information:get_census_households_nuts3_raw() )
path_to_dist_data = "data/data_bundle/1000A-3098_de.xlsx"

# Define paths to shapefiles for the federal state and its NUTS3 districts,
# used to clip/assign the bundled nationwide census data to the project region.
# Currently Schleswig-Holstein specific: a Stadtwerk outside SH must replace
# both shapefiles with the equivalent boundaries for its own federal state.
path_to_sh_shape = 'data/data_bundle/schleswig_holstein.shp'
path_to_nuts3_sh_shapes = 'data/data_bundle/Nuts3_SH.shp'



def clean(x):
    """Clean zensus household data row-wise

    Clean dataset by

    * converting '.' and '-' to str(0)
    * removing brackets

    Table can be converted to int/floats afterwards

    Parameters
    ----------
    x: pd.Series
        It is meant to be used with :code:`df.applymap()`

    Returns
    -------
    pd.Series
        Re-formatted data row
    """
    try:
        # convert to string for consistent cleaning
        x = str(x).strip()
        x = x.replace("-", "0").replace(".", "0").strip("()")

        # try converting cleaned value to int
        return int(float(x))  # allows conversion of '123.0' or 'nan' safely
    except (ValueError, TypeError):
        return 0

def download_and_check(url, target_file, max_iteration=5):
    """Download file from url (http) if it doesn't exist and check afterwards.
    If bad zip remove file and re-download. Repeat until file is fine or
    reached maximum iterations."""
    
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    
    bad_file = True
    count = 0
    while bad_file:

        # download file if it doesn't exist
        if not os.path.isfile(target_file):
            # check if url
            if url.lower().startswith("http"):
                urlretrieve(url, target_file)
            else:
                raise ValueError("No http url")

        # check zipfile
        try:
            with zipfile.ZipFile(target_file):
                print(f"Zip file {target_file} is good.")
            bad_file = False
        except zipfile.BadZipFile as ex:
            os.remove(target_file)
            count += 1
            if count > max_iteration:
                raise StopIteration(
                    f"Max iteration of {max_iteration} is exceeded"
                ) from ex
                
def read_csv_from_zip(zip_path, csv_filename_in_zip):
    """Read CSV from inside ZIP, create geometry from x/y columns."""
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_filename_in_zip) as f:
            df = pd.read_csv(f, delimiter=';')
    return df



def create_geometry_for_census_cells(df, path_sh_shape, path_nuts3_shape):
    """
    Create a GeoDataFrame with census cell points and spatially join them 
    with NUTS3 regions, filtered by a given SH boundary.

    Parameters
    ----------
    df : pd.DataFrame
    
    path_sh_shape : str or Path
        File path to a shapefile containing the SH boundary (used to filter points).
    
    path_nuts3_shape : str or Path
        File path to a shapefile with NUTS3 geometries for spatial join.

    Returns
    -------
    gdf_sh_with_nuts3 : gpd.GeoDataFrame
        GeoDataFrame containing census points located within SH and enriched
        with corresponding NUTS3 region names.
    """
    # Create GeoDataFrame from x/y coordinates
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['x_mp_100m'], df['y_mp_100m']),
        crs="EPSG:3035"
    )

    # Transform to WGS84 for compatibility with most shapefiles
    gdf = gdf.to_crs("EPSG:4326")

    # Load SH region boundary and reduce to a single geometry
    sh_geometry = gpd.read_file(path_sh_shape)
    sh_geometry = sh_geometry.union_all()

    # Keep only points within the SH boundary
    gdf_sh = gdf[gdf.geometry.intersects(sh_geometry)]

    # Load NUTS3 geometries for spatial join
    geom_nuts3 = gpd.read_file(path_nuts3_shape)
    geom_nuts3 = geom_nuts3.to_crs('EPSG:4326')
    
    # Spatial join with NUTS3 regions (add 'gen' column from NUTS3 shapefile)
    gdf_sh_with_nuts3 = gpd.sjoin(
        gdf_sh,
        geom_nuts3[["gen", "geometry"]],
        how="left",
        predicate="intersects"
    )

    return gdf_sh_with_nuts3

def get_census_households_nuts3_raw_V2(path_to_file):
    """Get zensus age x household type data from LoMa-data-bundle

    Dataset about household size with information about the categories:

    * family type
    * age class
    * household size

    for Germany in spatial resolution of federal states NUTS-3.

    Data manually selected and retrieved from:
    https://ergebnisse.zensus2022.de/datenbank/online/statistic/1000A/table/1000A-3098/ 
    For reproducing data selection, please do:

    * Search for: "1000A-3098"
    * or choose topic: "Bevölkerung kompakt"
    * Choose table code: "1000A-3098" with title "PPersonen: Alter (10er-Jahresgruppen)/Alter (11 Altersklassen) - 
    Größe des privaten Haushalts - Typ des privaten Haushalts (Familien)"
    - Change setting "Merkmal: Deutschland" to "Landkreise u. krsfr.Städte"
    - Filter only for districts in Schleswig-Holstein
    - Change setting "Merkmal: Alter" to "Alter (11 Altersklassen)"


    Returns
    -------
    pd.DataFrame
        Pre-processed zensus household data
    """

    households_raw = pd.read_excel(path_to_file,
                        skiprows=2, 
                        skipfooter=2,
                        index_col = [2,3,4],
                        header = [0,1,2])

    return households_raw


def process_nuts3_hh_distribution_data_V2(df_household_age_dist):
    """Make data compatible with household demand profile categories for new dataset

    Removes and reorders categories which are not needed to fit data to
    household types of IEE electricity demand time series generated by
    demand-profile-generator (DPG).

    * Kids (<15) are excluded as they are also excluded in DPG origin dataset
    * Adults (15<65)
    * Seniors (<65)

    Parameters
    ----------
    zensus_nuts1_family_data: pd.DataFrame
        zensus data with MultiIndex columns and rows

    Returns
    -------
    pd.DataFrame
        Aggregated zensus household data on NUTS-1 level
    """
    # Clean data to int only
    
    df_household_age_dist_clean = df_household_age_dist.applymap(clean).applymap(int)

    # Define the age categories
    kids = ["Unter 3 Jahre", "3 bis 5 Jahre", "6 bis 14 Jahre"]  # < 15
    adults = ["15 bis 17 Jahre", "18 bis 24 Jahre", "25 bis 29 Jahre", "30 bis 39 Jahre", "40 bis 49 Jahre", "50 bis 64 Jahre"]  # 15 < x < 65
    seniors = ["65 bis 74 Jahre", "75 Jahre und älter"]  # > 65

    # Group data based on age categories
    df_kids = df_household_age_dist_clean.loc[
     df_household_age_dist_clean.index.get_level_values(0).isin(kids)]  
    df_adults = df_household_age_dist_clean.loc[
     df_household_age_dist_clean.index.get_level_values(0).isin(adults)] 
    df_seniors = df_household_age_dist_clean.loc[
        df_household_age_dist_clean.index.get_level_values(0).isin(seniors)]
    
    # Group by remaining levels (persons, household_type), sum across age
    df_kids = df_kids.groupby(level=[1, 2]).sum()
    df_adults = df_adults.groupby(level=[1, 2]).sum()
    df_seniors = df_seniors.groupby(level=[1, 2]).sum()
    
    # Concatenate the age groups into a new DataFrame
    df_census_households = pd.concat(
        [df_kids, df_adults, df_seniors],
        axis=0,
        keys=["Kids", "Adults", "Seniors"],
        names=["age", "persons", "household_type"]
    )
    
    df_census_households = df_census_households.sort_index(level=["age", "persons", "household_type"])
    
    #deleting unnecessary columns 
    df_census_households = df_census_households.iloc[:, 2:]
    # delete unnecessary columns-level
    df_census_households.columns = df_census_households.columns.get_level_values(0)
    # delete duplicate columns
    df_census_households = df_census_households.loc[:, ~df_census_households.columns.duplicated(keep='first')]


    return df_census_households


def regroup_nuts3_census_data_V2(df_census_households_nuts3): 
    """
    Regroup census data (on Kreis level) and map according to demand-profile types.
    
    Parameters
    ----------
    df_census_households_nuts1: pd.DataFrame
        census household data on Kreis level in absolute values

    Returns
    ----------
    df_hh_distribution_abs: pd.DataFrame
        Distribution of household types per district (columns = Kreise, rows = demand profile types)
    """
    
    # Mapping dict
    profile_types_map = {
    "SR": [
        ("Seniors", "Insgesamt", "Einpersonenhaushalte (Singlehaushalte)"),
        ("Seniors", "Insgesamt", "Alleinerziehende Elternteile"),
    ],
    "SO": [
        ("Adults", "Insgesamt", "Einpersonenhaushalte (Singlehaushalte)")
    ],
    "SK": [
        ("Adults", "Insgesamt", "Alleinerziehende Elternteile")
    ],
    "PR": [
        ("Seniors", "2 Personen", "Paare ohne Kind(er)"),
        ("Seniors", "2 Personen", "Mehrpersonenhaushalte ohne Kernfamilie"),
    ],
    "PO": [
        ("Adults", "2 Personen", "Paare ohne Kind(er)"),
        ("Adults", "2 Personen", "Mehrpersonenhaushalte ohne Kernfamilie"),
    ],
    "P1": [
        ("Adults", "3 Personen", "Paare mit Kind(ern)")
    ],
    "P2": [
        ("Adults", "4 Personen", "Paare mit Kind(ern)")
    ],
    "P3": [
        ("Adults", "5 Personen", "Paare mit Kind(ern)"),
        ("Adults", "6 und mehr Personen", "Paare mit Kind(ern)"),
    ],
    "OR": [
        ("Seniors", "3 Personen", "Mehrpersonenhaushalte ohne Kernfamilie"),
        ("Seniors", "4 Personen", "Mehrpersonenhaushalte ohne Kernfamilie"),
        ("Seniors", "5 Personen", "Mehrpersonenhaushalte ohne Kernfamilie"),
        ("Seniors", "6 und mehr Personen", "Mehrpersonenhaushalte ohne Kernfamilie"),
        ("Seniors", "3 Personen", "Paare mit Kind(ern)"),
        ("Seniors", "3 Personen", "Paare ohne Kind(er)"),
        ("Seniors", "4 Personen", "Paare mit Kind(ern)"),
        ("Seniors", "4 Personen", "Paare ohne Kind(er)"),
        ("Seniors", "5 Personen", "Paare mit Kind(ern)"),
        ("Seniors", "5 Personen", "Paare ohne Kind(er)"),
        ("Seniors", "6 und mehr Personen", "Paare mit Kind(ern)"),
        ("Seniors", "6 und mehr Personen", "Paare ohne Kind(er)"),
    ],
    "OO": [
        ("Adults", "3 Personen", "Mehrpersonenhaushalte ohne Kernfamilie"),
        ("Adults", "4 Personen", "Mehrpersonenhaushalte ohne Kernfamilie"),
        ("Adults", "5 Personen", "Mehrpersonenhaushalte ohne Kernfamilie"),
        ("Adults", "6 und mehr Personen", "Mehrpersonenhaushalte ohne Kernfamilie"),
        ("Adults", "3 Personen", "Paare ohne Kind(er)"),
        ("Adults", "4 Personen", "Paare ohne Kind(er)"),
        ("Adults", "5 Personen", "Paare ohne Kind(er)"),
        ("Adults", "6 und mehr Personen", "Paare ohne Kind(er)"),
    ],
    }

    df_hh_distribution_abs = pd.DataFrame(index=profile_types_map.keys(), columns=df_census_households_nuts3.columns)
    
    #sum up defined values for each category defined in profile_types_map
    for profile_type, conditions in profile_types_map.items():
        idx_to_sum = [idx for idx in df_census_households_nuts3.index if idx in conditions]
        df_hh_distribution_abs.loc[profile_type] = df_census_households_nuts3.loc[idx_to_sum].sum()

    # Delete empty columns
    df_hh_distribution_abs = df_hh_distribution_abs.loc[:, (df_hh_distribution_abs != 0).any(axis=0)]

    return df_hh_distribution_abs

def inhabitants_to_households_V2(df_hh_people_distribution_abs):
    """
    Convert number of inhabitants to number of household types.

    Takes the distribution of people living in household types to
    calculate a distribution of household types by using a people-in-household
    mapping. Results are not rounded to int as they are used to calculate
    a relative distribution anyway.
    The CSV household size data is used to determine an average wherever
    the number of people is not trivial (OR, OO). Kids are not counted.

    Parameters
    ----------
    df_hh_people_distribution_abs: pd.DataFrame
        Grouped census household data on grid level (10km), 
        in absolute numbers of people per household type

    Returns
    ----------
    df_dist_households: pd.DataFrame
        Estimated number of households per type
    """
    #calculate average number of people living in OO/OR Households
    download_and_check("https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Groesse_des_privaten_Haushalts_in_Gitterzellen.zip", target_file_hh_size)
    df_hh_size = read_csv_from_zip(target_file_hh_size, "Zensus2022_Groesse_des_privaten_Haushalts_10km-Gitter.csv")
    columns = ["3_Personen", "4_Personen", "5_Personen", "6_Personen_und_mehr"]
    df_hh_size = df_hh_size[columns]
    df_hh_size = df_hh_size.replace("–", 0)
    df_hh_size = df_hh_size.apply(pd.to_numeric, errors='coerce')
    df_hh_size = df_hh_size.fillna(0)

    # Sum across all grid cells to get total number of people by household size
    total_households_per_size = df_hh_size.sum()
    OO_factor = (
        sum(total_households_per_size * [3, 4, 5, 6]) / total_households_per_size.sum()
    )

    # Mapping: household code → average number of people per household
    mapping_people_in_households = {
        "SR": 1,
        "SO": 1,
        "SK": 1,  # kids are excluded
        "PR": 2,
        "PO": 2,
        "P1": 2,  # kids are excluded
        "P2": 2,
        "P3": 2,
        "OR": OO_factor,
        "OO": OO_factor,
    }

    # Remove any keys in the DataFrame index that aren't in the mapping
    diff = set(df_hh_people_distribution_abs.index) ^ set(mapping_people_in_households.keys())

    if diff:
        for key in diff:
            mapping_people_in_households = dict(mapping_people_in_households)
            mapping_people_in_households.pop(key, None)
        print(f"Removed {diff} from mapping!")

    # Divide people by average people per household
    df_dist_households = df_hh_people_distribution_abs.div(
        mapping_people_in_households, axis=0
    )

    return df_dist_households



def filter_df_for_focus_region(df, path):
    "Filter DataFrame for focus region to accelerate the methodology"
    
    region = gpd.read_file(path)
    region = region.to_crs('EPSG:4326')
    region_geom = region.iloc[0].geometry
    df_region = df[df.geometry.within(region_geom)]
    
    return df_region



def impute_missing_hh_in_populated_cells(df_census_population_nuts3, df_census_household_nuts3):
    """
    Fill household data in populated cells without household information.

    For each grid cell with a population but missing household data, this function assigns
    a household distribution sampled from other cells with the same population.
    If no exact population match is available, the distribution from the next lower
    available population is used as a fallback.

    Parameters
    ----------
    df_census_population_nuts3 : pd.DataFrame
        Census population data at 100x100m resolution.
    df_census_household_nuts3 : pd.DataFrame
        Census household data at 100x100m resolution.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with imputed household data for previously missing cells.
    """
    # Merge population into household data
    df_w_hh = df_census_household_nuts3.merge(
        df_census_population_nuts3[['GITTER_ID_100m', 'Einwohner']],
        on='GITTER_ID_100m',
        how='left'
    )

    # Identify grid cells with population but no household data
    df_wo_hh = df_census_population_nuts3[
        ~df_census_population_nuts3['GITTER_ID_100m'].isin(df_census_household_nuts3['GITTER_ID_100m'])
    ].copy()
    df_wo_hh = df_wo_hh.sort_values('Einwohner').reset_index(drop=True)

    fallback_value = None
    columns_to_add = [
        'Insgesamt_Haushalte',
        'EinpersHH_SingleHH',
        'Paare_ohneKind',
        'Paare_mitKind',
        'Alleinerziehende',
        'MehrpersHHohneKernfam'
    ]

    for population in sorted(df_wo_hh['Einwohner'].unique(), reverse=True):
        available_populations = df_w_hh['Einwohner'].unique()

        if population in available_populations:
            fallback_value = population
            population_value = population
        elif fallback_value is not None:
            population_value = fallback_value
        else:
            # Skip if there's no fallback yet (i.e. population too high with no matches below)
            continue

        df_w_hh_pop = df_w_hh[df_w_hh['Einwohner'] == population_value]
        df_wo_hh_pop = df_wo_hh[df_wo_hh['Einwohner'] == population].copy()

        np.random.seed(42)
        for index, row in df_wo_hh_pop.iterrows():
            # Select a random cell
            random_id = np.random.choice(df_w_hh_pop['GITTER_ID_100m'].unique())
            random_distribution = df_w_hh_pop[df_w_hh_pop['GITTER_ID_100m'] == random_id]

            # Assign values
            df_wo_hh_pop.at[index, 'random_id'] = random_id
            for col in columns_to_add:
                df_wo_hh_pop.at[index, col] = random_distribution[col].values[0]

        df_w_hh = pd.concat([df_w_hh, df_wo_hh_pop], ignore_index=True)

    return df_w_hh



def add_missing_household_dist(df_census_household):
    """
    Fills missing household type distributions based on similar total household counts.
    Cells without household type data but with a valid total household count will receive 
    a distribution from a randomly selected cell with a matching or near-matching total.

    Parameters
    ----------
    df_census_household : pd.DataFrame
        Census household data at 100x100m resolution, 

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed household type distributions.
    """
    # Replace missing values ("–") with zero and reset index
    df = df_census_household.replace("–", 0).reset_index(drop=True)

    hh_cols = [
        "EinpersHH_SingleHH", "Paare_ohneKind", "Paare_mitKind",
        "Alleinerziehende", "MehrpersHHohneKernfam"
    ]

    df_with_dist = df[~(df[hh_cols] == 0).all(axis=1)]
    df_without_dist = df[(df[hh_cols] == 0).all(axis=1)]

    def assign_distribution(row, reference_df):
        """Finds a similar household count row and returns its distribution."""
        target = row["Insgesamt_Haushalte"]
        for tolerance in range(6):
            candidates = reference_df[
                (reference_df["Insgesamt_Haushalte"] >= target - tolerance) &
                (reference_df["Insgesamt_Haushalte"] <= target + tolerance)
            ]
            if not candidates.empty:
                return candidates.sample(1, random_state=42).iloc[0][hh_cols]
        print(f"⚠️ No matching cell found for GITTER_ID {row.get('GITTER_ID_100m', 'Unknown')}")
        return row[hh_cols]  # Keep as is if no match found

    # Apply imputation to first N rows only (adjust/remove condition as needed)
    for idx, row in df_without_dist.iterrows():
        df.loc[idx, hh_cols] = assign_distribution(row, df_with_dist)

    return df



def calculate_distribution_10_types_per_cell(df_household_grid_region, df_dist_households):
    """
    Disaggregates 5 household types into 10 subtypes per grid cell based on regional distributions.
    Uses predefined mappings to distribute the counts of 5 household types into 10 finer subtypes
    using region-specific probability weights.

    Parameters
    ----------
    df_household_grid_region : pd.DataFrame
        Grid-based household data at 100x100m resolution.

    df_dist_households : pd.DataFrame
        Relative distributions of 10 household subtypes per NUTS3 region.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with 10 household subtype columns and normalized shares.
    """

    # Mapping from 5 household types to 10 detailed subtypes
    mapping = {
        "EinpersHH_SingleHH": ["SR", "SO"],
        "Paare_ohneKind": ["PR", "PO"],
        "Paare_mitKind": ["P1", "P2", "P3"],
        "Alleinerziehende": ["SK"],
        "MehrpersHHohneKernfam": ["OR", "OO"],
    }

    hh_10_types_all = sum(mapping.values(), [])

    # Ensure clean indexing and initialize subtype columns
    df = df_household_grid_region.reset_index(drop=True).replace("–", 0)
    for col in hh_10_types_all:
        df[col] = 0.0

    # Normalize the household subtype distributions by region
    for subtypes in mapping.values():
        df_dist_households.loc[subtypes] = df_dist_households.loc[subtypes].div(
            df_dist_households.loc[subtypes].sum()
        )

    # Assign counts to subtype columns based on the probability distributions
    for hh_5_type, subtypes in mapping.items():
        for idx, row in df.iterrows():
            count = int(row[hh_5_type])
            nuts3 = row['gen']  # NUTS3 region name

            matching_cols = [col for col in df_dist_households.columns if f" {nuts3}" in col]

            if not matching_cols:
                print(f"⚠️ No matching region found for index {idx}, NUTS3: '{nuts3}'")
                continue

            probs = df_dist_households.loc[subtypes, matching_cols]

            for subtype in subtypes:
                if subtype not in probs.index or probs.empty:
                    print(f"⚠️ Missing probability for {subtype} in {nuts3}")
                    continue
                count_subtype = count * probs.loc[subtype].iloc[0]
                df.at[idx, subtype] = float(count_subtype)

    # Normalize subtype values to relative shares per cell
    df['sum'] = df[hh_10_types_all].sum(axis=1)
    for subtype in hh_10_types_all:
        df[subtype] = df[subtype] / df['sum'].replace(0, 1)  # Avoid division by zero

    return df

def create_household_dist(path_to_MV_district):
    """
    Execute all data processes and combine them to create output dataframe.
    Final dataframe has probabilities for household profiles (10 types) at
    100x100m resolution.
    Only runs if the output CSV does not already exist.
    """
    output_path = 'data/data_bundle/household_dist_df.csv'
    
    if os.path.exists(output_path):
        print(f"Household_distribution-File already exists at {output_path}. Loading existing file.")
        final_df = pd.read_csv(output_path)
        return final_df

    print("File not found. Starting data processing...")

    download_and_check('https://www.destatis.de/static/DE/zensus/gitterdaten/Typ_des_privaten_Haushalts_Familien.zip', target_file_household_type) 
    df_census_household_grid = read_csv_from_zip(target_file_household_type, "Typ_des_privaten_Haushalts_Familien/Zensus2022_Typ_priv_HH_Familie_100m-Gitter.csv")
    df_census_household_grid = create_geometry_for_census_cells(df_census_household_grid, path_to_sh_shape, path_to_nuts3_sh_shapes)
    
    download_and_check("https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Bevoelkerungszahl.zip", target_file_population)
    df_census_population_grid = read_csv_from_zip(target_file_population, "Zensus2022_Bevoelkerungszahl_100m-Gitter.csv")
    df_census_population_grid = create_geometry_for_census_cells(df_census_population_grid, path_to_sh_shape, path_to_nuts3_sh_shapes)

    df_census_household_grid_region = filter_df_for_focus_region(df_census_household_grid, path_to_MV_district)  
    df_census_population_grid_region = filter_df_for_focus_region(df_census_population_grid, path_to_MV_district)
    df_census_household_grid_region = impute_missing_hh_in_populated_cells(df_census_population_grid_region, df_census_household_grid_region)    
    df_census_household_grid_region = add_missing_household_dist(df_census_household_grid_region)
    
    df_household_age_dist = get_census_households_nuts3_raw_V2(path_to_dist_data) 
    dist_data = process_nuts3_hh_distribution_data_V2(df_household_age_dist)
    dist_data = regroup_nuts3_census_data_V2(dist_data)
    dist_data = inhabitants_to_households_V2(dist_data)
    
    final_df = calculate_distribution_10_types_per_cell(df_census_household_grid_region, dist_data)
    final_df.to_csv(output_path, index=False)

    print(f"Processing complete. File saved at {output_path}.")
    return final_df
