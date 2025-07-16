#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 14:23:23 2025

@author: student
"""

import osmnx as ox
import geopandas as gpd
from sqlalchemy import create_engine
from geoalchemy2.types import Geometry
import pandas as pd

region_nuts3 = "Nordfriesland, Schleswig-Holstein, Germany"
path_to_MV_district = '/home/student/Documents/LoMa/shape_files/husum_district.shp'
path_to_nuts3_sh_shapes = '/home/student/Documents/LoMa/shape_files/Nuts3_SH.shp'
path_to_industrial_demandregio = '/home/student/Documents/LoMa/Code/data/data_bundle/demand_regio_industrial.csv'

#toDo: generalize the code for diffferent regions
#toDo: filter out just buildings
def download_osm_industrial_areas(region_nuts3):
    
    tags = {"landuse": "industrial"}
    
    # extract data
    gdf = ox.features_from_place(region_nuts3, tags)
    gdf = gdf.to_crs("EPSG:4326")
    gdf["area_ha"] = gdf.geometry.area / 10_000  # Fläche in Hektar

    #filter out data for defined region
    sh_nuts3 = gpd.read_file(path_to_nuts3_sh_shapes)
    region_name = region_nuts3.split(',')[0]
    sh_region = sh_nuts3[sh_nuts3.gen == region_name]
    gdf = gdf[gdf.geometry.intersects(sh_region.iloc[0].geometry)]
    
    #filter out several company-types which are not part of demand regio data but tagged as industrial in osm (equal to methodology in egon-data)
    blacklist = [
        "kraftwerk", "stadtwerke", "müllverbrennung", "müllverwertung", "abfall", "wertstoff",
        "olarpark", "gewerbegebiet", "gewerbepark", "heizwerk", "kläranlage", "klärwerk",
        "biogasanlage", "wasserwerk", "recyclinghof", "recyclingpark"
    ]

    def is_blacklisted(name):
        if pd.isna(name):
            return False
        name = name.lower()
        return any(b in name for b in blacklist)

    gdf = gdf[~gdf["name"].apply(is_blacklisted)]
    
    gdf["osm_id"] = gdf.index.get_level_values("id")
    ####todo: check if blacklisted methiod nis working
    
    
    #optional export for checking
    gdf.to_file('/home/student/Documents/LoMa/shape_files/test.shp')
    
    
    return gdf[['osm_id', 'name', 'landuse', 'operator', 'substance', 'substation', 'area_ha', 'geometry']]


gdf=download_osm_industrial_areas(region_nuts3)
    
demand_regio_gdf = pd.read_csv(path_to_industrial_demandregio)
SH_kreise = ['Dithmarschen', 'Herzogtum Lauenburg', 'Nordfriesland', 'Ostholstein', 'Pinneberg', 'Plön', 'Rendsburg-Eckernförde', 'Schleswig-Flensburg', 'Segeberg', 'Steinburg', 'Stormarn']

def download_demandregio_data(scn):
    df_demand = pd.read_sql(
    f"""
    SELECT nuts3, wz, demand
    FROM demand.egon_demandregio_cts_ind
    WHERE scenario = '{scn}'
    AND demand > 0
    AND wz IN (SELECT wz FROM demand.egon_demandregio_wz WHERE sector = 'industry')
    """,
    con=engine2
    ) 
    
    sh_nuts3 = gpd.read_file(path_to_nuts3_sh_shapes)
    
    merged_gdf = sh_nuts3.merge(df_demand, left_on="nuts", right_on="nuts3", how="left")
    
    return merged_gdf
    
demand_gdf = download_demandregio_data('status2019')
osm_gdf=gdf


def distribute_ind_demand(osm_gdf, demand_gdf):
    region_name = region_nuts3.split(',')[0]
    demand_region = demand_gdf[demand_gdf.gen == region_name]
    
    total_area = osm_gdf["area_ha"].sum()
    
    results = []
    for _, row in demand_region.iterrows():    
        wz = row["wz"]
        demand_total = row["demand"]
        demand_per_ha = demand_total / total_area
    
        df = osm_gdf.copy()
        df["wz"] = wz
        df["demand"] = df["area_ha"] * demand_per_ha
    
        results.append(df)
    
    final_df = pd.concat(results, ignore_index=True) 
    final_gdf = gpd.GeoDataFrame(
        final_df,
        geometry='geometry',       
        crs="EPSG:4326"  )          
    
    mv_district = gpd.read_file(path_to_MV_district)
    mv_district_geom = mv_district.iloc[0].geometry
    final_gdf_mv_district = final_gdf[final_gdf.geometry.intersects(mv_district_geom)]
        
    final_gdf_mv_district.to_postgis(
    name="osm_industrial_demand",
    con=engine,
    schema="public",
    index=False,
    if_exists="replace",
    )
    
    return final_gdf_mv_district

def calc_load_curves_from_osm_demand(final_gdf_mv_district, load_profiles_df):
    """
    Erzeuge stündliche Lastprofile je Gebäude (osm_id), 
    basierend auf WZ-spezifischen Jahresnachfragen.

    Parameters
    ----------
    osm_demand_df : pd.DataFrame
        Spalten: ['osm_id', 'wz', 'demand'] (in MWh pro Jahr)

    load_profiles_df : pd.DataFrame
        Spalten: ['wz', 'timestamp', 'load_factor'] (normiert auf 1/Jahr)

    Returns
    -------
    pd.DataFrame
        DataFrame mit Spalten ['osm_id', 'timestamp', 'load_mw']
        (Last je Gebäude je Stunde)
    """
    
    load_profiles = pd.read_sql(
        f"""SELECT wz, load_curve
        FROM demand.egon_demandregio_timeseries_cts_ind""",
        index_col="wz",
        con=engine2
    ).transpose()
    
    idx = pd.DatetimeIndex(
        pd.date_range(
            start=f"01/01/2011",   #2011 standart weather year
            end=f"01/01/2012",
            freq="H",
            inclusive="left",
        )
    )

    df = pd.DataFrame(index=idx, columns=load_profiles.columns)
    for col in df.columns:
        df[col] = load_profiles[col].load_curve
        
    
    timeseries = []
    for idx, row in final_gdf_mv_district.iterrows():
        wz = row['wz']
        osm_id = row['osm_id']
        profile = df[wz] * row['demand']  
        
        timeseries.append({
            "osm_id": osm_id,
            "wz": wz,
            "timeseries": profile.values  # Array oder Liste
        })
    
    load_profiles_osm = pd.DataFrame(timeseries)
        
    timeseries_final = load_profiles_osm.groupby('osm_id')['timeseries'].sum()  
    
    
    
    
    
    
    
    
    