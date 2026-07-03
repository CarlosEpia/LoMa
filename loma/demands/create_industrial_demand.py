#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Downloads OSM industrial areas for a NUTS3 region and assigns industrial electricity demand profiles to buildings."""

import geopandas as gpd
import osmnx as ox
import pandas as pd


def download_osm_industrial_areas(region_nuts3):
    """Download OSM "landuse=industrial" areas for a NUTS3 region, clip them
    to the project's federal state, and filter out non-industrial company
    types (power plants, utilities, waste/water treatment, ...)."""

    tags = {"landuse": "industrial"}

    # extract data
    gdf = ox.features_from_place(region_nuts3, tags)
    gdf = gdf.to_crs("EPSG:32632")
    gdf["area_ha"] = gdf.geometry.area / 10_000  # area in hectares

    #filter out data for defined region
    # Currently Schleswig-Holstein specific: a Stadtwerk outside SH must
    # replace this NUTS3-district shapefile with its own federal state's.
    sh_nuts3 = gpd.read_file('data/data_bundle/Nuts3_SH.shp')
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
        """Check whether `name` contains any blacklisted non-industrial keyword."""
        if pd.isna(name):
            return False
        name = name.lower()
        return any(b in name for b in blacklist)

    gdf = gdf[~gdf["name"].apply(is_blacklisted)]   
    gdf["osm_id"] = gdf.index.get_level_values("id")
    
    return gdf[['osm_id', 'name', 'landuse', 'operator', 'substance', 'substation', 'area_ha', 'geometry']]
    
    
def distribute_ind_demand(path_to_MV_district, region_nuts3):
    """Distribute demand_regio's industrial electricity demand (per sector,
    "wz") evenly by area across all OSM industrial buildings in the MV
    grid district."""

    #load demand_regio data
    demand_regio_ind = pd.read_csv('data/data_bundle/demand_regio_industrial.csv')
    region_name = region_nuts3.split(",")[0]
    demand_regio_ind = demand_regio_ind[demand_regio_ind.gen==region_name]
    
    #distribute industrial demand among industrial buildings
    ####ToDo: categorize industrial buildings (currently all buildings will have the same profile)
    osm_ind_df = download_osm_industrial_areas(region_nuts3)
    total_area = osm_ind_df["area_ha"].sum()
    
    results = []
    for _, row in demand_regio_ind.iterrows():    
        wz = row["wz"]
        demand_total = row["demand"]
        demand_per_ha = demand_total / total_area
        
        osm_ind_df_2 = osm_ind_df.copy()
        osm_ind_df_2['wz'] = wz
        osm_ind_df_2["demand"] = osm_ind_df["area_ha"] * demand_per_ha
    
        results.append(osm_ind_df_2)
    
    final_df = pd.concat(results, ignore_index=True) 
    final_gdf = gpd.GeoDataFrame(
        final_df,
        geometry='geometry',       
        crs="EPSG:32632"  )    
    
    mv_district = gpd.read_file(path_to_MV_district)
    mv_district_geom = mv_district.iloc[0].geometry
    final_gdf_mv_district = final_gdf[final_gdf.geometry.intersects(mv_district_geom)]
    
    return final_gdf_mv_district


def calc_load_curves_from_osm_demand(path_to_MV_district, region_nuts3):
    """Build an hourly load timeseries for each industrial building from its
    annual demand and its sector's standard load curve."""

    ind_demand_per_building = distribute_ind_demand(path_to_MV_district, region_nuts3)
   
    load_profiles_ind = pd.read_hdf('data/data_bundle/load_profiles_ind.hdf', index_col=0)    
    idx = pd.DatetimeIndex(
        pd.date_range(
            start="01/01/2023",
            end="01/01/2024",
            freq="h",
            inclusive="left",
        )
    )

    df = pd.DataFrame(index=idx, columns=load_profiles_ind.columns)
    for col in df.columns:
        df[col] = load_profiles_ind[col].load_curve
            
    timeseries = []
    for idx, row in ind_demand_per_building.iterrows():
        wz = row['wz']
        osm_id = row['osm_id']
        geom = row['geometry']
        profile = df[wz] * row['demand']  
        
        timeseries.append({
            "osm_id": osm_id,
            "wz": wz,
            "timeseries": profile.values ,
            "geometry": geom
        })
    
    load_profiles_osm = pd.DataFrame(timeseries)      
    timeseries_final = load_profiles_osm.groupby('osm_id', as_index=False).agg({
        'timeseries': 'sum',
        'geometry': 'first'   
    })
    
    return timeseries_final

def insert_ind_demand_per_building(n, path_to_MV_district, region_nuts3):
    """
    Connect industrial loads to the closest bus of the network 
    (if distance is not too big).

    Connection logic:
      - Industrial loads <= 100 kW peak load:
          use existing connection logic
          house_connection or trafo bus inside polygon,
          otherwise nearest such bus within 100 m

      - Industrial loads > 100 kW peak load:
          connect to closest trafo station
    """

    MV_THRESHOLD_MW = 0.1  # 100 kW = 0.1 MW
    
    # prepare timeseries
    timeseries = calc_load_curves_from_osm_demand(path_to_MV_district, region_nuts3)
    timeseries_gdf = gpd.GeoDataFrame(timeseries, geometry="geometry", crs="EPSG:32632")
    
    # prepare relevant buses for connection
    buses_gdf = gpd.GeoDataFrame(
        n.buses,
        geometry=gpd.points_from_xy(n.buses.x, n.buses.y),
        crs="EPSG:32632",
    )

    # Existing candidate buses for small industrial loads
    con_buses_gdf = buses_gdf[
        buses_gdf.comp_type.isin(["house_connection", "trafo"])
    ]

    # Candidate buses for large industrial loads
    # Prefer transformer MV-side buses, analogous to the EV logic.
    if (
        hasattr(n, "transformers")
        and not n.transformers.empty
        and "bus0" in n.transformers.columns
    ):
        trafo_bus_names = n.transformers["bus0"].dropna().unique()
        trafo_buses_gdf = buses_gdf.loc[
            buses_gdf.index.intersection(trafo_bus_names)
        ].copy()
    else:
        trafo_buses_gdf = gpd.GeoDataFrame(
            columns=buses_gdf.columns,
            geometry="geometry",
            crs="EPSG:32632",
        )

    # Fallback: use buses marked as trafo if transformer bus0 is not available
    if trafo_buses_gdf.empty:
        trafo_buses_gdf = buses_gdf[buses_gdf.comp_type == "trafo"].copy()

    # connect each load to closest bus
    for _, row in timeseries_gdf.iterrows():
        geom = row["geometry"]
        ts = row["timeseries"]
        polygon_centroid = geom.centroid

        peak_load_mw = pd.Series(ts).max()
        peak_load_kw = peak_load_mw * 1000.0

        chosen_bus = None

        # Loads > 100 kW are connected to the closest trafo station
        if peak_load_mw > MV_THRESHOLD_MW:
            if not trafo_buses_gdf.empty:
                distances = trafo_buses_gdf.geometry.distance(polygon_centroid)
                chosen_bus = distances.idxmin()
                print(f"{row['osm_id']} connected to MV-grid")
            else:
                print(
                    f"No trafo station found for industrial load {row['osm_id']} "
                    f"with peak load {peak_load_kw:.2f} kW; ...skipped"
                )

        # Existing logic for loads <= 100 kW
        else:
            buses_in_polygon = con_buses_gdf[
                con_buses_gdf.geometry.within(geom)
            ]

            if not buses_in_polygon.empty:
                distances = buses_in_polygon.geometry.distance(polygon_centroid)
                chosen_bus = distances.idxmin()
            else:
                distances = con_buses_gdf.geometry.distance(polygon_centroid)
                nearest_buses = distances[distances < 100]

                if not nearest_buses.empty:
                    chosen_bus = nearest_buses.idxmin()
                else:
                    print(
                        f"No nearby bus found for connecting industrial load "
                        f"{row['osm_id']} (osm_id); ...skipped"
                    )

        if chosen_bus is not None:
            load_name = f"Ind_Load_{chosen_bus}_{row['osm_id']}"

            n.add(
                "Load",
                name=load_name,
                bus=chosen_bus,
                carrier="conventional_load",
                p_set=0.0,
            )

            n.loads_t.p_set[load_name] = pd.Series(ts, index=n.snapshots)
            
            # secure that no household demand is connected to the same bus
            if "household_count" in n.buses.columns:
                n.buses.loc[chosen_bus, "household_count"] = 0
   
    print(
        '''
          Industrial loads are successfully imported
          '''
    )
            
    return n
