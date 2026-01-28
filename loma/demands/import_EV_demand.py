#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree

def import_charging_points(n, input_folder):
    """
    Reads charging point positions from the shape file and adds a static load
    where p_set is the kW amount of the charging station extracted from the 'BEMERKUNG' column.
    If 'BEMERKUNG' contains e.g. '3x11kW', three individual charging points
    with 11 kW each are created.
    """

    EV_locations = gpd.read_file(
        os.path.join(input_folder, "Gis ST Ladesäule Position.shp")
    ).to_crs("EPSG:32632")

    con_buses = n.buses[n.buses.comp_type == "house_connection"].copy()

    MAX_ASSIGN_DIST = 100.0  # meters (50 -> 357 records) (100 -> 359 records)

    # ------------------------------------------------------------------
    # Parser: extract number of charging points and kW per point
    # ------------------------------------------------------------------
    def parse_bemerkung(value: str):
        """
        Returns (n_points, kw_per_point)
        """
        if not isinstance(value, str):
            return 1, None

        text = value.replace(",", ".").lower()

        # case: nxp kW, e.g. "3x11kW", "2x22kW / 50Amp"
        match = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*kw", text)
        if match:
            n, p = match.groups()
            return int(n), float(p)

        # case: single kW value, e.g. "11kW", "22kW / 10 A"
        match = re.search(r"(\d+(?:\.\d+)?)\s*kw", text)
        if match:
            return 1, float(match.group(1))

        # fallback
        return 1, None

    # ------------------------------------------------------------------
    # Spatial index
    # ------------------------------------------------------------------
    con_buses_gdf = gpd.GeoDataFrame(
        con_buses,
        geometry=con_buses["geom"],
        crs="EPSG:32632",
    ).reset_index(names="bus_name")

    tree = STRtree(con_buses_gdf.geometry.values)

    bus_load_counter = defaultdict(int)
    buses_with_charging = set() #export related

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for _, ev in EV_locations.iterrows():
        
        nearest_idx = tree.nearest(ev.geometry)
        bus_geom = con_buses_gdf.loc[nearest_idx, "geometry"]
        bus_name = con_buses_gdf.loc[nearest_idx, "bus_name"]

        dist = ev.geometry.distance(bus_geom)
        
        if dist > MAX_ASSIGN_DIST:
            print(
                f"SKIPPED EV: no bus within {MAX_ASSIGN_DIST:.0f} m "
                f"(nearest = {dist:.1f} m)"
            )
            continue

        buses_with_charging.add(bus_name) #export related

        n_points, kw_per_point = parse_bemerkung(ev.get("BEMERKUNG"))

        if kw_per_point is None:
            kw_per_point = 11.0  # default fallback

        for _ in range(n_points):

            bus_load_counter[bus_name] += 1
            counter = bus_load_counter[bus_name]

            p_set = kw_per_point / 1000.0  # kW to MW
            load_name = f"EV_load_{bus_name}_{counter}"

            n.add(
                "Load",
                load_name,
                bus=bus_name,
                p_set=p_set,
                carrier="charging_point",
            )
    '''
    export_buses_with_charging( #export related
        con_buses,
        buses_with_charging,
    )
    '''
    return n


def import_EV_demands(
    n,
    *,
    master_seed=42,
    export_profiles=True,
    export_path="results",
):
    """
    Updates load of charging points with dummy timeseries.
    This would be replaced as soon as there are timeseries from SimBEV and TracBEV.

    Optional: Saves ev_profles_df as CSV in results folder.
    """

    ev_loads = n.loads[n.loads["carrier"] == "charging_point"]

    # Creates dummy EV loads
    def ev_profile_stochastic(
        snapshots,
        charger_mw=0,
        mean_arrival_hour=18,
        std_arrival=2,
        mean_energy_mwh=0.015,
        std_energy_mwh=0.003,
        max_hours=6,
        efficiency=1.0,
        rng_local=None,
    ):
        if rng_local is None:
            rng_local = np.random.default_rng()

        # load profile
        ev_series = pd.Series(0.0, index=snapshots, dtype=float)

        per_day = pd.Index(ev_series.index.normalize().unique())

        for day in per_day:
            h = int(
                np.clip(
                    rng_local.normal(mean_arrival_hour, std_arrival), 0, 23
                )
            )  # random time of arrival
            e_need_mwh = max(
                0.0, rng_local.normal(mean_energy_mwh, std_energy_mwh)
            )  # needed energy

            charger_mw_safe = max(charger_mw, 1e-9)  # avoid division by zero
            hours_needed = int(
                np.ceil(e_need_mwh / (charger_mw_safe * efficiency))
            )
            hours_charge = min(hours_needed, max_hours)

            # creating load profile
            for k in range(hours_charge):
                ts_charge = day + pd.Timedelta(hours=h + k)
                if ts_charge in ev_series.index:
                    ev_series.loc[ts_charge] = charger_mw

        return ev_series

    # DataFrame for profiles
    ev_profiles = {}

    # For each EV: own load profile + bus
    for i, row in enumerate(ev_loads.itertuples(index=True)):
        load_name = row.Index

        # Aktueller statischer p_set (float)
        try:
            static_p_set = float(row.p_set)
        except Exception:
            # falls p_set schon eine Series ist, nimm den Max als charger_mw
            if isinstance(row.p_set, pd.Series):
                static_p_set = float(row.p_set.max())
            else:
                static_p_set = 0.011

        rng_ev = np.random.default_rng(master_seed + 1000 + i)  # RNG per EV

        # generating ev profile
        ev_profile = ev_profile_stochastic(
            n.snapshots, charger_mw=static_p_set, rng_local=rng_ev
        )

        # Saving profil in DataFrame
        ev_profiles[load_name] = ev_profile
    
    ev_profiles_df = pd.DataFrame(ev_profiles, index=n.snapshots)

    # ensure columns exist and correct order
    missing_cols = ev_profiles_df.columns.difference(n.loads_t.p_set.columns)

    if not missing_cols.empty:
        n.loads_t.p_set = pd.concat(
            [n.loads_t.p_set, ev_profiles_df[missing_cols]],
            axis=1,
        )
    else:
        n.loads_t.p_set.loc[:, ev_profiles_df.columns] = ev_profiles_df
    
    '''
    ############### OPTIONAL EXPORT TO CHECK RESULTS ###############
    # Exportiere ev_profiles_df, falls gewünscht
    if export_profiles:
        os.makedirs(export_path, exist_ok=True)
        csv_path = os.path.join(export_path, 'ev_profiles.csv')

        try:
            ev_profiles_df.to_csv(csv_path, index=True)
            print(f"ev_profiles_df als CSV gespeichert: {csv_path}")
        except Exception as e:
            print(f"Fehler beim Speichern als CSV: {e}")
    '''
    
    return n

'''
####### For checking if Ev loads are distributet as expected ########
def export_buses_with_charging(con_buses, buses_with_charging):
    buses_ev = con_buses.loc[
        con_buses.index.isin(buses_with_charging)
    ].copy()

    # Convert to GeoDataFrame explicitly
    buses_ev = gpd.GeoDataFrame(
        buses_ev,
        geometry=buses_ev["geom"],
        crs="EPSG:32632",
    )

    os.makedirs("results", exist_ok=True)
    buses_ev.to_file("results/buses_with_charging_points.shp")
'''