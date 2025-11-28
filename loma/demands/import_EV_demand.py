#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree

#input_folder = 'data/Input_files/Filtered_data_Kronenburg_V3'

def import_EV_loads(n, input_folder):
    """
    Reads EV-Positions from the shape file and adds a static load 
    where p_set is the kW amount of the charging station extracted from the 'BEMERKUNG' column.
    """
    
    EV_locations = gpd.read_file(os.path.join(input_folder, "Gis ST Ladesäule Position.shp"))
    EV_locations = EV_locations.to_crs('EPSG:32632')

    con_buses = n.buses[n.buses.comp_type == 'house_connection'].copy()

    # extracting kW from 'BEMERKUNG' column
    def extract_kw(value: str) -> float | None:
        if not isinstance(value, str):
            return None
        text = value.replace(",", ".").lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*kw", text)
        return float(match.group(1)) if match else None

    EV_locations["kw"] = EV_locations.get("BEMERKUNG", pd.Series(index=EV_locations.index)).apply(extract_kw)

    # spatial queries with STRtree
    geoms = con_buses.geom.values
    tree = STRtree(geoms)
    index_to_bus = dict(enumerate(con_buses.index.values))

    # Counter in case multiple EV-chargers for one bus
    bus_load_counter = defaultdict(int)

    for idx, ev in EV_locations.iterrows():
        nearest_geom_idx = tree.nearest(ev.geometry)
        bus_name = index_to_bus[nearest_geom_idx]

        # in case two EV-chargers are connected to the same bus
        bus_load_counter[bus_name] += 1
        counter = bus_load_counter[bus_name]

        kW = ev["kw"] if not pd.isna(ev["kw"]) else 11.0  # Default 11 kW if unknown
        p_set = kW / 1000.0  # MW

        load_name = f"EV_load_{bus_name}_{counter}"

        n.add("Load",
              load_name,
              bus=bus_name,
              p_set=p_set,
              carrier="land_transport_EV",
              )

    return n


def import_EV_demands(n, *, factor=1.0, e_nom_par14_store=1.0, master_seed=42,
                      charge_efficiency=0.98, export_profiles=True, export_path='results'):
    """
    Updates EV loads with timeseries and implements buses, links and stores necessary for the flexibilities.
    Creates dummy EV Load (this would be replaced as soon as there are timeseries from SimBEV and TracBEV)
    
    Optional: Saves ev_profles_df as CSV in results folder.
    """

    ev_loads = n.loads[n.loads['carrier'] == 'land_transport_EV']

    # Creates dummy EV loads and connection parameter
    def ev_profile_stochastic_with_connection(
            snapshots,
            charger_mw=0,
            mean_arrival_hour=18,
            std_arrival=2,
            mean_energy_mwh=0.015 * factor,
            std_energy_mwh=0.003 * factor,
            max_hours=6,
            efficiency=1.0,
            connection_hours=8, # EV stays connected for 8 hours after arrival
            rng_local=None
    ):
        if rng_local is None:
            rng_local = np.random.default_rng()

        # load profile
        ev_series = pd.Series(0.0, index=snapshots, dtype=float)
        # binary variable: EV connected (1) / not connected (0)
        connected = pd.Series(0, index=snapshots, dtype=int)
        
        per_day = pd.Index(ev_series.index.normalize().unique())

        for day in per_day:
            h = int(np.clip(rng_local.normal(mean_arrival_hour, std_arrival), 0, 23))   # random time of arrival
            e_need_mwh = max(0.0, rng_local.normal(mean_energy_mwh, std_energy_mwh))    # needed energy

            charger_mw_safe = max(charger_mw, 1e-9) # avoid division by zero
            hours_needed = int(np.ceil(e_need_mwh / (charger_mw_safe * efficiency)))
            hours_charge = min(hours_needed, max_hours)

            # creating load profile
            for k in range(hours_charge):
                ts_charge = day + pd.Timedelta(hours=h + k)
                if ts_charge in ev_series.index:
                    ev_series.loc[ts_charge] = charger_mw
            
            # marking the EV as connected for connection_hours after arrival
            for k in range(connection_hours):
                ts_connect = day + pd.Timedelta(hours=h + k)
                if ts_connect in connected.index:
                    connected.loc[ts_connect] = 1

        return ev_series, connected

    # DataFrame for profiles
    ev_profiles_df = pd.DataFrame(index=n.snapshots)

    # For each EV: own load profile + bus + links + store
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

        rng_ev = np.random.default_rng(master_seed + 1000 + i) # RNG per EV

        # generating ev profile and connection parameter
        ev_profile, ev_connection = ev_profile_stochastic_with_connection(
            n.snapshots, charger_mw=static_p_set, rng_local=rng_ev
        )

        # Saving profil in DataFrame
        ev_profiles_df[load_name] = ev_profile

        # Replace static p_set with timeseries
        if load_name not in n.loads_t.p_set.columns:
            n.loads_t.p_set[load_name] = 0.0
        n.loads_t.p_set[load_name] = ev_profile
    

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

    # Kurzer Check: Ausgabe der ersten 10 Loads mit Profilinfo
    print('\n=== Kurzer EV-Load Check ===')
    for load_name, row in n.loads[n.loads['carrier'] == 'land_transport_EV'].head(10).iterrows():
        p = row['p_set']
        if isinstance(p, pd.Series):
            print(f"{load_name}: Zeitprofil mit {len(p)} Zeitpunkten, peak={p.max():.4f} MW at {p.idxmax()}")
        else:
            print(f"{load_name}: statischer p_set = {p}")
    ################################################################


    return n
