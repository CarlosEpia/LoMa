#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 08:34:28 2025

@author: student
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import os
from collections import defaultdict
from shapely.strtree import STRtree


#input_folder = 'data/Input_files/Filtered_data_Kronenburg_V3'

def import_EV_loads(n, input_folder):
    EV_locations = gpd.read_file(os.path.join(input_folder, "Gis ST Ladesäule Position.shp"))
    EV_locations = EV_locations.to_crs('EPSG:32632')
    con_buses = n.buses[n.buses.comp_type == 'house_connection'].copy()
    
    # spartial queries with STRtree
    bus_ids = con_buses.index.values
    tree = STRtree(con_buses.geom)

    # Counter in case multiple EV-chargers for one bus
    bus_load_counter = defaultdict(int)
   
    for ev_geom in EV_locations.geometry:
        nearest_geom = tree.nearest(ev_geom)
        bus_name = bus_ids[nearest_geom]

        # in case two EV-chargers are connected to the same bus 
        bus_load_counter[bus_name] += 1
        counter = bus_load_counter[bus_name]
        load_name = f"EV_load_{bus_name}_{counter}"
        n.add("Load",
              load_name,
              bus=bus_name,
              p_set=0,       
              carrier="land_transport_EV")
        
    return n


def import_EV_demands(n):
    """
    Implements EV loads, buses, links and stores necessary for the flexibilities.
    Creates dummy EV Load (this would be replaced as soon as there are timeseries from SimBEV and TracBEV)
    """

    # -----------------------
    # Assumptions
    # -----------------------
    ev_loads = n.loads[n.loads['carrier'] == 'land_transport_EV']   # CSV-Loads mit EV-Carrier
    factor = 1.0                                                # nur um schneller EV load zu erhöhen und wieder normal zu machen um zu sehen wann Stores genutzt werden
    bev_charger_rate = 0.011*factor                             # 11 kW pro EV
    e_nom_par14_store = 1.0                                     # Speichergröße (MWh)
    master_seed = 42                                            # RNG-Seed
    charge_efficiency = 0.98

    # Dummy EV Loads and connection parameter
    def ev_profile_stochastic_with_connection(
            snapshots,
            charger_mw=bev_charger_rate,
            mean_arrival_hour=18,
            std_arrival=2,
            mean_energy_mwh=0.015*factor,
            std_energy_mwh=0.003*factor,
            max_hours=6,
            efficiency=1.0,
            connection_hours=8,  # EV stays connected for 8 hours after arrival
            rng_local=None
            ):
        if rng_local is None:
            rng_local = np.random.default_rng()
            
        # Ladeprofil
        s = pd.Series(0.0, index=snapshots, dtype=float)
        # binary variable: EV connected (1) / not connected (0)
        connected = pd.Series(0, index=snapshots, dtype=int)
        
        per_day = pd.Index(s.index.normalize().unique())
                
        for day in per_day:
            # Zufällige Ankunftsstunde
            h = int(np.clip(rng_local.normal(mean_arrival_hour, std_arrival), 0, 23))
            # Benötigte Energie
            e_need_mwh = max(0.0, rng_local.normal(mean_energy_mwh, std_energy_mwh))
            hours_needed = int(np.ceil(e_need_mwh / (charger_mw * efficiency)))
            hours_charge = min(hours_needed, max_hours)
                
            # Ladeprofil erstellen
            for k in range(hours_charge):
                ts_charge = day + pd.Timedelta(hours=h + k)
                if ts_charge in s.index:
                    s.loc[ts_charge] = charger_mw
                            
            # Marking the EV as connected for connection_hours after arrival
            for k in range(connection_hours):
                ts_connect = day + pd.Timedelta(hours=h + k)
                if ts_connect in connected.index:
                    connected.loc[ts_connect] = 1
                                    
        return s, connected

    # --- DataFrame für Profile vorbereiten ---
    ev_profiles_df = pd.DataFrame(index=n.snapshots)

    # --- For each EV: own Profile + Bus + Load + Links + Store ---
    for i, row in enumerate(ev_loads.itertuples(index=True)):
        
        # RNG pro EV
        rng_ev = np.random.default_rng(master_seed + 1000 + i)
        
        # Ladeprofil erzeugen
        ev_profile, ev_connection = ev_profile_stochastic_with_connection(n.snapshots,
                                           charger_mw=bev_charger_rate,
                                           rng_local=rng_ev)

        # Profil in DataFrame speichern
        ev_profiles_df[row.Index] = ev_profile

        # Coordinates of the network bus
        x = n.buses.at[row.bus, "x"]
        y = n.buses.at[row.bus, "y"]
        
        # --- EV-Bus hinzufügen ---
        ev_bus_name = f"EV_{row.bus}_{i}"
        n.add("Bus", 
              ev_bus_name,
              x=x,
              y=y)
        
        # --- EV-Load hinzufügen ---
        n.add("Load",
              name=f"EV_Load_{ev_bus_name}",
              bus=ev_bus_name,
              carrier=row.carrier,
              p_set=ev_profile)

        # --- Link: Network-Bus -> EV-Bus ---
        n.add("Link",
              f"Link_{ev_bus_name}_{i}",
              bus0=row.bus,
              bus1=ev_bus_name,
              p_nom=ev_profile.max()/charge_efficiency,
              p_nom_extendable=False,
              efficiency=charge_efficiency,
              p_max_pu=ev_connection,
              )

        # --- Store-Bus hinzufügen ---
        par14_store_bus = f"Par14_Store_{row.bus}_{i}"
        n.add("Bus", 
              par14_store_bus,
              x=x,
              y=y)

        # --- Store hinzufügen ---
        n.add("Store",
              f"Par14_Store_at_{row.bus}_{i}",
              bus=par14_store_bus,
              e_nom=e_nom_par14_store,
              e_initial=e_nom_par14_store)  
        
        # --- Link: Store-Bus -> EV-Bus ---
        n.add("Link",
              f"Par14_StoreDischarge_Link_{row.bus}_{i}",
              bus0=par14_store_bus,
              bus1=ev_bus_name,
              p_nom=bev_charger_rate,
              p_max_pu=0.6,             
              efficiency = np.sqrt(charge_efficiency),
              )          
        
        # --- Link: EV-Bus -> Store-Bus ---
        n.add("Link",
              f"EV_to_StoreCharge_Link_{row.bus}_{i}",
              bus0=ev_bus_name,
              bus1=par14_store_bus,
              p_nom=bev_charger_rate,
              p_max_pu=ev_connection,             # ev_connection = 1 if EV is currently plugged in
              efficiency = np.sqrt(charge_efficiency),
              )    

    return n