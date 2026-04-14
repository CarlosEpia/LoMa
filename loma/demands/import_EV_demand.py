#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from collections import defaultdict

import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree


def import_charging_points(
    n,
    input_folder,
    *,
    export_bus_shapefile=False,
    export_debug_csv=False,
    export_charging_profiles=False,
    export_path="results",
    debug_csv_name="debug_charging_parser.csv",
    charging_profile_csv_name="charging_point_profiles.csv",
    charging_profile_unit="kW",   # "kW" or "MW"
    prefer_44_as="4x11",          # "4x11" or "2x22"
):
    """
    Reads charging point positions from the shape file and adds charging loads.

    Parsing priority:
      1) BEMERKUNG
      2) NENNLEISTU
      3) Amp-only heuristic
      4) Fallback = 11 kW

    Notes:
      - If export_charging_profiles=True:
          * exports flat profiles from static n.loads.p_set
      - Connection logic:
          * <= 100 kW -> nearest LV house_connection bus
          * > 100 kW  -> nearest MV trafo bus
    """

    EV_locations = gpd.read_file(
        os.path.join(input_folder, "Gis ST Ladesäule Position.shp")
    ).to_crs("EPSG:32632")

    # get house_connection buses
    house_buses = n.buses[n.buses.comp_type == "house_connection"].copy()
    
    # get MV side (bus0) of the trafo buses
    mv_busses = n.transformers["bus0"].unique()
    trafo_buses = n.buses.loc[n.buses.index.intersection(mv_busses)].copy()

    MAX_ASSIGN_DIST_LV = 100.0   # m
    MAX_ASSIGN_DIST_MV = 300.0   # m
    MV_THRESHOLD_KW = 100.0      # kW

    def _normalize_text(s: str) -> str:
        s = s.strip().lower()
        s = s.replace(",", ".").replace("×", "x").replace("*", "x")
        s = re.sub(r"\s+", " ", s)
        return s

    def _clamp_kw(p: float) -> float:
        return 11.0 if p < 3.0 else p

    def _dedupe_same_value(loads: list, tol: float = 1e-9) -> list:
        if len(loads) <= 1:
            return loads

        if all(abs(loads[i] - loads[0]) <= tol for i in range(1, len(loads))):
            return [loads[0]]

        return loads

    def parse_loads_from_text(value: object) -> list:
        """
        Parses per-load powers in kW from a free text field.
        Returns [] if nothing meaningful could be extracted.
        """
        if not isinstance(value, str):
            return []

        t = _normalize_text(value)
        if not t:
            return []

        # Example: "2x11 63A" -> 2x11kW
        m = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\b(?!\s*k(?:w|va))", t)
        if m and re.search(r"\b\d+(?:\.\d+)?\s*a(?:mp(?:ere)?)?\b", t):
            n_mult = max(int(m.group(1)), 1)
            p = _clamp_kw(float(m.group(2)))
            return [p] * n_mult

        # Example: "3x11kW"
        m = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*k(?:w|va)\b", t)
        if m:
            n_mult = max(int(m.group(1)), 1)
            p = _clamp_kw(float(m.group(2)))
            return [p] * n_mult

        # Example: "22kW 11kW", "22kW 22kVA"
        vals = re.findall(r"(\d+(?:\.\d+)?)\s*k(?:w|va)\b", t)
        if vals:
            loads = [_clamp_kw(float(v)) for v in vals]

            if len(loads) == 1 and abs(loads[0] - 44.0) < 0.51:
                if prefer_44_as == "2x22":
                    return [22.0, 22.0]
                return [11.0, 11.0, 11.0, 11.0]

            loads = _dedupe_same_value(loads, tol=1e-6)
            return loads

        return []

    def parse_from_nennleistung(value: object) -> list:
        """
        Parses NENNLEISTU if BEMERKUNG has no usable result.
        """
        if isinstance(value, (int, float)) and value is not None:
            return [_clamp_kw(float(value))]

        if isinstance(value, str):
            t = _normalize_text(value)
            if not t:
                return []

            m = re.search(r"(\d+(?:\.\d+)?)\s*k(?:w|va)\b", t)
            if m:
                return [_clamp_kw(float(m.group(1)))]

            m = re.search(r"(\d+(?:\.\d+)?)", t)
            if m:
                return [_clamp_kw(float(m.group(1)))]

        return []

    def parse_amps_only(value: object) -> list:
        """
        Amp-only heuristic:
          - >= 240A -> 150kW
          - <= 50A  -> 11kW
          - > 50A   -> 22kW
        """
        if not isinstance(value, str):
            return []

        t = _normalize_text(value)
        if not t:
            return []

        amps = re.findall(r"(\d+(?:\.\d+)?)\s*a(?:mp(?:ere)?)?\b", t)
        if not amps:
            return []

        a = max(float(x) for x in amps)

        if a >= 240.0:
            return [150.0]
        if a <= 50.0:
            return [11.0]
        return [22.0]

    def parse_charger_loads(ev_row) -> tuple[list, str]:
        """
        Returns:
            (loads_kw, source)

        source in:
            {"BEMERKUNG", "NENNLEISTU", "AMPS_BEM", "AMPS_NENN", "FALLBACK"}
        """
        loads = parse_loads_from_text(ev_row.get("BEMERKUNG"))
        if loads:
            return loads, "BEMERKUNG"

        loads = parse_from_nennleistung(ev_row.get("NENNLEISTU"))
        if loads:
            return loads, "NENNLEISTU"

        loads = parse_amps_only(ev_row.get("BEMERKUNG"))
        if loads:
            return loads, "AMPS_BEM"

        loads = parse_amps_only(ev_row.get("NENNLEISTU"))
        if loads:
            return loads, "AMPS_NENN"

        return [11.0], "FALLBACK"

    def build_gdf_and_tree(bus_df):
        gdf = gpd.GeoDataFrame(
            bus_df,
            geometry=bus_df["geom"],
            crs="EPSG:32632",
        ).reset_index(names="bus_name")

        tree = STRtree(gdf.geometry.values)
        return gdf, tree

    def export_charging_point_profiles(
        n,
        load_names,
        export_path,
        csv_name,
        unit="kW",
    ):
        """
        Exports time series of the created charging loads.
        Uses n.loads_t.p_set if present, otherwise static p_set as flat profile.
        """
        if not load_names:
            print("No charging point loads available for profile export.")
            return

        load_names = [ln for ln in load_names if ln in n.loads.index]
        if not load_names:
            print("No valid charging point loads found in network.")
            return

        if len(n.snapshots) == 0:
            print("No snapshots available. Skipping charging profile export.")
            return

        os.makedirs(export_path, exist_ok=True)

        static_vals = n.loads.loc[load_names, "p_set"].astype(float)
        profiles = pd.DataFrame(
            index=n.snapshots,
            data={ln: static_vals.loc[ln] for ln in load_names},
        )

        try:
            ts = getattr(getattr(n, "loads_t", None), "p_set", None)
            if ts is not None and not ts.empty:
                common_cols = [c for c in load_names if c in ts.columns]
                if common_cols:
                    profiles.loc[:, common_cols] = ts.loc[:, common_cols]
        except Exception as e:
            print(f"Warning: could not read dynamic charging profiles: {e}")

        unit_upper = unit.upper()
        if unit_upper == "KW":
            profiles = profiles * 1000.0
        elif unit_upper != "MW":
            raise ValueError("charging_profile_unit must be 'kW' or 'MW'")

        profiles.index.name = "snapshot"

        out_path = os.path.join(export_path, csv_name)
        profiles.to_csv(out_path)
        print(f"Charging point profiles exported: {out_path}")

    house_buses_gdf, house_tree = build_gdf_and_tree(house_buses)
    trafo_buses_gdf, trafo_tree = build_gdf_and_tree(trafo_buses)

    bus_load_counter = defaultdict(int)
    buses_with_charging = set()
    charging_load_names = set()
    debug_rows = []

    for idx, ev in EV_locations.iterrows():
        loads_kw, src = parse_charger_loads(ev)
        kw_ref = max(loads_kw) if loads_kw else 11.0

        if kw_ref >= MV_THRESHOLD_KW:
            target_gdf = trafo_buses_gdf
            target_tree = trafo_tree
            max_dist = MAX_ASSIGN_DIST_MV
            level = "MV"
        else:
            target_gdf = house_buses_gdf
            target_tree = house_tree
            max_dist = MAX_ASSIGN_DIST_LV
            level = "LV"

        nearest_idx = target_tree.nearest(ev.geometry)
        bus_geom = target_gdf.loc[nearest_idx, "geometry"]
        bus_name = target_gdf.loc[nearest_idx, "bus_name"]
        dist = ev.geometry.distance(bus_geom)

        if dist > max_dist:
            print(
                f"SKIPPED EV ({level}): no bus within {max_dist:.0f} m "
                f"(nearest = {dist:.1f} m)"
            )

            if export_debug_csv:
                debug_rows.append({
                    "feature_idx": idx,
                    "status": "SKIPPED_DISTANCE",
                    "source": src,
                    "level": level,
                    "bus_name": None,
                    "dist_m": float(dist),
                    "max_dist_m": float(max_dist),
                    "BEMERKUNG": ev.get("BEMERKUNG"),
                    "NENNLEISTU": ev.get("NENNLEISTU"),
                    "loads_kw": "|".join(str(x) for x in loads_kw),
                    "p_set_mw_each": "|".join(str(x / 1000.0) for x in loads_kw),
                    "n_loads": len(loads_kw),
                })
            continue

        buses_with_charging.add(bus_name)

        created_load_names = []
        for kw in loads_kw:
            bus_load_counter[bus_name] += 1
            counter = bus_load_counter[bus_name]

            p_set = kw / 1000.0
            load_name = f"Existing_Charging_Point_{bus_name}_{counter}"

            n.add(
                "Load",
                load_name,
                bus=bus_name,
                p_set=p_set,
                carrier="charging_point",
            )

            created_load_names.append(load_name)
            charging_load_names.add(load_name)

        if export_debug_csv:
            debug_rows.append({
                "feature_idx": idx,
                "status": "CREATED",
                "source": src,
                "level": level,
                "bus_name": bus_name,
                "dist_m": float(dist),
                "max_dist_m": float(max_dist),
                "BEMERKUNG": ev.get("BEMERKUNG"),
                "NENNLEISTU": ev.get("NENNLEISTU"),
                "loads_kw": "|".join(str(x) for x in loads_kw),
                "p_set_mw_each": "|".join(str(x / 1000.0) for x in loads_kw),
                "n_loads": len(loads_kw),
                "created_load_names": "|".join(created_load_names),
            })

    if export_bus_shapefile:
        export_buses_with_charging(
            house_buses=house_buses,
            trafo_buses=trafo_buses,
            buses_with_charging=buses_with_charging,
            export_path=export_path,
        )

    if export_debug_csv:
        os.makedirs(export_path, exist_ok=True)
        dbg_df = pd.DataFrame(debug_rows)

        preferred_cols = [
            "feature_idx", "status", "source", "level", "bus_name",
            "dist_m", "max_dist_m", "BEMERKUNG", "NENNLEISTU",
            "loads_kw", "p_set_mw_each", "n_loads", "created_load_names",
        ]
        cols = [c for c in preferred_cols if c in dbg_df.columns] + [
            c for c in dbg_df.columns if c not in preferred_cols
        ]
        dbg_df = dbg_df[cols]

        dbg_path = os.path.join(export_path, debug_csv_name)
        try:
            dbg_df.to_csv(dbg_path, index=False)
            print(f"Debug charging parser export saved: {dbg_path}")
        except Exception as e:
            print(f"Error saving debug charging parser export: {e}")

    if export_charging_profiles:
        try:
            export_charging_point_profiles(
                n=n,
                load_names=sorted(charging_load_names),
                export_path=export_path,
                csv_name=charging_profile_csv_name,
                unit=charging_profile_unit,
            )
        except Exception as e:
            print(f"Error saving charging point profiles: {e}")

    return n


def export_buses_with_charging(
    house_buses,
    trafo_buses,
    buses_with_charging,
    export_path="results",
    filename="buses_with_charging_points.shp",
):
    """
    Exports all buses (house + trafo) that received EV charging loads.
    """
    all_buses = pd.concat([house_buses, trafo_buses])

    buses_ev = all_buses.loc[all_buses.index.isin(buses_with_charging)].copy()
    if buses_ev.empty:
        print("No buses with charging points to export.")
        return

    buses_ev["geometry"] = buses_ev["geom"].apply(
        lambda g: g if g.geom_type == "Point" else g.centroid
    )

    buses_ev = gpd.GeoDataFrame(
        buses_ev,
        geometry="geometry",
        crs="EPSG:32632",
    )

    os.makedirs(export_path, exist_ok=True)
    out_path = os.path.join(export_path, filename)
    buses_ev.to_file(out_path)
    print(f"Buses with charging points exported: {out_path}")