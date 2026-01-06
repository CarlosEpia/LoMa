#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 09:49:16 2025

@author: student
"""

import os
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from shapely.ops import linemerge, unary_union
import networkx as nx

from loma.demands.import_hp_demand import check_heat_pumps
from loma.demands.household_count import count_households_per_bus_input_file
from loma.demands.household_count import count_households_per_bus_census_data


def create_gdf_from_shape(input_folder):
    """
    Creates GeoDataFrame for pypsa network components based on shapefiles

    Returns
    -------
    buses : GeoDataFrame
        contains all potential bus-components
    lines : GeoDataFrame
        contains all potential line-components

    """

    def safe_read(path):
        """Safely read a shapefile; return empty GeoDataFrame if not found."""
        if os.path.exists(path):
            return gpd.read_file(path)
        else:
            print(f"⚠️  File not found: {os.path.basename(path)} — skipping.")
            return gpd.GeoDataFrame()

    # --- try reading all files (some might not exist) ---
    LV_lines = safe_read(
        os.path.join(input_folder, "Gis NSP Kabelabschnitt Verlauf.shp")
    )
    MV_lines = safe_read(
        os.path.join(input_folder, "Gis MSP Kabelabschnitt Verlauf.shp")
    )
    HA_lines = safe_read(
        os.path.join(input_folder, "Gis NSP HA Abschnitt Verlauf.shp")
    )
    HA_Bus = safe_read(
        os.path.join(input_folder, "Gis NSP HA Kasten Position.shp")
    )
    distributors = safe_read(
        os.path.join(input_folder, "Gis ST Kabelverteiler Position.shp")
    )
    joints_LV = safe_read(
        os.path.join(input_folder, "Gis NSP Muffe Position.shp")
    )
    joints_MV = safe_read(
        os.path.join(input_folder, "Gis MSP Muffe Position.shp")
    )
    if not joints_MV.empty:
          joints_MV = joints_MV.to_crs(joints_LV.crs) ##necessary for concat joints
          joints_MV['comp_type']= 'MV_Muffe'  # to distinguish MV Muffen from LV_Muffen for line splitting method
    else: 
          joints_LV['comp_type'] = None
    joints = pd.concat([joints_LV, joints_MV], ignore_index=True)
    
    MVLV_trafos = safe_read(
        os.path.join(input_folder, "Gis ST Station Fläche.shp")
    )

    #delete lines which are "out of service"
    LV_lines = LV_lines[~(LV_lines.STATUS.isin(['außer Betrieb', 'Vorverlegung', 'stillgelegt']))]
    HA_lines = HA_lines[~(HA_lines.STATUS.isin(['außer Betrieb', 'Vorverlegung', 'stillgelegt']))]
    if not MV_lines.empty: 
          MV_lines = MV_lines[~(MV_lines.STATUS.isin(['außer Betrieb', 'stillgelegt', 'Vorverlegung']))]
    
    #delete ditributors with type 'Beleuchtung' (leads to wrong connnection in some cases)
    distributors = distributors[distributors.STATUS!='Beleuchtung']
    
    # rename columns to generalize the names
    for df in [HA_Bus, distributors, joints]:
        df.rename(
            columns={"LOKATION_S": "Straße", "HAUSNUMMER": "Hausnummer"},
            inplace=True,
        )
    for df in [MVLV_trafos]:
        df.rename(
            columns={
                "LOKATION_S": "Straße",
                "HAUSNUMMER": "Hausnummer",
                "TRAFOBELAS": "s_nom",
            },
            inplace=True,
        )
   
    # component-type-column for distinguish the components
    LV_lines["comp_type"] = "lv_line"
    if not MV_lines.empty:
        MV_lines["comp_type"] = "mv_line"
    HA_lines["comp_type"] = "hc_line"
    mask = joints["comp_type"].isna()
    joints.loc[mask, "comp_type"] = joints.loc[mask, "ART"]  ##ToDo: Generalize for usage in other regions than husum
    distributors["comp_type"] = "distributor"
    MVLV_trafos["comp_type"] = MVLV_trafos["ART"].apply(
          lambda x: "trafo_HV" if x == "Umspannwerk" else "trafo") #distuingish MV/HV_trafo
    HA_Bus["comp_type"] = "house_connection"

    ##buses
    bus_columns = ["comp_type", "Straße", "Hausnummer", "geometry"]
    trafo_columns = ["comp_type", "Straße", "Hausnummer", "s_nom", "geometry"]

    def ensure_columns(df, columns):
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan  # oder np.nan
        return df[columns]

    joints_clean = ensure_columns(joints, bus_columns)
    distributors_clean = ensure_columns(distributors, bus_columns)
    MVLV_trafos_clean = ensure_columns(MVLV_trafos, trafo_columns)
    HA_Bus_clean = ensure_columns(HA_Bus, bus_columns)

    # secure same crs
    target_crs = "EPSG:32632"
    for gdf in [
        LV_lines,
        MV_lines,
        HA_lines,
        joints_clean,
        distributors_clean,
        HA_Bus_clean,
        MVLV_trafos_clean,
    ]:
        if not gdf.empty:
            gdf.to_crs(target_crs, inplace=True)

    # combine all bus-datfarmes
    buses = pd.concat(
        [joints_clean, distributors_clean, MVLV_trafos_clean, HA_Bus_clean],
        ignore_index=True,
    )
    buses["bus_id"] = [f"bus_{i}" for i in range(len(buses))]
    buses = buses.reset_index(drop=True)

    ##lines
    line_columns = ["comp_type", "KABELTYP", "geometry"]
    # combine all line-dataframes
    lines_list = [df for df in [LV_lines, MV_lines, HA_lines] if not df.empty]
    lines = pd.concat(
        [df[line_columns] for df in lines_list], ignore_index=True
    )
    lines["line_id"] = [f"line_{i}" for i in range(len(lines))]
    lines = lines.reset_index(drop=True)

    return buses, lines

#not used currently
def merge_connected_mv_lines(lines, tolerance=0.001):
    """
    If MV-Lines are just splitted by "Muffe" the lines will be merged together and treated as one line
    """

    lines = lines.copy()
    lines_mv = lines[lines.comp_type == "mv_line"].reset_index(drop=True)
    merged_lines = []

    while len(lines_mv) > 0:
        # Nimm die erste Linie als Basis
        base_line = lines_mv.iloc[0]
        base_geom = base_line.geometry
        base_attrs = base_line.drop(labels="geometry").to_dict()
        lines_mv = lines_mv.drop(0).reset_index(drop=True)

        changed = True
        while changed:
            changed = False
            for idx, other_line in lines_mv.iterrows():
                other_geom = other_line.geometry

                # Check line end and beginning
                base_start = Point(base_geom.coords[0])
                base_end = Point(base_geom.coords[-1])
                other_start = Point(other_geom.coords[0])
                other_end = Point(other_geom.coords[-1])

                if (
                    base_start.distance(other_start) < tolerance
                    or base_start.distance(other_end) < tolerance
                    or base_end.distance(other_start) < tolerance
                    or base_end.distance(other_end) < tolerance
                ):

                    # Merge Linien korrekt orientiert
                    merged_geom = linemerge(
                        unary_union([base_geom, other_geom])
                    )
                    # Falls MultiLineString, längste Linie wählen
                    if merged_geom.geom_type == "MultiLineString":
                        merged_geom = max(
                            merged_geom.geoms, key=lambda x: x.length
                        )

                    base_geom = merged_geom
                    # Die verbundene Linie aus dem DataFrame entfernen
                    lines_mv = lines_mv.drop(idx).reset_index(drop=True)
                    changed = True
                    break  # for-Schleife neu starten

        # Fertige gemergte Linie speichern
        merged_lines.append({**base_attrs, "geometry": base_geom})

    if len(merged_lines) > 0:
        merged_gdf = gpd.GeoDataFrame(
            merged_lines, geometry="geometry", crs=lines.crs
        )
    else:
        merged_gdf = gpd.GeoDataFrame(
            columns=lines.columns, geometry="geometry", crs=lines.crs
        )
    # add adjusted mv_lines in original dataframe
    lines = lines[lines.comp_type != "mv_line"]
    lines = pd.concat([lines, merged_gdf], ignore_index=True)
    return lines


def merge_lines_splitted_by_bus(n, remove_buses=True, tolerance=0.001):
    """
    Iteratively merge lines in a PyPSA Network which are only separated by
    degree-2 buses without attached components. Modifies `n` in-place.
    """

    total_merged = 0

    def single_pass():
        merged_count = 0
        candidates = []

        # --- collect merge candidates ---
        for bus in list(n.buses.index):
            incident = n.lines[(n.lines.bus0 == bus) | (n.lines.bus1 == bus)]
            if len(incident) != 2:
                continue

            # skip buses with attached components
            attached = False
            for comp in ["Generator", "Load", "StorageUnit", "Link", "Transformer"]:
                df = n.df(comp)
                if df is None or df.empty:
                    continue
                if comp in ["Transformer", "Link"]:
                    if ((df.get("bus0") == bus) | (df.get("bus1") == bus)).any():
                        attached = True
                        break
                else:
                    if (df.get("bus") == bus).any():
                        attached = True
                        break

            if attached:
                continue

            l1, l2 = incident.index.tolist()
            geom1 = incident.loc[l1, "geom"]
            geom2 = incident.loc[l2, "geom"]
            bus_geom = n.buses.loc[bus, "geom"]

            s1, e1 = Point(geom1.coords[0]), Point(geom1.coords[-1])
            s2, e2 = Point(geom2.coords[0]), Point(geom2.coords[-1])

            p1 = s1 if s1.distance(bus_geom) <= tolerance else e1
            p2 = s2 if s2.distance(bus_geom) <= tolerance else e2

            if p1.distance(p2) <= tolerance:
                candidates.append((bus, l1, l2))

        # --- process candidates ---
        for bus, l1, l2 in candidates:
            if bus not in n.buses.index:
                continue
            if l1 not in n.lines.index or l2 not in n.lines.index:
                continue

            def other_bus(line, mid):
                row = n.lines.loc[line]
                return row.bus1 if row.bus0 == mid else row.bus0

            bus_a = other_bus(l1, bus)
            bus_c = other_bus(l2, bus)
            if bus_a == bus_c:
                continue

            geom1 = n.lines.at[l1, "geom"]
            geom2 = n.lines.at[l2, "geom"]

            merged_geom = linemerge(unary_union([geom1, geom2]))
            if merged_geom.geom_type == "MultiLineString":
                merged_geom = max(merged_geom.geoms, key=lambda g: g.length)

            # aggregate attributes
            r_new = n.lines.at[l1, "r"] + n.lines.at[l2, "r"]
            x_new = n.lines.at[l1, "x"] + n.lines.at[l2, "x"]
            length_new = (
                n.lines.at[l1, "length"] + n.lines.at[l2, "length"]
            )

            s1 = n.lines.at[l1, "s_nom"]
            s2 = n.lines.at[l2, "s_nom"]
            s_nom_new = min(s1, s2)
            comp_type_new = n.lines.at[l1, "comp_type"]
            cable_type_new = n.lines.at[l1, "cable_type"]
            capital_new = 0.0
            if "capital_cost" in n.lines.columns:
                capital_new = (
                    float(n.lines.at[l1, "capital_cost"] or 0)
                    + float(n.lines.at[l2, "capital_cost"] or 0)
                )

            base_name = f"{l1}_{l2}_merged"
            new_name = base_name
            i = 0
            while new_name in n.lines.index:
                i += 1
                new_name = f"{base_name}_{i}"

            n.remove("Line", [l1, l2])

            n.add(
                "Line",
                new_name,
                bus0=bus_a,
                bus1=bus_c,
                carrier="AC",
                r=r_new,
                x=x_new,
                s_nom=s_nom_new,
                s_nom_extendable = True,
                s_nom_min=s_nom_new,
                capital_cost=capital_new,
                length=length_new,
            )

            n.lines.at[new_name, "geom"] = merged_geom
            n.lines.at[new_name, "comp_type"] = comp_type_new
            n.lines.at[new_name, "cable_type"] = cable_type_new

            if remove_buses:
                try:
                    n.remove("Bus", [bus])
                except Exception:
                    pass

            merged_count += 1

        return merged_count

    # --- iterative merging ---
    while True:
        merged = single_pass()
        if merged == 0:
            break
        total_merged += merged

    print(f"angepasste lines (gesamt): {total_merged}")


###LV-lines
def cut_line_at_points(line, cutting_points):
    """
    Split line at each given point (points have to be on the line)
    """
    # Calculate position along the line
    distances = [line.project(point) for point in cutting_points]
    distances = sorted(set(distances))  # sort points/delete duplicated points

    # Add starting and endpoint to the split-line
    if 0.0 not in distances:
        distances = [0.0] + distances
    if line.length not in distances:
        distances.append(line.length)

    # interpolate points
    points = [line.interpolate(distance) for distance in distances]

    # create segments of each line
    segments = []
    for i in range(len(points) - 1):
        segment = cut_line_between_distances(
            line, distances[i], distances[i + 1]
        )
        if segment is not None:
            segments.append(segment)

    return segments


def cut_line_between_distances(
    line, start_distance, end_distance, tolerance=0.01
):
    """
    Extracts geometry of one segment of splitted line from given end and start-distance
    """
    if end_distance - start_distance < tolerance:
        return None

    coords = [line.interpolate(start_distance).coords[0]]

    for point in line.coords:
        point_distance = line.project(Point(point))
        if start_distance < point_distance < end_distance:
            coords.append(point)

    coords.append(line.interpolate(end_distance).coords[0])

    return LineString(coords)


def split_lines_on_joints(lines, buses, tolerance=0.1):
    """
    Splits LV-, HC-lines at joint buses (snapped to the line) and returns a new GeoDataFrame
    with all lines (LV, HC, MV).
    """

    lines_to_split = lines[
        lines.comp_type.isin(["lv_line", "hc_line"])
    ].copy()

    other_lines = lines[
        ~lines.comp_type.isin(["lv_line", "hc_line"])
    ].copy()

    joint_types = [
        "Hausanschlußmuffe",
        "Verbindungsmuffe",
        "Endmuffe",
        "Übergangsmuffe",
        "Reparaturmuffe",
        "Abzweigmuffe",
        "vorverlegtes Ende",
        "HA-Kombimuffe",
        "distributor",
    ]

    joint_buses = buses[buses.comp_type.isin(joint_types)].copy()
    joint_sindex = joint_buses.sindex

    split_lines = []

    for _, row in lines_to_split.iterrows():
        line_geom = row.geometry

        # --- spatial pre-filter (bounding box) ---
        bbox = line_geom.buffer(tolerance).bounds
        candidate_idx = list(joint_sindex.intersection(bbox))
        candidates = joint_buses.iloc[candidate_idx]

        if candidates.empty:
            split_lines.append(
                {
                    "geometry": line_geom,
                    "comp_type": row.comp_type,
                    "KABELTYP": row.KABELTYP,
                }
            )
            continue

        # --- exact distance check ---
        near_joints = candidates[
            candidates.geometry.distance(line_geom) < tolerance
        ]

        if near_joints.empty:
            split_lines.append(
                {
                    "geometry": line_geom,
                    "comp_type": row.comp_type,
                    "KABELTYP": row.KABELTYP,
                }
            )
        else:
            split_segments = cut_line_at_points(
                line_geom, list(near_joints.geometry)
            )

            for segment in split_segments:
                split_lines.append(
                    {
                        "geometry": segment,
                        "comp_type": row.comp_type,
                        "KABELTYP": row.KABELTYP,
                    }
                )

    split_lines_gdf = gpd.GeoDataFrame(
        split_lines, geometry="geometry", crs=lines.crs
    )

    final_gdf = pd.concat(
        [split_lines_gdf, other_lines], ignore_index=True
    )

    final_gdf["line_id"] = [
        f"line_{i}" for i in range(len(final_gdf))
    ]

    return final_gdf


def snap_joint_buses_to_lines(lines, buses, tolerance=0.1):
    """
    Snaps all 'joint'-buses directly on the closest line, if distance < tolerance.
    Secures that coords from buses are directly on the line to avoid skipping inaccurate data.
    """
    joint_buses = buses[
        buses.comp_type.isin(
            [
                "Hausanschlußmuffe",
                "Verbindungsmuffe",
                "Endmuffe",
                "Übergangsmuffe",
                "Reparaturmuffe",
                "vorverlegtes Ende",
                "HA-Kombimuffe",
            ]
        )
    ].copy()

    other_buses = buses[
        ~buses.comp_type.isin(
            [
                "Hausanschlußmuffe",
                "Verbindungsmuffe",
                "Endmuffe",
                "Übergangsmuffe",
                "Reparaturmuffe",
                "vorverlegtes Ende",
                "HA-Kombimuffe",
            ]
        )
    ].copy()
        
    lines = lines[lines.comp_type.isin(["lv_line", "hc_line", "mv_line"])]
    line_geometries = list(lines.geometry)
    str_tree = STRtree(line_geometries)

    snapped_indices = []
    for idx, bus in joint_buses.iterrows():
        nearest_idx = str_tree.nearest(bus.geometry)
        nearest_geom = line_geometries[nearest_idx]
        min_dist = bus.geometry.distance(nearest_geom)

        if 0 < min_dist <= tolerance:
            projected_distance = nearest_geom.project(bus.geometry)
            snapped_point = nearest_geom.interpolate(projected_distance)
            print(f"{bus.bus_id} is projected to the corresponding line")
            joint_buses.at[idx, "geometry"] = snapped_point
            snapped_indices.append(idx)

    buses_updated = pd.concat([joint_buses, other_buses]).sort_index()

    # buses_updated.to_file('/home/student/Documents/LoMa/Code/test_grid_buses_before_network.shp')

    return buses_updated


#####house connection buses
def find_connected_HA_line(
    current_point, HA_lines, visited_lines, tolerance=0.1
):
    """
    Searches for another HA line connected to the current point
    that has not yet been visited.

    Parameters:
    - current_point: Shapely Point to check connections from.
    - HA_lines: GeoDataFrame of house connection lines.
    - visited_lines: Set of already visited line indices.
    - tolerance: Maximum distance to consider points as connected.

    Returns:
    - Index of the next HA line and the opposite endpoint as the new reference point.
    """
    for idx, line in HA_lines.iterrows():
        if idx in visited_lines:
            continue

        start_point = Point(line.geometry.coords[0])
        end_point = Point(line.geometry.coords[-1])

        if current_point.distance(start_point) < tolerance:
            return idx, end_point
        elif current_point.distance(end_point) < tolerance:
            return idx, start_point

    return None, None  # No further connection found


def assign_house_connections_to_joints(lines, buses, tolerance=0.1):
    """
    Assigns each house connection bus to the nearest joint bus,
    following the connected HA lines starting from the end opposite the house connection.

    Parameters:
    - HA_lines: GeoDataFrame of house connection lines.
    - buses: GeoDataFrame of all buses with 'comp_type' attribute.
    - tolerance: Maximum distance to consider connections between lines.

    Returns:
    - mapping: Dictionary mapping house connection bus indices to joint bus indices.
    """
    HA_lines = lines[lines.comp_type == "HA_Line"]
    joint_buses = buses[buses.comp_type == "joint"].copy()
    house_buses = buses[buses.comp_type == "house_connection"].copy()

    mapping = {}

    for idx, house_bus in house_buses.iterrows():
        # Find the closest HA line to the house connection bus
        min_dist = float("inf")
        closest_line_idx = None

        for line_idx, line in HA_lines.iterrows():
            dist = line.geometry.distance(house_bus.geometry)
            if dist < min_dist:
                min_dist = dist
                closest_line_idx = line_idx

        if closest_line_idx is None:
            print(f"No HA line found for house connection {idx}")
            continue

        # Determine the end of the line opposite to the house connection bus
        line = HA_lines.loc[closest_line_idx]
        start_point = Point(line.geometry.coords[0])
        end_point = Point(line.geometry.coords[-1])

        if house_bus.geometry.distance(
            start_point
        ) < house_bus.geometry.distance(end_point):
            current_point = end_point
        else:
            current_point = start_point

        visited_lines = {closest_line_idx}

        # Follow connected HA lines until no further connection is found
        while True:
            next_line_idx, next_point = find_connected_HA_line(
                current_point, HA_lines, visited_lines, tolerance
            )
            if next_line_idx is None:
                break
            visited_lines.add(next_line_idx)
            current_point = next_point

        # Find the nearest joint bus to the final reference point
        joint_buses["distance"] = joint_buses.geometry.apply(
            lambda p: p.distance(current_point)
        )
        closest_joint_idx = joint_buses["distance"].idxmin()

        # Map the house connection bus to the closest joint bus
        mapping[idx] = closest_joint_idx

    return mapping


def map_load_bus_to_network_bus(buses, lines):
    """
    Adds house connection assignments and filters out buses that do not lie on any line.
    Buses with comp_type 'house connection' are filtered out beforehand.

    Parameters:
    - buses: GeoDataFrame containing all buses (with 'bus_id', 'geometry', 'comp_type')
    - lines: GeoDataFrame of lines

    Returns:
    - Filtered GeoDataFrame containing buses that have house connections
      or lie on any line, excluding buses with comp_type 'house connection' from input
    """
    # First filter out buses with comp_type 'house connection'
    buses_filtered = buses[buses["comp_type"] != "house_connection"]

    # Get assignments from house connections to network buses (based on original buses)
    mapping = assign_house_connections_to_joints(lines, buses)

    # Group house connections per network bus
    grouped_mapping = defaultdict(
        lambda: {"house_connection_bus_ids": [], "house_connection_geoms": []}
    )
    for house_bus_id, network_bus_id in mapping.items():
        house_bus = buses.loc[buses.index == house_bus_id].iloc[0]
        grouped_mapping[network_bus_id]["house_connection_bus_ids"].append(
            house_bus.bus_id
        )
        grouped_mapping[network_bus_id]["house_connection_geoms"].append(
            house_bus.geometry
        )

    # Add house connection info (empty lists if no assignment)
    buses_filtered = buses_filtered.copy()
    buses_filtered["house_connection_bus_ids"] = buses_filtered.index.map(
        lambda idx: (
            grouped_mapping[idx]["house_connection_bus_ids"]
            if idx in grouped_mapping
            else []
        )
    )
    buses_filtered["house_connection_geoms"] = buses_filtered.index.map(
        lambda idx: (
            grouped_mapping[idx]["house_connection_geoms"]
            if idx in grouped_mapping
            else []
        )
    )

    # Combine all line geometries for fast distance queries
    all_lines_geom = lines.unary_union

    def on_line(geom, tolerance=1e-6):
        return all_lines_geom.distance(geom) < tolerance

    # Keep only buses that lie on any line
    filtered_buses = buses_filtered[buses_filtered.geometry.apply(on_line)]

    return filtered_buses



def get_nearest_bus_robust(point, bus_gdf, tree, k=5):
    """
    Find nearest bus using KDTree (for candidate selection)
    + exact geometry distance (final refinement).
    """
    if bus_gdf.empty:
        return None, np.inf

    # k nächste Kandidaten über Tree
    _, idxs = tree.query([point.x, point.y], k=min(k, len(bus_gdf)))

    # exakte Distanz zur Geometrie
    candidates = bus_gdf.iloc[idxs]
    dists = candidates.geometry.apply(lambda g: point.distance(g))

    best = dists.idxmin()
    nearest_bus_id = bus_gdf.loc[best, "bus_id"]
    nearest_dist = dists[best]

    return nearest_bus_id, nearest_dist


def get_nearest_bus(point, bus_tree, buses_df):
    """
    Fast nearest bus by KDTree centroid distance.
    """
    dist, idx = bus_tree.query([point.x, point.y])
    return buses_df.iloc[idx]["bus_id"], dist


###creating pypsa grid
def import_grid_infrastructure(n, buses, lines, cable_types):
    """
    Based on exported shapefiles recreate the grid infrastructure as pysa_network

    Parameters
    ----------
    n : Pypsa_network

    buses : GeoDataFrame
        contains all potential bus-components
    lines : GeoDataFrame
        contains all potential line-components

    Returns
    -------
    None.

    """

    #### ------- add buses to network -------##
    buses["centroid"] = buses.geometry.centroid
    for _, row in buses.iterrows():
        n.add("Bus", row["bus_id"], x=row.centroid.x, y=row.centroid.y)
        n.buses.at[row["bus_id"], "comp_type"] = row["comp_type"]
        n.buses.at[row["bus_id"], "household_count"] = row["household_count"]
        n.buses.at[row["bus_id"], "trafo_cap"] = row["s_nom"]
        n.buses.at[row["bus_id"], "geom"] = row["geometry"]
        
    mask_mv = n.buses.comp_type == "MV_Muffe"
    n.buses.loc[mask_mv, "v_nom"] = 20
    n.buses.loc[~mask_mv, "v_nom"] = 0.4
    

    #### ------- add lines to network ------###
    # prepare KDTree
    lv_buses = buses[~buses.comp_type.isin(["MV_Muffe"])]
    lv_coords = np.array(
        [[geom.centroid.x, geom.centroid.y] for geom in lv_buses.geometry]
    )
    lv_tree = cKDTree(lv_coords)

    # trafo / distributor set + KDTree
    traf_buses = buses[buses.comp_type.isin(["distributor", "trafo"])]
    
    traf_coords = np.array(
        [[geom.centroid.x, geom.centroid.y] for geom in traf_buses.geometry]
    )
    traf_tree = cKDTree(traf_coords)
    
    #mv_buses (mv_joints) set
    mv_buses = buses[
    buses.comp_type.isin(["MV_Muffe", "trafo", "distributor", "trafo_HV"])
      ]
      
    mv_coords = np.array(
        [[geom.centroid.x, geom.centroid.y] for geom in mv_buses.geometry]
    )
    mv_tree = cKDTree(mv_coords)

    def process_line(row):
        line_id = row["line_id"]
        line_geom = row.geometry
        start_point = Point(line_geom.coords[0])
        end_point = Point(line_geom.coords[-1])

        comp_type = row["comp_type"]
        cable_type = row["KABELTYP"]
        
        if comp_type == "mv_line":
              buses_use = mv_buses
              tree_use = mv_tree
        else:
              buses_use = lv_buses
              tree_use = lv_tree

        # nearest bus to line_start
        bus0, dist0 = get_nearest_bus_robust(start_point, buses_use, tree=tree_use)

        # nearest bus to line_end
        bus1, dist1 = get_nearest_bus_robust(end_point, buses_use, tree=tree_use)

        # ---- Starting point ----
        if 10 > dist0 > 0.1:
            traf_bus, traf_dist = get_nearest_bus_robust(
                start_point, traf_buses, tree=traf_tree
            )
            if traf_dist < 10:
                bus0, dist0 = traf_bus, traf_dist
            else:
                bus0, dist0 = get_nearest_bus_robust(
                        start_point, buses_use, tree=tree_use
                    )
                print(
                        f"Check line {row['line_id']} – no nearby trafo/distributor (<10 m), using nearest bus instead."
                    )
        elif dist0 >= 10:
            print(
                f"Line {row['line_id']} skipped – no trafo/bus nearby at beginning of line."
            )
            return None  # skip

        # ---- End point ----
        if 10 > dist1 > 0.1:
            traf_bus, traf_dist = get_nearest_bus_robust(
                end_point, traf_buses, tree=traf_tree
            )
            if traf_dist < 10:
                bus1, dist1 = traf_bus, traf_dist
            else:
                bus1, dist1 = get_nearest_bus_robust(
                        end_point, buses_use, tree=tree_use
                    )
                print(
                        f"Check line {row['line_id']} – no nearby trafo/distributor (<10 m), using nearest bus instead."
                    )
        elif dist1 >= 10:
            print(
                f"Line {row['line_id']} skipped – no trafo/bus nearby at end of line."
            )
            return None  # skip

        # ------------------------------------------------------------------
        # Cable parameter
        # ------------------------------------------------------------------

        length_km = line_geom.length / 1000

        if cable_type in cable_types:
            r = cable_types[cable_type]["R"] * length_km
            x = (
                cable_types[cable_type]["L"]
                / 1000
                * 50
                * 2
                * np.pi
                * length_km
            )
            s_nom = (
                cable_types[cable_type]["U"]
                * cable_types[cable_type]["I_max"]
                * np.sqrt(3)
                / 1e6
            )
        else:
            r = 0.3 * length_km
            x = 0.05* length_km
            s_nom = 1

        capital_costs = 100000  # ToDo: adjust default values!!!

        # results
        return {
            "line_id": line_id,
            "bus0": bus0,
            "bus1": bus1,
            "r": r,
            "x": x,
            "s_nom": s_nom,
            "s_nom_extendable": True,
            "capital_cost": capital_costs,
            "length": length_km,
            "comp_type": comp_type,
            "cable_type": cable_type,
            "geom": line_geom,
        }

    ### parallelize line processing for faster model building
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_line)(row)
        for idx, row in lines.iterrows()
    )

    results_filtered = [res for res in results if res is not None]
    # list of parameters
    line_ids = [res["line_id"] for res in results_filtered]
    bus0_list = [res["bus0"] for res in results_filtered]
    bus1_list = [res["bus1"] for res in results_filtered]
    r_list = [res["r"] for res in results_filtered]
    x_list = [res["x"] for res in results_filtered]
    s_nom_list = [res["s_nom"] for res in results_filtered]
    s_nom_extendable_list = [
        res.get("s_nom_extendable", True) for res in results_filtered
    ]
    capital_cost_list = [res["capital_cost"] for res in results_filtered]
    length_list = [res["length"] for res in results_filtered]
    comp_type_list = [res["comp_type"] for res in results_filtered]
    cable_type_list = [res["cable_type"] for res in results_filtered]
    geom_list = [res["geom"] for res in results_filtered]

    n.add(
        "Line",
        line_ids,
        bus0=bus0_list,
        bus1=bus1_list,
        carrier="AC",
        r=r_list,
        x=x_list,
        s_nom=s_nom_list,
        s_nom_min=s_nom_list,
        s_nom_extendable=s_nom_extendable_list,
        capital_cost=capital_cost_list,
        length=length_list,
    )
    
    # additional attributes (useful for distinguish components / create ding0 shape )
    n.lines["comp_type"] = comp_type_list
    n.lines["cable_type"] = cable_type_list
    n.lines["geom"] = geom_list

    # store bus0 and bus1 for validation of the lines
    results_df = pd.DataFrame(results_filtered)
    lines = lines.merge(
        results_df[["line_id", "bus0", "bus1"]], on="line_id", how="left"
    )

    ### ---- add transformator to network (connect an generator at each trafo for test reasons -----###
    trafo_buses = n.buses[n.buses.comp_type.str.contains("trafo")]
    n.buses = n.buses.drop(
        "trafo_cap", axis="columns"
    )  # trafo_cap column isn't used anymore
    for idx, bus in trafo_buses.iterrows():
        comp = bus.comp_type
        if bus.comp_type =='trafo':
              bus1 = bus.name
              bus0 = f"{bus1}_MV"  # Same bus for MV level
              s_nom = bus.trafo_cap / 1e3 if bus.trafo_cap != 0 else 0.63
        else: 
              bus1 = f"{bus.name}_MV"
              bus0 = f"{bus.name}_HV"  # Same bus for HV level
              s_nom = bus.trafo_cap / 1e3 if bus.trafo_cap != 0 else 63
              ### to add both MV- and HV-bus to network
              n.add(
                  "Bus",
                  name=bus1,
                  v_nom=110,
                  carrier="AC",
                  household_count=bus.household_count,
                  x=bus.x,
                  y=bus.y,
                  geom=bus.geom,
                  comp_type=comp,
              )
              
              n.add(
                  "Generator",
                  name=f"gen_{idx}",
                  bus=bus1,
                  carrier="AC",
                  p_nom=1e9,
                  marginal_cost=100,
                  #p_nom_extendable =True,
              )
              

        n.add(
            "Bus",
            name=bus0,
            v_nom=20,
            carrier="AC",
            household_count=bus.household_count,
            x=bus.x,
            y=bus.y,
            geom=bus.geom,
            comp_type=comp,
        )

        n.add(
            "Transformer",
            name=f"trafo_{bus1}",
            bus0=bus0,
            bus1=bus1,
            x=0.03864647477581,  # example vlaues from dingo
            r=0.0103174603174603,
            s_nom=s_nom,
            s_nom_extendable=True,
            comp_type = comp
        )
        
        
    #connect MV_lines to correct side of trafo
    # --- build LV -> MS bus mapping ---
    mv_buses = n.buses.index[n.buses.index.str.endswith("_MV")]
      
    lv_to_mv = {
          mv_bus.replace("_MV", ""): mv_bus
          for mv_bus in mv_buses
    }
      
    # --- update MV lines ---
    mv_lines = n.lines[n.lines.comp_type == "mv_line"]
      
    for line_name, line in mv_lines.iterrows():
      
        bus0 = line.bus0
        bus1 = line.bus1
             
        new_bus0 = lv_to_mv.get(bus0, bus0)
        new_bus1 = lv_to_mv.get(bus1, bus1)
    
        if new_bus0 != bus0 or new_bus1 != bus1:
            n.lines.at[line_name, "bus0"] = new_bus0
            n.lines.at[line_name, "bus1"] = new_bus1

    # add carriers

    carriers = [
        "AC",
        "land_transport_EV",
        "14a",
        "home_battery",
        "solar_rooftop",
        "heat_pump",
        "charging_point",
        "conventional_load"      
    ]
    for c in carriers:
        n.add("Carrier", c)
    
    return buses, lines


def open_LV_circle(n, lv_line_idx):
    if lv_line_idx in n.lines.index:
        n.lines.drop(lv_line_idx, inplace=True)
        print(f"Leitung '{lv_line_idx}' wurde entfernt.")
    else:
        print(f"Leitung '{lv_line_idx}' nicht gefunden im Netzwerk.")

    return n


def implement_switches_LV(n, input_path):
    try:
        lines_with_switches = gpd.read_file(input_path)
    except Exception as e:
        print(f"Not possible to read switches Input file {input_path}: {e}")
        return n

    open_lines_geoms = lines_with_switches["geometry"].tolist()

    if not open_lines_geoms:
        print(
            "Keine offenen (zu löschenden) Leitungsgeometrien in der Shapefile gefunden. Netzwerk bleibt unverändert."
        )
        return n

    #mask = n.lines.geom.apply(
     #   lambda g: any(g.equals(o) for o in open_lines_geoms)
    #)
    # deleted_lines = n.lines[mask]
    # print(f"This lines are deleted:{deleted_lines.index.tolist()}")

    initial_line_count = len(n.lines)
    n.lines = n.lines[~n.lines.geom.isin(open_lines_geoms)]

    lines_deleted = initial_line_count - len(n.lines)

    print(
        f"Switches implemented. {lines_deleted} lines were deleted due to open switches."
    )
    print(f"Remaining lines: {len(n.lines)}.")

    n = fix_grid_infrastructure(n)

    return n



def fix_grid_infrastructure(n, min_size=2600):

    # Delete loop lines
    loop_lines = n.lines[n.lines.bus0 == n.lines.bus1]
    if not loop_lines.empty:
        print("⚠️ Loop line IDs found:")
        for idx in loop_lines.index:
            print(idx)
        print("⚠️ This lines will be deleted.")
        n.remove("Line", loop_lines.index.tolist())

    # Lösche unverbundene Busse (ohne Subnetz)
    line_buses = pd.concat([n.lines.bus0, n.lines.bus1]).unique()
    connected_transformers = n.transformers[
        n.transformers.bus0.isin(line_buses)
        | n.transformers.bus1.isin(line_buses)
    ]
    con_buses = pd.concat(
        [
            n.lines.bus0,
            n.lines.bus1,
            connected_transformers.bus0,
            connected_transformers.bus1,
        ]
    ).unique()

    uncon_buses = n.buses[~n.buses.index.isin(con_buses)]
    if not uncon_buses.empty:
        print(f"⚠️ Found not connected busses:\n{uncon_buses.index.tolist()}")
        print("⚠️ Buses and according components will be deleted.")
        n.remove("Bus", uncon_buses.index.tolist())

        # Lösche alle Komponenten, die an unverbundenen Bussen hängen
        for comp in [
            "Generator",
            "Transformer",
            "Load",
            "StorageUnit",
            "Link",
        ]:
            df = n.df(comp)

            if comp in ["Transformer", "Link"]:
                mask = df.bus0.isin(uncon_buses.index) | df.bus1.isin(
                    uncon_buses.index
                )
            else:
                mask = df.bus.isin(uncon_buses.index)

            to_remove = df[mask].index.tolist()

            if to_remove:
                # Finde die betroffenen Busse
                if comp in ["Transformer", "Link"]:
                    buses_used = (
                        pd.concat(
                            [
                                df.loc[to_remove, "bus0"],
                                df.loc[to_remove, "bus1"],
                            ]
                        )
                        .unique()
                        .tolist()
                    )
                else:
                    buses_used = df.loc[to_remove, "bus"].unique().tolist()

                print(
                    f"⚠️ Remove {len(to_remove)} {comp}(s) at unconnected buses: {buses_used}"
                )

                n.remove(comp, to_remove)

    # Erkenne Subnetzwerke
    G = n.graph()
    components = list(nx.connected_components(G))

    for comp in components:
        if len(comp) < min_size:
            print(
                f"⚠️ Small subnetwork with {len(comp)} Buses found: {sorted(list(comp))[:5]} ..."
            )
            # Lösche alle Busse im Subnetz
            n.remove("Bus", list(comp))
            # Lösche alle Komponenten, die an diesen Bussen hängen
            for comp_name in [
                "Generator",
                "Transformer",
                "Load",
                "StorageUnit",
                "Link",
                "Line",
            ]:
                df = n.df(comp_name)
                if comp_name in ["Transformer", "Link", "Line"]:
                    to_remove = df[
                        df.bus0.isin(comp) | df.bus1.isin(comp)
                    ].index.tolist()
                else:
                    to_remove = df[df.bus.isin(comp)].index.tolist()
                if to_remove:
                    print(
                        f"⚠️ Remove {len(to_remove)} {comp_name}(s) inside of the subnetwork"
                    )
                    n.remove(comp_name, to_remove)
        else:
            print(
                f"⚠️ Main Subnetwork with {len(comp)} Buses found – will be maintained. Example buses:{sorted(list(comp))[:5]}"
            )

    print("Infrastructure of network is fixed.")

    return n


def import_ev_chargers(n):

    # use shapefile for ladesäulen
    return

def export_shape_files_from_network(n, output_path):
      buses_path = os.path.join(output_path, "buses_final.shp")
      buses = n.buses.copy()

      buses["geometry"] = [
            Point(xy) for xy in zip(buses["x"], buses["y"])
            ]

      gdf_buses = gpd.GeoDataFrame(
          buses,
          geometry="geometry",
          crs="EPSG:32632"  
      )  
      
      gdf_buses.to_file(buses_path)
      
      lines_path = os.path.join(output_path, "lines_final.shp")
      lines = n.lines.copy()
      
      gdf_lines = gpd.GeoDataFrame(
          lines,
          geometry="geom",
          crs="EPSG:32632"
      )
      gdf_lines.to_file(lines_path)
      
      
      
      
def create_pypsa_network(
    shape_files_folder,
    q_households_folder,
    heat_pump_folder,
    cable_types,
    household_count,
    export_shape_files,
    switches_folder,
    census_data,
):
    print("=== [1/10] Initializing PyPSA network ===")
    n = pypsa.Network()
    n.to_crs(32632)

    time_index = pd.date_range("2023-01-01", periods=8760, freq="h")
    n.snapshots = time_index
    print(f"    -> Snapshots set: {len(time_index)} hours")

    print("=== [2/10] Reading grid shape files ===")
    buses, lines = create_gdf_from_shape(shape_files_folder)
    print(f"    -> Loaded {len(buses)} buses, {len(lines)} lines")

    print("=== [3/10] Snapping joint buses to lines ===")
    buses = snap_joint_buses_to_lines(lines, buses)

    print("=== [4/10] Counting households per bus ===")
    if household_count:
        print("    -> Using census data")
        buses = count_households_per_bus_census_data(buses, census_data)
    else:
        print("    -> Using input household file")
        buses = count_households_per_bus_input_file(buses, q_households_folder)

    print("=== [5/10] Splitting lines at joint buses ===")
    split_lines = split_lines_on_joints(lines, buses)
    print(f"    -> Lines after splitting: {len(split_lines)}")

    print("=== [6/10] Importing grid infrastructure into PyPSA ===")
    buses, lines = import_grid_infrastructure(
        n, buses, split_lines, cable_types
    )
    print(
        f"    -> PyPSA now contains "
        f"{len(n.buses)} buses, "
        f"{len(n.lines)} lines"
    )

    print("=== [7/10] Implementing LV switches ===")
    n = implement_switches_LV(n, switches_folder)

    print("=== [8/10] Fixing grid infrastructure ===")
    fix_grid_infrastructure(n)
    
    print("=== [9/10] Merge lines, just seperated by unused bus ===")
    #merge_lines_splitted_by_bus(n)
    
    if export_shape_files:
        print("=== [10/10] Exporting grid shapefiles ===")
        os.makedirs("results", exist_ok=True)
        export_shape_files_from_network(n, "./results")
        print("    -> Shapefiles written to ./results")
    

    print("=== Network creation finished successfully ===")

    return n

