#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Counts households per bus, either from a manual street/house-number input file or from census cell data."""
import re
import pandas as pd
import geopandas as gpd
from rapidfuzz import process, fuzz
import numpy as np


def hausnummer_split(hn):
    """
    Transfer housenumbers into number and letter
    Example:
    '5a'   -> (5, 'a')
    '5'    -> (5, '')
    """
    if pd.isna(hn):
        return None, None
    m = re.match(r"(\d+)\s*([a-zA-Z]*)", str(hn).strip())
    if m:
        num = int(m.group(1))
        letter = m.group(2).lower() if m.group(2) else ""
        return num, letter
    return None, None


def parse_bus_numbers(hn_entry):
    """
    Wandelt komplexe Hausnummern-Einträge in eine Liste von (nummer, letter)-Tupeln um.
    Beispiele:
    '14, 14a, 14b' -> [(14,''), (14,'a'), (14,'b')]
    '39-43' -> [(39,''), (41,''), (43,'')]
    '21-21a' -> [(21,''), (21,'a')]
    """
    if not hn_entry or pd.isna(hn_entry):
        return []

    parts = [p.strip() for p in str(hn_entry).split(",")]
    numbers = []

    for p in parts:
        if "-" in p:
            left, right = [s.strip() for s in p.split("-", 1)]
            left_num, left_letter = hausnummer_split(left)
            right_num, right_letter = hausnummer_split(right)

            # case 1: both sides are just numbers
            if (
                left_num
                and right_num
                and left_letter == ""
                and right_letter == ""
            ):
                step = 2
                for num in range(left_num, right_num + 1, step):
                    numbers.append((num, ""))

            # case 2: right side is combination of numbers and letters
            elif left_num == right_num and right_letter != "":
                numbers.append((left_num, ""))
                numbers.append((right_num, right_letter))

            else:
                # fallback
                if left_num:
                    numbers.append((left_num, left_letter))
                if right_num:
                    numbers.append((right_num, right_letter))

        else:
            # just single number
            numbers.append(hausnummer_split(p))

    return numbers


def count_households_per_bus_input_file(buses, path, threshold=80):
    # load households
    q_households = pd.read_csv(path)

    # parse house numbers
    buses["parsed_numbers"] = buses["Hausnummer"].apply(parse_bus_numbers)
    q_households[["Hausnummer_int", "Hausnummer_letter"]] = q_households[
        "Nummer"
    ].apply(lambda x: pd.Series(hausnummer_split(x)))

    # initialize counts
    buses["household_count"] = 0
    mask_house_conn = buses["comp_type"] == "house_connection"

    # match each household
    for _, hh_row in q_households.iterrows():
        street = hh_row["Straße"]
        num = hh_row["Hausnummer_int"]
        letter = hh_row["Hausnummer_letter"]
        if pd.isna(num):
            continue

        # fuzzy match street name
        street_choices = buses.loc[mask_house_conn, "Straße"].tolist()
        best_match, score, _ = process.extractOne(
            street, street_choices, scorer=fuzz.ratio
        )
        if score < threshold:
            print(street, "übersprungen")
            continue  # kein aktzeptables Match

        # filter buses on that street
        street_buses = buses[mask_house_conn & (buses["Straße"] == best_match)]
        if street_buses.empty:
            print("Adresse:", street, num, letter, "übersprungen")
            continue

        # 1) exact match (number + letter)
        exact_matches = street_buses[
            street_buses["parsed_numbers"].apply(
                lambda nums: (num, letter) in nums
            )
        ]
        if not exact_matches.empty:
            buses.loc[exact_matches.index, "household_count"] += 1
            continue

        # 2) fallback: match number only
        number_matches = street_buses[
            street_buses["parsed_numbers"].apply(
                lambda nums: any(n == num for n, _ in nums)
            )
        ]
        if not number_matches.empty:
            buses.loc[number_matches.index, "household_count"] += 1
            continue

        # 3) fallback: nearest number on same side
        valid_street_buses = street_buses[
            street_buses["parsed_numbers"].apply(
                lambda nums: nums is not None
                and any(
                    n is not None
                    and not (isinstance(n, float) and np.isnan(n))
                    for n, _ in nums
                )
            )
        ]

        same_side = valid_street_buses[
            valid_street_buses["parsed_numbers"].apply(
                lambda nums: any(n % 2 == num % 2 for n, _ in nums)
            )
        ]

        if same_side.empty:
            same_side = (
                valid_street_buses  # if no one on same side use all buses
            )

        def min_diff(nums):
            return min(abs(n - num) for n, _ in nums)

        same_side = same_side.assign(
            diff=same_side["parsed_numbers"].apply(min_diff)
        )
        nearest_idx = same_side["diff"].idxmin()
        buses.loc[nearest_idx, "household_count"] += 1


    # secure that every house_connection bus has at least one one household
    buses.loc[
        mask_house_conn & (buses["household_count"] == 0), "household_count"
    ] = 1

    return buses


def count_households_per_bus_census_data(buses, census_data):
    # create GeoDataFrames
    census_data = gpd.GeoDataFrame(
        census_data,
        geometry=gpd.GeoSeries.from_wkt(census_data["geometry"]),
        crs="EPSG:4326",
    )
    census_data = census_data.to_crs("EPSG:32632")
    buses = gpd.GeoDataFrame(
        buses.copy(), geometry="geometry", crs="EPSG:32632"
    )

    # find corresponding cell for each house_bus
    house_buses = buses[buses["comp_type"] == "house_connection"].copy()
    house_buses = gpd.sjoin_nearest(
        house_buses,
        census_data[["geometry", "Insgesamt_Haushalte"]],
        how="left",
        distance_col="dist",
    )
    house_buses = house_buses.rename(columns={"index_right": "closest_cell"})

    # distribute amount of households for each cell equaly under the buses
    house_buses["household_count"] = 0
    for cell_idx, group in house_buses.groupby("closest_cell").groups.items():
        n_buses = len(group)
        households_in_cell = census_data.loc[cell_idx, "Insgesamt_Haushalte"]
        base = households_in_cell // n_buses
        rest = households_in_cell % n_buses

        counts = np.full(n_buses, base, dtype=int)
        counts[: int(rest)] += 1  # Rest gleichmäßig verteilen

        house_buses.loc[group, "household_count"] = counts

    # set amount of households in bus-dataframe
    house_buses["household_count"] = house_buses["household_count"].apply(
        lambda x: max(1, x)
    )
    buses = buses.merge(
        house_buses[["household_count"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    # secure that no NaN values ecist
    buses["household_count"] = buses["household_count"].fillna(0).astype(int)

    return buses
