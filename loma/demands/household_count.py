#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 09:10:51 2025

@author: student
"""
import re

import geopandas as gpd
#### manual household count based on inout file
import pandas as pd


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
        letter = m.group(2).lower() if m.group(2) else ''
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
    
    parts = [p.strip() for p in str(hn_entry).split(',')]
    numbers = []
    
    for p in parts:
        if '-' in p:
            left, right = [s.strip() for s in p.split('-', 1)]
            left_num, left_letter = hausnummer_split(left)
            right_num, right_letter = hausnummer_split(right)

            # case 1: both sides are just numbers
            if left_num and right_num and left_letter == '' and right_letter == '':
                step = 2 
                for num in range(left_num, right_num + 1, step):
                    numbers.append((num, ''))
            
            # case 2: right side is combination of numbers and letters
            elif left_num == right_num and right_letter != '':
                numbers.append((left_num, ''))
                numbers.append((right_num, right_letter))
            
            else:
                # fallback
                if left_num: numbers.append((left_num, left_letter))
                if right_num: numbers.append((right_num, right_letter))
        
        else:
            # just single number
            numbers.append(hausnummer_split(p))
    
    return numbers

'''
def parse_bus_numbers(hn_entry):
    """
    Transfer special housenumber entries into a list of numbers/letters-tuples
    Examples:
    '14, 14a, 14b' -> [(14,''), (14,'a'), (14,'b')]
    '39-43' -> [(39,''), (41,''), (43,'')]
    """
    if pd.isna(hn_entry):
        return []
    
    parts = [p.strip() for p in str(hn_entry).split(',')]
    numbers = []
    
    for p in parts:
        if '-' in p:
            bounds = re.findall(r'\d+', p)
            if len(bounds) == 2:
                start, end = map(int, bounds)
                step = 2 if start % 2 == 0 else 2
                for num in range(start, end + 1, step):
                    numbers.append((num, ''))
        else:
            numbers.append(hausnummer_split(p))
    return numbers

'''


def count_households_per_bus(buses, path):
    #load and prepare data
    q_households = pd.read_csv(path)
    buses = buses.rename(columns={'HAUSNUMMER': 'Hausnummer', 'LOKATION_S': 'Strasse'})
    buses['parsed_numbers'] = buses['Hausnummer'].apply(parse_bus_numbers)
    q_households[['Hausnummer_int', 'Hausnummer_letter']] = q_households['Hausnummer'].apply(lambda x: pd.Series(hausnummer_split(x)))
    
    buses['house_count'] = 0
    mask_house_conn = buses['comp_type'] == 'house_connection'
    buses.loc[mask_house_conn, 'house_count'] = 1
    
    # match each entry of q_households with one bus
    for _, hh_row in q_households.iterrows():
        street = hh_row['Strasse']
        num = hh_row['Hausnummer_int']
        letter = hh_row['Hausnummer_letter']
        
        if pd.isna(num):
            continue
        
        # 1) filter buses from the same street
        street_buses = buses[mask_house_conn & (buses['Strasse'] == street)]
        if street_buses.empty:
            continue
        
        # 2) try to find exact match (Number, letter))
        exact_matches = street_buses[
            street_buses['parsed_numbers'].apply(lambda nums: (num, letter) in nums)
        ]
        
        if not exact_matches.empty:
            buses.loc[exact_matches.index, 'house_count'] += 1
            continue
        
        # 3) Fallback: just compare number without letter
        number_matches = street_buses[
            street_buses['parsed_numbers'].apply(lambda nums: any(n == num for n, _ in nums))
        ]
        if not number_matches.empty:
            buses.loc[number_matches.index, 'house_count'] += 1
            continue
        
        # 4) Fallback: close number on the same street side if available
        same_side = street_buses[
            street_buses['parsed_numbers'].apply(lambda nums: any(n % 2 == num % 2 for n, _ in nums))
        ]
        if same_side.empty:
            same_side = street_buses  # if no one on same side use all buses
        
        def min_diff(nums):
            return min(abs(n - num) for n, _ in nums)
        
        same_side = same_side.assign(diff=same_side['parsed_numbers'].apply(min_diff))
        nearest_idx = same_side['diff'].idxmin()
        buses.loc[nearest_idx, 'house_count'] += 1
        
    return buses
               
               
               
               
    
#