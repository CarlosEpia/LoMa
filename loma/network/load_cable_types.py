#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:17:09 2026

@author: student
"""

# 1. Import hinzugefügt
import yaml

# 2. Hilfsfunktion
def load_kabeltypen(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)