#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loads cable type electrical parameters (voltage, current, resistance, inductance) from a YAML file."""

import yaml


def load_kabeltypen(yaml_path):
    """Load the cable type definitions from the given YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)