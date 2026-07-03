#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loads the project config YAML that bundles all Stadtwerk/region-specific settings (see docs/project_config.md)."""

import yaml


def load_project_config(yaml_path):
    """Load the project config from the given YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
