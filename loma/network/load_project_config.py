#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml


def load_project_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
