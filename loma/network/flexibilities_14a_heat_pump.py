#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements German §14a EnWG flexibility handling for heat pump loads."""
import numpy as np


def insert_heat_pump_flexibilities_14a(n):
    """For each heat pump load, add a "14a" generator that can supply the
    part of the load above the guaranteed minimum (4.2 kW, per §14a EnWG)
    as flexibility, with a high marginal cost so it's only used when line
    capacity would otherwise be exceeded."""
    heat_loads = n.loads[n.loads.index.str.contains("heat")]

    for idx, row in heat_loads.iterrows():
        bus = row.bus

        max_load_allowed = 0.0042  # 4.2kW maximum reduction of the load

        # max capacity of generator: load_timeseries - 4.2 kW (not < 0)
        p_gen_14a = np.maximum(n.loads_t.p_set[idx] - max_load_allowed, 0)
        p_max = p_gen_14a.max()
        if p_max > 0:
            n.add("Generator",
                  name=f"Gen_14a_{idx}",
                  bus=bus,
                  carrier = "14a",
                  p_nom=p_max,
                  p_max_pu=p_gen_14a / p_gen_14a.max(),
                  marginal_cost=200,   # high value, that generator is used only if line capacity is not enough
                  overwrite=True
                  )

    return n