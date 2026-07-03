#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 17:32:48 2025

@author: paul
"""
import os
import matplotlib.pyplot as plt
import geopandas as gpd


def plot_results(n):
    n.lines["lines_exp"] = n.lines["s_nom_opt"] - n.lines["s_nom"]
    lines_exp = n.lines["lines_exp"][n.lines["lines_exp"] > 0.001]
    lines_exp_t = n.lines_t["p0"].loc[:, lines_exp.index]
    lines_exp_t = lines_exp_t.loc[n.snapshots, :]
    lines_exp_t = lines_exp_t.apply(abs)

    gen_14a = n.generators[n.generators.carrier == "14a"]
    gen_14a_t = n.generators_t["p"].loc[:, gen_14a.index]
    gen_14a_t = gen_14a_t.loc[n.snapshots, :]

    fig, axL = plt.subplots(figsize=(12, 6))
    lines_exp_t.plot(ax=axL, legend=False)
    gen_14a_t.sum(axis=1).plot(ax=axL, label="14a - Heat Pump")

    handles, labels = axL.get_legend_handles_labels()
    filtered = [
        (h, l) for h, l in zip(handles, labels) if not l.startswith("line_")
    ]
    if filtered:
        h, l = zip(*filtered)
        axL.legend(h, l, loc="upper left")

    axL.set_ylabel("Power [MW]")
    axL.set_xlabel("Time")
    plt.tight_layout()

    # export plot to results folder
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/time_series_plot.png", dpi=300)

    plt.close(fig)

    lines = gpd.GeoDataFrame(n.lines, geometry="geom", crs=32632)
    lines.plot()
    lines.to_file("gis_files/lines_loma.geojson", driver="GeoJSON")
    buses = gpd.GeoDataFrame(n.buses, geometry="geom", crs=32632)
    buses.plot()
    buses.to_file("gis_files/buses_loma.geojson", driver="GeoJSON")
