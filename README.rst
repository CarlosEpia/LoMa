LoMa
====

LoMa builds a `PyPSA <https://pypsa.org/>`_ grid model of a low-voltage/
medium-voltage distribution grid from GIS shapefiles and demand data. It was
originally developed to evaluate the potential of `§14a EnWG
<https://www.gesetze-im-internet.de/enwg_2005/__14a.html>`_ to provide
flexibility (heat pumps, wallboxes, home batteries) in German LV/MV grids,
using the town of Husum as the reference case, and has since been
generalized so other Stadtwerke (municipal utilities) can build a model for
their own grid by supplying their own shapefiles and a project config.

What it does
------------

Starting from a Stadtwerk's GIS export (cable lines, joints, distributors,
transformer stations) and a set of bundled/regional datasets (census
household data, industrial/CTS demand, heat pump and EV projections, PV
rooftop and home battery inventories), LoMa:

- builds the LV/MV network topology (buses, lines, transformers) as a PyPSA
  network,
- distributes household, commercial/industrial and heat pump demand across
  the network's buses,
- inserts solar rooftop generators and home battery storage units, scaled to
  a scenario target,
- inserts EV charging point loads, scaled to a scenario target,
- optionally applies the §14a load-reduction constraint and flexibility
  generators.

Installation
------------

LoMa targets Python >= 3.10. Install it in editable mode into a virtual
environment::

    python -m venv .venv
    source .venv/bin/activate
    pip install -e .

Usage
-----

1. Create a project config YAML for your Stadtwerk/region, following the
   reference example and full key-by-key documentation in
   `docs/project_config.md <docs/project_config.md>`_. This bundles all
   region-specific settings: shapefile paths, the GIS schema (file/column
   names), CRS, voltage levels, and scenario targets.
2. Point ``PROJECT_CONFIG_PATH`` in ``loma/appl.py`` to your config file.
3. Run ``loma/appl.py`` from a working directory that contains your
   ``data/`` folder (shapefiles, cable types, bundled datasets) as
   referenced by the config's paths::

       python loma/appl.py

   This builds the network and exports it to a ``results/`` folder as a
   PyPSA CSV export.

See `docs/project_config.md <docs/project_config.md>`_ for the full
configuration reference, including known limitations of the current
generalization (remaining region-specific assumptions that a new Stadtwerk
may still need to adjust).

License
-------

LoMa is licensed under AGPL-3.0-or-later.
