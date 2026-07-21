# Projekt-Konfiguration

LoMa baut aus Shapefiles + Datenpaketen ein PyPSA-Netzmodell für eine Region auf.
Alle Stadtwerke-/regionsspezifischen Werte, die früher im Code hartcodiert waren
(ursprünglich nur für Husum), sind in einer YAML-Datei gebündelt: der
**Projekt-Config**. Für ein neues Stadtwerk wird eine Kopie dieser Datei mit
eigenen Pfaden/Werten angelegt (z. B. `project_config_musterstadt.yaml`) und in
`appl.py` über `PROJECT_CONFIG_PATH` referenziert.

Referenzbeispiel: `data/Input_files/project_config_husum.yaml` (liegt im
Ausführungsverzeichnis, nicht im Code-Repository, analog zu `cable_types.yaml`).

## Struktur

- **`project`**
  - `name`: Stadtwerk/Region, nur informativ.
  - `scenario`: Szenario-Bezeichnung (z. B. `statusQuo`, `2035`), ebenfalls nur
    informativ – wird an keiner Stelle im Code nachgeschlagen.
  - `crs`: Ziel-CRS der Shapefiles, z. B. `"EPSG:32632"` (UTM-Zone). Husum liegt
    in UTM 32N; andere Regionen müssen ihre eigene UTM-Zone eintragen.
  - `is_test_model`: `true` für kleine Testmodelle/Teilausschnitte. Ersetzt die
    frühere Heuristik `len(n.buses) < 1000` und steuert u. a., ob CTS-Loads
    entfernt werden und ob ein Dummy-MV-Netz statt eines echten Slack-Busses
    verwendet wird.
  - `voltage_levels.lv` / `voltage_levels.mv`: Spannungsebenen in kV (Husum:
    0.4 / 20). Das Modell geht weiterhin von genau zwei Ebenen aus – eine
    echte Mehr-Ebenen-Architektur ist bewusst nicht Teil dieser
    Generalisierung.

- **`paths`**: alle projektspezifischen Dateipfade (Shapefiles, Kabeltypen,
  Wärmepumpen-Shapefile, PV/Batterie-Datensätze etc.), relativ zum
  Ausführungsverzeichnis. `heat_pump_shapefile` kann ein regionsweiter
  Datensatz sein (z. B. SH-weit statt nur für das eigene Projektgebiet
  vorgefiltert) – siehe Wärmepumpen-Abschnitt unten.

- **`gis_source_files`**: Mapping von logischer Rolle (z. B. `lv_lines`,
  `mv_lv_stations`) auf den tatsächlichen Dateinamen im GIS-Export-Ordner.

- **`gis_column_mapping`**: Spaltennamen und erwartete Werte im GIS-Export
  (z. B. `status_column`, `joint_type_column`, `transformer_hv_marker_value`).

- **`thresholds`**: geometrische Toleranzen beim Netzaufbau
  (`line_endpoint_snap_m`, `switch_buffer_m`).

- **`scenario_targets`**: Zielwerte für das im Config-File hinterlegte
  Szenario (Wärmepumpen, Ladepunkte, PV-/Batterie-Kapazität,
  Zeitindex-Startjahr). Diese Werte werden direkt verwendet, **ohne**
  automatische Skalierung nach Modellgröße – sie sollten also bereits auf die
  tatsächliche Größe des jeweiligen Modells kalibriert sein.

- **`nuts3_focus_region`**: NUTS3-Region für die Industrie-Bedarfsermittlung
  (`demand_regio`-Daten).

## Offene Frage: Herkunft der GIS-Dateinamen/Spaltennamen

Die Dateinamen (z. B. `Gis NSP Kabelabschnitt Verlauf.shp`) und Spaltennamen
(`STATUS`, `ART`, `LOKATION_S`, `HAUSNUMMER`, `TRAFOBELAS`, `KABELTYP`) sehen
nach einem standardisierten Export einer bestimmten GIS-/Netzinformations-
Software aus, nicht nach einer Husum-Eigenheit. Falls ein neues Stadtwerk
dieselbe Software nutzt, sind die Husum-Werte vermutlich ein guter
Startpunkt; andernfalls müssen `gis_source_files` und `gis_column_mapping`
vollständig an das eigene Exportformat angepasst werden.

## Bekannte Grenzen dieser Generalisierung

- **CRS**: Im Kernmodul (`import_network_from_shape_files.py`) und in
  `pv_rooftop_and_home_battery.py` kommt das CRS aus der Config. In folgenden
  Dateien ist `EPSG:32632` weiterhin hartcodiert und müsste für Regionen
  außerhalb UTM-Zone 32N ebenfalls angepasst werden:
  - `loma/plot_results.py`
  - `loma/pypsa_model_into_ding0_shape.py`
  - `loma/demands/cts_demands.py`
  - `loma/demands/import_household_demand.py`
  - `loma/demands/create_industrial_demand.py`
  - `loma/demands/import_EV_demand.py`
  - `loma/demands/household_count.py`
- **Regionaldaten**: `create_household_distribution.py` und
  `create_industrial_demand.py` nutzen Schleswig-Holstein-spezifische
  Grenz-Shapefiles (`schleswig_holstein.shp`, `Nuts3_SH.shp`) zum Zuschneiden
  der bundesweiten Zensus-/OSM-Daten. Für Stadtwerke außerhalb SH müssen diese
  beiden Pfade durch die Grenzen des jeweiligen Bundeslands ersetzt werden.
- **Wärmepumpen-Standorte**: `demands/import_hp_demand.py` liest den unter
  `paths.heat_pump_shapefile` konfigurierten Datensatz und filtert ihn zur
  Laufzeit per `mask=` auf die MV-Grid-Grenze des jeweiligen Projekts
  (`paths.mv_grid_boundary`). `heat_pump_shapefile` kann dadurch ein
  regionsweiter Datensatz sein (z. B. SH-weit); eine pro Projekt manuell
  vorgefilterte Datei ist nicht mehr nötig, da die Filterung jetzt im Code
  passiert.
- **Spannungsebenen**: nur LV/MV, keine weiteren Zwischenebenen.
- **ding0/eDisGo-Export**: `pypsa_model_into_ding0_shape.py` wird aktuell
  nicht aus `appl.py` aufgerufen und ist nicht Teil dieser Generalisierung
  (weiterhin mit Husum-spezifischen Werten wie `mv_grid_id=35725`).
