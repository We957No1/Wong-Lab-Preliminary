# Data exports

Running all non-interactive cells in `five_mirror_EUV_projection_simulation.ipynb`
populates this directory with the CSV and JSON files documented in `README.tex`.

The notebook intentionally writes numeric fractions (0 to 1) for columns whose
names end in `_fraction`; columns ending in `_percent` are already multiplied by
100. Spatial ray-intercept coordinates remain in millimetres unless the column
name explicitly ends in `_nm`.
