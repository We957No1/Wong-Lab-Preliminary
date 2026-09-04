# Ray-optics notebooks

Files:

- `Plane Mirror.ipynb` — plane mirror with Optiland geometry/coating and a full xrt/raycing source–mirror–screen trace.
- `lens_group.ipynb` — editable two- or three-lens Optiland model, exact spot diagrams, paraxial movement comparison, sweep, and widget.
- `ray_optics_simulation_report.tex` — compile-ready report; missing notebook figures appear as placeholders until the notebooks are run.

Recommended setup: Python 3.11 or 3.12 in a virtual environment. This workspace
now contains a tested `.venv`; install or refresh its dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then select `.venv\Scripts\python.exe` as the VS Code Jupyter kernel and run both notebooks from top to bottom. Compile with the verified local Tectonic engine:

```powershell
$env:TECTONIC_CACHE_DIR = "$PWD\.tectonic-cache"
.\.tools\tectonic\tectonic.exe ray_optics_simulation_report.tex --keep-logs --synctex
```

If a full TeX distribution with `latexmk` is already installed, this is equivalent:

```powershell
latexmk -pdf ray_optics_simulation_report.tex
```

The mirror notebook defaults to an X-ray wavelength so xrt can use its material tables. For visible or infrared light, set `reflectance_model='constant'` and enter measured reflectance/absorptance values.
