27 OCT — INTERACTIVE MIRROR DEFECTS

OPEN THESE NOTEBOOKS
1. mirror_dent_zernike.ipynb
   Condensed visible-light experiment: default 550 nm, editable complex field
   reflectance, local Gaussian dent to global quadratic curvature, Zernike fit,
   direct/reconstructed point and object images, integer-order convergence.
   C04 shows all 45 Zernike components through n=8: a full OPD coefficient
   table, signed bar chart and coefficient map. The same decomposition updates
   in the live controls and exports as zernike_decomposition_n8.csv.
   Edit C02; live controls in C07; baseline export in C08.
   C09 adds three facing mirrors: 0-3 identical dents, two equal adjustable
   gaps, common incidence/complex reflectance, and plots only. The ideal
   reference is the same three-mirror train without dents. Fresnel propagation
   includes the two gaps before a common ideal focusing step.

2. euv_bilayer_zernike.ipynb
   New two-finite-layer experiment: default 40 nm, unnamed arbitrary materials
   A and B and an unnamed substrate. Complex refractive indices are editable
   illustrative assumptions, with no named-material or measured-data claim.
   A calculated total thickness makes equal physical layers a reflectance
   maximum for this chosen model. Edit C02; live controls C06; export C07.

3. optics_validation.ipynb
   Separate physical, numerical and Python widget-callback checks for both models.
   Verification is not run in the two demonstration notebooks.

RUN
Select the existing optics Python kernel:
C:\Users\fxl04\Desktop\Prof. Wong Grp Papers\Ray Optics\.venv\Scripts\python.exe
Run All in the selected notebook. Saved figure output is included. Live sliders
require a running kernel and frontend widget support. Change values, then click
Compute. Computational helper source is in mirror_dent_optics.py and
euv_bilayer_optics.py; helper implementations and tests are separate from the
main demonstration.

To execute all notebooks from a clean kernel, run in this directory:
& '..\..\Ray Optics\.venv\Scripts\python.exe' -X utf8 execute_notebook.py
Pass a notebook filename to execute just that file. requirements.txt records the
validated package versions for another compatible Python environment.

DEFINITIONS THAT MATTER
Visible ideal: the same system without surface figure error. Pupil, illumination,
wavelength, complex reflectance and chosen defocus are identical. The uniform
reflection phase changes no isolated image intensity; magnitude changes brightness.
The local/global shape slider mixes a localized Gaussian with a global quadratic
bow. Global quadratic curvature adds defocus, rather than primary spherical error.
At fixed depth, widening the Gaussian itself reduces its curvature magnitude.

Two-layer ideal: equal, spatially uniform layer thicknesses at the SAME optical
settings. Total coating thickness stays constant at every pupil location. The
nominal split deviation runs from -100% (all B) to +100% (all A). The spatial
nonuniformity control separates uniform brightness/global phase changes from
actual image-shape changes. The exact bounded map is derived in the notebook.
The reflection phase is referred to one fixed top plane: no surface-sag phase is
added to the buried-interface thickness error.

40 NM DESIGN
The arbitrary indices are nA=.920+.006i, nB=.995+.002i, ns=.980+.010i at 40 nm.
They are held fixed during wavelength sweeps; real materials require dispersion.
The exact two-film design gives total thickness ~20.9881 nm, equal layers
~10.4941 nm, and power reflectance ~0.7729%. Equal thickness is a conditional
maximum over thickness split for these selected settings, not a universal rule.
Changing wavelength, indices or total thickness can move that maximum. The
uniform equal-layer pupil remains the ideal imaging reference regardless.

NORMALIZATION AND RECONSTRUCTION
All raw images use a common unit-reflector, in-focus incident-light normalization;
reflected-power changes are retained, including in extended-object images. The
EUV plot also labels a separate unit-flux comparison solely to compare shapes.
Visible coefficients describe OPD in nm. Coating coefficients describe the
complex relative pupil r/r_reference and are dimensionless, with real and
imaginary parts exported. A phase-only fit would miss coating amplitude errors.

OUTPUTS AND REPRODUCIBILITY
results/visible and results/euv contain the current figures, parameters,
coefficients, image metrics, arrays, convention records and file hashes.
checks contains clean-kernel execution and the separate validation reports.
Previous notebook/helpers/results are preserved under archive. An unused named-
material draft and its source tables are also archived; they are not used by the
current arbitrary-material simulation. No network access is needed to rerun.

LIMITS
These are monochromatic scalar paraxial models. The coating model is locally
planar at normal incidence and omits roughness, interdiffusion, vector/lateral
coupling and material dispersion. It is two finite layers, not a many-period
multilayer coating. Independent tests validate the declared calculations rather
than a particular manufactured mirror.

MODULE RELOAD FIX
The saved AttributeError arose from an older cached mirror_dent_optics import.
C01, C04 and C07 now reload it explicitly; C09 reloads the train helper too.
Reopen this updated notebook and Run All once in an existing kernel. The
observed traceback and reproducible reload check are retained under checks/.

THREE-MIRROR OUTPUT
results/visible_three_mirrors contains the initial three-mirror plots and settings.
The panel applies the same dent to the first K of the three mirrors. All three
share the 5-degree incidence and complex reflectance. The folded path is
unfolded for scalar Fresnel propagation, with registered projected pupils.
Object images use a local shift-invariant approximation. No tables are shown
in the three-mirror panel. The numerical implementation and checks are in
mirror_train_optics.py and verify_mirror_train.py.

The three-mirror addition occupies ONE code cell, C09. Its controls and plots
share one output area. Nominal segments are planar and followed by one common
ideal focus; projected circular footprints and registered local dents are
small-angle assumptions. Fresnel propagation reference:
https://ocw.mit.edu/courses/2-71-optics-spring-2009/resources/mit2_71s09_lec18/

LIVE PLOT UPDATES AND DENT POSITION
C07 and C09 use persistent Image widgets rather than captured Matplotlib
display calls. Compute replaces each plot once; rerunning a cell disconnects
and closes its old panel. C09 still occupies exactly one notebook cell.
Shared centre sliders u=x/a and v=y/a move the local Gaussian on every dented
mirror in the same registered coordinates; keep u^2+v^2 <= 1. The global bow
remains centred. |r| is field reflection magnitude, so per-mirror power
reflectance is |r|^2. At |r|=.95 this is 90.25%; three bounces retain 73.509%
before clipping. Regression checks are in verify_render_updates.py.
