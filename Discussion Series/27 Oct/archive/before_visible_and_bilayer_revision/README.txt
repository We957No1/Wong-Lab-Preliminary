27 OCT — GAUSSIAN MIRROR DENT / ZERNIKE RECONSTRUCTION

START
Open mirror_dent_zernike.ipynb in Jupyter or VS Code and choose the existing
optics Python kernel:
C:\Users\fxl04\Desktop\Prof. Wong Grp Papers\Ray Optics\.venv\Scripts\python.exe
Run all cells in order. Saved output is included for viewing without a kernel.
Interactive controls need a running kernel and frontend widget support.

EDIT
C02: all baseline physical, defect, pupil, fitting, sampling and object parameters.
The preceding table defines every parameter and its units. Then rerun C07 onward.
C15: live controls. Change sliders, then click Compute selected case. Other settings
come from C02. The baseline is retained. Invalid combinations display a message.
C16: export baseline results. To save an interactive case after successful computation:
export_study(DASHBOARD['state']['study'], directory=PROJECT_DIR/'interactive_results')

MODEL
IDEAL means the SAME system with the dent removed: wavelength, pupil, illumination
and chosen added defocus are identical. The input object is separate from its
diffraction-limited image. Intensity units use a common dent-free IN-FOCUS peak;
a selected out-of-focus dent-free reference can have a peak below one.

An ideal focusing mirror/relay has a Gaussian surface departure h. Positive height
points toward incident space; a dent of positive depth d has h<0. Reflected OPD is
W=-2*h*cos(i); at normal incidence the wavefront departure is twice the surface
height in magnitude. The default 3 nm dent gives 6 nm peak OPD at 13.5 nm wavelength.

10% AREA means the region within the Gaussian 1/e DEPTH contour: radius sqrt(0.1)
times mirror radius for a centred circular dent. Tails extend beyond the contour.
A centred circular dent produces only even-n, m=0 modes in an ideal radial pupil;
it generally contains many radial/spherical orders, rather than one named mode.

The model is scalar, monochromatic, paraxial and uses a fixed mapped pupil. The
incidence angle changes the path factor only, not an oblique footprint or coating.
Point images use coherent diffraction. Extended objects are spatially incoherent
intensity distributions convolved with a unit-energy PSF. Every PSF uses one
matching ideal-peak normalization; true/reconstructed PSFs are never individually
normalized to their peaks. There is no automatic refocus in the baseline.

FIT / ACCURACY
Zernike convention: full-disk unit mean square, m>0 cosine, m<0 sine. Coefficients
are OPD in nm and waves, labelled (n,m), not Noll/Fringe indices. Clear pupils use
independent polar quadrature projection; obscured pupils use area-weighted QR.
Amplitude apodization changes propagation separately from geometric-area fitting.
Noise is synthetic unwrapped OPD metrology noise, not photon noise. This is forward
modelling from a known wavefront, not unique phase retrieval from one intensity image.

C12 compares successive orders; C13 separately demonstrates RMS refocus. C14 checks
independent physical/numerical invariants and image-grid refinement. Increasing
pupil resolution, coefficient quadrature, and FFT padding test different errors.

FILES
results/: 8 figures (PNG/PDF), all coefficients, metrics, order sweep, cropped PSF
and full pupil arrays, exact parameters, dependency versions and SHA-256 hashes.
checks/: execution/validation records and initial construction data.
mirror_dent_optics.py: script-use copy of the notebook's starting helper functions.
verify_mirror_dent.py: independent command-line scientific checks using that module.
execute_notebook.py: clean-kernel execution preserving notebook output.
The notebook includes its own helper definitions and validates those definitions
in C14. If editing functions, edit notebook cells; update the module too only if
you also want the command-line copy to follow those changes.

RERUN FROM POWERSHELL (existing optics environment)
& '..\..\Ray Optics\.venv\Scripts\python.exe' -X utf8 execute_notebook.py
& '..\..\Ray Optics\.venv\Scripts\python.exe' -X utf8 verify_mirror_dent.py

For another Python environment, install requirements.txt and select that kernel.
Use a Python version compatible with the pinned packages (validated environment
is recorded in results/provenance.json). Notebook sources include Chapter 7 of the
local textbook and the primary reference/documentation links used.
