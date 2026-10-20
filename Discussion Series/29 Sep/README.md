# Study notes for 29 September

Open `optics_study_notes_29_sep.pdf` to read, or edit
`optics_study_notes_29_sep.tex`.

The note develops Maxwell's equations, intensity normalization, angular-spectrum
and Fresnel propagation, focusing and the PSF, coherence, the OTF overlap
derivation, the sinusoidal-mask example, and their role in EUV optical design.

Keep the `figures` folder beside the `.tex` file. To compile, run:

```text
latexmk -pdf optics_study_notes_29_sep.tex
```

Alternatively, run pdfLaTeX twice or use Tectonic. The bibliography is embedded;
BibTeX and shell escape are unnecessary. Upload the `.tex` file and the four
figure PDFs together when using Overleaf.

`tools/make_figures.py` regenerates the Airy and pupil-overlap figures and
independently checks the sinusoidal result by numerical integration over the
illumination directions. It requires NumPy, SciPy, and Matplotlib.
`numerical_checks.json` records the values and the source-integration residual.
The supplied `figures/coherence_comparison.pdf` is ready to use.

`tools/make_pupil_example.py` regenerates the pupil-focused ray diagrams and checks
the pupil images, source images, and clear-aperture constraints with paraxial
ray matrices. Its numerical results are in `pupil_example_checks.json`.

Temporary build and layout-review files are kept in `.work` and are unnecessary
for compilation.
