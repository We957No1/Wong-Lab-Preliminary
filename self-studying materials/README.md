# Self-study optics textbooks

Start with either book's Part I if you have not studied optics before. The foundations are repeated so each textbook can be read independently. Then study the Zernike book for wavefront decomposition and metrology, and the aberrations book for field dependence, aberration orders, and design calculations.

The existing research-folder notes were used only to identify relevant interests. These books develop their own derivations and worked examples, with university and primary-source references in their bibliographies.

## Textbooks

| Topic | Main LaTeX file | Compiled reading copy |
| --- | --- | --- |
| Zernike coefficients for mirrors and lenses | `Zernike Constants/zernike_constants_textbook.tex` | `Zernike Constants/zernike_constants_textbook.pdf` |
| Optical aberrations and their orders | `abberations/aberrations_textbook.tex` | `abberations/aberrations_textbook.pdf` |

The folder spelling `abberations` follows the requested name. The textbook uses the standard spelling “aberrations.”

The Zernike textbook has 75 pages and 18 numbered chapters; the aberrations textbook has 73 pages and 17 numbered chapters. Each includes the beginner foundations, detailed derivations, worked numerical examples, exercises with solutions, and a bibliography.

Each main `.tex` file includes `notation.tex`, `foundations.tex`, and its subject-specific `*_chapters.tex`. Compile the main file, not a chapter fragment. Keep the `figures` folder beside the two book folders; the subject chapters reference its vector PDF illustrations.

## Build with a standard LaTeX installation

From the directory containing the corresponding main file:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error zernike_constants_textbook.tex
pdflatex -interaction=nonstopmode -halt-on-error zernike_constants_textbook.tex
```

For the second book, replace the filename with `aberrations_textbook.tex`. A third pass may be needed if the contents or cross-references change. `latexmk -pdf` can manage these passes automatically. Both books use embedded `thebibliography` entries and need no BibTeX run. Their diagrams use TikZ and the standard packages declared in the main files.

The existing local Tectonic executable also builds them. From the research-folder root:

```powershell
$env:TECTONIC_CACHE_DIR = Join-Path (Get-Location) 'Ray Optics/.tectonic-cache'
& '.\Ray Optics\.tools\tectonic\tectonic.exe' 'self-studying materials/Zernike Constants/zernike_constants_textbook.tex' --keep-logs
& '.\Ray Optics\.tools\tectonic\tectonic.exe' 'self-studying materials/abberations/aberrations_textbook.tex' --keep-logs
```

Tectonic downloads standard LaTeX dependencies when they are not already cached. The sources use no machine-specific fonts or absolute paths to figures.

## Reproduce the worked calculations

`verify_optics.py` performs independent symbolic and numerical checks and saves `numerical_checks.txt`. It requires Python, NumPy, and SymPy; Matplotlib additionally regenerates the three illustrations. Using the existing workspace Python environment, run from the research-folder root:

```powershell
& '.\Ray Optics\.venv\Scripts\python.exe' '.\self-studying materials\verify_optics.py'
```

The checks cover full-disk Zernike normalization, modal recovery on a masked pupil, quartic and sixth-degree balancing, exact spherical reflection, pupil-coordinate changes, a thin phase-screen lens, and an exact finite-thickness biconvex singlet. Prescriptions and units are stated in the script and report. These are reproducible teaching calculations, not measurements from a commercial instrument.

When adapting the examples, record the physical pupil radius, coordinate surface, wavelength, OPD sign, modal normalization, reference focus, and removed modes. The books explain why every one of these choices changes the interpretation of an exported coefficient table.

## Verification of this edition

Both complete books compiled successfully with Tectonic 0.17.0. Their final logs contain no unresolved references or citations, missing characters, or overfull/underfull layout boxes. All pages were inspected in rendered contact sheets and selected dense pages at full resolution. The mathematical derivations and solved numerical exercises were independently reviewed, and the verification script and embedded Zernike laboratory passed their assertions. The numerical checks validate the stated examples and conventions; they are not a certification of arbitrary future changes to the models.
