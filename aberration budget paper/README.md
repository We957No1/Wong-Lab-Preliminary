# Aberration budget paper: self-study textbook

Open **aberration_budget_textbook.pdf** to read. The editable main source
is **aberration_budget_textbook.tex**. Keep the chapters and figures
folders and bibliography.tex beside it.

The book begins with waves and basic mathematics, derives the optical
and learning models, explains all 12 paper figures, all 4 tables, and all
3 numbered equations, and includes 25 solved exercises. It records the
paper's ambiguities instead of silently correcting them.

The compiled edition has 98 pages, 17 main chapters, 3 appendices, and
8 original scientific figures. Start with **How to study this book**,
then Chapters 1–3. Chapter 15 is the guide to keep beside the paper;
Appendix C contains the textbook reading map and symbol dictionary.

The reading map uses the exact supplied scans. The Saleh–Teich excerpt
ends at printed page 278. Part of the Born–Wolf scan runs in reverse page
order; the book provides verified PDF entry points.

## Compile

From this folder with a standard LaTeX installation:

    latexmk -pdf aberration_budget_textbook.tex

Or run this command three times:

    pdflatex -interaction=nonstopmode -halt-on-error aberration_budget_textbook.tex

The bibliography is embedded; BibTeX is not needed. Standard LaTeX
packages and local PDF figures are used. No shell escape or
machine-specific fonts are required. The whole folder can be uploaded
to Overleaf; the hidden .work folder is unnecessary for compilation.

With the existing workspace Tectonic, run from the research workspace root:

    $env:TECTONIC_CACHE_DIR = Join-Path (Get-Location) 'Ray Optics/.tectonic-cache'
    & '.\Ray Optics\.tools\tectonic\tectonic.exe' 'aberration budget paper/aberration_budget_textbook.tex' --keep-logs

## Reproduce the teaching calculations

With Python, NumPy, SciPy, and Matplotlib:

    python tools/reproduce_examples.py

Or use the existing workspace Python from the research workspace root:

    & '.\Ray Optics\.venv\Scripts\python.exe' '.\aberration budget paper\tools\reproduce_examples.py'

This regenerates the original scientific figures in PDF and PNG,
numerical_checks.txt, and teaching_model_results.json.

The contact example is scalar and uses a binary thin mask, periodic
boundaries, a declared quasar quadrature, and a fixed threshold. It is an
educational model, not a reproduction of the authors' commercial EUV
mask simulation. Their data, trained weights, and full settings were
not supplied.

Source-inspection and rendering intermediates are organized under .work.
The original source PDFs and the pre-existing study notes are unchanged.

## Verification

The final PDF was compiled with Tectonic 0.17 and visually reviewed using
full-document contact sheets and detailed page renders. The final LaTeX
log has no warnings, unresolved references, missing citations, or
overfull boxes. Automated inspection found no text outside page bounds.
The teaching calculation's assertions passed, including Zernike
orthogonality, balanced spherical aberration, the analytic tilt shift,
and preservation of registered feature widths under tilt.

These checks concern this textbook and its explicitly stated teaching
model. They do not independently reproduce or certify the paper's
unreleased commercial simulations or trained neural network.
