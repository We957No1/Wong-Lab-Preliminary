$ErrorActionPreference = 'Stop'
$taskFolder = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = (Resolve-Path -LiteralPath (Join-Path $taskFolder '..\..')).Path
$pythonExe = Join-Path $workspace 'Ray Optics\.venv\Scripts\python.exe'
$texExe = Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64\pdflatex.exe'
Push-Location -LiteralPath $taskFolder
try {
    & $pythonExe -X utf8 verify_calculations.py
    if ($LASTEXITCODE -ne 0) { throw 'Scientific checks failed.' }
    & $pythonExe -X utf8 generate_figures.py
    if ($LASTEXITCODE -ne 0) { throw 'Figure generation failed.' }
    foreach ($pass in 1..3) {
        & $texExe -no-shell-escape -interaction=nonstopmode -halt-on-error -synctex=1 study_notes_20_oct.tex
        if ($LASTEXITCODE -ne 0) { throw 'PDF build failed.' }
    }
    if ($LASTEXITCODE -ne 0) { throw 'PDF build failed.' }
} finally { Pop-Location }
