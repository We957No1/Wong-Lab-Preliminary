from pathlib import Path
import json
import re

import pymupdf as fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / ".work" / "qa"
QA.mkdir(exist_ok=True)
name = "optics_study_notes_29_sep"
doc = fitz.open(ROOT / f"{name}.pdf")
if doc.page_count == 0:
    raise RuntimeError("PDF has no readable pages; an editor build may still be writing it.")
issues = []
pages = []
texts = []
for i, page in enumerate(doc):
    text = page.get_text()
    texts.append(text)
    if not text.strip():
        issues.append({"page": i+1, "problem": "empty text page"})
    if "\ufffd" in text:
        issues.append({"page": i+1, "problem": "replacement character"})
    for w in page.get_text("words"):
        x0, y0, x1, y1, word, *_ = w
        if x0 < 0 or y0 < 0 or x1 > page.rect.width+1 or y1 > page.rect.height+1:
            issues.append({"page": i+1, "problem": "text outside page", "word": word})
    path = QA / f"page_{i+1:02d}.png"
    page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False).save(path)
    pages.append(path)
    body = text.splitlines()[2:]
    print(f"Page {i+1}: {len(text.split())} extracted words; begins: {' '.join(body[:3])[:160]}")

for start in range(0, len(pages), 4):
    group = pages[start:start+4]
    sheet = Image.new("RGB", (1700, 2450), "#e1e5e8")
    draw = ImageDraw.Draw(sheet)
    for j, path in enumerate(group):
        pic = Image.open(path)
        pic.thumbnail((825, 1175))
        x = (j % 2)*850+(850-pic.width)//2
        y = (j // 2)*1225+30
        sheet.paste(pic, (x,y))
        draw.text(((j % 2)*850+15, (j // 2)*1225+8), f"Page {start+j+1}", fill="black")
    sheet.save(QA / f"contact_{start+1:02d}_{start+len(group):02d}.png")

tex = (ROOT / f"{name}.tex").read_text(encoding="utf-8")
labels = re.findall(r"\\label\{([^}]+)\}", tex)
refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex)
cites = {x.strip() for g in re.findall(r"\\cite\{([^}]+)\}", tex) for x in g.split(",")}
bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
log = (ROOT / f"{name}.log").read_text(encoding="utf-8", errors="replace")
warnings = [x for x in log.splitlines() if any(t in x for t in
            ("Overfull", "Underfull", "Warning:", "Missing character:", "! "))]
result = {"pages": len(doc), "missing_references": sorted(set(refs)-set(labels)),
          "missing_citations": sorted(cites-bibs),
          "duplicate_labels": sorted({x for x in labels if labels.count(x)>1}),
          "build_warnings": warnings, "layout_issues": issues,
          "pdf_text_word_count": sum(len(t.split()) for t in texts),
          "textbook_mentions": re.findall(r"(?i)textbook|section\s+4\.4|chapter\s+5", tex)}
(QA / "verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
(QA / "extracted_text.txt").write_text("\n\n".join(f"PAGE {i+1}\n{t}" for i,t in enumerate(texts)), encoding="utf-8")
print(json.dumps(result, indent=2))
