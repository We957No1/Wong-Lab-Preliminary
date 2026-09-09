"""Inspect compiled textbook pages and prepare visual-review contact sheets."""
from pathlib import Path
import json
import re
import pymupdf
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[1]
BOOKS = {
    "zernike": (BASE / "Zernike Constants", "zernike_constants_textbook", "zernike_chapters.tex"),
    "aberrations": (BASE / "abberations", "aberrations_textbook", "aberrations_chapters.tex"),
}
report = {}
for key, (folder, stem, core) in BOOKS.items():
    pdf = folder / (stem + ".pdf")
    if not pdf.exists():
        continue
    target = BASE / ".qa" / key
    target.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    source = "\n".join((folder / f).read_text(encoding="utf-8-sig")
                       for f in [stem + ".tex", "notation.tex", "foundations.tex", core])
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    refs = re.findall(r"\\(?:eqref|ref|nameref)\{([^}]+)\}", source)
    citations = [c.strip() for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", source)
                 for c in group.split(",")]
    bib = re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", source)
    page_text = [p.get_text() for p in doc]
    outside = []
    for i, page in enumerate(doc):
        for word in page.get_text("words"):
            x0, y0, x1, y1, text, *_ = word
            if x0 < 24 or x1 > page.rect.width - 24 or y0 < 12 or y1 > page.rect.height - 12:
                outside.append({"page": i + 1, "text": text, "box": [x0, y0, x1, y1]})
    thumbs = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(0.62, 0.62), alpha=False)
        im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        tile = Image.new("RGB", (390, 570), "#e4e6e8")
        im.thumbnail((374, 535))
        tile.paste(im, ((390-im.width)//2, 24))
        ImageDraw.Draw(tile).text((12, 6), f"{key} PDF page {i+1}", fill="black")
        thumbs.append(tile)
    sheets = []
    for start in range(0, len(thumbs), 12):
        sheet = Image.new("RGB", (1170, 2280), "white")
        for offset, im in enumerate(thumbs[start:start+12]):
            sheet.paste(im, ((offset % 3)*390, (offset//3)*570))
        filename = target / f"contact_{start+1:03}_{min(start+12,len(doc)):03}.png"
        sheet.save(filename)
        sheets.append(str(filename))
    report[key] = {
        "pdf_pages": len(doc),
        "chapters": len(re.findall(r"\\chapter\{", source)),
        "source_whitespace_words": len(source.split()),
        "duplicate_labels": sorted({x for x in labels if labels.count(x)>1}),
        "unresolved_source_references": sorted(set(refs)-set(labels)),
        "unresolved_source_citations": sorted(set(citations)-set(bib)),
        "replacement_glyph_pages": [i+1 for i,t in enumerate(page_text) if "\ufffd" in t],
        "page_edge_text": outside,
        "very_sparse_pages": [i+1 for i,t in enumerate(page_text) if len(t.strip())<100],
        "contact_sheets": sheets,
    }
    print(key, json.dumps({k:v for k,v in report[key].items() if k!="contact_sheets"}, indent=2))
(BASE / ".qa" / "inspection_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
