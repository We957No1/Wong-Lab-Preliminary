"""Render the complete book for visual QA and inspect basic PDF integrity."""
from pathlib import Path
import re, json
import pymupdf as fitz
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/".work"/"qa"
QA.mkdir(parents=True,exist_ok=True)
doc=fitz.open(ROOT/"aberration_budget_textbook.pdf")
texts=[]
issues=[]
pages=[]
for i,page in enumerate(doc):
    text=page.get_text()
    texts.append(text)
    if not text.strip():
        issues.append({"page":i+1,"issue":"empty text page"})
    if "\ufffd" in text:
        issues.append({"page":i+1,"issue":"replacement character"})
    for word in page.get_text("words"):
        x0,y0,x1,y1,*_=word
        if x0 < -1 or y0 < -1 or x1>page.rect.width+1 or y1>page.rect.height+1:
            issues.append({"page":i+1,"issue":"word outside page","word":word})
    pix=page.get_pixmap(matrix=fitz.Matrix(.65,.65),alpha=False)
    path=QA/f"page_{i+1:03}.png"
    pix.save(path)
    pages.append(path)
for start in range(0,len(pages),16):
    group=pages[start:start+16]
    sheet=Image.new("RGB",(1600,2312),"#d9dfe3")
    draw=ImageDraw.Draw(sheet)
    for j,path in enumerate(group):
        im=Image.open(path)
        im.thumbnail((390,545))
        x=(j%4)*400+(400-im.width)//2
        y=(j//4)*578+25
        sheet.paste(im,(x,y))
        draw.text(((j%4)*400+10,(j//4)*578+6),f"PDF page {start+j+1}",fill="black")
    sheet.save(QA/f"contact_{start+1:03}_{start+len(group):03}.png")
for number in (1,22,35,41,61,72,82,84,85,92,93):
    if number <= len(doc):
        doc[number-1].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(
            QA/f"detail_{number:03}.png")
(QA/"extracted_text.txt").write_text("\n".join(f"\n=== PDF PAGE {i+1} ===\n{t}" for i,t in enumerate(texts)),encoding="utf-8")
texfiles=[ROOT/"aberration_budget_textbook.tex",ROOT/"bibliography.tex",*sorted((ROOT/"chapters").glob("*.tex"))]
source="\n".join(p.read_text(encoding="utf-8") for p in texfiles)
labels=re.findall(r"\\label\{([^}]+)\}",source)
refs=re.findall(r"\\(?:eqref|ref)\{([^}]+)\}",source)
cites=[item.strip() for group in re.findall(r"\\cite\{([^}]+)\}",source) for item in group.split(",")]
bibs=re.findall(r"\\bibitem\{([^}]+)\}",source)
log=(ROOT/"aberration_budget_textbook.log").read_text(encoding="utf-8",errors="replace")
build_warnings=[line for line in log.splitlines()
                if any(token in line for token in
                       ("Overfull", "Underfull", "Warning:", "Missing character:", "! "))]
result={"pdf_pages":len(doc),"main_chapters":len(re.findall(r"\\chapter\{",source))-3,
        "appendices":3,"source_word_tokens":len(source.split()),
        "pdf_text_words":sum(len(t.split()) for t in texts),
        "missing_labels":sorted(set(refs)-set(labels)),
        "duplicate_labels":sorted({x for x in labels if labels.count(x)>1}),
        "missing_citations":sorted(set(cites)-set(bibs)),
        "page_issues":issues,"latex_warnings":build_warnings,
        "figures":len(list((ROOT/"figures").glob("*.pdf"))),
        "contact_sheets":[p.name for p in sorted(QA.glob("contact_*.png"))]}
(QA/"inspection_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
