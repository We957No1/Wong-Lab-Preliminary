from pathlib import Path
import pymupdf
from PIL import Image,ImageDraw
import json
base=Path(__file__).resolve().parents[1]
qa=base/'checks/pdf_pages';qa.mkdir(exist_ok=True)
d=pymupdf.open(base/'study_notes_20_oct.pdf')
records=[]
for i,p in enumerate(d):
    pix=p.get_pixmap(matrix=pymupdf.Matrix(1.15,1.15),alpha=False)
    pix.save(qa/f'page_{i+1:02}.png')
    text=p.get_text()
    out=[]
    for b in p.get_text('dict')['blocks']:
        for l in b.get('lines',[]):
            for s in l['spans']:
                r=pymupdf.Rect(s['bbox'])
                if r.x0<15 or r.x1>p.rect.width-15 or r.y0<15 or r.y1>p.rect.height-15:out.append(s['text'])
    records.append({'page':i+1,'characters':len(text),'out_of_page_margin_spans':out})
for offset in range(0,len(d),9):
    sheet=Image.new('RGB',(900,1290),'#dbe3e8');draw=ImageDraw.Draw(sheet)
    for j in range(9):
        idx=offset+j
        if idx>=len(d):break
        im=Image.open(qa/f'page_{idx+1:02}.png');im.thumbnail((282,400))
        x=(j%3)*300+(300-im.width)//2;y=(j//3)*430+24
        sheet.paste(im,(x,y));draw.text((x,y-18),f'PAGE {idx+1}',fill='black')
    sheet.save(qa/f'contact_{offset+1:02}.jpg',quality=92)
(base/'checks/pdf_layout_checks.json').write_text(json.dumps({'pages':len(d),'records':records},indent=2),encoding='utf-8')
print('Pages:',len(d));print('Margin flags:',sum(len(x['out_of_page_margin_spans']) for x in records));print('Page character counts:',[x['characters'] for x in records])
