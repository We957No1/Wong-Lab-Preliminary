from pathlib import Path
import sys
import pymupdf as fitz
from PIL import Image, ImageOps, ImageDraw

root = Path(__file__).resolve().parents[2]
out = root / 'aberration budget paper' / '.work' / 'sources'
key = sys.argv[1]
names = {'bw':'Born & Wolf Principles of Optics.pdf', 'st':'Saleh & Teich Foundations of Photonics.pdf', 'paper':'Abberation Budget paper.pdf'}
pages=[]
for arg in sys.argv[2:]:
    if '-' in arg:
        a,b=map(int,arg.split('-')); pages.extend(range(a,b+1))
    else: pages.append(int(arg))
doc=fitz.open(root/names[key])
folder=out/key
folder.mkdir(exist_ok=True)
for num in pages:
    doc[num-1].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(folder/f'{num:04}.png')
for start in range(0,len(pages),9):
    group=pages[start:start+9]
    sheet=Image.new('RGB',(1260,1770),'#dddddd')
    draw=ImageDraw.Draw(sheet)
    for j,num in enumerate(group):
        im=Image.open(folder/f'{num:04}.png'); im.thumbnail((412,560))
        x=(j%3)*420+(420-im.width)//2; y=(j//3)*590+25
        sheet.paste(im,(x,y)); draw.text(((j%3)*420+10,(j//3)*590+5),f'{key} PDF {num}',fill='black')
    sheet.save(folder/f'contact_{group[0]:04}_{group[-1]:04}.png')
print(key, pages)
