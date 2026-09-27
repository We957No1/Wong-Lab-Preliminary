"""Extract local source text and render reading copies for source verification."""
from pathlib import Path
import json
import fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'aberration budget paper' / '.work' / 'sources'
OUT.mkdir(parents=True, exist_ok=True)
sources = {
    'paper': 'Abberation Budget paper.pdf',
    'born_wolf': 'Born & Wolf Principles of Optics.pdf',
    'saleh_teich': 'Saleh & Teich Foundations of Photonics.pdf',
}
metadata = {}
for key, filename in sources.items():
    doc = fitz.open(ROOT / filename)
    pages = [page.get_text(sort=True) for page in doc]
    (OUT / f'{key}.txt').write_text('\n'.join(
        f'\n===== PDF PAGE {i+1} =====\n{text}' for i, text in enumerate(pages)),
        encoding='utf-8')
    (OUT / f'{key}_pages.json').write_text(json.dumps(pages, ensure_ascii=False), encoding='utf-8')
    metadata[key] = {'file': filename, 'pages': len(doc), 'metadata': doc.metadata,
                     'toc': doc.get_toc(), 'text_chars': sum(map(len, pages))}
    if key == 'paper':
        render = OUT / 'paper_pages'
        render.mkdir(exist_ok=True)
        for i, page in enumerate(doc):
            page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False).save(render / f'page_{i+1:02}.png')
(OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
for key, data in metadata.items():
    print(key, 'pages=', data['pages'], 'text_chars=', data['text_chars'])
    print('metadata:', data['metadata'])
    print('toc:', data['toc'][:35])
