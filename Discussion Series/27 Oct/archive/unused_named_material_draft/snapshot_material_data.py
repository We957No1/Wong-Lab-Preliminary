from pathlib import Path
from urllib.request import Request,urlopen
from datetime import datetime,timezone
import hashlib,json
base=Path(__file__).resolve().parents[1]/'material_data'
base.mkdir(exist_ok=True)
rows=[]
for symbol in ('mo','si','o'):
    url=f'https://henke.lbl.gov/optical_constants/sf/{symbol}.nff'
    data=urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30).read()
    target=base/(symbol+'.nff');target.write_bytes(data)
    rows.append({'symbol':symbol,'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
(base/'sources.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))
