from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mirror_dent_optics as m
from dataclasses import asdict
import json,io,contextlib
from pathlib import Path
ui=m.make_dashboard(m.Parameters(n_pupil=128,fft_pad=2),render=True)
values={'depth_nm':4.,'area_fraction':.12,'center_u':.2,'center_v':-.1,'aspect_ratio':1.4,'orientation_deg':30.,'added_defocus_nm':.1,'noise_rms_nm':.03,'n_max':6,'wavelength_nm':20.,'fit_family':'all'}
for name,value in values.items():ui['controls'][name].value=value
with contextlib.redirect_stdout(io.StringIO()):ui['button'].click()
assert ui['state']['error'] is None,ui['state']['error']
p=ui['state']['study']['p']
assert all(getattr(p,k)==v for k,v in values.items())
report={'status':'PASS','forwarded_controls':values,'render_callback':'All three comparison figures generated without error','frontend':'Python callback and widget state tested; no GUI browser automation'}
Path('checks/widget_render_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
