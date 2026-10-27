"""Clean-kernel execution for one or all of the 27 Oct optics notebooks."""
from pathlib import Path
import sys,os,json,time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

base=Path(__file__).resolve().parent
os.environ['MPLBACKEND']='Agg'
files=sys.argv[1:] or ['mirror_dent_zernike.ipynb','euv_bilayer_zernike.ipynb','optics_validation.ipynb']
(base/'checks').mkdir(exist_ok=True)
reports=[]
for filename in files:
    path=base/filename
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,cell.get('id','cell'),'exec')
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(base)}})
    client.km=manager;start=time.time()
    try:client.execute()
    finally:nbformat.write(nb,path)
    errors=[{'cell':i,'ename':o.get('ename'),'message':o.get('evalue')} for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if o.output_type=='error']
    report={'file':filename,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'errors':errors,
            'seconds':time.time()-start,'python':sys.executable,
            'embedded_png_figures':sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[])),
            'executions':[c.execution_count for c in nb.cells if c.cell_type=='code']}
    (base/'checks'/f'{path.stem}_execution.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    reports.append(report);print(json.dumps(report,indent=2),flush=True)
    if errors:raise RuntimeError(f'{filename} has errors.')
# Retain reports for other notebooks when only one file was rerun.
all_reports=[]
for name in ['mirror_dent_zernike','euv_bilayer_zernike','optics_validation']:
    report_file=base/'checks'/f'{name}_execution.json'
    if report_file.exists():all_reports.append(json.loads(report_file.read_text(encoding='utf-8')))
(base/'checks/notebook_execution.json').write_text(json.dumps(all_reports,indent=2),encoding='utf-8')
