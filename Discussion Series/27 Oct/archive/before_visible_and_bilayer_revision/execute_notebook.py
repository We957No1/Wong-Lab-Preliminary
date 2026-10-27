"""Execute the teaching notebook from a clean Python kernel; retain all results."""
from pathlib import Path
import sys, os, json, time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

base=Path(__file__).resolve().parent
os.environ['MPLBACKEND']='Agg'
path=base/'mirror_dent_zernike.ipynb'
nb=nbformat.read(path,as_version=4)
nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type=='code':compile(cell.source,cell.get('id','cell'),'exec')
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(base)}})
client.km=manager
start=time.time()
try:
    client.execute()
finally:
    nbformat.write(nb,path)
errors=[{'cell':i,'ename':o.get('ename'),'evalue':o.get('evalue')}
        for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if o.output_type=='error']
report={'code_cells':sum(c.cell_type=='code' for c in nb.cells),'errors':errors,
        'seconds':time.time()-start,'python':sys.executable,
        'executions':[c.execution_count for c in nb.cells if c.cell_type=='code']}
(base/'checks/notebook_execution.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
if errors:raise RuntimeError('Notebook has execution errors.')
