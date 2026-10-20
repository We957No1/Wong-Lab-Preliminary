from pathlib import Path
import sys,nbformat
from nbclient import NotebookClient
base=Path(__file__).resolve().parents[1]
p=base/'optics_demonstrations.ipynb'
nb=nbformat.read(p,as_version=4)
client=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(base)}})
client.km=client.create_kernel_manager()
client.km.kernel_spec.argv[0]=sys.executable
client.execute()
nbformat.write(nb,p)
print('Notebook executed:',sum(c.cell_type=='code' for c in nb.cells),'code cells; no errors.')
