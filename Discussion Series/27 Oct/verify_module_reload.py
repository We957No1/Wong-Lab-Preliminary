"""Run in an isolated process: reproduce the user's stale-import error."""
from pathlib import Path
import importlib.util,sys,json,os
import nbformat


def verify():
    base=Path(__file__).resolve().parent
    os.chdir(base)
    old_path=base/'archive/before_n8_decomposition/mirror_dent_optics.py'
    spec=importlib.util.spec_from_file_location('mirror_dent_optics',old_path)
    cached=importlib.util.module_from_spec(spec)
    sys.modules['mirror_dent_optics']=cached
    spec.loader.exec_module(cached)
    assert not hasattr(cached,'decomposition_through_order')
    try:cached.decomposition_through_order
    except AttributeError as exc:reproduced=str(exc)
    nb=nbformat.read(base/'mirror_dent_zernike.ipynb',as_version=4)
    setup=next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith('# C01'))
    namespace={}
    exec(compile(setup,'C01','exec'),namespace)
    assert hasattr(namespace['optics'],'decomposition_through_order')
    assert Path(namespace['optics'].__file__).resolve()==(base/'mirror_dent_optics.py').resolve()
    delattr(namespace['optics'],'decomposition_through_order')
    decomp=next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith('# C04'))
    exec(decomp.split('DECOMPOSITION_N8=')[0],namespace)
    assert hasattr(namespace['optics'],'decomposition_through_order')
    return {'status':'PASS','reproduced':reproduced,
            'cause':'An older module remains in sys.modules in a running notebook.',
            'C01_reload_fixed':True,'C04_reload_fixed':True,
            'loaded_file':str(Path(namespace['optics'].__file__).resolve())}


if __name__=='__main__':
    result=verify()
    Path('checks/module_reload_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
