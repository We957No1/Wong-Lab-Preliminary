"""Regression checks: repeated clicks replace image payloads, without displays."""
import hashlib
import matplotlib.pyplot as plt
from IPython.utils.capture import capture_output
import mirror_dent_optics as common
import mirror_train_optics as train


def verify():
    report={}
    p=common.Parameters(n_pupil=128,fft_pad=2,incidence_deg=5.)
    panels=[('single_mirror',common.make_dashboard(p),4),
            ('three_mirrors',train.make_train_dashboard(p),3)]
    for name,panel,expected in panels:
        initial_ids=tuple(im.model_id for im in panel['plot_images'])
        figures_before=set(plt.get_fignums())
        payloads=[]
        with capture_output(display=True) as captured:
            for j,(u,v) in enumerate([(0.,0.),(.20,-.10),(.35,.15)],start=1):
                panel['controls']['center_u'].value=u
                panel['controls']['center_v'].value=v
                panel['button'].click()
                assert panel['state']['error'] is None,panel['state']['error']
                assert panel['state']['compute_count']==j
                assert panel['state']['render_count']==j
                assert tuple(im.model_id for im in panel['plot_images'])==initial_ids
                assert len(panel['plot_images'])==expected
                hashes=[]
                for image in panel['plot_images']:
                    data=bytes(image.value)
                    assert data.startswith(b'\x89PNG\r\n\x1a\n')
                    hashes.append(hashlib.sha256(data).hexdigest())
                assert len(set(hashes))==expected
                payloads.append(tuple(hashes))
        assert not captured.outputs,'A widget update also emitted inline display output.'
        assert len(set(payloads))==3,'Position changes did not replace plot contents.'
        assert set(plt.get_fignums())==figures_before,'An unclosed figure can be auto-displayed.'
        assert len(panel['button']._click_handlers.callbacks)==1
        report[name]={'status':'PASS','compute_clicks':3,'persistent_image_slots':expected,
                      'unchanged_widget_ids':True,'inline_display_events':0,
                      'distinct_position_updates':3,'unclosed_figures':0}
        common.close_dashboard(panel)
        assert not panel['button']._click_handlers.callbacks
        assert panel['ui'].comm is None
    report['status']='PASS'
    report['scope']='Kernel display capture, widget payload replacement and callback cleanup; frontend not automated.'
    return report
