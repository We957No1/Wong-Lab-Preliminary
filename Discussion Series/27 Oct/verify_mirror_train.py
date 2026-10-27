"""Independent invariants and numerical checks for the three-reflection model."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import mirror_dent_optics as common
import mirror_train_optics as train


def verify_train():
    report={}
    rng=np.random.default_rng(2703)
    field=rng.normal(size=(64,64))+1j*rng.normal(size=(64,64))
    H=train.fresnel_transfer(field.shape,.2,550.,200.)
    propagated=train.fresnel_step(field,H)
    report['fresnel_relative_power_error']=float(abs(np.sum(abs(propagated)**2)/np.sum(abs(field)**2)-1))
    report['fresnel_forward_backward_max_error']=float(np.max(abs(train.fresnel_step(propagated,H.conj())-field)))
    assert report['fresnel_relative_power_error']<1e-12
    assert report['fresnel_forward_backward_max_error']<1e-12
    x=(np.arange(64)-32)*.2;kx,ky=3,-2
    plane=np.exp(2j*np.pi*(kx*x[None,:]+ky*x[:,None])/(64*.2))
    expected=np.exp(-1j*np.pi*550e-6*200*(kx*kx+ky*ky)/(64*.2)**2)*plane
    report['fresnel_plane_wave_max_error']=float(np.max(abs(train.fresnel_step(plane,H)-expected)))
    assert report['fresnel_plane_wave_max_error']<1e-12
    p=common.Parameters(n_pupil=128,fft_pad=2,incidence_deg=5.)
    config=train.TrainParameters(3,200.)
    case=train.three_mirror_study(p,config)
    u,v,support,A,du=common.cartesian_pupil(p)
    W=common.true_wavefront_nm(p,u,v)
    r=p.reflection_magnitude*np.exp(1j*np.deg2rad(p.reflection_phase_deg))
    coincident=train.three_mirror_study(p,replace(config,normal_spacing_mm=0))
    errors=[]
    for k in range(4):
        s,expected=common.propagate_psf(k*W,A,p,r**3)
        errors.append(float(np.max(abs(expected-coincident['count_psfs'][k]))))
    report['image_coordinate_vs_single_mirror_error']=float(np.max(abs(s-coincident['s'])))
    assert report['image_coordinate_vs_single_mirror_error']<1e-12
    report['zero_gap_vs_added_opd_max_error']=max(errors)
    assert max(errors)<1e-11
    shifted_p=replace(p,center_u=.30,center_v=-.20)
    shifted=train.three_mirror_study(shifted_p,replace(config,normal_spacing_mm=0))
    shifted_W=common.true_wavefront_nm(shifted_p,u,v)
    shifted_errors=[]
    for k in range(4):
        _,expected=common.propagate_psf(k*shifted_W,A,shifted_p,r**3)
        shifted_errors.append(float(np.max(abs(expected-shifted['count_psfs'][k]))))
    report['shared_off_axis_dent_zero_gap_error']=max(shifted_errors)
    assert max(shifted_errors)<1e-11
    angular=[abs(c) for (n,m),c in zip(shifted['fit']['modes'],shifted['fit']['coeff_nm']) if m!=0]
    report['off_axis_angular_coefficient_max_nm']=float(max(angular))
    assert max(angular)>.1
    report['zero_gap_power_vs_abs_r6_error']=float(abs(coincident['powers']['true']-abs(r)**6))
    assert report['zero_gap_power_vs_abs_r6_error']<1e-12
    report['aperture_power_upper_bound']=float(abs(r)**6)
    report['powers_by_dent_count']=case['count_powers']
    assert all(0<=power<=abs(r)**6+1e-12 for power in case['count_powers'].values())
    side=p.n_pupil*p.fft_pad
    power_from_fft=case['true_psf'].sum()*A.sum()**2/(side**2*np.sum(A*A))
    report['focal_vs_pupil_power_error']=float(abs(power_from_fft-case['powers']['true']))
    assert report['focal_vs_pupil_power_error']<1e-12
    no_dent=train.three_mirror_study(replace(p,depth_nm=0),config)
    report['zero_depth_all_counts_psf_error']=float(max(np.max(abs(no_dent['count_psfs'][k]-no_dent['ideal_psf'])) for k in range(4)))
    assert report['zero_depth_all_counts_psf_error']<1e-12
    selected_zero=train.three_mirror_study(p,replace(config,dented_mirrors=0))
    report['zero_selected_reconstruction_error']=selected_zero['relative_psf_error']
    assert report['zero_selected_reconstruction_error']<1e-12
    ref_phase=train.three_mirror_study(replace(p,reflection_phase_deg=-30.),config)
    report['common_reflection_phase_intensity_error']=float(np.max(abs(ref_phase['true_psf']-case['true_psf'])))
    assert report['common_reflection_phase_intensity_error']<1e-12
    unit=train.three_mirror_study(replace(p,reflection_magnitude=1.),config)
    half=train.three_mirror_study(replace(p,reflection_magnitude=.5),config)
    report['three_reflectance_factors_error']=float(np.max(abs(half['true_psf']-.5**6*unit['true_psf'])))
    assert report['three_reflectance_factors_error']<1e-12
    bow=train.three_mirror_study(replace(p,global_curvature=1.),config)
    report['n8_exact_global_bow_psf_error']=bow['relative_psf_error']
    assert report['n8_exact_global_bow_psf_error']<1e-10
    report['finite_gap_changes_ideal_psf']=float(np.max(abs(case['ideal_psf']-coincident['ideal_psf'])))
    assert report['finite_gap_changes_ideal_psf']>1e-6
    refinement=[]
    for n,pad in [(128,2),(128,4),(256,4)]:
        q=replace(p,n_pupil=n,fft_pad=pad)
        result=train.three_mirror_study(q,config)
        mid=len(result['s'])//2
        refinement.append({'pupil_pixels':n,'padding':pad,'on_axis':float(result['true_psf'][mid,mid]),
                           'throughput':result['powers']['true']})
    report['grid_refinement']=refinement
    report['padding_on_axis_difference']=abs(refinement[0]['on_axis']-refinement[1]['on_axis'])
    report['grid_on_axis_difference']=abs(refinement[1]['on_axis']-refinement[2]['on_axis'])
    assert report['padding_on_axis_difference']<.005
    assert report['grid_on_axis_difference']<.01
    report['status']='PASS'
    return report


def verify_train_widgets(render=False):
    p=common.Parameters(n_pupil=128,fft_pad=2,incidence_deg=5.)
    dashboard=train.make_train_dashboard(p,render=render)
    for k in range(4):
        dashboard['controls']['dented_mirrors'].value=k
        dashboard['controls']['depth_nm'].value=90.
        dashboard['controls']['normal_spacing_mm'].value=250.
        dashboard['controls']['center_u'].value=.30
        dashboard['controls']['center_v'].value=-.20
        dashboard['button'].click()
        assert dashboard['state']['error'] is None,dashboard['state']['error']
        case=dashboard['state']['study']
        assert case['train'].dented_mirrors==k and case['train'].normal_spacing_mm==250.
        assert case['p'].depth_nm==90.
        assert case['p'].center_u==.30 and case['p'].center_v==-.20
        assert dashboard['state']['compute_count']==k+1
        if render:
            assert len(dashboard['state']['figures'])==3
            assert len(dashboard['plot_images'])==3
            assert all(bytes(im.value).startswith(b'\x89PNG') for im in dashboard['plot_images'])
    common.close_dashboard(dashboard)
    return {'status':'PASS','dented_counts_tested':[0,1,2,3],
            'shared_depth_nm':90.,'shared_centre':[.30,-.20],'gap_mm':250.,'rendered':render,
            'scope':'Python widget state and callbacks; frontend not automated.'}


if __name__=='__main__':
    result={'optics':verify_train(),'widgets':verify_train_widgets()}
    Path('checks').mkdir(exist_ok=True)
    Path('checks/three_mirror_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
