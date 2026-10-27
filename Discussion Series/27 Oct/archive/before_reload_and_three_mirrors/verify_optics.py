"""Physical/numerical checks, kept out of the two demonstration notebooks."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import mirror_dent_optics as m
import euv_bilayer_optics as c
from verify_mirror_dent import verify_all as verify_visible_basics


def verify_visible():
    report=verify_visible_basics(include_widget=False)
    p=m.Parameters(n_pupil=128,reflection_magnitude=1.)
    b=m.build_study(replace(p,global_curvature=1.))
    f=b['fit'];coeff=dict(zip(f['modes'],f['coeff_nm']))
    report['global_bow_c00_error_nm']=abs(coeff[0,0]-p.depth_nm)
    report['global_bow_c20_error_nm']=abs(coeff[2,0]+p.depth_nm/np.sqrt(3))
    report['global_bow_residual_nm']=f['residual_rms_nm']
    assert max(report['global_bow_c00_error_nm'],report['global_bow_c20_error_nm'],report['global_bow_residual_nm'])<1e-9
    s1=m.build_study(replace(p,reflection_magnitude=.4,reflection_phase_deg=-40.))
    s2=m.build_study(replace(p,reflection_magnitude=.4,reflection_phase_deg=125.))
    full=m.build_study(p)
    report['uniform_reflection_phase_intensity_error']=float(np.max(abs(s1['true_psf']-s2['true_psf'])))
    report['reflectance_power_scaling_error']=float(np.max(abs(s1['true_psf']-.16*full['true_psf'])))
    assert report['uniform_reflection_phase_intensity_error']<1e-12
    assert report['reflectance_power_scaling_error']<1e-12
    report['throughput_error']=abs(s1['powers']['true']-.16);assert report['throughput_error']<1e-12
    conv=m.convergence_study(p,(0,2,4,6))
    fig=m.plot_convergence(conv,full)
    report['integer_order_ticks']=all(np.allclose(ax.get_xticks(),np.round(ax.get_xticks())) for ax in fig.axes)
    plt.close(fig);assert report['integer_order_ticks']
    broad=m.fit_wavefront(replace(p,area_fraction=100.))
    cc=dict(zip(broad['modes'],broad['coeff_nm']))
    report['broad_gaussian_spherical_to_defocus']=float(abs(cc[4,0]/cc[2,0]));assert report['broad_gaussian_spherical_to_defocus']<.01
    report['default_wavelength_nm']=p.wavelength_nm;assert 380<=p.wavelength_nm<=780
    decomposition=m.decomposition_through_order(full,8)
    modes=list(zip(decomposition['coefficients'].n,decomposition['coefficients'].m))
    report['n8_decomposition_mode_count']=len(modes)
    assert modes==m.mode_list(8) and len(modes)==45
    low_order=m.build_study(replace(p,n_max=2))
    dlow=m.decomposition_through_order(low_order,8)
    report['n8_independent_of_image_order_error_nm']=float(np.max(abs(
        dlow['coefficients'].coefficient_nm-decomposition['coefficients'].coefficient_nm)))
    assert report['n8_independent_of_image_order_error_nm']<1e-10
    high_masked=m.build_study(replace(p,n_max=12,obscuration_ratio=.25))
    masked_decomposition=m.decomposition_through_order(high_masked,8)
    masked_reference=m.fit_wavefront(replace(p,n_max=8,obscuration_ratio=.25))
    report['n8_masked_refit_error_nm']=float(np.max(abs(
        masked_decomposition['fit']['coeff_nm']-masked_reference['coeff_nm'])))
    assert report['n8_masked_refit_error_nm']<1e-10
    report['status']='PASS';return report


def characteristic_r(wavelength,T,split,na,nb,ns):
    # Independent two-layer characteristic matrix, not the reflection recursion.
    M=np.eye(2,dtype=complex)
    for n,d in [(na,T*(1+split)/2),(nb,T*(1-split)/2)]:
        delta=2*np.pi*n*d/wavelength
        layer=np.array([[np.cos(delta),-1j*np.sin(delta)/n],[-1j*n*np.sin(delta),np.cos(delta)]])
        M=M@layer
    Y=(M[1,0]+M[1,1]*ns)/(M[0,0]+M[0,1]*ns)
    return (1-Y)/(1+Y)


def single_film_r(wavelength,T,n,ns):
    a=(1-n)/(1+n);b=(n-ns)/(n+ns);phase=np.exp(4j*np.pi*n*T/wavelength)
    return (a+b*phase)/(1+a*b*phase)


def verify_coating():
    report={};rng=np.random.default_rng(2027);errors=[]
    for _ in range(120):
        wave=rng.uniform(20,120);T=rng.uniform(2,100);q=rng.uniform(-1,1)
        na=rng.uniform(.65,1.2)+1j*rng.uniform(0,.08)
        nb=rng.uniform(.65,1.2)+1j*rng.uniform(0,.08)
        ns=rng.uniform(.65,1.2)+1j*rng.uniform(0,.08)
        errors.append(abs(c.two_film_r(wave,T,q,na,nb,ns)-characteristic_r(wave,T,q,na,nb,ns)))
    report['recursion_vs_matrix_120_cases']=float(max(errors));assert max(errors)<1e-11
    T=c.equal_split_design(40.)
    p=c.CoatingParameters(wavelength_nm=40.,total_thickness_nm=T,n_pupil=128)
    na,nb,ns=c.indices(p);q=np.linspace(-1,1,1001)
    rr=c.r_for_split(p,q);R=abs(rr)**2
    report['design_total_nm']=T;report['design_wavelength_nm']=p.wavelength_nm
    report['design_reflectance']=float(abs(c.r_for_split(p,0))**2)
    report['equal_split_slope']=c.reflection_split_slope(T)
    report['sampled_optimum_split']=float(q[np.argmax(R)])
    step=1e-3
    report['equal_split_second_derivative']=float((abs(c.r_for_split(p,step))**2-2*abs(c.r_for_split(p,0))**2+abs(c.r_for_split(p,-step))**2)/step**2)
    assert abs(report['equal_split_slope'])<1e-8 and abs(report['sampled_optimum_split'])<.002
    assert report['equal_split_second_derivative']<0 and np.max(R)<=1+1e-12
    report['all_A_endpoint_error']=float(abs(c.r_for_split(p,1)-single_film_r(40,T,na,ns)))
    report['all_B_endpoint_error']=float(abs(c.r_for_split(p,-1)-single_film_r(40,T,nb,ns)))
    assert max(report['all_A_endpoint_error'],report['all_B_endpoint_error'])<1e-12
    same=.94+.005j
    r_same=c.two_film_r(40,T,q,same,same,ns)
    report['identical_material_split_invariance']=float(np.max(abs(r_same-single_film_r(40,T,same,ns))))
    assert report['identical_material_split_invariance']<1e-12
    report['fixed_index_wavelength_thickness_scaling']=float(np.max(abs(rr-c.two_film_r(80,2*T,q,na,nb,ns))))
    assert report['fixed_index_wavelength_thickness_scaling']<1e-12
    base=c.coating_study(p)
    geometry_errors=[]
    for deviation in (-100,-55,0,35,100):
        pp=replace(p,deviation_percent=deviation);split=c.split_map(pp,base['u'],base['v'])
        da=T*(1+split)/2;db=T*(1-split)/2
        assert np.min(da)>=-1e-12 and np.min(db)>=-1e-12
        geometry_errors.append(float(np.max(abs(da+db-T))))
        if abs(deviation)==100:assert np.all(split==np.sign(deviation))
    report['local_total_thickness_error_nm']=max(geometry_errors);assert max(geometry_errors)<1e-11
    ideal=c.coating_study(replace(p,deviation_percent=0,added_defocus_nm=1.))
    report['zero_deviation_same_focus_psf_error']=float(np.max(abs(ideal['true_psf']-ideal['ideal_psf'])))
    assert report['zero_deviation_same_focus_psf_error']<1e-12
    uniform=c.coating_study(replace(p,nonuniformity=0.))
    report['uniform_split_unit_flux_shape_error']=float(np.max(abs(uniform['true_psf']/uniform['true_psf'].sum()-uniform['ideal_psf']/uniform['ideal_psf'].sum())))
    assert report['uniform_split_unit_flux_shape_error']<1e-12
    assert abs(uniform['powers']['true']-uniform['powers']['ideal'])>1e-4
    report['uniform_split_true_throughput']=uniform['powers']['true']
    incident=(p.n_pupil*p.fft_pad)**2*np.sum(base['amplitude']**2)/base['amplitude'].sum()**2
    report['image_power_vs_pupil_power_error']=abs(base['true_psf'].sum()/incident-base['powers']['true'])
    assert report['image_power_vs_pupil_power_error']<1e-12
    conv=c.coating_convergence(p,(0,2,4,8,12,16))
    assert np.all(np.diff(conv.complex_pupil_rms)<1e-10)
    report['n16_complex_pupil_rms']=float(conv.iloc[-1].complex_pupil_rms)
    report['n16_image_error']=float(conv.iloc[-1].psf_relative_L2)
    fig=c.plot_coating_convergence(conv);report['integer_order_ticks']=all(np.allclose(ax.get_xticks(),np.round(ax.get_xticks())) for ax in fig.axes);plt.close(fig)
    assert report['integer_order_ticks'];report['status']='PASS';return report


def verify_widgets(render=False):
    vis=m.make_dashboard(m.Parameters(n_pupil=128,fft_pad=2),render=render)
    values={'wavelength_nm':600.,'global_curvature':1.,'reflection_magnitude':.5,'reflection_phase_deg':80.,'depth_nm':100.,'n_max':2}
    for k,v in values.items():vis['controls'][k].value=v
    vis['button'].click();assert vis['state']['error'] is None,vis['state']['error']
    assert all(getattr(vis['state']['study']['p'],k)==v for k,v in values.items())
    decomposition=vis['state']['decomposition_n8']
    assert len(decomposition['coefficients'])==45 and decomposition['p'].n_max==8
    c20=decomposition['coefficients'].query('n==2 and m==0').coefficient_nm.iloc[0]
    assert abs(c20+100/np.sqrt(3))<1e-9
    uv=c.coating_dashboard(c.CoatingParameters(total_thickness_nm=c.equal_split_design(),n_pupil=128,fft_pad=2),render=render)
    values={'wavelength_nm':45.,'deviation_percent':-100.,'nA_real':.91,'nB_imag':.003}
    for k,v in values.items():uv['controls'][k].value=v
    uv['button'].click();assert uv['state']['error'] is None,uv['state']['error']
    assert np.all(uv['state']['study']['dA_nm']==0)
    return {'visible_controls':'PASS','visible_decomposition_n8_modes':45,'coating_controls':'PASS','rendered':render,'scope':'Python widget state/callbacks; no frontend browser automation','status':'PASS'}

if __name__=='__main__':
    result={'visible':verify_visible(),'coating':verify_coating(),'widgets':verify_widgets()}
    Path('checks').mkdir(exist_ok=True)
    Path('checks/combined_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
