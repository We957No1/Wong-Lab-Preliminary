"""Independent physical and numerical checks for the mirror-dent notebook."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from dataclasses import replace
import mirror_dent_optics as m


def verify_all(output=None,include_widget=True):
    r={}
    p=m.Parameters(n_pupil=128,fft_pad=4,reflection_magnitude=1.0)
    s=m.build_study(p);f=s['fit']; w=f['weights']
    r['gram_max_abs_error']=f['gram_error']; assert r['gram_max_abs_error']<1e-11
    r['weight_sum_error']=abs(w.sum()-1);assert r['weight_sum_error']<1e-14
    n,midx=4,2
    synthetic=1.25*m.zernike(n,midx,f['u'],f['v'])-.3*m.zernike(2,0,f['u'],f['v'])
    coefficients=f['B'].T@(w*synthetic)
    expected=np.array([1.25 if mode==(4,2) else -.3 if mode==(2,0) else 0 for mode in f['modes']])
    r['known_mode_recovery_nm']=float(np.max(abs(coefficients-expected)));assert r['known_mode_recovery_nm']<1e-11
    amplitude=2*p.depth_nm*np.cos(np.deg2rad(p.incidence_deg));F=p.area_fraction
    mean=amplitude*F*(-np.expm1(-1/F))
    second=amplitude**2*F/2*(-np.expm1(-2/F))
    analytic_rms=np.sqrt(second-mean**2)
    r['gaussian_mean_nm_error']=abs(np.dot(w,f['true_nm'])-mean)
    r['gaussian_rms_nm_error']=abs(f['true_rms_nm']-analytic_rms)
    assert max(r['gaussian_mean_nm_error'],r['gaussian_rms_nm_error'])<1e-11
    independent=[]
    for (n,mm),c in zip(f['modes'],f['coeff_nm']):
        if mm==0:
            exact=amplitude*np.sqrt(n+1)*quad(lambda t:np.exp(-t/F)*m.eval_legendre(n//2,2*t-1),0,1,epsabs=1e-12)[0]
            independent.append(abs(c-exact))
    r['independent_radial_integral_coeff_nm_error']=float(max(independent));assert r['independent_radial_integral_coeff_nm_error']<1e-10
    angular=np.array([abs(c) for (n,mm),c in zip(f['modes'],f['coeff_nm']) if mm!=0])
    r['centered_dent_nonradial_coeff_nm_max']=float(angular.max());assert r['centered_dent_nonradial_coeff_nm_max']<1e-12
    variance=f['true_rms_nm']**2
    modal_variance=sum(c*c for mode,c in zip(f['modes'],f['coeff_nm']) if mode!=(0,0))
    r['parseval_residual_variance_nm2_error']=abs(f['residual_rms_nm']**2-(variance-modal_variance))
    assert r['parseval_residual_variance_nm2_error']<1e-10
    # Piston invariance and the DFT shift theorem are independent physical invariants.
    _,piston=m.propagate_psf(np.full_like(s['u'],2.73),s['amplitude'],p)
    r['piston_psf_max_abs_error']=float(np.max(abs(piston-s['ideal_psf'])));assert r['piston_psf_max_abs_error']<1e-12
    ds=s['s'][1]-s['s'][0]; pixel_shift=3
    _,tilt=m.propagate_psf(p.wavelength_nm*pixel_shift*ds*s['u'],s['amplitude'],p)
    r['positive_tilt_roll_max_abs_error']=float(np.max(abs(tilt-np.roll(s['ideal_psf'],pixel_shift,axis=1))))
    assert r['positive_tilt_roll_max_abs_error']<1e-12
    r['phase_only_total_energy_relative_error']=max(abs(s[key].sum()/s['ideal_psf'].sum()-1) for key in ['true_psf','recon_psf'])
    assert r['phase_only_total_energy_relative_error']<1e-12
    perfect=m.build_study(replace(p,depth_nm=0))
    r['zero_dent_psf_error']=float(np.max(abs(perfect['recon_psf']-perfect['ideal_psf'])));assert r['zero_dent_psf_error']<1e-14
    # The ideal comparison removes the dent ONLY, including at nonzero selected focus.
    no_dent_defocused=m.build_study(replace(p,depth_nm=0,added_defocus_nm=60.0))
    r['no_dent_same_focus_psf_error']=float(max(np.max(abs(no_dent_defocused[k]-no_dent_defocused['ideal_psf'])) for k in ('true_psf','recon_psf')))
    assert r['no_dent_same_focus_psf_error']<1e-12
    assert np.max(abs(no_dent_defocused['ideal_psf']-perfect['ideal_psf']))>1e-3
    offset=m.fit_wavefront(replace(p,center_u=.35))
    cx=offset['coeff_nm'][offset['modes'].index((1,1))]
    cy=offset['coeff_nm'][offset['modes'].index((1,-1))]
    assert abs(cx)>.05 and abs(cy)<1e-12
    r['offset_x_tilt_nm']=float(cx);r['offset_y_tilt_nm']=float(cy)
    # An annular support does not preserve disk orthogonality: recover a known polynomial by QR.
    ann=m.polar_basis(8,'all',.3,64,128);aw=ann['weights'];AB=ann['B']
    target=.7*m.zernike(4,0,ann['u'],ann['v'])-.2*m.zernike(2,2,ann['u'],ann['v'])
    recovered,_,rank,_=m.linalg.lstsq(np.sqrt(aw)[:,None]*AB,np.sqrt(aw)*target,lapack_driver='gelsy')
    ex=np.array([.7 if mode==(4,0) else -.2 if mode==(2,2) else 0 for mode in ann['modes']])
    r['annular_QR_recovery_nm_error']=float(np.max(abs(recovered-ex)));assert r['annular_QR_recovery_nm_error']<1e-10
    r['annular_gram_nonidentity']=float(np.max(abs(ann['G']-np.eye(len(ex)))));assert r['annular_gram_nonidentity']>.01
    # Check even-sized kernel centering in the extended-object calculation.
    obj=np.zeros((33,33));obj[16,16]=1
    im=m.incoherent_image(obj,s['ideal_psf']);center=len(s['s'])//2
    expected=s['ideal_psf'][center-16:center+17,center-16:center+17]/s['ideal_psf'].sum()
    r['delta_object_convolution_alignment_error']=float(np.max(abs(im-expected)));assert r['delta_object_convolution_alignment_error']<1e-12
    conv=m.convergence_study(p,orders=(0,2,4,8,12,16,20))
    assert np.all(np.diff(conv.residual_rms_nm)<=1e-10)
    r['n20_residual_nm']=float(conv.iloc[-1].residual_rms_nm);assert r['n20_residual_nm']<.01
    r['n20_psf_relative_L2']=float(conv.iloc[-1].psf_relative_L2);assert r['n20_psf_relative_L2']<1e-6
    # Independent quadrature refinement, including an asymmetric/elliptical dent.
    q=replace(p,center_u=.2,center_v=-.1,aspect_ratio=1.7,orientation_deg=31)
    fq=m.fit_wavefront(q);fine=m.fit_wavefront(replace(q,quad_radial=96,quad_angular=192))
    r['quadrature_refinement_coeff_nm_error']=float(np.max(abs(fq['coeff_nm']-fine['coeff_nm'])));assert r['quadrature_refinement_coeff_nm_error']<1e-9
    noisy=replace(p,noise_rms_nm=.1,seed=123)
    r['noise_seed_repeat_error']=float(np.max(abs(m.fit_wavefront(noisy)['coeff_nm']-m.fit_wavefront(noisy)['coeff_nm'])));assert r['noise_seed_repeat_error']==0
    # Imaging sampling check: on-axis diffraction also has an independent disk integral.
    exact_axis=abs(np.dot(w,np.exp(2j*np.pi*f['true_nm']/p.wavelength_nm)))**2
    grid_results=[]
    for ng in (128,256,384):
        sg=m.build_study(replace(p,n_pupil=ng))
        value=float(sg['metrics'].loc['True dent','on_axis_ratio'])
        grid_results.append({'pupil_pixels':ng,'on_axis_ratio':value,'absolute_error':abs(value-exact_axis)})
    r['independent_on_axis_ratio']=float(exact_axis)
    r['image_grid_refinement']=grid_results
    assert grid_results[-1]['absolute_error']<grid_results[0]['absolute_error']
    assert grid_results[1]['absolute_error']<3e-4
    airy_study=m.build_study(replace(p,n_pupil=256))
    axis=airy_study['s'];sel=(axis>0)&(axis<=3)
    exact_airy=(2*m.j1(2*np.pi*axis[sel])/(2*np.pi*axis[sel]))**2
    r['ideal_airy_cut_max_abs_error']=float(np.max(abs(airy_study['ideal_psf'][len(axis)//2,sel]-exact_airy)))
    assert r['ideal_airy_cut_max_abs_error']<2e-3
    _,coarse=m.propagate_psf(s['actual_nm'],s['amplitude'],replace(p,fft_pad=2))
    r['zero_padding_common_samples_max_error']=float(np.max(abs(coarse-s['true_psf'][::2,::2])))
    assert r['zero_padding_common_samples_max_error']<1e-12
    if include_widget:
        dash=m.make_dashboard(p,render=False)
        dash['controls']['center_u'].value=.25;dash['controls']['n_max'].value=6
        dash['button'].click()
        assert dash['state']['error'] is None and dash['state']['study']['p'].center_u==.25
        assert dash['state']['study']['p'].n_max==6
        r['ipywidgets_button_callback']='PASS (Python backend; no frontend browser automation)'
    r['status']='PASS'
    if output:
        Path(output).write_text(json.dumps(r,indent=2),encoding='utf-8')
    return r

if __name__=='__main__':
    print(json.dumps(verify_all('checks/verification.json'),indent=2))
