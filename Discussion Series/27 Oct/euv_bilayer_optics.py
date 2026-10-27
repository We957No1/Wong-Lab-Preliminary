"""Arbitrary absorbing two-layer mirror at longer EUV wavelengths.
Indices are EDITABLE ILLUSTRATIVE ASSUMPTIONS, not measured material data.
Time convention exp(-i omega t), passive n=n_real+i*kappa with kappa >= 0.
"""
from dataclasses import dataclass,replace,asdict
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import brentq
import mirror_dent_optics as common


def two_film_r(wavelength_nm,total_nm,split,na,nb,ns):
    """Vacuum | film A | film B | semi-infinite substrate at normal incidence.
    split=(d_A-d_B)/total: -1 gives pure B, +1 pure A.
    Reflection is referred to one fixed, flat top plane; total thickness is fixed.
    """
    da=total_nm*(1+np.asarray(split))/2;db=total_nm*(1-np.asarray(split))/2
    rab=(na-nb)/(na+nb);rbs=(nb-ns)/(nb+ns);r0a=(1-na)/(1+na)
    eb=np.exp(4j*np.pi*nb*db/wavelength_nm)
    lower=(rab+rbs*eb)/(1+rab*rbs*eb)
    ea=np.exp(4j*np.pi*na*da/wavelength_nm)
    return (r0a+lower*ea)/(1+r0a*lower*ea)


@dataclass(frozen=True)
class CoatingParameters:
    wavelength_nm:float=40.0
    total_thickness_nm:float=20.98814668672224  # the notebook computes the exact design below
    nA_real:float=.920
    nA_imag:float=.006
    nB_real:float=.995
    nB_imag:float=.002
    nS_real:float=.980
    nS_imag:float=.010
    deviation_percent:float=40.0
    nonuniformity:float=1.0
    area_fraction:float=.10
    center_u:float=0.0
    center_v:float=0.0
    mirror_diameter_mm:float=20.
    focal_length_mm:float=200.
    obscuration_ratio:float=0.
    edge_amplitude:float=1.
    added_defocus_nm:float=0.
    n_max:int=8
    n_pupil:int=192
    fft_pad:int=4
    quad_radial:int=64
    quad_angular:int=128
    psf_half_width_s:float=4.
    psf_log_floor:float=1e-5
    scene_half_width_s:float=6.
    scene_kind:str='mixed'
    point_flux_s2:float=.2
    point_separation_s:float=1.
    bar_width_s:float=.5
    bar_gap_s:float=.5
    @property
    def top_material(self):return 'Material A'
    @property
    def bottom_material(self):return 'Material B'
    @property
    def substrate(self):return 'Unnamed substrate'
    @property
    def radius_mm(self):return self.mirror_diameter_mm/2
    @property
    def numerical_aperture(self):return self.radius_mm/self.focal_length_mm
    @property
    def image_unit_nm(self):return self.wavelength_nm/self.numerical_aperture


def indices(p):return complex(p.nA_real,p.nA_imag),complex(p.nB_real,p.nB_imag),complex(p.nS_real,p.nS_imag)

def r_for_split(p,split):return two_film_r(p.wavelength_nm,p.total_thickness_nm,split,*indices(p))


def reflection_split_slope(total,wavelength=40.,na=.92+.006j,nb=.995+.002j,ns=.98+.01j,step=1e-5):
    return float((abs(two_film_r(wavelength,total,step,na,nb,ns))**2-abs(two_film_r(wavelength,total,-step,na,nb,ns))**2)/(2*step))


def equal_split_design(wavelength=40.,na=.92+.006j,nb=.995+.002j,ns=.98+.01j):
    """Choose T so equal physical thickness maximizes R over split for this model.
    Uses an exact Fresnel calculation, not a generic equal-thickness claim.
    """
    estimate=wavelength/(na.real+nb.real)
    grid=np.linspace(.6*estimate,1.4*estimate,101)
    slopes=[reflection_split_slope(T,wavelength,na,nb,ns) for T in grid]
    candidates=[]
    for lo,hi,a,b in zip(grid[:-1],grid[1:],slopes[:-1],slopes[1:]):
        if a*b<0:
            total=brentq(lambda T:reflection_split_slope(T,wavelength,na,nb,ns),lo,hi,xtol=1e-10)
            q=np.linspace(-1,1,2001);R=abs(two_film_r(wavelength,total,q,na,nb,ns))**2
            if abs(q[np.argmax(R)])<=.002:candidates.append(total)
    if not candidates:raise ValueError('No equal-split maximum found. Choose other illustrative indices, or use equal thickness only as a uniform imaging reference.')
    return min(candidates,key=lambda T:abs(T-estimate))


def validate_coating(p):
    if not 20<=p.wavelength_nm<=120:raise ValueError('Use 20 <= wavelength <= 120 nm for this longer-EUV example.')
    if not .1<=p.total_thickness_nm<=200:raise ValueError('Use positive total thickness from .1 to 200 nm.')
    if not -100<=p.deviation_percent<=100 or not 0<=p.nonuniformity<=1:raise ValueError('Deviation must be -100..100%, nonuniformity 0..1.')
    if not .005<=p.area_fraction<=4 or np.hypot(p.center_u,p.center_v)>1:raise ValueError('Invalid defect width or centre.')
    if not 0<=p.n_max<=24 or type(p.n_max) is not int:raise ValueError('Use integer n_max from 0 to 24.')
    if p.quad_radial<p.n_max+4 or p.quad_angular<2*p.n_max+3:raise ValueError('Refine coefficient quadrature.')
    if p.n_pupil%2 or p.n_pupil<64 or p.n_pupil*p.fft_pad>3072:raise ValueError('Use an even pupil grid and FFT side <= 3072.')
    if not 0<=p.obscuration_ratio<=.6 or not .05<=p.edge_amplitude<=1:raise ValueError('Invalid pupil support/amplitude.')
    if not 0<p.numerical_aperture<=.2:raise ValueError('Keep paraxial NA <= .2.')
    if min(p.nA_real,p.nB_real,p.nS_real)<=0 or min(p.nA_imag,p.nB_imag,p.nS_imag)<0:raise ValueError('Use positive real indices and nonnegative absorption indices.')
    return p


def split_map(p,u,v):
    t=p.deviation_percent/100
    G=np.exp(-((u-p.center_u)**2+(v-p.center_v)**2)/p.area_fraction)
    return t+p.nonuniformity*t*(1-abs(t))*G


def coating_study(p):
    validate_coating(p)
    basis=common.polar_basis(p.n_max,'all',p.obscuration_ratio,p.quad_radial,p.quad_angular)
    uq,vq,w,B=(basis[k] for k in ('u','v','weights','B'))
    rref=complex(r_for_split(p,0))
    if abs(rref)<1e-8:raise ValueError('Reference reflection is near zero; choose another design for relative-pupil fitting.')
    ratio=r_for_split(p,split_map(p,uq,vq))/rref
    if p.obscuration_ratio==0:coeff=B.T@(w*ratio)
    else:coeff=common.linalg.lstsq(np.sqrt(w)[:,None]*B,np.sqrt(w)*ratio,lapack_driver='gelsy')[0]
    fit_rms=float(np.sqrt(np.dot(w,abs(ratio-B@coeff)**2)))
    u,v,support,A,du=common.cartesian_pupil(p)
    split=split_map(p,u,v);rtrue=r_for_split(p,split)
    relative=np.zeros_like(u,dtype=complex)
    for value,(n,m) in zip(coeff,basis['modes']):
        if abs(value)>1e-13:relative+=value*common.zernike(n,m,u,v)
    rrecon=rref*relative
    focus=p.added_defocus_nm*common.zernike(2,0,u,v)
    s,Iref=common.propagate_psf(focus,A,p,rref)
    _,Itrue=common.propagate_psf(focus,A,p,rtrue)
    _,Irecon=common.propagate_psf(focus,A,p,rrecon)
    powers={key:float(np.sum(A*A*abs(r)**2)/np.sum(A*A)) for key,r in [('ideal',rref),('true',rtrue),('recon',rrecon)]}
    rows=[{'case':key,**common.psf_metrics(s,I),'throughput':powers[key]} for key,I in [('ideal',Iref),('true',Itrue),('recon',Irecon)]]
    table=pd.DataFrame([{'n':n,'m':m,'Re_c':z.real,'Im_c':z.imag,'abs_c':abs(z)} for (n,m),z in zip(basis['modes'],coeff)])
    return {'p':p,'u':u,'v':v,'support':support,'amplitude':A,'s':s,'split':split,
            'dA_nm':p.total_thickness_nm*(1+split)/2,'dB_nm':p.total_thickness_nm*(1-split)/2,
            'r_reference':rref,'r_true':rtrue,'r_reconstructed':rrecon,'coefficients':table,
            'coefficient_values':coeff,'fit_basis':basis,'complex_pupil_rms':fit_rms,
            'relative_psf_error':float(np.linalg.norm(Irecon-Itrue)/max(np.linalg.norm(Itrue),1e-30)),
            'ideal_psf':Iref,'true_psf':Itrue,'recon_psf':Irecon,'powers':powers,
            'metrics':pd.DataFrame(rows).set_index('case'),
            'case_names':['Ideal: equal layers','Thickness deviation','Zernike complex-pupil fit']}


def coating_convergence(p,orders=(0,2,4,6,8,10,12,16)):
    rows=[]
    for n in sorted(set(orders)):
        case=coating_study(replace(p,n_max=int(n)))
        rows.append({'n_max':n,'mode_count':len(case['coefficient_values']),
                     'complex_pupil_rms':case['complex_pupil_rms'],'psf_relative_L2':case['relative_psf_error']})
    return pd.DataFrame(rows)


def plot_design(p):
    wavelength=np.linspace(max(20,p.wavelength_nm*.7),min(120,p.wavelength_nm*1.3),241)
    spectra=np.array([complex(r_for_split(replace(p,wavelength_nm=float(w)),0)) for w in wavelength])
    q=np.linspace(-1,1,501);r=r_for_split(p,q);r0=complex(r_for_split(p,0))
    fig,axes=plt.subplots(2,2,figsize=(11,6.8),constrained_layout=True)
    axes[0,0].plot(wavelength,100*abs(spectra)**2);axes[0,0].axvline(p.wavelength_nm,color='#c54b31',ls='--')
    axes[0,0].set(xlabel='Wavelength (nm)',ylabel='Equal-layer reflectance (%)',title='Equal films: wavelength scan at fixed indices')
    axes[0,1].plot(wavelength,np.rad2deg(np.unwrap(np.angle(spectra))));axes[0,1].axvline(p.wavelength_nm,color='#c54b31',ls='--')
    axes[0,1].set(xlabel='Wavelength (nm)',ylabel='Reflection phase (degrees)',title='Complex reflection changes with wavelength')
    axes[1,0].plot(100*q,100*abs(r)**2);axes[1,0].axvline(0,color='.4',ls=':');axes[1,0].axvline(p.deviation_percent,color='#c54b31',ls='--')
    axes[1,0].scatter([0],[100*abs(r0)**2],color='#087e8b',zorder=5)
    axes[1,0].set(xlabel='Uniform thickness split deviation (%)',ylabel='Reflectance (%)',title='Equal split is the calibrated design reference')
    axes[1,1].plot(100*q,np.rad2deg(np.unwrap(np.angle(r))));axes[1,1].axvline(0,color='.4',ls=':')
    axes[1,1].set(xlabel='Uniform thickness split deviation (%)',ylabel='Reflection phase (degrees)',title='A uniform phase changes no isolated image intensity')
    for ax in axes.ravel():ax.grid(alpha=.2)
    for ax in axes[1]:ax.set_xticks([-100,-50,0,50,100])
    fig.suptitle(f'{p.top_material} / {p.bottom_material} on {p.substrate}: total thickness {p.total_thickness_nm:g} nm; wavelength {p.wavelength_nm:.5f} nm')
    return fig


def plot_coating_maps(study):
    p=study['p'];fig,axes=plt.subplots(2,3,figsize=(13,7.4),constrained_layout=True)
    du=2/p.n_pupil;extent=np.array([-1-du/2,1-du/2,-1-du/2,1-du/2])*p.radius_mm
    support=study['support']
    maps=[(study['dA_nm'],f'Top layer {p.top_material}','Thickness (nm)','viridis',0,p.total_thickness_nm),
          (study['dB_nm'],f'Bottom layer {p.bottom_material}','Thickness (nm)','viridis',0,p.total_thickness_nm),
          (100*abs(study['r_true'])**2,'True reflected power','Reflectance (%)','viridis',0,None),
          (np.where(abs(study['r_true'])>1e-10,np.angle(study['r_true']/study['r_reference'],deg=True),np.nan),'Phase relative to equal layers (wrapped)','Degrees','RdBu_r',-180,180),
          (abs(study['r_true']-study['r_reconstructed'])/abs(study['r_reference']),'Complex-pupil reconstruction error','Absolute error / |r_reference|','viridis',0,None)]
    for ax,(data,title,label,cmap,vmin,vmax) in zip(axes.ravel(),maps):
        im=ax.imshow(np.where(support,data,np.nan),origin='lower',extent=extent,cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set(title=title,xlabel='Pupil x (mm)',ylabel='Pupil y (mm)');fig.colorbar(im,ax=ax,label=label,shrink=.8)
    ax=axes[1,2];mid=p.n_pupil//2;xx=study['u'][mid]*p.radius_mm
    for name,title in [('dA_nm','d_A'),('dB_nm','d_B')]:ax.plot(xx,study[name][mid],label=title)
    ax.plot(xx,study['dA_nm'][mid]+study['dB_nm'][mid],label='d_A + d_B',ls='--',color='.3')
    ax.set(xlabel='x through pupil centre (mm)',ylabel='Thickness (nm)',title='Total thickness stays fixed at every location')
    ax.legend();ax.grid(alpha=.2)
    fig.suptitle(f'Nominal split deviation {p.deviation_percent:+g}%; spatial nonuniformity {p.nonuniformity:.2f}; pupil-fit RMS {study["complex_pupil_rms"]:.4g}')
    return fig


def plot_coating_psfs(study):
    p=study['p'];s=study['s'];select=abs(s)<=p.psf_half_width_s;sc=s[select];ds=s[1]-s[0]
    extent=[sc[0]-ds/2,sc[-1]+ds/2,sc[0]-ds/2,sc[-1]+ds/2]
    images=[study[key+'_psf'] for key in ('ideal','true','recon')]
    maximum=max(float(i.max()) for i in images);floor=max(maximum*p.psf_log_floor,1e-20)
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    colors=['#333333','#c54b31','#087e8b']
    for ax,I,title in zip(axes[0],images,study['case_names']):
        im=ax.imshow(np.maximum(I[np.ix_(select,select)],floor),origin='lower',extent=extent,cmap='magma',norm=LogNorm(floor,max(maximum,floor*10)))
        ax.set(title=title,xlabel='s_x = NA x / wavelength',ylabel='s_y');fig.colorbar(im,ax=ax,label='I / unit-reflector peak',shrink=.8)
    mid=len(s)//2
    for I,title,color in zip(images,study['case_names'],colors):
        axes[1,0].plot(sc,I[mid,select],label=title,color=color)
        axes[1,1].plot(sc,I[mid,select]/max(I.sum(),1e-30),label=title,color=color)
    axes[1,0].set(xlabel='s_x at s_y = 0',ylabel='Absolute normalized intensity',title='Brightness and shape together')
    axes[1,0].legend(fontsize=7)
    axes[1,1].set(xlabel='s_x at s_y = 0',ylabel='Intensity / full-array sum',title='Unit-flux comparison isolates shape')
    axes[1,1].legend(fontsize=7)
    diff=(images[2]-images[1])[np.ix_(select,select)];limit=max(float(np.max(abs(diff))),1e-15)
    im=axes[1,2].imshow(diff,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit)
    axes[1,2].set(title='Reconstructed minus true image',xlabel='s_x',ylabel='s_y');fig.colorbar(im,ax=axes[1,2],label='Intensity difference',shrink=.8)
    for ax in axes[1,:2]:ax.grid(alpha=.2)
    fig.suptitle(f'1 image unit = {p.image_unit_nm:.4g} nm; PSF relative L2 error = {study["relative_psf_error"]:.4g}')
    return fig


def plot_coating_convergence(table):
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),constrained_layout=True)
    axes[0].semilogy(table.n_max,np.maximum(table.complex_pupil_rms,1e-15),'o-')
    axes[0].set(ylabel='RMS error of r / r_reference',title='Complex-pupil reconstruction')
    axes[1].semilogy(table.n_max,np.maximum(table.psf_relative_L2,1e-15),'o-',color='#c54b31')
    axes[1].set(ylabel='Relative L2 error of intensity',title='Image reconstruction')
    for ax in axes:ax.set(xlabel='Maximum integer degree n_max',xticks=table.n_max.astype(int));ax.grid(alpha=.2)
    return fig


def coating_dashboard(base,render=True):
    import ipywidgets as widgets
    from IPython.display import display,clear_output
    style={'description_width':'165px'};layout=widgets.Layout(width='390px')
    def fs(label,value,lo,hi,step):return widgets.FloatSlider(description=label,value=value,min=lo,max=hi,step=step,continuous_update=False,readout_format='.4g',style=style,layout=layout)
    controls={
      'wavelength_nm':fs('EUV wavelength (nm)',base.wavelength_nm,20,120,1),
      'total_thickness_nm':fs('Total thickness (nm)',base.total_thickness_nm,2,100,.2),
      'deviation_percent':fs('Nominal split deviation (%)',base.deviation_percent,-100,100,1),
      'nonuniformity':fs('Spatial nonuniformity',base.nonuniformity,0,1,.05),
      'area_fraction':widgets.FloatLogSlider(description='Defect width area ratio',value=base.area_fraction,base=10,min=-2,max=.6,step=.05,continuous_update=False,style=style,layout=layout),
      'n_max':widgets.IntSlider(description='Maximum degree n',value=base.n_max,min=0,max=24,continuous_update=False,style=style,layout=layout),
      'center_u':fs('Defect centre u',base.center_u,-1,1,.05),
      'center_v':fs('Defect centre v',base.center_v,-1,1,.05),
      'nA_real':fs('Material A: real n',base.nA_real,.5,1.5,.005),
      'nA_imag':fs('Material A: absorption',base.nA_imag,0,.1,.001),
      'nB_real':fs('Material B: real n',base.nB_real,.5,1.5,.005),
      'nB_imag':fs('Material B: absorption',base.nB_imag,0,.1,.001)}
    button=widgets.Button(description='Compute selected case',button_style='primary',icon='play')
    status=widgets.HTML('Change controls, then click Compute.');out=widgets.Output();state={'study':None,'error':None}
    def compute(_):
        button.disabled=True;state['error']=None;status.value='Computing...'
        try:
            p=replace(base,**{key:w.value for key,w in controls.items()});case=coating_study(p);state['study']=case
            if render:
                with out:
                    clear_output(wait=True);display(case['metrics'].round(7))
                    for make in (plot_coating_maps,plot_coating_psfs,common.plot_scene):
                        fig=make(case);display(fig);plt.close(fig)
            r0=case['r_reference'];status.value=(f'Equal-layer r = {r0.real:+.5f}{r0.imag:+.5f}i; R={100*abs(r0)**2:.4g}%; '
                           f'true throughput={100*case["powers"]["true"]:.4g}%; image-fit error={case["relative_psf_error"]:.4g}.')
        except Exception as exc:state['error']=str(exc);status.value='Parameter error: '+str(exc)
        finally:button.disabled=False
    button.on_click(compute)
    left=widgets.VBox([controls[k] for k in ('wavelength_nm','total_thickness_nm','deviation_percent','nonuniformity')])
    right=widgets.VBox([controls[k] for k in ('area_fraction','n_max','center_u','center_v')])
    advanced=widgets.Accordion(children=[widgets.VBox([controls[k] for k in ('nA_real','nA_imag','nB_real','nB_imag')])]);advanced.set_title(0,'Advanced: arbitrary complex material indices');advanced.selected_index=None
    ui=widgets.VBox([widgets.HBox([left,right],layout=widgets.Layout(flex_flow='row wrap')),advanced,button,status,out])
    return {'ui':ui,'controls':controls,'button':button,'state':state,'compute':compute}


def export_coating(study,convergence=None,figures=None,directory='results/euv'):
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    (path/'parameters.json').write_text(json.dumps(asdict(study['p']),indent=2),encoding='utf-8')
    study['coefficients'].to_csv(path/'complex_zernike_coefficients.csv',index=False)
    study['metrics'].to_csv(path/'image_metrics.csv')
    if convergence is not None:convergence.to_csv(path/'order_convergence.csv',index=False)
    crop=abs(study['s'])<=study['p'].psf_half_width_s
    np.savez_compressed(path/'coating_arrays.npz',u=study['u'],v=study['v'],support=study['support'],
                        dA_nm=study['dA_nm'],dB_nm=study['dB_nm'],r_true=study['r_true'],
                        r_reconstructed=study['r_reconstructed'],s_crop=study['s'][crop],
                        ideal_psf_crop=study['ideal_psf'][np.ix_(crop,crop)],
                        true_psf_crop=study['true_psf'][np.ix_(crop,crop)],recon_psf_crop=study['recon_psf'][np.ix_(crop,crop)])
    for name,fig in (figures or {}).items():fig.savefig(path/(name+'.png'),bbox_inches='tight');fig.savefig(path/(name+'.pdf'),bbox_inches='tight')
    record={'reference':'Two spatially uniform equal-thickness films at the same settings; not the input object.',
            'time_convention':'exp(-i omega t); n=1-delta+i beta; passive beta positive',
            'phase_and_amplitude':'Fit the complex relative pupil r/r_reference, not phase alone.',
            'normalization':'Common unit-reflector incident-aperture in-focus peak; no throughput renormalization.',
            'material_model':'Arbitrary unnamed illustrative indices; held fixed during wavelength sweeps (no measured dispersion model).',
            'r_reference':[study['r_reference'].real,study['r_reference'].imag],
            'model':'Exactly two finite films on an unnamed semi-infinite substrate, locally planar, normal incidence.',
            'equal_split_design_scope':'Equal-split maximum is calibrated at 40 nm for the declared arbitrary indices and computed total thickness. Changing them can move the optimum.'}
    (path/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    rows=[{'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(path.iterdir()) if f.is_file() and f.name!='manifest.json']
    (path/'manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print('Saved to',path.resolve());return path
