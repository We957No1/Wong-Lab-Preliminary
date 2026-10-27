"""Three sequential reflections with Fresnel propagation between mirrors.

Scalar, unfolded, registered-pupil model. Each mirror has the same uniform r
and incidence angle; the first K mirrors carry identical surface departures.
The two gaps are free space. The final image uses a common ideal focusing step.
This is one three-bounce route, with no resonator feedback.
"""
from dataclasses import dataclass,replace,asdict
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
import mirror_dent_optics as common

FRESNEL_REFERENCE='https://ocw.mit.edu/courses/2-71-optics-spring-2009/resources/mit2_71s09_lec18/'


@dataclass(frozen=True)
class TrainParameters:
    dented_mirrors:int=3
    normal_spacing_mm:float=200.0


def centered_fft(field):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def centered_ifft(spectrum):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))


def fresnel_transfer(shape,pixel_mm,wavelength_nm,distance_mm):
    """H=exp[-i*pi*lambda*L*(fx^2+fy^2)], omitting a uniform exp(ikL).
    Spatial frequencies are cycles/mm, lambda is converted from nm to mm.
    The Fourier-domain operator is unitary on the padded numerical grid.
    """
    fy=np.fft.fftfreq(shape[0],d=pixel_mm)
    fx=np.fft.fftfreq(shape[1],d=pixel_mm)
    return np.exp(-1j*np.pi*(wavelength_nm*1e-6)*distance_mm*
                  (fy[:,None]**2+fx[None,:]**2))


def fresnel_step(field,transfer):
    if transfer is None:return field.copy()
    return np.fft.fftshift(np.fft.ifft2(
        np.fft.fft2(np.fft.ifftshift(field))*transfer))


def _embed(a,side):
    result=np.zeros((side,side),dtype=a.dtype)
    begin=(side-a.shape[0])//2
    result[begin:begin+a.shape[0],begin:begin+a.shape[1]]=a
    return result


def _three_reflections(incident,aperture,screen,r,count,transfer):
    field=incident.astype(complex,copy=True)
    for j in range(3):
        field*=r*aperture
        if j<count:field*=screen
        if j<2:field=fresnel_step(field,transfer)
    return field


def three_mirror_study(p,train=TrainParameters()):
    common.validate_parameters(p)
    if type(train.dented_mirrors) is not int or not 0<=train.dented_mirrors<=3:
        raise ValueError('The number of dented mirrors must be an integer from 0 to 3.')
    if not np.isfinite(train.normal_spacing_mm) or not 0<=train.normal_spacing_mm<=1000:
        raise ValueError('Normal mirror spacing must be between 0 and 1000 mm.')
    if p.incidence_deg>10:
        raise ValueError('Use a common incidence angle <=10 degrees for this projected-pupil model.')
    u,v,support,A,du=common.cartesian_pupil(p)
    side=p.n_pupil*p.fft_pad
    pixel_mm=du*p.radius_mm
    incident=_embed(A,side)
    aperture=_embed(support.astype(float),side)
    W=common.true_wavefront_nm(p,u,v)
    phase=np.exp(2j*np.pi*W/p.wavelength_nm)
    # All 45 coefficients through n=8 are used on each dented mirror.
    fit_p=replace(p,n_max=8,fit_family='all',noise_rms_nm=0.,
                  quad_radial=max(p.quad_radial,12),quad_angular=max(p.quad_angular,19))
    fit=common.fit_wavefront(fit_p)
    Wfit=common.reconstruct_wavefront(fit['coeff_nm'],fit['modes'],u,v)
    fitted_phase=np.exp(2j*np.pi*Wfit/p.wavelength_nm)
    screen=np.ones((side,side),dtype=complex)
    fitted_screen=screen.copy()
    begin=(side-p.n_pupil)//2;cut=np.s_[begin:begin+p.n_pupil,begin:begin+p.n_pupil]
    screen[cut]=phase;fitted_screen[cut]=fitted_phase
    distance=train.normal_spacing_mm/np.cos(np.deg2rad(p.incidence_deg))
    transfer=fresnel_transfer((side,side),pixel_mm,p.wavelength_nm,distance) if distance else None
    r=p.reflection_magnitude*np.exp(1j*np.deg2rad(p.reflection_phase_deg))
    focus=np.ones((side,side),dtype=complex)
    focus[cut]=np.exp(2j*np.pi*p.added_defocus_nm*
                      common.zernike(2,0,u,v)/p.wavelength_nm)
    normalization=float(A.sum()**2)
    incident_power=float(np.sum(A*A))
    # With kernel exp(-2*pi*i*s*u), s=NA*x/lambda equals cycles per u.
    s=np.fft.fftshift(np.fft.fftfreq(side,d=du))
    images={};powers={};selected_field=None
    for k in range(4):
        field=_three_reflections(incident,aperture,screen,r,k,transfer)
        images[k]=abs(centered_fft(field*focus))**2/normalization
        powers[k]=float(np.sum(abs(field)**2)/incident_power)
        if k==train.dented_mirrors:selected_field=field
    reconstructed=_three_reflections(incident,aperture,fitted_screen,r,
                                     train.dented_mirrors,transfer)
    Irecon=abs(centered_fft(reconstructed*focus))**2/normalization
    reconstruction_power=float(np.sum(abs(reconstructed)**2)/incident_power)
    truth=images[train.dented_mirrors]
    relative_error=float(np.linalg.norm(Irecon-truth)/max(np.linalg.norm(truth),1e-30))
    return {'p':p,'train':train,'u':u,'v':v,'support':support,'amplitude':A,
            'height_nm':common.gaussian_height_nm(p,u,v),'fit':fit,
            's':s,'ideal_psf':images[0],'true_psf':truth,'recon_psf':Irecon,
            'count_psfs':images,'count_powers':powers,
            'exit_field':selected_field,'reconstructed_exit_field':reconstructed,
            'powers':{'ideal':powers[0],'true':powers[train.dented_mirrors],'recon':reconstruction_power},
            'relative_psf_error':relative_error,'pixel_mm':pixel_mm,'path_gap_mm':distance,
            'case_names':['Ideal: three mirrors, no dents',
                          f'True: {train.dented_mirrors} of 3 dented',
                          'Reconstructed: n<=8 per mirror']}


def plot_train_overview(study):
    p=study['p'];t=study['train']
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    ax=axes[0]
    positions=np.array([[0.,1.],[1.,-1.],[2.,1.]])
    route=np.vstack([[-.4,.2],positions,[2.6,-.2]])
    for a,b in zip(route[:-1],route[1:]):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'.35','lw':1.5})
    for j,(x,y) in enumerate(positions):
        dented=j<t.dented_mirrors
        xx=np.linspace(x-.3,x+.3,120)
        yy=np.full_like(xx,y)
        if dented:yy+=np.sign(y)*.1*np.exp(-((xx-x-.20*p.center_u)/.10)**2)
        ax.plot(xx,yy,color='#bd4a2d' if dented else '#455a64',lw=4)
        ax.text(x,y+(.22 if y>0 else -.37),
                f'M{j+1}: '+('dent' if dented else 'ideal'),ha='center',fontsize=9)
    ax.annotate('',xy=(2.85,1),xytext=(2.85,-1),
                arrowprops={'arrowstyle':'<->','color':'#087e8b'})
    ax.text(2.95,0,f'g={t.normal_spacing_mm:g} mm',rotation=90,va='center',fontsize=9)
    ax.text(2.05,-.65,'to ideal\nfocus',ha='left',fontsize=8)
    ax.text(-.35,-1.68,f'Equal incidence: {p.incidence_deg:g} deg; same r on all 3\n'
            f'One pass: M1 -> M2 -> M3; geometry schematic',fontsize=8)
    ax.set(xlim=(-.55,3.15),ylim=(-1.85,1.75),
           title='Three facing mirrors: first K carry the dent')
    ax.axis('off')
    s=study['s'];mask=abs(s)<=p.psf_half_width_s;mid=len(s)//2
    colors=['#333333','#cc8b24','#087e8b','#bd4a2d']
    for k,color in enumerate(colors):
        axes[1].semilogy(s[mask],np.maximum(study['count_psfs'][k][mid,mask],1e-14),
                         label=f'{k} dented',color=color)
    axes[1].set(xlabel='s_x at s_y=0',ylabel='I / unit-reflector in-focus peak',
                title='0, 1, 2 or 3 identical dents',ylim=(p.psf_log_floor,1.2))
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    onaxis=[study['count_psfs'][k][mid,mid] for k in range(4)]
    axes[2].plot(range(4),onaxis,'o-',color='#087e8b')
    selected=t.dented_mirrors
    axes[2].scatter([selected],[onaxis[selected]],color='#bd4a2d',s=65,zorder=3,
                    label='Selected case')
    axes[2].set_xticks(range(4))
    axes[2].set(xlabel='Number of dented mirrors',ylabel='On-axis intensity / unit-reflector peak',
                title='Common focus and incident-light scale')
    axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    fig.suptitle(f'Three reflections, wavelength {p.wavelength_nm:g} nm; '
                 f'each ray-path gap = {study["path_gap_mm"]:.3g} mm\n'
                 f'Shared local dent centre (u,v) = ({p.center_u:+.2f}, {p.center_v:+.2f})')
    return common._finish(fig)


def make_train_dashboard(base_parameters,train_parameters=TrainParameters(),render=True):
    import ipywidgets as widgets
    p=base_parameters;t=train_parameters
    style={'description_width':'165px'};layout=widgets.Layout(width='390px')
    def slider(label,value,lo,hi,step):
        return widgets.FloatSlider(description=label,value=value,min=lo,max=hi,step=step,
            continuous_update=False,style=style,layout=layout,readout_format='.3g')
    controls={
        'dented_mirrors':widgets.IntSlider(description='Dented mirrors (of 3)',value=t.dented_mirrors,
            min=0,max=3,continuous_update=False,style=style,layout=layout),
        'normal_spacing_mm':slider('Facing-plane gap (mm)',t.normal_spacing_mm,125,1000,5),
        'depth_nm':slider('Shared dent depth (nm)',p.depth_nm,-550,550,5),
        'center_u':slider('Shared dent centre u',p.center_u,-1,1,.05),
        'center_v':slider('Shared dent centre v',p.center_v,-1,1,.05),
        'area_fraction':widgets.FloatLogSlider(description='Shared dent area ratio',value=p.area_fraction,
            base=10,min=-2.3,max=2,step=.05,continuous_update=False,style=style,layout=layout),
        'global_curvature':slider('Local dent -> global bow',p.global_curvature,0,1,.05),
        'wavelength_nm':slider('Shared wavelength (nm)',p.wavelength_nm,380,780,5),
        'reflection_magnitude':slider('Each mirror field |r|',p.reflection_magnitude,0,1,.01),
        'reflection_phase_deg':slider('Each mirror phase (deg)',p.reflection_phase_deg,-180,180,5),
    }
    button=widgets.Button(description='Compute three-mirror case',button_style='primary',icon='play',
                          layout=widgets.Layout(width='235px'))
    status=widgets.HTML(f'All mirrors: incidence {p.incidence_deg:g} deg. Change values, then Compute.')
    description=widgets.HTML(
        '<b>Three facing mirrors, one three-bounce path.</b> The first K mirrors '
        'carry the same dent. Ideal = the same three mirrors and gaps with no dents. '
        'Field propagation across both gaps precedes a common ideal focus. '
        'The shared centre is u=x/a, v=y/a in each registered pupil; keep u²+v²≤1. '
        'Position applies to the local dent; the global bow stays centred. '
        '|r| is the field reflection amplitude; power reflectance per mirror is |r|².')
    plot_box,plot_images=common.plot_image_slots(
        ['Mirror layout and dent-count comparison','Point-image comparison','Object-image comparison'])
    state={'study':None,'figures':{},'error':None,'compute_count':0,'render_count':0}
    def compute(_):
        if button.disabled:return
        button.disabled=True;state['error']=None;status.value='Computing three reflections...'
        state['compute_count']+=1
        try:
            keys=('depth_nm','area_fraction','global_curvature','wavelength_nm',
                  'reflection_magnitude','reflection_phase_deg','center_u','center_v')
            q=replace(p,**{key:controls[key].value for key in keys})
            config=replace(t,**{key:controls[key].value for key in ('dented_mirrors','normal_spacing_mm')})
            case=three_mirror_study(q,config)
            if render:
                figures={};pngs=[]
                with plt.ioff():
                    for name,make in [('01_facing_mirrors_and_counts',plot_train_overview),
                                      ('02_point_images',common.plot_psfs),
                                      ('03_object_images',common.plot_scene)]:
                        fig=make(case)
                        try:
                            figures[name]=fig;pngs.append(common.figure_png_bytes(fig))
                        finally:plt.close(fig)
                for view,png in zip(plot_images,pngs):view.value=png
                plot_box.layout.display=''
                state['figures']=figures;state['render_count']+=1
            state['study']=case
            status.value=(f'Complete: {config.dented_mirrors}/3 dented; shared centre '
                          f'({q.center_u:+.2f}, {q.center_v:+.2f}); '
                          f'power reflectance per mirror={100*q.reflection_magnitude**2:.2f}%.')
        except Exception as exc:
            state['error']=str(exc);status.value='Parameter error: '+str(exc)
        finally:button.disabled=False
    button.on_click(compute)
    left=widgets.VBox([controls[k] for k in ('dented_mirrors','center_u','center_v','depth_nm','area_fraction')])
    right=widgets.VBox([controls[k] for k in ('normal_spacing_mm','global_curvature','wavelength_nm','reflection_magnitude','reflection_phase_deg')])
    ui=widgets.VBox([description,widgets.HBox([left,right],layout=widgets.Layout(flex_flow='row wrap')),
                     button,status,plot_box])
    return {'ui':ui,'controls':controls,'button':button,'state':state,'compute':compute,
            'plot_images':plot_images}


def export_train(case,figures,directory='results/visible_three_mirrors'):
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    parameters={'mirror':asdict(case['p']),'train':asdict(case['train'])}
    (path/'parameters.json').write_text(json.dumps(parameters,indent=2),encoding='utf-8')
    for name,fig in figures.items():
        for ext in ('png','pdf'):fig.savefig(path/f'{name}.{ext}',bbox_inches='tight')
    provenance={
        'model':'One three-bounce folded route; scalar unfolded Fresnel propagation; common final ideal focus.',
        'ideal':'Same three mirrors, gaps, shared angle/reflectance and final focus, with no dents.',
        'dent_assignment':'First K mirrors; identical departures in registered unfolded pupil coordinates.',
        'reconstruction':'Each dented mirror uses its own identical n<=8 OPD reconstruction.',
        'field_propagation':'H=exp[-i*pi*lambda*L*(fx^2+fy^2)]; omit only uniform free-space phase.',
        'gap_ray_length_mm':case['path_gap_mm'],
        'normalization':'Common initial incident-aperture unit-reflector peak; keep all three reflectance factors and clipping.',
        'object_plot':'Local shift-invariant incoherent approximation; off-axis vignetting not traced.',
        'approximations':'Small equal incidence, projected circular footprints, registered local dents, planar nominal segments.',
        'fresnel_reference':FRESNEL_REFERENCE,
        'model_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'common_model_sha256':hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest()}
    (path/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    files=[]
    for f in sorted(path.iterdir()):
        if f.is_file() and f.name!='manifest.json':
            files.append({'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    (path/'manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    return path
