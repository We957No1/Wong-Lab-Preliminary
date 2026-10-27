from __future__ import annotations
from dataclasses import dataclass, replace, asdict
from functools import lru_cache
from pathlib import Path
import json, hashlib, platform, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.special import eval_jacobi, eval_legendre, j1
from scipy import linalg, signal, integrate

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                     'figure.dpi': 115, 'savefig.dpi': 180,
                     'axes.spines.top': False, 'axes.spines.right': False})

@dataclass(frozen=True)
class Parameters:
    # Physical lengths have explicit units. Positive depth means a depression.
    wavelength_nm: float = 13.5
    mirror_diameter_mm: float = 20.0
    focal_length_mm: float = 200.0
    incidence_deg: float = 0.0  # measured from the surface normal
    depth_nm: float = 3.0
    area_fraction: float = 0.10  # nominal area INSIDE the 1/e depth contour
    center_u: float = 0.0      # offsets in pupil-radius units
    center_v: float = 0.0
    aspect_ratio: float = 1.0  # sigma_major/sigma_minor; preserves nominal area
    orientation_deg: float = 0.0
    # Advanced support/illumination settings; keep defaults for Chapter 7's disk.
    obscuration_ratio: float = 0.0  # inner radius / outer pupil radius
    edge_amplitude: float = 1.0    # A(rho)=edge_amplitude**rho**2
    # Wavefront fit; n_max is RADIAL DEGREE, not a Noll/Fringe index.
    n_max: int = 8
    fit_family: str = 'all'         # 'all' or deliberately restrict to 'axisymmetric'
    noise_rms_nm: float = 0.0       # synthetic OPD measurement noise at fit nodes
    seed: int = 27
    added_defocus_nm: float = 0.0  # common coefficient multiplying Z_2^0 in imaging
    # Numerical sampling: fit quadrature and image grid are independent.
    n_pupil: int = 256
    fft_pad: int = 4
    quad_radial: int = 64
    quad_angular: int = 128
    # Display and test-object settings, in s = NA*x/lambda0 coordinates.
    psf_half_width_s: float = 5.0
    psf_log_floor: float = 1e-5
    scene_half_width_s: float = 7.0
    scene_kind: str = 'mixed'       # 'mixed', 'two_points', or 'bars'
    point_flux_s2: float = 0.2       # each point's integrated intensity in s-coordinate area
    point_separation_s: float = 1.0
    bar_width_s: float = 0.50
    bar_gap_s: float = 0.50

    @property
    def radius_mm(self): return self.mirror_diameter_mm/2
    @property
    def numerical_aperture(self): return self.radius_mm/self.focal_length_mm
    @property
    def image_unit_nm(self): return self.wavelength_nm/self.numerical_aperture
    @property
    def sigma_major(self): return np.sqrt(self.area_fraction*self.aspect_ratio/2)
    @property
    def sigma_minor(self): return np.sqrt(self.area_fraction/(2*self.aspect_ratio))


def validate_parameters(p):
    numeric = [v for v in asdict(p).values() if isinstance(v, (int,float))]
    if not np.all(np.isfinite(numeric)): raise ValueError('All numeric parameters must be finite.')
    if p.wavelength_nm <= 0 or p.mirror_diameter_mm <= 0 or p.focal_length_mm <= 0:
        raise ValueError('Wavelength, diameter and focal length must be positive.')
    if p.numerical_aperture > 0.2: raise ValueError('This paraxial model requires NA <= 0.2.')
    if not 0 <= p.incidence_deg <= 30: raise ValueError('Use 0 <= incidence_deg <= 30; footprint mapping is idealized.')
    if not .005 <= p.area_fraction <= .8: raise ValueError('Use .005 <= area_fraction <= .8; increase sampling for smaller dents.')
    if not 1 <= p.aspect_ratio <= 4: raise ValueError('Use 1 <= aspect_ratio <= 4.')
    if np.hypot(p.center_u,p.center_v) > 1: raise ValueError('Dent centre must lie inside or on the pupil.')
    if not 0 <= p.obscuration_ratio <= .6: raise ValueError('Use 0 <= obscuration_ratio <= .6.')
    if not .05 <= p.edge_amplitude <= 1: raise ValueError('Use .05 <= edge_amplitude <= 1.')
    if type(p.n_max) is not int or not 0 <= p.n_max <= 24: raise ValueError('n_max must be an integer from 0 to 24.')
    if p.fit_family not in ('all','axisymmetric'): raise ValueError('Unknown fit_family.')
    if p.noise_rms_nm < 0: raise ValueError('noise_rms_nm must be nonnegative.')
    if type(p.n_pupil) is not int or p.n_pupil < 64 or p.n_pupil > 512 or p.n_pupil%2:
        raise ValueError('n_pupil must be an even integer from 64 to 512.')
    if type(p.fft_pad) is not int or not 2 <= p.fft_pad <= 8 or p.n_pupil*p.fft_pad > 3072:
        raise ValueError('Use integer fft_pad from 2 to 8 and FFT side <= 3072.')
    if p.quad_radial < p.n_max+4 or p.quad_angular < 2*p.n_max+3:
        raise ValueError('Increase quadrature: radial >= n_max+4, angular >= 2*n_max+3.')
    if p.scene_kind not in ('mixed','two_points','bars'): raise ValueError('Unknown scene_kind.')
    if min(p.psf_half_width_s,p.scene_half_width_s,p.point_flux_s2,p.point_separation_s,p.bar_width_s,p.bar_gap_s) <= 0:
        raise ValueError('View and object dimensions must be positive.')
    if max(p.psf_half_width_s,p.scene_half_width_s) >= p.n_pupil/4:
        raise ValueError('Requested view must fit inside the FFT field.')
    if not 0 < p.psf_log_floor < 1: raise ValueError('psf_log_floor must lie between 0 and 1.')
    # Estimate the largest Gaussian surface slope; mm -> nm factor is explicit.
    slope = abs(p.depth_nm)*np.exp(-.5)/(p.sigma_minor*p.radius_mm*1e6)
    if slope > .01: warnings.warn('Large surface slopes: the phase-screen model needs review.')
    if p.incidence_deg > 10: warnings.warn('Only the cos(i) phase factor is included; no oblique footprint/vector model.')
    return p


def mode_list(n_max, family='all'):
    return [(n,m) for n in range(n_max+1) for m in range(-n,n+1,2)
            if family=='all' or m==0]


def zernike(n,m,u,v):
    """Real, unit-mean-square FULL-DISK modes; m>0 cosine, m<0 sine."""
    M=abs(m)
    if n < 0 or M > n or (n-M)%2: raise ValueError('Invalid (n,m).')
    rho=np.hypot(u,v); theta=np.arctan2(v,u)
    radial=rho**M*eval_jacobi((n-M)//2,0,M,2*rho*rho-1)
    if m==0: return np.sqrt(n+1)*radial
    angular=np.cos(M*theta) if m>0 else np.sin(M*theta)
    return np.sqrt(2*(n+1))*radial*angular


def mode_name(n,m):
    if (n,m)==(0,0): return 'Piston'
    if n==1: return ('x' if m>0 else 'y')+' tilt'
    if (n,m)==(2,0): return 'Defocus'
    if n==2 and abs(m)==2: return 'Primary astigmatism'
    if n==3 and abs(m)==1: return 'Balanced coma'
    if n==3 and abs(m)==3: return 'Trefoil'
    if (n,m)==(4,0): return 'Balanced primary spherical'
    if (n,m)==(6,0): return 'Balanced secondary spherical'
    if m==0: return 'Higher radial / spherical family'
    return 'Higher angular mode'


def weighted_mean(x,w): return float(np.dot(w,np.ravel(x)))
def weighted_rms(x,w,remove_mean=False):
    x=np.ravel(x)
    if remove_mean: x=x-weighted_mean(x,w)
    return float(np.sqrt(max(0,np.dot(w,x*x))))


def gaussian_height_nm(p,u,v):
    """Height relative to the ideal mirror: positive points toward incident space."""
    angle=np.deg2rad(p.orientation_deg)
    dx=u-p.center_u; dy=v-p.center_v
    qx=np.cos(angle)*dx+np.sin(angle)*dy
    qy=-np.sin(angle)*dx+np.cos(angle)*dy
    return -p.depth_nm*np.exp(-.5*((qx/p.sigma_major)**2+(qy/p.sigma_minor)**2))


def true_wavefront_nm(p,u,v):
    # A depression lengthens the reflected path. Constant coating phase is omitted.
    return -2*np.cos(np.deg2rad(p.incidence_deg))*gaussian_height_nm(p,u,v)


@lru_cache(maxsize=5)
def polar_basis(n_max,family,obscuration,nr,nt):
    """Gauss-Legendre in q=rho^2 and periodic angular quadrature.
    Weights sum to one over the admitted geometric pupil, independent of A.
    """
    x,wx=np.polynomial.legendre.leggauss(nr)
    q=obscuration**2+(1-obscuration**2)*(x+1)/2
    rho,theta=np.meshgrid(np.sqrt(q),2*np.pi*np.arange(nt)/nt,indexing='ij')
    u=(rho*np.cos(theta)).ravel(); v=(rho*np.sin(theta)).ravel()
    weights=np.repeat(wx/2/nt,nt)
    modes=mode_list(n_max,family)
    B=np.column_stack([zernike(n,m,u,v) for n,m in modes])
    G=B.T@(weights[:,None]*B)
    return {'u':u,'v':v,'weights':weights,'modes':modes,'B':B,'G':G}


def fit_wavefront(p):
    basis=polar_basis(p.n_max,p.fit_family,p.obscuration_ratio,p.quad_radial,p.quad_angular)
    u,v,w,B=(basis[k] for k in ('u','v','weights','B'))
    truth=true_wavefront_nm(p,u,v)
    rng=np.random.default_rng(p.seed)
    noise=rng.normal(0,p.noise_rms_nm,truth.shape)
    measured=truth+noise
    if p.obscuration_ratio==0:
        coeff=B.T@(w*measured)  # c_j=<W Z_j>; full disk area weights, NOT amplitude weights
        method='Full-disk orthogonal projection'
    else:
        rootw=np.sqrt(w)
        # QR with column pivoting; do not explicitly invert the normal equations.
        coeff,_,rank,_=linalg.lstsq(rootw[:,None]*B,rootw*measured,lapack_driver='gelsy')
        if rank < len(coeff): raise ValueError('Masked-pupil fit is rank deficient; reduce n_max.')
        method='Area-weighted least squares (pivoted QR) on admitted pupil'
    reconstructed=B@coeff
    error=truth-reconstructed
    eigenvalues=np.linalg.eigvalsh(basis['G'])
    condition=float(np.sqrt(max(eigenvalues[-1],0)/max(eigenvalues[0],np.finfo(float).tiny)))
    if condition>1e7: warnings.warn('Fit is ill conditioned; reduce n_max or obscuration.')
    return {**basis,'true_nm':truth,'measured_nm':measured,'fit_nm':reconstructed,
            'coeff_nm':coeff,'error_nm':error,'method':method,'condition_design':condition,
            'gram_error':float(np.max(abs(basis['G']-np.eye(len(coeff))))),
            'true_rms_nm':weighted_rms(truth,w,True),
            'residual_rms_nm':weighted_rms(error,w,True),
            'raw_residual_rms_nm':weighted_rms(error,w),
            'noise_realized_rms_nm':weighted_rms(noise,w,True)}


def reconstruct_wavefront(coeff,modes,u,v):
    result=np.zeros_like(u,dtype=float)
    # Ignore numerical angular roundoff at the 1e-13 relative coefficient level.
    tol=1e-13*max(float(np.max(abs(coeff))),1e-20)
    for c,(n,m) in zip(coeff,modes):
        if abs(c)>tol: result+=c*zernike(n,m,u,v)
    return result


def coefficient_table(p,fit):
    rows=[{'n':n,'m':m,'mode':mode_name(n,m),'coefficient_nm':c,
           'coefficient_waves':c/p.wavelength_nm}
          for (n,m),c in zip(fit['modes'],fit['coeff_nm'])]
    return pd.DataFrame(rows)


def cartesian_pupil(p):
    du=2/p.n_pupil
    coord=(np.arange(p.n_pupil)-p.n_pupil//2)*du
    u,v=np.meshgrid(coord,coord); rho=np.hypot(u,v)
    support=(rho<=1)&(rho>=p.obscuration_ratio)
    amplitude=np.where(support,p.edge_amplitude**(rho*rho),0.)
    return u,v,support,amplitude,du


def propagate_psf(wavefront_nm,amplitude,p):
    """Negative Fourier kernel; intensity uses the SAME ideal peak normalization.
    s=NA*x/lambda0. Zero padding interpolates the FFT; it adds no pupil detail.
    """
    side=p.n_pupil*p.fft_pad; start=(side-p.n_pupil)//2
    field=amplitude*np.exp(2j*np.pi*wavefront_nm/p.wavelength_nm)
    padded=np.zeros((side,side),dtype=complex)
    padded[start:start+p.n_pupil,start:start+p.n_pupil]=field
    amp=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    intensity=(amp.real**2+amp.imag**2)/amplitude.sum()**2
    s=np.fft.fftshift(np.fft.fftfreq(side,d=2/p.n_pupil))
    return s,intensity


def psf_metrics(s,I):
    center=len(s)//2
    disk=(s[:,None]**2+s[None,:]**2)<=.61**2
    return {'on_axis_ratio':float(I[center,center]),
            'sampled_peak_ratio':float(I.max()),
            'energy_in_0p61':float(I[disk].sum()/I.sum()),
            'discrete_total_intensity':float(I.sum())}


def build_study(p):
    validate_parameters(p)
    fit=fit_wavefront(p)
    u,v,support,A,du=cartesian_pupil(p)
    height=gaussian_height_nm(p,u,v)
    actual=true_wavefront_nm(p,u,v)
    reconstructed=reconstruct_wavefront(fit['coeff_nm'],fit['modes'],u,v)
    common_focus=p.added_defocus_nm*zernike(2,0,u,v)
    phase=2*np.pi*(actual+common_focus)/p.wavelength_nm
    phase_steps=[np.max(abs(np.diff(phase,axis=axis))[np.take(support,range(p.n_pupil-1),axis=axis)&np.take(support,range(1,p.n_pupil),axis=axis)]) for axis in (0,1)]
    max_phase_step=float(max(phase_steps))
    if max_phase_step>np.pi/2: warnings.warn('Large phase change per pupil pixel; increase n_pupil and check convergence.')
    if p.sigma_minor*p.n_pupil/2<5: warnings.warn('Gaussian width has fewer than five pixels; refine the pupil grid.')
    s,ideal=propagate_psf(common_focus,A,p)  # ideal comparison = dent removed, same focus
    _,truth=propagate_psf(actual+common_focus,A,p)
    _,recon=propagate_psf(reconstructed+common_focus,A,p)
    metric_rows=[]
    for label,I in [('Ideal',ideal),('True dent',truth),('Zernike reconstruction',recon)]:
        metric_rows.append({'case':label,**psf_metrics(s,I)})
    metrics=pd.DataFrame(metric_rows).set_index('case')
    relative_psf_error=float(np.linalg.norm(recon-truth)/np.linalg.norm(truth))
    # Actual clipped area: full geometric mirror, not the illuminated annular area.
    q=(u-p.center_u)*np.cos(np.deg2rad(p.orientation_deg))+(v-p.center_v)*np.sin(np.deg2rad(p.orientation_deg))
    r=-(u-p.center_u)*np.sin(np.deg2rad(p.orientation_deg))+(v-p.center_v)*np.cos(np.deg2rad(p.orientation_deg))
    inside_contour=(q/p.sigma_major)**2+(r/p.sigma_minor)**2<=2
    full_disk=(u*u+v*v)<=1
    clipped_area_fraction=float(np.count_nonzero(inside_contour&full_disk)/np.count_nonzero(full_disk))
    return {'p':p,'fit':fit,'u':u,'v':v,'support':support,'amplitude':A,
            'height_nm':height,'actual_nm':actual,'reconstructed_nm':reconstructed,
            'focus_nm':common_focus,'s':s,'ideal_psf':ideal,'true_psf':truth,'recon_psf':recon,
            'coefficients':coefficient_table(p,fit),'metrics':metrics,
            'relative_psf_error':relative_psf_error,'clipped_area_fraction':clipped_area_fraction,
            'max_phase_step_rad':max_phase_step}


def make_scene(p,ds):
    half=int(np.ceil(p.scene_half_width_s/ds))
    x=np.arange(-half,half+1)*ds; xx,yy=np.meshgrid(x,x); obj=np.zeros_like(xx)
    if p.scene_kind in ('two_points','mixed'):
        # Deposit point intensities by bilinear interpolation; preserve separation below one pixel.
        y0=2.0 if p.scene_kind=='mixed' else 0.0
        for x0 in (-p.point_separation_s/2,p.point_separation_s/2):
            fx=x0/ds+half; fy=y0/ds+half; ix=int(np.floor(fx)); iy=int(np.floor(fy))
            for dy in (0,1):
                for dx in (0,1):
                    if 0<=iy+dy<len(x) and 0<=ix+dx<len(x):
                        obj[iy+dy,ix+dx]+=p.point_flux_s2/ds**2*(1-abs(fy-(iy+dy)))*(1-abs(fx-(ix+dx)))
    if p.scene_kind in ('bars','mixed'):
        center_y=-1.5 if p.scene_kind=='mixed' else 0
        for center_x in (-1.5,0,1.5):
            # Supersample object pixels; narrow bars should not disappear by pixel coincidence.
            coverage=np.zeros_like(obj)
            for sy in (-.375,-.125,.125,.375):
                for sx in (-.375,-.125,.125,.375):
                    mask=(abs(xx+sx*ds-center_x*(p.bar_width_s+p.bar_gap_s)/1.5)<=p.bar_width_s/2)&(abs(yy+sy*ds-center_y)<=1.0)
                    coverage+=mask/16
            obj+=coverage
    return x,obj


def incoherent_image(obj,I):
    """Zero-extended linear convolution; explicitly align even-sized FFT kernels.
    PSF has unit DISCRETE sum. Object is mutually incoherent intensity.
    """
    kernel=I/I.sum()
    full=signal.fftconvolve(obj,kernel,mode='full')
    cy,cx=np.array(kernel.shape)//2
    image=full[cy:cy+obj.shape[0],cx:cx+obj.shape[1]]
    return np.maximum(image,0)  # remove floating-point negative roundoff only


def scene_images(study):
    p=study['p']; ds=study['s'][1]-study['s'][0]
    x,obj=make_scene(p,ds)
    return x,obj,{key:incoherent_image(obj,study[key+'_psf']) for key in ('ideal','true','recon')}


def convergence_study(p,orders=(0,2,4,6,8,10,12,16,20)):
    # Same noise realization and truth at each order; all fits are repeated honestly.
    base=build_study(p); rows=[]
    for nmax in sorted(set(orders)):
        q=replace(p,n_max=int(nmax))
        validate_parameters(q)
        fit=fit_wavefront(q)
        W=reconstruct_wavefront(fit['coeff_nm'],fit['modes'],base['u'],base['v'])
        _,I=propagate_psf(W+base['focus_nm'],base['amplitude'],q)
        true_var=fit['true_rms_nm']**2
        rows.append({'n_max':nmax,'mode_count':len(fit['modes']),
                     'residual_rms_nm':fit['residual_rms_nm'],
                     'residual_rms_waves':fit['residual_rms_nm']/p.wavelength_nm,
                     'variance_explained':1-fit['residual_rms_nm']**2/true_var if true_var>0 else 1.,
                     'psf_relative_L2':np.linalg.norm(I-base['true_psf'])/np.linalg.norm(base['true_psf']),
                     **psf_metrics(base['s'],I)})
    return pd.DataFrame(rows)


def area_rms_refocus_coefficient(study):
    """Additional Z_2^0 coefficient minimizing geometric-area OPD variance.
    This is NOT an optimization of peak intensity, and uses known simulated truth.
    """
    fit=study['fit']; w=fit['weights']
    z=zernike(2,0,fit['u'],fit['v']); z-=np.dot(w,z)
    W=fit['true_nm']+study['p'].added_defocus_nm*zernike(2,0,fit['u'],fit['v'])
    W=W-np.dot(w,W)
    return -float(np.dot(w,W*z)/np.dot(w,z*z))


def _finish(fig):
    fig.tight_layout()
    return fig


def _disk_map(ax,study,data,title,limit=None,cmap='RdBu_r'):
    p=study['p']; a=p.radius_mm
    masked=np.where(study['support'],data,np.nan)
    if limit is None: limit=max(float(np.nanmax(abs(masked))),1e-12)
    du=2/p.n_pupil
    extent=np.array([-1-du/2,1-du/2,-1-du/2,1-du/2])*a
    im=ax.imshow(masked,origin='lower',extent=extent,cmap=cmap,vmin=-limit,vmax=limit)
    ax.set(title=title,xlabel='Pupil x (mm)',ylabel='Pupil y (mm)')
    ax.set_aspect('equal')
    return im


def plot_geometry(study):
    p=study['p']; u,v=study['u'],study['v']
    fig=plt.figure(figsize=(14,4.2))
    ax=fig.add_subplot(131)
    im=_disk_map(ax,study,study['height_nm'],'Surface departure from ideal mirror')
    fig.colorbar(im,ax=ax,label='Height h (nm); negative = dent',shrink=.8)
    ax=fig.add_subplot(132,projection='3d')
    stride=max(1,p.n_pupil//65)
    h=np.where(study['support'],study['height_nm'],np.nan)
    ax.plot_surface(u[::stride,::stride]*p.radius_mm,v[::stride,::stride]*p.radius_mm,
                    h[::stride,::stride],cmap='viridis',linewidth=0,antialiased=True)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',zlabel='h (nm)',title='Dent height: vertical scale exaggerated')
    ax.view_init(elev=28,azim=-60)
    ax=fig.add_subplot(133)
    line=np.linspace(-1,1,800)
    hline=gaussian_height_nm(p,line,np.full_like(line,p.center_v))
    Wline=true_wavefront_nm(p,line,np.full_like(line,p.center_v))
    ax.plot(line*p.radius_mm,hline,label='Surface h')
    ax.plot(line*p.radius_mm,Wline,label='Reflected OPD W = -2 h cos(i)')
    ax.axhline(0,color='.6',lw=.6)
    ax.set(xlabel='x (mm), through dent centre',ylabel='Departure (nm)',title='Surface height and wavefront are different')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Gaussian mirror defect: nominal 1/e area = '+f'{100*p.area_fraction:.1f}% of the mirror',y=1.03)
    return _finish(fig)


def plot_wavefronts(study):
    fit=study['fit']; w=fit['weights']
    a=study['actual_nm']-np.dot(w,fit['true_nm'])
    b=study['reconstructed_nm']-np.dot(w,fit['fit_nm'])
    residual=a-b
    limit=max(float(np.max(abs(a[study['support']]))),float(np.max(abs(b[study['support']]))),1e-9)
    fig,axes=plt.subplots(1,3,figsize=(13,4.0))
    for ax,data,title,lim in zip(axes,[a,b,residual],
       ['True OPD (piston removed)','Zernike OPD (piston removed)','True minus reconstructed OPD'],
       [limit,limit,None]):
        im=_disk_map(ax,study,data,title,lim)
        fig.colorbar(im,ax=ax,label='OPD (nm)',shrink=.8)
    fig.suptitle(f'n_max={study["p"].n_max}; residual area RMS = {fit["residual_rms_nm"]:.5g} nm. Residual uses its own labelled scale.',y=1.02)
    return _finish(fig)


def plot_coefficients(study):
    table=study['coefficients']; p=study['p']
    fig,axes=plt.subplots(1,2,figsize=(13,4.1))
    nonzero=table.loc[table['coefficient_nm'].abs().sort_values(ascending=False).index].head(15)
    axes[0].bar(np.arange(len(nonzero)),nonzero.coefficient_nm,color='#177b91')
    axes[0].set_xticks(np.arange(len(nonzero)),[f'({n},{m})' for n,m in zip(nonzero.n,nonzero.m)],rotation=60)
    axes[0].set(ylabel='OPD coefficient (nm)',xlabel='(n,m), ranked by absolute coefficient',title='Largest fitted coefficients (includes piston)')
    axes[0].axhline(0,color='.3',lw=.7)
    maxc=max(float(table.coefficient_nm.abs().max()),1e-12)
    im=axes[1].scatter(table.m,table.n,c=table.coefficient_nm,cmap='RdBu_r',vmin=-maxc,vmax=maxc,s=170,edgecolors='.3',lw=.3)
    axes[1].set(xlabel='Angular index m',ylabel='Radial degree n',title='All retained modes; only m=0 survives ideal symmetry')
    fig.colorbar(im,ax=axes[1],label='OPD coefficient (nm)')
    axes[1].set_yticks(range(0,p.n_max+1,2));axes[1].grid(alpha=.2)
    return _finish(fig)


def plot_modal_contributions(study):
    coeff=study['fit']['coeff_nm']; modes=study['fit']['modes']
    ranked=np.argsort(abs(coeff))[::-1]
    valid=[i for i in ranked if abs(coeff[i])>max(abs(coeff).max()*1e-9,1e-13)][:6]
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    maps=[coeff[i]*zernike(*modes[i],study['u'],study['v']) for i in valid]
    limit=max([float(np.max(abs(a[study['support']]))) for a in maps]+[1e-12])
    for ax,i,data in zip(axes.ravel(),valid,maps):
        n,m=modes[i]
        im=_disk_map(ax,study,data,f'c Z_({n},{m}); c = {coeff[i]:+.4g} nm',limit)
    for ax in axes.ravel()[len(valid):]:ax.axis('off')
    if valid:
        fig.colorbar(im,ax=list(axes.ravel()),label='Modal OPD (nm): one shared numerical scale',shrink=.85,pad=.02)
    else:
        axes[0,0].text(.5,.5,'All retained coefficients are zero.',ha='center',transform=axes[0,0].transAxes)
    fig.suptitle('Largest individual modal contributions: sum these maps to reconstruct W.')
    return fig


def _psf_crop(study):
    p=study['p']; s=study['s']; mask=abs(s)<=p.psf_half_width_s
    return mask,s[mask]


def _ee(s,I,rmax=5):
    radii=np.hypot(s[:,None],s[None,:])
    edges=np.linspace(0,rmax,151)
    values,_=np.histogram(radii,bins=edges,weights=I)
    return edges[1:],np.cumsum(values)/I.sum()


def plot_psfs(study):
    p=study['p']; s=study['s']; mask,sc=_psf_crop(study)
    ds=s[1]-s[0]; extent=[sc[0]-ds/2,sc[-1]+ds/2,sc[0]-ds/2,sc[-1]+ds/2]
    images=[study[k+'_psf'] for k in ('ideal','true','recon')]
    titles=['Ideal: no dent','True dent: direct diffraction','From Zernike coefficients']
    fig,axes=plt.subplots(2,3,figsize=(13,8.0))
    for ax,I,title in zip(axes[0],images,titles):
        im=ax.imshow(np.maximum(I[np.ix_(mask,mask)],p.psf_log_floor),origin='lower',extent=extent,
                     cmap='magma',norm=LogNorm(p.psf_log_floor,1))
        ax.set(title=title,xlabel='s_x = NA x / wavelength',ylabel='s_y = NA y / wavelength')
        fig.colorbar(im,ax=ax,label='I / no-dent in-focus peak',shrink=.8)
    mid=len(s)//2
    colors=['#333333','#c54b31','#087e8b']
    for I,title,color in zip(images,titles,colors):
        axes[1,0].semilogy(s[mask],np.maximum(I[mid,mask],1e-14),label=title,color=color)
        rr,ee=_ee(s,I,p.psf_half_width_s);axes[1,1].plot(rr,ee,label=title,color=color)
    axes[1,0].set(xlabel='s_x at s_y=0',ylabel='I / no-dent in-focus peak',ylim=(p.psf_log_floor,1.2),title='Same normalization: no rescaling of each peak')
    axes[1,0].legend(fontsize=7);axes[1,0].grid(alpha=.2)
    axes[1,1].axvline(.61,color='.5',ls=':',lw=1)
    axes[1,1].set(xlabel='Radius s',ylabel='Fraction of total FFT-plane energy',ylim=(0,1.01),title='Encircled energy about the reference origin')
    axes[1,1].grid(alpha=.2)
    diff=(images[2]-images[1])[np.ix_(mask,mask)]
    limit=max(float(np.max(abs(diff))),1e-12)
    im=axes[1,2].imshow(diff,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit)
    axes[1,2].set(xlabel='s_x',ylabel='s_y',title='Reconstructed minus true intensity')
    fig.colorbar(im,ax=axes[1,2],label='Difference / no-dent in-focus peak',shrink=.8)
    fig.suptitle(f'1 image unit = {p.image_unit_nm:.4g} nm; clear-disk Airy reference = {0.61*p.image_unit_nm:.4g} nm. PSF relative L2 error = {study["relative_psf_error"]:.4g}',y=1.02)
    return _finish(fig)


def plot_scene(study):
    p=study['p']; x,obj,images=scene_images(study); ds=x[1]-x[0]
    extent=[x[0]-ds/2,x[-1]+ds/2,x[0]-ds/2,x[-1]+ds/2]
    fig,axes=plt.subplots(2,3,figsize=(13,7.7))
    ax=axes[0,0];im=ax.imshow(np.maximum(obj,1e-3),origin='lower',extent=extent,cmap='gray',norm=LogNorm(vmin=1e-3,vmax=max(float(obj.max()),.01)))
    ax.set(title='Input object intensity (own logarithmic scale)',xlabel='s_x',ylabel='s_y')
    fig.colorbar(im,ax=ax,label='Object intensity (arbitrary units)',shrink=.75)
    vmax=max(float(I.max()) for I in images.values())
    for ax,key,title in [(axes[0,1],'ideal','Ideal: no dent'),(axes[0,2],'true','True dent: incoherent image'),(axes[1,0],'recon','Reconstructed incoherent image')]:
        im=ax.imshow(images[key],origin='lower',extent=extent,cmap='gray',vmin=0,vmax=max(vmax,1e-12))
        ax.set(title=title,xlabel='s_x',ylabel='s_y')
        fig.colorbar(im,ax=ax,label='Image intensity (common scale)',shrink=.75)
    diff=images['recon']-images['true'];limit=max(float(np.max(abs(diff))),1e-12)
    im=axes[1,1].imshow(diff,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit)
    axes[1,1].set(title='Reconstructed minus true image',xlabel='s_x',ylabel='s_y')
    fig.colorbar(im,ax=axes[1,1],label='Intensity difference',shrink=.75)
    row=int(np.argmin(abs(x-(-1.5 if p.scene_kind=='mixed' else 0))))
    for key,label in [('ideal','Ideal: no dent'),('true','True dent'),('recon','Reconstructed')]:
        axes[1,2].plot(x,images[key][row],label=label)
    axes[1,2].set(xlabel='s_x',ylabel='Image intensity',title=f'Cross section at s_y={x[row]:.3g}')
    axes[1,2].legend(fontsize=8);axes[1,2].grid(alpha=.2)
    error=np.linalg.norm(diff)/max(np.linalg.norm(images['true']),1e-30)
    fig.suptitle(f'Mutually incoherent, shift-invariant imaging: I_image = I_object * unit-energy PSF. Image relative L2 error = {error:.4g}',y=1.02)
    return _finish(fig)


def plot_convergence(table,study):
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].semilogy(table.n_max,np.maximum(table.residual_rms_nm,1e-14),'o-')
    axes[0,0].set(ylabel='Piston-removed residual OPD RMS (nm)',title='Wavefront reconstruction')
    axes[0,1].semilogy(table.n_max,np.maximum(table.psf_relative_L2,1e-15),'o-',color='#c54b31')
    axes[0,1].set(ylabel='Relative L2 error of full PSF array',title='Imaging reconstruction')
    axes[1,0].plot(table.n_max,100*table.variance_explained,'o-')
    axes[1,0].set(ylabel='Geometric-area OPD variance explained (%)',title='Noiseless nested fits improve wavefront RMS')
    axes[1,1].plot(table.n_max,table.on_axis_ratio,'o-',label='Reconstructed on-axis ratio')
    axes[1,1].axhline(study['metrics'].loc['True dent','on_axis_ratio'],ls='--',color='#c54b31',label='True on-axis ratio')
    axes[1,1].set(ylabel='I(0) / ideal peak',title='A single image metric can converge non-monotonically')
    axes[1,1].legend(fontsize=8)
    for ax in axes.ravel():ax.set_xlabel('Maximum radial degree n_max');ax.grid(alpha=.2)
    return _finish(fig)


def plot_refocus(study):
    extra=area_rms_refocus_coefficient(study); p=study['p']
    W=study['actual_nm']+study['focus_nm']+extra*zernike(2,0,study['u'],study['v'])
    _,I=propagate_psf(W,study['amplitude'],p)
    mask,sc=_psf_crop(study);ds=study['s'][1]-study['s'][0]
    extent=[sc[0]-ds/2,sc[-1]+ds/2,sc[0]-ds/2,sc[-1]+ds/2]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,img,title in zip(axes[:2],[study['true_psf'],I],['True dent at chosen reference focus','True dent after area-RMS refocus']):
        im=ax.imshow(np.maximum(img[np.ix_(mask,mask)],p.psf_log_floor),origin='lower',extent=extent,cmap='magma',norm=LogNorm(p.psf_log_floor,1))
        ax.set(title=title,xlabel='s_x',ylabel='s_y');fig.colorbar(im,ax=ax,label='I / no-dent in-focus peak',shrink=.8)
    fit=study['fit'];w=fit['weights']
    Wq=fit['true_nm']+(p.added_defocus_nm+extra)*zernike(2,0,fit['u'],fit['v'])
    im=_disk_map(axes[2],study,W-np.dot(w,Wq),'Residual wavefront after refocus')
    fig.colorbar(im,ax=axes[2],label='OPD (nm)',shrink=.8)
    sigma=weighted_rms(Wq,w,True)
    dz_nm=-4*np.sqrt(3)*extra/p.numerical_aperture**2
    fig.suptitle(f'Added Z_2^0 coefficient = {extra:+.5g} nm; equivalent plane shift = {dz_nm:+.5g} nm; remaining RMS = {sigma:.5g} nm',y=1.05)
    return _finish(fig)


def make_dashboard(base_parameters,render=True):
    """Button-driven widgets; control changes never silently overwrite the base study."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    style={'description_width':'155px'}; layout=widgets.Layout(width='370px')
    def fs(description,value,minimum,maximum,step):
        return widgets.FloatSlider(description=description,value=value,min=minimum,max=maximum,step=step,
                                   continuous_update=False,readout_format='.3g',style=style,layout=layout)
    p=base_parameters
    controls={
      'depth_nm':fs('Dent depth (nm)',p.depth_nm,-max(10.,abs(p.depth_nm)),max(10.,abs(p.depth_nm)),.1),
      'area_fraction':fs('1/e area fraction',p.area_fraction,.005,.8,.005),
      'center_u':fs('Centre u / radius',p.center_u,-1,1,.05),
      'center_v':fs('Centre v / radius',p.center_v,-1,1,.05),
      'aspect_ratio':fs('Gaussian aspect ratio',p.aspect_ratio,1,4,.1),
      'orientation_deg':fs('Ellipse rotation (deg)',p.orientation_deg,-180,180,5),
      'added_defocus_nm':fs('Added Z20 (nm OPD)',p.added_defocus_nm,-max(5.,abs(p.added_defocus_nm)),max(5.,abs(p.added_defocus_nm)),.1),
      'noise_rms_nm':fs('OPD noise RMS (nm)',p.noise_rms_nm,0,max(2.,p.noise_rms_nm),.01),
      'n_max':widgets.IntSlider(description='Maximum degree n',value=p.n_max,min=0,max=24,continuous_update=False,style=style,layout=layout),
      'wavelength_nm':widgets.FloatLogSlider(description='Wavelength (nm)',value=p.wavelength_nm,base=10,min=np.log10(min(1.,p.wavelength_nm)),max=np.log10(max(1000.,p.wavelength_nm)),step=.02,continuous_update=False,style=style,layout=layout),
      'fit_family':widgets.Dropdown(description='Fitted modes',value=p.fit_family,options=['all','axisymmetric'],style=style,layout=layout)}
    button=widgets.Button(description='Compute selected case',button_style='primary',icon='play')
    status=widgets.HTML('<i>Change controls, then click Compute. Baseline results above remain available.</i>')
    output=widgets.Output(); state={'study':None,'error':None}
    def compute(_):
        button.disabled=True;status.value='<b>Computing...</b>';state['error']=None
        try:
            q=replace(p,**{name:c.value for name,c in controls.items()})
            result=build_study(q);state['study']=result
            if render:
                with output:
                    clear_output(wait=True)
                    display(result['metrics'].round(6))
                    display(result['coefficients'].sort_values('coefficient_nm',key=abs,ascending=False).head(12))
                    for make in (plot_wavefronts,plot_psfs,plot_scene):
                        fig=make(result);display(fig);plt.close(fig)
            status.value=(f'<b>Complete:</b> {len(result["fit"]["modes"])} modes; '
                          f'residual RMS {result["fit"]["residual_rms_nm"]:.5g} nm; '
                          f'PSF relative L2 error {result["relative_psf_error"]:.5g}.')
        except Exception as exc:
            state['error']=str(exc);status.value='<b>Parameter error:</b> '+str(exc)
        finally:button.disabled=False
    button.on_click(compute)
    left=widgets.VBox([controls[k] for k in ['depth_nm','area_fraction','center_u','center_v','aspect_ratio','orientation_deg']])
    right=widgets.VBox([controls[k] for k in ['n_max','fit_family','wavelength_nm','added_defocus_nm','noise_rms_nm']])
    ui=widgets.VBox([widgets.HBox([left,right],layout=widgets.Layout(flex_flow='row wrap')),button,status,output])
    return {'ui':ui,'controls':controls,'button':button,'state':state,'compute':compute}


def export_study(study,convergence=None,figures=None,directory='results'):
    import importlib.metadata as metadata
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    p=study['p']
    (path/'parameters.json').write_text(json.dumps(asdict(p),indent=2),encoding='utf-8')
    study['coefficients'].to_csv(path/'zernike_coefficients.csv',index=False)
    study['metrics'].to_csv(path/'image_metrics.csv')
    if convergence is not None:convergence.to_csv(path/'convergence.csv',index=False)
    crop,sc=_psf_crop(study)
    np.savez_compressed(path/'reconstruction_arrays.npz',u=study['u'],v=study['v'],
                        support=study['support'],amplitude=study['amplitude'],
                        height_nm=study['height_nm'],true_wavefront_nm=study['actual_nm'],
                        reconstructed_wavefront_nm=study['reconstructed_nm'],
                        s_crop=sc,ideal_psf_crop=study['ideal_psf'][np.ix_(crop,crop)],
                        true_psf_crop=study['true_psf'][np.ix_(crop,crop)],
                        reconstructed_psf_crop=study['recon_psf'][np.ix_(crop,crop)])
    for name,fig in (figures or {}).items():
        fig.savefig(path/(name+'.png'),bbox_inches='tight')
        fig.savefig(path/(name+'.pdf'),bbox_inches='tight')
    versions={name:metadata.version(name) for name in ['numpy','scipy','matplotlib','pandas','ipywidgets','nbformat','nbclient','ipykernel']}
    provenance={'python':sys.version,'platform':platform.platform(),'versions':versions,
                'normalization':'Zernike full-disk unit mean square; m>0 cosine, m<0 sine',
                'height_to_opd':'W_nm = -2*h_nm*cos(incidence_deg); positive h toward incident space',
                'pupil_model':'Scalar paraxial; fixed mapped pupil; phase-only defect; uniform coating',
                'psf_normalization':'Common dent-free in-focus peak; no per-image peak normalization',
                'ideal_comparison':'Dent removed; identical pupil, amplitude, wavelength and added defocus',
                'object_model':'Spatially incoherent, shift-invariant intensity convolution',
                'fit_method':study['fit']['method'],
                'fit_gram_error':study['fit']['gram_error'],
                'fit_condition_design':study['fit']['condition_design'],
                'source':'Local aberration_budget_textbook.pdf Chapter 7; 27 Oct teaching model',
                'array_export':'PSF views are cropped; scalar metrics use the full FFT plane.'}
    (path/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    files=[]
    for f in sorted(path.iterdir()):
        if f.is_file() and f.name!='manifest.json':
            data=f.read_bytes();files.append({'path':f.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    (path/'manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    print(f'Exported {len(files)} files to {path.resolve()}')
    return path
