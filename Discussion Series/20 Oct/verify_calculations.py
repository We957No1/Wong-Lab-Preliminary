from pathlib import Path
import json, math, hashlib
import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss
from optics_calculations import *
BASE=Path(__file__).resolve().parent
checks={}
print('STAGE: matrices',flush=True)
rng=np.random.default_rng(20261020)
# Independent transfer matrices, mapping bottom tangential fields to the top.
def matrix_rt(indices,thickness,wavelength,theta,pol):
    y,kz=admittances(indices,theta,pol); M=np.eye(2,dtype=complex)
    for j,d in enumerate(thickness,1):
        psi=2*np.pi*kz[j]*d/wavelength
        layer=np.array([[np.cos(psi),-1j*np.sin(psi)/y[j]],[-1j*y[j]*np.sin(psi),np.cos(psi)]])
        M=M@layer
    den=M[0,0]+M[0,1]*y[-1]; yeff=(M[1,0]+M[1,1]*y[-1])/den
    r=(y[0]-yeff)/(y[0]+yeff); t=(1+r)/den
    return r,t,y
errors=[]; conservation=[]
for absorbing in [False,True]:
    for k in range(80):
        nf=int(rng.integers(1,6)); n=np.r_[1.,rng.uniform(1.05,2.2,nf),1.5].astype(complex)
        if absorbing:n[1:-1]+=1j*rng.uniform(.001,.08,nf)
        thick=rng.uniform(.01,.4,nf); theta=rng.uniform(0,.8)
        for pol in ['s','p']:
            r=stack_reflection(n,thick,1,theta,pol); rm,tm,y=matrix_rt(n,thick,1,theta,pol)
            errors.append(abs(r-rm))
            if not absorbing:conservation.append(abs(abs(rm)**2+(y[-1].real/y[0].real)*abs(tm)**2-1))
checks['stack_vs_transfer_matrix_320_cases_max_abs_error']=max(errors)
checks['lossless_energy_max_abs_error']=max(conservation)
assert max(errors)<1e-11 and max(conservation)<1e-11
n=np.array([1,1.4+.07j,1.7]);y,_=admittances(n);r01=(y[0]-y[1])/(y[0]+y[1]);r12=(y[1]-y[2])/(y[1]+y[2]);p=np.exp(2j*np.pi*n[1]*.18);q=-r01*r12*p*p
series=r01+(1-r01*r01)*r12*p*p*sum(q**j for j in range(80))
checks['film_series_error']=float(abs(series-stack_reflection(n,[.18],1)))
checks['zero_thickness_film_error']=float(abs(stack_reflection(n,[0],1)-(n[0]-n[-1])/(n[0]+n[-1])))
assert checks['film_series_error']<1e-12 and checks['zero_thickness_film_error']<1e-12
print('STAGE: symbolic',flush=True)
# Exact symbolic normalization link and polynomial admissibility.
u,v=sp.symbols('u v',real=True);rho=sp.symbols('rho',nonnegative=True)
for n0 in range(9):
    for m0 in range(-n0,n0+1,2):
        Z=polynomial(n0,m0); assert sp.Poly(Z,u,v).total_degree()==n0
        for (p0,q0),_ in sp.Poly(Z,u,v).terms():assert p0>=0 and q0>=0 and (n0-p0-q0)%2==0
    if n0%2==0:assert sp.expand(polynomial(n0,0).subs({u:rho,v:0})/sp.sqrt(n0+1)-sp.legendre(n0//2,2*rho*rho-1))==0
checks['symbolic_selection_rules_and_legendre_through_n8']='PASS'
print('STAGE: Gram matrix',flush=True)
# Full two-dimensional quadrature of 45 real modes through n=8.
x,w=leggauss(40);t=(x+1)/2;wt=w/2;theta=2*np.pi*np.arange(128)/128
qu=np.sqrt(t)[:,None]*np.cos(theta);qv=np.sqrt(t)[:,None]*np.sin(theta)
modes=[(n,m)for n in range(9)for m in range(-n,n+1,2)]
Z=np.stack([zernike(n,m,qu,qv).ravel()for n,m in modes]);weights=np.broadcast_to(wt[:,None]/128,qu.shape).ravel()
gram=(Z*weights)@Z.T
checks['45_mode_gram_matrix_max_abs_error']=float(np.max(abs(gram-np.eye(len(modes)))))
assert checks['45_mode_gram_matrix_max_abs_error']<1e-11
print('STAGE: FFT',flush=True)
# FFT sign, piston invariance, and energy; no independent peak normalization.
u,v,mask,du=pupil_grid(128)
s,I0=psf_from_waves(np.zeros_like(u),mask,du,8)
_,Ip=psf_from_waves(np.full_like(u,.37),mask,du,8)
_,It=psf_from_waves(u,mask,du,8) # W/lambda = u -> positive image displacement s_x=1
expected_shift=round(1/(s[1]-s[0]))
checks['piston_intensity_max_abs_error']=float(np.max(abs(I0-Ip)))
checks['positive_tilt_fft_shift_max_abs_error']=float(np.max(abs(It-np.roll(I0,expected_shift,axis=1))))
_,Ic=psf_from_waves(.15*zernike(3,1,u,v),mask,du,8)
checks['phase_only_fft_energy_relative_error']=float(abs(Ic.sum()/I0.sum()-1))
assert max(checks[k] for k in ['piston_intensity_max_abs_error','positive_tilt_fft_shift_max_abs_error','phase_only_fft_energy_relative_error'])<1e-11
print('STAGE: decompositions',flush=True)
# Exact Seidel-to-Zernike decompositions.
u,v=sp.symbols('u v',real=True)
assert sp.simplify(u-polynomial(1,1)/2)==0
assert sp.simplify((u*u+v*v)**2-(sp.Rational(1,3)+polynomial(2,0)/(2*sp.sqrt(3))+polynomial(4,0)/(6*sp.sqrt(5))))==0
assert sp.simplify(u*(u*u+v*v)-(polynomial(1,1)/3+polynomial(3,1)/(3*sp.sqrt(8))))==0
assert sp.simplify(u*u-(sp.Rational(1,4)+polynomial(2,0)/(4*sp.sqrt(3))+polynomial(2,2)/(2*sp.sqrt(6))))==0
checks['seidel_zernike_exact_symbolic_decompositions']='PASS'
checks['status']='PASS'
(BASE/'checks/verification_results.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
