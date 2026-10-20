"""Reproducible scalar-pupil teaching calculations; lengths may use any consistent unit."""
from pathlib import Path
from functools import lru_cache
import math
import numpy as np
from scipy.special import eval_jacobi
import sympy as sp

ATLAS = [(0,0,'Piston'),(1,1,'x-tilt'),(1,-1,'y-tilt'),(2,0,'Defocus'),
 (2,2,'Astigmatism'),(2,-2,'45-degree astigmatism'),(3,1,'Balanced x-coma'),(3,-1,'Balanced y-coma'),
 (3,3,'Trefoil'),(4,0,'Primary spherical'),(4,2,'Secondary astigmatism'),(6,0,'Secondary spherical')]

def zernike(n,m,u,v):
    M=abs(m)
    if M>n or (n-M)%2: raise ValueError('Invalid Zernike indices')
    r=np.hypot(u,v); theta=np.arctan2(v,u)
    R=r**M*eval_jacobi((n-M)//2,0,M,2*r*r-1)
    if m==0: return np.sqrt(n+1)*R
    return np.sqrt(2*(n+1))*R*(np.cos(M*theta) if m>0 else np.sin(M*theta))

@lru_cache(None)
def polynomial(n,m):
    u,v=sp.symbols('u v',real=True); M=abs(m); k=(n-M)//2
    harmonic=sp.re(sp.expand((u+sp.I*v)**M)) if m>=0 else sp.im(sp.expand((u+sp.I*v)**M))
    radial=sum((-1)**j*sp.factorial(n-j)/(sp.factorial(j)*sp.factorial(k-j)*sp.factorial(M+k-j))*(u*u+v*v)**(k-j) for j in range(k+1))
    return sp.expand(sp.sqrt(n+1 if m==0 else 2*(n+1))*harmonic*radial)

@lru_cache(None)
def gradient_function(n,m):
    u,v=sp.symbols('u v',real=True); Z=polynomial(n,m)
    return sp.lambdify((u,v),(sp.diff(Z,u),sp.diff(Z,v)),'numpy')

def pupil_grid(n=256):
    x=(np.arange(n)-n/2)*(2/n); u,v=np.meshgrid(x,x)
    return u,v,u*u+v*v<=1,2/n

def psf_from_waves(waves,mask,du,pad=8):
    """Negative Fourier kernel. s = NA*x/lambda0; ideal peak is one."""
    field=mask*np.exp(2j*np.pi*waves)
    size=field.shape[0]*pad
    padded=np.zeros((size,size),complex)
    start=(size-field.shape[0])//2
    padded[start:start+field.shape[0],start:start+field.shape[1]]=field
    amp=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    intensity=np.abs(amp)**2/np.count_nonzero(mask)**2
    s=np.fft.fftshift(np.fft.fftfreq(size,d=du))
    return s,intensity

def admittances(indices,theta0=0,pol='s'):
    n=np.asarray(indices,complex); kpar=n[0]*np.sin(theta0)
    kz=np.sqrt(n*n-kpar*kpar+0j)
    kz=np.where((kz.imag<0)|((abs(kz.imag)<1e-14)&(kz.real<0)),-kz,kz)
    y=kz if pol=='s' else n*n/kz # common vacuum admittance cancels
    return y,kz

def stack_reflection(indices,thickness,wavelength,theta0=0,pol='s'):
    """N physical layer thicknesses; indices has N+2 entries. Passive layers."""
    if len(indices)!=len(thickness)+2: raise ValueError('Need N+2 indices for N films')
    y,kz=admittances(indices,theta0,pol); r=(y[:-1]-y[1:])/(y[:-1]+y[1:])
    effective=r[-1]
    for j in range(len(thickness)-1,-1,-1):
        phase=np.exp(4j*np.pi*kz[j+1]*thickness[j]/wavelength)
        effective=(r[j]+effective*phase)/(1+r[j]*effective*phase)
    return effective
