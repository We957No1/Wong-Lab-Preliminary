import numpy as np
from scipy.constants import physical_constants,Avogadro,h,c,e
from pathlib import Path
re=physical_constants['classical electron radius'][0]
data={sym:np.loadtxt(Path('material_data')/(sym+'.nff'),skiprows=1) for sym in ('mo','si','o')}
def ff(sym,w):
 a=data[sym];E=h*c/e/(w*1e-9);return np.interp(E,a[:,0],a[:,1]),np.interp(E,a[:,0],a[:,2])
def n(sym,w):
 rho,M={'mo':(10.22,.09595),'si':(2.329,.028085),'o':(1,.016)}[sym];f1,f2=ff(sym,w);f=re*(w*1e-9)**2/(2*np.pi)*(rho*1000/M*Avogadro);return 1-f*f1+1j*f*f2
def ns(w):
 fsi=ff('si',w);fo=ff('o',w);fac=re*(w*1e-9)**2/(2*np.pi)*(2200/.060083*Avogadro);return 1-fac*(fsi[0]+2*fo[0])+1j*fac*(fsi[1]+2*fo[1])
def rr(w,t=7.,split=0,N=1):
 na,nb=n('mo',w),n('si',w); inds=[1]+[na,nb]*N+[ns(w)];ds=[t*(1+split)/2,t*(1-split)/2]*N;r=(inds[-2]-inds[-1])/(inds[-2]+inds[-1])
 for j in range(len(ds)-1,-1,-1):
  a,b=inds[j],inds[j+1];rab=(a-b)/(a+b);phase=np.exp(4j*np.pi*b*ds[j]/w);r=(rab+r*phase)/(1+rab*r*phase)
 return r
for N in [1]:
 wav=np.linspace(12.6,16,341);R=np.array([abs(rr(w,N=N))**2 for w in wav]);i=R.argmax();d=np.linspace(-1,1,401);rs=np.array([abs(rr(13.5,split=s,N=N))**2 for s in d]);print(N,'nMo,nSi,nSub',n('mo',13.5),n('si',13.5),ns(13.5),'eqR13.5',abs(rr(13.5,N=N))**2,'peaklambda',wav[i],R[i],'splitmax',d[rs.argmax()],rs.max())

from scipy.optimize import brentq

def slope(w,T=7,N=1):
 e=.00001
 return (abs(rr(w,T,e,N))**2-abs(rr(w,T,-e,N))**2)/(2*e)
ws=np.linspace(12.6,16,100)
for N in [1]:
 roots=[]
 for lo,hi in zip(ws[:-1],ws[1:]):
  if slope(lo,N=N)*slope(hi,N=N)<0:
   w0=brentq(lambda w:slope(w,N=N),lo,hi,xtol=1e-10)
   d=np.linspace(-1,1,2001);R=np.array([abs(rr(w0,split=q,N=N))**2 for q in d]);roots.append((w0,abs(rr(w0,N=N))**2,d[R.argmax()],R.max()))
 print('stationary equal',N,roots)
