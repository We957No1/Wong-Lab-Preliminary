from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LogNorm
from optics_calculations import *
BASE=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':11,'savefig.dpi':190})
u,v,mask,du=pupil_grid(320)
fig,axes=plt.subplots(3,4,figsize=(11.5,8.8),layout='constrained')
cmap=plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#f5f7fa')
for ax,(n,m,name) in zip(axes.flat,ATLAS):
    z=np.where(mask,zernike(n,m,u,v),np.nan)
    im=ax.imshow(z,origin='lower',extent=(-1,1,-1,1),cmap=cmap,norm=Normalize(-3.2,3.2))
    ax.set_title(f'({n}, {m})  {name}',fontsize=12)
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); ax.set_xlabel('u'); ax.set_ylabel('v')
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='#52687a',lw=.6))
fig.colorbar(im,ax=axes,shrink=.8,label='Dimensionless normalized Zernike value Z (shared scale)',pad=.02)
fig.savefig(BASE/'figures/zernike_atlas.png'); plt.close(fig)

cases=[(1,1,'Tilt'),(2,0,'Defocus'),(2,2,'Astigmatism'),(3,1,'Balanced coma'),(4,0,'Balanced spherical'),(3,3,'Trefoil')]
all_cases=cases
for part,cases in enumerate([all_cases[:3],all_cases[3:]],1):
    u,v,mask,du=pupil_grid(256); coeff=.15
    fig,axs=plt.subplots(3,3,figsize=(10.5,6.8),layout='constrained')
    r=np.sqrt(np.linspace(.002,.995,20)); th=np.linspace(0,2*np.pi,65,endpoint=False)
    ru=(r[:,None]*np.cos(th)).ravel(); rv=(r[:,None]*np.sin(th)).ravel()
    for row,(n,m,name) in enumerate(cases):
        z=zernike(n,m,u,v); w=coeff*z
        im=axs[row,0].imshow(np.where(mask,w,np.nan),origin='lower',extent=(-1,1,-1,1),cmap=cmap,vmin=-.5,vmax=.5)
        axs[row,0].set_title(name+' | W / wavelength',fontsize=12)
        axs[row,0].set_xticks([-1,0,1]); axs[row,0].set_yticks([-1,0,1]); axs[row,0].set_xlabel('u'); axs[row,0].set_ylabel('v')
        gx,gy=gradient_function(n,m)(ru,rv); gx=np.broadcast_to(gx,ru.shape); gy=np.broadcast_to(gy,rv.shape)
        axs[row,1].scatter(coeff*gx,coeff*gy,s=1.2,c='#087e8b',alpha=.45,rasterized=True)
        axs[row,1].set_title('Geometric ray intercepts',fontsize=12)
        s,I=psf_from_waves(w,mask,du,8); cut=abs(s)<=3
        pi=axs[row,2].imshow(I[np.ix_(cut,cut)],origin='lower',extent=(s[cut][0],s[cut][-1],s[cut][0],s[cut][-1]),cmap='magma',norm=LogNorm(vmin=1e-4,vmax=1))
        axs[row,2].set_title('Diffraction: log intensity',fontsize=12)
        for ax in axs[row,1:]:
            ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.set_aspect('equal'); ax.set_xticks([-2,0,2]); ax.set_yticks([-2,0,2]); ax.set_xlabel('s_x'); ax.set_ylabel('s_y')
    fig.savefig(BASE/f'figures/pupil_ray_psf_{part}.png'); plt.close(fig)
    
print('Generated 12-panel atlas and two pupil/ray/PSF comparison figures.')
