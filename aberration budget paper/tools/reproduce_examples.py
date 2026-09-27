"""Reproduce this study guide's teaching calculations, not the authors' data.

Requires numpy, scipy, matplotlib. All optical lengths are in nm unless stated.
The contact model is scalar, thin-mask, periodic, and partially coherent.
It is NOT the rigorous reflective-mask model used in the research paper.
"""
from pathlib import Path
import json
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import j1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge

OUT = Path(__file__).resolve().parents[1]
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
LAMBDA, NA = 13.5, 0.33
FC = NA / LAMBDA
REPORT = []

def say(text):
    REPORT.append(str(text))
    print(text)

def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"), dpi=160)
    plt.close(fig)

def radial(n, m, rho):
    m = abs(m)
    return sum((-1)**s * factorial(n-s) /
               (factorial(s)*factorial((n+m)//2-s)*factorial((n-m)//2-s))
               * rho**(n-2*s) for s in range((n-m)//2+1))

def zernike(n, m, rho, theta):
    r = radial(n, m, rho)
    if m == 0:
        return np.sqrt(n+1)*r
    a = np.cos(m*theta) if m > 0 else np.sin(abs(m)*theta)
    return np.sqrt(2*(n+1))*r*a

def verify_zernike():
    q,w = leggauss(60)
    rho = np.sqrt((q+1)/2)[:,None]
    theta = np.linspace(0, 2*np.pi, 128, endpoint=False)[None,:]
    modes = [(n,m) for n in range(7) for m in range(-n,n+1,2)]
    basis = np.array([zernike(n,m,rho,theta)+np.zeros_like(theta) for n,m in modes])
    gram = np.einsum("irt,jrt,r->ij",basis,basis,w/2)/theta.size
    error = np.max(np.abs(gram-np.eye(len(modes))))
    assert error < 2e-12
    say(f"Zernike Gram matrix, 28 modes n<=6: max error {error:.3g}")
    # Explicit low-order balancing identities and piston-removed variances.
    avg = lambda a: float(np.sum(a*w[:,None]/2))
    r = rho
    assert np.isclose(avg((r**2-.5)**2),1/12)
    assert np.isclose(avg((r**4-r**2+1/6)**2),1/180)
    say("Balanced quartic RMS = |B|/sqrt(180): verified.")
    return modes

def quasar():
    radii = np.sqrt(.6**2+(np.arange(3)+.5)/3*(.8**2-.6**2))
    angles = np.deg2rad(np.concatenate(
        [a+np.linspace(-12,12,5) for a in [45,135,225,315]]))
    rr,aa = np.meshgrid(radii,angles)
    return np.c_[rr.ravel()*np.cos(aa.ravel()),rr.ravel()*np.sin(aa.ravel())]

def amplitudes(pitch=38., widths=(14.,14.), coeff=None, focus=0.):
    """Source index, order index. Frequency-pupil W has +i 2pi W/lambda."""
    coeff = coeff or {}
    limit = int(np.ceil(1.8*FC*pitch))
    mm,nn = np.meshgrid(np.arange(-limit,limit+1),np.arange(-limit,limit+1))
    orders = np.c_[mm.ravel(),nn.ravel()]
    a,b = widths
    mask = (a*b/pitch**2*np.sinc(orders[:,0]*a/pitch)
            *np.sinc(orders[:,1]*b/pitch))
    source=quasar()
    uv = source[:,None,:]+orders[None,:,:]/(pitch*FC)
    rho=np.linalg.norm(uv,axis=-1)
    theta=np.arctan2(uv[:,:,1],uv[:,:,0])
    wave=np.zeros_like(rho)
    for (n,m),value in coeff.items():
        wave += value*zernike(n,m,rho,theta)
    wave -= focus*NA**2*rho**2/2
    amp=mask[None,:]*(rho<=1)*np.exp(2j*np.pi*wave/LAMBDA)
    return orders,amp

def profile(pitch=38., widths=(14.,14.), coeff=None, focus=0.,
            axis=0, points=None, other=0.):
    if points is None:
        points=np.linspace(-pitch/2,pitch/2,4097)
    orders,amp=amplitudes(pitch,widths,coeff,focus)
    phase=np.exp(2j*np.pi*(orders[:,axis,None]*points[None,:]
                         +orders[:,1-axis,None]*other)/pitch)
    return points,np.mean(np.abs(amp@phase)**2,axis=0)

def edges(x,intensity,threshold):
    crossing=np.where(np.diff(np.sign(intensity-threshold))!=0)[0]
    roots=np.array([x[i]+(threshold-intensity[i])*(x[i+1]-x[i])
                    /(intensity[i+1]-intensity[i]) for i in crossing])
    left=roots[roots<0];right=roots[roots>0]
    if len(left)==0 or len(right)==0:
        raise ValueError("Central feature has no pair of threshold crossings")
    return float(left[-1]),float(right[0])

def metrics(coeff=None,focus=0.,dose=1.,threshold=None):
    coeff=coeff or {}
    vals=[]
    for axis in [0,1]:
        # Cuts pass through the known rigidly translated centre for pure tilt.
        # For these examples all remaining modes preserve the feature centre.
        opposite_tilt = (1,-1) if axis==0 else (1,1)
        other=-2*coeff.get(opposite_tilt,0.)/NA
        x,y=profile(coeff=coeff,focus=focus,axis=axis,other=other)
        l,r=edges(x,dose*y,threshold)
        vals.extend([r-l,(r+l)/2])
    return vals

def main():
    modes=verify_zernike()
    say(f"lambda/NA = {LAMBDA/NA:.9f} nm")
    say(f"lambda/NA^2 = {LAMBDA/NA**2:.9f} nm")
    say(f"1.5 nm OPD at 193 nm = {1000*1.5/193:.9f} m-waves")
    say(f"1.5 nm OPD at 13.5 nm = {1000*1.5/LAMBDA:.9f} m-waves")
    say(f"20 m-waves = {LAMBDA*.020:.9f} nm")
    say(f"24 coefficients all at 20 m-waves: RMS {np.sqrt(24)*20:.9f} m-waves")
    say(f"15 nm defocus, paraxial disk RMS = {15*NA**2/(4*np.sqrt(3)):.9f} nm")
    say(f"RMS-normalized tilt, 1 m-wave: |shift| = {2*.001*LAMBDA/NA:.9f} nm")
    say(f"RMS-normalized tilt bound for 0.4 nm shift = {.4*NA/(2*LAMBDA)*1000:.9f} m-waves")
    say(f"Data rows before filtering = {984+77+972}; full grid = {41**24}")
    sizes=[84,84,64,64,50,36,32,25,25]
    say(f"Parameters for literal fully connected architecture = {sum((a+1)*b for a,b in zip(sizes,sizes[1:]))}")
    say(f"4x-scaled multilayer Bragg estimate = {2*(.725+1)*4*np.cos(np.deg2rad(6)):.9f} nm")
    say(f"Geometric 60 nm absorber shadow /4 = {60*np.tan(np.deg2rad(6))/4:.9f} nm")

    # Original educational pupil atlas.
    grid=np.linspace(-1,1,241)
    xx,yy=np.meshgrid(grid,grid); rr=np.hypot(xx,yy);tt=np.arctan2(yy,xx)
    chosen=[(0,0),(1,1),(1,-1),(2,0),(2,2),(2,-2),(3,1),(3,-1),(3,3),(4,0),(4,2),(6,0)]
    fig,axes=plt.subplots(3,4,figsize=(9.3,7.2),layout="constrained")
    for ax,(n,m) in zip(axes.flat,chosen):
        z=np.ma.masked_where(rr>1,zernike(n,m,rr,tt))
        ax.imshow(z,extent=(-1,1,-1,1),origin="lower",cmap="RdBu_r",vmin=-3.2,vmax=3.2)
        ax.set_title(f"(n,m) = ({n},{m})");ax.set_xticks([]);ax.set_yticks([])
        ax.add_patch(Circle((0,0),1,fill=False,ec="#23384e",lw=.8))
    save(fig,"zernike_atlas")

    # Illumination pupil and diffraction-order admission at one source point.
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.2),layout="constrained")
    for ax in axes:
        ax.add_patch(Circle((0,0),1,fill=False,color="#1b3c59"))
        ax.set(xlim=(-1.35,1.35),ylim=(-1.35,1.35),aspect="equal",
               xlabel="normalized pupil x",ylabel="normalized pupil y")
    for a in [45,135,225,315]:
        axes[0].add_patch(Wedge((0,0),.8,a-15,a+15,width=.2,facecolor="#147d92"))
    axes[0].set_title("Idealized quasar illumination")
    s=np.array([.7/np.sqrt(2),.7/np.sqrt(2)])
    for m in range(-1,2):
        for n in range(-1,2):
            p=s+np.array([m,n])/(38*FC)
            if np.max(np.abs(p))<1.35:
                inside=np.linalg.norm(p)<=1
                axes[1].plot(*p,"o",color="#147d92" if inside else "#c35b38")
                axes[1].annotate(f"({m},{n})",p,xytext=(4,5),textcoords="offset points",fontsize=9)
    axes[1].set_title("38 nm pitch: orders for one source point")
    save(fig,"quasar_orders")

    # Exact two-order phase mechanism.
    x=np.linspace(-30,30,1000);pitch=38.;a,b=1.,.7
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),layout="constrained")
    for phase,label in [(0,"zero differential phase"),(.2,"0.2 rad differential phase")]:
        axes[0].plot(x,a*a+b*b+2*a*b*np.cos(2*np.pi*x/pitch+phase),label=label)
    axes[0].set(xlabel="x (nm)",ylabel="intensity (arb. units)",title="Phase moves the fringe")
    axes[0].legend(fontsize=8)
    b1=1.;phis=[0,.4,.8]
    for phi in phis:
        field=a+2*b1*np.exp(1j*phi)*np.cos(2*np.pi*x/pitch)
        axes[1].plot(x,np.abs(field)**2,label=f"even-order phase {phi:.1f} rad")
    axes[1].set(xlabel="x (nm)",ylabel="intensity (arb. units)",title="Even phase changes the profile")
    axes[1].legend(fontsize=8)
    save(fig,"phase_to_image")

    # Threshold example, same-RMS aberrations and process variation.
    _,ref_at_edge=profile(points=np.array([7.]))
    threshold=float(ref_at_edge[0])
    baseline=metrics(threshold=threshold)
    assert np.isclose(baseline[0],14,atol=1e-5)
    cases={"reference":{},"x tilt, 20 m-waves":{(1,1):.27},
           "defocus, 20 m-waves":{(2,0):.27},
           "astigmatism, 20 m-waves":{(2,2):.27}}
    data={}
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.9),layout="constrained")
    for label,c in cases.items():
        x,y=profile(coeff=c)
        axes[0].plot(x,y/threshold,label=label)
        data[label]=dict(zip(["CDx_nm","PSx_nm","CDy_nm","PSy_nm"],metrics(c,threshold=threshold)))
    axes[0].axhline(1,color=".4",ls="--",lw=.8)
    axes[0].set(xlim=(-13,13),xlabel="x (nm)",ylabel="intensity / threshold",
                title="Same RMS, different printed edges")
    axes[0].legend(fontsize=7)
    for f in [-15,0,15]:
        x,y=profile(focus=f)
        axes[1].plot(x,y/threshold,label=f"focus {f:+} nm")
    axes[1].axhline(1,color=".4",ls="--",lw=.8)
    axes[1].set(xlim=(6.90,7.05),ylim=(.99,1.012),
                xlabel="x (nm), right-edge close-up",ylabel="intensity / threshold",
                title="Focus moves the threshold crossing")
    axes[1].ticklabel_format(axis="y",style="plain",useOffset=False)
    axes[1].legend(fontsize=8)
    save(fig,"contact_profiles")
    # Analytic shift check at very fine spatial sampling.
    observed=data["x tilt, 20 m-waves"]["PSx_nm"]
    assert np.isclose(observed,-2*.27/NA,atol=2e-5)
    assert np.isclose(data["x tilt, 20 m-waves"]["CDx_nm"],14,atol=3e-5)
    assert np.isclose(data["x tilt, 20 m-waves"]["CDy_nm"],14,atol=3e-5)
    say(f"Teaching contact model threshold = {threshold:.12g}")
    say(f"Tilt shift analytic {-2*.27/NA:.9f} nm; numerical {observed:.9f} nm")
    for label,item in data.items():
        say(label+": "+json.dumps(item))

    fig,axes=plt.subplots(1,2,figsize=(9.3,3.9),layout="constrained")
    focuses=np.linspace(-25,25,41)
    for dose in [.95,1,1.05]:
        cds=[metrics(focus=float(f),dose=dose,threshold=threshold)[0] for f in focuses]
        axes[0].plot(focuses,cds,label=f"dose {dose:.2f}")
    axes[0].set(xlabel="focus offset (nm)",ylabel="CD on central cut (nm)",
                title="Teaching Bossung curves")
    axes[0].legend(fontsize=8)
    all_edges=[]
    for f in [-15,0,15]:
        for dose in [.95,1,1.05]:
            x,y=profile(focus=f)
            all_edges.append(edges(x,dose*y,threshold))
    all_edges=np.array(all_edges)
    leftband=np.ptp(all_edges[:,0]);rightband=np.ptp(all_edges[:,1])
    say(f"Teaching 3x3 focus-dose edge-band widths: left {leftband:.9f}, right {rightband:.9f} nm")
    for i,(l,r) in enumerate(all_edges):
        axes[1].plot([l,r],[i,i],"o-",color="#147d92",ms=3)
    axes[1].axvspan(all_edges[:,0].min(),all_edges[:,0].max(),alpha=.18,color="#c35b38")
    axes[1].axvspan(all_edges[:,1].min(),all_edges[:,1].max(),alpha=.18,color="#c35b38")
    axes[1].set(xlabel="threshold edge coordinate (nm)",ylabel="process sample",
                title="Separate edge bands")
    save(fig,"focus_dose_bands")

    # Simple geometry of a correlated feasible set.
    c1=np.linspace(-1.3,1.3,401);c2=np.linspace(-1.3,1.3,401)
    xx,yy=np.meshgrid(c1,c2)
    feasible=(np.abs(xx+yy)<=.4)&(np.abs(xx-yy)<=2.0)
    fig,ax=plt.subplots(figsize=(6.1,4.7),layout="constrained")
    ax.contourf(xx,yy,feasible,levels=[.5,1.5],colors=["#bedee3"])
    ax.plot([-1,1],[1,-1],"o",color="#147d92",label="two feasible candidates")
    ax.plot(1,1,"x",color="#c35b38",ms=10,label="mixed coordinate maxima: infeasible")
    ax.add_patch(Rectangle((-1,-1),2,2,fill=False,ls="--",ec=".4",label="coordinate min/max box"))
    ax.set(xlabel="coefficient 1 (arbitrary units)",ylabel="coefficient 2 (arbitrary units)",
           title="Candidate ranges do not define a safe box",aspect="equal")
    ax.legend(loc="lower left",fontsize=8)
    save(fig,"feasible_budget")

    # Relative error near zero.
    actual=np.logspace(-4,0,300);absolute=.01
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),layout="constrained")
    axes[0].loglog(actual,100*absolute/actual,color="#147d92")
    axes[0].set(xlabel="reference shift magnitude (nm)",ylabel="percentage error (%)",
                title="Fixed 0.01 nm error")
    axes[1].semilogx(actual,np.full_like(actual,absolute),color="#c35b38")
    axes[1].set(xlabel="reference shift magnitude (nm)",ylabel="absolute error (nm)",
                ylim=(0,.025),title="The physical error stays constant")
    save(fig,"percentage_error")
    # Budget arithmetic and nonlinear inverse branch failure.
    assert abs((1+1))>.4 and abs((1-1))<=.4
    assert abs(1-(-1))<=2.0
    assert (1**2+(-1)**2)/2 == 1 and ((1+(-1))/2)**2 == 0
    say("Inverse-branch conditional-mean counterexample verified: mean target 0 is infeasible for a^2=1.")
    say("All assertions passed. These checks validate teaching examples only.")
    (OUT/"numerical_checks.txt").write_text("\n".join(REPORT)+"\n",encoding="utf-8")
    (OUT/"teaching_model_results.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
