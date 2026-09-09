"""Reproduce numerical and symbolic examples in the two optics textbooks.

Run with a Python environment containing NumPy and SymPy, for example:
    python "self-studying materials/verify_optics.py"

All lengths in the ray tracer are millimetres. Optical path difference (OPD)
means actual accumulated optical path minus the chosen ideal spherical-wave
optical path. Zernike coefficients have the same units as the supplied OPD.
The basis is real, RMS-normalized on the full unit disk: m>0 means cosine and
m<0 means sine; no single-index convention is used or silently assumed.

This is an educational calculation, not a production optical-design program.
It deliberately keeps surface intersection and refraction equations visible.
Running it writes numerical_checks.txt beside this script and to stdout.
"""

from __future__ import annotations

from math import factorial, pi, sqrt
from pathlib import Path

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss


REPORT: list[str] = []


def say(line: str = "") -> None:
    REPORT.append(line)
    print(line)


def radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """R_n^{|m|}; only admissible (n,m) with n-|m| even are accepted."""
    m = abs(m)
    if n < m or (n - m) % 2:
        raise ValueError("Need n >= |m| and n-|m| even.")
    out = np.zeros_like(rho, dtype=float)
    for k in range((n - m) // 2 + 1):
        coefficient = ((-1) ** k * factorial(n - k) /
                       (factorial(k) * factorial((n + m) // 2 - k) *
                        factorial((n - m) // 2 - k)))
        out += coefficient * rho ** (n - 2 * k)
    return out


def zernike(n: int, m: int, rho: np.ndarray,
            theta: np.ndarray) -> np.ndarray:
    R = radial(n, m, rho)
    if m == 0:
        return sqrt(n + 1) * R
    angle = np.cos(m * theta) if m > 0 else np.sin(abs(m) * theta)
    return sqrt(2 * (n + 1)) * R * angle


def disk_quadrature(n_radial: int = 48, n_angular: int = 128):
    """Quadrature for <f> = (1/pi) integral_disk f dA.

    u=rho^2 makes 2*rho*drho=du, hence <f>=integral_0^1 mean_theta(f) du.
    Gauss-Legendre nodes avoid an overrepresented central pixel.
    """
    x, w = leggauss(n_radial)
    rho, theta = np.meshgrid(np.sqrt((x + 1) / 2),
                             np.arange(n_angular) * 2 * pi / n_angular,
                             indexing="ij")
    weights = np.broadcast_to((w / (2 * n_angular))[:, None], rho.shape)
    return rho.ravel(), theta.ravel(), weights.ravel()


def check_zernike() -> None:
    say("1. Real, full-disk, RMS-normalized Zernike checks")
    rho, theta, weights = disk_quadrature()
    modes = [(n, m) for n in range(7) for m in range(-n, n + 1, 2)]
    design = np.column_stack([zernike(n, m, rho, theta) for n, m in modes])
    gram = design.T @ (weights[:, None] * design)
    error = np.max(np.abs(gram - np.eye(len(modes))))
    say(f"  Modes through n=6: {len(modes)}; quadrature samples: {len(rho)}")
    say(f"  Maximum full-disk Gram-matrix error: {error:.3e}")
    assert abs(weights.sum() - 1) < 1e-13
    assert error < 1e-11

    # Known synthetic OPD in micrometres; residual coefficients are zero.
    truth = np.zeros(len(modes))
    specified = {(0, 0): 0.020, (1, 1): -0.035, (2, 0): 0.080,
                 (2, -2): 0.042, (3, 1): -0.025, (4, 0): 0.060,
                 (6, 0): 0.008}
    for mode, value in specified.items():
        truth[modes.index(mode)] = value
    opd = design @ truth
    projected = design.T @ (weights * opd)
    assert np.max(np.abs(projected - truth)) < 1e-11
    say(f"  Synthetic OPD full-disk RMS including piston: {np.sqrt(weights @ opd**2):.9f} um")
    say(f"  Coefficient Euclidean norm: {np.linalg.norm(truth):.9f} um")
    assert abs(np.sqrt(weights @ opd**2) - np.linalg.norm(truth)) < 1e-12

    # The sample mask includes a central obstruction, a vertical support, and
    # missing measurements in one region. Ordinary disk orthogonality is lost.
    x, y = rho * np.cos(theta), rho * np.sin(theta)
    valid = (rho >= 0.25) & (np.abs(x) >= 0.04) & ~((x > 0.30) & (y > 0.25))
    A, w, measured = design[valid], weights[valid], opd[valid]
    sw = np.sqrt(w)
    fitted, _, rank, _ = np.linalg.lstsq(A * sw[:, None], measured * sw,
                                        rcond=None)
    # Incorrect on this mask: applying the full-disk projection rule after
    # merely normalizing valid area to one. This is shown to expose cross-talk.
    naive = A.T @ (w * measured) / w.sum()
    fit_error = np.max(np.abs(fitted - truth))
    say(f"  Masked samples: {valid.sum()}; sampled area fraction: {w.sum():.6f}")
    say(f"  Weighted-design rank: {rank}; condition number: {np.linalg.cond(A * sw[:, None]):.6f}")
    say(f"  Correct weighted least-squares maximum coefficient error: {fit_error:.3e} um")
    say(f"  Incorrect masked-projection maximum coefficient error: {np.max(np.abs(naive-truth)):.6f} um")
    assert rank == len(modes)
    assert fit_error < 1e-11
    say("  (n,m)       prescribed [um]        recovered [um]")
    for mode, value in specified.items():
        say(f"  {str(mode):9s} {value: .9f}            {fitted[modes.index(mode)]: .9f}")
    say()


def check_polynomial_balancing() -> None:
    say("2. Quartic/sixth-order expansion and least-RMS focus")
    q, A, B = sp.symbols("rho A B", real=True)
    R2 = 2*q**2 - 1
    R4 = 6*q**4 - 6*q**2 + 1
    R6 = 20*q**6 - 30*q**4 + 12*q**2 - 1
    expansion = ((A/3 + B/4) + (A/2 + 9*B/20)*R2 +
                 (A/6 + B/4)*R4 + B/20*R6)
    assert sp.expand(expansion - A*q**4 - B*q**6) == 0
    D, P = -A - sp.Rational(9, 10)*B, A/6 + B/5
    balanced = A*q**4 + B*q**6 + D*q**2 + P
    assert sp.simplify(2*sp.integrate(balanced*q, (q, 0, 1))) == 0
    assert sp.simplify(2*sp.integrate(balanced*R2*q, (q, 0, 1))) == 0
    variance = (A/6 + B/4)**2/5 + B**2/2800
    assert sp.simplify(2*sp.integrate(balanced**2*q, (q, 0, 1))-variance) == 0
    say("  W = A rho^4 + B rho^6")
    say("  c00=A/3+B/4; c20=(A/2+9B/20)/sqrt(3)")
    say("  c40=(A/6+B/4)/sqrt(5); c60=B/(20sqrt(7))")
    say("  Add D rho^2 + P, with D=-A-9B/10 and P=A/6+B/5.")
    say("  Balanced RMS = sqrt[(A/6+B/4)^2/5 + B^2/2800].")
    say("  Here P is the added piston, not the original piston coefficient.")
    say("  Symbolic expansion, zero mean, zero defocus and RMS identities: PASS")
    example = {A: -0.15625, B: 0.0001953125}  # um: R=100 mm, a=5 mm mirror
    say("  Example (micrometres): A=-0.156250000, B=+0.0001953125")
    for label, expr in [("c00", A/3+B/4), ("c20", (A/2+9*B/20)/sp.sqrt(3)),
                        ("c40", (A/6+B/4)/sp.sqrt(5)),
                        ("c60", B/(20*sp.sqrt(7))), ("added D", D),
                        ("added P", P), ("balanced RMS", sp.sqrt(variance))]:
        say(f"    {label:12s} = {float(expr.subs(example)):+.12f} um")
    say()


def check_sphere_and_phase_screen() -> None:
    say("3. Spherical mirror: exact geometry, series and exact reflected rays")
    r, R = sp.symbols("r R", positive=True)
    z = R - sp.sqrt(R**2-r**2)
    f = R/2
    W = -z + sp.sqrt(r**2 + (f-z)**2) - f
    series = sp.series(W, r, 0, 8).removeO().expand()
    expected = -r**4/(4*R**3) + r**6/(8*R**5)
    assert sp.simplify(series-expected) == 0
    say("  Incoming direction (dy,dz)=(0,-1); sag z=R-sqrt(R^2-r^2).")
    say("  Reference focus (0,R/2), mirror vertex at z=0.")
    say("  W(r)=-z+sqrt[r^2+(R/2-z)^2]-R/2.")
    say("  Symbolic W=-r^4/(4R^3)+r^6/(8R^5)+O(r^8): PASS")
    say("  R=100 mm, paraxial f=50 mm; ray table:")
    say("   height [mm]   axis crossing z [mm]   intercept at f [mm]   exact W [um]  W4+W6 [um]")
    radius = 100.0
    for height in [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]:
        sag = height**2/(radius+sqrt(radius**2-height**2))
        normal = np.array([-height/radius, sqrt(1-(height/radius)**2)])
        incident = np.array([0., -1.])
        reflected = incident-2*np.dot(incident, normal)*normal
        axis = sag-height*reflected[1]/reflected[0]
        spot = height+(radius/2-sag)*reflected[0]/reflected[1]
        opd = -sag+np.hypot(height, radius/2-sag)-radius/2
        poly = -height**4/(4*radius**3)+height**6/(8*radius**5)
        assert abs(np.linalg.norm(reflected)-1) < 1e-13
        assert abs(axis-(radius-radius**2/(2*sqrt(radius**2-height**2)))) < 1e-10
        assert spot < 0 and opd < 0
        say(f"   {height:8.3f}       {axis:15.9f}         {spot: .9f}        {1000*opd: .9f}  {1000*poly: .9f}")
    height = 0.5
    sag = height**2/(radius+sqrt(radius**2-height**2))
    v = np.array([-2*height*sqrt(radius**2-height**2)/radius**2,
                  1-2*(height/radius)**2])
    exact_spot = height+(radius/2-sag)*v[0]/v[1]
    gradient_prediction = -height**3/(2*radius**2)
    assert abs(exact_spot/gradient_prediction-1) < 1e-4
    say(f"  Leading spot f*dW/dr at r=0.5 mm: {gradient_prediction:+.12e} mm")
    say(f"  Exact spot at r=0.5 mm:          {exact_spot:+.12e} mm")
    say("  The slope relation here is a paraxial leading-order check, not an exact surface-coordinate identity.")
    # Backward continuation of the reflected eikonal to the vertex plane z=0
    # changes both the pupil coordinate and the accumulated optical path.
    # This exposes why higher-order coefficients require a specified plane.
    q, u = sp.symbols("q u", real=True)
    sag_scaled = 1-sp.sqrt(1-q*q)
    vy, vz = -2*q*sp.sqrt(1-q*q), 1-2*q*q
    x_scaled = q-sag_scaled*vy/vz
    x_series = sp.series(x_scaled,q,0,7).removeO().expand()
    assert x_series == q+q**3+sp.Rational(7,4)*q**5
    inverse = u-u**3+sp.Rational(5,4)*u**5
    assert sp.series(x_series.subs(q,inverse)-u,u,0,7).removeO().expand() == 0
    # S(z=0)/R = -sag/R - (sag/R)/vz. Add ideal-sphere reference term.
    W_vertex_scaled = -sag_scaled*(1+1/vz)+sp.sqrt(x_scaled*x_scaled+sp.Rational(1,4))-sp.Rational(1,2)
    W_in_q = sp.series(W_vertex_scaled,q,0,8).removeO()
    W_in_u = sp.series(W_in_q.subs(q,inverse),u,0,8).removeO().expand()
    assert W_in_u == -u**4/4+sp.Rational(9,8)*u**6
    spot_scaled = x_scaled+vy/(2*vz)
    spot_q = sp.series(spot_scaled,q,0,7).removeO()
    spot_u = sp.series(spot_q.subs(q,inverse),u,0,7).removeO().expand()
    assert spot_u == -u**3/2+sp.Rational(3,8)*u**5
    say("  Vertex comparison plane uses reflected rays continued backward to z=0.")
    say("  Its radial coordinate is x=r+r^3/R^2+7r^5/(4R^4)+O(r^7).")
    say("  Its wavefront is W0(x)=-x^4/(4R^3)+9x^6/(8R^5)+O(x^8).")
    say("  The exact ray spot expanded in x is -x^3/(2R^2)+3x^5/(8R^4)+O(x^7).")
    say("  Coordinate inversion and independent vertex-plane series: PASS")
    say("  The surface-footprint and plane-coordinate sixth coefficients differ legitimately.")
    say()

    say("4. A thin spherical phase-screen lens (not a finite-thickness singlet)")
    n = sp.symbols("n", positive=True)
    thickness_change = -2*(R-sp.sqrt(R**2-r**2))
    f_lens = R/(2*(n-1))
    # f_lens>0 is imposed through the explicit n=1.5 example below.
    phase_delay = (n-1)*thickness_change
    say("  Symmetric spherical thickness change: t(r)-t0=-2[R-sqrt(R^2-r^2)].")
    say("  Air phase delay: P(r)=(n-1)[t(r)-t0], f=R/[2(n-1)].")
    say("  Target-focus OPD: W=P(r)+sqrt(f^2+r^2)-f.")
    say("  Quartic coefficient: -(n-1)/(4R^3)-1/(8f^3).")
    say("  Sixth coefficient:   -(n-1)/(8R^5)+1/(16f^5).")
    W_screen = phase_delay.subs(n, sp.Rational(3, 2)) + sp.sqrt(R**2+r**2)-R
    screen_series = sp.series(W_screen, r, 0, 8).removeO().expand()
    assert sp.simplify(screen_series+r**4/(4*R**3)) == 0
    say("  At n=1.5, f=R: W=-r^4/(4R^3)+0*r^6+O(r^8): PASS")
    say("  At R=50 mm, a=5 mm: quartic edge OPD A=-1.250000 um.")
    say("  c40=A/(6sqrt(5))=-0.093169499062 um after piston/defocus removal.")
    say("  This screen omits travel inside a thick lens; section 5 traces that travel explicitly.")
    say()


def refract(direction: np.ndarray, normal_forward: np.ndarray,
            n_before: float, n_after: float) -> np.ndarray:
    """Snell's law with a normal pointing into the transmitted medium.

    Preserve tangential optical momentum n*s_t; choose the forward normal
    component. These vectors are in meridional coordinates (height, z).
    """
    ci = float(np.dot(direction, normal_forward))
    if ci <= 0:
        raise ValueError("Normal must point into transmitted medium.")
    tangent = (n_before/n_after)*(direction-ci*normal_forward)
    normal_squared = 1-float(np.dot(tangent, tangent))
    if normal_squared < 0:
        raise ValueError("Total internal reflection.")
    outgoing = tangent+sqrt(normal_squared)*normal_forward
    assert abs(np.linalg.norm(outgoing)-1) < 1e-12
    assert np.linalg.norm(n_after*(outgoing-np.dot(outgoing,normal_forward)*normal_forward)
                          - n_before*(direction-ci*normal_forward)) < 1e-12
    return outgoing


def forward_sphere_intersection(point: np.ndarray, direction: np.ndarray,
                                centre: np.ndarray, radius: float) -> tuple[np.ndarray,float]:
    delta = point-centre
    b = float(np.dot(delta,direction))
    discriminant = b*b-float(np.dot(delta,delta))+radius*radius
    if discriminant < 0:
        raise ValueError("Ray misses sphere.")
    candidates = [-b-sqrt(discriminant), -b+sqrt(discriminant)]
    forward = [s for s in candidates if s > 1e-10]
    if not forward:
        raise ValueError("No forward intersection.")
    s = min(forward)
    return point+s*direction, s


def singlet_trace(height: float, focal_z: float, exit_z: float = 6.0) -> dict[str,float]:
    """Air -> n=1.5 glass -> air, R1=+50, R2=-50, vertex spacing 5 mm.

    Incident rays start at z=0 with direction +z. Both actual optical path and
    ideal reference path are compared on the SAME output plane z=exit_z.
    W=OPL_to_exit + distance(exit_point,ideal_focus) - OPL_on_axis_to_focus.
    """
    radius, thickness, index = 50.0, 5.0, 1.5
    z1 = height*height/(radius+sqrt(radius*radius-height*height))
    p1 = np.array([height,z1])
    norm1 = (np.array([0.,radius])-p1)/radius
    inside = refract(np.array([0.,1.]), norm1, 1., index)
    centre2 = np.array([0.,thickness-radius])
    p2, glass_length = forward_sphere_intersection(p1,inside,centre2,radius)
    assert thickness-1 < p2[1] <= thickness+1e-12  # physical rear cap
    norm2 = (p2-centre2)/radius
    outgoing = refract(inside,norm2,index,1.)
    air_length = (exit_z-p2[1])/outgoing[1]
    pexit = p2+air_length*outgoing
    opl = z1+index*glass_length+air_length
    distance_to_focus = np.hypot(pexit[0],focal_z-exit_z)
    on_axis_opl_to_focus = index*thickness+(focal_z-thickness)
    opd = opl+distance_to_focus-on_axis_opl_to_focus
    axis = np.nan if height == 0 else p2[1]-p2[0]*outgoing[1]/outgoing[0]
    target_intercept = p2[0]+(focal_z-p2[1])*outgoing[0]/outgoing[1]
    return dict(height=height, second_height=float(p2[0]), second_z=float(p2[1]),
                exit_height=float(pexit[0]), outgoing_y=float(outgoing[0]),
                outgoing_z=float(outgoing[1]), axis=float(axis),
                target_intercept=float(target_intercept), opl=float(opl), opd=float(opd))


def check_thick_lens() -> None:
    say("5. Exact symmetric biconvex singlet: vector Snell trace and common-plane OPD")
    n, t, R1, R2 = 1.5, 5., 50., -50.
    surface1 = np.array([[1.,0.],[-(n-1)/R1,1.]])
    travel = np.array([[1.,t/n],[0.,1.]])
    surface2 = np.array([[1.,0.],[-(1-n)/R2,1.]])
    matrix = surface2@travel@surface1
    A, B, C, D = matrix.ravel()
    efl, bfl = -1/C, -A/C
    focus = t+bfl
    assert abs(np.linalg.det(matrix)-1) < 1e-14
    say("  Coordinates (height,z), propagation +z; lengths mm; ray state (height,n*theta).")
    say("  R1=+50 mm, R2=-50 mm, thickness=5 mm, n=1.5, air on both sides.")
    say(f"  ABCD = [[{A:.12f}, {B:.12f}], [{C:.12f}, {D:.12f}]]")
    say(f"  Effective focal length (EFL): {efl:.12f} mm")
    say(f"  Back focal length (from rear vertex at z=5): {bfl:.12f} mm")
    say(f"  Paraxial focus coordinate: z={focus:.12f} mm")
    say("  Wavefront comparison plane: z=6 mm.")
    say("  W=OPL_to_z6 + sqrt(y6^2+(focus_z-6)^2) - [1.5*5+(focus_z-5)].")
    say("  The exact rays reach their own axis crossings; the added distance to the")
    say("  common ideal focus defines a reference-sphere OPD and is not the ray's continuation.")
    say("  input h    rear h      exit h       ray axis z     spot at focus     W at z6")
    say("     mm        mm          mm             mm              mm             um")
    for h in [0.,0.1,1.,2.,3.,4.,5.]:
        ray = singlet_trace(h,focus)
        say(f"  {h:6.2f}  {ray['second_height']:10.7f}  {ray['exit_height']:10.7f}  "
            f"{ray['axis']:14.9f}  {ray['target_intercept']: .10f}  {1000*ray['opd']: .9f}")
        if h == 0:
            assert abs(ray['opd']) < 1e-11
        else:
            assert ray['axis'] < focus
            assert ray['target_intercept'] < 0
    tiny = singlet_trace(0.001,focus)
    assert abs(tiny['axis']-focus) < 1e-5

    # Direct differential eikonal check on a common output plane. Parameter h
    # is input-pupil radius, while the required gradient is dW/d(exit height).
    # Exact identity: W_y=s_actual,y - s_ideal,y, both in air.
    probe, dh = 3., 1e-3
    left, mid, right = (singlet_trace(probe+delta,focus) for delta in [-dh,0.,dh])
    derivative = (right['opd']-left['opd'])/(right['exit_height']-left['exit_height'])
    distance = np.hypot(mid['exit_height'],focus-6.)
    ideal_y = -mid['exit_height']/distance
    momentum_difference = mid['outgoing_y']-ideal_y
    say(f"  At input h=3 mm, finite-difference dW/dy6={derivative:+.12e}.")
    say(f"  Actual minus ideal transverse direction cosine={momentum_difference:+.12e}.")
    assert abs(derivative-momentum_difference) < 2e-9
    say("  Differential OPD/ray-direction consistency: PASS")

    # Fit a radially symmetric polynomial in the ENTRANCE ray label rho=h/5.
    # These coefficients are not asserted to be exit-pupil Zernike data.
    # All pupil coordinates and weights must be specified when comparing data.
    u,w = leggauss(100)
    u,w = (u+1)/2,w/2
    rays = [singlet_trace(5*sqrt(value),focus) for value in u]
    values = 1000*np.array([ray['opd'] for ray in rays])
    rho = np.sqrt(u)
    Z = np.column_stack([zernike(order,0,rho,np.zeros_like(rho)) for order in (0,2,4,6,8)])
    coeff = Z.T@(w*values)
    residual = values-Z@coeff
    say("  Illustrative pullback: W sampled uniformly in entrance disk rho=h/5.")
    say("  This labels output OPD by entrance rays; it is NOT an exit-pupil fit.")
    for order,value in zip((0,2,4,6,8),coeff):
        say(f"    c({order},0) = {value:+.12f} um")
    say(f"  Fit residual RMS through n=8: {sqrt(w@(residual*residual)):.3e} um")
    say("  Reported c(2,0) includes the defocus projection of higher radial powers")
    say("  even though the reference focus was fixed at the paraxial focus.")
    say()


def main() -> None:
    say("Optics textbook reproducibility checks")
    say("Dependencies: NumPy and SymPy. All checks run locally; no network required.")
    say("Numbers rounded for display; assertions use unrounded values.")
    say()
    check_zernike()
    check_polynomial_balancing()
    check_sphere_and_phase_screen()
    check_thick_lens()
    generate_figures()
    say("ALL CHECKS PASSED")
    output = Path(__file__).resolve().with_name("numerical_checks.txt")
    output.write_text("\n".join(REPORT)+"\n",encoding="utf-8")


def generate_figures() -> None:
    """Optional, reproducible vector figures; numerical checks need no plotting."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        say("Matplotlib is absent: numerical checks passed; optional figures skipped.")
        say()
        return
    directory = Path(__file__).resolve().parent / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "figure.dpi":140, "savefig.facecolor":"white",
                         "axes.titleweight":"semibold"})

    # Values retain the same full-disk RMS normalization in every panel.
    grid = np.linspace(-1,1,241)
    x,y = np.meshgrid(grid,grid)
    rho,theta = np.hypot(x,y),np.arctan2(y,x)
    panels = [(0,0,"Piston"),(1,1,"x tilt"),(1,-1,"y tilt"),
              (2,0,"Defocus"),(2,2,"Cosine astigmatism"),(2,-2,"Sine astigmatism"),
              (3,1,"Cosine coma"),(3,-1,"Sine coma"),(3,3,"Cosine trefoil"),
              (4,0,"Primary spherical"),(4,2,"Secondary astigmatism"),(6,0,"Secondary spherical")]
    fig,axes = plt.subplots(4,3,figsize=(9,10.7),layout="constrained")
    for ax,(n,m,title) in zip(axes.ravel(),panels):
        values = np.ma.masked_where(rho>1,zernike(n,m,rho,theta))
        artist = ax.pcolormesh(x,y,values,cmap="RdBu_r",vmin=-3.2,vmax=3.2,
                              shading="auto",rasterized=True)
        ax.add_patch(plt.Circle((0,0),1,fill=False,color="#424242",linewidth=.7))
        ax.set(aspect="equal",xlim=(-1.06,1.06),ylim=(-1.06,1.06),xticks=[],yticks=[],
               title=f"{title}\n$Z_{{{n}}}^{{{m}}}$")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(artist,ax=axes,shrink=.6,pad=.015,ticks=[-3,-2,-1,0,1,2,3],
                 label="Dimensionless mode value (unit disk RMS = 1)")
    fig.suptitle("Real Zernike modes: signs, azimuths and radial structure",fontsize=15)
    fig.savefig(directory/"zernike_mode_atlas.pdf")
    fig.savefig(directory/"zernike_mode_atlas.png",dpi=130)
    plt.close(fig)

    # The sphere plot deliberately names the surface coordinate r, since the
    # plane-coordinate sixth term is different and verified separately above.
    heights = np.linspace(-5,5,601)
    radius = 100.
    sag = heights**2/(radius+np.sqrt(radius**2-heights**2))
    vy = -2*heights*np.sqrt(radius**2-heights**2)/radius**2
    vz = 1-2*(heights/radius)**2
    spots = heights+(radius/2-sag)*vy/vz
    opd = -sag+np.hypot(heights,radius/2-sag)-radius/2
    A,B = -(5**4)/(4*radius**3),5**6/(8*radius**5)
    norm = heights/5
    balanced = A*(norm**4-norm**2+1/6)+B*(norm**6-.9*norm**2+.2)
    fig,axes = plt.subplots(1,2,figsize=(10.4,4.3),layout="constrained")
    axes[0].plot(heights,1000*opd,color="#2166ac",label="Exact target-focus OPD")
    axes[0].plot(heights,1000*balanced,color="#b2182b",label="Balanced quartic + sixth")
    axes[0].set(xlabel="Signed meridional footprint coordinate (mm)",ylabel="OPD ($\\mu$m)",title="Spherical mirror wavefront")
    axes[0].legend(fontsize=8)
    axes[1].plot(heights,1000*spots,color="#2166ac",label="Exact reflected ray")
    axes[1].plot(heights,-1000*heights**3/(2*radius**2),"--",color="#b2182b",label="Leading cubic")
    axes[1].set(xlabel="Signed meridional footprint coordinate (mm)",ylabel="Intercept at z = 50 mm ($\\mu$m)",title="Transverse ray fan")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(0,color=".6",lw=.6)
        ax.grid(alpha=.18)
    fig.suptitle("Spherical mirror: R = 100 mm, aperture radius = 5 mm",fontsize=13)
    fig.savefig(directory/"spherical_mirror_checks.pdf")
    fig.savefig(directory/"spherical_mirror_checks.png",dpi=130)
    plt.close(fig)

    focus = 5+49.152542372881356
    heights = np.linspace(-5,5,401)
    rays = [singlet_trace(float(h),focus) for h in heights]
    fig,axes = plt.subplots(1,2,figsize=(10.4,4.3),layout="constrained")
    axes[0].plot(heights,[1000*ray["opd"] for ray in rays],color="#2166ac")
    axes[0].set(xlabel="Entrance ray height h (mm)",ylabel="OPD at z = 6 mm ($\\mu$m)",
                title="OPD relative to paraxial focus")
    axes[1].plot(heights,[1000*ray["target_intercept"] for ray in rays],color="#b2182b")
    axes[1].set(xlabel="Entrance ray height h (mm)",ylabel="Intercept at paraxial focus ($\\mu$m)",
                title="Exact refracted-ray fan")
    for ax in axes:
        ax.axhline(0,color=".6",lw=.6)
        ax.grid(alpha=.18)
    fig.suptitle("Biconvex singlet: R1 = +50 mm, R2 = -50 mm, t = 5 mm, n = 1.5",fontsize=12)
    fig.savefig(directory/"thick_singlet_checks.pdf")
    fig.savefig(directory/"thick_singlet_checks.png",dpi=130)
    plt.close(fig)
    say("6. Reproducible illustrations (Matplotlib)")
    say("  figures/zernike_mode_atlas.pdf")
    say("  figures/spherical_mirror_checks.pdf")
    say("  figures/thick_singlet_checks.pdf")
    say("  PNG previews are also written beside the PDF figures.")
    say()


if __name__ == "__main__":
    main()
