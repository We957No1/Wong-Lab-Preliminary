"""Reproduce the Airy and overlap figures; check the sinusoidal-mask calculation.

Run with Python, NumPy, SciPy, and Matplotlib. All plotting coordinates are
dimensionless unless stated. No network access or original source folder is
needed; coherence_comparison.pdf is supplied with the note.
"""
from pathlib import Path
import json

import numpy as np
from scipy.integrate import quad
from scipy.special import j1, jn_zeros
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
INK, TEAL, RUST = "#17354c", "#087e8b", "#a84429"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 10.5, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=180)
    plt.close(fig)


def airy(r):
    v = 2 * np.pi * np.asarray(r, dtype=float)
    amplitude = np.ones_like(v)
    np.divide(2 * j1(v), v, out=amplitude, where=(v != 0))
    return amplitude**2


def otf(q):
    nu = np.clip(np.abs(np.asarray(q, dtype=float)) / 2, 0, 1)
    return 2 / np.pi * (np.arccos(nu) - nu * np.sqrt(1 - nu**2))


def draw_airy():
    first_zero = jn_zeros(1, 1)[0] / (2 * np.pi)
    first_peak = jn_zeros(2, 1)[0] / (2 * np.pi)
    coordinates = np.linspace(-2.4, 2.4, 801)
    xx, yy = np.meshgrid(coordinates, coordinates)
    intensity = airy(np.hypot(xx, yy))
    r = np.linspace(0, 2.5, 6001)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.4), layout="constrained",
                             gridspec_kw={"height_ratios": [1.25, .85]})
    for ax, logarithmic, panel in zip(axes[0], (False, True), ("a", "b")):
        options = {"norm": LogNorm(1e-5, 1)} if logarithmic else {"vmin": 0, "vmax": 1}
        im = ax.imshow(intensity, origin="lower", extent=(-2.4, 2.4, -2.4, 2.4),
                       interpolation="nearest", cmap="magma", **options)
        ax.set(title=f"({panel}) Image plane: {'logarithmic' if logarithmic else 'linear'} intensity",
               xlabel=r"$x/(\lambda/\mathrm{NA})$", ylabel=r"$y/(\lambda/\mathrm{NA})$",
               xticks=(-2, -1, 0, 1, 2), yticks=(-2, -1, 0, 1, 2))
        if not logarithmic:
            ax.add_patch(Circle((0, 0), first_zero, fill=False, ec="white", lw=.7, ls="--"))
        ax.annotate("first dark ring", xy=(-first_zero/np.sqrt(2), first_zero/np.sqrt(2)),
                    xytext=(-2.15, 1.9), color="white", fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": "white", "lw": .8})
        bar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=.05, fraction=.055,
                          ticks=(1e-5, 1e-3, 1e-1, 1) if logarithmic else (0, .25, .5, .75, 1))
        bar.set_label(r"$I/I(0)$" + (" (log scale)" if logarithmic else " (linear scale)"))

    for ax in axes[1]:
        ax.plot(r, airy(r), color=TEAL, lw=1.6)
        ax.axvline(first_zero, color=RUST, ls="--", lw=1)
        ax.set(xlim=(0, 2.5), xlabel=r"Radius $r/(\lambda/\mathrm{NA})$", ylabel=r"$I(r)/I(0)$")
    axes[1, 0].set(title="(c) Radial intensity: linear scale", ylim=(-.025, 1.05))
    axes[1, 0].axvspan(0, first_zero, color=TEAL, alpha=.08)
    axes[1, 0].annotate(r"$r_0=0.6098\,\lambda/\mathrm{NA}$", xy=(first_zero, 0),
                        xytext=(.88, .45), color=RUST, fontsize=9,
                        arrowprops={"arrowstyle": "->", "color": RUST, "lw": .8})
    axes[1, 1].set(title="(d) Same radial intensity: logarithmic scale",
                   yscale="log", ylim=(1e-5, 1.4))
    axes[1, 1].grid(axis="y", alpha=.15)
    axes[1, 1].plot(first_peak, airy(first_peak), "o", color=RUST, ms=4)
    axes[1, 1].annotate("first bright ring: 1.75%", xy=(first_peak, airy(first_peak)),
                        xytext=(.95, .2), color=RUST, fontsize=9,
                        arrowprops={"arrowstyle": "->", "color": RUST, "lw": .8})
    save(fig, "airy_linear_and_log")
    return float(first_zero), float(airy(first_peak))


def draw_overlap():
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.75), layout="constrained")
    for ax, q in zip(axes, (0, 1.5, 2.2)):
        left, right = Circle((-q/2, 0), 1), Circle((q/2, 0), 1)
        ax.add_patch(Circle((-q/2, 0), 1, fc=INK, alpha=.10, ec="none"))
        ax.add_patch(Circle((q/2, 0), 1, fc=TEAL, alpha=.10, ec="none"))
        overlap = Circle((-q/2, 0), 1, fc=TEAL, alpha=.55, ec="none")
        ax.add_patch(overlap)
        overlap.set_clip_path(right.get_path(), right.get_transform() + ax.transData)
        for cx, color in ((-q/2, INK), (q/2, TEAL)):
            ax.add_patch(Circle((cx, 0), 1, fill=False, ec=color, lw=1.4))
        ax.plot((-q/2, q/2), (0, 0), ".", color=INK, ms=4)
        if q > 0:
            ax.annotate("", xy=(-q/2, -1.28), xytext=(q/2, -1.28),
                        arrowprops={"arrowstyle": "<->", "color": INK, "lw": .9})
            ax.text(0, -1.48, r"separation $q$", ha="center", va="center", fontsize=9)
        else:
            ax.annotate("", xy=(0, 0), xytext=(.8, .6),
                        arrowprops={"arrowstyle": "<->", "color": INK, "lw": .9})
            ax.text(.15, .55, r"$f_c$", fontsize=10, color=INK)
        ax.set(aspect="equal", xlim=(-2.2, 2.2), ylim=(-1.72, 1.2),
               title=fr"$q={q:g}f_c$: $T(q)={float(otf(q)):.4f}$")
        ax.axis("off")
    save(fig, "pupil_overlap")


def check_example():
    # Integrate the source area admitting each pair of diffraction orders.
    # This is independent of the analytic disk-overlap OTF formula.
    t0, eps, q, source_radius = .5, .05, 1.5, 3.
    frequencies = np.array([-q, 0, q])
    coefficients = np.array([eps/2, t0, eps/2])
    kernel = np.zeros((3, 3))
    for i, fi in enumerate(frequencies):
        for j, fj in enumerate(frequencies):
            low = max(-source_radius, -fi-1, -fj-1)
            high = min(source_radius, -fi+1, -fj+1)
            if high <= low:
                continue
            def height(sx):
                y2 = min(source_radius**2-sx**2, 1-(sx+fi)**2, 1-(sx+fj)**2)
                return 2 * np.sqrt(max(0., y2))
            split = -(fi+fj)/2
            kernel[i, j] = quad(height, low, high,
                                points=[split] if low < split < high else None,
                                epsabs=1e-12, epsrel=1e-12)[0] / np.pi
    cycles = np.linspace(0, 3, 1201)
    phases = np.exp(2j*np.pi*frequencies[:, None]*cycles[None, :]/q)
    direct = np.einsum("i,j,ij,ix,jx->x", coefficients, coefficients, kernel,
                       phases, phases.conj()).real
    prediction = t0**2 + eps**2/2 + 2*t0*eps*otf(q)*np.cos(2*np.pi*cycles)
    error = float(np.max(np.abs(direct-prediction)))
    if error > 1e-10:
        raise AssertionError(f"Source integral disagrees with intensity-OTF result: {error}")
    return {
        "coherent_cutoff_per_um": .05/.5,
        "mask_period_um": 1/(q*.05/.5),
        "otf_at_1_5_fc": float(otf(q)),
        "coherent_intensity": t0**2,
        "incoherent_mean": t0**2+eps**2/2,
        "incoherent_cosine_amplitude": float(2*t0*eps*otf(q)),
        "incoherent_Michelson_contrast": float(2*t0*eps*otf(q)/(t0**2+eps**2/2)),
        "source_integral_max_abs_error": error,
        "source_pair_kernel": kernel.tolist(),
        "uniform_mask_throughput_at_equal_incident_power": 1/source_radius**2,
    }


if __name__ == "__main__":
    zero, ring = draw_airy()
    draw_overlap()
    results = check_example()
    results.update({"airy_first_zero_lambda_over_NA": zero,
                    "airy_first_zero_nm_at_13_5_nm_NA_0_33": zero*13.5/.33,
                    "airy_first_bright_ring_relative_intensity": ring})
    (ROOT / "numerical_checks.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
