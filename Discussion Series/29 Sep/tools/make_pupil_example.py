"""Draw the physical stop and its two apparent openings through positive lenses.

All distances are in mm, and paraxial ray slopes are dy/dz. This script
produces vector PDF figures and verifies the constructions by ray matrices.
"""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path as PlotPath

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
INK, RUST = "#17354c", "#b05d25"
EP_COLOR, XP_COLOR = "#7955a2", "#087e8b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                     "axes.titlesize": 12, "pdf.fonttype": 42,
                     "savefig.bbox": "tight"})

Z_OBJECT, Z_L1, Z_STOP, Z_L2 = -160., 0., 40., 100.
F1, F2, R_STOP, R_LENS = 80., 100., 2., 6.
Z_EP, R_EP, Z_XP, R_XP = 80., 4., -50., 5.
Z_IMAGE, MAG = 137.5, -5/8


def translate(distance):
    return np.array([[1., distance], [0., 1.]])


def lens(focal_length):
    return np.array([[1., 0.], [-1/focal_length, 1.]])


TO_STOP = translate(Z_STOP-Z_L1) @ lens(F1) @ translate(Z_L1-Z_OBJECT)
TO_IMAGE = (translate(Z_IMAGE-Z_L2) @ lens(F2)
            @ translate(Z_L2-Z_L1) @ lens(F1) @ translate(Z_L1-Z_OBJECT))


def ray(object_height, stop_height):
    # Solve for the launch slope from the desired stop crossing, using ABCD.
    slope = (stop_height-TO_STOP[0, 0]*object_height)/TO_STOP[0, 1]
    initial = np.array([object_height, slope])
    before1 = translate(Z_L1-Z_OBJECT) @ initial
    after1 = lens(F1) @ before1
    at_stop = translate(Z_STOP-Z_L1) @ after1
    before2 = translate(Z_L2-Z_STOP) @ at_stop
    after2 = lens(F2) @ before2
    at_image = translate(Z_IMAGE-Z_L2) @ after2
    return {
        "z": np.array([Z_OBJECT, Z_L1, Z_STOP, Z_L2, Z_IMAGE]),
        "y": np.array([object_height, before1[0], at_stop[0], before2[0], at_image[0]]),
        "input_slope": slope, "output_slope": after2[1],
        "input_at_EP": before1[0]+(Z_EP-Z_L1)*slope,
        "output_at_XP": after2[0]+(Z_XP-Z_L2)*after2[1],
    }


def draw():
    """Make pupil size and position prominent within two sparse ray diagrams."""
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.4))
    fig.subplots_adjust(left=.02, right=.99, bottom=.025, top=.99, hspace=.045)

    def hardware(ax, second=False):
        ax.set(xlim=(-184, 164), ylim=(-9.2, 10.8))
        ax.axis("off")
        ax.plot([-176, 153], [0, 0], color="#d2d7da", lw=.65,
                ls=(0, (4, 4)), zorder=0)
        for z, f, index in ((Z_L1, F1, 1), (Z_L2, F2, 2)):
            vertices = [
                (z, -R_LENS), (z-3.0, -3.6), (z-3.0, 3.6), (z, R_LENS),
                (z+3.0, 3.6), (z+3.0, -3.6), (z, -R_LENS), (z, -R_LENS),
            ]
            codes = [PlotPath.MOVETO] + [PlotPath.CURVE4]*6 + [PlotPath.CLOSEPOLY]
            ax.add_patch(PathPatch(PlotPath(vertices, codes),
                                  facecolor="#f6f7f8", edgecolor="#c0c8ce",
                                  lw=1.0, zorder=1))
            label = (fr"$L_{index}$: $f={f:g}$ mm" + "\n" + fr"$z={z:g}$ mm"
                     if not second else fr"$L_{index}$")
            ax.text(z, -7.25, label, ha="center", va="top",
                    fontsize=10, color="#858e95")

        # Circular aperture outlines, shown obliquely. Fills lie behind rays.
        for z, radius, edge, fill, title in (
            (Z_XP, R_XP, XP_COLOR, "#eaf6f5", "Exit pupil (XP)"),
            (Z_STOP, R_STOP, "#39454d", "#f7f7f7", "Stop"),
            (Z_EP, R_EP, EP_COLOR, "#f3eef9", "Entrance pupil (EP)"),
        ):
            virtual = z != Z_STOP
            ax.add_patch(Ellipse((z, 0), width=5.2*radius, height=2*radius,
                                 facecolor=fill, edgecolor="none", zorder=1.5))
            ax.add_patch(Ellipse((z, 0), width=5.2*radius, height=2*radius,
                                 facecolor="none", edgecolor=edge,
                                 lw=2.2, linestyle=(0, (4, 3)) if virtual else "-",
                                 zorder=6))
            if virtual:
                ax.text(z, 8.6, title, ha="center", va="top", color=edge,
                        fontsize=13.5, fontweight="bold")
                ax.text(z, 7.05, fr"$D={2*radius:g}$ mm,  $z={z:g}$ mm",
                        ha="center", va="top", color=edge, fontsize=11)
            else:
                ax.text(z, 5.85, title, ha="center", va="top", color=edge,
                        fontsize=12, fontweight="bold")
                ax.text(z, -4.9, fr"$D={2*radius:g}$ mm" + "\n" + fr"$z={z:g}$ mm",
                        ha="center", va="top", color=edge, fontsize=11)

    def bundle(ax, height, *, extensions, faint=False):
        ray_color = "#bfc6cb" if faint else "#757f88"
        ray_width = .8 if faint else 1.05
        for stop_y in (-R_STOP, R_STOP):
            r = ray(height, stop_y)
            ax.plot(r["z"], r["y"], color=ray_color, lw=ray_width, zorder=2.5)
            if extensions:
                # The input and output straight-line segments locate EP and XP.
                ax.plot([Z_L1, Z_EP], [r["y"][1], r["input_at_EP"]],
                        color=EP_COLOR, lw=1.05, ls=(0, (3, 3)), alpha=.8, zorder=3)
                ax.plot([Z_XP, Z_L2], [r["output_at_XP"], r["y"][3]],
                        color=XP_COLOR, lw=1.05, ls=(0, (3, 3)), alpha=.8, zorder=3)
                ax.plot(Z_EP, r["input_at_EP"], "o", ms=3.5, color=EP_COLOR, zorder=7)
                ax.plot(Z_XP, r["output_at_XP"], "o", ms=3.5, color=XP_COLOR, zorder=7)
        ax.plot(Z_OBJECT, height, "o", ms=3.7, color=ray_color, zorder=5)
        ax.plot(Z_IMAGE, MAG*height, "o", ms=3.0, color=ray_color, zorder=5)
        ax.annotate("on-axis source" if height == 0 else "off-axis source",
                    xy=(Z_OBJECT, height), xytext=(-3, -13 if height == 0 else 8),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=10, color="#838c93" if faint else "#606b74")

    hardware(axes[0])
    axes[0].text(-179, 10.6, "(a) One on-axis source", ha="left", va="top",
                 fontsize=11, color="#4d5860")
    bundle(axes[0], 0., extensions=True)

    hardware(axes[1], second=True)
    axes[1].text(-179, 10.6, "(b) Add an off-axis source", ha="left", va="top",
                 fontsize=11, color="#4d5860")
    bundle(axes[1], 0., extensions=False, faint=True)
    bundle(axes[1], 6., extensions=True)

    fig.savefig(FIG / "two_lens_pupils.pdf")
    fig.savefig(FIG / "two_lens_pupils.png", dpi=180)
    plt.close(fig)


def verify():
    # Locate pupil images independently from the ray matrices: require B=0.
    ep_object_distance = Z_STOP-Z_L1
    xp_object_distance = Z_L2-Z_STOP
    ep_image_distance = 1/(1/F1-1/ep_object_distance)
    xp_image_distance = 1/(1/F2-1/xp_object_distance)
    ep_matrix = translate(ep_image_distance) @ lens(F1) @ translate(ep_object_distance)
    xp_matrix = translate(xp_image_distance) @ lens(F2) @ translate(xp_object_distance)
    np.testing.assert_allclose([ep_matrix[0, 1], xp_matrix[0, 1]], 0, atol=1e-12)
    np.testing.assert_allclose([Z_L1-ep_image_distance, Z_L2+xp_image_distance], [Z_EP, Z_XP])
    np.testing.assert_allclose([ep_matrix[0, 0]*R_STOP, xp_matrix[0, 0]*R_STOP], [R_EP, R_XP])
    np.testing.assert_allclose(TO_IMAGE[0], [MAG, 0.], atol=1e-12)
    max_ep_error = max_xp_error = max_image_error = 0.
    max_lens_heights = np.zeros(2)
    for height in (0., 6.):
        for stop_y in np.linspace(-R_STOP, R_STOP, 101):
            r = ray(height, stop_y)
            max_ep_error = max(max_ep_error, abs(r["input_at_EP"]-2*stop_y))
            max_xp_error = max(max_xp_error, abs(r["output_at_XP"]-2.5*stop_y))
            max_image_error = max(max_image_error, abs(r["y"][-1]-MAG*height))
            max_lens_heights = np.maximum(max_lens_heights, np.abs(r["y"][[1, 3]]))
    np.testing.assert_allclose([max_ep_error, max_xp_error, max_image_error], 0, atol=1e-12)
    assert np.all(max_lens_heights < R_LENS), "Lens rim clips an admitted stop ray"
    results = {
        "entrance_pupil": {"z_mm": Z_EP, "radius_mm": R_EP, "virtual": True},
        "exit_pupil": {"z_mm": Z_XP, "radius_mm": R_XP, "virtual": True},
        "object_plane_z_mm": Z_OBJECT, "image_plane_z_mm": Z_IMAGE,
        "lateral_magnification": MAG, "off_axis_source_height_mm": 6.,
        "off_axis_image_height_mm": MAG*6.,
        "entrance_cone_half_angle_rad_paraxial": R_EP/(Z_EP-Z_OBJECT),
        "exit_cone_half_angle_rad_paraxial": R_XP/(Z_IMAGE-Z_XP),
        "off_axis_input_chief_slope": ray(6., 0.)["input_slope"],
        "off_axis_output_chief_slope": ray(6., 0.)["output_slope"],
        "max_height_at_lenses_mm": max_lens_heights.tolist(),
        "lens_clear_radius_mm": R_LENS,
        "max_pupil_image_and_source_image_error_mm": max(max_ep_error, max_xp_error, max_image_error),
        "object_to_image_ABCD": TO_IMAGE.tolist(),
    }
    (ROOT / "pupil_example_checks.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    verify()
    draw()
