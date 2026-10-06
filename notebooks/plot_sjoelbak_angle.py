# -*- coding: utf-8 -*-
"""Plot the angled sjoelbak potential for a range of tilt angles.

Produces two figures next to this script:

* ``sjoelbak_angle_maps.png``     -- the 2D surface at eight angles, including
  every edge case (flat barrier, the corridor-opening threshold, the maximum
  angle at full width, the auto-shrunk regime and a mirrored barrier).
* ``sjoelbak_angle_profiles.png`` -- the projected free energy F(y), the
  effective potential along the ridge, and the crossing-probability estimate
  as a function of angle.

Run with::

    python notebooks/plot_sjoelbak_angle.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential as Pot

HERE = Path(__file__).resolve().parent
KT = 0.05          # temperature of the 2D runs in staple_test.py
# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid.
trapezoid = getattr(np, "trapezoid", None) or np.trapz
INK = "#1a1a1a"
MUTED = "#6b6b6b"
STROKE = [pe.withStroke(linewidth=2.0, foreground="#00000055")]


def grid_eval(pot, xs, ys):
    """Effective potential (the one consistent with the force) on a grid.

    Points outside the box are masked out; the harmonic walls would otherwise
    swamp the colour scale.
    """
    out = np.full((ys.size, xs.size), np.nan)
    for j, y in enumerate(ys):
        if not 0.0 <= y <= pot.Ly:
            continue
        for i, x in enumerate(xs):
            if 0.0 <= x <= pot.Lx:
                out[j, i] = pot.effective_potential((np.array([y, x]), np.zeros(2)))
    return out


def free_energy(pot, ys, nx=400):
    """F(y) = -kT ln int dx exp(-U_eff/kT), the projection onto lambda = y."""
    xs = np.linspace(-0.03, pot.Lx + 0.03, nx)
    out = np.empty(ys.size)
    for i, y in enumerate(ys):
        u = np.array([pot.effective_potential((np.array([y, x]), np.zeros(2)))
                      for x in xs])
        out[i] = -KT * np.log(trapezoid(np.exp(-u / KT), xs))
    return out


def ridge_partition(pot, n=4001):
    """Boltzmann weight integrated along the ridge (the dividing surface)."""
    s, _, u = pot.barrier_top_profile(n)
    return trapezoid(np.exp(-u / KT), s)


def basin_width(pot):
    """int dx exp(-U/kT) at an interface in the A basin.

    The potential there is x-independent inside the channel, so this is the
    channel width plus the two soft-wall tails.  P(lambda_B|lambda_A) is
    Z_ridge divided by *this*, not by Z_ridge alone -- it matters as soon as
    the auto-shrink changes Lx.
    """
    return pot.Lx + np.sqrt(2 * np.pi * KT / pot.k)


def p_cross(pot):
    """TST estimate of the crossing probability, up to an angle-free constant."""
    return ridge_partition(pot) / basin_width(pot)


def tilt_factors(pot):
    """The two analytic factors behind p_cross: crest length and background tilt."""
    grad = pot.gravity + pot.slope_strength_y / (pot.slope_end_y - pot.slope_start_y)
    u = pot.Lx * abs(np.tan(pot.angle)) * grad / KT
    sinh_fac = 1.0 if u < 1e-9 else 2.0 * np.sinh(0.5 * u) / u
    return 1.0 / np.cos(pot.angle), sinh_fac


# ----------------------------------------------------------------------
# Figure 1 - the surface at eight angles
# ----------------------------------------------------------------------
def figure_maps():
    ref = Pot()
    cases = [
        (0.0, "flat ridge, $\\perp$ to $\\lambda$"),
        (ref.angle_gap_deg, "corridor opens"),
        (30.0, "corridor in the $y$ projection"),
        (45.0, "the original sjoelbak, $r=-1$"),
        (ref.angle_threshold_deg, "widest tilt at full $L_x$"),
        (65.0, "past threshold, $L_x$ shrunk"),
        (80.0, "near hard limit, $L_x$ tiny"),
        (-45.0, "negative angle mirrors it"),
    ]

    xs = np.linspace(-0.015, 0.315, 165)
    ys = np.linspace(-0.03, 0.93, 420)

    fig, axes = plt.subplots(4, 2, figsize=(8.8, 7.6), sharex=True, sharey=True)
    vmin, vmax = -0.15, 0.50
    levels = np.arange(-0.15, 0.51, 0.05)

    for ax, (ang, note) in zip(axes.ravel(), cases):
        pot = Pot(angle=ang)
        # plotted with y (the order parameter) horizontal, as in plot_potential
        field = grid_eval(pot, xs, ys).T
        mesh = ax.pcolormesh(ys, xs, field, cmap="viridis", vmin=vmin, vmax=vmax,
                             shading="nearest", rasterized=True)
        ax.contour(ys, xs, np.nan_to_num(field, nan=vmax), levels=levels,
                   colors="white", linewidths=0.3, alpha=0.35)

        # the ridge crest and its two edges
        xl = np.array([0.0, pot.Lx])
        crest = pot.r * xl + pot.b
        half = (pot.L / 4) * np.sqrt(pot.r**2 + 1)
        ax.plot(crest, xl, color="white", lw=1.5, ls="--", path_effects=STROKE)
        for off in (-half, half):
            ax.plot(crest + off, xl, color="white", lw=0.7, ls=":", alpha=0.85)

        # state minima, and the well edges the barrier may not cross
        for y in pot.state_y:
            ax.axvline(y, color="#e8590c", lw=1.2)
        for y in pot.band:
            ax.axvline(y, color="#e8590c", lw=0.7, ls=":", alpha=0.85)

        # the channel walls, and the width that was given up when shrinking
        if pot.shrunk:
            ax.axhspan(pot.Lx, pot.Lx_requested, facecolor="#d9d9d9", alpha=0.55,
                       lw=0, hatch="///", edgecolor="#9a9a9a")
            ax.text(0.885, 0.5 * (pot.Lx + pot.Lx_requested),
                    "$L_x\\!:{:.3f}\\to{:.3f}$".format(pot.Lx_requested, pot.Lx),
                    fontsize=6.5, color=INK, ha="right", va="center")
        for h in (0.0, pot.Lx):
            ax.axhline(h, color=INK, lw=0.9)

        ax.set_title("$\\theta={:.2f}\\degree$  ·  ".format(ang) + note,
                     fontsize=8, color=INK, pad=3, loc="left")
        ax.set_ylim(-0.015, 0.315)
        ax.set_xlim(ys[0], ys[-1])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7, colors=MUTED, length=2)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("$y$   (order parameter $\\lambda$)", fontsize=8, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("$x$", fontsize=8, color=INK)

    fig.subplots_adjust(left=0.085, right=0.99, top=0.895, bottom=0.145,
                        hspace=0.36, wspace=0.07)

    cax = fig.add_axes([0.085, 0.052, 0.33, 0.014])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label("effective potential  $U+gy$   (inside the box only)",
                   fontsize=7.5, color=INK, labelpad=2)
    cbar.ax.tick_params(labelsize=7, colors=MUTED, length=2)

    handles = [
        Line2D([], [], color="white", lw=1.6, ls="--", label="barrier crest",
               path_effects=[pe.withStroke(linewidth=3.2, foreground="#3d3d3d")]),
        Line2D([], [], color="#9a9a9a", lw=0.9, ls=":", label="ridge edge ($D=L/4$)"),
        Line2D([], [], color="#e8590c", lw=1.2, label="state A / B minimum"),
        Line2D([], [], color="#e8590c", lw=0.7, ls=":", label="well edge = barrier limit"),
        Patch(facecolor=MUTED, alpha=0.18, hatch="///", edgecolor=MUTED,
              label="width given up by shrinking"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=2, fontsize=7,
               frameon=False, bbox_to_anchor=(0.995, 0.012),
               columnspacing=1.2, handlelength=1.8)

    fig.suptitle("Sjoelbak barrier vs. tilt angle\n"
                 "walls, states, barrier height $A$ and thickness $L$ all held fixed",
                 fontsize=10.5, color=INK, x=0.085, ha="left", y=0.975)
    out = HERE / "sjoelbak_angle_maps.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 2 - what the tilt does to the barrier
# ----------------------------------------------------------------------
def figure_profiles():
    ref = Pot()
    angles = [0.0, ref.angle_gap_deg, 30.0, 45.0, 55.0, ref.angle_threshold_deg]
    ramp = plt.get_cmap("viridis")(np.linspace(0.05, 0.78, len(angles)))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    ax1, ax2, ax3 = axes

    # (a) projected free energy along the order parameter
    ys = np.linspace(0.08, 0.82, 260)
    for ang, col in zip(angles, ramp):
        f = free_energy(Pot(angle=ang), ys) / KT
        ax1.plot(ys, f - f[0], color=col, lw=2.0)
    ax1.set_title("(a)  the $y$ projection hides the tilt", fontsize=9.5,
                  color=INK, loc="left")
    ax1.set_xlabel("$y$", fontsize=8.5, color=INK)
    ax1.set_ylabel("$F(y)/k_BT$", fontsize=8.5, color=INK)
    ax1.annotate("only $\\theta=0$ shows the ridge.\nOnce a corridor opens, every\n"
                 "tilt gives the same profile,\npeaking at the top of the\n"
                 "background ramp - not at\nthe barrier.",
                 xy=(0.03, 0.97), xycoords="axes fraction", fontsize=7.5,
                 color=MUTED, va="top", linespacing=1.45)

    # (b) effective potential along the ridge itself
    for ang, col in zip(angles, ramp):
        s, _, u = Pot(angle=ang).barrier_top_profile(400)
        ax2.plot(s, u, color=col, lw=2.0)
        ax2.plot([s[np.argmin(u)]], [u.min()], "o", ms=5.5, color=col,
                 mec="white", mew=1.2, zorder=3)
    ax2.set_title("(b)  the crest lengthens, its cheapest point sinks",
                  fontsize=9.5, color=INK, loc="left")
    ax2.set_xlabel("arc length along the crest, from $x=0$", fontsize=8.5, color=INK)
    ax2.set_ylabel("$U_{\\mathrm{eff}}$ on the crest", fontsize=8.5, color=INK)
    ax2.annotate("markers: cheapest crossing point.\nIt slides to the low-$y$ wall, "
                 "and\nsinks 2.6 $k_BT$ from $0\\degree$ to $61\\degree$.",
                 xy=(0.03, 0.10), xycoords="axes fraction", fontsize=7.5,
                 color=MUTED, linespacing=1.45)

    # (c) crossing probability estimate over the whole admissible range
    scan = np.concatenate([np.linspace(0.0, ref.angle_threshold_deg, 46),
                           np.linspace(ref.angle_threshold_deg, 81.6, 30)[1:]])
    base = p_cross(Pot(angle=0.0))
    full, shrunk, fac_len, fac_tilt = [], [], [], []
    for ang in scan:
        pot = Pot(angle=ang)
        (shrunk if pot.shrunk else full).append((ang, p_cross(pot) / base))
        a, b = tilt_factors(pot)
        fac_len.append((ang, a))
        fac_tilt.append((ang, b))
    full, shrunk = np.array(full), np.array(shrunk)
    fac_len, fac_tilt = np.array(fac_len), np.array(fac_tilt)

    ax3.plot(fac_len[:, 0], fac_len[:, 1], color="#5a5a5a", lw=1.1)
    ax3.plot(fac_tilt[:, 0], fac_tilt[:, 1], color="#5a5a5a", lw=1.1, ls=(0, (4, 2)))
    ax3.plot(full[:, 0], full[:, 1], color="#1f6feb", lw=2.4)
    ax3.plot(shrunk[:, 0], shrunk[:, 1], color="#1f6feb", lw=2.4, ls="--")
    ax3.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    ax3.set_yscale("log")
    ax3.set_xlim(-2, 88)
    ax3.set_ylim(0.55, 9.0)
    ax3.set_yticks([1, 2, 3, 5, 8])
    ax3.set_yticklabels(["1", "2", "3", "5", "8"])
    for ang, lab, col, at_top in [
            (ref.angle_gap_deg, "corridor opens", "#e8590c", True),
            (ref.angle_threshold_deg, "$L_x$ must shrink", "#c05621", True),
            (ref.angle_hard_limit_deg, "hard limit", "#9b2c2c", True)]:
        ax3.axvline(ang, color=col, lw=1.0)
        ax3.annotate("{}  {:.2f}$\\degree$".format(lab, ang),
                     (ang, 8.7 if at_top else 0.575),
                     xytext=(-4, 0), textcoords="offset points", rotation=90,
                     fontsize=7.5, color=col,
                     va="top" if at_top else "bottom", ha="right")
    ax3.set_title("(c)  crossing probability is not conserved", fontsize=9.5,
                  color=INK, loc="left")
    ax3.set_xlabel("barrier angle $\\theta$  (degrees)", fontsize=8.5, color=INK)
    ax3.set_ylabel("$P_{\\mathrm{cross}}(\\theta)\\,/\\,P_{\\mathrm{cross}}(0\\degree)$",
                   fontsize=8.5, color=INK)

    for ax in axes:
        ax.tick_params(labelsize=7.5, colors=MUTED, length=2)
        ax.grid(True, color="#ececec", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#cfcfcf")

    angle_handles = [Line2D([], [], color=c, lw=2.2,
                            label="$\\theta={:.1f}\\degree$".format(a))
                     for a, c in zip(angles, ramp)]
    c_handles = [
        Line2D([], [], color="#1f6feb", lw=2.4, label="$P_{\\mathrm{cross}}$, $L_x=0.3$ kept"),
        Line2D([], [], color="#1f6feb", lw=2.4, ls="--", label="$P_{\\mathrm{cross}}$, $L_x$ auto-shrunk"),
        Line2D([], [], color="#5a5a5a", lw=1.1, label="factor: crest length $1/\\cos\\theta$"),
        Line2D([], [], color="#5a5a5a", lw=1.1, ls=(0, (4, 2)),
               label="factor: background tilt"),
    ]
    fig.legend(handles=c_handles, loc="upper left", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.645, 0.945), title="panel (c)",
               title_fontsize=8, alignment="left", columnspacing=1.4)
    fig.legend(handles=angle_handles, loc="upper left", ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.006, 0.945),
               title="panels (a) and (b)", title_fontsize=8, alignment="left")

    fig.suptitle("Why the tilt changes the physics   ($k_BT=0.05$, the 2D "
                 "setting in staple_test.py)",
                 fontsize=10.5, color=INK, x=0.006, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.865))
    out = HERE / "sjoelbak_angle_profiles.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", figure_maps())
    print("wrote", figure_profiles())
