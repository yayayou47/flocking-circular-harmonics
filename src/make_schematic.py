"""Schema de la regle de mise a jour a deux zones (Fig. 1 du manuscrit).

Extrait du generateur de figures de la version longue du projet, reduit
ici a la seule figure utilisee par l'article.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style

FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
style.apply()


def _save(fig, name: str):
    out = FIG / name
    fig.savefig(out)
    plt.close(fig)
    print(f"saved: {out}")


def fig_model_schematic():
    """Schematic of the two-zone Vicsek update rule (no blind sector)."""
    from matplotlib.patches import Wedge, Patch

    # Zones enlarged by 20%; neighbours and arrows scaled to keep
    # categorisation correct. R_r = 0.45 is the canonical operating
    # point of the manuscript (set by the L=22 chi-peak scan).
    R_r = 0.45 * 1.2
    R_a = 0.7 * 1.2

    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    rep_color = "#e07b7b"
    ali_color = "#9bb8de"

    # Full 360 deg vision: both wedges span the entire disk.
    rep = Wedge((0, 0), R_r, 0, 360,
                facecolor=rep_color, alpha=0.55,
                edgecolor="#9c3a3a", lw=0.9, zorder=1)
    ali = Wedge((0, 0), R_a, 0, 360, width=R_a - R_r,
                facecolor=ali_color, alpha=0.55,
                edgecolor="#3a4a78", lw=0.9, zorder=1)
    for patch in (rep, ali):
        ax.add_patch(patch)

    # Focal particle i: arrow at origin pointing +x. All interior
    # elements (circles, arrowheads, labels, vectors) are sized 20%
    # larger than the previous schematic for legibility at 0.33 line
    # width.
    head_len = 0.20 * 1.2 * 1.2
    ax.annotate("", xy=(head_len, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=style.PARTICLE_BLUE,
                                lw=3.1), zorder=6)
    ax.scatter([0], [0], s=108, color=style.PARTICLE_BLUE,
               edgecolor="white", lw=0.96, zorder=7)
    ax.text(0.05, -0.13, r"$i$", fontsize=16, fontweight="bold", zorder=7)
    ax.text(head_len + 0.02, 0.05, r"$\vec e_i(t)$", fontsize=14,
            color=style.PARTICLE_BLUE, zorder=7)

    arrow_len = 0.14 * 1.2 * 1.2

    def neighbour(x, y, theta_deg, label, color=style.PARTICLE_BLUE,
                  alpha=1.0):
        th = np.deg2rad(theta_deg)
        ax.annotate("",
                    xy=(x + arrow_len * np.cos(th), y + arrow_len * np.sin(th)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.92,
                                    alpha=alpha), zorder=6)
        ax.scatter([x], [y], s=50, color=color, alpha=alpha,
                   edgecolor="white", lw=0.72, zorder=7)
        ax.text(x + 0.05, y + 0.08, label, fontsize=13, alpha=alpha,
                zorder=7)

    # Repulsion neighbour: triggers a turn-away vector.
    j1 = (-0.18 * 1.2, 0.22 * 1.2)
    neighbour(*j1, theta_deg=90, label=r"$j_1$")
    nrm = np.hypot(*j1)
    away = (-j1[0] / nrm * 0.30 * 1.2 * 1.2, -j1[1] / nrm * 0.30 * 1.2 * 1.2)
    ax.annotate("", xy=away, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#9c3a3a", lw=1.8,
                                ls=(0, (3, 2))), zorder=6)
    ax.text(away[0] + 0.02, away[1] - 0.06, "repulse", fontsize=12,
            color="#9c3a3a", style="italic", zorder=7)

    # Alignment neighbours (no longer restricted to a forward cone).
    neighbour(0.46 * 1.2, 0.42 * 1.2, theta_deg=20, label=r"$j_2$")
    neighbour(-0.50 * 1.2, -0.34 * 1.2, theta_deg=200, label=r"$j_3$")
    neighbour(-0.55 * 1.2, 0.06 * 1.2, theta_deg=0, label=r"$j_4$")

    # Outside-R_a neighbour: position out of perception range.
    neighbour(0.85 * 1.2, 0.55 * 1.2, theta_deg=60, label=r"$j_\infty$",
              color="#666", alpha=0.55)

    # Radius labels.
    ax.annotate(r"$R_r$",
                xy=(R_r * np.cos(np.deg2rad(-50)),
                    R_r * np.sin(np.deg2rad(-50))),
                xytext=(0.66, -1.14), fontsize=14,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))
    ax.annotate(r"$R_a$",
                xy=(R_a * np.cos(np.deg2rad(-30)),
                    R_a * np.sin(np.deg2rad(-30))),
                xytext=(1.26, -0.74), fontsize=14,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))

    # Legend (zone colours).
    handles = [
        Patch(facecolor=rep_color, alpha=0.55, edgecolor="#9c3a3a",
              label=r"Repulsion ($d<R_r$)"),
        Patch(facecolor=ali_color, alpha=0.55, edgecolor="#3a4a78",
              label=r"Alignment ($R_r \leq d < R_a$)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=12,
              framealpha=0.92)

    ax.set_xlim(-1.86, 1.86)
    ax.set_ylim(-1.44, 1.44)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"(a) Two-zone Vicsek update "
                 r"(full $360^\circ$ vision)",
                 fontsize=14)

    fig.tight_layout()
    _save(fig, "fig_model_schematic.pdf")


FOCAL_RED = "#c83a3a"
REP_COLOR = "#e07b7b"
ALI_COLOR = "#9bb8de"

R_R_SCHEMA = 0.45 * 1.2   # zone radii used in panel (a)
R_A_SCHEMA = 0.7  * 1.2
V_VIS      = 0.40          # visualisation step: 8 x v_0 (= 0.05); the
                           # angular update is the standard rule, only
                           # the displacement is inflated for legibility


if __name__ == "__main__":
    fig_model_schematic()
