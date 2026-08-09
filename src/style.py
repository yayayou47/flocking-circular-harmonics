"""Shared matplotlib style for journal figures.

PRE/PRL-style sizing: single-column 3.4 in, two-column 7.0 in. Cream
background applied to figure and axes. Tick/label sizes kept at
8-9 pt so text remains legible at print size.
"""
from __future__ import annotations

import matplotlib as mpl

CREAM = "#FFF8E1"
PARTICLE_BLUE = "#1f4ea1"
WHITE = "#FFFFFF"

#: Palette Wong (2011), colorblind-safe. Convention CLAUDE.md §2.2 pour
#: toute figure d'analyse (courbes, heatmaps, distributions).
WONG = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "sky":       "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
}
WONG_CYCLE = [WONG[k] for k in
              ("blue", "vermilion", "green", "orange",
               "purple", "sky", "black", "yellow")]

SINGLE_COL = (3.4, 2.6)
DOUBLE_COL = (7.0, 3.0)
SQUARE = (3.4, 3.4)


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "savefig.facecolor": CREAM,
        "savefig.edgecolor": CREAM,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    })


def apply_analysis() -> None:
    """Style des figures d'ANALYSE (courbes, heatmaps, distributions).

    Convention CLAUDE.md §2.2 : fond blanc, palette Wong, serif 9 pt,
    ticks majeurs 0.7 pt et mineurs 0.5 pt visibles. Les snapshots de
    particules gardent le fond crème via apply().
    """
    import matplotlib as mpl
    from cycler import cycler

    apply()
    mpl.rcParams.update({
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.edgecolor": WHITE,
        "axes.prop_cycle": cycler(color=WONG_CYCLE),
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "axes.grid": False,
    })
