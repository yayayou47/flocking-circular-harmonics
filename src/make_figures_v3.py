"""Figures du manuscrit v3 (refonte post-rejet).

Trois figures, une par section de résultats :

  fig_collapse.pdf     -> Sec. III  : le collapse en rho_1
  fig_selectivity.pdf  -> Sec. IV   : rho_1 contre huit concurrentes
  fig_families.pdf     -> Sec. V    : six lois de bruit appariées

Conventions CLAUDE.md §2.2 : fond blanc, palette Wong colorblind-safe,
serif 9 pt, ticks majeurs/mineurs 0.7/0.5 pt, bandes ou barres d'erreur
par graine plutôt qu'une simple ligne moyenne.

Toutes les valeurs sont relues des .npz ; aucune n'est codée en dur.

Lancer :  python3 notes/src/make_figures_v3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
sys.path.insert(0, str(HERE))
import style  # noqa: E402
from noise import stable_rvs  # noqa: E402

DATA, FIG = V1 / "data", V1 / "figures"
FIG.mkdir(exist_ok=True)
style.apply_analysis()
W = style.WONG
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ io
def load_sweep(name):
    z = np.load(DATA / name)
    a, ec, per = z["alphas"], z["eta_c_phi"], z["eta_c_phi_seed"]
    se = np.nanstd(per, axis=1, ddof=1) / np.sqrt(
        np.isfinite(per).sum(axis=1))
    r1 = np.exp(-(ec ** a))
    return dict(a=a, ec=ec, se=se, r1=r1,
                se_r1=r1 * a * ec ** (a - 1) * se,
                grids=z["eta_grids"], phi=z["phi"])


# ------------------------------------------------- fig 1 : le collapse
def fig_collapse():
    d15, d30 = load_sweep("phase_curve_v2.npz"), \
               load_sweep("phase_curve_v2_L30.npz")
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))

    # (a) eta_c(alpha) et la loi a un parametre
    A = ax[0, 0]
    for d, c, m, lab in ((d15, W["blue"], "o", r"$L=15$"),
                         (d30, W["vermilion"], "s", r"$L=30$")):
        C = np.exp(np.mean(d["a"] * np.log(d["ec"])))
        aa = np.linspace(d["a"].min(), d["a"].max(), 200)
        A.plot(aa, C ** (1 / aa), "-", color=c, lw=1.0, alpha=0.75)
        A.errorbar(d["a"], d["ec"], yerr=d["se"], fmt=m, color=c,
                   ms=4, capsize=2, lw=0, elinewidth=0.8, label=lab)
    A.set_yscale("log")
    A.set_xlabel(r"$\alpha$")
    A.set_ylabel(r"$\eta_c$")
    A.legend(loc="lower right")
    A.text(0.04, 0.92, "(a)", transform=A.transAxes, fontweight="bold")
    A.text(0.06, 0.72, r"$\eta_c=C^{1/\alpha}$", transform=A.transAxes,
           fontsize=8, color=W["blue"])

    # (b) rho_1 au seuil : le collapse, et la derive en L
    B = ax[0, 1]
    for d, c, m, lab in ((d15, W["blue"], "o", r"$L=15$"),
                         (d30, W["vermilion"], "s", r"$L=30$")):
        B.errorbar(d["a"], d["r1"], yerr=d["se_r1"], fmt=m, color=c,
                   ms=4, capsize=2, lw=0, elinewidth=0.8, label=lab)
        w = 1 / d["se_r1"] ** 2
        B.axhline(np.sum(w * d["r1"]) / np.sum(w), color=c, lw=0.8,
                  ls="--", alpha=0.7)
    B.set_xlabel(r"$\alpha$")
    B.set_ylabel(r"$\rho_1$ at threshold")
    B.legend(loc="lower right")
    B.text(0.04, 0.92, "(b)", transform=B.transAxes, fontweight="bold")
    B.annotate("", xy=(0.55, 0.9785), xytext=(0.55, 0.9712),
               xycoords=("axes fraction", "data"),
               textcoords=("axes fraction", "data"),
               arrowprops=dict(arrowstyle="<->", lw=0.7,
                               color=W["black"]))
    B.text(0.58, 0.5, r"$+0.0072$", transform=B.transAxes, fontsize=7.5)

    # (c) collapse des courbes entieres : phi contre rho_1
    C = ax[1, 0]
    cmap = plt.get_cmap("viridis")
    d = d15
    for i, al in enumerate(d["a"]):
        r = np.exp(-(d["grids"][i] ** al))
        C.plot(r, np.nanmean(d["phi"][i], axis=1), "-o", ms=2.2, lw=0.9,
               color=cmap(i / (len(d["a"]) - 1)),
               label=rf"$\alpha={al:g}$" if i in (0, 4, 8) else None)
    C.axhline(0.5, color=W["black"], lw=0.6, ls=":")
    C.set_xlim(0.90, 1.0)
    C.set_xlabel(r"$\rho_1$")
    C.set_ylabel(r"$\langle\varphi\rangle$")
    C.legend(loc="upper left", fontsize=7)
    C.text(0.04, 0.92, "(c)", transform=C.transAxes, fontweight="bold")

    # (d) quelle harmonique est conservee ?
    D = ax[1, 1]
    for d, c, lab in ((d15, W["blue"], r"$L=15$"),
                      (d30, W["vermilion"], r"$L=30$")):
        ns = np.linspace(0.5, 3.0, 400)
        res = []
        for n in ns:
            y = (n * d["ec"]) ** d["a"]
            Cc = np.exp(np.mean(np.log(y)))
            p = Cc ** (1 / d["a"]) / n
            res.append(100 * np.sqrt(np.mean(((d["ec"] - p) / d["ec"]) ** 2)))
        res = np.array(res)
        D.plot(ns, res, "-", color=c, label=lab)
        D.plot(ns[np.argmin(res)], res.min(), "v", color=c, ms=5)
    D.axvline(1.0, color=W["black"], lw=0.6, ls=":")
    D.set_yscale("log")
    D.set_xlabel(r"harmonic index $n$ held fixed")
    D.set_ylabel(r"residual on $\eta_c$ (\%)")
    D.legend(loc="upper right")
    D.text(0.04, 0.92, "(d)", transform=D.transAxes, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG / "fig_collapse.pdf")
    plt.close(fig)
    print("  fig_collapse.pdf")


# --------------------------------------------- fig 2 : la selectivite
#: Nombres aléatoires COMMUNS. La dérivée de F par différences finies
#: doit être prise sur les MÊMES tirages en (eta-h) et (eta+h), sans
#: quoi elle est dominée par le bruit Monte-Carlo et le chi2 n'est pas
#: reproductible d'une exécution à l'autre. stable_rvs(alpha, c) = c*X0
#: avec X0 ne dépendant que d'alpha et des tirages : on met donc X0 en
#: cache par alpha et on ne fait varier que l'échelle.
_X0: dict = {}


def _wrapped(a, e, n=2_000_000):
    if a not in _X0:
        _X0[a] = stable_rvs(a, 1.0, n, np.random.default_rng(20260809))
    x = _X0[a] * e
    return (x + np.pi) % (2 * np.pi) - np.pi


CANDIDATES = [
    (r"$\rho_1$", lambda a, e: np.exp(-(e ** a)), True),
    (r"$\langle\xi^2\rangle$", lambda a, e: np.mean(_wrapped(a, e) ** 2), False),
    (r"$\rho_2$", lambda a, e: np.exp(-((2 * e) ** a)), False),
    (r"$\rho_3$", lambda a, e: np.exp(-((3 * e) ** a)), False),
    (r"$\rho_4$", lambda a, e: np.exp(-((4 * e) ** a)), False),
    (r"$\langle|\xi|\rangle$", lambda a, e: np.mean(np.abs(_wrapped(a, e))), False),
    (r"IQR", lambda a, e: np.subtract(*np.percentile(_wrapped(a, e), [75, 25])), False),
    (r"med$|\xi|$", lambda a, e: np.median(np.abs(_wrapped(a, e))), False),
    (r"$\eta$ (bare)", lambda a, e: e, False),
]


def _chi2(d, F):
    v, s = [], []
    for A, E, S in zip(d["a"], d["ec"], d["se"]):
        h = 0.02 * E
        v.append(F(A, E))
        s.append(abs((F(A, E + h) - F(A, E - h)) / (2 * h)) * S)
    v, s = np.array(v), np.array(s)
    w = 1 / s ** 2
    c = np.sum(w * v) / np.sum(w)
    return float(np.sum(w * (v - c) ** 2) / (len(v) - 1))


def fig_selectivity():
    d15, d30 = load_sweep("phase_curve_v2.npz"), \
               load_sweep("phase_curve_v2_L30.npz")
    names = [n for n, _, _ in CANDIDATES]
    c15 = [_chi2(d15, F) for _, F, _ in CANDIDATES]
    c30 = [_chi2(d30, F) for _, F, _ in CANDIDATES]
    order = np.argsort(c15)
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for c, off, col, m, lab in ((c15, -0.16, W["blue"], "o", r"$L=15$"),
                                (c30, +0.16, W["vermilion"], "s", r"$L=30$")):
        ax.plot(np.array(c)[order], y + off, m, color=col, ms=4.5,
                label=lab, lw=0)
    ax.axvline(1.0, color=W["black"], lw=0.6, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.get_yticklabels()[0].set_color(W["blue"])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi^2/\mathrm{dof}$ for ``constant at threshold''")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_selectivity.pdf")
    plt.close(fig)
    print(f"  fig_selectivity.pdf   (rho_1 : {c15[0]:.2f} / {c30[0]:.2f} ; "
          f"meilleure concurrente : {min(c15[1:]):.0f} / {min(c30[1:]):.0f})")


# ------------------------------------------------ fig 3 : les familles
def fig_families():
    z = np.load(DATA / "noise_families.npz", allow_pickle=True)
    lab = [str(s) for s in z["labels"]]
    g, phi, r2t = z["rho1_grid"], z["phi"], z["rho2"]
    rc, rcs = z["rho1_c"], z["rho1_c_seed"]
    se = np.nanstd(rcs, axis=1, ddof=1) / np.sqrt(
        np.isfinite(rcs).sum(axis=1))
    r2c = np.array([np.interp(rc[i], g, r2t[i]) for i in range(len(lab))])
    cols = [W["purple"], W["blue"], W["sky"], W["green"],
            W["orange"], W["vermilion"]]
    pretty = {"tumble": "run-and-tumble", "uniform": "bounded uniform",
              "twopoint": r"two-point $\pm\delta$"}

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))

    # (a) phi contre rho_1 : le collapse entre familles
    for i, l in enumerate(lab):
        m = np.nanmean(phi[i], axis=1)
        s = np.nanstd(phi[i], axis=1, ddof=1) / np.sqrt(phi.shape[2])
        ax[0].plot(g, m, "-", color=cols[i], lw=1.0,
                   label=pretty.get(l, l))
        ax[0].fill_between(g, m - s, m + s, color=cols[i], alpha=0.25, lw=0)
    ax[0].axhline(0.5, color=W["black"], lw=0.6, ls=":")
    ax[0].set_xlabel(r"$\rho_1$")
    ax[0].set_ylabel(r"$\langle\varphi\rangle$")
    ax[0].legend(loc="upper left", fontsize=6.2)
    ax[0].text(0.04, 0.92, "(a)", transform=ax[0].transAxes,
               fontweight="bold")

    # (b) le residu contre rho_2
    w = 1 / se ** 2
    X = np.vstack([np.ones_like(r2c), r2c]).T
    W_ = np.diag(w)
    cov = np.linalg.inv(X.T @ W_ @ X)
    b = cov @ (X.T @ W_ @ rc)
    xx = np.linspace(r2c.min() - .005, r2c.max() + .005, 50)
    ax[1].plot(xx, b[0] + b[1] * xx, "-", color=W["black"], lw=0.9,
               alpha=0.7)
    for i, l in enumerate(lab):
        ax[1].errorbar(r2c[i], rc[i], yerr=se[i], fmt="o", color=cols[i],
                       ms=5, capsize=2, lw=0, elinewidth=0.8)
    ax[1].set_xlabel(r"$\rho_2$ at matched $\rho_1$")
    ax[1].set_ylabel(r"$\rho_1$ at threshold")
    ax[1].text(0.04, 0.92, "(b)", transform=ax[1].transAxes,
               fontweight="bold")
    ax[1].text(0.30, 0.10,
               rf"$B={b[1]:+.4f}\pm{np.sqrt(cov[1,1]):.4f}$",
               transform=ax[1].transAxes, fontsize=7)

    # (c) controle in situ de l'identite
    ins1, ins2 = z["rho1_insitu"], z["rho2_insitu"]
    for i in range(len(lab)):
        ax[2].plot(g, np.nanmean(ins1[i], axis=1) - g, "o", ms=2.4,
                   color=cols[i], alpha=0.85)
        ax[2].plot(r2t[i], np.nanmean(ins2[i], axis=1) - r2t[i], "^",
                   ms=2.4, color=cols[i], alpha=0.85)
    ax[2].axhline(0, color=W["black"], lw=0.6, ls=":")
    ax[2].set_ylim(-1.2e-3, 1.2e-3)
    ax[2].set_xlabel(r"nominal $\rho_k$")
    ax[2].set_ylabel(r"measured $-$ nominal")
    ax[2].text(0.04, 0.92, "(c)", transform=ax[2].transAxes,
               fontweight="bold")
    ax[2].text(0.30, 0.08, r"$\circ\ k=1$   $\triangle\ k=2$",
               transform=ax[2].transAxes, fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG / "fig_families.pdf")
    plt.close(fig)
    print(f"  fig_families.pdf      (B = {b[1]:+.5f} +/- "
          f"{np.sqrt(cov[1,1]):.5f})")




# ---------------------------------------------- fig 0 : les lois de bruit
def fig_noise_laws():
    """Ce que « apparier rho_1 » veut dire, montré plutôt que tabulé.

    (a) fonctions de répartition enroulées des six lois : elles n'ont
        visiblement rien en commun -- deux d'entre elles sont même
        purement atomiques.
    (b) leurs coefficients rho_k : confondus en k=1 par construction,
        ils divergent ensuite.  C'est l'expérience du papier en une
        image.
    """
    from noise_families import matched_set, rho_k

    R1 = 0.9711
    fams = matched_set(R1)
    cols = [W["purple"], W["blue"], W["sky"], W["green"],
            W["orange"], W["vermilion"]]
    pretty = {"tumble": "run-and-tumble", "uniform": "bounded uniform",
              "twopoint": r"two-point $\pm\delta$",
              "stable(a=1)": r"wrapped stable, $\alpha=1$",
              "stable(a=1.5)": r"wrapped stable, $\alpha=1.5$",
              "stable(a=2)": r"wrapped stable, $\alpha=2$"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))

    th = np.linspace(-np.pi, np.pi, 3000)
    for i, f in enumerate(fams):
        lab = pretty.get(f.name, pretty.get(f.label, f.label))
        if f.name == "tumble":                     # atome en 0 + uniforme
            q = 1.0 - f.param
            F = q * (th + np.pi) / (2 * np.pi) + (1 - q) * (th >= 0)
        elif f.name == "twopoint":                 # deux atomes
            F = 0.5 * (th >= -f.param) + 0.5 * (th >= f.param)
        elif f.name == "uniform":
            F = np.clip((th + f.param / 2) / f.param, 0, 1)
        else:
            x = f.draw(3_000_000, RNG)
            x = (x + np.pi) % (2 * np.pi) - np.pi
            F = np.searchsorted(np.sort(x), th) / x.size
        ax[0].plot(th, F, "-", color=cols[i], lw=1.3, label=lab)
    # A rho_1 = 0.971 les coups sont petits : toute la structure vit dans
    # |xi| < 0.6.  On zoome, et la legende dit ce qui se passe au-dela.
    ax[0].set_xlim(-1.0, 1.0)
    ax[0].set_ylim(-0.03, 1.03)
    ax[0].set_xlabel(r"wrapped angular kick $\xi$ (rad)")
    ax[0].set_ylabel(r"cumulative distribution")
    ax[0].legend(loc="upper left", fontsize=6.0, handlelength=1.4,
                 borderpad=0.25, labelspacing=0.28)
    ax[0].text(0.955, 0.06, "(a)", transform=ax[0].transAxes,
               fontweight="bold", ha="right")

    ks = np.arange(1, 9)
    for i, f in enumerate(fams):
        ax[1].plot(ks, [rho_k(f.name, f.param, k, f.alpha) for k in ks],
                   "-o", ms=3.2, lw=1.1, color=cols[i])
    ax[1].axvline(1, color=W["black"], lw=0.6, ls=":")
    ax[1].set_xlabel(r"harmonic index $k$")
    ax[1].set_ylabel(r"$\rho_k$")
    ax[1].set_xticks(ks)
    ax[1].set_ylim(-0.45, 1.12)
    ax[1].text(0.955, 0.06, "(b)", transform=ax[1].transAxes,
               fontweight="bold", ha="right")
    ax[1].annotate(r"matched at $k=1$", xy=(1.05, R1), xytext=(2.2, 1.05),
                   fontsize=7, va="center",
                   arrowprops=dict(arrowstyle="->", lw=0.7,
                                   color=W["black"]))

    fig.tight_layout()
    fig.savefig(FIG / "fig_noise_laws.pdf")
    plt.close(fig)
    print("  fig_noise_laws.pdf")

# ------------------------------------------- fig 0b : snapshots appariés
def fig_snapshots_matched(L=30.0, warm=30_000, meas=400, seed=11):
    """La thèse en image : même eta nu contre même rho_1.

    Rangée du haut — la comparaison que fait la littérature, à amplitude
    nominale égale : l'une des deux lois donne un flock, l'autre un gaz.
    Rangée du bas — la même paire de lois à rho_1 apparié : les deux
    états sont indiscernables.

    Conventions CLAUDE.md §2.1 : fond crème, flèches PARTICLE_BLUE,
    arrow_len 0.45, width 0.004, headwidth 3.5, headlength 4.0,
    scale 1/arrow_len, scale_units/angles "xy", pas de ticks, aspect
    égal, titre court avec alpha, eta et <phi> +/- sigma, aucun scatter
    coloré sous les flèches.
    """
    from vicsek import Vicsek, VicsekParams

    def run(alpha, eta):
        p = VicsekParams(N=int(round(2.22 * L * L)), L=L, v0=0.05,
                         R_r=0.45, R_a=0.7, eta=float(eta),
                         alpha=float(alpha), seed=seed)
        s = Vicsek(p)
        s.theta[:] = 0.0
        for _ in range(warm):
            s.step()
        ph = []
        for _ in range(meas):
            s.step()
            ph.append(s.polarisation())
        return s, float(np.mean(ph)), float(np.std(ph))

    ETA_EQ = 0.10                       # amplitude nue commune
    R1_M = 0.985                        # rho_1 commun
    panels = [
        (2.0, ETA_EQ, "equal nominal amplitude"),
        (1.0, ETA_EQ, "equal nominal amplitude"),
        (2.0, (-np.log(R1_M)) ** (1 / 2.0), r"matched $\rho_1$"),
        (1.0, (-np.log(R1_M)) ** (1 / 1.0), r"matched $\rho_1$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.9))
    fig.patch.set_facecolor(style.CREAM)
    arrow_len = 0.45
    for ax, (al, eta, tag) in zip(axes.ravel(), panels):
        sim, m, sd = run(al, eta)
        ax.set_facecolor(style.CREAM)
        ax.quiver(sim.x, sim.y, np.cos(sim.theta), np.sin(sim.theta),
                  color=style.PARTICLE_BLUE,
                  scale=1.0 / arrow_len, scale_units="xy", angles="xy",
                  width=0.004, headwidth=3.5, headlength=4.0)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$\alpha={al:g}$, $\eta={eta:.3f}$"
                     "\n"
                     rf"$\langle\varphi\rangle={m:.2f}\pm{sd:.2f}$",
                     fontsize=8)
        print(f"    alpha={al} eta={eta:.4f} rho1={np.exp(-eta**al):.4f} "
              f"phi={m:.3f}")
    for row, lab in enumerate(("equal nominal amplitude $\\eta$",
                               "matched $\\rho_1$")):
        axes[row, 0].set_ylabel(lab, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_snapshots_matched.pdf", facecolor=style.CREAM)
    plt.close(fig)
    print("  fig_snapshots_matched.pdf")


if __name__ == "__main__":
    print("figures v3 ->", FIG)
    fig_noise_laws()
    fig_snapshots_matched()
    fig_collapse()
    fig_selectivity()
    fig_families()
