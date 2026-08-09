"""Test des familles de bruit appariees sur rho_1.

Pourquoi ce run est decisif
---------------------------
Le balayage en alpha (data/phase_curve_v2.npz) etablit deux choses :

  1. le seuil est fixe par rho_1 : eta_c varie d'un facteur 187 quand
     alpha va de 0.5 a 2, et rho_1 au seuil ne bouge que de 0.25 % ;
  2. il reste un residu STATISTIQUEMENT RESOLU : rho_1^c = const est
     rejete (chi2/dof = 4.15), et rho_1^c = A + B rho_2 l'ajuste
     parfaitement (chi2/dof = 0.87, B = -0.0249).

Mais dans la famille stable enroulee rho_2 = rho_1^(2^alpha) est une
fonction deterministe de alpha : "le residu suit rho_2" et "le residu
suit une fonction monotone quelconque de alpha" sont indiscernables sur
ce seul jeu.  La degenerescence ne peut etre levee qu'en changeant de
famille de bruit a rho_1 fixe.

C'est ce que fait ce run.  Six familles appariees sur rho_1 balayent
rho_2 de 0.886 a 0.971 (cf. noise_families.py).  Si l'interpretation
rho_2 est juste, leurs seuils doivent tomber sur la MEME droite
rho_1^c = A + B rho_2, avec le B = -0.0249 mesure independamment sur le
balayage en alpha.  C'est une prediction SANS PARAMETRE LIBRE.

Amplitude attendue : B x (rho_2^max - rho_2^min) = -0.0249 x 0.085
= -0.0021 sur rho_1^c, contre une incertitude de ~0.00035 a 20 graines,
soit un effet a ~6 sigma.

Le run repond du meme coup au grief 5 du rapporteur (le bruit uniforme
borne de Vicsek 1995 n'avait jamais ete simule) et a son grief sur la
nouveaute face aux Vicsek generalises : on n'ajoute pas une famille de
bruit de plus, on identifie ce qui, dans toutes ces familles, compte.

Test in situ du theoreme, gratuit
---------------------------------
La factorisation c_k' = rho_k <exp(i k theta*)> vaut pour TOUT
pre-bruit theta*, la seule hypothese etant l'independance de xi.  On la
verifie donc directement dans la simulation, en estimant

    rho_k^mes = sum_t Re(Z_k conj(Z*_k)) / sum_t |Z*_k|^2

avec Z_k = (1/N) sum_j exp(i k theta_j) apres bruit et Z*_k avant.
Conditionnellement a theta*, E[Z_k | theta*] = rho_k Z*_k exactement.

Sortie : data/noise_families.npz

Lancer :
    python3 run_noise_families.py --dry-run
    nohup python3 run_noise_families.py > ../../logs/noise_families.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
V1_ROOT = HERE.parent
DATA = V1_ROOT / "data"
DATA.mkdir(exist_ok=True)

L, SIGMA = 15.0, 2.22
V0, R_R, R_A = 0.05, 0.45, 0.7
N_WARM, N_MEAS = 30_000, 10_000

#: Grille commune en rho_1. Resserree (pas 0.0016) sur [0.960, 0.982],
#: ou tombe le croisement phi = 1/2 : l'effet attendu entre familles est
#: de 0.0021 sur rho_1^c, il faut donc un pas nettement inferieur pour
#: que l'interpolation ne le limite pas.
RHO1_GRID = np.unique(np.concatenate([
    np.linspace(0.935, 0.9575, 4),
    np.linspace(0.960, 0.982, 15),
    np.linspace(0.9845, 0.993, 4),
]))

SEEDS = np.array([11, 23, 41, 67, 89, 101, 137, 211, 307, 401,
                  433, 509, 601, 691, 787, 863, 941, 1013, 1097, 1187])


def _job(args):
    """Une (famille, rho_1, graine)."""
    import numpy as np
    from vicsek import Vicsek, VicsekParams, _HAS_NUMBA
    from vicsek import _build_cells, _zonal_update, _zonal_update_np
    from noise_families import Family

    name, alpha, param, rho1, seed, n_warm, n_meas = args
    fam = Family(name, alpha, param, rho1)
    N = int(round(SIGMA * L * L))

    # eta/alpha des params sont inutilises : le bruit est tire par fam.
    p = VicsekParams(N=N, L=L, v0=V0, R_r=R_R, R_a=R_A,
                     eta=0.0, alpha=2.0, seed=int(seed))
    sim = Vicsek(p)
    sim.theta[:] = 0.0

    def one_step():
        """Copie de Vicsek.step() avec le bruit de la famille, et
        renvoie le pre-bruit theta* pour le test in situ."""
        if _HAS_NUMBA:
            head, nxt = _build_cells(sim.x, sim.y, p.L, sim.n_cell)
            target = _zonal_update(sim.x, sim.y, sim.theta, p.L,
                                   sim.R_r, sim.R_a, head, nxt, sim.n_cell)
        else:
            target = _zonal_update_np(sim.x, sim.y, sim.theta, p.L,
                                      sim.R_r, sim.R_a)
        xi = fam.draw(p.N, sim.rng)
        sim.theta = (target + xi + np.pi) % (2 * np.pi) - np.pi
        sim.x = (sim.x + p.v0 * np.cos(sim.theta)) % p.L
        sim.y = (sim.y + p.v0 * np.sin(sim.theta)) % p.L
        sim.t += 1
        return target

    for _ in range(n_warm):
        one_step()

    phis = np.empty(n_meas)
    c2s = np.empty(n_meas)
    num1 = num2 = den1 = den2 = 0.0
    for k in range(n_meas):
        target = one_step()
        z1s = np.exp(1j * target).mean()
        z2s = np.exp(2j * target).mean()
        z1 = np.exp(1j * sim.theta).mean()
        z2 = np.exp(2j * sim.theta).mean()
        phis[k] = abs(z1)
        c2s[k] = abs(z2)
        num1 += (z1 * np.conj(z1s)).real
        den1 += abs(z1s) ** 2
        num2 += (z2 * np.conj(z2s)).real
        den2 += abs(z2s) ** 2

    m2 = float(np.mean(phis ** 2))
    m4 = float(np.mean(phis ** 4))
    return (float(phis.mean()), float(N * phis.var()),
            float(1.0 - m4 / (3.0 * m2 ** 2)) if m2 > 0 else np.nan,
            float(c2s.mean()),
            float(num1 / den1) if den1 > 0 else np.nan,
            float(num2 / den2) if den2 > 0 else np.nan)


def _cross(grid, phi, target=0.5):
    """rho_1 ou phi croise 1/2. grid decroit en desordre : phi croit
    avec rho_1. Renvoie (valeur, bord?)."""
    idx = np.where((phi[:-1] - target) * (phi[1:] - target) <= 0)[0]
    if not idx.size:
        return float("nan"), True
    k = int(idx[0])
    t = (target - phi[k]) / (phi[k + 1] - phi[k])
    return float(grid[k] + t * (grid[k + 1] - grid[k])), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    import multiprocessing as mp
    from noise_families import SPEC, invert_rho1

    seeds = SEEDS[:3] if args.dry_run else SEEDS
    n_warm = 2000 if args.dry_run else N_WARM
    n_meas = 800 if args.dry_run else N_MEAS
    out = DATA / ("noise_families_dry.npz" if args.dry_run
                  else "noise_families.npz")
    ckpt = out.with_suffix(".partial.npz")

    labels = [f"stable(a={a:g})" if n == "stable" else n for n, a in SPEC]
    n_f, n_r, n_s = len(SPEC), len(RHO1_GRID), len(seeds)

    # parametre de chaque famille pour chaque rho_1 de la grille
    params = np.array([[invert_rho1(n, r, a) for r in RHO1_GRID]
                       for n, a in SPEC])
    from noise_families import rho_k
    rho2 = np.array([[rho_k(SPEC[i][0], params[i, j], 2, SPEC[i][1])
                      for j in range(n_r)] for i in range(n_f)])

    shape = (n_f, n_r, n_s)
    phi = np.full(shape, np.nan)
    chi = np.full(shape, np.nan)
    u4 = np.full(shape, np.nan)
    c2 = np.full(shape, np.nan)
    r1m = np.full(shape, np.nan)
    r2m = np.full(shape, np.nan)

    if ckpt.exists():
        z = np.load(ckpt)
        if z["phi"].shape == shape:
            phi, chi, u4, c2 = z["phi"], z["chi"], z["u4"], z["c2"]
            r1m, r2m = z["r1m"], z["r2m"]
            print(f"reprise : {int(np.isfinite(phi).sum())}/{phi.size}")

    todo = [(i, j, k) for i in range(n_f) for j in range(n_r)
            for k in range(n_s) if not np.isfinite(phi[i, j, k])]
    jobs = [(SPEC[i][0], SPEC[i][1], float(params[i, j]),
             float(RHO1_GRID[j]), int(seeds[k]), n_warm, n_meas)
            for i, j, k in todo]

    nw = args.workers or max(1, (mp.cpu_count() or 2) - 1)
    print(f"{len(jobs)} jobs, {nw} workers, {n_warm}+{n_meas} pas, "
          f"{n_s} graines, {n_f} familles x {n_r} valeurs de rho_1")
    print(f"sortie : {out}")

    t0, done = time.perf_counter(), 0
    with mp.Pool(nw) as pool:
        for (i, j, k), r in zip(todo, pool.imap(_job, jobs, chunksize=1)):
            phi[i, j, k], chi[i, j, k], u4[i, j, k] = r[0], r[1], r[2]
            c2[i, j, k], r1m[i, j, k], r2m[i, j, k] = r[3], r[4], r[5]
            done += 1
            if done % 50 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                print(f"  {done}/{len(jobs)}  {el/60:.1f} min  "
                      f"reste ~{el/done*(len(jobs)-done)/60:.1f} min",
                      flush=True)
                np.savez_compressed(ckpt, phi=phi, chi=chi, u4=u4,
                                    c2=c2, r1m=r1m, r2m=r2m)

    phi_m = np.nanmean(phi, axis=2)
    rho1_c = np.zeros(n_f)
    edge = np.zeros(n_f, bool)
    rho1_c_seed = np.full((n_f, n_s), np.nan)
    for i in range(n_f):
        rho1_c[i], edge[i] = _cross(RHO1_GRID, phi_m[i])
        for k in range(n_s):
            rho1_c_seed[i, k], _ = _cross(RHO1_GRID, phi[i, :, k])

    np.savez_compressed(
        out, labels=np.array(labels), spec_names=np.array([s[0] for s in SPEC]),
        spec_alphas=np.array([s[1] for s in SPEC]),
        rho1_grid=RHO1_GRID, params=params, rho2=rho2, seeds=seeds,
        phi=phi, chi=chi, u4=u4, c2=c2, rho1_insitu=r1m, rho2_insitu=r2m,
        rho1_c=rho1_c, rho1_c_seed=rho1_c_seed, edge=edge,
        setup=np.array([int(round(SIGMA * L * L)), L, V0, R_R, R_A,
                        n_warm, n_meas, n_s], dtype=float))
    if ckpt.exists():
        ckpt.unlink()

    print(f"\nsauve : {out}\n")
    print(f"{'famille':>18} {'rho_2 @seuil':>13} {'rho_1^c':>10} "
          f"{'SE':>9} {'rho_1 in situ':>14} {'ecart':>10}")
    for i in range(n_f):
        j = int(np.argmin(abs(RHO1_GRID - rho1_c[i])))
        se = np.nanstd(rho1_c_seed[i], ddof=1) / np.sqrt(
            max(np.isfinite(rho1_c_seed[i]).sum(), 1))
        ins = np.nanmean(r1m[i, j])
        print(f"{labels[i]:>18} {rho2[i, j]:13.5f} {rho1_c[i]:10.5f} "
              f"{se:9.5f} {ins:14.5f} {ins - RHO1_GRID[j]:+10.2e}")
    if edge.any():
        print("\n[!] croisement hors fenetre pour "
              f"{[labels[i] for i in range(n_f) if edge[i]]} : elargir "
              "RHO1_GRID.")


if __name__ == "__main__":
    main()
