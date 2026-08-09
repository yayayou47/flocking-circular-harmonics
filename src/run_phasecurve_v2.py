"""Frontiere de phase eta_c(alpha) -- refonte pour la revision radicale.

Remplace run_phasecurve.py, dont les limites bloquaient le test de
l'invariant rho_1 (cf. version1/PLAN_REVISION_RADICALE.md, §D bis) :

  * 3 graines, sans stockage par graine  -> aucune barre d'erreur sur
    eta_c(alpha), donc residu du collapse incomparable a l'incertitude ;
  * grille eta commune a tous les alpha, lineaire de 0.02 a 0.40, soit
    un seul point sous eta_c a alpha = 1 -> pic de chi censure sur le
    bord (defaut §B.2) ;
  * levier en alpha limite a [1, 2], soit un facteur 5.8 sur eta_c ;
  * ecrivait dans notes/data/ apres la reorg src -> notes/src.

Ce que fait cette version :

  * 20 graines, valeurs par graine stockees (phi, chi, et le secteur
    nematique c_2 = <exp(2 i theta)>, sonde de rho_2) ;
  * grille eta LOG-espacee et PROPRE A CHAQUE ALPHA, centree sur la
    prediction eta_c(alpha) = C^(1/alpha).  Elle n'impose rien : la
    localisation se fait par interpolation a l'interieur de la fenetre,
    et le script signale tout croisement ou tout pic qui tomberait sur
    un bord -- il est donc auto-diagnostiquant, pas auto-confirmant.
    Si la prediction etait fausse, la fenetre le montrerait ;
  * alpha etendu a [0.5, 2.0], soit un facteur ~200 attendu sur eta_c
    au lieu de 5.8.  Controle numerique prealable : l'identite
    rho_k = exp(-(k eta)^alpha) tient a 1e-5 pres jusqu'a alpha = 0.5
    (6 chiffres significatifs conserves a l'enroulement) ;
  * thermalisation 30000 pas (contre 1500), mesure 10000 (contre 1200) ;
  * checkpoint/reprise, parallelise sur les coeurs physiques.

Sortie : data/phase_curve_v2.npz  (n'ecrase PAS phase_curve.npz).

Lancer :
    cd version1/notes/src
    nohup python3 run_phasecurve_v2.py > ../../logs/phasecurve_v2.log 2>&1 &
    python3 run_phasecurve_v2.py --dry-run     # sanity, quelques minutes
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Numba doit rester mono-thread : le parallelisme est au niveau des jobs.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
V1_ROOT = HERE.parent
DATA = V1_ROOT / "data"          # canonique : version1/data, pas notes/data
DATA.mkdir(exist_ok=True)

# Constante de la prediction eta_c(alpha) = C^(1/alpha), ajustee sur les
# six points de phase_curve.npz (critere <phi> = 1/2). Elle ne sert ICI
# qu'a placer la fenetre de balayage, jamais a analyser le resultat.
C_PRED = 0.02934

ALPHAS = np.array([0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9, 2.0])
N_ETA = 20
ETA_LO, ETA_HI = 0.25, 4.0       # fenetre en unites de eta_c predit
SEEDS = np.array([11, 23, 41, 67, 89, 101, 137, 211, 307, 401,
                  433, 509, 601, 691, 787, 863, 941, 1013, 1097, 1187])

L, SIGMA = 15.0, 2.22
V0, R_R, R_A = 0.05, 0.45, 0.7
N_WARM, N_MEAS = 30_000, 10_000

# Les cinq constantes ci-dessus sont surchargeables en ligne de commande
# (--L, --alphas, --n-eta, --n-seeds) SANS toucher au reste du protocole :
# meme fenetre relative, meme thermalisation, meme localisateur, memes
# graines. C'est ce qui rend comparables les balayages a L different --
# le defaut §B.1 du plan etait precisement d'avoir compare des leviers
# mesures avec des protocoles differents. Les valeurs par defaut
# reproduisent exactement data/phase_curve_v2.npz.


def eta_grid(alpha: float) -> np.ndarray:
    """Fenetre log-espacee autour de eta_c predit pour cet alpha."""
    c = C_PRED ** (1.0 / alpha)
    return np.geomspace(ETA_LO * c, ETA_HI * c, N_ETA)


def _job(args):
    """Une (alpha, eta, graine). Renvoie (phi_moy, chi, c2)."""
    from vicsek import Vicsek, VicsekParams

    alpha, eta, seed, n_warm, n_meas = args
    N = int(round(SIGMA * L * L))
    p = VicsekParams(N=N, L=L, v0=V0, R_r=R_R, R_a=R_A,
                     eta=float(eta), alpha=float(alpha), seed=int(seed))
    sim = Vicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    phis = np.empty(n_meas)
    c2s = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        z = np.exp(1j * sim.theta)
        phis[k] = abs(z.mean())
        c2s[k] = abs((z ** 2).mean())
    return float(phis.mean()), float(N * phis.var()), float(c2s.mean())


def _locate(etas, phi, chi):
    """eta_c par croisement <phi> = 1/2 et par sommet du pic de chi.
    Renvoie aussi des drapeaux de bord : un resultat au bord de la
    fenetre est un resultat censure, pas une mesure."""
    out = {}

    # Croisement phi = 1/2, interpole en log(eta).
    idx = np.where((phi[:-1] - 0.5) * (phi[1:] - 0.5) <= 0)[0]
    if idx.size:
        k = int(idx[0])
        t = (0.5 - phi[k]) / (phi[k + 1] - phi[k])
        le = np.log(etas)
        out["eta_c_phi"] = float(np.exp(le[k] + t * (le[k + 1] - le[k])))
        out["phi_edge"] = False
    else:
        out["eta_c_phi"] = float("nan")
        out["phi_edge"] = True          # pas de croisement dans la fenetre

    # Sommet du pic de chi, parabole sur log(eta) autour du max.
    j = int(np.argmax(chi))
    out["chi_edge"] = (j == 0 or j == len(etas) - 1)
    if not out["chi_edge"]:
        x = np.log(etas[j - 1:j + 2])
        y = chi[j - 1:j + 2]
        den = y[0] - 2 * y[1] + y[2]
        if abs(den) > 1e-12:
            shift = 0.5 * (y[0] - y[2]) / den
            out["eta_c_chi"] = float(np.exp(x[1] + shift * (x[1] - x[0])))
        else:
            out["eta_c_chi"] = float(etas[j])
    else:
        out["eta_c_chi"] = float(etas[j])
    return out


def main():
    global L, N_ETA, ALPHAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="3 graines, 2000/800 pas : valide la chaine")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = cpu_count()-1")
    ap.add_argument("--L", type=float, default=L,
                    help="taille de boite (defaut 15)")
    ap.add_argument("--alphas", type=str, default="",
                    help="liste separee par des virgules, ex 0.75,1,1.5,2")
    ap.add_argument("--n-eta", type=int, default=N_ETA)
    ap.add_argument("--n-seeds", type=int, default=len(SEEDS))
    ap.add_argument("--tag", type=str, default="",
                    help="suffixe du fichier de sortie")
    args = ap.parse_args()

    import multiprocessing as mp

    L = args.L
    N_ETA = args.n_eta
    if args.alphas:
        ALPHAS = np.array([float(s) for s in args.alphas.split(",")])

    seeds = SEEDS[:3] if args.dry_run else SEEDS[:args.n_seeds]
    n_warm = 2000 if args.dry_run else N_WARM
    n_meas = 800 if args.dry_run else N_MEAS
    tag = args.tag or (f"_L{L:g}" if L != 15.0 else "")
    out = DATA / (f"phase_curve_v2{tag}_dry.npz" if args.dry_run
                  else f"phase_curve_v2{tag}.npz")
    ckpt = out.with_suffix(".partial.npz")

    n_a, n_e, n_s = len(ALPHAS), N_ETA, len(seeds)
    grids = np.stack([eta_grid(a) for a in ALPHAS])
    print(f"L = {L:g}, N = {int(round(SIGMA*L*L))}, "
          f"alphas = {list(ALPHAS)}, {n_e} eta, {n_s} graines")
    phi = np.full((n_a, n_e, n_s), np.nan)
    chi = np.full((n_a, n_e, n_s), np.nan)
    c2 = np.full((n_a, n_e, n_s), np.nan)

    if ckpt.exists():
        z = np.load(ckpt)
        if z["phi"].shape == phi.shape:
            phi, chi, c2 = z["phi"], z["chi"], z["c2"]
            print(f"reprise depuis {ckpt.name} : "
                  f"{int(np.isfinite(phi).sum())}/{phi.size} jobs faits")

    todo = [(ia, ie, isd)
            for ia in range(n_a) for ie in range(n_e) for isd in range(n_s)
            if not np.isfinite(phi[ia, ie, isd])]
    jobs = [(float(ALPHAS[ia]), float(grids[ia, ie]), int(seeds[isd]),
             n_warm, n_meas) for ia, ie, isd in todo]

    nw = args.workers or max(1, (mp.cpu_count() or 2) - 1)
    print(f"{len(jobs)} jobs restants, {nw} workers, "
          f"{n_warm}+{n_meas} pas, {n_s} graines, "
          f"alpha in [{ALPHAS[0]}, {ALPHAS[-1]}]")
    print(f"sortie : {out}")

    t0 = time.perf_counter()
    done = 0
    with mp.Pool(nw) as pool:
        for (ia, ie, isd), (p_, c_, n_) in zip(
                todo, pool.imap(_job, jobs, chunksize=1)):
            phi[ia, ie, isd] = p_
            chi[ia, ie, isd] = c_
            c2[ia, ie, isd] = n_
            done += 1
            if done % 50 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                eta_s = el / done * (len(jobs) - done)
                print(f"  {done}/{len(jobs)}  ecoule {el/60:.1f} min  "
                      f"reste ~{eta_s/60:.1f} min", flush=True)
                np.savez_compressed(ckpt, phi=phi, chi=chi, c2=c2)

    # ---- localisation, par graine (pour les barres d'erreur) et en moyenne
    phi_m, chi_m = np.nanmean(phi, axis=2), np.nanmean(chi, axis=2)
    eta_c_phi = np.zeros(n_a)
    eta_c_chi = np.zeros(n_a)
    edge_phi = np.zeros(n_a, bool)
    edge_chi = np.zeros(n_a, bool)
    eta_c_phi_seed = np.full((n_a, n_s), np.nan)

    for ia in range(n_a):
        r = _locate(grids[ia], phi_m[ia], chi_m[ia])
        eta_c_phi[ia], eta_c_chi[ia] = r["eta_c_phi"], r["eta_c_chi"]
        edge_phi[ia], edge_chi[ia] = r["phi_edge"], r["chi_edge"]
        for isd in range(n_s):
            rs = _locate(grids[ia], phi[ia, :, isd], chi[ia, :, isd])
            eta_c_phi_seed[ia, isd] = rs["eta_c_phi"]

    np.savez_compressed(
        out,
        alphas=ALPHAS, eta_grids=grids, seeds=seeds,
        phi=phi, chi=chi, c2=c2,
        eta_c_phi=eta_c_phi, eta_c_chi=eta_c_chi,
        eta_c_phi_seed=eta_c_phi_seed,
        edge_phi=edge_phi, edge_chi=edge_chi,
        params=np.array([int(round(SIGMA * L * L)), L, V0, R_R, R_A,
                         n_warm, n_meas, n_s], dtype=float),
    )
    if ckpt.exists():
        ckpt.unlink()

    print(f"\nsauve : {out}")
    print(f"{'alpha':>6} {'eta_c(phi)':>11} {'+/- SE':>9} "
          f"{'eta_c(chi)':>11} {'rho_1':>8} {'bord ?':>8}")
    for ia, a in enumerate(ALPHAS):
        se = np.nanstd(eta_c_phi_seed[ia]) / np.sqrt(
            max(np.isfinite(eta_c_phi_seed[ia]).sum(), 1))
        rho1 = np.exp(-(eta_c_phi[ia] ** a))
        flag = ("PHI " if edge_phi[ia] else "") + \
               ("CHI" if edge_chi[ia] else "")
        print(f"{a:6.2f} {eta_c_phi[ia]:11.5f} {se:9.5f} "
              f"{eta_c_chi[ia]:11.5f} {rho1:8.5f} {flag or '-':>8}")
    if edge_phi.any() or edge_chi.any():
        print("\n[!] au moins un point est tombe sur un bord de fenetre :")
        print("    resultat censure, elargir ETA_LO / ETA_HI et relancer.")


if __name__ == "__main__":
    main()
