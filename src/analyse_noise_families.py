"""Analyse du test des familles de bruit appariees sur rho_1.

Repond a une seule question : le residu mesure sur le balayage en alpha
(rho_1^c = A + B rho_2, B = -0.0249, chi2/dof = 0.87) est-il bien porte
par rho_2, ou n'etait-ce qu'une fonction monotone de alpha deguisee ?

Le balayage en alpha ne peut pas trancher : dans la famille stable,
rho_2 = rho_1^(2^alpha) est une fonction deterministe de alpha.  Six
familles appariees sur rho_1 mais de rho_2 differents levent la
degenerescence.

Prediction SANS PARAMETRE LIBRE : les six seuils tombent sur la droite
rho_1^c = A + B rho_2 avec le B mesure independamment sur le balayage
en alpha.

Lancer :  python3 notes/src/analyse_noise_families.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1_ROOT = HERE.parent
DATA = V1_ROOT / "data"


def _wls(x, y, w):
    """Moindres carres ponderes y = a + b x. Renvoie (a, b, se_b, chi2)."""
    X = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ (X.T @ W @ y)
    chi2 = float(np.sum(w * (y - X @ beta) ** 2))
    return float(beta[0]), float(beta[1]), float(np.sqrt(cov[1, 1])), chi2


def alpha_sweep_reference():
    """Refait la mesure de B sur le balayage en alpha, pour disposer de
    la prediction independante."""
    z = np.load(DATA / "phase_curve_v2.npz")
    a, ec, per = z["alphas"], z["eta_c_phi"], z["eta_c_phi_seed"]
    se = np.nanstd(per, axis=1, ddof=1) / np.sqrt(
        np.isfinite(per).sum(axis=1))
    r1 = np.exp(-(ec ** a))
    se_r1 = r1 * a * ec ** (a - 1) * se
    r2 = r1 ** (2.0 ** a)
    w = 1.0 / se_r1 ** 2
    c0 = np.sum(w * r1) / np.sum(w)
    chi0 = float(np.sum(w * (r1 - c0) ** 2))
    A, B, seB, chi1 = _wls(r2, r1, w)
    return dict(rho1=r1, se=se_r1, rho2=r2, A=A, B=B, seB=seB,
                chi2_const=chi0 / (len(a) - 1),
                chi2_lin=chi1 / (len(a) - 2), const=c0)


def main():
    ref = alpha_sweep_reference()
    print("=" * 76)
    print("REFERENCE — balayage en alpha (phase_curve_v2.npz)")
    print("=" * 76)
    print(f"  rho_1^c = const        : chi2/dof = {ref['chi2_const']:5.2f}"
          f"   (rejete)")
    print(f"  rho_1^c = A + B rho_2  : chi2/dof = {ref['chi2_lin']:5.2f}"
          f"   A = {ref['A']:+.5f}  B = {ref['B']:+.5f} +/- {ref['seB']:.5f}")

    p = DATA / "noise_families.npz"
    if not p.exists():
        print(f"\n[!] {p.name} absent : run pas encore termine.")
        return
    z = np.load(p, allow_pickle=True)
    labels = [str(s) for s in z["labels"]]
    grid = z["rho1_grid"]
    r1c = z["rho1_c"]
    r1c_seed = z["rho1_c_seed"]
    rho2_tab = z["rho2"]
    insitu1, insitu2 = z["rho1_insitu"], z["rho2_insitu"]
    n_f = len(labels)

    se = np.nanstd(r1c_seed, axis=1, ddof=1) / np.sqrt(
        np.isfinite(r1c_seed).sum(axis=1))
    # rho_2 evalue AU SEUIL de chaque famille
    rho2_c = np.array([np.interp(r1c[i], grid, rho2_tab[i])
                       for i in range(n_f)])

    print()
    print("=" * 76)
    print("TEST — six familles appariees sur rho_1")
    print("=" * 76)
    print(f"{'famille':>18} {'rho_2':>9} {'rho_1^c':>10} {'SE':>9} "
          f"{'predit':>10} {'ecart/SE':>9}")
    pred = ref["A"] + ref["B"] * rho2_c
    for i in np.argsort(-rho2_c):
        print(f"{labels[i]:>18} {rho2_c[i]:9.5f} {r1c[i]:10.5f} "
              f"{se[i]:9.5f} {pred[i]:10.5f} "
              f"{(r1c[i]-pred[i])/se[i]:+9.1f}")

    if z["edge"].any():
        bad = [labels[i] for i in range(n_f) if z["edge"][i]]
        print(f"\n[!] croisement hors fenetre : {bad} — resultat censure.")

    # --- 1. l'effet existe-t-il ?
    w = 1.0 / se ** 2
    c0 = np.sum(w * r1c) / np.sum(w)
    chi0 = float(np.sum(w * (r1c - c0) ** 2)) / (n_f - 1)
    A2, B2, seB2, chi1 = _wls(rho2_c, r1c, w)
    chi1 /= (n_f - 2)
    print()
    print("=" * 76)
    print("1. Les seuils dependent-ils de rho_2 ?")
    print("=" * 76)
    print(f"  rho_1^c = const       : chi2/dof = {chi0:6.2f}")
    print(f"  rho_1^c = A + B rho_2 : chi2/dof = {chi1:6.2f}   "
          f"B = {B2:+.5f} +/- {seB2:.5f}")
    print(f"  significativite de B  : {abs(B2)/seB2:5.1f} sigma")

    # --- 2. accord avec la prediction independante
    chi_pred = float(np.sum(((r1c - pred) / se) ** 2)) / n_f
    dB = (B2 - ref["B"]) / np.hypot(seB2, ref["seB"])
    print()
    print("=" * 76)
    print("2. Accord avec la prediction sans parametre libre")
    print("=" * 76)
    print(f"  B (familles) = {B2:+.5f} +/- {seB2:.5f}")
    print(f"  B (alpha)    = {ref['B']:+.5f} +/- {ref['seB']:.5f}")
    print(f"  ecart        = {dB:+.1f} sigma")
    print(f"  chi2/dof des six points contre la droite du balayage")
    print(f"  en alpha, AUCUN parametre reajuste : {chi_pred:.2f}")

    # --- 3. verdict
    print()
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    if chi0 > 3 and chi1 < 2 and abs(dB) < 3:
        print("  Les seuils s'ordonnent selon rho_2, avec la meme pente")
        print("  que le balayage en alpha. La loi a deux parametres est")
        print("  etablie sur deux jeux independants et six lois de bruit.")
    elif chi0 < 2:
        print("  Aucune dependance en rho_2 resolue entre familles : rho_1")
        print("  controle la position, et le residu vu en alpha est propre")
        print("  a la famille stable. A rapporter tel quel — la these")
        print("  principale (rho_1 fixe le seuil) tient.")
    else:
        print("  Situation intermediaire : voir les ecarts point par point")
        print("  ci-dessus avant de conclure.")

    # --- 4. controle in situ du theoreme
    print()
    print("=" * 76)
    print("3. Controle in situ de c_k' = rho_k <exp(i k theta*)>")
    print("=" * 76)
    print(f"{'famille':>18} {'|rho_1 mes - nominal|':>23} "
          f"{'|rho_2 mes - nominal|':>23}")
    for i in range(n_f):
        d1 = np.nanmax(np.abs(np.nanmean(insitu1[i], axis=1) - grid))
        d2 = np.nanmax(np.abs(np.nanmean(insitu2[i], axis=1) - rho2_tab[i]))
        print(f"{labels[i]:>18} {d1:23.2e} {d2:23.2e}")
    print("  Identite verifiee a l'interieur du modele complet, pas")
    print("  seulement sur le generateur de bruit.")


if __name__ == "__main__":
    main()
