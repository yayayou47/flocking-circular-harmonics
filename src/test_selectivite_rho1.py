"""Sélectivité de rho_1 : est-il le SEUL résumé à un nombre qui collapse ?

C'est la revendication centrale du papier v2, et celle qui nous distingue
de Porfiri & Ariel (Chaos 26, 043109, 2016), qui montrent que rho_1
*marche* mais pas qu'il est le *seul* à marcher.

Méthode
-------
Pour chaque fonctionnelle candidate F de la loi de bruit enroulée, on
évalue F au seuil mesuré eta_c(alpha) et on teste l'hypothèse
« F est constant au seuil » par un chi2 pondéré, l'incertitude étant
propagée depuis celle de eta_c :  sigma_F = |dF/d eta| * sigma_eta.

Piège à éviter
--------------
Toute fonction strictement monotone de rho_1 donne EXACTEMENT le même
chi2 (« f(rho_1) = const » equivaut à « rho_1 = const »). Les
concurrentes honnêtes sont donc les fonctionnelles qui dépendent
réellement de alpha A rho_1 FIXÉ -- rho_2, rho_3, les mesures de
dispersion dans l'espace réel, etc. La variance circulaire V = 1 - rho_1
et u = -ln rho_1 sont incluses comme TÉMOINS : elles doivent donner le
même chi2 que rho_1, ce qui valide la propagation d'erreurs.

Lancer :  python3 notes/src/test_selectivite_rho1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from noise import stable_rvs  # noqa: E402

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


# --- fonctionnelles candidates : F(alpha, eta) -----------------------
def F_rho(k):
    return lambda a, e: float(np.exp(-((k * e) ** a)))


def F_u(a, e):                       # -ln rho_1  (témoin monotone)
    return float(e ** a)


def F_V(a, e):                       # variance circulaire (témoin)
    return float(1.0 - np.exp(-(e ** a)))


def F_eta(a, e):                     # le choix naif : l'amplitude nue
    return float(e)


def F_absdev(a, e):                  # E|xi| enroulé
    return float(np.mean(np.abs(_wrapped(a, e))))


def F_mad(a, e):                     # médiane de |xi| enroulé
    return float(np.median(np.abs(_wrapped(a, e))))


def F_var(a, e):                     # E[xi^2] enroulé
    return float(np.mean(_wrapped(a, e) ** 2))


def F_tail(a, e):                    # P(|xi| > pi/2)
    return float(np.mean(np.abs(_wrapped(a, e)) > np.pi / 2))


def F_iqr(a, e):                     # écart interquartile
    w = _wrapped(a, e)
    return float(np.percentile(w, 75) - np.percentile(w, 25))


CANDIDATS = [
    ("rho_1 = <cos xi>", F_rho(1), "notre invariant"),
    ("V = 1 - rho_1", F_V, "TÉMOIN (fonction monotone de rho_1)"),
    ("u = -ln rho_1", F_u, "TÉMOIN (fonction monotone de rho_1)"),
    ("rho_2", F_rho(2), "concurrente"),
    ("rho_3", F_rho(3), "concurrente"),
    ("rho_4", F_rho(4), "concurrente"),
    ("E|xi| enroulé", F_absdev, "concurrente"),
    ("médiane |xi|", F_mad, "concurrente"),
    ("E[xi^2] enroulé", F_var, "concurrente"),
    ("P(|xi| > pi/2)", F_tail, "concurrente"),
    ("écart interquartile", F_iqr, "concurrente"),
    ("eta (amplitude nue)", F_eta, "le choix naïf de la littérature"),
]


def evaluate(path, label):
    z = np.load(path)
    a = z["alphas"]
    ec = z["eta_c_phi"]
    per = z["eta_c_phi_seed"]
    se = np.nanstd(per, axis=1, ddof=1) / np.sqrt(
        np.isfinite(per).sum(axis=1))

    print("=" * 74)
    print(f"{label} — {len(a)} valeurs de alpha, "
          f"gamme de eta_c = {ec.max() / ec.min():.0f}x")
    print("=" * 74)
    print(f"{'fonctionnelle':>22} {'chi2/dof':>10} {'p':>10}   remarque")

    rows = []
    for name, F, note in CANDIDATS:
        vals, sig = [], []
        for A, E, S in zip(a, ec, se):
            f0 = F(A, E)
            # dérivée par différences centrées, pas relatif 2 %
            h = 0.02 * E
            df = (F(A, E + h) - F(A, E - h)) / (2 * h)
            vals.append(f0)
            sig.append(abs(df) * S)
        vals, sig = np.array(vals), np.array(sig)
        if not np.all(sig > 0):
            continue
        w = 1.0 / sig ** 2
        c = np.sum(w * vals) / np.sum(w)
        chi2 = float(np.sum(w * (vals - c) ** 2))
        dof = len(a) - 1
        rows.append((name, chi2 / dof, note))

    ref = rows[0][1]
    for name, cd, note in rows:
        ratio = f"  ({cd / ref:.0f}x rho_1)" if cd > 2 * ref else ""
        print(f"{name:>22} {cd:10.2f} {'':10}   {note}{ratio}")
    return rows


def main():
    r15 = evaluate(V1_ROOT / "data" / "phase_curve_v2.npz",
                   "L = 15  (data/phase_curve_v2.npz)")
    print()
    r30 = evaluate(V1_ROOT / "data" / "phase_curve_v2_L30.npz",
                   "L = 30  (data/phase_curve_v2_L30.npz)")

    print()
    print("=" * 74)
    print("LECTURE")
    print("=" * 74)
    for lab, rows in (("L=15", r15), ("L=30", r30)):
        ref = rows[0][1]
        conc = [c for n, c, note in rows if note == "concurrente"]
        temoins = [c for n, c, note in rows if note.startswith("TÉMOIN")]
        ok = all(abs(t - ref) < 0.01 * max(ref, 1e-9) for t in temoins)
        print(f"  {lab} : rho_1 chi2/dof = {ref:.2f} ; "
              f"meilleure concurrente = {min(conc):.2f} "
              f"(facteur {min(conc)/ref:.0f})")
        print(f"        témoins monotones identiques a rho_1 : "
              f"{'oui' if ok else 'NON — propagation d erreur suspecte'}")


if __name__ == "__main__":
    main()
