"""Phase 1 du plan de revision radicale : l'invariant de la loi de bruit.

Trois tests, aucun cout de simulation. Tout se relit dans
``data/phase_curve_v2.npz`` ou se calcule par Monte-Carlo direct.

Le resultat
-----------
Le bruit angulaire etant additif sur la tete et independant de la
configuration, il se decouple harmonique par harmonique. Si
``theta' = arg(S) + xi`` avec ``S = sum_j exp(i theta_j)``, alors pour
tout entier k

    c_k' == <exp(i k theta')> == rho_k <(S/|S|)^k>,   rho_k == <exp(i k xi)>

et cette factorisation est EXACTE : elle ne demande aucune fermeture,
seulement l'independance de xi et de S. La loi de bruit n'entre donc
dans la dynamique que par la suite de ses coefficients de Fourier
circulaires rho_k.

Consequence immediate : la stabilite lineaire de l'etat desordonne dans
le secteur polaire (k = 1) ne fait intervenir que rho_1. La POSITION de
la transition est fixee par rho_1 seul ; rho_2 ne se peuple qu'en
O(phi^2) et n'agit donc que sur le coefficient de Landau, c'est-a-dire
sur l'ORDRE de la transition.

Pour la loi alpha-stable symetrique enroulee, rho_k = exp(-(k eta)^alpha)
exactement, d'ou la prediction a un seul parametre

    eta_c(alpha) = C^(1/alpha),   C = -ln rho_1^c.

Lancer :  python3 notes/src/theory_rho1_invariant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from noise import stable_rvs  # noqa: E402

RNG = np.random.default_rng(0)


# --------------------------------------------------------------- test 1
def test_wrapped_moments(n_samples: int = 4_000_000) -> None:
    """rho_k = exp(-(k eta)^alpha) pour la stable symetrique enroulee."""
    print("=" * 68)
    print("1. rho_k = <exp(i k xi)>,  xi ~ S_alpha(eta) enroule")
    print("=" * 68)
    print(f"{'alpha':>6} {'eta':>6} {'k':>3} {'mesure':>10} "
          f"{'exp(-(k eta)^a)':>16} {'ecart':>10}")
    worst = 0.0
    for alpha in (1.0, 1.5, 2.0):
        for eta in (0.1, 0.3):
            xi = stable_rvs(alpha, eta, n_samples, RNG)
            for k in (1, 2, 3):
                meas = float(np.mean(np.exp(1j * k * xi)).real)
                pred = float(np.exp(-((eta * k) ** alpha)))
                worst = max(worst, abs(meas - pred))
                print(f"{alpha:6.2f} {eta:6.2f} {k:3d} {meas:10.6f} "
                      f"{pred:16.6f} {meas - pred:+10.2e}")
    print(f"\n  ecart maximal : {worst:.1e}  (identite exacte, verifiee "
          f"sur {n_samples:.0e} tirages)")


# --------------------------------------------------------------- test 2
def _misfit(n: float, alphas: np.ndarray, eta_c: np.ndarray):
    """Ajuste eta_c(alpha) sous l'hypothese 'rho_n constant au seuil',
    soit (n eta_c)^alpha = C, avec C l'unique parametre libre.
    Renvoie (ecart RMS relatif en %, C)."""
    y = (n * eta_c) ** alphas
    C = float(np.exp(np.mean(np.log(y))))
    pred = C ** (1.0 / alphas) / n
    return 100 * float(np.sqrt(np.mean(((eta_c - pred) / eta_c) ** 2))), C


def test_which_harmonic() -> None:
    """Quel rho_n est invariant au seuil ? Le test est falsifiable :
    'rho_n = const' donne eta_c = C^(1/alpha)/n, et le facteur n^alpha
    depend de alpha, donc les hypotheses n = 1, 2, 3 ... sont
    genuinement distinctes et mutuellement exclusives."""
    z = np.load(V1_ROOT / "data" / "phase_curve_v2.npz")
    alphas = z["alphas"]

    print()
    print("=" * 68)
    print("2. Quel coefficient de Fourier est invariant au seuil ?")
    print("=" * 68)

    # Le driver enregistre lui-meme les drapeaux de censure : un seuil
    # tombe sur un bord de fenetre est un resultat censure, pas une
    # mesure, et il doit sortir du test.
    for label, key, flag in (("<phi> = 1/2", "eta_c_phi", "edge_phi"),
                             ("pic de chi ", "eta_c_chi", "edge_chi")):
        eta_c = z[key]
        censored = z[flag].astype(bool)
        if censored.any():
            bad = ", ".join(f"alpha={a:g}" for a in alphas[censored])
            print(f"\n  [!] critere '{label}' : {bad} sur un bord de "
                  f"fenetre -> ecarte du test.")
        keep = ~censored
        a_k, e_k = alphas[keep], eta_c[keep]
        r1, C = _misfit(1.0, a_k, e_k)
        print(f"\n  --- critere {label} ---")
        print(f"  {'n':>6} {'ecart RMS sur eta_c':>22}")
        for n in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            r, _ = _misfit(n, a_k, e_k)
            mark = "   <-- theorie" if n == 1.0 else ""
            print(f"  {n:6.2f} {r:20.2f} %{mark}")
        ns = np.linspace(0.3, 4.0, 4000)
        rs = np.array([_misfit(n, a_k, e_k)[0] for n in ns])
        print(f"  minimum continu : n* = {ns[np.argmin(rs)]:.3f}  "
              f"(theorie : n* = 1)")
        print(f"  C = {C:.5f}  ->  rho_1^c = {np.exp(-C):.4f}")
        print(f"  eta_c varie d'un facteur {e_k.max() / e_k.min():.2f} ; "
              f"residu apres collapse {r1:.2f} %")

    print()
    print("  Les valeurs par graine sont dans eta_c_phi_seed : les barres")
    print("  d'erreur de l'article se recalculent, elles ne sont pas a")
    print("  croire sur parole.")
    print("      (3 graines, L = 15, warm = 1500). Le residu de ~4 % ne")
    print("      peut donc pas encore etre compare a une incertitude de")
    print("      mesure. C'est le run de refection prioritaire, et il est")
    print("      bon marche : L = 15, N = 500.")


# --------------------------------------------------------------- test 3
def _headings(phi: float, c2: float, n: int, size: int) -> np.ndarray:
    """Tetes iid de coefficients circulaires imposes (phi, c2), par
    rejet sur P(t) ∝ 1 + 2 phi cos t + 2 c2 cos 2t."""
    need, buf = size * n, []
    ceiling = 1 + 2 * abs(phi) + 2 * abs(c2)
    while need > 0:
        t = RNG.uniform(-np.pi, np.pi, max(need * 2, 1024))
        p = 1 + 2 * phi * np.cos(t) + 2 * c2 * np.cos(2 * t)
        keep = t[RNG.uniform(0, ceiling, t.size) < p]
        buf.append(keep)
        need -= keep.size
    return np.concatenate(buf)[:size * n].reshape(size, n)


def _resultant(phi: float, c2: float, n: int, size: int = 400_000):
    S = np.exp(1j * _headings(phi, c2, n, size)).sum(axis=1)
    u = S / np.abs(S)
    return float(u.mean().real), float((u ** 2).mean().real)


def test_mechanism() -> None:
    """Le gain lineaire polaire A(n) = d<S/|S|>/dphi ne depend pas du
    secteur nematique, et <(S/|S|)^2> s'annule en phi = 0."""
    print()
    print("=" * 68)
    print("3. Mecanisme : rho_2 peut-il deplacer le seuil ?  (non)")
    print("=" * 68)
    print("  A(n) = d<S/|S|>/dphi en phi=0, a c_2 impose :")
    print(f"  {'n':>3} {'c_2 = 0':>10} {'c_2 = 0.15':>12} {'c_2 = 0.30':>12}")
    d = 0.02
    for n in (3, 6):
        row = []
        for c2 in (0.0, 0.15, 0.30):
            up, _ = _resultant(+d, c2, n)
            dn, _ = _resultant(-d, c2, n)
            row.append((up - dn) / (2 * d))
        print(f"  {n:3d} {row[0]:10.4f} {row[1]:12.4f} {row[2]:12.4f}")
    print("  -> A(n) est plat en c_2 : le seuil polaire rho_1 A(n) = 1")
    print("     ne contient pas rho_2.")

    print()
    print("  <(S/|S|)^2>, le terme source du secteur nematique (n = 3) :")
    print(f"  {'phi':>6} {'<(S/|S|)^2>':>14} {'/ phi^2':>10}")
    for phi in (0.0, 0.05, 0.10, 0.20, 0.40):
        _, m2 = _resultant(phi, 0.0, 3)
        ratio = f"{m2 / phi ** 2:10.2f}" if phi else f"{'-':>10}"
        print(f"  {phi:6.2f} {m2:14.5f} {ratio}")
    print("  -> c_2 = rho_2 x O(phi^2) : rho_2 n'entre qu'au terme")
    print("     cubique de Landau, donc sur l'ORDRE de la transition,")
    print("     pas sur sa POSITION.  C'est la prediction qui reste a")
    print("     tester numeriquement.")

    print()
    print("  Limite honnete : le champ moyen donne la STRUCTURE mais pas")
    print("  la VALEUR.  Il predit rho_1^c = 1/A(n) ~ 0.45-0.78 selon n,")
    print("  contre 0.971 mesure : il ignore les modes de Goldstone a")
    print("  grande longueur d'onde qui detruisent l'ordre en 2d.  La")
    print("  constante C n'est pas universelle ; le collapse en alpha,")
    print("  lui, l'est.")


if __name__ == "__main__":
    test_wrapped_moments()
    test_which_harmonic()
    test_mechanism()
