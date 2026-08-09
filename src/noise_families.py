"""Familles de bruit angulaire a coefficients de Fourier circulaires
prescrits.

Motivation (cf. PLAN_REVISION_RADICALE.md, §D bis) : la mise a jour
etant ``theta' = arg(S) + xi`` avec xi independant de la configuration,
on a EXACTEMENT, pour tout entier k,

    c_k' = rho_k <(S/|S|)^k>,     rho_k = <exp(i k xi)>.

La loi de bruit n'entre donc dans la dynamique que par la suite rho_k.
Le secteur polaire (k=1) ne contient que rho_1 : la POSITION de la
transition est fixee par rho_1 seul.  rho_2 ne se peuple qu'en O(phi^2)
et n'agit que sur le coefficient de Landau, donc sur l'ORDRE.

Ce module permet de tester les deux affirmations separement, en
comparant des familles APPARIEES SUR rho_1 mais de rho_2 differents.

Pourquoi rho_2 est le bon axe
-----------------------------
Pour tout bruit de faible amplitude et de variance angulaire finie,
rho_k = 1 - k^2 s^2/2 + ..., donc rho_2 -> 1 - 4(1 - rho_1) : toutes les
lois "etroites" s'ecrasent sur la meme valeur.  En sortir demande des
sauts rares et grands.  A rho_1 fixe, l'axe rho_2 EST l'axe
"beaucoup de petits ecarts" contre "quelques grands sauts" -- ce que
l'indice de stabilite alpha parametre continument, et dont le tumble
est le point extreme.  C'est la reformulation correcte de ce que le
manuscrit appelait, a tort, "heavy-tailed angular noise".

Familles
--------
=================  ==================================  ==================
famille            rho_k                               rho_2 a rho_1=.9711
=================  ==================================  ==================
tumble             rho_1  (independant de k)           0.9711
stable(alpha=1)    rho_1**k                            0.9430
stable(alpha=1.5)  rho_1**(k**1.5)                     0.9199
stable(alpha=2)    rho_1**(k**2)                       0.8894
uniform            sin(k D/2)/(k D/2)                  0.8880
twopoint           cos(k d)                            0.8853
=================  ==================================  ==================

Le "tumble" est le modele run-and-tumble : avec probabilite q la tete
est completement randomisee, sinon elle est laissee intacte.  C'est
exactement la cinematique d'E. coli citee en introduction du
manuscrit -- la famille est donc motivee, pas ad hoc.

Auto-test :  python3 notes/src/noise_families.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noise import stable_rvs

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------
# rho_k analytiques
# --------------------------------------------------------------------
def rho_k(family: str, param: float, k: int, alpha: float = 2.0) -> float:
    """k-ieme coefficient de Fourier circulaire, forme fermee."""
    k = int(k)
    if family == "stable":
        return float(np.exp(-((param * k) ** alpha)))
    if family == "uniform":                      # xi ~ U(-param/2, param/2)
        x = k * param / 2.0
        return 1.0 if x == 0 else float(np.sin(x) / x)
    if family == "tumble":                       # param = rho_1 = 1 - q
        return float(param)
    if family == "twopoint":                     # xi = +/- param
        return float(np.cos(k * param))
    raise ValueError(f"famille inconnue : {family}")


def invert_rho1(family: str, target: float, alpha: float = 2.0) -> float:
    """Parametre de la famille donnant rho_1 = target."""
    if not 0.0 < target < 1.0:
        raise ValueError("rho_1 doit etre dans (0, 1)")
    if family == "stable":
        return float((-np.log(target)) ** (1.0 / alpha))
    if family == "tumble":
        return float(target)
    if family == "twopoint":
        return float(np.arccos(target))
    if family == "uniform":
        # sin(x)/x decroit strictement de 1 a 0 sur (0, pi) ; x = D/2.
        lo, hi = 1e-12, np.pi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.sin(mid) / mid > target:
                lo = mid
            else:
                hi = mid
        return float(2.0 * 0.5 * (lo + hi))
    raise ValueError(f"famille inconnue : {family}")


# --------------------------------------------------------------------
# echantillonneurs
# --------------------------------------------------------------------
def sample(family: str, param: float, size: int, rng: np.random.Generator,
           alpha: float = 2.0) -> np.ndarray:
    """Tire l'increment angulaire (non enroule ; l'appelant enroule)."""
    if family == "stable":
        return stable_rvs(alpha, param, size, rng)
    if family == "uniform":
        return rng.uniform(-param / 2.0, param / 2.0, size)
    if family == "tumble":
        q = 1.0 - param
        xi = np.zeros(size)
        hit = rng.random(size) < q
        xi[hit] = rng.uniform(-np.pi, np.pi, int(hit.sum()))
        return xi
    if family == "twopoint":
        return param * np.where(rng.random(size) < 0.5, -1.0, 1.0)
    raise ValueError(f"famille inconnue : {family}")


@dataclass(frozen=True)
class Family:
    """Une famille appariee a une valeur cible de rho_1."""
    name: str
    alpha: float
    param: float
    rho1: float

    @property
    def label(self) -> str:
        if self.name == "stable":
            return f"stable(a={self.alpha:g})"
        return self.name

    def rho(self, k: int) -> float:
        return rho_k(self.name, self.param, k, self.alpha)

    def draw(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return sample(self.name, self.param, size, rng, self.alpha)


#: jeu standard, du rho_2 maximal au minimal a rho_1 donne
SPEC = [("tumble", 2.0), ("stable", 1.0), ("stable", 1.5),
        ("stable", 2.0), ("uniform", 2.0), ("twopoint", 2.0)]


def matched_set(rho1: float, spec=SPEC) -> list[Family]:
    """Construit les familles appariees sur rho_1, triees par rho_2
    decroissant."""
    fams = [Family(n, a, invert_rho1(n, rho1, a), rho1) for n, a in spec]
    return sorted(fams, key=lambda f: -f.rho(2))


# --------------------------------------------------------------------
# auto-test
# --------------------------------------------------------------------
def _selftest(n: int = 4_000_000) -> None:
    rng = np.random.default_rng(0)
    rho1 = 0.9711                      # valeur mesuree au seuil
    fams = matched_set(rho1)

    print("=" * 74)
    print(f"Familles appariees a rho_1 = {rho1}")
    print("=" * 74)
    print(f"{'famille':>18} {'parametre':>11} "
          f"{'rho_1 mes':>10} {'rho_2 the':>10} {'rho_2 mes':>10} "
          f"{'rho_3 the':>10}")
    for f in fams:
        xi = f.draw(n, rng)
        m1 = float(np.mean(np.exp(1j * xi)).real)
        m2 = float(np.mean(np.exp(2j * xi)).real)
        assert abs(m1 - rho1) < 5e-3, (f.label, m1)
        assert abs(m2 - f.rho(2)) < 5e-3, (f.label, m2, f.rho(2))
        print(f"{f.label:>18} {f.param:11.5f} {m1:10.5f} "
              f"{f.rho(2):10.5f} {m2:10.5f} {f.rho(3):10.5f}")

    span = max(f.rho(2) for f in fams) - min(f.rho(2) for f in fams)
    print(f"\n  rho_1 identique a {rho1}; rho_2 balaye {span:.4f} "
          f"({100 * span / rho1:.1f} % de rho_1).")
    print("  Prediction : le seuil est le MEME pour les six familles.")
    print("  Tout residu doit s'ordonner selon rho_2, et se voir dans")
    print("  l'ORDRE de la transition (U_4, bimodalite de P(phi)),")
    print("  pas dans sa POSITION.")

    print()
    print("=" * 74)
    print("Limite petit bruit : toutes les lois etroites s'ecrasent")
    print("=" * 74)
    print(f"{'rho_1':>8} {'1-4(1-rho_1)':>14} "
          + " ".join(f"{f.label:>16}" for f in fams))
    for r in (0.999, 0.99, 0.9711, 0.90, 0.70):
        row = " ".join(f"{Family(f.name, f.alpha, invert_rho1(f.name, r, f.alpha), r).rho(2):16.5f}"
                       for f in fams)
        print(f"{r:8.4f} {1 - 4 * (1 - r):14.5f} {row}")
    print("\n  -> a rho_1 -> 1 les familles etroites convergent vers")
    print("     1 - 4(1 - rho_1) ; seul le tumble s'en detache.  L'ecart")
    print("     est donc maximal a bruit fort, ce qui est justement le")
    print("     regime de la transition.")


if __name__ == "__main__":
    _selftest()
