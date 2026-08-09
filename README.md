# Matching one circular harmonic collapses the flocking threshold across angular-noise laws

Code and data to reproduce every figure and every number in the
manuscript of the same name (Y. Youssouf Yaya and D. Abakar, submitted
to *Chaos*).

## What the paper is about

Vicsek-type models of collective motion each pick an angular-noise
distribution — bounded uniform in the original model, Gaussian in the
hydrodynamic theories, something else again in most later variants — and
the literature compares them at equal *noise amplitude*. That quantity
is not comparable across distributions: width means different things for
different shapes.

What the transition responds to is the first circular Fourier
coefficient of the wrapped noise,

```
rho_1 = <cos(xi)>
```

and, as far as we can measure, nothing else. Sweeping wrapped
alpha-stable noise moves the critical amplitude by a factor of 187 while
`rho_1` at threshold holds to a few percent, and six noise laws matched
on `rho_1` alone — from run-and-tumble reorientation to a bounded
uniform kick — place the threshold within 1.4 parts in a thousand of one
another. Eight rival one-number summaries of the same law do far worse,
by factors of 60 to 3237 in chi-squared.

## Reproducing the paper

Requires Python 3.10+, then `pip install -r requirements.txt`.

```
make figures    # the six figures of the manuscript, into figures/
make numbers    # every number quoted in the text, printed to stdout
make check      # both
```

Neither target runs a simulation. They read the sweep data deposited in
`data/` and rebuild everything from it, which takes a few minutes — the
only slow part is the snapshot panel of Fig. 2, which re-simulates four
short runs because a snapshot is a configuration and not a summary.

To regenerate the sweep data from scratch:

```
make sweeps     # hours of CPU: ~3.5 h + ~4 h + ~3.5 h on 7 cores
```

## What is here

```
src/noise.py                    symmetric alpha-stable sampler (Chambers-Mallows-Stuck)
src/vicsek.py                   two-zone Vicsek-Couzin simulator, numba-accelerated
src/noise_families.py           the six noise laws, their rho_k, and the rho_1 matching
src/style.py                    figure conventions (Wong palette, journal sizing)

src/run_phasecurve_v2.py        alpha sweep; --L sets the system size
src/run_noise_families.py       the six laws on a common rho_1 grid

src/theory_rho1_invariant.py    which harmonic is conserved at threshold
src/test_selectivite_rho1.py    rho_1 against eight rival summaries
src/analyse_noise_families.py   the residual and its ordering by rho_2
src/make_figures_v3.py          Figs. 2-6
src/make_schematic.py           Fig. 1

data/phase_curve_v2.npz         9 alphas from 0.5 to 2, L = 15, 20 seeds
data/phase_curve_v2_L30.npz     4 alphas, L = 30, 12 seeds, identical protocol
data/noise_families.npz         6 laws x 23 values of rho_1 x 20 seeds
```

Each `.npz` stores per-seed values, not just means, so the error bars in
the paper can be recomputed rather than taken on trust.

## The protocol, in one paragraph

Every sweep uses the same one: warm-up of 3e4 steps from an aligned
start, measurement window of 1e4 steps, independent seeds, and a
threshold located from the crossing `<phi> = 1/2` interpolated in
`log(eta)` and computed once per seed, so the quoted uncertainty is a
seed-to-seed standard error rather than a fit error. The noise grid is
logarithmic and specific to each alpha, spanning a fixed window in units
of the expected threshold. A threshold falling on a window edge would be
a censored result rather than a measurement; the drivers detect that and
flag it, and no measurement in the paper is censored.

This uniformity is deliberate. Comparing sweeps taken under different
protocols is the failure this design exists to avoid.

## Two things the data will not tell you

The critical value of `rho_1` is **not** universal. It drifts by +0.0072
between `L = 15` and `L = 30`, fourteen to nineteen standard errors,
which is 3.4 times the entire spread produced by varying alpha. The
crossing of `<phi> = 1/2` at these sizes is a crossover contour, not a
critical point. What survives a change of size is the *relation* between
noise laws, not the constant.

A small residual survives the matching: the six laws do not share a
threshold exactly, and the departures are ordered by the second
harmonic. It is reproducible in sign and structure across both sizes and
all six laws, but its amplitude roughly doubles between them, so it is
not a coefficient of the noise law. We report it and do not interpret
it.

## Numerical notes

`rho_k = exp(-(k*eta)^alpha)` holds for the wrapped symmetric
alpha-stable law and is verified against the sampler to better than
1e-5 for `k = 1, 2, 3` over the whole range used, including
`alpha = 0.5`, where the largest unwrapped variate reaches ~4e9 and the
wrapping still retains six significant digits of the angle.

`test_selectivite_rho1.py` uses common random numbers for the
finite-difference derivatives. Without that, the Monte Carlo noise
dominates and the chi-squared values are not reproducible between runs.

## Licence

MIT. If you use this, please cite the paper.
