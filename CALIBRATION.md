# CALIBRATION.md -- Bayesian calibration to experiment (auto-generated)

Calibrated (omega_s0, R, lambda_c) to Petitjean/Breunlich: omega_s_eff = 0.45+-0.05 %, X_mu = 113+-12,
via numpyro NUTS. See `openmucf/calibrate.py`.

## Weak omega_s0 prior (experimental data only) -- exposes the degeneracy
| parameter | mean | sd | 95% CI |
|---|---|---|---|
| omega_s_eff_pct | 0.461 | 0.045 | [0.371, 0.55] |
| lambda_c | 1.14e+08 | 2.02e+07 | [8.25e+07, 1.54e+08] |
| omega_s0_pct | 0.817 | 0.142 | [0.608, 1.08] |
| R | 0.419 | 0.112 | [0.184, 0.588] |

**omega_s0 - R correlation = 0.84.** The effective sticking (product) and lambda_c
are well constrained, but omega_s0 and R are strongly correlated (a positive ridge along fixed
omega_s0*(1-R)): the yield/sticking data pin the product, NOT the split. (Figure `figures/calibration.png`.)

## Informative omega_s0 prior (Kamimura 0.857+-0.03 %) -- partially resolves it
| parameter | mean | sd | 95% CI |
|---|---|---|---|
| omega_s0_pct | 0.856 | 0.0296 | [0.797, 0.914] |
| R | 0.461 | 0.0569 | [0.341, 0.567] |
| omega_s_eff_pct | 0.461 | 0.0463 | [0.374, 0.556] |

The Kamimura *theory* input tightens omega_s0 (sd 0.142 -> 0.0296 %)
and hence R -- but R still inherits that uncertainty.

## Finding
Experiment alone determines **effective sticking and the cycling rate**, not the microscopic
sticking/reactivation split. Separating omega_s0 from R -- and predicting how R changes at high density --
requires an independent microscopic calculation. **That is exactly the Phase-3 reactivation surrogate,**
and this degeneracy is the quantitative reason it is needed.
