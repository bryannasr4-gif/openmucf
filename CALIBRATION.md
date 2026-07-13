# CALIBRATION.md -- Bayesian calibration to experiment (auto-generated)

Calibrated (omega_s0, R, lambda_c) to Petitjean/Breunlich: omega_s_eff = 0.45+-0.05 %, X_mu = 113+-12,
via numpyro NUTS (4 sequential chains). See `openmucf/calibrate.py`.

The default prior boxes were WIDENED 2026-07-12 (disclosed statistical correction, I2-clean -- no target
involved): the previous boxes provably RAILED -- the old weak-chain 95% CI had R hi 0.588 against the old
R bound 0.60, and omega_s0 lo 0.608 against the old omega_s0 bound 0.60. New boxes: `R ~ Uniform(0.00,
0.80)` (was 0.10-0.60) and weak `omega_s0_pct ~ Uniform(0.50, 1.20)` (was 0.60-1.10); the lambda_c box is
UNCHANGED (0.8-1.6e8, a measured-band prior, not railing).

## Weak omega_s0 prior (experimental data only) -- exposes the degeneracy
| parameter | mean | sd | mcse | 95% CI |
|---|---|---|---|---|
| omega_s_eff_pct | 0.458 | 0.0456 | 0.000389 | [0.369, 0.548] |
| lambda_c | 1.14e+08 | 2.04e+07 | 2.31e+05 | [8.26e+07, 1.55e+08] |
| omega_s0_pct | 0.805 | 0.202 | 0.00279 | [0.513, 1.18] |
| R | 0.394 | 0.161 | 0.00228 | [0.0682, 0.639] |

**omega_s0 - R correlation = 0.91** (prior-conditional). The posterior concentrates
on the CURVE omega_s0*(1-R) = omega_s_eff (the product is pinned by the data); the linear (Pearson)
correlation is a descriptive statistic of that curved ridge and is prior-support-dependent -- it ranges
[0.83, 0.92] across the prior-sensitivity sweep below. The effective sticking (product)
and lambda_c are well constrained; omega_s0 and R are not separable. (Figure `figures/calibration.png`.)

## Informative omega_s0 prior (Kamimura 0.857+-0.03 %) -- partially resolves it
| parameter | mean | sd | mcse | 95% CI |
|---|---|---|---|---|
| omega_s0_pct | 0.856 | 0.0301 | 0.000313 | [0.798, 0.915] |
| R | 0.462 | 0.0585 | 0.000589 | [0.346, 0.576] |
| omega_s_eff_pct | 0.46 | 0.0473 | 0.000446 | [0.367, 0.552] |

The Kamimura *theory* input tightens omega_s0 (sd 0.202 -> 0.0301 %)
and hence R -- but R still inherits that uncertainty.

## Finding
Experiment alone determines **effective sticking and the cycling rate**, not the microscopic
sticking/reactivation split. Separating omega_s0 from R -- and predicting how R changes at high density --
requires an independent microscopic calculation. **That is exactly the Phase-3 reactivation surrogate,**
and this degeneracy is the quantitative reason it is needed.

## Convergence diagnostics (4 chains, sequential)
| chain | max r_hat | min ess | divergences |
|---|---|---|---|
| weak | 1.000 | 5e+03 | 0 |
| Kamimura | 1.000 | 9.2e+03 | 0 |

Convergence gate (`tests/test_calibrate.py::test_multichain_diagnostics`): max r_hat < 1.01, min ess > 400, divergences == 0 on the default (widened-box) chains.

## Prior-sensitivity sweep (weak-prior mode; 4 chains x 1000)
| config | boxes | corr | R sd | ose mean | ose sd | rails? |
|---|---|---|---|---|---|---|
| R=legacy/os0=legacy/lc=narrow | R[0.10,0.60] os0[0.60,1.10] lc[8e+07,2e+08] | 0.84 | 0.109 | 0.462 | 0.0443 | no |
| R=legacy/os0=legacy/lc=wide | R[0.10,0.60] os0[0.60,1.10] lc[6e+07,2e+08] | 0.83 | 0.11 | 0.462 | 0.0463 | no |
| R=legacy/os0=wide/lc=narrow | R[0.10,0.60] os0[0.50,1.20] lc[8e+07,2e+08] | 0.89 | 0.136 | 0.46 | 0.0446 | no |
| R=legacy/os0=wide/lc=wide | R[0.10,0.60] os0[0.50,1.20] lc[6e+07,2e+08] | 0.88 | 0.136 | 0.46 | 0.0449 | no |
| R=legacy/os0=widest/lc=narrow | R[0.10,0.60] os0[0.40,1.30] lc[8e+07,2e+08] | 0.90 | 0.145 | 0.459 | 0.0458 | no |
| R=legacy/os0=widest/lc=wide | R[0.10,0.60] os0[0.40,1.30] lc[6e+07,2e+08] | 0.89 | 0.142 | 0.459 | 0.0466 | no |
| R=wide/os0=legacy/lc=narrow | R[0.05,0.70] os0[0.60,1.10] lc[8e+07,2e+08] | 0.85 | 0.118 | 0.46 | 0.047 | no |
| R=wide/os0=legacy/lc=wide | R[0.05,0.70] os0[0.60,1.10] lc[6e+07,2e+08] | 0.84 | 0.115 | 0.458 | 0.0485 | no |
| R=wide/os0=wide/lc=narrow | R[0.05,0.70] os0[0.50,1.20] lc[8e+07,2e+08] | 0.90 | 0.156 | 0.457 | 0.0458 | no |
| R=wide/os0=wide/lc=wide | R[0.05,0.70] os0[0.50,1.20] lc[6e+07,2e+08] | 0.90 | 0.157 | 0.456 | 0.0471 | no |
| R=wide/os0=widest/lc=narrow | R[0.05,0.70] os0[0.40,1.30] lc[8e+07,2e+08] | 0.92 | 0.171 | 0.455 | 0.0458 | no |
| R=wide/os0=widest/lc=wide | R[0.05,0.70] os0[0.40,1.30] lc[6e+07,2e+08] | 0.92 | 0.173 | 0.455 | 0.0478 | no |
| R=widest/os0=legacy/lc=narrow | R[0.00,0.80] os0[0.60,1.10] lc[8e+07,2e+08] | 0.84 | 0.114 | 0.46 | 0.0468 | no |
| R=widest/os0=legacy/lc=wide | R[0.00,0.80] os0[0.60,1.10] lc[6e+07,2e+08] | 0.84 | 0.116 | 0.458 | 0.0481 | no |
| R=widest/os0=wide/lc=narrow | R[0.00,0.80] os0[0.50,1.20] lc[8e+07,2e+08] | 0.90 | 0.16 | 0.459 | 0.046 | no |
| R=widest/os0=wide/lc=wide | R[0.00,0.80] os0[0.50,1.20] lc[6e+07,2e+08] | 0.90 | 0.166 | 0.459 | 0.0482 | no |
| R=widest/os0=widest/lc=narrow | R[0.00,0.80] os0[0.40,1.30] lc[8e+07,2e+08] | 0.92 | 0.186 | 0.455 | 0.0451 | no |
| R=widest/os0=widest/lc=wide | R[0.00,0.80] os0[0.40,1.30] lc[6e+07,2e+08] | 0.92 | 0.184 | 0.455 | 0.0472 | no |
| default box; rho_obs=+0.5 | default; MultivariateNormal obs likelihood | 0.92 | 0.158 | 0.46 | 0.0388 | no |
| default box; rho_obs=-0.5 | default; MultivariateNormal obs likelihood | 0.90 | 0.161 | 0.453 | 0.0481 | no |

The product omega_s0(1-R) (= ose mean) and lambda_c are box-invariant; corr and R-width are support-dependent (corr range [0.83, 0.92] across the sweep) -- the degeneracy is the finding, its two-decimal value is not. `rails?` flags a 95% CI endpoint within 1% of its OWN box bound; it is `no` for every config here, INCLUDING the legacy boxes -- the old boxes' railing was the looser near-bound truncation quoted above (old committed CI R hi 0.588 just below 0.60; omega_s0 lo 0.608 just above 0.60, the old bound falling inside the posterior), removed by the widening; the residual support-dependence now shows as growing R-width, not hard railing. The last two rows treat the two observations as correlated (MultivariateNormal, rho_obs=+/-0.5) instead of independent Gaussians: the product and lambda_c stay pinned; the corr/width shift is the covariance sensitivity.

## Posterior pushforward (state-of-knowledge X_mu and Q_net)
| quantity | mean | sd | 95% CI |
|---|---|---|---|
| X_mu (weak-box posterior) | 115.7 | 9.25 | [98.71, 134.1] |
| X_mu (Kamimura posterior) | 115.6 | 9.12 | [98.7, 133.7] |
| Q_net (hybrid: posterior kinetics x ignorance-box economics) | 0.04905 | 0.0324 | [0.01178, 0.1387] |

X_mu is the DEFAULT-box weak-chain (and Kamimura-chain) `(omega_s_eff, lambda_c)` joint draws pushed through `1/(ose/100 + lambda_0/lambda_c)` -- the state-of-knowledge (posterior) interval, as opposed to the ignorance-box propagation in FINDINGS.md. The two chains agree (the product is data-pinned). Q_net multiplies the weak-box posterior X_mu by seeded-uniform draws over the FROZEN uq boxes for E_mu / eta_acc / eta_thermal (posterior kinetics x ignorance-box economics -- hybrid; the economics is an ignorance box, not a posterior).

## Channels-on re-attribution (ttmu) -- blocked

The joint ttmu loss RE-ATTRIBUTION refit is NOT run: the ttmu formation rate `lambda_ttmu` is blocked (0.0, needs_verification) -- pending acquisition of the Matsuzaki/Bom tt-fusion tables (*Muon Catal. Fusion*). When it lands, this chain adds the tt channel to BOTH likelihood terms (obs_ose observes the TOTAL per-cycle loss ose_pct + tt_pc*100 = 0.45%, and X_mu carries tt_pc), so the omega_s0(1-R) posterior shifts DOWN by the tt share while X_mu stays ~113 -- a joint refit under the anchor-total constraint, NOT new physics. See `docs/accounting.md` and MODEL_SPEC.md sec.4.1.
