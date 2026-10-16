# CALIBRATION.md -- Bayesian calibration to experiment (auto-generated)

Calibrated (omega_s0, R, lambda_c, L_other) to the three summaries Breunlich et al. (1987) print for liquid D/T at 23 K: the effective sticking omega_s = 0.45+-0.05 % (p.331), the maximum cycle rate phi*lambda_c = 145+-12 us^-1 and the yield X_mu = 113+-10 (p.332), with their common +-8 % absolute calibration (Fig. 4 caption) entering all three.
The yield relation sees the total loss per cycle W = omega_s0*(1-R) + L_other, where L_other is the side-channel loss at the anchor fill (`openmucf.sidechannels`); 0.45 % measures the sticking alone, because the source subtracted those losses from the raw loss per cycle (p.331).
Estimator: deterministic quadrature (`openmucf.calibrate.posterior_quadrature`), which `make audit` checks within stated tolerances; NUTS on the same model is a cross-check (section "NUTS cross-check").

## AMENDMENT 2026-10-14 -- the anchor basis

Until this amendment the calibration fed the measured effective sticking into the yield relation as the total loss per cycle and left the measured cycle rate out of the likelihood; the yield's uncertainty was taken as +-12, where the source prints +-10.
With the paired values (0.45 %, 1.45e8 s^-1) that relation gives a yield of 130.9, 1.49 sigma from the measured 113 at +-12 and 1.79 sigma at +-10: the measurements alone did not force this correction; the source's definition of the quantities did.
The table compares each published cell before and after.

| cell | as printed before | now |
|---|---|---|
| convergence: Kamimura max r_hat | 1.000 | 1.002 |
| convergence: Kamimura min ess | 1e+04 | 3577 |
| convergence: weak max r_hat | 1.000 | 1.001 |
| convergence: weak min ess | 5e+03 | 2616 |
| Kamimura: R 95% CI | [0.346, 0.575] | [0.354, 0.586] |
| Kamimura: R mcse | 0.000556 | (column removed) |
| Kamimura: R mean | 0.462 | 0.475 |
| Kamimura: R sd | 0.058 | 0.0591 |
| Kamimura: omega_s0_pct 95% CI | [0.798, 0.914] | [0.797, 0.915] |
| Kamimura: omega_s0_pct mcse | 0.000291 | (column removed) |
| Kamimura: omega_s0_pct sd | 0.0298 | 0.03 |
| Kamimura: omega_s_eff_pct 95% CI | [0.367, 0.552] | [0.358, 0.546] |
| Kamimura: omega_s_eff_pct mcse | 0.000428 | (column removed) |
| Kamimura: omega_s_eff_pct mean | 0.46 | 0.449 |
| Kamimura: omega_s_eff_pct sd | 0.047 | 0.048 |
| pushforward: Q_net (hybrid: posterior kinetics x ignorance-box economics) 95% CI | [0.01178, 0.1387] | [0.01141, 0.1329] |
| pushforward: Q_net (hybrid: posterior kinetics x ignorance-box economics) mean | 0.04905 | 0.04728 |
| pushforward: Q_net (hybrid: posterior kinetics x ignorance-box economics) sd | 0.0324 | 0.0312 |
| pushforward: X_mu (Kamimura posterior) 95% CI | [98.6, 133.7] | [94.62, 129.6] |
| pushforward: X_mu (Kamimura posterior) mean | 115.6 | 111.2 |
| pushforward: X_mu (Kamimura posterior) sd | 9.18 | 8.96 |
| pushforward: X_mu (weak-box posterior) 95% CI | [98.71, 134.1] | [95, 129.7] |
| pushforward: X_mu (weak-box posterior) mean | 115.7 | 111.3 |
| pushforward: X_mu (weak-box posterior) sd | 9.25 | 8.88 |
| sweep: R=legacy/os0=legacy/lc=narrow R sd | 0.109 | 0.107 |
| sweep: R=legacy/os0=legacy/lc=narrow corr | 0.84 | 0.82 |
| sweep: R=legacy/os0=legacy/lc=narrow ose mean | 0.462 | 0.453 |
| sweep: R=legacy/os0=legacy/lc=narrow ose sd | 0.0443 | 0.0458 |
| sweep: R=legacy/os0=legacy/lc=wide R sd | 0.11 | 0.107 |
| sweep: R=legacy/os0=legacy/lc=wide boxes | R[0.10,0.60] os0[0.60,1.10] lc[6e+07,2e+08] | R[0.10,0.60] os0[0.60,1.10] lc[6e+07,2.4e+08] |
| sweep: R=legacy/os0=legacy/lc=wide corr | 0.83 | 0.82 |
| sweep: R=legacy/os0=legacy/lc=wide ose mean | 0.462 | 0.453 |
| sweep: R=legacy/os0=legacy/lc=wide ose sd | 0.0463 | 0.0458 |
| sweep: R=legacy/os0=wide/lc=narrow R sd | 0.136 | 0.135 |
| sweep: R=legacy/os0=wide/lc=narrow corr | 0.89 | 0.87 |
| sweep: R=legacy/os0=wide/lc=narrow ose mean | 0.46 | 0.451 |
| sweep: R=legacy/os0=wide/lc=narrow ose sd | 0.0446 | 0.0455 |
| sweep: R=legacy/os0=wide/lc=wide R sd | 0.136 | 0.135 |
| sweep: R=legacy/os0=wide/lc=wide boxes | R[0.10,0.60] os0[0.50,1.20] lc[6e+07,2e+08] | R[0.10,0.60] os0[0.50,1.20] lc[6e+07,2.4e+08] |
| sweep: R=legacy/os0=wide/lc=wide corr | 0.88 | 0.87 |
| sweep: R=legacy/os0=wide/lc=wide ose mean | 0.46 | 0.451 |
| sweep: R=legacy/os0=wide/lc=wide ose sd | 0.0449 | 0.0455 |
| sweep: R=legacy/os0=widest/lc=narrow R sd | 0.145 | 0.143 |
| sweep: R=legacy/os0=widest/lc=narrow corr | 0.90 | 0.89 |
| sweep: R=legacy/os0=widest/lc=narrow ose mean | 0.459 | 0.449 |
| sweep: R=legacy/os0=widest/lc=narrow ose sd | 0.0458 | 0.0474 |
| sweep: R=legacy/os0=widest/lc=wide R sd | 0.142 | 0.143 |
| sweep: R=legacy/os0=widest/lc=wide boxes | R[0.10,0.60] os0[0.40,1.30] lc[6e+07,2e+08] | R[0.10,0.60] os0[0.40,1.30] lc[6e+07,2.4e+08] |
| sweep: R=legacy/os0=widest/lc=wide ose mean | 0.459 | 0.449 |
| sweep: R=legacy/os0=widest/lc=wide ose sd | 0.0466 | 0.0474 |
| sweep: R=wide/os0=legacy/lc=narrow R sd | 0.118 | 0.115 |
| sweep: R=wide/os0=legacy/lc=narrow corr | 0.85 | 0.84 |
| sweep: R=wide/os0=legacy/lc=narrow ose mean | 0.46 | 0.449 |
| sweep: R=wide/os0=legacy/lc=narrow ose sd | 0.047 | 0.0479 |
| sweep: R=wide/os0=legacy/lc=wide boxes | R[0.05,0.70] os0[0.60,1.10] lc[6e+07,2e+08] | R[0.05,0.70] os0[0.60,1.10] lc[6e+07,2.4e+08] |
| sweep: R=wide/os0=legacy/lc=wide ose mean | 0.458 | 0.449 |
| sweep: R=wide/os0=legacy/lc=wide ose sd | 0.0485 | 0.0479 |
| sweep: R=wide/os0=wide/lc=narrow R sd | 0.156 | 0.154 |
| sweep: R=wide/os0=wide/lc=narrow ose mean | 0.457 | 0.447 |
| sweep: R=wide/os0=wide/lc=narrow ose sd | 0.0458 | 0.0468 |
| sweep: R=wide/os0=wide/lc=wide R sd | 0.157 | 0.154 |
| sweep: R=wide/os0=wide/lc=wide boxes | R[0.05,0.70] os0[0.50,1.20] lc[6e+07,2e+08] | R[0.05,0.70] os0[0.50,1.20] lc[6e+07,2.4e+08] |
| sweep: R=wide/os0=wide/lc=wide ose mean | 0.456 | 0.447 |
| sweep: R=wide/os0=wide/lc=wide ose sd | 0.0471 | 0.0468 |
| sweep: R=wide/os0=widest/lc=narrow R sd | 0.171 | 0.175 |
| sweep: R=wide/os0=widest/lc=narrow ose mean | 0.455 | 0.445 |
| sweep: R=wide/os0=widest/lc=narrow ose sd | 0.0458 | 0.0469 |
| sweep: R=wide/os0=widest/lc=wide R sd | 0.173 | 0.175 |
| sweep: R=wide/os0=widest/lc=wide boxes | R[0.05,0.70] os0[0.40,1.30] lc[6e+07,2e+08] | R[0.05,0.70] os0[0.40,1.30] lc[6e+07,2.4e+08] |
| sweep: R=wide/os0=widest/lc=wide ose mean | 0.455 | 0.445 |
| sweep: R=wide/os0=widest/lc=wide ose sd | 0.0478 | 0.0469 |
| sweep: R=widest/os0=legacy/lc=narrow R sd | 0.114 | 0.115 |
| sweep: R=widest/os0=legacy/lc=narrow ose mean | 0.46 | 0.449 |
| sweep: R=widest/os0=legacy/lc=narrow ose sd | 0.0468 | 0.048 |
| sweep: R=widest/os0=legacy/lc=wide R sd | 0.116 | 0.115 |
| sweep: R=widest/os0=legacy/lc=wide boxes | R[0.00,0.80] os0[0.60,1.10] lc[6e+07,2e+08] | R[0.00,0.80] os0[0.60,1.10] lc[6e+07,2.4e+08] |
| sweep: R=widest/os0=legacy/lc=wide ose mean | 0.458 | 0.449 |
| sweep: R=widest/os0=legacy/lc=wide ose sd | 0.0481 | 0.048 |
| sweep: R=widest/os0=wide/lc=narrow ose mean | 0.459 | 0.448 |
| sweep: R=widest/os0=wide/lc=narrow ose sd | 0.046 | 0.0473 |
| sweep: R=widest/os0=wide/lc=wide R sd | 0.166 | 0.16 |
| sweep: R=widest/os0=wide/lc=wide boxes | R[0.00,0.80] os0[0.50,1.20] lc[6e+07,2e+08] | R[0.00,0.80] os0[0.50,1.20] lc[6e+07,2.4e+08] |
| sweep: R=widest/os0=wide/lc=wide ose mean | 0.459 | 0.448 |
| sweep: R=widest/os0=wide/lc=wide ose sd | 0.0482 | 0.0473 |
| sweep: R=widest/os0=widest/lc=narrow R sd | 0.186 | 0.189 |
| sweep: R=widest/os0=widest/lc=narrow ose mean | 0.455 | 0.445 |
| sweep: R=widest/os0=widest/lc=narrow ose sd | 0.0451 | 0.0472 |
| sweep: R=widest/os0=widest/lc=wide R sd | 0.184 | 0.189 |
| sweep: R=widest/os0=widest/lc=wide boxes | R[0.00,0.80] os0[0.40,1.30] lc[6e+07,2e+08] | R[0.00,0.80] os0[0.40,1.30] lc[6e+07,2.4e+08] |
| sweep: R=widest/os0=widest/lc=wide ose mean | 0.455 | 0.445 |
| sweep: default box; rho_obs=+0.5 R sd | 0.158 | 0.156 |
| sweep: default box; rho_obs=+0.5 boxes | default; MultivariateNormal obs likelihood | default |
| sweep: default box; rho_obs=+0.5 ose mean | 0.46 | 0.445 |
| sweep: default box; rho_obs=+0.5 ose sd | 0.0388 | 0.0383 |
| sweep: default box; rho_obs=+0.5 rails? | no | - |
| sweep: default box; rho_obs=-0.5 R sd | 0.161 | 0.158 |
| sweep: default box; rho_obs=-0.5 boxes | default; MultivariateNormal obs likelihood | default |
| sweep: default box; rho_obs=-0.5 corr | 0.90 | 0.91 |
| sweep: default box; rho_obs=-0.5 ose mean | 0.453 | 0.447 |
| sweep: default box; rho_obs=-0.5 ose sd | 0.0481 | 0.043 |
| sweep: default box; rho_obs=-0.5 rails? | no | - |
| weak: R 95% CI | [0.0682, 0.639] | [0.0794, 0.648] |
| weak: R mcse | 0.00228 | (column removed) |
| weak: R mean | 0.394 | 0.405 |
| weak: R sd | 0.161 | 0.16 |
| weak: lambda_c 95% CI | [8.26e+07, 1.55e+08] | [1.22e+08, 1.67e+08] |
| weak: lambda_c mcse | 2.31e+05 | (column removed) |
| weak: lambda_c mean | 1.14e+08 | 1.44e+08 |
| weak: lambda_c sd | 2.04e+07 | 1.15e+07 |
| weak: omega_s0_pct 95% CI | [0.513, 1.18] | [0.512, 1.17] |
| weak: omega_s0_pct mcse | 0.00279 | (column removed) |
| weak: omega_s0_pct mean | 0.805 | 0.802 |
| weak: omega_s0_pct sd | 0.202 | 0.2 |
| weak: omega_s_eff_pct 95% CI | [0.369, 0.548] | [0.358, 0.543] |
| weak: omega_s_eff_pct mcse | 0.000389 | (column removed) |
| weak: omega_s_eff_pct mean | 0.458 | 0.448 |
| weak: omega_s_eff_pct sd | 0.0456 | 0.0473 |

## Prior boxes

The default prior boxes were WIDENED 2026-07-12 (a disclosed statistical correction -- no target
involved): the previous boxes provably RAILED -- the old weak-chain 95% CI had R hi 0.588 against the old
R bound 0.60, and omega_s0 lo 0.608 against the old omega_s0 bound 0.60. New boxes: `R ~ Uniform(0.00,
0.80)` (was 0.10-0.60) and weak `omega_s0_pct ~ Uniform(0.50, 1.20)` (was 0.60-1.10); the lambda_c box
was then unchanged (0.8-1.6e8); with lambda_c in the likelihood since the amendment above it is
(0.8-2.0e8), because the narrower box truncated the posterior.

## Weak omega_s0 prior (experimental data only) -- exposes the degeneracy
| parameter | mean | sd | 2.5% | median | 97.5% |
|---|---|---|---|---|---|
| omega_s_eff_pct | 0.448 | 0.0473 | 0.358 | 0.447 | 0.543 |
| lambda_c | 1.44e+08 | 1.15e+07 | 1.22e+08 | 1.43e+08 | 1.67e+08 |
| omega_s0_pct | 0.802 | 0.2 | 0.512 | 0.777 | 1.17 |
| R | 0.405 | 0.16 | 0.0794 | 0.426 | 0.648 |
| W_pct | 0.585 | 0.0486 | 0.495 | 0.584 | 0.684 |
| L_other_pct | 0.137 | 0.0183 | 0.101 | 0.137 | 0.173 |

**omega_s0 - R correlation = 0.90** (prior-conditional). The posterior concentrates
on the CURVE omega_s0*(1-R) = omega_s_eff (the product is pinned by the data); the linear (Pearson)
correlation is a descriptive statistic of that curved ridge and is prior-support-dependent -- it ranges
[0.82, 0.92] across the prior-sensitivity sweep below. The effective sticking (product)
and lambda_c are well constrained; omega_s0 and R are not separable. (Figure `figures/calibration.png`.)

## Informative omega_s0 prior (Kamimura 0.857+-0.03 %) -- partially resolves it
| parameter | mean | sd | 2.5% | median | 97.5% |
|---|---|---|---|---|---|
| omega_s0_pct | 0.856 | 0.03 | 0.797 | 0.856 | 0.915 |
| R | 0.475 | 0.0591 | 0.354 | 0.477 | 0.586 |
| omega_s_eff_pct | 0.449 | 0.048 | 0.358 | 0.448 | 0.546 |

The Kamimura *theory* input tightens omega_s0 (sd 0.2 -> 0.03 %)
and hence R -- but R still inherits that uncertainty.

## AMENDMENT 2026-09-22 -- the Kamimura prior bounded to the weak-prior support

The Kamimura chain uses a Normal with location 0.857 and scale 0.03, truncated to the weak prior's support [0.5, 1.2].
Within that support the prior is unchanged at double precision; the bound removes the negative-omega_s0 region in which a chain could trap, so the convergence cells and their audit describe this sampler across seeds.
The Kamimura cells are a new realization replacing those in this document as released in v1.2.0.
The FC-001 forecast retains its registered untruncated prior.

## Finding
Experiment alone determines **effective sticking and the cycling rate**, not the microscopic
sticking/reactivation split. Separating omega_s0 from R -- and predicting how R changes at high density --
requires an independent microscopic calculation. **That is exactly the Phase-3 reactivation surrogate,**
and this degeneracy is the quantitative reason it is needed.

## Estimator and convergence

The cells are computed by `openmucf.calibrate.posterior_quadrature` at res 1, on midpoint grids of 8000 x 1600 x 120 cells in (omega_s_eff, lambda_c, L_other) and 1600 x 1600 cells in (omega_s0, R), and are stored at full precision in `CALIBRATION_CELLS.json`.
The table gives each posterior's largest relative change between that resolution and twice it, per class of cell, and the criterion of each class.

| posterior | means and sds | quantiles (not L_other) | L_other quantiles |
|---|---|---|---|
| fc001_card_support | 4e-07 | 2.4e-05 | 4.9e-05 |
| fc001_widened_support | 3.4e-07 | 2e-05 | 0.00017 |
| kamimura | 3.1e-07 | 1.9e-05 | 0.00017 |
| weak | 1.1e-07 | 1.9e-05 | 0.00015 |
| criterion | 1e-06 | 0.001 | 0.005 |

`make audit` recomputes every cell and accepts it within min(cap, max(10 x the measured cross-platform scatter, 1e-06)) relative; the scatter was measured between Windows AMD64 (Python 3.12.10, numpy 2.5.0) and Linux x86_64 (Python 3.14.4, numpy 2.5.3).

| class | cap | measured scatter | tolerance |
|---|---|---|---|
| means, sds and correlations | 1e-05 | 2.3e-14 | 1e-06 |
| quantiles (not L_other) | 0.005 | 4.2e-15 | 1e-06 |
| L_other quantiles | 0.02 | 0 | 1e-06 |

## NUTS cross-check (4 chains, sequential)
| chain | max r_hat | min ess | divergences | max offset of a mean from quadrature, in MCSE |
|---|---|---|---|---|
| weak | 1.001 | 2616 | 0 | 1.15 |
| Kamimura | 1.002 | 3577 | 0 | 1.09 |

NUTS on the same priors and likelihood (`openmucf.calibrate.model_w`, 1000 warmup and 2000 draws per chain) is an independent check of the quadrature: it passes when max r_hat < 1.01, min ess > 400, no draw diverges and every mean lies within 3 MCSE of the quadrature mean.
`make audit` reruns it, accepting max r_hat within 2% of this record, a fresh min ess of at least 2000 (`min ess` above describes this run and is not a reproduction target), divergences exactly as recorded, and the same agreement with the quadrature.

## Prior-sensitivity sweep (weak-prior mode; quadrature at res 0.5)
| config | boxes | corr | R sd | ose mean | ose sd | rails? |
|---|---|---|---|---|---|---|
| R=legacy/os0=legacy/lc=narrow | R[0.10,0.60] os0[0.60,1.10] lc[8e+07,2e+08] | 0.82 | 0.107 | 0.453 | 0.0458 | no |
| R=legacy/os0=legacy/lc=wide | R[0.10,0.60] os0[0.60,1.10] lc[6e+07,2.4e+08] | 0.82 | 0.107 | 0.453 | 0.0458 | no |
| R=legacy/os0=wide/lc=narrow | R[0.10,0.60] os0[0.50,1.20] lc[8e+07,2e+08] | 0.87 | 0.135 | 0.451 | 0.0455 | no |
| R=legacy/os0=wide/lc=wide | R[0.10,0.60] os0[0.50,1.20] lc[6e+07,2.4e+08] | 0.87 | 0.135 | 0.451 | 0.0455 | no |
| R=legacy/os0=widest/lc=narrow | R[0.10,0.60] os0[0.40,1.30] lc[8e+07,2e+08] | 0.89 | 0.143 | 0.449 | 0.0474 | no |
| R=legacy/os0=widest/lc=wide | R[0.10,0.60] os0[0.40,1.30] lc[6e+07,2.4e+08] | 0.89 | 0.143 | 0.449 | 0.0474 | no |
| R=wide/os0=legacy/lc=narrow | R[0.05,0.70] os0[0.60,1.10] lc[8e+07,2e+08] | 0.84 | 0.115 | 0.449 | 0.0479 | no |
| R=wide/os0=legacy/lc=wide | R[0.05,0.70] os0[0.60,1.10] lc[6e+07,2.4e+08] | 0.84 | 0.115 | 0.449 | 0.0479 | no |
| R=wide/os0=wide/lc=narrow | R[0.05,0.70] os0[0.50,1.20] lc[8e+07,2e+08] | 0.90 | 0.154 | 0.447 | 0.0468 | no |
| R=wide/os0=wide/lc=wide | R[0.05,0.70] os0[0.50,1.20] lc[6e+07,2.4e+08] | 0.90 | 0.154 | 0.447 | 0.0468 | no |
| R=wide/os0=widest/lc=narrow | R[0.05,0.70] os0[0.40,1.30] lc[8e+07,2e+08] | 0.92 | 0.175 | 0.445 | 0.0469 | no |
| R=wide/os0=widest/lc=wide | R[0.05,0.70] os0[0.40,1.30] lc[6e+07,2.4e+08] | 0.92 | 0.175 | 0.445 | 0.0469 | no |
| R=widest/os0=legacy/lc=narrow | R[0.00,0.80] os0[0.60,1.10] lc[8e+07,2e+08] | 0.84 | 0.115 | 0.449 | 0.048 | no |
| R=widest/os0=legacy/lc=wide | R[0.00,0.80] os0[0.60,1.10] lc[6e+07,2.4e+08] | 0.84 | 0.115 | 0.449 | 0.048 | no |
| R=widest/os0=wide/lc=narrow | R[0.00,0.80] os0[0.50,1.20] lc[8e+07,2e+08] | 0.90 | 0.16 | 0.448 | 0.0473 | no |
| R=widest/os0=wide/lc=wide | R[0.00,0.80] os0[0.50,1.20] lc[6e+07,2.4e+08] | 0.90 | 0.16 | 0.448 | 0.0473 | no |
| R=widest/os0=widest/lc=narrow | R[0.00,0.80] os0[0.40,1.30] lc[8e+07,2e+08] | 0.92 | 0.189 | 0.445 | 0.0472 | no |
| R=widest/os0=widest/lc=wide | R[0.00,0.80] os0[0.40,1.30] lc[6e+07,2.4e+08] | 0.92 | 0.189 | 0.445 | 0.0472 | no |
| default box; L2 likelihood | default | 0.92 | 0.156 | 0.445 | 0.0383 | - |
| default box; L1 likelihood | default | 0.91 | 0.158 | 0.447 | 0.043 | - |

The product omega_s0(1-R) (= ose mean) spans [0.445, 0.453] across the sweep; corr and R-width are support-dependent (corr range [0.82, 0.92] across the sweep) -- the degeneracy is the finding, its two-decimal value is not. `rails?` flags a 95% interval endpoint within 1% of its OWN box bound; it is `no` for every config here. The last two rows refit the default box with the likelihood's covariance replaced: L2 keeps only the cycle-rate/yield correlation the yield formula implies, L1 treats the three summaries as independent.


## Posterior pushforward (state-of-knowledge X_mu and Q_net)
| quantity | mean | sd | 95% CI |
|---|---|---|---|
| X_mu (weak-box posterior) | 111.3 | 8.88 | [95, 129.7] |
| X_mu (Kamimura posterior) | 111.2 | 8.96 | [94.62, 129.6] |
| Q_net (posterior X_mu x frozen uq economics boxes; deterministic) | 0.04728 | 0.0312 | [0.01141, 0.1329] |

X_mu is the weak-prior (and Kamimura-prior) posterior of `1/(W/100 + lambda_0/lambda_c)` -- the state-of-knowledge (posterior) interval, as opposed to the ignorance-box propagation in FINDINGS.md. Q_net multiplies the weak-box posterior X_mu by the FROZEN uq boxes for E_mu / eta_acc / eta_thermal, each uniform: mean and sd in closed form, quantiles by Gauss-Legendre quadrature over the three boxes against the posterior X_mu distribution (posterior kinetics x ignorance-box economics; the economics is an ignorance box, not a posterior).

## Side-channel loss at the anchor fill
| channel | loss per cycle (%) |
|---|---|
| tt (tmut) | 0.0641 |
| dd (dmud, 3He + n branch) | 0.0154 |
| protium (pmud + pmut) | 0.0493 |
| impurities and 3He | 0.0100 |
| total | 0.1388 |
| sd | 0.0197 |

L_other is computed at the anchor fill's conditions (c_t, normalized cycle rate and protium fraction) and is not carried to other conditions; FINDINGS.md section 3 uses it only where it says so.
