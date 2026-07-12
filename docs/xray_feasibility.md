# X-ray/neutron-ratio degeneracy-breaker — feasibility scan

> **Exploratory note — NOT part of the reproducibility audit.** Every number here is NUTS-derived and
> reproduces only to Monte-Carlo noise (≈±1–3 percentage points on a contraction), not byte-identically.
> This document ships no manifest and is not byte-diffed by `make audit`. It is a go/no-go scan run
> *before* any effort is spent acquiring the (blocked) κ branching literature or engineering a likelihood
> term. Regenerate with `python scripts/xray_feasibility.py` (seeded, deterministic on a fixed platform).

## Verdict (first)

**Feasible.** Adding an X-ray/neutron-ratio observable (a count of K X-rays per fusion neutron, ∝ κ·ω_s⁰)
to the calibration contracts the posterior sd of the reactivation coefficient **R** by **≈43 %** at the
best grid cell (κ-band relative half-width `w=0.10`, measurement precision `σ_rel=0.02`) **in the
weak-prior calibration — the chain that actually exposes the ω_s⁰/R degeneracy** this observable exists to
break. In the informative-theory-prior (Kamimura) calibration the same measurement is **redundant (≈1 %)**,
because that chain has already pinned ω_s⁰ from theory. Against the pre-registered threshold (a best-cell
contraction of **≥ 15 %** counts as feasible), the degeneracy-exposing chain gives **42.95 % ≥ 15 % →
pursue**; the theory-prior chain gives **1.41 % < 15 %**.

The split between the two chains is itself the finding: **the X-ray observable is valuable precisely to the
extent one does *not* assume the contested theoretical sticking value** — i.e. its worth is as an
*independent* check of that value, not as an add-on when the value is taken as known. Because the value of a
measurement is judged against the state of knowledge one is willing to defend *without* it, and the
initial-sticking figure ω_s⁰ = 0.857 % is itself flagged *contested* in the ledger, the operative verdict
is **pursue the κ literature and define a Phase-3 acceptance test** (below).

## The construction (why the blocked κ papers are not needed to decide this)

The calibration in `openmucf.calibrate` constrains the product ω_s^eff = ω_s⁰·(1−R) and λ_c, but not ω_s⁰
and R separately — they sit on a strongly correlated (+0.84) ridge. A K-X-ray-per-neutron ratio is
proportional to κ·ω_s⁰, where κ (K X-rays emitted per initial sticking event) is known only to a band. We
add one observable to `calibrate.model`:

```
obs_ratio ~ Normal(kappa * omega_s0_fraction, sigma_rel * ratio_true)
kappa     ~ Uniform(kappa_mid*(1 - w), kappa_mid*(1 + w))
```

where `ratio_true = kappa_true * omega_s0_true_fraction` is the data-generating value of the **observable**
(so the noise is relative to the observable, not to ω_s⁰ alone or a fixed constant), `kappa_true =
kappa_mid`, and the synthetic observation is placed at its expected value `obs_ratio = ratio_true`. Because
the term constrains the *product* κ·ω_s⁰, the achievable contraction depends only on the **relative** κ-band
width `w` and the **relative** measurement precision `σ_rel` — **not** on κ's unknown central value.
`kappa_mid` is therefore an arbitrary scale that divides out; we verify this numerically below (κ_mid = 1.0
vs 0.5 give identical contractions to Monte-Carlo noise). This is what makes the scan decidable while the
Cohen-1988 / Markushin-1988 / Rafelski-1989 κ figures remain unavailable.

**Settings (seeded, deterministic):** NUTS warmup = 1000, samples = 4000, seed = 0 (matching the repo
calibration chains); ω_s⁰_true = 0.857 % (the physical data-generating sticking, used identically for both
chains — only the ω_s⁰ *prior* differs); grid `w ∈ {0.10, 0.30, 0.60} × σ_rel ∈ {0.02, 0.05, 0.10}`.
A run-time guard asserts the extended model with the ratio term switched off reproduces the package
`calibrate.model` sd(R) exactly (0.108847 vs 0.108847, |Δ| = 0).

## Results — sd(R) contraction vs the no-ratio baseline (%)

**Weak-prior chain** (exposes the ω_s⁰/R degeneracy) — baseline sd(R) = 0.10885:

| w \ σ_rel | 0.02 | 0.05 | 0.10 |
|---|---|---|---|
| **0.10** | **42.95** | 38.99 | 25.74 |
| **0.30** | 14.61 | 11.36 | 10.32 |
| **0.60** | −2.30 | −5.33 | 1.07 |

**Kamimura theory-prior chain** — baseline sd(R) = 0.05750:

| w \ σ_rel | 0.02 | 0.05 | 0.10 |
|---|---|---|---|
| **0.10** | **1.41** | 2.88 | 1.34 |
| **0.30** | 0.02 | 1.15 | 1.75 |
| **0.60** | 0.52 | 1.21 | 0.62 |

The contraction grows monotonically as the κ-band tightens (`w` ↓) and the measurement sharpens (`σ_rel` ↓),
as expected. The near-zero and slightly negative entries in the widest-band row are Monte-Carlo noise around
"no contraction": a maximally broad κ prior carries essentially no information about the product κ·ω_s⁰, so
adding the term does nothing and the finite-sample sd estimate fluctuates by a few percent either way. That
noise floor (≈±3 pp for the weak chain, ≈±1–2 pp for the Kamimura chain) is far below the 43 % signal and
far below the 15 % threshold, so neither verdict is a knife-edge.

**κ-scale-invariance check** (best cell, κ_mid = 1.0 vs 0.5):

| chain | contraction (κ_mid=1.0) | contraction (κ_mid=0.5) | \|Δ\| |
|---|---|---|---|
| weak-prior | 42.95 % | 42.94 % | 0.02 pp |
| Kamimura | 1.41 % | 1.28 % | 0.13 pp |

Identical to Monte-Carlo noise — confirming the contraction is set by the *relative* band width and
precision, not by κ's central value.

**Sample-count robustness** (best cell, seed 0, samples 4000 vs 8000): weak-prior 42.95 % → 43.99 %
(|Δ| = 1.04 pp); Kamimura 1.41 % → 0.32 % (|Δ| = 1.09 pp). Both verdicts are stable to doubling the sample
count. (The fixed seed 0 is used throughout: both chains converge cleanly at seed 0, whereas an exploratory
off-seed sweep found the Kamimura *baseline* chain can occasionally stick at the R = 0.10 prior boundary — a
convergence pathology of the base calibration, unrelated to the added term — which is exactly why the seed
is pinned.)

## Honesty constraints (binding)

1. **κ is computed by the SAME transport that determines R (structurally anti-correlated) — every
   contraction claim is conditional-on-κ-band and says so.** The X-ray branching and the reactivation
   coefficient are not independent measurements of independent quantities; a first-principles transport
   that raises R also changes κ. The contraction figures above are the *statistical* leverage of an
   idealized independent ratio measurement given an *assumed* κ band, and must always be quoted as
   conditional on that band.
2. **κ is density-dependent — no diamond-anvil-cell (DAC) extension pre-Phase-3.** The κ band drawn from
   the 1980s literature is a low-density / clean-target figure; it may not transfer to the high-density DAC
   regime, and no extension to DAC conditions is claimed before the Phase-3 transport exists.
3. **TES precision assumptions must reference D-T-environment performance (tritium beta background under an
   8.2 keV line), not clean-target resolution.** The `σ_rel` values above are *illustrative* measurement
   precisions; any real feasibility claim for a transition-edge-sensor X-ray measurement must use the
   achievable precision on the 8.2 keV Kα line *in a D-T cell with its tritium beta continuum*, not a
   clean-target energy resolution.

## Dual use — the Phase-3 acceptance test

This scan defines a **Phase-3 acceptance test**: the Phase-3 reactivation transport must predict κ and R
*consistently* (the same transport produces both), so the long-standing (~37-year) Rafelski X-ray/sticking
tension becomes a **pre-committed, reportable outcome** rather than a post-hoc narrative — Phase 3 either
reconciles the measured X-ray branching with its own R, or it reports the residual tension with a quantified
reason. It also **rescues the (ω_s⁰, R) estimand**: the split that neutron-yield data alone cannot resolve
becomes identifiable given an independent X-ray/neutron constraint of the precision studied here.
