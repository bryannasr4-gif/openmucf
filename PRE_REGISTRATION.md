# Pre-registration (locked before the engine is built — Phase 1.3)

Locking what counts as **validated** and **falsified** *before* writing the cycle engine, so results can't be
back-fit. Inputs are never tuned to hit a target; discrepancies are documented, not fudged.

## Validation targets (engine must reproduce within tolerance)
Formalized in `openmucf/data/validation_targets.csv`; backbone is `MODEL_SPEC.md` §7 (V1–V5).

| id | what | tolerance |
|---|---|---|
| V1 | analytic X_μ == ODE N_fus(∞), single-pool limit | < 1% (numerical) |
| V_petitjean_omega / _Xmu | ω_s^eff ≈ 0.45% and X_μ ≈ 113 at liquid density | ±0.05% band; X_μ ∈ [100,150] |
| V_kouchen_base / _best | fed Kou–Chen inputs, reproduce X_μ = 112.6 and 156.5 | ±10% |
| V_yamashita_lcT | λ_c(T) rises monotonically 20→800 K | shape monotone; ratio ±30% (graphical source) |
| V_nagamine_trend | solid D-T: cycling↑ & loss↓ as T↓ | qualitative monotone trend |
| V_faifman_peak | λ_dtμ resonance peak (8.7±2.1)×10⁹ s⁻¹ at 0.42 eV | ±25% |

> **Amendment (2026-06-30, disclosed):** the registered peak value above was a
> secondary-source transcription error; the primary source (Fujiwara et al., PRL 85, 1642 (2000))
> reports **(7.1±1.8)×10⁹ s⁻¹ at 0.423±0.037 eV**, which the ledger/validation now use. 7.1e9 lies
> inside the originally registered ±25% band of 8.7e9 (18.4% off), so the correction never converted
> a FAIL to a PASS. Note also (2026-07-01): the executed V_yamashita check currently tests X_μ(T)
> monotonicity at 200–800 K as a λ_c(T) proxy; the registered 20 K lower bound and ±30% ratio clause
> are pending (the computed λ_c(800)/λ_c(300) ≈ 1.31 vs ~1.45 digitized would pass), and
> V_faifman_peak is an anchor-consistency check (the peak amplitude is the inserted measured value).
> Executed (2026-07-08): the ±30% ratio clause now runs as target V_yamashita_ratio (engine ratio
> ~1.31 vs ~1.45 digitized = PASS); the 20 K lower bound remains pending (low-T formation is Phase-3
> condensed-phase scope).

> **Amendment (2026-07-12, disclosed — validation-surface hardening):** three pre-existing
> registered quantities are promoted to executed validation targets BEFORE the implementing code
> lands, with pre-framed expected outcomes. (1) `V_petitjean_omega` now runs as an independent
> PREDICTION of effective sticking from the ledger microphysics: ω_s0·(1−R_col) = 0.557% against
> the registered 0.45±0.05% band — expected **FAIL** (~+24%); the gap is the un-attributed
> side-channel share already described in `docs/accounting.md` (ttμ + recapture), i.e. a finding
> about the v1 effective parameters, not a regression. (2) `V_faifman_900K` and (3) `V_faifman_lowT`
> execute the ledger's own Faifman1989 rows (2.3e9 at 900 K Maxwellian; 2e10 at E=0.2 eV) against
> the v1 formation model at ±50% — expected **FAIL** by ~20×/~17×: the registered, quantified
> distance of the v1 placeholder formation model from the field's rates, standing motivation for
> the sourced-table upgrade and Phase 3. Additionally the validation scoreboard now carries a class
> column (self-consistency / reproduction / anchor-consistency / shape / independent) and counts
> the two Yamashita rows as one test. No input, tolerance, or observation was changed to make any
> row pass; the three new rows are expected to fail and their failure is the reported result.

## Falsification target (the headline)
Re-create, **inside the same transparent model**, the 2026 projections:
- Yin–Kou–Chen (2605.26432): N_μ "> 500", Q > 2 under four-dimensional synergy.
- Kou–Chen (2606.07077): N_fus,μ 112.6 → 156.5.

Then **propagate honest rate uncertainties** and report: does Q > 1 survive realistic error bars? *Which*
single-parameter assumptions carry each projection, and are they physically supported? Output the achievable
(X_μ, Q) posterior + a "what-would-have-to-be-true" table. Cite and differentiate from the source papers; do
**not** drift into "I re-coded their model."

## Pre-committed honest outcome
The high-density ω_s^eff(φ,T,c_t) forecast **may resolve to "currently unconstrained"** once cross-section
uncertainty above ~1.45 φ is propagated into the DAC regime. **That is a reported result, not a failure** — it
tells the field that open theory cannot yet adjudicate Acceleron's bet, and identifies which measurement buys
the most credibility per dollar. We commit to reporting it either way.

## Global-UQ / sensitivity plan
- **Local:** exact autodiff gradients ∂X_μ/∂θ, ∂Q/∂θ for every contested input θ at the operating point.
- **Global:** Sobol first/total indices (SALib) + PCE over the joint uncertainty ranges of the `contested`
  rows in `openmucf/data/rates.csv` (ω_s0, R_col, λ_dtμ scale, λ_dt, λ_10, E_μ, …). Rank the dominant 2–3.

## GEANT4 interop contract (complement, never compete)
- **Export:** differentiable surrogate rates — ω_s^eff(φ,T), λ_dtμ(E,φ,T,F) — as tables + a callable API a
  GEANT4 muonic-atom run can consume.
- **Ingest:** GEANT4 / experiment spectra (e.g. μ–He sticking X-rays, neutron-time spectra) as validation data.
- **Never:** re-implement the particle-transport GEANT4 already does. OpenMuCF is the rate/kinetics/UQ layer.

> **Note (2026-07-12):** the thermal formation export is named `lambda_form_eff` (an effective
> cycle-scale rate, 300 K-anchored placeholder — see `formation.py` / `MODEL_SPEC.md` §5), **not** the
> bare Faifman λ_dtμ; the old `export_lambda_dtmu_thermal` function name and the `lambda_dtmu` callable
> key remain as deprecated aliases (removed in v2.0.0). Once sourced formation tables land, the bare
> λ_dtμ export resumes under its true name.
