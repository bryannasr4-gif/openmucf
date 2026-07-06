# Changelog

All notable changes to OpenMuCF are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- Author: Bryan Nasr (ORCID: 0009-0008-2360-7522). -->
<!-- Repository: https://github.com/bryannasr4-gif/openmucf (Zenodo DOI added at first release). -->

## [Unreleased]

### Added
- **Forecast registry (`forecasts/`, `FORECASTS.md`).** `openmucf/forecast.py` builds pre-registered,
  hash-stamped probabilistic forecast cards as a pushforward of the existing calibrated posterior through the
  analytic map (no new physics), with CRPS + interval-coverage scoring. First card: **FC-001** — effective
  sticking `ω_s^eff` and cycling rate `λ_c` at high density (`φ ∈ {1.2, 2.0, 2.4}`) under a calibrated-model
  scenario A and an honest ignorance-bound scenario B, with a `payload`/`generation`/`registration` card split
  (environment-portable `payload_sha256`) and a `forecasts/FORECAST_PROTOCOL.md` pre-registration (basis
  conversion, T caveat, exclusion fence, determinism). Regenerate with `make forecast`; the card ships in
  `draft` status (registration DOI/tag added at first release). Adds 20 tests (63 collected).

### Planned
- **Phase 3 — compute-trained effective-sticking/reactivation surrogate `ω_s^eff(φ,T,c_t)`.** The one dominant
  rate that every group currently hard-codes, so that the auditor *produces* it instead of importing a
  contested constant. This is the quantitative motivation surfaced by the v1 calibration finding: experiment
  pins `ω_s^eff` and `λ_c` but not the `ω_s0`/`R` split (corr +0.84). Requires HPC/multi-GPU (cross-section
  training set + slowing-down Monte Carlo); a gold-standard close-coupling/R-matrix benchmark is the gating
  acquisition.

## [1.0.0] - 2026-06-30

First public release: the minimum-useful, validated **v1 spine** — FAIR rate ledger → analytic closed form →
differentiable cycle ODE → net-electrical energy balance → global UQ auditor → Bayesian calibration, all
provenance-clean and reproducible.

### Added
- **FAIR rate ledger (`openmucf/data/`).** `rates.csv` with 13 input rates (9 contested, 4 established; each carrying per-row provenance,
  conditions, uncertainty, an established/contested tag, and a validity range), `validation_targets.csv`
  with 10 reproduction anchors, `references.bib`, and `rates.schema.json`. Loaded by `openmucf/rates.py`,
  which enforces schema validation and a provenance cross-check against `references.bib` and returns
  autodiff-friendly float64 rates.
- **`openmucf/analytic.py`** — the closed-form yield `X_μ = 1/(ω_s^eff + λ₀/(φ·λ̃_c))` with
  `ω_s^eff = ω_s0·(1−R)`, plus scientific and net-electrical breakeven. Reproduces the differentiable ODE to
  `rel.diff 0.000%` at the V1 gate.
- **`openmucf/cycle.py`** — the differentiable JAX/diffrax cycle-kinetics ODE network (6 components: 3
  dynamical states + 3 accumulators; Kvaerno5 stiff solver; fast-fusion/adiabatic elimination). Probability
  conserved to `<1e-4`.
- **`openmucf/formation.py`** — a physically-grounded resonance-averaged `λ_dtμ(T,φ,F)`: energy-resolved
  Vesman resonances (peak 7.1e9 s⁻¹ at 0.423 eV, Fujiwara 2000) with a Maxwellian average, thermal scale
  calibrated to the ~1e8 room-temperature anchor.
- **`openmucf/energy.py`** — a transparent scientific and **net-electrical** `Q` chain
  (`η_acc·η_thermal·M`), yielding the energy ladder: record ~150 | scientific breakeven ~284 |
  net-electrical breakeven ~2367.
- **`openmucf/uq.py`** — the uncertainty auditor: autodiff local elasticities, SALib global Sobol indices,
  Monte-Carlo forward UQ, breakeven falsification, and an ODE-vs-analytic gradient cross-check.
- **`openmucf/calibrate.py`** — numpyro (NUTS) Bayesian calibration and the `ω_s0`/`R` identifiability
  analysis.
- **`openmucf/validate.py`** — reproduces the pre-registered literature targets and auto-generates
  `VALIDATION.md` from real engine output.
- **`openmucf/interop.py`** — a GEANT4 / external-tool interop stub (complement, never compete): exports the
  differentiable rates ω_s^eff(φ,T) and λ_dtμ(E,φ,T,F) as CSV/JSON `RateTable`s, a `geant4_callables` API,
  and `ingest_spectrum` for validation data. Honors the pre-registered interop contract.
- **Auto-generated findings docs.**
  - `VALIDATION.md` — **6 pass / 1 deferred / 0 fail** against the pre-registered targets (Kou–Chen baseline
    112.6→114.5, Kou–Chen best 156.5→160.3, Petitjean ~113→130.5, Yamashita λ_c(T) monotone rise,
    Faifman epithermal peak), no input tuned to hit a target.
  - `FINDINGS.md` — sensitivity split (X_μ variance driven by reactivation R, Sobol S_T=0.62), forward-UQ
    credible intervals, and the density-scoped breakeven result `P(X_μ>500)=0` at liquid density (φ≤1.45, unpolarized) — reported as requirements (reaching 500 needs R≥0.77).
  - `CALIBRATION.md` — the `ω_s0`/`R` degeneracy (corr +0.84) that motivates Phase 3.
- **4 figures** — `figures/sobol.png`, `figures/forward_uq.png`, `figures/breakeven.png`, and the
  calibration figure — generated by `scripts/generate_findings.py`.
- **Test suite** — 43 tests across the ledger, analytic, cycle, energy, formation, UQ, calibration,
  validation, and interop modules.
- **Tooling & CI** — `ruff` (clean), GitHub Actions CI (`.github/workflows/ci.yml`), a `Makefile`
  (`make validate` / `make findings` / `make calibration`), a pinned `requirements-lock.txt` for
  reproducible installs, `pyproject.toml` (package `openmucf`, license Apache-2.0), and an expanded
  `README.md`.
- **Positioning docs** — `MODEL_SPEC.md`, `LITERATURE.md`, `PRE_REGISTRATION.md`, `CREDIBILITY_FIREWALL.md`,
  and `ADOPTERS.md`. OpenMuCF introduces **no new fundamental μCF physics**; the cycle is textbook and the
  reactivation transport is Stodden (1990) / Rafelski–Müller (1988/89). The contribution is open, reproducible,
  differentiable, UQ-bearing infrastructure plus honest findings.

[Unreleased]: https://github.com/bryannasr4-gif/openmucf/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/bryannasr4-gif/openmucf/releases/tag/v1.0.0
