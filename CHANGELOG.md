# Changelog

All notable changes to OpenMuCF are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- Author: Bryan Nasr (ORCID: 0009-0008-2360-7522). -->
<!-- Repository: https://github.com/bryannasr4-gif/openmucf | Zenodo concept DOI 10.5281/zenodo.21251511 (v1.0.0 version DOI 10.5281/zenodo.21251512; v1.1.0 version DOI 10.5281/zenodo.21316574). -->

## [Unreleased]

### Added — D1 nuclear-capture dataset in `parity` mode (2026-08-14)
The first `G4MuonicData` dataset carrying real content. `data/g4/d1/` reproduces the muon-capture
data compiled into Geant4 v11.4.2 — a 90-record `{Z, A, rate, error}` table and a 101-value
effective-charge table — bit-for-bit, together with the Goulard–Primakoff analytic fallback,
declared as data in a `#FALLBACK` directive carrying all eight of the constants it needs.
- **Generated, never transcribed.** The pinned upstream source is vendored at
  `third_party/geant4/v11.4.2/`, unmodified, and pinned by its upstream **git blob id** — upstream's
  own object name for those bytes, verifiable against `github.com/Geant4/geant4` with no Geant4
  checkout and no `git` binary. Both layers are parsed out of it at build time by
  `openmucf/g4/sources/`, and no record count appears as a literal anywhere in the chain.
- **Parity checked exhaustively.** A Geant4-linked driver harvested 36000 `(Z, A)` points from the
  built library; a pure-Python evaluation preserving the C++ association order reproduced **every
  one bit-for-bit, maximum 0 ulp**. The committed oracle makes that a CI check on every platform
  with **no Geant4 present**.
- **Five upstream findings registered and disclosed, not fixed** (`DATASET_D1.md`): negative capture
  rates on 6325 of 36000 fallback points including ³H; non-finite returns at degenerate inputs with
  no coded rejection; a fallback that moves by up to **2980 ulp** between two conforming compiler
  configurations of the same source, which is why the declared model contract forbids floating-point
  contraction; an attribution that does not reconcile with the table's 74 distinct Z; and a
  non-monotonic step in the effective-charge table near the Z=82 shell closure.
- **The whole harvest chain is committed**, not only its C++ half: `cpp/tools/build_oracle.py` is
  the script that turns a driver's raw `%a` output into the committed oracle, and it reproduces that
  file byte for byte when run on the build named in the oracle's own header. A committed harvested
  artifact whose producing code is missing is the reproducibility hole vendoring the upstream source
  exists to close.
- Layer-2 row keys are now defined for tables whose primary key is a **single column** (a previously
  registered undefined case, whose first consumer is the effective-charge table).
- `FORMAT_SPEC.md` §2.2's example header no longer violates its own advisory that every `#UNITS`
  name be a `#COLUMNS` name, and now states that a `#FALLBACK` model's documentation must pin the
  formula **and its evaluation order**.
- `third_party/geant4/` is the first third-party licence in this repository: Geant4 Software License
  v1.0, applying to that directory only.

### Fixed — cross-architecture reproducibility (2026-08-09)
An independent reproduction of the full audit battery on Apple Silicon (arm64), against an x86-64
reference, found three defects that a single-architecture CI could not have surfaced. All three are fixed
here, and the gap that hid them is closed with a standing arm64 CI job.
- **Silent float32 sampling under a shadowed import (correctness, all platforms).** `jax_enable_x64=True`
  was set only in `openmucf/__init__.py`. Running any script from a directory that also contains a *clone*
  of this repository binds `openmucf` to a namespace package, so `__init__.py` never executes while the
  submodules still import — every NUTS chain then ran in float32, silently, on rates spanning ~7 decades.
  x64 is now enabled by `openmucf/_jaxcfg.py`, imported on every path into the package, and
  `require_x64()` hard-fails at each sampler entry. Pinned by `tests/test_x64_guard.py`, which reproduces
  the shadow and asserts the samplers still run in float64.
- **`DESIGN.md`'s `--audit` tolerance was smaller than the estimator's own Monte-Carlo error.** The
  sd-contraction cells were checked against a fixed 3 pp absolute band, justified by a "±1–3 pp
  Monte-Carlo floor" quoted from `docs/xray_feasibility.md` that had never been measured for these cells;
  the true per-refit spread is 4–18 pp. Such a band can only be met by regenerating the identical
  pseudo-random realization, so it passed on every x86-64 host and failed on 5 of 12 cells the first time
  another architecture ran it. Now: `n_synth` 8 → 64, every cell published as `value ± bootstrap MC
  standard error`, the audit band **derived** as 4σ of those SEs (floored at 0.01 — 4σ, not 3σ, because
  a gate that runs on every push across 12 cells must not cry wolf), and the *structural*
  claims the document makes — C4's lead, the ω_s^eff ranking, class-independence — gated at their own
  separations. The class-contrast prose is now generated from the measured paired difference and its SE,
  so the file cannot again assert a separation the data do not support.
- **The published ± omitted the base chain's own error, and the fix above would have failed cross-platform
  without it.** Every contraction is `1 − s_j/S` with one shared denominator `S = sd(base posterior)`, so a
  bootstrap over the synthetic datasets is blind to `S`'s Monte-Carlo error — the one term that actually
  moves when the run is reproduced on another machine. Measured by batch means on the pinned chain, that
  error is ~1.6% of `sd(ω_s^eff)` and ~1.7% of `sd(R)`, which alone shifts `ose_C1` by 0.011 against a band
  of 0.013. Every cell now publishes the two components combined in quadrature. It cancels to first order
  in the paired class contrast, which is unaffected. Also: the fresh run's SE — half of every band, and
  published nowhere — is now gated against the committed one, and any cell whose band exceeds its own value
  is reported as NOT INFORMATIVE instead of counting as a silent pass.
- **The FC-001 reproduction tests contradicted FC-001's own determinism statement.** `FORECAST_PROTOCOL.md`
  §7 claims bit-identity *under the recorded environment (including platform)* and Monte-Carlo agreement
  cross-platform; two tests asserted bit-identity unconditionally, so a fresh clone failed on any non-x86-64
  host. The strict gate now runs only on the platform the card itself records, and a new always-on gate
  checks every registered number (98 cells) against the pre-registered 2 % Monte-Carlo band and the card's
  structure exactly.
- **Instrumentation.** `generate_calibration.py --audit` now prints the per-class worst margin (as a
  percentage of band) whether it passes or fails, so every CI run on every platform is evidence about how
  the bands are actually sized — previously an "OK" on one architecture proved determinism, not that a
  tolerance was correctly sized. `generate_design.py --audit` prints its worst margin likewise.

### Fixed — the EIG audit band, measured rather than assumed (2026-08-12)
- **`DESIGN.md`'s EIG cells were audited against a 5 % *relative* constant, which is the wrong shape.**
  The sd-contraction cells got a measured, per-cell band on 2026-08-09; the EIG-in-bits cells kept the
  original pre-registered 5 % relative tolerance, annotated "held cross-arch 2026-07-23" — an annotation
  now **deleted**, because the artifacts of that reproduction were never tracked in this repository and
  the claim does not survive measurement. A 200-realization sweep over the base chain's seed (analysis
  seed held fixed, so the nested-Monte-Carlo draws are identical and only the posterior being integrated
  over changes) puts the per-cell realization noise at **0.042–0.068 bits = 1.40 %–6.10 % relative**: the
  noise is *absolute*, its relative size varying 4.4× across cells while its absolute size varies 1.6×.
  Against an independent realization the 5 % band therefore reds **49.5 % of runs**, worst on `eig_C3`,
  whose own noise (6.10 %) exceeds the entire band — and no relative constant fixes it, since the ≥ 34.5 %
  needed to cover `eig_C3` makes `eig_C4`'s band 24.8 σ of its own noise. The one arm64 EIG flag of
  2026-07-23 (|Δ| = 0.088 bits on `eig_C2`) sits at the 92nd percentile of that distribution, z = +1.48:
  an ordinary draw, reproduced on x86-64 by nothing more than a different chain seed.
  The EIG cells now use the same construction as the contraction cells, with the SE **measured in-run**:
  20 replicate base chains per pass (seeds 1–20, the committed seed excluded from its own error bar),
  `sd` with ddof = 1, published as `value ± se` in `DESIGN.md` and tracked in the manifest (31 → 37
  entries). Simulated false alarm **0.555 % per run**, inside the 0.12–0.60 %/run regime already chosen
  for the contraction cells at 4 σ; cost ≈ +20 s per pass. A fixed *absolute* band (0.383 bits) was
  measured and rejected — safe but ~10× more conservative than the project's own 4 σ design point, with
  7.7–18.6 % detection power at a 0.30-bit shift against 22.6–84.3 %, and stale the moment `n_outer`,
  `n_inner` or the chain length changes. **No published value moves and no finding changes**; the SE-ratio
  guard is now enforced only where a band is SE-governed, so the identically-zero `zero_eig` cell keeps
  the 0.01 absolute floor it already had.

### Fixed — the `min ess` audit cell, re-registered as a one-sided floor (2026-08-13)
- **`CALIBRATION.md`'s `min ess` cells were tolerance-audited against the committed realization, which
  is the wrong shape for a convergence diagnostic.** The symmetric 20 % band shared with the `mcse` cells
  reds when a *fresh* realization converges **better** than the committed one — a gate that punishes
  improvement is measuring the sampler's luck, not the artifact's correctness. Measured over a
  100-realization chain-seed sweep of both main chains, that band reds 3.0 % of the weak chain's
  realizations and 5.6 % of the Kamimura chain's converged, physical ones, and it held the worst margin
  of this file's whole battery in the 2026-07-23
  arm64 reproduction — 81.8 % of band, which was `Kamimura.min ess` at 9200 vs 11000: an ordinary draw of
  a diagnostic, not a defect in anything.
  `min ess` therefore leaves the tolerance-audited set. The committed value stays published as a
  *description* of that realization; what `--audit` now gates is the **fresh** run clearing a structural
  floor, `AUDIT_ESS_FLOOR = 2000`. Sizing: the 136 converged-and-physical realizations of that sweep span
  min ess 3885–11398 while the non-convergent mode sits at 2.0, so the floor separates the two regimes by
  three orders of magnitude rather than cutting through either; and because `mcse = sd/√ess`, a chain
  sitting at the floor carries an mcse inflated ×1.39 even against the lowest converged realization
  measured — outside the 20 % `mcse` band, which stays as it was. A run that clears the floor while
  genuinely mis-converged therefore still fails, on its own `mcse` cells. **No published value moves**;
  the `mean`/`sd`/`corr`/`r_hat`/`divergences` classes are untouched, and asking the audit for a
  committed-vs-fresh distance on a one-sided cell is now a hard error rather than a silent number.

### Added
- **`G4MuonicData`: an external-data format for muonic-atom physics, with its reference implementation
  (`FORMAT_SPEC.md` + `openmucf/g4/`).** A dependency-free way to ship muonic-atom data to Geant4 — or to
  any transport code — **with its provenance and its uncertainty attached**, which no dataset of this kind
  currently carries. Two layers, shipped together: Layer 1 (`*.g4dat`) is plain US-ASCII directives and
  numeric records that a C++ reader parses with nothing but its standard library; Layer 2 (`*.prov.json`)
  holds the bibliographic source, uncertainty type, competing-evaluation identity and disclosure flags that
  Layer 1 deliberately leaves out. The two are bound by an invariant: Layer 1 is generated from Layer 2 and
  its `#SOURCEDIGEST` is the SHA-256 of the Layer-2 file's bytes.
  - `FORMAT_SPEC.md` is normative and states the grammar, the Layer-2 schema, **sixteen exact error codes**
    (each carrying a 1-based line number), the reporting order two implementations must agree on, the
    archive format, and the rules a C and C++ reader has to follow — above all *not* `strtod`/`atof`, which
    honour `LC_NUMERIC` and silently return `0` for every value under a comma-decimal locale.
  - `openmucf/g4/{spec,provenance,emit}.py` is the reference implementation: parse/render/validate,
    `%.17g` floats that round-trip every finite double exactly, the Layer-2 schema and digest, and a
    deterministic `.tar.gz` plus the `geant4_add_dataset(...)` registration snippet. Standard library only,
    and fenced by test from importing this project's kinetics modules so the layer stays liftable.
  - `data/g4/` ships a worked example that carries **no physics and says so in the file itself** (every
    Layer-2 row reads "format example, not evaluated physics"), regenerated and byte-diffed by `make
    g4data` inside `make audit`. Float formatting is proven identical on Linux, macOS/arm64 and
    Windows on every CI run. The comma-decimal-locale check is *enforced* on Linux, where CI installs
    `de_DE.UTF-8` and verifies the install, so the test fails rather than skips; on the other two it
    runs only where a comma-decimal locale happens to be present.
  - Names are **provisional**: `G4MuonicData` and `G4MUONICDATA` are placeholders, and the C++ reader and
    its standalone validation application are specified but not yet part of this release.
- **Open muon-cost ledger (`openmucf/data/muon_cost.csv` + `openmucf/mucost.py` + `MUON_COST.md`).** A
  curated compilation-with-provenance of the muon-production energy cost on one auditable basis (beam GeV
  per muon; wall-plug and recapture credits kept in separate flagged columns, never folded). Ten rows across
  three tiers — design studies (anchor: Kelly–Hart–Rose 4.70 GeV/μ, open access; corroborated by
  full-text-verified Bertin 1987 and Eliezer–Henis 1994), demonstrated technology, and operating facilities
  (mu2e/COMET/MuSIC/HIMB — original derivations with the arithmetic shown). The 10³ simulation-to-facility
  gap is proved from the table itself and drawn in `figures/muon_cost_gap.png`.
- **`FINDINGS.md` §2b — Q_net by muon-cost tier.** The forward-UQ Q_net is re-run under T1/T2/T3 E_μ priors
  (via `uq.qnet_tier_panel`), holding every measured input fixed; the median Q_net collapses ~10⁵× from
  design-study to facility muons — the 10³ gap in energy-return form. The default flat [2, 10] GeV E_μ box
  in §1/§2 is unchanged (the tier panel is an added section, not a replacement).
- `MUON_COST.md` + `MUON_COST_MANIFEST.json` join `make audit` (regenerated + byte-diffed; the PNG is not
  byte-diffed); the provenance manifest check now covers the muon-cost manifest too.
- **The Q Rosetta stone + energy-balance graph (`openmucf/systems.py` + `SYSTEMS.md`).** `SystemChain` is a
  strict superset of the frozen `energy.EnergyChain` — a differentiable `jax.numpy` graph exposing every
  node (wall-plug → muon → fusion(+breeding) → blanket → thermal → electric → recirculation) as a named
  knob, plus two explicit, flagged, default-off factors (a tritium-breeding energy credit and a
  recirculating-power fraction). At the defaults it reproduces the v1 chain to machine precision (the
  G-legacy anchor: scientific breakeven 284.09, net-electrical 2367.42). `rosetta_table` + the `QBasis`
  registry convert v1's scientific/net-electrical gains, Kelly–Hart–Rose's electrical gain, and an
  efficiency-free gain onto one comparable reference basis.
- **η_acc self-correction finding (`SYSTEMS.md`).** Our v1 default η_acc = 0.30 was optimistic; Kelly's
  PSI-measured 0.18 moves the net-electrical breakeven ~2367 → ~3946 fusions/muon (linear in η_acc). The v1
  code default is unchanged this release; the finding carries the correction. The G-Kelly cross-basis check
  reproduces Kelly's electrical-gain chain (Eq. 2 + Table 1) at 15.7% (a documented result vs their 14%
  figure-3-curve headline, not tuned).
- `SYSTEMS.md` + `SYSTEMS_MANIFEST.json` join `make audit` (both regenerated + byte-diffed; closed-form
  algebra, cross-arch stable); the provenance manifest check now covers the systems manifest too.
- **Neutrons-per-joule league table (`NEUTRONOMICS.md` + `scripts/generate_neutronomics.py`).** Places μCF
  as a 14 MeV neutron source against the established incumbents on one basis: neutrons per joule of primary
  beam energy. μCF appears as **three tier-separated rows** — one per muon-cost tier (`MUON_COST.md`),
  never a single blended row — computed as X_μ / (E_μ,tier in J) with the **measured** record yield
  X_μ = 113 (`calibrate.OBS['xmu_obs']` / ledger target `V_petitjean_Xmu`, not the forward-UQ median). At
  the design-study muon cost μCF is competitive with a spallation source (~43 MeV of beam per neutron) and
  ~10³× better than a sealed-tube D-T generator; at the operating-facility muon cost the ~10³ muon-cost gap
  transfers one-for-one to the neutron economy. A short sourced table of alternative 14 MeV/n sources
  (Thermo P385 sealed tube, FNG, RTNS-II, ISIS spallation) is included, each n/J derived from published
  beam parameters. Beam basis only (wall-plug kept separate, I5); neutron-source economics, not breakeven
  (I9); no new physics (I1). `NEUTRONOMICS.md` + `NEUTRONOMICS_MANIFEST.json` join `make audit`.
- **Inverse-design frontiers (`openmucf/frontier.py` + `FRONTIER.md`).** "What would have to be true"
  breakeven frontiers over the energy-balance graph: closed-form requirement curves (`r_required`,
  `lambda_c_required`, `frontier_lambda_c_R`) plus an `optimistix` Newton solver (`solve_inverse`) that
  agree to ~1e-14. Reports the R ≳ 0.77 reactivation a MuFusE-scale programme would need for scientific
  breakeven — bit-identical to the existing forward-UQ audit. Framed strictly as requirements, never
  verdicts (the scenario-verdict registry is deliberately NOT built). `FRONTIER.md` + `FRONTIER_MANIFEST.json`
  join `make audit`; `optimistix` is promoted to an explicit dependency.
- **X-ray/neutron-ratio degeneracy-breaker feasibility scan (`docs/xray_feasibility.md`).** An exploratory
  (not-audited) scan of whether adding an X-ray-per-fusion-neutron observable to the calibration would break
  the `ω_s0`/`R` degeneracy. Best-cell posterior sd(R) contraction 42.95% in the weak-prior chain (≥ a 15%
  feasibility threshold), with its Monte-Carlo noise documented (the "±3 pp" figure originally quoted here
  is scoped to that study's asymptotic setting — see the Fixed section above). Exploratory only; the κ-band
  likelihood term is specced, not built, pending acquisition of a measured κ.
- **²²⁵Ac reproduction notebook (`scripts/parisi_ac225.py` + `notebooks/parisi_ac225.ipynb`).** A forward,
  factor-by-factor reproduction of Parisi & Rutkowski's (arXiv:2511.20951) headline — ~20 mg/yr of ²²⁵Ac from
  a 10 g ²²⁶Ra feedstock at 10¹² muons/s — from their published factors, each cited to its locator; lands at
  20.5 mg/yr (+2.6% vs the 20 mg/yr headline, +0.2% vs their Table-I value), with P_fus ≈ 564 W and ~400× the
  2024 global supply as cross-checks. Explicitly a reproduction of an *external* result — their "viable before
  energy breakeven" framing is quoted as theirs, not an OpenMuCF claim. CI-tested.
- **Bayesian experimental-design ranking (`openmucf/design.py` + `DESIGN.md`).** Ranks which next μCF
  experiment would most sharpen the partly-degenerate `(ω_s^eff, R)` estimand, over the existing calibration
  posterior, by a primary preposterior sd-contraction metric and a secondary nested-Monte-Carlo EIG. The
  X-ray/neutron ratio is the decisive, structural-class-robust R-sharpener; R is reported class-conditionally
  (constant-R vs R(φ)-inflated) because neutron-only observables do not identify R without an assumed
  structural form, and the class **contrast** is reported against its own Monte-Carlo error — on the shipped
  run it is *not* resolved, and the document says so. An internal planning instrument, not a verdict. `DESIGN.md` +
  `DESIGN_MANIFEST.json` carry NUTS-derived numbers, so `make audit` tolerance-checks them
  (`generate_design.py --audit`: every cell, EIG and sd-contraction alike, at 4σ of its own published
  Monte-Carlo SE — both the fixed 3 pp contraction band and the 5 % relative EIG band this release
  originally shipped are superseded, see the Fixed sections above) rather than byte-diffing.

### Changed
- **Class-tiered, falsifiable validation scoreboard.** Every `VALIDATION.md` row now carries a claim
  tier (`self-consistency` / `reproduction (fed input)` / `anchor-consistency` /
  `shape (calibrated model)` / `independent`) in a new `class` column, and the three Yamashita rows count
  as one shape test. Three registered `independent`-tier prediction targets now run and **FAIL by
  design** — `V_petitjean_omega` (derived effective sticking ω_s0·(1−R_col) = 0.557% vs the 0.45% band)
  and `V_faifman_900K` / `V_faifman_lowT` (the placeholder formation model vs the ledger's own
  Faifman1989 rows, ~20×/~17× low) — each a pre-registered, quantified measure of the v1 placeholder's
  distance from the field's rates (`PRE_REGISTRATION.md` amendment). No input, tolerance, or observation
  was changed to make any row pass.
- **Public surface aligned with what the code delivers.** A README trust map ("what you may cite":
  GREEN / AMBER / RED), a reworded status badge and value-prop stating that the Phase-3 surrogate is
  planned (today ω_s0 and R are ledger scalars), and `formation.py` truth-labels for every unsourced
  placeholder resonance plus a RED-tier runtime warning off its 300 K anchor (φ > 1.45 or T < 100 K).
- **Interop thermal export renamed** to `export_lambda_form_eff_thermal` (an effective cycle-scale rate,
  not the bare Faifman λ_dtμ); the old `export_lambda_dtmu_thermal` name and the `lambda_dtmu`
  `geant4_callables` key remain as deprecated aliases (removed in v2.0.0).
- **Sourced temperature-shape comparator (Yamashita–Kino 2022 Fig. 3a, digitized).** A deterministic,
  matplotlib-free digitizer (`scripts/digitize_yamashita_fig3a.py`) extracts λ_c(T) at c_t=0.5
  (`openmucf/data/yamashita_kino_lc_T.csv`, 14 points, CC-BY-4.0). `V_yamashita_ratio` is **re-anchored**
  from the earlier ~1.45 under-read to the full-curve digitized ratio λ_c(800 K)/λ_c(300 K) = 2.358
  (band [2.09, 2.62]; solid-line 2.235); the ±30% tolerance is unchanged, so the strictly-harder target
  flips the engine ratio ~1.31 **PASS→FAIL** and the row is re-tiered `shape (calibrated model)` →
  `independent` (a registered finding). A new `V_yamashita_curve` checks the engine against the digitized
  curve at 200/400/600/800 K (±30% per point; the 800 K point is a registered expected-FAIL). The
  scoreboard is now **6 pass / 5 registered-FAIL findings / 1 deferred**; the three Yamashita rows count
  as one test. No model input, prior, or tolerance was changed (`formation.py` untouched); the FAIL is
  the reported result.

### Planned
- **Phase 3 — compute-trained effective-sticking/reactivation surrogate `ω_s^eff(φ,T,c_t)`.** The one dominant
  rate that every group currently hard-codes, so that the auditor *produces* it instead of importing a
  contested constant. This is the quantitative motivation surfaced by the v1 calibration finding: experiment
  pins `ω_s^eff` and `λ_c` but not the `ω_s0`/`R` split (corr +0.84). Requires HPC/multi-GPU (cross-section
  training set + slowing-down Monte Carlo); a gold-standard close-coupling/R-matrix benchmark is the gating
  acquisition.

## [1.1.0] - 2026-07-11

### Added
- **Machine-checkable provenance (`openmucf/provenance.py` + `FINDINGS_MANIFEST.json`).** Every headline
  number in `FINDINGS.md` carries a typed manifest entry (formatted value + anchoring regex + source
  type), generated by construction from the same values the document uses; `python -m openmucf.provenance
  --check FINDINGS_MANIFEST.json` fails if any value drifts from its doc.
- **Single-sourced physical constants (`openmucf/constants.py`).** `λ₀`, `E_f`, and the muon-cost default
  are read once from the rate ledger and re-exported to the engine modules, so no module forks a literal
  and a broken ledger fails fast at import (zero numeric change to any result).
- **Registered UQ-priors file (`openmucf/data/uq_priors.csv`).** The uncertainty-box priors are now
  machine-sourced from a registered-priors file via `uq.params_from_ledger()` (regression-locked to the
  frozen literals); the box values are unchanged.
- **Typed ledger columns + the liquid cycling-rate row.** `rates.csv` gains
  `distribution`/`dist_lo`/`dist_hi`/`recommendation`/`phase`/`target_molecule` (schema + loader
  validation) and a first-class `lambda_c_liquid` measured-cycling-rate row (closing the long-missing
  cycling-rate row); the `eta_dtmu` row now carries an asymmetric [1, 5] interval rather than a Gaussian ±4.
- **η structural bracket (`FINDINGS.md` §1c).** The epithermal enhancement η (`eta_dtmu`) is threaded
  through the cycle engine and reported as a structural bracket beside the credible interval (X_μ at η=1
  vs η=5), with provenance-manifest entries — deliberately not folded into the UQ box, since the measured
  λ_c band already contains η as it occurred at the anchors.
- **New validation target `V_yamashita_ratio`.** Executes the pre-registered ±30% λ_c(800 K)/λ_c(300 K)
  ratio clause (engine ratio ~1.31 vs ~1.45 digitized = PASS); the validation scoreboard is now
  **7 pass / 1 deferred / 0 fail**.
- **Two absorbing loss channels in the cycle ODE (`cycle.py`) + the accounting table (`docs/accounting.md`).**
  The ttμ side-branch and ³He scavenging are added as explicit, opt-in (`include_loss_channels=True`)
  absorbing channels; the engine default stays channels-OFF and reduces to the v1 network bit-for-bit
  (reduction gate, pure atol 1e-9). `docs/accounting.md` is the single one-channel-one-home table (I5)
  recording where each deferred channel lives today and its re-attribution rule. **Framing: loss
  RE-ATTRIBUTION under the constraint that anchor-condition totals still match the measured effective
  sticking — a joint refit, not "more physics moved the numbers."**
- **Three loss-channel ledger rows (`lambda_ttmu`, `omega_tt`, `lambda_dhe3`).** `lambda_dhe3` = 1.92e8 s⁻¹
  from a live open source (Fotev et al., *Search for muon catalyzed d³He fusion*, arXiv:2001.09927);
  `omega_tt` = 0.14 (corroborated ω_tt=13.9%); `lambda_ttmu` ships the documented I10 blocked fallback
  (0.0, `needs_verification`) pending the Matsuzaki/Bom tt-fusion tables (*Muon Catal. Fusion*).
- **Extended closed form (`analytic.fusions_per_muon_v2`).** Adds the ttμ competing-hazard term
  `ω_tt·λ_tt/λ_c` (derived in `MODEL_SPEC.md` §4.1, validated to <1% against the ODE); ³He scavenging is
  intentionally omitted from the closed form (dμ-pool hazard) and documented.
- **Channels-on scoreboard (`VALIDATION_CHANNELS.md`).** The trust gate re-run with channels ON, in the
  `make audit` regenerate+diff list. With the tt channel blocked and the anchors He-purged it reproduces
  the channels-OFF 7/1/0 scoreboard exactly; the channels-OFF `VALIDATION.md` remains the trust gate.
- **muCF-Bench case registry + `openmucf` CLI (`openmucf/bench.py`, `openmucf/cli.py`, `BENCHMARKS.md`).**
  One registry exposes both the pre-registered validation trust gate (the 8 result ids `openmucf.validate`
  emits) and self-contained JSON reproduction cases (`openmucf/data/benchmarks/*.json`, shipped as package
  data) through a single runner and the `openmucf reproduce <case-id>` / `reproduce --all` / `validate`
  console script. `validation_targets.csv` remains the single source of validation truth (the runner
  re-exposes engine results, it does not re-decide any verdict). Two reproduction cases ship: `kou-chen-2026`
  (a friendly reproduction of Kou–Chen's published 112.6/156.5 fusions-per-muon, PASS within ±10%) and
  `jones-1986` (registered PENDING as blocked-acquisition — the record operating point cannot be pinned from
  open sources, so no conditions are guessed). `BENCHMARKS.md` is regenerated and diffed by `make bench` /
  `make audit`.
- **Counts-level twin: neutron time-spectrum forward model + likelihood (`openmucf/twin.py`,
  `openmucf/likelihood.py`, `TWIN_AUDIT.md`).** A fuel-component neutron time-spectrum expectation from the
  v1 cycle (channels-OFF; reduces to the established engine), a Poisson sampler for raw histograms, the
  idealized two-exponential estimator experimenters fit, and a counts-level numpyro likelihood. `TWIN_AUDIT.md`
  (generated, in `make audit`) reports the closed-form disappearance gate (recovers λ_n to <1%), the
  estimator-bias sweep over `t_min × c_t` on synthetic v1 truth, and FC-001 card-interval fuel-component
  disappearance bands. Identifiability is stated honestly: a delta-pulse histogram constrains the muon
  disappearance rate λ_n; ω_s^eff and λ_c are separated only through the informative measured-λ_c prior. A
  200-replica interval-calibration test (`slow`-marked, deselected by default and in CI) checks the λ_n 95%
  credible interval is calibrated. Fenced v0 — no detector response, no real-data fit, no dataset-specific claim.
- **Structural sensitivity brackets (`scripts/generate_materiality.py`, `MATERIALITY.md`).** One-at-a-time
  absorbing-loss-channel toggles at four fixed operating points (OP-A anchor-adjacent/non-headline, plus
  high-T / MuFusE-mid / MuFusE-peak), reported as **one-sided brackets** `X_μ^with − X_μ^without` beside the
  §2 forward-UQ CI width for scale — never convolved into any likelihood or CI (side-by-side combination rule
  only). The ³He scavenging channel is live (`c_He ∈ {1e-4, 1e-3}`; brackets ≤ ~0.18 X_μ units, under ~0.6%
  of the parametric CI width); the ttμ side-branch is rendered **"blocked — pending acquisition of the
  Matsuzaki/Bom tt-fusion tables"** — the generator detects the ledger row's `blocked:` marker and never
  emits a misleading zero bracket. Deterministic (no MCMC), in the `make audit` regenerate+diff list with
  `MATERIALITY_MANIFEST.json`.

### Changed
- **Cross-model review hardening.** Fail-loud capture of the solver's default error norm, the
  breakeven R-requirement computed from the registered omega_s0 nominal (never transcribed), a
  parallel-make-safe audit dependency, and a negative-background guard in the twin estimator — zero
  numeric change to any shipped result.
- **Extended reproducibility audit (`make audit`).** Now also verifies the provenance manifest (across
  `FINDINGS_MANIFEST.json`, `TWIN_MANIFEST.json`, and `MATERIALITY_MANIFEST.json`), exact-diffs
  `FINDINGS_MANIFEST.json`, `VALIDATION_CHANNELS.md`, `TWIN_AUDIT.md`/`TWIN_MANIFEST.json`, and
  `MATERIALITY.md`/`MATERIALITY_MANIFEST.json`, and re-checks the `CALIBRATION.md` MCMC tables (now including
  the channels-on re-attribution section, currently blocked) within a documented tolerance.
- **`slow` pytest marker.** Long-running tests (the twin coverage run) are marked `slow` and deselected from
  the default `pytest` (and CI) via `addopts`; run them with `pytest -m slow`.
- **Formation quadrature grid.** `formation._EGRID` switched from linear to geometric spacing for low-T
  convergence (a grid doubling now moves λ_dtμ(30 K) by <0.5%, previously ~7%); `formation._CALIB` was
  re-anchored so the disclosed 300 K rates are preserved bit-exactly (no 300 K result moved; off-anchor
  temperatures shift slightly, the intended better-quadrature improvement).

## [1.0.0] - 2026-07-07

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
- **Forecast registry (`forecasts/`, `openmucf/forecast.py`, `FORECASTS.md`).** Pre-registered, hash-stamped
  probabilistic forecast cards as a pushforward of the calibrated posterior through the analytic map (no new
  physics), scored later by CRPS + interval coverage. First card **FC-001** — effective sticking `ω_s^eff` and
  cycling rate `λ_c` at high density (`φ ∈ {1.2, 2.0, 2.4}`) under a calibrated-model scenario A and an honest
  ignorance-bound scenario B — **registered at this tag** (Zenodo DOI 10.5281/zenodo.21251512). Adds 20 forecast
  tests (**63 total**).

[Unreleased]: https://github.com/bryannasr4-gif/openmucf/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/bryannasr4-gif/openmucf/releases/tag/v1.1.0
[1.0.0]: https://github.com/bryannasr4-gif/openmucf/releases/tag/v1.0.0
