# FORECAST_PROTOCOL.md — pre-registration for the OpenMuCF forecast registry (FC-001)

This document locks **what is forecast, how it is scored, and how a published MuFusE result maps onto each
target** — *before* Acceleron publishes its effective-sticking / cycling-rate analysis. It is the
pre-registration home for `forecasts/FC-001-mufuse.json`. (The engine's earlier reproduction targets live in
`PRE_REGISTRATION.md`; this file governs the forward forecast only and never amends that one.)

FC-001 is a **pushforward of the existing calibrated posterior** (`openmucf.calibrate`, Kamimura-informative
chain) **through the existing analytic map** (`openmucf.analytic`) — it introduces **no new physics**. Every
card number is a posterior/ledger pushforward or a clearly-labelled registered prior. Reactivation transport
lineage: Stodden 1990 / Rafelski–Müller 1988–89.

---

## 0. The forecast in one paragraph

Once a muon-catalyzed-fusion target is compressed to liquid-hydrogen-density multiple `φ` and temperature `T`,
two quantities dominate the yield: the **effective α–μ sticking** `ω_s^eff` (a per-cycle loss, in %) and the
**cycling rate** `λ_c` (s⁻¹). OpenMuCF's v1 posterior — calibrated to the liquid-density Petitjean/Breunlich
data — pins `ω_s^eff` and the cycling rate at the liquid anchor. FC-001 propagates that posterior to the
higher densities the Acceleron/MuFusE diamond-anvil cell reaches (`φ ∈ {1.2, 2.0, 2.4}`) under two scenarios
(§3), and pre-commits the numbers, the hashes, and the scoring rules.

## 1. Targets and the conditions grid (D7)

Six scoring targets: `{ω_s^eff [percent], λ_c [s⁻¹]} × φ ∈ {1.2, 2.0, 2.4}`. Each target carries a stated
**temperature envelope** `T_K = [100, 150, 300]` labelled *"non-simultaneous envelope; forecast is conditional
on the (φ,T) Acceleron actually reports."*

**Why φ-only scoring, T as an envelope.** MuFusE's peak density and peak temperature are **not simultaneous**:
publicly, `933 MPa @ 100 K ≈ 2.4 φ` while `385 MPa @ 300 K ≈ 1.14 φ`. The v1 map has **no `ω_s^eff`
T-resolution** and only a weak `λ_c(T)` dependence (through resonant formation), so **producing T-resolved
predictions would be fabrication**. We therefore score on `φ` and state the `(φ,T)` the prediction is
conditional on. This is a deliberate, disclosed reduction of a fuller `φ×T` grid — not a hidden simplification.
**T caveat:** if Acceleron reports a quantity at a `(φ,T)` combination whose `φ` is in the grid, the target is
scored at that `φ`; the temperature is recorded but not used to reshape the prediction.

## 2. Basis-conversion rules (fixed BEFORE any MuFusE data exists)

Acceleron will publish **model-dependent fitted values in their own basis** — raw neutron-disappearance rates,
a normalized `λ_c`, and a **run-averaged `φ`** (their gasket-permeation discussion notes 2025 runs lost ~50%
of sample over 24 h at T > 150 K, so `φ(t)` drifts within a run). The following conversions are pre-fixed:

1. **`ω_s^eff`.** Map Acceleron's published/derived per-cycle effective sticking (in %) directly to the
   `ω_s_eff@φ` target at the run's run-averaged `φ`. If they publish only initial sticking `ω_s^0` and a
   reactivation `R`, form `ω_s^eff = ω_s^0·(1−R)` before comparison (the same map the engine uses).
2. **`λ_c`.** Map Acceleron's published cycling rate (their basis) to the `λ_c@φ` target at the run-averaged
   `φ`. If they publish a **density-normalized** rate `λ̃_c`, multiply by the stated run-averaged `φ` to obtain
   the actual rate before comparison (`λ_c = φ·λ̃_c`).
3. **`φ(t)` drift.** Use Acceleron's **run-averaged `φ` as stated**. Do not attempt to re-derive an
   instantaneous `φ` from pressure/temperature logs; the forecast is conditional on the reported average.
4. **Nearest-grid rule.** A reported `φ` within ±0.1 of a grid point is scored at that grid point; otherwise
   the target resolves *not scoreable* (rule 6).
5. **Model-basis mismatch.** If Acceleron reports a quantity that is not convertible to `ω_s^eff` or an actual
   `λ_c` under rules 1–3 (e.g. an unfactorable composite loss), that target is *not scoreable* (rule 6).
6. **The legitimate `"not scoreable under pre-fixed conversion rules"` outcome.** If no pre-fixed rule maps a
   published quantity to a target, the target resolves to **"not scoreable under pre-fixed conversion rules."**
   This is a **declared, honest resolution** — a model-vs-model basis mismatch is reported, never spun into a
   pass or a fail.

## 3. The two scenarios (exactly two; no hypothesis branches)

**Both scenarios are always computed, scored, and reported together.** Scenario A is the calibrated-model
forecast; Scenario B is the honest ignorance bound. **There is no post-hoc selection or reweighting** between
them (this is also a payload field, `scoring_rules.no_post_hoc_selection`). No stopping-power, Wallenius–Fröhlich,
or epithermal hypothesis branches enter FC-001 — those are out of scope by construction.

### 3.1 Scenario A — constant-R (calibrated-model forecast). Assumptions, verbatim:
- **"R held constant beyond its stated liquid validity range (`rates.csv`: 'liquid; phi<=1.45')."**
- **"λ_c assumed φ-linear beyond measured liquid support (φ ≈ 1.2)."**
- The joint posterior `(ω_s^0, R)` from the Kamimura-informative chain is held fixed; `ω_s^eff` is the
  posterior effective-sticking marginal (φ-independent under constant R), and `λ_c(φ) = φ·λ̃_c`.

### 3.2 Scenario B — honest ignorance (the "currently unconstrained" statement, quantified). Honesty block:
- **Registered prior, not a measurement.** `R` is replaced by a **registered prior (assumption)**
  `Uniform(0.15, 0.60)`. Motivation (all public sources): `[0.15, 0.60]` is a slightly floor-trimmed version
  of the calibration model's **own** `R` prior `Uniform(0.10, 0.60)` (`openmucf/calibrate.py`); its interior
  covers the Kou–Chen model value `R = 0.35 ± 0.05` (`rates.csv` `R_col`; arXiv:2606.07077 Eq. 33) and the
  documented liquid-density d-t reactivation spread ≈ 0.25–0.35 (`LITERATURE.md`; Breunlich 1989; Cohen;
  Markushin; Stodden 1990); its upper edge deliberately covers the **enhanced-reactivation direction** implied
  by the reported high-density sticking anomaly (measured `ω_s` at high density typically **10–50 % below**
  standard theory; arXiv:2606.05333). The Kamimura-prior posterior itself places **~25 % of its mass at
  `R > 0.5`**, so an interval truncated at 0.5 would be dishonest.
- **A sensitivity cut, not a coherent posterior.** Scenario B keeps the **posterior `ω_s^0` marginal** and
  replaces `R`. This deliberately **breaks the calibration's `ω_s^0`–`R` correlation** (+0.84 on the weak-prior
  chain, lower on the Kamimura chain), so B is a *sensitivity cut*, not a re-derived joint posterior.
- **Midpoint-of-ignorance artifact.** B's central value is a **midpoint-of-ignorance artifact**, not a central
  estimate. At `φ = 1.2` B's `ω_s^eff` center sits **above** Scenario A's, because the uniform-R midpoint
  (0.375) is **below** the posterior R mean (≈ 0.46) — expected, documented, **not a bug**.
- **What B does NOT cover.** B covers **nothing above `R = 0.60`**. If reactivation at high density exceeds
  0.60, the true `ω_s^eff` falls **outside both scenarios**; that outcome resolves outside FC-001 and is
  reported as such, not stretched to fit.

### 3.3 The λ_c structural bracket above the validity edge (D3)
`φ = 1.45` is the **model-validity edge** — the stated validity of the `R_col` reactivation input
(`rates.csv`: "liquid; phi<=1.45") — while the **measured liquid support ends at `φ ≈ 1.2`** (the D1 anchor).
(`φ = 1.45` is *not* a "measured edge", and it is unrelated to the numerically coincidental
`λ_c = 1.45×10⁸ s⁻¹`.) Beyond the validity edge the honest statement is a **bracket, not a distribution**:
- **ceiling** = full φ-linearity, `λ_c = φ·λ̃_c`;
- **floor** = saturation at the validity edge, `λ_c = 1.45·λ̃_c` for `φ > 1.45`.

Every prediction carries `prediction_type ∈ {ensemble, bracket}`. A **bracket** reports
`median_range = [floor_median, ceiling_median]` and `ci68`/`ci95` as **envelope unions**
(`[min of the two lower quantiles, max of the two upper quantiles]`) plus per-limb stats — and **no single
median and no headline CRPS**. At `φ = 1.2` (≤ the validity edge) the two limbs coincide, so Scenario B's
`λ_c` prediction there is an ordinary **ensemble identical to Scenario A's**. Only Scenario-B `λ_c` at
`φ ∈ {2.0, 2.4}` is a bracket; every other prediction (all `ω_s^eff`; all Scenario-A `λ_c`; Scenario-B `λ_c`
at `φ = 1.2`) is an ensemble. (`ω_s^eff` has no `φ` dependence in the map, so it is always an ensemble.)

## 4. The λ̃_c derivation, its cross-check, and a documented tension (D1)

The calibration model (`openmucf/calibrate.py`) samples `λ_c` as the **actual** cycling rate at the liquid
measurement conditions (it appears in the yield as `λ_0/λ_c`, with no separate `φ` factor). The repo's
**canonical liquid operating point is `φ = 1.2`** (`openmucf/validate.py` `_OP = dict(T=300.0, phi=1.2,
c_t=0.5)`, "canonical liquid"; `MODEL_SPEC.md` §7 gate V2; `validation_targets.csv` `V_petitjean_omega`
conditions "liquid D-T; phi~1.2"; `LITERATURE.md`). We therefore define the density-normalized cycling rate

    λ̃_c = λ_c / 1.2,

and propagate the actual rate at density `φ` as `λ_c(φ) = φ·λ̃_c` (`openmucf.analytic.cycling_rate`). The
Kamimura-chain posterior gives `λ_c ≈ 1.15×10⁸ s⁻¹`, hence **`λ̃_c ≈ 0.95×10⁸ s⁻¹`**.

**Cross-check (report-only; it never selects the anchor).** The maximum measured liquid cycling rate is
`λ_c ≤ 1.45×10⁸ s⁻¹` (Breunlich 1989; `validation_targets.csv` `V_breunlich_lambdac`, ±30 %). Normalized,
`1.45×10⁸ / 1.2 ≈ 1.21×10⁸ s⁻¹`; our `λ̃_c ≈ 0.95×10⁸` sits inside the ±30 % band `[0.85, 1.57]×10⁸` (our
posterior mean is expected to be *below* the measured *maximum*). ✔

**Documented tension (a finding, not something to "fix").** `λ̃_c ≈ 0.95×10⁸` sits marginally **below the
`1.0×10⁸` lower edge** of the Yamashita–Kino normalized cycling-rate band discussed at `LITERATURE.md`
(the "≈ 1.0–1.45×10⁸ s⁻¹" band). That reading already carries the repo's own **[VERIFY P1]** caveat: the
Yamashita–Kino band is quoted **density-normalized at φ = 0.4**, whereas the UQ layer conservatively treats
`1.45×10⁸` as the *actual* liquid maximum. The two conventions are not identical, and the ~5 % shortfall of
our posterior `λ̃_c` below the band's lower edge is exactly the kind of normalization ambiguity that
`[VERIFY P1]` flags. We **report** this tension rather than tune the anchor; the `φ = 1.2` anchor is fixed by
convention (§4, above) **before** any `λ̃_c` computation, and a failed cross-check would be a documented
finding, never a reason to move it.

## 5. Scoring (fixed BEFORE resolution)

**Estimator conventions (verbatim):** `ci68 = [16th, 84th]` and `ci95 = [2.5th, 97.5th]` equal-tailed
percentiles via `numpy.percentile` default ("linear") method; `median = numpy.median`. Predictions are rounded
to **4 significant figures** before serialization/hashing.

**CRPS (empirical-CDF / energy form), verbatim:**

    CRPS(F, y) = mean|X − y| − (1/(2 n²)) · Σ_{i,j} |x_i − x_j|

where `X = {x_1,…,x_n}` are the predictive draws and `y` is the resolved truth (implemented via the sorted
`Σ_{i,j}|x_i−x_j| = 2 Σ_i (2i−n−1) x_(i)` identity). A point-mass predictive at `c` gives `CRPS = |c − y|`.

- **Ensemble targets** → CRPS + interval coverage (`ci68`, `ci95`).
- **Bracket targets** → interval coverage (envelope-union `ci68`/`ci95`) + **per-limb CRPS reported as a
  `[best, worst]` pair**; **no headline CRPS** is defined for a bracket.

> **Amendment (2026-07-12, disclosed — Wave-3 research-grade hardening):** a **strictly-proper interval
> score** is added as the **headline bracket metric** (and is also reported for ensemble targets, so
> Scenario A and Scenario B are comparable on ONE proper score). The (negatively-oriented) **Winkler
> interval score** of a central `(1−α)` interval `[lo, hi]` vs truth `y` is
> `IS_α = (hi − lo) + (2/α)(lo − y)·[y<lo] + (2/α)(y − hi)·[y>hi]` (lower is better; rewards sharpness,
> penalizes a miss); it is reported at `interval_score_68` (α = 0.32) and `interval_score_95` (α = 0.05)
> on the envelope-union interval. Per-limb CRPS and coverage are retained as **diagnostics**. This applies
> to **FC-001's resolution-time scoring** (the scoring *code*, `openmucf.forecast.interval_score` /
> `score_card`) and to all future cards; **the registered FC-001 card file is unchanged on disk** — the
> `scoring_rules` field of NEW cards records the interval-score headline, but the frozen card is never
> regenerated (WAVE3 G-R4). Implemented `openmucf/forecast.py`; tests in `tests/test_forecast.py`.

**Both scenarios are always scored and reported together** — A = calibrated-model forecast, B = ignorance
bound — with **no post-hoc selection or reweighting**.

**Coverage caveat.** The six targets **share the same posterior draws** (and `ω_s^eff` is φ-independent, so it
repeats across the three φ), so coverage counts are **descriptive, not independent trials** — do not read them
as a calibration test with six degrees of freedom.

## 6. The excluded point (postdiction fence)

Acceleron has **publicly** presented a preliminary DT cycling datum near **2.2× liquid density at ~50 K**.
Because that number is **visible before this card is registered**, scoring it would be a **postdiction**, and a
single contaminated card discredits the whole registry. It is therefore **omitted from the scoring grid
entirely** (`payload.resolution_criteria.excluded_points`). This is a deliberate integrity fence, not an
oversight.

## 7. Hashing, determinism, and the registration-ready draft

The card has three sections:
- **`payload`** — everything scientific (targets, scenarios, predictions, scoring rules, provenance). It is
  **hashed and immutable**.
- **`generation`** — environment metadata (Python/JAX/jaxlib/numpyro/NumPy versions, platform, machine,
  `jax_enable_x64`). It is **not hashed**, precisely so the payload hash is **environment- and
  platform-portable**.
- **`registration`** — **not hashed, mutable**: `payload_sha256` (populated at build), and
  `registered_utc` / `code_version_tag` / `zenodo_doi` (all `null`), with `status = "draft"`.

**`payload_sha256`** = `SHA-256` over `canonical_json(payload)`, where
`canonical_json = json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=True)`. Because only the
payload is covered, **edits to `generation` or `registration` cannot change it** (a registration edit at tag
time leaves the scientific hash intact).

**`ledger_sha256`** = `SHA-256` over the **LF-normalized UTF-8 text** of `openmucf/data/rates.csv` (decode →
replace `\r\n` with `\n` → encode → hash). This normalization is pre-registered so a CRLF checkout on Windows
produces the **same** ledger hash as an LF checkout.

**Determinism statement.** The card regenerates **bit-identically under the recorded environment (including
platform)**; **cross-platform regeneration reproduces to Monte-Carlo error**. We claim no more than that.

**Registration is out of scope for this draft.** The card ships **registration-ready** in `draft` status:
`registered_utc`, `code_version_tag`, and `zenodo_doi` stay `null` until the first tagged release, when the
registration block is filled and `status` flips to `registered` — the immutable `payload_sha256` is what a
future reader checks the timestamp chain against.

## 8. What FC-001 claims and does not claim

- **Claims:** a pre-registered, hash-stamped, honestly-scored forward forecast of `ω_s^eff` and `λ_c` at
  stated high-density conditions, propagated from the existing calibrated posterior.
- **Does not claim:** any new microphysics; any resolution of the α-sticking discrepancy; any prediction at
  `(φ,T)` combinations the v1 map cannot resolve; any endorsement of a particular high-density reactivation
  value. Scenario B's whole point is that the high-density regime **may be currently unconstrained** — and if
  so, the forecast says so in quantitative form.
