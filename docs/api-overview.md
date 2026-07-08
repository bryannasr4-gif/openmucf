# API overview

One short section per `openmucf` module, naming the key public
functions/classes and what they return. Grounded in the module docstrings and
signatures; nothing here is invented.

Note on imports: the package top-level (`openmucf/__init__.py`) re-exports a convenience
surface — the ledger types (`Rate`, `RatesTable`, `load_rates`, `omega_fraction`, `bibkeys`),
`EnergyChain`, the submodules `analytic`, `cycle`, `energy`, `formation`, `interop`, `uq`, and
`__version__` — so `from openmucf import cycle, EnergyChain` works directly. `calibrate`,
`validate`, and `forecast` are reached as submodules (e.g. `from openmucf import calibrate`).

## `openmucf.rates` — the FAIR rate-ledger loader (single source of truth)

Loads and validates `openmucf/data/rates.csv` (microscopic input constants) and
`openmucf/data/validation_targets.csv` (observations to reproduce), enforcing per-row
provenance.

- `load_rates(csv_path=..., schema_path=..., check_refs=True) -> RatesTable` —
  load + validate the ledger; raises `ValueError` listing every problem.
- `RatesTable` — validated, dict-like collection of `Rate` rows. Key methods:
  `value(sym) -> float`, `__getitem__(sym) -> Rate`, `symbols()`, `contested()`,
  `needs_verification()`, `nominal_vector(symbols)` and
  `uncertainty_vector(symbols)` (jnp float64 vectors for autodiff/UQ), `get(sym)`.
- `Rate` — frozen dataclass for one ledger row (`symbol`, `description`, `value`,
  `unit`, `unc`, `unc_type`, `conditions`, `source_bibkey`, `source_locator`,
  `status`, `validity_range`, `single_source`, `needs_verification`, `notes`).
- `omega_fraction(rate_or_value) -> float` — convert a sticking value stored in
  percent to a bare fraction.
- `bibkeys(bib_path=...) -> set` — all citation keys defined in `references.bib`.

## `openmucf.constants` — cross-module physical constants (single source)

Re-exports, read once from the ledger at import, the three constants that multiple engine modules need
— so no module carries a forked literal, and a broken ledger fails fast at import. Reached as a
submodule (`from openmucf import constants`); not in the eager `__all__` surface.

- `LAMBDA_0` — muon decay rate `λ₀` [s⁻¹] (ledger row `lambda_mu_decay`).
- `E_F_MEV` — d-t fusion energy `E_f` [MeV] (ledger row `E_fusion`).
- `E_MU_GEV_DEFAULT` — muon-production cost default [GeV] (ledger row `E_mu_cost`).

## `openmucf.analytic` — closed-form steady-state yield and energy balance

Pure, JAX-differentiable backbone: `ω_s_eff = ω_s0·(1−R)`,
`X_μ = 1/(ω_s_eff + λ_0/λ_c)`, `Q = X_μ·E_f·η_conv/E_mu`.

- `effective_sticking(omega_s0, R)` — `ω_s0·(1−R)` (bare fractions).
- `cycling_rate(phi, lambda_c_tilde)` — `λ_c = φ·λ̃_c`.
- `fusions_per_muon(omega_s_eff, lambda_c, lambda_0=...)` — `X_μ`.
- `energy_gain(x_mu, eta_conv, E_f_MeV=..., E_mu_GeV=...)` — `Q`.
- `breakeven_xmu(E_f_MeV=..., E_mu_GeV=..., eta_conv=1.0)` — fusions-per-muon at
  Q = 1 (~284 for 5 GeV, η = 1).
- `from_ledger(rates, phi, lambda_c_tilde, use_legacy_sticking=False)` — `X_μ`
  using ledger values for `ω_s0`, `R_col`, `λ_0`.

## `openmucf.cycle` — differentiable cycle-kinetics ODE network (diffrax)

The v1 six-component network (states `x_dmu, x_tmu1, x_tmu0` and accumulators
`N_fus, stuck, dec`); Gate V1 is that it reproduces `analytic.fusions_per_muon`
to < 1% in the single-pool limit.

- `solve_cycle(lambda_0, lambda_dt, lambda_10, lambda_form1, lambda_form0,
  omega_s_eff, ...) -> diffrax solution` — integrate the cycle to `t1`.
- `fusions_per_muon_ode(lambda_0, lambda_dt, lambda_10, lambda_form1,
  lambda_form0, omega_s_eff, **kw) -> X_μ` — `X_μ = N_fus(t1)` from the full ODE.
- `params_from_conditions(rates, T, phi, c_t, omega_s_eff=None,
  use_legacy_sticking=False) -> dict` — assemble the cycle rate arguments from the
  ledger + physical conditions.
- `fusions_per_muon_from_conditions(rates, T, phi, c_t, **kw) -> X_μ` — one-call
  yield from (T, φ, c_t); the README quickstart entry point (≈ 114 at T=300, φ=1.2,
  c_t=0.5).
- `conservation_residual(sol) -> float` — invariant check (should be ~0).
- `STATE_LABELS` — the ODE state ordering.

## `openmucf.formation` — resonant dtμ (Vesman) formation rate

Data-anchored resonance-averaged v1 model, anchored to the measured 7.1e9 s⁻¹
beam resonance at 0.423 eV (Fujiwara 2000).

- `lambda_dtmu_energy(E, F=1) -> float [s^-1]` — energy-resolved resonant formation
  rate (sum of Vesman Gaussian resonances); autodiff-friendly.
- `lambda_dtmu(T, phi=1.0, F=0, eta=1.0) -> float [s^-1]` — thermally-averaged
  formation rate at temperature T, density φ, hyperfine F, with epithermal
  enhancement `eta`.

## `openmucf.energy` — transparent scientific + net-electrical energy balance

Keeps the scientific and net-electrical gains as separate, fully-knobbed layers.

- `EnergyChain` — frozen dataclass (`E_f_MeV=17.6`, `E_mu_GeV=5.0`,
  `eta_acc=0.30`, `eta_thermal=0.40`, `blanket_M=1.0`). Methods:
  - `Q_sci(x_mu)` — scientific gain (fusion energy out / muon-beam energy in).
  - `Q_net_electrical(x_mu)` — net-electrical gain through the full efficiency chain.
  - `breakeven_xmu_sci() -> float` — X_μ at Q_sci = 1 (~284).
  - `breakeven_xmu_net() -> float` — X_μ at Q_net = 1 (~2367).
  - `E_mu_MeV` — property, muon energy in MeV.

## `openmucf.uq` — UQ, global sensitivity, breakeven falsification

Runs on the closed-form map (validated < 1% vs the ODE); priors are uniform over
each input's contested range, defined in `PARAMS`.

- `Param` / `PARAMS` — dataclass and the list of six inputs (`omega_s0_pct`, `R`,
  `lambda_c`, `E_mu_GeV`, `eta_acc`, `eta_thermal`) with nominal/low/high ranges;
  `NAMES`, `NOMINAL`.
- `xmu(...)`, `q_sci(...)`, `q_net(...)` — vectorized numpy forward maps.
- `local_sensitivities() -> dict` — autodiff elasticities dln(Y)/dln(θ) for X_μ and
  Q_net at the nominal point.
- `cross_check_gradient(ose=0.005, lambda_c=1.30e8, tol=0.03) -> dict` — gradient of
  X_μ w.r.t. effective sticking, analytic vs autodiff-through-the-ODE (keys
  `grad_ode`, `grad_analytic`, `rel_diff`, `agree`).
- `sobol_indices(N=4096, output="X_mu") -> dict` — SALib first (`S1`) and total
  (`ST`) order Sobol indices for "X_mu" or "Q_net".
- `forward_uq(n=400_000, seed=0, blanket_M=1.0) -> dict` — Monte-Carlo credible
  intervals for X_mu, Q_sci, Q_net plus `P_Qsci_gt1`, `P_Qnet_gt1`.
- `breakeven_audit(n=400_000, seed=1) -> dict` — uncertainty-propagated verdict on
  the 2026 N_μ>500 / Q>2 claims plus the "what-would-have-to-be-true" required
  (R, λ_c).

## `openmucf.calibrate` — Bayesian calibration of cycle parameters (numpyro)

Calibrates (ω_s0, R, λ_c) to the measured effective sticking (0.45 ± 0.05 %) and
yield (113 ± 12); exposes the ω_s0/R identifiability degeneracy.

- `model(omega_s_eff_obs=0.45, omega_s_eff_sd=0.05, xmu_obs=113.0, xmu_sd=12.0,
  omega_s0_prior=("uniform", 0.60, 1.10))` — the numpyro model; `omega_s0_prior`
  can be `("uniform", lo, hi)` (weak; exposes the degeneracy) or `("normal", mu,
  sd)` (informative Kamimura prior).
- `run_mcmc(num_warmup=800, num_samples=2000, seed=0, omega_s0_prior=..., **obs)
  -> samples` — run NUTS and return posterior samples.
- `summarize(samples) -> dict` — per-parameter mean/sd/2.5%/97.5% for `omega_s0_pct`,
  `R`, `lambda_c`, `omega_s_eff_pct`, `X_mu`, plus `corr_omega_s0_R`.

## `openmucf.validate` — reproduce the pre-registered validation targets

The Phase 2.4 trust gate; honest by construction (unhittable v1 targets are marked
DEFERRED, not silently passed). Operating point T=300 K, φ=1.2, c_t=0.5.

- `Result` — dataclass per target (`target_id`, `observed`, `predicted`,
  `tolerance`, `passed` [None == DEFERRED], `note`).
- `run(rates) -> list[Result]` — evaluate the engine against every target
  (V_kouchen_base, V_kouchen_best, V_petitjean, V_yamashita_lcT, V_yamashita_ratio, V_breunlich_lambdac, V_faifman_peak).
- `report_markdown(results) -> str` — render the results table and pass/deferred/
  fail summary (the `VALIDATION.md` content: 7 pass / 1 deferred / 0 fail).

## `openmucf.forecast` — pre-registered, hash-stamped forecast cards (FC-001)

Builds probabilistic forecast cards as a pushforward of the existing calibrated posterior
(`openmucf.calibrate`, Kamimura chain) through the analytic map (`openmucf.analytic`) — no new
physics. Two scenarios (A = constant-R calibrated-model forecast; B = honest ignorance bound with
`R ∼ Uniform(0.15, 0.60)` and a structural `λ_c` bracket above `φ = 1.45`). See
`forecasts/FORECAST_PROTOCOL.md`.

- `posterior_samples() -> dict` — draw the calibrated posterior once (D6 chain: warmup 1000, samples
  4000, seed 0, Kamimura prior) and derive `lambda_tilde_c = lambda_c / 1.2`.
- `pushforward(samples, phi, scenario) -> dict` — typed predictions for one (φ, scenario):
  `{omega_s_eff, lambda_c}`, each an ensemble ({median, ci68, ci95, n_samples}) or a bracket
  ({median_range, envelope-union ci68/ci95, limbs}).
- `build_card(samples=None) -> dict` — the full card (`payload` hashed / `generation` env /
  `registration` mutable), with `payload_sha256` populated.
- `canonical_json(obj) -> str`, `payload_sha256(card) -> str`, `ledger_sha256(path=None) -> str`
  (LF-normalized) — deterministic serialization + hashing.
- `validate_card(card, schema_path=None)` — hand-rolled structural + fence checks (raises `ValueError`
  listing every problem; no new runtime dependency).
- `crps_empirical(samples, y) -> float` and `coverage(intervals, ys) -> dict` — scoring primitives;
  `score_card(card, resolved, samples=None) -> dict` — CRPS + interval coverage per target.
- `write_card(path=...)`, `regenerate(path=...)` (refreshes `payload_sha256`, preserves other
  registration fields), `render_forecasts_md(card_paths) -> str` (the `FORECASTS.md` registry table,
  read from on-disk cards only).

## `openmucf.provenance` — machine-checkable provenance for headline numbers

Records, for every headline number in a generated doc, a typed manifest entry (formatted value +
anchoring regex + source type), so a number cannot silently diverge from its recorded source. Paired
with `make audit`'s regeneration diff, this closes the doc-drift bug class. Reached as a submodule or
via `python -m openmucf.provenance --check FINDINGS_MANIFEST.json`.

- `ManifestEntry` — frozen dataclass for one tracked number (`id`, `value`, `pattern`, `source_type`
  ∈ {`derivation`, `ledger_row`, `registered_prior`}, `source`, `doc`).
- `file_sha256(path) -> str` — SHA-256 over the LF-normalized UTF-8 text of a file (same recipe as
  `forecast.ledger_sha256`, so digests are comparable).
- `write_manifest(path, entries, inputs) -> None` — write `{generated_by, inputs, entries}` JSON with
  sorted keys + 2-space indent for reviewable diffs.
- `check_manifest(manifest_path, repo_root=".") -> list[str]` — verify every entry's `value` is present
  (anchored by its `pattern`) in its doc; returns the list of failures (empty == all pass).
- `main(argv=None) -> int` — the `--check <manifest.json>` CLI (0 = pass, 1 = failure).

## `openmucf.interop` — GEANT4 / external-tool interoperability (v1 stub)

Honors the pre-registered interop contract: export the differentiable rates as tables + a callable
API, ingest external spectra as validation data, and never re-implement transport.

- `RateTable` — a named rate sampled on a 1-D/2-D grid; `to_csv(path)` (long format) and
  `to_json(path)` (axes + values + `openmucf_version`).
- `export_omega_s_eff(rates, T_grid, phi_grid, use_legacy_sticking=False) -> RateTable` — the
  ω_s^eff(φ,T) surface (flat in v1; the (φ,T,c_t) dependence is what Phase 3 fills in).
- `export_lambda_dtmu_thermal(T_grid, phi_grid, F=1, eta=1.0) -> RateTable` and
  `export_lambda_dtmu_energy(E_grid, F=1) -> RateTable` — the formation-rate surfaces.
- `export_all(rates, outdir, ...) -> dict` — write the standard CSV/JSON bundle; returns `{name: [paths]}`.
- `geant4_callables(rates, ...) -> dict` — plain-float callables `omega_s_eff(phi, T)` and
  `lambda_dtmu(E=None, phi, T, F)` for an in-process GEANT4 embedding.
- `ingest_spectrum(path, kind="neutron_tof", ...) -> Spectrum` — parse a two-column validation
  spectrum; `Spectrum.area()` / `.normalized()` for shape comparison.

## `openmucf` (package top level)

Enables JAX float64 (mandatory — rates span ~7 decades) and re-exports a convenience surface: the
ledger types `Rate`, `RatesTable`, `load_rates`, `omega_fraction`, `bibkeys`; `EnergyChain`; the
submodules `analytic`, `cycle`, `energy`, `formation`, `interop`, `uq`; and `__version__`.
(`calibrate`, `validate`, and `forecast` are reached as submodules.)
