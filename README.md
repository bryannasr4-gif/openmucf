# OpenMuCF

**Open FAIR rate ledger + differentiable cycle/energy-balance auditor for muon-catalyzed fusion (μCF).**

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Tests](https://img.shields.io/badge/tests-274%2F275%20(Linux%20%2B%20Windows%20%2B%20macOS%20arm64)-brightgreen.svg)
![Status](https://img.shields.io/badge/status-v1%20open%20infrastructure%20%2B%20honest%20findings-blue.svg)
[![CI](https://github.com/bryannasr4-gif/openmucf/actions/workflows/ci.yml/badge.svg)](https://github.com/bryannasr4-gif/openmucf/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21251511.svg)](https://doi.org/10.5281/zenodo.21251511)

> Status: **v1 spine complete** (Phases 0–2): it reproduces the pre-registered reproduction/consistency
> targets (see the class column in `VALIDATION.md`); independent-prediction targets are registered and
> currently FAIL by design against the v1 placeholder formation model — the quantified motivation for the
> sourced-table upgrade and the Phase-3 effective-sticking surrogate (planned, not yet started; needs HPC
> + a gold-standard cross-section source). See `MODEL_SPEC.md` for the model formulation, `LITERATURE.md`
> for the sourced rate ledger, and `CHANGELOG.md` for release history.

μCF had a 2026 renaissance — J-PARC's direct ddμ\* resonance observation (Toyama et al., *Sci. Adv.* 2026),
Acceleron Fusion's high-density diamond-anvil-cell program ([arXiv:2606.05333](https://arxiv.org/abs/2606.05333)),
and theory projecting fusions-per-muon > 500 and gain Q > 2 ([arXiv:2605.26432](https://arxiv.org/abs/2605.26432)) —
yet, to our knowledge (systematic search of GitHub/Zenodo/PyPI/IAEA, 2026-07), there is **no maintained,
citable open code for the μCF cycle that is reproducible, differentiable, and uncertainty-bearing**: the
published kinetics models (Yamashita–Kino 2022, Kou–Chen 2026, Bystritsky 2007, Stodden 1990) carry no
accompanying code releases. (Adjacent open tools exist — Geant4's open muonic-atom classes, with "catalyzed
fusion physics" on its 2024 work plan, and standalone muon-target sims — but none cover cycle kinetics /
energy balance / UQ; OpenMuCF is that layer, complementary to Geant4 transport.) OpenMuCF is the neutral
shared substrate:

1. **FAIR rate ledger** (`openmucf/data/`) — every rate with per-row provenance, conditions, uncertainty, an
   established/contested tag, and a validity range (the v1 seed/schema of the ENDF/IMAS-analog ledger the
   field lacks: 13 curated scalar rates today; T/φ/F-dependent tables are the Phase-2 milestone).
2. **Differentiable (JAX/diffrax) cycle-kinetics + net-electrical energy-balance engine** + a global UQ/Sobol
   auditor that turns point-estimate breakeven claims into **error-barred, falsifiable** verdicts.
3. **(Phase 3, planned — not yet built)** a compute-trained effective-sticking/reactivation surrogate
   `ω_s^eff(φ,T,c_t)`, so that the auditor will *produce* the dominant rate instead of hard-coding it.
   Today ω_s0 and R are ledger scalars.

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                 # 278 tests: 275 in the default run (274 pass, 1 skipped-blocked), 3 slow deselected
pytest -m slow         # the ~9-min twin interval-calibration coverage run (200 seeded MCMC fits)
```
Verified platforms: Linux CI (py3.11/3.12/3.13), Windows x64 (py3.12), and **macOS arm64 / Apple Silicon
(py3.12)** — every job runs the full suite, and the arm64 job additionally runs `make audit`, so the
cross-architecture claim below is a standing CI gate rather than a one-off check. All 18 byte-diffed
audited artifacts regenerate **byte-identically on arm64 and x86-64** (independently reproduced on Apple
Silicon, 2026-07-23; `git diff --exit-code` clean). The NUTS/MCMC-derived docs (`CALIBRATION.md`,
`DESIGN.md`) and the FC-001 card are not bit-portable across architectures and are checked against
pre-registered tolerance bands instead; each audit prints its per-cell margins so the headroom is visible
in every CI log.
Windows note: enable long-path support (or use a short venv path) for the JAX install.
The twin coverage test is marked `slow` and deselected from the default run (and CI); run it with `pytest -m slow`.

## Quickstart
```python
from openmucf import load_rates, analytic, cycle
from openmucf.energy import EnergyChain

rates = load_rates()                                   # validated FAIR ledger
xmu = cycle.fusions_per_muon_from_conditions(rates, T=300, phi=1.2, c_t=0.5)
print(float(xmu))                                      # ~114 fusions per muon
# NOTE: ~114 is the ledger pushforward omega_s0*(1-R_col)=0.557% through the closed form -- the same
# quantity the V_kouchen_base reproduction target checks; it is NOT an independent prediction (trust map below).
print(EnergyChain().breakeven_xmu_sci())               # ~284 (scientific breakeven)
print(EnergyChain().breakeven_xmu_net())               # ~2367 (net-electrical breakeven)
```
Reproduce the findings and figures:
```bash
make validate      # reproduce the pre-registered targets (VALIDATION.md: 6 pass, 5 registered-FAIL findings, 1 deferred; class-tiered)
make findings      # sensitivity ranking + breakeven falsification -> FINDINGS.md
make calibration   # Bayesian calibration + identifiability -> CALIBRATION.md
make systems       # Q Rosetta stone + energy-balance graph -> SYSTEMS.md
make mucost        # open muon-cost ledger + the 10^3 gap figure -> MUON_COST.md
make frontier      # inverse-design "what would have to be true" frontiers -> FRONTIER.md
make neutronomics  # neutrons-per-joule league table -> NEUTRONOMICS.md
make design        # Bayesian experimental-design ranking -> DESIGN.md
make audit         # regenerate every deterministic doc + tolerance-check the MCMC docs; fail on drift
```

## Headline results (see `FINDINGS.md`, `MUON_COST.md`, `CALIBRATION.md`)
- **Sensitivity split:** yield X_μ is controlled by reactivation R (Sobol S_T=0.62), λ_c, ω_s0; net-electrical
  Q is controlled by muon cost and accelerator efficiency. Different levers for yield vs energy.
- **Breakeven audit:** at liquid-scale density (φ ≤ ~1.45), under measured, unpolarized ranges, P(X_μ>500)=0 —
  structural (outside the prior's support), not a Monte-Carlo estimate. Density scaling could supply the
  cycling-rate factor at DAC φ≈2.4, but even at infinite λ_c the projection needs reactivation R ≥ 0.77
  (R ≈ 0.94 at λ_c=3e8) vs the model-derived ~0.35. A falsifiable, quantified bet that rides on reactivation.
- **Identifiability:** experiment pins ω_s^eff (and only loosely bounds λ_c) but not the ω_s0/R split — the
  posterior concentrates on the curve ω_s0(1−R)=ω_s^eff, whose linear correlation is prior-conditional
  (corr ≈ +0.8; see the prior-sensitivity sweep in `CALIBRATION.md`) — the quantitative reason the Phase-3
  microscopic calculation is needed.
- **Muon-cost normalization (`MUON_COST.md`):** a curated, provenance-tagged compilation of the
  muon-production energy cost on one auditable basis. Design studies sit at a few GeV per muon (anchor:
  Kelly–Hart–Rose 4.70 GeV/μ, open-access, G4Beamline), while operating facilities are ~10³× worse
  (mu2e ~5×10³, COMET ~2.3×10³, MuSIC ~6×10³ GeV/μ — original derivations, arithmetic shown). Re-running
  Q_net under each cost tier (`FINDINGS.md` §2b) collapses the median Q_net ~10⁵× from design-study to
  facility muons — the 10³ simulation-to-facility gap in energy-return form. *The floor is unvalidated,
  not impossible.*
- **The Q Rosetta stone (`SYSTEMS.md`):** a differentiable energy-balance graph (`openmucf.systems`, a
  superset of the frozen `EnergyChain`) that places the several muCF "Q" conventions — scientific gain,
  net-electrical gain, Kelly–Hart–Rose's electrical gain, an efficiency-free gain — on one comparable
  basis, so a dimensionless "Q" is never quoted without its accounting. The self-first finding: our v1
  default η_acc = 0.30 was optimistic; Kelly's PSI-measured 0.18 moves the net-electrical breakeven
  ~2367 → ~3946 fusions/muon (linear in η_acc).

Structural, one-sided: the parametric intervals above sit on the v1 reduced network; the known deferred
channels bias X_μ DOWNWARD by up to ≈15% combined (ttμ side-cycle, un-pinned pending acquisition;
d-recapture, bracketed in `MATERIALITY.md`), so intervals are best read as upper-edge-faithful.

Headline findings run on the closed form with measured-band inputs; the differentiable ODE engine is the
structural workhorse for trajectories/twin/UQ cross-checks and is gated against an exact linear-algebra
oracle (`openmucf/exact.py`; tests), but no headline number depends on its multi-pool structure.

## What you may cite (trust map)

| tier | outputs | why |
|---|---|---|
| **GREEN — citable as-is** | muon-cost ledger + 10³-gap (`MUON_COST.md`), Q Rosetta stone (`SYSTEMS.md`), neutrons-per-joule table (`NEUTRONOMICS.md`), breakeven falsification & requirements form (`FINDINGS.md` §3: caps, R ≥ 0.77 algebra), sensitivity split with error bars, forecast-registry machinery (FC-001) | transparent accounting / algebra on measured bands + provenance-tagged compilations; no dependence on the v1 formation model |
| **AMBER — citable with the stated basis** | calibrated ω_s^eff and λ_c posterior (`CALIBRATION.md`; basis: two published summary statistics, stated error bars, prior-sensitivity table), X_μ at the 300 K liquid anchor | statistically sound but summary-statistic-based; cite WITH the basis caveat |
| **RED — illustrative only, do not cite** | λ_c(T) / X_μ(T) temperature shape, anything at φ > 1.45, the ω_s0/R split as separate values, all `formation.py` outputs off the 300 K anchor | placeholder resonance geometry (unsourced positions/widths), linear-in-φ construction, ω_s0/R degenerate (corr ≈ +0.8); the λ_c(T) shape runs −41% to −44% below the digitized Yamashita–Kino 2022 curve (sourced comparator `V_yamashita_ratio`/`_curve`, fails ±30%) — a runtime warning fires in the RED regime |

## Forecast registry
OpenMuCF keeps a registry of **pre-registered, hash-stamped probabilistic forecasts** in `forecasts/`
(index: [`FORECASTS.md`](FORECASTS.md); pre-registration + basis-conversion rules + scoring conventions:
[`forecasts/FORECAST_PROTOCOL.md`](forecasts/FORECAST_PROTOCOL.md)). Each card is a **pushforward of the
existing calibrated posterior through the analytic map** (no new physics), scored later by **CRPS + interval
coverage** once the experiment publishes. The first card, **FC-001**, forecasts effective sticking `ω_s^eff`
and cycling rate `λ_c` at high density (`φ ∈ {1.2, 2.0, 2.4}`) under a calibrated-model scenario and an honest
ignorance bound. The card's `payload_sha256` covers only the scientific payload (environment metadata is
excluded), so the hash is portable; regenerate with `make forecast`.
FC-001 is **registered** at `v1.0.0` — Zenodo DOI [10.5281/zenodo.21251512](https://doi.org/10.5281/zenodo.21251512), payload SHA-256 `19291472309b1fe57c968bffc96ba56c7113b0be068686cf75b19fc6a2f14f59`.

> **Scope & intended use.** OpenMuCF is a neutron-economics auditor, not a reactor design. The energy chain
> includes an optional hybrid-blanket multiplier `M` purely as a transparent accounting term; below breakeven,
> μCF's defensible near-term utility is as a neutron / medical-isotope source (e.g. Ac-225) — the framing this
> project uses — not a fissile-breeding (Pu-239) pathway, which OpenMuCF does not model or endorse.

## Repository map
| path | what |
|---|---|
| `openmucf/rates.py` | FAIR ledger loader (provenance-enforced, autodiff-ready) |
| `openmucf/analytic.py` | closed-form X_μ, breakeven |
| `openmucf/cycle.py` | differentiable diffrax cycle ODE network |
| `openmucf/formation.py` | resonance-averaged λ_dtμ(T,φ,F) |
| `openmucf/energy.py` | transparent scientific + net-electrical Q |
| `openmucf/uq.py` | Sobol / forward-UQ / breakeven falsification |
| `openmucf/systems.py` | differentiable energy-balance graph + Q Rosetta stone (a superset of `energy.py`) |
| `openmucf/mucost.py` | open muon-cost ledger loader (tiered, provenance-tagged) |
| `openmucf/frontier.py` | inverse-design breakeven frontiers (closed-form + `optimistix` solver) |
| `openmucf/design.py` | Bayesian experimental-design ranking (nested-MC EIG + sd-contraction) |
| `openmucf/calibrate.py` | numpyro Bayesian calibration |
| `openmucf/validate.py` | reproduce the pre-registered targets |
| `openmucf/forecast.py` | pre-registered forecast cards (posterior pushforward, hashing, CRPS/coverage scoring) |
| `openmucf/interop.py` | GEANT4 interop stub — export rates (CSV/JSON), ingest validation spectra |
| `openmucf/g4/`, `FORMAT_SPEC.md` | the `G4MuonicData` external-data format: `.g4dat` grammar, Layer-2 provenance schema, deterministic archive |
| `openmucf/g4/sources/`, `third_party/geant4/`, `cpp/tools/` | structural extractors for the vendored upstream source, the pinned source itself, and the harvest drivers |
| `data/g4/d1/`, `DATASET_D1.md` | the D1 nuclear-capture dataset in `parity` mode + its findings (see below) |
| `openmucf/data/` | `rates.csv`, `validation_targets.csv`, `references.bib`, schema |
| `forecasts/`, `FORECASTS.md` | pre-registered, hash-stamped forecast cards (FC-001) + protocol + registry table |
| `MUON_COST.md`, `SYSTEMS.md`, `FRONTIER.md`, `NEUTRONOMICS.md`, `DESIGN.md`, `docs/xray_feasibility.md` | auto-generated analysis docs: muon-cost ledger + 10³ gap, energy-balance/Rosetta, inverse-design frontiers, neutrons-per-joule league table, experiment-design ranking, X-ray-feasibility scan |
| `examples/`, `notebooks/` | runnable `quickstart.py` + `quickstart.ipynb` |
| `docs/` | getting-started + API overview |
| `MODEL_SPEC.md`, `LITERATURE.md`, `PRE_REGISTRATION.md` | the physics, numbers, and locked targets |
| `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff` | how to contribute, what changed, how to cite |

## `G4MuonicData` — muonic-atom data as an external, versioned dataset
Geant4 carries its muon-capture data **compiled in**: 90 `{Z, A, rate, error}` records and a
101-value effective-charge table, with a Goulard–Primakoff analytic fallback for everything else.
Changing any of it means recompiling the toolkit, the per-record uncertainties it already stores are
never used, and nothing tells a user which rows rest on an isotope-resolved measurement.

`FORMAT_SPEC.md` defines a two-layer external-data format for that seam — a `.g4dat` grammar a
transport code can read with nothing but its standard library, plus a `*.prov.json` layer carrying
bibliography, uncertainty type and evaluation identity, bound to it by a SHA-256 digest.
`data/g4/d1/` is the first dataset with real content: a **`parity` profile** that reproduces Geant4
v11.4.2's compiled-in values bit-for-bit, generated at build time from the vendored upstream source
rather than transcribed. Its parity was checked against a Geant4-linked binary over **36000 (Z, A)
points at zero ulp**, and the committed oracle makes that checkable in CI with no Geant4 installed.

Building it surfaced five defects in the upstream seam, all **registered and disclosed rather than
fixed** — a parity dataset that "corrected" its source would stop being one. They include a fallback
that returns **negative capture rates on 6325 of those 36000 points** (³H among them, where a
negative rate silently disables capture entirely), non-finite returns at degenerate inputs with no
coded rejection, and a fallback whose result moves by up to **2980 ulp** between two conforming
compiler configurations of the same source. See [`DATASET_D1.md`](DATASET_D1.md).

> This product includes software developed by Members of the Geant4 Collaboration
> ( http://cern.ch/geant4 ).

The dataset name `G4MuonicData` and the environment variable `G4MUONICDATA` are **provisional
placeholders**, pending discussion with the Geant4 collaboration.

## Honest positioning
OpenMuCF introduces **no new fundamental μCF physics** — the cycle is textbook and the reactivation transport
is Stodden (1990) / Rafelski–Müller (1988/89). Its contribution is **open, reproducible, differentiable, UQ-bearing
infrastructure** plus honest findings. It **complements** GEANT4; it does not compete with it. See
`CREDIBILITY_FIREWALL.md` for what is deliberately excluded, and `ADOPTERS.md` for who it is for.

## Contributing
See [`CONTRIBUTING.md`](CONTRIBUTING.md) — in particular how to add a rate to the ledger with full
provenance, and the credibility-firewall policy. Release history is in [`CHANGELOG.md`](CHANGELOG.md).

## How to cite
If you use OpenMuCF, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this
repository" button from it). Archived on Zenodo — cite the exact release **v1.1.0** via DOI
[10.5281/zenodo.21316574](https://doi.org/10.5281/zenodo.21316574), or the version-independent concept DOI
[10.5281/zenodo.21251511](https://doi.org/10.5281/zenodo.21251511) to always resolve to the latest version.

## License
- **Code** — the `openmucf/` package, `scripts/`, tests, and all software: **Apache-2.0** (see [`LICENSE`](LICENSE)).
- **Data** — the rate ledger `openmucf/data/*` and the generated data docs: **CC-BY-4.0** (see [`LICENSE-DATA`](LICENSE-DATA)), so the compiled, provenance-tagged rates can be reused and cited with attribution.
- **Third party** — `third_party/geant4/` holds one unmodified Geant4 source file, redistributed under the **Geant4 Software License v1.0** ([`third_party/geant4/LICENSE`](third_party/geant4/LICENSE)). Those terms apply to that directory only.
