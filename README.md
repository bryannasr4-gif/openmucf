# OpenMuCF

**Open FAIR rate ledger + differentiable cycle/energy-balance auditor for muon-catalyzed fusion (μCF).**

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Tests](https://img.shields.io/badge/tests-129%2F131%20(macOS%20%2B%20Windows)-brightgreen.svg)
![Status](https://img.shields.io/badge/status-v1%20research--grade-brightgreen.svg)
[![CI](https://github.com/bryannasr4-gif/openmucf/actions/workflows/ci.yml/badge.svg)](https://github.com/bryannasr4-gif/openmucf/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21251511.svg)](https://doi.org/10.5281/zenodo.21251511)

> Status: **v1 spine complete and validated** (Phases 0–2). The high-density effective-sticking surrogate
> (Phase 3) is the next capability — planned, not yet started (needs HPC + a gold-standard cross-section
> source). See `MODEL_SPEC.md` for the model formulation, `LITERATURE.md` for the sourced rate ledger, and `CHANGELOG.md` for release history.

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
3. **A compute-trained effective-sticking/reactivation surrogate** `ω_s^eff(φ,T,c_t)` *(Phase 3)* so the
   auditor *produces* the dominant rate instead of hard-coding it.

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                 # 131 tests (129 pass, 1 skipped-blocked, 1 slow deselected by default)
pytest -m slow         # the ~9-min twin interval-calibration coverage run (200 seeded MCMC fits)
```
Verified platforms: macOS arm64 (py3.13) and Windows x64 (py3.12) — 129/131 tests, `VALIDATION.md` regenerates
identically on both. Windows note: enable long-path support (or use a short venv path) for the JAX install.
The twin coverage test is marked `slow` and deselected from the default run (and CI); run it with `pytest -m slow`.

## Quickstart
```python
from openmucf import load_rates, analytic, cycle
from openmucf.energy import EnergyChain

rates = load_rates()                                   # validated FAIR ledger
xmu = cycle.fusions_per_muon_from_conditions(rates, T=300, phi=1.2, c_t=0.5)
print(float(xmu))                                      # ~114 fusions per muon
print(EnergyChain().breakeven_xmu_sci())               # ~284 (scientific breakeven)
print(EnergyChain().breakeven_xmu_net())               # ~2367 (net-electrical breakeven)
```
Reproduce the findings and figures:
```bash
make validate      # reproduce the literature (VALIDATION.md: 7 pass / 1 deferred / 0 fail)
make findings      # sensitivity ranking + breakeven falsification -> FINDINGS.md
make calibration   # Bayesian calibration + identifiability -> CALIBRATION.md
```

## Headline results (see `FINDINGS.md`, `MUON_COST.md`, `CALIBRATION.md`)
- **Sensitivity split:** yield X_μ is controlled by reactivation R (Sobol S_T=0.62), λ_c, ω_s0; net-electrical
  Q is controlled by muon cost and accelerator efficiency. Different levers for yield vs energy.
- **Breakeven audit:** at liquid-scale density (φ ≤ ~1.45), under measured, unpolarized ranges, P(X_μ>500)=0 —
  structural (outside the prior's support), not a Monte-Carlo estimate. Density scaling could supply the
  cycling-rate factor at DAC φ≈2.4, but even at infinite λ_c the projection needs reactivation R ≥ 0.77
  (R ≈ 0.94 at λ_c=3e8) vs the model-derived ~0.35. A falsifiable, quantified bet that rides on reactivation.
- **Identifiability:** experiment pins ω_s^eff (and only loosely bounds λ_c) but not the ω_s0/R split
  (corr +0.84) — the quantitative reason the Phase-3 microscopic calculation is needed.
- **Muon-cost normalization (`MUON_COST.md`):** a curated, provenance-tagged compilation of the
  muon-production energy cost on one auditable basis. Design studies sit at a few GeV per muon (anchor:
  Kelly–Hart–Rose 4.70 GeV/μ, open-access, G4Beamline), while operating facilities are ~10³× worse
  (mu2e ~5×10³, COMET ~2.3×10³, MuSIC ~6×10³ GeV/μ — original derivations, arithmetic shown). Re-running
  Q_net under each cost tier (`FINDINGS.md` §2b) collapses the median Q_net ~10⁵× from design-study to
  facility muons — the 10³ simulation-to-facility gap in energy-return form. *The floor is unvalidated,
  not impossible.*

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
| `openmucf/calibrate.py` | numpyro Bayesian calibration |
| `openmucf/validate.py` | reproduce the pre-registered targets |
| `openmucf/forecast.py` | pre-registered forecast cards (posterior pushforward, hashing, CRPS/coverage scoring) |
| `openmucf/interop.py` | GEANT4 interop stub — export rates (CSV/JSON), ingest validation spectra |
| `openmucf/data/` | `rates.csv`, `validation_targets.csv`, `references.bib`, schema |
| `forecasts/`, `FORECASTS.md` | pre-registered, hash-stamped forecast cards (FC-001) + protocol + registry table |
| `examples/`, `notebooks/` | runnable `quickstart.py` + `quickstart.ipynb` |
| `docs/` | getting-started + API overview |
| `MODEL_SPEC.md`, `LITERATURE.md`, `PRE_REGISTRATION.md` | the physics, numbers, and locked targets |
| `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff` | how to contribute, what changed, how to cite |

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
Apache-2.0.
