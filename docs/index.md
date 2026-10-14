# OpenMuCF

**Open FAIR rate ledger + differentiable cycle / energy-balance auditor for
muon-catalyzed fusion (μCF), and external muonic-atom data for Geant4.**

> Status: **v1 spine complete and validated** (Phases 0–2). The high-density
> effective-sticking surrogate (Phase 3) is planned (not yet started; needs HPC + a
> gold-standard cross-section source). See `MODEL_SPEC.md` for the model
> formulation, `LITERATURE.md` for the sourced rate ledger, and `CHANGELOG.md` for release history.

<!-- Author: Bryan Nasr (ORCID: 0009-0008-2360-7522). -->
<!-- Repository: https://github.com/bryannasr4-gif/openmucf -->

## What OpenMuCF is

OpenMuCF is neutral, shared substrate for the μCF cycle. It is three things:

1. **A FAIR rate ledger** (`openmucf/data/`) — every rate carries per-row provenance,
   conditions, an uncertainty, an *established / contested* tag, and a validity
   range. This is the ENDF/IMAS-analog the field has lacked.
2. **A differentiable (JAX/diffrax) cycle-kinetics + net-electrical energy-balance
   engine** plus a global UQ / Sobol auditor that turns point-estimate breakeven
   claims into **error-barred, falsifiable** verdicts.
3. **A compute-trained effective-sticking / reactivation surrogate**
   `ω_s^eff(φ, T, c_t)` *(Phase 3, planned)* so the auditor *produces* the
   dominant rate instead of hard-coding it.

## Why it exists — the open / differentiable / UQ gap

μCF had a 2026 renaissance — J-PARC's direct ddμ\* resonance observation
(Toyama et al., *Sci. Adv.* 2026), Acceleron Fusion's high-density
diamond-anvil-cell program ([arXiv:2606.05333](https://arxiv.org/abs/2606.05333)),
and theory projecting fusions-per-muon > 500 and gain Q > 2
([arXiv:2605.26432](https://arxiv.org/abs/2605.26432)). Yet there was **no open,
reproducible, differentiable, uncertainty-bearing code** for the μCF cycle, and
every group re-codes the rate ODEs privately from 40-year-old tables. OpenMuCF
fills that gap: one auditable ledger, one differentiable engine, one honest UQ
verdict everyone can share, cite, and falsify.

## Honest positioning — infrastructure + a finding, not new physics

OpenMuCF introduces **no new fundamental μCF physics**. The cycle is textbook and
the reactivation transport is Stodden (1990) / Rafelski–Müller (1988/89). The
contribution is **open, reproducible, differentiable, UQ-bearing infrastructure**
plus honest findings that fall out of it — for example, that under the *prior*
uncertainty ranges the 2026 N_μ > 500 projection sits at probability zero and
requires two microscopic quantities to move far outside anything measured. It
**complements** GEANT4; it does not compete with it. See `CREDIBILITY_FIREWALL.md`
for what is deliberately excluded, and `ADOPTERS.md` for who it is for.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                 # the default run deselects the slow marker; run the slow tests, the twin coverage among them, with `pytest -m slow`
```

Reproduce the ledger, findings, and figures:

```bash
make validate      # reproduce the pre-registered targets (VALIDATION.md: 6 pass, 5 registered-FAIL findings, 1 deferred; class-tiered)
make findings      # sensitivity ranking + breakeven falsification -> FINDINGS.md
make calibration   # Bayesian calibration + identifiability -> CALIBRATION.md
```

## Documentation

| doc | what it covers |
|---|---|
| [getting-started.md](getting-started.md) | narrative walkthrough: load the ledger, compute X_μ ≈ 114, scientific breakeven ≈ 284 and net-electrical breakeven ≈ 2367, and reproduce a finding with `make findings` |
| [api-overview.md](api-overview.md) | one short section per `openmucf` module — key public functions/classes and what they return |
| [geant4-developers.md](geant4-developers.md) | for Geant4 developers: the muonic-atom data layer, how to try it, what has been checked, what building it showed about Geant4 itself, and taking it upstream |

See also the in-repo references: `README.md`, `MODEL_SPEC.md`, `LITERATURE.md`,
`PRE_REGISTRATION.md`, `FINDINGS.md`, `VALIDATION.md`, `CALIBRATION.md`,
`CREDIBILITY_FIREWALL.md`, `ADOPTERS.md`.

## License

Apache-2.0 for code; CC-BY-4.0 for the rate ledger under `openmucf/data/`, the data documents
generated from it and the files under `data/g4/` that this project computed, harvested or wrote, other
than those named below or in the License section of `README.md`; the Geant4 Software License for the
Geant4 patches under `cpp/patches/`, the vendored Geant4 files under `third_party/geant4/`, the `parity`
tables `data/g4/d1/d1_capture.g4dat` and `data/g4/d1/d1_zeff.g4dat` with their Layer-2 files
`data/g4/d1/d1_capture.prov.json` and `data/g4/d1/d1_zeff.prov.json`, and the natural isotopic
compositions and the decay rates read from Geant4 builds, `data/g4/d1/geant4_natural_isotopes.csv` and
`data/g4/d1/d1_decay_rate.oracle`; MuDirac's MIT License for `data/g4/d3/mudirac_inputs.csv`, which
repeats the radii MuDirac bundles, and for `data/g4/d3/LICENSE-MUDIRAC`, the text of that licence. The
License section of `README.md` names, by path, the muonic-atom files offered under no licence of this
repository.
