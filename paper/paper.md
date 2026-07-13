---
title: 'OpenMuCF: An open FAIR rate ledger and differentiable cycle/energy-balance auditor for muon-catalyzed fusion'
tags:
  - Python
  - muon-catalyzed fusion
  - nuclear fusion
  - JAX
  - differentiable programming
  - uncertainty quantification
  - FAIR data
  - reproducible research
authors:
  - name: Bryan Nasr
    orcid: 0009-0008-2360-7522
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 13 July 2026
bibliography: paper.bib
---

# Summary

`OpenMuCF` is open, reproducible, differentiable, uncertainty-bearing infrastructure for the
muon-catalyzed-fusion ($\mu$CF) cycle. It combines three things the field has lacked in open,
citable form: (1) a **FAIR rate ledger** in which every microscopic rate carries per-row provenance,
physical conditions, an uncertainty and its type, an `established`/`contested` tag, and a validity
range; (2) a **differentiable (JAX/diffrax) cycle-kinetics and net-electrical energy-balance engine**
gated against an exact linear-algebra oracle; and (3) a **global (Sobol) forward-uncertainty auditor**
that turns point-estimate breakeven claims into error-barred, falsifiable verdicts, together with a
registry of pre-registered, hash-stamped probabilistic forecasts.

`OpenMuCF` introduces **no new fundamental $\mu$CF physics** — the cycle is textbook and the
reactivation transport follows @Stodden1990 and @Rafelski1989. Its contribution is the open shared
substrate plus a set of honest, quantified findings, and a disciplined separation of what is citable
from what is illustrative. It is designed to *complement* transport codes such as Geant4 rather than
compete with them, supplying the cycle-kinetics/energy-balance/uncertainty layer that those engines
do not provide.

# Statement of need

Muon-catalyzed fusion had a 2026 renaissance: J-PARC reported the first direct observation of a
muonic-molecule resonance state [@Toyama2026]; the Acceleron/MuFusE collaboration is pushing $\mu$CF
into diamond-anvil-cell densities and temperatures [@Kalow2026]; and theory now projects fusions per
muon above 500 with gain $Q>2$ under combined levers [@YinKouChen2026; @KouChen2026]. Yet, to our
knowledge (a systematic search of GitHub, Zenodo, PyPI, and IAEA resources in mid-2026), there is
**no maintained, citable open code for the $\mu$CF cycle that is simultaneously reproducible,
differentiable, and uncertainty-bearing**: the published kinetics models — for example
@YamashitaKino2022 and @KouChen2026 — carry no accompanying code releases.

The audiences that need such a layer are concrete. Rate-network modelers who treat effective
cross-sections as scan parameters [@KouChen2026] need propagated error bars over those ranges;
`OpenMuCF` reproduces their published fusions-per-muon benchmarks within a few percent and adds
uncertainty propagation. Groups that own the canonical multi-population kinetics [@YamashitaKino2022]
need a consistent, sourced, uncertainty-tagged set of input rates. Experimental programs at J-PARC,
PSI, and RIKEN-RAL need a transparent forward model to price yield and cycling-rate measurements
above the noise. Funding agencies and technical reviewers need an auditable, end-to-end,
uncertainty-propagated **net-electrical** energy ledger rather than a bare scientific-gain number.
Finally, evaluated-data efforts such as the Muon Nuclear Data Development Project [@Watanabe2026]
cover muon-induced reactions but not $\mu$CF cycle kinetics — the complementary slice `OpenMuCF`
occupies.

# State of the field

The microscopic ingredients of the $\mu$CF cycle are individually well studied. Effective sticking
and its reduction by reactivation were treated by @Breunlich1989 and, at the transport level, by
@Stodden1990 and @Rafelski1989, whose density-dependent stripping/excitation cross sections remain
the practical basis for the reactivation fraction. Initial sticking is taken from modern few-body
calculations [@Kamimura2023]. Resonant $dt\mu$ formation has direct experimental anchors
[@Fujiwara2000], and the density anomaly that motivates high-density programs was first seen decades
ago [@Jones1986]. Modern kinetics models [@YamashitaKino2022] assemble these into multi-population
networks. No quantum calculation of the actual muonic system in the high-density stripping regime has
been published, so the reactivation fraction above roughly liquid density is a genuine open question,
not a solved one.

What is missing is not another physics model but shared, open *infrastructure*: a provenance-tagged
rate ledger, a differentiable forward map, and honest uncertainty propagation, all reproducible from a
pinned environment. `OpenMuCF` provides exactly that layer and is explicitly complementary to
transport engines and to evaluated-data projects [@Watanabe2026].

# Software design

`OpenMuCF` is a small Python package with a layered design:

- **FAIR ledger** (`openmucf/data/`, loaded by `openmucf.rates`): a curated compilation with
  provenance of the microscopic rates (13 curated scalar rates in v1), validation targets,
  uncertainty priors, and a muon-cost table. The loader refuses any row that is unsourced or fails
  the schema. The ledger is licensed CC-BY-4.0 and described by a `datapackage.json`; it is a
  compilation with provenance, **not** an independently evaluated nuclear-data library.
- **Closed-form and differentiable ODE cycle** (`openmucf.analytic`, `openmucf.cycle`): the analytic
  fusions-per-muon map and a JAX/diffrax multi-pool network. The ODE is gated against an exact
  matrix-exponential oracle (`openmucf.exact`), so no headline number depends on the integrator's
  internal structure.
- **Energy balance** (`openmucf.energy`, `openmucf.systems`): a transparent scientific and
  net-electrical $Q$ accounting graph that places the several $\mu$CF "$Q$" conventions on one
  comparable basis, so a dimensionless gain is never quoted without its accounting.
- **Uncertainty and calibration** (`openmucf.uq`, `openmucf.calibrate`): global Sobol sensitivity,
  forward-UQ breakeven falsification, and Bayesian calibration with identifiability diagnostics.
- **Forecast registry** (`openmucf.forecast`, `forecasts/`): pre-registered, hash-stamped forecast
  cards that are pushforwards of the calibrated posterior through the analytic map, to be scored by
  CRPS and interval coverage once the relevant experiment publishes.

**Honest validation (the trust map is the honesty section).** `OpenMuCF` ships a class-tiered
validation scoreboard in which only rows tagged `independent` are genuine predictions. As of v1 the
passing set contains **no** independent rows: three registered `independent` targets FAIL by design,
each a quantified measure of the distance between the v1 placeholder formation model and the field's
own rates. Reproducing this scoreboard verbatim, including the failing rows, is a deliberate part of
the artifact:

| target | class | predicted | tolerance | verdict |
|---|---|---|---|---|
| V_kouchen_base | reproduction (fed input) | 114.5 | +-10% | PASS |
| V_kouchen_best | reproduction (fed input) | 160.3 | +-10% | PASS |
| V_petitjean | reproduction (fed input) | 130.5 | [100,150] | PASS |
| V_yamashita_lcT | shape (calibrated model) | monotone X_mu(T) rise | monotone | PASS |
| V_breunlich_lambdac | anchor-consistency | 1.44e8 s^-1 | +-30% | PASS |
| V_yamashita_ratio | shape (calibrated model) | 1.3 | +-30% | PASS |
| V_faifman_peak | anchor-consistency | 7.1e9 s^-1 | +-25% | PASS |
| V_petitjean_omega | **independent** | 0.56% | [0.40,0.50] | **FAIL** |
| V_faifman_900K | **independent** | 1.06e8 s^-1 | +-50% | **FAIL** |
| V_faifman_lowT | **independent** | 1.16e9 s^-1 | +-50% | **FAIL** |
| V_nagamine_trend | **independent** | n/a | qualitative | DEFERRED |

**Summary: 7 pass (0 independent), 3 fail (3 registered placeholder-distance findings), 1 deferred.**
The three failures are pre-registered findings, not bugs: a *pass* on any of them would be the thing
to investigate. They are the standing, quantified motivation for a sourced-formation upgrade. In the
repository, a companion trust map sorts every output into **GREEN** (citable as-is: the muon-cost
compilation, the energy-balance "Rosetta stone", the breakeven falsification and requirements form,
the sensitivity split with error bars, the forecast-registry machinery), **AMBER** (citable with the
stated basis: the calibrated effective-sticking and cycling-rate posterior), and **RED** (illustrative
only: the temperature-dependence shape and anything above roughly liquid density, where the v1
formation geometry is an unsourced placeholder that emits a runtime warning).

# Research use

`OpenMuCF` supports three concrete uses today. First, **breakeven auditing**: at liquid-scale density
under measured, unpolarized ranges the probability of exceeding 500 fusions per muon is structurally
zero, and even with an unbounded cycling rate the projection requires a reactivation fraction
$R \gtrsim 0.77$ versus the model-derived $\approx 0.35$ — a falsifiable, quantified bet stated as
requirements rather than a verdict on any group's work. Second, a **muon-cost compilation** on one
auditable basis: design studies sit at a few GeV per muon [@KellyHartRose2021], while operating
facilities are roughly three orders of magnitude worse, and re-running the net-electrical gain under
each cost tier exposes that gap in energy-return form (reported as "floor unvalidated, not
impossible"). Third, a **pre-registered forecast registry** (FC-001) that commits probabilistic
predictions with a portable payload hash before the experiments report, so the model can later be
scored honestly rather than fit after the fact.

The package is tested on Python 3.11–3.13 with a pinned lock-file for byte-reproducible results, and
its deterministic analysis documents are regenerated and diffed in continuous integration to prevent
doc-versus-code drift.

# Acknowledgements

We acknowledge the open $\mu$CF experimental and theory community whose published rates and
measurements make a sourced, provenance-tagged ledger possible.

# References
