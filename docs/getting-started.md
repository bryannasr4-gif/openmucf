# Getting started

A narrative walkthrough that mirrors the `README.md` quickstart. Every number
below is reproduced by the shipped code and the auto-generated
`VALIDATION.md` / `FINDINGS.md` / `CALIBRATION.md`; nothing here is hand-tuned.

## 0. Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                 # 113 tests (112 pass, 1 skipped-blocked)
```

## 1. Load the validated FAIR rate ledger

Everything starts from the ledger. `load_rates()` reads `openmucf/data/rates.csv`,
validates each row against `openmucf/data/rates.schema.json`, and cross-checks that every
`source_bibkey` resolves in `openmucf/data/references.bib`. If anything is missing it
raises a `ValueError` listing every problem — provenance is enforced, not
optional.

```python
from openmucf import load_rates

rates = load_rates()          # a validated RatesTable (dict-like)
```

## 2. Fusions per muon from physical conditions → X_μ ≈ 114

The one-call forward map assembles the cycle rates from the ledger plus the
physical conditions — temperature `T` [K], density `φ`, and tritium fraction
`c_t` — and integrates the differentiable diffrax cycle ODE network. At the
canonical liquid operating point **T = 300 K, φ = 1.2, c_t = 0.5** it returns
about **114 fusions per muon**:

```python
from openmucf import cycle

xmu = cycle.fusions_per_muon_from_conditions(rates, T=300, phi=1.2, c_t=0.5)
print(float(xmu))             # ~114 fusions per muon
```

That value is the anchor of the trust gate: `VALIDATION.md` reports the
collision-only baseline (`V_kouchen_base`) as predicted **114.5** vs observed
112.6, inside its ±10% tolerance.

## 3. Energy balance → scientific breakeven ≈ 284, net-electrical breakeven ≈ 2367

`EnergyChain` keeps the two energy questions separate and transparent so an
auditor sees every assumption. The **scientific** breakeven asks only "fusion
energy out vs muon-production *beam* energy in"; the **net-electrical** breakeven
runs the full documented efficiency chain (wall-plug `eta_acc`, thermal-to-
electric `eta_thermal`, blanket multiplication `M`), and is far more brutal:

```python
from openmucf.energy import EnergyChain

chain = EnergyChain()
print(chain.breakeven_xmu_sci())      # ~284  (X_mu at Q_sci = 1)
print(chain.breakeven_xmu_net())      # ~2367 (X_mu at Q_net = 1, the honest target)
```

Comparing to step 2, X_μ ≈ 114 is well short of even the scientific breakeven of
≈ 284 — and roughly a factor of 20 short of the net-electrical ≈ 2367. Making
that gap unavoidable is the whole point of the module.

## 4. Reproduce a finding → `make findings`

The headline findings run on the closed-form forward map (validated to < 1%
against the ODE) so millions of Monte-Carlo evaluations are laptop-tractable:

```bash
make findings      # sensitivity ranking + breakeven falsification -> FINDINGS.md
```

This regenerates `FINDINGS.md`, whose marquee results are:

- **Gradient cross-check:** analytic vs autodiff-through-the-stiff-ODE gradient of
  X_μ w.r.t. effective sticking agree to **0.00%** (analytic = −1.384e+04, ODE =
  −1.384e+04) — the cheap map is a faithful stand-in for the ODE.
- **Sensitivity split (Sobol total-order S_T):** X_μ is controlled by reactivation
  `R` (S_T = 0.620), then `λ_c` (0.254) and `ω_s0` (0.131); net-electrical `Q_net`
  flips to being controlled by muon cost `E_mu_GeV` (0.631) and wall-plug
  efficiency `eta_acc` (0.427). Different levers for yield vs energy.
- **Propagated uncertainty (95% CI):** X_μ = [89, 104, 122]; Q_sci =
  [0.177, 0.305, 0.839]; Q_net = [0.0107, 0.0364, 0.1246]. P(Q_sci > 1) = 0.2%,
  P(Q_net > 1) = 0.0%.
- **Breakeven falsification:** under the measured ranges **P(X_μ > 500) = 0.0%**.
  Even at zero sticking the best measured cycling rate (λ_c = 1.45e8) caps the
  yield at **X_μ = 319**, so 500 is arithmetically impossible without also roughly
  doubling λ_c *and* pushing reactivation R from ~0.35 toward ~0.94.

Two more reproducible entry points round out the trust story:

```bash
make validate      # reproduce the pre-registered targets -> VALIDATION.md (7 pass / 1 deferred / 0 fail)
make calibration   # Bayesian calibration + identifiability -> CALIBRATION.md
```

`make calibration` regenerates `CALIBRATION.md`, which shows the key
identifiability result: experiment pins the effective sticking (ω_s^eff ≈ 0.461%)
and λ_c, but the ω_s0/R split stays degenerate (correlation +0.84) — the
quantitative reason the Phase-3 microscopic surrogate is needed.

## Where to go next

- [api-overview.md](api-overview.md) — the public function/class for every module.
- `MODEL_SPEC.md` — the physics and the two derivations of the closed-form yield.
- `FINDINGS.md`, `VALIDATION.md`, `CALIBRATION.md` — the auto-generated result docs.
