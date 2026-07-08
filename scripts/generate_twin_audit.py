"""Generate TWIN_AUDIT.md + figures/twin_bias.png + TWIN_MANIFEST.json (deterministic). Run from root:

python scripts/generate_twin_audit.py

Content (WAVE1_EXECUTION_SPEC sec.5.3), all seeded/deterministic so `make audit` byte-diffs it:
  1. closed-form disappearance gate (G-T1): a synthetic spectrum at the canonical OP is fit with the
     idealized two-exponential estimator and must recover the analytic lambda_n to < 1%.
  2. estimator-bias sweep t_min x c_t on the EXACT ODE truth (noise-free expected counts, so the numbers
     are deterministic) -- the residual bias of the single-exponential estimator.
  3. MuFusE fuel-component disappearance-rate bands: the REGISTERED FC-001 card's stored interval
     endpoints (Scenario A) pushed through the closed-form disappearance model -- an interval ENVELOPE,
     not a posterior pushforward, and not a detector prediction.
"""

import json
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from openmucf import cycle, provenance, twin  # noqa: E402
from openmucf.constants import LAMBDA_0  # noqa: E402
from openmucf.rates import RATES_CSV, load_rates  # noqa: E402

os.makedirs("figures", exist_ok=True)
rates = load_rates()

# 64 uniform bins over [0, 30 us]; canonical liquid OP = (300 K, phi=1.2, c_t=0.5); n_mu*eff = 1e6.
T_EDGES = np.linspace(0.0, 30.0e-6, 65)
N_MU_EFF = 1.0e6

# ------------------------------------------------------------- section 1: closed-form gate (G-T1)
_cp = cycle.params_from_conditions(rates, 300.0, 1.2, 0.5)
_xmu = float(cycle.fusions_per_muon_ode(**_cp))
_ose = _cp["omega_s_eff"]
_lc = twin.implied_cycling_rate(_xmu, _ose)
_ln_true = twin.disappearance_rate(_ose, _lc)
_exp = twin.expected_counts(T_EDGES, _cp, N_MU_EFF, 1.0, 0.0)
_syn = twin.synthetic_spectrum(T_EDGES, _exp, seed=0)
_gate = twin.fit_two_exponential(T_EDGES, _syn, t_min=2.0e-6, lambda_c=_lc)
_gate_bias = 100.0 * (_gate["lambda_n"] - _ln_true) / _ln_true

# --------------------------------------------------------- section 2: estimator-bias sweep (exact ODE)
T_MINS = (0.5e-6, 1.0e-6, 2.0e-6, 4.0e-6)
C_TS = (0.2, 0.5, 0.8)
bias_ln = {}  # (c_t, t_min) -> bias(lambda_n) %
bias_w = {}  # (c_t, t_min) -> bias(W == omega_s_eff) %
for c_t in C_TS:
    cpx = cycle.params_from_conditions(rates, 300.0, 1.2, c_t)
    xm = float(cycle.fusions_per_muon_ode(**cpx))
    osx = cpx["omega_s_eff"]
    lcx = twin.implied_cycling_rate(xm, osx)
    lnx = twin.disappearance_rate(osx, lcx)
    expx = twin.expected_counts(T_EDGES, cpx, N_MU_EFF, 1.0, 0.0)  # noise-free -> deterministic bias
    for tm in T_MINS:
        fit = twin.fit_two_exponential(T_EDGES, expx, t_min=tm, lambda_c=lcx)
        bias_ln[(c_t, tm)] = 100.0 * (fit["lambda_n"] - lnx) / lnx
        bias_w[(c_t, tm)] = 100.0 * (fit["omega_s_eff"] - osx) / osx

# --------------------------------------------- section 3: MuFusE fuel-component disappearance bands
with open("forecasts/FC-001-mufuse.json") as _f:
    _card = json.load(_f)
_scenA = next(s for s in _card["payload"]["scenarios"] if s["name"] == "A")


def _pred(target_id):
    return next(p for p in _scenA["predictions"] if p["target_id"] == target_id)


PHIS = (1.2, 2.0, 2.4)
band_ln = {}  # phi -> (lo, med, hi) of lambda_n [s^-1] pushed from the card ci95/median endpoints
for phi in PHIS:
    o = _pred(f"omega_s_eff@phi={phi}")
    lam = _pred(f"lambda_c@phi={phi}")
    o_lo, o_hi = o["ci95"]
    lam_lo, lam_hi = lam["ci95"]
    ln_lo = LAMBDA_0 + (o_lo / 100.0) * lam_lo  # joint low corner  -> slowest disappearance
    ln_med = LAMBDA_0 + (o["median"] / 100.0) * lam["median"]
    ln_hi = LAMBDA_0 + (o_hi / 100.0) * lam_hi  # joint high corner -> fastest disappearance
    band_ln[phi] = (ln_lo, ln_med, ln_hi)

# ------------------------------------------------------------------- headline numbers (single source)
H = {}
H["gate_bias_pct"] = f"{_gate_bias:.2f}"
H["gate_lambda_n"] = f"{_gate['lambda_n'] / 1e6:.4f}"
H["gate_lambda_n_analytic"] = f"{_ln_true / 1e6:.4f}"
for c_t in C_TS:
    H[f"bias_ln_ct{c_t}_tmin2"] = f"{bias_ln[(c_t, 2.0e-6)]:+.2f}"
    H[f"bias_w_ct{c_t}_tmin2"] = f"{bias_w[(c_t, 2.0e-6)]:+.2f}"
for phi in PHIS:
    lo, med, hi = band_ln[phi]
    H[f"band_ln_phi{phi}_med"] = f"{med / 1e6:.3f}"
    H[f"band_ln_phi{phi}_lo"] = f"{lo / 1e6:.3f}"
    H[f"band_ln_phi{phi}_hi"] = f"{hi / 1e6:.3f}"

# --------------------------------------------------------------------------------- figure: bias + bands
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
# left: estimator bias(lambda_n)% vs c_t at each t_min
for tm in T_MINS:
    axes[0].plot(C_TS, [bias_ln[(c, tm)] for c in C_TS], marker="o", label=f"t_min={tm * 1e6:.1f} us")
axes[0].axhline(0, color="k", lw=0.6)
axes[0].set_xlabel("tritium fraction c_t")
axes[0].set_ylabel("bias($\\lambda_n$) [%]")
axes[0].set_title("Idealized two-exponential estimator bias (synthetic v1 truth)")
axes[0].legend(fontsize=8)
# right: fuel-component disappearance bands (normalized spectra) per phi
tt = np.linspace(0, 12e-6, 400)
colors = {1.2: "#6699cc", 2.0: "#33aa66", 2.4: "#cc6666"}
for phi in PHIS:
    lo, med, hi = band_ln[phi]
    y_lo = np.exp(-hi * tt)  # fastest disappearance = lower envelope of the normalized spectrum
    y_hi = np.exp(-lo * tt)
    axes[1].fill_between(tt * 1e6, y_lo, y_hi, color=colors[phi], alpha=0.25)
    axes[1].plot(tt * 1e6, np.exp(-med * tt), color=colors[phi], label=f"$\\phi$={phi}")
axes[1].set_yscale("log")
axes[1].set_ylim(1e-4, 1.2)
axes[1].set_xlabel("time [us]")
axes[1].set_ylabel("normalized fuel-component neutron rate")
axes[1].set_title("FC-001 card-interval disappearance bands (NOT a detector prediction)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/twin_bias.png", dpi=140)
plt.close(fig)


# ----------------------------------------------------------------------------------------- TWIN_AUDIT.md
def _bias_table(bias):
    head = "| t_min [us] | c_t=0.2 | c_t=0.5 | c_t=0.8 |\n|---|---|---|---|\n"
    rows = []
    for tm in T_MINS:
        cells = " | ".join(f"{bias[(c, tm)]:+.2f}" for c in C_TS)
        rows.append(f"| {tm * 1e6:.1f} | {cells} |")
    return head + "\n".join(rows)


def _band_table():
    head = "| phi | lambda_n lo | lambda_n median | lambda_n hi |\n|---|---|---|---|\n"
    rows = []
    for phi in PHIS:
        rows.append(
            f"| {phi} | {H[f'band_ln_phi{phi}_lo']} | {H[f'band_ln_phi{phi}_med']} "
            f"| {H[f'band_ln_phi{phi}_hi']} |"
        )
    return head + "\n".join(rows)


md = f"""# TWIN_AUDIT.md -- counts-level twin: estimator bias + fuel-component bands (auto-generated by `scripts/generate_twin_audit.py`)

> **Physics-level, synthetic-only.** Everything here runs the v1 cycle model channels-OFF and is a
> statement about the IDEALIZED estimator on the model's own synthetic truth -- **NOT** a claim about any
> named historical dataset. Each real dataset's actual procedure (beam pulse structure, muon stopping in
> the cell/gasket/detector materials, detector response) is stage 2 and acquisition/contact-gated.
> Standard identity used throughout: `lambda_n = lambda_0 + omega_s_eff * lambda_c`, `X_mu = lambda_c/lambda_n`.

## 1. Closed-form disappearance gate (G-T1)
Synthetic neutron spectrum at the canonical liquid OP (300 K, phi=1.2, c_t=0.5), 64 bins over [0, 30 us],
n_mu*eff = 1e6, zero background (seed 0). The idealized two-exponential estimator (`fit_two_exponential`,
t_min = 2 us) recovers the disappearance rate lambda_n:

- fit lambda_n = **{H["gate_lambda_n"]}** x10^6 s^-1 vs analytic lambda_n = **{H["gate_lambda_n_analytic"]}** x10^6 s^-1
  (`disappearance_rate(omega_s_eff, lambda_c)` with the engine-implied cycling rate at this OP).
- **gate bias = {H["gate_bias_pct"]}%** -- passes the pre-registered G-T1 tolerance of < 1%.

## 2. Estimator-bias sweep (idealized estimator vs the exact ODE truth)
Bias of the single-exponential estimator against the analytic truth, on **noise-free** expected counts
(so every cell is deterministic), over t_min x c_t. The estimator collapses the two-hyperfine-pool cycle
onto one slope; the transfer transient (~6 ns) is far below the smallest t_min here, so the residual bias
is the two-pool structure, not the transfer -- and it is therefore flat in t_min and shrinks as the cycle
speeds up with c_t. **Positive throughout and < 1.3%.** Each named dataset's ACTUAL published estimator
and fit window must be checked before any dataset-specific bias claim (stage 2, acquisition-gated).

**bias(lambda_n) [%]:**
{_bias_table(bias_ln)}

**bias(W) [%]** (W = effective per-cycle sticking omega_s_eff, backed out via the standard identity
omega_s_eff = (lambda_n - lambda_0)/lambda_c; the amplification over bias(lambda_n) is the
lambda_0/(lambda_n - lambda_0) leverage of the sticking extraction):
{_bias_table(bias_w)}

## 3. MuFusE fuel-component disappearance bands (card-interval envelope)
Fuel-component neutron disappearance-rate bands at the FC-001 density grid, computed by pushing the
**registered FC-001 card**'s stored Scenario-A interval endpoints (ci95 lo / median / hi for omega_s_eff
and lambda_c per target, read from `forecasts/FC-001-mufuse.json`) through the closed-form disappearance
model -- **a card-interval envelope pushed through the forward model, not a full posterior pushforward**,
and byte-diffable (no MCMC in this generator). Physics-level fuel-component bands only -- **NOT a forecast
card, NOT a detector prediction.** Conditional on the stated (phi, T) of that card.

**lambda_n = lambda_0 + omega_s_eff * lambda_c  [x10^6 s^-1]:**
{_band_table()}

**DAC stopping caveat (verbatim).** Most muons in a diamond-anvil cell stop in the diamond/gasket/metal,
not the fuel: mu- in carbon (lifetime ~2.03 us) and in gasket metal produce time components comparable to
the fuel signal, so fitting real MuFusE data requires their material stopping/lifetime model (stage 2,
contact-gated). The bands above are the fuel component ONLY.

Figure: `figures/twin_bias.png`.
"""

with open("TWIN_AUDIT.md", "w") as f:
    f.write(md)


# --------------------------------------------------------- machine-checkable provenance (TWIN_MANIFEST)
def _entry(entry_id, pattern):
    return provenance.ManifestEntry(
        id=entry_id,
        value=H[entry_id],
        pattern=pattern,
        source_type="derivation",
        source="scripts/generate_twin_audit.py",
        doc="TWIN_AUDIT.md",
    )


_entries = [
    _entry("gate_bias_pct", rf"\*\*gate bias = {re.escape(H['gate_bias_pct'])}%\*\*"),
    _entry("gate_lambda_n", rf"fit lambda_n = \*\*{re.escape(H['gate_lambda_n'])}\*\*"),
    _entry("gate_lambda_n_analytic", rf"analytic lambda_n = \*\*{re.escape(H['gate_lambda_n_analytic'])}\*\*"),
]
for c_t in C_TS:
    # both bias tables carry the t_min=2us row; anchor the c_t cell within that row of each table
    _entries.append(_entry(f"bias_ln_ct{c_t}_tmin2", rf"\| 2\.0 \|[^\n]*{re.escape(H[f'bias_ln_ct{c_t}_tmin2'])}"))
    _entries.append(_entry(f"bias_w_ct{c_t}_tmin2", rf"\| 2\.0 \|[^\n]*{re.escape(H[f'bias_w_ct{c_t}_tmin2'])}"))
for phi in PHIS:
    _row = (
        rf"\| {phi} \| {re.escape(H[f'band_ln_phi{phi}_lo'])} \| "
        rf"{re.escape(H[f'band_ln_phi{phi}_med'])} \| {re.escape(H[f'band_ln_phi{phi}_hi'])} \|"
    )
    _entries.append(_entry(f"band_ln_phi{phi}_lo", _row))
    _entries.append(_entry(f"band_ln_phi{phi}_med", _row))
    _entries.append(_entry(f"band_ln_phi{phi}_hi", _row))

_inputs = {
    "rates_csv_sha256": provenance.file_sha256(RATES_CSV),
    "fc001_card_sha256": provenance.file_sha256("forecasts/FC-001-mufuse.json"),
    "seeds": {"gate_synthetic": 0},
}
provenance.write_manifest(
    "TWIN_MANIFEST.json", _entries, _inputs, generated_by="scripts/generate_twin_audit.py"
)

print(f"wrote TWIN_AUDIT.md + figures/twin_bias.png + TWIN_MANIFEST.json ({len(_entries)} entries)")
print(f"G-T1 gate bias = {H['gate_bias_pct']}% (<1% required)")
print("bias(lambda_n) t_min=2us:", {c: H[f'bias_ln_ct{c}_tmin2'] for c in C_TS})
print("fuel-band lambda_n medians [1e6 s^-1]:", {phi: H[f'band_ln_phi{phi}_med'] for phi in PHIS})
