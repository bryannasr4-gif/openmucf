"""Generate FINDINGS.md + publication figures from the UQ auditor (Phase 2.3). Run from repo root:

python scripts/generate_findings.py
"""

import hashlib
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from openmucf import cycle, provenance, uq  # noqa: E402
from openmucf.rates import RATES_CSV, TARGETS_CSV, load_rates  # noqa: E402

os.makedirs("figures", exist_ok=True)

sens = uq.local_sensitivities()
sob_x = uq.sobol_indices(N=8192, output="X_mu")
sob_q = uq.sobol_indices(N=8192, output="Q_net")
rob = uq.sobol_robustness(N=8192, output="X_mu")
fw = uq.forward_uq(n=400_000)
be = uq.breakeven_audit(n=400_000)
xchk = uq.cross_check_gradient()
# Fable hotfix 2026-07-08 (ws-fix-gradient-pin): rel_diff is autodiff-vs-analytic machine noise
# (~3e-13); its leading digit is env-dependent, so FINDINGS emits the asserted bound, never raw digits.
assert xchk["rel_diff"] < 1e-11, (
    f"gradient cross-check degraded: rel_diff={xchk['rel_diff']:.3e} (bound 1e-11)"
)

# eta=1-vs-5 structural bracket (section 1c): X_mu through the full ODE at the canonical OP, eta=1 vs 5.
# These 300 K numbers are grid-stable by the formation _CALIB anchor procedure (see formation.py).
_rates = load_rates()
_xmu_eta1 = float(cycle.fusions_per_muon_from_conditions(_rates, 300.0, 1.2, 0.5, eta=1.0))
_xmu_eta5 = float(cycle.fusions_per_muon_from_conditions(_rates, 300.0, 1.2, 0.5, eta=5.0))

# section 2b: Q_net under three muon-cost E_mu-tier priors (WS-E, deviation E1). The default flat
# [2,10] box in sections 1/2 is UNCHANGED; this panel ADDS a per-tier Q_net view (T1 Uniform(3.0,6.0),
# T2 Uniform(1e2,1e3), T3 Uniform(2.3e3,1e6)). Seeded forward-UQ MC (byte-stable like section 2).
_TIER_BOXES = {"T1": (3.0, 6.0), "T2": (1.0e2, 1.0e3), "T3": (2.3e3, 1.0e6)}
_tiers = {k: uq.qnet_tier_panel(lo, hi) for k, (lo, hi) in _TIER_BOXES.items()}

# ----------------------------------------------------------- headline numbers (single source of truth)
# Every number that appears BOTH in FINDINGS.md and FINDINGS_MANIFEST.json is formatted exactly once
# here; the document f-string and the manifest entries below consume the SAME strings from H, so a value
# can never differ between the doc and its recorded provenance (see openmucf/provenance.py).
H = {}


def _srow(name, s):
    """One Sobol table row: '| name | S1 +/- conf | ST +/- conf |' (3 decimals; seeded => byte-stable)."""
    return (f"| {name} | {s['S1'][name]:.3f} +/- {s['S1_conf'][name]:.3f} | "
            f"{s['ST'][name]:.3f} +/- {s['ST_conf'][name]:.3f} |")


for _name in ("R", "lambda_c", "omega_s0_pct"):
    H[f"sobol_xmu_ST_{_name}"] = f"{sob_x['ST'][_name]:.3f}"
for _name in ("E_mu_GeV", "eta_acc"):
    H[f"sobol_qnet_ST_{_name}"] = f"{sob_q['ST'][_name]:.3f}"
# ST - S1 interaction share for the top X_mu driver (the omega_s0 x R bilinear interaction)
H["sobol_xmu_interaction_R"] = f"{sob_x['ST']['R'] - sob_x['S1']['R']:.3f}"
# N-stability of the top-driver ST across N in {4096, 8192} x seed in {0, 1} (ST only; seeded, byte-stable)
_NSTAB = {(N, seed): uq.sobol_indices(N=N, output="X_mu", seed=seed)["ST"]
          for N in (4096, 8192) for seed in (0, 1)}
for (N, seed), st in _NSTAB.items():
    H[f"nstab_ST_R_{N}_{seed}"] = f"{st['R']:.3f}"
_NSTAB_TOP = {max(st, key=st.get) for st in _NSTAB.values()}
for _name in ("R", "lambda_c", "omega_s0_pct"):
    H[f"robustness_{_name}_box_i"] = f"{rob['contested_box'][_name]:.3f}"
    H[f"robustness_{_name}_box_ii"] = f"{rob['equal_relative_box'][_name]:.3f}"
H["xmu_ci_lo"] = f"{fw['X_mu']['lo']:.0f}"
H["xmu_ci_med"] = f"{fw['X_mu']['med']:.0f}"
H["xmu_ci_hi"] = f"{fw['X_mu']['hi']:.0f}"
H["qsci_ci_lo"] = f"{fw['Q_sci']['lo']:.3f}"
H["qsci_ci_med"] = f"{fw['Q_sci']['med']:.3f}"
H["qsci_ci_hi"] = f"{fw['Q_sci']['hi']:.3f}"
H["qnet_ci_lo"] = f"{fw['Q_net']['lo']:.4f}"
H["qnet_ci_med"] = f"{fw['Q_net']['med']:.4f}"
H["qnet_ci_hi"] = f"{fw['Q_net']['hi']:.4f}"
H["P_qsci_gt1"] = f"{fw['P_Qsci_gt1'] * 100:.1f}%"
H["P_qnet_gt1"] = f"{fw['P_Qnet_gt1'] * 100:.1f}%"
H["P_xmu_gt500"] = f"{be['P_xmu_gt500'] * 100:.1f}%"
H["cap_zero_sticking"] = f"{be['xmu_cap_at_measured_lambda_c']:.0f}"
H["R_required"] = f"{be['R_required_at_infinite_lambda_c']:.2f}"  # computed from the omega_s0 nominal
# the R>=0.77 point value carries an omega_s0-box band (higher initial sticking needs more R)
H["R_required_band"] = f"{be['R_required_band_lo']:.2f}-{be['R_required_band_hi']:.2f}"
H["eta_bracket_lo"] = f"{_xmu_eta1:.1f}"
H["eta_bracket_hi"] = f"{_xmu_eta5:.1f}"
H["eta_bracket_width"] = f"{_xmu_eta5 - _xmu_eta1:.1f}"
for _t in ("T1", "T2", "T3"):
    H[f"tier_qnet_Pgt1_{_t}"] = f"{_tiers[_t]['P_gt1'] * 100:.1f}%"
    H[f"tier_qnet_median_{_t}"] = f"{_tiers[_t]['median']:.2e}"


def _rank(d):
    return sorted(d.items(), key=lambda kv: -abs(kv[1]))


# ---------------------------------------------------------------- figure 1: Sobol total-order indices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (title, s) in zip(axes, [("X_mu", sob_x), ("Q_net", sob_q)], strict=False):
    items = _rank(s["ST"])
    ax.barh([k for k, _ in items], [v for _, v in items], color="#33aa66")
    ax.set_title(f"Global sensitivity (Sobol $S_T$): {title}")
    ax.set_xlabel("$S_T$")
    ax.invert_yaxis()
fig.tight_layout()
fig.savefig("figures/sobol.png", dpi=140)
plt.close(fig)

# ------------------------------------------------------------------------- figure 2: forward-UQ posteriors
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(fw["samples"]["X_mu"], bins=80, color="#6699cc")
axes[0].axvline(fw["X_mu"]["med"], color="k", label=f"median {fw['X_mu']['med']:.0f}")
axes[0].axvline(150, color="green", ls="--", label="record ~150 (high-T/c_t, outside liquid box)")
axes[0].set_title("prior-propagated $X_\\mu$ (measured liquid ranges)")
axes[0].set_xlabel("fusions per muon")
axes[0].legend(fontsize=8)
axes[1].hist(fw["samples"]["Q_net"], bins=80, color="#cc9966")
axes[1].axvline(1.0, color="r", ls="--", label="net-electrical breakeven")
axes[1].set_title("prior-propagated net-electrical $Q$")
axes[1].set_xlabel("$Q_{net}$")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/forward_uq.png", dpi=140)
plt.close(fig)

# ------------------------------------------------------ figure 3: breakeven "what-would-have-to-be-true"
fig, ax = plt.subplots(figsize=(6.5, 4.2))
R = np.linspace(0.0, 0.99, 240)
for lc in [1.0e8, 1.45e8, 2.3e8, 3.0e8]:
    ax.plot(R, uq.xmu(0.857, R, lc), label=f"$\\lambda_c$={lc:.2g} s$^{{-1}}$")
ax.axhline(500, color="k", ls=":", label="$N_\\mu$=500 (2026 claim)")
ax.axvspan(0.20, 0.45, color="green", alpha=0.15, label="$R$ band (model-derived)")
ax.set_xlabel("reactivation $R$")
ax.set_ylabel("$X_\\mu$")
ax.set_ylim(0, 700)
ax.legend(fontsize=8)
ax.set_title("What would have to be true for $N_\\mu$=500")
fig.tight_layout()
fig.savefig("figures/breakeven.png", dpi=140)
plt.close(fig)


# --------------------------------------------------------------------------------------- FINDINGS.md
def _tbl(ranked_names, s):
    header = "| input | $S_1$ (95% boot CI) | $S_T$ (95% boot CI) |\n|---|---|---|\n"
    return header + "\n".join(_srow(k, s) for k in ranked_names)


def _tbl2(dc, de, rel):
    header = f"| input | $S_T$ (contested box) | $S_T$ (equal-relative +/-{int(rel * 100)}%) |\n|---|---|---|\n"
    keys = sorted(dc, key=lambda k: -abs(dc[k]))
    rows = []
    for k in keys:
        ci = H.get(f"robustness_{k}_box_i", f"{dc[k]:.3f}")
        cii = H.get(f"robustness_{k}_box_ii", f"{de[k]:.3f}")
        rows.append(f"| {k} | {ci} | {cii} |")
    return header + "\n".join(rows)


xmu_rank = _rank(sob_x["ST"])
q_rank = _rank(sob_q["ST"])
xmu_names = [k for k, _ in xmu_rank]
q_names = [k for k, _ in q_rank]

md = f"""# FINDINGS.md -- the OpenMuCF headline results (auto-generated by `scripts/generate_findings.py`)

> Runs on the closed-form forward map X_mu = 1/(omega_s_eff + lambda_0/lambda_c) with the MEASURED
> lambda_c band -- so every result below depends only on measured rates and the muon lifetime, not on
> the v1 ODE network's structure (the network reduces exactly to this form in the single-pool V1 gate).
> Priors are **uniform over each input's measured/contested range** (see `openmucf/uq.py` `PARAMS`) --
> maximally honest about what is actually known. No number here is tuned.

## 0. Solver/autodiff cross-check (exact algebraic limit)
Analytic vs autodiff-through-the-stiff-ODE gradient of X_mu w.r.t. effective sticking, evaluated in the
single-pool limit where the network equals the closed form by construction:
analytic = {xchk["grad_analytic"]:.4g}, ODE = {xchk["grad_ode"]:.4g}, relative difference
**< 1e-11** (asserted at generation time; the measured value sits at the autodiff machine-noise floor,
whose leading digit is environment-dependent, so this doc reports the stable bound rather than raw
noise digits) -> this verifies the diffrax solver + JAX autodiff machinery (it is not an
independent test of the closed-form reduction; that reduction is the pre-registered V1 gate).

## 1. Which uncertainty actually controls the yield (global sensitivity)
Sobol first-order $S_1$ and total-order $S_T$ indices (fraction of output variance each input drives; $S_T$
includes interactions), each with a 95% bootstrap CI (200 resamples, seeded => byte-stable):

**X_mu (fusions per muon):**
{_tbl(xmu_names, sob_x)}

**Q_net (net-electrical gain):**
{_tbl(q_names, sob_q)}

The $S_T-S_1$ gap is the interaction share. For the top X_mu driver R it is
**{H["sobol_xmu_interaction_R"]}** (the omega_s0 x R bilinear interaction): small, because the omega_s0 box
is narrow, so R acts almost first-order here. Total-order ranking is stable across sample size and seed --
$S_T(R)$ over N in {{4096, 8192}} x seed in {{0, 1}} =
{H["nstab_ST_R_4096_0"]}, {H["nstab_ST_R_4096_1"]}, {H["nstab_ST_R_8192_0"]}, {H["nstab_ST_R_8192_1"]},
top driver **{sorted(_NSTAB_TOP)[0]}** stable across all four.

**Finding.** Under the contested-range priors (see section 1b for the prior-width caveat), X_mu is controlled by the sticking/reactivation pair (`omega_s0`, `R`) and the cycling
rate `lambda_c`; the muon cost and efficiencies do not enter it. But the *energy* question flips the
priority: Q_net is dominated by **`{q_rank[0][0]}`** and **`{q_rank[1][0]}`** -- the muon-production
cost and wall-plug efficiency swamp the microscopic sticking physics. **So "reduce sticking" is the
lever for yield, but "cheaper muons + higher efficiency" is the lever for energy gain.** That
reprioritization is not visible in any single-point projection.

## 1b. Is that ranking a physics fact, or a prior-width artifact?
The variance split in section 1 uses the **contested-range** box, where `R`'s measured range is relatively
wide (~+/-36% of nominal) while `omega_s0`'s is narrow (~+/-9%). Re-running Sobol under an **equal-relative**
box (each input +/-{int(rob["rel"] * 100)}% of its nominal) reorders the drivers -- **`{_rank(rob["equal_relative_box"])[0][0]}` now leads** --
so part of the contested-box ranking reflects how wide each *measured range* is, not physics alone:

{_tbl2(rob["contested_box"], rob["equal_relative_box"], rob["rel"])}

The prior-independent statements are therefore the *local elasticity* ranking at the operating point
(|dlnX_mu/dln omega_s0| > |dlnX_mu/dln lambda_c| > |dlnX_mu/dln R|) and the requirement-form result in
section 3 -- not "R is the dominant driver" as an unconditional claim.

## 1c. The eta=1-vs-5 formation debate (structural bracket, not a prior)
The epithermal enhancement eta (ledger row `eta_dtmu`) rescales the resonant dt-mu FORMATION rate; the
literature spans eta=1 (bare Faifman theory) to eta~5 (Yamashita-Kino fit). Recomputing X_mu through the
full cycle ODE at the canonical operating point (300 K, phi=1.2, c_t=0.5):

| eta | X_mu |
|---|---|
| 1 (bare theory) | {H["eta_bracket_lo"]} |
| 5 (Yamashita-Kino fit) | {H["eta_bracket_hi"]} |

so the structural bracket is X_mu(eta=5) - X_mu(eta=1) = **{H["eta_bracket_width"]}**.

eta rescales the FORMATION pathway; the measured lambda_c band in sections 1/2 already contains eta as it
occurred in the anchor experiments (accounting rule I5), so eta is reported as a structural bracket beside
the CI, never convolved into it.

## 2. Propagated uncertainty (what we can actually say today)
Monte-Carlo propagation of the measured liquid-density ranges (95% intervals; prior propagation, not a
posterior). Note the propagated interval deliberately reflects LIQUID conditions (phi ~ 1.2, T ~ 300 K):
the record X_mu ~ 150 (Jones 1986, high-T/high-c_t conditions) lies outside it by construction, as does
the Kou-Chen best case -- both correspond to conditions the liquid box excludes.

| quantity | 2.5% | median | 97.5% |
|---|---|---|---|
| X_mu | {H["xmu_ci_lo"]} | {H["xmu_ci_med"]} | {H["xmu_ci_hi"]} |
| Q_sci | {H["qsci_ci_lo"]} | {H["qsci_ci_med"]} | {H["qsci_ci_hi"]} |
| Q_net | {H["qnet_ci_lo"]} | {H["qnet_ci_med"]} | {H["qnet_ci_hi"]} |

P(Q_sci > 1) = {H["P_qsci_gt1"]} ; P(Q_net > 1) = {H["P_qnet_gt1"]}.

State-of-knowledge (posterior) X_mu and Q intervals -- as opposed to the ignorance-box propagation above --
are reported in CALIBRATION.md ("Posterior pushforward").

Structural, one-sided: the parametric intervals above sit on the v1 reduced network; the known deferred
channels bias X_mu DOWNWARD by up to ~15% combined (ttmu side-cycle, un-pinned pending acquisition;
d-recapture, bracketed in MATERIALITY.md), so intervals are best read as upper-edge-faithful.

## 2b. Q_net by muon-cost tier
Sections 1 and 2 use the default flat E_mu = [2, 10] GeV design-study box (UNCHANGED). To make the
muon-cost gap (`MUON_COST.md`) legible as an energy-return statement, the SAME seeded forward-UQ Q_net is
re-run under three tier-specific E_mu priors, with every other input (the measured omega_s0 / R / lambda_c /
eta boxes) held fixed:

| E_mu prior (muon-cost tier) | P(Q_net > 1) | median Q_net |
|---|---|---|
| T1 design studies, Uniform(3.0, 6.0) GeV | {H["tier_qnet_Pgt1_T1"]} | {H["tier_qnet_median_T1"]} |
| T2 demonstrated tech, Uniform(1e2, 1e3) GeV | {H["tier_qnet_Pgt1_T2"]} | {H["tier_qnet_median_T2"]} |
| T3 operating facilities, Uniform(2.3e3, 1e6) GeV | {H["tier_qnet_Pgt1_T3"]} | {H["tier_qnet_median_T3"]} |

**Finding.** The open-access anchor for the muon cost is Kelly, Hart & Rose (2021) at 4.70 GeV/muon
(full-text-verified; see `MUON_COST.md`). P(Q_net > 1) is {H["tier_qnet_Pgt1_T1"]} in every tier -- even the
cheapest design-study muons cap Q_net well below 1 at liquid density -- so the tier signal lives in the
MEDIAN Q_net, which collapses by ~5 orders of magnitude from T1 ({H["tier_qnet_median_T1"]}) to T3
({H["tier_qnet_median_T3"]}): the ~10^3 muon-cost gap expressed in energy-return form.

**T1 box-edge provenance.** The T1 box edges are 3.0 GeV (the Acceleron 2025 active-target slide value --
simulated, unvalidated, company slide) and 6.0 GeV (a design-study upper value). The full-text-pinned
Bertin et al. (1987) per-stopped-muon cost at liquid density is ~7.8 GeV (ABOVE this edge), with a ~3 GeV
ideal all-collected floor, and Eliezer-Henis (1994) is ~5 GeV; the box [3.0, 6.0] spans the low/central
design-study range, its edges are disclosed alongside the pinned values, and it is left UNCHANGED
(pre-registered; a discrepant pin is disclosed, never tuned away -- I2). Replacing the flat [2, 10] default
with a tiered prior is deferred to Phase-4 findings-v2 (deviation E1).

## 3. Breakeven audit (the marquee result)
The 2026 projections (Yin-Kou-Chen arXiv:2605.26432): $N_\\mu > 500$, $Q > 2$. Under the **measured,
liquid-density (phi <= ~1.45), unpolarized** uncertainty ranges:

- **P(X_mu > 500) = {H["P_xmu_gt500"]}**, P(Q_sci > 2) = {be["P_qsci_gt2"] * 100:.1f}%,
  P(Q_net > 1) = {be["P_qnet_gt1"] * 100:.1f}%. These zeros are STRUCTURAL, not Monte-Carlo estimates:
  500 lies outside the prior box's support entirely (max supported X_mu ~ 133).
- Even at **zero sticking**, the best measured cycling rate ($\\lambda_c$=1.45e8) caps the yield at
  **X_mu = {H["cap_zero_sticking"]}** at liquid density; even at the +30% reproduction
  band on lambda_c the cap is ~414 < 500. Density scaling (lambda_c = phi*lambda_c_tilde) at the
  demonstrated DAC phi=2.4 would lift the decay-only cap to ~530-640 *if phi-linearity holds there* --
  which is precisely the unmeasured question the MuFusE program tests.
- **What would have to be true** for $N_\\mu$=500: the (lambda_c, R) frontier runs from
  (2.28e8, R -> 1) to (3e8, R = {be["R_required_at_lambda_c_3e8"]:.2f}); and even at infinite lambda_c,
  omega_s_eff <= 0.2% i.e. **R >= {H["R_required"]}** is required (R >= {H["R_required_band"]} across the
  omega_s0-box band -- higher initial sticking needs more reactivation). For reference R ~ 0.35 is the model-derived
  collisional value (Kou-Chen Eq.33) -- experiment pins only the product omega_s_eff ~ 0.45%, and our
  Kamimura-prior calibration posterior gives R = 0.46 +- 0.06.

**Verdict.** The 2026 breakeven projection is *not falsified in principle* -- and this audit does not
evaluate the polarization / field-assisted-recovery mechanisms it invokes -- but expressed as
requirements, any such mechanism must push reactivation to R ~ 0.9+ (density can supply the cycling-rate
factor, sticking it cannot). That turns a headline into a falsifiable, quantitative bet on **exactly the
quantity Acceleron's diamond-anvil program measures and the Phase-3 sticking surrogate will forecast.**
(Figure `figures/breakeven.png`.)

## Honest caveats
- These use the closed-form yield map with uniform priors over contested ranges; the
  sticking/reactivation inputs are the v1 literature band, not yet the Phase-3 surrogate. The
  falsification result depends only on measured lambda_c and lambda_0 -- not on the v1 network
  structure -- which is the strongest defense of its robustness.
- All probability statements are scoped to liquid-scale density and unpolarized targets; the
  Yin-Kou-Chen projection's polarization levers are outside this model's support and are audited as
  REQUIREMENTS (R >= 0.77 at any density), not refuted mechanisms.
- Q_net uses the transparent default efficiency chain (`energy.py`); every factor is a documented knob.
  E_mu is beam energy per muon delivered (Breunlich 1989 convention); wall-plug efficiency enters
  separately as eta_acc.
- **Muon-cost caveat (the Q_net floor).** The 2-10 GeV E_mu range is a *design-study* figure for an
  unbuilt, purpose-built muon source; existing facilities are ~10^3x worse per delivered muon (they
  optimize beam brightness, not muons-per-watt). So the Q_net interval above is a best-case floor
  conditional on such a source existing -- real-facility Q_net today would be far lower. (The
  efficiency-free Q_sci comparison to Yin-Kou-Chen is unaffected: it is genuinely same-basis.)
- The blanket multiplier M=1 (pure muCF); a fission/breeding hybrid (M>1) is a separate, explicit knob.

Figures: `figures/sobol.png`, `figures/forward_uq.png`, `figures/breakeven.png`.
"""

with open("FINDINGS.md", "w") as f:
    f.write(md)


# ---------------------------------------------------- machine-checkable provenance manifest (WS-A)
# Built from the SAME H used for the document above, so every tracked value is identical by construction.
def _entry(entry_id, pattern):
    return provenance.ManifestEntry(
        id=entry_id,
        value=H[entry_id],
        pattern=pattern,
        source_type="derivation",
        source="scripts/generate_findings.py",
        doc="FINDINGS.md",
    )


_entries = [
    # each ST value is anchored to its FULL Sobol row (now '| name | S1 +/- conf | ST +/- conf |')
    _entry("sobol_xmu_ST_R", re.escape(_srow("R", sob_x))),
    _entry("sobol_xmu_ST_lambda_c", re.escape(_srow("lambda_c", sob_x))),
    _entry("sobol_xmu_ST_omega_s0_pct", re.escape(_srow("omega_s0_pct", sob_x))),
    _entry("sobol_qnet_ST_E_mu_GeV", re.escape(_srow("E_mu_GeV", sob_q))),
    _entry("sobol_qnet_ST_eta_acc", re.escape(_srow("eta_acc", sob_q))),
]
for _name in ("R", "lambda_c", "omega_s0_pct"):
    _bi = re.escape(H[f"robustness_{_name}_box_i"])
    _bii = re.escape(H[f"robustness_{_name}_box_ii"])
    _row = rf"\| {_name} \| {_bi} \| {_bii} \|"
    _entries.append(_entry(f"robustness_{_name}_box_i", _row))
    _entries.append(_entry(f"robustness_{_name}_box_ii", _row))
_xmu_row = rf"\| X_mu \| {re.escape(H['xmu_ci_lo'])} \| {re.escape(H['xmu_ci_med'])} \| {re.escape(H['xmu_ci_hi'])} \|"
for _k in ("xmu_ci_lo", "xmu_ci_med", "xmu_ci_hi"):
    _entries.append(_entry(_k, _xmu_row))
_qsci_row = rf"\| Q_sci \| {re.escape(H['qsci_ci_lo'])} \| {re.escape(H['qsci_ci_med'])} \| {re.escape(H['qsci_ci_hi'])} \|"
for _k in ("qsci_ci_lo", "qsci_ci_med", "qsci_ci_hi"):
    _entries.append(_entry(_k, _qsci_row))
_qnet_row = rf"\| Q_net \| {re.escape(H['qnet_ci_lo'])} \| {re.escape(H['qnet_ci_med'])} \| {re.escape(H['qnet_ci_hi'])} \|"
for _k in ("qnet_ci_lo", "qnet_ci_med", "qnet_ci_hi"):
    _entries.append(_entry(_k, _qnet_row))
_entries += [
    _entry("P_qsci_gt1", rf"P\(Q_sci > 1\) = {re.escape(H['P_qsci_gt1'])}"),
    _entry("P_qnet_gt1", rf"P\(Q_net > 1\) = {re.escape(H['P_qnet_gt1'])}"),
    _entry("P_xmu_gt500", rf"P\(X_mu > 500\) = {re.escape(H['P_xmu_gt500'])}"),
    _entry("cap_zero_sticking", rf"\*\*X_mu = {re.escape(H['cap_zero_sticking'])}\*\*"),
    _entry("R_required", rf"\*\*R >= {re.escape(H['R_required'])}\*\*"),
    _entry("R_required_band", rf"R >= {re.escape(H['R_required_band'])} across the"),
    _entry("eta_bracket_lo", rf"\| 1 \(bare theory\) \| {re.escape(H['eta_bracket_lo'])} \|"),
    _entry("eta_bracket_hi", rf"\| 5 \(Yamashita-Kino fit\) \| {re.escape(H['eta_bracket_hi'])} \|"),
    _entry("eta_bracket_width", rf"X_mu\(eta=5\) - X_mu\(eta=1\) = \*\*{re.escape(H['eta_bracket_width'])}\*\*"),
]
# section 2b (WS-E tier panel): anchor each tracked number to its row of the Q_net-by-tier table.
_tier_rows = {
    "T1": r"T1 design studies, Uniform\(3\.0, 6\.0\) GeV",
    "T2": r"T2 demonstrated tech, Uniform\(1e2, 1e3\) GeV",
    "T3": r"T3 operating facilities, Uniform\(2\.3e3, 1e6\) GeV",
}
for _t in ("T1", "T2", "T3"):
    _p = _tier_rows[_t]
    _entries.append(_entry(f"tier_qnet_Pgt1_{_t}", rf"{_p} \| {re.escape(H[f'tier_qnet_Pgt1_{_t}'])} \|"))
    _entries.append(
        _entry(f"tier_qnet_median_{_t}", rf"{_p}[^\n]*\| {re.escape(H[f'tier_qnet_median_{_t}'])} \|")
    )

_manifest_inputs = {
    "rates_csv_sha256": provenance.file_sha256(RATES_CSV),
    "validation_targets_csv_sha256": provenance.file_sha256(TARGETS_CSV),
    "uq_params_repr_sha256": hashlib.sha256(repr(uq.PARAMS).encode("utf-8")).hexdigest(),
    "seeds": {"sobol": 0, "forward_uq": 0, "breakeven": 1, "tier_panel": 0},
}
provenance.write_manifest("FINDINGS_MANIFEST.json", _entries, _manifest_inputs)

print("wrote FINDINGS.md and figures/{sobol,forward_uq,breakeven}.png")
print(f"wrote FINDINGS_MANIFEST.json ({len(_entries)} provenance entries)")
print(f"gradient cross-check rel.diff = {xchk['rel_diff']:.1e}")
print("X_mu Sobol ST:", {k: round(v, 3) for k, v in xmu_rank})
print("X_mu Sobol ST (equal-relative box):", {k: round(v, 3) for k, v in _rank(rob["equal_relative_box"])})
print("Q_net Sobol ST:", {k: round(v, 3) for k, v in q_rank})
print(f"X_mu 95% CI = [{fw['X_mu']['lo']:.0f}, {fw['X_mu']['hi']:.0f}], median {fw['X_mu']['med']:.0f}")
print(f"Q_net 95% CI = [{fw['Q_net']['lo']:.4f}, {fw['Q_net']['hi']:.4f}]")
print(
    f"P(X_mu>500)={be['P_xmu_gt500']:.3f}  xmu_cap@measured_lc={be['xmu_cap_at_measured_lambda_c']:.0f}"
    f"  R_req@3e8={be['R_required_at_lambda_c_3e8']:.2f}"
)
