"""Generate FINDINGS.md + publication figures from the UQ auditor (Phase 2.3). Run from repo root:

python scripts/generate_findings.py
"""

import csv
import hashlib
import math
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from openmucf import cycle, mucost, provenance, uq  # noqa: E402
from openmucf.constants import LAMBDA_0  # noqa: E402
from openmucf.rates import RATES_CSV, TARGETS_CSV, load_rates  # noqa: E402

os.makedirs("figures", exist_ok=True)

sens = uq.local_sensitivities()
sob_x = uq.sobol_indices(N=8192, output="X_mu")
sob_q = uq.sobol_indices(N=8192, output="Q_net")
rob = uq.sobol_robustness(N=8192, output="X_mu")
fw = uq.forward_uq(n=400_000)
be = uq.breakeven_audit(n=400_000)
xchk = uq.cross_check_gradient()
# The gradient cross-check ships as a BOUND, never a pinned digit: rel_diff is autodiff-vs-analytic noise
# (~3e-13); its leading digit is env-dependent, so FINDINGS emits the asserted bound, never raw digits.
assert xchk["rel_diff"] < 1e-11, (
    f"gradient cross-check degraded: rel_diff={xchk['rel_diff']:.3e} (bound 1e-11)"
)

# eta=1-vs-5 structural bracket (section 1c): X_mu through the full ODE at the canonical OP, eta=1 vs 5.
# These 300 K numbers are grid-stable by the formation _CALIB anchor procedure (see formation.py).
_rates = load_rates()
_xmu_eta1 = float(cycle.fusions_per_muon_from_conditions(_rates, 300.0, 1.2, 0.5, eta=1.0))
_xmu_eta5 = float(cycle.fusions_per_muon_from_conditions(_rates, 300.0, 1.2, 0.5, eta=5.0))

# section 2b: Q_net under three muon-cost E_mu-tier priors. The default flat [2,10] box in sections
# 1/2 is UNCHANGED; this panel ADDS a per-tier Q_net view. Seeded forward-UQ MC (byte-stable like
# section 2). The boxes come from `mucost.panel_tier_boxes`, which carries each edge's provenance and
# enforces that no edge -- and no box's support -- may take in a row barred from muCF cost aggregates;
# the paragraphs below render that provenance rather than assert it. Every edge read off a row is that
# row's own pinned value, so the ledger moving moves this section and the byte-diff sees it.
_MU_COST = mucost.load_muon_cost()
_TIER_BOXES = mucost.panel_tier_boxes(_MU_COST)
_TIER_NAMES = {"T1": "design studies", "T2": "demonstrated tech", "T3": "operating facilities"}
_tiers = {k: uq.qnet_tier_panel(lo.value, hi.value) for k, (lo, hi) in _TIER_BOXES.items()}

# The section-2b provenance prose contrasts the two T3 edge-setting rows by charge and stage.
# Everything those sentences state is either rendered from the row below or asserted here, so a
# ledger move that falsifies a sentence refuses to write the document instead of shipping a
# self-contradiction the byte-diff cannot see.
_T3_LO_ROW = _MU_COST[_TIER_BOXES["T3"][0].source_id]
_T3_HI_ROW = _MU_COST[_TIER_BOXES["T3"][1].source_id]
assert _T3_HI_ROW.charge_basis == "mixed", (
    f"the MIXED-charge upper-edge sentence is no longer true: {_T3_HI_ROW.source_id} is "
    f"{_T3_HI_ROW.charge_basis!r} -- re-word the T3 box-edges paragraph before regenerating"
)
assert _T3_LO_ROW.stage in mucost.OFF_CHAIN_STAGES and _T3_LO_ROW.stage != _T3_HI_ROW.stage, (
    "the DIFFERENT-stages sentence is no longer true -- re-word the T3 box-edges paragraph"
)
# The AMENDMENT quotes the excluded figure as it stood at the 2026-08-19 retraction. It is rendered
# from the row AND pinned here: if the row ever moves, regeneration must not silently rewrite the
# history the amendment records -- re-word it to past-tense the figure instead.
_PSI = _MU_COST["psi_himb"]
assert (_PSI.normalized_GeV_per_mu, _PSI.stage, _PSI.charge_basis) == (
    890000.0,
    "transported",
    "mu_plus_only",
), "psi_himb moved: the AMENDMENT quotes its 2026-08-19 coordinates; re-word it, do not regenerate"

# Section 2 says the Jones record and the Kou-Chen best case both need an effective sticking BELOW
# this box's support, and that neither exits on conditions. Both halves are read from the data and
# pinned here: if a prior moves, regeneration fails loudly rather than shipping a stale reason.
# Re-word the paragraph; do not regenerate past this.
_OS0_P = next(p for p in uq.PARAMS if p.name == "omega_s0_pct")
_R_P = next(p for p in uq.PARAMS if p.name == "R")
_LC_P = next(p for p in uq.PARAMS if p.name == "lambda_c")
_OSE_SUPPORT_LO = _OS0_P.low / 100.0 * (1.0 - _R_P.high)
_JONES_NEEDS_OSE = 1.0 / 150.0 - LAMBDA_0 / _LC_P.high
with open(TARGETS_CSV, encoding="utf-8") as _f:
    _KOU_BEST = next(r for r in csv.DictReader(_f) if r["target_id"] == "V_kouchen_best")
_KOU_OSE = float(re.search(r"omega_s_eff\s*=\s*([0-9.]+)\s*%", _KOU_BEST["conditions"]).group(1)) / 100.0
assert _JONES_NEEDS_OSE < _OSE_SUPPORT_LO and _KOU_OSE < _OSE_SUPPORT_LO, (
    "the section-2 sentence says both the Jones record and the Kou-Chen best case need an effective "
    "sticking below this box's support; that is no longer true -- re-word the paragraph"
)
# ... and its conclusion, which the premise above does not by itself pin: both figures must
# still lie ABOVE the box. This reads the row's stated value, not just its conditions, so the two
# cannot drift apart silently.
_BOX_MAX_XMU = 1.0 / (_OSE_SUPPORT_LO + LAMBDA_0 / _LC_P.high)
assert _BOX_MAX_XMU < 150.0 and float(_KOU_BEST["value"]) > _BOX_MAX_XMU, (
    "the section-2 sentence says both figures lie ABOVE this box's attainable maximum; that is "
    "no longer true -- re-word the paragraph"
)


def _box_label(t):
    """'T3 operating facilities, Uniform(2286, 6002) GeV' -- the table row's own subject, rendered once.

    Used for BOTH the document row and the manifest anchor, so a box edge cannot move in one and not
    the other; each edge renders itself (`BoxEdge.render`) rather than being formatted here.
    """
    lo, hi = _TIER_BOXES[t]
    return f"{t} {_TIER_NAMES[t]}, Uniform({lo.render()}, {hi.render()}) GeV"


def _box_composition(t):
    """One markdown line per edge of box ``t``: what sets it, and its full basis coordinate.

    Every edge gets a line, including a declared constant -- which is named as declared rather than
    left out, because an edge with nothing said about it is exactly the state this section is
    amending. A ledger edge prints the row's stage, numeraire and charge basis, read from the row
    rather than described. Line breaks are fixed here so the rendered document wraps the same way on
    every run.
    """
    lo, hi = _TIER_BOXES[t]
    lines = []
    for edge, which in ((lo, "lower"), (hi, "upper")):
        if not edge.from_ledger:
            lines.append(
                f"- {which} edge {edge.render()} GeV -- a DECLARED constant, not a ledger row; its "
                f"reason is\n  stated above."
            )
            continue
        r = _MU_COST[edge.source_id]
        lines.append(
            f"- {which} edge {edge.render()} GeV -- ledger row `{r.source_id}`: stage `{r.stage}`, "
            f"numeraire\n  `{r.numeraire}`, charge basis `{r.charge_basis}`."
        )
    return "\n".join(lines)


# The Finding sentence below states one P(Q_net > 1) for all three tiers. That is a claim about the
# rendered numbers, so it is checked here instead of being trusted: if the tiers ever separate, the
# sentence stops being true and this stops the document being written rather than shipping it.
_pgt1 = {k: f"{v['P_gt1'] * 100:.1f}%" for k, v in _tiers.items()}
assert len(set(_pgt1.values())) == 1, f"the tiers no longer share one P(Q_net>1): {_pgt1}"

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
# the cap is lambda_c/lambda_0 and therefore CONDITION-dependent; the ledger carries a second,
# condition-tagged anchor (SIN 12 K solid) whose sticking is paired to the SAME measurement
H["cap_zero_sticking_solid"] = f"{be['xmu_cap_at_solid_lambda_c']:.0f}"
H["yield_solid_pair"] = f"{be['yield_at_solid_anchor_pair']:.0f}"
H["R_required"] = f"{be['R_required_at_infinite_lambda_c']:.2f}"  # computed from the omega_s0 nominal
# the R>=0.77 point value carries an omega_s0-box band (higher initial sticking needs more R)
H["R_required_band"] = f"{be['R_required_band_lo']:.2f}-{be['R_required_band_hi']:.2f}"
# the same requirement in the ledger's TWO-FACTOR form: R_col and R_X are successive, never comparable
H["R_col_ref"] = f"{be['R_col_reference']:.2f}"
H["R_X_required"] = f"{be['R_X_required_given_R_col']:.2f}"
# The rendered box edges, so the document, the provenance paragraphs and the manifest all print the
# same strings from one place (each edge renders itself; nothing here reformats a value).
for _t, (_lo, _hi) in _TIER_BOXES.items():
    H[f"{_t.lower()}_lo"] = _lo.render()
    H[f"{_t.lower()}_hi"] = _hi.render()
H["eta_bracket_lo"] = f"{_xmu_eta1:.1f}"
H["eta_bracket_hi"] = f"{_xmu_eta5:.1f}"
H["eta_bracket_width"] = f"{_xmu_eta5 - _xmu_eta1:.1f}"
for _t in ("T1", "T2", "T3"):
    H[f"tier_qnet_Pgt1_{_t}"] = f"{_tiers[_t]['P_gt1'] * 100:.1f}%"
    H[f"tier_qnet_median_{_t}"] = f"{_tiers[_t]['median']:.2e}"
# What the T1 -> T3 fall actually tracks. Q_net goes as 1/E_mu at fixed other inputs, so each row is
# governed by where its box sits; these two ratios are published side by side so a reader can see that
# the fall is a property of the chosen support rather than a measurement of the muon-cost spread. Both
# are computed, never typed, and a test binds their agreement so the sentence cannot go stale quietly.
_MIDPOINT = {t: (lo.value + hi.value) / 2.0 for t, (lo, hi) in _TIER_BOXES.items()}
H["tier_span_oom"] = f"{math.log10(_tiers['T1']['median'] / _tiers['T3']['median']):.0f}"


def _two_sig_figs(x):
    """A value at TWO significant figures, rendered without an exponent.

    Both published ratios are quoted at this precision and not finer, because the medians in the
    table above are printed to three figures: a ratio quoted to one decimal cannot be reproduced from
    them, and a reader who tries gets a different number in the last two digits. The paragraph exists
    to be checked, so it may not quote a precision its own inputs cannot support.
    """
    return f"{float(f'{x:.2g}'):.0f}"


H["tier_median_ratio"] = _two_sig_figs(_tiers["T1"]["median"] / _tiers["T3"]["median"])
H["tier_midpoint_T1"] = f"{_MIDPOINT['T1']:g}"
H["tier_midpoint_T3"] = f"{_MIDPOINT['T3']:g}"
H["tier_midpoint_ratio"] = _two_sig_figs(_MIDPOINT["T3"] / _MIDPOINT["T1"])
# The check a reader would actually run: divide the two medians AS PRINTED. It must land on the same
# published figure, or this document quotes a number its own table cannot reproduce.
_printed_ratio = float(H["tier_qnet_median_T1"]) / float(H["tier_qnet_median_T3"])
assert _two_sig_figs(_printed_ratio) == H["tier_median_ratio"], (
    f"the printed medians give {_printed_ratio}, which does not round to the published "
    f"{H['tier_median_ratio']} -- a reader checking this paragraph would get a different number"
)
# "at that precision they are the same number" is a claim about the rendered strings; enforce it.
# If the two two-figure renderings ever diverge, the sentence is false and the document must be
# re-worded, never written around it.
assert H["tier_median_ratio"] == H["tier_midpoint_ratio"], (
    f"the two published ratios no longer agree at two significant figures "
    f"({H['tier_median_ratio']} vs {H['tier_midpoint_ratio']}) -- re-word the paragraph"
)


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
> lambda_c band -- so the headline results in sections 1 and 2 do not depend on the v1 ODE
> network's structure (the network reduces exactly to this form in the single-pool V1 gate).
> Priors are **uniform over each input's own range** (see `openmucf/uq.py` `PARAMS`), whose
> provenance is recorded per row in `openmucf/data/uq_priors.csv` --
> maximally honest about what is actually known. No input was tuned to hit a validation target, with
> one disclosed anchor: the formation model's overall scale (`formation._CALIB`) is set to the
> room-temperature thermal rate, so the section-1c eta bracket, the one result here that reaches
> `_CALIB`, is not fully independent of it.

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
occurred in the anchor experiments (one channel, one accounting home), so eta is reported as a structural bracket beside
the CI, never convolved into it.

## 2. Propagated uncertainty (what we can actually say today)
Monte-Carlo propagation of the measured liquid-density ranges (95% intervals; prior propagation, not a
posterior). The interval deliberately reflects LIQUID conditions (phi ~ 1.2, T ~ 300 K). The
record X_mu ~ 150 (Jones 1986, a liquefied d-t target at c_t = 0.3) and the Kou-Chen best case
both lie above it, and for the same reason: each needs an effective sticking below the
omega_s0/R support the box samples. Jones reports omega_s_eff "as small as 0.35%" and still
falling with density (p.590); these priors carry no density dependence.

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

> **AMENDMENT (2026-08-19).** This section previously ran its T3 row on Uniform(2.3e3, 1e6) GeV and
> closed its Finding by calling the resulting median collapse -- then reported as ~5 orders of
> magnitude -- "the ~10^3 muon-cost gap expressed in energy-return form". Both are retracted. Neither
> old T3 edge carried recorded provenance anywhere in this document or its generator, and the box's
> support ran past the PSI HIMB figure ({_PSI.normalized_GeV_per_mu:g} GeV per mu+, at its row's
> `{_PSI.stage}` stage), which counts mu+ only and which
> the muon-cost ledger's schema bars from any muCF cost aggregate -- a prior support drawn over a tier
> being no exception. The T3 box is now the min and the max of the pinned beam-kinetic T3 rows that
> rule admits: **Uniform({H["t3_lo"]}, {H["t3_hi"]}) GeV**. Dropping the T3 support from 1e6 GeV to
> {H["t3_hi"]} GeV RAISES the T3 median Q_net, from **4.39e-07 to {H["tier_qnet_median_T3"]}**, since
> cheaper assumed muons buy more return. Its effect on what this section reports is the T1-to-T3
> FALL: about {H["tier_span_oom"]} orders of magnitude, where the retracted text said five. That
> sentence also read the fall as the muon-cost spread restated in energy-return units; it is not that
> quantity, and "What sets the panel's spread" below says what it is.
> T1 and T2 are UNCHANGED -- no barred row touches either -- and the full-text-pinned Bertin et al.
> (1987) value still sits ABOVE the T1 box, where a discrepant pin is disclosed and never tuned away.

Sections 1 and 2 use the default flat E_mu = [2, 10] GeV design-study box (UNCHANGED). To show how
Q_net responds to the assumed muon cost, the SAME seeded forward-UQ Q_net is re-run under three
tier-specific E_mu priors, with every other input (the omega_s0 / R / lambda_c / eta boxes)
held fixed. This is a sensitivity-of-Q_net-to-E_mu panel: the boxes are disclosed modelling choices,
with their provenance below. It measures no cost gap, computes no cost ratio, and makes
no same-basis comparison. (Arithmetic ON ITS OWN OUTPUTS is a different thing and is reported below;
it says something about the boxes this document chose, and nothing about the muon-cost data.)

| E_mu prior (muon-cost tier) | P(Q_net > 1) | median Q_net |
|---|---|---|
| {_box_label("T1")} | {H["tier_qnet_Pgt1_T1"]} | {H["tier_qnet_median_T1"]} |
| {_box_label("T2")} | {H["tier_qnet_Pgt1_T2"]} | {H["tier_qnet_median_T2"]} |
| {_box_label("T3")} | {H["tier_qnet_Pgt1_T3"]} | {H["tier_qnet_median_T3"]} |

**Finding.** The open-access anchor for the muon cost is Kelly, Hart & Rose (2021) at 4.70 GeV/muon
(full-text-verified; see `MUON_COST.md`). P(Q_net > 1) is {H["tier_qnet_Pgt1_T1"]} in every tier -- even the
cheapest design-study muons cap Q_net well below 1 at liquid density -- so the tier signal lives in the
MEDIAN Q_net, which falls by about {H["tier_span_oom"]} orders of magnitude from
T1 ({H["tier_qnet_median_T1"]}) to T3 ({H["tier_qnet_median_T3"]}). That fall is a property of the
E_mu boxes chosen here, NOT a measurement of the muon-cost spread: no same-basis T1-vs-T3 cost ratio
is computable from the ledger rows at all (`MUON_COST.md`), and this panel computes none.

**What sets the panel's spread.** At fixed other inputs Q_net goes as 1/E_mu, so where each box sits
governs its row. Dividing this panel's own outputs by each other: the T1-to-T3 ratio of the medians
above is **{H["tier_median_ratio"]}**, and the ratio of the two boxes' midpoints
-- {H["tier_midpoint_T1"]} GeV and {H["tier_midpoint_T3"]} GeV -- is {H["tier_midpoint_ratio"]}. Both
are quoted to two significant figures and no finer -- the medians in the table above carry three, so
a ratio quoted finer could not be reproduced by dividing the printed values -- and at that precision
they are the same number; that agreement is the point. What the panel shows is a property of the support
this document chose, not of the muon-cost data. It is a different quantity from the muon-cost
tier-median ratio reported in `MUON_COST.md` (which is itself a mixed-basis, order-of-magnitude
observation and not a same-basis ratio), and the two do not take the same value, so any resemblance
between them is a coincidence of where the boxes were drawn and never corroboration of either.

**Box-edge provenance.** A box edge is a prior-support choice for a sensitivity scan, and the
ledger's aggregate rule binds every edge -- enforced by a test rather than promised here: no
edge may be read off a row barred from muCF cost aggregates, and no box's support may CONTAIN such a
row's value. Each box below prints every edge it has, and what set it.

**T1 box edges.** {H["t1_lo"]} GeV is the Acceleron 2025 active-target slide value -- simulated,
unvalidated, a company slide. `MUON_COST.md` records that the slide-tier Acceleron row is not one
of its named headline anchors; letting it set the lower edge of a prior support is a JUDGMENT CALL,
disclosed here as one, because the edge of a sensitivity box is not a headline figure and no rule
bars the row it is read off from muCF cost aggregates. {H["t1_hi"]} GeV is a declared design-study
upper constant and is not a ledger row. The full-text-pinned Bertin et al. (1987) per-stopped-muon
cost at liquid density is ~7.8 GeV (ABOVE the upper edge), with a ~3 GeV ideal all-collected floor,
and Eliezer-Henis (1994) is ~5 GeV; the box spans the low/central design-study range, its edges are
disclosed alongside the pinned values, and it is left UNCHANGED (pre-registered; a discrepant pin is
disclosed, never tuned away).

{_box_composition("T1")}

**T2 box edges.** {H["t2_lo"]} and {H["t2_hi"]} GeV are declared decade constants bracketing the
tier's single pinned row, the muon-collider front end at 178 GeV. Neither edge is a ledger row, and
this document previously recorded no provenance for either; they are recorded now as what they are, a
bracket around a one-row tier, and left UNCHANGED.

{_box_composition("T2")}

**T3 box edges.** This box previously shipped as [2.3e3, 1e6] GeV with no recorded provenance for
either edge. It is now a pure function of the ledger: the min and the max of the pinned
`beam_kinetic` T3 rows carrying a charge basis a muCF cost aggregate admits --
{mucost.panel_t3_membership(_MU_COST)}.
That row list is rendered from the same aggregate the edges are read off, so it moves when the
ledger moves, and so does the exclusion that follows.
{mucost.panel_t3_exclusion_clause(_MU_COST)}
Two things about the surviving edges are disclosed rather than smoothed over: the upper edge comes
from a MIXED-charge row ({mucost.PANEL_ROW_LABELS[_T3_HI_ROW.source_id]} counts mu+ and mu-
together, so the mu--only cost it implies is higher by a factor this ledger does not source), and the two edge-setting rows sit
at DIFFERENT stages -- one `{_T3_LO_ROW.stage}`, which is not a point on the muCF chain at all, the
other `{_T3_HI_ROW.stage}` -- so this box spans heterogeneous accounting bases and supports a
sensitivity scan only, never a cost statement.

{_box_composition("T3")}

Replacing the flat [2, 10] default with a tiered prior is deferred to Phase-4 findings-v2.

## 3. Breakeven audit (the marquee result)
The 2026 projections (Yin-Kou-Chen arXiv:2605.26432): $N_\\mu > 500$, $Q > 2$. Under the **measured,
liquid-density (phi <= ~1.45), unpolarized** uncertainty ranges:

- **P(X_mu > 500) = {H["P_xmu_gt500"]}**, P(Q_sci > 2) = {be["P_qsci_gt2"] * 100:.1f}%,
  P(Q_net > 1) = {be["P_qnet_gt1"] * 100:.1f}%. These zeros are STRUCTURAL, not Monte-Carlo estimates:
  500 lies outside the prior box's support entirely (max supported X_mu ~ 133).
- Even at **zero sticking** the cycling rate caps the yield at lambda_c/lambda_0, so this cap is
  **condition-dependent, never universal**. At the liquid anchor ($\\lambda_c$=1.45e8) it is
  **X_mu = {H["cap_zero_sticking"]}**; at the SIN 12 K solid non-equilibrated anchor
  ($\\lambda_c$=1.93e8, ledger `lambda_c_solid_12K`) it is **X_mu = {H["cap_zero_sticking_solid"]}**.
  Those two anchors are **condition-PAIRED, not independently selectable**: the condition that bought
  the faster cycle also carried higher measured sticking (0.57% vs 0.45%), so the measured yield rose
  only 113 -> {H["yield_solid_pair"]}. Even at the +30% reproduction
  band on lambda_c the liquid cap is ~414 < 500. Density scaling (lambda_c = phi*lambda_c_tilde) at the
  demonstrated DAC phi=2.4 would lift the decay-only cap to ~530-640 *if phi-linearity holds there* --
  which is precisely the unmeasured question the MuFusE program tests.
- **What would have to be true** for $N_\\mu$=500: the (lambda_c, R) frontier runs from
  (2.28e8, R -> 1) to (3e8, R = {be["R_required_at_lambda_c_3e8"]:.2f}); and even at infinite lambda_c,
  omega_s_eff <= 0.2% i.e. **R >= {H["R_required"]}** is required (R >= {H["R_required_band"]} across the
  omega_s0-box band -- higher initial sticking needs more reactivation). That **R** is the TOTAL
  reactivation. It is NOT comparable to the model-derived collisional value R_col = {H["R_col_ref"]}
  (Kou-Chen Eq.33): the two act as SUCCESSIVE factors, omega_s_eff = omega_s0 (1-R_col)(1-R_X), so at
  R_col = {H["R_col_ref"]} the field-assisted factor alone would have to reach
  **R_X >= {H["R_X_required"]}**. Experiment pins only the product omega_s_eff ~ 0.45%, and our
  Kamimura-prior calibration posterior gives a total R = 0.46 +- 0.06.

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
- Q_net uses the transparent efficiency chain in `openmucf/uq.py`; every factor is a documented knob
  (`openmucf/data/uq_priors.csv`).
  E_mu is beam energy per muon delivered (Breunlich 1989 convention); wall-plug efficiency enters
  separately as eta_acc.
- **Muon-cost caveat (the Q_net floor).** The 2-10 GeV E_mu range is a *design-study* figure for an
  unbuilt, purpose-built muon source, and the operating facilities in the muon-cost ledger sit about
  three orders of magnitude above it per muon. That spread is a MIXED-BASIS, order-of-magnitude
  observation and never a same-basis ratio: no accounting stage is even shared between the two tiers
  (`MUON_COST.md`), and facilities optimize beam brightness and purity rather than muons-per-watt. So
  the Q_net interval above is a best-case floor conditional on such a source existing -- real-facility
  Q_net today would be far lower. (The efficiency-free Q_sci comparison to Yin-Kou-Chen is unaffected:
  it is genuinely same-basis.)
- The blanket multiplier M=1 (pure muCF); a fission/breeding hybrid (M>1) is a separate, explicit knob.

Figures: `figures/sobol.png`, `figures/forward_uq.png`, `figures/breakeven.png`.
"""

with open("FINDINGS.md", "w") as f:
    f.write(md)


# --------------------------------------------------------- machine-checkable provenance manifest
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
    _entry(
        "cap_zero_sticking_solid",
        rf"anchor\s*\n?\s*\(\$\\lambda_c\$=1\.93e8[^\n]*\n?[^\n]*\*\*X_mu = "
        rf"{re.escape(H['cap_zero_sticking_solid'])}\*\*",
    ),
    _entry("R_required", rf"\*\*R >= {re.escape(H['R_required'])}\*\*"),
    _entry("R_required_band", rf"R >= {re.escape(H['R_required_band'])} across the"),
    _entry("R_X_required", rf"\*\*R_X >= {re.escape(H['R_X_required'])}\*\*"),
    _entry("eta_bracket_lo", rf"\| 1 \(bare theory\) \| {re.escape(H['eta_bracket_lo'])} \|"),
    _entry("eta_bracket_hi", rf"\| 5 \(Yamashita-Kino fit\) \| {re.escape(H['eta_bracket_hi'])} \|"),
    _entry("eta_bracket_width", rf"X_mu\(eta=5\) - X_mu\(eta=1\) = \*\*{re.escape(H['eta_bracket_width'])}\*\*"),
]
# section 2b (the muon-cost tier panel): anchor each tracked number to its row of the Q_net-by-tier table.
_tier_rows = {_t: re.escape(_box_label(_t)) for _t in ("T1", "T2", "T3")}
for _t in ("T1", "T2", "T3"):
    _p = _tier_rows[_t]
    _entries.append(_entry(f"tier_qnet_Pgt1_{_t}", rf"{_p} \| {re.escape(H[f'tier_qnet_Pgt1_{_t}'])} \|"))
    _entries.append(
        _entry(f"tier_qnet_median_{_t}", rf"{_p}[^\n]*\| {re.escape(H[f'tier_qnet_median_{_t}'])} \|")
    )

_entries += [
    # section 2b's two published ratios and the midpoints they are read against. Tracked because the
    # sentence they sit in is the one that replaces a retracted claim: if either moved without the
    # other, the paragraph would still read as though they agreed.
    _entry("tier_span_oom", rf"falls by about {re.escape(H['tier_span_oom'])} orders of magnitude"),
    _entry(
        "tier_median_ratio",
        rf"ratio of the medians\s*\n?\s*above is \*\*{re.escape(H['tier_median_ratio'])}\*\*",
    ),
    _entry("tier_midpoint_T1", rf"-- {re.escape(H['tier_midpoint_T1'])} GeV and"),
    _entry("tier_midpoint_T3", rf"and {re.escape(H['tier_midpoint_T3'])} GeV -- is"),
    _entry(
        "tier_midpoint_ratio",
        rf"GeV -- is {re.escape(H['tier_midpoint_ratio'])}\. Both",
    ),
]
# Every box edge, anchored to its own table row. The declared constants (T1_hi, T2_lo, T2_hi) have
# no ledger row to answer for them, so the manifest is where their published values are recorded as
# provenance-tracked; the literal pin that stops one moving silently lives in
# tests/test_mucost.py::test_declared_edges_and_published_boxes_are_pinned.
for _t in ("T1", "T2", "T3"):
    _lbl = re.escape(_box_label(_t))
    _entries.append(_entry(f"{_t.lower()}_lo", _lbl))
    _entries.append(_entry(f"{_t.lower()}_hi", _lbl))

_manifest_inputs = {
    "rates_csv_sha256": provenance.file_sha256(RATES_CSV),
    "validation_targets_csv_sha256": provenance.file_sha256(TARGETS_CSV),
    # section 2b's tier boxes are read off the muon-cost ledger, so this document's bytes depend on
    # that CSV as directly as they do on the rate ledger.
    "muon_cost_csv_sha256": provenance.file_sha256(mucost.MUON_COST_CSV),
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
