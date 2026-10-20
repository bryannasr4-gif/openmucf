"""Generate MATERIALITY.md + MATERIALITY_MANIFEST.json (deterministic). Run from repo root:

    python scripts/generate_materiality.py

Content: one-at-a-time absorbing-loss-channel toggles at fixed
operating points, reported as ONE-SIDED "structural sensitivity brackets" (X_mu^with - X_mu^without)
beside the section-2 forward-UQ CI width for scale, and, at the headline operating points, the response
of the three channels switched on together in one solve beside the sum of their brackets (section 5).
None of these is an error bar, and none is ever combined into any likelihood or CI (pre-registered
combination rule: side-by-side only).

Two absorbing channels, per the loss-channel network, each toggled alone:
  * He-3 scavenging (dmu + 3He): LIVE (ledger `lambda_dhe3`=1.92e8 s^-1, Fotev et al. 2020). Toggled at
    two illustrative static helium fractions c_He in {1e-4, 1e-3}, with the tt channel off.
  * ttmu side-branch: LIVE (ledger `lambda_ttmu`, the JINR preprint's rate read in the target). Toggled
    with c_He = 0 and recapture off, beside the density-normalized reading of the same table (the
    tabled rate, `openmucf/data/side_channel_inputs.csv`), a registered systematic.

Everything here runs the v1 cycle model with the loss channel as the ONLY toggle, so "channels OFF" is
the byte-exact v1 engine and the bracket isolates the single channel's structural effect. Fully
deterministic (no MCMC/sampling), so `make audit` byte-diffs MATERIALITY.md + MATERIALITY_MANIFEST.json.

The computation lives in importable helpers (no side effects on import); file I/O + printing are guarded
behind ``main()`` so the test suite can import and assert on the operating-point/channel definitions
without regenerating the doc (mirrors scripts/generate_calibration.py).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from openmucf import cycle, provenance, sidechannels
from openmucf.rates import RATES_CSV, load_rates

# --------------------------------------------------------------------------- DECIDED spec parameters
# Operating points. OP-A is anchor-adjacent: kept IN the table for
# completeness (accounting.md exists) but flagged and EXCLUDED from every headline/summary sentence.
# (phi, T [K], c_t)
OPERATING_POINTS: dict[str, tuple[float, float, float]] = {
    "OP-A": (1.25, 300.0, 0.5),  # Kou-Chen-like, anchor-adjacent (NON-HEADLINE)
    "OP-B": (1.2, 800.0, 0.5),  # high-T
    "OP-C": (2.0, 150.0, 0.5),  # MuFusE mid
    "OP-D": (2.4, 100.0, 0.5),  # MuFusE peak
}
ANCHOR_ADJACENT = "OP-A"  # excluded from headline sentences
HEADLINE_OPS = ("OP-B", "OP-C", "OP-D")
# Two illustrative tritium-decay-accumulation levels (static per-run helium fraction).
C_HE_LEVELS: tuple[float, ...] = (1e-4, 1e-3)
TRITIUM_HALFLIFE_YR = 12.32  # 3H beta decay -> 3He ingrowth driver (12.32 yr, standard)
# d-recapture / q_1s routing levels (contested cascade ground-state fraction). f_d = (1 - c_t) * q_1s.
Q1S_LEVELS: tuple[float, ...] = (0.4, 0.7, 1.0)
# The settings of the three-channel runs of section 5: the larger illustrative helium level and the maximal
# q_1s. The sum of the one-at-a-time brackets (combined_band) reads the same two constants, so the joint
# response and the sum it is printed beside are taken at one setting.
JOINT_C_HE = 1e-3
JOINT_Q1S = 1.0

FINDINGS_MANIFEST = "FINDINGS_MANIFEST.json"


def _che_tag(c_he: float) -> str:
    """Filesystem/id-safe tag for a helium fraction, e.g. 1e-4 -> 'c1em4'."""
    return "c" + f"{c_he:.0e}".replace("-", "m").replace("+", "p").replace(".", "")


def _che_label(c_he: float) -> str:
    """Human label, e.g. 1e-4 -> '1e-4' (the exponent without its leading zeros)."""
    return re.sub(r"e([+-])0*(\d)", r"e\1\2", f"{c_he:.0e}")


def op_label(op: str) -> str:
    """ASCII table label for an operating point, e.g. 'OP-B (phi=1.2, T=800 K, c_t=0.5)'."""
    phi, T, c_t = OPERATING_POINTS[op]
    label = f"{op} (phi={phi:g}, T={T:g} K, c_t={c_t:g})"
    if op == ANCHOR_ADJACENT:
        label += " -- anchor-adjacent"
    return label


def he_brackets(rates) -> dict:
    """Compute the He-3 (dmu + 3He scavenging) structural brackets at every operating point.

    Returns ``{op: {"off": X_off, che_level: {"with": X_on, "bracket": X_on - X_off}, ...}}``.
    "off" is the channels-OFF v1 engine (byte-exact v1); "with" toggles ONLY the He-3 channel on at the
    given static c_He, with ``lambda_tt`` zeroed. The bracket is one-sided and negative (an absorbing loss
    can only reduce X_mu).
    """
    out: dict = {}
    for op, (phi, T, c_t) in OPERATING_POINTS.items():
        x_off = float(cycle.fusions_per_muon_from_conditions(rates, T, phi, c_t))
        rec = {"off": x_off}
        for c_he in C_HE_LEVELS:
            p = cycle.params_from_conditions(rates, T, phi, c_t, include_loss_channels=True, c_he=c_he)
            p["lambda_tt"] = 0.0
            x_on = float(cycle.fusions_per_muon_ode(**p))
            rec[c_he] = {"with": x_on, "bracket": x_on - x_off}
        out[op] = rec
    return out


def tt_brackets(rates) -> dict:
    """Compute the ttmu side-branch brackets at every operating point, the tt channel toggled alone
    (c_He = 0, recapture off).

    Returns ``{op: {"off": X_off, "in_target": {...}, "density_normalized": {...}}}``, each reading
    ``{"with": X_on, "bracket": X_on - X_off}``: the ledger's ``lambda_ttmu`` (in-target reading) and the
    tabled rate of ``side_channel_inputs.csv`` unscaled (density-normalized reading), both x phi x c_t.
    """
    tabled = sidechannels.load_inputs()["lambda_ttmu_table"]["value"]
    out: dict = {}
    for op, (phi, T, c_t) in OPERATING_POINTS.items():
        x_off = float(cycle.fusions_per_muon_from_conditions(rates, T, phi, c_t))
        p = cycle.params_from_conditions(rates, T, phi, c_t, include_loss_channels=True, c_he=0.0)
        x_in = float(cycle.fusions_per_muon_ode(**p))
        p["lambda_tt"] = tabled * phi * c_t
        x_dn = float(cycle.fusions_per_muon_ode(**p))
        out[op] = {"off": x_off, "in_target": {"with": x_in, "bracket": x_in - x_off},
                   "density_normalized": {"with": x_dn, "bracket": x_dn - x_off}}
    return out


def _q1s_tag(q: float) -> str:
    """Id-safe tag for a q_1s level, e.g. 0.4 -> 'q04', 1.0 -> 'q10'."""
    return f"q{int(round(q * 10)):02d}"


def q1s_brackets(rates) -> dict:
    """Compute the d-recapture / q_1s routing brackets at every operating point.

    Returns ``{op: {"off": X_off, q_1s: {"with": X_on, "bracket": X_on - X_off}, ...}}``. "off" is the
    recapture-OFF v1 engine (byte-exact v1); "with" routes a fraction ``f_d = (1 - c_t) * q_1s`` of the
    surviving-sticking flux through the dmu pool (one extra transfer that races decay), so the bracket is
    one-sided and negative. This is a re-routing, NOT an absorbing loss (the muon is not removed).
    """
    out: dict = {}
    for op, (phi, T, c_t) in OPERATING_POINTS.items():
        x_off = float(cycle.fusions_per_muon_from_conditions(rates, T, phi, c_t))
        rec = {"off": x_off}
        for q in Q1S_LEVELS:
            x_on = float(cycle.fusions_per_muon_from_conditions(rates, T, phi, c_t, q_1s=q))
            rec[q] = {"with": x_on, "bracket": x_on - x_off}
        out[op] = rec
    return out


def combined_band(he_br: dict, q1s_br: dict, tt_br: dict) -> dict:
    """The sum of the one-at-a-time brackets at the HEADLINE operating points: the ttmu side-branch
    (in-target reading), 3He scavenging at c_He = JOINT_C_HE and d-recapture at q_1s = JOINT_Q1S, each
    toggled alone. It is not the response of the three channels together (``joint_response``).

    Returns ``{op: {"tt", "he", "q1s": <brackets>, "combined": <sum>, "pct": <% of X_off>}}``.
    """
    out: dict = {}
    for op in HEADLINE_OPS:
        x_off = he_br[op]["off"]
        tt_b = tt_br[op]["in_target"]["bracket"]
        he_b = he_br[op][JOINT_C_HE]["bracket"]  # 3He at the larger illustrative helium level
        q_b = q1s_br[op][JOINT_Q1S]["bracket"]  # d-recapture at the maximal q_1s
        combined = tt_b + he_b + q_b
        out[op] = {"tt": tt_b, "he": he_b, "q1s": q_b, "combined": combined, "pct": 100.0 * combined / x_off}
    return out


def structural_band_pct(rates) -> float:
    """The largest magnitude of the summed brackets over the headline operating points, in % of X_mu (OFF)."""
    comb = combined_band(he_brackets(rates), q1s_brackets(rates), tt_brackets(rates))
    return max(abs(comb[op]["pct"]) for op in HEADLINE_OPS)


def joint_params(rates, op: str, *, tt: str | None = "in_target", c_he: float = JOINT_C_HE,
                 q_1s: float | None = JOINT_Q1S) -> dict:
    """Cycle parameters at ``op`` with the loss channels switched on together.

    ``tt`` is the ttmu reading (``"in_target"``, the ledger's ``lambda_ttmu`` x phi x c_t;
    ``"density_normalized"``, the tabled rate of ``side_channel_inputs.csv`` x phi x c_t, as
    ``tt_brackets`` builds it; ``None``, the side-branch off). ``c_he = 0`` turns 3He off and
    ``q_1s = None`` turns the d-recapture routing off.
    """
    phi, T, c_t = OPERATING_POINTS[op]
    p = cycle.params_from_conditions(rates, T, phi, c_t, include_loss_channels=True, c_he=c_he, q_1s=q_1s)
    if tt is None:
        p["lambda_tt"] = 0.0
    elif tt == "density_normalized":
        p["lambda_tt"] = sidechannels.load_inputs()["lambda_ttmu_table"]["value"] * phi * c_t
    elif tt != "in_target":
        raise ValueError(f"unknown ttmu reading {tt!r}")
    return p


def joint_response(rates) -> dict:
    """The three channels switched on together at the HEADLINE operating points, in one solve each.

    Returns ``{op: {"off", "joint", "delta", "pct", "joint_dn", "delta_dn", "pct_dn", "given": {...}}}``:
    X_mu with the channels off; X_mu with the ttmu side-branch (in-target reading), 3He at c_He =
    JOINT_C_HE and d-recapture at q_1s = JOINT_Q1S all on; its change and that change in % of X_off; the
    same with the ttmu side-branch at its density-normalized reading; and, under ``"given"``, each
    channel's effect with the other two on, X_joint minus X_mu with that channel alone switched off.
    The ``"sol"`` entry keeps the in-target joint solve, for the conservation check.
    """
    out: dict = {}
    for op in HEADLINE_OPS:
        phi, T, c_t = OPERATING_POINTS[op]
        x_off = float(cycle.fusions_per_muon_from_conditions(rates, T, phi, c_t))
        sol = cycle.solve_cycle(**joint_params(rates, op))
        x_joint = float(sol.ys[-1, 3])
        x_dn = float(cycle.fusions_per_muon_ode(**joint_params(rates, op, tt="density_normalized")))
        without = {
            "tt": float(cycle.fusions_per_muon_ode(**joint_params(rates, op, tt=None))),
            "he": float(cycle.fusions_per_muon_ode(**joint_params(rates, op, c_he=0.0))),
            "q1s": float(cycle.fusions_per_muon_ode(**joint_params(rates, op, q_1s=None))),
        }
        out[op] = {
            "off": x_off,
            "joint": x_joint,
            "delta": x_joint - x_off,
            "pct": 100.0 * (x_joint - x_off) / x_off,
            "joint_dn": x_dn,
            "delta_dn": x_dn - x_off,
            "pct_dn": 100.0 * (x_dn - x_off) / x_off,
            "given": {k: x_joint - v for k, v in without.items()},
            "sol": sol,
        }
    return out


def read_forward_uq_ci(manifest_path=FINDINGS_MANIFEST) -> tuple[int, int]:
    """Read the section-2 forward-UQ 95% CI on X_mu (lo, hi) from the committed FINDINGS manifest.

    FINDINGS_MANIFEST.json is deterministic and byte-stable across environments, so this is an
    order-independent read (the CI is a fixed scale reference, not recomputed here).
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    vals = {e["id"]: e["value"] for e in manifest["entries"]}
    return (int(vals["xmu_ci_lo"]), int(vals["xmu_ci_hi"]))


def ingrowth_pct_per_month(halflife_yr: float = TRITIUM_HALFLIFE_YR) -> float:
    """Fractional 3He ingrowth per month from tritium decay: ln2/halflife, expressed as %/month."""
    return 100.0 * (math.log(2.0) / halflife_yr) / 12.0


def build_headline(rates) -> dict:
    """Assemble the single-source-of-truth formatted strings shared by the doc and the manifest."""
    br = he_brackets(rates)
    q1s_br = q1s_brackets(rates)
    tt_br = tt_brackets(rates)
    comb = combined_band(br, q1s_br, tt_br)
    jr = joint_response(rates)
    ci_lo, ci_hi = read_forward_uq_ci()

    H: dict[str, str] = {}
    for op in OPERATING_POINTS:
        H[f"{op}_off"] = f"{br[op]['off']:.3f}"
        for c_he in C_HE_LEVELS:
            tag = _che_tag(c_he)
            H[f"{op}_with_{tag}"] = f"{br[op][c_he]['with']:.3f}"
            H[f"{op}_bracket_{tag}"] = f"{br[op][c_he]['bracket']:+.3f}"
        for q in Q1S_LEVELS:
            qt = _q1s_tag(q)
            H[f"{op}_{qt}_with"] = f"{q1s_br[op][q]['with']:.3f}"
            H[f"{op}_{qt}_bracket"] = f"{q1s_br[op][q]['bracket']:+.3f}"
        for rd in ("in_target", "density_normalized"):
            H[f"{op}_tt_{rd}_with"] = f"{tt_br[op][rd]['with']:.3f}"
            H[f"{op}_tt_{rd}_bracket"] = f"{tt_br[op][rd]['bracket']:+.3f}"
    H["joint_c_he"] = _che_label(JOINT_C_HE)
    H["joint_q1s"] = f"{JOINT_Q1S:.1f}"
    for op in HEADLINE_OPS:
        # section 5, first table: the three channels together beside the sum of their brackets
        H[f"{op}_joint_with"] = f"{jr[op]['joint']:.3f}"
        H[f"{op}_joint"] = f"{jr[op]['delta']:+.3f}"
        H[f"{op}_joint_pct"] = f"{jr[op]['pct']:+.2f}"
        H[f"{op}_joint_mag"] = f"{abs(jr[op]['pct']):.2f}"
        H[f"{op}_comb_total"] = f"{comb[op]['combined']:+.3f}"
        H[f"{op}_comb_pct"] = f"{comb[op]['pct']:+.2f}"
        H[f"{op}_joint_dn"] = f"{jr[op]['delta_dn']:+.3f}"
        H[f"{op}_joint_dn_pct"] = f"{jr[op]['pct_dn']:+.2f}"
        # section 5, second table: each channel alone (its bracket) and with the other two on
        H[f"{op}_alone_tt"] = f"{comb[op]['tt']:+.3f}"
        H[f"{op}_alone_q1s"] = f"{comb[op]['q1s']:+.3f}"
        H[f"{op}_alone_he"] = f"{comb[op]['he']:+.3f}"
        for ch in ("tt", "q1s", "he"):
            H[f"{op}_given_{ch}"] = f"{jr[op]['given'][ch]:+.3f}"
    # the extremes the section-5 and section-6 sentences state (magnitudes; every value is one-sided <= 0)
    jmin = min(HEADLINE_OPS, key=lambda op: abs(jr[op]["pct"]))
    jmax = max(HEADLINE_OPS, key=lambda op: abs(jr[op]["pct"]))
    H["joint_pct_min"], H["joint_pct_min_op"] = f"{abs(jr[jmin]['pct']):.2f}", jmin
    H["joint_pct_max"], H["joint_pct_max_op"] = f"{abs(jr[jmax]['pct']):.2f}", jmax
    H["comb_worst_pct2"] = f"{max(abs(comb[op]['pct']) for op in HEADLINE_OPS):.2f}"
    he_given = [abs(jr[op]["given"]["he"]) for op in HEADLINE_OPS]
    he_given_pct = [100.0 * abs(jr[op]["given"]["he"]) / jr[op]["off"] for op in HEADLINE_OPS]
    H["he_given_min"], H["he_given_max"] = f"{min(he_given):.3f}", f"{max(he_given):.3f}"
    H["he_given_pct_min"], H["he_given_pct_max"] = f"{min(he_given_pct):.2f}", f"{max(he_given_pct):.2f}"
    H["ci_lo"] = f"{ci_lo}"
    H["ci_hi"] = f"{ci_hi}"
    H["ci_width"] = f"{ci_hi - ci_lo}"
    # 3He toggled alone: section 2 states the largest |bracket| over every operating point and both helium
    # levels; sections 5 and 6 the largest over the headline points at c_He = JOINT_C_HE.
    he_all = max(abs(br[op][c]["bracket"]) for op in OPERATING_POINTS for c in C_HE_LEVELS)
    he_head = max(abs(br[op][JOINT_C_HE]["bracket"]) for op in HEADLINE_OPS)
    H["he_alone_max"] = f"{he_all:.3f}"
    H["he_alone_ci_pct"] = f"{100.0 * he_all / (ci_hi - ci_lo):.2f}"
    H["he_head_max"] = f"{he_head:.3f}"
    H["he_head_ci_pct"] = f"{100.0 * he_head / (ci_hi - ci_lo):.2f}"
    H["he_alone_xmu_pct"] = (
        f"{max(100.0 * abs(br[op][JOINT_C_HE]['bracket']) / br[op]['off'] for op in HEADLINE_OPS):.2f}"
    )
    H["ingrowth_pct_month"] = f"{ingrowth_pct_per_month():.2f}"
    H["lambda_ttmu"] = f"{rates.value('lambda_ttmu'):.4g}"  # the ledger value the section-3 prose restates
    H["omega_tt"] = f"{rates.value('omega_tt')}"  # the ledger value the section-3 prose restates
    tt_worst = min(HEADLINE_OPS, key=lambda op: tt_br[op]["in_target"]["bracket"])
    H["tt_max"] = f"{tt_br[tt_worst]['in_target']['bracket']:+.3f}"
    return H


# ------------------------------------------------------------------------------------- MATERIALITY.md
def _he_table(H: dict) -> str:
    head = (
        "| operating point | X_mu (channels OFF) | X_mu (3He, c_He=1e-4) | bracket "
        "| X_mu (3He, c_He=1e-3) | bracket |\n|---|---|---|---|---|---|\n"
    )
    lo, hi = _che_tag(1e-4), _che_tag(1e-3)
    rows = []
    for op in OPERATING_POINTS:
        rows.append(
            f"| {op_label(op)} | {H[f'{op}_off']} | {H[f'{op}_with_{lo}']} | {H[f'{op}_bracket_{lo}']} "
            f"| {H[f'{op}_with_{hi}']} | {H[f'{op}_bracket_{hi}']} |"
        )
    return head + "\n".join(rows)


def _tt_table(H: dict) -> str:
    head = (
        "| operating point | X_mu (channels OFF) | X_mu (tt, in-target reading) | bracket "
        "| X_mu (tt, density-normalized reading) | bracket |\n|---|---|---|---|---|---|\n"
    )
    rows = []
    for op in OPERATING_POINTS:
        rows.append(
            f"| {op_label(op)} | {H[f'{op}_off']} | {H[f'{op}_tt_in_target_with']} "
            f"| {H[f'{op}_tt_in_target_bracket']} | {H[f'{op}_tt_density_normalized_with']} "
            f"| {H[f'{op}_tt_density_normalized_bracket']} |"
        )
    return head + "\n".join(rows)


def _q1s_table(H: dict) -> str:
    head = (
        "| operating point | X_mu (recapture OFF) | X_mu (q_1s=0.4) | bracket | X_mu (q_1s=0.7) "
        "| bracket | X_mu (q_1s=1.0) | bracket |\n|---|---|---|---|---|---|---|---|\n"
    )
    q04, q07, q10 = _q1s_tag(0.4), _q1s_tag(0.7), _q1s_tag(1.0)
    rows = []
    for op in OPERATING_POINTS:
        rows.append(
            f"| {op_label(op)} | {H[f'{op}_off']} | {H[f'{op}_{q04}_with']} | {H[f'{op}_{q04}_bracket']} "
            f"| {H[f'{op}_{q07}_with']} | {H[f'{op}_{q07}_bracket']} "
            f"| {H[f'{op}_{q10}_with']} | {H[f'{op}_{q10}_bracket']} |"
        )
    return head + "\n".join(rows)


#: The header cells after "operating point" that open each section-5 table; ``_cell_pattern`` anchors a
#: manifest value to the table whose header starts with them.
JOINT_HEAD = "X_mu (channels OFF) | X_mu (all three on)"
GIVEN_HEAD = "ttmu alone"


def _joint_table(H: dict) -> str:
    head = (
        f"| operating point | {JOINT_HEAD} | joint response | % of X_mu(OFF) "
        "| sum of the brackets | % of X_mu(OFF) | joint, density-normalized ttmu | % of X_mu(OFF) |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for op in HEADLINE_OPS:
        rows.append(
            f"| {op_label(op)} | {H[f'{op}_off']} | {H[f'{op}_joint_with']} | {H[f'{op}_joint']} "
            f"| {H[f'{op}_joint_pct']}% | {H[f'{op}_comb_total']} | {H[f'{op}_comb_pct']}% "
            f"| {H[f'{op}_joint_dn']} | {H[f'{op}_joint_dn_pct']}% |"
        )
    return head + "\n".join(rows)


def _given_table(H: dict) -> str:
    head = (
        f"| operating point | {GIVEN_HEAD} | ttmu, other two on | d-recapture alone | d-recapture, other two on "
        "| 3He alone | 3He, other two on |\n|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for op in HEADLINE_OPS:
        rows.append(
            f"| {op_label(op)} | {H[f'{op}_alone_tt']} | {H[f'{op}_given_tt']} | {H[f'{op}_alone_q1s']} "
            f"| {H[f'{op}_given_q1s']} | {H[f'{op}_alone_he']} | {H[f'{op}_given_he']} |"
        )
    return head + "\n".join(rows)


def build_markdown(H: dict) -> str:
    return f"""# MATERIALITY.md -- structural sensitivity brackets (auto-generated by `scripts/generate_materiality.py`)

> Structural sensitivity brackets: one-sided channel toggles, each channel alone, and the response of \
the three channels switched on together at the headline operating points (section 5). These are NOT error bars \
and are never combined into the parametric CI; the anchor-adjacent row is listed for completeness only \
(see docs/accounting.md).

## 1. What this is (and the combination rule)
Each bracket is `X_mu^with - X_mu^without` for ONE absorbing loss channel toggled on at a fixed operating
point, with the v1 cycle model otherwise unchanged ("without" == the byte-exact v1 engine). They are
**structural sensitivity brackets**, not a model-form error budget and not error bars: the pre-registered
combination rule is **side-by-side only** -- a bracket is NEVER convolved into any likelihood, posterior,
or the section-2 forward-UQ CI. A loss channel can only remove muons, so every live bracket is one-sided
(<= 0).

Operating points (fixed conditions; c_t=0.5 throughout):

- **OP-A** (phi=1.25, T=300 K) -- Kou-Chen-like, **anchor-adjacent**: kept in the tables for completeness
  but EXCLUDED from every headline sentence (its conditions are those the formation model's thermal
  scale is set at: openmucf.formation.CALIB_ANCHOR).
- **OP-B** (phi=1.2, T=800 K) -- high-T; **OP-C** (phi=2.0, T=150 K) -- MuFusE mid;
  **OP-D** (phi=2.4, T=100 K) -- MuFusE peak.

Channels covered: the two absorbing loss channels in the v1.1 network (ttmu side-branch; 3He scavenging),
plus the per-cycle d-recapture / q_1s routing (section 4 -- a re-routing, not an absorbing loss: the freed
muon detours through the dmu pool and races decay one extra transfer). The ddmu / d-d branch and the
epithermal-formation (eta) enhancement remain OUT of scope here: the ddmu channel does not exist in the
engine yet (documented -5..-15% headroom, see docs/accounting.md), and eta is already reported as its own
structural bracket in FINDINGS.md section 1c (one channel, one accounting home).

Section 5 also runs the three channels together at the headline operating points. They act in one network, so
their joint response is not the sum of their brackets; section 5 gives both, and each channel's effect alone and
with the other two on.

## 2. 3He scavenging channel (LIVE): dmu + 3He
The dmu + 3He scavenging rate `lambda_dhe3` = 1.92e8 s^-1 (Fotev et al. 2020, arXiv:2001.09927; open) is
live in the ledger; it acts on the dmu pool and is engaged at a STATIC per-run helium fraction c_He
(never time-evolved inside the single-muon ODE). 3He grows in from tritium beta decay at
ln(2) / {TRITIUM_HALFLIFE_YR:g} yr = {100.0 * math.log(2.0) / TRITIUM_HALFLIFE_YR:.4f}%/yr, i.e.
**{H["ingrowth_pct_month"]}%/month** of the tritium inventory; the two levels below (c_He = 1e-4 and 1e-3,
~0.01% and ~0.1% helium) are illustrative low accumulations bracketing the first days-to-weeks of a sealed
DT fill -- both well under one month's ingrowth.

{_he_table(H)}

For scale: the FINDINGS.md section-2 forward-UQ 95% CI on X_mu (liquid box) spans
[{H["ci_lo"]}, {H["ci_hi"]}] -- a width of **{H["ci_width"]}** X_mu units. Every 3He bracket above is at most {H["he_alone_max"]} in magnitude, {H["he_alone_ci_pct"]}% of
that parametric CI width. These brackets toggle 3He with the d-recapture routing off: the engine then returns
every freed muon straight to the tmu pool, so a muon is in the dmu pool, where 3He acts, only until its
transfer to tritium. With the routing of section 4 on, a share (1 - c_t) q_1s of the freed muons re-enters the
dmu pool every cycle; section 5 gives the 3He channel's effect with the other two channels on. Either way the
channel is reported beside the CI and never folded into it.

## 3. ttmu side-branch channel (LIVE)
The ttmu formation rate `lambda_ttmu` = {H["lambda_ttmu"]} s^-1 (JINR preprint E15-2008-99, Table 1, read as the rate in
the target; rate ledger) and `omega_tt` = {H["omega_tt"]} are live. Each bracket toggles the tt channel alone (c_He = 0,
recapture off). The density-normalized reading of the same table, which the preprint's systematic (a) supports, is
a registered systematic and is shown beside it. lambda_ttmu(T) is not modelled: every bracket applies the rate
measured in liquid T2 at 22.5 K unchanged at 100-800 K. The v1 engine recycles a freed muon straight to the tmu
pool, which lengthens the tmu dwell and so leans these brackets large.

{_tt_table(H)}

## 4. d-recapture / q_1s routing bracket (LIVE)
The freed muon that survives sticking is, in v1, recycled straight back to the tmu pool (3/4 F=1, 1/4 F=0).
The per-cycle **d-recapture** correction routes a fraction `f_d = (1 - c_t) * q_1s` of that surviving flux
through the **dmu** pool instead (the muon recaptures on deuterium and must transfer again, racing decay
one extra step); `q_1s` is the contested cascade ground-state fraction. This is a re-routing, not an
absorbing loss -- but it lowers X_mu, so the bracket is one-sided and negative. It is now explicit in
`cycle.py` (`params_from_conditions(q_1s=...)`); the `_CALIB` unfolding that would re-attribute this leg
against the 300 K anchor stays acquisition-gated (docs/accounting.md).

{_q1s_table(H)}

At the anchor-condition (300 K, 1.2 phi, c_t=0.5) MODEL_SPEC section-8 estimated this leg at -6% .. -14%
for q_1s=0.4..1.0; the explicit computation confirms it to rounding (-5.96% .. -13.68%; see the MODEL_SPEC
dated note). Above the compressed-gas onset the formation model is a placeholder (RED tier), so OP-C/OP-D
q_1s brackets share the off-anchor caveat of every off-anchor number here.

## 5. The three channels together (headline operating points)
The ttmu side-branch (in-target reading), 3He scavenging (c_He={H["joint_c_he"]}) and d-recapture (q_1s={H["joint_q1s"]}) are switched
on together in one run of the v1 cycle model. One table gives that joint response beside the sum of the
three brackets of sections 2-4, each toggled alone, and the joint response with the ttmu side-branch at its
density-normalized reading (section 3); the other gives each channel's effect alone (its bracket) and with the
other two on (X_mu with all three on minus X_mu with the other two).

{_joint_table(H)}

{_given_table(H)}

Toggled one at a time, the sum is dominated by the ttmu side-branch and by d-recapture at the maximal q_1s, and
3He alone removes at most {H["he_alone_xmu_pct"]}% of X_mu at these helium levels. The channels interact through the pools they
share: with the other two on, 3He lowers X_mu by {H["he_given_min"]} to {H["he_given_max"]} X_mu units ({H["he_given_pct_min"]}% to {H["he_given_pct_max"]}%) at these points,
because the d-recapture routing returns freed muons to the dmu pool, where 3He acts. The joint response at the
in-target reading runs from {H["joint_pct_min"]}% ({H["joint_pct_min_op"]}) to {H["joint_pct_max"]}% ({H["joint_pct_max_op"]}) of X_mu, above the ~10-15% one-sided structural
headroom named in MODEL_SPEC section 8; the one-at-a-time brackets sum to at most {H["comb_worst_pct2"]}%. These responses are
reported beside the parametric CI, never convolved into it.

## 6. Summary (headline operating points OP-B / OP-C / OP-D only)
Across the headline operating points OP-B (high-T), OP-C (MuFusE mid) and OP-D (MuFusE peak), the 3He
scavenging channel toggled alone contributes a one-sided structural bracket of at most {H["he_head_max"]} X_mu units in
magnitude at 0.1% helium -- {H["he_head_ci_pct"]}% of the section-2 forward-UQ CI width ({H["ci_width"]}) -- and smaller still at 0.01%
helium; with the d-recapture routing and the ttmu side-branch on, it lowers X_mu by {H["he_given_min"]} to {H["he_given_max"]} X_mu units
(section 5). Switched on together, the ttmu side-branch, the per-cycle d-recapture routing (section 4) and 3He
lower X_mu by {H["OP-B_joint_mag"]}%, {H["OP-C_joint_mag"]}% and {H["OP-D_joint_mag"]}% at OP-B, OP-C and OP-D (section 5), one-sided downward, where their
one-at-a-time brackets sum to at most {H["comb_worst_pct2"]}%. The ttmu side-branch alone, at its in-target reading, adds up to
{H["tt_max"]} X_mu at the headline points (section 3). The anchor-adjacent OP-A row is listed for completeness only
and carries no headline claim (docs/accounting.md). No bracket on this page is combined into any CI or
likelihood.
"""


# ------------------------------------------------------------- machine-checkable provenance (MANIFEST)
def build_manifest_entries(H: dict) -> list:
    def _entry(entry_id, pattern, key=None):
        return provenance.ManifestEntry(
            id=entry_id,
            value=H[key or entry_id],
            pattern=pattern,
            source_type="derivation",
            source="scripts/generate_materiality.py",
            doc="MATERIALITY.md",
        )

    def _cell(entry_id, section, head, k, suffix=""):
        return _entry(entry_id, _cell_pattern(section, head, entry_id.split("_", 1)[0], k, H[entry_id] + suffix))

    entries = []
    lo, hi = _che_tag(1e-4), _che_tag(1e-3)
    off_head = "X_mu (channels OFF)"
    for op in OPERATING_POINTS:
        # Every per-row table value is anchored to its section, its table, its operating point's row and its
        # column, so a value repeated in another table (section 5 repeats brackets of sections 2-4) cannot
        # stand in for it. Each channels-OFF value is anchored to the section-2 table.
        entries.append(_cell(f"{op}_off", 2, off_head, 0))
        entries.append(_cell(f"{op}_with_{lo}", 2, off_head, 1))
        entries.append(_cell(f"{op}_bracket_{lo}", 2, off_head, 2))
        entries.append(_cell(f"{op}_with_{hi}", 2, off_head, 3))
        entries.append(_cell(f"{op}_bracket_{hi}", 2, off_head, 4))
        for k, rd in enumerate(("in_target", "density_normalized")):  # ttmu side-branch (section 3)
            entries.append(_cell(f"{op}_tt_{rd}_with", 3, off_head, 1 + 2 * k))
            entries.append(_cell(f"{op}_tt_{rd}_bracket", 3, off_head, 2 + 2 * k))
        for k, q in enumerate(Q1S_LEVELS):  # d-recapture / q_1s bracket (section 4)
            qt = _q1s_tag(q)
            entries.append(_cell(f"{op}_{qt}_with", 4, "X_mu (recapture OFF)", 1 + 2 * k))
            entries.append(_cell(f"{op}_{qt}_bracket", 4, "X_mu (recapture OFF)", 2 + 2 * k))
    for op in HEADLINE_OPS:  # the three channels together (section 5)
        for k, (key, suffix) in enumerate((("joint_with", ""), ("joint", ""), ("joint_pct", "%"),
                                           ("comb_total", ""), ("comb_pct", "%"), ("joint_dn", ""),
                                           ("joint_dn_pct", "%")), start=1):
            entries.append(_cell(f"{op}_{key}", 5, JOINT_HEAD, k, suffix))
        for k, key in enumerate(("alone_tt", "given_tt", "alone_q1s", "given_q1s", "alone_he", "given_he")):
            entries.append(_cell(f"{op}_{key}", 5, GIVEN_HEAD, k))
    e = re.escape
    entries += [
        _entry("ci_lo", rf"spans\s*\[{e(H['ci_lo'])}, {e(H['ci_hi'])}\]"),
        _entry("ci_hi", rf"spans\s*\[{e(H['ci_lo'])}, {e(H['ci_hi'])}\]"),
        _entry("ci_width", rf"width of \*\*{e(H['ci_width'])}\*\* X_mu units"),
        _entry("ingrowth_pct_month", rf"\*\*{e(H['ingrowth_pct_month'])}%/month\*\*"),
        # section 2: 3He toggled alone, against the CI width
        _entry("he_alone_max", rf"Every 3He bracket above is at most {e(H['he_alone_max'])} in magnitude"),
        _entry("he_alone_ci_pct", rf"in magnitude, {e(H['he_alone_ci_pct'])}%\s+of\s+that parametric CI width"),
        # section 5
        _entry("joint_c_he", rf"3He scavenging \(c_He={e(H['joint_c_he'])}\) and d-recapture"),
        _entry("joint_q1s", rf"and d-recapture \(q_1s={e(H['joint_q1s'])}\) are switched"),
        _entry("he_alone_xmu_pct", rf"3He alone removes at most {e(H['he_alone_xmu_pct'])}% of X_mu"),
        _entry("he_given_min", rf"3He lowers X_mu by {e(H['he_given_min'])} to {e(H['he_given_max'])} X_mu units \("),
        _entry("he_given_max", rf"3He lowers X_mu by {e(H['he_given_min'])} to {e(H['he_given_max'])} X_mu units \("),
        _entry("he_given_pct_min", rf"X_mu units \({e(H['he_given_pct_min'])}% to {e(H['he_given_pct_max'])}%\)"),
        _entry("he_given_pct_max", rf"X_mu units \({e(H['he_given_pct_min'])}% to {e(H['he_given_pct_max'])}%\)"),
        _entry("joint_pct_min", rf"runs from {e(H['joint_pct_min'])}% \({e(H['joint_pct_min_op'])}\) to"),
        _entry("joint_pct_min_op", rf"runs from {e(H['joint_pct_min'])}% \({e(H['joint_pct_min_op'])}\) to"),
        _entry("joint_pct_max", rf"\) to {e(H['joint_pct_max'])}% \({e(H['joint_pct_max_op'])}\) of X_mu"),
        _entry("joint_pct_max_op", rf"\) to {e(H['joint_pct_max'])}% \({e(H['joint_pct_max_op'])}\) of X_mu"),
        _entry("comb_worst_pct2", rf"MODEL_SPEC section 8; the one-at-a-time brackets sum to at most "
                                  rf"{e(H['comb_worst_pct2'])}%\."),
        # section 6
        _entry("he_head_max", rf"structural bracket of at most {e(H['he_head_max'])} X_mu units in"),
        _entry("he_head_ci_pct", rf"helium -- {e(H['he_head_ci_pct'])}% of the section-2 forward-UQ CI width"),
        _entry("tt_max", rf"in-target reading, adds up to\s+{e(H['tt_max'])} X_mu at the headline points"),
    ]
    summary = (rf"lower X_mu by {e(H['OP-B_joint_mag'])}%, {e(H['OP-C_joint_mag'])}% and "
               rf"{e(H['OP-D_joint_mag'])}% at OP-B, OP-C and OP-D \(section 5\)")
    entries += [_entry(f"{op}_joint_mag", summary) for op in HEADLINE_OPS]
    entries.append(_entry(
        "comb_worst_pct2_summary",
        rf"one-at-a-time brackets sum to at most {e(H['comb_worst_pct2'])}%\. The ttmu side-branch alone",
        key="comb_worst_pct2",
    ))
    return entries


def _cell_pattern(section: int, head: str, op: str, k: int, value: str) -> str:
    """A pattern that matches ``value`` only as cell ``k`` (0-based, counted after the operating-point cell) of
    ``op``'s row, in the table of section ``section`` whose header cells after "operating point" start with
    ``head``. The skip over the table's rows consumes table lines only, so it cannot leave the table."""
    return (
        rf"\n## {section}\. [^\n]*\n(?:(?!\n## )[\s\S])*?\n\| operating point \| {re.escape(head)} \|[^\n]*\n"
        rf"(?:\|[^\n]*\n)*?\| {re.escape(op)} [^|\n]*\|(?:[^|\n]*\|){{{k}}} {re.escape(value)} \|"
    )


def main() -> None:
    rates = load_rates()
    H = build_headline(rates)
    Path("MATERIALITY.md").write_text(build_markdown(H), encoding="utf-8")
    entries = build_manifest_entries(H)
    inputs = {
        "rates_csv_sha256": provenance.file_sha256(RATES_CSV),
        "findings_manifest_sha256": provenance.file_sha256(FINDINGS_MANIFEST),
    }
    provenance.write_manifest(
        "MATERIALITY_MANIFEST.json", entries, inputs, generated_by="scripts/generate_materiality.py"
    )
    hi_tag = _che_tag(1e-3)
    br_summary = {op: H[f"{op}_bracket_{hi_tag}"] for op in OPERATING_POINTS}
    print(f"wrote MATERIALITY.md + MATERIALITY_MANIFEST.json ({len(entries)} entries)")
    print(f"3He brackets @ c_He=1e-3: {br_summary}")
    print(f"ttmu brackets (in-target): { {op: H[f'{op}_tt_in_target_bracket'] for op in OPERATING_POINTS} }")
    print(f"forward-UQ CI width for scale = {H['ci_width']} (X_mu [{H['ci_lo']}, {H['ci_hi']}])")


if __name__ == "__main__":
    main()
