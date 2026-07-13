"""openmucf.validate -- reproduce the pre-registered validation targets (Phase 2.4 trust gate).

Loads each target's observed value and pre-registered tolerance from
``openmucf/data/validation_targets.csv`` and runs the engine against them, reporting per target
predicted vs observed within tolerance. Mutating a CSV target value/tolerance changes the verdict
(see ``tests/test_validate.py``). Honest by construction: targets the v1 model cannot yet hit
(e.g. the solid-phase condensed-matter trend) are marked DEFERRED, not silently passed.

Every result carries a ``category`` tier (see ``CATEGORIES``) so the scoreboard is falsifiable rather
than flattering: only rows marked ``independent`` are genuine predictions. As of v1 the passing set
contains no ``independent`` row -- three ``independent`` targets run and FAIL by design (registered,
pre-framed placeholder-distance findings); a PASS on any of them is a bug or a tolerance error, not a
success (see ``tests/test_validate.py::test_expected_fail_guard``).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from . import cycle
from .rates import TARGETS_CSV, omega_fraction

# Canonical liquid operating point for sticking-controlled checks.
_OP = dict(T=300.0, phi=1.2, c_t=0.5)

# Digitized Yamashita-Kino 2022 Fig.3a lambda_c(T) (c_t=0.5 EVM-SPM-FIF band centreline), the sourced
# temperature-shape comparator for V_yamashita_ratio / V_yamashita_curve.
_YAMASHITA_CSV = Path(__file__).resolve().parent / "data" / "yamashita_kino_lc_T.csv"

# Validation tiers, weakest-claim to strongest. Only `independent` rows are genuine predictions;
# the rest test self-consistency, a fed reproduction, an inserted anchor, or a calibrated-model shape.
CATEGORIES = (
    "self-consistency",
    "reproduction (fed input)",
    "anchor-consistency",
    "shape (calibrated model)",
    "independent",
)


@dataclass
class Result:
    target_id: str
    observed: str
    predicted: float
    tolerance: str
    passed: bool | None  # None == DEFERRED (honest, not a pass)
    note: str
    category: str  # one of CATEGORIES (test-pinned per-id map)
    dedup_group: str = ""  # rows sharing a non-empty group count once in "distinct tests"


def _xmu(rates, omega_s_eff, T=_OP["T"], include_loss_channels=False):
    return float(
        cycle.fusions_per_muon_from_conditions(
            rates,
            T,
            _OP["phi"],
            _OP["c_t"],
            omega_s_eff=omega_s_eff,
            include_loss_channels=include_loss_channels,
        )
    )


def _load_targets(path=TARGETS_CSV):
    """Load pre-registered targets (observed value + tolerance) from validation_targets.csv, keyed by id."""
    with open(path, newline="") as f:
        return {row["target_id"].strip(): row for row in csv.DictReader(f)}


def _load_yamashita_curve(path=_YAMASHITA_CSV):
    """Digitized Yamashita-Kino 2022 Fig.3a lambda_c(T) [s^-1], keyed by T [K] (band centreline)."""
    with open(path, newline="") as f:
        return {int(r["T_K"]): float(r["lambda_c_s^-1"]) for r in csv.DictReader(f)}


def _within(pred, value, tol):
    """Pass/fail from a CSV tolerance string: interval '[a,b]' or relative '+-N%'."""
    m_iv = re.match(r"\s*\[([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\]", tol)
    if m_iv:
        return float(m_iv.group(1)) <= pred <= float(m_iv.group(2))
    m_pct = re.search(r"([\d.]+)\s*%", tol)
    if m_pct:
        return abs(pred - value) / abs(value) < float(m_pct.group(1)) / 100.0
    raise ValueError(f"unparseable tolerance {tol!r}")


def run(rates, targets_csv=None, channels="off"):
    """Run the engine against the pre-registered targets.

    ``channels`` in {"off","on"}: "off" is the v1 trust gate (committed VALIDATION.md); "on" switches
    ``include_loss_channels=True`` at the SAME operating points. At the anchor conditions the He-3 channel
    is inert (He-purged fills, c_He=0) and -- while the ttmu formation rate is blocked (lambda_ttmu=0.0) --
    the tt channel is a no-op, so the channels-on scoreboard currently reproduces channels-off exactly.
    Once lambda_ttmu is acquired the Kou-Chen +-10% PASSes may convert to documented discrepancies (a
    FINDING about the tt channel's share, not a regression) -- see the doc header and accounting.md.
    """
    inc = channels == "on"
    tgt = _load_targets(targets_csv or TARGETS_CSV)
    out = []

    x = _xmu(rates, 0.00557, include_loss_channels=inc)
    out.append(
        Result(
            "V_kouchen_base",
            "112.6 (fed omega_s_eff=0.557%)",
            x,
            "+-10%",
            _within(x, float(tgt["V_kouchen_base"]["value"]), tgt["V_kouchen_base"]["tolerance"]),
            "collision-only baseline",
            "reproduction (fed input)",
        )
    )

    x = _xmu(rates, 0.00308, include_loss_channels=inc)
    out.append(
        Result(
            "V_kouchen_best",
            "156.5 (fed omega_s_eff=0.308%)",
            x,
            "+-10%",
            _within(x, float(tgt["V_kouchen_best"]["value"]), tgt["V_kouchen_best"]["tolerance"]),
            "best external-field scenario",
            "reproduction (fed input)",
        )
    )

    x = _xmu(rates, 0.0045, include_loss_channels=inc)
    out.append(
        Result(
            "V_petitjean",
            "113 (fed omega_s_eff=0.45%)",
            x,
            "[100,150]",
            _within(x, float(tgt["V_petitjean_Xmu"]["value"]), tgt["V_petitjean_Xmu"]["tolerance"]),
            "measured liquid effective sticking",
            "reproduction (fed input)",
        )
    )

    xs = [_xmu(rates, 0.00557, T=T, include_loss_channels=inc) for T in (200.0, 400.0, 600.0, 800.0)]
    out.append(
        Result(
            "V_yamashita_lcT",
            "monotone X_mu(T) rise",
            xs[-1],
            "monotone",
            all(b > a for a, b in zip(xs, xs[1:], strict=False)),
            f"X_mu(200,400,600,800 K)={[round(v, 1) for v in xs]}. The three Yamashita rows "
            "(lcT monotonicity + ratio/curve vs the digitized full curve) are one test "
            "(the same lambda_c(T)/X_mu(T) inversion); counted as a single shape check in the summary.",
            "shape (calibrated model)",
            "yamashita_shape",
        )
    )

    x = _xmu(rates, 0.00557, include_loss_channels=inc)
    lam0 = rates.value("lambda_mu_decay")
    implied_lc = lam0 / (1.0 / x - 0.00557)
    _bl = tgt["V_breunlich_lambdac"]
    out.append(
        Result(
            "V_breunlich_lambdac",
            "1.45e8 s^-1 (max measured cycling rate, liquid; Breunlich 1989)",
            implied_lc,
            "+-30%",
            _within(implied_lc, float(_bl["value"]), _bl["tolerance"]),
            "engine-implied cycling rate at (300 K, 1.2 phi, c_t=0.5), inverted from the closed form; "
            "inherits the formation-scale anchor (see gate rule)",
            "anchor-consistency",
        )
    )

    # Pre-registered ratio clause: engine-implied lambda_c(800 K)/lambda_c(300 K) vs the digitized
    # Yamashita-Kino FULL-CURVE centreline ratio, +-30%. Re-anchored 2026-07-13: the comparator is the
    # sourced full-curve digitized value (2.358; band [2.09, 2.62], solid-line 2.235), NOT the earlier
    # ~1.45 under-read. The 300 K scale is anchor-fixed (formation._CALIB), so this tests the SHAPE of the
    # lambda_c(T) rise; the corrected target is strictly harder and the engine (~1.31) FAILs it.
    def _implied_lc(xval):
        return lam0 / (1.0 / xval - 0.00557)

    yk = _load_yamashita_curve()
    x300 = _xmu(rates, 0.00557, T=300.0, include_loss_channels=inc)
    x800 = _xmu(rates, 0.00557, T=800.0, include_loss_channels=inc)
    lc300 = _implied_lc(x300)
    ratio = _implied_lc(x800) / lc300
    _yr = tgt["V_yamashita_ratio"]
    out.append(
        Result(
            "V_yamashita_ratio",
            f"{float(_yr['value']):.3f} (digitized full-curve lambda_c(800 K)/lambda_c(300 K) centreline; "
            "Yamashita-Kino Fig.3a, data/yamashita_kino_lc_T.csv)",
            ratio,
            "+-30%",
            _within(ratio, float(_yr["value"]), _yr["tolerance"]),
            "engine ratio of implied lambda_c at 800 K vs 300 K (same inversion as V_breunlich) vs the "
            "digitized Yamashita-Kino full-curve centreline ratio 2.358 (band [2.091, 2.625]; solid-line "
            "2.235). Re-anchored 2026-07-13: the earlier ~1.45 comparator was a digitization under-read, so "
            "the corrected, strictly-harder target flips this row PASS->FAIL (engine ~1.31 is -44% vs the "
            "centreline / -41% vs the solid line -- outside +-30% across the whole digitization band). "
            "Registered expected-FAIL finding: the first sourced quantification of the v1 placeholder's "
            "temperature-shape deficit (no input tuned; formation.py untouched). The three Yamashita rows "
            "(lcT, ratio, curve) are one test (the same lambda_c(T)/X_mu(T) inversion), counted once below.",
            "independent",
            "yamashita_shape",
        )
    )

    # V_yamashita_curve: engine lambda_c(T)/lambda_c(300) vs the digitized centreline ratio at
    # 200/400/600/800 K, +-30% per point. The 800 K point is a registered expected-FAIL; 200/400/600 K are
    # either-outcome-acceptable pre-run (PRE_REGISTRATION.md amendment 2026-07-13).
    yk300 = yk[300]
    _curve = []
    for T in (200.0, 400.0, 600.0, 800.0):
        eng = _implied_lc(_xmu(rates, 0.00557, T=T, include_loss_channels=inc)) / lc300
        dig = yk[int(T)] / yk300
        _curve.append((int(T), eng, dig, _within(eng, dig, "+-30%")))
    curve_passed = all(row[3] for row in _curve)
    _curve_note = "; ".join(
        f"{T} K {'PASS' if p else 'FAIL'} (engine {e:.3f} vs digitized {d:.3f}, {(e - d) / d * 100:+.1f}%)"
        for T, e, d, p in _curve
    )
    out.append(
        Result(
            "V_yamashita_curve",
            "digitized lambda_c(T)/lambda_c(300) at 200/400/600/800 K "
            "(Yamashita-Kino Fig.3a, data/yamashita_kino_lc_T.csv)",
            _curve[-1][1],  # engine 800/300 ratio -- the headline registered-FAIL point
            "+-30% per point",
            curve_passed,
            "independent per-point shape check vs the digitized full curve -- " + _curve_note + ". "
            "Registered: the 800 K point is an expected-FAIL (the placeholder T-shape is too flat vs the "
            "sourced curve); 200/400/600 K were either-outcome-acceptable pre-run (PRE_REGISTRATION.md "
            "amendment 2026-07-13). One test with V_yamashita_lcT / V_yamashita_ratio (counted once below).",
            "independent",
            "yamashita_shape",
        )
    )

    import jax.numpy as jnp

    from . import formation

    peak = float(jnp.max(formation.lambda_dtmu_energy(jnp.linspace(0.2, 0.7, 400), F=1)))
    out.append(
        Result(
            "V_faifman_peak",
            "7.1e9 s^-1 @ 0.423 eV (Fujiwara 2000)",
            peak,
            "+-25%",
            _within(peak, float(tgt["V_faifman_peak"]["value"]), tgt["V_faifman_peak"]["tolerance"]),
            "anchor-consistency check: the F=1 peak amplitude IS the inserted measured value "
            "(validates the resonance-model construction, not independent physics)",
            "anchor-consistency",
        )
    )

    # --- Three registered independent-prediction targets (pre-framed to FAIL; see PRE_REGISTRATION.md).
    # They measure the v1 placeholder's distance from the field's own rates; a PASS is a bug, not a win.
    omega_pred = omega_fraction(rates["omega_s0"]) * (1.0 - rates.value("R_col")) * 100.0
    _po = tgt["V_petitjean_omega"]
    out.append(
        Result(
            "V_petitjean_omega",
            "0.45% measured effective sticking (band [0.40,0.50]; Breunlich/Petitjean)",
            omega_pred,
            _po["tolerance"],
            _within(omega_pred, float(_po["value"]), _po["tolerance"]),
            "independent prediction of effective sticking from the ledger microphysics "
            "(omega_s0 x (1-R_col)); FAIL is the registered finding: the ~+24% gap to the "
            "measured 0.45% is the side-channel share the v1 effective parameters absorb "
            "(accounting.md re-attribution; pending lambda_ttmu acquisition).",
            "independent",
        )
    )

    faif_900 = 0.75 * float(formation.lambda_dtmu(900.0, 1.0, 1)) + 0.25 * float(
        formation.lambda_dtmu(900.0, 1.0, 0)
    )
    _f9 = tgt["V_faifman_900K"]
    out.append(
        Result(
            "V_faifman_900K",
            "2.3e9 s^-1 (Faifman1989 Maxwellian, 900 K; ledger lambda_dtmu_900K)",
            faif_900,
            _f9["tolerance"],
            _within(faif_900, float(_f9["value"]), _f9["tolerance"]),
            "independent thermal-formation check: statistical hyperfine mix 3/4 F=1 + 1/4 F=0 of the v1 "
            "formation model vs the ledger's own Faifman1989 900 K rate; registered expected-FAIL (~20x low) "
            "-- the placeholder's thermal scale is anchored near the measured lambda_c(300 K), not the bare "
            "Faifman lambda_dtmu (needs_verification carried from rates.csv).",
            "independent",
        )
    )

    faif_lowT = float(formation.lambda_dtmu_energy(0.2, F=0))
    _fl = tgt["V_faifman_lowT"]
    out.append(
        Result(
            "V_faifman_lowT",
            "2e10 s^-1 (Faifman1989 energy-resolved, E=0.2 eV; ledger lambda_dtmu_lowT)",
            faif_lowT,
            _fl["tolerance"],
            _within(faif_lowT, float(_fl["value"]), _fl["tolerance"]),
            "independent energy-resolved check: the v1 resonance-model F=0 branch at E=0.2 eV vs the "
            "ledger's own Faifman1989 lambda_dtmu_lowT; registered expected-FAIL (~17x low) -- the "
            "placeholder near-threshold F=0 resonances are unsourced geometry, not the bare Faifman rate "
            "(needs_verification carried).",
            "independent",
        )
    )

    out.append(
        Result(
            "V_nagamine_trend",
            "solid D-T (5-16 K): cycling rises & losses fall as T falls (RIKEN-RAL)",
            float("nan"),
            "qualitative monotone",
            None,
            "DEFERRED: v1 is a thermalized gas/liquid model; solid-phase condensed-matter "
            "formation is Phase-3 scope (pre-registered in PRE_REGISTRATION.md)",
            "independent",
        )
    )
    return out


def report_markdown(results, channels="off") -> str:
    if channels == "on":
        # channels-ON diagnostic variant (VALIDATION_CHANNELS.md); the committed channels-OFF
        # VALIDATION.md below is byte-for-byte unchanged.
        lines = [
            "# VALIDATION_CHANNELS.md -- channels-ON variant (auto-generated by `openmucf.validate`)",
            "",
            "**Loss RE-ATTRIBUTION, not new physics.** This is the same trust gate run with "
            "`include_loss_channels=True` (ttmu side-branch + He-3 scavenging) at the SAME operating "
            "points. The channels-OFF `VALIDATION.md` remains the v1 trust gate; this doc is diagnostic.",
            "",
            "**Pre-framed outcome:** the ttmu formation rate `lambda_ttmu` is currently BLOCKED "
            "(0.0, needs_verification -- pending acquisition of the Matsuzaki/Bom tt-fusion tables, "
            "*Muon Catal. Fusion*) and the validation anchors are He-purged (c_He=0), so the two channels "
            "are inert here and this scoreboard reproduces the channels-OFF one **exactly**. Once "
            "`lambda_ttmu` is pinned, the Kou-Chen +-10% PASSes may convert to documented discrepancies -- "
            "that will be a FINDING about the tt channel's share of their effective numbers, NOT a "
            "regression (the channels-OFF scoreboard stays the trust gate). See `docs/accounting.md`.",
            "",
            "Operating point for sticking-controlled checks: **T=300 K, phi=1.2, c_t=0.5** "
            "(canonical liquid).",
            "",
            "| target | class | observed | predicted | tolerance | verdict | note |",
            "|---|---|---|---|---|---|---|",
        ]
    else:
        lines = [
            "# VALIDATION.md -- trust gate (auto-generated by `openmucf.validate`)",
            "",
            "Engine reproduction of the pre-registered targets in `openmucf/data/validation_targets.csv` "
            "(see `PRE_REGISTRATION.md`). The `class` column is the claim tier: only `independent` rows are "
            "genuine predictions -- and as of v1 the passing set contains none (the three `independent` rows "
            "are registered, pre-framed FAILs, below).",
            "Operating point for sticking-controlled checks: **T=300 K, phi=1.2, c_t=0.5** "
            "(canonical liquid).",
            "",
            "| target | class | observed | predicted | tolerance | verdict | note |",
            "|---|---|---|---|---|---|---|",
        ]
    for r in results:
        verdict = "DEFERRED" if r.passed is None else ("PASS" if r.passed else "**FAIL**")
        pred = "n/a" if r.predicted != r.predicted else f"{r.predicted:.1f}"
        lines.append(
            f"| {r.target_id} | {r.category} | {r.observed} | {pred} | {r.tolerance} | {verdict} | {r.note} |"
        )
    n_pass = sum(1 for r in results if r.passed is True)
    n_indep_pass = sum(1 for r in results if r.passed is True and r.category == "independent")
    n_fail = sum(1 for r in results if r.passed is False)
    n_fail_reg = sum(1 for r in results if r.passed is False and r.category == "independent")
    n_defer = sum(1 for r in results if r.passed is None)
    _seen: set[str] = set()
    n_distinct = 0
    for r in results:
        if r.dedup_group:
            if r.dedup_group in _seen:
                continue
            _seen.add(r.dedup_group)
        n_distinct += 1
    lines += [
        "",
        f"**Summary: {n_pass} pass ({n_indep_pass} independent), {n_fail} fail "
        f"({n_fail_reg} registered placeholder-distance findings -- see notes), {n_defer} deferred. "
        f"Distinct tests: {n_distinct} (the three Yamashita rows are one test, disclosed above).**",
        "",
        "Gate rule: a target passes only within its pre-registered tolerance; deferred items are documented "
        "limitations of the v1 model, not silent passes. No input was tuned to hit a validation target, with "
        "one disclosed anchor: the formation model's overall scale (`formation._CALIB`) is set to the "
        "room-temperature thermal rate (see formation.py), so yield-level targets at 300 K are not fully "
        "independent of that anchor.",
        "",
        "The T-shape rows are likewise not independent: they test the calibrated resonance-model "
        "construction (hand-placed, unsourced placeholder resonance positions -- see formation.py), not a "
        "sourced temperature dependence. The only independent rows are those marked `independent` in the "
        "class column; as of this version the passing set contains none -- the three FAILING independent "
        "rows are pre-registered findings that measure the v1 placeholder's distance from the field's own "
        "rates (PRE_REGISTRATION.md amendment, 2026-07-12), and are the standing, quantified motivation for "
        "the sourced-formation upgrade and the Phase-3 reactivation module.",
    ]
    return "\n".join(lines) + "\n"
