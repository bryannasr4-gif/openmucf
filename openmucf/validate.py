"""openmucf.validate -- reproduce the pre-registered validation targets (Phase 2.4 trust gate).

Loads each target's observed value and pre-registered tolerance from
``openmucf/data/validation_targets.csv`` and runs the engine against them, reporting per target
predicted vs observed within tolerance. Mutating a CSV target value/tolerance changes the verdict
(see ``tests/test_validate.py``). Honest by construction: targets the v1 model cannot yet hit
(e.g. the solid-phase condensed-matter trend) are marked DEFERRED, not silently passed.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass

from . import cycle
from .rates import TARGETS_CSV

# Canonical liquid operating point for sticking-controlled checks.
_OP = dict(T=300.0, phi=1.2, c_t=0.5)


@dataclass
class Result:
    target_id: str
    observed: str
    predicted: float
    tolerance: str
    passed: bool | None  # None == DEFERRED (honest, not a pass)
    note: str


def _xmu(rates, omega_s_eff, T=_OP["T"]):
    return float(
        cycle.fusions_per_muon_from_conditions(rates, T, _OP["phi"], _OP["c_t"], omega_s_eff=omega_s_eff)
    )


def _load_targets(path=TARGETS_CSV):
    """Load pre-registered targets (observed value + tolerance) from validation_targets.csv, keyed by id."""
    with open(path, newline="") as f:
        return {row["target_id"].strip(): row for row in csv.DictReader(f)}


def _within(pred, value, tol):
    """Pass/fail from a CSV tolerance string: interval '[a,b]' or relative '+-N%'."""
    m_iv = re.match(r"\s*\[([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\]", tol)
    if m_iv:
        return float(m_iv.group(1)) <= pred <= float(m_iv.group(2))
    m_pct = re.search(r"([\d.]+)\s*%", tol)
    if m_pct:
        return abs(pred - value) / abs(value) < float(m_pct.group(1)) / 100.0
    raise ValueError(f"unparseable tolerance {tol!r}")


def run(rates, targets_csv=None):
    tgt = _load_targets(targets_csv or TARGETS_CSV)
    out = []

    x = _xmu(rates, 0.00557)
    out.append(
        Result(
            "V_kouchen_base",
            "112.6 (fed omega_s_eff=0.557%)",
            x,
            "+-10%",
            _within(x, float(tgt["V_kouchen_base"]["value"]), tgt["V_kouchen_base"]["tolerance"]),
            "collision-only baseline",
        )
    )

    x = _xmu(rates, 0.00308)
    out.append(
        Result(
            "V_kouchen_best",
            "156.5 (fed omega_s_eff=0.308%)",
            x,
            "+-10%",
            _within(x, float(tgt["V_kouchen_best"]["value"]), tgt["V_kouchen_best"]["tolerance"]),
            "best external-field scenario",
        )
    )

    x = _xmu(rates, 0.0045)
    out.append(
        Result(
            "V_petitjean",
            "113 (fed omega_s_eff=0.45%)",
            x,
            "[100,150]",
            _within(x, float(tgt["V_petitjean_Xmu"]["value"]), tgt["V_petitjean_Xmu"]["tolerance"]),
            "measured liquid effective sticking",
        )
    )

    xs = [_xmu(rates, 0.00557, T=T) for T in (200.0, 400.0, 600.0, 800.0)]
    out.append(
        Result(
            "V_yamashita_lcT",
            "monotone X_mu(T) rise",
            xs[-1],
            "monotone",
            all(b > a for a, b in zip(xs, xs[1:], strict=False)),
            f"X_mu(200,400,600,800 K)={[round(v, 1) for v in xs]}",
        )
    )

    x = _xmu(rates, 0.00557)
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
        )
    )
    return out


def report_markdown(results) -> str:
    lines = [
        "# VALIDATION.md -- trust gate (auto-generated by `openmucf.validate`)",
        "",
        "Engine reproduction of the pre-registered targets in `openmucf/data/validation_targets.csv` "
        "(see `PRE_REGISTRATION.md`).",
        "Operating point for sticking-controlled checks: **T=300 K, phi=1.2, c_t=0.5** (canonical liquid).",
        "",
        "| target | observed | predicted | tolerance | verdict | note |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        verdict = "DEFERRED" if r.passed is None else ("PASS" if r.passed else "**FAIL**")
        pred = "n/a" if r.predicted != r.predicted else f"{r.predicted:.1f}"
        lines.append(f"| {r.target_id} | {r.observed} | {pred} | {r.tolerance} | {verdict} | {r.note} |")
    n_pass = sum(1 for r in results if r.passed is True)
    n_defer = sum(1 for r in results if r.passed is None)
    n_fail = sum(1 for r in results if r.passed is False)
    lines += [
        "",
        f"**Summary: {n_pass} pass, {n_defer} deferred (honest placeholder limits), {n_fail} fail.**",
        "",
        "Gate rule: a target passes only within its pre-registered tolerance; deferred items are documented "
        "limitations of the v1 model, not silent passes. No input was tuned to hit a validation target, with "
        "one disclosed anchor: the formation model's overall scale (`formation._CALIB`) is set to the "
        "room-temperature thermal rate (see formation.py), so yield-level targets at 300 K are not fully "
        "independent of that anchor; the monotone-rise shape and the beam-resonance energy are.",
    ]
    return "\n".join(lines) + "\n"
