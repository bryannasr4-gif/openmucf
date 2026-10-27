"""Preserve the historical common-pair policy/results and generate their scoped documentation.

The policy and results retain the legacy getter model. The section renderer also
describes the separate current experimental linear-recoil evaluator; it does not
regenerate the historical records under that formula.

``policy`` writes ``data/g4/d1/bound_rates_policy.json``: the prescription and its inputs, the frozen
reference pair of each production isotope, the 12 groups with their seeds and start-time strata, the limits of
the four tests of each group (the capture count's binomial interval and the scaled sum of all delays' gamma
interval; the two branch-conditioned sums are limited at evaluation time, from the observed branch counts),
the critical value of the distribution family, the power of the four tests against the stated alternative, and
the tolerances. It uses scipy's binomial and gamma functions (a development dependency of this repository).

``results`` writes ``data/g4/d1/bound_rates_results.json`` from ``data/g4/d1/bound_rates_harvest.csv``, the
policy and ``data/g4/d1/bound_rates_theory.json``.
"""

from __future__ import annotations

import argparse
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

from openmucf.g4 import d1_boundrates as bound
from openmucf.g4 import d1_mixture as mixture

ROOT = Path(__file__).resolve().parents[1]


def _stats():  # type: ignore[no-untyped-def]
    from scipy import stats  # development dependency

    return stats


def primary_limits(pair: bound.Pair, events: int, alpha: float) -> dict[str, Any]:
    """The inclusive binomial interval of the capture count and the gamma interval of ``Lambda * sum(T)``."""
    stats = _stats()
    p = float(pair.p_capture)
    lo = int(stats.binom.ppf(alpha / 2, events, p))
    hi = int(stats.binom.isf(alpha / 2, events, p))
    size = float(stats.binom.cdf(lo - 1, events, p) + stats.binom.sf(hi, events, p))
    q_lo = float(stats.gamma.ppf(alpha / 2, events))
    q_hi = float(stats.gamma.isf(alpha / 2, events))
    return {"capture_count": [lo, hi], "capture_count_null_size": repr(size), "sum_all": [repr(q_lo), repr(q_hi)]}


def power(pair: bound.Pair, alternative_decay: float, events: int, alpha: float,
          limits: dict[str, Any]) -> dict[str, str]:
    """Rejection probabilities of the four tests when the decay rate is ``alternative_decay`` and the capture
    rate is unchanged; the branch-conditioned tests summed over the alternative's binomial branch count, the
    omitted count probability bounded and reported."""
    import numpy as np

    stats = _stats()
    c, d0, d1_ = pair.capture, pair.decay, alternative_decay
    lam0, lam1 = c + d0, c + d1_
    p1 = c / lam1
    lo, hi = limits["capture_count"]
    branch = float(stats.binom.cdf(lo - 1, events, p1) + stats.binom.sf(hi, events, p1))
    scale = lam0 / lam1
    q_lo, q_hi = (float(x) for x in limits["sum_all"])
    all_t = float(stats.gamma.cdf(q_lo, events, scale=scale) + stats.gamma.sf(q_hi, events, scale=scale))
    sd = math.sqrt(events * p1 * (1 - p1))
    k = np.arange(max(1, int(events * p1 - 12 * sd)), int(events * p1 + 12 * sd) + 1)
    w = stats.binom.pmf(k, events, p1)
    omitted = float(stats.binom.cdf(k[0] - 1, events, p1) + stats.binom.sf(k[-1], events, p1))
    out = {"branch": repr(branch), "sum_all": repr(all_t), "omitted_count_probability": repr(omitted)}
    for name, shape in (("sum_capture", k), ("sum_decay", events - k)):
        a_lo, a_hi = stats.gamma.ppf(alpha / 2, shape), stats.gamma.isf(alpha / 2, shape)
        # one minus the acceptance probability, which is small, so the power is not rounded above one
        accept = stats.gamma.cdf(a_hi, shape, scale=scale) - stats.gamma.cdf(a_lo, shape, scale=scale)
        out[name] = repr(1.0 - math.fsum(w * accept) - omitted)
    return out


def build_policy(root: Path) -> dict[str, Any]:
    from openmucf.g4 import d1_lifetime as lifetime
    from openmucf.g4.sources import d1_nuclear_capture as d1

    policy = bound.policy_core(root)
    n_primary = len(policy["groups"]) * len(bound.PRIMARY_TESTS)
    n_distribution = len(policy["groups"]) * len(bound.STREAMS)
    alpha_primary = bound.per_test_alpha(n_primary)
    alpha_distribution = bound.per_test_alpha(n_distribution)
    helper = d1.load_helper(root / d1.HELPER_RELPATH)
    zmin, zmax = lifetime.helper_clamp(helper)
    limits, powers = {}, {}
    for label, z, _a, events in bound.ISOTOPES:
        pair = bound.frozen_pair(policy, label)
        limits[label] = primary_limits(pair, events, float(alpha_primary))
        old_helper = lifetime.helper_decay_rate(z, helper.zeff, zmin, zmax)
        powers[label] = power(pair, old_helper, events, float(alpha_primary), limits[label])
        powers[label]["alternative_decay"] = old_helper.hex()
    crit = mixture.critical(bound.TIME_BINS - 1, alpha_distribution)
    policy["statistics"] = {
        "family_alpha": str(bound.FAMILY_ALPHA),
        "primary": {"tests": list(bound.PRIMARY_TESTS), "count": n_primary,
                    "per_test_alpha": str(alpha_primary), "two_sided": True, "limits": limits,
                    "conditional": "sum_capture under Gamma(K, 1) and sum_decay under Gamma(N - K, 1), K the "
                                   "observed capture count; inclusive quantile interval at per_test_alpha / 2 "
                                   "in each tail; an empty branch fails"},
        "distribution": {"streams": list(bound.STREAMS), "bins": bound.TIME_BINS, "count": n_distribution,
                         "per_test_alpha": str(alpha_distribution), "dof": bound.TIME_BINS - 1,
                         "critical": [str(crit.lo), str(crit.hi)]},
        "power": {"alternative": "the atomic route's former decay rate with the capture rate unchanged",
                  "values": powers},
    }
    _ = Decimal  # the policy's alphas are decimal strings
    return policy


def primary_tests(stats: Any, g: dict[str, Any], pair: bound.Pair, limits: dict[str, Any],
                  alpha: float) -> dict[str, dict[str, Any]]:
    """The four registered tests of one group, from its merged statistics: the capture count against its
    frozen binomial interval; ``Lambda * sum(T)`` over all stops against the frozen Gamma(N, 1) interval; over the
    capture and the decay stops against Gamma(K, 1) and Gamma(N - K, 1) at the observed K. Two-sided p-values are
    reported beside each verdict; an empty branch fails."""
    n, k = int(g["events"]), int(g["captures"])
    lam = pair.total
    u = {name: lam * float.fromhex(g["sums"][name]) for name in bound.STREAMS}
    p = float(pair.p_capture)
    lo, hi = limits["capture_count"]
    out: dict[str, dict[str, Any]] = {}
    p_k = min(1.0, 2 * min(float(stats.binom.cdf(k, n, p)), float(stats.binom.sf(k - 1, n, p))))
    out["capture_count"] = {"statistic": str(k), "interval": [str(lo), str(hi)], "p_two_sided": repr(p_k),
                            "verdict": "pass" if lo <= k <= hi else "fail"}
    for name, shape, frozen in (("sum_all", n, limits["sum_all"]), ("sum_capture", k, None),
                                ("sum_decay", n - k, None)):
        stream = {"sum_all": "all", "sum_capture": "capture", "sum_decay": "decay"}[name]
        if shape <= 0:
            out[name] = {"statistic": repr(u[stream]), "interval": None, "p_two_sided": None,
                         "verdict": "fail", "reason": "empty branch"}
            continue
        if frozen is not None:
            q_lo, q_hi = (float(x) for x in frozen)
        else:
            q_lo, q_hi = float(stats.gamma.ppf(alpha / 2, shape)), float(stats.gamma.isf(alpha / 2, shape))
        pv = min(1.0, 2 * min(float(stats.gamma.cdf(u[stream], shape)), float(stats.gamma.sf(u[stream], shape))))
        out[name] = {"statistic": repr(u[stream]), "shape": str(shape), "interval": [repr(q_lo), repr(q_hi)],
                     "p_two_sided": repr(pv), "verdict": "pass" if q_lo <= u[stream] <= q_hi else "fail"}
    return out


def build_results(root: Path) -> dict[str, Any]:
    """The four result families from the committed summary, the policy and the theory status, each with its own
    status: the rates (and the wiring and completeness they rest on), the transport sampling, the clocks, and the
    theory. One family's status never stands in for another's. The summary is authenticated against the policy's
    registered roster first (:func:`openmucf.g4.d1_boundrates.roster_problems`): any problem is named in every
    family and makes its status INCOMPLETE, never PASS. Every registered stratum is graded for completeness and
    rates on its own, and the clocks of every registered stratum and category are required."""
    import json

    stats = _stats()
    policy = bound.load_policy(root)
    harvest = bound.load_harvest(root)
    theory = json.loads((root / bound.THEORY_RELPATH).read_text(encoding="utf-8"))
    roster = bound.roster_problems(policy, harvest)
    alpha = float(policy["statistics"]["primary"]["per_test_alpha"])
    n_dist = int(policy["statistics"]["distribution"]["count"])
    rate: dict[str, Any] = {}
    transport: dict[str, Any] = {}
    clock: dict[str, Any] = {}
    for group in policy["groups"]:
        name = bound.group_name(group)
        registered = [int(st["t0_ns"]) for st in group["strata"]]
        if name not in harvest or "group" not in harvest[name] or "meta" not in harvest[name]:
            missing = {"verdict": "fail", "problems": ["the summary has no complete cell for this group"]}
            rate[name] = {"complete": missing, "rate": missing, "strata": {}}
            transport[name] = {"primary": {t: {"verdict": "fail", "reason": "missing cell"}
                                           for t in bound.PRIMARY_TESTS},
                               "distribution": {s: {"verdict": "fail", "reason": "missing cell"}
                                                for s in bound.STREAMS}, "strata": {}}
            clock[name] = bound.gate_clocks({}, group["route"], registered)
            continue
        cell = harvest[name]
        g = cell["group"]
        switch_on = cell["meta"]["switch_on"] == "True"
        pair = bound.frozen_pair(policy, group["isotope"])
        present = {int(stat.split("_", 1)[1]): v for stat, v in cell.items() if stat.startswith("stratum_")}
        strata_gates = {}
        for st in group["strata"]:
            t0 = int(st["t0_ns"])
            part = present.get(t0)
            if part is None:
                gap = {"verdict": "fail", "problems": [f"stratum {t0} missing"]}
                strata_gates[t0] = {"complete": gap, "rate": gap}
            else:
                strata_gates[t0] = {"complete": bound.gate_complete(part, int(st["count"])),
                                    "rate": bound.gate_rate(part, group["route"], switch_on)}
        rate[name] = {"complete": bound.gate_complete(g, int(group["events"])),
                      "rate": bound.gate_rate(g, group["route"], switch_on), "strata": strata_gates}
        transport[name] = {"primary": primary_tests(stats, g, pair,
                                                    policy["statistics"]["primary"]["limits"][group["isotope"]],
                                                    alpha),
                           "distribution": bound.gate_distribution(g, n_dist),
                           "strata": {t0: {"events": v["events"], "captures": v["captures"]}
                                      for t0, v in sorted(present.items())}}
        clock[name] = bound.gate_clocks(present, group["route"], registered)

    def status(ok: bool) -> str:
        return "INCOMPLETE" if roster else ("PASS" if ok else "FAIL")

    rate_ok = all(v["complete"]["verdict"] == "pass" and v["rate"]["verdict"] == "pass"
                  and all(x["complete"]["verdict"] == "pass" and x["rate"]["verdict"] == "pass"
                          for x in v["strata"].values()) and v["strata"]
                  for v in rate.values()) and bool(rate)
    rejections = [f"{name}:{test}" for name, v in transport.items() for test, r in v["primary"].items()
                  if r["verdict"] != "pass"]
    dist_rejections = [f"{name}:{stream}" for name, v in transport.items()
                       for stream, r in v["distribution"].items() if r["verdict"] != "pass"]
    delivery_ok = all(v["delivery"]["verdict"] == "pass" for v in clock.values()) and bool(clock)
    physical_fail = sorted(f"{name}:{key}" for name, v in clock.items() for key, r in v["physical"].items()
                           if r["verdict"] != "pass")
    unexpected = sorted(f"{name}:{key}" for name, v in clock.items() for key, r in v["physical"].items()
                        if not r["as_predicted"])
    if roster or not clock or any(v["physical_verdict"] == "incomplete" for v in clock.values()):
        physical_status = "INCOMPLETE"
    else:
        physical_status = "FAIL" if physical_fail else "PASS"
    admission = bound.theory_admission(theory)
    return {
        "RATE_RESULTS": {"status": status(rate_ok), "roster_problems": roster, "groups": rate},
        "TRANSPORT_RESULTS": {"status": status(bool(transport) and not rejections and not dist_rejections),
                              "roster_problems": roster,
                              "primary_rejections": rejections, "distribution_rejections": dist_rejections,
                              "groups": transport},
        "CLOCK_RESULTS": {"delivery_status": status(delivery_ok), "physical_status": physical_status,
                          "roster_problems": roster,
                          "physical_failures": physical_fail, "not_as_predicted": unexpected, "groups": clock},
        "THEORY_STATUS": {"status": admission["status"], "problems": admission["problems"],
                          "accuracy": theory["accuracy"], "qualified_roster": admission["roster"],
                          "screen": theory["screen"]},
    }

# --------------------------------------------------------------------------------------------
# section 14 of DATASET_D1.md, generated from the committed files
# --------------------------------------------------------------------------------------------

DOCUMENT = "DATASET_D1.md"
SECTION_HEADING = "## 14. One bound-decay rate pair for both stopped-muon routes"
RULINGS_RELPATH = "data/g4/d1/bound_rates_drill_rulings.csv"
WORKERS_RELPATH = "data/g4/d1/bound_rates_workers.csv"
#: The results of the first summary of the same raw records, kept as they were committed.
HISTORY_RESULTS_RELPATH = "data/g4/d1/history/bound_rates_results.analysis1.json"
#: What each physical-clock category compares, in words.
CLOCK_WORDS = {
    "atom_creation": "the muonic atom's creation",
    "atom_dio": "the atomic route's decay electron",
    "atom_capture": "the atomic route's capture products",
    "stop_dio": "the stopping route's decay electron",
    "stop_capture": "the stopping route's capture products",
}


def _csv_rows(root: Path, relpath: str) -> list[dict[str, str]]:
    import csv

    with (root / relpath).open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def _thousands(n: int) -> str:
    return f"{n:,}"


def drill_tally(root: Path, drills: list[dict[str, str]], ruled: dict[str, dict[str, str]]) -> dict[str, int]:
    """The outcome of every drill and control run (``d1_boundrates.drill_families`` / ``drill_outcome``, a ruled
    run graded on its ruled families): ``red`` -- runs with a nonempty intended set, ruled runs excluded, that
    turned every intended family red; ``unaffected_green`` -- drill runs with the intended set "none" (an
    unaffected route or isotope) that stayed green; ``controls_green``; ``ruled_red``. Any other outcome stops the
    rendering: the section states only these."""
    stats = _stats()
    ref = bound.Reference(root)
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in drills:
        grouped.setdefault(r["cell"], []).append(r)
    tally = {"red": 0, "unaffected_green": 0, "controls_green": 0, "ruled_red": 0}
    for cell, rows in sorted(grouped.items()):
        meta = {r["key"]: r["value"] for r in rows if r["stat"] == "meta"}
        st = bound.rows_stats([r for r in rows if r["stat"] == "group"])
        families = bound.drill_families(st, meta, ref.pair(int(meta["z"]), int(meta["a"])), stats)
        intended = ruled[cell]["ruled_intended"] if cell in ruled else meta["intended"]
        outcome = bound.drill_outcome(meta["kind"], intended, families)
        if meta["kind"] == "control":
            key = "controls_green" if outcome == "green" else ""
        elif cell in ruled:
            key = "ruled_red" if outcome == "red" else ""
        elif intended == "none":
            key = "unaffected_green" if outcome == "green" else ""
        else:
            key = "red" if outcome == "red" else ""
        if not key:
            raise SystemExit(f"{cell}: outcome {outcome} with intended {intended!r}; the section does not state it")
        tally[key] += 1
    return tally


def registered_outcomes(results: dict[str, Any]) -> dict[str, Any]:
    """Every registered outcome of a results file, without the problem texts and lists a later analysis adds:
    each family's status; per group the completeness and rate verdicts with the exact-event count, the event
    count and the largest relative deviation; each registered test's verdict, statistic and interval; each
    distribution test's verdict, statistic and count; the rejections; the delivered clock's verdict and each
    physical-clock cell's verdict, item count, failures and largest residual; the failing and unexpected
    physical cells; and the theory status."""
    rate, transport, clock = results["RATE_RESULTS"], results["TRANSPORT_RESULTS"], results["CLOCK_RESULTS"]
    return {
        "rate_status": rate["status"],
        "rate": {name: (g["complete"]["verdict"], g["rate"]["verdict"], g["rate"].get("exact_events"),
                        g["rate"].get("events"), g["rate"].get("max_relative_deviation"))
                 for name, g in rate["groups"].items()},
        "transport_status": transport["status"],
        "primary_rejections": transport["primary_rejections"],
        "distribution_rejections": transport["distribution_rejections"],
        "primary": {name: {t: (v["verdict"], v.get("statistic"), v.get("interval")) for t, v in g["primary"].items()}
                    for name, g in transport["groups"].items()},
        "distribution": {name: {k: (v["verdict"], v.get("x2"), v.get("n")) for k, v in g["distribution"].items()}
                         for name, g in transport["groups"].items()},
        "delivery_status": clock["delivery_status"],
        "physical_status": clock["physical_status"],
        "physical_failures": clock["physical_failures"],
        "not_as_predicted": clock["not_as_predicted"],
        "clocks": {name: (g["delivery"]["verdict"],
                          {k: (v["verdict"], v["n"], v["fail"], v["max_abs_residual_ns"])
                           for k, v in g["physical"].items()})
                   for name, g in clock["groups"].items()},
        "theory_status": results["THEORY_STATUS"]["status"],
    }


def render_section(root: Path) -> str:
    """Section 14 of DATASET_D1.md: every number read from the committed files."""
    import json

    from openmucf.g4 import d1_lifetime as lifetime
    from openmucf.g4.sources import d1_nuclear_capture as d1

    policy = bound.load_policy(root)
    results = json.loads((root / bound.RESULTS_RELPATH).read_text(encoding="utf-8"))
    theory = json.loads((root / bound.THEORY_RELPATH).read_text(encoding="utf-8"))
    sweep = _csv_rows(root, bound.SWEEP_RELPATH)
    identity = _csv_rows(root, bound.IDENTITY_RELPATH)
    identity_drills = _csv_rows(root, bound.IDENTITY_DRILLS_RELPATH)
    drills = bound.read_csv(root / bound.DRILLS_RELPATH, bound.HARVEST_COLUMNS)
    rulings = _csv_rows(root, RULINGS_RELPATH)
    workers = _csv_rows(root, WORKERS_RELPATH)
    ref = bound.Reference(root)
    grid = sorted(k for k in ref.oracle.light if bound.in_domain(*k))
    if grid != sorted((z, a) for z in range(2, 14) for a in range(z, 3 * z + 1)):
        raise SystemExit("the oracle's keys inside the domain are not the grid this section describes")
    refused = sorted({key for r in sweep if r["switch"] == "on" and r["profile"] == "none"
                      for key in r["refused_keys"].split()}, key=lambda k: tuple(int(x) for x in k.split("-")))
    helper = d1.load_helper(root / d1.HELPER_RELPATH)
    zmin, zmax = lifetime.helper_clamp(helper)
    us = float(d1.MICROSECOND)
    lines = [
        SECTION_HEADING,
        "",
        "The current default-off experimental common provider uses "
        "`low-z-inclusive-tree-linear-recoil-v1`: "
        "`Q = 1 - (Z alpha)^2 * (1/2 - eta)`, `eta = muonMass/nuclearMass`, "
        "multiplying the unchanged free normalization `0.45517005 / microsecond`. Capture still comes "
        "from `G4MuonicAtomHelper::GetMuonCaptureRate`, and both consumers use their existing mass APIs.",
        "With the existing option on, both routes, including the stopping route, now receive this "
        "decay rate, which differs from the stopping getter's legacy value. The static getters remain "
        "unchanged. The earlier opt-in changelog entry and the historical verification below describe "
        "the former pair.",
        "This is leading binding and linear eta for the fully inclusive tree width of an unpolarized "
        "Coulomb 1s muon with a point spinless spectator. The guard requires finite positive masses "
        "and finite `eta < 1/2`, with `0 < decay <= free`; it is a consistency guard, not an accuracy "
        "bound. Physical use needs `eta << 1`. Higher recoil/binding, radiative and nuclear corrections "
        "have no assigned physical uncertainty. See [the derivation](docs/low-z-bound-muon-decay.md).",
        "`openmucf/g4/d1_linear_recoil.py` and the standalone C++ checker test the current formula. "
        "The legacy getters, `d1_lifetime.py`, `d1_boundrates.Reference`, frozen policy and all numerical "
        "harvest/results below retain their historical meanings. No new physical-accuracy or transport "
        "claim follows from those old records.",
        "",
        "The rest of this section records the historical common provider: the helper capture resolver "
        "and the stopping getter's legacy low-Z decay expression "
        "`Q_old = 1 - (Z alpha)^2 * (1/2 + 0.06*eta)`. Its code comment attributes that expression to "
        "von Baeyer and Leiter (Phys. Rev. A **19**, 1371 (1979)). That source was not read for the "
        "historical harvest. Its contrary coefficient and the scope of the new finite-M derivation "
        "are discussed in the linked derivation. The current-minus-legacy difference "
        "`1.06*(Z alpha)^2*eta` is a model difference, not an uncertainty estimate.",
        "The provider answers for 2 ≤ Z ≤ 13 and A ≥ Z. Hydrogen, whose atomic-route rate is a tabulated value, "
        "Z ≥ 14, where the stopping route uses a fit, and a natural-composition request keep each route's own "
        "pair.",
        f"Inside that domain a negative capture rate is refused with a fatal `G4Exception` rather than clipped: "
        f"of the {len(grid)} keys with 2 ≤ Z ≤ 13 and Z ≤ A ≤ 3Z that the decay-rate oracle and the sweep below "
        f"cover (not all of them bound nuclei), the compiled fallback gives one to {len(refused)}, all "
        f"neutron-rich ({', '.join(refused)}).",
        "The option is off by default, the dataset opt-in does not set it, and the getters `GetMuonDecayRate` of "
        "both classes are unchanged.",
        "",
        "### 14.1 Historical option behavior and numerical rows",
        "",
        "On the stopping route it changes nothing: the pair is that route's own, bit for bit "
        "(`bound_rates_identity.csv`). On the atomic route the muonic atom's lifetimes come from the pair instead "
        "of the helper's former decay rate:",
        "",
        "| nuclide | capture (µs⁻¹) | decay, with the option (µs⁻¹) | decay, atomic route without it (µs⁻¹) | "
        "lifetime with the option (ns) | atomic lifetime without it (ns) | capture probability with the option | "
        "atomic, without it |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, z, _, _ in bound.ISOTOPES:
        p = bound.frozen_pair(policy, label)
        former = lifetime.helper_decay_rate(z, helper.zeff, zmin, zmax)
        lines.append(
            f"| {label} | {p.capture * us:.5f} | {p.decay * us:.9f} | {former * us:.9f} | {1 / p.total:.3f} | "
            f"{1 / (p.capture + former):.3f} | {float(p.p_capture):.6f} | "
            f"{p.capture / (p.capture + former):.6f} |")
    n_cells = len({(r["revision"], r["route"], r["nuclide"], r["t0_ns"]) for r in identity})
    keys = sorted({int(r["keys"]) for r in sweep})
    events = sum(int(g["events"]) for g in policy["groups"])
    tr = results["TRANSPORT_RESULTS"]
    n_primary = int(policy["statistics"]["primary"]["count"])
    n_dist = int(policy["statistics"]["distribution"]["count"])
    meta: dict[str, dict[str, str]] = {}
    for r in drills:
        if r["stat"] == "meta":
            meta.setdefault(r["cell"], {})[r["key"]] = r["value"]
    ruled = {r["item"]: r for r in rulings if r["kind"] == "statistical"}
    ruled_cells = set(ruled)
    reruns = sorted({r["rerun"] for r in ruled.values()})
    absent = sorted((ruled_cells | set(reruns)) - set(meta))
    if absent:
        raise SystemExit(f"a ruled drill run or its rerun is not in the drill summary: {', '.join(absent)}")
    unchecked = sorted(c for c, m in meta.items()
                       if m.get("provenance_problems") != "0" or m.get("raw_digest_problems") != "0")
    if unchecked:
        raise SystemExit(f"a drill run whose run record or raw files were not checked clean: {unchecked[0]}")
    rerun_sizes = sorted({int(meta[c]["events"]) for c in reruns})
    ruled_sizes = sorted({int(meta[c]["events"]) for c in ruled_cells})
    tally = drill_tally(root, drills, ruled)
    n_red, n_unaffected, n_controls = tally["red"], tally["unaffected_green"], tally["controls_green"]
    for family in ("RATE_RESULTS", "TRANSPORT_RESULTS", "CLOCK_RESULTS"):
        if results[family]["roster_problems"]:
            raise SystemExit(f"{family}: the summary does not match the registered roster")
    first = json.loads((root / HISTORY_RESULTS_RELPATH).read_text(encoding="utf-8"))
    if registered_outcomes(first) != registered_outcomes(results):
        raise SystemExit("a registered outcome differs from the first summary's: this section does not describe it")
    harvest = bound.load_harvest(root)
    if {cell["meta"].get("analysis_version") for cell in harvest.values()} != {str(bound.ANALYSIS_VERSION)}:
        raise SystemExit("a harvest cell from another analysis version")
    for group in policy["groups"]:
        g = harvest[bound.group_name(group)]["group"]
        events_ = int(group["events"])
        if group["route"] == bound.STOPPING:
            if not (int(g["rate_exact"]) == events_ and int(g["wiring_fail"]) == 0):
                raise SystemExit(f"{bound.group_name(group)}: not every stopping event received the pair exactly")
        elif not (int(g["rate_exact"]) == events_ and int(g["recovered_fail"]) == 0):
            raise SystemExit(f"{bound.group_name(group)}: an atomic lifetime or recovered rate disagrees")
    ruled_checks = [r for r in rulings if r["kind"] == "identity"]
    n_identity = len(identity_drills) - len(ruled_checks)
    split_rows = [r for r in workers if r["on1_vs_on4_events"]]
    split_events = sorted({int(e) for r in split_rows for e in r["on1_vs_on4_events"].split()})
    split_where = sorted({(r["route"], r["target"]) for r in split_rows})
    labels = {f"{z}-{a}": label for label, z, a, _ in bound.ISOTOPES} | {"13-27": "Al-27"}
    if [route for route, _ in split_where] != ["muonic_atom_helper"] * len(split_where):
        raise SystemExit("a thread split outside the atomic route: this section does not describe it")
    events_words = (", ".join(str(e) for e in split_events[:-1]) + " and " + str(split_events[-1])
                    if len(split_events) > 1 else ", ".join(str(e) for e in split_events))
    lines += [
        "",
        "### 14.2 What the historical harvest verified",
        "",
        f"`bound_rates_policy.json` fixed the harvest before it ran. With the option off, {n_cells} "
        "configurations (the roster of `bound_rates_identity.csv`, both routes, both revisions, start times of "
        "0, 250 and 1000 ns) wrote the same stops, transported records and random-engine states in the patched "
        "build as in the unpatched one, traced or not, and the dataset opt-in under each capture profile wrote "
        "the same as the previous patches.",
        f"Through the patched libraries, {len(sweep)} sweeps of {', '.join(str(k) for k in keys)} keys each "
        "(`bound_rates_sweep.csv`) found the two routes' capture rates equal bit for bit, the provider's pair "
        "equal to the helper's capture rate and the stopping expression bit for bit, and the muonic atom storing "
        "1/(c + d), 1/d and 1/c of that pair.",
        "At 1 and 4 worker threads (`bound_rates_workers.csv`) the option adds no dependence on the thread count: "
        f"with it on, the records that differ between 1 and 4 threads (events {events_words} of the atomic route "
        f"on {' and '.join(labels[t] for _, t in split_where)}, the first events of 4-thread workers) are "
        "exactly those that differ in the unpatched build, where Geant4's atomic route itself depends on the "
        "thread count.",
        f"The harvest transported {_thousands(events)} stopped muons on single-isotope targets in "
        f"{len(policy['groups'])} groups (`bound_rates_harvest.csv`), with a build that records every call the "
        "routes make to the provider. Every stopping event made exactly one provider call and received the "
        "registered rate pair bit for bit. Every atomic event stored the three registered reciprocal lifetimes bit "
        "for bit; the rates G4MuonicAtomDecay recovers from those lifetimes agree with the registered pair within "
        f"the registered relative tolerance. The {n_primary} registered tests rejected "
        f"{len(tr['primary_rejections'])} and the {n_dist} tests of the delay distribution rejected "
        f"{len(tr['distribution_rejections'])} (`bound_rates_results.json`).",
        "The summary was recomputed from the same raw records with a stricter reader, which refuses any unobserved "
        "or nonfinite field, grades every registered stratum and clock category on its own, and re-hashes every "
        "raw file it reads, checking it against its run's record and the committed digests "
        "(`bound_rates_raw_digests.csv`); with no new draws, seeds or tolerances, every registered outcome is the "
        "one the first summary gave (`history/bound_rates_results.analysis1.json`).",
        f"{n_red} runs with nonempty intended failure sets turned every intended family red; {n_unaffected} runs "
        f"on unaffected routes or isotopes and all {n_controls} matched controls stayed green. Among the red runs "
        "(`bound_rates_drills.csv`) are a stopping route disconnected from the provider, which only the recorded "
        "calls reveal, and a constant delay with the correct mean, which only the distribution tests reveal. "
        f"{len(ruled_cells)} further runs of {', '.join(_thousands(n) for n in ruled_sizes)} events, "
        "an atomic consumer given the former decay rate and one with a stale decay-in-orbit lifetime, turned the "
        "rate and call checks red but not the transport tests, which have little power against those defects at "
        "that size; their reruns at "
        f"{' and '.join(_thousands(n) for n in rerun_sizes)} events turned the transport tests red as well "
        "(`bound_rates_drill_rulings.csv`).",
        f"{n_identity} identity checks (`bound_rates_identity_drills.csv`), drills and their controls, came out "
        "as expected. An extra random number drawn in the stopping consumer leaves the atomic route unchanged "
        f"({len(ruled_checks)} checks, not applicable: that route never runs it), and one drawn in the atomic "
        "consumer's inserted block is detected on both revisions.",
        "",
        "### 14.3 What the historical harvest does not establish",
        "",
        "The option leaves both routes' product times as they were. Against the correct composition of the stop "
        "time, the sampled delay and any model delay, the harvest's clock check fails exactly where it was "
        "predicted to fail before the harvest ran:",
        "",
    ]
    if results["CLOCK_RESULTS"]["not_as_predicted"]:
        raise SystemExit("a clock result not as predicted: this section does not describe it")
    clock_groups = results["CLOCK_RESULTS"]["groups"]
    failing: dict[str, dict[str, list[int | float]]] = {}
    for g in clock_groups.values():
        for key, r in g["physical"].items():
            if r["verdict"] == "fail":
                cat, t0 = key.split("@")
                acc = failing.setdefault(cat, {}).setdefault(t0, [0, 0, 0.0])
                acc[0] += int(r["n"])
                acc[1] += int(r["fail"])
                acc[2] = max(float(acc[2]), float.fromhex(r["max_abs_residual_ns"]))
    for cat in ("atom_creation", "atom_dio", "atom_capture", "stop_dio", "stop_capture"):
        for t0, (n, fail, worst) in sorted(failing.get(cat, {}).items(), key=lambda kv: int(kv[0])):
            lines.append(f"- {CLOCK_WORDS[cat]}, start time {t0} ns: {_thousands(int(fail))} of "
                         f"{_thousands(int(n))} items, maximum recorded residual approximately {worst:g} ns")
    lines += [
        "",
        "The atomic route creates the muonic atom at twice the stop time, so every atomic product of a stop at a "
        "nonzero start time is late by that time; the stopping route lets a capture product keep the negative "
        "local time the cascade model gives it, so it comes before the capture.",
        f"Physical accuracy: {theory['status']}; {theory['accuracy']}. The admitted roster is empty, so the "
        f"accuracy screen is {theory['screen']} (`bound_rates_theory.json`).",
        "",
        "### 14.4 Evaluated pair rows",
        "",
        "The dataset carries a third kind of table, `bound_rate_primitives`, under the profile `evaluated-bound` "
        "(`d1_bound_rate.evaluated-bound.g4dat` and its provenance file): a tuple `(tau, q, free_tau)` for each "
        "admitted isotope, `tau` and `free_tau` in ns and `q` dimensionless, from which a consumer forms both rates "
        "together: the decay rate `q/free_tau`, and the capture rate, the inverse of `tau` less the decay rate. The "
        "admitted roster is aluminium alone; oxygen and carbon are evaluated as decay factors only and get no row. "
        "`tau` is the late-time lifetime Mizuno et al. (2025) fit for their self-supported plate (section 9) and `q` "
        "the evaluated decay factor; the evaluation, its regions and its decisions are in "
        "`bound_rate_evaluation.json`, generated and checked by `scripts/generate_g4evaluated_rates.py`, and the page "
        "[`docs/evaluated-bound-rates.md`](docs/evaluated-bound-rates.md) renders them.",
        "",
        "A build reads the table only when `G4MUONICDATA_D1_PAIR_PROFILE` names its profile, independently of "
        "`G4MUONICDATA_D1_PROFILE`, and only with both the dataset opt-in and the common-rate option above on: then a "
        "key the table holds takes the pair on both stopped-muon routes, in place of the common formula, and every "
        "other key keeps its capture profile's rate and the common formula. The variable that spans both seams never "
        "selects it, and a pair profile with the common-rate option off, with no pair table, or with a malformed or "
        "empty pair table is refused; `cpp/patches/README.md` states how. The empty admitted roster above concerns "
        "the common formula, not the pair.",
        "",
        "What the evaluation decides, under its evaluator-assessed regions: the late lifetime and the decay rate meet "
        "provisional budgets (conditional; the late-lifetime decision has no validation power, because the delivered "
        "lifetime is the source's own), and the stopped mean and the capture probability are not established: the "
        "hyperfine capture contrast is unresolved. On the atomic route the pair robustly improves the late lifetime "
        "over the stock pair; no other improvement over stock is established. The comparison of section 12 restates "
        "the pair's rows under its profile. The pair is not an independent measurement of the capture branching.",
    ]
    return "\n".join(lines) + "\n"


def write_section(root: Path) -> None:
    """Write section 14 into DATASET_D1.md: replace the section if it is there (it is the last), else append it."""
    path = root / DOCUMENT
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    marker = "\n" + SECTION_HEADING.split(". ", 1)[0] + ". "
    head = text.split(marker, 1)[0].rstrip("\n") + "\n\n"
    path.write_bytes((head + render_section(root)).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("steps", nargs="+", choices=("policy", "results", "section"))
    args = parser.parse_args()
    if "policy" in args.steps:
        policy = build_policy(ROOT)
        (ROOT / bound.POLICY_RELPATH).write_bytes(bound.render_policy(policy))
        for label, lim in policy["statistics"]["primary"]["limits"].items():
            pw = policy["statistics"]["power"]["values"][label]
            print(f"{label} capture_count {lim['capture_count']} sum_all {lim['sum_all']} power {pw}")
        print(f"policy groups={len(policy['groups'])} primary={policy['statistics']['primary']['count']} "
              f"distribution={policy['statistics']['distribution']['count']}")
    if "results" in args.steps:
        import json

        results = build_results(ROOT)
        (ROOT / bound.RESULTS_RELPATH).write_text(json.dumps(results, indent=1, sort_keys=True) + "\n",
                                                  encoding="ascii", newline="\n")
        for family in ("RATE_RESULTS", "TRANSPORT_RESULTS", "CLOCK_RESULTS", "THEORY_STATUS"):
            r = results[family]
            print(family, {k: v for k, v in r.items() if k != "groups"})
    if "section" in args.steps:
        write_section(ROOT)
        print(f"section written into {DOCUMENT}")


if __name__ == "__main__":
    main()
