"""Generate the verification policy of the common bound-rate pair, and its results from the committed summary.

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
    theory. One family's status never stands in for another's."""
    import json

    stats = _stats()
    policy = bound.load_policy(root)
    harvest = bound.load_harvest(root)
    theory = json.loads((root / bound.THEORY_RELPATH).read_text(encoding="utf-8"))
    alpha = float(policy["statistics"]["primary"]["per_test_alpha"])
    n_dist = int(policy["statistics"]["distribution"]["count"])
    rate: dict[str, Any] = {}
    transport: dict[str, Any] = {}
    clock: dict[str, Any] = {}
    for group in policy["groups"]:
        name = bound.group_name(group)
        if name not in harvest:
            missing = {"verdict": "fail", "problems": ["the summary has no cell for this group"]}
            rate[name] = {"complete": missing, "rate": missing}
            transport[name] = {"primary": {t: {"verdict": "fail", "reason": "missing cell"}
                                           for t in bound.PRIMARY_TESTS},
                               "distribution": {s: {"verdict": "fail", "reason": "missing cell"}
                                                for s in bound.STREAMS}, "strata": {}}
            clock[name] = {"delivery": missing, "physical": {}, "physical_verdict": "fail"}
            continue
        cell = harvest[name]
        g = cell["group"]
        switch_on = cell["meta"]["switch_on"] == "True"
        pair = bound.frozen_pair(policy, group["isotope"])
        rate[name] = {"complete": bound.gate_complete(g, int(group["events"])),
                      "rate": bound.gate_rate(g, group["route"], switch_on)}
        transport[name] = {"primary": primary_tests(stats, g, pair,
                                                    policy["statistics"]["primary"]["limits"][group["isotope"]],
                                                    alpha),
                           "distribution": bound.gate_distribution(g, n_dist),
                           "strata": {t0: {"events": cell[f"stratum_{t0}"]["events"],
                                           "captures": cell[f"stratum_{t0}"]["captures"]}
                                      for t0 in bound.T0_NS if f"stratum_{t0}" in cell}}
        clock[name] = bound.gate_clocks({t0: cell[f"stratum_{t0}"] for t0 in bound.T0_NS
                                         if f"stratum_{t0}" in cell}, group["route"])

    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    rate_ok = all(v["complete"]["verdict"] == "pass" and v["rate"]["verdict"] == "pass" for v in rate.values())
    rejections = [f"{name}:{test}" for name, v in transport.items() for test, r in v["primary"].items()
                  if r["verdict"] != "pass"]
    dist_rejections = [f"{name}:{stream}" for name, v in transport.items()
                       for stream, r in v["distribution"].items() if r["verdict"] != "pass"]
    delivery_ok = all(v["delivery"]["verdict"] == "pass" for v in clock.values())
    physical_fail = sorted(f"{name}:{key}" for name, v in clock.items() for key, r in v["physical"].items()
                           if r["verdict"] != "pass")
    unexpected = sorted(f"{name}:{key}" for name, v in clock.items() for key, r in v["physical"].items()
                        if not r["as_predicted"])
    return {
        "RATE_RESULTS": {"status": status(rate_ok), "groups": rate},
        "TRANSPORT_RESULTS": {"status": status(not rejections and not dist_rejections),
                              "primary_rejections": rejections, "distribution_rejections": dist_rejections,
                              "groups": transport},
        "CLOCK_RESULTS": {"delivery_status": status(delivery_ok),
                          "physical_status": "FAIL" if physical_fail else "PASS",
                          "physical_failures": physical_fail, "not_as_predicted": unexpected, "groups": clock},
        "THEORY_STATUS": {"status": theory["status"], "accuracy": theory["accuracy"],
                          "qualified_roster": theory["qualified_roster"], "screen": theory["screen"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("steps", nargs="+", choices=("policy", "results"))
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


if __name__ == "__main__":
    main()
