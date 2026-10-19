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
        raise SystemExit("results: the harvest summary is not committed yet")


if __name__ == "__main__":
    main()
