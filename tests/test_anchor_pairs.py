"""The two condition-tagged cycling-rate anchors on a common basis: the total loss per cycle W.

The yield relation of Breunlich et al. (1987, p.330 Eq. (2), p.332) sees the total loss per cycle W, the
sticking plus the side-channel losses, so each anchor's W is 1/X_mu - lambda_0/lambda_c from its own published
yield and cycle rate. The decay-only yield cap ``lambda_c / lambda_0`` is a CONDITION-dependent number.
"""

from __future__ import annotations

import csv
import math

from openmucf.analytic import fusions_per_muon
from openmucf.rates import TARGETS_CSV, load_rates

# The 12 K solid anchor's published yield (Crowe LBL-23816 1987 abstract; rates.csv omega_s_eff_solid_12K).
PUBLISHED_SOLID_YIELD = (124.0, 10.0)


def _liquid_pair():
    with open(TARGETS_CSV, newline="", encoding="utf-8") as f:
        t = {r["target_id"]: r for r in csv.DictReader(f)}
    return float(t["V_petitjean_Xmu"]["value"]), float(t["V_breunlich_lambdac"]["value"])


def _w_liquid():
    x, lc = _liquid_pair()
    return 1.0 / x - load_rates().value("lambda_mu_decay") / lc


def test_liquid_pair_reproduces_its_yield_from_the_unrounded_total_loss():
    """W = 1/113 - lambda_0/1.45e8, unrounded, returns the measured 113 within 1e-4."""
    x, lc = _liquid_pair()
    assert abs(float(fusions_per_muon(_w_liquid(), lc)) - x) < 1e-4


def test_solid_pair_reproduces_its_published_yield():
    r = load_rates()
    got = float(fusions_per_muon(r.value("omega_s_eff_solid_12K") / 100.0, r.value("lambda_c_solid_12K")))
    centre, band = PUBLISHED_SOLID_YIELD
    assert round(got, 2) == 124.09
    assert centre - band <= got <= centre + band


def test_the_two_anchors_lose_the_same_fraction_per_cycle():
    """At the solid anchor's cycle rate, the liquid anchor's W gives the solid fill's yield to 0.13 %: on the
    common basis the yield rose from 113 to 124 through the cycle rate."""
    r = load_rates()
    lc_s = r.value("lambda_c_solid_12K")
    ratio = float(fusions_per_muon(_w_liquid(), lc_s)) / float(
        fusions_per_muon(r.value("omega_s_eff_solid_12K") / 100.0, lc_s)
    )
    assert round(ratio, 4) == 0.9987


def test_decay_only_cap_is_condition_dependent():
    """cap = lambda_c/lambda_0 differs between the two anchors -- 319 is NOT a universal cap."""
    r = load_rates()
    l0 = r.value("lambda_mu_decay")
    cap_liquid = _liquid_pair()[1] / l0
    cap_solid = r.value("lambda_c_solid_12K") / l0
    assert math.isclose(cap_liquid, 318.5, abs_tol=0.5)
    assert math.isclose(cap_solid, 424.0, abs_tol=0.5)
    assert cap_solid > cap_liquid


def test_anchors_are_condition_tagged_differently():
    """Both anchors carry distinct, explicit phase/validity tags so they cannot be silently mixed."""
    r = load_rates()
    liquid, solid = r["lambda_c_liquid"], r["lambda_c_solid_12K"]
    assert liquid.phase == "liquid" and solid.phase == "solid"
    assert liquid.validity_range != solid.validity_range
    assert "solid" in solid.validity_range.lower()
    assert r["omega_s_eff_solid_12K"].phase == "solid"
    assert r["omega_s_eff_solid_12K"].source_bibkey == solid.source_bibkey
