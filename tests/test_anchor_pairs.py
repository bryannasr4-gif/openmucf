"""Condition-paired cycling-rate / sticking anchors.

The decay-only yield cap is ``lambda_c / lambda_0``, so it is a CONDITION-dependent number, never a
universal one. The ledger carries two condition-tagged anchors from the same SIN programme:

* ``lambda_c_liquid``     -- the LIQUID maximum (23 K, phi~1.2), 1.45e8 s^-1, measured yield ~113
* ``lambda_c_solid_12K``  -- the 12 K SOLID non-equilibrated c_t=0.4 point, 1.93e8 s^-1, yield 124+-10

They are **paired with their own sticking values** and are not independently selectable: the condition
that bought the faster cycle also carried higher measured sticking, so the yield moved only 113 -> 124.
These tests lock that pairing so a future edit cannot silently combine a best-case rate with a
best-case sticking drawn from a different experiment.
"""

from __future__ import annotations

import math

from openmucf.analytic import fusions_per_muon
from openmucf.rates import load_rates

# Published values (Crowe LBL-23816 1987 abstract; Breunlich PRL 58, 329 (1987) abstract).
PUBLISHED_SOLID_YIELD = (124.0, 10.0)  # 124 +- 10 fusions per muon at the 12 K solid anchor


def test_solid_anchor_pair_reproduces_published_yield():
    """The closed form reproduces the SIN solid anchor from its OWN paired (lambda_c, omega_s_eff)."""
    r = load_rates()
    lc = r.value("lambda_c_solid_12K")
    ose = r.value("omega_s_eff_solid_12K") / 100.0
    got = float(fusions_per_muon(ose, lc))
    centre, band = PUBLISHED_SOLID_YIELD
    assert centre - band <= got <= centre + band, (
        f"solid pair gives {got}, outside published {PUBLISHED_SOLID_YIELD}"
    )


def test_decay_only_cap_is_condition_dependent():
    """cap = lambda_c/lambda_0 differs between the two anchors -- 319 is NOT a universal cap."""
    r = load_rates()
    l0 = r.value("lambda_mu_decay")
    cap_liquid = 1.45e8 / l0  # the liquid maximum quoted in FINDINGS sec.3
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
    # the solid rate is paired to a sticking measured at the SAME condition
    assert r["omega_s_eff_solid_12K"].phase == "solid"
    assert r["omega_s_eff_solid_12K"].source_bibkey == solid.source_bibkey


def test_faster_cycle_did_not_buy_a_proportional_yield_gain():
    """The physical point of the pairing: +33% in lambda_c bought only ~+10% in measured yield,
    because the same condition raised sticking. Mixing a best-case rate with a best-case sticking
    from a different condition would overstate the yield."""
    r = load_rates()
    lc_s = r.value("lambda_c_solid_12K")
    ose_s = r.value("omega_s_eff_solid_12K") / 100.0
    ose_liquid = 0.45 / 100.0  # Breunlich liquid intrinsic sticking, a DIFFERENT condition
    paired = float(fusions_per_muon(ose_s, lc_s))
    mismatched = float(fusions_per_muon(ose_liquid, lc_s))  # best rate + best sticking (invalid)
    assert paired < mismatched
    # measured: 124.1 paired vs 145.8 mismatched -- combining the solid-anchor rate with the LIQUID
    # sticking overstates the yield by ~17%, which is the error this pairing lock exists to prevent
    assert mismatched / paired > 1.15, "mixing conditions materially overstates the yield"
