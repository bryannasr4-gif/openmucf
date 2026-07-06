"""Tests for the transparent energy balance (Phase 2.3)."""

import math

from openmucf.energy import EnergyChain


def test_scientific_breakeven_is_284():
    ch = EnergyChain()
    assert math.isclose(ch.breakeven_xmu_sci(), 5000.0 / 17.6, rel_tol=1e-9)
    assert 280 <= ch.breakeven_xmu_sci() <= 290


def test_q_sci_unity_at_scientific_breakeven():
    ch = EnergyChain()
    assert math.isclose(ch.Q_sci(ch.breakeven_xmu_sci()), 1.0, rel_tol=1e-9)


def test_net_electrical_breakeven_is_much_harder():
    ch = EnergyChain()
    # net-electrical breakeven is far above scientific breakeven AND the ~150 record
    assert ch.breakeven_xmu_net() > 1000
    assert ch.breakeven_xmu_net() > ch.breakeven_xmu_sci()


def test_record_yield_is_subunity_net_electrical():
    ch = EnergyChain()
    assert ch.Q_net_electrical(150.0) < 1.0


def test_blanket_multiplier_helps_monotonically():
    plain = EnergyChain()
    hybrid = EnergyChain(blanket_M=10.0)  # fission-hybrid-style multiplication
    assert hybrid.Q_net_electrical(150.0) > plain.Q_net_electrical(150.0)
