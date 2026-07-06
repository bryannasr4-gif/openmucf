"""Tests for the differentiable cycle ODE network (Phase 2.2), incl. the V1 gate."""

import jax.numpy as jnp

from openmucf import analytic as A
from openmucf import cycle, formation, load_rates


def test_v1_gate_ode_matches_analytic_under_1pct():
    """GATE V1: single-pool limit -> ODE N_fus(inf) == analytic X_mu to < 1%."""
    lam0, lam_form, ose = 4.552e5, 1.2e8, 0.005
    y0 = jnp.array([0.0, 0.75, 0.25, 0.0, 0.0, 0.0])  # start in tmu pool, hyperfine-independent
    x_ode = float(cycle.fusions_per_muon_ode(lam0, 0.0, 1e9, lam_form, lam_form, ose, y0=y0))
    x_an = float(A.fusions_per_muon(ose, lam_form, lam0))
    assert abs(x_ode - x_an) / x_an < 0.01, f"V1 fail: ODE {x_ode} vs analytic {x_an}"


def test_v1_gate_holds_across_conditions():
    for lam_form, ose in [(0.8e8, 0.0045), (1.5e8, 0.0057), (3.0e8, 0.003)]:
        y0 = jnp.array([0.0, 0.75, 0.25, 0.0, 0.0, 0.0])
        x_ode = float(cycle.fusions_per_muon_ode(4.552e5, 0.0, 1e9, lam_form, lam_form, ose, y0=y0))
        x_an = float(A.fusions_per_muon(ose, lam_form, 4.552e5))
        assert abs(x_ode - x_an) / x_an < 0.01


def test_probability_conserved():
    sol = cycle.solve_cycle(4.552e5, 2.8e8, 7e8, 3e8, 1.5e8, 0.005)
    assert abs(cycle.conservation_residual(sol)) < 1e-4


def test_formation_monotone_in_T():
    vals = [float(formation.lambda_dtmu(T, 1.0, 0)) for T in (50.0, 100.0, 300.0, 500.0, 800.0)]
    assert all(b > a for a, b in zip(vals, vals[1:], strict=False))  # strictly rising (V_yamashita shape)


def test_realistic_xmu_is_sane():
    rt = load_rates()
    x = float(cycle.fusions_per_muon_from_conditions(rt, T=500.0, phi=1.2, c_t=0.5))
    assert 50 <= x <= 200


def test_xmu_rises_with_temperature():
    rt = load_rates()
    x300 = float(cycle.fusions_per_muon_from_conditions(rt, 300.0, 1.2, 0.5))
    x700 = float(cycle.fusions_per_muon_from_conditions(rt, 700.0, 1.2, 0.5))
    assert x700 > x300
