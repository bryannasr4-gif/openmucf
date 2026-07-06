"""Tests for the closed-form analytic backbone (Phase 2.1)."""

import math

import jax

from openmucf import analytic as A
from openmucf import load_rates


def test_effective_sticking():
    assert math.isclose(A.effective_sticking(0.00857, 0.35), 0.00857 * 0.65, rel_tol=1e-12)


def test_xmu_reproduces_kouchen_baseline_when_fed_their_eff_sticking():
    # Kou-Chen baseline: omega_s_eff = 0.557%, and a cycling rate giving N=112.6.
    x = A.fusions_per_muon(0.00557, 1.371e8)
    assert math.isclose(float(x), 112.6, rel_tol=0.02)  # within 2%


def test_xmu_in_canonical_liquid_range():
    x = A.fusions_per_muon(0.0050, 1.3e8)
    assert 100 <= float(x) <= 160


def test_breakeven_is_284():
    assert math.isclose(A.breakeven_xmu(), 5000.0 / 17.6, rel_tol=1e-9)
    assert 280 <= A.breakeven_xmu() <= 290


def test_energy_gain_subunity_today():
    # X_mu ~ 113, eta_conv ~ 0.4 (thermal) -> Q well below 1 at 5 GeV/muon
    q = A.energy_gain(113.0, eta_conv=0.4)
    assert float(q) < 1.0


def test_from_ledger_runs_and_is_reasonable():
    rt = load_rates()
    x = A.from_ledger(rt, phi=1.2, lambda_c_tilde=1.0e8)
    assert 80 <= float(x) <= 200


def test_differentiable_signs():
    # More sticking -> fewer fusions; faster cycling -> more fusions.
    dX_dsticking = jax.grad(lambda o: A.fusions_per_muon(o, 1.3e8))(0.005)
    dX_dlambda = jax.grad(lambda lc: A.fusions_per_muon(0.005, lc))(1.3e8)
    assert float(dX_dsticking) < 0
    assert float(dX_dlambda) > 0
