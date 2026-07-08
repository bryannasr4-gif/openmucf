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


def test_eta_threads_through_params():
    """params_from_conditions(eta=5.0) scales both formation rates by exactly 5x vs eta=1."""
    import pytest

    rt = load_rates()
    p1 = cycle.params_from_conditions(rt, 300.0, 1.2, 0.5, eta=1.0)
    p5 = cycle.params_from_conditions(rt, 300.0, 1.2, 0.5, eta=5.0)
    assert p5["lambda_form1"] == pytest.approx(5.0 * p1["lambda_form1"], rel=1e-12)
    assert p5["lambda_form0"] == pytest.approx(5.0 * p1["lambda_form0"], rel=1e-12)


def test_wsn_norm_excludes_loss_accumulators_bit_exact():
    """PIN (Fable amendment 2026-07-08, §3.4; diagnostic de-pinned same day, hotfix ws-fix-gradient-pin):
    step-error is controlled over the 6 v1 states only, so the channels-OFF 8-component network reduces
    to the v1 reference BIT-FOR-BIT (reduction gate G-N1 at PURE atol 1e-9, rtol=0); reverting to
    diffrax's default 8-component norm regresses N_fus to ~5e-9 and fails that gate. The through-the-ODE
    diagnostic (uq.cross_check_gradient) is machine noise ~3e-13 whose leading digit drifts across
    jax/numpy builds (2.9e-13 locked env vs 2.8e-13 on unpinned CI 3.12), so it is BOUNDED (< 1e-11),
    not byte-pinned; FINDINGS.md emits the same bound (see scripts/generate_findings.py)."""
    import json
    from pathlib import Path

    import numpy as np

    from openmucf.uq import cross_check_gradient

    ref = json.loads((Path(__file__).parent / "cycle_v1_reference.json").read_text())
    args6 = ref["args6"]
    ref6 = np.array(ref["ys"])[:, :6]
    got6 = np.array([np.asarray(cycle.solve_cycle(*args6, t1=float(t)).ys[-1])[:6] for t in ref["ts"]])
    # G-N1 as PURE atol (rtol=0): only the v1-sliced norm makes this pass (default 8-norm -> ~5e-9).
    assert np.max(np.abs(got6 - ref6)) < 1e-9
    # Through-the-ODE diagnostic: autodiff-vs-analytic agreement stays at the machine-noise floor
    # (~3e-13 measured; its leading digit is env-dependent, so bound it ~2 decades above the floor).
    assert cross_check_gradient()["rel_diff"] < 1e-11
