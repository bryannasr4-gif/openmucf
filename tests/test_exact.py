"""Exact linear-algebra oracle for the cycle network -- gates the diffrax integrator.

The network is linear constant-coefficient, so ``openmucf.exact`` solves it in closed form
(numpy.linalg.solve + jax.scipy.linalg.expm). These tests prove the oracle is consistent with the
ODE integrand (M rebuilt from ``_field``), that the ODE endpoint matches the oracle to <= 1e-8, that
conservation closes to 1e-12, that t1=3e-5 truncates negligibly, that the solver has step headroom,
and that the Maxwell-average quadrature is grid-converged.
"""

import jax.numpy as jnp
import numpy as np

from openmucf import cycle, exact, formation, load_rates

# DECIDED oracle grid (T [K], phi, c_t): canonical + a low-T, a high-T, and the MuFusE peak.
_GRID = [(300.0, 1.2, 0.5), (100.0, 1.2, 0.5), (800.0, 1.2, 0.5), (300.0, 2.4, 0.9)]


def _M_from_field(args):
    """Rebuild the 3x3 generator numerically from cycle._field via three unit-vector evaluations."""
    M = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(8)
        e[j] = 1.0
        M[:, j] = np.asarray(cycle._field(0.0, e, args))[:3]
    return M


def _grid_cases(rates):
    """Yield (label, args) over the grid x {channels off, on (hand-forced)} x {recapture off, q_1s=0.7}.

    The ledger ttmu row is blocked (0.0), so the channels-ON cases hand-force representative
    loss-channel rates -- this exercises the loss accumulators against the oracle, not the ledger.
    """
    for (T, phi, c_t) in _GRID:
        for chan in (False, True):
            for q in (None, 0.7):
                p = cycle.params_from_conditions(rates, T, phi, c_t, q_1s=q)
                if chan:
                    p["lambda_tt"], p["omega_tt"], p["lambda_he"] = 1.0e6, 0.14, 1.0e5 * c_t
                yield (T, phi, c_t, chan, q), exact.pack_args(**p)


def _solve(args):
    """Full 8-state ODE solution at the default t1 for a 10-tuple of args."""
    return cycle.solve_cycle(
        args[0], args[1], args[2], args[3], args[4], args[5],
        lambda_tt=args[6], omega_tt=args[7], lambda_he=args[8], f_d=args[9],
    )


def test_field_matrix_consistency():
    """The closed-form generator M equals the M rebuilt from _field, bit-for-bit (keeps them in sync)."""
    rates = load_rates()
    for label, args in _grid_cases(rates):
        Ma = exact.field_matrix(args)
        Mf = _M_from_field(args)
        assert np.array_equal(Ma, Mf), (label, float(np.max(np.abs(Ma - Mf))))


def test_ode_matches_exact_oracle():
    """GATE: the diffrax ODE endpoint matches the exact finite-time oracle to <= 1e-8 (relative) on the
    error-controlled accumulators (N_fus, stuck, dec) over the decided grid."""
    rates = load_rates()
    worst = 0.0
    for label, args in _grid_cases(rates):
        ode = np.asarray(_solve(args).ys[-1])
        ex = exact.finite_totals(args, 3.0e-5)
        for key, idx in (("X_mu", 3), ("stuck", 4), ("dec", 5)):
            rel = abs(float(ode[idx]) - ex[key]) / abs(ex[key])
            worst = max(worst, rel)
            assert rel <= 1e-8, (label, key, rel)
    print(f"\nODE-vs-exact-oracle: worst relative residual over grid = {worst:.6e} (gate 1e-8)")


def test_exact_conservation_closure():
    """Oracle self-consistency: x_sum(t1) + stuck + dec + loss_tt + loss_he == 1 to 1e-12."""
    rates = load_rates()
    worst = 0.0
    for label, args in _grid_cases(rates):
        resid = abs(exact.finite_totals(args, 3.0e-5)["closure"] - 1.0)
        worst = max(worst, resid)
        assert resid < 1e-12, (label, resid)
    print(f"\nexact conservation closure: worst |closure - 1| = {worst:.3e} (gate 1e-12)")


def test_t1_truncation_bound():
    """t1 = 3e-5 truncates negligibly: |exact(inf) - exact(t1)| / exact(inf) < 1e-5 across the grid."""
    rates = load_rates()
    worst = 0.0
    for label, args in _grid_cases(rates):
        x_inf = exact.steady_totals(args)["X_mu"]
        x_t1 = exact.finite_totals(args, 3.0e-5)["X_mu"]
        rel = abs(x_inf - x_t1) / abs(x_inf)
        worst = max(worst, rel)
        assert rel < 1e-5, (label, rel)
    print(f"\nt1=3e-5 truncation: worst |exact(inf) - exact(t1)| / exact(inf) = {worst:.3e} (gate 1e-5)")


def test_max_steps_headroom():
    """At the stiffest decided point (100 K, 2.4, 0.9, channels on) the adaptive solver stays well under
    the 1e6 step ceiling (num_steps < 5e5)."""
    rates = load_rates()
    p = cycle.params_from_conditions(rates, 100.0, 2.4, 0.9, q_1s=0.7)
    p["lambda_tt"], p["omega_tt"], p["lambda_he"] = 1.0e6, 0.14, 1.0e5 * 0.9
    sol = cycle.solve_cycle(
        p["lambda_0"], p["lambda_dt"], p["lambda_10"], p["lambda_form1"], p["lambda_form0"],
        p["omega_s_eff"], lambda_tt=p["lambda_tt"], omega_tt=p["omega_tt"], lambda_he=p["lambda_he"],
        f_d=p["f_d"],
    )
    n = int(sol.stats["num_steps"])
    assert n < 5e5, n
    print(f"\nnum_steps at (100 K, 2.4, 0.9, channels on) = {n} (ceiling 5e5)")


def test_quadrature_grid_doubling():
    """The Maxwell average (RAW, upstream of _CALIB) shifts < 0.05% under a grid doubling 800 -> 1600
    points at T in {20, 300, 900} x F in {0, 1}: the geometric quadrature is converged, so _CALIB
    anchors a converged average rather than shadowing a grid-truncation error."""
    def avg_on_grid(T, F, n):
        grid = jnp.geomspace(1.0e-4, 2.0, n)
        lam = formation.lambda_dtmu_energy(grid, F)
        f = formation._maxwell_pdf(grid, T)
        return float(formation._trapz(lam * f, grid))

    worst = 0.0
    for T in (20.0, 300.0, 900.0):
        for F in (0, 1):
            a800 = avg_on_grid(T, F, 800)
            a1600 = avg_on_grid(T, F, 1600)
            rel = abs(a1600 - a800) / abs(a800)
            worst = max(worst, rel)
            assert rel < 5e-4, (T, F, rel)
    print(f"\nquadrature grid-doubling (800 vs 1600): worst relative shift = {worst:.3e} (gate 5e-4)")
