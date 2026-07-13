"""Tests for the resonant formation model: the disclosed 300 K anchor and the log-grid convergence."""

import jax.numpy as jnp
import pytest

import openmucf.formation as formation

# Disclosed 300 K anchor values (formation._CALIB re-anchor, 2026-07-08; see formation.py). The grid
# change from linear to geomspace re-derived _CALIB to reproduce THESE exactly, so no 300 K number moved.
A0 = 130059577.61950417  # lambda_dtmu(300, 1, F=0)
A1 = 53094361.62224904  # lambda_dtmu(300, 1, F=1)


def test_calib_anchor_preserved():
    """The 300 K anchor rates are preserved by the _CALIB re-derivation (I2: anchor honored, not tuned)."""
    assert float(formation.lambda_dtmu(300.0, 1.0, 0)) == pytest.approx(A0, rel=1e-9)
    assert float(formation.lambda_dtmu(300.0, 1.0, 1)) == pytest.approx(A1, rel=1e-9)


def test_lowT_grid_converged():
    """Doubling the geomspace grid changes lambda_dtmu(30 K, 1, F=0) by < 0.5%.

    On the OLD linear grid this was ~7.1% (grid-unconverged at low T, where the Maxwell weight
    concentrates below 0.01 eV); the log grid resolves that region.
    """
    base = float(formation.lambda_dtmu(30.0, 1.0, 0))
    orig = formation._EGRID
    try:
        formation._EGRID = jnp.geomspace(1.0e-4, 2.0, 1600)
        fine = float(formation.lambda_dtmu(30.0, 1.0, 0))
    finally:
        formation._EGRID = orig
    assert abs(fine - base) / abs(base) < 0.005


def test_egrid_is_log():
    """A geometric grid has a constant successive-point ratio (log-uniform spacing)."""
    r = formation._EGRID[1:] / formation._EGRID[:-1]
    assert float(jnp.max(r) - jnp.min(r)) < 1e-9


def test_formation_scope_warning():
    """A concrete off-anchor call (phi>1.45 or T<100 K) fires a one-shot RED-tier scope warning;
    an in-scope call does not warn."""
    import warnings

    formation._SCOPE_WARNED = False
    with pytest.warns(UserWarning, match="300 K-anchored placeholder"):
        formation.lambda_dtmu(300.0, 2.0, 0)

    # in-scope call (phi=1.2, T=300) must NOT warn (reset the one-shot flag first)
    formation._SCOPE_WARNED = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        formation.lambda_dtmu(300.0, 1.2, 0)  # would raise if a warning fired
    formation._SCOPE_WARNED = False  # leave the module flag clean for other tests


def test_scope_guard_is_jit_safe():
    """The off-anchor scope guard is skipped under jit tracing (never breaks a traced computation)."""
    import warnings

    import jax

    formation._SCOPE_WARNED = False
    jitted = jax.jit(lambda T, phi: formation.lambda_dtmu(T, phi, 0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val = float(jitted(30.0, 2.0))  # T<100 and phi>1.45: the guard must skip the tracer
    assert not any("300 K-anchored placeholder" in str(w.message) for w in caught)
    assert val > 0.0
    # the guard has no numeric effect: equals the eager value (suppress the eager warning first)
    formation._SCOPE_WARNED = True
    assert val == pytest.approx(float(formation.lambda_dtmu(30.0, 2.0, 0)))
    formation._SCOPE_WARNED = False
