"""openmucf.frontier -- inverse design: "what would have to be true" for a target yield / gain.

Two layers over the SAME physics as the forward map (``analytic`` + ``systems``), never new physics (I1):

  * CLOSED-FORM frontiers where exact algebra exists (no solver): :func:`r_required`,
    :func:`lambda_c_required`, :func:`frontier_lambda_c_R`. Pure-Python ``float`` (float64) arithmetic --
    byte-stable cross-architecture, so their values may ship at full precision (like
    ``systems.rosetta_table``). :func:`r_required` reproduces ``uq.breakeven_audit``'s R-floor identity
    bit-for-bit, hence the FINDINGS.md sec.3 "R >= 0.77" headline.
  * A SOLVER-BACKED general inverse: :func:`solve_inverse` uses ``optimistix.root_find`` (Newton) over the
    differentiable ``systems.q_net`` graph to solve for ONE free variable among ``FREE_VARS`` =
    (E_mu_GeV, R, lambda_c, eta_acc) at a ``q_net`` target with the others fixed. In this analytically
    invertible model every case also has a closed form, which is exactly why the two paths can be gated to
    agree to < 1e-9. Per the Wave-2 solver-printing rule, any solver output that ships is quantised to <= 6
    significant figures; this module keeps the shipped digits closed-form (byte-stable) and gate-tests the
    solver against them, so nothing byte-diffed depends on iterative-solver noise.

INVERSE-DESIGN ONLY (I1/I8). Given a target it reports the *required* parameter value -- exactly the
requirement form of FINDINGS.md sec.3 ("R >= 0.77 is required"). It renders NO verdict on any external
projection and encodes NO scenario/verdict registry; that is fenced OUT (WAVE2 sec.3.2). A required value
that lands outside [0, 1] (R, eta_acc) or at ``math.inf`` (lambda_c) is the honest readout that the target
is unreachable in that variable -- reported, never clamped.

Not part of the eager-import surface (a submodule like ``calibrate`` / ``systems`` / ``forecast``);
reached as ``openmucf.frontier`` or ``from openmucf import frontier``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import jax.numpy as jnp
import optimistix as optx

from .analytic import effective_sticking, fusions_per_muon
from .constants import LAMBDA_0
from .systems import SystemChain, q_net

# Nominal initial-sticking fraction: the Kamimura theory value 0.857 % expressed as a BARE FRACTION
# (matching analytic.effective_sticking's convention). Single-sourced to the same nominal that
# uq.NOMINAL["omega_s0_pct"] (= 0.857) drives, so r_required reproduces uq.breakeven_audit exactly.
OMEGA_S0_DEFAULT = 0.00857

# Nominal fixed-kinetics operating point for the solver worked examples (uq_priors.csv nominals):
# collisional reactivation R_col and the measured liquid-condition cycling rate.
R_NOMINAL = 0.35
LAMBDA_C_NOMINAL = 1.30e8

# The one free variable the solver may invert for (same basis as uq/calibrate: lambda_c is the ACTUAL rate).
FREE_VARS = ("E_mu_GeV", "R", "lambda_c", "eta_acc")

# Documented Newton starts (one per free variable); smooth 1-D roots, so these converge across the grid.
_Y0_DEFAULT = {"E_mu_GeV": 5.0, "eta_acc": 0.30, "R": 0.5, "lambda_c": 1.5e8}

# Positive, unbounded-above variables that span orders of magnitude across targets: Newton is run in
# LOG space (value = exp(z)) so a step can never cross into the unphysical value <= 0 region where the
# 1/E_mu or lambda_0/lambda_c terms flip sign and diverge. R and eta_acc are O(1) and solved directly.
_LOG_VARS = ("E_mu_GeV", "lambda_c")


# --------------------------------------------------------------------------------------------------
# Closed-form frontiers (pure-Python float64; byte-stable, may print at full precision)
# --------------------------------------------------------------------------------------------------
def r_required(x_mu_target, lambda_c, omega_s0=OMEGA_S0_DEFAULT, lambda_0=LAMBDA_0):
    """Reactivation ``R`` required to reach ``x_mu_target`` fusions/muon at cycling rate ``lambda_c``.

    The FINDINGS.md sec.3 identity generalised (inverting ``X = 1/(omega_s0*(1-R) + lambda_0/lambda_c)``):

        R = 1 - (1/X - lambda_0/lambda_c) / omega_s0

    ``lambda_c`` may be ``math.inf`` -- the decay-free limit that gives the density-independent R floor
    (``R = 1 - (1/X)/omega_s0``). ``omega_s0`` is a bare fraction (default = Kamimura 0.857 %). The result
    can exceed 1 or go negative: the honest readout that the target is unreachable / trivially reachable at
    the given ``(lambda_c, omega_s0)``. Pure ``float`` arithmetic -> byte-stable cross-architecture.
    """
    inv_lam = 0.0 if math.isinf(lambda_c) else lambda_0 / lambda_c
    return 1.0 - (1.0 / x_mu_target - inv_lam) / omega_s0


def lambda_c_required(x_mu_target, R, omega_s0=OMEGA_S0_DEFAULT, lambda_0=LAMBDA_0):
    """Cycling rate ``lambda_c`` required to reach ``x_mu_target`` at reactivation ``R``.

    Inverse of the same forward map: ``lambda_c = lambda_0 / (1/X - omega_s_eff)`` with
    ``omega_s_eff = omega_s0*(1-R)``. Returns ``math.inf`` when the sticking floor alone already caps the
    yield at or below the target (``1/X <= omega_s_eff`` i.e. ``x_mu_target >= 1/omega_s_eff``): no finite
    cycling rate can reach it -- the honest "unreachable at this R" readout. Pure ``float`` arithmetic.
    """
    ose = effective_sticking(omega_s0, R)
    denom = 1.0 / x_mu_target - ose
    return math.inf if denom <= 0.0 else lambda_0 / denom


def frontier_lambda_c_R(x_mu_target, grid, omega_s0=OMEGA_S0_DEFAULT, lambda_0=LAMBDA_0):
    """The ``(lambda_c, R_required)`` frontier curve for ``x_mu_target`` over ``grid``.

    For each ``lambda_c`` in ``grid``, the reactivation ``R`` that would be needed to hit ``x_mu_target``.
    Returns a list of ``(lambda_c, R)`` ``float`` pairs -- pure Python, byte-stable, deterministic. This is
    the closed-form spine plotted in ``figures/frontier.png``: R falls monotonically as ``lambda_c`` rises
    (``d R_required / d lambda_c < 0``), because a faster cycle lets decay-loss carry more of the budget.
    """
    return [(float(lc), r_required(x_mu_target, float(lc), omega_s0, lambda_0)) for lc in grid]


# --------------------------------------------------------------------------------------------------
# Solver-backed general inverse (optimistix Newton over the differentiable systems.q_net graph)
# --------------------------------------------------------------------------------------------------
def _q_net_of_free(free_var, value, chain, omega_s0, R, lambda_c):
    """``q_net`` as a differentiable function of the single ``free_var`` (all others fixed).

    ``x_mu`` is reconstructed from the kinetics via ``analytic.fusions_per_muon`` (so R and lambda_c enter
    the ENERGY gain through the yield); ``E_mu_GeV`` and ``eta_acc`` are ``SystemChain`` knobs at fixed
    ``x_mu``. Pure ``jax.numpy`` in the free variable -> Newton gets its derivative by autodiff.
    """
    if free_var == "E_mu_GeV":
        chain = replace(chain, E_mu_GeV=value)
        x_mu = fusions_per_muon(effective_sticking(omega_s0, R), lambda_c)
    elif free_var == "eta_acc":
        chain = replace(chain, eta_acc=value)
        x_mu = fusions_per_muon(effective_sticking(omega_s0, R), lambda_c)
    elif free_var == "R":
        x_mu = fusions_per_muon(effective_sticking(omega_s0, value), lambda_c)
    elif free_var == "lambda_c":
        x_mu = fusions_per_muon(effective_sticking(omega_s0, R), value)
    else:  # pragma: no cover - guarded by solve_inverse
        raise ValueError(f"free_var must be one of {FREE_VARS}, got {free_var!r}")
    return q_net(chain, x_mu)


def solve_inverse(
    target_q_net,
    free_var,
    *,
    chain=None,
    omega_s0=OMEGA_S0_DEFAULT,
    R=R_NOMINAL,
    lambda_c=LAMBDA_C_NOMINAL,
    y0=None,
    rtol=1e-12,
    atol=1e-12,
    max_steps=256,
):
    """Solve ``q_net(...) == target_q_net`` for the single ``free_var`` by Newton root-find.

    ``optimistix.root_find`` (Newton) is driven over the differentiable ``systems.q_net`` graph. The fixed
    variables come from ``chain`` (defaults to the v1 ``SystemChain``) plus ``R`` / ``lambda_c`` (the fixed
    kinetics; ignored when they are themselves the free variable). ``y0`` overrides the documented Newton
    start for ``free_var``. Returns the required value as a Python ``float``.

    In this analytically invertible model the answer also has a closed form (:func:`r_required` /
    :func:`lambda_c_required` / the linear ``E_mu_GeV`` / ``eta_acc`` inverses); the two agree to < 1e-9,
    which is the consistency gate. The solver exists as the GENERAL capability (it would carry over to a
    future non-invertible extended graph); nothing byte-diffed depends on its iterative output.
    """
    if free_var not in FREE_VARS:
        raise ValueError(f"free_var must be one of {FREE_VARS}, got {free_var!r}")
    chain = SystemChain() if chain is None else chain
    y_start = _Y0_DEFAULT[free_var] if y0 is None else y0
    in_log = free_var in _LOG_VARS

    def residual(y, args):
        value = jnp.exp(y) if in_log else y
        return _q_net_of_free(free_var, value, chain, omega_s0, R, lambda_c) - target_q_net

    z0 = math.log(y_start) if in_log else y_start
    solver = optx.Newton(rtol=rtol, atol=atol)
    sol = optx.root_find(
        residual, solver, jnp.asarray(z0, dtype=jnp.float64), max_steps=max_steps, throw=False
    )
    if sol.result != optx.RESULTS.successful:
        raise RuntimeError(
            f"Newton did not converge for free_var={free_var!r} at target_q_net={target_q_net} "
            f"(optimistix result: {sol.result}). The root may be ill-conditioned -- e.g. a lambda_c "
            f"target within the yield-saturation cliff (x_mu -> 1/omega_s_eff) where d q_net/d lambda_c "
            f"-> 0. Pass a closer y0 or invert a different free variable."
        )
    return float(jnp.exp(sol.value)) if in_log else float(sol.value)


def q_net_at(
    free_var, value, *, chain=None, omega_s0=OMEGA_S0_DEFAULT, R=R_NOMINAL, lambda_c=LAMBDA_C_NOMINAL
):
    """Forward ``q_net`` at a single ``free_var`` value (the map :func:`solve_inverse` inverts).

    Convenience wrapper used to verify ``q_net(inverse solution) == target`` and to compute the nominal
    operating point. Returns a Python ``float``.
    """
    chain = SystemChain() if chain is None else chain
    return float(_q_net_of_free(free_var, value, chain, omega_s0, R, lambda_c))
