"""openmucf.exact -- exact linear-algebra oracle for the cycle-kinetics network.

The cycle network (:func:`openmucf.cycle._field`) is **linear constant-coefficient** in the three
dynamical muonic-atom states ``y_d = (x_dmu, x_tmu1, x_tmu0)``; the five accumulators
(``N_fus, stuck, dec, loss_tt, loss_he``) integrate linear functionals of ``y_d``. So the whole
network has a closed matrix-exponential solution, and this module computes it directly -- an exact
oracle that gates the diffrax integrator (see tests/test_exact.py: the ODE endpoint must match this
oracle to <= 1e-8, and conservation closes to 1e-12).

Let ``M`` (3x3) be the generator ``dy_d/dt = M y_d`` (built here in closed form; a test rebuilds it
numerically from ``_field`` via three unit-vector evaluations and asserts bit-exact agreement, so the
oracle and the ODE integrand can never silently diverge). With ``y_d(0) = (1, 0, 0)`` (one muon born
on deuterium):

* steady time-integral ``I_inf = integral_0^inf y_d dt = (-M)^{-1} y_d(0)`` (numpy.linalg.solve);
* finite-time state ``y_d(t1) = expm(M t1) y_d(0)`` and integral
  ``I(t1) = M^{-1} (expm(M t1) - I) y_d(0)`` (expm via jax.scipy.linalg);
* accumulator totals are linear functionals of the relevant integral: e.g. ``X_mu = [0, lf1, lf0].I``.

No new runtime dependency: the matrix exponential is ``jax.scipy.linalg.expm`` (x64 enabled at import
in ``openmucf/__init__``) and the linear solves are ``numpy.linalg.solve`` -- scipy is NOT declared.
"""

from __future__ import annotations

import numpy as np
from jax.scipy.linalg import expm as _expm

# One muon enters on deuterium: y_d(0) = (x_dmu, x_tmu1, x_tmu0) = (1, 0, 0).
_Y0 = np.array([1.0, 0.0, 0.0])


def pack_args(
    lambda_0,
    lambda_dt,
    lambda_10,
    lambda_form1,
    lambda_form0,
    omega_s_eff,
    lambda_tt=0.0,
    omega_tt=0.0,
    lambda_he=0.0,
    f_d=0.0,
):
    """Pack the cycle rates into the exact 10-tuple order used by ``cycle._field``.

    Accepts the keyword names emitted by ``cycle.params_from_conditions`` (so
    ``pack_args(**params_from_conditions(...))`` works directly)."""
    return (
        lambda_0, lambda_dt, lambda_10, lambda_form1, lambda_form0, omega_s_eff,
        lambda_tt, omega_tt, lambda_he, f_d,
    )


def field_matrix(args):
    """Closed-form 3x3 generator ``M`` (``dy_d/dt = M y_d``) for the dynamical block ``y_d``.

    Assembled by inspection of ``cycle._field`` and kept in sync with it: the arithmetic is written in
    the SAME operation order the field uses per unit vector, so a unit-vector rebuild from ``_field``
    matches this matrix bit-for-bit (test_field_matrix_consistency).
    """
    (lam0, lam_dt, lam10, lf1, lf0, ose, lam_tt, w_tt, lam_he, f_d) = args
    recyc1 = (1.0 - ose) * lf1  # surviving-sticking recycle flux per unit x_tmu1
    recyc0 = (1.0 - ose) * lf0  # ... per unit x_tmu0
    ttret = (1.0 - w_tt) * lam_tt  # tt-branch return flux per unit tmu occupancy
    d_dmu = -(lam_dt + lam0 + lam_he)
    return np.array(
        [
            [d_dmu, f_d * recyc1, f_d * recyc0],
            [
                lam_dt,
                -(lam10 + lf1 + lam0 + lam_tt) + 0.75 * ((1.0 - f_d) * recyc1 + ttret),
                0.75 * ((1.0 - f_d) * recyc0 + ttret),
            ],
            [
                0.0,
                lam10 + 0.25 * ((1.0 - f_d) * recyc1 + ttret),
                -(lf0 + lam0 + lam_tt) + 0.25 * ((1.0 - f_d) * recyc0 + ttret),
            ],
        ]
    )


def _functionals(args):
    """The linear accumulator functionals (row vectors over ``y_d``)."""
    (lam0, lam_dt, lam10, lf1, lf0, ose, lam_tt, w_tt, lam_he, f_d) = args
    f_fus = np.array([0.0, lf1, lf0])  # dN_fus/dt = f_fus . y_d
    f_dec = lam0 * np.array([1.0, 1.0, 1.0])  # ddec/dt
    f_tt = (w_tt * lam_tt) * np.array([0.0, 1.0, 1.0])  # dloss_tt/dt
    f_he = lam_he * np.array([1.0, 0.0, 0.0])  # dloss_he/dt
    return f_fus, f_dec, f_tt, f_he, ose


def _totals_from_integral(integral, args):
    """Accumulator totals from a time-integral of ``y_d`` (steady I_inf or finite I(t1))."""
    f_fus, f_dec, f_tt, f_he, ose = _functionals(args)
    x_mu = float(f_fus @ integral)
    return {
        "X_mu": x_mu,
        "stuck": float(ose * x_mu),  # same functional as X_mu, scaled by omega_s_eff
        "dec": float(f_dec @ integral),
        "loss_tt": float(f_tt @ integral),
        "loss_he": float(f_he @ integral),
    }


def steady_totals(args, y0=None):
    """Exact totals over ``t -> inf``: X_mu, stuck, dec, loss_tt, loss_he from ``I_inf = (-M)^{-1} y0``.

    ``y0`` is the dynamical-block initial condition (default = one muon on deuterium, (1, 0, 0)); pass
    ``(0, 0.75, 0.25)`` for the tmu-pool single-pool V1 gate.
    """
    y0 = _Y0 if y0 is None else np.asarray(y0, dtype=float)
    M = field_matrix(args)
    i_inf = np.linalg.solve(-M, y0)  # M I_inf = -y0  (since y_d(inf)=0)
    out = _totals_from_integral(i_inf, args)
    out["I_inf"] = i_inf
    return out


def finite_totals(args, t1, y0=None):
    """Exact state and totals at finite ``t1``: y_d(t1) = expm(M t1) y0, I(t1) = M^{-1}(expm(M t1)-I) y0.

    Also returns ``x_sum`` (= sum of the three dynamical occupancies at t1) and the conservation
    ``closure`` = x_sum + stuck + dec + loss_tt + loss_he, which equals ``sum(y0)`` (= 1 for the default
    single-muon start) to machine precision.
    """
    y0 = _Y0 if y0 is None else np.asarray(y0, dtype=float)
    M = field_matrix(args)
    exp_mt = np.asarray(_expm(M * t1), dtype=float)
    y_t1 = exp_mt @ y0
    integral = np.linalg.solve(M, (exp_mt - np.eye(3)) @ y0)
    out = _totals_from_integral(integral, args)
    out["y_d"] = y_t1
    out["I"] = integral
    x_sum = float(y_t1.sum())
    out["x_sum"] = x_sum
    out["closure"] = x_sum + out["stuck"] + out["dec"] + out["loss_tt"] + out["loss_he"]
    return out


def xmu_steady(args, y0=None):
    """Convenience: exact steady X_mu (fusions per muon) only."""
    return steady_totals(args, y0=y0)["X_mu"]
