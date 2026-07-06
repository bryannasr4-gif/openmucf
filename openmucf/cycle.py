"""openmucf.cycle -- differentiable muCF cycle-kinetics ODE network (Phase 2.2).

Implements the v1 network of MODEL_SPEC.md sec. 3 in JAX/diffrax. Six components:
  dynamical muonic-atom states  x_dmu, x_tmu(F=1), x_tmu(F=0)
  accumulators                  N_fus (the observable X_mu), stuck, dec
The dt-mu molecule is adiabatically eliminated (fast-fusion limit): ``lambda_form^F`` already
denotes the formation-limited rate, so the 7-decade stiffness from lambda_f ~ 1e12 is removed.

Conserved invariant: x_dmu + x_tmu1 + x_tmu0 + stuck + dec = 1 (N_fus is an event counter).

Gate V1: in the single-pool limit (lf0 == lf1, muon started in the tmu pool) N_fus(inf) must
reproduce analytic.fusions_per_muon(omega_s_eff, lambda_form) to < 1%.
"""

from __future__ import annotations

import diffrax
import jax.numpy as jnp

STATE_LABELS = ("x_dmu", "x_tmu1", "x_tmu0", "N_fus", "stuck", "dec")


def _field(t, y, args):
    x_dmu, x_tmu1, x_tmu0 = y[0], y[1], y[2]
    lam0, lam_dt, lam10, lf1, lf0, ose = args
    F = lf1 * x_tmu1 + lf0 * x_tmu0  # fusion (= formation) flux out of the tmu pool
    recyc = (1.0 - ose) * F  # muons that survive sticking re-form tmu (3/4 F=1, 1/4 F=0)
    dx_dmu = -(lam_dt + lam0) * x_dmu
    dx_tmu1 = lam_dt * x_dmu - (lam10 + lf1 + lam0) * x_tmu1 + 0.75 * recyc
    dx_tmu0 = lam10 * x_tmu1 - (lf0 + lam0) * x_tmu0 + 0.25 * recyc
    dN_fus = F
    dstuck = ose * F
    ddec = lam0 * (x_dmu + x_tmu1 + x_tmu0)
    return jnp.stack([dx_dmu, dx_tmu1, dx_tmu0, dN_fus, dstuck, ddec])


def solve_cycle(
    lambda_0,
    lambda_dt,
    lambda_10,
    lambda_form1,
    lambda_form0,
    omega_s_eff,
    y0=None,
    t1=3.0e-5,
    rtol=1e-9,
    atol=1e-12,
    max_steps=1_000_000,
):
    """Integrate the cycle to ``t1`` (long enough that the muon is gone). Returns the diffrax solution."""
    if y0 is None:
        y0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # muon enters on deuterium
    args = (lambda_0, lambda_dt, lambda_10, lambda_form1, lambda_form0, omega_s_eff)
    return diffrax.diffeqsolve(
        diffrax.ODETerm(_field),
        diffrax.Kvaerno5(),
        t0=0.0,
        t1=t1,
        dt0=1e-10,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=max_steps,
    )


def fusions_per_muon_ode(lambda_0, lambda_dt, lambda_10, lambda_form1, lambda_form0, omega_s_eff, **kw):
    """X_mu = N_fus(t1) from the full ODE network."""
    sol = solve_cycle(lambda_0, lambda_dt, lambda_10, lambda_form1, lambda_form0, omega_s_eff, **kw)
    return sol.ys[-1, 3]


def conservation_residual(sol):
    """x_dmu + x_tmu1 + x_tmu0 + stuck + dec - 1 at the final time (should be ~0)."""
    y = sol.ys[-1]
    return float(y[0] + y[1] + y[2] + y[4] + y[5] - 1.0)


def params_from_conditions(rates, T, phi, c_t, omega_s_eff=None, use_legacy_sticking=False):
    """Assemble cycle rates from the ledger + physical conditions (T [K], density phi, tritium fraction c_t).

    Documented v1 density scalings: transfer ~ phi*c_t, spin-flip ~ phi, formation ~ phi (inside
    formation.lambda_dtmu). omega_s_eff defaults to omega_s0*(1-R_col) from the ledger.
    """
    from . import formation
    from .analytic import effective_sticking
    from .rates import omega_fraction

    lambda_0 = rates.value("lambda_mu_decay")
    lambda_dt = rates.value("lambda_dt_transfer") * phi * c_t
    lambda_10 = rates.value("lambda_10_spinflip") * phi
    lf0 = formation.lambda_dtmu(T, phi, 0)
    lf1 = formation.lambda_dtmu(T, phi, 1)
    if omega_s_eff is None:
        os0 = omega_fraction(rates["omega_s0_legacy" if use_legacy_sticking else "omega_s0"])
        omega_s_eff = effective_sticking(os0, rates.value("R_col"))
    return dict(
        lambda_0=lambda_0,
        lambda_dt=lambda_dt,
        lambda_10=lambda_10,
        lambda_form1=lf1,
        lambda_form0=lf0,
        omega_s_eff=omega_s_eff,
    )


def fusions_per_muon_from_conditions(rates, T, phi, c_t, **kw):
    """One-call X_mu from (T, phi, c_t), using the ledger and the v1 formation model."""
    return fusions_per_muon_ode(**params_from_conditions(rates, T, phi, c_t, **kw))
