"""openmucf.cycle -- differentiable muCF cycle-kinetics ODE network (Phase 2.2).

Implements the v1 network of MODEL_SPEC.md sec. 3 in JAX/diffrax. Six components:
  dynamical muonic-atom states  x_dmu, x_tmu(F=1), x_tmu(F=0)
  accumulators                  N_fus (the observable X_mu), stuck, dec
The dt-mu molecule is adiabatically eliminated (fast-fusion limit): ``lambda_form^F`` already
denotes the formation-limited rate, so the 7-decade stiffness from lambda_f ~ 1e12 is removed.

Conserved invariant: x_dmu + x_tmu1 + x_tmu0 + stuck + dec + loss_tt + loss_he = 1
(N_fus is an event counter, not an occupancy). The two loss accumulators are the WS-N absorbing
channels (ttmu side-branch, He-3 scavenging); with their rates at 0 they stay identically zero and the
network reduces bit-exactly to the v1 six-state system (reduction gate G-N1).

Gate V1: in the single-pool limit (lf0 == lf1, muon started in the tmu pool) N_fus(inf) must
reproduce analytic.fusions_per_muon(omega_s_eff, lambda_form) to < 1%.
"""

from __future__ import annotations

import inspect as _inspect  # stdlib; used only to capture diffrax's default error norm below

import diffrax
import jax.numpy as jnp

# indices 0..5 are the v1 states (unchanged); 6,7 are the WS-N absorbing loss accumulators (append-only,
# so ``sol.ys[-1, 3]`` stays N_fus and every existing caller keeps working).
STATE_LABELS = ("x_dmu", "x_tmu1", "x_tmu0", "N_fus", "stuck", "dec", "loss_tt", "loss_he")

# --- WS-N adaptive-step error-norm scope (Fable amendment 2026-07-08, WAVE1_EXECUTION_SPEC.md §3.4) ---
# diffrax's PIDController averages its RMS step-error norm over ALL state components. The two WS-N
# absorbing OUTPUT accumulators (loss_tt, loss_he) would change that average (/8 vs /6) and shift v1's
# exact step sequence at ~1e-13 -- enough to move the one FINDINGS number computed THROUGH the ODE
# (uq.cross_check_gradient: 2.9e-13 -> 3.2e-13) and to break the channels-OFF bit-exact reduction.
# We therefore control error over exactly the six v1 states, reusing diffrax's OWN default norm
# (captured below -- NOT imported from optimistix, so the pinned dependency set is untouched) sliced to
# those states. Nothing in _field reads loss_tt/loss_he, so their accuracy rides on the already
# error-controlled driving fluxes; excluding pure output-only states from step control is standard and
# a true no-op on the base integration. This does NOT re-decide v1: N_fus/stuck/dec (3/4/5) stay in the
# norm exactly as in v1 -- we only decline to EXTEND error control to the new outputs. Numerics only;
# no physics input is tuned (I2). Result: channels-OFF reduces to v1 bit-for-bit (G-N1 pure atol 1e-9)
# and every locked number stays byte-identical. Locked by test_wsn_norm_excludes_loss_accumulators_bit_exact.
_N_V1_STATES = 6
# Introspection (not a formula copy) is deliberate: it is guaranteed bit-identical to whatever this
# diffrax version uses, so the locked step sequence cannot drift by a re-implementation ulp. The cost is
# reliance on PIDController's signature; if a future diffrax renames/moves the `norm` default, fail LOUD
# at import with instructions rather than with an opaque KeyError (cross-vendor review hardening, 2026-07-08).
try:
    _DIFFRAX_DEFAULT_NORM = _inspect.signature(diffrax.PIDController.__init__).parameters["norm"].default
    if not callable(_DIFFRAX_DEFAULT_NORM):
        raise TypeError(f"expected a callable norm default, got {_DIFFRAX_DEFAULT_NORM!r}")
except (KeyError, TypeError) as exc:  # pragma: no cover - trips only on a diffrax API change
    raise ImportError(
        "diffrax.PIDController's `norm` default could not be captured for the v1-sliced error norm "
        "(diffrax API changed?). Update _v1_error_norm in openmucf/cycle.py to the new default norm; "
        "it MUST match diffrax's own default bit-for-bit or the locked v1 step sequence will drift "
        "(see test_wsn_norm_excludes_loss_accumulators_bit_exact)."
    ) from exc


def _v1_error_norm(y):
    """Step-error norm over the v1 states only; trailing WS-N loss accumulators (y[6:]) are excluded."""
    return _DIFFRAX_DEFAULT_NORM(y[:_N_V1_STATES])


def _field(t, y, args):
    x_dmu, x_tmu1, x_tmu0 = y[0], y[1], y[2]
    lam0, lam_dt, lam10, lf1, lf0, ose, lam_tt, w_tt, lam_he, f_d = args
    F = lf1 * x_tmu1 + lf0 * x_tmu0  # fusion (= formation) flux out of the tmu pool
    recyc = (1.0 - ose) * F  # muons that survive sticking re-form tmu (3/4 F=1, 1/4 F=0)
    tt1 = lam_tt * x_tmu1  # ttmu formation flux out of each tmu hyperfine pool
    tt0 = lam_tt * x_tmu0
    tt_return = (1.0 - w_tt) * (tt1 + tt0)  # non-stuck tt-branch muons re-enter the tmu pool
    he = lam_he * x_dmu  # He-3 scavenging out of the dmu pool (absorbing)
    # d-recapture routing: a fraction f_d of the surviving-sticking flux re-enters via the dmu pool
    # (the freed muon recaptures on d and must transfer again) instead of re-forming tmu directly.
    # f_d=0.0 (default) makes the two new terms IEEE-exact identities (x + 0.0 == x; 1.0*x == x), so the
    # channels-/recapture-off engine reduces to the locked v1 step sequence bit-for-bit (reduction gate).
    dx_dmu = -(lam_dt + lam0 + lam_he) * x_dmu + f_d * recyc
    dx_tmu1 = (
        lam_dt * x_dmu - (lam10 + lf1 + lam0 + lam_tt) * x_tmu1 + 0.75 * ((1.0 - f_d) * recyc + tt_return)
    )
    dx_tmu0 = lam10 * x_tmu1 - (lf0 + lam0 + lam_tt) * x_tmu0 + 0.25 * ((1.0 - f_d) * recyc + tt_return)
    dN_fus = F  # d-t 14-MeV-neutron counter ONLY; tt-branch fusion neutrons are out of scope in v1
    dstuck = ose * F
    ddec = lam0 * (x_dmu + x_tmu1 + x_tmu0)
    dloss_tt = w_tt * (tt1 + tt0)  # muons lost to tt-branch sticking (absorbing)
    dloss_he = he
    return jnp.stack([dx_dmu, dx_tmu1, dx_tmu0, dN_fus, dstuck, ddec, dloss_tt, dloss_he])


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
    *,
    lambda_tt=0.0,
    omega_tt=0.0,
    lambda_he=0.0,
    f_d=0.0,
    saveat=None,
):
    """Integrate the cycle to ``t1`` (long enough that the muon is gone). Returns the diffrax solution.

    The three WS-N loss-channel rates (``lambda_tt``, ``omega_tt``, ``lambda_he``) are keyword-only and
    default to 0.0, so the engine default is channels-OFF and reduces bit-exactly to the v1 network
    (reduction gate G-N1). ``f_d`` (default 0.0) is the d-recapture routing fraction: the share of the
    surviving-sticking flux re-entering via the dmu pool instead of re-forming tmu directly; f_d=0.0 is an
    IEEE-exact identity, so the default engine is unchanged. ``saveat`` defaults to
    ``diffrax.SaveAt(t1=True)`` (today's behavior exactly); pass an explicit ``diffrax.SaveAt(ts=...)`` to
    record a trajectory (the hook WS-N's reduction check and WS-T's twin need).

    y0 compatibility: a length-6 ``y0`` is padded with two zeros to length 8 before solving, so every
    existing 6-element caller keeps working unchanged; the default ``y0`` is the 8-element vector.
    """
    if y0 is None:
        y0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # muon on deuterium; +2 loss accumulators
    else:
        y0 = jnp.asarray(y0)
        if y0.shape[0] == 6:  # pad legacy 6-element state to length 8 (the two loss accumulators start at 0)
            y0 = jnp.concatenate([y0, jnp.zeros(2, dtype=y0.dtype)])
    args = (
        lambda_0, lambda_dt, lambda_10, lambda_form1, lambda_form0, omega_s_eff,
        lambda_tt, omega_tt, lambda_he, f_d,
    )
    return diffrax.diffeqsolve(
        diffrax.ODETerm(_field),
        diffrax.Kvaerno5(),
        t0=0.0,
        t1=t1,
        dt0=1e-10,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol, norm=_v1_error_norm),
        saveat=diffrax.SaveAt(t1=True) if saveat is None else saveat,
        max_steps=max_steps,
    )


def fusions_per_muon_ode(lambda_0, lambda_dt, lambda_10, lambda_form1, lambda_form0, omega_s_eff, **kw):
    """X_mu = N_fus(t1) from the full ODE network."""
    sol = solve_cycle(lambda_0, lambda_dt, lambda_10, lambda_form1, lambda_form0, omega_s_eff, **kw)
    return sol.ys[-1, 3]


def conservation_residual(sol):
    """x_dmu + x_tmu1 + x_tmu0 + stuck + dec + loss_tt + loss_he - 1 at the final time (should be ~0).

    Includes the two WS-N loss accumulators (y[6], y[7]); with channels off they are 0 and this reduces
    to the v1 five-term sum. N_fus (y[3]) is an event counter and is deliberately excluded.
    """
    y = sol.ys[-1]
    return float(y[0] + y[1] + y[2] + y[4] + y[5] + y[6] + y[7] - 1.0)


def params_from_conditions(
    rates,
    T,
    phi,
    c_t,
    omega_s_eff=None,
    use_legacy_sticking=False,
    eta=None,
    c_he=0.0,
    include_loss_channels=False,
    q_1s=None,
):
    """Assemble cycle rates from the ledger + physical conditions (T [K], density phi, tritium fraction c_t).

    Documented v1 density scalings: transfer ~ phi*c_t, spin-flip ~ phi, formation ~ phi (inside
    formation.lambda_dtmu). omega_s_eff defaults to omega_s0*(1-R_col) from the ledger.

    ``eta`` is the epithermal formation enhancement (ledger row ``eta_dtmu``); ``None`` reads it from the
    ledger (= 1.0, so behavior is unchanged), and it is a structural knob (eta=1 bare theory .. ~5 fit),
    NOT a UQ prior -- the measured lambda_c band already contains eta as it occurred at the anchors (I5).

    ``include_loss_channels`` (default False = channels OFF) pulls the WS-N loss-channel rates from the
    ledger: ``lambda_tt = lambda_ttmu * phi * c_t`` (ttmu side-branch) and ``lambda_he = lambda_dhe3 * phi
    * c_he`` (He-3 scavenging; ``c_he`` is a STATIC per-run helium fraction, never time-evolved). Channels
    OFF returns all three at 0.0, so the engine default reproduces v1 exactly (the three ledger rows are
    needs_verification and I3 forbids introducing new physics on unverified values as a silent default).

    ``q_1s`` (default None = recapture OFF, f_d=0.0) is the contested cascade ground-state fraction of the
    d-recapture routing: ``None`` keeps v1's direct recycle-to-tmu, a number sets the routing fraction
    ``f_d = (1 - c_t) * q_1s`` -- the freed muon recaptures on deuterium with probability ~ the deuterium
    fraction (1 - c_t), ground-state fraction q_1s re-enters via the dmu pool. This is a first-order
    construction (docs/accounting.md d-recapture row); the deferred _CALIB unfolding stays acquisition-gated.
    """
    from . import formation
    from .analytic import effective_sticking
    from .rates import omega_fraction

    if eta is None:
        eta = rates.value("eta_dtmu")
    lambda_0 = rates.value("lambda_mu_decay")
    lambda_dt = rates.value("lambda_dt_transfer") * phi * c_t
    lambda_10 = rates.value("lambda_10_spinflip") * phi
    lf0 = formation.lambda_dtmu(T, phi, 0, eta)
    lf1 = formation.lambda_dtmu(T, phi, 1, eta)
    if omega_s_eff is None:
        os0 = omega_fraction(rates["omega_s0_legacy" if use_legacy_sticking else "omega_s0"])
        omega_s_eff = effective_sticking(os0, rates.value("R_col"))
    if include_loss_channels:
        lambda_tt = rates.value("lambda_ttmu") * phi * c_t
        omega_tt = rates.value("omega_tt")
        lambda_he = rates.value("lambda_dhe3") * phi * c_he
    else:
        lambda_tt = omega_tt = lambda_he = 0.0
    f_d = 0.0 if q_1s is None else (1.0 - c_t) * q_1s
    return dict(
        lambda_0=lambda_0,
        lambda_dt=lambda_dt,
        lambda_10=lambda_10,
        lambda_form1=lf1,
        lambda_form0=lf0,
        omega_s_eff=omega_s_eff,
        lambda_tt=lambda_tt,
        omega_tt=omega_tt,
        lambda_he=lambda_he,
        f_d=f_d,
    )


def fusions_per_muon_from_conditions(rates, T, phi, c_t, **kw):
    """One-call X_mu from (T, phi, c_t), using the ledger and the v1 formation model."""
    return fusions_per_muon_ode(**params_from_conditions(rates, T, phi, c_t, **kw))
