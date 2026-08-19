"""openmucf.analytic -- closed-form steady-state muCF yield and energy balance.

The analytic backbone (derived two ways in ``MODEL_SPEC.md`` sec. 4: absorbing-Markov ODE
and renewal sum):

    omega_s_eff = omega_s0 * (1 - R),   R = 1 - (1 - R_col)(1 - R_X)
    X_mu        = 1 / (omega_s_eff + lambda_0 / lambda_c),   lambda_c = phi * lambda_c_tilde
    Q           = X_mu * E_f * eta_conv / E_mu

``R`` here is the TOTAL reactivation probability. It decomposes into SUCCESSIVE factors -- the
collisional/stripping fraction ``R_col`` and any external-field-assisted contribution ``R_X`` --
which compose multiplicatively on the SURVIVING sticking, not additively (see
``combine_reactivation``). A required total R must therefore never be compared directly against a
collision-only ``R_col`` as though they were the same quantity.

Every function is pure and JAX-differentiable (scalars or arrays). ``cycle.py`` (the diffrax
ODE network) must reproduce ``fusions_per_muon`` to < 1% in the single-pool limit -- gate V1.
"""

from __future__ import annotations

from .constants import E_F_MEV, E_MU_GEV_DEFAULT, LAMBDA_0


def effective_sticking(omega_s0, R):
    """omega_s_eff = omega_s0 (1 - R). Inputs are bare fractions (not percent).

    ``R`` is the TOTAL reactivation probability. To build it from the two-factor decomposition, use
    :func:`combine_reactivation` -- do not pass a collision-only ``R_col`` where a total is meant.
    """
    return omega_s0 * (1.0 - R)


def combine_reactivation(R_col, R_X=0.0):
    """Total reactivation from SUCCESSIVE factors: R = 1 - (1 - R_col)(1 - R_X).

    The collisional fraction ``R_col`` and any external-field-assisted fraction ``R_X`` act in
    sequence on the muon that is still stuck, so their SURVIVAL probabilities multiply:
    ``omega_s_eff = omega_s0 (1 - R_col)(1 - R_X)``. They are not interchangeable and never add.

    Domain: both inputs in [0, 1], where the result is also in [0, 1] and monotone non-decreasing in
    each argument (``R_col + R_X - R_col*R_X``). Kept branch-free so it stays JAX-traceable; the
    ledger values are range-checked by ``tests/test_analytic.py`` rather than by a runtime raise.
    ``R_X = 0`` is the field-off baseline and reduces this to ``R = R_col`` exactly.
    """
    return 1.0 - (1.0 - R_col) * (1.0 - R_X)


def ledger_reactivation(rates):
    """Total reactivation R from the ledger's two-factor rows (``R_col``, ``R_X``)."""
    return combine_reactivation(rates.value("R_col"), rates.value("R_X"))


def cycling_rate(phi, lambda_c_tilde):
    """Actual cycling rate from the density-normalized rate: lambda_c = phi * lambda_c_tilde."""
    return phi * lambda_c_tilde


def fusions_per_muon(omega_s_eff, lambda_c, lambda_0=LAMBDA_0):
    """X_mu = 1 / (omega_s_eff + lambda_0 / lambda_c).

    ``lambda_c`` is the *actual* cycling rate (already = phi * lambda_c_tilde).
    """
    return 1.0 / (omega_s_eff + lambda_0 / lambda_c)


def fusions_per_muon_v2(omega_s_eff, lambda_c, lambda_0=LAMBDA_0, tt_loss_rate=0.0, omega_tt=0.0):
    """Extended closed form with the ttmu side-branch competing hazard (v2).

        X_mu = 1 / (omega_s_eff + omega_tt * tt_loss_rate / lambda_c + lambda_0 / lambda_c)

    ``tt_loss_rate`` is the actual ttmu formation rate (= lambda_ttmu * phi * c_t); ``omega_tt`` the
    tt-branch muon-loss fraction. The extra per-cycle loss ``omega_tt * tt_loss_rate / lambda_c`` is the
    ``tt_pc`` share of the re-attribution (accounting.md). Derived in ``MODEL_SPEC.md`` sec.4 (dated
    subsection) as a renewal sum over tmu episodes with d-t formation, tt formation and decay as three
    competing first-order hazards; validated against the ODE to <1% by the analytic-vs-ODE gate.

    DOCUMENTED ASYMMETRY: the He-3 scavenging channel is OMITTED here. It removes muons from the *dmu*
    pool, but the single-pool closed form has already collapsed the dmu/tmu structure, so there is no
    clean single-pool representation of a dmu-only hazard; He-3 scavenging is available in ``cycle.py``
    (the ODE) only. With ``tt_loss_rate=0`` (or ``omega_tt=0``) this reduces exactly to
    :func:`fusions_per_muon`.
    """
    return 1.0 / (omega_s_eff + omega_tt * tt_loss_rate / lambda_c + lambda_0 / lambda_c)


def energy_gain(x_mu, eta_conv, E_f_MeV=E_F_MEV, E_mu_GeV=E_MU_GEV_DEFAULT):
    """Q = X_mu * E_f * eta_conv / E_mu (units reconciled internally)."""
    E_mu_MeV = E_mu_GeV * 1.0e3
    return x_mu * E_f_MeV * eta_conv / E_mu_MeV


def breakeven_xmu(E_f_MeV=E_F_MEV, E_mu_GeV=E_MU_GEV_DEFAULT, eta_conv=1.0):
    """Fusions-per-muon at Q = 1: X_mu = E_mu / (E_f * eta_conv).  ~284 for 5 GeV, eta=1."""
    return (E_mu_GeV * 1.0e3) / (E_f_MeV * eta_conv)


def from_ledger(rates, phi, lambda_c_tilde, use_legacy_sticking=False):
    """Compute X_mu using ledger values for omega_s0, R_col, lambda_0.

    ``rates`` is an :class:`openmucf.rates.RatesTable`.
    """
    from .rates import omega_fraction

    omega_s0 = omega_fraction(rates["omega_s0_legacy" if use_legacy_sticking else "omega_s0"])
    R = ledger_reactivation(rates)
    lambda_0 = rates.value("lambda_mu_decay")
    lambda_c = cycling_rate(phi, lambda_c_tilde)
    return fusions_per_muon(effective_sticking(omega_s0, R), lambda_c, lambda_0)
