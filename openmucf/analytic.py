"""openmucf.analytic -- closed-form steady-state muCF yield and energy balance.

The analytic backbone (derived two ways in ``MODEL_SPEC.md`` sec. 4: absorbing-Markov ODE
and renewal sum):

    omega_s_eff = omega_s0 * (1 - R)
    X_mu        = 1 / (omega_s_eff + lambda_0 / lambda_c),   lambda_c = phi * lambda_c_tilde
    Q           = X_mu * E_f * eta_conv / E_mu

Every function is pure and JAX-differentiable (scalars or arrays). ``cycle.py`` (the diffrax
ODE network) must reproduce ``fusions_per_muon`` to < 1% in the single-pool limit -- gate V1.
"""

from __future__ import annotations

from .constants import E_F_MEV, E_MU_GEV_DEFAULT, LAMBDA_0


def effective_sticking(omega_s0, R):
    """omega_s_eff = omega_s0 (1 - R). Inputs are bare fractions (not percent)."""
    return omega_s0 * (1.0 - R)


def cycling_rate(phi, lambda_c_tilde):
    """Actual cycling rate from the density-normalized rate: lambda_c = phi * lambda_c_tilde."""
    return phi * lambda_c_tilde


def fusions_per_muon(omega_s_eff, lambda_c, lambda_0=LAMBDA_0):
    """X_mu = 1 / (omega_s_eff + lambda_0 / lambda_c).

    ``lambda_c`` is the *actual* cycling rate (already = phi * lambda_c_tilde).
    """
    return 1.0 / (omega_s_eff + lambda_0 / lambda_c)


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
    R = rates.value("R_col")
    lambda_0 = rates.value("lambda_mu_decay")
    lambda_c = cycling_rate(phi, lambda_c_tilde)
    return fusions_per_muon(effective_sticking(omega_s0, R), lambda_c, lambda_0)
