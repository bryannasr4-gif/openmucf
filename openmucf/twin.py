"""openmucf.twin -- counts-level neutron time-spectrum forward model + the idealized estimator (v0).

The piece an experimenter actually runs: a fuel-component neutron time-spectrum expectation from the v1
cycle ODE, a Poisson sampler for raw histograms, and the IDEALIZED two-exponential estimator everyone
fits -- so its bias against the model truth can be quantified on synthetic data (see
``scripts/generate_twin_audit.py`` and ``TWIN_AUDIT.md``).

Fenced v0 (DECIDED, WAVE1_EXECUTION_SPEC.md sec.5.1): constant density phi; d-t only; delta beam pulse
(the muon starts on deuterium at t=0); flat background; NO detector-response prediction; NO
dataset-specific claims. Each named historical dataset's ACTUAL published procedure (pulse structure,
material stopping components, detector response) is stage 2 and acquisition/contact-gated -- nothing
here is a claim about any specific published spectrum.

Runs CHANNELS-OFF by default: the forward model calls ``cycle.solve_cycle`` with the loss channels at
their 0.0 defaults, so the twin reduces to the established v1 engine (the ``saveat`` trajectory hook it
uses is the one authorized for WS-N + WS-T). ``likelihood.py`` holds the Bayesian counts-level model.

Standard muCF disappearance identity used throughout:

    lambda_n = lambda_0 + omega_s_eff * lambda_c        (muon disappearance rate)
    X_mu     = lambda_c / lambda_n                      (so lambda_n = lambda_c / X_mu)

i.e. the neutron time spectrum decays as ~ lambda_c * exp(-lambda_n t); the late-time slope is
lambda_n and the total yield per muon is X_mu.
"""

from __future__ import annotations

import diffrax
import jax.numpy as jnp
import numpy as np

from . import cycle
from .constants import LAMBDA_0

# solve_cycle keyword names that params_from_conditions emits (channels included, all 0.0 by default).
_CYCLE_KEYS = (
    "lambda_0",
    "lambda_dt",
    "lambda_10",
    "lambda_form1",
    "lambda_form0",
    "omega_s_eff",
    "lambda_tt",
    "omega_tt",
    "lambda_he",
)


def _solve_on_grid(t_grid, cycle_params):
    """Solve the v1 cycle ODE saving the full state at each ``t_grid`` point (channels-off by default)."""
    ts = jnp.asarray(t_grid, dtype=float)
    kw = {k: cycle_params[k] for k in _CYCLE_KEYS if k in cycle_params}
    return cycle.solve_cycle(**kw, t1=float(ts[-1]), saveat=diffrax.SaveAt(ts=ts))


def fusion_rate_density(t_grid, **cycle_params) -> jnp.ndarray:
    """Fuel-component d-t fusion rate density F(t) = lf1*x_tmu1(t) + lf0*x_tmu0(t) on ``t_grid`` [s].

    This is exactly ``dN_fus/dt`` of the cycle ODE (per incident muon). ``cycle_params`` is a
    ``cycle.params_from_conditions(...)`` dict. Units: s^-1 (fusions per muon per second).
    """
    sol = _solve_on_grid(t_grid, cycle_params)
    lf1 = cycle_params["lambda_form1"]
    lf0 = cycle_params["lambda_form0"]
    return lf1 * sol.ys[:, 1] + lf0 * sol.ys[:, 2]


def expected_counts(t_edges, cycle_params, n_mu, efficiency, background_rate) -> jnp.ndarray:
    """Expected neutron counts per histogram bin defined by ``t_edges`` [s] (length = len(edges)-1).

        counts_i = n_mu * efficiency * (N_fus(e_{i+1}) - N_fus(e_i)) + background_rate * (e_{i+1} - e_i)

    The per-bin fusion yield is read from the ODE's N_fus accumulator (state index 3) at the bin edges,
    so it is EXACT (no quadrature). ``background_rate`` is a flat rate [counts/s]; ``n_mu * efficiency``
    is the incident-muon count times detector efficiency.
    """
    te = jnp.asarray(t_edges, dtype=float)
    sol = _solve_on_grid(te, cycle_params)
    n_fus = sol.ys[:, 3]
    # N_fus is a monotone accumulator; clip the per-bin yield at 0 to drop O(1e-14 relative) solver
    # round-off in the depleted late-time tail, so the expectation stays a physical (non-negative) count.
    d_fus = jnp.clip(jnp.diff(n_fus), 0.0, None)
    widths = jnp.diff(te)
    return n_mu * efficiency * d_fus + background_rate * widths


def synthetic_spectrum(t_edges, expected, seed) -> np.ndarray:
    """Poisson-sample a raw histogram from ``expected`` counts (host-side; seeded, reproducible).

    ``t_edges`` is accepted for API symmetry and shape-checked against ``expected``.
    """
    exp = np.asarray(expected, dtype=float)
    if exp.shape[0] != np.asarray(t_edges).shape[0] - 1:
        raise ValueError("expected must have length len(t_edges) - 1")
    rng = np.random.default_rng(seed)
    return rng.poisson(exp)


def fit_two_exponential(t_edges, counts, t_min, lambda_c=None, background_rate=0.0, lambda_0=LAMBDA_0):
    """The IDEALIZED estimator: weighted least squares of log(count-rate) on [t_min, t_edges[-1]].

    Model fitted: rate_i - background_rate = amplitude * exp(-lambda_n * t_center_i), a single decaying
    exponential over a flat background. Returns ``{lambda_n, amplitude, omega_s_eff, loss_per_cycle}``.

    Idealizations (each a documented v0 fence; ``TWIN_AUDIT.md`` quantifies the resulting bias against the
    full ODE truth):
      * a single exponential -- ignores the early dmu->tmu transfer transient and the two-hyperfine-pool
        structure of the real cycle (the residual bias is what the audit sweep reports);
      * a delta beam pulse at t=0 and a flat, time-independent background;
      * log-linear WLS with Poisson weights (weight_i = counts_i), so empty/negative-rate bins are dropped.

    ``lambda_c`` (optional) enables the standard muCF back-out of the per-cycle quantities from the fitted
    disappearance rate: ``omega_s_eff = (lambda_n - lambda_0) / lambda_c`` (effective per-cycle sticking)
    and ``loss_per_cycle = lambda_n / lambda_c`` (= 1/X_mu, total loss per cycle). Both need the cycling
    rate, which a single histogram cannot supply on its own (it is degenerate with the muon count through
    the amplitude); without ``lambda_c`` they are returned as NaN.
    """
    te = np.asarray(t_edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    centers = 0.5 * (te[:-1] + te[1:])
    widths = np.diff(te)
    rate = counts / widths - background_rate
    keep = (centers >= t_min) & (rate > 0.0)
    if keep.sum() < 2:
        raise ValueError("fewer than 2 usable bins in the fit window [t_min, t_edges[-1]]")
    tc = centers[keep]
    y = np.log(rate[keep])
    w = np.sqrt(counts[keep])  # polyfit minimizes sum((w*(y-yhat))^2); w=sqrt(counts) => Poisson weight
    slope, intercept = np.polyfit(tc, y, 1, w=w)
    lambda_n = float(-slope)
    amplitude = float(np.exp(intercept))
    if lambda_c is not None and lambda_c > 0:
        omega_s_eff = (lambda_n - lambda_0) / lambda_c
        loss_per_cycle = lambda_n / lambda_c
    else:
        omega_s_eff = float("nan")
        loss_per_cycle = float("nan")
    return {
        "lambda_n": lambda_n,
        "amplitude": amplitude,
        "omega_s_eff": float(omega_s_eff),
        "loss_per_cycle": float(loss_per_cycle),
    }


def disappearance_rate(omega_s_eff, lambda_c, lambda_0=LAMBDA_0) -> float:
    """Analytic muon disappearance rate lambda_n = lambda_0 + omega_s_eff * lambda_c.

    ``omega_s_eff`` is a bare fraction (not percent). This is the closed-form gate target: the two-
    exponential fit of a synthetic spectrum must recover it to < 1% (gate G-T1).
    """
    return float(lambda_0 + omega_s_eff * lambda_c)


def implied_cycling_rate(x_mu, omega_s_eff, lambda_0=LAMBDA_0) -> float:
    """Engine-implied actual cycling rate lambda_c = lambda_0 / (1/X_mu - omega_s_eff).

    The closed-form inversion of ``X_mu = 1/(omega_s_eff + lambda_0/lambda_c)`` (same inversion as
    ``validate.py``'s implied_lc). With this lambda_c, ``disappearance_rate`` equals ``lambda_c / X_mu``.
    """
    return float(lambda_0 / (1.0 / x_mu - omega_s_eff))
