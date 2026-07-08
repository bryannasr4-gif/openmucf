"""openmucf.likelihood -- counts-level Bayesian model for a neutron time-spectrum histogram (WS-T v0).

The likelihood an experimenter would actually put on a raw histogram: a Poisson observation of per-bin
neutron counts whose expectation is the CLOSED-FORM single-exponential limit of the v1 cycle,

    expected_i = amplitude * X_mu * (exp(-lambda_n e_i) - exp(-lambda_n e_{i+1})) + background_rate * w_i,
    lambda_n   = lambda_0 + omega_s_eff * lambda_c ,   X_mu = lambda_c / lambda_n

(this is the late-time limit of ``twin.expected_counts``; it is cheap and JAX-differentiable, so NUTS is
feasible where solving the stiff ODE inside every leapfrog step would not be). ``twin.py`` holds the
exact-ODE forward model + the idealized estimator; the small closed-form-vs-ODE gap is the < 1% bias
that ``TWIN_AUDIT.md`` quantifies.

IDENTIFIABILITY (the honest treatment, stated because a counts histogram genuinely cannot do more):
  * A single delta-pulse histogram constrains the muon DISAPPEARANCE RATE lambda_n (the decay slope) and
    the total signal yield -- nothing more. Along the line lambda_n = lambda_0 + omega_s_eff * lambda_c,
    omega_s_eff and lambda_c trade off freely: the data fix their COMBINATION, not each separately.
  * They are separated ONLY through the informative lambda_c prior (the MEASURED liquid cycling band from
    the ledger row ``lambda_c_liquid``). With a flat lambda_c prior only the product-form lambda_n is
    identified. This is the same omega_s0/R-style degeneracy calibrate.py documents, one level up.
  * The amplitude (n_mu * efficiency) and lambda_c are additionally degenerate through the total yield
    (yield = amplitude * lambda_c / lambda_n): absent an independent muon count the amplitude absorbs the
    lambda_c scale, which is why lambda_c leans on its prior. amplitude/background are weak nuisances.

Fenced v0 (see ``twin.py``): constant phi, d-t only, delta pulse, flat background, no detector response,
no dataset-specific claim. Not part of the eager-import surface; reached as ``openmucf.likelihood``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .constants import LAMBDA_0


def _data_scales(t_edges, counts):
    """Weakly-informative nuisance scales from the histogram (host-side; concrete arrays only).

    ``amp_center`` ~ total signal / a nominal X_mu ~ 100 (a scale for the broad amplitude prior);
    ``background_scale`` ~ mean overall count-rate (the Exponential background prior's mean). Derived from
    the DATA, never from the truth. Computed here so the numpyro model never converts a traced value.
    """
    te = np.asarray(t_edges, dtype=float)
    total = float(np.sum(np.asarray(counts, dtype=float)))
    total_width = float(te[-1] - te[0])
    amp_center = max(total / 114.0, 1.0)  # X_mu ~ O(100); a scale only -- the prior stays broad
    background_scale = max(total / total_width, 1.0)  # mean overall rate; Exponential mean = this
    return amp_center, background_scale


def ledger_lambda_c_bounds(phi: float = 1.2):
    """Actual-rate lambda_c prior bounds at density ``phi`` from the ledger ``lambda_c_liquid`` row.

    The ledger band is the ACTUAL liquid-condition rate at phi ~ 1.2; scaling to another density uses the
    same phi-linearity as the engine: (phi/1.2) * [dist_lo, dist_hi]. At phi=1.2 this is exactly the
    measured [1.00e8, 1.45e8] band (consistent with calibrate.py's Uniform(0.8e8, 1.6e8) support).
    """
    from .rates import load_rates

    lo, hi = load_rates().dist_bounds("lambda_c_liquid")
    scale = phi / 1.2
    return (scale * lo, scale * hi)


def expected_counts_closed_form(t_edges, omega_s_eff_frac, lambda_c, amplitude, background_rate,
                                lambda_0=LAMBDA_0):
    """Closed-form per-bin expected counts (single-exponential limit). JAX-differentiable.

    ``omega_s_eff_frac`` is a bare fraction (not percent); ``amplitude`` is n_mu * efficiency.
    """
    te = jnp.asarray(t_edges, dtype=float)
    lambda_n = lambda_0 + omega_s_eff_frac * lambda_c
    x_mu = lambda_c / lambda_n
    surv = jnp.exp(-lambda_n * te)
    per_bin = x_mu * (surv[:-1] - surv[1:])  # fusions per muon in each bin
    widths = jnp.diff(te)
    return amplitude * per_bin + background_rate * widths


def spectrum_model(t_edges, counts, phi=1.2, lambda_c_bounds=None, lambda_0=LAMBDA_0,
                   amp_center=None, amp_log_sd=2.0, background_scale=None):
    """numpyro model for a raw neutron histogram ``counts`` on bins ``t_edges``.

    Priors (DECIDED, sec.5.2): omega_s_eff_pct ~ Uniform(0.2, 0.8) %; lambda_c ~ Uniform over the ledger
    ``lambda_c_liquid`` band scaled to ``phi`` (the informative prior that breaks the identifiability
    degeneracy above); amplitude = n_mu*efficiency ~ LogNormal (weak, data-scaled); background_rate ~
    Exponential (weak). Observation: counts ~ Poisson(expected).
    """
    if lambda_c_bounds is None:
        lambda_c_bounds = ledger_lambda_c_bounds(phi)
    counts_j = jnp.asarray(counts, dtype=float)
    if amp_center is None or background_scale is None:
        # only reached on a direct (untraced) call, where counts is concrete; under fit_spectrum/MCMC
        # these are passed pre-computed so the traced model never converts a tracer to float.
        _amp, _bg = _data_scales(t_edges, counts)
        amp_center = _amp if amp_center is None else amp_center
        background_scale = _bg if background_scale is None else background_scale

    ose_pct = numpyro.sample("omega_s_eff_pct", dist.Uniform(0.2, 0.8))
    lambda_c = numpyro.sample("lambda_c", dist.Uniform(lambda_c_bounds[0], lambda_c_bounds[1]))
    amplitude = numpyro.sample("amplitude", dist.LogNormal(jnp.log(amp_center), amp_log_sd))
    background_rate = numpyro.sample("background_rate", dist.Exponential(1.0 / background_scale))

    expected = expected_counts_closed_form(
        t_edges, ose_pct / 100.0, lambda_c, amplitude, background_rate, lambda_0
    )
    lambda_n = lambda_0 + (ose_pct / 100.0) * lambda_c
    numpyro.deterministic("lambda_n", lambda_n)
    numpyro.deterministic("X_mu", lambda_c / lambda_n)
    numpyro.sample("counts", dist.Poisson(expected), obs=counts_j)


def fit_spectrum(t_edges, counts, phi=1.2, lambda_c_bounds=None, num_warmup=300, num_samples=800,
                 seed=0, **model_kw):
    """NUTS posterior over (omega_s_eff_pct, lambda_c, amplitude, background_rate) given a histogram.

    Matches calibrate.py's conventions (seeded PRNGKey, progress_bar=False). ``num_warmup``/``num_samples``
    default to the reduced coverage-test settings; pass smaller values for smoke fits.
    """
    if lambda_c_bounds is None:
        lambda_c_bounds = ledger_lambda_c_bounds(phi)
    amp_center, background_scale = _data_scales(t_edges, counts)
    model_kw = {"amp_center": amp_center, "background_scale": background_scale, **model_kw}
    mcmc = MCMC(NUTS(spectrum_model), num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(seed),
        t_edges=jnp.asarray(t_edges, dtype=float),
        counts=jnp.asarray(counts, dtype=float),
        phi=phi,
        lambda_c_bounds=lambda_c_bounds,
        **model_kw,
    )
    return mcmc.get_samples()
