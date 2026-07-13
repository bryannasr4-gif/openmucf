"""openmucf.calibrate -- Bayesian calibration of the cycle parameters to experiment (numpyro).

Calibrates (omega_s0, R, lambda_c) to the measured effective sticking and yield (Petitjean/Breunlich):
    omega_s_eff = omega_s0 * (1 - R) = 0.45 +- 0.05 %
    X_mu        = 1/(omega_s_eff + lambda_0/lambda_c) = 113 +- 12

Finding (identifiability): the yield/sticking data constrain the PRODUCT omega_s0*(1-R) (= omega_s_eff)
and lambda_c, but NOT omega_s0 and R separately -- the posterior concentrates on the CURVE
omega_s0*(1-R) = omega_s_eff (a product pinned by the data). The linear (Pearson) correlation of that
curved ridge is a descriptive statistic and is prior-support-dependent; the constraint, not its
two-decimal correlation, is the finding. Breaking the degeneracy needs an independent microscopic
constraint on the split, i.e. the Phase-3 reactivation calculation. Run with an informative omega_s0
prior (Kamimura) to see it partially resolve.

Sampling API
------------
* ``run_mcmc`` / ``run_mcmc_full`` default to 4 sequential chains x 2000 draws (multi-chain diagnostics).
* ``summarize(samples, mcmc=...)`` optionally attaches split-R_hat / ESS / divergence diagnostics via
  ``numpyro.diagnostics`` (no new runtime dependency).
* The default prior boxes were WIDENED 2026-07-12 (a disclosed statistical correction, I2-clean -- no
  target involved) because the previous boxes provably railed: R ~ Uniform(0.00, 0.80) (was 0.10-0.60;
  the weak-chain 95% CI hi 0.588 sat against the old 0.60 bound) and weak omega_s0_pct ~ Uniform(0.50,
  1.20) (was 0.60-1.10; the weak-chain 95% CI lo 0.608 sat against the old 0.60 bound). The lambda_c box
  is unchanged (0.8-1.6e8; a measured-band prior, not railing). See CALIBRATION.md.
* The FC-001 registered realization is PINNED in ``openmucf.forecast.posterior_samples`` (single chain,
  the OLD R box, the registered seed/warmup/samples) -- never changed while a card is registered against
  it (the FC-001 registered-card freeze). FC-002+ use the new defaults.

Convergence caveat (Kamimura chain): the informative omega_s0 prior is an UNBOUNDED Normal, so on the thin
degeneracy ridge a chain can occasionally get trapped in a zero-prior-mass artifact basin at negative
omega_s0 for some seeds (a NUTS init pathology, NOT a real posterior mode -- its prior log-density is
~-1200). The shipped chains use the pre-registered seed 0, which converges (r_hat ~1.00); CALIBRATION.md
reports and audits that r_hat, and the SBC / convergence gates use the BOUNDED weak prior, which cannot
trap. The init strategy is left at numpyro's default so the registered FC-001 realization is unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS

from .constants import LAMBDA_0

# observations (Petitjean/Breunlich 1989)
OBS = dict(omega_s_eff_obs=0.45, omega_s_eff_sd=0.05, xmu_obs=113.0, xmu_sd=12.0)

# WIDENED default prior boxes (2026-07-12; see the module docstring + CALIBRATION.md). The lambda_c box
# is unchanged. Parameters (R_prior, lambda_c_prior) are exposed so that (a) the FC-001 realization can be
# pinned to its registered OLD R box and (b) the prior-sensitivity sweep can vary each box.
WEAK_OMEGA_S0_PRIOR = ("uniform", 0.50, 1.20)
R_PRIOR_DEFAULT = (0.00, 0.80)
LAMBDA_C_PRIOR_DEFAULT = (0.8e8, 1.6e8)

# multi-chain defaults (multi-chain diagnostics gated in CI). FC-001 is pinned to 1 chain in forecast.py.
NUM_CHAINS_DEFAULT = 4
CHAIN_METHOD_DEFAULT = "sequential"

# diagnostics are reported for these sites (the 3 sampled parameters + the 2 deterministics).
_DIAG_SITES = ("omega_s0_pct", "R", "lambda_c", "omega_s_eff_pct", "X_mu")


def model(
    omega_s_eff_obs=0.45,
    omega_s_eff_sd=0.05,
    xmu_obs=113.0,
    xmu_sd=12.0,
    omega_s0_prior=WEAK_OMEGA_S0_PRIOR,
    R_prior=R_PRIOR_DEFAULT,
    lambda_c_prior=LAMBDA_C_PRIOR_DEFAULT,
    obs_corr=0.0,
):
    """omega_s0_prior: ('uniform', lo, hi) [weak; exposes the degeneracy] or ('normal', mu, sd) [Kamimura].

    ``R_prior`` / ``lambda_c_prior`` are (lo, hi) uniform boxes. ``obs_corr`` (default 0.0) treats the two
    observations as independent Gaussians -- published as separate summary statistics, primary covariance
    unobtainable pre-acquisition. A nonzero ``obs_corr`` switches to a MultivariateNormal
    likelihood with off-diagonal correlation rho_obs (the covariance-sensitivity cut, CALIBRATION.md); it
    is NEVER used by the default/registered chains, so the obs_corr=0.0 path is byte-identical to the
    original two-independent-Normal likelihood.
    """
    if omega_s0_prior[0] == "normal":
        omega_s0 = numpyro.sample("omega_s0_pct", dist.Normal(omega_s0_prior[1], omega_s0_prior[2]))
    else:
        omega_s0 = numpyro.sample("omega_s0_pct", dist.Uniform(omega_s0_prior[1], omega_s0_prior[2]))
    R = numpyro.sample("R", dist.Uniform(R_prior[0], R_prior[1]))
    lambda_c = numpyro.sample("lambda_c", dist.Uniform(lambda_c_prior[0], lambda_c_prior[1]))

    ose_pct = omega_s0 * (1.0 - R)
    numpyro.deterministic("omega_s_eff_pct", ose_pct)
    xmu = 1.0 / (ose_pct / 100.0 + LAMBDA_0 / lambda_c)
    numpyro.deterministic("X_mu", xmu)

    if obs_corr == 0.0:
        numpyro.sample("obs_ose", dist.Normal(ose_pct, omega_s_eff_sd), obs=omega_s_eff_obs)
        numpyro.sample("obs_xmu", dist.Normal(xmu, xmu_sd), obs=xmu_obs)
    else:
        cov = obs_corr * omega_s_eff_sd * xmu_sd
        mean = jnp.stack([ose_pct, xmu])
        covariance = jnp.array(
            [[omega_s_eff_sd**2, cov], [cov, xmu_sd**2]]
        )
        numpyro.sample(
            "obs_joint",
            dist.MultivariateNormal(mean, covariance_matrix=covariance),
            obs=jnp.array([omega_s_eff_obs, xmu_obs]),
        )


def run_mcmc_full(
    num_warmup=1000,
    num_samples=2000,
    seed=0,
    omega_s0_prior=WEAK_OMEGA_S0_PRIOR,
    R_prior=R_PRIOR_DEFAULT,
    lambda_c_prior=LAMBDA_C_PRIOR_DEFAULT,
    obs_corr=0.0,
    num_chains=NUM_CHAINS_DEFAULT,
    chain_method=CHAIN_METHOD_DEFAULT,
    **obs,
):
    """Run NUTS and return ``(mcmc, samples)``. ``samples`` is the pooled (across-chain) draws dict.

    ``extra_fields=("diverging",)`` is collected for the divergence diagnostic; it does not alter the
    parameter draws (so the obs_corr=0.0 / single-chain / OLD-box path reproduces the FC-001 realization
    byte-for-byte -- verified by ``git diff --exit-code forecasts/`` and the FC-001 pin test).
    """
    data = {
        **OBS,
        **obs,
        "omega_s0_prior": omega_s0_prior,
        "R_prior": R_prior,
        "lambda_c_prior": lambda_c_prior,
        "obs_corr": obs_corr,
    }
    mcmc = MCMC(
        NUTS(model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",), **data)
    return mcmc, mcmc.get_samples()


def run_mcmc(*args, **kwargs):
    """Back-compatible wrapper: return only the pooled samples dict (see :func:`run_mcmc_full`)."""
    return run_mcmc_full(*args, **kwargs)[1]


def summarize(samples, mcmc=None):
    """Posterior summary. With ``mcmc`` given, also attach multi-chain convergence diagnostics.

    Adds ``"diagnostics"`` = {site: {"r_hat", "ess", "mcse"}} (split-Gelman-Rubin / effective sample size
    via ``numpyro.diagnostics``; mcse = sd / sqrt(ess)) and ``"n_divergences"`` (int). No new dependency.
    """

    def stats(a):
        a = np.asarray(a)
        return {
            "mean": float(a.mean()),
            "sd": float(a.std()),
            "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)),
        }

    out = {k: stats(samples[k]) for k in ("omega_s0_pct", "R", "lambda_c", "omega_s_eff_pct", "X_mu")}
    out["corr_omega_s0_R"] = float(
        np.corrcoef(np.asarray(samples["omega_s0_pct"]), np.asarray(samples["R"]))[0, 1]
    )
    if mcmc is not None:
        grouped = mcmc.get_samples(group_by_chain=True)
        diags = {}
        for site in _DIAG_SITES:
            x = np.asarray(grouped[site])  # shape (num_chains, num_samples)
            ess = float(effective_sample_size(x))
            sd = out[site]["sd"]
            diags[site] = {
                "r_hat": float(split_gelman_rubin(x)),
                "ess": ess,
                "mcse": (sd / np.sqrt(ess)) if ess > 0 else float("nan"),
            }
        out["diagnostics"] = diags
        out["n_divergences"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return out
