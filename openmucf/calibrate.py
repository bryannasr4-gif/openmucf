"""openmucf.calibrate -- Bayesian calibration of the cycle parameters to experiment (numpyro).

Calibrates (omega_s0, R, lambda_c) to the measured effective sticking and yield (Petitjean/Breunlich):
    omega_s_eff = omega_s0 * (1 - R) = 0.45 +- 0.05 %
    X_mu        = 1/(omega_s_eff + lambda_0/lambda_c) = 113 +- 12

Finding (identifiability): the yield/sticking data constrain the PRODUCT omega_s0*(1-R) (= omega_s_eff)
and lambda_c, but NOT omega_s0 and R separately -- they come out strongly correlated (a ridge along
fixed omega_s0*(1-R): raising R forces raising omega_s0 to keep the product, hence positive correlation).
Breaking that degeneracy needs an independent microscopic constraint on the split, i.e. the Phase-3
reactivation calculation. Run with an informative omega_s0 prior (Kamimura) to see it partially resolve.
"""

from __future__ import annotations

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .constants import LAMBDA_0

# observations (Petitjean/Breunlich 1989)
OBS = dict(omega_s_eff_obs=0.45, omega_s_eff_sd=0.05, xmu_obs=113.0, xmu_sd=12.0)


def model(
    omega_s_eff_obs=0.45,
    omega_s_eff_sd=0.05,
    xmu_obs=113.0,
    xmu_sd=12.0,
    omega_s0_prior=("uniform", 0.60, 1.10),
):
    """omega_s0_prior: ('uniform', lo, hi) [weak; exposes the degeneracy] or ('normal', mu, sd) [Kamimura]."""
    if omega_s0_prior[0] == "normal":
        omega_s0 = numpyro.sample("omega_s0_pct", dist.Normal(omega_s0_prior[1], omega_s0_prior[2]))
    else:
        omega_s0 = numpyro.sample("omega_s0_pct", dist.Uniform(omega_s0_prior[1], omega_s0_prior[2]))
    R = numpyro.sample("R", dist.Uniform(0.10, 0.60))
    lambda_c = numpyro.sample("lambda_c", dist.Uniform(0.8e8, 1.6e8))

    ose_pct = omega_s0 * (1.0 - R)
    numpyro.deterministic("omega_s_eff_pct", ose_pct)
    xmu = 1.0 / (ose_pct / 100.0 + LAMBDA_0 / lambda_c)
    numpyro.deterministic("X_mu", xmu)

    numpyro.sample("obs_ose", dist.Normal(ose_pct, omega_s_eff_sd), obs=omega_s_eff_obs)
    numpyro.sample("obs_xmu", dist.Normal(xmu, xmu_sd), obs=xmu_obs)


def run_mcmc(num_warmup=800, num_samples=2000, seed=0, omega_s0_prior=("uniform", 0.60, 1.10), **obs):
    data = {**OBS, **obs, "omega_s0_prior": omega_s0_prior}
    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), **data)
    return mcmc.get_samples()


def summarize(samples):
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
    return out
