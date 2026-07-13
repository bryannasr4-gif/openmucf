"""Simulation-Based Calibration (SBC) for openmucf.calibrate -- a stronger correctness check than r_hat/ESS:
it verifies the whole draw->simulate->refit->rank loop is UNBIASED (the posterior is well-calibrated).

For each of N rounds: draw theta* ~ default priors, simulate the two observations from the model's OWN
likelihood, refit, and record the rank of theta*_i among the posterior draws. If the sampler is calibrated
those ranks are uniform; a systematic bias (a coding error in the model or a mis-scaled likelihood) skews
them. We test uniformity with a 20-bin chi-square per parameter (p > 0.005).

SLOW (~7 min): marked `slow`, so it is deselected from default CI (`addopts = -m 'not slow'`); run once in
session and paste the output. No committed artifact -> zero audit surface. No new runtime dependency:
numpyro chains + the already-locked scipy (transitive dep of numpyro/SALib) for the chi-square p-value.
"""

import numpy as np
import pytest

from openmucf import calibrate
from openmucf.constants import LAMBDA_0

pytestmark = pytest.mark.slow

# default prior boxes (the WIDENED defaults; SBC must pass on the SHIPPED priors)
_OS0 = calibrate.WEAK_OMEGA_S0_PRIOR[1:]     # (0.50, 1.20)
_R = calibrate.R_PRIOR_DEFAULT               # (0.00, 0.80)
_LC = calibrate.LAMBDA_C_PRIOR_DEFAULT       # (0.8e8, 1.6e8)
_OSE_SD, _XMU_SD = calibrate.OBS["omega_s_eff_sd"], calibrate.OBS["xmu_sd"]


def test_sbc_rank_uniformity():
    import scipy.stats as st  # test-only; present in the locked env (numpyro/SALib transitive dependency)

    n_rounds = 200
    rng = np.random.default_rng(0)
    ranks = {"omega_s0_pct": [], "R": [], "lambda_c": []}
    n_draws = None
    for i in range(n_rounds):
        os0 = rng.uniform(*_OS0)
        r = rng.uniform(*_R)
        lc = rng.uniform(*_LC)
        ose = os0 * (1.0 - r)                                   # effective sticking [percent]
        xmu = 1.0 / (ose / 100.0 + LAMBDA_0 / lc)
        y_ose = float(rng.normal(ose, _OSE_SD))
        y_xmu = float(rng.normal(xmu, _XMU_SD))
        s = calibrate.run_mcmc(
            num_warmup=500, num_samples=1000, seed=i, num_chains=2,
            omega_s_eff_obs=y_ose, xmu_obs=y_xmu,
        )
        n_draws = np.asarray(s["R"]).size
        for name, true in (("omega_s0_pct", os0), ("R", r), ("lambda_c", lc)):
            ranks[name].append(int((np.asarray(s[name]) < true).sum()))

    print(f"\nSBC: {n_rounds} rounds x 2 chains x 1000 draws (n_draws/round={n_draws})")
    for name in ("omega_s0_pct", "R", "lambda_c"):
        counts, _ = np.histogram(ranks[name], bins=20, range=(0, n_draws + 1))
        chi2, pval = st.chisquare(counts)                       # expected = n_rounds/20 per bin, dof=19
        print(f"  {name:14s}: chi2={chi2:6.2f}  p={pval:.4f}  (bins n={counts.sum()})")
        assert pval > 0.005, f"SBC rank non-uniform for {name}: chi2={chi2:.2f}, p={pval:.4g}"
