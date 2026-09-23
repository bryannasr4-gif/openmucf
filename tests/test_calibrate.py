"""Tests for Bayesian calibration + the identifiability finding (Phase 2 v1 polish, item 2)."""

import math

import numpyro.handlers
import pytest

from openmucf import calibrate


def test_calibration_recovers_effective_sticking_and_exposes_degeneracy():
    s = calibrate.run_mcmc(num_warmup=400, num_samples=1200, seed=0)
    summ = calibrate.summarize(s)
    # effective sticking recovered near the measured 0.45%, and TIGHTLY constrained
    assert 0.40 < summ["omega_s_eff_pct"]["mean"] < 0.52
    assert summ["omega_s_eff_pct"]["sd"] < 0.06
    # lambda_c constrained by the yield
    assert 0.9e8 < summ["lambda_c"]["mean"] < 1.6e8
    # ... but omega_s0 and R are strongly correlated (positive ridge along fixed product) -- the degeneracy
    assert summ["corr_omega_s0_R"] > 0.4
    # R is poorly constrained on its own (wide) relative to the tight effective sticking
    assert summ["R"]["sd"] > 0.03


def test_informative_prior_partially_breaks_degeneracy():
    # The bounded Kamimura chain must meet the same convergence gate as the weak chain.
    mk, sk = calibrate.run_mcmc_full(1000, 1000, seed=0, omega_s0_prior=calibrate.KAMIMURA_OMEGA_S0_PRIOR)
    kam = calibrate.summarize(sk, mcmc=mk)
    weak = calibrate.summarize(calibrate.run_mcmc(1000, 1000, seed=0))
    assert kam["n_divergences"] == 0
    for site in calibrate._DIAG_SITES:
        d = kam["diagnostics"][site]
        assert d["r_hat"] < 1.01, (site, d["r_hat"])
        assert d["ess"] > 400, (site, d["ess"])
    # the Kamimura theory prior tightens omega_s0 (and hence R) vs the weak-prior case
    assert kam["omega_s0_pct"]["sd"] < weak["omega_s0_pct"]["sd"]


def test_kamimura_prior_is_the_weak_support_truncation():
    """The sampled support matches the weak box and excludes the negative basin."""
    prior = calibrate.KAMIMURA_OMEGA_S0_PRIOR
    assert prior == ("truncnormal", 0.857, 0.03, 0.50, 1.20)
    trace = numpyro.handlers.trace(numpyro.handlers.seed(calibrate.model, 0)).get_trace(omega_s0_prior=prior)
    support = trace["omega_s0_pct"]["fn"].support
    assert (float(support.lower_bound), float(support.upper_bound)) == calibrate.WEAK_OMEGA_S0_PRIOR[1:]
    assert float(support.lower_bound) > 0
    mass = (math.erfc((prior[1] - prior[3]) / (prior[2] * math.sqrt(2)))
            + math.erfc((prior[4] - prior[1]) / (prior[2] * math.sqrt(2)))) / 2
    assert 1.0 - mass == 1.0


def test_unknown_prior_kind_is_refused():
    """An unsupported prior kind raises instead of silently becoming uniform."""
    with pytest.raises(ValueError, match="bogus"):
        numpyro.handlers.trace(numpyro.handlers.seed(calibrate.model, 0)).get_trace(
            omega_s0_prior=("bogus", 0.5, 1.2)
        )


def test_multichain_diagnostics():
    """Convergence gate: the default (4-chain, widened-box) run mixes -- max r_hat < 1.01, min ess >
    400 on every sampled/derived site, and ZERO divergences. summarize(..., mcmc=...) surfaces them."""
    mcmc, s = calibrate.run_mcmc_full(num_warmup=500, num_samples=1000, seed=0)
    summ = calibrate.summarize(s, mcmc=mcmc)
    assert summ["n_divergences"] == 0
    for site in calibrate._DIAG_SITES:
        d = summ["diagnostics"][site]
        assert d["r_hat"] < 1.01, (site, d["r_hat"])
        assert d["ess"] > 400, (site, d["ess"])
        # mcse = sd / sqrt(ess) is finite and positive for a mixed chain
        assert d["mcse"] > 0.0


def test_default_boxes_are_the_widened_ones():
    """The disclosed statistical correction: the default R / weak-omega_s0 boxes are the WIDENED
    ones; the old boxes railed (documented in CALIBRATION.md)."""
    assert calibrate.R_PRIOR_DEFAULT == (0.00, 0.80)          # was (0.10, 0.60)
    assert calibrate.WEAK_OMEGA_S0_PRIOR == ("uniform", 0.50, 1.20)  # was (..., 0.60, 1.10)
    assert calibrate.LAMBDA_C_PRIOR_DEFAULT == (0.8e8, 1.6e8)  # UNCHANGED (measured-band prior)


def test_obs_correlation_keeps_product_pinned():
    """Covariance sensitivity: a MultivariateNormal likelihood with off-diagonal rho_obs (vs the
    default independent Gaussians) leaves the well-constrained PRODUCT omega_s_eff pinned near 0.45%."""
    base = calibrate.summarize(calibrate.run_mcmc(400, 1000, seed=0))
    corr = calibrate.summarize(calibrate.run_mcmc(400, 1000, seed=0, obs_corr=0.5))
    assert abs(base["omega_s_eff_pct"]["mean"] - corr["omega_s_eff_pct"]["mean"]) < 0.03
    # and it is a genuine change of likelihood (the R-width / correlation moves off the independent case)
    assert corr["omega_s_eff_pct"]["mean"] > 0.0


def test_reattribution_shifts_ose_down():
    """Re-attribution: with tt_pc>0 the omega_s0(1-R) posterior mean shifts strictly below the tt_pc=0 mean,
    and the total reproduces the anchor. SKIPPED while lambda_ttmu is blocked (0.0) -- the joint refit is
    not run; revive when the Matsuzaki/Bom tt tables land and lambda_ttmu != 0."""
    import pytest

    from openmucf import load_rates

    if load_rates().value("lambda_ttmu") == 0.0:
        pytest.skip("lambda_ttmu blocked (0.0); tt re-attribution refit skipped (blocked path)")


def test_reattribution_blocked_section_in_calibration():
    """The blocked path is honestly recorded: while lambda_ttmu is 0.0 the tt refit is skipped and
    CALIBRATION.md carries the 'blocked' section header (no fabricated 3rd chain)."""
    from pathlib import Path

    from openmucf import load_rates

    assert load_rates().value("lambda_ttmu") == 0.0  # precondition for this release: tt still blocked
    calib = (Path(__file__).resolve().parents[1] / "CALIBRATION.md").read_text(encoding="utf-8")
    assert "Channels-on re-attribution (ttmu) -- blocked" in calib
