"""Tests for Bayesian calibration + the identifiability finding (Phase 2 v1 polish, item 2)."""

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
    weak = calibrate.summarize(calibrate.run_mcmc(400, 1200, seed=1))
    kam = calibrate.summarize(calibrate.run_mcmc(400, 1200, seed=1, omega_s0_prior=("normal", 0.857, 0.03)))
    # the Kamimura theory prior tightens omega_s0 (and hence R) vs the weak-prior case
    assert kam["omega_s0_pct"]["sd"] < weak["omega_s0_pct"]["sd"]
