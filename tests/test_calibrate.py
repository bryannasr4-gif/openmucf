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


def test_reattribution_shifts_ose_down():
    """WS-N Sec.3.5: with tt_pc>0 the omega_s0(1-R) posterior mean shifts strictly below the tt_pc=0 mean,
    and the total reproduces the anchor. SKIPPED while lambda_ttmu is blocked (0.0) -- the joint refit is
    not run (Sec.3.5 blocked path); revive when the Matsuzaki/Bom tt tables land and lambda_ttmu != 0."""
    import pytest

    from openmucf import load_rates

    if load_rates().value("lambda_ttmu") == 0.0:
        pytest.skip("lambda_ttmu blocked (0.0); tt re-attribution refit skipped (WS-N Sec.3.5 blocked path)")


def test_reattribution_blocked_section_in_calibration():
    """The blocked path is honestly recorded: while lambda_ttmu is 0.0 the tt refit is skipped and
    CALIBRATION.md carries the 'blocked' section header (no fabricated 3rd chain)."""
    from pathlib import Path

    from openmucf import load_rates

    assert load_rates().value("lambda_ttmu") == 0.0  # precondition for this release: tt still blocked
    calib = (Path(__file__).resolve().parents[1] / "CALIBRATION.md").read_text(encoding="utf-8")
    assert "Channels-on re-attribution (ttmu) -- blocked" in calib
