"""Tests for the UQ / sensitivity / breakeven-falsification layer (Phase 2.3)."""

from openmucf import uq


def test_local_sensitivity_signs():
    s = uq.local_sensitivities()
    assert s["X_mu"]["omega_s0_pct"] < 0  # more sticking -> fewer fusions
    assert s["X_mu"]["R"] > 0  # more reactivation -> less effective sticking -> more fusions
    assert s["X_mu"]["lambda_c"] > 0  # faster cycling -> more fusions
    assert s["Q_net"]["E_mu_GeV"] < 0  # costlier muons -> lower gain
    assert s["Q_net"]["eta_acc"] > 0


def test_gradient_cross_check_ode_vs_analytic():
    r = uq.cross_check_gradient()
    assert r["agree"], r


def test_sobol_isolates_the_drivers_of_xmu():
    s = uq.sobol_indices(N=1024, output="X_mu")
    for irrelevant in ("E_mu_GeV", "eta_acc", "eta_thermal"):
        assert abs(s["ST"][irrelevant]) < 0.05
    assert s["ST"]["omega_s0_pct"] > 0.1
    assert s["ST"]["lambda_c"] > 0.1


def test_sobol_cis_present_and_ranking_stable():
    """sobol_indices returns bootstrap CIs, and the top-1 total-order driver is identical across
    N in {4096, 8192} x seed in {0, 1} (the ranking is not a sampling artifact)."""
    tops = set()
    for N in (4096, 8192):
        for seed in (0, 1):
            s = uq.sobol_indices(N=N, output="X_mu", seed=seed)
            assert set(s) == {"S1", "ST", "S1_conf", "ST_conf"}
            assert all(s["ST_conf"][k] >= 0 for k in s["ST_conf"])
            tops.add(max(s["ST"], key=s["ST"].get))
    assert tops == {"R"}, tops  # one identical top driver across all four cells


def test_breakeven_R_required_band():
    """The R>=0.77 requirement carries an omega_s0-box band (rises with initial sticking)."""
    r = uq.breakeven_audit(n=50_000)
    assert r["R_required_band_lo"] < r["R_required_at_infinite_lambda_c"] < r["R_required_band_hi"]
    assert 0.70 < r["R_required_band_lo"] < r["R_required_band_hi"] < 0.85


def test_forward_uq_net_electrical_below_breakeven():
    r = uq.forward_uq(n=50_000)
    assert 50 < r["X_mu"]["med"] < 400
    assert r["Q_net"]["hi"] < 1.0
    assert r["P_Qnet_gt1"] == 0.0


def test_breakeven_projection_is_falsified_under_measured_uncertainty():
    r = uq.breakeven_audit(n=50_000)
    assert r["P_xmu_gt500"] == 0.0
    assert r["P_qnet_gt1"] == 0.0
    # even zero sticking at the best measured cycling rate cannot reach 500
    assert r["xmu_cap_at_measured_lambda_c"] < 500
    # reaching 500 needs reactivation far above the measured ~0.35
    assert r["R_required_at_lambda_c_3e8"] > 0.85
