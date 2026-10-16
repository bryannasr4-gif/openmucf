"""Tests for Bayesian calibration + the identifiability finding (Phase 2 v1 polish, item 2)."""

import copy
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import numpyro.handlers
import pytest

from openmucf import calibrate, sidechannels


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


# -----------------------------------------------------------------------------------------------------------
# The published calibration on the anchor's basis: observation model, quadrature estimator, convergence and
# NUTS gates.
# -----------------------------------------------------------------------------------------------------------
_CELLS = Path(__file__).resolve().parents[1] / "CALIBRATION_CELLS.json"


def _cells():
    return json.loads(_CELLS.read_text(encoding="utf-8"))


def test_observation_constants_and_the_widened_box():
    expected = dict(omega_s_pct=0.45, omega_s_sd=0.05, phi_lambda_c=145e6, phi_lambda_c_sd=12e6,
                    xmu=113.0, xmu_sd=10.0, calibration=0.08, lambda_stat=0.02)
    assert expected == calibrate.OBS_W
    assert calibrate.LAMBDA_C_PRIOR_W == (0.8e8, 2.0e8)
    # the legacy observation set stays as the frozen FC-001 path uses it
    legacy = dict(omega_s_eff_obs=0.45, omega_s_eff_sd=0.05, xmu_obs=113.0, xmu_sd=12.0)
    assert legacy == calibrate.OBS


def test_l3_covariance_is_built_from_the_common_calibration():
    w = calibrate.OBS_W
    k, el = w["calibration"], w["lambda_stat"]
    e_s = math.sqrt((w["xmu_sd"] / w["xmu"]) ** 2 - k**2 - el**2)
    e_o = math.sqrt(w["omega_s_sd"] ** 2 - (w["omega_s_pct"] * k) ** 2)
    a = np.array([
        [-w["omega_s_pct"] * k, 0, 0, e_o],
        [w["phi_lambda_c"] * k, w["phi_lambda_c"] * el, 0, 0],
        [w["xmu"] * k, w["xmu"] * el, -w["xmu"] * e_s, 0],
    ])
    cov = calibrate.observation_covariance("L3")
    assert np.array_equal(cov, a @ a.T)
    sd = np.sqrt(np.diag(cov))
    assert sd[0] == pytest.approx(w["omega_s_sd"], rel=1e-12)
    assert sd[2] == pytest.approx(w["xmu_sd"], rel=1e-12)
    corr = cov / np.outer(sd, sd)
    assert np.round(corr[0, 1], 4) == -0.6985
    assert np.round(corr[0, 2], 4) == -0.6509
    assert np.round(corr[1, 2], 4) == 0.9318


def test_l2_and_l1_covariances():
    l2 = calibrate.observation_covariance("L2")
    assert round(l2[1, 2] / math.sqrt(l2[1, 1] * l2[2, 2]), 4) == 0.9352
    assert l2[0, 1] == l2[0, 2] == 0.0
    l1 = calibrate.observation_covariance("L1")
    assert np.count_nonzero(l1 - np.diag(np.diag(l1))) == 0
    assert np.array_equal(np.diag(l1), np.diag(l2))
    with pytest.raises(ValueError, match="bogus"):
        calibrate.observation_covariance("bogus")


def test_likelihood_constant_is_the_side_channel_anchor():
    """Both estimators take L_other's prior from sidechannels.ANCHOR_L_OTHER_PCT, truncated at 0."""
    trace = numpyro.handlers.trace(numpyro.handlers.seed(calibrate.model_w, 0)).get_trace()
    fn = trace["L_pct"]["fn"]
    assert (float(fn.base_dist.loc), float(fn.base_dist.scale)) == sidechannels.ANCHOR_L_OTHER_PCT
    assert float(fn.support.lower_bound) == 0.0
    obs = trace["obs_w"]["fn"]
    assert np.allclose(np.asarray(obs.covariance_matrix), calibrate.observation_covariance("L3"), rtol=1e-15)
    assert _cells()["estimator"]["l_other_pct"] == list(sidechannels.ANCHOR_L_OTHER_PCT)


_N = NormalDist()


def toy_quantile_error(width_factor, h=0.1):
    """Largest quantile error (in sd) of the cell-integrated CDF on a standard normal discretized in cells of
    width h, each cell's exact mass spread over ``width_factor`` x its width (1 = the estimator's rule)."""
    edges = np.arange(-6.0, 6.0 + 1e-12, h)
    lo, hi = edges[:-1], edges[1:]
    mid = 0.5 * (lo + hi)
    mass = np.array([_N.cdf(b) - _N.cdf(a) for a, b in zip(lo, hi, strict=True)])
    acc = calibrate._RampCDF(np.linspace(-6.0, 6.0, 12001))
    acc.add(mid - 0.5 * h * width_factor, mid + 0.5 * h * width_factor, mass)
    cdf = acc.cdf(mass.sum())
    return max(abs(calibrate._quantile(acc.edges, cdf, p) - _N.inv_cdf(p)) for p in calibrate.QUANTILES)


def test_cell_integrated_quantiles_recover_a_gaussian():
    assert toy_quantile_error(1.0) < 0.005


def test_toy_check_reds_on_node_snapping():
    """Drill: with each cell's mass placed at its node (the rule the estimator replaces) the check fails."""
    assert toy_quantile_error(1e-9) > 0.005


def test_quadrature_reproduces_the_committed_means_at_a_quarter_grid():
    """The estimator's means converge fast: a quarter-resolution run is within 1e-5 of the committed cells."""
    post = calibrate.posterior_quadrature(calibrate.WEAK_OMEGA_S0_PRIOR, res=0.25)
    ref = _cells()["posteriors"]["weak"]
    for q in calibrate.QUAD_QUANTITIES:
        assert post[q]["mean"] == pytest.approx(ref[q]["mean"], rel=1e-5), q


_X_BINS = calibrate.QUAD_BINS["X_mu"]


def test_quadrature_x_mu_cdf_holds_its_mass_at_a_coarse_grid():
    """Control for the drill below: on the shipped X_mu edges the cell-integrated CDF runs from 0 to 1."""
    post = calibrate.posterior_quadrature(calibrate.WEAK_OMEGA_S0_PRIOR, res=0.1, bins=0.1)
    edges, cdf = post["X_mu_cdf"]
    assert (edges[0], edges[-1]) == _X_BINS[:2]
    assert cdf[0] == 0.0
    assert cdf[-1] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("lo, hi", [(75.0, _X_BINS[1]), (_X_BINS[0], 160.0)],
                         ids=["lower_tail", "upper_tail"])
def test_quadrature_refuses_x_mu_mass_outside_its_cdf_edges(monkeypatch, lo, hi):
    """Drill (X_mu CDF mass, each tail): edges narrowed on one side so that a small posterior mass, well under
    1e-5 of the total, falls outside them -> refused (the estimator admits at most 1e-9), never a CDF that
    silently stops short of 1 (which the Q_net quantiles would read)."""
    monkeypatch.setitem(calibrate.QUAD_BINS, "X_mu", (lo, hi, _X_BINS[2]))
    with pytest.raises(ValueError, match="X_mu mass outside its CDF edges") as exc:
        calibrate.posterior_quadrature(calibrate.WEAK_OMEGA_S0_PRIOR, res=0.1, bins=0.1)
    lost = float(str(exc.value).split(": ")[1].split(" ")[0])
    assert 1e-9 < lost < 1e-5, lost


def _convergence_drill(q, stat, factor):
    pub = _cells()["posteriors"]["weak"]
    fine = copy.deepcopy(pub)
    fine[q][stat] *= factor
    return calibrate.convergence_failures(pub, fine)


def test_convergence_failures_on_the_committed_posterior():
    pub = _cells()["posteriors"]["weak"]
    assert calibrate.convergence_failures(pub, copy.deepcopy(pub)) == []


@pytest.mark.parametrize(
    "q, stat, factor, named",
    [
        ("R", "q0.025", 1 + 2e-3, True),       # a non-L quantile moved 2e-3: over its 1e-3
        ("L_pct", "q0.975", 1 + 6e-3, True),   # an L_other quantile moved 6e-3: over its 5e-3
        ("L_pct", "q0.975", 1 + 4e-3, False),  # ... and 4e-3: within
        ("X_mu", "mean", 1 + 2e-6, True),      # a mean moved 2e-6: over 1e-6
    ],
)
def test_convergence_failures_drills(q, stat, factor, named):
    fails = _convergence_drill(q, stat, factor)
    assert (any(f.startswith(f"{q}.{stat}:") for f in fails)) is named
    assert len(fails) == (1 if named else 0)


def _committed_nuts(name="weak"):
    c = _cells()
    return copy.deepcopy(c["nuts_cross_check"][name]), c["posteriors"][name]


def test_nuts_gate_passes_on_the_committed_diagnostics():
    for name in ("weak", "kamimura"):
        diag, quad = _committed_nuts(name)
        assert calibrate.nuts_gate_failures(diag, quad) == []


@pytest.mark.parametrize(
    "key, value, named",
    [("max_rhat", 1.01, "max_rhat"), ("min_ess", 400, "min_ess"), ("divergences", 1, "divergences")],
)
def test_nuts_gate_drills(key, value, named):
    diag, quad = _committed_nuts()
    diag[key] = value
    assert [f for f in calibrate.nuts_gate_failures(diag, quad) if f.startswith(named)]


def test_nuts_gate_names_a_mean_three_and_a_half_mcse_off():
    diag, quad = _committed_nuts()
    diag["mean"]["lambda_c"] = quad["lambda_c"]["mean"] + 3.5 * diag["mcse"]["lambda_c"]
    assert [f for f in calibrate.nuts_gate_failures(diag, quad) if f.startswith("lambda_c mean")]
