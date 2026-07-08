"""WS-T: the counts-level twin -- neutron time-spectrum forward model + idealized estimator (fast tests).

The forward model runs the v1 cycle CHANNELS-OFF (the loss channels stay at their 0.0 defaults), so the
twin reduces to the established engine. The slow interval-calibration coverage test lives in
tests/test_twin_coverage.py (marked `slow`, deselected from the default run)."""

from pathlib import Path

import numpy as np
from numpyro.infer.util import log_density

import openmucf
from openmucf import analytic, cycle, likelihood, load_rates, provenance, twin
from openmucf.constants import LAMBDA_0
from openmucf.rates import RATES_CSV

REPO_ROOT = Path(openmucf.__file__).resolve().parent.parent
TWIN_MANIFEST = REPO_ROOT / "TWIN_MANIFEST.json"
FC001_CARD = REPO_ROOT / "forecasts" / "FC-001-mufuse.json"

# Canonical liquid operating point + histogram used across the twin (matches generate_twin_audit.py).
T_EDGES = np.linspace(0.0, 30.0e-6, 65)


def _canonical():
    rates = load_rates()
    cp = cycle.params_from_conditions(rates, 300.0, 1.2, 0.5)
    xmu = float(cycle.fusions_per_muon_ode(**cp))
    ose = cp["omega_s_eff"]
    lc = twin.implied_cycling_rate(xmu, ose)
    return cp, xmu, ose, lc


def test_fusion_rate_density_integrates_to_xmu():
    """F(t) = dN_fus/dt integrates back to the yield X_mu (a fine-grid trapezoid; the residual is the
    quadrature over the sharp early rise, not the model -- the EXACT accumulator check is the next test)."""
    cp, xmu, _, _ = _canonical()
    tg = np.linspace(0.0, 30.0e-6, 20001)
    integ = np.trapezoid(np.asarray(twin.fusion_rate_density(tg, **cp)), tg)
    assert abs(integ - xmu) / xmu < 1e-4


def test_expected_counts_sums_to_yield_plus_background():
    """expected_counts reads the N_fus accumulator at the edges (exact), so the histogram sums to
    n_mu*eff*X_mu + background_rate*total_width to ~1e-6."""
    cp, xmu, _, _ = _canonical()
    n_mu, eff, bkg = 1.0e6, 1.0, 5.0e9
    total = float(np.sum(np.asarray(twin.expected_counts(T_EDGES, cp, n_mu, eff, bkg))))
    expect = n_mu * eff * xmu + bkg * (T_EDGES[-1] - T_EDGES[0])
    assert abs(total - expect) / expect < 1e-6


def test_synthetic_spectrum_reproducible():
    """Same seed -> identical Poisson draw; a different seed differs."""
    cp, _, _, _ = _canonical()
    exp = twin.expected_counts(T_EDGES, cp, 1.0e6, 1.0, 0.0)
    a = twin.synthetic_spectrum(T_EDGES, exp, seed=0)
    b = twin.synthetic_spectrum(T_EDGES, exp, seed=0)
    c = twin.synthetic_spectrum(T_EDGES, exp, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_two_exponential_recovers_lambda_n_transient_free():
    """On transient-free (closed-form single-exponential) synthetic truth, the idealized estimator recovers
    lambda_n to < 1% -- the in-suite twin of gate G-T1 with the two-pool transient removed."""
    _, _, ose, lc = _canonical()
    ln_true = twin.disappearance_rate(ose, lc)
    exp = np.asarray(likelihood.expected_counts_closed_form(T_EDGES, ose, lc, 1.0e6, 0.0))
    syn = twin.synthetic_spectrum(T_EDGES, exp, seed=0)
    fit = twin.fit_two_exponential(T_EDGES, syn, t_min=2.0e-6, lambda_c=lc)
    assert abs(fit["lambda_n"] - ln_true) / ln_true < 0.01


def test_gate_g_t1_ode_synthetic_under_1pct():
    """Gate G-T1: synthetic spectrum from the FULL ODE at the canonical OP, fit at t_min=2 us, recovers the
    analytic disappearance rate to < 1% (the residual is the two-hyperfine-pool structure)."""
    cp, xmu, ose, lc = _canonical()
    ln_true = twin.disappearance_rate(ose, lc)
    exp = twin.expected_counts(T_EDGES, cp, 1.0e6, 1.0, 0.0)
    syn = twin.synthetic_spectrum(T_EDGES, exp, seed=0)
    fit = twin.fit_two_exponential(T_EDGES, syn, t_min=2.0e-6, lambda_c=lc)
    assert abs(fit["lambda_n"] - ln_true) / ln_true < 0.01


def test_bias_positive_and_monotone_in_ct():
    """At t_min=0.5 us the idealized estimator's bias(lambda_n) is positive for every c_t and monotone
    (decreasing) as the cycle speeds up with c_t -- sign/direction only, on noise-free expected counts."""
    rates = load_rates()
    biases = []
    for c_t in (0.2, 0.5, 0.8):
        cp = cycle.params_from_conditions(rates, 300.0, 1.2, c_t)
        xm = float(cycle.fusions_per_muon_ode(**cp))
        ose = cp["omega_s_eff"]
        lc = twin.implied_cycling_rate(xm, ose)
        ln = twin.disappearance_rate(ose, lc)
        exp = twin.expected_counts(T_EDGES, cp, 1.0e6, 1.0, 0.0)
        fit = twin.fit_two_exponential(T_EDGES, exp, t_min=0.5e-6, lambda_c=lc)
        biases.append((fit["lambda_n"] - ln) / ln)
    assert all(b > 0 for b in biases), biases
    assert biases[0] > biases[1] > biases[2], biases


def test_disappearance_rate_matches_inverse_xmu():
    """lambda_n = lambda_0 + omega_s_eff*lambda_c equals lambda_c / X_mu (the closed-form identity)."""
    ose, lc = 0.0055705, 1.4e8
    ln = twin.disappearance_rate(ose, lc)
    x_an = float(analytic.fusions_per_muon(ose, lc, LAMBDA_0))
    assert abs(ln - lc / x_an) / (lc / x_an) < 1e-12


def test_spectrum_model_log_density_finite_at_truth():
    """The counts-level model has a finite log-density at the generating truth (well-posed likelihood)."""
    ose, lc, amp, bkg = 0.557, 1.30e8, 1.0e5, 3.0e7
    exp = np.asarray(likelihood.expected_counts_closed_form(T_EDGES, ose / 100.0, lc, amp, bkg))
    counts = np.random.default_rng(0).poisson(exp)
    params = dict(omega_s_eff_pct=ose, lambda_c=lc, amplitude=amp, background_rate=bkg)
    lp, _ = log_density(likelihood.spectrum_model, (), dict(t_edges=T_EDGES, counts=counts), params)
    assert np.isfinite(float(lp))


def test_fit_spectrum_smoke_recovers_omega_s_eff():
    """A tiny-settings NUTS fit recovers omega_s_eff within 3 posterior sd (mixing/plumbing smoke test).

    The posterior sits slightly high of truth because omega_s_eff is separated from lambda_c only through
    the informative lambda_c prior (the identifiability documented in likelihood.py); 3 sd is the honest
    loose bound for a smoke fit."""
    ose_true, lc, amp, bkg = 0.557, 1.30e8, 1.0e5, 3.0e7
    exp = np.asarray(likelihood.expected_counts_closed_form(T_EDGES, ose_true / 100.0, lc, amp, bkg))
    counts = np.random.default_rng(7).poisson(exp)
    s = likelihood.fit_spectrum(T_EDGES, counts, num_warmup=250, num_samples=500, seed=0)
    mean = float(np.mean(s["omega_s_eff_pct"]))
    sd = float(np.std(s["omega_s_eff_pct"]))
    assert abs(mean - ose_true) < 3.0 * sd, (mean, sd)


def test_ledger_lambda_c_prior_bounds_from_ledger():
    """The likelihood's lambda_c prior bounds at phi=1.2 equal the ledger lambda_c_liquid dist bounds
    [1.00e8, 1.45e8]; scaling to another density is phi-linear."""
    lo, hi = likelihood.ledger_lambda_c_bounds(1.2)
    led_lo, led_hi = load_rates().dist_bounds("lambda_c_liquid")
    assert (lo, hi) == (led_lo, led_hi)
    lo2, hi2 = likelihood.ledger_lambda_c_bounds(2.4)
    assert abs(lo2 - 2.0 * led_lo) < 1e-3 and abs(hi2 - 2.0 * led_hi) < 1e-3


def test_twin_manifest_consistent_with_committed_doc():
    """TWIN_MANIFEST.json verifies against the committed TWIN_AUDIT.md and records current input SHAs
    (in-suite proxy for the byte-identical regeneration that `make audit` enforces)."""
    assert provenance.check_manifest(TWIN_MANIFEST, repo_root=REPO_ROOT) == []
    import json

    manifest = json.loads(TWIN_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["inputs"]["rates_csv_sha256"] == provenance.file_sha256(RATES_CSV)
    assert manifest["inputs"]["fc001_card_sha256"] == provenance.file_sha256(FC001_CARD)
    ids = {e["id"] for e in manifest["entries"]}
    assert {"gate_bias_pct", "band_ln_phi1.2_med", "bias_ln_ct0.5_tmin2"} <= ids
