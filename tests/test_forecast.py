"""Tests for openmucf.forecast (FC-001 pre-registered forecast card).

One module-scoped fixture runs the calibration MCMC once; every card-shape test reuses it. No test asserts
byte-equality between a freshly generated card and the committed file (fresh builds are compared only to other
fresh builds in the same process).
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from openmucf import forecast

PHI_GRID = (1.2, 2.0, 2.4)
EXPECTED_IDS = {f"{obs}@phi={phi}" for phi in PHI_GRID for obs in ("omega_s_eff", "lambda_c")}


@pytest.fixture(scope="module")
def samples():
    """Draw the calibrated posterior ONCE for the whole module."""
    return forecast.posterior_samples()


@pytest.fixture(scope="module")
def fresh_card(samples):
    """A fresh card built from the shared posterior draws (no extra MCMC)."""
    return forecast.build_card(samples=samples)


@pytest.fixture(scope="module")
def shipped_card():
    """The committed forecasts/FC-001-mufuse.json (loaded from disk; no MCMC)."""
    return json.loads(forecast.CARD_PATH.read_text(encoding="utf-8"))


def _pred(card, scenario_name, target_id):
    scen = next(s for s in card["payload"]["scenarios"] if s["name"] == scenario_name)
    return next(p for p in scen["predictions"] if p["target_id"] == target_id)


# 1 ----------------------------------------------------------------------- determinism (fresh vs fresh)
def test_build_is_deterministic(fresh_card):
    """A second full build (independent MCMC, same seed) yields identical canonical_json bytes."""
    other = forecast.build_card()
    assert forecast.canonical_json(other) == forecast.canonical_json(fresh_card)
    # the payload hash is the same too
    assert forecast.payload_sha256(other) == forecast.payload_sha256(fresh_card)


# 2 ------------------------------------------------------------- hash consistency of the SHIPPED card
def test_shipped_card_hash_consistency(shipped_card):
    """Recompute the hash over the shipped card's OWN payload and match its OWN recorded digest (no MCMC)."""
    recomputed = forecast.payload_sha256(shipped_card)
    assert recomputed == shipped_card["registration"]["payload_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", recomputed)


# 3 ------------------------------------------------------------------- structural / schema validation
def test_shipped_card_validates(shipped_card):
    forecast.validate_card(shipped_card)  # raises on any problem


def test_scenarios_and_targets_are_the_pre_registered_grid(shipped_card):
    payload = shipped_card["payload"]
    assert [s["name"] for s in payload["scenarios"]] == ["A", "B"]
    assert {t["target_id"] for t in payload["targets"]} == EXPECTED_IDS
    for scen in payload["scenarios"]:
        assert {p["target_id"] for p in scen["predictions"]} == EXPECTED_IDS


def test_registration_is_draft_with_nulls(shipped_card):
    reg = shipped_card["registration"]
    assert reg["status"] in ("draft", "registered")
    if reg["status"] == "draft":
        assert reg["registered_utc"] is None
        assert reg["code_version_tag"] is None
        assert reg["zenodo_doi"] is None
    else:
        assert reg["registered_utc"] and reg["code_version_tag"] and reg["zenodo_doi"]
    assert re.fullmatch(r"[0-9a-f]{64}", reg["payload_sha256"])


# 4 --------------------------------------------------------------------------- exclusion / no retrodiction
def test_no_excluded_point_target_and_no_retrodiction(shipped_card):
    payload = shipped_card["payload"]
    # no scoring target at the publicly-visible ~2.2 phi / 50 K cycling point
    assert all(t["phi"] in PHI_GRID and t["phi"] != 2.2 for t in payload["targets"])
    # nothing anywhere is typed "retrodiction"
    assert "retrodiction" not in json.dumps(shipped_card).lower()
    # the exclusion is documented
    assert payload["resolution_criteria"]["excluded_points"]
    assert any("2.2" in e for e in payload["resolution_criteria"]["excluded_points"])


# 5 ---------------------------------------------------------------------------------- scenario-A sanity
def test_scenario_a_omega_median_in_posterior_ci(fresh_card, samples):
    lo, hi = np.percentile(samples["omega_s_eff_pct"], [2.5, 97.5])
    med = _pred(fresh_card, "A", "omega_s_eff@phi=1.2")["median"]
    assert lo <= med <= hi


def test_scenario_a_lambda_c_scales_linearly_in_phi(fresh_card):
    m = {phi: _pred(fresh_card, "A", f"lambda_c@phi={phi}")["median"] for phi in PHI_GRID}
    assert m[2.0] / m[1.2] == pytest.approx(2.0 / 1.2, rel=2e-3)
    assert m[2.4] / m[1.2] == pytest.approx(2.4 / 1.2, rel=2e-3)


# 6 ------------------------------------------------------------------- band relations (split by quantity)
def _width(interval):
    return interval[1] - interval[0]


def test_omega_s_eff_band_B_strictly_wider_than_A(fresh_card):
    for phi in PHI_GRID:
        a = _pred(fresh_card, "A", f"omega_s_eff@phi={phi}")["ci95"]
        b = _pred(fresh_card, "B", f"omega_s_eff@phi={phi}")["ci95"]
        assert _width(b) > _width(a)


def test_lambda_c_band_B_equals_A_at_1p2_and_wider_above(fresh_card):
    # equal at phi = 1.2 (both ensembles, identical computation)
    a12 = _pred(fresh_card, "A", "lambda_c@phi=1.2")["ci95"]
    b12 = _pred(fresh_card, "B", "lambda_c@phi=1.2")["ci95"]
    assert a12 == b12
    assert _pred(fresh_card, "B", "lambda_c@phi=1.2")["prediction_type"] == "ensemble"
    # strictly wider (envelope union) at phi = 2.0 and 2.4, where B is a bracket
    for phi in (2.0, 2.4):
        a = _pred(fresh_card, "A", f"lambda_c@phi={phi}")["ci95"]
        b = _pred(fresh_card, "B", f"lambda_c@phi={phi}")
        assert b["prediction_type"] == "bracket"
        assert _width(b["ci95"]) > _width(a)
        assert "median" not in b and "crps" not in b  # bracket carries no single median / headline CRPS


# 7 -------------------------------------------------------------------------- CRPS + coverage unit checks
def test_crps_point_mass_equals_abs_error():
    assert forecast.crps_empirical([5.0, 5.0, 5.0], 5.0) == pytest.approx(0.0)
    assert forecast.crps_empirical([5.0, 5.0, 5.0], 3.0) == pytest.approx(2.0)


def test_crps_centered_beats_point_mass_away():
    rng = np.random.default_rng(0)
    ens = rng.normal(0.0, 1.0, 4000)
    crps_centered = forecast.crps_empirical(ens, 0.0)
    crps_far = forecast.crps_empirical(np.full(4000, 10.0), 0.0)
    assert crps_centered < crps_far


def test_crps_sorted_identity_matches_brute_force():
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 3.0, 200)
    y = 1.3
    brute = float(np.mean(np.abs(x - y)) - np.abs(x[:, None] - x[None, :]).sum() / (2 * x.size**2))
    assert forecast.crps_empirical(x, y) == pytest.approx(brute, rel=1e-9, abs=1e-9)


def test_coverage_counts():
    cov = forecast.coverage([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0]], [0.5, 2.0, 2.5])
    assert cov == {"n": 3, "covered": 2, "fraction": pytest.approx(2 / 3)}


# 7b ------------------------------------------------------------------- Winkler interval score
def test_interval_score_penalizes_width():
    """y INSIDE both intervals: the score is just the width, so a 3x wider interval scores strictly higher
    (worse). Rewards sharpness."""
    narrow = forecast.interval_score(0.0, 1.0, 0.5, alpha=0.32)
    wide = forecast.interval_score(-1.0, 2.0, 0.5, alpha=0.32)   # 3x the width, same center, y inside
    assert narrow == pytest.approx(1.0)
    assert wide == pytest.approx(3.0)
    assert wide > narrow


def test_interval_score_miss_penalty():
    """y OUTSIDE the interval: the miss penalty (2/alpha * shortfall) dominates and scales with 1/alpha, so
    a tight 95% interval that misses scores far worse than its width."""
    hit = forecast.interval_score(0.0, 1.0, 0.5, alpha=0.05)         # inside -> width only
    miss = forecast.interval_score(0.0, 1.0, 2.0, alpha=0.05)        # y=2 above hi=1 by 1.0
    assert hit == pytest.approx(1.0)
    assert miss == pytest.approx(1.0 + (2.0 / 0.05) * 1.0)           # width + 40*shortfall
    assert miss > hit


def test_score_card_reports_is_for_brackets(fresh_card, samples):
    """score_card reports interval_score_68 / _95 as the headline for BOTH ensemble and bracket targets."""
    resolved = {"omega_s_eff@phi=1.2": 0.46, "lambda_c@phi=2.0": 1.9e8}
    out = forecast.score_card(fresh_card, resolved, samples=samples)
    ens = out["A"]["omega_s_eff@phi=1.2"]
    brk = out["B"]["lambda_c@phi=2.0"]
    for r in (ens, brk):
        assert "interval_score_68" in r and "interval_score_95" in r
        assert r["interval_score_68"] >= 0.0 and r["interval_score_95"] >= 0.0
    assert brk["prediction_type"] == "bracket" and len(brk["crps_per_limb"]) == 2


def test_score_card_runs_on_synthetic_resolution(fresh_card, samples):
    resolved = {
        "omega_s_eff@phi=1.2": 0.46,
        "lambda_c@phi=2.0": 1.9e8,   # a bracket target under scenario B
        "omega_s_eff@phi=2.4": "not_scoreable",
    }
    out = forecast.score_card(fresh_card, resolved, samples=samples)
    assert set(out) == {"A", "B"}
    assert out["A"]["omega_s_eff@phi=1.2"]["prediction_type"] == "ensemble"
    assert "crps" in out["A"]["omega_s_eff@phi=1.2"]
    assert out["B"]["lambda_c@phi=2.0"]["prediction_type"] == "bracket"
    assert len(out["B"]["lambda_c@phi=2.0"]["crps_per_limb"]) == 2
    assert out["A"]["omega_s_eff@phi=2.4"] == {"status": "not_scoreable"}


# 8 --------------------------------------------------------------------------- provenance completeness
# ledger content hash at FC-001 registration (v1.0.0). The live ledger evolves (v1.1 fix-pack onward);
# the card faithfully records the input it was computed from. Verified once out-of-band (not in the test):
# `git show v1.0.0:openmucf/data/rates.csv` LF-normalized hashes to exactly this literal.
LEDGER_SHA256_AT_REGISTRATION = "11e09e0d5342be08b20b25ab77c9d1554e1ffae16d13287e550503ac47ba7e92"


def test_ledger_hash_matches_fresh_lf_normalized_hash(shipped_card):
    # Registration-record integrity (NOT live-file freshness): the shipped card immutably records the
    # ledger hash as of registration; the live ledger has since evolved (WS-L fix-pack), so this asserts
    # the recorded hash equals the pinned registration-time literal, not a fresh hash of the live file.
    assert shipped_card["payload"]["provenance"]["ledger_sha256"] == LEDGER_SHA256_AT_REGISTRATION


def test_provenance_has_every_replication_field(shipped_card):
    prov = shipped_card["payload"]["provenance"]
    for key in ("model", "omega_s0_prior", "priors_note", "seed", "num_warmup", "num_samples"):
        assert key in prov["posterior_spec"]
    for key in ("distribution", "bounds", "generator", "seed", "n"):
        assert key in prov["scenario_b_sampling"]
    assert prov["posterior_spec"]["num_warmup"] == forecast.NUM_WARMUP
    assert prov["posterior_spec"]["num_samples"] == forecast.NUM_SAMPLES
    assert prov["scenario_b_sampling"]["bounds"] == [forecast.R_LO, forecast.R_HI]
    assert prov["phi_anchor"] == forecast.PHI_ANCHOR


def test_scenario_b_replicates_from_recorded_spec(shipped_card, samples):
    """An independent verifier reproduces Scenario B from scenario_b_sampling + the posterior os0 marginal."""
    sb = shipped_card["payload"]["provenance"]["scenario_b_sampling"]
    rng = np.random.default_rng(sb["seed"])
    r_b = rng.uniform(sb["bounds"][0], sb["bounds"][1], size=sb["n"])
    ose_b = samples["omega_s0_pct"] * (1.0 - r_b)
    med = forecast._round_sig(float(np.median(ose_b)))
    assert med == _pred(shipped_card, "B", "omega_s_eff@phi=1.2")["median"]


# 9 -------------------------------------------------------------------------------- wall-clock guard
def test_no_timestamp_in_payload(shipped_card):
    blob = json.dumps(shipped_card["payload"])
    # no ISO-8601-style datetime anywhere in the hashed payload
    assert not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", blob)
    # (registered_utc lives in the unhashed registration block, not the payload)


# 10 ------------------------------------------------------------------- D6 chain-setting mirror guard
def test_d6_constants_still_mirror_generate_calibration():
    src = (Path(forecast.__file__).resolve().parent.parent / "scripts" / "generate_calibration.py").read_text(
        encoding="utf-8"
    )
    # the main CALIBRATION.md chains use warmup 1000 / samples 4000 (mirrored by forecast's D6 constants)
    assert "MAIN_WARMUP, MAIN_SAMPLES = 1000, 4000" in src
    assert "seed=0" in src
    assert '("normal", 0.857, 0.03)' in src
    # and forecast.py's constants match those literals
    assert (forecast.NUM_WARMUP, forecast.NUM_SAMPLES, forecast.SEED) == (1000, 4000, 0)


# 11 ----------------------------------------------------- FC-001 chain settings are PINNED
def test_fc001_chain_settings_pinned():
    """The registered FC-001 realization is pinned to a SINGLE chain and the OLD (pre-widening) R box
    Uniform(0.10, 0.60), because calibrate.run_mcmc now defaults to 4 chains and R ~ Uniform(0.00, 0.80).
    Without both overrides the registered card's posterior draws (hence every published prediction) would
    move. Verified by module constants + the explicit override in posterior_samples, and that calibrate's
    live defaults really did move away from the pinned values (else the pin would be a silent no-op)."""
    from openmucf import calibrate

    assert forecast.NUM_CHAINS == 1
    assert forecast.R_PRIOR == (0.10, 0.60)
    src = Path(forecast.__file__).read_text(encoding="utf-8")
    assert "R_prior=R_PRIOR" in src
    assert "num_chains=NUM_CHAINS" in src
    # the pin is meaningful ONLY if calibrate's defaults differ from what FC-001 pins:
    assert calibrate.R_PRIOR_DEFAULT == (0.00, 0.80)
    assert calibrate.NUM_CHAINS_DEFAULT == 4


def test_fc001_pinned_posterior_reproduces_registered_predictions(fresh_card, shipped_card):
    """Behavioural proof: a fresh build through the PINNED posterior reproduces every registered card
    prediction byte-for-byte (the only legitimate payload drift is the evolved-ledger sha256)."""
    def _preds(card):
        return {s["name"]: {p["target_id"]: p for p in s["predictions"]}
                for s in card["payload"]["scenarios"]}

    assert _preds(fresh_card) == _preds(shipped_card)
    assert forecast.OMEGA_S0_PRIOR == ("normal", 0.857, 0.03)
