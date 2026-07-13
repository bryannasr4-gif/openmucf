"""Phase 2.4 trust gate: the reproducible validation targets must pass; the class-tiered
scoreboard exposes the three registered independent targets that FAIL by design."""

import pytest

from openmucf import load_rates, validate


def _by_id():
    return {r.target_id: r for r in validate.run(load_rates())}


def test_reproducible_targets_pass():
    res = _by_id()
    for tid in (
        "V_kouchen_base",
        "V_kouchen_best",
        "V_petitjean",
        "V_yamashita_lcT",
        "V_breunlich_lambdac",
        "V_faifman_peak",
    ):
        r = res[tid]
        assert r.passed is True, f"{tid}: predicted {r.predicted} (tol {r.tolerance})"


def test_faifman_peak_now_reproduced_by_resonance_model():
    # Previously DEFERRED (placeholder was thermal-only); the resonance model now hits the peak.
    r = _by_id()["V_faifman_peak"]
    assert r.passed is True
    assert abs(r.predicted - 7.1e9) / 7.1e9 < 0.25


def test_report_renders():
    md = validate.report_markdown(validate.run(load_rates()))
    assert "VALIDATION.md" in md and "PASS" in md
    # the class column and the new class-tiered summary format
    assert "| target | class | observed |" in md
    assert "Distinct tests:" in md
    assert "**FAIL**" in md  # the three registered independent rows fail by design


def test_verdict_is_csv_driven(tmp_path):
    # Proves the trust gate actually reads validation_targets.csv: mutate a target value -> verdict flips.
    import csv

    from openmucf.rates import TARGETS_CSV

    with open(TARGETS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["target_id"].strip() == "V_kouchen_base":
            row["value"] = "999"  # engine predicts ~114.5; 999 +-10% cannot contain it -> FAIL
    mutated = tmp_path / "validation_targets_mutated.csv"
    with open(mutated, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    mutated_res = {r.target_id: r for r in validate.run(load_rates(), targets_csv=str(mutated))}
    assert mutated_res["V_kouchen_base"].passed is False
    # control: the unmutated CSV still passes it
    base_res = {r.target_id: r for r in validate.run(load_rates())}
    assert base_res["V_kouchen_base"].passed is True


def test_yamashita_ratio_target():
    """The re-anchored ratio clause FAILs by design against the digitized full curve (registered finding).

    Re-anchored 2026-07-13: the comparator moved from the ~1.45 under-read to the sourced digitized
    centreline (2.358), which is strictly harder; the engine ratio ~1.31 now flips PASS->FAIL.
    """
    res = {r.target_id: r for r in validate.run(load_rates())}
    assert "V_yamashita_ratio" in res
    assert res["V_yamashita_ratio"].passed is False
    assert "registered" in res["V_yamashita_ratio"].note.lower()


def test_categories_pinned():
    """Every result carries its DECIDED category tier (test-pinned literals)."""
    res = _by_id()
    expected = {
        "V_kouchen_base": "reproduction (fed input)",
        "V_kouchen_best": "reproduction (fed input)",
        "V_petitjean": "reproduction (fed input)",
        "V_yamashita_lcT": "shape (calibrated model)",
        "V_yamashita_ratio": "independent",
        "V_yamashita_curve": "independent",
        "V_breunlich_lambdac": "anchor-consistency",
        "V_faifman_peak": "anchor-consistency",
        "V_petitjean_omega": "independent",
        "V_faifman_900K": "independent",
        "V_faifman_lowT": "independent",
        "V_nagamine_trend": "independent",
    }
    assert {tid: r.category for tid, r in res.items()} == expected
    for r in res.values():
        assert r.category in validate.CATEGORIES


def test_categories_are_the_pinned_literals():
    """The claim-tier literals are pinned (VALIDATION.md, the README trust map, and docs depend on them)."""
    assert validate.CATEGORIES == (
        "self-consistency",
        "reproduction (fed input)",
        "anchor-consistency",
        "shape (calibrated model)",
        "independent",
    )


def test_distinct_test_count_dedups_shape_rows():
    """The three Yamashita rows share a dedup group and are counted once (10 distinct of 12)."""
    results = validate.run(load_rates())
    assert len(results) == 12
    groups = [
        r.dedup_group
        for r in results
        if r.target_id in ("V_yamashita_lcT", "V_yamashita_ratio", "V_yamashita_curve")
    ]
    assert len(groups) == 3 and all(g and g == groups[0] for g in groups)  # shared, non-empty group
    assert "Distinct tests: 10" in validate.report_markdown(results)


def test_registered_fail_targets():
    """The three independent targets FAIL by design and are labelled registered findings."""
    res = _by_id()
    for tid in ("V_petitjean_omega", "V_faifman_900K", "V_faifman_lowT"):
        r = res[tid]
        assert r.passed is False, f"{tid} unexpectedly passed (predicted {r.predicted})"
        assert "registered" in r.note
    # the ledger pushforward omega_s0*(1-R_col) = 0.55705% against the [0.40,0.50] band
    assert res["V_petitjean_omega"].predicted == pytest.approx(0.55705, rel=1e-3)


def test_summary_counts():
    """The summary line reports the class-tiered counts, including zero passing independent rows."""
    md = validate.report_markdown(validate.run(load_rates()))
    summary = next(ln for ln in md.splitlines() if ln.startswith("**Summary"))
    assert "(0 independent)" in summary
    assert "5 fail" in summary  # 3 original registered + the re-anchored ratio + the curve
    assert "Distinct tests:" in summary


def test_expected_fail_guard():
    """A registered expected-FAIL that PASSES is a bug/tolerance error, not a success (never ship it)."""
    res = _by_id()
    for tid in ("V_petitjean_omega", "V_faifman_900K", "V_faifman_lowT"):
        assert res[tid].passed is False, (
            f"surprise PASS on a registered expected-FAIL target {tid} -- STOP and root-cause "
            "(these are pre-registered to FAIL; see PRE_REGISTRATION.md)"
        )


def test_yamashita_expected_fail_guard():
    """The two sourced Yamashita expected-FAILs (the re-anchored ratio row and the 800 K curve point)
    are pre-registered to FAIL; a surprise PASS on either is a bug/tolerance error, not a success."""
    res = _by_id()
    assert res["V_yamashita_ratio"].passed is False, (
        "surprise PASS on the re-anchored V_yamashita_ratio -- STOP and root-cause "
        "(pre-registered expected-FAIL; see PRE_REGISTRATION.md)"
    )
    assert res["V_yamashita_curve"].passed is False, (
        "surprise PASS on V_yamashita_curve -- STOP and root-cause "
        "(pre-registered expected-FAIL; see PRE_REGISTRATION.md)"
    )
    # the 800 K point specifically is the registered expected-FAIL
    assert "800 K FAIL" in res["V_yamashita_curve"].note


def test_yamashita_ratio_band_robustness():
    """The PASS->FAIL flip is robust to the digitization band: the engine ratio fails +-30% at BOTH
    edges of the band (centreline +- the digitization half-width propagated from the +-8% per point)."""
    import csv as _csv

    from openmucf.rates import TARGETS_CSV

    pred = _by_id()["V_yamashita_ratio"].predicted  # engine 800/300 ratio ~1.311
    with open(TARGETS_CSV, newline="") as f:
        row = next(r for r in _csv.DictReader(f) if r["target_id"].strip() == "V_yamashita_ratio")
    centre = float(row["value"])
    rel = 2**0.5 * 0.08  # per-point 8% digitization uncertainty propagated to the ratio
    lo, hi = centre * (1.0 - rel), centre * (1.0 + rel)
    assert not validate._within(pred, lo, "+-30%"), f"ratio unexpectedly PASSes at band-low {lo}"
    assert not validate._within(pred, hi, "+-30%"), f"ratio unexpectedly PASSes at band-high {hi}"


def test_yamashita_ratio_value_matches_digitized_csv():
    """The committed ratio anchor equals the digitized CSV's own lambda_c(800)/lambda_c(300) -- one
    source of truth (the digitized data file), no drift between the anchor and the curve."""
    import csv as _csv

    from openmucf.rates import TARGETS_CSV

    yk = validate._load_yamashita_curve()
    dig_ratio = yk[800] / yk[300]
    with open(TARGETS_CSV, newline="") as f:
        row = next(r for r in _csv.DictReader(f) if r["target_id"].strip() == "V_yamashita_ratio")
    assert abs(float(row["value"]) - dig_ratio) / dig_ratio < 1e-3


def test_physics_mutation_flips_verdicts():
    """The suite CAN fail on a physics change: mutate an input/resonance -> a verdict moves."""
    import dataclasses

    import openmucf.formation as formation
    from openmucf.rates import RatesTable, omega_fraction

    base = load_rates()
    base_pred = {r.target_id: r for r in validate.run(base)}["V_petitjean_omega"].predicted

    # (i) R_col -> 0.60 drives V_petitjean_omega's derived sticking down toward/through the band.
    patched = dict(base._rates)
    patched["R_col"] = dataclasses.replace(patched["R_col"], value=0.60)
    mut = RatesTable(patched)
    pred = {r.target_id: r for r in validate.run(mut)}["V_petitjean_omega"].predicted
    assert pred == pytest.approx(omega_fraction(base["omega_s0"]) * (1.0 - 0.60) * 100.0)
    assert pred < base_pred  # moved toward the pass side (physics-driven, CSV/ledger-sourced)

    # (ii) halve the measured F=1 peak amplitude -> V_faifman_peak drops out of its +-25% band -> FAIL.
    e_r, amp, wid = formation._RESONANCES[1][0]
    orig_f1 = list(formation._RESONANCES[1])
    try:
        formation._RESONANCES[1][0] = (e_r, 0.5 * amp, wid)
        res = {r.target_id: r for r in validate.run(load_rates())}
        assert res["V_faifman_peak"].passed is False
    finally:
        formation._RESONANCES[1][:] = orig_f1
