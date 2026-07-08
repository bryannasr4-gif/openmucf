"""Phase 2.4 trust gate: the reproducible validation targets must pass."""

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
    """The executed pre-registered ratio clause appears as V_yamashita_ratio and passes."""
    res = {r.target_id: r for r in validate.run(load_rates())}
    assert "V_yamashita_ratio" in res
    assert res["V_yamashita_ratio"].passed is True
