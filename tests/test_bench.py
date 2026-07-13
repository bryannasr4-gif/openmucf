"""muCF-Bench: the case registry (openmucf.bench) over the validation trust gate + JSON reproductions."""

import json

import pytest

from openmucf import bench, load_rates, validate

JSON_IDS = {"jones-1986", "kou-chen-2026"}


def _rates():
    return load_rates()


def test_case_ids_are_7_validation_plus_2_json():
    ids = bench.case_ids(_rates())
    assert len(ids) == 9
    assert set(bench.VALIDATION_IDS) | JSON_IDS == set(ids)
    assert len(bench.VALIDATION_IDS) == 7


def test_validation_ids_match_validate_emission():
    """VALIDATION_IDS mirrors the reproduction/consistency-tier results validate.run() emits; the
    registered independent-prediction FAIL findings are executed in VALIDATION.md but deliberately
    excluded from this friendly-reproduction registry (guards drift both ways)."""
    results = validate.run(_rates())
    emitted = {r.target_id for r in results}
    registered_fail = {r.target_id for r in results if r.passed is False and r.category == "independent"}
    assert registered_fail == {
        "V_petitjean_omega",
        "V_faifman_900K",
        "V_faifman_lowT",
        "V_yamashita_ratio",
        "V_yamashita_curve",
    }
    assert set(bench.VALIDATION_IDS) == emitted - registered_fail


def test_run_case_validation_id_reproduces_validate_verdict():
    r = _rates()
    by_id = {res.target_id: res for res in validate.run(r)}
    for vid in bench.VALIDATION_IDS:
        want = by_id[vid].passed
        expect = "DEFERRED" if want is None else ("PASS" if want else "FAIL")
        got = bench.run_case(r, vid)
        assert got.verdict == expect, f"{vid}: {got.verdict} != {expect}"
        assert got.type == "validation"


def test_kouchen_case_passes_and_reproduces_committed_values():
    got = bench.run_case(_rates(), "kou-chen-2026")
    assert got.verdict == "PASS"
    assert got.type == "reproduction"
    # reproduces the committed VALIDATION.md predictions 114.5 / 160.3
    assert got.predicted == "114.5 / 160.3"
    assert got.expected == "112.6 / 156.5"


def test_jones_case_ships_blocked_pending():
    got = bench.run_case(_rates(), "jones-1986")
    assert got.verdict == "PENDING"
    assert got.type == "reproduction"
    # the blocking DOCUMENT is named publicly (never ACQUISITIONS.md); no guessed conditions
    assert "Phys. Rev. Lett. 56, 588" in got.source
    assert "ACQUISITIONS" not in got.note
    case = bench.load_cases()["jones-1986"]
    assert case["status"] == "blocked-acquisition"
    assert case["inputs"] == []


def test_run_all_and_report_markdown_render_with_footnotes():
    results = bench.run_all(_rates())
    assert len(results) == 9
    md = bench.report_markdown(results)
    assert "BENCHMARKS.md" in md
    # the mapping/exclusion footnotes (now five excluded registered-FAIL findings)
    assert "V_petitjean_omega" in md and "V_yamashita_ratio" in md
    assert "A_acceleron_density" in md and "A_acceleron_anomaly" in md
    # the friendly-reproduction registry excludes every registered FAIL, so it stays 0 fail
    assert "0 fail" in md
    # a generated public doc must carry no internal workstream id
    assert "WS-" not in md


def test_load_cases_returns_two_schema_valid_cases():
    cases = bench.load_cases()
    assert set(cases) == JSON_IDS
    for case in cases.values():
        assert set(case) >= bench._REQUIRED_KEYS


def test_run_case_unknown_id_raises():
    with pytest.raises(KeyError):
        bench.run_case(_rates(), "does-not-exist")


def test_schema_check_catches_a_bad_case_file(tmp_path):
    # A structurally broken case must be rejected by the loader (stdlib schema check).
    bad = {"id": "bad", "type": "reproduction", "engine": "warp", "status": "active", "inputs": []}
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="bad bench case"):
        bench.load_cases(cases_dir=tmp_path)


def test_fail_case_yields_fail_verdict(tmp_path):
    # A synthetic active case whose published value the engine cannot hit must FAIL (not silently pass).
    case = {
        "id": "synthetic-fail",
        "type": "reproduction",
        "title": "synthetic",
        "engine": "analytic",
        "inputs": [
            {"label": "x", "omega_s_eff_pct": 0.45, "lambda_c": 1e8,
             "published_value": 1.0, "tolerance": "+-1%"}
        ],
        "provenance": {"source_bibkey": "none", "locator": "synthetic", "input_basis": "synthetic test"},
        "status": "active",
        "notes": "synthetic FAIL case",
    }
    (tmp_path / "synthetic-fail.json").write_text(json.dumps(case), encoding="utf-8")
    got = bench.run_case(_rates(), "synthetic-fail", cases_dir=tmp_path)
    assert got.verdict == "FAIL"
