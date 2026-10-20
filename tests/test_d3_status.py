"""The failed fresh-solve record is validated and reaches the component header."""
import importlib.util
import json
import shutil
import sys
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from openmucf.g4 import d3_contract as contract
from openmucf.g4.sources import mudirac130 as md

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def keys():
    return {tuple(k) for k in json.loads((ROOT / "data/g4/d3/reference/expected_keys.json").read_text())}


@pytest.fixture
def status():
    return json.loads((ROOT / contract.NUMERICAL_CHECK_STATUS_RELPATH).read_text())


def write_status(root, status):
    path = root / contract.NUMERICAL_CHECK_STATUS_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(status).encode("ascii"))


def witness(status):
    return next(r for r in status["fresh_solve"]["failures"]
                if (r["Z"], r["A"], r["uehling"], r["quantity"]) == (4, 9, "off", "e3-e4"))


def test_status_accepts_record_and_keeps_decimal_numbers(tmp_path, status, keys):
    write_status(tmp_path, status)
    loaded = contract.load_numerical_check_status(tmp_path, keys)
    assert loaded["status"] == "REPRODUCTION_FAILED"
    assert isinstance(loaded["fresh_solve"]["maximum_change_of_target"], Decimal)
    assert loaded["fresh_solve"]["failed_rows"] == len(loaded["fresh_solve"]["failures"]) > 0


@pytest.mark.parametrize("change", ["json", "missing", "shape", "criterion"])
def test_unusable_status_is_refused(tmp_path, status, keys, change):
    if change == "missing":
        del status["fresh_solve"]
    elif change == "shape":
        status["fresh_solve"] = None
    elif change == "criterion":
        status["fresh_solve"]["criteria"]["reference_limit_of_target"] = "not numeric"
    write_status(tmp_path, status)
    if change == "json":
        (tmp_path / contract.NUMERICAL_CHECK_STATUS_RELPATH).write_bytes(b"{")
    with pytest.raises(md.CellError, match="numerical check status"):
        contract.load_numerical_check_status(tmp_path, keys)


@pytest.mark.parametrize("change,match", [
    ("schema", "schema"), ("token", "status vocabulary"), ("empty", "nonempty failure count"),
    ("count", "nonempty failure count"), ("outside", "failure inventory"),
    ("duplicate", "failure inventory"), ("switch", "failure inventory"),
    ("rows", "population"), ("cases", "population"),
    ("sharpness_removed", "derived failure flags"), ("sharpness_inconsistent", "derived failure flags"),
    ("ref_flag", "derived failure flags"), ("num_flag", "derived failure flags"),
    ("certificate_flag", "derived failure flags"), ("flags_order", "derived failure flags"),
    ("flags_duplicate", "derived failure flags"), ("empty_flags", "derived failure flags"),
    ("verdict", "fresh verdict vocabulary"), ("maximum", "maximum change"),
    ("maximum_q", "maximum change"),
    ("negative", "finite nonnegative numbers"), ("nonfinite", "numerical check status"),
    ("negative_criterion", "finite nonnegative numbers"),
])
def test_status_rejects_inconsistent_records(tmp_path, status, keys, change, match):
    fresh = status["fresh_solve"]
    be = witness(status)
    row = fresh["failures"][0]
    if change == "schema":
        status["schema"] += 1
    elif change == "token":
        status["status"] = "REPRODUCED"
    elif change == "empty":
        fresh["failures"] = []
        fresh["failed_rows"] = 0
    elif change == "count":
        fresh["failed_rows"] -= 1
    elif change == "outside":
        row["Z"], row["A"] = 1, 1
    elif change == "duplicate":
        fresh["failures"].append(deepcopy(row))
        fresh["failed_rows"] += 1
    elif change == "switch":
        row["uehling"] = "unknown"
    elif change in ("rows", "cases"):
        fresh["population"][change] -= 1
    elif change == "sharpness_removed":
        be["flags"].remove("reference_sharpness")
    elif change == "sharpness_inconsistent":
        be["fresh_b_ref_of_target"] = 0.19
    elif change == "ref_flag":
        row["flags"].remove("b_ref_eV")
    elif change == "num_flag":
        row["flags"].remove("b_num_eV")
    elif change == "certificate_flag":
        be["flags"].remove("fresh_certificate")
    elif change == "flags_order":
        row["flags"].reverse()
    elif change == "flags_duplicate":
        row["flags"].append(row["flags"][0])
    elif change == "empty_flags":
        row.update(b_ref_change_of_target=0, b_num_change_of_target=0,
                   fresh_b_ref_of_target=0, fresh_verdict="QUALIFIED", flags=[])
    elif change == "verdict":
        be["fresh_verdict"] = "UNQUALIFIED"
    elif change == "maximum":
        fresh["maximum_change_of_target"] = 0
    elif change == "maximum_q":
        fresh["maximum_q_ref_change_of_target"] = fresh["maximum_change_of_target"] + 1
    elif change == "negative":
        fresh["maximum_q_ref_change_of_target"] = -1
    elif change == "negative_criterion":
        fresh["criteria"]["reference_limit_of_target"] = "-1"
    else:
        fresh["maximum_q_ref_change_of_target"] = float("inf")
    write_status(tmp_path, status)
    with pytest.raises(md.CellError, match=match):
        contract.load_numerical_check_status(tmp_path, keys)


def test_missing_status_stops_bundle_and_build(tmp_path, monkeypatch):
    shutil.copytree(ROOT / md.D3_RELDIR, tmp_path / md.D3_RELDIR)
    (tmp_path / contract.NUMERICAL_CHECK_STATUS_RELPATH).unlink()
    with pytest.raises(md.CellError, match="numerical check status"):
        contract.load_bundle(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "d3_status_generator", ROOT / "scripts/generate_g4data.py")
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generator
    spec.loader.exec_module(generator)
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    monkeypatch.setattr(generator.d1src, "load", lambda path: pytest.fail("status admission bypassed"))
    with pytest.raises(md.CellError, match="numerical check status"):
        generator.build_dataset_artifacts()


def test_component_status_and_digest_reach_serialized_header(tmp_path):
    shutil.copytree(ROOT / md.D3_RELDIR, tmp_path / md.D3_RELDIR)
    bundle = contract.load_bundle(tmp_path)
    before = json.loads(contract.render_components(tmp_path, bundle).splitlines()[0])
    assert before["numerical_check_status"] == bundle.numerical_check_status["status"]
    path = tmp_path / contract.NUMERICAL_CHECK_STATUS_RELPATH
    path.write_bytes(path.read_bytes() + b" ")
    after = json.loads(contract.render_components(tmp_path, bundle).splitlines()[0])
    assert after["inputs_sha256"] != before["inputs_sha256"]
    assert after["numerical_check_status"] == before["numerical_check_status"]
