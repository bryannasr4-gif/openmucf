"""The real loader compares every summary cell and propagates evaluated adverse records."""
from __future__ import annotations

import functools
from copy import deepcopy
from pathlib import Path

import pytest

from openmucf.g4 import d3_centroids as centroids
from openmucf.g4 import d3_contract as contract
from openmucf.g4 import d3_numerical as numerical
from openmucf.g4.sources import mudirac130 as md

ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def outputs():
    return md.load_outputs(ROOT)


@functools.lru_cache(maxsize=1)
def evaluated():
    return numerical.evaluate_all(ROOT)


def csv_rows():
    return list(md.read_rows(ROOT / contract.NUMERICAL_CHECK_RELPATH, contract.NUMERICAL_CHECK_COLUMNS))


def inject_rows(monkeypatch, rows):
    original = md.read_rows

    def read(path, columns):
        if Path(path).name == "numerical_check.csv":
            return iter(rows)
        return original(path, columns)

    monkeypatch.setattr(md, "read_rows", read)


@pytest.mark.parametrize("column", ["set", "Q_ref_eV", "b_ref_eV", "b_num_eV", "target_eV", "verdict",
                                    "bue_1_eV", "probe_c_bohr", "probe_t_bohr"])
def test_loader_rejects_forged_summary(monkeypatch, column):
    evaluated()
    rows = [(where, dict(row)) for where, row in csv_rows()]
    witness = next(row for _, row in rows if row["Z"] == "6" and row["A"] == "12"
                   and row["uehling"] == "on" and row["quantity"] == "K")
    if column == "set":
        witness[column] = "audit"
    elif column == "verdict":
        witness[column] = "NUMERICAL_FAIL"
    elif column == "Q_ref_eV":
        witness[column] = str(float(witness[column]) + 0.1)
    else:
        witness[column] = "0.123456789"
    # Keep twin consistency so the evidence comparator, rather than the old pairing guard,
    # is what refuses a forged probe value.
    if column.startswith("probe_"):
        for _, row in rows:
            if (row["Z"], row["A"]) == ("6", "12"):
                row[column] = witness[column]
    inject_rows(monkeypatch, rows)
    monkeypatch.setattr(numerical, "evaluate_all", lambda root: evaluated())
    with pytest.raises(md.CellError, match=f"{column} disagrees"):
        contract.load_numerical_check(ROOT, outputs())


def test_loader_requires_complete_expected_keys(monkeypatch):
    evaluated()
    inject_rows(monkeypatch, csv_rows()[1:])
    monkeypatch.setattr(numerical, "evaluate_all", lambda root: evaluated())
    with pytest.raises(md.CellError, match="keys differ"):
        contract.load_numerical_check(ROOT, outputs())


def test_loader_calls_evaluator_with_supplied_root(monkeypatch, tmp_path):
    inject_rows(monkeypatch, csv_rows())
    seen = []

    def evaluate(root):
        seen.append(root)
        raise numerical.EvidenceError("sentinel supplied root")

    monkeypatch.setattr(numerical, "evaluate_all", evaluate)
    with pytest.raises(md.CellError, match="sentinel supplied root"):
        contract.load_numerical_check(tmp_path, outputs())
    assert seen == [tmp_path]


def test_loader_exposes_unrounded_evaluated_quantities():
    values = evaluated()
    loaded = contract.load_numerical_check(ROOT, outputs())
    assert set(loaded) == set(values)
    for key, row in loaded.items():
        value = values[key]
        assert (row.q_full_ev, row.q_ref_full_ev, row.b_ref_full_ev, row.b_num_full_ev) == (
            value.q_ev, value.q_ref_ev, value.b_ref_ev, value.b_num_ev)
    assert any(row.b_num_full_ev != float(row.b_num_ev) for row in loaded.values())


def test_consistent_adverse_record_reaches_components_and_centroids(monkeypatch):
    evidence = deepcopy(numerical.load_evidence(ROOT))
    evidence.payloads[6, 12, True]["A"]["K1"]["prod"]["B_Ha"] -= 0.1 / numerical.EH
    result = numerical.evaluate_all(ROOT, evidence)
    key = 6, 12, True, "K-e2"
    assert result[key].prereg.gates["G1"] is True
    assert result[key].prereg.gates["G3"] is False
    rows = [(where, result[int(row["Z"]), int(row["A"]), row["uehling"] == "on",
                           row["quantity"]].historical_summary()) for where, row in csv_rows()]
    inject_rows(monkeypatch, rows)
    monkeypatch.setattr(numerical, "evaluate_all", lambda root: result)
    checks = contract.load_numerical_check(ROOT, outputs())
    assert checks[key].verdict == "NUMERICAL_FAIL"
    assert centroids.numerical_record(ROOT)[6, 12][-1] == "NUMERICAL_FAIL"
    assert next(r for r in centroids.centroid_rows(ROOT)
                if (r["Z"], r["A"]) == ("6", "12"))["numerical_verdict"] == "NUMERICAL_FAIL"
    # The same primitive perturbation also changes the exported K certificate.
    record = next(r for r in contract.components(ROOT)
                  if (r.get("Z"), r.get("A"), r.get("quantity")) == (6, 12, "K"))
    assert record["certificate"]["verdict"] == "NUMERICAL_FAIL"


def test_centroid_requires_evaluated_loader(monkeypatch):
    outputs()
    monkeypatch.setattr(md, "load_outputs", lambda root: outputs())

    def reject(root, output):
        raise md.CellError("sentinel evaluated centroid admission")

    monkeypatch.setattr(contract, "load_numerical_check", reject)
    with pytest.raises(md.CellError, match="sentinel evaluated centroid admission"):
        centroids.numerical_record(ROOT)


def test_centroid_consumes_the_supplied_evaluated_bundle(monkeypatch):
    bundle = contract.load_bundle(ROOT)

    def unexpected(*args):
        pytest.fail("evaluated bundle was discarded and inputs were loaded again")

    monkeypatch.setattr(md, "load_outputs", unexpected)
    monkeypatch.setattr(contract, "load_numerical_check", unexpected)
    assert centroids.render_centroids(ROOT, bundle) == (ROOT / centroids.CENTROIDS_RELPATH).read_bytes()
