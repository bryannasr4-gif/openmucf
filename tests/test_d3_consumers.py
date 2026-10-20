"""Primitive discrepancies reach the evaluated consumer and publication boundary."""
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from openmucf.g4 import d3_centroids as centroids
from openmucf.g4 import d3_contract as contract
from openmucf.g4 import d3_numerical as numerical
from openmucf.g4.sources import mudirac130 as md

ROOT = Path(__file__).resolve().parents[1]


def shifted(payload):
    for variant in numerical.VARIANTS:
        payload["A"]["K1"][variant]["B_Ha"] -= 0.1 / numerical.EH
    for basis in payload["B"].values():
        basis["K1"] -= 0.1 / numerical.EH


def copied_root(tmp_path):
    shutil.copytree(ROOT / md.D3_RELDIR, tmp_path / md.D3_RELDIR)
    return tmp_path


def test_all_variants_discrepancy_reaches_consumers_and_build(monkeypatch):
    evidence = deepcopy(numerical.load_evidence(ROOT))
    shifted(evidence.payloads[6, 12, True])
    result = numerical.evaluate_all(ROOT, evidence)
    witness = result[6, 12, True, "K"]
    for estimate in (witness.prereg, witness.a3):
        assert estimate.gates["G1"] is True
        assert estimate.gates["G2"] is True
        assert estimate.gates["G3"] is False
    assert witness.verdict == "NUMERICAL_FAIL"
    assert witness.b_num_ev == pytest.approx(0.1003157294326229)
    assert witness.b_num_ev > witness.target_ev
    original = md.read_rows

    def read(path, columns):
        if Path(path).name == "numerical_check.csv":
            return iter((where, result[int(r["Z"]), int(r["A"]), r["uehling"] == "on",
                                       r["quantity"]].historical_summary())
                        for where, r in original(path, columns))
        return original(path, columns)

    monkeypatch.setattr(md, "read_rows", read)
    monkeypatch.setattr(numerical, "evaluate_all", lambda root: result)
    bundle = contract.load_bundle(ROOT)
    assert bundle.checks[6, 12, True, "K"].verdict == "NUMERICAL_FAIL"
    record = next(r for r in contract.components(ROOT, bundle)
                  if (r.get("Z"), r.get("A"), r.get("quantity")) == (6, 12, "K"))
    assert record["certificate"]["verdict"] == "NUMERICAL_FAIL"
    assert next(r for r in centroids.centroid_rows(ROOT, bundle)
                if (r["Z"], r["A"]) == ("6", "12"))["numerical_verdict"] == "NUMERICAL_FAIL"
    spec = importlib.util.spec_from_file_location(
        "d3_consumer_generator", ROOT / "scripts/generate_g4data.py")
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generator
    spec.loader.exec_module(generator)
    monkeypatch.setattr(contract, "load_bundle", lambda root: bundle)
    with pytest.raises(md.CellError, match="publication refused"):
        generator.build_dataset_artifacts()


def test_changed_primitives_with_updated_digest_reach_real_loader(tmp_path):
    root = copied_root(tmp_path)
    path = root / "data/g4/d3/reference/payloads.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(r for r in rows if (r["z"], r["a"], r["uehling"]) == (6, 12, True))
    shifted(row["payload"])
    path.write_bytes(("\n".join(json.dumps(r) for r in rows) + "\n").encode("ascii"))
    protocol_path = path.with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    protocol["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol_path.write_bytes(json.dumps(protocol).encode("ascii"))
    with pytest.raises(md.CellError, match="(verdict|Q_ref_eV) disagrees"):
        contract.load_numerical_check(root, md.load_outputs(root))


def test_missing_and_duplicate_csv_rows_are_refused(tmp_path):
    root = copied_root(tmp_path)
    path = root / contract.NUMERICAL_CHECK_RELPATH
    original = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(original.decode("ascii"))))
    for changed, error_type, error in [([], md.EmptyError, "no rows"),
                                       (rows[1:], md.CellError, "keys differ"),
                                       (rows + rows[:1], md.DuplicateKeyError, "duplicate")]:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=contract.NUMERICAL_CHECK_COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(changed)
        path.write_bytes(stream.getvalue().encode("ascii"))
        with pytest.raises(error_type, match=error):
            contract.load_numerical_check(root, md.load_outputs(root))


def test_unusable_required_run_is_refused():
    evidence = deepcopy(numerical.load_evidence(ROOT))
    evidence.runs[6, 12, True, "phase2"]["output"]["GE"] = "INCOMPLETE"
    with pytest.raises(numerical.EvidenceError, match="unusable required run"):
        numerical.evaluate_all(ROOT, evidence, members={(6, 12)})


def test_file_loader_requires_primitive_joins_after_digest_update(tmp_path):
    root = copied_root(tmp_path)
    path = root / "data/g4/d3/reference/runs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(r for r in rows if (r["z"], r["a"], r["uehling"], r["role"])
               == (6, 12, True, "phase1"))
    assert row["public_run"] is None
    row["input_text"] = row["input_text"].replace("element: C", "element: N")
    row["input_sha256"] = hashlib.sha256(row["input_text"].encode("ascii")).hexdigest()
    path.write_bytes(("\n".join(json.dumps(r) for r in rows) + "\n").encode("ascii"))
    protocol_path = path.with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    protocol["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol_path.write_bytes(json.dumps(protocol).encode("ascii"))
    with pytest.raises(numerical.EvidenceError, match="input recipe"):
        numerical.load_evidence(root)
