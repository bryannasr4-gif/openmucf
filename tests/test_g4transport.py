"""Transport comparison guards and the committed run manifest."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cpp/transport/transport.py"
SPEC = importlib.util.spec_from_file_location("g4transport_workflow", SOURCE)
assert SPEC is not None and SPEC.loader is not None
transport = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = transport
SPEC.loader.exec_module(transport)
MATRIX = json.loads((ROOT / "cpp/transport/matrix.json").read_text(encoding="utf-8"))
EVIDENCE = ROOT / "cpp/transport/evidence"


def tiny_records() -> list[list[str]]:
    return [["T", "0", "1", "0", "13", "-", "-", "0x0p+0", "0x0p+0"],
            ["E", "0", str(MATRIX["seed_base"]), "1"]]


def tiny_config() -> dict[str, list[str]]:
    return {"PROCESSES": ["muMinusCaptureAtRest"],
            "MATERIAL": ["1", "1", "13", "27", float(1).hex()],
            "PARTICLES": ["ok"], "CONFIG": ["mizuno2025", "mudirac130", "set", "x"]}


def test_t149_parity_and_thread_digest_detect_one_field_corruption():
    records = tiny_records()
    assert transport.records_digest(records) == transport.records_digest(list(reversed(records)))
    changed = [row[:] for row in records]
    changed[0][7] = "0x1p+0"
    assert transport.records_digest(records) != transport.records_digest(changed)


def test_t149_missing_event_and_wrong_seed_fail():
    records = tiny_records()
    assert transport.event_check(records, 1, MATRIX["seed_base"])[0]
    assert not transport.event_check(records[:1], 1, MATRIX["seed_base"])[0]
    changed = [row[:] for row in records]
    changed[1][2] = str(MATRIX["seed_base"] + 1)
    assert not transport.event_check(changed, 1, MATRIX["seed_base"])[0]


def test_t149_duplicate_rest_process_and_two_isotope_material_fail():
    config = tiny_config()
    assert transport.route_check(config, tiny_records(), "bound_decay", 1)[0]
    config["PROCESSES"].append("muMinusCaptureAtRest")
    assert not transport.route_check(config, tiny_records(), "bound_decay", 1)[0]
    config = tiny_config()
    assert transport.target_check(config, {"Z": 13, "A": 27})[0]
    config["MATERIAL"][1] = "2"
    assert not transport.target_check(config, {"Z": 13, "A": 27})[0]


def test_t149_wrong_config_fails():
    config = tiny_config()
    assert transport.config_check(config, "enabled", "x")[0]
    config["CONFIG"][0] = "parity"
    assert not transport.config_check(config, "enabled", "x")[0]


def test_t149_wrong_lookup_origin_fails(tmp_path):
    (tmp_path / "d1_capture.mizuno2025.g4dat").write_text(
        "#COLUMNS Z A value\n13 27 0.7\n", encoding="ascii")
    (tmp_path / "d1_zeff.g4dat").write_text("#COLUMNS Z value\n13 11\n", encoding="ascii")
    (tmp_path / "d3_kshell.mudirac130.g4dat").write_text(
        "#COLUMNS Z A value\n13 27 400\n", encoding="ascii")
    config = {"RATE_BD": [(0.7 * 0.001).hex()], "RATE_HELPER": [(0.7 * 0.001).hex()],
              "ZEFF_BD": [float(11).hex()], "ZEFF_HELPER": [float(11).hex()],
              "KA": [(400 * 0.001).hex()],
              "_RES_LINES": [["rate", "13", "27", "exact", "mizuno2025", "1"],
                             ["zeff", "13", "0", "compiled", "mizuno2025", "0"],
                             ["kshell", "13", "27", "exact", "mudirac130", "1"]]}
    pristine = {"ZEFF_BD": [float(11).hex()], "ZEFF_HELPER": [float(11).hex()],
                "K1": [(400 * 0.001).hex()]}
    assert transport.lookup_check(config, pristine, {"Z": 13, "A": 27}, tmp_path)[0]
    config["_RES_LINES"][0][3] = "compiled"
    assert not transport.lookup_check(config, pristine, {"Z": 13, "A": 27}, tmp_path)[0]


def cascade_fixture(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    columns = " ".join(f"e{i}" for i in range(2, 9))
    values = " ".join(str(i * 1000) for i in range(13, 6, -1))
    (dataset / "d3_levels.mudirac130.g4dat").write_text(
        f"#COLUMNS Z A {columns}\n13 27 {values}\n", encoding="ascii")
    levels = [float(i) for i in range(14, 0, -1)]
    config = {"L": [value.hex() for value in levels]}
    records = [["C", "0", "1", "0", "11", "model_EMCascade", levels[13].hex(), "0x0p+0"]]
    for k in range(1, 14):
        records.append(["C", "0", "1", str(k), "11", "model_EMCascade", float(1).hex(), "0x0p+0"])
    return dataset, config, records


def test_t149_cascade_transition_bound_and_leftover_step(tmp_path):
    dataset, config, records = cascade_fixture(tmp_path)
    target = {"Z": 13, "A": 27}
    assert transport.d9_check(records, config, "bound_decay", target, dataset)[0]
    bound = 64 * sys.float_info.epsilon * 14
    exact = [row[:] for row in records]
    exact[1][6] = (1.0 + bound).hex()
    assert transport.d9_check(exact, config, "bound_decay", target, dataset)[0]
    wrong = [row[:] for row in records]
    wrong[1][6] = (1.0 + 2 * bound).hex()
    assert not transport.d9_check(wrong, config, "bound_decay", target, dataset)[0]
    leftover = [row[:] for row in records]
    leftover.append(["C", "0", "1", "14", "11", "model_EMCascade", float(1).hex(), "0x0p+0"])
    assert not transport.d9_check(leftover, config, "bound_decay", target, dataset)[0]


def test_t149_manifest_source_digest_guard():
    expected = {name: hashlib.sha256((ROOT / "cpp/transport" / name).read_bytes()).hexdigest()
                for name in transport.SOURCES}
    changed = dict(expected)
    changed["transport.py"] = "0" * len(changed["transport.py"])
    assert expected != changed
    if not (EVIDENCE / "manifest.json").exists():
        pytest.skip("manifest is written after transport runs")
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"] == expected


def test_t149_stage_refuses_wrong_archive_md5(monkeypatch, tmp_path):
    fake = SimpleNamespace(DATASET_NAME="G4MuonicData", DATASET_VERSION="bad",
                           emit=SimpleNamespace(archive_name=lambda name, version: "bad.tar.gz"),
                           build_dataset_artifacts=lambda: ({}, b"not the declared archive"))
    monkeypatch.setattr(transport, "generator", lambda: fake)
    with pytest.raises(RuntimeError, match="archive md5"):
        transport.stage_dataset(tmp_path / "stage")


def test_t149_nonempty_target_is_refused(tmp_path):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "keep").write_text("data", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty"):
        transport.fresh(target)


def test_t149_committed_manifest_ties_builds_dataset_and_cells():
    if not (EVIDENCE / "manifest.json").exists():
        pytest.skip("manifest is written after transport runs")
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    raw = (EVIDENCE / "manifest.json").read_text(encoding="utf-8")
    assert not any(token in raw.lower() for token in ("/root/", "/mnt/", "/home/", "u020", "bryan"))
    for build in manifest["builds"]:
        assert build["commit"] == MATRIX["revisions"][build["tag"]]
        assert build["compile_count"] == build["flag_count"]
        for patch in build["patches"]:
            assert patch["sha256"] == hashlib.sha256((ROOT / patch["path"]).read_bytes()).hexdigest()
        assert all(row["sha256"] for row in build["ldd"] if row["path"].startswith(("$WORK", "$PRESERVED")))
    generator_path = ROOT / "scripts/generate_g4data.py"
    spec = importlib.util.spec_from_file_location("transport_generator_test", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generator
    spec.loader.exec_module(generator)
    _, archive = generator.build_dataset_artifacts()
    assert hashlib.md5(archive).hexdigest() == manifest["dataset"]["md5"]
    snippet = (ROOT / "data/g4/d1/geant4_add_dataset.snippet").read_text(encoding="utf-8")
    assert manifest["dataset"]["md5"] in snippet
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = {member.name: hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                   for member in tar.getmembers() if member.isfile()}
    assert members == manifest["dataset"]["members"]
    with (EVIDENCE / "cells.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    expected_rows = set()
    for tag in MATRIX["revisions"]:
        for route in MATRIX["routes"]:
            for target in MATRIX["targets"]:
                for mode in MATRIX["modes"]:
                    for thread in ([1] if mode == "preserved" else MATRIX["threads"]):
                        for check in transport.CHECK_IDS:
                            expected_rows.add((tag, mode, route, str(target["Z"]), str(target["A"]),
                                               str(thread), check))
        for case in MATRIX["cases"]:
            if case != "P13" or tag == "v11.4.2":
                expected_rows.add((tag, "case", "bound_decay", "", "", "", case))
    actual_rows = {(row["tag"], row["mode"], row["route"], row["Z"], row["A"],
                    row["threads"], row["check"]) for row in rows}
    assert actual_rows == expected_rows
    assert len(rows) == len(actual_rows)
    assert all(row["status"] in ("PASS", "INFO") for row in rows)
