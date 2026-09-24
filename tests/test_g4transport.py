"""Transport comparison guards and the committed run manifest."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
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


def synthetic_check_work(tmp_path, monkeypatch, changes=None):
    changes = changes or {}
    tag = next(iter(MATRIX["revisions"]))
    route = MATRIX["routes"][0]
    target = MATRIX["targets"][0]
    minimal = dict(MATRIX)
    minimal.update(revisions={tag: MATRIX["revisions"][tag]}, routes=[route], targets=[target],
                   threads=[1, 4], events=1, cases={})
    monkeypatch.setattr(transport, "MATRIX", minimal)
    for name in ("route_check", "lookup_check", "level_check", "d9_check"):
        monkeypatch.setattr(transport, name, lambda *args: (True, "synthetic"))
    stage = tmp_path / "dataset"
    stage.mkdir()
    transport.save(stage / "stage.json", {"name": "G4MuonicData", "version": "synthetic"})
    (stage / "G4MuonicDatasynthetic").mkdir()
    transport.save(tmp_path / "cases.json", [])
    for mode in minimal["modes"]:
        for thread in ([1] if mode == "preserved" else minimal["threads"]):
            directory = transport.cell_dir(tmp_path, tag, mode, route, target, thread)
            output = directory / "attempt_1"
            output.mkdir(parents=True)
            transport.save(directory / "complete.json", {"output": "attempt_1"})
            record = tiny_records()
            if changes.get((mode, thread, "record")):
                record[0][7] = float(1).hex()
            (output / "records-0.txt").write_text(
                "\n".join(" ".join(row) for row in record) + "\n", encoding="utf-8")
            lines = ["PROCESSES muMinusCaptureAtRest",
                     f"MATERIAL 1 1 {target['Z']} {target['A']} {float(1).hex()}",
                     f"PARTICLES {changes.get((mode, thread, 'particles'), 'ok')}"]
            values = {"patched-off": ["compiled", "compiled", "none", "none"],
                      "patched-default": ["parity", "compiled", "set", "synthetic"],
                      "enabled": ["mizuno2025", "mudirac130", "set", "synthetic"]}.get(mode)
            if changes.get((mode, thread, "config")):
                values = None
            if values:
                lines.append("CONFIG " + " ".join(values))
            (output / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tag, route, target


def test_t149_check_wiring_rejects_gating_changes_and_reports_preserved(tmp_path, monkeypatch):
    tag, route, target = synthetic_check_work(
        tmp_path, monkeypatch, {("preserved", 1, "record"): True})
    out = tmp_path / "cells.csv"
    transport.check(tmp_path, out)
    with out.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    preserved = [row for row in rows if row["mode"] == "preserved" and row["check"] == "parity"]
    assert len(preserved) == 1 and preserved[0]["status"] == "INFO"
    assert preserved[0]["tag"] == tag and preserved[0]["route"] == route
    assert preserved[0]["Z"] == str(target["Z"])


@pytest.mark.parametrize("change", [
    ("patched-off", 1, "record"), ("patched-default", 1, "record"),
    "threads", ("patched-default", 1, "config"),
    ("pristine", 1, "particles")])
def test_t149_check_wiring_rejects_corruption(tmp_path, monkeypatch, change):
    changes = ({(mode, 4, "record"): True for mode in MATRIX["modes"] if mode != "preserved"}
               if change == "threads" else {change: True})
    synthetic_check_work(tmp_path, monkeypatch, changes)
    with pytest.raises(RuntimeError, match="first gating failure"):
        transport.check(tmp_path, tmp_path / "cells.csv")


def test_t149_parity_and_thread_digest_detect_one_field_corruption():
    records = tiny_records()
    assert transport.records_digest(records) == transport.records_digest(list(reversed(records)))
    changed = [row[:] for row in records]
    changed[0][7] = "0x1p+0"
    assert transport.records_digest(records) != transport.records_digest(changed)


def test_t149_same_writer_hex_spelling_remains_byte_distinct():
    records = tiny_records()
    records[0][7] = float(1).hex()
    respelled = [row[:] for row in records]
    respelled[0][7] = "0x1p+0"
    assert transport.records_digest(records) != transport.records_digest(respelled)


def test_t149_target_numeric_fields_compare_exact_values():
    config = tiny_config()
    target = {"Z": 13, "A": 27}
    config["MATERIAL"][-1] = "0x1p+0"
    assert transport.target_check(config, target)[0]
    config["MATERIAL"][0] = "01"
    assert transport.target_check(config, target)[0]
    config["MATERIAL"][-1] = math.nextafter(1.0, 2.0).hex()
    assert not transport.target_check(config, target)[0]
    config["MATERIAL"] = ["1", "1", "13", "27"]
    assert not transport.target_check(config, target)[0]
    config["MATERIAL"] = ["1", "1", "13", "27", "invalid"]
    assert not transport.target_check(config, target)[0]


def test_t149_same_writer_level_spelling_stays_distinct(tmp_path):
    pristine = {"L": [float(1).hex()]}
    config = {"L": ["0x1p+0"]}
    target = {"Z": 13, "A": 27}
    assert not transport.level_check(config, pristine, "patched-off", target, tmp_path)[0]
    (tmp_path / "d3_kshell.mudirac130.g4dat").write_text(
        "#COLUMNS Z A value\n82 208 1\n", encoding="ascii")
    (tmp_path / "d3_levels.mudirac130.g4dat").write_text(
        "#COLUMNS Z A e2\n82 208 1\n", encoding="ascii")
    assert not transport.level_check(config, pristine, "enabled", target, tmp_path)[0]


def test_t149_level_table_excludes_uncertainty_columns(tmp_path):
    source = tmp_path / "levels.g4dat"
    source.write_text("#COLUMNS Z A e2 e3 u2 u3\n13 27 100 50 0.1 0.2\n", encoding="ascii")
    assert transport.table_levels(source, (13, 27)) == [100.0, 50.0]
    for header in ("Z A u2 u3", "Z A e2 u2 e3", "Z A e2 e4 u2"):
        source.write_text(f"#COLUMNS {header}\n13 27 100 50 0.1\n", encoding="ascii")
        with pytest.raises(RuntimeError, match="bad level shape"):
            transport.table_levels(source, (13, 27))


def test_t149_same_writer_lookup_fallback_spelling_stays_distinct(tmp_path):
    (tmp_path / "d1_capture.mizuno2025.g4dat").write_text(
        "#COLUMNS Z A value\n82 208 1\n", encoding="ascii")
    (tmp_path / "d1_zeff.g4dat").write_text(
        "#COLUMNS Z value\n82 1\n", encoding="ascii")
    (tmp_path / "d3_kshell.mudirac130.g4dat").write_text(
        "#COLUMNS Z A value\n82 208 1\n", encoding="ascii")
    pristine = {key: [float(1).hex()] for key in
                ("RATE_BD", "RATE_HELPER", "ZEFF_BD", "ZEFF_HELPER", "K1")}
    config = {key: [float(1).hex()] for key in
              ("RATE_BD", "RATE_HELPER", "ZEFF_BD", "ZEFF_HELPER", "KA")}
    config["RATE_BD"] = ["0x1p+0"]
    config["_RES_LINES"] = [
        ["rate", "13", "27", "compiled", "mizuno2025", "0"],
        ["zeff", "13", "0", "compiled", "mizuno2025", "0"],
        ["kshell", "13", "27", "compiled", "mudirac130", "0"],
    ]
    assert not transport.lookup_check(config, pristine, {"Z": 13, "A": 27}, tmp_path)[0]
    config["RATE_BD"] = [float(1).hex()]
    config["RATE_HELPER"] = ["0x1p+0"]
    assert not transport.lookup_check(config, pristine, {"Z": 13, "A": 27}, tmp_path)[0]


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


def test_t149_helper_requires_master_precreation_marker():
    config = tiny_config()
    config["PROCESSES"] = ["muMinusAtomicCaptureAtRest"]
    config["MUATOM_PRECREATED"] = ["MuAl27"]
    atom = str(2_000_000_000 + int(config["MATERIAL"][2]) * 10000
               + int(config["MATERIAL"][3]) * 10)
    records = [["T", "0", "1", "0", atom, "muMinusAtomicCaptureAtRest", "-", "0x0p+0", "0x0p+0"]]
    assert transport.route_check(config, records, "muonic_atom_helper", 1)[0]
    del config["MUATOM_PRECREATED"]
    assert not transport.route_check(config, records, "muonic_atom_helper", 1)[0]


def test_t149_helper_counts_atom_pdg_amid_cascade_tracks():
    config = tiny_config()
    config["PROCESSES"] = ["muMinusAtomicCaptureAtRest"]
    config["MUATOM_PRECREATED"] = ["MuAl27"]
    atom = str(2_000_000_000 + int(config["MATERIAL"][2]) * 10000
               + int(config["MATERIAL"][3]) * 10)
    creator = "muMinusAtomicCaptureAtRest"
    def track(ident: int, pdg: str) -> list[str]:
        return ["T", "0", str(ident), "1", pdg, creator, "-", "0x0p+0", "0x0p+0"]
    records = [track(1, atom), track(2, "11"), track(3, "22")]
    assert transport.route_check(config, records, "muonic_atom_helper", 1)[0]
    assert not transport.route_check(config, records + [track(4, atom)], "muonic_atom_helper", 1)[0]
    assert not transport.route_check(config, [track(2, "22")], "muonic_atom_helper", 1)[0]
    wrong_creator = [track(2, "22")]
    wrong_creator.append(["T", "0", "1", "0", atom, "-", "-", "0x0p+0", "0x0p+0"])
    assert not transport.route_check(config, wrong_creator, "muonic_atom_helper", 1)[0]
    config["MUATOM_PRECREATED"] = ["MuAl28"]
    assert not transport.route_check(config, [track(1, atom)], "muonic_atom_helper", 1)[0]
    config["MUATOM_PRECREATED"] = ["MuSi27"]
    assert not transport.route_check(config, [track(1, atom)], "muonic_atom_helper", 1)[0]
    config["MUATOM_PRECREATED"] = ["MuAl27"]
    config["MATERIAL"] = ["1", "1", "13"]
    assert not transport.route_check(config, [track(1, atom)], "muonic_atom_helper", 1)[0]
    config["MATERIAL"] = ["1", "1", "bad", "27", float(1).hex()]
    assert not transport.route_check(config, [track(1, atom)], "muonic_atom_helper", 1)[0]
    config["MATERIAL"] = tiny_config()["MATERIAL"]
    assert not transport.route_check(config, [track(1, "bad")], "muonic_atom_helper", 1)[0]
    config["MATERIAL"] = ["1", "1", "99", "99", float(1).hex()]
    config["MUATOM_PRECREATED"] = ["MuNone99"]
    unknown_atom = str(2_000_000_000 + 99 * 10000 + 99 * 10)
    assert not transport.route_check(config, [track(1, unknown_atom)], "muonic_atom_helper", 1)[0]


@pytest.mark.parametrize(
    "target,symbol", tuple(zip(MATRIX["targets"], ("Al", "Si", "Ag", "Pb"), strict=True)))
def test_t149_helper_marker_matches_target_ion_name(target, symbol):
    config = tiny_config()
    config["PROCESSES"] = ["muMinusAtomicCaptureAtRest"]
    config["MATERIAL"][2:4] = [str(target["Z"]), str(target["A"])]
    config["MUATOM_PRECREATED"] = [f"Mu{symbol}{target['A']}"]
    atom = str(2_000_000_000 + target["Z"] * 10000 + target["A"] * 10)
    records = [["T", "0", "1", "0", atom, "muMinusAtomicCaptureAtRest", "-", "0x0p+0", "0x0p+0"]]
    assert transport.route_check(config, records, "muonic_atom_helper", 1)[0]


def test_t149_marker_symbols_cover_matrix_targets(monkeypatch):
    config = tiny_config()
    config["PROCESSES"] = ["muMinusAtomicCaptureAtRest"]
    config["MATERIAL"][2:4] = ["99", "99"]
    config["MUATOM_PRECREATED"] = ["MuNone99"]
    monkeypatch.setitem(transport.MATRIX, "targets", [*MATRIX["targets"], {"Z": 99, "A": 99}])
    with pytest.raises(ValueError, match="zip"):
        transport.route_check(config, [], "muonic_atom_helper", 1)


def test_t149_bound_route_rejects_atom_and_atomic_creator():
    config = tiny_config()
    atom = str(2_000_000_000 + int(config["MATERIAL"][2]) * 10000
               + int(config["MATERIAL"][3]) * 10)
    def track(pdg: str, creator: str) -> list[str]:
        return ["T", "0", "1", "0", pdg, creator, "-", "0x0p+0", "0x0p+0"]
    assert transport.route_check(config, [track("13", "-")], "bound_decay", 1)[0]
    assert not transport.route_check(config, [track("22", "muMinusAtomicCaptureAtRest")],
                                     "bound_decay", 1)[0]
    assert not transport.route_check(config, [track(atom, "-")], "bound_decay", 1)[0]


def test_t149_master_precreation_call_precedes_beamon():
    source = (ROOT / "cpp/transport/g4muonic_transport.cc").read_text(encoding="utf-8")
    assert source.index("manager->Initialize();") < source.index(
        "G4IonTable::GetIonTable()->GetMuonicAtom(options.z, options.a);") < source.index(
        "manager->BeamOn(options.events);")
    assert '"MUATOM_PRECREATED "' in source


def test_t149_new_harness_uses_fresh_run_tree(tmp_path):
    target = {"Z": 13, "A": 27}
    old = tmp_path / "v11.4.2/runs/pristine/muonic_atom_helper/13-27/1"
    old.mkdir(parents=True)
    (old / "complete.json").write_text("old harness", encoding="utf-8")
    chosen = transport.cell_dir(tmp_path, "v11.4.2", "pristine", "muonic_atom_helper", target, 1)
    assert chosen == tmp_path / "v11.4.2/runs_precreated/pristine/muonic_atom_helper/13-27/1"
    assert not (chosen / "complete.json").exists()


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
    config = {"RATE_BD": [(0.7 / 1000.0).hex()], "RATE_HELPER": [(0.7 / 1000.0).hex()],
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


def test_t149_compiled_helper_uses_its_own_pristine_kernel(tmp_path):
    (tmp_path / "d1_capture.mizuno2025.g4dat").write_text(
        "#COLUMNS Z A value\n13 27 0.7\n", encoding="ascii")
    (tmp_path / "d1_zeff.g4dat").write_text("#COLUMNS Z value\n13 11\n", encoding="ascii")
    (tmp_path / "d3_kshell.mudirac130.g4dat").write_text(
        "#COLUMNS Z A value\n13 27 400\n", encoding="ascii")
    config = {"RATE_BD": [(0.7 / 1000.0).hex()], "RATE_HELPER": [(0.7 / 1000.0).hex()],
              "ZEFF_BD": [float(11).hex()], "ZEFF_HELPER": [float(12).hex()],
              "KA": [(400 * 0.001).hex()],
              "_RES_LINES": [["rate", "13", "27", "exact", "mizuno2025", "1"],
                             ["zeff", "13", "0", "compiled", "mizuno2025", "0"],
                             ["kshell", "13", "27", "exact", "mudirac130", "1"]]}
    pristine = {"ZEFF_BD": [float(11).hex()], "ZEFF_HELPER": [float(12).hex()]}
    assert transport.lookup_check(config, pristine, {"Z": 13, "A": 27}, tmp_path)[0]


def test_t149_version_matches_numeric_release():
    assert transport.version_matches("v11.4.2", "11.4.2")
    assert transport.version_matches("v11.5.0.beta", "11.5.0")
    assert transport.version_matches("v11.5.0.beta.1", "11.5.0")
    assert not transport.version_matches("v11.5.1.beta", "11.5.0")
    assert not transport.version_matches("v11.5.0.beta", "11.5.0.beta")
    assert not transport.version_matches("v11.5.0.rc1", "11.5.0")


def test_t149_build_location_uses_successful_retry(tmp_path):
    tag = "v11.5.0.beta"
    assert transport.build_dir(tmp_path, tag, "pristine") == tmp_path / tag / "pristine"
    transport.record_build_dir(tmp_path, tag, "pristine", tmp_path / tag / "pristine_retry")
    assert transport.build_dir(tmp_path, tag, "pristine") == tmp_path / tag / "pristine_retry"


def test_t149_corrupt_levels_uses_last_tabulated_column(tmp_path):
    tag = "v11.4.2"
    stage = tmp_path / "dataset"
    data = stage / "G4MuonicDatasynthetic"
    data.mkdir(parents=True)
    transport.save(stage / "stage.json", {"name": "G4MuonicData", "version": "synthetic"})
    levels = data / "d3_levels.mudirac130.g4dat"
    levels.write_text("#COLUMNS Z A e2 e3 e4 e5 e6 e7 e8\n"
                      "82 0 700 600 500 400 350 300 250\n"
                      "82 208 700 600 500 400 350 300 250\n", encoding="ascii")
    (tmp_path / tag / "farm_nodata").mkdir(parents=True)
    cell = tmp_path / tag / transport.RUNS_DIR / "pristine/bound_decay/82-208/1"
    output = cell / "attempt_1"
    output.mkdir(parents=True)
    transport.save(cell / "complete.json", {"output": "attempt_1"})
    (output / "config.txt").write_text("L " + " ".join([1.0.hex()] * 8 + [0.4.hex()]),
                                       encoding="utf-8")
    corrupted = transport.corrupt_levels(tag, tmp_path)
    changed = (corrupted / data.name / levels.name).read_text(encoding="ascii")
    assert changed.count(" 200.0\n") == 2
    invalid = tmp_path / "invalid"
    shutil.copytree(stage, invalid / "dataset")
    (invalid / tag / "farm_nodata").mkdir(parents=True)
    bad_cell = invalid / tag / transport.RUNS_DIR / "pristine/bound_decay/82-208/1"
    shutil.copytree(cell, bad_cell)
    (bad_cell / "attempt_1/config.txt").write_text(
        "L " + " ".join([1.0.hex()] * 9), encoding="utf-8")
    with pytest.raises(RuntimeError, match="P12 precondition"):
        transport.corrupt_levels(tag, invalid)


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


def test_t149_rate_division_and_kshell_scale(tmp_path):
    source = ROOT / "data/g4/d1/d1_capture.mizuno2025.g4dat"
    value = transport.table_value(source, (14, 0), "value")
    assert value is not None and value / 1000.0 != value * 0.001
    (tmp_path / source.name).write_text(
        f"#COLUMNS Z A value\n14 0 {value!r}\n", encoding="ascii")
    (tmp_path / "d1_zeff.g4dat").write_text(
        "#COLUMNS Z value\n82 11\n", encoding="ascii")
    (tmp_path / "d3_kshell.mudirac130.g4dat").write_text(
        f"#COLUMNS Z A value\n14 0 {value!r}\n", encoding="ascii")
    config = {"RATE_BD": [(value / 1000.0).hex()],
              "RATE_HELPER": [(value / 1000.0).hex()],
              "ZEFF_BD": [float(11).hex()], "ZEFF_HELPER": [float(11).hex()],
              "KA": [(value * 0.001).hex()],
              "_RES_LINES": [["rate", "14", "0", "exact", "mizuno2025", "1"],
                             ["zeff", "14", "0", "compiled", "mizuno2025", "0"],
                             ["kshell", "14", "0", "exact", "mudirac130", "1"]]}
    pristine = {"ZEFF_BD": [float(11).hex()], "ZEFF_HELPER": [float(11).hex()]}
    target = {"Z": 14, "A": 0}
    assert transport.lookup_check(config, pristine, target, tmp_path)[0]
    config["RATE_BD"] = [(value * 0.001).hex()]
    assert not transport.lookup_check(config, pristine, target, tmp_path)[0]
    config["RATE_BD"] = [(value / 1000.0).hex()]
    config["RATE_HELPER"] = [(value * 0.001).hex()]
    assert not transport.lookup_check(config, pristine, target, tmp_path)[0]


def test_t149_manifest_resolves_libraries_from_build_install(monkeypatch, tmp_path):
    install = tmp_path / "install"
    lib = install / "lib/libprobe.so"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"library")
    binary = tmp_path / "binary"
    binary.write_bytes(b"executable")

    def ldd(argv, **kwargs):
        assert argv == ["ldd", str(binary)]
        assert kwargs["env"]["LD_LIBRARY_PATH"] == str(install / "lib")
        return f"libprobe.so => {lib} (0x000)\n"

    monkeypatch.setattr(transport, "command", ldd)
    found = transport.libraries(binary, tmp_path, {}, install)
    assert found == [{"soname": "libprobe.so", "path": "$WORK" + str(lib)[len(str(tmp_path)):],
                      "sha256": hashlib.sha256(b"library").hexdigest()}]


def test_t149_manifest_records_preserved_harness_and_libraries(monkeypatch, tmp_path):
    stage = tmp_path / "dataset"
    (stage / "G4MuonicDatasynthetic").mkdir(parents=True)
    transport.save(stage / "stage.json", {"name": "G4MuonicData", "version": "synthetic",
                                           "members": {}, "archive": "synthetic.tar.gz", "md5": "probe"})
    for tag in MATRIX["revisions"]:
        preserved = tmp_path / tag / "preserved"
        binary = preserved / "harness/executable"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(tag.encode())
        install = preserved / "install"
        (install / "lib").mkdir(parents=True)
        transport.save(preserved / "preserved.json",
                       {"binary": str(binary), "install": str(install)})
        transport.save(tmp_path / tag / "farm.json", {"entries": []})
        kinds = ("pristine", "patched", "mutant") if tag == "v11.4.2" else ("pristine", "patched")
        for kind in kinds:
            build = transport.build_dir(tmp_path, tag, kind)
            build.mkdir(parents=True)
            transport.save(build / "build.json", {
                "binary": str(binary), "install": str(install), "commit": MATRIX["revisions"][tag],
                "patches": [], "removed": False, "configure_argv": [], "cache_sha256": "probe",
                "compile_commands_sha256": "probe", "compile_count": 1, "flag_count": 1,
                "version": "synthetic", "wall_s": 0})

    def ldd(binary, work, preserved_paths, install):
        assert install == preserved_paths[next(tag for tag in MATRIX["revisions"]
                                               if tag in str(binary))]
        return [{"soname": "libprobe.so", "path": "$WORK/libprobe.so", "sha256": "probe"}]

    monkeypatch.setattr(transport, "libraries", ldd)
    monkeypatch.setattr(transport, "command", lambda *args, **kwargs: "synthetic version\n")
    output = tmp_path / "manifest.json"
    transport.manifest(tmp_path, output)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert set(manifest["preserved"]) == set(MATRIX["revisions"])
    for tag, entry in manifest["preserved"].items():
        assert entry["harness_path"].replace("\\", "/").startswith(f"$WORK/{tag}/preserved/")
        assert entry["harness_sha256"] == hashlib.sha256(tag.encode()).hexdigest()
        assert entry["ldd"] == [{"soname": "libprobe.so", "path": "$WORK/libprobe.so",
                                 "sha256": "probe"}]


def test_t149_manifest_normalizes_embedded_work_path(tmp_path):
    install = tmp_path / "tag/pristine/install"
    arg = f"-DCMAKE_INSTALL_PREFIX={install}"
    expected = "-DCMAKE_INSTALL_PREFIX=$WORK" + str(install)[len(str(tmp_path)):]
    assert transport.normalized(arg, tmp_path, {}) == expected


def test_t149_manifest_farm_target_is_inside_work():
    entry = {"name": "physics_data", "target": "/root/external/data", "sha256": "digest"}
    assert transport.farm_manifest_entry("tag", entry) == {
        "name": "physics_data", "target": "$WORK/tag/farm/physics_data", "sha256": "digest"}


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
    assert set(manifest["preserved"]) == set(MATRIX["revisions"])
    for item in manifest["preserved"].values():
        assert item["harness_path"].startswith("$WORK/")
        assert item["harness_sha256"]
        assert item["ldd"]
        assert all(row["sha256"] for row in item["ldd"]
                   if row["path"].startswith(("$WORK", "$PRESERVED")))
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
