"""Synthetic selector and capture-corpus checks."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import pytest

from openmucf.g4 import d2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/g4/d2"
REFERENCE = ROOT / "data/g4/reference/d2"


def _csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _reference(name: str) -> list[dict[str, str]]:
    with (REFERENCE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _element(z: int, density: float = 1.0, abundances: tuple[float, ...] = (1.0,)) -> d2.Element:
    return d2.Element(z, density, tuple(2 * z + i for i in range(len(abundances))), abundances)


def test_d2_selector_boundary_and_isotope_boundary() -> None:
    elements = (_element(1), _element(1))
    assert d2.selector_index(elements, 0.5) == 0
    assert d2.selector_counts(elements, 8) == (4, 4)
    assert d2.isotope_at_half(_element(8, abundances=(0.5, 0.5))) == 16


def test_d2_selector_weight_factors_and_bisection() -> None:
    for z, expected in ((9, 0.66), (17, 0.66), (35, 0.66), (53, 0.66),
                        (85, 0.66), (8, 0.56), (6, 1.0)):
        elements = (_element(z), _element(1))
        assert d2.factor(z) == expected
        brute = tuple(sum(d2.selector_index(elements, (k + 0.5) / 32) == i for k in range(32))
                      for i in range(2))
        assert d2.selector_counts(elements, 32) == brute
    assert d2.selector_index((_element(9),), 0.75) == 0


def test_d2_ratios_and_binary_conversion() -> None:
    assert d2.a_g4(9, 8) == Fraction(0.66 * 9) / Fraction(0.56 * 8)
    assert d2.h_share("H2O") == Fraction(2) / (Fraction(2) + Fraction(0.56 * 8))
    assert d2.atomic_ratio_to_probability(2, 1, 1) == (2 / 3, 1 / 3)
    for args in ((0, 1, 1), (1, 0, 1), (1, 1, -1), (1, 1, float("nan"))):
        with pytest.raises(ValueError):
            d2.atomic_ratio_to_probability(*args)
    a, b = d2.identifiability_counterexample()
    assert a["P_Z"] != b["P_Z"]
    assert (a["P_Z"] * a["T_Z"], a["P_O"] * a["T_O"]) == (
        b["P_Z"] * b["T_Z"], b["P_O"] * b["T_O"])


def _row() -> dict[str, str]:
    row = dict.fromkeys(d2.MEASUREMENT_COLUMNS, "")
    row.update({"source_id": "a", "material_formula": "SiO2", "phase": "solid",
                "observable": "atomic_capture_ratio", "inferable_parameter": "A(Si/O)",
                "reported_value": "1.0", "reported_uncertainty": "0.1",
                "qualification": "stage=terminal@a p.1;population=all_stops@a p.1;method=xray@a p.1;"
                                 "lineage=own@a p.1;inputs=self:a:cal@a p.1"})
    row.update({field: "not applied: stated a p.1" for field in d2.CORRECTION_COLUMNS})
    return row


def _source(source_id: str = "a") -> dict[str, str]:
    return {"source_id": source_id, "access": "AVAILABLE", "primary_read": "true",
            "kind": "measurement", "same_data_as": ""}


@pytest.mark.parametrize(("change", "reason"), [
    ({"observable": "xray_yield"}, "not a per-atom ratio"),
    ({"inferable_parameter": "A(Si)"}, "not a per-atom ratio"),
    ({"qualification": "stage=initial@a p.1;population=all_stops@a p.1;method=xray@a p.1"}, "stage initial"),
    ({"qualification": "stage=terminal@a p.1;population=transfer_only@a p.1;method=xray@a p.1"},
     "population transfer_only"),
    ({"qualification": "stage=terminal@a p.1;population=all_stops@a p.1;method=unstated@a p.1"},
     "method undocumented"),
    ({"efficiency_correction": ""}, "correction undocumented: efficiency_correction"),
    ({"attenuation_correction": ""}, "correction undocumented: attenuation_correction"),
    ({"cascade_correction": ""}, "correction undocumented: cascade_correction"),
    ({"transfer_assumption": ""}, "correction undocumented: transfer_assumption"),
    ({"reported_uncertainty": ""}, "no uncertainty"),
    ({"material_formula": "?"}, "unidentifiable class"),
])
def test_d2_each_gating_reason(change: dict[str, str], reason: str) -> None:
    row = _row()
    row.update(change)
    assert d2.gate_reason(row, _source()) == reason


def test_d2_source_gate_and_hydrogen_limit() -> None:
    row = _row()
    assert d2.gate_reason(row, _source()) == ""
    joint = dict(row, qualification=row["qualification"].replace("stage=terminal", "stage=initial=terminal"))
    assert d2.gate_reason(joint, _source()) == ""
    for field, value in (("access", "REQUEST_NEEDED"), ("primary_read", "false"), ("kind", "review")):
        source = _source()
        source[field] = value
        assert d2.gate_reason(row, source) == "access"
    row.update({"material_formula": "H2O", "inferable_parameter": "P(H)",
                "reported_value": "<0.5", "reported_uncertainty": ""})
    assert d2.gate_reason(row, _source()) == ""
    assert d2.compare_ratio(Fraction(1, 2), "<0.5", "") == "not_excluded"
    assert d2.compare_ratio(Fraction(3, 4), "<0.5", "") == "outside"
    row["material_formula"] = "SiO2"
    assert d2.gate_reason(row, _source()) == "not a per-atom ratio"


def test_d2_three_sigma_inclusive_and_asymmetric_side() -> None:
    assert d2.compare_ratio(Fraction(13, 10), "1.0", "0.1") == "inside"
    assert d2.compare_ratio(Fraction(131, 100), "1.0", "0.1") == "outside"
    assert d2.compare_ratio(Fraction(13, 10), "1.0", "+0.1-0.01") == "inside"
    assert d2.compare_ratio(Fraction(7, 10), "1.0", "+0.01-0.1") == "inside"


def test_d2_class_priority_and_subclasses() -> None:
    row = _row()
    for formula, phase, expected, subs in (
        ("SiO2", "solid", "d2-selector-oxides", {"d2-selector-oxides"}),
        ("LiF", "solid", "d2-selector-fluorides", {"d2-selector-fluorides"}),
        ("NaCl", "solid", "d2-selector-chlorides", {"d2-selector-chlorides"}),
        ("NaBr", "solid", "d2-selector-other-compounds", {"bromide"}),
        ("NaI", "solid", "d2-selector-other-compounds", {"iodide"}),
        ("FeS", "solid", "d2-selector-other-compounds", {"sulfide"}),
        ("BN", "solid", "d2-selector-other-compounds", {"nitride", "boride"}),
        ("Na2SO4", "solid", "d2-selector-other-compounds", {"ternary"}),
        ("SiO2", "gas", "d2-selector-gases", {"d2-selector-gases"}),
        ("alloy(Cu,Zn)", "solid", "d2-selector-alloys", {"alloy"}),
        ("intermetallic(Cu,Zn)", "solid", "d2-selector-alloys", {"intermetallic"}),
        ("NaCl+SiO2", "solid", "d2-selector-powder-mixtures", {"d2-selector-powder-mixtures"}),
    ):
        row.update({"material_formula": formula, "phase": phase})
        found, found_subs = d2.classify(row)
        assert found == expected
        assert found_subs == subs
    row.update({"material_formula": "H2O", "phase": "solid", "inferable_parameter": "P(H)"})
    assert d2.classify(row) == ("d2-selector-hydrogenous", frozenset(("d2-selector-hydrogenous",)))


def test_d2_every_other_condensed_compound_is_an_other_compound() -> None:
    row = _row()
    for formula in ("SiC", "GaAs", "H2O"):
        row["material_formula"] = formula
        assert d2.classify(row) == ("d2-selector-other-compounds", frozenset())
    row["inferable_parameter"] = "A(Si/C)"
    row["material_formula"] = "SiC"
    assert d2.gate_reason(row, _source()) == ""
    for formula in ("Fe", "solution(Cu,Au)"):
        row["material_formula"] = formula
        with pytest.raises(ValueError, match="unidentifiable class|unsupported formula"):
            d2.classify(row)
        assert d2.gate_reason(row, _source()) == "unidentifiable class"


def test_d2_independence_requires_lineage_and_disjoint_inputs() -> None:
    a, b = _row(), _row()
    b.update({"source_id": "b", "qualification": a["qualification"].replace("a p.1", "b p.1")
              .replace("self:a:cal", "self:b:cal")})
    sources = {"a": _source("a"), "b": _source("b")}
    assert d2.independent(a, b, sources)
    assert d2.class_outcome("d2-selector-oxides", [(a, "inside", frozenset(("d2-selector-oxides",))),
                                                      (b, "inside", frozenset(("d2-selector-oxides",)))],
                            sources) is True
    for edit in ("lineage=unstated", "inputs=unstated", "inputs=self:a:cal"):
        broken = dict(b)
        if edit.startswith("lineage"):
            broken["qualification"] = b["qualification"].replace("lineage=own", "lineage=unstated")
        else:
            broken["qualification"] = b["qualification"].replace("inputs=self:b:cal", edit)
        assert not d2.independent(a, broken, sources)
        pair = [(a, "inside", frozenset(("d2-selector-oxides",))),
                (broken, "inside", frozenset(("d2-selector-oxides",)))]
        assert d2.class_outcome("d2-selector-oxides", pair, sources) is None
    sources["b"]["same_data_as"] = "a"
    assert not d2.independent(a, b, sources)
    sources = {"a": _source("a"), "b": _source("b"), "c": _source("c")}
    sources["a"]["same_data_as"] = "c"
    sources["b"]["same_data_as"] = "c"
    assert not d2.independent(a, b, sources)
    sources["a"]["same_data_as"] = ""
    sources["b"]["same_data_as"] = ""
    empty = dict(b)
    empty["qualification"] = b["qualification"].replace("inputs=self:b:cal", "inputs=")
    assert not d2.independent(a, empty, sources)
    unlocated = dict(b, qualification=b["qualification"].replace("lineage=own@b p.1", "lineage=own@"))
    assert not d2.independent(a, unlocated, sources)


def test_d2_limit_rows_never_support_a_pass() -> None:
    a, b = _row(), _row()
    b.update({"source_id": "b", "qualification": a["qualification"].replace("a p.1", "b p.1")
              .replace("self:a:cal", "self:b:cal")})
    sources = {"a": _source("a"), "b": _source("b")}
    record = "d2-selector-hydrogenous"
    subs = frozenset((record,))
    assert d2.class_outcome(record, [(a, "inside", subs), (b, "inside", subs)], sources) is True
    for results in (("not_excluded", "not_excluded"), ("inside", "not_excluded")):
        rows = [(row, result, subs) for row, result in zip((a, b), results, strict=True)]
        assert d2.class_outcome(record, rows, sources) is None


def test_d2_class_coverage_and_outside_priority() -> None:
    row = _row()
    sources = {"a": _source()}
    one = [(row, "inside", frozenset(("alloy",)))]
    assert d2.class_outcome("d2-selector-alloys", one, sources) is None
    assert d2.class_outcome("d2-selector-alloys", one + [(row, "outside", frozenset())], sources) is False
    assert d2.class_outcome("d2-initial-capture-general", one, sources) is None
    assert d2.class_outcome("d2-selector-powder-mixtures", one, sources) is None


def test_d2_unrelated_inside_row_does_not_block_covered_subclasses() -> None:
    a, b, c = _row(), _row(), _row()
    b.update({"source_id": "b", "qualification": a["qualification"].replace("a p.1", "b p.1")
              .replace("self:a:cal", "self:b:cal")})
    c["source_id"] = "c"
    c["qualification"] = c["qualification"].replace("lineage=own", "lineage=unstated")
    sources = {"a": _source("a"), "b": _source("b"), "c": _source("c")}
    rows = [(row, "inside", frozenset((sub,))) for row in (a, b)
            for sub in ("alloy", "intermetallic")]
    rows.append((c, "inside", frozenset(("unrelated",))))
    assert d2.class_outcome("d2-selector-alloys", rows, sources) is True


def test_d2_harvest_exact_selector_parity() -> None:
    builds = _csv("harvest_builds.csv")
    rows = _csv("selector_harvest.csv")
    assert {row["build"] for row in builds} == {row["build"] for row in rows}
    assert len(builds) == len({row["build"] for row in builds}) == 4
    expected_materials = {
        "G4_TEFLON", "G4_LITHIUM_FLUORIDE", "G4_POLYVINYL_CHLORIDE",
        "G4_SODIUM_IODIDE", "G4_WATER", "G4_POLYETHYLENE", "G4_SILICON_DIOXIDE",
        *(f"zscan-{z}" for z in range(1, 101)),
    }
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["build"], row["material"]].append(row)
    for build in {row["build"] for row in builds}:
        assert {material for b, material in groups if b == build} == expected_materials
    for (build, material), group in groups.items():
        assert [int(row["element_index"]) for row in group] == list(range(len(group)))
        elements = tuple(d2.Element(
            int(row["z"]), float.fromhex(row["atom_density"]),
            tuple(map(int, row["isotope_n"].split(";"))),
            tuple(float.fromhex(value) for value in row["isotope_abundance"].split(";")),
        ) for row in group)
        assert tuple(int(row["count"]) for row in group) == d2.selector_counts(elements), (build, material)
        assert all(int(row["isotope_at_half"]) == d2.isotope_at_half(element)
                   for row, element in zip(group, elements, strict=True)), (build, material)
        draws = {int(row["draws"]) for row in group}
        isotope_draws = sum(int(row["count"]) for row, element in zip(group, elements, strict=True)
                            if len(element.isotope_n) > 1)
        assert draws == {d2.DRAW_COUNT + isotope_draws}, (build, material)


def test_d2_harvest_build_provenance() -> None:
    manifest = json.loads((ROOT / "cpp/transport/evidence/manifest.json").read_text(encoding="utf-8"))
    clean_versions = {build["tag"]: build["version"] for build in manifest["builds"]
                      if build["kind"] == "pristine"}
    harness_hash = hashlib.sha256((ROOT / "cpp/tools/harvest_d2.cc").read_bytes()).hexdigest()
    for build in _csv("harvest_builds.csv"):
        assert build["harness_sha256"] == harness_hash
        assert build["geant4_version"] == clean_versions[build["revision"]]
        assert build["compile_flags"] == "-std=c++17 -O2 -ffp-contract=off"


def test_d2_bindings_reach_static_trace() -> None:
    sites = _csv("selector_call_sites.csv")
    bindings = _csv("selector_bindings.csv")
    revisions = {row["revision"] for row in sites}
    assert revisions == {build["revision"] for build in _csv("harvest_builds.csv")}
    for revision in revisions:
        rev_sites = [row for row in sites if row["revision"] == revision]
        assert {Path(row["file"]).name for row in rev_sites if row["kind"] == "caller"} == {
            "G4HadronStoppingProcess.cc", "G4MuonMinusAtomicCapture.cc"}
        assert {row["kind"] for row in rev_sites} == {"caller", "derived"}
        assert len({row["commit"] for row in rev_sites}) == 1
        derived = {Path(row["file"]).stem for row in rev_sites if row["kind"] == "derived"}
        assert derived == {"G4HadronicAbsorptionBertini", "G4HadronicAbsorptionFritiof",
                           "G4HadronicAbsorptionFritiofWithBinaryCascade", "G4HadronicAbsorptionINCLXX",
                           "G4MuonMinusCapture"}
        for build in _csv("harvest_builds.csv"):
            if build["revision"] != revision:
                continue
            for physics_list in ("QBBC", "FTFP_BERT", "QBBC+helper"):
                selected = [row for row in bindings if row["build"] == build["build"]
                            and row["physics_list"] == physics_list]
                assert {row["particle"] for row in selected} == {"mu-", "pi-", "kaon-", "anti_proton"}
                assert {row["class"] for row in selected if row["stopping_process"] == "true"} <= derived
                for particle in ("mu-", "pi-", "kaon-", "anti_proton"):
                    assert sum(row["stopping_process"] == "true" or row["atomic_capture"] == "true"
                               for row in selected if row["particle"] == particle) == 1
                if physics_list == "QBBC+helper":
                    assert any(row["class"] == "G4MuonMinusAtomicCapture" and row["atomic_capture"] == "true"
                               for row in selected if row["particle"] == "mu-")


def test_d2_measurements_bind_every_printed_primary_cell() -> None:
    printed = [row for row in _reference("printed_rows.csv") if row["row_kind"] == "data"]
    measurements = _reference("measurements.csv")
    sources = {row["source_id"]: row for row in _reference("sources.csv")}
    assert len(measurements) == len(printed)
    by_id = {row["record_id"]: row for row in measurements}
    assert len(by_id) == len(measurements)
    for cell in printed:
        record_id = "-".join(cell[field] for field in ("source_id", "table", "printed_page", "row_ordinal"))
        row = by_id[record_id]
        source = sources[cell["source_id"]]
        assert row["reported_value"] == cell["value_raw"]
        assert row["reported_uncertainty"] == cell["unc_raw"]
        assert row["source_sha256"] == source["copy_sha256"]
        assert source["access"] == "AVAILABLE" and source["kind"] == "measurement"
        assert source["primary_read"] == row["primary_read"] == "true"


_NAMED_MATERIAL = re.compile(r"(alloy|intermetallic|solution)\(([A-Z][a-z]?),([A-Z][a-z]?)\)")


def test_d2_measured_targets_follow_the_formula_grammar() -> None:
    rows = _reference("measurements.csv")
    assert rows
    for row in rows:
        formula = row["material_formula"]
        named = _NAMED_MATERIAL.fullmatch(formula)
        symbols = [named[2], named[3]] if named else [
            symbol for part in formula.split("+") for symbol in d2.formula_atoms(part)]
        assert symbols and set(symbols) <= set(d2._SYMBOLS), (row["record_id"], formula)


def test_d2_baijal_oxygen_footnote_qualifies_cus_and_pbs() -> None:
    printed = _reference("printed_rows.csv")
    footnote = next(row for row in printed if row["source_id"] == "Baijal1962OSTI"
                    and row["table"] == "Table IV" and row["row_kind"] == "footnote"
                    and "captures in oxygen" in row["note"])
    footnote_id = "-".join(footnote[field] for field in
                           ("source_id", "table", "printed_page", "row_ordinal"))
    rows = [row for row in _reference("measurements.csv") if row["source_id"] == "Baijal1962OSTI"
            and row["material_formula"] in ("CuS", "PbS")]
    assert rows
    assert {row["material_formula"] for row in rows} == {"CuS", "PbS"}
    for row in rows:
        assert "oxygen-capture footnote" in row["normalization"]
        assert footnote_id in row["normalization"]
        assert footnote["note"] in row["normalization"]
        assert footnote_id in row["stopping_model"]
        assert row["observable"] == "other"


def test_d2_yoshida_delayed_table_stays_ungated() -> None:
    printed = [row for row in _reference("printed_rows.csv") if row["source_id"] == "Yoshida2016"
               and row["table"] == "Table 1" and row["row_kind"] == "data"]
    measured = {row["record_id"]: row for row in _reference("measurements.csv")}
    sources = {row["source_id"]: row for row in _reference("sources.csv")}
    compared = {row["record_id"]: row for row in _csv("selector_vs_primary.csv")}
    assert printed
    for cell in printed:
        record_id = "-".join(cell[field] for field in ("source_id", "table", "printed_page", "row_ordinal"))
        row = measured[record_id]
        result = compared[record_id]
        assert row["qualification"].startswith("stage=terminal@Yoshida2016 p.116;")
        assert "population=transfer_only@Yoshida2016 p.116" in row["qualification"]
        assert d2.gate_reason(row, sources[row["source_id"]]) == "population transfer_only"
        assert result["population"] == "transfer_only"
        assert result["gated"] == "false" and result["reason"] == "population transfer_only"


def test_d2_loader_refuses_secondary_rows(tmp_path: Path) -> None:
    row = _row()
    row["record_id"] = "secondary"
    path = tmp_path / "measurements.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, d2.MEASUREMENT_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    for source in ({**_source(), "kind": "review"}, {**_source(), "primary_read": "false"}):
        with pytest.raises(ValueError, match="not primary-read"):
            d2.load_measurements(path, {"a": source})


def test_d2_comparison_bytes_regenerate_without_review_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    path = DATA / "selector_vs_primary.csv"
    before = path.read_bytes()
    original_open = Path.open

    def checked_open(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == "review_quoted.csv":
            raise AssertionError("comparison read review quotations")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    script = ROOT / "scripts/generate_g4d2.py"
    spec = importlib.util.spec_from_file_location("generate_g4d2_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.compare()
    assert path.read_bytes() == before


def test_d2_independence_and_each_subclass_require_source_evidence() -> None:
    a, b = _row(), _row()
    a["locator"] = "facility A; year 2001"
    b.update({"source_id": "b", "qualification": a["qualification"].replace("a p.1", "b p.1")
              .replace("self:a:cal", "self:b:cal").replace("method=xray", "method=lifetime"),
              "locator": "facility B; year 2002"})
    sources = {"a": _source("a"), "b": _source("b")}
    for record, subclasses in d2.SUBCLASSES.items():
        pairs = [(dict(a), "inside", frozenset((sub,))) for sub in subclasses]
        pairs += [(dict(b), "inside", frozenset((sub,))) for sub in subclasses]
        assert d2.class_outcome(record, pairs, sources) is True
        missing = [(row, result, subs) for row, result, subs in pairs
                   if next(iter(subclasses)) not in subs]
        assert d2.class_outcome(record, missing, sources) is None
        for edit in ("lineage=unstated", "inputs=self:a:cal"):
            broken = []
            for row, result, subs in pairs:
                changed = dict(row)
                if changed["source_id"] == "b":
                    changed["qualification"] = changed["qualification"].replace(
                        "lineage=own" if edit.startswith("lineage") else "inputs=self:b:cal", edit)
                broken.append((changed, result, subs))
            assert d2.class_outcome(record, broken, sources) is None
