"""Printed-row transcription structure for the published capture tables."""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import io
import json
import re
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmucf.g4 import provenance, spec
from openmucf.g4.sources import d1_nuclear_capture as d1
from openmucf.g4.sources import mizuno2025, suzuki1987

DATA = Path(__file__).resolve().parents[1] / suzuki1987.PRINTED_ROWS_RELPATH


def _load_generator():
    """`scripts/generate_g4data.py`, loaded by path -- `scripts/` is a directory, not a package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_g4data.py"
    module_spec = importlib.util.spec_from_file_location("generate_g4data", path)
    assert module_spec and module_spec.loader, path
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


gen = _load_generator()


def _csv_bytes(rows: list[dict[str, str]], columns: tuple[str, ...] | None = None) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=columns or suzuki1987.PRINTED_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("ascii")


def _row() -> dict[str, str]:
    row = dict.fromkeys(suzuki1987.PRINTED_COLUMNS, "")
    row.update(
        table="III", printed_page="2217", page_row_ordinal="1", row_kind="data",
        zeff_underlined="false", copy_read="published-scan",
    )
    return row


def _corruption(case: str) -> bytes:
    first = _row()
    rows = [first]
    if case == "header":
        return _csv_bytes(rows).replace(b"table,", b"wrong,", 1)
    if case == "cr":
        return _csv_bytes(rows).replace(b"\n", b"\r\n")
    if case == "ascii":
        return _csv_bytes(rows).replace(b"data", "daté".encode())
    if case == "shape":
        return _csv_bytes(rows)[:-1] + b",extra\n"
    if case == "table":
        first["table"] = "II"
    elif case == "integer":
        first["printed_page"] = "x"
    elif case == "positive":
        first["printed_page"] = "-1"
    elif case == "ordinal":
        second = _row()
        second["page_row_ordinal"] = "3"
        rows.append(second)
    elif case == "duplicate":
        rows.append(_row())
    elif case == "kind":
        first["row_kind"] = "invented"
    elif case == "underline":
        first["zeff_underlined"] = "maybe"
    elif case == "copy":
        first["copy_read"] = "preprint-scan"
    elif case == "empty":
        rows.clear()
    return _csv_bytes(rows)


# T-131: every structural refusal must name the offending file and line.
@pytest.mark.parametrize(
    "case",
    (
        "header", "cr", "ascii", "shape", "table", "integer", "positive", "ordinal",
        "duplicate", "kind", "underline", "copy", "empty", "missing",
    ),
)
def test_loader_refusals(tmp_path: Path, case: str) -> None:
    path = tmp_path / "printed_rows.csv"
    if case != "missing":
        path.write_bytes(_corruption(case))
    with pytest.raises(suzuki1987.Suzuki1987Error, match=r"printed_rows\.csv line \d+"):
        suzuki1987.load_printed_rows(path)


# T-132: the committed ledger is one ordered record per printed text line.
def test_committed_ledger_loads() -> None:
    rows = suzuki1987.load_printed_rows(DATA)
    assert rows
    assert all(row.locator.startswith(f"Table {row.table} p.{row.printed_page} row ") for row in rows)
    assert all(row.copy_read == "published-scan" for row in rows)
    assert {row.row_kind for row in rows} == suzuki1987.ROW_KINDS


def _printed() -> tuple[suzuki1987.PrintedRow, ...]:
    return suzuki1987.load_printed_rows(DATA)


def _normalized() -> tuple[dict[str, str], ...]:
    return suzuki1987.normalize(_printed())


# T-133: every printed data row has one generated quantity row and its own locator.
def test_quantity_rows_follow_printed_data() -> None:
    printed, quantity = _printed(), _normalized()
    assert len(quantity) == sum(row.row_kind == "data" for row in printed)
    assert suzuki1987.quantity_bytes(quantity) == (DATA.parent / "suzuki1987_quantity_rows.csv").read_bytes()
    for row in quantity:
        assert set(row) == set(suzuki1987.QUANTITY_COLUMNS)
        assert row["copy_read"] == "published-scan"
        assert row["label_from"].startswith(f"Table {row['table']} p.")
        assert row["rate_unit"] == ("s^-1" if row["table"] == "III" else "10^6/s")


# T-134: printed labels, Z inheritance, Huff direction, and rate notation stay distinct.
def test_normalized_conventions() -> None:
    rows = _normalized()
    by_locator = {(r["table"], r["printed_page"], r["page_row_ordinal"]): r for r in rows}
    printed = _printed()
    for source in printed:
        if source.row_kind != "data":
            continue
        row = by_locator[(source.table, str(source.printed_page), str(source.page_row_ordinal))]
        if source.z_raw:
            assert row["Z"] == source.z_raw and row["z_from"] == "printed"
        else:
            assert row["z_from"].startswith(f"Table {source.table} p.")
        if source.huff_raw:
            assert row["huff_factor"] == source.huff_raw and row["huff_from"] == "printed"
        if source.rate_raw.startswith("("):
            assert row["rate_parenthesized"] == "true"
        if source.rate_raw:
            assert gen._published_rate(row) == source.rate_raw
        if "+" in source.rate_raw and "(-" in source.rate_raw:
            assert not row["rate_unc"] and row["rate_unc_plus"] and row["rate_unc_minus"].startswith("-")
    continuations = [r for r in printed if r.element_raw.endswith(" (con't)")]
    assert continuations
    for source in continuations:
        row = by_locator[(source.table, str(source.printed_page), str(source.page_row_ordinal))]
        assert row["target_label"] == source.element_raw.removesuffix(" (con't)")
    mixture = next(r for r in rows if "." in r["target_label"])
    assert mixture["target_basis"] == "enriched_mixture"
    assert mixture["basis_rule"] == "noninteger-mass-prefix"
    assert not mixture["huff_factor"]
    ar = [r for r in rows if r["target_label"] == "Ar"]
    assert not ar[0]["huff_factor"] and ar[1]["huff_from"] == "printed"
    assert any(r["target_basis"] == "natural" for r in rows)
    assert any(r["target_basis"] == "unspecified" for r in rows)


@pytest.mark.parametrize(
    "case",
    ("label", "life", "rate_power", "rate", "unit", "missing_label", "symbol", "z", "header"),
)
# T-135: corrupt printed semantics are refused before a generated quantity is written.
def test_normalize_refusals(case: str) -> None:
    rows = list(_printed())
    if case == "header":
        rows.pop(next(i for i, r in enumerate(rows) if r.row_kind == "header"))
    else:
        index = next(i for i, r in enumerate(rows) if r.row_kind == "data")
        edits = {
            "label": {"element_raw": "43.8.8Ca"},
            "life": {"mean_life_raw": "bad"},
            "rate_power": {"rate_raw": "12+-3x10^x"},
            "rate": {"rate_raw": "bad"},
            "unit": {},
            "missing_label": {"element_raw": ""},
            "symbol": {},
            "z": {"z_raw": "x"},
        }
        if case == "unit":
            index = next(i for i, r in enumerate(rows) if r.row_kind == "header")
            rows[index] = dataclasses.replace(rows[index], rate_raw="unknown")
        elif case == "symbol":
            index = next(
                i for i, r in enumerate(rows)
                if r.row_kind == "data" and not r.z_raw and r.element_raw
            )
            rows[index] = dataclasses.replace(rows[index], element_raw="Xx")
        else:
            rows[index] = dataclasses.replace(rows[index], **edits[case])
    with pytest.raises(suzuki1987.Suzuki1987Error):
        suzuki1987.normalize(tuple(rows))


# T-136: the profile keys are exactly the eligible rows, with exact decimal scaling.
def test_profile_selection_and_provenance() -> None:
    rows = _normalized()
    selected = suzuki1987.selected(rows)
    eligible = [r for r in rows if r["own_measurement"] == "true"
                and r["target_basis"] in ("isotope", "natural") and r["rate_unc"]
                and r["rate_parenthesized"] == "false"]
    assert len(selected) == len(eligible)
    layer1_bytes = (DATA.parent / "d1_capture.suzuki1987.g4dat").read_bytes()
    layer2_bytes = (DATA.parent / "d1_capture.suzuki1987.prov.json").read_bytes()
    layer1 = spec.parse(layer1_bytes.decode("ascii"))
    layer2 = provenance.from_json_obj(json.loads(layer2_bytes.decode("ascii")))
    assert provenance.document_bytes(gen.build_suzuki_capture_document(selected)) == layer2_bytes
    assert spec.render(gen.build_suzuki_capture_table(
        selected, provenance.source_digest(layer2_bytes)
    )).encode("ascii") == layer1_bytes
    assert layer1.directives["PROFILE"] == layer2.profile == suzuki1987.PROFILE
    assert "SOURCESHA" not in layer1.directives and "FALLBACK" not in layer1.directives
    assert layer2.precedence == (suzuki1987.PROFILE,)
    assert layer1.directives["VERSION"] == layer2.version
    assert set(layer2.rows) == {f"{z}-{a}" for z, a in selected}
    assert {(int(z), int(a)) for z, a, *_ in layer1.records} == set(selected)
    assert all(a != 0 for _, a in selected) == (spec.validity_assignments(layer1)["A"] == "listed")
    for z, a, value, unc in layer1.records:
        source = selected[(int(z), int(a))]
        assert Decimal(repr(value)) == Decimal(repr(float(suzuki1987.scaled(source, "rate"))))
        assert Decimal(repr(unc)) == Decimal(repr(float(suzuki1987.scaled(source, "rate_unc"))))
        proof = layer2.rows[f"{z}-{a}"]
        assert proof.source_bibkey == suzuki1987.BIBKEY
        assert proof.source_library == suzuki1987.PROFILE
        assert proof.evaluation_id == suzuki1987.EVALUATION_ID
        assert proof.source_locator == (
            f"Table {source['table']} p.{source['printed_page']} row {source['page_row_ordinal']} "
            "[copy read: published-scan]"
        )
        assert proof.unc_type == "table" and proof.needs_verification is False
        assert proof.single_source is (source["refs"] == "a")
        assert proof.isotope_resolved is (source["target_basis"] == "isotope")
        assert proof.validity_range == f"Z={z} A={a}"
        assert proof.recommendation == ""
        printed_huff = source["huff_factor"] if source["huff_from"] == "printed" else "none on this row"
        assert proof.conditions == (
            f"target label as printed: {source['target_label']}; mean life as printed: "
            f"{source['mean_life_ns']}+-{source['mean_life_unc_ns']} ns; Huff factor as printed: "
            f"{printed_huff}; copy read: page images of the published article"
        )
        assert proof.evaluation_method == (
            f"total capture rate as the primary prints it (Table {source['table']}, column "
            f"'Total capture rate ({source['rate_unit']})'), scaled exactly to 1e6/s; "
            "not re-derived here"
        )


# T-137: empty and repeated selections cannot become a profile.
@pytest.mark.parametrize("case", ("empty", "duplicate", "mass"))
def test_selection_refusals(case: str) -> None:
    rows = _normalized()
    eligible = next(r for r in rows if r["own_measurement"] == "true" and r["target_basis"] == "isotope"
                    and r["rate_unc"] and r["rate_parenthesized"] == "false")
    if case == "empty":
        source = tuple(dict(row, own_measurement="false") for row in rows)
    elif case == "duplicate":
        source = rows + (eligible,)
    else:
        source = (dict(eligible, target_label="Xx"),)
    with pytest.raises(suzuki1987.Suzuki1987Error):
        suzuki1987.selected(source)


# T-140: each eligibility clause excludes a row without altering the eligible key set.
@pytest.mark.parametrize("field,value", (
    ("own_measurement", "false"), ("target_basis", "enriched_mixture"),
    ("rate_unc", ""), ("rate_parenthesized", "true"),
))
def test_selection_filters(field: str, value: str) -> None:
    rows = _normalized()
    base = suzuki1987.selected(rows)
    eligible = next(row for row in rows if row in base.values())
    blocked = dict(eligible, **{field: value})
    assert suzuki1987.selected(rows + (blocked,)) == base


# T-138: comparisons enumerate every compiled-in row at a published Z and expose the printed 236U cell.
def test_comparison_files_and_mizuno_lifetimes() -> None:
    rows = _normalized()
    found = d1.load(gen.VENDORED_PATH)
    with (DATA.parent / "suzuki1987_parity_cells.csv").open(newline="", encoding="ascii") as stream:
        parity_rows = list(csv.DictReader(stream))
    published_z = {int(row["Z"]) for row in rows}
    assert {(int(row["Z"]), int(row["A"])) for row in parity_rows} == {
        (z, a) for z, a, _, _ in found.capture_records if z in published_z
    }
    uranium = next(r for r in rows if r["target_label"].startswith("236U"))
    parity = next(r for r in parity_rows if (r["Z"], r["A"]) == (uranium["Z"], "236"))
    assert int(parity["equal_count"]) > 0
    assert f"row {uranium['page_row_ordinal']}" in parity["equal_locators"]
    with (DATA.parent / "suzuki1987_preprint_differences.csv").open(newline="", encoding="ascii") as stream:
        differences = list(csv.DictReader(stream))
    assert differences
    assert {r["kind"] for r in differences} <= {
        "refs", "value", "unc", "label", "bracket", "zeff", "printed_z", "underline", "unmatched",
        "not_in_committed_preprint_cells",
    }
    assert any(r["kind"] == "not_in_committed_preprint_cells"
               and r["published_locator"].endswith(f"row {uranium['page_row_ordinal']}")
               for r in differences)
    assert all(row["preprint"] != row["published"] for row in differences)
    zeff = d1.load_zeff_audit(DATA.parents[3] / d1.ZEFF_AUDIT_RELPATH)
    printed_z = {
        int(row.z_raw): row for row in _printed()
        if row.row_kind == "data" and row.z_raw and row.zeff_raw
    }
    changed_z = {
        z for z, old in zeff.items()
        if z in printed_z and old.printed_z != int(printed_z[z].z_raw)
    }
    assert {int(row["Z"]) for row in differences if row["kind"] == "printed_z"} == changed_z
    for z in changed_z:
        matching = [row for row in differences if row["kind"] == "printed_z" and int(row["Z"]) == z]
        assert len(matching) == 1
        assert matching[0]["preprint"] == str(zeff[z].printed_z)
        assert matching[0]["published"] == printed_z[z].z_raw
    own = [r for r in rows if r["own_measurement"] == "true"]
    table1 = mizuno2025.load_table1(DATA.parent / "mizuno2025_table1.csv")
    for target in table1:
        if not target.has_suzuki_value:
            continue
        symbol = re.search(r"[A-Z][a-z]?$", target.nuclide)
        assert symbol is not None
        matches = []
        for row in own:
            printed_symbol = re.search(r"[A-Z][a-z]?(?:\^[a-z])?$", row["target_label"])
            if printed_symbol and printed_symbol.group().startswith(symbol.group()):
                matches.append(row)
        assert (target.suzuki_lifetime_ns, target.suzuki_lifetime_unc_ns) in {
            (r["mean_life_ns"], r["mean_life_unc_ns"]) for r in matches
        }


# T-139: each generated Suzuki output remains a required generator artifact.
def test_generator_includes_every_suzuki_output() -> None:
    artifacts, _archive = gen.build_dataset_artifacts()
    expected = (
        gen.D1_SUZUKI_LAYER1, gen.D1_SUZUKI_LAYER2, gen.SUZUKI_QUANTITY_PATH,
        gen.SUZUKI_PARITY_CELLS, gen.SUZUKI_PREPRINT_DIFF,
    )
    for path in expected:
        assert path in artifacts
        assert artifacts[path] == path.read_bytes()


# T-141: the committed Suzuki pair receives the same E009 audit as the other pairs.
def test_audit_checks_suzuki_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    original = gen.D1_SUZUKI_LAYER2
    corrupted = original.read_bytes().replace(
        b'"profile": "suzuki1987"', b'"profile": "corrupted"'
    )
    read_bytes = Path.read_bytes

    def read_corrupted(path: Path) -> bytes:
        if path == original:
            return corrupted
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_corrupted)
    with pytest.raises(SystemExit, match="E009"):
        gen.audit()


# T-142: a base release has three numeric components before its patch is advanced.
def test_version_bump_rejects_a_short_or_nonnumeric_base() -> None:
    parts = gen._BASE_DATASET_VERSION.split(".")
    assert f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}" == gen.DATASET_VERSION
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        gen._next_patch_version(".".join(parts[:-1]))
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        gen._next_patch_version(".".join((*parts, "0")))
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        gen._next_patch_version(".".join((parts[0], "word", parts[-1])))


# T-143: an equal value with a different uncertainty is not an equal published cell.
def test_parity_comparison_requires_both_printed_cells() -> None:
    source = next(row for row in _normalized() if row["rate"] and row["rate_unc"])
    value = float(suzuki1987.scaled(source, "rate"))
    uncertainty = float(suzuki1987.scaled(source, "rate_unc") * 2)
    fake = SimpleNamespace(
        capture_records=((int(source["Z"]), 0, value, uncertainty),),
        capture_literals=((repr(value), repr(uncertainty)),),
    )
    cells = gen.build_suzuki_parity_cells(fake, (source,))
    (comparison,) = list(csv.DictReader(io.StringIO(cells.decode("ascii"))))
    assert comparison["equal_count"] == "0" and not comparison["equal_locators"]


# T-149: each published-copy audit clause refuses its own isolated corruption.
@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("flag", "isotope_resolved must be true"),
        ("opening", "evidence must open with the separated isotope label"),
        ("no_equal", "exactly one published cell"),
        ("two_equal", "exactly one published cell"),
        ("basis", "matched cell must carry the isotope mass and symbol"),
        ("symbol", "matched cell must carry the isotope mass and symbol"),
        ("locator", "locator must name the matched published table and page"),
    ),
)
def test_t149_published_audit_guard_clauses(monkeypatch, case: str, message: str) -> None:
    found = d1.load(gen.VENDORED_PATH)
    audit = d1.load_isotope_audit(gen.ROOT / d1.AUDIT_RELPATH)
    published_keys = [key for key, finding in audit.items()
                      if finding.copy_read == suzuki1987.COPY_READ]
    assert len(published_keys) == 1
    (key,) = published_keys
    finding = audit[key]
    rows = list(_normalized())
    gen.build_capture_document(found, tuple(rows))
    z, a = key
    symbol = finding.evidence.split("separated isotope ", 1)[1].split("-", 1)[0]
    matching = next(row for row in rows if int(row["Z"]) == z
                    and row["target_label"].partition("^")[0] == f"{a}{symbol}")
    changed = dict(audit)
    if case == "flag":
        changed[key] = dataclasses.replace(finding, isotope_resolved=False)
    elif case == "opening":
        changed[key] = dataclasses.replace(finding, evidence="wrong opening; " + finding.evidence)
    elif case == "locator":
        changed[key] = dataclasses.replace(finding, locator=finding.locator + " wrong")
    elif case == "two_equal":
        rows.append(dict(matching))
    else:
        index = rows.index(matching)
        altered = dict(matching)
        if case == "no_equal":
            altered["rate_unc"] = ""
        elif case == "basis":
            altered["target_basis"] = "unspecified"
        elif case == "symbol":
            altered["target_label"] = f"{a}X"
        rows[index] = altered
    monkeypatch.setattr(d1, "load_isotope_audit", lambda _path: changed)
    with pytest.raises(SystemExit, match=message):
        gen.build_capture_document(found, tuple(rows))


# T-150: each printed-precision conjunct is necessary to identify an equal cell.
def test_t150_published_equal_cells_require_value_uncertainty_and_printed_uncertainty() -> None:
    source = next(row for row in _normalized() if row["rate"] and row["rate_unc"])
    value = float(suzuki1987.scaled(source, "rate"))
    unc = float(suzuki1987.scaled(source, "rate_unc"))
    assert gen._published_equal_cells(value, unc, [source]) == [source]
    assert not gen._published_equal_cells(value * 2, unc, [source])
    assert not gen._published_equal_cells(value, unc * 2, [source])
    no_unc = dict(source)
    no_unc["rate_unc"] = ""
    assert not gen._published_equal_cells(value, unc, [no_unc])
