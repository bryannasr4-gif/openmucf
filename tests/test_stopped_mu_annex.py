"""The section of `BENCHMARK_STOPPED_MU.md` on the other stages of a stopped muon
(`scripts/generate_stopped_mu.py`, `stage_inputs` and `render_stages`).

The section restates, from six committed files of the datasets, what `DATASET_D1.md`, `DATASET_D2.md` and
`DATASET_D3.md` state about the capturing atom, the X-ray energies, the capture record and the disappearance
rate. These tests hold each number it prints to the number the dataset document's own test holds, drill every
refusal of `stage_inputs` on a tampered copy of the six files (each against a green control on the untampered
copy), show that `compare` refuses before it writes any file, that the section reads no file of the
transported natural-target comparison, that the sentence on the routes is generated for every pattern, and
that a moved number moves the section. No count of a dataset is written here as a literal: every expectation
comes from a committed file, the owning module or the dataset document's own pins.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import io
import json
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import test_g4d2
import test_g4d3
import test_stopped_mu as tsm

from openmucf.g4 import d1_lifetime as lifetime
from openmucf.g4 import d1_mixture as mixture
from openmucf.g4 import d2
from openmucf.g4 import d3_contract as d3c

REPO = tsm.REPO
gen = tsm.gen
DOCUMENT = REPO / "BENCHMARK_STOPPED_MU.md"
CSV_NAMES = tuple(gen.STAGE_COLUMNS)
HEADING = "## Other stages of a stopped muon, from the dataset documents"


@functools.lru_cache(maxsize=1)
def committed_stages():
    return gen.stage_inputs(REPO)


def section(stages=None) -> str:
    return gen.stages_text(committed_stages() if stages is None else stages).text


def collapsed(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------------------------
# a copy of the six files, and edits of it
# --------------------------------------------------------------------------------------------
def copy_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for relpath in gen.STAGE_FILES.values():
        (root / relpath).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relpath, root / relpath)
    return root


def path_of(root: Path, name: str) -> Path:
    return root / gen.STAGE_FILES[name]


def edit_bytes(root: Path, name: str, change: Callable[[bytes], bytes]) -> None:
    path = path_of(root, name)
    before = path.read_bytes()
    after = change(before)
    assert after != before, name
    path.write_bytes(after)


def records(data: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))


def serialized(rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode("utf-8")


def edit_records(root: Path, name: str, change: Callable[[list[list[str]]], list[list[str]]]) -> None:
    edit_bytes(root, name, lambda data: serialized(change(records(data))))


def set_cell(
    root: Path, name: str, column: str, value: str, where: Callable[[dict[str, str]], bool] | None = None
) -> int:
    """Set ``column`` to ``value`` on every data record ``where`` selects (the first record when ``where`` is
    None); the number of records changed."""
    columns = gen.STAGE_COLUMNS[name]
    index = columns.index(column)
    hit = 0

    def change(rows: list[list[str]]) -> list[list[str]]:
        nonlocal hit
        for number, record in enumerate(rows[1:]):
            row = dict(zip(columns, record, strict=True))
            if (number == 0) if where is None else where(row):
                record[index] = value
                hit += 1
        return rows

    edit_records(root, name, change)
    assert hit, (name, column)
    return hit


def edit_status(root: Path, change: Callable[[dict], None]) -> None:
    def edit(data: bytes) -> bytes:
        record = json.loads(data)
        change(record)
        return json.dumps(record, indent=1).encode("utf-8")

    edit_bytes(root, "numerical_check_status", edit)


def refused(root: Path, pattern: str) -> None:
    with pytest.raises(gen.StageError, match=pattern):
        gen.stage_inputs(root)


# --------------------------------------------------------------------------------------------
# the committed section
# --------------------------------------------------------------------------------------------
def test_the_committed_files_are_admitted_and_a_copy_reads_the_same():
    """Every guard of stage_inputs admits the six committed files, and an untampered copy of them gives the
    same values: the green control of every drill below."""
    stages = committed_stages()
    assert stages.records and stages.selector_rows and stages.gated and stages.groups and stages.summary


def test_the_green_control_holds_on_a_copy(tmp_path):
    assert gen.stage_inputs(copy_root(tmp_path)) == committed_stages()


def test_rewriting_a_table_unchanged_keeps_its_bytes():
    """The record editor of the drills below rewrites each table byte for byte when it changes nothing, so
    every drill differs from the committed file only by its own edit."""
    for name in CSV_NAMES:
        data = path_of(REPO, name).read_bytes()
        assert serialized(records(data)) == data, name


def test_the_document_carries_the_section_between_the_cascade_and_reproducing():
    text = DOCUMENT.read_text(encoding="utf-8")
    assert text.count(section()) == 1
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    at = headings.index(HEADING)
    assert headings[at - 1] == "## The muonic X-ray cascade" and headings[at + 1] == "## Reproducing"
    subsections = [line for line in section().splitlines() if line.startswith("### ")]
    assert subsections == [
        "### The capturing atom",
        "### The X-ray energies",
        "### The capture record",
        "### The disappearance rate",
    ]


def test_the_section_uses_no_forbidden_phrase_and_names_no_transport_file():
    assert gen.forbidden_hits(section()) == []
    assert "both routes fail" in gen.FORBIDDEN
    assert "natural_mixture" not in section() and "natural_target_answers" not in section()


#: Clauses the section copies from a dataset document: each stands, whitespace-collapsed, in the section and
#: in its source.
COPIED = (
    ("adopted after the selector's ratio had already been compared with the rows it gates", "DATASET_D2.md"),
    (
        "the difference may come from such transfer rather than from the weights, and this comparison does "
        "not tell them apart",
        "DATASET_D2.md",
    ),
    ("a reconstructed combination, not a demonstrated history", "DATASET_D1.md"),
    (
        "The comparison is of the rates the code returns for a single nucleus, not of a transported "
        "simulation of a target",
        "DATASET_D1.md",
    ),
    (
        "a target of natural composition reaches an element's record only on the stops that draw the "
        "record's own mass number",
        "DATASET_D1.md",
    ),
    ("The compiled-in table answers only a nucleus whose Z and A both match a record", "DATASET_D1.md"),
)


@pytest.mark.parametrize(("clause", "source"), COPIED, ids=[f"{s}:{c[:24]}" for c, s in COPIED])
def test_each_copied_clause_stands_in_the_section_and_in_its_source(clause, source):
    assert clause in collapsed(section())
    assert clause in collapsed((REPO / source).read_text(encoding="utf-8"))


def test_the_class_table_prints_the_module_definitions():
    text = section()
    for name, count in committed_stages().classes:
        assert f"| `{name}` | {mixture.CLASS_TEXT[name]} | {count} |" in text


def test_every_status_of_the_contract_has_a_sentence():
    assert set(gen.STATUS_TEXT) == set(d3c.NUMERICAL_CHECK_STATUSES)
    assert set(gen.FAILING_STATUSES) <= set(gen.STATUS_TEXT)


# --------------------------------------------------------------------------------------------
# V6 -- every number equals the number the dataset document's own test holds
# --------------------------------------------------------------------------------------------
def test_the_routes_table_is_the_table_of_the_d1_module():
    rows = lifetime.read_comparison(REPO / lifetime.COMPARISON_RELPATH)
    assert section().count(lifetime.render_summary_table(rows) + "\n") == 1
    assert committed_stages().summary == tuple(lifetime.summary(rows))


def _expected(pins, names: dict[str, str]) -> dict[str, object]:
    by_name = {what: expected for what, _path, _pattern, _groups, expected in pins}
    return {field: by_name[what] for field, what in names.items()}


def test_the_selector_numbers_are_the_ones_the_d2_document_states():
    expected = _expected(
        test_g4d2.document_pins(),
        {
            "selector_rows": "method-gated rows",
            "ratio_min": "smallest selector ratio over the measured",
            "ratio_max": "largest selector ratio over the measured",
            "ratio_median": "median selector ratio over the measured",
            "selector_hydrogenous": "rows of a material that holds hydrogen",
        },
    )
    stages = committed_stages()
    assert {field: getattr(stages, field) for field in expected} == expected


def test_the_energy_counts_are_the_ones_the_d3_document_states():
    expected = _expected(
        test_g4d3.contract_pins(),
        {
            "gated": "gated rows in the projection",
            "consumer_within": "rows whose shell difference lies within the band",
            "solver_within": "rows whose solver line lies within the band",
            "stock_within": "rows whose unpatched cascade lies within the band",
            "consumer_closer": "rows whose shell difference is closer than the unpatched cascade",
            "groups_empty": "groups with an empty intersection",
            "groups": "multi-line groups",
        },
    )
    stages = committed_stages()
    assert {field: getattr(stages, field) for field in expected} == expected


def test_the_status_record_is_the_one_the_d3_document_states():
    """The status and its counts, read from the patterns of the D3 document's own numerical pins in its text
    (so no fresh load of the D3 bundle is needed)."""
    text = collapsed((REPO / "DATASET_D3.md").read_text(encoding="utf-8"))
    stated = {}
    for field, pattern in (
        ("check_rows", r"A fresh solve of all (\d+) rows"),
        ("failed_rows", r"of the target on (\d+) rows"),
        ("check_cases", r"The file carries \d+ rows over (\d+) nuclides"),
        ("status", r"gives the check the status `([A-Z_]+)`"),
    ):
        [hit] = re.findall(pattern, text)
        stated[field] = hit if field == "status" else int(hit)
    stages = committed_stages()
    assert {field: getattr(stages, field) for field in stated} == stated


def test_the_record_counts_equal_an_independent_recount_of_the_isotope_audit():
    """Recounted from `isotope_audit.csv` (the flag) and `d1_capture.prov.json` (`needs_verification`), files
    the section does not read."""
    with (REPO / "data/g4/d1/isotope_audit.csv").open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))
    prov = json.loads((REPO / "data/g4/d1/d1_capture.prov.json").read_text(encoding="utf-8"))["rows"]
    assert sorted(f"{row['Z']}-{row['A']}" for row in audit) == sorted(prov)
    settled = [row for row in audit if prov[f"{row['Z']}-{row['A']}"]["needs_verification"] is False]
    stages = committed_stages()
    assert (stages.records, stages.settled, stages.unsettled) == (
        len(audit),
        len(settled),
        len(audit) - len(settled),
    )
    assert stages.resolved == sum(row["isotope_resolved"] == "true" for row in settled)
    assert stages.unresolved == sum(row["isotope_resolved"] == "false" for row in settled)


def test_the_class_counts_are_the_parity_column_of_the_d1_class_table():
    rendered = mixture.render_section(REPO)
    table = {}
    for line in rendered.splitlines():
        hit = re.fullmatch(r"\| `([a-z_]+)` \| .* \| (\d+) \| (\d+) \|", line)
        if hit and hit[1] in mixture.CLASS_TEXT:
            table[hit[1]] = int(hit[2])
    header = next(line for line in rendered.splitlines() if line.startswith("| class |"))
    assert header.split("|")[-3].strip() == "records, `parity`", header
    assert dict(committed_stages().classes) == {name: count for name, count in table.items() if count}


# --------------------------------------------------------------------------------------------
# V1 -- the six files: structure
# --------------------------------------------------------------------------------------------
def _non_utf8(data: bytes) -> bytes:
    head, rest = data.split(b"\n", 1)
    return head + b"\n" + rest.replace(b",", b",\xff", 1)


def _header(data: bytes, change: Callable[[list[str]], list[str]]) -> bytes:
    rows = records(data)
    return serialized([change(rows[0]), *rows[1:]])


def _all(data: bytes, change: Callable[[list[str]], list[str]]) -> bytes:
    return serialized([change(record) for record in records(data)])


def _record(data: bytes, change: Callable[[list[str]], list[str]]) -> bytes:
    rows = records(data)
    return serialized([rows[0], change(rows[1]), *rows[2:]])


STRUCTURE = {
    "non-utf8": (_non_utf8, "the file is not UTF-8"),
    "bom": (lambda d: b"\xef\xbb\xbf" + d, "the header is not the schema"),
    "cr": (lambda d: d.replace(b"\n", b"\r\n", 1), "contains a CR byte"),
    "empty": (lambda d: b"", "the file is empty"),
    "header-only": (lambda d: d.split(b"\n", 1)[0] + b"\n", "a header and no rows"),
    "duplicated-header-name": (
        lambda d: _header(d, lambda h: [h[0], h[0], *h[2:]]),
        "the header is not the schema",
    ),
    "missing-column": (lambda d: _all(d, lambda r: r[:-1]), "the header is not the schema"),
    "extra-column": (lambda d: _all(d, lambda r: [*r, "x"]), "the header is not the schema"),
    "reordered-columns": (lambda d: _all(d, lambda r: [r[1], r[0], *r[2:]]), "the header is not the schema"),
    "short-record": (lambda d: _record(d, lambda r: r[:-1]), "do not have exactly"),
    "long-record": (lambda d: _record(d, lambda r: [*r, "x"]), "do not have exactly"),
    "duplicate-row": (lambda d: d + d.split(b"\n")[1] + b"\n", "repeats"),
}


@pytest.mark.parametrize("fault", list(STRUCTURE))
@pytest.mark.parametrize("name", CSV_NAMES)
def test_drill_a_table_with_a_broken_structure_is_refused(tmp_path, name, fault):
    change, pattern = STRUCTURE[fault]
    root = copy_root(tmp_path)
    edit_bytes(root, name, change)
    refused(root, re.escape(gen.STAGE_FILES[name]) + ".*" + re.escape(pattern))


#: For each table, a key collision whose other cells differ: the second row takes the key of the first and
#: keeps everything else of its own.
@pytest.mark.parametrize("name", CSV_NAMES)
def test_drill_two_rows_with_one_key_and_different_cells_are_refused(tmp_path, name):
    columns, key = gen.STAGE_COLUMNS[name], gen.STAGE_KEYS[name]

    def collide(rows: list[list[str]]) -> list[list[str]]:
        first, second = rows[1], rows[2]
        assert any(first[columns.index(c)] != second[columns.index(c)] for c in key)
        for column in key:
            second[columns.index(column)] = first[columns.index(column)]
        return rows

    root = copy_root(tmp_path)
    edit_records(root, name, collide)
    refused(root, re.escape(gen.STAGE_FILES[name]) + ": the key .* repeats")


# --------------------------------------------------------------------------------------------
# V1 -- the six files: fields and rosters
# --------------------------------------------------------------------------------------------
BAD_FLAGS = ("", "True", "1", "yes", " true", "maybe")
FLAG_CELLS = (
    ("record_classes", "isotope_resolved"),
    ("record_classes", "needs_verification"),
    ("shell_projection", "consumer_within"),
    ("shell_projection", "stock_within"),
    ("shell_projection", "solver_within"),
    ("shell_projection", "consumer_closer_than_stock"),
    ("incompatible_groups", "empty"),
)


@pytest.mark.parametrize("value", BAD_FLAGS, ids=repr)
@pytest.mark.parametrize(("name", "column"), FLAG_CELLS, ids=[f"{n}.{c}" for n, c in FLAG_CELLS])
def test_drill_a_flag_that_is_not_true_or_false_is_refused(tmp_path, name, column, value):
    """A flag cell other than `true` or `false` is refused, never counted as `false`."""
    root = copy_root(tmp_path)
    set_cell(root, name, column, value)
    refused(root, f"{column} {re.escape(repr(value))} is not true or false")


@pytest.mark.parametrize("value", BAD_FLAGS, ids=repr)
@pytest.mark.parametrize("route", [suffix for suffix, _ in lifetime.ROUTES])
def test_drill_a_compared_row_whose_route_cell_is_not_true_or_false_is_refused(tmp_path, route, value):
    root = copy_root(tmp_path)
    compared = [
        r for r in tsm.read_csv(path_of(REPO, "mean_life_comparison")) if r["population"] == "compared"
    ]
    target = compared[0]["locator"], compared[0]["profile"]
    set_cell(
        root,
        "mean_life_comparison",
        f"within_{route}",
        value,
        lambda row: (row["locator"], row["profile"]) == target,
    )
    refused(root, f"within_{route} {re.escape(repr(value))} on a compared row")


def test_the_route_cells_are_empty_only_where_the_comparison_leaves_them_empty():
    """Main has empty route cells only on rows whose population is not `compared`; the guard admits them
    there."""
    rows = tsm.read_csv(path_of(REPO, "mean_life_comparison"))
    empty = [r for r in rows if r["within_bound"] == "" or r["within_helper"] == ""]
    assert empty and all(r["population"] != lifetime.COMPARED for r in empty)


BAD_INTEGERS = ("+7", " 7", "1_0", "7.0", "-1", "nan", "", "07")
INTEGER_CELLS = (
    ("record_classes", "z"),
    ("shell_projection", "A"),
    ("selector_summary", "rows"),
    ("selector_summary", "hydrogenous"),
    ("incompatible_groups", "initial_n"),
)


@pytest.mark.parametrize("value", BAD_INTEGERS, ids=repr)
@pytest.mark.parametrize(("name", "column"), INTEGER_CELLS, ids=[f"{n}.{c}" for n, c in INTEGER_CELLS])
def test_drill_an_integer_cell_python_would_parse_but_the_tables_never_write_is_refused(
    tmp_path, name, column, value
):
    root = copy_root(tmp_path)
    set_cell(root, name, column, value)
    refused(root, f"{column} {re.escape(repr(value))} is not a non-negative integer")


@pytest.mark.parametrize("value", ("nan", "inf", "-1.59", "1e0", "", "1.", ".5"), ids=repr)
@pytest.mark.parametrize("column", ("ratio_min", "ratio_median", "ratio_max"))
def test_drill_a_ratio_cell_that_is_not_a_plain_decimal_is_refused(tmp_path, column, value):
    root = copy_root(tmp_path)
    set_cell(root, "selector_summary", column, value, lambda row: row["scope"] == "method_gated")
    refused(root, f"{column} {re.escape(repr(value))} is not a non-negative decimal ratio")


VOCABULARY = {
    "class": ("record_classes", "answer_class", "parity_guessed", "is not a class of the D1 module"),
    "evaluated-class-on-a-parity-record": (
        "record_classes",
        "answer_class",
        mixture.EVALUATED_NATURAL,
        "has no definition to print",
    ),
    "population": ("mean_life_comparison", "population", "compared-ish", "is not one of the D1 module's"),
    "source": ("shell_projection", "source", "Fricke1996", "has no citation"),
}


@pytest.mark.parametrize("case", list(VOCABULARY))
def test_drill_a_word_outside_its_vocabulary_is_refused(tmp_path, case):
    name, column, value, pattern = VOCABULARY[case]
    root = copy_root(tmp_path)
    set_cell(root, name, column, value)
    refused(root, pattern)


def test_the_class_vocabulary_admits_the_evaluated_natural_class_off_the_parity_profile():
    """`evaluated_natural` is not in `d1_mixture.CLASSES`, and committed rows of an evaluated profile carry
    it: the guard admits it there and refuses it only on a record of the compiled-in table, whose class it
    prints."""
    rows = tsm.read_csv(path_of(REPO, "record_classes"))
    assert mixture.EVALUATED_NATURAL not in mixture.CLASSES
    assert any(r["answer_class"] == mixture.EVALUATED_NATURAL for r in rows)
    assert all(r["profile"] != "parity" for r in rows if r["answer_class"] == mixture.EVALUATED_NATURAL)


def test_drill_the_rosters_the_section_needs_are_refused_when_missing(tmp_path):
    """No `method_gated` row, a `method_gated` row twice, no record of the compiled-in table, and a profile
    with no compared row: each is refused, never rendered as a count of nothing."""
    cases = []
    root = copy_root(tmp_path / "missing")
    set_cell(
        root, "selector_summary", "scope", "method_gated_old", lambda row: row["scope"] == "method_gated"
    )
    cases.append((root, "no row method_gated"))
    root = copy_root(tmp_path / "twice")
    edit_records(
        root, "selector_summary", lambda rows: [*rows, next(r for r in rows if r[0] == "method_gated")]
    )
    cases.append((root, "the key .*method_gated.* repeats"))
    root = copy_root(tmp_path / "parity")
    set_cell(root, "record_classes", "profile", "parity_old", lambda row: row["profile"] == "parity")
    cases.append((root, "no record of the profile parity"))
    root = copy_root(tmp_path / "profile")
    last = committed_stages().summary[-1][0]
    edit_records(
        root,
        "mean_life_comparison",
        lambda rows: [r for r in rows if not (r[7] == last and r[9] == lifetime.COMPARED)],
    )
    cases.append((root, f"the profile '{last}' has no compared row"))
    for root, pattern in cases:
        refused(root, pattern)
    assert gen.STAGE_COLUMNS["mean_life_comparison"][7] == "profile"
    assert gen.STAGE_COLUMNS["mean_life_comparison"][9] == "population"


@pytest.mark.parametrize("column", gen.SELECTOR_COUNTS)
def test_drill_a_selector_count_beyond_its_rows_or_no_rows_is_refused(tmp_path, column):
    gated = next(r for r in tsm.read_csv(path_of(REPO, "selector_summary")) if r["scope"] == "method_gated")
    root = copy_root(tmp_path / "beyond")
    set_cell(
        root, "selector_summary", column, str(int(gated["rows"]) + 1), lambda r: r["scope"] == "method_gated"
    )
    refused(root, f"{column} exceeds the rows of method_gated")
    root = copy_root(tmp_path / "none")
    set_cell(root, "selector_summary", "rows", "0", lambda r: r["scope"] == "method_gated")
    refused(root, "describes no row")


# --------------------------------------------------------------------------------------------
# V1 -- the status record
# --------------------------------------------------------------------------------------------
def _replace_text(old: str, new: str) -> Callable[[bytes], bytes]:
    def change(data: bytes) -> bytes:
        text = data.decode("utf-8")
        assert text.count(old) == 1, old
        return text.replace(old, new).encode("utf-8")

    return change


def _failed_rows() -> int:
    return committed_stages().failed_rows


@pytest.mark.parametrize("literal", ("NaN", "Infinity", "-Infinity"))
def test_drill_a_non_finite_literal_in_the_status_record_is_refused(tmp_path, literal):
    root = copy_root(tmp_path)
    edit_bytes(
        root,
        "numerical_check_status",
        _replace_text(f'"failed_rows": {_failed_rows()}', f'"failed_rows": {literal}'),
    )
    refused(root, f"{literal} is not a number")


def test_drill_a_repeated_key_in_the_status_record_is_refused(tmp_path):
    root = copy_root(tmp_path)
    edit_bytes(root, "numerical_check_status", lambda d: d.replace(b"{", b'{\n "status": "OTHER",', 1))
    refused(root, "the key 'status' repeats")


def test_drill_a_status_record_that_is_not_json_is_refused(tmp_path):
    root = copy_root(tmp_path)
    edit_bytes(root, "numerical_check_status", lambda d: b"\xff" + d)
    refused(root, "not a JSON record")
    edit_bytes(root, "numerical_check_status", lambda d: d[1:-3])
    refused(root, "not a JSON record")


STATUS_FAULTS: dict[str, tuple[Callable[[dict], None], str]] = {
    "unknown-status": (lambda r: r.update(status="REPRODUCED"), "is not a status of the D3 contract"),
    "failed-true": (
        lambda r: r["fresh_solve"].update(failed_rows=True),
        "failed_rows True is not an integer",
    ),
    "failed-float": (
        lambda r: r["fresh_solve"].update(failed_rows=float(r["fresh_solve"]["failed_rows"])),
        r"failed_rows \d+\.0 is not an integer",
    ),
    "failed-text": (
        lambda r: r["fresh_solve"].update(failed_rows=str(r["fresh_solve"]["failed_rows"])),
        r"failed_rows '\d+' is not an integer",
    ),
    "failed-null": (
        lambda r: r["fresh_solve"].update(failed_rows=None),
        "failed_rows None is not an integer",
    ),
    "failed-negative": (lambda r: r["fresh_solve"].update(failed_rows=-1), "failed_rows -1 is not the"),
    "failed-beyond-rows": (
        lambda r: r["fresh_solve"]["population"].update(rows=r["fresh_solve"]["failed_rows"] - 1, cases=1),
        "is not the .* failures within",
    ),
    "failed-not-failures": (
        lambda r: r["fresh_solve"].update(failed_rows=r["fresh_solve"]["failed_rows"] - 1),
        "is not the .* failures within",
    ),
    "no-rows": (lambda r: r["fresh_solve"]["population"].update(rows=0), "is empty or inconsistent"),
    "no-cases": (lambda r: r["fresh_solve"]["population"].update(cases=0), "is empty or inconsistent"),
    "cases-beyond-rows": (
        lambda r: r["fresh_solve"]["population"].update(cases=r["fresh_solve"]["population"]["rows"] + 1),
        "is empty or inconsistent",
    ),
    "rows-bool": (lambda r: r["fresh_solve"]["population"].update(rows=True), "population.rows True is not"),
    "cases-text": (
        lambda r: r["fresh_solve"]["population"].update(cases=str(r["fresh_solve"]["population"]["cases"])),
        r"population.cases '\d+' is not",
    ),
    "failures-not-list": (lambda r: r["fresh_solve"].update(failures={}), "failures is not a list"),
    "failing-status-without-failures": (
        lambda r: r["fresh_solve"].update(failed_rows=0, failures=[]),
        "records no failed row",
    ),
    "missing-status": (lambda r: r.pop("status"), "a field the section reads is missing"),
    "missing-population": (
        lambda r: r["fresh_solve"].pop("population"),
        "a field the section reads is missing",
    ),
    "population-not-object": (
        lambda r: r["fresh_solve"].update(population=[]),
        "a field the section reads is missing",
    ),
}


@pytest.mark.parametrize("fault", list(STATUS_FAULTS))
def test_drill_a_status_record_the_section_does_not_admit_is_refused(tmp_path, fault):
    change, pattern = STATUS_FAULTS[fault]
    root = copy_root(tmp_path)
    edit_status(root, change)
    refused(root, pattern)


def test_the_status_editor_keeps_an_unchanged_record_admitted(tmp_path):
    root = copy_root(tmp_path)
    edit_bytes(root, "numerical_check_status", lambda d: d + b"\n")
    assert gen.stage_inputs(root) == committed_stages()


# --------------------------------------------------------------------------------------------
# V1 -- drift of the owning modules
# --------------------------------------------------------------------------------------------
OWNERS = {
    "mean_life_comparison": (lifetime, "COLUMNS"),
    "record_classes": (mixture, "CLASS_COLUMNS"),
    "selector_summary": (d2, "SUMMARY_COLUMNS"),
    "shell_projection": (d3c, "PROJECTION_COLUMNS"),
    "incompatible_groups": (d3c, "GROUPS_COLUMNS"),
}


@pytest.mark.parametrize("change", ["extra", "renamed"])
@pytest.mark.parametrize("name", CSV_NAMES)
def test_drill_a_table_whose_module_changed_its_columns_is_refused(monkeypatch, name, change):
    """A redesigned table stops the section until its wording is re-read: an extra column in, or a renamed
    field of, the owning module's tuple is refused before the file is read."""
    module, attribute = OWNERS[name]
    columns = getattr(module, attribute)
    assert columns == gen.STAGE_COLUMNS[name]
    changed = (*columns, "extra") if change == "extra" else (columns[0] + "_renamed", *columns[1:])
    monkeypatch.setattr(module, attribute, changed)
    refused(REPO, re.escape(gen.STAGE_FILES[name]) + ": its module's columns are not")


def test_drill_a_status_the_contract_adds_with_no_sentence_is_refused(monkeypatch):
    monkeypatch.setattr(d3c, "NUMERICAL_CHECK_STATUSES", (*d3c.NUMERICAL_CHECK_STATUSES, "REPRODUCED"))
    refused(REPO, "the statuses of the D3 contract and the sentences of STATUS_TEXT differ")


# --------------------------------------------------------------------------------------------
# the order of compare: a refused input leaves every file untouched
# --------------------------------------------------------------------------------------------
def test_drill_compare_refuses_a_tampered_input_before_it_writes_any_file(tmp_path, monkeypatch):
    root = copy_root(tmp_path)
    set_cell(
        root, "shell_projection", "consumer_within", "maybe", lambda row: row["consumer_within"] == "false"
    )
    data_dir = tsm.DATA
    before = {p.name: p.read_bytes() for p in sorted(data_dir.iterdir()) if p.is_file()}
    writes = []
    monkeypatch.setattr(gen, "ROOT", root)
    monkeypatch.setattr(gen, "write_csv", lambda path, fields, rows: writes.append(path))
    document = tmp_path / "BENCHMARK_STOPPED_MU.md"
    monkeypatch.setattr(gen, "DOCUMENT", document)
    with pytest.raises(gen.StageError, match="consumer_within 'maybe' is not true or false"):
        gen.compare()
    assert writes == [] and not document.exists()
    assert {p.name: p.read_bytes() for p in sorted(data_dir.iterdir()) if p.is_file()} == before


# --------------------------------------------------------------------------------------------
# V4 -- the section reads no file of the transported natural-target comparison
# --------------------------------------------------------------------------------------------
_OPENED: list[str] | None = None


def _record_open(event: str, args: tuple) -> None:
    if _OPENED is not None and event == "open" and isinstance(args[0], (str, bytes, Path)):
        _OPENED.append(str(args[0]))


sys.addaudithook(_record_open)


def opened_by(call: Callable[[], object]) -> list[str]:
    """The paths ``call`` opened, recorded by an audit hook that is inert outside this function."""
    global _OPENED
    _OPENED = []
    try:
        call()
        return list(_OPENED)
    finally:
        _OPENED = None


def transport_files(paths: list[str]) -> list[str]:
    names = [Path(path).name for path in paths]
    return [n for n in names if n.startswith("natural_mixture") or n == "natural_target_answers.csv"]


def test_the_section_opens_its_six_files_and_no_transport_file():
    paths = opened_by(lambda: gen.stage_inputs(REPO))
    assert transport_files(paths) == []
    under_data = {Path(p).resolve().relative_to(REPO).as_posix() for p in paths if "data" in Path(p).parts}
    assert under_data == set(gen.STAGE_FILES.values())


def test_the_document_render_opens_no_transport_file():
    paths = opened_by(lambda: gen.render(gen.load_inputs()))
    assert paths and transport_files(paths) == []


def test_drill_a_section_that_reads_a_transport_file_is_caught(monkeypatch):
    original = gen.stage_inputs

    def reading(root: Path):
        (root / "data/g4/d1/natural_mixture_comparison.csv").read_bytes()
        return original(root)

    monkeypatch.setattr(gen, "stage_inputs", reading)
    assert transport_files(opened_by(lambda: gen.stage_inputs(REPO))) == ["natural_mixture_comparison.csv"]


# --------------------------------------------------------------------------------------------
# V5 -- the sentence on the routes, for every pattern
# --------------------------------------------------------------------------------------------
BOUND, HELPER = (name for _, name in lifetime.ROUTES)
ROUTE_PATTERNS = {
    "every-route-everywhere": (
        (("a", 3, 1, 2), ("b", 2, 0, 1)),
        "Under `a` and `b` each route leaves a compared row outside the band.",
    ),
    "no-route-anywhere": (
        (("a", 3, 3, 3), ("b", 1, 1, 1)),
        "Under `a` and `b` no route leaves a compared row outside the band.",
    ),
    "mixed": (
        (("p", 5, 4, 4), ("q", 5, 5, 4), ("r", 5, 4, 5), ("s", 5, 5, 5), ("t", 5, 3, 3)),
        f"Under `p` and `t` each route leaves a compared row outside the band; under `q` only `{HELPER}` "
        f"leaves a compared row outside the band; under `r` only `{BOUND}` leaves a compared row outside the "
        "band; under `s` no route leaves a compared row outside the band.",
    ),
}


@pytest.mark.parametrize("pattern", list(ROUTE_PATTERNS))
def test_the_route_sentence_is_generated_for_every_pattern(pattern):
    summary, sentence = ROUTE_PATTERNS[pattern]
    text = section(dataclasses.replace(committed_stages(), summary=summary))
    assert sentence in text.splitlines()
    assert gen.forbidden_hits(text) == []


def test_the_committed_route_sentence_follows_the_summary():
    """Each profile is named in the clause its own counts select, in the comparison's order."""
    line = next(line for line in section().splitlines() if line.startswith("Under "))
    for profile, compared, *within in committed_stages().summary:
        outside = [name for (_, name), count in zip(lifetime.ROUTES, within, strict=True) if count < compared]
        clause = next(c for c in line.rstrip(".").split("; ") if f"`{profile}`" in c)
        if len(outside) == len(lifetime.ROUTES):
            assert clause.endswith("each route leaves a compared row outside the band"), clause
        elif outside:
            assert clause.endswith(f"only `{outside[0]}` leaves a compared row outside the band"), clause
        else:
            assert clause.endswith("no route leaves a compared row outside the band"), clause


# --------------------------------------------------------------------------------------------
# V7 -- a moved number moves the section
# --------------------------------------------------------------------------------------------
def test_a_flipped_shell_difference_moves_the_section_with_the_d3_recount(tmp_path):
    root = copy_root(tmp_path)
    locator = next(
        r for r in tsm.read_csv(path_of(REPO, "shell_projection")) if r["consumer_within"] == "false"
    )
    set_cell(
        root,
        "shell_projection",
        "consumer_within",
        "true",
        lambda row: all(row[c] == locator[c] for c in gen.STAGE_KEYS["shell_projection"]),
    )
    moved = gen.stage_inputs(root)
    recount = sum(r["consumer_within"] == "true" for r in tsm.read_csv(path_of(root, "shell_projection")))
    assert moved.consumer_within == committed_stages().consumer_within + 1 == recount
    assert section(moved) != section() and section(moved) not in DOCUMENT.read_text(encoding="utf-8")


def test_a_changed_selector_ratio_moves_the_section(tmp_path):
    root = copy_root(tmp_path)
    set_cell(root, "selector_summary", "ratio_median", "9.99", lambda row: row["scope"] == "method_gated")
    moved = gen.stage_inputs(root)
    assert moved.ratio_median == "9.99" != committed_stages().ratio_median
    assert "at 9.99 times it at the median" in section(moved)
    assert section(moved) not in DOCUMENT.read_text(encoding="utf-8")


def test_a_moved_route_count_moves_the_table_and_the_sentence():
    profile, compared, first, second = committed_stages().summary[-1]
    moved = dataclasses.replace(
        committed_stages(), summary=(*committed_stages().summary[:-1], (profile, compared, first, compared))
    )
    text = section(moved)
    assert f"| `{profile}` | {compared} | {first} | {compared} |" in text and text != section()
    assert section(moved) not in DOCUMENT.read_text(encoding="utf-8")


def test_a_zero_count_and_no_hydrogenous_row_render_truthfully():
    stages = dataclasses.replace(committed_stages(), groups_empty=0, selector_hydrogenous=0, mixed=())
    text = section(stages)
    assert "intersect in each of the" in text and "empty intersection" not in text
    assert "holds hydrogen" not in text and "a reconstructed combination" not in text
