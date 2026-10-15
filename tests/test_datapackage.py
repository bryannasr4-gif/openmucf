"""FAIR datapackage.json: valid JSON, resource paths exist, and each resource's field list
matches the live CSV header. Hand-written descriptor -- no frictionless dependency."""

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATAPACKAGE = REPO / "datapackage.json"


def _load():
    return json.loads(DATAPACKAGE.read_text(encoding="utf-8"))


def test_datapackage_is_valid_json_and_cc_by():
    dp = _load()
    assert dp["name"] == "openmucf-rate-ledger"
    assert dp["licenses"][0]["name"] == "CC-BY-4.0"


def test_datapackage_vocabulary_is_compilation_not_evaluated_library():
    """Vocabulary discipline: 'curated compilation with provenance', never 'evaluated library'."""
    blob = json.dumps(_load()).lower()
    assert "curated compilation with provenance" in blob
    assert "evaluated library" not in blob


def test_datapackage_resource_paths_exist():
    for res in _load()["resources"]:
        assert (REPO / res["path"]).is_file(), f"missing resource file: {res['path']}"


def _key_repeats(res, rows):
    pk = res["schema"]["primaryKey"]
    seen = [r[pk] for r in rows]
    return sorted({k for k in seen if seen.count(k) > 1})


def _keyed_resources():
    return [res for res in _load()["resources"] if res["schema"].get("primaryKey")]


def test_datapackage_primary_keys_are_unique():
    """No value of a resource's primaryKey is on two rows: a reader that keys a table by it keeps one of those
    rows and drops the rest unread, unless it refuses the repeat as openmucf.rates.load_rates,
    openmucf.validate._load_targets and openmucf.sidechannels.load_inputs do."""
    keyed = _keyed_resources()
    assert keyed
    for res in keyed:
        with open(REPO / res["path"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, res["name"]
        assert _key_repeats(res, rows) == [], res["name"]


def test_a_repeated_primary_key_is_named():
    """Drill: each keyed resource with its first row stated twice is named."""
    for res in _keyed_resources():
        with open(REPO / res["path"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert _key_repeats(res, [*rows, rows[0]]) == [rows[0][res["schema"]["primaryKey"]]], res["name"]


def test_datapackage_fields_match_live_csv_headers():
    """Every resource schema's field names, in order, equal the live CSV header row."""
    for res in _load()["resources"]:
        with open(REPO / res["path"], newline="", encoding="utf-8") as f:
            headers = next(csv.reader(f))
        field_names = [fld["name"] for fld in res["schema"]["fields"]]
        assert field_names == headers, (
            f"{res['name']}: datapackage fields {field_names} != live CSV headers {headers}"
        )


def test_datapackage_primary_keys_are_real_columns():
    for res in _load()["resources"]:
        pk = res["schema"].get("primaryKey")
        if pk:
            names = {fld["name"] for fld in res["schema"]["fields"]}
            assert pk in names, f"{res['name']}: primaryKey {pk!r} not among fields"
