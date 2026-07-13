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
