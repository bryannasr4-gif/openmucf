"""The ``G4MuonicData`` format: Layer-1 grammar and Layer-2 provenance (see ``FORMAT_SPEC.md``).

One test per rule, in the order the document states them. The sixteen error tests each assert the
exact code *and* the exact 1-based line number, because a validator that rejects the right file for
the wrong reason -- or that cannot say where -- is not usable by whoever has to fix the file.

Line numbers below are literal on purpose: ``CANONICAL`` has a fixed 17-line layout (13 directives,
3 records, ``#END``), so a literal is auditable by reading the document and a shift in the layout
fails loudly instead of being absorbed by a helper that recomputes it.
"""

import dataclasses
import json
import locale
import math
import random
import re
import struct
import sys

import pytest

from openmucf.g4 import provenance, spec
from openmucf.g4.spec import G4DatFormatError, G4DatTable

DIRECTIVES = {
    "GRAMMAR": "1.0",  # line 1
    "DATASET": "G4MuonicData",  # line 2
    "VERSION": "1.0.0",  # line 3
    "PROFILE": "parity",  # line 4
    "SEAM": "d1_nuclear_capture",  # line 5
    "TABLE": "nuclear_capture_rate",  # line 6
    "GENERATOR": "openmucf-g4 1.1.0",  # line 7
    "SOURCEDIGEST": "0" * 64,  # line 8
    "SOURCESHA": "8cc04f65977807f1848da7b958c421cd5e162f26",  # line 9
    "UNITS": "rate=1e6/s",  # line 10
    "COLUMNS": "Z A value unc",  # line 11
    "VALIDITY": "Z:1-94 A:natural_and_listed",  # line 12
    "FALLBACK": "goulard_primakoff b0a=-0.03 b0b=-0.25 b0c=3.24 t1=875e-9",  # line 13
}
RECORDS = ((1, 1, 0.000725, 1.7e-05), (29, 63, 5.676, 0.041), (94, 242, 12.86, 0.19))  # lines 14-16
END_LINE = 17


def make_table(records=RECORDS, **overrides) -> G4DatTable:
    """The canonical table, with directive overrides; ``key=None`` drops a directive."""
    directives = dict(DIRECTIVES)
    for key, value in overrides.items():
        if value is None:
            directives.pop(key, None)
        else:
            directives[key] = value
    return G4DatTable(directives=directives, records=tuple(records))


CANONICAL = spec.render(make_table())


def drop_line(text: str, prefix: str) -> str:
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith(prefix))


def replace_line(text: str, prefix: str, replacement: str) -> str:
    return "".join(
        replacement + "\n" if line.startswith(prefix) else line
        for line in text.splitlines(keepends=True)
    )


def rejected(text: str) -> G4DatFormatError:
    with pytest.raises(G4DatFormatError) as caught:
        spec.parse(text)
    return caught.value


def rejected_table(table: G4DatTable) -> G4DatFormatError:
    with pytest.raises(G4DatFormatError) as caught:
        spec.validate(table)
    return caught.value


def digest_rejected(table: G4DatTable, layer2: bytes) -> G4DatFormatError:
    with pytest.raises(G4DatFormatError) as caught:
        provenance.check_source_digest(table, layer2)
    return caught.value


ROW = {
    "source_bibkey": "suzuki1987",
    "source_locator": "Table III",
    "unc_type": "exp",
    "conditions": "muonic atom, ground state",
    "validity_range": "Z 1-94",
    "evaluation_method": "measured total capture rate",
    "single_source": True,
    "needs_verification": False,
    "recommendation": "recommended",
    "evaluation_id": "suzuki1987-tableIII",
    "source_library": "suzuki1987",
    "isotope_resolved": True,
}


def make_document(rows=None, **overrides) -> provenance.ProvDocument:
    """The canonical Layer-2 document: one row per record of the canonical table."""
    if rows is None:
        rows = {
            f"{z}-{a}": provenance.ProvRow(**{**ROW, "evaluation_id": f"suzuki1987-{z}-{a}"})
            for z, a, *_ in RECORDS
        }
    document = provenance.ProvDocument(
        dataset=DIRECTIVES["DATASET"],
        version=DIRECTIVES["VERSION"],
        profile=DIRECTIVES["PROFILE"],
        seam=DIRECTIVES["SEAM"],
        precedence=("iwamoto2025", "suzuki1987", "geant4-compiled-in"),
        rows=rows,
    )
    return dataclasses.replace(document, **overrides)


# --------------------------------------------------------------------------------------------
# T-01..T-16 -- one per error code, code and line both asserted
# --------------------------------------------------------------------------------------------


def test_t01_e001_unknown_directive():
    """A `#` keyword we do not know is a hard error, not a skipped comment."""
    error = rejected(replace_line(CANONICAL, "#FALLBACK", "#COMMENT      not a comment field"))
    assert (error.code, error.line) == ("E001", 13)
    assert str(error) == "E001: unknown directive '#COMMENT' (line 13)"
    lowercase = rejected(replace_line(CANONICAL, "#UNITS", "#units        rate=1e6/s"))
    assert (lowercase.code, lowercase.line) == ("E001", 10)


def test_t02_e002_missing_required_directive():
    """A missing required directive is reported where the directive block ended."""
    error = rejected(drop_line(CANONICAL, "#VALIDITY"))
    assert (error.code, error.line) == ("E002", 13)  # 12 directives left, first record on line 13
    assert "'#VALIDITY'" in str(error)


def test_t03_e003_directive_out_of_order():
    version, profile = "#VERSION      1.0.0\n", "#PROFILE      parity\n"
    error = rejected(CANONICAL.replace(version + profile, profile + version))
    assert (error.code, error.line) == ("E003", 4)
    units = "#UNITS        rate=1e6/s\n"
    repeated = rejected(CANONICAL.replace(units, units + units))
    assert (repeated.code, repeated.line) == ("E003", 11)
    trailing = rejected(CANONICAL.replace("#END", "#UNITS        rate=1e6/s\n#END"))
    assert (trailing.code, trailing.line) == ("E003", 17)


def test_t04_e004_record_field_count():
    error = rejected(replace_line(CANONICAL, " 1", " 1   1 0.000725"))
    assert (error.code, error.line) == ("E004", 14)
    assert "3 field(s)" in str(error)
    blank = rejected(replace_line(CANONICAL, "29", ""))
    assert (blank.code, blank.line) == ("E004", 15)


def test_t05_e005_non_ascii_byte():
    error = rejected(CANONICAL.replace("rate=1e6/s", "rate=1e6/µs"))
    assert (error.code, error.line) == ("E005", 10)


def test_t06_e006_cr_line_ending():
    error = rejected(CANONICAL.replace("#UNITS        rate=1e6/s\n", "#UNITS        rate=1e6/s\r\n"))
    assert (error.code, error.line) == ("E006", 10)
    lone_cr = rejected(CANONICAL.replace("#UNITS        rate=1e6/s\n", "#UNITS        rate=1e6/s\r"))
    assert (lone_cr.code, lone_cr.line) == ("E006", 10)


def test_t07_e007_unparsable_float():
    """A comma decimal separator is a syntax error, never a silent truncation."""
    error = rejected(CANONICAL.replace("0.00072499999999999995", "0,00072499999999999995"))
    assert (error.code, error.line) == ("E007", 14)
    negative_z = rejected(replace_line(CANONICAL, " 1", "-1   1 0.000725 1.7e-05"))
    assert (negative_z.code, negative_z.line) == ("E007", 14)


def test_t08_e008_duplicate_key():
    """The duplicate is reported where it appears, naming the line the key was first seen on."""
    error = rejected(replace_line(CANONICAL, "94", " 1   1 12.86 0.19"))
    assert (error.code, error.line) == ("E008", 16)
    assert "first seen at line 14" in str(error)


def test_t09_e009_source_digest_mismatch():
    """The cross-layer code: the digest is checked against the Layer-2 file's bytes, not a copy."""
    error = digest_rejected(make_table(), provenance.document_bytes(make_document()))
    assert (error.code, error.line) == ("E009", 8)
    assert provenance.source_digest(make_document()) in str(error)
    missing = digest_rejected(make_table(SOURCEDIGEST=None), b"{}\n")
    assert (missing.code, missing.line) == ("E002", 8)


def test_t10_e010_unsupported_grammar_major():
    error = rejected(replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR      2.0"))
    assert (error.code, error.line) == ("E010", 1)
    unreadable = rejected(replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR      one"))
    assert (unreadable.code, unreadable.line) == ("E010", 1)
    assert spec.parse(replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR      1.7")).directives["GRAMMAR"] == "1.7"


def test_t11_e011_content_after_end():
    error = rejected(CANONICAL + "95 243 1.0 0.1\n")
    assert (error.code, error.line) == ("E011", END_LINE + 1)
    blank_after = rejected(CANONICAL + "\n")
    assert (blank_after.code, blank_after.line) == ("E011", END_LINE + 1)


def test_t12_e012_missing_end():
    error = rejected(drop_line(CANONICAL, "#END"))
    assert (error.code, error.line) == ("E012", END_LINE - 1)
    unterminated = rejected(CANONICAL.rstrip("\n"))
    assert (unterminated.code, unterminated.line) == ("E012", END_LINE)
    assert "newline" in str(unterminated)


def test_t13_e013_parity_without_sourcesha():
    """Reported on the `#PROFILE` line: that is the line that created the requirement."""
    error = rejected(drop_line(CANONICAL, "#SOURCESHA"))
    assert (error.code, error.line) == ("E013", 4)


def test_t14_e014_non_finite_float():
    error = rejected(replace_line(CANONICAL, "29", "29  63 nan 0.041"))
    assert (error.code, error.line) == ("E014", 15)
    infinity = rejected(replace_line(CANONICAL, "29", "29  63 -INF 0.041"))
    assert (infinity.code, infinity.line) == ("E014", 15)
    overflow = rejected(replace_line(CANONICAL, "29", "29  63 1e400 0.041"))
    assert (overflow.code, overflow.line) == ("E014", 15)


def test_t15_e015_records_not_sorted():
    swapped = CANONICAL.splitlines(keepends=True)
    swapped[13], swapped[15] = swapped[15], swapped[13]
    error = rejected("".join(swapped))
    assert (error.code, error.line) == ("E015", 15)


def test_t16_e016_profile_or_seam_outside_allowed_set():
    seam = rejected(replace_line(CANONICAL, "#SEAM", "#SEAM         d9_not_a_seam"))
    assert (seam.code, seam.line) == ("E016", 5)
    profile = rejected(replace_line(CANONICAL, "#PROFILE", "#PROFILE      Parity"))
    assert (profile.code, profile.line) == ("E016", 4)


# --------------------------------------------------------------------------------------------
# T-17..T-21 -- round-trip, float exactness, locale, determinism, ordering
# --------------------------------------------------------------------------------------------


def _random_double(rng: random.Random) -> float:
    """A finite double drawn from the whole bit space -- subnormals, huge exponents, signed zeros."""
    while True:
        value = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if math.isfinite(value):
            return value


def _draw_float(rng: random.Random) -> float:
    style = rng.randrange(5)
    if style == 0:
        return rng.uniform(-1.0e3, 1.0e3)
    if style == 1:
        return rng.uniform(0.0, 1.0) * 10.0 ** rng.randint(-300, 300)
    if style == 2:
        return float(rng.randint(-10**6, 10**6))
    if style == 3:
        return -0.0 if rng.random() < 0.5 else 0.0
    return _random_double(rng)


def seeded_tables(count: int, seed: int = 20260810):
    """A deterministic pseudo-random stream of *valid* tables -- no new test dependency."""
    rng = random.Random(seed)
    profiles = ["parity", "evaluated", "iwamoto2025", "jendl-mund", "abc", "z9_x-y"]
    for index in range(count):
        columns = ["Z", "A", "value", "unc"] + [f"extra{i}" for i in range(rng.randrange(3))]
        keys = sorted(rng.sample([(z, a) for z in range(1, 25) for a in range(1, 40)], rng.randrange(7)))
        records = tuple((z, a, *(_draw_float(rng) for _ in columns[2:])) for z, a in keys)
        profile = rng.choice(profiles)
        directives = {
            "GRAMMAR": "1.0",
            "DATASET": "G4MuonicData",
            "VERSION": f"{rng.randrange(3)}.{rng.randrange(10)}.{rng.randrange(10)}",
            "PROFILE": profile,
            "SEAM": rng.choice(spec.ALLOWED_SEAMS),
            "TABLE": rng.choice(["nuclear_capture_rate", "zeff", "k_shell_energy"]),
            "GENERATOR": f"openmucf-g4 {index}",
            "SOURCEDIGEST": f"{rng.getrandbits(256):064x}",
            "UNITS": rng.choice(["rate=1e6/s", "energy=keV", "probability=fraction"]),
            "COLUMNS": " ".join(columns),
            "VALIDITY": "Z:1-94 A:natural_and_listed",
        }
        if profile == spec.PARITY_PROFILE:
            directives["SOURCESHA"] = f"{rng.getrandbits(160):040x}"
        if rng.random() < 0.5:
            directives["FALLBACK"] = "goulard_primakoff b0a=-0.03 b0b=-0.25 b0c=3.24 t1=875e-9"
        yield G4DatTable(directives=directives, records=records)


def test_t17_round_trip_property():
    """For every table validate() accepts, parse(render(t)) == t -- 200 seeded tables."""
    count = 0
    for table in seeded_tables(200):
        assert spec.validate(table) is None
        text = spec.render(table)
        assert spec.parse(text) == table
        count += 1
    assert count == 200


def test_t18_format_float_is_exact():
    """float(format_float(x)) == x, including the extremes that break shorter formats."""
    fixed = [
        0.1,
        5e-324,  # smallest positive subnormal
        1.7976931348623157e308,  # largest finite double
        -0.0,
        0.0,
        1.0,
        2.2250738585072014e-308,  # smallest positive normal
        0.000725,
        12.86,
    ]
    rng = random.Random(18)
    for value in [*fixed, *(_random_double(rng) for _ in range(100))]:
        text = spec.format_float(value)
        assert float(text) == value, text
        assert math.copysign(1.0, float(text)) == math.copysign(1.0, value), text
    assert math.copysign(1.0, float(spec.format_float(-0.0))) == -1.0
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            spec.format_float(bad)


def test_t19_render_is_locale_independent():
    """Under a comma-decimal LC_NUMERIC the output must be byte-identical to the C-locale output.

    Unavailable locale: fail on Linux (CI installs de_DE.UTF-8, so a silent skip there would hide a
    regression on the platform that guarantees the locale), skip elsewhere with the reason.
    """
    reference = spec.render(make_table())
    reference_floats = [spec.format_float(x) for x in (0.1, 1e-300, -2.5e17, 0.000725)]
    previous = locale.setlocale(locale.LC_NUMERIC)
    chosen = None
    try:
        for candidate in ("de_DE.UTF-8", "de_DE.utf8", "de_DE", "German_Germany.1252", "de-DE"):
            try:
                locale.setlocale(locale.LC_NUMERIC, candidate)
            except locale.Error:
                continue
            chosen = candidate
            break
        if chosen is None:
            message = (
                "no comma-decimal locale available on this machine "
                "(tried de_DE.UTF-8, de_DE.utf8, de_DE, German_Germany.1252, de-DE)"
            )
            if sys.platform.startswith("linux"):
                pytest.fail(message)
            pytest.skip(message)
        assert locale.localeconv()["decimal_point"] == ","
        assert spec.render(make_table()) == reference
        assert [spec.format_float(x) for x in (0.1, 1e-300, -2.5e17, 0.000725)] == reference_floats
        assert spec.parse(reference) == make_table()
    finally:
        locale.setlocale(locale.LC_NUMERIC, previous)


def test_t20_render_is_deterministic_and_timestamp_free():
    table = make_table()
    first = spec.render(table)
    assert first == spec.render(table)
    shuffled = G4DatTable(
        directives=dict(reversed(list(table.directives.items()))), records=table.records
    )
    assert first == spec.render(shuffled)  # dict insertion order cannot leak into the bytes
    assert "#DATE" not in first and "#TIME" not in first
    without_sha = "".join(
        line for line in first.splitlines(keepends=True) if not line.startswith("#SOURCESHA")
    )
    assert re.search(r"\b(19|20)\d{2}\b", without_sha) is None


def test_t21_records_are_sorted_on_render():
    reversed_table = make_table(records=tuple(reversed(RECORDS)))
    text = spec.render(reversed_table)
    assert text == spec.render(make_table())
    keys = [tuple(int(field) for field in line.split()[:2]) for line in text.splitlines()[13:16]]
    assert keys == sorted(keys) == [(1, 1), (29, 63), (94, 242)]
    assert spec.parse(text) == make_table()
    unsorted_error = rejected_table(reversed_table)
    assert (unsorted_error.code, unsorted_error.line) == ("E015", 15)


# --------------------------------------------------------------------------------------------
# T-26..T-30 -- Layer 2: schema, the digest binding, precedence, and the isotope disclosure
# --------------------------------------------------------------------------------------------


def test_t26_layer2_schema_round_trip():
    document = make_document()
    assert provenance.from_json_obj(provenance.to_json_obj(document)) == document
    text = provenance.render_json(document)
    assert json.loads(text) == provenance.to_json_obj(document)
    assert text.endswith("\n") and text.isascii()
    assert provenance.document_bytes(document) == text.encode("ascii")
    assert provenance.render_json(document) == text  # canonical: sorted keys, fixed indent
    with pytest.raises(ValueError, match="unknown field"):
        provenance.from_json_obj({**provenance.to_json_obj(document), "extra": 1})


def test_t27_source_digest_matches():
    document = make_document()
    payload = provenance.document_bytes(document)
    table = make_table(SOURCEDIGEST=provenance.source_digest(payload))
    assert provenance.check_source_digest(table, payload) is None
    assert provenance.source_digest(document) == provenance.source_digest(payload)
    assert provenance.check_against_table(table, document) is None
    with pytest.raises(ValueError, match="does not match Layer-1"):
        provenance.check_against_table(make_table(VERSION="9.9.9"), document)


def test_t28_digest_drift_raises_e009():
    """Regenerate Layer 2 with one field changed and the two layers can no longer both be right."""
    document = make_document()
    table = make_table(SOURCEDIGEST=provenance.source_digest(provenance.document_bytes(document)))
    drifted = dataclasses.replace(
        document,
        rows={**document.rows, "1-1": dataclasses.replace(document.rows["1-1"], isotope_resolved=False)},
    )
    payload = provenance.document_bytes(drifted)
    assert payload != provenance.document_bytes(document)
    error = digest_rejected(table, payload)
    assert (error.code, error.line) == ("E009", 8)


def test_t29_precedence_is_an_ordered_list_of_known_libraries():
    document = make_document()
    assert provenance.to_json_obj(document)["precedence"] == list(document.precedence)
    reordered = dataclasses.replace(document, precedence=tuple(reversed(document.precedence)))
    assert provenance.render_json(reordered) != provenance.render_json(document)  # order is data
    for bad, match in (
        (["not-a-library"], "not one of"),
        (["suzuki1987", "suzuki1987"], "repeated"),
        ([], "at least one"),
        ("suzuki1987", "ordered list"),
    ):
        with pytest.raises(ValueError, match=match):
            provenance.validate_document({**provenance.to_json_obj(document), "precedence": bad})


def test_t30_isotope_resolved_required_on_every_row():
    """The disclosure cannot be omitted: a row without it is not a valid Layer-2 row."""
    obj = provenance.to_json_obj(make_document())
    stripped = {
        key: {k: v for k, v in row.items() if k != "isotope_resolved"} for key, row in obj["rows"].items()
    }
    with pytest.raises(ValueError, match="isotope_resolved"):
        provenance.validate_document({**obj, "rows": stripped})
    with pytest.raises(ValueError, match="must be a boolean"):
        provenance.validate_document(
            {**obj, "rows": {**obj["rows"], "1-1": {**obj["rows"]["1-1"], "isotope_resolved": "yes"}}}
        )
    assert all(row["isotope_resolved"] is True for row in obj["rows"].values())
