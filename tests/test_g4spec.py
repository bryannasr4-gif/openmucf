"""The ``G4MuonicData`` format: Layer-1 grammar and Layer-2 provenance (see ``FORMAT_SPEC.md``).

One test per rule, in the order the document states them. The sixteen error tests each assert the
exact code *and* the exact 1-based line number, because a validator that rejects the right file for
the wrong reason -- or that cannot say where -- is not usable by whoever has to fix the file.

Line numbers below are literal on purpose: ``CANONICAL`` has a fixed 17-line layout (13 directives,
3 records, ``#END``), so a literal is auditable by reading the document and a shift in the layout
fails loudly instead of being absorbed by a helper that recomputes it.
"""

import ast
import dataclasses
import io
import json
import locale
import math
import pathlib
import random
import re
import struct
import subprocess
import sys
import tarfile
import tempfile
import tomllib

import pytest

from openmucf.g4 import emit, provenance, spec
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


def test_t05_e005_byte_outside_the_allowed_set():
    """E005 covers the whole byte set `{TAB, LF, CR, 0x20-0x7E}`, not just non-ASCII.

    A VT or FF is ASCII, so an `isascii()` check waves it through -- and then Python's `str.split()`
    treats it as a separator while a space/tab C++ reader treats it as one more character of the
    field. Same file, two different field counts. Rejecting the byte is what stops that.
    """
    error = rejected(CANONICAL.replace("rate=1e6/s", "rate=1e6/µs"))
    assert (error.code, error.line) == ("E005", 10)

    for control in ("\x0b", "\x0c", "\x00", "\x1f", "\x7f"):
        bad = rejected(replace_line(CANONICAL, " 1", f" 1{control}1 0.000725 1.7e-05"))
        assert (bad.code, bad.line) == ("E005", 14), control
        assert f"{ord(control):02X}" in str(bad), control

    # The disagreement the rule exists to prevent, stated as an assertion rather than a comment.
    separator = "\x0b"
    vt_separated = f"1{separator}1{separator}2.5"
    assert vt_separated.split() == ["1", "1", "2.5"]  # what str.split() would have made of it
    assert spec._split_fields(vt_separated) == [vt_separated]  # what a space/tab reader sees

    # A VT inside a directive value is ASCII and survives .strip(); validate() must reject it too, or
    # a table it accepts would render to a file parse() rejects, breaking the round-trip guarantee.
    in_directive = rejected_table(make_table(UNITS="rate=1e6\x0b/s"))
    assert (in_directive.code, in_directive.line) == ("E005", 10)


def test_t06_e006_cr_line_ending():
    error = rejected(CANONICAL.replace("#UNITS        rate=1e6/s\n", "#UNITS        rate=1e6/s\r\n"))
    assert (error.code, error.line) == ("E006", 10)
    lone_cr = rejected(CANONICAL.replace("#UNITS        rate=1e6/s\n", "#UNITS        rate=1e6/s\r"))
    assert (lone_cr.code, lone_cr.line) == ("E006", 10)


def test_t07_e007_field_outside_its_column_lexical_class():
    """A comma decimal separator is a syntax error, never a silent truncation."""
    error = rejected(CANONICAL.replace("0.00072499999999999995", "0,00072499999999999995"))
    assert (error.code, error.line) == ("E007", 14)
    negative_z = rejected(replace_line(CANONICAL, " 1", "-1   1 0.000725 1.7e-05"))
    assert (negative_z.code, negative_z.line) == ("E007", 14)

    # Integer columns are BOUNDED, so a C++ reader's integer width is not implementation-defined.
    assert (spec.INTEGER_MIN, spec.INTEGER_MAX) == (0, 9999)
    for field in ("10000", "99999", "18446744073709551616"):
        over = rejected(replace_line(CANONICAL, "94", f"{field} 242 12.86 0.19"))
        assert (over.code, over.line) == ("E007", 16), field
        assert "integer out of range" in str(over), field
    at_bound = spec.parse(replace_line(CANONICAL, "94", "9999 242 12.86 0.19"))
    assert at_bound.records[2][0] == 9999  # the bound itself is inclusive

    # Same rule on the in-memory path, where the value is an int rather than a field.
    in_memory = rejected_table(make_table(records=((1, 1, 0.5, 0.1), (10**5, 1, 0.5, 0.1))))
    assert (in_memory.code, in_memory.line) == ("E007", 15)
    assert "integer out of range" in str(in_memory)


def test_t08_e008_duplicate_key():
    """The duplicate is reported where it appears, naming the line the key was first seen on."""
    error = rejected(replace_line(CANONICAL, "94", " 1   1 12.86 0.19"))
    assert (error.code, error.line) == ("E008", 16)
    assert "first seen at line 14" in str(error)
    assert "duplicate key (Z=1, A=1)" in str(error)

    # The label comes from the columns the table DECLARES. An A-only table's duplicate is `A=5`; the
    # first implementation read the names off INTEGER_COLUMNS and called it `Z=5`, sending whoever
    # had to fix the file to a column that is not there.
    a_only = rejected_table(
        make_table(COLUMNS="A value unc", records=((5, 1.0, 0.1), (5, 2.0, 0.1)))
    )
    assert (a_only.code, a_only.line) == ("E008", 14 + 1)
    assert "duplicate key (A=5)" in str(a_only)
    z_only = rejected_table(
        make_table(COLUMNS="Z value unc", records=((5, 1.0, 0.1), (5, 2.0, 0.1)))
    )
    assert "duplicate key (Z=5)" in str(z_only)


def test_t09_e009_source_digest_mismatch():
    """The cross-layer code: the digest is checked against the Layer-2 file's bytes, not a copy."""
    error = digest_rejected(make_table(), provenance.document_bytes(make_document()))
    assert (error.code, error.line) == ("E009", 8)
    assert provenance.source_digest(make_document()) in str(error)
    missing = digest_rejected(make_table(SOURCEDIGEST=None), b"{}\n")
    assert (missing.code, missing.line) == ("E002", 8)


def test_t10_e010_unsupported_grammar_major():
    """E010 is raised EAGERLY, at the `#GRAMMAR` line, and therefore preempts every later defect.

    Every other diagnosis is only meaningful under a grammar this reader implements, so reporting
    one of them first is a true statement about the wrong problem. The two preemption cases below
    are the ones that were actually wrong before: an out-of-order directive further down reported
    E003, and a file written to a grammar we do not know reported its new directive as "unknown".
    """
    error = rejected(replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR      2.0"))
    assert (error.code, error.line) == ("E010", 1)
    unreadable = rejected(replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR      one"))
    assert (unreadable.code, unreadable.line) == ("E010", 1)
    assert spec.parse(replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR      1.7")).directives["GRAMMAR"] == "1.7"

    validity = "#VALIDITY     Z:1-94 A:natural_and_listed\n"
    fallback = "#FALLBACK     goulard_primakoff b0a=-0.03 b0b=-0.25 b0c=3.24 t1=875e-9\n"
    out_of_order = replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR      9.0").replace(
        validity + fallback, fallback + validity
    )
    assert rejected(out_of_order.replace("#GRAMMAR      9.0", "#GRAMMAR      1.0")).code == "E003"
    preempted = rejected(out_of_order)
    assert (preempted.code, preempted.line) == ("E010", 1)  # not E003 on the swapped line

    future = replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR      2.0").replace(
        "#DATASET", "#NEWTHING     x\n#DATASET"
    )
    assert rejected(future.replace("#GRAMMAR      2.0", "#GRAMMAR      1.0")).code == "E001"
    forward = rejected(future)
    assert (forward.code, forward.line) == ("E010", 1)  # not "unknown directive '#NEWTHING'"


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

    # An EMPTY value counts as absent. `#SOURCESHA` with nothing after it is a parity file claiming
    # to reproduce nothing -- the exact claim E013 exists to stop -- and taking the key's presence
    # as satisfaction let it validate clean.
    empty = rejected(replace_line(CANONICAL, "#SOURCESHA", "#SOURCESHA"))
    assert (empty.code, empty.line) == ("E013", 4)
    in_memory = rejected_table(make_table(SOURCESHA=""))
    assert (in_memory.code, in_memory.line) == ("E013", 4)


class _UnderflowingInt(int):
    """An int whose ``__float__`` disagrees with its value -- the only witness that reaches the
    in-memory half of the underflow rule, since no plain int or float can."""

    def __float__(self) -> float:
        return 0.0


def test_t14_e014_non_finite_float():
    error = rejected(replace_line(CANONICAL, "29", "29  63 nan 0.041"))
    assert (error.code, error.line) == ("E014", 15)
    infinity = rejected(replace_line(CANONICAL, "29", "29  63 -INF 0.041"))
    assert (infinity.code, infinity.line) == ("E014", 15)
    overflow = rejected(replace_line(CANONICAL, "29", "29  63 1e400 0.041"))
    assert (overflow.code, overflow.line) == ("E014", 15)

    # UNDERFLOW is the same condition seen from the other end. Python's float("1e-999") is 0.0,
    # silently; C++'s std::from_chars reports result_out_of_range for the same text. Accepting it
    # would mean a conforming C++ reader and this reference implementation read one file differently.
    underflow = rejected(replace_line(CANONICAL, "29", "29  63 1e-999 0.041"))
    assert (underflow.code, underflow.line) == ("E014", 15)
    assert "underflows to zero" in str(underflow)
    for lexically_zero in ("0", "0.0", "-0.0", "0e-999", "0.000e-999"):
        table = spec.parse(replace_line(CANONICAL, "29", f"29  63 {lexically_zero} 0.041"))
        assert table.records[1][2] == 0.0, lexically_zero  # a genuine zero is not an underflow
    assert spec.parse(  # nor is the smallest positive subnormal, which round-trips exactly
        replace_line(CANONICAL, "29", "29  63 4.9406564584124654e-324 0.041")
    ).records[1][2] == 5e-324

    # The in-memory path. A Python int too large to convert used to escape as a raw OverflowError --
    # a stack trace where the format promises a coded, located rejection -- on validate() and
    # render() alike. Unreachable from any file, so only a hand-built table finds it.
    huge = make_table(records=((1, 1, 10**400, 0.1),))
    from_validate = rejected_table(huge)
    assert (from_validate.code, from_validate.line) == ("E014", 14)
    assert "overflows to infinity" in str(from_validate)
    with pytest.raises(G4DatFormatError) as from_render:
        spec.render(huge)
    assert (from_render.value.code, from_render.value.line) == ("E014", 14)

    tiny = rejected_table(make_table(records=((1, 1, _UnderflowingInt(3), 0.1),)))
    assert (tiny.code, tiny.line) == ("E014", 14)
    assert "underflows to zero" in str(tiny)


def test_t15_e015_records_not_sorted():
    swapped = CANONICAL.splitlines(keepends=True)
    swapped[13], swapped[15] = swapped[15], swapped[13]
    error = rejected("".join(swapped))
    assert (error.code, error.line) == ("E015", 15)
    assert "ascending Z/A order" in str(error)

    # As with E008, the ordering message names the declared key columns, not INTEGER_COLUMNS.
    a_only = rejected_table(
        make_table(COLUMNS="A value unc", records=((7, 1.0, 0.1), (5, 2.0, 0.1)))
    )
    assert (a_only.code, a_only.line) == ("E015", 15)
    assert "record (A=5) is not in ascending A order" in str(a_only)
    z_only = rejected_table(
        make_table(COLUMNS="Z value unc", records=((7, 1.0, 0.1), (5, 2.0, 0.1)))
    )
    assert "ascending Z order" in str(z_only)


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
# T-22..T-25 -- profiles and seams
# --------------------------------------------------------------------------------------------


def test_t22_profile_allowed_set_accepted():
    """The profile token set is open on purpose: N competing evaluations must be able to coexist."""
    for profile in ("parity", "evaluated", "iwamoto2025", "jendl-mund", "abc", "a" * 32, "z9_x-y"):
        table = make_table(PROFILE=profile, SOURCESHA=None)
        if profile == spec.PARITY_PROFILE:
            table = make_table(PROFILE=profile)
        assert spec.validate(table) is None
        assert spec.parse(spec.render(table)).directives["PROFILE"] == profile


def test_t23_profile_outside_allowed_set_rejected():
    for profile in ("Parity", "ab", "a" * 33, "9lives", "has space", "trailing!", ""):
        error = rejected_table(make_table(PROFILE=profile, SOURCESHA=None))
        assert (error.code, error.line) == ("E016", 4), profile


def test_t24_seam_allowed_set():
    for seam in spec.ALLOWED_SEAMS:
        assert spec.validate(make_table(SEAM=seam)) is None
    assert spec.ALLOWED_SEAMS == (
        "d1_nuclear_capture",
        "d2_atomic_capture",
        "d3_transitions",
        "d4_mucf_cycle",
    )
    for seam in ("d5_unknown", "D1_NUCLEAR_CAPTURE", "nuclear_capture", ""):
        error = rejected_table(make_table(SEAM=seam))
        assert (error.code, error.line) == ("E016", 5), seam


def test_t25_sourcesha_required_iff_parity():
    """A parity file must name the revision it reproduces; a file reproducing nothing must not."""
    assert spec.validate(make_table(PROFILE="parity")) is None
    assert spec.validate(make_table(PROFILE="evaluated", SOURCESHA=None)) is None

    missing = rejected_table(make_table(PROFILE="parity", SOURCESHA=None))
    assert (missing.code, missing.line) == ("E013", 4)
    unexpected = rejected_table(make_table(PROFILE="evaluated"))
    assert (unexpected.code, unexpected.line) == ("E013", 9)
    assert "'#SOURCESHA'" in str(unexpected)

    # An empty value is absent for BOTH directions of the iff: it cannot satisfy parity, and it
    # cannot violate the non-parity half either -- there is no revision being claimed.
    assert spec.validate(make_table(PROFILE="evaluated", SOURCESHA="")) is None
    empty_parity = rejected_table(make_table(PROFILE="parity", SOURCESHA=""))
    assert (empty_parity.code, empty_parity.line) == ("E013", 4)
    assert spec.parse(spec.render(make_table(PROFILE="evaluated", SOURCESHA=""))) == make_table(
        PROFILE="evaluated", SOURCESHA=""
    )  # an empty directive still round-trips


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


# --------------------------------------------------------------------------------------------
# T-31..T-34 -- hand-written malformed files on disk, and the import fence
# --------------------------------------------------------------------------------------------

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "g4dat_bad"


def fixture_bytes(name: str) -> bytes:
    """Read the fixture as bytes. Text mode would translate CRLF into LF and hide the very defect
    one of these files exists to carry."""
    return (FIXTURES / name).read_bytes()


def test_t31_fixture_crlf_line_endings():
    raw = fixture_bytes("crlf_line_endings.g4dat")
    assert b"\r\n" in raw, "the checkout stripped the CR this fixture is made of"
    error = rejected(raw.decode("ascii"))
    assert (error.code, error.line) == ("E006", 1)


def test_t32_fixture_comma_decimal():
    raw = fixture_bytes("comma_decimal.g4dat")
    assert b"\r" not in raw, "the checkout rewrote this LF fixture and it now fails as E006"
    assert b"2,5" in raw
    error = rejected(raw.decode("ascii"))
    assert (error.code, error.line) == ("E007", 13)


def test_t33_fixture_stray_comment():
    raw = fixture_bytes("stray_comment.g4dat")
    assert b"\r" not in raw, "the checkout rewrote this LF fixture and it now fails as E006"
    error = rejected(raw.decode("ascii"))
    assert (error.code, error.line) == ("E001", 10)
    assert str(error) == "E001: unknown directive '#COMMENT' (line 10)"


def test_t34_import_fence():
    """openmucf/g4/ may not import the kinetics modules, so the data layer stays liftable.

    A source-level rule, checked with the AST: the package __init__ eagerly imports that stack, so
    a sys.modules check would prove nothing. What matters is that no module *here* names it.
    """
    banned = {f"openmucf.{name}" for name in ("cycle", "uq", "calibrate", "formation")}
    root = pathlib.Path(spec.__file__).resolve().parent
    checked = set()
    for path in sorted(root.rglob("*.py")):
        parts = ["openmucf", "g4", *path.relative_to(root).with_suffix("").parts]
        package = parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = ".".join(package[: len(package) - node.level + 1] if node.level else [])
                base = f"{base}.{node.module}" if base and node.module else (base or node.module or "")
                targets = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            else:
                continue
            for target in targets:
                offenders = [name for name in banned if target == name or target.startswith(f"{name}.")]
                assert not offenders, f"{path.name} imports {target!r} (line {node.lineno})"
        checked.add(path.name)
    assert checked == {"__init__.py", "spec.py", "provenance.py", "emit.py"}


# --------------------------------------------------------------------------------------------
# T-35..T-37 -- the archive, the digest on disk, and the committed example
# --------------------------------------------------------------------------------------------

REPO = pathlib.Path(__file__).resolve().parents[1]
G4DIR = REPO / "data" / "g4"


def example_members() -> dict[str, bytes]:
    """The two files the shipped archive holds, read as bytes from the committed dataset."""
    return {
        "example.g4dat": (G4DIR / "example.g4dat").read_bytes(),
        "example.prov.json": (G4DIR / "example.prov.json").read_bytes(),
    }


def test_t35_archive_is_deterministic():
    """Two builds of the same members are byte-identical, and nothing about this machine is in them.

    A tar entry carries an mtime, a uid/gid, owner names and a mode; a gzip container carries an
    mtime and can carry the source filename. Each is a channel for the builder to leak into the
    artifact, and an artifact whose bytes depend on who built it cannot be checksummed once and
    shipped. The gzip header is asserted field by field so that a future determinism failure can be
    attributed to the DEFLATE stream (a zlib build difference) rather than to this code.
    """
    members = example_members()
    first = emit.build_tarball(members)
    assert first == emit.build_tarball(members)
    assert first == emit.build_tarball(dict(reversed(list(members.items()))))  # insertion order

    header = emit.gzip_header(first)
    assert header["mtime"] == 0, "the gzip container is stamped with the build time"
    assert header["flags"] & 0x08 == 0, "the gzip container carries a source filename"
    assert (header["method"], header["xfl"], header["os"]) == (8, 2, 255)

    with tempfile.TemporaryDirectory() as scratch:
        path = pathlib.Path(scratch) / f"example.{emit.ARCHIVE_EXTENSION}"
        path.write_bytes(first)
        with tarfile.open(path) as archive:
            entries = archive.getmembers()
        assert [entry.name for entry in entries] == sorted(members)  # sorted, and nothing else
        for entry in entries:
            assert (entry.mtime, entry.uid, entry.gid, entry.uname, entry.gname) == (0, 0, 0, "", "")
            assert entry.mode == 0o644
    assert len(emit.tarball_md5(first)) == 32


def test_t36_source_digest_survives_a_round_trip_through_the_filesystem():
    """The digest binds Layer 1 to the BYTES OF A FILE, so prove it against a real file.

    Everything up to here checked the digest in memory, where the bytes cannot be mangled on the way
    out. The failure this guards is mundane and platform-specific: write the Layer-2 file in text
    mode on Windows and every LF becomes CRLF, the file grows, the digest no longer matches, and
    nothing notices until a consumer's validation fails. Unpack the archive, re-read the member from
    disk, re-hash -- and then do the same with a text-mode copy and require E009.
    """
    members = example_members()
    archive = emit.build_tarball(members)

    with tempfile.TemporaryDirectory() as scratch:
        root = pathlib.Path(scratch)
        with tarfile.open(fileobj=io.BytesIO(archive)) as opened:
            opened.extractall(root, filter="data")

        layer1 = spec.parse((root / "example.g4dat").read_bytes().decode("ascii"))
        from_disk = (root / "example.prov.json").read_bytes()
        assert from_disk == members["example.prov.json"]
        assert layer1.directives["SOURCEDIGEST"] == provenance.source_digest(from_disk)
        assert provenance.check_source_digest(layer1, from_disk) is None

        # Negative control: the same bytes written through a CRLF-translating text stream. Without
        # it this test would pass on a build that never writes the file correctly in the first place.
        crlf_path = root / "example.crlf.prov.json"
        with open(crlf_path, "w", encoding="ascii", newline="\r\n") as handle:
            handle.write(from_disk.decode("ascii"))
        crlf = crlf_path.read_bytes()
        assert crlf != from_disk and crlf.count(b"\r\n") == from_disk.count(b"\n")
        error = digest_rejected(layer1, crlf)
        assert error.code == "E009"


def test_t37_committed_example_regenerates_and_ships():
    """The committed dataset is exactly what the generator produces, and the wheel gets the code.

    `packages` in pyproject.toml is an explicit list, so a new subpackage is shipped only if someone
    adds it: that is the property under test, pinned here rather than discovered by a user whose
    `import openmucf.g4` fails after a clean install.
    """
    generator = REPO / "scripts" / "generate_g4data.py"
    result = subprocess.run(
        [sys.executable, str(generator), "--audit"], capture_output=True, text=True, cwd=REPO
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "g4data audit OK" in result.stdout

    committed = (G4DIR / "example.g4dat").read_bytes()
    assert b"\r" not in committed, "the checkout rewrote the dataset's line endings"
    table = spec.parse(committed.decode("ascii"))
    assert table.directives["DATASET"] == "G4MuonicData"
    assert table.directives["SOURCEDIGEST"] == provenance.source_digest(
        (G4DIR / "example.prov.json").read_bytes()
    )
    # Nothing synthetic wears a physics label: every row says what it is, in the file itself.
    document = provenance.from_json_obj(json.loads((G4DIR / "example.prov.json").read_text("ascii")))
    assert len(document.rows) == len(table.records) == 3
    for row in document.rows.values():
        assert row.needs_verification is True
        assert row.evaluation_method == "format example, not evaluated physics"
        assert row.source_library == "openmucf"

    snippet = (G4DIR / "geant4_add_dataset.snippet").read_text("ascii")
    assert "geant4_add_dataset(" in snippet and "ENVVAR    G4MUONICDATA" in snippet

    declared = tomllib.loads((REPO / "pyproject.toml").read_text("utf-8"))
    assert "openmucf.g4" in declared["tool"]["setuptools"]["packages"]
