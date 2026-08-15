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
import gzip
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

import openmucf
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
#: The 13 directive lines alone -- no records, no `#END`. A file that never closes its directive
#: block, which is the shape a truncated download has and the case section 4's reporting order turns
#: on.
HEADER_ONLY = "".join(
    line for line in CANONICAL.splitlines(keepends=True) if line.startswith("#") and line != "#END\n"
)


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
    """A missing directive is reported at the line it WOULD have occupied, on both entry points.

    It has no line of its own, so a convention is needed; `parse()` used the line where the block
    closed and `validate()` used the would-be slot, which meant identical content came back with the
    same code and two different lines for every one of the required directives. The would-be slot
    wins: it points at where the directive should go, which is what the reader has to act on.
    """
    error = rejected(drop_line(CANONICAL, "#VALIDITY"))
    assert (error.code, error.line) == ("E002", 12)  # 11 directives precede it, so it belongs on 12
    assert "'#VALIDITY'" in str(error)

    # The comparison that was missing: the two entry points, on the same content, for every one of
    # the eleven always-required directives.
    for keyword in spec.REQUIRED_DIRECTIVES:
        from_file = rejected(drop_line(CANONICAL, f"#{keyword}"))
        in_memory = rejected_table(make_table(**{keyword: None}))
        assert str(from_file) == str(in_memory), keyword
        assert from_file.code == "E002", keyword

    # An EMPTY value counts as ABSENT for every required directive, not only for `#SOURCESHA`, and
    # is reported at its own line. Six of the thirteen directives used to accept one in silence: a
    # `#SOURCEDIGEST` with nothing after it parsed, validated and round-tripped, carrying the
    # invariant that binds the two layers as the empty string.
    for keyword, line in (
        ("GRAMMAR", 1),
        ("DATASET", 2),
        ("VERSION", 3),
        ("TABLE", 6),
        ("GENERATOR", 7),
        ("SOURCEDIGEST", 8),
        ("UNITS", 10),
        ("COLUMNS", 11),
        ("VALIDITY", 12),
    ):
        from_file = rejected(replace_line(CANONICAL, f"#{keyword}", f"#{keyword}"))
        assert (from_file.code, from_file.line) == ("E002", line), keyword
        assert "empty value" in str(from_file), keyword
        in_memory = rejected_table(make_table(**{keyword: ""}))
        assert (in_memory.code, in_memory.line) == ("E002", line), keyword

    # `#FALLBACK` is optional, so an empty one is absent and that is simply allowed. It is still a
    # DISTINCT conforming file from one that omits the line -- two files, one meaning (section 2.2).
    assert spec.validate(make_table(FALLBACK="")) is None
    assert spec.render(make_table(FALLBACK="")) != spec.render(make_table(FALLBACK=None))
    assert spec.parse(spec.render(make_table(FALLBACK=""))) == make_table(FALLBACK="")

    # `#GRAMMAR` is the one required directive with a competing code, and the two entry points used
    # to disagree on it: parse() validates it eagerly at its own line and said E010 (an unreadable
    # version), while validate() reached the header loop first and said E002. An empty value declares
    # no version at all, so E002 governs -- and both paths must say so, or a C++ reader written from
    # section 4 has no tie-break rule and section 7's "report errors the same way" is false.
    from_file = rejected(replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR"))
    in_memory = rejected_table(make_table(GRAMMAR=""))
    assert (from_file.code, from_file.line) == ("E002", 1)
    assert str(from_file) == str(in_memory)

    # The `#GRAMMAR` line's verdict PREEMPTS what follows -- and the preemption belongs to the
    # position, not to the code, so the empty case preempts exactly as an unsupported major does.
    # validate() reached the grammar only at the end of the header rules, so a bad `#GRAMMAR` plus
    # any other header defect reported the other defect while parse() reported the grammar.
    for grammar, code in (("", "E002"), ("2.0", "E010"), ("one", "E010")):
        for other in ({"DATASET": ""}, {"XFOO": "x"}, {"VALIDITY": None}):
            table = make_table(GRAMMAR=grammar, **other)
            error = rejected_table(table)
            assert (error.code, error.line) == (code, 1), (grammar, other)


def test_t03_e003_directive_out_of_order():
    version, profile = "#VERSION      1.0.0\n", "#PROFILE      parity\n"
    error = rejected(CANONICAL.replace(version + profile, profile + version))
    assert (error.code, error.line) == ("E003", 4)
    units = "#UNITS        rate=1e6/s\n"
    repeated = rejected(CANONICAL.replace(units, units + units))
    assert (repeated.code, repeated.line) == ("E003", 11)
    assert "(repeated)" in str(repeated)
    trailing = rejected(CANONICAL.replace("#END", "#UNITS        rate=1e6/s\n#END"))
    assert (trailing.code, trailing.line) == ("E003", 17)

    # A duplicate is a duplicate wherever it lands. The detail used to say "repeated" only when the
    # copy immediately followed its twin and "must precede '#VALIDITY'" otherwise -- a true sentence
    # that hides the more useful fact, and one that made the message depend on incidental position.
    fallback = "#FALLBACK     goulard_primakoff b0a=-0.03 b0b=-0.25 b0c=3.24 t1=875e-9\n"
    separated = rejected(CANONICAL.replace(fallback, units + fallback))
    assert (separated.code, separated.line) == ("E003", 13)
    assert "(repeated)" in str(separated)


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

    # The same byte at the EDGE of a value. This format's whitespace is space and tab; Python's
    # `str.strip()` also eats VT, FF and CR, so validate()'s "no file can produce that" guard used to
    # swallow these and raise ValueError while parse() called the identical document E005/E006 --
    # the two entry points reporting different classes for one file.
    for value, code in (
        ("rate=1e6/s\x0b", "E005"),
        ("\x0brate=1e6/s", "E005"),
        ("rate=1e6/s\x0c", "E005"),
        ("rate=1e6/s\r", "E006"),
    ):
        edge = rejected_table(make_table(UNITS=value))
        assert (edge.code, edge.line) == (code, 10), value
        assert edge.code == rejected(replace_line(CANONICAL, "#UNITS", f"#UNITS        {value}")).code
    # Space and tab at the edge stay a programming error: no file can produce them (parse strips).
    for value in ("rate=1e6/s ", "\trate=1e6/s"):
        with pytest.raises(ValueError, match="leading or trailing whitespace"):
            spec.validate(make_table(UNITS=value))


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

    # One version, one spelling. `01.0` names the same major as `1.0`, so a reader that accepts both
    # lets two byte-different files declare the same grammar -- in a format whose entire identity
    # discipline is byte-exactness, and whose C++ readers would each pick their own rule for it.
    for malformed in ("01.0", "1.00", "1.0.0", "0001.0", "+1.0", "v1.0", "1.", ".0", "1"):
        bad = rejected(replace_line(CANONICAL, "#GRAMMAR", f"#GRAMMAR      {malformed}"))
        assert (bad.code, bad.line) == ("E010", 1), malformed
    two_digit = spec.parse(replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR      1.10"))
    assert two_digit.directives["GRAMMAR"] == "1.10"  # a two-digit MINOR is not a leading zero
    assert rejected(replace_line(CANONICAL, "#GRAMMAR", "#GRAMMAR      0.9")).code == "E010"  # major 0

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

    # Content ON the terminator line, which is the same defect one line earlier. `#END` is not a
    # directive, and letting the directive machinery diagnose it produced two different FALSE
    # messages for one file shape: "unknown directive '#END'" when no record preceded it, and
    # "'#END' appears after the record block began" when one did. Now: E011, at that line, always.
    with_records = rejected(CANONICAL.replace("#END", "#END x"))
    assert (with_records.code, with_records.line) == ("E011", END_LINE)
    assert str(with_records) == f"E011: the '#END' terminator line carries content: 'x' (line {END_LINE})"
    empty_table = HEADER_ONLY + "#END\n"
    assert empty_table.count("\n") == 14 and empty_table.endswith("#END\n")
    no_records = rejected(empty_table.replace("#END", "#END x"))
    assert (no_records.code, no_records.line) == ("E011", 14)
    assert str(no_records) == "E011: the '#END' terminator line carries content: 'x' (line 14)"

    # `#ENDX` is a different thing -- a directive claiming that name -- and stays diagnosed as one.
    assert rejected(CANONICAL.replace("#END", "#ENDX")).code == "E003"
    assert rejected(empty_table.replace("#END", "#ENDX")).code == "E001"
    # Trailing spaces and tabs after `#END` are still ignored (section 2.4).
    assert spec.parse(CANONICAL.replace("#END", "#END  \t")) == make_table()


def test_t12_e012_missing_end():
    error = rejected(drop_line(CANONICAL, "#END"))
    assert (error.code, error.line) == ("E012", END_LINE - 1)
    unterminated = rejected(CANONICAL.rstrip("\n"))
    assert (unterminated.code, unterminated.line) == ("E012", END_LINE)
    assert "newline" in str(unterminated)

    # The directive block closes at the FIRST RECORD LINE or at `#END`, whichever comes first, and
    # the header checks run there. So the same header defect reports differently depending on
    # whether the file has records -- and both answers are right. FORMAT_SPEC section 4 stated a
    # total order that got the truncated case backwards, which is the case Stage 3's C++ reader
    # would have inherited: a truncated download is the ordinary way this format breaks.
    for keyword in ("VALIDITY", "UNITS"):
        with_records = rejected(drop_line(CANONICAL, f"#{keyword}").replace("#END\n", ""))
        assert with_records.code == "E002", keyword  # the block closed at the first record line
        without_records = rejected(drop_line(HEADER_ONLY, f"#{keyword}"))
        assert without_records.code == "E012", keyword  # the block never closes; E012 wins
    bad_seam = replace_line(HEADER_ONLY, "#SEAM", "#SEAM         d9_not_a_seam")
    assert rejected(bad_seam).code == "E012"
    assert rejected(bad_seam + "#END\n").code == "E016"  # the control: close the block and E016 fires
    parity_gap = drop_line(HEADER_ONLY, "#SOURCESHA")
    assert rejected(parity_gap).code == "E012"
    assert rejected(parity_gap + "#END\n").code == "E013"
    # E010 stays the deliberate exception: eager at its own line, so it preempts even this.
    assert rejected(replace_line(HEADER_ONLY, "#GRAMMAR", "#GRAMMAR      9.0")).code == "E010"


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

    # The non-finite literals are checked BEFORE the lexical float pattern, and the order is not
    # cosmetic: none of them matches that pattern, so a reader applying the pattern first reports
    # E007 where this format requires E014. Section 2.3 rule 4 now says so.
    assert spec._FLOAT_PATTERN.match("inf") is None
    for literal in ("inf", "+inf", "-inf", "infinity", "nan", "NaN", "INF"):
        error = rejected(replace_line(CANONICAL, "29", f"29  63 {literal} 0.041"))
        assert (error.code, error.line) == ("E014", 15), literal
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


def test_t08b_key_rules_run_after_field_rules_and_e008_before_e015():
    """Section 4's phases are global, not per record -- and `parse` and `validate` must agree.

    `_check_records` used to check each record's fields and then its key before moving on, which
    broke section 4 twice over. A duplicate key on an early record preempted a malformed field on a
    later one, so `validate()` said E008 where `parse()` -- which lexes during its own line scan --
    said E014 on the identical document. And an ordering break preempted a genuine duplicate further
    down, though section 4 orders E008 before E015 globally and gives the reason.
    """

    def both(records, raw_rows):
        header = [line for line in CANONICAL.splitlines(keepends=True) if line.startswith("#")][:13]
        text = "".join(header) + "".join(row + "\n" for row in raw_rows) + "#END\n"
        return rejected(text), rejected_table(make_table(records=records))

    # A duplicate on line 15, an overflowing field on line 16: the field rule is phase 2 and wins.
    parsed, validated = both(
        ((1, 1, 0.5, 0.1), (1, 1, 0.5, 0.1), (3, 3, 10**400, 0.1)),
        [" 1 1 0.5 0.1", " 1 1 0.5 0.1", " 3 3 1e400 0.1"],
    )
    assert (parsed.code, parsed.line) == ("E014", 16)
    assert (validated.code, validated.line) == ("E014", 16)

    # An ordering break on line 15, a genuine duplicate on line 17: E008 comes first, globally.
    parsed, validated = both(
        ((5, 5, 0.5, 0.1), (2, 2, 0.5, 0.1), (9, 9, 0.5, 0.1), (2, 2, 0.5, 0.1)),
        [" 5 5 0.5 0.1", " 2 2 0.5 0.1", " 9 9 0.5 0.1", " 2 2 0.5 0.1"],
    )
    assert (parsed.code, parsed.line) == ("E008", 17)
    assert (validated.code, validated.line) == ("E008", 17)
    assert "first seen at line 15" in str(parsed)


def test_t08c_a_table_with_no_primary_key():
    """A table declaring neither `Z` nor `A` has no key, so rules 6 and 7 have nothing to check.

    Section 2.3 says so explicitly. The branch existed and worked but nothing exercised it: every
    other table in this file, generated or hand-built, declares at least one of `Z`/`A`.
    """
    keyless = make_table(COLUMNS="energy value", records=((9.0, 1.0), (2.0, 2.0), (9.0, 3.0)))
    assert spec._key_columns(["energy", "value"]) == []
    assert spec.validate(keyless) is None  # duplicated AND descending, both fine without a key
    assert spec.parse(spec.render(keyless)) == keyless  # and render must not reorder them
    assert [line.split()[0] for line in spec.render(keyless).splitlines()[13:16]] == ["9", "2", "9"]


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


def test_t16_e016_directive_value_outside_its_allowed_form():
    seam = rejected(replace_line(CANONICAL, "#SEAM", "#SEAM         d9_not_a_seam"))
    assert (seam.code, seam.line) == ("E016", 5)
    profile = rejected(replace_line(CANONICAL, "#PROFILE", "#PROFILE      Parity"))
    assert (profile.code, profile.line) == ("E016", 4)

    # `#SOURCEDIGEST` carries section 1's binding invariant, and E009 -- the only check it had --
    # needs the Layer-2 file. Stage 3's standalone Layer-1 validator will not have one, so without a
    # lexical rule the field is unchecked: `not-a-sha256`, 63 hex, and 64 UPPERCASE hex all parsed,
    # validated AND round-tripped.
    for digest in (
        "not-a-sha256",
        "0" * 63,
        "0" * 65,
        "0" * 63 + "g",
        ("0" * 63 + "A").upper(),
        "0" * 32 + " " + "0" * 31,
    ):
        error = rejected_table(make_table(SOURCEDIGEST=digest))
        assert (error.code, error.line) == ("E016", 8), digest
        assert "64 lowercase hex" in str(error), digest
    assert spec.validate(make_table(SOURCEDIGEST="0123456789abcdef" * 4)) is None

    # `#COLUMNS` names must be well formed and DISTINCT. `Z A Z value` used to validate, with the
    # key silently taking the first `Z` and the second column unreachable -- while section 2.3 rule 6
    # says the key is "whichever of Z and A the table declares", which that table makes meaningless.
    repeated = rejected_table(make_table(COLUMNS="Z A Z value", records=((1, 1, 1, 0.5),)))
    assert (repeated.code, repeated.line) == ("E016", 11)
    assert "repeats the name(s) Z" in str(repeated)
    for name, arity in (("9value", 3), ("val-ue", 3), ("val.ue", 3), ("value!", 3)):
        columns = f"Z A {name}"
        error = rejected_table(make_table(COLUMNS=columns, records=((1, 1, 0.5),) * 1))
        assert (error.code, error.line) == ("E016", 11), columns
        assert len(spec._split_fields(columns)) == arity, columns
    # A non-ASCII column name is caught one phase earlier, by the byte-set rule, and stays E005.
    assert rejected_table(make_table(COLUMNS="Z A µ", records=((1, 1, 0.5),))).code == "E005"
    assert spec.validate(make_table(COLUMNS="Z A value _unc2", records=((1, 1, 0.5, 0.1),))) is None


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


def test_t20b_record_lines_tolerate_edge_whitespace_and_sort_numerically():
    """Two rules an independent reader would otherwise have to guess, now stated in section 2.3.

    Both were true of this implementation and unstated in the document, which is the same defect as
    stating them wrongly: a C++ reader that splits a record line without stripping the END of it sees
    a trailing separator run as one more empty field and reports E004 on a conforming file, and a
    reader that compares the key columns as TEXT sorts `10` before `2` and reports E015 on a file
    this one accepts.
    """
    for suffix in ("", " ", "   ", "\t", " \t "):
        text = replace_line(CANONICAL, " 1", " 1   1 0.000725 1.7e-05" + suffix)
        assert spec.parse(text).records[0] == (1, 1, 0.000725, 1.7e-05), repr(suffix)
    leading = replace_line(CANONICAL, " 1", "\t 1   1 0.000725 1.7e-05")
    assert spec.parse(leading).records[0] == (1, 1, 0.000725, 1.7e-05)

    # Numeric, not lexicographic: (2, 2) then (10, 10) is ascending; the reverse is E015. Compared as
    # text, "10" sorts before "2" and the two orders would swap.
    ascending = make_table(records=((2, 2, 0.5, 0.1), (10, 10, 0.5, 0.1)))
    assert spec.validate(ascending) is None
    descending = rejected_table(make_table(records=((10, 10, 0.5, 0.1), (2, 2, 0.5, 0.1))))
    assert (descending.code, descending.line) == ("E015", 15)
    assert [line.split()[0] for line in spec.render(ascending).splitlines()[13:15]] == ["2", "10"]


def test_t21_records_are_sorted_on_render():
    reversed_table = make_table(records=tuple(reversed(RECORDS)))
    text = spec.render(reversed_table)
    assert text == spec.render(make_table())
    keys = [tuple(int(field) for field in line.split()[:2]) for line in text.splitlines()[13:16]]
    assert keys == sorted(keys) == [(1, 1), (29, 63), (94, 242)]
    assert spec.parse(text) == make_table()
    unsorted_error = rejected_table(reversed_table)
    assert (unsorted_error.code, unsorted_error.line) == ("E015", 15)

    # render() validates the table AS GIVEN and tolerates only E015. Validating the sorted copy let
    # the sort decide which record was reached first, so render() and validate() named different
    # defects on one table: E004 from render, E007 from validate. Sorting is the single liberty
    # render() takes; reporting a different diagnosis is not part of it.
    multi = make_table(records=((2.5, 1, 0.5, 0.1), (2, 2, 0.5)))
    with pytest.raises(G4DatFormatError) as from_render:
        spec.render(multi)
    from_validate = rejected_table(multi)
    assert str(from_render.value) == str(from_validate)


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
    for profile in ("Parity", "ab", "a" * 33, "9lives", "has space", "trailing!"):
        error = rejected_table(make_table(PROFILE=profile, SOURCESHA=None))
        assert (error.code, error.line) == ("E016", 4), profile
    # An EMPTY value is ABSENT, not malformed: this case moved from E016 to E002 deliberately, under
    # the rule that generalizes section 2.2's empty-counts-as-absent to every required directive.
    # "You have not said which evaluation this file carries" beats "'' is not a token".
    empty = rejected_table(make_table(PROFILE="", SOURCESHA=None))
    assert (empty.code, empty.line) == ("E002", 4)


def test_t24_seam_allowed_set():
    for seam in spec.ALLOWED_SEAMS:
        assert spec.validate(make_table(SEAM=seam)) is None
    assert spec.ALLOWED_SEAMS == (
        "d1_nuclear_capture",
        "d2_atomic_capture",
        "d3_transitions",
        "d4_mucf_cycle",
    )
    for seam in ("d5_unknown", "D1_NUCLEAR_CAPTURE", "nuclear_capture"):
        error = rejected_table(make_table(SEAM=seam))
        assert (error.code, error.line) == ("E016", 5), seam
    empty = rejected_table(make_table(SEAM=""))  # absent, not malformed -- see T-23
    assert (empty.code, empty.line) == ("E002", 5)


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

    # The canonical form is NORMATIVE (section 3) because the digest is taken over these bytes, and
    # the check has to live in the shipped package: `packages` ships openmucf and openmucf.g4, not
    # scripts/, so a rule enforced only by a build script is a rule no installed consumer can apply.
    assert "check_canonical_bytes" in provenance.__all__
    canonical = provenance.document_bytes(document)
    assert provenance.check_canonical_bytes(canonical) is None
    obj = provenance.to_json_obj(document)
    for label, variant in (
        ("re-indented", json.dumps(obj, sort_keys=True, indent=4, ensure_ascii=True) + "\n"),
        ("compact", json.dumps(obj, sort_keys=True, ensure_ascii=True) + "\n"),
        ("keys unsorted", json.dumps({"rows": obj["rows"], **obj}, indent=2, ensure_ascii=True) + "\n"),
        ("no trailing newline", json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True)),
        ("CRLF", (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n").replace("\n", "\r\n")),
    ):
        with pytest.raises(ValueError, match="canonical"):
            provenance.check_canonical_bytes(variant.encode("ascii"))
        assert provenance.source_digest(variant.encode("ascii")) != provenance.source_digest(canonical), label
    with pytest.raises(ValueError, match="not ASCII JSON"):
        provenance.check_canonical_bytes(b"\xef\xbb\xbf{}")
    with pytest.raises(ValueError, match="not ASCII JSON"):
        provenance.check_canonical_bytes(b"{not json")

    # Row keys carry no zero padding. JSON object keys are strings and nothing normalizes them, so a
    # lax pattern let "1-1", "01-1" and "001-001" be three distinct keys for one record -- the exact
    # collision section 3's paragraph says it exists to prevent, while its own parenthesized regex
    # permitted it. (Layer 1's integer FIELDS stay lax on purpose: they are converted to integers and
    # re-emitted canonically, so no two spellings survive a round-trip.)
    for bad_key in ("001-001", "01-1", "1-01", "+1-1", "1 - 1", "1-", "-1", "1--1", "01-01"):
        with pytest.raises(ValueError, match="not of the form"):
            provenance.validate_document({**obj, "rows": {bad_key: obj["rows"]["1-1"]}})
    for good_key in ("1-1", "0-0", "94-242", "10-100"):
        provenance.validate_document({**obj, "rows": {good_key: obj["rows"]["1-1"]}})


def test_t27_source_digest_matches():
    document = make_document()
    payload = provenance.document_bytes(document)
    table = make_table(SOURCEDIGEST=provenance.source_digest(payload))
    assert provenance.check_source_digest(table, payload) is None
    assert provenance.source_digest(document) == provenance.source_digest(payload)
    assert provenance.check_against_table(table, document) is None
    with pytest.raises(ValueError, match="does not match Layer-1"):
        provenance.check_against_table(make_table(VERSION="9.9.9"), document)

    # Section 3's "one object per Layer-1 record", enforced in the SHIPPED package. It used to live
    # only in scripts/generate_g4data.py, which `packages` does not ship -- the same gap that put
    # check_canonical_bytes here.
    with pytest.raises(ValueError, match="one-for-one"):
        provenance.check_against_table(make_table(records=(*RECORDS, (7, 14, 1.0, 0.1))), document)
    short = dataclasses.replace(
        document, rows={k: v for k, v in document.rows.items() if k != "94-242"}
    )
    with pytest.raises(ValueError, match="one-for-one"):
        provenance.check_against_table(table, short)
    # A SINGLE-key table is now keyed by that one column's integer, so the one-for-one rule applies
    # to it like any other -- it is no longer the registered-undefined case it was when only the
    # `"Z-A"` form existed. Here the table's key set is {"1"} and the document's is {"1-1", ...},
    # so every row and every record is unmatched, in both directions at once.
    with pytest.raises(ValueError, match="one-for-one"):
        provenance.check_against_table(
            make_table(COLUMNS="Z value unc", records=((1, 0.5, 0.1),)), document
        )
    # A table declaring NEITHER key column still has no primary key, and must SAY so rather than
    # silently no-op -- which let such a table carry any rows at all.
    empty_doc = dataclasses.replace(document, rows={})
    with pytest.raises(ValueError, match="at least one of"):
        provenance.check_against_table(
            make_table(COLUMNS="energy value", records=((1.0, 0.5),)), empty_doc
        )
    # BOTH directions. Guarding only "rows that cannot be keyed" left the worse half open: records
    # shipping with no provenance row at all, which is exactly what section 3 forbids.
    with pytest.raises(ValueError, match="one-for-one"):
        provenance.check_against_table(
            make_table(COLUMNS="Z value unc", records=((1, 0.5, 0.1),)), empty_doc
        )
    with pytest.raises(ValueError, match="at least one of"):
        provenance.check_against_table(make_table(COLUMNS="energy value", records=()), document)
    # A table with neither records nor rows is vacuously fine.
    assert provenance.check_against_table(
        make_table(COLUMNS="energy value", records=()), empty_doc
    ) is None


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
        if "__pycache__" in path.parts:
            continue
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
        checked.add(path.relative_to(root).as_posix())
    # Relative paths, not bare names: `sources/` brought a second `__init__.py` into the package, and
    # a set of names would have silently collapsed the two into one entry -- so a new fenced module
    # could be added under `sources/` without this inventory noticing it had arrived.
    assert checked == {
        "__init__.py",
        "spec.py",
        "provenance.py",
        "emit.py",
        "sources/__init__.py",
        "sources/d1_nuclear_capture.py",
    }

    # A layout invariant that is currently satisfied with exactly one space to spare, and that a
    # future directive would break silently: `#SOURCEDIGEST` is 13 characters, the pad is 14, so the
    # longest directive line has a single separator. A 13-character directive NAME would render with
    # none at all, and the line would come back as E001.
    longest = max(len("#" + keyword) for keyword in spec.DIRECTIVE_ORDER)
    assert longest + 1 <= spec._KEYWORD_WIDTH, "a directive name has outgrown the keyword pad"


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

    # Member names are flat and ASCII. A separator would silently produce a nested archive where
    # section 8 promises a flat one; the ASCII rule is about the MESSAGE, since the ustar length
    # check already rejects a non-ASCII name, as a UnicodeEncodeError rather than as a statement
    # about archive names.
    for bad in ("sub/dir.g4dat", "sub\\dir.g4dat", ""):
        with pytest.raises(ValueError, match="flat name"):
            emit.build_tarball({bad: b"x"})
    # Matched on the MESSAGE, not on the type: UnicodeEncodeError IS a ValueError, so a bare
    # `pytest.raises(ValueError)` here passed before the ASCII guard existed and pinned nothing.
    with pytest.raises(ValueError, match="US-ASCII"):
        emit.build_tarball({"examplé.g4dat": b"x"})
    with pytest.raises(ValueError, match="ustar"):
        emit.build_tarball({"n" * 101: b"x"})

    # The ustar rows of section 8, asserted against the real bytes rather than only written down.
    # Each of these has a legal alternative that a different writer picks, and each changes the MD5.
    decompressed = gzip.decompress(first)
    header = decompressed[:512]  # the first member's ustar header block
    assert header[257:265] == b"ustar\x0000"
    assert header[100:108] == b"0000644\x00" and header[136:148] == b"00000000000\x00"
    assert header[148:156].endswith(b"\x00 ") and len(header[148:156]) == 8  # 6 octal, NUL, space
    assert header[329:345] == b"\x00" * 16, "devmajor/devminor must be NUL, not octal zero"
    assert header[156:157] == b"0", "typeflag must be '0', not the equally legal NUL spelling"
    assert len(decompressed) % 10240 == 0 and decompressed.endswith(b"\x00" * 1024)


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
    layer2 = (G4DIR / "example.prov.json").read_bytes()
    # `.gitattributes` marks `data/g4/* -text` and that line is load-bearing on any checkout with
    # core.autocrlf set: example.prov.json IS the byte range #SOURCEDIGEST is taken over, so a CRLF
    # checkout changes the digest, and example.g4dat is LF-only by E006. Assert both directly, so
    # deleting the attribute names its own cause on the windows job instead of surfacing as a
    # digest mismatch nobody can explain.
    assert b"\r" not in committed, "the checkout rewrote the dataset's line endings"
    assert b"\r" not in layer2, "the checkout rewrote the Layer-2 file the digest is taken over"
    provenance.check_canonical_bytes(layer2)
    table = spec.parse(committed.decode("ascii"))
    assert table.directives["DATASET"] == "G4MuonicData"
    assert table.directives["SOURCEDIGEST"] == provenance.source_digest(layer2)

    # #GENERATOR embeds openmucf.__version__, deliberately: a consumer holding a broken file needs
    # to know which tool version made it. The cost is that a version bump changes these bytes and
    # the archive MD5, so `make audit` goes red until the dataset is regenerated. That coupling is
    # made loud HERE, with the remedy in the message, rather than discovered as a byte-diff.
    assert table.directives["GENERATOR"] == f"openmucf-g4 {openmucf.__version__}", (
        "openmucf.__version__ has moved but data/g4/ was not regenerated: run "
        "`python scripts/generate_g4data.py` and commit example.g4dat AND "
        "geant4_add_dataset.snippet (its MD5SUM changes too)"
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


# --------------------------------------------------------------------------------------------
# T-38..T-39 -- diagnosis determinism, and the one write path that actually exists
# --------------------------------------------------------------------------------------------


def test_t38_validate_diagnosis_is_independent_of_insertion_order():
    """Two tables holding the same directives must be rejected identically, however they were built.

    Section 4 exists so that two implementations cannot disagree about a file they both reject; a
    diagnosis that depends on `dict` insertion order fails that inside a single implementation. It
    did: the same directives inserted forward and reversed reported E005 on line 9 versus line 12,
    and two unknown directives reported E001 on line 13 versus line 12. T-20 pins the same property
    for `render`, which is where the requirement comes from -- validate() and render() must agree
    about what "canonical order" means, because validate() reports the lines render() would emit.
    """

    def both_orders(**overrides) -> tuple[str, str]:
        forward = make_table(**overrides)
        reversed_table = G4DatTable(
            directives=dict(reversed(list(forward.directives.items()))), records=forward.records
        )
        assert forward.directives == reversed_table.directives  # same content, opposite layout
        assert list(forward.directives) != list(reversed_table.directives)
        return str(rejected_table(forward)), str(rejected_table(reversed_table))

    # Two E005 candidates on different lines: whichever is canonically FIRST must win, both times.
    forward, backward = both_orders(UNITS="rate=1e6\x0b/s", FALLBACK="model\x0cx=1")
    assert forward == backward == "E005: control character '\\x0b' (0x0B) in directive '#UNITS' (line 10)"

    # Two unknown directives: the canonical order sorts them, so the answer cannot depend on which
    # was inserted first. (Their line numbers are notional either way -- render never emits one.)
    forward, backward = both_orders(AAA="x", ZZZ="y")
    assert forward == backward and forward.startswith("E001: unknown directive '#AAA'")

    # And a header defect competing with a record defect resolves the same way from either layout.
    forward, backward = both_orders(SEAM="d9_nope", records=((1, 1, 0.5, 0.1), (1, 1, 0.5, 0.1)))
    assert forward == backward and forward.startswith("E016: '#SEAM'")


def test_t39_the_generator_reads_layer_2_as_bytes():
    """The digest is over the file's bytes, so the one product read path must be binary.

    R3 asked for proof that the write path is binary; the project has no Layer-2 writer at all --
    Layer 2 is hand-authored and only ever read -- so T-36 exercises `tarfile`'s writer plus its own
    text-mode control. The property that DOES exist in product code is this one, and a `read_text`
    here would rewrite CRLF on Windows and hand back bytes the file does not contain.
    """
    source = (REPO / "scripts" / "generate_g4data.py").read_text("utf-8")
    tree = ast.parse(source)
    reads = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"LAYER2_PATH", "LAYER1_PATH"}
        and node.func.attr.startswith(("read", "write", "open"))
    }
    assert reads == {"read_bytes"}, f"Layer-1/Layer-2 I/O must be binary, found {sorted(reads)}"
    assert "read_text" not in source and "open(" not in source
    # The write side of the same rule, on the generated artifacts.
    assert "write_bytes" in source and "write_text" not in source
