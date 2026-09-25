"""T-74/T-75, T-78/T-79.

`tests/test_g4parity.py` pins the counts it knows how to compute (T-63), and its docstring says what
that leaves open: a pin table is not a census, so a number nobody thought to pin drifts unwatched.
This file closes the complement. It enumerates, in the documents named by `PROSE_PATHS`, every
numeric token and every word `_WORDS` lists (the spelled numbers), and admits each one only through
one of three doors, tried in order:

1. **a pin** -- the token lies inside the captured group of a pattern whose value is computed at
   run time (T-63's tables, the F-3 figures `cpp/test/check_f3.py` reads, and `INTERNAL_PINS`
   below);
2. **a class** -- a row of `g4_prose_classes.tsv` whose glob matches the file and whose regex,
   anchored on a non-numeric marker, matches around the token: dates, DOIs, digests, version
   strings, identifiers and the like, each row with a written reason;
3. **the registry** -- `g4_prose_registry.tsv` keys the line by `(path, sha1 of the normalised
   line)`, names every token the first two doors did not admit, and says why it may stand.

Anything else fails, naming file, line, token and the line. Nothing is excluded for sitting in a
code span or a fenced block: the F-3 block is data. The drills in T-75 plant an unpinned digit and a
spelled-number decoy in in-memory copies and require the failure to name them.

Door 3 is then walked the other way (T-78): every file a registry reason names must resolve to one
file of the tree, and every figure a reason types must stand verbatim in the ruled line or, for a
digit or superscript figure, in a file the reason names -- a reason may point at a number, never
restate one unchecked, and a spelled figure is prose that no home vouches for. T-79 drills that
with a missing home, an ambiguous one, a figure no home carries and a spelled figure a home does.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import functools
import hashlib
import importlib.util
import os
import pathlib
import re
import subprocess
import sys

import test_g4d3 as d3
import test_g4parity as parity

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The documents under the check. Every other public document is out of its reach and says nothing
#: this check would vouch for.
PROSE_PATHS = (
    "DATASET_D1.md", "README.md", "cpp/tools/README.md", "cpp/README.md", "CHANGELOG.md",
    "third_party/geant4/README.md", "cpp/patches/README.md", "DATASET_D3.md",
    "cpp/transport/README.md", "DATASET_D2.md",
    "paper/muonic-data/paper.md", "FORMAT_SPEC.md",
)
#: Documents that may carry no registry row: every token in them is pinned or class-admitted.
REGISTRY_FREE = ("cpp/README.md",)


def test_t74_transport_readme_enumerated():
    assert "cpp/transport/README.md" in PROSE_PATHS


def test_t74_paper_draft_enumerated():
    assert "paper/muonic-data/paper.md" in PROSE_PATHS


def test_t74_format_specification_enumerated():
    assert "FORMAT_SPEC.md" in PROSE_PATHS


CLASSES = pathlib.Path(__file__).with_name("g4_prose_classes.tsv")
REGISTRY = pathlib.Path(__file__).with_name("g4_prose_registry.tsv")
CHECK_F3 = REPO / "cpp" / "test" / "check_f3.py"

#: How many registry rows may still be `UNREVIEWED`. Zero: this registry was ruled line by line
#: when it was written, and a new line is ruled in the commit that adds it.
UNREVIEWED_CEILING = 0
VALID_PREFIXES = ("EXERCISED:", "REGISTERED:")

# --------------------------------------------------------------------------------------------
# The tokenizer
# --------------------------------------------------------------------------------------------

NUMERIC = re.compile(r"[0-9][0-9,.]*[0-9]|[0-9]|[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million",
]
_WORD = "|".join(_WORDS)
#: A hyphenated compound (`ninety-one`) is one token.
SPELLED = re.compile(rf"\b(?:{_WORD})(?:-(?:{_WORD}))*\b", re.IGNORECASE)


@dataclasses.dataclass(frozen=True, order=True)
class Token:
    path: str
    lineno: int
    col: int
    text: str

    @property
    def end(self) -> int:
        return self.col + len(self.text)


def tokenize(path: str, text: str) -> list[Token]:
    """Every numeric token and spelled number `_WORDS` lists in `text`, in file order."""
    out: list[Token] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        found = [(m.start(), m.group()) for m in NUMERIC.finditer(line)]
        found += [(m.start(), m.group()) for m in SPELLED.finditer(line)]
        out.extend(Token(path, lineno, col, tok) for col, tok in sorted(found))
    return out


# --------------------------------------------------------------------------------------------
# Door 1 -- pins, matched on the whitespace-collapsed document and mapped back to (line, column)
# --------------------------------------------------------------------------------------------


def collapse(text: str) -> tuple[str, list[tuple[int, int] | None]]:
    """`" ".join(text.split())` together with, for each collapsed character, its raw `(line, col)`.

    The collapsed form is the one T-63's patterns are written against; the map is what lets a match
    in it name a token in the file. Built by one walk so the two cannot disagree, and checked
    against `str.split` so the convention is the library's, not this file's.
    """
    out: list[str] = []
    origin: list[tuple[int, int] | None] = []
    pending = False
    for lineno, line in enumerate(text.splitlines(), 1):
        for col, ch in enumerate(line):
            if ch.isspace():
                pending = True
                continue
            if pending and out:
                out.append(" ")
                origin.append(None)
            pending = False
            out.append(ch)
            origin.append((lineno, col))
        pending = True
    collapsed = "".join(out)
    assert collapsed == " ".join(text.split())
    return collapsed, origin


@dataclasses.dataclass(frozen=True)
class Pin:
    what: str
    path: str
    pattern: str
    groups: tuple[int, ...]
    #: The value the captured group must state, when this file is the one asserting it; `None`
    #: for a pattern whose value another test already asserts (T-63, `check_f3.py`) --
    #: `check_f3.py`'s document-to-producer comparison runs only where a producer exists
    #: (`f3_check`, Linux x86-64 in CI); its README-to-document comparison is repeated here.
    expected: object = None
    #: Decimal places the document rounds `expected` to; `None` compares as an integer count.
    places: int | None = None


def _load_check_f3():
    spec = importlib.util.spec_from_file_location("check_f3", CHECK_F3)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def pin_table() -> list[Pin]:
    """Every pin, mechanically: T-63's tables through `document_pins`, the two F-3 figure patterns
    of `check_f3.py`, and `INTERNAL_PINS`. No pattern is restated here."""
    pins = parity.document_pins()
    check_f3 = _load_check_f3()
    table: list[Pin] = []
    for path, rows in (
        ("DATASET_D1.md", pins.claims),
        ("DATASET_D1.md", pins.rounded),
        ("CHANGELOG.md", pins.changelog_claims),
        ("CHANGELOG.md", pins.changelog_rounded),
        ("README.md", pins.readme_claims),
        ("cpp/tools/README.md", pins.tools_readme_claims),
        ("DATASET_D1.md", pins.crosscheck_rows),
        ("DATASET_D1.md", pins.suzuki_crosscheck_rows),
        ("DATASET_D1.md", pins.crosscheck_unpartnered),
        ("CHANGELOG.md", pins.string_claims),
        ("DATASET_D1.md", pins.open_row_comparisons),
        ("DATASET_D1.md", pins.settled_by_value),
    ):
        for row in rows:
            table.append(Pin(row[0], path, row[1], (1,)))
    document_figures = check_f3.DOCUMENT_FIGURES
    readme_figures = check_f3.README_FIGURES
    table.append(Pin("F-3 figures, the document's block", "DATASET_D1.md",
                     document_figures.pattern, tuple(range(1, document_figures.groups + 1))))
    table.append(Pin("F-3 figures, the harvest tooling's restatement", "cpp/tools/README.md",
                     readme_figures.pattern, tuple(range(1, readme_figures.groups + 1))))
    table.append(Pin("F-3 maximum relative difference, the document's line", "DATASET_D1.md",
                     check_f3.DOCUMENT_MAX_REL.pattern, (1,)))
    table.extend(internal_pins(pins, check_f3))
    table.extend(paper_pins(pins, check_f3))
    # The D3 document's pins, as its own test module builds them.
    for what, path, pattern, groups, expected in d3.document_pins():
        table.append(Pin(what, path, pattern, groups, expected))
    return table


def _computed(pins: parity.DocumentPins, what: str) -> int:
    """The value T-63 computed for the count row it labels `what` -- looked up, never retyped."""
    for table in (pins.claims, pins.changelog_claims, pins.readme_claims, pins.tools_readme_claims):
        for row in table:
            if row[0] == what:
                return int(row[2])
    raise KeyError(what)


def _rounded(pins: parity.DocumentPins, what: str) -> tuple[float, int]:
    """`(value, places)` of the rounded-figure row T-63 labels `what`."""
    for row in pins.rounded:
        if row[0] == what:
            return float(row[2]), int(row[3])
    raise KeyError(what)


def internal_pins(pins: parity.DocumentPins, check_f3) -> list[Pin]:
    """Restatements of computed values that T-63's tables do not reach. Every `expected` is a value
    another module computed at run time; this list adds patterns, never numbers."""
    figures = check_f3.document_figures(REPO / "DATASET_D1.md")
    assert check_f3.readme_figures(REPO / "cpp" / "tools" / "README.md") == (
        figures[3], figures[1], figures[0]
    ), "cpp/tools/README.md restates F-3 figures the dataset document does not state"
    max_ulp = figures[3]
    d1 = parity.d1
    records = _computed(pins, "capture record count, section 1")
    distinct_z = _computed(pins, "distinct Z, section 1")
    swept = _computed(pins, "swept points, section 4")
    zeff_entries = _computed(pins, "effective-charge array length, section 3")
    named = _computed(pins, "elements the primary's sentence names")
    carrying = _computed(pins, "named elements carrying a separated-isotope record")
    zeff_56 = _rounded(pins, "the effective charge at Z=56")
    zeff_81 = _rounded(pins, "the effective charge at Z=81")
    zeff_82 = _rounded(pins, "the effective charge at Z=82")
    zeff_83 = _rounded(pins, "the effective charge at Z=83")
    # The unit conversion the document states is the module's own constant: nanoseconds per
    # microsecond, which is what `value / 1000` divides by.
    per_microsecond = d1.MICROSECOND
    assert per_microsecond == int(per_microsecond)
    # The vendored README's size cells are the bytes and newlines of the vendored files themselves.
    vendored_readme = "third_party/geant4/README.md"
    bd = parity.VENDORED.read_bytes()
    hp = parity.HELPER.read_bytes()
    beta_bd = parity.BETA_BOUND_DECAY.read_bytes()
    beta_hp = parity.BETA_HELPER.read_bytes()
    # The error codes the format defines are the codes its specification names, and the reference
    # implementation raises the same set; the changelog's spelled count is held to it.
    codes_specified = set(re.findall(r"\bE0\d\d\b", (REPO / "FORMAT_SPEC.md").read_text("utf-8")))
    codes_raised = set(re.findall(r"\bE0\d\d\b", (REPO / "openmucf" / "g4" / "spec.py").read_text("utf-8")))
    assert codes_specified == codes_raised, sorted(codes_specified ^ codes_raised)
    error_codes = len(codes_specified)
    return [
        Pin("F-3 maximum restated in section 2", "DATASET_D1.md",
            r"changes it by up to (\d+) ulp", (1,), max_ulp),
        Pin("capture record count, F-7", "DATASET_D1.md",
            r"For \*\*\d+ of the (\d+) records\*\* the primary shows", (1,), records),
        Pin("capture record count, section 6", "DATASET_D1.md",
            r"Each of the (\d+) records was checked", (1,), records),
        Pin("distinct Z, F-4 page reading", "DATASET_D1.md",
            r"every one of the (\d+) was read", (1,), distinct_z),
        Pin("the maximum Z, F-5", "DATASET_D1.md",
            r"gaps above, and Z > (\d+)\) are in neither table", (1,),
            _computed(pins, "the maximum Z, section 1")),
        Pin("records out of place, section 3's opening sentence", "DATASET_D1.md",
            r"and exactly (\w+) record sits out of", (1,),
            _computed(pins, "records out of place in the upstream declaration order")),
        Pin("swept points, section 3's box", "DATASET_D1.md",
            r"at every point of a (\d+)-point box", (1,), swept),
        Pin("effective-charge entries, the array as declared", "DATASET_D1.md",
            r'because "(\d+)/\d+ bit-identical" means', (1,), zeff_entries),
        Pin("effective-charge entries, the array as declared, denominator", "DATASET_D1.md",
            r'because "\d+/(\d+) bit-identical" means', (1,), zeff_entries),
        Pin("nanoseconds per microsecond, the table-hit conversion", "DATASET_D1.md",
            r"the rate is instead `value / (\d+)`", (1,), int(per_microsecond)),
        Pin("named elements without a separated-isotope record, F-6", "DATASET_D1.md",
            r"The (\w+) exceptions are instructive", (1,), named - carrying),
        Pin("mononuclidic route, section 6 restated", "DATASET_D1.md",
            r"the (\d+) mononuclidic calls above", (1,), _computed(pins, "mononuclidic route")),
        Pin("the effective charge at Z=56, the barium sentence", "DATASET_D1.md",
            r"barium is Z = \d+, and ([\d.]+) sits correctly", (1,), zeff_56[0], zeff_56[1]),
        Pin("the effective charge at Z=82, the second descent's start", "DATASET_D1.md",
            r"Z=82→83 \(([\d.]+) →", (1,), zeff_82[0], zeff_82[1]),
        # F-5 quotes the primary's three lead-region entries and says the shipped array matches
        # them; that is a claim about the shipped array, so each is held to it.
        Pin("the effective charge at Z=81, as the primary prints it", "DATASET_D1.md",
            r"Table IV prints \*\*81\(([\d.]+)\), 82\([\d.]+\), 83\([\d.]+\)\*\*", (1,),
            zeff_81[0], zeff_81[1]),
        Pin("the effective charge at Z=82, as the primary prints it", "DATASET_D1.md",
            r"Table IV prints \*\*81\([\d.]+\), 82\(([\d.]+)\), 83\([\d.]+\)\*\*", (1,),
            zeff_82[0], zeff_82[1]),
        Pin("the effective charge at Z=83, as the primary prints it", "DATASET_D1.md",
            r"Table IV prints \*\*81\([\d.]+\), 82\([\d.]+\), 83\(([\d.]+)\)\*\*", (1,),
            zeff_83[0], zeff_83[1]),
        Pin("F-3 maximum, the README's restatement", "README.md",
            r"moves by up to \*\*(\d+) ulp\*\* between two conforming", (1,), max_ulp),
        Pin("F-3 maximum, the changelog's restatement", "CHANGELOG.md",
            r"moves by up to \*\*(\d+) ulp\*\* between two conforming", (1,), max_ulp),
        Pin("sweep box, Z lower bound, section 4", "DATASET_D1.md",
            r"over \*\*Z ∈ \[(\d+),\d+\] × A ∈ \[\d+,\d+\] = \d+ points\*\*", (1,), d1.SWEEP_Z_MIN),
        Pin("sweep box, Z upper bound, section 4", "DATASET_D1.md",
            r"over \*\*Z ∈ \[\d+,(\d+)\] × A ∈ \[\d+,\d+\] = \d+ points\*\*", (1,), d1.SWEEP_Z_MAX),
        Pin("sweep box, A lower bound, section 4", "DATASET_D1.md",
            r"over \*\*Z ∈ \[\d+,\d+\] × A ∈ \[(\d+),\d+\] = \d+ points\*\*", (1,), d1.SWEEP_A_MIN),
        Pin("sweep box, A upper bound, section 4", "DATASET_D1.md",
            r"over \*\*Z ∈ \[\d+,\d+\] × A ∈ \[\d+,(\d+)\] = \d+ points\*\*", (1,), d1.SWEEP_A_MAX),
        Pin("harvest box, Z lower bound", "cpp/tools/README.md",
            r"over Z (\d+)\.\.\d+ × A \d+\.\.\d+", (1,), d1.SWEEP_Z_MIN),
        Pin("harvest box, Z upper bound", "cpp/tools/README.md",
            r"over Z \d+\.\.(\d+) × A \d+\.\.\d+", (1,), d1.SWEEP_Z_MAX),
        Pin("harvest box, A lower bound", "cpp/tools/README.md",
            r"over Z \d+\.\.\d+ × A (\d+)\.\.\d+", (1,), d1.SWEEP_A_MIN),
        Pin("harvest box, A upper bound", "cpp/tools/README.md",
            r"over Z \d+\.\.\d+ × A \d+\.\.(\d+)", (1,), d1.SWEEP_A_MAX),
        Pin("capture record count, the vendored README", vendored_readme,
            r"a (\d+)-record", (1,), records),
        Pin("effective-charge entries, the vendored README", vendored_readme,
            r"a (\d+)-value effective-charge", (1,), zeff_entries),
        Pin("vendored BoundDecay size, bytes", vendored_readme,
            r"\| size \| (\d+) bytes, \d+ lines \|", (1,), len(bd)),
        Pin("vendored BoundDecay size, lines", vendored_readme,
            r"\| size \| \d+ bytes, (\d+) lines \|", (1,), bd.count(b"\n")),
        Pin("vendored helper size, bytes", vendored_readme,
            r"\| `G4MuonicAtomHelper\.cc` size \| (\d+) bytes, \d+ lines \|", (1,), len(hp)),
        Pin("vendored helper size, lines", vendored_readme,
            r"\| `G4MuonicAtomHelper\.cc` size \| \d+ bytes, (\d+) lines \|", (1,), hp.count(b"\n")),
        Pin("vendored beta BoundDecay size, bytes", vendored_readme,
            r"\| `v11\.5\.0\.beta/G4MuonMinusBoundDecay\.cc` size \| (\d+) bytes, \d+ lines \|",
            (1,), len(beta_bd)),
        Pin("vendored beta BoundDecay size, lines", vendored_readme,
            r"\| `v11\.5\.0\.beta/G4MuonMinusBoundDecay\.cc` size \| \d+ bytes, (\d+) lines \|",
            (1,), beta_bd.count(b"\n")),
        Pin("vendored beta helper size, bytes", vendored_readme,
            r"\| `v11\.5\.0\.beta/G4MuonicAtomHelper\.cc` size \| (\d+) bytes, \d+ lines \|",
            (1,), len(beta_hp)),
        Pin("vendored beta helper size, lines", vendored_readme,
            r"\| `v11\.5\.0\.beta/G4MuonicAtomHelper\.cc` size \| \d+ bytes, (\d+) lines \|",
            (1,), beta_hp.count(b"\n")),
        Pin("error codes the format defines, the changelog's count", "CHANGELOG.md",
            r"\*\*(\w+) exact error codes\*\*", (1,), error_codes),
        *d3_seam_size_pins(vendored_readme),
        *format_spec_pins(error_codes),
    ]


FORMAT_SPEC = "FORMAT_SPEC.md"


def format_spec_pins(error_codes: int) -> list[Pin]:
    """The format specification's restatements of what the reference implementation carries: the
    grammar version, each directive's place in the order, the digest length, the patterns the
    reader compiles, the integer-column bounds, the error-code count and the archive's pinned
    member fields. Every `expected` is read from `openmucf.g4` or computed; none is typed here."""
    from openmucf.g4 import emit, spec

    grammar = spec.GRAMMAR_VERSION
    hex_digits = len(hashlib.sha256().hexdigest())
    integer = spec._INTEGER_PATTERN.pattern
    column_name = spec._COLUMN_NAME_PATTERN.pattern
    pins = [
        Pin(f"directive order, #{name}", FORMAT_SPEC, rf"\| (\d+) \| `#{name}` \| ", (1,), place)
        for place, name in enumerate(spec.DIRECTIVE_ORDER, 1)
    ]
    for what, pattern, expected in (
        ("grammar version, the document's own",
         r"Version of this document: \*\*grammar (\d+\.\d+)\*\*", grammar),
        ("grammar version, the directive table", r"`MAJOR\.MINOR`; currently `(\d+\.\d+)`", grammar),
        ("grammar version, the unconstrained directives",
         r"grammar (\d+\.\d+) pins no internal syntax", grammar),
        ("grammar version, the example header", r"#GRAMMAR (\d+\.\d+) #DATASET", grammar),
        ("grammar pattern, the checked values", r"\| `#GRAMMAR` \| `([^`]+)` \(`E010`\)",
         spec._GRAMMAR_PATTERN.pattern.replace("|", "\\|")),
        ("grammar pattern, section 2.7", r"lexically `([^`]+)` -- two runs",
         spec._GRAMMAR_PATTERN.pattern),
        ("digest length, the directive table",
         r"SHA-256 of the Layer-2 file, (\d+) lowercase hex", hex_digits),
        ("digest length, the checked values", r"exactly (\d+) lowercase hex characters", hex_digits),
        ("digest length, the example header", r"\.\.\.(\d+) lowercase hex total\.\.\.", hex_digits),
        ("digest pattern, the checked values", r"lowercase hex characters, `([^`]+)` \(`E016`\)",
         spec._SOURCEDIGEST_PATTERN.pattern),
        ("column-name pattern, the checked values",
         r"one or more names matching `([^`]+)`, all", column_name),
        ("column-name pattern, the #UNITS name",
         r"`NAME=UNIT`, `NAME` matching `([^`]+)`", column_name),
        ("profile pattern, section 2.5", r"`#PROFILE` is a token matching `([^`]+)`",
         spec.PROFILE_PATTERN.pattern),
        ("integer pattern, section 2.3", r"the field must match `([^`]+)` and its value", integer),
        ("integer lower bound, section 2.3",
         r"must lie in \*\*`(\d+)`-`\d+` inclusive\*\*", spec.INTEGER_MIN),
        ("integer upper bound, section 2.3",
         r"must lie in \*\*`\d+`-`(\d+)` inclusive\*\*", spec.INTEGER_MAX),
        ("integer pattern, the E007 row", r"or `([^`]+)` within `\d+`-`\d+` for `Z`", integer),
        ("integer lower bound, the E007 row", r"within `(\d+)`-`\d+` for `Z`", spec.INTEGER_MIN),
        ("integer upper bound, the E007 row", r"within `\d+`-`(\d+)` for `Z`", spec.INTEGER_MAX),
        ("integer pattern, the Layer-2 key note", r"are laxer — `([^`]+)`, section", integer),
        ("error codes the format defines, section 7",
         r"the section-4 codes remain exactly the (\w+) file-level", error_codes),
        ("stored member name limit, section 8", r"at most \*\*(\d+) bytes\*\*", emit._MAX_MEMBER_NAME),
        ("member mode, section 8", r"\| tar \| `mode` \| `(\d+)` \|", f"{emit._MEMBER_MODE:04o}"),
        ("member mtime, section 8", r"\| tar \| `mtime` \| `(\d+)` \|", emit._EPOCH),
        ("gzip mtime, section 8", r"\| gzip \| `mtime` \| `(\d+)` \|", emit._EPOCH),
    ):
        pins.append(Pin(what, FORMAT_SPEC, pattern, (1,), expected))
    pins.extend(format_spec_derived_pins())
    return pins


def format_spec_derived_pins() -> list[Pin]:
    """The format specification's restatements of values the reference implementation holds in a
    compiled pattern, passes as a literal argument or writes through its own code: the bytes the
    reader permits and splits on, the digit range of its underflow test, the Layer-2 key pattern,
    the float precision and the text the emitter writes, the Layer-2 layout, the archive member's
    owner, mode and type, the gzip compression level and header bytes, and the line a directive occupies
    in the order.
    Every `expected` is read from `openmucf.g4` at run time -- a pattern parsed, an output measured,
    a call's argument read from its source -- and none is typed here."""
    import gzip
    import inspect
    import io
    import json
    import math
    import tarfile
    from re import _parser as sre_parse

    from openmucf.g4 import emit, provenance, spec

    doc = collapse((REPO / FORMAT_SPEC).read_text(encoding="utf-8"))[0]
    hexa = "0x[0-9A-F]{2}"

    # `_FORBIDDEN_BYTE` negates a class of single bytes and one range: the bytes the reader permits.
    [(_, forbidden)] = list(sre_parse.parse(spec._FORBIDDEN_BYTE.pattern))
    tab, lf, cr = (f"0x{v:02X}" for v in sorted(v for op, v in forbidden if op is sre_parse.LITERAL))
    [(low, high)] = [v for op, v in forbidden if op is sre_parse.RANGE]
    # `_FIELD_SEPARATOR` repeats a class of the separator bytes, space first.
    [(_, (_, _, repeated))] = list(sre_parse.parse(spec._FIELD_SEPARATOR.pattern))
    [(_, separators)] = list(repeated)
    space, separator_tab = (f"0x{v:02X}" for op, v in separators)
    # `_NONZERO_DIGIT` is one range of digit characters.
    [(_, [(_, (first, last))])] = list(sre_parse.parse(spec._NONZERO_DIGIT.pattern))
    # `_ROW_KEY_PATTERN` is the single-column form with the both-columns suffix optional.
    key = re.fullmatch(r"\^(\([^()]*\))\(\?:-(\([^()]*\))\)\?\$", provenance._ROW_KEY_PATTERN.pattern)
    assert key is not None, provenance._ROW_KEY_PATTERN.pattern
    both_keys = f"^{key[1]}-{key[2]}$".replace("|", "\\|")
    one_key = f"^{key[1]}$".replace("|", "\\|")
    # The precision `format_float` writes, measured on a value that needs every digit.
    precision = len(spec.format_float(math.pi).replace(".", ""))
    subnormal = spec.format_float(math.ulp(0.0))
    entered = re.search(r"entered as `([^`]+)` is emitted as", doc)
    assert entered is not None, "section 2.6 no longer states an entered value"
    emitted = spec.format_float(float(entered[1]))
    # The Layer-2 layout `render_json` writes, measured on a shipped Layer-2 document.
    shipped = (REPO / "data" / "g4" / "d1" / "d1_zeff.prov.json").read_text(encoding="ascii")
    rendered = provenance.render_json(provenance.from_json_obj(json.loads(shipped)))
    nested = rendered.splitlines()[1]
    indent = len(nested) - len(nested.lstrip(" "))
    newlines = len(rendered) - len(rendered.rstrip("\n"))
    # The member `build_tarball` writes, read back; the compression level, read from its call; the
    # gzip header bytes, as `gzip_header` reads them.
    archive = emit.build_tarball(
        {FORMAT_SPEC: b""}, directory=emit.dataset_directory("G4MuonicData", spec.GRAMMAR_VERSION)
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as unpacked:
        [member] = unpacked.getmembers()
    typeflag = member.type.decode("ascii")
    # The mode, uid and gid fields of that member's ustar header as `build_tarball` wrote them: seven
    # octal digits at the offsets the header layout fixes (tests/test_g4spec.py reads the same slices).
    header = gzip.decompress(archive)[:512]
    encoded_mode, encoded_uid, encoded_gid = (header[at:at + 7].decode("ascii") for at in (100, 108, 116))
    assert encoded_uid == encoded_gid, (encoded_uid, encoded_gid)
    gzip_header = emit.gzip_header(archive)
    [level] = re.findall(r"\bcompresslevel=(\d+)", inspect.getsource(emit.build_tarball))
    place = {name: line for line, name in enumerate(spec.DIRECTIVE_ORDER, 1)}

    return [
        Pin("permitted byte, TAB, section 2.1", FORMAT_SPEC,
            rf"\*\*`({hexa})` \(TAB\), `{hexa}` \(LF\)", (1,), tab),
        Pin("permitted byte, LF, section 2.1", FORMAT_SPEC,
            rf"`{hexa}` \(TAB\), `({hexa})` \(LF\)", (1,), lf),
        Pin("permitted byte, CR, section 2.1", FORMAT_SPEC, rf"\(LF\), `({hexa})` \(CR\)", (1,), cr),
        Pin("permitted range, low end, section 2.1", FORMAT_SPEC,
            rf"\(CR\), and `({hexa})`-`{hexa}`\*\*", (1,), f"0x{low:02X}"),
        Pin("permitted range, high end, section 2.1", FORMAT_SPEC,
            rf"\(CR\), and `{hexa}`-`({hexa})`\*\*", (1,), f"0x{high:02X}"),
        Pin("permitted range, low end, the E005 row", FORMAT_SPEC,
            rf"`\{{TAB, LF, CR, ({hexa})-{hexa}\}}`", (1,), f"0x{low:02X}"),
        Pin("permitted range, high end, the E005 row", FORMAT_SPEC,
            rf"`\{{TAB, LF, CR, {hexa}-({hexa})\}}`", (1,), f"0x{high:02X}"),
        Pin("field separator, space, section 2.3", FORMAT_SPEC,
            rf"\*\*space \(`({hexa})`\) and tab", (1,), space),
        Pin("field separator, tab, section 2.3", FORMAT_SPEC,
            rf"\) and tab \(`({hexa})`\) only\*\*", (1,), separator_tab),
        Pin("nonzero digits, low end, section 2.3", FORMAT_SPEC,
            r"no digit `(\d)`-`\d` before the exponent", (1,), chr(first)),
        Pin("nonzero digits, high end, section 2.3", FORMAT_SPEC,
            r"no digit `\d`-`(\d)` before the exponent", (1,), chr(last)),
        Pin("nonzero digits, low end, section 6", FORMAT_SPEC,
            r"\(a digit `(\d)`-`\d` before the exponent\)", (1,), chr(first)),
        Pin("nonzero digits, high end, section 6", FORMAT_SPEC,
            r"\(a digit `\d`-`(\d)` before the exponent\)", (1,), chr(last)),
        Pin("Layer-2 key pattern, both key columns, section 3", FORMAT_SPEC,
            r"then `A` \S+ `([^`]+)` \|", (1,), both_keys),
        Pin("Layer-2 key pattern, one key column, section 3", FORMAT_SPEC,
            r"that column's integer \S+ `([^`]+)` \|", (1,), one_key),
        Pin("float precision, section 2.6", FORMAT_SPEC, r"Floats are written with `%\.(\d+)g`", (1,),
            precision),
        Pin("float precision in words, section 2.6", FORMAT_SPEC,
            r"IEEE-754 double\. (\w+) significant decimal digits", (1,), precision),
        Pin("float precision, section 7", FORMAT_SPEC, r"the `%\.(\d+)g` float syntax of section", (1,),
            precision),
        Pin("the smallest subnormal as emitted, section 2.3", FORMAT_SPEC,
            r"subnormal\*\*: `([^`]+)` is representable", (1,), subnormal),
        Pin("the smallest subnormal as emitted, section 6", FORMAT_SPEC,
            r"`[^`]+` is emitted as `([^`]+)`\);", (1,), subnormal),
        Pin("the entered value as emitted, section 2.6", FORMAT_SPEC,
            r"is emitted as `([^`]+)`: the file records", (1,), emitted),
        Pin("the entered value as emitted, section 6", FORMAT_SPEC,
            r'`strtod\("([^"]+)"\)` stops', (1,), emitted),
        Pin("Layer-2 indent in words, section 3", FORMAT_SPEC,
            r"at every level, (\w+)-space indentation", (1,), indent),
        Pin("Layer-2 trailing newlines, section 3", FORMAT_SPEC, r"exactly (\w+) trailing newline\.", (1,),
            newlines),
        Pin("Layer-2 indent, the digest invariant", FORMAT_SPEC, r"indent=(\d+), ensure_ascii=True\)",
            (1,), indent),
        Pin("member uid, section 8", FORMAT_SPEC, r"\| tar \| `uid`, `gid` \| `(\d+)`, `\d+` \|", (1,),
            member.uid),
        Pin("member gid, section 8", FORMAT_SPEC, r"\| tar \| `uid`, `gid` \| `\d+`, `(\d+)` \|", (1,),
            member.gid),
        Pin("member typeflag, section 8", FORMAT_SPEC, r"\| tar \| typeflag \| the byte `'(.)'`", (1,),
            typeflag),
        Pin("member typeflag in hexadecimal, section 8", FORMAT_SPEC,
            rf"the byte `'.'` \(`({hexa})`\)", (1,), f"0x{ord(typeflag):02X}"),
        Pin("member mode as encoded, section 8", FORMAT_SPEC,
            r"as 7 digits \+ NUL \(`(\d+)`, `\d+`\)", (1,), encoded_mode),
        Pin("member uid and gid as encoded, section 8", FORMAT_SPEC,
            r"as 7 digits \+ NUL \(`\d+`, `(\d+)`\)", (1,), encoded_uid),
        Pin("gzip compression level, section 8", FORMAT_SPEC,
            r"\| gzip \| compression level \| `(\d+)` with", (1,), int(level)),
        Pin("gzip XFL byte, section 8", FORMAT_SPEC, r"and therefore `XFL` = `(\d+)`", (1,),
            gzip_header["xfl"]),
        Pin("gzip OS byte, section 8", FORMAT_SPEC, r"\| gzip \| `OS` byte \| \*\*(\d+)\*\*", (1,),
            gzip_header["os"]),
        Pin("the #GRAMMAR line, section 4's lexical example", FORMAT_SPEC,
            r"an unreadable `#GRAMMAR` on line (\d+) reports", (1,), place["GRAMMAR"]),
        Pin("the #PROFILE line, section 4's priority example", FORMAT_SPEC,
            r"an `E013` whose fault line is (\d+)\.", (1,), place["PROFILE"]),
        Pin("the #SOURCEDIGEST line, section 4's priority example", FORMAT_SPEC,
            r"an `E016` on line (\d+) is reported ahead", (1,), place["SOURCEDIGEST"]),
        Pin("the #PROFILE line, section 4's preemption example", FORMAT_SPEC,
            r"whose `#PROFILE` on line (\d+) lacks", (1,), place["PROFILE"]),
        Pin("the E013 line, section 4's preemption example", FORMAT_SPEC,
            r"not `E013` on line (\d+), because", (1,), place["PROFILE"]),
    ]


PAPER = "paper/muonic-data/paper.md"


def _d3_expected(what: str) -> object:
    """The value `tests/test_g4d3.py` computes for the `DATASET_D3.md` pin it labels `what` --
    looked up, never retyped."""
    hits = [row[4] for row in d3.document_pins() if row[0] == what]
    assert len(hits) == 1, (what, len(hits))
    return hits[0]


def paper_pins(pins: parity.DocumentPins, check_f3) -> list[Pin]:
    """The draft paper's numbers. Each restates a value another pin already holds its home
    document to; every `expected` is looked up from the module that computes it."""
    max_ulp = check_f3.document_figures(REPO / "DATASET_D1.md")[3]
    swept = _computed(pins, "swept points at zero ulp")
    return [
        Pin("capture record count, the paper", PAPER,
            r"(\d+) `\{Z, A, rate, error\}` records", (1,), _computed(pins, "capture record count")),
        Pin("effective-charge entries, the paper", PAPER,
            r"records and a (\d+)-value effective-charge table", (1,),
            _computed(pins, "effective-charge record count")),
        Pin("swept points, the paper's box", PAPER,
            r"over a (\d+)-point box", (1,), swept),
        Pin("swept points returning a negative rate, the paper", PAPER,
            r"negative capture rates on (\d+) of the \d+ swept points", (1,),
            _computed(pins, "swept points returning a negative rate")),
        Pin("swept points, the paper's finding", PAPER,
            r"negative capture rates on \d+ of the (\d+) swept points", (1,), swept),
        Pin("F-3 maximum, the paper", PAPER,
            r"moves its result by up to (\d+) ulp", (1,), max_ulp),
        Pin("the D3 tolerance factor, the paper", PAPER,
            r"at a tolerance of (\d+) times the printed standard", (1,),
            _d3_expected("the tolerance factor")),
        Pin("D3 gated rows, the paper", PAPER,
            r"of the (\d+) gated rows, the line MuDirac prints", (1,),
            _d3_expected("gated rows in the projection")),
        Pin("D3 rows whose solver line lies within the band, the paper", PAPER,
            r"the line MuDirac prints lies within it for (\d+),", (1,),
            _d3_expected("rows whose solver line lies within the band")),
        Pin("D3 rows whose shell difference lies within the band, the paper", PAPER,
            r"the shell difference the patched cascade receives for (\d+),", (1,),
            _d3_expected("rows whose shell difference lies within the band")),
        Pin("D3 rows whose unpatched cascade lies within the band, the paper", PAPER,
            r"the energy the unpatched cascade emits for (\d+)\.", (1,),
            _d3_expected("rows whose unpatched cascade lies within the band")),
    ]


def d3_seam_size_pins(vendored_readme: str) -> list[Pin]:
    """The vendored README's size cells of the cascade and decay copies: bytes and newlines of each
    vendored file, the v11.4.2 rows labelled by file name and the beta rows by tag and file name."""
    out = []
    for tag, directory in parity.D3_SEAM_DIRS.items():
        for name in sorted(parity.D3_SEAM_BLOB_IDS[tag]):
            data = (directory / name).read_bytes()
            label = re.escape(f"`{name}`" if tag == parity.d1.UPSTREAM_TAG else f"`{tag}/{name}`")
            out.append(Pin(f"vendored {tag}/{name} size, bytes", vendored_readme,
                           rf"\| {label} size \| (\d+) bytes, \d+ lines \|", (1,), len(data)))
            out.append(Pin(f"vendored {tag}/{name} size, lines", vendored_readme,
                           rf"\| {label} size \| \d+ bytes, (\d+) lines \|", (1,), data.count(b"\n")))
    return out


@dataclasses.dataclass(frozen=True)
class PinProblem:
    pin: Pin
    detail: str


def pin_spans(
    texts: dict[str, str], pins: list[Pin]
) -> tuple[dict[str, set[tuple[int, int]]], list[PinProblem]]:
    """The raw `(line, col)` positions every pin covers in `texts`, and every pin that did not match
    exactly once or did not state its expected value. A pin whose file is absent from `texts` is
    skipped: the check is over the files it was handed."""
    covered: dict[str, set[tuple[int, int]]] = {path: set() for path in texts}
    problems: list[PinProblem] = []
    collapsed = {path: collapse(text) for path, text in texts.items()}
    for pin in pins:
        if pin.path not in texts:
            continue
        doc, origin = collapsed[pin.path]
        hits = list(re.finditer(pin.pattern, doc))
        if len(hits) != 1:
            problems.append(PinProblem(pin, f"matched {len(hits)} times, expected exactly one"))
            continue
        hit = hits[0]
        if pin.expected is not None:
            stated: object
            expected: object
            if isinstance(pin.expected, str):
                stated, expected = hit.group(1), pin.expected
            elif pin.places is None:
                stated, expected = parity._stated(hit.group(1)), pin.expected
            else:
                stated, expected = float(hit.group(1)), round(float(pin.expected), pin.places)
            if stated != expected:
                problems.append(PinProblem(
                    pin, f"states {hit.group(1)!r}; the shipped data says {expected}"
                ))
        for group in pin.groups:
            start, end = hit.span(group)
            for index in range(start, end):
                position = origin[index]
                if position is not None:
                    covered[pin.path].add(position)
    return covered, problems


# --------------------------------------------------------------------------------------------
# Door 2 -- classes
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ClassRow:
    id: str
    glob: str
    regex: re.Pattern[str]
    reason: str
    example: str


def read_classes(path: pathlib.Path = CLASSES) -> list[ClassRow]:
    """Rows of `id glob regex reason example`; the header names the columns, `#` lines comment."""
    rows: list[ClassRow] = []
    lines = [raw for raw in path.read_text(encoding="utf-8").splitlines()
             if raw.strip() and not raw.startswith("#")]
    assert lines and lines[0].split("\t") == ["id", "glob", "regex", "reason", "example"], (
        f"{path.name} must open with the header row"
    )
    for raw in lines[1:]:
        parts = raw.split("\t")
        assert len(parts) == 5, f"class row must have 5 tab-separated fields: {raw!r}"
        row_id, glob, regex, reason, example = parts
        assert row_id not in {r.id for r in rows}, f"duplicate class id {row_id}"
        assert reason.strip(), f"class {row_id} has no reason"
        rows.append(ClassRow(row_id, glob, re.compile(regex), reason, example))
    return rows


def class_admitting(token: Token, line: str, classes: list[ClassRow]) -> ClassRow | None:
    """The first class row whose glob matches the file and whose regex matches around the token."""
    for row in classes:
        if not fnmatch.fnmatchcase(token.path, row.glob):
            continue
        for match in row.regex.finditer(line):
            if match.start() <= token.col and token.end <= match.end():
                return row
    return None


# --------------------------------------------------------------------------------------------
# Door 3 -- the registry
# --------------------------------------------------------------------------------------------


def _normalize(line: str) -> str:
    return " ".join(line.split())


def claim_sha1(line: str) -> str:
    return hashlib.sha1(_normalize(line).encode("utf-8"), usedforsecurity=False).hexdigest()


def read_registry(path: pathlib.Path = REGISTRY) -> dict[tuple[str, str], str]:
    """{(path, sha1): status}. Three tab-separated fields per row: sha1, path, status.

    The registry carries no copy of the matched text: that lives in the file the path names, and
    the enumerator prints it on failure. Rows are kept sorted by (path, sha1) so a row has one
    place to be.
    """
    rows: dict[tuple[str, str], str] = {}
    keys: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        assert len(parts) == 3, f"registry row must have 3 tab-separated fields: {raw!r}"
        sha, rel, status = parts
        assert (rel, sha) not in rows, f"duplicate registry row for {(rel, sha)}"
        rows[(rel, sha)] = status
        keys.append((rel, sha))
    assert keys == sorted(keys), f"{path.name} rows are not sorted by (path, sha1)"
    return rows


def status_tokens(status: str) -> tuple[str, set[str], str]:
    """`(kind, tokens named, tail)` of a status: `EXERCISED: t1, t2 -- tests/f.py::test` or
    `REGISTERED: t1, t2 -- reason`; `UNREVIEWED` names nothing."""
    if status == "UNREVIEWED":
        return "UNREVIEWED", set(), ""
    assert status.startswith(VALID_PREFIXES), (
        f"status must be UNREVIEWED or start with {VALID_PREFIXES}: {status!r}"
    )
    kind, _, rest = status.partition(":")
    named, sep, tail = rest.strip().partition(" -- ")
    assert sep and tail.strip(), f"status must read '<tokens> -- <reason or test>': {status!r}"
    tokens = {t.strip() for t in named.split(", ")}
    assert tokens and all(tokens), f"status names no token: {status!r}"
    return kind, tokens, tail.strip()


def check_registry_form(registry: dict[tuple[str, str], str]) -> list[str]:
    """Statuses well formed, an EXERCISED row naming a test that exists, no row for a
    registry-free file, and UNREVIEWED under its ceiling."""
    problems: list[str] = []
    for (rel, sha), status in sorted(registry.items()):
        if rel in REGISTRY_FREE:
            problems.append(f"{sha} {rel}: this file may carry no registry row")
        if rel not in PROSE_PATHS:
            problems.append(f"{sha} {rel}: not a file this check reads")
        kind, _, tail = status_tokens(status)
        if kind != "EXERCISED":
            continue
        file_part, _, func = tail.partition("::")
        target = REPO / file_part
        if not target.is_file():
            problems.append(f"{sha} {rel}: EXERCISED names a missing file: {tail}")
        elif not func or f"def {func}(" not in target.read_text(encoding="utf-8"):
            problems.append(f"{sha} {rel}: EXERCISED names a test that does not exist: {tail}")
    unreviewed = sorted(k for k, st in registry.items() if st == "UNREVIEWED")
    if len(unreviewed) > UNREVIEWED_CEILING:
        problems.append(
            f"{len(unreviewed)} UNREVIEWED rows exceed the ceiling {UNREVIEWED_CEILING}; the "
            "ceiling is monotone non-increasing -- rule the rows instead of raising it"
        )
    return problems


# --------------------------------------------------------------------------------------------
# Door 3, walked -- a reason's homes resolve, and the figures it types are carried (T-78)
# --------------------------------------------------------------------------------------------

HOME_SUFFIXES = "md|py|csv|json|tsv|bib|txt|cc|hh|yml|yaml|toml|snippet|oracle|g4dat|cmake|patch|ipynb|mac"
#: A file name in a reason: path segments of word characters, `.`, `+` and `-`, ending in one of
#: the suffixes above, optionally followed by `:<line>`, and not glued to a longer path or word.
HOME = re.compile(r"(?<![\w/.+-])((?:[\w.+-]+/)*[\w.+-]+\.(?:" + HOME_SUFFIXES + r"))(?::\d+)?(?![\w/])")
_PRUNED_DIRS = {"dist", "__pycache__", "htmlcov", "node_modules"}
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _walked(directory: str) -> bool:
    """Directories no tracked file lives in are pruned: `.git`, virtual environments, build trees,
    tool caches, egg metadata and the usual output directories."""
    return not (
        directory == ".git"
        or directory.startswith(".venv")
        or directory.startswith("build")
        or directory.endswith("_cache")
        or directory.endswith(".egg-info")
        or directory in _PRUNED_DIRS
    )


@functools.lru_cache(maxsize=1)
def tree_files() -> tuple[str, ...]:
    """Every file under the repository as a POSIX path relative to it, pruned by `_walked`. In a
    fresh clone this is `git ls-files`; walking rather than asking git keeps the check free of a
    `git` binary, like the blob computation in `test_g4parity.py`."""
    out: list[str] = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = sorted(d for d in dirs if _walked(d))
        for name in sorted(files):
            out.append((pathlib.Path(root) / name).relative_to(REPO).as_posix())
    return tuple(out)


def resolve_home(name: str, files: tuple[str, ...]) -> str | None:
    """`name` when it is a file of the tree; else the one file whose path ends in `/name`; else
    `None` -- for a name no file carries and, alike, for one that several files carry."""
    if name in files:
        return name
    hits = [path for path in files if path.endswith("/" + name)]
    return hits[0] if len(hits) == 1 else None


def figure_present(token: str, text: str) -> bool:
    """`token` occurs in `text` as a whole figure: a superscript run bounded by non-superscripts, a
    digit run bounded by neither digit nor `.` (so `56` is not found inside `56.7`), or a spelled
    number as a whole word, case-insensitively."""
    if token[0] in _SUPERSCRIPTS:
        pattern = f"(?<![{_SUPERSCRIPTS}])" + re.escape(token) + f"(?![{_SUPERSCRIPTS}])"
        flags = 0
    elif token[0].isdigit():
        pattern = r"(?<![\d.])" + re.escape(token) + r"(?![\d.])"
        flags = 0
    else:
        pattern = r"\b" + re.escape(token) + r"\b"
        flags = re.IGNORECASE
    return re.search(pattern, text, flags) is not None


def check_registry_homes(
    registry: dict[tuple[str, str], str], texts: dict[str, str], classes: list[ClassRow]
) -> list[str]:
    """For every row: each file name its reason gives resolves to one file of the tree, and each
    figure its reason types -- a token `tokenize` finds that the row does not rule and no class
    admits -- stands verbatim in the ruled line or, when it begins with a digit or a superscript,
    in a file the reason names. A spelled figure is admitted by the ruled set or the ruled line
    only: a home's prose carries small spelled numbers by accident (`one`, `two`, `three` occur in
    nearly every document), so a home vouches for nothing spelled. A figure carried by neither is
    a restatement nothing checks, and is named."""
    files = tree_files()
    lines_by_key: dict[tuple[str, str], str] = {}
    for path, text in texts.items():
        for line in text.splitlines():
            lines_by_key.setdefault((path, claim_sha1(line)), line)
    cache: dict[str, str] = {}

    def home_text(rel: str) -> str:
        if rel not in cache:
            cache[rel] = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        return cache[rel]

    problems: list[str] = []
    for (path, sha), status in sorted(registry.items()):
        _, ruled, reason = status_tokens(status)
        ruled_lower = {t.lower() for t in ruled}
        homes: list[str] = []
        for match in HOME.finditer(reason):
            home = resolve_home(match.group(1), files)
            if home is None:
                problems.append(f"{sha} {path}: names a home that does not resolve: {match.group(1)}")
            else:
                homes.append(home)
        line = lines_by_key.get((path, sha), "")
        for token in tokenize(path, reason):
            if token.text.lower() in ruled_lower:
                continue
            if class_admitting(token, reason, classes) is not None:
                continue
            if figure_present(token.text, line):
                continue
            if (token.text[0].isdigit() or token.text[0] in _SUPERSCRIPTS) and any(
                figure_present(token.text, home_text(home)) for home in homes
            ):
                continue
            problems.append(f"{sha} {path}: types {token.text}, carried by no home it names")
    return problems


# --------------------------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Miss:
    token: Token
    line: str

    def __str__(self) -> str:
        return f"{self.token.path}:{self.token.lineno} {self.token.text} | {self.line.strip()}"


def tree_texts() -> dict[str, str]:
    return {rel: (REPO / rel).read_text(encoding="utf-8") for rel in PROSE_PATHS}


def pinned(token: Token, covered: set[tuple[int, int]]) -> bool:
    """Every digit of the token lies inside a pinned span. The separators a token may carry
    (`1,120`, `1..300`, `3.5`) are not digits, so two adjacent pins admit the pair they join and
    a token with one unpinned digit is refused."""
    digits = {
        (token.lineno, token.col + offset)
        for offset, ch in enumerate(token.text)
        if ch not in ",."
    }
    return digits <= covered


def unadmitted_by_pins_and_classes(
    texts: dict[str, str], pins: list[Pin], classes: list[ClassRow]
) -> tuple[dict[tuple[str, int], list[Token]], list[PinProblem]]:
    """{(path, lineno): tokens} for every line carrying a token neither door 1 nor door 2 admits."""
    covered, problems = pin_spans(texts, pins)
    open_tokens: dict[tuple[str, int], list[Token]] = {}
    for path, text in texts.items():
        lines = text.splitlines()
        for token in tokenize(path, text):
            if pinned(token, covered[path]):
                continue
            if class_admitting(token, lines[token.lineno - 1], classes) is not None:
                continue
            open_tokens.setdefault((path, token.lineno), []).append(token)
    return open_tokens, problems


def unadmitted(
    texts: dict[str, str],
    pins: list[Pin],
    classes: list[ClassRow],
    registry: dict[tuple[str, str], str],
) -> tuple[list[Miss], list[str]]:
    """Every token no door admits, and every registry problem: a stale row (its line no longer
    carries an open token), a row naming a token its line does not leave open, an unregistered
    line."""
    open_tokens, _ = unadmitted_by_pins_and_classes(texts, pins, classes)
    lines = {path: text.splitlines() for path, text in texts.items()}
    enumerated = {(path, claim_sha1(lines[path][lineno - 1])): (path, lineno)
                  for path, lineno in open_tokens}
    misses: list[Miss] = []
    problems: list[str] = []
    for key in sorted(registry):
        if key not in enumerated:
            problems.append(f"stale registry row -- delete it: {key[1]} {key[0]}")
    for (path, lineno), tokens in sorted(open_tokens.items()):
        line = lines[path][lineno - 1]
        key = (path, claim_sha1(line))
        status = registry.get(key)
        named = status_tokens(status)[1] if status is not None else set()
        carried = {t.text for t in tokens}
        for token in tokens:
            if token.text not in named:
                misses.append(Miss(token, line))
        for extra in sorted(named - carried):
            problems.append(f"{key[1]} {path}:{lineno} names {extra!r}, which the line does not leave open")
    return misses, problems


def check_tree(
    texts: dict[str, str] | None = None,
) -> tuple[list[Miss], list[str], list[PinProblem]]:
    texts = tree_texts() if texts is None else texts
    pins = pin_table()
    _, pin_problems = pin_spans(texts, pins)
    registry = read_registry()
    classes = read_classes()
    misses, problems = unadmitted(texts, pins, classes, registry)
    problems = check_registry_form(registry) + problems + check_registry_homes(registry, texts, classes)
    return misses, problems, pin_problems


# --------------------------------------------------------------------------------------------
# T-74 -- the tree
# --------------------------------------------------------------------------------------------


def test_t74_every_number_in_the_documents_prose_paths_names_is_computed_or_listed_with_a_reason():
    misses, problems, pin_problems = check_tree()
    assert not pin_problems, "\n".join(
        f"{p.pin.path}: {p.pin.what}: {p.detail} -- pattern {p.pin.pattern!r}" for p in pin_problems
    )
    assert not problems, "\n".join(problems)
    assert not misses, (
        f"{len(misses)} token(s) neither pinned, class-admitted nor registered:\n"
        + "\n".join(str(m) for m in misses)
    )


# --------------------------------------------------------------------------------------------
# T-75 -- the drills
# --------------------------------------------------------------------------------------------


def _drill(texts: dict[str, str]) -> tuple[list[Miss], list[str]]:
    return unadmitted(texts, pin_table(), read_classes(), read_registry())


def test_t75_drill_a_spelled_number_decoy_is_named():
    texts = tree_texts()
    texts["DATASET_D1.md"] += "\nthe table has ninety-one records\n"
    lineno = len(texts["DATASET_D1.md"].splitlines())
    misses, _ = _drill(texts)
    named = {(m.token.path, m.token.lineno, m.token.text) for m in misses}
    assert ("DATASET_D1.md", lineno, "ninety-one") in named, sorted(named)
    assert ("DATASET_D1.md", lineno, "one") not in named, "the compound was split"


def test_t75_drill_an_unpinned_digit_beside_a_pinned_one_is_named():
    texts = tree_texts()
    original = texts["cpp/tools/README.md"]
    # The pinned figure is read from the file through the F-3 checker, never typed here.
    max_ulp = _load_check_f3().readme_figures(REPO / "cpp" / "tools" / "README.md")[0]
    assert original.count(f"{max_ulp} ulp") == 1
    # Beside the pin, as the attack states it: the figure pattern itself then no longer matches
    # and every figure on the line surfaces, the decoy among them.
    texts["cpp/tools/README.md"] = original.replace(
        f"{max_ulp} ulp", f"{max_ulp} ulp and {max_ulp + 1} ulp"
    )
    lineno = next(
        i for i, line in enumerate(original.splitlines(), 1) if f"{max_ulp} ulp" in line
    )
    misses, _ = _drill(texts)
    assert ("cpp/tools/README.md", lineno, str(max_ulp + 1)) in {
        (m.token.path, m.token.lineno, m.token.text) for m in misses
    }
    # Past the pin's span, so the pin still matches: the decoy is then the only new failure.
    texts["cpp/tools/README.md"] = original.replace(
        "swept points", f"swept points and {max_ulp + 1} ulp", 1
    )
    misses, _ = _drill(texts)
    assert [(m.token.path, m.token.lineno, m.token.text) for m in misses] == [
        ("cpp/tools/README.md", lineno, str(max_ulp + 1))
    ], [str(m) for m in misses]


def test_t75_drill_a_changed_digit_in_a_registered_line_goes_stale_and_unadmitted():
    registry = read_registry()
    texts = tree_texts()
    lines = {path: text.splitlines() for path, text in texts.items()}
    # Any registered line carrying a registered digit token will do; the first in file order.
    chosen = None
    for path in PROSE_PATHS:
        for lineno, line in enumerate(lines[path], 1):
            status = registry.get((path, claim_sha1(line)))
            if status is None:
                continue
            digits = [t for t in status_tokens(status)[1] if t[0].isdigit() and t in line]
            if digits:
                chosen = (path, lineno, line, digits[0])
                break
        if chosen:
            break
    assert chosen, "no registered line carries a digit token -- the drill has nothing to corrupt"
    path, lineno, line, token = chosen
    flipped = str((int(token[0]) + 1) % 10) + token[1:]
    mutated = line.replace(token, flipped, 1)
    lines[path][lineno - 1] = mutated
    texses = dict(texts)
    texses[path] = "\n".join(lines[path]) + "\n"
    misses, problems = _drill(texses)
    assert any(p.startswith("stale registry row") and claim_sha1(line) in p for p in problems), problems
    assert (path, lineno, flipped) in {(m.token.path, m.token.lineno, m.token.text) for m in misses}


def test_t75_every_class_row_matches_its_example_and_admits_a_live_token():
    classes = read_classes()
    texts = tree_texts()
    admitted: dict[str, int] = {row.id: 0 for row in classes}
    for row in classes:
        assert row.regex.search(row.example), f"class {row.id}: its example does not match its regex"
        assert re.search(r"[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]", row.example) or SPELLED.search(row.example), (
            f"class {row.id}: its example carries no token"
        )
    covered, _ = pin_spans(texts, pin_table())
    for path, text in texts.items():
        lines = text.splitlines()
        for token in tokenize(path, text):
            if pinned(token, covered[path]):
                continue
            row = class_admitting(token, lines[token.lineno - 1], classes)
            if row is not None:
                admitted[row.id] += 1
    idle = sorted(row_id for row_id, n in admitted.items() if n == 0)
    assert not idle, f"class rows admitting no live token -- delete them: {idle}"


def test_t75_every_pin_pattern_matches_exactly_once():
    _, problems = pin_spans(tree_texts(), pin_table())
    assert not problems, "\n".join(
        f"{p.pin.path}: {p.pin.what}: {p.detail}" for p in problems
    )


def test_t75_drill_a_format_spec_bound_that_disagrees_with_the_package_is_named():
    """Raise the integer-column bound section 2.3 of `FORMAT_SPEC.md` states by one in an in-memory
    copy: the pin holding it to `openmucf.g4.spec.INTEGER_MAX` names the disagreement, and no other
    pin moves."""
    from openmucf.g4 import spec

    texts = tree_texts()
    stated = f"`{spec.INTEGER_MIN}`-`{spec.INTEGER_MAX}` inclusive"
    assert texts[FORMAT_SPEC].count(stated) == 1
    texts[FORMAT_SPEC] = texts[FORMAT_SPEC].replace(
        stated, f"`{spec.INTEGER_MIN}`-`{spec.INTEGER_MAX + 1}` inclusive"
    )
    _, problems = pin_spans(texts, pin_table())
    assert [p.pin.what for p in problems] == ["integer upper bound, section 2.3"], [
        f"{p.pin.what}: {p.detail}" for p in problems
    ]


def _wrong(stated: str) -> str:
    """A different value of the same kind and width as `stated`: a number one higher (wrapping), a
    hexadecimal byte one higher, another spelled number, or -- inside a pattern or a literal -- its
    first digit moved."""
    if stated.isdigit():
        return str((int(stated) + 1) % 10 ** len(stated)).zfill(len(stated))
    if re.fullmatch(r"0x[0-9A-F]{2}", stated):
        return f"0x{int(stated, 16) + 1:02X}"
    if stated.isalpha():
        return "three" if stated.lower() != "three" else "four"
    digit = re.search(r"\d", stated)
    assert digit is not None, stated
    return stated[: digit.start()] + str((int(digit[0]) + 1) % 10) + stated[digit.end():]


def test_t75_drill_each_derived_format_spec_figure_that_disagrees_with_the_package_is_named():
    """For every pin of `format_spec_derived_pins`, state a different value where its figure stands
    in an in-memory copy of `FORMAT_SPEC.md`: exactly that pin names the disagreement."""
    texts = tree_texts()
    pins = format_spec_derived_pins()
    doc, origin = collapse(texts[FORMAT_SPEC])
    for pin in pins:
        [hit] = re.finditer(pin.pattern, doc)
        start, end = hit.span(1)
        (line, col), (last_line, last_col) = origin[start], origin[end - 1]
        assert line == last_line, pin.what
        lines = texts[FORMAT_SPEC].split("\n")
        lines[line - 1] = lines[line - 1][:col] + _wrong(hit[1]) + lines[line - 1][last_col + 1:]
        _, problems = pin_spans({FORMAT_SPEC: "\n".join(lines)}, pins)
        assert [p.pin.what for p in problems] == [pin.what], [f"{p.pin.what}: {p.detail}" for p in problems]


def _rebuilt_tarball(members, *, directory):
    """A writer that stores its members owned by uid and gid 1, with the NUL typeflag, at
    compression level 6 (so another XFL byte) -- each a legal choice section 8 excludes."""
    import gzip
    import io
    import tarfile

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(f"{directory}/{name}")
            info.size = len(payload)
            info.uid = info.gid = 1
            info.type = tarfile.AREGTYPE
            archive.addfile(info, io.BytesIO(payload))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=6, mtime=0, filename="") as stream:
        stream.write(raw.getvalue())
    return compressed.getvalue()


def _drill_package(monkeypatch, case: str) -> None:
    import json

    from openmucf.g4 import emit, provenance, spec

    if case == "permitted bytes":
        monkeypatch.setattr(spec, "_FORBIDDEN_BYTE", re.compile(r"[^\t\n\r\x20-\x7F]"))
    elif case == "field separators":
        monkeypatch.setattr(spec, "_FIELD_SEPARATOR", re.compile(r"[\x0b\t]+"))
    elif case == "nonzero digits":
        monkeypatch.setattr(spec, "_NONZERO_DIGIT", re.compile(r"[2-9]"))
    elif case == "row keys":
        monkeypatch.setattr(provenance, "_ROW_KEY_PATTERN", re.compile(r"^([0-9]+)(?:-([0-9]+))?$"))
    elif case == "float syntax":
        monkeypatch.setattr(spec, "format_float", lambda x: f"{float(x):.16g}")
    elif case == "Layer-2 layout":
        monkeypatch.setattr(provenance, "render_json", lambda document: json.dumps(
            provenance.to_json_obj(document), sort_keys=True, indent=4, ensure_ascii=True) + "\n\n")
    elif case == "archive":
        monkeypatch.setattr(emit, "build_tarball", _rebuilt_tarball)
    elif case == "directive order":
        order = [name for name in spec.DIRECTIVE_ORDER if name != "PROFILE"]
        order.insert(order.index("SEAM") + 1, "PROFILE")
        monkeypatch.setattr(spec, "DIRECTIVE_ORDER", tuple(order))
    elif case == "member mode":
        monkeypatch.setattr(emit, "_MEMBER_MODE", 0o600)
    elif case == "digest line":
        order = list(spec.DIRECTIVE_ORDER)
        first, second = order.index("GENERATOR"), order.index("SOURCEDIGEST")
        order[first], order[second] = order[second], order[first]
        monkeypatch.setattr(spec, "DIRECTIVE_ORDER", tuple(order))
    else:
        raise AssertionError(case)


#: Each in-memory change of `_drill_package`, and the derived pins that must then name it, in order.
PACKAGE_DRILLS = {
    "permitted bytes": ["permitted range, high end, section 2.1", "permitted range, high end, the E005 row"],
    "field separators": ["field separator, space, section 2.3"],
    "nonzero digits": ["nonzero digits, low end, section 2.3", "nonzero digits, low end, section 6"],
    "row keys": ["Layer-2 key pattern, both key columns, section 3",
                 "Layer-2 key pattern, one key column, section 3"],
    "float syntax": ["float precision, section 2.6", "float precision in words, section 2.6",
                     "float precision, section 7", "the smallest subnormal as emitted, section 2.3",
                     "the smallest subnormal as emitted, section 6",
                     "the entered value as emitted, section 2.6", "the entered value as emitted, section 6"],
    "Layer-2 layout": ["Layer-2 indent in words, section 3", "Layer-2 trailing newlines, section 3",
                       "Layer-2 indent, the digest invariant"],
    "archive": ["member uid, section 8", "member gid, section 8", "member typeflag, section 8",
                "member typeflag in hexadecimal, section 8", "member uid and gid as encoded, section 8",
                "gzip compression level, section 8", "gzip XFL byte, section 8"],
    "member mode": ["member mode as encoded, section 8"],
    "digest line": ["the #SOURCEDIGEST line, section 4's priority example"],
    "directive order": ["the #PROFILE line, section 4's priority example",
                        "the #PROFILE line, section 4's preemption example",
                        "the E013 line, section 4's preemption example"],
}


def test_t75_drill_a_package_value_a_derived_format_spec_pin_reads_is_named(monkeypatch):
    """Change, in memory, each value in `openmucf.g4` a derived pin reads: exactly the pins that
    restate it name the disagreement."""
    texts = tree_texts()
    for case, named in PACKAGE_DRILLS.items():
        _drill_package(monkeypatch, case)
        _, problems = pin_spans(texts, format_spec_derived_pins())
        monkeypatch.undo()
        assert [p.pin.what for p in problems] == named, (
            case, [f"{p.pin.what}: {p.detail}" for p in problems]
        )


def test_t75_the_enumerator_prints_nothing_on_the_tree():
    result = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__)), "--tokens"],
        capture_output=True, text=True, cwd=REPO, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", result.stdout


def test_t75_drill_check_f3_refuses_a_document_whose_relative_difference_disagrees_at_two_figures(tmp_path):
    """A producer line built from the live document's own figures passes `--exact`; the same line
    with its relative difference doubled is refused -- so the figure is compared, not only printed."""
    check_f3 = _load_check_f3()
    document = REPO / "DATASET_D1.md"
    points, differ, over_one, max_ulp, z, a = check_f3.document_figures(document)
    rel = check_f3.document_max_rel(document)
    producer = tmp_path / "f3_output.txt"

    def run(max_rel: str) -> int:
        producer.write_text(
            f"F3 cxx=GNU points={points} differ={differ} over1ulp={over_one} max_ulp={max_ulp} "
            f"at=({z},{a}) max_rel={max_rel}\n", "utf-8",
        )
        return check_f3.main(["--producer-output", str(producer), "--document", str(document), "--exact"])

    assert run(rel) == 0
    assert run(f"{float(rel) * 2:.3e}") == 1


# --------------------------------------------------------------------------------------------
# T-78 -- the registry reasons, walked; T-79 -- its drill
# --------------------------------------------------------------------------------------------


def test_t78_every_registry_reason_resolves_its_homes_and_types_no_figure_they_do_not_carry():
    problems = check_registry_homes(read_registry(), tree_texts(), read_classes())
    assert not problems, "\n".join(problems)


def test_t79_drill_a_reason_naming_a_missing_home_an_ambiguous_home_or_an_unfound_figure_is_named():
    """Four one-row copies of the registry, each planted with one defect the walk must name: a
    file no tree carries, a bare name several files carry (picked from the tree, never typed),
    the spelled decoy T-75 also uses, and a spelled figure that a named home's text carries but
    the ruled line does not (picked from that home's text, never typed) -- the case the home
    branch used to admit. The unplanted row passes first, so each failure is the plant's.
    """
    texts = tree_texts()
    classes = read_classes()
    key, status = next((k, s) for k, s in sorted(read_registry().items()) if s != "UNREVIEWED")
    assert check_registry_homes({key: status}, texts, classes) == []

    missing = "benchmarks/nonexistent.json"
    assert resolve_home(missing, tree_files()) is None
    problems = check_registry_homes({key: status + f" (see {missing})"}, texts, classes)
    assert problems and all(missing in p and "does not resolve" in p for p in problems), problems

    files = tree_files()
    ambiguous = sorted(
        name for name in {path.rsplit("/", 1)[-1] for path in files if "/" in path}
        if name not in files and sum(path.endswith("/" + name) for path in files) > 1
    )
    assert ambiguous, "no bare file name has several homes -- the drill has nothing to plant"
    problems = check_registry_homes({key: status + " " + ambiguous[0]}, texts, classes)
    assert problems and all(ambiguous[0] in p and "does not resolve" in p for p in problems), problems

    problems = check_registry_homes({key: status + " ninety-one"}, texts, classes)
    assert problems == [f"{key[1]} {key[0]}: types ninety-one, carried by no home it names"], problems

    # (d) The first row naming a resolvable home, and the first spelled number of `_WORDS` that
    # home's text carries as a whole word while neither the ruled line, the ruled set nor the reason
    # itself does: appended to the reason, it must be named exactly like the decoy.
    lines_by_key = {
        (path, claim_sha1(line)): line for path, text in texts.items() for line in text.splitlines()
    }
    planted = next(
        (k, s, word)
        for k, s in sorted(read_registry().items())
        if s != "UNREVIEWED"
        for home in [resolve_home(m.group(1), files) for m in HOME.finditer(status_tokens(s)[2])]
        if home is not None
        for word in _WORDS
        if figure_present(word, (REPO / home).read_text(encoding="utf-8", errors="replace"))
        and not figure_present(word, lines_by_key.get(k, ""))
        and word not in {t.lower() for t in status_tokens(s)[1]}
        and not figure_present(word, s)
    )
    key_d, status_d, word = planted
    assert check_registry_homes({key_d: status_d}, texts, classes) == []
    problems = check_registry_homes({key_d: status_d + " " + word}, texts, classes)
    assert problems == [f"{key_d[1]} {key_d[0]}: types {word}, carried by no home it names"], problems


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # python tests/test_g4prose.py           -> the registry skeleton for every line left open
    # python tests/test_g4prose.py --tokens  -> every token no door admits, one per line
    # The documents carry superscripts and dashes a Windows console code page cannot encode.
    sys.stdout.reconfigure(encoding="utf-8")
    _texts = tree_texts()
    if "--tokens" in sys.argv:
        _misses, _problems, _pin_problems = check_tree(_texts)
        for _p in _pin_problems:
            print(f"PIN {_p.pin.path}: {_p.pin.what}: {_p.detail}")
        for _line in _problems:
            print(f"REGISTRY {_line}")
        for _m in _misses:
            print(_m)
        sys.exit(1 if (_misses or _problems or _pin_problems) else 0)
    else:
        _open, _ = unadmitted_by_pins_and_classes(_texts, pin_table(), read_classes())
        _lines = {p: t.splitlines() for p, t in _texts.items()}
        _keys = sorted({(p, claim_sha1(_lines[p][n - 1])) for p, n in _open})
        for _p, _s in _keys:
            print(f"{_s}\t{_p}\tUNREVIEWED")
