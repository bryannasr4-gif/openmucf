"""T-74/T-75 -- every number in the dataset documents is computed from shipped data or listed with a reason.

`tests/test_g4parity.py` pins the counts it knows how to compute (T-63), and its docstring says what
that leaves open: a pin table is not a census, so a number nobody thought to pin drifts unwatched.
This file closes the complement. It enumerates every numeric token and every spelled number in the
documents named by `PROSE_PATHS` and admits each one only through one of three doors, tried in
order:

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
"""

from __future__ import annotations

import dataclasses
import fnmatch
import functools
import hashlib
import importlib.util
import pathlib
import re
import subprocess
import sys

import test_g4parity as parity

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The documents under the check. Every other public document is out of its reach and says nothing
#: this check would vouch for.
PROSE_PATHS = ("DATASET_D1.md", "README.md", "cpp/tools/README.md")
#: Documents that may carry no registry row: every token in them is pinned or class-admitted.
REGISTRY_FREE = ("cpp/README.md",)

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
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
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
    """Every numeric token and spelled number in `text`, in file order."""
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
    #: for a pattern whose value another test already asserts (T-63, `check_f3.py`).
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
        ("README.md", pins.readme_claims),
        ("cpp/tools/README.md", pins.tools_readme_claims),
    ):
        for row in rows:
            table.append(Pin(row[0], path, row[1], (1,)))
    document_figures = check_f3.DOCUMENT_FIGURES
    readme_figures = check_f3.README_FIGURES
    table.append(Pin("F-3 figures, the document's block", "DATASET_D1.md",
                     document_figures.pattern, tuple(range(1, document_figures.groups + 1))))
    table.append(Pin("F-3 figures, the harvest tooling's restatement", "cpp/tools/README.md",
                     readme_figures.pattern, tuple(range(1, readme_figures.groups + 1))))
    table.extend(internal_pins(pins, check_f3))
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
    max_ulp = figures[3]
    d1 = parity.d1
    records = _computed(pins, "capture record count, section 1")
    distinct_z = _computed(pins, "distinct Z, section 1")
    swept = _computed(pins, "swept points, section 4")
    zeff_entries = _computed(pins, "effective-charge array length, section 3")
    open_rows = _computed(pins, "open rows")
    named = _computed(pins, "elements the primary's sentence names")
    carrying = _computed(pins, "named elements carrying a separated-isotope record")
    zeff_56 = _rounded(pins, "the effective charge at Z=56")
    zeff_81 = _rounded(pins, "the effective charge at Z=81")
    zeff_82 = _rounded(pins, "the effective charge at Z=82")
    zeff_83 = _rounded(pins, "the effective charge at Z=83")
    # The unit conversion the document states is the module's own constant: microseconds per
    # nanosecond, which is what `value / 1000` divides by.
    per_microsecond = d1.MICROSECOND
    assert per_microsecond == int(per_microsecond)
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
        Pin("microseconds per nanosecond, the table-hit conversion", "DATASET_D1.md",
            r"the rate is instead `value / (\d+)`", (1,), int(per_microsecond)),
        Pin("named elements without a separated-isotope record, F-6", "DATASET_D1.md",
            r"The (\w+) exceptions are instructive", (1,), named - carrying),
        Pin("open rows, F-6's strontium sentence", "DATASET_D1.md",
            r"it is one of the (\w+) records this dataset \*\*cannot settle\*\*", (1,), open_rows),
        Pin("mononuclidic route, section 6 restated", "DATASET_D1.md",
            r"the (\d+) mononuclidic calls above", (1,), _computed(pins, "mononuclidic route")),
        Pin("open rows, section 6 closing sentence", "DATASET_D1.md",
            r"Three of the (\w+) unsettled rows appear in that list", (1,), open_rows),
        Pin("open rows, against the disagreements", "DATASET_D1.md",
            r"The (\w+) open rows and the \d+ disagreements", (1,), open_rows),
        Pin("rows the old rule disagrees with, restated", "DATASET_D1.md",
            r"The \w+ open rows and the (\d+) disagreements", (1,),
            _computed(pins, "rows the old rule disagrees with")),
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
        Pin("open rows, section 6's naming sentence", "DATASET_D1.md",
            r"\*\*The (\w+) unsettled rows say so with an empty locator", (1,), open_rows),
        Pin("F-3 maximum, the README's restatement", "README.md",
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
    ]


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
            if pin.places is None:
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
    misses, problems = unadmitted(texts, pins, read_classes(), registry)
    return misses, check_registry_form(registry) + problems, pin_problems


# --------------------------------------------------------------------------------------------
# T-74 -- the tree
# --------------------------------------------------------------------------------------------


def test_t74_every_number_in_the_dataset_documents_is_computed_or_listed_with_a_reason():
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
    assert original.count("2980 ulp") == 1
    # Beside the pin, as the attack states it: the figure pattern itself then no longer matches
    # and every figure on the line surfaces, the decoy among them.
    texts["cpp/tools/README.md"] = original.replace("2980 ulp", "2980 ulp and 2981 ulp")
    lineno = next(i for i, line in enumerate(original.splitlines(), 1) if "2980 ulp" in line)
    misses, _ = _drill(texts)
    assert ("cpp/tools/README.md", lineno, "2981") in {
        (m.token.path, m.token.lineno, m.token.text) for m in misses
    }
    # Past the pin's span, so the pin still matches: the decoy is then the only new failure.
    texts["cpp/tools/README.md"] = original.replace("swept points", "swept points and 2981 ulp", 1)
    misses, _ = _drill(texts)
    assert [(m.token.path, m.token.lineno, m.token.text) for m in misses] == [
        ("cpp/tools/README.md", lineno, "2981")
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


def test_t75_the_enumerator_prints_nothing_on_the_tree():
    result = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__)), "--tokens"],
        capture_output=True, text=True, cwd=REPO, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", result.stdout


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
