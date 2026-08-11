"""openmucf.g4.spec -- the Layer-1 ``.g4dat`` grammar: parse, render, validate, format floats.

``FORMAT_SPEC.md`` is the normative document and this module is its reference implementation: the
format *is* this file. Everything it enforces is stated there, and every rule stated there is
tested in ``tests/test_g4spec.py``.

The three guarantees the rest of the toolchain leans on:

* **round-trip** -- for every table :func:`validate` accepts, ``parse(render(t)) == t``;
* **determinism** -- :func:`render` is a pure function of its argument (no timestamp, no locale
  dependence, no dict-ordering dependence), so a regenerated file byte-diffs cleanly against the
  committed one;
* **only conforming output** -- :func:`render` sorts the records and then validates, so anything it
  emits is accepted by :func:`parse`.

Rejections carry an exact code from ``FORMAT_SPEC.md`` section 4 and a 1-based line number, so a
malformed dataset produces an actionable message rather than a stack trace. :func:`validate` reports
the line an offending item *would* occupy in the emitted file, so an in-memory table and a parsed
file report the same way.

Standard library only, and no import of the kinetics modules (enforced by test, not by comment).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["G4DatFormatError", "G4DatTable", "format_float", "parse", "render", "validate"]

#: Version of the *format* this module implements (``#GRAMMAR``), distinct from a dataset's
#: ``#VERSION``. A reader must reject a major it does not know (E010); any minor of a known major
#: is accepted -- see ``FORMAT_SPEC.md`` section 2.7.
GRAMMAR_VERSION = "1.0"
SUPPORTED_GRAMMAR_MAJOR = 1

#: Directives in their one legal order. A repeat, or a directive after one that should follow it,
#: is E003.
DIRECTIVE_ORDER = (
    "GRAMMAR",
    "DATASET",
    "VERSION",
    "PROFILE",
    "SEAM",
    "TABLE",
    "GENERATOR",
    "SOURCEDIGEST",
    "SOURCESHA",
    "UNITS",
    "COLUMNS",
    "VALIDITY",
    "FALLBACK",
)
#: Always required. ``SOURCESHA`` is required if and only if the profile is ``parity`` (E013);
#: ``FALLBACK`` is optional.
REQUIRED_DIRECTIVES = tuple(k for k in DIRECTIVE_ORDER if k not in ("SOURCESHA", "FALLBACK"))

#: Columns carrying an unsigned integer; every other column is a float column. ``(Z, A)`` is the
#: primary key, in this order.
INTEGER_COLUMNS = ("Z", "A")
ALLOWED_SEAMS = ("d1_nuclear_capture", "d2_atomic_capture", "d3_transitions", "d4_mucf_cycle")
PARITY_PROFILE = "parity"
END_MARKER = "#END"

#: A profile token. The set is deliberately open so that N competing evaluations coexist, each as
#: its own file rather than as extra columns (``FORMAT_SPEC.md`` section 2.5).
PROFILE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,31}$")

_DIRECTIVE_PATTERN = re.compile(r"^#([A-Z][A-Z0-9]*)(?:[ \t]+(.*))?$")
_GRAMMAR_PATTERN = re.compile(r"^(\d+)\.(\d+)$")
_INTEGER_PATTERN = re.compile(r"^[0-9]+$")
#: Strict C-locale float. A comma decimal separator does not match, and that is the point: it is a
#: syntax error, never a silent truncation.
_FLOAT_PATTERN = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_NON_FINITE_TOKENS = frozenset({"inf", "infinity", "nan"})

#: ``#KEYWORD`` is padded to this width, so every directive value starts in the same column.
_KEYWORD_WIDTH = 14

Number = int | float
Key = tuple[int, ...]
_DirectiveLine = Callable[[str], int]
_RecordLine = Callable[[int], int]


class G4DatFormatError(Exception):
    """A Layer-1 rejection: an exact code, a 1-based line number, and a human explanation.

    ``str(exc)`` is exactly ``"{code}: {message} (line {line})"``.
    """

    def __init__(self, code: str, line: int, message: str) -> None:
        self.code = code
        self.line = line
        self.message = message
        super().__init__(f"{code}: {message} (line {line})")


@dataclass(frozen=True)
class G4DatTable:
    """A parsed Layer-1 table: the directives by keyword (no leading ``#``) and the records.

    ``directives`` is compared by content, not by insertion order -- :func:`render` always emits the
    canonical order -- and each record is a tuple with one entry per ``#COLUMNS`` name.
    """

    directives: dict[str, str]
    records: tuple[tuple[Number, ...], ...]


# --------------------------------------------------------------------------------------------
# floats
# --------------------------------------------------------------------------------------------


def format_float(x: float) -> str:
    """Format ``x`` as ``%.17g``: locale-independent, and exact (``float(format_float(x)) == x``).

    Seventeen significant digits round-trip every finite ``binary64`` value, subnormals included.
    Raises ``ValueError`` on a non-finite input: this layer has no line number to report, so
    ``inf``/``nan`` become E014 in :func:`validate` and :func:`parse`, which do.
    """
    value = float(x)
    if not math.isfinite(value):
        raise ValueError(f"cannot format the non-finite value {value!r}; inf and nan are not representable")
    return f"{value:.17g}"  # identical output to the C-style "%.17g", and locale-independent


# --------------------------------------------------------------------------------------------
# shared rule checks (one implementation, two line mappings: file lines vs canonical lines)
# --------------------------------------------------------------------------------------------


def _check_grammar(value: str, line: int) -> None:
    """E010: reject a ``#GRAMMAR`` major we do not implement, or a version we cannot read at all."""
    match = _GRAMMAR_PATTERN.match(value)
    if match is None:
        raise G4DatFormatError("E010", line, f"unreadable '#GRAMMAR' version {value!r}; expected MAJOR.MINOR")
    major = int(match.group(1))
    if major != SUPPORTED_GRAMMAR_MAJOR:
        raise G4DatFormatError(
            "E010",
            line,
            f"unsupported '#GRAMMAR' major version {major}; this reader implements "
            f"major {SUPPORTED_GRAMMAR_MAJOR}",
        )


def _check_header(directives: dict[str, str], dline: _DirectiveLine) -> None:
    """E002, E010, E013, E016 -- the directive-block rules, independent of where the lines are."""
    for keyword in REQUIRED_DIRECTIVES:
        if keyword not in directives:
            raise G4DatFormatError("E002", dline(keyword), f"missing required directive '#{keyword}'")

    _check_grammar(directives["GRAMMAR"], dline("GRAMMAR"))

    profile = directives["PROFILE"]
    if PROFILE_PATTERN.match(profile) is None:
        raise G4DatFormatError(
            "E016", dline("PROFILE"), f"'#PROFILE' value {profile!r} is not a {PROFILE_PATTERN.pattern} token"
        )
    seam = directives["SEAM"]
    if seam not in ALLOWED_SEAMS:
        raise G4DatFormatError(
            "E016", dline("SEAM"), f"'#SEAM' value {seam!r} is not one of {', '.join(ALLOWED_SEAMS)}"
        )

    # "required iff parity", enforced in both directions: a parity file must name the revision it
    # reproduces, and a file that is not reproducing anything must not claim to.
    if profile == PARITY_PROFILE and "SOURCESHA" not in directives:
        raise G4DatFormatError(
            "E013", dline("PROFILE"), f"'#PROFILE {PARITY_PROFILE}' requires '#SOURCESHA'"
        )
    if profile != PARITY_PROFILE and "SOURCESHA" in directives:
        raise G4DatFormatError(
            "E013",
            dline("SOURCESHA"),
            f"'#SOURCESHA' is only allowed under '#PROFILE {PARITY_PROFILE}', not {profile!r}",
        )


def _is_integer_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, float):
        return math.isfinite(value) and value >= 0.0 and value.is_integer()
    return False


def _is_real_value(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _key_indices(columns: Sequence[str]) -> list[int]:
    """Positions of ``Z`` then ``A``. Empty when the table declares neither, in which case it has no
    primary key and E008/E015 have nothing to check (``FORMAT_SPEC.md`` section 2.3)."""
    return [columns.index(name) for name in INTEGER_COLUMNS if name in columns]


def _check_records(
    records: Sequence[tuple[Number, ...]], columns: Sequence[str], rline: _RecordLine
) -> None:
    """E004, E007, E014, E008, E015 -- the record rules.

    E008 is checked before E015 on purpose: a duplicate key is also not strictly ascending, so the
    other order would make E008 unreachable.
    """
    key_indices = _key_indices(columns)
    first_seen: dict[Key, int] = {}
    previous: Key | None = None

    for index, record in enumerate(records):
        line = rline(index)
        if len(record) != len(columns):
            raise G4DatFormatError(
                "E004",
                line,
                f"record has {len(record)} field(s), '#COLUMNS' declares {len(columns)}",
            )
        for name, value in zip(columns, record, strict=True):
            if name in INTEGER_COLUMNS:
                if not _is_integer_value(value):
                    raise G4DatFormatError(
                        "E007", line, f"column {name!r} needs an unsigned integer, got {value!r}"
                    )
            elif not _is_real_value(value):
                raise G4DatFormatError("E007", line, f"column {name!r} needs a number, got {value!r}")
            elif not math.isfinite(float(value)):
                raise G4DatFormatError("E014", line, f"non-finite value in column {name!r}: {value!r}")

        if not key_indices:
            continue
        key: Key = tuple(int(record[i]) for i in key_indices)
        printed = ", ".join(f"{name}={part}" for name, part in zip(INTEGER_COLUMNS, key, strict=False))
        if key in first_seen:
            raise G4DatFormatError(
                "E008", line, f"duplicate key ({printed}); first seen at line {first_seen[key]}"
            )
        if previous is not None and key < previous:
            raise G4DatFormatError(
                "E015", line, f"record ({printed}) is not in ascending {'/'.join(INTEGER_COLUMNS)} order"
            )
        first_seen[key] = line
        previous = key


# --------------------------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------------------------


def _check_encoding(text: str) -> None:
    """E005 -- decoding precedes line splitting, so this whole-text check runs first."""
    if text.isascii():
        return
    index = next(i for i, ch in enumerate(text) if not ch.isascii())
    character = text[index]
    raise G4DatFormatError(
        "E005",
        text.count("\n", 0, index) + 1,
        f"non-ASCII character {character!r} (U+{ord(character):04X})",
    )


def _check_line_endings(text: str) -> None:
    """E006 -- CRLF and lone CR alike; the format is LF-only."""
    index = text.find("\r")
    if index < 0:
        return
    kind = "CRLF" if text[index : index + 2] == "\r\n" else "CR"
    raise G4DatFormatError(
        "E006", text.count("\n", 0, index) + 1, f"{kind} line ending; this format is LF-only"
    )


def _split_directive(line: str, lineno: int) -> tuple[str, str]:
    """Split ``#KEYWORD value`` -- E001 for anything that is not a keyword we know."""
    match = _DIRECTIVE_PATTERN.match(line)
    if match is None:
        raise G4DatFormatError("E001", lineno, f"unreadable directive line {line.strip()!r}")
    keyword = match.group(1)
    if keyword not in DIRECTIVE_ORDER:
        raise G4DatFormatError("E001", lineno, f"unknown directive '#{keyword}'")
    return keyword, (match.group(2) or "").strip()


def _lex_record(line: str, lineno: int, columns: Sequence[str]) -> tuple[Number, ...]:
    """Lex one record line -- E004, E007, E014. Leading whitespace and column alignment are fine;
    any run of whitespace is one separator."""
    fields = line.split()
    if len(fields) != len(columns):
        raise G4DatFormatError(
            "E004", lineno, f"record has {len(fields)} field(s), '#COLUMNS' declares {len(columns)}"
        )
    values: list[Number] = []
    for name, field in zip(columns, fields, strict=True):
        if name in INTEGER_COLUMNS:
            if _INTEGER_PATTERN.match(field) is None:
                raise G4DatFormatError(
                    "E007", lineno, f"unreadable unsigned integer in column {name!r}: {field!r}"
                )
            values.append(int(field))
            continue
        bare = field[1:] if field[:1] in "+-" else field
        if bare.lower() in _NON_FINITE_TOKENS:
            raise G4DatFormatError("E014", lineno, f"non-finite value in column {name!r}: {field!r}")
        if _FLOAT_PATTERN.match(field) is None:
            raise G4DatFormatError("E007", lineno, f"unreadable float in column {name!r}: {field!r}")
        number = float(field)
        if not math.isfinite(number):
            raise G4DatFormatError(
                "E014", lineno, f"value in column {name!r} overflows to infinity: {field!r}"
            )
        values.append(number)
    return tuple(values)


def parse(text: str) -> G4DatTable:
    """Parse a Layer-1 ``.g4dat`` document, raising :class:`G4DatFormatError` on the first violation.

    ``text`` is the decoded file. Read the file as bytes and decode it yourself: reading it in text
    mode with universal newlines rewrites CRLF into LF and hides E006.
    """
    _check_encoding(text)
    _check_line_endings(text)

    raw = text.split("\n")
    newline_terminated = bool(raw) and raw[-1] == ""
    body = raw[:-1] if newline_terminated else raw

    directives: dict[str, str] = {}
    directive_lines: dict[str, int] = {}
    records: list[tuple[Number, ...]] = []
    record_lines: list[int] = []
    columns: list[str] | None = None
    order_index = -1
    end_line: int | None = None

    def close_header(at_line: int) -> list[str]:
        """Run the header rules once the directive block is over, then fix the column list."""
        _check_header(directives, lambda keyword: directive_lines.get(keyword, at_line))
        return directives["COLUMNS"].split()

    for lineno, line in enumerate(body, start=1):
        if end_line is not None:
            raise G4DatFormatError("E011", lineno, f"content after '{END_MARKER}': {line.strip()!r}")

        if line.rstrip() == END_MARKER:
            if columns is None:
                columns = close_header(lineno)
            end_line = lineno
            continue

        if line.startswith("#"):
            if columns is not None:
                raise G4DatFormatError(
                    "E003", lineno, f"directive {line.split()[0]!r} appears after the record block began"
                )
            keyword, value = _split_directive(line, lineno)
            index = DIRECTIVE_ORDER.index(keyword)
            if index <= order_index:
                previous = DIRECTIVE_ORDER[order_index]
                detail = "repeated" if keyword == previous else f"must precede '#{previous}'"
                raise G4DatFormatError("E003", lineno, f"directive '#{keyword}' is out of order ({detail})")
            order_index = index
            directives[keyword] = value
            directive_lines[keyword] = lineno
            continue

        if columns is None:
            columns = close_header(lineno)
        records.append(_lex_record(line, lineno, columns))
        record_lines.append(lineno)

    if end_line is None:
        raise G4DatFormatError("E012", max(len(body), 1), f"missing '{END_MARKER}' terminator")
    if not newline_terminated:
        raise G4DatFormatError(
            "E012",
            len(body),
            f"the '{END_MARKER}' line is not newline-terminated; the file must end with a newline",
        )

    if columns is None:  # pragma: no cover -- the #END branch closes the header before setting it
        columns = close_header(end_line)
    _check_records(records, columns, lambda index: record_lines[index])
    return G4DatTable(directives=dict(directives), records=tuple(records))


# --------------------------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------------------------


def _canonical_lines(directives: dict[str, str]) -> tuple[_DirectiveLine, _RecordLine]:
    """Line numbers as :func:`render` would lay the table out, so validate and parse report alike."""
    lines: dict[str, int] = {}
    count = 0
    for keyword in DIRECTIVE_ORDER:
        if keyword in directives:
            count += 1
            lines[keyword] = count
    for keyword in sorted(k for k in directives if k not in DIRECTIVE_ORDER):
        count += 1
        lines[keyword] = count
    header_lines = count

    def dline(keyword: str) -> int:
        if keyword in lines:
            return lines[keyword]
        if keyword not in DIRECTIVE_ORDER:  # pragma: no cover -- unknown keys are always present
            return header_lines + 1
        # A missing directive is reported at the line it would have occupied.
        preceding = DIRECTIVE_ORDER[: DIRECTIVE_ORDER.index(keyword)]
        return sum(1 for k in preceding if k in directives) + 1

    def rline(index: int) -> int:
        return header_lines + 1 + index

    return dline, rline


def validate(table: G4DatTable) -> None:
    """Check an in-memory table against ``FORMAT_SPEC.md`` section 2. Returns ``None`` when clean.

    Raises :class:`G4DatFormatError` with the line the offending item would occupy in the emitted
    file. A table no file could ever produce -- a directive value with leading or trailing
    whitespace, or an embedded newline -- raises ``ValueError`` instead: that is a programming
    error, and the section-4 codes stay exactly the sixteen file-level conditions.
    """
    directives = table.directives
    for keyword, value in directives.items():
        if not isinstance(keyword, str) or not isinstance(value, str):
            raise ValueError(f"directive {keyword!r} must map a str keyword to a str value, got {value!r}")
        if value != value.strip():
            raise ValueError(
                f"directive '#{keyword}' value {value!r} carries leading or trailing whitespace; "
                "no file can produce that, so the table would not survive a round-trip"
            )
        if "\n" in value:
            raise ValueError(f"directive '#{keyword}' value {value!r} contains a newline")

    dline, rline = _canonical_lines(directives)

    for keyword, value in directives.items():
        if not keyword.isascii() or not value.isascii():
            raise G4DatFormatError("E005", dline(keyword), f"non-ASCII character in directive '#{keyword}'")
        if "\r" in value:
            raise G4DatFormatError("E006", dline(keyword), f"carriage return in directive '#{keyword}'")
    for keyword in directives:
        if keyword not in DIRECTIVE_ORDER:
            raise G4DatFormatError("E001", dline(keyword), f"unknown directive '#{keyword}'")

    _check_header(directives, dline)
    _check_records(table.records, directives["COLUMNS"].split(), rline)


# --------------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------------


def _sorted_records(
    records: tuple[tuple[Number, ...], ...], columns: Sequence[str]
) -> tuple[tuple[Number, ...], ...]:
    """Records ascending by ``(Z, A)``. Best effort: a table too broken to sort is left alone and
    :func:`validate` reports what is actually wrong with it."""
    key_indices = _key_indices(columns)
    if not key_indices:
        return records
    try:
        return tuple(sorted(records, key=lambda record: tuple(record[i] for i in key_indices)))
    except (IndexError, TypeError):
        return records


def _render_field(name: str, value: Number) -> str:
    return str(int(value)) if name in INTEGER_COLUMNS else format_float(float(value))


def render(table: G4DatTable) -> str:
    """Emit ``table`` as a Layer-1 document, records ascending by ``(Z, A)``.

    Pure: two calls give identical bytes, there is no timestamp anywhere in the output, and the
    float syntax does not depend on ``LC_NUMERIC``. The table is validated after sorting, so
    everything this returns is accepted by :func:`parse` -- an unsorted input is normalized rather
    than rejected, which is the one way :func:`render` is more permissive than :func:`validate`.
    """
    columns = table.directives.get("COLUMNS", "").split()
    records = _sorted_records(table.records, columns)
    validate(G4DatTable(directives=dict(table.directives), records=records))

    lines = [
        f"{'#' + keyword:<{_KEYWORD_WIDTH}}{table.directives[keyword]}".rstrip()
        for keyword in DIRECTIVE_ORDER
        if keyword in table.directives
    ]
    cells = [
        [_render_field(name, value) for name, value in zip(columns, record, strict=True)]
        for record in records
    ]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))] if cells else []
    lines += [" ".join(cell.rjust(width) for cell, width in zip(row, widths, strict=True)) for row in cells]
    lines.append(END_MARKER)
    return "\n".join(lines) + "\n"
