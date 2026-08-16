"""D1 (nuclear capture): extract Geant4's compiled-in data, and reproduce its compiled behaviour.

Two halves, and the separation is the point.

**The extractor** parses ``third_party/geant4/v11.4.2/G4MuonMinusBoundDecay.cc`` -- the vendored,
blob-pinned upstream file -- and returns what it found: the ``{Z, A, cRate, cRErr}`` capture records
with the exact float literals they were written as, the effective-charge (``zeff``) table, and every
constant the Goulard-Primakoff fallback needs. It is **structural**: it anchors on declarations and
brace-matches to their ends. It never uses a line number to find anything, because a parser keyed to
line numbers keeps parsing at the next upstream release and silently extracts the wrong thing, while
a parser keyed to declarations either finds them or fails loudly. Nothing here is transcribed: a
number typed by a human anywhere in this chain would be a defect.

**The reference implementation** evaluates the same function Geant4 compiles -- table lookup with
Geant4's own early-exit scan, Goulard-Primakoff otherwise -- and it is bit-exact against the
compiled library over the whole ``(Z, A)`` box the dataset claims. Getting that requires two things
that read as pedantry and are not:

* **the association order is part of the specification.** Floating-point ``+`` and ``*`` are not
  associative, so ``(a*b)*c`` and ``a*(b*c)`` are different functions. :meth:`GoulardPrimakoff.rate`
  reproduces the C++ expression's parenthesisation exactly, and says so at each step.
* **no floating-point contraction.** The identical C++ expression compiled with FMA contraction
  enabled -- which is the compiler default wherever FMA exists, including the baseline aarch64 ISA
  -- returns results up to thousands of ulp away. CPython evaluates each operation separately and
  rounds each one, so this module *is* the no-contraction contract rather than merely obeying it.

Standard library only, and no import of the kinetics modules.
"""

from __future__ import annotations

import csv
import hashlib
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AUDIT_COLUMNS",
    "AUDIT_RELPATH",
    "D1Extraction",
    "GoulardPrimakoff",
    "IsotopeAuditRow",
    "SourceExtractionError",
    "capture_rate",
    "extract",
    "load",
    "load_isotope_audit",
    "parse_fallback_directive",
    "render_fallback_directive",
    "sweep_digest",
]


# --------------------------------------------------------------------------------------------
# the pins -- the identity of the file everything below is derived from
# --------------------------------------------------------------------------------------------

#: Where the vendored copy lives, relative to the repository root.
VENDORED_RELPATH = "third_party/geant4/v11.4.2/G4MuonMinusBoundDecay.cc"
#: The upstream revision the ``parity`` profile reproduces; this is what ``#SOURCESHA`` carries.
UPSTREAM_COMMIT = "8cc04f65977807f1848da7b958c421cd5e162f26"
UPSTREAM_TAG = "v11.4.2"
UPSTREAM_PATH = "source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc"
#: Upstream's own object name for the vendored bytes -- verifiable against github.com/Geant4/geant4
#: with no Geant4 checkout, no ``git`` binary, and no trust in this project.
UPSTREAM_BLOB_ID = "29bd73719cd619de34ef83ca5ca076ceadf1cc5a"
UPSTREAM_SHA256 = "860dcdb53167c6437484b12c05ac1ab2eae4a6a52886af83fcf4394611882813"

#: The model name the dataset declares in ``#FALLBACK``, and the coefficients it must carry. All
#: eight are needed: a consumer handed only the four ``b0*``/``t1`` coefficients cannot evaluate the
#: formula, which defeats the purpose of declaring the fallback as data at all.
FALLBACK_MODEL = "goulard_primakoff"
FALLBACK_NAMES = ("b0a", "b0b", "b0c", "t1", "xmu_coeff", "mix", "zmin", "zmax")

#: The ``(Z, A)`` box the parity claim is made over. Z to 120 because ``G4IonTable`` admits Z up to
#: 118 and 120 leaves margin; A to 300 because it covers every known nuclide with margin. Degenerate
#: inputs (Z <= 0, A = 0) are excluded and recorded separately: they return non-finite values, and a
#: NaN has no single bit pattern to hash.
SWEEP_Z_MIN, SWEEP_Z_MAX = 1, 120
SWEEP_A_MIN, SWEEP_A_MAX = 1, 300

#: Geant4's time unit, written exactly as CLHEP's ``SystemOfUnits.h`` derives it
#: (``nanosecond = 1.``, ``second = 1.e+9*nanosecond``, ``microsecond = 1.e-6*second``) rather than
#: as the constant it happens to equal. The table branch returns ``cRate / microsecond``, so this
#: divisor sits directly in the parity chain and deserves to be derived rather than asserted.
_NANOSECOND = 1.0
_SECOND = 1.0e9 * _NANOSECOND
MICROSECOND = 1.0e-6 * _SECOND


class SourceExtractionError(RuntimeError):
    """The vendored source is not shaped the way the extractor requires.

    Always raised naming the construct that was being looked for. There is no silent-skip path in
    this module: an extractor that quietly returns fewer records than the array holds produces a
    dataset that is wrong, self-consistent, and passes every downstream check.
    """


# --------------------------------------------------------------------------------------------
# lexical helpers -- comment/string masking, brace matching
# --------------------------------------------------------------------------------------------

#: A C float literal, in the same shape the format's own section-2.3 rule 4 accepts.
_FLOAT = r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"


def _mask(text: str) -> str:
    """``text`` with every comment and string/char literal blanked, newlines and offsets preserved.

    Structural scanning runs over this rather than over the raw text, so a brace inside a comment or
    a string can never terminate an array body early. Offsets are unchanged, so a match found here
    indexes the original text directly, and line numbers are computable from either.
    """
    out = list(text)
    index, end = 0, len(text)

    def blank(start: int, stop: int) -> None:
        for position in range(start, min(stop, end)):
            if out[position] != "\n":  # keep the line map intact
                out[position] = " "

    while index < end:
        char = text[index]
        pair = text[index : index + 2]
        if pair == "//":
            newline = text.find("\n", index)
            stop = end if newline < 0 else newline
            blank(index, stop)
            index = stop
        elif pair == "/*":
            closing = text.find("*/", index + 2)
            stop = end if closing < 0 else closing + 2
            blank(index, stop)
            index = stop
        elif char in "\"'":
            cursor = index + 1
            while cursor < end:
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if text[cursor] == char:
                    cursor += 1
                    break
                cursor += 1
            blank(index, cursor)
            index = min(cursor, end)
        else:
            index += 1
    return "".join(out)


def _matching_brace(masked: str, opening: int, construct: str) -> int:
    """Index of the ``}`` closing the ``{`` at ``opening``, scanning the masked text."""
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise SourceExtractionError(f"{construct}: no closing brace for the initialiser opened here")


def _line_of(text: str, offset: int) -> int:
    """The 1-based line ``offset`` falls on. Used for provenance locators, never for parsing."""
    return text.count("\n", 0, offset) + 1


def _find(pattern: re.Pattern[str], masked: str, construct: str) -> re.Match[str]:
    match = pattern.search(masked)
    if match is None:
        raise SourceExtractionError(
            f"{construct}: not found in the vendored source. This extractor is structural on "
            f"purpose -- it will not guess at a moved or renamed construct, because guessing "
            f"produces a dataset that is wrong and self-consistent."
        )
    return match


def _array_body(masked: str, declaration: re.Pattern[str], construct: str) -> tuple[int, int]:
    """The half-open span of an array initialiser's body, brace-matched from its declaration."""
    match = _find(declaration, masked, construct)
    opening = masked.index("{", match.end() - 1)
    return opening + 1, _matching_brace(masked, opening, construct)


# --------------------------------------------------------------------------------------------
# the constructs this module knows how to find
# --------------------------------------------------------------------------------------------

_CAPRATES_DECL = re.compile(r"static\s+const\s+capRate\s+capRates\s*\[\s*\]\s*=\s*\{")
_ZEFF_DECL = re.compile(r"static\s+const\s+G4double\s+zeff\s*\[\s*\]\s*=\s*\{")
_MAXZ_DECL = re.compile(r"static\s+const\s+G4int\s+maxZ\s*=\s*([0-9]+)\s*;")
_CAPTURE_RECORD = re.compile(
    r"\{\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*(" + _FLOAT + r")\s*,\s*(" + _FLOAT + r")\s*\}"
)
#: What may legally remain between records once every record span is removed.
_RECORD_SEPARATOR = re.compile(r"^[\s,]*$")

_CAPTURE_FUNCTION = re.compile(r"G4MuonMinusBoundDecay::GetMuonCaptureRate\s*\(")
_ZEFF_FUNCTION = re.compile(r"G4MuonMinusBoundDecay::GetMuonZeff\s*\(")
_STRUCT_CAPRATE = re.compile(r"struct\s+capRate\s*\{")

#: The eight fallback inputs, each anchored on the tokens that surround it in the source. ``zmin``
#: and ``zmax`` are the clamp `GetMuonZeff` applies before indexing, which is as much a part of the
#: model as the coefficients: without them a consumer cannot evaluate the formula outside 1..100.
_COEFFICIENT_ANCHORS: tuple[tuple[str, str], ...] = (
    ("b0a", r"static\s+const\s+G4double\s+b0a\s*=\s*(" + _FLOAT + r")\s*;"),
    ("b0b", r"static\s+const\s+G4double\s+b0b\s*=\s*(" + _FLOAT + r")\s*;"),
    ("b0c", r"static\s+const\s+G4double\s+b0c\s*=\s*(" + _FLOAT + r")\s*;"),
    ("t1", r"static\s+const\s+G4double\s+t1\s*=\s*(" + _FLOAT + r")\s*;"),
    ("xmu_coeff", r"xmu\s*=\s*zeff2\s*\*\s*(" + _FLOAT + r")\s*;"),
    ("mix", r"\(\s*1\.0\s*-\s*\(\s*1\.0\s*-\s*xmu\s*\)\s*\*\s*(" + _FLOAT + r")\s*\)"),
    ("zmin", r"std::max\s*\(\s*std::min\s*\(\s*ZZ\s*,\s*maxZ\s*\)\s*,\s*([0-9]+)\s*\)"),
    ("zmax", r"static\s+const\s+G4int\s+maxZ\s*=\s*([0-9]+)\s*;"),
)


@dataclass(frozen=True)
class D1Extraction:
    """Everything D1 needs, read out of one vendored source file.

    Both the parsed values *and* the literals they were written as are carried. The literals are not
    redundant: they are what lets a test prove the shipped double came from that text rather than
    from a transcription that happens to round the same way, and what lets ``#FALLBACK`` declare each
    coefficient in the source's own spelling instead of a re-rendered one.

    Line numbers are here for provenance locators only. **Nothing in the parse uses them**, by
    design: a parser that keyed on line numbers would break the moment upstream added a blank line.
    But a Layer-2 ``source_locator`` that cannot point at a line is a locator a reader cannot
    follow, so they are carried, and carried separately.
    """

    capture_records: tuple[tuple[int, int, float, float], ...]
    capture_literals: tuple[tuple[str, str], ...]
    capture_lines: tuple[int, ...]
    capture_comment_lines: tuple[str, ...]
    zeff: tuple[float, ...]
    zeff_literals: tuple[str, ...]
    zeff_lines: tuple[int, ...]
    zeff_comment_lines: tuple[str, ...]
    zeff_max_z: int
    fallback_coefficients: tuple[tuple[str, str], ...]

    @property
    def coefficients(self) -> dict[str, str]:
        """The fallback coefficients as ``{name: source text}``."""
        return dict(self.fallback_coefficients)

    @property
    def distinct_capture_z(self) -> tuple[int, ...]:
        """The distinct Z the capture table covers, ascending."""
        return tuple(sorted({z for z, _, _, _ in self.capture_records}))

    def capture_rows_per_z(self) -> dict[int, int]:
        """How many capture records each Z carries -- the input to the isotope-resolution rule."""
        counts: dict[int, int] = {}
        for z, _, _, _ in self.capture_records:
            counts[z] = counts.get(z, 0) + 1
        return counts


def _comment_lines(segment: str) -> tuple[str, ...]:
    """The ``//`` comment text in ``segment``, stripped of markers and blank lines, in order.

    Read from the RAW text, never from the masked copy: these lines are the upstream attribution the
    Layer-2 rows quote verbatim, so blanking them would be exactly backwards.
    """
    lines = []
    for raw in segment.splitlines():
        stripped = raw.strip()
        if not stripped.startswith("//"):
            continue
        body = stripped[2:].strip()
        if body:
            lines.append(body)
    return tuple(lines)


def _parse_capture_records(
    text: str, masked: str
) -> tuple[
    tuple[tuple[int, int, float, float], ...], tuple[tuple[str, str], ...], tuple[int, ...]
]:
    """Every ``{Z, A, cRate, cRErr}`` record of ``capRates[]``, in source order.

    The completeness check below is the load-bearing part. Matching records with a regex is easy;
    proving that the regex matched *all* of them is the difference between a parity dataset and a
    silently truncated one. So every matched span is removed from the array body and the residue
    must be nothing but whitespace and commas -- a record the pattern failed to match leaves its own
    text behind and fails here, naming what was left over.
    """
    start, stop = _array_body(masked, _CAPRATES_DECL, "capRates[]")
    body = masked[start:stop]

    records: list[tuple[int, int, float, float]] = []
    literals: list[tuple[str, str]] = []
    lines: list[int] = []
    residue: list[str] = []
    cursor = 0
    for match in _CAPTURE_RECORD.finditer(body):
        residue.append(body[cursor : match.start()])
        cursor = match.end()
        z_text, a_text, rate_text, error_text = match.groups()
        records.append((int(z_text), int(a_text), float(rate_text), float(error_text)))
        literals.append((rate_text, error_text))
        lines.append(_line_of(text, start + match.start()))
    residue.append(body[cursor:])

    if not records:
        raise SourceExtractionError("capRates[]: the initialiser matched no records at all")
    leftover = "".join(residue)
    if _RECORD_SEPARATOR.match(leftover) is None:
        raise SourceExtractionError(
            "capRates[]: the initialiser holds text the record pattern did not match, so this parse "
            f"is SHORT and every count derived from it would be wrong. Unmatched: {leftover.strip()!r}"
        )
    return tuple(records), tuple(literals), tuple(lines)


def _parse_zeff(text: str, masked: str) -> tuple[tuple[float, ...], tuple[str, ...], tuple[int, ...]]:
    """The ``zeff[]`` initialiser: every value, its literal, and the line it sits on."""
    start, stop = _array_body(masked, _ZEFF_DECL, "zeff[]")
    values: list[float] = []
    literals: list[str] = []
    lines: list[int] = []
    offset = start
    for chunk in masked[start:stop].split(","):
        literal = chunk.strip()
        if literal:
            if re.fullmatch(_FLOAT, literal) is None:
                raise SourceExtractionError(
                    f"zeff[]: {literal!r} is not a float literal; the initialiser is not the flat "
                    "comma-separated list this extractor requires"
                )
            values.append(float(literal))
            literals.append(literal)
            lines.append(_line_of(text, offset + chunk.index(literal)))
        offset += len(chunk) + 1
    if not values:
        raise SourceExtractionError("zeff[]: the initialiser matched no values at all")
    return tuple(values), tuple(literals), tuple(lines)


def extract(source_text: str) -> D1Extraction:
    """Parse the vendored ``G4MuonMinusBoundDecay.cc`` into everything D1 is generated from."""
    masked = _mask(source_text)

    records, literals, lines = _parse_capture_records(source_text, masked)
    zeff, zeff_literals, zeff_lines = _parse_zeff(source_text, masked)

    max_z_match = _find(_MAXZ_DECL, masked, "maxZ")
    max_z = int(max_z_match.group(1))

    coefficients: list[tuple[str, str]] = []
    for name, pattern in _COEFFICIENT_ANCHORS:
        match = _find(re.compile(pattern), masked, f"the {name!r} fallback constant")
        coefficients.append((name, match.group(1)))

    capture_function = _find(_CAPTURE_FUNCTION, masked, "GetMuonCaptureRate")
    capture_struct = _find(_STRUCT_CAPRATE, masked, "struct capRate")
    zeff_function = _find(_ZEFF_FUNCTION, masked, "GetMuonZeff")

    return D1Extraction(
        capture_records=records,
        capture_literals=literals,
        capture_lines=lines,
        capture_comment_lines=_comment_lines(
            source_text[capture_function.end() : capture_struct.start()]
        ),
        zeff=zeff,
        zeff_literals=zeff_literals,
        zeff_lines=zeff_lines,
        zeff_comment_lines=_comment_lines(
            source_text[zeff_function.end() : max_z_match.start()]
        ),
        zeff_max_z=max_z,
        fallback_coefficients=tuple(coefficients),
    )


def check_pins(data: bytes) -> None:
    """Fail loudly if the vendored bytes are not the pinned upstream blob.

    Called on every generation, so a re-pin cannot happen by accident: the sequence is forced to be
    T-40 fails, the extraction yields different counts, the generated bytes move, and the byte-diff
    audit reports them -- in that order, and never silently.
    """
    if b"\r" in data:
        raise SourceExtractionError(
            f"{VENDORED_RELPATH} carries CR bytes: the checkout rewrote its line endings and both "
            "pins below are now meaningless. Check `.gitattributes` for `third_party/geant4/** -text`."
        )
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
    if blob != UPSTREAM_BLOB_ID:
        raise SourceExtractionError(
            f"{VENDORED_RELPATH} has git blob id {blob}, not the pinned {UPSTREAM_BLOB_ID}. "
            "A deliberate re-pin belongs in a NEW third_party/geant4/<tag>/ directory with a new "
            "#SOURCESHA -- never as an overwrite, which destroys the evidence that the previously "
            "published dataset was faithful to the version it claimed."
        )
    digest = hashlib.sha256(data).hexdigest()
    if digest != UPSTREAM_SHA256:  # pragma: no cover -- unreachable while the blob id matches
        raise SourceExtractionError(f"{VENDORED_RELPATH} has sha256 {digest}, not {UPSTREAM_SHA256}")


def load(path: str | Path) -> D1Extraction:
    """Read the vendored source, check its pins, and extract it."""
    data = Path(path).read_bytes()  # binary: the pins are over these exact bytes
    check_pins(data)
    return extract(data.decode("ascii"))


# --------------------------------------------------------------------------------------------
# the declared fallback -- parsed from the directive, evaluated in the compiled association order
# --------------------------------------------------------------------------------------------


def parse_fallback_directive(value: str) -> tuple[str, dict[str, str]]:
    """Split a ``#FALLBACK`` value into its model name and its ``NAME=VALUE`` assignments."""
    fields = value.split()
    if not fields:
        raise ValueError("a '#FALLBACK' value must name a model")
    model, assignments = fields[0], fields[1:]
    coefficients: dict[str, str] = {}
    for assignment in assignments:
        name, separator, text = assignment.partition("=")
        if not separator or not name or not text:
            raise ValueError(f"'#FALLBACK' assignment {assignment!r} is not NAME=VALUE")
        coefficients[name] = text
    return model, coefficients


def render_fallback_directive(model: str, coefficients: Mapping[str, str]) -> str:
    """The ``#FALLBACK`` directive value: the model, then each coefficient in :data:`FALLBACK_NAMES`.

    Every value is the **source text** of the constant, not a re-rendered float. A record column is
    ``%.17g`` because the format mandates it there; a directive value is one opaque string to the
    reader, so the faithful spelling is available here and is what is used -- ``875.e-9`` ships as
    ``875.e-9``, and a test asserts each string occurs verbatim in the vendored source.
    """
    missing = [name for name in FALLBACK_NAMES if name not in coefficients]
    if missing:
        raise ValueError(f"the {model} fallback is missing coefficient(s): {', '.join(missing)}")
    fields = [model] + [f"{name}={coefficients[name]}" for name in FALLBACK_NAMES]
    return " ".join(fields)


@dataclass(frozen=True)
class GoulardPrimakoff:
    """The declared analytic fallback, evaluated exactly as the compiled expression associates.

    Constructed from the **parsed** ``#FALLBACK`` string and the dataset's own ``muon_zeff`` table,
    never from module-level literals. That is what makes the directive load-bearing rather than
    decorative: change a coefficient in the file and the model moves with it.
    """

    b0a: float
    b0b: float
    b0c: float
    t1: float
    xmu_coeff: float
    mix: float
    zmin: int
    zmax: int
    zeff: tuple[float, ...]

    @classmethod
    def from_directive(cls, value: str, zeff: Sequence[float]) -> GoulardPrimakoff:
        """Build the model from a ``#FALLBACK`` directive value and a ``muon_zeff`` table."""
        model, coefficients = parse_fallback_directive(value)
        if model != FALLBACK_MODEL:
            raise ValueError(f"expected the {FALLBACK_MODEL!r} model, got {model!r}")
        missing = [name for name in FALLBACK_NAMES if name not in coefficients]
        if missing:
            raise ValueError(f"'#FALLBACK' is missing coefficient(s): {', '.join(missing)}")
        # The clamp bounds must index the table they are declared against. Without this a dataset
        # whose `#FALLBACK` and `muon_zeff` table disagreed would fail with a bare IndexError from
        # inside the model, which tells a consumer nothing about which of the two files is wrong.
        zmin, zmax = int(coefficients["zmin"]), int(coefficients["zmax"])
        if not 0 <= zmin <= zmax < len(zeff):
            raise ValueError(
                f"'#FALLBACK' declares the clamp [{zmin}, {zmax}], which does not index the "
                f"{len(zeff)}-entry muon_zeff table it is evaluated against"
            )
        return cls(
            b0a=float(coefficients["b0a"]),
            b0b=float(coefficients["b0b"]),
            b0c=float(coefficients["b0c"]),
            t1=float(coefficients["t1"]),
            xmu_coeff=float(coefficients["xmu_coeff"]),
            mix=float(coefficients["mix"]),
            zmin=int(coefficients["zmin"]),
            zmax=int(coefficients["zmax"]),
            zeff=tuple(zeff),
        )

    def muon_zeff(self, z: int) -> float:
        """``GetMuonZeff``: clamp into ``[zmin, zmax]``, then index. The clamp IS the model here."""
        return self.zeff[max(min(z, self.zmax), self.zmin)]

    def rate(self, z: int, a: int) -> float:
        """The Goulard-Primakoff capture rate in ns^-1, within the model's **declared domain**.

        The model is declared valid for ``Z >= 1`` and ``A >= 1``. Outside it, this raises rather
        than returning something: Geant4 returns NaN for ``Z = 0``, ``+inf`` for ``A = 0`` and a
        plausible-looking finite *negative* rate for ``Z < 0``, in every case with no coded
        rejection at all -- and a value that looks like a rate but is not one is worse than an
        error, because it propagates. Our own Layer-1 format rejects non-finite floats outright, so
        a conforming consumer has to report a domain error here; this is that consumer.

        What Geant4 actually does at those inputs is not hidden, it is recorded: see
        :meth:`evaluate_unchecked`, ``data/g4/d1/d1_gp_sweep.oracle``, and the finding in
        ``DATASET_D1.md``. Reproducing the library and declaring a safer contract than the library
        are different jobs, and this is the only place the two deliberately differ.
        """
        if z < 1 or a < 1:
            raise ValueError(
                f"the {FALLBACK_MODEL} model is declared for Z >= 1 and A >= 1; (Z={z}, A={a}) is "
                "outside its domain. Geant4 returns a non-finite or negative value here instead of "
                "rejecting the input -- a registered finding, reproduced in evaluate_unchecked()."
            )
        return self.evaluate_unchecked(z, a)

    def evaluate_unchecked(self, z: int, a: int) -> float:
        """The compiled expression itself, domain check and all judgement suspended.

        This is what Geant4 computes, in the compiled association order, for any input the
        arithmetic survives. It exists so that the parity claim covers the library's edges too --
        and so that the difference between "what the library does" and "what this dataset says a
        conforming consumer should do" is visible in the code rather than only in prose.

        The parenthesisation below is not style. C++ ``*`` and ``+`` are left-associative and
        floating-point arithmetic is not associative, so re-grouping any of these products changes
        the result -- and the whole parity claim is that it does not. Each step mirrors one
        sub-expression of the source, in the order the compiler evaluates them.
        """
        r1 = self.muon_zeff(z)
        zeff2 = r1 * r1
        xmu = zeff2 * self.xmu_coeff
        a2ze = 0.5 * float(a) / float(z)
        r2 = 1.0 - xmu
        # `2 * (A - Z)` is INTEGER arithmetic before it meets the double, exactly as written
        # upstream; `std::abs` on a double is `fabs`, which is Python's `abs` on a float.
        neutron_excess = float(2 * (a - z) + abs(a2ze - 1.0))
        bracket = (
            a2ze * self.b0a + 1.0 - (a2ze - 1.0) * self.b0b - neutron_excess * self.b0c / float(a * 4)
        )
        return self.t1 * zeff2 * zeff2 * (r2 * r2) * (1.0 - (1.0 - xmu) * self.mix) * bracket


def capture_rate(
    z: int, a: int, records: Sequence[tuple[int, int, float, float]], model: GoulardPrimakoff
) -> float:
    """``GetMuonCaptureRate(Z, A)`` in ns^-1: Geant4's own scan, then the declared fallback.

    The loop is upstream's, early exit included -- ``records`` must therefore be in **source order**,
    which is sorted by Z and not by ``(Z, A)``. The exit is sound only because the array is sorted by
    Z; reproducing it rather than substituting a dictionary is what makes this a reimplementation of
    the compiled function rather than of what the compiled function was probably meant to do.
    """
    rate = -1.0
    for record_z, record_a, value, _ in records:
        if record_z == z and record_a == a:
            rate = value / MICROSECOND
            break
        if record_z > z:
            break
    if rate < 0.0:
        rate = model.rate(z, a)
    return rate


def sweep_digest(
    records: Sequence[tuple[int, int, float, float]], model: GoulardPrimakoff
) -> str:
    """SHA-256 over the whole ``(Z, A)`` box, as big-endian IEEE-754 binary64 bytes.

    Bytes rather than text, and big-endian rather than native: the digest must mean the same thing
    in C++ and in Python, and a text serialisation would have formatting degrees of freedom while a
    native byte order would depend on the machine. Row-major, Z ascending outermost.
    """
    running = hashlib.sha256()
    for z in range(SWEEP_Z_MIN, SWEEP_Z_MAX + 1):
        for a in range(SWEEP_A_MIN, SWEEP_A_MAX + 1):
            running.update(struct.pack(">d", capture_rate(z, a, records, model)))
    return running.hexdigest()


# --------------------------------------------------------------------------------------------
# the isotope audit -- the one hand-authored input in this chain
# --------------------------------------------------------------------------------------------

#: Where the audit lives, relative to the repository root.
AUDIT_RELPATH = "data/g4/d1/isotope_audit.csv"
#: Its columns, in order. The header must match exactly: a reordered or renamed column is a
#: silent re-interpretation of hand-entered data, which is the one thing this file cannot survive.
AUDIT_COLUMNS = ("Z", "A", "isotope_resolved", "evidence", "locator", "copy_read")


@dataclass(frozen=True)
class IsotopeAuditRow:
    """One capture record's isotope-resolution finding, read from a primary.

    ``locator`` is not decoration. It names the table and page that ESTABLISH this row's flag, and
    it is empty on exactly those rows no primary settles -- so ``settled`` is a property of the
    data rather than a second, driftable column that could disagree with it.
    """

    z: int
    a: int
    isotope_resolved: bool
    evidence: str
    locator: str
    copy_read: str

    @property
    def settled(self) -> bool:
        """True when a primary establishes this row's flag, in either direction."""
        return bool(self.locator)


class IsotopeAuditError(RuntimeError):
    """The audit is malformed. Raised with the row named, never swallowed."""


def _audit_bool(text: str, where: str) -> bool:
    if text not in ("true", "false"):
        raise IsotopeAuditError(f"{where}: isotope_resolved must be 'true' or 'false', got {text!r}")
    return text == "true"


def load_isotope_audit(path: Path) -> dict[tuple[int, int], IsotopeAuditRow]:
    """Parse the audit, refusing anything a generator could otherwise carry into shipped bytes.

    The checks are deliberately unforgiving. This is the only file in the D1 chain a human typed,
    so it is the only one where a mistake cannot be caught by comparing against the vendored
    source -- which makes the structural invariants the whole of the protection available.
    """
    raw = path.read_bytes()
    if b"\r" in raw:
        raise IsotopeAuditError(f"{path.name} contains CR; the audit is committed LF-only")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise IsotopeAuditError(f"{path.name} is not ASCII: {exc}") from None

    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != AUDIT_COLUMNS:
        raise IsotopeAuditError(
            f"{path.name} header is {tuple(reader.fieldnames or ())!r}, expected {AUDIT_COLUMNS!r}"
        )

    rows: dict[tuple[int, int], IsotopeAuditRow] = {}
    for number, record in enumerate(reader, start=2):
        where = f"{path.name} line {number}"
        try:
            z, a = int(record["Z"]), int(record["A"])
        except (TypeError, ValueError):
            raise IsotopeAuditError(f"{where}: Z and A must be integers") from None
        row = IsotopeAuditRow(
            z=z,
            a=a,
            isotope_resolved=_audit_bool(record["isotope_resolved"], where),
            evidence=record["evidence"],
            locator=record["locator"],
            copy_read=record["copy_read"],
        )
        if (z, a) in rows:
            raise IsotopeAuditError(f"{where}: duplicate key ({z}, {a})")
        if not row.evidence:
            raise IsotopeAuditError(f"{where}: every row must state its evidence")
        # The rule this file's own documentation states, enforced rather than trusted.
        if row.isotope_resolved and not row.locator:
            raise IsotopeAuditError(
                f"{where}: isotope_resolved is true with no locator -- a flag with nothing behind it"
            )
        # A locator names a copy that was read; recording one without the other loses exactly the
        # provenance the column pair exists to carry.
        if bool(row.locator) != bool(row.copy_read):
            raise IsotopeAuditError(
                f"{where}: locator and copy_read must be present or absent together, got "
                f"{row.locator!r} and {row.copy_read!r}"
            )
        rows[(z, a)] = row
    if not rows:
        raise IsotopeAuditError(f"{path.name} carries no rows")
    return rows
