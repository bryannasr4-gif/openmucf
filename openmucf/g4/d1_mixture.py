"""What answers a target of natural composition, and the mixture Geant4 makes of it.

A Geant4 user who stops negative muons in a ``G4_<El>`` material that ``G4NistManager`` builds gets a mixture:
``G4ElementSelector::SelectZandA`` draws an isotope with the element's relative abundance, and each isotope's
capture rate comes from whichever rung of the lookup answers it (:func:`openmucf.g4.d1_natural.lookup`): the
selected profile's row keyed by that nuclide, else the compiled-in record keyed by it, else the
Goulard-Primakoff formula. Under every profile but ``parity`` a nuclide the profile does not key falls through
to the compiled-in record, so an evaluated profile can answer a stop from a record measured on a target of
natural composition.

This module classifies each record that can answer by two axes the committed public evidence establishes, the
kind of target its value was measured on and the lineage of the value itself, and maps the pair to an answer
class:

* ``evaluated_isotope``: a row of an evaluated profile keyed by a mass number (``evaluated_natural`` is a row
of one keyed ``A = 0``, which no transported nucleus requests);
* ``parity_isotope``: a compiled-in record of a separated isotope or other isotopically defined target whose
value equals a printed cell exactly or comes from another primary, or of an element with a single natural
isotope;
* ``parity_isotope_lineage_open``: a compiled-in record of a separated isotope whose value equals a printed
cell only at that cell's printed precision, or equals no printed cell;
* ``parity_mixed_reconstructed``: a compiled-in record whose evidence states, and whose stated
inverse-variance combination of printed rows of differently composed targets this module recomputes and
reproduces, that its value is that combination;
* ``parity_natural``: a compiled-in record of a target printed with no mass number;
* ``parity_unsettled``: a compiled-in record whose flag is not established and whose evidence states no
  combination;
* ``formula``: the Goulard-Primakoff formula, which answers every nuclide no record keys.

Standard library only.
"""

from __future__ import annotations

import collections
import csv
import functools
import hashlib
import io
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

from openmucf.g4 import d1_natural as natural
from openmucf.g4 import spec
from openmucf.g4.sources import d1_nuclear_capture as d1
from openmucf.g4.sources import mizuno2025, suzuki1987

#: The generated files, relative to the repository root.
CLASSES_RELPATH = "data/g4/d1/record_classes.csv"
ANSWERS_RELPATH = "data/g4/d1/natural_target_answers.csv"
#: The Layer-2 files of each profile.
PROV_RELPATHS = {
    spec.PARITY_PROFILE: "data/g4/d1/d1_capture.prov.json",
    mizuno2025.PROFILE: "data/g4/d1/d1_capture.mizuno2025.prov.json",
    suzuki1987.PROFILE: "data/g4/d1/d1_capture.suzuki1987.prov.json",
}
MIZUNO_TABLE3_RELPATH = "data/g4/d1/mizuno2025_table3.csv"

TARGET_KINDS = (
    "separated_isotope",
    "single_isotope_element",
    "other_primary_isotopic",
    "natural_composition",
    "mixed_composition",
    "unsettled",
)
LINEAGES = ("printed_exact", "printed_rounded", "combination_stated", "no_cell", "other_source")
EVALUATED_ISOTOPE, EVALUATED_NATURAL = "evaluated_isotope", "evaluated_natural"
PARITY_ISOTOPE, LINEAGE_OPEN = "parity_isotope", "parity_isotope_lineage_open"
MIXED, PARITY_NATURAL, UNSETTLED, FORMULA = (
    "parity_mixed_reconstructed",
    "parity_natural",
    "parity_unsettled",
    "formula",
)
#: The answer classes, in the order every table gives them.
CLASSES = (EVALUATED_ISOTOPE, PARITY_ISOTOPE, LINEAGE_OPEN, MIXED, PARITY_NATURAL, UNSETTLED, FORMULA)
#: The classes a comparison row with a nonzero share in can never qualify for an accuracy verdict.
DISQUALIFYING = (LINEAGE_OPEN, MIXED, UNSETTLED)

CLASS_COLUMNS = (
    "profile",
    "z",
    "a",
    "isotope_resolved",
    "needs_verification",
    "target_kind",
    "target_evidence",
    "value_lineage",
    "lineage_evidence",
    "answer_class",
)
ANSWER_COLUMNS = ("profile", "z", "symbol", "a", "route", "answer_class", "record")


class MixtureError(RuntimeError):
    """A committed input this module cannot classify, or a combination the evidence states and the
    recomputation does not reproduce."""


# --------------------------------------------------------------------------------------------
# the printed cells
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One printed rate cell of Tables III and IV of the published article, scaled to 1e6/s with its
    printed digits kept (``Decimal`` exponent)."""

    locator: str
    z: int
    label: str
    basis: str
    own: bool
    rate: Decimal
    unc: Decimal
    parenthesized: bool


def _quantity_rows(root: Path) -> list[dict[str, str]]:
    raw = (root / suzuki1987.QUANTITY_ROWS_RELPATH).read_bytes()
    return list(csv.DictReader(io.StringIO(raw.decode("ascii"))))


def printed_cells(root: Path) -> dict[str, Cell]:
    """Every printed rate cell with a symmetric uncertainty, by locator."""
    out: dict[str, Cell] = {}
    for row in _quantity_rows(root):
        if not (row["rate"] and row["rate_unc"]):
            continue
        locator = f"Table {row['table']} p.{row['printed_page']} row {row['page_row_ordinal']}"
        out[locator] = Cell(
            locator,
            int(row["Z"]),
            row["target_label"],
            row["target_basis"],
            row["own_measurement"] == "true",
            suzuki1987.scaled(row, "rate"),
            suzuki1987.scaled(row, "rate_unc"),
            row["rate_parenthesized"] == "true",
        )
    return out


def digit_equal(value: str, unc: str, cell: Cell) -> bool:
    """True when the literal pair equals the cell exactly as decimal numbers, the value and the uncertainty:
    no digit of the literal differs from the cell's once trailing zeros are read as zeros (``1.800`` equals a
    printed
    ``1.80``; ``7.547`` does not equal a printed ``7.55``, which it matches only at the cell's precision)."""
    return Decimal(value) == cell.rate and Decimal(unc) == cell.unc


def label_composition(label: str) -> str:
    """The composition a printed target label states: ``A<mass>`` for a mass-number prefix, ``radiogenic``
    for the ``R`` prefix, ``natural`` for ``nat``, ``element`` for a bare symbol."""
    match = re.fullmatch(r"(nat|R|[0-9]+)?([A-Z][a-z]?)(?:\^[a-z])?", label)
    if match is None:
        raise MixtureError(f"unreadable printed label {label!r}")
    prefix = match[1]
    if prefix is None:
        return "element"
    if prefix == "nat":
        return "natural"
    if prefix == "R":
        return "radiogenic"
    return f"A{prefix}"


# --------------------------------------------------------------------------------------------
# stated combinations
# --------------------------------------------------------------------------------------------

_IVW = "inverse-variance weighted mean"
_ROWS_BY_NUMBER = re.compile(r"Table (III|IV) p\.([0-9]+) rows ([0-9]+) and ([0-9]+)")
_ROWS_BY_LABEL = re.compile(r"weighted mean of the ([A-Z][a-z]?-[0-9]+) row and the ([A-Z][a-z]?-[0-9]+) row")
_ROW_THAT = re.compile(r"weighted mean of that row and the ([A-Z][a-z]?-[0-9]+) row")
_NEAREST_OWN = re.compile(r"the nearest in value being the ([A-Z][a-z]?) row of this experiment")
_AT_PRECISION = "at the precision it is printed with"
_WITHIN_UNIT = "to within a unit of their last printed digit"


@dataclass(frozen=True)
class Combination:
    """A combination a record's evidence states: the printed cells it names, the tolerance it states, the
    recomputed inverse-variance weighted mean and its uncertainty, and whether it reproduces the literal."""

    cells: tuple[Cell, ...]
    tolerance: str
    mean: Decimal
    unc: Decimal
    reproduced: bool

    @property
    def mixed(self) -> bool:
        """True when the named cells were measured on targets of different composition."""
        return len({label_composition(cell.label) for cell in self.cells}) > 1


def _published_cell(cells: Mapping[str, Cell], z: int, label: str) -> Cell:
    """The one published cell at ``z`` printed as ``label`` (``Br-79`` is printed ``79Br``)."""
    symbol, _, mass = label.partition("-")
    printed = f"{mass}{symbol}"
    found = [c for c in cells.values() if c.z == z and re.sub(r"\^[a-z]$", "", c.label) == printed]
    if len(found) != 1:
        raise MixtureError(f"Z={z}: {len(found)} published cells print {printed!r}")
    return found[0]


def ivw(cells: Sequence[Cell]) -> tuple[Decimal, Decimal]:
    """The inverse-variance weighted mean of the cells and its uncertainty, at 50 significant digits."""
    with localcontext() as context:
        context.prec = 50
        weights = [1 / (cell.unc * cell.unc) for cell in cells]
        total = sum(weights, Decimal(0))
        mean = sum((w * cell.rate for w, cell in zip(weights, cells, strict=True)), Decimal(0)) / total
        return +mean, (1 / total).sqrt()


def _matches(value: str, computed: Decimal, tolerance: str) -> bool:
    literal = Decimal(value)
    quantum = Decimal(1).scaleb(literal.as_tuple().exponent)  # type: ignore[arg-type]
    if tolerance == _AT_PRECISION:
        return computed.quantize(quantum, rounding=ROUND_HALF_EVEN) == literal
    return abs(computed - literal) <= quantum


def stated_combination(
    z: int, value: str, unc: str, evaluation_method: str, cells: Mapping[str, Cell]
) -> Combination | None:
    """The combination ``evaluation_method`` states, recomputed from the printed cells it names; ``None`` when
    it
    states none. A statement this module cannot resolve to printed cells is refused."""
    if _IVW not in evaluation_method:
        return None
    named: list[Cell] = []
    by_number = _ROWS_BY_NUMBER.search(evaluation_method)
    by_label = _ROWS_BY_LABEL.search(evaluation_method)
    that = _ROW_THAT.search(evaluation_method)
    if by_number:
        table, page = by_number[1], by_number[2]
        named = [cells[f"Table {table} p.{page} row {by_number[n]}"] for n in (3, 4)]
    elif by_label:
        named = [_published_cell(cells, z, by_label[n]) for n in (1, 2)]
    elif that:
        nearest = _NEAREST_OWN.search(evaluation_method)
        if nearest is None:
            raise MixtureError(f"Z={z}: 'that row' has no antecedent")
        own = [c for c in cells.values() if c.z == z and c.own and c.label == nearest[1]]
        if len(own) != 1:
            raise MixtureError(f"Z={z}: {len(own)} rows of this experiment print {nearest[1]!r}")
        named = [own[0], _published_cell(cells, z, that[1])]
    else:
        raise MixtureError(f"Z={z}: a stated combination names no printed rows this module can read")
    if _AT_PRECISION in evaluation_method:
        tolerance = _AT_PRECISION
    elif _WITHIN_UNIT in evaluation_method:
        tolerance = _WITHIN_UNIT
    else:
        raise MixtureError(f"Z={z}: a stated combination states no tolerance")
    mean, sigma = ivw(named)
    reproduced = _matches(value, mean, tolerance) and _matches(unc, sigma, tolerance)
    return Combination(tuple(named), tolerance, mean, sigma, reproduced)


# --------------------------------------------------------------------------------------------
# the record classes
# --------------------------------------------------------------------------------------------


def answer_class(kind: str, lineage: str) -> str:
    """The answer class of a compiled-in record of target kind ``kind`` and value lineage ``lineage``."""
    if kind == "mixed_composition":
        return MIXED
    if kind == "natural_composition":
        return PARITY_NATURAL
    if kind == "unsettled":
        return UNSETTLED
    if kind == "single_isotope_element":
        return PARITY_ISOTOPE
    if kind in ("separated_isotope", "other_primary_isotopic"):
        return PARITY_ISOTOPE if lineage in ("printed_exact", "other_source") else LINEAGE_OPEN
    raise MixtureError(f"no answer class for target kind {kind!r}")


def _prov_rows(root: Path, profile: str) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = json.loads(
        (root / PROV_RELPATHS[profile]).read_text(encoding="utf-8")
    )["rows"]
    return rows


def _single_isotopes(isotopes: Mapping[int, Sequence[natural.Isotope]]) -> dict[int, int]:
    return {z: members[0].a for z, members in isotopes.items() if len(members) == 1}


def _parity_equal_locators(root: Path) -> dict[tuple[int, int], list[str]]:
    raw = (root / suzuki1987.PARITY_CELLS_RELPATH).read_bytes()
    out: dict[tuple[int, int], list[str]] = {}
    for row in csv.DictReader(io.StringIO(raw.decode("ascii"))):
        out[(int(row["Z"]), int(row["A"]))] = [x for x in row["equal_locators"].split("|") if x]
    return out


def compiled_classes(root: Path) -> list[dict[str, str]]:
    """One row per compiled-in record."""
    found = d1.load(root / d1.VENDORED_RELPATH)
    prov = _prov_rows(root, spec.PARITY_PROFILE)
    cells = printed_cells(root)
    equal = _parity_equal_locators(root)
    single = _single_isotopes(natural.load_isotopes(root / natural.ISOTOPES_RELPATH))
    rows: list[dict[str, str]] = []
    for (z, a, _, _), (value, unc) in zip(found.capture_records, found.capture_literals, strict=True):
        record = prov[f"{z}-{a}"]
        method = str(record["evaluation_method"])
        resolved, verify = bool(record["isotope_resolved"]), bool(record["needs_verification"])
        combination = stated_combination(z, value, unc, method, cells)
        if combination is not None and not combination.reproduced:
            raise MixtureError(f"({z}, {a}): the stated combination does not reproduce {value} +- {unc}")
        if combination is not None and combination.mixed:
            kind, kind_evidence = "mixed_composition", "evaluation_method states the combination; recomputed"
            lineage = "combination_stated"
            lineage_evidence = " + ".join(f"{c.locator} {c.label}" for c in combination.cells) + (
                f"; inverse-variance weighted mean {combination.mean:.6f} +- {combination.unc:.6f}"
            )
        else:
            if single.get(z) == a:
                kind, kind_evidence = (
                    "single_isotope_element",
                    "geant4_natural_isotopes.csv: one natural isotope",
                )
            elif resolved and "the cited primary" in method:
                kind, kind_evidence = "other_primary_isotopic", "evaluation_method: the cited primary"
            elif resolved and "separated isotope" in method:
                kind, kind_evidence = (
                    "separated_isotope",
                    "evaluation_method: the primary lists the separated isotope",
                )
            elif not resolved and not verify and "no mass number" in method:
                kind, kind_evidence = (
                    "natural_composition",
                    "evaluation_method: the primary lists the element with no mass number",
                )
            elif verify:
                kind, kind_evidence = "unsettled", "needs_verification true and no stated combination"
            else:
                raise MixtureError(f"({z}, {a}): no target kind follows from the evidence")
            locators = equal.get((z, a), [])
            matched = [cells[loc] for loc in locators if loc in cells]
            if len(matched) != len(locators):
                raise MixtureError(f"({z}, {a}): an equal locator is not a printed rate cell")
            if z <= 2:
                lineage, lineage_evidence = "other_source", "a primary other than the published tables"
            elif any(digit_equal(value, unc, cell) for cell in matched):
                lineage = "printed_exact"
                lineage_evidence = "; ".join(
                    f"{c.locator} {c.label}" for c in matched if digit_equal(value, unc, c)
                )
            elif matched:
                lineage = "printed_rounded"
                lineage_evidence = "; ".join(f"{c.locator} {c.label}" for c in matched)
            else:
                lineage, lineage_evidence = "no_cell", "equal to no published cell at its printed precision"
            for cell in matched:
                composition = label_composition(cell.label)
                if kind == "separated_isotope" and composition != f"A{a}":
                    raise MixtureError(f"({z}, {a}): equal cell {cell.locator} prints {cell.label!r}")
                if kind == "natural_composition" and composition not in ("element", "natural"):
                    raise MixtureError(f"({z}, {a}): equal cell {cell.locator} prints {cell.label!r}")
        rows.append(
            {
                "profile": spec.PARITY_PROFILE,
                "z": str(z),
                "a": str(a),
                "isotope_resolved": str(resolved).lower(),
                "needs_verification": str(verify).lower(),
                "target_kind": kind,
                "target_evidence": kind_evidence,
                "value_lineage": lineage,
                "lineage_evidence": lineage_evidence,
                "answer_class": answer_class(kind, lineage),
            }
        )
    return rows


def _profile_label_kind(label: str, z: int, a: int, single: Mapping[int, int]) -> str:
    composition = label_composition(label)
    if composition == "natural":
        if a != 0:
            raise MixtureError(f"({z}, {a}): {label!r} is a natural-composition label keyed by a mass number")
        return "natural_composition"
    if composition.startswith("A"):
        if int(composition[1:]) != a:
            raise MixtureError(f"({z}, {a}): {label!r} is keyed by another mass number")
        return "separated_isotope"
    if composition == "element" and single.get(z) == a:
        return "single_isotope_element"
    raise MixtureError(f"({z}, {a}): {label!r} admits no key")


def evaluated_rows(root: Path) -> dict[str, list[tuple[int, int, str, str, str, str]]]:
    """Each evaluated profile's rows as ``(z, a, value, unc, printed label, locator)``: the row of the table
    the profile is generated from whose value and uncertainty equal the profile row's, found by value, not by
    key, so that a row keyed wrongly is still found."""
    tables = natural.capture_profiles(root / "data" / "g4" / "d1")
    out: dict[str, list[tuple[int, int, str, str, str, str]]] = {}
    mizuno = mizuno2025.load_table3(root / MIZUNO_TABLE3_RELPATH)
    own = [row for row in _quantity_rows(root) if row["own_measurement"] == "true" and row["rate_unc"]]
    for name, table in tables.items():
        if name == spec.PARITY_PROFILE:
            continue
        found: list[tuple[int, int, str, str, str, str]] = []
        for z, a, value, unc in table.records:
            if name == mizuno2025.PROFILE:
                hits = [
                    (r.nuclide, r.locator, r.rate, r.rate_unc)
                    for r in mizuno
                    if r.z == int(z)
                    and Decimal(r.rate) == Decimal(repr(value))
                    and Decimal(r.rate_unc) == Decimal(repr(unc))
                ]
            elif name == suzuki1987.PROFILE:
                hits = [
                    (
                        r["target_label"],
                        f"Table {r['table']} p.{r['printed_page']} row {r['page_row_ordinal']}",
                        str(suzuki1987.scaled(r, "rate")),
                        str(suzuki1987.scaled(r, "rate_unc")),
                    )
                    for r in own
                    if int(r["Z"]) == int(z)
                    and r["rate_parenthesized"] == "false"
                    and suzuki1987.scaled(r, "rate") == Decimal(repr(value))
                    and suzuki1987.scaled(r, "rate_unc") == Decimal(repr(unc))
                ]
            else:
                raise MixtureError(f"profile {name!r}: no source table is known for its rows")
            if len(hits) != 1:
                raise MixtureError(f"{name} ({z}, {a}): {len(hits)} printed rows carry its value")
            label, locator, printed_value, printed_unc = hits[0]
            found.append((int(z), int(a), printed_value, printed_unc, label, locator))
        out[name] = found
    return out


def profile_classes(root: Path) -> list[dict[str, str]]:
    """One row per row of an evaluated profile. The printed label must agree with the key: a mass-number label
    with that mass number, ``nat`` with ``A = 0``, a bare symbol only for an element with a single natural
    isotope
    and keyed by it."""
    single = _single_isotopes(natural.load_isotopes(root / natural.ISOTOPES_RELPATH))
    rows: list[dict[str, str]] = []
    for name, found in evaluated_rows(root).items():
        prov = _prov_rows(root, name)
        for z, a, _, _, label, locator in found:
            record = prov.get(f"{z}-{a}")
            if record is None:
                raise MixtureError(f"{name} ({z}, {a}): no provenance row")
            kind = _profile_label_kind(label, z, a, single)
            rows.append(
                {
                    "profile": name,
                    "z": str(z),
                    "a": str(a),
                    "isotope_resolved": str(bool(record["isotope_resolved"])).lower(),
                    "needs_verification": str(bool(record["needs_verification"])).lower(),
                    "target_kind": kind,
                    "target_evidence": f"printed label {label!r}",
                    "value_lineage": "printed_exact",
                    "lineage_evidence": locator,
                    "answer_class": EVALUATED_NATURAL if a == 0 else EVALUATED_ISOTOPE,
                }
            )
    return rows


def record_classes(root: Path) -> list[dict[str, str]]:
    """Every compiled-in record, then every evaluated-profile row."""
    return compiled_classes(root) + profile_classes(root)


# --------------------------------------------------------------------------------------------
# what answers each isotope
# --------------------------------------------------------------------------------------------


def answers(root: Path, classes: Sequence[Mapping[str, str]] | None = None) -> list[dict[str, str]]:
    """One row per profile, element (the parity table's validity range) and isotope of its natural
    composition:
    the route the lookup takes, the answer class and the answering record."""
    classes = record_classes(root) if classes is None else classes
    by_key = {(row["profile"], int(row["z"]), int(row["a"])): row["answer_class"] for row in classes}
    found = d1.load(root / d1.VENDORED_RELPATH)
    model = natural.reference_model(found)
    isotopes = natural.load_isotopes(root / natural.ISOTOPES_RELPATH)
    rows: list[dict[str, str]] = []
    for name, table in natural.capture_profiles(root / "data" / "g4" / "d1").items():
        reach = None if name == spec.PARITY_PROFILE else natural.profile_rows(table)
        for z in sorted(isotopes):
            for isotope in isotopes[z]:
                route = natural.lookup(z, isotope.a, reach, found.capture_records, model).route
                if route == natural.PROFILE_ROUTE:
                    klass, record = by_key[(name, z, isotope.a)], f"{name}:{z}-{isotope.a}"
                elif route == natural.TABLE_ROUTE:
                    klass, record = (
                        by_key[(spec.PARITY_PROFILE, z, isotope.a)],
                        f"{spec.PARITY_PROFILE}:{z}-{isotope.a}",
                    )
                else:
                    klass, record = FORMULA, ""
                rows.append(
                    {
                        "profile": name,
                        "z": str(z),
                        "symbol": isotope.symbol,
                        "a": str(isotope.a),
                        "route": route,
                        "answer_class": klass,
                        "record": record,
                    }
                )
    return rows


def class_shares(
    rows: Sequence[Mapping[str, str]], isotopes: Mapping[int, Sequence[natural.Isotope]]
) -> dict[tuple[str, int], dict[str, float]]:
    """``(profile, z)`` to the share of stops each answer class answers: the correctly rounded sum of the
    abundances ``isotopes`` gives the isotopes that class answers."""
    abundance = {(z, i.a): i.abundance for z, members in isotopes.items() for i in members}
    groups: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rows:
        group = groups.setdefault((row["profile"], int(row["z"])), {klass: [] for klass in CLASSES})
        group[row["answer_class"]].append(abundance[(int(row["z"]), int(row["a"]))])
    return {key: {k: math.fsum(v) for k, v in group.items()} for key, group in groups.items()}


def render_csv(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> bytes:
    """``rows`` as LF-terminated CSV bytes."""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("ascii")


def read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """A committed CSV with exactly ``columns``."""
    raw = path.read_bytes()
    if b"\r" in raw:
        raise MixtureError(f"{path.name} contains CR")
    reader = csv.DictReader(io.StringIO(raw.decode("ascii")))
    if tuple(reader.fieldnames or ()) != tuple(columns):
        raise MixtureError(f"{path.name} header is {reader.fieldnames!r}")
    return list(reader)


# --------------------------------------------------------------------------------------------
# the mixture a natural target makes, on each route
# --------------------------------------------------------------------------------------------

STOPPING, ATOMIC = "stopping", "atomic"
ROUTES = (STOPPING, ATOMIC)


@dataclass(frozen=True)
class Component:
    """One isotope of a natural target as one route receives it.

    ``capture`` and ``decay`` are the binary64 rates (ns^-1) the call site computes. On the stopping route
    ``total`` is ``fl(capture + decay)``, the rate ``G4MuonMinusBoundDecay::ApplyYourself`` draws the delay
    from; on the atomic route ``tau``, ``tau_dio`` and ``tau_nc`` are the lifetimes
    ``G4MuonicAtomHelper::ConstructMuonicAtom`` stores, and the delay is exponential with rate ``1 / tau``.
    ``rate`` is the exact rate of the delay distribution and ``p_dio`` the exact probability of the decay
    branch, from those binary64 values in call-site order. ``abundance`` is the exact probability with which
    the selector draws the isotope."""

    a: int
    abundance: Fraction
    capture: float
    decay: float
    total: float
    tau: float
    tau_dio: float
    tau_nc: float
    rate: Fraction
    p_dio: Fraction
    lookup: str
    answer_class: str
    record: str
    record_unc: Decimal | None


def stopping_branch(capture: float, total: float) -> Fraction:
    """The decay-branch probability of ``G4MuonMinusBoundDecay::ApplyYourself``: capture iff
    ``U * total < capture``."""
    return 1 - Fraction(capture) / Fraction(total)


def atomic_branch(tau_dio: float, tau_nc: float) -> Fraction:
    """The decay-branch probability of ``G4MuonicAtomDecay::DecayIt``: ``l1 = 1/tau_dio``, ``l2 = 1/tau_nc``,
    ``l = l1 + l2`` in binary64 in that order, DIO iff ``U * l < l1``."""
    l1 = 1.0 / tau_dio
    l2 = 1.0 / tau_nc
    return Fraction(l1) / Fraction(l1 + l2)


class Oracle:
    """The analytic mixture model from the committed tables: never from a harvest."""

    def __init__(self, root: Path) -> None:
        from openmucf.g4 import d1_lifetime as lifetime

        self.root = root
        self.found = d1.load(root / d1.VENDORED_RELPATH)
        self.helper_found = d1.load_helper(root / d1.HELPER_RELPATH)
        self.model = natural.reference_model(self.found)
        self.helper_model = natural.reference_model(self.helper_found)
        self.zmin, self.zmax = lifetime.helper_clamp(self.helper_found)
        self.decay_oracle = lifetime.load_oracle(root / lifetime.ORACLE_RELPATH)
        self.mu_mass = self.decay_oracle.constants["mu_mass"]
        self.isotopes = natural.load_isotopes(root / natural.ISOTOPES_RELPATH)
        self.tables = natural.capture_profiles(root / "data" / "g4" / "d1")
        self.classes = {
            (r["profile"], int(r["z"]), int(r["a"])): r["answer_class"]
            for r in read_csv(root / CLASSES_RELPATH, CLASS_COLUMNS)
        }
        self.compiled_unc = {(z, a): unc for z, a, _, unc in self.found.capture_records}
        self._lifetime = lifetime

    def components(self, profile: str, route: str, z: int) -> tuple[Component, ...]:
        """The isotopes of element ``z`` as ``route`` receives them under ``profile``."""
        lifetime = self._lifetime
        table = self.tables[profile]
        reach = None if profile == spec.PARITY_PROFILE else natural.profile_rows(table)
        profile_unc = {(int(z_), int(a_)): float(u) for z_, a_, _, u in table.records}
        members = self.isotopes[z]
        weight_sum = sum((Fraction(m.abundance) for m in members), Fraction(0))
        out: list[Component] = []
        for member in members:
            a = member.a
            found = self.found if route == STOPPING else self.helper_found
            model = self.model if route == STOPPING else self.helper_model
            looked = natural.lookup(z, a, reach, found.capture_records, model)
            capture = looked.rate
            if route == STOPPING:
                mass = self.decay_oracle.light[(z, a)][0] if z < lifetime.BOUND_THRESHOLD_Z else math.nan
                decay = lifetime.bound_decay_rate(z, a, self.mu_mass, mass)
                total = capture + decay
                tau = tau_dio = tau_nc = math.nan
                rate, p_dio = Fraction(total), stopping_branch(capture, total)
            else:
                decay = lifetime.helper_decay_rate(z, self.helper_found.zeff, self.zmin, self.zmax)
                total = capture + decay
                tau = 1.0 / total
                tau_dio, tau_nc = 1.0 / decay, 1.0 / capture
                rate, p_dio = 1 / Fraction(tau), atomic_branch(tau_dio, tau_nc)
            if looked.route == natural.PROFILE_ROUTE:
                klass, record = self.classes[(profile, z, a)], f"{profile}:{z}-{a}"
                unc: Decimal | None = Decimal(repr(profile_unc[(z, a)]))
            elif looked.route == natural.TABLE_ROUTE:
                klass, record = self.classes[(spec.PARITY_PROFILE, z, a)], f"{spec.PARITY_PROFILE}:{z}-{a}"
                unc = Decimal(repr(self.compiled_unc[(z, a)]))
            else:
                klass, record, unc = FORMULA, "", None
            out.append(
                Component(
                    a,
                    Fraction(member.abundance) / weight_sum,
                    capture,
                    decay,
                    total,
                    tau,
                    tau_dio,
                    tau_nc,
                    rate,
                    p_dio,
                    looked.route,
                    klass,
                    record,
                    unc,
                )
            )
        return tuple(out)


def mean_rate(components: Sequence[Component]) -> Fraction:
    """``sum a_i rate_i``: the mixture's initial disappearance rate (ns^-1), exact."""
    return sum((c.abundance * c.rate for c in components), Fraction(0))


def capture_fraction(components: Sequence[Component]) -> Fraction:
    """``sum a_i (1 - p_dio,i)``, exact."""
    return sum((c.abundance * (1 - c.p_dio) for c in components), Fraction(0))


def _dec(value: Fraction | Decimal | float | int) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(value)


def survival(components: Sequence[Component], t: float, prec: int = 50) -> Decimal:
    """``S(t) = sum a_i exp(-rate_i t)`` at ``prec`` digits."""
    with localcontext() as context:
        context.prec = prec
        return +sum((_dec(c.abundance) * (-_dec(c.rate) * Decimal(t)).exp() for c in components), Decimal(0))


def dio_probabilities(
    components: Sequence[Component], edges: Sequence[float], prec: int = 50
) -> list[Decimal]:
    """The probability that a stop decays in orbit with its electron in each interval of ``edges`` (the last
    interval open to infinity), then the capture probability: ``sum a_i p_i (e^{-r_i t_j} - e^{-r_i t_j+1})``.
    Each term is evaluated at ``prec`` digits from the exact inputs; the relative error of a probability is
    below
    ``10**(5 - prec)``."""
    with localcontext() as context:
        context.prec = prec
        bounds = [Decimal(t) for t in edges]
        out = [Decimal(0)] * len(bounds)
        for c in components:
            weight = _dec(c.abundance * c.p_dio)
            rate = _dec(c.rate)
            tails = [(-rate * b).exp() for b in bounds] + [Decimal(0)]
            for j in range(len(bounds)):
                out[j] += weight * (tails[j] - tails[j + 1])
        out.append(_dec(capture_fraction(components)))
        return [+p for p in out]


# --------------------------------------------------------------------------------------------
# the estimator: the rate a lifetime fit extracts from an expected time spectrum
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitSpec:
    """One fit of an expected decay-electron time spectrum.

    The spectrum is the target's DIO-electron density, aged by each beam pulse (``pulses``: weight and age of
    the stops at the time origin), scaled to the count share left by the nuisance components, plus each
    nuisance component ``(rate, count share)`` (rates in ns^-1; a count share is of the total count from the
    time origin to infinity). The model fitted on ``[t1, t2]`` is a free-rate, free-amplitude exponential for
    the target, plus, for each nuisance component, an exponential of its rate with a free amplitude
    (``free_nuisance``) or with its amplitude fixed at the true value, plus a free constant when ``constant``.
    ``method`` is ``ll`` (expected Poisson log-likelihood) or ``ls`` (least squares weighted by the expected
    counts)."""

    method: str
    t1: float
    t2: float
    constant: bool
    pulses: tuple[tuple[float, float], ...]
    nuisance: tuple[tuple[float, float, bool], ...]


def _gauss_legendre(order: int, conv, exp_unused=None):  # type: ignore[no-untyped-def]
    """Nodes and weights of ``order``-point Gauss-Legendre quadrature on [-1, 1], by Newton iteration on the
    Legendre polynomial, in the number type ``conv`` makes."""
    nodes, weights = [], []
    one, two = conv(1), conv(2)
    pi = _pi(conv)
    for i in range(1, order + 1):
        x = _cos(pi * (conv(i) - conv(1) / conv(4)) / (conv(order) + conv(1) / conv(2)), conv)
        for _ in range(100):
            p0, p1 = one, x
            for k in range(2, order + 1):
                p0, p1 = p1, ((two * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = order * (x * p1 - p0) / (x * x - one)
            step = p1 / dp
            x -= step
            if abs(step) < conv(10) ** -60 or (isinstance(x, float) and abs(step) < 1e-16):
                break
        p0, p1 = one, x
        for k in range(2, order + 1):
            p0, p1 = p1, ((two * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = order * (x * p1 - p0) / (x * x - one)
        nodes.append(x)
        weights.append(two / ((one - x * x) * dp * dp))
    return nodes, weights


def _pi(conv):  # type: ignore[no-untyped-def]
    if conv is float:
        return math.pi

    # Machin: pi = 16 atan(1/5) - 4 atan(1/239)
    def atan_inv(n: int) -> Decimal:
        x = Decimal(1) / n
        x2 = x * x
        term, total, k = x, x, 1
        limit = Decimal(10) ** -(localcontext_prec() + 5)
        while True:
            term *= -x2
            add = term / (2 * k + 1)
            if abs(add) < limit:
                break
            total += add
            k += 1
        return total

    return 16 * atan_inv(5) - 4 * atan_inv(239)


def _cos(x, conv):  # type: ignore[no-untyped-def]
    if conv is float:
        return math.cos(x)
    term, total, k = Decimal(1), Decimal(1), 0
    while True:
        k += 2
        term *= -x * x / ((k - 1) * k)
        if term == 0 or abs(term) < Decimal(10) ** -(2 * localcontext_prec()):
            break
        total += term
    return total


def localcontext_prec() -> int:
    import decimal

    return decimal.getcontext().prec


def _exp(x, conv):  # type: ignore[no-untyped-def]
    return math.exp(x) if conv is float else x.exp()


class _Quadrature:
    def __init__(self, t1, t2, fastest, conv, order: int = 16) -> None:  # type: ignore[no-untyped-def]
        base_nodes, base_weights = _gauss_legendre(order, conv)
        span = t2 - t1
        panels = max(4, int(math.ceil(float(span) * float(fastest) / 4.0)))
        h = span / panels
        self.t, self.w = [], []
        for p in range(panels):
            lo = t1 + h * p
            for x, w in zip(base_nodes, base_weights, strict=True):
                self.t.append(lo + h * (x + 1) / 2)
                self.w.append(w * h / 2)


def fit_rate(components: Sequence[Component], spec_: FitSpec, conv=float, start: float = 1.0):  # type: ignore[no-untyped-def]
    """The target rate (ns^-1) the fit ``spec_`` extracts from the expected spectrum of ``components``, in the
    number type ``conv`` makes (``float`` or ``Decimal``); Newton's iteration starts from ``start`` times the
    spectrum's initial target rate."""
    rates = [conv(c.rate.numerator) / conv(c.rate.denominator) for c in components]
    weights = [
        conv((c.abundance * c.p_dio * c.rate).numerator) / conv((c.abundance * c.p_dio * c.rate).denominator)
        for c in components
    ]
    return _fit(rates, weights, spec_, conv, start)


def _fit(rates, weights, spec_: FitSpec, conv, start: float = 1.0):  # type: ignore[no-untyped-def]
    zero, one = conv(0), conv(1)
    # age the target by each pulse and normalise it to the count share the nuisance components leave
    aged = [
        sum((conv(q) * w * _exp(-r * conv(s), conv) for q, s in spec_.pulses), zero)
        for r, w in zip(rates, weights, strict=True)
    ]
    count = sum((w / r for w, r in zip(aged, rates, strict=True)), zero)
    share = one - sum((conv(f) for _, f, _ in spec_.nuisance), zero)
    target_amp = [w * share / count for w in aged]
    nuis = [(conv(mu), conv(f) * conv(mu), free) for mu, f, free in spec_.nuisance]
    fastest = max([float(r) for r in rates] + [float(mu) for mu, _, _ in nuis])
    quad = _Quadrature(conv(spec_.t1), conv(spec_.t2), fastest, conv)
    nodes, qw = quad.t, quad.w
    truth = [
        sum((a * _exp(-r * t, conv) for a, r in zip(target_amp, rates, strict=True)), zero)
        + sum((b * _exp(-mu * t, conv) for mu, b, _ in nuis), zero)
        for t in nodes
    ]
    fixed = [sum((b * _exp(-mu * t, conv) for mu, b, free in nuis if not free), zero) for t in nodes]
    free_basis = [[_exp(-mu * t, conv) for t in nodes] for mu, _, free in nuis if free]
    if spec_.constant:
        free_basis.append([one for _ in nodes])
    # parameters: rate, target amplitude, then one amplitude per free basis function
    lam = (
        conv(start)
        * sum((a * r for a, r in zip(target_amp, rates, strict=True)), zero)
        / sum(target_amp, zero)
    )
    amp = sum(target_amp, zero)
    coef = [b for mu, b, free in nuis if free] + ([zero] if spec_.constant else [])
    n_par = 2 + len(coef)
    tol = conv(10) ** -(localcontext_prec() - 6) if conv is not float else 1e-15
    slack = conv(10) ** -(localcontext_prec() - 4) if conv is not float else 1e-13

    def evaluate(lam, amp, coef):  # type: ignore[no-untyped-def]
        """The objective, its gradient and Hessian; ``None`` where the model is not positive on a node of a
        likelihood fit."""
        grad = [zero] * n_par
        hess = [[zero] * n_par for _ in range(n_par)]
        objective = zero
        magnitude = zero
        for k, t in enumerate(nodes):
            e = _exp(-lam * t, conv)
            m = amp * e + fixed[k] + sum((c * col[k] for c, col in zip(coef, free_basis, strict=True)), zero)
            g = truth[k]
            d = [-amp * t * e, e] + [col[k] for col in free_basis]
            d2_ll = amp * t * t * e
            d2_la = -t * e
            w = qw[k]
            if spec_.method == "ll":
                if m <= 0:
                    return None
                term = w * (g * _log(m, conv) - m)
                objective += term
                magnitude += abs(term) + w * m
                r = g / m - one
                s = g / (m * m)
                factor = one
            else:
                objective -= w * (g - m) * (g - m) / g
                magnitude += w * (g + m)
                r = (g - m) / g
                s = one / g
                factor = conv(2)
            for p in range(n_par):
                grad[p] += factor * w * r * d[p]
                for q in range(p, n_par):
                    hess[p][q] -= factor * w * s * d[p] * d[q]
            hess[0][0] += factor * w * r * d2_ll
            hess[0][1] += factor * w * r * d2_la
        for p in range(n_par):
            for q in range(p):
                hess[p][q] = hess[q][p]
        return objective, grad, hess, magnitude

    state = evaluate(lam, amp, coef)
    if state is None:
        raise MixtureError("the starting model is not positive")
    # Newton's method with Levenberg-Marquardt damping: a step that does not ascend is retried with the
    # diagonal of the negated Hessian scaled up, which tames the near-degenerate directions free nuisance
    # amplitudes and a free constant can make over a short window
    damping = zero
    for _ in range(400):
        objective, grad, hess, magnitude = state
        neg = [[-h for h in row] for row in hess]
        for _retry in range(80):
            matrix = [
                [neg[p][q] * (one + damping) if p == q else neg[p][q] for q in range(n_par)]
                for p in range(n_par)
            ]
            try:
                step = _solve(matrix, grad)
            except MixtureError:
                step = None
            if step is not None:
                trial = (lam + step[0], amp + step[1], [c + s for c, s in zip(coef, step[2:], strict=True)])
                new_state = evaluate(*trial)
                if new_state is not None and new_state[0] >= objective - slack * magnitude:
                    break
            damping = conv(1) / 1000 if damping == 0 else damping * 10
        else:
            raise MixtureError("the fit found no ascent step")
        moved = abs(trial[0] - lam)
        lam, amp, coef = trial
        state = new_state
        if moved <= tol * abs(lam) and damping < conv(1) / 1000:
            return lam
        damping = zero if damping <= conv(1) / 1000 else damping / 100
    raise MixtureError("the fit did not converge")


def _log(x, conv):  # type: ignore[no-untyped-def]
    return math.log(x) if conv is float else x.ln()


def _solve(matrix, vector):  # type: ignore[no-untyped-def]
    """Gaussian elimination with partial pivoting."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if a[col][col] == 0:
            raise MixtureError("singular normal matrix")
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    x = [a[0][0] * 0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - sum((a[r][c] * x[c] for c in range(r + 1, n)), a[0][0] * 0)) / a[r][r]
    return x


# --------------------------------------------------------------------------------------------
# exact statistics
# --------------------------------------------------------------------------------------------


@functools.cache
def _decimal_pi(prec: int) -> Decimal:
    with localcontext() as context:
        context.prec = prec + 10
        value = _pi(Decimal)
    with localcontext() as context:
        context.prec = prec
        return +value


def _erfc(z: Decimal, prec: int) -> Decimal:
    """``erfc(z)`` for ``z >= 0`` by the Maclaurin series of erf at enough guard digits to absorb its
    cancellation."""
    guard = int(float(z) ** 2 / 2.302585) + 20
    with localcontext() as context:
        context.prec = prec + guard
        z2 = z * z
        term, total, n = z, z, 0
        while True:
            n += 1
            term *= -z2 / n
            add = term / (2 * n + 1)
            total += add
            if abs(add) < Decimal(10) ** -(prec + guard):
                break
        pi = _decimal_pi(prec + guard)
        value = 1 - 2 * total / pi.sqrt()
    with localcontext() as context:
        context.prec = prec
        return +value


def chi2_sf(x: Decimal, dof: int, prec: int = 60) -> Decimal:
    """The upper tail ``P(X > x)`` of the chi-square distribution with ``dof`` degrees of freedom: for even
    ``dof = 2m`` it is ``e^{-y} sum_{j<m} y^j / j!`` with ``y = x/2``; for odd ``dof = 2m + 1`` it is
    ``erfc(sqrt y) + e^{-y} sum_{j<m} y^{j+1/2} / Gamma(j + 3/2)``."""
    with localcontext() as context:
        context.prec = prec + 20
        y = x / 2
        if dof % 2 == 0:
            term, total = Decimal(1), Decimal(1)
            for j in range(1, dof // 2):
                term *= y / j
                total += term
            value = (-y).exp() * total
        else:
            pi = _decimal_pi(prec + 20)
            term = y.sqrt() / (pi.sqrt() / 2)  # y^{1/2} / Gamma(3/2)
            total = Decimal(0)
            for j in range(dof // 2):
                total += term
                term *= y / (Decimal(j) + Decimal(3) / 2)
            value = _erfc(y.sqrt(), prec + 20) + (-y).exp() * total
    with localcontext() as context:
        context.prec = prec
        return +value


@dataclass(frozen=True)
class Critical:
    """An enclosure ``[lo, hi]`` of the chi-square critical value at upper-tail probability ``alpha``,
    certified
    by ``sf(lo) > alpha > sf(hi)``."""

    dof: int
    alpha: Decimal
    lo: Decimal
    hi: Decimal


def critical(dof: int, alpha: Decimal, width: Decimal = Decimal("1e-20")) -> Critical:
    lo, hi = Decimal(0), Decimal(max(10, 4 * dof + 200))
    while hi - lo > width:
        mid = (lo + hi) / 2
        if chi2_sf(mid, dof) > alpha:
            lo = mid
        else:
            hi = mid
    if not (chi2_sf(lo, dof) > alpha > chi2_sf(hi, dof)):
        raise MixtureError(f"the critical value for {dof} dof is not enclosed")
    return Critical(dof, alpha, lo, hi)


def pearson(counts: Sequence[int], probabilities: Sequence[Fraction | Decimal]) -> tuple[Decimal, Decimal]:
    """Pearson's X^2 of ``counts`` against ``probabilities`` (which sum to one), as an enclosure: exact from
    fractions, or from 50-digit decimals with a bound of ``1e-30`` on the error."""
    n = sum(counts)
    fractions = [p for p in probabilities if isinstance(p, Fraction)]
    if len(fractions) == len(probabilities):
        exact = sum(
            ((Fraction(k) - n * p) ** 2 / (n * p) for k, p in zip(counts, fractions, strict=True)),
            Fraction(0),
        )
        with localcontext() as context:
            context.prec = 60
            lo = Decimal(exact.numerator) / Decimal(exact.denominator)
        return lo - Decimal("1e-50"), lo + Decimal("1e-50")
    with localcontext() as context:
        context.prec = 60
        value = sum(
            (
                (Decimal(k) - n * _dec(p)) ** 2 / (n * _dec(p))
                for k, p in zip(counts, probabilities, strict=True)
            ),
            Decimal(0),
        )
    return value - Decimal("1e-30"), value + Decimal("1e-30")


def verdict(x2: tuple[Decimal, Decimal], crit: Critical) -> str:
    """``pass`` when the statistic lies below the critical value, ``fail`` when above it, ``refuse`` when the
    enclosures overlap."""
    if x2[1] < crit.lo:
        return "pass"
    if x2[0] > crit.hi:
        return "fail"
    return "refuse"


# --------------------------------------------------------------------------------------------
# power (sizing only, binary64)
# --------------------------------------------------------------------------------------------


def _gamma_q(s: float, x: float) -> float:
    """The regularized upper incomplete gamma function Q(s, x), by the series or the continued fraction."""
    if x <= 0:
        return 1.0
    if x < s + 1:
        term = total = 1.0 / s
        k = s
        while abs(term) > abs(total) * 1e-17:
            k += 1
            term *= x / k
            total += term
        return 1.0 - total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    b = x + 1 - s
    c = 1e300
    d = 1 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(-x + s * math.log(x) - math.lgamma(s)) * h


def noncentral_sf(x: float, dof: int, nc: float) -> float:
    """``P(X > x)`` for the noncentral chi-square with ``dof`` degrees of freedom and noncentrality ``nc``:
    the
    Poisson-weighted sum of central tails."""
    half = nc / 2
    total = 0.0
    j_max = int(half + 12 * math.sqrt(half + 1) + 40)
    for j in range(j_max):
        log_w = (
            -half + (j * math.log(half) if half > 0 else (0.0 if j == 0 else -math.inf)) - math.lgamma(j + 1)
        )
        total += math.exp(log_w) * _gamma_q(dof / 2 + j, x / 2)
    return total


def required_noncentrality(dof: int, crit: float, power: float = 0.99) -> float:
    """The noncentrality at which a test with ``dof`` degrees of freedom and critical value ``crit`` has the
    given power."""
    lo, hi = 0.0, 1.0
    while noncentral_sf(crit, dof, hi) < power:
        hi *= 2
    while hi - lo > 1e-7 * hi:
        mid = (lo + hi) / 2
        if noncentral_sf(crit, dof, mid) < power:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------------------------
# the measured mean lives and the estimators their sources describe
# --------------------------------------------------------------------------------------------

ESTIMATORS_RELPATH = "data/g4/d1/mean_life_estimators.csv"
POLICY_RELPATH = "data/g4/d1/natural_mixture_policy.json"
ESTIMATOR_COLUMNS = ("source", "scope", "fact", "value", "status", "locator")
ESTIMATOR_STATUSES = ("printed", "not_stated")
#: The inputs an estimator's asymptotic expectation needs (section 4 of the scientific contract): it is
#: ESTABLISHED for a row only when each is printed for that row.
ESTABLISHING_FACTS = (
    "components",
    "fit_method",
    "weighting",
    "fit_range",
    "nontarget_amplitudes",
    "beam_structure",
)
ESTABLISHED, SURROGATE = "ESTABLISHED", "SURROGATE"
SUZUKI, MIZUNO = suzuki1987.PROFILE, mizuno2025.PROFILE


@dataclass(frozen=True)
class Reference:
    """One measured mean life of a target of natural composition: a source's own row of an element with more
    than one natural isotope, printed with no mass number or as natural."""

    source: str
    locator: str
    label: str
    z: int
    basis: str
    key_a: int | None
    measured_rate: str
    measured_rate_unc: str


def references(root: Path) -> list[Reference]:
    """The rows of ``mean_life_comparison.csv`` under ``parity`` whose target is printed with no mass number
    or
    as natural, of an element whose natural composition has more than one isotope, in the comparison's
    order."""
    from openmucf.g4 import d1_lifetime as lifetime

    isotopes = natural.load_isotopes(root / natural.ISOTOPES_RELPATH)
    out: list[Reference] = []
    for row in lifetime.read_comparison(root / lifetime.COMPARISON_RELPATH):
        if row["profile"] != spec.PARITY_PROFILE or row["basis"] not in ("unspecified", "natural"):
            continue
        if len(isotopes[int(row["z"])]) < 2:
            continue
        out.append(
            Reference(
                row["source"],
                row["locator"],
                row["label"],
                int(row["z"]),
                row["basis"],
                int(row["key_a"]) if row["key_a"] else None,
                row["measured_rate"],
                row["measured_rate_unc"],
            )
        )
    return out


def load_estimators(root: Path) -> list[dict[str, str]]:
    """The committed transcription of each source's estimator, refusing an unknown status or a printed fact
    with
    no value."""
    rows = read_csv(root / ESTIMATORS_RELPATH, ESTIMATOR_COLUMNS)
    for row in rows:
        if row["status"] not in ESTIMATOR_STATUSES:
            raise MixtureError(f"estimator fact {row['fact']!r}: status {row['status']!r}")
        if (row["status"] == "printed") != bool(row["value"]):
            raise MixtureError(
                f"estimator fact {row['fact']!r}: a printed fact carries its value, no other does"
            )
        if not row["locator"]:
            raise MixtureError(f"estimator fact {row['fact']!r}: no locator")
    return rows


def _facts(
    rows: Sequence[Mapping[str, str]], source: str, scopes: Sequence[str]
) -> dict[str, Mapping[str, str]]:
    """A source's facts for a row, the most specific scope winning."""
    out: dict[str, Mapping[str, str]] = {}
    for scope in scopes:
        for row in rows:
            if row["source"] == source and row["scope"] == scope:
                out[row["fact"]] = row
    return out


def _mizuno_form(root: Path, locator: str) -> str:
    """``in_case`` or ``self_supported``, from the target form Table 1 prints for the row."""
    number = int(locator.rsplit(" ", 1)[1])
    rows = mizuno2025.load_table1(root / "data" / "g4" / "d1" / "mizuno2025_table1.csv")
    return "in_case" if "case" in rows[number - 1].form else "self_supported"


def row_facts(root: Path, ref: Reference, rows: Sequence[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    if ref.source == SUZUKI:
        return _facts(rows, SUZUKI, ("all", ref.locator))
    return _facts(rows, MIZUNO, ("all", _mizuno_form(root, ref.locator), ref.locator))


def estimator_label(facts: Mapping[str, Mapping[str, str]]) -> tuple[str, list[str]]:
    """ESTABLISHED when every establishing fact is printed and states what the expectation needs; otherwise
    SURROGATE, with the facts that are missing. A fact whose printed value says a needed part is not printed
    (the
    nominal fit range, an amplitude, the pulse separation) counts as missing."""
    missing: list[str] = []
    for fact in ESTABLISHING_FACTS:
        row = facts.get(fact)
        if (
            row is None
            or row["status"] != "printed"
            or "not printed" in row["value"]
            or "not stated" in row["value"]
        ):
            missing.append(fact)
    for fact in ("cno_fraction", "ag_fraction", "bg_fraction", "carbon_fraction"):
        row = facts.get(fact)
        if row is not None and (
            row["status"] != "printed" or "approximately" in row["value"] or "typically" in row["value"]
        ):
            missing.append(fact)
    return (ESTABLISHED if not missing else SURROGATE), missing


def _quantity_mean_life(root: Path, locator: str) -> float:
    for row in _quantity_rows(root):
        if f"Table {row['table']} p.{row['printed_page']} row {row['page_row_ordinal']}" == locator:
            return float(row["mean_life_ns"])
    raise MixtureError(f"no mean life at {locator}")


def _printed_ns(text: str) -> float:
    """The central value of a printed lifetime such as ``87.0(5) ns`` or ``1.94(8) us``, in ns."""
    match = re.match(r"([0-9.]+)\([0-9]+\) (ns|us)", text)
    if match is None:
        raise MixtureError(f"unreadable printed lifetime {text!r}")
    return float(match[1]) * (1000.0 if match[2] == "us" else 1.0)


# --------------------------------------------------------------------------------------------
# the surrogate brackets
# --------------------------------------------------------------------------------------------

#: The compound each element of a Suzuki row was measured in, where the source's list of powders and liquids
#: names one (p.2213), and the Cr2O3 of its Fig. 3 (p.2214): the partner whose muonic decay the fit carries.
SUZUKI_PARTNERS = {17: "C", 20: "F", 24: "O", 32: "O", 56: "O", 60: "O", 80: "O", 82: "F"}
#: The Table III rows whose mean life gives each partner's lifetime (carbon, oxygen, fluorine as printed).
PARTNER_LOCATORS = {
    "C": "Table III p.2217 row 37",
    "O": "Table III p.2217 row 49",
    "F": "Table III p.2217 row 54",
}
DEFAULT_SHARES = (0.0, 0.05, 0.20)
# : The beam of a pulsed source when the separation is not printed: one prompt pulse, or two equal pulses this
# far
#: apart (a policy value: the source prints no separation).
PULSE_SEPARATION_NS = 600.0
MIZUNO_ORIGIN_NS = 40.0


def bracket(root: Path, ref: Reference, tau_bar: float) -> tuple[list[tuple[str, FitSpec]], str]:
    """Every fit of the row's surrogate bracket as ``(label, spec)``, and the label of the nominal fit (the
    one the tabulated-uncertainty derivatives are taken at). Lifetimes come from the printed rows and the
    transcription; shares and ranges not printed take the policy's values."""
    rows = load_estimators(root)
    facts = row_facts(root, ref, rows)
    out: list[tuple[str, FitSpec]] = []
    ranges = [(t1, t2) for t1 in (0.0, 2.0) for t2 in (6.0, 12.0)]
    if ref.source == SUZUKI:
        carbon_rate = 1.0 / _quantity_mean_life(root, PARTNER_LOCATORS["C"])
        partner = SUZUKI_PARTNERS.get(ref.z)
        partner_shares = DEFAULT_SHARES if partner else (0.0,)
        # a carbon target's scintillator background is muonic carbon itself: a free carbon amplitude would be
        # the target's own exponential, so a carbon row carries no carbon component
        carbon_options: list[tuple[bool | None, float]] = (
            [(None, 0.0)]
            if ref.z == 6
            else [(free, share) for free in (True, False) for share in (0.01, 0.02)]
        )
        for method in ("ll", "ls"):
            for constant in (False, True):
                for t1, t2 in ranges:
                    for carbon_free, carbon in carbon_options:
                        for share in partner_shares:
                            nuis: list[tuple[float, float, bool]] = []
                            c_share = carbon + (share if partner == "C" else 0.0)
                            if carbon_free is not None:
                                nuis.append((carbon_rate, c_share, carbon_free))
                            if partner and partner != "C":
                                rate = 1.0 / _quantity_mean_life(root, PARTNER_LOCATORS[partner])
                                nuis.append((rate, share, True))
                            label = (
                                f"{method} const={int(constant)} t1={t1:g} t2={t2:g} carbon="
                                + (
                                    "none"
                                    if carbon_free is None
                                    else f"{'free' if carbon_free else 'fixed'}:{carbon:g}"
                                )
                                + (f" {partner}={share:g}" if partner else "")
                            )
                            out.append(
                                (
                                    label,
                                    FitSpec(
                                        method,
                                        t1 * tau_bar,
                                        t2 * tau_bar,
                                        constant,
                                        ((1.0, 0.0),),
                                        tuple(nuis),
                                    ),
                                )
                            )
        nominal = (
            "ll const=0 t1=0 t2=12 carbon="
            + ("none" if ref.z == 6 else "free:0.01")
            + (f" {partner}=0.05" if partner else "")
        )
        return out, nominal
    form = _mizuno_form(root, ref.locator)
    tau_ag = _printed_ns(facts["tau_Ag"]["value"])
    tau_cno = _printed_ns(facts["tau_CNO"]["value"])
    tau_bg = _printed_ns(facts["tau_bg"]["value"])
    printed_cno = 0.02 if form == "in_case" else 0.01
    natag = ref.label == "natAg"
    ag_shares = DEFAULT_SHARES if (form == "in_case" and not natag) else (None,)
    beams = {
        "single": ((1.0, MIZUNO_ORIGIN_NS),),
        "double": ((0.5, MIZUNO_ORIGIN_NS), (0.5, MIZUNO_ORIGIN_NS + PULSE_SEPARATION_NS)),
    }
    for t1, t2 in ranges:
        for cno in (printed_cno / 2, printed_cno, 2 * printed_cno):
            for ag in ag_shares:
                for bg in DEFAULT_SHARES:
                    for beam_name, pulses in beams.items():
                        nuis = [(1.0 / tau_cno, cno, True), (1.0 / tau_bg, bg, True)]
                        if ag is not None:
                            nuis.insert(0, (1.0 / tau_ag, ag, True))
                        label = (
                            f"ll t1={t1:g} t2={t2:g} cno={cno:g}"
                            + (f" ag={ag:g}" if ag is not None else "")
                            + f" bg={bg:g} beam={beam_name}"
                        )
                        out.append(
                            (label, FitSpec("ll", t1 * tau_bar, t2 * tau_bar, False, pulses, tuple(nuis)))
                        )
    nominal = (
        f"ll t1=0 t2=12 cno={printed_cno:g}"
        + (" ag=0.05" if ag_shares != (None,) else "")
        + " bg=0.05 beam=single"
    )
    return out, nominal


# --------------------------------------------------------------------------------------------
# the harvest roster and the policy
# --------------------------------------------------------------------------------------------

REVISIONS = ("v11.4.2", "v11.5.0.beta")
SEED_BASE, Z_STRIDE, ROUTE_STRIDE = 1_000_000_000, 10_000_000, 5_000_000
IDENTITY_SEED_BASE, PILOT_SEED_BASE = 2_000_000_000, 3_000_000_000


def seed_base(z: int, route: str, base: int = SEED_BASE) -> int:
    """The seed of event 0 of a cell; event ``k`` is seeded ``seed_base + k``. Every profile, revision and
    kind
    of build uses the same seeds for one element and route, so their cells are comparable stop for stop."""
    return base + z * Z_STRIDE + ROUTES.index(route) * ROUTE_STRIDE


def roster(root: Path) -> dict[str, list[int]]:
    """The elements harvested under each profile, by rule: under ``parity`` every element of a reference row,
    lithium and boron, and aluminium and gold (single-isotope controls); under each evaluated profile every
    element with more than one natural isotope that the profile's rows answer some stops of, and one element
    it answers none of (``suzuki1987``: lead; ``mizuno2025``: magnesium, whose natural-composition row no
    transported
    nucleus requests)."""
    isotopes = natural.load_isotopes(root / natural.ISOTOPES_RELPATH)
    refs = references(root)
    parity = sorted({r.z for r in refs} | {3, 5, 13, 79})
    out = {spec.PARITY_PROFILE: parity}
    answers_rows = read_csv(root / ANSWERS_RELPATH, ANSWER_COLUMNS)
    controls = {SUZUKI: 82, MIZUNO: 12}
    for name in natural.capture_profiles(root / "data" / "g4" / "d1"):
        if name == spec.PARITY_PROFILE:
            continue
        touched = sorted(
            {
                int(r["z"])
                for r in answers_rows
                if r["profile"] == name
                and r["route"] == natural.PROFILE_ROUTE
                and len(isotopes[int(r["z"])]) > 1
            }
        )
        if controls[name] in touched:
            raise MixtureError(f"{name}: the control element is touched")
        out[name] = sorted(touched + [controls[name]])
    return out


def cells(root: Path, policy: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    """Every production cell: profile, route, revision, element, events, seed."""
    out: list[dict[str, object]] = []
    big = (
        {}
        if policy is None
        else {
            (d["profile"], d["z"], d["route"]): d["events"]
            for d in policy["drill_cells"]  # type: ignore[attr-defined]
        }
    )
    isotopes = natural.load_isotopes(root / natural.ISOTOPES_RELPATH)
    for profile, zs in roster(root).items():
        for z in zs:
            for route in ROUTES:
                for revision in REVISIONS:
                    out.append(
                        {
                            "profile": profile,
                            "route": route,
                            "revision": revision,
                            "z": z,
                            "symbol": isotopes[z][0].symbol,
                            "events": big.get((profile, z, route), 100_000),
                            "seed_base": seed_base(z, route),
                        }
                    )
    return out


#: The bins of the harvest's distribution tests.
TIME_BINS, DIO_BINS, WINDOW_BINS = 20, 20, 20
WINDOW = (2.0, 12.0)
FAMILY_ALPHA = Decimal("0.001")
POWER = 0.99


def time_edges() -> list[Decimal]:
    """The edges of the equal-probability bins of Exp(1): ``u_j = -ln(1 - j/B)`` for ``j = 1..B-1``."""
    with localcontext() as context:
        context.prec = 50
        return [-(1 - Decimal(j) / TIME_BINS).ln() for j in range(1, TIME_BINS)]


def dio_edges(components: Sequence[Component]) -> list[float]:
    """The binary64 edges of the DIO-time bins of a cell: ``0`` and ``-ln(1 - j/B) / lambda_bar`` for
    ``j = 1..B-1``, the last bin open."""
    lam = _dec(mean_rate(components))
    with localcontext() as context:
        context.prec = 50
        return [0.0] + [float(-(1 - Decimal(j) / DIO_BINS).ln() / lam) for j in range(1, DIO_BINS)]


def window_edges(components: Sequence[Component]) -> list[float]:
    """The binary64 edges of the window bins: equal probability for Exp(lambda_bar) truncated to
    ``[2 tau_bar, 12 tau_bar]``, the window's ends included."""
    lam = _dec(mean_rate(components))
    with localcontext() as context:
        context.prec = 50
        lo, hi = Decimal(WINDOW[0]), Decimal(WINDOW[1])
        span = 1 - (lo - hi).exp()
        return [
            float((lo - (1 - span * Decimal(j) / WINDOW_BINS).ln()) / lam) for j in range(WINDOW_BINS + 1)
        ]


def window_probabilities(components: Sequence[Component], edges: Sequence[float]) -> list[Decimal]:
    """The DIO-time distribution conditional on the window, in the window's bins."""
    full = dio_probabilities(components, list(edges))
    inside = full[:-2]  # the last edge's open interval and the capture category are outside
    with localcontext() as context:
        context.prec = 50
        total = sum(inside, Decimal(0))
        return [+(p / total) for p in inside]


def _float_probs_time_alt(components: Sequence[Component]) -> list[float]:
    """G-time bin probabilities when every delay is drawn from Exp(lambda_bar) instead of the drawn isotope's
    rate, the statistic still scaling each delay by the drawn isotope's rate."""
    lam = float(mean_rate(components))
    edges = [0.0] + [float(u) for u in time_edges()] + [math.inf]
    out = []
    for j in range(TIME_BINS):
        p = 0.0
        for c in components:
            r = float(c.rate)
            lo = math.exp(-edges[j] * lam / r)
            hi = 0.0 if edges[j + 1] == math.inf else math.exp(-edges[j + 1] * lam / r)
            p += float(c.abundance) * (lo - hi)
        out.append(p)
    return out


def _single_exponential(components: Sequence[Component]) -> list[Component]:
    """The mixture replaced by one component at its mean rate with its DIO fraction: the drill oracle."""
    lam = mean_rate(components)
    dio = 1 - capture_fraction(components)
    c = components[0]
    return [
        Component(
            0,
            Fraction(1),
            c.capture,
            c.decay,
            c.total,
            c.tau,
            c.tau_dio,
            c.tau_nc,
            lam,
            dio,
            "",
            "",
            "",
            None,
        )
    ]


def _without_fallback(components: Sequence[Component]) -> list[Component]:
    """The mixture without its formula-answered components, renormalized: the omitted-fallback drill
    oracle."""
    kept = [c for c in components if c.answer_class != FORMULA]
    total = sum((c.abundance for c in kept), Fraction(0))
    return [
        Component(
            c.a,
            c.abundance / total,
            c.capture,
            c.decay,
            c.total,
            c.tau,
            c.tau_dio,
            c.tau_nc,
            c.rate,
            c.p_dio,
            c.lookup,
            c.answer_class,
            c.record,
            c.record_unc,
        )
        for c in kept
    ]


def _nc(observed: Sequence[float], expected: Sequence[float], n: float) -> float:
    return n * sum((o - e) ** 2 / e for o, e in zip(observed, expected, strict=True) if e > 0)


def drill_noncentralities(components: Sequence[Component], n: float) -> dict[str, float]:
    """The noncentrality each statistical drill gives its gate on a cell of ``n`` stops."""
    out: dict[str, float] = {}
    out["G-time:exp_lambda_bar"] = _nc(_float_probs_time_alt(components), [1.0 / TIME_BINS] * TIME_BINS, n)
    edges = dio_edges(components)
    truth = [float(p) for p in dio_probabilities(components, edges, prec=30)]
    single = [float(p) for p in dio_probabilities(_single_exponential(components), edges, prec=30)]
    out["G-DIO:single_exponential"] = _nc(truth, single, n)
    if any(c.answer_class == FORMULA for c in components) and any(
        c.answer_class != FORMULA for c in components
    ):
        omitted = [float(p) for p in dio_probabilities(_without_fallback(components), edges, prec=30)]
        out["G-DIO:omitted_fallback"] = _nc(truth, omitted, n)
        out["G-sel:formula_stops_removed"] = n * float(
            sum(c.abundance for c in components if c.answer_class == FORMULA)
        )
    wedges = window_edges(components)
    wtruth = [float(p) for p in window_probabilities(components, wedges)]
    wsingle = [float(p) for p in window_probabilities(_single_exponential(components), wedges)]
    inside = float(sum(dio_probabilities(components, wedges, prec=30)[:-2]))
    out["G-win:single_exponential"] = _nc(wtruth, wsingle, n * inside)
    return out


DRILL_DOF = {
    "G-time:exp_lambda_bar": TIME_BINS - 1,
    "G-DIO:single_exponential": DIO_BINS,
    "G-DIO:omitted_fallback": DIO_BINS,
    "G-win:single_exponential": WINDOW_BINS - 1,
}
#: The fraction of one isotope's captures the branch drill turns into decays, and the share of stops the time
#: drill draws its resampled delays with (the seed of its generator).
BRANCH_SWAP_FRACTION = Fraction(1, 20)
DRILL_SEED = 20261017


#: Every drill of the harvest gates: the gate it targets, the cell it runs on (the statistical ones are the
#: cells the power calculation names) and what it plants. Each runs through the whole chain of gates,
#: recording the first that turns red, and against its target gate alone, which must turn red.
DRILL_PLAN = [
    {
        "drill": "rate_substitution",
        "gate": "G-rate",
        "z": 26,
        "side": "oracle",
        "plants": "the natural iron record's capture rate given to iron-54",
    },
    {
        "drill": "fallback_dropped",
        "gate": "G-set",
        "z": 50,
        "side": "oracle",
        "plants": "the formula-answered components dropped and the rest renormalized",
    },
    {
        "drill": "formula_stops_removed",
        "gate": "G-sel",
        "z": 50,
        "side": "harvest copy",
        "plants": "the formula-answered stops removed",
    },
    {
        "drill": "branch_swap",
        "gate": "G-branch",
        "z": 29,
        "side": "harvest copy",
        "plants": "a twentieth of the most abundant isotope's captures turned into decays",
    },
    {
        "drill": "exp_lambda_bar",
        "gate": "G-time",
        "z": 29,
        "side": "harvest copy",
        "plants": "every delay drawn again from Exp(lambda_bar), the statistic still scaling by the drawn "
        "isotope's rate",
    },
    {
        "drill": "single_exponential",
        "gate": "G-DIO",
        "z": 17,
        "side": "oracle",
        "plants": "the mixture replaced by Exp(lambda_bar) with the same decay fraction",
    },
    {
        "drill": "omitted_fallback",
        "gate": "G-DIO",
        "z": 12,
        "side": "oracle",
        "plants": "the oracle without its formula-answered components",
    },
    {
        "drill": "single_exponential_window",
        "gate": "G-win",
        "z": 17,
        "side": "oracle",
        "plants": "the mixture replaced by Exp(lambda_bar) with the same decay fraction, in the window",
    },
    {
        "drill": "atomic_expectation",
        "gate": "G-branch",
        "z": 29,
        "side": "oracle",
        "plants": "the atomic branch expectation replaced by lambda_d / (lambda_c + lambda_d): information "
        "only, the largest binary64 difference is reported",
    },
]


def tests_of(n_isotopes: int) -> tuple[str, ...]:
    """The distribution tests of a cell of an element with ``n_isotopes`` natural isotopes."""
    return (("G-sel",) if n_isotopes > 1 else ()) + ("G-branch", "G-time", "G-DIO", "G-win")


def build_policy(root: Path, throughput_ms_per_stop: float) -> tuple[dict[str, object], list[str]]:
    """The comparison policy, and the lines of the power calculation it rests on."""
    oracle = Oracle(root)
    rost = roster(root)
    base_cells = cells(root)
    isotopes = oracle.isotopes
    n_tests = sum(len(tests_of(len(isotopes[int(str(c["z"]))]))) for c in base_cells)
    alpha = FAMILY_ALPHA / n_tests
    dofs = sorted(
        {TIME_BINS - 1, DIO_BINS, WINDOW_BINS - 1}
        | {len(isotopes[z]) - 1 for z in rost[spec.PARITY_PROFILE] if len(isotopes[z]) > 1}
        | {len(isotopes[z]) for zs in rost.values() for z in zs}
    )
    crits = {dof: critical(dof, alpha) for dof in dofs}
    log: list[str] = [
        f"tests M={n_tests} over {len(base_cells)} production cells; family alpha {FAMILY_ALPHA}; "
        f"per-test alpha {alpha:.6e}"
    ]
    for dof, crit in crits.items():
        log.append(f"critical dof={dof} [{crit.lo:.12f}, {crit.hi:.12f}]")
    required = {dof: required_noncentrality(dof, float(crit.hi), POWER) for dof, crit in crits.items()}
    for dof, value in required.items():
        log.append(f"required noncentrality for power {POWER} at dof={dof}: {value:.3f}")
    # every parity cell: the noncentrality of each statistical drill at 1e5 stops, the omitted share G-sel
    # detects
    # and the single-exponential distinguishability
    drills: dict[tuple[str, str], list[tuple[float, int]]] = {}
    per_cell: list[dict[str, object]] = []
    for z in rost[spec.PARITY_PROFILE]:
        for route in ROUTES:
            comps = oracle.components(spec.PARITY_PROFILE, route, z)
            ncs = drill_noncentralities(comps, 1.0e5)
            k = len(comps)
            sel_dof = k - 1
            min_share = (required[sel_dof] / (1.0e5 + required[sel_dof])) if k > 1 else None
            single = ncs["G-DIO:single_exponential"]
            per_cell.append(
                {
                    "z": z,
                    "route": route,
                    "n": 100_000,
                    "min_detectable_omitted_share": (
                        None if min_share is None else float(f"{min_share:.6g}")
                    ),
                    "single_exponential_noncentrality": float(f"{single:.6g}"),
                    "single_exponential_power": float(
                        f"{noncentral_sf(float(crits[DIO_BINS].hi), DIO_BINS, single):.6g}"
                    ),
                }
            )
            for name, value in ncs.items():
                drills.setdefault((name, route), []).append((value, z))
            log.append(
                f"cell parity z={z} {route}: "
                + " ".join(f"{k_}={v:.3f}" for k_, v in ncs.items())
                + (f" min_omitted_share={min_share:.3e}" if min_share is not None else "")
            )
    drill_cells: dict[tuple[str, int, str], int] = {}
    chosen: list[dict[str, object]] = []
    for (name, route), values in sorted(drills.items()):
        best_value, best_z = max(values)
        # removing the formula-answered stops: the statistic's dof is that cell's isotope count less one
        dof = len(isotopes[best_z]) - 1 if name.startswith("G-sel") else DRILL_DOF[name]
        crit_hi = float(crits[dof].hi)
        events = 100_000
        power = noncentral_sf(crit_hi, dof, best_value)
        if power < POWER:
            events = 1_000_000
            power = noncentral_sf(crit_hi, dof, best_value * 10)
        if power < POWER:
            raise MixtureError(f"{name} {route}: no cell reaches power {POWER} at 1e6 stops")
        if events > 100_000:
            drill_cells[(spec.PARITY_PROFILE, best_z, route)] = events
        chosen.append(
            {
                "drill": name,
                "route": route,
                "z": best_z,
                "events": events,
                "dof": dof,
                "noncentrality_at_1e5": float(f"{best_value:.6g}"),
                "power": float(f"{power:.6g}"),
            }
        )
        log.append(f"drill {name} {route}: z={best_z} events={events} power={power:.6f}")
    policy_cells = [
        {"profile": p, "z": z, "route": r, "events": e} for (p, z, r), e in sorted(drill_cells.items())
    ]
    total_events = sum(
        drill_cells.get((str(c["profile"]), int(str(c["z"])), str(c["route"])), 100_000) for c in base_cells
    )
    pristine_events = 4 * len(ROUTES) * len(REVISIONS) * 10_000
    identity_events = len(REVISIONS) * len(ROUTES) * (4 * 3 + 2) * 2_500
    core_hours = (total_events + pristine_events + 3 * identity_events) * throughput_ms_per_stop / 1000 / 3600
    log.append(
        f"compute: production stops {total_events}, pristine {pristine_events}, identity {identity_events} "
        f"(counted 3x for state hashing) at {throughput_ms_per_stop:.4f} ms per stop = {core_hours:.2f} "
        "core-hours"
    )
    refs = references(root)
    est_rows = load_estimators(root)
    labels = []
    for ref in refs:
        label, missing = estimator_label(row_facts(root, ref, est_rows))
        labels.append(
            {
                "source": ref.source,
                "locator": ref.locator,
                "label": ref.label,
                "z": ref.z,
                "estimator": label,
                "missing": missing,
            }
        )
    policy: dict[str, object] = {
        "contract": "data/g4/d1 natural-target mixtures: harvest gates, statistics, drills and the "
        "comparison of measured mean lives; every value fixed before any harvest output or comparison "
        "exists",
        "roster": {k: v for k, v in rost.items()},
        "routes": list(ROUTES),
        "revisions": list(REVISIONS),
        "events_default": 100_000,
        "drill_cells": policy_cells,
        "pristine_controls": {"elements": [3, 32, 47, 82], "events": 10_000},
        "identity": {"elements": [3, 32, 47, 82], "events": 2_500, "drill_element": 32, "drill_a": 74},
        "seeds": {
            "production_base": SEED_BASE,
            "z_stride": Z_STRIDE,
            "route_stride": ROUTE_STRIDE,
            "identity_base": IDENTITY_SEED_BASE,
            "pilot_base": PILOT_SEED_BASE,
        },
        "bins": {"time": TIME_BINS, "dio": DIO_BINS, "window": WINDOW_BINS},
        "window_over_tau_bar": list(WINDOW),
        "branch_min_expected": 5,
        "statistics": {
            "tests": n_tests,
            "family_alpha": str(FAMILY_ALPHA),
            "per_test_alpha": str(alpha),
            "critical": {str(d): [str(c.lo), str(c.hi)] for d, c in crits.items()},
            "rule": "pass if the X^2 enclosure lies below the critical enclosure, fail if above, "
            "refuse if they overlap",
        },
        "power": {"target": POWER, "drills": chosen, "cells": per_cell},
        "drills": {"branch_swap_fraction": str(BRANCH_SWAP_FRACTION), "seed": DRILL_SEED, "plan": DRILL_PLAN},
        "estimators": {
            "labels": labels,
            "rule": "ESTABLISHED only when the component set, method, weighting, fit range, non-target "
            "amplitudes and beam structure are all printed; otherwise SURROGATE",
            "surrogate": {
                "suzuki1987": {
                    "method": ["ll", "ls"],
                    "constant": [False, True],
                    "t1_over_tau_bar": [0, 2],
                    "t2_over_tau_bar": [6, 12],
                    "carbon": ["free", "fixed"],
                    "carbon_share": [0.01, 0.02],
                    "carbon_target": "a carbon row carries no carbon component: its background is muonic "
                    "carbon",
                    "partners": {str(k): v for k, v in SUZUKI_PARTNERS.items()},
                    "partner_share": list(DEFAULT_SHARES),
                    "partner_lifetime_rows": PARTNER_LOCATORS,
                    "beam": "one stop per clock start (printed)",
                },
                "mizuno2025": {
                    "method": ["ll"],
                    "constant": [False],
                    "t1_over_tau_bar": [0, 2],
                    "t2_over_tau_bar": [6, 12],
                    "cno_share": "printed approximate share x {0.5, 1, 2}",
                    "ag_share": list(DEFAULT_SHARES),
                    "bg_share": list(DEFAULT_SHARES),
                    "origin_ns": MIZUNO_ORIGIN_NS,
                    "beam": f"one prompt pulse, or two equal prompt pulses {PULSE_SEPARATION_NS:g} ns apart "
                    ""
                    "(policy value: the separation is not printed)",
                    "natAg": "the CNO and background lifetimes fixed at the printed values of the free fit",
                },
            },
            "lambda_fit": "the midpoint of the bracket's range of fitted rates",
            "sigma_win": "half the bracket's range (0 for an ESTABLISHED estimator)",
            "sigma_tab": "linear propagation of each answering record's tabulated uncertainty, fully "
            "correlated over the isotopes one record answers and independent across records, at the "
            "nominal fit",
        },
        "band": {
            "k": 3,
            "rule": "WITHIN iff |lambda_fit - lambda_meas| <= 3 sqrt(sigma_meas^2 + sigma_tab^2 + "
            "sigma_win^2), decided on squares in exact decimals",
        },
        "qualification": "printed nat, formula share 0, estimator ESTABLISHED, and no answering record in "
        + ", ".join(DISQUALIFYING),
        "science": "per profile and route: family all SCIENCE_FAIL if any distinct row is OUTSIDE, else "
        "UNQUALIFIED; family qualified SCIENCE_PASS or SCIENCE_FAIL, or SCREEN_VACUOUS when it admits no "
        "row",
        "compute": {
            "ms_per_stop": throughput_ms_per_stop,
            "core_hours": float(f"{core_hours:.4g}"),
            "stop_threshold_core_hours": 150,
        },
    }
    return policy, log


def render_policy(policy: Mapping[str, object]) -> bytes:
    return (json.dumps(policy, indent=1, sort_keys=True) + "\n").encode("ascii")


def load_policy(root: Path) -> dict[str, object]:
    raw = (root / POLICY_RELPATH).read_bytes()
    if b"\r" in raw:
        raise MixtureError("the policy contains CR")
    policy: dict[str, object] = json.loads(raw.decode("ascii"))
    return policy


# --------------------------------------------------------------------------------------------
# the harvest: raw cells to the committed summary
# --------------------------------------------------------------------------------------------

HARVEST_RELPATH = "data/g4/d1/natural_mixture_harvest.csv"
BUILDS_RELPATH = "data/g4/d1/natural_mixture_builds.csv"
HARNESS_RELPATH = "cpp/tools/harvest_mixture.cc"
HARVEST_COLUMNS = ("cell", "stat", "key", "value")
BUILD_COLUMNS = (
    "build",
    "revision",
    "kind",
    "drill",
    "geant4_version",
    "gcc",
    "compile_flags",
    "source_sha256",
    "binary_sha256",
    "libraries_sha256",
)
#: The lines of a cell whose stream decides the pristine-against-patched identity of the controls.
CONTROL_LINES = 10_000


def cell_name(cell: Mapping[str, object]) -> str:
    return f"{cell['profile']}:{cell['route']}:{cell['revision']}:{cell['z']}"


def _bin_of_time(u_float: float, exact: Callable[[], Fraction]) -> int | None:
    """The G-time bin of ``u`` (equal-probability bins of Exp(1)): decided from the binary64 product
    ``u_float`` unless it lies within 1e-9 of an edge, and then from the exact ``u`` in 60-digit decimals, or
    ``None`` when
    ``u`` lies within 1e-40 of an edge (the statistic then refuses rather than guess)."""
    q = 1.0 - math.exp(-u_float)
    j = int(q * TIME_BINS)
    if abs(q * TIME_BINS - round(q * TIME_BINS)) > 1e-9:
        return min(j, TIME_BINS - 1)
    u = exact()
    with localcontext() as context:
        context.prec = 60
        value = 1 - (-(Decimal(u.numerator) / Decimal(u.denominator))).exp()
        scaled = value * TIME_BINS
        nearest = scaled.to_integral_value()
        if abs(scaled - nearest) < Decimal("1e-40"):
            return None
        return min(int(scaled), TIME_BINS - 1)


def _exact_product(a: float, b: float) -> Fraction:
    return Fraction(a) * Fraction(b)


def _exact_quotient(a: float, b: float) -> Fraction:
    return Fraction(a) / Fraction(b)


def _bin_of(t: float, edges: Sequence[float]) -> int:
    """The index of the interval of ``edges`` holding ``t`` (the last open), by binary64 comparison."""
    j = 0
    while j + 1 < len(edges) and t >= edges[j + 1]:
        j += 1
    return j


def rate_digest(values: Iterable[str]) -> str:
    """``<number of distinct values>:<SHA-256>`` of the distinct values a call site returned for one isotope
    and field, each read as a binary64 and written canonically (``float.hex``), joined by ``|`` in sorted
    order: the
    summary keeps the identity of the values, not the values."""
    canonical = sorted({float.fromhex(v).hex() for v in values})
    return f"{len(canonical)}:{hashlib.sha256('|'.join(canonical).encode('ascii')).hexdigest()}"


def summarize_cell(
    raw_stops: bytes, route: str, components: Sequence[Component]
) -> dict[str, dict[str, str]]:
    """The statistics of one cell from its stops file: per-isotope counts, branch counts and distinct rates
    from the call sites, the G-time bins of every stop, the G-DIO and window bins of the transported decay
    electrons,
    the consistency of the call sites with each other and with the transported output, and digests."""
    counts: dict[int, int] = {}
    branches: dict[str, int] = {}
    rates: dict[str, set[str]] = {}
    time_bins = [0] * TIME_BINS
    ties = 0
    dedges = dio_edges(components)
    wedges = window_edges(components)
    dio_bins = [0] * (len(dedges) + 1)
    win_bins = [0] * (len(wedges) - 1)
    outside = 0
    delays: list[float] = []
    electron_times: list[float] = []
    mismatch = {"selector_vs_decay": 0, "branch_vs_outcome": 0, "delay_vs_electron": 0}
    events = 0
    text = raw_stops.decode("ascii")
    for line in text.splitlines():
        f = line.split()
        events += 1
        sel_z, sel_a, z, a = int(f[1]), int(f[2]), int(f[3]), int(f[4])
        if (sel_z, sel_a) != (z, a):
            mismatch["selector_vs_decay"] += 1
        counts[a] = counts.get(a, 0) + 1
        if route == STOPPING:
            capture, decay, before, after = (float.fromhex(x) for x in f[5:9])
            branch = "D" if f[9] == "0" else "N"
            for name, value in (("capture", f[5]), ("decay", f[6])):
                rates.setdefault(f"{a}:{name}", set()).add(value)
            total = capture + decay
            delay = after - before
            u_float = total * delay
            exact = functools.partial(_exact_product, total, delay)

            outcome, electron = f[10], f[11]
            if electron != "-" and float.fromhex(electron) != after:
                mismatch["delay_vs_electron"] += 1
        else:
            tau = float.fromhex(f[5])
            for name, value in (("tau", f[5]), ("tau_dio", f[6]), ("tau_nc", f[7])):
                rates.setdefault(f"{a}:{name}", set()).add(value)
            delay = float.fromhex(f[9])
            atom_time = float.fromhex(f[10])
            branch = f[11]
            u_float = delay / tau
            exact = functools.partial(_exact_quotient, delay, tau)

            outcome, electron = f[12], f[13]
            if electron != "-" and float.fromhex(electron) != atom_time + delay:
                mismatch["delay_vs_electron"] += 1
        key = f"{a}:{branch}"
        branches[key] = branches.get(key, 0) + 1
        if branch != outcome:
            mismatch["branch_vs_outcome"] += 1
        delays.append(delay)
        j = _bin_of_time(u_float, exact)
        if j is None:
            ties += 1
        else:
            time_bins[j] += 1
        if outcome == "D" and electron != "-":
            t = float.fromhex(electron)
            electron_times.append(t)
            dio_bins[_bin_of(t, dedges)] += 1
            if wedges[0] <= t < wedges[-1]:
                win_bins[_bin_of(t, wedges[:-1])] += 1
            else:
                outside += 1
        else:
            dio_bins[-1] += 1
            outside += 1
    first = "\n".join(text.splitlines()[:CONTROL_LINES]) + "\n"
    out: dict[str, dict[str, str]] = {
        "meta": {
            "events": str(events),
            "stops_sha256": hashlib.sha256(raw_stops).hexdigest(),
            "first_lines_sha256": hashlib.sha256(first.encode("ascii")).hexdigest(),
            "time_ties": str(ties),
            "mean_delay_ns": repr(math.fsum(delays) / len(delays)),
            "mean_electron_time_ns": repr(math.fsum(electron_times) / len(electron_times)),
            **{f"mismatch_{k}": str(v) for k, v in mismatch.items()},
        },
        "count": {str(a): str(n) for a, n in sorted(counts.items())},
        "branch": {
            k: str(v) for k, v in sorted(branches.items(), key=lambda kv: (int(kv[0].split(":")[0]), kv[0]))
        },
        "rate": {
            k: rate_digest(v)
            for k, v in sorted(rates.items(), key=lambda kv: (int(kv[0].split(":")[0]), kv[0]))
        },
        "time": {str(j): str(n) for j, n in enumerate(time_bins)},
        "dio": {**{str(j): str(n) for j, n in enumerate(dio_bins[:-1])}, "capture": str(dio_bins[-1])},
        "dio_edge": {str(j): e.hex() for j, e in enumerate(dedges)},
        "win": {**{str(j): str(n) for j, n in enumerate(win_bins)}, "outside": str(outside)},
        "win_edge": {str(j): e.hex() for j, e in enumerate(wedges)},
    }
    return out


def summarize_cells(root: Path, plan: Mapping[str, Any], out_dir: Path) -> None:
    """The committed harvest and build files from the raw cells ``plan`` lists (``cells``: each with its
    ``dir``;
    ``builds``: each build directory; ``identity``: the W1 comparison)."""
    import gzip

    oracle = Oracle(root)
    policy = load_policy(root)
    rows: list[dict[str, str]] = []
    cells_ = plan["cells"]
    for cell in cells_:
        name = cell_name(cell)
        directory = Path(cell["dir"])
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        build = json.loads((Path(cell["build_dir"]) / "build.json").read_text(encoding="utf-8"))
        raw = gzip.decompress((directory / "raw" / "stops.txt.gz").read_bytes())
        if hashlib.sha256(raw).hexdigest() != run["files"]["stops.txt"]:
            raise MixtureError(f"{name}: the stops file is not the one the run recorded")
        profile = str(cell["profile"]) if cell["profile"] != "pristine" else spec.PARITY_PROFILE
        comps = oracle.components(profile, str(cell["route"]), int(cell["z"]))
        stats = summarize_cell(raw, str(cell["route"]), comps)
        stats["meta"].update(
            {
                "build": Path(cell["build_dir"]).name,
                "seed_base": str(cell["seed_base"]),
                "wall_s": f"{run['wall_s']:.3f}",
                "compile_flags": build["compile_flags"],
                "binary_sha256": build["binary_sha256"],
            }
        )
        verdicts = evaluate_cell(stats, str(cell["route"]), comps, policy)
        stats["verdict"] = {gate: v["verdict"] for gate, v in verdicts.items()}
        stats["x2"] = {gate: v["x2"] for gate, v in verdicts.items() if "x2" in v}
        stats["dof"] = {gate: v["dof"] for gate, v in verdicts.items() if "dof" in v}
        print(
            f"CELL {name} events={stats['meta']['events']} "
            + " ".join(f"{g}={v}" for g, v in stats["verdict"].items()),
            flush=True,
        )
        for stat, values in stats.items():
            for key, value in values.items():
                rows.append({"cell": name, "stat": stat, "key": key, "value": value})
    identity = plan.get("identity")
    if identity:
        for entry in identity["cells"]:
            name = f"identity:{entry['route']}:{entry['revision']}:{entry['symbol']}"
            for key in sorted(entry):
                if key not in ("route", "revision", "symbol"):
                    rows.append({"cell": name, "stat": "identity", "key": key, "value": str(entry[key])})
        for entry in identity["drills"]:
            name = f"identity_drill:{entry['route']}:{entry['revision']}"
            for key in sorted(entry):
                if key not in ("route", "revision"):
                    rows.append({"cell": name, "stat": "identity", "key": key, "value": str(entry[key])})
    source = hashlib.sha256((root / HARNESS_RELPATH).read_bytes()).hexdigest()
    builds: list[dict[str, str]] = []
    for build_dir in plan["builds"]:
        info = json.loads((Path(build_dir) / "build.json").read_text(encoding="utf-8"))
        if info["source_sha256"] != source:
            raise MixtureError(f"{build_dir}: the build compiled another source than {HARNESS_RELPATH}")
        libs = hashlib.sha256(
            json.dumps([(x["soname"], x["sha256"]) for x in info["libraries"]]).encode()
        ).hexdigest()
        name = Path(build_dir).name
        builds.append(
            {
                "build": name,
                "revision": name.split("-")[0],
                "kind": info["kind"],
                "drill": info["drill"],
                "geant4_version": info["geant4_version"],
                "gcc": info["gcc"],
                "compile_flags": info["compile_flags"],
                "source_sha256": info["source_sha256"],
                "binary_sha256": info["binary_sha256"],
                "libraries_sha256": libs,
            }
        )
    (out_dir / Path(HARVEST_RELPATH).name).write_bytes(render_csv(rows, HARVEST_COLUMNS))
    (out_dir / Path(BUILDS_RELPATH).name).write_bytes(render_csv(builds, BUILD_COLUMNS))
    print(f"SUMMARIZED cells={len(cells_)} rows={len(rows)} builds={len(builds)}")


def load_harvest(root: Path) -> dict[str, dict[str, dict[str, str]]]:
    """The committed harvest by cell, then statistic, then key."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for row in read_csv(root / HARVEST_RELPATH, HARVEST_COLUMNS):
        out.setdefault(row["cell"], {}).setdefault(row["stat"], {})[row["key"]] = row["value"]
    return out


# --------------------------------------------------------------------------------------------
# the gates, recomputed from the committed harvest, the policy and the oracle
# --------------------------------------------------------------------------------------------

GATES = ("G-rate", "G-set", "G-sel", "G-branch", "G-time", "G-DIO", "G-win")


@functools.cache
def _critical(dof: int, alpha: str) -> Critical:
    return critical(dof, Decimal(alpha))


def gate_rate(
    stats: Mapping[str, Mapping[str, str]], route: str, components: Sequence[Component]
) -> list[str]:
    """Every isotope and field whose harvested call-site values are not exactly the oracle's binary64 value:
    the
    summary's digest must be that of the one value the oracle computes."""
    failures = []
    by_a = {c.a: c for c in components}
    for key, value in stats["rate"].items():
        a, field = key.split(":")
        c = by_a.get(int(a))
        if c is None:
            failures.append(f"{key}: no oracle component")
            continue
        expected = {
            "capture": c.capture,
            "decay": c.decay,
            "tau": c.tau,
            "tau_dio": c.tau_dio,
            "tau_nc": c.tau_nc,
        }[field]
        if value != rate_digest([expected.hex()]):
            failures.append(f"{key}: harvested {value} oracle {rate_digest([expected.hex()])}")
    return failures


def gate_set(stats: Mapping[str, Mapping[str, str]], components: Sequence[Component]) -> list[str]:
    harvested = {int(a) for a in stats["count"]}
    model = {c.a for c in components}
    return [] if harvested == model else [f"harvested {sorted(harvested)} oracle {sorted(model)}"]


def _x2_verdict(counts: Sequence[int], probs: Sequence[Fraction | Decimal], alpha: str) -> dict[str, str]:
    dof = len(counts) - 1
    lo, hi = pearson(counts, probs)
    crit = _critical(dof, alpha)
    return {
        "x2": f"{(lo + hi) / 2:.6f}",
        "dof": str(dof),
        "critical": f"{crit.lo:.6f}",
        "verdict": verdict((lo, hi), crit),
    }


def gate_sel(
    stats: Mapping[str, Mapping[str, str]], components: Sequence[Component], alpha: str
) -> dict[str, str]:
    counts = [int(stats["count"].get(str(c.a), "0")) for c in components]
    return _x2_verdict(counts, [c.abundance for c in components], alpha)


def gate_branch(
    stats: Mapping[str, Mapping[str, str]],
    components: Sequence[Component],
    alpha: str,
    min_expected: int = 5,
    branch_expectation=None,
) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Pearson's X^2 over the isotopes, each a binomial of its own count with the exact decay probability; an
    isotope whose smaller expected count is below ``min_expected`` is left out and counted."""
    x2_lo = x2_hi = Decimal(0)
    used = left = 0
    for c in components:
        dio = int(stats["branch"].get(f"{c.a}:D", "0"))
        cap = int(stats["branch"].get(f"{c.a}:N", "0"))
        n = dio + cap
        p = c.p_dio if branch_expectation is None else branch_expectation(c)
        if n == 0 or min(n * p, n * (1 - p)) < min_expected:
            left += 1
            continue
        lo, hi = pearson([dio, cap], [p, 1 - p])
        x2_lo += lo
        x2_hi += hi
        used += 1
    crit = _critical(used, alpha)
    return {
        "x2": f"{(x2_lo + x2_hi) / 2:.6f}",
        "dof": str(used),
        "left_out": str(left),
        "critical": f"{crit.lo:.6f}",
        "verdict": verdict((x2_lo, x2_hi), crit),
    }


def gate_time(stats: Mapping[str, Mapping[str, str]], alpha: str) -> dict[str, str]:
    counts = [int(stats["time"][str(j)]) for j in range(TIME_BINS)]
    result = _x2_verdict(counts, [Fraction(1, TIME_BINS)] * TIME_BINS, alpha)
    if int(stats["meta"]["time_ties"]):
        result["verdict"] = "refuse"
    return result


def gate_dio(
    stats: Mapping[str, Mapping[str, str]], components: Sequence[Component], alpha: str
) -> dict[str, str]:
    edges = [float.fromhex(stats["dio_edge"][str(j)]) for j in range(len(stats["dio_edge"]))]
    if edges != dio_edges(components):
        return {"verdict": "fail", "reason": "the summary's bins are not the oracle's"}
    probs = dio_probabilities(components, edges)
    counts = [int(stats["dio"][str(j)]) for j in range(len(edges))] + [int(stats["dio"]["capture"])]
    return _x2_verdict(counts, probs, alpha)


def gate_win(
    stats: Mapping[str, Mapping[str, str]], components: Sequence[Component], alpha: str
) -> dict[str, str]:
    edges = [float.fromhex(stats["win_edge"][str(j)]) for j in range(len(stats["win_edge"]))]
    if edges != window_edges(components):
        return {"verdict": "fail", "reason": "the summary's bins are not the oracle's"}
    probs = window_probabilities(components, edges)
    counts = [int(stats["win"][str(j)]) for j in range(len(edges) - 1)]
    return _x2_verdict(counts, probs, alpha)


def evaluate_cell(
    stats: Mapping[str, Mapping[str, str]],
    route: str,
    components: Sequence[Component],
    policy: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Every gate of one cell, in chain order."""
    alpha = str(policy["statistics"]["per_test_alpha"])
    rate = gate_rate(stats, route, components)
    out: dict[str, dict[str, str]] = {
        "G-rate": {"verdict": "pass" if not rate else "fail", "detail": "; ".join(rate)}
    }
    setf = gate_set(stats, components)
    out["G-set"] = {"verdict": "pass" if not setf else "fail", "detail": "; ".join(setf)}
    if len(components) > 1:
        out["G-sel"] = gate_sel(stats, components, alpha)
    out["G-branch"] = gate_branch(stats, components, alpha, int(policy["branch_min_expected"]))
    out["G-time"] = gate_time(stats, alpha)
    out["G-DIO"] = gate_dio(stats, components, alpha)
    out["G-win"] = gate_win(stats, components, alpha)
    return out


def first_red(verdicts: Mapping[str, Mapping[str, str]]) -> str | None:
    for gate in GATES:
        if gate in verdicts and verdicts[gate]["verdict"] != "pass":
            return gate
    return None


# --------------------------------------------------------------------------------------------
# the comparison with the measured mean lives
# --------------------------------------------------------------------------------------------

COMPARISON_RELPATH = "data/g4/d1/natural_mixture_comparison.csv"
COMPARISON_COLUMNS = (
    "source",
    "locator",
    "label",
    "z",
    "profile",
    "route",
    "estimator",
    "fit_rate",
    "fit_rate_min",
    "fit_rate_max",
    "sigma_tab",
    "sigma_win",
    "mean_rate",
    "alias_rate",
    "difference",
    "difference_sigma",
    "part_estimator",
    "part_composition",
    "part_single_nucleus",
    "part_merged",
    "formula_share",
    "same_source_share",
    "classes",
    "qualified",
    "verdict",
    "comparison",
)
#: Rates in the comparison are in 1e6/s; the oracle's are ns^-1.
PER_MICROSECOND = d1.MICROSECOND


def _perturbed(components: Sequence[Component], record: str, route: str, h: float) -> list[Component]:
    """The components with the capture rate of every isotope ``record`` answers scaled by ``1 + h``, the
    route's
    rate and decay probability recomputed in exact arithmetic (a real-valued perturbation, for
    derivatives)."""
    out = []
    for c in components:
        if c.record != record:
            out.append(c)
            continue
        capture = Fraction(c.capture) * (1 + Fraction(h))
        total = capture + Fraction(c.decay)
        out.append(
            Component(
                c.a,
                c.abundance,
                c.capture,
                c.decay,
                c.total,
                c.tau,
                c.tau_dio,
                c.tau_nc,
                total,
                Fraction(c.decay) / total,
                c.lookup,
                c.answer_class,
                c.record,
                c.record_unc,
            )
        )
    return out


def _within(rate: str, measured: str, sigmas: Sequence[str], k: int = 3) -> bool:
    with localcontext() as context:
        context.prec = 400
        difference = Decimal(rate) - Decimal(measured)
        bound = sum((Decimal(s) ** 2 for s in sigmas), Decimal(0))
        return difference * difference <= k * k * bound


def comparison_rows(root: Path) -> list[dict[str, str]]:
    """One row per measured mean life of a natural target, profile and route: the rate the source's estimator
    (or its surrogate bracket) extracts from the transported mixture's expected spectrum, its uncertainties,
    the
    verdict and the parts of the difference."""
    from openmucf.g4 import d1_lifetime as lifetime

    oracle = Oracle(root)
    policy = load_policy(root)
    labels = {(r["source"], r["locator"]): r for r in policy["estimators"]["labels"]}  # type: ignore[index]
    measured_rows = {
        (r["source"], r["locator"], r["profile"]): r
        for r in lifetime.read_comparison(root / lifetime.COMPARISON_RELPATH)
    }
    cache: dict[tuple[object, ...], tuple[float, float, float, list[tuple[str, float, Decimal]]]] = {}
    rows: list[dict[str, str]] = []
    ids: dict[tuple[object, ...], int] = {}
    for ref in references(root):
        label = labels[(ref.source, ref.locator)]
        for profile in oracle.tables:
            for route in ROUTES:
                comps = oracle.components(profile, route, ref.z)
                key = (
                    ref.source,
                    ref.locator,
                    route,
                    tuple((c.a, c.abundance, c.rate, c.p_dio) for c in comps),
                )
                if key not in cache:
                    tau_bar = float(1 / mean_rate(comps))
                    fits, nominal = bracket(root, ref, tau_bar)
                    values = {name: fit_rate(comps, spec_) for name, spec_ in fits}
                    lo, hi = min(values.values()), max(values.values())
                    spec_nominal = dict(fits)[nominal]
                    contributions: list[tuple[str, float, Decimal]] = []
                    for record in sorted({c.record for c in comps if c.record}):
                        unc = next(c.record_unc for c in comps if c.record == record)
                        h = 1e-6
                        up = fit_rate(_perturbed(comps, record, route, h), spec_nominal)
                        down = fit_rate(_perturbed(comps, record, route, -h), spec_nominal)
                        capture = next(c.capture for c in comps if c.record == record)
                        derivative = (up - down) / (2 * h * capture)
                        contributions.append((record, derivative, unc))  # type: ignore[arg-type]
                    cache[key] = (lo, hi, values[nominal], contributions)
                lo, hi, _, contributions = cache[key]
                fit = (lo + hi) / 2 * PER_MICROSECOND
                sigma_win = (hi - lo) / 2 * PER_MICROSECOND
                sigma_tab = math.sqrt(math.fsum((d * float(u)) ** 2 for _, d, u in contributions))
                lam_bar = float(mean_rate(comps)) * PER_MICROSECOND
                measured = float(ref.measured_rate)
                sigma_meas = float(ref.measured_rate_unc)
                alias = ""
                parts = {
                    "part_estimator": fit - lam_bar,
                    "part_composition": math.nan,
                    "part_single_nucleus": math.nan,
                    "part_merged": math.nan,
                }
                source_row = measured_rows[(ref.source, ref.locator, profile)]
                if source_row["key_a"]:
                    alias_value = float(source_row["rate_bound" if route == STOPPING else "rate_helper"])
                    alias = repr(alias_value)
                    parts["part_composition"] = lam_bar - alias_value
                    parts["part_single_nucleus"] = alias_value - measured
                else:
                    parts["part_merged"] = lam_bar - measured
                sigma = math.sqrt(sigma_meas**2 + sigma_tab**2 + sigma_win**2)
                formula = sum((c.abundance for c in comps if c.answer_class == FORMULA), Fraction(0))
                # a record whose value the compared source's own measurements produced: the compiled-in table,
                # whose source comment credits Suzuki, Measday and Roalsvig from Z = 3 on, and each evaluated
                # profile's rows for its own source; a capture rate such a source derives from its own mean
                # life
                # is not an independent check of that mean life
                same_source = sum(
                    (
                        c.abundance
                        for c in comps
                        if (c.lookup == natural.TABLE_ROUTE and ref.source == SUZUKI and ref.z >= 3)
                        or (c.lookup == natural.PROFILE_ROUTE and profile == ref.source)
                    ),
                    Fraction(0),
                )
                shares: dict[str, Fraction] = {}
                for c in comps:
                    shares[c.answer_class] = shares.get(c.answer_class, Fraction(0)) + c.abundance
                touched = ";".join(f"{k}={float(v):.6g}" for k, v in sorted(shares.items()) if v)
                qualified = (
                    ref.basis == "natural"
                    and formula == 0
                    and label["estimator"] == ESTABLISHED
                    and not any(shares.get(k, 0) for k in DISQUALIFYING)
                )
                inside = _within(
                    repr(fit), ref.measured_rate, (ref.measured_rate_unc, repr(sigma_tab), repr(sigma_win))
                )
                identity = (
                    ref.source,
                    ref.locator,
                    route,
                    key[3],
                    repr(fit),
                    repr(sigma_win),
                    tuple((r, repr(d), str(u)) for r, d, u in contributions),
                    ref.measured_rate,
                    ref.measured_rate_unc,
                )
                comparison = ids.setdefault(identity, len(ids) + 1)
                rows.append(
                    {
                        "source": ref.source,
                        "locator": ref.locator,
                        "label": ref.label,
                        "z": str(ref.z),
                        "profile": profile,
                        "route": route,
                        "estimator": label["estimator"],
                        "fit_rate": repr(fit),
                        "fit_rate_min": repr(lo * PER_MICROSECOND),
                        "fit_rate_max": repr(hi * PER_MICROSECOND),
                        "sigma_tab": repr(sigma_tab),
                        "sigma_win": repr(sigma_win),
                        "mean_rate": repr(lam_bar),
                        "alias_rate": alias,
                        "difference": repr(fit - measured),
                        "difference_sigma": repr((fit - measured) / sigma),
                        **{k: ("" if math.isnan(v) else repr(v)) for k, v in parts.items()},
                        "formula_share": repr(float(formula)),
                        "same_source_share": repr(float(same_source)),
                        "classes": touched,
                        "qualified": "true" if qualified else "false",
                        "verdict": "WITHIN" if inside else "OUTSIDE",
                        "comparison": str(comparison),
                    }
                )
    return rows


def with_measured(root: Path, rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """The comparison rows with each measurement's rate and uncertainty joined from
    ``mean_life_comparison.csv``,
    which restates them; the comparison file itself carries only this project's values."""
    measured = {(r.source, r.locator): (r.measured_rate, r.measured_rate_unc) for r in references(root)}
    out = []
    for row in rows:
        rate, unc = measured[(row["source"], row["locator"])]
        out.append({**row, "measured_rate": rate, "measured_rate_unc": unc})
    return out


def science(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, object]]:
    """The verdict of each family per profile and route, over distinct comparisons."""
    out: dict[str, dict[str, object]] = {}
    for profile in dict.fromkeys(r["profile"] for r in rows):
        for route in ROUTES:
            mine = {r["comparison"]: r for r in rows if r["profile"] == profile and r["route"] == route}
            outside = sorted({r["comparison"] for r in mine.values() if r["verdict"] == "OUTSIDE"}, key=int)
            qualified = [r for r in mine.values() if r["qualified"] == "true"]
            if not qualified:
                family_q = "SCREEN_VACUOUS"
            else:
                family_q = (
                    "SCIENCE_FAIL" if any(r["verdict"] == "OUTSIDE" for r in qualified) else "SCIENCE_PASS"
                )
            out[f"{profile}:{route}"] = {
                "all": "SCIENCE_FAIL" if outside else "UNQUALIFIED",
                "qualified": family_q,
                "distinct": len(mine),
                "outside": len(outside),
                "qualified_rows": len(qualified),
            }
    return out


# --------------------------------------------------------------------------------------------
# the document: DATASET_D1.md section 13, generated
# --------------------------------------------------------------------------------------------

SECTION_HEADING = "## 13. Natural targets: what answers them, the call-site harvest and measured mean lives"
CLASS_TEXT = {
    EVALUATED_ISOTOPE: "a row of the selected evaluated profile keyed by the nuclide",
    PARITY_ISOTOPE: (
        "a compiled-in record of a separated isotope, of an element with a single natural isotope or of an "
        "isotope "
        "another primary tabulates, whose value equals a printed cell"
    ),
    LINEAGE_OPEN: (
        "a compiled-in record of a separated isotope whose value equals a printed cell only at that cell's "
        "printed "
        "precision, or equals no printed cell"
    ),
    MIXED: (
        "a compiled-in record whose value is a stated inverse-variance combination of printed rows measured "
        ""
        "on "
        "targets of different composition, recomputed and reproduced"
    ),
    PARITY_NATURAL: "a compiled-in record of the element printed with no mass number",
    UNSETTLED: "a compiled-in record whose flag is not established and whose evidence states no combination",
    FORMULA: "the Goulard-Primakoff formula",
}
GATE_TEXT = {
    "G-rate": "every harvested rate or lifetime against the model's binary64 value",
    "G-set": "the isotopes drawn against the model's components",
    "G-sel": "the isotope counts against the abundances",
    "G-branch": "each isotope's branch counts against its exact decay probability",
    "G-time": "every stop's delay scaled by its isotope's rate, in equal-probability bins of Exp(1)",
    "G-DIO": "the transported decay-electron times and the capture count against the mixture",
    "G-win": "the transported decay-electron times inside the window from two to twelve mean lives",
}


def _pct(value: float) -> str:
    text = format(100.0 * value, ".1f")
    return "0.0" if text == "-0.0" else text


def _sigma(value: str, sd: float) -> str:
    """``value / sd`` to a tenth, signed; a value that rounds to zero is ``0.0``."""
    text = format(float(value) / sd, "+.1f")
    return "0.0" if text in ("+0.0", "-0.0") else text


def _row_sd(row: Mapping[str, str]) -> float:
    return math.sqrt(sum(float(row[k]) ** 2 for k in ("measured_rate_unc", "sigma_tab", "sigma_win")))


def _keys(rows: Sequence[Mapping[str, str]], klass: str) -> str:
    keys = [
        f"`({r['z']}, {r['a']})`"
        for r in rows
        if r["profile"] == spec.PARITY_PROFILE and r["answer_class"] == klass
    ]
    return ", ".join(keys[:-1]) + (" and " if len(keys) > 1 else "") + keys[-1]


def _section_classes(root: Path, refs: Sequence[Reference]) -> list[str]:
    classes = read_csv(root / CLASSES_RELPATH, CLASS_COLUMNS)
    answers_rows = read_csv(root / ANSWERS_RELPATH, ANSWER_COLUMNS)
    isotopes = natural.load_isotopes(root / natural.ISOTOPES_RELPATH)
    shares = class_shares(answers_rows, isotopes)
    profiles = list(dict.fromkeys(r["profile"] for r in answers_rows))
    lines = [
        "### 13.1 What answers a natural target",
        "",
        "| class | answer | records, `parity` | rows, evaluated profiles |",
        "|---|---|---|---|",
    ]
    for klass in CLASSES:
        if klass == FORMULA:
            counts = ("-", "-")
        else:
            counts = (
                str(
                    sum(
                        1
                        for r in classes
                        if r["profile"] == spec.PARITY_PROFILE and r["answer_class"] == klass
                    )
                ),
                str(
                    sum(
                        1
                        for r in classes
                        if r["profile"] != spec.PARITY_PROFILE and r["answer_class"] == klass
                    )
                ),
            )
        lines.append(f"| `{klass}` | {CLASS_TEXT[klass]} | {counts[0]} | {counts[1]} |")
    n_natural_rows = sum(1 for r in classes if r["answer_class"] == EVALUATED_NATURAL)
    open_records = [
        r for r in classes if r["profile"] == spec.PARITY_PROFILE and r["answer_class"] == LINEAGE_OPEN
    ]
    lines += [
        "",
        f"The evaluated profiles' {n_natural_rows} rows keyed `A = 0` answer no transported nucleus, which "
        "always "
        "carries a mass number (section 11).",
        "The `isotope_resolved` flag of section 6 says what kind of target a record's value was measured "
        "on; "
        "it does "
        "not establish the value's lineage.",
        f"The {len(open_records)} records of a separated isotope whose value equals a published cell only "
        "at "
        "the "
        f"cell's printed precision, or equals none, are `{LINEAGE_OPEN}`: {_keys(classes, LINEAGE_OPEN)}.",
        f"The records {_keys(classes, MIXED)} are `{MIXED}`: the evidence of each states, and the generator "
        ""
        "recomputes and reproduces within the tolerance it states, an inverse-variance weighted mean of "
        "printed rows "
        "measured on targets of different composition (sections 6 and 11).",
        f"The records {_keys(classes, UNSETTLED)} are `{UNSETTLED}`.",
        "",
        "Under every profile but `parity`, a nuclide the profile keys no row for falls through to the "
        "compiled-in "
        "record of that nuclide where there is one, and to the formula where there is none: the parity path "
        ""
        "(`cpp/patches/README.md`).",
    ]
    for name in profiles:
        if name == spec.PARITY_PROFILE:
            continue
        own = sum(1 for (p, _), d in shares.items() if p == name and d[EVALUATED_ISOTOPE] > 0)
        aliased = sum(1 for (p, _), d in shares.items() if p == name and d[PARITY_NATURAL] > 0)
        lines.append(
            f"Under `{name}`, its own rows answer some stops of {own} of the {len(isotopes)} elements, and "
            "a "
            f"`{PARITY_NATURAL}` record answers some stops of {aliased}."
        )
    lines += [
        "The next table gives, for each element of a measured mean life of section 13.3, the share of stops "
        ""
        "each "
        "class answers under `parity`, and under an evaluated profile wherever it differs, in percent "
        "rounded "
        ""
        "to a "
        "tenth.",
        "",
        "| Z | element | profile | " + " | ".join(f"`{k}`" for k in CLASSES) + " |",
        "|---|---|---|" + "---|" * len(CLASSES),
    ]
    for z in sorted({r.z for r in refs}):
        base = shares[(spec.PARITY_PROFILE, z)]
        for name in profiles:
            d = shares[(name, z)]
            if name != spec.PARITY_PROFILE and d == base:
                continue
            cells_ = " | ".join(_pct(d[k]) for k in CLASSES)
            lines.append(f"| {z} | {isotopes[z][0].symbol} | `{name}` | {cells_} |")
    return lines


def _section_harvest(policy: Mapping[str, Any], harvest: Mapping[str, Any]) -> list[str]:
    cells_ = [k for k in harvest if not k.startswith("identity")]
    production = [k for k in cells_ if not k.startswith("pristine:")]
    controls = [k for k in cells_ if k.startswith("pristine:")]
    roster_ = policy["roster"]
    big = policy["drill_cells"]
    stops = sum(int(harvest[k]["meta"]["events"]) for k in production)
    wall = math.fsum(float(harvest[k]["meta"]["wall_s"]) for k in cells_)
    gates: dict[str, collections.Counter[str]] = {g: collections.Counter() for g in GATES}
    for k in cells_:
        for gate, v in harvest[k]["verdict"].items():
            gates[gate][v] += 1
    identity = [v["identity"] for k, v in harvest.items() if k.startswith("identity:")]
    identical = sum(
        1
        for e in identity
        if e["rng_identical"] == e["base_delegation"] == e["transport_identical"] == "True"
    )
    streams: dict[tuple[str, str, str], set[str]] = {}
    for k in production:
        profile, route, _, z = k.split(":")
        streams.setdefault((profile, route, z), set()).add(harvest[k]["meta"]["stops_sha256"])
    same = sum(1 for v in streams.values() if len(v) == 1)
    lines = [
        "### 13.2 The harvest at the call sites",
        "",
        "This harvest checks the sampler against the analytic model and nothing about physics.",
        "`cpp/tools/harvest_mixture.cc` transports negative muons that start at rest at t = 0 in a box of "
        "the "
        ""
        "NIST "
        "material of one element, one single-threaded process per cell, through the patched builds of "
        "`v11.4.2` and "
        "`v11.5.0.beta` on the stopping route and the atomic route.",
        f"It harvests {len(roster_['parity'])} elements under `parity`, {len(roster_['suzuki1987'])} under "
        f"`suzuki1987` and {len(roster_['mizuno2025'])} under `mizuno2025`, with the stops "
        f"`natural_mixture_policy.json` fixed before any harvest ran: {policy['events_default']} per cell, "
        "and "
        f"{big[0]['events']} on the {len(big)} cells its power calculation names; {stops} stops in "
        f"{len(production)} cells.",
        "Classes derived from the stock `G4ElementSelector`, `G4MuonMinusBoundDecay` and "
        "`G4MuonicAtomDecay` "
        "record "
        "the nucleus drawn, the rates or lifetimes used, the delay sampled and the branch taken at each "
        "call "
        "site, "
        "then call the stock method once; the transported decay electrons and capture outcomes come from "
        "the "
        "secondaries alone.",
        "`natural_mixture_harvest.csv` holds the summary of every cell and `natural_mixture_builds.csv` the "
        ""
        "builds; "
        "a test recomputes every gate exactly from them, the policy and the analytic model.",
        "In every cell the selector's nucleus is the decay's, the call site's branch is the transported "
        "outcome and "
        "the transported decay electron leaves at the delay the call site drew, on every stop.",
        "",
        "| gate | compares | cells | pass | fail | refuse |",
        "|---|---|---|---|---|---|",
    ]
    for gate in GATES:
        c = gates[gate]
        lines.append(
            f"| {gate} | {GATE_TEXT[gate]} | {sum(c.values())} | {c['pass']} | {c['fail']} | {c['refuse']} |"
        )
    lines += [
        "",
        f"The {len(controls)} control cells, run on the unpatched builds, write the same stops as the first "
        ""
        "stops of "
        "the patched `parity` cells, and the evaluated-profile cells of the element each profile does not "
        "touch write "
        "the same stops as its `parity` cells.",
        f"In {identical} of {len(identity)} identity runs the random engine's state at every track start, "
        "step end "
        "and event end is the same with the hooks on as with the stock classes alone, each hook calls its "
        "base method "
        "once per call, and the transported records are the same hooked, unhooked and on the unpatched "
        "build.",
        f"Of the {len(streams)} groups of one profile, route and element, {same} wrote the same stops on "
        "both "
        ""
        ""
        "revisions, so their two cells are one sample.",
        f"The cells took {wall / 3600:.1f} core-hours.",
    ]
    return lines


def _section_comparison(
    root: Path,
    policy: Mapping[str, Any],
    harvest: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    refs: Sequence[Reference],
) -> list[str]:
    distinct = len({r["comparison"] for r in rows})
    labels = {(x["source"], x["locator"]): x["estimator"] for x in policy["estimators"]["labels"]}
    surrogate = sum(1 for r in refs if labels[(r.source, r.locator)] == SURROGATE)
    lines = [
        "### 13.3 Measured mean lives",
        "",
        f"`natural_mixture_comparison.csv` compares {len(refs)} measured mean lives of targets of natural "
        f"composition, of {len({r.z for r in refs})} elements, with the transported mixture: the rows of "
        "section 12 "
        "printed with no mass number or as natural, of an element with more than one natural isotope.",
        "The rate compared is the one the source's lifetime fit would extract from the expected "
        "decay-electron "
        "spectrum of the mixture, not the mixture's initial rate.",
        f"`mean_life_estimators.csv` records what each source prints about its fit; for {surrogate} of the "
        f"{len(refs)} it does not print everything the fit's expectation needs, so the comparison uses a "
        "surrogate "
        "estimator: the source's stated components with each unprinted input bracketed as the policy fixes, "
        ""
        "the "
        "fitted rate the middle of the bracket's range and half the range an uncertainty of its own.",
        "A row is within the band when the rate differs from the measured rate by at most three times "
        "the root "
        "sum of squares of the measurement's uncertainty, the propagated tabulated uncertainty of the "
        "records "
        ""
        "that "
        "answer it and the surrogate's.",
        f"Profiles and routes that give a measurement the same mixture and the same uncertainties give it "
        "one "
        ""
        ""
        f"comparison: {distinct} distinct comparisons.",
        "",
        "| profile | route | distinct comparisons | within | outside | qualified |",
        "|---|---|---|---|---|---|",
    ]
    for key, v in science(rows).items():
        profile, route = key.split(":")
        within = int(str(v["distinct"])) - int(str(v["outside"]))
        lines.append(
            f"| `{profile}` | {route} | {v['distinct']} | {within} | {v['outside']} | {v['qualified_rows']} |"
        )
    lines += [
        "",
        "No row qualifies for an accuracy verdict, which needs a target printed as natural, no stop "
        "answered "
        "by the "
        f"formula, an established estimator and no answering record in `{LINEAGE_OPEN}`, `{MIXED}` or "
        f"`{UNSETTLED}`: "
        "the qualified family admits no row and is reported as vacuous, not as a pass.",
    ]
    same = [r for r in rows if float(r["same_source_share"]) > 0]
    lines += [
        f"On {len(same)} of the {len(rows)} rows some stops are answered by a record whose value the "
        "compared "
        ""
        ""
        "source's own measurements produced (the compiled-in table, whose source comment credits Suzuki, "
        "Measday and "
        "Roalsvig, for their rows; an evaluated profile for its own source's rows), and `same_source_share` "
        ""
        "gives "
        "that share.",
        "Such a source derives its capture rate from its own mean life (Suzuki, Measday & Roalsvig, Eq. "
        "(4)), "
        ""
        "so on "
        "those shares a row compares a source with itself, not with an independent measurement, and the "
        "band "
        "errs "
        "wide.",
    ]
    oracle = Oracle(root)
    for route in ROUTES:
        comps = oracle.components(spec.PARITY_PROFILE, route, 12)
        lam = float(mean_rate(comps)) * PER_MICROSECOND
        life = float(sum((c.abundance / c.rate for c in comps), Fraction(0)))
        delay = float(harvest[f"parity:{route}:v11.4.2:12"]["meta"]["mean_delay_ns"])
        parts = [
            f"`{r['source']}` {r['locator']} measured {1000 / float(r['measured_rate']):.1f} ns and fitted "
            f"{1000 / float(r['fit_rate']):.1f} ns, {_sigma(r['difference'], _row_sd(r))} σ"
            for r in rows
            if r["z"] == "12" and r["profile"] == spec.PARITY_PROFILE and r["route"] == route
        ]
        lines.append(
            f"For natural magnesium on the {route} route under `parity`, the mixture's initial rate is "
            f"{lam:.4f} per µs "
            f"and its mean life {life:.1f} ns, where the transported stops' mean delay is {delay:.1f} ns; "
            + "; ".join(parts)
            + "."
        )
    outside_rows = [r for r in rows if r["verdict"] == "OUTSIDE"]
    lines += [
        "",
        "Each row outside the band splits its difference into the estimator's part (the fitted rate less "
        "the "
        "mixture's initial rate), the composition's part (that initial rate less the route's rate of the "
        "nucleus "
        "section 12 compares) and section 12's single-nucleus part, or, for a row section 12 gives no "
        "nucleus, the "
        "last two merged; each is in units of the band's standard deviation and rounded to a tenth, and "
        "every "
        ""
        "row's "
        "estimator is a surrogate.",
        "",
        "| measurement | target | route | profiles | difference (σ) | estimator (σ) | composition (σ) | "
        "single nucleus (σ) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, ...], list[str]] = {}
    for r in outside_rows:
        sd = _row_sd(r)
        merged = not r["part_single_nucleus"]
        group_key: tuple[str, ...] = (
            f"`{r['source']}` {r['locator']}",
            r["label"],
            r["route"],
            _sigma(r["difference"], sd),
            _sigma(r["part_estimator"], sd),
            _sigma(r["part_merged"] if merged else r["part_composition"], sd),
            "merged" if merged else _sigma(r["part_single_nucleus"], sd),
        )
        groups.setdefault(group_key, []).append(f"`{r['profile']}`")
    for entry, profs in groups.items():
        lines.append(
            f"| {entry[0]} | {entry[1]} | {entry[2]} | {', '.join(profs)} | " + " | ".join(entry[3:]) + " |"
        )
    lines += [
        "",
        "Every reference is exposed: the profiles and the compiled-in records rest on the same "
        "measurements, "
        "so the "
        "comparison is retrospective, never a validation.",
        "Compounds, enriched targets, stops after t = 0, detector acceptance and hyperfine effects lie "
        "outside it, and "
        "a sample printed with no mass number is taken to be of natural composition.",
        "No capture rate or profile of the dataset changes with this section.",
    ]
    return lines


def render_section(root: Path) -> str:
    """Section 13 of DATASET_D1.md, every number generated from the committed files."""
    policy = load_policy(root)
    harvest = load_harvest(root)
    rows = with_measured(root, read_csv(root / COMPARISON_RELPATH, COMPARISON_COLUMNS))
    refs = references(root)
    lines = [
        SECTION_HEADING,
        "",
        "A stop in a target of natural composition draws an isotope with the element's relative abundance, "
        "and the "
        "isotope's capture rate comes from whichever rung of the lookup answers it (section 11).",
        "`record_classes.csv` classifies every compiled-in record and every row of an evaluated profile by "
        "two axes "
        "the committed evidence establishes, the kind of target the value was measured on and the lineage "
        "of "
        "the "
        "value, and gives each an answer class; `natural_target_answers.csv` gives, for each profile and "
        "each "
        ""
        "isotope "
        "of each element of the `parity` table's validity range, the route the lookup takes and the class "
        "that "
        "answers it.",
        "`scripts/generate_g4mixture.py`, a step of `make audit`, writes both from the provenance rows, the "
        ""
        "published "
        "cells each record equals (`suzuki1987_parity_cells.csv`) and the transcriptions, and a test that "
        "shares no "
        "code with it asserts the class of every compiled-in record from that evidence.",
        "",
    ]
    lines += _section_classes(root, refs) + [""]
    lines += _section_harvest(policy, harvest) + [""]
    lines += _section_comparison(root, policy, harvest, rows, refs)
    return "\n".join(lines) + "\n"
