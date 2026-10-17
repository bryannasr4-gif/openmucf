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

import csv
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path

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
