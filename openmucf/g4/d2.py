"""Reproduce the capturing-atom selector and compare primary capture rows."""

from __future__ import annotations

import csv
import math
import re
from bisect import bisect_left
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

HALOGENS = frozenset((9, 17, 35, 53, 85))
DRAW_COUNT = 1 << 20
MEASUREMENT_COLUMNS = (
    "record_id", "source_id", "source_sha256", "locator", "material_formula", "phase",
    "isotope_composition", "temperature_K", "density_g_cm3", "beam_conditions",
    "observable", "normalization", "reported_value", "reported_uncertainty", "units",
    "efficiency_correction", "attenuation_correction", "cascade_correction",
    "transfer_assumption", "stopping_model", "covariance_reference", "primary_read",
    "inferable_parameter", "qualification", "missing_inputs",
)
CORRECTION_COLUMNS = (
    "efficiency_correction", "attenuation_correction", "cascade_correction", "transfer_assumption",
)
RECORDS = (
    "d2-initial-capture-general", "d2-selector-oxides", "d2-selector-fluorides",
    "d2-selector-chlorides", "d2-selector-other-compounds", "d2-selector-alloys",
    "d2-selector-gases", "d2-selector-hydrogenous", "d2-selector-powder-mixtures",
)
SUBCLASSES = {
    "d2-selector-alloys": frozenset(("alloy", "intermetallic")),
    "d2-selector-other-compounds": frozenset((
        "bromide", "iodide", "sulfide", "nitride", "boride", "ternary",
    )),
}
_ELEMENT_SYMBOLS = (
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
)
_SYMBOLS = {symbol: z for z, symbol in enumerate(_ELEMENT_SYMBOLS, 1)}


@dataclass(frozen=True)
class Element:
    """An element in material order, with Geant4's exact density and abundances."""

    z: int
    atom_density: float
    isotope_n: tuple[int, ...]
    isotope_abundance: tuple[float, ...]


def factor(z: int) -> float:
    if z in HALOGENS:
        return 0.66
    if z == 8:
        return 0.56
    return 1.0


def selector_index(elements: tuple[Element, ...], draw: float) -> int:
    """Select the first cumulative bin that contains the supplied uniform draw."""
    if not elements or not 0.0 <= draw < 1.0:
        raise ValueError("invalid selector input")
    if len(elements) == 1:
        return 0
    cumulative: list[float] = []
    total = 0.0
    for element in elements:
        total += factor(element.z) * element.z * element.atom_density
        cumulative.append(total)
    needle = total * draw
    return bisect_left(cumulative, needle)


def isotope_at_half(element: Element) -> int:
    if not element.isotope_n or len(element.isotope_n) != len(element.isotope_abundance):
        raise ValueError("invalid isotope input")
    if len(element.isotope_n) == 1:
        return element.isotope_n[0]
    remaining = 0.5
    for n, abundance in zip(element.isotope_n, element.isotope_abundance, strict=True):
        remaining -= abundance
        if remaining <= 0.0:
            return n
    raise ValueError("isotope abundances do not cover the draw")


def selector_counts(elements: tuple[Element, ...], draws: int = DRAW_COUNT) -> tuple[int, ...]:
    """Count a fixed midpoint ladder by bisection on each cumulative boundary."""
    if draws <= 0 or not elements:
        raise ValueError("invalid count input")
    boundaries = [0]
    for index in range(len(elements) - 1):
        low, high = 0, draws
        while low < high:
            mid = (low + high) // 2
            if selector_index(elements, (mid + 0.5) / draws) <= index:
                low = mid + 1
            else:
                high = mid
        boundaries.append(low)
    boundaries.append(draws)
    return tuple(b - a for a, b in zip(boundaries, boundaries[1:], strict=False))


def a_g4(z1: int, z2: int) -> Fraction:
    return Fraction(factor(z1) * z1) / Fraction(factor(z2) * z2)


def formula_atoms(formula: str) -> dict[str, int]:
    """Parse a plain formula without silently accepting unparsed syntax."""
    atoms: dict[str, int] = {}
    position = 0
    for match in re.finditer(r"([A-Z][a-z]?)([0-9]*)", formula):
        if match.start() != position:
            raise ValueError(f"unsupported formula: {formula}")
        atoms[match[1]] = atoms.get(match[1], 0) + int(match[2] or "1")
        position = match.end()
    if not formula or position != len(formula):
        raise ValueError(f"unsupported formula: {formula}")
    return atoms


def h_share(formula: str) -> Fraction:
    atoms = formula_atoms(formula)
    denominator = sum(Fraction(factor(_SYMBOLS[s]) * _SYMBOLS[s]) * n for s, n in atoms.items())
    return Fraction(atoms.get("H", 0)) / denominator


def atomic_ratio_to_probability(m: float, n: float, ratio: float) -> tuple[float, float]:
    if not all(math.isfinite(x) for x in (m, n, ratio)) or m <= 0 or n <= 0 or ratio < 0:
        raise ValueError("m and n must be positive and R finite nonnegative")
    denominator = m * ratio + n
    return m * ratio / denominator, n / denominator


def identifiability_counterexample() -> tuple[dict[str, Fraction], dict[str, Fraction]]:
    """Distinct capture probabilities with the same two line yields."""
    return (
        {"P_Z": Fraction(1, 2), "P_O": Fraction(1, 2), "T_Z": Fraction(1), "T_O": Fraction(1)},
        {"P_Z": Fraction(1, 4), "P_O": Fraction(3, 4), "T_Z": Fraction(2), "T_O": Fraction(2, 3)},
    )


def qualification(value: str) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for item in value.split(";"):
        if "=" in item and "@" in item:
            key, rest = item.split("=", 1)
            result[key] = tuple(rest.rsplit("@", 1))  # type: ignore[assignment]
    return result


def classify(row: dict[str, str]) -> tuple[str, frozenset[str]]:
    formula = row["material_formula"]
    if row["inferable_parameter"] == "P(H)" and "H" in formula_atoms(formula):
        return "d2-selector-hydrogenous", frozenset(("d2-selector-hydrogenous",))
    if "+" in formula:
        return "d2-selector-powder-mixtures", frozenset(("d2-selector-powder-mixtures",))
    if row["phase"] == "gas":
        return "d2-selector-gases", frozenset(("d2-selector-gases",))
    if formula.startswith(("alloy(", "intermetallic(")):
        sub = "alloy" if formula.startswith("alloy(") else "intermetallic"
        return "d2-selector-alloys", frozenset((sub,))
    atoms = formula_atoms(formula)
    if len(atoms) == 2 and "O" in atoms and "H" not in atoms:
        return "d2-selector-oxides", frozenset(("d2-selector-oxides",))
    if len(atoms) == 2 and "F" in atoms:
        return "d2-selector-fluorides", frozenset(("d2-selector-fluorides",))
    if len(atoms) == 2 and "Cl" in atoms:
        return "d2-selector-chlorides", frozenset(("d2-selector-chlorides",))
    subs = {name for symbol, name in (("Br", "bromide"), ("I", "iodide"),
            ("S", "sulfide"), ("N", "nitride"), ("B", "boride")) if len(atoms) == 2 and symbol in atoms}
    if len(atoms) >= 3:
        subs.add("ternary")
    if subs:
        return "d2-selector-other-compounds", frozenset(subs)
    raise ValueError(f"unidentifiable class: {formula}")


def gate_reason(row: dict[str, str], source: dict[str, str]) -> str:
    if source["access"] != "AVAILABLE" or source["primary_read"] != "true" or source["kind"] != "measurement":
        return "access"
    parameter = row["inferable_parameter"]
    is_ratio = row["observable"] == "atomic_capture_ratio" and bool(
        re.fullmatch(r"A\([A-Za-z]+/[A-Za-z]+\)", parameter))
    is_hydrogen = parameter == "P(H)" and "H" in formula_atoms(row["material_formula"])
    if not (is_ratio or is_hydrogen):
        return "not a per-atom ratio"
    qual = qualification(row["qualification"])
    stage = qual.get("stage", ("unstated", ""))[0]
    if stage not in ("terminal", "initial=terminal"):
        return f"stage {stage}"
    population = qual.get("population", ("unstated", ""))[0]
    if population != "all_stops":
        return f"population {population}"
    if qual.get("method", ("unstated", ""))[0] == "unstated":
        return "method undocumented"
    for field in CORRECTION_COLUMNS:
        if not row[field]:
            return f"correction undocumented: {field}"
    is_hydrogen_limit = parameter == "P(H)" and row["reported_value"].startswith("<")
    if not row["reported_uncertainty"] and not is_hydrogen_limit:
        return "no uncertainty"
    try:
        classify(row)
    except ValueError:
        return "unidentifiable class"
    return ""


def independent(a: dict[str, str], b: dict[str, str], sources: dict[str, dict[str, str]]) -> bool:
    if a["source_id"] == b["source_id"]:
        return False

    def data_chain(source_id: str) -> set[str]:
        seen = set()
        while source_id and source_id not in seen:
            seen.add(source_id)
            source_id = sources.get(source_id, {}).get("same_data_as", "")
        return seen

    if data_chain(a["source_id"]) & data_chain(b["source_id"]):
        return False
    qa, qb = qualification(a["qualification"]), qualification(b["qualification"])
    if any(q.get("lineage", ("", ""))[0] != "own" or not q.get("lineage", ("", ""))[1] for q in (qa, qb)):
        return False
    ia, ib = (q.get("inputs", ("unstated", ""))[0] for q in (qa, qb))
    if not ia or not ib or "unstated" in (ia, ib):
        return False
    return set(ia.split("+" )).isdisjoint(ib.split("+"))


def class_outcome(record: str, rows: list[tuple[dict[str, str], str, frozenset[str]]],
                  sources: dict[str, dict[str, str]]) -> bool | None:
    if record in ("d2-initial-capture-general", "d2-selector-powder-mixtures"):
        return None
    if any(result == "outside" for _, result, _ in rows):
        return False
    inside = [(row, subs) for row, result, subs in rows if result == "inside"]
    required = SUBCLASSES.get(record, frozenset((record,)))
    supporting = [(row, subs) for row, subs in inside if subs & required]
    if not all(any(sub in subs for _, subs in supporting) for sub in required):
        return None
    for row, _ in supporting:
        if not any(independent(row, other, sources) for other, _ in supporting):
            return None
    return True


def decimal_fraction(value: str) -> Fraction:
    return Fraction(Decimal(value))


def compare_ratio(predicted: Fraction, value: str, uncertainty: str) -> str:
    if value.startswith("<"):
        return "outside" if predicted > decimal_fraction(value[1:]) else "not_excluded"
    observed = decimal_fraction(value)
    if uncertainty.startswith("+") and "-" in uncertainty[1:]:
        plus, minus = uncertainty[1:].split("-", 1)
        sigma = decimal_fraction(plus if predicted >= observed else minus)
    else:
        sigma = decimal_fraction(uncertainty.removeprefix("+-"))
    return "inside" if abs(predicted - observed) <= 3 * sigma else "outside"


def load_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"{path}: unexpected columns")
        return [dict(row) for row in reader]


def load_measurements(path: Path, sources: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows = load_csv(path, MEASUREMENT_COLUMNS)
    for row in rows:
        source = sources[row["source_id"]]
        if source["kind"] == "review" or source["primary_read"] != "true":
            raise ValueError(f"{row['record_id']}: source is not primary-read")
    return rows
