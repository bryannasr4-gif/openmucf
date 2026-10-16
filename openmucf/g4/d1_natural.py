"""The capture rate a Geant4 build gives a target of natural isotopic composition.

Geant4 picks the nucleus that captures a stopped negative muon in two steps.
``G4ElementSelector::SelectZandA`` draws the element, then draws one of its isotopes with the
probabilities of the element's relative abundance vector. ``GetMuonCaptureRate(Z, A)`` is then asked
about that one nucleus. The compiled-in table answers only an exact ``(Z, A)``, and every other
nucleus gets the Goulard-Primakoff formula. Where the table's record for an element was measured on
a target of natural composition, a natural target therefore reaches that measurement only on the
stops that draw the record's own mass number.

This module computes, per element and per profile of the capture table, the share of stops each
route answers and the abundance-weighted capture rate ``sum_i w_i * lambda_i``. Each
``lambda_i`` follows the lookup a patched build runs for that isotope. Under ``parity`` that is the
compiled-in chain, reproduced by :func:`openmucf.g4.sources.d1_nuclear_capture.capture_rate`. Under
any other profile, the profile's own row keyed by exactly that nuclide answers first, and the
compiled-in chain answers the rest. The weighted rate is compared with the compiled-in record where
the isotope audit establishes that the record rests on a measurement of the element printed with no mass
number (for lead, a record that matches a reconstructed combination with a radiogenic-lead row:
DATASET_D1.md section 11). It is a first-order estimate: a natural target decays in Geant4 as a mixture of
exponentials, one per isotope, and the weighted rate is the capture part of that mixture's initial
rate.

The isotopic compositions are the ones ``cpp/tools/harvest_natural.cc`` read from Geant4 builds and
``data/g4/d1/geant4_natural_isotopes.csv`` commits. Nothing here changes how Geant4 picks or rates a
nucleus.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from openmucf.g4 import spec
from openmucf.g4.sources import d1_nuclear_capture as d1

#: The committed isotopic compositions, relative to the repository root.
ISOTOPES_RELPATH = "data/g4/d1/geant4_natural_isotopes.csv"
#: The builds the compositions were harvested from.
BUILDS_RELPATH = "data/g4/d1/natural_harvest_builds.csv"
#: The generated study.
STUDY_RELPATH = "data/g4/d1/natural_target_capture.csv"
#: The harvest driver, whose SHA-256 every build record carries.
HARNESS_RELPATH = "cpp/tools/harvest_natural.cc"
#: The capture tables of the D1 directory, every profile's.
CAPTURE_GLOB = "d1_capture*.g4dat"

ISOTOPE_COLUMNS = ("z", "symbol", "a", "abundance")
BUILD_COLUMNS = ("build", "revision", "geant4_version", "gcc", "harness_sha256", "compile_flags")
STUDY_COLUMNS = (
    "profile", "z", "symbol", "isotopes", "profile_share", "table_share", "formula_share",
    "nonpositive_share", "weighted_rate", "relative_spread", "reference_a", "reference_rate",
    "reference_unc", "relative_difference", "formula_at_reference",
)
#: The builds the harvest runs on: one unpatched install per Geant4 revision the patches serve.
BUILDS = (
    ("unpatched-v11.4.2", "v11.4.2"),
    ("unpatched-v11.5.0.beta", "v11.5.0.beta"),
)
#: The flags every build record states; the driver does no arithmetic, so they are a record only.
COMPILE_FLAGS = "-std=c++17 -O2 -ffp-contract=off"

#: The three routes a lookup can take.
PROFILE_ROUTE, TABLE_ROUTE, FORMULA_ROUTE = "profile", "table", "formula"

_ELEMENT_LINE = re.compile(r"E\t([0-9]+)\t([A-Z][a-z]?)\t([0-9]+)\t([0-9]+)\t([0-9]+)")
_ISOTOPE_LINE = re.compile(r"I\t([0-9]+)\t([0-9]+)\t(\S+)")


class NaturalTargetError(RuntimeError):
    """A harvest, a committed file or a profile this study cannot be computed from."""


@dataclass(frozen=True)
class Isotope:
    """One isotope of an element's natural composition, as a Geant4 build stores it."""

    z: int
    symbol: str
    a: int
    abundance: float


@dataclass(frozen=True)
class Route:
    """How a lookup answered one isotope: the route, and the rate in ns^-1."""

    route: str
    rate: float


# --------------------------------------------------------------------------------------------
# the harvest
# --------------------------------------------------------------------------------------------


def parse_harvest(text: str) -> tuple[Isotope, ...]:
    """The isotopes ``harvest_natural.cc`` printed, refusing any line it does not write and any
    element whose isotope lines disagree with its element line."""
    elements: dict[int, tuple[str, int]] = {}
    isotopes: list[Isotope] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line:
            continue
        element = _ELEMENT_LINE.fullmatch(line)
        isotope = _ISOTOPE_LINE.fullmatch(line)
        if element:
            z = int(element[1])
            if z in elements or (elements and z != max(elements) + 1):
                raise NaturalTargetError(f"line {number}: element Z={z} out of order")
            elements[z] = (element[2], int(element[5]))
        elif isotope:
            z, a = int(isotope[1]), int(isotope[2])
            if z not in elements or z != max(elements):
                raise NaturalTargetError(f"line {number}: isotope of Z={z} outside its element")
            abundance = float.fromhex(isotope[3])
            if not (math.isfinite(abundance) and abundance > 0.0):
                raise NaturalTargetError(f"line {number}: abundance {isotope[3]} is not positive")
            isotopes.append(Isotope(z, elements[z][0], a, abundance))
        else:
            raise NaturalTargetError(f"line {number}: not a harvest line: {line!r}")
    for z, (_, natural) in elements.items():
        if sum(1 for isotope in isotopes if isotope.z == z) != natural:
            raise NaturalTargetError(f"Z={z}: the isotope lines do not match the element line")
    return tuple(isotopes)


def harvest(raw: Path, root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The committed rows, from one raw directory per build of :data:`BUILDS`: the isotopes, which
    every build must print byte for byte alike, and one record per build."""
    harness = hashlib.sha256((root / HARNESS_RELPATH).read_bytes()).hexdigest()
    outputs: list[bytes] = []
    builds: list[dict[str, str]] = []
    for build, revision in BUILDS:
        folder = raw / build
        digest = (folder / "sha256.txt").read_text(encoding="ascii").split()[0]
        if digest != harness:
            raise NaturalTargetError(f"{build}: the compiled driver is not {HARNESS_RELPATH}")
        if (folder / "stderr.txt").read_bytes():
            raise NaturalTargetError(f"{build}: the driver wrote to stderr")
        outputs.append((folder / "natural.txt").read_bytes())
        builds.append({
            "build": build, "revision": revision,
            "geant4_version": (folder / "version.txt").read_text(encoding="ascii").strip(),
            "gcc": (folder / "gcc.txt").read_text(encoding="ascii").strip(),
            "harness_sha256": harness, "compile_flags": COMPILE_FLAGS,
        })
    if any(output != outputs[0] for output in outputs):
        raise NaturalTargetError("the builds printed different compositions")
    isotopes = parse_harvest(outputs[0].decode("ascii"))
    rows = [{"z": str(i.z), "symbol": i.symbol, "a": str(i.a), "abundance": repr(i.abundance)}
            for i in isotopes]
    return rows, builds


# --------------------------------------------------------------------------------------------
# the committed inputs
# --------------------------------------------------------------------------------------------


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if b"\r" in raw:
        raise NaturalTargetError(f"{path.name} contains CR")
    reader = csv.DictReader(io.StringIO(raw.decode("ascii")))
    if tuple(reader.fieldnames or ()) != columns:
        raise NaturalTargetError(f"{path.name} header is {reader.fieldnames!r}, expected {columns!r}")
    return list(reader)


def load_isotopes(path: Path) -> dict[int, tuple[Isotope, ...]]:
    """The committed compositions by Z, each element's isotopes in the order Geant4 stores them."""
    out: dict[int, list[Isotope]] = {}
    for row in _read_csv(path, ISOTOPE_COLUMNS):
        isotope = Isotope(int(row["z"]), row["symbol"], int(row["a"]), float(row["abundance"]))
        if repr(isotope.abundance) != row["abundance"] or not isotope.abundance > 0.0:
            raise NaturalTargetError(f"{path.name}: abundance {row['abundance']!r} is not canonical")
        out.setdefault(isotope.z, []).append(isotope)
    return {z: tuple(isotopes) for z, isotopes in out.items()}


def load_builds(path: Path) -> list[dict[str, str]]:
    """The build records."""
    return _read_csv(path, BUILD_COLUMNS)


def profile_rows(table: spec.G4DatTable) -> dict[tuple[int, int], float]:
    """A capture profile's rows a transported nucleus can reach: those keyed by a mass number. A
    natural-composition row (``A = 0``) answers only a request that names it, and a transported
    nucleus always carries a mass number."""
    return {(int(z), int(a)): float(value) for z, a, value, _ in table.records if int(a) != 0}


def capture_profiles(directory: Path) -> dict[str, spec.G4DatTable]:
    """Every capture table of the D1 directory by profile, ``parity`` first."""
    tables: dict[str, spec.G4DatTable] = {}
    for path in sorted(directory.glob(CAPTURE_GLOB)):
        table = spec.parse(path.read_bytes().decode("ascii"))
        tables[table.directives["PROFILE"]] = table
    if spec.PARITY_PROFILE not in tables:
        raise NaturalTargetError("the D1 directory carries no parity capture table")
    return {name: tables[name] for name in sorted(tables, key=lambda n: (n != spec.PARITY_PROFILE, n))}


def element_label_records(audit: Mapping[tuple[int, int], d1.IsotopeAuditRow]) -> dict[int, int]:
    """Z to the mass number of the compiled-in record the isotope audit settles as not isotope-resolved:
    one whose value rests on a measurement of the element printed with no mass number, which for lead
    matches a reconstructed combination with a measurement of radiogenic lead (DATASET_D1.md section 11)."""
    out: dict[int, int] = {}
    for (z, a), row in audit.items():
        if row.settled and not row.isotope_resolved:
            if z in out:
                raise NaturalTargetError(f"Z={z} carries two natural-composition records")
            out[z] = a
    return out


# --------------------------------------------------------------------------------------------
# the lookup and the study
# --------------------------------------------------------------------------------------------


def lookup(
    z: int, a: int, profile: Mapping[tuple[int, int], float] | None,
    records: Sequence[tuple[int, int, float, float]], model: d1.GoulardPrimakoff,
) -> Route:
    """The rate a patched build returns for ``(z, a)``, and the route that returned it. ``profile``
    is ``None`` under ``parity``, whose table is the compiled-in one."""
    if profile is not None and (z, a) in profile:
        return Route(PROFILE_ROUTE, profile[(z, a)] / d1.MICROSECOND)
    keys = {(record_z, record_a) for record_z, record_a, _, _ in records}
    route = TABLE_ROUTE if (z, a) in keys else FORMULA_ROUTE
    return Route(route, d1.capture_rate(z, a, records, model))


def _share(isotopes: Iterable[Isotope]) -> float:
    return math.fsum(isotope.abundance for isotope in isotopes)


def study_rows(
    isotopes: Mapping[int, Sequence[Isotope]],
    profiles: Mapping[str, Mapping[tuple[int, int], float] | None],
    found: d1.D1Extraction,
    model: d1.GoulardPrimakoff,
    references: Mapping[int, int],
) -> list[dict[str, str]]:
    """One row per profile and element: the share of stops each route answers, the share whose rate
    is not a positive finite rate, the abundance-weighted rate and the weighted spread, and, where
    the element has a natural-composition record, that record and the relative difference."""
    records = found.capture_records
    compiled = {(z, a): (value, unc) for z, a, value, unc in records}
    rows: list[dict[str, str]] = []
    for name, profile in profiles.items():
        for z in sorted(isotopes):
            members = isotopes[z]
            routes = [lookup(z, i.a, profile, records, model) for i in members]
            by_route = {
                key: _share(i for i, r in zip(members, routes, strict=True) if r.route == key)
                for key in (PROFILE_ROUTE, TABLE_ROUTE, FORMULA_ROUTE)
            }
            nonpositive = _share(
                i for i, r in zip(members, routes, strict=True)
                if not (math.isfinite(r.rate) and r.rate > 0.0)
            )
            weighted = math.fsum(i.abundance * r.rate for i, r in zip(members, routes, strict=True))
            spread = math.sqrt(math.fsum(
                i.abundance * (r.rate - weighted) ** 2 for i, r in zip(members, routes, strict=True)
            )) / weighted
            row = {
                "profile": name, "z": str(z), "symbol": members[0].symbol, "isotopes": str(len(members)),
                "profile_share": repr(by_route[PROFILE_ROUTE]), "table_share": repr(by_route[TABLE_ROUTE]),
                "formula_share": repr(by_route[FORMULA_ROUTE]), "nonpositive_share": repr(nonpositive),
                "weighted_rate": repr(weighted * d1.MICROSECOND), "relative_spread": repr(spread),
                "reference_a": "", "reference_rate": "", "reference_unc": "",
                "relative_difference": "", "formula_at_reference": "",
            }
            if z in references:
                a = references[z]
                value, unc = compiled[(z, a)]
                tabulated = value / d1.MICROSECOND
                row.update({
                    "reference_a": str(a), "reference_rate": repr(value), "reference_unc": repr(unc),
                    "relative_difference": repr(weighted / tabulated - 1.0),
                    "formula_at_reference": repr(model.rate(z, a) / tabulated - 1.0),
                })
            rows.append(row)
    return rows


def reference_model(found: d1.D1Extraction) -> d1.GoulardPrimakoff:
    """The declared fallback, built from the ``#FALLBACK`` string as a consumer builds it."""
    return d1.GoulardPrimakoff.from_directive(
        d1.render_fallback_directive(d1.FALLBACK_MODEL, found.coefficients), found.zeff
    )


def compute(root: Path) -> list[dict[str, str]]:
    """The study, from the committed files under ``root``."""
    found = d1.load(root / d1.VENDORED_RELPATH)
    isotopes = load_isotopes(root / ISOTOPES_RELPATH)
    parity = capture_profiles(root / "data" / "g4" / "d1")
    if any(int(record[1]) == 0 for record in parity[spec.PARITY_PROFILE].records):
        # Under `parity` a patched build answers a nucleus the table does not key from the element's
        # natural-composition row, so the compiled-in chain would no longer be the lookup.
        raise NaturalTargetError("the parity capture table carries a natural-composition row")
    validity = parity[spec.PARITY_PROFILE].directives["VALIDITY"]
    span = re.match(r"Z:([0-9]+)-([0-9]+) ", validity)
    if span is None or sorted(isotopes) != list(range(int(span[1]), int(span[2]) + 1)):
        raise NaturalTargetError(f"the compositions do not cover the parity table's {validity!r}")
    profiles = {
        name: None if name == spec.PARITY_PROFILE else profile_rows(table) for name, table in parity.items()
    }
    references = element_label_records(d1.load_isotope_audit(root / d1.AUDIT_RELPATH))
    return study_rows(isotopes, profiles, found, reference_model(found), references)


def render_csv(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> bytes:
    """``rows`` as LF-terminated CSV bytes."""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("ascii")


# --------------------------------------------------------------------------------------------
# the document table
# --------------------------------------------------------------------------------------------

TABLE_HEADER = (
    "| Z | element | record A | stops the table answers (%) | relative difference (%) | "
    "tabulated relative uncertainty (%) |\n|---|---|---|---|---|---|"
)


def _percent(value: float, sign: bool = False) -> str:
    """``value`` in percent to a tenth, signed when ``sign``; a value that rounds to zero is ``0.0``."""
    text = format(100.0 * value, "+.1f" if sign else ".1f")
    return "0.0" if text in ("+0.0", "-0.0") else text


def render_table(rows: Sequence[Mapping[str, str]]) -> str:
    """The parity rows that carry a natural-composition record, as the document's table."""
    lines = [TABLE_HEADER]
    for row in rows:
        if row["profile"] != spec.PARITY_PROFILE or not row["reference_a"]:
            continue
        lines.append(
            f"| {row['z']} | {row['symbol']} | {row['reference_a']} | "
            f"{_percent(float(row['table_share']))} | "
            f"{_percent(float(row['relative_difference']), sign=True)} | "
            f"{_percent(float(row['reference_unc']) / float(row['reference_rate']))} |"
        )
    return "\n".join(lines)


def render_reach_sentence(tables: Mapping[str, spec.G4DatTable]) -> str:
    """The sentence of the document that counts, for each profile, the rows a transported nucleus can reach:
    those :func:`profile_rows` keeps, of all the table's rows."""
    parts = [f"{len(profile_rows(table))} of the {len(table.records)} rows of `{name}`"
             for name, table in tables.items()]
    return ("A transported nucleus always carries a mass number, so it can reach the rows of a profile "
            "keyed by a mass number: " + ", ".join(parts[:-1]) + " and " + parts[-1] + ".")


PROFILE_TABLE_HEADER = (
    "| profile | Z | element | stops the profile's rows answer (%) | relative difference (%) |\n"
    "|---|---|---|---|---|"
)


def render_profile_table(rows: Sequence[Mapping[str, str]]) -> str:
    """The rows of another profile that carry a natural-composition record and whose weighted rate
    differs from the parity row's."""
    parity = {row["z"]: row for row in rows if row["profile"] == spec.PARITY_PROFILE}
    lines = [PROFILE_TABLE_HEADER]
    for row in rows:
        if row["profile"] == spec.PARITY_PROFILE or not row["reference_a"]:
            continue
        if row["weighted_rate"] == parity[row["z"]]["weighted_rate"]:
            continue
        lines.append(
            f"| `{row['profile']}` | {row['z']} | {row['symbol']} | "
            f"{_percent(float(row['profile_share']))} | "
            f"{_percent(float(row['relative_difference']), sign=True)} |"
        )
    return "\n".join(lines)
