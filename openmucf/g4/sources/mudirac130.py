"""The ``mudirac130`` profile: MuDirac 1.3.0 run as a generator whose inputs and printed outputs are data.

The D3 energy tables are not transcribed and not computed here. MuDirac 1.3.0 is run outside this
package on inputs this module renders from ``data/g4/d3/mudirac_inputs.csv``; every state header and
every line it prints is committed as a CSV beside them, and this module turns those printed strings
into the two tables by exact decimal arithmetic whose checks raise rather than warn.

Beside the energies, the directory carries the measured transition energies the tables are compared
with, transcribed from their sources' printed tables, and the comparison is re-derived from the
committed files alone.

Standard library only, no import of any other ``openmucf`` module, and loadable by file path by an
interpreter that does not have the package installed -- ``scripts/mudirac_d3.py`` imports it that way
where the MuDirac runs happen.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

PROFILE = "mudirac130"
SEAM = "d3_transitions"
#: The bibliography keys of the transition energies the tables are compared with.
VALIDATION_SOURCES = ("Fricke1995", "Saito2025")

D3_RELDIR = "data/g4/d3"
CELLS_RELPATH = f"{D3_RELDIR}/validation_cells.csv"
ORIGIN_RELPATH = f"{D3_RELDIR}/validation_radius_origin.csv"

CELLS_COLUMNS = (
    "source", "Z", "A", "transition", "quantity", "value_keV", "unc_keV", "unc_label", "npol_keV",
    "gated", "reason", "locator", "copy_read",
)
ORIGIN_COLUMNS = ("Z", "A", "origin", "locator")
#: Why a printed value is reported but not compared: the source states no weighting for its
#: centre of gravity, or the value is a hyperfine component (or a line its source says is split),
#: which a model without hyperfine structure cannot test.
UNGATED_REASONS = ("centroid", "hyperfine")
#: Where a validation nuclide's charge radius comes from, by the source's own tables.
RADIUS_ORIGINS = ("muonic", "e-scattering", "other")

#: A transition in MuDirac's IUPAC spelling, orbit to orbit (``K1-L3``).
_LINE = re.compile(r"[K-Z][0-9]+-[K-Z][0-9]+")
_INTEGER = re.compile(r"[1-9][0-9]*")
_DECIMAL = re.compile(r"-?[0-9]+\.[0-9]+")
_UNSIGNED_DECIMAL = re.compile(r"[0-9]+\.[0-9]+")
_BOOL = {"true": True, "false": False}


class Mudirac130Error(RuntimeError):
    """A committed D3 file is malformed. Raised with the file and row named, never swallowed."""


class CarriageReturnError(Mudirac130Error):
    """The file carries a CR byte; every D3 file is committed LF-only."""


class NonAsciiError(Mudirac130Error):
    """The file carries a byte outside US-ASCII."""


class HeaderError(Mudirac130Error):
    """The header row is not the file's column tuple, in order."""


class CellError(Mudirac130Error):
    """A cell does not have its column's type, or a row breaks a rule between its cells."""


class DuplicateKeyError(Mudirac130Error):
    """Two rows carry the same key."""


class OrderError(Mudirac130Error):
    """The rows are not in the file's declared order."""


class EmptyError(Mudirac130Error):
    """The file carries a header and no rows."""


def read_rows(path: Path, columns: tuple[str, ...]) -> list[tuple[str, dict[str, str]]]:
    """The rows of a committed D3 CSV as ``(where, record)`` pairs, after the byte-level checks."""
    raw = Path(path).read_bytes()
    name = Path(path).name
    if b"\r" in raw:
        raise CarriageReturnError(f"{name} contains CR; the file is committed LF-only")
    if not raw.isascii():
        raise NonAsciiError(f"{name} carries a byte outside US-ASCII")
    lines = raw.decode("ascii").split("\n")
    if lines[-1] == "":
        lines.pop()
    rows = list(csv.reader(lines))
    if not rows or tuple(rows[0]) != columns:
        raise HeaderError(f"{name} header is {tuple(rows[0]) if rows else ()!r}, expected {columns!r}")
    out = []
    for number, cells in enumerate(rows[1:], start=2):
        where = f"{name} line {number}"
        if len(cells) != len(columns):
            raise CellError(f"{where}: {len(columns)} cells expected, got {len(cells)}")
        out.append((where, dict(zip(columns, cells, strict=True))))
    if not out:
        raise EmptyError(f"{name} carries no rows")
    return out


def integer(text: str, where: str, column: str) -> int:
    if not _INTEGER.fullmatch(text):
        raise CellError(f"{where}: {column} must be an integer without sign or leading zero, got {text!r}")
    return int(text)


def boolean(text: str, where: str, column: str) -> bool:
    if text not in _BOOL:
        raise CellError(f"{where}: {column} must be true or false, got {text!r}")
    return _BOOL[text]


def _decimals(text: str) -> int:
    return len(text.split(".")[1])


# --------------------------------------------------------------------------------------------
# the measured transition energies
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One printed transition energy, every cell as its source prints it."""

    source: str
    z: int
    a: int
    transition: str
    quantity: str
    value_kev: str
    unc_kev: str
    unc_label: str
    npol_kev: str
    gated: bool
    reason: str
    locator: str
    copy_read: str

    @property
    def nuclide(self) -> tuple[int, int]:
        return (self.z, self.a)


@dataclass(frozen=True)
class RadiusOrigin:
    z: int
    a: int
    origin: str
    locator: str


def load_cells(path: Path) -> tuple[Cell, ...]:
    """Parse the transcription: printed decimals, the uncertainty at its value's decimals, a quantity
    exactly on the gated rows, a reason exactly on the others, rows ordered by (source, Z, A)."""
    cells: list[Cell] = []
    seen: dict[tuple[str, int, int, str, str], str] = {}
    previous: tuple[int, int, int] | None = None
    for where, r in read_rows(path, CELLS_COLUMNS):
        if r["source"] not in VALIDATION_SOURCES:
            raise CellError(f"{where}: source {r['source']!r} is not one of {VALIDATION_SOURCES!r}")
        z, a = integer(r["Z"], where, "Z"), integer(r["A"], where, "A")
        if not _UNSIGNED_DECIMAL.fullmatch(r["value_keV"]):
            raise CellError(f"{where}: value_keV must be digits, a point and digits, got {r['value_keV']!r}")
        if not _UNSIGNED_DECIMAL.fullmatch(r["unc_keV"]) or Decimal(r["unc_keV"]) <= 0:
            raise CellError(f"{where}: unc_keV must be a positive printed decimal, got {r['unc_keV']!r}")
        if _decimals(r["unc_keV"]) != _decimals(r["value_keV"]):
            raise CellError(f"{where}: unc_keV {r['unc_keV']!r} is not at the decimals of {r['value_keV']!r}")
        if r["npol_keV"] and not _DECIMAL.fullmatch(r["npol_keV"]):
            raise CellError(f"{where}: npol_keV must be empty or a printed decimal, got {r['npol_keV']!r}")
        gated = boolean(r["gated"], where, "gated")
        if gated != bool(_LINE.fullmatch(r["quantity"])) or (not gated and r["quantity"]):
            raise CellError(f"{where}: a gated row names its MuDirac line in quantity, and only a gated row")
        if gated == (r["reason"] in UNGATED_REASONS) or (gated and r["reason"]):
            raise CellError(
                f"{where}: an ungated row carries one of {UNGATED_REASONS!r} in reason, and only it"
            )
        for column in ("transition", "unc_label", "locator", "copy_read"):
            if not r[column]:
                raise CellError(f"{where}: every row must carry a {column}")
        order = (VALIDATION_SOURCES.index(r["source"]), z, a)
        if previous is not None and order < previous:
            raise OrderError(f"{where}: rows are ordered by (source, Z, A)")
        previous = order
        key = (r["source"], z, a, r["transition"], r["locator"])
        if key in seen:
            raise DuplicateKeyError(f"{where}: duplicate row {key!r}, first seen at {seen[key]}")
        seen[key] = where
        cells.append(Cell(r["source"], z, a, r["transition"], r["quantity"], r["value_keV"], r["unc_keV"],
                          r["unc_label"], r["npol_keV"], gated, r["reason"], r["locator"], r["copy_read"]))
    return tuple(cells)


def load_radius_origins(path: Path) -> tuple[RadiusOrigin, ...]:
    """Parse the radius-origin file: one row per nuclide, ascending by (Z, A)."""
    rows: list[RadiusOrigin] = []
    previous: tuple[int, int] | None = None
    for where, r in read_rows(path, ORIGIN_COLUMNS):
        z, a = integer(r["Z"], where, "Z"), integer(r["A"], where, "A")
        if r["origin"] not in RADIUS_ORIGINS:
            raise CellError(f"{where}: origin {r['origin']!r} is not one of {RADIUS_ORIGINS!r}")
        if not r["locator"]:
            raise CellError(f"{where}: every row must carry a locator")
        if previous is not None and (z, a) == previous:
            raise DuplicateKeyError(f"{where}: duplicate key ({z}, {a})")
        if previous is not None and (z, a) < previous:
            raise OrderError(f"{where}: rows are ascending by (Z, A)")
        previous = (z, a)
        rows.append(RadiusOrigin(z, a, r["origin"], r["locator"]))
    return tuple(rows)


def gated_nuclides(cells: tuple[Cell, ...]) -> list[tuple[int, int]]:
    """Every (Z, A) with at least one gated row, ascending."""
    return sorted({cell.nuclide for cell in cells if cell.gated})


def load_validation(
    cells_path: Path, origin_path: Path
) -> tuple[tuple[Cell, ...], dict[tuple[int, int], RadiusOrigin]]:
    """Both transcriptions, cross-checked: the origin file names exactly the gated nuclides."""
    cells = load_cells(cells_path)
    origins = {(row.z, row.a): row for row in load_radius_origins(origin_path)}
    if set(origins) != set(gated_nuclides(cells)):
        raise CellError(
            f"{Path(origin_path).name} names {sorted(origins)}, the gated nuclides of "
            f"{Path(cells_path).name} are {gated_nuclides(cells)}"
        )
    return cells, origins
