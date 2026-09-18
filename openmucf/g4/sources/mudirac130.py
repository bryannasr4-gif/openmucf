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
import hashlib
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

PROFILE = "mudirac130"
SEAM = "d3_transitions"
#: The bibliography key of the generator's reference.
BIBKEY = "Sturniolo2021"
MUDIRAC_VERSION = "1.3.0"
#: The pinned identities of the generator's release and of the three input files the input set is
#: built from: the release archive, the two data files it bundles (by git blob id) and the charge
#: radii table whose uncertainty column gives the radius shift of the `rsig` runs.
TARBALL_SHA256 = "e50e346c3b9dedb29c11685d42fa9042ee92fd89240aaf07c8fd2342187bece0"
RADII_BLOB = "92c402f80690adcefc6aaa6e23529418c4660160"
ABUNDANT_BLOB = "2f714ca6282681fa73488189e197eceab3ab6dcd"
IAEA_SHA256 = "683d028dae93235376ddf7a345f736561ee7c0dba070091912c034bdf740eee6"
#: The highest shell of the circular chain the tables carry.
N_MAX = 8
#: The lowest shell whose circular states are checked against the hydrogen-like solution.
IDEAL_FROM = 6
#: The Fermi skin thickness passed on every run, in fm, as the input file spells it.
FERMI_T = "2.3"
#: The factor on the rms radius of the `r101` runs.
SIZE_FLOOR = Decimal("1.01")
#: tol = TOL_FACTOR * sqrt(sigma_meas**2 + SIGMA_CALC**2).
TOL_FACTOR = 3
SIGMA_CALC = 0
#: The largest relative difference a circular state may show against the hydrogen-like run.
NMAX_BOUND = Decimal("0.01")
K_TABLE = "k_shell_energy"
LEVEL_TABLE = "level_energy"
#: The bibliography keys of the transition energies the tables are compared with.
VALIDATION_SOURCES = ("Fricke1995", "Saito2025")

D3_RELDIR = "data/g4/d3"
INPUTS_RELPATH = f"{D3_RELDIR}/mudirac_inputs.csv"
RUNS_RELPATH = f"{D3_RELDIR}/mudirac_runs.csv"
STATES_RELPATH = f"{D3_RELDIR}/mudirac_states.csv"
LINES_RELPATH = f"{D3_RELDIR}/mudirac_lines.csv"
NMAX_RELPATH = f"{D3_RELDIR}/mudirac_nmax_check.csv"
CELLS_RELPATH = f"{D3_RELDIR}/validation_cells.csv"
ORIGIN_RELPATH = f"{D3_RELDIR}/validation_radius_origin.csv"
VALIDATION_RELPATH = f"{D3_RELDIR}/validation.csv"

#: The run kinds whose results are committed, in their order: the bundled radius, the rms radius
#: moved by its uncertainty, the rms radius times SIZE_FLOOR (validation nuclides only), and one
#: hydrogen-like run per checked shell.
IDEAL_KINDS = tuple(f"ideal{n}" for n in range(IDEAL_FROM, N_MAX + 1))
COMMITTED_KINDS = ("base", "rsig", "r101", *IDEAL_KINDS)
#: Runs made for the evidence only and never committed: the base input a second time, and the
#: base input without its radius line.
EVIDENCE_KINDS = ("again", "defaultradius")
#: The kinds whose printed states and lines are committed.
PRINTED_KINDS = ("base", "rsig", "r101")

INPUTS_COLUMNS = (
    "Z", "A", "symbol", "rms_fm", "sigma_rms_fm", "radius_fm", "fermi_t_fm", "most_abundant",
    "iaea_rms_fm", "source",
)

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


# --------------------------------------------------------------------------------------------
# the inputs
# --------------------------------------------------------------------------------------------

#: The sphere-equivalent radius MuDirac takes is the rms radius times this factor.
SPHERE_FACTOR = math.sqrt(5.0 / 3.0)
_SYMBOL = re.compile(r"[A-Z][a-z]?")


def git_blob_id(payload: bytes) -> str:
    """git's object name for a blob holding ``payload`` (``git hash-object``)."""
    return hashlib.sha1(b"blob %d\0" % len(payload) + payload).hexdigest()


def input_source() -> str:
    """The ``source`` cell every input row carries: the two pinned files the row is read from."""
    return f"nuclear_radii.dat blob {RADII_BLOB}; charge_radii.csv sha256 {IAEA_SHA256}"


@dataclass(frozen=True)
class InputRow:
    """One member of the input set: the bundled radius passed to MuDirac and what it is checked with."""

    z: int
    a: int
    symbol: str
    rms_fm: str
    sigma_rms_fm: str
    radius_fm: str
    fermi_t_fm: str
    most_abundant: int
    iaea_rms_fm: str
    source: str

    @property
    def nuclide(self) -> tuple[int, int]:
        return (self.z, self.a)

    def cells(self) -> list[str]:
        return [str(self.z), str(self.a), self.symbol, self.rms_fm, self.sigma_rms_fm, self.radius_fm,
                self.fermi_t_fm, str(self.most_abundant), self.iaea_rms_fm, self.source]


def build_inputs(radii: bytes, abundant: bytes, iaea: bytes) -> tuple[InputRow, ...]:
    """The input set from the three pinned files, after checking each file's pin.

    Every (Z, A) with 1 <= Z <= 92 the bundled radius file lists, ascending; the symbol and the rms
    radius with its uncertainty from the charge-radii table's row for that (Z, A); the most abundant
    isotope of the Z from the bundled abundance file.
    """
    if git_blob_id(radii) != RADII_BLOB:
        raise Mudirac130Error(f"nuclear_radii.dat is blob {git_blob_id(radii)}, pinned {RADII_BLOB}")
    if git_blob_id(abundant) != ABUNDANT_BLOB:
        raise Mudirac130Error(f"abundant.dat is blob {git_blob_id(abundant)}, pinned {ABUNDANT_BLOB}")
    digest = hashlib.sha256(iaea).hexdigest()
    if digest != IAEA_SHA256:
        raise Mudirac130Error(f"charge_radii.csv is sha256 {digest}, pinned {IAEA_SHA256}")
    bundled: dict[tuple[int, int], str] = {}
    for line in radii.decode("ascii").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        z, a, radius = line.split()
        bundled[(int(z), int(a))] = radius
    most: dict[int, int] = {}
    for line in abundant.decode("ascii").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        z, a = line.split()
        most[int(z)] = int(a)
    table = {(int(r["z"]), int(r["a"])): r for r in csv.DictReader(iaea.decode("ascii").splitlines())}
    rows = []
    for (z, a), radius in sorted(bundled.items()):
        if not 1 <= z <= 92:
            continue
        measured = table[(z, a)]
        rows.append(InputRow(
            z=z, a=a, symbol=measured["symbol"], rms_fm=repr(float(radius) / SPHERE_FACTOR),
            sigma_rms_fm=measured["radius_unc"], radius_fm=radius, fermi_t_fm=FERMI_T,
            most_abundant=most[z], iaea_rms_fm=measured["radius_val"], source=input_source(),
        ))
    return tuple(rows)


def render_inputs(rows: tuple[InputRow, ...]) -> bytes:
    """``mudirac_inputs.csv``: LF, ASCII, the header then one row per member in order."""
    lines = [",".join(INPUTS_COLUMNS)] + [",".join(row.cells()) for row in rows]
    return ("\n".join(lines) + "\n").encode("ascii")


def load_inputs(path: Path) -> tuple[InputRow, ...]:
    """Parse ``mudirac_inputs.csv``: typed cells, the rms radius the bundled radius implies, the
    pinned fermi_t and source, one row per (Z, A), ascending."""
    rows: list[InputRow] = []
    previous: tuple[int, int] | None = None
    for where, r in read_rows(path, INPUTS_COLUMNS):
        z, a = integer(r["Z"], where, "Z"), integer(r["A"], where, "A")
        if not _SYMBOL.fullmatch(r["symbol"]):
            raise CellError(f"{where}: symbol must be an element symbol, got {r['symbol']!r}")
        for column in ("rms_fm", "sigma_rms_fm", "radius_fm", "iaea_rms_fm"):
            if not _UNSIGNED_DECIMAL.fullmatch(r[column]):
                raise CellError(f"{where}: {column} must be digits, a point and digits, got {r[column]!r}")
        if r["rms_fm"] != repr(float(r["radius_fm"]) / SPHERE_FACTOR):
            raise CellError(f"{where}: rms_fm is not radius_fm / sqrt(5/3) as Python prints it")
        if r["fermi_t_fm"] != FERMI_T:
            raise CellError(f"{where}: fermi_t_fm must be {FERMI_T}, got {r['fermi_t_fm']!r}")
        most = integer(r["most_abundant"], where, "most_abundant")
        if r["source"] != input_source():
            raise CellError(f"{where}: source must name the two pinned files, got {r['source']!r}")
        if previous is not None and (z, a) == previous:
            raise DuplicateKeyError(f"{where}: duplicate key ({z}, {a})")
        if previous is not None and (z, a) < previous:
            raise OrderError(f"{where}: rows are ascending by (Z, A)")
        previous = (z, a)
        rows.append(InputRow(z, a, r["symbol"], r["rms_fm"], r["sigma_rms_fm"], r["radius_fm"],
                             r["fermi_t_fm"], most, r["iaea_rms_fm"], r["source"]))
    return tuple(rows)


def radius_disagreements(rows: tuple[InputRow, ...]) -> list[tuple[int, int, str, str]]:
    """The members whose bundled rms radius and charge-radii table value differ at 4 decimals:
    ``(Z, A, bundled rms, table value)``. Listed, never edited: the bundled value is the one passed."""
    return [(row.z, row.a, f"{float(row.rms_fm):.4f}", row.iaea_rms_fm) for row in rows
            if round(float(row.rms_fm), 4) != round(float(row.iaea_rms_fm), 4)]


# --------------------------------------------------------------------------------------------
# the circular chain and the run inputs
# --------------------------------------------------------------------------------------------


def shell(n: int) -> str:
    """The IUPAC shell letter of principal quantum number ``n`` (K for 1)."""
    return chr(ord("J") + n)


def orbit(n: int, upper: bool) -> str:
    """The IUPAC orbit of the circular state of shell ``n`` (l = n - 1): number 2l for j = l - 1/2,
    2l + 1 for j = l + 1/2; K1 for n = 1."""
    if n == 1:
        return "K1"
    return f"{shell(n)}{2 * (n - 1) + (1 if upper else 0)}"


def circular_orbits(n: int) -> tuple[str, ...]:
    return ("K1",) if n == 1 else (orbit(n, False), orbit(n, True))


def chain_specs() -> list[str]:
    """``xr_lines`` for the circular chain to N_MAX, in MuDirac's range spelling."""
    def spec(n: int) -> str:
        return ":".join(circular_orbits(n))
    return [f"{spec(n - 1)}-{spec(n)}" for n in range(2, N_MAX + 1)]


def chain_lines() -> set[str]:
    """Every orbit-to-orbit line the chain specs name."""
    return {f"{lo}-{hi}" for n in range(2, N_MAX + 1) for lo in circular_orbits(n - 1)
            for hi in circular_orbits(n)}


def extra_lines(cells: tuple[Cell, ...], nuclide: tuple[int, int]) -> list[str]:
    """The gated quantities of ``nuclide`` the chain does not already name, in first-seen order."""
    out: list[str] = []
    chain = chain_lines()
    for cell in cells:
        if cell.gated and cell.nuclide == nuclide and cell.quantity not in chain and cell.quantity not in out:
            out.append(cell.quantity)
    return out


def run_id(row: InputRow, kind: str) -> str:
    return f"{row.symbol}{row.a}_{kind}"


def run_kinds(row: InputRow, validation: set[tuple[int, int]]) -> list[str]:
    """The committed kinds run for ``row``: every kind, `r101` only for a validation nuclide."""
    return [kind for kind in COMMITTED_KINDS if kind != "r101" or row.nuclide in validation]


def run_radius(row: InputRow, kind: str) -> str | None:
    """The radius line's value for ``kind``; None where the input carries no radius line."""
    rms = float(row.radius_fm) / SPHERE_FACTOR
    if kind == "rsig":
        return repr((rms + float(row.sigma_rms_fm)) * SPHERE_FACTOR)
    if kind == "r101":
        return repr(rms * float(SIZE_FLOOR) * SPHERE_FACTOR)
    if kind == "defaultradius":
        return None
    return row.radius_fm


def render_input(row: InputRow, kind: str, extra: list[str]) -> str:
    """The MuDirac input file of ``row`` under ``kind``: the fixed settings in their order, the kind's
    radius, the circular chain plus ``extra`` (none on a hydrogen-like run), and on a hydrogen-like
    run the shell from which the atom is hydrogen-like."""
    if kind not in COMMITTED_KINDS + EVIDENCE_KINDS:
        raise ValueError(f"unknown run kind {kind!r}")
    lines = [
        f"element: {row.symbol}",
        f"isotope: {row.a}",
        "nuclear_model: FERMI2",
        "uehling_correction: TRUE",
        "reduced_mass: TRUE",
        "optimise_fermi_parameters: FALSE",
    ]
    radius = run_radius(row, kind)
    if radius is not None:
        lines.append(f"radius: {radius}")
    ideal = kind in IDEAL_KINDS
    lines += [
        f"fermi_t: {row.fermi_t_fm}",
        "output: 2",
        "xr_print_precision: 6",
        "state_print_precision: 6",
        "xr_lines: " + ",".join(chain_specs() + ([] if ideal else extra)),
    ]
    if ideal:
        lines.append(f"ideal_atom_minshell: {shell(int(kind[len('ideal'):]))}")
    return "\n".join(lines) + "\n"


def mudirac_argv(binary: str, infile: str) -> list[str]:
    """The one command line a run uses: the binary and its input file, never a second argument --
    the second argument is the only route by which MuDirac reads measured energies."""
    return [binary, infile]
