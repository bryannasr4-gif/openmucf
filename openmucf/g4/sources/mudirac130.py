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
import io
import math
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
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
#: The label each source table gives its printed uncertainty, keyed by the source and the table its
#: locator leads with: Fricke's Table IIIA errors are statistical, its Table IIIB errors include the
#: fit error, and Saito's tables state statistical and systematic uncertainties.
UNC_LABELS = {
    ("Fricke1995", "Table IIIA"): "statistical",
    ("Fricke1995", "Table IIIB"): "statistical and fit",
    ("Saito2025", "Table III"): "statistical and systematic",
    ("Saito2025", "Table IV"): "statistical and systematic",
}
#: The table a locator leads with.
_LOCATOR_TABLE = re.compile(r"Table (?:IIIA|IIIB|III|IV)(?![A-Za-z])")
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
        table = _LOCATOR_TABLE.match(r["locator"])
        table_name = table.group(0) if table else ""
        label = UNC_LABELS.get((r["source"], table_name))
        if label is None:
            raise CellError(
                f"{where}: locator {r['locator']!r} leads with no table of {r['source']} in UNC_LABELS"
            )
        if r["unc_label"] != label:
            raise CellError(
                f"{where}: unc_label {r['unc_label']!r} is not {label!r}, the label of {table_name} "
                f"of {r['source']}"
            )
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
        z_text, a_text, radius = line.split()
        bundled[(int(z_text), int(a_text))] = radius
    most: dict[int, int] = {}
    for line in abundant.decode("ascii").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        z_text, a_text = line.split()
        most[int(z_text)] = int(a_text)
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


# --------------------------------------------------------------------------------------------
# the printed outputs
# --------------------------------------------------------------------------------------------

RUNS_COLUMNS = ("run", "Z", "A", "kind", "rc", "err_bytes")
STATES_COLUMNS = ("run", "Z", "A", "kind", "state", "n", "l", "s", "binding_eV", "total_eV")
LINES_COLUMNS = ("run", "Z", "A", "kind", "line", "delta_e_eV", "w12_per_s")
NMAX_COLUMNS = ("Z", "A", "n", "state", "full_binding_eV", "ideal_binding_eV")

_ORBIT = re.compile(r"[K-Z][0-9]+")
_COUNT = re.compile(r"0|[1-9][0-9]*")
_SIGNED_COUNT = re.compile(r"0|-?[1-9][0-9]*")
#: A state energy as MuDirac prints it: six significant digits, in either %g form.
_PRINTED = re.compile(r"-?[0-9]+(\.[0-9]+)?(e[+-][0-9]+)?")
#: A line energy as MuDirac prints it at the precision the inputs request.
_LINE_ENERGY = re.compile(r"[0-9]+\.[0-9]{6}")
#: Half a unit of the last decimal a line energy is printed with, in eV.
LINE_HALF_UNIT = Decimal("0.0000005")
#: The least uncertainty a `unc` or `u<n>` cell carries, in keV: the most by which the rounding of
#: the printed line energies can move a change measured through them. A quantity measured with the
#: outermost circular state held fixed sums at most twice (N_MAX - 1) printed lines -- the
#: N_MAX - 1 upper-chain lines down to K1, then the N_MAX - 1 lower-chain lines out again -- each
#: within LINE_HALF_UNIT of the value MuDirac computed; a (2j+1) mean is a convex combination of two
#: such sums and is bounded the same way; and the change is taken between two runs, each rounded
#: independently.
UNC_FLOOR_KEV = 4 * (N_MAX - 1) * LINE_HALF_UNIT / 1000
#: Why a nuclide the keep rule refuses is dropped, when its failure is that the Fermi parameter c
#: MuDirac computes by default from the sphere radius is not a real number.
DROP_FERMI2_C = "FERMI2 default c not real: sphere radius below sqrt(7/3)*pi*t/(4 ln 3) at t = fermi_t"
#: The lowest mass number for which MuDirac's documented default c is the square-root form.
FERMI2_SQRT_FROM_A = 5
#: Why a member is dropped when MuDirac's documented default c is set by its mass number alone.
DROP_FERMI2_FROM_A = (
    "FERMI2 default c set by the mass number alone (A below FERMI2_SQRT_FROM_A)"
)


@dataclass(frozen=True)
class RunRow:
    run: str
    z: int
    a: int
    kind: str
    rc: int
    err_bytes: int

    @property
    def clean(self) -> bool:
        return self.rc == 0 and self.err_bytes == 0


@dataclass(frozen=True)
class NmaxRow:
    z: int
    a: int
    n: int
    state: str
    full: str
    ideal: str


@dataclass(frozen=True)
class Outputs:
    """The committed run table, printed state headers, printed lines and hydrogen-like comparison."""

    inputs: tuple[InputRow, ...]
    runs: dict[str, RunRow]
    #: ``{run: {orbit: printed header energy E}}`` of every base, rsig and r101 run.
    headers: dict[str, dict[str, str]]
    #: ``{run: {line: printed transition energy}}`` of the same runs.
    lines: dict[str, dict[str, str]]
    nmax: tuple[NmaxRow, ...]


def _orbit_order(state: str) -> tuple[int, int]:
    return (ord(state[0]), int(state[1:]))


def _run_key(z: int, a: int, kind: str) -> tuple[int, int, int]:
    return (z, a, COMMITTED_KINDS.index(kind))


def _nuclide_kind(where: str, r: dict[str, str], kinds: tuple[str, ...]) -> tuple[int, int, str]:
    z, a = integer(r["Z"], where, "Z"), integer(r["A"], where, "A")
    if r["kind"] not in kinds:
        raise CellError(f"{where}: kind {r['kind']!r} is not one of {kinds!r}")
    return z, a, r["kind"]


def load_runs(path: Path) -> dict[str, RunRow]:
    """Parse ``mudirac_runs.csv``: one row per committed run, ordered by (Z, A, kind order)."""
    out: dict[str, RunRow] = {}
    previous: tuple[int, int, int] | None = None
    for where, r in read_rows(path, RUNS_COLUMNS):
        z, a, kind = _nuclide_kind(where, r, COMMITTED_KINDS)
        if not r["run"].endswith(f"{a}_{kind}"):
            raise CellError(f"{where}: run {r['run']!r} does not name A={a} and kind {kind}")
        if not _SIGNED_COUNT.fullmatch(r["rc"]) or not _COUNT.fullmatch(r["err_bytes"]):
            raise CellError(
                f"{where}: rc and err_bytes must be integers, got {r['rc']!r}, {r['err_bytes']!r}"
            )
        if r["run"] in out:
            raise DuplicateKeyError(f"{where}: duplicate run {r['run']!r}")
        order = _run_key(z, a, kind)
        if previous is not None and order < previous:
            raise OrderError(f"{where}: rows are ordered by (Z, A, kind)")
        previous = order
        out[r["run"]] = RunRow(r["run"], z, a, kind, int(r["rc"]), int(r["err_bytes"]))
    return out


def load_states(path: Path) -> dict[str, dict[str, str]]:
    """Parse ``mudirac_states.csv``: printed header energies, by run then orbit, ordered."""
    out: dict[str, dict[str, str]] = {}
    previous: tuple[int, int, int, int, int] | None = None
    for where, r in read_rows(path, STATES_COLUMNS):
        z, a, kind = _nuclide_kind(where, r, PRINTED_KINDS)
        if not _ORBIT.fullmatch(r["state"]):
            raise CellError(f"{where}: state must be an IUPAC orbit, got {r['state']!r}")
        for column in ("n", "l", "s"):
            if not _SIGNED_COUNT.fullmatch(r[column]):
                raise CellError(f"{where}: {column} must be an integer, got {r[column]!r}")
        for column in ("binding_eV", "total_eV"):
            if not _PRINTED.fullmatch(r[column]):
                raise CellError(f"{where}: {column} must be a printed number, got {r[column]!r}")
        if r["state"] in out.get(r["run"], {}):
            raise DuplicateKeyError(f"{where}: duplicate state {r['run']} {r['state']}")
        order = (*_run_key(z, a, kind), *_orbit_order(r["state"]))
        if previous is not None and order < previous:
            raise OrderError(f"{where}: rows are ordered by (Z, A, kind, orbit)")
        previous = order
        out.setdefault(r["run"], {})[r["state"]] = r["binding_eV"]
    return out


def load_lines(path: Path) -> dict[str, dict[str, str]]:
    """Parse ``mudirac_lines.csv``: printed line energies, by run then line, runs ordered."""
    out: dict[str, dict[str, str]] = {}
    previous: tuple[int, int, int] | None = None
    for where, r in read_rows(path, LINES_COLUMNS):
        z, a, kind = _nuclide_kind(where, r, PRINTED_KINDS)
        if not _LINE.fullmatch(r["line"]):
            raise CellError(f"{where}: line must be orbit-orbit, got {r['line']!r}")
        if not _LINE_ENERGY.fullmatch(r["delta_e_eV"]):
            raise CellError(f"{where}: delta_e_eV must be printed with six decimals, got {r['delta_e_eV']!r}")
        if not _UNSIGNED_DECIMAL.fullmatch(r["w12_per_s"]):
            raise CellError(f"{where}: w12_per_s must be a printed decimal, got {r['w12_per_s']!r}")
        if r["line"] in out.get(r["run"], {}):
            raise DuplicateKeyError(f"{where}: duplicate line {r['run']} {r['line']}")
        order = _run_key(z, a, kind)
        if previous is not None and order < previous:
            raise OrderError(f"{where}: rows are ordered by (Z, A, kind)")
        previous = order
        out.setdefault(r["run"], {})[r["line"]] = r["delta_e_eV"]
    return out


def load_nmax(path: Path) -> tuple[NmaxRow, ...]:
    """Parse ``mudirac_nmax_check.csv``: both circular states of every checked shell, per member."""
    out: list[NmaxRow] = []
    previous: tuple[int, int, int, int, int] | None = None
    for where, r in read_rows(path, NMAX_COLUMNS):
        z, a, n = integer(r["Z"], where, "Z"), integer(r["A"], where, "A"), integer(r["n"], where, "n")
        if not IDEAL_FROM <= n <= N_MAX or r["state"] not in circular_orbits(n):
            raise CellError(f"{where}: shell {n} state {r['state']!r} is not a checked circular state")
        for column in ("full_binding_eV", "ideal_binding_eV"):
            if r[column] and not _PRINTED.fullmatch(r[column]):
                raise CellError(f"{where}: {column} must be empty or a printed number, got {r[column]!r}")
        order = (z, a, n, *_orbit_order(r["state"]))
        if previous is not None and order == previous:
            raise DuplicateKeyError(f"{where}: duplicate row ({z}, {a}, {r['state']})")
        if previous is not None and order < previous:
            raise OrderError(f"{where}: rows are ordered by (Z, A, n, orbit)")
        previous = order
        out.append(NmaxRow(z, a, n, r["state"], r["full_binding_eV"], r["ideal_binding_eV"]))
    return tuple(out)


def expected_runs(
    inputs: tuple[InputRow, ...], validation: set[tuple[int, int]]
) -> list[tuple[str, int, int, str]]:
    """Every committed run the input set implies, ``(run, Z, A, kind)``, in the run table's order."""
    return [(run_id(row, kind), row.z, row.a, kind) for row in inputs for kind in run_kinds(row, validation)]


def load_outputs(root: Path) -> Outputs:
    """Every committed output CSV, cross-checked against the input set: the run table holds exactly
    the runs the inputs imply, every printed-kind run's states and lines belong to a listed run, and
    the hydrogen-like comparison covers every member's checked shells."""
    root = Path(root)
    inputs = load_inputs(root / INPUTS_RELPATH)
    cells, _origins = load_validation(root / CELLS_RELPATH, root / ORIGIN_RELPATH)
    runs = load_runs(root / RUNS_RELPATH)
    expected = expected_runs(inputs, set(gated_nuclides(cells)))
    listed = [(r.run, r.z, r.a, r.kind) for r in runs.values()]
    if listed != expected:
        missing = sorted(set(expected) - set(listed))[:3]
        extra = sorted(set(listed) - set(expected))[:3]
        raise CellError(f"{Path(RUNS_RELPATH).name} does not list exactly the implied runs: "
                        f"missing {missing}, unexpected {extra}")
    headers = load_states(root / STATES_RELPATH)
    lines = load_lines(root / LINES_RELPATH)
    for name, table in ((STATES_RELPATH, headers), (LINES_RELPATH, lines)):
        stray = sorted(run for run in table if run not in runs or runs[run].kind not in PRINTED_KINDS)
        if stray:
            raise CellError(
                f"{Path(name).name} carries runs the run table does not list as printed: {stray[:3]}"
            )
    nmax = load_nmax(root / NMAX_RELPATH)
    keys = [(r.z, r.a, r.n, r.state) for r in nmax]
    implied = [(row.z, row.a, n, state) for row in inputs for n in range(IDEAL_FROM, N_MAX + 1)
               for state in circular_orbits(n)]
    if keys != implied:
        raise CellError(
            f"{Path(NMAX_RELPATH).name} does not hold both circular states of every checked shell"
        )
    return Outputs(inputs, runs, headers, lines, nmax)


# --------------------------------------------------------------------------------------------
# the derivation of the tables from the printed strings
# --------------------------------------------------------------------------------------------

#: Enough digits that no quotient below is rounded before its final conversion to a float.
_PRECISION = 50


class DerivationError(Mudirac130Error):
    """A run's printed states and lines do not give binding energies."""


def half_unit_6sig(text: str) -> Decimal:
    """Half a unit of the sixth significant digit of a printed state energy."""
    return Decimal(5) * Decimal(10) ** (Decimal(text).adjusted() - 6)


def upper_chain_lines() -> list[str]:
    """The lines from orbit 2l+1 of shell n to orbit 2l+3 of shell n+1, from K1 outward."""
    return [f"{orbit(n, True)}-{orbit(n + 1, True)}" for n in range(1, N_MAX)]


def lower_chain_lines() -> list[str]:
    """The lines from orbit 2l of shell n to orbit 2l+2 of shell n+1, from K1 outward."""
    return [f"{orbit(n, False)}-{orbit(n + 1, False)}" for n in range(1, N_MAX)]


def cross_lines() -> list[str]:
    """The lines from orbit 2l+1 of shell n to orbit 2l+2 of shell n+1, n = 2 .. N_MAX - 1."""
    return [f"{orbit(n, True)}-{orbit(n + 1, False)}" for n in range(2, N_MAX)]


def chain_states() -> list[str]:
    """Every circular orbit of shells 1 .. N_MAX, from K1 outward."""
    return [state for n in range(1, N_MAX + 1) for state in circular_orbits(n)]


def header_bound(header: str, anchor: str) -> Decimal:
    """How far a derived binding energy may lie from its printed header: half a sixth-digit unit of
    the header and of the anchor, plus half a printed unit for each line summed along the chain."""
    return half_unit_6sig(header) + half_unit_6sig(anchor) + 2 * N_MAX * LINE_HALF_UNIT


def derive_bindings(run: str, headers: dict[str, str], lines: dict[str, str]) -> dict[str, Decimal]:
    """Positive binding energies (eV) of every circular orbit of one run, from its printed strings.

    Anchor: orbit 2l+1 of shell N_MAX, B = -(its header). Upper orbits downward by adding each
    printed line; lower orbits outward from K1 by subtracting each. Raises unless every derived
    energy lies within ``header_bound`` of its own header and every cross line closes within the
    rounding of the lines summed around it.
    """
    for state in chain_states():
        if state not in headers:
            raise DerivationError(f"{run}: no printed header for {state}")
    for line in upper_chain_lines() + lower_chain_lines() + cross_lines():
        if line not in lines:
            raise DerivationError(f"{run}: no printed line {line}")
    anchor = orbit(N_MAX, True)
    with localcontext() as context:
        context.prec = _PRECISION
        b: dict[str, Decimal] = {anchor: -Decimal(headers[anchor])}
        for n in range(N_MAX - 1, 0, -1):
            lower, upper = orbit(n, True), orbit(n + 1, True)
            b[lower] = b[upper] + Decimal(lines[f"{lower}-{upper}"])
        for n in range(1, N_MAX):
            lower, upper = orbit(n, False), orbit(n + 1, False)
            b[upper] = b[lower] - Decimal(lines[f"{lower}-{upper}"])
        for state in chain_states():
            gap = abs(b[state] + Decimal(headers[state]))
            if gap > header_bound(headers[state], headers[anchor]):
                raise DerivationError(f"{run}: derived {state} lies {gap} eV from its printed header")
        closure = (2 * N_MAX + 1) * LINE_HALF_UNIT
        for line in cross_lines():
            lower, upper = line.split("-")
            residual = abs(b[lower] - b[upper] - Decimal(lines[line]))
            if residual > closure:
                raise DerivationError(f"{run}: cross line {line} closes to {residual} eV")
    return b


def level_mean(b: dict[str, Decimal], n: int) -> Decimal:
    """The (2j+1)-weighted mean binding energy (eV) of the circular state of shell ``n`` >= 2."""
    ell = n - 1
    with localcontext() as context:
        context.prec = _PRECISION
        return (2 * ell * b[orbit(n, False)] + (2 * ell + 2) * b[orbit(n, True)]) / (4 * ell + 2)


def quantities(b: dict[str, Decimal]) -> tuple[Decimal, tuple[Decimal, ...]]:
    """``(K, (e2 .. e<N_MAX>))`` in keV: the 1s binding energy and each circular level's mean."""
    with localcontext() as context:
        context.prec = _PRECISION
        return b["K1"] / 1000, tuple(level_mean(b, n) / 1000 for n in range(2, N_MAX + 1))


def relative(b: dict[str, Decimal]) -> dict[str, Decimal]:
    """Every state's binding energy (eV) minus the anchor's: what the printed lines alone give, with
    the energy of the outermost circular state held fixed, so the anchor's printed header drops out."""
    anchor = b[orbit(N_MAX, True)]
    with localcontext() as context:
        context.prec = _PRECISION
        return {state: energy - anchor for state, energy in b.items()}


def uncertainties(
    base: dict[str, Decimal], moved: dict[str, Decimal]
) -> tuple[Decimal, tuple[Decimal, ...]]:
    """``(unc, (u2 .. u<N_MAX>))`` in keV: the absolute change of each quantity between the base and
    the moved run, both measured relative to the anchor, and never less than UNC_FLOOR_KEV."""
    k, levels = quantities(relative(base))
    k_moved, levels_moved = quantities(relative(moved))
    with localcontext() as context:
        context.prec = _PRECISION
        return max(abs(k_moved - k), UNC_FLOOR_KEV), tuple(
            max(abs(m - e), UNC_FLOOR_KEV) for m, e in zip(levels_moved, levels, strict=True)
        )


# --------------------------------------------------------------------------------------------
# which nuclides the tables keep
# --------------------------------------------------------------------------------------------


def fermi2_c_threshold(fermi_t: str) -> float:
    """The sphere radius (fm) below which MuDirac's documented default Fermi parameter
    c = sqrt(R^2 - 7/3 (pi t / (4 ln 3))^2) is not a real number."""
    return math.sqrt(7.0 / 3.0) * math.pi * float(fermi_t) / (4.0 * math.log(3.0))


def fermi2_c_from_a(row: InputRow) -> bool:
    return row.a < FERMI2_SQRT_FROM_A


def fermi2_c_not_real(row: InputRow) -> bool:
    return row.a >= FERMI2_SQRT_FROM_A and float(row.radius_fm) < fermi2_c_threshold(row.fermi_t_fm)


def keep_failures(out: Outputs, row: InputRow) -> list[str]:
    """Every clause of the keep rule ``row`` fails; empty when it is kept.

    The base, rsig and hydrogen-like runs exit 0 with an empty error file; for every checked shell
    both circular states of the base run lie within NMAX_BOUND (relative) of the hydrogen-like run's;
    the derivation holds on the base and rsig runs.
    """
    failures = []
    for kind in ("base", "rsig", *IDEAL_KINDS):
        run = out.runs[run_id(row, kind)]
        if not run.clean:
            failures.append(f"{kind} rc={run.rc} err_bytes={run.err_bytes}")
    for check in out.nmax:
        if check.z != row.z or check.a != row.a:
            continue
        if not check.full or not check.ideal:
            failures.append(f"shell {check.n} {check.state} header missing")
            continue
        with localcontext() as context:
            context.prec = _PRECISION
            relative = abs(Decimal(check.full) - Decimal(check.ideal)) / abs(Decimal(check.ideal))
        if relative >= NMAX_BOUND:
            failures.append(f"shell {check.n} {check.state} differs from hydrogen-like by {relative:.6f}")
    for kind in ("base", "rsig"):
        name = run_id(row, kind)
        try:
            derive_bindings(name, out.headers.get(name, {}), out.lines.get(name, {}))
        except DerivationError as error:
            failures.append(str(error))
    return failures


def drop_reasons(out: Outputs) -> dict[tuple[int, int], str]:
    """``{(Z, A): reason}`` of every member dropped. Two classes are computed from the inputs: a member
    whose default Fermi parameter is set by its mass number alone is dropped with DROP_FERMI2_FROM_A
    whether or not it fails the keep rule; a member the keep rule drops has DROP_FERMI2_C when its
    default Fermi parameter is not real at its sphere radius, else the failed clauses."""
    reasons = {}
    for row in out.inputs:
        if fermi2_c_from_a(row):
            reasons[row.nuclide] = DROP_FERMI2_FROM_A
            continue
        failures = keep_failures(out, row)
        if failures:
            reasons[row.nuclide] = DROP_FERMI2_C if fermi2_c_not_real(row) else "; ".join(failures)
    return reasons


# --------------------------------------------------------------------------------------------
# the two tables
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TableRow:
    """One (Z, A) row of both tables; ``carries`` is the isotope an A = 0 row copies (else A)."""

    z: int
    a: int
    carries: int
    k: float
    k_unc: float
    levels: tuple[float, ...]
    level_uncs: tuple[float, ...]


def table_rows(out: Outputs) -> tuple[list[TableRow], dict[tuple[int, int], str]]:
    """The rows both tables carry, ascending by (Z, A), and the dropped members with their reasons.

    Every kept member gives its (Z, A) row; the (Z, 0) row of a Z copies the row of the isotope the
    abundance file names for it, when that isotope is kept.
    """
    dropped = drop_reasons(out)
    isotopes: list[TableRow] = []
    for row in out.inputs:
        if row.nuclide in dropped:
            continue
        base, moved = (derive_bindings(run_id(row, kind), out.headers[run_id(row, kind)],
                                       out.lines[run_id(row, kind)]) for kind in ("base", "rsig"))
        k, levels = quantities(base)
        k_unc, level_uncs = uncertainties(base, moved)
        isotopes.append(TableRow(
            row.z, row.a, row.a, float(k), float(k_unc),
            tuple(float(e) for e in levels),
            tuple(float(u) for u in level_uncs),
        ))
    most: dict[int, int] = {}
    for row in out.inputs:
        if most.setdefault(row.z, row.most_abundant) != row.most_abundant:
            raise CellError(f"Z={row.z} names two most abundant isotopes")
    kept = {(r.z, r.a): r for r in isotopes}
    rows = list(isotopes)
    for z, a in sorted(most.items()):
        if (z, a) in kept:
            r = kept[(z, a)]
            rows.append(TableRow(z, 0, a, r.k, r.k_unc, r.levels, r.level_uncs))
    rows.sort(key=lambda r: (r.z, r.a))
    return rows, dropped


# --------------------------------------------------------------------------------------------
# the comparison with the measured transition energies
# --------------------------------------------------------------------------------------------

VALIDATION_COLUMNS = (
    "source", "Z", "A", "transition", "quantity", "measured_keV", "unc_keV", "unc_label", "tol_keV",
    "model_keV", "residual_keV", "dE_sigma_keV", "dE_1pct_keV", "label", "within", "npol_keV",
    "radius_origin", "gated", "reason", "locator",
)
#: The comparison labels: a row whose model value moves by at least a third of its tolerance when
#: the radius moves (by its uncertainty, or by the SIZE_FLOOR factor) is size-dominated.
SIZE_DOMINATED = "size-dominated"
WEAKLY_SENSITIVE = "weakly sensitive"


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def line_energy_kev(out: Outputs, row: InputRow, kind: str, line: str) -> Decimal:
    """The printed energy of ``line`` in ``row``'s run of ``kind``, in keV. Raises when MuDirac did
    not print that line for that run."""
    run = run_id(row, kind)
    printed = out.lines.get(run, {})
    if line not in printed:
        raise CellError(f"{run} printed no line {line}")
    return Decimal(printed[line]) / 1000


def validation_rows(
    out: Outputs, cells: tuple[Cell, ...], origins: dict[tuple[int, int], RadiusOrigin]
) -> list[dict[str, str]]:
    """One row per transcribed cell, in the transcription's order: the measured value, the tolerance
    TOL_FACTOR * sqrt(sigma**2 + SIGMA_CALC**2), and on a gated row the model value, the residual,
    the model's shift when the radius moves, the label and whether the residual is within tolerance.
    Nothing is altered for lying outside tolerance."""
    inputs = {row.nuclide: row for row in out.inputs}
    rows = []
    for cell in cells:
        sigma = Decimal(cell.unc_kev)
        with localcontext() as context:
            context.prec = _PRECISION
            tol = TOL_FACTOR * (sigma * sigma + Decimal(SIGMA_CALC) ** 2).sqrt()
        row = {
            "source": cell.source, "Z": str(cell.z), "A": str(cell.a), "transition": cell.transition,
            "quantity": cell.quantity, "measured_keV": cell.value_kev, "unc_keV": cell.unc_kev,
            "unc_label": cell.unc_label, "tol_keV": _decimal_text(tol), "model_keV": "",
            "residual_keV": "", "dE_sigma_keV": "", "dE_1pct_keV": "", "label": "", "within": "",
            "npol_keV": cell.npol_kev,
            "radius_origin": origins[cell.nuclide].origin if cell.nuclide in origins else "",
            "gated": "true" if cell.gated else "false", "reason": cell.reason, "locator": cell.locator,
        }
        if cell.gated:
            member = inputs[cell.nuclide]
            model = line_energy_kev(out, member, "base", cell.quantity)
            residual = model - Decimal(cell.value_kev)
            d_sigma = line_energy_kev(out, member, "rsig", cell.quantity) - model
            d_floor = line_energy_kev(out, member, "r101", cell.quantity) - model
            size = max(abs(d_sigma), abs(d_floor)) >= tol / 3
            row.update({
                "model_keV": _decimal_text(model), "residual_keV": _decimal_text(residual),
                "dE_sigma_keV": _decimal_text(d_sigma), "dE_1pct_keV": _decimal_text(d_floor),
                "label": SIZE_DOMINATED if size else WEAKLY_SENSITIVE,
                "within": "true" if abs(residual) <= tol else "false",
            })
        rows.append(row)
    return rows


def render_validation(rows: list[dict[str, str]]) -> bytes:
    """``validation.csv``: LF, ASCII, the header then one row per transcribed cell."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(VALIDATION_COLUMNS)
    for row in rows:
        writer.writerow([row[column] for column in VALIDATION_COLUMNS])
    return buffer.getvalue().encode("ascii")


def build_validation(root: Path) -> bytes:
    """The comparison file from the committed files under ``root`` alone."""
    root = Path(root)
    cells, origins = load_validation(root / CELLS_RELPATH, root / ORIGIN_RELPATH)
    return render_validation(validation_rows(load_outputs(root), cells, origins))


#: The columns of the document's comparison table, each a column of ``validation.csv``.
TABLE_COLUMNS = (
    ("source", "source"), ("Z", "Z"), ("A", "A"), ("transition", "transition"), ("line", "quantity"),
    ("measured (keV)", "measured_keV"), ("sigma (keV)", "unc_keV"), ("model (keV)", "model_keV"),
    ("residual (keV)", "residual_keV"), ("tol (keV)", "tol_keV"), ("label", "label"),
    ("within", "within"), ("NPol (keV)", "npol_keV"),
)


def render_validation_table(rows: list[dict[str, str]]) -> str:
    """The document's comparison table: one Markdown row per gated row of ``validation.csv``, in its
    order, every cell copied from the file."""
    lines = [
        "| " + " | ".join(title for title, _ in TABLE_COLUMNS) + " |",
        "|" + "---|" * len(TABLE_COLUMNS),
    ]
    for row in rows:
        if row["gated"] != "true":
            continue
        cells = [f"`{row[column]}`" if column == "quantity" else row[column] for _, column in TABLE_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
