"""MuDirac 1.3.0, with the patch in ``third_party/mudirac/``, run as a generator whose inputs and printed
outputs are data: the ``mudirac130p1`` profile of the D3 tables.

The D3 energy tables are not transcribed and not computed here. The patched MuDirac 1.3.0 is run outside
this package on inputs this module renders from ``data/g4/d3/mudirac_inputs.csv``; every state header and
every line it prints is committed as a CSV beside them, and this module turns those printed strings
into the two tables by exact decimal arithmetic whose checks raise rather than warn.

Beside the energies, the directory carries the measured energies the tables are compared
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
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

#: The profile the tables carry: MuDirac 1.3.0 with the patch set in ``third_party/mudirac/``, ``p1`` its
#: first published revision.
PROFILE = "mudirac130p1"
SEAM = "d3_transitions"
#: The bibliography key of the generator's reference.
BIBKEY = "Sturniolo2021"
MUDIRAC_VERSION = "1.3.0"
#: The pinned identities of the generator's release and of the three input files the input set is
#: built from: the release archive the patch applies to, the two data files it bundles (by git blob id)
#: and the charge radii table whose uncertainty column gives the radius shift of the `rsig` runs.
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
#: The bibliography keys of the energies the tables are compared with.
VALIDATION_SOURCES = ("Fricke1995", "Saito2025")

D3_RELDIR = "data/g4/d3"
INPUTS_RELPATH = f"{D3_RELDIR}/mudirac_inputs.csv"
RUNS_RELPATH = f"{D3_RELDIR}/mudirac_runs.csv"
STATES_RELPATH = f"{D3_RELDIR}/mudirac_states.csv"
LINES_RELPATH = f"{D3_RELDIR}/mudirac_lines.csv"
CELLS_RELPATH = f"{D3_RELDIR}/validation_cells.csv"
ORIGIN_RELPATH = f"{D3_RELDIR}/validation_radius_origin.csv"
VALIDATION_RELPATH = f"{D3_RELDIR}/validation.csv"
GEANT4_LEVELS_RELPATH = f"{D3_RELDIR}/geant4_cascade_levels.csv"

#: The kinds the renderer writes for each member, in their order: the bundled radius, the rms radius
#: moved by its uncertainty, the rms radius times SIZE_FLOOR (validation nuclides only), and one
#: hydrogen-like run per checked shell. The recipe runs the first as the anchor kind.
IDEAL_KINDS = tuple(f"ideal{n}" for n in range(IDEAL_FROM, N_MAX + 1))
COMMITTED_KINDS = ("base", "rsig", "r101", *IDEAL_KINDS)
#: The kind run beside them to measure the other sign of the radius response: the rms radius moved
#: down by its uncertainty, where `rsig` moves it up.
RESPONSE_KINDS = ("rminus",)

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
# the measured energies
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One printed energy, every cell as its source prints it."""

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


def centroid_nuclides(cells: tuple[Cell, ...]) -> list[tuple[int, int]]:
    """Nuclides with a source-labelled center of gravity."""
    return sorted({cell.nuclide for cell in cells if cell.reason == "centroid"})


def stock_nuclides(cells: tuple[Cell, ...]) -> list[tuple[int, int]]:
    """Gated and source-labelled nuclides eligible for the stock comparison."""
    return sorted(set(gated_nuclides(cells)) | set(centroid_nuclides(cells)))


def doublet_determinations(cells: tuple[Cell, ...]) -> list[tuple[Cell, Cell]]:
    """Complete gated doublets paired by their source, nuclide and printed locator."""
    groups: dict[tuple[str, int, int, str], list[Cell]] = {}
    for cell in cells:
        if cell.gated and cell.quantity in ("K1-L2", "K1-L3"):
            groups.setdefault((cell.source, cell.z, cell.a, cell.locator), []).append(cell)
    return [(l2, l3) for group in groups.values()
            if len(group) == 2
            for l2, l3 in [(next((c for c in group if c.quantity == "K1-L2"), None),
                            next((c for c in group if c.quantity == "K1-L3"), None))]
            if l2 is not None and l3 is not None]


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


def run_radius(row: InputRow, kind: str) -> str:
    """The radius line's value for ``kind``."""
    rms = float(row.radius_fm) / SPHERE_FACTOR
    if kind == "rsig":
        return repr((rms + float(row.sigma_rms_fm)) * SPHERE_FACTOR)
    if kind == "rminus":
        return repr((rms - float(row.sigma_rms_fm)) * SPHERE_FACTOR)
    if kind == "r101":
        return repr(rms * float(SIZE_FLOOR) * SPHERE_FACTOR)
    return row.radius_fm


def render_input(row: InputRow, kind: str, extra: list[str]) -> str:
    """The MuDirac input file of ``row`` under ``kind``: the fixed settings in their order, the kind's
    radius, the circular chain plus ``extra`` (none on a hydrogen-like run), and on a hydrogen-like
    run the shell from which the atom is hydrogen-like."""
    if kind not in COMMITTED_KINDS + RESPONSE_KINDS:
        raise ValueError(f"unknown run kind {kind!r}")
    lines = [
        f"element: {row.symbol}",
        f"isotope: {row.a}",
        "nuclear_model: FERMI2",
        "uehling_correction: TRUE",
        "reduced_mass: TRUE",
        "optimise_fermi_parameters: FALSE",
    ]
    lines.append(f"radius: {run_radius(row, kind)}")
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
# the committed outputs of the recipe runs
# --------------------------------------------------------------------------------------------

#: One row per run of every recipe batch, in the batches' order; what the collector recorded, the
#: domain label of a run whose c is solved from its intended rms radius, and the printed total
#: energy of its outermost circular state, from which the float resolution of its energies follows.
RUNS_COLUMNS = (
    "run", "Z", "A", "kind", "stage", "uehling", "variant", "density_domain", "rc", "err_bytes", "wall_s",
    "input_sha256", "r15_total_eV",
)
#: The printed state headers and printed lines of every run, as printed, grouped by run in the run
#: table's order: what the tables and their checks read (``committed_view``). The transition rates
#: MuDirac prints beside the lines are not committed: no table reads them.
STATES_COLUMNS = ("run", "state", "n", "l", "s", "binding_eV")
LINES_COLUMNS = ("run", "line", "delta_e_eV")
#: The circular states of the checked shells, the only printed values of a hydrogen-like run read.
CHECKED_STATES = tuple(state for n in range(IDEAL_FROM, N_MAX + 1) for state in circular_orbits(n))


def committed_view(kind: str, printed_lines: Sequence[str]) -> tuple[list[str], set[str]]:
    """``(lines, states)`` of a complete run that are committed: of a hydrogen-like run, no line and the
    circular states of the checked shells (the keep rule's comparison reads nothing else); of every other
    run, every printed line and the states those lines join."""
    if kind in IDEAL_KINDS:
        return [], set(CHECKED_STATES)
    return list(printed_lines), {state for line in printed_lines for state in line.split("-")}

_ORBIT = re.compile(r"[K-Z][0-9]+")
_COUNT = re.compile(r"0|[1-9][0-9]*")
_SIGNED_COUNT = re.compile(r"0|-?[1-9][0-9]*")
#: A state energy as MuDirac prints it, in either %g form.
_PRINTED = re.compile(r"-?[0-9]+(\.[0-9]+)?(e[+-][0-9]+)?")
_SHA256_TEXT = re.compile(r"[0-9a-f]{64}")
_WALL = re.compile(r"[0-9]+\.[0-9]{3}")
#: The decimals of a printed line of a run of the shipped stage (the v1.3.0 input) and of every other run.
SHIPPED_XR_PRECISION = 6
#: Half a unit of the last decimal a line of the shipped stage is printed with, in eV.
LINE_HALF_UNIT = Decimal("0.0000005")


@dataclass(frozen=True)
class RunRow:
    run: str
    z: int
    a: int
    kind: str
    stage: str
    uehling: bool
    variant: str
    density_domain: str
    rc: int
    err_bytes: int
    wall_s: str
    input_sha256: str
    r15_total: str

    @property
    def clean(self) -> bool:
        return self.rc == 0 and self.err_bytes == 0


@dataclass(frozen=True, eq=False)
class Outputs:
    """The committed run table, printed state headers and printed lines, cross-checked against the
    runs the input set implies. Compared by identity, so the derivations below are cached per instance."""

    inputs: tuple[InputRow, ...]
    cells: tuple[Cell, ...]
    runs: dict[str, RunRow]
    #: ``{run: {orbit: printed binding energy E}}``.
    headers: dict[str, dict[str, str]]
    #: ``{run: {line: printed level difference}}``.
    lines: dict[str, dict[str, str]]


def _orbit_order(state: str) -> tuple[int, int]:
    return (ord(state[0]), int(state[1:]))


def _grouped(path: Path, columns: tuple[str, ...], order: dict[str, int], check) -> dict[str, dict[str, str]]:
    """Rows of a states or lines file grouped by run, each run's rows contiguous and the runs in the
    run table's order; ``check(where, record)`` returns ``(key, value)`` for one row."""
    out: dict[str, dict[str, str]] = {}
    previous = -1
    for where, r in read_rows(path, columns):
        run = r["run"]
        if run not in order:
            raise CellError(f"{where}: run {run!r} is not in {Path(RUNS_RELPATH).name}")
        position = order[run]
        if position < previous or (position != previous and run in out):
            raise OrderError(f"{where}: rows are grouped by run in the run table's order")
        previous = position
        key, value = check(where, r)
        if key in out.setdefault(run, {}):
            raise DuplicateKeyError(f"{where}: duplicate {key} of {run}")
        out[run][key] = value
    return out


def load_outputs(root: Path) -> Outputs:
    """Every committed output CSV, cross-checked against the input set.

    The run table lists exactly the runs ``recipe_runs`` implies for every batch, in order, each with
    the digest of the input the renderer makes now, its implied kind, stage, switch, variant and
    domain label, a clean exit and an empty ``.err``. Each run's committed lines and state headers are
    exactly its ``committed_view`` of the lines its input's ``xr_lines`` implies, the lines printed with
    its input's decimals and every header with (n, l, s) matching its label.
    """
    root = Path(root)
    inputs = load_inputs(root / INPUTS_RELPATH)
    cells, _origins = load_validation(root / CELLS_RELPATH, root / ORIGIN_RELPATH)
    expected = [r for batch in RECIPE_BATCHES for r in recipe_runs(inputs, cells, batch)]
    listed = read_rows(root / RUNS_RELPATH, RUNS_COLUMNS)
    if len(listed) != len(expected):
        raise CellError(
            f"{Path(RUNS_RELPATH).name} lists {len(listed)} runs; the input set implies {len(expected)}"
        )
    runs: dict[str, RunRow] = {}
    texts: dict[str, str] = {}
    for (where, r), want in zip(listed, expected, strict=True):
        if r["run"] != want.run:
            raise CellError(f"{where}: run {r['run']!r} where the input set implies {want.run!r}")
        domain = density_domain(want.row, want.kind) if want.stage in ("rms", "mass", "recipe") else ""
        implied = {"Z": str(want.row.z), "A": str(want.row.a), "kind": want.kind, "stage": want.stage,
                   "uehling": "on" if want.uehling else "off", "variant": want.variant,
                   "density_domain": domain,
                   "input_sha256": hashlib.sha256(want.text.encode("ascii")).hexdigest()}
        for column, value in implied.items():
            if r[column] != value:
                raise CellError(f"{where}: {column} {r[column]!r}, implied {value!r}")
        if not _SIGNED_COUNT.fullmatch(r["rc"]) or not _COUNT.fullmatch(r["err_bytes"]):
            raise CellError(
                f"{where}: rc and err_bytes must be integers, got {r['rc']!r}, {r['err_bytes']!r}"
            )
        if not _WALL.fullmatch(r["wall_s"]) or not _PRINTED.fullmatch(r["r15_total_eV"]):
            raise CellError(f"{where}: wall_s and r15_total_eV must be printed numbers")
        row = RunRow(r["run"], want.row.z, want.row.a, want.kind, want.stage, want.uehling, want.variant,
                     domain,
                     int(r["rc"]), int(r["err_bytes"]), r["wall_s"], r["input_sha256"], r["r15_total_eV"])
        if not row.clean:
            raise CellError(
                f"{where}: rc={row.rc} err_bytes={row.err_bytes}; the collector admits clean runs only"
            )
        runs[row.run] = row
        texts[row.run] = want.text
    order = {run: i for i, run in enumerate(runs)}

    def state_row(where: str, r: dict[str, str]) -> tuple[str, str]:
        if not _ORBIT.fullmatch(r["state"]):
            raise CellError(f"{where}: state must be an IUPAC orbit, got {r['state']!r}")
        n, ell, two_j = iupac_state(r["state"])
        s_expected = 1 if (ell == 0 or two_j == 2 * ell + 1) else 0
        if (r["n"], r["l"], r["s"]) != (str(n), str(ell), str(s_expected)):
            raise CellError(f"{where}: n, l, s {r['n']}, {r['l']}, {r['s']} do not match {r['state']}")
        if not _PRINTED.fullmatch(r["binding_eV"]):
            raise CellError(f"{where}: binding_eV must be a printed number, got {r['binding_eV']!r}")
        return r["state"], r["binding_eV"]

    def line_row(where: str, r: dict[str, str]) -> tuple[str, str]:
        if not _LINE.fullmatch(r["line"]):
            raise CellError(f"{where}: line must be orbit-orbit, got {r['line']!r}")
        decimals = SHIPPED_XR_PRECISION if runs[r["run"]].stage == "shipped" else RECIPE_XR_PRECISION
        if not re.fullmatch(rf"[0-9]+\.[0-9]{{{decimals}}}", r["delta_e_eV"]):
            raise CellError(f"{where}: delta_e_eV must carry {decimals} decimals, got {r['delta_e_eV']!r}")
        return r["line"], r["delta_e_eV"]

    headers = _grouped(root / STATES_RELPATH, STATES_COLUMNS, order, state_row)
    lines = _grouped(root / LINES_RELPATH, LINES_COLUMNS, order, line_row)
    for run in runs:
        want_lines, want_states = committed_view(
            runs[run].kind, expected_printed_lines(input_keywords(texts[run])["xr_lines"]))
        if sorted(lines.get(run, {})) != sorted(want_lines):
            raise CellError(f"{run}: the committed lines are not the lines its input implies")
        if set(headers.get(run, {})) != want_states:
            raise CellError(f"{run}: the committed states are not the states its committed view names")
        if list(headers[run]) != sorted(headers[run], key=_orbit_order):
            raise OrderError(f"{run}: state rows are in orbit order")
    return Outputs(inputs, cells, runs, headers, lines)


# --------------------------------------------------------------------------------------------
# the derivation of binding energies from the printed strings
# --------------------------------------------------------------------------------------------

#: Enough digits that no quotient below is rounded before its final conversion to a float.
_PRECISION = 50


class DerivationError(Mudirac130Error):
    """A run's printed states and lines do not give binding energies."""


def half_unit_6sig(text: str) -> Decimal:
    """Half a unit of the sixth significant digit of a printed state energy (the shipped stage)."""
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
    """Shipped stage: how far a derived binding energy may lie from its printed header -- half a
    sixth-digit unit of the header and of the anchor, plus half a printed unit per line summed."""
    return half_unit_6sig(header) + half_unit_6sig(anchor) + 2 * N_MAX * LINE_HALF_UNIT


def derive_bindings(run: str, headers: dict[str, str], lines: dict[str, str]) -> dict[str, Decimal]:
    """The shipped stage's derivation (the v1.3.0 tables'): positive binding energies (eV) of every
    circular orbit of one run of the shipped input, anchored on the printed six-digit header of orbit
    2l+1 of shell N_MAX; upper orbits downward by adding each printed line, lower orbits outward from
    K1 by subtracting each. Raises unless every derived energy lies within ``header_bound`` of its own
    header and every cross line closes within the rounding of the lines summed around it."""
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
    """Every state's binding energy (eV) minus the outermost circular state's: what the printed lines
    alone give, so the anchor drops out."""
    anchor = b[orbit(N_MAX, True)]
    with localcontext() as context:
        context.prec = _PRECISION
        return {state: energy - anchor for state, energy in b.items()}


def uncertainties(
    base: dict[str, Decimal], moved: dict[str, Decimal], floor_kev: Decimal
) -> tuple[Decimal, tuple[Decimal, ...]]:
    """``(unc, (u2 .. u<N_MAX>))`` in keV: the absolute change of each quantity between the base and
    the moved run, both measured relative to the outermost circular state, and never less than
    ``floor_kev``."""
    k, levels = quantities(relative(base))
    k_moved, levels_moved = quantities(relative(moved))
    with localcontext() as context:
        context.prec = _PRECISION
        return max(abs(k_moved - k), floor_kev), tuple(
            max(abs(m - e), floor_kev) for m, e in zip(levels_moved, levels, strict=True)
        )


def eh_of(stage: str) -> float:
    """The Hartree energy in eV a run's printed energies were converted with."""
    return E_H_EV_2022 if stage == "recipe" else E_H_EV_MUDIRAC


_BINDINGS: weakref.WeakKeyDictionary[Outputs, dict[str, dict[str, Decimal]]] = weakref.WeakKeyDictionary()
_TABLES: weakref.WeakKeyDictionary[Outputs, tuple[list[TableRow], dict[tuple[int, int], str]]] = (
    weakref.WeakKeyDictionary()
)


def run_bindings(out: Outputs, run: str) -> dict[str, Decimal]:
    """The derived binding energies (eV) of every circular orbit of one committed run: the shipped
    stage by the v1.3.0 derivation, every other run by ``derive_recipe_bindings``."""
    cache = _BINDINGS.setdefault(out, {})
    if run not in cache:
        r = out.runs[run]
        if r.stage == "shipped":
            cache[run] = derive_bindings(run, out.headers[run], out.lines[run])
        else:
            b = derive_recipe_bindings(run, r.kind, out.headers[run], out.lines[run], r.r15_total,
                                       eh_of(r.stage))
            cache[run] = {state: value for state, (value, _rounding) in b.items()}
    return cache[run]


def line_rounding_ev(out: Outputs, run: str) -> Decimal:
    """What one printed line of a recipe run carries: half a unit of its last decimal plus the float
    resolution of the total energies it is the difference of."""
    r = out.runs[run]
    with localcontext() as context:
        context.prec = _PRECISION
        half_unit = Decimal(5) * Decimal(10) ** (-RECIPE_XR_PRECISION - 1)
        return half_unit + float_floor_eV(r.r15_total, eh_of(r.stage))


def unc_floor_kev(out: Outputs, runs: Sequence[str]) -> Decimal:
    """The least uncertainty a ``unc`` or ``u<n>`` cell carries, in keV: a quantity measured with the
    outermost circular state held fixed sums at most twice (N_MAX - 1) printed lines, each within one
    line's rounding of the value MuDirac computed; a (2j+1) mean is bounded the same way; and the
    change is taken between two runs, each rounded independently."""
    with localcontext() as context:
        context.prec = _PRECISION
        return 4 * (N_MAX - 1) * max(line_rounding_ev(out, run) for run in runs) / 1000


# --------------------------------------------------------------------------------------------
# which nuclides the tables keep
# --------------------------------------------------------------------------------------------


def keep_failures(out: Outputs, row: InputRow) -> list[str]:
    """Every clause of the keep rule ``row`` fails; empty when it is kept.

    The anchor and rsig runs' intended rms radii lie inside the 2pF domain; for every checked shell,
    on both Uehling switches, both circular states of the anchor run lie within NMAX_BOUND (relative)
    of the hydrogen-like run's; the derivation holds on the anchor and rsig runs, on both switches.
    The collector admits only complete runs, and ``load_outputs`` refuses any other.
    """
    failures = []
    for kind in ("anchor", "rsig"):
        domain = density_domain(row, kind)
        if domain != INSIDE_2PF:
            failures.append(
                f"{kind} density_domain {domain}: intended rms {recipe_rms_fm(row, kind)!r} fm at or below "
                "the "
                f"2pF floor {fermi2_rms_floor_fm(row.fermi_t_fm)!r} fm at t = {row.fermi_t_fm} fm"
            )
    if failures:
        return failures
    for uehling in (True, False):
        anchor = recipe_run_id(row, "anchor", uehling)
        for n in range(IDEAL_FROM, N_MAX + 1):
            ideal = recipe_run_id(row, f"ideal{n}", uehling)
            for state in circular_orbits(n):
                full, hydrogenic = Decimal(out.headers[anchor][state]), Decimal(out.headers[ideal][state])
                with localcontext() as context:
                    context.prec = _PRECISION
                    rel = abs(full - hydrogenic) / abs(hydrogenic)
                if rel >= NMAX_BOUND:
                    failures.append(f"{ideal} shell {n} {state} differs from hydrogen-like by {rel:.6f}")
        for kind in ("anchor", "rsig"):
            try:
                run_bindings(out, recipe_run_id(row, kind, uehling))
            except DerivationError as error:
                failures.append(str(error))
    return failures


def drop_reasons(out: Outputs) -> dict[tuple[int, int], str]:
    """``{(Z, A): reason}`` of every member the keep rule drops: its failed clauses."""
    reasons = {}
    for row in out.inputs:
        failures = keep_failures(out, row)
        if failures:
            reasons[row.nuclide] = "; ".join(failures)
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

    Every kept member gives its (Z, A) row from its anchor run with the Uehling potential: K and the
    level means of ``quantities``, and as uncertainties the change of each in its rsig run, floored at
    ``unc_floor_kev``. The (Z, 0) row of a Z copies the row of the isotope the abundance file names
    for it, when that isotope is kept.
    """
    if out in _TABLES:
        cached_rows, cached_dropped = _TABLES[out]
        return list(cached_rows), dict(cached_dropped)
    dropped = drop_reasons(out)
    isotopes: list[TableRow] = []
    for row in out.inputs:
        if row.nuclide in dropped:
            continue
        base_run, moved_run = recipe_run_id(row, "anchor", True), recipe_run_id(row, "rsig", True)
        base, moved = run_bindings(out, base_run), run_bindings(out, moved_run)
        k, levels = quantities(base)
        k_unc, level_uncs = uncertainties(base, moved, unc_floor_kev(out, (base_run, moved_run)))
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
    _TABLES[out] = (list(rows), dict(dropped))
    return rows, dropped


def kept_members(out: Outputs) -> list[InputRow]:
    """The members the tables keep -- the A > 0 rows of ``table_rows`` -- as input rows, in order."""
    rows, _dropped = table_rows(out)
    inputs = {row.nuclide: row for row in out.inputs}
    return [inputs[(r.z, r.a)] for r in rows if r.a > 0]


# --------------------------------------------------------------------------------------------
# the level energies Geant4's own muonic cascade uses, kept for context beside the comparison
# --------------------------------------------------------------------------------------------

GEANT4_LEVELS_COLUMNS = ("Z", "A", "n", "level_MeV")
#: The levels `G4EmCaptureCascade` carries, `fLevelEnergy[0]` through `fLevelEnergy[13]`; a row's `n`
#: is the index plus one, the shell whose energy the cascade holds there.
CASCADE_LEVELS = 14
#: A positive normal double as C's ``%a`` prints it.
_HEXFLOAT = re.compile(r"0x1(?:\.[0-9a-f]+)?p[+-][0-9]+")


def render_geant4_levels(
    harvest_text: str, nuclides: list[tuple[int, int]], optional: Sequence[tuple[int, int]] = (),
) -> bytes:
    """``geant4_cascade_levels.csv`` from the output of ``cpp/tools/harvest_d3.cc``: for each of
    ``nuclides`` and present ``optional`` keys, ascending, the CASCADE_LEVELS level energies its ``C``
    line prints, each the ``%a`` token verbatim, in MeV. Every other line is passed over. Required
    keys need a ``C`` line; repeated lines and malformed levels are refused."""
    wanted = set(nuclides) | set(optional)
    found: dict[tuple[int, int], list[str]] = {}
    for number, line in enumerate(harvest_text.split("\n"), start=1):
        fields = line.split(" ")
        if fields[0] != "C":
            continue
        where = f"harvest line {number}"
        key = (integer(fields[1], where, "Z"), integer(fields[2], where, "A"))
        if key not in wanted:
            continue
        if key in found:
            raise DuplicateKeyError(f"{where}: a second C line for (Z, A) = {key}")
        levels = fields[3:3 + CASCADE_LEVELS]
        if len(levels) != CASCADE_LEVELS or not all(_HEXFLOAT.fullmatch(token) for token in levels):
            raise CellError(
                f"{where}: {CASCADE_LEVELS} level energies printed with %a expected after Z and A"
            )
        found[key] = levels
    missing = sorted(set(nuclides) - set(found))
    if missing:
        raise CellError(f"the harvest has no C line for {missing}")
    rows = [",".join(GEANT4_LEVELS_COLUMNS)]
    for z, a in sorted(found):
        rows += [f"{z},{a},{n},{token}" for n, token in enumerate(found[(z, a)], start=1)]
    return ("\n".join(rows) + "\n").encode("ascii")


def load_geant4_levels(path: Path) -> dict[tuple[int, int], tuple[float, ...]]:
    """Parse ``geant4_cascade_levels.csv``: ``{(Z, A): (level 1, ..., level CASCADE_LEVELS)}`` in MeV,
    each read exactly from its ``%a`` token. Rows ascend by (Z, A, n), and n runs 1 through
    CASCADE_LEVELS for every nuclide."""
    out: dict[tuple[int, int], list[float]] = {}
    previous: tuple[int, int, int] | None = None
    for where, r in read_rows(path, GEANT4_LEVELS_COLUMNS):
        z, a, n = integer(r["Z"], where, "Z"), integer(r["A"], where, "A"), integer(r["n"], where, "n")
        if not _HEXFLOAT.fullmatch(r["level_MeV"]):
            raise CellError(
                f"{where}: level_MeV must be a positive double printed with %a, got {r['level_MeV']!r}"
            )
        key = (z, a, n)
        if previous is not None and key == previous:
            raise DuplicateKeyError(f"{where}: duplicate row for (Z, A, n) = {key}")
        if previous is not None and key < previous:
            raise OrderError(f"{where}: rows are ordered by (Z, A, n)")
        levels = out.setdefault((z, a), [])
        if n != len(levels) + 1 or n > CASCADE_LEVELS:
            raise CellError(f"{where}: n runs 1 through {CASCADE_LEVELS} for every (Z, A), got n = {n}")
        levels.append(float.fromhex(r["level_MeV"]))
        previous = key
    short = sorted(key for key, levels in out.items() if len(levels) != CASCADE_LEVELS)
    if short:
        raise CellError(
            f"{Path(path).name}: n runs 1 through {CASCADE_LEVELS} for every (Z, A); {short} stop short"
        )
    return {key: tuple(levels) for key, levels in out.items()}


def line_shells(line: str) -> tuple[int, int]:
    """The (lower, upper) shells of a line in MuDirac's IUPAC spelling: K is shell 1, L shell 2, ..."""
    lower, upper = (ord(orbit[0]) - ord("K") + 1 for orbit in line.split("-"))
    if not lower < upper:
        raise CellError(f"line {line!r} does not go from a lower to a higher shell")
    return lower, upper


def shell_difference_kev(levels: tuple[float, ...], lower: int, upper: int) -> Decimal:
    """The photon the unpatched cascade emits between the two shells: the difference of its two
    level energies taken in doubles, in keV at nine decimals."""
    if not lower < upper:
        raise CellError(f"shells {lower} and {upper} do not go from a lower to a higher shell")
    photon = levels[lower - 1] - levels[upper - 1]
    with localcontext() as context:
        context.prec = 1000
        return (Decimal(photon) * 1000).quantize(Decimal("1e-9"))


def geant4_line_kev(levels: tuple[float, ...], line: str) -> Decimal:
    """The photon Geant4's cascade emits between the two shells of ``line`` -- the difference of its
    two level energies, taken in doubles as the cascade takes it -- in keV, at nine decimals."""
    return shell_difference_kev(levels, *line_shells(line))


# --------------------------------------------------------------------------------------------
# the comparison with the measured energies
# --------------------------------------------------------------------------------------------

VALIDATION_COLUMNS = (
    "source", "Z", "A", "transition", "quantity", "measured_keV", "unc_keV", "unc_label", "tol_keV",
    "model_keV", "residual_keV", "dE_sigma_keV", "dE_1pct_keV", "label", "within", "npol_keV",
    "radius_origin", "gated", "reason", "locator", "geant4_keV", "geant4_residual_keV",
)
#: The comparison labels: a row whose model value moves by at least a third of its tolerance when
#: the radius moves (by its uncertainty, or by the SIZE_FLOOR factor) is size-dominated.
SIZE_DOMINATED = "size-dominated"
WEAKLY_SENSITIVE = "weakly sensitive"


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def line_energy_kev(out: Outputs, row: InputRow, kind: str, line: str) -> Decimal:
    """The printed energy of ``line`` in ``row``'s production run of ``kind`` with the Uehling potential,
    in keV. Raises when MuDirac did not print that line for that run."""
    run = recipe_run_id(row, kind, True)
    printed = out.lines.get(run, {})
    if line not in printed:
        raise CellError(f"{run} printed no line {line}")
    return Decimal(printed[line]) / 1000


def validation_rows(
    out: Outputs, cells: tuple[Cell, ...], origins: dict[tuple[int, int], RadiusOrigin],
    geant4_levels: dict[tuple[int, int], tuple[float, ...]],
) -> list[dict[str, str]]:
    """One row per transcribed cell, in the transcription's order: the measured value, the tolerance
    TOL_FACTOR * sqrt(sigma**2 + SIGMA_CALC**2), and on a gated row the model value, the residual,
    the model's shift when the radius moves, the label and whether the residual is within tolerance.
    Nothing is altered for lying outside tolerance. For context, a gated row also carries the energy
    Geant4's own cascade emits between the line's two shells and its difference from the measured
    value; the label and the tolerance test use MuDirac's residual alone."""
    gated = set(gated_nuclides(cells))
    if not gated <= set(geant4_levels) or not set(geant4_levels) <= set(stock_nuclides(cells)):
        raise CellError(
            f"{Path(GEANT4_LEVELS_RELPATH).name} names {sorted(geant4_levels)}, "
            f"the gated nuclides are {sorted(gated)} and stock nuclides are {stock_nuclides(cells)}"
        )
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
            "geant4_keV": "", "geant4_residual_keV": "",
        }
        if cell.gated:
            member = inputs[cell.nuclide]
            model = line_energy_kev(out, member, "anchor", cell.quantity)
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
            geant4 = geant4_line_kev(geant4_levels[cell.nuclide], cell.quantity)
            row.update({
                "geant4_keV": _decimal_text(geant4),
                "geant4_residual_keV": _decimal_text(geant4 - Decimal(cell.value_kev)),
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
    geant4_levels = load_geant4_levels(root / GEANT4_LEVELS_RELPATH)
    return render_validation(validation_rows(load_outputs(root), cells, origins, geant4_levels))


#: The columns of the document's comparison table, each a column of ``validation.csv``.
TABLE_COLUMNS = (
    ("source", "source"), ("Z", "Z"), ("A", "A"), ("transition", "transition"), ("line", "quantity"),
    ("measured (keV)", "measured_keV"), ("sigma (keV)", "unc_keV"), ("model (keV)", "model_keV"),
    ("residual (keV)", "residual_keV"), ("Geant4 residual (keV)", "geant4_residual_keV"),
    ("tol (keV)", "tol_keV"), ("label", "label"),
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


# --------------------------------------------------------------------------------------------
# the patched recipe: MuDirac 1.3.0 with the patch in third_party/mudirac/
# --------------------------------------------------------------------------------------------

#: The profile the patched recipe's tables carry: MuDirac 1.3.0 with the patch set in
#: ``third_party/mudirac/``, ``p1`` its first published revision.
RECIPE_PROFILE = "mudirac130p1"
PATCH_RELPATH = "third_party/mudirac/mudirac-1.3.0-p1.patch"
#: Decimals of every printed line, digits of every printed state energy, and significant digits
#: of every state file's energy header in a recipe run.
RECIPE_XR_PRECISION = 12
RECIPE_STATE_PRECISION = 12
RECIPE_HEADER_PRECISION = 17
#: The Hartree energy in eV under ``ev_codata2022`` (CODATA 2022) and MuDirac 1.3.0's own
#: (``lib/constants.hpp``: ``eV = 1.0 / 27.211385``).
E_H_EV_2022 = 27.211386245981
E_H_EV_MUDIRAC = 27.211385
#: The digits of the Riemann zeta function at 3 and 5 that fix the rms radius of the 2pF density
#: with c = 0 (``fermi2_rms_floor_fm``).
_ZETA3 = 1.2020569031595942853997381615114
_ZETA5 = 1.0369277551433699263313654864570


@dataclass(frozen=True)
class Settings:
    """The numerical settings of a recipe run: ``loggrid_step = 0.005 / 2**k``, ``uehling_steps``
    ``n_u``, ``uehling_shell_nodes`` ``n_s`` and ``energy_tol = 1e-<t>``."""

    k: int
    n_u: int
    n_s: int
    t: int

    def keywords(self) -> list[tuple[str, str]]:
        return [("loggrid_step", repr(0.005 / 2 ** self.k)), ("energy_tol", repr(float(f"1e-{self.t}"))),
                ("uehling_steps", str(self.n_u)), ("uehling_shell_nodes", str(self.n_s))]


#: The settings of every production run.
RECIPE_SETTINGS = Settings(k=3, n_u=60, n_s=12, t=7)
#: The settings of the `numerics` stage: the uniform-sphere Uehling kernel by Gauss-Legendre in theta
#: at 64 nodes and the grid and tolerance of the recipe (``uehling_shell_nodes`` has no effect there).
NUMERICS_STAGE_SETTINGS = Settings(k=3, n_u=64, n_s=12, t=7)
#: One step finer in each setting, one at a time, beside every production anchor run; the Uehling-free
#: twin takes only the two that act without the Uehling potential.
REFINEMENTS = (
    ("k4", Settings(k=4, n_u=60, n_s=12, t=7)),
    ("tol8", Settings(k=3, n_u=60, n_s=12, t=8)),
    ("nu90", Settings(k=3, n_u=90, n_s=12, t=7)),
    ("ns16", Settings(k=3, n_u=60, n_s=16, t=7)),
)
NOU_REFINEMENTS = ("k4", "tol8")
#: The stages from the shipped input to the recipe, in order: the v1.3.0 input exactly; the numerics
#: repair (Gauss-Legendre sphere Uehling kernel, Hermite 2pF Coulomb read-back, the hydrogen-like anchor,
#: 12-decimal prints); the one 2pF density for the Coulomb and Uehling potentials together with the
#: trapezoidal Uehling quadrature; the 2pF c solved from the intended rms radius; the nuclear mass; and
#: (the production stage `recipe`) the CODATA 2022 Hartree energy.
STAGES = ("shipped", "numerics", "density", "rms", "mass")
STAGE_KEYWORDS: dict[str, tuple[tuple[str, str], ...]] = {
    "numerics": (("uehling_gauss", "TRUE"),),
    "density": (("uehling_density", "fermi2"),),
    "rms": (("uehling_density", "fermi2"), ("fermi2_rms_exact", "TRUE")),
    "mass": (("uehling_density", "fermi2"), ("fermi2_rms_exact", "TRUE"), ("nuclear_mass", "TRUE")),
    "recipe": (("uehling_density", "fermi2"), ("fermi2_rms_exact", "TRUE"), ("nuclear_mass", "TRUE"),
               ("ev_codata2022", "TRUE")),
}
#: The production kinds: the anchor kind (the bundled radius, the chain extended by one line to the
#: hydrogen-like state of shell N_MAX + 1) in place of `base`, then the committed kinds.
RECIPE_KINDS = ("anchor", "rsig", "r101", *IDEAL_KINDS)
RECIPE_BATCHES = ("prod", "refine", "rminus", "stages")
INSIDE_2PF = "INSIDE"


def anchor_line() -> str:
    """The line the anchor kind adds: orbit 2l+1 of shell N_MAX to that of shell N_MAX + 1."""
    return f"{orbit(N_MAX, True)}-{orbit(N_MAX + 1, True)}"


def recipe_rms_fm(row: InputRow, kind: str) -> float:
    """The intended rms charge radius (fm) of ``kind``, as the float the radius line is made from."""
    rms = float(row.radius_fm) / SPHERE_FACTOR
    if kind == "rsig":
        return rms + float(row.sigma_rms_fm)
    if kind == "rminus":
        return rms - float(row.sigma_rms_fm)
    if kind == "r101":
        return rms * float(SIZE_FLOOR)
    return rms


def fermi2_rms_floor_fm(fermi_t: str) -> float:
    """The rms radius (fm) of the 2pF density with c = 0 at thickness ``fermi_t``:
    a sqrt(12 eta(5)/eta(3)), a = t/(4 ln 3), eta(s) = (1 - 2^(1-s)) zeta(s). No c >= 0 gives a
    smaller one, so an intended rms at or below it is outside the 2pF domain."""
    a = float(fermi_t) / (4.0 * math.log(3.0))
    eta3, eta5 = 0.75 * _ZETA3, (15.0 / 16.0) * _ZETA5
    return a * math.sqrt(12.0 * eta5 / eta3)


def density_domain(row: InputRow, kind: str) -> str:
    """``INSIDE`` when the kind's intended rms radius exceeds the 2pF floor at the row's thickness,
    else the label the patched binary gives (``OUTSIDE_2PF_T<t>``) when it keeps MuDirac's own c."""
    if recipe_rms_fm(row, kind) > fermi2_rms_floor_fm(row.fermi_t_fm):
        return INSIDE_2PF
    return f"OUTSIDE_2PF_T{float(row.fermi_t_fm):g}"


def render_recipe_input(row: InputRow, kind: str, stage: str, uehling: bool, settings: Settings,
                        extra: list[str]) -> str:
    """The input of a patched-recipe run: the renderer's own input for the kind (`base` for the anchor
    kind), printed at RECIPE_XR_PRECISION / RECIPE_STATE_PRECISION, the anchor kind's extra line and
    hydrogen-like shell, ``uehling_correction: FALSE`` for an Uehling-free twin, then the header
    precision, the Hermite read-back, the stage's keywords, the intended rms where the stage solves c
    from it, and the settings; each key set once."""
    if kind not in RECIPE_KINDS + RESPONSE_KINDS:
        raise ValueError(f"unknown recipe kind {kind!r}")
    if stage not in STAGE_KEYWORDS:
        raise ValueError(f"unknown recipe stage {stage!r}")
    out = []
    for line in render_input(row, "base" if kind == "anchor" else kind, extra).splitlines():
        if line == "xr_print_precision: 6":
            line = f"xr_print_precision: {RECIPE_XR_PRECISION}"
        elif line == "state_print_precision: 6":
            line = f"state_print_precision: {RECIPE_STATE_PRECISION}"
        elif line == "uehling_correction: TRUE" and not uehling:
            line = "uehling_correction: FALSE"
        elif line.startswith("xr_lines: ") and kind == "anchor":
            line = f"{line},{anchor_line()}"
        out.append(line)
    if kind == "anchor":
        out.append(f"ideal_atom_minshell: {shell(N_MAX + 1)}")
    keywords = [("state_header_precision", str(RECIPE_HEADER_PRECISION)), ("fermi2_hermite", "TRUE"),
                *STAGE_KEYWORDS[stage]]
    if ("fermi2_rms_exact", "TRUE") in keywords:
        keywords.append(("fermi2_rms_fm", repr(recipe_rms_fm(row, kind))))
    keywords += settings.keywords()
    for key, value in keywords:
        if any(line.startswith(f"{key}:") for line in out):
            raise ValueError(f"{key} set twice")
        out.append(f"{key}: {value}")
    return "\n".join(out) + "\n"


@dataclass(frozen=True)
class RecipeRun:
    """One run of a recipe batch: its id, member, kind, stage, Uehling switch, variant and input."""

    run: str
    row: InputRow
    kind: str
    stage: str
    uehling: bool
    variant: str
    text: str


def recipe_run_id(row: InputRow, kind: str, uehling: bool, variant: str = "") -> str:
    """``<Sym><A>_<kind>_<on|off>[_<variant>]``; a stage run is ``<Sym><A>_<stage>``."""
    if kind == "stage":
        return run_id(row, variant)
    return run_id(row, f"{kind}_{'on' if uehling else 'off'}" + (f"_{variant}" if variant else ""))


def compared_nuclides(cells: tuple[Cell, ...]) -> list[tuple[int, int]]:
    """Every (Z, A) the validation cells name, gated or not, ascending."""
    return sorted({cell.nuclide for cell in cells})


def recipe_runs(rows: Sequence[InputRow], cells: tuple[Cell, ...], batch: str) -> list[RecipeRun]:
    """Every run of ``batch``, in order: `prod` (each member's kinds, the anchor kind for `base`, each
    with its Uehling-free twin), `refine` (each member's anchor run one step finer per setting, and its
    twin per twin setting), `rminus` (the radius moved down, where its rms stays inside the 2pF domain)
    or `stages` (each compared nuclide's run at every stage before the recipe: the shipped input, then
    the anchor kind)."""
    validation = set(gated_nuclides(cells))
    compared = set(compared_nuclides(cells))
    out: list[RecipeRun] = []

    def add(row: InputRow, kind: str, stage: str, uehling: bool, settings: Settings,
            variant: str = "") -> None:
        text = render_recipe_input(row, kind, stage, uehling, settings, extra_lines(cells, row.nuclide))
        name = recipe_run_id(row, "stage" if stage != "recipe" else kind, uehling, variant)
        out.append(RecipeRun(name, row, kind, stage, uehling, variant or "prod", text))

    if batch == "prod":
        for row in rows:
            for kind in run_kinds(row, validation):
                for uehling in (True, False):
                    add(row, "anchor" if kind == "base" else kind, "recipe", uehling, RECIPE_SETTINGS)
    elif batch == "refine":
        for row in rows:
            for tag, settings in REFINEMENTS:
                add(row, "anchor", "recipe", True, settings, tag)
            for tag, settings in REFINEMENTS:
                if tag in NOU_REFINEMENTS:
                    add(row, "anchor", "recipe", False, settings, tag)
    elif batch == "rminus":
        for row in rows:
            if density_domain(row, "rminus") == INSIDE_2PF:
                add(row, "rminus", "recipe", True, RECIPE_SETTINGS)
    elif batch == "stages":
        for row in rows:
            if row.nuclide not in compared:
                continue
            text = render_input(row, "base", extra_lines(cells, row.nuclide))
            out.append(RecipeRun(run_id(row, "shipped"), row, "base", "shipped", True, "shipped", text))
            add(row, "anchor", "numerics", True, NUMERICS_STAGE_SETTINGS, "numerics")
            for stage in ("density", "rms", "mass"):
                add(row, "anchor", stage, True, RECIPE_SETTINGS, stage)
    else:
        raise ValueError(f"unknown recipe batch {batch!r}")
    return out


# --------------------------------------------------------------------------------------------
# the printed outputs of a recipe run: completeness first, then the strings
# --------------------------------------------------------------------------------------------

class IncompleteRunError(Mudirac130Error):
    """A run's printed outputs lack an expected line or state header, or are malformed."""


_SHELLS = "KLMNOPQRS"


def iupac_state(label: str) -> tuple[int, int, int]:
    """``(n, l, 2j)`` of an IUPAC orbit label (orbit 1 is s1/2, 2l is j = l - 1/2, 2l + 1 is j = l + 1/2)."""
    n = _SHELLS.index(label[0]) + 1
    number = int(label[1:])
    ell = number // 2
    return n, ell, 1 if number == 1 else (2 * ell + 1 if number % 2 else 2 * ell - 1)


def _iupac_range(spec: str) -> list[str]:
    """MuDirac's range spelling: ``L2:L3`` is every orbit from L2 to L3; a single orbit is itself."""
    parts = spec.split(":")
    if len(parts) == 1:
        return parts
    (n1, _l1, _j1), (n2, _l2, _j2) = iupac_state(parts[0]), iupac_state(parts[1])
    first, last = int(parts[0][1:]), int(parts[1][1:])
    out = []
    for n in range(n1, n2 + 1):
        for number in range(first if n == n1 else 1, (last if n == n2 else 2 * n - 1) + 1):
            out.append(f"{_SHELLS[n - 1]}{number}")
    return out


def expected_printed_lines(xr_lines: str) -> list[str]:
    """Every line a run prints for its ``xr_lines`` keyword, as MuDirac expands it: n2 >= n1,
    |l2 - l1| = 1 and |j2 - j1| <= 1 (a line with Delta j = 2 has no E1 rate and is not printed)."""
    out: list[str] = []
    for item in xr_lines.split(","):
        low, high = item.split("-")
        for lo in _iupac_range(low):
            for hi in _iupac_range(high):
                (na, la, ja), (nb, lb, jb) = iupac_state(lo), iupac_state(hi)
                if nb < na or abs(lb - la) != 1 or abs(jb - ja) > 2:
                    continue
                out.append(f"{lo}-{hi}")
    return out


def input_keywords(text: str) -> dict[str, str]:
    """``{key: value}`` of an input file's ``key: value`` lines."""
    pairs = (line.split(":", 1) for line in text.splitlines() if ":" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def read_recipe_run(directory: Path, run: str, text: str, rc: str, err: str) -> tuple[
        dict[str, tuple[str, ...]], list[tuple[str, str, str]]]:
    """The printed state headers ``{orbit: (n, l, s, E, E + mc^2)}`` and lines ``[(line, dE, rate)]`` of
    one run, after its completeness check: exit status 0 and an empty ``.err``; the ``.xr.out`` ends
    with a newline after its two header lines; every printed line has exactly the input's decimals and
    the printed lines are exactly the expected ones (``expected_printed_lines``), once each; every state
    of an expected line has a header with (n, l, s) matching its label and both energies printed. A
    run that fails any clause raises IncompleteRunError naming the run and the clause; nothing of it is
    returned."""
    def refuse(why: str) -> IncompleteRunError:
        return IncompleteRunError(f"{run}: {why}")
    if rc != "0" or err != "0":
        raise refuse(f"rc={rc} err_bytes={err}")
    keywords = input_keywords(text)
    decimals = int(keywords["xr_print_precision"])
    xr = directory / f"{run}.xr.out"
    if not xr.is_file():
        raise refuse("no .xr.out")
    raw = xr.read_bytes()
    if not raw.endswith(b"\n"):
        raise refuse(".xr.out does not end with a newline")
    rows = raw.decode("ascii").split("\n")[:-1]
    if len(rows) < 2 or not rows[0].startswith("# Z = ") or not rows[1].startswith("Line\tDeltaE"):
        raise refuse(".xr.out header malformed")
    energy = re.compile(rf"[0-9]+\.[0-9]{{{decimals}}}")
    lines: list[tuple[str, str, str]] = []
    for row in rows[2:]:
        fields = [f for f in row.split("\t") if f != ""]
        if len(fields) != 3 or not _LINE.fullmatch(fields[0]) or not energy.fullmatch(fields[1]) \
                or not energy.fullmatch(fields[2]):
            raise refuse(f"malformed printed line {row!r}")
        lines.append((fields[0], fields[1], fields[2]))
    expected = expected_printed_lines(keywords["xr_lines"])
    if sorted(name for name, _e, _r in lines) != sorted(expected):
        missing = sorted(set(expected) - {name for name, _e, _r in lines})
        raise refuse(f"printed lines differ from the expected ones (missing {missing[:4]})")
    headers: dict[str, tuple[str, ...]] = {}
    for state in sorted({s for line in expected for s in line.split("-")}, key=_orbit_order):
        path = directory / f"{run}.{state}.out"
        if not path.is_file():
            raise refuse(f"no state file for {state}")
        head = path.read_bytes().decode("ascii").split("\n")[:5]
        match_state = _STATE_HEADER_LINE.match(head[1]) if len(head) > 2 else None
        match_energy = _ENERGY_HEADER_LINE.match(head[2]) if len(head) > 2 else None
        if not (match_state and match_energy):
            raise refuse(f"state {state} has no complete header")
        n, ell, two_j = iupac_state(state)
        s_expected = 1 if (ell == 0 or two_j == 2 * ell + 1) else 0
        if match_state.groups() != (str(n), str(ell), str(s_expected)):
            raise refuse(f"state {state} header names n, l, s = {match_state.groups()}")
        headers[state] = (*match_state.groups(), *match_energy.groups())
    return headers, lines


_STATE_HEADER_LINE = re.compile(r"^# DiracState with n = (\d+), l = (\d+), s = (\d+)$")
_ENERGY_HEADER_LINE = re.compile(r"^# E = (\S+) \+ mc\^2 = (\S+) eV$")


# --------------------------------------------------------------------------------------------
# the derivation of binding energies from a recipe run's printed strings
# --------------------------------------------------------------------------------------------

def float_floor_eV(total_eV: str, eh: float) -> Decimal:
    """ulp(mu c^2), in eV: the spacing of the doubles the binary holds a total energy (binding plus
    mu c^2, in Hartree) on, which every printed line and header carries whatever its decimals."""
    total_ha = float(total_eV) / eh
    with localcontext() as context:
        context.prec = _PRECISION
        return Decimal(2.0 ** (math.floor(math.log2(total_ha)) - 52)) * Decimal(repr(eh))


def header_half_unit(text: str) -> Decimal:
    """Half a unit of the last printed digit of a header energy (fixed or exponent form)."""
    mantissa, _sep, exponent = text.lstrip("-").lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return Decimal(5) * Decimal(10) ** (int(exponent or 0) - decimals - 1)


def derive_recipe_bindings(run: str, kind: str, headers: dict[str, str], lines: dict[str, str],
                           total_eV: str, eh: float) -> dict[str, tuple[Decimal, Decimal]]:
    """``{orbit: (B, rounding)}`` in eV for every circular orbit of shells 1 .. N_MAX of one recipe run.

    The outermost circular orbit (2l+1 of shell N_MAX): for the anchor kind, minus the printed header of
    the hydrogen-like orbit of shell N_MAX + 1 plus the printed line between them; for every other kind,
    minus its own printed header. Then the upper orbits downward by adding each printed line and the
    lower orbits outward from K1 by subtracting each. A printed line carries half a unit of its last
    decimal plus ``float_floor_eV`` of ``total_eV``, the run's printed total energy of that orbit;
    a header the same with its own last digit. Raises DerivationError
    unless every derived energy lies within its header's rounding, its own rounding and one more line's
    of its printed header, and every cross line closes within the rounding of its two ends and itself.
    """
    chain = chain_states()
    needed = list(chain) + ([orbit(N_MAX + 1, True)] if kind == "anchor" else [])
    for state in needed:
        if state not in headers:
            raise DerivationError(f"{run}: no printed header for {state}")
    wanted = upper_chain_lines() + lower_chain_lines() + cross_lines()
    if kind == "anchor":
        wanted.append(anchor_line())
    for line in wanted:
        if line not in lines:
            raise DerivationError(f"{run}: no printed line {line}")
    top = orbit(N_MAX, True)
    with localcontext() as context:
        context.prec = _PRECISION
        floor = float_floor_eV(total_eV, eh)
        rho_line = Decimal(5) * Decimal(10) ** (-RECIPE_XR_PRECISION - 1) + floor
        if kind == "anchor":
            outer = orbit(N_MAX + 1, True)
            b = {top: (-Decimal(headers[outer]) + Decimal(lines[anchor_line()]),
                       header_half_unit(headers[outer]) + floor + rho_line)}
        else:
            b = {top: (-Decimal(headers[top]), header_half_unit(headers[top]) + floor)}
        for n in range(N_MAX - 1, 0, -1):
            lower, upper = orbit(n, True), orbit(n + 1, True)
            b[lower] = (b[upper][0] + Decimal(lines[f"{lower}-{upper}"]), b[upper][1] + rho_line)
        for n in range(1, N_MAX):
            lower, upper = orbit(n, False), orbit(n + 1, False)
            b[upper] = (b[lower][0] - Decimal(lines[f"{lower}-{upper}"]), b[lower][1] + rho_line)
        for state in chain:
            value, rounding = b[state]
            header = headers[state]
            gap = abs(value + Decimal(header))
            if gap > header_half_unit(header) + floor + rounding + rho_line:
                raise DerivationError(f"{run}: derived {state} lies {gap} eV from its printed header")
        for line in cross_lines():
            lower, upper = line.split("-")
            residual = abs(b[lower][0] - b[upper][0] - Decimal(lines[line]))
            if residual > b[lower][1] + b[upper][1] + rho_line:
                raise DerivationError(f"{run}: cross line {line} closes to {residual} eV")
    return b


def recipe_quantities(b: dict[str, tuple[Decimal, Decimal]]) -> tuple[Decimal, tuple[Decimal, ...]]:
    """``(K, (e2 .. e<N_MAX>))`` in keV from ``derive_recipe_bindings``: the exact decimal eV-to-keV
    division of the 1s binding energy and of each circular level's (2j+1)-weighted mean."""
    return quantities({state: value for state, (value, _rounding) in b.items()})
