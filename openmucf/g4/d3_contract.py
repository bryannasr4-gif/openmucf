"""The representation contract of the D3 tables: what a cascade receives, measured beside the tables.

The shipped tables are not touched. This module reads them as text, joins every gated measured line
to the shell difference a patched cascade emits for it, intersects the bands of the lines that share
a shell pair, loads what is recorded of the radius each compared member was run with, and writes one
component record per emitted value: the signed radius responses in both directions, the rounding
bounds of the printed anchor and of the summed lines, the shifts observed when MuDirac's numerical
settings are refined, and the legacy ``unc``/``u<n>`` cell under the name of what it is. No total
uncertainty is produced, no tolerance moves and no central value moves.

Standard library plus ``openmucf.g4.sources.mudirac130`` only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

from openmucf.g4.sources import mudirac130 as md

PROJECTION_RELPATH = f"{md.D3_RELDIR}/shell_projection.csv"
GROUPS_RELPATH = f"{md.D3_RELDIR}/incompatible_groups.csv"
KSHELL_RELPATH = f"{md.D3_RELDIR}/d3_kshell.{md.PROFILE}.g4dat"
LEVELS_RELPATH = f"{md.D3_RELDIR}/d3_levels.{md.PROFILE}.g4dat"

#: What the cascade receives for a measured line: the difference of two shell energies.
CONSUMER_QUANTITY = "shell_difference"
#: The emitted quantities of one (Z, A) row, in the tables' column order.
QUANTITIES = ("K", *(f"e{n}" for n in range(2, md.N_MAX + 1)))

PROJECTION_COLUMNS = (
    "source", "Z", "A", "transition", "quantity", "consumer_quantity", "initial_n", "final_n",
    "measured_keV", "unc_keV", "tol_keV", "solver_keV", "consumer_keV", "stock_keV",
    "solver_residual_keV", "consumer_residual_keV", "stock_residual_keV", "solver_within",
    "consumer_within", "stock_within", "consumer_closer_than_stock", "representation_error_keV",
    "solver_numeric_target_keV", "solver_numeric_shifts_keV", "solver_numeric_qualification",
    "consumer_numeric_target_keV", "consumer_numeric_shifts_keV", "consumer_numeric_qualification",
    "locator", "copy_read",
)
GROUPS_COLUMNS = (
    "source", "Z", "A", "initial_n", "final_n", "lines", "quantities", "lower_keV", "upper_keV",
    "gap_keV", "empty", "basis",
)

#: The numeric-qualification tokens: observed stability under refinement, never a true error.
INSUFFICIENT_LEVELS = "INSUFFICIENT_LEVELS"
RUN_FAILED = "RUN_FAILED"
QUALIFIED = "QUALIFIED_AT_LEVEL_"
NOT_QUALIFIED = "NOT_QUALIFIED_THROUGH_LEVEL_"
#: The fewest valid levels a qualification needs: two shifts.
MIN_VALID_LEVELS = 3
#: The fraction of the smallest printed uncertainty a compared line's resolution target is.
LINE_TARGET_FRACTION = Decimal("0.1")
#: A declared engineering target for an emitted binding: the larger of an absolute and a relative
#: value, neither an experimental uncertainty.
BINDING_TARGET_ABSOLUTE_KEV = Decimal("1e-6")
BINDING_TARGET_RELATIVE = Decimal("1e-6")


def _text(value: Decimal) -> str:
    return format(value, "f")


def _shell_index(orbit: str) -> int:
    """The principal quantum number an IUPAC orbit's shell letter names: K is 1."""
    return ord(orbit[0]) - ord("K") + 1


def _shell_quantity(n: int) -> str:
    return "K" if n == 1 else f"e{n}"


# --------------------------------------------------------------------------------------------
# the tables as text
# --------------------------------------------------------------------------------------------


def read_table_text(path: Path) -> dict[tuple[int, int], list[str]]:
    """The record cells of a ``.g4dat`` as text, keyed by (Z, A): every line that does not start
    with ``#`` split on whitespace, the first two fields the key. Nothing is parsed as a number."""
    out: dict[tuple[int, int], list[str]] = {}
    for number, line in enumerate(Path(path).read_bytes().decode("ascii").split("\n"), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        key = (int(fields[0]), int(fields[1]))
        if key in out:
            raise md.DuplicateKeyError(f"{Path(path).name} line {number}: a second record for {key}")
        out[key] = fields[2:]
    return out


@dataclass(frozen=True)
class Tables:
    """The two tables' cells as text."""

    kshell: dict[tuple[int, int], list[str]]
    levels: dict[tuple[int, int], list[str]]

    def central(self, key: tuple[int, int], quantity: str) -> str:
        """The ``value`` or ``e<n>`` cell of ``key``, as the table prints it."""
        if quantity == "K":
            return self.kshell[key][0]
        return self.levels[key][int(quantity[1:]) - 2]

    def legacy(self, key: tuple[int, int], quantity: str) -> str:
        """The ``unc`` or ``u<n>`` cell of ``key``, as the table prints it."""
        if quantity == "K":
            return self.kshell[key][1]
        return self.levels[key][len(QUANTITIES) - 1 + int(quantity[1:]) - 2]

    def shell(self, key: tuple[int, int], n: int) -> str:
        """The binding energy the cascade receives for shell ``n``: ``value`` for 1, ``e<n>`` above."""
        return self.central(key, _shell_quantity(n))


# --------------------------------------------------------------------------------------------
# the refined runs of one member, and the qualification of what they show
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Refinement:
    """One member's refinement levels: the levels run, the derived bindings and printed lines of
    each valid level, and whether any level's run failed."""

    levels: list[int]
    bindings: dict[int, dict[str, Decimal]]
    lines: dict[int, dict[str, str]]
    failed: bool

    @property
    def valid(self) -> list[int]:
        return sorted(self.bindings)


def path_lengths() -> dict[str, int]:
    """How many printed lines the derivation sums from the anchor to each circular state: down the
    upper chain to K1, then out the lower chain."""
    path = {md.orbit(md.N_MAX, True): 0}
    for line in reversed(md.upper_chain_lines()):
        lower, upper = line.split("-")
        path[lower] = path[upper] + 1
    for line in md.lower_chain_lines():
        lower, upper = line.split("-")
        path[upper] = path[lower] + 1
    return path


def line_rounding_bound_kev(quantity: str) -> Decimal:
    """Half a printed unit per line on the derivation path, in keV: K1's path for ``K``, and for
    ``e<n>`` the (2j+1)-weighted sum of its two circular states' paths -- absolute coefficients."""
    path = path_lengths()
    with localcontext() as context:
        context.prec = md._PRECISION
        unit = md.LINE_HALF_UNIT / 1000
        if quantity == "K":
            return path["K1"] * unit
        n = int(quantity[1:])
        ell = n - 1
        weighted = 2 * ell * path[md.orbit(n, False)] + (2 * ell + 2) * path[md.orbit(n, True)]
        return Decimal(weighted) / (4 * ell + 2) * unit


def quantity_value(b: dict[str, Decimal], quantity: str) -> Decimal:
    """``K`` or ``e<n>`` in keV from a run's derived bindings."""
    k, levels = md.quantities(b)
    return k if quantity == "K" else levels[int(quantity[1:]) - 2]


def binding_target_kev(central: str) -> Decimal:
    with localcontext() as context:
        context.prec = md._PRECISION
        return max(BINDING_TARGET_ABSOLUTE_KEV, BINDING_TARGET_RELATIVE * abs(Decimal(central)))


def shifts(values: dict[int, Decimal]) -> list[Decimal]:
    """The differences between consecutive valid levels, oldest first."""
    levels = sorted(values)
    with localcontext() as context:
        context.prec = md._PRECISION
        return [values[b] - values[a] for a, b in zip(levels, levels[1:], strict=False)]


def qualification(observed: list[Decimal], valid: list[int], failed: bool, target: Decimal,
                  allowance: Decimal) -> str:
    """Observed stability only: with fewer than MIN_VALID_LEVELS valid levels the record is
    INSUFFICIENT_LEVELS; with any level's run failed it is RUN_FAILED; else it is qualified at the
    highest valid level iff the last two shifts both lie within target plus allowance and the
    newer is no larger than the older."""
    if len(valid) < MIN_VALID_LEVELS:
        return INSUFFICIENT_LEVELS
    if failed:
        return RUN_FAILED
    k = valid[-1]
    older, newer = observed[-2], observed[-1]
    with localcontext() as context:
        context.prec = md._PRECISION
        bound = target + allowance
        if abs(older) <= bound and abs(newer) <= bound and abs(newer) <= abs(older):
            return f"{QUALIFIED}{k}"
    return f"{NOT_QUALIFIED}{k}"


# --------------------------------------------------------------------------------------------
# everything the artifacts read, loaded once
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bundle:
    outputs: md.Outputs
    numerics: md.NumericsOutputs
    tables: Tables
    cells: tuple[md.Cell, ...]
    origins: dict[tuple[int, int], md.RadiusOrigin]
    stock: dict[tuple[int, int], tuple[float, ...]]
    kept: list[md.InputRow]
    #: Per kept member: the derived bindings of the base, rsig and (where made) rminus runs.
    bindings: dict[tuple[int, int], dict[str, dict[str, Decimal]]]
    refinements: dict[tuple[int, int], Refinement]

    def row(self, key: tuple[int, int]) -> md.InputRow:
        return next(row for row in self.kept if row.nuclide == key)


def _derive(
    run: str, headers: dict[str, dict[str, str]], lines: dict[str, dict[str, str]]
) -> dict[str, Decimal]:
    return md.derive_bindings(run, headers.get(run, {}), lines.get(run, {}))


def _refinement(row: md.InputRow, numerics: md.NumericsOutputs) -> Refinement:
    levels = md.numerics_levels(row)
    bindings: dict[int, dict[str, Decimal]] = {}
    lines: dict[int, dict[str, str]] = {}
    failed = False
    for level in levels:
        run = numerics.runs[md.numerics_run_id(row, "grid", level)]
        if not run.clean:
            failed = True
            continue
        try:
            bindings[level] = _derive(run.run, numerics.headers, numerics.lines)
        except md.DerivationError:
            continue
        lines[level] = numerics.lines[run.run]
    return Refinement(levels, bindings, lines, failed)


def load_bundle(root: Path) -> Bundle:
    """Every committed file the contract reads, cross-checked by its own loader."""
    root = Path(root)
    outputs = md.load_outputs(root)
    numerics = md.load_numerics_outputs(root, outputs)
    tables = Tables(read_table_text(root / KSHELL_RELPATH), read_table_text(root / LEVELS_RELPATH))
    cells, origins = md.load_validation(root / md.CELLS_RELPATH, root / md.ORIGIN_RELPATH)
    stock = md.load_geant4_levels(root / md.GEANT4_LEVELS_RELPATH)
    kept = md.kept_members(outputs)
    bindings: dict[tuple[int, int], dict[str, dict[str, Decimal]]] = {}
    refinements: dict[tuple[int, int], Refinement] = {}
    for row in kept:
        derived = {kind: _derive(md.run_id(row, kind), outputs.headers, outputs.lines)
                   for kind in ("base", "rsig")}
        minus = numerics.runs[md.numerics_run_id(row, "rminus", 0)]
        if minus.clean:
            derived["rminus"] = _derive(minus.run, numerics.headers, numerics.lines)
        bindings[row.nuclide] = derived
        refinements[row.nuclide] = _refinement(row, numerics)
    return Bundle(outputs, numerics, tables, cells, origins, stock, kept, bindings, refinements)


def _bundle(root: Path, bundle: Bundle | None) -> Bundle:
    return bundle if bundle is not None else load_bundle(root)


# --------------------------------------------------------------------------------------------
# the projection: every gated line beside the shell difference the cascade emits for it
# --------------------------------------------------------------------------------------------


def _line_target_kev(cells: tuple[md.Cell, ...], cell: md.Cell) -> Decimal:
    """A tenth of the smallest positive printed uncertainty among the gated cells of the cell's
    (Z, A, quantity)."""
    sigmas = [Decimal(c.unc_kev) for c in cells
              if c.gated and c.nuclide == cell.nuclide and c.quantity == cell.quantity
              and Decimal(c.unc_kev) > 0]
    with localcontext() as context:
        context.prec = md._PRECISION
        return LINE_TARGET_FRACTION * min(sigmas)


def _anchor_bound_kev(bundle: Bundle, key: tuple[int, int]) -> Decimal:
    header = bundle.outputs.headers[md.run_id(bundle.row(key), "base")][md.orbit(md.N_MAX, True)]
    with localcontext() as context:
        context.prec = md._PRECISION
        return md.half_unit_6sig(header) / 1000


def _binding_allowance_kev(bundle: Bundle, key: tuple[int, int], quantity: str) -> Decimal:
    with localcontext() as context:
        context.prec = md._PRECISION
        return 2 * (line_rounding_bound_kev(quantity) + _anchor_bound_kev(bundle, key))


def _shifts_text(observed: list[Decimal]) -> str:
    return ";".join(_text(s) for s in observed)


def _stock_kev(levels: tuple[float, ...], lower_n: int, upper_n: int) -> Decimal:
    """The photon the unpatched cascade emits between the two shells: the difference of its two
    level energies taken in doubles, in keV at nine decimals."""
    photon = levels[lower_n - 1] - levels[upper_n - 1]
    with localcontext() as context:
        context.prec = 1000
        return (Decimal(photon) * 1000).quantize(Decimal("1e-9"))


def project_shell_rows(root: Path, bundle: Bundle | None = None) -> list[dict[str, str]]:
    """One row per gated cell, in the transcription's order: the measured line beside the solver
    line MuDirac printed, the shell difference a patched cascade emits between the line's two
    shells, and the shell difference the unpatched cascade emits; residuals against the measured
    value, the frozen band, and the observed stability of the solver line and of the shell
    difference under the refined settings. The measured value keeps its label; the consumer
    quantity is named ``shell_difference`` on every row."""
    bundle = _bundle(root, bundle)
    if md.SIGMA_CALC != 0:
        raise md.CellError("the frozen band is TOL_FACTOR * unc; SIGMA_CALC is not 0")
    inputs = {row.nuclide: row for row in bundle.outputs.inputs}
    rows: list[dict[str, str]] = []
    for cell in bundle.cells:
        if not cell.gated:
            continue
        key = cell.nuclide
        lower_n, upper_n = (_shell_index(orbit) for orbit in cell.quantity.split("-"))
        pair = (lower_n, upper_n)
        refinement = bundle.refinements[key]
        with localcontext() as context:
            context.prec = md._PRECISION
            measured = Decimal(cell.value_kev)
            tol = md.TOL_FACTOR * Decimal(cell.unc_kev)
            solver = Decimal(bundle.outputs.lines[md.run_id(inputs[key], "base")][cell.quantity]) / 1000
            consumer = Decimal(bundle.tables.shell(key, lower_n)) - Decimal(bundle.tables.shell(key, upper_n))
            stock = _stock_kev(bundle.stock[key], lower_n, upper_n)
            residuals = {name: value - measured for name, value in
                         (("solver", solver), ("consumer", consumer), ("stock", stock))}
            solver_levels = {level: Decimal(lines[cell.quantity]) / 1000
                             for level, lines in refinement.lines.items() if cell.quantity in lines}
            solver_shifts = shifts(solver_levels)
            solver_target = _line_target_kev(bundle.cells, cell)
            solver_allowance = 2 * md.LINE_HALF_UNIT / 1000
            lower_q, upper_q = _shell_quantity(lower_n), _shell_quantity(upper_n)
            consumer_levels = {level: quantity_value(b, lower_q) - quantity_value(b, upper_q)
                               for level, b in refinement.bindings.items()}
            consumer_shifts = shifts(consumer_levels)
            consumer_target = sum((binding_target_kev(bundle.tables.shell(key, n)) for n in pair), Decimal(0))
            consumer_allowance = sum(
                (_binding_allowance_kev(bundle, key, _shell_quantity(n)) for n in pair), Decimal(0)
            )
        rows.append({
            "source": cell.source, "Z": str(cell.z), "A": str(cell.a), "transition": cell.transition,
            "quantity": cell.quantity, "consumer_quantity": CONSUMER_QUANTITY,
            "initial_n": str(upper_n), "final_n": str(lower_n),
            "measured_keV": cell.value_kev, "unc_keV": cell.unc_kev, "tol_keV": _text(tol),
            "solver_keV": _text(solver), "consumer_keV": _text(consumer), "stock_keV": _text(stock),
            "solver_residual_keV": _text(residuals["solver"]),
            "consumer_residual_keV": _text(residuals["consumer"]),
            "stock_residual_keV": _text(residuals["stock"]),
            "solver_within": str(abs(residuals["solver"]) <= tol).lower(),
            "consumer_within": str(abs(residuals["consumer"]) <= tol).lower(),
            "stock_within": str(abs(residuals["stock"]) <= tol).lower(),
            "consumer_closer_than_stock": str(abs(residuals["consumer"]) < abs(residuals["stock"])).lower(),
            "representation_error_keV": _text(consumer - solver),
            "solver_numeric_target_keV": _text(solver_target),
            "solver_numeric_shifts_keV": _shifts_text(solver_shifts),
            "solver_numeric_qualification": qualification(
                solver_shifts, sorted(solver_levels), refinement.failed, solver_target, solver_allowance),
            "consumer_numeric_target_keV": _text(consumer_target),
            "consumer_numeric_shifts_keV": _shifts_text(consumer_shifts),
            "consumer_numeric_qualification": qualification(
                consumer_shifts, refinement.valid, refinement.failed, consumer_target, consumer_allowance),
            "locator": cell.locator, "copy_read": cell.copy_read,
        })
    return rows


def incompatible_groups(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Every (source, Z, A, initial_n, final_n) group of the projection holding more than one
    distinct measured line, with the intersection of the lines' bands: ``lower`` = the largest
    lower band edge, ``upper`` = the smallest upper edge, ``gap`` = lower - upper (signed), and
    ``empty`` when the gap is positive. ``basis`` is what the cells were read from, never a new
    reading. Ordered by key."""
    groups: dict[tuple[str, int, int, int, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["source"], int(row["Z"]), int(row["A"]), int(row["initial_n"]), int(row["final_n"]))
        groups.setdefault(key, []).append(row)
    out = []
    for key in sorted(groups):
        members = groups[key]
        if len({row["quantity"] for row in members}) < 2:
            continue
        with localcontext() as context:
            context.prec = md._PRECISION
            lower = max(Decimal(row["measured_keV"]) - Decimal(row["tol_keV"]) for row in members)
            upper = min(Decimal(row["measured_keV"]) + Decimal(row["tol_keV"]) for row in members)
            gap = lower - upper
        basis: list[str] = []
        for row in members:
            if row["copy_read"] not in basis:
                basis.append(row["copy_read"])
        out.append({
            "source": key[0], "Z": str(key[1]), "A": str(key[2]), "initial_n": str(key[3]),
            "final_n": str(key[4]), "lines": str(len(members)),
            "quantities": ";".join(row["quantity"] for row in members),
            "lower_keV": _text(lower), "upper_keV": _text(upper), "gap_keV": _text(gap),
            "empty": str(gap > 0).lower(), "basis": ";".join(basis),
        })
    return out


# --------------------------------------------------------------------------------------------
# the renderers and the summary
# --------------------------------------------------------------------------------------------


def _render_csv(columns: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[column] for column in columns])
    return buffer.getvalue().encode("ascii")


def render_projection(root: Path, bundle: Bundle | None = None) -> bytes:
    """``shell_projection.csv``: LF, ASCII, the header then one row per gated cell."""
    return _render_csv(PROJECTION_COLUMNS, project_shell_rows(root, bundle))


def render_groups(root: Path, bundle: Bundle | None = None) -> bytes:
    """``incompatible_groups.csv``: LF, ASCII, the header then one row per multi-line group."""
    return _render_csv(GROUPS_COLUMNS, incompatible_groups(project_shell_rows(root, bundle)))


#: The columns of the document's groups table, each a column of ``incompatible_groups.csv``.
GROUPS_TABLE_COLUMNS = (
    ("source", "source"), ("Z", "Z"), ("A", "A"), ("initial n", "initial_n"), ("final n", "final_n"),
    ("lines", "lines"), ("quantities", "quantities"), ("lower (keV)", "lower_keV"),
    ("upper (keV)", "upper_keV"), ("gap (keV)", "gap_keV"), ("empty", "empty"),
)


def render_groups_table(rows: list[dict[str, str]]) -> str:
    """The document's groups table: one Markdown row per row of ``incompatible_groups.csv``, in its
    order, every cell copied from the file."""
    lines = [
        "| " + " | ".join(title for title, _ in GROUPS_TABLE_COLUMNS) + " |",
        "|" + "---|" * len(GROUPS_TABLE_COLUMNS),
    ]
    for row in rows:
        cells = [f"`{row[column]}`" if column == "quantities" else row[column]
                 for _, column in GROUPS_TABLE_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _histogram(tokens: list[str]) -> str:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return ", ".join(f"{token} {count}" for token, count in sorted(counts.items()))


def summary_lines(root: Path, bundle: Bundle | None = None) -> list[str]:
    """What the contract measured, one line per artifact -- printed by the regeneration and the audit."""
    root = Path(root)
    bundle = _bundle(root, bundle)
    rows = project_shell_rows(root, bundle)
    groups = incompatible_groups(rows)
    empty = [g for g in groups if g["empty"] == "true"]
    widest = max(groups, key=lambda g: Decimal(g["gap_keV"]))
    return [
        f"d3 contract: gated {len(rows)} solver_within {sum(r['solver_within'] == 'true' for r in rows)} "
        f"consumer_within {sum(r['consumer_within'] == 'true' for r in rows)} "
        f"stock_within {sum(r['stock_within'] == 'true' for r in rows)} "
        f"consumer_closer_than_stock {sum(r['consumer_closer_than_stock'] == 'true' for r in rows)}",
        f"d3 contract: multi-line groups {len(groups)} empty {len(empty)}; widest "
        f"({widest['source']}, {widest['Z']}, {widest['A']}, {widest['initial_n']}, {widest['final_n']}) "
        f"gap {widest['gap_keV']} keV",
        f"d3 contract: numeric qualification over {len(rows)} gated solver lines: "
        + _histogram([r["solver_numeric_qualification"] for r in rows]),
        lineage_summary(root),
        components_summary(root, bundle),
    ]


# --------------------------------------------------------------------------------------------
# the radius lineage: what is known of the experiments behind each compared member's radius
# --------------------------------------------------------------------------------------------

LINEAGE_RELPATH = f"{md.D3_RELDIR}/radius_lineage.csv"
LINEAGE_COLUMNS = (
    "Z", "A", "input_source", "primary_experiment_id", "method", "calibration_inputs",
    "covariance_source", "comparison_experiment_id", "dependency_state", "locator",
)
#: What is known of the experiments behind a member's radius and behind its compared lines:
#: traced and shared, traced and disjoint, or not traced.
SHARED = "SHARED"
DISJOINT_DOCUMENTED = "DISJOINT_DOCUMENTED"
UNKNOWN = "UNKNOWN"
DEPENDENCY_STATES = (SHARED, DISJOINT_DOCUMENTED, UNKNOWN)
#: The cell a lineage row carries where a primary or its calibration inputs are not in hand.
NOT_TRACED = "not traced"


@dataclass(frozen=True)
class Lineage:
    z: int
    a: int
    input_source: str
    primary_experiment_id: str
    method: str
    calibration_inputs: str
    covariance_source: str
    comparison_experiment_id: str
    dependency_state: str
    locator: str

    @property
    def nuclide(self) -> tuple[int, int]:
        return (self.z, self.a)


def load_radius_lineage(path: Path) -> tuple[Lineage, ...]:
    """Parse ``radius_lineage.csv``: one row per gated nuclide of the ``validation_cells.csv``
    beside it, in that order; a state among DEPENDENCY_STATES; SHARED and DISJOINT_DOCUMENTED only
    with a traced primary and traced calibration inputs."""
    path = Path(path)
    gated = md.gated_nuclides(md.load_cells(path.with_name(Path(md.CELLS_RELPATH).name)))
    rows: list[Lineage] = []
    for where, r in md.read_rows(path, LINEAGE_COLUMNS):
        z, a = md.integer(r["Z"], where, "Z"), md.integer(r["A"], where, "A")
        if r["dependency_state"] not in DEPENDENCY_STATES:
            raise md.CellError(
                f"{where}: dependency_state {r['dependency_state']!r} is not one of {DEPENDENCY_STATES!r}"
            )
        if r["dependency_state"] != UNKNOWN:
            for column in ("primary_experiment_id", "calibration_inputs"):
                if r[column] in ("", NOT_TRACED):
                    raise md.CellError(
                        f"{where}: {r['dependency_state']} needs a traced {column}, got {r[column]!r}"
                    )
        for column in ("input_source", "method", "covariance_source", "comparison_experiment_id", "locator"):
            if not r[column]:
                raise md.CellError(f"{where}: every row must carry a {column}")
        if any(row.nuclide == (z, a) for row in rows):
            raise md.DuplicateKeyError(f"{where}: duplicate key ({z}, {a})")
        rows.append(Lineage(z, a, r["input_source"], r["primary_experiment_id"], r["method"],
                            r["calibration_inputs"], r["covariance_source"],
                            r["comparison_experiment_id"], r["dependency_state"], r["locator"]))
    if [row.nuclide for row in rows] != gated:
        raise md.CellError(
            f"{path.name} names {[row.nuclide for row in rows]}, the gated nuclides in order are {gated}"
        )
    return tuple(rows)


def audit_dependencies(root: Path) -> list[dict[str, str]]:
    """The lineage rows as loaded, each with ``gated_rows``, the count of gated cells of its
    nuclide. Nothing is inferred: a row's state is what the file records."""
    root = Path(root)
    cells = md.load_cells(root / md.CELLS_RELPATH)
    out = []
    for row in load_radius_lineage(root / LINEAGE_RELPATH):
        record = {column: getattr(row, column.lower() if column in ("Z", "A") else column)
                  for column in LINEAGE_COLUMNS}
        record["Z"], record["A"] = str(row.z), str(row.a)
        record["gated_rows"] = str(sum(1 for cell in cells if cell.gated and cell.nuclide == row.nuclide))
        out.append(record)
    return out


def lineage_summary(root: Path) -> str:
    lineage = load_radius_lineage(Path(root) / LINEAGE_RELPATH)
    return (f"d3 contract: lineage states {_histogram([row.dependency_state for row in lineage])} "
            f"over {len(lineage)} rows")


# --------------------------------------------------------------------------------------------
# the component records: per emitted value, what is known of what can move it
# --------------------------------------------------------------------------------------------

COMPONENTS_RELPATH = f"{md.D3_RELDIR}/components.jsonl"
#: What no component record accounts for.
OMITTED_COMPONENTS = [
    "model discrepancy", "nuclear polarization", "electron screening", "higher-order QED",
    "hyperfine structure", "numerical settings beyond the observed levels",
]
RADIUS_PERTURBATION = "source_rms_sigma"
#: The files the records are computed from, digested together into the header record.
COMPONENT_INPUTS = (
    KSHELL_RELPATH, LEVELS_RELPATH, md.INPUTS_RELPATH, md.STATES_RELPATH, md.LINES_RELPATH,
    md.NUMERICS_RUNS_RELPATH, md.NUMERICS_STATES_RELPATH, md.NUMERICS_LINES_RELPATH,
)


def inputs_sha256(root: Path) -> str:
    """One digest over the sorted digests of COMPONENT_INPUTS, one per line, LF."""
    root = Path(root)
    digests = sorted(hashlib.sha256((root / relpath).read_bytes()).hexdigest()
                     for relpath in COMPONENT_INPUTS)
    return hashlib.sha256(("\n".join(digests) + "\n").encode("ascii")).hexdigest()


def header_record(root: Path) -> dict[str, object]:
    return {
        "record": "header", "profile": md.PROFILE, "radius_perturbation": RADIUS_PERTURBATION,
        "omitted_components": list(OMITTED_COMPONENTS), "total_sigma_keV": None,
        "model_uncertainty": "unknown", "inputs_sha256": inputs_sha256(root),
    }


def response_kev(
    bundle: Bundle, key: tuple[int, int], kind: str, quantity: str, relative: bool
) -> str | None:
    """The signed change of ``quantity`` between the base run and the ``kind`` run, from the
    bindings as derived (absolute) or with the anchor subtracted (relative); None where the run
    was not made."""
    derived = bundle.bindings[key]
    if kind not in derived:
        return None
    with localcontext() as context:
        context.prec = md._PRECISION
        moved, base = derived[kind], derived["base"]
        if relative:
            moved, base = md.relative(moved), md.relative(base)
        return _text(quantity_value(moved, quantity) - quantity_value(base, quantity))


def components(root: Path, bundle: Bundle | None = None) -> list[dict[str, object]]:
    """The header record, one record per kept (Z, A) and emitted quantity in (Z, A, quantity)
    order, then one reference record per A = 0 row naming the isotope it carries."""
    root = Path(root)
    bundle = _bundle(root, bundle)
    out: list[dict[str, object]] = [header_record(root)]
    for row in bundle.kept:
        key = row.nuclide
        refinement = bundle.refinements[key]
        anchor = _anchor_bound_kev(bundle, key)
        for quantity in QUANTITIES:
            central = bundle.tables.central(key, quantity)
            values = {level: quantity_value(b, quantity) for level, b in refinement.bindings.items()}
            observed = shifts(values)
            target = binding_target_kev(central)
            out.append({
                "Z": row.z, "A": row.a, "quantity": quantity, "central_keV": central,
                "sigma_rms_fm": row.sigma_rms_fm,
                "radius_plus_relative_response_keV": response_kev(bundle, key, "rsig", quantity, True),
                "radius_minus_relative_response_keV": response_kev(bundle, key, "rminus", quantity, True),
                "radius_plus_absolute_response_keV": response_kev(bundle, key, "rsig", quantity, False),
                "radius_minus_absolute_response_keV": response_kev(bundle, key, "rminus", quantity, False),
                "anchor_rounding_bound_keV": _text(anchor),
                "anchor_coefficient": "1",
                "line_rounding_bound_keV": _text(line_rounding_bound_kev(quantity)),
                "numeric_observed_shifts_keV": [_text(s) for s in observed],
                "numeric_resolution_target_keV": _text(target),
                "numeric_qualification": qualification(
                    observed, refinement.valid, refinement.failed, target,
                    _binding_allowance_kev(bundle, key, quantity)),
                "correlation_groups": [f"anchor:{md.run_id(row, 'base')}", f"radius:{row.symbol}{row.a}"],
                "legacy_radius_response_magnitude_with_floor": bundle.tables.legacy(key, quantity),
            })
    table_rows, _dropped = md.table_rows(bundle.outputs)
    for table_row in table_rows:
        if table_row.a == 0:
            out.append({"record": "reference", "Z": table_row.z, "A": 0,
                        "references": {"Z": table_row.z, "A": table_row.carries}})
    return out


def render_components(root: Path, bundle: Bundle | None = None) -> bytes:
    """``components.jsonl``: one compact ASCII JSON record per line, LF."""
    lines = [json.dumps(record, separators=(",", ":"), ensure_ascii=True)
             for record in components(root, bundle)]
    return ("\n".join(lines) + "\n").encode("ascii")


def components_summary(root: Path, bundle: Bundle | None = None) -> str:
    records = [r for r in components(root, bundle) if "quantity" in r]
    return (f"d3 contract: numeric qualification over {len(records)} components: "
            + _histogram([str(r["numeric_qualification"]) for r in records]))
