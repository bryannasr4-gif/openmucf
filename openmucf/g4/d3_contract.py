"""The representation contract of the D3 tables: what a cascade receives, measured beside the tables.

The shipped tables are not touched. This module reads them as text, joins every gated measured line
to the shell difference a patched cascade emits for it, intersects the bands of the lines that share
a shell pair, loads what is recorded of the radius each compared member was run with, and writes one
component record per emitted value: the signed radius responses in both directions, the rounding
bounds of the anchor and of the summed lines, the whole Uehling contribution between a run and its
Uehling-free twin, the changes of a one-step refinement of each numerical setting, the comparison with
an independent solver where ``numerical_check.csv`` carries one, and the legacy ``unc``/``u<n>`` cell
under the name of what it is. No total uncertainty is produced, no tolerance moves and no central
value moves.

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
NUMERICAL_CHECK_RELPATH = f"{md.D3_RELDIR}/numerical_check.csv"

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

#: The one-step refinement verdicts: every change within its limit, or not. Observed stability of
#: one step, never an error bound on the table.
ONE_STEP_STABLE = "ONE_STEP_STABLE"
ONE_STEP_UNSTABLE = "ONE_STEP_UNSTABLE"
#: The limit on each one-step change, as a fraction of the value's resolution target: a tenth for the
#: grid step and the two Uehling quadratures, a hundredth for the energy tolerance.
REFINEMENT_LIMITS = {
    "k4": Decimal("0.1"), "tol8": Decimal("0.01"), "nu90": Decimal("0.1"), "ns16": Decimal("0.1"),
}
#: The fraction of a compared nuclide's smallest printed uncertainty that is its resolution target.
LINE_TARGET_FRACTION = Decimal("0.1")
#: A declared engineering target for an emitted binding: the larger of an absolute and a relative
#: value, neither an experimental uncertainty.
BINDING_TARGET_ABSOLUTE_KEV = Decimal("1e-6")
BINDING_TARGET_RELATIVE = Decimal("1e-6")


def _text(value: Decimal) -> str:
    """A decimal in fixed notation without trailing zeros (``0`` for zero)."""
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


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
# resolution targets, rounding bounds and the one-step refinement
# --------------------------------------------------------------------------------------------


def nuclide_target_kev(cells: tuple[md.Cell, ...], key: tuple[int, int]) -> Decimal | None:
    """A tenth of the smallest printed uncertainty among the nuclide's gated and centroid cells, in
    keV; None for a nuclide no such cell names."""
    sigmas = [Decimal(c.unc_kev) for c in cells if c.nuclide == key and (c.gated or c.reason == "centroid")]
    if not sigmas:
        return None
    with localcontext() as context:
        context.prec = md._PRECISION
        return LINE_TARGET_FRACTION * min(sigmas)


def binding_target_kev(central: Decimal | str) -> Decimal:
    with localcontext() as context:
        context.prec = md._PRECISION
        return max(BINDING_TARGET_ABSOLUTE_KEV, BINDING_TARGET_RELATIVE * abs(Decimal(central)))


def resolution_target_kev(cells: tuple[md.Cell, ...], key: tuple[int, int], quantity: str,
                          value_kev: Decimal | str) -> Decimal:
    """The resolution target of one value: a printed line takes the nuclide target; ``K``, ``e<n>``
    and every shell difference the smaller of the nuclide target and the engineering target; a
    nuclide without a compared cell the engineering target alone."""
    nuclide = nuclide_target_kev(cells, key)
    if quantity.startswith("line:") and nuclide is not None:
        return nuclide
    engineering = binding_target_kev(value_kev)
    return engineering if nuclide is None else min(nuclide, engineering)


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


def line_rounding_bound_kev(quantity: str, rho_line_ev: Decimal) -> Decimal:
    """One line's rounding per line on the derivation path, in keV: K1's path for ``K``, and for
    ``e<n>`` the (2j+1)-weighted sum of its two circular states' paths -- absolute coefficients."""
    path = path_lengths()
    with localcontext() as context:
        context.prec = md._PRECISION
        unit = rho_line_ev / 1000
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


def refinement_verdict(changes: dict[str, Decimal], target: Decimal) -> str:
    """ONE_STEP_STABLE when every one-step change lies within its REFINEMENT_LIMITS fraction of the
    target, else ONE_STEP_UNSTABLE."""
    with localcontext() as context:
        context.prec = md._PRECISION
        ok = all(abs(change) <= REFINEMENT_LIMITS[tag] * target for tag, change in changes.items())
    return ONE_STEP_STABLE if ok else ONE_STEP_UNSTABLE


def _changes_text(changes: dict[str, Decimal]) -> str:
    return ";".join(f"{tag}={_text(value)}" for tag, value in changes.items())


# --------------------------------------------------------------------------------------------
# the independent check of the numerical solution, as committed
# --------------------------------------------------------------------------------------------

NUMERICAL_CHECK_COLUMNS = (
    "set", "Z", "A", "kind", "uehling", "quantity", "Q_eV", "Q_ref_eV", "b_ref_eV", "b_num_eV", "target_eV",
    "verdict", "bue_1_eV", "run", "input_sha256", "probe_c_bohr", "probe_t_bohr",
)
NUMERICAL_CHECK_SETS = ("pilot_D", "pilot_RA", "pilot_RB", "audit")


@dataclass(frozen=True)
class CheckRow:
    set: str
    z: int
    a: int
    uehling: bool
    quantity: str
    q_ev: str
    q_ref_ev: str
    b_ref_ev: str
    b_num_ev: str
    target_ev: str
    verdict: str
    bue_1_ev: str
    run: str
    input_sha256: str
    probe_c: str
    probe_t: str


def committed_q_ev(outputs: md.Outputs, run: str, quantity: str) -> Decimal:
    """The value of a checked quantity in eV, from one committed run: ``level:<orbit>`` its derived
    binding energy, ``line:<line>`` its printed line, ``K``/``e<n>`` the emitted binding and ``X-Y`` the
    difference of two emitted bindings."""
    b = md.run_bindings(outputs, run)
    with localcontext() as context:
        context.prec = md._PRECISION
        if quantity.startswith("level:"):
            return b[quantity[len("level:"):]]
        if quantity.startswith("line:"):
            return Decimal(outputs.lines[run][quantity[len("line:"):]])
        if "-" in quantity:
            first, second = quantity.split("-")
            return (quantity_value(b, first) - quantity_value(b, second)) * 1000
        return quantity_value(b, quantity) * 1000


def load_numerical_check(root: Path, outputs: md.Outputs) -> dict[tuple[int, int, bool, str], CheckRow]:
    """Parse ``numerical_check.csv``: rows of the anchor kind, each naming the production run of its
    (Z, A) and switch with that run's committed input digest; ``Q_eV`` the value that run gives
    (``committed_q_ev``) to within half a unit of the last decimal ``Q_eV`` prints; one row per (Z, A,
    switch, quantity); the on- and off-run of one (Z, A) carry the same probed c and t (the pairing of
    the twins)."""
    out: dict[tuple[int, int, bool, str], CheckRow] = {}
    inputs = {row.nuclide: row for row in outputs.inputs}
    for where, r in md.read_rows(Path(root) / NUMERICAL_CHECK_RELPATH, NUMERICAL_CHECK_COLUMNS):
        z, a = md.integer(r["Z"], where, "Z"), md.integer(r["A"], where, "A")
        if r["set"] not in NUMERICAL_CHECK_SETS or r["kind"] != "anchor" or r["uehling"] not in ("on", "off"):
            raise md.CellError(f"{where}: set, kind or uehling outside the file's vocabulary")
        uehling = r["uehling"] == "on"
        run = md.recipe_run_id(inputs[(z, a)], "anchor", uehling)
        if r["run"] != run or r["input_sha256"] != outputs.runs[run].input_sha256:
            raise md.CellError(f"{where}: run and input_sha256 must name the production run {run}")
        if not md._UNSIGNED_DECIMAL.fullmatch(r["Q_eV"]):
            raise md.CellError(f"{where}: Q_eV must be digits '.' digits, got {r['Q_eV']!r}")
        with localcontext() as context:
            context.prec = md._PRECISION
            half_unit = Decimal(5) * Decimal(10) ** (-len(r["Q_eV"].split(".")[1]) - 1)
            try:
                committed = committed_q_ev(outputs, run, r["quantity"])
            except (KeyError, IndexError, ValueError) as error:
                raise md.CellError(f"{where}: quantity {r['quantity']!r} is not one {run} gives") from error
            if abs(Decimal(r["Q_eV"]) - committed) > half_unit:
                raise md.CellError(f"{where}: Q_eV {r['Q_eV']} is not {run}'s committed {r['quantity']} "
                                   f"{committed} eV at the decimals it prints")
        if r["verdict"] != "QUALIFIED" and r["verdict"] != "NUMERICAL_FAIL":
            raise md.CellError(f"{where}: verdict {r['verdict']!r}")
        if bool(r["bue_1_eV"]) != uehling:
            raise md.CellError(f"{where}: B-UE_1 is carried on the Uehling-on rows and only there")
        key = (z, a, uehling, r["quantity"])
        if key in out:
            raise md.DuplicateKeyError(f"{where}: duplicate row {key}")
        out[key] = CheckRow(r["set"], z, a, uehling, r["quantity"], r["Q_eV"], r["Q_ref_eV"], r["b_ref_eV"],
                            r["b_num_eV"], r["target_eV"], r["verdict"], r["bue_1_eV"], run,
                            r["input_sha256"],
                            r["probe_c_bohr"], r["probe_t_bohr"])
    pairs: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for row in out.values():
        pairs.setdefault((row.z, row.a), set()).add((row.probe_c, row.probe_t))
    for nuclide, probed in pairs.items():
        if len(probed) != 1:
            raise md.CellError(f"{nuclide}: the twins carry different probed (c, t): {sorted(probed)}")
    return out


# --------------------------------------------------------------------------------------------
# everything the artifacts read, loaded once
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bundle:
    outputs: md.Outputs
    tables: Tables
    cells: tuple[md.Cell, ...]
    origins: dict[tuple[int, int], md.RadiusOrigin]
    stock: dict[tuple[int, int], tuple[float, ...]]
    kept: list[md.InputRow]
    #: Per kept member: the derived bindings of the anchor runs (both switches), the rsig run and,
    #: where it was made, the rminus run.
    bindings: dict[tuple[int, int], dict[str, dict[str, Decimal]]]
    #: Per kept member and switch: the derived bindings of each one-step refinement.
    refinements: dict[tuple[int, int], dict[bool, dict[str, dict[str, Decimal]]]]
    checks: dict[tuple[int, int, bool, str], CheckRow]

    def row(self, key: tuple[int, int]) -> md.InputRow:
        return next(row for row in self.kept if row.nuclide == key)


def load_bundle(root: Path) -> Bundle:
    """Every committed file the contract reads, cross-checked by its own loader."""
    root = Path(root)
    outputs = md.load_outputs(root)
    tables = Tables(read_table_text(root / KSHELL_RELPATH), read_table_text(root / LEVELS_RELPATH))
    cells, origins = md.load_validation(root / md.CELLS_RELPATH, root / md.ORIGIN_RELPATH)
    stock = md.load_geant4_levels(root / md.GEANT4_LEVELS_RELPATH)
    kept = md.kept_members(outputs)
    bindings: dict[tuple[int, int], dict[str, dict[str, Decimal]]] = {}
    refinements: dict[tuple[int, int], dict[bool, dict[str, dict[str, Decimal]]]] = {}
    for row in kept:
        derived = {
            "anchor": md.run_bindings(outputs, md.recipe_run_id(row, "anchor", True)),
            "anchor_off": md.run_bindings(outputs, md.recipe_run_id(row, "anchor", False)),
            "rsig": md.run_bindings(outputs, md.recipe_run_id(row, "rsig", True)),
        }
        minus = md.recipe_run_id(row, "rminus", True)
        if minus in outputs.runs:
            derived["rminus"] = md.run_bindings(outputs, minus)
        bindings[row.nuclide] = derived
        refinements[row.nuclide] = {
            uehling: {tag: md.run_bindings(outputs, md.recipe_run_id(row, "anchor", uehling, tag))
                      for tag, _settings in md.REFINEMENTS if uehling or tag in md.NOU_REFINEMENTS}
            for uehling in (True, False)
        }
    checks = load_numerical_check(root, outputs)
    return Bundle(outputs, tables, cells, origins, stock, kept, bindings, refinements, checks)


def _bundle(root: Path, bundle: Bundle | None) -> Bundle:
    return bundle if bundle is not None else load_bundle(root)


def refinement_changes(bundle: Bundle, key: tuple[int, int], uehling: bool,
                       quantity: str) -> dict[str, Decimal]:
    """The change of ``K``/``e<n>`` (keV) in each one-step refinement run against the production run."""
    base = bundle.bindings[key]["anchor" if uehling else "anchor_off"]
    with localcontext() as context:
        context.prec = md._PRECISION
        return {tag: quantity_value(b, quantity) - quantity_value(base, quantity)
                for tag, b in bundle.refinements[key][uehling].items()}


def _line_changes(bundle: Bundle, key: tuple[int, int], line: str) -> dict[str, Decimal]:
    """The change of a printed line (keV) in each one-step refinement run with the Uehling potential."""
    row = bundle.row(key)
    base = Decimal(bundle.outputs.lines[md.recipe_run_id(row, "anchor", True)][line])
    with localcontext() as context:
        context.prec = md._PRECISION
        return {tag: (Decimal(bundle.outputs.lines[md.recipe_run_id(row, "anchor", True, tag)][line]) - base)
                / 1000 for tag, _settings in md.REFINEMENTS}


# --------------------------------------------------------------------------------------------
# the projection: every gated line beside the shell difference the cascade emits for it
# --------------------------------------------------------------------------------------------


def stock_kev(levels: tuple[float, ...], lower_n: int, upper_n: int) -> Decimal:
    """The photon the unpatched cascade emits between the two shells: the difference of its two
    level energies taken in doubles, in keV at nine decimals."""
    return md.shell_difference_kev(levels, lower_n, upper_n)


def project_shell_rows(root: Path, bundle: Bundle | None = None) -> list[dict[str, str]]:
    """One row per gated cell, in the transcription's order: the measured line beside the solver
    line the patched MuDirac printed in the anchor run, the shell difference a patched cascade emits
    between the line's two shells, and the shell difference the unpatched cascade emits; residuals
    against the measured value, the frozen band, and the one-step refinement changes of the solver
    line and of the shell difference with their verdicts. The measured value keeps its label; the
    consumer quantity is named ``shell_difference`` on every row."""
    bundle = _bundle(root, bundle)
    if md.SIGMA_CALC != 0:
        raise md.CellError("the frozen band is TOL_FACTOR * unc; SIGMA_CALC is not 0")
    rows: list[dict[str, str]] = []
    for cell in bundle.cells:
        if not cell.gated:
            continue
        key = cell.nuclide
        lower_n, upper_n = md.line_shells(cell.quantity)
        member = bundle.row(key)
        with localcontext() as context:
            context.prec = md._PRECISION
            measured = Decimal(cell.value_kev)
            tol = md.TOL_FACTOR * Decimal(cell.unc_kev)
            anchor_lines = bundle.outputs.lines[md.recipe_run_id(member, "anchor", True)]
            solver = Decimal(anchor_lines[cell.quantity]) / 1000
            consumer = Decimal(bundle.tables.shell(key, lower_n)) - Decimal(bundle.tables.shell(key, upper_n))
            stock = stock_kev(bundle.stock[key], lower_n, upper_n)
            residuals = {name: value - measured for name, value in
                         (("solver", solver), ("consumer", consumer), ("stock", stock))}
            solver_changes = _line_changes(bundle, key, cell.quantity)
            solver_target = resolution_target_kev(bundle.cells, key, f"line:{cell.quantity}", solver)
            lower_q, upper_q = _shell_quantity(lower_n), _shell_quantity(upper_n)
            lower_c = refinement_changes(bundle, key, True, lower_q)
            upper_c = refinement_changes(bundle, key, True, upper_q)
            consumer_changes = {tag: lower_c[tag] - upper_c[tag] for tag in lower_c}
            consumer_target = resolution_target_kev(bundle.cells, key, f"{lower_q}-{upper_q}", consumer)
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
            "solver_numeric_shifts_keV": _changes_text(solver_changes),
            "solver_numeric_qualification": refinement_verdict(solver_changes, solver_target),
            "consumer_numeric_target_keV": _text(consumer_target),
            "consumer_numeric_shifts_keV": _changes_text(consumer_changes),
            "consumer_numeric_qualification": refinement_verdict(consumer_changes, consumer_target),
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
        f"d3 contract: one-step refinement over {len(rows)} gated solver lines: "
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
    "model discrepancy", "nuclear polarization", "self energy", "higher-order vacuum polarization",
    "recoil beyond the reduced mass", "electron screening", "hyperfine structure",
    "nuclear charge distributions other than 2pF", "numerical settings beyond one refinement step",
]
RADIUS_PERTURBATION = "source_rms_sigma"
#: The baseline every value carries: the Uehling potential to all orders (each value is solved with
#: it in the Hamiltonian), so nothing may add a further Uehling term to a value.
BASELINE_INCLUDES_UEHLING = "all_orders"
ASSEMBLY_RULE = ("add no further Uehling, iterated-Uehling or 'Ue' term to a value: the Uehling potential is "
                 "in its Hamiltonian to all orders")
#: What is not computed, and why. The first-order Uehling expectation comes with the independent
#: solution; a record carries it where numerical_check.csv does and says NOT_COMPUTED elsewhere.
NOT_COMPUTED = "NOT_COMPUTED"
NOT_COMPUTED_REASONS = {
    "bue_1": ("the first-order Uehling expectation in the Uehling-free eigenstates is computed by the "
              "independent solver only for the values numerical_check.csv carries; no table and no check "
              "reads it"),
    "shape_exports": ("the Uehling contribution under other nuclear charge shapes is not computed: no table "
                      "and no check reads it, and alone it would omit the larger Coulomb-shape sensitivity"),
}
#: The files the records are computed from, digested together into the header record.
COMPONENT_INPUTS = (
    KSHELL_RELPATH, LEVELS_RELPATH, md.INPUTS_RELPATH, md.RUNS_RELPATH, md.STATES_RELPATH, md.LINES_RELPATH,
    NUMERICAL_CHECK_RELPATH,
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
        "baseline_includes_uehling": BASELINE_INCLUDES_UEHLING, "assembly_rule": ASSEMBLY_RULE,
        "not_computed": dict(NOT_COMPUTED_REASONS),
        "refinement_limits": {k: str(v) for k, v in REFINEMENT_LIMITS.items()},
        "omitted_components": list(OMITTED_COMPONENTS), "total_sigma_keV": None,
        "model_uncertainty": "unknown", "inputs_sha256": inputs_sha256(root),
    }


def response_kev(
    bundle: Bundle, key: tuple[int, int], kind: str, quantity: str, relative: bool
) -> str | None:
    """The signed change of ``quantity`` between the anchor run and the ``kind`` run (Uehling on), from
    the bindings as derived (absolute) or with the outermost circular state subtracted (relative);
    None where the run was not made."""
    derived = bundle.bindings[key]
    if kind not in derived:
        return None
    with localcontext() as context:
        context.prec = md._PRECISION
        moved, base = derived[kind], derived["anchor"]
        if relative:
            moved, base = md.relative(moved), md.relative(base)
        return _text(quantity_value(moved, quantity) - quantity_value(base, quantity))


def anchor_rounding_kev(bundle: Bundle, key: tuple[int, int]) -> Decimal:
    """The rounding the anchor carries, in keV: half a unit of the last digit of the printed
    hydrogen-like header of orbit S17, the float resolution, and the printed line R15-S17."""
    run = md.recipe_run_id(bundle.row(key), "anchor", True)
    r = bundle.outputs.runs[run]
    header = bundle.outputs.headers[run][md.orbit(md.N_MAX + 1, True)]
    with localcontext() as context:
        context.prec = md._PRECISION
        floor = md.float_floor_eV(r.r15_total, md.eh_of(r.stage))
        return (md.header_half_unit(header) + floor + md.line_rounding_ev(bundle.outputs, run)) / 1000


def components(root: Path, bundle: Bundle | None = None) -> list[dict[str, object]]:
    """The header record; per kept (Z, A), in order, one member record -- the pairing of its anchor run
    with its Uehling-free twin (both runs and their input digests, and the probed (c, t) where the
    independent check was made), its radius uncertainty, whether its rminus run was made, its correlation
    groups -- then one record per emitted quantity; then one reference record per A = 0 row naming the
    isotope it carries."""
    root = Path(root)
    bundle = _bundle(root, bundle)
    out: list[dict[str, object]] = [header_record(root)]
    for row in bundle.kept:
        key = row.nuclide
        on_run, off_run = md.recipe_run_id(row, "anchor", True), md.recipe_run_id(row, "anchor", False)
        rho = md.line_rounding_ev(bundle.outputs, on_run)
        anchor = anchor_rounding_kev(bundle, key)
        minus = md.recipe_run_id(row, "rminus", True)
        probe = bundle.checks.get((row.z, row.a, True, "K"))
        out.append({
            "record": "member", "Z": row.z, "A": row.a,
            "pairing": {"on_run": on_run, "off_run": off_run,
                        "on_input_sha256": bundle.outputs.runs[on_run].input_sha256,
                        "off_input_sha256": bundle.outputs.runs[off_run].input_sha256,
                        "probe_c_bohr": probe.probe_c if probe else None,
                        "probe_t_bohr": probe.probe_t if probe else None},
            "sigma_rms_fm": row.sigma_rms_fm,
            "radius_minus_status": ("RAN" if minus in bundle.outputs.runs
                                    else "INVALID_PERTURBATION_DOMAIN"),
            "correlation_groups": [f"anchor:{on_run}", f"radius:{row.symbol}{row.a}"],
        })
        for quantity in QUANTITIES:
            central = bundle.tables.central(key, quantity)
            on_changes = refinement_changes(bundle, key, True, quantity)
            off_changes = refinement_changes(bundle, key, False, quantity)
            with localcontext() as context:
                context.prec = md._PRECISION
                q_on = quantity_value(bundle.bindings[key]["anchor"], quantity)
                q_off = quantity_value(bundle.bindings[key]["anchor_off"], quantity)
                bue_np = q_on - q_off
                target = resolution_target_kev(bundle.cells, key, quantity, q_on)
                target_off = resolution_target_kev(bundle.cells, key, quantity, q_off)
            check = bundle.checks.get((row.z, row.a, True, quantity))
            out.append({
                "Z": row.z, "A": row.a, "quantity": quantity, "central_keV": central,
                "radius_plus_relative_response_keV": response_kev(bundle, key, "rsig", quantity, True),
                "radius_minus_relative_response_keV": response_kev(bundle, key, "rminus", quantity, True),
                "radius_plus_absolute_response_keV": response_kev(bundle, key, "rsig", quantity, False),
                "radius_minus_absolute_response_keV": response_kev(bundle, key, "rminus", quantity, False),
                "anchor_rounding_bound_keV": _text(anchor),
                "anchor_coefficient": "1",
                "line_rounding_bound_keV": _text(line_rounding_bound_kev(quantity, rho)),
                "bue_np_keV": _text(bue_np),
                "bue_1_keV": (_text(Decimal(check.bue_1_ev) / 1000) if check is not None else NOT_COMPUTED),
                "refinement_changes_keV": {tag: _text(v) for tag, v in on_changes.items()},
                "noU_refinement_changes_keV": {tag: _text(v) for tag, v in off_changes.items()},
                "refinement_target_keV": _text(target),
                "refinement_verdict": refinement_verdict(on_changes, target),
                "noU_refinement_verdict": refinement_verdict(off_changes, target_off),
                "certificate": None if check is None else {
                    "set": check.set, "verdict": check.verdict, "Q_keV": _text(Decimal(check.q_ev) / 1000),
                    "Q_ref_keV": _text(Decimal(check.q_ref_ev) / 1000),
                    "b_num_keV": _text(Decimal(check.b_num_ev) / 1000),
                    "target_keV": _text(Decimal(check.target_ev) / 1000),
                },
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
    certified = [r for r in records if r["certificate"] is not None]
    return (f"d3 contract: one-step refinement over {len(records)} components: "
            + _histogram([str(r["refinement_verdict"]) for r in records])
            + "; Uehling-free twins: " + _histogram([str(r["noU_refinement_verdict"]) for r in records])
            + f"; independent check on {len(certified)}: "
            + _histogram([str(r["certificate"]["verdict"]) for r in certified]))  # type: ignore[index]
