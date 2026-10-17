"""Each change of a compared D3 energy from the v1.3.0 tables to the ``mudirac130p1`` tables, by cause.

Every compared nuclide was run once more at each stage between the input of the v1.3.0 tables and the
production input (the ``stages`` batch of ``mudirac_runs.csv``): the v1.3.0 input exactly (``shipped``); the
numerics repair (``numerics``); the one 2pF density for the Coulomb and Uehling potentials together with the
trapezoidal Uehling quadrature (``density``); the 2pF parameter c solved from the intended rms radius
(``rms``); the nuclear mass in the reduced mass (``mass``); and the production run, which adds the CODATA 2022
Hartree energy. ``change_attribution.csv`` gives, for every gated line, the K-L shell difference of every
nuclide the centroid comparison reads, and K and e2 of every compared nuclide, the value at the first stage,
the change at each later one and the value at the last; the changes sum to the difference exactly.

The second change is named ``delta_density_and_quadrature``: it moves the density of the Uehling potential
and its quadrature at once (Gauss-Legendre in theta at 64 nodes for the uniform sphere, to the trapezoidal
rule at 60 steps and 12 shell nodes per panel for the 2pF density), so it is a combined implementation
change and not the physical contribution of the density alone.

``history/validation.mudirac130.csv`` is the comparison file of release v1.3.0, byte for byte.

Standard library plus ``openmucf.g4.sources.mudirac130`` only.
"""

from __future__ import annotations

import csv
import hashlib
import io
from decimal import Decimal, localcontext
from pathlib import Path

from openmucf.g4.sources import mudirac130 as md

ATTRIBUTION_RELPATH = f"{md.D3_RELDIR}/change_attribution.csv"
HISTORY_RELPATH = f"{md.D3_RELDIR}/history/validation.mudirac130.csv"
#: The git blob id of release v1.3.0's ``data/g4/d3/validation.csv``.
HISTORY_BLOB = "c16d0085317c3ae0e1b2fc45e394fffc9aa9084c"
#: The stages in order, the production run last; each change is named for the stage it leads to.
STAGE_RUNS = ("shipped", "numerics", "density", "rms", "mass", "recipe")
DELTAS = ("delta_numerics_keV", "delta_density_and_quadrature_keV", "delta_rms_keV", "delta_mass_keV",
          "delta_eV_keV")
ATTRIBUTION_COLUMNS = ("Z", "A", "quantity", "shipped_keV", *DELTAS, "new_keV")
#: Every value is carried at this many decimals of a keV before any change is taken, so the changes of a
#: row sum to its difference exactly.
DECIMALS = Decimal("1e-18")


def history_problems(root: Path) -> list[str]:
    """The history copy is release v1.3.0's comparison file: its git blob id is HISTORY_BLOB."""
    payload = (Path(root) / HISTORY_RELPATH).read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(payload) + payload, usedforsecurity=False).hexdigest()
    return [] if blob == HISTORY_BLOB else [f"{HISTORY_RELPATH} is blob {blob}, not {HISTORY_BLOB}"]


def _stage_run(row: md.InputRow, stage: str) -> str:
    return md.recipe_run_id(row, "anchor", True) if stage == "recipe" else md.run_id(row, stage)


def _value(out: md.Outputs, run: str, quantity: str) -> Decimal:
    """A quantity of one run in keV: a printed line ``line:<line>``, an emitted binding ``K``/``e<n>``, or the
    difference ``K-e2`` of two."""
    with localcontext() as context:
        context.prec = md._PRECISION
        if quantity.startswith("line:"):
            value = Decimal(out.lines[run][quantity[len("line:"):]]) / 1000
        else:
            k, levels = md.quantities(md.run_bindings(out, run))
            parts = [k if name == "K" else levels[int(name[1:]) - 2] for name in quantity.split("-")]
            value = parts[0] - parts[1] if len(parts) == 2 else parts[0]
        return value.quantize(DECIMALS)


def _text(value: Decimal) -> str:
    text = format(value, "f").rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def quantities(cells: tuple[md.Cell, ...], key: tuple[int, int]) -> list[str]:
    """The quantities of a compared nuclide, in order: its gated lines as the transcription first names
    them, K-e2 where the centroid comparison reads it, then K and e2."""
    out: list[str] = []
    for cell in cells:
        if cell.gated and cell.nuclide == key and f"line:{cell.quantity}" not in out:
            out.append(f"line:{cell.quantity}")
    centroid = set(md.centroid_nuclides(cells)) | {l2.nuclide for l2, _l3 in md.doublet_determinations(cells)}
    if key in centroid:
        out.append("K-e2")
    return [*out, "K", "e2"]


def attribution_rows(out: md.Outputs) -> list[dict[str, str]]:
    """One row per compared nuclide and quantity, ascending by (Z, A)."""
    inputs = {row.nuclide: row for row in out.inputs}
    rows = []
    for key in md.compared_nuclides(out.cells):
        row = inputs[key]
        for quantity in quantities(out.cells, key):
            values = [_value(out, _stage_run(row, stage), quantity) for stage in STAGE_RUNS]
            record = {"Z": str(row.z), "A": str(row.a), "quantity": quantity, "shipped_keV": _text(values[0]),
                      "new_keV": _text(values[-1])}
            for name, before, after in zip(DELTAS, values[:-1], values[1:], strict=True):
                record[name] = _text(after - before)
            rows.append(record)
    return rows


def render_attribution(root: Path, out: md.Outputs | None = None) -> bytes:
    """``change_attribution.csv``: LF, ASCII, the header then one row per compared nuclide and quantity."""
    out = out if out is not None else md.load_outputs(Path(root))
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(ATTRIBUTION_COLUMNS)
    for record in attribution_rows(out):
        writer.writerow([record[column] for column in ATTRIBUTION_COLUMNS])
    return buffer.getvalue().encode("ascii")


def summary_lines(root: Path, out: md.Outputs | None = None) -> list[str]:
    """The largest change by cause over the gated lines, and the v1.3.0 and current comparison counts."""
    root = Path(root)
    out = out if out is not None else md.load_outputs(root)
    rows = attribution_rows(out)
    lines = [r for r in rows if r["quantity"].startswith("line:")]
    largest = []
    for name in DELTAS:
        top = lines[0]
        for r in lines:
            if abs(Decimal(r[name])) > abs(Decimal(top[name])):
                top = r
        largest.append(f"{name} {top[name]} (Z={top['Z']} A={top['A']} {top['quantity']})")
    history = list(csv.DictReader(io.StringIO((root / HISTORY_RELPATH).read_bytes().decode("ascii"))))
    current = list(csv.DictReader(io.StringIO((root / md.VALIDATION_RELPATH).read_bytes().decode("ascii"))))

    def within(table: list[dict[str, str]]) -> str:
        gated = [r for r in table if r["gated"] == "true"]
        return f"{sum(r['within'] == 'true' for r in gated)}/{len(gated)}"

    return [
        f"d3 attribution: rows {len(rows)} over {len(md.compared_nuclides(out.cells))} compared nuclides; "
        "largest change on a gated line: " + "; ".join(largest),
        f"d3 attribution: within tolerance v1.3.0 {within(history)}, {md.PROFILE} {within(current)}",
    ]
