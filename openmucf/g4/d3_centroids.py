"""Retrospective shell-difference comparisons derived from committed D3 records."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path

from openmucf.g4 import d3_contract as contract
from openmucf.g4.sources import mudirac130 as md

CENTROIDS_RELPATH = f"{md.D3_RELDIR}/consumer_centroids.csv"
MARGINS_RELPATH = f"{md.D3_RELDIR}/centroid_margins.csv"
RATIOS_RELPATH = f"{md.D3_RELDIR}/intensity_ratios.csv"
RATIO_COLUMNS = ("source", "Z", "A", "quantity", "ratio", "ratio_unc", "label", "target", "locator",
                 "copy_read")
CENTROID_COLUMNS = (
    "cohort", "source", "Z", "A", "locator", "comparator", "representative", "c_keV", "sigma_keV",
    "sigma_max_keV", "sigma_min_keV", "sigma_0_keV", "q_keV", "s_keV", "r_d3_keV", "r_stock_keV",
    "margin_keV", "stock_reason", "screen", "dU_keV", "dgrid_keV", "dnum_keV", "u_certified",
    "screen_at_reference", "npol_keV", "npol_l2_keV", "npol_l3_keV", "ratio", "ratio_unc",
    "ratio_source", "ratio_locator", "ratio_label", "illustrative_c_keV", "illustrative_q_minus_c_keV",
    "excluded_gated", "two_p_rows",
)
MARGIN_COLUMNS = ("cohort", "source", "isotopes", "margin_min_keV", "margin_median_keV",
                  "margin_max_keV", "max_abs_r_d3_keV", "max_abs_r_d3_Z", "max_abs_r_d3_A")
CENTROID_TABLE_COLUMNS = (
    ("cohort", "cohort"), ("source", "source"), ("Z", "Z"), ("A", "A"),
    ("value (keV)", "c_keV"), ("sigma (keV)", "sigma_keV"),
    ("sigma min (keV)", "sigma_min_keV"), ("sigma max (keV)", "sigma_max_keV"),
    ("shell difference (keV)", "q_keV"), ("residual (keV)", "r_d3_keV"),
    ("unpatched residual (keV)", "r_stock_keV"), ("margin (keV)", "margin_keV"),
    ("screen", "screen"), ("dnum (keV)", "dnum_keV"), ("u_certified", "u_certified"),
    ("screen at reference", "screen_at_reference"),
)
MARGIN_TABLE_COLUMNS = (
    ("cohort", "cohort"), ("source", "source"), ("isotopes", "isotopes"),
    ("margin min (keV)", "margin_min_keV"), ("margin median (keV)", "margin_median_keV"),
    ("margin max (keV)", "margin_max_keV"), ("largest residual (keV)", "max_abs_r_d3_keV"),
    ("Z", "max_abs_r_d3_Z"), ("A", "max_abs_r_d3_A"),
)
_UNIT = Decimal("1e-9")


def _within_unit(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= _UNIT


def _text(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 50
        return format(value.quantize(_UNIT, rounding=ROUND_HALF_EVEN), ".9f")


def _csv(columns: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([row.get(column, "") for column in columns] for row in rows)
    return out.getvalue().encode("ascii")


def load_intensity_ratios(path: Path, cells: tuple[md.Cell, ...]) -> dict[tuple[int, int], dict[str, str]]:
    """Only positive printed ratios for complete constructed isotopes are admitted."""
    eligible = {l2.nuclide for l2, _l3 in md.doublet_determinations(cells)}
    out: dict[tuple[int, int], dict[str, str]] = {}
    for where, row in md.read_rows(path, RATIO_COLUMNS):
        key = (md.integer(row["Z"], where, "Z"), md.integer(row["A"], where, "A"))
        if key not in eligible:
            raise md.CellError(f"{where}: ratio isotope is not constructed")
        if key in out:
            raise md.DuplicateKeyError(f"{where}: duplicate ratio isotope {key}")
        for column in ("ratio", "ratio_unc"):
            if not md._UNSIGNED_DECIMAL.fullmatch(row[column]) or Decimal(row[column]) <= 0:
                raise md.CellError(f"{where}: {column} must be a positive printed decimal")
        out[key] = row
    return out


def _screen(residual: Decimal, sigma: Decimal, minimum: Decimal | None = None) -> str:
    if minimum is None:
        return "inside" if abs(residual) <= md.TOL_FACTOR * sigma else "outside"
    if abs(residual) > md.TOL_FACTOR * sigma:
        return "outside"
    if abs(residual) <= md.TOL_FACTOR * minimum:
        return "inside"
    return "correlation-dependent"


def onset_and_certificate(values: dict[int, Decimal], target: Decimal) -> tuple[int | None, int | None]:
    """First unclean or inconsistent U, and the earliest clean doubling certificate below it."""
    with localcontext() as context:
        context.prec = 50
        return _onset_and_certificate(values, target)


def _onset_and_certificate(values: dict[int, Decimal], target: Decimal) -> tuple[int | None, int | None]:
    floor = target / 200
    onset: int | None = None
    reference_sign: int | None = None
    previous: Decimal | None = None
    for u in range(100, 2401, 100):
        if u not in values:
            onset = min(onset, u) if onset is not None else u
            continue
        if u == 100 or u - 100 not in values:
            continue
        step = values[u] - values[u - 100]
        if u == 200 and abs(step) >= floor:
            reference_sign = 1 if step > 0 else -1
        opposite = (abs(step) >= floor and reference_sign is not None and
                    (1 if step > 0 else -1) != reference_sign)
        growth = u >= 300 and previous is not None and abs(step) - abs(previous) >= floor
        if opposite or growth:
            onset = min(onset, u) if onset is not None else u
        previous = step
    for u in (50, 150, 250, 350, 450, 550):
        if u not in values:
            onset = min(onset, u) if onset is not None else u
    certified = None
    for u in range(100, 1201, 100):
        if onset is not None and 2 * u >= onset:
            continue
        if not all(k in values for k in (u // 2, u, 2 * u)):
            continue
        low = values[u] - values[u // 2]
        high = values[2 * u] - values[u]
        if abs(high) < target and abs(high) <= abs(low):
            certified = u
            break
    return onset, certified


def _settings_q(settings: md.SettingsOutputs) -> dict[tuple[int, int, str], Decimal]:
    result: dict[tuple[int, int, str], Decimal] = {}
    for run in settings.runs.values():
        if not run.clean:
            continue
        try:
            binding = md.derive_bindings(run.run, settings.headers.get(run.run, {}),
                                         settings.lines.get(run.run, {}))
            k, outer = md.quantities(binding)
        except Exception:
            continue
        result[(run.z, run.a, run.setting)] = k - outer[0]
    return result


def _targets(cells: tuple[md.Cell, ...]) -> dict[tuple[int, int], Decimal]:
    targets = {cell.nuclide: Decimal(cell.unc_kev) / 10 for cell in cells if cell.reason == "centroid"}
    for l2, l3 in md.doublet_determinations(cells):
        with localcontext() as context:
            context.prec = 50
            sigma = ((Decimal(l2.unc_kev) / 3) ** 2 + (2 * Decimal(l3.unc_kev) / 3) ** 2).sqrt()
            target = sigma / 10
        targets[l2.nuclide] = min(targets.get(l2.nuclide, target), target)
    return targets


def numerical_components(root: Path, cells: tuple[md.Cell, ...]) -> tuple[
    dict[tuple[int, int], tuple[Decimal, Decimal, Decimal, int | None, int | None]], int]:
    with localcontext() as context:
        context.prec = 50
        return _numerical_components(root, cells)


def _numerical_components(root: Path, cells: tuple[md.Cell, ...]) -> tuple[
    dict[tuple[int, int], tuple[Decimal, Decimal, Decimal, int | None, int | None]], int]:
    settings = md.load_settings_outputs(root, cells)
    q = _settings_q(settings)
    targets = _targets(cells)
    result: dict[tuple[int, int], tuple[Decimal, Decimal, Decimal, int | None, int | None]] = {}
    for z, a in md.settings_nuclides(cells):
        ladder = {u: q[(z, a, md.setting_id(0, u))] for u in md.SETTINGS_UEHLING_G0
                  if (z, a, md.setting_id(0, u)) in q}
        onset, certified = onset_and_certificate(ladder, targets[(z, a)])
        if onset is not None and onset <= 1000:
            raise md.CellError(f"Z={z} A={a}: reference U at or above onset {onset}")
        if (z, a, md.setting_id(2, 1000)) not in q:
            raise md.CellError(f"Z={z} A={a}: fine-grid reference is unclean")
        try:
            low, mid = ladder[100], ladder[1000]
        except KeyError as exc:
            raise md.CellError(f"Z={z} A={a}: numerical endpoint is unclean") from exc
        fine = q[(z, a, md.setting_id(2, 1000))]
        du, dgrid = mid - low, fine - mid
        result[(z, a)] = du, dgrid, du + dgrid, certified, onset
    return result, sum((run.z, run.a, run.setting) not in q for run in settings.runs.values()
                       if run.setting.startswith("g0"))


def _illustrative(l2: md.Cell, l3: md.Cell, ratio: dict[str, str], q: Decimal) -> tuple[str, str]:
    r, unc = Decimal(ratio["ratio"]), Decimal(ratio["ratio_unc"])
    first, second = Decimal(l2.value_kev), Decimal(l3.value_kev)
    values = [first + x / (1 + x) * (second - first) for x in (r - unc, r, r + unc)]
    return ";".join(_text(v) for v in values), ";".join(_text(q - v) for v in values)


def centroid_rows(root: Path) -> list[dict[str, str]]:
    """The labelled, complete-doublet and excluded cohorts in the frozen order."""
    root = Path(root)
    cells = md.load_cells(root / md.CELLS_RELPATH)
    ratios = load_intensity_ratios(root / RATIOS_RELPATH, cells)
    components, _unclean = numerical_components(root, cells)
    tables = contract.Tables(contract.read_table_text(root / contract.KSHELL_RELPATH),
                             contract.read_table_text(root / contract.LEVELS_RELPATH))
    stock = md.load_geant4_levels(root / md.GEANT4_LEVELS_RELPATH)
    if md.SIGMA_CALC != 0:
        raise md.CellError("the centroid screen freezes SIGMA_CALC at zero")
    if not set(md.gated_nuclides(cells)) <= set(stock) or not set(stock) <= set(md.stock_nuclides(cells)):
        raise md.CellError("stock levels omit a gated key or name an unlisted key")
    projection = {
        (r["source"], int(r["Z"]), int(r["A"]), r["locator"], r["quantity"]): r
        for _where, r in md.read_rows(root / contract.PROJECTION_RELPATH, contract.PROJECTION_COLUMNS)
    }
    rows: list[dict[str, str]] = []

    def common(cohort: str, source: str, z: int, a: int, locator: str, c: Decimal,
               sigma: Decimal, sigma_min: Decimal | None, dnum: Decimal) -> dict[str, str]:
        key = z, a
        q = Decimal(tables.shell(key, 1)) - Decimal(tables.shell(key, 2))
        stock_value = contract.stock_kev(stock[key], 1, 2) if key in stock else None
        residual = q - c
        stock_residual = stock_value - c if stock_value is not None else None
        du, dgrid, _shift, certified, _onset = components[key]
        row = dict.fromkeys(CENTROID_COLUMNS, "")
        row.update(cohort=cohort, source=source, Z=str(z), A=str(a), locator=locator,
                   c_keV=_text(c), q_keV=_text(q), sigma_keV=_text(sigma) if sigma_min is None else "",
                   s_keV=_text(stock_value) if stock_value is not None else "",
                   r_d3_keV=_text(residual),
                   r_stock_keV=_text(stock_residual) if stock_residual is not None else "",
                   margin_keV=(_text(abs(stock_residual) - abs(residual))
                               if stock_residual is not None else ""),
                   stock_reason="" if stock_value is not None else "no C line in the unpatched harvest",
                   screen=_screen(residual, sigma, sigma_min), dU_keV=_text(du), dgrid_keV=_text(dgrid),
                   dnum_keV=_text(dnum), u_certified=str(certified) if certified is not None else "none",
                   screen_at_reference=_screen(residual + dnum, sigma, sigma_min))
        row["_residual_raw"] = str(residual)
        row["_margin_raw"] = str(abs(stock_residual) - abs(residual)) if stock_residual is not None else ""
        return row

    with localcontext() as context:
        context.prec = 50
        for cell in sorted((c for c in cells if c.reason == "centroid"), key=lambda c: c.nuclide):
            du, dgrid, dnum, certified, onset = components[cell.nuclide]
            row = common("labelled", cell.source, cell.z, cell.a, cell.locator,
                         Decimal(cell.value_kev), Decimal(cell.unc_kev), None, dnum)
            row.update(comparator="center of gravity", representative="true", npol_keV=cell.npol_kev)
            rows.append(row)
        grouped: dict[tuple[str, int, int], list[tuple[dict[str, str], Decimal]]] = defaultdict(list)
        for l2, l3 in md.doublet_determinations(cells):
            key = l2.nuclide
            c = (Decimal(l2.value_kev) + 2 * Decimal(l3.value_kev)) / 3
            u2, u3 = Decimal(l2.unc_kev), Decimal(l3.unc_kev)
            sigma_max = (u2 + 2 * u3) / 3
            sigma_min = abs(u2 - 2 * u3) / 3
            sigma_zero = ((u2 / 3) ** 2 + (2 * u3 / 3) ** 2).sqrt()
            dnum = components[key][2]
            row = common("constructed", l2.source, l2.z, l2.a, l2.locator, c, sigma_max,
                         sigma_min, dnum)
            row.update(comparator="degeneracy-convention centroid", sigma_max_keV=_text(sigma_max),
                       sigma_min_keV=_text(sigma_min), sigma_0_keV=_text(sigma_zero),
                       npol_l2_keV=l2.npol_kev, npol_l3_keV=l3.npol_kev)
            q = Decimal(tables.shell(key, 1)) - Decimal(tables.shell(key, 2))
            try:
                pair = [projection[(c.source, c.z, c.a, c.locator, c.quantity)] for c in (l2, l3)]
            except KeyError as exc:
                raise md.CellError(f"{contract.PROJECTION_RELPATH}: missing {key}") from exc
            if any(Decimal(p["consumer_keV"]) != q for p in pair):
                raise md.CellError(f"{key}: shell projection disagrees with the consumer quantity")
            solver = (Decimal(pair[0]["solver_keV"]) + 2 * Decimal(pair[1]["solver_keV"])) / 3
            if not _within_unit(q, solver):
                raise md.CellError(f"{key}: weighted solver lines disagree with the shell difference")
            if any(Decimal(p["stock_keV"]) != contract.stock_kev(stock[key], 1, 2) for p in pair):
                raise md.CellError(f"{key}: stock level difference disagrees with the shell projection")
            if key in ratios:
                ratio = ratios[key]
                illustrative, residuals = _illustrative(l2, l3, ratio, q)
                row.update(ratio=ratio["ratio"], ratio_unc=ratio["ratio_unc"], ratio_source=ratio["source"],
                           ratio_locator=ratio["locator"], ratio_label=ratio["label"],
                           illustrative_c_keV=illustrative, illustrative_q_minus_c_keV=residuals)
            grouped[(l2.source, l2.z, l2.a)].append((row, sigma_max))
            rows.append(row)
        for determinations in grouped.values():
            representative = min(determinations, key=lambda item: item[1])[0]
            representative["representative"] = "true"
        excluded = {(c.source, c.z, c.a) for c in cells
                    if c.gated and c.quantity in ("K1-L2", "K1-L3") and
                    (c.source, c.z, c.a) not in grouped}
        for source, z, a in sorted(excluded, key=lambda key: (key[1], key[2])):
            members = [c for c in cells if (c.source, c.z, c.a) == (source, z, a)]
            row = dict.fromkeys(CENTROID_COLUMNS, "")
            row.update(cohort="excluded", source=source, Z=str(z), A=str(a),
                       excluded_gated=";".join(c.quantity for c in members
                                                 if c.gated and c.quantity in ("K1-L2", "K1-L3")),
                        two_p_rows=_two_p_rows(members))
            rows.append(row)
    return rows


def _two_p_rows(members: list[md.Cell]) -> str:
    return ";".join(f"{c.transition}:{c.reason or 'gated'}" for c in members
                    if c.transition.startswith("2p") and c.reason != "centroid")


def margin_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups = (("labelled", "Fricke1995"), ("constructed", "Fricke1995"),
              ("constructed", "Saito2025"))
    out: list[dict[str, str]] = []
    for cohort, source in groups:
        members = [r for r in rows if r["cohort"] == cohort and r["source"] == source]
        margins = sorted(Decimal(r.get("_margin_raw") or r["margin_keV"]) for r in members
                         if r["representative"] == "true" and r["margin_keV"])
        if not margins:
            raise md.CellError(f"{cohort} {source}: no stock margins")
        middle = len(margins) // 2
        median = (_text(margins[middle]) if len(margins) % 2 else
                  f"{_text(margins[middle - 1])}..{_text(margins[middle])}")
        largest = max(members, key=lambda r: abs(Decimal(r.get("_residual_raw") or r["r_d3_keV"])))
        out.append(dict(cohort=cohort, source=source, isotopes=str(len(margins)),
                        margin_min_keV=_text(margins[0]), margin_median_keV=median,
                        margin_max_keV=_text(margins[-1]),
                        max_abs_r_d3_keV=_text(abs(Decimal(largest.get("_residual_raw") or
                                                          largest["r_d3_keV"]))),
                        max_abs_r_d3_Z=largest["Z"], max_abs_r_d3_A=largest["A"]))
    return out


def render_centroids(root: Path) -> bytes:
    return _csv(CENTROID_COLUMNS, centroid_rows(root))


def render_margins(root: Path) -> bytes:
    return _csv(MARGIN_COLUMNS, margin_rows(centroid_rows(root)))


def _render_table(rows: list[dict[str, str]], columns: tuple[tuple[str, str], ...]) -> str:
    lines = ["| " + " | ".join(title for title, _ in columns) + " |",
             "|" + "---|" * len(columns)]
    lines.extend("| " + " | ".join(row[column] for _, column in columns) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def render_centroids_table(rows: list[dict[str, str]]) -> str:
    """Copy compared cells into the document's centroid table."""
    return _render_table([row for row in rows if row["cohort"] != "excluded"], CENTROID_TABLE_COLUMNS)


def render_margins_table(rows: list[dict[str, str]]) -> str:
    """Copy margin cells into the document's summary table."""
    return _render_table(rows, MARGIN_TABLE_COLUMNS)


def summary_lines(root: Path) -> list[str]:
    rows = centroid_rows(root)
    compared = [r for r in rows if r["cohort"] != "excluded"]
    labelled = [r for r in compared if r["cohort"] == "labelled"]
    constructed = [r for r in compared if r["cohort"] == "constructed"]
    excluded = [r for r in rows if r["cohort"] == "excluded"]
    components, _unclean = numerical_components(root, md.load_cells(root / md.CELLS_RELPATH))
    onsets = [(onset, z, a) for (z, a), (_du, _dg, _dn, _cert, onset) in components.items()
              if onset is not None]
    first = min(onsets)
    above = sum(abs(Decimal(r["dnum_keV"])) >
                Decimal(r["sigma_keV"] or r["sigma_max_keV"]) / 10 for r in compared)
    lines = [
        f"d3 centroids: labelled rows {len(labelled)} inside "
        f"{sum(r['screen'] == 'inside' for r in labelled)} "
        f"outside {sum(r['screen'] == 'outside' for r in labelled)}; at reference inside "
        f"{sum(r['screen_at_reference'] == 'inside' for r in labelled)} outside "
        f"{sum(r['screen_at_reference'] == 'outside' for r in labelled)}",
        f"d3 centroids: constructed rows {len(constructed)} isotopes "
        f"{len({(r['Z'], r['A']) for r in constructed})} inside "
        f"{sum(r['screen'] == 'inside' for r in constructed)} outside "
        f"{sum(r['screen'] == 'outside' for r in constructed)} correlation-dependent "
        f"{sum(r['screen'] == 'correlation-dependent' for r in constructed)}; at reference inside "
        f"{sum(r['screen_at_reference'] == 'inside' for r in constructed)} outside "
        f"{sum(r['screen_at_reference'] == 'outside' for r in constructed)} correlation-dependent "
        f"{sum(r['screen_at_reference'] == 'correlation-dependent' for r in constructed)}",
        f"d3 centroids: excluded {len(excluded)}: " + "; ".join(
            f"{r['source']} {r['Z']}-{r['A']}" for r in excluded),
        f"d3 centroids: nuclides {len(components)} u_certified none "
        f"{sum(cert is None for _du, _dg, _dn, cert, _onset in components.values())}; "
        f"smallest G0 onset {first[0]} at Z={first[1]} A={first[2]}; "
        f"|dnum| above a tenth of the largest sigma on {above} of {len(compared)} rows",
    ]
    for row in margin_rows(rows):
        lines.append(f"d3 centroids: margin {row['cohort']} {row['source']} isotopes {row['isotopes']} "
                     f"min {row['margin_min_keV']} median {row['margin_median_keV']} "
                     f"max {row['margin_max_keV']}; max |r_D3| {row['max_abs_r_d3_keV']} "
                     f"at Z={row['max_abs_r_d3_Z']} A={row['max_abs_r_d3_A']}")
    ratios = load_intensity_ratios(root / RATIOS_RELPATH, md.load_cells(root / md.CELLS_RELPATH))
    lines.append(f"d3 centroids: intensity ratio rows {len(ratios)}")
    return lines
