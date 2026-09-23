from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

PRINTED_ROWS_RELPATH = "data/g4/d1/suzuki1987_printed_rows.csv"
QUANTITY_ROWS_RELPATH = "data/g4/d1/suzuki1987_quantity_rows.csv"
PARITY_CELLS_RELPATH = "data/g4/d1/suzuki1987_parity_cells.csv"
PREPRINT_DIFF_RELPATH = "data/g4/d1/suzuki1987_preprint_differences.csv"
PROFILE = "suzuki1987"
BIBKEY = "Suzuki1987"
EVALUATION_ID = "suzuki1987-tables-iii-iv"
PRINTED_COLUMNS = (
    "table", "printed_page", "page_row_ordinal", "row_kind", "z_raw", "zeff_raw",
    "zeff_underlined", "element_raw", "mean_life_raw", "rate_raw", "huff_raw",
    "refs_raw", "note", "copy_read",
)
ROW_KINDS = frozenset(("caption", "header", "data", "positive_muon", "note", "footnote"))
QUANTITY_COLUMNS = (
    "table", "printed_page", "page_row_ordinal", "Z", "z_from", "target_label", "label_from",
    "target_basis", "basis_rule", "mean_life_ns", "mean_life_unc_ns", "rate", "rate_unc",
    "rate_unc_plus", "rate_unc_minus", "rate_exp10", "rate_unit", "rate_parenthesized",
    "huff_factor", "huff_from", "refs", "own_measurement", "copy_read",
)
_DECIMAL = r"[0-9]+(?:\.[0-9]+)?"
_SYMMETRIC = re.compile(rf"({_DECIMAL})\+\-({_DECIMAL})")
_ASYMMETRIC = re.compile(rf"({_DECIMAL})\+({_DECIMAL})\(-({_DECIMAL})\)")
_LABEL = re.compile(r"(?:(nat|R|[0-9]+(?:\.[0-9]+)?)?)([A-Z][a-z]?)(?:\^[a-z])?")


class Suzuki1987Error(RuntimeError):
    pass


@dataclass(frozen=True)
class PrintedRow:
    table: str
    printed_page: int
    page_row_ordinal: int
    row_kind: str
    z_raw: str
    zeff_raw: str
    zeff_underlined: str
    element_raw: str
    mean_life_raw: str
    rate_raw: str
    huff_raw: str
    refs_raw: str
    note: str
    copy_read: str

    @property
    def locator(self) -> str:
        return f"Table {self.table} p.{self.printed_page} row {self.page_row_ordinal}"


def load_printed_rows(path: Path) -> tuple[PrintedRow, ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Suzuki1987Error(f"{path} line 0: {exc}") from None
    if b"\r" in raw:
        line = raw[:raw.index(b"\r")].count(b"\n") + 1
        raise Suzuki1987Error(f"{path} line {line}: CR is forbidden")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        line = raw[:exc.start].count(b"\n") + 1
        raise Suzuki1987Error(f"{path} line {line}: non-ASCII byte") from None
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != PRINTED_COLUMNS:
        raise Suzuki1987Error(f"{path} line 1: header differs from printed-row schema")
    rows: list[PrintedRow] = []
    next_ordinal: dict[tuple[str, int], int] = {}
    for line, record in enumerate(reader, 2):
        where = f"{path} line {line}"
        if None in record or any(value is None for value in record.values()):
            raise Suzuki1987Error(f"{where}: wrong cell count")
        if record["table"] not in ("III", "IV"):
            raise Suzuki1987Error(f"{where}: table must be III or IV")
        try:
            page = int(record["printed_page"])
            ordinal = int(record["page_row_ordinal"])
        except ValueError:
            raise Suzuki1987Error(f"{where}: page and ordinal must be integers") from None
        if page <= 0 or ordinal <= 0:
            raise Suzuki1987Error(f"{where}: page and ordinal must be positive")
        group = (record["table"], page)
        expected = next_ordinal.get(group, 1)
        if ordinal != expected:
            raise Suzuki1987Error(f"{where}: ordinal {ordinal} follows {expected - 1}")
        next_ordinal[group] = expected + 1
        if record["row_kind"] not in ROW_KINDS:
            raise Suzuki1987Error(f"{where}: unrecognized row_kind {record['row_kind']!r}")
        if record["zeff_underlined"] not in ("true", "false", ""):
            raise Suzuki1987Error(f"{where}: zeff_underlined must be true, false, or blank")
        if record["copy_read"] != "published-scan":
            raise Suzuki1987Error(f"{where}: copy_read must be published-scan")
        rows.append(PrintedRow(
            table=record["table"], printed_page=page, page_row_ordinal=ordinal,
            row_kind=record["row_kind"], z_raw=record["z_raw"], zeff_raw=record["zeff_raw"],
            zeff_underlined=record["zeff_underlined"], element_raw=record["element_raw"],
            mean_life_raw=record["mean_life_raw"], rate_raw=record["rate_raw"],
            huff_raw=record["huff_raw"], refs_raw=record["refs_raw"], note=record["note"],
            copy_read=record["copy_read"],
        ))
    if not rows:
        raise Suzuki1987Error(f"{path} line 1: no printed rows")
    return tuple(rows)


def _label_parts(label: str, locator: str) -> tuple[str, str, str]:
    clean = label.removesuffix(" (con't)")
    match = _LABEL.fullmatch(clean)
    if match is None:
        raise Suzuki1987Error(f"{locator}: target label {label!r} has no supported basis")
    prefix, symbol = match.groups()
    if prefix is None:
        return clean, symbol, "unspecified"
    if prefix == "nat":
        return clean, symbol, "natural"
    if prefix == "R" or "." in prefix:
        return clean, symbol, "enriched_mixture"
    return clean, symbol, "isotope"


def _basis_rule(label: str, basis: str) -> str:
    if basis == "isotope":
        return "integer-mass-prefix"
    if basis == "natural":
        return "nat-prefix"
    if basis == "enriched_mixture":
        return "R-prefix" if label.startswith("R") else "noninteger-mass-prefix"
    return "bare-symbol"


def _pair(raw: str, locator: str, column: str) -> tuple[str, str]:
    if not raw:
        return "", ""
    match = _SYMMETRIC.fullmatch(raw)
    if match is None:
        raise Suzuki1987Error(f"{locator}: malformed {column} {raw!r}")
    return match.group(1), match.group(2)


def _rate(raw: str, locator: str) -> tuple[str, str, str, str, str, str]:
    if not raw:
        return "", "", "", "", "", "false"
    parenthesized = raw.startswith("(") and raw.endswith(")")
    core = raw[1:-1] if parenthesized else raw
    exponent = "0"
    if "x10^" in core:
        core, exponent = core.rsplit("x10^", 1)
        if not exponent.isdecimal():
            raise Suzuki1987Error(f"{locator}: malformed rate power {raw!r}")
    symmetric = _SYMMETRIC.fullmatch(core)
    asymmetric = _ASYMMETRIC.fullmatch(core)
    if symmetric:
        rate, unc = symmetric.groups()
        return rate, unc, "", "", exponent, str(parenthesized).lower()
    if asymmetric:
        rate, plus, minus = asymmetric.groups()
        return rate, "", plus, f"-{minus}", exponent, str(parenthesized).lower()
    raise Suzuki1987Error(f"{locator}: malformed rate {raw!r}")


def normalize(rows: tuple[PrintedRow, ...]) -> tuple[dict[str, str], ...]:
    units: dict[tuple[str, int], str] = {}
    for row in rows:
        if row.row_kind != "header":
            continue
        if "10^6/s" in row.rate_raw:
            unit = "10^6/s"
        elif "s^-1" in row.rate_raw:
            unit = "s^-1"
        else:
            raise Suzuki1987Error(f"{row.locator}: capture-rate unit absent from header")
        units[(row.table, row.printed_page)] = unit

    result: list[dict[str, str]] = []
    current_z = ""
    z_origin = ""
    group_symbol = ""
    current_label = ""
    label_origin = ""
    huff = ""
    huff_origin = ""
    for row in rows:
        if row.row_kind != "data":
            continue
        locator = row.locator
        if row.element_raw:
            label, symbol, basis = _label_parts(row.element_raw, locator)
        elif current_label:
            label, symbol, basis = _label_parts(current_label, locator)
        else:
            raise Suzuki1987Error(f"{locator}: missing target label")
        if row.z_raw:
            if not row.z_raw.isdecimal():
                raise Suzuki1987Error(f"{locator}: Z is not a printed integer")
            current_z, z_origin, group_symbol = row.z_raw, locator, symbol
            huff = huff_origin = ""
        elif not current_z or symbol != group_symbol:
            raise Suzuki1987Error(f"{locator}: inherited Z crosses an element symbol")
        if row.element_raw:
            if label != current_label:
                huff = huff_origin = ""
            current_label, label_origin = label, locator
        if row.huff_raw:
            huff, huff_origin = row.huff_raw, locator
        mean, mean_unc = _pair(row.mean_life_raw, locator, "mean life")
        rate, unc, plus, minus, exp10, parenthesized = _rate(row.rate_raw, locator)
        page_unit = units.get((row.table, row.printed_page))
        if page_unit is None:
            raise Suzuki1987Error(f"{locator}: page lacks a rate-unit header")
        result.append(dict(zip(QUANTITY_COLUMNS, (
            row.table, str(row.printed_page), str(row.page_row_ordinal), current_z,
            "printed" if row.z_raw else z_origin, label, label_origin, basis,
            _basis_rule(label, basis), mean, mean_unc, rate, unc, plus, minus, exp10, page_unit,
            parenthesized, huff, "printed" if row.huff_raw else huff_origin, row.refs_raw,
            str("a" in row.refs_raw.split(",")).lower(), row.copy_read,
        ), strict=True)))
    return tuple(result)


def quantity_bytes(rows: tuple[dict[str, str], ...]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=QUANTITY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("ascii")


def scaled(row: dict[str, str], column: str) -> Decimal:
    shift = int(row["rate_exp10"]) - (6 if row["rate_unit"] == "s^-1" else 0)
    return Decimal(row[column]).scaleb(shift)


def selected(rows: tuple[dict[str, str], ...]) -> dict[tuple[int, int], dict[str, str]]:
    result: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if not (row["own_measurement"] == "true" and row["target_basis"] in ("isotope", "natural")
                and row["rate_unc"] and row["rate_parenthesized"] == "false"):
            continue
        z = int(row["Z"])
        mass = re.match(r"[0-9]+", row["target_label"])
        if row["target_basis"] == "isotope" and mass is None:
            raise Suzuki1987Error(f"{row['target_label']!r}: isotope lacks an integer mass prefix")
        a = 0 if row["target_basis"] == "natural" else int(mass.group())  # type: ignore[union-attr]
        key = z, a
        if key in result:
            raise Suzuki1987Error(
                f"Table {row['table']} p.{row['printed_page']} row {row['page_row_ordinal']}: "
                f"duplicate selected key {key}"
            )
        result[key] = row
    if not result:
        raise Suzuki1987Error("suzuki1987 printed rows line 1: empty own-measurement selection")
    return result
