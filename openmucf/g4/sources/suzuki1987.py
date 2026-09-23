from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

PRINTED_ROWS_RELPATH = "data/g4/d1/suzuki1987_printed_rows.csv"
PRINTED_COLUMNS = (
    "table", "printed_page", "page_row_ordinal", "row_kind", "z_raw", "zeff_raw",
    "zeff_underlined", "element_raw", "mean_life_raw", "rate_raw", "huff_raw",
    "refs_raw", "note", "copy_read",
)
ROW_KINDS = frozenset(("caption", "header", "data", "positive_muon", "note", "footnote"))


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
