"""Printed-row transcription structure for the published capture tables."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from openmucf.g4.sources import suzuki1987

DATA = Path(__file__).resolve().parents[1] / suzuki1987.PRINTED_ROWS_RELPATH


def _csv_bytes(rows: list[dict[str, str]], columns: tuple[str, ...] | None = None) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=columns or suzuki1987.PRINTED_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("ascii")


def _row() -> dict[str, str]:
    row = dict.fromkeys(suzuki1987.PRINTED_COLUMNS, "")
    row.update(
        table="III", printed_page="2217", page_row_ordinal="1", row_kind="data",
        zeff_underlined="false", copy_read="published-scan",
    )
    return row


def _corruption(case: str) -> bytes:
    first = _row()
    rows = [first]
    if case == "header":
        return _csv_bytes(rows).replace(b"table,", b"wrong,", 1)
    if case == "cr":
        return _csv_bytes(rows).replace(b"\n", b"\r\n")
    if case == "ascii":
        return _csv_bytes(rows).replace(b"data", "daté".encode())
    if case == "shape":
        return _csv_bytes(rows)[:-1] + b",extra\n"
    if case == "table":
        first["table"] = "II"
    elif case == "integer":
        first["printed_page"] = "x"
    elif case == "positive":
        first["printed_page"] = "-1"
    elif case == "ordinal":
        second = _row()
        second["page_row_ordinal"] = "3"
        rows.append(second)
    elif case == "duplicate":
        rows.append(_row())
    elif case == "kind":
        first["row_kind"] = "invented"
    elif case == "underline":
        first["zeff_underlined"] = "maybe"
    elif case == "copy":
        first["copy_read"] = "preprint-scan"
    elif case == "empty":
        rows.clear()
    return _csv_bytes(rows)


# T-131: every structural refusal must name the offending file and line.
@pytest.mark.parametrize(
    "case",
    (
        "header", "cr", "ascii", "shape", "table", "integer", "positive", "ordinal",
        "duplicate", "kind", "underline", "copy", "empty", "missing",
    ),
)
def test_loader_refusals(tmp_path: Path, case: str) -> None:
    path = tmp_path / "printed_rows.csv"
    if case != "missing":
        path.write_bytes(_corruption(case))
    with pytest.raises(suzuki1987.Suzuki1987Error, match=r"printed_rows\.csv line \d+"):
        suzuki1987.load_printed_rows(path)


# T-132: the committed ledger is one ordered record per printed text line.
def test_committed_ledger_loads() -> None:
    rows = suzuki1987.load_printed_rows(DATA)
    assert rows
    assert all(row.locator.startswith(f"Table {row.table} p.{row.printed_page} row ") for row in rows)
    assert all(row.copy_read == "published-scan" for row in rows)
    assert {row.row_kind for row in rows} == suzuki1987.ROW_KINDS
