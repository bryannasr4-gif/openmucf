"""The ``mizuno2025`` profile's input: two transcriptions of Mizuno et al. (2025), Tables 1 and 3.

Unlike the ``parity`` profile, whose every number is parsed out of a vendored source file, this
profile rests on two files a human typed from the primary's tables -- so, as with the isotope audit,
the structural invariants are the whole of the protection available, and every one of them is
enforced here rather than trusted:

* Table 1 is transcribed row for row as printed, one row per target, and a nuclide the primary
  measured on two targets has two rows.
* Table 3's ``Exp.`` column is the primary's own per-nuclide value: for a nuclide printed once in
  Table 1 it repeats that row's value and uncertainty, string for string; for a nuclide printed
  twice it carries the primary's footnote and is the primary's average, never ours.

Standard library only, and no import of the kinetics modules -- the fence ``openmucf/g4/__init__.py``
states and ``tests/test_g4spec.py`` enforces.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

PROFILE = "mizuno2025"
#: The bibliography key of the primary both tables are read from.
BIBKEY = "Mizuno2025"
TABLE1_RELPATH = "data/g4/d1/mizuno2025_table1.csv"
TABLE3_RELPATH = "data/g4/d1/mizuno2025_table3.csv"
#: The columns of each file, in order. The header must match exactly: a reordered or renamed column
#: is a silent re-interpretation of hand-entered data.
TABLE1_COLUMNS = (
    "nuclide", "form", "size_mm", "weight_g", "time_h", "huff_factor", "lifetime_ns",
    "lifetime_unc_ns", "suzuki_lifetime_ns", "suzuki_lifetime_unc_ns", "rate", "rate_unc",
    "locator", "copy_read",
)
TABLE3_COLUMNS = ("Z", "A", "nuclide", "rate", "rate_unc", "note", "locator", "copy_read")
#: Every numeric column of Table 1 that must always parse; the two Suzuki columns are either both
#: present or both empty, since the primary leaves them blank for the enriched targets.
TABLE1_NUMERIC = ("weight_g", "time_h", "huff_factor", "lifetime_ns", "lifetime_unc_ns", "rate", "rate_unc")
TABLE1_PAIRED = ("suzuki_lifetime_ns", "suzuki_lifetime_unc_ns")
TABLE3_NUMERIC = ("rate", "rate_unc")
#: The one column the primary prints with characters outside US-ASCII (a diameter sign and a
#: multiplication sign in the target dimensions); every other cell of both files is ASCII.
TABLE1_NON_ASCII = ("size_mm",)


class Mizuno2025Error(RuntimeError):
    """A transcription is malformed. Raised with the file and row named, never swallowed."""


@dataclass(frozen=True)
class Table1Row:
    """One printed row of Table 1, every cell as the primary prints it (``v(u)`` split in two)."""

    nuclide: str
    form: str
    size_mm: str
    weight_g: str
    time_h: str
    huff_factor: str
    lifetime_ns: str
    lifetime_unc_ns: str
    suzuki_lifetime_ns: str
    suzuki_lifetime_unc_ns: str
    rate: str
    rate_unc: str
    locator: str
    copy_read: str

    @property
    def has_suzuki_value(self) -> bool:
        """True when the primary prints a comparison value from its reference for this target."""
        return bool(self.suzuki_lifetime_ns)


@dataclass(frozen=True)
class Table3Row:
    """One row of Table 3's ``Exp.`` column, keyed to the dataset's ``(Z, A)`` scheme."""

    z: int
    a: int
    nuclide: str
    rate: str
    rate_unc: str
    note: str
    locator: str
    copy_read: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.z, self.a)


@dataclass(frozen=True)
class Mizuno2025Extraction:
    """Both transcriptions, cross-checked: the rows of Table 1 and Table 3 in printed order."""

    table1: tuple[Table1Row, ...]
    table3: tuple[Table3Row, ...]

    def table1_rows(self, nuclide: str) -> tuple[Table1Row, ...]:
        """Every Table-1 row printed under ``nuclide``, in printed order."""
        return tuple(row for row in self.table1 if row.nuclide == nuclide)

    @property
    def keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(row.key for row in self.table3)


def _read(
    path: Path, columns: tuple[str, ...], non_ascii: tuple[str, ...]
) -> list[tuple[str, dict[str, str]]]:
    """The rows of a transcription as ``(where, record)`` pairs, after the byte-level checks."""
    raw = path.read_bytes()
    if b"\r" in raw:
        raise Mizuno2025Error(f"{path.name} contains CR; the transcription is committed LF-only")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Mizuno2025Error(f"{path.name} is not UTF-8: {exc}") from None
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != columns:
        raise Mizuno2025Error(
            f"{path.name} header is {tuple(reader.fieldnames or ())!r}, expected {columns!r}"
        )
    rows: list[tuple[str, dict[str, str]]] = []
    for number, record in enumerate(reader, start=2):
        where = f"{path.name} line {number}"
        if None in record or any(value is None for value in record.values()):
            raise Mizuno2025Error(f"{where}: {len(columns)} cells expected")
        for name, value in record.items():
            if name not in non_ascii and not value.isascii():
                raise Mizuno2025Error(f"{where}: column {name!r} must be ASCII, got {value!r}")
        for name in ("locator", "copy_read"):
            if not record[name]:
                raise Mizuno2025Error(f"{where}: every row must carry a {name}")
        rows.append((where, record))
    if not rows:
        raise Mizuno2025Error(f"{path.name} carries no rows")
    return rows


def _decimal(text: str, where: str, column: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation:
        raise Mizuno2025Error(f"{where}: {column} must be a decimal number, got {text!r}") from None


def load_table1(path: Path) -> tuple[Table1Row, ...]:
    """Parse Table 1's transcription, refusing anything the generator could carry into shipped bytes."""
    rows = []
    for where, record in _read(path, TABLE1_COLUMNS, TABLE1_NON_ASCII):
        for column in TABLE1_NUMERIC:
            _decimal(record[column], where, column)
        present = [bool(record[column]) for column in TABLE1_PAIRED]
        if any(present) and not all(present):
            raise Mizuno2025Error(
                f"{where}: {TABLE1_PAIRED[0]} and {TABLE1_PAIRED[1]} must be present or absent together"
            )
        if all(present):
            for column in TABLE1_PAIRED:
                _decimal(record[column], where, column)
        rows.append(Table1Row(**record))
    return tuple(rows)


def load_table3(path: Path) -> tuple[Table3Row, ...]:
    """Parse Table 3's transcription: integer keys, unique, decimal values."""
    rows: list[Table3Row] = []
    seen: dict[tuple[int, int], str] = {}
    for where, record in _read(path, TABLE3_COLUMNS, ()):
        try:
            z, a = int(record["Z"]), int(record["A"])
        except ValueError:
            raise Mizuno2025Error(f"{where}: Z and A must be integers") from None
        for column in TABLE3_NUMERIC:
            _decimal(record[column], where, column)
        if (z, a) in seen:
            raise Mizuno2025Error(f"{where}: duplicate key ({z}, {a}), first seen at {seen[(z, a)]}")
        seen[(z, a)] = where
        rows.append(
            Table3Row(
                z=z, a=a, nuclide=record["nuclide"], rate=record["rate"], rate_unc=record["rate_unc"],
                note=record["note"], locator=record["locator"], copy_read=record["copy_read"],
            )
        )
    return tuple(rows)


def load(table1_path: Path, table3_path: Path) -> Mizuno2025Extraction:
    """Both tables, cross-checked row against row.

    The two files describe one set of nuclides. A Table-3 row without a footnote repeats exactly one
    Table-1 row's value and uncertainty as printed; a Table-3 row with the footnote averages at
    least two Table-1 rows, and the average is the primary's, so nothing here recomputes it.
    """
    table1 = load_table1(table1_path)
    table3 = load_table3(table3_path)
    labels1 = {row.nuclide for row in table1}
    labels3 = {row.nuclide for row in table3}
    if labels1 != labels3:
        raise Mizuno2025Error(
            f"nuclide labels differ between the two tables: only in {table1_path.name} "
            f"{sorted(labels1 - labels3)}, only in {table3_path.name} {sorted(labels3 - labels1)}"
        )
    for row in table3:
        printed = [r for r in table1 if r.nuclide == row.nuclide]
        where = f"{table3_path.name} row {row.nuclide}"
        if not row.note:
            if len(printed) != 1:
                raise Mizuno2025Error(
                    f"{where}: note is empty but {table1_path.name} prints {len(printed)} rows for "
                    f"{row.nuclide}; a value over more than one target carries the primary's footnote"
                )
            (only,) = printed
            if (only.rate, only.rate_unc) != (row.rate, row.rate_unc):
                raise Mizuno2025Error(
                    f"{where}: rate {row.rate}({row.rate_unc}) is not {table1_path.name}'s "
                    f"{only.rate}({only.rate_unc}) for {row.nuclide}, string for string"
                )
        elif len(printed) < 2:
            raise Mizuno2025Error(
                f"{where}: note is set but {table1_path.name} prints only {len(printed)} row(s) for "
                f"{row.nuclide}; the footnote marks an average over more than one target"
            )
    return Mizuno2025Extraction(table1=table1, table3=table3)
