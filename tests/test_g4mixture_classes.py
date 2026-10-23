"""The answer class of every compiled-in record, asserted from the public evidence alone.

This file shares no code with the module that generates ``record_classes.csv`` and
``natural_target_answers.csv`` (``openmucf.g4.d1_mixture``): it reads the provenance rows, the published cells
each record equals, the printed rows and the vendored literals itself, decides each record's class by its own
reading of that evidence, and requires the generated files to carry that class for every record. The drills
change one class at a time, in the generated rows and in the generator's mapping, and require the check to
name the record they changed.
"""

from __future__ import annotations

import csv
import io
import json
import re
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
D1 = ROOT / "data" / "g4" / "d1"


def _rows(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((D1 / name).read_bytes().decode("ascii"))))


def _exact(text: str) -> Fraction:
    return Fraction(text)


def _literals() -> dict[tuple[int, int], tuple[str, str]]:
    """The compiled-in records as the vendored source writes them: ``{Z, A, value, unc}`` literals."""
    source = (ROOT / "third_party" / "geant4" / "v11.4.2" / "G4MuonMinusBoundDecay.cc").read_text(
        encoding="ascii"
    )
    start = re.search(r"capRates\s*\[\]\s*=\s*\{", source)
    assert start is not None
    body = source[start.end() :]
    body = body[: body.index("};")]
    out: dict[tuple[int, int], tuple[str, str]] = {}
    for z, a, value, unc in re.findall(
        r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\}", body
    ):
        out[(int(z), int(a))] = (value, unc)
    return out


def _cells() -> dict[str, dict[str, object]]:
    """Every printed rate cell of the published tables with a symmetric uncertainty: its label, Z, whether it
    is
    this experiment's, and its value and uncertainty in 1e6/s as exact fractions."""
    out: dict[str, dict[str, object]] = {}
    for row in _rows("suzuki1987_quantity_rows.csv"):
        if not (row["rate"] and row["rate_unc"]):
            continue
        scale = Fraction(10) ** (int(row["rate_exp10"]) - (6 if row["rate_unit"] == "s^-1" else 0))
        out[f"Table {row['table']} p.{row['printed_page']} row {row['page_row_ordinal']}"] = {
            "label": row["target_label"],
            "z": int(row["Z"]),
            "own": row["own_measurement"] == "true",
            "rate": Fraction(row["rate"]) * scale,
            "unc": Fraction(row["rate_unc"]) * scale,
        }
    return out


def _kind_of_label(label: str) -> str:
    """``isotope`` for a printed mass number, ``radiogenic`` for the R prefix, ``natural`` for nat or a bare
    symbol."""
    stripped = re.sub(r"\^[a-z]$", "", label)
    if re.match(r"[0-9]", stripped):
        return "isotope:" + re.match(r"[0-9]+", stripped).group()  # type: ignore[union-attr]
    if re.fullmatch(r"R[A-Z][a-z]?", stripped):
        return "radiogenic"
    return "natural"


def _single_isotope() -> dict[int, int]:
    counts: dict[int, list[int]] = {}
    for row in _rows("geant4_natural_isotopes.csv"):
        counts.setdefault(int(row["z"]), []).append(int(row["a"]))
    return {z: a[0] for z, a in counts.items() if len(a) == 1}


def _named_combination(
    z: int, method: str, cells: dict[str, dict[str, object]]
) -> list[dict[str, object]] | None:
    """The printed cells a stated inverse-variance combination names, read from the evaluation method."""
    if "inverse-variance weighted mean" not in method:
        return None
    pages = re.search(r"Table (III|IV) p\.(\d+) rows (\d+) and (\d+)", method)
    if pages:
        return [cells[f"Table {pages[1]} p.{pages[2]} row {pages[k]}"] for k in (3, 4)]
    found: list[dict[str, object]] = []
    for symbol, mass in re.findall(r"the ([A-Z][a-z]?)-(\d+) row", method):
        hits = [
            c
            for c in cells.values()
            if c["z"] == z and re.sub(r"\^[a-z]$", "", str(c["label"])) == mass + symbol
        ]
        assert len(hits) == 1, (z, symbol, mass)
        found.append(hits[0])
    own = re.search(r"being the ([A-Z][a-z]?) row of this experiment", method)
    if own and "of that row and" in method:
        hits = [c for c in cells.values() if c["z"] == z and c["own"] and c["label"] == own[1]]
        assert len(hits) == 1, (z, own[1])
        found.insert(0, hits[0])
    assert len(found) == 2, (z, method[:200])
    return found


def _reproduces(literal: str, computed: Fraction, method: str) -> bool:
    digits = len(literal.split(".")[1]) if "." in literal else 0
    unit = Fraction(1, 10**digits)
    if "at the precision it is printed with" in method:
        # round half to even at the literal's last digit
        scaled = computed / unit
        floor = scaled.numerator // scaled.denominator
        rest = scaled - floor
        nearest = floor + (1 if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and floor % 2) else 0)
        return nearest * unit == Fraction(literal)
    assert "to within a unit of their last printed digit" in method
    return abs(computed - Fraction(literal)) <= unit


def _sqrt_close(square: Fraction, literal: str, method: str) -> bool:
    """Whether the square root of ``square`` reproduces ``literal`` under the stated tolerance, decided by
    bounding
    the root between two decimals 1e-30 apart."""
    lo, hi = Fraction(0), Fraction(max(1, int(square) + 1))
    while hi - lo > Fraction(1, 10**30):
        mid = (lo + hi) / 2
        if mid * mid <= square:
            lo = mid
        else:
            hi = mid
    return _reproduces(literal, lo, method) and _reproduces(literal, hi, method)


def expected_classes() -> dict[tuple[int, int], str]:
    """Each compiled-in record's class, from the evidence."""
    prov = json.loads((D1 / "d1_capture.prov.json").read_text(encoding="utf-8"))["rows"]
    cells = _cells()
    literals = _literals()
    single = _single_isotope()
    equal = {
        (int(r["Z"]), int(r["A"])): [x for x in r["equal_locators"].split("|") if x]
        for r in _rows("suzuki1987_parity_cells.csv")
    }
    out: dict[tuple[int, int], str] = {}
    for key, (value, unc) in literals.items():
        z, a = key
        row = prov[f"{z}-{a}"]
        method = row["evaluation_method"]
        named = _named_combination(z, method, cells)
        if named is not None:
            weights = [1 / (c["unc"] * c["unc"]) for c in named]  # type: ignore[operator]
            total = sum(weights)
            mean = sum(w * c["rate"] for w, c in zip(weights, named, strict=True)) / total  # type: ignore[operator]
            assert _reproduces(value, mean, method) and _sqrt_close(1 / total, unc, method), key
            if len({_kind_of_label(str(c["label"])) for c in named}) > 1:
                out[key] = "parity_mixed_reconstructed"
                continue
        if single.get(z) == a:
            out[key] = "parity_isotope"
            continue
        resolved, verify = row["isotope_resolved"], row["needs_verification"]
        if verify:
            out[key] = "parity_unsettled"
            continue
        if not resolved:
            for loc in equal[key]:
                assert _kind_of_label(str(cells[loc]["label"])) == "natural", (key, loc)
            out[key] = "parity_natural"
            continue
        if z <= 2:
            out[key] = "parity_isotope"
            continue
        matched = [cells[loc] for loc in equal[key]]
        for cell in matched:
            assert _kind_of_label(str(cell["label"])) == f"isotope:{a}", (key, cell["label"])
        exact = any(_exact(value) == c["rate"] and _exact(unc) == c["unc"] for c in matched)
        out[key] = "parity_isotope" if exact else "parity_isotope_lineage_open"
    return out


def a_key_failures(classes: list[dict[str, str]], answers: list[dict[str, str]]) -> list[str]:
    """Every record whose class in ``classes`` or in ``answers`` differs from the class its evidence
    supports."""
    expected = expected_classes()
    failures: list[str] = []
    generated = {(int(r["z"]), int(r["a"])): r["answer_class"] for r in classes if r["profile"] == "parity"}
    if set(generated) != set(expected):
        failures.append("record sets differ")
    for key, klass in expected.items():
        if generated.get(key) != klass:
            failures.append(f"record_classes {key}: {generated.get(key)} != {klass}")
    for row in answers:
        if row["record"].startswith("parity:"):
            z, a = (int(x) for x in row["record"].split(":")[1].split("-"))
            if row["answer_class"] != expected[(z, a)]:
                where = f"{row['profile']} {row['z']}-{row['a']}"
                failures.append(f"answers {where}: {row['answer_class']} != {expected[(z, a)]}")
    return failures


def test_a_key_every_record() -> None:
    assert a_key_failures(_rows("record_classes.csv"), _rows("natural_target_answers.csv")) == []


def test_a_key_counts_the_five_mixed_records() -> None:
    expected = expected_classes()
    assert {k for k, v in expected.items() if v == "parity_mixed_reconstructed"} == {
        (29, 63),
        (35, 79),
        (82, 207),
    }
    assert {k for k, v in expected.items() if v == "parity_unsettled"} == {(24, 52), (38, 88)}
    assert expected[(35, 81)] == "parity_isotope_lineage_open"


def _relabel(rows: list[dict[str, str]], key: tuple[int, int], klass: str) -> list[dict[str, str]]:
    out = [dict(r) for r in rows]
    for r in out:
        if r["profile"] == "parity" and (int(r["z"]), int(r["a"])) == key:
            r["answer_class"] = klass
    return out


@pytest.mark.parametrize(
    ("key", "klass"),
    [
        ((26, 56), "parity_isotope"),
        ((35, 81), "parity_isotope"),
        ((82, 207), "parity_natural"),
        ((82, 207), "parity_isotope"),
        ((29, 63), "parity_unsettled"),
    ],
)
def test_a_key_drill_relabel_in_the_generated_file(key: tuple[int, int], klass: str) -> None:
    failures = a_key_failures(
        _relabel(_rows("record_classes.csv"), key, klass), _rows("natural_target_answers.csv")
    )
    assert any(str(key) in f for f in failures), failures


def test_a_key_drill_flip_in_the_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """A generator whose mapping sends a natural-composition record to ``parity_isotope`` must fail A-key, in
    both
    generated files."""
    from openmucf.g4 import d1_mixture

    original = d1_mixture.answer_class

    def flipped(kind: str, lineage: str) -> str:
        return "parity_isotope" if kind == "natural_composition" else original(kind, lineage)

    monkeypatch.setattr(d1_mixture, "answer_class", flipped)
    classes = d1_mixture.record_classes(ROOT)
    answers = d1_mixture.answers(ROOT, classes)
    failures = a_key_failures(classes, answers)
    assert any(f.startswith("record_classes (26, 56)") for f in failures), failures[:5]
    assert any(f.startswith("answers parity 26-56") for f in failures), failures[:5]
