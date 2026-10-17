"""The natural-target answer classes and the mixture Geant4 makes of a natural target.

The classes and answers tests regenerate ``record_classes.csv`` and ``natural_target_answers.csv`` byte for
byte, check that every evaluated-profile row's printed label agrees with its key (A-lab), and that the class
shares partition each element's abundance and aggregate to the route shares ``natural_target_capture.csv``
states (A-share). Each gate has a drill that must turn it red.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

from openmucf.g4 import d1_mixture as mixture
from openmucf.g4 import d1_natural as natural
from openmucf.g4 import spec

ROOT = Path(__file__).resolve().parents[1]
D1 = ROOT / "data" / "g4" / "d1"


def _rows(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((D1 / name).read_bytes().decode("ascii"))))


# --------------------------------------------------------------------------------------------
# classes and answers: the committed files
# --------------------------------------------------------------------------------------------


def test_classes_regenerate_byte_for_byte() -> None:
    classes = mixture.record_classes(ROOT)
    assert mixture.render_csv(classes, mixture.CLASS_COLUMNS) == (ROOT / mixture.CLASSES_RELPATH).read_bytes()


def test_answers_regenerate_byte_for_byte() -> None:
    rows = mixture.answers(ROOT)
    assert mixture.render_csv(rows, mixture.ANSWER_COLUMNS) == (ROOT / mixture.ANSWERS_RELPATH).read_bytes()


def test_classes_cover_every_record_of_every_profile() -> None:
    classes = _rows("record_classes.csv")
    tables = natural.capture_profiles(D1)
    for name, table in tables.items():
        keys = {(int(z), int(a)) for z, a, _, _ in table.records}
        assert keys == {(int(r["z"]), int(r["a"])) for r in classes if r["profile"] == name}, name
    assert {r["value_lineage"] for r in classes} <= set(mixture.LINEAGES)
    assert {r["target_kind"] for r in classes} <= set(mixture.TARGET_KINDS)


# --------------------------------------------------------------------------------------------
# A-lab: an evaluated row's printed label agrees with its key
# --------------------------------------------------------------------------------------------


def a_lab_failures(profiles: dict[str, list[tuple[int, int, float, float]]]) -> list[str]:
    """Every evaluated-profile row whose printed label, found by value, does not admit its key: a mass-number
    label keys that mass number, ``nat`` keys ``A = 0``, a bare symbol keys the single natural isotope of its
    element and
    a bare symbol of an element with more than one is not admitted."""
    single = {
        z: m[0].a for z, m in natural.load_isotopes(D1 / "geant4_natural_isotopes.csv").items() if len(m) == 1
    }
    mizuno = _rows("mizuno2025_table3.csv")
    quantity = [
        r for r in _rows("suzuki1987_quantity_rows.csv") if r["own_measurement"] == "true" and r["rate_unc"]
    ]
    from openmucf.g4.sources import suzuki1987

    failures: list[str] = []
    for name, records in profiles.items():
        for z, a, value, unc in records:
            if name == "mizuno2025":
                labels = [
                    r["nuclide"]
                    for r in mizuno
                    if int(r["Z"]) == z and float(r["rate"]) == value and float(r["rate_unc"]) == unc
                ]
            else:
                labels = [
                    r["target_label"]
                    for r in quantity
                    if int(r["Z"]) == z
                    and r["rate_parenthesized"] == "false"
                    and float(suzuki1987.scaled(r, "rate")) == value
                    and float(suzuki1987.scaled(r, "rate_unc")) == unc
                ]
            if len(labels) != 1:
                failures.append(f"{name} ({z}, {a}): {len(labels)} printed labels")
                continue
            label = labels[0]
            if label.startswith("nat"):
                ok = a == 0
            elif label[0].isdigit():
                ok = int("".join(ch for ch in label if ch.isdigit())) == a
            else:
                ok = single.get(z) == a
            if not ok:
                failures.append(f"{name} ({z}, {a}): printed label {label!r} does not admit the key")
    return failures


def _evaluated_records() -> dict[str, list[tuple[int, int, float, float]]]:
    tables = natural.capture_profiles(D1)
    return {
        name: [(int(z), int(a), float(v), float(u)) for z, a, v, u in t.records]
        for name, t in tables.items()
        if name != spec.PARITY_PROFILE
    }


def test_classes_a_lab_every_evaluated_row() -> None:
    assert a_lab_failures(_evaluated_records()) == []


def test_classes_a_lab_drill_natural_row_rekeyed() -> None:
    """Rekey mizuno2025's natural-silicon row `14 0` to `14 28` and drop the enriched 28Si row: the format
    layer and
    the lookup accept the table, and A-lab must not."""
    profiles = _evaluated_records()
    records = [r for r in profiles["mizuno2025"] if (r[0], r[1]) != (14, 28)]
    records = [(14, 28, v, u) if (z, a) == (14, 0) else (z, a, v, u) for z, a, v, u in records]
    table = natural.capture_profiles(D1)["mizuno2025"]
    rekeyed = spec.G4DatTable(directives=table.directives, records=tuple(sorted(records)))
    reparsed = spec.parse(spec.render(rekeyed))
    assert (14, 28) in natural.profile_rows(reparsed)
    failures = a_lab_failures({**profiles, "mizuno2025": records})
    assert failures == ["mizuno2025 (14, 28): printed label 'natSi' does not admit the key"], failures


# --------------------------------------------------------------------------------------------
# A-share: the class shares partition each element and aggregate to the route shares
# --------------------------------------------------------------------------------------------

ROUTE_OF_CLASS = {mixture.EVALUATED_ISOTOPE: natural.PROFILE_ROUTE, mixture.FORMULA: natural.FORMULA_ROUTE}


def a_share_failures(answers: list[dict[str, str]]) -> list[str]:
    isotopes = natural.load_isotopes(D1 / "geant4_natural_isotopes.csv")
    study = {(r["profile"], int(r["z"])): r for r in _rows("natural_target_capture.csv")}
    failures: list[str] = []
    seen: dict[tuple[str, int], list[int]] = {}
    for row in answers:
        seen.setdefault((row["profile"], int(row["z"])), []).append(int(row["a"]))
    for key, members in seen.items():
        if sorted(members) != sorted(i.a for i in isotopes[key[1]]):
            failures.append(f"{key}: the rows do not partition the element's isotopes")
    shares = mixture.class_shares(answers, isotopes)
    routes = (natural.PROFILE_ROUTE, natural.TABLE_ROUTE, natural.FORMULA_ROUTE)
    for key, by_class in shares.items():
        # each class share is the correctly rounded sum of its isotopes' abundances, so the shares of an
        # element differ from its total abundance by at most one rounding per class
        total = math.fsum(i.abundance for i in isotopes[key[1]])
        if abs(math.fsum(by_class.values()) - total) > len(by_class) * math.ulp(total):
            failures.append(f"{key}: the class shares do not sum to the element's abundance")
        abundance = {i.a: i.abundance for i in isotopes[key[1]]}
        exact = {
            route: math.fsum(
                abundance[int(r["a"])]
                for r in answers
                if (r["profile"], int(r["z"])) == key
                and ROUTE_OF_CLASS.get(r["answer_class"], natural.TABLE_ROUTE) == route
            )
            for route in routes
        }
        for route, column in (
            (natural.PROFILE_ROUTE, "profile_share"),
            (natural.TABLE_ROUTE, "table_share"),
            (natural.FORMULA_ROUTE, "formula_share"),
        ):
            if repr(exact[route]) != study[key][column]:
                failures.append(f"{key}: {route} share {exact[route]!r} != {study[key][column]}")
    return failures


def test_answers_a_share_partition_and_aggregate() -> None:
    assert a_share_failures(_rows("natural_target_answers.csv")) == []


def test_answers_a_share_drill_moved_abundance() -> None:
    """Move iron-56 from its natural-composition record to the formula class without changing the route column
    or any total's source: A-share must fail on that element and on no other."""
    answers = _rows("natural_target_answers.csv")
    for row in answers:
        if (row["profile"], row["z"], row["a"]) == ("parity", "26", "56"):
            row["answer_class"] = mixture.FORMULA
    failures = a_share_failures(answers)
    assert failures and all(f.startswith("('parity', 26)") for f in failures), failures


def test_answers_route_and_class_agree() -> None:
    for row in _rows("natural_target_answers.csv"):
        assert ROUTE_OF_CLASS.get(row["answer_class"], natural.TABLE_ROUTE) == row["route"], row
