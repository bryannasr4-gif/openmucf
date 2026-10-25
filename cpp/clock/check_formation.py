"""Validate independent reaction-product timing metadata at the wrapper and atomic consumer."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from itertools import product
from pathlib import Path

REVISIONS = ("v11.4.2", "v11.5.0.beta")
ORIGINS = (0, 250, 1000)


def wrapper_records(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line.startswith(("INPUT ", "FINAL ", "SECONDARY ", "ENGINE ", "META ", "FRAGMENT ", "AUDIT_"))
    ]


def invariant(lines: list[str]) -> list[str]:
    return [
        re.sub(r" time \S+", " time <clock>", line) for line in lines if not line.startswith("AUDIT_PICK ")
    ]


def assess_wrapper(lines: list[str], mode: str, origin: float, expected_calls: int) -> dict:
    inputs, finals, products, metadata, fragments, engines = {}, {}, {}, {}, {}, set()
    supplied = {}
    for line in lines:
        fields = line.split()
        tag = fields[0]
        if tag == "AUDIT_PICK":
            continue
        if tag in ("INPUT", "FINAL"):
            event = int(fields[1])
            target = inputs if tag == "INPUT" else finals
            if event in target:
                raise AssertionError("duplicate wrapper case/final")
            target[event] = dict(zip(fields[2::2], fields[3::2], strict=True))
        elif tag in ("SECONDARY", "META"):
            key = (int(fields[1]), int(fields[2]))
            target = products if tag == "SECONDARY" else supplied
            if key in target:
                raise AssertionError("duplicate wrapper product/metadata")
            target[key] = dict(zip(fields[3::2], fields[4::2], strict=True))
        elif tag == "AUDIT_RP":
            row = dict(zip(fields[1::2], fields[2::2], strict=True))
            key = (int(row["event"]), int(row["index"]))
            if key in metadata:
                raise AssertionError("duplicate raw reaction-product metadata")
            metadata[key] = row
        elif tag == "AUDIT_FRAGMENT":
            event = int(fields[2])
            if event in fragments:
                raise AssertionError("duplicate incoming fragment")
            fragments[event] = float.fromhex(fields[4])
        elif tag == "ENGINE":
            event = int(fields[1])
            if event in engines:
                raise AssertionError("duplicate engine observation")
            engines.add(event)
    expected = set(range(expected_calls))
    if expected_calls <= 0 or set(inputs) != expected or set(finals) != expected or engines != expected:
        raise AssertionError("missing declared wrapper calls or engine states")
    failures, positive, count = [], 0, 0
    grid = Counter(product((-1.0, 0.0, 1.0, 7.0), (-2.0, 0.0, 0.125, 3.0)))
    for event in expected:
        incoming, final = inputs[event], finals[event]
        if incoming["mode"] != mode or float.fromhex(incoming["origin"]) != origin:
            raise AssertionError("wrong wrapper mode or origin")
        size = int(final["count"])
        if {(event, i) for i in range(size)} != {key for key in products if key[0] == event}:
            raise AssertionError("missing or extra wrapper secondary")
        if mode == "light":
            if (
                int(final["dependency_calls"]) != 0
                or event in fragments
                or any(k[0] == event for k in metadata)
            ):
                raise AssertionError("light-target bypass called the dependency")
        else:
            if fragments.get(event) != 0.0:
                raise AssertionError("incoming fragment origin was not independently observed at zero")
            if {(event, i) for i in range(size - 1)} != {key for key in metadata if key[0] == event}:
                raise AssertionError("missing raw reaction-product metadata")
            if mode == "controlled":
                if size != 17 or int(final["dependency_calls"]) != 1:
                    raise AssertionError("incomplete supplied dependency call")
                observed_grid = Counter()
                if {(event, i) for i in range(16)} != {key for key in supplied if key[0] == event}:
                    raise AssertionError("missing supplied reaction-product metadata")
                for i in range(16):
                    supplied_row, raw = supplied[(event, i)], metadata[(event, i)]
                    if any(supplied_row[key] != raw[key] for key in ("name", "tof", "formation")):
                        raise AssertionError("supplied and consumed metadata differ")
                    observed_grid[(float.fromhex(raw["tof"]), float.fromhex(raw["formation"]))] += 1
                if observed_grid != grid:
                    raise AssertionError("missing independent TOF/formation grid point")
        for index in range(size):
            row = products[(event, index)]
            formation = 0.0
            if index and mode != "light":
                raw = metadata[(event, index - 1)]
                formation = float.fromhex(raw["formation"])
                if row["name"] != raw["name"] or float.fromhex(raw["origin"]) != origin:
                    raise AssertionError("wrong metadata/product handoff")
            actual, predicted = float.fromhex(row["time"]), origin + max(formation, 0.0)
            if not all(math.isfinite(x) for x in (actual, predicted, formation)):
                raise AssertionError("nonfinite formation clock")
            positive += formation > 0.0
            count += 1
            if not math.isclose(actual, predicted, rel_tol=0.0, abs_tol=1.0e-9):
                failures.append({"case": event, "secondary": index, "actual": actual, "expected": predicted})
    if mode != "light" and positive == 0:
        raise AssertionError("vacuous positive formation-time coverage")
    return {
        "calls": expected_calls,
        "products": count,
        "positive_formation_products": positive,
        "failures": len(failures),
        "first_failures": failures[:3],
    }


def assess_integrated(text: str, origin: float, expected_events: int) -> dict:
    atoms, waits, metadata, fragments, products, engines = {}, {}, {}, {}, {}, set()
    current = None
    for line in text.splitlines():
        fields = line.split()
        if line.startswith("ATOM_START "):
            current = int(fields[1])
            if current in atoms:
                raise AssertionError("duplicate atom origin")
            atoms[current] = float.fromhex(fields[2])
        elif line.startswith("G4MuonicAtomDecay::DecayIt time set to:") and current not in waits:
            waits[current] = float(line.split(": ")[-1].split("[")[0])
        elif line.startswith("AUDIT_RP "):
            row = dict(zip(fields[1::2], fields[2::2], strict=True))
            key = (int(row["event"]), int(row["index"]))
            if key in metadata:
                raise AssertionError("duplicate integrated metadata")
            metadata[key] = row
        elif line.startswith("AUDIT_FRAGMENT "):
            event = int(fields[2])
            if event in fragments:
                raise AssertionError("duplicate integrated incoming fragment")
            fragments[event] = float.fromhex(fields[4])
        elif line.startswith("PRODUCT "):
            key = (int(fields[1]), fields[5], int(fields[2]))
            if key in products:
                raise AssertionError("duplicate integrated product")
            products[key] = {"name": fields[3], "time": float.fromhex(fields[7])}
        elif line.startswith("AUDIT_RNG "):
            event = int(fields[1])
            if event in engines:
                raise AssertionError("duplicate integrated engine state")
            engines.add(event)
    expected = set(range(expected_events))
    if set(atoms) != expected or set(waits) != expected or engines != expected:
        raise AssertionError("missing integrated events/wait observations")
    if any(not math.isfinite(wait) or wait < 0.0 for wait in waits.values()):
        raise AssertionError("invalid independently observed waiting time")
    captures = {event for event, index in metadata}
    decays = expected - captures
    if not captures or not decays or any(fragments.get(event) != 0.0 for event in captures):
        raise AssertionError("missing integrated branch or zero incoming fragment origin")
    for event in captures:
        indices = {index for ev, index in metadata if ev == event}
        daughter_indices = {
            index for ev, phase, index in products if ev == event and phase == "atom_daughter"
        }
        if indices != set(range(len(indices))) or daughter_indices != set(range(len(indices) + 1)):
            raise AssertionError("missing integrated metadata/product handoff")
    failures, positive = [], 0
    for (event, phase, index), row in products.items():
        formation = 0.0
        if phase == "atom_daughter":
            if event in captures and index:
                raw = metadata[(event, index - 1)]
                formation = float.fromhex(raw["formation"])
                if row["name"] != raw["name"] or float.fromhex(raw["origin"]) != 0.0:
                    raise AssertionError("integrated metadata/product mismatch")
            elif event in decays and not any(
                value["name"] == "anti_nu_e"
                for key, value in products.items()
                if key[0] == event and key[1] == "atom_daughter"
            ):
                raise AssertionError("missing DIO branch marker")
            predicted = origin + (waits[event] + max(formation, 0.0))
        else:
            predicted = origin
        actual = row["time"]
        if not all(math.isfinite(x) for x in (actual, predicted, formation)):
            raise AssertionError("nonfinite integrated formation clock")
        positive += formation > 0.0
        if not math.isclose(actual, predicted, rel_tol=0.0, abs_tol=1.0e-9):
            failures.append({"event": event, "phase": phase, "actual": actual, "expected": predicted})
    if positive == 0 or any(atoms[event] != origin for event in expected):
        raise AssertionError("missing positive formation witness or incorrect atom origin")
    return {
        "events": expected_events,
        "capture_events": len(captures),
        "DIO_events": len(decays),
        "positive_formation_products": positive,
        "failures": len(failures),
        "first_failures": failures[:3],
    }


def check(root: Path, controlled_calls: int, default_calls: int, integrated_events: int) -> dict:
    reports = {}
    for revision in REVISIONS:
        cells = {}
        for mode, count in (("controlled", controlled_calls), ("default", default_calls), ("light", 4)):
            for origin in ORIGINS:
                loaded = {
                    v: wrapper_records((root / revision / f"{v}-{mode}-{origin}.log").read_text())
                    for v in ("old", "new", "zero")
                }
                if not invariant(loaded["old"]) == invariant(loaded["new"]) == invariant(loaded["zero"]):
                    raise AssertionError("matched wrapper non-time output, metadata or full RNG changed")
                results = {v: assess_wrapper(lines, mode, origin, count) for v, lines in loaded.items()}
                if results["new"]["failures"]:
                    raise AssertionError("formation getter correction failed")
                if mode == "light":
                    if loaded["old"] != loaded["new"] or loaded["new"] != loaded["zero"]:
                        raise AssertionError("light-target bypass output changed")
                elif not all(results[v]["failures"] > 0 for v in ("old", "zero")):
                    raise AssertionError("old getter or all-zero formation gate did not go red")
                cells[f"{mode}-{origin}"] = results
        integrated = {
            v: assess_integrated(
                (root / revision / f"{v}-integrated-250.log").read_text(), 250.0, integrated_events
            )
            for v in ("old", "new", "zero")
        }
        if integrated["new"]["failures"] or not all(integrated[v]["failures"] > 0 for v in ("old", "zero")):
            raise AssertionError("integrated formation-clock acceptance/adverse check failed")
        reports[revision] = {
            "wrapper": cells,
            "integrated": integrated,
            "matched_non_time_and_rng_identical": True,
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--controlled-calls", type=int, required=True)
    parser.add_argument("--default-calls", type=int, required=True)
    parser.add_argument("--integrated-events", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.log_root, args.controlled_calls, args.default_calls, args.integrated_events)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for revision, report in result.items():
        print(
            revision,
            "formation propagation PASS; old getter/zero-all red; matched non-time/full-RNG identical",
        )
        print("integrated", {variant: cell["failures"] for variant, cell in report["integrated"].items()})


if __name__ == "__main__":
    main()
