"""Validate the delivered clock and matched-input invariants from clock_probe logs.

Usage: python check_clock.py LOG_ROOT [--output result.json]
LOG_ROOT/<revision>/<variant>-<route>-<t0>.log contains the probe's stdout.
No Geant4/Python binding or statistical accuracy claim is involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

REVISIONS = ("v11.4.2", "v11.5.0.beta")
VARIANTS = ("installed", "pristine", "corrected", "wrong-clamp", "duplicate-stop", "omit-stop")
ROUTES = ("atomic", "stopping", "fixture")
STOPS = (0, 250, 1000)


def records(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(("CASE ", "SEC ", "RNG ", "RATES "))]


def invariant(lines: list[str]) -> list[str]:
    # All non-time observables, including the full random engine state.
    return [re.sub(r" time \S+", " time <clock>", line) if line.startswith("SEC ") else line for line in lines]


def assess(lines: list[str], route: str, *, old: bool = False) -> dict:
    case = None
    counts: Counter = Counter()
    failures = []
    seen_cases = set()
    seen_rng = set()
    rng_records = 0
    expected_count = None
    actual_count = 0
    for line in lines:
        fields = line.split()
        if fields[0] == "CASE":
            if expected_count is not None and actual_count != expected_count:
                raise AssertionError("missing or extra secondary record")
            case = {fields[k]: fields[k+1] for k in range(3, len(fields), 2)}
            seen_cases.add(int(fields[2]))
            case["id"] = int(fields[2])
            expected_count, actual_count = int(case["count"]), 0
            counts["cases"] += 1
        elif fields[0] == "RNG":
            seen_rng.add(int(fields[1]))
            rng_records += 1
        elif fields[0] == "SEC":
            if case is None or int(fields[1]) != case["id"]:
                raise AssertionError("secondary without matching case")
            actual_count += 1
            values = {fields[k]: fields[k+1] for k in range(4, len(fields), 2)}
            phase = values["phase"]
            t0, wait, local = (float.fromhex(case[key]) for key in ("t0", "wait", "local"))
            actual = float.fromhex(values["time"])
            if phase == "atom":
                expected = 2*t0 if old else t0
            elif phase == "em":
                expected = t0 + (0.375 if route == "fixture" else 0.)
            elif phase == "decay":
                expected = t0 + wait
            elif phase == "capture":
                local = local if route == "fixture" else -1.
                expected = t0 + (max(0., wait+local) if old else wait+max(0.,local))
                if local > 0.:
                    counts["positive_model_delay"] += 1
            else:
                raise AssertionError(f"unknown phase {phase}")
            if not all(math.isfinite(x) for x in (actual, expected)):
                raise AssertionError("nonfinite clock")
            counts[phase] += 1
            counts["products"] += 1
            if not math.isclose(actual, expected, rel_tol=0., abs_tol=1.e-9):
                failures.append({"case": case["id"], "phase": phase, "actual": actual, "expected": expected})
    if expected_count is None or actual_count != expected_count:
        raise AssertionError("empty or incomplete probe log")
    if (seen_cases != seen_rng or seen_cases != set(range(len(seen_cases)))
            or counts["cases"] != len(seen_cases) or rng_records != len(seen_cases)):
        raise AssertionError("missing, duplicate, or disordered event/RNG records")
    if counts["products"] == 0 or counts["em"] == 0:
        raise AssertionError("vacuous probe")
    if route == "fixture":
        if counts["cases"] != 160 or not all(counts[k] for k in ("capture", "decay", "positive_model_delay")):
            raise AssertionError("incomplete deterministic/generated fixture grid")
    elif route == "atomic" and counts["atom"] != counts["cases"]:
        raise AssertionError("missing muonic atom")
    elif route == "stopping" and not all(counts[k] for k in ("capture", "decay")):
        raise AssertionError("stock capture/decay branches were not both observed")
    if route != "fixture" and counts["cases"] != 300:
        raise AssertionError("incomplete stock process sample")
    return {"counts": dict(counts), "failures": len(failures), "first_failures": failures[:3]}


def check(root: Path) -> dict:
    result = {"scope": "application object-overlay, actual Geant4 process boundary", "revisions": {}}
    for rev in REVISIONS:
        cells = {}
        drill_failures = Counter()
        for route in ROUTES:
            for stop in STOPS:
                loaded = {v: records((root/rev/f"{v}-{route}-{stop}.log").read_text()) for v in VARIANTS}
                if loaded["installed"] != loaded["pristine"]:
                    raise AssertionError(f"{rev} {route} {stop}: pristine object overlay changes baseline")
                base = invariant(loaded["installed"])
                for variant in VARIANTS[2:]:
                    if invariant(loaded[variant]) != base:
                        raise AssertionError(f"{rev} {route} {stop} {variant}: non-time observables or RNG changed")
                old_law = assess(loaded["installed"], route, old=True)
                if old_law["failures"]:
                    raise AssertionError(f"baseline does not reproduce known old law: {rev} {route} {stop}")
                reports = {v: assess(lines,route) for v,lines in loaded.items()}
                if reports["corrected"]["failures"]:
                    raise AssertionError(f"corrected clock failed: {rev} {route} {stop}")
                for variant in ("installed", "wrong-clamp", "duplicate-stop", "omit-stop"):
                    drill_failures[variant] += reports[variant]["failures"]
                cells[f"{route}-{stop}"] = reports
        if not all(drill_failures[v] > 0 for v in ("installed", "wrong-clamp", "duplicate-stop", "omit-stop")):
            raise AssertionError(f"a timing gate failed to reject its adverse variant: {rev}")
        result["revisions"][rev] = {"cells": cells, "adverse_product_failures": dict(drill_failures),
            "installed_equals_pristine_overlay": True, "matched_non_time_and_rng_identical": True}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.log_root)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    for rev,report in result["revisions"].items():
        print(rev, "corrected physical clock PASS; baseline/three adverse variants rejected; non-time+RNG identical",
              report["adverse_product_failures"])


if __name__ == "__main__":
    main()
