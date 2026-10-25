"""Check controlled and GPIL-sampled calls through the actual muonic-atom decay process."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

from check_clock import FIXTURE_LOCALS, FIXTURE_WAITS, REVISIONS, STOPS, invariant
from check_transport import assess as assess_transport
from check_transport import parse as parse_transport


def selected(text: str) -> list[str]:
    return [
        line for line in text.splitlines() if line.startswith(("WAIT ", "CASE ", "SEC ", "RNG ", "RATES "))
    ]


def unchanged_clock_subset(lines: list[str]) -> list[str]:
    keep = False
    result = []
    for line in lines:
        fields = line.split()
        if fields[0] == "CASE":
            case = {fields[i]: fields[i + 1] for i in range(3, len(fields), 2)}
            keep = (
                case["branch"] == "1"
                or float.fromhex(case["local"]) >= 0.0
                or float.fromhex(case["wait"]) == 0.0
            )
        elif fields[0] == "SEC" and keep:
            result.append(line)
    return result


def assess(lines: list[str], stop: float, sampled_events: int, *, old: bool = False) -> dict:
    waits, cases, secondaries, rng = {}, {}, {}, set()
    rates = 0
    for line in lines:
        fields = line.split()
        if fields[0] == "WAIT":
            event = int(fields[1])
            if event in waits:
                raise AssertionError("duplicate waiting-time observation")
            waits[event] = {fields[i]: fields[i + 1] for i in range(2, len(fields), 2)}
        elif fields[0] == "CASE":
            event = int(fields[2])
            if event in cases or fields[1] != "atom_decay":
                raise AssertionError("duplicate or wrong-route case")
            cases[event] = {fields[i]: fields[i + 1] for i in range(3, len(fields), 2)}
            secondaries[event] = []
        elif fields[0] == "SEC":
            event = int(fields[1])
            if event not in cases or int(fields[2]) != len(secondaries[event]):
                raise AssertionError("secondary without ordered matching case")
            secondaries[event].append({fields[i]: fields[i + 1] for i in range(4, len(fields), 2)})
        elif fields[0] == "RNG":
            event = int(fields[1])
            if event in rng:
                raise AssertionError("duplicate RNG observation")
            rng.add(event)
        else:
            rates += 1
    expected_events = set(range(30 + sampled_events))
    if set(cases) != expected_events or set(waits) != expected_events or rng != expected_events or rates != 1:
        raise AssertionError("missing process/wait/RNG/rate observations")
    grid = Counter()
    counts = Counter()
    failures = []
    for event, case in cases.items():
        observed = waits[event]
        wait, local, origin = (float.fromhex(case[key]) for key in ("wait", "local", "t0"))
        mean, independent_wait = (float.fromhex(observed[key]) for key in ("mean", "observed"))
        if not all(math.isfinite(x) for x in (wait, local, origin, mean, independent_wait)):
            raise AssertionError("nonfinite process input or independent waiting observation")
        if origin != stop or wait < 0.0 or mean <= 0.0 or wait != independent_wait:
            raise AssertionError("wrong origin or inconsistent independent GPIL waiting observation")
        captured = case["branch"] == "0"
        phase = "capture" if captured else "decay"
        if event < 30:
            declared = float.fromhex(observed["declared"])
            if observed["mode"] != "controlled" or not captured:
                raise AssertionError("controlled nuclear fixture did not take capture branch")
            if not math.isclose(wait, declared, rel_tol=0.0, abs_tol=1.0e-12):
                raise AssertionError("controlled waiting observation disagrees with declared input")
            grid[(declared, local)] += 1
        elif observed["mode"] != "sampled":
            raise AssertionError("missing actual GPIL-sampled waiting case")
        else:
            counts[f"sampled_{phase}"] += 1
        products = secondaries[event]
        if len(products) != int(case["count"]) or not products:
            raise AssertionError("missing decay-process products")
        for secondary in products:
            actual = float.fromhex(secondary["time"])
            expected = (
                origin + (max(0.0, wait + local) if old else wait + max(0.0, local))
                if captured
                else origin + wait
            )
            if not math.isfinite(actual) or not math.isfinite(expected):
                raise AssertionError("nonfinite delivered decay-process clock")
            if secondary["phase"] != phase:
                raise AssertionError("inconsistent branch classification")
            counts[phase] += 1
            if captured and local > 0.0:
                counts["positive_model_local_time"] += 1
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-9):
                failures.append({"event": event, "actual": actual, "expected": expected})
    if grid != Counter(product(FIXTURE_WAITS, FIXTURE_LOCALS)):
        raise AssertionError("missing controlled atomic-decay grid point")
    if not all(counts[key] for key in ("sampled_capture", "sampled_decay", "positive_model_local_time")):
        raise AssertionError("vacuous branch or positive-local-time coverage")
    return {
        "events": len(cases),
        "counts": dict(counts),
        "failures": len(failures),
        "first_failures": failures[:3],
    }


def check(root: Path, prior_transport: Path, sampled_events: int, transport_events: int) -> dict:
    reports = {}
    for revision in REVISIONS:
        cells, transports = {}, {}
        for stop in STOPS:
            loaded = {
                variant: selected((root / revision / f"{variant}-atom_decay-{stop}.log").read_text())
                for variant in ("installed", "clock1", "clock2")
            }
            if loaded["installed"] != loaded["clock1"]:
                raise AssertionError("unmodified atomic-decay source overlay changes installed baseline")
            if invariant(loaded["clock1"]) != invariant(loaded["clock2"]):
                raise AssertionError("atomic-decay correction changes non-time observables or RNG state")
            if unchanged_clock_subset(loaded["clock1"]) != unchanged_clock_subset(loaded["clock2"]):
                raise AssertionError("nonnegative model, zero-wait, or DIO product clocks changed")
            old = assess(loaded["clock1"], stop, sampled_events, old=True)
            if old["failures"]:
                raise AssertionError("old atomic-decay composition not reproduced")
            baseline = assess(loaded["clock1"], stop, sampled_events)
            corrected = assess(loaded["clock2"], stop, sampled_events)
            if baseline["failures"] == 0 or corrected["failures"]:
                raise AssertionError("old atomic-decay defect must fail and corrected clock must pass")
            cells[str(stop)] = {"baseline": baseline, "corrected": corrected}
            modes = {}
            for mode in ("spy", "plain"):
                clock1 = parse_transport((root / revision / f"clock1-atomic-{stop}-{mode}.log").read_text())
                clock2 = parse_transport((root / revision / f"clock2-atomic-{stop}-{mode}.log").read_text())
                prior = parse_transport(
                    (prior_transport / revision / f"corrected-atomic-{stop}-{mode}.log").read_text()
                )
                if clock1 != clock2 or clock1 != prior:
                    raise AssertionError("default nonnegative-model atomic transport changed")
                if mode == "spy":
                    report = assess_transport(clock2, "atomic", stop, transport_events)
                    if report["failures"]:
                        raise AssertionError("default atomic transport clock failed")
                else:
                    # Plain mode has no model-local spy; compare its products
                    # to the fully assessed spy mode below, as the original check did.
                    report = {"events": transport_events, "products": len(clock2["products"])}
                modes[mode] = clock2
                transports[f"{stop}-{mode}"] = report
            if modes["spy"]["products"] != modes["plain"]["products"]:
                raise AssertionError("atomic transport spy changes product observations")
        reports[revision] = {
            "process": cells,
            "default_transport": transports,
            "installed_equals_clock1": True,
            "matched_non_time_and_rng_identical": True,
            "default_nonnegative_transport_equals_prior": True,
            "nonnegative_zero_wait_and_dio_product_records_identical": True,
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path)
    parser.add_argument("prior_transport", type=Path)
    parser.add_argument("--sampled-events", required=True, type=int)
    parser.add_argument("--transport-events", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.log_root, args.prior_transport, args.sampled_events, args.transport_events)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for revision, report in result.items():
        print(revision, "atomic-decay corrected clock PASS; old defect red; matched non-time/RNG identical")
        print(
            "baseline failures by stop",
            {stop: cell["baseline"]["failures"] for stop, cell in report["process"].items()},
        )
        print("default nonnegative-model atomic transport exactly preserves prior parsed observations")


if __name__ == "__main__":
    main()
