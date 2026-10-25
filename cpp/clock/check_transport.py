"""Check passive clock_transport records, including atom-decay descendants."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REVISIONS = ("v11.4.2", "v11.5.0.beta")


def parse(text: str) -> dict:
    result = {"products": [], "atoms": {}, "waits": {}, "locals": defaultdict(dict), "branches": {}}
    current = None
    for line in text.splitlines():
        f = line.split()
        if line.startswith("ATOM_START "):
            current = int(f[1])
            result["atoms"][current] = float.fromhex(f[2])
        elif line.startswith("G4MuonicAtomDecay::DecayIt time set to:") and current not in result["waits"]:
            result["waits"][current] = float(line.split(": ")[-1].split("[")[0])
        elif line.startswith("MODEL_LOCAL "):
            result["locals"][int(f[1])][int(f[2])] = float.fromhex(f[3])
        elif line.startswith("BOUND_WAIT "):
            result["waits"][int(f[1])] = float.fromhex(f[2])
            result["branches"][int(f[1])] = int(f[4])
        elif line.startswith("PRODUCT "):
            result["products"].append(
                {
                    "event": int(f[1]),
                    "index": int(f[2]),
                    "name": f[3],
                    "phase": f[5],
                    "time": float.fromhex(f[7]),
                    "energy": float.fromhex(f[9]),
                }
            )
    if not result["products"]:
        raise AssertionError("empty transport observation")
    if any(not math.isfinite(p["time"]) for p in result["products"]):
        raise AssertionError("nonfinite transported clock")
    return result


def assess(data: dict, route: str, stop: float, expected_events: int, *, old: bool = False) -> dict:
    counts = Counter()
    failed = []
    events = set()
    nc_events, dio_events = set(), set()
    for p in data["products"]:
        event, phase = p["event"], p["phase"]
        events.add(event)
        if route == "atomic":
            atom_start = 2 * stop if old else stop
            if phase == "atom":
                expected = atom_start
            elif phase == "other":
                expected = stop  # instantaneous stock formation-cascade secondaries
            elif phase == "atom_daughter":
                wait = data["waits"][event]
                if not math.isfinite(wait) or wait < 0.0:
                    raise AssertionError("invalid observed atom waiting time")
                if event in data["locals"]:
                    tau = data["locals"][event][p["index"]]
                    if not math.isfinite(tau) or tau < 0.0:
                        raise AssertionError("atomic precompound witness outside tested nonnegative domain")
                    nc_events.add(event)
                    counts["positive_model_delay"] += tau > 0.0
                else:
                    tau = 0.0
                    dio_events.add(event)
                expected = atom_start + wait + tau
            else:
                raise AssertionError(f"unrecognized atomic product {phase}")
        else:
            if phase == "em":
                expected = stop
            elif phase in ("capture", "decay"):
                wait = data["waits"][event]
                if not math.isfinite(wait) or wait < 0.0:
                    raise AssertionError("invalid observed bound waiting time")
                if phase == "capture":
                    nc_events.add(event)
                    expected = stop + (max(0.0, wait - 1.0) if old else wait)
                else:
                    dio_events.add(event)
                    expected = stop + wait
            else:
                raise AssertionError(f"unrecognized stopping product {phase}")
        counts[phase] += 1
        if not math.isclose(p["time"], expected, rel_tol=0.0, abs_tol=1.0e-9):
            failed.append({"event": event, "phase": phase, "actual": p["time"], "expected": expected})
    if expected_events <= 0 or events != set(range(expected_events)) or not nc_events or not dio_events:
        raise AssertionError("missing transport events or either branch")
    if route == "atomic" and (len(data["atoms"]) != len(events) or not counts["positive_model_delay"]):
        raise AssertionError("missing atomic or positive-delay witness")
    return {
        "counts": dict(counts),
        "events": len(events),
        "capture_events": len(nc_events),
        "decay_events": len(dio_events),
        "failures": len(failed),
        "first_failures": failed[:3],
    }


def check(root: Path, expected_events: dict[tuple[str, int], int]) -> dict:
    reports = {}
    for rev in REVISIONS:
        cells = {}
        for variant in ("installed", "corrected"):
            for route in ("atomic", "stopping"):
                for stop in (0, 250, 1000):
                    prefix = root / rev / f"{variant}-{route}-{stop}"
                    spy = parse(Path(str(prefix) + "-spy.log").read_text())
                    plain = parse(Path(str(prefix) + "-plain.log").read_text())
                    if spy["products"] != plain["products"]:
                        raise AssertionError(f"spy changes transported products: {prefix}")
                    report = assess(spy, route, stop, expected_events[(route, stop)])
                    if variant == "corrected" and report["failures"]:
                        raise AssertionError(f"corrected transported clock failed: {prefix}")
                    if variant == "installed":
                        if assess(spy, route, stop, expected_events[(route, stop)], old=True)["failures"]:
                            raise AssertionError(f"old transported law not reproduced: {prefix}")
                        if (route == "stopping" or stop > 0) and not report["failures"]:
                            raise AssertionError(f"old defect did not go red: {prefix}")
                    cells[f"{variant}-{route}-{stop}"] = report
        reports[rev] = cells
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--atomic-events",
        type=int,
        required=True,
        help="Exact event count passed to every atomic transport invocation",
    )
    parser.add_argument(
        "--stopping-events",
        type=int,
        required=True,
        help="Exact event count passed to stopping transport invocations",
    )
    parser.add_argument(
        "--stopping-250-events", type=int, help="Exact count for an extended stopping t0=250 ns invocation"
    )
    args = parser.parse_args()
    expected_events = {
        (route, stop): count
        for route, count in (("atomic", args.atomic_events), ("stopping", args.stopping_events))
        for stop in (0, 250, 1000)
    }
    if args.stopping_250_events is not None:
        expected_events[("stopping", 250)] = args.stopping_250_events
    result = check(args.log_root, expected_events)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for rev, cells in result.items():
        print(
            rev,
            "transported atomic creation, DIO/capture descendants, stopping products PASS; "
            "spies identical; old defects red",
        )
        print(
            "  corrected cells",
            {
                name: {k: v for k, v in report.items() if k.endswith("events")}
                for name, report in cells.items()
                if name.startswith("corrected")
            },
        )


if __name__ == "__main__":
    main()
