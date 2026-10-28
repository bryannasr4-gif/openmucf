"""Differential check of the compiled consumer transform (G4MuonicDataSemantics::BoundPairProblem).

The compiled function is run over a boundary corpus (subnormals, the reciprocal-overflow edge, nextafter
neighbours of every edge, zero capture, q = 1, DBL_MAX, NaN and both infinities in every primitive) and a
seeded random corpus. Each tuple must be admitted exactly when an independent simulation of the listed
consumer expressions -- the stopping route's total rate and waiting-time expression, the atomic route's
lifetime, stored partial lifetimes and the rates its decay step reads back from them -- admits it, and an
admitted tuple's capture and decay rates must be the simulation's to the bit.

usage: check_pair_transform.py <g4muonicdata_bound_rate_check>
"""

from __future__ import annotations

import math
import random
import subprocess
import sys

DBL_MAX = sys.float_info.max
VARIATE_MAX = 745.0


def divide(a: float, b: float) -> float:
    """a / b as binary64 hardware gives it: a nonzero number over zero is an infinity."""
    if b == 0:
        return math.copysign(math.inf, a) * math.copysign(1.0, b) if a != 0 else math.nan
    return a / b


def simulate(tau: float, q: float, free_tau: float) -> tuple[float, float] | None:
    """(capture, decay) when the listed transform is admissible, else None."""
    if not all(math.isfinite(v) for v in (tau, q, free_tau)) or not (tau > 0 and free_tau > 0 and 0 < q <= 1):
        return None
    decay = q / free_tau
    capture = 1 / tau - decay
    rate = capture + decay
    lifetime = divide(1.0, rate)
    dio, nc = divide(1.0, decay), divide(1.0, capture)
    read_rate = divide(1.0, dio) + divide(1.0, nc)
    quantities = (
        decay,
        1 / tau,
        capture,
        rate,
        lifetime,
        dio,
        read_rate,
        divide(VARIATE_MAX, rate),
        VARIATE_MAX * lifetime,
    )
    usable = (
        all(math.isfinite(x) for x in quantities)
        and 0 < decay <= 1 / free_tau
        and capture >= 0
        and rate > 0
        and lifetime > 0
        and 0 <= divide(capture, rate) <= 1
        and dio > 0
        and read_rate > 0
        and (capture == 0 and nc == math.inf or math.isfinite(nc) and nc > 0)
    )
    return (capture, decay) if usable else None


def neighbours(x: float) -> list[float]:
    return [math.nextafter(x, -math.inf), x, math.nextafter(x, math.inf)]


def corpus() -> list[tuple[float, float, float]]:
    scale = [
        5e-324,
        2.2250738585072014e-308,
        1 / DBL_MAX,
        1e-300,
        1e-3,
        1.0,
        863.7,
        2196.9811,
        1e300,
        1e306,
        DBL_MAX / VARIATE_MAX,
        DBL_MAX,
    ]
    times = [v for s in scale for v in neighbours(s)] + [0.0, -0.0, -1.0, math.nan, math.inf, -math.inf]
    qs = [0.0, 5e-324, 1e-300, 0.5, 0.993598, *neighbours(1.0), -0.5, math.nan, math.inf, -math.inf]
    tuples = [(tau, q, free_tau) for tau in times for q in qs for free_tau in times]
    for free_tau in times:
        tuples += [(tau, 1.0, free_tau) for tau in neighbours(free_tau)]
    tuples += [
        (DBL_MAX, 0.5, DBL_MAX),
        (863.7, 0.5, DBL_MAX),
        (math.nextafter(1e293, 0), 1.0, 1e293),
        (2196.9811, 1.0, 2196.9811),
    ]
    rng = random.Random(4172)
    for _ in range(40000):
        free_tau = (
            math.ldexp(rng.random(), rng.randint(-1074, 1024)) if rng.random() < 0.5 else rng.uniform(0, 1e4)
        )
        tau = rng.choice(
            [
                free_tau,
                math.nextafter(free_tau, 0),
                math.nextafter(free_tau, math.inf),
                math.ldexp(rng.random(), rng.randint(-1074, 1024)),
                rng.uniform(0, 1e4),
            ]
        )
        q = rng.choice(
            [rng.random(), 1.0, math.nextafter(1.0, 0), math.ldexp(rng.random(), rng.randint(-1074, 0))]
        )
        tuples.append((tau, q, free_tau))
    return tuples


def main() -> int:
    tuples = corpus()
    text = "".join(f"{float.hex(t)} {float.hex(q)} {float.hex(f)}\n" for t, q, f in tuples)
    run = subprocess.run(
        [sys.argv[1], "pair-transform"], input=text, capture_output=True, text=True, check=False
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith(("ADMIT", "REFUSE"))]
    if run.returncode != 0 or len(lines) != len(tuples):
        print(f"FAIL: rc {run.returncode}, {len(lines)} answers for {len(tuples)} tuples")
        return 1
    mismatches = admitted = 0
    for (tau, q, free_tau), line in zip(tuples, lines, strict=True):
        expected = simulate(tau, q, free_tau)
        if expected is None:
            ok = line == "REFUSE"
        else:
            admitted += 1
            parts = line.split()
            ok = parts[0] == "ADMIT" and (float.fromhex(parts[1]), float.fromhex(parts[2])) == expected
        if not ok:
            mismatches += 1
            if mismatches <= 20:
                print(
                    f"MISMATCH tau={tau!r} q={q!r} free_tau={free_tau!r}: "
                    f"compiled {line!r}, simulation {expected}"
                )
    print(
        f"tuples {len(tuples)} admitted {admitted} refused {len(tuples) - admitted} mismatches {mismatches}"
    )
    return 1 if mismatches or not 0 < admitted < len(tuples) else 0


if __name__ == "__main__":
    sys.exit(main())
