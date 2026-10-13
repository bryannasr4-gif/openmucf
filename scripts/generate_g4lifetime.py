"""Convert the decay-rate harvest into its oracle, and generate the mean-life comparison.

``harvest --raw <dir>`` reads one directory per build that ``cpp/tools/harvest_decay.cc`` was run on and
writes ``data/g4/d1/d1_decay_rate.oracle``. ``compare`` writes ``data/g4/d1/mean_life_comparison.csv`` from
the committed files; it is the step ``make audit`` runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openmucf.g4 import d1_lifetime as lifetime

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    harvest = sub.add_parser("harvest")
    harvest.add_argument("--raw", type=Path, required=True)
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.command == "harvest":
        oracle = lifetime.harvest(args.raw, ROOT)
        (ROOT / lifetime.ORACLE_RELPATH).write_bytes(oracle)
        lines = oracle.count(b"\n")
        print(f"oracle lines={lines} out={lifetime.ORACLE_RELPATH}")
        return
    rows = lifetime.comparison_rows(ROOT)
    (ROOT / lifetime.COMPARISON_RELPATH).write_bytes(lifetime.render_csv(rows))
    compared = [row for row in rows if row["population"] == lifetime.COMPARED]
    print(f"comparison rows={len(rows)} compared={len(compared)} out={lifetime.COMPARISON_RELPATH}")


if __name__ == "__main__":
    main()
