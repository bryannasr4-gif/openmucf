"""Compare the committed stopped-muon cells with the measured yields under the pre-registration's rule.

  python scripts/generate_stopped_mu.py compare [--prereg <file>]

reads `data/g4/stopped_mu/preregistration.json` and `cells.jsonl`, admits every cell, builds one comparison
row per row of the pre-registration and judges it in exact arithmetic, and writes `comparison.csv` and
`verdicts.csv` beside them. No float decides a byte of either file.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "g4" / "stopped_mu"
MODULE = ROOT / "openmucf" / "g4" / "stopped_mu.py"
spec = importlib.util.spec_from_file_location("g4_stopped_mu_reference", MODULE)
assert spec is not None and spec.loader is not None
sm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sm
spec.loader.exec_module(sm)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def comparison(prereg: Path = DATA / "preregistration.json", cells: Path = DATA / "cells.jsonl"):
    """(policy, cells, comparison rows, verdicts) from the committed files."""
    policy = sm.load_policy(prereg)
    admitted = sm.load_cells(cells, policy)
    rows = sm.comparison_rows(policy, admitted)
    return policy, admitted, rows, sm.judge(policy, rows)


def compare(prereg: Path = DATA / "preregistration.json") -> None:
    _, _, rows, verdicts = comparison(prereg)
    write_csv(DATA / "comparison.csv", sm.COMPARISON_COLUMNS, rows)
    write_csv(
        DATA / "verdicts.csv",
        sm.VERDICT_COLUMNS,
        [{"row_id": v.row_id, "verdict": v.verdict, "kind": v.kind, "pull": v.pull} for v in verdicts],
    )
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    print(f"stopped_mu compare rows={len(rows)} " + " ".join(f"{k}={counts[k]}" for k in sorted(counts)))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--prereg", default=str(DATA / "preregistration.json"))
    args = parser.parse_args()
    if args.cmd == "compare":
        compare(Path(args.prereg))


if __name__ == "__main__":
    main()
