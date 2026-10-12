"""Convert the natural-composition harvest, and generate the natural-target capture study from it.

``harvest --raw <dir>`` reads one directory per build that ``cpp/tools/harvest_natural.cc`` was run
on and writes ``data/g4/d1/geant4_natural_isotopes.csv`` and ``data/g4/d1/natural_harvest_builds.csv``.
``compare`` writes ``data/g4/d1/natural_target_capture.csv`` from the committed files; it is the step
``make audit`` runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openmucf.g4 import d1_natural as natural

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    harvest = sub.add_parser("harvest")
    harvest.add_argument("--raw", type=Path, required=True)
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.command == "harvest":
        isotopes, builds = natural.harvest(args.raw, ROOT)
        (ROOT / natural.ISOTOPES_RELPATH).write_bytes(natural.render_csv(isotopes, natural.ISOTOPE_COLUMNS))
        (ROOT / natural.BUILDS_RELPATH).write_bytes(natural.render_csv(builds, natural.BUILD_COLUMNS))
        print(f"harvest isotopes={len(isotopes)} elements={len({row['z'] for row in isotopes})} builds={len(builds)}")
        return
    rows = natural.compute(ROOT)
    (ROOT / natural.STUDY_RELPATH).write_bytes(natural.render_csv(rows, natural.STUDY_COLUMNS))
    compared = [row for row in rows if row["reference_a"]]
    print(f"study rows={len(rows)} compared={len(compared)} out={natural.STUDY_RELPATH}")


if __name__ == "__main__":
    main()
