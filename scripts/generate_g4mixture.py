"""Generate the natural-target answer classes from the committed D1 files.

``classes`` writes ``data/g4/d1/record_classes.csv``; ``answers`` writes ``data/g4/d1/natural_target_answers.csv``.
Both are steps ``make audit`` runs.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from openmucf.g4 import d1_mixture as mixture

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("steps", nargs="+", choices=("classes", "answers"))
    args = parser.parse_args()
    classes = mixture.record_classes(ROOT)
    if "classes" in args.steps:
        (ROOT / mixture.CLASSES_RELPATH).write_bytes(mixture.render_csv(classes, mixture.CLASS_COLUMNS))
        counts = collections.Counter((row["profile"], row["answer_class"]) for row in classes)
        print(
            f"classes rows={len(classes)} " + " ".join(f"{p}:{k}={n}" for (p, k), n in sorted(counts.items()))
        )
    if "answers" in args.steps:
        rows = mixture.answers(ROOT, classes)
        (ROOT / mixture.ANSWERS_RELPATH).write_bytes(mixture.render_csv(rows, mixture.ANSWER_COLUMNS))
        print(f"answers rows={len(rows)} elements={len({row['z'] for row in rows})}")


if __name__ == "__main__":
    main()
