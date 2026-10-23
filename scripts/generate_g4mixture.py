"""Generate the natural-target answer classes and the mixture comparison policy from the committed D1 files.

``classes`` writes ``data/g4/d1/record_classes.csv``; ``answers`` writes ``data/g4/d1/natural_target_answers.csv``;
both are steps ``make audit`` runs. ``estimators`` prints, for each measured mean life of a natural target, the
facts its source prints about its fit and the label (ESTABLISHED or SURROGATE) they give it. ``policy --ms-per-stop
<ms>`` writes ``data/g4/d1/natural_mixture_policy.json`` from the committed files and the measured cost of a
transported stop, and prints the power calculation behind it. ``compare`` writes
``data/g4/d1/natural_mixture_comparison.csv``, a step ``make audit`` runs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
from pathlib import Path

from openmucf.g4 import d1_mixture as mixture

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("steps", nargs="+", choices=("classes", "answers", "estimators", "policy", "compare"))
    parser.add_argument("--ms-per-stop", type=float)
    args = parser.parse_args()
    if "classes" in args.steps or "answers" in args.steps:
        classes = mixture.record_classes(ROOT)
    if "classes" in args.steps:
        (ROOT / mixture.CLASSES_RELPATH).write_bytes(mixture.render_csv(classes, mixture.CLASS_COLUMNS))
        counts = collections.Counter((row["profile"], row["answer_class"]) for row in classes)
        listed = " ".join(f"{p}:{k}={n}" for (p, k), n in sorted(counts.items()))
        print(f"classes rows={len(classes)} {listed}")
    if "answers" in args.steps:
        rows = mixture.answers(ROOT, classes)
        (ROOT / mixture.ANSWERS_RELPATH).write_bytes(mixture.render_csv(rows, mixture.ANSWER_COLUMNS))
        print(f"answers rows={len(rows)} elements={len({row['z'] for row in rows})}")
    if "estimators" in args.steps:
        facts = mixture.load_estimators(ROOT)
        for row in facts:
            print(
                f"FACT {row['source']} [{row['scope']}] {row['fact']}: {row['status']}: {row['value'] or '-'} "
                f"({row['locator']})"
            )
        refs = mixture.references(ROOT)
        labels: collections.Counter[str] = collections.Counter()
        for ref in refs:
            label, missing = mixture.estimator_label(mixture.row_facts(ROOT, ref, facts))
            labels[label] += 1
            print(
                f"ROW {ref.source} {ref.locator} {ref.label} Z={ref.z}: {label}; missing: {', '.join(missing)}"
            )
        print(
            f"rows={len(refs)} elements={len({r.z for r in refs})} "
            + " ".join(f"{k}={v}" for k, v in labels.items())
        )
    if "policy" in args.steps:
        if args.ms_per_stop is None:
            parser.error("policy needs --ms-per-stop")
        policy, log = mixture.build_policy(ROOT, args.ms_per_stop)
        data = mixture.render_policy(policy)
        (ROOT / mixture.POLICY_RELPATH).write_bytes(data)
        print("\n".join(log))
        print(f"policy {mixture.POLICY_RELPATH} sha256 {hashlib.sha256(data).hexdigest()}")
    if "compare" in args.steps:
        compare()


def compare() -> None:
    rows = mixture.comparison_rows(ROOT)
    (ROOT / mixture.COMPARISON_RELPATH).write_bytes(mixture.render_csv(rows, mixture.COMPARISON_COLUMNS))
    print(f"comparison rows={len(rows)} distinct={len({r['comparison'] for r in rows})}")
    for key, value in mixture.science(rows).items():
        print(f"science {key}: {value}")


if __name__ == "__main__":
    main()
