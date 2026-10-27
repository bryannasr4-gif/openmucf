"""Generate and check the evaluated bound-muon rate pair.

``generate`` writes ``data/g4/d1/bound_rate_evaluation.json`` (every derived quantity and decision of the
evaluated pair) and ``data/g4/d1/bound_decay_response_v1.json`` (the D1 roles of the shared D1/D2 convention
record) from the committed primitives ``data/g4/d1/bound_rate_inputs.json``, the decay-rate oracle and the
compendium rows. ``check`` recomputes both from the primitives and refuses a committed file whose fields differ
(numbers to 1e-12 relative: the reference ranges come from a numerical search, so they are not byte-pinned), and
checks that the dataset's pair table carries exactly the production tuples. ``make audit`` runs ``check``. The
work is delegated to :mod:`openmucf.g4.d1_lifetime`, which owns the measurement records this pair extends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openmucf.g4 import d1_evaluated as evaluated
from openmucf.g4 import d1_lifetime as lifetime
from openmucf.g4 import provenance, spec

ROOT = Path(__file__).resolve().parents[1]


def render(obj: object) -> bytes:
    """Indented, sorted, LF-terminated JSON; floats by their shortest round-trip repr."""
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def generate(root: Path) -> None:
    record = lifetime.evaluation(root)
    convention = lifetime.convention(root, record)
    (root / lifetime.EVALUATION_RELPATH).write_bytes(render(record))
    (root / lifetime.CONVENTION_RELPATH).write_bytes(render(convention))
    decisions = record["decisions"]
    assert isinstance(decisions, dict)
    print(f"evaluation out={lifetime.EVALUATION_RELPATH} convention out={lifetime.CONVENTION_RELPATH} "
          f"payload_sha256={convention['payload_sha256']}")
    for name, value in decisions.items():
        print(f"decision {name} {value['verdict']} L={value['L']!r} U={value['U']!r}")


def check(root: Path) -> None:
    record = lifetime.evaluation(root)
    evaluated.check_evaluation((root / lifetime.EVALUATION_RELPATH).read_bytes(), record)
    convention = lifetime.convention(root, record)
    committed = evaluated.parse_json((root / lifetime.CONVENTION_RELPATH).read_bytes())
    if evaluated.payload_digest(committed.get("payload")) != committed.get("payload_sha256"):
        raise SystemExit("evaluated check FAILED: the convention record's payload digest does not match its payload")
    if evaluated.canonical_bytes(committed) != evaluated.canonical_bytes(convention):
        raise SystemExit("evaluated check FAILED: the convention record differs from its recomputation")
    table = spec.parse((root / lifetime.PAIR_LAYER1_RELPATH).read_bytes().decode("ascii"))
    raw = (root / lifetime.PAIR_LAYER2_RELPATH).read_bytes()
    provenance.check_canonical_bytes(raw)
    provenance.check_source_digest(table, raw)
    provenance.check_against_table(table, provenance.from_json_obj(evaluated.parse_json(raw)))
    expected = [(z, a, float(tau), float(q), float(free_tau)) for z, a, tau, q, free_tau in
                lifetime.evaluated_tuples(root)]
    declared = {key: table.directives.get(key) for key in ("PROFILE", "TABLE", "UNITS", "COLUMNS", "SEAM")}
    wanted = {"PROFILE": evaluated.PAIR_PROFILE, "TABLE": evaluated.PAIR_TABLE, "UNITS": evaluated.PAIR_UNITS,
              "COLUMNS": " ".join(evaluated.PAIR_COLUMNS), "SEAM": "d1_nuclear_capture"}
    if declared != wanted or not table.directives.get("VALIDITY", "").endswith(" A:listed"):
        raise SystemExit(f"evaluated check FAILED: the pair table declares {declared}, not {wanted} with A:listed")
    if [tuple(record) for record in table.records] != expected:
        raise SystemExit("evaluated check FAILED: the pair table does not carry exactly the production tuples")
    print(f"evaluated check OK: {lifetime.EVALUATION_RELPATH}, {lifetime.CONVENTION_RELPATH} and "
          f"{lifetime.PAIR_LAYER1_RELPATH} equal their recomputation")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "check"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.root)
    else:
        check(args.root)


if __name__ == "__main__":
    main()
