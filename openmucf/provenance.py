"""openmucf.provenance -- machine-checkable provenance for shipped headline numbers.

Doc-drift (stale scoreboards, miscounted rows) is this project's recurring bug class. ``make audit``
already catches *regeneration* drift for the generated docs; this module adds the missing half: a typed
manifest recording, for every headline number, the formatted value as it appears in a doc, a regex that
anchors it in context, and a source type (``derivation`` | ``ledger_row`` | ``registered_prior``).
``check_manifest`` fails if any tracked value no longer appears in its doc, so a number cannot silently
diverge between a doc and its recorded source.

Not part of the eager-import surface (like ``calibrate``/``validate``/``forecast``); reached as a
submodule or via ``python -m openmucf.provenance --check FINDINGS_MANIFEST.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SOURCE_TYPES = ("derivation", "ledger_row", "registered_prior")


@dataclass(frozen=True)
class ManifestEntry:
    id: str  # e.g. "sobol_xmu_ST_R"
    value: str  # the FORMATTED string as it appears in the doc, e.g. "0.620"
    pattern: str  # regex that must match >=1 time in `doc` (anchors the value in context)
    source_type: str  # "derivation" | "ledger_row" | "registered_prior"
    source: str  # script path, ledger symbol, or prior name
    doc: str  # "FINDINGS.md"


def file_sha256(path) -> str:
    """SHA-256 over the LF-normalized UTF-8 text of ``path`` (immune to CRLF checkouts).

    Same recipe as ``openmucf.forecast.ledger_sha256`` so digests are comparable across the codebase.
    """
    text = Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(obj) -> str:
    """Deterministic JSON for reviewable diffs: sorted keys, 2-space indent, ASCII-only."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True)


def write_manifest(path, entries, inputs: dict, generated_by="scripts/generate_findings.py") -> None:
    """Write the provenance manifest as ``{generated_by, inputs, entries}`` JSON.

    ``inputs`` records the SHA-256 of every deterministic input (ledger CSVs, ``repr(uq.PARAMS)``) plus
    the seeds, so a manifest built from different inputs diffs visibly. ``entries`` is a list of
    :class:`ManifestEntry`. ``generated_by`` names the producing script (defaults to the findings
    generator; the twin audit passes its own).
    """
    payload = {
        "generated_by": generated_by,
        "inputs": inputs,
        "entries": [asdict(e) for e in entries],
    }
    Path(path).write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def check_manifest(manifest_path, repo_root=".") -> list[str]:
    """Verify every manifest entry against its doc. Returns a list of failures (empty == all pass).

    For each entry: read ``entry["doc"]``, require ``re.search(entry["pattern"])`` finds >=1 match, and
    require ``entry["value"]`` is a substring of the matched text.
    """
    failures: list[str] = []
    root = Path(repo_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    doc_cache: dict[str, str | None] = {}
    for entry in manifest["entries"]:
        doc = entry["doc"]
        if doc not in doc_cache:
            doc_path = root / doc
            doc_cache[doc] = (
                doc_path.read_bytes().decode("utf-8").replace("\r\n", "\n")
                if doc_path.exists()
                else None
            )
        text = doc_cache[doc]
        if text is None:
            failures.append(f"{entry['id']}: doc '{doc}' not found under {root}")
            continue
        m = re.search(entry["pattern"], text)
        if m is None:
            failures.append(f"{entry['id']}: pattern {entry['pattern']!r} not found in {doc}")
            continue
        if entry["value"] not in m.group(0):
            failures.append(
                f"{entry['id']}: value {entry['value']!r} not in matched text {m.group(0)!r} ({doc})"
            )
    return failures


def main(argv=None) -> int:
    """`python -m openmucf.provenance --check A.json [B.json ...]` -> 0 on success, 1 on any failure.

    Accepts MULTIPLE manifest paths after ``--check`` (each is verified against its docs), so one audit
    line covers every manifest in the tree (e.g. FINDINGS_MANIFEST.json + TWIN_MANIFEST.json).
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) >= 2 and argv[0] == "--check":
        manifest_paths = [a for a in argv[1:] if a != "--check"]  # tolerate repeated --check flags
    else:
        print(
            "usage: python -m openmucf.provenance --check MANIFEST.json [MANIFEST2.json ...]",
            file=sys.stderr,
        )
        return 2
    all_failures: list[str] = []
    total_entries = 0
    for mp in manifest_paths:
        failures = check_manifest(mp)
        all_failures += [f"{mp}: {f}" for f in failures]
        total_entries += len(json.loads(Path(mp).read_text(encoding="utf-8"))["entries"])
    if all_failures:
        print(f"provenance check FAILED ({len(all_failures)} problem(s)):", file=sys.stderr)
        for f in all_failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        f"provenance OK: {total_entries} manifest entries across "
        f"{len(manifest_paths)} manifest(s) verified against their docs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
