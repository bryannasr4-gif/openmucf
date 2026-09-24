"""Convert the compiled selector harvest and derive its source call sites."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "openmucf/g4/d2.py"
spec = importlib.util.spec_from_file_location("g4_d2_reference", MODULE)
assert spec is not None and spec.loader is not None
d2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = d2
spec.loader.exec_module(d2)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def call_sites(sources: list[str], out: Path) -> None:
    rows: list[dict[str, str]] = []
    for item in sources:
        revision, directory = item.split("=", 1)
        source_root = Path(directory)
        commit = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        derived: set[str] = set()
        for path in sorted((source_root / "source").rglob("*")):
            if path.suffix not in (".cc", ".hh"):
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            rel = path.relative_to(source_root).as_posix()
            for number, line in enumerate(lines, 1):
                kind = ""
                if re.search(r"(?:->|\.)SelectZandA\(", line) and path.name not in ("G4ElementSelector.cc", "G4ElementSelector.hh"):
                    kind = "caller"
                elif re.search(r"public\s+G4HadronStoppingProcess\b", line):
                    kind = "derived"
                    derived.add(path.stem)
                elif re.search(r"(?:->|\.)SetElementSelector\(", line):
                    kind = "setter_call"
                elif "AtRestDoIt(" in line and path.stem in derived and path.suffix == ".hh":
                    kind = "atrest_declaration"
                if kind:
                    rows.append({"revision": revision, "commit": commit, "kind": kind,
                                 "file": rel, "line": str(number), "text": line.strip()})
    write_csv(out, ("revision", "commit", "kind", "file", "line", "text"), rows)
    print(f"call_sites rows={len(rows)} revisions={len(sources)}")


def harvest(raw: Path, out: Path) -> None:
    selector: list[dict[str, str]] = []
    bindings: list[dict[str, str]] = []
    builds: list[dict[str, str]] = []
    configs = (
        ("preserved-v11.4.2", "v11.4.2", "preserved"),
        ("preserved-v11.5.0.beta", "v11.5.0.beta", "preserved"),
        ("clean-v11.4.2", "v11.4.2", "clean"),
        ("clean-v11.5.0.beta", "v11.5.0.beta", "clean"),
    )
    harness_hash = hashlib.sha256((ROOT / "cpp/tools/harvest_d2.cc").read_bytes()).hexdigest()
    for build, revision, install_kind in configs:
        folder = raw / build
        hashes = (folder / "sha256.txt").read_text(encoding="utf-8").splitlines()
        if not any(harness_hash == line.split()[0] for line in hashes):
            raise ValueError(f"{build}: compiled harness source hash differs")
        builds.append({"build": build, "revision": revision, "install_kind": install_kind,
                       "geant4_version": (folder / "version.txt").read_text().strip(),
                       "gcc": (folder / "gcc.txt").read_text().strip(),
                       "harness_sha256": harness_hash,
                       "compile_flags": "-std=c++17 -O2 -ffp-contract=off"})
        for line in (folder / "selector.txt").read_text().splitlines():
            if not line.startswith("M\t"):
                continue
            parts = line.split("\t")
            if len(parts) != 10 or parts[0] != "M":
                raise ValueError(f"{build}: malformed selector line: {line}")
            selector.append(dict(zip(("build", "material", "element_index", "z", "atom_density",
                                      "isotope_n", "isotope_abundance", "isotope_at_half", "count", "draws"),
                                     (build, *parts[1:]), strict=True)))
        for physics_list in ("QBBC", "FTFP_BERT", "QBBC+helper"):
            for line in (folder / f"bindings_{physics_list}.txt").read_text().splitlines():
                if not line.startswith("B\t"):
                    continue
                parts = line.split("\t")
                if len(parts) != 6 or parts[0] != "B":
                    raise ValueError(f"{build}: malformed binding line: {line}")
                bindings.append(dict(zip(("build", "physics_list", "particle", "process", "class",
                                          "stopping_process", "atomic_capture"),
                                         (build, physics_list, *parts[1:]), strict=True)))
    write_csv(out / "selector_harvest.csv", ("build", "material", "element_index", "z", "atom_density",
                 "isotope_n", "isotope_abundance", "isotope_at_half", "count", "draws"), selector)
    write_csv(out / "selector_bindings.csv", ("build", "physics_list", "particle", "process", "class",
                 "stopping_process", "atomic_capture"), bindings)
    write_csv(out / "harvest_builds.csv", ("build", "revision", "install_kind", "geant4_version", "gcc",
                 "harness_sha256", "compile_flags"), builds)
    print(f"harvest builds={len(builds)} selector_rows={len(selector)} binding_rows={len(bindings)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sites = sub.add_parser("call-sites")
    sites.add_argument("--src", action="append", required=True)
    sites.add_argument("--out", type=Path, required=True)
    harvest_parser = sub.add_parser("harvest")
    harvest_parser.add_argument("--raw", type=Path, required=True)
    harvest_parser.add_argument("--out", type=Path, required=True)
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.command == "call-sites":
        call_sites(args.src, args.out)
    elif args.command == "harvest":
        harvest(args.raw, args.out)
    else:
        raise NotImplementedError("comparison requires the primary-row ledger")


if __name__ == "__main__":
    main()
