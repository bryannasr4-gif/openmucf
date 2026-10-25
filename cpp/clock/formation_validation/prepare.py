"""Prepare fresh, test-only formation-time comparator sources from a pristine Geant4 tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISIONS = ("v11.4.2", "v11.5.0.beta")
PROCESS_DIR = Path("source/processes/hadronic/stopping/src")
PROCESS_NAMES = (
    "G4MuonMinusAtomicCapture.cc",
    "G4HadronStoppingProcess.cc",
    "G4MuonicAtomDecay.cc",
    "G4MuMinusCapturePrecompound.cc",
)
CORRECTED = "fTime = time0 + std::max(rp->GetFormationTime(), 0.0);"


def apply_patch_file(patch: Path, kernel: Path, ceiling: Path) -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_CEILING_DIRECTORIES"] = str(ceiling.resolve())
    for operation in (("--check",), ()):
        subprocess.run(
            ["git", "-c", "core.autocrlf=false", "apply", *operation, str(patch)],
            cwd=kernel,
            env=env,
            check=True,
            timeout=60,
        )


def variants(traced: str) -> dict[str, str]:
    if traced.count(CORRECTED) != 1 or traced.count("#include <algorithm>\n") != 1:
        raise ValueError("expected exactly one corrected getter and explicit algorithm include")
    original = traced.replace("#include <algorithm>\n", "")
    return {
        "old": original.replace(CORRECTED, "fTime = time0 + rp->GetTOF();"),
        "new": traced,
        "zero": original.replace(CORRECTED, "fTime = time0;"),
    }


def prepare(revision: str, source: Path, output: Path) -> Path:
    if revision not in REVISIONS:
        raise ValueError("unsupported Geant4 revision")
    source, output = source.resolve(), output.resolve()
    # Refuse reuse: source trees and previous validation artifacts are never changed.
    output.mkdir(parents=True, exist_ok=False)
    kernel, app = output / "kernel", output / "app"
    (kernel / PROCESS_DIR).mkdir(parents=True)
    app.mkdir()
    for name in PROCESS_NAMES:
        text = (source / PROCESS_DIR / name).read_text(encoding="utf-8")
        (kernel / PROCESS_DIR / name).write_bytes(text.encode("utf-8"))
    production = HERE.parents[1] / "patches" / f"g4-{revision}-stopped-muon-clock-v3.patch"
    trace = HERE / f"trace-g4-{revision}.patch"
    for patch in (production, trace):
        apply_patch_file(patch, kernel, output)
        body = (kernel / PROCESS_DIR / PROCESS_NAMES[-1]).read_text(encoding="utf-8")
        marker = CORRECTED if patch == production else "AUDIT_PICK"
        if body.count(marker) != 1:
            raise RuntimeError(f"patch {patch.name} was not applied")
    for name in PROCESS_NAMES:
        path = kernel / PROCESS_DIR / name
        path.write_bytes(path.read_text(encoding="utf-8").encode("utf-8"))
    tested = json.loads((HERE / "tested_sources.json").read_text(encoding="utf-8"))
    for name in ("formation_observer.cc", "audit_transport.cc", "CMakeLists.txt"):
        raw = (HERE / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != tested["files"][name]:
            raise ValueError(f"published observer/build source changed: {name}")
        shutil.copyfile(HERE / name, app / name)
    traced = (kernel / PROCESS_DIR / PROCESS_NAMES[-1]).read_text(encoding="utf-8")
    hashes = {}
    for variant, text in variants(traced).items():
        raw = text.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != tested["revisions"][revision][variant]:
            raise ValueError(f"source does not match tested {revision} {variant} body")
        (app / f"model-{variant}.cc").write_bytes(raw)
        hashes[variant] = digest
    (output / "prepared_sources.json").write_text(
        json.dumps({"revision": revision, "model_sha256": hashes}, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", choices=REVISIONS, required=True)
    parser.add_argument("--geant4-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.revision, args.geant4_source, args.output))


if __name__ == "__main__":
    main()
