"""Build, run, and check the declared Geant4 transport matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MATRIX = json.loads((HERE / "matrix.json").read_text(encoding="utf-8"))
PATCHES = ROOT / "cpp" / "patches"
SNIPPET = ROOT / "data" / "g4" / "d1" / "geant4_add_dataset.snippet"
SOURCES = ("matrix.json", "g4muonic_transport.cc", "CMakeLists.txt", "transport.py")
RUNS_DIR = "runs_precreated"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    pairs = [(p.relative_to(path).as_posix(), digest(p)) for p in sorted(path.rglob("*")) if p.is_file()]
    payload = json.dumps(pairs, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"non-empty target directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
            stdout: Path | None = None, history: list[dict[str, Any]] | None = None) -> str:
    start = time.monotonic()
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    record = {"argv": argv, "rc": result.returncode, "wall_s": time.monotonic() - start}
    if history is not None:
        history.append(record)
    if stdout is not None:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed {record}: {(result.stdout + result.stderr)[-2500:]}")
    return result.stdout


def generator() -> Any:
    path = ROOT / "scripts" / "generate_g4data.py"
    spec = importlib.util.spec_from_file_location("transport_generate_g4data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stage_dataset(out: Path) -> None:
    fresh(out)
    gen = generator()
    _, archive = gen.build_dataset_artifacts()
    name = gen.emit.archive_name(gen.DATASET_NAME, gen.DATASET_VERSION)
    actual = hashlib.md5(archive).hexdigest()
    declared = re.search(r"^\s*MD5SUM\s+([0-9a-f]{32})$", SNIPPET.read_text(encoding="utf-8"), re.M)
    if declared is None or actual != declared.group(1):
        raise RuntimeError(f"archive md5 {actual} disagrees with snippet")
    (out / name).write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (out / member.name).resolve()
            if not target.is_relative_to(out.resolve()):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tar.extractall(out, filter="data")
    members = {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob("*"))
               if p.is_file() and p.name != name}
    save(out / "stage.json", {"name": gen.DATASET_NAME, "version": gen.DATASET_VERSION,
                              "archive": name, "md5": actual, "members": members})
    print(f"STAGED {name} md5={actual} members={len(members)}")


def dataset_dir(out: Path) -> Path:
    info = json.loads((out / "stage.json").read_text(encoding="utf-8"))
    return out / f"{info['name']}{info['version']}"


def farm(tag: str, physics_data: Path, dataset: Path, work: Path) -> None:
    parent = work / tag
    target = parent / "farm"
    nodata = parent / "farm_nodata"
    fresh(target)
    fresh(nodata)
    staged = work / "dataset"
    if not staged.exists():
        shutil.copytree(dataset, staged)
    elif tree_digest(staged) != tree_digest(dataset):
        raise RuntimeError("staged dataset differs from the work copy")
    entries: list[dict[str, str]] = []
    for item in sorted(physics_data.iterdir()):
        if not item.is_dir():
            continue
        for root in (target, nodata):
            (root / item.name).symlink_to(item.resolve(), target_is_directory=True)
        entries.append({"name": item.name, "target": str(item.resolve()), "sha256": tree_digest(item)})
    source = dataset_dir(staged)
    shutil.copytree(source, target / source.name)
    entries.append({"name": source.name, "target": str(source), "sha256": tree_digest(source)})
    save(parent / "farm.json", {"entries": entries, "farm": str(target), "nodata": str(nodata)})
    print(f"FARM {tag} physics={len(entries) - 1} dataset={source.name}")


def geant4_dir(install: Path) -> Path:
    found = list(install.rglob("Geant4Config.cmake"))
    if len(found) != 1:
        raise RuntimeError(f"expected one Geant4Config.cmake under {install}, found {len(found)}")
    return found[0].parent


def harness(install: Path, target: Path, patched: bool, history: list[dict[str, Any]]) -> Path:
    fresh(target)
    command(["cmake", "-S", str(HERE), "-B", str(target),
             f"-DGeant4_DIR={geant4_dir(install)}",
             f"-DG4MUONIC_TRANSPORT_PATCHED={'ON' if patched else 'OFF'}",
             "-DCMAKE_BUILD_TYPE=Release"], history=history, stdout=target / "configure.log")
    command(["cmake", "--build", str(target), "-j6"], history=history, stdout=target / "build.log")
    binary = target / "g4muonic_transport"
    if not binary.is_file():
        raise RuntimeError(f"missing harness executable: {binary}")
    return binary


def preserved(tag: str, install: Path, work: Path) -> None:
    target = work / tag / "preserved"
    fresh(target)
    history: list[dict[str, Any]] = []
    binary = harness(install, target / "harness", False, history)
    save(target / "preserved.json", {"install": str(install), "binary": str(binary),
                                     "binary_sha256": digest(binary), "commands": history})
    print(f"PRESERVED {tag} harness={binary} sha256={digest(binary)}")


def env_for(install: Path, data: Path, work: Path, profiles: dict[str, str] | None = None) -> dict[str, str]:
    result = {"PATH": os.environ["PATH"], "HOME": str(work),
              "LD_LIBRARY_PATH": str(install / "lib"), "GEANT4_DATA_DIR": str(data)}
    if profiles:
        result.update(profiles)
    return result


def cell_argv(binary: Path, route: str, target: dict[str, int], events: int, threads: int,
              opt_in: str, out: Path, config_only: bool = False) -> list[str]:
    argv = [str(binary), "--route", route, "--opt-in", opt_in,
            "--Z", str(target["Z"]), "--A", str(target["A"]),
            "--events", str(events), "--threads", str(threads),
            "--seed-base", str(MATRIX["seed_base"]), "--seed-stride", str(MATRIX["seed_stride"]),
            "--out", str(out)]
    if config_only:
        argv.append("--config-only")
    return argv


def throughput(tag: str, work: Path, events: int) -> None:
    parent = work / tag
    spec = json.loads((parent / "preserved/preserved.json").read_text(encoding="utf-8"))
    binary = Path(spec["binary"])
    install = Path(spec["install"])
    data = parent / "farm"
    samples: list[dict[str, Any]] = []
    for route in MATRIX["routes"]:
        for target in MATRIX["targets"]:
            out = parent / "probe" / route / f"{target['Z']}-{target['A']}" / "config"
            fresh(out)
            argv = cell_argv(binary, route, target, events, 1, "off", out, True)
            command(argv, env=env_for(install, data, work), stdout=out / "stdout.log")
            config = (out / "config.txt").read_text(encoding="utf-8")
            print(f"CONFIG_ONLY {tag} {route} {target['Z']},{target['A']} {config.splitlines()[0]}")
            sample_dir = out.parent / "sample"
            fresh(sample_dir)
            argv = cell_argv(binary, route, target, events, 1, "off", sample_dir)
            start = time.monotonic()
            command(argv, env=env_for(install, data, work), stdout=sample_dir / "stdout.log")
            elapsed = time.monotonic() - start
            size = sum(p.stat().st_size for p in sample_dir.glob("records-*.txt"))
            row = {"route": route, "Z": target["Z"], "A": target["A"], "events": events,
                   "events_per_s": events / elapsed, "bytes_per_event": size / events,
                   "wall_s": elapsed}
            samples.append(row)
            print(f"THROUGHPUT {tag} {route} {target['Z']},{target['A']} "
                  f"events_per_s={row['events_per_s']:.3f} bytes_per_event={row['bytes_per_event']:.1f}")
    factor = len([m for m in MATRIX["modes"] if m != "preserved"]) * len(MATRIX["threads"])
    projected = sum(row["wall_s"] * MATRIX["events"] / events * factor for row in samples)
    save(parent / "throughput.json", {"samples": samples, "projected_w4_s": projected})
    print(f"PROJECTED_W4_S {tag} {projected:.1f}")


def patch_paths(tag: str) -> list[Path]:
    return [PATCHES / f"g4-{tag}-{kind}.patch" for kind in ("muonicdata", "register-dataset")]


def build_dir(work: Path, tag: str, kind: str) -> Path:
    locations = work / tag / "build_locations.json"
    names = json.loads(locations.read_text(encoding="utf-8")) if locations.exists() else {}
    return work / tag / names.get(kind, kind)


def record_build_dir(work: Path, tag: str, kind: str, parent: Path) -> None:
    locations = work / tag / "build_locations.json"
    names = json.loads(locations.read_text(encoding="utf-8")) if locations.exists() else {}
    names[kind] = parent.name
    save(locations, names)


def version_matches(tag: str, version: str) -> bool:
    match = re.fullmatch(r"v(\d+\.\d+\.\d+)(?:\.beta(?:\.\d+)?)?", tag)
    return match is not None and version == match.group(1)


def build(tag: str, source: Path, kind: str, work: Path, slot: str | None = None) -> None:
    parent = work / tag / (kind if slot is None else f"{kind}_{slot}")
    fresh(parent)
    src = parent / "src"
    src.mkdir()
    history: list[dict[str, Any]] = []
    actual = command(["git", "-c", "safe.directory=*", "-C", str(source), "rev-parse", f"{tag}^{{commit}}"],
                     history=history)
    commit = actual.strip()
    if commit != MATRIX["revisions"][tag]:
        raise RuntimeError(f"tag {tag} resolves to {commit}, expected {MATRIX['revisions'][tag]}")
    archive = parent / "source.tar"
    archive_start = time.monotonic()
    archive_argv = ["git", "-c", "safe.directory=*", "-C", str(source), "archive", tag]
    with archive.open("wb") as output:
        result = subprocess.run(archive_argv,
                                stdout=output, check=False)
    history.append({"argv": archive_argv, "rc": result.returncode,
                    "wall_s": time.monotonic() - archive_start})
    if result.returncode != 0:
        raise RuntimeError(f"git archive failed: {result.returncode}")
    with tarfile.open(archive) as tar:
        tar.extractall(src, filter="data")
    used_patches: list[dict[str, str]] = []
    if kind != "pristine":
        for patch in patch_paths(tag):
            command(["git", "apply", "--check", str(patch)], cwd=src, history=history)
            command(["git", "apply", str(patch)], cwd=src, history=history)
            used_patches.append({"path": patch.relative_to(ROOT).as_posix(), "sha256": digest(patch)})
    removed = ""
    if kind == "mutant":
        path = src / "source/processes/hadronic/stopping/src/G4EmCaptureCascade.cc"
        needle = b"    if (overlaid) G4MuonicDataOverlay::CheckCascadeLevels(Z, A, fLevelEnergy, 14);\n"
        data = path.read_bytes()
        if data.count(needle) != 1:
            raise RuntimeError("mutant call site count is not one")
        path.write_bytes(data.replace(needle, b""))
        if path.read_bytes().count(needle) != 0:
            raise RuntimeError("mutant call site survived")
        removed = needle.decode().strip()
    install = parent / "install"
    builddir = parent / "build"
    options = [f"-D{name}={value}" for name, value in MATRIX["configure"].items()]
    options += [f"-DCMAKE_INSTALL_PREFIX={install}"]
    configure_argv = ["cmake", "-S", str(src), "-B", str(builddir), *options]
    command(configure_argv, history=history,
            stdout=parent / "configure.log")
    command(["cmake", "--build", str(builddir), "-j" + str(MATRIX["jobs"])], history=history,
            stdout=parent / "compile.log")
    command(["cmake", "--install", str(builddir)], history=history, stdout=parent / "install.log")
    compile_commands = json.loads((builddir / "compile_commands.json").read_text(encoding="utf-8"))
    flagged = sum("-ffp-contract=off" in row.get("command", "") for row in compile_commands)
    if flagged != len(compile_commands):
        raise RuntimeError(f"flagged {flagged}/{len(compile_commands)} compile commands")
    cache = (builddir / "CMakeCache.txt").read_text(encoding="utf-8")
    if "CMAKE_BUILD_TYPE:STRING=Release" not in cache or "GEANT4_BUILD_MULTITHREADED:BOOL=ON" not in cache:
        raise RuntimeError("CMake cache lacks Release or MT ON")
    version = command([str(install / "bin/geant4-config"), "--version"], history=history).strip()
    if not version_matches(tag, version):
        raise RuntimeError(f"Geant4 version {version} disagrees with {tag}")
    if kind != "pristine":
        headers = list(builddir.rglob("G4FindDataDir.hh"))
        version_line = re.search(r"^\s*VERSION\s+(\S+)$", SNIPPET.read_text(encoding="utf-8"), re.M)
        if version_line is None:
            raise RuntimeError("snippet has no dataset version")
        expected = version_line.group(1)
        if not any("G4MUONICDATA" in h.read_text(encoding="utf-8") and
                   f"G4MuonicData{expected}" in h.read_text(encoding="utf-8") for h in headers):
            raise RuntimeError("G4FindDataDir header lacks the registered dataset")
    binary = harness(install, parent / "harness", kind != "pristine", history)
    info = {"tag": tag, "commit": commit, "kind": kind, "patches": used_patches,
            "removed": removed, "configure_argv": configure_argv,
            "commands": history, "cache_sha256": digest(builddir / "CMakeCache.txt"),
            "compile_commands_sha256": digest(builddir / "compile_commands.json"),
            "compile_count": len(compile_commands), "flag_count": flagged, "version": version,
            "harness_sha256": digest(binary), "binary": str(binary), "install": str(install),
            "wall_s": sum(item["wall_s"] for item in history)}
    save(parent / "build.json", info)
    if slot is not None:
        record_build_dir(work, tag, kind, parent)
    print(f"BUILD {tag} {kind} rc=0 wall_s={info['wall_s']:.1f} flags={flagged}/{len(compile_commands)} "
          f"Release MT=ON version={version}")


def run(tag: str, mode: str, work: Path, route_only: str | None) -> None:
    spec = MATRIX["modes"][mode]
    kind = spec["kind"]
    parent = work / tag
    build_file = (parent / "preserved/preserved.json" if kind == "preserved"
                  else build_dir(work, tag, kind) / "build.json")
    info = json.loads(build_file.read_text(encoding="utf-8"))
    install = Path(info["install"])
    binary = Path(info["binary"])
    data = parent / "farm"
    profiles = spec.get("profiles", {})
    threads = [1] if mode == "preserved" else MATRIX["threads"]
    for route in MATRIX["routes"]:
        if route_only is not None and route != route_only:
            continue
        for target in MATRIX["targets"]:
            for count in threads:
                target_dir = parent / RUNS_DIR / mode / route / f"{target['Z']}-{target['A']}" / str(count)
                marker = target_dir / "complete.json"
                if marker.exists():
                    print(f"SKIP {tag} {mode} {route} {target['Z']},{target['A']} t={count}")
                    continue
                target_dir.mkdir(parents=True, exist_ok=True)
                attempts = sorted(target_dir.glob("attempt_*"))
                output_dir = target_dir / f"attempt_{len(attempts) + 1}"
                fresh(output_dir)
                argv = cell_argv(binary, route, target, MATRIX["events"], count, spec["opt_in"], output_dir)
                start = time.monotonic()
                command(argv, env=env_for(install, data, work, profiles), stdout=output_dir / "stdout.log")
                elapsed = time.monotonic() - start
                files = sorted(output_dir.glob("records-*.txt"))
                if not files:
                    raise RuntimeError(f"no records in {output_dir}")
                save(marker, {"argv": argv, "rc": 0, "wall_s": elapsed,
                              "output": output_dir.name,
                              "record_files": [{"name": p.name, "sha256": digest(p)} for p in files]})
                print(f"CELL {tag} {mode} {route} {target['Z']},{target['A']} t={count} "
                      f"rc=0 marker={marker} wall_s={elapsed:.1f}", flush=True)


def config_map(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            if parts[0] == "RES":
                continue
            if parts[0] in result:
                raise RuntimeError(f"duplicate config key {parts[0]} in {path}")
            result[parts[0]] = parts[1:]
    return result


def corrupt_levels(tag: str, work: Path) -> Path:
    root = work / tag
    corrupted = root / "farm_corrupt"
    fresh(corrupted)
    nodata = root / "farm_nodata"
    for item in nodata.iterdir():
        (corrupted / item.name).symlink_to(item.resolve(), target_is_directory=True)
    source = dataset_dir(work / "dataset")
    shutil.copytree(source, corrupted / source.name)
    config = config_map(cell_output(root / RUNS_DIR / "pristine/bound_decay/82-208/1") / "config.txt")
    pristine_l8 = float.fromhex(config["L"][8])
    e8 = pristine_l8 * 1000.0 / 2.0
    path = corrupted / source.name / "d3_levels.mudirac130.g4dat"
    old = path.read_text(encoding="ascii")
    columns_line = next(line for line in old.splitlines() if line.startswith("#COLUMNS"))
    columns = columns_line.split()[1:]
    index = columns.index("e8")
    seen: set[int] = set()
    lines: list[str] = []
    for line in old.splitlines(keepends=True):
        tokens = line.split()
        if len(tokens) > index and tokens[0] == "82" and tokens[1] in ("0", "208"):
            mass = int(tokens[1])
            if not (0 < e8 < float(tokens[columns.index("e7")]) and
                    e8 * 0.001 < pristine_l8):
                raise RuntimeError(f"P12 precondition fails for {mass}")
            tokens[index] = repr(e8)
            line = " ".join(tokens) + "\n"
            seen.add(mass)
        lines.append(line)
    if seen != {0, 208}:
        raise RuntimeError(f"P12 changed wrong rows: {seen}")
    path.write_text("".join(lines), encoding="ascii")
    print(f"P12_PRECONDITION {tag} e8={e8!r} 0<e8<e7 and e8<pristine_L8 "
          f"rows={sorted(seen)}", flush=True)
    return corrupted


def cases(work: Path) -> None:
    outcomes: list[dict[str, Any]] = []
    route = "bound_decay"
    target = next(t for t in MATRIX["targets"] if t["Z"] == 47)
    definitions = [
        ("P1", {}, "parity", "compiled", "set", None),
        ("P2", {"G4MUONICDATA_PROFILE": "mudirac130"}, "compiled", "mudirac130", "set", None),
        ("P3", {"G4MUONICDATA_D1_PROFILE": "mizuno2025"}, "mizuno2025", "compiled", "set", None),
        ("P4", {"G4MUONICDATA_D3_PROFILE": "mudirac130"}, "parity", "mudirac130", "set", None),
        ("P5", {"G4MUONICDATA_D1_PROFILE": "mizuno2025", "G4MUONICDATA_D3_PROFILE": "mudirac130"},
         "mizuno2025", "mudirac130", "set", None),
        ("P6", {"G4MUONICDATA_PROFILE": "compiled", "G4MUONICDATA_D1_PROFILE": "mizuno2025"},
         "mizuno2025", "compiled", "set", None),
        ("P7", {"G4MUONICDATA_PROFILE": "compiled", "G4MUONICDATA_D3_PROFILE": "mudirac130"},
         "compiled", "mudirac130", "set", None),
        ("P8", {"G4MUONICDATA_PROFILE": "compiled"}, "compiled", "compiled", "none", "nodata"),
        ("P9", {"G4MUONICDATA_D3_PROFILE": "mudirac130"}, "parity", "mudirac130", "set", "explicit"),
        ("P10", {"G4MUONICDATA_D3_PROFILE": "mudirac130"}, "", "", "", "nodata"),
        ("P11", {"G4MUONICDATA_PROFILE": "unknown_profile"}, "", "", "", None),
    ]
    for tag in MATRIX["revisions"]:
        root = work / tag
        build_info = json.loads((build_dir(work, tag, "patched") / "build.json").read_text(encoding="utf-8"))
        install = Path(build_info["install"])
        binary = Path(build_info["binary"])
        farm_path = root / "farm"
        explicit_dir = root / "explicit_dataset"
        fresh(explicit_dir)
        source = dataset_dir(work / "dataset")
        shutil.copytree(source, explicit_dir / source.name)
        corrupt = corrupt_levels(tag, work)
        for case_id, profiles, d1, d3, directory, special in definitions:
            cell = root / "cases" / case_id
            fresh(cell)
            chosen_data = root / "farm_nodata" if special in ("nodata", "explicit") else farm_path
            variables = dict(profiles)
            if special == "explicit":
                variables["G4MUONICDATA"] = str(explicit_dir / source.name)
            env = env_for(install, chosen_data, work, variables)
            argv = cell_argv(binary, route, target, 1, 1, "on", cell, True)
            result = subprocess.run(argv, env=env, capture_output=True, text=True, check=False)
            output = result.stdout + result.stderr
            (cell / "stdout.log").write_text(output, encoding="utf-8")
            if case_id in ("P10", "P11"):
                code = "G4MuonicData001" if case_id == "P10" else "G4MuonicData004"
                ok = result.returncode != 0 and code in output
            else:
                if result.returncode == 0:
                    config = config_map(cell / "config.txt")
                    version = "none" if case_id == "P8" else json.loads(
                        (work / "dataset/stage.json").read_text(encoding="utf-8"))["version"]
                    ok = config.get("CONFIG") == [d1, d3, directory, version]
                    if case_id == "P8":
                        ok &= config.get("DIRECTORY") == ["none"]
                    elif case_id == "P9":
                        ok &= config.get("DIRECTORY") == [str(explicit_dir / source.name)]
                    else:
                        ok &= config.get("DIRECTORY") == [str(farm_path / source.name)]
                else:
                    ok = False
            outcomes.append({"tag": tag, "case": case_id, "pass": bool(ok), "rc": result.returncode})
            print(f"CASE {tag} {case_id} {'PASS' if ok else 'FAIL'} rc={result.returncode}", flush=True)
            if not ok:
                raise RuntimeError(f"{tag} {case_id} failed: {output[-1200:]}")
        for case_id, build_kind in (("P12", "patched"), ("P13", "mutant")):
            if case_id == "P13" and tag != "v11.4.2":
                continue
            spec = json.loads((build_dir(work, tag, build_kind) / "build.json").read_text(encoding="utf-8"))
            cell = root / "cases" / case_id
            fresh(cell)
            args = cell_argv(Path(spec["binary"]), route, {"Z": 82, "A": 208}, 1, 1, "on", cell)
            env = env_for(Path(spec["install"]), corrupt, work,
                          {"G4MUONICDATA_D3_PROFILE": "mudirac130"})
            result = subprocess.run(args, env=env, capture_output=True, text=True, check=False)
            output = result.stdout + result.stderr
            (cell / "stdout.log").write_text(output, encoding="utf-8")
            records = "".join(p.read_text(encoding="utf-8") for p in cell.glob("records-*.txt"))
            stop_lines = [line for line in records.splitlines() if line.startswith("C 0 1 ")]
            refusal = "the muonic cascade assembled levels that do not fall"
            if case_id == "P12":
                ok = result.returncode != 0 and refusal in output and "S006" not in output and not stop_lines
            else:
                ok = (result.returncode == 0 and bool(stop_lines) and refusal not in output
                      and "S006" not in output)
            outcomes.append({"tag": tag, "case": case_id, "pass": bool(ok), "rc": result.returncode,
                             "stop_c_lines": len(stop_lines)})
            print(f"CASE {tag} {case_id} {'PASS' if ok else 'FAIL'} rc={result.returncode} "
                  f"stop_c_lines={len(stop_lines)}", flush=True)
            if not ok:
                raise RuntimeError(f"{tag} {case_id} failed: {output[-1200:]}")
    save(work / "cases.json", outcomes)


def parsed_records(directory: Path) -> list[list[str]]:
    files = sorted(directory.glob("records-*.txt"))
    if not files:
        raise RuntimeError(f"no records in {directory}")
    records = [line.split() for file in files for line in file.read_text(encoding="utf-8").splitlines()]
    sizes = {"T": 9, "S": 8, "C": 8, "E": 4}
    for record in records:
        if not record or len(record) != sizes.get(record[0]):
            raise RuntimeError(f"malformed record in {directory}: {record}")
    return records


def record_key(record: list[str]) -> tuple[int, int, int, int, str]:
    kind = {"T": 0, "S": 1, "C": 2, "E": 3}[record[0]]
    track = int(record[2]) if record[0] != "E" else 0
    step = int(record[3]) if record[0] == "S" else 0
    k = int(record[3]) if record[0] == "C" else 0
    return int(record[1]), kind, track, step * 1000000 + k, " ".join(record)


def canonical(records: list[list[str]]) -> bytes:
    return ("\n".join(" ".join(row) for row in sorted(records, key=record_key)) + "\n").encode()


def records_digest(records: list[list[str]]) -> str:
    return hashlib.sha256(canonical(records)).hexdigest()


def event_check(records: list[list[str]], expected: int, base: int) -> tuple[bool, str]:
    ends = [r for r in records if r[0] == "E"]
    ids = [int(r[1]) for r in ends]
    if sorted(ids) != list(range(expected)):
        return False, f"E ids differ: count={len(ids)} first={sorted(ids)[:8]}"
    tracks: dict[int, int] = {}
    for row in records:
        if row[0] == "T":
            event = int(row[1])
            tracks[event] = tracks.get(event, 0) + 1
    for row in ends:
        event = int(row[1])
        if int(row[2]) != base + event or int(row[3]) != tracks.get(event, 0):
            return False, f"E seed/track count differs at event {event}: {row}"
    return True, f"events={expected}"


def route_check(config: dict[str, list[str]], records: list[list[str]], route: str,
                events: int) -> tuple[bool, str]:
    names = config.get("PROCESSES", [])
    bound = names.count("muMinusCaptureAtRest")
    helper = names.count("muMinusAtomicCaptureAtRest")
    expected = (1, 0) if route == "bound_decay" else (0, 1)
    if (bound, helper) != expected:
        return False, f"rest process counts {(bound, helper)} expected {expected}"
    if route == "muonic_atom_helper" and len(config.get("MUATOM_PRECREATED", [])) != 1:
        return False, f"master muonic atom marker {config.get('MUATOM_PRECREATED')}"
    by_event: dict[int, int] = {}
    for row in records:
        if row[0] == "T" and row[5] == "muMinusAtomicCaptureAtRest":
            event = int(row[1])
            by_event[event] = by_event.get(event, 0) + 1
    if route == "bound_decay" and by_event:
        return False, f"bound route created atomic tracks: {by_event}"
    if route == "muonic_atom_helper" and any(by_event.get(e, 0) != 1 for e in range(events)):
        return False, f"helper atomic-track counts differ: {list(by_event.items())[:8]}"
    return True, f"rest={expected} atomic_tracks={sum(by_event.values())}"


def target_check(config: dict[str, list[str]], target: dict[str, int]) -> tuple[bool, str]:
    material = config.get("MATERIAL", [])
    expected = ["1", "1", str(target["Z"]), str(target["A"]), float(1).hex()]
    return material == expected, f"material={material} expected={expected}"


def config_check(config: dict[str, list[str]], mode: str, version: str) -> tuple[bool, str]:
    expected = {
        "pristine": None,
        "preserved": None,
        "patched-off": ["compiled", "compiled", "none", "none"],
        "patched-default": ["parity", "compiled", "set", version],
        "enabled": ["mizuno2025", "mudirac130", "set", version],
    }[mode]
    actual = config.get("CONFIG")
    return actual == expected, f"config={actual} expected={expected}"


def table(path: Path) -> tuple[list[str], dict[tuple[int, ...], list[float]]]:
    columns: list[str] = []
    rows: dict[tuple[int, ...], list[float]] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("#COLUMNS"):
            columns = line.split()[1:]
        elif line and not line.startswith("#"):
            parts = line.split()
            keys = 2 if "A" in columns else 1
            rows[tuple(int(value) for value in parts[:keys])] = [float(value) for value in parts[keys:]]
    if not columns or not rows:
        raise RuntimeError(f"table empty or lacks columns: {path}")
    return columns, rows


def table_value(path: Path, key: tuple[int, ...], column: str) -> float | None:
    columns, rows = table(path)
    values = rows.get(key)
    if values is None:
        return None
    nonkeys = [name for name in columns if name not in ("Z", "A")]
    return values[nonkeys.index(column)]


def table_levels(path: Path, key: tuple[int, int]) -> list[float] | None:
    columns, rows = table(path)
    values = rows.get(key)
    if values is None:
        return None
    if columns[:2] != ["Z", "A"] or not columns[2:] or columns[2] != "e2":
        raise RuntimeError(f"bad level shape: {path}")
    return values


def same_float(actual: str, expected: float) -> bool:
    return float.fromhex(actual).hex() == expected.hex()


def lookup_check(config: dict[str, list[str]], pristine: dict[str, list[str]],
                 target: dict[str, int], dataset: Path) -> tuple[bool, str]:
    z, a = target["Z"], target["A"]
    requests = {
        "rate": (dataset / "d1_capture.mizuno2025.g4dat", (z, a), "value", "RATE_BD", 0.001),
        "zeff": (dataset / "d1_zeff.g4dat", (z,), "value", "ZEFF_BD", 1.0),
        "kshell": (dataset / "d3_kshell.mudirac130.g4dat", (z, a), "value", "KA", 0.001),
    }
    for name, (path, key, column, config_key, scale) in requests.items():
        hit = None if name == "zeff" else table_value(path, key, column)
        origin = "exact" if hit is not None else "compiled"
        line = next((row for row in config.get("_RES_LINES", []) if row[0] == name), None)
        if line is None or line[3] != origin:
            return False, f"{name} origin {line} expected {origin}"
        if name == "zeff":
            expected_profile = "mizuno2025"
        else:
            expected_profile = "mizuno2025" if name == "rate" else "mudirac130"
        if line[4] != expected_profile or int(line[5]) != int(hit is not None):
            return False, f"{name} resolution {line}"
        if hit is None:
            fallback_key = "K1" if name == "kshell" else config_key
            expected = float.fromhex(pristine[fallback_key][0])
        else:
            expected = hit * scale
        if config_key not in config or not same_float(config[config_key][0], expected):
            return False, f"{config_key} {config.get(config_key)} expected {expected.hex()}"
        if name in ("rate", "zeff"):
            helper_key = "RATE_HELPER" if name == "rate" else "ZEFF_HELPER"
            expected_helper = (float.fromhex(pristine[helper_key][0]) if hit is None else hit * scale)
            if helper_key not in config or not same_float(config[helper_key][0], expected_helper):
                return False, f"{helper_key} {config.get(helper_key)} expected {expected_helper.hex()}"
    return True, f"rate zeff kshell checked for {z},{a}"


def level_check(config: dict[str, list[str]], pristine: dict[str, list[str]],
                mode: str, target: dict[str, int], dataset: Path) -> tuple[bool, str]:
    actual = [float.fromhex(value) for value in config["L"]]
    stock = [float.fromhex(value) for value in pristine["L"]]
    if len(actual) != len(stock):
        return False, "level count differs from pristine"
    if mode != "enabled":
        return actual == stock, "off/default level array vs pristine"
    key = target["Z"], target["A"]
    k = table_value(dataset / "d3_kshell.mudirac130.g4dat", key, "value")
    levels = table_levels(dataset / "d3_levels.mudirac130.g4dat", key)
    if k is None or levels is None:
        return actual == stock, "compiled level array vs pristine"
    expected = stock[:]
    expected[0] = k * 0.001
    for i, level in enumerate(levels, start=1):
        expected[i] = level * 0.001
    if actual != expected:
        differing = [i for i, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b]
        return False, f"level array differs at {differing}"
    res = next((row for row in config.get("_RES_LINES", []) if row[0] == "levels"), None)
    if res is None or res[3:] != ["exact", "mudirac130", str(len(levels))]:
        return False, f"level resolution {res}"
    return True, f"levels=table:{len(levels)} formula:{len(actual) - len(levels) - 1}"


def d9_check(records: list[list[str]], config: dict[str, list[str]], route: str,
             target: dict[str, int], dataset: Path) -> tuple[bool, str]:
    levels = [float.fromhex(value) for value in config["L"]]
    tolerance = 64 * sys.float_info.epsilon * max(levels[0], 0.001)
    tabulated = table_levels(dataset / "d3_levels.mudirac130.g4dat", (target["Z"], target["A"]))
    count = len(tabulated or [])
    tab = [levels[0]] + [v * 0.001 for v in (tabulated or [])]
    by_event: dict[int, list[list[str]]] = {}
    for row in records:
        if row[0] == "C" and row[2] == "1":
            by_event.setdefault(int(row[1]), []).append(row)
    classes = {"table": 0, "mixed": 0, "formula": 0}
    for event, all_secondaries in by_event.items():
        if route == "bound_decay":
            cascade = [r for r in all_secondaries if r[5].endswith("_EMCascade")]
        else:
            cascade = []
            for row in all_secondaries:
                if row[4] not in ("11", "22"):
                    break
                cascade.append(row)
        if not cascade or cascade[0][4] != "11" or abs(float.fromhex(cascade[0][6]) - levels[13]) > tolerance:
            return False, f"event {event} lacks initial cascade electron"
        n = 13
        for row in cascade[1:]:
            energy = float.fromhex(row[6])
            if row[4] == "11":
                i = n - 1
                if i < 0 or abs(energy - (levels[i] - levels[n])) > tolerance:
                    return False, f"event {event} electron at level {n} carries {energy.hex()}"
            elif row[4] == "22":
                matches = [j for j in range(n) if abs(energy - (levels[j] - levels[n])) <= tolerance]
                if len(matches) != 1:
                    return False, f"event {event} gamma at level {n} matches {matches}"
                i = matches[0]
            else:
                return False, f"event {event} cascade particle {row[4]}"
            if n <= count:
                category = "table"
                if abs(energy - (tab[i] - tab[n])) > tolerance:
                    return False, f"event {event} table transition {n}->{i} differs"
            elif i <= count:
                category = "mixed"
            else:
                category = "formula"
            classes[category] += 1
            n = i
        if n != 0:
            return False, f"event {event} cascade ended at level {n}"
    if not by_event or classes["table"] == 0:
        return False, f"no table-class transition: {classes}"
    return True, f"transitions table={classes['table']} mixed={classes['mixed']} formula={classes['formula']}"


CHECK_IDS = ("events", "route", "target", "particles", "config", "lookups", "levels", "d9",
             "parity", "threads")


def cell_dir(work: Path, tag: str, mode: str, route: str, target: dict[str, int], threads: int) -> Path:
    return work / tag / RUNS_DIR / mode / route / f"{target['Z']}-{target['A']}" / str(threads)


def cell_output(directory: Path) -> Path:
    marker = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    return directory / marker["output"]


def check(work: Path, out: Path) -> None:
    dataset = dataset_dir(work / "dataset")
    version = json.loads((work / "dataset/stage.json").read_text(encoding="utf-8"))["version"]
    rows: list[dict[str, str]] = []
    for tag in MATRIX["revisions"]:
        for route in MATRIX["routes"]:
            for target in MATRIX["targets"]:
                for mode in MATRIX["modes"]:
                    threads = [1] if mode == "preserved" else MATRIX["threads"]
                    for thread in threads:
                        directory = cell_dir(work, tag, mode, route, target, thread)
                        if not (directory / "complete.json").is_file():
                            raise RuntimeError(f"missing completion marker: {directory}")
                        directory = cell_output(directory)
                        records = parsed_records(directory)
                        config = config_map(directory / "config.txt")
                        config_text = (directory / "config.txt").read_text(encoding="utf-8")
                        config["_RES_LINES"] = [line.split()[1:] for line in config_text.splitlines()
                                                if line.startswith("RES ")]
                        pristine_thread = thread if mode != "preserved" else 1
                        pristine_dir = cell_dir(work, tag, "pristine", route, target, pristine_thread)
                        pristine_dir = cell_output(pristine_dir)
                        pristine = config_map(pristine_dir / "config.txt")
                        results: dict[str, tuple[bool, str]] = {}
                        results["events"] = event_check(records, MATRIX["events"], MATRIX["seed_base"])
                        results["route"] = route_check(config, records, route, MATRIX["events"])
                        results["target"] = target_check(config, target)
                        results["particles"] = (config.get("PARTICLES") == ["ok"], "particle table")
                        results["config"] = config_check(config, mode, version)
                        results["lookups"] = (lookup_check(config, pristine, target, dataset)
                                              if mode == "enabled" else (True, "not enabled"))
                        results["levels"] = (level_check(config, pristine, mode, target, dataset)
                                             if mode not in ("pristine", "preserved")
                                             else (True, "reference"))
                        results["d9"] = (d9_check(records, config, route, target, dataset)
                                         if mode == "enabled" else (True, "not enabled"))
                        comparison_modes = ("patched-off", "patched-default", "preserved")
                        reference = (records_digest(parsed_records(pristine_dir))
                                     if mode in comparison_modes else records_digest(records))
                        same = records_digest(records) == reference
                        results["parity"] = (same, f"digest={records_digest(records)} reference={reference}")
                        if thread == 4:
                            single = cell_dir(work, tag, mode, route, target, 1)
                            single = cell_output(single)
                            thread_same = records_digest(records) == records_digest(parsed_records(single))
                            results["threads"] = (thread_same, "sorted worker records vs one thread")
                        else:
                            results["threads"] = (True, "one-thread reference")
                        for check_id in CHECK_IDS:
                            passed, detail = results[check_id]
                            status = "INFO" if mode == "preserved" and check_id == "parity" else (
                                "PASS" if passed else "FAIL")
                            row = {"tag": tag, "mode": mode, "route": route,
                                   "Z": str(target["Z"]), "A": str(target["A"]), "threads": str(thread),
                                   "check": check_id, "status": status, "detail": detail}
                            rows.append(row)
                            print(f"CHECK {tag} {mode} {route} {target['Z']},{target['A']} t={thread} "
                                  f"{check_id} {status} {detail}", flush=True)
                            if status == "FAIL":
                                raise RuntimeError(f"first gating failure: {row}")
    for result in json.loads((work / "cases.json").read_text(encoding="utf-8")):
        row = {"tag": result["tag"], "mode": "case", "route": "bound_decay", "Z": "",
               "A": "", "threads": "", "check": result["case"],
               "status": "PASS" if result["pass"] else "FAIL", "detail": f"rc={result['rc']}"}
        rows.append(row)
        print(f"CHECK {result['tag']} case {result['case']} {row['status']}", flush=True)
        if row["status"] == "FAIL":
            raise RuntimeError(f"case failed: {row}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("tag", "mode", "route", "Z", "A", "threads",
                                                  "check", "status", "detail"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CHECK COMPLETE rows={len(rows)} gating_failures=0")


def normalized(path: str, work: Path, preserved_paths: dict[str, Path]) -> str:
    if path.startswith(str(work)):
        return "$WORK" + path[len(str(work)):]
    for tag, install in preserved_paths.items():
        if path.startswith(str(install)):
            return f"$PRESERVED_{tag}" + path[len(str(install)):]
    return path


def libraries(binary: Path, work: Path, preserved_paths: dict[str, Path]) -> list[dict[str, str | None]]:
    output = command(["ldd", str(binary)])
    found: list[dict[str, str | None]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=>" in line:
            soname, remainder = line.split("=>", 1)
            path = remainder.strip().split()[0]
        else:
            parts = line.split()
            soname, path = parts[0], parts[0]
        if path == "not":
            raise RuntimeError(f"unresolved ldd dependency: {line}")
        if Path(path).is_file():
            found.append({"soname": soname.strip(), "path": normalized(path, work, preserved_paths),
                          "sha256": digest(Path(path))})
        else:
            found.append({"soname": soname.strip(), "path": path, "sha256": None})
    return found


def manifest(work: Path, out: Path) -> None:
    stage = json.loads((work / "dataset/stage.json").read_text(encoding="utf-8"))
    dataset = dataset_dir(work / "dataset")
    members = {p.relative_to(dataset.parent).as_posix(): digest(p)
               for p in sorted(dataset.rglob("*")) if p.is_file()}
    if members != stage["members"]:
        raise RuntimeError("staged dataset members changed")
    preserved_paths = {tag: Path(json.loads((work / tag / "preserved/preserved.json").read_text(
        encoding="utf-8"))["install"]) for tag in MATRIX["revisions"]}
    builds: list[dict[str, Any]] = []
    for tag in MATRIX["revisions"]:
        for kind in (("pristine", "patched", "mutant") if tag == "v11.4.2" else ("pristine", "patched")):
            info = json.loads((build_dir(work, tag, kind) / "build.json").read_text(encoding="utf-8"))
            binary = Path(info["binary"])
            builds.append({"tag": tag, "commit": info["commit"], "kind": kind,
                           "patches": info["patches"], "removed": info["removed"],
                           "configure_argv": [normalized(arg, work, preserved_paths)
                                              for arg in info["configure_argv"]],
                           "cache_sha256": info["cache_sha256"],
                           "compile_commands_sha256": info["compile_commands_sha256"],
                           "compile_count": info["compile_count"], "flag_count": info["flag_count"],
                           "version": info["version"], "wall_s": info["wall_s"],
                           "harness_sha256": digest(binary),
                           "harness_path": normalized(str(binary), work, preserved_paths),
                           "ldd": libraries(binary, work, preserved_paths)})
    farms = {}
    for tag in MATRIX["revisions"]:
        info = json.loads((work / tag / "farm.json").read_text(encoding="utf-8"))
        farms[tag] = [{"name": entry["name"],
                       "target": (f"$WORK/{tag}/farm/{entry['name']}"
                                  if entry["name"].startswith(stage["name"])
                                  else normalized(entry["target"], work, preserved_paths)),
                       "sha256": entry["sha256"]} for entry in info["entries"]]
    host = {"arch": platform.machine(), "gcc": command(["gcc", "--version"]).splitlines()[0],
            "cmake": command(["cmake", "--version"]).splitlines()[0],
            "glibc": " ".join(platform.libc_ver()), "python": platform.python_version()}
    value = {"builds": builds, "farms": farms,
             "dataset": {"name": stage["name"], "version": stage["version"],
                         "archive": stage["archive"], "md5": stage["md5"], "members": members},
             "sources": {name: digest(HERE / name) for name in SOURCES}, "host": host}
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    for forbidden in ("/root/", "/mnt/", "/home/", "u020", "bryan"):
        if forbidden in raw.lower():
            raise RuntimeError(f"manifest contains private path token {forbidden}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(raw, encoding="utf-8")
    print(f"MANIFEST builds={len(builds)} patches={sum(len(b['patches']) for b in builds)} "
          f"dataset_members={len(members)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    stage = sub.add_parser("stage-dataset")
    stage.add_argument("--out", type=Path, required=True)
    farm_parser = sub.add_parser("farm")
    farm_parser.add_argument("--tag", required=True)
    farm_parser.add_argument("--physics-data", type=Path, required=True)
    farm_parser.add_argument("--dataset", type=Path, required=True)
    farm_parser.add_argument("--work", type=Path, required=True)
    preserved_parser = sub.add_parser("preserved")
    preserved_parser.add_argument("--tag", required=True)
    preserved_parser.add_argument("--install", type=Path, required=True)
    preserved_parser.add_argument("--work", type=Path, required=True)
    probe = sub.add_parser("throughput")
    probe.add_argument("--tag", required=True)
    probe.add_argument("--work", type=Path, required=True)
    probe.add_argument("--events", type=int, required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--tag", required=True)
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--kind", choices=("pristine", "patched", "mutant"), required=True)
    build_parser.add_argument("--work", type=Path, required=True)
    build_parser.add_argument("--slot")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--tag", required=True)
    run_parser.add_argument("--mode", choices=MATRIX["modes"], required=True)
    run_parser.add_argument("--route", choices=MATRIX["routes"])
    run_parser.add_argument("--work", type=Path, required=True)
    for name in ("cases", "manifest", "check"):
        p = sub.add_parser(name)
        p.add_argument("--work", type=Path, required=True)
        if name != "cases":
            p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "stage-dataset":
        stage_dataset(args.out)
    elif args.action == "farm":
        farm(args.tag, args.physics_data, args.dataset, args.work)
    elif args.action == "preserved":
        preserved(args.tag, args.install, args.work)
    elif args.action == "throughput":
        throughput(args.tag, args.work, args.events)
    elif args.action == "build":
        build(args.tag, args.source, args.kind, args.work, args.slot)
    elif args.action == "run":
        run(args.tag, args.mode, args.work, args.route)
    elif args.action == "cases":
        cases(args.work)
    elif args.action == "check":
        check(args.work, args.out)
    elif args.action == "manifest":
        manifest(args.work, args.out)
    else:
        raise RuntimeError(f"{args.action} is not implemented")


if __name__ == "__main__":
    main()
