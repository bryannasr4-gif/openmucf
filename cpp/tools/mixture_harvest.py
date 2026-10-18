"""Build ``harvest_mixture.cc`` against a Geant4 installation, run it on one cell, and summarize raw cells.

  build      --install <dir> --kind pristine|patched --drill none|extra_draw|early_return --out <dir>
  run        --build <dir> --route stopping|atomic --symbol <El> --z <Z> --events <n> --seed-base <s>
             --data <GEANT4_DATA_DIR> --profile parity|<name>|pristine --hooks on|off --identity on|off
             --out <dir> [--drill-a <A>]
  summarize  --root <repository> --cells <cells.json> --out <dir>

``build`` compiles the committed source with the flags it records in its own ``build.json`` (with the
compiler, the Geant4 version, the source and binary SHA-256 and the SHA-256 of every library the binary
loads); a summary reads each cell's flags from that record, never from a constant. ``run`` records the argv,
the environment it sets, the return code, the wall time and the SHA-256 of what the binary wrote.
``summarize`` turns raw cells into the committed CSV files through :mod:`openmucf.g4.d1_mixture`, imported
from the repository without running the parent package's ``__init__`` (which needs packages a Geant4 host may
lack). Standard library only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "harvest_mixture.cc"
#: The flags every build compiles the source with; the build record keeps what was used.
FLAGS = ("-std=c++17", "-O2", "-ffp-contract=off")
DRILL_DEFINES = {
    "none": [],
    "extra_draw": ["-DHARVEST_MIXTURE_DRILL_EXTRA_DRAW"],
    "early_return": ["-DHARVEST_MIXTURE_DRILL_EARLY_RETURN"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(argv: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr[-2000:]}")
    return result.stdout


def libraries(binary: Path, install: Path) -> list[dict[str, str]]:
    """The shared libraries the binary loads, each with its SHA-256."""
    env = {**os.environ, "LD_LIBRARY_PATH": str(install / "lib")}
    out = []
    for line in output(["ldd", str(binary)], env).splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(\S+)\s+\(", line)
        if match:
            out.append({"soname": match[1], "path": match[2], "sha256": sha256(Path(match[2]))})
    return out


def build(install: Path, kind: str, drill: str, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"non-empty build directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = install / "bin" / "geant4-config"
    cflags = output([str(config), "--cflags"]).split()
    libs = output([str(config), "--libs"]).split()
    defines = (["-DHARVEST_MIXTURE_PATCHED"] if kind == "patched" else []) + DRILL_DEFINES[drill]
    binary = out / "harvest_mixture"
    argv = [
        "g++",
        *FLAGS,
        *defines,
        str(SOURCE),
        "-o",
        str(binary),
        *cflags,
        *libs,
        f"-Wl,-rpath,{install / 'lib'}",
    ]
    start = time.monotonic()
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    (out / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"compile failed: {result.stderr[-3000:]}")
    record = {
        "kind": kind,
        "drill": drill,
        "install": str(install),
        "geant4_version": output([str(config), "--version"]).strip(),
        "gcc": output(["g++", "--version"]).splitlines()[0],
        "compile_flags": " ".join([*FLAGS, *defines]),
        "argv": argv,
        "source_sha256": sha256(SOURCE),
        "binary_sha256": sha256(binary),
        "libraries": libraries(binary, install),
        "wall_s": time.monotonic() - start,
    }
    (out / "build.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"BUILD {out} kind={kind} drill={drill} binary_sha256={record['binary_sha256']} "
        f"source_sha256={record['source_sha256']} flags={record['compile_flags']!r}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    info = json.loads((args.build / "build.json").read_text(encoding="utf-8"))
    out: Path = args.out
    if (out / "run.json").exists():
        print(f"SKIP {out} (complete)", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(out),
        "GEANT4_DATA_DIR": str(args.data),
        "LD_LIBRARY_PATH": str(Path(info["install"]) / "lib"),
    }
    opt_in = "off"
    if args.profile != "pristine":
        if info["kind"] != "patched":
            raise RuntimeError("a profile needs a patched build")
        opt_in = "on"
        if args.profile != "parity":
            env["G4MUONICDATA_D1_PROFILE"] = args.profile
    argv = [
        str(args.build / "harvest_mixture"),
        "--route",
        args.route,
        "--symbol",
        args.symbol,
        "--Z",
        str(args.z),
        "--events",
        str(args.events),
        "--seed-base",
        str(args.seed_base),
        "--opt-in",
        opt_in,
        "--hooks",
        args.hooks,
        "--identity",
        args.identity,
        "--out",
        str(out / "raw"),
    ]
    if args.drill_a:
        argv += ["--drill-a", str(args.drill_a)]
    start = time.monotonic()
    with (out / "stdout.log").open("wb") as handle:
        rc = subprocess.call(argv, env=env, stdout=handle, stderr=subprocess.STDOUT)
    wall = time.monotonic() - start
    raw = out / "raw"
    files = {}
    for name in ("stops.txt", "records.txt", "config.txt", "summary.txt"):
        path = raw / name
        if path.is_file():
            files[name] = sha256(path)
    record = {
        "argv": argv,
        "env": {k: v for k, v in env.items() if k != "PATH"},
        "rc": rc,
        "wall_s": wall,
        "build_binary_sha256": info["binary_sha256"],
        "files": files,
    }
    if rc != 0:
        (out / "failed.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"cell {out} failed rc={rc}; see stdout.log")
    for name in ("stops.txt", "records.txt"):
        path = raw / name
        if path.is_file():
            with (
                path.open("rb") as src,
                gzip.GzipFile(str(path) + ".gz", "wb", compresslevel=6, mtime=0) as dst,
            ):
                dst.write(src.read())
            path.unlink()
    (out / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CELL {out} rc=0 wall_s={wall:.1f} events={args.events}", flush=True)


def repository_module(root: Path, name: str) -> types.ModuleType:
    """``openmucf.g4.<name>`` from ``root`` with the parent package bound to its directory alone, so that its
    ``__init__`` (which imports packages a Geant4 host may lack) does not run; ``openmucf.g4`` and its modules
    are standard-library only."""
    if "openmucf" not in sys.modules:
        parent = types.ModuleType("openmucf")
        parent.__path__ = [str(root / "openmucf")]
        sys.modules["openmucf"] = parent
    return importlib.import_module(f"openmucf.g4.{name}")


def summarize(root: Path, cells: Path, out: Path) -> None:
    mixture: Any = repository_module(root, "d1_mixture")
    mixture.summarize_cells(root, json.loads(cells.read_text(encoding="utf-8")), out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--install", type=Path, required=True)
    b.add_argument("--kind", choices=("pristine", "patched"), required=True)
    b.add_argument("--drill", choices=tuple(DRILL_DEFINES), default="none")
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--build", type=Path, required=True)
    r.add_argument("--route", choices=("stopping", "atomic"), required=True)
    r.add_argument("--symbol", required=True)
    r.add_argument("--z", type=int, required=True)
    r.add_argument("--events", type=int, required=True)
    r.add_argument("--seed-base", type=int, required=True)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--profile", required=True)
    r.add_argument("--hooks", choices=("on", "off"), default="on")
    r.add_argument("--identity", choices=("on", "off"), default="off")
    r.add_argument("--drill-a", type=int, default=0)
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--root", type=Path, default=ROOT)
    s.add_argument("--cells", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.install, args.kind, args.drill, args.out)
    elif args.command == "run":
        run(args)
    else:
        summarize(args.root, args.cells, args.out)


if __name__ == "__main__":
    main()
