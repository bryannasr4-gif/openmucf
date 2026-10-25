"""Build ``harvest_mixture.cc`` against a Geant4 installation, run it on one cell, and summarize raw cells.

  build      --install <dir> --kind pristine|patched|bound --drill none|extra_draw|early_return
             [--trace on|off] --out <dir>
  run        --build <dir> --route stopping|atomic --symbol <El> --z <Z> --events <n> --seed-base <s>
             --data <GEANT4_DATA_DIR> --profile parity|<name>|pristine --hooks on|off --identity on|off
             --out <dir> [--drill-a <A>] [--isotope <A>] [--t0 <ns>] [--event-offset <i0>]
             [--bound-rates on|off] [--bound-record on|off] [--rng-states on|off]
             [--synthetic-capture-scale <s>]
  summarize  --root <repository> --cells <cells.json> --out <dir>
  summarize-bound  --root <repository> --plan <plan.json> --out <dir> [--jobs <n>]

``build`` compiles the committed source with the flags it records in its own ``build.json`` (with the
compiler, the Geant4 version, the source and binary SHA-256 and the SHA-256 of every library the binary
loads); a summary reads each cell's flags from that record, never from a constant. ``run`` records the argv,
the environment it sets, the return code, the wall time and the SHA-256 of what the binary wrote.
``summarize-bound`` turns the chunks of a bound-rate harvest into ``bound_rates_harvest.csv`` and
``bound_rates_builds.csv`` through :mod:`openmucf.g4.d1_boundrates`, summarizing chunks in ``--jobs``
processes first. ``summarize`` turns raw cells into the committed CSV files through
:mod:`openmucf.g4.d1_mixture`, imported from the repository without running the parent package's ``__init__``
(which needs packages a Geant4 host may lack). Standard library only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import re
import shutil
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
#: The defines of each build kind: an unpatched install, an install carrying the muonic-data patch, and an
#: install carrying the patch with the bound-rate provider.
KIND_DEFINES = {
    "pristine": [],
    "patched": ["-DHARVEST_MIXTURE_PATCHED"],
    "bound": ["-DHARVEST_MIXTURE_PATCHED", "-DHARVEST_MIXTURE_BOUND_RATES"],
}
#: The functions a trace build defines and exports, so that the Geant4 libraries' calls bind to them.
TRACED_SYMBOLS = (
    "_ZN25G4MuonicBoundRateProvider5RatesEiiddR18G4MuonicBoundRates",
    "_ZN18G4CascadeInterface13ApplyYourselfERK15G4HadProjectileR9G4Nucleus",
    "_ZN27G4MuMinusCapturePrecompound13ApplyYourselfERK15G4HadProjectileR9G4Nucleus",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def build(install: Path, kind: str, drill: str, out: Path, trace: bool = False) -> None:
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"non-empty build directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = install / "bin" / "geant4-config"
    cflags = output([str(config), "--cflags"]).split()
    libs = output([str(config), "--libs"]).split()
    if trace and kind != "bound":
        raise RuntimeError("a trace build needs the bound kind")
    defines = KIND_DEFINES[kind] + DRILL_DEFINES[drill] + (["-DHARVEST_MIXTURE_TRACE"] if trace else [])
    exports = [f"-Wl,--export-dynamic-symbol={name}" for name in TRACED_SYMBOLS] if trace else []
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
        *exports,
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
        "trace": trace,
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
        f"BUILD {out} kind={kind} drill={drill} trace={trace} binary_sha256={record['binary_sha256']} "
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
        if info["kind"] not in ("patched", "bound"):
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
    # The extended options, passed only when away from their defaults, so a default run's argv is unchanged.
    for flag, value, default in (
        ("--isotope", args.isotope, 0),
        ("--t0", args.t0, "0"),
        ("--event-offset", args.event_offset, 0),
        ("--bound-rates", args.bound_rates, "off"),
        ("--bound-record", args.bound_record, "off"),
        ("--rng-states", args.rng_states, "off"),
        ("--synthetic-capture-scale", args.synthetic_capture_scale, "0"),
    ):
        if value != default:
            argv += [flag, str(value)]
    start = time.monotonic()
    with (out / "stdout.log").open("wb") as handle:
        rc = subprocess.call(argv, env=env, stdout=handle, stderr=subprocess.STDOUT)
    wall = time.monotonic() - start
    raw = out / "raw"
    files = {}
    for name in ("stops.txt", "records.txt", "config.txt", "summary.txt", "bound.txt", "rng.txt",
                 "extended.txt", "bound_summary.txt"):
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
    for name in ("stops.txt", "records.txt", "bound.txt", "rng.txt"):
        path = raw / name
        if path.is_file():
            with (
                path.open("rb") as src,
                gzip.GzipFile(str(path) + ".gz", "wb", compresslevel=6, mtime=0) as dst,
            ):
                shutil.copyfileobj(src, dst, 1 << 20)
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


def _bound_chunk(job: tuple[str, dict[str, Any], bool]) -> str:
    root, chunk, switch_on = job
    bound: Any = repository_module(Path(root), "d1_boundrates")
    policy = bound.load_policy(Path(root))
    group = policy["groups"][int(chunk["group"])]
    cell = Path(chunk["cell"])
    stats = bound.summarize_chunk(cell / "raw", group["route"], bound.frozen_pair(policy, group["isotope"]),
                                  float(chunk["t0_ns"]), int(chunk["first"]), int(chunk["count"]), switch_on)
    (cell / "bound_stats.json").write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="ascii")
    return f"CHUNK {cell} events={stats['events']} malformed={stats['malformed']}"


def summarize_bound(root: Path, plan_path: Path, out: Path, jobs: int) -> None:
    import multiprocessing

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    work = [(str(root), chunk, bool(plan["switch_on"])) for chunk in plan["chunks"]]
    with multiprocessing.Pool(jobs) as pool:
        for line in pool.imap_unordered(_bound_chunk, work):
            print(line, flush=True)
    bound: Any = repository_module(root, "d1_boundrates")
    bound.summarize_plan(root, {**plan, "reuse": True}, out)
    print(f"SUMMARY {out} chunks={len(work)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--install", type=Path, required=True)
    b.add_argument("--kind", choices=tuple(KIND_DEFINES), required=True)
    b.add_argument("--drill", choices=tuple(DRILL_DEFINES), default="none")
    b.add_argument("--trace", choices=("on", "off"), default="off")
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
    r.add_argument("--isotope", type=int, default=0)
    r.add_argument("--t0", default="0")
    r.add_argument("--event-offset", type=int, default=0)
    r.add_argument("--bound-rates", choices=("on", "off"), default="off")
    r.add_argument("--bound-record", choices=("on", "off"), default="off")
    r.add_argument("--rng-states", choices=("on", "off"), default="off")
    r.add_argument("--synthetic-capture-scale", default="0")
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--root", type=Path, default=ROOT)
    s.add_argument("--cells", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    sb = sub.add_parser("summarize-bound")
    sb.add_argument("--root", type=Path, default=ROOT)
    sb.add_argument("--plan", type=Path, required=True)
    sb.add_argument("--out", type=Path, required=True)
    sb.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.command == "build":
        build(args.install, args.kind, args.drill, args.out, args.trace == "on")
    elif args.command == "run":
        run(args)
    elif args.command == "summarize-bound":
        summarize_bound(args.root, args.plan, args.out, args.jobs)
    else:
        summarize(args.root, args.cells, args.out)


if __name__ == "__main__":
    main()
