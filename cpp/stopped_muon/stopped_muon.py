"""Build the stopped-muon application, run and reduce a cell, and re-run a committed cell to compare its
records.

  python3 stopped_muon.py build --install <prefix> --build <dir>
  python3 stopped_muon.py run --binary <exe> --install <prefix> --data <dir> --work <dir> --version <v>
                              --config <config> --target <target> [--prereg <file>]
  python3 stopped_muon.py reproduce --cell <version>/<config>/<target> --binary <exe> --install <prefix>
                              --data <dir> --work <dir> --out <csv> [--prereg <file>] [--cells <file>]

Standard library only; the classifier, the reducer and the pre-registration's rule come from
``openmucf/g4/stopped_mu.py``, loaded by path, so the package need not be installed. ``--prereg`` defaults to
the committed ``data/g4/stopped_mu/preregistration.json``. Every Geant4 call runs with the environment
``environment`` builds: no inherited dataset or cascade variable, the twelve dataset variables bound to
``<data>/<directory>`` as the pre-registration binds them for the revision, ``G4CASCADE_USE_PRECOMPOUND=1``
for ``bertini_pc`` only, and ``LD_LIBRARY_PATH=<install>/lib``. Before a run the directories the harness
reports must equal that binding, and after it the ``DATASET`` lines of ``config.txt`` must equal it again; a
difference refuses the chunk.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = ROOT / "openmucf" / "g4" / "stopped_mu.py"
PREREG = ROOT / "data" / "g4" / "stopped_mu" / "preregistration.json"
CELLS = ROOT / "data" / "g4" / "stopped_mu" / "cells.jsonl"
EXECUTABLE = "g4muonic_transport"

#: The run plan: chunk 0 holds PILOT_STOPS stops; the next chunk is sized from chunk 0's capture fraction to
#: reach the pre-registration's captures_per_configuration with MARGIN to spare; a cell still short takes one
#: extension chunk of EXTENSION_FACTOR times the shortfall over the cell's capture fraction. Seeds are per
#: event, so the plan changes which events are run, never what an event gives.
PILOT_STOPS = 2000
MARGIN = Fraction(3, 100)
EXTENSION_FACTOR = Fraction(11, 10)
UNSET = re.compile(
    r"^(?:G4[A-Za-z0-9_]*DATA|GEANT4_DATA_DIR|G4CASCADE_[A-Za-z0-9_]*|G4NUCMODEL_[A-Za-z0-9_]*)$"
)
REPRODUCED_COLUMNS = (
    "version",
    "config",
    "target",
    "chunk",
    "offset",
    "events",
    "compact_text_sha256",
    "binary_sha256",
)
#: The fields of a cell record that a reproduction compares with the committed record.
COMPARED = (
    "quantities",
    "neutron_hist",
    "f2",
    "events_total",
    "captures_total",
    "decays_total",
    "X",
    "N_capt",
    "events_scanned",
    "extension_chunks",
    "stops_total",
    "n_no_outcome",
    "no_outcome_events",
    "gamma",
    "line_guard",
    "tag_width",
    "problems",
)


def load_module(path: Path = MODULE) -> ModuleType:
    spec = importlib.util.spec_from_file_location("stopped_mu", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sm = load_module()


class Refusal(Exception):
    """A step the driver refuses; the message names the reason."""


# ------------------------------------------------------------------------------------------------- arguments
def target_arguments(policy: dict[str, Any], target: str) -> list[str]:
    for t in policy["setup"]["targets"]:
        if t["target"] == target:
            return list(t["arguments"])
    raise Refusal(f"unknown target {target!r}")


def route(policy: dict[str, Any], config: str) -> str:
    try:
        return sm.route_of(policy, config)
    except sm.Refused as e:
        raise Refusal(str(e)) from None


def seed_base(policy: dict[str, Any], offset: int) -> int:
    """The --seed-base of a chunk starting at global event ``offset``: event i of the cell is seeded with
    seed_base + i."""
    return int(policy["setup"]["seed_base"]) + offset


def harness_argv(
    binary: str,
    policy: dict[str, Any],
    *,
    config: str,
    target: str,
    offset: int,
    events: int,
    out: str,
    print_datasets: bool = False,
) -> list[str]:
    argv = [
        binary,
        "--route",
        route(policy, config),
        *target_arguments(policy, target),
        "--events",
        str(events),
        "--seed-base",
        str(seed_base(policy, offset)),
        "--seed-stride",
        str(policy["setup"]["seed_stride"]),
        "--out",
        out,
    ]
    options = list(policy["setup"]["harness_options"])
    if print_datasets:
        options = [o for o in options if o not in ("--momenta", "--record-datasets")] + ["--print-datasets"]
    return argv + options


def binding(policy: dict[str, Any], version: str, data: str) -> dict[str, str]:
    """The twelve dataset variables of ``version``, each bound to ``<data>/<directory>``."""
    datasets = policy["setup"]["datasets"]
    try:
        dirs = datasets["binding"][version]
    except KeyError:
        raise Refusal(f"unknown version {version!r}") from None
    if list(dirs) != list(datasets["variables"]):
        raise Refusal("the binding's variables differ from the pre-registration's list")
    return {var: f"{data.rstrip('/')}/{d}" for var, d in dirs.items()}


def environment(
    policy: dict[str, Any], base: dict[str, str], *, version: str, config: str, install: str, data: str
) -> dict[str, str]:
    route(policy, config)
    env = {k: v for k, v in base.items() if not UNSET.match(k)}
    env.update(binding(policy, version, data))
    env.update(policy["setup"]["configurations"][config]["environment"])
    env["LD_LIBRARY_PATH"] = f"{install.rstrip('/')}/lib"
    return env


def dataset_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        f = line.split()
        if f and f[0] == "DATASET":
            if len(f) != 3 or f[1] in out:
                raise Refusal(f"malformed DATASET line {line!r}")
            out[f[1]] = f[2]
    return out


def check_datasets(text: str, expected: dict[str, str], when: str) -> None:
    """Refuse unless the DATASET lines of ``text`` are exactly the expected variables, in order, bound
    alike."""
    got = dataset_lines(text)
    if list(got) != list(expected):
        raise Refusal(f"{when}: DATASET lines are not the {len(expected)} bound variables: {sorted(got)}")
    for var, path in expected.items():
        if got[var] != path:
            raise Refusal(f"{when}: {var} {got[var]} != binding {path}")


def second_chunk_events(policy: dict[str, Any], pilot_captures: int) -> int:
    """Events of the chunk after the pilot: enough stops, at the pilot's capture fraction, for the
    pre-registration's captures_per_configuration with MARGIN to spare."""
    if pilot_captures <= 0:
        raise Refusal("the pilot chunk holds no capture")
    want = Fraction(int(policy["setup"]["captures_per_configuration"])) * (1 + MARGIN)
    assert want.denominator == 1
    f = pilot_captures / PILOT_STOPS
    return max(0, math.ceil(int(want) / f) - PILOT_STOPS)


def extension_events(policy: dict[str, Any], captures: int, events: int) -> int:
    f = captures / events
    if f <= 0:
        raise Refusal("capture fraction 0")
    n_sel = int(policy["setup"]["captures_per_configuration"])
    return math.ceil((n_sel - captures) / f * float(EXTENSION_FACTOR))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="stopped_muon.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="configure and build the application against a Geant4 install")
    b.add_argument("--install", required=True)
    b.add_argument("--build", required=True)
    b.add_argument("--jobs", type=int, default=6)
    for name in ("run", "reproduce"):
        p = sub.add_parser(name)
        p.add_argument("--binary", required=True)
        p.add_argument("--install", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--work", required=True)
        p.add_argument("--prereg", default=str(PREREG))
    run = sub.choices["run"]
    run.add_argument("--version", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--target", required=True)
    rep = sub.choices["reproduce"]
    rep.add_argument("--cell", required=True, help="<version>/<config>/<target>")
    rep.add_argument("--cells", default=str(CELLS))
    rep.add_argument("--out", required=True, help="the file each re-run chunk's line is appended to")
    return ap


# ------------------------------------------------------------------------------------------------- build
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_flags(install: Path) -> tuple[str, str]:
    """Geant4_CXX_FLAGS and the multithreading flag as the install's Geant4Config.cmake sets them."""
    text = (install / "lib" / "cmake" / "Geant4" / "Geant4Config.cmake").read_text(encoding="utf-8")
    flags = re.search(r'set\(Geant4_CXX_FLAGS "([^"]*)"\)', text)
    mt = re.search(r"set\(Geant4_multithreaded_FOUND (\w+)\)", text)
    if flags is None or mt is None:
        raise Refusal("Geant4Config.cmake does not set Geant4_CXX_FLAGS and Geant4_multithreaded_FOUND")
    return flags.group(1), mt.group(1)


def first_line(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.splitlines()[0].strip()


def build(args: argparse.Namespace) -> int:
    install, out = Path(args.install), Path(args.build)
    if not (install / "lib" / "cmake" / "Geant4").is_dir():
        raise Refusal(f"no Geant4 CMake package under {install}")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(HERE),
            "-B",
            str(out),
            f"-DGeant4_DIR={install}/lib/cmake/Geant4",
            "-DG4MUONIC_TRANSPORT_PATCHED=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(out), f"-j{args.jobs}"], check=True)
    binary = out / EXECUTABLE
    flags, mt = install_flags(install)
    print(f"SOURCE g4muonic_transport.cc sha256 {sha256(HERE / 'g4muonic_transport.cc')}")
    print(f"SOURCE CMakeLists.txt sha256 {sha256(HERE / 'CMakeLists.txt')}")
    print(f"GEANT4 {first_line([str(install / 'bin' / 'geant4-config'), '--version'])}")
    print(f"COMPILER {first_line(['c++', '--version'])}")
    print(f"CMAKE {first_line(['cmake', '--version'])}")
    print(f"INSTALL Geant4_CXX_FLAGS {flags!r} multithreaded {mt}")
    print(f"BINARY {binary} sha256 {sha256(binary)}")
    return 0


# ------------------------------------------------------------------------------------------------ run a chunk
def run_chunk(
    policy: dict[str, Any],
    *,
    binary: str,
    install: str,
    data: str,
    work: Path,
    version: str,
    config: str,
    target: str,
    offset: int,
    events: int,
    chunk: int,
    extension: bool = False,
) -> dict[str, Any]:
    """Run, classify and record one chunk; the records file is deleted once classified."""
    wd = work / version / config / target / f"chunk{chunk}"
    if wd.exists():
        raise Refusal(f"chunk directory exists: {wd}")
    wd.mkdir(parents=True)
    env = environment(policy, dict(os.environ), version=version, config=config, install=install, data=data)
    expected = binding(policy, version, data)
    pre = subprocess.run(
        harness_argv(
            binary,
            policy,
            config=config,
            target=target,
            offset=offset,
            events=events,
            out=str(wd),
            print_datasets=True,
        ),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    check_datasets(pre, expected, "before the run")
    (wd / "datasets_pre.txt").write_text(pre, encoding="utf-8")
    with open(wd / "stdout.txt", "w", encoding="utf-8") as log:
        rc = subprocess.run(
            harness_argv(
                binary, policy, config=config, target=target, offset=offset, events=events, out=str(wd)
            ),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    if rc != 0:
        raise Refusal(f"harness rc {rc} (see {wd / 'stdout.txt'})")
    records = sorted(wd.glob("records-*.txt"))
    if [r.name for r in records] != ["records-0.txt"]:
        raise Refusal(f"expected one worker record file, found {[r.name for r in records]}")
    check_datasets((wd / "config.txt").read_text(encoding="utf-8"), expected, "after the run")
    with (
        open(records[0], encoding="ascii", newline="\n") as src,
        open(wd / "compact.txt", "w", encoding="ascii", newline="\n") as dst,
    ):
        stats = sm.classify_stream(
            src,
            dst,
            route=route(policy, config),
            version=version,
            config=config,
            target=target,
            offset=offset,
            events=events,
            policy=policy,
        )
    records[0].unlink()
    text = (wd / "compact.txt").read_bytes()
    text_sha, body_sha = sm.compact_digests(text)
    with open(wd / "compact.txt.gz", "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(text)
    (wd / "compact.txt").unlink()
    meta = {
        "version": version,
        "config": config,
        "target": target,
        "chunk": chunk,
        "offset": offset,
        "events": events,
        "extension": extension,
        "binary_sha256": sha256(Path(binary)),
        "compact_sha256": sha256(wd / "compact.txt.gz"),
        "compact_text_sha256": text_sha,
        "compact_body_sha256": body_sha,
        "stats": stats,
    }
    (wd / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"CHUNK {version} {config} {target} chunk{chunk} offset {offset} events {events} "
        f"C {stats['C']} D {stats['D']} X {stats['X']} text sha256 {text_sha}"
    )
    return meta


def gz_lines(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="ascii", newline="\n") as f:
        yield from f


def reduce_work(
    policy: dict[str, Any],
    work: Path,
    version: str,
    config: str,
    target: str,
    metas: list[dict[str, Any]],
    no_outcome: str = "exclude",
) -> tuple[dict[str, Any], str]:
    chunks = [
        sm.Chunk(
            chunk=f"chunk{m['chunk']}",
            offset=m["offset"],
            events=m["events"],
            extension=m["extension"],
            lines=gz_lines(work / version / config / target / f"chunk{m['chunk']}" / "compact.txt.gz"),
            compact_sha256=m["compact_sha256"],
            binary_sha256=m["binary_sha256"],
        )
        for m in metas
    ]
    return sm.reduce_cell(
        chunks, version=version, config=config, target=target, policy=policy, no_outcome=no_outcome
    )


def run(args: argparse.Namespace) -> int:
    policy = sm.load_policy(args.prereg)
    work = Path(args.work)
    common = dict(
        binary=args.binary,
        install=args.install,
        data=args.data,
        work=work,
        version=args.version,
        config=args.config,
        target=args.target,
    )
    target_arguments(policy, args.target)
    route(policy, args.config)
    metas = [run_chunk(policy, offset=0, events=PILOT_STOPS, chunk=0, **common)]
    nxt = second_chunk_events(policy, metas[0]["stats"]["C"])
    if nxt:
        metas.append(run_chunk(policy, offset=PILOT_STOPS, events=nxt, chunk=1, **common))
    cell, status = reduce_work(policy, work, args.version, args.config, args.target, metas)
    if status == "short":
        offset = sum(m["events"] for m in metas)
        ext = extension_events(policy, cell["captures_total"], cell["events_total"])
        metas.append(run_chunk(policy, offset=offset, events=ext, chunk=len(metas), extension=True, **common))
        cell, status = reduce_work(policy, work, args.version, args.config, args.target, metas)
    for m, c in zip(metas, cell["chunks"], strict=True):
        c["compact_text_sha256"], c["compact_body_sha256"] = (
            m["compact_text_sha256"],
            m["compact_body_sha256"],
        )
    out = work / args.version / args.config / args.target / "cell.json"
    out.write_text(json.dumps(cell, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"CELL {args.version} {args.config} {args.target}: {status.upper()} N_capt {cell['N_capt']} "
        f"events {cell['events_total']} chunks {len(metas)} problems {cell['problems']} -> {out}"
    )
    return 0 if status == "accepted" else 2


# ------------------------------------------------------------------------------------------------- reproduce
def committed_cell(path: str, version: str, config: str, target: str) -> dict[str, Any]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cell = json.loads(line)
        if (cell["version"], cell["config"], cell["target"]) == (version, config, target):
            return cell
    raise Refusal(f"no committed cell {version}/{config}/{target} in {path}")


def compare_chunk(committed: dict[str, Any], text_sha: str) -> str | None:
    """None if the re-run chunk's compact text equals the committed chunk's, else what differs."""
    if committed["compact_text_sha256"] != text_sha:
        return f"{committed['chunk']} compact_text_sha256 {text_sha} != {committed['compact_text_sha256']}"
    return None


def compare_cell(new: dict[str, Any], committed: dict[str, Any]) -> list[str]:
    keys = [k for k in COMPARED if k in committed or k in new]
    return sm.differing_fields(new, committed, keys)


def reproduce(args: argparse.Namespace) -> int:
    policy = sm.load_policy(args.prereg)
    try:
        version, config, target = args.cell.split("/")
    except ValueError:
        raise Refusal("--cell is <version>/<config>/<target>") from None
    committed = committed_cell(args.cells, version, config, target)
    work = Path(args.work)
    metas, differs = [], []
    lines = []
    for c in committed["chunks"]:
        k = int(c["chunk"].removeprefix("chunk"))
        m = run_chunk(
            policy,
            binary=args.binary,
            install=args.install,
            data=args.data,
            work=work,
            version=version,
            config=config,
            target=target,
            offset=c["offset"],
            events=c["events"],
            chunk=k,
            extension=c["extension"],
        )
        metas.append(m)
        lines.append(
            ",".join(
                str(x)
                for x in (
                    version,
                    config,
                    target,
                    c["chunk"],
                    c["offset"],
                    c["events"],
                    m["compact_text_sha256"],
                    m["binary_sha256"],
                )
            )
        )
        bad = compare_chunk(c, m["compact_text_sha256"])
        if bad:
            differs.append(bad)
    out = Path(args.out)
    new_file = not out.exists() or out.stat().st_size == 0
    with open(out, "a", encoding="utf-8", newline="\n") as h:
        if new_file:
            h.write(",".join(REPRODUCED_COLUMNS) + "\n")
        for line in lines:
            h.write(line + "\n")
    cell, status = reduce_work(policy, work, version, config, target, metas)
    differs += [f"field {k}" for k in compare_cell(cell, committed)]
    if "alternative" in committed:
        alt, _ = reduce_work(policy, work, version, config, target, metas, no_outcome="as_capture")
        differs += [
            f"alternative {k}"
            for k in sm.differing_fields(alt, committed["alternative"], sm.ALTERNATIVE_FIELDS)
        ]
    if status != "accepted":
        differs.append(f"status {status}")
    if differs:
        for d in differs:
            print(f"DIFFERS {args.cell} {d}")
        return 1
    print(f"REPRODUCED {args.cell}: {len(metas)} chunks, compact text and every compared field equal")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return {"build": build, "run": run, "reproduce": reproduce}[args.cmd](args)
    except (Refusal, sm.Refused, sm.Malformed) as e:
        print(f"REFUSED: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
