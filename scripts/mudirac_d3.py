"""Run MuDirac 1.3.0 over the D3 input set and commit what it prints.

    python3 scripts/mudirac_d3.py inputs --radii R --abundant A --iaea I --out data/g4/d3
    python3 scripts/mudirac_d3.py write --kind K --dir D
    python3 scripts/mudirac_d3.py run --mudirac BIN --dir D --jobs N
    python3 scripts/mudirac_d3.py collect --dir D [D ...] --out data/g4/d3

`inputs` checks the three pinned source files and writes ``mudirac_inputs.csv``. `write` renders one
``<run>/<run>.in`` per member for one run kind. `run` runs MuDirac once per input with exactly one
argument, the input file, and records each exit status and ``.err`` size in ``D/runs.tsv``. `collect`
reads the run directories of the committed kinds and writes the run table, every printed state header,
every printed line and the hydrogen-like comparison of the checked shells, each in a fixed order.

The generator itself is not run by the test suite or the audit: its committed outputs are the input
of record, and ``scripts/generate_g4data.py`` builds the D3 tables from them. This script imports
``openmucf/g4/sources/mudirac130.py`` by path, so it runs under an interpreter without the package.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "mudirac130", ROOT / "openmucf" / "g4" / "sources" / "mudirac130.py"
)
assert _SPEC is not None and _SPEC.loader is not None
md = importlib.util.module_from_spec(_SPEC)
sys.modules["mudirac130"] = md
_SPEC.loader.exec_module(md)

_STATE_FILE = re.compile(r"\.([K-Z][0-9]+)\.out$")
_STATE_HEADER = re.compile(r"^# DiracState with n = (\S+), l = (\S+), s = (\S+)$")
_ENERGY_HEADER = re.compile(r"^# E = (\S+) \+ mc\^2 = (\S+) eV$")


def _write(path: Path, rows: list[list[str]]) -> None:
    path.write_bytes(("\n".join(",".join(row) for row in rows) + "\n").encode("ascii"))


def cmd_inputs(args: argparse.Namespace) -> None:
    rows = md.build_inputs(
        Path(args.radii).read_bytes(), Path(args.abundant).read_bytes(), Path(args.iaea).read_bytes()
    )
    out = Path(args.out) / Path(md.INPUTS_RELPATH).name
    out.write_bytes(md.render_inputs(rows))
    print(f"wrote {out}: {len(rows)} members")
    for z, a, bundled, table in md.radius_disagreements(rows):
        print(f"bundled rms differs from the charge-radii table at 4 decimals: Z={z} A={a} {bundled} vs {table}")


def _inputs_and_cells():
    rows = md.load_inputs(ROOT / md.INPUTS_RELPATH)
    cells, _origins = md.load_validation(ROOT / md.CELLS_RELPATH, ROOT / md.ORIGIN_RELPATH)
    return rows, cells


def cmd_write(args: argparse.Namespace) -> None:
    rows, cells = _inputs_and_cells()
    validation = set(md.gated_nuclides(cells))
    written = 0
    for row in rows:
        if args.kind in md.COMMITTED_KINDS and args.kind not in md.run_kinds(row, validation):
            continue
        run = md.run_id(row, args.kind)
        directory = Path(args.dir) / run
        directory.mkdir(parents=True, exist_ok=True)
        text = md.render_input(row, args.kind, md.extra_lines(cells, row.nuclide))
        (directory / f"{run}.in").write_bytes(text.encode("ascii"))
        written += 1
    print(f"wrote {written} inputs of kind {args.kind} under {args.dir}")


def _run_one(binary: str, directory: Path) -> tuple[str, int, str]:
    run = directory.name
    with open(directory / "stdout.txt", "wb") as sink:
        rc = subprocess.run(
            md.mudirac_argv(binary, f"{run}.in"), cwd=directory, stdout=sink, stderr=subprocess.STDOUT
        ).returncode
    err = directory / f"{run}.err"
    return run, rc, str(err.stat().st_size) if err.exists() else "missing"


def cmd_run(args: argparse.Namespace) -> None:
    directories = sorted(p for p in Path(args.dir).iterdir() if (p / f"{p.name}.in").is_file())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = sorted(pool.map(lambda d: _run_one(args.mudirac, d), directories))
    lines = [f"{run}\t{rc}\t{err}" for run, rc, err in results]
    (Path(args.dir) / "runs.tsv").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    failed = [r for r in results if r[1] != 0 or r[2] != "0"]
    print(f"ran {len(results)} inputs; rc != 0 or non-empty .err: {len(failed)}")
    for run, rc, err in failed:
        print(f"  {run} rc={rc} err_bytes={err}")


def _state_headers(directory: Path) -> dict[str, tuple[str, ...]]:
    """``{orbit: (n, l, s, E, E + mc^2)}`` from every state file of one run, as printed."""
    out = {}
    for path in directory.iterdir():
        match = _STATE_FILE.search(path.name)
        if not match or not path.name.startswith(directory.name + "."):
            continue
        state = energy = None
        for line in path.read_text("ascii").splitlines()[:4]:
            state = state or _STATE_HEADER.match(line)
            energy = energy or _ENERGY_HEADER.match(line)
        if not (state and energy):
            raise SystemExit(f"{path}: no state header")
        out[match.group(1)] = (*state.groups(), *energy.groups())
    return out


def _orbit_order(orbit: str) -> tuple[int, int]:
    return (ord(orbit[0]), int(orbit[1:]))


def _lines(directory: Path) -> list[tuple[str, str, str]]:
    """Every row of the run's ``.xr.out`` after its two header lines, as printed and in order."""
    path = directory / f"{directory.name}.xr.out"
    if not path.exists():
        return []
    out = []
    for line in path.read_text("ascii").splitlines()[2:]:
        fields = line.split()
        if len(fields) != 3:
            raise SystemExit(f"{path}: unexpected row {line!r}")
        out.append((fields[0], fields[1], fields[2]))
    return out


def cmd_collect(args: argparse.Namespace) -> None:
    rows, cells = _inputs_and_cells()
    validation = set(md.gated_nuclides(cells))
    by_kind = {}
    for directory in args.dir:
        for line in (Path(directory) / "runs.tsv").read_text("ascii").splitlines():
            run, rc, err = line.split("\t")
            by_kind[run] = (Path(directory) / run, rc, err)
    runs = [["run", "Z", "A", "kind", "rc", "err_bytes"]]
    states = [["run", "Z", "A", "kind", "state", "n", "l", "s", "binding_eV", "total_eV"]]
    lines = [["run", "Z", "A", "kind", "line", "delta_e_eV", "w12_per_s"]]
    nmax = [["Z", "A", "n", "state", "full_binding_eV", "ideal_binding_eV"]]
    headers = {}
    for row in rows:
        key = [str(row.z), str(row.a)]
        for kind in md.run_kinds(row, validation):
            run = md.run_id(row, kind)
            if run not in by_kind:
                raise SystemExit(f"no result for {run}: run every committed kind before collecting")
            directory, rc, err = by_kind[run]
            runs.append([run, *key, kind, rc, err])
            headers[run] = _state_headers(directory)
            if kind in md.PRINTED_KINDS:
                for orbit in sorted(headers[run], key=_orbit_order):
                    states.append([run, *key, kind, orbit, *headers[run][orbit]])
                for name, delta, rate in _lines(directory):
                    lines.append([run, *key, kind, name, delta, rate])
        for n in range(md.IDEAL_FROM, md.N_MAX + 1):
            full, ideal = headers[md.run_id(row, "base")], headers[md.run_id(row, f"ideal{n}")]
            for orbit in md.circular_orbits(n):
                nmax.append([*key, str(n), orbit, full.get(orbit, ("",) * 5)[3],
                             ideal.get(orbit, ("",) * 5)[3]])
    out = Path(args.out)
    for relpath, table in ((md.RUNS_RELPATH, runs), (md.STATES_RELPATH, states), (md.LINES_RELPATH, lines),
                           (md.NMAX_RELPATH, nmax)):
        _write(out / Path(relpath).name, table)
        print(f"wrote {Path(relpath).name}: {len(table) - 1} rows")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("--radii", required=True)
    p.add_argument("--abundant", required=True)
    p.add_argument("--iaea", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("write")
    p.add_argument("--kind", required=True, choices=md.COMMITTED_KINDS + md.EVIDENCE_KINDS)
    p.add_argument("--dir", required=True)
    p = sub.add_parser("run")
    p.add_argument("--mudirac", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--jobs", type=int, required=True)
    p = sub.add_parser("collect")
    p.add_argument("--dir", required=True, nargs="+")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"inputs": cmd_inputs, "write": cmd_write, "run": cmd_run, "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
