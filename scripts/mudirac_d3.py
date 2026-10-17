"""Run the patched MuDirac 1.3.0 over the D3 input set and commit what it prints.

    python3 scripts/mudirac_d3.py inputs --radii R --abundant A --iaea I --out data/g4/d3
    python3 scripts/mudirac_d3.py write-recipe --batch B --dir D
    python3 scripts/mudirac_d3.py run --mudirac BIN --dir D --jobs N [--shard i/n] [--time-v] [--heads-only]
    python3 scripts/mudirac_d3.py collect-recipe --dir D [D ...] --out data/g4/d3
    python3 scripts/mudirac_d3.py geant4-levels --harvest H --out data/g4/d3/geant4_cascade_levels.csv

`inputs` checks the three pinned source files and writes ``mudirac_inputs.csv``. `write-recipe` renders
the inputs of one batch of the recipe (MuDirac 1.3.0 built with the patch in ``third_party/mudirac/``):
``prod`` (every member's kinds, the anchor kind in place of ``base``, each with its Uehling-free twin),
``refine`` (each anchor run one step finer in each setting), ``rminus`` (the radius moved down, where
its rms stays inside the 2pF domain) or ``stages`` (each compared nuclide's run at every stage from the
shipped input to the recipe). `run` runs MuDirac once per input with exactly one argument, the input
file, and records each exit status, ``.err`` size, wall time and input digest in ``D/runs.tsv``
(``D/runs.<i>of<n>.tsv`` for the shard ``i/n`` of the sorted inputs); ``--time-v`` wraps each run in
``/usr/bin/time -v`` so its peak memory is kept beside it, and ``--heads-only`` keeps, of every state
file a run writes, its five header lines and drops the transition-matrix files. `collect-recipe`
re-renders every batch and requires every run to be listed with the digest of its re-rendered input
and to be complete -- exit status 0, an empty ``.err``, every expected line printed once at its input's
decimals and every expected state's header -- and refuses, naming the run, otherwise; then it writes
the run table and, of each run, the printed state headers and line energies its committed view names,
in the batches' order: of a hydrogen-like run the checked shells' circular states, of every other run
every printed line and the states it joins (the printed transition rates are not committed: no table
reads them). `geant4-levels` reads the output of
``cpp/tools/harvest_d3.cc`` on a Geant4 build without the overlay and writes the cascade's level
energies for every gated validation nuclide.

The generator itself is not run by the test suite or the audit: its committed outputs are the input
of record, and ``scripts/generate_g4data.py`` builds the D3 tables from them. This script imports
``openmucf/g4/sources/mudirac130.py`` by path, so it runs under an interpreter without the package.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "mudirac130", ROOT / "openmucf" / "g4" / "sources" / "mudirac130.py"
)
assert _SPEC is not None and _SPEC.loader is not None
md = importlib.util.module_from_spec(_SPEC)
sys.modules["mudirac130"] = md
_SPEC.loader.exec_module(md)


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


def _keep_heads(directory: Path) -> None:
    """Cut every state file of the run in ``directory`` to its five header lines and drop its
    transition-matrix files; the ``.xr.out`` is left whole."""
    run = directory.name
    for path in directory.glob(f"{run}.*.out"):
        if path.name == f"{run}.xr.out":
            continue
        if path.name.endswith(".tmat.out"):
            path.unlink()
            continue
        with open(path, "rb") as source:
            head = b"".join(source.readline() for _ in range(5))
        path.write_bytes(head)


def _run_one(binary: str, directory: Path, time_v: bool, timeout: int | None,
             heads_only: bool = False) -> tuple[str, int, str, str, str]:
    """Run one input: ``(run, rc, err_bytes, wall_s, input_sha256)``; MuDirac gets exactly one
    argument whether or not ``/usr/bin/time -v`` wraps it."""
    run = directory.name
    digest = hashlib.sha256((directory / f"{run}.in").read_bytes()).hexdigest()
    argv = md.mudirac_argv(binary, f"{run}.in")
    if time_v:
        argv = ["/usr/bin/time", "-v", "-o", "time_v.txt", *argv]
    start = time.monotonic()
    with open(directory / "stdout.txt", "wb") as sink:
        try:
            rc = subprocess.run(argv, cwd=directory, stdout=sink, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    wall = time.monotonic() - start
    if heads_only:
        _keep_heads(directory)
    err = directory / f"{run}.err"
    return run, rc, str(err.stat().st_size) if err.exists() else "missing", f"{wall:.3f}", digest


def cmd_run(args: argparse.Namespace) -> None:
    index, count = (int(part) for part in args.shard.split("/"))
    if not 0 <= index < count:
        raise SystemExit(f"--shard must be i/n with 0 <= i < n, got {args.shard}")
    directories = sorted(p for p in Path(args.dir).iterdir() if (p / f"{p.name}.in").is_file())
    directories = [d for position, d in enumerate(directories) if position % count == index]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = sorted(pool.map(lambda d: _run_one(args.mudirac, d, args.time_v, args.timeout, args.heads_only),
                                  directories))
    lines = ["\t".join(fields) for run, rc, err, wall, digest in results
             for fields in [(run, str(rc), err, wall, digest)]]
    name = "runs.tsv" if count == 1 else f"runs.{index}of{count}.tsv"
    (Path(args.dir) / name).write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    failed = [r for r in results if r[1] != 0 or r[2] != "0"]
    total = sum(float(r[3]) for r in results)
    print(f"ran {len(results)} inputs (shard {args.shard}, wall {total:.1f} s summed); "
          f"rc != 0 or non-empty .err: {len(failed)}")
    for run, rc, err, _wall, _digest in failed:
        print(f"  {run} rc={rc} err_bytes={err}")


def _results(directories: list[str]) -> list[tuple[str, Path, str, str, str, str]]:
    """Every ``(run, directory, rc, err_bytes, wall_s, input_sha256)`` the ``runs*.tsv`` files of
    ``directories`` record."""
    out = []
    for directory in directories:
        for table in sorted(Path(directory).glob("runs*.tsv")):
            for line in table.read_text("ascii").splitlines():
                run, rc, err, wall, digest = line.split("\t")
                out.append((run, Path(directory) / run, rc, err, wall, digest))
    return out


def cmd_write_recipe(args: argparse.Namespace) -> None:
    rows, cells = _inputs_and_cells()
    runs = md.recipe_runs(rows, cells, args.batch)
    for r in runs:
        directory = Path(args.dir) / r.run
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{r.run}.in").write_bytes(r.text.encode("ascii"))
    print(f"wrote {len(runs)} inputs of recipe batch {args.batch} under {args.dir}")


def cmd_collect_recipe(args: argparse.Namespace) -> None:
    rows, cells = _inputs_and_cells()
    results = {}
    for run, directory, rc, err, wall, digest in _results(args.dir):
        if run in results:
            raise SystemExit(f"{run}: listed twice in the runs tables")
        results[run] = (directory, rc, err, wall, digest)
    runs = [list(md.RUNS_COLUMNS)]
    states = [list(md.STATES_COLUMNS)]
    lines = [list(md.LINES_COLUMNS)]
    made: dict[str, int] = {}
    for batch in md.RECIPE_BATCHES:
        for r in md.recipe_runs(rows, cells, batch):
            if r.run not in results:
                raise SystemExit(f"no result for {r.run}: run every batch before collecting")
            directory, rc, err, wall, digest = results.pop(r.run)
            if digest != hashlib.sha256(r.text.encode("ascii")).hexdigest():
                raise SystemExit(f"{r.run}: the input it ran is not the input the renderer makes now")
            try:
                headers, printed = md.read_recipe_run(directory, r.run, r.text, rc, err)
            except md.IncompleteRunError as error:
                raise SystemExit(f"collect refused: {error}") from error
            domain = md.density_domain(r.row, r.kind) if r.stage in ("rms", "mass", "recipe") else ""
            runs.append([r.run, str(r.row.z), str(r.row.a), r.kind, r.stage, "on" if r.uehling else "off", r.variant,
                         domain, rc, err, wall, digest, headers[md.orbit(md.N_MAX, True)][4]])
            view_lines, view_states = md.committed_view(r.kind, [name for name, _e, _r in printed])
            for orbit in sorted(view_states, key=lambda s: (ord(s[0]), int(s[1:]))):
                states.append([r.run, orbit, *headers[orbit][:4]])
            for name, delta, _rate in printed:
                if name in view_lines:
                    lines.append([r.run, name, delta])
            made[batch] = made.get(batch, 0) + 1
    if results:
        raise SystemExit(f"runs in no batch: {sorted(results)[:3]}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for relpath, table in ((md.RUNS_RELPATH, runs), (md.STATES_RELPATH, states), (md.LINES_RELPATH, lines)):
        path = out / Path(relpath).name
        _write(path, table)
        print(f"wrote {path.name}: {len(table) - 1} rows, {path.stat().st_size} bytes")
    print("runs complete per batch: " + ", ".join(f"{b} {n}" for b, n in made.items()))


def cmd_geant4_levels(args: argparse.Namespace) -> None:
    _rows, cells = _inputs_and_cells()
    nuclides = md.gated_nuclides(cells)
    optional = sorted(set(md.centroid_nuclides(cells)) - set(nuclides))
    payload = md.render_geant4_levels(Path(args.harvest).read_bytes().decode("ascii"), nuclides, optional)
    Path(args.out).write_bytes(payload)
    present = set(md.load_geant4_levels(Path(args.out)))
    for z, a in optional:
        if (z, a) not in present:
            print(f"no C line: Z={z} A={a} (leaves the stock comparison)")
    print(f"wrote {args.out}: {len(present)} nuclides")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("--radii", required=True)
    p.add_argument("--abundant", required=True)
    p.add_argument("--iaea", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("write-recipe")
    p.add_argument("--batch", required=True, choices=md.RECIPE_BATCHES)
    p.add_argument("--dir", required=True)
    p = sub.add_parser("run")
    p.add_argument("--mudirac", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--jobs", type=int, required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--time-v", action="store_true")
    p.add_argument("--timeout", type=int)
    p.add_argument("--heads-only", action="store_true")
    p = sub.add_parser("collect-recipe")
    p.add_argument("--dir", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p = sub.add_parser("geant4-levels")
    p.add_argument("--harvest", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"inputs": cmd_inputs, "write-recipe": cmd_write_recipe, "run": cmd_run, "collect-recipe": cmd_collect_recipe,
     "geant4-levels": cmd_geant4_levels}[args.command](args)


if __name__ == "__main__":
    main()
