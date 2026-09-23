"""Run MuDirac 1.3.0 over the D3 input set and commit what it prints.

    python3 scripts/mudirac_d3.py inputs --radii R --abundant A --iaea I --out data/g4/d3
    python3 scripts/mudirac_d3.py write --kind K [--level L] --dir D
    python3 scripts/mudirac_d3.py run --mudirac BIN --dir D --jobs N [--shard i/n] [--time-v]
    python3 scripts/mudirac_d3.py collect --dir D [D ...] --out data/g4/d3
    python3 scripts/mudirac_d3.py collect-numerics --dir D [D ...] --out data/g4/d3
    python3 scripts/mudirac_d3.py geant4-levels --harvest H --out data/g4/d3/geant4_cascade_levels.csv

`inputs` checks the three pinned source files and writes ``mudirac_inputs.csv``. `write` renders one
``<run>/<run>.in`` per member for one run kind: a committed or evidence kind over the input set, or
``rminus`` (the radius moved down, over the kept members whose moved radius stays in the model's
domain) or ``grid`` at ``--level`` (the base input with its three numerical settings refined, over
the kept members that level is run for). `run` runs MuDirac once per input with exactly one
argument, the input file, and records each exit status, ``.err`` size, wall time and input digest in
``D/runs.tsv`` (``D/runs.<i>of<n>.tsv`` for the shard ``i/n`` of the sorted inputs); ``--time-v``
wraps each run in ``/usr/bin/time -v`` so its peak memory is kept beside it. `collect` reads the run
directories of the committed kinds and writes the run table, every printed state header, every
printed line and the hydrogen-like comparison of the checked shells, each in a fixed order.
`collect-numerics` does the same for the ``rminus`` and ``grid`` directories into the three numerics
tables, with a row of status ``INVALID_PERTURBATION_DOMAIN`` for each member whose ``rminus`` run
was not made. `geant4-levels` reads the output of ``cpp/tools/harvest_d3.cc`` on a Geant4 build
without the overlay and writes the cascade's level energies for every gated validation nuclide.

The generator itself is not run by the test suite or the audit: its committed outputs are the input
of record, and ``scripts/generate_g4data.py`` builds the D3 tables from them. This script imports
``openmucf/g4/sources/mudirac130.py`` by path, so it runs under an interpreter without the package.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import re
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


def _numerics_members(args: argparse.Namespace, cells) -> list[tuple[str, object, str]]:
    """``(run, row, input text)`` of every ``rminus`` or ``grid`` run of ``args.level`` to make."""
    out = []
    for row in md.kept_members(md.load_outputs(ROOT)):
        extra = md.extra_lines(cells, row.nuclide)
        if args.kind == "rminus":
            if not md.rminus_in_domain(row):
                continue
            out.append((md.numerics_run_id(row, "rminus", 0), row, md.render_input(row, "rminus", extra)))
        elif args.level in md.numerics_levels(row):
            out.append((md.numerics_run_id(row, "grid", args.level), row,
                        md.render_numerics_input(row, args.level, extra)))
    return out


def cmd_write(args: argparse.Namespace) -> None:
    rows, cells = _inputs_and_cells()
    validation = set(md.gated_nuclides(cells))
    if args.kind in md.RESPONSE_KINDS + ("grid",):
        inputs = _numerics_members(args, cells)
    else:
        inputs = [(md.run_id(row, args.kind), row, md.render_input(row, args.kind, md.extra_lines(cells, row.nuclide)))
                  for row in rows
                  if args.kind not in md.COMMITTED_KINDS or args.kind in md.run_kinds(row, validation)]
    for run, _row, text in inputs:
        directory = Path(args.dir) / run
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{run}.in").write_bytes(text.encode("ascii"))
    print(f"wrote {len(inputs)} inputs of kind {args.kind} level {args.level} under {args.dir}")


def cmd_write_settings(args: argparse.Namespace) -> None:
    _rows, cells = _inputs_and_cells()
    kept = {row.nuclide: row for row in md.kept_members(md.load_outputs(ROOT))}
    wanted = md.settings_nuclides(cells)
    if not set(wanted) <= set(kept):
        raise SystemExit(f"settings nuclides not kept: {sorted(set(wanted) - set(kept))}")
    made = 0
    for key in wanted:
        row = kept[key]
        extra = md.extra_lines(cells, key)
        for level, uehling in md.SETTINGS:
            setting = md.setting_id(level, uehling)
            run = md.settings_run_id(row, level, uehling)
            directory = Path(args.dir) / setting / run
            directory.mkdir(parents=True, exist_ok=True)
            text = md.render_settings_input(row, level, uehling, extra)
            (directory / f"{run}.in").write_bytes(text.encode("ascii"))
            made += 1
    print(f"wrote {made} settings inputs for {len(wanted)} nuclides under {args.dir}")


def _run_one(binary: str, directory: Path, time_v: bool, timeout: int | None) -> tuple[str, int, str, str, str]:
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
    err = directory / f"{run}.err"
    return run, rc, str(err.stat().st_size) if err.exists() else "missing", f"{wall:.3f}", digest


def cmd_run(args: argparse.Namespace) -> None:
    index, count = (int(part) for part in args.shard.split("/"))
    if not 0 <= index < count:
        raise SystemExit(f"--shard must be i/n with 0 <= i < n, got {args.shard}")
    directories = sorted(p for p in Path(args.dir).iterdir() if (p / f"{p.name}.in").is_file())
    directories = [d for position, d in enumerate(directories) if position % count == index]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = sorted(pool.map(lambda d: _run_one(args.mudirac, d, args.time_v, args.timeout), directories))
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


def _results(directories: list[str]) -> list[tuple[str, Path, str, str, str, str]]:
    """Every ``(run, directory, rc, err_bytes, wall_s, input_sha256)`` the ``runs*.tsv`` files of
    ``directories`` record; a file written before wall time and digest were kept gives empty ones."""
    out = []
    for directory in directories:
        for table in sorted(Path(directory).glob("runs*.tsv")):
            for line in table.read_text("ascii").splitlines():
                fields = line.split("\t")
                run, rc, err = fields[:3]
                wall, digest = (fields[3], fields[4]) if len(fields) >= 5 else ("", "")
                out.append((run, Path(directory) / run, rc, err, wall, digest))
    return out


def cmd_collect_numerics(args: argparse.Namespace) -> None:
    results = {run: (directory, rc, err, wall, digest)
               for run, directory, rc, err, wall, digest in _results(args.dir)}
    kept = md.kept_members(md.load_outputs(ROOT))
    runs = [list(md.NUMERICS_RUNS_COLUMNS)]
    states = [list(md.NUMERICS_STATES_COLUMNS)]
    lines = [list(md.NUMERICS_LINES_COLUMNS)]
    made = {}
    invalid = 0
    rows = {row.nuclide: row for row in kept}
    for run, z, a, kind, level in md.expected_numerics_runs(kept):
        key = [str(z), str(a), kind, str(level)]
        if kind == "rminus" and not md.rminus_in_domain(rows[(z, a)]):
            runs.append([run, *key, md.INVALID_PERTURBATION_DOMAIN, "", "", "", ""])
            invalid += 1
            continue
        if run not in results:
            raise SystemExit(f"no result for {run}: run every kind and level before collecting")
        directory, rc, err, wall, digest = results[run]
        if not wall or not digest:
            raise SystemExit(f"{run}: its runs table carries no wall time or input digest")
        runs.append([run, *key, md.RAN, rc, err, wall, digest])
        made[f"{kind}{level}"] = made.get(f"{kind}{level}", 0) + 1
        headers = _state_headers(directory)
        for orbit in sorted(headers, key=_orbit_order):
            states.append([run, *key, orbit, *headers[orbit]])
        for name, delta, rate in _lines(directory):
            lines.append([run, *key, name, delta, rate])
    out = Path(args.out)
    for relpath, table in ((md.NUMERICS_RUNS_RELPATH, runs), (md.NUMERICS_STATES_RELPATH, states),
                           (md.NUMERICS_LINES_RELPATH, lines)):
        _write(out / Path(relpath).name, table)
        print(f"wrote {Path(relpath).name}: {len(table) - 1} rows, "
              f"{(out / Path(relpath).name).stat().st_size} bytes")
    print(f"kept members {len(kept)}; runs made per kind and level: "
          + ", ".join(f"{k} {v}" for k, v in sorted(made.items()))
          + f"; {md.INVALID_PERTURBATION_DOMAIN} {invalid}")


def cmd_collect_settings(args: argparse.Namespace) -> None:
    _rows, cells = _inputs_and_cells()
    kept = {row.nuclide: row for row in md.kept_members(md.load_outputs(ROOT))}
    wanted = md.settings_nuclides(cells)
    results = {run: (directory, rc, err, wall, digest)
               for run, directory, rc, err, wall, digest in _results(
                   [str(p) for p in Path(args.dir).glob("g*") if p.is_dir()])}
    runs = [list(md.SETTINGS_RUNS_COLUMNS)]
    states = [list(md.SETTINGS_STATES_COLUMNS)]
    lines = [list(md.SETTINGS_LINES_COLUMNS)]
    for key in wanted:
        row = kept[key]
        for level, uehling in md.SETTINGS:
            setting = md.setting_id(level, uehling)
            run = md.settings_run_id(row, level, uehling)
            if run not in results:
                raise SystemExit(f"no result for {run}")
            directory, rc, err, wall, digest = results.pop(run)
            runs.append([run, str(row.z), str(row.a), setting, *md.settings_values(level, uehling),
                         rc, err, wall, digest])
            if rc == "0" and err == "0":
                for orbit, header in sorted(_state_headers(directory).items(), key=lambda item: _orbit_order(item[0])):
                    states.append([run, str(row.z), str(row.a), setting, orbit, *header])
                for name, delta, rate in _lines(directory):
                    lines.append([run, str(row.z), str(row.a), setting, name, delta, rate])
    if results:
        raise SystemExit(f"unexpected settings results: {sorted(results)[:3]}")
    out = Path(args.out)
    for relpath, table in ((md.SETTINGS_RUNS_RELPATH, runs), (md.SETTINGS_STATES_RELPATH, states),
                           (md.SETTINGS_LINES_RELPATH, lines)):
        _write(out / Path(relpath).name, table)
        print(f"wrote {Path(relpath).name}: {len(table) - 1} rows")


def cmd_collect(args: argparse.Namespace) -> None:
    rows, cells = _inputs_and_cells()
    validation = set(md.gated_nuclides(cells))
    by_kind = {}
    for run, directory, rc, err, _wall, _digest in _results(args.dir):
        by_kind[run] = (directory, rc, err)
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
    p = sub.add_parser("write")
    p.add_argument("--kind", required=True,
                   choices=md.COMMITTED_KINDS + md.EVIDENCE_KINDS + md.RESPONSE_KINDS + ("grid",))
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--dir", required=True)
    p = sub.add_parser("write-settings")
    p.add_argument("--dir", required=True)
    p = sub.add_parser("run")
    p.add_argument("--mudirac", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--jobs", type=int, required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--time-v", action="store_true")
    p.add_argument("--timeout", type=int)
    p = sub.add_parser("collect")
    p.add_argument("--dir", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p = sub.add_parser("collect-numerics")
    p.add_argument("--dir", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p = sub.add_parser("collect-settings")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("geant4-levels")
    p.add_argument("--harvest", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"inputs": cmd_inputs, "write": cmd_write, "write-settings": cmd_write_settings,
     "run": cmd_run, "collect": cmd_collect, "collect-numerics": cmd_collect_numerics,
     "collect-settings": cmd_collect_settings, "geant4-levels": cmd_geant4_levels}[args.command](args)


if __name__ == "__main__":
    main()
