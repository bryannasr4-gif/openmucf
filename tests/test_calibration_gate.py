"""The calibration cells file has one admission, ``admit_calibration_cells`` in
scripts/generate_calibration.py, and every reader that renders, audits or publishes from the file takes only
what it admits.

Two kinds of test hold that. A structural scan (`cells_file_readers`) over every module of openmucf/ and
scripts/ fails if a module other than the gate's names the file, reads it, or takes anything from the
calibration script but its gate, and if a function of the script outside the gate and its three measurement
steps reads it; its drills plant each kind of read and must see it named. The consumer matrix then runs the
actual entry points -- `--render`, `--audit`, the FRONTIER.md generator's `main` and the FINDINGS.md generator
itself -- on a working copy whose cells file carries one defect at a time: every print of the documents as
released replaced, every one omitted, the record or the released documents damaged, each input digest
stale, missing or unknown, and each whole record absent. Each must stop, naming the defect, before it
writes or reports success.
"""

import ast
import copy
import functools
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from openmucf import provenance, uq

#: the repository this test file is in (nothing is read from it when the module is collected)
_ROOT = Path(__file__).resolve().parents[1]
GATE = "scripts/generate_calibration.py"
CELLS = "CALIBRATION_CELLS"
#: what a module outside the gate may take from the calibration script
GATE_API = frozenset({"admit_calibration_cells", "AdmittedCells", "CellsRefused"})
#: the functions of the calibration script that read the cells file: the gate, and the three measurement
#: steps that rewrite it (each admits what it carries or binds; see their docstrings)
GATE_READERS = frozenset({"admit_calibration_cells", "regenerate", "record_convergence", "record_scatter"})
_READ_CALLS = frozenset({"open", "read_text", "read_bytes", "load", "loads", "load_json", "loads_json",
                         "file_sha256"})


# ============================================================================ the structural scan
def _modules(root):
    return sorted(p.relative_to(root).as_posix()
                  for d in ("openmucf", "scripts") for p in (root / d).rglob("*.py"))


def _call_name(node):
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def _docstrings(tree):
    """The docstring nodes of a module and of its classes and functions, and the lines they span."""
    nodes, lines = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                nodes.add(id(first.value))
                lines |= set(range(first.lineno, first.end_lineno + 1))
    return nodes, lines


def _gate_module_names(tree):
    """Names bound, in source order, to a module loaded from a path that names generate_calibration."""
    specs, mods = set(), set()
    calls = (n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call))
    assigns = sorted(calls, key=lambda n: (n.lineno, n.col_offset))
    for a in assigns:
        targets = [t.id for t in a.targets if isinstance(t, ast.Name)]
        name = _call_name(a.value)
        if name == "spec_from_file_location":
            gate = any(isinstance(c, ast.Constant) and isinstance(c.value, str)
                       and "generate_calibration" in c.value for c in ast.walk(a.value))
            for t in targets:
                (specs.add if gate else specs.discard)(t)
        elif name == "module_from_spec":
            arg = a.value.args[0] if a.value.args else None
            gate = isinstance(arg, ast.Name) and arg.id in specs
            for t in targets:
                (mods.add if gate else mods.discard)(t)
    return mods


def _reads_the_cells_file(call):
    """A call that reads the cells file by the script's own name for it: `load()` with no argument (its
    default is the cells file), or a reading call whose function or arguments name CELLS_JSON."""
    name = _call_name(call)
    if name == "load" and not call.args and not call.keywords:
        return True
    if name not in _READ_CALLS:
        return False
    parts = [call.func, *call.args, *(k.value for k in call.keywords)]
    return any(isinstance(n, ast.Name) and n.id == "CELLS_JSON" for p in parts for n in ast.walk(p))


def _gate_findings(rel, tree):
    out = []
    for top in tree.body:
        is_def = isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        owner = top.name if is_def else "<module>"
        for n in ast.walk(top):
            if isinstance(n, ast.Call) and _reads_the_cells_file(n) and owner not in GATE_READERS:
                out.append(f"{rel}:{n.lineno}: {owner} reads the cells file outside the gate and the "
                           "measurement steps")
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and n.name not in ("load", "admit_calibration_cells"):
                for d in [*n.args.defaults, *n.args.kw_defaults]:
                    if isinstance(d, ast.Name) and d.id == "CELLS_JSON":
                        out.append(f"{rel}:{n.lineno}: {n.name} defaults to the cells file")
    return out


def cells_file_readers(root):
    """Every way a module of openmucf/ or scripts/ under ``root`` reads the cells file other than through the
    gate, as 'path:line: what'. Outside the calibration script: a string or bytes constant naming the file
    (docstrings aside), any use of the script's CELLS_JSON, anything taken from a module loaded from the
    script but GATE_API, and, by a text scan, any line naming the file that is neither in a docstring nor
    after a comment sign. Inside it: a read of the cells file in a function that is not one of
    GATE_READERS, and a function other than `load` and the gate whose argument defaults to the file."""
    out = set()
    for rel in _modules(root):
        src = (root / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        if rel == GATE:
            out |= set(_gate_findings(rel, tree))
            continue
        doc_nodes, doc_lines = _docstrings(tree)
        gate_mods = _gate_module_names(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)) and id(n) not in doc_nodes:
                text = n.value if isinstance(n.value, str) else n.value.decode("latin-1")
                if CELLS in text:
                    out.add(f"{rel}:{n.lineno}: names the cells file")
            elif isinstance(n, ast.Attribute) and n.attr == "CELLS_JSON":
                out.add(f"{rel}:{n.lineno}: uses the calibration script's CELLS_JSON")
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in gate_mods \
                    and n.attr not in GATE_API:
                out.add(f"{rel}:{n.lineno}: takes {n.attr!r} from the calibration script, not its gate")
            if isinstance(n, ast.ImportFrom) and n.module and "generate_calibration" in n.module:
                out |= {f"{rel}:{n.lineno}: imports {a.name!r} from the calibration script"
                        for a in n.names if a.name not in GATE_API}
            if isinstance(n, ast.Import) and any("generate_calibration" in a.name for a in n.names):
                out.add(f"{rel}:{n.lineno}: imports the calibration script as a module")
        for i, line in enumerate(src.splitlines(), 1):
            at = line.find(CELLS)
            if at >= 0 and i not in doc_lines and not 0 <= line.find("#") < at:
                out.add(f"{rel}:{i}: names the cells file in code")
    return sorted(out)


def test_no_module_reads_the_cells_file_but_through_the_gate():
    assert cells_file_readers(_ROOT) == []


def test_the_scan_covers_every_module_and_finds_the_gate_and_its_readers():
    """The scan is not vacuous: it walks every tracked module of openmucf/ and scripts/ (where git is
    present), the gate module is among them, FINDINGS.md's and FRONTIER.md's generators load the script under
    a name the scan recognizes, and inside the script it finds exactly the three measurement steps reading
    the file."""
    mods = _modules(_ROOT)
    assert GATE in mods and "openmucf/uq.py" in mods
    git = subprocess.run(["git", "-C", str(_ROOT), "ls-files", "openmucf", "scripts"],
                         capture_output=True, text=True)
    if git.returncode == 0 and git.stdout:
        assert {f for f in git.stdout.splitlines() if f.endswith(".py")} <= set(mods)
    for rel in ("scripts/generate_findings.py", "scripts/generate_frontier.py"):
        tree = ast.parse((_ROOT / rel).read_text(encoding="utf-8"))
        assert _gate_module_names(tree), rel
    gate_tree = ast.parse((_ROOT / GATE).read_text(encoding="utf-8"))
    readers = {top.name for top in gate_tree.body if isinstance(top, ast.FunctionDef)
               for n in ast.walk(top) if isinstance(n, ast.Call) and _reads_the_cells_file(n)}
    assert readers == {"regenerate", "record_convergence", "record_scatter"}


_LOAD_GATE = ('import importlib.util\nfrom pathlib import Path\n'
              'spec = importlib.util.spec_from_file_location("g", Path(__file__).parent / '
              '"generate_calibration.py")\n'
              'gen = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(gen)\n')
_PLANTED = [
    ("a path read", "scripts/planted.py",
     'import json\nfrom pathlib import Path\n'
     'cells = json.loads(Path("CALIBRATION_CELLS.json").read_text())\n',
     "scripts/planted.py:3: names the cells file"),
    ("a package module opening the file", "openmucf/planted.py",
     'from pathlib import Path\nROOT = Path(__file__).parent.parent\n'
     'def peek():\n    return open(ROOT / "CALIBRATION_CELLS.json").read()\n',
     "openmucf/planted.py:4: names the cells file"),
    ("a bytes literal", "scripts/planted.py", 'NAME = b"CALIBRATION_CELLS.json"\n',
     "scripts/planted.py:1: names the cells file"),
    ("the script's raw loader", "scripts/planted.py", _LOAD_GATE + "cells = gen.load()\n",
     "scripts/planted.py:6: takes 'load' from the calibration script, not its gate"),
    ("the script's checks without the gate", "scripts/planted.py",
     _LOAD_GATE + "probs = gen.admission_problems({})\n",
     "scripts/planted.py:6: takes 'admission_problems' from the calibration script, not its gate"),
    ("the script's path constant", "scripts/planted.py", _LOAD_GATE + "p = gen.CELLS_JSON\n",
     "scripts/planted.py:6: uses the calibration script's CELLS_JSON"),
    ("an import of the script", "scripts/planted.py", "from generate_calibration import load\n",
     "scripts/planted.py:1: imports 'load' from the calibration script"),
]
_AUDIT_TRY = ("    try:                # a record the gate refuses is not rendered, recomputed or sampled "
              "against\n")
_GATE_PLANTS = [
    ("the audit reading the file itself",
     _AUDIT_TRY + "        committed = admit_calibration_cells().data\n",
     _AUDIT_TRY + "        committed = load()\n",
     "audit reads the cells file outside the gate and the measurement steps"),
    ("a renderer reading the file", "def _estimator_section(cells):\n",
     "def _estimator_section(cells):\n    cells = provenance.load_json(CELLS_JSON)\n",
     "_estimator_section reads the cells file outside the gate and the measurement steps"),
    ("a reader defaulting to the file", "def render(cells):\n",
     "def peek(path=CELLS_JSON):\n    return path\n\n\ndef render(cells):\n",
     "peek defaults to the cells file"),
]


def _tree(tmp_path):
    root = tmp_path / "tree"
    for d in ("openmucf", "scripts"):
        (root / d).mkdir(parents=True)
    shutil.copyfile(_ROOT / GATE, root / GATE)
    return root


@pytest.mark.parametrize("label, rel, code, named", _PLANTED, ids=[p[0] for p in _PLANTED])
def test_a_planted_read_outside_the_gate_is_named(tmp_path, label, rel, code, named):
    """Drill: each planted way of reading the file outside the gate is named; the tree without it is clean."""
    root = _tree(tmp_path)
    assert cells_file_readers(root) == []
    (root / rel).write_text(code, encoding="utf-8")
    assert named in cells_file_readers(root)


@pytest.mark.parametrize("label, old, new, named", _GATE_PLANTS, ids=[p[0] for p in _GATE_PLANTS])
def test_a_planted_read_inside_the_script_is_named(tmp_path, label, old, new, named):
    root = _tree(tmp_path)
    src = (root / GATE).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count(old) == 1, label
    (root / GATE).write_text(src.replace(old, new), encoding="utf-8")
    assert [f for f in cells_file_readers(root) if named in f], cells_file_readers(root)


def test_mentions_that_read_nothing_are_not_named(tmp_path):
    """Control: a docstring or comment naming the file, an unrelated JSON read and the gate's own API are
    clean."""
    root = _tree(tmp_path)
    (root / "scripts" / "clean.py").write_text(
        '"""Reads CALIBRATION_CELLS.json only through the gate."""\n' + _LOAD_GATE
        + "import json\n\n\ndef f(p):\n    \"\"\"Not the CALIBRATION_CELLS.json file.\"\"\"\n"
        + "    return json.load(open(p))  # never CALIBRATION_CELLS.json\n\n\n"
        + "cells = gen.admit_calibration_cells()\nrefused = gen.CellsRefused\n", encoding="utf-8")
    assert cells_file_readers(root) == []


# ============================================================================ the consumer matrix
_FILES = ("CALIBRATION.md", "CALIBRATION_CELLS.json", "VALIDATION_CHANNELS.md",
          "forecasts/FC-001-mufuse.json", "openmucf/data/side_channel_inputs.csv",
          "openmucf/data/uq_priors.csv",
          "data/calibration/released_v1.3.0.json")
_RELEASED = _FILES[-1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Reached(Exception):
    """FINDINGS.md's generator went past its admission (its first computation was called)."""


def _raise_reached(*args, **kwargs):
    raise _Reached


@pytest.fixture
def work(tmp_path, monkeypatch):
    """A working copy of every file the gate and the two calibration documents read, as the working
    directory; FRONTIER.md's figure and FINDINGS.md's first computation stand in as markers."""
    gate = _load(_ROOT / GATE, "_gen_calibration")
    for rel in (*_FILES, gate.NOTE_PATH):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / rel, tmp_path / rel)
    monkeypatch.chdir(tmp_path)
    fr = _load(_ROOT / "scripts" / "generate_frontier.py", "_gen_frontier")
    monkeypatch.setattr(fr, "_write_figure", lambda H: pytest.fail("figure drawn"))
    monkeypatch.setattr(uq, "local_sensitivities", _raise_reached)
    return gate, fr, tmp_path


_OUTPUTS = ("CALIBRATION.md", "FRONTIER.md", "FRONTIER_MANIFEST.json", "FINDINGS.md",
            "FINDINGS_MANIFEST.json")


def _outputs(root, gate):
    """Each output's size and modification time, or None where it does not exist (a write changes the
    time)."""
    out = {}
    for p in (*_OUTPUTS, gate.NOTE_PATH):
        try:
            st = os.stat(root / p)
            out[p] = (st.st_size, st.st_mtime_ns)
        except FileNotFoundError:
            out[p] = None
    return out


def _render(gate, fr, root):
    gate.main(["--render"])


def _audit(gate, fr, root):
    compute, sample = gate.compute_cells, gate.nuts_cross_check
    gate.compute_cells = lambda posts=None: pytest.fail("recomputed")
    gate.nuts_cross_check = lambda posteriors: pytest.fail("sampled")
    try:
        gate.audit()
    finally:
        gate.compute_cells, gate.nuts_cross_check = compute, sample


def _frontier(gate, fr, root):
    fr.main()


def _findings(gate, fr, root):
    runpy.run_path(str(_ROOT / "scripts" / "generate_findings.py"), run_name="__main__")


CONSUMERS = {"render": _render, "audit": _audit, "frontier": _frontier, "findings": _findings}


def _refusals(work, fixtures, consumer):
    """Run ``consumer`` on each (label, write-the-defect, named) fixture, each on the pristine copy but for
    the files its defect touches: return the labels it did not refuse by name, or after which it wrote or
    changed a file or went past its admission, with what it did instead."""
    gate, fr, root = work
    pristine = {rel: (root / rel).read_bytes() for rel in _FILES}
    misses, touched = [], ()
    for label, corrupt, named in fixtures:
        for rel in touched:
            (root / rel).write_bytes(pristine[rel])
        touched = tuple(corrupt(root))
        before = _outputs(root, gate)
        try:
            CONSUMERS[consumer](gate, fr, root)
            got = "returned"
        except _Reached:
            got = "admitted (FINDINGS went past its admission)"
        except SystemExit as e:
            got = str(e.code)
        after = _outputs(root, gate)
        if named not in got or after != before or (root / "figures").exists():
            misses.append(f"{label}: {got[:300]}")
            for p, sig in after.items():            # leave the next fixture the copy it expects
                if sig is not None and before[p] is None:
                    (root / p).unlink()
            shutil.rmtree(root / "figures", ignore_errors=True)
    return misses


@functools.lru_cache(maxsize=1)
def _committed_text():
    return (_ROOT / "CALIBRATION_CELLS.json").read_text(encoding="utf-8")


def _committed():
    return json.loads(_committed_text())


def _cells_fixture(edit):
    """A defect written into the cells file (the committed record, edited); it touches that file alone."""
    def corrupt(root):
        cells = json.loads(_committed_text())
        edit(cells)
        (root / "CALIBRATION_CELLS.json").write_text(json.dumps(cells), encoding="utf-8")
        return ("CALIBRATION_CELLS.json",)
    return corrupt


def _released():
    """The released prints, derived by the gate from the released documents, and equal to the committed
    record."""
    gate = _load(_ROOT / GATE, "_gen_calibration")
    cwd = Path.cwd()
    os.chdir(_ROOT)
    try:
        prints, problems = gate.released_prints()
    finally:
        os.chdir(cwd)
    assert problems == [] and prints == _committed()["shipped_v1_3_0"]
    return prints


def _wrong_print_fixtures():
    out = []
    for key, val in sorted(_released().items()):
        bad = "999" if val != "999" else "998"
        def edit(c, k=key, b=bad):
            c["shipped_v1_3_0"].update({k: b})

        out.append((f"wrong print {key}", _cells_fixture(edit),
                    f"shipped_v1_3_0[{key!r}]: {bad!r} is not the released print {val!r}"))
    return out


def _omitted_print_fixtures():
    return [(f"omitted print {key}", _cells_fixture(lambda c, k=key: c["shipped_v1_3_0"].pop(k)),
             f"shipped_v1_3_0: no {key!r}") for key in sorted(_released())]


def _record_fixtures():
    k_cal = sorted(k for k in _released() if k.startswith("CALIBRATION.md|"))
    k_val = "VALIDATION_CHANNELS.md|V_breunlich_lambdac|predicted"

    def drop_calibration(c):
        for k in k_cal:
            del c["shipped_v1_3_0"][k]

    def edit_released(old, new):
        def corrupt(root):
            p = root / _RELEASED
            rec = json.loads(p.read_text(encoding="utf-8"))
            assert old in rec["documents"]["CALIBRATION.md"]
            rec["documents"]["CALIBRATION.md"] = rec["documents"]["CALIBRATION.md"].replace(old, new, 1)
            p.write_text(json.dumps(rec), encoding="utf-8")
            return (_RELEASED,)
        return corrupt

    def drop_released(root):
        (root / _RELEASED).unlink()
        return (_RELEASED,)

    out = [
        ("the shipped record absent", _cells_fixture(lambda c: c.pop("shipped_v1_3_0")),
         "shipped_v1_3_0: NoneType is not a record"),
        ("the shipped record empty", _cells_fixture(lambda c: c["shipped_v1_3_0"].clear()),
         f"shipped_v1_3_0: no {k_cal[0]!r}"),
        ("every CALIBRATION.md print removed", _cells_fixture(drop_calibration),
         f"shipped_v1_3_0: no {k_cal[-1]!r}"),
        ("a print the release does not have", _cells_fixture(
            lambda c: c["shipped_v1_3_0"].update({"CALIBRATION.md|Finding|weak|mean": "0.46"})),
         "not a cell of the released documents"),
        ("the V_breunlich_lambdac print moved", _cells_fixture(
            lambda c: c["shipped_v1_3_0"].update({k_val: "143822639.3"})),
         f"shipped_v1_3_0[{k_val!r}]: '143822639.3' is not the released print '143822639.2'"),
        ("a released document altered", edit_released("| 0.46 |", "| 0.47 |"),
         "CALIBRATION.md is not the document released at v1.3.0"),
        ("the released documents absent", drop_released, "the released documents are unreadable"),
    ]
    out.append(("the inputs not a record", _cells_fixture(lambda c: c.update(inputs=[])),
                "inputs: [] is not a record of input digests"))
    for key in ("convergence", "nuts_cross_check"):
        out.append((f"the {key} record absent", _cells_fixture(lambda c, k=key: c.pop(k)),
                    "convergence: no record" if key == "convergence" else "nuts_cross_check: chains None"))
    for key in ("schema", "estimator", "posteriors", "sweep", "q_net", "legacy_tension", "inputs",
                "computed_on"):
        out.append((f"the computed member {key} absent", _cells_fixture(lambda c, k=key: c.pop(k)),
                    f"cells: no {key!r}"))
    return out


_INPUTS = {"side_channel_inputs_csv_sha256": "openmucf/data/side_channel_inputs.csv",
           "uq_priors_csv_sha256": "openmucf/data/uq_priors.csv",
           "fc001_card_sha256": "forecasts/FC-001-mufuse.json"}


def _input_fixtures():
    """Each recorded input digest stale (its live file gains a byte) and missing, and one unknown digest."""
    def stale(rel):
        def corrupt(root):
            with open(root / rel, "ab") as f:
                f.write(b"\n")
            return (rel,)
        return corrupt

    out = [(f"{k} stale", stale(rel), f"inputs.{k}: recorded") for k, rel in _INPUTS.items()]
    out += [(f"{k} missing", _cells_fixture(lambda c, k=k: c["inputs"].pop(k)),
             f"inputs.{k}: recorded None vs live") for k in _INPUTS]
    out.append(("an unknown input", _cells_fixture(lambda c: c["inputs"].update(extra_csv_sha256="0" * 64)),
                "inputs.extra_csv_sha256: no such input"))
    return out


def test_the_input_fixtures_are_every_recorded_input():
    gate = _load(_ROOT / GATE, "_gen_calibration")
    assert _INPUTS == gate.INPUT_FILES
    assert set(_committed()["inputs"]) == set(gate.INPUT_FILES)
    assert len(_input_fixtures()) == 2 * len(gate.INPUT_FILES) + 1


@pytest.mark.parametrize("consumer", sorted(CONSUMERS))
def test_every_wrong_released_print_is_refused_by_every_consumer(work, consumer):
    fixtures = _wrong_print_fixtures()
    assert len(fixtures) == len(_committed()["shipped_v1_3_0"])
    assert _refusals(work, fixtures, consumer) == []


@pytest.mark.parametrize("consumer", sorted(CONSUMERS))
def test_every_omitted_released_print_is_refused_by_every_consumer(work, consumer):
    fixtures = _omitted_print_fixtures()
    assert len(fixtures) == len(_committed()["shipped_v1_3_0"])
    assert _refusals(work, fixtures, consumer) == []


@pytest.mark.parametrize("consumer", sorted(CONSUMERS))
def test_every_damaged_or_absent_record_is_refused_by_every_consumer(work, consumer):
    assert _refusals(work, _record_fixtures(), consumer) == []


@pytest.mark.parametrize("consumer", sorted(CONSUMERS))
def test_every_input_digest_fixture_is_refused_by_every_consumer(work, consumer):
    assert _refusals(work, _input_fixtures(), consumer) == []


def _same_text(a, b):
    return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")


def test_every_consumer_takes_the_genuine_record(work, monkeypatch):
    """Control: on the unaltered copy, `--render` writes the committed documents, FRONTIER.md's generator
    writes the committed FRONTIER.md and manifest, and FINDINGS.md's generator goes past its admission to its
    first computation. (`--audit` on the same record, with the known-good outputs, is
    tests/test_calibration_audit.py::test_audit_accepts_the_committed_record.)"""
    gate, fr, root = work
    for p in (gate.CALIBRATION_MD, gate.NOTE_PATH):
        (root / p).write_text("sentinel\n", encoding="utf-8")
    gate.main(["--render"])
    for p in (gate.CALIBRATION_MD, gate.NOTE_PATH):
        assert _same_text(root / p, _ROOT / p), p
    figures = []
    monkeypatch.setattr(fr, "_write_figure", figures.append)
    fr.main()
    for p in ("FRONTIER.md", "FRONTIER_MANIFEST.json"):
        assert _same_text(root / p, _ROOT / p), p
    assert len(figures) == 1
    with pytest.raises(_Reached):
        runpy.run_path(str(_ROOT / "scripts" / "generate_findings.py"), run_name="__main__")
    assert not (root / "FINDINGS.md").exists()


# ============================================================================ the measurement steps
def _genuine_run_files(gate, root):
    conv = _committed()["convergence"]
    paths = []
    for n in gate.CONVERGENCE_POSTERIORS:
        run = {"name": n, "res_pub": conv["res_pub"], "res_fine": conv["res_fine"],
               "pub": conv["runs"][n]["pub"], "fine": conv["runs"][n]["fine"]}
        p = root / f"run_{n}.json"
        p.write_text(json.dumps(run), encoding="utf-8")
        paths.append(str(p))
    return paths


@pytest.mark.parametrize("label, corrupt, named", _input_fixtures(), ids=[f[0] for f in _input_fixtures()])
def test_both_recorders_refuse_a_cells_file_whose_inputs_are_not_live(work, label, corrupt, named):
    """The convergence and scatter recorders rewrite the cells file; holding computed cells, it must carry the
    live input digests, or both refuse by name and write nothing."""
    gate, fr, root = work
    other = root / "other.json"
    shutil.copyfile(root / "CALIBRATION_CELLS.json", other)
    corrupt(root)
    before = (root / "CALIBRATION_CELLS.json").read_bytes()
    runs = _genuine_run_files(gate, root)
    for argv, refusal in ((["--record-convergence", *runs], "convergence record refused"),
                          (["--record-scatter", str(other)], "scatter record refused")):
        with pytest.raises(SystemExit) as exc:
            gate.main(argv)
        assert refusal in str(exc.value) and named in str(exc.value), str(exc.value)
        assert (root / "CALIBRATION_CELLS.json").read_bytes() == before


def test_the_convergence_recorder_states_its_two_stages(work):
    """Before the first regeneration the cells file holds no computed member, and the runs are recorded alone;
    a file holding some computed members but not all is refused naming the missing ones; the committed record
    (its convergence record removed) takes the genuine runs back exactly."""
    gate, fr, root = work
    committed = _committed()
    runs = _genuine_run_files(gate, root)
    cells_path = root / "CALIBRATION_CELLS.json"
    cells_path.write_text(json.dumps({"shipped_v1_3_0": committed["shipped_v1_3_0"]}), encoding="utf-8")
    gate.main(["--record-convergence", *runs])
    assert json.loads(cells_path.read_text(encoding="utf-8"))["convergence"] == committed["convergence"]
    partial = {k: committed[k] for k in ("shipped_v1_3_0", "posteriors")}
    cells_path.write_text(json.dumps(partial), encoding="utf-8")
    before = cells_path.read_bytes()
    with pytest.raises(SystemExit, match="convergence record refused:\n  cells: no 'schema'"):
        gate.main(["--record-convergence", *runs])
    assert cells_path.read_bytes() == before
    whole = {k: v for k, v in committed.items() if k != "convergence"}
    cells_path.write_text(json.dumps(whole), encoding="utf-8")
    gate.main(["--record-convergence", *runs])
    assert json.loads(cells_path.read_text(encoding="utf-8")) == committed


_PUSH_KEY = ("CALIBRATION.md|Posterior pushforward (state-of-knowledge X_mu and Q_net)|"
             "X_mu (weak-box posterior)|mean")


def _carried_shipped_defects():
    k_val = "VALIDATION_CHANNELS.md|V_breunlich_lambdac|predicted"
    k_new = "CALIBRATION.md|No such|weak|mean"
    return [
        ("a section the release does not have", lambda sh: sh.update({k_new: "1"}),
         f"shipped_v1_3_0[{k_new!r}]: not a cell of the released documents"),
        ("a print replaced", lambda sh: sh.update({_PUSH_KEY: "999"}),
         f"shipped_v1_3_0[{_PUSH_KEY!r}]: '999' is not the released print"),
        ("a print removed", lambda sh: sh.pop(k_val), f"shipped_v1_3_0: no {k_val!r}"),
    ]


@pytest.mark.parametrize("label, corrupt, named", _carried_shipped_defects(),
                         ids=[d[0] for d in _carried_shipped_defects()])
def test_regeneration_refuses_a_carried_shipped_record_before_it_samples(work, monkeypatch, label, corrupt,
                                                                         named):
    """The shipped record the regeneration carries forward is admitted against the released documents before
    the NUTS cross-check is sampled: a defect in it stops the regeneration, named, with the sampler never
    called and nothing written."""
    gate, fr, root = work
    committed = _committed()
    assert committed["shipped_v1_3_0"].get(_PUSH_KEY)
    cells = _committed()
    corrupt(cells["shipped_v1_3_0"])
    (root / "CALIBRATION_CELLS.json").write_text(json.dumps(cells), encoding="utf-8")
    before = _outputs(root, gate), (root / "CALIBRATION_CELLS.json").read_bytes()
    carried = ("shipped_v1_3_0", "convergence", "audit_scatter", "nuts_cross_check")
    fresh = {k: copy.deepcopy(v) for k, v in committed.items() if k not in carried}

    def posterior(name, res=None, **kw):
        return copy.deepcopy(committed["posteriors"][name])

    monkeypatch.setattr(gate, "_posterior", posterior)
    monkeypatch.setattr(gate, "compute_cells", lambda posts=None: copy.deepcopy(fresh))
    monkeypatch.setattr(gate, "nuts_cross_check", lambda p: pytest.fail("sampled"))
    monkeypatch.setattr(gate, "_write_figure", lambda draws: pytest.fail("figure drawn"))
    with pytest.raises(SystemExit) as exc:
        gate.regenerate()
    assert str(exc.value).startswith("regeneration refused: before sampling"), str(exc.value)
    assert named in str(exc.value), str(exc.value)
    assert (_outputs(root, gate), (root / "CALIBRATION_CELLS.json").read_bytes()) == before


@pytest.mark.parametrize("label, corrupt, named", _input_fixtures(), ids=[f[0] for f in _input_fixtures()])
def test_regeneration_records_the_live_input_digests(work, monkeypatch, label, corrupt, named):
    """The regeneration replaces the computed record, its digests included, from the live files: a cells file
    whose recorded inputs are not the live ones is rewritten with the live digests, never carried forward, and
    what it writes is what the gate admits. (The quadrature, the sweep, Q_net and the sampler return the
    genuine outputs here; the rest of the computed record, its digests included, is computed.)"""
    gate, fr, root = work
    committed = _committed()
    corrupt(root)

    def posterior(name, res=None, **kw):
        return copy.deepcopy(committed["posteriors"][name])

    monkeypatch.setattr(gate, "_posterior", posterior)
    monkeypatch.setattr(gate, "_sweep_rows", lambda: copy.deepcopy(committed["sweep"]))
    monkeypatch.setattr(gate, "q_net_cells", lambda weak, n=48: copy.deepcopy(committed["q_net"]))
    monkeypatch.setattr(gate, "nuts_cross_check",
                        lambda p: (copy.deepcopy(committed["nuts_cross_check"]), None))
    monkeypatch.setattr(gate, "_write_figure", lambda draws: None)
    gate.regenerate()
    written = gate.admit_calibration_cells()
    assert dict(written.data["inputs"]) == gate._inputs()
    assert written.sha256 == provenance.file_sha256(root / "CALIBRATION_CELLS.json")
