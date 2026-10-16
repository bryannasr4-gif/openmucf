"""The calibration cells file has one admission, ``admit_calibration_cells`` in
scripts/generate_calibration.py, and every reader that renders, audits or publishes from the file takes only
what it admits.

The runtime ownership test and the consumer matrix hold that. An advisory static scan flags common direct
reads of the cells file; it is not exhaustive. The runtime ownership test checks that every shipped entry
point opens the file only through the gate. It launches each mode of the calibration script's command line,
its regeneration, and the FINDINGS.md and FRONTIER.md generators as their own processes, `python <script>
<arguments>`, on a working copy that carries the openmucf package. A recorder that Python imports at
interpreter start, before the package or the script, records by Python's `open` audit event every open of
the file for the whole life of each process and of any Python process it starts; the test accepts only the
gate's read and the measurement steps' own reads and writes, each as often as stated, and names any process
started and any fault of the recording's lifetime. The script's numerical engines return the committed
values in those runs from the moment the script calls its `main`, and run for real, at a reduced size, in a
process of their own.

The consumer matrix then runs the actual entry points -- `--render`, `--audit`, the FRONTIER.md generator's
`main` and the FINDINGS.md generator itself -- on a working copy whose cells file carries one defect at a
time: every print of the documents as released replaced, every one omitted, the record or the released
documents damaged, each input digest stale, missing or unknown, and each whole record absent. Each must
stop, naming the defect, before it writes or reports success.

The forms the advisory scan (`advisory_cells_file_scan`) flags are listed in its docstring, and its
planted-reader tests drill them.
"""

import ast
import copy
import functools
import importlib.util
import inspect
import json
import os
import runpy
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from openmucf import provenance, uq

#: the repository this test file is in (nothing is read from it when the module is collected)
_ROOT = Path(__file__).resolve().parents[1]
GATE = "scripts/generate_calibration.py"
CELLS = "CALIBRATION_CELLS"
#: the calibration script's module name, as an import, a module-name string or a file location gives it
SCRIPT = "generate_calibration"
#: the calls that load a module named by a string
_IMPORT_CALLS = frozenset({"import_module", "__import__"})
#: what a module outside the gate may take from the calibration script
GATE_API = frozenset({"admit_calibration_cells", "AdmittedCells", "CellsRefused"})
#: the functions of the calibration script that read the cells file: the gate, and the three measurement
#: steps that rewrite it (each admits what it carries or binds; see their docstrings)
GATE_READERS = frozenset({"admit_calibration_cells", "regenerate", "record_convergence", "record_scatter"})
_READ_CALLS = frozenset({"open", "read_text", "read_bytes", "load", "loads", "load_json", "loads_json",
                         "file_sha256"})


# ============================================================================ the advisory static scan
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


def _names_the_file(node):
    """A string or bytes constant naming the cells file."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, (str, bytes)):
        return False
    text = node.value if isinstance(node.value, str) else node.value.decode("latin-1")
    return CELLS in text


def _names_the_script(node):
    """An expression holding a string constant that names the calibration script."""
    return any(isinstance(c, ast.Constant) and isinstance(c.value, str) and SCRIPT in c.value
               for c in ast.walk(node))


def _dotted(node):
    """'a.b.c' for a name or a chain of attributes on a name, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else None


def _bindings(tree):
    """(names bound, value, statement) for each assignment, annotated assignment, assignment expression and
    import of a module, in source order."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            out.append(([t.id for t in n.targets if isinstance(t, ast.Name)], n.value, n))
        elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)) and n.value is not None:
            out.append(([n.target.id] if isinstance(n.target, ast.Name) else [], n.value, n))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.append(([], None, n))
    return sorted(out, key=lambda b: (b[2].lineno, b[2].col_offset))


def _gate_module_names(tree):
    """The names bound to the calibration script's module by each binding form the scan supports, and the ids
    of the string constants those bindings name the script with. The forms: `import
    [package.]generate_calibration [as m]` (the dotted name, or m); `from <package> import
    generate_calibration [as m]`, relative imports included; `m = importlib.import_module(...)` or `m =
    __import__(...)` with a string naming the script; and `m = importlib.util.module_from_spec(s)` where `s =
    importlib.util.spec_from_file_location(...)` names it, in source order (a later spec or module of
    another file under the same name drops it)."""
    specs, mods, named = set(), set(), set()
    for targets, value, node in _bindings(tree):
        if isinstance(node, ast.Import):
            mods |= {a.asname or a.name for a in node.names if a.name.split(".")[-1] == SCRIPT}
            continue
        if isinstance(node, ast.ImportFrom):
            mods |= {a.asname or a.name for a in node.names if a.name == SCRIPT}
            continue
        if not isinstance(value, ast.Call):
            continue
        name, gate = _call_name(value), _names_the_script(value)
        if name == "spec_from_file_location":
            for t in targets:
                (specs.add if gate else specs.discard)(t)
        elif name == "module_from_spec":
            arg = value.args[0] if value.args else None
            for t in targets:
                (mods.add if isinstance(arg, ast.Name) and arg.id in specs else mods.discard)(t)
        elif name in _IMPORT_CALLS and gate:
            mods |= set(targets)
        if gate and targets and name in {"spec_from_file_location", *_IMPORT_CALLS}:
            named |= {id(c) for c in ast.walk(value) if isinstance(c, ast.Constant)}
    return mods, named


def _module_uses(tree, mods):
    """(line, attribute) for each load of a name in ``mods``: the attribute taken from it, or None for any
    other use (passing the module on, binding it to another name, getattr on it); running it by `exec_module`
    is not a use."""
    parents = {id(c): p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, (ast.Name, ast.Attribute)) or not isinstance(n.ctx, ast.Load) \
                or _dotted(n) not in mods:
            continue
        p = parents.get(id(n))
        if isinstance(p, ast.Attribute) and p.value is n:
            out.append((n.lineno, p.attr))
        elif not (isinstance(p, ast.Call) and _call_name(p) == "exec_module" and any(a is n for a in p.args)):
            out.append((n.lineno, None))
    return out


def _path_definition(tree):
    """The constant of the calibration script's module-level `CELLS_JSON = "..."` statement, or None."""
    for top in tree.body:
        if isinstance(top, ast.Assign) and [getattr(t, "id", None) for t in top.targets] == ["CELLS_JSON"] \
                and isinstance(top.value, ast.Constant):
            return top.value
    return None


#: the calls that build a path, or a string, from a path they are given
_PATH_CALLS = frozenset({"Path", "PurePath", "str", "fspath", "join", "joinpath", "abspath", "realpath",
                         "normpath", "resolve", "absolute", "expanduser"})


def _is_path(node, paths):
    """An expression that is the cells file's path by a supported transform: a constant naming the file, a
    name in ``paths``, an f-string, a `+` or `/` with such an operand, or a path-building call (`_PATH_CALLS`)
    given one, as an argument or as the object it is called on."""
    if _names_the_file(node) or isinstance(node, ast.Name) and node.id in paths:
        return True
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) and _is_path(v.value, paths) for v in node.values)
    if isinstance(node, ast.BinOp):
        return _is_path(node.left, paths) or _is_path(node.right, paths)
    if isinstance(node, ast.Call) and _call_name(node) in _PATH_CALLS:
        on = node.func.value if isinstance(node.func, ast.Attribute) else None
        return any(_is_path(a, paths) for a in [*node.args, *(k.value for k in node.keywords)]) \
            or on is not None and _is_path(on, paths)
    return False


def _cells_path_names(tree, scope=None):
    """Inside the calibration script, the names that may hold the cells file's path in ``scope`` (a top-level
    statement; None for the module level alone) -- every name bound there or at the module level to an
    expression `_is_path` accepts, CELLS_JSON among them -- and the names of the raw loader there: `load` and
    every name bound to it or to such a name. Found to a fixed point; scopes nested in ``scope`` are not told
    apart from it."""
    paths, loaders = set(), {"load"}
    tops = [t for t in tree.body if not isinstance(t, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if scope is not None and scope not in tops:
        tops.append(scope)
    binds = [(targets, value) for top in tops for targets, value, _ in _bindings(top) if value is not None]
    changed = True
    while changed:
        changed = False
        for targets, value in binds:
            refers = _is_path(value, paths)
            alias = isinstance(value, ast.Name) and value.id in loaders
            for t in targets:
                for found, flag in ((paths, refers), (loaders, alias)):
                    if flag and t not in found:
                        found.add(t)
                        changed = True
    return paths, loaders


def _reads_the_cells_file(call, paths=frozenset({"CELLS_JSON"}), loaders=frozenset({"load"})):
    """A call that reads the cells file: the raw loader (``loaders``) with no argument (its default is the
    cells file), or a reading call or the loader whose function or arguments name the file, or use a name in
    ``paths``."""
    name = _call_name(call)
    if name in loaders and not call.args and not call.keywords:
        return True
    if name not in _READ_CALLS and name not in loaders:
        return False
    parts = [call.func, *call.args, *(k.value for k in call.keywords)]
    return any(_names_the_file(n) or isinstance(n, ast.Name) and n.id in paths
               for p in parts for n in ast.walk(p))


def _gate_findings(rel, tree):
    out = []
    for top in tree.body:
        is_def = isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        owner = top.name if is_def else "<module>"
        paths, loaders = _cells_path_names(tree, top)
        for n in ast.walk(top):
            if isinstance(n, ast.Call) and owner not in GATE_READERS:
                if _reads_the_cells_file(n, paths, loaders):
                    out.append(f"{rel}:{n.lineno}: {owner} reads the cells file outside the gate and the "
                               "measurement steps")
                elif _call_name(n) != "admit_calibration_cells" and any(
                        _is_path(a, paths) and not isinstance(a, ast.JoinedStr)
                        for a in [*n.args, *(k.value for k in n.keywords)]):
                    out.append(f"{rel}:{n.lineno}: {owner} passes the cells file's path to "
                               f"{_call_name(n)!r} outside the gate and the measurement steps")
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and n.name not in ("load", "admit_calibration_cells"):
                for d in [*n.args.defaults, *n.args.kw_defaults]:
                    if isinstance(d, ast.Name) and d.id == "CELLS_JSON":
                        out.append(f"{rel}:{n.lineno}: {n.name} defaults to the cells file")
    doc_nodes, _ = _docstrings(tree)
    definition = _path_definition(tree)
    out += [f"{rel}:{n.lineno}: names the cells file outside its CELLS_JSON definition"
            for n in ast.walk(tree) if _names_the_file(n) and id(n) not in doc_nodes and n is not definition]
    return out


def advisory_cells_file_scan(root):
    """What the advisory scan flags in the modules of openmucf/ and scripts/ under ``root``, as 'path:line:
    what'. An advisory static scan flags common direct reads of the cells file; it is not exhaustive. The
    runtime ownership test checks that every shipped entry point opens the file only through the gate.

    Outside the calibration script it flags: a string or bytes constant naming the file (docstrings aside);
    any attribute CELLS_JSON; for each name `_gate_module_names` binds to the script's module, every
    attribute taken from it but GATE_API and every other use of it (`_module_uses`); a string naming the
    script outside those bindings; an import of names from the script but GATE_API, and an `import`
    statement of it; and, by a text scan, any line naming the file that is neither in a docstring nor after a
    comment sign. Inside the script it flags: a read of the cells file (`_reads_the_cells_file`, with the path
    and loader names of `_cells_path_names`) in a function that is not one of GATE_READERS, or the file's
    path passed by such a function to any call but the gate; a function other than `load` and the gate whose
    argument defaults to the file; and a string or bytes constant naming the file other than the CELLS_JSON
    definition and docstrings."""
    out = set()
    for rel in _modules(root):
        src = (root / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        if rel == GATE:
            out |= set(_gate_findings(rel, tree))
            continue
        doc_nodes, doc_lines = _docstrings(tree)
        gate_mods, named = _gate_module_names(tree)
        for line, attr in _module_uses(tree, gate_mods):
            if attr is None:
                out.add(f"{rel}:{line}: uses the calibration script's module other than for its gate")
            elif attr not in GATE_API:
                out.add(f"{rel}:{line}: takes {attr!r} from the calibration script, not its gate")
        for n in ast.walk(tree):
            if _names_the_file(n) and id(n) not in doc_nodes:
                out.add(f"{rel}:{n.lineno}: names the cells file")
            elif isinstance(n, ast.Attribute) and n.attr == "CELLS_JSON":
                out.add(f"{rel}:{n.lineno}: uses the calibration script's CELLS_JSON")
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and SCRIPT in n.value \
                    and id(n) not in doc_nodes and id(n) not in named:
                out.add(f"{rel}:{n.lineno}: names the calibration script outside a supported binding")
            if isinstance(n, ast.ImportFrom) and n.module and SCRIPT in n.module:
                out |= {f"{rel}:{n.lineno}: imports {a.name!r} from the calibration script"
                        for a in n.names if a.name not in GATE_API}
            if isinstance(n, ast.Import) and any(SCRIPT in a.name for a in n.names):
                out.add(f"{rel}:{n.lineno}: imports the calibration script as a module")
        for i, line in enumerate(src.splitlines(), 1):
            at = line.find(CELLS)
            if at >= 0 and i not in doc_lines and not 0 <= line.find("#") < at:
                out.add(f"{rel}:{i}: names the cells file in code")
    return sorted(out)


def test_the_advisory_scan_flags_nothing_in_the_tree():
    assert advisory_cells_file_scan(_ROOT) == []


def test_the_scan_covers_every_module_and_finds_the_gate_and_its_readers():
    """The advisory scan is not vacuous: it walks every tracked module of openmucf/ and scripts/ (where git is
    present), the gate module is among them, FINDINGS.md's and FRONTIER.md's generators load the script under
    a name the scan recognizes and take only the gate from it, and inside the script the file's name is
    defined once, as CELLS_JSON, and exactly the three measurement steps read the file."""
    mods = _modules(_ROOT)
    assert GATE in mods and "openmucf/uq.py" in mods
    git = subprocess.run(["git", "-C", str(_ROOT), "ls-files", "openmucf", "scripts"],
                         capture_output=True, text=True)
    if git.returncode == 0 and git.stdout:
        assert {f for f in git.stdout.splitlines() if f.endswith(".py")} <= set(mods)
    for rel in ("scripts/generate_findings.py", "scripts/generate_frontier.py"):
        tree = ast.parse((_ROOT / rel).read_text(encoding="utf-8"))
        names, named = _gate_module_names(tree)
        assert names and named, rel
        assert {attr for _, attr in _module_uses(tree, names)} == {"admit_calibration_cells"}, rel
    gate_tree = ast.parse((_ROOT / GATE).read_text(encoding="utf-8"))
    assert _path_definition(gate_tree) is not None
    paths, loaders = _cells_path_names(gate_tree)
    assert "CELLS_JSON" in paths and "load" in loaders
    readers = {top.name for top in gate_tree.body if isinstance(top, ast.FunctionDef)
               for n in ast.walk(top)
               if isinstance(n, ast.Call) and _reads_the_cells_file(n, *_cells_path_names(gate_tree, top))}
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
    ("an import of the loader under another name", "scripts/planted.py",
     "from scripts.generate_calibration import load as raw\ncells = raw()\n",
     "scripts/planted.py:1: imports 'load' from the calibration script"),
    ("the script imported from its package under another name", "scripts/planted.py",
     "from scripts import generate_calibration as g\ncells = g.load()\n",
     "scripts/planted.py:2: takes 'load' from the calibration script, not its gate"),
    ("the script imported from its package", "scripts/planted.py",
     "from scripts import generate_calibration\ncells = generate_calibration.load()\n",
     "scripts/planted.py:2: takes 'load' from the calibration script, not its gate"),
    ("the script imported relatively", "scripts/planted.py",
     "from . import generate_calibration as g\ncells = g.load()\n",
     "scripts/planted.py:2: takes 'load' from the calibration script, not its gate"),
    ("the script imported by its dotted name", "scripts/planted.py",
     "import scripts.generate_calibration\ncells = scripts.generate_calibration.load()\n",
     "scripts/planted.py:2: takes 'load' from the calibration script, not its gate"),
    ("the script imported by its dotted name under another name", "scripts/planted.py",
     "import scripts.generate_calibration as g\ncells = g.load()\n",
     "scripts/planted.py:2: takes 'load' from the calibration script, not its gate"),
    ("the script imported by its own name", "scripts/planted.py",
     "import generate_calibration\ncells = generate_calibration.load()\n",
     "scripts/planted.py:2: takes 'load' from the calibration script, not its gate"),
    ("the script loaded by import_module", "scripts/planted.py",
     'import importlib\ng = importlib.import_module("scripts.generate_calibration")\ncells = g.load()\n',
     "scripts/planted.py:3: takes 'load' from the calibration script, not its gate"),
    ("the script loaded by __import__", "scripts/planted.py",
     'g = __import__("generate_calibration")\ncells = g.load()\n',
     "scripts/planted.py:2: takes 'load' from the calibration script, not its gate"),
    ("the script bound by an annotated assignment", "scripts/planted.py",
     'import importlib\ng: object = importlib.import_module("generate_calibration")\ncells = g.load()\n',
     "scripts/planted.py:3: takes 'load' from the calibration script, not its gate"),
    ("the script bound by an assignment expression", "scripts/planted.py",
     'if (g := __import__("generate_calibration")):\n    cells = g.load()\n',
     "scripts/planted.py:2: takes 'load' from the calibration script, not its gate"),
    ("the loader bound to another name", "scripts/planted.py", _LOAD_GATE + "raw = gen.load\ncells = raw()\n",
     "scripts/planted.py:6: takes 'load' from the calibration script, not its gate"),
    ("the script's module bound to another name", "scripts/planted.py",
     _LOAD_GATE + "other = gen\ncells = other.load()\n",
     "scripts/planted.py:6: uses the calibration script's module other than for its gate"),
    ("the loader taken by getattr", "scripts/planted.py", _LOAD_GATE + 'cells = getattr(gen, "load")()\n',
     "scripts/planted.py:6: uses the calibration script's module other than for its gate"),
    ("the script loaded without a binding", "scripts/planted.py",
     'import importlib\ncells = importlib.import_module("scripts.generate_calibration").load()\n',
     "scripts/planted.py:2: names the calibration script outside a supported binding"),
    ("the script run by its path", "scripts/planted.py",
     'import runpy\ncells = runpy.run_path("scripts/generate_calibration.py")["load"]()\n',
     "scripts/planted.py:2: names the calibration script outside a supported binding"),
    ("the script taken from the loaded modules", "scripts/planted.py",
     'import sys\ncells = sys.modules["generate_calibration"].load()\n',
     "scripts/planted.py:2: names the calibration script outside a supported binding"),
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
    ("a renderer reading the file by a literal name", "def _estimator_section(cells):\n",
     'def _estimator_section(cells):\n    cells = provenance.load_json("CALIBRATION_CELLS.json")\n',
     "_estimator_section reads the cells file outside the gate and the measurement steps"),
    ("a renderer reading the file through a copy of its path", "def _estimator_section(cells):\n",
     "def _estimator_section(cells):\n    p = CELLS_JSON\n    cells = load(p)\n",
     "_estimator_section reads the cells file outside the gate and the measurement steps"),
    ("a renderer reading the file through a path joined to it", "def _estimator_section(cells):\n",
     'def _estimator_section(cells):\n    p = Path(".") / CELLS_JSON\n'
     "    cells = json.loads(p.read_text())\n",
     "_estimator_section reads the cells file outside the gate and the measurement steps"),
    ("a renderer reading the file through a path object of it", "def _estimator_section(cells):\n",
     "def _estimator_section(cells):\n    p = Path(CELLS_JSON).resolve()\n    cells = load(p)\n",
     "_estimator_section reads the cells file outside the gate and the measurement steps"),
    ("a renderer reading the file through an f-string of it", "def _estimator_section(cells):\n",
     'def _estimator_section(cells):\n    p = f"{CELLS_JSON}"\n    cells = load(p)\n',
     "_estimator_section reads the cells file outside the gate and the measurement steps"),
    ("a renderer handing the path to a helper", "def _estimator_section(cells):\n",
     "def _estimator_section(cells):\n    cells = _peek(CELLS_JSON)\n",
     "_estimator_section passes the cells file's path to '_peek' outside the gate"),
    ("a renderer handing a copy of the path to a helper", "def _estimator_section(cells):\n",
     "def _estimator_section(cells):\n    p = CELLS_JSON\n    cells = _peek(path=p)\n",
     "_estimator_section passes the cells file's path to '_peek' outside the gate"),
    ("a renderer calling the raw loader under another name", "def _estimator_section(cells):\n",
     "def _estimator_section(cells):\n    raw = load\n    cells = raw()\n",
     "_estimator_section reads the cells file outside the gate and the measurement steps"),
    ("a second literal of the file's name", "def render(cells):\n",
     'OTHER = "CALIBRATION_CELLS.json"\n\n\ndef render(cells):\n',
     "names the cells file outside its CELLS_JSON definition"),
]
#: the two reader forms, verbatim, that an earlier version of the scan missed: the script's module bound by
#: a package import under another name in FRONTIER.md's generator, and a literal read in a function of the
#: script outside the gate; each is appended to the module it was planted in
_RETAINED_PLANTS = [
    ("a package import of the script under another name", "scripts/generate_frontier.py",
     "\nfrom scripts import generate_calibration as _raw_probe\ndef _unchecked_probe():\n"
     "    return _raw_probe.load()\n",
     "    return _raw_probe.load()", "takes 'load' from the calibration script, not its gate"),
    ("a literal read in a function of the script outside the gate", GATE,
     '\n\ndef _unchecked_probe():\n    return provenance.load_json("CALIBRATION_CELLS.json")\n',
     '    return provenance.load_json("CALIBRATION_CELLS.json")',
     "_unchecked_probe reads the cells file outside the gate and the measurement steps"),
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
    assert advisory_cells_file_scan(root) == []
    (root / rel).write_text(code, encoding="utf-8")
    assert named in advisory_cells_file_scan(root)


@pytest.mark.parametrize("label, old, new, named", _GATE_PLANTS, ids=[p[0] for p in _GATE_PLANTS])
def test_a_planted_read_inside_the_script_is_named(tmp_path, label, old, new, named):
    root = _tree(tmp_path)
    src = (root / GATE).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count(old) == 1, label
    (root / GATE).write_text(src.replace(old, new), encoding="utf-8")
    assert [f for f in advisory_cells_file_scan(root) if named in f], advisory_cells_file_scan(root)


_GATE_CLEAN = [
    ("the gate given the path", "def _estimator_section(cells):\n",
     "def _estimator_section(cells):\n    cells = admit_calibration_cells(CELLS_JSON).data\n"),
    ("a message naming the path", "def _estimator_section(cells):\n",
     'def _estimator_section(cells):\n    print(f"from {CELLS_JSON}")\n'),
]


@pytest.mark.parametrize("label, old, new", _GATE_CLEAN, ids=[p[0] for p in _GATE_CLEAN])
def test_a_use_of_the_path_inside_the_script_that_reads_nothing_is_clean(tmp_path, label, old, new):
    """Control: a function of the script that hands the path to the gate, or prints it, is not named."""
    root = _tree(tmp_path)
    src = (root / GATE).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count(old) == 1, label
    (root / GATE).write_text(src.replace(old, new), encoding="utf-8")
    assert advisory_cells_file_scan(root) == []


@pytest.mark.parametrize("label, rel, plant, at, named", _RETAINED_PLANTS,
                         ids=[p[0] for p in _RETAINED_PLANTS])
def test_a_retained_reader_appended_to_its_module_is_named(tmp_path, label, rel, plant, at, named):
    """Drill: each retained reader, appended verbatim to the module it was planted in, is named at its line;
    the module without it is clean."""
    root = _tree(tmp_path)
    src = (_ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
    (root / rel).write_text(src, encoding="utf-8")
    assert advisory_cells_file_scan(root) == []
    planted = src + plant
    (root / rel).write_text(planted, encoding="utf-8")
    line = planted.splitlines().index(at) + 1
    assert f"{rel}:{line}: {named}" in advisory_cells_file_scan(root), advisory_cells_file_scan(root)


_CLEAN_BINDINGS = [
    ("by file location", _LOAD_GATE + "cells = gen.admit_calibration_cells()\nrefused = gen.CellsRefused\n"),
    ("from its package",
     "from scripts import generate_calibration as g\ncells = g.admit_calibration_cells()\n"),
    ("by import_module", 'import importlib\ng = importlib.import_module("scripts.generate_calibration")\n'
                         "admitted = g.AdmittedCells\n"),
    ("its gate imported by name", "from scripts.generate_calibration import admit_calibration_cells\n"),
]


@pytest.mark.parametrize("label, code", _CLEAN_BINDINGS, ids=[c[0] for c in _CLEAN_BINDINGS])
def test_the_gate_taken_by_each_supported_binding_is_clean(tmp_path, label, code):
    """Control: a module that binds the script by a supported form and takes only GATE_API from it is
    clean."""
    root = _tree(tmp_path)
    (root / "scripts" / "clean.py").write_text(code, encoding="utf-8")
    assert advisory_cells_file_scan(root) == []


def test_mentions_that_read_nothing_are_not_named(tmp_path):
    """Control: a docstring or comment naming the file, an unrelated JSON read and the gate's own API are
    clean."""
    root = _tree(tmp_path)
    (root / "scripts" / "clean.py").write_text(
        '"""Reads CALIBRATION_CELLS.json only through the gate."""\n' + _LOAD_GATE
        + "import json\n\n\ndef f(p):\n    \"\"\"Not the CALIBRATION_CELLS.json file.\"\"\"\n"
        + "    return json.load(open(p))  # never CALIBRATION_CELLS.json\n\n\n"
        + "cells = gen.admit_calibration_cells()\nrefused = gen.CellsRefused\n", encoding="utf-8")
    assert advisory_cells_file_scan(root) == []


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


# ============================================================================ the runtime ownership test
#: the opens of the cells file that the measurement steps make themselves, each as (read or write, the
#: calibration script's function that opens it, the script's function that called that one): each step reads
#: the file raw, admits what it carries or binds (see their docstrings) and writes it
MEASUREMENT_OPENS = frozenset({
    ("read", "load", "regenerate"), ("write", "_write_text", "regenerate"),
    ("read", "load", "record_convergence"), ("write", "dump", "record_convergence"),
    ("read", "load", "record_scatter"), ("write", "dump", "record_scatter"),
})
_CELLS_FILE = CELLS + ".json"
#: the calibration script's numerical engines: in the entry-point runs they return the committed record's
#: values; `test_the_numerical_engines_open_no_cells_file` runs each for real, at a reduced size
ENGINES = ("_posterior", "_sweep_rows", "q_net_cells", "nuts_cross_check", "_write_figure")
#: the environment variables that configure the recorder in an entry point's process
_CONFIG_VAR, _ROOT_VAR = "OPENMUCF_CELLS_OPENS", "OPENMUCF_CELLS_OPENS_ROOT"


def _recorder():
    """The recorder, written as its own source and a call into ``sitecustomize.py`` in a directory first on
    the PYTHONPATH of an entry point's process. Python's `site` imports it at interpreter start, before the
    script, the openmucf package or anything they import runs, and every Python process started from there
    inherits it through that PYTHONPATH. It adds an audit hook that reports to the file the test names, one
    JSON line per event:

    - each open of a cells file: read or write, the qualified names of the calibration script's functions on
      the stack (innermost first) and the path. A cells file is any path whose name is CALIBRATION_CELLS.json,
      and any path that is the same file on disk as one of the configured cells files (a hard link, a short
      name). A function is the script's when its code is in one of the configured scripts;
    - each process started, and each error of the hook;
    - the start of the process, with any openmucf module already imported, and its normal end, with the file
      the openmucf package was imported from.

    Where the test asks, it makes the calibration script's ENGINES return the committed values from the moment
    the script, run as itself, calls its `main`, and reports that it did."""
    import atexit
    import json
    import os
    import sys

    cfg = json.loads(os.environ["OPENMUCF_CELLS_OPENS"])
    root = os.environ.pop("OPENMUCF_CELLS_OPENS_ROOT", "") == "1"     # the processes it starts are not
    fd = os.open(cfg["report"], os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0))

    def emit(**event):
        os.write(fd, (json.dumps({"pid": os.getpid(), "root": root, **event}) + "\n").encode())

    def real(path):
        return os.path.normcase(os.path.realpath(path))

    scripts = {real(s) for s in cfg["scripts"]}
    ids = {(st.st_dev, st.st_ino) for st in map(os.stat, cfg["cells"]) if st.st_ino}
    name = os.path.normcase("CALIBRATION_CELLS.json")
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND
    spawn_events = frozenset({"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
                              "os.fork", "os.forkpty", "os.startfile", "_winapi.CreateProcess"})
    in_script = {}
    busy = []

    def is_cells(path):
        if os.path.normcase(os.path.basename(path)) == name:
            return True
        try:
            st = os.stat(path)
        except (OSError, ValueError):
            return False
        return (st.st_dev, st.st_ino) in ids

    def record(path, mode, flags, frame):
        if path is None or isinstance(path, int):       # a descriptor: recorded when opened by its path
            return
        path = os.fsdecode(os.fspath(path))
        if not is_cells(path):
            return
        write = bool((flags or 0) & write_flags) or any(c in (mode or "") for c in "wax+")
        chain = []
        while frame is not None:
            f = frame.f_code.co_filename
            if f not in in_script:
                in_script[f] = real(f) in scripts
            if in_script[f]:
                chain.append(frame.f_code.co_qualname)
            frame = frame.f_back
        emit(event="open", kind="write" if write else "read", chain=chain, path=path)

    def hook(event, args):
        if busy or (event != "open" and event not in spawn_events):
            return
        busy.append(1)
        try:
            if event == "open":
                try:
                    frame = sys._getframe(1)
                except ValueError:      # opened by the interpreter itself (the script), no Python frame
                    frame = None
                record(*args[:3], frame)
            else:
                emit(event="spawn", what=f"{event} {args!r}"[:300])
        except Exception as e:      # raised here, it would fail the open it audits instead of the test
            emit(event="error", what=f"{event}: {type(e).__name__}: {e}")
        finally:
            busy.clear()

    if cfg.get("engines"):
        with open(cfg["engines"]["values"], encoding="utf-8") as f:
            values = f.read()
        gate = real(cfg["engines"]["script"])
        stand_ins = {
            "_posterior": lambda name, res=None, **kw: json.loads(values)["posteriors"][name],
            "_sweep_rows": lambda: json.loads(values)["sweep"],
            "q_net_cells": lambda weak, n=48: json.loads(values)["q_net"],
            "nuts_cross_check": lambda posteriors: (json.loads(values)["nuts_cross_check"], None),
            "_write_figure": lambda draws: None,
        }

        def profile(frame, event, arg):
            code = frame.f_code
            if event == "call" and code.co_name == "main" and frame.f_globals.get("__name__") == "__main__" \
                    and real(code.co_filename) == gate:
                sys.setprofile(None)
                missing = sorted(n for n in stand_ins if not callable(frame.f_globals.get(n)))
                frame.f_globals.update(stand_ins)
                emit(event="engines", names=sorted(stand_ins), missing=missing)

        sys.setprofile(profile)
    preloaded = sorted(m for m in sys.modules if m.split(".")[0] == "openmucf")
    sys.addaudithook(hook)
    emit(event="start", preloaded=preloaded)
    atexit.register(lambda: emit(event="end",
                                 openmucf=getattr(sys.modules.get("openmucf"), "__file__", None)))


def _engines_run():
    """Written as its own source and a call into a script of the test's, run as its own process in the working
    copy: it loads the working copy's calibration script by its path, runs each of ENGINES for real at a
    reduced size (the same code, on smaller grids and shorter chains) and prints, as JSON, what each
    returned against what the script says it must."""
    import importlib.util
    import json
    import sys

    spec = importlib.util.spec_from_file_location("_gen_calibration_engines",
                                                  "scripts/generate_calibration.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SWEEP_RES = 0.05
    gate.NUTS_WARMUP = 20
    gate.NUTS_SAMPLES = 20
    with open(sys.argv[1], encoding="utf-8") as f:
        posteriors = json.load(f)["posteriors"]
    ran = {"_posterior": {name: gate._posterior(name, res=0.05) for name in gate.POSTERIORS}}
    ran["_sweep_rows"] = gate._sweep_rows()
    ran["q_net_cells"] = gate.q_net_cells(ran["_posterior"]["weak"], n=4)
    ran["nuts_cross_check"] = gate.nuts_cross_check(posteriors)
    ran["_write_figure"] = gate._write_figure(ran["nuts_cross_check"][1])
    print(json.dumps({"ran": sorted(ran),
                      "sweep rows": [len(ran["_sweep_rows"]), len(gate._sweep_grid())],
                      "Q_net": [sorted(ran["q_net_cells"]), sorted(gate.Q_NET_STATS)],
                      "NUTS chains": [sorted(ran["nuts_cross_check"][0]), sorted(gate.NUTS_CHAINS)]}))


def _as_script(f, prefix=""):
    return prefix + inspect.getsource(f) + f"\n\n{f.__name__}()\n"


class _Report:
    """One entry point's recording, read from the recorder's report: the opens of a cells file as (read or
    write, the calibration script's functions on the stack, the path), the processes started, the errors of
    the hook, and what is wrong with the recording's lifetime: the entry point's process not recorded from its
    start, any openmucf module imported before the recorder, no normal end, the openmucf package not the
    working copy's, or the numerical engines asked to stand in and not standing in from `main` on."""

    def __init__(self, path, root, engines):
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] \
            if path.exists() else []
        self.events = events
        self.opens = [(e["kind"], tuple(e["chain"]), e["path"]) for e in events if e["event"] == "open"]
        self.spawns = [e["what"] for e in events if e["event"] == "spawn"]
        self.errors = [e["what"] for e in events if e["event"] == "error"]
        mine = [e for e in events if e["root"]]
        copy_root = os.path.normcase(os.path.realpath(root)) + os.sep
        lifetime = []
        starts = [e for e in mine if e["event"] == "start"]
        if len(starts) != 1:
            lifetime.append(f"the entry point's process was not recorded from its start "
                            f"({len(starts)} starts)")
        elif starts[0]["preloaded"]:
            lifetime.append(f"openmucf was imported before the recording started: {starts[0]['preloaded']}")
        ends = [e for e in mine if e["event"] == "end"]
        if len(ends) != 1:
            lifetime.append(f"the recording did not last until the process exited ({len(ends)} ends)")
        elif not os.path.normcase(os.path.realpath(ends[0]["openmucf"] or "")).startswith(copy_root):
            lifetime.append(f"openmucf was not imported from the working copy: {ends[0]['openmucf']}")
        stood = [e for e in mine if e["event"] == "engines"]
        if engines and [(e["names"], e["missing"]) for e in stood] != [(sorted(ENGINES), [])]:
            lifetime.append(f"the numerical engines did not stand in from `main` on: {stood}")
        self.lifetime = lifetime


def _owner(kind, chain):
    """'the gate' for a read whose innermost function of the calibration script is `admit_calibration_cells`;
    the step for a measurement step's own open (MEASUREMENT_OPENS); else None."""
    if kind == "read" and chain[:1] == ("admit_calibration_cells",):
        return "the gate"
    if len(chain) >= 2 and (kind, *chain[:2]) in MEASUREMENT_OPENS:
        return chain[1]
    return None


def _owned(rec):
    """The owned opens in ``rec``, counted as (read or write, the opening function, its caller in the
    script)."""
    return Counter((kind, *chain[:2]) for kind, chain, _ in rec.opens if _owner(kind, chain) is not None)


def ownership_problems(rec, expected):
    """Every open of a cells file in ``rec`` that is neither the gate's read nor a measurement step's own
    open, every process started, every error of the hook and every fault of the recording's lifetime; and
    the owned opens, counted, when they are not ``expected``."""
    probs = [f"{kind} of {path} outside the gate (the calibration script's functions on the stack: "
             f"{', '.join(chain) or 'none'})"
             for kind, chain, path in rec.opens if _owner(kind, chain) is None]
    probs += [f"started a process: {s}" for s in rec.spawns]
    probs += [f"the recording failed: {e}" for e in rec.errors]
    probs += rec.lifetime
    owned = _owned(rec)
    if owned != expected:
        probs.append(f"the owned opens {sorted(owned.items())} are not {sorted(expected.items())}")
    return probs


#: the scripts an entry point takes from the working copy (FINDINGS.md's generator loads the materiality
#: script by its path from the working directory)
_SCRIPTS = (GATE, "scripts/generate_findings.py", "scripts/generate_frontier.py",
            "scripts/generate_materiality.py")


class _EntryCopy:
    """A working copy of every file the entry points read, the openmucf package included, as the working
    directory; a copy of the cells file under its own name elsewhere, a hard link to it, the convergence runs
    and a copy of the cells file for the two recorders; and, apart from it, the recorder and the engines'
    committed values."""

    def __init__(self, root, aside):
        self.root, self.aside = root, aside
        self.cells = root / _CELLS_FILE
        self.watch = {"scripts": [str(root / GATE), str(_ROOT / GATE)],
                      "cells": [str(self.cells), str(_ROOT / _CELLS_FILE)]}
        self.runs = []
        self.launches = 0

    def prepare(self):
        """Write the files the recorders and the drills take, from the working copy's cells file as it is."""
        self.runs = [os.path.basename(p) for p in _genuine_run_files(_load(_ROOT / GATE, "_gen_calibration"),
                                                                     self.root)]
        shutil.copyfile(self.cells, self.root / "other_platform.json")
        (self.root / "stale").mkdir()
        shutil.copyfile(self.cells, self.root / "stale" / _CELLS_FILE)
        os.link(self.cells, self.root / "linked.json")
        committed = _committed()
        (self.aside / "engine_values.json").write_text(
            json.dumps({k: committed[k] for k in ("posteriors", "sweep", "q_net", "nuts_cross_check")}),
            encoding="utf-8")
        return self

    def launch(self, argv, *, engines=None, recorder_prefix="", package_from_copy=True):
        """Run ``python <argv>`` in the working copy as its own process, recorded from interpreter start by
        `_recorder`; with ``engines`` (by default, when ``argv`` runs the calibration script) the script's
        ENGINES stand in from its `main` on. Return the `_Report` and the completed process."""
        self.launches += 1
        shim = self.aside / f"launch_{self.launches}"
        shim.mkdir()
        (shim / "sitecustomize.py").write_text(_as_script(_recorder, recorder_prefix), encoding="utf-8")
        engines = argv[0] == GATE if engines is None else engines
        cfg = dict(self.watch, report=str(shim / "opens.jsonl"))
        if engines:
            cfg["engines"] = {"script": str(self.root / GATE),
                              "values": str(self.aside / "engine_values.json")}
        path = [str(shim), *([str(self.root)] if package_from_copy else [])]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(path), PYTHONIOENCODING="utf-8",
                   **{_CONFIG_VAR: json.dumps(cfg), _ROOT_VAR: "1"})
        done = subprocess.run([sys.executable, *argv], cwd=self.root, env=env, capture_output=True,
                              encoding="utf-8", errors="replace", timeout=1800)
        return _Report(shim / "opens.jsonl", self.root, engines), done


@pytest.fixture
def entry_copy(tmp_path):
    # matplotlib builds its font cache the first time it is imported without one, and on Linux runs `fc-list`
    # to do so: built here, outside the recordings, so that a launch starts no process of matplotlib's
    importlib.import_module("matplotlib.font_manager")
    root, aside = tmp_path / "copy", tmp_path / "aside"
    for rel in (*_FILES, *_SCRIPTS, _load(_ROOT / GATE, "_gen_calibration").NOTE_PATH):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / rel, root / rel)
    shutil.copytree(_ROOT / "openmucf", root / "openmucf", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    aside.mkdir()
    return _EntryCopy(root, aside)


def _calibration(*args):
    return lambda w: [GATE, *args]


_BY_GATE = ("read", "admit_calibration_cells")
#: the shipped entry points, each: (its command line, `python` and this, given the working copy; the owned
#: opens of the cells file it makes, counted); they are every mode of the calibration script and each module
#: that binds the script by a form the advisory scan recognizes
#: (`test_the_entry_points_are_every_mode_of_the_script_and_every_module_taking_its_gate`)
ENTRY_POINTS = {
    "generate_calibration.py --audit": (_calibration("--audit"), Counter({(*_BY_GATE, "audit"): 1})),
    "generate_calibration.py --render": (_calibration("--render"), Counter({(*_BY_GATE, "main"): 1})),
    "generate_calibration.py --cells-out": (_calibration("--cells-out", "computed_cells.json"), Counter()),
    "generate_calibration.py --converge": (_calibration("--converge", "weak", "--out", "run_new.json"),
                                           Counter()),
    "generate_calibration.py --record-convergence": (
        lambda w: [GATE, "--record-convergence", *w.runs],
        Counter({("read", "load", "record_convergence"): 1, ("write", "dump", "record_convergence"): 1})),
    "generate_calibration.py --record-scatter": (
        _calibration("--record-scatter", "other_platform.json"),
        Counter({("read", "load", "record_scatter"): 1, ("write", "dump", "record_scatter"): 1})),
    "generate_calibration.py (regeneration)": (
        _calibration(),
        Counter({("read", "load", "regenerate"): 1, ("write", "_write_text", "regenerate"): 1})),
    "generate_findings.py": (lambda w: ["scripts/generate_findings.py"], Counter({_BY_GATE: 1})),
    "generate_frontier.py": (lambda w: ["scripts/generate_frontier.py"], Counter({_BY_GATE: 1})),
}


def test_the_entry_points_are_every_mode_of_the_script_and_every_module_taking_its_gate(tmp_path):
    """The runtime test launches every mode the calibration script's `main` tests for in its arguments, its
    regeneration (no mode), and each module of openmucf/ and scripts/ that binds the script by a form the
    advisory scan recognizes, each as a script of the working copy run by `python`."""
    tree = ast.parse((_ROOT / GATE).read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    modes = {n.left.value for n in ast.walk(main) if isinstance(n, ast.Compare)
             and isinstance(n.left, ast.Constant) and isinstance(n.ops[0], ast.In)}
    assert len(modes) >= 6
    assert {f"generate_calibration.py {m}" for m in modes} | {"generate_calibration.py (regeneration)"} \
        == {e for e in ENTRY_POINTS if e.startswith("generate_calibration.py")}
    binders = {rel for rel in _modules(_ROOT) if rel != GATE
               and _gate_module_names(ast.parse((_ROOT / rel).read_text(encoding="utf-8")))[0]}
    assert binders == {f"scripts/{e}" for e in ENTRY_POINTS if not e.startswith("generate_calibration.py")}
    w = _EntryCopy(tmp_path, tmp_path)
    w.runs = ["run.json"]
    for entry, (argv, _) in ENTRY_POINTS.items():
        line = argv(w)
        assert line[0] in _SCRIPTS and line[0].endswith(entry.split()[0]), entry
        assert [a for a in line[1:] if a in modes] == [m for m in modes if m in entry.split()], entry


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
def test_every_entry_point_opens_the_cells_file_only_through_the_gate(entry_copy, entry):
    """Each entry point, launched on the genuine working copy and recorded from interpreter start, opens a
    cells file only by the gate's read and the measurement steps' own opens, exactly as often as
    `ENTRY_POINTS` states, starts no process, and exits 0."""
    entry_copy.prepare()
    argv, expected = ENTRY_POINTS[entry]
    rec, done = entry_copy.launch(argv(entry_copy))
    assert ownership_problems(rec, expected) == []
    assert done.returncode == 0, done.stderr[-3000:]


def _nonfinite_q_net(root):
    """The committed record but for Q_net's mean, which the gate refuses: not a finite number."""
    cells = _committed()
    cells["q_net"]["mean"] = float("nan")
    (root / _CELLS_FILE).write_text(json.dumps(cells), encoding="utf-8")


_NAN_REFUSED = "q_net.mean = nan is not a finite number"


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
def test_no_entry_point_takes_a_refused_record_but_through_its_refusal(entry_copy, entry):
    """With a cells file whose Q_net mean is not a finite number, each entry point, recorded from interpreter
    start, opens it only by the gate's read and the measurement steps' own opens, never more often than
    `ENTRY_POINTS` states, and starts no process. The regeneration, which computes Q_net afresh, writes the
    computed value in its place; every other entry point that opens the file stops, naming that scalar; one
    that does not open it exits 0. So no consumer, at start-up or after, takes the refused value
    unobserved."""
    _nonfinite_q_net(entry_copy.root)
    entry_copy.prepare()
    argv, expected = ENTRY_POINTS[entry]
    rec, done = entry_copy.launch(argv(entry_copy))
    owned = _owned(rec)
    assert ownership_problems(rec, owned) == []
    assert owned <= expected
    if entry == "generate_calibration.py (regeneration)":
        assert done.returncode == 0, done.stderr[-3000:]
        assert json.loads(entry_copy.cells.read_text(encoding="utf-8"))["q_net"] == _committed()["q_net"]
    elif expected:
        assert done.returncode != 0 and _NAN_REFUSED in done.stderr, (done.returncode, done.stderr[-3000:])
    else:
        assert not owned and done.returncode == 0, done.stderr[-3000:]


def test_the_numerical_engines_open_no_cells_file(entry_copy):
    """The ENGINES the entry-point runs stand in for run here for real, as their own process recorded from
    interpreter start, at a reduced size: each published posterior's quadrature, the prior-sensitivity sweep,
    Q_net, the NUTS cross-check and the figure. None opens a cells file or starts a process, and each returns
    what the calibration script says it must."""
    entry_copy.prepare()
    (entry_copy.aside / "engines_run.py").write_text(_as_script(_engines_run), encoding="utf-8")
    rec, done = entry_copy.launch([str(entry_copy.aside / "engines_run.py"),
                                   str(entry_copy.aside / "engine_values.json")], engines=False)
    assert ownership_problems(rec, Counter()) == []
    assert done.returncode == 0, done.stderr[-3000:]
    ran = json.loads(done.stdout.strip().splitlines()[-1])
    assert ran.pop("ran") == sorted(ENGINES)
    for what, (got, due) in ran.items():
        assert got == due and got, what
    assert (entry_copy.root / "figures" / "calibration.png").stat().st_size > 0


_FRONTIER = "scripts/generate_frontier.py"
_FRONTIER_AT = "    cells = admitted_calibration_cells()      # before anything is written\n"
_RENDER_AT = "def _estimator_section(cells):\n"
_LAUNCH_AT = 'if __name__ == "__main__":\n    main()\n'
_UQ = "openmucf/uq.py"
_UQ_AT = "import csv\n"
#: (label, file of the working copy, text, the text that replaces it, entry point, what the problem names,
#: what the planted reader prints on a record the gate refuses): two plant a reader that runs before the
#: calibration script's `main`, one when the script is launched, one when the openmucf package is initialized
_STARTUP_PLANTS = [
    ("the raw loader under another name at launch", GATE, _LAUNCH_AT,
     'if __name__ == "__main__":\n    reader, = (load,)\n    unchecked = reader()\n'
     '    print("unchecked Q_net mean:", unchecked["q_net"]["mean"])\n    main()\n',
     "generate_calibration.py --render",
     "read of CALIBRATION_CELLS.json outside the gate (the calibration script's functions on the stack: "
     "load, <module>)", "unchecked Q_net mean: nan"),
    ("a raw read when the package is initialized", _UQ, _UQ_AT,
     _UQ_AT + 'import json as _cells_json\n'
     'with open("_".join(("CALIBRATION", "CELLS")) + ".json") as _cells_fp:\n'
     '    _unchecked_qnet = _cells_json.load(_cells_fp)["q_net"]["mean"]\n'
     '    print("unchecked Q_net at import:", _unchecked_qnet)\n',
     "generate_calibration.py --render",
     "read of CALIBRATION_CELLS.json outside the gate (the calibration script's functions on the stack: "
     "<module>)", "unchecked Q_net at import: nan"),
]
#: (label, file of the working copy, text, the text that replaces it, entry point, what the problem names):
#: each plants an open of the cells file outside the gate, or a fault of the recording, in the working copy;
#: the advisory scan misses several
_OWNERSHIP_PLANTS = [p[:6] for p in _STARTUP_PLANTS] + [
    ("plain open() in FRONTIER.md's generator", _FRONTIER, _FRONTIER_AT,
     _FRONTIER_AT + '    _raw = open("CALIBRATION_CELLS.json").read()\n', "generate_frontier.py",
     "read of CALIBRATION_CELLS.json outside the gate (the calibration script's functions on the stack: "
     "none)"),
    ("Path.read_text through an f-string in the render", GATE, _RENDER_AT,
     _RENDER_AT + '    _raw = Path(f"{CELLS_JSON}").read_text()\n', "generate_calibration.py --render",
     "read of CALIBRATION_CELLS.json outside the gate (the calibration script's functions on the stack: "
     "_estimator_section, "),
    ("the raw loader under another name in the render", GATE, _RENDER_AT,
     _RENDER_AT + "    reader = load\n    _raw = reader()\n", "generate_calibration.py --render",
     "functions on the stack: load, _estimator_section, "),
    ("the raw loader under another name in FRONTIER.md's generator", _FRONTIER,
     "    return gen.admit_calibration_cells()\n",
     "    raw = gen.load\n    raw()\n    return gen.admit_calibration_cells()\n", "generate_frontier.py",
     "functions on the stack: load)"),
    ("a partial of the raw loader", GATE, _RENDER_AT,
     _RENDER_AT + '    _raw = __import__("functools").partial(load)()\n', "generate_calibration.py --render",
     "functions on the stack: load, _estimator_section, "),
    ("the raw loader's __call__", GATE, _RENDER_AT, _RENDER_AT + "    _raw = load.__call__()\n",
     "generate_calibration.py --render", "functions on the stack: load, _estimator_section, "),
    ("the path handed to a reading helper as an f-string", GATE, _RENDER_AT,
     _RENDER_AT + '    _raw = provenance.load_json(f"{CELLS_JSON}")\n', "generate_calibration.py --render",
     "functions on the stack: _estimator_section, "),
    ("the raw loader as a default argument", GATE, "def render(cells):\n",
     "def _probe(reader=load):\n    return reader()\n\n\ndef render(cells):\n    _probe()\n",
     "generate_calibration.py --render", "functions on the stack: load, _probe, render, main, <module>)"),
    ("a raw read inside the gate's own checks", GATE, "def admission_problems(cells, *, sampled=True):\n",
     "def admission_problems(cells, *, sampled=True):\n    _raw = load()\n",
     "generate_calibration.py --render",
     "functions on the stack: load, admission_problems, _admit_bytes, admit_calibration_cells, main, "
     "<module>)"),
    ("os.open in the audit", GATE, "def audit():\n",
     'def audit():\n    __import__("os").close(__import__("os").open(CELLS_JSON, 0))\n',
     "generate_calibration.py --audit", "functions on the stack: audit, main, <module>)"),
    ("a hard link to the file", _FRONTIER, _FRONTIER_AT,
     _FRONTIER_AT + '    _raw = open("linked.json").read()\n',
     "generate_frontier.py", "read of linked.json outside the gate"),
    ("a copy of the file under its own name elsewhere", _FRONTIER, _FRONTIER_AT,
     _FRONTIER_AT + '    _raw = open("stale/CALIBRATION_CELLS.json").read()\n', "generate_frontier.py",
     "of stale/CALIBRATION_CELLS.json outside the gate"),
    ("a write outside the measurement steps", GATE, "def render(cells):\n",
     'def render(cells):\n    dump({}, "CALIBRATION_CELLS.json")\n', "generate_calibration.py --render",
     "write of CALIBRATION_CELLS.json outside the gate (the calibration script's functions on the stack: "
     "dump, render, main, <module>)"),
    ("a second raw read in a measurement step", GATE, "    cells, other = load(), load(path)\n",
     "    cells, other = load(), load(path)\n    _again = load()\n",
     "generate_calibration.py --record-scatter",
     "the owned opens [(('read', 'load', 'record_scatter'), 2), "),
    ("a process that reads the file", _FRONTIER, _FRONTIER_AT,
     _FRONTIER_AT + '    __import__("subprocess").run([__import__("sys").executable, "-c", '
                    '"open(\'CALIBRATION_CELLS.json\').read()"], check=True)\n',
     "generate_frontier.py", "started a process: subprocess.Popen"),
    ("the file read by a process the entry point starts", _FRONTIER, _FRONTIER_AT,
     _FRONTIER_AT + '    __import__("subprocess").run([__import__("sys").executable, "-c", '
                    '"open(\'CALIBRATION_CELLS.json\').read()"], check=True)\n',
     "generate_frontier.py",
     "read of CALIBRATION_CELLS.json outside the gate (the calibration script's functions on the stack: "
     "none)"),
    ("an open the recorder cannot read", _FRONTIER, _FRONTIER_AT,
     _FRONTIER_AT + '    __import__("sys").audit("open", 0.5, "r", 0)\n', "generate_frontier.py",
     "the recording failed: open: TypeError"),
    ("the process ending without its exit", _FRONTIER, _FRONTIER_AT,
     _FRONTIER_AT + '    __import__("os")._exit(0)\n', "generate_frontier.py",
     "the recording did not last until the process exited (0 ends)"),
]


def _plant(w, rel, old, new, label):
    path = w.root / rel
    src = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count(old) == 1, label
    path.write_text(src.replace(old, new), encoding="utf-8")


@pytest.mark.parametrize("label, rel, old, new, entry, named", _OWNERSHIP_PLANTS,
                         ids=[p[0] for p in _OWNERSHIP_PLANTS])
def test_a_planted_open_of_the_cells_file_is_named_at_run_time(entry_copy, label, rel, old, new, entry,
                                                               named):
    """Drill: each planted open of a cells file outside the gate, or fault of the recording, launched through
    its entry point, is named by the same check the entry points pass."""
    _plant(entry_copy, rel, old, new, label)
    entry_copy.prepare()
    argv, expected = ENTRY_POINTS[entry]
    rec, _ = entry_copy.launch(argv(entry_copy))
    probs = ownership_problems(rec, expected)
    assert [p for p in probs if named in p], probs


@pytest.mark.parametrize("label, rel, old, new, entry, named, printed", _STARTUP_PLANTS,
                         ids=[p[0] for p in _STARTUP_PLANTS])
def test_a_start_up_reader_of_a_refused_record_is_named(entry_copy, label, rel, old, new, entry, named,
                                                        printed):
    """Drill: a reader planted at launch or at the package's initialization, on a record the gate refuses,
    receives and prints the refused value before the gate refuses it, and the check names its read."""
    _plant(entry_copy, rel, old, new, label)
    _nonfinite_q_net(entry_copy.root)
    entry_copy.prepare()
    argv, expected = ENTRY_POINTS[entry]
    rec, done = entry_copy.launch(argv(entry_copy))
    assert printed in done.stdout and _NAN_REFUSED in done.stderr and done.returncode != 0, done
    probs = ownership_problems(rec, expected)
    assert [p for p in probs if named in p], probs


#: (label, how the launch departs from the entry points', what the problem names): each fault of the
#: recording's lifetime is named
_LIFETIME_FAULTS = [
    ("the package imported before the recorder", {"recorder_prefix": "import openmucf\n"},
     "openmucf was imported before the recording started: ['openmucf'"),
    ("the package not the working copy's", {"package_from_copy": False},
     "openmucf was not imported from the working copy"),
    ("the engines asked for and not standing in", {"engines": True, "argv": ["scripts/generate_frontier.py"]},
     "the numerical engines did not stand in from `main` on: []"),
]


@pytest.mark.parametrize("label, how, named", _LIFETIME_FAULTS, ids=[f[0] for f in _LIFETIME_FAULTS])
def test_a_fault_of_the_recordings_lifetime_is_named(entry_copy, label, how, named):
    """Drill: a launch whose recording starts after the openmucf package was imported, that imports another
    openmucf than the working copy's, or whose engines were asked to stand in and do not, is named."""
    entry_copy.prepare()
    how = dict(how)
    rec, _ = entry_copy.launch(how.pop("argv", [GATE, "--render"]), **how)
    probs = ownership_problems(rec, ENTRY_POINTS["generate_calibration.py --render"][1])
    assert [p for p in probs if named in p], probs
