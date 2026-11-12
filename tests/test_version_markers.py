"""Every version marker of the repository carries the package's version.

The package's version is `openmucf.__version__` as this tree defines it (never the installed distribution's
metadata, which an editable install can map to another checkout), and `pyproject.toml` carries the same
`version`. The markers held to it: the `version` of `CITATION.cff` (exactly one such line), `version` and
`softwareVersion` of `codemeta.json`, the `version` of `.zenodo.json`, `.zenodo-data.json` and
`datapackage.json`, and the `#GENERATOR` line of every table under `data/g4/` -- the tables this repository's
generators write, which name the tool and the version that produced them (FORMAT_SPEC.md). The `.g4dat` files
under `tests/fixtures/` are not markers: their bytes are the test (a fixed `openmucf-g4 1.1.0` or
`openmucf-g4 conformance` stamp), so no release re-stamps them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import openmucf

REPO = Path(__file__).resolve().parents[1]

#: The files that carry a release version, each with how to read it.
JSON_MARKERS = (
    ("codemeta.json", "version"), ("codemeta.json", "softwareVersion"), (".zenodo.json", "version"),
    (".zenodo-data.json", "version"), ("datapackage.json", "version"),
)
GENERATOR = re.compile(r"^#GENERATOR[ \t]+(.*?)[ \t]*$", re.M)
STAMP = re.compile(r"openmucf-g4 (\S+)")


def generated_tables(root: Path) -> list[str]:
    """Every `.g4dat` under `data/g4/`, as a path relative to ``root``."""
    return sorted(p.relative_to(root).as_posix() for p in (root / "data" / "g4").rglob("*.g4dat"))


def version_markers(texts: dict[str, str]) -> tuple[list[tuple[str, str]], list[str]]:
    """The markers ``texts`` (path to text) carry, as (where, version) pairs, and every way a marker is
    missing, repeated or malformed. A marker that cannot be read is a problem, never a skipped marker."""
    markers: list[tuple[str, str]] = []
    problems: list[str] = []
    try:
        project = tomllib.loads(texts["pyproject.toml"])["project"]
        markers.append(("pyproject.toml version", project["version"]))
    except (KeyError, tomllib.TOMLDecodeError) as error:
        problems.append(f"pyproject.toml: no project version ({error})")
    cff = re.findall(r"^version:[ \t]*(\S*)[ \t]*$", texts.get("CITATION.cff", ""), re.M)
    if len(cff) != 1:
        problems.append(f"CITATION.cff: {len(cff)} `version:` lines, not one")
    markers.extend(("CITATION.cff version", v) for v in cff)
    for path, key in JSON_MARKERS:
        try:
            value = json.loads(texts[path])[key]
        except (KeyError, json.JSONDecodeError) as error:
            problems.append(f"{path}: no `{key}` ({error!r})")
            continue
        markers.append((f"{path} {key}", value))  # a value that is not the version's string is refused below
    tables = [path for path in texts if path.startswith("data/g4/") and path.endswith(".g4dat")]
    if not tables:
        problems.append("no generated table to read a #GENERATOR from")
    for path in sorted(tables):
        lines = GENERATOR.findall(texts[path])
        stamps = [STAMP.fullmatch(line) for line in lines]
        if len(lines) != 1 or stamps[0] is None:
            problems.append(f"{path}: #GENERATOR lines {lines!r}, not one `openmucf-g4 <version>`")
            continue
        markers.append((f"{path} #GENERATOR", stamps[0].group(1)))
    return markers, problems


def version_problems(texts: dict[str, str], version: str) -> list[str]:
    """Each marker of ``texts`` that is missing, malformed or other than ``version``."""
    markers, problems = version_markers(texts)
    problems.extend(f"{where}: {value!r}, not {version!r}" for where, value in markers if value != version)
    return problems


def _texts(root: Path = REPO) -> dict[str, str]:
    paths = ["pyproject.toml", "CITATION.cff", *dict.fromkeys(p for p, _ in JSON_MARKERS),
             *generated_tables(root)]
    return {p: (root / p).read_bytes().decode("utf-8").replace("\r\n", "\n") for p in paths}


def test_every_version_marker_carries_the_package_version():
    texts = _texts()
    assert version_problems(texts, openmucf.__version__) == []
    markers, _ = version_markers(texts)
    # Non-vacuity, counted from the tree: one marker per JSON key, one per table, and the two text markers.
    assert len(markers) == len(JSON_MARKERS) + len(generated_tables(REPO)) + 2
    assert generated_tables(REPO), "no generated table under data/g4"


def _first_table(t: dict[str, str]) -> str:
    return next(p for p in sorted(t) if p.endswith(".g4dat"))


def _last_table(t: dict[str, str]) -> str:
    return [p for p in sorted(t) if p.endswith(".g4dat")][-1]


def _json(path: str, key: str, value: object):
    def drill(t: dict[str, str]) -> dict[str, str]:
        record = json.loads(t[path])
        record[key] = value
        return dict(t, **{path: json.dumps(record)})
    return drill


def _drop_json(path: str, key: str):
    def drill(t: dict[str, str]) -> dict[str, str]:
        record = json.loads(t[path])
        del record[key]
        return dict(t, **{path: json.dumps(record)})
    return drill


def _cff(old: str, new: str):
    return lambda t: dict(t, **{"CITATION.cff": t["CITATION.cff"].replace(old, new, 1)})


def _table(old: str, new: str, which=_first_table):
    def drill(t: dict[str, str]) -> dict[str, str]:
        path = which(t)
        return dict(t, **{path: t[path].replace(old, new, 1)})
    return drill


V = openmucf.__version__
#: (name, how the drill alters the texts): each must be refused
DRILLS = [
    ("pyproject version differs", lambda t: dict(t, **{"pyproject.toml": t["pyproject.toml"].replace(
        f'version = "{V}"', 'version = "0.0.1"', 1)})),
    ("pyproject version missing", lambda t: dict(t, **{"pyproject.toml": t["pyproject.toml"].replace(
        f'version = "{V}"', "", 1)})),
    ("CITATION version differs", _cff(f"version: {V}", "version: 0.0.1")),
    ("CITATION version missing", _cff(f"version: {V}\n", "")),
    ("CITATION version repeated", _cff(f"version: {V}\n", f"version: {V}\nversion: {V}\n")),
    ("codemeta version differs", _json("codemeta.json", "version", "0.0.1")),
    ("codemeta softwareVersion differs", _json("codemeta.json", "softwareVersion", "0.0.1")),
    ("codemeta softwareVersion missing", _drop_json("codemeta.json", "softwareVersion")),
    ("zenodo version differs", _json(".zenodo.json", "version", "0.0.1")),
    ("zenodo version a number", _json(".zenodo.json", "version", 2.0)),
    ("zenodo-data version differs", _json(".zenodo-data.json", "version", "0.0.1")),
    ("datapackage version differs", _json("datapackage.json", "version", "0.0.1")),
    ("datapackage version missing", _drop_json("datapackage.json", "version")),
    ("a table's generator differs", _table(f"openmucf-g4 {V}", "openmucf-g4 0.0.1")),
    ("the last table's generator differs",
     _table(f"openmucf-g4 {V}", "openmucf-g4 0.0.1", _last_table)),
    ("a table's generator line missing", _table(f"#GENERATOR    openmucf-g4 {V}\n", "")),
    ("a table's generator line repeated",
     _table(f"#GENERATOR    openmucf-g4 {V}\n", f"#GENERATOR    openmucf-g4 {V}\n" * 2)),
    ("a table's generator names another tool", _table(f"openmucf-g4 {V}", f"other-tool {V}")),
    ("no generated table", lambda t: {p: s for p, s in t.items() if not p.endswith(".g4dat")}),
]


@pytest.mark.parametrize("name, drill", DRILLS, ids=[d[0] for d in DRILLS])
def test_version_drill_is_refused(name, drill):
    """Drill: each alteration of one marker changes the texts, and is refused."""
    t = _texts()
    drilled = drill(t)
    assert drilled != t, name
    assert version_problems(drilled, V), name


def test_the_measured_oxide_tables_take_the_package_version():
    """The `#GENERATOR` the measured oxide tables are written with is derived from the package's version,
    not typed: a package of another version stamps that version (in a fresh interpreter, so that no module
    is reloaded)."""
    code = ("import openmucf; openmucf.__version__ = '9.9.9'; from openmucf.g4 import d2_measured; "
            "print(d2_measured.GENERATOR)")
    path = os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH", "")) if p)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                            env={**os.environ, "PYTHONPATH": path})
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "openmucf-g4 9.9.9"


def test_the_version_is_read_from_this_tree():
    """The version the markers are held to is the one this tree's package defines."""
    init = (REPO / "openmucf" / "__init__.py").read_text(encoding="utf-8")
    assert re.search(rf'^__version__ = "{re.escape(openmucf.__version__)}"$', init, re.M)
    assert Path(openmucf.__file__).resolve().parent == REPO / "openmucf"
