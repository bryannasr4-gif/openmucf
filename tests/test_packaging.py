"""Packaging + lazy public API (PEP 562) guards.

The heavy public submodules load lazily on first attribute access, so a bare `import openmucf`
never pays the numpyro/statistics import cost. These tests assert the marker file is packaged,
the lazy names resolve, and importing the package does not eager-load the heavy stack.

2026-08-12 amendment: also guards the DEPENDENCY DECLARATION itself. numpy and Pillow were imported
by shipped code while arriving only transitively; the last test below makes that class of omission a
test failure instead of a latent packaging bug, so it cannot recur silently as more code lands."""

import ast
import json
import re
import subprocess
import sys
import tomllib
import urllib.parse
from pathlib import Path

import pytest

import openmucf

REPO = Path(__file__).resolve().parents[1]

# Directories whose imports must be covered by a declaration (runtime deps or an extra).
IMPORT_SCAN_DIRS = ("openmucf", "scripts", "tests", "examples")

# Import name -> distribution name, for the cases where they differ. Kept explicit and static rather
# than read from the installed environment: the `locked` CI job installs a lockfile that does not
# contain every declared distribution, so an environment-derived map would make this test env-dependent.
IMPORT_TO_DISTRIBUTION = {"PIL": "pillow"}

# `conftest` is a repo-local module under tests/, not a distribution -- pytest makes it importable by
# name, so a test that exercises the hooks in it reads to the static scan below like a third-party
# import that nothing declares. Listing it here fixes that misclassification; it exempts no actual
# distribution, because there is no package named `conftest` in any dependency table.
# `test_g4parity` and `test_g4d3` are the same case: tests/test_g4prose.py imports each by name to
# reuse its pin tables.
FIRST_PARTY = {"openmucf", "scripts", "conftest", "test_g4parity", "test_g4d3"}

LAZY = (
    "calibrate",
    "validate",
    "forecast",
    "systems",
    "mucost",
    "frontier",
    "twin",
    "likelihood",
    "bench",
    "design",
)


def test_py_typed_marker_present_and_declared():
    assert (REPO / "openmucf" / "py.typed").is_file()
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"py.typed"' in pyproject, "py.typed not declared in [tool.setuptools.package-data]"


def test_all_exports_include_lazy_submodules():
    for name in LAZY:
        assert name in openmucf.__all__, f"{name} missing from __all__"


def test_lazy_getattr_resolves_each_submodule():
    for name in LAZY:
        module = getattr(openmucf, name)
        assert module.__name__ == f"openmucf.{name}"


def test_unknown_attribute_still_raises_attribute_error():
    try:
        openmucf.does_not_exist  # noqa: B018
    except AttributeError:
        return
    raise AssertionError("expected AttributeError for an unknown attribute")


def test_bare_import_does_not_eager_load_heavy_stack():
    """Deterministic laziness guard: a fresh `import openmucf` must NOT pull numpyro or any lazy
    submodule into sys.modules; access triggers the load."""
    code = (
        "import sys, openmucf\n"
        "print(int('numpyro' in sys.modules))\n"
        f"print(int(any(f'openmucf.{{n}}' in sys.modules for n in {LAZY!r})))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["0", "0"], f"heavy stack eager-loaded: {out.stdout!r}"


def test_import_walltime_within_2x_eager_spine():
    """Wall-time guard against an accidental eager heavy import: a bare `import openmucf` must stay
    within 2x the time to import its eager dependency spine (jax + diffrax). If numpyro (or another
    heavy dep pulled only by the lazy submodules) were eager-imported, this ratio would blow past 2x."""

    def _min_time(imports, n=3):
        code = f"import time; _t=time.perf_counter(); import {imports}; print(time.perf_counter()-_t)"
        times = []
        for _ in range(n):
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
            times.append(float(out.stdout.strip().splitlines()[-1]))
        return min(times)

    baseline = _min_time("jax, diffrax")
    package = _min_time("openmucf")
    assert package < 2.0 * baseline, (
        f"import openmucf ({package:.3f}s) exceeds 2x the eager-spine baseline ({baseline:.3f}s) "
        "-- something heavy is being eager-imported"
    )


def _normalize(name: str) -> str:
    """PEP 503 distribution-name normalization ('SALib' and 'salib' are the same project)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_distributions() -> set[str]:
    """Every distribution pyproject.toml declares, runtime or extra, normalized."""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    declared = set()
    for req in requirements:
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
        assert match, f"unparseable requirement {req!r} in pyproject.toml"
        declared.add(_normalize(match.group(1)))
    return declared


def _third_party_imports() -> dict[str, list[str]]:
    """Top-level third-party import name -> the repo-relative files that import it.

    Static (ast) on purpose: it sees imports inside functions and inside `if` branches, and it does
    not require the imported package to be installed in the environment running the test.
    """
    stdlib = set(sys.stdlib_module_names)
    imports: dict[str, list[str]] = {}
    for directory in IMPORT_SCAN_DIRS:
        for path in sorted((REPO / directory).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.add(node.module.split(".")[0])
            for name in names:
                if name in stdlib or name in FIRST_PARTY:
                    continue
                imports.setdefault(name, []).append(path.relative_to(REPO).as_posix())
    return imports


def test_every_third_party_import_is_a_declared_dependency():
    """No scanned module may import a distribution nothing declares.

    Guards the omission class found on 2026-08-12: numpy (openmucf/) and Pillow (scripts/) were both
    imported by shipped code while arriving only as transitive installs of SALib/matplotlib, so a
    resolver change could have broken the package with no declaration to point at.

    Two limits, stated so nobody reads more into a pass than is there:
      * It checks that an import is declared SOMEWHERE -- runtime table or any extra -- never that it
        is declared in the RIGHT one. Moving scipy to [project.dependencies] would not fail this test;
        that placement judgement is made in pyproject.toml's comments, not enforced here.
      * It is static, so it sees only real import statements. importlib.import_module(name) and
        __import__ with a computed name are invisible to it, as are notebooks.
    """
    declared = _declared_distributions()
    undeclared = {
        name: files
        for name, files in sorted(_third_party_imports().items())
        if _normalize(IMPORT_TO_DISTRIBUTION.get(name, name)) not in declared
    }
    assert not undeclared, (
        "imports with no declared distribution in pyproject.toml (add the dependency, or map the "
        f"import name in IMPORT_TO_DISTRIBUTION): {undeclared}"
    )


def test_the_distribution_licence_metadata_names_what_it_packages():
    """The distribution declares `Apache-2.0 AND CC-BY-4.0` and ships both licence files: the package's
    code is Apache-2.0 (`LICENSE`) and the files under `openmucf/data` it packages are CC-BY-4.0
    (`LICENSE-DATA`). The configuration that decides what it packages is held here as well -- an
    explicit package list under `openmucf`, package data declared for `openmucf` alone, no
    `MANIFEST.in` -- because under it a build packages nothing from `cpp/` or `third_party/`, the
    paths that carry the Geant4 Software License, so the expression needs no term for that licence.
    The build floor is held too: setuptools 76.1.0 refuses `license` as a string, and 77.0.1 builds
    this file."""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["license"] == "Apache-2.0 AND CC-BY-4.0"
    assert project["license-files"] == ["LICENSE", "LICENSE-DATA"]
    assert all((REPO / name).is_file() for name in project["license-files"])
    setuptools = pyproject["tool"]["setuptools"]
    assert all(name == "openmucf" or name.startswith("openmucf.") for name in setuptools["packages"])
    assert set(setuptools["package-data"]) == {"openmucf"}
    assert "data/*" in setuptools["package-data"]["openmucf"]
    assert not (REPO / "MANIFEST.in").exists()
    assert pyproject["build-system"]["requires"] == ["setuptools>=77"]


#: The licence names a per-path statement may use.
LICENCES = ("Apache-2.0", "CC-BY-4.0", "Geant4 Software License")
#: Each path token of the per-path statements, and the one licence a clause naming it must name.
LICENCE_BY_PATH = {
    "(LICENSE)": "Apache-2.0",
    "(LICENSE-DATA)": "CC-BY-4.0",
    "openmucf/data": "CC-BY-4.0",
    "cpp/patches/": "Geant4 Software License",
    "third_party/geant4/": "Geant4 Software License",
}


def check_licences_by_path(text: str, paths: tuple[str, ...]) -> None:
    """Every path in `paths` occurs in `text`, and every clause that names a path of
    :data:`LICENCE_BY_PATH` names exactly one licence of :data:`LICENCES`: the one that path carries.
    A clause is the text between semicolons, colons and sentence ends, whitespace collapsed; a clause
    naming a path and no known licence (an unknown one, say) fails as well as a clause naming the wrong
    one."""
    flat = " ".join(text.split())
    assert all(path in flat for path in paths), [path for path in paths if path not in flat]
    for clause in re.split(r"[;:]|\.\s", flat):
        named = [licence for licence in LICENCES if licence in clause]
        for path, licence in LICENCE_BY_PATH.items():
            if path in clause:
                assert named == [licence], f"{path!r} is given {named}, not [{licence!r}]: {clause!r}"


def check_licence_badge(readme: str) -> None:
    """The README's licence badge links to its License section and lists exactly the licences of
    :data:`LICENCES`, read back from the badge's own URL (shields.io writes `--` for `-` and `%xx`
    escapes)."""
    badge = r"\[!\[Licenses by path\]\((https://img\.shields\.io/badge/[^)]+)\)\]\(#license\)"
    match = re.search(badge, readme)
    assert match, "no `Licenses by path` badge linking to #license"
    assert re.search(r"^## License$", readme, re.MULTILINE), "no License section for the badge to link to"
    segment = match.group(1).split("/badge/", 1)[1].removesuffix(".svg")
    _label, message, _colour = re.split(r"(?<!-)-(?!-)", segment)
    listed = [urllib.parse.unquote(part).replace("--", "-") for part in message.split("%20%C2%B7%20")]
    assert listed == list(LICENCES), listed


def citation_comment(citation: str) -> str:
    """The comment lines of `CITATION.cff`, without their `#`, as one text with whitespace collapsed."""
    lines = [line.lstrip().lstrip("#") for line in citation.splitlines() if line.lstrip().startswith("#")]
    return " ".join(" ".join(lines).split())


def test_the_citation_and_archive_metadata_state_the_licences_by_path():
    """The repository's files carry licences by path, and each metadata file says so in the form its
    format allows. `codemeta.json` lists the three licence documents. `.zenodo.json` takes one licence
    id, which Zenodo applies to every file of the record, so it carries `other-open` and its notes give
    each path its licence. CFF reads a list of `license` ids as alternatives, so `CITATION.cff` sets no
    `license` key at any level and its comment gives each path its licence. The README badge lists the
    licences and links to the section that names their paths."""
    codemeta = json.loads((REPO / "codemeta.json").read_text(encoding="utf-8"))
    assert codemeta["license"] == [
        "https://spdx.org/licenses/Apache-2.0",
        "https://spdx.org/licenses/CC-BY-4.0",
        "http://cern.ch/geant4/license",
    ]
    zenodo = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    assert zenodo["license"] == "other-open"
    zenodo_paths = ("(LICENSE)", "(LICENSE-DATA)", "cpp/patches/", "third_party/geant4/")
    check_licences_by_path(zenodo["notes"], zenodo_paths)
    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert not re.search(r"^\s*license\s*:", citation, re.MULTILINE)
    check_licences_by_path(citation_comment(citation), tuple(LICENCE_BY_PATH))
    check_licence_badge((REPO / "README.md").read_text(encoding="utf-8"))


def test_drill_a_wrong_or_swapped_licence_by_path_is_refused():
    """The per-path and badge guards, shown to fire on in-memory corruptions of the shipped texts: an
    unknown licence put in place of one, two licences swapped between paths, the patches given the code's
    licence, a licence dropped from the badge, and the badge's link moved off the License section."""
    notes = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))["notes"]
    comment = citation_comment((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    swapped = notes.replace("Apache-2.0 for code", "CC-BY-4.0 for code")
    swapped = swapped.replace("CC-BY-4.0 for data", "Apache-2.0 for data")
    corruptions = [
        (notes, "Apache-2.0 for code", "MIT for code"),
        (notes, notes, swapped),
        (comment, "The code is Apache-2.0", "The code is MIT"),
        (comment, "Geant4 Software License (cpp/patches/LICENSE)", "Apache-2.0 (cpp/patches/LICENSE)"),
    ]
    for text, old, new in corruptions:
        assert text.count(old) == 1, old
        with pytest.raises(AssertionError):
            check_licences_by_path(text.replace(old, new), ())
    for old, new in (("%20%C2%B7%20CC--BY--4.0", ""), ("](#license)", "](#top)")):
        assert readme.count(old) == 1, old
        with pytest.raises(AssertionError):
            check_licence_badge(readme.replace(old, new))
