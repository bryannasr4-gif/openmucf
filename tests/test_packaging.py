"""Packaging + lazy public API (PEP 562) guards.

The heavy public submodules load lazily on first attribute access, so a bare `import openmucf`
never pays the numpyro/statistics import cost. These tests assert the marker file is packaged,
the lazy names resolve, and importing the package does not eager-load the heavy stack.

2026-08-12 amendment: also guards the DEPENDENCY DECLARATION itself. numpy and Pillow were imported
by shipped code while arriving only transitively; the last test below makes that class of omission a
test failure instead of a latent packaging bug, so it cannot recur silently as more code lands."""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import openmucf

REPO = Path(__file__).resolve().parents[1]

# Directories whose imports must be covered by a declaration (runtime deps or an extra).
IMPORT_SCAN_DIRS = ("openmucf", "scripts", "tests", "examples")

# Import name -> distribution name, for the cases where they differ. Kept explicit and static rather
# than read from the installed environment: the `locked` CI job installs a lockfile that does not
# contain every declared distribution, so an environment-derived map would make this test env-dependent.
IMPORT_TO_DISTRIBUTION = {"PIL": "pillow"}

FIRST_PARTY = {"openmucf"}

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
