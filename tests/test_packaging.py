"""Packaging + lazy public API (PEP 562) guards.

The heavy public submodules load lazily on first attribute access, so a bare `import openmucf`
never pays the numpyro/statistics import cost. These tests assert the marker file is packaged,
the lazy names resolve, and importing the package does not eager-load the heavy stack.

2026-08-12 amendment: also guards the DEPENDENCY DECLARATION itself. numpy and Pillow were imported
by shipped code while arriving only transitively; the last test below makes that class of omission a
test failure instead of a latent packaging bug, so it cannot recur silently as more code lands."""

import ast
import hashlib
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

# A module under tests/ that a test imports by name -- `conftest`, which pytest makes importable, or a test
# module whose tables another test reuses -- is first-party, not a distribution: the static scan below
# takes an imported name as first-party whenever tests/ holds a file of that name, so a new sibling import
# needs no entry here. `openmucf` and `scripts` are this repository's own packages.
FIRST_PARTY = {"openmucf", "scripts"}

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
                if name in stdlib or name in FIRST_PARTY or (REPO / "tests" / f"{name}.py").is_file():
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
    check_build_floor(pyproject["build-system"]["requires"])


def check_build_floor(requires: list[str]) -> None:
    """The build needs one requirement: a setuptools floor above 76.1.0, which refuses `license` as a string.
    A higher floor is admitted."""
    (requirement,) = requires
    floor = re.fullmatch(r"setuptools>=(\d+(?:\.\d+)*)", requirement)
    assert floor, requirement
    assert tuple(int(part) for part in floor.group(1).split(".")) > (76, 1, 0), requirement


#: The licence names a per-path statement may use, each matched as a whole word.
LICENCES = ("Apache-2.0", "CC-BY-4.0", "Geant4 Software License", "MIT")
#: Each path token of the per-path statements, and the one licence of :data:`LICENCES` a clause naming that
#: path must name -- or `None` for a path this repository offers under no licence of its own, beside which a
#: clause names no licence. A token is a path, a directory ending in `/`, or a pattern whose `*` stands for
#: any run of characters other than `/`. An occurrence of a token inside an occurrence of a longer one names
#: only the longer token's path, so the tables named inside `data/g4/` keep their own licence there.
LICENCE_BY_PATH: dict[str, str | None] = {
    "(LICENSE)": "Apache-2.0",
    "(LICENSE-DATA)": "CC-BY-4.0",
    "openmucf/data": "CC-BY-4.0",
    "cpp/patches/": "Geant4 Software License",
    "third_party/geant4/": "Geant4 Software License",
    "third_party/mudirac/": "MIT",
    "data/g4/": "CC-BY-4.0",
    "data/g4/d1/capture_rate_cells.csv": None,
    "data/g4/d1/zeff_audit.csv": None,
    "data/g4/d1/suzuki1987_*.csv": None,
    "data/g4/d1/mizuno2025_table*.csv": None,
    "data/g4/d1/mean_life_comparison.csv": None,
    "data/g4/d1/d1_capture.suzuki1987.*": None,
    "data/g4/d1/d1_capture.mizuno2025.*": None,
    "data/g4/reference/d2/printed_rows.csv": None,
    "data/g4/reference/d2/measurements.csv": None,
    "data/g4/d2/selector_vs_primary.csv": None,
    "data/g4/d3/validation_cells.csv": None,
    "data/g4/d3/intensity_ratios.csv": None,
    "data/g4/d3/validation.csv": None,
    "data/g4/d3/shell_projection.csv": None,
    "data/g4/d3/consumer_centroids.csv": None,
    "data/g4/d3/history/validation.mudirac130.csv": None,
    "data/g4/stopped_mu/preregistration.json": None,
    "data/g4/stopped_mu/comparison.csv": None,
    "data/g4/stopped_mu/cascade_reference.csv": None,
    "data/g4/stopped_mu/cascade_comparison.csv": None,
    "data/g4/d1/d1_capture.g4dat": "Geant4 Software License",
    "data/g4/d1/d1_capture.prov.json": "Geant4 Software License",
    "data/g4/d1/d1_zeff.g4dat": "Geant4 Software License",
    "data/g4/d1/d1_zeff.prov.json": "Geant4 Software License",
    "data/g4/d1/geant4_natural_isotopes.csv": "Geant4 Software License",
    "data/g4/d1/d1_decay_rate.oracle": "Geant4 Software License",
    "data/g4/d3/mudirac_inputs.csv": "MIT",
    "data/g4/d3/LICENSE-MUDIRAC": "MIT",
}
#: The licence of Geant4 itself, which the patches, the vendored source and the files that restate it carry.
GEANT4_LICENCE = "Geant4 Software License"
#: The `parity` tables, which carry the Geant4 Software License inside `data/g4/`.
PARITY = (
    "data/g4/d1/d1_capture.g4dat", "data/g4/d1/d1_capture.prov.json",
    "data/g4/d1/d1_zeff.g4dat", "data/g4/d1/d1_zeff.prov.json",
)
#: MuDirac's input radii and the text of its MIT License, under which they are carried.
#: The natural isotopic compositions read from Geant4 builds, which restate values compiled into Geant4 and
#: carry the Geant4 Software License, as the `parity` tables do.
GEANT4_COMPOSITIONS = ("data/g4/d1/geant4_natural_isotopes.csv",)
#: The decay rates and nuclear masses read from Geant4 builds, which carry the Geant4 Software License, as the
#: `parity` tables do.
GEANT4_DECAY_RATES = ("data/g4/d1/d1_decay_rate.oracle",)
MUDIRAC = ("data/g4/d3/mudirac_inputs.csv", "data/g4/d3/LICENSE-MUDIRAC")
#: The files under `data/g4/` that restate cells of tables a third party publishes -- as printed, rescaled or
#: normalized, whether alone, as the values of a table generated from them or beside this project's values in
#: a comparison -- and so carry no licence of this repository. A file that repeats a third-party value only
#: as an input a computation was run with is this project's record of that computation.
RESTATED = (
    "data/g4/d1/capture_rate_cells.csv",
    "data/g4/d1/d1_capture.mizuno2025.g4dat",
    "data/g4/d1/d1_capture.mizuno2025.prov.json",
    "data/g4/d1/d1_capture.suzuki1987.g4dat",
    "data/g4/d1/d1_capture.suzuki1987.prov.json",
    "data/g4/d1/mean_life_comparison.csv",
    "data/g4/d1/mizuno2025_table1.csv",
    "data/g4/d1/mizuno2025_table3.csv",
    "data/g4/d1/suzuki1987_parity_cells.csv",
    "data/g4/d1/suzuki1987_preprint_differences.csv",
    "data/g4/d1/suzuki1987_printed_rows.csv",
    "data/g4/d1/suzuki1987_quantity_rows.csv",
    "data/g4/d1/zeff_audit.csv",
    "data/g4/d2/selector_vs_primary.csv",
    "data/g4/d3/consumer_centroids.csv",
    "data/g4/d3/history/validation.mudirac130.csv",
    "data/g4/d3/intensity_ratios.csv",
    "data/g4/d3/shell_projection.csv",
    "data/g4/d3/validation.csv",
    "data/g4/d3/validation_cells.csv",
    "data/g4/reference/d2/measurements.csv",
    "data/g4/stopped_mu/comparison.csv",
    "data/g4/stopped_mu/preregistration.json",
    "data/g4/stopped_mu/cascade_comparison.csv",
    "data/g4/stopped_mu/cascade_reference.csv",
    "data/g4/reference/d2/printed_rows.csv",
)
#: The other files under `data/g4/`: what this project computed, harvested from Geant4 or MuDirac, or wrote.
PROJECT_AUTHORED = (
    "data/g4/d1/bound_rates_builds.csv",
    "data/g4/d1/bound_rates_drill_rulings.csv",
    "data/g4/d1/bound_rates_drills.csv",
    "data/g4/d1/bound_rates_harness_defaults.csv",
    "data/g4/d1/bound_rates_harvest.csv",
    "data/g4/d1/bound_rates_identity.csv",
    "data/g4/d1/bound_rates_identity_drills.csv",
    "data/g4/d1/bound_rates_identity_verdicts.csv",
    "data/g4/d1/bound_rates_policy.json",
    "data/g4/d1/bound_rates_results.json",
    "data/g4/d1/bound_rates_sweep.csv",
    "data/g4/d1/bound_rates_theory.json",
    "data/g4/d1/bound_rates_workers.csv",
    "data/g4/d1/bound_rates_raw_digests.csv",
    "data/g4/d1/history/bound_rates_chunk_files.analysis1.csv",
    "data/g4/d1/history/bound_rates_results.analysis1.json",
    "data/g4/d1/d1_gp_sweep.oracle",
    "data/g4/d1/geant4_add_dataset.snippet",
    "data/g4/d1/isotope_audit.csv",
    "data/g4/d1/natural_harvest_builds.csv",
    "data/g4/d1/mean_life_estimators.csv",
    "data/g4/d1/natural_mixture_builds.csv",
    "data/g4/d1/natural_mixture_comparison.csv",
    "data/g4/d1/natural_mixture_harvest.csv",
    "data/g4/d1/natural_mixture_policy.json",
    "data/g4/d1/natural_target_answers.csv",
    "data/g4/d1/natural_target_capture.csv",
    "data/g4/d1/record_classes.csv",
    "data/g4/d2/harvest_builds.csv",
    "data/g4/d2/selector_bindings.csv",
    "data/g4/d2/selector_call_sites.csv",
    "data/g4/d2/selector_harvest.csv",
    "data/g4/d2/selector_summary.csv",
    "data/g4/d3/centroid_margins.csv",
    "data/g4/d3/change_attribution.csv",
    "data/g4/d3/components.jsonl",
    "data/g4/d3/d3_kshell.mudirac130p1.g4dat",
    "data/g4/d3/d3_kshell.mudirac130p1.prov.json",
    "data/g4/d3/d3_levels.mudirac130p1.g4dat",
    "data/g4/d3/d3_levels.mudirac130p1.prov.json",
    "data/g4/d3/geant4_cascade_levels.csv",
    "data/g4/d3/incompatible_groups.csv",
    "data/g4/d3/mudirac_lines.csv",
    "data/g4/d3/mudirac_runs.csv",
    "data/g4/d3/mudirac_states.csv",
    "data/g4/d3/numerical_check.csv",
    "data/g4/d3/numerical_check_status.json",
    "data/g4/d3/reference/README.md",
    "data/g4/d3/reference/cases.json",
    "data/g4/d3/reference/expected_keys.json",
    "data/g4/d3/reference/payloads.jsonl",
    "data/g4/d3/reference/protocol.json",
    "data/g4/d3/reference/runs.jsonl",
    "data/g4/d3/radius_lineage.csv",
    "data/g4/d3/validation_radius_origin.csv",
    "data/g4/example.g4dat",
    "data/g4/example.prov.json",
    "data/g4/geant4_add_dataset.snippet",
    "data/g4/reference/d2/sources.csv",
    "data/g4/stopped_mu/builds.csv",
    "data/g4/stopped_mu/cells.jsonl",
    "data/g4/stopped_mu/reproduced.csv",
    "data/g4/stopped_mu/verdicts.csv",
    "data/g4/stopped_mu/cascade_rule.json",
    "data/g4/stopped_mu/cascade_harvest.csv",
    "data/g4/stopped_mu/cascade_transitions.csv",
    "data/g4/stopped_mu/cascade_consumer.csv",
)
#: Every file under `data/g4/`, with the licence it carries. A file added there needs its line here.
G4_DATA: dict[str, str | None] = {
    **dict.fromkeys(RESTATED, None),
    **dict.fromkeys(PROJECT_AUTHORED, "CC-BY-4.0"),
    **dict.fromkeys(PARITY, "Geant4 Software License"),
    **dict.fromkeys(GEANT4_COMPOSITIONS, "Geant4 Software License"),
    **dict.fromkeys(GEANT4_DECAY_RATES, "Geant4 Software License"),
    **dict.fromkeys(MUDIRAC, "MIT"),
}
#: The words in which a statement says that a path carries no licence of this repository.
NO_LICENCE = "no licence of this repository"
#: The paths the comment of `CITATION.cff` names; the README's License section names the rest.
CITATION_PATHS = ("(LICENSE)", "(LICENSE-DATA)", "openmucf/data", "cpp/patches/", "third_party/geant4/")
#: A `license` or `license-url` key at any level of a CFF file, quoted or not.
CFF_LICENSE_KEY = re.compile(r"""^\s*["']?license(?:-url)?["']?\s*:""", re.MULTILINE)
#: The SHA-256 of MuDirac 1.3.0's `LICENSE` as the release archive whose SHA-256
#: `openmucf.g4.sources.mudirac130.TARBALL_SHA256` pins carries it.
MUDIRAC_LICENSE_SHA256 = "496542fd7321f869d0ba12efcc808c62a73ea5d7240b2ca28f55a305fbed27fa"


def licences_named(clause: str) -> list[str]:
    """The licences of :data:`LICENCES` that `clause` names, in that order, each matched as a whole word."""
    return [name for name in LICENCES if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", clause)]


def paths_named(clause: str) -> set[str]:
    """The paths of :data:`LICENCE_BY_PATH` that `clause` names, counting no occurrence of a token that lies
    inside an occurrence of a longer one."""
    spans = [
        (match.start(), match.end(), path)
        for path in LICENCE_BY_PATH
        for match in re.finditer(re.escape(path), clause)
    ]
    return {
        path for start, end, path in spans
        if not any(s <= start and end <= e and e - s > end - start for s, e, _ in spans)
    }


def licence_of(path: str) -> str | None:
    """The licence :data:`LICENCE_BY_PATH` gives the file at `path`, relative to the repository: that of the
    token equal to it, else of the longest token whose pattern or directory takes it in."""
    if path in LICENCE_BY_PATH:
        return LICENCE_BY_PATH[path]
    tokens = [
        token for token in LICENCE_BY_PATH
        if re.fullmatch(re.escape(token).replace(r"\*", "[^/]*") + (".*" if token.endswith("/") else ""),
                        path)
    ]
    assert tokens, path
    return LICENCE_BY_PATH[max(tokens, key=len)]


def g4_data_files() -> set[str]:
    """Every file under `data/g4/`, as a path relative to the repository."""
    return {path.relative_to(REPO).as_posix() for path in (REPO / "data" / "g4").rglob("*") if path.is_file()}


def check_g4_data_inventory(files: set[str], inventory: dict[str, str | None]) -> None:
    """`files` are exactly the files `inventory` lists, and :func:`licence_of` gives each the licence
    `inventory` gives it."""
    assert files == set(inventory), sorted(files ^ set(inventory))
    for path, licence in inventory.items():
        assert licence_of(path) == licence, (path, licence_of(path), licence)


def clauses(text: str) -> list[str]:
    """`text` with its whitespace collapsed, split at semicolons and at sentence ends."""
    return re.split(r";|\.\s", " ".join(text.split()))


def check_licences_by_path(text: str, paths: tuple[str, ...]) -> None:
    """Every path in `paths` occurs in `text`, and every clause that names a path of :data:`LICENCE_BY_PATH`
    names exactly the licence that path carries: none at all for a path carrying `None`. A clause therefore
    fails when it names, beside a path, a licence other than the path's, a second licence, no licence where
    the path carries one, or a licence where the path carries none -- and when it names two paths that carry
    different licences."""
    flat = " ".join(text.split())
    assert all(path in flat for path in paths), [path for path in paths if path not in flat]
    for clause in clauses(text):
        named = licences_named(clause)
        for path in paths_named(clause):
            licence = LICENCE_BY_PATH[path]
            want = [] if licence is None else [licence]
            assert named == want, f"{path!r} is given {named}, not {want}: {clause!r}"


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


def _section(text: str, heading: str) -> str:
    """The body of the section that `heading` opens, up to the next heading of any level or the end."""
    match = re.search(rf"^{re.escape(heading)}\n(.*?)(?=^#+ |\Z)", text, re.MULTILINE | re.DOTALL)
    assert match, heading
    return match.group(1)


def _note(text: str, marker: str) -> str:
    """What follows `marker`, which opens the note that closes a licence file, and occurs once."""
    assert text.count(marker) == 1, marker
    return text.split(marker, 1)[1]


def _tokens_under(*directories: str) -> tuple[str, ...]:
    """The tokens of :data:`LICENCE_BY_PATH` that lie in one of `directories`, and `data/g4/` itself."""
    return ("data/g4/", *(token for token in LICENCE_BY_PATH if token.startswith(directories)))


def _licensed_as(directory: str, licence: str) -> tuple[str, ...]:
    """The tokens of :data:`LICENCE_BY_PATH` inside `directory`, the directory itself excepted, that carry
    `licence`: what a statement that says where `licence` applies must name path by path."""
    return tuple(token for token, carried in LICENCE_BY_PATH.items()
                 if token.startswith(directory) and token != directory and carried == licence)


def _licensed_otherwise(directory: str) -> tuple[str, ...]:
    """The tokens of :data:`LICENCE_BY_PATH` inside `directory`, the directory itself excepted, that carry a
    licence of their own other than the directory's: what a statement that grants the directory's licence to
    the files in it by what they are must name path by path. A token carrying no licence is left to the
    README's list, which such a statement cites and :func:`check_licences_by_path` holds to every token."""
    granted = LICENCE_BY_PATH[directory]
    return tuple(token for token, carried in LICENCE_BY_PATH.items()
                 if token.startswith(directory) and token != directory and carried not in (None, granted))


def licence_statements() -> dict[str, tuple[str, tuple[str, ...]]]:
    """The prose statements of licences by path that :func:`check_licences_by_path` reads, keyed by file:
    each with the text it reads and the paths that text must name. A dataset document names every token of
    its own directories; a statement that grants the licence of `data/g4/` to the files there by what they
    are names every file in it that carries another licence, and one that says where the Geant4 Software
    License applies names every file in `data/g4/` that carries it."""

    def read(name: str) -> str:
        return (REPO / name).read_text(encoding="utf-8")

    return {
        "README.md": (_section(read("README.md"), "## License"), tuple(LICENCE_BY_PATH)),
        "LICENSE": (
            _note(read("LICENSE"), "NOTE (not part of the Apache License)"),
            ("third_party/geant4/", "cpp/patches/"),
        ),
        "LICENSE-DATA": (
            _note(read("LICENSE-DATA"), "NOTE (not part of the Creative Commons licence)"),
            ("openmucf/data", "data/g4/", "third_party/geant4/", "cpp/patches/",
             *_licensed_otherwise("data/g4/")),
        ),
        "DATASET_D1.md": (
            _section(read("DATASET_D1.md"), "## 8. Licensing"),
            (*_tokens_under("data/g4/d1/"), "third_party/geant4/", "cpp/patches/"),
        ),
        "DATASET_D2.md": (
            _section(read("DATASET_D2.md"), "## Licensing"),
            _tokens_under("data/g4/d2/", "data/g4/reference/d2/"),
        ),
        "DATASET_D3.md": (_section(read("DATASET_D3.md"), "## 9. Licensing"), _tokens_under("data/g4/d3/")),
        "docs/index.md": (
            _section(read("docs/index.md"), "## License"),
            ("openmucf/data", "cpp/patches/", "third_party/geant4/", "data/g4/",
             *_licensed_otherwise("data/g4/")),
        ),
        "FORMAT_SPEC.md": (
            _section(read("FORMAT_SPEC.md"), "## 9. Attribution"),
            ("third_party/geant4/", "cpp/patches/", *_licensed_as("data/g4/", GEANT4_LICENCE)),
        ),
        "third_party/geant4/README.md": (
            _section(read("third_party/geant4/README.md"), "## Licensing"),
            ("cpp/patches/", *_licensed_as("data/g4/", GEANT4_LICENCE)),
        ),
        "CONTRIBUTING.md": (
            _section(read("CONTRIBUTING.md"), "## 7. Licensing (by artifact type, on purpose)"),
            ("openmucf/data", "cpp/patches/"),
        ),
    }


def patch_offer(text: str) -> str:
    """The clause of `cpp/patches/README.md` that offers the patch files under a licence."""
    (offer,) = [clause for clause in clauses(text) if clause.startswith("The patch files in this directory")]
    return offer


def test_the_citation_and_archive_metadata_state_the_licences_by_path():
    """The repository's files carry licences by path, and each metadata file says so in the form its
    format allows. `codemeta.json` lists the licence documents. `.zenodo.json` takes one licence id, which
    Zenodo applies to every file of the record, so it carries `other-open` and its notes give each path of
    :data:`LICENCE_BY_PATH` that carries a licence that licence. CFF reads a list of `license` ids as
    alternatives, so `CITATION.cff` sets no `license` or `license-url` key at any level and its comment gives
    each path it names its licence. The README badge lists the licences and links to the section that names
    their paths."""
    codemeta = json.loads((REPO / "codemeta.json").read_text(encoding="utf-8"))
    assert codemeta["license"] == [
        "https://spdx.org/licenses/Apache-2.0",
        "https://spdx.org/licenses/CC-BY-4.0",
        "http://cern.ch/geant4/license",
        "https://spdx.org/licenses/MIT",
    ]
    zenodo = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    assert zenodo["license"] == "other-open"
    licensed = tuple(path for path, licence in LICENCE_BY_PATH.items() if licence is not None)
    check_licences_by_path(zenodo["notes"], licensed)
    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert not CFF_LICENSE_KEY.search(citation)
    check_licences_by_path(citation_comment(citation), CITATION_PATHS)
    check_licence_badge((REPO / "README.md").read_text(encoding="utf-8"))


def test_every_licence_statement_gives_each_path_it_names_its_licence():
    """Each statement :func:`licence_statements` reads names the paths it must and gives each path of
    :data:`LICENCE_BY_PATH` it names that path's licence; the README's License section names, in one clause
    that says in so many words that they carry no licence of this repository, exactly the paths that carry
    none; and the offer in `cpp/patches/README.md` names the Geant4 Software License and no other licence."""
    statements = licence_statements()
    for name, (text, paths) in statements.items():
        try:
            check_licences_by_path(text, paths)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from None
    readme = statements["README.md"][0]
    unlicensed = {path for path, licence in LICENCE_BY_PATH.items() if licence is None}
    assert any(NO_LICENCE in clause and paths_named(clause) == unlicensed for clause in clauses(readme))
    offer = patch_offer((REPO / "cpp" / "patches" / "README.md").read_text(encoding="utf-8"))
    assert licences_named(offer) == ["Geant4 Software License"], offer


def test_every_file_under_data_g4_carries_the_licence_its_class_gives_it():
    """The files under `data/g4/` are exactly those :data:`G4_DATA` lists, so a file added there is classed
    before it ships, and the paths of the statements give each file the licence its class gives it: none for
    a file of :data:`RESTATED`, CC-BY-4.0 for one of :data:`PROJECT_AUTHORED`, the Geant4 Software License for
    the `parity` tables, the natural isotopic compositions and the decay rates read from Geant4 builds, and
    the MIT License for MuDirac's radii and their licence text."""
    check_g4_data_inventory(g4_data_files(), G4_DATA)


def _archive_generator():
    """`scripts/generate_g4data.py`, loaded by path -- `scripts/` is a directory, not a package."""
    import importlib.util

    module_spec = importlib.util.spec_from_file_location("generate_g4data_licence", REPO / "scripts" /
                                                         "generate_g4data.py")
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


#: The words of a `LICENSE`-member statement that give each class of :data:`G4_DATA`.
MEMBER_CLASS = (("Geant4 Software License", "Geant4 Software License"), ("CC-BY-4.0", "CC-BY-4.0"),
                ("offered under no licence of the openmucf repository", None))


def licence_member_problems(text: str, members: list[str], texts: dict[str, bytes]) -> list[str]:
    """What makes the archive's `LICENSE` member (``text``) false against the repository: a member other than
    `LICENSE` that no statement line names, or that two name; a statement whose opening words give a member a
    class other than the one :data:`G4_DATA` gives the file of that name under `data/g4/` (`README` and
    `History`, generated for the archive, are project-authored: CC-BY-4.0); a licence text not byte-equal to
    the repository file it copies (``texts``: title to bytes)."""
    problems: list[str] = []
    named: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"  - ([^:]+): (.+)", line)
        if match:
            for name in match.group(1).split(", "):
                if name in named:
                    problems.append(f"{name} named twice")
                named[name] = match.group(2)
    if sorted(named) != sorted(m for m in members if m != "LICENSE"):
        problems.append(f"named {sorted(named)} != members {sorted(members)}")
    for name, statement in named.items():
        classes = [cls for words, cls in MEMBER_CLASS if statement.startswith(words)]
        paths = [path for path in G4_DATA if path.endswith(f"/{name}")]
        if name in ("README", "History"):
            want: str | None = "CC-BY-4.0"
        else:
            want = G4_DATA[paths[0]] if len(paths) == 1 else "?"
        if classes != [want]:
            problems.append(f"{name}: statement {statement[:60]!r} gives {classes}, G4_DATA gives {want!r}")
    for title, data in texts.items():
        body = data.replace(b"\r\n", b"\n").decode("ascii").rstrip("\n")
        if f"\n--- {title} ---\n\n{body}\n" not in text + "\n":
            problems.append(f"the text of {title} is not the repository copy")
    return problems


def test_the_archive_licence_member_gives_each_member_the_licence_of_its_file():
    """The `LICENSE` member of the archive the snippet registers names every other member once, gives each
    the licence class :data:`G4_DATA` gives the file of that name under `data/g4/`, copies the Geant4
    Software License and MuDirac's MIT License byte for byte, and the README names it; one profile given
    another licence in `PROFILE_LICENCES` is refused."""
    import io
    import tarfile

    gen = _archive_generator()
    _, archive = gen.build_dataset_artifacts()
    with tarfile.open(fileobj=io.BytesIO(archive)) as opened:
        members = {m.name.split("/", 1)[1]: opened.extractfile(m).read() for m in opened.getmembers()}
    texts = {"Geant4 Software License v1.0": (REPO / "third_party" / "geant4" / "LICENSE").read_bytes(),
             "MuDirac's MIT License": (REPO / "data" / "g4" / "d3" / "LICENSE-MUDIRAC").read_bytes()}
    licence = members["LICENSE"].decode("ascii")
    assert licence_member_problems(licence, list(members), texts) == []
    readme = members["README"].decode("ascii").splitlines()
    assert "  - LICENSE: the licence of each file of this directory" in readme
    listed = re.compile(r"  - (\S+\.g4dat): #TABLE (\S+), #PROFILE (\S+), #SEAM (\S+), (\d+) records")
    files = [(m.group(1), f"{m.group(1)[:-len('.g4dat')]}.prov.json", m.group(2), m.group(3), m.group(4),
              int(m.group(5)))
             for line in readme if (m := listed.fullmatch(line))]
    rebuilt = gen.emit.license_member(
        name=gen.DATASET_NAME, version=gen.DATASET_VERSION, statements=gen._licence_statements(files),
        texts=tuple(texts.items()),
    )
    assert rebuilt == members["LICENSE"]
    profile = "suzuki1987"
    flipped = dict(gen.PROFILE_LICENCES, **{profile: gen.CC_BY})
    gen.PROFILE_LICENCES = flipped
    drilled = gen.emit.license_member(
        name=gen.DATASET_NAME, version=gen.DATASET_VERSION, statements=gen._licence_statements(files),
        texts=tuple(texts.items()),
    ).decode("ascii")
    assert drilled != licence
    assert any(problem.startswith(f"d1_capture.{profile}") for problem in
               licence_member_problems(drilled, list(members), texts))


def _licence_member_and_texts() -> tuple[str, list[str], dict[str, bytes]]:
    """The `LICENSE` member of the archive the generator builds, the archive's member names, and the two
    licence texts it must copy, by title."""
    import io
    import tarfile

    _, archive = _archive_generator().build_dataset_artifacts()
    with tarfile.open(fileobj=io.BytesIO(archive)) as opened:
        members = {m.name.split("/", 1)[1]: opened.extractfile(m).read() for m in opened.getmembers()}
    texts = {"Geant4 Software License v1.0": (REPO / "third_party" / "geant4" / "LICENSE").read_bytes(),
             "MuDirac's MIT License": (REPO / "data" / "g4" / "d3" / "LICENSE-MUDIRAC").read_bytes()}
    return members["LICENSE"].decode("ascii"), list(members), texts


@pytest.mark.parametrize("case", ("member not named", "member absent", "named twice", "another class",
                                  "text altered", "text missing"))
def test_drill_each_licence_member_check_refuses_its_own_fault_alone(case: str):
    """Each check of :func:`licence_member_problems`, given a `LICENSE` member that is correct in every
    other respect: a member no statement names (`History` dropped from its line), a statement naming a member
    the archive lacks, a member named on a second line too, a statement giving a profile another licence
    class, and a licence text altered by one character or left out. Each is refused, and by that check
    alone."""
    licence, members, texts = _licence_member_and_texts()
    assert licence_member_problems(licence, members, texts) == []
    lines = licence.split("\n")
    (generated,) = [i for i, line in enumerate(lines) if line.startswith("  - README, History: ")]
    cc_by = lines[generated].split(": ", 1)[1]
    if case == "member not named":
        lines[generated] = f"  - README: {cc_by}"
    elif case == "member absent":
        members = [name for name in members if name != "History"]
    elif case == "named twice":
        lines.insert(generated + 1, f"  - README: {cc_by}")
    elif case == "another class":
        (profile,) = [i for i, line in enumerate(lines) if line.startswith("  - d1_capture.suzuki1987.g4dat")]
        lines[profile] = lines[profile].split(": ", 1)[0] + f": {cc_by}"
    elif case == "text altered":
        (title,) = [i for i, line in enumerate(lines) if line == "--- Geant4 Software License v1.0 ---"]
        body = title + 2
        assert lines[body]
        lines[body] = lines[body][:-1] + ("x" if lines[body][-1] != "x" else "y")
    else:
        (title,) = [i for i, line in enumerate(lines) if line == "--- MuDirac's MIT License ---"]
        lines = lines[:title]
    problems = licence_member_problems("\n".join(lines), members, texts)
    if case in ("member not named", "member absent"):
        assert len(problems) == 1 and problems[0].startswith("named "), problems
    elif case == "named twice":
        assert problems == ["README named twice"], problems
    elif case == "another class":
        assert [p.split(":", 1)[0] for p in problems] == [
            "d1_capture.suzuki1987.g4dat", "d1_capture.suzuki1987.prov.json"], problems
        assert all("gives ['CC-BY-4.0'], G4_DATA gives None" in p for p in problems), problems
    elif case == "text altered":
        assert problems == ["the text of Geant4 Software License v1.0 is not the repository copy"], problems
    else:
        assert problems == ["the text of MuDirac's MIT License is not the repository copy"], problems


#: The README sentences that name columns of a file under `data/g4/`, each with that file. Each pattern
#: captures, as ``columns``, the clause that lists the file's columns; the file's own name, and the
#: third-party file the mudirac clause cites, lie outside the capture, so every backquoted token inside it is
#: read as a column name, whatever it ends with.
COLUMN_SENTENCES = (
    (r"Of `mudirac_inputs\.csv`'s columns, (?P<columns>[^.]+?) repeat the IAEA charge-radii table "
     r"\(`charge_radii\.csv`\)", "data/g4/d3/mudirac_inputs.csv"),
    (r"in `suzuki1987_parity_cells\.csv` those are the (?P<columns>[^.]+) columns\.",
     "data/g4/d1/suzuki1987_parity_cells.csv"),
)


def column_clauses(readme: str) -> dict[str, list[str]]:
    """Each file of :data:`COLUMN_SENTENCES` to the column-list clauses the README states for it."""
    flat = " ".join(readme.split())
    return {path: [m.group("columns") for m in re.finditer(pattern, flat)]
            for pattern, path in COLUMN_SENTENCES}


def column_problems(readme: str, headers: dict[str, list[str]]) -> list[str]:
    """What makes the README's column sentences false against the files' headers (``headers``, path to
    columns): a sentence the README lacks or states more than once, a clause with an unpaired backquote or
    with no backquoted name, and each backquoted name in the clause that the file's header does not carry."""
    problems = []
    for path, clauses in column_clauses(readme).items():
        if len(clauses) != 1:
            problems.append(f"{path}: {len(clauses)} sentences match")
            continue
        clause = clauses[0]
        if clause.count("`") % 2:
            problems.append(f"{path}: an unpaired backquote in {clause!r}")
            continue
        named = re.findall(r"`([^`]*)`", clause)
        if not named:
            problems.append(f"{path}: the sentence names no column")
        problems += [f"{path}: no column {name!r}" for name in named if name not in headers[path]]
    return problems


def _readme_and_headers() -> tuple[str, dict[str, list[str]]]:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    headers = {path: (REPO / path).read_text(encoding="utf-8").splitlines()[0].split(",")
               for _, path in COLUMN_SENTENCES}
    return readme, headers


def test_the_mudirac_inputs_columns_the_readme_names_are_the_files_header():
    """The columns the License section of the README names -- those of `data/g4/d3/mudirac_inputs.csv`
    that repeat MuDirac's radii or the IAEA table, and those of `data/g4/d1/suzuki1987_parity_cells.csv`
    that repeat compiled-in values -- are columns of each file's header. Every name the sentences give, read
    from the README here, is drilled twice: misspelled in the README, and renamed in its file's header; each
    is refused by the header check alone."""
    readme, headers = _readme_and_headers()
    assert column_problems(readme, headers) == []
    flat = " ".join(readme.split())
    drilled = 0
    for path, (clause,) in column_clauses(readme).items():
        names = re.findall(r"`([^`]*)`", clause)
        assert names, path
        for name in names:
            assert clause.count(f"`{name}`") == 1, (path, name)
            misspelled = flat.replace(clause, clause.replace(f"`{name}`", f"`{name}_bogus`"))
            assert column_problems(misspelled, headers) == [f"{path}: no column {name + '_bogus'!r}"], name
            renamed = dict(headers, **{path: [column if column != name else column + "_x"
                                              for column in headers[path]]})
            assert column_problems(readme, renamed) == [f"{path}: no column {name!r}"], name
            drilled += 1
    assert drilled == sum(len(re.findall(r"`([^`]*)`", c)) for (c,) in column_clauses(readme).values())


@pytest.mark.parametrize("case", ("twice", "no backquoted name", "an unknown name alone", "unpaired"))
def test_drill_a_repeated_column_sentence_or_one_without_a_column_is_refused(case: str):
    """Each sentence, otherwise correct, stated twice; its clause with no backquoted name; its clause naming
    only a name its header lacks; and its clause with an unpaired backquote: each is refused, and for that
    sentence's file alone."""
    readme, headers = _readme_and_headers()
    flat = " ".join(readme.split())
    for pattern, path in COLUMN_SENTENCES:
        match = re.search(pattern, flat)
        assert match is not None, path
        clause = match.group("columns")
        sentence = match.group(0)
        if case == "twice":
            drilled, want = flat + " " + sentence, [f"{path}: 2 sentences match"]
        elif case == "no backquoted name":
            drilled = flat.replace(sentence, sentence.replace(clause, "the radii"))
            want = [f"{path}: the sentence names no column"]
        elif case == "an unknown name alone":
            drilled = flat.replace(sentence, sentence.replace(clause, "`bogus`"))
            want = [f"{path}: no column 'bogus'"]
        else:
            drilled = flat.replace(sentence, sentence.replace(clause, clause + " `"))
            want = [f"{path}: an unpaired backquote in {clause + ' `'!r}"]
        assert column_problems(drilled, headers) == want, (case, path)


def test_the_mudirac_licence_text_is_the_release_copy():
    """`data/g4/d3/LICENSE-MUDIRAC` is MuDirac 1.3.0's `LICENSE` byte for byte, as the release archive whose
    SHA-256 `openmucf.g4.sources.mudirac130.TARBALL_SHA256` pins carries it."""
    payload = (REPO / "data" / "g4" / "d3" / "LICENSE-MUDIRAC").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == MUDIRAC_LICENSE_SHA256


#: The six inputs of the MuDirac patch's default-output check, which its README names by count.
MUDIRAC_DEFAULT_CHECK = (
    "Ag107_ideal7.in", "He3_base.in", "Pb208_g0u0100.in", "Si28_g2u1000.in", "U238_rsig.in",
    "Zr90_g0u1000.in",
)


def test_the_mudirac_patch_directory_carries_the_release_licence_and_its_six_check_inputs():
    """`third_party/mudirac/` carries MuDirac 1.3.0's `LICENSE` byte for byte; the patch has LF line ends,
    as `patch -p1` applies it to the release archive's sources, and creates only its two option headers;
    the README's default-output sentence names six inputs, and `default_check/` holds exactly those six
    MuDirac inputs, each setting none of the keywords the patch adds."""
    directory = REPO / "third_party" / "mudirac"
    licence = (directory / "LICENSE").read_bytes()
    assert licence == (REPO / "data" / "g4" / "d3" / "LICENSE-MUDIRAC").read_bytes()
    assert hashlib.sha256(licence).hexdigest() == MUDIRAC_LICENSE_SHA256
    patch = (directory / "mudirac-1.3.0-p1.patch").read_bytes()
    assert b"\r" not in patch
    created = re.findall(rb"^--- /dev/null\n\+\+\+ b/(\S+)$", patch, re.MULTILINE)
    assert sorted(created) == [b"lib/consistency.hpp", b"lib/numerics.hpp"]
    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "on the six inputs in\n[`default_check/`](default_check/)" in readme
    assert tuple(sorted(p.name for p in (directory / "default_check").iterdir())) == MUDIRAC_DEFAULT_CHECK
    added = set(re.findall(r"^\| `([a-z0-9_]+)`", readme, re.MULTILINE))
    added |= {"domain_in_eps", "domain_out_eps"}
    for name in MUDIRAC_DEFAULT_CHECK:
        text = (directory / "default_check" / name).read_bytes()
        assert b"\r" not in text
        keys = {line.split(":")[0] for line in text.decode("ascii").splitlines()}
        assert keys & added == set(), (name, keys & added)


def test_drill_a_file_given_a_licence_of_its_own_must_be_named_by_each_statement_that_reaches_it(monkeypatch):
    """A file under `data/g4/d1/` that :data:`LICENCE_BY_PATH` gives the Geant4 Software License joins the
    paths that the README's License section, the licensing section of `DATASET_D1.md`, the note of
    `LICENSE-DATA`, the License section of `docs/index.md`, section 9 of `FORMAT_SPEC.md` and the Licensing
    section of `third_party/geant4/README.md` must name; one given MuDirac's MIT License joins those of the
    first four; and each statement that does not name it is refused."""
    planted = "data/g4/d1/planted_table.csv"
    reaches = {
        GEANT4_LICENCE: {"README.md", "DATASET_D1.md", "LICENSE-DATA", "docs/index.md", "FORMAT_SPEC.md",
                         "third_party/geant4/README.md"},
        "MIT": {"README.md", "DATASET_D1.md", "LICENSE-DATA", "docs/index.md"},
    }
    for licence, expected in reaches.items():
        monkeypatch.setitem(LICENCE_BY_PATH, planted, licence)
        refused = set()
        for name, (text, paths) in licence_statements().items():
            try:
                check_licences_by_path(text, paths)
            except AssertionError:
                refused.add(name)
        assert refused == expected, (licence, sorted(refused))


def test_drill_a_wrong_or_swapped_licence_by_path_is_refused():
    """The per-path, inventory, badge, CFF-key and build-floor guards, shown to fire on in-memory corruptions
    of the shipped texts: an unknown licence put in place of one, two licences swapped between paths, a
    second licence beside the right one, the patches given the code's licence, the CC-BY grant extended to the
    restated tables, the project's own files under `data/g4/` given the code's licence, a `parity` table left
    with no licence, MuDirac's radii given the code's licence, a path dropped from the README, the patch offer
    given another licence, a file under `data/g4/` that the inventory does not list, a restated table listed
    as CC-BY-4.0, a project-authored file listed with no licence, a quoted and a URL licence key in a CFF
    file, a licence dropped from the badge, the badge's link moved off the License section, and a build floor
    at setuptools 76.1.0."""
    notes = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))["notes"]
    comment = citation_comment((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = licence_statements()["README.md"][0]
    swapped = notes.replace("Apache-2.0 for code", "CC-BY-4.0 for code")
    swapped = swapped.replace("CC-BY-4.0 for the rate ledger", "Apache-2.0 for the rate ledger")
    parity_offer = (
        "offered under the **Geant4 Software License v1.0** "
        "([`third_party/geant4/LICENSE`](third_party/geant4/LICENSE)), as the patches are"
    )
    project_grant = "other than those the items below name: **CC-BY-4.0**"
    corruptions = [
        (notes, "Apache-2.0 for code", "BSD-3-Clause for code"),
        (notes, notes, swapped),
        (notes, "Apache-2.0 for code", "Apache-2.0 or MIT for code"),
        (comment, "The code is Apache-2.0", "The code is MIT"),
        (comment, "Geant4 Software License (cpp/patches/LICENSE)", "Apache-2.0 (cpp/patches/LICENSE)"),
        (section, f"offered under {NO_LICENCE}", "offered under CC-BY-4.0"),
        (section, project_grant, project_grant.replace("CC-BY-4.0", "Apache-2.0")),
        (section, parity_offer, f"offered under {NO_LICENCE}"),
        (section, "under MuDirac's **MIT** License", "under MuDirac's **Apache-2.0** License"),
    ]
    for text, old, new in corruptions:
        assert text.count(old) == 1, old
        with pytest.raises(AssertionError):
            check_licences_by_path(text.replace(old, new), ())
    dropped = "`data/g4/d3/mudirac_inputs.csv` "
    assert section.count(dropped) == 1
    with pytest.raises(AssertionError):
        check_licences_by_path(section.replace(dropped, ""), tuple(LICENCE_BY_PATH))
    offer = patch_offer((REPO / "cpp" / "patches" / "README.md").read_text(encoding="utf-8"))
    assert licences_named(offer.replace("Geant4 Software License", "Apache-2.0")) == ["Apache-2.0"]
    files = g4_data_files()
    for corrupted_files, inventory in (
        (files | {"data/g4/d1/new_table.csv"}, G4_DATA),
        (files, dict(G4_DATA, **{RESTATED[0]: "CC-BY-4.0"})),
        (files, dict(G4_DATA, **{PROJECT_AUTHORED[0]: None})),
    ):
        with pytest.raises(AssertionError):
            check_g4_data_inventory(corrupted_files, inventory)
    for key in ('"license": Apache-2.0', "license-url: https://example.org/licence"):
        assert CFF_LICENSE_KEY.search(f"cff-version: 1.2.0\n{key}\n"), key
    for old, new in (("%20%C2%B7%20CC--BY--4.0", ""), ("](#license)", "](#top)")):
        assert readme.count(old) == 1, old
        with pytest.raises(AssertionError):
            check_licence_badge(readme.replace(old, new))
    check_build_floor(["setuptools>=78"])
    with pytest.raises(AssertionError):
        check_build_floor(["setuptools>=76.1.0"])
