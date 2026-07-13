"""Packaging + lazy public API (PEP 562) guards.

The heavy public submodules load lazily on first attribute access, so a bare `import openmucf`
never pays the numpyro/statistics import cost. These tests assert the marker file is packaged,
the lazy names resolve, and importing the package does not eager-load the heavy stack."""

import subprocess
import sys
from pathlib import Path

import openmucf

REPO = Path(__file__).resolve().parents[1]

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
