"""float64 is MANDATORY (rates span ~7 decades) -- and it must survive a SHADOWED package import.

Regression test for the failure found during the 2026-07-23 cross-architecture audit: a script run from a
directory that also contains a CLONE of this repository binds the top-level name ``openmucf`` to a
NAMESPACE package (the clone root has no ``__init__.py``), while an editable install's meta-path finder
still resolves the SUBMODULES to the real files. ``openmucf/__init__.py`` -- which used to be the ONLY
place ``jax_enable_x64`` was set -- then never runs, and every chain silently samples in float32.

The fix (``openmucf/_jaxcfg.py``) is two-layered, and both layers are tested here:
  1. every import path into the package enables x64 (test 1-3),
  2. the samplers hard-fail if it is ever off (test 4-5).
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import openmucf
from openmucf import _jaxcfg

_REPO_ROOT = Path(openmucf.__file__).resolve().parent.parent


def _run_isolated(code: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with ``cwd`` as the working directory (so cwd is sys.path[0])."""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd,
                          capture_output=True, text=True, timeout=300)


def test_plain_import_enables_x64():
    assert _jaxcfg.x64_enabled() is True


@pytest.mark.parametrize("module", ["rates", "cycle", "exact", "formation", "calibrate"])
def test_every_entry_module_enables_x64_without_the_package_init(module, tmp_path):
    """Importing a SUBMODULE first (never touching openmucf/__init__.py's body via a plain import) must
    still leave x64 on. Run in a subprocess so this process's already-configured jax cannot mask it."""
    proc = _run_isolated(f"""
        import importlib, jax
        importlib.import_module("openmucf.{module}")
        assert jax.config.read("jax_enable_x64") is True, "x64 OFF after importing openmucf.{module}"
        print("OK")
    """, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_shadowed_namespace_import_still_enables_x64(tmp_path):
    """THE regression: reproduce the shadow by placing a bare ``openmucf/`` directory (no __init__.py --
    exactly what a clone ROOT looks like) on sys.path[0], then import a submodule and sample.

    Under the pre-fix code this bound ``openmucf`` to a namespace package, skipped ``__init__.py``, and
    ran float32 silently. Now either the import path enables x64 anyway, or ``require_x64`` raises -- what
    must NEVER happen again is a quiet float32 run.
    """
    (tmp_path / "openmucf").mkdir()  # a directory named like the package, with no __init__.py
    proc = _run_isolated("""
        import jax, openmucf.calibrate as c
        shadowed = getattr(__import__("openmucf"), "__file__", None) is None
        x64 = bool(jax.config.read("jax_enable_x64"))
        print(f"shadowed={shadowed} x64={x64}")
        assert x64, "SILENT FLOAT32: shadowed import left jax_enable_x64 OFF"
        c.run_mcmc(num_warmup=20, num_samples=20, num_chains=1)   # must not raise
        print("OK")
    """, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_require_x64_raises_with_a_diagnosis(monkeypatch):
    """The layer-2 guard fires (and names the shadowing cause) when x64 is somehow off."""
    monkeypatch.setattr(_jaxcfg, "x64_enabled", lambda: False)
    with pytest.raises(RuntimeError, match="jax_enable_x64"):
        _jaxcfg.require_x64("a test")
    with pytest.raises(RuntimeError, match="SHADOWED"):
        _jaxcfg.require_x64("a test")


def test_every_sampler_entry_calls_the_guard():
    """Every file that builds an ``MCMC(NUTS(...))`` also calls ``require_x64`` in that file.

    Stated as a property over the tree rather than as a list of known entry points: a sampler added
    later without the guard fails here, which a list cannot catch. Whitespace is normalized first
    because the constructor is written across lines in some of them.
    """
    files = sorted(p for d in ("openmucf", "scripts") for p in (_REPO_ROOT / d).rglob("*.py"))
    entries = [p for p in files
               if "MCMC(NUTS(" in re.sub(r"\s+", "", p.read_text(encoding="utf-8"))]
    assert entries, "no sampler entry found at all -- this guard would pass vacuously"
    unguarded = sorted(str(p.relative_to(_REPO_ROOT)) for p in entries
                       if "require_x64(" not in p.read_text(encoding="utf-8"))
    assert not unguarded, f"NUTS entry with no x64 guard: {unguarded}"
