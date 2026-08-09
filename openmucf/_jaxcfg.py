"""Single source of the MANDATORY JAX float64 configuration, plus a hard guard.

Rates in this package span ~7 decades (``lambda_f`` ~1e12 vs ``lambda_0`` ~1e5), so float64 is not an
optimisation -- a float32 run silently loses the small channels and quietly biases every MCMC posterior.
``jax_enable_x64`` used to be set only in ``openmucf/__init__.py``. That is one line short of safe:

    A directory that contains a CLONE of this repository shadows the installed package. Python sees
    ``<cwd>/openmucf/`` (the clone ROOT, which has no ``__init__.py``) as a NAMESPACE package and binds
    the top-level name to it, while an editable install's meta-path finder still resolves the SUBMODULES
    to the real files. Result: ``openmucf.calibrate`` imports and runs, but ``openmucf/__init__.py``
    NEVER EXECUTES -- so x64 was never enabled and the chain silently ran in float32.

That is not hypothetical: it happened during the 2026-07-23 cross-architecture audit and produced
non-reproducing numbers before the layout was diagnosed. The failure is silent, architecture-independent,
and reachable by any third party who runs a script from a directory holding a clone of this repo.

Defence is therefore two-layered:

1. **Importing this module enables x64.** It is imported by ``rates`` (the dependency root) and by the
   three jax-using leaf modules that import nothing else from the package (``cycle``, ``exact``,
   ``formation``), so EVERY import path into the package enables x64 before any array is created.
2. :func:`require_x64` hard-fails at the entry to any sampler/solver, so if a future refactor breaks
   layer 1 the run STOPS with a diagnosis instead of returning quietly-wrong numbers.
"""

from __future__ import annotations

import jax as _jax

# Rates span ~7 decades (lambda_f ~1e12 vs lambda_0 ~1e5); float64 is mandatory.
_jax.config.update("jax_enable_x64", True)


def x64_enabled() -> bool:
    """True iff JAX is currently configured for float64."""
    return bool(_jax.config.read("jax_enable_x64"))


def require_x64(context: str = "this computation") -> None:
    """Hard-fail unless float64 is on. Called at the entry of every sampler/solver.

    Raises ``RuntimeError`` with the shadowing diagnosis, because that is by far the most likely cause
    (a plain ``import openmucf`` cannot reach this state).
    """
    if not x64_enabled():
        raise RuntimeError(
            f"openmucf requires jax_enable_x64=True for {context}; it is OFF. The usual cause is a "
            "SHADOWED package import: running a script from a directory that contains a clone of this "
            "repository binds 'openmucf' to a namespace package, so openmucf/__init__.py never runs. "
            "Run from a different working directory (or 'import openmucf' first and check "
            "openmucf.__file__ is not None). Rates span ~7 decades, so float32 results are invalid."
        )
