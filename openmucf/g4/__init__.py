"""openmucf.g4 -- the ``G4MuonicData`` external-data layer (see ``FORMAT_SPEC.md``).

Two layers, one source of truth: ``spec`` implements the Layer-1 ``.g4dat`` grammar that a
transport code reads with nothing but its standard library, and ``provenance`` implements the
Layer-2 ``*.prov.json`` schema that carries the bibliography, the uncertainty type and the
evaluation identity Layer 1 deliberately does not. ``emit`` packages the two into the deterministic
archive that ships, and writes the registration snippet that describes it.

This subpackage is **self-contained by rule**: it may not import ``openmucf.cycle``, ``.uq``,
``.calibrate`` or ``.formation``. The rule is enforced by a test, not by this docstring, so that the
data layer can be lifted out into its own distribution without dragging a fusion-kinetics stack
behind it. Nothing is imported eagerly here; the submodules load on first attribute access.
"""

from __future__ import annotations

import importlib as _importlib
from types import ModuleType

_SUBMODULES = ("spec", "provenance", "emit")


def __getattr__(name: str) -> ModuleType:
    """PEP 562 lazy loader: ``openmucf.g4.spec`` imports on first access, then caches."""
    if name in _SUBMODULES:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))


__all__ = ["emit", "provenance", "spec"]
