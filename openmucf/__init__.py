"""OpenMuCF -- open FAIR rate ledger + differentiable muon-catalyzed-fusion
cycle/energy auditor.

Status: v1 complete (spine: FAIR ledger + analytic + differentiable cycle ODE + net-electrical
energy + global UQ auditor + Bayesian calibration + validation). See README.md and MODEL_SPEC.md; the
Phase-3 omega_s^eff(phi,T,c_t) reactivation surrogate is the next capability. Validation is class-tiered
(see VALIDATION.md; the independent-prediction targets currently fail by design against the placeholder
formation model).
"""

import importlib as _importlib

import jax as _jax

# Rates span ~7 decades (lambda_f ~1e12 vs lambda_0 ~1e5); float64 is mandatory.
_jax.config.update("jax_enable_x64", True)

from . import analytic, cycle, energy, formation, interop, uq  # noqa: E402, F401
from .energy import EnergyChain  # noqa: E402
from .rates import Rate, RatesTable, bibkeys, load_rates, omega_fraction  # noqa: E402

__version__ = "1.1.0"

# Heavy public submodules loaded lazily (PEP 562) on first attribute access, so a bare
# `import openmucf` never pays the numpyro/statistics import cost. Accessing e.g.
# `openmucf.calibrate` imports it on demand and caches it on the package namespace.
_LAZY_SUBMODULES = (
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


def __getattr__(name: str):
    """PEP 562 lazy loader for the heavy public submodules (see `_LAZY_SUBMODULES`)."""
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache: subsequent access bypasses __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))


__all__ = [
    "Rate",
    "RatesTable",
    "EnergyChain",
    "bibkeys",
    "load_rates",
    "omega_fraction",
    "analytic",
    "cycle",
    "energy",
    "formation",
    "interop",
    "uq",
    *_LAZY_SUBMODULES,
    "__version__",
]
