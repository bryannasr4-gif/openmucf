"""OpenMuCF -- open FAIR rate ledger + differentiable muon-catalyzed-fusion
cycle/energy auditor.

Status: v1 complete (spine: FAIR ledger + analytic + differentiable cycle ODE + net-electrical
energy + global UQ auditor + Bayesian calibration + validation). See README.md and MODEL_SPEC.md; the
Phase-3 omega_s^eff(phi,T,c_t) reactivation surrogate is the next capability.
"""

import jax as _jax

# Rates span ~7 decades (lambda_f ~1e12 vs lambda_0 ~1e5); float64 is mandatory.
_jax.config.update("jax_enable_x64", True)

from . import analytic, cycle, energy, formation, interop, uq  # noqa: E402, F401
from .energy import EnergyChain  # noqa: E402
from .rates import Rate, RatesTable, bibkeys, load_rates, omega_fraction  # noqa: E402

__version__ = "1.0.0"
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
    "__version__",
]
