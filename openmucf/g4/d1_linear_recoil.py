"""Current experimental inclusive tree prescription, separate from historical harvests.

The coefficient is leading in binding and linear in eta=m_mu/M. See
``docs/low-z-bound-muon-decay.md`` for the observable, derivation and unassigned
remainders. This is not the legacy getter, an all-orders reduced-mass width, or
a physical-accuracy certificate. No historical policy is reinterpreted here.
"""

from __future__ import annotations

import math
from pathlib import Path

from openmucf.g4 import d1_boundrates as historical
from openmucf.g4.d1_lifetime import ALPHA
from openmucf.g4.sources import d1_nuclear_capture as d1

PRESCRIPTION_ID = "low-z-inclusive-tree-linear-recoil-v1"


def decay_rate(z: int, muon_mass: float, nuclear_mass: float) -> float:
    """Decay rate in ns^-1, or NaN on unsupported Z/masses/eta.

    The eta < 1/2 guard enforces internal suppression consistency only. The
    physical expansion requires eta << 1; the guard assigns no error bound.
    Positive masses whose quotient underflows to zero use the static limit.
    """
    if type(z) is not int or not 2 <= z <= 13:
        return math.nan
    for mass in (muon_mass, nuclear_mass):
        if isinstance(mass, bool) or not isinstance(mass, (int, float)):
            return math.nan
        try:
            if not math.isfinite(mass) or mass <= 0:
                return math.nan
        except OverflowError:
            return math.nan
    eta = muon_mass / nuclear_mass
    if not math.isfinite(eta) or eta >= 0.5:
        return math.nan
    x = z * ALPHA
    return (1.0 - x * x * (0.5 - eta)) * (0.45517005 / float(d1.MICROSECOND))


def current_pair(root: Path, z: int, a: int) -> historical.Pair:
    """Deterministic current pair using the archived mass inputs and capture resolver.

    The old decay oracle supplies constants/masses only; it is not an oracle for
    this prescription. Unsupported keys and unusable capture/decay are refused.
    No current transport-harvest claim follows from evaluating this pair.
    """
    if type(z) is not int or type(a) is not int or not historical.in_domain(z, a):
        raise historical.BoundRateError(f"Z={z} A={a} lies outside the domain")
    inputs = historical.Reference(root)
    if (z, a) not in inputs.oracle.light:
        raise historical.BoundRateError(f"no archived mass input for Z={z} A={a}")
    mass = inputs.oracle.light[(z, a)][0]
    capture = d1.capture_rate(z, a, inputs.helper.capture_records, inputs.model)
    decay = decay_rate(z, inputs.mu_mass, mass)
    free = 0.45517005 / float(d1.MICROSECOND)
    if not math.isfinite(capture) or capture < 0 or not 0 < decay <= free:
        raise historical.BoundRateError(f"unusable current pair for Z={z} A={a}")
    return historical.Pair(z, a, mass, capture, decay)
