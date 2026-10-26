"""Current inclusive linear-recoil model; old harvests keep their separate oracle."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import pytest

from openmucf.g4 import d1_boundrates as historical
from openmucf.g4 import d1_lifetime as legacy
from openmucf.g4 import d1_linear_recoil as current
from openmucf.g4.sources import d1_nuclear_capture as d1

ROOT = Path(__file__).resolve().parents[1]
FREE = 0.45517005 / d1.MICROSECOND


@pytest.mark.parametrize("z", [2, 6, 13])
def test_static_limit_and_independent_affine_recoil_slopes(z: int) -> None:
    # The matched kinetic operator has coefficient -1/2; differentiating its
    # Coulomb reduced-mass matrix element at eta=0 gives positive unit slope.
    z2 = Fraction(z * legacy.ALPHA) ** 2
    static = 1 - z2 / 2
    q0 = current.decay_rate(z, math.ulp(0.0), float.fromhex("0x1.fffffffffffffp+1023")) / FREE
    assert q0 == pytest.approx(float(static), abs=3e-16)
    samples = [(eta, current.decay_rate(z, 1.0, 1.0 / eta) / FREE)
               for eta in (0.001, 0.01, 0.125, 0.375)]
    for eta, q in samples:
        assert (q - q0) / float(z2) / eta == pytest.approx(1, abs=2e-9)
        assert q0 < q <= 1
    # Independent secants also exclude any all-orders reduced-mass resummation.
    for (a, qa), (b, qb) in zip(samples, samples[1:], strict=False):
        assert (qb - qa) / (b - a) == pytest.approx(float(z2), rel=1e-9)


def test_archived_getter_delta_and_capture_forwarding() -> None:
    old = historical.Reference(ROOT)
    count = 0
    for z, a in old.oracle.light:
        if not historical.in_domain(z, a):
            continue
        mass, decay = old.oracle.light[z, a]
        got = current.decay_rate(z, old.mu_mass, mass)
        expected_delta = 1.06 * (z * legacy.ALPHA) ** 2 * old.mu_mass / mass * FREE
        assert got - decay == pytest.approx(expected_delta, abs=2e-19)
        capture = d1.capture_rate(z, a, old.helper.capture_records, old.model)
        if capture >= 0:
            pair = current.current_pair(ROOT, z, a)
            assert pair.capture == capture
            assert pair.mass == mass and pair.decay == got
            assert pair.lifetimes[:2] == (1 / (capture + got), 1 / got)
            count += 1
        else:
            with pytest.raises(historical.BoundRateError, match="unusable current pair"):
                current.current_pair(ROOT, z, a)
    assert count > 0


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf, True, "1", None, 10**1000])
def test_direct_helper_refuses_invalid_masses(bad: object) -> None:
    assert math.isnan(current.decay_rate(6, bad, 1000.0))  # type: ignore[arg-type]
    assert math.isnan(current.decay_rate(6, 1.0, bad))  # type: ignore[arg-type]


@pytest.mark.parametrize("z", [-1, 0, 1, 14, 10**1000, True, 6.0, "6", None])
def test_direct_helper_refuses_unsupported_or_mistyped_z(z: object) -> None:
    assert math.isnan(current.decay_rate(z, 1.0, 1000.0))  # type: ignore[arg-type]


def test_eta_boundary_overflow_and_finite_extremes() -> None:
    just_below = math.nextafter(0.5, 0)
    assert 0 < current.decay_rate(13, just_below, 1) <= FREE
    for mu, mass in [(0.5, 1), (math.nextafter(0.5, 1), 1), (1, 1), (1e308, 1e-308)]:
        assert math.isnan(current.decay_rate(6, mu, mass))
    # Same finite ratio at extreme absolute scales must have the same result.
    assert current.decay_rate(6, 1e-300, 1e-299) == current.decay_rate(6, 1e300, 1e301)
    assert current.PRESCRIPTION_ID == "low-z-inclusive-tree-linear-recoil-v1"


@pytest.mark.parametrize("z,a", [(1, 1), (14, 28), (6, 0), (6, 5), (True, 2), (6, 12.0)])
def test_current_pair_domain(z: int, a: int) -> None:
    with pytest.raises(historical.BoundRateError, match="outside the domain"):
        current.current_pair(ROOT, z, a)


def test_current_pair_requires_an_observed_mass_input() -> None:
    with pytest.raises(historical.BoundRateError, match="no archived mass input"):
        current.current_pair(ROOT, 6, 1000)
