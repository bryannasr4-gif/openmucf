"""Adverse drills for the physical-time observation gate (no Geant4 dependency)."""

from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("clock_checker", ROOT / "cpp/clock/check_clock.py")
clock = module_from_spec(spec)
spec.loader.exec_module(clock)
transport_spec = spec_from_file_location("transport_checker", ROOT / "cpp/clock/check_transport.py")
transport = module_from_spec(transport_spec)
transport_spec.loader.exec_module(transport)
decay_spec = spec_from_file_location("atomic_decay_checker", ROOT / "cpp/clock/check_atomic_decay.py")
decay = module_from_spec(decay_spec)
with patch.dict("sys.modules", {"check_clock": clock, "check_transport": transport}):
    decay_spec.loader.exec_module(decay)


def atomic_records(atom_time: str = "0x1.f4p+7") -> list[str]:
    lines = ["RATES stopping_capture 0x1p+0 stopping_decay 0x1p+0 atomic_capture 0x1p+0 atomic_decay 0x1p+0"]
    for i in range(300):
        lines += [
            f"CASE atomic {i} t0 0x1.f4p+7 wait 0x0p+0 local 0x0p+0 branch 0 count 2 status 2 edep 0x0p+0",
            f"SEC {i} 0 gamma time 0x1.f4p+7 energy 0x1p+0 px 0x0p+0 py 0x0p+0 "
            "pz 0x1p+0 weight 0x1p+0 model -1 phase em",
            f"SEC {i} 1 MuPb208 time {atom_time} energy 0x0p+0 px 0x0p+0 py 0x0p+0 "
            "pz 0x0p+0 weight 0x1p+0 model -1 phase atom",
            f"RNG {i} fixed-test-engine-state",
        ]
    return lines


def test_physical_clock_gate_distinguishes_old_and_correct_time():
    assert clock.assess(atomic_records(), "atomic")["failures"] == 0
    doubled = atomic_records("0x1.f4p+8")
    assert clock.assess(doubled, "atomic")["failures"] == 300
    assert clock.assess(doubled, "atomic", old=True)["failures"] == 0
    assert clock.assess(atomic_records("0x0p+0"), "atomic")["failures"] == 300


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_observed_clock_is_not_a_pass(value):
    with pytest.raises(AssertionError, match="nonfinite"):
        clock.assess(atomic_records(value), "atomic")


def test_empty_missing_and_duplicate_observations_are_rejected():
    for records in ([], atomic_records()[:-1], atomic_records() + atomic_records()[-4:]):
        with pytest.raises(AssertionError):
            clock.assess(records, "atomic")


def test_invariant_comparison_retains_rng_and_kinematics():
    records = atomic_records()
    assert clock.invariant(records) == clock.invariant(atomic_records("0x1.f4p+8"))
    for old, new in (("fixed-test-engine-state", "extra-draw-state"), ("energy 0x1p+0", "energy 0x1p+1")):
        changed = [line.replace(old, new) for line in records]
        assert clock.invariant(records) != clock.invariant(changed)


def fixture_records(grid: list[tuple[int, float, float]]) -> list[str]:
    lines = ["RATES stopping_capture 0x1p+0 stopping_decay 0x1p+0 atomic_capture 0x1p+0 atomic_decay 0x1p+0"]
    cases = grid + [(i % 2, 2.0, 0.0) for i in range(100)]
    for i, (branch, wait, local) in enumerate(cases):
        phase = "decay" if branch else "capture"
        arrival = 250.0 + wait + (0.0 if branch else max(local, 0.0))
        lines += [
            f"CASE fixture {i} t0 {(250.0).hex()} wait {wait.hex()} local {local.hex()} "
            f"branch {branch} count 2 status 2 edep 0x0p+0",
            f"SEC {i} 0 gamma time {(250.375).hex()} energy 0x1p+0 phase em",
            f"SEC {i} 1 gamma time {arrival.hex()} energy 0x1p+0 phase {phase}",
            f"RNG {i} fixed-test-engine-state",
        ]
    return lines


def test_declared_fixture_grid_rejects_missing_or_replaced_point():
    grid = list(product((0, 1), (0.0, 0.25, 1.0, 2.0, 123.0), (-3.0, -1.0, -0.1, 0.0, 0.125, 3.0)))
    assert clock.assess(fixture_records(grid), "fixture")["failures"] == 0
    missing = grid[:1] + grid[2:]
    replaced = grid.copy()
    replaced[1] = replaced[0]
    for altered in (missing, replaced):
        with pytest.raises(AssertionError, match="fixture grid"):
            clock.assess(fixture_records(altered), "fixture")


def test_rate_readout_must_be_present_exactly_once():
    valid = atomic_records()
    for altered in (valid[1:], valid + valid[:1]):
        with pytest.raises(AssertionError, match="exactly one RATES"):
            clock.assess(altered, "atomic")


def test_transport_truncation_fails_even_with_both_branches():
    lines = []
    for i in range(500):
        branch = i % 2
        phase = "decay" if branch else "capture"
        lines += [
            f"BOUND_WAIT {i} 0x1p+0 branch {branch}",
            f"PRODUCT {i} 0 gamma phase {phase} time {(251.0).hex()} energy 0x1p+0",
        ]
    complete = transport.parse("\n".join(lines))
    prefix = transport.parse("\n".join(lines[:240]))
    assert transport.assess(complete, "stopping", 250.0, 500)["failures"] == 0
    assert transport.assess(prefix, "stopping", 250.0, 120)["failures"] == 0
    with pytest.raises(AssertionError, match="missing transport events"):
        transport.assess(prefix, "stopping", 250.0, 500)


def atomic_decay_records(*, old: bool = False) -> list[str]:
    grid = list(product((0.0, 0.25, 1.0, 2.0, 123.0), (-3.0, -1.0, -0.1, 0.0, 0.125, 3.0)))
    lines = ["RATES stopping_capture 0x1p+0 stopping_decay 0x1p+0 atomic_capture 0x1p+0 atomic_decay 0x1p+0"]
    for i, (wait, local) in enumerate(grid + [(2.5, -1.0)] * 4):
        branch = i % 2 if i >= 30 else 0
        phase = "decay" if branch else "capture"
        mode = "sampled" if i >= 30 else "controlled"
        elapsed = wait if branch else max(0.0, wait + local) if old else wait + max(0.0, local)
        lines += [
            f"WAIT {i} mode {mode} declared {wait.hex()} mean 0x1p+0 observed {wait.hex()}",
            f"CASE atom_decay {i} t0 {(250.0).hex()} wait {wait.hex()} local {local.hex()} "
            f"branch {branch} count 1 status 2 edep 0x0p+0",
            f"SEC {i} 0 gamma time {(250.0 + elapsed).hex()} energy 0x1p+0 phase {phase}",
            f"RNG {i} fixed-test-engine-state",
        ]
    return lines


def test_atomic_decay_gate_uses_independent_wait_and_rejects_old_composition():
    correct, old = atomic_decay_records(), atomic_decay_records(old=True)
    assert decay.assess(correct, 250.0, 4)["failures"] == 0
    assert decay.assess(old, 250.0, 4, old=True)["failures"] == 0
    assert decay.assess(old, 250.0, 4)["failures"] > 0
    altered = correct.copy()
    altered[1] = altered[1].replace("observed 0x0.0p+0", "observed 0x1p+0")
    with pytest.raises(AssertionError, match="independent GPIL"):
        decay.assess(altered, 250.0, 4)


def test_atomic_decay_gate_rejects_missing_wait_and_nonfinite_product():
    correct = atomic_decay_records()
    with pytest.raises(AssertionError, match="missing process/wait"):
        decay.assess(correct[:1] + correct[2:], 250.0, 4)
    altered = correct.copy()
    altered[3] = altered[3].replace(f"time {(250.0).hex()}", "time nan")
    with pytest.raises(AssertionError, match="nonfinite delivered"):
        decay.assess(altered, 250.0, 4)
