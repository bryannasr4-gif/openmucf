"""Adverse drills for the physical-time observation gate (no Geant4 dependency)."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("clock_checker", ROOT / "cpp/clock/check_clock.py")
clock = module_from_spec(spec)
spec.loader.exec_module(clock)


def atomic_records(atom_time: str = "0x1.f4p+7") -> list[str]:
    lines = []
    for i in range(300):
        lines += [
            f"CASE atomic {i} t0 0x1.f4p+7 wait 0x0p+0 local 0x0p+0 branch 0 count 2 status 2 edep 0x0p+0",
            f"SEC {i} 0 gamma time 0x1.f4p+7 energy 0x1p+0 px 0x0p+0 py 0x0p+0 pz 0x1p+0 weight 0x1p+0 model -1 phase em",
            f"SEC {i} 1 MuPb208 time {atom_time} energy 0x0p+0 px 0x0p+0 py 0x0p+0 pz 0x0p+0 weight 0x1p+0 model -1 phase atom",
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
    for records in ([], atomic_records()[:-1], atomic_records()+atomic_records()[-4:]):
        with pytest.raises(AssertionError):
            clock.assess(records, "atomic")


def test_invariant_comparison_retains_rng_and_kinematics():
    records = atomic_records()
    assert clock.invariant(records) == clock.invariant(atomic_records("0x1.f4p+8"))
    for old, new in (("fixed-test-engine-state", "extra-draw-state"), ("energy 0x1p+0", "energy 0x1p+1")):
        changed = [line.replace(old,new) for line in records]
        assert clock.invariant(records) != clock.invariant(changed)
