"""Independent-field drills for the reaction-product formation-time observation gate."""

from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("formation_checker", ROOT / "cpp/clock/check_formation.py")
formation = module_from_spec(spec)
spec.loader.exec_module(formation)


def records(variant="new"):
    origin = 250.0
    lines = [
        f"INPUT 0 mode controlled origin {origin.hex()} Z 82 A 208",
        "AUDIT_FRAGMENT event 0 creation 0x0p+0",
        "FINAL 0 count 17 status 2 edep 0x0p+0 dependency_calls 1",
        f"SECONDARY 0 0 name nu_mu time {origin.hex()} energy 0x1p+0",
    ]
    for i, (tof, local) in enumerate(product((-1.0, 0.0, 1.0, 7.0), (-2.0, 0.0, 0.125, 3.0))):
        elapsed = max(local, 0.0)
        if variant == "old":
            elapsed = tof
        elif variant == "zero":
            elapsed = 0.0
        elif variant == "fallback" and local == 0.0:
            elapsed = tof
        lines += [
            f"META 0 {i} name gamma tof {tof.hex()} formation {local.hex()}",
            f"AUDIT_RP event 0 index {i} name gamma tof {tof.hex()} "
            f"formation {local.hex()} origin {origin.hex()}",
            f"SECONDARY 0 {i + 1} name gamma time {(origin + elapsed).hex()} energy 0x1p+0",
        ]
    return lines + ["ENGINE 0 complete-test-engine-state"]


def test_formation_gate_rejects_old_zero_and_zero_formation_fallback():
    assert formation.assess_wrapper(records(), "controlled", 250.0, 1)["failures"] == 0
    for variant in ("old", "zero", "fallback"):
        assert formation.assess_wrapper(records(variant), "controlled", 250.0, 1)["failures"] > 0
        assert formation.invariant(records(variant)) == formation.invariant(records())


def test_formation_gate_requires_independent_metadata_and_finite_times():
    missing = [line for line in records() if not line.startswith("META 0 13 ")]
    with pytest.raises(AssertionError, match="missing supplied"):
        formation.assess_wrapper(missing, "controlled", 250.0, 1)
    altered = [line.replace(f"time {(250.0).hex()}", "time nan") for line in records()]
    with pytest.raises(AssertionError, match="nonfinite"):
        formation.assess_wrapper(altered, "controlled", 250.0, 1)


def test_formation_gate_requires_declared_count_and_unchanged_rng():
    with pytest.raises(AssertionError, match="missing declared wrapper"):
        formation.assess_wrapper(records(), "controlled", 250.0, 2)
    extra_draw = [line.replace("complete-test-engine-state", "extra-draw-state") for line in records()]
    assert formation.invariant(extra_draw) != formation.invariant(records())
