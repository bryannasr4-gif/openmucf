"""WS-S: systems.py -- the full energy-balance graph + the Q Rosetta stone.

These tests lock: the differentiable graph's gradient SIGNS (jax.grad); the G-legacy degenerate special
case reproduces the frozen v1 EnergyChain breakevens to rel 1e-12; the G-Kelly cross-basis value is the
faithful Eq.(2)+Table-1 reproduction (15.69%), which lands JUST ABOVE the pre-registered band -- a
documented finding, not tuned (I2); every QBasis round-trips its own reference conversion; rosetta_table
covers all four bases; the two flagged knobs (breeding credit, recirculating fraction) are degenerate-off;
and SYSTEMS.md + its manifest regenerate deterministically and verify.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import jax
import pytest

import openmucf
from openmucf import provenance
from openmucf.energy import EnergyChain
from openmucf.systems import (
    BASES,
    KELLY,
    REFERENCE_BASIS,
    SystemChain,
    _native_value,
    _native_ykc,
    q_net,
    q_sci,
    rosetta_table,
)

REPO = Path(openmucf.__file__).resolve().parent.parent
_SCRIPT = REPO / "scripts" / "generate_systems.py"


def _load_generator():
    """Import the generator by path (file I/O is guarded behind main(), so import is inert)."""
    spec = importlib.util.spec_from_file_location("_gen_systems", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------------
# Differentiable graph: gradient signs (the graph is a real jnp function of every knob)
# --------------------------------------------------------------------------------------------------
def test_q_net_gradient_signs():
    """q_net is monotone increasing in x_mu, eta_acc, eta_thermal, blanket_M, breeding_credit_MeV and
    decreasing in E_mu_GeV, recirc_fraction -- checked by autodiff on the graph."""
    sc = SystemChain()
    x = 150.0
    assert float(jax.grad(lambda v: q_net(sc, v))(x)) > 0.0  # d/dx_mu
    assert float(jax.grad(lambda v: q_net(replace(sc, eta_acc=v), x))(sc.eta_acc)) > 0.0
    assert float(jax.grad(lambda v: q_net(replace(sc, eta_thermal=v), x))(sc.eta_thermal)) > 0.0
    assert float(jax.grad(lambda v: q_net(replace(sc, blanket_M=v), x))(sc.blanket_M)) > 0.0
    assert float(jax.grad(lambda v: q_net(replace(sc, breeding_credit_MeV=v), x))(0.0)) > 0.0
    assert float(jax.grad(lambda v: q_net(replace(sc, recirc_fraction=v), x))(0.0)) < 0.0
    assert float(jax.grad(lambda v: q_net(replace(sc, E_mu_GeV=v), x))(sc.E_mu_GeV)) < 0.0


def test_q_sci_gradient_signs():
    """q_sci rises with x_mu and the breeding credit, falls with E_mu; it is efficiency-free (no eta,
    no blanket_M dependence)."""
    sc = SystemChain()
    x = 150.0
    assert float(jax.grad(lambda v: q_sci(sc, v))(x)) > 0.0
    assert float(jax.grad(lambda v: q_sci(replace(sc, breeding_credit_MeV=v), x))(0.0)) > 0.0
    assert float(jax.grad(lambda v: q_sci(replace(sc, E_mu_GeV=v), x))(sc.E_mu_GeV)) < 0.0
    # q_sci carries no efficiency chain: gradient wrt eta_acc / blanket_M is exactly zero
    assert float(jax.grad(lambda v: q_sci(replace(sc, eta_acc=v), x))(sc.eta_acc)) == 0.0
    assert float(jax.grad(lambda v: q_sci(replace(sc, blanket_M=v), x))(sc.blanket_M)) == 0.0


# --------------------------------------------------------------------------------------------------
# G-legacy: the degenerate special case reproduces the frozen v1 EnergyChain (no-tuning anchor)
# --------------------------------------------------------------------------------------------------
def test_g_legacy_breakevens_reproduce_v1():
    """With the flagged knobs off, SystemChain breakevens equal EnergyChain's to rel 1e-12 (in fact
    bit-identical): scientific ~284.09 and net-electrical ~2367.42."""
    sc, ec = SystemChain(), EnergyChain()
    assert math.isclose(sc.breakeven_xmu_sci(), ec.breakeven_xmu_sci(), rel_tol=1e-12)
    assert math.isclose(sc.breakeven_xmu_net(), ec.breakeven_xmu_net(), rel_tol=1e-12)
    # the actual v1 anchors (5000/17.6 and 5000/(17.6*0.40*0.30))
    assert math.isclose(sc.breakeven_xmu_sci(), 5000.0 / 17.6, rel_tol=1e-12)
    assert math.isclose(sc.breakeven_xmu_net(), 5000.0 / (17.6 * 0.40 * 0.30), rel_tol=1e-12)


def test_g_legacy_q_values_reproduce_v1():
    """q_sci/q_net at defaults equal the frozen EnergyChain.Q_sci/Q_net_electrical for arbitrary x_mu."""
    sc, ec = SystemChain(), EnergyChain()
    for x in (113.0, 150.0, 284.0, 500.0):
        assert math.isclose(float(q_sci(sc, x)), ec.Q_sci(x), rel_tol=1e-12)
        assert math.isclose(float(q_net(sc, x)), ec.Q_net_electrical(x), rel_tol=1e-12)


# --------------------------------------------------------------------------------------------------
# The eta_acc self-correction (the headline finding): 2367 -> 3946 at eta_acc 0.30 -> 0.18
# --------------------------------------------------------------------------------------------------
def test_eta_acc_self_correction_3946():
    """Correcting eta_acc 0.30 -> 0.18 (Kelly PSI-measured) moves the net-electrical breakeven to
    5000/(17.6*0.40*0.18); the directly-computed value rounds to 3946 (never the double-rounded 3945),
    and the v1 code default is unchanged at 0.30."""
    be = SystemChain(eta_acc=0.18).breakeven_xmu_net()
    assert math.isclose(be, 5000.0 / (17.6 * 0.40 * 0.18), rel_tol=1e-12)
    assert round(be) == 3946
    assert f"{be:.0f}" == "3946"
    # exactly linear in 1/eta_acc: the ratio is 0.30/0.18 = 5/3
    assert math.isclose(be / SystemChain(eta_acc=0.30).breakeven_xmu_net(), 0.30 / 0.18, rel_tol=1e-12)
    # the v1 default in code is untouched this wave
    assert SystemChain().eta_acc == 0.30


# --------------------------------------------------------------------------------------------------
# G-Kelly: cross-basis validation against Kelly-Hart-Rose 2021 Eq.(2) + Table 1
# --------------------------------------------------------------------------------------------------
def test_g_kelly_reproduces_eq2_from_cited_numbers():
    """KELLY.q_elec() reproduces Kelly Eq.(2) evaluated on the independently-written Table-1 numbers."""
    F, H, B = 2991.0, 3743.0, 3606.0  # row (A); rows (B)-(E)=2664+23+526+530; row (G)
    eta_mu, eta_rec, eta_acc, eta_heat = 0.50, 1.00, 0.18, 0.60
    expected = ((F * eta_mu + H * eta_rec) / B) * eta_acc * eta_heat
    assert math.isclose(KELLY.q_elec(), expected, rel_tol=1e-12)
    assert math.isclose(KELLY.q_elec(), 0.15689, abs_tol=5e-5)


def test_g_kelly_band_is_a_documented_finding_not_tuned():
    """The faithful reproduction (15.69%) lands JUST ABOVE the pre-registered band [12.6%, 15.4%]. Per I2
    this is a documented finding (Kelly's max-Q Table-1 config vs his 14% figure-3 curve headline), NOT a
    value tuned to hit 14%. This test LOCKS that we did not tune down into the band."""
    band_hi = 15.4  # pre-registered upper edge of [12.6%, 15.4%]
    pct = 100.0 * KELLY.q_elec()
    assert pct > band_hi  # outside (above), reported as a finding -- never adjusted to land inside
    assert band_hi < pct < 15.9  # just above the upper edge (0.29pp), not a gross discrepancy
    # scientific-gain reference for Kelly's config (X_mu=150): x_mu*17.6/E_mu with E_mu=4.70 GeV
    assert math.isclose(KELLY.q_sci(), 150.0 * 17.6 / 4700.0, rel_tol=1e-12)


def test_g_kelly_q_elec_affine_in_x_mu():
    """Kelly's Q_elec is affine (not linear) in X_mu: fusion heat scales, recoverable heat H is fixed --
    reproducing his figure-3 curve (rising, sub-linear)."""
    lo, hi = KELLY.q_elec(100.0), KELLY.q_elec(600.0)
    assert lo < KELLY.q_elec(150.0) < hi
    # sub-linear: doubling fusions/muon less than doubles Q_elec (H offset dominates at low X_mu)
    assert KELLY.q_elec(300.0) < 2.0 * KELLY.q_elec(150.0)


# --------------------------------------------------------------------------------------------------
# The Q Rosetta stone: bases round-trip; the table covers all four
# --------------------------------------------------------------------------------------------------
def test_every_basis_round_trips_to_reference():
    """Each QBasis.convert_to_reference maps its own native value back to the efficiency-free scientific
    gain reference (exact algebra; for Kelly, his cited electrical multiplier)."""
    sc = SystemChain()
    x = 150.0
    ref = float(q_sci(sc, x))
    for name, basis in BASES.items():
        if name == "kelly_Q_elec":
            native = KELLY.q_elec()
            got = basis.convert_to_reference(native, sc)
            assert math.isclose(got, KELLY.q_sci(), rel_tol=1e-9)  # Kelly's own reference gain
        else:
            native = float(_native_value(name, sc, x))
            got = float(basis.convert_to_reference(native, sc))
            assert math.isclose(got, ref, rel_tol=1e-9), (name, got, ref)


def test_q_net_basis_conversion_is_exact_inverse():
    """The q_net_v1 basis conversion genuinely inverts the efficiency chain (not a trivial identity)."""
    sc = SystemChain()
    x = 284.0
    native = float(q_net(sc, x))
    ref = float(q_sci(sc, x))
    assert native < ref  # net gain is far below the scientific gain
    assert math.isclose(BASES["q_net_v1"].convert_to_reference(native, sc), ref, rel_tol=1e-12)


def test_rosetta_table_covers_four_bases():
    """rosetta_table yields one row per X_mu with all four bases + the reference; deterministic."""
    grid = [113.0, 150.0, 284.0, 500.0]
    rows = rosetta_table(grid)
    assert [r["x_mu"] for r in rows] == grid
    for r in rows:
        for key in ("q_sci_v1", "q_net_v1", "kelly_Q_elec", "ykc_efficiency_free", "reference_q_sci"):
            assert key in r
        # the three SystemChain-native bases collapse to the reference; q_net is far below it
        assert math.isclose(r["q_sci_v1"], r["reference_q_sci"], rel_tol=1e-12)
        assert math.isclose(r["ykc_efficiency_free"], r["reference_q_sci"], rel_tol=1e-12)
        assert r["q_net_v1"] < r["reference_q_sci"]
    assert rosetta_table(grid) == rows  # deterministic
    assert set(BASES) == {"q_sci_v1", "q_net_v1", "kelly_Q_elec", "ykc_efficiency_free"}
    assert REFERENCE_BASIS == "q_sci_v1"


# --------------------------------------------------------------------------------------------------
# The two flagged knobs are degenerate-off (breeding credit, recirculating fraction)
# --------------------------------------------------------------------------------------------------
def test_breeding_credit_flag_default_off_equals_v1():
    """breeding_credit_MeV defaults to 0 -> E_per_fusion == E_fusion == v1; turning it on raises both
    gains monotonically."""
    sc = SystemChain()
    assert sc.breeding_credit_MeV == 0.0
    assert sc.E_per_fusion_MeV == sc.E_fusion_MeV
    assert math.isclose(float(q_net(sc, 150.0)), EnergyChain().Q_net_electrical(150.0), rel_tol=1e-12)
    hot = SystemChain(breeding_credit_MeV=8.4)  # Kelly's fusion-neutron breeding credit
    assert float(q_net(hot, 150.0)) > float(q_net(sc, 150.0))
    assert float(q_sci(hot, 150.0)) > float(q_sci(sc, 150.0))


def test_recirc_fraction_degenerate_at_zero():
    """recirc_fraction defaults to 0 (v1-degenerate); a positive fraction reduces net output linearly and
    leaves the efficiency-free scientific gain untouched."""
    sc = SystemChain()
    assert sc.recirc_fraction == 0.0
    assert math.isclose(float(q_net(sc, 150.0)), EnergyChain().Q_net_electrical(150.0), rel_tol=1e-12)
    r = SystemChain(recirc_fraction=0.25)
    assert math.isclose(float(q_net(r, 150.0)), 0.75 * float(q_net(sc, 150.0)), rel_tol=1e-12)
    # the efficiency-free scientific gain is untouched by recirculation
    assert math.isclose(float(q_sci(r, 150.0)), float(q_sci(sc, 150.0)), rel_tol=1e-12)


# --------------------------------------------------------------------------------------------------
# Packaging + doc/manifest determinism
# --------------------------------------------------------------------------------------------------
def test_systems_is_lazy_public_api():
    """systems is a lazily-loaded public submodule: exported in __all__ but not eager-imported
    (the PEP 562 __getattr__ resolves it on first access; see tests/test_packaging.py for the
    deterministic no-eager-load and wall-time guards)."""
    assert "systems" in getattr(openmucf, "__all__", [])
    assert openmucf.systems.__name__ == "openmucf.systems"


def test_ykc_equals_q_sci_at_default_credit():
    """The Rosetta identity finding: YKC's efficiency-free gain equals our q_sci when no breeding credit
    is flagged (default), and diverges (by exactly the credit ratio) when one is."""
    sc = SystemChain()
    assert math.isclose(float(_native_ykc(sc, 150.0)), float(q_sci(sc, 150.0)), rel_tol=1e-12)
    hot = SystemChain(breeding_credit_MeV=8.4)
    assert float(_native_ykc(hot, 150.0)) < float(q_sci(hot, 150.0))  # ykc is credit-free


def test_systems_md_regenerates_byte_identical():
    """SYSTEMS.md rebuilt from the generator equals the committed file byte-for-byte (LF-normalized)."""
    gen = _load_generator()
    rebuilt = gen.build_markdown(gen.build_headline())
    committed = (REPO / "SYSTEMS.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert rebuilt == committed


def test_systems_manifest_verifies():
    """The committed SYSTEMS_MANIFEST.json verifies against the committed SYSTEMS.md (no doc drift)."""
    failures = provenance.check_manifest(REPO / "SYSTEMS_MANIFEST.json", repo_root=REPO)
    assert failures == [], failures


def test_kelly_numbers_are_paper_cited_constants():
    """Guard the paper-cited Kelly primitives so a silent edit to any digit fails a test."""
    assert (KELLY.F_fusion_MeV_ref, KELLY.H_recover_MeV, KELLY.B_beam_MeV) == (2991.0, 3743.0, 3606.0)
    assert (KELLY.eta_mu, KELLY.eta_rec, KELLY.eta_acc, KELLY.eta_heat) == (0.50, 1.00, 0.18, 0.60)
    assert KELLY.x_mu_ref == 150.0 and KELLY.E_mu_GeV == 4.70
    # H is exactly the sum of Table-1 rows (B)-(E)
    assert KELLY.H_recover_MeV == 2664.0 + 23.0 + 526.0 + 530.0


@pytest.mark.parametrize("bad_basis", ["not_a_basis"])
def test_native_value_rejects_unknown_basis(bad_basis):
    with pytest.raises(KeyError):
        _native_value(bad_basis, SystemChain(), 150.0)
