"""Tests for openmucf.interop -- the GEANT4/external-tool data bridge (v1 stub)."""

import json
import math

import pytest

from openmucf import formation, interop, load_rates
from openmucf.analytic import effective_sticking
from openmucf.rates import omega_fraction

RATES = load_rates()


def _expected_omega_s_eff(use_legacy=False):
    os0 = omega_fraction(RATES["omega_s0_legacy" if use_legacy else "omega_s0"])
    return float(effective_sticking(os0, RATES.value("R_col")))


def test_export_omega_s_eff_matches_analytic_and_is_flat():
    T = (100.0, 300.0, 800.0)
    phi = (1.0, 1.2, 1.45)
    tab = interop.export_omega_s_eff(RATES, T, phi)
    assert tab.axis_names == ("phi", "T")
    assert len(tab.values) == len(phi) and len(tab.values[0]) == len(T)
    expected = _expected_omega_s_eff()
    # v1 contract: omega_s^eff is condition-independent -> every cell equals the ledger value.
    for row in tab.values:
        for v in row:
            assert math.isclose(v, expected, rel_tol=1e-12)


def test_export_lambda_form_eff_thermal_shape_and_positive():
    T = (100.0, 300.0, 800.0)
    phi = (1.0, 1.45)
    tab = interop.export_lambda_form_eff_thermal(T, phi, F=1)
    assert tab.name == "lambda_form_eff_thermal"
    assert tab.unit == "s^-1"
    assert len(tab.values) == len(phi)
    for i, p in enumerate(phi):
        for j, t in enumerate(T):
            assert tab.values[i][j] > 0.0
            # matches the underlying formation model exactly
            assert math.isclose(tab.values[i][j], float(formation.lambda_dtmu(t, p, 1, 1.0)), rel_tol=1e-9)


def test_export_lambda_dtmu_thermal_is_deprecated_alias():
    """The old name warns (DeprecationWarning) but returns the renamed effective-rate table."""
    with pytest.deprecated_call():
        tab = interop.export_lambda_dtmu_thermal((300.0,), (1.2,), F=1)
    assert tab.name == "lambda_form_eff_thermal"


def test_export_lambda_dtmu_energy_peaks_near_resonance():
    E = [i * 1.0 / 200 for i in range(201)]  # 0..1 eV
    tab = interop.export_lambda_dtmu_energy(E, F=1)
    assert tab.axis_names == ("E_eV",)
    peak_i = max(range(len(E)), key=lambda k: tab.values[k])
    # F=1 measured Vesman resonance sits at 0.423 eV (Fujiwara 2000)
    assert abs(E[peak_i] - 0.423) < 0.03


def test_export_all_writes_csv_and_json(tmp_path):
    written = interop.export_all(RATES, tmp_path, fmt="both")
    assert set(written) == {"omega_s_eff", "lambda_form_eff_thermal", "lambda_dtmu_energy"}
    for paths in written.values():
        assert len(paths) == 2
        for p in paths:
            assert p.exists() and p.stat().st_size > 0
    # JSON round-trips to a valid payload carrying provenance
    payload = json.loads((tmp_path / "lambda_form_eff_thermal.json").read_text())
    assert payload["unit"] == "s^-1"
    assert "openmucf_version" in payload
    assert len(payload["axis_grids"]) == 2


def test_csv_long_format_row_count(tmp_path):
    tab = interop.export_lambda_form_eff_thermal((100.0, 300.0), (1.0, 1.2, 1.45), F=1)
    p = tab.to_csv(tmp_path / "t.csv")
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    assert lines[0] == "phi,T,lambda_form_eff_thermal"
    assert len(lines) == 1 + 3 * 2  # header + phi*T rows


def test_geant4_callables_are_plain_floats():
    api = interop.geant4_callables(RATES)
    ose = api["omega_s_eff"](phi=1.45, T=800.0)
    assert isinstance(ose, float)
    assert math.isclose(ose, _expected_omega_s_eff(), rel_tol=1e-12)
    thermal = api["lambda_form_eff"](phi=1.2, T=300.0, F=1)
    energy = api["lambda_form_eff"](E=0.423, F=1)
    assert isinstance(thermal, float) and thermal > 0.0
    assert isinstance(energy, float) and energy > 0.0
    # energy-resolved peak >> thermal average (resonance vs Maxwell tail)
    assert energy > thermal


def test_geant4_callables_both_keys():
    """The canonical `lambda_form_eff` and the legacy `lambda_dtmu` keys are the same callable."""
    api = interop.geant4_callables(RATES)
    assert "lambda_form_eff" in api and "lambda_dtmu" in api
    assert api["lambda_form_eff"] is api["lambda_dtmu"]
    a = api["lambda_form_eff"](phi=1.2, T=300.0, F=1)
    b = api["lambda_dtmu"](phi=1.2, T=300.0, F=1)
    assert a == b and isinstance(a, float) and a > 0.0


def test_ingest_spectrum_roundtrip_and_normalization(tmp_path):
    p = tmp_path / "neutron_tof.csv"
    p.write_text("# t_us, counts\n0.0,0.0\n1.0,10.0\n2.0,0.0\n")
    spec = interop.ingest_spectrum(p, kind="neutron_tof", source="unit-test", x_unit="us", y_unit="counts")
    assert len(spec) == 3
    assert spec.kind == "neutron_tof"
    assert math.isclose(spec.area(), 10.0, rel_tol=1e-9)  # triangle area = 0.5*base(2)*height(10)
    assert math.isclose(spec.normalized().area(), 1.0, rel_tol=1e-9)


def test_ingest_whitespace_and_comments(tmp_path):
    p = tmp_path / "spec.dat"
    p.write_text("# header\n0.0   1.0\n\n1.0   2.0\n2.0   1.0\n")
    spec = interop.ingest_spectrum(p, kind="x_ray")
    assert len(spec) == 3
    assert spec.x == (0.0, 1.0, 2.0)
    assert spec.y == (1.0, 2.0, 1.0)
