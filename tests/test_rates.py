"""Validation tests for the FAIR rate ledger (Phase 1)."""

import csv
import math

from openmucf import load_rates
from openmucf.rates import TARGETS_CSV, bibkeys


def test_ledger_loads_and_validates():
    rt = load_rates()  # raises ValueError on any provenance/schema problem
    assert len(rt) >= 10


def test_settled_values():
    rt = load_rates()
    assert math.isclose(rt.value("lambda_mu_decay"), 4.552e5, rel_tol=1e-3)
    assert math.isclose(rt.value("omega_s0"), 0.857, rel_tol=1e-9)
    assert math.isclose(rt.value("E_fusion"), 17.6, rel_tol=1e-9)
    assert rt["omega_s0"].source_bibkey == "Kamimura2023"


def test_every_rate_is_sourced_and_in_bib():
    rt = load_rates()
    known = bibkeys()
    for r in rt._rates.values():
        assert r.source_bibkey, f"{r.symbol} has no source"
        for key in r.source_bibkey.replace(";", ",").split(","):
            key = key.strip()
            if key:
                assert key in known, f"{r.symbol}: bibkey {key} missing from references.bib"


def test_contested_and_unverified_are_flagged_not_hidden():
    rt = load_rates()
    # The two single-quoted/placeholder formation points must be flagged for Phase-2 digitization.
    for sym in ("lambda_dtmu_lowT", "lambda_dt_transfer", "lambda_10_spinflip"):
        assert rt[sym].needs_verification is True


def test_nominal_vector_is_float64():
    rt = load_rates()
    v = rt.nominal_vector(["lambda_mu_decay", "omega_s0", "R_col", "lambda_f_dtmu"])
    assert str(v.dtype) == "float64"
    assert v.shape == (4,)
    # ordering preserved
    assert math.isclose(float(v[0]), 4.552e5, rel_tol=1e-3)


def test_validation_targets_present_and_well_formed():
    assert TARGETS_CSV.exists()
    with open(TARGETS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = {r["target_id"] for r in rows}
    # the two marquee Kou-Chen reproduction anchors and the Acceleron regime anchor
    assert {"V_kouchen_base", "V_kouchen_best", "A_acceleron_anomaly"} <= ids


def test_ledger_new_columns_validate(tmp_path):
    """The loader accepts the extended ledger; a bad `distribution` enum value raises."""
    from pathlib import Path

    import pytest

    from openmucf.rates import RATES_CSV

    r = load_rates()
    assert "lambda_c_liquid" in r
    assert r["eta_dtmu"].distribution == "asym_interval"
    assert r.dist_bounds("lambda_c_liquid") == (1.00e8, 1.45e8)

    text = Path(RATES_CSV).read_bytes().decode("utf-8").replace("\r\n", "\n")
    bad = text.replace(",asym_interval,1.0,5.0,,gas,D2", ",NOTADIST,1.0,5.0,,gas,D2")
    assert bad != text  # the target substring was actually present
    p = tmp_path / "bad_rates.csv"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        load_rates(csv_path=p)


def test_recommended_superseded_pair():
    """omega_s0 is `recommended`, omega_s0_legacy is `superseded`; exactly one `recommended` in the family."""
    r = load_rates()
    assert r["omega_s0"].recommendation == "recommended"
    assert r["omega_s0_legacy"].recommendation == "superseded"
    fam_recommended = [
        s
        for s in r.symbols()
        if r[s].recommendation == "recommended" and s.replace("_legacy", "") == "omega_s0"
    ]
    assert fam_recommended == ["omega_s0"]
