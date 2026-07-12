"""WS-Y1: the Parisi-Rutkowski Ac-225 reproduction (scripts/parisi_ac225.py).

These tests lock the forward reproduction of arXiv:2511.20951v2's headline: the entered factor chain
reproduces the ~20 mg/yr Ac-225 number within +/-5% (invariant I2 -- a FORWARD computation, not a tuned
fit), matches the OpenMC Table-I value to <1%, every factor row carries a non-empty citation (invariant
I3), the P_fus=564 W cross-check holds, and the MUON_COST.md tier cross-reference points at rows that
actually exist in the repo's muon-cost ledger.

The script imports with NO side effects (all I/O is behind ``main()``), so these tests exercise the pure
factor arithmetic directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_parisi():
    """Import scripts/parisi_ac225.py (scripts/ is not a package) as a module, no side effects."""
    path = REPO / "scripts" / "parisi_ac225.py"
    spec = importlib.util.spec_from_file_location("parisi_ac225", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec so @dataclass can resolve its own module
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def parisi():
    return _load_parisi()


def test_reproduces_headline_within_5pct(parisi):
    """The forward factor chain reproduces the 20 mg/yr headline within +/-5% (I2: forward, not tuned)."""
    mg_yr = parisi.ac225_mg_per_year()
    lo = parisi.HEADLINE_MG_PER_YEAR * (1.0 - parisi.RECON_TOL_FRAC)
    hi = parisi.HEADLINE_MG_PER_YEAR * (1.0 + parisi.RECON_TOL_FRAC)
    assert lo <= mg_yr <= hi, f"{mg_yr} mg/yr outside [{lo}, {hi}]"
    # explicit band the spec names: [19.0, 21.0] mg/yr
    assert 19.0 <= mg_yr <= 21.0, mg_yr


def test_matches_table_I_openmc_value(parisi):
    """Given eta_pro, the analytic chain reproduces the OpenMC Table-I value (20,480 ug/yr) to <1%."""
    ug_yr = parisi.ac225_mg_per_year() * 1000.0
    rel = abs(ug_yr - parisi.TABLE_I_UG_PER_YEAR) / parisi.TABLE_I_UG_PER_YEAR
    assert rel < 0.01, (ug_yr, parisi.TABLE_I_UG_PER_YEAR, rel)


def test_every_factor_has_citation(parisi):
    """I3: every factor row carries a non-empty citation string (and a name, unit, positive value)."""
    assert len(parisi.FACTORS) >= 6
    for f in parisi.FACTORS:
        assert isinstance(f.citation, str) and f.citation.strip(), f"{f.symbol}: empty citation"
        assert f.name.strip(), f.symbol
        assert f.unit.strip(), f.symbol
        assert f.value > 0.0, f.symbol


def test_pfus_cross_check(parisi):
    """P_fus = Ndot_n * E_fus reproduces the paper's ~564 W (abstract 'half a kilowatt') to <1%."""
    p_fus = parisi.fusion_power_W()
    rel = abs(p_fus - parisi.TABLE_I_PFUS_W) / parisi.TABLE_I_PFUS_W
    assert rel < 0.01, (p_fus, parisi.TABLE_I_PFUS_W)


def test_global_supply_multiple(parisi):
    """The yield / 2024-global-supply ratio reproduces the abstract's '~400x' (loose band)."""
    r = parisi.reproduce()
    assert 350.0 <= r["global_supply_multiple"] <= 450.0, r["global_supply_multiple"]


def test_muon_cost_tier_crossref_is_grounded(parisi):
    """The tier cross-ref names a REAL tier and REAL rows in the repo's muon-cost ledger (not a bare
    string): T1-design-study exists, and the Acceleron (3.0 GeV) and Eliezer-Henis (5.0 GeV) rows the
    note cites are present at the values the note quotes."""
    from openmucf.mucost import load_muon_cost

    table = load_muon_cost()
    assert parisi.MUON_COST_TIER == "T1-design-study"
    assert len(table.tier("T1-design-study")) >= 1
    assert table["acceleron_2025"].normalized_GeV_per_stopped_mu == 3.0
    assert table["eliezer_henis_1994"].normalized_GeV_per_stopped_mu == 5.0


def test_import_has_no_side_effects(parisi, capsys):
    """Importing the module prints nothing (all output is guarded behind main())."""
    # the fixture already imported it; assert nothing leaked to stdout during collection/import
    out = capsys.readouterr().out
    assert out == "", out
    # and calling the pure helpers is silent too
    parisi.reproduce()
    assert capsys.readouterr().out == ""
