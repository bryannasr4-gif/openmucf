"""WS-E: the open muon-cost ledger + tier panel.

A curated compilation with provenance, not an evaluation. These tests lock: the loader validates and
rejects a bad tier; every bibkey resolves; the needs_verification flags match the A8-committed set
(Jandel is the only nv=true row); normalized values are positive and tier-ordered (the 10^3 gap, G-E2);
every T3 row carries its derivation; recapture is recorded-not-folded; the FINDINGS section-2b tier panel
regenerates deterministically; and the muon-cost manifest verifies against MUON_COST.md.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import pytest

import openmucf
from openmucf import provenance, uq
from openmucf.mucost import MUON_COST_CSV, MUON_COST_SCHEMA, MuonCostTable, load_muon_cost

REPO = Path(__file__).resolve().parents[1]

# The nv-flag set committed this session (WAVE2 sec.0-A A8): only Jandel is needs_verification.
EXPECTED_NV = {
    "kelly_hart_rose_2021": False,
    "bertin_1987": False,
    "eliezer_henis_1994": False,
    "jandel_1989": True,
    "acceleron_2025": False,
    "muon_collider_front_end": False,
    "mu2e": False,
    "comet": False,
    "music": False,
    "psi_himb": False,
}


@pytest.fixture(scope="module")
def table() -> MuonCostTable:
    return load_muon_cost()


def test_loader_validates_and_loads(table):
    assert len(table) == 10
    assert set(table.ids()) == set(EXPECTED_NV)
    # tier partition covers every row
    n = sum(len(table.tier(t)) for t in ("T1-design-study", "T2-demonstrated-tech", "T3-operating-facility"))
    assert n == len(table)


def test_loader_rejects_bad_tier(tmp_path):
    """A row with a tier outside the enum must raise (schema/enum validation)."""
    bad = tmp_path / "bad.csv"
    header = list(csv.reader([MUON_COST_CSV.read_text(encoding="utf-8").splitlines()[0]]))[0]
    row = {c: "" for c in header}  # empty defaults isolate the tier-enum error
    row.update(
        source_id="bogus", citation="c", year="2020", tier="T9-not-a-tier",
        basis_as_published="b", derivation="d", source_bibkey="Bertin1987", needs_verification="false",
        recapture_credit_applied="false", normalized_GeV_per_stopped_mu="5.0",
    )
    with bad.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        w.writeheader()
        w.writerow(row)
    with pytest.raises(ValueError, match="bad tier"):
        load_muon_cost(csv_path=bad, schema_path=MUON_COST_SCHEMA, check_refs=False)


def test_every_bibkey_resolves(table):
    """Every source_bibkey resolves in references.bib (loading with check_refs=True already enforces
    this; assert it explicitly too)."""
    from openmucf.rates import bibkeys

    keys = bibkeys()
    for r in table:
        for k in re.split(r"[;,]", r.source_bibkey):
            k = k.strip()
            assert k and k in keys, f"{r.source_id}: bibkey {k!r} not in references.bib"


def test_nv_flags_match_committed_set(table):
    """nv flags == the A8-committed set; Jandel is the ONLY needs_verification row (T1 and overall)."""
    got = {r.source_id: r.needs_verification for r in table}
    assert got == EXPECTED_NV
    nv_rows = [r.source_id for r in table.needs_verification()]
    assert nv_rows == ["jandel_1989"]
    # the invariant A8 sec.1.5 asserts: Jandel is the only nv=true row in T1
    t1_nv = [r.source_id for r in table.tier("T1-design-study") if r.needs_verification]
    assert t1_nv == ["jandel_1989"]


def test_normalized_positive_and_tier_ordered(table):
    """Every pinned normalized value is > 0, and the tier medians are ordered T1 < T2 < T3 (G-E2)."""
    for r in table:
        if r.has_normalized:
            assert r.normalized_GeV_per_stopped_mu > 0.0, r.source_id
    m1 = table.tier_median("T1-design-study")
    m2 = table.tier_median("T2-demonstrated-tech")
    m3 = table.tier_median("T3-operating-facility")
    assert m1 < m2 < m3, (m1, m2, m3)


def test_ten_to_the_three_gap_from_the_table(table):
    """G-E2: the tier-median gap T3/T1 proves the ~10^3 simulation-to-facility gap from the table itself."""
    ratio = table.tier_median("T3-operating-facility") / table.tier_median("T1-design-study")
    assert ratio >= 1.0e3, ratio


def test_t3_rows_carry_derivation(table):
    """Every T3 (facility) row is an 'implied, derived here' row -- derivation must be non-empty and
    show the arithmetic; and every row's derivation is non-empty."""
    for r in table.tier("T3-operating-facility"):
        assert r.derivation.strip(), r.source_id
        assert "derived here" in r.derivation, r.source_id
    for r in table:
        assert r.derivation.strip(), r.source_id


def test_recapture_recorded_not_folded(table):
    """recapture consistency: applied=true requires a factor; Kelly's x2.5 is RECORDED (factor present)
    but applied=false, and the normalized 4.70 is the pre-credit value (never silently folded)."""
    for r in table:
        if r.recapture_credit_applied:
            assert not math.isnan(r.recapture_factor), r.source_id
    kelly = table["kelly_hart_rose_2021"]
    assert kelly.recapture_credit_applied is False
    assert kelly.recapture_factor == 2.5
    assert kelly.normalized_GeV_per_stopped_mu == 4.70  # the pre-credit beam-energy-per-muon value


def test_jandel_unpinned(table):
    """Jandel is nv=true with NO pinned normalized value (digit not in hand; not invented)."""
    j = table["jandel_1989"]
    assert j.needs_verification is True
    assert j.has_normalized is False


def test_anchor_values_pinned(table):
    """The three full-text-verified T1 anchors carry their pinned digits (nv=false)."""
    assert table["kelly_hart_rose_2021"].normalized_GeV_per_stopped_mu == 4.70
    assert table["bertin_1987"].normalized_GeV_per_stopped_mu == 7.8
    assert table["eliezer_henis_1994"].normalized_GeV_per_stopped_mu == 5.0
    for sid in ("kelly_hart_rose_2021", "bertin_1987", "eliezer_henis_1994"):
        assert table[sid].needs_verification is False


def test_mucost_is_lazy_public_api():
    """mucost is a lazily-loaded public submodule: exported in __all__ but not eager-imported
    (the PEP 562 __getattr__ resolves it on first access; see tests/test_packaging.py for the
    deterministic no-eager-load and wall-time guards)."""
    assert "mucost" in getattr(openmucf, "__all__", [])
    assert openmucf.mucost.__name__ == "openmucf.mucost"


def test_muon_cost_manifest_verifies():
    """G-E1 kernel: the committed manifest verifies against the committed MUON_COST.md."""
    failures = provenance.check_manifest(REPO / "MUON_COST_MANIFEST.json", repo_root=REPO)
    assert failures == [], failures


def test_tier_panel_deterministic():
    """FINDINGS section-2b regenerates deterministically: qnet_tier_panel is seed-stable and matches the
    committed FINDINGS.md T1/T2/T3 rows."""
    a = uq.qnet_tier_panel(3.0, 6.0)
    b = uq.qnet_tier_panel(3.0, 6.0)
    assert a == b
    boxes = {"T1": (3.0, 6.0), "T2": (1.0e2, 1.0e3), "T3": (2.3e3, 1.0e6)}
    findings = (REPO / "FINDINGS.md").read_text(encoding="utf-8")
    labels = {
        "T1": r"T1 design studies, Uniform\(3\.0, 6\.0\) GeV",
        "T2": r"T2 demonstrated tech, Uniform\(1e2, 1e3\) GeV",
        "T3": r"T3 operating facilities, Uniform\(2\.3e3, 1e6\) GeV",
    }
    for t, (lo, hi) in boxes.items():
        med = f"{uq.qnet_tier_panel(lo, hi)['median']:.2e}"
        assert re.search(rf"{labels[t]}[^\n]*\| {re.escape(med)} \|", findings), (t, med)
