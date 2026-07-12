"""neutronomics Layer 1: the neutrons-per-joule league table.

A curated compilation with provenance, not an evaluation (I8); no new physics (I1). These tests lock:
muCF is THREE tier-separated rows, each tier-labelled T1/T2/T3 (never one blended row); every row (muCF
AND alternative-source) carries a non-empty source; X_mu = 113 is the MEASURED Petitjean/Breunlich value
imported from ``calibrate.OBS`` (NOT the forward-UQ median 104, NOT a bare literal); the n/J values match
independent hand arithmetic (X_mu / (E_mu_tier in J), E_mu_tier = MUON_COST tier median); each
alternative-source n/J matches its own published (yield, I, V) / (n-per-proton, E) triple; the doc
regenerates byte-identically; the committed manifest verifies; and the "dropped for unsourceability" list
is present in the doc.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import openmucf
from openmucf import mucost, provenance
from openmucf.calibrate import OBS

REPO = Path(openmucf.__file__).resolve().parent.parent
_SCRIPT = REPO / "scripts" / "generate_neutronomics.py"

GEV_TO_J = 1.602176634e-10
EV_TO_J = 1.602176634e-19


def _load_generator():
    """Import the generator by path (file I/O + printing are guarded behind main(), so import is inert)."""
    spec = importlib.util.spec_from_file_location("_gen_neutronomics", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # so the module-level @dataclass can resolve cls.__module__ (py3.12)
    spec.loader.exec_module(mod)
    return mod


def _committed_doc() -> str:
    return (REPO / "NEUTRONOMICS.md").read_bytes().decode("utf-8").replace("\r\n", "\n")


# --------------------------------------------------------------------------------------------------
# 1. Three tier-separated muCF rows, each tier-labelled T1/T2/T3 (never one blended row)
# --------------------------------------------------------------------------------------------------
def test_three_mucf_rows_tier_labeled():
    mod = _load_generator()
    rows = mod.mucf_rows(mucost.load_muon_cost())
    assert len(rows) == 3, rows
    assert [r["short"] for r in rows] == ["T1", "T2", "T3"]
    tier_ids = {r["tier_id"] for r in rows}
    assert tier_ids == {"T1-design-study", "T2-demonstrated-tech", "T3-operating-facility"}
    doc = _committed_doc()
    # each tier appears as its own labelled table row (three separate rows, not one blended row)
    for r in rows:
        assert f"MUON_COST.md {r['tier_id']} median" in doc, r["tier_id"]


# --------------------------------------------------------------------------------------------------
# 2. Every row -- muCF AND alternative-source -- carries a non-empty source/citation
# --------------------------------------------------------------------------------------------------
def test_every_row_sourced():
    mod = _load_generator()
    doc = _committed_doc()
    # muCF rows: each cites its E_mu source (MUON_COST tier) AND the X_mu source (Petitjean/calibrate.OBS)
    for r in mod.mucf_rows(mucost.load_muon_cost()):
        assert f"MUON_COST.md {r['tier_id']} median" in doc
    assert "V_petitjean_Xmu" in doc and "calibrate.OBS" in doc
    # alternative-source rows: each carries a non-empty source AND a non-empty locator
    alts = mod.build_alt_sources()
    assert len(alts) >= 3
    for a in alts:
        assert a.source.strip(), a.key
        assert a.locator.strip(), a.key
        assert a.locator in doc, a.key  # the locator is actually rendered in the table


# --------------------------------------------------------------------------------------------------
# 3. Deterministic: regenerate twice -> byte-identical, and the committed doc matches a fresh build
# --------------------------------------------------------------------------------------------------
def test_doc_and_manifest_deterministic():
    mod = _load_generator()
    table = mucost.load_muon_cost()
    a = mod.build_markdown(table, mod.build_headline(table))
    b = mod.build_markdown(table, mod.build_headline(table))
    assert a == b
    assert a == _committed_doc(), "committed NEUTRONOMICS.md is not byte-identical to a fresh build"


def test_committed_manifest_verifies():
    """The committed NEUTRONOMICS_MANIFEST.json verifies against the committed NEUTRONOMICS.md."""
    failures = provenance.check_manifest(REPO / "NEUTRONOMICS_MANIFEST.json", repo_root=REPO)
    assert failures == [], failures


# --------------------------------------------------------------------------------------------------
# 4. X_mu = 113 is the MEASURED value from calibrate.OBS (NOT the forward-UQ median 104, NOT a literal)
# --------------------------------------------------------------------------------------------------
def test_xmu_grounded_in_calibrate_obs_not_104():
    mod = _load_generator()
    assert mod.XMU == OBS["xmu_obs"] == 113.0
    doc = _committed_doc()
    assert "X_mu = 113 fusions per muon" in doc
    # the doc must explicitly disclaim the forward-UQ median 104 (I2/I3)
    assert "104" in doc and "NOT" in doc


def test_npj_matches_hand_arithmetic():
    """Each muCF n/J equals X_mu / (E_mu_tier in J), E_mu_tier = mucost.tier_median -- recomputed here."""
    mod = _load_generator()
    table = mucost.load_muon_cost()
    expected_medians = {"T1": 4.85, "T2": 178.0, "T3": 5497.5}
    for r in mod.mucf_rows(table):
        med = table.tier_median(r["tier_id"])
        assert med == expected_medians[r["short"]] == r["emu_GeV"]
        hand = 113.0 / (med * GEV_TO_J)
        assert abs(r["n_per_joule"] - hand) < 1e-3 * hand
    # ordering: the ~10^3 muon-cost gap transfers to the neutron economy (T1 > T2 > T3)
    npj = {r["short"]: r["n_per_joule"] for r in mod.mucf_rows(table)}
    assert npj["T1"] > npj["T2"] > npj["T3"]
    assert npj["T1"] / npj["T3"] > 1.0e3


# --------------------------------------------------------------------------------------------------
# 5. Alternative-source n/J each match their own published triple; dropped-list present
# --------------------------------------------------------------------------------------------------
def test_alt_source_derivations_correct():
    mod = _load_generator()
    by_key = {a.key: a for a in mod.build_alt_sources()}
    # each n/J re-derived here from the row's own published triple (I3 "derived here" arithmetic)
    expected = {
        "dt_generator": 8.2e8 / (90e-6 * 140e3),  # yield / (I * V)
        "fng": 1e11 / (1e-3 * 300e3),
        "rtns2": 2.1e11 / (1e-3 * 400e3),
        "spallation": 20.0 / (800e6 * EV_TO_J),  # n_per_proton / (E_proton in J)
    }
    for key, hand in expected.items():
        got = by_key[key].n_per_joule
        assert abs(got - hand) < 1e-3 * hand, key
    # spallation is honestly flagged as NOT monoenergetic 14 MeV
    assert "NOT 14 MeV" in by_key["spallation"].neutron_energy


def test_dropped_list_present_in_doc():
    """The 'dropped for unsourceability' list (empty here) MUST appear in the doc (I3)."""
    doc = _committed_doc()
    assert "Dropped for unsourceability" in doc
    mod = _load_generator()
    assert mod.DROPPED == []  # all rows sourced this session; nothing approximated
