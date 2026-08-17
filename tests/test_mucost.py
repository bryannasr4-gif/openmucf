"""WS-E: the open muon-cost ledger + tier panel.

A curated compilation with provenance, not an evaluation. These tests lock: the loader validates and
rejects a bad tier; every bibkey resolves; the needs_verification flags match the A8-committed set
(Jandel is the only nv=true row); normalized values are positive and tier-ordered (the order-of-magnitude
mixed-basis tier spread, G-E2);
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
from openmucf import mucost, provenance, uq
from openmucf.mucost import MUON_COST_CSV, MUON_COST_SCHEMA, BasisError, MuonCostTable, load_muon_cost

REPO = Path(__file__).resolve().parents[1]


def _load_generator():
    """Import scripts/generate_mucost.py by path (scripts/ is not an importable package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_mucost", REPO / "scripts" / "generate_mucost.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# The nv-flag set committed this session (WAVE2 sec.0-A A8): only Jandel is needs_verification.
EXPECTED_NV = {
    "kelly_hart_rose_2021": False,
    "kelly_electrical_minimal": False,
    "kelly_electrical_site": False,
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

# Every normalized value that existed BEFORE the (stage, numeraire) axes were added. The axes were a
# pure re-labelling: adding them may not move a single published number, and this dict is the pin that
# proves it (see test_relabelling_moved_no_committed_value).
PRE_AXIS_VALUES = {
    "kelly_hart_rose_2021": 4.70,
    "bertin_1987": 7.8,
    "eliezer_henis_1994": 5.0,
    "acceleron_2025": 3.0,
    "muon_collider_front_end": 178.0,
    "mu2e": 4993.0,
    "comet": 2286.0,
    "music": 6002.0,
    "psi_himb": 890000.0,
}


@pytest.fixture(scope="module")
def table() -> MuonCostTable:
    return load_muon_cost()


def test_loader_validates_and_loads(table):
    assert len(table) == 12
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
        recapture_credit_applied="false", normalized_GeV_per_mu="5.0",
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
            assert r.normalized_GeV_per_mu > 0.0, r.source_id
    m1 = table.tier_median("T1-design-study")
    m2 = table.tier_median("T2-demonstrated-tech")
    m3 = table.tier_median("T3-operating-facility")
    assert m1 < m2 < m3, (m1, m2, m3)


def test_tier_spread_is_about_three_orders_of_magnitude_on_mixed_bases(table):
    """The tier-median spread T3/T1 is ~10^3 -- an ORDER-OF-MAGNITUDE, MIXED-BASIS OBSERVATION.

    RE-SPECIFIED (was ``test_ten_to_the_three_gap_from_the_table``, which asserted ``ratio >= 1.0e3``
    under the name of a "10^3 gap"). That framing pinned a claim the repo has now retracted: the name
    said *gap*, i.e. a same-basis ratio, while MUON_COST.md's own text denied it was one, and the
    phrasing propagated into other documents as though it were a result.

    What is actually true, and all that is asserted here: within the SINGLE ``beam_kinetic`` numeraire,
    the T3 and T1 medians differ by about three orders of magnitude. It is NOT a same-basis ratio --
    ``test_no_basis_class_spans_T1_and_T3`` proves no accounting stage is even shared between the two
    tiers, so the quantity has no common denominator to be a ratio *of*. The numeric check is kept
    (deleting it would drop the only guard on the spread) but deliberately loosened to an
    order-of-magnitude band, because a tight bound would once again be pinning a precision the bases
    do not support.
    """
    m1 = table.tier_median("T1-design-study")
    m3 = table.tier_median("T3-operating-facility")
    spread = m3 / m1
    assert 1.0e2 <= spread <= 1.0e4, f"tier spread {spread} left its order-of-magnitude band"
    # and it is only ever computed within one numeraire -- mixing them would be a units error
    assert {r.numeraire for r in table.rows_in_numeraire("beam_kinetic")} == {"beam_kinetic"}


def test_relabelling_moved_no_committed_value(table):
    """Invariant: adding the (stage, numeraire, evidence_status) axes was a PURE re-labelling.

    Not one published number may have moved when the axes were introduced. If this fails, a
    "classification" change silently edited a result.
    """
    for sid, expected in PRE_AXIS_VALUES.items():
        assert table[sid].normalized_GeV_per_mu == expected, sid
    assert table["jandel_1989"].has_normalized is False  # unpinned before, unpinned after


def test_every_pinned_row_declares_its_full_basis(table):
    """A pinned row must state BOTH coordinates of the grid plus how well-founded its number is."""
    for r in table:
        assert r.evidence_status in mucost.VALID_EVIDENCE_STATUS, r.source_id
        if r.has_normalized:
            assert r.numeraire, r.source_id
            assert r.stage, r.source_id
    # and the deprecated alias must still agree with the axis that superseded it, row by row
    for r in table:
        assert r.stage == mucost.STAGE_FROM_BASIS_CLASS[r.basis_class], r.source_id


def test_aggregates_are_numeraire_restricted(table):
    """The units guard: medians are taken WITHIN one numeraire, never across.

    The ledger holds beam-kinetic and electrical rows. Medianing them together would be a units error
    on top of the stage-basis error the document already discloses -- and it would silently move the
    committed T1 median, which NEUTRONOMICS.md consumes.
    """
    assert table.numeraires() == {"beam_kinetic", "electrical_minimal", "electrical_site"}
    # the committed medians are beam-kinetic and are unmoved by the electrical rows
    assert table.tier_median("T1-design-study") == 4.85
    assert table.tier_median("T2-demonstrated-tech") == 178.0
    assert table.tier_median("T3-operating-facility") == 5497.5
    # the electrical rows exist in T1 and would move that median if they were let in
    elec = [r.normalized_GeV_per_mu for r in table.tier("T1-design-study") if r.numeraire != "beam_kinetic"]
    assert elec, "expected electrical-numeraire rows in T1"
    import statistics

    mixed = statistics.median(table.normalized_values("T1-design-study") + elec)
    assert mixed != table.tier_median("T1-design-study"), (
        "the numeraire guard is vacuous if mixing changes nothing -- this test would protect nothing"
    )


def test_derived_electrical_rows_are_pure_numeraire_conversions(table):
    """Both derived rows are Kelly's beam figure re-expressed, at the SAME stage.

    Each is beam / eta_acc, stored at the ledger's two-decimal convention. Nothing about the muon's
    position on the chain changes -- only the units -- so both stay at stage 'produced'.
    """
    beam = table["kelly_hart_rose_2021"].normalized_GeV_per_mu
    for sid, eta_acc in (("kelly_electrical_minimal", 0.18), ("kelly_electrical_site", 0.104)):
        r = table[sid]
        assert r.eta_acc_assumption == eta_acc, sid
        assert r.normalized_GeV_per_mu == round(beam / eta_acc, 2), sid
        assert r.stage == "produced", sid  # a numeraire change is NOT a stage advance
        assert r.charge_basis == "mu_minus", sid
        assert r.evidence_status == "derived_here", sid
        assert "derived here" in r.derivation, sid
    assert table["kelly_electrical_minimal"].numeraire == "electrical_minimal"
    assert table["kelly_electrical_site"].numeraire == "electrical_site"
    # the site denominator is the primary's own arithmetic: 1.3 MW beam / 12.5 MW facility draw
    assert table["kelly_electrical_site"].eta_acc_assumption == pytest.approx(1.3 / 12.5)


def test_eta_mu_is_recorded_arbitrary_and_never_folded(table):
    """Kelly's eta_mu is carried, graded arbitrary, and folded into nothing."""
    kelly = table["kelly_hart_rose_2021"]
    assert kelly.eta_mu_assumption == 0.50
    assert kelly.eta_mu_evidence_status == "author_declared_arbitrary"
    assert kelly.eta_mu_is_sourced is False
    assert "arbitrary but reasonable assumption" in kelly.notes  # the authors' own words, verbatim
    # never folded: the row's value is still the pre-delivery beam figure
    assert kelly.normalized_GeV_per_mu == 4.70
    # and no OTHER row quietly folded it either
    for r in table:
        if r.has_normalized and not math.isnan(r.eta_mu_assumption):
            assert r.eta_mu_evidence_status, r.source_id


def test_no_headline_number_depends_on_an_arbitrary_row(table):
    """Nothing the generator publishes may be composed from a non-sourced factor.

    Every manifest-pinned headline string is recomputed from sourced ledger rows only. Here we assert
    the complement directly: composing the arbitrary eta_mu produces a figure that appears NOWHERE in
    the committed document, in any rendering.
    """
    gen = _load_generator()
    H = gen.build_headline(table)
    doc = (REPO / "MUON_COST.md").read_text(encoding="utf-8")
    kelly = table["kelly_hart_rose_2021"]
    for sid in gen.CHAIN_POINT_IDS:
        composed = table[sid].chain_point().compose(
            kelly.eta_mu_assumption, "stopped_useful_in_dt", kelly.eta_mu_evidence_status, "eta_mu"
        )
        assert composed.is_bound, sid
        for text in (f"{composed.value_GeV:.2f}", f"{composed.value_GeV:.1f}"):
            assert text not in doc, f"an eta_mu-composed figure ({text}) reached the document via {sid}"
            assert text not in H.values(), f"an eta_mu-composed figure ({text}) reached a headline"


def test_bases_are_heterogeneous_and_declared(table):
    """Every pinned row declares basis_class + charge_basis, and the table is NOT basis-homogeneous."""
    for r in table:
        if r.has_normalized:
            assert r.basis_class, r.source_id
            assert r.charge_basis, r.source_id
    assert not table.is_basis_homogeneous(), "bases are known to be mixed; see MUON_COST.md"


def test_no_basis_class_spans_T1_and_T3(table):
    """The honest finding: a same-basis T1-vs-T3 ratio is NOT COMPUTABLE from these rows.

    T1 holds per-produced / per-stopped-in-D-T figures; T3 holds per-collected /
    per-stopped-in-another-target ones. With no shared class, any cross-tier ratio mixes bases -- which
    is exactly what the MUON_COST.md headline now discloses instead of asserting a precise gap.
    """
    shared = table.basis_classes("T1-design-study") & table.basis_classes("T3-operating-facility")
    assert shared == set(), f"a shared basis class appeared: {shared} -- update the headline disclosure"


def test_lower_bound_rows_are_flagged(table):
    """per-produced / per-collected / mixed-charge rows understate the per-stopped-in-D-T cost."""
    assert table["kelly_hart_rose_2021"].understates_stopped_in_dt_cost is True  # per produced
    assert table["music"].understates_stopped_in_dt_cost is True  # counts mu+ AND mu-
    assert table["bertin_1987"].understates_stopped_in_dt_cost is False  # already per stopped in D-T


def test_kelly_wallplug_is_a_one_sided_bound(table):
    """wall-plug = beam / eta_acc is an ENERGY conversion, not a stopping correction.

    Kelly is the only row stating its own eta_acc (0.18, PSI-measured), so it is the only fully-sourced
    wall-plug figure: 4.70/0.18 = 26.1 GeV per muon PRODUCED. Because that row is per-produced, the
    remaining (sub-unity) collection and stopping fractions can only raise the true per-stopped-in-D-T
    cost -- the bound is one-sided.
    """
    kelly = table["kelly_hart_rose_2021"]
    assert kelly.eta_acc_assumption == 0.18
    assert math.isclose(kelly.wallplug_lower_bound_GeV, 4.70 / 0.18, rel_tol=1e-12)
    assert kelly.wallplug_lower_bound_GeV > kelly.normalized_GeV_per_mu
    assert kelly.understates_stopped_in_dt_cost is True
    # rows that state no eta_acc cannot be converted, and must report NaN rather than a guess
    assert math.isnan(table["mu2e"].wallplug_lower_bound_GeV)


def test_psi_himb_is_mu_plus_only(table):
    """PSI HIMB is mu+-only and therefore irrelevant to muCF; the flag must be machine-readable."""
    assert table["psi_himb"].charge_basis == "mu_plus_only"


def test_music_is_mixed_charge(table):
    """MuSIC's published rate counts mu+ and mu- together -- pinned to the source, not just to the doc.

    ``test_published_values_are_read_from_the_ledger_not_typed`` binds document to ledger; this binds
    ledger to source, which is the other half. Without it the CSV could be set to a wrong charge basis
    and every document-vs-ledger check would stay green because both moved together. PSI HIMB already
    had this pin (``test_psi_himb_is_mu_plus_only``); MuSIC did not.
    """
    assert table["music"].charge_basis == "mixed"


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
    assert kelly.normalized_GeV_per_mu == 4.70  # the pre-credit beam-energy-per-muon value


def test_jandel_unpinned(table):
    """Jandel is nv=true with NO pinned normalized value (digit not in hand; not invented)."""
    j = table["jandel_1989"]
    assert j.needs_verification is True
    assert j.has_normalized is False


def test_anchor_values_pinned(table):
    """The three full-text-verified T1 anchors carry their pinned digits (nv=false)."""
    assert table["kelly_hart_rose_2021"].normalized_GeV_per_mu == 4.70
    assert table["bertin_1987"].normalized_GeV_per_mu == 7.8
    assert table["eliezer_henis_1994"].normalized_GeV_per_mu == 5.0
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


def test_eq15_ceiling_recomputes_from_the_criterion_constants(table):
    """The Kou-Chen eq.(15) ceiling is COMPUTED, never a transcribed digit.

    E_cost,max = (eta_sys * E_use / G_mu) * N_fus,mu, checked by hand here against the generator's own
    helper for BOTH useful-energy conventions, and cross-checked against the committed document.
    """
    gen = _load_generator()
    # hand arithmetic, written out: 1 * 20.4 MeV * 150 / 1 = 3060 MeV = 3.06 GeV
    assert gen.eq15_max_muon_cost_GeV(gen.E_USE_KOUCHEN_MEV) == pytest.approx(
        1.0 * 20.4 * 150.0 / 1.0 / 1000.0
    )
    assert gen.eq15_max_muon_cost_GeV(gen.E_USE_KOUCHEN_MEV) == pytest.approx(3.06)
    # Kelly's E_use is itself derived, not pasted: 17.6 + 1.75 * 4.8 = 26.0 MeV -> 3.90 GeV
    assert gen.E_USE_KELLY_MEV == pytest.approx(17.6 + 1.75 * 4.8) == pytest.approx(26.0)
    assert gen.eq15_max_muon_cost_GeV(gen.E_USE_KELLY_MEV) == pytest.approx(3.90)
    # the rest of the criterion, reproduced from the paper's own sec.IV worked values
    n_L = gen.eq9_cycle_demand(5.0, gen.E_USE_KOUCHEN_MEV)
    assert n_L == pytest.approx(5000.0 / 20.4) == pytest.approx(245.1, abs=0.05)
    assert gen.eq12_omega_crit(n_L) * 100.0 == pytest.approx(0.408, abs=0.001)
    assert gen.eq10_one_muon_gain(n_L) == pytest.approx(150.0 / n_L)
    # The eta_sys = 0.4 case is PUBLISHED in the eta_sys provenance bullet as the paper's own sec.IV
    # arithmetic, and it was the one quoted source pair with no guard: reproduce both digits here so a
    # constant cannot move underneath a sentence that attributes them to Kou & Chen.
    n_L_conservative = gen.eq9_cycle_demand(5.0, gen.E_USE_KOUCHEN_MEV, eta_sys=0.4)
    assert n_L_conservative == pytest.approx(613.0, abs=0.5)
    assert gen.eq12_omega_crit(n_L_conservative) * 100.0 == pytest.approx(0.16, abs=0.005)
    # and the committed document carries exactly these, to the precision it prints
    H = gen.build_headline(table)
    assert H["ceiling_kc"] == "3.06" and H["ceiling_kelly"] == "3.90"
    doc = (REPO / "MUON_COST.md").read_text(encoding="utf-8")
    assert "**3.06 GeV**" in doc and "**3.90 GeV**" in doc


def test_published_values_are_read_from_the_ledger_not_typed(table):
    """Every value this document publishes about the ledger must move when the ledger moves.

    A recurring defect class: a structured column (``eta_mu``, ``recapture_factor``, ``charge_basis``)
    or a derived ratio published as a literal typed into the template, so the CSV could move while the
    document kept printing the old value and every guard stayed green -- ``provenance --check`` compares
    the manifest to the document, and a literal that appears in both stays self-consistent while both
    are stale. This asserts the rendered values equal a recomputation from the ledger, which is what
    makes the byte-diff a live drift detector for them.
    """
    gen = _load_generator()
    H = gen.build_headline(table)
    assert H["charge_music"] == table["music"].charge_basis
    assert H["charge_psi_himb"] == table["psi_himb"].charge_basis
    conv = gen.KOUCHEN_CONVENTIONAL_COST_GEV
    assert H["optimism_low"] == f"{table['kelly_electrical_minimal'].normalized_GeV_per_mu / conv:.0f}"
    assert H["optimism_high"] == f"{table['kelly_electrical_site'].normalized_GeV_per_mu / conv:.0f}"
    # and the document actually prints them, so the assertion is about shipped bytes
    # whitespace-collapsed, because the template's line wrapping shifts with the rendered widths
    doc = " ".join((REPO / "MUON_COST.md").read_text(encoding="utf-8").split())
    # Anchored to each row's OWN sentence, not to a bare token: `charge_basis` draws from a small
    # vocabulary, so a bare-token check is satisfied by whichever row happens to render that value --
    # setting MuSIC to `mu_plus_only` would then match PSI HIMB's token and the check would pass.
    assert f"MuSIC's figure is `{H['charge_music']}`" in doc
    assert f"PSI HIMB is `{H['charge_psi_himb']}`" in doc
    assert f"~{H['optimism_low']}-{H['optimism_high']}x optimistic relative to" in doc


#: Spelled row counts, because CHANGELOG.md writes the count as a word.
_ROW_COUNT_WORDS = {9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen"}


def test_hand_written_docs_restate_the_ledger_correctly():
    """The hand-written docs carry ledger facts that nothing regenerates. Bind the facts.

    ``MUON_COST.md`` is generated, byte-diffed and provenance-checked, so a stale digit there dies at
    the gate. ``CHANGELOG.md`` and ``README.md`` restate the same facts by hand with no such binding,
    which is how a row count and a retracted single-basis premise both survived a change that moved
    them -- twice, in the same bullet. ``tests/test_g4parity.py`` does this for the dataset counts;
    this is the ledger's equivalent.

    **What this binds, exactly.** The three COUNTS below are read from the ledger and asserted against
    the prose, in both directions: the right one must be present and no other spelled count may appear
    in the same construction. That is a real guard.

    **What it does not bind, and why.** It cannot detect the retracted *premise* by substring, because
    a substring cannot tell an assertion from its negation -- ``datapackage.json`` correctly contains
    "NOT a single common basis", and banning that string would fail on correct text. The list below is
    therefore a **regression guard for the four wordings this repo has actually shipped**, not a
    detector for the class; paraphrases will pass it and are left to review. Do not describe it as
    more than that.
    """
    table = mucost.load_muon_cost()
    rows = list(table)
    n_rows = len(rows)
    n_tiers = len({r.tier for r in rows})
    n_numeraires = len({r.numeraire for r in rows if r.numeraire})
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"{_count_word(n_rows)} rows across" in changelog, (
        f"CHANGELOG.md must say '{_count_word(n_rows)} rows across' -- the ledger has {n_rows} rows"
    )
    assert f"across {_count_word(n_tiers).lower()} tiers" in changelog, (
        f"CHANGELOG.md must say 'across {_count_word(n_tiers).lower()} tiers' -- the ledger has {n_tiers}"
    )
    # and no OTHER spelled count may appear in the same construction, so a stale restatement cannot
    # simply coexist beside the correct one (asserting presence alone does not assert absence)
    for n, word in _COUNT_WORDS.items():
        if n != n_rows:
            assert f"{word} rows across" not in changelog, f"CHANGELOG.md also says '{word} rows across'"
        if n != n_tiers:
            stale = f"across {word.lower()} tiers"
            assert stale not in changelog, f"CHANGELOG.md also says '{stale}'"
    assert n_numeraires == 3, f"expected 3 numeraires in the ledger, found {n_numeraires}"

    for name in ("CHANGELOG.md", "README.md", "ADOPTERS.md", "paper/paper.md"):
        text = (REPO / name).read_text(encoding="utf-8").lower()
        for claim in _SHIPPED_BASIS_CLAIMS:
            assert claim not in text, f"{name} still asserts '{claim}'; the rows are not on a common basis"


#: Spelled counts, because the prose writes them as words. Wide enough that growth does not KeyError.
_COUNT_WORDS = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
    9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
    15: "Fifteen", 16: "Sixteen", 17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty",
}


def _count_word(n: int) -> str:
    """Spelled form of ``n``, or the digits if the ledger grows past the table."""
    return _COUNT_WORDS.get(n, str(n))


#: The exact wordings of the retracted premise that this repo has actually shipped. A REGRESSION guard
#: for these four strings only -- see the docstring above for why it cannot be more than that.
_SHIPPED_BASIS_CLAIMS = (
    "one auditable basis",
    "single auditable basis",
    "single normalized basis",
    "one normalized basis",
)


def test_a_non_sourced_chain_renders_as_a_bound_not_a_value(table):
    """The API must REFUSE to render an incomplete or unsourced chain as a value.

    This is the actual contribution of the basis work -- it makes the error unrepresentable rather than
    merely discouraged. Three cases: incomplete chain, arbitrary factor, and off-chain row.
    """
    # 1. incomplete chain -- Kelly stops at 'produced', four conversions short of the terminal stage
    beam = table["kelly_hart_rose_2021"].chain_point()
    assert beam.is_bound and beam.bias_direction == "lower"
    assert beam.render().startswith(">= ")
    with pytest.raises(BasisError, match="refusing to render a bound as a value"):
        beam.render_value()

    # 2. an author-declared-arbitrary factor poisons the chain even though it REACHES the terminal stage
    composed = beam.compose(0.50, "stopped_useful_in_dt", "author_declared_arbitrary", "Kelly eta_mu")
    assert composed.missing_stages == ()  # it did reach the end...
    assert composed.is_bound  # ...and is still only a bound
    assert "author_declared_arbitrary" in composed.why_bound()
    with pytest.raises(BasisError):
        composed.render_value()

    # 3. a row stopped outside D-T fuel is not on the chain at all and cannot become a chain point
    with pytest.raises(BasisError, match="not on the muCF chain"):
        table["mu2e"].chain_point()

    # 4. the refusal is NOT vacuous: a complete, fully-sourced chain does render as a value
    ok = mucost.ChainValue(
        value_GeV=42.0,
        stage=mucost.TERMINAL_STAGE,
        numeraire=mucost.BEAM_KINETIC,
        charge_basis="mu_minus",
        statuses=("primary", "primary"),
        provenance=("synthetic-complete-chain",),
    )
    assert not ok.is_bound
    assert ok.render_value() == "42.00 GeV"

    # 5. no row in the real ledger reaches that state today -- which is the finding, and it is asserted
    for r in table:
        if r.has_normalized and r.stage in mucost.MUCF_CHAIN:
            assert r.chain_point().is_bound, f"{r.source_id} claims a fully-sourced chain; verify it"


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
