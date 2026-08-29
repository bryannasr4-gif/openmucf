"""The open muon-cost ledger + tier panel.

A curated compilation with provenance, not an evaluation. These tests lock: the loader validates and
rejects a bad tier; every bibkey resolves; the needs_verification flags match the committed set
(Jandel is the only nv=true row); normalized values are positive and tier-ordered (the order-of-magnitude
mixed-basis tier spread);
every T3 row carries its derivation; recapture is recorded-not-folded; the FINDINGS section-2b tier panel
regenerates deterministically; and the muon-cost manifest verifies against MUON_COST.md.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import re
import statistics
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

# The committed needs_verification set: only Jandel is flagged.
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


def test_the_schema_required_columns_are_pinned_and_enforced(tmp_path):
    """The ledger's required columns, pinned as a SET and drilled one at a time through the loader.

    ``load_muon_cost`` reads ``required`` out of the schema at run time, so a column quietly dropped
    from that list would stop being enforced with nothing failing. Pinned as the set itself and never
    as a count: swapping one name for another leaves a count unchanged. The drill blanks one required
    cell at a time in an otherwise-loadable row, so the negative control (the full row) proves the
    drill isolates the fault it names.
    """
    schema = json.loads(Path(MUON_COST_SCHEMA).read_text(encoding="utf-8"))
    required = set(schema["required"])
    assert required == {
        "source_id", "citation", "year", "tier", "basis_as_published", "derivation",
        "source_bibkey", "needs_verification", "evidence_status",
    }
    assert required <= set(schema["properties"]), "a required column with no property to describe it"

    header = list(csv.reader([MUON_COST_CSV.read_text(encoding="utf-8").splitlines()[0]]))[0]
    assert required <= set(header), "a required column the shipped CSV does not carry"
    full = {c: "" for c in header}
    full.update(
        source_id="bogus", citation="c", year="2020", tier="T1-design-study",
        basis_as_published="b", derivation="d", source_bibkey="Bertin1987",
        needs_verification="false", evidence_status="primary",
        recapture_credit_applied="false", normalized_GeV_per_mu="5.0", numeraire="beam_kinetic",
        stage="produced", basis_class="produced", charge_basis="mu_minus",
    )

    def _write(row, name):
        path = tmp_path / name
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
            w.writeheader()
            w.writerow(row)
        return path

    control = _write(full, "control.csv")
    load_muon_cost(csv_path=control, schema_path=MUON_COST_SCHEMA, check_refs=False)

    for col in sorted(required):
        row = dict(full, **{col: ""})
        path = _write(row, f"missing_{col}.csv")
        with pytest.raises(ValueError) as exc:
            load_muon_cost(csv_path=path, schema_path=MUON_COST_SCHEMA, check_refs=False)
        assert f"missing required '{col}'" in str(exc.value)


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
    """nv flags == the committed set; Jandel is the ONLY needs_verification row (T1 and overall)."""
    got = {r.source_id: r.needs_verification for r in table}
    assert got == EXPECTED_NV
    nv_rows = [r.source_id for r in table.needs_verification()]
    assert nv_rows == ["jandel_1989"]
    # the committed set asserts: Jandel is the only nv=true row in T1
    t1_nv = [r.source_id for r in table.tier("T1-design-study") if r.needs_verification]
    assert t1_nv == ["jandel_1989"]


def test_normalized_positive_and_tier_ordered(table):
    """Every pinned normalized value is > 0, and the tier medians are ordered T1 < T2 < T3."""
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
    assert table.tier_median("T3-operating-facility") == 4993.0
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
    # The edge table made this reachable a second way: `compose_path` will compose the same arbitrary
    # factor from `muon_cost_chain.csv`, so the rule is asserted over EVERY path the edge table can
    # build and not only over the eta_mu column it was written for.
    edges = mucost.load_muon_cost_chain()
    for r in table:
        if not r.has_normalized or r.stage not in mucost.MUCF_CHAIN:
            continue
        if r.chain_point().charge_basis not in mucost.COMPOSABLE_CHARGE_BASIS:
            continue
        for path in mucost.enumerate_chain_paths(r.chain_point(), list(edges)):
            if all(edges[e].is_sourced for e in path.edge_ids):
                continue
            for text in (f"{path.value.value_GeV:.2f}", f"{path.value.value_GeV:.1f}"):
                assert text not in doc, (
                    f"an unsourced-path figure ({text}) reached the document via {path.describe()}"
                )


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

    Kelly's beam row states an eta_acc (0.18, PSI-measured), so the conversion is available for it, and
    the ledger publishes the converted figures as rows of their own rather than folding them into a
    beam-kinetic value -- which is why more than one row carries an eta_acc. Do not restate a count or
    a uniqueness claim here: the document renders those from the ledger, and this docstring cannot.
    Because Kelly's beam row is per-produced, the
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
    # nor can a row that is ALREADY counted in an electrical numeraire: it carries an eta_acc *and*
    # has had that conversion applied by construction, so dividing again double-counts it and returns
    # a cost of nothing -- 26.11 / 0.18 = 145.06 for the minimal row, 45.19 / 0.104 = 434.52 for the
    # site one. The loop covers both, and all three preconditions are asserted per row so it cannot
    # pass vacuously if any of those fields is later emptied.
    for sid in ("kelly_electrical_minimal", "kelly_electrical_site"):
        elec = table[sid]
        assert elec.numeraire != mucost.BEAM_KINETIC
        assert not math.isnan(elec.eta_acc_assumption)
        assert elec.has_normalized  # else NaN would come from the missing value, not the guard
        assert math.isnan(elec.wallplug_lower_bound_GeV)


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
    """Manifest-verification kernel: the committed manifest verifies against the committed MUON_COST.md."""
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
    assert gen.eq8_one_muon_gain(n_L) == pytest.approx(150.0 / n_L)
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


def test_hand_written_docs_restate_the_ledger_correctly():
    """The hand-written docs carry ledger facts that nothing regenerates. Bind the facts.

    ``MUON_COST.md`` is generated, byte-diffed and provenance-checked, so a stale digit there dies at
    the gate. ``CHANGELOG.md`` and ``README.md`` restate the same facts by hand with no such binding,
    which is how a row count and a retracted single-basis premise both survived a change that moved
    them -- twice, in the same bullet. ``tests/test_g4parity.py`` does this for the dataset counts;
    this is the ledger's equivalent.

    **What this binds, exactly.** TWO counts -- rows and tiers -- are read from the ledger and asserted
    against the prose in both directions: the right one must be present, and no other spelled count may
    appear in the same construction. The numeraire count is NOT prose-bound, because no hand-written
    document states one; the assertion on it is a ledger invariant that trips if a fourth numeraire
    appears without this test being revisited. Do not describe it as a prose binding.

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
    # ledger invariant, not a prose binding: no hand-written document states a numeraire count
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


def test_csv_is_structurally_well_formed():
    """Every row must have exactly the header's field count, and the resource must be declared.

    A free-text cell that gains an unquoted comma silently becomes two fields: the row shifts, later
    columns take the wrong values and the last one is dropped entirely. None of the other guards can
    see it -- the loader coerces what it is handed, the affected columns are not rendered into
    ``MUON_COST.md``, so the byte-diff has nothing to compare, and a shifted boolean happened to coerce
    to the value it should have had. That is how a 27-field row shipped.

    Scope, exactly: this binds each row's field COUNT against the header, and nothing else. The
    header's field NAMES and their order are bound against ``datapackage.json`` by
    ``tests/test_datapackage.py::test_datapackage_fields_match_live_csv_headers``, for every resource;
    do not restate that check here in a weaker form.
    """
    with MUON_COST_CSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    n_fields = len(rows[0])
    # r may be empty (a blank line): report it rather than indexing into it
    malformed = [(r[0] if r else "<blank row>", len(r)) for r in rows[1:] if len(r) != n_fields]
    assert not malformed, f"rows whose field count != header's {n_fields}: {malformed}"

    # ...and the resource must be DECLARED at all. `test_datapackage.py` compares names and order for
    # every resource that exists, so it iterates past a resource that has been deleted outright; that
    # existence check lives nowhere else.
    package = json.loads((REPO / "datapackage.json").read_text(encoding="utf-8"))
    declared = [r for r in package["resources"] if str(r.get("path", "")).endswith("muon_cost.csv")]
    assert len(declared) == 1, "muon_cost.csv must be declared exactly once in datapackage.json"


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


def test_tier_panel_deterministic(table):
    """FINDINGS section-2b regenerates deterministically, and its rows match the LEDGER's own boxes.

    The boxes are no longer copied into this test. They are read from :func:`mucost.panel_tier_boxes`,
    which is the single place they are defined, and the committed document is then required to carry
    the panel value each of those boxes produces. A box edited in one place and not the other used to
    leave this test green against a stale copy of itself.
    """
    a = uq.qnet_tier_panel(3.0, 6.0)
    b = uq.qnet_tier_panel(3.0, 6.0)
    assert a == b
    boxes = mucost.panel_tier_boxes(table)
    names = {"T1": "design studies", "T2": "demonstrated tech", "T3": "operating facilities"}
    findings = (REPO / "FINDINGS.md").read_text(encoding="utf-8")
    for t, (lo, hi) in boxes.items():
        label = re.escape(f"{t} {names[t]}, Uniform({lo.render()}, {hi.render()}) GeV")
        med = f"{uq.qnet_tier_panel(lo.value, hi.value)['median']:.2e}"
        assert re.search(rf"{label}[^\n]*\| {re.escape(med)} \|", findings), (t, med)
    # and the T3 box is exactly the admitted rows' range -- the property that replaced two edges no
    # document could account for. Asserted on values, so a relabelled edge cannot satisfy it.
    admitted = [r.normalized_GeV_per_mu for r in table.aggregate_rows(tier="T3-operating-facility")]
    assert (boxes["T3"][0].value, boxes["T3"][1].value) == (min(admitted), max(admitted))


def test_the_panel_spread_tracks_its_boxes_and_not_the_ledger(table):
    """FINDINGS section-2b prints two ratios and asserts they agree; both are checked here.

    The paragraph exists to defuse a coincidence. The panel's T1-to-T3 median ratio lands within about
    10% of the muon-cost tier-median ratio, and the retracted sentence read that as the panel
    MEASURING the cost spread. It does not: Q_net goes as 1/E_mu, so the panel ratio tracks the ratio
    of the two boxes' MIDPOINTS, which is a modelling choice this document makes. Three things are
    bound, because the prose asserts all three: the panel ratio agrees with the midpoint ratio, it
    does NOT equal the ledger ratio, and the document prints the values this recomputes.
    """
    boxes = mucost.panel_tier_boxes(table)
    med = {t: uq.qnet_tier_panel(lo.value, hi.value)["median"] for t, (lo, hi) in boxes.items()}
    mid = {t: (lo.value + hi.value) / 2.0 for t, (lo, hi) in boxes.items()}
    panel_ratio = med["T1"] / med["T3"]
    midpoint_ratio = mid["T3"] / mid["T1"]
    ledger_ratio = table.tier_median("T3-operating-facility") / table.tier_median("T1-design-study")

    assert panel_ratio == pytest.approx(midpoint_ratio, rel=0.01), (panel_ratio, midpoint_ratio)
    assert panel_ratio != pytest.approx(ledger_ratio, rel=1e-3), (
        "the panel ratio and the ledger ratio have converged -- the document says they are different "
        "quantities that merely resemble each other, and that sentence would now be misleading"
    )

    def two_sig(x):
        return f"{float(f'{x:.2g}'):.0f}"

    # THE CHECK A READER WOULD RUN. The medians ship at three figures, so dividing the two PRINTED
    # values is the only recomputation available from the page -- and it must land on the published
    # ratio. Quoting a finer precision than the table supports made that check fail while every other
    # guard stayed green, which is how a paragraph written to be verified became unverifiable.
    printed = {t: float(f"{v:.2e}") for t, v in med.items()}
    assert two_sig(printed["T1"] / printed["T3"]) == two_sig(panel_ratio), (
        printed["T1"] / printed["T3"], panel_ratio
    )

    doc = " ".join((REPO / "FINDINGS.md").read_text(encoding="utf-8").split())
    assert f"ratio of the medians above is **{two_sig(panel_ratio)}**" in doc
    assert f"midpoints -- {mid['T1']:g} GeV and {mid['T3']:g} GeV -- is {two_sig(midpoint_ratio)}." in doc
    # and the fall the Finding states is the order of magnitude of that same ratio
    assert f"falls by about {math.log10(panel_ratio):.0f} orders of magnitude" in doc
    # the amendment must not describe the T3 move as a fall: excluding the barred row RAISED it
    assert "RAISES the T3 median Q_net" in doc
    assert med["T3"] > 4.39e-07, "the T3 median rose; a sentence calling that a fall would be wrong"


def test_no_aggregate_admits_a_mu_plus_only_row(table):
    """The schema rule, drilled: EVERY aggregate must move (or red) when a row is recharged.

    ``muon_cost.schema.json`` says a ``mu_plus_only`` figure "must never enter a muCF cost aggregate".
    That was prose, and the tier median did not honour it. Here the rule is exercised three ways: the
    excluded row is named and still present; flipping an ADMITTED row to ``mu_plus_only`` moves the
    median and shrinks the box, so the filter is load-bearing rather than decorative; and flipping the
    excluded row BACK restores exactly the figures that used to ship, which is what makes the
    correction's before/after reproducible rather than remembered.
    """
    excluded = table.rows_excluded_from_aggregates()
    assert [r.source_id for r in excluded] == ["psi_himb"], "the drill below assumes one excluded row"
    assert excluded[0].source_id in table, "an excluded row is still IN the ledger; it is not hidden"
    assert excluded[0] in list(table), "and it is still rendered by any caller iterating the table"

    def _recharged(source_id, charge_basis):
        rows = [
            dataclasses.replace(r, charge_basis=charge_basis) if r.source_id == source_id else r
            for r in table
        ]
        return MuonCostTable(rows)

    # 1. flipping an ADMITTED row out moves the median and pulls the box edge in
    without_music = _recharged("music", "mu_plus_only")
    assert without_music.tier_median("T3-operating-facility") != table.tier_median(
        "T3-operating-facility"
    )
    assert mucost.panel_tier_boxes(without_music)["T3"][1].value == 4993.0

    # 2. flipping the EXCLUDED row back in reproduces the pre-correction figures exactly
    as_shipped_before = _recharged("psi_himb", "mu_minus")
    m3_before = as_shipped_before.tier_median("T3-operating-facility")
    m1 = table.tier_median("T1-design-study")
    assert m3_before == 5497.5
    assert f"{m3_before / m1:.1f}" == "1133.5"
    assert f"{table.tier_median('T3-operating-facility') / m1:.1f}" == "1029.5"

    # 3. both box rules fire, and on the two different faults they exist for. Recharging the row that
    #    SETS the T1 lower edge trips the attribution rule; recharging an INTERIOR T3 row leaves the
    #    edges where they are and trips the containment rule instead -- which is the fault the box
    #    this replaced actually had, and the one an attribution check alone would have missed.
    with pytest.raises(BasisError, match="may never set the edge of a muCF cost aggregate"):
        mucost.panel_tier_boxes(_recharged("acceleron_2025", "mu_plus_only"))
    with pytest.raises(BasisError, match="which may never enter a"):
        mucost.panel_tier_boxes(_recharged("mu2e", "mu_plus_only"))


def test_deleting_a_barred_row_changes_no_aggregate(table):
    """The invariant behind "every aggregate", stated once instead of enumerated.

    Naming each aggregate and checking it individually is only as complete as the list, and the list
    is what went wrong: the tier median and the prior box were two aggregates, and one of them was
    not on anybody's list. This asserts the property directly -- DELETING the barred row from the
    ledger entirely must leave every aggregate value identical -- across the aggregate surfaces the
    module exposes: ``tier_median``, ``aggregate_values`` and ``panel_tier_boxes``, all of which
    route through ``aggregate_rows``. An aggregate added later is covered only if it reads
    ``aggregate_rows`` too (the module contract) AND is added to this drill; the docstring used to
    claim the drill reached later additions by itself, which was more than the body checks.
    Non-vacuous because the row exists and its value is three orders of magnitude from its tier's
    others.
    """
    barred = table.rows_excluded_from_aggregates()
    assert barred, "no barred rows: this invariant would be trivially true"
    without = MuonCostTable([r for r in table if r not in barred])
    assert len(list(without)) == len(list(table)) - len(barred)
    for tier in mucost.TIER_ORDER:
        assert without.tier_median(tier) == table.tier_median(tier), tier
        assert without.aggregate_values(tier) == table.aggregate_values(tier), tier
    boxes_with = mucost.panel_tier_boxes(table)
    boxes_without = mucost.panel_tier_boxes(without)
    assert {t: (lo.value, hi.value, lo.source_id, hi.source_id)
            for t, (lo, hi) in boxes_with.items()} == {
        t: (lo.value, hi.value, lo.source_id, hi.source_id)
        for t, (lo, hi) in boxes_without.items()
    }
    # and the row IS still visible everywhere a reader would look for it
    for r in barred:
        assert r.source_id in table
        assert r in table.rows_in_numeraire(mucost.BEAM_KINETIC, r.tier)
        assert r.basis_class in table.basis_classes(r.tier)


def test_no_box_edge_is_set_by_a_barred_row(table):
    """No prior-box edge is read off a barred row, and no box CONTAINS one.

    Two rules, because the edge that shipped was a round literal rather than a row's value: an
    attribution check alone would have called it clean. The containment rule is the one that catches
    it, and the historical box is used here as the negative control -- if [2.3e3, 1e6] did not fail,
    this guard would be asserting nothing.
    """
    boxes = mucost.panel_tier_boxes(table)
    barred = table.rows_excluded_from_aggregates()
    assert barred, "no barred rows: both rules below would pass vacuously"
    for t, (lo, hi) in boxes.items():
        for edge in (lo, hi):
            if edge.from_ledger:
                assert table[edge.source_id].charge_basis not in (
                    mucost.AGGREGATE_EXCLUDED_CHARGE_BASIS
                ), (t, edge.source_id)
                assert table[edge.source_id].normalized_GeV_per_mu == edge.value
        for r in barred:
            assert not lo.value <= r.normalized_GeV_per_mu <= hi.value, (t, r.source_id)
    # the negative control: the box this replaced DID contain a barred row
    old_t3 = (2.3e3, 1.0e6)
    assert any(old_t3[0] <= r.normalized_GeV_per_mu <= old_t3[1] for r in barred), (
        "the historical T3 box no longer trips the containment rule -- this guard has gone vacuous"
    )
    # every edge is accounted for: a ledger row, or a declared constant. There is no third state.
    # (Membership in the dict alone was circular -- the dict could move WITH the edge and this line
    # stayed green. The dict itself is now pinned by literals in
    # test_declared_edges_and_published_boxes_are_pinned, which is what closes the loop.)
    for t, (lo, hi) in boxes.items():
        for edge in (lo, hi):
            assert edge.from_ledger or edge.value in mucost._DECLARED_EDGES.values(), (t, edge)


def test_declared_edges_and_published_boxes_are_pinned(table):
    """Every published prior-box edge, pinned by literals OUTSIDE the module that defines it.

    ``_DECLARED_EDGES`` used to be checked only by ``edge.value in mucost._DECLARED_EDGES.values()``
    (in test_no_box_edge_is_set_by_a_barred_row), which is circular: mutate the dict, regenerate,
    and every gate stays green while the published T1 row silently becomes Uniform(3.0, 7.0) --
    measured on this branch, twice, before this pin existed. The section-2b stop says the T1 and T2
    rows are byte-identical or the run halts; this is the test that halts it. The T3 edges are
    ledger values and are pinned here too: a ledger move is a published-number move, and a published
    number moves only through a conscious two-place edit (ledger + this literal), never through a
    coordinated regeneration alone.
    """
    assert mucost._DECLARED_EDGES == {"T1_hi": 6.0, "T2_lo": 1.0e2, "T2_hi": 1.0e3}
    boxes = mucost.panel_tier_boxes(table)
    assert {t: (lo.value, hi.value) for t, (lo, hi) in boxes.items()} == {
        "T1": (3.0, 6.0),
        "T2": (100.0, 1000.0),
        "T3": (2286.0, 6002.0),
    }


def test_the_t3_provenance_paragraph_is_derived_from_the_ledger(table):
    """FINDINGS section 2b's T3 provenance prose names rows and stages; that prose must BE the ledger.

    The paragraph calls the box "a pure function of the ledger", and its row list was hand-typed: a
    ledger mutation moved the box and both manifests while the sentence three lines below kept the
    old figures. Same contract as test_the_median_membership_sentence_is_derived_from_the_ledger,
    in both directions: every admitted row is named with its ledger value, no excluded row appears
    in the admitted list, the excluded row is named with its value AND the stage read off its own
    row -- it shipped twice as "per mu+ produced" against a row whose stage is ``transported`` --
    and the rendered clauses appear verbatim in the committed document.
    """
    membership = mucost.panel_t3_membership(table)
    exclusion = mucost.panel_t3_exclusion_clause(table)
    admitted = table.aggregate_rows(tier="T3-operating-facility")
    excluded = table.rows_excluded_from_aggregates(tier="T3-operating-facility")
    assert admitted and excluded, "either list empty: the two-direction checks below would be vacuous"
    for r in admitted:
        assert f"{mucost.PANEL_ROW_LABELS[r.source_id]} at {r.normalized_GeV_per_mu:g} GeV" in membership
    for r in excluded:
        assert mucost.PANEL_ROW_LABELS[r.source_id] not in membership
        assert f"{r.normalized_GeV_per_mu:g} GeV" in exclusion
        assert f"`{r.stage}` stage" in exclusion, "the stage must be the row's own, never assumed"
    doc = " ".join((REPO / "FINDINGS.md").read_text(encoding="utf-8").split())
    assert " ".join(membership.split()) in doc
    assert " ".join(exclusion.split()) in doc
    # the wording the ledger refutes must not come back, anywhere in the document
    assert "per mu+ produced" not in doc
    # and a ledger move moves the sentence: the same mutation an adversarial re-read ran, in memory
    mutated = MuonCostTable(
        [
            dataclasses.replace(r, normalized_GeV_per_mu=6500.0) if r.source_id == "music" else r
            for r in table
        ]
    )
    assert "MuSIC at 6500 GeV" in mucost.panel_t3_membership(mutated)
    assert " ".join(mucost.panel_t3_membership(mutated).split()) not in doc


def test_the_median_membership_sentence_is_derived_from_the_ledger(table):
    """MUON_COST.md names WHICH rows each median is taken over; that list must BE the aggregate.

    A membership sentence written by hand can name a set the median is not actually computed on --
    the same class of defect as a median that admitted a barred row, one level up in the prose. Both
    clauses are rendered from ``aggregate_rows`` and are checked here against it, in both directions:
    every admitted row is named, and no excluded row is.
    """
    gen = _load_generator()
    H = gen.build_headline(table)
    doc = " ".join((REPO / "MUON_COST.md").read_text(encoding="utf-8").split())
    for tier, key in (
        ("T1-design-study", "t1_median_rows"),
        ("T3-operating-facility", "t3_median_rows"),
    ):
        rows = table.aggregate_rows(tier=tier)
        assert rows, tier
        for r in rows:
            assert gen.LABELS[r.source_id] in H[key], (tier, r.source_id)
        for r in table.rows_excluded_from_aggregates(tier=tier):
            assert gen.LABELS[r.source_id] not in H[key], (tier, r.source_id)
        assert " ".join(H[key].split()) in doc
    excluded = table.rows_excluded_from_aggregates()
    assert excluded, "no excluded rows: the disclosure clause below would pass vacuously"
    clause = " ".join(H["aggregate_excluded_clause"].split())
    assert clause and clause in doc
    for r in excluded:
        assert gen.LABELS[r.source_id] in clause, r.source_id


def test_published_tier_ratio_is_bound_to_the_ledger(table):
    """MUON_COST.md's published ratio is a recomputation of the ledger, not a typed digit.

    Both the current figure and the pre-correction figure the amendment quotes are checked, because a
    retraction that misquotes what it retracts is its own defect. The historical value is recovered
    from the ledger by re-admitting the excluded row, never by trusting the string in the document.
    """
    gen = _load_generator()
    H = gen.build_headline(table)
    m1 = table.tier_median("T1-design-study")
    m3 = table.tier_median("T3-operating-facility")
    assert H["gap_ratio"] == f"{m3 / m1:.1f}" == "1029.5"
    assert H["t3_median"] == "4993"
    doc = " ".join((REPO / "MUON_COST.md").read_text(encoding="utf-8").split())
    assert f"nominally **{H['gap_ratio']}x**" in doc
    # the amendment's own arithmetic, recomputed rather than read
    rows = [
        dataclasses.replace(r, charge_basis="mu_minus") if r.source_id == "psi_himb" else r
        for r in table
    ]
    before = MuonCostTable(rows).tier_median("T3-operating-facility")
    assert f"T3 tier median: {before:g} -> {H['t3_median']} GeV" in doc
    assert f"Tier ratio: {before / m1:.1f}x -> {H['gap_ratio']}x" in doc


# ---- basis and loader contracts, each exercised by the input that breaks it -----------------------


def _bad_row(header: list[str], **overrides: str) -> dict[str, str]:
    """A syntactically complete CSV row with empty defaults, so one fault can be isolated."""
    row = {c: "" for c in header}
    row.update(
        source_id="bogus", citation="c", year="2020", tier="T1-design-study",
        basis_as_published="b", derivation="d", source_bibkey="Bertin1987",
        needs_verification="false", recapture_credit_applied="false",
    )
    row.update(overrides)
    return row


def _write_csv(path, header: list[str], row: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        w.writeheader()
        w.writerow(row)


def test_loader_lists_every_problem_including_unparseable_numbers(tmp_path):
    """The loader's contract -- one raise listing EVERY problem -- exercised, not read.

    Four numeric columns were converted outside the accumulating block, so a single unparseable cell
    raised a bare ``float()``/``int()`` error that listed nothing and hid every other fault in the
    file. Reading ``errors.append`` and the closing ``raise`` passes this contract; one bad float
    fails it, which is why it is fed one here for each column at once, alongside an unrelated enum
    error that must survive into the same message.
    """
    header = list(csv.reader([MUON_COST_CSV.read_text(encoding="utf-8").splitlines()[0]]))[0]
    bad = tmp_path / "bad.csv"
    _write_csv(bad, header, _bad_row(
        header,
        tier="T9-not-a-tier",                # the unrelated fault that must not be masked
        year="two thousand twenty",
        recapture_factor="2.5x",
        normalized_GeV_per_mu="4.70 GeV",
        eta_mu_assumption="fifty percent",
        eta_acc_assumption="18%",
    ))
    with pytest.raises(ValueError) as exc:
        load_muon_cost(csv_path=bad, schema_path=MUON_COST_SCHEMA, check_refs=False)
    msg = str(exc.value)
    for column in ("year", "recapture_factor", "normalized_GeV_per_mu",
                   "eta_mu_assumption", "eta_acc_assumption"):
        assert f"{column} is not a number" in msg or f"{column} is not an integer" in msg, (
            f"{column} did not reach the accumulated report:\n{msg}"
        )
    assert "bad tier" in msg, f"an unrelated fault was masked by the bad numbers:\n{msg}"
    # ...and the malformed cell must not ALSO raise the "empty is allowed only when
    # needs_verification=true" complaint, which would describe a cell that is not empty.
    assert "empty normalized_GeV_per_mu" not in msg


def test_loader_still_reports_a_good_row_with_one_bad_number(tmp_path):
    """A single unparseable number must not swallow the rest of the row's validation.

    The control for the test above: with only ``normalized_GeV_per_mu`` malformed, the row's own
    stage/numeraire faults must still be listed rather than lost to an early raise.
    """
    header = list(csv.reader([MUON_COST_CSV.read_text(encoding="utf-8").splitlines()[0]]))[0]
    bad = tmp_path / "bad.csv"
    _write_csv(bad, header, _bad_row(
        header, normalized_GeV_per_mu="not-a-number", numeraire="furlongs", stage="nowhere",
    ))
    with pytest.raises(ValueError) as exc:
        load_muon_cost(csv_path=bad, schema_path=MUON_COST_SCHEMA, check_refs=False)
    msg = str(exc.value)
    assert "normalized_GeV_per_mu is not a number" in msg
    assert "bad numeraire" in msg and "bad stage" in msg


def test_bound_message_names_the_bound_not_the_direction_of_the_truth():
    """``bias_direction`` names the BOUND's type; the truth lies the other way.

    The refusal used to end "the true cost is one-sided (lower)", the inverse of the module's own
    "the true cost can only be higher": the FIGURE is the lower bound, the truth is above it.
    """
    cv = mucost.ChainValue(
        value_GeV=4.70, stage="produced", numeraire=mucost.BEAM_KINETIC,
        charge_basis="mu_minus", statuses=("primary",), provenance=("kelly",),
    )
    assert cv.is_bound and cv.bias_direction == "lower"
    with pytest.raises(BasisError) as exc:
        cv.render_value()
    msg = str(exc.value)
    assert "can only be higher" in msg, msg
    assert "the true cost is one-sided" not in msg, msg


def test_numeraires_is_tier_scoped(table):
    """:meth:`is_basis_homogeneous` prescribes comparing ``numeraires`` for the same tier, so the
    method must accept one.

    Un-scoped it returned every numeraire in the table for every tier, which turns T2 -- which really
    does hold one kind of energy -- into an apparent mixture and inverts its genuinely-safe answer.
    """
    assert table.numeraires() == {"beam_kinetic", "electrical_minimal", "electrical_site"}
    assert table.numeraires("T2-demonstrated-tech") == {"beam_kinetic"}
    assert table.numeraires("T3-operating-facility") == {"beam_kinetic"}
    assert table.numeraires("T1-design-study") == {
        "beam_kinetic", "electrical_minimal", "electrical_site"
    }
    # T2 is safe on BOTH axes; T1 is homogeneous within beam-kinetic yet spans three numeraires, so
    # the basis_class answer alone would not have told a caller that.
    assert table.is_basis_homogeneous("T2-demonstrated-tech")
    assert len(table.numeraires("T2-demonstrated-tech")) == 1
    assert len(table.numeraires("T1-design-study")) > 1
    with pytest.raises(KeyError):
        table.numeraires("T9-not-a-tier")


def test_generator_refuses_an_empty_headline_anchor_set(table):
    """An empty subject must fail loudly rather than render prose about nothing."""
    gen = _load_generator()
    gen.HEADLINE_ANCHOR_IDS = ()
    with pytest.raises(ValueError, match="HEADLINE_ANCHOR_IDS is empty"):
        gen.build_headline(table)


def test_generator_refuses_an_empty_chain_point_set(table):
    """The same guard on the parallel construct, which did not have one.

    ``CHAIN_POINT_IDS`` feeds a clause built by indexing the derived stage list; emptied, it rendered
    "they stop at  rather than at one stage" -- prose about no rows at all -- instead of failing.
    """
    gen = _load_generator()
    gen.CHAIN_POINT_IDS = ()
    with pytest.raises(ValueError, match="CHAIN_POINT_IDS is empty"):
        gen.build_headline(table)


def test_exactly_one_cost_source_states_a_beam_to_electrical_conversion(table):
    """MUON_COST.md's typed uniqueness claim, exercised against the ledger rather than read.

    The claim is about this compilation's COST sources -- the primary each row is a row *of*. A row
    may additionally cite a supporting source for its denominator (the site-wide figure is the PSI
    primary's, not Kelly's), which is why the assertion is on the leading bibkey and why the document
    no longer says Kelly's statement is why all three rows carry one.
    """
    carriers = [r for r in table if not math.isnan(r.eta_acc_assumption)]
    assert len(carriers) == 3
    assert {r.source_bibkey.split(";")[0].strip() for r in carriers} == {"KellyHartRose2021"}
    # the beam row states its own; the two electrical rows are re-expressions of that same row
    assert table["kelly_hart_rose_2021"].numeraire == mucost.BEAM_KINETIC
    assert {r.numeraire for r in carriers} == {
        "beam_kinetic", "electrical_minimal", "electrical_site"
    }


#: The retracted wording, as one phrase. Compared against whitespace-normalized text, so
#: a re-wrap does not defeat it; a restatement in other words is a different string and is not caught.
RETRACTED_LOWER_BOUND_UNIVERSAL = "every published figure in this table is a **lower bound**"

#: The rendered document and the template it comes from. A sentence deleted from one and left in the
#: other still ships, so both are read.
LOWER_BOUND_PROSE_PATHS = ("MUON_COST.md", "scripts/generate_mucost.py")


def _normalized(rel: str) -> str:
    return " ".join((REPO / rel).read_text(encoding="utf-8").split())


def test_the_lower_bound_universal_is_scoped_to_the_rows_it_reaches(table):
    """The closing universal reaches chain rows that count mu-, and only those.

    Two axes exclude a row. A row stopped outside D-T fuel is not on the chain: the loader refuses it
    a chain point and ``understates_stopped_in_dt_cost`` is False, so the sub-unity-factors argument
    never reaches it. A ``mu_plus_only`` row IS on the chain and does get a chain point, but prices no
    mu- at all, so a bound on its mu--only cost holds without saying anything -- the shape the
    retracted wording had one axis over. Both exclusions are derived from the ledger, not typed.
    """
    gen = _load_generator()
    prose = {rel: _normalized(rel) for rel in LOWER_BOUND_PROSE_PATHS}
    doc = prose["MUON_COST.md"]
    H = gen.build_headline(table)

    off_chain = [r for r in table if r.has_normalized and r.stage in mucost.OFF_CHAIN_STAGES]
    assert off_chain, "no off-chain rows: this guard would pass vacuously"
    for r in off_chain:
        assert r.understates_stopped_in_dt_cost is False, r.source_id
        with pytest.raises(BasisError):
            r.chain_point()

    on_chain = [r for r in table if r.has_normalized and r.stage in mucost.MUCF_CHAIN]
    mu_plus_only = [r for r in on_chain if r.charge_basis == "mu_plus_only"]
    assert mu_plus_only, "no mu+-only chain rows: the scope clause would pass vacuously"
    for r in mu_plus_only:
        r.chain_point()  # on the chain: excluded by what it counts, not by where it stops
    excluded = {r.source_id for r in mu_plus_only}
    reached = [r for r in on_chain if r.source_id not in excluded]
    assert reached, "no rows left for the universal to reach"
    # the complement: what the scope word claims of the rows it does keep
    assert {r.charge_basis for r in reached} <= {"mu_minus", "mixed"}
    clause = " ".join(H["mu_plus_only_clause"].split())
    assert clause and clause in doc
    for r in mu_plus_only:
        assert gen.LABELS[r.source_id] in clause

    for rel, text in prose.items():
        assert RETRACTED_LOWER_BOUND_UNIVERSAL not in text, rel
    assert "on the muCF chain** and counts mu- is a **lower bound**" in doc
    assert f"The {len(off_chain)} off-chain" in doc
    assert H["n_offchain_rows"] == str(len(off_chain))


def test_to_numeraire_refuses_the_ways_it_must_and_carries_what_it_must():
    """The numeraire axis gets the same refusals as the stage axis -- drilled in both directions.

    ``compose`` has always refused an off-chain stage, a non-advancing stage, an unknown status and a
    factor outside (0, 1]. The numeraire axis had no typed conversion at all until this test's
    subject existed, so the refusals are drilled here rather than assumed to mirror.
    """
    beam = mucost.ChainValue(
        value_GeV=4.70,
        stage="produced",
        numeraire=mucost.BEAM_KINETIC,
        charge_basis="mu_minus",
        statuses=("primary",),
        provenance=("kelly_hart_rose_2021",),
    )

    # -- the refusals -------------------------------------------------------------------------
    with pytest.raises(BasisError, match="cannot convert to numeraire"):
        beam.to_numeraire(0.18, "wall_plug", "primary_cited", "typo")
    with pytest.raises(BasisError, match="cannot convert to numeraire"):
        beam.to_numeraire(0.18, "", "primary_cited", "empty is not a numeraire")
    with pytest.raises(BasisError, match="must change the numeraire"):
        beam.to_numeraire(0.18, mucost.BEAM_KINETIC, "primary_cited", "no-op")
    with pytest.raises(BasisError, match="unknown evidence_status"):
        beam.to_numeraire(0.18, "electrical_minimal", "probably_fine", "bad status")
    for bad in (0.0, -0.18, 1.5):
        with pytest.raises(BasisError, match=r"must lie in \(0, 1\]"):
            beam.to_numeraire(bad, "electrical_minimal", "primary_cited", "bad factor")

    # -- and the other way: the legal conversion does what it says --------------------------------
    elec = beam.to_numeraire(0.18, "electrical_minimal", "primary_cited", "Kelly eta_acc")
    assert elec.value_GeV == 4.70 / 0.18
    assert elec.stage == "produced"  # a numeraire change is NOT a stage advance
    assert elec.numeraire == "electrical_minimal"
    assert elec.charge_basis == "mu_minus"
    assert elec.statuses == ("primary", "primary_cited")
    assert elec.provenance == ("kelly_hart_rose_2021", "Kelly eta_acc")
    # still a bound, because the chain has not reached the terminal stage
    assert elec.is_bound and elec.missing_stages == ("captured", "transported", "moderated",
                                                     "stopped_useful_in_dt")
    # an unsourced conversion poisons the result exactly as an unsourced delivery factor does
    poisoned = beam.to_numeraire(0.18, "electrical_site", "assumption", "guessed eta_acc")
    assert "assumption" in poisoned.unsourced_statuses
    # a factor of exactly 1 is legal on both axes (an efficiency of 100% is a real, stated value)
    assert beam.to_numeraire(1.0, "electrical_site", "assumption", "unity").value_GeV == 4.70


def test_the_float_wallplug_path_cannot_disagree_with_the_typed_conversion(table):
    """``wallplug_lower_bound_GeV`` and ``ChainValue.to_numeraire`` must give the same magnitude.

    The float property predates the typed conversion and drops the statuses, the provenance and the
    bound flag on the way out. It is kept because downstream code reads it; this pins it to the typed
    path bit-for-bit so the two arithmetics can never drift, which is the guarantee that lets the
    property stay a float.
    """
    checked = 0
    for r in table:
        wp = r.wallplug_lower_bound_GeV
        if math.isnan(wp):
            continue
        typed = r.chain_point().to_numeraire(
            r.eta_acc_assumption, "electrical_minimal", "primary_cited", f"{r.source_id}:eta_acc"
        )
        assert typed.value_GeV == wp, f"{r.source_id}: float {wp} != typed {typed.value_GeV}"
        # the magnitude does not depend on WHICH electrical denominator the label names; the label
        # does, which is why the property cannot pick one for itself and this test supplies it.
        other = r.chain_point().to_numeraire(
            r.eta_acc_assumption, "electrical_site", "primary_cited", f"{r.source_id}:eta_acc"
        )
        assert other.value_GeV == typed.value_GeV
        checked += 1
    assert checked == 1, "exactly one committed row yields a wall-plug figure (Kelly's beam row)"
    # and the typed conversion reproduces the shipped electrical rows to the digits they publish
    kelly = table["kelly_hart_rose_2021"].chain_point()
    for sid, eta in (("kelly_electrical_minimal", 0.18), ("kelly_electrical_site", 0.104)):
        got = kelly.to_numeraire(eta, table[sid].numeraire, "primary_cited", sid).value_GeV
        assert round(got, 2) == table[sid].normalized_GeV_per_mu


# --------------------------------------------------------------------------------------------------
# The EDGE table (muon_cost_chain.csv): the conversions that join the cost grid's points.
# --------------------------------------------------------------------------------------------------

CHAIN_COLUMNS = [
    "edge_id", "from_stage", "from_numeraire", "to_stage", "to_numeraire",
    "factor", "factor_lo", "factor_hi", "bias_direction", "charge_basis",
    "conditions", "evidence_status", "source_bibkey", "source_locator", "derivation", "notes",
]

#: A loadable numeraire edge, used as the negative control every drill below mutates one cell of.
CHAIN_CONTROL = {
    "edge_id": "control_edge", "from_stage": "*", "from_numeraire": "beam_kinetic",
    "to_stage": "*", "to_numeraire": "electrical_minimal",
    "factor": "0.5", "factor_lo": "", "factor_hi": "",
    "bias_direction": "none", "charge_basis": "*", "conditions": "synthetic",
    "evidence_status": "primary", "source_bibkey": "Kovach2017",
    "source_locator": "synthetic fixture, not a reading of the source",
    "derivation": "synthetic fixture", "notes": "",
}


@pytest.fixture(scope="module")
def chain() -> mucost.ChainEdgeTable:
    return mucost.load_muon_cost_chain()


def _write_chain(tmp_path, rows, name="chain.csv"):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHAIN_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CHAIN_COLUMNS})
    return path


def _load_chain(path):
    return mucost.load_muon_cost_chain(
        csv_path=path, schema_path=mucost.MUON_COST_CHAIN_SCHEMA, check_refs=False
    )


def test_the_edge_table_loads_and_every_stated_factor_cites_a_primary(chain):
    """The committed edge table, and the rule that is the whole reason it can be trusted.

    Every edge either carries a factor WITH a bibkey and a locator, or carries evidence_status
    'absent' and no number in any of its three numeric cells. A plausible invented factor filling a
    hole is the failure this table exists to prevent, and this is where that is checked rather than
    promised.
    """
    assert len(chain) == 8
    assert chain.ids() == [
        "eta_acc_kelly_psi_minimal", "eta_acc_kovach_minimal", "eta_acc_kovach_site",
        "delivery_kelly_eta_mu",
        "produced_to_captured_absent", "captured_to_transported_absent",
        "transported_to_moderated_absent", "moderated_to_stopped_useful_absent",
    ]
    # the one stated stage factor spans the WHOLE delivery segment, which is what its source's own
    # eq.(2) makes it: a factor multiplying the fusion energy is the useful-stopped fraction, not an
    # arrival fraction. Composing it alongside that source's earlier 100% placeholder for the same
    # conversion would apply one factor twice, so only one of the two is an edge.
    delivery = chain["delivery_kelly_eta_mu"]
    assert (delivery.from_stage, delivery.to_stage) == ("produced", mucost.TERMINAL_STAGE)
    assert delivery.spans == mucost.MUCF_CHAIN[1:]
    known = mucost.bibkeys()
    for e in chain:
        if e.has_factor:
            assert e.evidence_status != "absent"
            assert e.source_bibkey and e.source_locator, f"{e.edge_id}: a factor with no provenance"
            assert 0.0 < e.factor <= 1.0
            assert e.derivation, f"{e.edge_id}: no derivation recorded"
        else:
            assert e.evidence_status == "absent"
            assert math.isnan(e.factor) and math.isnan(e.factor_lo) and math.isnan(e.factor_hi)
        for key in e.source_bibkey.split(";"):
            if key.strip():
                assert key.strip() in known, f"{e.edge_id}: {key} does not resolve"
    # the numeraire conversion is the only one the literature sources at all
    assert {e.edge_id for e in chain.sourced()} == {
        "eta_acc_kelly_psi_minimal", "eta_acc_kovach_minimal", "eta_acc_kovach_site"
    }
    # an absent row may cite the sources READ AND FOUND SILENT; the shipped ones all do, and the
    # schema says so. (This is the half of the provenance rule that is easy to get backwards: the
    # bibkey on such a row names what was checked, never the source of a number there is none of.)
    for e in chain:
        if not e.has_factor:
            assert e.source_bibkey and e.source_locator, f"{e.edge_id}: an unchecked absence"
    schema = json.loads(Path(mucost.MUON_COST_CHAIN_SCHEMA).read_text(encoding="utf-8"))
    assert "READ AND FOUND SILENT" in schema["properties"]["source_bibkey"]["description"]
    assert all(e.kind == mucost.NUMERAIRE_EDGE for e in chain.sourced())
    assert chain.of_kind(mucost.STAGE_EDGE) and chain.of_kind(mucost.NUMERAIRE_EDGE)
    with pytest.raises(KeyError, match="unknown edge kind"):
        chain.of_kind("diagonal")


def test_the_chain_schema_required_columns_are_pinned_and_enforced(tmp_path):
    """The edge table's required columns, pinned as a SET and drilled one at a time.

    Same discipline as the node table's: the loader reads `required` from the schema at run time, so
    a name quietly dropped there would stop being enforced with nothing failing.
    """
    schema = json.loads(Path(mucost.MUON_COST_CHAIN_SCHEMA).read_text(encoding="utf-8"))
    required = set(schema["required"])
    assert required == {
        "edge_id", "from_stage", "from_numeraire", "to_stage", "to_numeraire",
        "bias_direction", "charge_basis", "evidence_status", "derivation",
    }
    assert required <= set(schema["properties"])
    assert set(schema["properties"]) == set(CHAIN_COLUMNS)
    header = mucost.MUON_COST_CHAIN_CSV.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == CHAIN_COLUMNS, "the shipped edge CSV's header must match the schema's fields"

    _load_chain(_write_chain(tmp_path, [CHAIN_CONTROL], "control.csv"))  # negative control
    for col in sorted(required):
        path = _write_chain(tmp_path, [dict(CHAIN_CONTROL, **{col: ""})], f"missing_{col}.csv")
        with pytest.raises(ValueError) as exc:
            _load_chain(path)
        assert f"missing required '{col}'" in str(exc.value)


def test_an_edge_moves_exactly_one_axis(tmp_path, chain):
    """One axis per edge, and the wildcard on the other -- drilled in every way it can be wrong.

    This is what keeps wall-plug a numeraire rather than a chain node. An edge that moved both axes
    would be a source's collapsed factor smuggled in as one conversion; an edge that pinned the axis
    it does not move would make the same conversion inexpressible everywhere else on the chain.
    """
    for e in chain:
        if e.kind == mucost.NUMERAIRE_EDGE:
            assert e.from_stage == e.to_stage == mucost.ANY
            assert e.from_numeraire != e.to_numeraire
        else:
            assert e.from_numeraire == e.to_numeraire == mucost.ANY
            assert mucost.MUCF_CHAIN.index(e.to_stage) > mucost.MUCF_CHAIN.index(e.from_stage)

    cases = [
        (dict(from_stage="produced", to_stage="captured"), "moves both"),
        (dict(from_numeraire="*", to_numeraire="*"), "must move one axis"),
        (dict(from_stage="produced", to_stage="produced", from_numeraire="*", to_numeraire="*"),
         "must advance the chain"),
        (dict(from_stage="captured", to_stage="produced", from_numeraire="*", to_numeraire="*"),
         "must advance the chain"),
        (dict(to_numeraire="beam_kinetic"), "must change the numeraire"),
        (dict(from_stage="produced", to_stage="*", from_numeraire="*", to_numeraire="*"),
         "needs both endpoints on the chain"),
        (dict(from_numeraire="*"), "needs both endpoints named"),
        (dict(from_stage="produced", to_stage="captured", from_numeraire="*", to_numeraire="*",
              factor="", evidence_status="absent", bias_direction="lower", source_bibkey="",
              source_locator=""), None),  # a legal stage edge: the drill's own control
        (dict(from_stage="stopped_other_target", to_stage="*", from_numeraire="*",
              to_numeraire="*"), "bad from_stage"),
    ]
    for i, (mutation, expected) in enumerate(cases):
        path = _write_chain(tmp_path, [dict(CHAIN_CONTROL, **mutation)], f"axis_{i}.csv")
        if expected is None:
            _load_chain(path)
            continue
        with pytest.raises(ValueError, match=re.escape(expected)):
            _load_chain(path)


def test_an_absent_edge_carries_no_number_and_refuses_to_compose(tmp_path, chain):
    """'absent' means the hole is recorded, not filled -- enforced on load and again on composition."""
    absent = chain["moderated_to_stopped_useful_absent"]
    assert not absent.has_factor and not absent.is_sourced
    start = mucost.ChainValue(1.0, "moderated", mucost.BEAM_KINETIC, "mu_minus", ("primary",), ("x",))
    with pytest.raises(BasisError, match="carries no factor"):
        absent.apply_to(start)
    with pytest.raises(BasisError, match="carries no factor"):
        mucost.compose_path(start, [absent])

    stage_absent = dict(
        CHAIN_CONTROL, edge_id="a", from_stage="produced", to_stage="captured",
        from_numeraire="*", to_numeraire="*", evidence_status="absent", bias_direction="lower",
        factor="", source_bibkey="", source_locator="",
    )
    _load_chain(_write_chain(tmp_path, [stage_absent], "absent_ok.csv"))  # control
    for mutation, expected in (
        (dict(factor="0.9"), "the factor cell must be empty"),
        (dict(factor_lo="0.2"), "must be empty"),
        (dict(factor_hi="0.9"), "must be empty"),
    ):
        with pytest.raises(ValueError, match=re.escape(expected)):
            _load_chain(_write_chain(tmp_path, [dict(stage_absent, **mutation)], "absent_bad.csv"))
    # and the other way: a status that CLAIMS a stated factor may not leave the cell empty
    with pytest.raises(ValueError, match="may not be empty"):
        _load_chain(_write_chain(tmp_path, [dict(CHAIN_CONTROL, factor="")], "no_factor.csv"))
    # ...nor may it leave the factor uncited. This is the rule the whole table rests on -- a number
    # with no primary behind it is the invented factor the 'absent' status exists to make
    # unnecessary -- so it is drilled from both missing halves, not assumed from one.
    for missing in ("source_bibkey", "source_locator"):
        with pytest.raises(ValueError, match="every edge value comes from a primary"):
            _load_chain(_write_chain(
                tmp_path, [dict(CHAIN_CONTROL, **{missing: ""})], f"uncited_{missing}.csv"
            ))


def test_bias_direction_must_agree_with_the_evidence_status(tmp_path, chain):
    """A sourced factor biases nothing; an unsourced one always biases somehow. Both ways drilled."""
    for e in chain:
        assert (e.bias_direction == "none") == e.is_sourced, e.edge_id
    with pytest.raises(ValueError, match="contradicts evidence_status"):
        _load_chain(_write_chain(tmp_path, [dict(CHAIN_CONTROL, bias_direction="lower")], "b1.csv"))
    with pytest.raises(ValueError, match="contradicts evidence_status"):
        _load_chain(_write_chain(
            tmp_path,
            [dict(CHAIN_CONTROL, evidence_status="assumption", bias_direction="none")],
            "b2.csv",
        ))
    with pytest.raises(ValueError, match="bad bias_direction"):
        _load_chain(_write_chain(tmp_path, [dict(CHAIN_CONTROL, bias_direction="up")], "b3.csv"))


def test_a_stated_interval_must_bracket_its_factor(tmp_path):
    """factor_lo/factor_hi are empty on every committed edge, so their rules are drilled here.

    No primary read for this table states a range for any conversion, which is itself part of the
    coverage finding. The columns exist for the source that does, and an unexercised validator is
    not a validator, so the bracket rules are drilled on a synthetic edge instead of assumed.
    """
    ok = dict(CHAIN_CONTROL, factor="0.5", factor_lo="0.4", factor_hi="0.6")
    edge = _load_chain(_write_chain(tmp_path, [ok], "iv_ok.csv"))["control_edge"]
    assert (edge.factor_lo, edge.factor_hi) == (0.4, 0.6)
    for mutation, expected in (
        (dict(factor_lo="0.7"), "lies below factor_lo"),
        (dict(factor_hi="0.4"), "lies above factor_hi"),
        (dict(factor_lo="0.8", factor_hi="0.3", factor="0.5"), "exceeds factor_hi"),
        (dict(factor_hi="1.5"), "factor_hi must lie in (0, 1]"),
        (dict(factor="1.5", factor_lo="", factor_hi=""), "factor must lie in (0, 1]"),
    ):
        with pytest.raises(ValueError, match=re.escape(expected)):
            _load_chain(_write_chain(tmp_path, [dict(ok, **mutation)], "iv_bad.csv"))


def test_compose_path_refuses_every_join_that_would_misrepresent_the_figure(table, chain):
    """D5, drilled: the joins must match, the charge must be mu-, and no factor is applied twice."""
    beam = table["kelly_hart_rose_2021"].chain_point()
    eta_acc = chain["eta_acc_kelly_psi_minimal"]
    delivery = chain["delivery_kelly_eta_mu"]

    # a numeraire edge whose from_numeraire is not where the figure sits
    already_electrical = mucost.compose_path(beam, [eta_acc]).value
    with pytest.raises(BasisError, match="does not join here"):
        mucost.compose_path(already_electrical, [eta_acc])
    # a stage edge whose from_stage is not where the figure sits
    with pytest.raises(BasisError, match="does not join here"):
        mucost.compose_path(mucost.compose_path(beam, [delivery]).value, [delivery])
    # the same edge twice on one path
    with pytest.raises(BasisError, match="appears twice"):
        mucost.compose_path(beam, [eta_acc, eta_acc])
    # a starting figure that prices no mu-
    mu_plus = dataclasses.replace(beam, charge_basis="mu_plus_only")
    with pytest.raises(BasisError, match="must be counted on mu-"):
        mucost.compose_path(mu_plus, [eta_acc])
    # an edge stated for a charge basis a muCF chain may not compose
    mixed = dataclasses.replace(eta_acc, charge_basis="mixed")
    with pytest.raises(BasisError, match="may not enter a muCF chain"):
        mucost.compose_path(beam, [mixed])
    # the legal path still composes -- the refusals above are not refusing everything
    assert mucost.compose_path(beam, [eta_acc, delivery]).value.stage == mucost.TERMINAL_STAGE


def test_an_unsourced_path_is_a_bound_and_a_fully_sourced_synthetic_one_is_not(tmp_path):
    """The D5 refusal, both ways, on a SYNTHETIC edge set.

    The positive half is deliberately synthetic. Asserting it against the committed table would
    assert that the literature has a fully-sourced chain, which it does not -- so the test would
    either fail today or, worse, be written to pass by grading a real hole as sourced. The synthetic
    edges prove the refusal is not vacuous without making any claim about the literature.
    """
    edges = [
        dict(CHAIN_CONTROL, edge_id="syn_numeraire", factor="0.20"),
        dict(CHAIN_CONTROL, edge_id="syn_delivery", from_stage="produced",
             to_stage="stopped_useful_in_dt", from_numeraire="*", to_numeraire="*",
             factor="0.25", charge_basis="mu_minus"),
    ]
    syn = _load_chain(_write_chain(tmp_path, edges, "synthetic.csv"))
    start = mucost.ChainValue(
        value_GeV=4.0, stage="produced", numeraire=mucost.BEAM_KINETIC, charge_basis="mu_minus",
        statuses=("primary",), provenance=("synthetic_start",),
    )
    good = mucost.compose_path(start, list(syn))
    assert good.value.value_GeV == 4.0 / 0.20 / 0.25
    assert good.value.stage == mucost.TERMINAL_STAGE and good.value.missing_stages == ()
    assert not good.value.is_bound
    assert good.bias_direction == "none"
    assert good.render() == good.render_value() == "80.00 GeV"

    # ...and one unsourced factor anywhere on the same path takes the value away again
    edges[1] = dict(edges[1], evidence_status="assumption", bias_direction="lower")
    poisoned = mucost.compose_path(start, list(_load_chain(_write_chain(tmp_path, edges, "syn2.csv"))))
    assert poisoned.value.value_GeV == good.value.value_GeV  # same number...
    assert poisoned.bias_direction == "lower"  # ...different epistemic status
    with pytest.raises(BasisError, match="refusing to render"):
        poisoned.render_value()
    with pytest.raises(BasisError, match="refusing to render a bound as a value"):
        poisoned.value.render_value()
    assert poisoned.render() == ">= 80.00 GeV"


def test_competing_edges_are_a_set_of_terminal_figures_and_never_a_mean(table, chain):
    """Where two sources give the same conversion, both survive to the end as separate figures.

    This is the reason the edges are a second TABLE rather than more columns on the node table: a
    per-source column set would force one path per source, and the two readings of the same
    conversion could not both exist. No mean is formed and no value is preferred.
    """
    competing = chain.competing()
    assert list(competing) == [("*", "beam_kinetic", "*", "electrical_minimal")]
    rival_ids = [e.edge_id for e in competing[("*", "beam_kinetic", "*", "electrical_minimal")]]
    assert rival_ids == ["eta_acc_kelly_psi_minimal", "eta_acc_kovach_minimal"]
    rivals = [chain[i].factor for i in rival_ids]
    assert rivals == [0.18, 0.183]

    beam = table["kelly_hart_rose_2021"].chain_point()
    paths = mucost.enumerate_chain_paths(beam, list(chain))
    figures = [round(p.value.value_GeV, 2) for p in paths]
    assert len(figures) == len(set(figures)) == 3, "three maximal paths, three distinct figures"
    # the two rival readings of one conversion reach the SAME coordinate with DIFFERENT numbers
    same_coord = [p for p in paths if p.value.numeraire == "electrical_minimal"]
    assert len(same_coord) == 2
    assert {round(p.value.value_GeV, 2) for p in same_coord} == {52.22, 51.37}
    assert statistics.mean(rivals) not in rivals  # the mean is not a value this table holds
    for p in paths:  # every figure carries the edges it was built from
        # one numeraire conversion and the one collapsed delivery factor: the whole chain the
        # literature can build, which is two edges long and ends on an arbitrary one
        assert len(p.edge_ids) == 2 and p.describe().startswith("kelly_hart_rose_2021 -> ")
        assert p.value.stage == mucost.TERMINAL_STAGE


def test_composition_is_order_independent_which_is_why_paths_dedupe_by_edge_set(table, chain):
    """Applying the same edges in a different order gives the same figure, exactly.

    ``enumerate_chain_paths`` deduplicates by edge SET on that basis, so this is the property the
    deduplication rests on rather than an incidental one. Bit-identical, not close: the composition
    is a sequence of divisions by the same floats.
    """
    beam = table["kelly_hart_rose_2021"].chain_point()
    a = chain["eta_acc_kovach_site"]
    b = chain["delivery_kelly_eta_mu"]
    first = mucost.compose_path(beam, [a, b])
    second = mucost.compose_path(beam, [b, a])
    assert first.value.value_GeV == second.value.value_GeV
    assert first.value.stage == second.value.stage == mucost.TERMINAL_STAGE
    assert first.value.numeraire == second.value.numeraire == "electrical_site"
    assert sorted(first.value.statuses) == sorted(second.value.statuses)
    assert first.bias_direction == second.bias_direction
    paths = mucost.enumerate_chain_paths(beam, list(chain))
    assert len({frozenset(p.edge_ids) for p in paths}) == len(paths)


def test_no_committed_path_reaches_a_figure_this_ledger_may_render_as_a_value(table, chain):
    """The finding, asserted: not one path over the committed edges earns a plain number.

    Every path either stops short of the terminal stage or runs through a factor whose own authors
    call it arbitrary. That is a statement about the literature, not a gap in the code, and it is why
    both refusals are exercised on every committed path here.
    """
    seen = 0
    for r in table:
        if not r.has_normalized or r.stage not in mucost.MUCF_CHAIN:
            continue
        start = r.chain_point()
        if start.charge_basis not in mucost.COMPOSABLE_CHARGE_BASIS:
            with pytest.raises(BasisError, match="must be counted on mu-"):
                mucost.enumerate_chain_paths(start, list(chain))
            continue
        for p in mucost.enumerate_chain_paths(start, list(chain)):
            assert p.bias_direction in {"lower", "unknown"}, f"{p.describe()} claims a value"
            with pytest.raises(BasisError):
                p.render_value()
            seen += 1
    assert seen, "no committed row produced a path, so this test asserted nothing"


def test_a_direction_unknown_path_is_never_printed_with_a_bound_marker(table, chain):
    """An arbitrary factor is weaker than an omitted one, and the rendering must say so.

    An OMITTED factor is bounded above by 1, so leaving it out can only understate the cost: that is
    a one-sided lower bound and prints with ">=". A factor a source states while saying it does not
    know the value can move the figure either way, so the same marker would be a false claim. The
    edge carries the direction and the path reads it; ``ChainValue.bias_direction`` cannot see it,
    which is precisely why ``ChainPath`` exists.
    """
    beam = table["kelly_hart_rose_2021"].chain_point()
    delivery = chain["delivery_kelly_eta_mu"]
    assert delivery.bias_direction == "unknown"

    partial = mucost.compose_path(beam, [chain["eta_acc_kovach_site"]])
    assert partial.bias_direction == "lower" and partial.render().startswith(">= ")

    through_arbitrary = mucost.compose_path(beam, [delivery])
    assert through_arbitrary.bias_direction == "unknown"
    assert not through_arbitrary.render().startswith(">= ")
    assert through_arbitrary.render().endswith("(direction unknown)")
    assert "direction unknown: delivery_kelly_eta_mu" in through_arbitrary.why_bound()
    # the underlying ChainValue still grades itself 'lower'; the two answer different questions and
    # the path's answer is the one a document may print
    assert through_arbitrary.value.bias_direction == "lower"
    with pytest.raises(BasisError, match=re.escape("refusing to render a figure graded 'unknown'")):
        through_arbitrary.render_value()


def test_the_edge_table_is_declared_in_the_data_package(chain):
    """The FAIR descriptor must list the edge CSV as a resource, with its fields and its licence.

    Same rule the node table is held to: a shipped CC-BY data file that the data package does not
    declare is undiscoverable by the machinery that makes the package worth having.
    """
    package = json.loads((REPO / "datapackage.json").read_text(encoding="utf-8"))
    declared = [r for r in package["resources"]
                if str(r.get("path", "")).endswith("muon_cost_chain.csv")]
    assert len(declared) == 1, "muon_cost_chain.csv must be declared exactly once"
    resource = declared[0]
    assert resource["schema"]["primaryKey"] == "edge_id"
    assert [f["name"] for f in resource["schema"]["fields"]] == CHAIN_COLUMNS
    assert {lic["name"] for lic in package["licenses"]} == {"CC-BY-4.0"}


def test_the_edge_csv_is_a_generator_input_bound_by_the_manifest():
    """The edge table must be a declared INPUT, or the rendered conversions float free of their bytes.

    ``provenance --check`` binds each manifest entry to the document that renders it; what binds the
    document to the DATA is the inputs digest. A new CSV that fed the document without appearing here
    could move under a green gate, which is exactly the hole this records.
    """
    manifest = json.loads((REPO / "MUON_COST_MANIFEST.json").read_text(encoding="utf-8"))
    inputs = manifest["inputs"]
    assert set(inputs) == {"muon_cost_csv_sha256", "muon_cost_chain_csv_sha256"}
    assert inputs["muon_cost_chain_csv_sha256"] == provenance.file_sha256(
        mucost.MUON_COST_CHAIN_CSV
    )
    assert inputs["muon_cost_csv_sha256"] == provenance.file_sha256(MUON_COST_CSV)


def test_the_chain_coverage_sentence_is_derived_from_the_two_tables(table, chain):
    """The document's coverage headline is computed from the CSVs, never typed into the template.

    It is the sentence a reader is most likely to quote, so it is the one that must not be able to go
    stale: every count in it is recomputed here from the tables, and the rendered sentence is then
    required to be present in the committed document verbatim.
    """
    gen = _load_generator()
    H = gen.build_headline(table, chain)
    sentence = H["chain_coverage_sentence"]
    doc = _normalized("MUON_COST.md")
    assert " ".join(sentence.split()) in doc

    cov = gen.edge_coverage_rows(table, chain)
    links = [c for c in cov if c["kind"] == mucost.STAGE_EDGE]
    numeraire = [c for c in cov if c["kind"] == mucost.NUMERAIRE_EDGE]
    # the four consecutive chain links, derived from MUCF_CHAIN rather than listed
    assert [c["label"] for c in links] == [
        f"`{a}` -> `{b}`"
        for a, b in zip(mucost.MUCF_CHAIN[:-1], mucost.MUCF_CHAIN[1:], strict=True)
    ]
    # the numeraire conversions are the ones the LEDGER's own pinned rows are counted in
    assert {c["label"] for c in numeraire} == {
        f"`{mucost.BEAM_KINETIC}` -> `{n}`"
        for n in {r.numeraire for r in table if r.has_normalized} - {mucost.BEAM_KINETIC}
    }
    # ...and the finding itself: the sourced conversions are exactly the numeraire ones
    assert [c["label"] for c in cov if c["sourced"]] == [c["label"] for c in numeraire]
    assert H["n_sourced_stage_conversions"] == "0"
    assert H["n_sourced_conversions"] == str(len(numeraire))
    # every stage link IS covered by an edge -- by an arbitrary or absent one, which is the point
    for c in links:
        assert c["edges"], f"{c['label']} has no edge at all, not even an absent one"
        assert not c["sourced"]
    # the sentence agrees with the node table's own count of fully-sourced chains
    assert H["n_fully_sourced_chains"] == "0"
    assert f"**{H['n_fully_sourced_chains']} of the {H['n_chain_rows']} pinned rows" in \
        (REPO / "MUON_COST.md").read_text(encoding="utf-8")


def test_the_published_figure_set_is_sourced_only_and_is_never_reduced(table, chain):
    """What the document prints: every fully-sourced path, as a SET, and no reconciled number.

    Two rules meet here. The competing readings of one conversion are the deliverable, so no mean,
    midpoint or preferred value may be rendered. And nothing printed may be composed through a factor
    its own authors call arbitrary, which is why the published set is built from
    ``chain.sourced()`` while the API composes the rest.
    """
    gen = _load_generator()
    H = gen.build_headline(table, chain)
    doc = _normalized("MUON_COST.md")
    paths = gen.sourced_paths(table, chain)
    assert len(paths) == int(H["n_sourced_paths"]) >= 2
    values = [p.value.value_GeV for p in paths]
    assert values == sorted(values), "the rendered set must be ordered deterministically"
    for i, p in enumerate(paths, 1):
        assert H[f"sourced_path_{i}"] == p.render()
        assert p.render() in doc
        assert " -> ".join(f"`{e}`" for e in p.edge_ids) in doc
        assert all(chain[e].is_sourced for e in p.edge_ids), "a published path used an unsourced edge"
        assert p.bias_direction == "lower" and p.render().startswith(">= ")
    # the two rival readings of one conversion land at the same coordinate with different numbers,
    # and their mean is nowhere in the document
    rivals = [p for p in paths if p.value.numeraire == "electrical_minimal"]
    assert len(rivals) == 2
    mean = statistics.mean(p.value.value_GeV for p in rivals)
    assert f"{mean:.2f}" not in doc, "a mean of two competing readings must never be rendered"
    assert H["competing_clause"] and " ".join(H["competing_clause"].split()) in doc
    # the composed figures reproduce the ledger's own electrical rows, to the digits it publishes
    published = {round(p.value.value_GeV, 2) for p in paths}
    ledger = {r.normalized_GeV_per_mu for r in table
              if r.has_normalized and r.numeraire != mucost.BEAM_KINETIC}
    assert ledger < published, "the composed set must contain the ledger rows and at least one more"


def test_the_generated_edge_section_says_what_stops_the_chain(table, chain):
    """The document must name the conversions that would continue the chain and may not be printed.

    A set of lower bounds with no statement of what stops them reads as though nothing more is
    known, when in fact the next factor is published and disowned by its own authors. That
    distinction -- an omitted factor is bounded above by 1, an arbitrary one is not bounded at all --
    is the sharpest thing the edge layer has to say, so it is stated where the figures are.
    """
    gen = _load_generator()
    H = gen.build_headline(table, chain)
    doc = _normalized("MUON_COST.md")
    assert "direction unknown" in doc
    assert "every conversion it omits is <= 1" in doc
    blocked = gen.blocked_extensions(chain)
    assert blocked, "nothing is blocked, so the clause would pass vacuously"
    assert int(H["n_blocked_extensions"]) == len(blocked)
    for e in blocked:
        assert not e.is_sourced and e.has_factor
        assert f"`{e.edge_id}`" in doc
    assert " ".join(H["blocked_clause"].split()) in doc
    # and the marker is not decoration: a sourced partial path earns '>=', an arbitrary one does not
    beam = table[gen.CHAIN_ANCHOR_ID].chain_point()
    assert mucost.compose_path(beam, [chain["eta_acc_kovach_site"]]).render().startswith(">= ")
    assert not mucost.compose_path(beam, [chain["delivery_kelly_eta_mu"]]).render().startswith(">= ")
