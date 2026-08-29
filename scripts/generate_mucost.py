"""Generate MUON_COST.md + figures/muon_cost_gap.png + MUON_COST_MANIFEST.json (deterministic).

    python scripts/generate_mucost.py

Content: the open muon-cost ledger rendered as tier tables + a
normalization-basis explainer + the tier-spread figure. This is a **curated
compilation with provenance, not an evaluation**: each cost is carried at its OWN
(stage, numeraire) coordinate and the rows are NOT on a common basis, so every aggregate is computed
within a single numeraire; wall-plug is a numeraire rather than a stage, so applying eta_acc produces a
separate row (kept separate); T3 facility rows are original derivations ("implied, derived here, formula
shown"); an accounting credit (Kelly's x2.5 recapture, stated in his abstract) is recorded in its own
flagged column, never folded into the normalized value.

Paywall/headline rule: the headline sentence cites Kelly (4.70, open access) FIRST as the named
anchor, then the full-text-verified Bertin and Eliezer-Henis values as corroboration (by DOI). Rows that
are needs_verification (Jandel) or slide-tier (Acceleron) are not in ``HEADLINE_ANCHOR_IDS``; a
slide-tier row can still sit inside a tier median.

Audit wiring: this generator regenerates all three artifacts; only MUON_COST.md +
MUON_COST_MANIFEST.json join the `git diff --exit-code` list -- the PNG is never byte-diffed
(matplotlib/freetype bytes are not cross-platform stable). All committed numbers are pure deterministic
arithmetic on the committed CSV (no MCMC/solver), so the two byte-diffed artifacts are cross-arch stable.

Computation lives in importable helpers (no side effects on import); file I/O + the figure + printing are
guarded behind ``main()`` so tests import and assert on the tables without regenerating the doc/figure.
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path

from openmucf import mucost, provenance
from openmucf.mucost import MUON_COST_CHAIN_CSV, MUON_COST_CSV

# The disarmament sentence -- goes VERBATIM in the figure caption. Em-dash intentional.
DISARMAMENT = (
    "Facilities optimize brightness/purity, not muons-per-watt "
    "— the floor is unvalidated, not impossible."
)

TIER_TITLES = {
    "T1-design-study": "Tier 1 -- purpose-built muon-source design studies",
    "T2-demonstrated-tech": "Tier 2 -- demonstrated technology",
    "T3-operating-facility": "Tier 3 -- operating facilities (GeV/muon derived here, at each row's own stage)",
}

# Short display labels used in the tables AND as manifest row anchors (unique per row).
LABELS = {
    "kelly_hart_rose_2021": "Kelly, Hart & Rose (2021)",
    "kelly_electrical_minimal": "Kelly / eta_acc minimal-subsystem",
    "kelly_electrical_site": "Kelly / eta_acc site-wide",
    "bertin_1987": "Bertin et al. (1987)",
    "eliezer_henis_1994": "Eliezer & Henis (1994)",
    "jandel_1989": "Jandel (1989)",
    "acceleron_2025": "Acceleron (2025 deck)",
    "muon_collider_front_end": "muon-collider front end",
    "mu2e": "mu2e (Fermilab)",
    "comet": "COMET (J-PARC)",
    "music": "MuSIC (RCNP)",
    "psi_himb": "PSI HIMB",
}

# --------------------------------------------------------------------------------------------------
# The Kou-Chen (2026) cycle-closure criterion, arXiv:2607.10989 (bibkey KouChenLawson2026).
#
# These are THEIR accounting choices, declared here as inputs so the comparison runs on their terms.
# None is fitted to anything of ours, and each carries its provenance in CRITERION_PROVENANCE below --
# including the two that are NOT sourced in their paper, which is itself part of the result.
# --------------------------------------------------------------------------------------------------
ETA_SYS = 1.0  # their sec.IV illustrative system efficiency (they also show 0.4)
G_MU_TARGET = 1.0  # the TARGET one-muon gain in eq.(15)/eq.(12); G_mu = 1 is breakeven
N_FUS = 150.0  # their Table I LAMPF/Jones row, which that table types a "literature anchor"
E_USE_KOUCHEN_MEV = 20.4  # useful energy per fusion cycle -- NOT sourced in their paper
# Kelly, Hart & Rose sec.2 derive a larger useful energy from the same fusion: the 17.6 MeV of fusion
# kinetic energy PLUS the exothermic tritium-breeding reactions each fusion neutron drives. Derived
# here from its three inputs rather than pasted, so the 26.0 MeV is checkable rather than asserted.
E_FUSION_KINETIC_MEV = 17.6
BREEDING_REACTIONS_PER_FUSION = 1.75
E_BREEDING_MEV = 4.8  # n + 6Li -> t + 4He + 4.8 MeV
E_USE_KELLY_MEV = E_FUSION_KINETIC_MEV + BREEDING_REACTIONS_PER_FUSION * E_BREEDING_MEV
MEV_PER_GEV = 1000.0

# The historically demonstrated effective sticking this is measured against, taken from OUR OWN rate
# ledger rather than retyped: the SIN campaign value that is also Kou-Chen's Table I SIN/Crowe anchor.
STICKING_RATE_SYMBOL = "omega_s_eff_solid_12K"

# The three ledger rows that form the sourced part of the chain, in chain order.
CHAIN_POINT_IDS = ("kelly_hart_rose_2021", "kelly_electrical_minimal", "kelly_electrical_site")
#: Short, stable manifest-key suffixes for those rows (source_ids are too long for readable keys).
_SHORT = {
    "kelly_hart_rose_2021": "beam",
    "kelly_electrical_minimal": "elecmin",
    "kelly_electrical_site": "elecsite",
}
#: The muon cost the criterion paper actually adopts (their historical Jones anchor), for contrast.
KOUCHEN_CONVENTIONAL_COST_GEV = 5.0

#: The rows the Headline names as design-study anchors, in the order it names them.
HEADLINE_ANCHOR_IDS = ("kelly_hart_rose_2021", "bertin_1987", "eliezer_henis_1994")

#: The ledger row the terminal-figure set is composed FROM. Not a preference among rows: it is the
#: only row whose own source states any conversion at all, so it is the only starting point for which
#: a composed figure uses that row's own literature rather than borrowing another paper's factor.
CHAIN_ANCHOR_ID = "kelly_hart_rose_2021"


def _join(items: list[str]) -> str:
    """'A' / 'A and B' / 'A, B and C' -- deterministic, no Oxford comma (the document's style)."""
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]

CRITERION_PROVENANCE = {
    "eta_sys": (
        "Kou-Chen sec.IV illustrative value; a lumped system efficiency. **They also report 0.4** -- in "
        "their sec.IV running text and as a Fig.2(a) legend entry, NOT in a table: their only table, "
        "Table I, states in its caption that its gain column is evaluated at `eta_sys` = 1 -- and "
        "`N_L` is linear in `1/eta_sys` exactly as it is in `1/E_use`, so this choice moves the answer the "
        "same way: on their own sec.IV arithmetic `eta_sys` = 0.4 at 5 GeV raises `N_L` to 613 and lowers "
        "`omega_crit` to 0.16%. Reported here at 1 to reproduce their headline panel, never as the only "
        "defensible value."
    ),
    "G_mu": "The TARGET one-muon gain in eq.(15) and the eq.(12) no-go boundary; G_mu = 1 is breakeven.",
    "N_fus": (
        "Kou-Chen Table I, LAMPF/Jones row. That table types it a *literature anchor* at `Y_f` ~ 150 "
        "with an adopted sticking value, not a measurement; its one row typed *experiment*, SIN/Crowe, "
        "is `Y_f` = 124 +/- 10. Adopted here as their own headline-panel input, not as a demonstrated yield."
    ),
    "E_use_kouchen": (
        "**not sourced in arXiv:2607.10989.** The paper attributes 'about 20 MeV' to the same Jones "
        "accounting that supplies its 5 GeV muon cost, so the useful-energy input rests on exactly the "
        "convention this axis is questioning. Adopted here only to reproduce their numbers on their terms."
    ),
    "E_use_kelly": (
        "primary-derived, and larger, because the same fusion also breeds tritium exothermically: "
        "17.6 MeV of fusion kinetic energy + 1.75 breeding reactions per fusion neutron x 4.8 MeV each "
        "(Kelly, Hart & Rose sec.2). Reported alongside the 20.4 MeV convention, never substituted for it."
    ),
}


def eq15_max_muon_cost_GeV(
    e_use_MeV: float, eta_sys: float = ETA_SYS, n_fus: float = N_FUS, g_mu: float = G_MU_TARGET
) -> float:
    """Kou-Chen eq.(15)'s ceiling, in the N_fus,mu form: (eta_sys * E_use / G_mu) * N_fus,mu.

    Their printed eq.(15) carries the cycle-strength factor L_mu / (1 + omega_eff * L_mu); that factor
    IS N_fus,mu by their eq.(2), so this is an exact rewriting on their algebra, NOT a form the paper
    prints -- the same disclosure MUON_COST.md and the bibliography carry for it.
    """
    return (eta_sys * e_use_MeV / g_mu) * n_fus / MEV_PER_GEV


def eq9_cycle_demand(e_cost_GeV: float, e_use_MeV: float, eta_sys: float = ETA_SYS) -> float:
    """Kou-Chen eq.(9): the cycle demand N_L = E_cost / (eta_sys * E_use) -- fusions one muon must buy."""
    return (e_cost_GeV * MEV_PER_GEV) / (eta_sys * e_use_MeV)


def eq8_one_muon_gain(n_L: float, n_fus: float = N_FUS) -> float:
    """Kou-Chen eq.(8)'s gain, in the N_fus,mu / N_L form: their eq.(8) taken with their eq.(9).

    Eq.(8) prints G_mu = (eta_sys * E_use / E_cost) * N_fus,mu and eq.(9) defines
    N_L = E_cost / (eta_sys * E_use); together they give this quotient, which the paper prints
    nowhere -- an exact rewriting on their algebra, exactly as the eq.(15) ceiling above is one.
    Their eq.(10) writes the same gain from the cycle-strength side,
    G_mu = (1/N_L) * L_mu / (1 + omega_eff * L_mu), which they introduce in their own words as
    "Substituting Eq. (2) into Eq. (8) gives"; eq.(2) collapses that bracket to N_fus,mu, so
    eq.(10) and this quotient are the same quantity written two ways.
    """
    return n_fus / n_L


def eq12_omega_crit(n_L: float, g_mu: float = G_MU_TARGET) -> float:
    """Kou-Chen eq.(12): the sticking no-go boundary, omega_eff < 1 / (G_mu * N_L). Returns a FRACTION."""
    return 1.0 / (g_mu * n_L)


def _fmt(v: float) -> str:
    """Deterministic display of a normalized GeV/muon value (byte-stable; pure arithmetic)."""
    if v >= 100:
        return f"{v:.0f}"
    if v == int(v):
        return f"{v:.1f}"
    return f"{v:.2f}"


def _fmt_med(v: float) -> str:
    """Deterministic display of a tier median (keeps the .5 on 5497.5; drops trailing on 178.0)."""
    return f"{v:g}"


def build_headline(
    table: mucost.MuonCostTable, chain: mucost.ChainEdgeTable | None = None
) -> dict[str, str]:
    """Single source of truth: the formatted strings shared by MUON_COST.md and the manifest."""
    if chain is None:
        chain = mucost.load_muon_cost_chain()
    H: dict[str, str] = {}
    # per-row normalized display (pinned rows only; Jandel has no value)
    for r in table:
        if r.has_normalized:
            H[f"norm_{r.source_id}"] = _fmt(r.normalized_GeV_per_mu)
    # The Headline's basis claim is a statement about OUR OWN rows, so it is DERIVED from their
    # `stage` rather than typed. A typed universal here read "Those single-GeV figures are beam
    # energy per muon PRODUCED" over all three anchors while the ledger carried one of them at the
    # terminal stage -- a basis claim the same document contradicted sixty lines below it.
    at_produced = [LABELS[s] for s in HEADLINE_ANCHOR_IDS if table[s].stage == "produced"]
    off_produced = [
        f"{LABELS[s]} (`{table[s].stage}`)"
        for s in HEADLINE_ANCHOR_IDS
        if table[s].stage != "produced"
    ]
    # The heterogeneity claim is derived too, not just the row lists: a typed "are not on one basis"
    # prefix in front of a derived clause would contradict itself the moment every anchor shared a
    # stage, which is the same shape of defect one axis down.
    if not at_produced and not off_produced:  # no anchors at all: fail loudly rather than render prose
        raise ValueError("HEADLINE_ANCHOR_IDS is empty; the Headline has no anchors to describe")
    if at_produced and off_produced:
        H["anchor_basis_sentence"] = (
            f"Those single-GeV figures are not on one basis: {_join(at_produced)} "
            f"{'is' if len(at_produced) == 1 else 'are'} beam energy per muon PRODUCED, while "
            f"{_join(off_produced)} {'is' if len(off_produced) == 1 else 'are'} carried at a "
            "different stage"
        )
    elif at_produced:
        H["anchor_basis_sentence"] = (
            f"Those single-GeV figures are all at stage `produced`: {_join(at_produced)} "
            f"{'is' if len(at_produced) == 1 else 'are'} beam energy per muon PRODUCED"
        )
    else:
        H["anchor_basis_sentence"] = (
            f"None of those single-GeV figures is at stage `produced`: {_join(off_produced)} "
            f"{'is' if len(off_produced) == 1 else 'are'} carried at another stage"
        )
    # The chain-point table's stage claim, derived for the same reason.
    chain_stages = sorted({table[s].stage for s in CHAIN_POINT_IDS})
    if not chain_stages:  # no chain points at all: fail loudly rather than render prose
        raise ValueError("CHAIN_POINT_IDS is empty; the chain-point table has no stages to describe")
    H["chain_points_stage_clause"] = (
        f"they all stop at stage `{chain_stages[0]}`"
        if len(chain_stages) == 1
        else "they stop at " + _join([f"`{s}`" for s in chain_stages]) + " rather than at one stage"
    )
    # tier medians + the gap
    m1 = table.tier_median("T1-design-study")
    m2 = table.tier_median("T2-demonstrated-tech")
    m3 = table.tier_median("T3-operating-facility")
    H["t1_median"] = _fmt_med(m1)
    H["t2_median"] = _fmt_med(m2)
    H["t3_median"] = _fmt_med(m3)
    H["gap_ratio"] = f"{m3 / m1:.1f}"
    H["disarmament"] = DISARMAMENT
    # WHICH rows each median is taken over, and which the charge-basis rule kept out. Both are derived
    # from the ledger rather than described, because a median whose membership is asserted in prose can
    # drift from the rows it is actually computed on -- which is the defect this document is amending.
    H["t1_median_rows"] = _median_membership(table, "T1-design-study")
    H["t3_median_rows"] = _median_membership(table, "T3-operating-facility")
    H["aggregate_excluded_clause"] = _aggregate_exclusion_clause(table)
    # Basis composition -- computed, so the disclosure can never drift from the CSV. No basis_class is
    # shared between T1 and T3, so a same-basis T1-vs-T3 ratio is NOT COMPUTABLE from these rows.
    H["t1_classes"] = ", ".join(sorted(table.basis_classes("T1-design-study")))
    H["t3_classes"] = ", ".join(sorted(table.basis_classes("T3-operating-facility")))
    shared = table.basis_classes("T1-design-study") & table.basis_classes("T3-operating-facility")
    H["shared_classes"] = ", ".join(sorted(shared)) if shared else "none"
    # The wall-plug-equivalent figures, i.e. the ones whose numeraire conversion is sourced. Each is a
    # LEDGER ROW of its own rather than an inline recomputation, so the repo carries each number once,
    # on its own electrical denominator. ("fully-sourced CHAIN" is a different and stronger property,
    # counted separately below; no row in this ledger has one.)
    kelly = table["kelly_hart_rose_2021"]
    H["kelly_eta_acc"] = f"{kelly.eta_acc_assumption:g}"
    H["kelly_wallplug"] = _fmt(table["kelly_electrical_minimal"].normalized_GeV_per_mu)
    H["kelly_eta_acc_site"] = f"{table['kelly_electrical_site'].eta_acc_assumption:g}"
    H["kelly_wallplug_site"] = _fmt(table["kelly_electrical_site"].normalized_GeV_per_mu)
    # eta_mu is PUBLISHED twice in the prose below, so it must be read from the ledger like every other
    # published number. Typing it into the template would let the CSV move while the document kept
    # printing the old digit, and no guard would notice: `provenance --check` only compares the manifest
    # against the document, so a literal that appears in both stays self-consistent while both are stale.
    H["kelly_eta_mu"] = f"{kelly.eta_mu_assumption:.2f}"
    H["kelly_recapture"] = f"{kelly.recapture_factor:g}"
    # charge_basis is PUBLISHED in the accounting-basis prose, so it is read from the ledger for the
    # same reason eta_mu is: typing it would let the CSV move while the document kept asserting the
    # old basis, and both the document and the manifest would stay self-consistent while both were stale.
    H["charge_music"] = table["music"].charge_basis
    H["charge_psi_himb"] = table["psi_himb"].charge_basis

    # ---- the Kou-Chen cycle-closure comparison (all derived, nothing transcribed) ----
    H["eta_sys"] = f"{ETA_SYS:g}"
    H["g_mu_target"] = f"{G_MU_TARGET:g}"
    H["n_fus"] = f"{N_FUS:g}"
    # one decimal on both, so the primary-derived 17.6 + 1.75*4.8 reads as 26.0 rather than a bare "26"
    H["e_use_kouchen"] = f"{E_USE_KOUCHEN_MEV:.1f}"
    H["e_use_kelly"] = f"{E_USE_KELLY_MEV:.1f}"
    for key, e_use in (("kc", E_USE_KOUCHEN_MEV), ("kelly", E_USE_KELLY_MEV)):
        H[f"ceiling_{key}"] = f"{eq15_max_muon_cost_GeV(e_use):.2f}"

    omega_anchor = _sticking_anchor()
    H["omega_anchor"] = f"{omega_anchor:g}"
    for sid in CHAIN_POINT_IDS:
        cv = table[sid].chain_point()
        short = _SHORT[sid]
        H[f"chain_{short}"] = cv.render()  # ">= X.XX GeV" -- the bound marker is part of the pinned string
        for key, e_use in (("kc", E_USE_KOUCHEN_MEV), ("kelly", E_USE_KELLY_MEV)):
            n_L = eq9_cycle_demand(cv.value_GeV, e_use)
            H[f"ratio_{short}_{key}"] = f"{cv.value_GeV / eq15_max_muon_cost_GeV(e_use):.2f}"
            H[f"nl_{short}_{key}"] = f"{n_L:.1f}"
            H[f"gmu_{short}_{key}"] = f"{eq8_one_muon_gain(n_L):.3f}"
            H[f"omegacrit_{short}_{key}"] = f"{eq12_omega_crit(n_L) * 100.0:.3g}"
            H[f"overshoot_{short}_{key}"] = f"{omega_anchor / (eq12_omega_crit(n_L) * 100.0):.1f}"
    # their own 5 GeV convention, for the same three columns -- the row this program is questioning
    n_L_conv = eq9_cycle_demand(KOUCHEN_CONVENTIONAL_COST_GEV, E_USE_KOUCHEN_MEV)
    H["conventional_cost"] = f"{KOUCHEN_CONVENTIONAL_COST_GEV:g}"
    H["nl_conventional"] = f"{n_L_conv:.1f}"
    H["gmu_conventional"] = f"{eq8_one_muon_gain(n_L_conv):.3f}"
    H["omegacrit_conventional"] = f"{eq12_omega_crit(n_L_conv) * 100.0:.3g}"
    H["overshoot_conventional"] = f"{omega_anchor / (eq12_omega_crit(n_L_conv) * 100.0):.1f}"
    # The optimism factor and the one-decimal conventional gain are PUBLISHED in the claim paragraph, so
    # they are computed from the same two ledger rows and the same constant the table above prints. Typed
    # as literals they would keep asserting "~5-9x" and "~0.6" after the CSV moved the costs they are
    # ratios of -- which is exactly the defect eta_mu, x2.5 and charge_basis each turned out to be.
    H["optimism_low"] = (
        f"{table['kelly_electrical_minimal'].normalized_GeV_per_mu / KOUCHEN_CONVENTIONAL_COST_GEV:.0f}"
    )
    H["optimism_high"] = (
        f"{table['kelly_electrical_site'].normalized_GeV_per_mu / KOUCHEN_CONVENTIONAL_COST_GEV:.0f}"
    )
    H["gmu_conventional_1dp"] = f"{eq8_one_muon_gain(n_L_conv):.1f}"

    # Coverage is the deliverable. Count what is actually sourced across the whole ledger.
    cov = coverage_rows(table)
    H["n_chain_rows"] = str(sum(1 for c in cov if c["on_chain"]))
    H["n_offchain_rows"] = str(sum(1 for c in cov if not c["on_chain"]))
    H["n_fully_sourced_chains"] = str(sum(1 for c in cov if c["complete"]))
    H["n_numeraire_sourced"] = str(sum(1 for c in cov if c["numeraire_sourced"]))
    # The closing universal reaches only rows that count mu-. A `mu_plus_only` row prices no mu- at
    # all, so a bound on its mu--only cost holds without saying anything; the exclusion is derived
    # from `charge_basis` so that recharging a row rewrites the sentence instead of leaving it stale.
    mu_plus_only = [
        LABELS[r.source_id]
        for r in table
        if r.has_normalized and r.stage in mucost.MUCF_CHAIN and r.charge_basis == "mu_plus_only"
    ]
    H["mu_plus_only_clause"] = (
        ""
        if not mu_plus_only
        else "The mu+-only chain {} ({}) {} excluded on the other axis: {} no mu-, so {}\n"
             "mu--only cost is unbounded and any figure bounds it vacuously.".format(
                 "figure" if len(mu_plus_only) == 1 else "figures",
                 _join(mu_plus_only),
                 "is" if len(mu_plus_only) == 1 else "are",
                 "it counts" if len(mu_plus_only) == 1 else "they count",
                 "its" if len(mu_plus_only) == 1 else "their",
             )
    )

    # ---- the EDGE layer: what the conversions themselves are, and which of them are sourced ----
    # The headline sentence below is DERIVED from the two tables at generation time and not typed.
    # A typed version of it would keep asserting a coverage the CSVs had moved out from under, which
    # is the same defect the derived membership and basis sentences above already retired one layer
    # up; and the sentence is the one a reader is most likely to quote, so it is the one that must
    # not be able to go stale.
    cov = edge_coverage_rows(table, chain)
    links = [c for c in cov if c["kind"] == mucost.STAGE_EDGE]
    numeraire_convs = [c for c in cov if c["kind"] == mucost.NUMERAIRE_EDGE]
    sourced = [c for c in cov if c["sourced"]]
    sourced_links = [c for c in links if c["sourced"]]
    H["n_chain_edges"] = str(len(chain))
    H["n_absent_edges"] = str(sum(1 for e in chain if not e.has_factor))
    H["n_required_conversions"] = str(len(cov))
    H["n_stage_conversions"] = str(len(links))
    H["n_numeraire_conversions"] = str(len(numeraire_convs))
    H["n_sourced_conversions"] = str(len(sourced))
    H["n_sourced_stage_conversions"] = str(len(sourced_links))

    # which cost sources state their own beam-to-electrical conversion, read off the ledger
    carriers = [r for r in table if not math.isnan(r.eta_acc_assumption)]
    carrier_names = _join(
        sorted({LABELS[r.source_id] for r in carriers if r.numeraire == mucost.BEAM_KINETIC})
    )
    carrier_keys = {r.source_bibkey.split(";")[0].strip() for r in carriers}
    # ...and whether those same sources state a delivery factor they call unsourced
    declared_delivery = [
        e for e in chain
        if e.kind == mucost.STAGE_EDGE
        and e.has_factor
        and not e.is_sourced
        and e.source_bibkey.split(";")[0].strip() in carrier_keys
    ]
    n_carriers = len({r.source_bibkey.split(";")[0].strip() for r in carriers})

    if sourced_links:
        links_clause = (
            f"**{len(sourced_links)} of the {len(links)} stage advances** {'is' if len(sourced_links) == 1 else 'are'} "
            f"sourced: {_join([c['label'] for c in sourced_links])}"
        )
    else:
        links_clause = (
            f"**not one of the {len(links)} stage advances the chain requires is sourced by any "
            f"primary read here**"
        )
    kind_clause = (
        "every one of them is a numeraire change"
        if sourced and all(c["kind"] == mucost.NUMERAIRE_EDGE for c in sourced)
        else _join([c["label"] for c in sourced]) or "none of them"
    )
    one_carrier = n_carriers == 1
    one_delivery = len(declared_delivery) == 1
    how_many_delivery = (
        "the chain's one delivery factor"
        if one_delivery
        else f"{len(declared_delivery)} of the chain's delivery factors"
    )
    delivery_clause = (
        ""
        if not declared_delivery
        else (
            f" {'That source' if one_carrier else 'Those sources'} "
            f"also state{'s' if one_carrier else ''} "
            f"{how_many_delivery} "
            f"({_join([f'`{e.edge_id}`' for e in declared_delivery])}), and "
            f"grade{'s' if one_carrier else ''} "
            f"{'it' if one_delivery else 'every one of them'} "
            f"{_join(sorted({f'`{e.evidence_status}`' for e in declared_delivery}))} -- so composing "
            f"{'it' if one_delivery else 'them'} yields a bound and never a value."
        )
    )
    H["chain_coverage_sentence"] = (
        f"Of the {len(cov)} conversions a fully-sourced chain needs -- {len(links)} stage advances "
        f"along the muCF chain and {len(numeraire_convs)} numeraire "
        f"{'change' if len(numeraire_convs) == 1 else 'changes'} out of `beam_kinetic` -- "
        f"**{len(sourced)}** carry a factor from a primary read here, and {kind_clause}: "
        f"{links_clause}. Exactly {n_carriers} cost "
        f"{'source' if n_carriers == 1 else 'sources'} in this compilation "
        f"({carrier_names}) {'states its' if n_carriers == 1 else 'state their'} own "
        f"beam-to-electrical conversion.{delivery_clause}"
    )

    # the figures the competing conversions lead to: a SET, with its provenance. Sourced paths only
    # -- the document prints no figure composed through a factor its own authors call arbitrary.
    paths = sourced_paths(table, chain)
    H["n_sourced_paths"] = str(len(paths))
    for i, p in enumerate(paths, 1):
        H[f"sourced_path_{i}"] = p.render()
        H[f"sourced_path_coord_{i}"] = f"{p.value.stage} / {p.value.numeraire}"
    blocked = blocked_extensions(chain)
    # NOT every blocked conversion is direction-unknown, and saying so of all of them would be the
    # overclaim this layer exists to catch: a factor a source states at its ceiling while saying the
    # truth lies below it still biases one way. The two are separated by their own bias_direction.
    undirected = [e for e in blocked if e.bias_direction == "unknown"]
    H["n_blocked_extensions"] = str(len(blocked))
    blocked_clause = (
        "No stated conversion in the edge table is barred from these paths."
        if not blocked
        else "{} the chain could be continued with {} stated but not sourced: {}. None of the "
             "figures here is composed through {}.".format(
                 "Beyond that," if len(blocked) == 1 else "Beyond those,",
                 "one conversion" if len(blocked) == 1 else f"{len(blocked)} conversions",
                 _join([
                     f"`{e.edge_id}` ({e.evidence_status}, bias `{e.bias_direction}`)"
                     for e in blocked
                 ]),
                 "it" if len(blocked) == 1 else "any of them",
             )
    )
    if undirected:
        blocked_clause += (
            " {} not even one-sided: {} own authors state they do not know the {}, so a figure "
            "built through {} could be too high or too low rather than bounded below."
        ).format(
            f"`{undirected[0].edge_id}` is"
            if len(undirected) == 1
            else _join([f"`{e.edge_id}`" for e in undirected]) + " are",
            "its" if len(undirected) == 1 else "their",
            "value" if len(undirected) == 1 else "values",
            "it" if len(undirected) == 1 else "them",
        )
    H["blocked_clause"] = blocked_clause
    # The marker sentence is derived for the same reason the coverage sentence is. "Each figure
    # prints >= because it is a one-sided lower bound" is true of every published figure today and
    # silently false the day one of them is not -- which is exactly the staleness the paragraph
    # above it was built to retire.
    printed_biases = sorted({p.bias_direction for p in paths})
    if printed_biases == ["lower"]:
        H["marker_clause"] = (
            "Each figure above prints `>=` because it is a one-sided **lower** bound: every "
            "conversion it omits is <= 1, so leaving it out can only understate the cost."
        )
    elif printed_biases == ["none"]:
        H["marker_clause"] = (
            "Each figure above prints as a plain value: its path reaches the terminal stage with "
            "every conversion sourced, so nothing about it is a bound."
        )
    else:
        H["marker_clause"] = (
            "The figures above do not all carry the same marker ({}). A `>=` is a one-sided "
            "**lower** bound -- every conversion that figure omits is <= 1, so leaving it out can "
            "only understate the cost -- while a figure marked *direction unknown* is composed "
            "through a factor its own authors call arbitrary and is bounded in neither "
            "direction.".format(_join([f"`{b}`" for b in printed_biases]))
        )
    competing = chain.competing()
    H["n_competing_conversions"] = str(len(competing))
    H["competing_clause"] = (
        "No conversion in the edge table carries more than one stated value today."
        if not competing
        else "{} {} more than one stated value: {}. Both readings are carried to the end as "
             "separate figures -- no mean is formed and no value is preferred.".format(
                 len(competing),
                 "conversion carries" if len(competing) == 1 else "conversions carry",
                 _join([
                     "{} ({})".format(
                         f"`{k[1]}` -> `{k[3]}`" if k[0] == mucost.ANY else f"`{k[0]}` -> `{k[2]}`",
                         _join([f"`{e.edge_id}` = {e.factor:g}" for e in v]),
                     )
                     for k, v in competing.items()
                 ]),
             )
    )
    return H


def _median_membership(table: mucost.MuonCostTable, tier: str) -> str:
    """'LABEL v, LABEL v and LABEL v GeV' for the rows a tier median is actually taken over.

    Ordered by value so the rendering is deterministic and a reader can see the median position
    directly. Reads :meth:`~openmucf.mucost.MuonCostTable.aggregate_rows`, i.e. exactly the set the
    median is computed on -- not a hand-kept list that could disagree with it.
    """
    rows = sorted(table.aggregate_rows(tier=tier), key=lambda r: r.normalized_GeV_per_mu)
    if not rows:  # an empty aggregate must fail loudly rather than render prose about no rows
        raise ValueError(f"tier {tier!r} has no aggregable rows; the median has no membership to state")
    return _join([f"{LABELS[r.source_id]} {_fmt(r.normalized_GeV_per_mu)}" for r in rows]) + " GeV"


def _aggregate_exclusion_clause(table: mucost.MuonCostTable) -> str:
    """The sentence naming every row the charge-basis rule keeps OUT of every aggregate.

    Derived, so recharging a row rewrites this sentence instead of leaving it stale, and so the
    exclusion is disclosed rather than silently applied: the rows named here are still rendered in
    their tier tables with their own labels, they simply enter no median, spread or ratio.
    """
    excluded = sorted(
        table.rows_excluded_from_aggregates(), key=lambda r: r.normalized_GeV_per_mu
    )
    if not excluded:
        return (
            "No row in this ledger carries a charge basis the aggregate rule excludes, so every "
            "pinned `beam_kinetic` row enters the medians above."
        )
    names = _join([f"{LABELS[r.source_id]} (`{r.charge_basis}`)" for r in excluded])
    is_are = "is" if len(excluded) == 1 else "are"
    it_they = "it prices" if len(excluded) == 1 else "they price"
    return (
        f"{names} {is_are} kept OUT of every aggregate here -- {it_they} no mu- at all, and the "
        f"ledger schema bars such a figure from any muCF cost aggregate -- any statistic formed over "
        f"rows: a tier median, a spread, a ratio, or a prior-box edge -- so the row stays in its "
        f"tier table for scale and enters no median, spread or ratio. The exclusion is applied at "
        f"the aggregate, never at the row."
    )


def _sticking_anchor() -> float:
    """The demonstrated effective sticking (percent), read from OUR rate ledger, never retyped."""
    from openmucf import load_rates

    return float(load_rates()[STICKING_RATE_SYMBOL].value)


def coverage_rows(table: mucost.MuonCostTable) -> list[dict]:
    """Per source: how far the chain actually gets, and which conversions are sourced.

    A row is ``complete`` only if it reaches the terminal stage with every factor sourced -- i.e. only
    if its :class:`~openmucf.mucost.ChainValue` is not a bound. Rows stopped outside D-T fuel are not
    on the muCF chain at all and are reported separately rather than silently dropped.
    """
    rows = []
    for r in table:
        if not r.has_normalized:
            continue
        on_chain = r.stage in mucost.MUCF_CHAIN
        cv = r.chain_point() if on_chain else None
        if r.stage == mucost.TERMINAL_STAGE:
            delivery = (
                "reached"
                if r.useful_fraction_sourced
                else "reached, but the source never establishes the 'useful' qualifier"
            )
        elif r.eta_mu_evidence_status:
            delivery = f"stated as one collapsed factor, {r.eta_mu_evidence_status}"
        else:
            delivery = "absent"
        rows.append(
            {
                "source_id": r.source_id,
                "label": LABELS[r.source_id],
                "on_chain": on_chain,
                "stage": r.stage,
                "numeraire": r.numeraire,
                "numeraire_sourced": not math.isnan(r.eta_acc_assumption),
                "delivery": delivery,
                "complete": bool(cv is not None and not cv.is_bound),
            }
        )
    return rows


def required_conversions(table: mucost.MuonCostTable) -> list[dict]:
    """The conversions a fully-sourced chain would need, derived rather than listed.

    The stage advances are the consecutive links of :data:`~openmucf.mucost.MUCF_CHAIN`; the numeraire
    changes are the ones out of ``beam_kinetic`` into each electrical numeraire the ledger's own
    pinned rows are actually counted in. Deriving the second from the ledger means a new numeraire in
    the CSV enters the coverage table instead of being silently uncounted.
    """
    out = [
        {"kind": mucost.STAGE_EDGE, "label": f"`{a}` -> `{b}`", "key": (a, b)}
        for a, b in zip(mucost.MUCF_CHAIN[:-1], mucost.MUCF_CHAIN[1:], strict=True)
    ]
    electrical = sorted(
        {r.numeraire for r in table if r.has_normalized and r.numeraire}
        - {mucost.BEAM_KINETIC}
    )
    out += [
        {
            "kind": mucost.NUMERAIRE_EDGE,
            "label": f"`{mucost.BEAM_KINETIC}` -> `{n}`",
            "key": (mucost.BEAM_KINETIC, n),
        }
        for n in electrical
    ]
    return out


def edge_coverage_rows(table: mucost.MuonCostTable, chain: mucost.ChainEdgeTable) -> list[dict]:
    """Per required conversion: which edges cover it, and whether any of them is sourced.

    A stage edge covers every link it spans, so a source that collapses several conversions into one
    factor is credited against each of them -- and, because the factor keeps its own evidence status,
    a collapsed arbitrary factor covers those links without sourcing any of them. That distinction is
    the whole content of this table.
    """
    rows = []
    for conv in required_conversions(table):
        if conv["kind"] == mucost.STAGE_EDGE:
            covering = [
                e for e in chain
                if e.kind == mucost.STAGE_EDGE and conv["key"][1] in e.spans
            ]
        else:
            covering = [
                e for e in chain
                if e.kind == mucost.NUMERAIRE_EDGE
                and (e.from_numeraire, e.to_numeraire) == conv["key"]
            ]
        rows.append(
            {
                "kind": conv["kind"],
                "label": conv["label"],
                "edges": covering,
                "sourced": [e for e in covering if e.is_sourced],
                "stated": [e for e in covering if e.has_factor],
            }
        )
    return rows


def sourced_paths(table: mucost.MuonCostTable, chain: mucost.ChainEdgeTable) -> list:
    """Every maximal path out of the anchor row **built only from sourced conversions**.

    The restriction to sourced edges is the document's rule, not the API's: ``compose_path`` will
    compose an author-declared-arbitrary factor and mark the result *direction unknown*, and the edge
    table carries those conversions so a reader can see them. What this document may PRINT is
    narrower -- no figure here is composed through a factor whose own authors say they do not know
    it, which is the rule ``test_no_headline_number_depends_on_an_arbitrary_row`` enforces and the
    reason the accounting-basis section can say the delivery factor is folded into nothing.

    Sorted by figure so the set reads as a set; the tie-break on the edge ids keeps the order total
    (and the rendered document deterministic) if two paths ever agree to the last bit.
    """
    paths = mucost.enumerate_chain_paths(table[CHAIN_ANCHOR_ID].chain_point(), chain.sourced())
    return sorted(paths, key=lambda p: (p.value.value_GeV, p.edge_ids))


def blocked_extensions(chain: mucost.ChainEdgeTable) -> list:
    """The conversions that would continue a sourced path, and cannot: stated but not sourced.

    Derived so the document can say WHAT stops the chain rather than merely that something does.
    """
    return [e for e in chain if e.has_factor and not e.is_sourced]


def _edge_table(chain: mucost.ChainEdgeTable) -> str:
    head = (
        "| edge | conversion | factor | evidence | bias | source |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = []
    for e in chain:
        if e.kind == mucost.NUMERAIRE_EDGE:
            conv = f"`{e.from_numeraire}` -> `{e.to_numeraire}` (at any stage)"
        else:
            conv = f"`{e.from_stage}` -> `{e.to_stage}` (in any numeraire)"
        factor = f"{e.factor:g}" if e.has_factor else "-- (no number)"
        source = e.source_bibkey.replace(";", ", ") if e.source_bibkey else "--"
        lines.append(
            f"| `{e.edge_id}` | {conv} | {factor} | {e.evidence_status} | {e.bias_direction} | "
            f"{source} |"
        )
    return head + "\n".join(lines)


def _edge_coverage_table(rows: list[dict]) -> str:
    head = (
        "| conversion | kind | edges that cover it | any sourced? |\n"
        "|---|---|---|---|\n"
    )
    lines = []
    for c in rows:
        covering = ", ".join(f"`{e.edge_id}` ({e.evidence_status})" for e in c["edges"]) or "none"
        lines.append(
            f"| {c['label']} | {c['kind']} | {covering} | "
            f"{'yes' if c['sourced'] else '**no**'} |"
        )
    return head + "\n".join(lines)


def _sourced_path_table(paths: list) -> str:
    head = "| conversions applied | coordinate reached | figure |\n|---|---|---|\n"
    lines = []
    for p in paths:
        applied = " -> ".join(f"`{i}`" for i in p.edge_ids)
        lines.append(f"| {applied} | `{p.value.stage}` / `{p.value.numeraire}` | {p.render()} |")
    return head + "\n".join(lines)


def _tier_table(table: mucost.MuonCostTable, tier: str, H: dict[str, str]) -> str:
    head = (
        "| source | value as published | GeV/muon | numeraire | stage | charge | evidence | nv |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in table.tier(tier):
        norm = H[f"norm_{r.source_id}"] if r.has_normalized else "-- (not pinned)"
        nv = "**yes**" if r.needs_verification else "no"
        num = r.numeraire or "--"
        st = r.stage or "--"
        ch = r.charge_basis or "--"
        # flag the rows whose figure understates the cost per mu- stopped and useful in D-T fuel
        if r.understates_stopped_in_dt_cost:
            st += " (lower bound)"
        elif r.stage == mucost.TERMINAL_STAGE and not r.useful_fraction_sourced:
            st += " ('useful' not established)"
        if r.charge_basis == "mu_plus_only":
            ch += " (not muCF)"
        if r.stage in mucost.OFF_CHAIN_STAGES:
            st += " (off-chain)"
        rows.append(
            f"| {LABELS[r.source_id]} | {r.value_as_published} | {norm} | {num} | {st} | {ch} | "
            f"{r.evidence_status} | {nv} |"
        )
    return head + "\n".join(rows)


def _coverage_table(table: mucost.MuonCostTable) -> str:
    """Print, per source, how far the chain gets and which conversions are sourced."""
    head = (
        "| source | stage reached | numeraire | beam -> electrical sourced? | "
        "produced -> stopped & useful in D-T | fully sourced? |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = []
    for c in coverage_rows(table):
        stage = c["stage"] if c["on_chain"] else f"{c['stage']} (NOT on the muCF chain)"
        conv = "yes (`eta_acc`, from the PSI primary)" if c["numeraire_sourced"] else "absent"
        lines.append(
            f"| {c['label']} | {stage} | `{c['numeraire']}` | {conv} | {c['delivery']} | "
            f"{'yes' if c['complete'] else '**no**'} |"
        )
    return head + "\n".join(lines)


def build_markdown(
    table: mucost.MuonCostTable, H: dict[str, str], chain: mucost.ChainEdgeTable | None = None
) -> str:
    if chain is None:
        chain = mucost.load_muon_cost_chain()
    t1 = _tier_table(table, "T1-design-study", H)
    t2 = _tier_table(table, "T2-demonstrated-tech", H)
    t3 = _tier_table(table, "T3-operating-facility", H)
    return f"""# MUON_COST.md -- the open muon-cost ledger (auto-generated by `scripts/generate_mucost.py`)

> **Curated compilation with provenance, NOT an evaluation.** `normalized_GeV_per_mu` is energy per muon
> in GeV at **each row's own (`stage`, `numeraire`) coordinate** -- these rows are **NOT on a common basis
> and are not commensurable across either axis**, so every aggregate below is computed within one
> numeraire and discloses its stage composition. T3 facility rows are ORIGINAL DERIVATIONS ("implied,
> derived here, formula shown") -- no operating facility reports a per-stopped-muon energy cost. An
> accounting credit (Kelly's x{H['kelly_recapture']} recapture) is recorded in its own flagged column, never folded into the
> normalized value, and a factor whose own authors call it arbitrary (Kelly's `eta_mu`) is recorded but
> **never composed into any figure quoted here**.

## Headline
The purpose-built muon-source **design studies** put the muon cost at a few GeV per muon. The open-access
anchor is **{LABELS['kelly_hart_rose_2021']}: {H['norm_kelly_hart_rose_2021']} GeV/muon** (G4Beamline;
deuteron on a tungsten target; DOI 10.1088/2515-7655/abfb4b) -- the only fully reader-checkable row,
which reports its own eta_acc={H['kelly_eta_acc']} (PSI-measured) and Q_elec=14% at X_mu=150. Two further
full-text-verified design studies corroborate the same single-GeV scale: **{LABELS['bertin_1987']},
~{H['norm_bertin_1987']} GeV/muon** at liquid density (DOI 10.1209/0295-5075/4/8/003; ~3 GeV ideal
all-collected) and **{LABELS['eliezer_henis_1994']}, ~{H['norm_eliezer_henis_1994']} GeV/muon**
(DOI 10.13182/FST94-A30300).

**{H['anchor_basis_sentence']}.** On Kelly's own
accelerator efficiency the same muon costs {H['norm_kelly_hart_rose_2021']} /
{H['kelly_eta_acc']} = **{H['kelly_wallplug']} GeV per muon produced** in ELECTRICAL energy, and on the
same primary's site-wide denominator ({H['kelly_eta_acc_site']}) it costs **{H['kelly_wallplug_site']} GeV**
-- and both are still LOWER BOUNDS on the electrical cost per mu- actually stopped and useful in D-T,
because the capture, transport, moderation and stopping factors (all <= 1) have not been applied. Every
factor omitted here pushes the cost UP, so the bound is one-sided.

**The tier spread is about three orders of magnitude, on MIXED bases.** The tier-median rises from **{H['t1_median']}
GeV** (design studies) through **{H['t2_median']} GeV** (demonstrated technology, collected-not-stopped)
to **{H['t3_median']} GeV** (operating facilities) -- nominally **{H['gap_ratio']}x**. **That ratio is
NOT a same-basis comparison and must not be quoted as one.** T1 contains {{{H['t1_classes']}}} rows and T3
contains {{{H['t3_classes']}}} rows; basis classes shared between the two tiers: **{H['shared_classes']}**.
With no shared class, a same-basis T1-vs-T3 ratio is **not computable from these rows** -- and because
both the numerator and the denominator contain lower-bound (per-produced / per-collected) figures, the
ratio is not cleanly bounded in either direction. The defensible statement is the spread's **order of
magnitude**, driven by technology, with the basis composition disclosed above. Neither the
needs_verification row (Jandel) nor the slide-tier row (Acceleron) is a named headline anchor: the
Jandel row has no pinned value and enters no aggregate, while the Acceleron row is one of the T1
median rows named next.

**Which rows each median is taken over.** T1: {H['t1_median_rows']}. T3: {H['t3_median_rows']}.
{H['aggregate_excluded_clause']}

## Accounting basis (read before the tables)
A muon cost is only meaningful as a point on a **2-D grid**, and both coordinates are carried per row:

**Axis 1 -- `stage`: how far along the chain the muon has got.** The muCF chain is
`produced -> captured -> transported -> moderated -> stopped_useful_in_dt`, following the five verbs by
which the cycle-closure literature defines the muon cost. Only the terminal stage is what a muCF energy
balance actually needs; every earlier stage is a lower bound on it, because each omitted
conversion factor is <= 1. `stopped_other_target` (mu2e, COMET -- muons stopped in aluminium) is **not a
point on this chain at all**: it prices stopping a muon somewhere that is not D-T fuel, so no chain of
sub-unity factors connects it to a muCF cost and it is not a bound on one. Such a row is still carried in
its tier table and read by the aggregates over that tier -- part of why the tier spread above is on MIXED
bases -- while `chain_point()` refuses to compose it.

**Axis 2 -- `numeraire`: what kind of energy is being counted.** `beam_kinetic` is beam energy;
`electrical_minimal` and `electrical_site` are electrical energy on the two different facility
denominators the same PSI primary supplies (see the tables). **Wall-plug is a numeraire, not a stage:**
dividing by an accelerator efficiency changes the units and applies at *any* stage, so treating it as a
step along the chain would make "electrical energy per transported muon" inexpressible. Consequently
every aggregate in this document -- every tier median, and the figure -- is computed **within a single
numeraire** (`beam_kinetic`); medianing beam-kinetic against electrical figures would be a units error
on top of the stage-basis error.

`basis_class` is the deprecated 1-D predecessor of `stage`, kept as an alias and validated against it
(`produced -> produced`, `collected -> transported`, `stopped_in_dt -> stopped_useful_in_dt`).
`charge_basis` records what is counted, and is read from the ledger here rather than described:
MuSIC's figure is `{H['charge_music']}` (mu+ and mu- together, so the mu--only cost is higher by a
factor this ledger does not source) and PSI HIMB is `{H['charge_psi_himb']}` -- irrelevant to muCF, which needs mu-, and
listed for scale only. `evidence_status` grades each number: `primary` / `primary_cited` /
`derived_here` are sourced; `author_declared_arbitrary` / `assumption` / `absent` are **not**, and any
figure composing one of those is reported as a **bound, never a value**.

Three factors are deliberately kept in their own flagged columns rather than folded into any row's
value: the **accelerator efficiency** `eta_acc` (Kelly's {H['kelly_eta_acc']}) -- applying it produces a
*separate row in a different numeraire*, never a silent edit to the beam-kinetic one, so both readings
stay side by side and auditable; the **recapture/breeding credit** `recapture_factor` (Kelly's x{H['kelly_recapture']},
recorded with `recapture_credit_applied=false`); and the **delivery factor** `eta_mu` (Kelly's {H['kelly_eta_mu']}),
whose authors call it an "arbitrary but reasonable assumption" and state they do not know its value --
so it carries `eta_mu_evidence_status = author_declared_arbitrary` and is **never composed into any
figure in this document**. T3 facility rows report no cost of this kind themselves, so their GeV/muon is
an ORIGINAL DERIVATION with the arithmetic shown in the row's `derivation` field (verbatim in the CSV).

## {TIER_TITLES['T1-design-study']}
{t1}

## {TIER_TITLES['T2-demonstrated-tech']}
{t2}

## {TIER_TITLES['T3-operating-facility']}
Each GeV/muon below is *implied, derived here* from public beam-power / muon-rate numbers (the full
arithmetic is in the CSV `derivation` column); no facility reports this quantity. PSI HIMB is mu+-ONLY
(surface muons) and thus irrelevant to muCF, which needs mu- -- listed for scale only.

{t3}

## The tier spread: an order-of-magnitude, mixed-basis observation
![muon-cost tier spread](figures/muon_cost_gap.png)

**Figure `figures/muon_cost_gap.png` (log-scale beam GeV/muon by tier).** Caption: *{DISARMAMENT}*

> **This section previously headlined "the 10^3 simulation-to-facility gap".** That heading asserted a
> same-basis ratio which the text below it already denied, and the phrasing had propagated into other
> documents as though it were a result. It is retracted here: the tier spread is an order-of-magnitude
> **mixed-basis** observation, not a measured gap.

> **AMENDMENT (2026-08-19) -- the T3 tier median and the tier ratio both moved, and the correction
> makes this document's own spread SMALLER.** The median took a tier, a numeraire and a pinned status,
> and applied no charge-basis filter -- so the PSI HIMB figure, which counts mu+ only and which this
> ledger's schema bars from any muCF cost aggregate, entered the published T3 median. It is now
> excluded where an aggregate is formed, and only there: the row is still rendered in the T3 table
> with its own label. **T3 tier median: 5497.5 -> {H['t3_median']} GeV.** Before, the median ran over
> four values -- 2286, 4993, 6002 and 890000 GeV -- and was the mean of the two middle ones
> (4993 + 6002 = 10995 GeV, and 10995 / 2 = 5497.5 GeV). After, it runs over three --
> {H['t3_median_rows']} -- and is the middle one. **Tier ratio: 1133.5x -> {H['gap_ratio']}x**
> (5497.5 / 4.85 = 1133.5 before, {H['t3_median']} / {H['t1_median']} = {H['gap_ratio']} after; the T1
> median is unchanged). The correction was taken BECAUSE it weakens the spread this document reports,
> not despite it. Nothing was tuned: the discrepant PSI HIMB figure is disclosed in the T3 table and in
> the sentence above, never deleted and never argued down.

The ~{H['gap_ratio']}x tier-median spread ({H['t1_median']} GeV design-study -> {H['t3_median']} GeV
facility) is **mixed-basis** (see the Headline: shared basis classes between T1 and T3 =
{H['shared_classes']}), so it is quoted here as an order of magnitude and never as a same-basis ratio.
Its basis composition is printed rather than summarised: T1 = {{{H['t1_classes']}}},
T3 = {{{H['t3_classes']}}}, all rows in the `beam_kinetic` numeraire.
It is also not a claim that the design-study floor is unreachable: existing facilities are built for beam
brightness and purity, not muons-per-watt; the floor is **unvalidated, not impossible**. This is a
normalization no facility publishes, presented as a reader-checkable compilation, not a verdict on any
program. (E_mu single accounting home: the rate-ledger `E_mu_cost` row points here.)

## What a muon is allowed to cost: the Kou-Chen cycle-closure ceiling
Kou & Chen (arXiv:2607.10989) close the muCF cycle the way Lawson closes a thermonuclear one. Their
eq.(15) gives the **maximum tolerable muon cost**. Their printed form carries the cycle-strength factor
`L_mu / (1 + omega_eff * L_mu)`; that factor *is* `N_fus,mu` by their own eq.(2), and it is substituted
here to read `E_cost,max = (eta_sys * E_use / G_mu) * N_fus,mu` -- an exact rewriting on their algebra,
not a form the paper prints.
At their own accounting -- `eta_sys` = {H['eta_sys']}, `G_mu` = {H['g_mu_target']} (breakeven),
`N_fus,mu` = {H['n_fus']} (their Table I LAMPF/Jones row, which that table types a *literature
anchor*; its one *experiment* row, SIN/Crowe, is `Y_f` = 124 +/- 10):

| `E_use` per fusion cycle | source | ceiling `E_cost,max` |
|---|---|---|
| {H['e_use_kouchen']} MeV | Kou & Chen's own choice -- **unsourced in their paper** | **{H['ceiling_kc']} GeV** |
| {H['e_use_kelly']} MeV | Kelly, Hart & Rose sec.2 -- primary-derived | **{H['ceiling_kelly']} GeV** |

- **{H['e_use_kouchen']} MeV:** {CRITERION_PROVENANCE['E_use_kouchen']}
- **{H['e_use_kelly']} MeV:** {CRITERION_PROVENANCE['E_use_kelly']}
- **`eta_sys` = {H['eta_sys']}:** {CRITERION_PROVENANCE['eta_sys']}

Both conventions are reported because **`N_L` is linear in `1/E_use`**, so the choice moves the answer by
the ratio of the two: our axis fixes the *cost* input of `N_L = E_cost / (eta_sys * E_use)` and leaves the
other two convention-set.

Against that ceiling, the chain points built from the open-access anchor -- every one of them a
**bound**, because {H['chain_points_stage_clause']}:

| chain point | numeraire | figure | vs {H['ceiling_kc']} GeV ceiling | vs {H['ceiling_kelly']} GeV ceiling |
|---|---|---|---|---|
| beam kinetic per mu- produced | `beam_kinetic` | {H['chain_beam']} | {H['ratio_beam_kc']}x | {H['ratio_beam_kelly']}x |
| electrical per mu- produced (minimal-subsystem) | `electrical_minimal` | {H['chain_elecmin']} | {H['ratio_elecmin_kc']}x | {H['ratio_elecmin_kelly']}x |
| electrical per mu- produced (site-wide) | `electrical_site` | {H['chain_elecsite']} | {H['ratio_elecsite_kc']}x | {H['ratio_elecsite_kelly']}x |

Read in the criterion's own coordinates, at `E_use` = {H['e_use_kouchen']} MeV. **`G_mu` carries two
distinct meanings in eq.(12) and they must not be substituted into each other:** the row below reports
the gain a chain point *achieves* at `N_fus` = {H['n_fus']}, whereas the boundary
`omega_crit = 1 / (G_mu * N_L)` is evaluated at the *target* gain, here breakeven
`G_mu` = {H['g_mu_target']}, so the `omega_crit` row is `1 / N_L`. Putting the achieved gain into the
boundary formula answers a different question and yields a different number:

| quantity | their {H['conventional_cost']} GeV convention | at {H['kelly_wallplug']} GeV | at {H['kelly_wallplug_site']} GeV |
|---|---|---|---|
| cycle demand `N_L` | {H['nl_conventional']} | {H['nl_elecmin_kc']} | {H['nl_elecsite_kc']} |
| one-muon gain `G_mu` at `N_fus` = {H['n_fus']} | {H['gmu_conventional']} | **{H['gmu_elecmin_kc']}** | **{H['gmu_elecsite_kc']}** |
| `omega_crit` | {H['omegacrit_conventional']}% | **{H['omegacrit_elecmin_kc']}%** | **{H['omegacrit_elecsite_kc']}%** |
| demonstrated sticking {H['omega_anchor']}% sits | {H['overshoot_conventional']}x over | **{H['overshoot_elecmin_kc']}x over** | **{H['overshoot_elecsite_kc']}x over** |

(The demonstrated sticking is this repo's own ledger row `{STICKING_RATE_SYMBOL}` = {H['omega_anchor']}%,
the SIN campaign value that is also Kou & Chen's Table I SIN/Crowe anchor -- read from `rates.csv`, not
retyped.)

**The claim, stated exactly.** This is **not** a finding that muCF fails, and it is **not** a correction
to Kou & Chen's mathematics, every digit of which reproduces. Their own sec.IV places the historical
anchors below breakeven, saying they "fall near the `G_mu` ~ 0.5-0.6 region"; their Table I `G_mu` column
prints 0.51 (SIN/Crowe) and 0.61 (LAMPF/Jones), and 0.41 (Petitjean low) is the nearest value it prints
outside that range. The contribution here is narrower and sharper:
**the cost convention that puts them there is itself ~{H['optimism_low']}-{H['optimism_high']}x
optimistic relative to the only anchor in the field whose beam-to-electrical conversion is
sourced**, so that anchor's `G_mu` is
~{H['gmu_elecsite_kc']}-{H['gmu_elecmin_kc']} rather than ~{H['gmu_conventional_1dp']},
and the demonstrated sticking sits {H['overshoot_elecmin_kc']}-{H['overshoot_elecsite_kc']}x on the
forbidden side of the no-go line rather than {H['overshoot_conventional']}x. **Every omitted factor pushes
the same way**, so this reading is rigorously one-sided: the chain points above are lower bounds, and
the true costs can only be higher.

## Chain coverage: which conversions are actually sourced
{_coverage_table(table)}

**{H['n_fully_sourced_chains']} of the {H['n_chain_rows']} pinned rows that sit on the muCF chain have a
fully-sourced chain to a useful stopped muon** (a further {H['n_offchain_rows']} rows are not on the chain
at all -- they stop muons outside D-T fuel). Exactly **one source** in this compilation states its own
beam-to-electrical conversion -- Kelly, Hart & Rose, who take 18% from the PSI primary they cite -- and
that same source states the delivery factor is unknown. {H['n_numeraire_sourced']} rows carry an
`eta_acc`: his beam row and its two electrical re-expressions, on the two denominators that same PSI
primary supplies. Kelly adopts the minimally-required-subsystem one; the site-wide figure is the
primary's, not his. So **no row in this
compilation supports a *value* for the quantity a muCF energy balance needs, only a bound.** That is the
honest state of the literature, not a gap in this ledger, and it is why
`ChainValue.render_value()` raises rather than prints for every row here.

**What would close this properly.** Each row would have to reach `stopped_useful_in_dt` in a stated
numeraire with every conversion sourced: beam GeV per mu- produced -> / eta_acc (a numeraire change)
-> / capture -> / transport -> / moderation -> / stopping-in-D-T -> mu--only. Only the numeraire change
is sourceable for any row today (Kelly: {H['norm_kelly_hart_rose_2021']} / {H['kelly_eta_acc']} =
{H['kelly_wallplug']} GeV). Kelly, Hart & Rose do quote a single collapsed delivery factor,
`eta_mu` = {H['kelly_eta_mu']}, but describe it verbatim as an "arbitrary but reasonable assumption" and state they do
not know its value; it is therefore recorded in the ledger as `author_declared_arbitrary`, **never folded
into any figure above and never allowed to headline**. The remaining factors are all <= 1, so every
figure above that sits **on the muCF chain** and counts mu- is a **lower bound** on the cost at the
terminal stage. The {H['n_offchain_rows']} off-chain
figures are not bounds on a muCF cost at all: they price stopping a muon somewhere that is not D-T
fuel, so no chain of sub-unity factors connects them to the quantity this table is about.
{H['mu_plus_only_clause']}

## The conversions themselves: the edge table
A cost row above is a **point** on the grid. A conversion between two points is an **edge**, and the
edges live in their own table (`openmucf/data/muon_cost_chain.csv`) rather than in extra columns on
the ledger. That is not tidiness: one source supplies several conversions, and one conversion takes
COMPETING values from different sources, so a per-source column set would force exactly one path per
source and make the second reading of a conversion inexpressible. An edge moves exactly one axis and
writes `*` on the other, which reads *any* -- a delivery fraction is dimensionless and holds in any
numeraire, and an accelerator efficiency applies at any stage, which is the same fact that makes
wall-plug a numeraire rather than a chain node. Where no source read here states a conversion, the row
carries `absent` and **no number**: {H['n_absent_edges']} of the {H['n_chain_edges']} edges
are that kind, and filling one of those cells with a plausible factor is the failure this table
exists to prevent.

{_edge_table(chain)}

### Which conversions are actually sourced
{_edge_coverage_table(edge_coverage_rows(table, chain))}

{H['chain_coverage_sentence']}

### Where the competing readings lead
{H['competing_clause']}

Composing every path out of the open-access anchor row ({LABELS[CHAIN_ANCHOR_ID]}) that uses **only
sourced conversions** gives {H['n_sourced_paths']} figures, and the deliverable is the SET of them
with their provenance -- never a mean, and never one of them singled out:

{_sourced_path_table(sourced_paths(table, chain))}

**Read the marker, and read what is missing.** {H['marker_clause']}
{H['blocked_clause']} The edge table carries every one of those conversions and the API
will compose them, marking a result *direction unknown* wherever its edges cannot bound it -- which
is how this compilation records a factor it may not publish a number from.
"""


def build_figure(table: mucost.MuonCostTable, path: str = "figures/muon_cost_gap.png") -> None:
    """Render the log-scale beam-GeV/muon-by-tier spread figure. NEVER byte-diffed (matplotlib bytes).

    Restricted to the ``beam_kinetic`` numeraire: the ledger now also holds electrical-numeraire rows,
    and plotting those on the same axis would put two different kinds of energy on one scale.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("figures").mkdir(exist_ok=True)
    colors = {"T1-design-study": "#33aa66", "T2-demonstrated-tech": "#cc9966", "T3-operating-facility": "#6699cc"}
    xpos = {"T1-design-study": 0, "T2-demonstrated-tech": 1, "T3-operating-facility": 2}
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for r in table:
        if not r.has_normalized or r.numeraire != mucost.BEAM_KINETIC:
            continue
        # tiny deterministic jitter: crc32 is stable across runs/platforms, unlike the built-in hash()
        # (PYTHONHASHSEED-randomized per process), so the figure regenerates identically run-to-run.
        x = xpos[r.tier] + (zlib.crc32(r.source_id.encode()) % 21 - 10) * 0.012
        ax.scatter([x], [r.normalized_GeV_per_mu], s=70, color=colors[r.tier], edgecolor="k", zorder=3)
        ax.annotate(LABELS[r.source_id].split(" (")[0], (x, r.normalized_GeV_per_mu),
                    fontsize=7, ha="left", va="bottom", xytext=(4, 2), textcoords="offset points")
    for tier, x in xpos.items():
        ax.hlines(table.tier_median(tier), x - 0.28, x + 0.28, color="k", lw=2, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(["T1 design\nstudies", "T2 demonstrated\ntech", "T3 operating\nfacilities"])
    ax.set_ylabel("beam kinetic energy per muon (GeV, log scale)")
    ax.set_title("Muon-cost tier spread (mixed basis): design studies vs operating facilities")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.text(0.5, 0.01, DISARMAMENT, ha="center", fontsize=8, style="italic", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_manifest_entries(H: dict[str, str], table: mucost.MuonCostTable) -> list:
    import re

    def _entry(entry_id, pattern):
        return provenance.ManifestEntry(
            id=entry_id, value=H[entry_id], pattern=pattern,
            source_type="derivation", source="scripts/generate_mucost.py", doc="MUON_COST.md",
        )

    entries = [
        _entry("t1_median", rf"from \*\*{re.escape(H['t1_median'])}\s*\n?GeV\*\* \(design studies\)"),
        _entry("t2_median", rf"through \*\*{re.escape(H['t2_median'])} GeV\*\* \(demonstrated"),
        _entry("t3_median", rf"to \*\*{re.escape(H['t3_median'])}\s*\n?GeV\*\* \(operating"),
        _entry("gap_ratio", rf"nominally \*\*{re.escape(H['gap_ratio'])}x\*\*"),
        _entry("disarmament", rf"Caption: \*{re.escape(H['disarmament'])}\*"),
        # the basis disclosure is a shipped claim and is pinned like any other number
        _entry("shared_classes", rf"shared between the two tiers: \*\*{re.escape(H['shared_classes'])}\*\*"),
        _entry("kelly_wallplug", rf"\*\*{re.escape(H['kelly_wallplug'])} GeV per muon produced\*\*"),
        _entry("kelly_wallplug_site", rf"it costs \*\*{re.escape(H['kelly_wallplug_site'])} GeV\*\*"),
        # eta_mu is published but never composed, which is exactly why it needs a pin: an unpinned
        # published number can drift from the CSV with every existing guard still green.
        _entry("kelly_eta_mu", rf"`eta_mu` = {re.escape(H['kelly_eta_mu'])}, but describe it verbatim"),
        _entry("kelly_recapture", rf"`recapture_factor` \(Kelly's x{re.escape(H['kelly_recapture'])},"),
        # the Kou-Chen ceiling comparison: every cell is a shipped number and is pinned like any other
        _entry("ceiling_kc", rf"\| {re.escape(H['e_use_kouchen'])} MeV \|[^\n]*\| \*\*{re.escape(H['ceiling_kc'])} GeV\*\* \|"),
        _entry("ceiling_kelly", rf"\| {re.escape(H['e_use_kelly'])} MeV \|[^\n]*\| \*\*{re.escape(H['ceiling_kelly'])} GeV\*\* \|"),
        _entry("omega_anchor", rf"= {re.escape(H['omega_anchor'])}%,\s*\n?the SIN campaign value"),
        _entry("nl_conventional", rf"cycle demand `N_L` \| {re.escape(H['nl_conventional'])} \|"),
        _entry("gmu_conventional", rf"\| {re.escape(H['gmu_conventional'])} \| \*\*{re.escape(H['gmu_elecmin_kc'])}\*\*"),
        _entry("omegacrit_conventional", rf"`omega_crit` \| {re.escape(H['omegacrit_conventional'])}% \|"),
        _entry("overshoot_conventional", rf"\| {re.escape(H['overshoot_conventional'])}x over \|"),
        _entry("n_fully_sourced_chains", rf"\*\*{re.escape(H['n_fully_sourced_chains'])} of the {re.escape(H['n_chain_rows'])} pinned rows"),
        # the edge layer: the coverage counts and every terminal figure the competing edges produce
        _entry("n_chain_edges", rf"of the {re.escape(H['n_chain_edges'])} edges\s*\n?are that kind"),
        _entry("n_absent_edges", rf"\*\*no number\*\*: {re.escape(H['n_absent_edges'])} of the"),
        _entry("n_sourced_conversions", rf"\*\*{re.escape(H['n_sourced_conversions'])}\*\* carry a factor from a primary"),
        _entry("n_sourced_paths", rf"gives {re.escape(H['n_sourced_paths'])} figures"),
    ]
    for i in range(1, int(H["n_sourced_paths"]) + 1):
        entries.append(_entry(f"sourced_path_{i}", rf"\| {re.escape(H[f'sourced_path_{i}'])} \|"))
    # the three sourced chain points, their bound-marked figures, ratios and criterion coordinates
    for short in _SHORT.values():
        entries.append(_entry(f"chain_{short}", rf"\| {re.escape(H[f'chain_{short}'])} \|"))
        for key in ("kc", "kelly"):
            entries.append(
                _entry(f"ratio_{short}_{key}", rf"\| {re.escape(H[f'ratio_{short}_{key}'])}x \|")
            )
    for short in ("elecmin", "elecsite"):
        entries.append(_entry(f"nl_{short}_kc", rf"\| {re.escape(H[f'nl_{short}_kc'])} \|"))
        entries.append(_entry(f"gmu_{short}_kc", rf"\*\*{re.escape(H[f'gmu_{short}_kc'])}\*\*"))
        entries.append(_entry(f"omegacrit_{short}_kc", rf"\*\*{re.escape(H[f'omegacrit_{short}_kc'])}%\*\*"))
        entries.append(_entry(f"overshoot_{short}_kc", rf"\*\*{re.escape(H[f'overshoot_{short}_kc'])}x over\*\*"))
    # every pinned row's normalized value, anchored to its table-row label
    for r in table:
        if not r.has_normalized:
            continue
        eid = f"norm_{r.source_id}"
        label = re.escape(LABELS[r.source_id])
        entries.append(_entry(eid, rf"{label}[^\n]*\| {re.escape(H[eid])} \|"))
    return entries


def main() -> None:
    table = mucost.load_muon_cost()
    H = build_headline(table)
    Path("MUON_COST.md").write_text(build_markdown(table, H), encoding="utf-8")
    build_figure(table)
    entries = build_manifest_entries(H, table)
    inputs = {
        "muon_cost_csv_sha256": provenance.file_sha256(MUON_COST_CSV),
        # the edge table is a generator INPUT like the node table: its hash lands in the manifest, so
        # `provenance --check` binds the rendered conversions to the bytes they were rendered from.
        "muon_cost_chain_csv_sha256": provenance.file_sha256(MUON_COST_CHAIN_CSV),
    }
    provenance.write_manifest(
        "MUON_COST_MANIFEST.json", entries, inputs, generated_by="scripts/generate_mucost.py"
    )
    print(f"wrote MUON_COST.md + figures/muon_cost_gap.png + MUON_COST_MANIFEST.json ({len(entries)} entries)")
    print(f"tier medians GeV/muon: T1={H['t1_median']} T2={H['t2_median']} T3={H['t3_median']} (gap {H['gap_ratio']}x)")
    print(f"anchor: Kelly {H['norm_kelly_hart_rose_2021']} | Bertin {H['norm_bertin_1987']} | Eliezer {H['norm_eliezer_henis_1994']}")


if __name__ == "__main__":
    main()
