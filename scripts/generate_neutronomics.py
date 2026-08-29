"""Generate NEUTRONOMICS.md + NEUTRONOMICS_MANIFEST.json (deterministic, pure arithmetic).

    python scripts/generate_neutronomics.py

Content (neutronomics, Layer 1): the neutrons-per-joule "league table" that places muon-catalyzed
fusion, as a 14 MeV neutron source, against the established incumbents. This is a **curated
compilation with provenance, not an evaluation** and asserts **no new physics**: every number
is either a measured yield (X_mu), a tier median already published in MUON_COST.md, or a transparent
"derived here" ratio of published beam parameters.

muCF appears as THREE tier-separated rows -- one per MUON_COST.md muon-cost tier -- NEVER one blended
row. The muon cost is not uncertain-around-a-mean; it is SELECTED by which muon source you build (a
design study vs a demonstrated collider front end vs an operating facility). Averaging the tiers would
be UQ theater over a decision variable, so each tier gets its own row.

BASIS (read before trusting a number): every value is neutrons per joule of PRIMARY BEAM kinetic energy
(beam basis, NOT wall-plug). For muCF the primary-beam joule is the MUON_COST.md tier median, at
whatever accounting stage that tier's rows carry (``normalized_GeV_per_mu``); for the alternative
sources it is the deuteron or proton beam kinetic energy delivered to the target. Wall-plug = this /
eta_acc (< 1), kept SEPARATE (the single-accounting-home rule; applying eta_acc is a numeraire change
that produces its own row, never folded into a beam-kinetic value). A wall-plug muCF column would rest
on an unsourced per-tier efficiency and is deliberately NOT tabulated.

Grounding -- every number sourced, one channel one accounting home: X_mu = 113 is imported from
``openmucf.calibrate.OBS['xmu_obs']`` -- the MEASURED
Petitjean/Breunlich yield (rate-ledger validation target ``V_petitjean_Xmu``, band
[100,150]) -- NOT the forward-UQ posterior median (104). The three E_mu tier medians come from
``openmucf.mucost.load_muon_cost().tier_median(...)`` (the muon-cost ledger's single accounting home),
never a hardcoded literal. GeV -> J via 1 GeV = 1.602176634e-10 J (CODATA elementary charge).

Audit wiring: this generator regenerates BOTH artifacts and BOTH join the ``git diff --exit-code`` list
and the ``provenance --check`` line -- all committed numbers are pure deterministic arithmetic on the
committed muon_cost.csv + the ledger X_mu + the published beam parameters of the alternative sources
(no MCMC/solver), so both are byte-stable cross-arch (A2).

Computation lives in importable helpers (no file I/O or printing on import); file I/O + printing are
guarded behind ``main()`` so tests import and assert on the tables without regenerating the doc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from openmucf import mucost, provenance
from openmucf.calibrate import OBS  # X_mu = OBS['xmu_obs'] = 113.0 (Petitjean/Breunlich measured value)
from openmucf.mucost import MUON_COST_CSV

# CODATA 2018 elementary charge -> energy conversions (exact, byte-stable).
GEV_TO_J = 1.602176634e-10
EV_TO_J = 1.602176634e-19

# X_mu: the MEASURED Petitjean/Breunlich yield, imported from the ledger (calibrate.OBS),
# NOT the forward-UQ posterior median 104. Kept as the single ground so the doc cannot drift from it.
XMU = OBS["xmu_obs"]

# muCF muon-cost tiers (MUON_COST.md). Short label -> (mucost tier id, human tier label).
TIERS = (
    ("T1", "T1-design-study", "T1 design study"),
    ("T2", "T2-demonstrated-tech", "T2 demonstrated tech"),
    ("T3", "T3-operating-facility", "T3 operating facility"),
)


@dataclass(frozen=True)
class AltSource:
    """One alternative 14 MeV (or, for spallation, broad-spectrum) neutron source.

    ``n_per_joule`` is neutrons per joule of PRIMARY BEAM kinetic energy, derived here from the
    published (yield, beam-current, beam-energy) or (neutrons-per-proton, proton-energy) triple. Every
    row carries a ``source`` + ``locator``; a row that cannot be sourced is DROPPED, never
    approximated (see ``DROPPED`` below).
    """

    key: str
    label: str
    neutron_energy: str
    published: str  # the verbatim published inputs (the triple)
    derivation: str  # the "derived here" arithmetic string
    n_per_joule: float
    source: str
    locator: str


def _iv(label, key, neutron_energy, yield_per_s, current_A, voltage_V, published, source, locator):
    """Beam-target / sealed-tube row: n/J = yield_per_s / (I * V) [beam power in W]."""
    power_W = current_A * voltage_V
    npj = yield_per_s / power_W
    deriv = f"{published.split(';')[0]}: n/J = yield / (I x V) = {yield_per_s:g} / ({power_W:g} W)"
    return AltSource(key, label, neutron_energy, published, deriv, npj, source, locator)


def _per_proton(label, key, neutron_energy, n_per_proton, proton_E_eV, published, source, locator):
    """Spallation row: n/J = neutrons_per_proton / (E_proton in J)."""
    npj = n_per_proton / (proton_E_eV * EV_TO_J)
    deriv = f"n/J = n per proton / E_proton = {n_per_proton:g} / ({proton_E_eV:g} eV)"
    return AltSource(key, label, neutron_energy, published, deriv, npj, source, locator)


def build_alt_sources() -> list[AltSource]:
    """The 3-4 alternative 14 MeV/n sources, each with a live-verified primary source (values entered
    from the cited locators, never approximated). Every row below carries a full triple; a source that
    cannot supply one is dropped and named, never approximated.
    """
    return [
        _iv(
            "Sealed-tube D-T generator (Thermo Sci. P385)", "dt_generator", "14.1 MeV (monoenergetic)",
            yield_per_s=8.2e8, current_A=90e-6, voltage_V=140e3,
            published="8.2e8 n/s at 90 uA, 140 kV (A3083 tube, measured)",
            source="Nowak et al., arXiv:2406.18607 (Thermo Scientific P385, A3083 sealed tube)",
            locator="arXiv:2406.18607 (2024): ~8.2e8 n/s at 90 uA / 140 kV; manufacturer spec 5e8 n/s",
        ),
        _iv(
            "FNG (Frascati Neutron Generator, ENEA)", "fng", "14 MeV (T(d,n)alpha)",
            yield_per_s=1e11, current_A=1e-3, voltage_V=300e3,
            published="1e11 n/s at 1 mA, 300 keV deuterons on a tritiated target",
            source="Pillon et al., J. Phys. Conf. Ser. 1021 (2018) 012004 (FNG, ENEA Frascati)",
            locator="J.Phys.Conf.Ser. 1021 012004: 'D+ accelerated up to 300 keV and 1 mA'; 1e11 n/s at 14 MeV",
        ),
        _iv(
            "RTNS-II (LLNL, beam-target 14 MeV source)", "rtns2", "14 MeV (T(d,n)alpha)",
            yield_per_s=2.1e11, current_A=1e-3, voltage_V=400e3,
            published="2.1e11 n/s per mA of 400 keV D+ (fresh Ti-tritide target; up to 150 mA -> 3e13 n/s)",
            source="RTNS-II status reports (LLNL): J. Nucl. Mater. 108-109 (1982) 29; UCRL-84554",
            locator="400 kV, 150 mA D+ accelerator; initial yield 2.1e11 n/s-mA (RTNS-II operational summary)",
        ),
        _per_proton(
            "Spallation source (ISIS, W target)", "spallation", "broad evaporation spectrum (NOT 14 MeV)",
            n_per_proton=20.0, proton_E_eV=800e6,
            published="~20 neutrons per 800 MeV proton on a tungsten target (broad spectrum, ~1-2 MeV peak)",
            source="ISIS Neutron and Muon Source (STFC), 'How ISIS works'",
            locator="isis.stfc.ac.uk: 800 MeV proton accelerator; ~20 neutrons/proton on the W target",
        ),
    ]


# Sources considered but DROPPED for unsourceability. Empty here: every included row carries a
# full triple; no row required approximation. (Kept as an explicit, possibly-empty list so
# the discipline is visible in the doc regardless of outcome.)
DROPPED: list[str] = []


def _fmt_npj(v: float) -> str:
    """Deterministic display of a neutrons-per-joule value (scientific, byte-stable cross-arch)."""
    return f"{v:.3e}"


def _fmt_gev(v: float) -> str:
    """Deterministic display of a muon-cost tier median in GeV (keeps 5497.5, drops trailing on 178)."""
    return f"{v:g}"


def _fmt_j(v: float) -> str:
    return f"{v:.3e}"


#: Spelled orders of magnitude for the section-1 span sentence. Deliberately narrow: a tier spread
#: outside this range is a different document and must be re-worded, not silently re-numbered.
_OOM_WORDS = {2: "two", 3: "three", 4: "four"}


def mucf_rows(table: mucost.MuonCostTable) -> list[dict]:
    """The three tier-separated muCF rows: n/J = X_mu / (E_mu_tier in J), E_mu_tier = MUON_COST median."""
    rows = []
    for short, tier_id, label in TIERS:
        emu_GeV = table.tier_median(tier_id)
        emu_J = emu_GeV * GEV_TO_J
        npj = XMU / emu_J
        rows.append(
            {
                "short": short,
                "tier_id": tier_id,
                "label": label,
                "emu_GeV": emu_GeV,
                "emu_J": emu_J,
                "n_per_joule": npj,
            }
        )
    return rows


def build_headline(table: mucost.MuonCostTable) -> dict[str, str]:
    """Single source of truth: the formatted strings shared by NEUTRONOMICS.md and the manifest."""
    H: dict[str, str] = {"xmu": f"{XMU:g}"}
    for r in mucf_rows(table):
        s = r["short"]
        H[f"emu_{s}"] = _fmt_gev(r["emu_GeV"])
        H[f"emuJ_{s}"] = _fmt_j(r["emu_J"])
        H[f"npj_{s}"] = _fmt_npj(r["n_per_joule"])
    # The T1-to-T3 neutron-column ratio, COMPUTED. n/J = X_mu / E_mu, so this is exactly the muon-cost
    # tier-median ratio and nothing else -- an inheritance, not a second measurement. It is published
    # here so the sentence below states the factor it actually carries rather than an approximation of
    # it, and so it moves when the ledger moves.
    _by_short = {r["short"]: r for r in mucf_rows(table)}
    _exact_ratio = _by_short["T1"]["n_per_joule"] / _by_short["T3"]["n_per_joule"]
    H["npj_ratio_T1_T3"] = f"{_exact_ratio:.1f}"
    # The reader's two recomputation routes from the printed page, published beside the factor (the
    # same rule the FINDINGS ratio carries: a paragraph written to be checked may not quote a number
    # its own printed inputs cannot reproduce). The E_mu column must reproduce the published figure
    # EXACTLY -- the generator refuses to write the page otherwise -- and the n/J column's quotient
    # is printed too, because its cells are rounded independently to four significant figures, so a
    # reader dividing them can land one last-digit step away and must not have to guess why.
    H["emu_ratio_from_printed"] = f"{float(H['emu_T3']) / float(H['emu_T1']):.1f}"
    assert H["emu_ratio_from_printed"] == H["npj_ratio_T1_T3"], (
        f"the printed E_mu cells give {H['emu_ratio_from_printed']}, which does not reproduce the "
        f"published {H['npj_ratio_T1_T3']} -- a reader checking the sentence gets a different number"
    )
    H["npj_ratio_from_printed"] = f"{float(H['npj_T1']) / float(H['npj_T3']):.1f}"
    # each .3e cell carries a relative rounding error of at most 5e-4, so their quotient sits within
    # ~1e-3 of the exact factor; outside that bound the disclosure sentence itself is false.
    assert abs(float(H["npj_ratio_from_printed"]) / _exact_ratio - 1.0) < 1.1e-3, (
        f"the printed n/J cells no longer reproduce the factor to display rounding "
        f"({H['npj_ratio_from_printed']} vs {_exact_ratio}) -- re-word the disclosure sentence"
    )
    # The order-of-magnitude word in section 1 is derived from the same factor, never typed. Strict
    # map: a spread drifting outside it must be re-worded consciously, not printed as a bare digit.
    H["tier_span_words"] = _OOM_WORDS[round(math.log10(_exact_ratio))]
    for a in build_alt_sources():
        H[f"npj_alt_{a.key}"] = _fmt_npj(a.n_per_joule)
    return H


def _mucf_table(rows: list[dict], H: dict[str, str]) -> str:
    head = (
        "| tier (muon source) | E_mu per muon (GeV) | E_mu per muon (J) | n per beam joule |\n"
        "|---|---|---|---|\n"
    )
    out = []
    for r in rows:
        s = r["short"]
        out.append(
            f"| {r['label']} (MUON_COST.md {r['tier_id']} median) "
            f"| {H[f'emu_{s}']} | {H[f'emuJ_{s}']} | {H[f'npj_{s}']} |"
        )
    return head + "\n".join(out)


def _alt_table(alts: list[AltSource], H: dict[str, str]) -> str:
    head = (
        "| source | neutron energy | published inputs | n per beam joule (derived here) | source |\n"
        "|---|---|---|---|---|\n"
    )
    out = []
    for a in alts:
        out.append(
            f"| {a.label} | {a.neutron_energy} | {a.published} "
            f"| {H[f'npj_alt_{a.key}']} | {a.locator} |"
        )
    return head + "\n".join(out)


def _dropped_block() -> str:
    if not DROPPED:
        return (
            "**Dropped for unsourceability: none.** Every alternative-source row above carries a "
            "primary (yield, beam-current, beam-energy) or (neutrons-per-proton, "
            "proton-energy) triple; no row required approximation."
        )
    lines = "\n".join(f"- {d}" for d in DROPPED)
    return "**Dropped for unsourceability (named, not silently omitted):**\n" + lines


def _stage_mix_clause(table: mucost.MuonCostTable) -> str:
    """The within-tier composition disclosure: a tier median can itself mix accounting stages.

    Rendered from the same rows each median is computed over (``aggregate_rows``), never typed, and
    over EVERY tier rather than one example -- naming a single mixed tier invites the reader to
    conclude the others are stage-homogeneous, which is a claim about the ledger this docstring
    has no business making on its behalf. Conditional: a
    tier whose aggregate holds one stage is left out of the sentence, and if no tier mixes, the
    homogeneous form is emitted instead. This is the disclosure that keeps the one-row-at-a-time
    reading honest -- each row is conditional on its tier median AS COMPOSED.
    """
    mixed = []
    for short, tier_id, _label in TIERS:
        stages = sorted({r.stage for r in table.aggregate_rows(tier=tier_id)})
        if len(stages) > 1:
            joined = " and ".join(f"`{s}`" for s in stages)
            mixed.append(f"the {short} median over rows at {joined}")
    if not mixed:
        return (
            "(Every tier median here is taken over rows at a single accounting stage, so each row "
            "conditions on a stage-homogeneous median.)"
        )
    listed = ";\n".join(mixed)
    return (
        f"(One more inherited property: a tier median can itself mix accounting stages within its\n"
        f"tier -- {listed} --\n"
        f"so each row conditions on its tier median as composed, with the per-row stages printed\n"
        f"in MUON_COST.md.)"
    )


def build_markdown(table: mucost.MuonCostTable, H: dict[str, str]) -> str:
    rows = mucf_rows(table)
    alts = build_alt_sources()
    return f"""# NEUTRONOMICS.md -- the neutrons-per-joule league table (auto-generated by `scripts/generate_neutronomics.py`)

> **Curated compilation with provenance, NOT an evaluation; no new physics.** muCF is shown as
> THREE tier-separated rows -- one per MUON_COST.md muon-cost tier -- NEVER one blended row: the muon
> cost is SELECTED by which muon source you build, not uncertain around a mean, so averaging the tiers
> would be UQ theater over a decision variable. Framed around neutron-source economics, not energy
> breakeven.

## Basis: neutrons per joule of PRIMARY BEAM energy (beam basis, not wall-plug)
Every value below is `neutrons / (joule of primary accelerator beam kinetic energy)`. For muCF the
primary-beam joule is the MUON_COST.md tier median, at whatever accounting stage that tier's rows carry
(MUON_COST.md `normalized_GeV_per_mu`; the per-row stages are printed there, and a tier median can mix
them); for the alternative sources it is the deuteron or proton beam kinetic
energy delivered to the target. This is a **beam basis**: the wall-plug figure is this divided by the
accelerator efficiency eta_acc (< 1), kept SEPARATE (the single-accounting-home rule; in MUON_COST.md
applying eta_acc is a numeraire change that produces its own row and is never folded into a
beam-kinetic value). A wall-plug muCF column would rest on an unsourced per-tier
efficiency for T2/T3 and is deliberately NOT tabulated; on a wall-plug basis every muCF value below
falls by 1/eta_acc (e.g. ~5.6x at eta_acc = 0.18).

> **Basis note (transparency; a genuine spec-vs-basis reconciliation).** The muCF formula is
> n/J = X_mu / (E_mu_tier in J) with E_mu_tier taken DIRECTLY from the MUON_COST.md tier median, which is
> a BEAM energy per muon. The table is therefore `n per BEAM joule`, labelled as such -- not `n per
> wall-plug joule`. Reaching a wall-plug basis would require dividing by eta_acc, which is kept in its own
> column upstream and pinned to a primary source only for the Kelly row; it is reported separately here,
> never folded into a tier value: one channel has one accounting home, and an unsourceable row is
> named rather than approximated.

## 1. muCF as a 14 MeV neutron source (three tier-separated rows)
**X_mu = {H['xmu']} fusions per muon** -- the MEASURED Petitjean/Breunlich yield
(`openmucf.calibrate.OBS['xmu_obs']`; rate-ledger validation target `V_petitjean_Xmu`, Breunlich et al.,
Annu. Rev. Nucl. Part. Sci. 39 (1989) 311, band [100,150]). This is the measured value, **NOT** the
forward-UQ posterior median (104): the league table is a tier-SELECTED accounting of the measured yield,
not a UQ pushforward. Each E_mu tier median is sourced from the muon-cost ledger (MUON_COST.md /
`openmucf.mucost`), converted at 1 GeV = 1.602176634e-10 J.

{_mucf_table(rows, H)}

The tier medians span about {H['tier_span_words']} orders of magnitude, and `n/J = X_mu / E_mu` is exactly inversely
proportional to the muon cost, so the neutron column spans the SAME factor **by construction**:
{H['npj_ratio_T1_T3']}x, which is the muon-cost tier-median ratio itself and not a second measurement
of anything. Checked from the page itself: the E_mu column reproduces the factor exactly
({H['emu_T3']} / {H['emu_T1']} = {H['emu_ratio_from_printed']}x), while dividing the printed n/J
column gives {H['npj_ratio_from_printed']}x -- those cells carry four significant figures and are
rounded independently, so their quotient can differ from the factor in the last digit; the E_mu
column is the one that carries the factor's own inputs.
It runs from **{H['npj_T1']} n/J** at the design-study muon cost (E_mu {H['emu_T1']} GeV,
~{rows[0]['emu_GeV'] * 1000 / XMU:.0f} MeV of beam per neutron) down to **{H['npj_T3']} n/J** at the
operating-facility muon cost (E_mu {H['emu_T3']} GeV).

**That spread is inherited, and it inherits the basis it came from.** MUON_COST.md records that its
tier medians are a MIXED-BASIS, order-of-magnitude observation -- no accounting stage is shared
between T1 and T3 at all, so a same-basis cost ratio between them is not computable from those rows.
Dividing one row of this table by another therefore does not produce a same-basis neutron-economy
comparison either; it reproduces the cost spread together with its heterogeneity. Each row remains a
well-defined statement conditional on ITS OWN tier median, which is why the table is tier-separated
and never blended. {_stage_mix_clause(table)}
Which row is real depends entirely on whether a purpose-built muon source at the
design-study cost is ever demonstrated -- the same open question MUON_COST.md flags ("the floor is
unvalidated, not impossible").

## 2. Alternative 14 MeV/n sources (sourced comparison)
Established neutron sources, each `n per beam joule` derived here from published beam parameters (the
arithmetic is in each row's inputs). Spallation is included per the neutronomics landscape but is a
**broad evaporation spectrum, NOT monoenergetic 14 MeV** -- flagged in its row.

{_alt_table(alts, H)}

{_dropped_block()}

## 3. What the table says (honest reading)
On a beam-energy basis, muCF at the **design-study** muon cost ({H['npj_T1']} n/J, ~{rows[0]['emu_GeV'] * 1000 / XMU:.0f}
MeV of beam per neutron) is competitive with a spallation source and ~10^3 better than a sealed-tube D-T
generator -- because one expensive muon catalyzes ~{H['xmu']} fusions. At the **operating-facility** muon
cost ({H['npj_T3']} n/J) that advantage is gone: because `n/J` is inversely proportional to `E_mu`,
the muon-cost tier spread carries into the neutron column exactly ({H['npj_ratio_T1_T3']}x), and it
carries its accounting with it -- that spread is MIXED-BASIS and an order of magnitude, never a
same-basis ratio (MUON_COST.md), so neither is the ratio between two rows of the muCF table. The
comparison against the alternative sources in section 2 is a different one and IS same-basis: every
`n per beam joule` in this document, muCF row and alternative alike, is counted on beam energy and
on nothing else (no wall-plug for any row; the
alternatives' accelerator efficiencies are likewise not folded), the spallation spectrum is not 14 MeV,
and none of this is an energy-breakeven claim -- it is neutron-source accounting, and no new physics
is introduced. E_mu single accounting home: MUON_COST.md; X_mu single ground: the measured `V_petitjean_Xmu`.
"""


def build_manifest_entries(H: dict[str, str], table: mucost.MuonCostTable, alts: list[AltSource]) -> list:
    import re

    def _entry(entry_id, pattern, source_type="derivation", source="scripts/generate_neutronomics.py"):
        return provenance.ManifestEntry(
            id=entry_id, value=H[entry_id], pattern=pattern,
            source_type=source_type, source=source, doc="NEUTRONOMICS.md",
        )

    entries = [
        _entry(
            "xmu", rf"\*\*X_mu = {re.escape(H['xmu'])} fusions per muon\*\*",
            source_type="ledger_row",
            source="openmucf/data/validation_targets.csv:V_petitjean_Xmu / openmucf.calibrate.OBS['xmu_obs']",
        ),
    ]
    # muCF rows: E_mu median (ledger-derived) + n/J (derivation), anchored to the tier's table row.
    for short, tier_id, _label in TIERS:
        emu, npj = H[f"emu_{short}"], H[f"npj_{short}"]
        row_re = rf"MUON_COST\.md {re.escape(tier_id)} median\) \| {re.escape(emu)} \|"
        entries.append(
            _entry(
                f"emu_{short}", row_re,
                source_type="ledger_row", source="openmucf/data/muon_cost.csv (mucost.tier_median)",
            )
        )
        entries.append(_entry(f"npj_{short}", rf"{re.escape(emu)} \| {re.escape(H[f'emuJ_{short}'])} \| {re.escape(npj)} \|"))
    # The published tier ratio. Tracked because it is the number the retracted "transfers one-for-one"
    # sentence used to assert without printing: it is now printed, and bound to the ledger by the
    # manifest as well as by the test that recomputes it from the muon-cost medians.
    entries.append(
        _entry(
            "npj_ratio_T1_T3",
            rf"\*\*by construction\*\*:\s*\n?\s*{re.escape(H['npj_ratio_T1_T3'])}x",
            source_type="ledger_row", source="openmucf/data/muon_cost.csv (mucost.tier_median)",
        )
    )
    # The two reader-recomputation figures published beside the factor: the E_mu-column quotient
    # (must equal the published factor -- the generator asserts it) and the n/J-column quotient
    # (display-rounded, disclosed so a reader's division is explained rather than surprising).
    entries.append(
        _entry("emu_ratio_from_printed", rf"= {re.escape(H['emu_ratio_from_printed'])}x\)")
    )
    entries.append(
        _entry("npj_ratio_from_printed", rf"gives {re.escape(H['npj_ratio_from_printed'])}x")
    )
    # alt-source rows: each derived n/J, anchored to its label
    for a in alts:
        eid = f"npj_alt_{a.key}"
        entries.append(_entry(eid, rf"{re.escape(a.label)} \|[^\n]*\| {re.escape(H[eid])} \|"))
    return entries


def main() -> None:
    table = mucost.load_muon_cost()
    H = build_headline(table)
    alts = build_alt_sources()
    Path("NEUTRONOMICS.md").write_text(build_markdown(table, H), encoding="utf-8")
    entries = build_manifest_entries(H, table, alts)
    inputs = {
        "muon_cost_csv_sha256": provenance.file_sha256(MUON_COST_CSV),
        "xmu_obs": f"{XMU:g}",
        "gev_to_j": f"{GEV_TO_J:.9e}",
    }
    provenance.write_manifest(
        "NEUTRONOMICS_MANIFEST.json", entries, inputs, generated_by="scripts/generate_neutronomics.py"
    )
    print(f"wrote NEUTRONOMICS.md + NEUTRONOMICS_MANIFEST.json ({len(entries)} entries)")
    rows = mucf_rows(table)
    print(f"muCF n/J (beam basis, X_mu={H['xmu']}): "
          + " ".join(f"{r['short']}={H['npj_' + r['short']]}" for r in rows))
    print("alt n/J: " + " ".join(f"{a.key}={H['npj_alt_' + a.key]}" for a in alts))
    if DROPPED:
        print(f"dropped (unsourceable): {DROPPED}")
    else:
        print("dropped (unsourceable): none")


if __name__ == "__main__":
    main()
