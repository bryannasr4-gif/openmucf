"""Generate MUON_COST.md + figures/muon_cost_gap.png + MUON_COST_MANIFEST.json (deterministic).

    python scripts/generate_mucost.py

Content (WAVE2_EXECUTION_SPEC sec.1, WS-E): the open muon-cost ledger rendered as tier tables + a
normalization-basis explainer + the 10^3 simulation-to-facility gap figure. This is a **curated
compilation with provenance, not an evaluation** (I8): the single auditable basis is beam energy per
muon in GeV; wall-plug = that / eta_acc (kept separate); T3 facility rows are original derivations
("implied, derived here, formula shown"); an accounting credit (Kelly's x2.5 recapture) is recorded in
its own flagged column, never folded into the normalized value.

Paywall/headline rule (I3): the headline sentence cites Kelly (4.70, open access) FIRST as the named
anchor, then the full-text-verified Bertin and Eliezer-Henis values as corroboration (by DOI). Rows that
are needs_verification (Jandel) or slide-tier (Acceleron) appear only in the tables with visible flags
and NEVER headline.

Audit wiring (WS-E): this generator regenerates all three artifacts; only MUON_COST.md +
MUON_COST_MANIFEST.json join the `git diff --exit-code` list -- the PNG is never byte-diffed
(matplotlib/freetype bytes are not cross-platform stable). All committed numbers are pure deterministic
arithmetic on the committed CSV (no MCMC/solver), so the two byte-diffed artifacts are cross-arch stable.

Computation lives in importable helpers (no side effects on import); file I/O + the figure + printing are
guarded behind ``main()`` so tests import and assert on the tables without regenerating the doc/figure.
"""

from __future__ import annotations

import zlib
from pathlib import Path

from openmucf import mucost, provenance
from openmucf.mucost import MUON_COST_CSV

# The disarmament sentence -- goes VERBATIM in the figure caption (WS-E spec). Em-dash intentional.
DISARMAMENT = (
    "Facilities optimize brightness/purity, not muons-per-watt "
    "— the floor is unvalidated, not impossible."
)

TIER_TITLES = {
    "T1-design-study": "Tier 1 -- purpose-built muon-source design studies",
    "T2-demonstrated-tech": "Tier 2 -- demonstrated technology",
    "T3-operating-facility": "Tier 3 -- operating facilities (GeV/stopped-mu derived here)",
}

# Short display labels used in the tables AND as manifest row anchors (unique per row).
LABELS = {
    "kelly_hart_rose_2021": "Kelly, Hart & Rose (2021)",
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


def build_headline(table: mucost.MuonCostTable) -> dict[str, str]:
    """Single source of truth: the formatted strings shared by MUON_COST.md and the manifest."""
    H: dict[str, str] = {}
    # per-row normalized display (pinned rows only; Jandel has no value)
    for r in table:
        if r.has_normalized:
            H[f"norm_{r.source_id}"] = _fmt(r.normalized_GeV_per_mu)
    # tier medians + the gap
    m1 = table.tier_median("T1-design-study")
    m2 = table.tier_median("T2-demonstrated-tech")
    m3 = table.tier_median("T3-operating-facility")
    H["t1_median"] = _fmt_med(m1)
    H["t2_median"] = _fmt_med(m2)
    H["t3_median"] = _fmt_med(m3)
    H["gap_ratio"] = f"{m3 / m1:.1f}"
    H["disarmament"] = DISARMAMENT
    # Basis composition -- computed, so the disclosure can never drift from the CSV. No basis_class is
    # shared between T1 and T3, so a same-basis T1-vs-T3 ratio is NOT COMPUTABLE from these rows.
    H["t1_classes"] = ", ".join(sorted(table.basis_classes("T1-design-study")))
    H["t3_classes"] = ", ".join(sorted(table.basis_classes("T3-operating-facility")))
    shared = table.basis_classes("T1-design-study") & table.basis_classes("T3-operating-facility")
    H["shared_classes"] = ", ".join(sorted(shared)) if shared else "none"
    # The one fully-sourced wall-plug-equivalent figure (Kelly states its own eta_acc).
    kelly = table["kelly_hart_rose_2021"]
    H["kelly_wallplug"] = f"{kelly.wallplug_lower_bound_GeV:.1f}"
    H["kelly_eta_acc"] = f"{kelly.eta_acc_assumption:g}"
    return H


def _tier_table(table: mucost.MuonCostTable, tier: str, H: dict[str, str]) -> str:
    head = (
        "| source | value as published | GeV/muon (normalized) | basis_class | charge | nv | "
        "basis / notes |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in table.tier(tier):
        norm = H[f"norm_{r.source_id}"] if r.has_normalized else "-- (not pinned)"
        nv = "**yes**" if r.needs_verification else "no"
        bc = r.basis_class or "--"
        ch = r.charge_basis or "--"
        # flag the rows whose figure understates the cost per mu- stopped in D-T fuel
        if r.understates_stopped_in_dt_cost:
            bc += " (lower bound)"
        if r.charge_basis == "mu_plus_only":
            ch += " (not muCF)"
        # keep table cells single-line: collapse the basis to its first clause
        basis = r.basis_as_published.split(";")[0].strip()
        rows.append(
            f"| {LABELS[r.source_id]} | {r.value_as_published} | {norm} | {bc} | {ch} | {nv} | {basis} |"
        )
    return head + "\n".join(rows)


def build_markdown(table: mucost.MuonCostTable, H: dict[str, str]) -> str:
    t1 = _tier_table(table, "T1-design-study", H)
    t2 = _tier_table(table, "T2-demonstrated-tech", H)
    t3 = _tier_table(table, "T3-operating-facility", H)
    return f"""# MUON_COST.md -- the open muon-cost ledger (auto-generated by `scripts/generate_mucost.py`)

> **Curated compilation with provenance, NOT an evaluation.** `normalized_GeV_per_mu` is BEAM energy per
> muon in GeV **on each row's own accounting basis** (`basis_class`) -- these rows are **NOT all on a
> common basis and are not commensurable across classes**. Wall-plug = normalized / eta_acc (kept in its
> own column, never folded). T3 facility rows are ORIGINAL DERIVATIONS ("implied, derived here, formula
> shown") -- no operating facility reports GeV-per-stopped-muon. An accounting credit (Kelly's x2.5
> recapture) is recorded in its own flagged column, never folded into the normalized value.

## Headline
The purpose-built muon-source **design studies** put the muon cost at a few GeV per muon. The open-access
anchor is **{LABELS['kelly_hart_rose_2021']}: {H['norm_kelly_hart_rose_2021']} GeV/muon** (G4Beamline;
deuteron on a tungsten target; DOI 10.1088/2515-7655/abfb4b) -- the only fully reader-checkable row,
which reports its own eta_acc={H['kelly_eta_acc']} (PSI-measured) and Q_elec=14% at X_mu=150. Two further
full-text-verified design studies corroborate the same single-GeV scale: **{LABELS['bertin_1987']},
~{H['norm_bertin_1987']} GeV/muon** at liquid density (DOI 10.1209/0295-5075/4/8/003; ~3 GeV ideal
all-collected) and **{LABELS['eliezer_henis_1994']}, ~{H['norm_eliezer_henis_1994']} GeV/muon**
(DOI 10.13182/FST94-A30300).

**Those single-GeV figures are beam energy per muon PRODUCED, not the loaded cost of a muon stopped in
D-T fuel.** On Kelly's own numbers the wall-plug-equivalent is {H['norm_kelly_hart_rose_2021']} /
{H['kelly_eta_acc']} = **{H['kelly_wallplug']} GeV per muon produced** -- and that is still a LOWER BOUND
on the wall-plug cost per mu- actually stopped in D-T, because the collection and stopping fractions
(both < 1) have not been applied. Every factor omitted here pushes the cost UP, so the bound is one-sided.

**Operating muon facilities are ~10^3 worse.** The tier-median rises from **{H['t1_median']}
GeV** (design studies) through **{H['t2_median']} GeV** (demonstrated technology, collected-not-stopped)
to **{H['t3_median']} GeV** (operating facilities) -- nominally **{H['gap_ratio']}x**. **That ratio is
NOT a same-basis comparison and must not be quoted as one.** T1 contains {{{H['t1_classes']}}} rows and T3
contains {{{H['t3_classes']}}} rows; basis classes shared between the two tiers: **{H['shared_classes']}**.
With no shared class, a same-basis T1-vs-T3 ratio is **not computable from these rows** -- and because
both the numerator and the denominator contain lower-bound (per-produced / per-collected) figures, the
ratio is not cleanly bounded in either direction. The defensible statement is the **order of magnitude**
(~10^3), driven by technology, with the basis composition disclosed above. needs_verification (Jandel)
and slide-tier (Acceleron) rows carry visible flags below and never headline.

## Normalization basis (read before the tables)
Each row's `normalized_GeV_per_mu` is **beam (kinetic) energy per muon on that row's own basis**, recorded
machine-readably in `basis_class`:

- `produced` -- per muon created (Kelly, Eliezer-Henis, Acceleron)
- `stopped_in_dt` -- per muon stopped in the D-T target: the quantity a muCF energy balance actually
  needs (Bertin)
- `collected` -- transported/collected but never stopped in fuel (muon-collider front end, MuSIC, PSI HIMB)
- `stopped_other_target` -- stopped in a NON-D-T target, e.g. mu2e's aluminium stopping target (mu2e, COMET)

A `produced` or `collected` figure is a **lower bound** on the `stopped_in_dt` cost, since collection and
stopping fractions are both < 1. `charge_basis` records what is counted: MuSIC's figure is `mixed`
(mu+ and mu- together, so the mu--only cost is roughly 2x higher) and PSI HIMB is `mu_plus_only` --
irrelevant to muCF, which needs mu-, and listed for scale only.

Two factors are deliberately kept SEPARATE and never folded into the normalized value: **wall-plug
efficiency** (`eta_acc`, the electrical -> muon-beam efficiency, e.g. Kelly's PSI-measured
{H['kelly_eta_acc']}; wall-plug per muon = normalized / eta_acc -- this is an energy-accounting
conversion, **not** a collection or stopping correction) and any **recapture/breeding credit**
(`recapture_factor`: Kelly's x2.5 is recorded but `recapture_credit_applied=false`). T3 facility rows
report no such cost themselves, so their GeV/muon is an ORIGINAL DERIVATION with the arithmetic shown in
the row's `derivation` field (verbatim in the CSV).

## {TIER_TITLES['T1-design-study']}
{t1}

## {TIER_TITLES['T2-demonstrated-tech']}
{t2}

## {TIER_TITLES['T3-operating-facility']}
Each GeV/muon below is *implied, derived here* from public beam-power / muon-rate numbers (the full
arithmetic is in the CSV `derivation` column); no facility reports this quantity. PSI HIMB is mu+-ONLY
(surface muons) and thus irrelevant to muCF, which needs mu- -- listed for scale only.

{t3}

## The 10^3 simulation-to-facility gap
![muon-cost gap by tier](figures/muon_cost_gap.png)

**Figure `figures/muon_cost_gap.png` (log-scale GeV/muon by tier).** Caption: *{DISARMAMENT}*

The ~{H['gap_ratio']}x tier-median spread ({H['t1_median']} GeV design-study -> {H['t3_median']} GeV
facility) is **mixed-basis** (see the Headline: shared basis classes between T1 and T3 =
{H['shared_classes']}), so it is quoted here as an order of magnitude and never as a same-basis ratio.
It is also not a claim that the design-study floor is unreachable: existing facilities are built for beam
brightness and purity, not muons-per-watt; the floor is **unvalidated, not impossible**. This is a
normalization no facility publishes, presented as a reader-checkable compilation, not a verdict on any
program. (E_mu single accounting home: the rate-ledger `E_mu_cost` row points here.)

**What would close this properly.** A same-basis comparison needs every row expressed as wall-plug energy
per mu- *stopped in D-T fuel*: beam GeV per mu- produced -> / eta_acc (wall-plug) -> / collection
fraction -> / stopping fraction in D-T -> mu--only. Only the first two steps are sourceable for any row
today (Kelly: {H['norm_kelly_hart_rose_2021']} / {H['kelly_eta_acc']} = {H['kelly_wallplug']} GeV). The
remaining factors are all < 1, so every published figure in this table is a **lower bound** on that
quantity -- which is why the one-sided reading above is the only defensible one.
"""


def build_figure(table: mucost.MuonCostTable, path: str = "figures/muon_cost_gap.png") -> None:
    """Render the log-scale GeV/muon-by-tier gap figure. NEVER byte-diffed (matplotlib bytes)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("figures").mkdir(exist_ok=True)
    colors = {"T1-design-study": "#33aa66", "T2-demonstrated-tech": "#cc9966", "T3-operating-facility": "#6699cc"}
    xpos = {"T1-design-study": 0, "T2-demonstrated-tech": 1, "T3-operating-facility": 2}
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for r in table:
        if not r.has_normalized:
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
    ax.set_ylabel("beam energy per muon (GeV, log scale)")
    ax.set_title("The 10$^3$ muon-cost gap: design studies vs operating facilities")
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
    ]
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
    inputs = {"muon_cost_csv_sha256": provenance.file_sha256(MUON_COST_CSV)}
    provenance.write_manifest(
        "MUON_COST_MANIFEST.json", entries, inputs, generated_by="scripts/generate_mucost.py"
    )
    print(f"wrote MUON_COST.md + figures/muon_cost_gap.png + MUON_COST_MANIFEST.json ({len(entries)} entries)")
    print(f"tier medians GeV/muon: T1={H['t1_median']} T2={H['t2_median']} T3={H['t3_median']} (gap {H['gap_ratio']}x)")
    print(f"anchor: Kelly {H['norm_kelly_hart_rose_2021']} | Bertin {H['norm_bertin_1987']} | Eliezer {H['norm_eliezer_henis_1994']}")


if __name__ == "__main__":
    main()
