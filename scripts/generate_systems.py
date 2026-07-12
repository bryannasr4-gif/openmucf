"""Generate SYSTEMS.md + SYSTEMS_MANIFEST.json (deterministic; closed-form, no MCMC/solver).

    python scripts/generate_systems.py

Content (WAVE2_EXECUTION_SPEC sec.2, WS-S): the full differentiable energy-balance graph as a "Q
Rosetta stone" -- one table putting the several muCF Q conventions on ONE comparable reference basis
(the efficiency-free scientific gain) -- plus the eta_acc self-correction finding, the G-Kelly
cross-basis validation, and the Acceleron worked example.

Framing (I1/I8): no new physics -- this is a transparent ACCOUNTING instrument over the frozen v1
energy chain, presented as a curated compilation, not an evaluation. Every shipped number is CLOSED-FORM
algebra over ``openmucf.systems`` (a superset of the untouched ``energy.EnergyChain``); the two byte-diffed
artifacts (SYSTEMS.md + SYSTEMS_MANIFEST.json) are therefore cross-architecture stable.

Computation lives in importable helpers (no side effects on import); file I/O + printing are guarded
behind ``main()`` so tests import and assert on the tables without regenerating the doc.
"""

from __future__ import annotations

from pathlib import Path

from openmucf import provenance
from openmucf.energy import EnergyChain
from openmucf.rates import RATES_CSV
from openmucf.systems import KELLY, SystemChain, rosetta_table

# The Rosetta X_mu grid: the measured record (Petitjean ~113), the Jones/Kelly record (150), the
# scientific breakeven (~284), and a frontier target (500). All lie within Kelly's fig-3 fit range
# [100, 600], so the kelly_Q_elec column is interpolation, never extrapolation.
X_MU_GRID = [113.0, 150.0, 284.0, 500.0]

# Pre-registered G-Kelly band (WS-S sec.2.2): reproduce Kelly's 14% within +/-10% relative.
KELLY_BAND_LO, KELLY_BAND_HI = 12.6, 15.4

# Acceleron ARPA-E slide inputs (muon_cost.csv acceleron_2025 row; slide-tier, NOT a gate).
ACCELERON_E_MU_GEV = 3.0
ACCELERON_NET_EXPORT_GEV = 3.4


def build_headline() -> dict[str, str]:
    """Single source of truth: every formatted string shared by SYSTEMS.md and the manifest."""
    H: dict[str, str] = {}
    sc, ec = SystemChain(), EnergyChain()

    # G-legacy: the degenerate special case reproduces v1's breakevens exactly.
    H["breakeven_sci"] = f"{sc.breakeven_xmu_sci():.2f}"          # 284.09
    H["breakeven_net"] = f"{sc.breakeven_xmu_net():.2f}"          # 2367.42
    H["breakeven_sci_v1"] = f"{ec.breakeven_xmu_sci():.2f}"
    H["breakeven_net_v1"] = f"{ec.breakeven_xmu_net():.2f}"

    # eta_acc self-correction: v1 default 0.30 -> Kelly PSI-measured 0.18.
    be_018 = SystemChain(eta_acc=0.18).breakeven_xmu_net()
    H["eta_acc_v1"] = "0.30"
    H["eta_acc_kelly"] = "0.18"
    H["breakeven_net_018_precise"] = f"{be_018:.2f}"             # 3945.71 (computed live)
    H["breakeven_net_018"] = f"{be_018:.0f}"                     # 3946 (rounded; used in the finding)
    H["breakeven_net_v1_round"] = f"{sc.breakeven_xmu_net():.0f}"  # 2367

    # G-Kelly cross-basis validation (Eq. 2 + Table 1; every number paper-cited).
    qe = KELLY.q_elec()
    H["kelly_q_elec_pct"] = f"{100 * qe:.2f}"                    # 15.69
    H["kelly_q_sci"] = f"{KELLY.q_sci():.4f}"                    # 0.5617
    H["kelly_F"] = f"{KELLY.F_fusion_MeV_ref:.0f}"              # 2991
    H["kelly_H"] = f"{KELLY.H_recover_MeV:.0f}"                 # 3743
    H["kelly_B"] = f"{KELLY.B_beam_MeV:.0f}"                    # 3606
    H["kelly_band_lo"] = f"{KELLY_BAND_LO:.1f}"                  # 12.6
    H["kelly_band_hi"] = f"{KELLY_BAND_HI:.1f}"                  # 15.4
    H["kelly_E_mu"] = f"{KELLY.E_mu_GeV:.2f}"                   # 4.70

    # Acceleron worked example (slide-tier).
    acc = SystemChain(E_mu_GeV=ACCELERON_E_MU_GEV)
    H["acc_e_mu"] = f"{ACCELERON_E_MU_GEV:.1f}"
    H["acc_net_export"] = f"{ACCELERON_NET_EXPORT_GEV:.1f}"
    H["acc_q_sci_150"] = f"{150.0 * acc.E_per_fusion_MeV / acc.E_mu_MeV:.4f}"
    H["acc_q_net_150"] = f"{150.0 * acc.E_per_fusion_MeV * acc._net_efficiency_factor() / acc.E_mu_MeV:.4f}"
    H["acc_breakeven_net"] = f"{acc.breakeven_xmu_net():.0f}"

    # Rosetta cells (anchored so the doc table cannot silently drift from the code).
    for r in rosetta_table(X_MU_GRID):
        x = f"{r['x_mu']:.0f}"
        H[f"ros_{x}_q_sci"] = f"{r['q_sci_v1']:.4f}"
        H[f"ros_{x}_q_net"] = f"{r['q_net_v1']:.5f}"
        H[f"ros_{x}_kelly"] = f"{100 * r['kelly_Q_elec']:.2f}"
        H[f"ros_{x}_ykc"] = f"{r['ykc_efficiency_free']:.4f}"
    return H


def _rosetta_table_md(H: dict[str, str]) -> str:
    head = (
        "| X_mu (fusions/muon) | q_sci_v1 | q_net_v1 | ykc_efficiency_free | kelly_Q_elec | "
        "reference (q_sci) |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in rosetta_table(X_MU_GRID):
        x = f"{r['x_mu']:.0f}"
        rows.append(
            f"| {x} | {H[f'ros_{x}_q_sci']} | {H[f'ros_{x}_q_net']} | {H[f'ros_{x}_ykc']} | "
            f"{H[f'ros_{x}_kelly']}% | {H[f'ros_{x}_q_sci']} |"
        )
    return head + "\n".join(rows)


def build_markdown(H: dict[str, str]) -> str:
    table = _rosetta_table_md(H)
    return f"""# SYSTEMS.md -- the energy-balance graph + the Q Rosetta stone \
(auto-generated by `scripts/generate_systems.py`)

> **A transparent accounting instrument, NOT new physics (I1) and NOT an evaluation (I8).**
> `openmucf.systems.SystemChain` is a strict SUPERSET of the frozen v1 `energy.EnergyChain`: it exposes
> every node of the wall-plug -> muon -> fusion(+breeding) -> blanket -> thermal -> electric ->
> recirculation chain as a named, differentiable knob, and adds two explicit, FLAGGED, default-OFF
> factors (a tritium-breeding energy credit and a recirculating-power fraction). At the defaults the
> graph reproduces v1 exactly (the G-legacy anchor). Every number below is closed-form algebra.

## The Q Rosetta stone (the deliverable)
The muCF literature reports "Q" in several incompatible conventions. The same physics that reads as a
scientific gain **q_sci_v1** = X_mu*E_f/E_mu also reads as a much smaller net-electrical gain
**q_net_v1** (after the efficiency chain), as Kelly-Hart-Rose's electrical gain **kelly_Q_elec**, and as
a Yin-Kou-Chen-style **ykc_efficiency_free** gain. This table places all four on ONE comparable
reference basis -- the efficiency-free scientific gain q_sci. Each SystemChain-native column
(q_sci_v1, q_net_v1, ykc_efficiency_free) collapses to the reference under its documented
`convert_to_reference`; the collapse IS the point.

{table}

The three SystemChain-native columns are evaluated on the v1-default chain (E_mu = 5.0 GeV); each maps
back to the reference q_sci by exact algebra (identity for q_sci_v1 and ykc_efficiency_free at the
default breeding credit; dividing out `blanket_M*eta_thermal*eta_acc*(1-recirc_fraction)` for q_net_v1).
**kelly_Q_elec** is Kelly-Hart-Rose's published external convention on THEIR chain (E_mu = {H['kelly_E_mu']}
GeV), affine in X_mu over their fitted range [100, 600] fusions/muon; it converts to their own scientific
gain in the G-Kelly section below. The lesson: a single dimensionless "Q" is meaningless without its
basis -- the Rosetta table makes the basis explicit.

## The eta_acc self-correction finding (framed self-first)
*Our v1 default eta_acc = 0.30 was optimistic: Kelly-Hart-Rose's PSI-measured value is 0.18. Corrected,
the net-electrical breakeven moves ~{H['breakeven_net_v1_round']} -> ~{H['breakeven_net_018']}
(= 5000/(17.6x0.40x0.18); linear in eta_acc). We correct our own default's implication first; this is
the single sharpest demonstration of why the transparent chain matters.*

Precisely (closed form, `SystemChain(eta_acc=0.18).breakeven_xmu_net()`): the net-electrical breakeven
is E_mu/(E_f*eta_thermal*eta_acc) = 5000/(17.6x0.40x0.18) = **{H['breakeven_net_018_precise']}**, i.e.
**{H['breakeven_net_018']}** fusions/muon (vs the v1-default {H['breakeven_net_v1']} at eta_acc=0.30) --
a 5/3x rise, exactly linear in 1/eta_acc. The v1 code default STAYS 0.30 this wave (changing it would
re-anchor the FINDINGS; that is a later decision); the FINDING carries the correction.

## G-legacy: the degenerate special case (no-tuning anchor)
With the flagged knobs off (`breeding_credit_MeV=0`, `recirc_fraction=0`), SystemChain reproduces the
frozen v1 EnergyChain breakevens exactly: scientific **{H['breakeven_sci']}** (v1 {H['breakeven_sci_v1']})
and net-electrical **{H['breakeven_net']}** (v1 {H['breakeven_net_v1']}), to relative 1e-12. The superset
adds accounting transparency, never a numeric change to the v1 chain.

## G-Kelly: cross-basis validation against Kelly-Hart-Rose 2021
Kelly, Hart & Rose (J. Phys. Energy 3, 035003, 2021; open access, DOI 10.1088/2515-7655/abfb4b) define an
electrical gain via their Eq. (2): `Q_elec = ((F*eta_mu + H*eta_rec)/B)*eta_acc*eta_heat`. Reading their
exact Table-1 config (deuteron -> tungsten; 150 fusions/muon; their highest-Q config, Q=1.87): F (fusion
heat, row A) = **{H['kelly_F']}** MeV [17.6 MeV D-T kinetic + 8.4 MeV fusion-neutron tritium breeding =
26.0 MeV/fusion, folded per the paper]; H (recoverable heat, rows B-E) = **{H['kelly_H']}** MeV
(2664+23+526+530); B (beam energy, row G) = **{H['kelly_B']}** MeV; efficiencies eta_mu=0.50
("arbitrary but reasonable"), eta_rec=1.00 (100% capture), eta_acc={H['eta_acc_kelly']} (PSI 590 MeV
accelerator), eta_heat=0.60 (heat->electricity, ">60%"; eta_acc*eta_heat = 10.8%). Their beam energy per
muon is 3.61 GeV / 0.77 muons-per-beam-particle = {H['kelly_E_mu']} GeV.

Evaluating Eq. (2) on these cited numbers reproduces **Q_elec = {H['kelly_q_elec_pct']}%**.

**Finding (reported, not tuned -- I2).** {H['kelly_q_elec_pct']}% is **just above** the pre-registered
band **[{H['kelly_band_lo']}%, {H['kelly_band_hi']}%]** (Kelly's 14% headline +/-10% relative). The
residual is INTERNAL to Kelly, not a conversion error on our side: Table 1 is his single **highest-Q**
illustrative config (Q=1.87), whereas his 14% headline is read off his figure-3 Q_elec-vs-fusions/muon
curve at 150 -- a smoothed value from a relationship the paper itself calls "quite volatile". Our
transparent basis reproduces his own formula on his own tabulated numbers exactly; per I2 we report the
{H['kelly_q_elec_pct']}% -- {H['kelly_band_hi']}% gap as a finding about config selection rather than
adjusting any input to land on 14%. Converted to the reference (efficiency-free scientific) basis,
Kelly's config sits at q_sci = **{H['kelly_q_sci']}** -- commensurable with our own q_sci at X_mu=150.

## Acceleron worked example (slide-tier inputs; NOT a validation gate)
Acceleron Fusion's ARPA-E deck (July 2025) asserts **{H['acc_e_mu']} GeV/muon** production and a
**{H['acc_net_export']} GeV net-export** flow. Treated ONLY as a labeled worked example (slide-tier,
simulated-unvalidated inputs; never a gate -- it would be unevaluable-as-silent-failure): through the
transparent chain at E_mu = {H['acc_e_mu']} GeV, X_mu = 150, q_sci = **{H['acc_q_sci_150']}** and
q_net (v1 default efficiencies) = **{H['acc_q_net_150']}** -- still sub-unity. Reaching net-electrical
export (q_net > 1) at their {H['acc_e_mu']} GeV/muon needs X_mu >= **{H['acc_breakeven_net']}**
fusions/muon, ~{H['acc_breakeven_net']}/150 = far above today's record. Their claim is recorded, not
endorsed.
"""


def build_manifest_entries(H: dict[str, str]) -> list:
    import re

    def _entry(entry_id, pattern):
        return provenance.ManifestEntry(
            id=entry_id, value=H[entry_id], pattern=pattern,
            source_type="derivation", source="scripts/generate_systems.py", doc="SYSTEMS.md",
        )

    entries = [
        _entry("breakeven_sci", rf"scientific \*\*{re.escape(H['breakeven_sci'])}\*\*"),
        _entry("breakeven_net", rf"net-electrical \*\*{re.escape(H['breakeven_net'])}\*\*"),
        _entry("breakeven_net_018", rf"-> ~{re.escape(H['breakeven_net_018'])}\b"),
        _entry("breakeven_net_018_precise", rf"= \*\*{re.escape(H['breakeven_net_018_precise'])}\*\*"),
        _entry("eta_acc_kelly", rf"PSI-measured value is {re.escape(H['eta_acc_kelly'])}\b"),
        _entry("kelly_q_elec_pct", rf"reproduces \*\*Q_elec = {re.escape(H['kelly_q_elec_pct'])}%"),
        _entry("kelly_band_lo", rf"\[{re.escape(H['kelly_band_lo'])}%,"),
        _entry("kelly_band_hi", rf"{re.escape(H['kelly_band_hi'])}%\]"),
        _entry("kelly_F", rf"row A\) = \*\*{re.escape(H['kelly_F'])}\*\* MeV"),
        _entry("kelly_H", rf"rows B-E\) = \*\*{re.escape(H['kelly_H'])}\*\* MeV"),
        _entry("kelly_B", rf"row G\) = \*\*{re.escape(H['kelly_B'])}\*\* MeV"),
        _entry("kelly_q_sci", rf"q_sci = \*\*{re.escape(H['kelly_q_sci'])}\*\*"),
        _entry("acc_q_net_150", rf"q_net \(v1 default efficiencies\) = \*\*{re.escape(H['acc_q_net_150'])}\*\*"),
    ]
    # every Rosetta cell, anchored to its table row
    for r in rosetta_table(X_MU_GRID):
        x = f"{r['x_mu']:.0f}"
        entries.append(
            provenance.ManifestEntry(
                id=f"ros_{x}_q_net", value=H[f"ros_{x}_q_net"],
                pattern=rf"\| {x} \| {re.escape(H[f'ros_{x}_q_sci'])} \| {re.escape(H[f'ros_{x}_q_net'])} \|",
                source_type="derivation", source="scripts/generate_systems.py", doc="SYSTEMS.md",
            )
        )
    return entries


def main() -> None:
    H = build_headline()
    Path("SYSTEMS.md").write_text(build_markdown(H), encoding="utf-8")
    entries = build_manifest_entries(H)
    inputs = {
        "rates_csv_sha256": provenance.file_sha256(RATES_CSV),
        "E_mu_GeV_default": float(SystemChain().E_mu_GeV),
        "eta_acc_v1_default": 0.30,
        "eta_thermal_default": 0.40,
        "kelly_table1_F_H_B_MeV": [KELLY.F_fusion_MeV_ref, KELLY.H_recover_MeV, KELLY.B_beam_MeV],
        "kelly_efficiencies_mu_rec_acc_heat": [KELLY.eta_mu, KELLY.eta_rec, KELLY.eta_acc, KELLY.eta_heat],
    }
    provenance.write_manifest(
        "SYSTEMS_MANIFEST.json", entries, inputs, generated_by="scripts/generate_systems.py"
    )
    print(f"wrote SYSTEMS.md + SYSTEMS_MANIFEST.json ({len(entries)} entries)")
    print(f"G-legacy breakevens: sci={H['breakeven_sci']} net={H['breakeven_net']} (v1-exact)")
    print(f"eta_acc self-correction: {H['breakeven_net_v1_round']} -> {H['breakeven_net_018']} "
          f"(precise {H['breakeven_net_018_precise']}) at eta_acc {H['eta_acc_v1']}->{H['eta_acc_kelly']}")
    print(f"G-Kelly: Q_elec={H['kelly_q_elec_pct']}% band [{H['kelly_band_lo']},{H['kelly_band_hi']}]% "
          f"({'INSIDE' if KELLY_BAND_LO <= 100 * KELLY.q_elec() <= KELLY_BAND_HI else 'OUTSIDE -> finding'})")


if __name__ == "__main__":
    main()
