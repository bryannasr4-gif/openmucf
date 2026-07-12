"""Reproduce the Parisi-Rutkowski Ac-225 headline from their published factor chain.

Paper: J. F. Parisi & A. Rutkowski, "Isotope Production in Muon-Catalyzed-Fusion Systems",
arXiv:2511.20951v2 (Marathon Fusion, 25 Dec 2025). Fetched live and read in full for this script.

The headline (abstract + Section IV + Table I): a muCF neutron source with a 10 g ^226Ra feedstock and a
steady-state rate of 1e12 muons/s -- roughly half a kilowatt of D-T fusion power -- produces ~20 mg/yr of
^225Ac, "comparable to 400 times global supply in 2024". This script reproduces that number as a genuine
FORWARD computation from the paper's own published factors (no factor is back-solved to hit 20 mg/yr --
invariant I2). The transmutation pathway is

    ^226Ra(n,2n)^225Ra --beta^- (15 d)--> ^225Ac                                            [paper eq. (28)]

The forward chain (each factor cited to its locator in ``FACTORS`` below):

    Ndot_n   = R_mu * N_fus_mu                     fusion (= 14.1 MeV D-T neutron) rate   [eq. (3), (A9)]
    Ndot_225 = eta_pro * Ndot_n                    ^225Ra(->^225Ac) production rate       [eq. (39)]
    Mdot     = Ndot_225 * (A_Ac / N_A) * s_year    mass rate -> mg/yr                     [eq. (14)]

Two published cross-checks fall out of the same factors: the fusion power P_fus = Ndot_n * E_fus lands at
564 W (Table I; abstract "half a kilowatt"), and the yield over the 2024 global supply (51 ug/yr, ref. [33])
lands at ~400x (abstract).

I2 / honest-finding note. eta_pro = 0.0087 is taken as the paper's published Table-I value. It is the one
factor NOT re-derived here from first principles: the authors obtain it from the blanket geometry
(eqs. (33)-(38)) folded with an EXTERNAL ^226Ra(n,2n) microscopic cross section (an ENDF/B-class nuclear-data
input; the paper defines sigma_(n,2n) in eq. (36) but prints no numeric value) and validate it with an
OpenMC transport run. We reproduce the headline at the level the paper's printed factors support and flag
that external dependency explicitly (invariant I3: every number sourced). Given eta_pro, the analytic factor
chain (eqs. (3),(14),(39)) reproduces the OpenMC Table-I value 20,480 ug/yr to < 1% -- an internal
consistency check on the paper, computed forward.

Positioning (invariant I8/I9): this is a reproduction of an EXTERNAL group's result, not an OpenMuCF claim.
See the framing paragraph printed by ``main()``; the "viable well before energy breakeven" language is
Parisi & Rutkowski's, quoted as theirs.

Run from the repo root:

    python scripts/parisi_ac225.py
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

# --- paper headline targets (the numbers we reproduce; NOT inputs to the forward chain) ----------------
HEADLINE_MG_PER_YEAR = 20.0  # abstract: "20 mg of 225Ac per year"
TABLE_I_UG_PER_YEAR = 20_480.0  # Table I (high-flux RaO OpenMC run): 20,480 ug/yr
TABLE_I_PFUS_W = 564.0  # Table I / Sec IV: P_fus ~= 564 W ("roughly half a kilowatt")
GLOBAL_2024_UG_PER_YEAR = 51.0  # 2024 global 225Ac production ~3 Ci/yr ~= 51 ug/yr (ref. [33])
GLOBAL_2024_MULTIPLE = 400.0  # abstract: "comparable to 400 times global supply in 2024"

RECON_TOL_FRAC = 0.05  # +/-5% reproduction band on the headline (the I2 forward-reproduction tolerance)


@dataclass(frozen=True)
class Factor:
    """One published factor in the forward chain: value, unit, and its in-paper (or standard) locator."""

    symbol: str
    name: str
    value: float
    unit: str
    citation: str  # non-empty by construction; asserted in the tests (invariant I3)


# --- THE FACTOR CHAIN (every row carries a locator; arXiv:2511.20951v2 unless noted) -------------------
FACTORS: tuple[Factor, ...] = (
    Factor(
        "R_mu",
        "muon stopping/absorption rate at the design point",
        1.0e12,
        "muons/s",
        "arXiv:2511.20951v2, abstract + Sec IV ('10^12 muons/second beam rate'); R_mu = I_mu*f_stop, "
        "eq. (A2). At the design point beam rate == absorbed rate (f_stop=1), pinned by the P_fus=564 W "
        "consistency check (eq. (10)).",
    ),
    Factor(
        "N_fus_mu",
        "catalyzed D-T fusions per muon",
        200.0,
        "dimensionless",
        "arXiv:2511.20951v2, Sec IV ('Assuming N_fus,mu = 200'), Table I, Fig. 3 caption.",
    ),
    Factor(
        "eta_pro",
        "neutron (n,2n) transmutation fraction, 226Ra->225Ra (high-flux RaO)",
        0.0087,
        "dimensionless",
        "arXiv:2511.20951v2, Table I (eta_pro = 0.0087). Authors' value from blanket geometry "
        "eqs. (33)-(38) x an EXTERNAL 226Ra(n,2n) cross section (ENDF/B-class; sigma_(n,2n) defined "
        "eq. (36) but not printed) validated by OpenMC; taken as published, not re-derived here.",
    ),
    Factor(
        "A_Ac",
        "molar mass of 225Ac (== 225Ra parent; 1:1 via beta^- decay)",
        225.0,
        "g/mol",
        "Mass number of the A=225 chain (eq. (28) 226Ra(n,2n)225Ra -> 225Ac); m_pro = A_Ac/N_A, eq. (14).",
    ),
    Factor(
        "N_A",
        "Avogadro constant",
        6.02214076e23,
        "1/mol",
        "CODATA 2018 exact value (used in the paper's number-density relation, eq. (35)).",
    ),
    Factor(
        "s_per_year",
        "seconds per year (Julian year, 365.25 d)",
        3.15576e7,
        "s/yr",
        "Standard Julian year; consistent with the paper's per-second rates and T_hour=3600 s (eq. (13)).",
    ),
    Factor(
        "E_fus_MeV",
        "energy released per D-T fusion (3.5 MeV alpha + 14.1 MeV n)",
        17.6,
        "MeV",
        "arXiv:2511.20951v2, eq. (1) and eq. (2) ('E_fus = 17.6 MeV'). Used for the P_fus cross-check.",
    ),
    Factor(
        "MeV_to_J",
        "MeV -> joule conversion",
        1.602176634e-13,
        "J/MeV",
        "CODATA 2018 exact elementary charge (1 MeV = 1.602176634e-13 J). For the P_fus cross-check.",
    ),
)

# lookup for the forward chain: values keyed by symbol (the same Factor rows the tests assert citations on)
_V: dict[str, float] = {f.symbol: f.value for f in FACTORS}


def neutron_rate(r_mu: float = _V["R_mu"], n_fus_mu: float = _V["N_fus_mu"]) -> float:
    """Fusion (= 14.1 MeV D-T neutron) rate Ndot_n = R_mu * N_fus_mu, in neutrons/s [eq. (3), (A9)]."""
    return r_mu * n_fus_mu


def fusion_power_W(
    r_mu: float = _V["R_mu"],
    n_fus_mu: float = _V["N_fus_mu"],
    e_fus_mev: float = _V["E_fus_MeV"],
    mev_to_j: float = _V["MeV_to_J"],
) -> float:
    """Total fusion power P_fus = Ndot_n * E_fus, in watts [eq. (10)] -- cross-check against Table I 564 W."""
    return neutron_rate(r_mu, n_fus_mu) * e_fus_mev * mev_to_j


def ac225_atoms_per_s(
    r_mu: float = _V["R_mu"],
    n_fus_mu: float = _V["N_fus_mu"],
    eta_pro: float = _V["eta_pro"],
) -> float:
    """225Ac production rate Ndot_225 = eta_pro * Ndot_n, in atoms/s [eq. (39)]."""
    return eta_pro * neutron_rate(r_mu, n_fus_mu)


def atom_mass_g(a_molar: float = _V["A_Ac"], n_a: float = _V["N_A"]) -> float:
    """Mass of a single product atom m_pro = A / N_A, in grams [eq. (14)]."""
    return a_molar / n_a


def ac225_mg_per_year(
    r_mu: float = _V["R_mu"],
    n_fus_mu: float = _V["N_fus_mu"],
    eta_pro: float = _V["eta_pro"],
    a_molar: float = _V["A_Ac"],
    n_a: float = _V["N_A"],
    s_per_year: float = _V["s_per_year"],
) -> float:
    """The headline forward computation: 225Ac mass production rate in mg/yr.

    Mdot = Ndot_225 * m_pro * s_per_year, converted g -> mg [eqs. (3), (14), (39)]. Every default is a
    cited ``FACTORS`` value; nothing is tuned to the 20 mg/yr target (invariant I2).
    """
    mdot_g_s = ac225_atoms_per_s(r_mu, n_fus_mu, eta_pro) * atom_mass_g(a_molar, n_a)
    return mdot_g_s * s_per_year * 1000.0


def reproduce() -> dict[str, float]:
    """Run the full forward chain and both published cross-checks; return every intermediate value."""
    ndot_n = neutron_rate()
    ndot_225 = ac225_atoms_per_s()
    m_pro = atom_mass_g()
    p_fus = fusion_power_W()
    mg_yr = ac225_mg_per_year()
    return {
        "Ndot_n_per_s": ndot_n,
        "P_fus_W": p_fus,
        "Ndot_225Ac_per_s": ndot_225,
        "m_pro_g": m_pro,
        "Ac225_mg_per_year": mg_yr,
        "Ac225_ug_per_year": mg_yr * 1000.0,
        "dev_vs_headline_pct": 100.0 * (mg_yr - HEADLINE_MG_PER_YEAR) / HEADLINE_MG_PER_YEAR,
        "dev_vs_table1_pct": 100.0 * (mg_yr * 1000.0 - TABLE_I_UG_PER_YEAR) / TABLE_I_UG_PER_YEAR,
        "global_supply_multiple": (mg_yr * 1000.0) / GLOBAL_2024_UG_PER_YEAR,
    }


# --- MUON_COST.md tier cross-reference for the 1e12 muons/s source assumption -------------------------
# The 1e12 muons/s in the headline is a RATE assumption; MUON_COST.md's tiers classify muon COST
# (GeV/muon), not rate, so the rate does not map onto a cost tier. What IS cross-referenceable is the
# paper's muon-COST assumption and the rate's standing relative to real facilities:
MUON_COST_TIER = "T1-design-study"
MUON_COST_TIER_NOTE = (
    "Cost axis: Parisi & Rutkowski take E_mu ~= 5 GeV/muon (their ref. [9] = Jandel 1989; "
    "MUON_COST.md T1 design-study band, consistent with Eliezer-Henis 5.0) and cite a 3.0 GeV "
    "active-target source (their ref. [14] = Newburg ARPA-E 2025 = the Acceleron deck; MUON_COST.md "
    "T1 'Acceleron (2025 deck), 3.0 GeV' row). Both are MUON_COST.md Tier 1 (purpose-built design "
    "studies). Rate axis: the 1e12 muons/s itself is HYPOTHETICAL -- Parisi's own survey puts current "
    "mu- sources at ~1e7/s (J-PARC/PSI/MuSIC) rising toward 1e8/s with upgrades, and MUON_COST.md's "
    "highest-rate T3 operating facility (mu2e, ~1e10 stopped mu/s) is still ~100x below 1e12/s. So the "
    "SOURCE COST maps to T1 design-study; the RATE maps to no tier (it exceeds every real facility)."
)


def main() -> None:
    line = "=" * 100
    print(line)
    print("Parisi & Rutkowski (arXiv:2511.20951v2): Ac-225 from muCF -- forward reproduction of the headline")
    print(line)
    print("Pathway: 226Ra(n,2n)225Ra --beta^- (15 d)--> 225Ac  [eq. (28)];  design point = Table I (RaO, high flux)\n")

    print("FACTOR CHAIN (every row cited to its locator):")
    for f in FACTORS:
        print(f"  {f.symbol:<11} = {f.value:.6g} {f.unit}")
        print(f"      {f.name}")
        print(f"      cite: {f.citation}")
    print()

    r = reproduce()
    print("FORWARD COMPUTATION:")
    print(f"  Ndot_n   = R_mu * N_fus_mu            = {r['Ndot_n_per_s']:.4e} neutrons/s      [eq. (3),(A9)]")
    print(f"  P_fus    = Ndot_n * E_fus             = {r['P_fus_W']:.1f} W                 "
          f"[eq. (10); Table I: {TABLE_I_PFUS_W:.0f} W, abstract 'half a kilowatt']")
    print(f"  Ndot_225 = eta_pro * Ndot_n           = {r['Ndot_225Ac_per_s']:.4e} atoms/s        [eq. (39)]")
    print(f"  m_pro    = A_Ac / N_A                 = {r['m_pro_g']:.4e} g/atom         [eq. (14)]")
    print(f"  Mdot     = Ndot_225 * m_pro * s_year  = {r['Ac225_mg_per_year']:.3f} mg/yr "
          f"(= {r['Ac225_ug_per_year']:.0f} ug/yr)")
    print()
    print("REPRODUCTION vs PAPER:")
    print(f"  headline (abstract): {HEADLINE_MG_PER_YEAR:.0f} mg/yr   -> forward = "
          f"{r['Ac225_mg_per_year']:.2f} mg/yr  ({r['dev_vs_headline_pct']:+.2f} %)")
    print(f"  Table I (OpenMC):    {TABLE_I_UG_PER_YEAR:.0f} ug/yr -> forward = "
          f"{r['Ac225_ug_per_year']:.0f} ug/yr ({r['dev_vs_table1_pct']:+.2f} %)")
    print(f"  vs 2024 global supply ({GLOBAL_2024_UG_PER_YEAR:.0f} ug/yr, ref. [33]): "
          f"{r['global_supply_multiple']:.0f}x   (abstract: ~{GLOBAL_2024_MULTIPLE:.0f}x)")
    within = abs(r["dev_vs_headline_pct"]) <= RECON_TOL_FRAC * 100.0
    print(f"  => within +/-{RECON_TOL_FRAC * 100:.0f}% of headline: {'YES' if within else 'NO'}")
    print()

    print("MUON SOURCE -> MUON_COST.md tier cross-reference:")
    print(f"  1e12 muons/s source, cost tier = {MUON_COST_TIER}")
    print(textwrap.fill(MUON_COST_TIER_NOTE, width=96, initial_indent="    ", subsequent_indent="    "))
    print()

    print("FRAMING (invariant I9; the claim below is Parisi & Rutkowski's, quoted as theirs -- I8):")
    print(
        "  Below energy breakeven, muCF's utility is NEUTRON-SOURCE / MEDICAL-ISOTOPE economics, not\n"
        "  electricity. Selling the 14.1 MeV D-T neutrons as Ac-225 (a targeted-alpha-therapy isotope in\n"
        "  chronic short supply) relaxes the per-muon requirements by orders of magnitude: Parisi &\n"
        "  Rutkowski report the breakeven fusions-per-muon falling from N_fus,mu >~ 415 (electricity-only)\n"
        "  to N_fus,mu >~ 5e-7 (225Ac transmutation) [Sec III, Fig. 2]. On that basis they argue muCF\n"
        "  'systems employing transmutation could be viable well before energy breakeven is possible' and\n"
        "  that the finding motivates muon-source development 'far before net energy generation is\n"
        "  possible' [arXiv:2511.20951v2, abstract + Sec V]. OpenMuCF reproduces their arithmetic; the\n"
        "  viability judgement is theirs, not ours."
    )
    print(line)


if __name__ == "__main__":
    main()
