# Step-0: (αμ)⁺ + D/T cross-section database & literature scan

> **Phase-3 due-diligence memo.** OpenMuCF's planned Phase-3 module computes the muon *reactivation*
> probability *R* — the fraction of muons stripped back off the fusion alpha while the muonic-helium ion
> (αμ)⁺ slows down through the D/T medium — from energy-resolved (αμ)⁺ + D/T excitation / ionization /
> stripping cross sections. Before any compute is scoped, this memo records what curated data and published
> calculations for those cross sections actually exist. It is the "secure the cross-section anchor first"
> step of the Phase-3 plan: the anchor, not the FLOPs, is the real gating acquisition.
>
> **Bottom line.** No atomic/molecular database curates the muonic-ion collisional data (it is out of scope
> and unrepresentable in the standard species notation). The field's cross-section anchor is a single
> mass-scaled 1990 *electronic-analog* calculation, backstopped by late-1980s muonic-helium kinetics and
> Stark-mixing fits; the *actual* muonic (αμ)⁺ + D/T system has never been recomputed in the stripping
> regime (2000–2026 search returned nothing). A modern direct calculation is therefore **desirable but not
> existence-critical** — Phase 3 can proceed on classical-trajectory Monte Carlo (CTMC) plus
> literature-bounded uncertainty, with a pre-registered *"currently unconstrained above φ ≈ 1.45"* outcome
> if the anchor cannot be tightened.

**Search date:** 2026-07-06.
**Method note:** the databases' coverage, species notation, and search interface were retrieved and
inspected directly. The interactive query endpoints could not be exercised programmatically, so the
database conclusion below rests on the databases' *documented scope and species conventions* (which
structurally exclude a muon-bound exotic species), not on an enumerated empty result set. Stated this way
deliberately rather than reporting a query that was not run.

## 1. Database search — IAEA CollisionDB and ALADDIN

- **IAEA CollisionDB** (`amdis.iaea.org/db/collisiondb/`, release v2024.1, ~122k datasets). Species are
  encoded in **PyValem** notation — ordinary element symbols with charge and isotope labels (`H+`, `D`,
  `T`, `Be+4`). Heavy-particle process codes cover exactly the processes of interest — `HST`
  (electron-stripping), `HIN` (ionization), `HEX` (excitation), `HCX` (charge transfer) — but only for
  standard atoms / ions / molecules. A muon-bound exotic species such as (αμ)⁺ / μHe⁺ **cannot be encoded**
  in the notation, and the database scope is explicitly electrons, photons, and heavy particles with atomic
  and molecular species for fusion-plasma applications. The *electronic analog* p + He⁺ heavy-particle
  channels are in scope (and plausibly present); the muonic system is structurally absent.
- **IAEA ALADDIN** (`amdis.iaea.org/ALADDIN/`) — the legacy IAEA numerical A+M database for
  fusion-relevant electron / heavy-particle collisions on ordinary atoms and ions, being superseded by
  CollisionDB. Same fusion-plasma scope; no exotic-atom / muonic collisional category.
- **Conclusion.** No curated muonic-ion (αμ)⁺ + H/D/T collisional dataset exists in the IAEA A+M
  databases — it is both out of scope and unrepresentable in the species notation. The closest in-scope
  curated data is the electronic-analog p + He⁺ heavy-particle set.

## 2. Existing literature anchor (citations verified 2026-07-06)

| Reference | What it provides | Notes |
|---|---|---|
| **Stodden, Monkhorst, Szalewicz & Winter, Phys. Rev. A 41, 1281 (1990)** | Impact-parameter coupled-state (Sturmian basis, up to 51 states) excitation / ionization / charge-transfer for the electronic analog p–He⁺, **mass-scaled to (αμ)⁺ + H**; kinetics solved for the muon-stripping probability and K/L X-ray yields | c.m. energy range **20–600 keV**. Stripping-probability error bars **9 % and 11 %**, quoted at densities **1.2 × and 0.05 × liquid-hydrogen density** (φ = density relative to LHD). This remains the field's cross-section anchor. |
| **Cohen, Phys. Rev. Lett. 58, 1407 (1987)** | State-resolved (n-, l-changing) **kinetics of muonic helium** in μCF d-d / d-t; density-dependent stripping and X-ray production | "Kinetics of muonic helium in muon-catalyzed d-d and d-t fusion." |
| **Struensee & Cohen, Phys. Rev. A 38, 44 (1988)** | 2S–2P **Stark mixing** of muonic helium in collisions with hydrogen; closed-form fit | σ₂ₛ₋₂ₚ ≈ **5.5 × 10⁻³ / v¹·⁸ a₀²** (v in atomic units, valid v ≳ 1). |
| **Rafelski, Müller, Rafelski, Trautmann & Viollier, Prog. Part. Nucl. Phys. 22, 279 (1989)** | Reanalysis of the density dependence of effective sticking; **density-dependent dense-hydrogen stopping power** as the controlling density lever | "Muon reactivation in muon-catalyzed d-t fusion," pp. 279–338. |

## 3. The modern-quantum-anchor gap (2000–2026)

No modern coupled-channel / quantum-scattering recomputation of the *actual* muonic (αμ)⁺ + D/T system in
the stripping regime was found for 2000–2026. The direct-muonic-system collisional work is all late-1980s
(Cohen's CTMC and coupled-equation treatments; the coupled-state numbers still trace to the mass-scaled
Stodden 1990 electronic analog). Post-2000 activity is adjacent rather than a fresh in-medium cross-section
calculation: external-field-assisted stripping (cyclotron-resonance schemes, PTEP 2021 093G01; the 2026
rate-network external-field-assisted reactivation study, arXiv:2606.07077), muonic-molecule resonance
observation (Sci. Adv. 2026), and high-density diamond-anvil-cell experiments (arXiv:2606.19304). None
recomputes the slowing-down (αμ)⁺ + D/T excitation / ionization / stripping cross sections quantum
mechanically.

## 4. Step-0 conclusion

A single mass-scaled 1990 electronic-analog set (Stodden et al., p–He⁺ → (αμ)⁺ + H) remains the field's
cross-section anchor, backstopped by 1980s Cohen / Struensee kinetics and Stark-mixing fits and the
Rafelski density / stopping-power lever; no modern direct-muonic quantum-scattering recompute exists
(2000–2026), and no IAEA database curates the muonic-ion collisional data. A modern direct calculation is
therefore desirable but not existence-critical. **Phase 3 can proceed** on CTMC + literature-bounded
uncertainty for *R*, with *"currently unconstrained above φ ≈ 1.45"* an acceptable, pre-registered outcome
rather than a blocker. The stopping power *S(v, φ)* is treated as a first-class uncertain input (Rafelski
1989), and any quantum anchor obtained later tightens — rather than gates — the result.
