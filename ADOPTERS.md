# Adopter map — who OpenMuCF is for, and what each could consume

The point of OpenMuCF is **adoption**, not citation. Honest tiering of the (small, ~tens-of-groups) μCF
community and its funders, with what each would actually consume and what would make them run it.

## Tier 1 — best-fit users (could run it directly)
| Who | What they could consume | Why it fits |
|---|---|---|
| **Kou & Chen** (IMP-CAS / UCAS) | the open UQ baseline; the ledger; (Phase 3, planned) a computed ω_s^eff cross-check | their rate-network paper (arXiv:2606.07077) treats S_eff(E), σ_cap^eff(E), σ_γ as scan parameters; OpenMuCF already reproduces its 112.6 / 156.5 fusions-per-muon benchmarks within ~2.5% (VALIDATION.md) and adds propagated error bars over those scan ranges. Separately, the Yin–Kou–Chen review (arXiv:2605.26432) projects >500 fusions/muon and Q>2 — OpenMuCF gives an independent, error-barred baseline for that projection — reported as requirements (what would have to be true), not a verdict on their work |
| **Kino & Yamashita** (Tohoku) | the FAIR rate ledger + UQ layer feeding *their* kinetics; (stretch) the η/S(q,ω) module | they fit η=5 as a constant that a condensed-matter formation module would compute; they own the canonical (~25-population) kinetics and need consistent, sourced inputs — the pitch is the ledger+UQ layer feeding their code, not that they run our reduced network |
| **J-PARC / Toyama; PSI; RIKEN-RAL** experimentalists | a transparent forward model for sticking and cycling-rate data | an open, error-barred model to interpret yield/cycling measurements. (State-resolved ddμ\*/TES x-ray observables are a named v2 goal — d-d chain kinetics + resonance-state populations → 1.6–2.0 keV line intensities — not a v1 capability.) |
| **Students / new entrants** | everything | the only open cycle-kinetics starting point we could find (systematic search 2026-07; public release pending, see README) |

## Tier 2 — would ADOPT the ledger + UQ methodology, NOT replace their engine
| Who | What they adopt | Caveat |
|---|---|---|
| **Acceleron / MuFusE collaboration** (e.g. J.D. Kalow, K.R. Lynch, N.J.L. MacFadden, L.E. & A.N. Knaian; arXiv:2606.05333) | the FAIR ledger; the UQ methodology; the (Phase-3, planned) sticking surrogate as an *external cross-check* on the DAC measurements | they have their own GEANT4-based μCF extension classes (PoS ICHEP2022 1232 — test code, not in any public Geant4 release as of 11.4); OpenMuCF complements (exports surrogate rates / ingests spectra), it does not compete with transport engines |

## Tier 3 — independent yardstick
| Who | What they get |
|---|---|
| **Funding agencies and technical reviewers; skeptics** | an auditable, end-to-end, uncertainty-propagated **net-electrical** Q ledger — a pre-registered, error-barred baseline that lets credible measured results be priced above the noise, plus a ranked list of which measurement buys the most credibility per dollar |

## Honest adoption risks (tracked, not hidden)
- The DAC-regime ω_s verdict may resolve to "currently unconstrained" once above-1.45φ cross-section
  uncertainty is propagated. Pre-registered as a still-useful result (it tells the field what to measure).
