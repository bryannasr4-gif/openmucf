# LITERATURE.md — verified facts & numbers (with conditions)

> Every number carries its conditions (temperature T, density φ relative to liquid-hydrogen density,
> tritium fraction c_t, hyperfine state F) and a source. Items marked **[VERIFY P1]** have the right
> source identified but the exact digit must be pinned from the full text in Phase 1.
> Citations verified against their primary sources on 2026-06-29; the 2026 arXiv papers were confirmed
> against their published listings.

## 0. The one equation everything orbits

Fusions per muon (steady-state catalytic yield):

    X_μ = 1 / ( ω_s^eff + λ_0 / (φ · λ̃_c) )

- `λ_0` = muon decay rate = 1/τ_μ. **τ_μ = 2.1969811 µs → λ_0 = 4.552×10⁵ s⁻¹** (PDG; exact, settled).
- `ω_s^eff` = effective α-μ sticking probability (the hard ceiling on X_μ).
- `λ̃_c` = density-normalized cycling rate; actual cycling rate `λ_c = φ·λ̃_c`.
- `φ` = target density in units of liquid-hydrogen density (LHD = 4.25×10²² atoms/cm³).
- `c_t` = atomic tritium fraction.

Derivation is in `MODEL_SPEC.md` (absorbing-Markov / renewal argument). The engine's `analytic.py`
must reproduce this; `cycle.py` must reduce to it in the fast-fusion, thermalized limit.

## 1. Sticking & reactivation (THE bottleneck)

| Quantity | Value | Conditions | Source |
|---|---|---|---|
| Initial sticking ω_s^0 | ≈ 0.886–0.912 % (older); refined **[VERIFY P1]** | dtμ, J=v=0 | Markushin; Petrov; **Kamimura–Kino–Yamashita PRC 107, 034607 (2023)** = arXiv:2112.08399 |
| Reactivation fraction R | ≈ 0.25–0.35 | liquid/solid density; rises with φ | Cohen; Markushin; Stodden |
| Effective sticking ω_s^eff | ≈ 0.45–0.60 % | φ ~ 1.2 (liquid) | Breunlich–Kammel–Cohen–Leon ARNPS **39**, 311 (1989) |
| Kou–Chen baseline ω_s^eff / X_μ | **N_fus,μ = 112.6** (collision-only) → **156.5** (best external-field) | their rate-network model | **Kou–Chen arXiv:2606.07077 (2026)** |
| Kou–Chen sticking decomposition | ω_s^eff = ω_s^0(1−R_col)(1−R_X), R_X = f_X·P_X·η_X | external-field stripping | arXiv:2606.07077 |
| Kou–Chen no-go criterion | η_X^crit > 1 ⇒ external field cannot help, scheme-independently | — | arXiv:2606.07077 |

**Note:** initial sticking is the single most-recomputed, still-not-fully-settled μCF number (40-year
theory–experiment tension). We do NOT try to resolve it; we quantify how much it moves X_μ and Q.

## 2. Cycle rates

| Quantity | Value | Conditions | Source |
|---|---|---|---|
| Intramolecular fusion rate λ_f(dtμ) | ≈ 1.1×10¹² s⁻¹ | J=v=0 ground; effectively instantaneous | Bogdanova; Kamimura |
| Resonant dtμ formation λ_dtμ(T,φ,F) | rises with T; ~10⁸–10¹⁰ s⁻¹ scale | Vesman mechanism; strong F (hyperfine) dependence | Faifman; Vesman; Yamashita–Kino 2022 |
| dtμ formation resonance peak | ~7.1×10⁹ s⁻¹ near **E_cm ≈ 0.423 eV** | atomic-beam (epithermal) | Faifman; used as the Phase-1 sanity anchor |
| Isotopic transfer λ_dt (dμ→tμ) | ≈ 2.8×10⁸ s⁻¹ (×φ·c_t) **[VERIFY P1]** | drives muon onto tritium | Standard tables |
| tμ hyperfine spin-flip λ_10 | ~10⁸–10⁹ s⁻¹ scale **[VERIFY P1]** | F=1 → F=0 | Faifman |
| Cycling rate λ̃_c (density-NORMALIZED; Fig. 3a, c_t=0.5 EVM-SPM-FIF panel) | digitized: ≈ 0.84×10⁸ (300 K) → 1.97×10⁸ (800 K) s⁻¹, 800/300 ratio ≈ 2.36 (`openmucf/data/yamashita_kino_lc_T.csv`) | c_t ~ 0.5; a φ-normalized *gas* cycle rate — **NOT** the Breunlich *liquid* λ_c max 1.45×10⁸ s⁻¹ (`V_breunlich_lambdac`). The earlier "≈1.0–1.45×10⁸" reading here was a digitization under-read (corrected 2026-07-13); its numeric coincidence with the separate Breunlich 1.45×10⁸ anchor is a documented hazard (see `forecasts/FORECAST_PROTOCOL.md`). | **Yamashita–Kino Sci. Rep. 12, 6393 (2022)** |

## 3. Energy balance

| Quantity | Value | Notes | Source |
|---|---|---|---|
| Fusion energy E_f | 17.6 MeV (α 3.5 + n 14.1) | d+t → α+n | settled |
| Muon production cost E_μ | ≈ 5 GeV (range ~2–10 GeV cited) | dominant cost; accelerator-dependent | Jändel; Petrov; reviews |
| Scientific breakeven X_μ | ≈ E_μ/E_f = 5000/17.6 ≈ **284** (for Q=1, η_conv=1) | the number current μCF cannot reach | derived |
| Current record X_μ | ≈ **150** (150±4±20) | best experimental; high-T/high-c_t conditions | Jones et al. PRL 56, 588 (1986); Yin–Kou–Chen 2026 |
| Energy gain Q (current) | ≈ 0.3–0.5 | with realistic η_conv | reviews |

## 4. The 2026 "breakeven" claims (the falsification targets)

- **Yin–Kou–Chen, arXiv:2605.26432 (2026), "Muon-Catalyzed Nuclear Fusion: Physical Mechanism,
  Bottleneck Breakthroughs, and an Engineering Pathway"** (IMP Lanzhou + GUCAS): formulates the cycle in
  four steps; identifies α-sticking as the central bottleneck; projects that a "four-dimensional
  synergistic scheme" (dual polarization + high-density confinement + electric-field-assisted muon
  recovery + resonant enhancement — per the abstract; in-flight μCF and heavy-ion-driven
  magneto-inertial fusion are separately-discussed breakthrough routes, §IV) could lift
  fusions-per-muon from the record ~150 to **"more than 500"**,
  **potentially enabling Q > 2**. These are single-point, error-bar-free projections. ← our marquee target.
- **Kou–Chen, arXiv:2606.07077 (2026)**: rate-network model of external-field-assisted reactivation;
  N_fus,μ 112.6 → 156.5; derives the η_X^crit > 1 no-go. ← our Phase-3 validation anchor + Phase-4 sibling.

## 5. The 2026 renaissance context (why now)

- **Toyama et al., Science Advances 12, eaed3321 (2026)**: first **direct observation of muonic molecules
  in resonance states (ddμ\*)** via transition-edge-sensor microcalorimeters (10× resolution), J-PARC;
  ddμ\* x-rays in the 1.6–2.0 keV range; excellent agreement with theory. Validates the resonant-molecule
  picture our kinetics rests on.
- **Acceleron Fusion** (US startup, ~$24M raise; IEEE Spectrum 2026): diamond-anvil-cell high-density DT
  μCF; μCF extension classes (MuonicAtomTransfer, MuonCatalyzedFusion — test code, PoS ICHEP2022 1232, not in
  any public Geant4 release as of 11.4) built on Geant4's OPEN muonic-atom classes (G4MuonMinusAtomicCapture,
  G4MuonicAtomDecay); "catalyzed fusion physics" is on Geant4's 2024 work plan. Confirms an active, partly-closed US revival.
- **Yamashita–Kino Sci. Rep. 12, 6393 (2022)**: the canonical modern kinetics model (3 resonant-molecule
  roles: isotopic-population change, epithermal muonic atoms, in-flight fusion); reproduces rising λ_c with
  T to 800 K. Closed-source, RK4, only a local 1-parameter sensitivity scan. ← our primary λ_c(T) benchmark.

## 6. SETTLED vs CONTESTED

| SETTLED (use as fixed inputs) | CONTESTED / model-dependent (carry uncertainty + sensitivity) |
|---|---|
| λ_0 = 4.552×10⁵ s⁻¹ | initial sticking ω_s^0 (0.857–0.93 %, method-dependent) |
| E_f = 17.6 MeV; breakeven X_μ ≈ 284 | reactivation R (density- and model-dependent) |
| λ_f ≈ 10¹² s⁻¹ (fusion ≫ formation) | λ_dtμ(T,φ,F) absolute scale & epithermal enhancement (η ~ 1 vs ~5 debate) |
| resonant (Vesman) formation is real (Toyama 2026) | muon production cost E_μ (2–10 GeV) and η_conv chain |
| record X_μ ≈ 150; Q < 1 today | whether 2026 N_μ>500 / Q>2 projections survive honest uncertainty ← WE TEST THIS |

## 7. The exact claim this project makes (and does not)

> **Claims** (to our knowledge; systematic search of GitHub/Zenodo/PyPI/IAEA, 2026-07): the first open,
> reproducible, differentiable muCF cycle + energy-balance engine; the first
> *global* sensitivity/identifiability ranking of which microscopic rate dominates X_μ and Q uncertainty
> with propagated error bars; and the first independent, uncertainty-quantified audit of the 2026
> Yin–Kou–Chen / Kou–Chen breakeven projections.
>
> **Does NOT claim:** any new fundamental rate or microphysics; any resolution of the α-sticking
> discrepancy; any endorsement of Q > 2, "cold", or ultradense μCF. The contribution is methodological
> infrastructure plus an honest finding, nothing more.

## 8. High-density / condensed-matter regime + reactivation (added 2026-06-29)

**Acceleron / MuFusE diamond-anvil-cell program (verified 2026-06-29):**
- **arXiv:2606.05333** — *"The MuFusE Large-Volume Diamond Anvil Cell for Exploring μCF at Higher Pressures
  and Temperatures"* (Kalow et al., ~46 authors): 19.2 mm³ sample at liquid density, ≤933 MPa, ≤400 K, 25 Ci
  tritium, in-situ laser spectroscopy. Exceeds prior static d-t target limits.
- **arXiv:2606.19304** — *"Design and Commissioning of a DT Gas Delivery System for μCF in a DAC"*: compresses
  DT to GPa, **>2× liquid density**, cryogenic through **500 K**. Expands the μCF kinetics/yield parameter range.
- **Motivation (the flagship's reason to exist):** the high-density dependence of effective sticking /
  reactivation is the live unknown the DAC was built to measure. **[VERIFIED verbatim, 2026-07-01, in the
  arXiv:2606.05333 introduction]**: "the measured values of ωs at high density are typically 10–50% lower than
  those predicted by standard theoretical models [Jones 1986; Rafelski et al. 1989], suggesting the presence of
  additional reactivation mechanisms" — note this is Acceleron summarizing PRIOR literature, not their own
  measurement (their ω_s/λ_c analysis is ongoing, no published value yet).

**Effective-sticking / reactivation prior art (the flagship is open+differentiable+UQ on THIS physics):**
| Source | Year | What it established |
|---|---|---|
| Stodden, Monkhorst, Szalewicz, Winter | 1990 | Energy-resolved (αμ)⁺ slowing-down transport with accurate stripping+excitation cross sections → reactivation R. **The exact method the flagship modernizes.** Closed Fortran, no UQ; two densities (1.2 and 0.05 LHD), no continuous φ dependence. |
| H.E. Rafelski & B. Müller (AIP Conf. Proc. 181, 355); H.E. Rafelski, Müller, J. Rafelski, Trautmann, Viollier (PPNP 22, 279) | 1988 / 1989 | Density-dependent stopping power of the (αμ)⁺ ion is the crucial factor controlling the density dependence of effective sticking. **The flagship's headline thesis** (we add modern inputs + error bars + extrapolation). |
| J. Cohen (PRA 35, 1419, CTMC) | 1987 | Tested target-structure effects on (αμ) stripping and found them NOT important at the densities studied — the open question is whether that survives the DAC regime. |
| Froelich & Larson | 1989 | (αμ) stripping by ionization in dense D/T (the dense-regime stripping problem). |
| Adamczak & Faifman | ~2000s | S(q,ω) / generalized–Van-Hove recasting of resonant dtμ formation (the basis of the *stretch* λ_dtμ(E,φ,T,phase) module; we'd extend to the anharmonic warm-dense fluid). |
| Petitjean / PSI | 1990s | Experimental ω_s ≈ 0.45% with weak density dependence to ~1.2–1.45 φ (the Phase-3 validation anchor below the DAC regime). |

**Honest originality position:** none of the above is ours; the contribution is the FIRST **open +
differentiable + propagated-UQ + provenance** artifact that ties a FAIR rate ledger to a falsification auditor
**and produces** ω_s^eff(φ,T,c_t) with a compute-trained surrogate (instead of hard-coding it), plus an
honest, pre-registered forecast for the DAC regime.
