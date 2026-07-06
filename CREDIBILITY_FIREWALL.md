# Credibility firewall

OpenMuCF models the **conventional** muon-catalyzed-fusion (μCF) cycle across the full physically-accessible
range — gas, liquid, cryogenic solid, and high-pressure diamond-anvil-cell (DAC) targets up to GPa pressures
and >2× liquid density — using **standard quantum mechanics, QED, and nuclear physics**. Every rate that enters
the ledger comes from a peer-reviewed or arXiv source using that standard framework, with provenance recorded
in `openmucf/data/references.bib`.

To keep the tool trustworthy as shared infrastructure, the following claim classes are **explicitly excluded**,
and excluded *on the record* (documented here) rather than silently omitted:

| Excluded | Why |
|---|---|
| **Holmlid "ultra-dense hydrogen" H(0) / Rydberg matter** muon and fusion claims | The proposed ~picometre H(0) state is inconsistent with standard QM; the muon/kaon and fusion yields have not been independently replicated and are widely disputed. |
| **LENR / "cold fusion" / electron-screening-as-fusion-enhancer** | No standard-model mechanism produces the claimed rates; not reproducible. (Ordinary electron screening of *real* μCF rates IS in scope — that is conventional physics; "screening as a cold-fusion mechanism" is not.) |
| **Piezonuclear / fracto-fusion** energy claims | Unsupported by standard nuclear physics; unreplicated. |
| **"Cold/ultradense μCF beats the sticking limit" press claims without a mechanism** | We model the conventional reactivation physics that *does* lower effective sticking at high density (Stodden 1990, Rafelski–Müller 1988/89); we do not import claims that bypass the ω_s ceiling without a standard mechanism. |

**In scope and welcome:** the genuine, debated, standard-physics questions — e.g. why measured high-density ω_s
runs 10–50% below standard theory (Acceleron/MuFusE, arXiv:2606.05333), the epithermal enhancement η, and the
density/temperature dependence of reactivation. Quantifying those honestly, *with* uncertainty, is the point.

**Rule:** any future rate or claim added to the ledger must cite a standard-physics source. If a number's only
provenance is an excluded class above, it does not go in the ledger — and the exclusion is noted here.
