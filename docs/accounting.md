# accounting.md — one channel, one accounting home

Measured effective parameters already CONTAIN deferred physics as it occurred in the anchor experiments
(`formation._CALIB`; the measured `lambda_c` band). The measured effective sticking is not one of them: Breunlich et al. (1987, p.331) corrected the raw loss per cycle for the dμd, tμt, pμd and pμt fusion losses before reporting it. Adding an
explicit channel therefore requires RE-ATTRIBUTION under the constraint that anchor-condition totals still
match — it is a **joint refit**, never "we added more physics so the numbers moved." This table is the
single source of truth for where each channel's effect lives; `cycle.py`, `MATERIALITY.md`, and Phase 4
all consume it. When a channel is made explicit in `cycle.py`, its ledger rate row (`rates.csv`) and its
entry here move together.

| channel | where its effect lives TODAY (v1) | evidence / source | re-attribution rule when made explicit | status |
|---|---|---|---|---|
| ttμ formation + tt-sticking | a side-channel loss per cycle, separate from the measured effective sticking (`omega_s_eff_obs`=0.45%), from which the source subtracted it | Breunlich et al., *Phys. Rev. Lett.* 58, 329 (1987), p.331 (the correction), p.332 (the yield 113 computed with the raw loss). tt rates: `lambda_ttmu` (JINR preprint E15-2008-99, Table 1, read as the rate in the target) and `omega_tt`=0.139, rate-ledger rows | none at the anchor: the calibration's loss per cycle is W = ω_s^eff + L_other, with the tt loss inside L_other at the anchor fill (`openmucf.sidechannels`) | explicit in `cycle.py` (v1.1) and live (`lambda_ttmu` > 0); off by default (`include_loss_channels=False`) |
| pμd + pμt (protium in the fill) | a side-channel loss per cycle at the anchor fill, which held about 1% protium, subtracted by the source with the tt and dd losses | Breunlich et al. (1987), p.331; coefficient k_p from Jones et al. (1986), p.590; `openmucf/data/side_channel_inputs.csv` | part of L_other at the anchor fill; sample-specific, never carried to other conditions | in `openmucf.sidechannels` only (not a `cycle.py` channel) |
| ³He scavenging (dμ + ³He) | absent from the anchors (fresh/He-purged fills, negligible c_He); NOT contained in `omega_s_eff_obs` | `lambda_dhe3` = 1.92(3)e8 s⁻¹ (³Heμd molecular-complex formation rate, λ_d³He=192(3)e6 s⁻¹; Fotev et al., arXiv:2001.09927, 2020 — open), matching MODEL_SPEC §8 `lambda_dHe`~1.9e8; carries needs_verification pending the full-text normalization pin; at the liquid anchor the losses to impurities and to ³He from tritium decay were bounded below 2×10⁻⁴ per cycle (Breunlich et al. 1987, p.331) | no re-attribution at the anchor (c_He≈0 there); enters only burn-time scenarios via a STATIC per-run `c_He` — never time-evolved inside the single-muon ODE (timescale separation ~9 orders) | explicit in `cycle.py` (v1.1), off by default (`c_he`=0.0) |
| d-recapture / q_1s routing | inside `formation._CALIB` + the measured `lambda_c` band (v1 recycles freed muons straight to the tμ pool, ¾/¼) | MODEL_SPEC §8 deferred-channel table (−6% to −14% for q_1s=0.4–1.0; the 300 K anchor absorbs much of it; computed 2026-07-13: −5.96%…−13.68% at 300 K/1.2 φ). Primary q_1s cascade literature (Cohen/Markushin/Rafelski κ set, *Muon Catal. Fusion*): **unknown — pending acquisition** | re-attribution deferred (would require unfolding the `_CALIB` anchor against the q_1s cascade fraction) | bracketed in `cycle.py` (first-order c_d·q_1s routing, `f_d = (1−c_t)·q_1s`; tabulated in `MATERIALITY.md`); `_CALIB` unfolding still deferred (pending acquisition of the primary q_1s / cascade literature, *Muon Catal. Fusion*) |
| ddμ / d-d branch | omitted from `cycle.py`; at the anchor fill it is part of L_other (`openmucf.sidechannels`); part of the documented −5…−15% one-sided structural headroom | MODEL_SPEC §8 (~−0.1% on X_μ at c_t≈0.5; matters for interpreting d-d datasets). Interpretation anchors: Toyama et al., *Sci. Adv.* 12, eaed3321 (2026) (ddμ\* resonance) + the MuFusE DD runs (arXiv:2606.05333) | needs the full ddμ/d-d branch (Toyama/TES + MuFusE DD interpretation) before it can be attributed | deferred (documented scope; MODEL_SPEC §8) |
| formation-rate multiplier (η) | the `eta_dtmu` ledger row, a multiplier on the measurement-anchored formation rate (η=1 the anchored rate … η=5 a five-times-anchored scenario); reported as a STRUCTURAL BRACKET in FINDINGS §1c | Yamashita & Kino, *Sci. Rep.* 12, 6393 (2022) (EVM η fit); FINDINGS §1c bracket | never convolved into the measured-`lambda_c` CI — the measured band already contains η as it occurred at the anchors (the one-home rule); η stays a bracket beside the CI | bracketed (FINDINGS §1c) |

## The anchor total (the binding rule)

At the liquid anchor of Breunlich et al. (1987) the yield relation sees the total loss per cycle, the effective
sticking plus the side-channel losses the source subtracted from it (p.330 Eq. (2), p.331-332):

```
W     =  omega_s0*(1 - R)  +  L_other      (0.45 % measures the first term; L_other is the anchor fill's)
X_mu  =  1 / (W + lambda_0/lambda_c)
```

Making a side channel explicit therefore adds to W and leaves the measured sticking where it is.
`CALIBRATION.md` fits the sticking, the cycle rate and the yield together, with L_other at the anchor fill as a
parameter.

The ³He channel is different: it needs **no** re-attribution —
it only ever adds loss in forward burn-time scenarios with a nonzero static `c_He`.
