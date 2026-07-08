# accounting.md — one channel, one accounting home (Invariant I5)

Measured effective parameters already CONTAIN deferred physics as it occurred in the anchor experiments
(`omega_s_eff` from Petitjean/Breunlich; `formation._CALIB`; the measured `lambda_c` band). Adding an
explicit channel therefore requires RE-ATTRIBUTION under the constraint that anchor-condition totals still
match — it is a **joint refit**, never "we added more physics so the numbers moved." This table is the
single source of truth for where each channel's effect lives; `cycle.py`, `MATERIALITY.md`, and Phase 4
all consume it. When a channel is made explicit in `cycle.py`, its ledger rate row (`rates.csv`) and its
entry here move together.

| channel | where its effect lives TODAY (v1) | evidence / source | re-attribution rule when made explicit | status |
|---|---|---|---|---|
| ttμ formation + tt-sticking | inside the measured `omega_s_eff_obs`=0.45% (the Petitjean/Breunlich anchor ran at c_t≈0.4–0.5, so the tt side-branch loss is already folded into the fitted effective sticking) | Breunlich et al., *Phys. Rev. A* 40, 1907 (1989) — anchor `omega_s_eff`=0.45% at c_t≈0.4–0.5. tt-branch rates (`lambda_ttmu`, `omega_tt`≈0.14): **unknown — pending acquisition of the Matsuzaki/Bom tt-fusion tables (*Muon Catal. Fusion*)**; the ledger `lambda_ttmu` row ships 0.0 (blocked) and `omega_tt` carries needs_verification | joint refit: the `omega_s0(1−R)` posterior shifts DOWN by the explicit tt per-cycle loss at anchor conditions so the TOTAL per-cycle loss still reproduces 0.45% | explicit in `cycle.py` (v1.1) but **INERT — blocked**: `lambda_ttmu`=0.0 pending the ttμ formation-rate tables, so the re-attribution refit is not yet run |
| ³He scavenging (dμ + ³He) | absent from the anchors (fresh/He-purged fills, negligible c_He); NOT contained in `omega_s_eff_obs` | `lambda_dhe3` = 1.92(3)e8 s⁻¹ (³Heμd molecular-complex formation rate, λ_d³He=192(3)e6 s⁻¹; Fotev et al., arXiv:2001.09927, 2020 — open), matching MODEL_SPEC §8 `lambda_dHe`~1.9e8; carries needs_verification pending the full-text normalization pin | no re-attribution at the anchor (c_He≈0 there); enters only burn-time scenarios via a STATIC per-run `c_He` — never time-evolved inside the single-muon ODE (timescale separation ~9 orders) | explicit in `cycle.py` (v1.1), off by default (`c_he`=0.0) |
| d-recapture / q_1s routing | inside `formation._CALIB` + the measured `lambda_c` band (v1 recycles freed muons straight to the tμ pool, ¾/¼) | MODEL_SPEC §8 deferred-channel table (−6% to −14% for q_1s=0.4–1.0; the 300 K anchor absorbs much of it). Primary q_1s cascade literature (Cohen/Markushin/Rafelski κ set, *Muon Catal. Fusion*): **unknown — pending acquisition** | re-attribution deferred (would require unfolding the `_CALIB` anchor against the q_1s cascade fraction) | deferred (pending acquisition of the primary q_1s / cascade literature, *Muon Catal. Fusion*) |
| ddμ / d-d branch | omitted (the c_d channel is not modeled); part of the documented −5…−15% one-sided structural headroom | MODEL_SPEC §8 (~−0.1% on X_μ at c_t≈0.5; matters for interpreting d-d datasets). Interpretation anchors: Toyama et al., *Sci. Adv.* 12, eaed3321 (2026) (ddμ\* resonance) + the MuFusE DD runs (arXiv:2606.05333) | needs the full ddμ/d-d branch (Toyama/TES + MuFusE DD interpretation) before it can be attributed | deferred (documented scope; MODEL_SPEC §8) |
| epithermal formation enhancement (η) | the `eta_dtmu` ledger row (η=1 bare theory … ~5 fit); reported as a STRUCTURAL BRACKET in FINDINGS §1c | Yamashita & Kino, *Sci. Rep.* 12, 6393 (2022) (EVM η fit); FINDINGS §1c bracket | never convolved into the measured-`lambda_c` CI — the measured band already contains η as it occurred at the anchors (one-home rule I5); η stays a bracket beside the CI | bracketed (FINDINGS §1c) |

## Re-attribution constraint (the binding rule)

For any channel currently folded into a measured effective parameter, making it explicit MUST preserve the
anchor-condition total. Concretely for the ttμ channel at the Petitjean/Breunlich anchor (φ≈1.2, c_t≈0.45):

```
omega_s0*(1 - R)  +  tt_pc   =   0.45%   (the measured effective per-cycle loss)
                     ^^^^^                where tt_pc = omega_tt * lambda_ttmu * phi * c_t / lambda_c
```

so introducing `tt_pc` > 0 forces the fitted `omega_s0(1−R)` DOWN by exactly that share; X_μ at the anchor
is unchanged by construction (it is pinned by the same 0.45% total plus λ₀/λ_c). This is loss
**re-attribution**, not new physics. Because `lambda_ttmu` is currently blocked (0.0), `tt_pc`=0 and the
refit is a no-op — the channels-off VALIDATION.md scoreboard remains the v1 trust gate, and the refit is
recorded as *blocked — pending acquisition of the Matsuzaki/Bom tt tables* until the primary is in hand.

The ³He channel is different: it is genuinely ABSENT from the anchors, so it needs **no** re-attribution —
it only ever adds loss in forward burn-time scenarios with a nonzero static `c_He`.
