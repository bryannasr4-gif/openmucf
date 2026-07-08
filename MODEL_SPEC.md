# MODEL_SPEC.md — the muCFkin cycle model (derived from scratch)

This document defines the model **before any code**. The engine (`cycle.py`) implements the ODE network in
§3; `analytic.py` implements the closed form in §4; Phase 3 validates against the targets pre-registered in §7.

---

## 1. Physical picture (one paragraph)

A μ⁻ stops in a dense D/T mixture and cascades (~10⁻¹¹ s) into a muonic atom. Because m_μ ≈ 207 m_e, the
muonic atom is ~207× smaller than ordinary hydrogen. The muon migrates down the isotope ladder to **tμ**
(more deeply bound). A tμ atom collides with a D₂ (or DT) molecule and forms the muonic molecular ion
**dtμ** by the **Vesman resonant mechanism**: dtμ has an anomalously weakly bound J=1, v=1 state (~0.66 eV)
whose binding energy is resonantly absorbed into rovibrational excitation of the host molecule. Inside dtμ
the d and t sit ~500 fm apart, so they **fuse essentially instantly** (λ_f ≈ 1.1×10¹² s⁻¹) →
**d + t → α + n + 17.6 MeV**. The muon is normally freed and catalyzes again — unless it **sticks** to the
α (forms αμ⁺, probability ω_s⁰ ≈ 0.9 %), partially undone by **reactivation** R as the αμ⁺ slows down,
leaving an effective loss ω_s^eff = ω_s⁰(1−R) ≈ 0.5 %. Two clocks bound the yield: muon **decay**
(λ₀ = 4.552×10⁵ s⁻¹) racing the **cycling rate** λ_c, and the per-cycle **sticking** loss.

---

## 2. State space (v1)

Muon-state occupation probabilities (each subject to decay λ₀ → `dec`), plus absorbing accumulators:

| symbol | meaning |
|---|---|
| `x_dμ`  | muon on a deuteron (post-cascade) |
| `x_tμ1` | muon on a triton, hyperfine F=1 (triplet, statistical weight 3) |
| `x_tμ0` | muon on a triton, hyperfine F=0 (singlet, statistical weight 1) |
| `x_mol` | dtμ molecule formed (transient; fuses at λ_f) |
| `N_fus` | cumulative fusions (the observable; X_μ = N_fus(∞) with one starting muon) |
| `stuck` | muon lost to α-sticking (absorbing) |
| `dec`   | muon lost to decay (absorbing) |

v1 lumps dμ hyperfine (F=1/2,3/2) into one `x_dμ`; tμ hyperfine is **kept** because λ_dtμ depends strongly
on F. The probability conservation invariant (checked numerically every run):
`x_dμ + x_tμ1 + x_tμ0 + x_mol + N_fus·0 + stuck + dec` is NOT conserved (N_fus counts events, not occupancy);
the conserved quantity is `x_dμ + x_tμ1 + x_tμ0 + x_mol + stuck + dec = 1` at all times.

---

## 3. Governing ODE network (full v1)

Rates (all s⁻¹; density-dependent ones already multiplied by φ and the relevant concentration):

- `λ₀` decay (acts on every muonic state),
- `λ_dt`  isotopic transfer dμ → tμ  (∝ φ·c_t),
- `λ₁₀`   tμ hyperfine spin-flip F=1 → F=0,
- `λ_f^F` = λ_dtμ^F(T,φ) effective formation+fusion rate from tμ(F) (fast-fusion limit folds λ_f in; see §5),
- `ω_s^eff` per-fusion sticking loss; `(1−ω_s^eff)` recycles the muon back to the tμ pool.

On recycle after fusion, the freed muon re-thermalizes and re-forms tμ; v1 returns it to the **statistical**
tμ hyperfine mix (¾ to F=1, ¼ to F=0). (A returned-muon-via-dμ refinement is a v2 option.)

```
dx_dμ/dt   = −(λ_dt + λ₀) x_dμ
dx_tμ1/dt  = +λ_dt x_dμ − (λ₁₀ + λ_f^{1} + λ₀) x_tμ1
             + ¾ (1−ω_s^eff) (λ_f^{1} x_tμ1 + λ_f^{0} x_tμ0)        # recycled muons, statistical split
dx_tμ0/dt  = +λ₁₀ x_tμ1 − (λ_f^{0} + λ₀) x_tμ0
             + ¼ (1−ω_s^eff) (λ_f^{1} x_tμ1 + λ_f^{0} x_tμ0)
dN_fus/dt  = +(λ_f^{1} x_tμ1 + λ_f^{0} x_tμ0)                        # every formation → prompt fusion
dstuck/dt  = +ω_s^eff (λ_f^{1} x_tμ1 + λ_f^{0} x_tμ0)
ddec/dt    = +λ₀ (x_dμ + x_tμ1 + x_tμ0)
```

Initial condition (v1): one muon entering the cycle on deuterium → `x_dμ(0)=1`, all else 0. (A capture-ratio
split between x_dμ and x_tμ at t=0 is a knob; it barely affects X_μ because transfer is fast.)

`X_μ ≡ N_fus(t→∞)`. This is a **linear** ODE system ⇒ X_μ has a closed form (§4), giving the engine a free
internal consistency check.

> **Stiffness:** λ_f ~ 10¹² vs λ₀ ~ 10⁵ spans 7 decades ⇒ stiff. In the fast-fusion limit we do NOT integrate
> λ_f explicitly; `λ_f^F` already denotes the *formation-limited* rate (formation ≪ fusion), so the stiff
> ratio is removed and a Kvaerno/implicit solver handles the residual 10⁵–10¹⁰ span. The molecule occupancy
> `x_mol` is adiabatically eliminated (it never accumulates because λ_f is huge). We keep `x_mol` only as an
> optional diagnostic, not a dynamical variable, in v1.

---

## 4. Closed-form reduction (the analytic backbone)

Collapse the hyperfine/isotopic structure into a single cycling pool `m(t)` (muon alive and able to fuse),
an effective cycle rate `λ_c`, and effective sticking `ω_s^eff`. Then:

```
dm/dt     = −(λ_c + λ₀) m + (1−ω_s^eff) λ_c m = −(λ₀ + ω_s^eff λ_c) m,   m(0)=1
dN_fus/dt = λ_c m
```

Solve: m(t) = exp[−(λ₀ + ω_s^eff λ_c) t], so

```
X_μ = ∫₀^∞ λ_c m dt = λ_c / (λ₀ + ω_s^eff λ_c) = 1 / ( ω_s^eff + λ₀/λ_c ).
```

**Equivalent renewal derivation (the whiteboard version):** per cycle, P(fusion before decay) =
p = λ_c/(λ_c+λ₀); given fusion, P(survive sticking) = s = 1−ω_s^eff. Then P(≥k fusions) = p (s p)^{k−1}, so

```
X_μ = Σ_{k≥1} p (s p)^{k−1} = p/(1 − s p) = λ_c/(λ₀ + ω_s^eff λ_c) = 1/(ω_s^eff + λ₀/λ_c).   ∎
```

With λ_c = φ·λ̃_c this is exactly the `LITERATURE.md` identity. **This is the formula the student must be
able to derive live.** `analytic.py` returns it; `cycle.py`'s `N_fus(∞)` must match it to <1 % in the
single-pool limit (regression gate).

### 4.1 Extended closed form with the ttμ side-branch (v2 — WS-N, 2026-07-08)

The v1 renewal derivation has **two** competing per-cycle outcomes for the muon in the cycling pool:
d-t formation-fusion (rate λ_c) and decay (λ₀). WS-N makes the **ttμ side-branch** explicit as a **third
competing first-order hazard** out of the same pool: a tμ atom can form **ttμ** (rate λ_tt = λ_ttμ·φ·c_t)
instead of dtμ, and after the tt fusion the muon is lost with probability ω_tt or returned with
probability (1−ω_tt). (³He scavenging is a *different* hazard — see the asymmetry note below.)

Per episode, with total exit rate Λ = λ_c + λ_tt + λ₀, the competing branch probabilities are
`p_dt = λ_c/Λ`, `p_tt = λ_tt/Λ`, `p_dec = λ₀/Λ`. A d-t fusion is counted iff the episode ends in the
d-t branch (prob p_dt). The muon **returns to the pool** (renewal) iff it survives whichever reaction
fired: after a d-t fusion with s = (1−ω_s^eff), after a tt fusion with (1−ω_tt). So the per-episode
return probability is `r = p_dt·(1−ω_s^eff) + p_tt·(1−ω_tt)`, and

```
X_μ = Σ_{k≥1} r^{k−1} p_dt = p_dt / (1 − r)
    = λ_c / ( Λ − λ_c(1−ω_s^eff) − λ_tt(1−ω_tt) )
    = λ_c / ( λ_c·ω_s^eff + λ_tt·ω_tt + λ₀ )
    = 1 / ( ω_s^eff + ω_tt·(λ_tt/λ_c) + λ₀/λ_c ).                         (v2)
```

The extra per-cycle loss term **`ω_tt·(λ_tt/λ_c)`** is exactly the `tt_pc` share used in the
re-attribution refit (`accounting.md`): introducing it forces the fitted ω_s^eff (= ω_s0(1−R)) down so the
anchor total still reproduces the measured 0.45 %. With λ_tt→0 this collapses to §4's v1 identity. The
implementation is `analytic.fusions_per_muon_v2(omega_s_eff, lambda_c, lambda_0, tt_loss_rate, omega_tt)`,
gated against the ODE (`cycle.py` channels-on, single-pool limit) to **<1 %** (G-N2; measured worst
0.0 % over the 3-point (λ_tt, ω_tt) grid).

**Documented asymmetry (³He scavenging omitted from the closed form).** The ³He channel removes the muon
from the **dμ** pool (rate λ_He = λ_d³He·φ·c_He), before it ever reaches the tμ cycling pool. The §4
closed form has *already collapsed* the dμ→tμ isotopic structure into the single pool `m(t)`, so a
dμ-only hazard has no clean single-pool representative and is **not** included in `fusions_per_muon_v2`.
³He scavenging is available only in the full ODE (`cycle.py`, `include_loss_channels=True` with `c_he>0`),
where the dμ pool is an explicit state. This asymmetry is intentional and is the reason the channels-on
scoreboard/refit (§WS-N) drive the ttμ re-attribution through the closed form but keep ³He an ODE-only,
burn-time-scenario channel.

---

## 5. Constructing the effective cycle rate λ_c

The cycle is an (approximately) serial chain; the effective rate is the harmonic-style combination of the
rate-limiting serial steps (transfer → spin-flip → resonant formation; fusion is instantaneous):

```
1/λ_c ≈ 1/λ_dt + (hyperfine-weighted) 1/λ_form + 0(fusion),
λ_form = effective tμ→dtμ formation rate = Σ_F w_F · λ_dtμ^F(T,φ),   w_F = thermal/kinetic F-population.
```

At the densities/temperatures of interest, **resonant formation λ_dtμ dominates** the cycle time, so
λ_c ≈ λ_form to leading order. The temperature dependence of λ_c (the headline Yamashita–Kino curve) comes
almost entirely from λ_dtμ(T): the Vesman resonance is thermally accessed, so λ_c **rises with T** to ~800 K.
`λ_dtμ^F(T,φ)` is supplied by the Phase-1 interpolant `rates.λ_dtμ(T,φ,F)`; the ODE network (§3) realizes the
serial combination automatically, while §4's λ_c is the lumped effective value used analytically.

---

## 6. Energy balance

```
Q = X_μ · E_f · η_conv / E_μ,
```
- E_f = 17.6 MeV, E_μ ≈ 5 GeV (knob; range 2–10 GeV), η_conv = explicit chain (η_acc·η_thermal·η_recirc·…,
  optional blanket multiplier M for a hybrid). Scientific breakeven (Q=1, η_conv=1): X_μ = E_μ/E_f ≈ 284.

> **Scope & intended use.** `M` is a transparent accounting multiplier for a fusion–fission *hybrid blanket*,
> not a breeding design — OpenMuCF models neutron economics only. Below electrical breakeven, the defensible
> near-term value of μCF is as a *neutron / medical-isotope source* (e.g. Ac-225), and any application should be
> framed that way — never as a fissile-material (Pu-239) breeding pathway, which this project does not model or endorse.

`energy.py` exposes η_conv as a transparent product so judges can see exactly what each factor assumes.

---

## 7. PRE-REGISTERED validation targets + tolerance bands (Phase 3 GATE)

Registered **now**, before fitting anything. A target "passes" only if met OR the discrepancy is documented
and physically explained.

| # | Target | Source | Tolerance band |
|---|---|---|---|
| V1 | analytic X_μ == ODE N_fus(∞) in single-pool limit | self-consistency | < 1 % (numerical) |
| V2 | X_μ ∈ [100, 150] at φ≈1.2, ω_s^eff≈0.45–0.5 %, λ_c≈10⁸ s⁻¹ | Breunlich 1989; record ~150 | within [80, 160] |
| V3 | λ_c(T) rises monotonically 20→800 K; λ_c(800)/λ_c(300) ratio | Yamashita–Kino 2022 (graphical) | shape monotone; ratio within ±30 % of digitized |
| V4 | Fed Kou–Chen ω_s^eff inputs, reproduce N_fus,μ 112.6 and 156.5 | Kou–Chen 2606.07077 | within ±10 % |
| V5 | Q ≈ 0.5 at X_μ≈150 (stated η_conv); breakeven X_μ | derived | breakeven within 1 % of 284 |

**Known reproduction risk (pre-registered, honest):** Yamashita–Kino results are graphical with
underspecified rate inputs, so V3 is a *tolerance-band* benchmark, not exact reproduction. If V2/V3 miss, we
publish a discrepancy analysis (likely cause: our lumped λ_c vs their full EVM-SPM-FIF side-paths), not a fudge.

---

## 8. Scope: in v1 vs deferred to v2

**In v1:** thermalized atoms; tμ hyperfine (F=0,1); isotopic transfer; resonant formation λ_dtμ(T,φ,F);
fast-fusion limit; reactivation folded into ω_s^eff = ω_s⁰(1−R); energy balance; full autodiff + UQ.

**Deferred to v2 (named, not silently dropped), with computed materiality at (300 K, 1.2 φ, c_t=0.5):**

| deferred channel | est. effect on X_μ | note |
|---|---|---|
| per-cycle d-recapture + q_1s (recycled muons re-enter via dμ) | −6% to −14% (q_1s = 0.4–1.0) | v1 recycles straight to tμ (§3); the formation-scale anchor (`formation._CALIB`) absorbs much of this at 300 K; q_1s (contested cascade ground-state fraction) is ~free for v1 X_μ but multiplies this leg |
| ttμ side cycle (μ loss to tt fusions, ω_tt ≈ 0.14 measured) | −5% (optimistic ω_tt) to −15% (measured) | the DOMINANT deferred side-cycle loss at c_t≈0.5; one-sided downward — would consume most of the ±10% V4 margin; tt neutrons (≤9.4 MeV continuum) are also the real neutron-count contaminant, not dd |
| ddμ branch / d-d cycle | ~−0.1% | one-time ~1% dμ branch; negligible for X_μ at c_t=0.5 (matters for interpreting d-d datasets, e.g. Toyama ddμ*/MuFusE DD runs — a v2 goal) |
| ³He scavenging (t→³He decay, ~0.47%/month; λ_dHe ~ 1.9e8 s⁻¹) + initial He purity + fusion ⁴He | ~−2% (3-day fuel) to ~−14% (30-day, gas/liquid; illustrative — solid targets shed ³He and experiments purge it) | validation anchors use He-purged fuel, so v1 passes are unaffected; a one-line absorbing channel λ_He·φ·c_He(t) is the v2 fix |
| epithermal/non-thermal distributions; in-flight μCF; SPM/FIF side-paths; dμ-hyperfine; energy-resolved reactivation R_X(E) | folded into the calibrated formation scale / η band | as before (Yamashita–Kino side-paths; Kou–Chen R_X(E)) |
| x-ray observables (state-resolved ddμ*/dtμ* populations → 1.6–2.0 keV line intensities) | n/a (new observable) | the J-PARC/TES-facing interop; requires the d-d branch first |

Each is a clean extension of the §3 linear network. v1's honest claim is bounded accordingly: the v1 network
is a reduced effective cycle whose yield-level numbers carry ~±10–15% structural headroom (one-sided downward
for the side-cycle/recapture items); the FINDINGS headline results are insulated from all of this because they
run on the closed form with the measured λ_c band (see FINDINGS.md caveats).
