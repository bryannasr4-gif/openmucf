"""Generate (or audit) DESIGN.md + DESIGN_MANIFEST.json -- the Bayesian experimental-design ranking.

    python scripts/generate_design.py            # regenerate DESIGN.md + DESIGN_MANIFEST.json (runs NUTS)
    python scripts/generate_design.py --audit    # re-run with pinned seeds; check committed numbers

Content (openmucf.design): rank candidate NEXT muCF experiments (C1 neutron disappearance slope @ phi=2.0;
C2 lambda_c(T) @ 800 K; C3 omega_s^eff @ phi=2.4 under scenario A AND B; C4 X-ray/neutron ratio) by the
PRIMARY preposterior sd-contraction metric and the SECONDARY nested-MC EIG, over the EXISTING weak-prior
calibrate posterior. R is reported class-conditionally (constant-R vs R(phi)-inflated); the class
CONTRAST -- resolved or not, against its own Monte-Carlo error -- is the finding.

Reproducibility (the CALIBRATION.md precedent, NOT the byte-diff pattern): DESIGN.md +
DESIGN_MANIFEST.json carry numpyro/NUTS-derived numbers that are NOT byte-stable cross-architecture, so
`make audit` byte-diffs NEITHER. Instead `--audit` re-runs with the pinned seeds and checks every
manifest-tracked number against a band DERIVED from that cell's OWN published Monte-Carlo standard error
-- K_SIGMA * sqrt(se_committed^2 + se_fresh^2), floored at AUDIT_ATOL_FLOOR -- never against a fixed
pre-registered constant. Both audited quantity classes use that one construction; they differ only in how
the SE is measured:
  * sd-contraction ratios (dimensionless, and legitimately passing through zero -- the estimand-discipline
    cells collapse to ~0 -- so a relative tolerance is ill-posed here): SE = the per-refit nonparametric
    bootstrap, combined in quadrature with the base chain's own posterior-sd error.
  * EIG-in-bits: SE measured IN-RUN as the spread over AUDIT_EIG_REPLICATES replicate base chains.
Every band is therefore itself auditable, and re-sizes automatically if the sampler settings change. See
the two AMENDMENTs below for what each band replaced and why.

AMENDMENT 2026-08-09 (cross-architecture reproduction, arm64 vs x86-64). The contraction band used to be
a fixed 3 pp absolute, justified by a "~+/-3 pp Monte-Carlo floor" quoted from docs/xray_feasibility.md.
That floor was not measured for THESE cells and is too small: at n_synth=8 the per-refit contraction
spread is 4-18 pp, so the reported medians carried a Monte-Carlo SE of 1.7-6.4 pp -- i.e. the band was
tighter than the estimator's own noise for 7 of 12 cells. A band below the estimator's noise cannot be
satisfied by an independent reproduction; it can only be satisfied by regenerating the identical
pseudo-random realization. It duly passed on every x86-64 host and failed on 5 of 12 cells the first time
a different architecture ran it (Apple Silicon, 2026-07-23: worst |delta| 0.1416 = 472% of the band, and
one sign flip), while a 20-seed sweep on that host reproduced the same dispersion from seed variation
ALONE -- so the failure was the band, not the architecture. Two changes follow, both pre-registered here:
  (a) n_synth 8 -> 64 (openmucf.design.N_SYNTH_DEFAULT), cutting the worst-cell SE ~3.3x;
  (b) the band is DERIVED per cell as K_SIGMA * sqrt(se_committed^2 + se_fresh^2) (floored at
      AUDIT_ATOL_FLOOR so a fluke-small SE cannot make it vacuous-tight), with each cell's SE published
      in DESIGN.md and tracked in the manifest -- so the tolerance is itself auditable, and can never
      again be tighter than the noise of the number it is checking.
The SE that sets the band was VALIDATED against an independent measurement rather than assumed: feeding
the arm64 run's own 8 per-refit contractions to `openmucf.design.median_se` reproduces the dispersion that
20 independent analysis seeds showed in the reported median, to a mean ratio of 1.03 (per-cell 0.39-1.24;
the low outlier is an n=8 bootstrap on a sample with one extreme value -- another reason n_synth=64). A
band built from a run's self-reported SE is therefore honestly sized, not self-serving.
Structural claims the doc actually makes (the C4 recommendation, the rankings, class-independence) are
additionally gated in `audit()` at their own separations, because those -- not 3-decimal noise cells --
are the deliverable.
Both are hard-failing and pinned by tests/test_design_audit.py (an audit tolerance may never be
softened silently). Doc<->manifest consistency is enforced by `provenance --check` (regenerated together).

AMENDMENT 2026-08-12 (the EIG band, measured rather than assumed). The EIG-in-bits cells kept a
pre-registered 5% RELATIVE band annotated "held cross-arch 2026-07-23". That annotation is DELETED, not
repaired: it is unsourceable -- the artifacts of that reproduction were never tracked in this repository --
and the substance is wrong anyway. A 200-realization sweep over the BASE CHAIN's seed (analysis seed held
fixed) measures the EIG cells' realization noise at 0.042-0.068 bits per cell, i.e. 1.40%-6.10% relative:
the noise is ABSOLUTE, and its relative size varies 4.4x across cells while its absolute size varies only
1.6x. A single relative constant is therefore the wrong SHAPE, not merely the wrong value. Against an
independent base-chain realization the 5% band reds 49.5% of runs (worst cell eig_C3, whose own noise --
6.10% -- exceeds the entire band); the smallest relative constant that would control that rate is >= 34.5%,
set by eig_C3, and it makes eig_C4's band 24.8 sigma of ITS own noise, i.e. unfalsifiable. The single
arm64 EIG flag of 2026-07-23 (|delta| = 0.088 bits on eig_C2) sits at the 92nd percentile of that
seed-perturbation distribution, z = +1.48 -- an ordinary draw of an ordinary quantity, and another base
chain on the x86-64 dev host reproduced essentially the same value.
The EIG cells therefore move to the SAME construction the contraction cells received on 2026-08-09, with
the SE MEASURED IN-RUN: AUDIT_EIG_REPLICATES independent base chains per pass (analysis seed unchanged),
sd with ddof=1 over each cell's replicate values, published as se_<cell> in DESIGN.md and tracked in the
manifest. Simulated false alarm 0.555% per run (20k replicates, both values AND both SEs resampled -- the
conservative both-sides-fresh convention), inside the 0.12-0.60%/run regime this file already chose for
the contraction cells at 4 sigma; the superseded 5% constant scores 73.4% per run on that same convention.
A fixed ABSOLUTE band (K*sqrt(2)*max-cell sigma = 0.383 bits) was measured and REJECTED: 0.010%/run, but
~10x more conservative than this file's own 4-sigma design point (8.5 sigma on eig_C4), detection power at
a 0.30-bit shift 7.7-18.6% against 22.6-84.3% for the measured band, and it is once more a constant that
goes stale the moment n_outer, n_inner or the chain length changes -- the failure mode the 2026-08-09
amendment exists to prevent. An in-run measurement re-sizes itself instead.
`zero_eig` is unaffected in practice: the replicate candidate's observable is constant, so its EIG is
identically zero and its replicate SE is numerically zero (<= 1e-17 across 200 realizations), leaving the
AUDIT_ATOL_FLOOR absolute floor it already had. One general rule follows, applied uniformly to every
SE-banded cell: the SE-ratio guard is enforced only where the band is SE-GOVERNED, since the ratio of two
numerically-zero SEs is meaningless. It is verified live that all 12 contraction cells remain SE-governed,
so this changes nothing for them.

Importable without side effects (all work is inside functions / main()); the tests import the pure
formatting/registry/audit helpers without running the NUTS refits.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from openmucf import design
from openmucf.provenance import ManifestEntry, file_sha256, write_manifest
from openmucf.rates import RATES_CSV

DESIGN_MD = "DESIGN.md"
DESIGN_MANIFEST = "DESIGN_MANIFEST.json"

# --- the C4 conditional input (public-sourced: docs/xray_feasibility.md) ------------------------------
# The X-ray/neutron feasibility study reports a best-cell (w=0.10, sigma_rel=0.02) posterior sd(R)
# contraction of 42.95% in the weak-prior (degeneracy-exposing) chain. 42.95% >= the 15% threshold, so
# candidate C4 (X-ray/neutron ratio) is INCLUDED. The value is passed to design.registry() -- the
# conditional is structural (a function of this number), exercised both ways by the tests.
XRAY_VERDICT_PCT = 42.95
XRAY_THRESHOLD_PCT = 15.0

# --- audit tolerances (pinned by tests/test_design_audit.py; hard-failing; never widen silently) ------
# Replicate base chains used to MEASURE each EIG-family cell's Monte-Carlo SE in-run (2026-08-12; see the
# AMENDMENT above). Base seeds 1..AUDIT_EIG_REPLICATES -- the committed seed stays out of its own SE --
# with the analysis seed unchanged. Sized from a 200-realization base-chain-seed sweep (2026-08-10): the
# EIG realization noise is ABSOLUTE (per-cell sigma 0.042-0.068 bits = 1.4%-6.1% relative), so the
# superseded 5%-relative constant reds ~50% of runs against an independent realization. Measured false
# alarm of THIS construction ~0.6%/run at R=20 (R=8/12 give 1.3%/0.8%, outside the 0.12-0.60%/run regime
# chosen for the contraction cells; R=32 buys ~0.15 pp for 60% more chains). Cost ~+20 s per pass: one
# base chain is ~1.0 s and the six EIG cells over it are ~3 ms.
AUDIT_EIG_REPLICATES = 20
AUDIT_ATOL_FLOOR = 0.01    # band floor: never tighter than 1 pp, whatever the measured SE says
# The DETECTION band and the CLAIM threshold are deliberately different numbers, because they trade off in
# opposite directions and were conflated in the superseded design:
#   * AUDIT_K_SIGMA sizes a gate that runs on EVERY push, on 12 cells, and must not cry wolf. Both sides
#     of the band are ESTIMATED SEs, so the nominal normal-tail rate understates the truth; MEASURED by
#     simulation (20k replicates, n=64, normal / lognormal / t3 samples, both SEs bootstrapped exactly as
#     here): 3 sigma gives ~0.35% per cell = 4.1-4.6% per 12-cell run, and 4 sigma gives ~0.01-0.05% per
#     cell = 0.12-0.60% per run -- against nominal 3.2% and 0.076%. A ~4% per-run false-alarm rate is
#     near-certain to fire spuriously within a few dozen CI runs, which is how a blocking gate gets
#     ignored or disabled; ~0.1-0.6% is not. That is the reason for 4 sigma. (The superseded version of
#     this comment claimed 0.08% per run and said the margin "absorbs" SE-estimation noise so that an
#     under-estimated SE "cannot by itself turn the gate red" -- backwards: SE-estimation noise is
#     precisely what inflates the rate 1.5-8x above nominal. The choice survives; its justification did not.)
#   * CLAIM_K_SIGMA is the bar for DESIGN.md to state a finding as resolved. Here the conservative
#     direction is the opposite one -- a lower bar over-claims -- so it stays at the conventional 3 sigma.
AUDIT_K_SIGMA = 4.0                # sd-contraction band = K sigma of the cell's OWN Monte-Carlo SE
CLAIM_K_SIGMA = 3.0                # a contrast/cell is only reported as RESOLVED above this
AUDIT_MIN_SEPARATION_SIGMA = 3.0   # C4's R-contraction lead over the runner-up, in pooled SE units
# Half the band comes from the FRESH run's SE, which is not published anywhere -- so without this gate a
# noisier platform would silently award itself a wider band. The bootstrap SE's own relative sd at
# n_synth=64 is ~25% (measured), so a 3x divergence in either direction is >5 sigma of estimator noise:
# it means the estimator, not the estimate, changed. Symmetric, because a fresh SE 3x TIGHTER than the
# committed one is equally a signal that the two runs are not measuring the same thing.
AUDIT_SE_RATIO_MAX = 3.0

CANDIDATE_ORDER = ("C1", "C2", "C3", "C4")


# ============================================================================ computation
def eig_family(cand_ids, samples: dict, seed: int) -> dict:
    """The six EIG-family cells for ONE base-posterior realization, keyed by manifest entry id.

    Shared by the committed pass and by the replicate-SE loop below, so the set of cells the band is
    measured over can never drift from the set of cells the band is applied to.
    """
    out = {f"eig_{cid}": design.eig_nested_mc(cid, samples=samples, seed=seed) for cid in cand_ids}
    out["eig_C3_inflated"] = design.eig_nested_mc("C3", cls="inflated", samples=samples, seed=seed)
    out["zero_eig"] = design.eig_nested_mc(design.replicate_candidate(), samples=samples, seed=seed)
    return out


def eig_replicate_se(cand_ids, analysis_seed: int) -> dict:
    """Measure each EIG-family cell's Monte-Carlo SE over AUDIT_EIG_REPLICATES replicate base chains.

    The quantity that moves between two honest reproductions of this document is the BASE CHAIN's
    realization: the nested-MC draw indices are generated from the analysis seed and are bit-identical
    everywhere, while the posterior the EIG integrates over is a fresh NUTS realization on every new
    platform, runner image or pin set. So the replicates perturb the base seed (1..R -- the committed
    seed 0 is deliberately excluded, an estimate must not be inside its own error bar) and hold the
    analysis seed fixed, which is exactly the sweep the band was sized from. Sequential, one chain at a
    time; ddof=1 because these R values are a sample, not the population.
    """
    reps: dict[str, list[float]] = {}
    for r in range(1, AUDIT_EIG_REPLICATES + 1):
        rep_samples = design.base_posterior(seed=r)
        for key, res in eig_family(cand_ids, rep_samples, analysis_seed).items():
            reps.setdefault(key, []).append(res["eig_bits"])
    return {key: float(np.std(vals, ddof=1)) for key, vals in reps.items()}


def compute(seed: int = 0) -> dict:
    """Run the full design analysis once (NUTS-heavy). Returns raw numbers keyed for build_headline."""
    samples = design.base_posterior(seed=seed)
    reg = design.registry(XRAY_VERDICT_PCT, XRAY_THRESHOLD_PCT)
    cand_ids = [c for c in CANDIDATE_ORDER if c in reg["candidates"]]

    fam = eig_family(cand_ids, samples, seed)
    eig = {cid: fam[f"eig_{cid}"] for cid in cand_ids}
    eig_c3_inflated = fam["eig_C3_inflated"]
    sdc = {cid: design.sd_contraction(cid, samples=samples, seed=seed) for cid in cand_ids}

    zero_eig = fam["zero_eig"]
    sobol = design.sobol_consistency(seed=seed)
    # Last, because it is the only part that draws chains other than the committed one: keeping it after
    # every committed cell is computed makes it structurally impossible for a replicate to leak into one.
    eig_se = eig_replicate_se(cand_ids, seed)

    return {
        "seed": seed,
        "registry": reg,
        "cand_ids": cand_ids,
        "eig": eig,
        "eig_c3_inflated": eig_c3_inflated,
        "eig_se": eig_se,
        "sdc": sdc,
        "zero_eig_bits": zero_eig["eig_bits"],
        "sobol": sobol,
        "settings": {
            "n_outer": eig[cand_ids[0]]["n_outer"],
            "n_inner": eig[cand_ids[0]]["n_inner"],
            "n_synth": sdc[cand_ids[0]]["n_synth"],
            "num_warmup": design.NUM_WARMUP,
            "num_samples": design.NUM_SAMPLES,
        },
    }


# ============================================================================ headline formatting
def _rank(scores: dict) -> list[str]:
    """Candidate ids ordered by DESCENDING score (ties broken by CANDIDATE_ORDER for determinism)."""
    return sorted(scores, key=lambda c: (-scores[c], CANDIDATE_ORDER.index(c)))


def build_headline(res: dict) -> tuple[dict, dict]:
    """Single source of truth: return (H, RAW). H maps id -> formatted string (for DESIGN.md + manifest
    provenance); RAW maps id -> float (for the --audit tolerance comparison). Also computes rankings and
    the class-contrast finding, resolved or not (put in H as strings)."""
    H: dict[str, str] = {}
    RAW: dict[str, float] = {}
    cand_ids = res["cand_ids"]

    for cid in cand_ids:
        sdc = res["sdc"][cid]
        RAW[f"eig_{cid}"] = res["eig"][cid]["eig_bits"]
        RAW[f"ose_{cid}"] = sdc["ose_contraction"]
        RAW[f"Rc_{cid}"] = sdc["R_contraction"]["constant"]
        RAW[f"Ri_{cid}"] = sdc["R_contraction"]["inflated"]
        # Per-cell Monte-Carlo SE: published, manifest-tracked, and the basis of the audit band.
        RAW[f"se_ose_{cid}"] = sdc["ose_contraction_se"]
        RAW[f"se_Rc_{cid}"] = sdc["R_contraction_se"]["constant"]
        RAW[f"se_Ri_{cid}"] = sdc["R_contraction_se"]["inflated"]
    RAW["eig_C3_inflated"] = res["eig_c3_inflated"]["eig_bits"]
    RAW["zero_eig"] = res["zero_eig_bits"]
    # Per-cell Monte-Carlo SE of every EIG-family cell, measured in-run over replicate base chains:
    # published, manifest-tracked, and the basis of that cell's audit band (2026-08-12 amendment).
    for key, se in res["eig_se"].items():
        RAW[f"se_{key}"] = se

    for k, v in RAW.items():
        s = f"{v:.3f}"
        H[k] = "0.000" if s == "-0.000" else s  # normalise negative zero (e.g. the ~0 zero-EIG cell)

    # rankings
    ose_rank = _rank({c: RAW[f"ose_{c}"] for c in cand_ids})
    eig_rank = _rank({c: RAW[f"eig_{c}"] for c in cand_ids})
    rc_rank = _rank({c: RAW[f"Rc_{c}"] for c in cand_ids})
    ri_rank = _rank({c: RAW[f"Ri_{c}"] for c in cand_ids})
    H["ose_rank"] = " > ".join(ose_rank)
    H["eig_rank"] = " > ".join(eig_rank)
    H["Rc_rank"] = " > ".join(rc_rank)
    H["Ri_rank"] = " > ".join(ri_rank)
    H["ose_eig_agree"] = "AGREE" if ose_rank == eig_rank else "DISAGREE"
    H["sobol_top"] = res["sobol"]["top_param"]
    H["c4_status"] = "included" if res["registry"]["c4_included"] else "dropped"

    # ---- resolution-aware findings (2026-08-09 amendment) -------------------------------------------
    # Everything below is DERIVED from the run's own Monte-Carlo SEs, never hard-coded, so the doc cannot
    # state a separation the data do not support. `sigma` = |estimate| / SE of that estimate.
    def _sig(value: float, se: float) -> float:
        return abs(value) / se if se and se == se and se > 0 else float("nan")

    # C4's lead on R-contraction (the recommendation) over the best non-C4 candidate, in pooled SE units.
    lead = {}
    for cls, key in (("constant", "Rc"), ("inflated", "Ri")):
        others = {c: RAW[f"{key}_{c}"] for c in cand_ids if c != "C4"}
        runner = max(others, key=others.get)
        gap = RAW[f"{key}_C4"] - others[runner]
        pooled = (RAW[f"se_{key}_C4"] ** 2 + RAW[f"se_{key}_{runner}"] ** 2) ** 0.5
        lead[cls] = {"runner": runner, "gap": gap, "sigma": _sig(gap, pooled)}
    # Both runner names are emitted: they happen to coincide on the shipped run, but a single name used
    # for both classes would silently mislabel the moment they diverge.
    H["C4_lead_runner_c"] = lead["constant"]["runner"]
    H["C4_lead_runner_i"] = lead["inflated"]["runner"]
    H["C4_lead_sigma_c"] = f"{lead['constant']['sigma']:.1f}"
    H["C4_lead_sigma_i"] = f"{lead['inflated']['sigma']:.1f}"

    # Class-flip: the PAIRED constant-minus-inflated difference and its own SE, per class-sensitive
    # candidate. A "flip" is claimed only when that difference is resolved at CLAIM_K_SIGMA.
    flips = {}
    for cid in cand_ids:
        d = res["sdc"][cid]["R_contraction_class_delta"]
        if not res["sdc"][cid]["class_sensitive"]:
            continue
        flips[cid] = {"delta": d["value"], "se": d["se"], "sigma": _sig(d["value"], d["se"])}
        H[f"delta_{cid}"] = f"{d['value']:+.3f}"
        H[f"se_delta_{cid}"] = f"{d['se']:.3f}"
        H[f"sigma_delta_{cid}"] = f"{flips[cid]['sigma']:.1f}"
    resolved = [c for c, f in flips.items() if f["sigma"] >= CLAIM_K_SIGMA]
    H["class_flip"] = ("RESOLVED for " + ", ".join(sorted(resolved))) if resolved else "NOT RESOLVED"
    H["class_flip_candidates"] = ", ".join(sorted(flips))
    H["class_delta_table"] = ", ".join(
        f"{c} {H[f'delta_{c}']} +- {H[f'se_delta_{c}']} ({H[f'sigma_delta_{c}']} sigma)"
        for c in sorted(flips)
    )

    # The interpretive sentence is DERIVED, never hard-coded: an earlier revision asserted a C1 collapse
    # "far larger than the noise floor" from a single realization whose separation was ~1.7 sigma of that
    # realization's own spread, and it did not survive an independent reproduction. Both branches below
    # support the same estimand-discipline conclusion; only one of them is licensed by any given run.
    #
    # AMENDMENT 2026-08-10 (merge gate). Both branches still ended in a HARD-CODED tail, and both tails
    # were wrong:
    #   * the unresolved branch closed with "the neutron-only candidates deliver no R information that
    #     survives its own Monte-Carlo error under EITHER structural class" -- a universal quantifier this
    #     run's own numbers refute (C3 is neutron-only and resolves from zero; it is absent from
    #     `unresolved_cells` in the same document, so the file contradicted itself);
    #   * the resolved branch closed with "the apparent constant-R information ... is an artifact of the
    #     assumed R(phi) form", which only parses when the contraction FALLS under inflation. `_dir()`
    #     already handled both signs, but the conclusion after it did not -- and the live C3 contrast is
    #     NEGATIVE at 2.8 sigma, i.e. 0.2 sigma from printing an explanation of the opposite phenomenon.
    # Every clause below is now derived from the cells, in both branches and for both signs.
    cands = res["registry"]["candidates"]

    def _resolved_from_zero(cid: str) -> bool:
        return (_sig(RAW[f"Rc_{cid}"], RAW[f"se_Rc_{cid}"]) >= CLAIM_K_SIGMA
                or _sig(RAW[f"Ri_{cid}"], RAW[f"se_Ri_{cid}"]) >= CLAIM_K_SIGMA)

    def _dir(cid: str) -> str:
        return "collapses under R(phi)-inflation" if flips[cid]["delta"] > 0 else "RISES under R(phi)-inflation"

    if resolved:
        H["class_flip_reading"] = (
            "The contrast is RESOLVED at >= {k:.0f} sigma for {names}: {detail}. The direction differs "
            "between candidates but the reading does not: an R contraction that MOVES with the assumed "
            "R(phi) form is a property of that assumption, not of the measurement, in either "
            "direction.".format(
                k=CLAIM_K_SIGMA, names=", ".join(sorted(resolved)),
                detail="; ".join(f"{c} {_dir(c)}" for c in sorted(resolved)))
        )
    else:
        # "neutron-only" is read from the registry (kind != xray_ratio), never hard-coded, so adding a
        # candidate cannot silently falsify the sentence.
        neutron_only = [c for c in cand_ids if cands[c].kind != "xray_ratio"]
        flat = [c for c in neutron_only if not _resolved_from_zero(c)]
        firm = [c for c in neutron_only if _resolved_from_zero(c)]
        clauses = []
        # Colon-list form throughout: the lists are of unknown length, and a generated sentence must not
        # depend on getting subject-verb agreement right for 1 vs many candidates.
        if flat:
            clauses.append(f"{', '.join(flat)} -- no R contraction distinguishable from zero under EITHER "
                           f"structural class")
        if firm:
            clauses.append(f"{', '.join(firm)} -- a nonzero R contraction does resolve, but "
                           f"class-conditionally, and with the class contrast itself unresolved this run "
                           f"cannot separate that information from the assumed R(phi) form")
        clean = [c for c in cand_ids if not res["sdc"][c]["class_sensitive"] and _resolved_from_zero(c)]
        H["class_flip_reading"] = (
            "NO class contrast is resolved at {k:.0f} sigma at this n_synth, so this document does NOT "
            "claim a measured collapse. What the run does support is narrower: {clauses}. {closer}".format(
                k=CLAIM_K_SIGMA,
                clauses="; ".join(clauses) if clauses else "nothing about the neutron-only candidates",
                closer=("Resolved at >= {k:.0f} sigma AND identical across both structural classes by "
                        "construction: {c} -- which is the recommendation this document makes."
                        .format(k=CLAIM_K_SIGMA, c=", ".join(clean)) if clean else
                        f"No candidate's R contraction is both resolved at >= {CLAIM_K_SIGMA:.0f} sigma "
                        f"and class-independent on this run."))
        )

    # Which R cells are resolvable from zero at all, at this n_synth?
    unresolved = [f"{k}_{c}" for c in cand_ids for k in ("Rc", "Ri")
                  if _sig(RAW[f"{k}_{c}"], RAW[f"se_{k}_{c}"]) < CLAIM_K_SIGMA]
    H["unresolved_cells"] = ", ".join(unresolved) if unresolved else "none"

    # The base-chain component of every published +-, surfaced so a reader can see which term dominates.
    # Which cell it dominates hardest is DERIVED, not asserted: an earlier revision hard-coded "on ose_C1
    # the base-chain term is ~5x the bootstrap term", which is a per-run measurement, not a constant.
    comp = res["sdc"][cand_ids[0]]["se_components"]["base_sd_mcse_rel"]
    H["base_mcse_ose"] = f"{comp['omega_s_eff_pct']:.2%}"
    H["base_mcse_R"] = f"{comp['R']:.2%}"
    ratios = {}
    for cid in cand_ids:
        boot = res["sdc"][cid]["se_components"]["boot"]
        ratios[f"ose_{cid}"] = abs(1 - RAW[f"ose_{cid}"]) * comp["omega_s_eff_pct"] / boot["ose"]
        for cls, key in (("constant", "Rc"), ("inflated", "Ri")):
            if cls in boot["R"]:
                ratios[f"{key}_{cid}"] = abs(1 - RAW[f"{key}_{cid}"]) * comp["R"] / boot["R"][cls]
    dom = max(ratios, key=ratios.get)
    H["base_dom_cell"] = dom
    H["base_dom_ratio"] = f"{ratios[dom]:.1f}"
    return H, RAW


# ============================================================================ DESIGN.md
def _sdc_row(H: dict, cid: str) -> str:
    """One PRIMARY-table row: every cell as ``value +- MC standard error`` (the audit band's basis)."""
    return (f"| {cid} | {H[f'ose_{cid}']} +- {H[f'se_ose_{cid}']} | "
            f"{H[f'Rc_{cid}']} +- {H[f'se_Rc_{cid}']} | {H[f'Ri_{cid}']} +- {H[f'se_Ri_{cid}']} |")


def _sdc_table(H: dict, cand_ids) -> str:
    return ("| candidate | omega_s^eff contraction | R contraction (constant-R) | "
            "R contraction (R(phi)-inflated) |\n"
            "|---|---|---|---|\n" + "\n".join(_sdc_row(H, cid) for cid in cand_ids))


def _eig_row(H: dict, label: str, entry_id: str) -> str:
    """One SECONDARY-table row: ``value +- MC standard error``, mirroring the PRIMARY table."""
    return f"| {label} | {H[entry_id]} +- {H[f'se_{entry_id}']} |"


def _eig_table(H: dict, cand_ids) -> str:
    rows = [_eig_row(H, cid, f"eig_{cid}") for cid in cand_ids]
    rows.append(_eig_row(H, "C3 (scenario-B, R(phi)-inflated)", "eig_C3_inflated"))
    return "| candidate | EIG [bits] |\n|---|---|\n" + "\n".join(rows)


def build_markdown(H: dict, res: dict) -> str:
    cand_ids = res["cand_ids"]
    s = res["settings"]
    reg = res["registry"]
    labels = {c: design._resolve(c) for c in cand_ids}
    reg_rows = "\n".join(
        f"| {c} | {labels[c].label} | {labels[c].design_point} | included |" for c in cand_ids
    )
    c4_line = (
        f"C4 (X-ray/neutron ratio) is **{H['c4_status']}**: the X-ray/neutron feasibility study "
        f"(`docs/xray_feasibility.md`) reports a best-cell (kappa-band w=0.10, sigma_rel=0.02) posterior "
        f"sd(R) contraction of **{reg['xray_verdict_pct']:.2f}%** in the weak-prior chain, which is "
        f">= the pre-registered **{reg['threshold_pct']:.0f}%** inclusion threshold. The conditional is "
        f"applied structurally in `openmucf.design.registry` (a pure function of that number)."
    )
    return f"""# DESIGN.md -- Bayesian experimental design: which next experiment sharpens (omega_s^eff, R)?
(auto-generated by `scripts/generate_design.py`)

> **INTERNAL DESIGN NOTE, NOT AN OUTBOUND ARTIFACT (I6).** This ranking is **never cold-mailed**. It
> attaches only to an ALREADY-WARM thread -- an existing exchange with a muCF laboratory or muon-source
> developer -- as a "here is how we would prioritise the next measurement" note. It is an internal planning
> instrument, not outreach.

> **Estimand discipline.** EIG on omega_s^eff at stated conditions is well-posed; EIG "on R" from
> neutron-only observables is generated by the ASSUMED structural form R(phi) -- we report it
> class-conditionally (constant-R vs R(phi)-inflated) and report the class contrast, RESOLVED OR NOT,
> as a finding.

**Method (no new physics).** Both metrics run over the EXISTING weak-prior calibration posterior
(`openmucf.calibrate`, Uniform omega_s0 prior -- the +0.84 omega_s0/R degeneracy ridge of
`CALIBRATION.md`, the same chain the X-ray verdict was decided on). A candidate is only an ADDED future
observable `y ~ Normal(mu(theta), sigma)` at a stated design point; nothing about the calibration data or
the forward map changes. The PRIMARY metric is the preposterior median posterior-sd contraction from
refitting with that observable appended (`openmucf.design.sd_contraction`); the SECONDARY metric is the
nested-Monte-Carlo Expected Information Gain in bits (`openmucf.design.eig_nested_mc`). Pinned settings:
n_outer={s['n_outer']}, n_inner={s['n_inner']} (EIG), n_synth={s['n_synth']} synthetic datasets, chains
num_warmup={s['num_warmup']}/num_samples={s['num_samples']}, seed={res['seed']}.

## Candidate registry
| candidate | observable | design point | status |
|---|---|---|---|
{reg_rows}

{c4_line}

## PRIMARY metric -- preposterior sd-contraction (median over {s['n_synth']} synthetic datasets)
{_sdc_table(H, cand_ids)}

Ranking by omega_s^eff contraction (the well-posed estimand): **{H['ose_rank']}**. Ranking by R
contraction under constant-R: **{H['Rc_rank']}**; under R(phi)-inflated: **{H['Ri_rank']}**.

**Read the +/- column before the ranking.** Each cell is a MEDIAN over {s['n_synth']} synthetic datasets
and the quoted +/- is that median's total Monte-Carlo standard error: the nonparametric bootstrap over
those datasets, combined in quadrature with the Monte-Carlo error of the shared denominator
`sd_before` -- the base chain's own posterior sd, which every contraction divides by and which a
bootstrap over the datasets cannot see (measured on this chain at {H['base_mcse_ose']} of `sd(omega_s^eff)`
and {H['base_mcse_R']} of `sd(R)`). Omitting that second term is what would make the band narrower than
the error it exists to absorb: on `{H['base_dom_cell']}` the base-chain term alone is
{H['base_dom_ratio']}x the bootstrap term.
Differences smaller than a few times these SEs are not findings. On this run the R cells NOT resolvable
from zero at {CLAIM_K_SIGMA:.0f} sigma are: **{H['unresolved_cells']}**.
This supersedes the "~+/-3 pp Monte-Carlo floor" quoted in earlier revisions of this file, which was
carried over from `docs/xray_feasibility.md` and was never measured for these cells; the true per-refit
spread is several times larger, and the `--audit` band is now derived from these SEs rather than from a
fixed constant (see the audit section).

**The recommended experiment depends on the estimand.** To break the omega_s0/R degeneracy (tighten R),
**C4 (X-ray/neutron ratio) wins decisively and ROBUSTLY across both structural classes** -- its R
contraction ({H['Rc_C4']} +- {H['se_Rc_C4']}) leads the runner-up by {H['C4_lead_sigma_c']} sigma under
constant-R (vs {H['C4_lead_runner_c']}) and by {H['C4_lead_sigma_i']} sigma under R(phi)-inflation
(vs {H['C4_lead_runner_i']}), and it is class-independent because it constrains omega_s0 DIRECTLY, not
through the R(phi) form. That
separation -- not the third decimal of any cell -- is the deliverable, and it is the one ordering the
`--audit` gate enforces structurally. To tighten omega_s^eff specifically, C3 (a direct high-density
re-measurement) leads.

**Class-conditional R: {H['class_flip']}.** The estimand-discipline note warns that a neutron-only
candidate's apparent R information is generated by the ASSUMED R(phi) form. The statistic that tests it is
the PAIRED per-dataset difference (constant-R minus R(phi)-inflated), which shares its theta*, its
standardized noise and its kappa draws between the two classes, so the class contrast is not swamped by
synthetic-dataset variation. Measured on this run ({H['class_flip_candidates']} are the class-sensitive
candidates): {H['class_delta_table']}. Note this is the median of the PER-DATASET differences, which is
not the difference of the two published per-class medians (medians do not subtract) -- subtracting the
table columns gives a different, untested number. {H['class_flip_reading']}
Independently of that contrast, C4's contraction ({H['Rc_C4']}) does not move between classes at all --
it constrains omega_s0 DIRECTLY -- so it remains the one candidate that identifies R without the
structural assumption, which is the recommendation this document actually makes.

## SECONDARY metric -- nested-Monte-Carlo EIG
{_eig_table(H, cand_ids)}

**The +/- here is the BASE-CHAIN realization error**, measured in-run rather than assumed: each cell is
re-evaluated over {AUDIT_EIG_REPLICATES} independent base chains (the analysis seed held fixed, so the
nested-MC draws are identical and only the posterior being integrated over changes), and the quoted +/- is
the sd of those {AUDIT_EIG_REPLICATES} values. That is the term that actually moves when this document is
reproduced on another machine -- the nested-MC draw indices do not -- and it is what sizes the `--audit`
band below. Differences between two runs smaller than a few times these SEs are not findings.

**Scenario-B disclaimer.** the scenario-B MuFusE EIG is large BY CONSTRUCTION (the widest prior wins);
this is a property of the prior, not of the experiment. C3's EIG rises from {H['eig_C3']} bits under
scenario A (constant-R) to {H['eig_C3_inflated']} bits under scenario B (R replaced by the wider
Uniform(0.15, 0.60) prior) purely because scenario B starts from a wider prior, so there is more entropy
to remove -- it is not evidence that the experiment is more informative.

**Nested-MC bias caveat.** the log-mean-exp marginal log-likelihood is NEGATIVELY biased by Jensen
(mean-of-logs < log-of-mean), so the reported EIG -- which subtracts it -- carries an O(1/n_inner)
POSITIVE bias: the bits are a slight over-estimate that shrinks with n_inner (={s['n_inner']} here); the
candidate RANKING -- the deliverable -- is robust to it, absolute bits are indicative only.

## sd-contraction vs EIG ranking
By omega_s^eff contraction: **{H['ose_rank']}**; by EIG: **{H['eig_rank']}** -- these **{H['ose_eig_agree']}**.
The two metrics answer different questions (total information vs contraction of a specific estimand), so
they need not agree; **where they disagree, sd_contraction (the estimand-specific metric) GOVERNS.**

## Sanity gates (all three are tests -- `tests/test_design.py`)
1. **zero-EIG for an exact-replicate measurement:** re-observing an already-pinned constant yields EIG =
   {H['zero_eig']} +- {H['se_zero_eig']} bits (identically zero: the replicate observable is constant, so
   this cell carries no realization noise at all and its audit band is the {AUDIT_ATOL_FLOOR} absolute floor).
2. **EIG monotone in stated precision:** a tighter measurement never lowers EIG (a 3-point sigma sweep).
3. **Sobol-consistency in the small-noise limit:** the parameter a tiny-sigma X_mu measurement informs
   most is **{H['sobol_top']}** -- the top Sobol driver of X_mu over the same `openmucf.uq` prior box.

## Reproducibility / audit
Every number here is NUTS/Monte-Carlo derived and reproduces to Monte-Carlo error, NOT byte-identically
(the `CALIBRATION.md` precedent). `make audit` byte-diffs NEITHER this file nor its manifest; instead
`python scripts/generate_design.py --audit` re-runs with the pinned seeds and checks:

1. **EIG bits** against a band DERIVED from the SEs published in the SECONDARY table above -- the same
   `{AUDIT_K_SIGMA:.0f} * sqrt(se_committed^2 + se_fresh^2)` construction as item 2, floored at
   {AUDIT_ATOL_FLOOR} absolute. Until 2026-08-12 these cells were checked against a fixed
   **5% RELATIVE** constant. That was the wrong SHAPE: a 200-realization sweep over the base chain's seed
   measures the per-cell realization noise at 0.042-0.068 bits (1.40%-6.10% relative), so the noise is
   ABSOLUTE -- its relative size varies 4.4x across cells while its absolute size varies 1.6x -- and the
   5% band would red **49.5% of runs** against an independent realization, worst on `eig_C3`, whose own
   noise exceeds the whole band. No relative constant fixes that: the >= 34.5% needed to cover `eig_C3`
   makes `eig_C4`'s band 24.8 sigma of its own noise. The measured band is 0.555%/run instead, and it
   re-sizes itself if the sampler settings change. See the 2026-08-12 AMENDMENT in
   `scripts/generate_design.py`.
2. **sd-contraction cells** against a band DERIVED from the numbers themselves --
   `{AUDIT_K_SIGMA:.0f} * sqrt(se_committed^2 + se_fresh^2)`, floored at {AUDIT_ATOL_FLOOR} absolute --
   using the per-cell Monte-Carlo SEs published in the PRIMARY table above and tracked in the manifest.
   The committed half of the band is therefore published; the fresh half is cross-checked against it
   (a fresh SE more than {AUDIT_SE_RATIO_MAX:.0f}x the committed one, in either direction, is itself a
   failure, so the band cannot quietly inflate on one platform). Two honest limits, stated rather than
   papered over: the band is a CONSERVATIVE proxy, not an exact `sd(committed - fresh)` -- the two runs
   share their pinned synthetic-dataset draws, so adding the two SEs in quadrature over-counts the dataset
   term -- and because a band scales with a cell's own noise, a genuinely noisy cell gets a wide one. Any
   cell whose band exceeds its own value is listed as NOT INFORMATIVE in the audit output rather than
   being silently counted as a pass.
3. **The structural claims** this document actually makes: C4 top-ranked on R contraction under BOTH
   classes and leading the runner-up by >= {AUDIT_MIN_SEPARATION_SIGMA:.0f} sigma; the omega_s^eff ranking
   top-2; C4/C2 class-independence exact; and the categorical Sobol top-parameter gate.

Until 2026-08-09 the contraction cells were checked against a FIXED 3 pp absolute band. That band was
smaller than the estimator's own Monte-Carlo error, so it could only be met by regenerating the identical
pseudo-random realization -- it passed on every x86-64 host and broke on the first genuinely independent
(arm64) reproduction. See the AMENDMENT block in `scripts/generate_design.py`. Regenerate with
`python scripts/generate_design.py`.
"""


# ============================================================================ manifest
def build_manifest_entries(H: dict, cand_ids) -> list[ManifestEntry]:
    def _e(entry_id, pattern):
        return ManifestEntry(id=entry_id, value=H[entry_id], pattern=pattern,
                             source_type="derivation", source="scripts/generate_design.py", doc="DESIGN.md")

    entries: list[ManifestEntry] = []
    # sd-contraction cells + their Monte-Carlo SEs (both live in the PRIMARY table row:
    # | Cx | ose +- se | Rc +- se | Ri +- se |). Tracking the SE makes the audit BAND auditable too.
    for cid in cand_ids:
        row = re.escape(_sdc_row(H, cid))
        for key in ("ose", "Rc", "Ri"):
            entries.append(_e(f"{key}_{cid}", row))
            entries.append(_e(f"se_{key}_{cid}", row))
    # EIG cells + their replicate-measured Monte-Carlo SEs (both live in the SECONDARY table row:
    # | Cx | eig +- se |), tracked for the same reason: the SE is what sets that cell's audit band.
    for cid, label in [(f"eig_{c}", c) for c in cand_ids] + [
            ("eig_C3_inflated", "C3 (scenario-B, R(phi)-inflated)")]:
        row = re.escape(_eig_row(H, label, cid))
        entries.append(_e(cid, row))
        entries.append(_e(f"se_{cid}", row))
    # sanity-gate headline claims
    zero_row = rf"yields EIG =\s*{re.escape(H['zero_eig'])} \+\- {re.escape(H['se_zero_eig'])} bits"
    entries.append(_e("zero_eig", zero_row))
    entries.append(_e("se_zero_eig", zero_row))
    entries.append(_e("sobol_top", rf"informs\s+most is \*\*{re.escape(H['sobol_top'])}\*\*"))
    return entries


def _manifest_inputs() -> dict:
    return {
        "rates_csv_sha256": file_sha256(RATES_CSV),
        "base_prior": list(design.BASE_PRIOR),
        "num_warmup": design.NUM_WARMUP,
        "num_samples": design.NUM_SAMPLES,
        "xray_verdict_pct": XRAY_VERDICT_PCT,
        "xray_threshold_pct": XRAY_THRESHOLD_PCT,
        "audit_eig_replicates": AUDIT_EIG_REPLICATES,
        "audit_k_sigma": AUDIT_K_SIGMA,
        "claim_k_sigma": CLAIM_K_SIGMA,
        "audit_atol_floor": AUDIT_ATOL_FLOOR,
        "audit_min_separation_sigma": AUDIT_MIN_SEPARATION_SIGMA,
        "n_synth": design.N_SYNTH_DEFAULT,
        "se_bootstrap": design.SE_BOOTSTRAP,
        "se_bootstrap_seed": design.SE_BOOTSTRAP_SEED,
        "seed": 0,
    }


# ============================================================================ regenerate / audit
def regenerate() -> None:
    res = compute()
    H, _ = build_headline(res)
    Path(DESIGN_MD).write_text(build_markdown(H, res), encoding="utf-8")
    entries = build_manifest_entries(H, res["cand_ids"])
    write_manifest(DESIGN_MANIFEST, entries, _manifest_inputs(), generated_by="scripts/generate_design.py")
    print(f"wrote {DESIGN_MD} + {DESIGN_MANIFEST} ({len(entries)} tracked numbers)")
    print(f"  primary (omega_s^eff) rank: {H['ose_rank']}   EIG rank: {H['eig_rank']}   "
          f"({H['ose_eig_agree']})")
    print(f"  R-contraction rank constant-R: {H['Rc_rank']}   R(phi)-inflated: {H['Ri_rank']}   "
          f"class-flip: {H['class_flip']}")
    print(f"  C4 {H['c4_status']} (X-ray verdict {XRAY_VERDICT_PCT:.2f}% >= {XRAY_THRESHOLD_PCT:.0f}%)")


def _read_committed_manifest() -> dict:
    import json
    data = json.loads(Path(DESIGN_MANIFEST).read_text(encoding="utf-8"))
    return {e["id"]: e["value"] for e in data["entries"]}


def _structural_gates(res: dict, RAW: dict) -> list[str]:
    """Check the claims DESIGN.md actually makes, at their own separations (2026-08-09 amendment).

    A 3-decimal cell at this Monte-Carlo resolution is not the deliverable; the recommendation and the
    class-independence are. These gates are what would catch a real regression -- e.g. C4 ceasing to
    dominate, or the X-ray ratio silently acquiring class sensitivity -- which the old per-cell band could
    not distinguish from noise.
    """
    problems: list[str] = []
    cand_ids = res["cand_ids"]
    if "C4" not in cand_ids:  # the registry conditional dropped C4; the gates below do not apply
        return problems
    for cls, key in (("constant", "Rc"), ("inflated", "Ri")):
        others = {c: RAW[f"{key}_{c}"] for c in cand_ids if c != "C4"}
        runner = max(others, key=others.get)
        gap = RAW[f"{key}_C4"] - others[runner]
        pooled = (RAW[f"se_{key}_C4"] ** 2 + RAW[f"se_{key}_{runner}"] ** 2) ** 0.5
        sigma = gap / pooled if pooled > 0 else float("inf")
        if sigma < AUDIT_MIN_SEPARATION_SIGMA:
            problems.append(
                f"structural[{cls}]: C4's R-contraction lead over {runner} is {gap:+.4f} = {sigma:.1f} "
                f"sigma (< {AUDIT_MIN_SEPARATION_SIGMA:.0f} required) -- the recommendation no longer holds"
            )
    # C4 and C2 are class-INSENSITIVE by construction: their two class cells must be bit-equal.
    for cid in cand_ids:
        if not res["sdc"][cid]["class_sensitive"] and RAW[f"Rc_{cid}"] != RAW[f"Ri_{cid}"]:
            problems.append(f"structural: {cid} is declared class-insensitive but Rc != Ri "
                            f"({RAW[f'Rc_{cid}']:.6g} vs {RAW[f'Ri_{cid}']:.6g})")
    # The well-posed-estimand headline: a direct high-density re-measurement leads on omega_s^eff, with
    # the neutron disappearance slope second. DESIGN.md's audit section claims the TOP-2 is gated, so gate
    # the top-2 (it previously checked only the leader, and the claim was therefore overstated).
    ose = {c: RAW[f"ose_{c}"] for c in cand_ids}
    top2 = sorted(ose, key=lambda c: -ose[c])[:2]
    if top2 != ["C3", "C1"]:
        problems.append(f"structural: the omega_s^eff top-2 is no longer C3 > C1 (fresh {top2}; "
                        f"full ranking {ose})")
    return problems


def audit() -> None:
    """Re-run with pinned seeds and check the committed DESIGN.md against a fresh computation.

    Three gates (see the module docstring + the AMENDMENTs there):
      * every tolerance-audited cell -- EIG-family and sd-contraction alike -- against AUDIT_K_SIGMA sigma
        of ITS OWN published Monte-Carlo SE, pooled committed-vs-fresh and floored at AUDIT_ATOL_FLOOR;
      * the fresh SE against the committed one (AUDIT_SE_RATIO_MAX) wherever the band is SE-governed,
        since the fresh half of every band is otherwise unpublished and unchecked;
      * the structural claims (:func:`_structural_gates`) + the categorical sobol_top.
    Every band and every margin is printed, so a CI log on ANY platform is evidence about how the
    tolerances are actually sized -- the instrumentation the 2026-07-23 cross-arch audit found missing.
    """
    committed = _read_committed_manifest()
    # EVERY audited cell's band is derived from a published per-cell SE, so every audited cell must have
    # one. Stated as a general rule rather than a list of prefixes (2026-08-12: the EIG family joined the
    # construction, and a future cell must not be able to slip in without its SE).
    missing_se = [k for k in committed
                  if not k.startswith("se_") and k != "sobol_top" and f"se_{k}" not in committed]
    if missing_se:
        raise SystemExit(
            "DESIGN_MANIFEST.json predates the Monte-Carlo-SE audit (no se_* entries for "
            f"{', '.join(sorted(missing_se))}). Every audit band is derived from those SEs, so it cannot "
            "run against this manifest: regenerate with `python scripts/generate_design.py`."
        )
    res = compute()
    _, RAW = build_headline(res)
    problems: list[str] = []
    margins: list[tuple[str, float, float]] = []        # (id, |delta|, band)
    uninformative: list[tuple[str, float, float]] = []  # (id, value, band) where band > |value|
    n_eig = n_con = 0
    for entry_id, committed_str in committed.items():
        if entry_id == "sobol_top":  # categorical (top Sobol driver), not a tolerance cell
            if res["sobol"]["top_param"] != committed_str:
                problems.append(f"sobol_top: committed {committed_str!r} vs fresh {res['sobol']['top_param']!r}")
            continue
        if entry_id.startswith("se_"):
            continue  # the SEs are checked implicitly: they SET the band for their own cell
        c = float(committed_str)
        f = RAW[entry_id]
        if entry_id.startswith("eig_") or entry_id == "zero_eig":
            n_eig += 1
        else:
            n_con += 1
        # ONE construction for every audited cell (2026-08-12): the band is K sigma of the cell's own
        # published Monte-Carlo SE, pooled committed-vs-fresh, floored so a fluke-small SE cannot make it
        # vacuous-tight. The two classes differ only in how their SE was measured, not in how it is used.
        se_c = float(committed[f"se_{entry_id}"])
        se_f = RAW[f"se_{entry_id}"]
        se_band = AUDIT_K_SIGMA * (se_c**2 + se_f**2) ** 0.5
        band = max(se_band, AUDIT_ATOL_FLOOR)
        se_governed = se_band >= AUDIT_ATOL_FLOOR
        # Half the band comes from se_f, which is published nowhere. Gate it against the committed SE so a
        # noisier run cannot quietly award itself a wider band (AUDIT_SE_RATIO_MAX) -- but only where the
        # band is actually SE-governed: on a floor-governed cell the SEs do not set the band at all, and
        # for the zero-EIG sanity cell they are two numerically-zero numbers whose ratio is meaningless.
        if (se_governed and se_c > 0 and se_f > 0
                and not (1 / AUDIT_SE_RATIO_MAX <= se_f / se_c <= AUDIT_SE_RATIO_MAX)):
            problems.append(
                f"se_{entry_id}: fresh SE {se_f:.4g} vs committed {se_c:.4g} "
                f"(ratio {se_f / se_c:.2f}x, allowed {1 / AUDIT_SE_RATIO_MAX:.2f}-"
                f"{AUDIT_SE_RATIO_MAX:.2f}x) -- the estimator changed, not just the estimate"
            )
        if se_governed:
            tol = f"{AUDIT_K_SIGMA:.0f}sig(se {se_c:.3f}/{se_f:.3f})"
            if band > abs(c):   # a band wider than the cell cannot falsify anything about that cell
                uninformative.append((entry_id, c, band))
        else:
            # A floor-governed cell is checked against the pre-registered absolute floor, so the
            # band-vs-value comparison above says nothing about it (and its value may be exactly 0).
            tol = f"floor {AUDIT_ATOL_FLOOR:g} abs (se {se_c:.3f}/{se_f:.3f})"
        margins.append((entry_id, abs(c - f), band))
        if abs(c - f) > band:
            problems.append(f"{entry_id}: committed {c:.4g} vs fresh {f:.4g}, "
                            f"|delta|={abs(c - f):.4g} > band {band:.4g} [{tol}]")
    problems += _structural_gates(res, RAW)
    worst = max(margins, key=lambda m: (m[1] / m[2]) if m[2] else 0.0, default=("-", 0.0, 1.0))
    if problems:
        raise SystemExit("DESIGN.md audit FAILED:\n  " + "\n  ".join(problems))
    print(f"design audit OK: {n_eig} EIG-family + {n_con} contraction cells within "
          f"{AUDIT_K_SIGMA:.0f} sigma of their published MC SE (floored at {AUDIT_ATOL_FLOOR:g} absolute; "
          f"EIG SEs measured over {AUDIT_EIG_REPLICATES} replicate base chains), "
          f"structural gates pass, sobol_top matches")
    print(f"  worst margin: {worst[0]} at {worst[1] / worst[2]:.1%} of its band "
          f"(|delta|={worst[1]:.4g}, band={worst[2]:.4g})")
    # Vacuity is reported, never silently counted as a pass: a band wider than the cell's own value means
    # that cell's per-cell check is uninformative and only the structural gates constrain it.
    if uninformative:
        print("  NOT INFORMATIVE (band exceeds the cell's own value -- per-cell check cannot falsify "
              "these; the structural gates are what constrain them):")
        for k, c, band in uninformative:
            print(f"    {k}: value {c:+.4g}, band {band:.4g} ({band / abs(c):.2f}x the value)")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--audit" in argv:
        audit()
    else:
        regenerate()


if __name__ == "__main__":
    main()
