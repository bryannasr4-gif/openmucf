"""Generate (or audit) DESIGN.md + DESIGN_MANIFEST.json -- the Bayesian experimental-design ranking.

    python scripts/generate_design.py            # regenerate DESIGN.md + DESIGN_MANIFEST.json (runs NUTS)
    python scripts/generate_design.py --audit    # re-run with pinned seeds; check committed numbers

Content (openmucf.design): rank candidate NEXT muCF experiments (C1 neutron disappearance slope @ phi=2.0;
C2 lambda_c(T) @ 800 K; C3 omega_s^eff @ phi=2.4 under scenario A AND B; C4 X-ray/neutron ratio) by the
PRIMARY preposterior sd-contraction metric and the SECONDARY nested-MC EIG, over the EXISTING weak-prior
calibrate posterior. R is reported class-conditionally (constant-R vs R(phi)-inflated); the class-flip is
a finding.

Reproducibility (WAVE2 A1/A2 -- the CALIBRATION.md precedent, NOT the byte-diff pattern): DESIGN.md +
DESIGN_MANIFEST.json carry numpyro/NUTS-derived numbers that are NOT byte-stable cross-architecture, so
`make audit` byte-diffs NEITHER. Instead `--audit` re-runs with the pinned seeds and compares every
manifest-tracked number at a pre-registered tolerance, split by quantity class exactly as the calibration
audit splits its mean/sd cells (2%/8%):
  * EIG-in-bits (a relative-scale quantity, averaged over n_outer*n_inner draws): 5% RELATIVE tolerance
    (the A1 pre-registered value).
  * sd-contraction ratios (dimensionless, legitimately passing through zero -- the estimand-discipline
    cells collapse to ~0 -- so a pure relative tolerance is ill-posed): 3 percentage-points ABSOLUTE,
    the X-ray-feasibility-documented contraction Monte-Carlo-noise floor (~+/-3 pp). For the largest
    shipped contraction (0.56) this equals 5.4% relative, i.e. the 5% intent; for the near-zero
    estimand-discipline cells it is the only well-posed tolerance.
Both are hard-failing and pinned by tests/test_design_audit.py (the never-soften-silently rule of
WAVE1 spec 1.5). Doc<->manifest consistency is enforced by `provenance --check` (regenerated together).

Importable without side effects (all work is inside functions / main()); the tests import the pure
formatting/registry/audit helpers without running the NUTS refits.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

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
AUDIT_RTOL_EIG = 0.05          # EIG-in-bits: 5% RELATIVE (WAVE2 A1 pre-registered)
AUDIT_ATOL_CONTRACTION = 0.03  # sd-contraction ratios: 3 pp ABSOLUTE (X-ray-feasibility MC-noise floor)

CANDIDATE_ORDER = ("C1", "C2", "C3", "C4")


# ============================================================================ computation
def compute(seed: int = 0) -> dict:
    """Run the full design analysis once (NUTS-heavy). Returns raw numbers keyed for build_headline."""
    samples = design.base_posterior(seed=seed)
    reg = design.registry(XRAY_VERDICT_PCT, XRAY_THRESHOLD_PCT)
    cand_ids = [c for c in CANDIDATE_ORDER if c in reg["candidates"]]

    eig = {cid: design.eig_nested_mc(cid, samples=samples, seed=seed) for cid in cand_ids}
    eig_c3_inflated = design.eig_nested_mc("C3", cls="inflated", samples=samples, seed=seed)
    sdc = {cid: design.sd_contraction(cid, samples=samples, seed=seed) for cid in cand_ids}

    zero_eig = design.eig_nested_mc(design.replicate_candidate(), samples=samples, seed=seed)
    sobol = design.sobol_consistency(seed=seed)

    return {
        "seed": seed,
        "registry": reg,
        "cand_ids": cand_ids,
        "eig": eig,
        "eig_c3_inflated": eig_c3_inflated,
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
    the class-flip finding (put in H as strings)."""
    H: dict[str, str] = {}
    RAW: dict[str, float] = {}
    cand_ids = res["cand_ids"]

    for cid in cand_ids:
        RAW[f"eig_{cid}"] = res["eig"][cid]["eig_bits"]
        RAW[f"ose_{cid}"] = res["sdc"][cid]["ose_contraction"]
        RAW[f"Rc_{cid}"] = res["sdc"][cid]["R_contraction"]["constant"]
        RAW[f"Ri_{cid}"] = res["sdc"][cid]["R_contraction"]["inflated"]
    RAW["eig_C3_inflated"] = res["eig_c3_inflated"]["eig_bits"]
    RAW["zero_eig"] = res["zero_eig_bits"]

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
    H["class_flip"] = "YES" if rc_rank != ri_rank else "NO"
    H["sobol_top"] = res["sobol"]["top_param"]
    H["c4_status"] = "included" if res["registry"]["c4_included"] else "dropped"
    return H, RAW


# ============================================================================ DESIGN.md
def _sdc_table(H: dict, cand_ids) -> str:
    rows = []
    for cid in cand_ids:
        rows.append(f"| {cid} | {H[f'ose_{cid}']} | {H[f'Rc_{cid}']} | {H[f'Ri_{cid}']} |")
    return ("| candidate | omega_s^eff contraction | R contraction (constant-R) | "
            "R contraction (R(phi)-inflated) |\n"
            "|---|---|---|---|\n" + "\n".join(rows))


def _eig_table(H: dict, cand_ids) -> str:
    rows = [f"| {cid} | {H[f'eig_{cid}']} |" for cid in cand_ids]
    rows.append(f"| C3 (scenario-B, R(phi)-inflated) | {H['eig_C3_inflated']} |")
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
> class-conditionally (constant-R vs R(phi)-inflated) and report the class-flip as a finding.

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
contraction under constant-R: **{H['Rc_rank']}**; under R(phi)-inflated: **{H['Ri_rank']}**. Contraction
cells carry a ~+/-3 pp Monte-Carlo floor over the {s['n_synth']} synthetic datasets, so separations below
that -- the near-zero estimand-discipline cells especially -- are not distinguishable from zero; the
deliverable is the LARGE-effect ranking and the structural collapse below (both reproduced within
tolerance by `--audit`), not the fine ordering of sub-floor cells.

**The recommended experiment depends on the estimand.** To break the omega_s0/R degeneracy (tighten R),
**C4 (X-ray/neutron ratio) wins decisively and ROBUSTLY across both structural classes** -- its R
contraction ({H['Rc_C4']}) is class-independent because it constrains omega_s0 DIRECTLY, not through the
R(phi) form. To tighten omega_s^eff specifically, C3 (a direct high-density re-measurement) leads.

**Class-flip finding ({H['class_flip']}).** The neutron-only candidates' apparent R information is
generated by the assumed R(phi) form, exactly as the estimand-discipline note warns. The neutron
disappearance slope C1 contracts R by {H['Rc_C1']} under constant-R but collapses to {H['Ri_C1']} under
R(phi)-inflation -- a drop far larger than the +/-3 pp contraction noise floor -- so its apparent
constant-R information is revealed as an artifact of the structural assumption. At the collapsed level C1
is **statistically indistinguishable from the R-independent control C2** ({H['Ri_C2']}): the nominal
ranking reads constant-R {H['Rc_rank']} vs R(phi)-inflated {H['Ri_rank']}, but that C1<->C2 crossing is
itself WITHIN the +/-3 pp noise floor and is NOT a robust ordering -- the robust finding is C1's collapse,
not the crossing. C4's contraction ({H['Rc_C4']}) does not move between classes: it constrains omega_s0
DIRECTLY, so it is the one candidate that identifies R without the structural assumption.

## SECONDARY metric -- nested-Monte-Carlo EIG
{_eig_table(H, cand_ids)}

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
   {H['zero_eig']} bits (<= the estimator's Monte-Carlo noise).
2. **EIG monotone in stated precision:** a tighter measurement never lowers EIG (a 3-point sigma sweep).
3. **Sobol-consistency in the small-noise limit:** the parameter a tiny-sigma X_mu measurement informs
   most is **{H['sobol_top']}** -- the top Sobol driver of X_mu over the same `openmucf.uq` prior box.

## Reproducibility / audit
Every number here is NUTS/Monte-Carlo derived and reproduces to Monte-Carlo error, NOT byte-identically
(the `CALIBRATION.md` precedent). `make audit` byte-diffs NEITHER this file nor its manifest; instead
`python scripts/generate_design.py --audit` re-runs with the pinned seeds and checks every manifest
number -- EIG bits at {AUDIT_RTOL_EIG:.0%} relative, sd-contraction ratios at {AUDIT_ATOL_CONTRACTION}
absolute ({AUDIT_ATOL_CONTRACTION * 100:.0f} percentage points, the contraction Monte-Carlo-noise floor;
a relative tolerance is ill-posed for the near-zero estimand-discipline cells). Regenerate with
`python scripts/generate_design.py`.
"""


# ============================================================================ manifest
def build_manifest_entries(H: dict, cand_ids) -> list[ManifestEntry]:
    def _e(entry_id, pattern):
        return ManifestEntry(id=entry_id, value=H[entry_id], pattern=pattern,
                             source_type="derivation", source="scripts/generate_design.py", doc="DESIGN.md")

    entries: list[ManifestEntry] = []
    # sd-contraction cells (each appears in the PRIMARY table row: | Cx | ose | Rc | Ri |)
    for cid in cand_ids:
        row = rf"\| {cid} \| {re.escape(H[f'ose_{cid}'])} \| {re.escape(H[f'Rc_{cid}'])} \| {re.escape(H[f'Ri_{cid}'])} \|"
        entries.append(_e(f"ose_{cid}", row))
        entries.append(_e(f"Rc_{cid}", row))
        entries.append(_e(f"Ri_{cid}", row))
    # EIG cells (each appears in the SECONDARY table row: | Cx | eig |)
    for cid in cand_ids:
        entries.append(_e(f"eig_{cid}", rf"\| {cid} \| {re.escape(H[f'eig_{cid}'])} \|"))
    entries.append(_e("eig_C3_inflated", rf"scenario-B, R\(phi\)-inflated\) \| {re.escape(H['eig_C3_inflated'])} \|"))
    # sanity-gate headline claims
    entries.append(_e("zero_eig", rf"yields EIG =\s*{re.escape(H['zero_eig'])} bits"))
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
        "audit_rtol_eig": AUDIT_RTOL_EIG,
        "audit_atol_contraction": AUDIT_ATOL_CONTRACTION,
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


def audit() -> None:
    """Re-run with pinned seeds; check every manifest number against a fresh computation at the
    class-appropriate tolerance (EIG 5% relative; contraction 3pp absolute). Hard-fail on any miss."""
    committed = _read_committed_manifest()
    res = compute()
    _, RAW = build_headline(res)
    problems: list[str] = []
    n_eig = n_con = 0
    for entry_id, committed_str in committed.items():
        if entry_id == "sobol_top":  # categorical (top Sobol driver), not a tolerance cell
            if res["sobol"]["top_param"] != committed_str:
                problems.append(f"sobol_top: committed {committed_str!r} vs fresh {res['sobol']['top_param']!r}")
            continue
        c = float(committed_str)
        f = RAW[entry_id]
        if entry_id.startswith("eig_"):
            n_eig += 1
            ok = abs(c - f) <= AUDIT_RTOL_EIG * max(abs(c), abs(f))
            tol = f"{AUDIT_RTOL_EIG:.0%} rel"
        else:  # ose_/Rc_/Ri_/zero_eig -> contraction/near-zero absolute tolerance
            n_con += 1
            ok = abs(c - f) <= AUDIT_ATOL_CONTRACTION
            tol = f"{AUDIT_ATOL_CONTRACTION} abs"
        if not ok:
            problems.append(f"{entry_id}: committed {c:.4g} vs fresh {f:.4g} (> {tol})")
    if problems:
        raise SystemExit("DESIGN.md audit FAILED:\n  " + "\n  ".join(problems))
    print(f"design audit OK: {n_eig} EIG cells within {AUDIT_RTOL_EIG:.0%} rel, "
          f"{n_con} contraction/near-zero cells within {AUDIT_ATOL_CONTRACTION} abs, sobol_top matches")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--audit" in argv:
        audit()
    else:
        regenerate()


if __name__ == "__main__":
    main()
