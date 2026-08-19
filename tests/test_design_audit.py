"""WS-D: the DESIGN.md audit wiring.

DESIGN.md + DESIGN_MANIFEST.json are NUTS-derived and NOT byte-diffed; `generate_design.py --audit`
tolerance-checks them. Here we (a) pin the audit tolerances against a SILENT softening (an audit
tolerance may never be softened silently; same guard as test_calibration_audit.py), and (b) verify the
doc<->manifest render deterministically, carry the mandated verbatim paragraphs, and pass
`provenance --check` -- all WITHOUT running the NUTS pipeline (a fixed synthetic ``res`` exercises
the pure rendering path).
"""

import importlib.util
from pathlib import Path

import pytest

import openmucf
from openmucf import design
from openmucf.provenance import check_manifest, write_manifest

_SCRIPT = Path(openmucf.__file__).resolve().parent.parent / "scripts" / "generate_design.py"


def _load_script():
    """Import the generator by path (no NUTS: all work is guarded behind main())."""
    spec = importlib.util.spec_from_file_location("_gen_design", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mock_res(*, resolved_flip: bool = False, negative_delta: bool = False,
              eig_se_scale: float = 1.0, eig_shift: dict | None = None):
    """A fixed synthetic result matching compute()'s structure -- no NUTS, fully deterministic.

    ``resolved_flip`` toggles the class-contrast between "not resolved" (the honest default: a small
    paired delta against a comparable SE) and "resolved at >3 sigma", so BOTH branches of the derived
    class-flip prose are exercised without running NUTS. ``negative_delta`` flips C1's contraction so the
    resolved contrast points the OTHER way (R information RISES under inflation) -- the sign the shipped
    C3 cell actually has, and the sign whose prose was wrong until 2026-08-10.

    ``eig_se_scale`` rescales the replicate-measured EIG SEs and ``eig_shift`` moves individual EIG cells,
    so a test can hold a cell's DELTA fixed while changing only the SE that sets its band (2026-08-12) --
    which is what distinguishes a derived band from a constant wearing a derived band's name.
    """
    cand_ids = ["C1", "C2", "C3", "C4"]
    eig_bits = {"C1": 1.629, "C2": 1.782, "C3": 1.081, "C4": 3.240}
    ose = {"C1": 0.231, "C2": 0.049, "C3": 0.560, "C4": -0.026}
    rc = {"C1": 0.070, "C2": 0.017, "C3": 0.101, "C4": 0.408}
    ri = {"C1": 0.002, "C2": 0.017, "C3": 0.103, "C4": 0.408}
    if negative_delta:
        rc["C1"], ri["C1"] = ri["C1"], rc["C1"]
    se = {"C1": 0.020, "C2": 0.012, "C3": 0.026, "C4": 0.021}
    sens = {"C1": True, "C2": False, "C3": True, "C4": False}
    delta_se = 0.004 if resolved_flip else 0.030
    # Replicate-measured EIG SEs (the 2026-08-12 band). zero_eig's is EXACTLY zero -- the replicate
    # observable is constant -- which is the floor-governed path the audit has to survive.
    eig_se = {"eig_C1": 0.046, "eig_C2": 0.037, "eig_C3": 0.058, "eig_C4": 0.033,
              "eig_C3_inflated": 0.041, "zero_eig": 0.0}
    eig_se = {k: v * eig_se_scale for k, v in eig_se.items()}
    shift = eig_shift or {}
    eig_bits = {c: eig_bits[c] + shift.get(f"eig_{c}", 0.0) for c in cand_ids}
    return {
        "seed": 0,
        "registry": design.registry(42.95, 15.0),
        "cand_ids": cand_ids,
        "eig": {c: {"eig_bits": eig_bits[c], "n_outer": 256, "n_inner": 256} for c in cand_ids},
        "eig_c3_inflated": {"eig_bits": 2.492 + shift.get("eig_C3_inflated", 0.0)},
        "eig_se": eig_se,
        "sdc": {c: {"ose_contraction": ose[c], "ose_contraction_se": se[c], "n_synth": 64,
                    "class_sensitive": sens[c],
                    "R_contraction": {"constant": rc[c], "inflated": ri[c]},
                    "R_contraction_se": {"constant": se[c], "inflated": se[c]},
                    "R_contraction_class_delta": {"value": rc[c] - ri[c], "se": delta_se, "paired": True},
                    "se_components": {"base_sd_mcse_rel": {"omega_s_eff_pct": 0.016, "R": 0.017},
                                      "boot": {"ose": 0.004,
                                               "R": {"constant": 0.006, "inflated": 0.006}}}}
                for c in cand_ids},
        "zero_eig_bits": -1e-7 + shift.get("zero_eig", 0.0),  # exercises the negative-zero normalisation
        "sobol": {"top_param": "R"},
        "settings": {"n_outer": 256, "n_inner": 256, "n_synth": 64,
                     "num_warmup": 1000, "num_samples": 4000},
    }


def test_audit_tolerances_pinned():
    """Any SILENT softening of the audit tolerances trips this test (same literal-substring guard as
    test_calibration_audit.py). Changing any of these requires deliberately editing this pin + a dated,
    written amendment in the generator that says what was measured and why the band moved.

    2026-08-09: the contraction band changed from a fixed ``AUDIT_ATOL_CONTRACTION = 0.03`` to
    ``AUDIT_K_SIGMA`` sigma of each cell's OWN published Monte-Carlo SE, floored at ``AUDIT_ATOL_FLOOR``.
    This is a deliberate, dated re-registration, not a softening: the old fixed band was SMALLER than the
    estimator's Monte-Carlo error on 7 of 12 cells, so it could only ever be met by regenerating the same
    pseudo-random realization (it failed on 5 cells the first time a different architecture ran it). The
    new band's committed half cannot be widened without publishing a larger SE in DESIGN.md, and n_synth
    was raised 8 -> 64 so the SEs -- hence the band -- actually shrank for every cell.

    2026-08-10: ``AUDIT_SE_RATIO_MAX`` added. The FRESH half of the band is published nowhere, so without
    it a noisier platform could silently award itself a wider band; and the published SE itself gained the
    base-chain term (``design.SE_BASE_BATCHES``) it had been omitting.

    2026-08-12: the EIG cells' ``AUDIT_RTOL_EIG = 0.05`` (5% RELATIVE) is DELETED and replaced by the same
    measured per-cell band, with the SE measured in-run over ``AUDIT_EIG_REPLICATES`` replicate base
    chains. Again a re-registration, not a softening: a 200-realization sweep over the base-chain seed puts
    those cells' realization noise at 0.042-0.068 bits (1.40%-6.10% RELATIVE), so the noise is ABSOLUTE and
    a single relative constant is the wrong SHAPE -- the 5% band reds 49.5% of runs against an independent
    realization (worst cell eig_C3, whose own noise exceeds the whole band), while the >= 34.5% that would
    cover eig_C3 makes eig_C4's band 24.8 sigma of its own noise. The measured band is 0.555%/run. The
    superseded constant is guarded like ``AUDIT_ATOL_CONTRACTION``: it must not come back.
    """
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "AUDIT_EIG_REPLICATES = 20" in src
    assert "AUDIT_K_SIGMA = 4.0" in src
    assert "CLAIM_K_SIGMA = 3.0" in src
    assert "AUDIT_ATOL_FLOOR = 0.01" in src
    assert "AUDIT_MIN_SEPARATION_SIGMA = 3.0" in src
    assert "AUDIT_SE_RATIO_MAX = 3.0" in src
    assert "AUDIT_ATOL_CONTRACTION" not in src, "the superseded fixed band must not come back"
    assert "AUDIT_RTOL_EIG" not in src, "the superseded 5% relative EIG band must not come back"


def test_n_synth_and_se_settings_pinned():
    """The Monte-Carlo resolution of the PRIMARY metric is pre-registered (openmucf.design)."""
    assert design.N_SYNTH_DEFAULT == 64
    assert design.SE_BOOTSTRAP == 10_000
    assert design.SE_BASE_BATCHES == 20
    src = (Path(design.__file__)).read_text(encoding="utf-8")
    assert "N_SYNTH_DEFAULT = 64" in src
    assert "SE_BASE_BATCHES = 20" in src


def test_published_se_carries_the_base_chain_term():
    """The published +- is bootstrap (+) base-chain sd error -- NOT the bootstrap alone.

    Without the second term the band is narrower than the error it exists to absorb: on the shipped run
    the base-chain term alone shifts ose_C1 by ~0.011 against a 4-sigma band of ~0.013, i.e. the
    cross-architecture audit this branch adds would have been ~50% likely to fail spuriously.
    """
    import numpy as np

    values = list(np.random.default_rng(3).normal(0.4, 0.12, 64))
    med = float(np.median(values))
    boot = design.median_se(values)
    total = design._cell_se(values, med, 0.016)
    assert total > boot                                                   # strictly larger
    assert total == pytest.approx(np.hypot(boot, (1 - med) * 0.016))      # ... by exactly that term
    assert design._cell_se(values, med, 0.0) == pytest.approx(boot)       # degenerates cleanly

    # the estimator of the base-chain term is deterministic and tracks a known chain's sd error
    rng = np.random.default_rng(5)
    draws = rng.normal(0.0, 1.0, 4000)
    d = design.base_sd_mcse_rel(draws)
    assert design.base_sd_mcse_rel(draws) == d                            # deterministic
    assert 0.3 / np.sqrt(4000) < d < 4.0 / np.sqrt(4000)  # iid theory: 1/sqrt(2n) = 1.1%, batch est ~that


def test_median_se_is_deterministic_and_tracks_dispersion():
    """The published +- is a real, reproducible SE: same input -> same value, and it falls like 1/sqrt(n)."""
    rng = __import__("numpy").random.default_rng(0)
    small = list(rng.normal(0.4, 0.15, 8))
    big = list(rng.normal(0.4, 0.15, 512))
    assert design.median_se(small) == design.median_se(small)      # deterministic
    assert design.median_se(big) < design.median_se(small)         # shrinks with n
    # within a factor ~1.5 of the normal-theory 1.2533*s/sqrt(n) on a normal sample
    import numpy as np
    theory = 1.2533 * np.std(big, ddof=1) / np.sqrt(len(big))
    assert 0.67 * theory <= design.median_se(big) <= 1.5 * theory


def test_design_doc_and_manifest_render_deterministically(tmp_path):
    """The doc + manifest render bit-identically from a fixed result, carry the verbatim paragraphs +
    the I6 fence, and every tracked manifest number is found in the doc (provenance green)."""
    mod = _load_script()
    res = _mock_res()
    h1, _ = mod.build_headline(res)
    h2, _ = mod.build_headline(res)
    md1 = mod.build_markdown(h1, res)
    md2 = mod.build_markdown(h2, res)
    assert md1 == md2  # deterministic given the result

    # verbatim estimand-discipline paragraph
    assert "EIG on omega_s^eff at stated conditions is well-posed" in md1
    assert "generated by the ASSUMED structural form R(phi)" in md1
    # 2026-08-09: the banner used to promise "report the class-flip as a finding", which asserted a flip
    # exists. A reader would take the opposite of what the body now reports when the contrast is not
    # resolved, so the promise is now symmetric in outcome.
    assert "report the class contrast, RESOLVED OR NOT,\n> as a finding" in md1
    assert "report the class-flip as a finding" not in md1
    # verbatim scenario-B disclaimer
    assert "the scenario-B MuFusE EIG is large BY CONSTRUCTION (the widest prior wins)" in md1
    assert "this is a property of the prior, not of the experiment." in md1
    # I6 fence in the header (generic warm-thread language; no named private outreach targets in a public doc)
    assert "never cold-mailed" in md1 and "ALREADY-WARM thread" in md1 and "not outreach" in md1
    assert "Antognini" not in md1 and "NCCR" not in md1  # public-hygiene: no private outreach target names
    # negative-zero normalised, class-contrast + C4 conditional surfaced
    assert "yields EIG =\n" in md1 or "yields EIG = " in md1
    assert "-0.000" not in md1
    assert "C4 (X-ray/neutron ratio) is **included**" in md1
    # every PRIMARY cell is published WITH its Monte-Carlo SE, and the superseded floor claim is gone
    assert "| C1 | 0.231 +- 0.020 | 0.070 +- 0.020 | 0.002 +- 0.020 |" in md1
    # the false floor is RETRACTED in place (named, so a reader of an old revision sees why), never
    # restated as current, and the hard-coded C1-collapse claim is gone
    assert "cells carry a ~+/-3 pp" not in md1
    assert "a drop far larger than" not in md1
    assert "supersedes the" in md1 and "~+/-3 pp Monte-Carlo" in md1
    # the +- is documented as BOTH components, and the paired-contrast estimand is not left to be
    # mis-derived by subtracting the table columns (2026-08-10)
    assert "Monte-Carlo error of the shared denominator" in md1
    assert "medians do not subtract" in md1

    # write doc + manifest to a temp repo root; provenance must find every tracked value in the doc
    (tmp_path / "DESIGN.md").write_text(md1, encoding="utf-8")
    entries = mod.build_manifest_entries(h1, res["cand_ids"])
    write_manifest(tmp_path / "DESIGN_MANIFEST.json", entries, mod._manifest_inputs(),
                   generated_by="scripts/generate_design.py")
    failures = check_manifest(tmp_path / "DESIGN_MANIFEST.json", repo_root=tmp_path)
    assert failures == [], failures
    # 4x3 sd-contraction cells + 4x3 their MC SEs + (4 EIG + inflated + zero-EIG) + those 6 cells' MC SEs
    # + the categorical sobol_top = 37 (31 before the EIG cells gained published SEs on 2026-08-12).
    assert len(entries) == 37
    # every EIG cell is published WITH its replicate-measured SE, in the SECONDARY table
    assert "| C1 | 1.629 +- 0.046 |" in md1
    assert "| C3 (scenario-B, R(phi)-inflated) | 2.492 +- 0.041 |" in md1


def test_class_contrast_prose_is_derived_not_asserted():
    """The class-flip sentence must follow the measured paired delta / SE, in BOTH directions.

    Regression guard for the defect the 2026-07-23 cross-arch reproduction exposed: the doc hard-coded a
    resolved C1 collapse ("a drop far larger than the +/-3 pp contraction noise floor") from a single
    realization whose separation was ~1.7 sigma of its own spread.
    """
    mod = _load_script()
    h_unres, _ = mod.build_headline(_mock_res(resolved_flip=False))
    h_res, _ = mod.build_headline(_mock_res(resolved_flip=True))

    assert h_unres["class_flip"] == "NOT RESOLVED"
    assert "does NOT" in h_unres["class_flip_reading"]

    assert h_res["class_flip"].startswith("RESOLVED for ")
    assert "C1" in h_res["class_flip"]
    assert "RESOLVED at >= 3 sigma" in h_res["class_flip_reading"]
    assert "collapses under R(phi)-inflation" in h_res["class_flip_reading"]

    # the sigma actually shown is |delta| / SE, not a narrative
    assert h_res["sigma_delta_C1"] == f"{abs(0.070 - 0.002) / 0.004:.1f}"
    assert h_unres["sigma_delta_C1"] == f"{abs(0.070 - 0.002) / 0.030:.1f}"


def test_unresolved_prose_never_generalises_past_the_resolved_cells():
    """The unresolved branch must not claim "no R information" for a candidate whose cells RESOLVE.

    Regression guard for the 2026-08-10 merge-gate defect: the branch ended in a hard-coded universal --
    "the neutron-only candidates deliver no R information that survives its own Monte-Carlo error under
    EITHER structural class" -- while the same document's `unresolved_cells` line (correctly) omitted C3,
    whose cells sit at 3.9 sigma in this very fixture. The document contradicted itself.
    """
    mod = _load_script()
    h, raw = mod.build_headline(_mock_res(resolved_flip=False))
    prose = h["class_flip_reading"].replace("\n", " ")

    def sig(cid, key):
        return abs(raw[f"{key}_{cid}"]) / raw[f"se_{key}_{cid}"]

    # ground truth for this fixture: C3 resolves under BOTH classes and C1 under constant-R, so the old
    # universal ("the neutron-only candidates deliver no R information") was false about two of the three.
    assert sig("C3", "Rc") >= 3.0 and sig("C3", "Ri") >= 3.0
    assert sig("C1", "Rc") >= 3.0
    assert sig("C2", "Rc") < 3.0 and sig("C2", "Ri") < 3.0

    # a resolved neutron-only candidate is NEVER swept into the "delivers nothing" clause ...
    flat_clause = prose.split("no R contraction distinguishable")[0]
    assert "C3" not in flat_clause and "C1" not in flat_clause, prose
    assert "C2" in flat_clause, prose
    # ... it is named as resolving, and the universal quantifier is gone for good
    assert "C1, C3 -- a nonzero R contraction does resolve" in prose, prose
    assert "the neutron-only candidates deliver no R information" not in prose, prose
    # C4 (class-insensitive AND resolved) carries the recommendation
    assert "identical across both structural classes by construction: C4" in prose, prose


def test_resolved_prose_is_correct_for_a_NEGATIVE_contrast():
    """A resolved contrast pointing the other way must not be explained as a collapse.

    The shipped C3 contrast is negative at 2.8 sigma -- 0.2 sigma from firing this branch. Until
    2026-08-10 the conclusion after ``_dir()`` was hard-coded to "the apparent constant-R information is
    an artifact", which only parses when the contraction FALLS under inflation.
    """
    mod = _load_script()
    h, _ = mod.build_headline(_mock_res(resolved_flip=True, negative_delta=True))
    prose = h["class_flip_reading"].replace("\n", " ")
    assert h["class_flip"].startswith("RESOLVED for ")
    assert "C1 RISES under R(phi)-inflation" in prose, prose
    assert "collapses under R(phi)-inflation" not in prose, prose
    # the conclusion must be sign-neutral -- no claim that the CONSTANT-R side is the artifact
    assert "in either direction" in prose, prose
    assert "apparent constant-R information" not in prose, prose


def test_structural_gates_catch_a_lost_recommendation():
    """The audit's structural gates fire when the DELIVERABLE breaks, not just when a decimal moves."""
    mod = _load_script()
    res = _mock_res()
    _, raw = mod.build_headline(res)
    assert mod._structural_gates(res, raw) == []          # baseline: the shipped claims hold

    # (a) C4 loses its lead -> the recommendation is gone
    broken = dict(raw, Rc_C4=0.105, Ri_C4=0.105)
    probs = mod._structural_gates(res, broken)
    assert any("recommendation no longer holds" in p for p in probs), probs

    # (b) a class-insensitive candidate acquires class sensitivity -> structural error
    probs = mod._structural_gates(res, dict(raw, Ri_C2=0.030))
    assert any("declared class-insensitive" in p for p in probs), probs

    # (c) the well-posed-estimand headline changes hands
    probs = mod._structural_gates(res, dict(raw, ose_C1=0.900))
    assert any("top-2 is no longer C3 > C1" in p for p in probs), probs

    # (d) the top-2 claim DESIGN.md makes is really gated at top-2, not just top-1: C3 still leads, but
    #     C2 displaces C1 in second place. The pre-2026-08-10 gate checked only the leader and passed.
    probs = mod._structural_gates(res, dict(raw, ose_C2=0.400))
    assert any("top-2 is no longer C3 > C1" in p for p in probs), probs


# ---------------------------------------------------------------- the 2026-08-12 measured EIG band
def test_eig_replicate_se_uses_seeds_1_to_R_with_ddof_1(monkeypatch):
    """The SE ESTIMATOR's own convention is part of the tolerance, so it is pinned like a constant.

    Every other test here injects ``eig_se`` through the fixture, which exercises how the band USES the
    SE but never how the SE is MADE. That leaves the estimator itself unguarded: silently changing
    ``ddof=1`` to ``ddof=0``, or the replicate seeds from ``1..R`` to ``0..R-1``, narrows every EIG band
    (~2.5% for ddof; eig_C3's SE 0.058 -> 0.049 for the seed range), and BOTH edits would regenerate the
    document cleanly and pass ``--audit``, because the committed and fresh sides move together. That is
    precisely the silent softening this guard forbids. This test runs the real ``eig_replicate_se`` with
    the NUTS calls replaced by a deterministic stand-in, so the seed set and the ddof are checked as
    BEHAVIOUR rather than as source text.
    """
    import numpy as np

    mod = _load_script()
    seen: list[int] = []

    def fake_base_posterior(seed):
        seen.append(seed)
        return {"marker": float(seed)}

    def fake_eig(candidate, samples=None, seed=0, cls="constant"):
        return {"eig_bits": samples["marker"]}      # cell value == the base seed, exactly

    monkeypatch.setattr(design, "base_posterior", fake_base_posterior)
    monkeypatch.setattr(design, "eig_nested_mc", fake_eig)
    monkeypatch.setattr(design, "replicate_candidate", lambda: "REPLICATE")

    se = mod.eig_replicate_se(["C1"], analysis_seed=0)
    R = mod.AUDIT_EIG_REPLICATES

    # the committed seed stays OUT of its own error bar, and the replicates are 1..R inclusive
    assert seen == list(range(1, R + 1)), seen
    assert 0 not in seen
    # ... and the spread is the ddof=1 sample sd of those values, not the ddof=0 population sd
    values = list(range(1, R + 1))
    assert se["eig_C1"] == pytest.approx(float(np.std(values, ddof=1)))
    assert se["eig_C1"] != pytest.approx(float(np.std(values, ddof=0)))
    # all six EIG-family cells are measured, not just the candidates
    assert set(se) == {"eig_C1", "eig_C3_inflated", "zero_eig"}


def _audit_against(tmp_path, monkeypatch, committed_res, fresh_res):
    """Run ``audit()`` with a manifest written from ``committed_res`` and a fresh run of ``fresh_res``.

    Neither side runs NUTS: ``compute()`` is replaced, so what is exercised is exactly the band
    arithmetic and the gates -- the part a test can own.
    """
    mod = _load_script()
    h, _ = mod.build_headline(committed_res)
    entries = mod.build_manifest_entries(h, committed_res["cand_ids"])
    monkeypatch.chdir(tmp_path)
    write_manifest(mod.DESIGN_MANIFEST, entries, mod._manifest_inputs(),
                   generated_by="scripts/generate_design.py")
    monkeypatch.setattr(mod, "compute", lambda *a, **k: fresh_res)
    return mod


def test_eig_band_is_derived_from_the_published_se_not_a_constant(tmp_path, monkeypatch, capsys):
    """The SAME |delta| must pass or fail according to the PUBLISHED SE, not any fixed tolerance.

    This is the property the 2026-08-12 re-registration exists to create, and the one a band that is
    secretly still a constant would fail. A 0.20-bit shift on ``eig_C3`` (committed 1.081) is 18.5%
    relative -- far outside the superseded 5% band, and outside 10% and 15% too -- yet it is only 1.7
    sigma of that cell's measured realization noise, so it must PASS. Shrink the SE tenfold on BOTH sides
    (so the ratio guard sees 1.0x and cannot be what fires) and the identical shift must FAIL.
    """
    shift = {"eig_C3": 0.20}

    # (a) wide measured SE -> the shift is inside the band
    mod = _audit_against(tmp_path, monkeypatch, _mock_res(), _mock_res(eig_shift=shift))
    mod.audit()
    out = capsys.readouterr().out
    assert "design audit OK" in out
    assert "20 replicate base chains" in out          # the band names its own provenance
    # the band really is 4 sigma of the two SEs, not a relative constant
    assert abs(0.20) > 0.15 * 1.081                   # >> any plausible relative constant
    assert 0.20 < 4.0 * (0.058**2 + 0.058**2) ** 0.5  # ... and inside 4 sigma of the measured SE

    # (b) same shift, SEs 10x tighter on BOTH sides -> the band shrinks with them and it fails
    mod = _audit_against(tmp_path, monkeypatch,
                         _mock_res(eig_se_scale=0.1), _mock_res(eig_se_scale=0.1, eig_shift=shift))
    with pytest.raises(SystemExit) as exc:
        mod.audit()
    msg = str(exc.value)
    assert "eig_C3" in msg and "band" in msg
    assert "ratio" not in msg, "must fail on the BAND, not on the SE-ratio guard"


def test_every_audited_cell_publishes_an_se(tmp_path, monkeypatch):
    """No audited cell may regress to a bare constant: each one needs a published ``se_<id>`` companion.

    The audit derives every band from a published SE, so an entry without one either crashes the audit or
    (worse) would have to be given a constant. The shipped manifest is checked, and the preflight is shown
    to hard-fail with the regenerate instruction -- a SystemExit, never a KeyError -- when one is missing.
    """
    import json

    committed = json.loads((_SCRIPT.parent.parent / "DESIGN_MANIFEST.json").read_text(encoding="utf-8"))
    ids = {e["id"] for e in committed["entries"]}
    audited = {i for i in ids if not i.startswith("se_") and i != "sobol_top"}
    assert audited, ids
    assert {i for i in audited if f"se_{i}" not in ids} == set()
    assert "audit_rtol_eig" not in committed["inputs"]
    assert committed["inputs"]["audit_eig_replicates"] == 20

    # an OLD manifest (EIG cells with no SEs) must SystemExit with the instruction, not KeyError
    mod = _load_script()
    h, _ = mod.build_headline(_mock_res())
    entries = [e for e in mod.build_manifest_entries(h, ["C1", "C2", "C3", "C4"])
               if not e.id.startswith("se_eig") and e.id != "se_zero_eig"]
    monkeypatch.chdir(tmp_path)
    write_manifest(mod.DESIGN_MANIFEST, entries, mod._manifest_inputs(),
                   generated_by="scripts/generate_design.py")
    monkeypatch.setattr(mod, "compute", lambda *a, **k: _mock_res())
    with pytest.raises(SystemExit) as exc:
        mod.audit()
    assert "regenerate" in str(exc.value) and "eig_C1" in str(exc.value)


def test_zero_eig_band_is_floor_governed_and_skips_the_ratio_guard(tmp_path, monkeypatch, capsys):
    """The zero-EIG sanity cell has an identically-zero SE; the audit must handle it without a special case.

    Its band reduces to AUDIT_ATOL_FLOOR, the SE-ratio guard is not applicable (a ratio of two zeros is
    meaningless), and it must not be swept into the NOT INFORMATIVE list -- which would both mislabel a
    working sanity gate and divide by its zero value.
    """
    mod = _audit_against(tmp_path, monkeypatch, _mock_res(), _mock_res(eig_shift={"zero_eig": 0.005}))
    mod.audit()                                  # inside the 0.01 floor: passes, no exception
    out = capsys.readouterr().out
    assert "design audit OK" in out
    vacuity_report = out.split("NOT INFORMATIVE")[-1] if "NOT INFORMATIVE" in out else ""
    assert "zero_eig" not in vacuity_report, out

    mod = _audit_against(tmp_path, monkeypatch, _mock_res(), _mock_res(eig_shift={"zero_eig": 0.02}))
    with pytest.raises(SystemExit) as exc:      # outside it: fails on the floor, with the floor named
        mod.audit()
    assert "zero_eig" in str(exc.value) and "floor 0.01 abs" in str(exc.value)
