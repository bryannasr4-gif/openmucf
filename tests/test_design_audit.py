"""WS-D: the DESIGN.md audit wiring (WAVE2 A1).

DESIGN.md + DESIGN_MANIFEST.json are NUTS-derived and NOT byte-diffed; `generate_design.py --audit`
tolerance-checks them. Here we (a) pin the audit tolerances against a SILENT softening (the WAVE1 1.5
never-soften rule; same guard as test_calibration_audit.py), and (b) verify the doc<->manifest render
deterministically, carry the mandated verbatim paragraphs, and pass `provenance --check` -- all WITHOUT
running the NUTS pipeline (a fixed synthetic ``res`` exercises the pure rendering path).
"""

import importlib.util
from pathlib import Path

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


def _mock_res(*, resolved_flip: bool = False):
    """A fixed synthetic result matching compute()'s structure -- no NUTS, fully deterministic.

    ``resolved_flip`` toggles the class-contrast between "not resolved" (the honest default: a small
    paired delta against a comparable SE) and "resolved at >3 sigma", so BOTH branches of the derived
    class-flip prose are exercised without running NUTS.
    """
    cand_ids = ["C1", "C2", "C3", "C4"]
    eig_bits = {"C1": 1.629, "C2": 1.782, "C3": 1.081, "C4": 3.240}
    ose = {"C1": 0.231, "C2": 0.049, "C3": 0.560, "C4": -0.026}
    rc = {"C1": 0.070, "C2": 0.017, "C3": 0.101, "C4": 0.408}
    ri = {"C1": 0.002, "C2": 0.017, "C3": 0.103, "C4": 0.408}
    se = {"C1": 0.020, "C2": 0.012, "C3": 0.026, "C4": 0.021}
    sens = {"C1": True, "C2": False, "C3": True, "C4": False}
    delta_se = 0.004 if resolved_flip else 0.030
    return {
        "seed": 0,
        "registry": design.registry(42.95, 15.0),
        "cand_ids": cand_ids,
        "eig": {c: {"eig_bits": eig_bits[c], "n_outer": 256, "n_inner": 256} for c in cand_ids},
        "eig_c3_inflated": {"eig_bits": 2.492},
        "sdc": {c: {"ose_contraction": ose[c], "ose_contraction_se": se[c], "n_synth": 64,
                    "class_sensitive": sens[c],
                    "R_contraction": {"constant": rc[c], "inflated": ri[c]},
                    "R_contraction_se": {"constant": se[c], "inflated": se[c]},
                    "R_contraction_class_delta": {"value": rc[c] - ri[c], "se": delta_se, "paired": True}}
                for c in cand_ids},
        "zero_eig_bits": -1e-7,  # exercises the negative-zero normalisation
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
    new band cannot be widened without publishing a larger SE in DESIGN.md, and n_synth was raised 8 -> 64
    so the SEs -- hence the band -- actually shrank for every cell.
    """
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "AUDIT_RTOL_EIG = 0.05" in src
    assert "AUDIT_K_SIGMA = 4.0" in src
    assert "CLAIM_K_SIGMA = 3.0" in src
    assert "AUDIT_ATOL_FLOOR = 0.01" in src
    assert "AUDIT_MIN_SEPARATION_SIGMA = 3.0" in src
    assert "AUDIT_ATOL_CONTRACTION" not in src, "the superseded fixed band must not come back"


def test_n_synth_and_se_settings_pinned():
    """The Monte-Carlo resolution of the PRIMARY metric is pre-registered (openmucf.design)."""
    assert design.N_SYNTH_DEFAULT == 64
    assert design.SE_BOOTSTRAP == 10_000
    src = (Path(design.__file__)).read_text(encoding="utf-8")
    assert "N_SYNTH_DEFAULT = 64" in src


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

    # verbatim estimand-discipline paragraph (WAVE2 6.1)
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

    # write doc + manifest to a temp repo root; provenance must find every tracked value in the doc
    (tmp_path / "DESIGN.md").write_text(md1, encoding="utf-8")
    entries = mod.build_manifest_entries(h1, res["cand_ids"])
    write_manifest(tmp_path / "DESIGN_MANIFEST.json", entries, mod._manifest_inputs(),
                   generated_by="scripts/generate_design.py")
    failures = check_manifest(tmp_path / "DESIGN_MANIFEST.json", repo_root=tmp_path)
    assert failures == [], failures
    # 4x3 sd-contraction cells + 4x3 their MC SEs + 4 EIG + inflated + 2 sanity claims
    assert len(entries) >= 27


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
    assert "no R information that survives" in h_unres["class_flip_reading"].replace("\n", " ")

    assert h_res["class_flip"].startswith("RESOLVED for ")
    assert "C1" in h_res["class_flip"]
    assert "RESOLVED at >= 3 sigma" in h_res["class_flip_reading"]
    assert "collapses under R(phi)-inflation" in h_res["class_flip_reading"]

    # the sigma actually shown is |delta| / SE, not a narrative
    assert h_res["sigma_delta_C1"] == f"{abs(0.070 - 0.002) / 0.004:.1f}"
    assert h_unres["sigma_delta_C1"] == f"{abs(0.070 - 0.002) / 0.030:.1f}"


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
    assert any("no longer led by C3" in p for p in probs), probs
