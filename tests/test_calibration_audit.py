"""The CALIBRATION --audit parser is generic (header-aware, extensible), and its per-quantity tolerance
CLASSES are pinned against a *silent* softening (the never-soften rule).

AMENDMENT 2026-08-13 -- `min ess` left the tolerance-audited set. It is a convergence DIAGNOSTIC, and a
symmetric relative band on one reds when a fresh realization converges BETTER than the committed one, i.e.
it measures the sampler's luck rather than the artifact's correctness. It is now gated ONE-SIDED against
`AUDIT_ESS_FLOOR`. The pins below therefore guard two things at once: the surviving bands against silent
widening (as before), and `min ess` against silently regaining a committed-vs-fresh band -- the
must-not-return direction, which is the one this amendment exists to hold.
"""

import importlib.util
from pathlib import Path

import pytest

import openmucf

_SCRIPT = Path(openmucf.__file__).resolve().parent.parent / "scripts" / "generate_calibration.py"


def _load_script():
    """Import the script module by path (no MCMC runs: its work is guarded behind main())."""
    spec = importlib.util.spec_from_file_location("_gen_calibration", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_tables_generic_header_aware():
    """_parse_tables returns (title, header, rows) for an arbitrary number of tables/columns -- so a chain
    or column added later is picked up without editing the parser."""
    mod = _load_script()
    md = (
        "# header\n\n"
        "## Chain one\n| parameter | mean | sd | mcse | 95% CI |\n|---|---|---|---|---|\n"
        "| a | 1.0 | 0.1 | 0.01 | [0.8, 1.2] |\n| b | 2.0 | 0.2 | 0.02 | [1.6, 2.4] |\n\n"
        "## Convergence\n| chain | max r_hat | min ess | divergences |\n|---|---|---|---|\n"
        "| weak | 1.001 | 2500 | 0 |\n"
    )
    sections = mod._parse_tables(md)
    assert [t for t, _, _ in sections] == ["Chain one", "Convergence"]
    assert sections[0][1] == ["parameter", "mean", "sd", "mcse", "95% CI"]
    assert sections[0][2][0] == ["a", "1.0", "0.1", "0.01", "[0.8, 1.2]"]
    assert sections[1][1] == ["chain", "max r_hat", "min ess", "divergences"]


def test_cell_specs_class_map():
    """Column-name -> tolerance-class routing is the classifier the audit relies on."""
    mod = _load_script()
    assert mod._cell_specs("mean") == [("mean", "mean")]
    assert mod._cell_specs("R sd") == [("sd", "sd")]
    assert mod._cell_specs("mcse") == [("mcse", "mcse")]
    assert mod._cell_specs("min ess") == [("ess", "ess_floor")]   # one-sided floor, NOT a band (2026-08-13)
    assert mod._cell_specs("max r_hat") == [("r_hat", "rhat")]
    assert mod._cell_specs("corr") == [("corr", "corr")]
    assert mod._cell_specs("divergences") == [("divergences", "div")]
    assert mod._cell_specs("95% CI") == []       # intervals are not audited
    assert mod._cell_specs("boxes") == []
    assert mod._cell_specs("rails?") == []


def test_audit_tolerances_pinned():
    """Any *silent* softening of the audit tolerance CLASSES trips this test (same literal-substring guard
    as test_forecast.py::test_d6). Changing them requires deliberately editing this pin + a dated RESULTS
    note (never-soften rule)."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "AUDIT_RTOL_MEAN = 0.02" in src        # mean cells 2%
    assert "AUDIT_RTOL_SD = 0.08" in src          # sd cells 8%
    assert "AUDIT_RTOL_CORR = 0.08" in src        # corr cells 8%
    assert "AUDIT_RTOL_ESS_MCSE = 0.20" in src    # mcse cells 20% (ess left this class 2026-08-13)
    assert "AUDIT_RTOL_RHAT = 0.02" in src        # r_hat cells 2%
    assert "AUDIT_ATOL_DIVERGENCES = 0" in src    # divergences EXACT == 0
    assert "AUDIT_ESS_FLOOR = 2000" in src        # min ess: one-sided structural floor, never widened


def test_min_ess_can_never_regain_a_committed_vs_fresh_band():
    """The must-not-return guard (the AUDIT_RTOL_EIG precedent): re-banding a convergence diagnostic
    against a committed realization is the defect this re-registration removed, so it is blocked
    structurally -- in the class map, in the tolerance table, and in _rel_distance itself.
    """
    mod = _load_script()
    src = _SCRIPT.read_text(encoding="utf-8")

    # (a) the superseded routing must not come back, and no ess tolerance may be introduced
    assert '("ess", "ess_mcse")' not in src
    assert "AUDIT_RTOL_ESS =" not in src
    assert "ess_floor" not in mod._TOL, "membership in _TOL is what makes a committed value a target"

    # (b) the floor is ONE-SIDED and reads only the fresh value: an identical committed value passes and
    #     fails purely on where the FRESH one sits, and a better-converged fresh run can never red.
    assert mod._within(5000.0, 9000.0, "ess_floor") is True     # converged BETTER -> must not red
    assert mod._within(5000.0, 2000.0, "ess_floor") is True     # exactly at the floor -> passes
    assert mod._within(5000.0, 1999.0, "ess_floor") is False
    assert mod._within(2.0, 9000.0, "ess_floor") is True        # committed collapsed, fresh fine -> passes
    assert mod._within(9000.0, 2.0, "ess_floor") is False       # the collapse mode is still caught

    # (c) asking for a relative distance on a one-sided cell is a hard error, not a silent number
    with pytest.raises(AssertionError, match="not a relative-tolerance class"):
        mod._rel_distance(5000.0, 9000.0, "ess_floor")


def _audit_with(tmp_path, monkeypatch, committed_md, fresh_md):
    """Drive ``audit()`` over a crafted committed/fresh pair -- no MCMC, so what is exercised is exactly
    the parse -> classify -> compare path the real audit runs (the test_design_audit.py pattern)."""
    mod = _load_script()
    monkeypatch.chdir(tmp_path)
    Path(mod.CALIBRATION_MD).write_text(committed_md, encoding="utf-8")
    monkeypatch.setattr(mod, "_all", lambda: (None,) * 8)
    monkeypatch.setattr(mod, "build_md", lambda *a, **k: fresh_md)
    return mod


def _convergence_md(min_ess_cell):
    return ("# CALIBRATION\n\n## Convergence diagnostics (4 chains, sequential)\n"
            "| chain | max r_hat | min ess | divergences |\n|---|---|---|---|\n"
            f"| weak | 1.000 | {min_ess_cell} | 0 |\n")


def test_audit_min_ess_floor_fails_below_and_passes_above(tmp_path, monkeypatch, capsys):
    """The floor path end-to-end: the verdict depends on the FRESH value only.

    A fresh chain below the floor fails even when the committed cell is identical; a fresh chain above it
    passes even when the committed cell is nowhere near -- which is the whole point of the 2026-08-13
    re-registration, and what a relative band (>= 20% apart in both directions here) would get wrong.
    """
    # (a) fresh below the floor -> FAIL, and the message names the floor, not a band
    mod = _audit_with(tmp_path, monkeypatch, _convergence_md("1.5e+03"), _convergence_md("1.5e+03"))
    with pytest.raises(SystemExit) as exc:
        mod.audit()
    msg = str(exc.value)
    assert "BELOW the structural floor 2000" in msg
    assert "rel." not in msg, "a one-sided cell must never fail with a relative-band message"

    # (b) fresh above the floor -> PASS, with the committed cell far away in BOTH directions (50% and 78%
    #     relative distance, either of which the superseded 20% band would have red)
    for committed, fresh in (("1e+04", "5e+03"), ("2.2e+03", "1e+04")):
        mod = _audit_with(tmp_path, monkeypatch, _convergence_md(committed), _convergence_md(fresh))
        mod.audit()
        out = capsys.readouterr().out
        assert "calibration audit OK" in out
        assert "one-sided min-ess floor" in out
        assert "floor=2000" in out and "x the floor" in out    # the headroom is instrumented, not silent
