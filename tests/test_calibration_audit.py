"""The CALIBRATION --audit parser is generic (header-aware, extensible), and its per-quantity tolerance
CLASSES are pinned against a *silent* softening (the never-soften rule).

AMENDMENT 2026-08-13 -- `min ess` left the tolerance-audited set. It is a convergence DIAGNOSTIC, and a
symmetric relative band on one reds when a fresh realization converges BETTER than the committed one, i.e.
it measures the sampler's luck rather than the artifact's correctness. It is now gated ONE-SIDED against
`AUDIT_ESS_FLOOR`. The pins below therefore guard two things at once: the surviving bands against silent
widening (as before), and `min ess` against silently regaining a committed-vs-fresh band -- the
must-not-return direction, which is the one this amendment exists to hold.
"""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

import openmucf

_ROOT = Path(openmucf.__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "generate_calibration.py"


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
    as test_forecast.py::test_d6_constants_still_mirror_generate_calibration). Changing them
    requires deliberately editing this pin, plus a dated
    written note saying what was measured and why the band moved -- the never-soften rule."""
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


def test_the_shipped_document_actually_gets_the_floor():
    """A floor that silently stops applying is exactly as bad as the band coming back.

    Renaming the `min ess` column on BOTH sides, or dropping the convergence table, would remove the
    one-sided cells with nothing to show for it but a changed count in a CI log -- the pins above all
    guard the RULE and none of them guards its REACH. So pin the shipped document against the audit's
    own routing: exactly two floor cells, one per main chain, and both clearing the floor.
    """
    mod = _load_script()
    calib = Path(openmucf.__file__).resolve().parent.parent / "CALIBRATION.md"
    found = []
    for _title, header, rows in mod._parse_tables(calib.read_text(encoding="utf-8")):
        if header is None:
            continue
        for row in rows:
            for j, col in enumerate(header[1:], start=1):
                for (_label, kind), value in zip(
                    mod._cell_specs(col), mod._cell_floats(row[j]), strict=False
                ):
                    if kind == "ess_floor":
                        found.append((row[0], value))
    assert sorted(chain for chain, _ in found) == ["Kamimura", "weak"], found
    # and the verdict on the REAL cells does not depend on their committed values: a nonsense
    # committed value still passes, because only the fresh side is read.
    assert all(mod._within(0.0, v, "ess_floor") for _, v in found), found


def _ncc_pair(min_ess_fresh, min_ess_committed=5000.0):
    """A committed and a fresh NUTS cross-check record that differ only in min ess (the committed quadrature
    means are the record's own, so the 3-MCSE gate is met)."""
    mean = {q: 1.0 for q in ("omega_s_eff_pct", "lambda_c", "omega_s0_pct", "R", "W_pct", "X_mu", "L_pct")}
    rec = {"max_rhat": 1.001, "min_ess": min_ess_committed, "divergences": 0, "mean": mean,
           "mcse": dict.fromkeys(mean, 0.01), "max_abs_dev_over_mcse": 0.0}
    fresh = copy.deepcopy(rec)
    fresh["min_ess"] = min_ess_fresh
    quad = {q: {"mean": 1.0} for q in mean}
    return {"weak": rec}, {"weak": fresh}, {"weak": quad}


def test_audit_min_ess_floor_fails_below_and_passes_above():
    """The floor path: the verdict depends on the FRESH value only.

    A fresh run below the floor fails even when the committed value is identical; a fresh run above it passes
    even when the committed value is nowhere near -- what a relative band would get wrong.
    """
    mod = _load_script()
    committed, fresh, quad = _ncc_pair(1500.0, 1500.0)
    msg = " ".join(mod.nuts_audit_problems(committed, fresh, quad))
    assert "BELOW the structural floor 2000" in msg
    assert "rel." not in msg, "a one-sided cell must never fail with a relative-band message"
    for c, f in ((1e4, 5e3), (2.2e3, 1e4)):
        committed, fresh, quad = _ncc_pair(f, c)
        assert mod.nuts_audit_problems(committed, fresh, quad) == []


def test_nuts_audit_names_rhat_and_divergence_drift():
    mod = _load_script()
    committed, fresh, quad = _ncc_pair(5000.0)
    fresh["weak"]["max_rhat"] = 1.001 * 1.03          # outside the 2% class
    fresh["weak"]["divergences"] = 1
    probs = mod.nuts_audit_problems(committed, fresh, quad)
    assert any("max_rhat" in p for p in probs) and any("divergences" in p for p in probs)


# ------------------------------------------------------------------ the quadrature cells (tolerance rule)
def test_quadrature_tolerance_rule_pinned():
    """The caps, the floor and the scatter factor of the quadrature audit are pinned; the rule is
    tol = min(cap, max(factor x measured scatter, floor))."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert 'QUAD_TOL_CAP = {"moment": 1e-5, "quantile": 5e-3, "quantile_L": 2e-2}' in src
    assert "QUAD_TOL_FLOOR = 1e-6" in src
    assert "QUAD_TOL_SCATTER_FACTOR = 10.0" in src
    mod = _load_script()
    assert mod.quad_tolerances({"moment": 0.0, "quantile": 1e-8, "quantile_L": 1.0}) == {
        "moment": 1e-6, "quantile": 1e-6, "quantile_L": 2e-2}
    assert mod.quad_tolerances({"moment": 3e-7, "quantile": 2e-4, "quantile_L": 1e-3}) == pytest.approx(
        {"moment": 3e-6, "quantile": 2e-3, "quantile_L": 1e-2})


def _committed_cells():
    return json.loads((_ROOT / "CALIBRATION_CELLS.json").read_text(encoding="utf-8"))


def test_committed_tolerances_follow_the_recorded_scatter():
    mod = _load_script()
    cells = _committed_cells()
    tol = mod.committed_tolerances(cells)
    assert tol == mod.quad_tolerances(cells["audit_scatter"]["scatter"])
    cells["audit_scatter"]["tolerance"]["quantile"] *= 2
    with pytest.raises(SystemExit, match="not the rule"):
        mod.committed_tolerances(cells)


@pytest.mark.parametrize(
    "path, factor, named",
    [
        (("posteriors", "weak", "R", "q0.025"), 1 + 6e-3, True),        # beyond the 5e-3 quantile cap
        (("posteriors", "kamimura", "lambda_c", "mean"), 1 + 2e-5, True),  # beyond the 1e-5 moment cap
        (("posteriors", "weak", "L_pct", "q0.5"), 1 + 1e-7, False),    # inside every tolerance
    ],
)
def test_compare_cells_drill(path, factor, named):
    """Drill (a): a fresh cell perturbed past its class tolerance is named; one inside it is not."""
    mod = _load_script()
    committed = _committed_cells()
    fresh = copy.deepcopy(committed)
    node = fresh
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] *= factor
    problems, worst = mod.compare_cells(committed, fresh, mod.committed_tolerances(committed))
    assert bool([p for p in problems if p.startswith(".".join(path))]) is named
    assert set(worst) == {"moment", "quantile", "quantile_L"}


def test_compare_cells_rails_are_exact():
    mod = _load_script()
    committed = _committed_cells()
    fresh = copy.deepcopy(committed)
    fresh["sweep"][0]["rails"] = not fresh["sweep"][0]["rails"]
    problems, _ = mod.compare_cells(committed, fresh, mod.committed_tolerances(committed))
    assert any(p.startswith(f"sweep[{committed['sweep'][0]['config']}].rails") for p in problems)


def _in_copy(tmp_path, monkeypatch):
    """Run the script's file-reading checks in a tmp copy of the files they read."""
    for rel in ("CALIBRATION.md", "CALIBRATION_CELLS.json", "VALIDATION_CHANNELS.md",
                "forecasts/FC-001-mufuse.json", "openmucf/data/side_channel_inputs.csv",
                "openmucf/data/uq_priors.csv"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / rel, tmp_path / rel)
    mod = _load_script()
    shutil.copyfile(_ROOT / mod.NOTE_PATH, tmp_path / mod.NOTE_PATH)
    monkeypatch.chdir(tmp_path)
    return mod


def test_committed_documents_are_the_render_of_the_committed_cells(tmp_path, monkeypatch):
    mod = _in_copy(tmp_path, monkeypatch)
    assert mod.render_problems(mod.load()) == []
    assert mod.input_problems(mod.load()) == []


def test_render_check_names_a_changed_byte(tmp_path, monkeypatch):
    """Drill (b): CALIBRATION.md one byte off its render is named."""
    mod = _in_copy(tmp_path, monkeypatch)
    p = tmp_path / "CALIBRATION.md"
    t = p.read_text(encoding="utf-8")
    p.write_text(t.replace("Estimator:", "Estimator;", 1), encoding="utf-8")
    expected = ["CALIBRATION.md: differs from the render of the committed cells"]
    assert mod.render_problems(mod.load()) == expected


def test_input_digest_check_names_a_changed_input_and_render_refuses(tmp_path, monkeypatch):
    """Drill (d): the side-channel inputs with one byte changed -> the digest check names it, and `--render`
    refuses to write."""
    mod = _in_copy(tmp_path, monkeypatch)
    p = tmp_path / "openmucf/data/side_channel_inputs.csv"
    t = p.read_text(encoding="utf-8")
    p.write_text(t.replace("0.3157", "0.3158", 1), encoding="utf-8")
    monkeypatch.setattr(mod, "INPUT_FILES", {k: str(tmp_path / v) for k, v in mod.INPUT_FILES.items()})
    probs = mod.input_problems(mod.load())
    assert [x for x in probs if x.startswith("inputs.side_channel_inputs_csv_sha256")]
    before = (tmp_path / "CALIBRATION.md").read_bytes()
    with pytest.raises(SystemExit, match="render refused"):
        mod.main(["--render"])
    assert (tmp_path / "CALIBRATION.md").read_bytes() == before


def test_fc_note_raises_when_a_corrected_median_leaves_the_card_interval(tmp_path, monkeypatch):
    """Drill (FC note): the widened-support lambda_c median set to 1.01 x the card's 95 % upper bound."""
    mod = _in_copy(tmp_path, monkeypatch)
    cells = mod.load()
    card = mod._card_scenario_a()
    upper = card["lambda_c@phi=1.2"]["ci95"][1]
    cells["posteriors"]["fc001_widened_support"]["lambda_c"]["q0.5"] = 1.01 * upper
    with pytest.raises(ValueError, match="fc001_widened_support.lambda_c.q0.5"):
        mod.build_note(cells)


def test_sweep_lambda_c_labels_print_true_bounds():
    """Every sweep box renders through lc_label, parses back to the box exactly, and appears in the
    committed CALIBRATION.md."""
    mod = _load_script()
    md = (_ROOT / "CALIBRATION.md").read_text(encoding="utf-8")
    for box, _name in mod._LC_BOXES:
        label = mod.lc_label(box)
        lo, hi = (float(x) for x in label[len("lc["):-1].split(","))
        assert (lo, hi) == box, label
        assert label in md, label
