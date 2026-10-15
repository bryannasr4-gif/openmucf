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
import re
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


# ------------------------------------------------------------------ the FC-001 note's interval check
_FC_TARGETS = {"lambda_c": "lambda_c@phi=1.2", "omega_s_eff_pct": "omega_s_eff@phi=1.2"}


@pytest.mark.parametrize("name", ["fc001_card_support", "fc001_widened_support"])
@pytest.mark.parametrize("q", sorted(_FC_TARGETS))
@pytest.mark.parametrize("side, factor", [("lower", 0.99), ("upper", 1.01)])
def test_fc_note_names_each_median_outside_the_card_interval(tmp_path, monkeypatch, name, q, side, factor):
    """Drill (FC note, every member): each corrected scenario-A median moved 1 % past either end of the card's
    95 % interval is refused by name, and `--render` then writes neither document."""
    mod = _in_copy(tmp_path, monkeypatch)
    cells = mod.load()
    lo, hi = mod._card_scenario_a()[_FC_TARGETS[q]]["ci95"]
    cells["posteriors"][name][q]["q0.5"] = factor * (lo if side == "lower" else hi)
    with pytest.raises(ValueError, match=f"{name}.{q}.q0.5 = .* outside the card's 95 % interval"):
        mod.build_note(cells)
    for run in cells["convergence"]["runs"][name].values():     # the convergence record stays bound
        run[q]["q0.5"] = cells["posteriors"][name][q]["q0.5"]
    assert mod.convergence_problems(cells, mod.committed_tolerances(cells)) == []
    mod.dump(cells, mod.CELLS_JSON)
    for path in (mod.CALIBRATION_MD, mod.NOTE_PATH):
        (tmp_path / path).write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"{name}.{q}.q0.5"):
        mod.main(["--render"])
    for path in (mod.CALIBRATION_MD, mod.NOTE_PATH):
        assert (tmp_path / path).read_text(encoding="utf-8") == "sentinel\n", path


def test_fc_note_accepts_the_committed_medians(tmp_path, monkeypatch):
    mod = _in_copy(tmp_path, monkeypatch)
    assert "Every corrected median lies inside the card's own 95 % interval." in mod.build_note(mod.load())


# ------------------------------------------------------------------ the sweep's railing screen
_RAIL_BOXES = {"omega_s0_pct": (0.50, 1.20), "R": (0.00, 0.80), "lambda_c": (0.8e8, 2.0e8)}


def _interior_post(boxes):
    """95 % intervals at 20 % and 80 % of each box: far from every bound."""
    return {k: {"q0.025": lo + 0.2 * (hi - lo), "q0.975": lo + 0.8 * (hi - lo)}
            for k, (lo, hi) in boxes.items()}


def test_rails_is_false_away_from_every_bound():
    assert _load_script()._rails(_interior_post(_RAIL_BOXES), _RAIL_BOXES) is False


@pytest.mark.parametrize("param", sorted(_RAIL_BOXES))
@pytest.mark.parametrize("end", ["q0.025", "q0.975"])
@pytest.mark.parametrize("frac, railed", [(0.005, True), (0.015, False)])
def test_rails_flags_one_endpoint_within_one_percent_of_its_own_bound(param, end, frac, railed):
    """Each of the six interval endpoints alone, 0.5 % of its box width from its bound, rails; at 1.5 % it
    does not (the screen's 1 % criterion)."""
    mod = _load_script()
    post = _interior_post(_RAIL_BOXES)
    lo, hi = _RAIL_BOXES[param]
    post[param][end] = lo + frac * (hi - lo) if end == "q0.025" else hi - frac * (hi - lo)
    assert mod._rails(post, _RAIL_BOXES) is railed


def test_sweep_rows_carry_the_railing_screen(monkeypatch):
    """The sweep writes `rails` from the screen: a quadrature whose R interval reaches one config's R bound
    rails that config alone."""
    mod = _load_script()
    target = (tuple(mod._R_BOXES[0][0]), tuple(mod._OS0_BOXES[0][0]), tuple(mod._LC_BOXES[0][0]))

    def fake_quadrature(prior, *, R_box, lam_box, cov, res):
        boxes = {"omega_s0_pct": tuple(prior[1:3]), "R": tuple(R_box), "lambda_c": tuple(lam_box)}
        post = _interior_post(boxes)
        if (tuple(R_box), tuple(prior[1:3]), tuple(lam_box)) == target:
            post["R"]["q0.025"] = R_box[0] + 0.005 * (R_box[1] - R_box[0])
        post.update(corr_omega_s0_R=0.5, omega_s_eff_pct={"mean": 0.45, "sd": 0.05})
        post["R"]["sd"] = 0.1
        return post

    monkeypatch.setattr(mod.calibrate, "posterior_quadrature", fake_quadrature)
    rows = mod._sweep_rows()
    assert len(rows) == len(mod._R_BOXES) * len(mod._OS0_BOXES) * len(mod._LC_BOXES)
    railed = [r["config"] for r in rows if r["rails"]]
    assert railed == [f"R={mod._R_BOXES[0][1]}/os0={mod._OS0_BOXES[0][1]}/lc={mod._LC_BOXES[0][1]}"]


# ------------------------------------------------------------------ the audit, through its entry point
_CARRIED = ("shipped_v1_3_0", "convergence", "audit_scatter", "nuts_cross_check")


def _fresh_and_ncc(mod):
    """The committed cells and NUTS record as a fresh run would return them: every audit check passes."""
    committed = mod.load()
    fresh = {k: copy.deepcopy(v) for k, v in committed.items() if k not in _CARRIED}
    return fresh, copy.deepcopy(committed["nuts_cross_check"])


def _audit_with(mod, monkeypatch, fresh, ncc):
    monkeypatch.setattr(mod, "compute_cells", lambda posts=None: copy.deepcopy(fresh))
    monkeypatch.setattr(mod, "nuts_cross_check", lambda posteriors: (copy.deepcopy(ncc), None))
    mod.audit()


def test_audit_accepts_the_committed_record(tmp_path, monkeypatch, capsys):
    mod = _in_copy(tmp_path, monkeypatch)
    fresh, ncc = _fresh_and_ncc(mod)
    _audit_with(mod, monkeypatch, fresh, ncc)
    assert "calibration audit OK" in capsys.readouterr().out


def _drift_input(mod, tmp_path, fresh, ncc):
    p = tmp_path / "openmucf/data/side_channel_inputs.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("0.3157", "0.3158", 1), encoding="utf-8")
    return "inputs.side_channel_inputs_csv_sha256: recorded"


def _unknown_input(mod, tmp_path, fresh, ncc):
    cells = mod.load()
    cells["inputs"]["extra_csv_sha256"] = "0" * 64
    mod.dump(cells, mod.CELLS_JSON)
    fresh["inputs"] = cells["inputs"]
    return "inputs.extra_csv_sha256: no such input"


def _drift_render(mod, tmp_path, fresh, ncc):
    p = tmp_path / "CALIBRATION.md"
    p.write_text(p.read_text(encoding="utf-8").replace("Estimator:", "Estimator;", 1), encoding="utf-8")
    return "CALIBRATION.md: differs from the render of the committed cells"


def _drift_cell(mod, tmp_path, fresh, ncc):
    fresh["posteriors"]["weak"]["R"]["q0.025"] *= 1 + 6e-3
    return "posteriors.weak.R.q0.025: committed"


def _drop_cells(mod, tmp_path, fresh, ncc):
    del fresh["posteriors"]["weak"]
    return "fresh: 50 scalar(s) missing"


def _drift_nuts(mod, tmp_path, fresh, ncc):
    ncc["kamimura"]["divergences"] = 1
    return "nuts_cross_check.kamimura.divergences: fresh 1"


@pytest.mark.parametrize("corrupt", [_drift_input, _unknown_input, _drift_render, _drift_cell, _drop_cells,
                                     _drift_nuts])
def test_audit_refuses_each_problem_alone(tmp_path, monkeypatch, capsys, corrupt):
    """Drill (audit stop): with every other check passing, one input, render, cell or NUTS problem alone stops
    `--audit` with that problem named, and nothing prints success."""
    mod = _in_copy(tmp_path, monkeypatch)
    fresh, ncc = _fresh_and_ncc(mod)
    named = corrupt(mod, tmp_path, fresh, ncc)
    with pytest.raises(SystemExit) as exc:
        _audit_with(mod, monkeypatch, fresh, ncc)
    assert str(exc.value).startswith("calibration audit FAILED")
    assert named in str(exc.value)
    assert "audit OK" not in capsys.readouterr().out


# ------------------------------------------------------------------ the compared and recorded cell sets
_POSTERIOR_NAMES = ("fc001_card_support", "fc001_widened_support", "kamimura", "l1_default", "l2_default",
                    "weak")
_OMISSIONS = ([("posteriors", n) for n in _POSTERIOR_NAMES]
              + [("q_net", k) for k in ("mean", "sd", "q0.025", "q0.5", "q0.975")]
              + [("legacy_tension", k) for k in ("x_mu", "z_sigma_at_10", "z_sigma_at_12")])


def test_omission_homes_are_every_compared_group():
    """The omission drills below cover every group `quad_cells` reads: each posterior, q_net stat and legacy
    tension key of the committed cells."""
    cells = _committed_cells()
    homes = {(g, k) for g in ("posteriors", "q_net", "legacy_tension") for k in cells[g]}
    assert homes == set(_OMISSIONS)


@pytest.mark.parametrize("group, key", _OMISSIONS)
@pytest.mark.parametrize("where", ["fresh", "committed"])
def test_compare_cells_names_a_missing_group(group, key, where):
    """Drill (compared coverage): a cell group absent on either side is named, never skipped."""
    mod = _load_script()
    committed = _committed_cells()
    fresh = copy.deepcopy(committed)
    del (fresh if where == "fresh" else committed)[group][key]
    problems, _ = mod.compare_cells(committed, fresh, mod.committed_tolerances(_committed_cells()))
    gone = f"posteriors.{key}.L_pct.mean" if group == "posteriors" else f"{group}.{key}"
    assert [p for p in problems if p.startswith(f"{where}: ") and "missing" in p and gone in p], problems


def test_input_check_names_an_unknown_recorded_input_and_render_refuses(tmp_path, monkeypatch, capsys):
    """Drill (unknown input): a digest recorded for a file the calibration does not read is named, and
    `--render` refuses without writing."""
    mod = _in_copy(tmp_path, monkeypatch)
    cells = mod.load()
    cells["inputs"]["extra_csv_sha256"] = "0" * 64
    assert mod.input_problems(cells) == ["inputs.extra_csv_sha256: no such input"]
    mod.dump(cells, mod.CELLS_JSON)
    (tmp_path / "CALIBRATION.md").write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="render refused: inputs.extra_csv_sha256: no such input"):
        mod.main(["--render"])
    assert (tmp_path / "CALIBRATION.md").read_text(encoding="utf-8") == "sentinel\n"
    assert "rendered" not in capsys.readouterr().out


# ------------------------------------------------------------------ regeneration, through its entry point
def _regenerate_with(mod, monkeypatch, ncc):
    committed = mod.load()
    fresh = {k: copy.deepcopy(v) for k, v in committed.items() if k not in _CARRIED}
    figures = []
    posts = committed["posteriors"]
    monkeypatch.setattr(mod, "_posterior", lambda name, res=None, **kw: copy.deepcopy(posts[name]))
    monkeypatch.setattr(mod, "compute_cells", lambda posts=None: copy.deepcopy(fresh))
    monkeypatch.setattr(mod, "nuts_cross_check", lambda posteriors: (copy.deepcopy(ncc), {"draws": None}))
    monkeypatch.setattr(mod, "_write_figure", figures.append)
    mod.regenerate()
    return figures


def _written(tmp_path, mod):
    return {p: (tmp_path / p).read_bytes() for p in (mod.CELLS_JSON, mod.CALIBRATION_MD, mod.NOTE_PATH)}


def test_regenerate_writes_the_committed_record_from_the_committed_cells(tmp_path, monkeypatch, capsys):
    """Control: a passing cross-check writes the cells, both documents and the figure, and the documents are
    the committed ones."""
    mod = _in_copy(tmp_path, monkeypatch)
    committed = mod.load()
    docs = {p: (tmp_path / p).read_text(encoding="utf-8") for p in (mod.CALIBRATION_MD, mod.NOTE_PATH)}
    figures = _regenerate_with(mod, monkeypatch, committed["nuts_cross_check"])
    assert figures == [{"draws": None}]
    assert mod.load() == committed
    for p, text in docs.items():
        assert (tmp_path / p).read_text(encoding="utf-8") == text, p
    assert "wrote CALIBRATION_CELLS.json" in capsys.readouterr().out


@pytest.mark.parametrize("chain, key, value, named", [
    ("weak", "divergences", 1, "divergences 1 != 0"),
    ("kamimura", "max_rhat", 1.01, "max_rhat 1.0100 >= 1.01"),
    ("weak", "min_ess", 400, "min_ess 400 <= 400"),
    ("kamimura", "mean.lambda_c", 3.5, "lambda_c mean 3.50 MCSE from the quadrature mean"),
])
def test_regenerate_stops_on_a_failed_cross_check_before_any_write(tmp_path, monkeypatch, capsys, chain, key,
                                                                    value, named):
    """Drill (regeneration stop): one failed NUTS gate member stops regeneration, named, before the cells, the
    documents or the figure are written."""
    mod = _in_copy(tmp_path, monkeypatch)
    committed = mod.load()
    ncc = copy.deepcopy(committed["nuts_cross_check"])
    if key == "mean.lambda_c":
        q = committed["posteriors"][chain]["lambda_c"]["mean"]
        ncc[chain]["mean"]["lambda_c"] = q + value * ncc[chain]["mcse"]["lambda_c"]
    else:
        ncc[chain][key] = value
    before = _written(tmp_path, mod)
    with pytest.raises(SystemExit, match="NUTS cross-check FAILED") as exc:
        _regenerate_with(mod, monkeypatch, ncc)
    assert named in str(exc.value)
    assert _written(tmp_path, mod) == before
    assert not (tmp_path / "figures").exists()
    assert "wrote" not in capsys.readouterr().out


def test_regenerate_refuses_an_fc_note_failure_before_any_write(tmp_path, monkeypatch):
    """A corrected median outside the card interval stops regeneration before the cells file is written."""
    mod = _in_copy(tmp_path, monkeypatch)
    committed = mod.load()
    upper = mod._card_scenario_a()["lambda_c@phi=1.2"]["ci95"][1]
    for run in committed["convergence"]["runs"]["fc001_card_support"].values():   # the record stays bound
        run["lambda_c"]["q0.5"] = 1.01 * upper
    mod.dump(committed, mod.CELLS_JSON)

    def fresh_with_moved_median(posts=None):
        fresh = {k: copy.deepcopy(v) for k, v in committed.items() if k not in _CARRIED}
        fresh["posteriors"]["fc001_card_support"]["lambda_c"]["q0.5"] = 1.01 * upper
        return fresh

    before = _written(tmp_path, mod)
    monkeypatch.setattr(mod, "_posterior", lambda name, res=None, **kw: None)
    monkeypatch.setattr(mod, "compute_cells", fresh_with_moved_median)
    ncc = committed["nuts_cross_check"]
    monkeypatch.setattr(mod, "nuts_cross_check", lambda p: (copy.deepcopy(ncc), None))
    monkeypatch.setattr(mod, "_write_figure", lambda draws: pytest.fail("figure written"))
    with pytest.raises(ValueError, match="fc001_card_support.lambda_c.q0.5"):
        mod.regenerate()
    assert _written(tmp_path, mod) == before


# ------------------------------------------------------------------ the measurement records
def test_expected_scalars_are_what_the_cells_carry():
    """The scalar set the recorders require is the committed cells' own, and its size follows from the
    definitions: per posterior every quantity x stat plus the correlation, four per sweep config, the Q_net
    stats and the legacy-tension keys."""
    mod = _load_script()
    cells = _committed_cells()
    want = mod.expected_quad_paths()
    assert {(p, c) for p, _, c in mod.quad_cells(cells)} == want
    n_sweep = len(mod._R_BOXES) * len(mod._OS0_BOXES) * len(mod._LC_BOXES)
    per_post = len(mod.calibrate.QUAD_QUANTITIES) * len(mod.calibrate.STATS) + 1
    assert len(want) == (len(mod.POSTERIORS) * per_post + 4 * n_sweep + len(mod.Q_NET_STATS)
                         + len(mod.legacy_tension()))
    assert sorted(cells["q_net"]) == sorted(mod.Q_NET_STATS)


def test_q_net_cells_write_the_declared_stats():
    mod = _load_script()
    post = mod.calibrate.posterior_quadrature(mod.calibrate.WEAK_OMEGA_S0_PRIOR, res=0.1, bins=0.1)
    assert sorted(mod.q_net_cells(post, n=8)) == sorted(mod.Q_NET_STATS)


def _fake_io(mod, monkeypatch, cells, files):
    """`load()` -> ``cells``; `load(path)` -> ``files[path]``; `dump` records instead of writing."""
    written = []
    monkeypatch.setattr(mod, "load", lambda path=None: cells if path is None else files[path])
    monkeypatch.setattr(mod, "dump", lambda obj, path: written.append((path, copy.deepcopy(obj))))
    return written


def _genuine_runs(cells, mod):
    """The four convergence runs as `--converge` writes them, from the record the committed cells carry."""
    conv = cells["convergence"]
    return {n: {"name": n, "res_pub": conv["res_pub"], "res_fine": conv["res_fine"],
                "pub": copy.deepcopy(conv["runs"][n]["pub"]), "fine": copy.deepcopy(conv["runs"][n]["fine"])}
            for n in mod.CONVERGENCE_POSTERIORS}


def test_record_convergence_reproduces_the_committed_record(monkeypatch, capsys):
    """Control: the genuine runs re-record exactly the committed convergence record."""
    mod = _load_script()
    cells = _committed_cells()
    committed = copy.deepcopy(cells["convergence"])
    del cells["convergence"]
    runs = _genuine_runs(_committed_cells(), mod)
    written = _fake_io(mod, monkeypatch, cells, runs)
    mod.record_convergence(list(runs))
    assert [p for p, _ in written] == [mod.CELLS_JSON]
    assert written[0][1]["convergence"] == committed
    assert mod.convergence_problems(written[0][1], mod.committed_tolerances(written[0][1])) == []


@pytest.mark.parametrize("res_pub, res_fine, named", [
    (1.0, 1.0, "res_fine 1.0 is not twice the publication resolution"),
    (2.0, 4.0, "res_pub 2.0 is not the publication resolution 1"),
    (1.0, 3.0, "res_fine 3.0 is not twice the publication resolution"),
])
def test_record_convergence_refuses_a_run_off_the_resolution_pair(monkeypatch, res_pub, res_fine, named):
    """Drill (resolution metadata, each posterior): a run whose fine resolution is not twice the publication
    resolution, or whose publication resolution is not the estimator's, is refused by name and nothing is
    written -- however small its recorded changes."""
    mod = _load_script()
    cells = _committed_cells()
    for name in mod.CONVERGENCE_POSTERIORS:
        runs = _genuine_runs(cells, mod)
        runs[name]["res_pub"], runs[name]["res_fine"] = res_pub, res_fine
        written = _fake_io(mod, monkeypatch, copy.deepcopy(cells), runs)
        with pytest.raises(SystemExit, match=f"{name}: {named}"):
            mod.record_convergence(list(runs))
        assert written == []


def _members(mod):
    return [(q, st) for q in mod.calibrate.QUAD_QUANTITIES for st in mod.calibrate.STATS]


def test_record_convergence_refuses_runs_of_other_posteriors(monkeypatch):
    """Drill (binding, all 4 x 7 x 7 members): a run whose publication and fine posteriors both moved 10 % in
    one member -- converged, but not the posterior the cells hold -- is refused naming that member, nothing
    written."""
    mod = _load_script()
    cells = _committed_cells()
    admitted = []
    for name in mod.CONVERGENCE_POSTERIORS:
        for q, st in _members(mod):
            runs = _genuine_runs(cells, mod)
            for side in ("pub", "fine"):
                runs[name][side][q][st] *= 1.1
            written = _fake_io(mod, monkeypatch, copy.deepcopy(cells), runs)
            try:
                mod.record_convergence(list(runs))
                admitted.append(f"{name}.{q}.{st}")
            except SystemExit as e:
                if f"{name}.{q}.{st}: run" not in str(e) or written:
                    admitted.append(f"{name}.{q}.{st} (not named, or written)")
    assert admitted == [], admitted
    assert len(mod.CONVERGENCE_POSTERIORS) * len(_members(mod)) == 196


def test_consumers_refuse_a_record_of_other_posteriors():
    """Drill (binding at consumption, all 196 members): a record carrying the runs of other posteriors --
    written before the cells existed -- is refused by the check the audit, the render and the regeneration
    call."""
    mod = _load_script()
    cells = _committed_cells()
    tol = mod.committed_tolerances(cells)
    admitted = []
    for name in mod.CONVERGENCE_POSTERIORS:
        for q, st in _members(mod):
            bad = copy.deepcopy(cells)
            for side in ("pub", "fine"):
                bad["convergence"]["runs"][name][side][q][st] *= 1.1
            if not [p for p in mod.convergence_problems(bad, tol) if f"{name}.{q}.{st}: run" in p]:
                admitted.append(f"{name}.{q}.{st}")
    assert admitted == [], admitted


@pytest.mark.parametrize("corrupt, named", [
    (lambda c: c["convergence"]["posteriors"]["kamimura"].update(moment=0.0), "kamimura: recorded maxima"),
    (lambda c: c["convergence"]["runs"].pop("weak"), "convergence: runs"),
    (lambda c: c["convergence"].pop("runs"), "convergence: runs"),
    (lambda c: c["convergence"].update(res_fine=1.0), "convergence.res_fine 1.0 is not twice"),
    (lambda c: c["estimator"].update(res=0.5), "estimator res 0.5"),
    (lambda c: c["convergence"]["criteria"].update(moment=1e-5), "convergence.criteria"),
    (lambda c: c["convergence"]["runs"]["weak"]["fine"]["R"].update(sd=float("nan")),
     "weak.fine.R.sd: nan is not a finite number"),
    (lambda c: c["convergence"]["runs"]["kamimura"]["pub"]["X_mu"].pop("q0.5"),
     "kamimura.pub.X_mu.q0.5: None is not a finite number"),
    (lambda c: c.pop("convergence"), "convergence: no record"),
    (lambda c: c["posteriors"].pop("weak"), "convergence: weak: the cells hold no 'weak' posterior"),
])
def test_convergence_check_names_each_defect(corrupt, named):
    mod = _load_script()
    cells = _committed_cells()
    assert mod.convergence_problems(cells, mod.committed_tolerances(cells)) == []
    corrupt(cells)
    assert [p for p in mod.convergence_problems(cells, mod.committed_tolerances(_committed_cells()))
            if named in p], mod.convergence_problems(cells, mod.committed_tolerances(_committed_cells()))


def _unbind(mod):
    cells = mod.load()
    cells["convergence"]["runs"]["weak"]["pub"]["lambda_c"]["mean"] *= 1.1
    cells["convergence"]["runs"]["weak"]["fine"]["lambda_c"]["mean"] *= 1.1
    mod.dump(cells, mod.CELLS_JSON)
    return cells


def test_audit_refuses_a_record_of_other_posteriors(tmp_path, monkeypatch, capsys):
    mod = _in_copy(tmp_path, monkeypatch)
    fresh, ncc = _fresh_and_ncc(mod)
    _unbind(mod)
    with pytest.raises(SystemExit, match="convergence: weak.lambda_c.mean: run"):
        _audit_with(mod, monkeypatch, fresh, ncc)
    assert "audit OK" not in capsys.readouterr().out


def test_render_refuses_a_record_of_other_posteriors(tmp_path, monkeypatch):
    mod = _in_copy(tmp_path, monkeypatch)
    _unbind(mod)
    (tmp_path / "CALIBRATION.md").write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="render refused: convergence: weak.lambda_c.mean: run"):
        mod.main(["--render"])
    assert (tmp_path / "CALIBRATION.md").read_text(encoding="utf-8") == "sentinel\n"


def test_regenerate_refuses_a_record_of_other_posteriors_before_sampling(tmp_path, monkeypatch):
    mod = _in_copy(tmp_path, monkeypatch)
    _unbind(mod)
    before = _written(tmp_path, mod)
    monkeypatch.setattr(mod, "nuts_cross_check", lambda p: pytest.fail("sampled"))
    with pytest.raises(SystemExit, match="regeneration refused: .*\n.*weak.lambda_c.mean: run"):
        cells = mod.load()
        posts = cells["posteriors"]
        fresh = {k: copy.deepcopy(v) for k, v in cells.items() if k not in _CARRIED}
        monkeypatch.setattr(mod, "_posterior", lambda name, res=None, **kw: copy.deepcopy(posts[name]))
        monkeypatch.setattr(mod, "compute_cells", lambda p=None: copy.deepcopy(fresh))
        mod.regenerate()
    assert _written(tmp_path, mod) == before


# --- the scatter record
def _set(rec, path, value):
    """Set the scalar at a quad_cells path ('posteriors.weak.R.sd', 'sweep[cfg].corr', 'q_net.sd', ...)."""
    if path.startswith("sweep["):
        cfg, key = path[len("sweep["):].rsplit("].", 1)
        row = [r for r in rec["sweep"] if r["config"] == cfg]
        assert len(row) == 1
        old, row[0][key] = row[0][key], value
        return old
    group, rest = path.split(".", 1)
    node, last = rec[group], rest
    if group == "posteriors":
        name, rest = rest.split(".", 1)
        node, last = rec["posteriors"][name], rest
        if rest != "corr_omega_s0_R":
            q, last = rest.split(".", 1)
            node = node[q]
    old, node[last] = node[last], value
    return old


def test_record_scatter_measures_a_known_scatter(monkeypatch):
    """Control: a complete, finite other-platform record that differs in one moment cell by 1e-12 relative
    records exactly that scatter and the floor tolerance."""
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    other["posteriors"]["weak"]["X_mu"]["mean"] *= 1 + 1e-12
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    mod.record_scatter("other")
    rec = written[0][1]["audit_scatter"]
    assert rec["scatter"]["moment"] == pytest.approx(1e-12, rel=1e-3)
    assert rec["scatter"]["quantile"] == rec["scatter"]["quantile_L"] == 0.0
    assert rec["tolerance"] == {"moment": 1e-6, "quantile": 1e-6, "quantile_L": 1e-6}


@pytest.mark.parametrize("group, key", _OMISSIONS)
@pytest.mark.parametrize("side", ["this", "other"])
def test_record_scatter_refuses_a_missing_group(monkeypatch, group, key, side):
    """Drill (scatter coverage, 14 homes x both records): a missing group is refused, never measured as
    zero."""
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    del (cells if side == "this" else other)[group][key]
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    with pytest.raises(SystemExit, match=f"{'this' if side == 'this' else 'the other'} platform: .*missing"):
        mod.record_scatter("other")
    assert written == []


def test_record_scatter_refuses_records_missing_the_same_groups(monkeypatch):
    """Both records without any posterior: equal sets, but 80 of the 380 scalars -- refused."""
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    cells["posteriors"], other["posteriors"] = {}, {}
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    with pytest.raises(SystemExit, match="300 scalar\\(s\\) missing"):
        mod.record_scatter("other")
    assert written == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "+inf", "-inf"])
@pytest.mark.parametrize("side", ["this", "other"])
def test_record_scatter_refuses_a_nonfinite_scalar_anywhere(monkeypatch, side, value):
    """Drill (scatter validity, every one of the scalars on both records): a non-finite measurement is refused
    by path, never reduced into a zero scatter."""
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    target = cells if side == "this" else other
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    paths = sorted(p for p, _ in mod.expected_quad_paths())
    admitted = []
    for path in paths:
        old = _set(target, path, value)
        try:
            mod.record_scatter("other")
            admitted.append(path)
        except SystemExit as e:
            if f"{path} = {value!r} is not a finite number" not in str(e):
                admitted.append(f"{path} (not named)")
        finally:
            _set(target, path, old)
    assert admitted == [] and written == [], admitted[:5]
    assert len(paths) == len(mod.quad_cells(_committed_cells()))


# ------------------------------------------------------------------ admission: the raw record first
# Every reader of a cells record (the scatter recorder, the audit's comparison, the renderer, the
# regeneration) admits the raw record first: the sweep's rows by config before anything is keyed by config,
# the measured scalars against the paths the definitions give, the stored runs, scatter and NUTS records
# member by member.
# Each matrix below is enumerated from the script's own definitions, and asserts its size, so it cannot pass
# by shrinking.
_SWEEP_MEMBERS = ("corr", "R_sd", "ose_mean", "ose_sd", "rails", "R_box", "os0_box", "lc_box")


def _perturbed(row, field):
    dup = copy.deepcopy(row)
    if field == "rails":
        dup[field] = not dup[field]
    elif isinstance(dup[field], list):
        dup[field] = [0.8 * dup[field][0], 1.2 * dup[field][1]]
    else:
        dup[field] *= 1.1
    return dup


def _repeat_row(rec, idx, field, order):
    """Insert a copy of sweep row ``idx`` with ``field`` changed, before it ("first") or after it ("last")."""
    extra = _perturbed(rec["sweep"][idx], field)
    rec["sweep"].insert(idx if order == "first" else idx + 1, extra)
    return extra["config"]


def test_sweep_definitions_are_not_vacuous():
    """The grid has one config per box triple, the row fields are the eight rendered members plus the config,
    and the committed record passes every admission check."""
    mod = _load_script()
    grid = mod._sweep_grid()
    assert len(grid) == len(mod._R_BOXES) * len(mod._OS0_BOXES) * len(mod._LC_BOXES)
    assert sorted(mod.SWEEP_FIELDS) == sorted(("config", *_SWEEP_MEMBERS))
    cells = _committed_cells()
    assert [r["config"] for r in cells["sweep"]] == list(grid)
    assert mod.sweep_problems(cells["sweep"]) == []
    assert mod.record_problems(cells, "cells") == []
    assert mod.nuts_record_problems(cells) == []
    assert mod.committed_problems(cells) == []
    assert mod.scatter_record_problems(cells["audit_scatter"]) == []


def test_sweep_rows_pass_the_sweep_check(monkeypatch):
    """What `_sweep_rows` computes is what `sweep_problems` admits."""
    mod = _load_script()

    def fake_quadrature(prior, *, R_box, lam_box, cov, res):
        boxes = {"omega_s0_pct": tuple(prior[1:3]), "R": tuple(R_box), "lambda_c": tuple(lam_box)}
        post = _interior_post(boxes)
        post.update(corr_omega_s0_R=0.5, omega_s_eff_pct={"mean": 0.45, "sd": 0.05})
        post["R"]["sd"] = 0.1
        return post

    monkeypatch.setattr(mod.calibrate, "posterior_quadrature", fake_quadrature)
    assert mod.sweep_problems(mod._sweep_rows()) == []


def test_record_scatter_refuses_a_repeated_sweep_config(monkeypatch):
    """Repeated identity, scatter: every config, every rendered member, either record, the changed
    copy before or after the original -- refused naming the config, nothing written. Keyed by config, the copy
    placed first would vanish behind the original and measure a zero scatter."""
    mod = _load_script()
    base = _committed_cells()
    admitted, n = [], 0
    for side in ("this", "other"):
        for order in ("first", "last"):
            for idx in range(len(base["sweep"])):
                for field in _SWEEP_MEMBERS:
                    cells, other = copy.deepcopy(base), copy.deepcopy(base)
                    cfg = _repeat_row(cells if side == "this" else other, idx, field, order)
                    written = _fake_io(mod, monkeypatch, cells, {"other": other})
                    n += 1
                    try:
                        mod.record_scatter("other")
                        admitted.append((side, order, cfg, field))
                    except SystemExit as e:
                        if f"sweep[{cfg}]: config repeated on 2 rows" not in str(e) or written:
                            admitted.append((side, order, cfg, field, "not named or written"))
    assert admitted == [], admitted[:5]
    assert n == 2 * 2 * len(mod._sweep_grid()) * len(_SWEEP_MEMBERS)


def test_compare_cells_refuses_a_repeated_sweep_config():
    """The same matrix against the audit's comparison, on either side."""
    mod = _load_script()
    base = _committed_cells()
    tol = mod.committed_tolerances(base)
    admitted, n = [], 0
    for where in ("committed", "fresh"):
        for order in ("first", "last"):
            for idx in range(len(base["sweep"])):
                for field in _SWEEP_MEMBERS:
                    committed, fresh = copy.deepcopy(base), copy.deepcopy(base)
                    cfg = _repeat_row(committed if where == "committed" else fresh, idx, field, order)
                    problems, _ = mod.compare_cells(committed, fresh, tol)
                    n += 1
                    if not [p for p in problems if p == f"{where}: sweep[{cfg}]: config repeated on 2 rows"]:
                        admitted.append((where, order, cfg, field))
    assert admitted == [], admitted[:5]
    assert n == 2 * 2 * len(mod._sweep_grid()) * len(_SWEEP_MEMBERS)


_SWEEP_READERS = ("_sweep_now", "_corr_range", "_sweep_section", "quad_cells", "_exact_cells", "build_md")


@pytest.mark.parametrize("reader", _SWEEP_READERS)
@pytest.mark.parametrize("order", ["first", "last"])
def test_every_sweep_reader_refuses_a_repeated_config(reader, order):
    """Each function that keys or renders the sweep reads it through `checked_sweep`: a repeated config
    raises, naming it, for every config."""
    mod = _load_script()
    base = _committed_cells()
    for idx in range(len(base["sweep"])):
        cells = copy.deepcopy(base)
        cfg = _repeat_row(cells, idx, "ose_mean", order)
        with pytest.raises(ValueError, match=re.escape(f"sweep[{cfg}]: config repeated on 2 rows")):
            getattr(mod, reader)(cells)


def _repeat_in_file(mod, tmp_path):
    """The cells file with a first copy of sweep row 0 carrying ose_mean 999 and rails yes."""
    cells = mod.load()
    extra = dict(copy.deepcopy(cells["sweep"][0]), ose_mean=999.0, rails=True)
    cells["sweep"].insert(0, extra)
    mod.dump(cells, mod.CELLS_JSON)
    return extra["config"]


def test_render_refuses_a_repeated_sweep_config_in_the_cells_file(tmp_path, monkeypatch, capsys):
    mod = _in_copy(tmp_path, monkeypatch)
    cfg = _repeat_in_file(mod, tmp_path)
    for path in (mod.CALIBRATION_MD, mod.NOTE_PATH):
        (tmp_path / path).write_text("sentinel\n", encoding="utf-8")
    named = f"render refused: cells: sweep[{cfg}]: config repeated on 2 rows"
    with pytest.raises(SystemExit, match=re.escape(named)):
        mod.main(["--render"])
    for path in (mod.CALIBRATION_MD, mod.NOTE_PATH):
        assert (tmp_path / path).read_text(encoding="utf-8") == "sentinel\n", path
    assert "rendered" not in capsys.readouterr().out


def test_audit_refuses_a_repeated_sweep_config_given_known_good_fresh_outputs(tmp_path, monkeypatch, capsys):
    """The entry-point witness: the committed file gains a first copy of a sweep row carrying ose_mean 999 and
    rails yes. With compute_cells and nuts_cross_check returning the known-good genuine outputs, `--audit`
    refuses the file naming the repeated config, and neither recomputes nor samples against it."""
    mod = _in_copy(tmp_path, monkeypatch)
    fresh, ncc = _fresh_and_ncc(mod)
    cfg = _repeat_in_file(mod, tmp_path)
    calls = []

    def compute(posts=None):
        calls.append("compute_cells")
        return copy.deepcopy(fresh)

    def sample(posteriors):
        calls.append("nuts_cross_check")
        return copy.deepcopy(ncc), None

    monkeypatch.setattr(mod, "compute_cells", compute)
    monkeypatch.setattr(mod, "nuts_cross_check", sample)
    with pytest.raises(SystemExit) as exc:
        mod.audit()
    assert str(exc.value).startswith("calibration audit FAILED")
    assert f"cells: sweep[{cfg}]: config repeated on 2 rows" in str(exc.value)
    assert calls == []
    assert "audit OK" not in capsys.readouterr().out


def test_audit_accepts_the_committed_file_with_the_same_outputs(tmp_path, monkeypatch, capsys):
    """Control for the witness above: the same known-good outputs against the unmodified file pass."""
    mod = _in_copy(tmp_path, monkeypatch)
    fresh, ncc = _fresh_and_ncc(mod)
    _audit_with(mod, monkeypatch, fresh, ncc)
    assert "calibration audit OK: 380 quadrature cells" in capsys.readouterr().out


def test_regenerate_refuses_repeated_sweep_rows_before_sampling(tmp_path, monkeypatch):
    mod = _in_copy(tmp_path, monkeypatch)
    committed = mod.load()
    fresh = {k: copy.deepcopy(v) for k, v in committed.items() if k not in _CARRIED}
    cfg = _repeat_row(fresh, 4, "ose_mean", "first")
    posts = committed["posteriors"]
    monkeypatch.setattr(mod, "_posterior", lambda name, res=None, **kw: copy.deepcopy(posts[name]))
    monkeypatch.setattr(mod, "compute_cells", lambda posts=None: copy.deepcopy(fresh))
    monkeypatch.setattr(mod, "nuts_cross_check", lambda p: pytest.fail("sampled"))
    before = _written(tmp_path, mod)
    named = f"computed cells: sweep[{cfg}]: config repeated on 2 rows"
    with pytest.raises(SystemExit, match=re.escape(named)):
        mod.regenerate()
    assert _written(tmp_path, mod) == before


def test_record_scatter_refuses_a_repeated_config_read_from_the_file(tmp_path, monkeypatch):
    mod = _in_copy(tmp_path, monkeypatch)
    other = mod.load()
    cfg = _repeat_row(other, 2, "corr", "first")
    mod.dump(other, str(tmp_path / "other.json"))
    before = (tmp_path / mod.CELLS_JSON).read_bytes()
    named = f"the other platform: sweep[{cfg}]: config repeated on 2 rows"
    with pytest.raises(SystemExit, match=re.escape(named)):
        mod.main(["--record-scatter", str(tmp_path / "other.json")])
    assert (tmp_path / mod.CELLS_JSON).read_bytes() == before


def _sweep_defects():
    """(label, corrupt(sweep) -> (sweep, the message it must produce)): each structural defect of the raw
    sweep."""
    def first_cfg(sw):
        return sw[0]["config"]
    return [
        ("not a list", lambda sw: ({}, "sweep: dict is not a list of rows")),
        ("row not an object", lambda sw: (sw[:3] + [1.0] + sw[4:], "sweep row 3: fields float are not")),
        ("field missing", lambda sw: ([*sw[:3], {k: v for k, v in sw[3].items() if k != "ose_sd"}, *sw[4:]],
                                      "sweep row 3: fields")),
        ("field added", lambda sw: ([*sw[:3], dict(sw[3], extra=1.0), *sw[4:]], "sweep row 3: fields")),
        ("config not a name", lambda sw: ([*sw[:3], dict(sw[3], config=3), *sw[4:]],
                                          "sweep row 3: config 3 is not a name")),
        ("config missing", lambda sw: (sw[:5] + sw[6:], f"sweep[{sw[5]['config']}]: missing")),
        ("config off the grid", lambda sw: ([*sw, dict(sw[0], config="R=x/os0=x/lc=x")],
                                            "sweep[R=x/os0=x/lc=x]: not a config of the grid")),
        ("rows reordered", lambda sw: ([sw[1], sw[0], *sw[2:]], "sweep: rows")),
        ("rails an int", lambda sw: ([dict(sw[0], rails=1), *sw[1:]],
                                     f"sweep[{first_cfg(sw)}].rails: 1 is not a Boolean")),
        ("rails text", lambda sw: ([dict(sw[0], rails="no"), *sw[1:]],
                                   f"sweep[{first_cfg(sw)}].rails: 'no' is not a Boolean")),
        ("rails None", lambda sw: ([dict(sw[0], rails=None), *sw[1:]],
                                   f"sweep[{first_cfg(sw)}].rails: None is not a Boolean")),
    ]


@pytest.mark.parametrize("label, corrupt", _sweep_defects(), ids=[d[0] for d in _sweep_defects()])
@pytest.mark.parametrize("side", ["this", "other"])
def test_each_sweep_defect_is_named_by_every_reader(monkeypatch, label, corrupt, side):
    """Each defect is named by `sweep_problems`, refused by the scatter recorder on either record with nothing
    written, named by the comparison, and raised by `checked_sweep`."""
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    target = cells if side == "this" else other
    target["sweep"], named = corrupt(copy.deepcopy(target["sweep"]))
    assert [p for p in mod.sweep_problems(target["sweep"]) if named in p], mod.sweep_problems(target["sweep"])
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    with pytest.raises(SystemExit, match=re.escape(named)):
        mod.record_scatter("other")
    assert written == []
    problems, _ = mod.compare_cells(cells, other, mod.committed_tolerances(_committed_cells()))
    assert [p for p in problems if named in p], problems
    with pytest.raises(ValueError, match=re.escape(named)):
        mod.checked_sweep(target)


def test_a_box_that_is_not_its_configs_is_refused_everywhere(monkeypatch):
    """Every config's three boxes, on either record: a widened upper bound is refused naming the box, nothing
    written."""
    mod = _load_script()
    base = _committed_cells()
    admitted, n = [], 0
    for side in ("this", "other"):
        for idx, row in enumerate(base["sweep"]):
            for key in ("R_box", "os0_box", "lc_box"):
                cells, other = copy.deepcopy(base), copy.deepcopy(base)
                target = cells if side == "this" else other
                target["sweep"][idx][key] = [row[key][0], 1.1 * row[key][1]]
                written = _fake_io(mod, monkeypatch, cells, {"other": other})
                n += 1
                try:
                    mod.record_scatter("other")
                    admitted.append((side, row["config"], key))
                except SystemExit as e:
                    named = f"sweep[{row['config']}].{key}:" in str(e) and "is not its config's box" in str(e)
                    if not named or written:
                        admitted.append((side, row["config"], key, "not named or written"))
    assert admitted == [] and n == 2 * 3 * len(mod._sweep_grid()), admitted[:5]


def test_a_railing_flag_that_differs_between_records_is_refused(monkeypatch):
    """Every config, either record: a railing flag the other record does not carry is refused by the scatter
    recorder and named by the comparison."""
    mod = _load_script()
    base = _committed_cells()
    tol = mod.committed_tolerances(base)
    admitted = []
    for side in ("this", "other"):
        for idx, row in enumerate(base["sweep"]):
            cells, other = copy.deepcopy(base), copy.deepcopy(base)
            (cells if side == "this" else other)["sweep"][idx]["rails"] = not row["rails"]
            written = _fake_io(mod, monkeypatch, cells, {"other": other})
            try:
                mod.record_scatter("other")
                admitted.append((side, row["config"]))
            except SystemExit as e:
                if f"sweep[{row['config']}].rails: this platform" not in str(e) or written:
                    admitted.append((side, row["config"], "not named or written"))
            problems, _ = mod.compare_cells(cells, other, tol)
            if not [p for p in problems if p.startswith(f"sweep[{row['config']}].rails: committed")]:
                admitted.append((side, row["config"], "comparison"))
    assert admitted == [], admitted[:5]


def _extra_scalars():
    return [
        ("posterior", lambda c: c["posteriors"].update(extra=copy.deepcopy(c["posteriors"]["weak"])),
         "posteriors.extra"),
        ("posterior member", lambda c: c["posteriors"]["weak"].update(extra=1.0), "posteriors.weak.extra"),
        ("quantity stat", lambda c: c["posteriors"]["kamimura"]["X_mu"].update({"q0.9": 1.0}),
         "posteriors.kamimura.X_mu.q0.9"),
        ("q_net stat", lambda c: c["q_net"].update(extra=1.0), "q_net.extra"),
        ("legacy key", lambda c: c["legacy_tension"].update(extra=1.0), "legacy_tension.extra"),
    ]


@pytest.mark.parametrize("label, add, path", _extra_scalars(), ids=[e[0] for e in _extra_scalars()])
@pytest.mark.parametrize("side", ["this", "other", "both"])
def test_an_unexpected_scalar_is_refused(monkeypatch, label, add, path, side):
    """A measured scalar outside `expected_quad_paths()`, on either record or on both (then the two records
    agree, so no other check sees it), is refused naming it, never measured or ignored, and the comparison
    names it too."""
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    for rec, s in ((cells, "this"), (other, "other")):
        if side in (s, "both"):
            add(rec)
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    with pytest.raises(SystemExit, match=r"unexpected scalar\(s\), e\.g\. \[.*" + re.escape(path)):
        mod.record_scatter("other")
    assert written == []
    problems, _ = mod.compare_cells(cells, other, mod.committed_tolerances(_committed_cells()))
    assert [p for p in problems if "unexpected scalar" in p and path in p], problems


@pytest.mark.parametrize("side", ["this", "other", "both"])
def test_a_sweep_config_off_the_grid_is_refused(monkeypatch, side):
    """A nineteenth sweep row whose config is not on the grid, on either record or on both, is refused naming
    it, nothing written; the comparison names it too."""
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    for rec, s in ((cells, "this"), (other, "other")):
        if side in (s, "both"):
            rec["sweep"].append(dict(copy.deepcopy(rec["sweep"][0]), config="R=x/os0=x/lc=x"))
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    with pytest.raises(SystemExit, match=re.escape("sweep[R=x/os0=x/lc=x]: not a config of the grid")):
        mod.record_scatter("other")
    assert written == []
    problems, _ = mod.compare_cells(cells, other, mod.committed_tolerances(_committed_cells()))
    assert [p for p in problems if "sweep[R=x/os0=x/lc=x]: not a config of the grid" in p], problems


@pytest.mark.parametrize("values", [(True, True), (False, False), (None, None), ("0.5", "0.5")],
                         ids=["True", "False", "None", "text"])
def test_record_scatter_refuses_a_typed_non_number_anywhere(monkeypatch, values):
    """Every expected scalar set to the same Boolean, None or text on both records -- equal, so a scatter of
    zero would follow -- is refused by path, never measured."""
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    paths = sorted(p for p, _ in mod.expected_quad_paths())
    admitted = []
    for path in paths:
        old_c, old_o = _set(cells, path, values[0]), _set(other, path, values[1])
        try:
            mod.record_scatter("other")
            admitted.append(path)
        except SystemExit as e:
            if f"{path} = {values[0]!r} is not a finite number" not in str(e):
                admitted.append(f"{path} (not named)")
        finally:
            _set(cells, path, old_c)
            _set(other, path, old_o)
    assert admitted == [] and written == [], admitted[:5]
    assert len(paths) == len(mod.quad_cells(_committed_cells()))


def _record_defects():
    """(label, corrupt(record), message) for the computed record's own members."""
    out = [(f"no {k}", (lambda c, k=k: c.pop(k)), f"no {k!r}") for k in
           ("schema", "estimator", "posteriors", "sweep", "q_net", "legacy_tension", "inputs", "computed_on")]
    out += [
        ("unexpected member", lambda c: c.update(extra={}), "unexpected member 'extra'"),
        ("schema", lambda c: c.update(schema="openmucf-calibration-cells/0"),
         "schema 'openmucf-calibration-cells/0'"),
        ("computed_on not a description", lambda c: c.update(computed_on="Linux"),
         "computed_on: 'Linux' is not"),
        ("computed_on member missing", lambda c: c["computed_on"].pop("numpy"), "computed_on: {"),
        ("computed_on not text", lambda c: c["computed_on"].update(python=3.12), "computed_on: {"),
    ]
    est = {"res": 0.5, "sweep_res": 1.0, "grid": {"ne": 1}, "bins": {}, "l_other_pct": [0.1388, 0.02],
           "lambda_c_box": [0.8e8, 1.6e8], "covariance": "L2"}
    out += [(f"estimator {k}", (lambda c, k=k, v=v: c["estimator"].update({k: v})),
             "is not the live estimator") for k, v in est.items()]
    return out


@pytest.mark.parametrize("label, corrupt, named", _record_defects(), ids=[d[0] for d in _record_defects()])
@pytest.mark.parametrize("side", ["this", "other"])
def test_record_scatter_refuses_each_record_defect(monkeypatch, label, corrupt, named, side):
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    corrupt(cells if side == "this" else other)
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    platform = "this platform" if side == "this" else "the other platform"
    with pytest.raises(SystemExit, match=re.escape(f"{platform}: ") + ".*" + re.escape(named)):
        mod.record_scatter("other")
    assert written == []


def test_expected_leaves_are_the_compared_scalars():
    mod = _load_script()
    assert {mod._path_text(p) for p in mod._expected_leaves()} == {p for p, _ in mod.expected_quad_paths()}
    assert len(mod._expected_leaves()) == len(mod.expected_quad_paths())


def test_a_dotted_key_cannot_stand_in_for_a_quantity(monkeypatch):
    """posteriors.weak without R, beside a posterior named "weak.R" holding R's stats: the dotted names would
    spell the expected paths, the key paths do not -- refused, the missing quantity and the stray posterior
    named."""
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    cells["posteriors"]["weak.R"] = cells["posteriors"]["weak"].pop("R")
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    with pytest.raises(SystemExit, match=r"(?s)this platform: 7 scalar\(s\) missing.*7 unexpected scalar"):
        mod.record_scatter("other")
    assert written == []


def test_record_problems_refuses_a_record_that_is_not_an_object():
    assert _load_script().record_problems([], "the other platform") == [
        "the other platform: list is not a cells record"]


def test_the_estimator_check_reaches_the_estimator_rendered(tmp_path, monkeypatch):
    """A committed estimator off the live one (the grid CALIBRATION.md prints) stops `--render`, nothing
    written, and `--audit` before it recomputes."""
    mod = _in_copy(tmp_path, monkeypatch)
    cells = mod.load()
    cells["estimator"]["grid"]["ne"] = 4000
    mod.dump(cells, mod.CELLS_JSON)
    (tmp_path / mod.CALIBRATION_MD).write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="render refused: cells: estimator .* is not the live estimator"):
        mod.main(["--render"])
    assert (tmp_path / mod.CALIBRATION_MD).read_text(encoding="utf-8") == "sentinel\n"
    monkeypatch.setattr(mod, "compute_cells", lambda posts=None: pytest.fail("recomputed"))
    with pytest.raises(SystemExit, match="calibration audit FAILED:\n  cells: estimator"):
        mod.audit()


@pytest.mark.parametrize("key", ["side_channel_inputs_csv_sha256", "uq_priors_csv_sha256",
                                 "fc001_card_sha256"])
def test_record_scatter_refuses_records_of_different_inputs(monkeypatch, key):
    mod = _load_script()
    cells, other = _committed_cells(), _committed_cells()
    other["inputs"][key] = "0" * 64
    written = _fake_io(mod, monkeypatch, cells, {"other": other})
    with pytest.raises(SystemExit, match="the platforms' inputs differ"):
        mod.record_scatter("other")
    assert written == []


# --- the stored convergence runs
_INVALID = [float("nan"), float("inf"), float("-inf"), None, True, False, "not-a-number"]


def test_record_convergence_refuses_an_invalid_correlation(monkeypatch):
    """Correlation admission: each posterior's stored correlation, publication or fine, set to each invalid
    value or removed, is refused by name at record time with nothing written, and by the check every consumer
    runs."""
    mod = _load_script()
    cells = _committed_cells()
    tol = mod.committed_tolerances(cells)
    admitted, n = [], 0
    for name in mod.CONVERGENCE_POSTERIORS:
        for side in ("pub", "fine"):
            for value in [*_INVALID, "missing"]:
                runs = _genuine_runs(cells, mod)
                bad = copy.deepcopy(cells)
                for post in (runs[name][side], bad["convergence"]["runs"][name][side]):
                    if value == "missing":
                        del post["corr_omega_s0_R"]
                    else:
                        post["corr_omega_s0_R"] = value
                shown = repr(None if value == "missing" else value)
                named = f"{name}.{side}.corr_omega_s0_R: {shown} is not a finite number"
                n += 1
                written = _fake_io(mod, monkeypatch, copy.deepcopy(cells), runs)
                try:
                    mod.record_convergence(list(runs))
                    admitted.append((name, side, shown, "recorded"))
                except SystemExit as e:
                    if named not in str(e) or written:
                        admitted.append((name, side, shown, "not named or written"))
                if not [p for p in mod.convergence_problems(bad, tol) if named in p]:
                    admitted.append((name, side, shown, "consumed"))
    assert admitted == [], admitted[:5]
    assert n == len(mod.CONVERGENCE_POSTERIORS) * 2 * (len(_INVALID) + 1)


@pytest.mark.parametrize("value", _INVALID[3:], ids=["None", "True", "False", "text"])
def test_record_convergence_refuses_a_typed_non_number_member(monkeypatch, value):
    """A run member that is a Boolean, None or text is refused by name, in every member of the fine run."""
    mod = _load_script()
    cells = _committed_cells()
    admitted = []
    for q, st in _members(mod):
        runs = _genuine_runs(cells, mod)
        runs["weak"]["fine"][q][st] = value
        written = _fake_io(mod, monkeypatch, copy.deepcopy(cells), runs)
        try:
            mod.record_convergence(list(runs))
            admitted.append(f"{q}.{st}")
        except SystemExit as e:
            if f"weak.fine.{q}.{st}: {value!r} is not a finite number" not in str(e) or written:
                admitted.append(f"{q}.{st} (not named or written)")
    assert admitted == [], admitted[:5]


def _run_member_defects():
    return [
        ("member added", lambda p: p.update(extra=1.0), "unexpected member(s) ['extra']"),
        ("stat added", lambda p: p["X_mu"].update({"q0.9": 1.0}), "X_mu: unexpected stat(s) ['q0.9']"),
    ]


@pytest.mark.parametrize("label, corrupt, named", _run_member_defects(),
                         ids=[d[0] for d in _run_member_defects()])
def test_a_run_member_outside_the_stored_set_is_refused(monkeypatch, label, corrupt, named):
    """Each posterior's publication or fine run with a member it does not store is refused at record time and
    by the consumers' check."""
    mod = _load_script()
    cells = _committed_cells()
    tol = mod.committed_tolerances(cells)
    admitted = []
    for name in mod.CONVERGENCE_POSTERIORS:
        for side in ("pub", "fine"):
            runs = _genuine_runs(cells, mod)
            bad = copy.deepcopy(cells)
            corrupt(runs[name][side])
            corrupt(bad["convergence"]["runs"][name][side])
            written = _fake_io(mod, monkeypatch, copy.deepcopy(cells), runs)
            try:
                mod.record_convergence(list(runs))
                admitted.append((name, side))
            except SystemExit as e:
                if f"{name}.{side}" not in str(e) or named not in str(e) or written:
                    admitted.append((name, side, "not named or written"))
            if not [p for p in mod.convergence_problems(bad, tol) if f"{name}.{side}" in p and named in p]:
                admitted.append((name, side, "consumed"))
    assert admitted == [], admitted


def test_a_run_side_that_is_not_a_posterior_is_refused(monkeypatch):
    mod = _load_script()
    cells = _committed_cells()
    runs = _genuine_runs(cells, mod)
    runs["kamimura"]["fine"] = None
    written = _fake_io(mod, monkeypatch, copy.deepcopy(cells), runs)
    with pytest.raises(SystemExit, match="kamimura.fine: None is not a posterior"):
        mod.record_convergence(list(runs))
    assert written == []


def test_record_convergence_refuses_every_unconverged_member(monkeypatch):
    """Criteria, all members of every posterior: the publication run is the cells' own and the fine run moved
    10 % in one member -- refused naming the change, nothing written; and by the consumers' check."""
    mod = _load_script()
    cells = _committed_cells()
    tol = mod.committed_tolerances(cells)
    admitted, n = [], 0
    for name in mod.CONVERGENCE_POSTERIORS:
        for q, st in _members(mod):
            runs = _genuine_runs(cells, mod)
            runs[name]["fine"][q][st] *= 1.1
            bad = copy.deepcopy(cells)
            bad["convergence"]["runs"][name]["fine"][q][st] *= 1.1
            named = f"{name}: {q}.{st}: relative change"
            n += 1
            written = _fake_io(mod, monkeypatch, copy.deepcopy(cells), runs)
            try:
                mod.record_convergence(list(runs))
                admitted.append(f"{name}.{q}.{st}")
            except SystemExit as e:
                if named not in str(e) or written:
                    admitted.append(f"{name}.{q}.{st} (not named or written)")
            if not [p for p in mod.convergence_problems(bad, tol) if named in p]:
                admitted.append(f"{name}.{q}.{st} (consumed)")
    assert admitted == [], admitted[:5]
    assert n == len(mod.CONVERGENCE_POSTERIORS) * len(_members(mod))


def test_the_correlation_is_bound_to_the_cells(monkeypatch):
    """Binding reach: a run whose publication correlation is not the cells' (the fine one unmoved,
    so no criterion sees it) is refused naming the correlation at record time and by the consumers' check."""
    mod = _load_script()
    cells = _committed_cells()
    tol = mod.committed_tolerances(cells)
    admitted = []
    for name in mod.CONVERGENCE_POSTERIORS:
        runs = _genuine_runs(cells, mod)
        runs[name]["pub"]["corr_omega_s0_R"] *= 1.1
        bad = copy.deepcopy(cells)
        bad["convergence"]["runs"][name]["pub"]["corr_omega_s0_R"] *= 1.1
        written = _fake_io(mod, monkeypatch, copy.deepcopy(cells), runs)
        try:
            mod.record_convergence(list(runs))
            admitted.append(name)
        except SystemExit as e:
            if f"{name}.corr_omega_s0_R: run" not in str(e) or written:
                admitted.append(f"{name} (not named or written)")
        if not [p for p in mod.convergence_problems(bad, tol) if f"{name}.corr_omega_s0_R: run" in p]:
            admitted.append(f"{name} (consumed)")
    assert admitted == [], admitted


@pytest.mark.parametrize("kind", ["repeated", "extra", "missing", "unnamed", "not an object"])
def test_record_convergence_refuses_any_set_of_runs_but_one_per_posterior(monkeypatch, kind):
    """For each posterior: its run given twice, a fifth named run, its run left out, or a run with no name --
    refused, nothing written (a repeated or extra run would otherwise be dropped unread)."""
    mod = _load_script()
    cells = _committed_cells()
    for name in mod.CONVERGENCE_POSTERIORS:
        runs = _genuine_runs(cells, mod)
        paths = list(runs)
        if kind == "repeated":
            paths.append(name)
        elif kind == "extra":
            runs["extra"] = dict(copy.deepcopy(runs[name]), name="extra")
            paths.append("extra")
        elif kind == "missing":
            paths.remove(name)
        elif kind == "unnamed":
            runs[name]["name"] = None
        else:
            runs[name] = [runs[name]]
        written = _fake_io(mod, monkeypatch, copy.deepcopy(cells), runs)
        with pytest.raises(SystemExit, match=r"convergence runs \["):
            mod.record_convergence(paths)
        assert written == []


@pytest.mark.parametrize("corrupt, named", [
    (lambda c: c["convergence"].update(extra=1), "convergence: unexpected member(s) ['extra']"),
    (lambda c: c["convergence"]["runs"]["weak"].pop("fine"), "convergence: weak: run members ['pub']"),
    (lambda c: c["convergence"]["runs"]["kamimura"].update(res_pub=2.0),
     "convergence: kamimura: run members ['fine', 'pub', 'res_pub']"),
])
def test_convergence_record_holds_only_what_it_stores(corrupt, named):
    mod = _load_script()
    cells = _committed_cells()
    corrupt(cells)
    probs = mod.convergence_problems(cells, mod.committed_tolerances(_committed_cells()))
    assert [p for p in probs if p.startswith(named)], probs


# --- the stored scatter record
def _scatter_defects():
    out = [
        ("member added", lambda s: s.update(extra=1), "audit_scatter: members"),
        ("platforms missing", lambda s: s.pop("platforms"), "audit_scatter: members"),
        ("one platform", lambda s: s.update(platforms=s["platforms"][:1]), "audit_scatter.platforms: "),
        ("platform malformed", lambda s: s["platforms"][1].pop("numpy"), "audit_scatter.platforms[1]: "),
        ("class missing", lambda s: s["scatter"].pop("quantile"), "audit_scatter.scatter: "),
        ("class added", lambda s: s["scatter"].update(other=0.0), "audit_scatter.scatter: "),
    ]
    for c in ("moment", "quantile", "quantile_L"):
        for v in (float("nan"), float("inf"), -1e-9, True, None, "0"):
            out.append((f"{c}={v!r}", (lambda s, c=c, v=v: s["scatter"].update({c: v})),
                        f"audit_scatter.scatter.{c}: {v!r} is not a finite number at least 0"))
    return out


@pytest.mark.parametrize("label, corrupt, named", _scatter_defects(), ids=[d[0] for d in _scatter_defects()])
def test_the_scatter_record_is_admitted_before_it_sets_a_tolerance(label, corrupt, named):
    mod = _load_script()
    cells = _committed_cells()
    corrupt(cells["audit_scatter"])
    with pytest.raises(SystemExit, match="(?s)audit_scatter refused:.*" + re.escape(named)):
        mod.committed_tolerances(cells)


def test_audit_refuses_a_nan_scatter_that_would_open_the_tolerance_to_its_cap(tmp_path, monkeypatch, capsys):
    """A recorded scatter of NaN with the tolerance written at the cap: unadmitted, the rule reads the cap and
    a fresh cell 5e-6 off its committed value (five times the measured tolerance) would pass. Refused
    instead."""
    mod = _in_copy(tmp_path, monkeypatch)
    fresh, ncc = _fresh_and_ncc(mod)
    cells = mod.load()
    cells["audit_scatter"]["scatter"]["moment"] = float("nan")
    cells["audit_scatter"]["tolerance"]["moment"] = mod.QUAD_TOL_CAP["moment"]
    mod.dump(cells, mod.CELLS_JSON)
    mod.render(cells)       # the documents agree with the record, so only admission can refuse it
    fresh["posteriors"]["weak"]["lambda_c"]["mean"] *= 1 + 5e-6
    with pytest.raises(SystemExit, match="audit_scatter refused:\n  audit_scatter.scatter.moment: nan"):
        _audit_with(mod, monkeypatch, fresh, ncc)
    assert "audit OK" not in capsys.readouterr().out


# --- the stored NUTS record
def _nuts_defects():
    out = [
        ("chain missing", lambda n: n.pop("kamimura"), "nuts_cross_check: chains ['weak'] are not"),
        ("chain added", lambda n: n.update(extra=copy.deepcopy(n["weak"])), "nuts_cross_check: chains"),
        ("field missing", lambda n: n["weak"].pop("min_ess"), "nuts_cross_check.weak: fields"),
        ("field added", lambda n: n["weak"].update(extra=1.0), "nuts_cross_check.weak: fields"),
        ("mean quantity missing", lambda n: n["kamimura"]["mean"].pop("R"),
         "nuts_cross_check.kamimura.mean: {"),
        ("mcse not an object", lambda n: n["weak"].update(mcse=0.1),
         "nuts_cross_check.weak.mcse: 0.1 does not"),
        ("mcse zero", lambda n: n["weak"]["mcse"].update(R=0.0),
         "nuts_cross_check.weak.mcse.R: 0.0 is not positive"),
        ("mcse negative", lambda n: n["weak"]["mcse"].update(R=-0.003),
         "nuts_cross_check.weak.mcse.R: -0.003 is not positive"),
        ("divergences a float", lambda n: n["weak"].update(divergences=0.0),
         "nuts_cross_check.weak.divergences: 0.0 is not a count"),
        ("divergences a Boolean", lambda n: n["weak"].update(divergences=False),
         "nuts_cross_check.weak.divergences: False is not a count"),
        ("gate: r_hat", lambda n: n["kamimura"].update(max_rhat=1.02),
         "nuts_cross_check.kamimura: max_rhat 1.0200"),
        ("gate: ess", lambda n: n["weak"].update(min_ess=300.0), "nuts_cross_check.weak: min_ess 300 <= 400"),
        ("gate: divergences", lambda n: n["weak"].update(divergences=2),
         "nuts_cross_check.weak: divergences 2"),
        ("offset not its means'", lambda n: n["weak"].update(max_abs_dev_over_mcse=0.5),
         "nuts_cross_check.weak.max_abs_dev_over_mcse: 0.5 is not its means' offset"),
    ]
    for key in ("max_rhat", "min_ess", "max_abs_dev_over_mcse"):
        for v in (float("nan"), True, None):
            out.append((f"{key}={v!r}", (lambda n, k=key, v=v: n["kamimura"].update({k: v})),
                        f"nuts_cross_check.kamimura.{key}: {v!r} is not a finite number"))
    for key in ("mean", "mcse"):
        out.append((f"{key} nan", (lambda n, k=key: n["weak"][k].update(X_mu=float("nan"))),
                    f"nuts_cross_check.weak.{key}.X_mu: nan is not a finite number"))
    return out


@pytest.mark.parametrize("label, corrupt, named", _nuts_defects(), ids=[d[0] for d in _nuts_defects()])
def test_the_nuts_record_is_admitted_as_rendered(label, corrupt, named):
    mod = _load_script()
    cells = _committed_cells()
    corrupt(cells["nuts_cross_check"])
    probs = mod.nuts_record_problems(cells)
    assert [p for p in probs if p.startswith(named)], probs
    assert [p for p in mod.committed_problems(cells) if p.startswith(named)]


def test_the_gate_mean_offset_is_checked_on_the_stored_record():
    """A stored mean 3.5 MCSE from the quadrature mean, its rendered offset made consistent, fails the
    gate."""
    mod = _load_script()
    cells = _committed_cells()
    d, quad = cells["nuts_cross_check"]["kamimura"], cells["posteriors"]["kamimura"]
    d["mean"]["lambda_c"] = quad["lambda_c"]["mean"] + 3.5 * d["mcse"]["lambda_c"]
    d["max_abs_dev_over_mcse"] = mod._nuts_offset(d, quad)
    assert mod.nuts_record_problems(cells) == ["nuts_cross_check.kamimura: lambda_c mean 3.50 MCSE from the "
                                               "quadrature mean"]


def test_render_and_audit_refuse_an_inconsistent_nuts_record(tmp_path, monkeypatch, capsys):
    mod = _in_copy(tmp_path, monkeypatch)
    cells = mod.load()
    cells["nuts_cross_check"]["weak"]["max_abs_dev_over_mcse"] = 0.5
    mod.dump(cells, mod.CELLS_JSON)
    (tmp_path / mod.CALIBRATION_MD).write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="render refused: nuts_cross_check.weak.max_abs_dev_over_mcse: 0.5"):
        mod.main(["--render"])
    assert (tmp_path / mod.CALIBRATION_MD).read_text(encoding="utf-8") == "sentinel\n"
    monkeypatch.setattr(mod, "nuts_cross_check", lambda posteriors: pytest.fail("sampled"))
    monkeypatch.setattr(mod, "compute_cells", lambda posts=None: pytest.fail("recomputed"))
    with pytest.raises(SystemExit, match="calibration audit FAILED:\n  nuts_cross_check.weak.max_abs_dev"):
        mod.audit()
    assert "audit OK" not in capsys.readouterr().out


def test_nuts_chains_are_the_chains_the_cross_check_records(monkeypatch):
    mod = _load_script()
    posts = _committed_cells()["posteriors"]
    rec = _committed_cells()["nuts_cross_check"]
    seen = []

    def fake_run(omega_s0_prior, **kw):
        seen.append(omega_s0_prior)
        return omega_s0_prior, {"draws": None}

    def fake_summary(mcmc):
        name = "weak" if mcmc == mod.calibrate.WEAK_OMEGA_S0_PRIOR else "kamimura"
        return {k: copy.deepcopy(v) for k, v in rec[name].items() if k != "max_abs_dev_over_mcse"}

    monkeypatch.setattr(mod.calibrate, "run_model_w", fake_run)
    monkeypatch.setattr(mod.calibrate, "summarize_w", fake_summary)
    got, _ = mod.nuts_cross_check(posts)
    assert tuple(got) == mod.NUTS_CHAINS
    assert got == rec


# --- the files the script reads
def _with_repeated_key(text, key):
    """The JSON text with its first top-level ``"key": ...`` member stated twice (the second copy first)."""
    obj = json.loads(text)
    head = json.dumps({key: obj[key]}, indent=2)[1:-1].strip()
    return text.replace("{", "{\n  " + head + ",", 1)


def test_load_refuses_a_repeated_key(tmp_path, monkeypatch):
    """A cells file stating q_net twice: refused naming the file and the key, so `--render` writes nothing and
    the scatter recorder refuses the other platform's file."""
    mod = _in_copy(tmp_path, monkeypatch)
    text = (tmp_path / mod.CELLS_JSON).read_text(encoding="utf-8")
    assert mod.load() == json.loads(text)
    (tmp_path / "other.json").write_text(_with_repeated_key(text, "q_net"), encoding="utf-8")
    with pytest.raises(ValueError, match=r"other.json: repeated key\(s\) \['q_net'\]"):
        mod.main(["--record-scatter", str(tmp_path / "other.json")])
    assert (tmp_path / mod.CELLS_JSON).read_text(encoding="utf-8") == text
    (tmp_path / mod.CELLS_JSON).write_text(_with_repeated_key(text, "legacy_tension"), encoding="utf-8")
    (tmp_path / mod.CALIBRATION_MD).write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"repeated key\(s\) \['legacy_tension'\]"):
        mod.main(["--render"])
    assert (tmp_path / mod.CALIBRATION_MD).read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize("defect", ["second scenario A", "repeated target"])
def test_the_card_reader_refuses_a_scenario_it_would_collapse(tmp_path, monkeypatch, defect):
    mod = _in_copy(tmp_path, monkeypatch)
    card = json.loads((tmp_path / "forecasts/FC-001-mufuse.json").read_text(encoding="utf-8"))
    scenarios = card["payload"]["scenarios"]
    a = [s for s in scenarios if s["name"] == "A"][0]
    if defect == "second scenario A":
        scenarios.append(copy.deepcopy(a))
        named = "2 scenarios named A"
    else:
        a["predictions"].append(copy.deepcopy(a["predictions"][0]))
        named = "scenario A repeats a target_id"
    (tmp_path / "card.json").write_text(json.dumps(card), encoding="utf-8")
    monkeypatch.setattr(mod.forecast, "CARD_PATH", str(tmp_path / "card.json"))
    with pytest.raises(ValueError, match=named):
        mod._card_scenario_a()


@pytest.mark.parametrize("rows", [0, 2])
def test_vb_figures_refuses_other_than_one_row(tmp_path, monkeypatch, rows):
    mod = _in_copy(tmp_path, monkeypatch)
    p = tmp_path / mod.VALIDATION_CHANNELS_MD
    lines = p.read_text(encoding="utf-8").splitlines()
    row = [ln for ln in lines if ln.startswith("| V_breunlich_lambdac |")]
    assert len(row) == 1
    lines = [ln for ln in lines if ln != row[0]] + row * rows
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"{rows} V_breunlich_lambdac rows"):
        mod.vb_figures(mod.load())


@pytest.mark.parametrize("key", ["shipped_v1_3_0", "convergence"])
def test_regenerate_refuses_a_cells_file_without_its_carried_records(tmp_path, monkeypatch, key):
    mod = _in_copy(tmp_path, monkeypatch)
    cells = mod.load()
    del cells[key]
    mod.dump(cells, mod.CELLS_JSON)
    before = _written(tmp_path, mod)
    monkeypatch.setattr(mod, "_posterior", lambda name, res=None, **kw: pytest.fail("computed"))
    with pytest.raises(SystemExit, match=f"has no {key!r}"):
        mod.regenerate()
    assert _written(tmp_path, mod) == before


def test_render_check_names_a_missing_document(tmp_path, monkeypatch):
    mod = _in_copy(tmp_path, monkeypatch)
    (tmp_path / mod.NOTE_PATH).unlink()
    assert mod.render_problems(mod.load()) == [f"{mod.NOTE_PATH}: missing"]


def test_the_card_reader_refuses_a_repeated_key(tmp_path, monkeypatch):
    """The FC-001 card read by the note with a top-level key stated twice is refused naming it."""
    mod = _in_copy(tmp_path, monkeypatch)
    text = (tmp_path / "forecasts/FC-001-mufuse.json").read_text(encoding="utf-8")
    (tmp_path / "card.json").write_text(_with_repeated_key(text, "payload"), encoding="utf-8")
    monkeypatch.setattr(mod.forecast, "CARD_PATH", str(tmp_path / "card.json"))
    with pytest.raises(ValueError, match=r"card.json: repeated key\(s\) \['payload'\]"):
        mod._card_scenario_a()


# --- the stored record of the documents as released
def _shipped_calibration_keys(cells):
    return sorted(k for k in cells["shipped_v1_3_0"] if k.startswith("CALIBRATION.md|"))


def test_the_shipped_record_is_admitted_and_not_vacuous():
    """Every CALIBRATION.md key of the committed shipped record names a cell of the record, and the two
    validation-page keys are present: the admission has nothing to refuse, over all of them."""
    mod = _load_script()
    cells = _committed_cells()
    keys = _shipped_calibration_keys(cells)
    assert len(keys) + len(mod.SHIPPED_VALIDATION_KEYS) == len(cells["shipped_v1_3_0"])
    assert keys and all(len(k.split("|")) == 4 for k in keys)
    assert mod.shipped_problems(cells) == []


def test_every_shipped_key_that_names_no_cell_is_refused():
    """Each CALIBRATION.md key of the shipped record, its row renamed to one the record does not hold, is
    named; `_moved_rows` would otherwise stop on it with a bare KeyError, or print a cell of another row."""
    mod = _load_script()
    base = _committed_cells()
    admitted = []
    keys = _shipped_calibration_keys(base)
    for key in keys:
        cells = copy.deepcopy(base)
        doc, section, _row, col = key.split("|")
        bad = f"{doc}|{section}|no such row|{col}"
        cells["shipped_v1_3_0"][bad] = cells["shipped_v1_3_0"].pop(key)
        named = f"shipped_v1_3_0[{bad!r}]: names no cell"
        if not [p for p in mod.shipped_problems(cells) if p.startswith(named)]:
            admitted.append(key)
    assert admitted == [], admitted[:5]
    assert len(keys) == len(base["shipped_v1_3_0"]) - len(mod.SHIPPED_VALIDATION_KEYS)


def _shipped_defects():
    k0 = "VALIDATION_CHANNELS.md|V_breunlich_lambdac|predicted"
    return [
        ("not a record", lambda sh: [], "shipped_v1_3_0: list is not a record"),
        ("value not a string", lambda sh: sh.update({sorted(sh)[0]: 0.46}), "is not a printed string"),
        ("five parts", lambda sh: sh.update({"CALIBRATION.md|a|b|c|d": "1"}),
         "not a CALIBRATION.md|section|row|column key"),
        ("other document", lambda sh: sh.update({"FINDINGS.md|a|b|c": "1"}),
         "not a CALIBRATION.md|section|row|column key"),
        ("unknown section", lambda sh: sh.update({"CALIBRATION.md|No such section|weak|mean": "1"}),
         "names no cell of this record"),
        ("validation key missing", lambda sh: sh.pop(k0), f"shipped_v1_3_0: no {k0!r}"),
        ("validation value nan", lambda sh: sh.update({k0: "nan"}), "'nan' is not a finite number"),
        ("validation value text", lambda sh: sh.update({k0: "n/a"}), "'n/a' is not a finite number"),
    ]


@pytest.mark.parametrize("label, corrupt, named", _shipped_defects(), ids=[d[0] for d in _shipped_defects()])
def test_each_shipped_record_defect_is_named(label, corrupt, named):
    mod = _load_script()
    cells = _committed_cells()
    out = corrupt(cells["shipped_v1_3_0"])
    if isinstance(out, list):
        cells["shipped_v1_3_0"] = out
    assert [p for p in mod.shipped_problems(cells) if named in p], mod.shipped_problems(cells)
    assert [p for p in mod.committed_problems(cells) if named in p]


def test_render_and_audit_refuse_a_shipped_key_that_names_no_cell(tmp_path, monkeypatch, capsys):
    mod = _in_copy(tmp_path, monkeypatch)
    cells = mod.load()
    cells["shipped_v1_3_0"]["CALIBRATION.md|No such section|weak|mean"] = "0.46"
    mod.dump(cells, mod.CELLS_JSON)
    for path in (mod.CALIBRATION_MD, mod.NOTE_PATH):
        (tmp_path / path).write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="render refused: shipped_v1_3_0.*names no cell"):
        mod.main(["--render"])
    for path in (mod.CALIBRATION_MD, mod.NOTE_PATH):
        assert (tmp_path / path).read_text(encoding="utf-8") == "sentinel\n", path
    monkeypatch.setattr(mod, "compute_cells", lambda posts=None: pytest.fail("recomputed"))
    with pytest.raises(SystemExit, match="calibration audit FAILED:\n  shipped_v1_3_0"):
        mod.audit()
    assert "audit OK" not in capsys.readouterr().out


# --- the regeneration admits the record it writes, and --cells-out the record it writes
def test_regenerate_refuses_a_record_its_readers_would_refuse(tmp_path, monkeypatch):
    """A NUTS record that passes the gate but carries a field the record does not hold: refused by the same
    admission the readers run, before the cells file, the documents or the figure are written."""
    mod = _in_copy(tmp_path, monkeypatch)
    ncc = copy.deepcopy(mod.load()["nuts_cross_check"])
    ncc["kamimura"]["extra"] = 1.0
    before = _written(tmp_path, mod)
    with pytest.raises(SystemExit, match="regeneration refused: the record to be written:\n  "
                                         "nuts_cross_check.kamimura: fields"):
        _regenerate_with(mod, monkeypatch, ncc)
    assert _written(tmp_path, mod) == before
    assert not (tmp_path / "figures").exists()


def test_cells_out_refuses_a_record_the_scatter_recorder_would_refuse(tmp_path, monkeypatch):
    mod = _in_copy(tmp_path, monkeypatch)
    fresh, _ = _fresh_and_ncc(mod)
    fresh["posteriors"]["weak"]["R"]["sd"] = float("nan")
    monkeypatch.setattr(mod, "compute_cells", lambda posts=None: copy.deepcopy(fresh))
    with pytest.raises(SystemExit, match="cells refused: computed cells: posteriors.weak.R.sd = nan"):
        mod.main(["--cells-out", str(tmp_path / "fresh.json")])
    assert not (tmp_path / "fresh.json").exists()
    good, _ = _fresh_and_ncc(mod)
    monkeypatch.setattr(mod, "compute_cells", lambda posts=None: copy.deepcopy(good))
    mod.main(["--cells-out", str(tmp_path / "fresh.json")])
    assert mod.record_problems(mod.load(str(tmp_path / "fresh.json")), "fresh") == []


# --- readers of the cells file outside this script
def test_admitted_cells_returns_the_committed_record_and_refuses_a_defect(tmp_path, monkeypatch):
    mod = _in_copy(tmp_path, monkeypatch)
    assert mod.admitted_cells() == _committed_cells()
    _repeat_in_file(mod, tmp_path)
    with pytest.raises(SystemExit, match=r"CALIBRATION_CELLS.json refused: cells: sweep\[.*config repeated"):
        mod.admitted_cells()
