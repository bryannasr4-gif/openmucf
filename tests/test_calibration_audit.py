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
    return "cell sets differ"


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
    assert [p for p in problems if p.startswith("cell sets differ") and gone in p], problems


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
