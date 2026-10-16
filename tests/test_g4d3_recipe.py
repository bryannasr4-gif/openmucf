"""The patched-recipe runs of the D3 tables: the inputs the renderer makes, the completeness check the
collector applies before it reads a printed string, and the derivation of binding energies from a run's
printed strings.

The fixture ``tests/fixtures/d3_recipe/Pb208_anchor_on`` holds one production run as the collector reads
it: its input, its ``.xr.out`` and the five header lines of every state file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import pathlib
import shutil
import sys
from decimal import Decimal

import pytest

from openmucf.g4.sources import mudirac130 as md

REPO = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "d3_recipe" / "Pb208_anchor_on"
RUN = "Pb208_anchor_on"


def _script():
    spec = importlib.util.spec_from_file_location("mudirac_d3_script", REPO / "scripts" / "mudirac_d3.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mudirac_d3_script"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inputs():
    rows = md.load_inputs(REPO / md.INPUTS_RELPATH)
    cells, _origins = md.load_validation(REPO / md.CELLS_RELPATH, REPO / md.ORIGIN_RELPATH)
    return rows, cells


def _row(rows, z, a):
    return next(r for r in rows if r.nuclide == (z, a))


def _copy_run(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / RUN
    shutil.copytree(FIXTURE, d)
    return d


def _text() -> str:
    return (FIXTURE / f"{RUN}.in").read_text("ascii")


# ----------------------------------------------------------------------------------------------------
# the inputs
# ----------------------------------------------------------------------------------------------------

def test_r01_the_production_input_of_the_fixture_is_the_one_the_renderer_makes(inputs):
    rows, cells = inputs
    row = _row(rows, 82, 208)
    extra = md.extra_lines(cells, row.nuclide)
    text = md.render_recipe_input(row, "anchor", "recipe", True, md.RECIPE_SETTINGS, extra)
    assert text == _text()
    run = next(r for r in md.recipe_runs([row], cells, "prod") if r.run == RUN)
    assert run.text == text and run.kind == "anchor" and run.uehling and run.variant == "prod"


def test_r02_the_recipe_input_adds_exactly_its_keywords_to_the_renderers_own_input(inputs):
    rows, cells = inputs
    row = _row(rows, 82, 208)
    extra = md.extra_lines(cells, row.nuclide)
    base = md.render_input(row, "base", extra).splitlines()
    new = md.render_recipe_input(row, "anchor", "recipe", True, md.RECIPE_SETTINGS, extra).splitlines()
    assert {line.split(":")[0] for line in base} <= {line.split(":")[0] for line in new}
    keys = [line.split(":")[0] for line in new]
    assert len(keys) == len(set(keys)), "a key set twice"
    values = dict(line.split(": ", 1) for line in new)
    assert values["xr_print_precision"] == str(md.RECIPE_XR_PRECISION)
    assert values["state_header_precision"] == str(md.RECIPE_HEADER_PRECISION)
    assert values["xr_lines"].endswith("," + md.anchor_line())
    assert values["ideal_atom_minshell"] == md.shell(md.N_MAX + 1)
    for key, value in md.STAGE_KEYWORDS["recipe"]:
        assert values[key] == value
    assert float(values["fermi2_rms_fm"]) == float(row.radius_fm) / md.SPHERE_FACTOR
    assert abs(float(values["radius"]) / math.sqrt(5 / 3) - float(values["fermi2_rms_fm"])) <= 1e-12 * float(
        values["fermi2_rms_fm"])
    assert values["loggrid_step"] == repr(0.005 / 2 ** md.RECIPE_SETTINGS.k)
    assert values["energy_tol"] == "1e-07"


def test_r03_the_uehling_free_twin_differs_only_in_the_uehling_switch(inputs):
    rows, cells = inputs
    for z, a in ((2, 3), (47, 107), (92, 238)):
        row = _row(rows, z, a)
        extra = md.extra_lines(cells, row.nuclide)
        for kind in md.RECIPE_KINDS:
            on = md.render_recipe_input(row, kind, "recipe", True, md.RECIPE_SETTINGS, extra).splitlines()
            off = md.render_recipe_input(row, kind, "recipe", False, md.RECIPE_SETTINGS, extra).splitlines()
            diff = [(x, y) for x, y in zip(on, off, strict=True) if x != y]
            assert diff == [("uehling_correction: TRUE", "uehling_correction: FALSE")]


def test_r04_each_batch_holds_the_runs_its_rule_implies(inputs):
    rows, cells = inputs
    validation = set(md.gated_nuclides(cells))
    prod = md.recipe_runs(rows, cells, "prod")
    assert len(prod) == 2 * sum(len(md.run_kinds(r, validation)) for r in rows)
    assert len({r.run for r in prod}) == len(prod)
    refine = md.recipe_runs(rows, cells, "refine")
    assert len(refine) == len(rows) * (len(md.REFINEMENTS) + len(md.NOU_REFINEMENTS))
    rminus = md.recipe_runs(rows, cells, "rminus")
    assert len(rminus) == sum(1 for r in rows if md.density_domain(r, "rminus") == md.INSIDE_2PF)
    stages = md.recipe_runs(rows, cells, "stages")
    assert len(stages) == len(md.STAGES) * len(md.compared_nuclides(cells))
    shipped = [r for r in stages if r.stage == "shipped"]
    assert all(r.text == md.render_input(r.row, "base", md.extra_lines(cells, r.row.nuclide))
               for r in shipped)


def test_r05_the_2pf_floor_is_the_c_zero_rms_and_leaves_three_members_outside(inputs):
    rows, _cells = inputs
    # rms(c = 0, t) for t = 2.53 fm, computed independently by quadrature of the 2pF moments
    a = 2.53 / (4 * math.log(3.0))
    h = a / 200
    moment2 = moment4 = 0.0
    for i in range(1, 60 * 200):
        s = i * h
        f = 1 / (1 + math.exp(s / a))
        moment2 += f * s ** 2
        moment4 += f * s ** 4
    assert md.fermi2_rms_floor_fm("2.53") == pytest.approx(math.sqrt(moment4 / moment2), rel=1e-6)
    outside = sorted((r.z, r.a) for r in rows if md.density_domain(r, "anchor") != md.INSIDE_2PF)
    assert outside == [(1, 1), (1, 3), (2, 4)]
    assert md.density_domain(_row(rows, 1, 1), "anchor") == "OUTSIDE_2PF_T2.3"


def test_r06_the_runner_keeps_only_the_headers_of_the_state_files(tmp_path):
    script = _script()
    d = tmp_path / "X1_anchor_on"
    d.mkdir()
    (d / "X1_anchor_on.xr.out").write_bytes(b"a\nb\nc\n")
    (d / "X1_anchor_on.K1.out").write_bytes(b"".join(b"%d\n" % i for i in range(20)))
    (d / "X1_anchor_on.K1-L2.tmat.out").write_bytes(b"m\n")
    script._keep_heads(d)
    assert (d / "X1_anchor_on.xr.out").read_bytes() == b"a\nb\nc\n"
    assert (d / "X1_anchor_on.K1.out").read_bytes() == b"0\n1\n2\n3\n4\n"
    assert not (d / "X1_anchor_on.K1-L2.tmat.out").exists()


# ----------------------------------------------------------------------------------------------------
# completeness before parsing
# ----------------------------------------------------------------------------------------------------

def test_r07_the_fixture_run_is_complete():
    headers, lines = md.read_recipe_run(FIXTURE, RUN, _text(), "0", "0")
    assert sorted(name for name, _e, _r in lines) == sorted(md.expected_printed_lines(
        md.input_keywords(_text())["xr_lines"]))
    assert md.orbit(md.N_MAX + 1, True) in headers and "K1" in headers
    assert all(len(h) == 5 for h in headers.values())


WHATS = ["line deleted", "rc", "err", "decimals", "state file", "header n", "truncated"]


@pytest.mark.parametrize("what", WHATS)
def test_r08_an_incomplete_or_malformed_run_is_refused_by_name(tmp_path, what):
    d = _copy_run(tmp_path)
    rc, err = "0", "0"
    xr = d / f"{RUN}.xr.out"
    rows = xr.read_bytes().decode("ascii").split("\n")
    if what == "line deleted":
        rows.pop(4)
        xr.write_bytes("\n".join(rows).encode("ascii"))
    elif what == "rc":
        rc = "1"
    elif what == "err":
        err = "12"
    elif what == "decimals":
        name, energy, rate = [f for f in rows[3].split("\t") if f]
        rows[3] = f"{name}\t{energy[:-1]}\t\t{rate}"
        xr.write_bytes("\n".join(rows).encode("ascii"))
    elif what == "state file":
        (d / f"{RUN}.M5.out").unlink()
    elif what == "header n":
        p = d / f"{RUN}.L2.out"
        p.write_bytes(p.read_bytes().replace(b"n = 2,", b"n = 3,"))
    elif what == "truncated":
        xr.write_bytes(xr.read_bytes()[:-1])
    clause = {"line deleted": "printed lines differ", "rc": "rc=1", "err": "err_bytes=12",
              "decimals": "malformed printed line", "state file": "no state file for M5",
              "header n": "state L2 header names", "truncated": "does not end with a newline"}[what]
    with pytest.raises(md.IncompleteRunError, match=f"{RUN}: .*{clause}"):
        md.read_recipe_run(d, RUN, _text(), rc, err)


def test_r09_the_collector_refuses_a_run_whose_input_is_not_the_one_rendered_now(
        tmp_path, inputs, monkeypatch):
    rows, cells = inputs
    script = _script()
    row = _row(rows, 82, 208)
    one = [r for r in md.recipe_runs([row], cells, "prod") if r.run == RUN]
    adapter = sys.modules["mudirac130"]   # the adapter module the script loaded by path
    monkeypatch.setattr(adapter, "recipe_runs", lambda _rows, _cells, _batch: one)
    d = _copy_run(tmp_path)
    digest = hashlib.sha256(_text().encode("ascii")).hexdigest()
    (tmp_path / "runs.tsv").write_bytes(f"{RUN}\t0\t0\t7.0\t{digest}\n".encode("ascii"))
    out = tmp_path / "out"
    script.main(["collect-recipe", "--batch", "prod", "--dir", str(tmp_path), "--out", str(out)])
    assert (out / "mudirac_recipe_prod_lines.csv").read_text().count("\n") == 1 + len(md.read_recipe_run(
        d, RUN, _text(), "0", "0")[1])
    (tmp_path / "runs.tsv").write_bytes(f"{RUN}\t0\t0\t7.0\t{'0' * 64}\n".encode("ascii"))
    with pytest.raises(SystemExit, match="not the input the renderer makes now"):
        script.main(["collect-recipe", "--batch", "prod", "--dir", str(tmp_path), "--out", str(out)])
    (tmp_path / "runs.tsv").write_bytes(f"{RUN}\t0\t0\t7.0\t{digest}\n".encode("ascii"))
    xr = d / f"{RUN}.xr.out"
    lines = xr.read_bytes().decode("ascii").split("\n")
    xr.write_bytes("\n".join(lines[:4] + lines[5:]).encode("ascii"))
    with pytest.raises(SystemExit, match=f"collect refused: {RUN}"):
        script.main(["collect-recipe", "--batch", "prod", "--dir", str(tmp_path), "--out", str(out)])


# ----------------------------------------------------------------------------------------------------
# the derivation
# ----------------------------------------------------------------------------------------------------

def _printed():
    headers, lines = md.read_recipe_run(FIXTURE, RUN, _text(), "0", "0")
    return headers, {name: energy for name, energy, _rate in lines}


def test_r10_the_anchor_is_the_hydrogen_like_header_plus_its_line_and_the_chain_telescopes():
    headers, lines = _printed()
    b = md.derive_recipe_bindings(RUN, "anchor", headers, lines, md.E_H_EV_2022)
    top, outer = md.orbit(md.N_MAX, True), md.orbit(md.N_MAX + 1, True)
    assert b[top][0] == -Decimal(headers[outer][3]) + Decimal(lines[md.anchor_line()])
    assert b["K1"][0] == b[top][0] + sum(Decimal(lines[ln]) for ln in md.upper_chain_lines())
    k, levels = md.recipe_quantities(b)
    assert k == b["K1"][0] / 1000
    assert len(levels) == md.N_MAX - 1
    floor = md.float_floor_eV(headers[top][4], md.E_H_EV_2022)
    assert Decimal("1.2e-8") < floor < Decimal("1.3e-8")


def test_r11_a_header_moved_beyond_its_rounding_breaks_the_derivation():
    headers, lines = _printed()
    outer = md.orbit(md.N_MAX + 1, True)
    moved = dict(headers)
    h = list(moved[outer])
    h[3] = str(Decimal(h[3]) + Decimal("1e-6"))
    moved[outer] = tuple(h)
    with pytest.raises(md.DerivationError, match=RUN):
        md.derive_recipe_bindings(RUN, "anchor", moved, lines, md.E_H_EV_2022)
    with pytest.raises(md.DerivationError, match="no printed line"):
        short = {k: v for k, v in lines.items() if k != md.anchor_line()}
        md.derive_recipe_bindings(RUN, "anchor", headers, short, md.E_H_EV_2022)


def test_r12_the_header_half_unit_is_half_the_last_printed_digit():
    assert md.header_half_unit("-116805.64593844541") == Decimal("5e-12")
    assert md.header_half_unit("1.2345e-05") == Decimal("5e-10")
    assert md.header_half_unit("-42") == Decimal("0.5")
