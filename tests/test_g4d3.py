"""The D3 energy tables of the ``mudirac130p1`` profile: their MuDirac inputs and printed outputs, the
derivation of the two tables from those printed strings, and the comparison with measured energies.

Every count here is derived at run time from the committed files; none is written down.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import json
import math
import pathlib
import re
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest
import test_g4parity as parity

from openmucf.g4 import d3_attribution as attribution
from openmucf.g4 import d3_centroids as centroids
from openmucf.g4 import d3_contract as d3c
from openmucf.g4 import provenance, spec
from openmucf.g4.sources import mudirac130 as md

REPO = pathlib.Path(__file__).resolve().parents[1]
D3DIR = REPO / md.D3_RELDIR
CELLS = REPO / md.CELLS_RELPATH
ORIGIN = REPO / md.ORIGIN_RELPATH
NL = "\n"


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def _lines(text: str) -> list[str]:
    return text.split(NL)


def _repeat_line(text: str, index: int) -> str:
    """The text with its line ``index`` written twice in a row."""
    lines = _lines(text)
    return NL.join([*lines[: index + 1], lines[index], *lines[index + 1 :]])


def _swap_lines(text: str, first: int) -> str:
    lines = _lines(text)
    lines[first], lines[first + 1] = lines[first + 1], lines[first]
    return NL.join(lines)


def _drill(tmp_path, source: pathlib.Path, mutate, error, message, loader):
    """Copy ``source``, check the copy loads, corrupt it one way, and require the named refusal."""
    text = source.read_bytes().decode("ascii")
    copy = tmp_path / source.name
    copy.write_bytes(text.encode("ascii"))
    loader(copy)
    mutated = mutate(text)
    assert mutated != text
    copy.write_bytes(mutated.encode("utf-8"))
    with pytest.raises(error, match=re.escape(message)):
        loader(copy)


# --------------------------------------------------------------------------------------------
# T-106 -- the measured energies (loaders)
# --------------------------------------------------------------------------------------------

_G = "Fricke1995,21,45,2p1/2-1s1/2,K1-L2,855.185,0.041,"
_U = "Fricke1995,4,9,2p-1s centre of gravity,,33.402,0.010,statistical,0.001,false,centroid,"

CELL_DRILLS = [
    ("a carriage return", lambda t: t.replace(NL, "\r" + NL, 1),
     md.CarriageReturnError, "contains CR"),
    ("a non-ASCII byte", lambda t: _replace_once(t, ",,33.402,", ",,33·402,"),
     md.NonAsciiError, "outside US-ASCII"),
    ("a renamed column", lambda t: _replace_once(t, "unc_keV,unc_label", "unc,unc_label"),
     md.HeaderError, "header is"),
    ("a missing cell", lambda t: _replace_once(t, _U, _U.replace(",false,", ",")),
     md.CellError, "cells expected"),
    ("an unknown source", lambda t: _replace_once(t, NL + _U, NL + "Fricke1996" + _U[len("Fricke1995"):]),
     md.CellError, "is not one of"),
    ("a signed Z", lambda t: _replace_once(t, "Fricke1995,4,9,", "Fricke1995,+4,9,"),
     md.CellError, "Z must be an integer"),
    ("a value in another form", lambda t: _replace_once(t, ",33.402,", ",33402e-3,"),
     md.CellError, "value_keV must be digits"),
    ("an uncertainty at other decimals", lambda t: _replace_once(t, ",33.402,0.010,", ",33.402,0.01,"),
     md.CellError, "is not at the decimals"),
    ("a malformed NPol", lambda t: _replace_once(t, _U, _U.replace(",0.001,", ",n/a,")),
     md.CellError, "npol_keV must be empty"),
    ("a gated row without its line", lambda t: _replace_once(t, _G, _G.replace(",K1-L2,", ",,")),
     md.CellError, "names its MuDirac line"),
    ("an ungated row without its reason", lambda t: _replace_once(t, _U, _U.replace(",centroid,", ",,")),
     md.CellError, "carries one of"),
    ("a gated row with a reason",
     lambda t: _replace_once(t, _G + "statistical,0.182,true,,", _G + "statistical,0.182,true,centroid,"),
     md.CellError, "carries one of"),
    ("an empty locator", lambda t: _replace_once(t, 'centroid,"Table IIIA, p. 205, row 9Be"', "centroid,"),
     md.CellError, "carry a locator"),
    ("a Table IIIA row relabelled",
     lambda t: _replace_once(t, _U, _U.replace(",statistical,", ",statistical and fit,")),
     md.CellError, "line 2: unc_label 'statistical and fit' is not 'statistical', the label of Table IIIA"),
    ("rows out of order", lambda t: _replace_once(t, NL + _U, NL + _U.replace(",4,9,", ",99,9,")),
     md.OrderError, "ordered by (source, Z, A)"),
    ("a duplicated key", lambda t: _repeat_line(t, 2),
     md.DuplicateKeyError, "duplicate row"),
    ("no rows", lambda t: _lines(t)[0] + NL,
     md.EmptyError, "carries no rows"),
]

ORIGIN_DRILLS = [
    ("an unknown origin", lambda t: _replace_once(t, "21,45,muonic,", "21,45,optical,"),
     md.CellError, "is not one of"),
    ("an empty locator", lambda t: NL.join([_lines(t)[0], "21,45,muonic,", *_lines(t)[2:]]),
     md.CellError, "carry a locator"),
    ("a duplicated key", lambda t: _repeat_line(t, 1),
     md.DuplicateKeyError, "duplicate key"),
    ("rows out of order", lambda t: _swap_lines(t, 1),
     md.OrderError, "ascending by (Z, A)"),
    ("no rows", lambda t: _lines(t)[0] + NL,
     md.EmptyError, "carries no rows"),
]


def test_t106_the_transcriptions_load_and_the_origin_file_names_exactly_the_gated_nuclides():
    cells, origins = md.load_validation(CELLS, ORIGIN)
    gated = [cell for cell in cells if cell.gated]
    assert set(origins) == set(md.gated_nuclides(cells))
    assert {cell.source for cell in cells} == set(md.VALIDATION_SOURCES)
    assert all(cell.reason in md.UNGATED_REASONS for cell in cells if not cell.gated)
    print(f"\nvalidation cells: rows {len(cells)} gated {len(gated)} nuclides {len(origins)}")


@pytest.mark.parametrize("label, mutate, error, message", CELL_DRILLS, ids=[d[0] for d in CELL_DRILLS])
def test_t106_drill_each_cells_rule_refuses_its_fixture(tmp_path, label, mutate, error, message):
    _drill(tmp_path, CELLS, mutate, error, message, md.load_cells)


@pytest.mark.parametrize("label, mutate, error, message", ORIGIN_DRILLS, ids=[d[0] for d in ORIGIN_DRILLS])
def test_t106_drill_each_origin_rule_refuses_its_fixture(tmp_path, label, mutate, error, message):
    _drill(tmp_path, ORIGIN, mutate, error, message, md.load_radius_origins)


def test_t106_drill_an_origin_file_that_misses_a_gated_nuclide_is_refused(tmp_path):
    lines = _lines(ORIGIN.read_bytes().decode("ascii"))
    copy = tmp_path / ORIGIN.name
    copy.write_bytes(NL.join(lines[:1] + lines[2:]).encode("ascii"))
    md.load_radius_origins(copy)
    with pytest.raises(md.CellError, match="the gated nuclides of"):
        md.load_validation(CELLS, copy)


# --------------------------------------------------------------------------------------------
# T-103 -- the MuDirac inputs
# --------------------------------------------------------------------------------------------

INPUTS = REPO / md.INPUTS_RELPATH
_I1 = "1,1,H,0.8783,0.0086,1.133880424324658,2.3,1,0.8783,"

INPUT_DRILLS = [
    ("a carriage return", lambda t: t.replace(NL, "\r" + NL, 1),
     md.CarriageReturnError, "contains CR"),
    ("a renamed column", lambda t: _replace_once(t, "most_abundant,", "abundant,"),
     md.HeaderError, "header is"),
    ("a lower-case symbol", lambda t: _replace_once(t, NL + _I1, NL + _I1.replace(",H,", ",h,")),
     md.CellError, "element symbol"),
    ("an rms radius the sphere radius does not imply",
     lambda t: _replace_once(t, NL + _I1, NL + _I1.replace(",H,0.8783,", ",H,0.8784,")),
     md.CellError, "rms_fm is not radius_fm"),
    ("another skin thickness", lambda t: _replace_once(t, NL + _I1, NL + _I1.replace(",2.3,", ",2.30,")),
     md.CellError, "fermi_t_fm must be"),
    ("another source", lambda t: t.replace("; charge_radii.csv", "; charge-radii.csv", 1),
     md.CellError, "source must name"),
    ("a duplicated key", lambda t: _repeat_line(t, 1),
     md.DuplicateKeyError, "duplicate key"),
    ("rows out of order", lambda t: _swap_lines(t, 1),
     md.OrderError, "ascending by (Z, A)"),
    ("no rows", lambda t: _lines(t)[0] + NL,
     md.EmptyError, "carries no rows"),
]


@pytest.mark.parametrize("label, mutate, error, message", INPUT_DRILLS, ids=[d[0] for d in INPUT_DRILLS])
def test_t103_drill_each_inputs_rule_refuses_its_fixture(tmp_path, label, mutate, error, message):
    _drill(tmp_path, INPUTS, mutate, error, message, md.load_inputs)


def test_t103_a_source_file_off_its_pin_is_refused():
    with pytest.raises(md.Mudirac130Error, match="nuclear_radii.dat is blob"):
        md.build_inputs(b"1 1 1.0\n", b"", b"")


def test_t103_every_input_is_fit_free_and_every_run_has_one_argument():
    """Every input a committed kind renders carries the fit keyword exactly once and set FALSE, the
    fixed settings, the chain, and on a validation nuclide every gated line the chain lacks; the
    command line holds the input file and nothing else."""
    rows = md.load_inputs(INPUTS)
    cells, _ = md.load_validation(CELLS, ORIGIN)
    validation = set(md.gated_nuclides(cells))
    chain = md.chain_lines()
    rendered = 0
    for row in rows:
        extra = md.extra_lines(cells, row.nuclide)
        for kind in md.run_kinds(row, validation):
            text = md.render_input(row, kind, extra)
            rendered += 1
            assert text.count("optimise_fermi_parameters") == 1, (row, kind)
            assert "optimise_fermi_parameters: FALSE\n" in text, (row, kind)
            xr = next(line for line in text.splitlines() if line.startswith("xr_lines: "))
            specs = xr[len("xr_lines: "):].split(",")
            assert specs[: len(md.chain_specs())] == md.chain_specs()
            assert specs[len(md.chain_specs()):] == ([] if kind in md.IDEAL_KINDS else extra)
            assert not set(extra) & chain
    for row in rows:
        gated = {cell.quantity for cell in cells if cell.gated and cell.nuclide == row.nuclide}
        assert gated <= chain | set(md.extra_lines(cells, row.nuclide)), row
    assert md.mudirac_argv("mudirac", "X1_base.in") == ["mudirac", "X1_base.in"]
    print(f"\nrendered inputs: {rendered} over {len(rows)} members; validation nuclides {len(validation)}")


def test_t103_the_radius_disagreements_are_listed_and_left_as_bundled():
    rows = md.load_inputs(INPUTS)
    listed = md.radius_disagreements(rows)
    for z, a, bundled, table in listed:
        (row,) = [r for r in rows if r.nuclide == (z, a)]
        assert row.radius_fm and row.iaea_rms_fm == table and bundled != table
    print(f"\nbundled rms against the charge-radii table at 4 decimals, differing: {listed}")


# --------------------------------------------------------------------------------------------
# T-104 -- the committed outputs: every run the input set implies, each input as rendered now
# --------------------------------------------------------------------------------------------

RUNS = REPO / md.RUNS_RELPATH
STATES = REPO / md.STATES_RELPATH
LINES = REPO / md.LINES_RELPATH


@functools.lru_cache(maxsize=1)
def _outputs() -> md.Outputs:
    """The committed outputs, loaded once for the tests that only read them."""
    return md.load_outputs(REPO)


def _committed_runs() -> list[dict[str, str]]:
    text = RUNS.read_bytes().decode("ascii")
    assert "\r" not in text
    return list(csv.DictReader(io.StringIO(text)))


def test_t104_every_committed_input_digest_is_the_input_the_renderer_makes_now():
    """G-stale, read with this test's own CSV reader: the run table lists, batch by batch and in order, the
    runs `recipe_runs` implies for the committed inputs, each with the sha256 of the input the renderer makes
    at this commit; and the renderer's settings are the ones the document states."""
    rows = md.load_inputs(INPUTS)
    cells, _ = md.load_validation(CELLS, ORIGIN)
    committed = _committed_runs()
    implied = [r for batch in md.RECIPE_BATCHES for r in md.recipe_runs(rows, cells, batch)]
    assert [r["run"] for r in committed] == [r.run for r in implied]
    for got, want in zip(committed, implied, strict=True):
        assert got["input_sha256"] == hashlib.sha256(want.text.encode("ascii")).hexdigest(), want.run
    assert md.Settings(k=3, n_u=60, n_s=12, t=7) == md.RECIPE_SETTINGS
    per_batch = {batch: len(md.recipe_runs(rows, cells, batch)) for batch in md.RECIPE_BATCHES}
    print(f"\nrun table rows {len(committed)} = rendered inputs {len(implied)}; per batch {per_batch}")


def _data_copy(tmp_path) -> pathlib.Path:
    """A copy of the committed D3 CSV files under ``tmp_path``, at their relative paths."""
    target = tmp_path / md.D3_RELDIR
    target.mkdir(parents=True)
    for path in D3DIR.glob("*.csv"):
        (target / path.name).write_bytes(path.read_bytes())
    return tmp_path


def test_t104_drill_a_stale_input_digest_and_a_renderer_off_its_settings_are_refused(tmp_path, monkeypatch):
    """G-stale's two drills: one committed digest replaced by that of the same input with `uehling_steps: 90`
    is refused; and with the committed table unchanged, a renderer whose setting is N_u = 90 refuses it."""
    rows = md.load_inputs(INPUTS)
    cells, _ = md.load_validation(CELLS, ORIGIN)
    first = md.recipe_runs(rows, cells, "prod")[0]
    root = _data_copy(tmp_path)
    md.load_outputs(root)
    digest = hashlib.sha256(first.text.encode("ascii")).hexdigest()
    moved = hashlib.sha256(_replace_once(first.text, "uehling_steps: 60\n", "uehling_steps: 90\n")
                           .encode("ascii")).hexdigest()
    runs = root / md.RUNS_RELPATH
    runs.write_bytes(_replace_once(runs.read_bytes().decode("ascii"), digest, moved).encode("ascii"))
    with pytest.raises(md.CellError, match=f"input_sha256 '{moved}', implied '{digest}'"):
        md.load_outputs(root)
    monkeypatch.setattr(md, "RECIPE_SETTINGS", md.Settings(k=3, n_u=90, n_s=12, t=7))
    with pytest.raises(md.CellError, match="input_sha256"):
        md.load_outputs(REPO)


_R1 = "H1_anchor_on,1,1,anchor,recipe,on,prod,OUTSIDE_2PF_T2.3,0,0,"
_S1 = "H1_anchor_on,K1,1,0,1,"
_L1 = "H1_anchor_on,K1-L2,1898.113802710279"


def _first_shipped_line(text: str) -> str:
    return next(line for line in _lines(text) if "_shipped," in line)


def _move_first_row_of_second_run_up(text: str) -> str:
    lines = _lines(text)
    first_run = lines[1].split(",")[0]
    index = next(i for i, line in enumerate(lines[1:], start=1) if not line.startswith(first_run + ","))
    return NL.join([lines[0], lines[index], *lines[1:index], *lines[index + 1:]])


def _extra_state_of_a_hydrogen_like_run(text: str) -> str:
    lines = _lines(text)
    index = next(i for i, line in enumerate(lines) if line.startswith("He3_ideal6_on,"))
    return NL.join([*lines[:index], "He3_ideal6_on,O9,5,4,1,-1.0", *lines[index:]])


OUTPUT_DRILLS = [
    (RUNS, "a carriage return", lambda t: t.replace(NL, "\r" + NL, 1), md.CarriageReturnError, "contains CR"),
    (RUNS, "a renamed column", lambda t: _replace_once(t, "rc,err_bytes", "rc,err"), md.HeaderError,
     "header is"),
    (RUNS, "a run the input set does not imply",
     lambda t: _replace_once(t, NL + _R1, NL + _R1.replace("H1_anchor_on", "H1_anchor_om")),
     md.CellError, "where the input set implies 'H1_anchor_on'"),
    (RUNS, "a run under another kind",
     lambda t: _replace_once(t, NL + _R1, NL + _R1.replace(",anchor,", ",rsig,")),
     md.CellError, "kind 'rsig', implied 'anchor'"),
    (RUNS, "a domain label the radius does not imply",
     lambda t: _replace_once(t, NL + _R1, NL + _R1.replace("OUTSIDE_2PF_T2.3", "INSIDE")),
     md.CellError, "density_domain 'INSIDE', implied 'OUTSIDE_2PF_T2.3'"),
    (RUNS, "a nonzero exit status", lambda t: _replace_once(t, NL + _R1, NL + _R1.replace(",0,0,", ",1,0,")),
     md.CellError, "the collector admits clean runs only"),
    (RUNS, "an error size that is not a count",
     lambda t: _replace_once(t, NL + _R1, NL + _R1.replace(",0,0,", ",0,missing,")),
     md.CellError, "must be integers"),
    (RUNS, "a missing run", lambda t: NL.join([*_lines(t)[:1], *_lines(t)[2:]]), md.CellError,
     "the input set implies"),
    (STATES, "a state that is not an orbit",
     lambda t: _replace_once(t, NL + _S1, NL + _S1.replace(",K1,", ",1s,")),
     md.CellError, "IUPAC orbit"),
    (STATES, "quantum numbers that do not match the orbit",
     lambda t: _replace_once(t, NL + _S1, NL + _S1.replace(",1,0,1,", ",1,0,0,")), md.CellError,
     "do not match K1"),
    (STATES, "an energy in another form",
     lambda t: _replace_once(t, NL + _S1 + "-2530.2622457005946" + NL, NL + _S1 + "-2530.26 eV" + NL),
     md.CellError, "printed number"),
    (STATES, "a duplicated state", lambda t: _repeat_line(t, 1), md.DuplicateKeyError,
     "duplicate K1 of H1_anchor_on"),
    (STATES, "rows not grouped by run", _move_first_row_of_second_run_up, md.OrderError,
     "grouped by run in the run table's order"),
    (STATES, "a missing state", lambda t: NL.join([*_lines(t)[:1], *_lines(t)[2:]]), md.CellError,
     "H1_anchor_on: the committed states are not the states its committed view names"),
    (STATES, "a state a hydrogen-like run does not commit", _extra_state_of_a_hydrogen_like_run, md.CellError,
     "He3_ideal6_on: the committed states are not the states its committed view names"),
    (STATES, "states out of orbit order", lambda t: _swap_lines(t, 1), md.OrderError,
     "state rows are in orbit order"),
    (LINES, "a line at other decimals", lambda t: _replace_once(t, NL + _L1, NL + _L1[:-1]), md.CellError,
     f"must carry {md.RECIPE_XR_PRECISION} decimals"),
    (LINES, "a v1.3.0 line at the recipe's decimals",
     lambda t: _replace_once(t, NL + _first_shipped_line(t) + NL,
                             NL + _first_shipped_line(t) + "000000" + NL),
     md.CellError, f"must carry {md.SHIPPED_XR_PRECISION} decimals"),
    (LINES, "a line that is not orbit to orbit",
     lambda t: _replace_once(t, NL + _L1, NL + _L1.replace(",K1-L2,", ",Ka1,")),
     md.CellError, "orbit-orbit"),
    (LINES, "a duplicated line", lambda t: _repeat_line(t, 1), md.DuplicateKeyError,
     "duplicate K1-L2 of H1_anchor_on"),
    (LINES, "a missing line", lambda t: NL.join([*_lines(t)[:1], *_lines(t)[2:]]), md.CellError,
     "H1_anchor_on: the committed lines are not the lines its input implies"),
]


@pytest.mark.parametrize("source, label, mutate, error, message", OUTPUT_DRILLS,
                         ids=[f"{d[0].stem}-{d[1]}" for d in OUTPUT_DRILLS])
def test_t104_drill_each_outputs_rule_refuses_its_fixture(tmp_path, source, label, mutate, error, message):
    """Each refusal of `load_outputs` on a copy of the committed files that carries only its defect: the
    unmutated copy loads first."""
    root = _data_copy(tmp_path)
    if label == OUTPUT_DRILLS[0][1]:
        md.load_outputs(root)
    copy = root / md.D3_RELDIR / source.name
    text = copy.read_bytes().decode("ascii")
    mutated = mutate(text)
    assert mutated != text
    copy.write_bytes(mutated.encode("utf-8"))
    with pytest.raises(error, match=re.escape(message)):
        md.load_outputs(root)


def test_t104_the_committed_view_keeps_every_line_and_only_the_checked_states_of_a_hydrogen_like_run():
    out = _outputs()
    kinds: dict[str, int] = {}
    for run, row in out.runs.items():
        if row.kind in md.IDEAL_KINDS:
            assert run not in out.lines and set(out.headers[run]) == set(md.CHECKED_STATES), run
        else:
            states = {state for line in out.lines[run] for state in line.split("-")}
            assert set(out.headers[run]) == states, run
        kinds[row.kind] = kinds.get(row.kind, 0) + 1
    print(f"\nruns by kind: {kinds}")


def test_t104_the_derivation_holds_on_the_anchor_and_rsig_runs_of_every_kept_member():
    out = _outputs()
    kept = md.kept_members(out)
    checked = 0
    for row in kept:
        for kind in ("anchor", "rsig"):
            for uehling in (True, False):
                run = md.recipe_run_id(row, kind, uehling)
                assert set(md.run_bindings(out, run)) == set(md.chain_states())
                checked += 1
    print(f"\nderivation checked on {checked} runs of {len(kept)} kept members")


def test_t104_drill_the_derivation_refuses_a_moved_cross_line_and_a_missing_state():
    out = _outputs()
    row = md.kept_members(out)[0]
    run = md.recipe_run_id(row, "anchor", True)
    r = out.runs[run]
    headers, lines = dict(out.headers[run]), dict(out.lines[run])
    md.derive_recipe_bindings(run, "anchor", headers, lines, r.r15_total, md.eh_of(r.stage))
    cross = md.cross_lines()[0]
    moved = {**lines, cross: format(Decimal(lines[cross]) + Decimal("1e-5"), f".{md.RECIPE_XR_PRECISION}f")}
    with pytest.raises(md.DerivationError, match=f"cross line {cross}"):
        md.derive_recipe_bindings(run, "anchor", headers, moved, r.r15_total, md.eh_of(r.stage))
    missing = {state: text for state, text in headers.items() if state != "L2"}
    with pytest.raises(md.DerivationError, match="no printed header for L2"):
        md.derive_recipe_bindings(run, "anchor", missing, lines, r.r15_total, md.eh_of(r.stage))


def test_t104_drill_a_header_just_past_its_rounding_is_refused_and_one_just_inside_passes():
    """G-anchor on a committed run: the printed S17 header of an anchor run moved by its whole allowance
    plus one unit of its 17th digit is refused, moved by its allowance less one unit passes; the run stays
    complete either way, so the collector's completeness check passes on both."""
    out = _outputs()
    row = md.kept_members(out)[0]
    run = md.recipe_run_id(row, "anchor", True)
    r = out.runs[run]
    headers, lines = dict(out.headers[run]), dict(out.lines[run])
    outer = md.orbit(md.N_MAX + 1, True)
    header = Decimal(headers[outer])
    unit = 2 * md.header_half_unit(headers[outer])
    floor = md.float_floor_eV(r.r15_total, md.eh_of(r.stage))
    rho = Decimal(5) * Decimal(10) ** (-md.RECIPE_XR_PRECISION - 1) + floor
    b = md.derive_recipe_bindings(run, "anchor", headers, lines, r.r15_total, md.eh_of(r.stage))
    top = md.orbit(md.N_MAX, True)
    with localcontext() as context:
        context.prec = 50
        # The top orbit's own check: |B(R15) + header(R15)| <= half(R15) + floor + rounding(R15) + rho, where
        # rounding(R15) = half(S17) + floor + rho. A shift s of the S17 header moves B(R15) by -s.
        gap = b[top][0] + Decimal(headers[top])
        allowance = (md.header_half_unit(headers[top]) + floor + md.header_half_unit(headers[outer]) + floor
                     + 2 * rho)
        room = allowance - abs(gap)
        steps_inside = int((room / unit).to_integral_value(rounding="ROUND_FLOOR")) - 1
        # A shift s of the S17 header moves the gap by -s; the sign that widens it is the gap's opposite.
        shift_sign = -1 if gap >= 0 else 1
        inside = header + shift_sign * steps_inside * unit
        outside = header + shift_sign * (steps_inside + 3) * unit
    md.derive_recipe_bindings(run, "anchor", {**headers, outer: str(inside)}, lines, r.r15_total,
                              md.eh_of(r.stage))
    with pytest.raises(md.DerivationError, match=f"derived {top} lies"):
        md.derive_recipe_bindings(run, "anchor", {**headers, outer: str(outside)}, lines, r.r15_total,
                                  md.eh_of(r.stage))


# --------------------------------------------------------------------------------------------
# T-105 -- the two tables, re-derived by an independent route
# --------------------------------------------------------------------------------------------

KSHELL_LAYER1 = D3DIR / f"d3_kshell.{md.PROFILE}.g4dat"
KSHELL_LAYER2 = D3DIR / f"d3_kshell.{md.PROFILE}.prov.json"
LEVELS_LAYER1 = D3DIR / f"d3_levels.{md.PROFILE}.g4dat"
LEVELS_LAYER2 = D3DIR / f"d3_levels.{md.PROFILE}.prov.json"
_CARRIES = re.compile(r"^Z=([1-9][0-9]*) A=0 [(]carries A=([1-9][0-9]*)[)]$")


def _shipped(layer1: pathlib.Path, layer2: pathlib.Path):
    table = spec.parse(layer1.read_bytes().decode("ascii"))
    document = provenance.from_json_obj(json.loads(layer2.read_bytes().decode("ascii")))
    return table, document


def _records(table) -> dict[tuple[int, int], tuple]:
    return {(record[0], record[1]): tuple(record[2:]) for record in table.records}


def _carried(document) -> dict[tuple[int, int], int]:
    """``{(Z, A): the isotope the row's values belong to}``, read from each row's validity range."""
    out = {}
    for key, row in document.rows.items():
        z, a = (int(part) for part in key.split("-"))
        match = _CARRIES.match(row.validity_range)
        out[(z, a)] = int(match.group(2)) if a == 0 else a
        assert (a == 0) is bool(match), (key, row.validity_range)
    return out


def _half_unit(text: str) -> Decimal:
    """Half a unit of the last digit ``text`` prints, read off the printed string itself."""
    return Decimal(5) * Decimal(10) ** (Decimal(text).as_tuple().exponent - 1)


#: The most printed lines the derivation sums to reach any circular state from the anchor: every
#: upper-chain line down to K1, then every lower-chain line out again.
_MOST_LINES = 2 * (md.N_MAX - 1)


def _ulp_floor(total_ev: str, eh: float) -> Decimal:
    """The spacing of the doubles a run holds its total energy on, in eV, computed here with math.ulp
    from the printed total of its outermost circular state."""
    return Decimal(math.ulp(float(total_ev) / eh)) * Decimal(repr(eh))


def _line_rounding(run: md.RunRow, lines: dict[str, str]) -> Decimal:
    """What one printed line of ``run`` carries: half a unit of its last printed decimal and the spacing of
    the doubles its total energy is held on."""
    eh = md.E_H_EV_2022 if run.stage == "recipe" else md.E_H_EV_MUDIRAC
    return max(_half_unit(text) for text in lines.values()) + _ulp_floor(run.r15_total, eh)


def _from_headers(out: md.Outputs, run: str) -> list[tuple[Decimal, Decimal]]:
    """``[(quantity, bound)]`` in keV for K then e2 .. e<N_MAX>, from the run's printed state headers alone:
    each quantity with the largest distance the derivation may put between it and this value -- the
    header's own rounding, the anchor's (the S17 header, its float spacing and one line), and one line's
    rounding for each line summed on the longest path, all read off the printed strings."""
    headers, lines, r = out.headers[run], out.lines[run], out.runs[run]
    rho = _line_rounding(r, lines)
    floor = _ulp_floor(r.r15_total, md.E_H_EV_2022 if r.stage == "recipe" else md.E_H_EV_MUDIRAC)
    anchor = _half_unit(headers[md.orbit(md.N_MAX + 1, True)]) + floor + rho

    def bound(state: str) -> Decimal:
        return _half_unit(headers[state]) + floor + anchor + (_MOST_LINES + 1) * rho

    with localcontext() as context:
        context.prec = 50
        result = [(-Decimal(headers["K1"]) / 1000, bound("K1") / 1000)]
        for n in range(2, md.N_MAX + 1):
            ell = n - 1
            lower, upper = md.orbit(n, False), md.orbit(n, True)
            mean = ((2 * ell * -Decimal(headers[lower]) + (2 * ell + 2) * -Decimal(headers[upper]))
                    / (4 * ell + 2))
            result.append((mean / 1000, max(bound(lower), bound(upper)) / 1000))
    return result


def _within(value: float, expected: Decimal, bound: Decimal) -> bool:
    """``value`` (as shipped, a double) within ``bound`` of ``expected``, allowing the double's own
    rounding of a decimal quotient."""
    shipped = Decimal(repr(value))
    return abs(shipped - expected) <= bound + abs(expected) * Decimal("1e-15")


def unit_problems(k_records: dict[tuple[int, int], tuple], level_records: dict[tuple[int, int], tuple],
                  carried: dict[tuple[int, int], int]) -> list[str]:
    """G-unit: every `value` and `e<n>` cell set beside the keV this test computes from the raw eV the anchor
    run printed in its state headers; every row whose cells it does not give, named."""
    out = _outputs()
    inputs = {row.nuclide: row for row in out.inputs}
    problems = []
    for (z, a), (value, _unc) in k_records.items():
        source = inputs[(z, carried[(z, a)])]
        run = md.recipe_run_id(source, "anchor", True)
        shipped = [value, *level_records[(z, a)][: md.N_MAX - 1]]
        for got, (quantity, bound) in zip(shipped, _from_headers(out, run), strict=True):
            if not _within(got, quantity, bound):
                problems.append(f"Z={z} A={a}: {got!r} is not {quantity} keV within {bound}")
                break
    return problems


def test_t105_every_value_is_the_kev_of_the_raw_ev_headers_within_their_rounding():
    """G-unit: each shipped value within the printed headers' own rounding of their eV / 1000; the header
    route sums no printed line, so it is not the generator's route."""
    kshell, kshell_doc = _shipped(KSHELL_LAYER1, KSHELL_LAYER2)
    levels, levels_doc = _shipped(LEVELS_LAYER1, LEVELS_LAYER2)
    carried = _carried(kshell_doc)
    assert carried == _carried(levels_doc)
    assert unit_problems(_records(kshell), _records(levels), carried) == []
    print(f"\nvalues checked against the eV headers: {len(_records(kshell))} rows x {md.N_MAX} quantities")


def test_t105_drill_a_table_in_ev_labelled_kev_parses_validates_and_is_refused(tmp_path):
    """G-unit's drill: a copy of `k_shell_energy` whose values are the committed ones times 1000 -- eV
    labelled keV -- still parses and validates, so the format's guards pass; the unit check names it."""
    text = KSHELL_LAYER1.read_bytes().decode("ascii")
    table = spec.parse(text)
    scaled = spec.G4DatTable(directives=table.directives,
                             records=tuple((z, a, value * 1000, unc) for z, a, value, unc in table.records))
    copy = tmp_path / KSHELL_LAYER1.name
    copy.write_bytes(spec.render(scaled).encode("ascii"))
    reparsed = spec.parse(copy.read_bytes().decode("ascii"))
    spec.validate(reparsed)
    levels, levels_doc = _shipped(LEVELS_LAYER1, LEVELS_LAYER2)
    carried = _carried(levels_doc)
    assert unit_problems(_records(table), _records(levels), carried) == []
    problems = unit_problems(_records(reparsed), _records(levels), carried)
    assert len(problems) == len(table.records), problems[:3]


def _route_relative(lines: dict[str, str]) -> dict[str, tuple[Decimal, int]]:
    """``{state: (binding energy relative to the anchor in eV, lines summed)}`` by a route other than
    the generator's: upper orbits down from the anchor by the upper-chain lines, the lower orbit of
    shell 2 from K1 by its line, and each lower orbit of shell n + 1 from the upper orbit of shell n
    by the cross line."""
    out = {md.orbit(md.N_MAX, True): (Decimal(0), 0)}
    for n in range(md.N_MAX - 1, 0, -1):
        lower, upper = md.orbit(n, True), md.orbit(n + 1, True)
        energy, count = out[upper]
        out[lower] = (energy + Decimal(lines[f"{lower}-{upper}"]), count + 1)
    energy, count = out["K1"]
    out[md.orbit(2, False)] = (energy - Decimal(lines[f"K1-{md.orbit(2, False)}"]), count + 1)
    for n in range(2, md.N_MAX):
        upper, lower = md.orbit(n, True), md.orbit(n + 1, False)
        energy, count = out[upper]
        out[lower] = (energy - Decimal(lines[f"{upper}-{lower}"]), count + 1)
    return out


def _route_quantities(lines: dict[str, str]) -> list[tuple[Decimal, int]]:
    """``[(quantity in keV, lines summed)]`` for K then e2 .. e<N_MAX>, on the other route."""
    with localcontext() as context:
        context.prec = 50
        rel = _route_relative(lines)
        result = [(rel["K1"][0] / 1000, rel["K1"][1])]
        for n in range(2, md.N_MAX + 1):
            ell = n - 1
            (low, low_count), (high, high_count) = rel[md.orbit(n, False)], rel[md.orbit(n, True)]
            mean = (2 * ell * low + (2 * ell + 2) * high) / (4 * ell + 2)
            result.append((mean / 1000, max(low_count, high_count)))
    return result


def test_t105_every_unc_is_floored_and_agrees_with_an_independent_route_within_its_rounding():
    """Every shipped `unc` and `u<n>` re-derived by the other route, anchor-relative: each cell is at
    least this test's floor, and lies within that floor plus the two routes' rounding bound of the
    other route's change (floored the same way). The largest difference and the floor are printed:
    the difference is the route noise, and it must lie below the floor."""
    out = _outputs()
    inputs = {row.nuclide: row for row in out.inputs}
    kshell, kshell_doc = _shipped(KSHELL_LAYER1, KSHELL_LAYER2)
    levels, _ = _shipped(LEVELS_LAYER1, LEVELS_LAYER2)
    k_records, level_records = _records(kshell), _records(levels)
    carried = _carried(kshell_doc)
    largest, floors, checked = Decimal(0), set(), 0
    for (z, a), (_value, unc) in k_records.items():
        source = inputs[(z, carried[(z, a)])]
        anchor, rsig = md.recipe_run_id(source, "anchor", True), md.recipe_run_id(source, "rsig", True)
        rho = max(_line_rounding(out.runs[anchor], out.lines[anchor]),
                  _line_rounding(out.runs[rsig], out.lines[rsig]))
        floor = 2 * _MOST_LINES * rho / 1000
        floors.add(floor)
        shipped_unc = [unc, *level_records[(z, a)][md.N_MAX - 1 :]]
        base, moved = _route_quantities(out.lines[anchor]), _route_quantities(out.lines[rsig])
        for got, (quantity, count), (quantity_moved, count_moved) in zip(
            shipped_unc, base, moved, strict=True
        ):
            shipped = Decimal(repr(got))
            assert shipped >= floor * (1 - Decimal("1e-15")), ((z, a), got, floor)
            expected = max(abs(quantity_moved - quantity), floor)
            route_bound = (2 * _MOST_LINES + count + count_moved) * rho / 1000
            difference = abs(shipped - expected)
            assert difference <= floor + route_bound + expected * Decimal("1e-15"), (
                (z, a), got, expected, route_bound)
            largest = max(largest, difference)
            checked += 1
    print(f"\nuncertainty cells checked on the other route: {checked}; largest route difference "
          f"{largest} keV; floors {sorted(floors)} keV")
    assert largest < min(floors), (largest, floors)


def test_t105_drill_a_moved_rsig_anchor_header_leaves_the_uncertainties_unchanged():
    """The outermost header of a kept member's `rsig` run moved by one printed unit: the absolute
    derivation's K moves with it, and the shipped uncertainties, which hold that state fixed, do not."""
    out = _outputs()
    kshell, _ = _shipped(KSHELL_LAYER1, KSHELL_LAYER2)
    levels, _ = _shipped(LEVELS_LAYER1, LEVELS_LAYER2)
    k_records, level_records = _records(kshell), _records(levels)
    top = md.orbit(md.N_MAX, True)
    for row in md.kept_members(out):
        anchor_run, rsig_run = md.recipe_run_id(row, "anchor", True), md.recipe_run_id(row, "rsig", True)
        r = out.runs[rsig_run]
        headers, lines = out.headers[rsig_run], out.lines[rsig_run]
        printed = Decimal(headers[top])
        moved_headers = {**headers, top: str(printed + 2 * md.header_half_unit(headers[top]))}
        try:
            moved = md.derive_recipe_bindings(rsig_run, "rsig", moved_headers, lines, r.r15_total,
                                              md.eh_of(r.stage))
        except md.DerivationError:
            continue
        original = md.run_bindings(out, rsig_run)
        moved_values = {state: value for state, (value, _rounding) in moved.items()}
        assert md.quantities(moved_values)[0] != md.quantities(original)[0]
        floor = md.unc_floor_kev(out, (anchor_run, rsig_run))
        unc, level_uncs = md.uncertainties(md.run_bindings(out, anchor_run), moved_values, floor)
        shipped = (k_records[row.nuclide][1], *level_records[row.nuclide][md.N_MAX - 1 :])
        assert tuple(float(u) for u in (unc, *level_uncs)) == shipped, row.nuclide
        print(f"\nZ={row.z} A={row.a}: rsig {top} header {headers[top]} -> {moved_headers[top]}; "
              "uncertainties unchanged")
        return
    pytest.fail("no kept member derives with its rsig anchor moved")


def test_t105_natural_rows_equal_their_carried_isotope_and_both_tables_share_one_key_set():
    """V-16's Python mirror, the (Z, 0) copy rule, and the kept set: the key set is every kept
    member plus a (Z, 0) row for each Z whose most abundant isotope is kept."""
    out = _outputs()
    kshell, kshell_doc = _shipped(KSHELL_LAYER1, KSHELL_LAYER2)
    levels, _ = _shipped(LEVELS_LAYER1, LEVELS_LAYER2)
    k_records, level_records = _records(kshell), _records(levels)
    assert set(k_records) == set(level_records)
    dropped = md.drop_reasons(out)
    kept = {row.nuclide for row in out.inputs if row.nuclide not in dropped}
    most = {row.z: row.most_abundant for row in out.inputs}
    natural = {(z, 0) for z, a in most.items() if (z, a) in kept}
    assert set(k_records) == kept | natural
    for key, carries in _carried(kshell_doc).items():
        if key[1] == 0:
            assert carries == most[key[0]]
            assert k_records[key] == k_records[(key[0], carries)], key
            assert level_records[key] == level_records[(key[0], carries)], key
    print(f"\nkeys {len(k_records)}: kept members {len(kept)}, natural rows {len(natural)}")


def test_t105_layer1_directives_and_layer2_invariants_hold_on_every_row():
    out = _outputs()
    cells, _ = md.load_validation(CELLS, ORIGIN)
    inputs = {row.nuclide: row for row in out.inputs}
    for layer1, layer2, name in ((KSHELL_LAYER1, KSHELL_LAYER2, md.K_TABLE),
                                 (LEVELS_LAYER1, LEVELS_LAYER2, md.LEVEL_TABLE)):
        table, document = _shipped(layer1, layer2)
        spec.validate(table)
        provenance.check_against_table(table, document)
        provenance.check_source_digest(table, layer2.read_bytes())
        directives = table.directives
        assert (directives["PROFILE"], directives["SEAM"]) == (md.PROFILE, md.SEAM)
        assert md.PROFILE == "mudirac130p1"
        assert directives["TABLE"] == name
        assert "FALLBACK" not in directives and "SOURCESHA" not in directives
        assert spec.validity_assignments(table)["A"] == spec.A_MOST_ABUNDANT_AND_LISTED
        columns = directives["COLUMNS"].split()
        assert set(directives["UNITS"].split()) == {f"{column}=keV" for column in columns[2:]}
        if name == md.LEVEL_TABLE:
            shells = range(2, md.N_MAX + 1)
            assert columns == ["Z", "A", *(f"e{n}" for n in shells), *(f"u{n}" for n in shells)]
        assert document.precedence == (md.PROFILE,)
        carried = _carried(document)
        for key, row in document.rows.items():
            z, a = (int(part) for part in key.split("-"))
            source = inputs[(z, carried[(z, a)])]
            rendered = md.render_recipe_input(source, "anchor", "recipe", True, md.RECIPE_SETTINGS,
                                              md.extra_lines(cells, source.nuclide))
            assert row.source_bibkey == md.BIBKEY
            assert row.unc_type == "model"
            expected = (f"MuDirac {md.MUDIRAC_VERSION} with {md.PATCH_RELPATH} applied, input: "
                        + "; ".join(rendered.splitlines()))
            assert row.conditions == expected
            assert row.conditions.count("optimise_fermi_parameters: FALSE") == 1
            assert (row.single_source, row.needs_verification, row.recommendation) == (False, False, "")
            assert row.evaluation_id == row.source_library == md.PROFILE
            assert row.isotope_resolved is (a != 0)
            anchor, rsig = md.recipe_run_id(source, "anchor", True), md.recipe_run_id(source, "rsig", True)
            assert f"runs {anchor} and {rsig}:" in row.source_locator
            assert rsig in row.evaluation_method and anchor in row.evaluation_method
            floor = md.unc_floor_kev(out, (anchor, rsig))
            assert f"never less than {format(floor, 'f')} keV" in row.evaluation_method
            assert (" A = 0 row carries the values of " in row.evaluation_method) is (a == 0), key
            if a == 0:
                assert f"A={carried[(z, a)]}," in row.evaluation_method, key
            else:
                assert row.validity_range == f"Z={z} A={a}"


def test_t105_the_widened_parity_rule_holds_over_the_assembled_dataset(tmp_path):
    """The Python mirror of V-14 over one directory holding every D1 and D3 member: every table
    another profile carries in a seam parity carries a table in is carried by parity too; the D3
    tables sit in a seam parity carries no table in, so the unscoped rule would refuse them."""
    for source in (REPO / "data" / "g4" / "d1", D3DIR):
        for path in [*source.glob("*.g4dat"), *source.glob("*.prov.json")]:
            (tmp_path / path.name).write_bytes(path.read_bytes())
    tables = spec.load_directory(tmp_path)
    parity_seams = {
        t.directives["SEAM"] for (profile, _), t in tables.items() if profile == spec.PARITY_PROFILE
    }
    for (profile, name), table in tables.items():
        if table.directives["SEAM"] in parity_seams:
            assert (spec.PARITY_PROFILE, name) in tables, (profile, name)
    unscoped = sorted(
        (profile, name) for profile, name in tables if (spec.PARITY_PROFILE, name) not in tables
    )
    assert unscoped == [(md.PROFILE, md.K_TABLE), (md.PROFILE, md.LEVEL_TABLE)]
    assert md.SEAM not in parity_seams


def test_t105_no_d3_file_carries_a_carriage_return():
    files = sorted(path for path in D3DIR.rglob("*") if path.is_file())
    assert files
    for path in files:
        assert b"\r" not in path.read_bytes(), path.name


# --------------------------------------------------------------------------------------------
# T-106 -- the comparison with the measured energies, re-derived
# --------------------------------------------------------------------------------------------

VALIDATION = REPO / md.VALIDATION_RELPATH


def _committed_validation() -> list[dict[str, str]]:
    text = VALIDATION.read_bytes().decode("ascii")
    assert "\r" not in text
    return list(csv.DictReader(io.StringIO(text)))


def test_t106_every_gated_quantity_is_a_line_its_anchor_rsig_and_r101_runs_printed():
    out = _outputs()
    cells, _ = md.load_validation(CELLS, ORIGIN)
    inputs = {row.nuclide: row for row in out.inputs}
    for cell in cells:
        if cell.gated:
            for kind in ("anchor", "rsig", "r101"):
                run = md.recipe_run_id(inputs[cell.nuclide], kind, True)
                assert cell.quantity in out.lines[run], (cell, kind)


def test_t106_the_comparison_file_equals_an_arithmetic_rederivation():
    """Every column recomputed here from the transcription and the printed lines of the runs with the
    Uehling potential, with the tolerance three printed standard deviations and no model term; the two
    Geant4 columns from the committed cascade levels, the photon between the line's two shells taken in
    doubles."""
    out = _outputs()
    geant4 = md.load_geant4_levels(GEANT4_LEVELS)
    cells, origins = md.load_validation(CELLS, ORIGIN)
    inputs = {row.nuclide: row for row in out.inputs}
    committed = _committed_validation()
    assert len(committed) == len(cells)
    assert list(committed[0]) == list(md.VALIDATION_COLUMNS)
    assert md.TOL_FACTOR == 3 and md.SIGMA_CALC == 0

    def printed(cell, kind):
        return Decimal(out.lines[md.recipe_run_id(inputs[cell.nuclide], kind, True)][cell.quantity]) / 1000

    for cell, row in zip(cells, committed, strict=True):
        sigma = Decimal(cell.unc_kev)
        tol = 3 * sigma
        assert Decimal(row["tol_keV"]) == tol, row
        assert (row["source"], int(row["Z"]), int(row["A"]), row["transition"]) == (
            cell.source, cell.z, cell.a, cell.transition)
        assert (row["measured_keV"], row["unc_keV"], row["npol_keV"]) == (
            cell.value_kev, cell.unc_kev, cell.npol_kev)
        assert row["radius_origin"] == (origins[cell.nuclide].origin if cell.nuclide in origins else "")
        derived = ("model_keV", "residual_keV", "dE_sigma_keV", "dE_1pct_keV", "label", "within",
                   "geant4_keV", "geant4_residual_keV")
        if not cell.gated:
            assert row["gated"] == "false" and row["reason"] == cell.reason
            assert all(row[column] == "" for column in derived), row
            continue
        model = printed(cell, "anchor")
        residual = model - Decimal(cell.value_kev)
        d_sigma, d_floor = printed(cell, "rsig") - model, printed(cell, "r101") - model
        assert Decimal(row["model_keV"]) == model
        assert Decimal(row["residual_keV"]) == residual
        assert Decimal(row["dE_sigma_keV"]) == d_sigma
        assert Decimal(row["dE_1pct_keV"]) == d_floor
        sized = max(abs(d_sigma), abs(d_floor)) * 3 >= tol
        assert row["label"] == ("size-dominated" if sized else "weakly sensitive"), row
        assert row["within"] == ("true" if abs(residual) <= tol else "false"), row
        shells = ["KLMNOPQRSTUVWXYZ".index(orbit[0]) + 1 for orbit in cell.quantity.split("-")]
        photon = geant4[cell.nuclide][min(shells) - 1] - geant4[cell.nuclide][max(shells) - 1]
        with localcontext() as context:
            context.prec = 1000
            photon_kev = (Decimal(photon) * 1000).quantize(Decimal("1e-9"))
        assert Decimal(row["geant4_keV"]) == photon_kev, row
        assert Decimal(row["geant4_residual_keV"]) == photon_kev - Decimal(cell.value_kev), row
    assert VALIDATION.read_bytes() == md.build_validation(REPO)


def test_t106_the_geant4_columns_move_neither_the_label_nor_the_tolerance_test():
    """Context only: with every Geant4 level doubled, the Geant4 columns change and nothing else."""
    out = _outputs()
    cells, origins = md.load_validation(CELLS, ORIGIN)
    geant4 = md.load_geant4_levels(GEANT4_LEVELS)
    doubled = {key: tuple(2 * level for level in levels) for key, levels in geant4.items()}
    rows = md.validation_rows(out, cells, origins, geant4)
    moved = md.validation_rows(out, cells, origins, doubled)
    context = ("geant4_keV", "geant4_residual_keV")
    gated = [i for i, row in enumerate(rows) if row["gated"] == "true"]
    assert gated and all(rows[i]["geant4_keV"] != moved[i]["geant4_keV"] for i in gated)
    for row, other in zip(rows, moved, strict=True):
        assert {k: v for k, v in row.items() if k not in context} == {
            k: v for k, v in other.items() if k not in context}


def test_t106_every_gated_measurement_outside_tolerance_carries_its_printed_npol_when_fricke_prints_one():
    committed = _committed_validation()
    gated = [row for row in committed if row["gated"] == "true"]
    outside = [row for row in gated if row["within"] == "false"]
    for row in outside:
        if row["source"] == "Fricke1995":
            assert row["npol_keV"], row
    weakly = [row for row in gated if row["label"] == "weakly sensitive"]
    print(f"\ngated {len(gated)} within {len(gated) - len(outside)} outside {len(outside)}; "
          f"weakly sensitive {len(weakly)} (within {sum(r['within'] == 'true' for r in weakly)})")


# --------------------------------------------------------------------------------------------
# T-107 -- the document: its comparison table is the generated block, and every number it states
# is pinned to the value the module or the committed files give
# --------------------------------------------------------------------------------------------

DOCUMENT = REPO / "DATASET_D3.md"


def _document_text() -> str:
    """The document as text, its line ends normalized (a Windows checkout writes CRLF)."""
    return DOCUMENT.read_bytes().decode("utf-8").replace("\r\n", "\n")


#: A numeric token as the prose check tokenizes one.
_NUMERIC = re.compile(r"[0-9][0-9,.]*[0-9]|[0-9]")


def _row_pattern(line: str) -> tuple[str, tuple[int, ...]]:
    """A pattern matching ``line`` whitespace-collapsed, with every numeric token a group."""
    collapsed = " ".join(line.split())
    parts, groups, at = [], [], 0
    for index, match in enumerate(_NUMERIC.finditer(collapsed), start=1):
        parts.append(re.escape(collapsed[at:match.start()]))
        parts.append(f"({re.escape(match.group())})")
        groups.append(index)
        at = match.end()
    parts.append(re.escape(collapsed[at:]))
    return "".join(parts), tuple(groups)


def _npol_exceeding(npol: list[dict[str, str]]) -> int:
    """The gated rows whose printed nuclear-polarization correction exceeds the tolerance in magnitude."""
    return sum(abs(Decimal(row["npol_keV"])) > Decimal(row["tol_keV"]) for row in npol)


def test_d3_an_npol_equal_to_the_tolerance_does_not_exceed_it() -> None:
    at = [{"npol_keV": "0.5", "tol_keV": "0.5"}, {"npol_keV": "-0.5", "tol_keV": "0.5"}]
    past = [{"npol_keV": "0.51", "tol_keV": "0.5"}, {"npol_keV": "-0.51", "tol_keV": "0.5"}]
    assert _npol_exceeding(at) == 0
    assert _npol_exceeding(past) == 2


#: The words the document spells a small count in.
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def _keyword(name: str) -> str:
    """The value the production inputs give ``name``, as the renderer spells it."""
    (value,) = [v for k, v in md.RECIPE_SETTINGS.keywords() if k == name]
    return value


@functools.lru_cache(maxsize=1)
def document_pins() -> list[tuple[str, str, str, tuple[int, ...], object]]:
    """``(what, path, pattern, groups, expected)`` for every number `DATASET_D3.md` states: one row
    per comparison-table row (every numeric token a group, its value the generated block's), and
    one per count or constant of the prose, `expected` read from the module or the committed files.
    Patterns match the whitespace-collapsed document."""
    out = _outputs()
    rows, dropped = md.table_rows(out)
    kept = sum(1 for row in rows if row.a != 0)
    inputs = {row.nuclide: row for row in out.inputs}
    zs = sorted({row.z for row in out.inputs})
    cells, _ = md.load_validation(CELLS, ORIGIN)
    batches = {batch: len(md.recipe_runs(out.inputs, cells, batch)) for batch in md.RECIPE_BATCHES}
    validation = _committed_validation()
    gated = [row for row in validation if row["gated"] == "true"]
    within = [row for row in gated if row["within"] == "true"]
    weak = [row for row in gated if row["label"] == md.WEAKLY_SENSITIVE]
    npol = [row for row in gated if row["npol_keV"]]
    names = [f"{inputs[key].symbol}{key[1]}" for key in sorted(dropped)]
    assert len(names) == 3, names
    refinements = dict(md.REFINEMENTS)
    path = "DATASET_D3.md"
    pins: list[tuple[str, str, str, tuple[int, ...], object]] = [
        ("the generator's version", path, r"derived from the output of MuDirac (\d+\.\d+\.\d+)", (1,),
         md.MUDIRAC_VERSION),
        ("the generator's version, in the profile's name", path,
         r"The profile's name says so: MuDirac (\d+\.\d+\.\d+) with", (1,), md.MUDIRAC_VERSION),
        ("the generator's version, of the replaced profile", path,
         r"runs of the unpatched MuDirac (\d+\.\d+\.\d+);", (1,), md.MUDIRAC_VERSION),
        ("the batches of the run table", path, r"for every run of (\w+) batches:", (1,),
         NUMBER_WORDS[len(md.RECIPE_BATCHES)]),
        ("the highest shell of the chain", path, r"The chain ends at shell (\d+);", (1,), md.N_MAX),
        ("members of the input set", path, r"Of the (\d+) members of the input set", (1,), len(out.inputs)),
        ("kept members", path, r"members of the input set, (\d+) are kept", (1,), kept),
        ("natural-composition rows", path, r"the tables carry (\d+) natural-composition rows", (1,),
         len(rows) - kept),
        ("the lowest Z of the input set", path, r"with Z from (\d+) through \d+", (1,), zs[0]),
        ("the highest Z of the input set", path, r"with Z from \d+ through (\d+)", (1,), zs[-1]),
        ("the skin thickness of every run", path, r"`fermi_t: (\d+\.\d+)`", (1,), md.FERMI_T),
        ("the grid step of the production runs", path, r"`loggrid_step: ([0-9.]+)`", (1,),
         _keyword("loggrid_step")),
        ("the energy tolerance of the production runs", path, r"`energy_tol: ([0-9.e-]+)`", (1,),
         _keyword("energy_tol")),
        ("the Uehling steps of the production runs", path, r"`uehling_steps: (\d+)`", (1,),
         _keyword("uehling_steps")),
        ("the Uehling shell nodes of the production runs", path, r"`uehling_shell_nodes: (\d+)`", (1,),
         _keyword("uehling_shell_nodes")),
        ("the decimals of a printed line", path, r"prints its lines to (\d+) decimals", (1,),
         md.RECIPE_XR_PRECISION),
        ("the significant digits of a state header", path, r"each state to (\d+) significant", (1,),
         md.RECIPE_HEADER_PRECISION),
        ("runs of the prod batch", path, r"The run table lists (\d+), \d+, \d+ and \d+ runs", (1,),
         batches["prod"]),
        ("runs of the refine batch", path, r"The run table lists \d+, (\d+), \d+ and \d+ runs", (1,),
         batches["refine"]),
        ("runs of the rminus batch", path, r"The run table lists \d+, \d+, (\d+) and \d+ runs", (1,),
         batches["rminus"]),
        ("runs of the stages batch", path, r"The run table lists \d+, \d+, \d+ and (\d+) runs", (1,),
         batches["stages"]),
        ("the shell of the outermost derived orbit", path, r"circular orbit of shell (\d+), `R15`", (1,),
         md.N_MAX),
        ("the 2pF rms floor", path, r"rms radius of its anchor and `rsig` runs exceeds ([0-9.]+) fm", (1,),
         repr(md.fermi2_rms_floor_fm(md.FERMI_T))),
        ("the thickness of the 2pF floor", path, r"the 2pF density of thickness ([0-9.]+) fm", (1,),
         md.FERMI_T),
        ("the lowest checked shell", path, r"circular state of shells (\d+) through \d+", (1,),
         md.IDEAL_FROM),
        ("the highest checked shell", path, r"circular state of shells \d+ through (\d+)", (1,), md.N_MAX),
        ("the hydrogen-like bound in percent", path, r"lies within (\d+) % of the same state", (1,),
         int(md.NMAX_BOUND * 100)),
        ("the first dropped member", path, r"The generator drops `([A-Z][a-z]?\d+)`, `", (1,), names[0]),
        ("the second dropped member", path, r"drops `[A-Z][a-z]?\d+`, `([A-Z][a-z]?\d+)` and", (1,),
         names[1]),
        ("the third dropped member", path, r"and `([A-Z][a-z]?\d+)`, whose intended rms radii", (1,),
         names[2]),
        ("the tolerance factor", path, r"The tolerance is (\d+) times", (1,), md.TOL_FACTOR),
        ("the model's own uncertainty", path, r"is set to (\d+) by decision", (1,), md.SIGMA_CALC),
        ("the size floor", path, r"multiplied by (\d+\.\d+) \(the `r101` runs", (1,), str(md.SIZE_FLOOR)),
        ("gated rows", path, r"Of the (\d+) gated rows, \d+ lie within tolerance", (1,), len(gated)),
        ("gated rows within tolerance", path, r"gated rows, (\d+) lie within tolerance", (1,), len(within)),
        ("gated rows outside tolerance", path, r"lie within tolerance and (\d+) outside it", (1,),
         len(gated) - len(within)),
        ("weakly sensitive gated rows", path,
         r"outside it; (\d+) of the gated rows are weakly sensitive", (1,), len(weak)),
        ("weakly sensitive rows within tolerance", path, r"and (\d+) of those lie within tolerance", (1,),
         sum(row["within"] == "true" for row in weak)),
        ("gated rows whose NPol exceeds the tolerance", path,
         r"On (\d+) of the \d+ gated rows that carry an `NPol`", (1,),
         _npol_exceeding(npol)),
        ("gated rows that carry an NPol", path, r"On \d+ of the (\d+) gated rows that carry an `NPol`", (1,),
         len(npol)),
        ("the Uehling steps one step finer, from", path, r"Uehling steps raised from (\d+) to \d+", (1,),
         md.RECIPE_SETTINGS.n_u),
        ("the Uehling steps one step finer, to", path, r"raised from \d+ to (\d+) \(`nu90`\)", (1,),
         refinements["nu90"].n_u),
        ("the shell nodes one step finer, from", path, r"the shell nodes from (\d+) to \d+", (1,),
         md.RECIPE_SETTINGS.n_s),
        ("the shell nodes one step finer, to", path, r"the shell nodes from \d+ to (\d+) \(`ns16`\)", (1,),
         refinements["ns16"].n_s),
        ("the absolute engineering target", path, r"the larger of ([0-9.]+) keV and", (1,),
         format(d3c.BINDING_TARGET_ABSOLUTE_KEV, "f")),
        ("the relative engineering target", path, r"keV and ([0-9.]+) times the value", (1,),
         format(d3c.BINDING_TARGET_RELATIVE, "f")),
    ]
    table = md.render_validation_table(validation).splitlines()[2:]
    for number, line in enumerate(table, start=1):
        pattern, groups = _row_pattern(line)
        pins.append((f"comparison table row {number}", path, pattern, groups, None))
    pins.extend(numerical_pins())
    pins.extend(contract_pins())
    pins.extend(centroid_pins())
    pins.extend(attribution_pins())
    return pins


def _committed_rows(relpath: str) -> list[dict[str, str]]:
    text = (REPO / relpath).read_bytes().decode("ascii")
    assert "\r" not in text
    return list(csv.DictReader(io.StringIO(text)))


def numerical_pins() -> list[tuple[str, str, str, tuple[int, ...], object]]:
    """The pins of the section on the numerical checks, read from `components.jsonl`, `shell_projection.csv`
    and `numerical_check.csv`; each count it states of a verdict is held to every record carrying it."""
    path = "DATASET_D3.md"
    records = [r for r in _component_records(COMPONENTS) if "quantity" in r]
    projection = _committed_rows(d3c.PROJECTION_RELPATH)
    assert all(r["refinement_verdict"] == r["noU_refinement_verdict"] == d3c.ONE_STEP_STABLE for r in records)
    assert all(r["solver_numeric_qualification"] == d3c.ONE_STEP_STABLE for r in projection)
    worst = (Decimal(-1), "", "", 0)
    with localcontext() as context:
        context.prec = 50
        for r in records:
            target = Decimal(r["refinement_target_keV"])
            for tag, change in r["refinement_changes_keV"].items():
                ratio = abs(Decimal(change)) / (d3c.REFINEMENT_LIMITS[tag] * target)
                if ratio > worst[0]:
                    worst = (ratio, tag, r["quantity"], (r["Z"], r["A"]))
    ratio, tag, quantity, key = worst
    assert (tag, quantity) == ("k4", "K"), worst
    symbols = {row.nuclide: row.symbol for row in _outputs().inputs}
    check = _committed_rows(d3c.NUMERICAL_CHECK_RELPATH)
    sets = {name: {(r["Z"], r["A"]) for r in check if r["set"] == name} for name in d3c.NUMERICAL_CHECK_SETS}
    pilots = sets["pilot_D"] | sets["pilot_RA"] | sets["pilot_RB"]
    assert all(r["verdict"] == "QUALIFIED" for r in check)
    with localcontext() as context:
        context.prec = 50
        largest_ratio = max(Decimal(r["b_num_eV"]) / Decimal(r["target_eV"]) for r in check)
        ceiling = (largest_ratio * 1000).to_integral_value(rounding="ROUND_CEILING") / 1000
        far = max(check, key=lambda r: abs(Decimal(r["Q_eV"]) - Decimal(r["Q_ref_eV"])))
        difference = abs(Decimal(far["Q_eV"]) - Decimal(far["Q_ref_eV"]))
    assert (far["uehling"], far["quantity"]) == ("on", "level:K1"), far
    compared = md.compared_nuclides(md.load_cells(CELLS))
    return [
        ("components with the Uehling potential, one step stable", path,
         r"All (\d+) emitted binding energies with the Uehling potential", (1,), len(records)),
        ("components without it, one step stable", path, r"their (\d+) twins without it", (1,), len(records)),
        ("gated solver lines, one step stable", path, r"and all (\d+) gated solver lines are", (1,),
         len(projection)),
        ("the largest one-step change against its limit", path,
         r"the largest change is ([0-9.]+) of its limit", (1,),
         format(ratio.quantize(Decimal("0.001")), "f")),
        ("the element of the largest one-step change", path, r"on `K` of ([A-Z][a-z]?)-\d+\.", (1,),
         symbols[key]),
        ("the mass number of the largest one-step change", path, r"on `K` of [A-Z][a-z]?-(\d+)\.", (1,),
         key[1]),
        ("rows of the independent check", path, r"The file carries (\d+) rows over \d+ nuclides", (1,),
         len(check)),
        ("nuclides of the independent check", path, r"The file carries \d+ rows over (\d+) nuclides", (1,),
         len(pilots | sets["audit"])),
        ("the sets of the independent check", path, r"over \d+ nuclides in (\w+) sets:", (1,),
         NUMBER_WORDS[len(d3c.NUMERICAL_CHECK_SETS)]),
        ("development nuclides", path, r"sets: (\d+) of the development set \(`pilot_D`\)", (1,),
         len(sets["pilot_D"])),
        ("reserved nuclides", path, r"made; (\d+) reserved from those choices \(`pilot_RA`\)", (1,),
         len(sets["pilot_RA"])),
        ("contingency nuclides", path, r"Hamiltonian; (\d+) of a contingency set \(`pilot_RB`\)", (1,),
         len(sets["pilot_RB"])),
        ("audit nuclides", path, r"and a sample of (\d+) \(`audit`\)", (1,), len(sets["audit"])),
        ("the highest Z of the compared nuclides", path, r"one member for each Z above (\d+), the highest Z",
         (1,), max(z for z, _a in compared)),
        ("the largest b_num against its target, rounded up", path,
         r"with `b_num_eV` at most ([0-9.]+) of the", (1,), format(ceiling, "f")),
        ("the largest difference from the independent solution", path,
         r"production value and the independent one is ([0-9.]+) eV", (1,), format(difference, "f")),
        ("its element", path, r"on the K level of ([A-Z][a-z]?)-\d+ with", (1,),
         symbols[(int(far["Z"]), int(far["A"]))]),
        ("its mass number", path, r"on the K level of [A-Z][a-z]?-(\d+) with", (1,), far["A"]),
        ("its target", path, r"against a target of ([0-9.]+) eV\.", (1,), far["target_eV"]),
    ]


def contract_pins() -> list[tuple[str, str, str, tuple[int, ...], object]]:
    """The pins of the section on the energy the cascade receives: every number it states, read
    from `shell_projection.csv`, `incompatible_groups.csv` and `radius_lineage.csv`,
    and one pin per row of the generated groups table."""
    path = "DATASET_D3.md"
    projection = _committed_rows(d3c.PROJECTION_RELPATH)
    groups = _committed_rows(d3c.GROUPS_RELPATH)
    lineage = d3c.load_radius_lineage(REPO / d3c.LINEAGE_RELPATH)
    assert all(row.dependency_state == d3c.UNKNOWN for row in lineage)
    widest = max(groups, key=lambda g: Decimal(g["gap_keV"]))
    pins: list[tuple[str, str, str, tuple[int, ...], object]] = [
        ("gated rows in the projection", path, r"Of the (\d+) gated rows, the shell difference lies", (1,),
         len(projection)),
        ("rows whose shell difference lies within the band", path,
         r"the shell difference lies within the band for (\d+),", (1,),
         sum(r["consumer_within"] == "true" for r in projection)),
        ("rows whose solver line lies within the band", path,
         r"lies within the band for \d+, the solver line for (\d+),", (1,),
         sum(r["solver_within"] == "true" for r in projection)),
        ("rows whose unpatched cascade lies within the band", path,
         r"the solver line for \d+, the unpatched cascade for (\d+),", (1,),
         sum(r["stock_within"] == "true" for r in projection)),
        ("rows whose shell difference is closer than the unpatched cascade", path,
         r"closer than the unpatched cascade for (\d+)\.", (1,),
         sum(r["consumer_closer_than_stock"] == "true" for r in projection)),
        ("groups with an empty intersection", path,
         r"intersects the lines' bands: (\d+) of the \d+ such groups", (1,),
         sum(g["empty"] == "true" for g in groups)),
        ("multi-line groups", path, r"bands: \d+ of the (\d+) such groups", (1,), len(groups)),
        ("the widest group's mass number", path, r"the widest, Pb-(\d+) K–L, by", (1,), widest["A"]),
        ("the widest gap", path, r"Pb-\d+ K–L, by ([0-9.]+) keV", (1,), widest["gap_keV"]),
    ]
    for number, line in enumerate(d3c.render_groups_table(groups).splitlines()[2:], start=1):
        pattern, row_groups = _row_pattern(line)
        pins.append((f"groups table row {number}", path, pattern, row_groups, None))
    return pins


def centroid_pins() -> list[tuple[str, str, str, tuple[int, ...], object]]:
    rows = _committed_rows(centroids.CENTROIDS_RELPATH)
    margins = _committed_rows(centroids.MARGINS_RELPATH)
    ratios = _committed_rows(centroids.RATIOS_RELPATH)
    compared = [row for row in rows if row["cohort"] != "excluded"]
    above = sum(Decimal(row["b_num_keV"]) > Decimal(row["sigma_keV"] or row["sigma_max_keV"]) / 10
                for row in compared)
    assert len(ratios) == 1
    path = "DATASET_D3.md"
    pins: list[tuple[str, str, str, tuple[int, ...], object]] = [
        ("centroid rows whose b_num exceeds a tenth of the uncertainty", path,
         r"correlation, on (\d+) of the \d+ rows\.", (1,), above),
        ("centroid compared rows", path, r"correlation, on \d+ of the (\d+) rows\.", (1,), len(compared)),
        ("illustrative ratio mass", path, r"For Pb-(\d+) the row also lists", (1,), ratios[0]["A"]),
    ]
    pins.extend(margin_summary_pins(compared, margins))
    for name, block in (("centroids", centroids.render_centroids_table(rows)),
                        ("margins", centroids.render_margins_table(margins))):
        for number, line in enumerate(block.splitlines()[2:], start=1):
            pattern, groups = _row_pattern(line)
            pins.append((f"{name} table row {number}", path, pattern, groups, None))
    return pins


#: The words the document may give a multiple in, read as the multiple each names.
MULTIPLES = {"once": 1, "twice": 2}


def margin_summary_pins(compared: list[dict[str, str]],
                        margins: list[dict[str, str]]) -> list[tuple[str, str, str, tuple[int, ...], object]]:
    """The pins of the two sentences after the margins table: the range of the margins over the isotopes
    `centroid_margins.csv` counts, the largest absolute residuals of the patched and the unpatched cascade
    with their nuclides, and the rows whose margin exceeds twice the tolerance, taken at its largest over
    every correlation, plus `b_num_keV` -- the most that moving the value within its band and the shell
    difference by its numerical bound can take from a margin -- and the word for the multiple of the
    tolerance that screen allows the value's band, read from the screen by probing it with no bound a step
    either side of each multiple `MULTIPLES` names; whether a margin at the multiple itself outlasts is left
    to the drill."""
    path = "DATASET_D3.md"
    symbol = {(row["Z"], row["A"]): row["symbol"] for row in _committed_rows(md.INPUTS_RELPATH)}
    patched = max(margins, key=lambda row: Decimal(row["max_abs_r_d3_keV"]))
    unpatched = max(compared, key=lambda row: abs(Decimal(row["r_stock_keV"])))

    def outlasts(row: dict[str, str]) -> bool:
        tolerance = md.TOL_FACTOR * Decimal(row["sigma_keV"] or row["sigma_max_keV"])
        return Decimal(row["margin_keV"]) > 2 * tolerance + Decimal(row["b_num_keV"])

    lasting = [row for row in compared if outlasts(row)]
    (short,) = [row for row in compared if not outlasts(row)]
    nuclide = symbol[(patched["max_abs_r_d3_Z"], patched["max_abs_r_d3_A"])]

    def probe(margin: Decimal) -> bool:
        return outlasts(dict(compared[0], b_num_keV="0", margin_keV=str(margin)))

    unit = md.TOL_FACTOR * Decimal(compared[0]["sigma_keV"] or compared[0]["sigma_max_keV"])
    (multiple,) = [word for word, times in MULTIPLES.items()
                   if not probe((times * unit).next_minus()) and probe((times * unit).next_plus())]
    return [
        ("isotopes the margins are over", path,
         r"the margins over the (\d+) isotopes of `centroid_margins\.csv`", (1,),
         sum(int(row["isotopes"]) for row in margins)),
        ("smallest margin", path, r"`centroid_margins\.csv` run from ([0-9.]+) keV to", (1,),
         min(margins, key=lambda row: Decimal(row["margin_min_keV"]))["margin_min_keV"]),
        ("largest margin", path, r"run from [0-9.]+ keV to ([0-9.]+) keV, and the patched", (1,),
         max(margins, key=lambda row: Decimal(row["margin_max_keV"]))["margin_max_keV"]),
        ("patched cascade's largest absolute residual", path,
         r"largest absolute residual is ([0-9.]+) keV, for", (1,), patched["max_abs_r_d3_keV"]),
        ("patched cascade's largest residual, mass number", path,
         rf"residual is [0-9.]+ keV, for {nuclide}-(\d+), where", (1,), patched["max_abs_r_d3_A"]),
        ("unpatched cascade's largest absolute residual", path,
         r"the unpatched cascade's reaches ([0-9.]+) keV", (1,), unpatched["r_stock_keV"].lstrip("-")),
        ("unpatched cascade's largest residual, mass number", path,
         rf"reaches [0-9.]+ keV, for {symbol[(unpatched['Z'], unpatched['A'])]}-(\d+)\.", (1,),
         unpatched["A"]),
        ("rows whose margin outlasts the band and the bound", path,
         r"the margin exceeds the sum of both on (\d+) of the \d+ rows", (1,), len(lasting)),
        ("rows of the margin screen", path,
         r"the margin exceeds the sum of both on \d+ of the (\d+) rows", (1,), len(compared)),
        ("the row whose margin does not", path,
         rf"and not on the row for {symbol[(short['Z'], short['A'])]}-(\d+)\.", (1,), short["A"]),
        ("the band's multiple of the tolerance", path,
         r"changes a margin by at most (\w+) the tolerance", (1,), multiple),
    ]


def test_t107_drill_a_margin_that_outlasts_nothing_moves_the_count() -> None:
    compared = [row for row in _committed_rows(centroids.CENTROIDS_RELPATH) if row["cohort"] != "excluded"]
    margins = _committed_rows(centroids.MARGINS_RELPATH)
    stated = {what: expected for what, *_, expected in margin_summary_pins(compared, margins)}
    first = next(i for i, row in enumerate(compared) if row["cohort"] == "labelled")
    shrunk = [dict(row, margin_keV="0.000000001") if i == first else row for i, row in enumerate(compared)]
    with pytest.raises(ValueError):
        margin_summary_pins(shrunk, margins)
    count = stated["rows whose margin outlasts the band and the bound"]
    assert isinstance(count, int) and count == len(compared) - 1
    # The edges of the screen, on that row with its committed sigma and bound: a margin one step above the
    # band's multiple of the tolerance plus the bound outlasts both; a margin equal to that sum, and one a
    # step below it (still above the band alone), does not, so a second row falls short and the single-row
    # unpacking refuses. Only the two cells the screen reads move.
    row = compared[first]
    tolerance = md.TOL_FACTOR * Decimal(row["sigma_keV"] or row["sigma_max_keV"])
    bound_num = Decimal(row["b_num_keV"])
    band = MULTIPLES[stated["the band's multiple of the tolerance"]] * tolerance
    threshold = band + bound_num
    assert band < threshold.next_minus() < threshold < threshold.next_plus()

    def screened(margin: Decimal) -> dict[str, object]:
        moved = [dict(other, margin_keV=str(margin)) if i == first else other
                 for i, other in enumerate(compared)]
        return {what: expected for what, *_, expected in margin_summary_pins(moved, margins)}

    above = screened(threshold.next_plus())
    assert above["rows whose margin outlasts the band and the bound"] == count
    for margin in (threshold, threshold.next_minus()):
        with pytest.raises(ValueError, match="too many values to unpack"):
            screened(margin)


def test_t107_the_centroid_tables_are_generated_blocks():
    rows = _committed_rows(centroids.CENTROIDS_RELPATH)
    margins = _committed_rows(centroids.MARGINS_RELPATH)
    text = _document_text()
    for block in (centroids.render_centroids_table(rows), centroids.render_margins_table(margins)):
        assert text.count(block) == 1
        data_row = block.splitlines()[2]
        mutated = _replace_once(text, data_row, data_row.replace(" | ", " | 1", 1))
        assert mutated.count(block) == 0
        assert _pin_problems(mutated)


def test_t107_the_groups_table_is_the_generated_block():
    block = d3c.render_groups_table(_committed_rows(d3c.GROUPS_RELPATH))
    assert _document_text().count(block) == 1


def test_t107_the_comparison_table_is_the_generated_block():
    block = md.render_validation_table(_committed_validation())
    text = _document_text()
    assert text.count(block) == 1


def _pin_problems(text: str, path: str = "DATASET_D3.md") -> list[str]:
    """Every pin of ``path`` that ``text``, that file's content, does not state exactly once."""
    collapsed = " ".join(text.split())
    problems = []
    for what, pin_path, pattern, _groups, expected in document_pins():
        if pin_path != path:
            continue
        hits = list(re.finditer(pattern, collapsed))
        if len(hits) != 1:
            problems.append(f"{what}: matched {len(hits)} times")
        elif expected is not None and hits[0].group(1) != str(expected):
            problems.append(f"{what}: states {hits[0].group(1)}, the committed files give {expected}")
    return problems


VERSION_STATEMENT = re.compile(r"the dataset's `#VERSION` becomes (\d+\.\d+\.\d+)")


def changelog_version_problems(text: str) -> list[str]:
    """The newest stated dataset version is shipped; earlier statements strictly decrease."""
    kshell_table = KSHELL_LAYER1.read_text("ascii")
    shipped = re.search(r"^#VERSION\s+(\S+)$", kshell_table, re.M)
    assert shipped, "the committed k-shell table declares no #VERSION"
    versions = [tuple(map(int, match.group(1).split(".")))
                for match in VERSION_STATEMENT.finditer(text)]
    if not versions:
        return ["the changelog states no dataset version"]
    problems = []
    if versions[0] != tuple(map(int, shipped.group(1).split("."))):
        problems.append("the topmost changelog version differs from the shipped version")
    if any(above <= below for above, below in zip(versions, versions[1:], strict=False)):
        problems.append("changelog versions do not strictly decrease")
    return problems


def test_t144_changelog_versions_track_the_shipped_version_and_history():
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    matches = list(VERSION_STATEMENT.finditer(text))
    assert len(matches) >= 2
    assert changelog_version_problems(text) == []
    top, historical = matches[:2]
    stale_top = text[:top.start(1)] + historical.group(1) + text[top.end(1):]
    assert changelog_version_problems(stale_top)
    top_version = tuple(map(int, top.group(1).split(".")))
    newer = ".".join(map(str, (top_version[0] + 1, *top_version[1:])))
    out_of_order = text[:historical.start(1)] + newer + text[historical.end(1):]
    assert "changelog versions do not strictly decrease" in changelog_version_problems(out_of_order)
    restated = text[:historical.start(1)] + top.group(1) + text[historical.end(1):]
    assert changelog_version_problems(restated) == ["changelog versions do not strictly decrease"]
    removed_top = text[:top.start()] + text[top.end():]
    assert "the topmost changelog version differs from the shipped version" in (
        changelog_version_problems(removed_top)
    )
    no_statements = VERSION_STATEMENT.sub("", text)
    assert changelog_version_problems(no_statements) == ["the changelog states no dataset version"]


def test_t107_every_pin_matches_once_and_states_its_value():
    assert not _pin_problems(_document_text())
    assert {pin[1] for pin in document_pins()} == {"DATASET_D3.md"}
    print(f"\ndocument pins: {len(document_pins())}")


def test_t107_drill_a_changed_count_and_a_changed_table_cell_are_refused():
    text = _document_text()
    validation = _committed_validation()
    gated = sum(row["gated"] == "true" for row in validation)
    within = sum(row["gated"] == "true" and row["within"] == "true" for row in validation)
    stated = f"Of the {gated} gated rows, {within} lie"
    mutated = _replace_once(text, stated, f"Of the {gated + 1} gated rows, {within} lie")
    assert any(problem.startswith("gated rows: states") for problem in _pin_problems(mutated))
    block = md.render_validation_table(_committed_validation())
    row = block.splitlines()[2]
    cell = row.split(" | ")[5]
    changed = _replace_once(text, row, row.replace(f" | {cell} | ", f" | {cell}1 | ", 1))
    assert changed.count(block) == 0
    assert any(problem.startswith("comparison table row 1:") for problem in _pin_problems(changed))


# --------------------------------------------------------------------------------------------
# T-110 -- the level energies Geant4's own cascade uses, beside the comparison
# --------------------------------------------------------------------------------------------

GEANT4_LEVELS = REPO / md.GEANT4_LEVELS_RELPATH


def cascade_k_energies(zs: list[int], literals: list[str]) -> dict[int, float]:
    """``fKLevelEnergy`` as the vendored cascade's constructor fills it, re-computed here by its own
    expression over its compiled-in table: the listed Z take their literal, the Z between two listed
    ones the interpolation of energy over Z squared."""
    energies = [float(token) for token in literals]
    k = {0: 0.0, 1: energies[0]}
    idx = 1
    for i in range(1, len(zs)):
        z1, z2 = zs[idx], zs[i]
        if z1 + 1 < z2:
            dz = float(z2 - z1)
            y1 = energies[idx] / float(z1 * z1)
            y2 = energies[i] / float(z2 * z2)
            for z in range(z1 + 1, z2):
                k[z] = (y1 + (y2 - y1) * (z - z1) / dz) * z * z
        k[z2] = energies[i]
        idx = i
    return k


def test_t110_drill_equal_empty_gated_stock_sets_are_admitted():
    out = md.load_outputs(REPO)
    assert md.validation_rows(out, (), {}, {}) == []


def test_t110_the_cascade_levels_are_the_vendored_cascades_for_every_gated_nuclide():
    levels = md.load_geant4_levels(GEANT4_LEVELS)
    cells, _ = md.load_validation(CELLS, ORIGIN)
    assert sorted(levels) == md.stock_nuclides(cells)
    cascade = parity.D3_SEAM_DIRS[parity.d1.UPSTREAM_TAG] / parity.CASCADE_NAME
    text = cascade.read_text("ascii")
    assert f"fLevelEnergy[{md.CASCADE_LEVELS - 1}]" in text and f"i<{md.CASCADE_LEVELS}" in text
    kshell = cascade_k_energies(*parity.k_table(text, parity.CASCADE_NAME))
    top = max(kshell)
    for (z, a), row in levels.items():
        assert len(row) == md.CASCADE_LEVELS
        assert all(upper < lower for lower, upper in zip(row, row[1:], strict=False)), (z, a)
        assert row[0] == kshell[min(z, top)], (z, a, row[0].hex(), kshell[min(z, top)].hex())
        e = 4 * row[1]
        for n in range(2, md.CASCADE_LEVELS + 1):
            assert e / float(n * n) == row[n - 1], (z, a, n)
    print(f"\ncascade levels: {len(levels)} stock nuclides, {md.CASCADE_LEVELS} levels each")


def _harvest_from_levels() -> str:
    """A harvest whose C lines carry the committed levels of every gated nuclide."""
    rows = GEANT4_LEVELS.read_bytes().decode("ascii").split(NL)[1:-1]
    tokens: dict[tuple[int, int], list[str]] = {}
    for row in rows:
        z, a, _n, level = row.split(",")
        tokens.setdefault((int(z), int(a)), []).append(level)
    lines = ["K 1 0x1p-1"]
    lines += [f"C {z} {a} " + " ".join(levels) + " 1 e:0x1p-1 edep 0x1p-1"
              for (z, a), levels in tokens.items()]
    return NL.join(lines) + NL


def test_t110_the_committed_file_is_what_the_renderer_writes_from_a_harvest():
    cells, _ = md.load_validation(CELLS, ORIGIN)
    harvest = _harvest_from_levels()
    gated = md.gated_nuclides(cells)
    optional = sorted(set(md.centroid_nuclides(cells)) - set(gated))
    assert md.render_geant4_levels(harvest, gated, optional) == GEANT4_LEVELS.read_bytes()


def test_t110_a_missing_labelled_c_line_leaves_only_that_stock_comparison(tmp_path):
    cells, origins = md.load_validation(CELLS, ORIGIN)
    gated = md.gated_nuclides(cells)
    optional = sorted(set(md.centroid_nuclides(cells)) - set(gated))
    key = optional[0]
    harvest = _harvest_from_levels()
    missing = next(line for line in harvest.split(NL) if line.startswith(f"C {key[0]} {key[1]} "))
    payload = md.render_geant4_levels(harvest.replace(missing + NL, ""), gated, optional)
    path = tmp_path / "levels.csv"
    path.write_bytes(payload)
    levels = md.load_geant4_levels(path)
    assert key not in levels
    assert sorted(levels) == sorted(set(md.stock_nuclides(cells)) - {key})
    md.validation_rows(md.load_outputs(REPO), cells, origins, levels)


def test_t110_a_missing_gated_key_or_an_unlisted_stock_key_is_refused():
    cells, origins = md.load_validation(CELLS, ORIGIN)
    levels = md.load_geant4_levels(GEANT4_LEVELS)
    out = md.load_outputs(REPO)
    first = md.gated_nuclides(cells)[0]
    with pytest.raises(md.CellError, match="gated nuclides"):
        md.validation_rows(out, cells, origins, {k: v for k, v in levels.items() if k != first})
    stray = (max(k[0] for k in levels) + 1, 1)
    with pytest.raises(md.CellError, match="stock nuclides"):
        md.validation_rows(out, cells, origins, {**levels, stray: next(iter(levels.values()))})


def test_t110_drill_a_missing_a_repeated_and_a_malformed_c_line_are_refused():
    cells, _ = md.load_validation(CELLS, ORIGIN)
    nuclides = md.gated_nuclides(cells)
    harvest = _harvest_from_levels()
    key = nuclides[0]
    first = next(line for line in harvest.split(NL) if line.startswith(f"C {key[0]} {key[1]} "))
    drills = [
        (harvest.replace(first + NL, ""), md.CellError, "no C line for"),
        (harvest + first + NL, md.DuplicateKeyError, "a second C line"),
        (harvest.replace(first, first.replace(" 0x", " 0.", 1)), md.CellError, "printed with %a"),
    ]
    for mutated, error, message in drills:
        assert mutated != harvest
        with pytest.raises(error, match=re.escape(message)):
            md.render_geant4_levels(mutated, nuclides)


_GL = "21,45,1,"
GEANT4_LEVELS_DRILLS = [
    ("a carriage return", lambda t: t.replace(NL, "\r" + NL, 1), md.CarriageReturnError, "contains CR"),
    ("a non-ASCII byte", lambda t: t.replace(NL + _GL, NL + "21,45,1·", 1),
     md.NonAsciiError, "outside US-ASCII"),
    ("a renamed column", lambda t: _replace_once(t, "n,level_MeV", "n,level"), md.HeaderError, "header is"),
    ("a signed Z", lambda t: _replace_once(t, NL + _GL, NL + "+21,45,1,"),
     md.CellError, "Z must be an integer"),
    ("a level in decimal", lambda t: NL.join([*_lines(t)[:1], _GL + "1.1", *_lines(t)[2:]]),
     md.CellError, "positive double printed with %a"),
    ("a missing level", lambda t: NL.join([*_lines(t)[:2], *_lines(t)[3:]]),
     md.CellError, "got n = 3"),
    ("a nuclide stopping short", lambda t: NL.join([*_lines(t)[:-2], ""]), md.CellError, "stop short"),
    ("a duplicated row", lambda t: _repeat_line(t, 1), md.DuplicateKeyError, "duplicate row"),
    ("rows out of order", lambda t: _swap_lines(t, md.CASCADE_LEVELS), md.OrderError, "ordered by (Z, A, n)"),
    ("no rows", lambda t: _lines(t)[0] + NL, md.EmptyError, "carries no rows"),
]


@pytest.mark.parametrize("label, mutate, error, message", GEANT4_LEVELS_DRILLS,
                         ids=[d[0] for d in GEANT4_LEVELS_DRILLS])
def test_t110_drill_each_loader_rule_refuses_its_fixture(tmp_path, label, mutate, error, message):
    _drill(tmp_path, GEANT4_LEVELS, mutate, error, message, md.load_geant4_levels)


# --------------------------------------------------------------------------------------------
# T-121 -- which nuclides the tables keep, and the runs made beside the production ones
# --------------------------------------------------------------------------------------------


def test_t121_the_dropped_set_is_the_members_outside_the_2pf_domain_and_the_tables_hold_none_of_them():
    """Re-derived from the committed inputs: a member is dropped exactly when the intended rms radius of its
    anchor run is at or below the rms radius of the 2pF density with c = 0 at its thickness, computed here by
    quadrature of the 2pF moments; no row of either table carries a dropped member."""
    out = _outputs()
    dropped = md.drop_reasons(out)
    thickness = {row.fermi_t_fm for row in out.inputs}
    assert thickness == {md.FERMI_T}
    a = float(md.FERMI_T) / (4 * math.log(3.0))
    h = a / 200
    moment2 = moment4 = 0.0
    for i in range(1, 60 * 200):
        s = i * h
        f = 1 / (1 + math.exp(s / a))
        moment2 += f * s ** 2
        moment4 += f * s ** 4
    floor = math.sqrt(moment4 / moment2)
    assert md.fermi2_rms_floor_fm(md.FERMI_T) == pytest.approx(floor, rel=1e-6)
    expected = {row.nuclide for row in out.inputs
                if float(row.radius_fm) / md.SPHERE_FACTOR <= md.fermi2_rms_floor_fm(md.FERMI_T)}
    assert set(dropped) == expected
    for key in expected:
        assert dropped[key].startswith("anchor density_domain OUTSIDE_2PF_T2.3: intended rms "), dropped[key]
    keys = {key for layer1 in (KSHELL_LAYER1, LEVELS_LAYER1)
            for key in _records(spec.parse(layer1.read_bytes().decode("ascii")))}
    assert not {key for key in keys if key in expected}
    print(f"\nmembers {len(out.inputs)} kept {len(out.inputs) - len(dropped)} dropped {sorted(dropped)}")


def test_t121_drill_a_hydrogen_like_state_moved_past_the_bound_drops_its_member_with_the_clause(tmp_path):
    """The keep rule's shell check, on a copy: one checked circular state of a kept member's hydrogen-like
    run moved to twice its printed energy is refused by that clause, and no other member moves."""
    out = _outputs()
    before = md.drop_reasons(out)
    row = md.kept_members(out)[0]
    run = md.recipe_run_id(row, f"ideal{md.IDEAL_FROM}", False)
    state = md.orbit(md.IDEAL_FROM, True)
    root = _data_copy(tmp_path)
    states = root / md.STATES_RELPATH
    text = states.read_bytes().decode("ascii")
    printed = out.headers[run][state]
    line = f"{run},{state},{md.IDEAL_FROM},{md.IDEAL_FROM - 1},1,{printed}"
    doubled = line.replace(printed, repr(2 * float(printed)))
    states.write_bytes(_replace_once(text, NL + line + NL, NL + doubled + NL).encode("ascii"))
    dropped = md.drop_reasons(md.load_outputs(root))
    assert set(dropped) == set(before) | {row.nuclide}
    clause = f"{run} shell {md.IDEAL_FROM} {state} differs from hydrogen-like by "
    assert dropped[row.nuclide].startswith(clause)


def test_t121_the_rminus_runs_are_the_members_whose_lowered_radius_keeps_the_2pf_density():
    out = _outputs()
    rows = out.inputs
    made = {r.run for r in md.recipe_runs(rows, out.cells, "rminus")}
    for row in rows:
        lowered = float(row.radius_fm) / md.SPHERE_FACTOR - float(row.sigma_rms_fm)
        inside = lowered > md.fermi2_rms_floor_fm(row.fermi_t_fm)
        assert (md.recipe_run_id(row, "rminus", True) in made) is inside, row.nuclide
        assert (md.recipe_run_id(row, "rminus", True) in out.runs) is inside, row.nuclide
    print(f"\nrminus runs {len(made)} of {len(rows)} members")


def test_t121_each_refinement_run_differs_from_its_production_run_in_its_one_setting_only():
    out = _outputs()
    cells = out.cells
    for row in out.inputs[:40]:
        extra = md.extra_lines(cells, row.nuclide)
        for uehling in (True, False):
            base = md.render_recipe_input(row, "anchor", "recipe", uehling, md.RECIPE_SETTINGS,
                                          extra).splitlines()
            for tag, settings in md.REFINEMENTS:
                if not uehling and tag not in md.NOU_REFINEMENTS:
                    assert md.recipe_run_id(row, "anchor", uehling, tag) not in out.runs
                    continue
                finer = md.render_recipe_input(row, "anchor", "recipe", uehling, settings, extra).splitlines()
                diff = [(x, y) for x, y in zip(base, finer, strict=True) if x != y]
                assert len(diff) == 1, (row.nuclide, tag, diff)
                assert md.recipe_run_id(row, "anchor", uehling, tag) in out.runs


# --------------------------------------------------------------------------------------------
# T-123 -- the projection of every gated line onto the shell difference the cascade receives,
# re-derived here from the tables' text and the committed files, with the counts pinned
# --------------------------------------------------------------------------------------------

PROJECTION = REPO / d3c.PROJECTION_RELPATH
GROUPS = REPO / d3c.GROUPS_RELPATH
SHELL_OF = {letter: n for n, letter in enumerate("KLMNOPQRSTUVWXYZ", start=1)}


def _committed_csv(path: pathlib.Path) -> list[dict[str, str]]:
    text = path.read_bytes().decode("ascii")
    assert "\r" not in text
    return list(csv.DictReader(io.StringIO(text)))


def _table_cells(path: pathlib.Path) -> dict[tuple[int, int], list[str]]:
    """The record cells of a .g4dat as text, by this test's own reading of the file."""
    out = {}
    for line in path.read_bytes().decode("ascii").split(NL):
        if line.strip() and not line.startswith("#"):
            fields = line.split()
            out[(int(fields[0]), int(fields[1]))] = fields[2:]
    return out


def _shell_text(kshell, levels, key: tuple[int, int], n: int) -> str:
    return kshell[key][0] if n == 1 else levels[key][n - 2]


def _quantity(b: dict[str, Decimal], quantity: str) -> Decimal:
    k, means = md.quantities(b)
    return k if quantity == "K" else means[int(quantity[1:]) - 2]


def _shell_quantity(n: int) -> str:
    return "K" if n == 1 else f"e{n}"


def _nuclide_target(cells, key: tuple[int, int]) -> Decimal | None:
    sigmas = [Decimal(c.unc_kev) for c in cells if c.nuclide == key and (c.gated or c.reason == "centroid")]
    return Decimal("0.1") * min(sigmas) if sigmas else None


def _target(cells, key: tuple[int, int], value: Decimal, line: bool) -> Decimal:
    """The resolution target as this test states the rule: a printed line, a tenth of the nuclide's smallest
    compared uncertainty; anything else, the smaller of that and max(1e-6 keV, 1e-6 |value|)."""
    nuclide = _nuclide_target(cells, key)
    engineering = max(Decimal("1e-6"), Decimal("1e-6") * abs(value))
    if line and nuclide is not None:
        return nuclide
    return engineering if nuclide is None else min(nuclide, engineering)


def _verdict(changes: dict[str, Decimal], target: Decimal) -> str:
    limits = {"k4": Decimal("0.1"), "tol8": Decimal("0.01"), "nu90": Decimal("0.1"), "ns16": Decimal("0.1")}
    stable = all(abs(change) <= limits[tag] * target for tag, change in changes.items())
    return "ONE_STEP_STABLE" if stable else "ONE_STEP_UNSTABLE"


def _changes_text(changes: dict[str, Decimal]) -> str:
    return ";".join(f"{tag}={d3c._text(value)}" for tag, value in changes.items())


def test_t123_drill_reversed_shell_pair_is_refused_through_stock_and_line_helpers():
    levels = (2.0, 1.0)
    assert md.geant4_line_kev(levels, "K-L") == d3c.stock_kev(levels, 1, 2)
    with pytest.raises(md.CellError, match="lower to a higher shell"):
        d3c.stock_kev(levels, 2, 1)
    with pytest.raises(md.CellError, match="lower to a higher shell"):
        md.geant4_line_kev(levels, "L-K")


def test_t123_drill_the_one_step_verdict_holds_each_limit_at_its_edge():
    target = Decimal("1")
    for tag, limit in d3c.REFINEMENT_LIMITS.items():
        assert d3c.refinement_verdict({tag: limit * target}, target) == d3c.ONE_STEP_STABLE
        assert d3c.refinement_verdict({tag: -limit * target}, target) == d3c.ONE_STEP_STABLE
        assert d3c.refinement_verdict({tag: (limit * target).next_plus()}, target) == d3c.ONE_STEP_UNSTABLE
    expected = {"k4": Decimal("0.1"), "tol8": Decimal("0.01"), "nu90": Decimal("0.1"), "ns16": Decimal("0.1")}
    assert expected == d3c.REFINEMENT_LIMITS


def test_t123_the_projection_equals_an_independent_rederivation_from_the_tables_text():
    out = _outputs()
    cells, _ = md.load_validation(CELLS, ORIGIN)
    stock = md.load_geant4_levels(GEANT4_LEVELS)
    kshell = _table_cells(KSHELL_LAYER1)
    levels = _table_cells(LEVELS_LAYER1)
    inputs = {row.nuclide: row for row in out.inputs}
    committed = _committed_csv(PROJECTION)
    gated = [cell for cell in cells if cell.gated]
    assert list(committed[0]) == list(d3c.PROJECTION_COLUMNS)
    assert len(committed) == len(gated)
    assert md.TOL_FACTOR == 3 and md.SIGMA_CALC == 0
    tags = [tag for tag, _settings in md.REFINEMENTS]
    for cell, row in zip(gated, committed, strict=True):
        key = cell.nuclide
        member = inputs[key]
        lower_n, upper_n = (SHELL_OF[orbit[0]] for orbit in cell.quantity.split("-"))
        assert (row["source"], row["Z"], row["A"], row["transition"], row["quantity"]) == (
            cell.source, str(cell.z), str(cell.a), cell.transition, cell.quantity)
        assert row["consumer_quantity"] == "shell_difference"
        assert (row["initial_n"], row["final_n"]) == (str(upper_n), str(lower_n))
        assert (row["measured_keV"], row["unc_keV"], row["locator"], row["copy_read"]) == (
            cell.value_kev, cell.unc_kev, cell.locator, cell.copy_read)
        anchor = md.recipe_run_id(member, "anchor", True)
        with localcontext() as context:
            context.prec = md._PRECISION
            measured = Decimal(cell.value_kev)
            tol = 3 * Decimal(cell.unc_kev)
            solver = Decimal(out.lines[anchor][cell.quantity]) / 1000
            consumer = (Decimal(_shell_text(kshell, levels, key, lower_n))
                        - Decimal(_shell_text(kshell, levels, key, upper_n)))
            photon = stock[key][lower_n - 1] - stock[key][upper_n - 1]
            with localcontext() as wide:
                wide.prec = 1000
                stock_kev = (Decimal(photon) * 1000).quantize(Decimal("1e-9"))
            assert Decimal(row["tol_keV"]) == tol
            assert Decimal(row["solver_keV"]) == solver
            assert Decimal(row["consumer_keV"]) == consumer
            assert Decimal(row["stock_keV"]) == stock_kev
            for name, value in (("solver", solver), ("consumer", consumer), ("stock", stock_kev)):
                assert Decimal(row[f"{name}_residual_keV"]) == value - measured
                assert row[f"{name}_within"] == ("true" if abs(value - measured) <= tol else "false")
            assert row["consumer_closer_than_stock"] == (
                "true" if abs(consumer - measured) < abs(stock_kev - measured) else "false")
            assert Decimal(row["representation_error_keV"]) == consumer - solver
            # the one-step changes, from the refine runs' printed lines and derived bindings
            line_changes = {}
            for tag in tags:
                finer_line = out.lines[md.recipe_run_id(member, "anchor", True, tag)][cell.quantity]
                line_changes[tag] = (Decimal(finer_line) - Decimal(out.lines[anchor][cell.quantity])) / 1000
            solver_target = _target(cells, key, solver, line=True)
            assert Decimal(row["solver_numeric_target_keV"]) == solver_target
            assert (row["solver_numeric_shifts_keV"], row["solver_numeric_qualification"]) == (
                _changes_text(line_changes), _verdict(line_changes, solver_target)), row
            base = md.run_bindings(out, anchor)
            consumer_changes = {}
            for tag in tags:
                finer = md.run_bindings(out, md.recipe_run_id(member, "anchor", True, tag))
                low, high = _shell_quantity(lower_n), _shell_quantity(upper_n)
                consumer_changes[tag] = ((_quantity(finer, low) - _quantity(base, low))
                                         - (_quantity(finer, high) - _quantity(base, high)))
            consumer_target = _target(cells, key, consumer, line=False)
            assert Decimal(row["consumer_numeric_target_keV"]) == consumer_target
            assert (row["consumer_numeric_shifts_keV"], row["consumer_numeric_qualification"]) == (
                _changes_text(consumer_changes), _verdict(consumer_changes, consumer_target)), row
    assert PROJECTION.read_bytes() == d3c.render_projection(REPO)


def test_t123_drill_a_refinement_line_shifted_by_a_fifth_of_the_target_is_unstable():
    """G-refine's drill, in memory on the loaded files: the grid-step refinement run's printed line of one
    gated row moved by 0.2 times that row's resolution target (the run stays complete) makes exactly that
    row's solver line `ONE_STEP_UNSTABLE`; the committed projection is unchanged."""
    bundle = d3c.load_bundle(REPO)
    committed = _committed_csv(PROJECTION)
    first = committed[0]
    member = bundle.row((int(first["Z"]), int(first["A"])))
    run = md.recipe_run_id(member, "anchor", True, "k4")
    printed = Decimal(bundle.outputs.lines[run][first["quantity"]])
    shift = Decimal("0.2") * Decimal(first["solver_numeric_target_keV"]) * 1000
    bundle.outputs.lines[run][first["quantity"]] = format(printed + shift, f".{md.RECIPE_XR_PRECISION}f")
    moved = d3c.project_shell_rows(REPO, bundle)
    assert moved[0]["solver_numeric_qualification"] == d3c.ONE_STEP_UNSTABLE
    rows_changed = [i for i, (a, b) in enumerate(zip(committed, moved, strict=True))
                    if a["solver_numeric_qualification"] != b["solver_numeric_qualification"]]
    assert rows_changed == [i for i, r in enumerate(committed)
                            if (r["Z"], r["A"], r["quantity"]) == (first["Z"], first["A"], first["quantity"])]


def _counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "gated": len(rows),
        "solver_within": sum(r["solver_within"] == "true" for r in rows),
        "consumer_within": sum(r["consumer_within"] == "true" for r in rows),
        "stock_within": sum(r["stock_within"] == "true" for r in rows),
        "consumer_closer_than_stock": sum(r["consumer_closer_than_stock"] == "true" for r in rows),
    }


def test_t123_the_projection_counts_are_pinned():
    """Regression pins, copied from the regeneration's own summary line -- not quotas: a change trips
    this test on purpose."""
    counts = _counts(_committed_csv(PROJECTION))
    assert counts == {"gated": 67, "solver_within": 30, "consumer_within": 1, "stock_within": 0,
                      "consumer_closer_than_stock": 42}
    print("\nprojection counts: " + " ".join(f"{k} {v}" for k, v in counts.items()))


def _d3_copy(tmp_path: pathlib.Path) -> pathlib.Path:
    """A root whose data/g4/d3 is a copy of the committed directory."""
    target = tmp_path / md.D3_RELDIR
    target.mkdir(parents=True)
    for source in D3DIR.iterdir():
        if source.is_file():
            (target / source.name).write_bytes(source.read_bytes())
    import shutil
    shutil.copytree(D3DIR / "reference", target / "reference")
    return tmp_path


CONSUMER_COLUMNS = {"consumer_keV", "consumer_residual_keV", "consumer_within", "consumer_closer_than_stock",
                    "representation_error_keV", "consumer_numeric_target_keV",
                    "consumer_numeric_qualification"}


def test_t123_drill_a_moved_e2_cell_changes_exactly_that_nuclides_consumer_columns(tmp_path):
    root = _d3_copy(tmp_path)
    levels_path = root / d3c.LEVELS_RELPATH
    text = levels_path.read_bytes().decode("ascii")
    line = next(ln for ln in text.split(NL) if ln.split()[:2] == ["82", "208"])
    fields = line.split()
    moved = line.replace(fields[2], "9" + fields[2], 1)
    assert moved != line
    levels_path.write_bytes(_replace_once(text, line, moved).encode("ascii"))
    before = _committed_csv(PROJECTION)
    after = d3c.project_shell_rows(root)
    changed = [(b, a) for b, a in zip(before, after, strict=True) if b != a]
    assert changed and all(b["Z"] == "82" and b["A"] == "208" for b, _ in changed)
    touching_e2 = {c["quantity"] for c in before if c["Z"] == "82" and c["A"] == "208"
                   and 2 in {SHELL_OF[o[0]] for o in c["quantity"].split("-")}}
    assert {b["quantity"] for b, _ in changed} == touching_e2
    for b, a in changed:
        differing = {column for column in b if b[column] != a[column]}
        assert differing <= CONSUMER_COLUMNS and "consumer_keV" in differing, differing


# --------------------------------------------------------------------------------------------
# T-124 -- the band intersections of the lines sharing a shell pair, re-derived from the cells
# --------------------------------------------------------------------------------------------


def test_t124_the_groups_equal_an_independent_rederivation_from_the_cells():
    cells, _ = md.load_validation(CELLS, ORIGIN)
    committed = _committed_csv(GROUPS)
    assert list(committed[0]) == list(d3c.GROUPS_COLUMNS)
    groups: dict[tuple[str, int, int, int, int], list[md.Cell]] = {}
    for cell in cells:
        if cell.gated:
            lower_n, upper_n = (SHELL_OF[orbit[0]] for orbit in cell.quantity.split("-"))
            groups.setdefault((cell.source, cell.z, cell.a, upper_n, lower_n), []).append(cell)
    expected = []
    for key in sorted(groups):
        members = groups[key]
        if len({cell.quantity for cell in members}) < 2:
            continue
        with localcontext() as context:
            context.prec = md._PRECISION
            lower = max(Decimal(c.value_kev) - 3 * Decimal(c.unc_kev) for c in members)
            upper = min(Decimal(c.value_kev) + 3 * Decimal(c.unc_kev) for c in members)
        basis = []
        for cell in members:
            if cell.copy_read not in basis:
                basis.append(cell.copy_read)
        expected.append((key, len(members), ";".join(c.quantity for c in members), lower, upper,
                         ";".join(basis)))
    assert len(committed) == len(expected)
    for row, (key, lines, quantities, lower, upper, basis) in zip(committed, expected, strict=True):
        assert (row["source"], int(row["Z"]), int(row["A"]), int(row["initial_n"]),
                int(row["final_n"])) == key
        assert (row["lines"], row["quantities"], row["basis"]) == (str(lines), quantities, basis)
        assert Decimal(row["lower_keV"]) == lower and Decimal(row["upper_keV"]) == upper
        assert Decimal(row["gap_keV"]) == lower - upper
        assert row["empty"] == ("true" if lower - upper > 0 else "false")
    assert GROUPS.read_bytes() == d3c.render_groups(REPO)


def test_t124_every_multi_line_group_is_empty_and_the_widest_is_pb208_k_to_l():
    """Pins copied from the regeneration's summary; the Pb-208 shell difference is an assertion target
    computed here from the tables' text, never an input."""
    committed = _committed_csv(GROUPS)
    empty = [row for row in committed if row["empty"] == "true"]
    assert (len(committed), len(empty)) == (29, 29)
    pb = next(row for row in committed
              if (row["source"], row["Z"], row["A"], row["initial_n"], row["final_n"])
              == ("Fricke1995", "82", "208", "2", "1"))
    assert Decimal(pb["gap_keV"]) == Decimal("184.226")
    assert pb == max(committed, key=lambda row: Decimal(row["gap_keV"]))
    kshell = _table_cells(KSHELL_LAYER1)
    levels = _table_cells(LEVELS_LAYER1)
    with localcontext() as context:
        context.prec = md._PRECISION
        k_to_l = (Decimal(_shell_text(kshell, levels, (82, 208), 1))
                  - Decimal(_shell_text(kshell, levels, (82, 208), 2)))
    assert abs(k_to_l - Decimal("5902.825353012")) <= Decimal("1e-9")
    projection = _committed_csv(PROJECTION)
    pb_k_lines = [row for row in projection if row["Z"] == "82" and row["quantity"].startswith("K1-L")]
    assert {row["consumer_keV"] for row in pb_k_lines} == {d3c._text(k_to_l)}
    print(f"\ngroups {len(committed)} empty {len(empty)}; Pb-208 K-L gap {pb['gap_keV']} keV; "
          f"consumer K-L {k_to_l}")


def test_t124_drill_a_widened_pb208_uncertainty_flips_the_group_to_non_empty(tmp_path):
    root = _d3_copy(tmp_path)
    cells_path = root / md.CELLS_RELPATH
    text = cells_path.read_bytes().decode("ascii")
    mutated = _replace_once(text, "Fricke1995,82,208,2p1/2-1s1/2,K1-L2,5778.058,0.100,",
                            "Fricke1995,82,208,2p1/2-1s1/2,K1-L2,5778.058,99.000,")
    cells_path.write_bytes(mutated.encode("ascii"))
    groups = d3c.incompatible_groups(d3c.project_shell_rows(root))
    pb = next(row for row in groups if (row["source"], row["Z"], row["A"], row["initial_n"], row["final_n"])
              == ("Fricke1995", "82", "208", "2", "1"))
    assert pb["empty"] == "false" and Decimal(pb["gap_keV"]) < 0
    assert sum(row["empty"] == "true" for row in groups) == len(groups) - 1


# --------------------------------------------------------------------------------------------
# T-125 -- the radius lineage: what is recorded of the experiments behind each compared radius
# --------------------------------------------------------------------------------------------

LINEAGE = REPO / d3c.LINEAGE_RELPATH


def test_t125_the_lineage_names_the_gated_nuclides_in_order_and_every_row_is_unknown_today():
    cells, origins = md.load_validation(CELLS, ORIGIN)
    rows = d3c.load_radius_lineage(LINEAGE)
    assert [row.nuclide for row in rows] == md.gated_nuclides(cells)
    sources = {row.input_source for row in rows}
    assert len(sources) == 1
    source = sources.pop()
    assert md.RADII_BLOB in source and source.startswith("nuclear_radii.dat blob ")
    assert "(Angeli and Marinova 2013)" in source
    for row in rows:
        assert row.dependency_state == d3c.UNKNOWN
        assert row.primary_experiment_id == d3c.NOT_TRACED and row.calibration_inputs == d3c.NOT_TRACED
        assert row.method == "evaluated compilation" and row.covariance_source == "none"
        seen: list[str] = []
        for cell in cells:
            if cell.gated and cell.nuclide == row.nuclide and cell.source not in seen:
                seen.append(cell.source)
        assert row.comparison_experiment_id == "; ".join(seen)
        assert row.locator == origins[row.nuclide].locator
    audit = d3c.audit_dependencies(REPO)
    assert [(r["Z"], r["A"]) for r in audit] == [(str(z), str(a)) for z, a in md.gated_nuclides(cells)]
    for record, row in zip(audit, rows, strict=True):
        assert record["dependency_state"] == row.dependency_state
        assert int(record["gated_rows"]) == sum(1 for c in cells if c.gated and c.nuclide == row.nuclide)
    states: dict[str, int] = {}
    for row in rows:
        states[row.dependency_state] = states.get(row.dependency_state, 0) + 1
    print(f"\nlineage rows {len(rows)}: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())))


def test_t125_the_weakly_sensitive_rows_are_isotopes_of_one_element():
    weak = [row for row in _committed_validation()
            if row["gated"] == "true" and row["label"] == md.WEAKLY_SENSITIVE]
    nuclides = sorted({(int(row["Z"]), int(row["A"])) for row in weak})
    assert weak and {z for z, _ in nuclides} == {46}
    out = md.load_outputs(REPO)
    symbols = {row.z: row.symbol for row in out.inputs}
    assert symbols[46] == "Pd"
    print(f"\nweakly sensitive rows {len(weak)} over {len(nuclides)} isotopes of Z=46 ({symbols[46]})")


def _lineage_drill(tmp_path, mutate, error, message):
    """Copy the lineage file and the cells beside it, check the copy loads, corrupt it, require
    the named refusal."""
    for source in (LINEAGE, CELLS):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    copy = tmp_path / LINEAGE.name
    text = LINEAGE.read_bytes().decode("ascii")
    d3c.load_radius_lineage(copy)
    mutated = mutate(text)
    assert mutated != text
    copy.write_bytes(mutated.encode("utf-8"))
    with pytest.raises(error, match=re.escape(message)):
        d3c.load_radius_lineage(copy)


def _lineage_cell(index: int, new: str):
    def edit(text: str) -> str:
        rows = list(csv.reader(io.StringIO(text)))
        rows[1][index] = new
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator=NL).writerows(rows)
        return buffer.getvalue()
    return edit


LINEAGE_DRILLS = [
    ("a carriage return", lambda t: t.replace(NL, "\r" + NL, 1), md.CarriageReturnError, "contains CR"),
    ("a non-ASCII byte", lambda t: t.replace("not traced", "not tracéd", 1), md.NonAsciiError,
     "outside US-ASCII"),
    ("a renamed column", lambda t: _replace_once(t, "dependency_state,locator", "state,locator"),
     md.HeaderError, "header is"),
    ("an unknown state", _lineage_cell(8, "TRACED"), md.CellError, "dependency_state"),
    ("SHARED with an untraced primary", _lineage_cell(8, d3c.SHARED), md.CellError,
     "SHARED needs a traced primary_experiment_id"),
    ("DISJOINT_DOCUMENTED with an untraced primary", _lineage_cell(8, d3c.DISJOINT_DOCUMENTED), md.CellError,
     "DISJOINT_DOCUMENTED needs a traced primary_experiment_id"),
    ("an empty locator", _lineage_cell(9, ""), md.CellError, "every row must carry a locator"),
    ("a missing nuclide", lambda t: NL.join([*_lines(t)[:1], *_lines(t)[2:]]), md.CellError,
     "the gated nuclides in order are"),
    ("rows out of order", lambda t: _swap_lines(t, 1), md.CellError, "the gated nuclides in order are"),
    ("a duplicated row", lambda t: _repeat_line(t, 1), md.DuplicateKeyError, "duplicate key"),
    ("no rows", lambda t: _lines(t)[0] + NL, md.EmptyError, "carries no rows"),
]


@pytest.mark.parametrize("label, mutate, error, message", LINEAGE_DRILLS, ids=[d[0] for d in LINEAGE_DRILLS])
def test_t125_drill_each_lineage_rule_refuses_its_fixture(tmp_path, label, mutate, error, message):
    _lineage_drill(tmp_path, mutate, error, message)


def test_t125_drill_a_shared_row_with_a_traced_primary_and_calibration_loads(tmp_path):
    for source in (LINEAGE, CELLS):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    copy = tmp_path / LINEAGE.name
    text = copy.read_bytes().decode("ascii")
    rows = list(csv.reader(io.StringIO(text)))
    rows[1][3], rows[1][5], rows[1][8] = "experiment X", "calibration Y", d3c.SHARED
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator=NL).writerows(rows)
    copy.write_bytes(buffer.getvalue().encode("ascii"))
    loaded = d3c.load_radius_lineage(copy)
    assert loaded[0].dependency_state == d3c.SHARED and loaded[0].primary_experiment_id == "experiment X"


# --------------------------------------------------------------------------------------------
# T-126 -- the component records: every field re-derived from the committed files
# --------------------------------------------------------------------------------------------

COMPONENTS = REPO / d3c.COMPONENTS_RELPATH
NUMERICAL_CHECK = REPO / d3c.NUMERICAL_CHECK_RELPATH
COMPONENT_KEYS = (
    "Z", "A", "quantity", "central_keV", "radius_plus_relative_response_keV",
    "radius_minus_relative_response_keV", "radius_plus_absolute_response_keV",
    "radius_minus_absolute_response_keV", "anchor_rounding_bound_keV", "anchor_coefficient",
    "line_rounding_bound_keV", "bue_np_keV", "bue_1_keV", "refinement_changes_keV",
    "noU_refinement_changes_keV",
    "refinement_target_keV", "refinement_verdict", "noU_refinement_verdict", "certificate",
    "legacy_radius_response_magnitude_with_floor",
)
MEMBER_KEYS = ("record", "Z", "A", "pairing", "sigma_rms_fm", "radius_minus_status", "correlation_groups")
PAIRING_KEYS = ("on_run", "off_run", "on_input_sha256", "off_input_sha256", "probe_c_bohr", "probe_t_bohr")


def _component_records(path: pathlib.Path) -> list[dict]:
    text = path.read_bytes().decode("ascii")
    assert "\r" not in text and text.endswith(NL)
    return [json.loads(line) for line in text.split(NL)[:-1]]


def _path_lengths() -> dict[str, int]:
    """Lines summed from the anchor orbit R15 to each circular state, counted here: down the upper chain, out
    the lower one."""
    path = {md.orbit(md.N_MAX, True): 0}
    for n in range(md.N_MAX - 1, 0, -1):
        path[md.orbit(n, True)] = path[md.orbit(n + 1, True)] + 1
    for n in range(2, md.N_MAX + 1):
        path[md.orbit(n, False)] = path[md.orbit(n - 1, n <= 2)] + 1
    return path


def _line_bound(rho_kev: Decimal, quantity: str) -> Decimal:
    path = _path_lengths()
    if quantity == "K":
        return path["K1"] * rho_kev
    n = int(quantity[1:])
    ell = n - 1
    weighted = 2 * ell * path[md.orbit(n, False)] + (2 * ell + 2) * path[md.orbit(n, True)]
    return Decimal(weighted) / (4 * ell + 2) * rho_kev


def _checks() -> dict[tuple[int, int, bool, str], dict[str, str]]:
    return {(int(r["Z"]), int(r["A"]), r["uehling"] == "on", r["quantity"]): r
            for r in _committed_rows(d3c.NUMERICAL_CHECK_RELPATH)}


def _component_problems(records: list[dict]) -> list[str]:
    """Every record of ``records`` that the committed files do not give, named."""
    out = _outputs()
    cells, _ = md.load_validation(CELLS, ORIGIN)
    kshell = _table_cells(KSHELL_LAYER1)
    levels = _table_cells(LEVELS_LAYER1)
    kept = md.kept_members(out)
    table_rows, _ = md.table_rows(out)
    checks = _checks()
    tags = [tag for tag, _settings in md.REFINEMENTS]
    problems: list[str] = []
    header, body = records[0], records[1:]
    digests = sorted(hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in d3c.COMPONENT_INPUTS)
    expected_header = {
        "record": "header", "profile": md.PROFILE, "radius_perturbation": "source_rms_sigma",
        "baseline_includes_uehling": "all_orders", "assembly_rule": d3c.ASSEMBLY_RULE,
        "not_computed": dict(d3c.NOT_COMPUTED_REASONS),
        "refinement_limits": {"k4": "0.1", "tol8": "0.01", "nu90": "0.1", "ns16": "0.1"},
        "omitted_components": ["model discrepancy", "nuclear polarization", "self energy",
                               "higher-order vacuum polarization", "recoil beyond the reduced mass",
                               "electron screening", "hyperfine structure",
                               "nuclear charge distributions other than 2pF",
                               "numerical settings beyond one refinement step"],
        "total_sigma_keV": None, "model_uncertainty": "unknown",
        "numerical_check_status": d3c.load_numerical_check_status(REPO, set(_checks()))["status"],
        "inputs_sha256": hashlib.sha256((NL.join(digests) + NL).encode("ascii")).hexdigest(),
    }
    if header != expected_header:
        problems.append("header")
    if set(d3c.NOT_COMPUTED_REASONS) != {"bue_1", "shape_exports"}:
        problems.append("not_computed")
    expected: list[dict] = []
    with localcontext() as context:
        context.prec = md._PRECISION
        for row in kept:
            key = row.nuclide
            on, off = md.recipe_run_id(row, "anchor", True), md.recipe_run_id(row, "anchor", False)
            minus_run = md.recipe_run_id(row, "rminus", True)
            probe = checks.get((row.z, row.a, True, "K"))
            expected.append({
                "record": "member", "Z": row.z, "A": row.a,
                "pairing": _expected_pairing(row, out, checks),
                "sigma_rms_fm": row.sigma_rms_fm,
                "radius_minus_status": "RAN" if minus_run in out.runs else "INVALID_PERTURBATION_DOMAIN",
                "correlation_groups": [f"anchor:{on}", f"radius:{row.symbol}{row.a}"],
            })
            assert probe is None or probe["run"] == on
            base, base_off = md.run_bindings(out, on), md.run_bindings(out, off)
            plus = md.run_bindings(out, md.recipe_run_id(row, "rsig", True))
            minus = md.run_bindings(out, minus_run) if minus_run in out.runs else None
            r = out.runs[on]
            floor = _ulp_floor(r.r15_total, md.E_H_EV_2022)
            rho = Decimal(5) * Decimal(10) ** (-md.RECIPE_XR_PRECISION - 1) + floor
            anchor = (_half_unit(out.headers[on][md.orbit(md.N_MAX + 1, True)]) + floor + rho) / 1000
            for index, quantity in enumerate(("K", *(f"e{n}" for n in range(2, md.N_MAX + 1)))):
                central = kshell[key][0] if quantity == "K" else levels[key][index - 1]
                legacy = kshell[key][1] if quantity == "K" else levels[key][md.N_MAX - 1 + index - 1]
                q_on, q_off = _quantity(base, quantity), _quantity(base_off, quantity)
                on_changes = {tag: _quantity(md.run_bindings(out, md.recipe_run_id(row, "anchor", True, tag)),
                                             quantity) - q_on for tag in tags}
                off_changes = {
                    tag: _quantity(md.run_bindings(out, md.recipe_run_id(row, "anchor", False, tag)),
                                   quantity) - q_off
                    for tag in tags if tag in md.NOU_REFINEMENTS}
                target, target_off = _target(cells, key, q_on, False), _target(cells, key, q_off, False)
                check = checks.get((row.z, row.a, True, quantity))

                def response(moved, relative, quantity=quantity, base=base):
                    if moved is None:
                        return None
                    if relative:
                        moved, base = md.relative(moved), md.relative(base)
                    return d3c._text(_quantity(moved, quantity) - _quantity(base, quantity))

                expected.append({
                    "Z": row.z, "A": row.a, "quantity": quantity, "central_keV": central,
                    "radius_plus_relative_response_keV": response(plus, True),
                    "radius_minus_relative_response_keV": response(minus, True),
                    "radius_plus_absolute_response_keV": response(plus, False),
                    "radius_minus_absolute_response_keV": response(minus, False),
                    "anchor_rounding_bound_keV": d3c._text(anchor),
                    "anchor_coefficient": "1",
                    "line_rounding_bound_keV": d3c._text(_line_bound(rho / 1000, quantity)),
                    "bue_np_keV": d3c._text(q_on - q_off),
                    "bue_1_keV": d3c._text(Decimal(check["bue_1_eV"]) / 1000) if check else "NOT_COMPUTED",
                    "refinement_changes_keV": {tag: d3c._text(v) for tag, v in on_changes.items()},
                    "noU_refinement_changes_keV": {tag: d3c._text(v) for tag, v in off_changes.items()},
                    "refinement_target_keV": d3c._text(target),
                    "refinement_verdict": _verdict(on_changes, target),
                    "noU_refinement_verdict": _verdict(off_changes, target_off),
                    "certificate": None if check is None else {
                        "set": check["set"], "verdict": check["verdict"],
                        "Q_keV": d3c._text(Decimal(check["Q_eV"]) / 1000),
                        "Q_ref_keV": d3c._text(Decimal(check["Q_ref_eV"]) / 1000),
                        "b_num_keV": d3c._text(Decimal(check["b_num_eV"]) / 1000),
                        "target_keV": d3c._text(Decimal(check["target_eV"]) / 1000)},
                    "legacy_radius_response_magnitude_with_floor": legacy,
                })
                with localcontext() as legacy_context:
                    legacy_context.prec = md._PRECISION
                    floor_kev = md.unc_floor_kev(out, (on, md.recipe_run_id(row, "rsig", True)))
                    plus_rel = Decimal(response(plus, True))
                assert float(legacy) == float(max(abs(plus_rel), floor_kev)), (key, quantity)
    for table_row in table_rows:
        if table_row.a == 0:
            expected.append({"record": "reference", "Z": table_row.z, "A": 0,
                             "references": {"Z": table_row.z, "A": table_row.carries}})
    if len(body) != len(expected):
        problems.append(f"record count {len(body)} != {len(expected)}")
    for got, want in zip(body, expected, strict=False):
        if got != want:
            name = want.get("quantity", want.get("record"))
            problems.append(f"record {want.get('Z')},{want.get('A')},{name}")
    return problems


def _expected_pairing(row: md.InputRow, out: md.Outputs, checks) -> dict[str, object]:
    """The pairing of a member's anchor run with its Uehling-free twin, rebuilt from the run inputs: both run
    ids, the sha256 of each input as the renderer makes it now, and the (c, t) the independent check probed,
    the same on both twins, where it checked the member."""
    cells = out.cells
    extra = md.extra_lines(cells, row.nuclide)
    on_text = md.render_recipe_input(row, "anchor", "recipe", True, md.RECIPE_SETTINGS, extra)
    off_text = md.render_recipe_input(row, "anchor", "recipe", False, md.RECIPE_SETTINGS, extra)
    on_probe, off_probe = checks.get((row.z, row.a, True, "K")), checks.get((row.z, row.a, False, "K"))
    if (on_probe is None) != (off_probe is None):
        raise AssertionError(f"{row.nuclide}: the independent check carries one twin only")
    if on_probe is not None and (on_probe["probe_c_bohr"], on_probe["probe_t_bohr"]) != (
            off_probe["probe_c_bohr"], off_probe["probe_t_bohr"]):
        raise AssertionError(f"{row.nuclide}: the twins were probed at different (c, t)")
    return {
        "on_run": md.recipe_run_id(row, "anchor", True), "off_run": md.recipe_run_id(row, "anchor", False),
        "on_input_sha256": hashlib.sha256(on_text.encode("ascii")).hexdigest(),
        "off_input_sha256": hashlib.sha256(off_text.encode("ascii")).hexdigest(),
        "probe_c_bohr": on_probe["probe_c_bohr"] if on_probe else None,
        "probe_t_bohr": on_probe["probe_t_bohr"] if on_probe else None,
    }


def test_t126_every_component_field_is_rederived_from_the_committed_files():
    records = _component_records(COMPONENTS)
    assert not _component_problems(records)
    body = [r for r in records[1:] if "quantity" in r]
    members = [r for r in records[1:] if r.get("record") == "member"]
    assert all(list(r) == list(COMPONENT_KEYS) for r in body)
    assert all(list(r) == list(MEMBER_KEYS) and list(r["pairing"]) == list(PAIRING_KEYS) for r in members)
    rows, _ = md.table_rows(_outputs())
    isotopes, natural = sum(1 for r in rows if r.a > 0), sum(1 for r in rows if r.a == 0)
    assert len(records) == isotopes * (len(d3c.QUANTITIES) + 1) + natural + 1
    assert COMPONENTS.read_bytes() == d3c.render_components(REPO)
    histogram: dict[str, int] = {}
    for r in body:
        histogram[r["refinement_verdict"]] = histogram.get(r["refinement_verdict"], 0) + 1
    minus_made = sum(r["radius_minus_relative_response_keV"] is not None for r in body)
    certified = sum(r["certificate"] is not None for r in body)
    print(f"\ncomponents {len(body)}; rminus responses present {minus_made}; certificates {certified}; "
          "one-step verdicts " + ", ".join(f"{k} {v}" for k, v in sorted(histogram.items())))


def test_t126_the_anchor_moves_every_absolute_binding_by_its_coefficient_and_cancels_in_every_difference():
    """A moved anchor header moves every derived binding, K and each (2j+1) mean alike, by exactly
    its shift (the weights sum to one); the anchor-subtracted values, and every difference of two
    derived values, do not move; a difference against a value computed elsewhere does."""
    out = _outputs()
    row = next(r for r in out.inputs if r.nuclide == (82, 208))
    run = md.recipe_run_id(row, "anchor", True)
    headers, lines = dict(out.headers[run]), out.lines[run]
    outer = md.orbit(md.N_MAX + 1, True)
    delta = Decimal("0.5")
    moved_headers = {**headers, outer: format(Decimal(headers[outer]) - delta, "f")}

    b = md.run_bindings(out, run)
    assert _unchecked_bindings(headers, lines) == b
    b2 = _unchecked_bindings(moved_headers, lines)
    quantities = ("K", *(f"e{n}" for n in range(2, md.N_MAX + 1)))
    synthetic_tail = Decimal("100")
    with localcontext() as context:
        context.prec = md._PRECISION
        for q in quantities:
            assert _quantity(b2, q) - _quantity(b, q) == delta / 1000, q
            assert _quantity(md.relative(b2), q) == _quantity(md.relative(b), q), q
            assert (_quantity(b2, q) - synthetic_tail) - (_quantity(b, q) - synthetic_tail) == delta / 1000
        for q1 in quantities:
            for q2 in quantities:
                assert _quantity(b2, q1) - _quantity(b2, q2) == _quantity(b, q1) - _quantity(b, q2)


def _unchecked_bindings(headers: dict[str, str], lines: dict[str, str]) -> dict[str, Decimal]:
    """The anchor-kind chain sums of ``derive_recipe_bindings`` without its header check: what a moved anchor
    gives every state."""
    with localcontext() as context:
        context.prec = md._PRECISION
        top = md.orbit(md.N_MAX, True)
        b = {top: -Decimal(headers[md.orbit(md.N_MAX + 1, True)]) + Decimal(lines[md.anchor_line()])}
        for n in range(md.N_MAX - 1, 0, -1):
            lower, upper = md.orbit(n, True), md.orbit(n + 1, True)
            b[lower] = b[upper] + Decimal(lines[f"{lower}-{upper}"])
        for n in range(1, md.N_MAX):
            b[md.orbit(n + 1, False)] = (b[md.orbit(n, False)]
                                          - Decimal(lines[f"{md.orbit(n, False)}-{md.orbit(n + 1, False)}"]))
    return b


def test_t126_drill_a_tampered_refinement_change_is_refused():
    records = _component_records(COMPONENTS)
    tampered = json.loads(json.dumps(records))
    target = next(r for r in tampered[1:] if "quantity" in r)
    target["refinement_changes_keV"]["k4"] += "1"
    assert _component_problems(tampered) == [f"record {target['Z']},{target['A']},{target['quantity']}"]


# --------------------------------------------------------------------------------------------
# T-127 -- the independent check of the numerical solution: every row bound to a committed run
# --------------------------------------------------------------------------------------------


def _committed_q(out: md.Outputs, run: str, quantity: str) -> Decimal:
    """The value a checked quantity has in a committed run, in eV, computed here."""
    b = md.run_bindings(out, run)
    with localcontext() as context:
        context.prec = md._PRECISION
        if quantity.startswith("level:"):
            return b[quantity.split(":")[1]]
        if quantity.startswith("line:"):
            return Decimal(out.lines[run][quantity.split(":")[1]])
        if "-" in quantity:
            first, second = quantity.split("-")
            return (_quantity(b, first) - _quantity(b, second)) * 1000
        return _quantity(b, quantity) * 1000


def test_t127_every_checked_value_is_the_committed_production_value_at_its_printed_decimals():
    """G-binding: each row's Q_eV is the value its production run gives, within half a unit of the last
    decimal it prints; its run and digest name that run; its twin carries the same probed (c, t)."""
    out = _outputs()
    inputs = {row.nuclide: row for row in out.inputs}
    rows = _committed_rows(d3c.NUMERICAL_CHECK_RELPATH)
    assert list(rows[0]) == list(d3c.NUMERICAL_CHECK_COLUMNS)
    largest = Decimal(0)
    for r in rows:
        row = inputs[(int(r["Z"]), int(r["A"]))]
        run = md.recipe_run_id(row, "anchor", r["uehling"] == "on")
        assert (r["run"], r["input_sha256"], r["kind"]) == (run, out.runs[run].input_sha256, "anchor")
        half = _half_unit(r["Q_eV"])
        difference = abs(Decimal(r["Q_eV"]) - _committed_q(out, run, r["quantity"]))
        assert difference <= half, (r["set"], run, r["quantity"], difference)
        largest = max(largest, difference / half)
    assert len(d3c.load_numerical_check(REPO, out)) == len(rows)
    print(f"\nnumerical_check rows {len(rows)}; largest |Q - committed| {largest} of half a printed unit")


def test_t127_drill_a_checked_value_moved_by_one_unit_of_its_last_digit_is_refused(tmp_path):
    """G-binding's drill: one row's Q_eV moved by one unit in its last printed digit."""
    root = _d3_copy(tmp_path)
    path = root / d3c.NUMERICAL_CHECK_RELPATH
    text = path.read_bytes().decode("ascii")
    line = _lines(text)[1]
    fields = line.split(",")
    q = Decimal(fields[6])
    fields[6] = str(q + 2 * _half_unit(fields[6]))
    path.write_bytes(_replace_once(text, line, ",".join(fields)).encode("ascii"))
    with pytest.raises(md.CellError, match=f"Q_eV {fields[6]} is not {fields[13]}'s committed"):
        d3c.load_numerical_check(root, _outputs())


NUMERICAL_CHECK_DRILLS = [
    ("another run's digest", "on", 14, lambda v: "0" * 64,
     "run and input_sha256 must name the production run"),
    ("an unknown set", "on", 0, lambda v: "pilot_X", "outside the file's vocabulary"),
    ("an unknown verdict", "on", 11, lambda v: "PASS", "verdict 'PASS'"),
    ("a first-order Uehling term on an Uehling-free row", "off", 12, lambda v: "1.0",
     "carried on the Uehling-on rows and only there"),
    ("a quantity the run does not give", "on", 5, lambda v: "level:Z99", "is not one"),
    ("twins probed at another c", "off", 15, lambda v: v + "1", "the twins carry different probed (c, t)"),
]


@pytest.mark.parametrize("label, switch, column, edit, message", NUMERICAL_CHECK_DRILLS,
                         ids=[d[0] for d in NUMERICAL_CHECK_DRILLS])
def test_t127_drill_each_numerical_check_rule_refuses_its_fixture(
    tmp_path, label, switch, column, edit, message
):
    root = _d3_copy(tmp_path)
    path = root / d3c.NUMERICAL_CHECK_RELPATH
    lines = _lines(path.read_bytes().decode("ascii"))
    index = next(i for i, line in enumerate(lines[1:], start=1) if line.split(",")[4] == switch)
    fields = lines[index].split(",")
    fields[column] = edit(fields[column])
    lines[index] = ",".join(fields)
    path.write_bytes(NL.join(lines).encode("ascii"))
    with pytest.raises(md.Mudirac130Error, match=re.escape(message)):
        d3c.load_numerical_check(root, _outputs())


def test_t127_the_sample_holds_every_compared_nuclide_the_pilots_lack_and_one_member_per_z_above_them():
    rows = _committed_rows(d3c.NUMERICAL_CHECK_RELPATH)
    sets = {name: {(int(r["Z"]), int(r["A"])) for r in rows if r["set"] == name}
            for name in d3c.NUMERICAL_CHECK_SETS}
    pilots = sets["pilot_D"] | sets["pilot_RA"] | sets["pilot_RB"]
    compared = set(md.compared_nuclides(md.load_cells(CELLS)))
    assert compared - pilots <= sets["audit"]
    top = max(z for z, _a in compared)
    heavy = sorted({row.z for row in _outputs().inputs if row.z > top})
    audit_heavy = sorted(z for z, _a in sets["audit"] if z > top)
    assert audit_heavy == heavy
    assert not pilots & sets["audit"]
    quantities = {name: {r["quantity"] for r in rows if r["set"] == name}
                  for name in d3c.NUMERICAL_CHECK_SETS}
    print(f"\npilots {len(pilots)}, audit {len(sets['audit'])}: compared without a pilot "
          f"{len(compared - pilots)}, Z above {top} {heavy}; quantities per set "
          f"{ {k: len(v) for k, v in quantities.items()} }")


# --------------------------------------------------------------------------------------------
# T-128 -- the pairing of each anchor run with its Uehling-free twin, rebuilt from the run inputs
# and the probe records and compared with the exported records
# --------------------------------------------------------------------------------------------


def pairing_problems(records: list[dict]) -> list[str]:
    """Every member record of ``records`` (the exported `components.jsonl`) whose pairing is not the one
    this test rebuilds from the run inputs and the independent check's probe records, named."""
    out = _outputs()
    checks = _checks()
    inputs = {row.nuclide: row for row in out.inputs}
    problems = []
    members = [r for r in records if r.get("record") == "member"]
    if [(r["Z"], r["A"]) for r in members] != [row.nuclide for row in md.kept_members(out)]:
        problems.append("the member records are not the kept members in order")
    for record in members:
        row = inputs[(record["Z"], record["A"])]
        expected = _expected_pairing(row, out, checks)
        if record["pairing"] != expected:
            problems.append(f"Z={row.z} A={row.a}: pairing {record['pairing']!r}")
        elif (out.runs[expected["on_run"]].input_sha256, out.runs[expected["off_run"]].input_sha256) != (
                expected["on_input_sha256"], expected["off_input_sha256"]):
            problems.append(f"Z={row.z} A={row.a}: the run table's digests are not the rendered inputs'")
    return problems


def test_t128_every_pairing_is_rebuilt_from_the_run_inputs_and_probe_records():
    records = _component_records(COMPONENTS)
    assert pairing_problems(records) == []
    members = [r for r in records if r.get("record") == "member"]
    probed = [r for r in members if r["pairing"]["probe_c_bohr"] is not None]
    keys = {json.dumps(r["pairing"], sort_keys=True) for r in members}
    assert len(keys) == len(members)
    print(f"\nmember pairings {len(members)}, distinct {len(keys)}; "
          f"probed by the independent check {len(probed)}")


def test_t128_drill_a_swapped_isotope_key_is_refused():
    """The pairings of two probed isotopes of one element exchanged in a copy of the exported records."""
    records = json.loads(json.dumps(_component_records(COMPONENTS)))
    members = [r for r in records if r.get("record") == "member" and r["pairing"]["probe_c_bohr"] is not None]
    first, second = next((a, b) for a in members for b in members if a["Z"] == b["Z"] and a["A"] < b["A"])
    first["pairing"], second["pairing"] = second["pairing"], first["pairing"]
    assert pairing_problems(records) == [f"Z={first['Z']} A={first['A']}: pairing {first['pairing']!r}",
                                         f"Z={second['Z']} A={second['A']}: pairing {second['pairing']!r}"]


def test_t128_drill_a_field_definition_written_over_the_concrete_pairing_is_refused():
    """The defect of an inherited serializer: every record's concrete pairing replaced by the list of the
    pairing's field names."""
    records = json.loads(json.dumps(_component_records(COMPONENTS)))
    members = [r for r in records if r.get("record") == "member"]
    for record in members:
        record["pairing"] = list(PAIRING_KEYS)
    problems = pairing_problems(records)
    assert len(problems) == len(members)
    assert all(p.endswith(f"pairing {list(PAIRING_KEYS)!r}") for p in problems)


# T-129 -- the retrospective K-L screen: the summary, its helpers and the frozen screen


def test_t129_drill_within_unit_includes_the_edge_only():
    with localcontext() as context:
        context.prec = 50
        unit = centroids._UNIT
        assert centroids._within_unit(Decimal(0), unit)
        assert not centroids._within_unit(Decimal(0), unit + Decimal("1e-40"))


def test_t129_drill_margin_max_uses_all_rows_even_when_not_representative():
    rows = []
    for cohort, source in (("labelled", "Fricke1995"), ("constructed", "Fricke1995"),
                           ("constructed", "Saito2025")):
        rows.extend((
            {"cohort": cohort, "source": source, "representative": "true", "margin_keV": "1",
             "r_d3_keV": "1", "Z": "1", "A": "1"},
            {"cohort": cohort, "source": source, "representative": "false", "margin_keV": "2",
             "r_d3_keV": "3", "Z": "2", "A": "2"},
        ))
    for row in centroids.margin_rows(rows):
        assert row["max_abs_r_d3_keV"] == "3.000000000"
        assert (row["max_abs_r_d3_Z"], row["max_abs_r_d3_A"]) == ("2", "2")


def test_t129_drill_summary_counts_shifts_above_the_tenth(monkeypatch):
    rows = []
    for cohort, source, shift in (("labelled", "Fricke1995", "2"),
                                  ("constructed", "Fricke1995", "0"),
                                  ("constructed", "Saito2025", "0")):
        rows.append({"cohort": cohort, "source": source, "Z": "1", "A": "1", "b_num_keV": shift,
                     "sigma_keV": "10", "sigma_max_keV": "", "screen": "inside",
                     "numerical_verdict": "QUALIFIED"})
    monkeypatch.setattr(centroids, "centroid_rows", lambda root: rows)
    monkeypatch.setattr(centroids, "margin_rows", lambda members: [])
    monkeypatch.setattr(md, "load_cells", lambda path: ())
    monkeypatch.setattr(centroids, "load_intensity_ratios", lambda path, cells: {})
    assert any("above a tenth of the largest sigma on 1 of 3 rows" in line
               for line in centroids.summary_lines(REPO))


def test_t129_drill_excluded_two_p_rows_omit_centroid_labels():
    from types import SimpleNamespace

    members = [SimpleNamespace(transition="2p1/2", reason=""),
               SimpleNamespace(transition="2p3/2", reason="centroid"),
               SimpleNamespace(transition="3p1/2", reason="gated")]
    assert centroids._two_p_rows(members) == "2p1/2:gated"


def test_t129_drill_summary_excludes_a_shift_exactly_at_the_tenth(monkeypatch):
    rows = [{"cohort": cohort, "source": "Fricke1995", "Z": "1", "A": "1", "b_num_keV": shift,
             "sigma_keV": "10", "sigma_max_keV": "", "screen": "inside", "numerical_verdict": "QUALIFIED"}
            for cohort, shift in (("labelled", "1"), ("constructed", "2"))]
    monkeypatch.setattr(centroids, "centroid_rows", lambda root: rows)
    monkeypatch.setattr(centroids, "margin_rows", lambda members: [])
    monkeypatch.setattr(md, "load_cells", lambda path: ())
    monkeypatch.setattr(centroids, "load_intensity_ratios", lambda path, cells: {})
    assert any("above a tenth of the largest sigma on 1 of 2 rows" in line
               for line in centroids.summary_lines(REPO))


def test_t129_screen_boundary_and_frozen_calculation_error(monkeypatch):
    sigma = Decimal("1")
    assert centroids._screen(3 * sigma, sigma) == "inside"
    assert centroids._screen(3 * sigma + Decimal("0.1"), sigma) == "outside"
    assert centroids._screen(Decimal("0.3"), sigma, Decimal("0.1")) == "inside"
    assert centroids._screen(Decimal("1"), sigma, Decimal("0.1")) == "correlation-dependent"
    assert centroids._screen(Decimal("3.1"), sigma, Decimal("0.1")) == "outside"
    assert centroids._screen(Decimal("3"), sigma, Decimal("0.1")) == "correlation-dependent"
    monkeypatch.setattr(md, "SIGMA_CALC", 1)
    with pytest.raises(md.CellError, match="freezes SIGMA_CALC"):
        centroids.centroid_rows(REPO)


def test_t129_drill_a_centroid_nuclide_the_independent_check_lacks_is_refused(monkeypatch):
    record = centroids.numerical_record(REPO)
    first = next(iter(record))
    monkeypatch.setattr(centroids, "numerical_record",
                        lambda root: {key: value for key, value in record.items() if key != first})
    message = f"Z={first[0]} A={first[1]}: numerical_check.csv carries no K-e2 row"
    with pytest.raises(md.CellError, match=message):
        centroids.centroid_rows(REPO)


# T-130 -- source ratios are copied and affect only illustrative columns


def test_t130_ratio_loader_refuses_bad_bytes_values_keys_and_duplicates(tmp_path):
    cells = md.load_cells(CELLS)
    source = REPO / centroids.RATIOS_RELPATH
    text = source.read_bytes().decode("ascii")
    path = tmp_path / source.name
    path.write_bytes(text.encode("ascii"))
    ratio = centroids.load_intensity_ratios(path, cells)
    assert len(ratio) == 1
    printed = next(iter(ratio.values()))
    assert (printed["source"], printed["Z"], printed["A"], printed["ratio"],
            printed["ratio_unc"], printed["label"]) == (
                "Jenkins1971", "82", "208", "1.67", "0.07", "This Experiment")
    assert printed["locator"] == "Table 9, PDF p. 46, row R_p"
    assert "Pb(NO3)2" in printed["target"] and "98.7%" in printed["target"]
    mutations = [
        (text.replace(NL, "\r" + NL, 1), md.CarriageReturnError),
        (text.replace("Jenkins1971", "JenkinsÂ·1971", 1), md.NonAsciiError),
        (text.replace("ratio_unc", "unc", 1), md.HeaderError),
        (text.replace(",1.67,", ",x,"), md.CellError),
        (text.replace(",1.67,", ",0.00,"), md.CellError),
        (text.replace(",0.07,", ",-0.07,"), md.CellError),
        (text.replace(",82,208,", ",82,209,"), md.CellError),
        (text + text.splitlines()[1] + NL, md.DuplicateKeyError),
    ]
    for changed, error in mutations:
        assert changed != text
        path.write_bytes(changed.encode("utf-8"))
        with pytest.raises(error):
            centroids.load_intensity_ratios(path, cells)


# T-129 -- independent re-derivation of the cohort columns and the margin groups


def _centroid_records(path):
    with path.open(newline="", encoding="ascii") as source:
        return list(csv.DictReader(source))


def _nine(value):
    with localcontext() as context:
        context.prec = 50
        return format(value.quantize(Decimal("1e-9")), ".9f")


def test_t129_drill_stock_set_equal_to_gated_set_is_admitted(monkeypatch):
    cells = md.load_cells(CELLS)
    gated = set(md.gated_nuclides(cells))
    original = md.load_geant4_levels
    monkeypatch.setattr(md, "load_geant4_levels", lambda path: {key: value for key, value in
                        original(path).items() if key in gated})
    assert set(md.load_geant4_levels(REPO / md.GEANT4_LEVELS_RELPATH)) == gated
    assert centroids.centroid_rows(REPO)


def test_t129_every_comparison_and_margin_cell_is_rederived_from_the_sources():
    cells = md.load_cells(CELLS)
    pairs = md.doublet_determinations(cells)
    labelled = [cell for cell in cells if cell.reason == "centroid"]
    rows = _centroid_records(REPO / centroids.CENTROIDS_RELPATH)
    margins = _centroid_records(REPO / centroids.MARGINS_RELPATH)
    assert tuple(rows[0]) == centroids.CENTROID_COLUMNS
    assert tuple(margins[0]) == centroids.MARGIN_COLUMNS
    assert (len([r for r in rows if r["cohort"] == "labelled"]),
            len([r for r in rows if r["cohort"] == "constructed"])) == (len(labelled), len(pairs))
    assert [(r["source"], r["Z"], r["A"], r["locator"]) for r in rows if r["cohort"] == "constructed"] == [
        (left.source, str(left.z), str(left.a), left.locator) for left, _right in pairs]
    paired = {(left.source, left.z, left.a) for left, _right in pairs}
    excluded = {(cell.source, cell.z, cell.a) for cell in cells
                if cell.gated and cell.quantity in ("K1-L2", "K1-L3") and
                (cell.source, cell.z, cell.a) not in paired}
    assert {(r["source"], int(r["Z"]), int(r["A"])) for r in rows
            if r["cohort"] == "excluded"} == excluded
    tables = d3c.Tables(d3c.read_table_text(REPO / d3c.KSHELL_RELPATH),
                        d3c.read_table_text(REPO / d3c.LEVELS_RELPATH))
    stock = md.load_geant4_levels(GEANT4_LEVELS)
    record = {(int(r["Z"]), int(r["A"])): r for r in _committed_rows(d3c.NUMERICAL_CHECK_RELPATH)
              if r["quantity"] == "K-e2" and r["uehling"] == "on"}
    ratios = centroids.load_intensity_ratios(REPO / centroids.RATIOS_RELPATH, cells)
    for row in rows:
        if row["cohort"] == "excluded":
            key = row["source"], int(row["Z"]), int(row["A"])
            present = [c for c in cells if (c.source, c.z, c.a) == key]
            assert row["excluded_gated"] == ";".join(c.quantity for c in present
                                                       if c.gated and c.quantity in ("K1-L2", "K1-L3"))
            assert row["two_p_rows"] == ";".join(
                f"{c.transition}:{c.reason or 'gated'}" for c in present
                if c.transition.startswith("2p") and c.reason != "centroid")
            assert all(value == "" for name, value in row.items() if name not in
                       {"cohort", "source", "Z", "A", "excluded_gated", "two_p_rows"})
            continue
        key = int(row["Z"]), int(row["A"])
        q = Decimal(tables.shell(key, 1)) - Decimal(tables.shell(key, 2))
        assert row["q_keV"] == _nine(q)
        check = record[key]
        assert (row["b_num_keV"], row["numerical_verdict"]) == (
            _nine(Decimal(check["b_num_eV"]) / 1000), check["verdict"])
        if row["cohort"] == "labelled":
            source = next(c for c in labelled if c.nuclide == key)
            c = Decimal(source.value_kev)
            sigma = Decimal(source.unc_kev)
            minimum = sigma
            assert row["sigma_keV"] == _nine(sigma)
            assert row["npol_keV"] == source.npol_kev
            assert row["representative"] == "true"
            assert row["comparator"] == "center of gravity"
        else:
            left, right = next((a, b) for a, b in pairs if (a.source, a.nuclide, a.locator) ==
                               (row["source"], key, row["locator"]))
            c = (Decimal(left.value_kev) + 2 * Decimal(right.value_kev)) / 3
            u2, u3 = Decimal(left.unc_kev), Decimal(right.unc_kev)
            sigma, minimum = (u2 + 2 * u3) / 3, abs(u2 - 2 * u3) / 3
            assert row["sigma_max_keV"] == _nine(sigma)
            assert row["sigma_min_keV"] == _nine(minimum)
            sigma_zero = ((u2 / 3) ** 2 + (2 * u3 / 3) ** 2).sqrt()
            assert row["sigma_0_keV"] == _nine(sigma_zero)
            assert row["npol_l2_keV"] == left.npol_kev
            assert row["npol_l3_keV"] == right.npol_kev
            assert row["comparator"] == "degeneracy-convention centroid"
            if key in ratios:
                ratio = ratios[key]
                assert (row["ratio"], row["ratio_unc"], row["ratio_source"], row["ratio_locator"],
                        row["ratio_label"]) == (ratio["ratio"], ratio["ratio_unc"], ratio["source"],
                                               ratio["locator"], ratio["label"])
                r, uncertainty = Decimal(ratio["ratio"]), Decimal(ratio["ratio_unc"])
                first, second = Decimal(left.value_kev), Decimal(right.value_kev)
                with localcontext() as context:
                    context.prec = 50
                    values = [first + trial / (1 + trial) * (second - first)
                              for trial in (r - uncertainty, r, r + uncertainty)]
                assert row["illustrative_c_keV"] == ";".join(_nine(value) for value in values)
                assert row["illustrative_q_minus_c_keV"] == ";".join(_nine(q - value) for value in values)
            else:
                assert not any(row[name] for name in ("ratio", "ratio_unc", "ratio_source",
                                                     "ratio_locator", "ratio_label", "illustrative_c_keV",
                                                     "illustrative_q_minus_c_keV"))
        residual = q - c
        assert row["c_keV"] == _nine(c)
        assert row["r_d3_keV"] == _nine(residual)
        screen = ("outside" if abs(residual) > 3 * sigma else
                  "inside" if abs(residual) <= 3 * minimum else "correlation-dependent")
        assert row["screen"] == screen
        if key in stock:
            stock_value = d3c.stock_kev(stock[key], 1, 2)
            stock_residual = stock_value - c
            assert row["s_keV"] == _nine(stock_value)
            assert row["r_stock_keV"] == _nine(stock_residual)
            assert row["margin_keV"] == _nine(abs(stock_residual) - abs(residual))
            assert row["stock_reason"] == ""
        else:
            assert (row["s_keV"], row["r_stock_keV"], row["margin_keV"]) == ("", "", "")
            assert row["stock_reason"] == "no C line in the unpatched harvest"
    grouped = {}
    for row in rows:
        if row["cohort"] == "constructed":
            grouped.setdefault((row["source"], row["Z"], row["A"]), []).append(row)
    for determinations in grouped.values():
        expected = min(determinations, key=lambda r: Decimal(r["sigma_max_keV"]))
        assert [r["locator"] for r in determinations if r["representative"] == "true"] == [
            expected["locator"]]
    groups = [("labelled", "Fricke1995"), ("constructed", "Fricke1995"),
              ("constructed", "Saito2025")]
    assert [(r["cohort"], r["source"]) for r in margins] == groups
    for summary in margins:
        group = [r for r in rows if (r["cohort"], r["source"]) ==
                 (summary["cohort"], summary["source"])]
        representative = [r for r in group if r["representative"] == "true" and r["margin_keV"]]
        values = sorted(Decimal(r["margin_keV"]) for r in representative)
        assert summary["isotopes"] == str(len({(r["Z"], r["A"]) for r in representative}))
        assert summary["margin_min_keV"] == _nine(values[0])
        assert summary["margin_max_keV"] == _nine(values[-1])
        mid = len(values) // 2
        median = _nine(values[mid]) if len(values) % 2 else f"{_nine(values[mid - 1])}..{_nine(values[mid])}"
        assert summary["margin_median_keV"] == median
        maximum = max(group, key=lambda r: abs(Decimal(r["r_d3_keV"])))
        assert (summary["max_abs_r_d3_keV"], summary["max_abs_r_d3_Z"], summary["max_abs_r_d3_A"]) == (
            _nine(abs(Decimal(maximum["r_d3_keV"]))), maximum["Z"], maximum["A"])


def test_t129_moved_measured_line_and_missing_or_extra_stock_key_are_local(monkeypatch):
    original = md.load_cells
    baseline = centroids.centroid_rows(REPO)
    cells = original(CELLS)
    target = next(c for c in cells if c.gated and c.quantity == "K1-L3")
    changed = tuple(replace(c, value_kev=str(Decimal(c.value_kev) + 1)) if c == target else c
                    for c in cells)
    monkeypatch.setattr(md, "load_cells", lambda path: (
        changed if pathlib.Path(path) == CELLS else original(path)))
    moved = centroids.centroid_rows(REPO)
    different = [i for i, (a, b) in enumerate(zip(baseline, moved, strict=True)) if a != b]
    assert len(different) == 1
    assert moved[different[0]]["locator"] == target.locator
    monkeypatch.undo()
    levels = md.load_geant4_levels(GEANT4_LEVELS)
    optional = next(key for key in md.centroid_nuclides(cells) if key not in md.gated_nuclides(cells))
    monkeypatch.setattr(md, "load_geant4_levels", lambda path: {
        k: v for k, v in levels.items() if k != optional})
    missing = centroids.centroid_rows(REPO)
    different = [i for i, (a, b) in enumerate(zip(baseline, missing, strict=True)) if a != b]
    assert len(different) == 1
    assert (int(missing[different[0]]["Z"]), int(missing[different[0]]["A"])) == optional
    assert missing[different[0]]["s_keV"] == missing[different[0]]["r_stock_keV"] == ""
    monkeypatch.setattr(md, "load_geant4_levels", lambda path: {
        **levels, (999, 999): next(iter(levels.values()))})
    with pytest.raises(md.CellError, match="unlisted key"):
        centroids.centroid_rows(REPO)
    required = md.gated_nuclides(cells)[0]
    monkeypatch.setattr(md, "load_geant4_levels", lambda path: {
        k: v for k, v in levels.items() if k != required})
    with pytest.raises(md.CellError, match="omit a gated key"):
        centroids.centroid_rows(REPO)


def test_t129_each_constructed_projection_identity_refuses_a_moved_cell(monkeypatch):
    cells = md.load_cells(CELLS)
    left, _right = md.doublet_determinations(cells)[0]
    original = md.read_rows
    projection_path = REPO / d3c.PROJECTION_RELPATH
    identity_cases = (
        ("consumer_keV", "shell projection disagrees"),
        ("solver_keV", "weighted solver lines disagree"),
        ("stock_keV", "stock level difference disagrees"),
    )
    for column, error in identity_cases:
        def moved(path, columns, field=column):
            rows = original(path, columns)
            if pathlib.Path(path) != projection_path:
                return rows
            changed = []
            for where, row in rows:
                if (row["source"], row["Z"], row["A"], row["locator"], row["quantity"]) == (
                        left.source, str(left.z), str(left.a), left.locator, left.quantity):
                    row = {**row, field: str(Decimal(row[field]) + 1)}
                changed.append((where, row))
            return changed

        with monkeypatch.context() as patcher:
            patcher.setattr(md, "read_rows", moved)
            with pytest.raises(md.CellError, match=error):
                centroids.centroid_rows(REPO)
    def missing(path, columns):
        rows = original(path, columns)
        if pathlib.Path(path) != projection_path:
            return rows
        return [(where, row) for where, row in rows
                if (row["source"], row["Z"], row["A"], row["locator"], row["quantity"]) !=
                (left.source, str(left.z), str(left.a), left.locator, left.quantity)]

    with monkeypatch.context() as patcher:
        patcher.setattr(md, "read_rows", missing)
        with pytest.raises(md.CellError, match="shell_projection.csv"):
            centroids.centroid_rows(REPO)


def test_t129_representative_uses_unrounded_sigma_max(monkeypatch):
    cells = md.load_cells(CELLS)
    grouped = {}
    for left, right in md.doublet_determinations(cells):
        grouped.setdefault((left.source, left.z, left.a), []).append((left, right))
    key, determinations = next((key, pairs) for key, pairs in grouped.items() if len(pairs) > 1)
    first, second = determinations[:2]
    first_locator, second_locator = first[0].locator, second[0].locator
    changed = tuple(replace(cell, unc_kev=("1.0000000003" if
                                   (cell.source, cell.z, cell.a, cell.locator) ==
                                   (*key, first_locator) and cell.quantity == "K1-L2" else "1"))
                    if (cell.source, cell.z, cell.a, cell.locator) in
                    {(*key, first_locator), (*key, second_locator)} else cell for cell in cells)
    monkeypatch.setattr(md, "load_cells", lambda path: changed)
    rows = [r for r in centroids.centroid_rows(REPO)
            if (r["source"], int(r["Z"]), int(r["A"])) == key and r["cohort"] == "constructed"]
    assert rows[0]["sigma_max_keV"] == rows[1]["sigma_max_keV"]
    assert [r["locator"] for r in rows if r["representative"] == "true"] == [second_locator]


def test_t130_a_changed_ratio_moves_only_the_illustrative_cells(monkeypatch):
    baseline = centroids.centroid_rows(REPO)
    original = centroids.load_intensity_ratios

    def moved(path, cells):
        ratios = original(path, cells)
        key = next(iter(ratios))
        ratios[key] = {**ratios[key], "ratio": str(Decimal(ratios[key]["ratio"]) + Decimal("0.1"))}
        return ratios

    monkeypatch.setattr(centroids, "load_intensity_ratios", moved)
    changed = centroids.centroid_rows(REPO)
    different = [(a, b) for a, b in zip(baseline, changed, strict=True) if a != b]
    assert len(different) == 1
    before, after = different[0]
    columns = {key for key in before if before[key] != after[key]}
    assert columns == {"ratio", "illustrative_c_keV", "illustrative_q_minus_c_keV"}


# --------------------------------------------------------------------------------------------
# T-131 -- each change of a compared energy by cause, and the v1.3.0 comparison kept beside the new one
# --------------------------------------------------------------------------------------------

ATTRIBUTION = REPO / attribution.ATTRIBUTION_RELPATH
HISTORY = REPO / attribution.HISTORY_RELPATH


def _stage_value(out: md.Outputs, row: md.InputRow, stage: str, quantity: str) -> Decimal:
    """A quantity of one stage run in keV, computed here: the stage runs of the `stages` batch, the production
    anchor run last."""
    run = md.recipe_run_id(row, "anchor", True) if stage == "recipe" else md.run_id(row, stage)
    with localcontext() as context:
        context.prec = 50
        if quantity.startswith("line:"):
            value = Decimal(out.lines[run][quantity.split(":")[1]]) / 1000
        else:
            b = md.run_bindings(out, run)
            names = quantity.split("-")
            value = _quantity(b, names[0]) - (_quantity(b, names[1]) if len(names) == 2 else 0)
        return value.quantize(Decimal("1e-18"))


def attribution_problems(rows: list[dict[str, str]]) -> list[str]:
    """Every row of ``rows`` (the committed `change_attribution.csv`) whose cells the committed stage runs do
    not give, or whose changes do not sum to its difference exactly, named."""
    out = _outputs()
    inputs = {row.nuclide: row for row in out.inputs}
    cells = out.cells
    stages = ("shipped", "numerics", "density", "rms", "mass", "recipe")
    expected_keys = []
    for key in md.compared_nuclides(cells):
        lines = []
        for cell in cells:
            if cell.gated and cell.nuclide == key and f"line:{cell.quantity}" not in lines:
                lines.append(f"line:{cell.quantity}")
        doublets = {l2.nuclide for l2, _ in md.doublet_determinations(cells)}
        centroid = set(md.centroid_nuclides(cells)) | doublets
        expected_keys += [(key, q) for q in [*lines, *(["K-e2"] if key in centroid else []), "K", "e2"]]
    problems = []
    if [((int(r["Z"]), int(r["A"])), r["quantity"]) for r in rows] != expected_keys:
        problems.append("the rows are not every compared nuclide's quantities in order")
    for r in rows:
        key = (int(r["Z"]), int(r["A"]))
        values = [_stage_value(out, inputs[key], stage, r["quantity"]) for stage in stages]
        with localcontext() as context:
            context.prec = 50
            want = [values[0], *(after - before for before, after in zip(values, values[1:], strict=False)),
                    values[-1]]
            got = [Decimal(r[column]) for column in attribution.ATTRIBUTION_COLUMNS[3:]]
            if got != want:
                problems.append(f"Z={key[0]} A={key[1]} {r['quantity']}: cells")
            elif got[0] + sum(got[1:-1]) != got[-1]:
                problems.append(f"Z={key[0]} A={key[1]} {r['quantity']}: the changes do not sum to the "
                                "difference")
    return problems


def test_t131_every_attribution_cell_is_rederived_from_the_stage_runs_and_telescopes_exactly():
    rows = _committed_csv(ATTRIBUTION)
    assert tuple(rows[0]) == attribution.ATTRIBUTION_COLUMNS
    assert "delta_density_and_quadrature_keV" in attribution.ATTRIBUTION_COLUMNS
    assert attribution_problems(rows) == []
    assert ATTRIBUTION.read_bytes() == attribution.render_attribution(REPO)
    compared = len(md.compared_nuclides(_outputs().cells))
    print(f"\nattribution rows {len(rows)} over {compared} compared nuclides")


def test_t131_drill_a_committed_value_moved_by_one_unit_of_its_last_digit_is_refused():
    """G-binding's drill on the attribution: one change moved by one unit in its last printed digit."""
    rows = _committed_csv(ATTRIBUTION)
    moved = [dict(row) for row in rows]
    cell = moved[0]["delta_rms_keV"]
    moved[0]["delta_rms_keV"] = str(Decimal(cell) + 2 * _half_unit(cell).copy_sign(1))
    first = moved[0]
    assert attribution_problems(moved) == [f"Z={first['Z']} A={first['A']} {first['quantity']}: cells"]


def test_t131_the_first_stage_is_the_v1_3_0_model_and_the_last_the_shipped_value():
    """G-binding: the `shipped_keV` of every gated line is the model line of release v1.3.0's comparison file;
    the `new_keV` of every gated line is this comparison's model line, of K and e2 the table cell, and of K-e2
    the shell difference of the centroid comparison at its nine decimals."""
    history = {(r["Z"], r["A"], r["quantity"]): r for r in _committed_csv(HISTORY) if r["gated"] == "true"}
    current = {(r["Z"], r["A"], r["quantity"]): r for r in _committed_validation() if r["gated"] == "true"}
    kshell, levels = _table_cells(KSHELL_LAYER1), _table_cells(LEVELS_LAYER1)
    centroid_q = {(r["Z"], r["A"]): r["q_keV"] for r in _committed_rows(centroids.CENTROIDS_RELPATH)
                  if r["q_keV"]}
    checked = 0
    for r in _committed_csv(ATTRIBUTION):
        key = (int(r["Z"]), int(r["A"]))
        if r["quantity"].startswith("line:"):
            line = r["quantity"].split(":")[1]
            assert Decimal(r["shipped_keV"]) == Decimal(history[(r["Z"], r["A"], line)]["model_keV"]), r
            assert Decimal(r["new_keV"]) == Decimal(current[(r["Z"], r["A"], line)]["model_keV"]), r
        elif r["quantity"] in ("K", "e2"):
            cell = kshell[key][0] if r["quantity"] == "K" else levels[key][0]
            assert float(r["new_keV"]) == float(cell), r
        else:
            assert Decimal(r["new_keV"]).quantize(Decimal("1e-9")) == Decimal(centroid_q[(r["Z"], r["A"])]), r
        checked += 1
    assert checked == len(_committed_csv(ATTRIBUTION))


def test_t131_the_history_copy_is_release_v1_3_0s_comparison_file():
    assert attribution.HISTORY_BLOB == "c16d0085317c3ae0e1b2fc45e394fffc9aa9084c"
    assert attribution.history_problems(REPO) == []
    history = _committed_csv(HISTORY)
    gated = [r for r in history if r["gated"] == "true"]
    print(f"\nv1.3.0 comparison: gated {len(gated)} within {sum(r['within'] == 'true' for r in gated)}")


def test_t131_drill_one_flipped_byte_of_the_history_copy_is_refused(tmp_path):
    root = tmp_path
    target = root / attribution.HISTORY_RELPATH
    target.parent.mkdir(parents=True)
    payload = bytearray(HISTORY.read_bytes())
    target.write_bytes(bytes(payload))
    assert attribution.history_problems(root) == []
    payload[100] ^= 0x01
    target.write_bytes(bytes(payload))
    assert attribution.history_problems(root) and "is blob" in attribution.history_problems(root)[0]


def attribution_pins() -> list[tuple[str, str, str, tuple[int, ...], object]]:
    """The pins of the subsection on the change from the `mudirac130` tables, read from the history copy,
    `change_attribution.csv` and the module's settings."""
    path = "DATASET_D3.md"
    history = [r for r in _committed_csv(HISTORY) if r["gated"] == "true"]
    rows = [r for r in _committed_csv(ATTRIBUTION) if r["quantity"].startswith("line:")]

    def largest(column: str) -> dict[str, str]:
        return max(rows, key=lambda r: abs(Decimal(r[column])))

    density, numerics = largest("delta_density_and_quadrature_keV"), largest("delta_numerics_keV")
    where = ("Z", "A", "quantity")
    assert [density[c] for c in where] == [numerics[c] for c in where]
    assert inputs_symbol(int(density["Z"])) == "Pb"
    rest = ("delta_rms_keV", "delta_mass_keV", "delta_eV_keV")
    other = max(abs(Decimal(r[c])) for r in rows for c in rest)
    return [
        ("v1.3.0 rows within tolerance", path, r"placed (\d+) of the \d+ gated rows within tolerance;", (1,),
         sum(r["within"] == "true" for r in history)),
        ("v1.3.0 gated rows", path, r"placed \d+ of the (\d+) gated rows within tolerance;", (1,),
         len(history)),
        ("the decimals of the numerics stage", path, r"printing to (\d+) decimals and the hydrogen-like",
         (1,), md.RECIPE_XR_PRECISION),
        ("the Gauss-Legendre nodes of the numerics stage", path, r"from Gauss–Legendre at (\d+) nodes", (1,),
         md.NUMERICS_STAGE_SETTINGS.n_u),
        ("the trapezoidal steps of the recipe", path, r"the trapezoidal rule at (\d+) steps", (1,),
         md.RECIPE_SETTINGS.n_u),
        ("the shell nodes of the recipe", path, r"with (\d+) Gauss–Legendre nodes per panel", (1,),
         md.RECIPE_SETTINGS.n_s),
        ("the largest density and quadrature change on a gated line", path,
         r"the largest changes are ([0-9.]+) keV from the density", (1,),
         density["delta_density_and_quadrature_keV"].lstrip("-")),
        ("the largest numerics change on a gated line", path,
         r"keV from the density and quadrature and ([0-9.]+) keV from the numerics", (1,),
         numerics["delta_numerics_keV"].lstrip("-")),
        ("their mass number", path, r"both on the Pb-(\d+) `", (1,), density["A"]),
        ("their line", path, r"both on the Pb-\d+ `([K-Z]\d+-[K-Z]\d+)` line", (1,), density["quantity"][5:]),
        ("the largest change of any other step on a gated line", path, r"by more than ([0-9.]+) keV\.", (1,),
         attribution._text(other)),
    ]


def inputs_symbol(z: int) -> str:
    return next(row.symbol for row in _outputs().inputs if row.z == z)
