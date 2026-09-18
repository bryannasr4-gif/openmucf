"""The D3 energy tables of the ``mudirac130`` profile: their MuDirac inputs and printed outputs, the
derivation of the two tables from those printed strings, and the comparison with measured energies.

Every count here is derived at run time from the committed files; none is written down.
"""

from __future__ import annotations

import pathlib
import re

import pytest

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
# T-106 -- the measured transition energies (loaders)
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


def test_t103_the_run_table_lists_exactly_the_rendered_inputs():
    """The committed run table holds one row per input the committed kinds render, in their order,
    and every row names a rendered run."""
    rows = md.load_inputs(INPUTS)
    cells, _ = md.load_validation(CELLS, ORIGIN)
    rendered = md.expected_runs(rows, set(md.gated_nuclides(cells)))
    runs = md.load_runs(REPO / md.RUNS_RELPATH)
    assert [(r.run, r.z, r.a, r.kind) for r in runs.values()] == rendered
    assert len(runs) == len(rendered)
    print(f"\nrun table rows {len(runs)} = rendered inputs {len(rendered)}")


# --------------------------------------------------------------------------------------------
# T-104 -- the printed outputs and the keep rule
# --------------------------------------------------------------------------------------------

RUNS = REPO / md.RUNS_RELPATH
STATES = REPO / md.STATES_RELPATH
LINES = REPO / md.LINES_RELPATH
NMAX = REPO / md.NMAX_RELPATH
_R1 = "H1_base,1,1,base,0,0"
_S1 = "H1_base,1,1,base,K1,1,0,1,-2530.37,9.49672e+07"
_L1 = "H1_base,1,1,base,K1-L2,1898.185553,116589628776.002548"
_N1 = "1,1,6,P10,-70.2398,-70.2398"

OUTPUT_DRILLS = [
    (RUNS, md.load_runs, "a carriage return", lambda t: t.replace(NL, "\r" + NL, 1),
     md.CarriageReturnError, "contains CR"),
    (RUNS, md.load_runs, "a renamed column", lambda t: _replace_once(t, "rc,err_bytes", "rc,err"),
     md.HeaderError, "header is"),
    (RUNS, md.load_runs, "an unknown kind", lambda t: _replace_once(t, NL + _R1, NL + "H1_base,1,1,bass,0,0"),
     md.CellError, "kind 'bass'"),
    (RUNS, md.load_runs, "a run named for another kind",
     lambda t: _replace_once(t, NL + _R1, NL + "H1_rsig,1,1,base,0,0"), md.CellError, "does not name"),
    (RUNS, md.load_runs, "an error size that is not a count",
     lambda t: _replace_once(t, NL + _R1, NL + "H1_base,1,1,base,0,missing"),
     md.CellError, "must be integers"),
    (RUNS, md.load_runs, "a duplicated run", lambda t: _repeat_line(t, 1),
     md.DuplicateKeyError, "duplicate run"),
    (RUNS, md.load_runs, "rows out of order", lambda t: _swap_lines(t, 1),
     md.OrderError, "ordered by (Z, A, kind)"),
    (RUNS, md.load_runs, "no rows", lambda t: _lines(t)[0] + NL, md.EmptyError, "carries no rows"),
    (STATES, md.load_states, "a state of a kind whose states are not committed",
     lambda t: _replace_once(t, NL + _S1, NL + _S1.replace(",base,", ",ideal6,")),
     md.CellError, "kind 'ideal6'"),
    (STATES, md.load_states, "a state that is not an orbit",
     lambda t: _replace_once(t, NL + _S1, NL + _S1.replace(",K1,", ",1s,")), md.CellError, "IUPAC orbit"),
    (STATES, md.load_states, "an energy in another form",
     lambda t: _replace_once(t, NL + _S1, NL + _S1.replace(",-2530.37,", ",-2530.37 eV,")),
     md.CellError, "printed number"),
    (STATES, md.load_states, "a duplicated state", lambda t: _repeat_line(t, 1),
     md.DuplicateKeyError, "duplicate state"),
    (STATES, md.load_states, "rows out of order", lambda t: _swap_lines(t, 1),
     md.OrderError, "ordered by (Z, A, kind, orbit)"),
    (LINES, md.load_lines, "a line at other decimals",
     lambda t: _replace_once(t, NL + _L1, NL + _L1.replace(",1898.185553,", ",1898.18555,")),
     md.CellError, "six decimals"),
    (LINES, md.load_lines, "a line that is not orbit to orbit",
     lambda t: _replace_once(t, NL + _L1, NL + _L1.replace(",K1-L2,", ",Ka1,")), md.CellError, "orbit-orbit"),
    (LINES, md.load_lines, "a duplicated line", lambda t: _repeat_line(t, 1),
     md.DuplicateKeyError, "duplicate line"),
    (NMAX, md.load_nmax, "a shell outside the checked range",
     lambda t: _replace_once(t, NL + _N1, NL + _N1.replace("1,1,6,P10,", "1,1,5,P10,")),
     md.CellError, "not a checked circular state"),
    (NMAX, md.load_nmax, "an energy in another form",
     lambda t: _replace_once(t, NL + _N1, NL + "1,1,6,P10,-70.2398,n/a"), md.CellError, "printed number"),
    (NMAX, md.load_nmax, "a duplicated row", lambda t: _repeat_line(t, 1),
     md.DuplicateKeyError, "duplicate row"),
    (NMAX, md.load_nmax, "rows out of order", lambda t: _swap_lines(t, 1),
     md.OrderError, "ordered by (Z, A, n, orbit)"),
]


@pytest.mark.parametrize("source, loader, label, mutate, error, message", OUTPUT_DRILLS,
                         ids=[f"{d[0].stem}-{d[2]}" for d in OUTPUT_DRILLS])
def test_t104_drill_each_outputs_rule_refuses_its_fixture(
    tmp_path, source, loader, label, mutate, error, message
):
    _drill(tmp_path, source, mutate, error, message, loader)


def _data_copy(tmp_path) -> pathlib.Path:
    """A copy of the committed D3 files under ``tmp_path``, at their relative paths."""
    target = tmp_path / md.D3_RELDIR
    target.mkdir(parents=True)
    for path in D3DIR.glob("*.csv"):
        (target / path.name).write_bytes(path.read_bytes())
    return tmp_path


def test_t104_drill_a_run_table_missing_a_rendered_run_is_refused(tmp_path):
    root = _data_copy(tmp_path)
    md.load_outputs(root)
    runs = root / md.RUNS_RELPATH
    lines = _lines(runs.read_bytes().decode("ascii"))
    runs.write_bytes(NL.join(lines[:1] + lines[2:]).encode("ascii"))
    with pytest.raises(md.CellError, match="does not list exactly the implied runs"):
        md.load_outputs(root)


def test_t104_the_derivation_holds_on_the_base_and_rsig_runs_of_every_kept_member():
    out = md.load_outputs(REPO)
    dropped = md.drop_reasons(out)
    checked = 0
    for row in out.inputs:
        if row.nuclide in dropped:
            continue
        for kind in ("base", "rsig"):
            run = md.run_id(row, kind)
            bindings = md.derive_bindings(run, out.headers[run], out.lines[run])
            assert set(bindings) == set(md.chain_states())
            checked += 1
    print(f"\nderivation checked on {checked} runs of {len(out.inputs) - len(dropped)} kept members")


def test_t104_drill_the_derivation_refuses_a_moved_line_a_moved_header_and_a_missing_state():
    out = md.load_outputs(REPO)
    run = md.run_id(out.inputs[0], "base")
    headers, lines = dict(out.headers[run]), dict(out.lines[run])
    md.derive_bindings(run, headers, lines)
    moved = dict(lines)
    moved["K1-L3"] = format(float(lines["K1-L3"]) + 1e-5, ".6f")
    with pytest.raises(md.DerivationError, match="cross line"):
        md.derive_bindings(run, headers, moved)
    shifted = dict(headers)
    shifted["K1"] = format(float(headers["K1"]) - 0.1, ".2f")
    with pytest.raises(md.DerivationError, match="from its printed header"):
        md.derive_bindings(run, shifted, lines)
    missing = {state: text for state, text in headers.items() if state != "L2"}
    with pytest.raises(md.DerivationError, match="no printed header for L2"):
        md.derive_bindings(run, missing, lines)


def test_t104_the_keep_rule_drops_exactly_the_member_whose_default_fermi_c_is_not_real():
    """Re-derived from the committed run table, comparison file and printed outputs: one member is
    dropped, for the reason its sphere radius gives no real default Fermi parameter c, and no kept
    member of the square-root range lies below that radius."""
    out = md.load_outputs(REPO)
    dropped = md.drop_reasons(out)
    below = [row for row in out.inputs if md.fermi2_c_not_real(row)]
    threshold = md.fermi2_c_threshold(md.FERMI_T)
    assert [row.nuclide for row in below] == list(dropped)
    assert set(dropped.values()) == {md.DROP_FERMI2_C}
    assert len(dropped) == 1
    for row in below:
        assert md.keep_failures(out, row)
        print(f"\ndropped: Z={row.z} A={row.a} sphere radius {row.radius_fm} fm below {threshold!r} fm; "
              f"{md.keep_failures(out, row)}")
    print(f"members {len(out.inputs)} kept {len(out.inputs) - len(dropped)} dropped {len(dropped)}")


def test_t104_drill_a_failed_run_of_another_member_is_dropped_with_its_clause(tmp_path):
    root = _data_copy(tmp_path)
    runs = root / md.RUNS_RELPATH
    runs.write_bytes(_replace_once(runs.read_bytes().decode("ascii"), NL + _R1 + NL,
                                   NL + "H1_base,1,1,base,0,5" + NL).encode("ascii"))
    dropped = md.drop_reasons(md.load_outputs(root))
    assert dropped[(1, 1)] == "base rc=0 err_bytes=5"
    assert len(dropped) == 2
