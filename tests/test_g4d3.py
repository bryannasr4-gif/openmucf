"""The D3 energy tables of the ``mudirac130`` profile: their MuDirac inputs and printed outputs, the
derivation of the two tables from those printed strings, and the comparison with measured energies.

Every count here is derived at run time from the committed files; none is written down.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import re
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest
import test_g4parity as parity

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


def test_t104_drill_a_header_just_past_its_bound_is_refused_and_one_just_inside_passes():
    """K1's header moved to exactly `header_bound` plus half a printed line unit from its derived
    value is refused; moved to `header_bound` minus that half unit, the derivation holds. Both
    moved headers keep the original's magnitude, so the bound they are held to is the same."""
    out = md.load_outputs(REPO)
    dropped = md.drop_reasons(out)
    row = next(row for row in out.inputs if row.nuclide not in dropped)
    run = md.run_id(row, "base")
    headers, lines = dict(out.headers[run]), dict(out.lines[run])
    derived = md.derive_bindings(run, headers, lines)
    anchor = headers[md.orbit(md.N_MAX, True)]
    bound = md.header_bound(headers["K1"], anchor)
    # The bound is no looser than its docstring reads off the printed strings: half a printed unit
    # of the header and of the anchor, plus half a printed line unit per line of a chain it allows.
    stated = (_half_unit(headers["K1"]) + _half_unit(anchor)
              + 2 * md.N_MAX * _line_half_unit(lines))
    assert bound <= stated, (bound, stated)
    with localcontext() as context:
        context.prec = 50
        outside = format(-(derived["K1"] + bound + md.LINE_HALF_UNIT), "f")
        inside = format(-(derived["K1"] + bound - md.LINE_HALF_UNIT), "f")
    for text in (outside, inside):
        assert Decimal(text).adjusted() == Decimal(headers["K1"]).adjusted()
        assert md.header_bound(text, anchor) == bound
    with pytest.raises(md.DerivationError, match="derived K1 lies"):
        md.derive_bindings(run, {**headers, "K1": outside}, lines)
    md.derive_bindings(run, {**headers, "K1": inside}, lines)


def test_t104_the_dropped_set_is_the_two_computed_classes_and_the_shipped_tables_hold_neither():
    """Re-derived from the committed inputs, run table and printed outputs: the members whose default
    Fermi parameter c is set by the mass number alone and the members whose sphere radius gives no
    real default c are dropped, each with its own reason, and no other member is; no row of either
    shipped table has a mass number below the square-root range."""
    out = md.load_outputs(REPO)
    dropped = md.drop_reasons(out)
    expected = {}
    for row in out.inputs:
        if md.fermi2_c_from_a(row):
            expected[row.nuclide] = md.DROP_FERMI2_FROM_A
        elif md.fermi2_c_not_real(row):
            expected[row.nuclide] = md.DROP_FERMI2_C
    assert dropped == expected
    light = [
        (layer1.name, key) for layer1 in (KSHELL_LAYER1, LEVELS_LAYER1)
        for key in _records(spec.parse(layer1.read_bytes().decode("ascii")))
        if 0 < key[1] < md.FERMI2_SQRT_FROM_A
    ]
    assert light == []
    below = [row for row in out.inputs if md.fermi2_c_not_real(row)]
    threshold = md.fermi2_c_threshold(md.FERMI_T)
    for row in below:
        assert md.keep_failures(out, row)
        print(f"\ndropped: Z={row.z} A={row.a} sphere radius {row.radius_fm} fm below {threshold!r} fm; "
              f"{md.keep_failures(out, row)}")
    print(f"members {len(out.inputs)} kept {len(out.inputs) - len(dropped)} dropped {len(dropped)}")


def test_t104_drill_a_failed_run_of_another_member_is_dropped_with_its_clause(tmp_path):
    out = md.load_outputs(REPO)
    before = md.drop_reasons(out)
    row = next(row for row in out.inputs if row.nuclide not in before)
    line = next(line for line in _lines(RUNS.read_bytes().decode("ascii"))
                if line.startswith(md.run_id(row, "base") + ","))
    root = _data_copy(tmp_path)
    runs = root / md.RUNS_RELPATH
    runs.write_bytes(_replace_once(runs.read_bytes().decode("ascii"), NL + line + NL,
                                   NL + line[: line.rindex(",")] + ",5" + NL).encode("ascii"))
    dropped = md.drop_reasons(md.load_outputs(root))
    assert dropped == {**before, row.nuclide: "base rc=0 err_bytes=5"}


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


def _header_binding(headers: dict[str, str], state: str) -> Decimal:
    return -Decimal(headers[state])


def _half_unit(text: str) -> Decimal:
    """Half a unit of the last digit ``text`` prints, read off the printed string itself."""
    return Decimal(5) * Decimal(10) ** (Decimal(text).as_tuple().exponent - 1)


def _line_half_unit(lines: dict[str, str]) -> Decimal:
    """The largest half unit among a run's printed line energies."""
    return max(_half_unit(text) for text in lines.values())


#: The most printed lines the derivation sums to reach any circular state from the anchor: every
#: upper-chain line down to K1, then every lower-chain line out again.
_MOST_LINES = 2 * (md.N_MAX - 1)


def _header_bound(headers: dict[str, str], lines: dict[str, str], state: str) -> Decimal:
    """How far a derived binding energy may lie from its printed header, written here from the
    printed strings: half a printed unit of the header and of the anchor, plus half a printed line
    unit for each line summed."""
    anchor = headers[md.orbit(md.N_MAX, True)]
    return _half_unit(headers[state]) + _half_unit(anchor) + _MOST_LINES * _line_half_unit(lines)


def _from_headers(headers: dict[str, str], lines: dict[str, str]) -> list[tuple[Decimal, Decimal]]:
    """``[(quantity, bound)]`` in keV for K then e2 .. e<N_MAX>, from the printed state headers alone:
    each quantity with the largest distance the derivation may put between it and this value."""
    with localcontext() as context:
        context.prec = 50
        out = [(_header_binding(headers, "K1") / 1000, _header_bound(headers, lines, "K1") / 1000)]
        for n in range(2, md.N_MAX + 1):
            ell = n - 1
            lower, upper = md.orbit(n, False), md.orbit(n, True)
            mean = (2 * ell * _header_binding(headers, lower)
                    + (2 * ell + 2) * _header_binding(headers, upper)) / (4 * ell + 2)
            bound = max(_header_bound(headers, lines, lower), _header_bound(headers, lines, upper))
            out.append((mean / 1000, bound / 1000))
    return out


def _floor(lines: dict[str, str]) -> Decimal:
    """The least uncertainty a cell may carry, in keV, written here from the printed lines: a
    change between two runs of a quantity that sums at most `_MOST_LINES` printed lines in each."""
    return 2 * _MOST_LINES * _line_half_unit(lines) / 1000


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
        out = [(rel["K1"][0] / 1000, rel["K1"][1])]
        for n in range(2, md.N_MAX + 1):
            ell = n - 1
            (low, low_count), (high, high_count) = rel[md.orbit(n, False)], rel[md.orbit(n, True)]
            mean = (2 * ell * low + (2 * ell + 2) * high) / (4 * ell + 2)
            out.append((mean / 1000, max(low_count, high_count)))
    return out


def _within(value: float, expected: Decimal, bound: Decimal) -> bool:
    """``value`` (as shipped, a double) within ``bound`` of ``expected``, allowing the double's own
    rounding of a decimal quotient."""
    shipped = Decimal(repr(value))
    return abs(shipped - expected) <= bound + abs(expected) * Decimal("1e-15")


def test_t105_every_value_and_unc_lies_within_the_header_precision_of_the_state_headers():
    out = md.load_outputs(REPO)
    inputs = {row.nuclide: row for row in out.inputs}
    kshell, kshell_doc = _shipped(KSHELL_LAYER1, KSHELL_LAYER2)
    levels, levels_doc = _shipped(LEVELS_LAYER1, LEVELS_LAYER2)
    k_records, level_records = _records(kshell), _records(levels)
    carried = _carried(kshell_doc)
    assert carried == _carried(levels_doc)
    checked = 0
    for (z, a), (value, unc) in k_records.items():
        source = inputs[(z, carried[(z, a)])]
        base_run, rsig_run = md.run_id(source, "base"), md.run_id(source, "rsig")
        base = _from_headers(out.headers[base_run], out.lines[base_run])
        moved = _from_headers(out.headers[rsig_run], out.lines[rsig_run])
        floor = max(_floor(out.lines[base_run]), _floor(out.lines[rsig_run]))
        shipped = [value, *level_records[(z, a)][: md.N_MAX - 1]]
        shipped_unc = [unc, *level_records[(z, a)][md.N_MAX - 1 :]]
        assert len(shipped) == len(shipped_unc) == len(base) == md.N_MAX
        for got, got_unc, (quantity, bound), (quantity_moved, bound_moved) in zip(
            shipped, shipped_unc, base, moved, strict=True
        ):
            assert _within(got, quantity, bound), ((z, a), got, quantity, bound)
            assert _within(got_unc, abs(quantity_moved - quantity), bound + bound_moved + floor), (
                (z, a), got_unc
            )
            checked += 1
    print(f"\nvalues and uncs checked against the headers: {checked} of {len(k_records)} rows")


def test_t105_every_unc_is_floored_and_agrees_with_an_independent_route_within_its_rounding():
    """Every shipped `unc` and `u<n>` re-derived by the other route, anchor-relative: each cell is at
    least this test's floor, and lies within that floor plus the two routes' rounding bound of the
    other route's change (floored the same way). The largest difference and the floor are printed:
    the difference is the route noise, and it must lie below the floor."""
    out = md.load_outputs(REPO)
    inputs = {row.nuclide: row for row in out.inputs}
    kshell, kshell_doc = _shipped(KSHELL_LAYER1, KSHELL_LAYER2)
    levels, _ = _shipped(LEVELS_LAYER1, LEVELS_LAYER2)
    k_records, level_records = _records(kshell), _records(levels)
    carried = _carried(kshell_doc)
    largest, floors, checked = Decimal(0), set(), 0
    for (z, a), (_value, unc) in k_records.items():
        source = inputs[(z, carried[(z, a)])]
        base_lines = out.lines[md.run_id(source, "base")]
        rsig_lines = out.lines[md.run_id(source, "rsig")]
        half = max(_line_half_unit(base_lines), _line_half_unit(rsig_lines))
        floor = max(_floor(base_lines), _floor(rsig_lines))
        floors.add(floor)
        shipped_unc = [unc, *level_records[(z, a)][md.N_MAX - 1 :]]
        base, moved = _route_quantities(base_lines), _route_quantities(rsig_lines)
        for got, (quantity, count), (quantity_moved, count_moved) in zip(
            shipped_unc, base, moved, strict=True
        ):
            shipped = Decimal(repr(got))
            assert shipped >= floor, ((z, a), got, floor)
            expected = max(abs(quantity_moved - quantity), floor)
            route_bound = (2 * _MOST_LINES + count + count_moved) * half / 1000
            difference = abs(shipped - expected)
            assert difference <= floor + route_bound + expected * Decimal("1e-15"), (
                (z, a), got, expected, route_bound
            )
            largest = max(largest, difference)
            checked += 1
    (floor,) = floors
    print(f"\nuncertainty cells checked on the other route: {checked}; largest route difference "
          f"{largest} keV; floor {floor} keV")
    assert largest < floor, (largest, floor)


def test_t105_drill_a_moved_rsig_anchor_header_leaves_the_uncertainties_unchanged():
    """The `rsig` anchor header of a member whose anchor prints the same in both runs, moved by one
    printed unit: the absolute derivation's K moves with it, and the shipped uncertainties do not."""
    out = md.load_outputs(REPO)
    dropped = md.drop_reasons(out)
    kshell, _ = _shipped(KSHELL_LAYER1, KSHELL_LAYER2)
    levels, _ = _shipped(LEVELS_LAYER1, LEVELS_LAYER2)
    k_records, level_records = _records(kshell), _records(levels)
    anchor = md.orbit(md.N_MAX, True)
    for row in out.inputs:
        if row.nuclide in dropped:
            continue
        base_run, rsig_run = md.run_id(row, "base"), md.run_id(row, "rsig")
        headers = out.headers[rsig_run]
        if out.headers[base_run][anchor] != headers[anchor]:
            continue
        unit = 2 * _half_unit(headers[anchor])
        for step in (unit, -unit):
            moved_headers = dict(headers)
            printed = Decimal(headers[anchor])
            moved_headers[anchor] = str((printed + step).quantize(printed))
            try:
                moved = md.derive_bindings(rsig_run, moved_headers, out.lines[rsig_run])
            except md.DerivationError:
                continue
            base = md.derive_bindings(base_run, out.headers[base_run], out.lines[base_run])
            original = md.derive_bindings(rsig_run, headers, out.lines[rsig_run])
            assert md.quantities(moved)[0] != md.quantities(original)[0]
            unc, level_uncs = md.uncertainties(base, moved)
            shipped = (k_records[row.nuclide][1], *level_records[row.nuclide][md.N_MAX - 1 :])
            assert tuple(float(u) for u in (unc, *level_uncs)) == shipped, row.nuclide
            print(f"\nZ={row.z} A={row.a}: rsig anchor {headers[anchor]} -> {moved_headers[anchor]}; "
                  "uncertainties unchanged")
            return
    pytest.fail("no member with an equal anchor header derives with its rsig anchor moved")


def test_t105_natural_rows_equal_their_carried_isotope_and_both_tables_share_one_key_set():
    """V-16's Python mirror, the (Z, 0) copy rule, and the kept set: the key set is every kept
    member plus a (Z, 0) row for each Z whose most abundant isotope is kept."""
    out = md.load_outputs(REPO)
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
    out = md.load_outputs(REPO)
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
            rendered = md.render_input(source, "base", md.extra_lines(cells, source.nuclide))
            assert row.source_bibkey == md.BIBKEY
            assert row.unc_type == "model"
            expected = f"MuDirac {md.MUDIRAC_VERSION} input: " + "; ".join(rendered.splitlines())
            assert row.conditions == expected
            assert row.conditions.count("optimise_fermi_parameters: FALSE") == 1
            assert (row.single_source, row.needs_verification, row.recommendation) == (False, False, "")
            assert row.evaluation_id == row.source_library == md.PROFILE
            assert row.isotope_resolved is (a != 0)
            assert md.run_id(source, "base") in row.source_locator
            assert md.run_id(source, "rsig") in row.source_locator
            assert md.run_id(source, "rsig") in row.evaluation_method
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
    files = sorted(D3DIR.iterdir())
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


def test_t106_every_gated_quantity_is_a_line_its_base_run_printed():
    out = md.load_outputs(REPO)
    cells, _ = md.load_validation(CELLS, ORIGIN)
    inputs = {row.nuclide: row for row in out.inputs}
    for cell in cells:
        if cell.gated:
            for kind in ("base", "rsig", "r101"):
                assert cell.quantity in out.lines[md.run_id(inputs[cell.nuclide], kind)], (cell, kind)


def test_t106_the_comparison_file_equals_an_arithmetic_rederivation():
    """Every column recomputed here from the transcription and the printed lines, with the
    tolerance three printed standard deviations and no model term; the two Geant4 columns from the
    committed cascade levels, the photon between the line's two shells taken in doubles."""
    out = md.load_outputs(REPO)
    geant4 = md.load_geant4_levels(GEANT4_LEVELS)
    cells, origins = md.load_validation(CELLS, ORIGIN)
    inputs = {row.nuclide: row for row in out.inputs}
    committed = _committed_validation()
    assert len(committed) == len(cells)
    assert list(committed[0]) == list(md.VALIDATION_COLUMNS)
    assert md.TOL_FACTOR == 3 and md.SIGMA_CALC == 0

    def printed(cell, kind):
        return Decimal(out.lines[md.run_id(inputs[cell.nuclide], kind)][cell.quantity]) / 1000

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
        model = printed(cell, "base")
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
    out = md.load_outputs(REPO)
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


def document_pins() -> list[tuple[str, str, str, tuple[int, ...], object]]:
    """``(what, path, pattern, groups, expected)`` for every number `DATASET_D3.md` states: one row
    per comparison-table row (every numeric token a group, its value the generated block's), and
    one per count or constant of the prose, `expected` read from the module or the committed files.
    Patterns match the whitespace-collapsed document."""
    out = md.load_outputs(REPO)
    rows, dropped = md.table_rows(out)
    kept = sum(1 for row in rows if row.a != 0)
    inputs = {row.nuclide: row for row in out.inputs}
    zs = sorted({row.z for row in out.inputs})
    validation = _committed_validation()
    gated = [row for row in validation if row["gated"] == "true"]
    within = [row for row in gated if row["within"] == "true"]
    weak = [row for row in gated if row["label"] == md.WEAKLY_SENSITIVE]
    path = "DATASET_D3.md"
    pins: list[tuple[str, str, str, tuple[int, ...], object]] = [
        ("the generator's version", path, r"derived from the output of MuDirac (\d+\.\d+\.\d+)", (1,),
         md.MUDIRAC_VERSION),
        ("the highest shell of the chain", path, r"The chain ends at shell (\d+),", (1,), md.N_MAX),
        ("members of the input set", path, r"Of the (\d+) members of the input set", (1,), len(out.inputs)),
        ("kept members", path, r"members of the input set, (\d+) are kept", (1,), kept),
        ("natural-composition rows", path, r"the tables carry (\d+) natural-composition rows", (1,),
         len(rows) - kept),
        ("the lowest Z of the input set", path, r"with Z from (\d+) through \d+", (1,), zs[0]),
        ("the highest Z of the input set", path, r"with Z from \d+ through (\d+)", (1,), zs[-1]),
        ("the skin thickness of every run", path, r"`fermi_t: (\d+\.\d+)`", (1,), md.FERMI_T),
        ("the lowest checked shell", path, r"circular state of shells (\d+) through \d+", (1,),
         md.IDEAL_FROM),
        ("the highest checked shell", path, r"circular state of shells \d+ through (\d+)", (1,), md.N_MAX),
        ("the hydrogen-like bound in percent", path, r"lies within (\d+) % of the same state", (1,),
         int(md.NMAX_BOUND * 100)),
        ("the uncertainty floor", path, r"No `unc` or `u<n>` cell is below (\d+\.\d+) keV", (1,),
         format(md.UNC_FLOOR_KEV, "f")),
        ("the dropped member", path, r"The generator drops `([A-Z][a-z]?\d+)`", (1,),
         "".join(md.run_id(inputs[key], "base")[: -len("_base")] for key, reason in dropped.items()
                 if reason == md.DROP_FERMI2_C)),
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
    ]
    table = md.render_validation_table(validation).splitlines()[2:]
    for number, line in enumerate(table, start=1):
        pattern, groups = _row_pattern(line)
        pins.append((f"comparison table row {number}", path, pattern, groups, None))
    pins.extend(contract_pins())
    pins.extend(centroid_pins())
    return pins


def _committed_rows(relpath: str) -> list[dict[str, str]]:
    text = (REPO / relpath).read_bytes().decode("ascii")
    assert "\r" not in text
    return list(csv.DictReader(io.StringIO(text)))


def contract_pins() -> list[tuple[str, str, str, tuple[int, ...], object]]:
    """The pins of the section on the energy the cascade receives: every number it states, read
    from `shell_projection.csv`, `incompatible_groups.csv` and `radius_lineage.csv`,
    and one pin per row of the generated groups table."""
    from openmucf.g4 import d3_contract as d3c

    path = "DATASET_D3.md"
    projection = _committed_rows(d3c.PROJECTION_RELPATH)
    groups = _committed_rows(d3c.GROUPS_RELPATH)
    lineage = d3c.load_radius_lineage(REPO / d3c.LINEAGE_RELPATH)
    assert all(row.dependency_state == d3c.UNKNOWN for row in lineage)
    widest = max(groups, key=lambda g: Decimal(g["gap_keV"]))
    pb_line = next(r for r in projection
                   if (r["Z"], r["A"], r["quantity"]) == (widest["Z"], widest["A"], "K1-L3"))
    with localcontext() as context:
        context.prec = md._PRECISION
        pb_shift = sum((Decimal(s) for s in pb_line["solver_numeric_shifts_keV"].split(";")), Decimal(0))
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
        ("the refined line's mass number", path, r"moves the Pb-(\d+) `K1-L3` line by", (1,), pb_line["A"]),
        ("the refined line's shift", path, r"`K1-L3` line by ([0-9.]+) keV against a printed", (1,),
         format(pb_shift, "f")),
        ("the refined line's printed uncertainty", path,
         r"against a printed uncertainty of ([0-9.]+) keV\.", (1,), pb_line["unc_keV"]),
    ]
    for number, line in enumerate(d3c.render_groups_table(groups).splitlines()[2:], start=1):
        pattern, row_groups = _row_pattern(line)
        pins.append((f"groups table row {number}", path, pattern, row_groups, None))
    return pins


def centroid_pins() -> list[tuple[str, str, str, tuple[int, ...], object]]:
    from openmucf.g4 import d3_centroids as centroids

    rows = _committed_rows(centroids.CENTROIDS_RELPATH)
    margins = _committed_rows(centroids.MARGINS_RELPATH)
    ratios = _committed_rows(centroids.RATIOS_RELPATH)
    compared = [row for row in rows if row["cohort"] != "excluded"]
    above = sum(abs(Decimal(row["dnum_keV"])) >
                Decimal(row["sigma_keV"] or row["sigma_max_keV"]) / 10 for row in compared)
    largest = max(compared, key=lambda row: abs(Decimal(row["dnum_keV"])))
    nuclides = {(row["Z"], row["A"]) for row in compared}
    uncertified = {(row["Z"], row["A"]) for row in compared if row["u_certified"] == "none"}
    assert len(ratios) == 1
    path = "DATASET_D3.md"
    pins: list[tuple[str, str, str, tuple[int, ...], object]] = [
        ("centroid shift count", path, r"on (\d+) of the \d+ rows and reaches", (1,), above),
        ("centroid compared rows", path, r"on \d+ of the (\d+) rows and reaches", (1,), len(compared)),
        ("largest centroid shift", path, r"reaches ([-0-9.]+) keV for Pb-", (1,),
         largest["dnum_keV"]),
        ("largest shift mass", path, r"keV for Pb-(\d+), and", (1,), largest["A"]),
        ("uncertified nuclides", path, r"is `none` for (\d+) of the \d+ nuclides", (1,),
         len(uncertified)),
        ("compared nuclides", path, r"is `none` for \d+ of the (\d+) nuclides", (1,),
         len(nuclides)),
        ("illustrative ratio mass", path, r"For Pb-(\d+) the row also lists", (1,), ratios[0]["A"]),
    ]
    for name, block in (("centroids", centroids.render_centroids_table(rows)),
                        ("margins", centroids.render_margins_table(margins))):
        for number, line in enumerate(block.splitlines()[2:], start=1):
            pattern, groups = _row_pattern(line)
            pins.append((f"{name} table row {number}", path, pattern, groups, None))
    return pins


def test_t107_the_centroid_tables_are_generated_blocks():
    from openmucf.g4 import d3_centroids as centroids

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
    from openmucf.g4 import d3_contract as d3c

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
    kshell_table = (D3DIR / "d3_kshell.mudirac130.g4dat").read_text("ascii")
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
# T-121 -- the runs made beside the committed ones: how their inputs are rendered
# --------------------------------------------------------------------------------------------


def _kept() -> tuple[md.Outputs, list[md.InputRow]]:
    out = md.load_outputs(REPO)
    return out, md.kept_members(out)


def test_t121_the_level_zero_input_is_the_base_input_plus_the_three_defaults_each_set_once():
    """MuDirac 1.3.0 documents the three keywords' defaults in its lib/config.cpp: energy_tol 1e-7
    (line 42), loggrid_step 0.005 (line 46), uehling_steps 100 (line 62); the base renderer emits
    none of them, so the level-0 input is the base input with each set once at its default."""
    _out, kept = _kept()
    cells, _ = md.load_validation(CELLS, ORIGIN)
    defaults = "loggrid_step: 0.005\nuehling_steps: 100\nenergy_tol: 1e-07\n"
    for row in kept:
        extra = md.extra_lines(cells, row.nuclide)
        base = md.render_input(row, "base", extra)
        assert not any(f"{key}:" in base for key in md.NUMERICS_KEYS)
        text = md.render_numerics_input(row, 0, extra)
        assert text == base + defaults
        keys = [line.split(":")[0] for line in text.splitlines()]
        assert [keys.count(key) for key in md.NUMERICS_KEYS] == [1, 1, 1]
    print(f"\nlevel-0 inputs rendered for {len(kept)} kept members")


def test_t121_each_level_halves_the_grid_step_doubles_the_uehling_steps_and_tightens_the_tolerance_tenfold():
    def values(level: int) -> tuple[float, int, float]:
        step, steps, tol = (line.split(": ")[1] for line in md.numerics_settings(level))
        return float(step), int(steps), float(tol)

    for level in (*md.NUMERICS_LEVELS_ALL, *md.NUMERICS_LEVELS_DEEP)[1:]:
        step, steps, tol = values(level)
        step0, steps0, tol0 = values(level - 1)
        assert step == step0 / 2 and steps == 2 * steps0
        assert abs(tol / tol0 - 0.1) < 1e-12


def test_t121_render_numerics_input_refuses_to_set_a_key_the_base_input_already_sets(monkeypatch):
    _out, kept = _kept()
    row = kept[0]
    base = md.render_input
    monkeypatch.setattr(md, "render_input", lambda r, k, e: base(r, k, e) + "energy_tol: 1\n")
    with pytest.raises(ValueError, match="already sets energy_tol"):
        md.render_numerics_input(row, 0, [])


def test_t121_the_rminus_radius_moves_the_rms_radius_down_by_its_uncertainty_and_nothing_else_moves():
    _out, kept = _kept()
    cells, _ = md.load_validation(CELLS, ORIGIN)
    for row in kept:
        rms = float(row.radius_fm) / md.SPHERE_FACTOR
        sigma = float(row.sigma_rms_fm)
        minus, plus = (float(md.run_radius(row, kind)) / md.SPHERE_FACTOR for kind in ("rminus", "rsig"))
        assert abs((rms - minus) - sigma) <= 1e-12 * rms and abs((plus - rms) - sigma) <= 1e-12 * rms
        extra = md.extra_lines(cells, row.nuclide)
        base, moved = (md.render_input(row, kind, extra).splitlines() for kind in ("base", "rminus"))
        differing = [(b, m) for b, m in zip(base, moved, strict=True) if b != m]
        assert differing == [(f"radius: {row.radius_fm}", f"radius: {md.run_radius(row, 'rminus')}")]
        assert md.numerics_run_id(row, "rminus", 0).endswith(f"{row.a}_rminus")
        assert md.numerics_run_id(row, "grid", 2).endswith(f"{row.a}_grid2")


def test_t121_the_domain_rule_refuses_a_nonpositive_rms_and_a_moved_radius_whose_fermi_c_is_not_real():
    _out, kept = _kept()
    inputs = {row.nuclide: row for row in kept}
    lead = inputs[(82, 208)]
    assert md.rminus_in_domain(lead)
    assert not md.rminus_in_domain(replace(lead, sigma_rms_fm=lead.rms_fm))
    threshold = md.fermi2_c_threshold(lead.fermi_t_fm)
    near = replace(lead, radius_fm=repr(threshold * 1.001), sigma_rms_fm=repr(threshold * 0.01))
    assert float(near.radius_fm) / md.SPHERE_FACTOR - float(near.sigma_rms_fm) > 0
    assert not md.fermi2_c_not_real(near) and not md.rminus_in_domain(near)
    outside = [row for row in kept if not md.rminus_in_domain(row)]
    print(f"\nkept members {len(kept)}; rminus outside the domain: {len(outside)} "
          + " ".join(md.run_id(row, 'base')[:-len('_base')] for row in outside))


def test_t121_the_implied_numerics_runs_are_four_per_kept_member_plus_two_per_deep_nuclide():
    _out, kept = _kept()
    runs = md.expected_numerics_runs(kept)
    deep = [row for row in kept if row.nuclide in md.NUMERICS_DEEP_NUCLIDES]
    assert len(deep) == len(md.NUMERICS_DEEP_NUCLIDES)
    per_member, per_deep = 1 + len(md.NUMERICS_LEVELS_ALL), len(md.NUMERICS_LEVELS_DEEP)
    assert len(runs) == len(kept) * per_member + len(deep) * per_deep
    assert len({run for run, *_ in runs}) == len(runs)
    for row in deep:
        assert md.numerics_levels(row) == [*md.NUMERICS_LEVELS_ALL, *md.NUMERICS_LEVELS_DEEP]
    print(f"\nimplied numerics runs: {len(runs)} over {len(kept)} kept members, {len(deep)} deep")


# --------------------------------------------------------------------------------------------
# T-122 -- the numerics tables: exactly the implied runs, every input re-rendered to its digest,
# level 0 equal to the committed base strings, and the loaders' refusals
# --------------------------------------------------------------------------------------------

NUMERICS_RUNS = REPO / md.NUMERICS_RUNS_RELPATH
NUMERICS_STATES = REPO / md.NUMERICS_STATES_RELPATH
NUMERICS_LINES = REPO / md.NUMERICS_LINES_RELPATH


def test_t122_the_run_table_lists_exactly_the_implied_runs_and_every_input_rerenders_to_its_digest():
    out, kept = _kept()
    numerics = md.load_numerics_outputs(REPO, out)
    listed = [(r.run, r.z, r.a, r.kind, r.level) for r in numerics.runs.values()]
    assert listed == md.expected_numerics_runs(kept)
    cells, _ = md.load_validation(CELLS, ORIGIN)
    inputs = {row.nuclide: row for row in kept}
    made: dict[str, int] = {}
    invalid = 0
    for run in numerics.runs.values():
        row = inputs[(run.z, run.a)]
        extra = md.extra_lines(cells, row.nuclide)
        if run.status == md.INVALID_PERTURBATION_DOMAIN:
            invalid += 1
            assert run.kind == "rminus" and not md.rminus_in_domain(row)
            assert run.run not in numerics.headers and run.run not in numerics.lines
            continue
        text = (md.render_input(row, "rminus", extra) if run.kind == "rminus"
                else md.render_numerics_input(row, run.level, extra))
        assert hashlib.sha256(text.encode("ascii")).hexdigest() == run.input_sha256, run.run
        made[f"{run.kind}{run.level}"] = made.get(f"{run.kind}{run.level}", 0) + 1
    failed = [r.run for r in numerics.runs.values() if r.status == md.RAN and not r.clean]
    counts = ", ".join(f"{k} {v}" for k, v in sorted(made.items()))
    print(f"\nnumerics runs {len(numerics.runs)}: {counts}; {md.INVALID_PERTURBATION_DOMAIN} {invalid}; "
          f"rc != 0 or non-empty .err: {len(failed)} {failed}")


def test_t122_every_grid0_state_header_and_line_equals_its_committed_base_row_string_for_string():
    base_states = {(r["run"], r["state"]): (r["n"], r["l"], r["s"], r["binding_eV"], r["total_eV"])
                   for _, r in md.read_rows(REPO / md.STATES_RELPATH, md.STATES_COLUMNS)
                   if r["kind"] == "base"}
    base_lines: dict[str, list[tuple[str, str, str]]] = {}
    for _, r in md.read_rows(REPO / md.LINES_RELPATH, md.LINES_COLUMNS):
        if r["kind"] == "base":
            base_lines.setdefault(r["run"], []).append((r["line"], r["delta_e_eV"], r["w12_per_s"]))
    grid0_states = {}
    for _, r in md.read_rows(NUMERICS_STATES, md.NUMERICS_STATES_COLUMNS):
        if r["kind"] == "grid" and r["level"] == "0":
            grid0_states[(r["run"].replace("_grid0", "_base"), r["state"])] = (
                r["n"], r["l"], r["s"], r["binding_eV"], r["total_eV"])
    grid0_lines: dict[str, list[tuple[str, str, str]]] = {}
    for _, r in md.read_rows(NUMERICS_LINES, md.NUMERICS_LINES_COLUMNS):
        if r["kind"] == "grid" and r["level"] == "0":
            grid0_lines.setdefault(r["run"].replace("_grid0", "_base"), []).append(
                (r["line"], r["delta_e_eV"], r["w12_per_s"]))
    _out, kept = _kept()
    runs = {md.run_id(row, "base") for row in kept}
    assert {run for run, _ in grid0_states} == runs and set(grid0_lines) == runs
    assert grid0_states == {key: value for key, value in base_states.items() if key[0] in runs}
    assert grid0_lines == {run: lines for run, lines in base_lines.items() if run in runs}
    print(f"\ngrid0 equals base on {len(runs)} runs: {len(grid0_states)} state headers, "
          f"{sum(len(v) for v in grid0_lines.values())} lines")


def _numerics_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """A root holding copies of the three numerics tables under data/g4/d3."""
    d3 = tmp_path / md.D3_RELDIR
    d3.mkdir(parents=True)
    for source in (NUMERICS_RUNS, NUMERICS_STATES, NUMERICS_LINES):
        (d3 / source.name).write_bytes(source.read_bytes())
    return tmp_path


def test_t122_drill_a_missing_run_a_status_against_the_domain_rule_and_a_stray_run_are_refused(tmp_path):
    out, _kept_rows = _kept()
    root = _numerics_root(tmp_path)
    runs_path = root / md.NUMERICS_RUNS_RELPATH
    states_path = root / md.NUMERICS_STATES_RELPATH
    md.load_numerics_outputs(root, out)
    text = runs_path.read_bytes().decode("ascii")
    lines = _lines(text)
    runs_path.write_bytes(NL.join([*lines[:1], *lines[2:]]).encode("ascii"))
    with pytest.raises(md.CellError, match="does not list exactly the implied runs"):
        md.load_numerics_outputs(root, out)
    first = lines[1].split(",")
    assert first[3] == "rminus" and first[5] == md.RAN
    flipped = ",".join([*first[:5], md.INVALID_PERTURBATION_DOMAIN, "", "", "", ""])
    runs_path.write_bytes(NL.join([lines[0], flipped, *lines[2:]]).encode("ascii"))
    with pytest.raises(md.CellError, match="disagrees with the domain rule"):
        md.load_numerics_outputs(root, out)
    runs_path.write_bytes(text.encode("ascii"))
    states = _lines(states_path.read_bytes().decode("ascii"))
    last = states[-2]
    level = last.split(",")[4]
    stray = last.replace(f"_grid{level},", "_grid9,", 1).replace(f",grid,{level},", ",grid,9,", 1)
    assert stray != last and "_grid9," in stray
    states_path.write_bytes(NL.join([*states[:-1], stray, ""]).encode("ascii"))
    with pytest.raises(md.CellError, match="does not list as made"):
        md.load_numerics_outputs(root, out)


def _one_row(text: str, edit) -> str:
    """The header of ``text`` and its first row after ``edit``."""
    lines = _lines(text)
    return lines[0] + NL + edit(lines[1]) + NL


def _cell(index: int, new: str):
    def edit(row: str) -> str:
        cells = row.split(",")
        cells[index] = new
        return ",".join(cells)
    return edit


def _not_made(row: str) -> list[str]:
    return [*row.split(",")[:5], md.INVALID_PERTURBATION_DOMAIN, "", "", "", ""]


NUMERICS_RUNS_DRILLS = [
    ("a carriage return", lambda t: t.replace(NL, "\r" + NL, 1), md.CarriageReturnError, "contains CR"),
    ("a non-ASCII byte", lambda t: t.replace(",rminus,", ",rminus·", 1), md.NonAsciiError,
     "outside US-ASCII"),
    ("a renamed column", lambda t: _replace_once(t, "wall_s,input_sha256", "wall,input_sha256"),
     md.HeaderError, "header is"),
    ("an unknown kind", lambda t: _one_row(t, _cell(3, "rplus")), md.CellError, "kind"),
    ("a level that is not a count", lambda t: _one_row(t, _cell(4, "x")), md.CellError,
     "level must be a count"),
    ("an rminus run at level 1", lambda t: _one_row(t, _cell(4, "1")), md.CellError,
     "an rminus run is level 0"),
    ("a run not naming its kind", lambda t: _one_row(t, _cell(3, "grid")), md.CellError, "does not name"),
    ("an unknown status", lambda t: _one_row(t, _cell(5, "OK")), md.CellError, "status"),
    ("a signed rc", lambda t: _one_row(t, _cell(6, "+0")), md.CellError, "rc and err_bytes must be integers"),
    ("a signed wall time", lambda t: _one_row(t, _cell(8, "-1.0")), md.CellError,
     "wall_s must be a printed decimal"),
    ("a short digest", lambda t: _one_row(t, lambda row: row[:-1]), md.CellError,
     "input_sha256 must be 64 hex"),
    ("a run not made that carries an rc",
     lambda t: _one_row(t, lambda row: ",".join([*_not_made(row)[:6], "0", "", "", ""])),
     md.CellError, "carries no rc"),
    ("a grid run not made",
     lambda t: _one_row(t, lambda row: ",".join(
         _not_made(row.replace("_rminus", "_grid0").replace(",rminus,", ",grid,")))),
     md.CellError, "only an rminus run"),
    ("a duplicated run", lambda t: _repeat_line(t, 1), md.DuplicateKeyError, "duplicate run"),
    ("rows out of order", lambda t: _swap_lines(t, 1), md.OrderError, "ordered by (Z, A, kind, level)"),
    ("no rows", lambda t: _lines(t)[0] + NL, md.EmptyError, "carries no rows"),
]


@pytest.mark.parametrize("label, mutate, error, message", NUMERICS_RUNS_DRILLS,
                         ids=[d[0] for d in NUMERICS_RUNS_DRILLS])
def test_t122_drill_each_run_table_rule_refuses_its_fixture(tmp_path, label, mutate, error, message):
    _drill(tmp_path, NUMERICS_RUNS, mutate, error, message, md.load_numerics_runs)


NUMERICS_STATES_DRILLS = [
    ("a state that is not an orbit", lambda t: _one_row(t, _cell(5, "K")), md.CellError, "IUPAC orbit"),
    ("an n that is not an integer", lambda t: _one_row(t, _cell(6, "1.0")), md.CellError,
     "n must be an integer"),
    ("a binding that is not printed", lambda t: _one_row(t, _cell(9, "x")), md.CellError,
     "binding_eV must be"),
    ("a duplicated state", lambda t: _repeat_line(t, 1), md.DuplicateKeyError, "duplicate state"),
    ("rows out of order", lambda t: _swap_lines(t, 1), md.OrderError,
     "ordered by (Z, A, kind, level, orbit)"),
]


@pytest.mark.parametrize("label, mutate, error, message", NUMERICS_STATES_DRILLS,
                         ids=[d[0] for d in NUMERICS_STATES_DRILLS])
def test_t122_drill_each_state_table_rule_refuses_its_fixture(tmp_path, label, mutate, error, message):
    _drill(tmp_path, NUMERICS_STATES, mutate, error, message, md.load_numerics_states)


NUMERICS_LINES_DRILLS = [
    ("a line that is not orbit-orbit", lambda t: _one_row(t, _cell(5, "K1L2")), md.CellError, "orbit-orbit"),
    ("a line energy at five decimals", lambda t: _one_row(t, _cell(6, "1.12345")), md.CellError,
     "six decimals"),
    ("a rate that is not a decimal", lambda t: _one_row(t, _cell(7, "1e5")), md.CellError, "w12_per_s"),
    ("a duplicated line", lambda t: _repeat_line(t, 1), md.DuplicateKeyError, "duplicate line"),
    ("runs out of order", lambda t: NL.join([_lines(t)[0], _lines(t)[-2], *_lines(t)[1:-2], ""]),
     md.OrderError, "ordered by (Z, A, kind, level)"),
]


@pytest.mark.parametrize("label, mutate, error, message", NUMERICS_LINES_DRILLS,
                         ids=[d[0] for d in NUMERICS_LINES_DRILLS])
def test_t122_drill_each_line_table_rule_refuses_its_fixture(tmp_path, label, mutate, error, message):
    _drill(tmp_path, NUMERICS_LINES, mutate, error, message, md.load_numerics_lines)


# --------------------------------------------------------------------------------------------
# T-123 -- the shell projection: every gated line beside the shell difference the cascade emits,
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


def _numerics_by_level(row: md.InputRow, numerics: md.NumericsOutputs):
    """Per refinement level of ``row``: (clean, bindings or None, printed lines or None)."""
    out = {}
    for level in md.numerics_levels(row):
        run = numerics.runs[md.numerics_run_id(row, "grid", level)]
        bindings = lines = None
        if run.clean:
            try:
                bindings = md.derive_bindings(run.run, numerics.headers.get(run.run, {}),
                                              numerics.lines.get(run.run, {}))
                lines = numerics.lines[run.run]
            except md.DerivationError:
                bindings = lines = None
        out[level] = (run.clean, bindings, lines)
    return out


def _qualify(
    values: dict[int, Decimal], failed: bool, target: Decimal, allowance: Decimal
) -> tuple[str, str]:
    """The rule as this test states it: (shifts text, token)."""
    levels = sorted(values)
    with localcontext() as context:
        context.prec = md._PRECISION
        observed = [values[b] - values[a] for a, b in zip(levels, levels[1:], strict=False)]
    text = ";".join(format(s, "f") for s in observed)
    if len(levels) < 3:
        return text, "INSUFFICIENT_LEVELS"
    if failed:
        return text, "RUN_FAILED"
    older, newer = observed[-2], observed[-1]
    ok = abs(older) <= target + allowance and abs(newer) <= target + allowance and abs(newer) <= abs(older)
    return text, f"{'QUALIFIED_AT_LEVEL_' if ok else 'NOT_QUALIFIED_THROUGH_LEVEL_'}{levels[-1]}"


def test_t123_drill_reversed_shell_pair_is_refused_through_stock_and_line_helpers():
    levels = (2.0, 1.0)
    assert md.geant4_line_kev(levels, "K-L") == d3c.stock_kev(levels, 1, 2)
    with pytest.raises(md.CellError, match="lower to a higher shell"):
        d3c.stock_kev(levels, 2, 1)
    with pytest.raises(md.CellError, match="lower to a higher shell"):
        md.geant4_line_kev(levels, "L-K")


def test_t123_drill_qualification_clauses_and_equality_edges():
    target = Decimal("1")
    valid = list(range(1, d3c.MIN_VALID_LEVELS + 1))
    qualified = f"{d3c.QUALIFIED}{valid[-1]}"
    unqualified = f"{d3c.NOT_QUALIFIED}{valid[-1]}"
    assert d3c.qualification([target, target], valid, False, target, Decimal(0)) == qualified
    assert (d3c.qualification([target, target], valid[:-1], False, target, Decimal(0)) ==
            d3c.INSUFFICIENT_LEVELS)
    assert d3c.qualification([target, target], valid, True, target, Decimal(0)) == d3c.RUN_FAILED
    above = target + Decimal("0.1")
    for shifts in ([above, Decimal(0)], [target, above], [Decimal("0.5"), Decimal("0.6")]):
        assert d3c.qualification(shifts, valid, False, target, Decimal(0)) == unqualified


def test_t123_the_projection_equals_an_independent_rederivation_from_the_tables_text():
    out = md.load_outputs(REPO)
    numerics = md.load_numerics_outputs(REPO, out)
    cells, _ = md.load_validation(CELLS, ORIGIN)
    stock = md.load_geant4_levels(GEANT4_LEVELS)
    kshell = _table_cells(D3DIR / "d3_kshell.mudirac130.g4dat")
    levels = _table_cells(D3DIR / "d3_levels.mudirac130.g4dat")
    inputs = {row.nuclide: row for row in out.inputs}
    committed = _committed_csv(PROJECTION)
    gated = [cell for cell in cells if cell.gated]
    assert list(committed[0]) == list(d3c.PROJECTION_COLUMNS)
    assert len(committed) == len(gated)
    assert md.TOL_FACTOR == 3 and md.SIGMA_CALC == 0
    path = d3c.path_lengths()
    assert path["K1"] == md.N_MAX - 1
    half_line = md.LINE_HALF_UNIT / 1000
    refinements = {}
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
        with localcontext() as context:
            context.prec = md._PRECISION
            measured = Decimal(cell.value_kev)
            tol = 3 * Decimal(cell.unc_kev)
            solver = Decimal(out.lines[md.run_id(member, "base")][cell.quantity]) / 1000
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
            # the observed stability, from the numerics tables and the rule as stated above
            if key not in refinements:
                refinements[key] = _numerics_by_level(member, numerics)
            by_level = refinements[key]
            failed = any(not clean for clean, _, _ in by_level.values())
            solver_values = {level: Decimal(lines[cell.quantity]) / 1000
                             for level, (_, _, lines) in by_level.items() if lines and cell.quantity in lines}
            sigmas = [Decimal(c.unc_kev) for c in gated if c.nuclide == key and c.quantity == cell.quantity]
            solver_target = Decimal("0.1") * min(sigmas)
            assert Decimal(row["solver_numeric_target_keV"]) == solver_target
            text, token = _qualify(solver_values, failed, solver_target, 2 * half_line)
            assert (row["solver_numeric_shifts_keV"], row["solver_numeric_qualification"]) == (
                text, token), row
            anchor_header = out.headers[md.run_id(member, "base")][md.orbit(md.N_MAX, True)]
            anchor_bound = md.half_unit_6sig(anchor_header) / 1000
            consumer_target = Decimal(0)
            consumer_allowance = Decimal(0)
            consumer_values: dict[int, Decimal] = {}
            for level, (_, bindings, _) in by_level.items():
                if bindings is not None:
                    k, means = md.quantities(bindings)
                    consumer_values[level] = (k if lower_n == 1 else means[lower_n - 2]) - means[upper_n - 2]
            for n in (lower_n, upper_n):
                central = Decimal(_shell_text(kshell, levels, key, n))
                consumer_target += max(Decimal("1e-6"), Decimal("1e-6") * abs(central))
                if n == 1:
                    line_bound = path["K1"] * half_line
                else:
                    ell = n - 1
                    weighted = 2 * ell * path[md.orbit(n, False)] + (2 * ell + 2) * path[md.orbit(n, True)]
                    line_bound = Decimal(weighted) / (4 * ell + 2) * half_line
                consumer_allowance += 2 * (line_bound + anchor_bound)
            assert Decimal(row["consumer_numeric_target_keV"]) == consumer_target
            text, token = _qualify(consumer_values, failed, consumer_target, consumer_allowance)
            assert (row["consumer_numeric_shifts_keV"], row["consumer_numeric_qualification"]) == (
                text, token), row
    assert PROJECTION.read_bytes() == d3c.render_projection(REPO)


def _counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "gated": len(rows),
        "solver_within": sum(r["solver_within"] == "true" for r in rows),
        "consumer_within": sum(r["consumer_within"] == "true" for r in rows),
        "stock_within": sum(r["stock_within"] == "true" for r in rows),
        "consumer_closer_than_stock": sum(r["consumer_closer_than_stock"] == "true" for r in rows),
    }


def test_t123_the_projection_counts_are_pinned():
    """Regression pins, copied from the unit's own measurement log -- not quotas: a change trips
    this test on purpose."""
    counts = _counts(_committed_csv(PROJECTION))
    assert counts == {"gated": 67, "solver_within": 26, "consumer_within": 1, "stock_within": 0,
                      "consumer_closer_than_stock": 42}
    print("\nprojection counts: " + " ".join(f"{k} {v}" for k, v in counts.items()))


def _d3_copy(tmp_path: pathlib.Path) -> pathlib.Path:
    """A root whose data/g4/d3 is a copy of the committed directory."""
    target = tmp_path / md.D3_RELDIR
    target.mkdir(parents=True)
    for source in D3DIR.iterdir():
        if source.is_file():
            (target / source.name).write_bytes(source.read_bytes())
    return tmp_path


CONSUMER_COLUMNS = {"consumer_keV", "consumer_residual_keV", "consumer_within", "consumer_closer_than_stock",
                    "representation_error_keV", "consumer_numeric_target_keV"}


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
    """Pins copied from the unit's own measurement log; the two Pb-208 constants are assertion
    targets computed here from the printed cells and the tables' text, never inputs."""
    committed = _committed_csv(GROUPS)
    empty = [row for row in committed if row["empty"] == "true"]
    assert (len(committed), len(empty)) == (29, 29)
    pb = next(row for row in committed
              if (row["source"], row["Z"], row["A"], row["initial_n"], row["final_n"])
              == ("Fricke1995", "82", "208", "2", "1"))
    assert Decimal(pb["gap_keV"]) == Decimal("184.226")
    assert pb == max(committed, key=lambda row: Decimal(row["gap_keV"]))
    kshell = _table_cells(D3DIR / "d3_kshell.mudirac130.g4dat")
    levels = _table_cells(D3DIR / "d3_levels.mudirac130.g4dat")
    with localcontext() as context:
        context.prec = md._PRECISION
        k_to_l = (Decimal(_shell_text(kshell, levels, (82, 208), 1))
                  - Decimal(_shell_text(kshell, levels, (82, 208), 2)))
    assert abs(k_to_l - Decimal("5902.236247053")) <= Decimal("1e-9")
    projection = _committed_csv(PROJECTION)
    pb_k_lines = [row for row in projection if row["Z"] == "82" and row["quantity"].startswith("K1-L")]
    assert {row["consumer_keV"] for row in pb_k_lines} == {format(k_to_l, "f")}
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
COMPONENT_KEYS = (
    "Z", "A", "quantity", "central_keV", "sigma_rms_fm", "radius_plus_relative_response_keV",
    "radius_minus_relative_response_keV", "radius_plus_absolute_response_keV",
    "radius_minus_absolute_response_keV", "anchor_rounding_bound_keV", "anchor_coefficient",
    "line_rounding_bound_keV", "numeric_observed_shifts_keV", "numeric_resolution_target_keV",
    "numeric_qualification", "correlation_groups", "legacy_radius_response_magnitude_with_floor",
)


def _component_records(path: pathlib.Path) -> list[dict]:
    text = path.read_bytes().decode("ascii")
    assert "\r" not in text and text.endswith(NL)
    return [json.loads(line) for line in text.split(NL)[:-1]]


def _quantity(b: dict[str, Decimal], quantity: str) -> Decimal:
    k, means = md.quantities(b)
    return k if quantity == "K" else means[int(quantity[1:]) - 2]


def _line_bound(path: dict[str, int], quantity: str) -> Decimal:
    half_line = md.LINE_HALF_UNIT / 1000
    if quantity == "K":
        return path["K1"] * half_line
    n = int(quantity[1:])
    ell = n - 1
    weighted = 2 * ell * path[md.orbit(n, False)] + (2 * ell + 2) * path[md.orbit(n, True)]
    return Decimal(weighted) / (4 * ell + 2) * half_line


def _component_problems(records: list[dict]) -> list[str]:
    """Every record of ``records`` that the committed files do not give, named."""
    out = md.load_outputs(REPO)
    numerics = md.load_numerics_outputs(REPO, out)
    kshell = _table_cells(D3DIR / "d3_kshell.mudirac130.g4dat")
    levels = _table_cells(D3DIR / "d3_levels.mudirac130.g4dat")
    kept = md.kept_members(out)
    table_rows, _ = md.table_rows(out)
    path = d3c.path_lengths()
    problems: list[str] = []
    header, body = records[0], records[1:]
    digests = sorted(hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in d3c.COMPONENT_INPUTS)
    expected_header = {
        "record": "header", "profile": md.PROFILE, "radius_perturbation": "source_rms_sigma",
        "omitted_components": ["model discrepancy", "nuclear polarization", "electron screening",
                               "higher-order QED", "hyperfine structure",
                               "numerical settings beyond the observed levels"],
        "total_sigma_keV": None, "model_uncertainty": "unknown",
        "inputs_sha256": hashlib.sha256((NL.join(digests) + NL).encode("ascii")).hexdigest(),
    }
    if header != expected_header:
        problems.append("header")
    expected: list[dict] = []
    with localcontext() as context:
        context.prec = md._PRECISION
        for row in kept:
            key = row.nuclide
            base = md.derive_bindings(md.run_id(row, "base"), out.headers[md.run_id(row, "base")],
                                      out.lines[md.run_id(row, "base")])
            plus = md.derive_bindings(md.run_id(row, "rsig"), out.headers[md.run_id(row, "rsig")],
                                      out.lines[md.run_id(row, "rsig")])
            minus_run = numerics.runs[md.numerics_run_id(row, "rminus", 0)]
            minus = None
            if minus_run.clean:
                minus = md.derive_bindings(minus_run.run, numerics.headers[minus_run.run],
                                           numerics.lines[minus_run.run])
            by_level = _numerics_by_level(row, numerics)
            failed = any(not clean for clean, _, _ in by_level.values())
            anchor = md.half_unit_6sig(out.headers[md.run_id(row, "base")][md.orbit(md.N_MAX, True)]) / 1000
            for index, quantity in enumerate(("K", *(f"e{n}" for n in range(2, md.N_MAX + 1)))):
                central = kshell[key][0] if quantity == "K" else levels[key][index - 1]
                legacy = kshell[key][1] if quantity == "K" else levels[key][md.N_MAX - 1 + index - 1]
                values = {level: _quantity(b, quantity) for level, (_, b, _) in by_level.items()
                          if b is not None}
                target = max(Decimal("1e-6"), Decimal("1e-6") * abs(Decimal(central)))
                line_bound = _line_bound(path, quantity)
                shifts_text, token = _qualify(values, failed, target, 2 * (line_bound + anchor))
                plus_rel = _quantity(md.relative(plus), quantity) - _quantity(md.relative(base), quantity)
                record = {
                    "Z": row.z, "A": row.a, "quantity": quantity, "central_keV": central,
                    "sigma_rms_fm": row.sigma_rms_fm,
                    "radius_plus_relative_response_keV": format(plus_rel, "f"),
                    "radius_minus_relative_response_keV": None if minus is None else format(
                        _quantity(md.relative(minus), quantity) - _quantity(md.relative(base), quantity),
                        "f"),
                    "radius_plus_absolute_response_keV": format(
                        _quantity(plus, quantity) - _quantity(base, quantity), "f"),
                    "radius_minus_absolute_response_keV": None if minus is None else format(
                        _quantity(minus, quantity) - _quantity(base, quantity), "f"),
                    "anchor_rounding_bound_keV": format(anchor, "f"),
                    "anchor_coefficient": "1",
                    "line_rounding_bound_keV": format(line_bound, "f"),
                    "numeric_observed_shifts_keV": shifts_text.split(";") if shifts_text else [],
                    "numeric_resolution_target_keV": format(target, "f"),
                    "numeric_qualification": token,
                    "correlation_groups": [f"anchor:{md.run_id(row, 'base')}", f"radius:{row.symbol}{row.a}"],
                    "legacy_radius_response_magnitude_with_floor": legacy,
                }
                assert float(legacy) == float(max(abs(plus_rel), md.UNC_FLOOR_KEV)), (key, quantity)
                expected.append(record)
    for table_row in table_rows:
        if table_row.a == 0:
            expected.append({"record": "reference", "Z": table_row.z, "A": 0,
                             "references": {"Z": table_row.z, "A": table_row.carries}})
    if len(body) != len(expected):
        problems.append(f"record count {len(body)} != {len(expected)}")
    for got, want in zip(body, expected, strict=False):
        if got != want:
            problems.append(f"record {want.get('Z')},{want.get('A')},{want.get('quantity', 'reference')}")
    return problems


def test_t126_every_component_field_is_rederived_from_the_committed_files():
    records = _component_records(COMPONENTS)
    assert not _component_problems(records)
    body = [r for r in records[1:] if "quantity" in r]
    assert all(list(r) == list(COMPONENT_KEYS) for r in body)
    out = md.load_outputs(REPO)
    rows, _ = md.table_rows(out)
    isotopes, natural = sum(1 for r in rows if r.a > 0), sum(1 for r in rows if r.a == 0)
    assert len(records) == isotopes * len(d3c.QUANTITIES) + natural + 1
    pb = next(r for r in body if (r["Z"], r["A"], r["quantity"]) == (82, 208, "K"))
    assert Decimal(pb["anchor_rounding_bound_keV"]) == Decimal("0.0005")
    assert Decimal(pb["anchor_rounding_bound_keV"]) > md.UNC_FLOOR_KEV
    assert COMPONENTS.read_bytes() == d3c.render_components(REPO)
    histogram: dict[str, int] = {}
    for r in body:
        histogram[r["numeric_qualification"]] = histogram.get(r["numeric_qualification"], 0) + 1
    minus_made = sum(r["radius_minus_relative_response_keV"] is not None for r in body)
    print(f"\ncomponents {len(body)}; rminus responses present {minus_made}; qualification "
          + ", ".join(f"{k} {v}" for k, v in sorted(histogram.items())))


def test_t126_the_anchor_moves_every_absolute_binding_by_its_coefficient_and_cancels_in_every_difference():
    """A moved anchor header moves every derived binding, K and each (2j+1) mean alike, by exactly
    its shift (the weights sum to one); the anchor-subtracted values, and every difference of two
    derived values, do not move; a difference against a value computed elsewhere does."""
    out = md.load_outputs(REPO)
    row = next(r for r in out.inputs if r.nuclide == (82, 208))
    run = md.run_id(row, "base")
    headers, lines = dict(out.headers[run]), out.lines[run]
    anchor = md.orbit(md.N_MAX, True)
    delta = Decimal("0.5")
    moved_headers = dict(headers)
    moved_headers[anchor] = format(Decimal(headers[anchor]) - delta, "f")
    b = md.derive_bindings(run, headers, lines)
    b2 = md.derive_bindings(run, moved_headers, lines)
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


def test_t126_drill_a_tampered_shift_is_refused(tmp_path):
    records = _component_records(COMPONENTS)
    tampered = json.loads(json.dumps(records))
    target = next(r for r in tampered[1:] if r.get("numeric_observed_shifts_keV"))
    shift = target["numeric_observed_shifts_keV"][0]
    target["numeric_observed_shifts_keV"][0] = shift + "1"
    problems = _component_problems(tampered)
    assert problems == [f"record {target['Z']},{target['A']},{target['quantity']}"]


# T-127 -- named settings inputs and their committed run records


def test_t127_settings_render_from_the_base_and_name_exactly_the_compared_kept_nuclides(monkeypatch):
    cells, _ = md.load_validation(CELLS, ORIGIN)
    out = md.load_outputs(REPO)
    kept = {row.nuclide: row for row in md.kept_members(out)}
    wanted = md.settings_nuclides(cells)
    expected = set(md.centroid_nuclides(cells)) | {
        left.nuclide for left, _right in md.doublet_determinations(cells)}
    assert set(wanted) == expected <= set(kept)
    assert len(md.SETTINGS) == len(md.SETTINGS_UEHLING_G0) + 1
    assert tuple(sorted(set(md.SETTINGS_UEHLING_G0))) == md.SETTINGS_UEHLING_G0
    for key in wanted:
        row = kept[key]
        extra = md.extra_lines(cells, key)
        base = md.render_input(row, "base", extra)
        for level, uehling in md.SETTINGS:
            values = md.settings_values(level, uehling)
            expected_text = base + "".join(f"{name}: {value}\n" for name, value in zip(
                md.NUMERICS_KEYS, values, strict=True))
            assert md.render_settings_input(row, level, uehling, extra) == expected_text
            assert md.settings_run_id(row, level, uehling).endswith(md.setting_id(level, uehling))
    first = kept[wanted[0]]
    extra = md.extra_lines(cells, first.nuclide)
    with pytest.raises(ValueError, match="unknown setting"):
        md.render_settings_input(first, -1, 100, extra)
    original = md.render_input
    monkeypatch.setattr(md, "render_input", lambda *args: original(*args) + "uehling_steps: 100\n")
    with pytest.raises(ValueError, match="already sets uehling_steps"):
        md.render_settings_input(first, 0, 100, extra)
    print(f"\nsettings nuclides {len(wanted)} runs {len(wanted) * len(md.SETTINGS)}")


def test_t127_settings_loaders_refuse_each_record_shape(tmp_path):
    setting = md.setting_id(*md.SETTINGS[0])
    values = md.settings_values(*md.SETTINGS[0])
    run = f"Be9_{setting}"
    digest = "a" * 64
    header = ",".join(md.SETTINGS_RUNS_COLUMNS) + NL
    row = ",".join([run, "4", "9", setting, *values, "0", "0", "0.1", digest]) + NL
    path = tmp_path / "runs.csv"
    path.write_bytes((header + row).encode("ascii"))
    assert list(md.load_settings_runs(path)) == [run]
    unlisted = md.setting_id(9, 9999)
    unlisted_row = ",".join([f"Be9_{unlisted}", "4", "9", unlisted,
                             *md.settings_values(9, 9999), "0", "0", "0.1", digest]) + NL
    bad = [
        (header.replace("setting", "kind", 1) + row, md.HeaderError),
        ((header + row).replace(NL, "\r" + NL, 1), md.CarriageReturnError),
        ((header + row).replace("Be9", "BeÂ·9", 1), md.NonAsciiError),
        (header + row.replace(setting, "g9u9999"), md.CellError),
        (header + unlisted_row, md.CellError),
        (header + row.replace("Be9_", "Be8_"), md.CellError),
        (header + row.replace(values[0], "0.1", 1), md.CellError),
        (header + row.replace(",0,0,0.1,", ",x,0,0.1,"), md.CellError),
        (header + row.replace(",0,0,0.1,", ",0,-1,0.1,"), md.CellError),
        (header + row.replace(",0,0,0.1,", ",0,0,-0.1,"), md.CellError),
        (header + row[:-2] + NL, md.CellError),
        (header + row + row, md.DuplicateKeyError),
    ]
    for changed, error in bad:
        path.write_bytes(changed.encode("utf-8"))
        with pytest.raises(error):
            md.load_settings_runs(path)
    second_setting = md.setting_id(*md.SETTINGS[1])
    second_values = md.settings_values(*md.SETTINGS[1])
    second = ",".join([f"Be9_{second_setting}", "4", "9", second_setting, *second_values,
                       "0", "0", "0.1", digest]) + NL
    path.write_bytes((header + second + row).encode("ascii"))
    with pytest.raises(md.OrderError):
        md.load_settings_runs(path)

    state_header = ",".join(md.SETTINGS_STATES_COLUMNS) + NL
    state = ",".join([run, "4", "9", setting, "K1", "1", "0", "1", "1.0", "1.0"]) + NL
    line_header = ",".join(md.SETTINGS_LINES_COLUMNS) + NL
    line = ",".join([run, "4", "9", setting, "K1-L2", "1.000000", "1.0"]) + NL
    state_path, line_path = tmp_path / "states.csv", tmp_path / "lines.csv"
    state_path.write_bytes((state_header + state).encode("ascii"))
    line_path.write_bytes((line_header + line).encode("ascii"))
    assert run in md._load_settings_printed(state_path, md.SETTINGS_STATES_COLUMNS, True)
    assert run in md._load_settings_printed(line_path, md.SETTINGS_LINES_COLUMNS, False)
    for changed in (state.replace("K1,", "bad,"), state.replace(",1,0,1,", ",x,0,1,"),
                    state.replace(",1.0,1.0", ",bad,1.0"), state + state):
        state_path.write_bytes((state_header + changed).encode("ascii"))
        with pytest.raises(md.Mudirac130Error):
            md._load_settings_printed(state_path, md.SETTINGS_STATES_COLUMNS, True)
    for changed in (line.replace("K1-L2", "bad"), line.replace("1.000000", "1"),
                    line.rsplit(",1.0", 1)[0] + ",-1.0" + NL, line + line):
        line_path.write_bytes((line_header + changed).encode("ascii"))
        with pytest.raises(md.Mudirac130Error):
            md._load_settings_printed(line_path, md.SETTINGS_LINES_COLUMNS, False)
    later_state = state.replace("K1,1,0,1", "L2,2,1,1")
    state_path.write_bytes((state_header + later_state + state).encode("ascii"))
    with pytest.raises(md.OrderError):
        md._load_settings_printed(state_path, md.SETTINGS_STATES_COLUMNS, True)
    later_line = line.replace(run, f"Be9_{second_setting}").replace("," + setting + ",",
                                                                    "," + second_setting + ",")
    line_path.write_bytes((line_header + later_line + line).encode("ascii"))
    with pytest.raises(md.OrderError):
        md._load_settings_printed(line_path, md.SETTINGS_LINES_COLUMNS, False)


def test_t127_committed_settings_are_the_implied_runs_and_the_default_equals_base():
    cells = md.load_cells(CELLS)
    settings = md.load_settings_outputs(REPO, cells)
    kept = {row.nuclide: row for row in md.kept_members(md.load_outputs(REPO))}
    expected = [(md.settings_run_id(kept[key], level, uehling), *key, md.setting_id(level, uehling))
                for key in md.settings_nuclides(cells) for level, uehling in md.SETTINGS]
    assert [(r.run, r.z, r.a, r.setting) for r in settings.runs.values()] == expected
    for run in settings.runs.values():
        row = kept[(run.z, run.a)]
        level, uehling = next(pair for pair in md.SETTINGS if md.setting_id(*pair) == run.setting)
        rendered = md.render_settings_input(row, level, uehling, md.extra_lines(cells, row.nuclide))
        assert hashlib.sha256(rendered.encode("ascii")).hexdigest() == run.input_sha256
        if not run.clean:
            assert run.run not in settings.headers and run.run not in settings.lines
    base_states = {(r["run"], r["state"]): (r["n"], r["l"], r["s"], r["binding_eV"], r["total_eV"])
                   for _, r in md.read_rows(REPO / md.STATES_RELPATH, md.STATES_COLUMNS)
                   if r["kind"] == "base"}
    base_lines = {(r["run"], r["line"]): (r["delta_e_eV"], r["w12_per_s"])
                  for _, r in md.read_rows(REPO / md.LINES_RELPATH, md.LINES_COLUMNS)
                  if r["kind"] == "base"}
    states = md.read_rows(REPO / md.SETTINGS_STATES_RELPATH, md.SETTINGS_STATES_COLUMNS)
    lines = md.read_rows(REPO / md.SETTINGS_LINES_RELPATH, md.SETTINGS_LINES_COLUMNS)
    for _where, r in states:
        if r["setting"] == md.setting_id(0, 100):
            base = r["run"].replace("_g0u0100", "_base")
            assert (r["n"], r["l"], r["s"], r["binding_eV"], r["total_eV"]) == (
                base_states[(base, r["state"])])
    for _where, r in lines:
        if r["setting"] == md.setting_id(0, 100):
            base = r["run"].replace("_g0u0100", "_base")
            assert (r["delta_e_eV"], r["w12_per_s"]) == base_lines[(base, r["line"])]
    print(f"\nsettings runs {len(settings.runs)} clean {sum(r.clean for r in settings.runs.values())}")


def test_t127_script_writes_named_inputs_and_reports_a_missing_optional_stock_line(
        monkeypatch, tmp_path, capsys):
    import runpy
    from types import SimpleNamespace

    script = runpy.run_path(str(REPO / "scripts" / "mudirac_d3.py"))
    cells = md.load_cells(CELLS)
    script["cmd_write_settings"](SimpleNamespace(dir=str(tmp_path / "settings")))
    wanted = md.settings_nuclides(cells)
    paths = list((tmp_path / "settings").glob("g*/*/*.in"))
    assert len(paths) == len(wanted) * len(md.SETTINGS)
    sample = paths[0]
    assert sample.stem == sample.parent.name
    with monkeypatch.context() as patcher:
        script_md = script["cmd_write_settings"].__globals__["md"]
        patcher.setattr(script_md, "settings_nuclides", lambda _cells: [*wanted, (999, 999)])
        with pytest.raises(SystemExit, match="settings nuclides not kept"):
            script["cmd_write_settings"](SimpleNamespace(dir=str(tmp_path / "unkept")))
    harvest = _harvest_from_levels()
    optional = next(key for key in md.centroid_nuclides(cells) if key not in md.gated_nuclides(cells))
    line = next(line for line in harvest.split(NL) if line.startswith(f"C {optional[0]} {optional[1]} "))
    source, out = tmp_path / "harvest.txt", tmp_path / "levels.csv"
    source.write_text(harvest.replace(line + NL, ""), encoding="ascii")
    script["cmd_geant4_levels"](SimpleNamespace(harvest=str(source), out=str(out)))
    assert optional not in md.load_geant4_levels(out)
    assert f"no C line: Z={optional[0]} A={optional[1]}" in capsys.readouterr().out


def test_t127_script_times_out_and_collects_only_clean_settings(monkeypatch, tmp_path):
    import runpy
    import subprocess
    from types import SimpleNamespace

    script = runpy.run_path(str(REPO / "scripts" / "mudirac_d3.py"))
    run_one = script["_run_one"]
    globals_ = run_one.__globals__
    directory = tmp_path / "Be9_g0u0050"
    directory.mkdir()
    (directory / "Be9_g0u0050.in").write_text("input\n", encoding="ascii")

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 1
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(globals_["subprocess"], "run", timeout)
    assert run_one("binary", directory, False, 1)[1] == 124

    collect = script["cmd_collect_settings"]
    collected = collect.__globals__
    script_md = collected["md"]
    cells = md.load_cells(CELLS)
    key = md.settings_nuclides(cells)[0]
    row = next(r for r in md.kept_members(md.load_outputs(REPO)) if r.nuclide == key)
    level, uehling = md.SETTINGS[0]
    run = md.settings_run_id(row, level, uehling)
    run_dir = tmp_path / "collected" / run
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(script_md, "settings_nuclides", lambda _cells: [key])
    monkeypatch.setattr(script_md, "SETTINGS", ((level, uehling),))
    monkeypatch.setitem(collected, "_results", lambda dirs: [(run, run_dir, "0", "0", "0.1", "a" * 64)])
    monkeypatch.setitem(collected, "_state_headers", lambda directory: {"K1": ("1", "0", "1", "1.0", "1.0")})
    monkeypatch.setitem(collected, "_lines", lambda directory: [("K1-L2", "1.000000", "1.0")])
    out = tmp_path / "out"
    out.mkdir()
    collect(SimpleNamespace(dir=str(tmp_path), out=str(out)))
    assert len(md.read_rows(out / pathlib.Path(md.SETTINGS_RUNS_RELPATH).name, md.SETTINGS_RUNS_COLUMNS)) == 1
    states_out = out / pathlib.Path(md.SETTINGS_STATES_RELPATH).name
    lines_out = out / pathlib.Path(md.SETTINGS_LINES_RELPATH).name
    assert len(md.read_rows(states_out, md.SETTINGS_STATES_COLUMNS)) == 1
    assert len(md.read_rows(lines_out, md.SETTINGS_LINES_COLUMNS)) == 1
    monkeypatch.setitem(collected, "_results", lambda dirs: [(run, run_dir, "124", "0", "0.1", "a" * 64)])
    collect(SimpleNamespace(dir=str(tmp_path), out=str(out)))
    assert states_out.read_text(encoding="ascii").splitlines() == [",".join(md.SETTINGS_STATES_COLUMNS)]
    assert lines_out.read_text(encoding="ascii").splitlines() == [",".join(md.SETTINGS_LINES_COLUMNS)]
    monkeypatch.setitem(collected, "_results", lambda dirs: [])
    with pytest.raises(SystemExit, match="no result"):
        collect(SimpleNamespace(dir=str(tmp_path), out=str(out)))
    monkeypatch.setitem(collected, "_results", lambda dirs: [
        (run, run_dir, "0", "0", "0.1", "a" * 64),
        ("unexpected", run_dir, "0", "0", "0.1", "a" * 64),
    ])
    with pytest.raises(SystemExit, match="unexpected settings results"):
        collect(SimpleNamespace(dir=str(tmp_path), out=str(out)))


def test_t127_drill_wanted_set_equal_to_kept_set_is_admitted(monkeypatch):
    cells = md.load_cells(CELLS)
    wanted = set(md.settings_nuclides(cells))
    out = md.load_outputs(REPO)
    original = md.kept_members
    monkeypatch.setattr(md, "kept_members", lambda loaded: tuple(row for row in original(loaded)
                                                            if row.nuclide in wanted))
    assert set(row.nuclide for row in md.kept_members(out)) == wanted
    assert md.load_settings_outputs(REPO, cells).runs


def test_t127_settings_crosscheck_refuses_missing_and_unclean_records(monkeypatch, tmp_path):
    cells = md.load_cells(CELLS)
    out = md.load_outputs(REPO)
    d3 = tmp_path / md.D3_RELDIR
    d3.mkdir(parents=True)
    paths = [md.SETTINGS_RUNS_RELPATH, md.SETTINGS_STATES_RELPATH, md.SETTINGS_LINES_RELPATH]
    for rel in paths:
        (tmp_path / rel).write_bytes((REPO / rel).read_bytes())
    monkeypatch.setattr(md, "load_outputs", lambda root: out)
    loaded = md.load_settings_outputs(tmp_path, cells)
    assert len(loaded.runs) == len(md.settings_nuclides(cells)) * len(md.SETTINGS)
    runs_path = tmp_path / md.SETTINGS_RUNS_RELPATH
    states_path = tmp_path / md.SETTINGS_STATES_RELPATH
    lines_path = tmp_path / md.SETTINGS_LINES_RELPATH
    runs, states, lines = (path.read_text(encoding="ascii") for path in
                           (runs_path, states_path, lines_path))
    run_rows = runs.splitlines()
    runs_path.write_bytes((NL.join([run_rows[0], *run_rows[2:]]) + NL).encode("ascii"))
    with pytest.raises(md.CellError, match="implied runs"):
        md.load_settings_outputs(tmp_path, cells)
    runs_path.write_bytes(runs.encode("ascii"))
    first = run_rows[1].split(",")
    assert first[7:9] == ["0", "0"]
    first[7] = "124"
    runs_path.write_bytes((NL.join([run_rows[0], ",".join(first), *run_rows[2:]]) + NL).encode("ascii"))
    with pytest.raises(md.CellError, match="unclean runs"):
        md.load_settings_outputs(tmp_path, cells)
    runs_path.write_bytes(runs.encode("ascii"))
    state_rows = states.splitlines()
    stray = state_rows[1].replace(state_rows[1].split(",")[0], "X9_g0u0050", 1)
    states_path.write_bytes((NL.join([state_rows[0], stray, *state_rows[2:]]) + NL).encode("ascii"))
    with pytest.raises(md.CellError, match="unlisted"):
        md.load_settings_outputs(tmp_path, cells)
    states_path.write_bytes(states.encode("ascii"))
    line_rows = lines.splitlines()
    stray = line_rows[1].replace(line_rows[1].split(",")[0], "X9_g0u0050", 1)
    lines_path.write_bytes((NL.join([line_rows[0], stray, *line_rows[2:]]) + NL).encode("ascii"))
    with pytest.raises(md.CellError, match="unlisted"):
        md.load_settings_outputs(tmp_path, cells)


def test_t127_drill_clean_settings_runs_need_printed_rows_in_both_tables(monkeypatch, tmp_path):
    cells = md.load_cells(CELLS)
    out = md.load_outputs(REPO)
    monkeypatch.setattr(md, "load_outputs", lambda root: out)
    (tmp_path / md.D3_RELDIR).mkdir(parents=True)
    for rel in (md.SETTINGS_RUNS_RELPATH, md.SETTINGS_STATES_RELPATH, md.SETTINGS_LINES_RELPATH):
        (tmp_path / rel).write_bytes((REPO / rel).read_bytes())
    runs = md.load_settings_runs(tmp_path / md.SETTINGS_RUNS_RELPATH)
    clean = next(run for run in runs if runs[run].clean)
    for rel in (md.SETTINGS_STATES_RELPATH, md.SETTINGS_LINES_RELPATH):
        path = tmp_path / rel
        original = path.read_text(encoding="ascii")
        rows = original.splitlines()
        path.write_bytes((NL.join(row for row in rows if row.split(",")[0] != clean) + NL).encode("ascii"))
        with pytest.raises(md.CellError, match="lacks the printed rows of clean runs"):
            md.load_settings_outputs(tmp_path, cells)
        path.write_bytes(original.encode("ascii"))


def test_t104_drill_equal_run_and_state_orders_are_admitted(tmp_path):
    for relpath, loader, change in (
        (md.RUNS_RELPATH, md.load_runs, lambda row: "X" + row),
        (md.STATES_RELPATH, md.load_states, lambda row: row.replace(",K1,", ",K01,")),
    ):
        header, first, *_ = (REPO / relpath).read_text(encoding="ascii").splitlines()
        second = change(first)
        assert second != first
        path = tmp_path / pathlib.Path(relpath).name
        path.write_bytes(f"{header}{NL}{first}{NL}{second}{NL}".encode("ascii"))
        loaded = loader(path)
        if loader is md.load_runs:
            assert len(loaded) == 2
        else:
            assert len(loaded[first.split(",")[0]]) == 2


def test_t104_drill_equal_nmax_order_is_a_duplicate(tmp_path):
    header, first, *_ = (REPO / md.NMAX_RELPATH).read_text(encoding="ascii").splitlines()
    path = tmp_path / "nmax.csv"
    path.write_bytes(f"{header}{NL}{first}{NL}{first}{NL}".encode("ascii"))
    with pytest.raises(md.DuplicateKeyError, match="duplicate row"):
        md.load_nmax(path)


def test_t122_drill_equal_numerics_run_and_state_orders_are_admitted(tmp_path):
    for relpath, loader, change in (
        (md.NUMERICS_RUNS_RELPATH, md.load_numerics_runs, lambda row: "X" + row),
        (md.NUMERICS_STATES_RELPATH, md.load_numerics_states,
         lambda row: row.replace(",K1,", ",K01,")),
    ):
        header, first, *_ = (REPO / relpath).read_text(encoding="ascii").splitlines()
        second = change(first)
        assert second != first
        path = tmp_path / pathlib.Path(relpath).name
        path.write_bytes(f"{header}{NL}{first}{NL}{second}{NL}".encode("ascii"))
        loaded = loader(path)
        if loader is md.load_numerics_runs:
            assert len(loaded) == 2
        else:
            assert len(loaded[first.split(",")[0]]) == 2


def test_t127_drill_equal_settings_order_and_noncanonical_setting(tmp_path):
    header, first, *_ = (REPO / md.SETTINGS_RUNS_RELPATH).read_text(encoding="ascii").splitlines()
    path = tmp_path / "settings.csv"
    path.write_bytes(f"{header}{NL}{first}{NL}X{first}{NL}".encode("ascii"))
    assert len(md.load_settings_runs(path)) == 2
    fields = first.split(",")
    setting = fields[3]
    noncanonical = setting.replace("g0", "g00", 1)
    assert noncanonical != setting
    fields[0] = fields[0].replace(setting, noncanonical)
    fields[3] = noncanonical
    path.write_bytes(f"{header}{NL}{','.join(fields)}{NL}".encode("ascii"))
    with pytest.raises(md.CellError, match="unknown setting"):
        md.load_settings_runs(path)


def test_t127_settings_loader_refuses_a_compared_nuclide_outside_kept_members(monkeypatch):
    cells = md.load_cells(CELLS)
    wanted = md.settings_nuclides(cells)
    monkeypatch.setattr(md, "settings_nuclides", lambda _cells: [*wanted, (999, 999)])
    with pytest.raises(md.CellError, match="settings nuclides not kept"):
        md.load_settings_outputs(REPO, cells)


# T-128 -- the onset and certificate use the frozen increment clause


def test_t128_synthetic_onsets_and_certificates_discriminate_each_clause():
    target = Decimal("2")
    values = {u: Decimal(0) for u in md.SETTINGS_UEHLING_G0}
    assert centroids.onset_and_certificate(values, target) == (None, 100)
    growth_below_floor = dict(values)
    growth_below_floor[200] = Decimal("1")
    growth_below_floor[300] = Decimal("2.001")
    for u in range(400, 2401, 100):
        growth_below_floor[u] = growth_below_floor[300]
    onset, _cert = centroids.onset_and_certificate(growth_below_floor, target)
    assert onset is None
    growth_at_floor = dict(growth_below_floor)
    for u in range(300, 2401, 100):
        growth_at_floor[u] += Decimal("0.009")
    assert centroids.onset_and_certificate(growth_at_floor, target)[0] == 300
    growth_below_by_less = dict(growth_below_floor)
    for u in range(300, 2401, 100):
        growth_below_by_less[u] += Decimal("0.008")
    assert centroids.onset_and_certificate(growth_below_by_less, target)[0] is None
    reversed_sign = dict(values)
    reversed_sign[200] = Decimal("1")
    reversed_sign[300] = Decimal("0")
    for u in range(400, 2401, 100):
        reversed_sign[u] = Decimal("0")
    assert centroids.onset_and_certificate(reversed_sign, target)[0] == 300
    reversed_at_floor = dict(values)
    reversed_at_floor[200] = Decimal("0.01")
    assert centroids.onset_and_certificate(reversed_at_floor, target)[0] == 300
    certificate_at_target = dict(values)
    certificate_at_target[100] = target
    for u in range(200, 2401, 100):
        certificate_at_target[u] = 2 * target
    assert centroids.onset_and_certificate(certificate_at_target, target) == (None, 200)
    missing_half = dict(values)
    del missing_half[50]
    assert centroids.onset_and_certificate(missing_half, target)[0] == 50
    missing_whole = dict(values)
    del missing_whole[400]
    assert centroids.onset_and_certificate(missing_whole, target)[0] == 400
    no_certificate = {u: Decimal(3) * Decimal(u) / 100 for u in md.SETTINGS_UEHLING_G0}
    assert centroids.onset_and_certificate(no_certificate, target) == (None, None)


def test_t128_drill_reference_sign_uses_the_exact_floor_in_precision_50():
    with localcontext() as context:
        context.prec = 50
        target = Decimal("2")
        phi = target / 200
        at_floor = {u: Decimal(0) for u in md.SETTINGS_UEHLING_G0}
        at_floor[200] = phi
        assert centroids.onset_and_certificate(at_floor, target)[0] == 300
        below_floor = dict(at_floor)
        below_floor[200] = phi.next_minus()
        below_floor[300] = below_floor[200] - phi
        for u in range(400, 2401, 100):
            below_floor[u] = below_floor[300]
        assert centroids.onset_and_certificate(below_floor, target)[0] is None


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
        rows.append({"cohort": cohort, "source": source, "Z": "1", "A": "1", "dnum_keV": shift,
                     "sigma_keV": "10", "sigma_max_keV": "", "screen": "inside",
                     "screen_at_reference": "inside"})
    monkeypatch.setattr(centroids, "centroid_rows", lambda root: rows)
    monkeypatch.setattr(centroids, "margin_rows", lambda members: [])
    monkeypatch.setattr(centroids, "numerical_components",
                        lambda root, cells: ({(1, 1): (0, 0, 0, None, 100)}, []))
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
    rows = [{"cohort": cohort, "source": "Fricke1995", "Z": "1", "A": "1", "dnum_keV": shift,
             "sigma_keV": "10", "sigma_max_keV": "", "screen": "inside",
             "screen_at_reference": "inside"}
            for cohort, shift in (("labelled", "1"), ("constructed", "2"))]
    monkeypatch.setattr(centroids, "centroid_rows", lambda root: rows)
    monkeypatch.setattr(centroids, "margin_rows", lambda members: [])
    monkeypatch.setattr(centroids, "numerical_components",
                        lambda root, cells: ({(1, 1): (0, 0, 0, None, 100)}, []))
    monkeypatch.setattr(md, "load_cells", lambda path: ())
    monkeypatch.setattr(centroids, "load_intensity_ratios", lambda path, cells: {})
    assert any("above a tenth of the largest sigma on 1 of 2 rows" in line
               for line in centroids.summary_lines(REPO))


def test_t128_a_component_at_onset_or_without_a_clean_fine_reference_is_refused(monkeypatch):
    key = (4, 9)
    settings = md.SettingsOutputs({}, {}, {})
    monkeypatch.setattr(md, "load_settings_outputs", lambda root, cells: settings)
    monkeypatch.setattr(md, "settings_nuclides", lambda cells: [key])
    monkeypatch.setattr(centroids, "_targets", lambda cells: {key: Decimal("0.1")})
    q = {(4, 9, md.setting_id(0, 100)): Decimal(0),
         (4, 9, md.setting_id(0, 1000)): Decimal(1),
         (4, 9, md.setting_id(2, 1000)): Decimal("1.1")}
    monkeypatch.setattr(centroids, "_settings_q", lambda loaded: q)
    monkeypatch.setattr(centroids, "onset_and_certificate", lambda values, target: (1000, None))
    with pytest.raises(md.CellError, match="at or above onset"):
        centroids.numerical_components(REPO, ())
    monkeypatch.setattr(centroids, "onset_and_certificate", lambda values, target: (None, 100))
    del q[(4, 9, md.setting_id(2, 1000))]
    with pytest.raises(md.CellError, match="fine-grid reference is unclean"):
        centroids.numerical_components(REPO, ())
    q[(4, 9, md.setting_id(2, 1000))] = Decimal("1.1")
    del q[(4, 9, md.setting_id(0, 100))]
    with pytest.raises(md.CellError, match="numerical endpoint is unclean"):
        centroids.numerical_components(REPO, ())


def test_t128_a_derivation_failure_is_an_unclean_setting(monkeypatch):
    clean = md.SettingsRun("Be9_g0u0100", 4, 9, "g0u0100", 0, 0, "0.1", "a" * 64)
    unclean = md.SettingsRun("Be9_g0u0200", 4, 9, "g0u0200", 124, 0, "0.1", "a" * 64)
    outputs = md.SettingsOutputs({clean.run: clean, unclean.run: unclean}, {}, {})
    calls = []

    def failed(run, headers, lines):
        calls.append(run)
        raise ValueError("unreadable binding")

    monkeypatch.setattr(md, "derive_bindings", failed)
    assert centroids._settings_q(outputs) == {}
    assert calls == [clean.run]


def test_t128_screen_boundary_and_frozen_calculation_error(monkeypatch):
    sigma = Decimal("1")
    assert centroids._screen(3 * sigma, sigma) == "inside"
    assert centroids._screen(3 * sigma + Decimal("0.1"), sigma) == "outside"
    assert centroids._screen(Decimal("0.3"), sigma, Decimal("0.1")) == "inside"
    assert centroids._screen(Decimal("1"), sigma, Decimal("0.1")) == "correlation-dependent"
    assert centroids._screen(Decimal("3.1"), sigma, Decimal("0.1")) == "outside"
    assert centroids._screen(Decimal("3"), sigma, Decimal("0.1")) == "correlation-dependent"
    cells = md.load_cells(CELLS)
    zeros = {key: (Decimal(0), Decimal(0), Decimal(0), None, None)
             for key in md.settings_nuclides(cells)}
    monkeypatch.setattr(centroids, "numerical_components", lambda root, cells: (zeros, 0))
    monkeypatch.setattr(md, "SIGMA_CALC", 1)
    with pytest.raises(md.CellError, match="freezes SIGMA_CALC"):
        centroids.centroid_rows(REPO)


def test_t128_committed_components_and_certificates_match_reproduced_cells():
    cells = md.load_cells(CELLS)
    components, _unclean = centroids.numerical_components(REPO, cells)
    expected = {
        (4, 9): ('0.000076749', '0.000000731', '0.000077480', '100'),
        (6, 12): ('0.000277602', '0.000003887', '0.000281489', '300'),
        (8, 16): ('0.000714321', '0.000014440', '0.000728762', '200'),
        (11, 23): ('0.002046892', '0.000057267', '0.002104159', '300'),
        (13, 27): ('0.003573340', '0.000108296', '0.003681636', '400'),
        (14, 28): ('0.004541285', '0.000144919', '0.004686204', '300'),
        (18, 40): ('0.009833462', '0.000387986', '0.010221448', '200'),
        (21, 45): ('0.015628270', '0.000664280', '0.016292550', '200'),
        (26, 56): ('0.028503393', '0.001346820', '0.029850213', '300'),
        (29, 63): ('0.037523574', '0.001907240', '0.039430815', '300'),
        (32, 74): ('0.046517982', '0.002593741', '0.049111723', '500'),
        (38, 88): ('0.069731116', '0.004132155', '0.073863272', 'none'),
        (40, 90): ('0.078123816', '0.004709744', '0.082833560', 'none'),
        (41, 93): ('0.081419659', '0.005023699', '0.086443358', '700'),
        (44, 102): ('0.091082539', '0.005993768', '0.097076307', '600'),
        (46, 104): ('0.100082807', '0.006634152', '0.106716959', '300'),
        (46, 106): ('0.099148665', '0.006641288', '0.105789953', '200'),
        (46, 108): ('0.098206686', '0.006649298', '0.104855984', '300'),
        (46, 110): ('0.097374805', '0.006657312', '0.104032117', '200'),
        (47, 107): ('0.103651034', '0.006968027', '0.110619061', 'none'),
        (49, 115): ('0.110816451', '0.007642388', '0.118458839', 'none'),
        (53, 127): ('0.124608964', '0.009005102', '0.133614066', 'none'),
        (55, 133): ('0.131755063', '0.009686152', '0.141441215', 'none'),
        (79, 197): ('0.192017477', '0.016690678', '0.208708155', '400'),
        (81, 205): ('0.195824922', '0.017130541', '0.212955462', '200'),
        (82, 208): ('0.197049118', '0.017330812', '0.214379930', '500'),
    }
    actual = {key: (centroids._text(du), centroids._text(grid), centroids._text(total),
                    str(cert) if cert is not None else "none")
              for key, (du, grid, total, cert, _onset) in components.items()}
    assert actual == expected
    assert all(onset is None or onset > 1000 for _du, _grid, _total, _cert, onset
               in components.values())


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
    components, _unclean = centroids.numerical_components(REPO, cells)
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
        du, grid, total, certified, _onset = components[key]
        assert (row["dU_keV"], row["dgrid_keV"], row["dnum_keV"], row["u_certified"]) == (
            _nine(du), _nine(grid), _nine(total), str(certified) if certified is not None else "none")
        assert Decimal(row["dnum_keV"]) == Decimal(row["dU_keV"]) + Decimal(row["dgrid_keV"]) or (
            abs(Decimal(row["dnum_keV"]) - Decimal(row["dU_keV"]) - Decimal(row["dgrid_keV"]))
            <= Decimal("1e-9"))
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
        reference = residual + Decimal(row["dnum_keV"])
        reference_screen = ("outside" if abs(reference) > 3 * sigma else
                            "inside" if abs(reference) <= 3 * minimum else "correlation-dependent")
        assert row["screen_at_reference"] == reference_screen
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
    zero = {nuclide: (Decimal(0), Decimal(0), Decimal(0), None, None)
            for nuclide in md.settings_nuclides(changed)}
    monkeypatch.setattr(centroids, "numerical_components", lambda root, source: (zero, 0))
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
