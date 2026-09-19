"""The D3 energy tables of the ``mudirac130`` profile: their MuDirac inputs and printed outputs, the
derivation of the two tables from those printed strings, and the comparison with measured energies.

Every count here is derived at run time from the committed files; none is written down.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import re
from decimal import Decimal, localcontext

import pytest

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
# T-106 -- the comparison with the measured transition energies, re-derived
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
    tolerance three printed standard deviations and no model term."""
    out = md.load_outputs(REPO)
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
        derived = ("model_keV", "residual_keV", "dE_sigma_keV", "dE_1pct_keV", "label", "within")
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
    assert VALIDATION.read_bytes() == md.build_validation(REPO)


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
        ("the generator's version", path, r"computed by MuDirac (\d+\.\d+\.\d+)", (1,), md.MUDIRAC_VERSION),
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
        ("gated rows", path, r"Of the (\d+) gated rows", (1,), len(gated)),
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
    return pins


def test_t107_the_comparison_table_is_the_generated_block():
    block = md.render_validation_table(_committed_validation())
    text = _document_text()
    assert text.count(block) == 1


def _pin_problems(text: str) -> list[str]:
    collapsed = " ".join(text.split())
    problems = []
    for what, _path, pattern, _groups, expected in document_pins():
        hits = list(re.finditer(pattern, collapsed))
        if len(hits) != 1:
            problems.append(f"{what}: matched {len(hits)} times")
        elif expected is not None and hits[0].group(1) != str(expected):
            problems.append(f"{what}: states {hits[0].group(1)}, the committed files give {expected}")
    return problems


def test_t107_every_pin_matches_once_and_states_its_value():
    assert not _pin_problems(_document_text())
    print(f"\ndocument pins: {len(document_pins())}")


def test_t107_drill_a_changed_count_and_a_changed_table_cell_are_refused():
    text = _document_text()
    gated = sum(row["gated"] == "true" for row in _committed_validation())
    stated = f"Of the {gated} gated rows"
    mutated = _replace_once(text, stated, f"Of the {gated + 1} gated rows")
    assert any(problem.startswith("gated rows: states") for problem in _pin_problems(mutated))
    block = md.render_validation_table(_committed_validation())
    row = block.splitlines()[2]
    cell = row.split(" | ")[5]
    changed = _replace_once(text, row, row.replace(f" | {cell} | ", f" | {cell}1 | ", 1))
    assert changed.count(block) == 0
    assert any(problem.startswith("comparison table row 1:") for problem in _pin_problems(changed))
