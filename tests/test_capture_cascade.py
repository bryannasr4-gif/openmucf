"""Tests of `openmucf/g4/capture_cascade.py`, the stock muonic cascade of Geant4 in exact arithmetic; of the
`cascade` command of the stopped-muon driver; of the harvested cascades and their exact binomial test; of the
comparison with measured K-series intensities; and of the section of `BENCHMARK_STOPPED_MU.md` they generate.

Every drill calls the function under test directly with the faulty input, so no earlier guard pre-empts it.
"""

from __future__ import annotations

import ast
import copy
import csv
import functools
import hashlib
import importlib.util
import io
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_stopped_mu as tsm

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "openmucf" / "g4" / "capture_cascade.py"
DATA = REPO / "data" / "g4" / "stopped_mu"
SOURCES = REPO / "third_party" / "geant4"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cc = _load("capture_cascade_under_test", MODULE_PATH)
gen, drv = tsm.gen, tsm.drv
RULES = dict(cc.RULES)
POLICY = json.loads((DATA / "preregistration.json").read_text(encoding="utf-8"))
V1, V2 = POLICY["setup"]["versions"]
TAGS = POLICY["setup"]["tags"]
SRC1 = (SOURCES / TAGS[V1] / "G4EmCaptureCascade.cc").read_text(encoding="utf-8")
SRC2 = (SOURCES / TAGS[V2] / "G4EmCaptureCascade.cc").read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def rendered():
    return gen.document()


def fresh_inputs():
    return copy.deepcopy(gen.load_inputs())


def section(text: str) -> str:
    start = text.index("## The muonic X-ray cascade")
    return text[start : text.index("\n## ", start + 1)]


# ============================================================================================
# The module and the driver's cascade command
# ============================================================================================
def test_t1_the_rule_constants_are_read_back_from_both_vendored_sources():
    b = cc.bind_source(SRC1, SRC2)
    assert b.rules == RULES
    assert cc.read_rules(SRC2).rules == RULES
    lines = SRC1.splitlines()
    assert lines[b.lines["n_elec"] - 1].split() == ["G4int", "nElec", "=", "Z;"]
    assert "while" in lines[b.lines["loop"] - 1]
    assert "G4UniformRand" in lines[b.lines["auger_test"] - 1]
    assert "G4UniformRand" in lines[b.lines["photon_draw"] - 1]
    assert b.lines["n_elec"] < b.lines["auger_test"] < b.lines["photon_draw"] < b.lines["loop"]


def edited(text: str, needle: str, old: str, new: str) -> str:
    lines = text.splitlines(keepends=True)
    [i] = [k for k, line in enumerate(lines) if needle in line and "//" not in line.split(needle)[0]]
    assert old in lines[i]
    lines[i] = lines[i].replace(old, new, 1)
    return "".join(lines)


SOURCE_DRILLS = [
    (
        "photon weight 9 in the draw",
        "v1",
        "G4double var",
        "10.0",
        "9.0",
        "^the photon draw uses 9.0, the jump 10.0 and 10.0$",
    ),
    (
        "photon weight 9 in the jump's offset",
        "v1",
        "iLevel -= G4int",
        "var-10.0",
        "var-9.0",
        "^the photon draw uses 10.0, the jump 10.0 and 9.0$",
    ),
    (
        "start level 12",
        "v1",
        "G4int nLevel",
        "13",
        "12",
        "^the capture electron is not emitted at the start level$",
    ),
    (
        "Auger weight 1000",
        "v1",
        "pGamma + 10000.0",
        "10000.0",
        "1000.0",
        "^the Auger test compares with 10000.0 after adding 1000.0$",
    ),
    (
        "Auger count 2 in the other copy",
        "v2",
        "G4int nAuger",
        "1",
        "2",
        "^copy 1: statement auger_count0 differs once whitespace is removed$",
    ),
    (
        "cube in the other copy",
        "v2",
        "pGamma = (Z",
        "Z * Z * Z * Z",
        "Z * Z * Z",
        "^copy 1: statement power differs once whitespace is removed$",
    ),
    (
        "photon jump",
        "v2",
        "iLevel -= G4int",
        "+ 1",
        "+ 2",
        "^statement photon_jump matches 0 lines, not one$",
    ),
]


@pytest.mark.parametrize(
    "name, which, needle, old, new, match", SOURCE_DRILLS, ids=[d[0] for d in SOURCE_DRILLS]
)
def test_t1_drill_an_edited_source_is_refused(name, which, needle, old, new, match):
    s1 = edited(SRC1, needle, old, new) if which == "v1" else SRC1
    s2 = edited(SRC2, needle, old, new) if which == "v2" else SRC2
    with pytest.raises(cc.SourceMismatch, match=match):
        cc.bind_source(s1, s2)


def test_t1_drill_photon_weight_nine_in_every_photon_statement_is_refused_as_another_rule():
    s1 = SRC1.replace("10.0", "9.0")
    assert cc.read_rules(s1).rules == {**RULES, "photon_weight": 9}
    with pytest.raises(cc.SourceMismatch, match="the source states"):
        cc.bind_source(s1, SRC2)


def test_t1_drill_a_mass_number_in_the_rule_block_is_refused():
    s1 = SRC1.replace("  // Capture on 14-th level", "  G4double unused = A;", 1)
    assert s1 != SRC1
    with pytest.raises(cc.SourceMismatch, match="mass-number identifier A$"):
        cc.bind_source(s1, SRC2)
    s1 = SRC1.replace("    // case of Auger electrons", "    G4double massCheck = massA;", 1)
    assert s1 != SRC1
    with pytest.raises(cc.SourceMismatch, match="mass-number identifier massA"):
        cc.read_rules(s1)


def test_t1_drill_statements_out_of_order_are_refused():
    lines = SRC1.splitlines(keepends=True)
    i = next(k for k, line in enumerate(lines) if "G4int nAuger" in line)
    j = next(k for k, line in enumerate(lines) if "G4int nLevel" in line)
    lines[i], lines[j] = lines[j], lines[i]
    with pytest.raises(cc.SourceMismatch, match="^statement start_level is out of order$"):
        cc.read_rules("".join(lines))


def test_t1_drill_a_constant_that_is_not_an_integer_is_refused():
    with pytest.raises(cc.SourceMismatch, match="^constant 10000.5 is not an integer$"):
        cc.read_rules(SRC1.replace("10000.0", "10000.5"))


def test_t1_drill_a_copy_that_writes_a_constant_otherwise_is_refused_though_its_value_is_equal():
    other = SRC2.replace("10000.0", "10000.00")
    assert cc.read_rules(other).rules == RULES
    with pytest.raises(
        cc.SourceMismatch, match="^copy 1: statement auger_test differs once whitespace is removed$"
    ):
        cc.bind_source(SRC1, other)


def test_t1_a_copy_that_differs_in_whitespace_alone_is_bound():
    spaced = SRC2.replace("G4int nAuger = 1;", "G4int   nAuger =   1 ;")
    assert spaced != SRC2
    assert cc.bind_source(SRC1, spaced).rules == RULES


def enumerate_paths(z: int, top: int = 13, w_photon: int = 10, w_auger: int = 10000, auger: bool = True):
    """Every labelled path from level index ``top`` to the K shell, written here from the source alone: the
    expected count of each (upper n, lower n, kind) transition is the sum of the probabilities of the paths
    that take it."""
    flows: dict[tuple[int, int, str], Fraction] = {}
    a = Fraction(w_auger, z**4 + w_auger) if auger else Fraction(0)
    stack = [(top, 1, Fraction(1), ())]
    while stack:
        lev, emitted, p, hist = stack.pop()
        if lev == 0:
            for t in hist:
                flows[t] = flows.get(t, Fraction(0)) + p
            continue
        pa = a if emitted < z else Fraction(0)
        if pa:
            stack.append((lev - 1, emitted + 1, p * pa, (*hist, (lev + 1, lev, "auger"))))
        tot = w_photon + lev - 1
        for low in range(lev):
            w = w_photon if low == lev - 1 else 1
            stack.append(
                (low, emitted, p * (1 - pa) * Fraction(w, tot), (*hist, (lev + 1, low + 1, "photon")))
            )
    return flows


def same_flows(module: dict, enumerated: dict) -> bool:
    keys = set(module) | set(enumerated)
    return all(module.get(k, Fraction(0)) == enumerated.get(k, Fraction(0)) for k in keys)


@pytest.mark.parametrize("z", [1, 13, 14])
def test_t2_the_module_equals_the_path_enumeration(z):
    flows = cc.transitions(z)
    assert same_flows(flows, enumerate_paths(z))
    into_k = sum(flows[(n, 1, "photon")] for n in range(2, cc.LEVELS + 1)) + flows[(2, 1, "auger")]
    assert into_k == 1


@pytest.mark.slow
@pytest.mark.parametrize("z", [6, 82])
def test_t2_the_module_equals_the_path_enumeration_slow(z):
    assert same_flows(cc.transitions(z), enumerate_paths(z))


def test_t2_a_hand_derived_chain():
    # from level index 3 (n = 4) without Auger: n = 2 is reached directly (1/12) or through n = 3
    # (10/12 * 10/11),
    # and from n = 2 the photon goes to the K shell with certainty
    rules = {**RULES, "start_level": 3}
    want = Fraction(1, 12) + Fraction(10, 12) * Fraction(10, 11)
    assert want == Fraction(37, 44)
    assert cc.transitions(1, rules)[(2, 1, "photon")] == want
    assert enumerate_paths(1, top=3, auger=False)[(2, 1, "photon")] == want


def test_t2_z_one_is_the_no_auger_limit():
    flows = cc.transitions(1)
    assert all(v == 0 for (u, low, kind), v in flows.items() if kind == "auger")
    assert same_flows(flows, enumerate_paths(1, auger=False))


def test_t2_drill_the_enumerator_with_photon_weight_nine_differs():
    assert same_flows(cc.transitions(1), enumerate_paths(1))
    assert not same_flows(cc.transitions(1), enumerate_paths(1, w_photon=9))
    rules = {**RULES, "start_level": 5, "photon_weight": 9}
    assert same_flows(cc.transitions(13, rules), enumerate_paths(13, top=5, w_photon=9))


@pytest.mark.parametrize("z", [0, cc.Z_MAX + 1, -3])
def test_t3_z_outside_the_domain_raises(z):
    with pytest.raises(ValueError, match="overflows"):
        cc.transitions(z)


def test_t3_a_bool_or_float_z_is_refused():
    with pytest.raises(TypeError, match="z must be an int, not bool"):
        cc.transitions(True)
    with pytest.raises(TypeError, match="z must be an int, not float"):
        cc.transitions(13.0)


def test_t3_the_observables_check_their_arguments_at_their_edges():
    top = cc.LEVELS
    flows = cc.transitions(13)
    assert cc.k_line(13, 2) == flows[(2, 1, "photon")] and cc.k_line(13, top) == flows[(top, 1, "photon")]
    for bad in (1, top + 1):
        with pytest.raises(ValueError, match="outside"):
            cc.k_line(13, bad)
    ka = flows[(2, 1, "photon")]
    assert cc.high_over_ka(13, 3) == sum(flows[(n, 1, "photon")] for n in range(3, top + 1)) / ka
    assert cc.high_over_ka(13, top) == flows[(top, 1, "photon")] / ka
    for bad in (2, top + 1):
        with pytest.raises(ValueError, match="outside"):
            cc.high_over_ka(13, bad)
    with pytest.raises(ValueError, match="rules must name"):
        cc.transitions(13, {"start_level": 13})


def test_t3_every_stop_enters_the_k_shell_by_a_photon_or_an_auger_electron():
    for z in (1, 6, 13, 82):
        assert cc.auger_into_1s(z) == cc.transitions(z)[(2, 1, "auger")]
        assert cc.auger_into_1s(z) + cc.k_total(z) == 1
        assert cc.ka_over_ktot(z) == cc.k_line(z, 2) / cc.k_total(z)
        assert cc.kb_over_ka(z) == cc.k_line(z, 3) / cc.k_line(z, 2)
    assert cc.auger_into_1s(1) == 0 and cc.auger_into_1s(13) > 0


L = [0.45] + [0.5 / (i + 1) ** 2 for i in range(1, 14)]


EM = tsm.EM


def h(x: float) -> str:
    return float(x).hex()


def event(
    ev: int,
    secs: list[tuple[int, float]],
    route: str = "bound_decay",
    seed: int | None = None,
    stops: int = 1,
) -> list[str]:
    """One event: the muon's T line, its at-rest S line (``stops`` of them) with the cascade secondaries and a
    capture product (bound_decay) or the muonic atom (muonic_atom_helper), and the E line."""
    out = [f"T {ev} 1 0 13 - - {h(0.0)} {h(0.0)}\n"]
    model = EM if route == "bound_decay" else "-"
    children = [(pdg, model, e) for pdg, e in secs]
    children.append((2112, tsm.NC, 5.0) if route == "bound_decay" else (2000130270, "-", 0.0))
    for step in range(1, stops + 1):
        out.append(f"S {ev} 1 {step} {cc.CAPTURE_PROCESS[route]} {h(0.0)} {h(0.0)} {len(children)}\n")
        out += [tsm.c_line(ev, 1, k, pdg, m, e, 0.0) for k, (pdg, m, e) in enumerate(children)]
    out.append(f"E {ev} {ev if seed is None else seed} 1\n")
    return out


def legal() -> list[tuple[int, float]]:
    return [(11, L[13]), (11, L[12] - L[13]), (22, L[1] - L[12]), (22, L[0] - L[1])]


@pytest.mark.parametrize("route", sorted(cc.CAPTURE_PROCESS))
def test_t5_a_legal_chain_is_rebuilt_on_each_route(route):
    hv = cc.reconstruct(event(0, legal(), route) + event(1, legal(), route), L, route, 13)
    assert (hv.events, hv.stops, hv.chains, dict(hv.refused)) == (2, 2, 2, {})
    assert dict(hv.transitions) == {"e14->13": 2, "g13->2": 2, "g2->1": 2}
    assert (hv.augers, hv.photons) == (2, 4)
    want = hashlib.sha256(b"0 e14->13 g13->2 g2->1\n1 e14->13 g13->2 g2->1\n").hexdigest()
    assert hv.sequences_sha256 == want


CHAIN_FAULTS = [
    (
        "an Auger energy one ulp off",
        lambda s: [s[0], (11, math.nextafter(s[1][1], 1.0)), *s[2:]],
        "electron energy is no Auger energy",
    ),
    (
        "a photon matching no level",
        lambda s: [*s[:2], (22, s[2][1] * 1.5), *s[3:]],
        "photon energy matches no single transition",
    ),
    ("a missing capture electron", lambda s: s[1:], "no capture electron"),
    ("a chain short of the K shell", lambda s: s[:3], "chain does not reach the K shell"),
    ("a secondary after the K shell", lambda s: [*s, (22, L[0] - L[1])], "secondary after the K shell"),
    (
        "a cascade particle of another kind",
        lambda s: [*s[:3], (2212, L[0] - L[1])],
        "cascade secondary of PDG",
    ),
]


@pytest.mark.parametrize("name, fault, reason", CHAIN_FAULTS, ids=[f[0] for f in CHAIN_FAULTS])
def test_t5_a_chain_that_departs_from_the_rules_is_refused_with_its_reason(name, fault, reason):
    with pytest.raises(ValueError, match=reason):
        cc.chain(fault(legal()), L, 13)
    hv = cc.reconstruct(event(0, fault(legal())), L, "bound_decay", 13)
    assert (hv.stops, hv.chains, dict(hv.refused)) == (1, 0, {next(iter(hv.refused)): 1})
    assert reason in next(iter(hv.refused))
    assert not hv.transitions


def test_t5_an_event_without_a_cascade_or_without_one_stop_is_refused():
    hv = cc.reconstruct(event(0, []), L, "bound_decay", 13)
    assert dict(hv.refused) == {"no cascade": 1}
    hv = cc.reconstruct(event(0, legal(), stops=2), L, "bound_decay", 13)
    assert dict(hv.refused) == {"at-rest steps of the muon: 2": 1}
    no_stop = [line for line in event(0, legal()) if line.startswith(("T", "E"))]
    hv = cc.reconstruct(no_stop, L, "bound_decay", 13)
    assert dict(hv.refused) == {"at-rest steps of the muon: 0": 1} and hv.stops == 0


def adjacent_augers(n: int) -> list[tuple[int, float]]:
    """The capture electron and n Augers down from n = 14, each with exactly its energy, then photons
    level by level to the K shell."""
    out = [(11, L[13])]
    lev = 13
    for _ in range(n):
        out.append((11, L[lev - 1] - L[lev]))
        lev -= 1
    while lev:
        out.append((22, L[lev - 1] - L[lev]))
        lev -= 1
    return out


def test_t5_drill_an_over_cap_chain_is_refused_by_the_counter_with_every_energy_matching():
    z = 6
    over = adjacent_augers(13)
    for k, (pdg, e) in enumerate(over[1:], 1):  # every energy is exactly an Auger energy
        assert pdg == 11 and e == L[13 - k] - L[14 - k]
    with pytest.raises(ValueError, match="over the Auger cap"):
        cc.chain(over, L, z)
    edge = adjacent_augers(z - 1)  # at most Z - 1 cascade Augers: the edge is admitted
    assert sum(1 for p, _ in edge[1:] if p == 11) == z - 1
    assert cc.chain(edge, L, z)[: z - 1] == [f"e{14 - k}->{13 - k}" for k in range(z - 1)]
    with pytest.raises(ValueError, match="over the Auger cap"):
        cc.chain(adjacent_augers(z), L, z)
    assert len(cc.chain(over, L, cc.LEVELS)) == 13  # the same chain is legal where the cap cannot bind
    hv = cc.reconstruct(event(0, over), L, "bound_decay", z)
    assert dict(hv.refused) == {"over the Auger cap": 1}


def test_t5_the_helper_route_admits_only_one_muonic_atom_and_its_cascade():
    lines = event(0, legal(), "muonic_atom_helper")
    two_atoms = [
        x.replace("2000130270", "11") if x.startswith("C") and "2000130270" in x else x for x in lines
    ]
    hv = cc.reconstruct(two_atoms, L, "muonic_atom_helper", 13)
    assert dict(hv.refused) == {"at-rest step is not one muonic atom and its cascade": 1}


def test_t5_a_stream_that_breaks_the_record_structure_raises():
    lines = event(0, legal())
    with pytest.raises(ValueError, match="numbered"):
        cc.reconstruct([lines[0], lines[1], lines[3], *lines[2:]], L, "bound_decay", 13)
    with pytest.raises(ValueError, match="declares"):
        cc.reconstruct([x for i, x in enumerate(lines) if i != len(lines) - 2], L, "bound_decay", 13)
    with pytest.raises(ValueError, match="where 0 expected"):
        cc.reconstruct(event(1, legal()), L, "bound_decay", 13)
    with pytest.raises(ValueError, match="after the last E line"):
        cc.reconstruct(lines[:-1], L, "bound_decay", 13)
    with pytest.raises(ValueError, match="route"):
        cc.reconstruct(lines, L, "other", 13)


def test_t5_other_steps_and_their_secondaries_are_read_but_never_part_of_a_cascade():
    lines = event(0, legal(), "muonic_atom_helper")
    muon_step = f"S 0 1 1 muIoni {h(0.0)} {h(0.0)} 0\n"
    atom = [
        f"T 0 10 1 2000130270 muMinusAtomicCaptureAtRest - {h(0.0)} {h(0.0)}\n",
        f"S 0 10 1 MuonicAtomDecay {h(0.0)} {h(0.0)} 2\n",
        tsm.c_line(0, 10, 0, 11, "-", L[13], 0.0),
        tsm.c_line(0, 10, 1, 22, "-", L[0] - L[1], 0.0),
    ]
    stream = [
        lines[0],
        muon_step,
        *[x.replace(" 1 1 muMinus", " 1 2 muMinus") for x in lines[1:-1]],
        *atom,
        lines[-1],
    ]
    hv = cc.reconstruct(stream, L, "muonic_atom_helper", 13)
    assert (hv.stops, hv.chains, dict(hv.refused)) == (1, 1, {})
    assert dict(hv.transitions) == {"e14->13": 1, "g13->2": 1, "g2->1": 1}
    with pytest.raises(ValueError, match="^S-line declares 2 secondaries, 1 C-lines follow it$"):
        cc.reconstruct([*stream[:-2], stream[-1]], L, "muonic_atom_helper", 13)


@pytest.mark.parametrize(
    "name, child, admitted",
    [
        ("a proton beside the cascade", (2212, "-", 1.0), False),
        ("an electron of a named model", (11, "model_x", 1.0), False),
        ("a photon of a named model", (22, "model_x", 1.0), False),
        ("a second muonic atom", (2000130270, "-", 0.0), False),
        ("the smallest nucleus code as the atom", None, True),
    ],
)
def test_t5_the_helper_route_admits_one_muonic_atom_and_cascade_secondaries_alone(name, child, admitted):
    lines = event(0, legal(), "muonic_atom_helper")
    if child is None:
        stream = [x.replace(" 2000130270 ", f" {cc.NUCLEUS_PDG_MIN} ") for x in lines]
    else:
        n = int(lines[1].split()[7])
        stream = [
            lines[0],
            lines[1].rsplit(" ", 1)[0] + f" {n + 1}\n",
            *lines[2:-1],
            tsm.c_line(0, 1, n, child[0], child[1], child[2], 0.0),
            lines[-1],
        ]
    hv = cc.reconstruct(stream, L, "muonic_atom_helper", 13)
    assert (hv.chains == 1) is admitted, dict(hv.refused)
    if not admitted:
        assert dict(hv.refused) == {"at-rest step is not one muonic atom and its cascade": 1}


def test_t5_record_lines_the_grammar_does_not_know_are_refused():
    lines = event(0, legal())
    with pytest.raises(ValueError, match="^C-line not after an at-rest S-line of the muon$"):
        cc.reconstruct([lines[0], lines[2], *lines[1:]], L, "bound_decay", 13)
    with pytest.raises(ValueError, match="^empty record line$"):
        cc.reconstruct([lines[0], "\n", *lines[1:]], L, "bound_decay", 13)
    with pytest.raises(ValueError, match="^unknown record line 'X 0 1"):
        cc.reconstruct([lines[0], "X 0 1\n", *lines[1:]], L, "bound_decay", 13)
    with pytest.raises(ValueError, match="^C-line numbered 2 where 1 is next$"):
        cc.reconstruct([lines[0], lines[1], lines[2], lines[4], lines[3], *lines[5:]], L, "bound_decay", 13)
    with pytest.raises(ValueError, match="^14 levels expected$"):
        cc.reconstruct(lines, L[:13], "bound_decay", 13)


def test_t5_record_lines_after_the_last_e_line_are_refused_whichever_kind_they_are():
    lines = event(0, legal())
    stop_then_track = [*lines[:-1], f"T 0 2 1 22 - - {h(0.0)} {h(0.0)}\n"]  # a stop is open, no step is
    with pytest.raises(ValueError, match="after the last E line"):
        cc.reconstruct(stop_then_track, L, "bound_decay", 13)
    other_step = [*lines, f"S 1 4 1 hIoni {h(0.0)} {h(0.0)} 1\n", tsm.c_line(1, 4, 0, 11, "-", 0.1, 0.0)]
    with pytest.raises(ValueError, match="after the last E line"):  # a step of another track is open
        cc.reconstruct(other_step, L, "bound_decay", 13)


def test_t5_levels_are_read_from_the_config_l_line():
    text = "VERSION x\nL " + " ".join(h(x) for x in L) + "\n"
    assert cc.levels_from_config(text) == L
    with pytest.raises(ValueError):
        cc.levels_from_config("VERSION x\n")


@pytest.mark.parametrize("side", ["hi", "lo"])
def test_t7_the_certified_tail_encloses_the_exact_tail(side):
    n, p = 30, Fraction(1, 3)
    pmf = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    mode = math.floor((n + 1) * p)
    xs = range(mode, n + 1) if side == "hi" else range(0, mode + 1)
    for x in xs:
        exact = sum(pmf[x:]) if side == "hi" else sum(pmf[: x + 1])
        lo, hi = cc.binomial_tail(n, p, x, side)
        assert lo <= exact <= hi, x
        assert hi - lo <= exact * Fraction(1, 2**100) + Fraction(1, 2**590), x
    with pytest.raises(ValueError):
        cc.binomial_tail(n, p, mode - 1, "hi")


def test_t7_every_interval_primitive_rounds_its_lower_end_down_and_its_upper_end_up():
    third, big = Fraction(1, 3), Fraction(3**200 + 1, 7**90)  # not dyadic; more than the mantissa holds
    for x in (third, big, Fraction(5, 7)):
        lo, hi = cc._frac(cc._from_fraction(x, False)), cc._frac(cc._from_fraction(x, True))
        assert lo < x < hi
        assert hi - lo <= x / 2**158
    for x in (Fraction(3, 4), Fraction(6), Fraction(0)):  # dyadic values are carried exactly
        assert cc._frac(cc._from_fraction(x, False)) == x == cc._frac(cc._from_fraction(x, True))
    with pytest.raises(ValueError, match="non-negative"):
        cc._from_fraction(Fraction(-1, 3), True)
    m = (1 << 170) + 1  # a mantissa beyond the precision, with a remainder
    lo, hi = cc._frac(cc._norm(m, -3, False)), cc._frac(cc._norm(m, -3, True))
    assert lo < Fraction(m, 8) < hi
    assert cc._frac(cc._norm(1 << 170, -3, True)) == Fraction(1 << 170, 8)  # no remainder: no rounding
    a, b = cc._from_fraction(third, True), cc._from_fraction(Fraction(5, 7), True)
    xa, xb = cc._frac(a), cc._frac(b)
    for op, exact in ((cc._mul, xa * xb), (cc._add, xa + xb), (cc._div, xa / xb)):
        lo, hi = cc._frac(op(a, b, False)), cc._frac(op(a, b, True))
        assert lo < exact < hi, op.__name__
    six, three = cc._from_fraction(Fraction(6), True), cc._from_fraction(Fraction(3), True)
    assert cc._frac(cc._div(six, three, False)) == 2 == cc._frac(cc._div(six, three, True))
    zero = cc._from_fraction(Fraction(0), True)
    assert cc._frac(cc._add(zero, a, True)) == xa == cc._frac(cc._add(a, zero, False))
    assert cc._frac(cc._div(zero, a, True)) == 0
    i, j = cc.Interval.of(third), cc.Interval.of(Fraction(5, 7))
    for iv, exact in ((i * j, third * Fraction(5, 7)), (i + j, third + Fraction(5, 7)), (i.pow(5), third**5)):
        lo, hi = iv.bounds()
        assert lo < exact < hi


def test_t7_rounding_is_directed_where_only_the_remainder_shows_the_loss():
    """The interval ends must round outward also when the bits normalisation drops are zero and only the
    division's remainder (or the product's or sum's extra bits) carries the loss."""
    b = cc._from_fraction(Fraction(7), True)
    for num in range(1, 4000, 2):
        a = cc._from_fraction(Fraction(num), True)
        shift = cc._P + b[0].bit_length() - a[0].bit_length() + 1
        q, r = divmod(a[0] << shift, b[0])
        extra = q.bit_length() - cc._P
        if r and extra > 0 and q % (1 << extra) == 0:
            break
    else:
        raise AssertionError("no division whose dropped bits are zero")
    exact = Fraction(num, 7)
    assert cc._frac(cc._div(a, b, False)) < exact < cc._frac(cc._div(a, b, True))
    x = Fraction((1 << 159) + 1, 1 << 159)  # a dyadic of the full mantissa: carried exactly
    i = cc.Interval.of(x)
    assert i.bounds() == (x, x)
    lo, hi = (i * i).bounds()
    assert lo < x * x < hi
    y = Fraction(1, 1 << 300)
    lo, hi = (i + cc.Interval.of(y)).bounds()
    assert lo < x + y < hi


@pytest.mark.parametrize(
    "args, message",
    [
        ((30, Fraction(0), 3, "hi"), "^0 < p < 1 required$"),
        ((30, Fraction(1), 3, "hi"), "^0 < p < 1 required$"),
        ((30, Fraction(1, 3), -1, "lo"), "^0 <= x <= n required$"),
        ((30, Fraction(1, 3), 31, "hi"), "^0 <= x <= n required$"),
        (
            (30, Fraction(1, 3), 20, "up"),
            "^" + re.escape("side is 'hi' (P(X >= x)) or 'lo' (P(X <= x))") + "$",
        ),
    ],
)
def test_t7_the_tail_refuses_arguments_outside_its_domain(args, message):
    with pytest.raises(ValueError, match=message):
        cc.binomial_tail(*args)


def test_t7_the_tail_admits_the_edges_of_its_domain():
    n, p = 30, Fraction(1, 3)
    assert cc.binomial_tail(n, p, n, "hi")[0] > 0
    assert cc.binomial_tail(n, p, 0, "lo")[0] > 0
    assert cc.decide(n, Fraction(1, 31), 0, Fraction(1, 1000)).decision == "pass"
    with pytest.raises(ValueError, match="^0 < alpha_side < 1 required$"):
        cc.decide(n, p, 3, Fraction(1))


def test_t7_a_decision_against_its_exact_tail():
    n, p = 30, Fraction(1, 3)
    pmf = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    for x in range(n + 1):
        mode = math.floor((n + 1) * p)
        d = cc.decide(n, p, x, Fraction(1, 1000))
        if x == mode:
            assert (d.side, d.bound_kind, d.decision) == ("mode", "none", "pass")
            continue
        tail = sum(pmf[x:]) if x > mode else sum(pmf[: x + 1])
        assert d.decision == ("reject" if tail <= Fraction(1, 1000) else "pass"), x
        assert (d.bound >= tail) if d.decision == "reject" else (d.bound <= tail)


RULE = {"holds_within": "2", "tension_within": "3"}


@pytest.mark.parametrize("side", ["hi", "lo"])
def test_t7_a_long_tail_is_closed_by_the_remainder_bound_and_still_encloses_the_exact_tail(side):
    """n = 2000: the walk stops on the remainder bound long before the end of the support (the path a
    harvested cell takes), and the bounds still enclose the exact tail."""
    n, p = 2000, Fraction(1, 3)
    mode = math.floor((n + 1) * p)
    x = mode + 60 if side == "hi" else mode - 60
    ks = range(x, n + 1) if side == "hi" else range(0, x + 1)
    exact = sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in ks)
    lo, hi = cc.binomial_tail(n, p, x, side)
    assert lo <= exact <= hi and lo < hi
    assert hi - lo <= exact / 2**100


def test_t7_the_exponent_of_a_displayed_bound_is_exact_where_the_float_estimate_is_not():
    x = Fraction(999999999999999, 10**14)  # just below 10: log10 of the parts rounds the estimate up to 1
    assert math.floor(math.log10(x.numerator) - math.log10(x.denominator)) == 1
    assert cc.sig_digits(x, 6, False) == "9.99999e+00"
    assert cc.sig_digits(x, 6, True) == "1.00000e+01"


def test_t7_bounds_are_displayed_with_directed_significant_digits():
    assert (
        cc.sig_digits(Fraction(1, 1000), 6, True)
        == "1.00000e-03"
        == cc.sig_digits(Fraction(1, 1000), 6, False)
    )
    assert cc.sig_digits(Fraction(123456789, 10**12), 6, False) == "1.23456e-04"
    assert cc.sig_digits(Fraction(123456789, 10**12), 6, True) == "1.23457e-04"
    assert cc.sig_digits(Fraction(9999999, 10**11), 6, True) == "1.00000e-04"  # rounding up carries a digit
    assert cc.sig_digits(Fraction(9999999, 10**11), 6, False) == "9.99999e-05"
    assert cc.sig_digits(Fraction(10), 1, True) == "1e+01"
    for bad in (Fraction(0), Fraction(-1, 3)):
        with pytest.raises(ValueError, match="^positive values only$"):
            cc.sig_digits(bad, 6, True)


def test_t7_consumer_rows_refuse_counts_other_than_the_retained_counts():
    counts = {k: 1 for k in cc.retained_counts(cc.transitions(13))}
    counts.pop("K>=6")
    with pytest.raises(ValueError, match="^the observed counts are not the retained counts$"):
        cc.consumer_rows(
            {"version": "v", "config": "c", "target": "t"},
            counts,
            13,
            100,
            {"budget_per_cell": "1/10000", "sides": 2},
        )


def test_t7_the_budget_must_lie_strictly_between_zero_and_one():
    for a in (Fraction(0), Fraction(1)):
        with pytest.raises(ValueError, match="^0 < alpha_side < 1 required$"):
            cc.decide(30, Fraction(1, 3), 3, a)


def test_t8_sigma_squared_zero_is_admitted_and_negative_refused():
    assert cc.verdict(Fraction(1), Fraction(1), Fraction(0), RULE) == "HOLDS"
    assert cc.verdict(Fraction(2), Fraction(1), Fraction(0), RULE) == "FAILS"
    with pytest.raises(ValueError, match="^sigma\\^2 < 0$"):
        cc.verdict(Fraction(1), Fraction(1), Fraction(-1, 10**30), RULE)


def ref_row(source, target, z, phase, n_upper="", n_from="", value="1", sigma="0.1"):
    return {
        "source": source,
        "target": target,
        "Z": str(z),
        "phase": phase,
        "n_upper": n_upper,
        "n_from": n_from,
        "value": value,
        "sigma": sigma,
    }


SYNTH_RULE = {
    "ratios": {
        "sources": {"o": "printed_ratio_percent", "a": "absolute_members", "x": "other"},
        "set_of_phase": {"solid": "primary", "gas": "secondary"},
        **RULE,
    }
}


def test_t8_the_rows_follow_the_members_a_source_prints():
    rows = cc.comparison_rows([ref_row("o", "T", 13, "solid", n_from="8", value="5")], SYNTH_RULE)
    assert [(r["observable"], r["k"]) for r in rows] == [("high_over_ka", "8")]
    rows = cc.comparison_rows([ref_row("o", "T", 13, "solid", n_upper="3", value="7")], SYNTH_RULE)
    assert [(r["observable"], r["k"]) for r in rows] == [("kb_over_ka", "")]
    members = [
        ref_row("a", "G", 13, "gas", n_upper="2", value="0.9"),
        ref_row("a", "G", 13, "gas", n_upper="4", value="0.1"),
    ]
    rows = cc.comparison_rows(members, SYNTH_RULE)
    assert [r["observable"] for r in rows] == ["ka_over_ktot"] and rows[0]["set"] == "secondary"


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [ref_row("o", "T", 13, "solid", n_from="8"), ref_row("o", "T", 13, "solid", n_from="7")],
            "^more than one grouped member$",
        ),
        (
            [ref_row("o", "T", 13, "solid", n_upper="3"), ref_row("o", "T", 14, "solid", n_from="8")],
            "^o T: rows disagree on Z or phase$",
        ),
        (
            [ref_row("o", "T", 13, "solid", n_upper="3"), ref_row("o", "T", 13, "gas", n_from="8")],
            "^o T: rows disagree on Z or phase$",
        ),
        ([ref_row("x", "T", 13, "solid", n_upper="3")], "^unknown source kind 'other'$"),
    ],
)
def test_t8_a_reference_the_rule_cannot_read_is_refused(rows, message):
    with pytest.raises(ValueError, match=message):
        cc.comparison_rows(rows, SYNTH_RULE)


@pytest.mark.parametrize(
    "offset, want",
    [
        (Fraction(2), "HOLDS"),
        (Fraction(2) + Fraction(1, 10**12), "TENSION"),
        (Fraction(3), "TENSION"),
        (Fraction(3) + Fraction(1, 10**12), "FAILS"),
        (-Fraction(2), "HOLDS"),
        (-Fraction(3) - Fraction(1, 10**12), "FAILS"),
    ],
)
def test_t8_the_band_edges_are_exact(offset, want):
    sigma = Fraction(3, 7)
    d = Fraction(1, 3)
    m = d + offset * sigma
    assert (m - d) ** 2 == offset**2 * sigma**2  # the fixture sits where it is meant to
    assert cc.verdict(m, d, sigma**2, RULE) == want


def test_t8_a_synthetic_policy_moves_the_bands():
    assert (
        cc.verdict(Fraction(5), Fraction(0), Fraction(1), {"holds_within": "5", "tension_within": "6"})
        == "HOLDS"
    )
    assert cc.verdict(Fraction(5), Fraction(0), Fraction(1), RULE) == "FAILS"


def test_t8_the_correlation_extremes_and_the_independent_propagation():
    a, sa, b, sb = Fraction(8), Fraction(1, 10), Fraction(2), Fraction(1, 20)
    r, ea, eb = b / a, sa / a, sb / b
    assert cc.ratio_sigma2(a, sa, b, sb) == r * r * (ea * ea + eb * eb)
    assert cc.ratio_sigma2(a, sa, b, sb, Fraction(-1)) == r * r * (ea + eb) ** 2
    assert cc.ratio_sigma2(a, sa, b, sb, Fraction(1)) == r * r * (ea - eb) ** 2
    members = [(Fraction(8), Fraction(1, 10)), (Fraction(1), Fraction(1, 20)), (Fraction(1), Fraction(1, 50))]
    rr, conv, ind = cc.knorm_sigma2(members)
    s = Fraction(10)
    assert rr == Fraction(8, 10)
    assert conv == (Fraction(1, 10) / s) ** 2
    # R = a / (a + rest): dR/da = rest / S^2, dR/drest = -a / S^2
    assert ind == ((2 * Fraction(1, 10)) ** 2 + 8**2 * (Fraction(1, 20) ** 2 + Fraction(1, 50) ** 2)) / s**4


def test_t8_the_pull_is_printed_half_even_and_decides_nothing():
    assert cc.pull(Fraction(1), Fraction(0), Fraction(4, 9)) == "+1.50"
    assert cc.pull(Fraction(0), Fraction(1, 8), Fraction(1, 16)) == "-0.50"
    assert cc.pull(Fraction(1, 800), Fraction(0), Fraction(1, 100)) == "+0.01"


def test_t13_the_module_imports_the_standard_library_alone():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names.add((node.module or "").split(".")[0])
    names.discard("__future__")
    assert names and names <= set(sys.stdlib_module_names), sorted(names - set(sys.stdlib_module_names))


def test_t13_the_module_loads_by_path_in_an_isolated_interpreter():
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('c', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['c'] = m; spec.loader.exec_module(m)\n"
        "print('LOADED', 'openmucf' in sys.modules, m.ka_over_ktot(6) < 1)\n"
    )
    p = subprocess.run(
        [sys.executable, "-I", "-c", code, str(MODULE_PATH)],
        capture_output=True,
        text=True,
        cwd=str(REPO / "cpp"),
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["LOADED", "False", "True"]


def test_t13_the_cascade_parser_takes_the_run_commands_options():
    ap = drv.parser()
    run = {a.dest for a in ap._subparsers._group_actions[0].choices["run"]._actions}
    cas = {a.dest for a in ap._subparsers._group_actions[0].choices["cascade"]._actions}
    assert run <= cas
    assert cas - run == {"out", "transitions", "rule"}


def test_t13_the_cascade_argv_is_the_run_argv_with_the_harvests_seed_base():
    pol = tsm.synthetic_policy()
    hv = {"events": 7, "seed_base": 123}
    a = drv.cascade_argv("/bin/x", pol, hv, config="bertini", target="Al-27", out="/o")
    b = drv.harness_argv("/bin/x", pol, config="bertini", target="Al-27", offset=0, events=7, out="/o")
    i = b.index("--seed-base") + 1
    assert a[:i] + a[i + 1 :] == b[:i] + b[i + 1 :] and a[i] == "123" and b[i] != "123"


class StandIn:
    """The harness for the cascade command: DATASET lines of the binding, then on a run ``records``, a
    config.txt
    with the MATERIAL and L lines, and the environment each call received."""

    def __init__(self, records: list[str], datasets: str, material: str = "MATERIAL 1 1 13 27 0x1p+0"):
        self.records, self.datasets, self.material, self.envs = records, datasets, material, []

    def run(self, argv, **kwargs):
        self.envs.append(kwargs.get("env"))
        if "--print-datasets" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=self.datasets, stderr="")
        out = Path(argv[argv.index("--out") + 1])
        (out / "records-0.txt").write_text("".join(self.records), encoding="ascii", newline="\n")
        config = f"{self.material}\nL {' '.join(h(x) for x in L)}\n{self.datasets}"
        (out / "config.txt").write_text(config, encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)


def cascade_setup(tmp_path, monkeypatch, records, cells=None, **kwargs):
    pol = tsm.synthetic_policy()
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps(pol), encoding="utf-8")
    rule = tmp_path / "rule.json"
    cells = cells or [{"version": "A", "config": "bertini", "target": "Al-27"}]
    rule.write_text(
        json.dumps({"harvest": {"cells": cells, "events": 2, "seed_base": 100}}), encoding="utf-8"
    )
    fake = StandIn(records, tsm.dataset_text(drv.binding(pol, "A", "/data")), **kwargs)
    monkeypatch.setattr(drv, "subprocess", SimpleNamespace(run=fake.run, STDOUT=subprocess.STDOUT))
    binary = tmp_path / "g4muonic_transport"
    binary.write_bytes(b"a stand-in binary")
    argv = [
        "cascade",
        "--binary",
        str(binary),
        "--install",
        "/inst",
        "--data",
        "/data",
        "--work",
        str(tmp_path / "work"),
        "--prereg",
        str(prereg),
        "--rule",
        str(rule),
        "--version",
        "A",
        "--config",
        "bertini",
        "--target",
        "Al-27",
        "--out",
        str(tmp_path / "h.csv"),
        "--transitions",
        str(tmp_path / "t.csv"),
    ]
    return fake, argv


def test_t13_the_cascade_command_runs_a_cell_through_the_stand_in(tmp_path, monkeypatch, capsys):
    records = event(0, legal(), seed=100) + event(1, legal(), seed=101)
    fake, argv = cascade_setup(tmp_path, monkeypatch, records)
    assert drv.main(argv) == 0
    [row] = gen.read_csv(tmp_path / "h.csv")
    assert (row["events"], row["stops"], row["chains"], row["refused"], row["augers"], row["photons"]) == (
        "2",
        "2",
        "2",
        "0",
        "2",
        "4",
    )
    assert row["binary_sha256"] == hashlib.sha256(b"a stand-in binary").hexdigest()
    trans = gen.read_csv(tmp_path / "t.csv")
    assert [t["transition"] for t in trans] == cc.transition_keys()
    assert {t["transition"]: t["stops"] for t in trans if t["stops"] != "0"} == {
        "e14->13": "2",
        "g13->2": "2",
        "g2->1": "2",
    }
    assert not (tmp_path / "work/A/bertini/Al-27/cascade/records-0.txt").exists()
    want = drv.environment(
        tsm.synthetic_policy(),
        dict(drv.os.environ),
        version="A",
        config="bertini",
        install="/inst",
        data="/data",
    )
    assert fake.envs == [want, want]  # the run command's environment, for both calls
    assert "CASCADE A bertini Al-27: events 2 stops 2 chains 2 refused 0" in capsys.readouterr().out


@pytest.mark.parametrize(
    "name, records, kwargs, match",
    [
        (
            "a refused chain",
            lambda: event(0, legal()[1:], seed=100) + event(1, legal(), seed=101),
            {},
            "refused stops",
        ),
        ("a wrong seed", lambda: event(0, legal(), seed=100) + event(1, legal(), seed=7), {}, "not seeded"),
        ("a short run", lambda: event(0, legal(), seed=100), {}, "2 expected"),
        (
            "another isotope",
            lambda: event(0, legal(), seed=100) + event(1, legal(), seed=101),
            {"material": "MATERIAL 1 1 13 28 0x1p+0"},
            "are not the single isotope Z = 13, A = 27",
        ),
        ("a malformed record", lambda: [*event(0, legal(), seed=100)[:-1], "E 0 100\n"], {}, "E-line with"),
    ],
)
def test_t13_the_cascade_command_refuses_a_bad_run(
    tmp_path, monkeypatch, capsys, name, records, kwargs, match
):
    _, argv = cascade_setup(tmp_path, monkeypatch, records(), **kwargs)
    assert drv.main(argv) == 2
    assert match in capsys.readouterr().out
    assert not (tmp_path / "h.csv").exists()


def test_t13_every_option_of_the_cascade_command_is_required_but_prereg_and_rule():
    sub = drv.parser()._subparsers._group_actions[0].choices["cascade"]
    required = {a.dest for a in sub._actions if a.required}
    assert required == {
        "binary",
        "install",
        "data",
        "work",
        "version",
        "config",
        "target",
        "out",
        "transitions",
    }


def test_t13_the_cascade_command_calls_the_harness_as_the_run_command_does(tmp_path, monkeypatch):
    calls = []
    fake, argv = cascade_setup(
        tmp_path, monkeypatch, event(0, legal(), seed=100) + event(1, legal(), seed=101)
    )
    original = fake.run

    def spy(a, **kwargs):
        calls.append({k: v for k, v in kwargs.items() if k not in ("env", "stdout")})
        return original(a, **kwargs)

    monkeypatch.setattr(drv, "subprocess", SimpleNamespace(run=spy, STDOUT=subprocess.STDOUT))
    assert drv.main(argv) == 0
    assert calls[0] == {"capture_output": True, "text": True, "check": True}
    assert calls[1] == {"stderr": subprocess.STDOUT}
    row = json.loads((tmp_path / "work/A/bertini/Al-27/cascade/harvest.json").read_text(encoding="utf-8"))
    assert list(row) == sorted(row) and row["chains"] == 2
    argv[argv.index("--work") + 1] = str(tmp_path / "work2")
    assert drv.main(argv) == 0  # a second cell appends below the header
    text = (tmp_path / "h.csv").read_text(encoding="utf-8").splitlines()
    assert text[0] == ",".join(drv.HARVEST_COLUMNS) and len(text) == 3 and text[1] == text[2]


@pytest.mark.parametrize(
    "name, setup, match",
    [
        ("an existing cascade directory", "exists", "^REFUSED: cascade directory exists"),
        ("a harness that fails", "rc", "^REFUSED: harness rc 3"),
        ("two worker record files", "files", "^REFUSED: expected one worker record file, found"),
        (
            "records out of the structure",
            "structure",
            "^REFUSED: records not readable as cascades: C-line not after",
        ),
    ],
)
def test_t13_the_cascade_command_refuses_a_run_it_cannot_read(
    tmp_path, monkeypatch, capsys, name, setup, match
):
    lines = event(0, legal(), seed=100) + event(1, legal(), seed=101)
    if setup == "structure":
        lines = [lines[0], lines[2], *lines[1:]]
    fake, argv = cascade_setup(tmp_path, monkeypatch, lines)
    original = fake.run
    if setup == "exists":
        (tmp_path / "work/A/bertini/Al-27/cascade").mkdir(parents=True)
    if setup in ("rc", "files"):

        def run(a, **kwargs):
            out = original(a, **kwargs)
            if "--print-datasets" in a:
                return out
            if setup == "files":
                (Path(a[a.index("--out") + 1]) / "records-1.txt").write_text("", encoding="ascii")
                return out
            return subprocess.CompletedProcess(a, 3)

        monkeypatch.setattr(drv, "subprocess", SimpleNamespace(run=run, STDOUT=subprocess.STDOUT))
    assert drv.main(argv) == 2
    assert re.search(match, capsys.readouterr().out, re.M)


def test_t13_the_cascade_command_refuses_a_cell_outside_the_rule_and_a_natural_target(
    tmp_path, monkeypatch, capsys
):
    _, argv = cascade_setup(
        tmp_path, monkeypatch, [], cells=[{"version": "A", "config": "precompound", "target": "Al-27"}]
    )
    assert drv.main(argv) == 2
    assert "not a cell of the rule's harvest" in capsys.readouterr().out
    _, argv = cascade_setup(
        tmp_path, monkeypatch, [], cells=[{"version": "A", "config": "bertini", "target": "natSi"}]
    )
    argv[argv.index("Al-27")] = "natSi"
    assert drv.main(argv) == 2
    assert "not a single-isotope target" in capsys.readouterr().out


# ============================================================================================
# The committed reference, rule and harvest, the exact binomial test of the harvested counts and the
# comparison
# ============================================================================================
REFERENCE_COLUMNS = (
    "source",
    "citation",
    "locator",
    "target",
    "Z",
    "phase",
    "quantity",
    "n_upper",
    "n_from",
    "value",
    "sigma",
    "note",
)


PINS = {
    "cascade_reference.csv": "09ad1807d7047b84743423d829fc7b62ff22b791f97ae1e48ceb0e9b865822e2",
    "cascade_rule.json": "cb62a783563b72519bb91d0915265e73514e96c79f2ce1d761bba6864aba0876",
    "cascade_harvest.csv": "234247d1765c3fd42ac9306c49266d28d847b3cd7f10362f26860ca23c3e77db",
    "cascade_transitions.csv": "ead93ad916a660fea1ef474bb2c4be9e65319a41818ca282abd0d588a3841abd",
}


@pytest.mark.parametrize("name", sorted(PINS))
def test_t4_the_committed_file_is_the_pinned_file(name):
    assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == PINS[name]


def reference() -> list[dict[str, str]]:
    return gen.read_csv(DATA / "cascade_reference.csv")


def test_t4_the_reference_schema_and_its_normalisations():
    rows = reference()
    with open(DATA / "cascade_reference.csv", encoding="utf-8", newline="") as f:
        assert tuple(next(csv.reader(f))) == REFERENCE_COLUMNS
    assert {r["source"] for r in rows} == {"orth1982", "ensdf_measday2007", "kirch1999"}
    for r in rows:
        assert bool(r["n_upper"]) != bool(r["n_from"]), r
        assert Fraction(r["value"]) > 0 and Fraction(r["sigma"]) > 0
    kirch: dict[str, Fraction] = {}
    ensdf: dict[str, Fraction] = {}
    for r in rows:
        if r["source"] == "kirch1999":
            kirch[r["target"]] = kirch.get(r["target"], Fraction(0)) + Fraction(r["value"])
        if r["source"] == "ensdf_measday2007":
            ensdf[r["target"]] = ensdf.get(r["target"], Fraction(0)) + Fraction(r["value"])
    assert len(kirch) == 7 and all(abs(s - 1) <= Fraction(1, 1000) for s in kirch.values()), kirch
    assert [cc.fixed(s, 2) for s in ensdf.values()] == ["99.98", "99.96"]
    [si] = {r["target"] for r in rows if r["source"] == "ensdf_measday2007" and r["Z"] == "14"}
    assert si == "natural silicon"


def test_t4_the_rule_was_fixed_in_order_with_its_post_data_part_marked():
    rule = json.loads((DATA / "cascade_rule.json").read_text(encoding="utf-8"))
    parts = rule["fixed"]
    assert [p["part"] for p in parts] == ["ratios", "k_normalised", "harvest"]
    assert [p["post_data"] for p in parts] == [False, True, False]
    times = [p["fixed_utc"] for p in parts]
    assert times == sorted(times) and len(set(times)) == len(times)
    assert parts[1]["relation_to_data"].startswith("fixed after data")


def test_t6_the_harvest_is_the_rules_cells_each_one_stop_per_event_and_refused_zero():
    inp = fresh_inputs()
    observed = gen.cascade_harvest(inp)
    rule = inp.cascade.rule["harvest"]
    assert list(observed) == [(c["version"], c["config"], c["target"]) for c in rule["cells"]]
    builds = {b["tag"]: b["rebuilt_binary_sha256"] for b in inp.builds}
    for r in inp.cascade.harvest:
        assert r["refused"] == "0" and r["stops"] == r["events"] == str(rule["events"])
        assert r["binary_sha256"] == builds[TAGS[r["version"]]]
    keys = cc.transition_keys()
    assert len(keys) == len(set(keys))
    for counts in observed.values():
        assert set(counts) == set(cc.retained_counts(cc.transitions(1)))
        for k in cc.UNIONS:
            assert counts[f"K>={k}"] == sum(counts[f"g{n}->1"] for n in range(k, cc.LEVELS + 1))


def _set(rows, cell_index, field, value):
    rows[cell_index][field] = value


HARVEST_FAULTS: list[tuple[str, Callable, str]] = [
    ("cells out of order", lambda c: c.harvest.insert(0, c.harvest.pop(1)), "rule's cells"),
    ("events", lambda c: _set(c.harvest, 0, "events", "99999"), "events"),
    ("seed base", lambda c: _set(c.harvest, 0, "seed_base", "1"), "seed_base"),
    ("route", lambda c: _set(c.harvest, 0, "route", "muonic_atom_helper"), "route"),
    ("Z", lambda c: _set(c.harvest, 0, "Z", "7"), "Z"),
    ("binary", lambda c: _set(c.harvest, 0, "binary_sha256", "0" * 64), "binary_sha256"),
    ("a refused stop", lambda c: _set(c.harvest, 0, "refused", "1"), "refused"),
    ("stops", lambda c: _set(c.harvest, 0, "stops", "99999"), "stops"),
    ("chains", lambda c: _set(c.harvest, 0, "chains", "99999"), "chains"),
    ("transition order", lambda c: c.transitions.insert(0, c.transitions.pop(1)), "transitions in its order"),
    (
        "a missing cell",
        lambda c: c.transitions.__delitem__(slice(0, len(cc.transition_keys()))),
        "rule's cells",
    ),
    ("a count above the stops", lambda c: _set(c.transitions, 0, "stops", "100001"), "outside 0..stops"),
    (
        "entries into the K shell",
        lambda c: _set(c.transitions, 0, "stops", str(int(c.transitions[0]["stops"]) + 1)),
        "entries into the K shell",
    ),
    ("photon total", lambda c: _set(c.harvest, 0, "photons", "1"), "totals"),
]


def test_t6_a_cell_at_the_edges_of_its_counts_is_admitted():
    inp = fresh_inputs()
    first = inp.cascade.harvest[0]
    stops = first["stops"]
    for t in inp.cascade.transitions:
        if gen.cell_of(t) == gen.cell_of(first):
            t["stops"] = stops if t["transition"] == "g2->1" else "0"
    first["augers"], first["photons"] = "0", stops
    observed = gen.cascade_harvest(inp)[gen.cell_of(first)]
    assert observed["g2->1"] == int(stops) and observed["g3->1"] == 0


@pytest.mark.parametrize(
    "name, fault, match",
    [
        (
            "cells out of order",
            lambda c: c.harvest.insert(0, c.harvest.pop(1)),
            "^cascade_harvest.csv does not hold the rule's cells in the rule's order$",
        ),
        (
            "a missing transitions cell",
            lambda c: c.transitions.__delitem__(slice(0, len(cc.transition_keys()))),
            "^cascade_transitions.csv does not hold the rule's cells$",
        ),
        (
            "transition order",
            lambda c: c.transitions.insert(0, c.transitions.pop(1)),
            ": the transition rows are not the module's transitions in its order$",
        ),
        (
            "totals",
            lambda c: _set(c.harvest, 0, "photons", "1"),
            ": Auger and photon totals are not the sums of the transition counts$",
        ),
    ],
)
def test_t6_each_refusal_names_its_clause(name, fault, match):
    inp = fresh_inputs()
    fault(inp.cascade)
    with pytest.raises(ValueError, match=match):
        gen.cascade_harvest(inp)


@pytest.mark.parametrize("name, fault, match", HARVEST_FAULTS, ids=[f[0] for f in HARVEST_FAULTS])
def test_t6_drill_each_admission_clause_refuses_its_fault(name, fault, match):
    inp = fresh_inputs()
    fault(inp.cascade)
    with pytest.raises(ValueError, match=match):
        gen.cascade_harvest(inp)


HARVEST_RULE = json.loads((DATA / "cascade_rule.json").read_text(encoding="utf-8"))["harvest"]


N_COUNTS = len(cc.transition_keys()) + len(cc.UNIONS)


A_SIDE = cc.alpha_side(HARVEST_RULE, N_COUNTS)


def test_t7_every_committed_cell_passes_and_the_consumer_file_is_its_regeneration():
    inp = fresh_inputs()
    rows = gen.cascade_consumer(inp)
    assert (
        cc.csv_text(cc.CONSUMER_COLUMNS, rows).encode("utf-8") == (DATA / "cascade_consumer.csv").read_bytes()
    )
    committed = gen.read_csv(DATA / "cascade_consumer.csv")
    assert tuple(committed[0]) == cc.CONSUMER_COLUMNS
    assert gen.consumer_problems(committed) == []
    assert len(committed) == N_COUNTS * len(HARVEST_RULE["cells"])
    assert {r["decision"] for r in committed} == {"pass"}
    assert {r["budget"] for r in committed} == {f"{A_SIDE.numerator}/{A_SIDE.denominator}"}


CONSUMER_FAULTS = [
    ("lower on a reject", {"decision": "reject"}),
    ("lower at the budget", {"bound_value": "{budget}"}),
    ("upper on a pass", {"bound_kind": "upper"}),
    ("none off the mode", {"bound_kind": "none", "bound_value": ""}),
    ("a value on none", {"side": "mode", "bound_kind": "none"}),
    ("an unknown kind", {"bound_kind": "exact"}),
]


@pytest.mark.parametrize("name, change", CONSUMER_FAULTS, ids=[f[0] for f in CONSUMER_FAULTS])
def test_t7_drill_each_consumer_row_rule_is_checked(name, change):
    rows = gen.read_csv(DATA / "cascade_consumer.csv")
    [row] = [r for r in rows if r["bound_kind"] == "lower"][:1]
    for k, v in change.items():
        row[k] = v.format(budget=row["budget"])
    assert gen.consumer_problems([row])


def test_t7_the_upper_rows_of_the_consumer_schema_are_checked_at_their_edges():
    row = dict(gen.read_csv(DATA / "cascade_consumer.csv")[0])
    up = dict(row, side="hi", bound_kind="upper", decision="reject", bound_value=row["budget"])
    assert gen.consumer_problems([up]) == []  # a reject at exactly the budget is a reject
    for change in ({"decision": "pass"}, {"side": "mode"}, {"bound_value": "1/1000"}):
        assert gen.consumer_problems([dict(up, **change)]) != []
    assert "bound_kind 'exact'" in gen.consumer_problems([dict(row, bound_kind="exact")])[0]
    assert "columns" in gen.consumer_problems([{**{k: row[k] for k in reversed(list(row))}}])[0]


def test_t7_alpha_side_is_the_cell_budget_over_the_counts_and_sides():
    assert len(cc.retained_counts(cc.transitions(13))) == N_COUNTS
    assert A_SIDE == Fraction(1, 10000) / (N_COUNTS * 2) == Fraction(1, 2140000)


def fixture(z: int) -> dict[str, int]:
    return {k: round(HARVEST_RULE["events"] * p) for k, p in cc.retained_counts(cc.transitions(z)).items()}


def rejected(z: int, counts: dict[str, int]) -> list[str]:
    cell = {"version": "v", "config": "c", "target": "t"}
    rows = cc.consumer_rows(cell, counts, z, HARVEST_RULE["events"], HARVEST_RULE)
    return [r["count"] for r in rows if r["decision"] == "reject"]


def beyond(z: int, name: str, x: int) -> bool:
    """The certified tail of count ``name`` at x is at most the budget."""
    p = cc.retained_counts(cc.transitions(z))[name]
    n = HARVEST_RULE["events"]
    side = "hi" if x > math.floor((n + 1) * p) else "lo"
    return cc.binomial_tail(n, p, x, side)[1] <= A_SIDE


TARGET_Z = sorted(
    {
        int(t["arguments"][1])
        for t in POLICY["setup"]["targets"]
        if t["target"] in {c["target"] for c in HARVEST_RULE["cells"]}
    }
)


@pytest.mark.parametrize("z", TARGET_Z)
def test_t7a_the_kb_shift_fixture_is_rejected_after_its_shift_is_proven_beyond_the_budget(z):
    s = HARVEST_RULE["drills"]["kb_shift"]
    base = fixture(z)
    assert rejected(z, base) == []  # every unshifted count proven inside
    f = dict(base, **{"g2->1": base["g2->1"] - s, "g3->1": base["g3->1"] + s})
    assert beyond(z, "g3->1", f["g3->1"])
    assert "g3->1" in rejected(z, f)


@pytest.mark.parametrize("z", TARGET_Z)
def test_t7_the_intra_n5_redistribution_fixture_is_rejected(z):
    s = HARVEST_RULE["drills"]["intra_shift"]
    base = fixture(z)
    f = dict(base, **{"g5->1": base["g5->1"] - s})
    for k in cc.UNIONS:
        f[f"K>={k}"] = base[f"K>={k}"] + s
    assert beyond(z, "g5->1", f["g5->1"])
    assert "g5->1" in rejected(z, f)


@pytest.mark.parametrize("z", TARGET_Z)
def test_t7_a_single_stop_moved_in_the_fixture_is_accepted(z):
    base = fixture(z)
    f = dict(base, **{"g2->1": base["g2->1"] - 1, "g3->1": base["g3->1"] + 1})
    assert not beyond(z, "g3->1", f["g3->1"]) and not beyond(z, "g2->1", f["g2->1"])
    assert rejected(z, f) == []


def test_t7_drill_the_real_counts_against_photon_weight_nine_reject_every_cell():
    inp = fresh_inputs()
    weight = HARVEST_RULE["drills"]["photon_weight_drill"]
    rows = gen.cascade_consumer(inp, {**RULES, "photon_weight": weight})
    cells = {gen.cell_of(r) for r in rows if r["decision"] == "reject"}
    assert cells == {gen.cell_of(r) for r in inp.cascade.harvest}
    assert len(cells) == len(HARVEST_RULE["cells"])


def test_t8_the_committed_comparison_is_its_regeneration_with_sixty_four_rows():
    inp = fresh_inputs()
    rows = gen.cascade_comparison(inp)
    text = cc.csv_text(cc.COMPARISON_COLUMNS, rows).encode("utf-8")
    assert text == (DATA / "cascade_comparison.csv").read_bytes()
    pairs = {(r["source"], r["target"], r["observable"], r["k"]) for r in rows}
    assert len(pairs) == len(rows)
    sources = {r["source"] for r in reference()}
    assert {r["source"] for r in rows} == sources
    assert all(r["sigma_independent"] for r in rows if r["observable"] == "ka_over_ktot")
    assert all(
        bool(r["verdict_rho_minus1"]) == (r["source"] != "orth1982" and r["observable"] != "ka_over_ktot")
        for r in rows
    )


def test_t9_compare_writes_the_cascade_files_byte_for_byte(tmp_path, monkeypatch):
    out = {}

    def capture(path, fields, rows):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        out[Path(path).name] = stream.getvalue().encode("utf-8")

    monkeypatch.setattr(gen, "write_csv", capture)
    monkeypatch.setattr(gen, "DOCUMENT", tmp_path / "BENCHMARK_STOPPED_MU.md")
    gen.compare()
    for name in ("comparison.csv", "verdicts.csv", "cascade_comparison.csv", "cascade_consumer.csv"):
        assert out[name] == (DATA / name).read_bytes(), name
    assert (tmp_path / "BENCHMARK_STOPPED_MU.md").read_bytes() == (
        REPO / "BENCHMARK_STOPPED_MU.md"
    ).read_text(encoding="utf-8").encode("utf-8")
