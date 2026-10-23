"""Tests of `openmucf/g4/capture_cascade.py`, the stock muonic cascade of Geant4 in exact arithmetic; of the
`cascade` command of the stopped-muon driver; of the harvested cascades and their exact binomial test; of the
comparison with measured K-series intensities; and of the section of `BENCHMARK_STOPPED_MU.md` they generate.

Every drill calls the function under test directly with the faulty input, so no earlier guard pre-empts it.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import csv
import functools
import hashlib
import importlib.util
import io
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from decimal import Decimal
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
    with pytest.raises(cc.DisplayDomainError, match="^z must be an int in .*, not bool$"):
        cc.transitions(True)
    with pytest.raises(cc.DisplayDomainError, match="^z must be an int in .*, not float$"):
        cc.transitions(13.0)


def test_t3_the_observables_check_their_arguments_at_their_edges():
    top = cc.LEVELS
    flows = cc.transitions(13)
    assert cc.k_line(13, 2) == flows[(2, 1, "photon")] and cc.k_line(13, top) == flows[(top, 1, "photon")]
    for bad in (1, top + 1):
        with pytest.raises(cc.DisplayDomainError, match=re.escape(f"n must be an int in [2, {top}]")):
            cc.k_line(13, bad)
    ka = flows[(2, 1, "photon")]
    assert cc.high_over_ka(13, 3) == sum(flows[(n, 1, "photon")] for n in range(3, top + 1)) / ka
    assert cc.high_over_ka(13, top) == flows[(top, 1, "photon")] / ka
    for bad in (2, top + 1):
        with pytest.raises(cc.DisplayDomainError, match=re.escape(f"k must be an int in [3, {top}]")):
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
        ((30, Fraction(0), 3, "hi"), "^" + re.escape("p must be an int or a Fraction in (0, 1)") + "$"),
        ((30, Fraction(1), 3, "hi"), "^" + re.escape("p must be an int or a Fraction in (0, 1)") + "$"),
        ((30, Fraction(1, 3), -1, "lo"), "^" + re.escape("x must be an int in [0, 30]") + "$"),
        ((30, Fraction(1, 3), 31, "hi"), "^" + re.escape("x must be an int in [0, 30]") + "$"),
        (
            (30, Fraction(1, 3), 20, "up"),
            "^" + re.escape("side is 'hi' (P(X >= x)) or 'lo' (P(X <= x))") + "$",
        ),
    ],
)
def test_t7_the_tail_refuses_arguments_outside_its_domain(args, message):
    with pytest.raises(cc.DisplayDomainError, match=message):
        cc.binomial_tail(*args)


def test_t7_the_tail_admits_the_edges_of_its_domain():
    n, p = 30, Fraction(1, 3)
    assert cc.binomial_tail(n, p, n, "hi")[0] > 0
    assert cc.binomial_tail(n, p, 0, "lo")[0] > 0
    assert cc.decide(n, Fraction(1, 31), 0, Fraction(1, 1000)).decision == "pass"
    with pytest.raises(
        cc.DisplayDomainError, match=re.escape("alpha_side must be an int or a Fraction in (0, 1)")
    ):
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
        with pytest.raises(cc.DisplayDomainError, match="^x must be an int or a Fraction > 0$"):
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
        with pytest.raises(
            cc.DisplayDomainError, match=re.escape("alpha_side must be an int or a Fraction in (0, 1)")
        ):
            cc.decide(30, Fraction(1, 3), 3, a)


def test_t8_sigma_squared_zero_is_admitted_and_negative_refused():
    assert cc.verdict(Fraction(1), Fraction(1), Fraction(0), RULE) == "HOLDS"
    assert cc.verdict(Fraction(2), Fraction(1), Fraction(0), RULE) == "FAILS"
    with pytest.raises(cc.DisplayDomainError, match="^sigma2 must be an int or a Fraction >= 0$"):
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
    assert gen.consumer_problems(committed, inp) == []
    assert len(committed) == N_COUNTS * len(HARVEST_RULE["cells"])
    assert {r["decision"] for r in committed} == {"pass"}
    assert {parse_exact(r, "budget") for r in committed} == {A_SIDE}


def _bump(text: str) -> str:
    return int_text_of(parse_int_text(text) + 1)


#: Adverse edits of one row of the committed consumer table: each must make the exact check fail.
CONSUMER_FAULTS = [
    ("a flipped decision", lambda r: {"decision": "reject"}),
    ("a corrupted bound numerator", lambda r: {"bound_num": _bump(r["bound_num"])}),
    ("a corrupted bound denominator", lambda r: {"bound_den": _bump(r["bound_den"])}),
    ("a corrupted p numerator", lambda r: {"p_num": _bump(r["p_num"])}),
    ("a corrupted p denominator", lambda r: {"p_den": _bump(r["p_den"])}),
    ("a corrupted budget", lambda r: {"budget_den": _bump(r["budget_den"])}),
    ("a zero denominator", lambda r: {"bound_den": "0"}),
    ("a negative denominator", lambda r: {"p_den": "-" + r["p_den"]}),
    (
        "an unreduced fraction",
        lambda r: {"bound_num": r["bound_num"] + "0", "bound_den": r["bound_den"] + "0"},
    ),
    ("a leading zero", lambda r: {"observed": "0" + r["observed"]}),
    ("a decimal in an int column", lambda r: {"n": r["n"] + ".0"}),
    ("a corrupted observed count", lambda r: {"observed": _bump(r["observed"])}),
    ("a corrupted n", lambda r: {"n": _bump(r["n"])}),
    ("a corrupted mode", lambda r: {"mode": _bump(r["mode"])}),
    ("another count's p", lambda r: {"count": "g3->1"}),
    ("upper on a pass", lambda r: {"bound_kind": "upper"}),
    ("none off the mode", lambda r: {"bound_kind": "none", "bound_num": "", "bound_den": ""}),
    ("an unknown kind", lambda r: {"bound_kind": "exact"}),
    ("a missing bound", lambda r: {"bound_num": ""}),
]


@pytest.mark.parametrize("name, change", CONSUMER_FAULTS, ids=[f[0] for f in CONSUMER_FAULTS])
def test_t7_drill_each_consumer_row_rule_is_checked(name, change):
    rows = gen.read_csv(DATA / "cascade_consumer.csv")
    [row] = [r for r in rows if r["bound_kind"] == "lower" and r["count"] == "g2->1"][:1]
    assert gen.consumer_problems([row], fresh_inputs()) == []
    assert gen.consumer_problems([{**row, **change(row)}], fresh_inputs())


def test_t7_the_display_columns_of_the_consumer_table_are_never_read():
    rows = gen.read_csv(DATA / "cascade_consumer.csv")
    edited = [{**r, "expected_display": "-1", "bound_display": "garbage"} for r in rows[:5]]
    assert gen.consumer_problems(edited, fresh_inputs()) == []


def test_t7_the_upper_rows_of_the_consumer_schema_are_checked_at_their_edges():
    rows = cc.consumer_rows(CELL, conserved(20000), 13, 20000, HARVEST_RULE)
    [up] = [r for r in rows if r["count"] == "g2->1"]
    assert (up["bound_kind"], up["decision"]) == ("upper", "reject") and cc.consumer_problems(
        [up], A_SIDE
    ) == []
    for change in ({"decision": "pass"}, {"side": "mode"}, {"bound_kind": "lower"}, {"bound_num": "1"}):
        assert cc.consumer_problems([dict(up, **change)]) != []
    assert "columns" in cc.consumer_problems([{**{k: up[k] for k in reversed(list(up))}}])[0]
    assert cc.consumer_problems([up], A_SIDE / 2)  # a budget that is not the rule's


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


# ============================================================================================
# The section of BENCHMARK_STOPPED_MU.md
# ============================================================================================
def test_t1_the_chapter_prints_the_lines_where_the_statements_are():
    b = cc.bind_source(SRC1, SRC2)
    text = section(rendered().text)
    assert f"lines {b.lines['n_elec']}–{b.lines['loop']}" in text
    assert f"(line {b.lines['auger_test']})" in text
    assert f"(line {b.lines['capture_electron']})" in text
    assert f"(lines {b.lines['photon_draw']}–{b.lines['photon_clamp']})" in text
    assert f"(lines {b.lines['n_elec']} and {b.lines['auger_count0']})" in text


def test_t3_the_domain_is_where_the_integer_power_fits_and_the_chapter_states_its_maximum():
    p = RULES["power"]
    assert cc.Z_MAX**p <= 2 ** (gen.INT_BITS - 1) - 1 < (cc.Z_MAX + 1) ** p
    ratio = {z: cc.ka_over_ktot(z) for z in range(1, cc.Z_MAX + 1)}
    zhi = max(ratio, key=ratio.__getitem__)
    assert ratio[zhi] <= Fraction(72732, 100000)
    assert f"is at most {cc.outward(100 * ratio[zhi], 3, True)} % (at Z = {zhi})" in rendered().text


def test_t4_the_chapter_states_the_ensdf_member_sums_it_reads_from_the_reference():
    sums: dict[str, Fraction] = {}
    for r in reference():
        if r["source"] == "ensdf_measday2007":
            sums[r["target"]] = sums.get(r["target"], Fraction(0)) + Fraction(r["value"])
    want = " and ".join(cc.fixed(v, 2) for v in sums.values())
    assert f"whose members sum to {want} %" in rendered().text


def test_t10_a_revision_pair_with_different_cascades_is_listed():
    text = section(rendered().text)
    assert "Both revisions rebuilt identical cascades" in text
    inp = fresh_inputs()
    inp.cascade.harvest[6]["sequences_sha256"] = "f" * 64
    out = section(gen.render(inp).text)
    assert "rebuilt different cascades on the revisions: bertini C-12;" in out
    assert "the other 6 pairs rebuilt identical cascades" in out


def test_t10_a_route_pair_with_different_cascades_is_listed():
    assert "Both routes rebuilt identical cascades on Al-27" in section(rendered().text)
    inp = fresh_inputs()
    inp.cascade.harvest[-1]["sequences_sha256"] = "f" * 64
    out = section(gen.render(inp).text)
    assert "The routes rebuilt different cascades on 11.5.0.beta Al-27." in out


def test_t10_the_independent_propagation_and_correlation_branches(monkeypatch):
    base = gen.cascade_comparison(fresh_inputs())
    assert "the verdict changes for 1 of the 9 Kα/ΣK rows: C4H10 goes from HOLDS" in rendered().text
    assert "changes no verdict." in rendered().text
    rows = copy.deepcopy(base)
    for r in rows:
        if r["observable"] == "ka_over_ktot":
            r["verdict_independent"] = r["verdict"]
    flip = next(r for r in rows if r["verdict_rho_minus1"])
    flip["verdict_rho_plus1"] = "HOLDS"
    monkeypatch.setattr(gen, "cascade_comparison", lambda inp: rows)
    monkeypatch.setattr(gen, "comparison_problems", lambda rows, inp: [])  # the wording branch, not the check
    out = section(gen.render(fresh_inputs()).text)
    assert "the verdict changes for none of the 9 Kα/ΣK rows." in out
    assert f"changes these verdicts: {flip['target']} Kβ/Kα (FAILS to HOLDS)." in out


def test_t10_the_ensdf_k_normalised_ratio_inside_the_codes_range_is_stated_so(monkeypatch):
    assert "above the largest value the code gives for any Z in its domain" in rendered().text
    rows = copy.deepcopy(gen.cascade_comparison(fresh_inputs()))
    for r in rows:
        if r["source"] == "ensdf_measday2007" and r["observable"] == "ka_over_ktot":
            r.update(measured_num="7", measured_den="10")
    monkeypatch.setattr(gen, "cascade_comparison", lambda inp: rows)
    monkeypatch.setattr(gen, "comparison_problems", lambda rows, inp: [])  # the wording branch, not the check
    assert "within the code's range." in section(gen.render(fresh_inputs()).text)


def test_t10_a_source_that_does_not_bind_raises_before_anything_is_stated():
    inp = fresh_inputs()
    inp.cascade.sources[V2] = edited(SRC2, "G4int nAuger", "1", "2")
    with pytest.raises(gen.cc.SourceMismatch):
        gen.render(inp)


def test_t10_a_rejecting_cell_raises_before_anything_is_stated():
    inp = fresh_inputs()
    rows = [t for t in inp.cascade.transitions if gen.cell_of(t) == gen.cell_of(inp.cascade.harvest[0])]
    by = {t["transition"]: t for t in rows}
    by["g2->1"]["stops"] = str(int(by["g2->1"]["stops"]) - 2000)
    by["g3->1"]["stops"] = str(int(by["g3->1"]["stops"]) + 2000)
    with pytest.raises(
        ValueError, match=r"^the harvested cascade rejects \d+ counts: the module does not describe it$"
    ):
        gen.render(inp)


def test_t10_a_neighbouring_rule_that_some_cell_passes_is_counted(monkeypatch):
    original = gen.cascade_consumer

    def drill(inp, rules=None):
        rows = original(inp, rules)
        if rules is not None:
            for r in rows:
                if gen.cell_of(r) == gen.cell_of(inp.cascade.harvest[0]):
                    r["decision"] = "pass"
        return rows

    monkeypatch.setattr(gen, "cascade_consumer", drill)
    out = section(gen.render(fresh_inputs()).text)
    assert "the test rejects 13 of the 14 cells." in out


def test_t10_a_domain_the_integer_type_does_not_bound_raises(monkeypatch):
    monkeypatch.setattr(gen, "INT_BITS", 33)
    with pytest.raises(
        ValueError, match="^the module's domain is not where the integer power fits the integer type$"
    ):
        gen.render(fresh_inputs())


def test_t10_consumer_rows_that_break_their_schema_raise(monkeypatch):
    monkeypatch.setattr(gen, "consumer_problems", lambda rows, inp: ["a row"])
    with pytest.raises(ValueError, match="^cascade_consumer rows break their schema"):
        gen.render(fresh_inputs())


@pytest.mark.parametrize("z_extreme", [1, 14, cc.Z_MAX])
def test_t10_the_domain_statements_read_every_z_of_their_range(monkeypatch, z_extreme):
    real_ratio, real_kb, real_hi = gen.cc.ka_over_ktot, gen.cc.kb_over_ka, gen.cc.high_over_ka
    monkeypatch.setattr(
        gen.cc, "ka_over_ktot", lambda z: Fraction(99, 100) if z == z_extreme else real_ratio(z)
    )
    monkeypatch.setattr(gen.cc, "kb_over_ka", lambda z: Fraction(1, 2) if z == z_extreme else real_kb(z))
    monkeypatch.setattr(
        gen.cc, "high_over_ka", lambda z, k: Fraction(3, 4) if z == z_extreme else real_hi(z, k)
    )
    text = section(gen.render(fresh_inputs()).text)
    assert f"is at most 99.000 % (at Z = {z_extreme})" in text
    if z_extreme >= cc.LEVELS:
        assert "Kβ/Kα lies in [" in text and "50.000 %]" in text and "75.000 %]" in text
    else:
        assert "50.000 %]" not in text


def test_t11_the_pins_cover_every_slot_line_of_the_section():
    r = rendered()
    lines = r.text.splitlines()
    start = lines.index("## The muonic X-ray cascade") + 1
    end = lines.index("## Reproducing") + 1
    slot_lines = {n for n, _, _ in r.slots if start <= n < end}
    pinned = {int(w.rsplit(" ", 1)[1]) for w, *_ in tsm.document_pins(None)}
    assert slot_lines and slot_lines <= pinned


NOVELTY = ("first", "novel", "new ", "never", "unprecedented", "pioneer", "for the first time")


def test_t12_the_section_states_no_k_alpha_per_stop_and_no_novelty():
    text = section(rendered().text)
    assert gen.forbidden_hits(rendered().text) == []
    for para in text.split("\n\n"):
        if "per stop" in para or "per muon stop" in para:
            assert para.startswith(("**Expected transitions.**", "**Limits.**")), para[:80]
    assert not [w for w in NOVELTY if w in text.lower()]
    assert "No Kα per stopped muon is compared" in text


# ============================================================================================
# Boundaries of the certified arithmetic and of the decimal display, each against an independent exact value
# ============================================================================================
def exact_tail(n: int, p: Fraction, x: int, side: str) -> Fraction:
    """P(X >= x) or P(X <= x) for X ~ Binomial(n, p), summed exactly from math.comb."""
    ks = range(x, n + 1) if side == "hi" else range(0, x + 1)
    return sum((math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in ks), Fraction(0))


@pytest.mark.parametrize(
    "n, p, x, side",
    [
        (1, Fraction(1, 2), 1, "lo"),  # the end of the support on the low side; the next ratio is exactly 1
        (1, Fraction(1, 3), 0, "hi"),  # the end of the support on the high side
        (1, Fraction(1, 2), 1, "hi"),
        (2, Fraction(1, 2), 0, "lo"),
        (3, Fraction(1, 2), 2, "lo"),  # a ratio of exactly 1 one step past the mode
        (4, Fraction(1, 2), 3, "hi"),
        (5, Fraction(2, 3), 5, "hi"),
        (7, Fraction(1, 7), 0, "lo"),
    ],
)
def test_t7_the_tail_encloses_the_exact_tail_at_the_ends_of_the_support(n, p, x, side):
    exact = exact_tail(n, p, x, side)
    lo, hi = cc.binomial_tail(n, p, x, side)
    assert lo <= exact <= hi
    assert hi - lo <= exact / 2**119 + Fraction(1, 2**599)


@pytest.mark.parametrize("n, x", [(30, 25), (2000, 760), (2000, 590)])
def test_t7_the_tail_keeps_its_documented_width(n, x):
    p = Fraction(1, 3)
    side = "hi" if x > (n + 1) * p else "lo"
    exact = exact_tail(n, p, x, side)
    lo, hi = cc.binomial_tail(n, p, x, side)
    assert lo <= exact <= hi
    assert hi - lo <= lo / 2**119 + Fraction(1, 2**599)


def test_t7_a_tail_exactly_at_the_budget_is_rejected_with_the_exact_bound():
    # X ~ Binomial(1, 1/4): P(X >= 1) = 1/4 exactly (dyadic, carried exactly), and the budget is 1/4
    assert exact_tail(1, Fraction(1, 4), 1, "hi") == Fraction(1, 4)
    d = cc.decide(1, Fraction(1, 4), 1, Fraction(1, 4))
    assert (d.side, d.bound_kind, d.bound, d.decision) == ("hi", "upper", Fraction(1, 4), "reject")
    # just above the budget the same tail passes with a lower bound above it
    d = cc.decide(1, Fraction(1, 4), 1, Fraction(1, 4) - Fraction(1, 2**200))
    assert (d.bound_kind, d.decision) == ("lower", "pass") and d.bound > Fraction(1, 4) - Fraction(1, 2**200)


def test_t7_the_remainder_bound_that_meets_the_budget_exactly_rejects():
    # n = 4, p = 1/2, x = 3: the first term 1/4, the ratio 1/4, the next term 1/16; the geometric remainder
    # after it is (1/16)(1/4)/(3/4) = 1/48, so the upper bound is 5/16 + 1/48 = 1/3, all carried exactly: at a
    # budget of 1/3 the remainder bound meets the budget and the count is rejected with it; the exact tail is
    # 5/16
    assert exact_tail(4, Fraction(1, 2), 3, "hi") == Fraction(5, 16)
    d = cc.decide(4, Fraction(1, 2), 3, Fraction(1, 3))
    assert (d.side, d.bound_kind, d.bound, d.decision) == ("hi", "upper", Fraction(1, 3), "reject")


def test_t7_a_tail_that_cannot_be_separated_from_the_budget_is_refused():
    a = Fraction(1, 2140000)
    assert exact_tail(1, a, 1, "hi") == a  # the tail is exactly the budget and a is not dyadic
    lo, hi = cc.binomial_tail(1, a, 1, "hi")
    assert lo < a < hi
    with pytest.raises(ArithmeticError, match="^the certified tail does not separate from the budget$"):
        cc.decide(1, a, 1, a)


def test_t7_every_primitive_keeps_its_160_bit_precision():
    """One rounding of a 160-bit mantissa moves a bound by at most one unit in its last place, 2^-159 of
    it."""
    third, five_sevenths = Fraction(1, 3), Fraction(5, 7)
    a, b = cc._from_fraction(third, True), cc._from_fraction(five_sevenths, True)
    xa, xb = cc._frac(a), cc._frac(b)
    for op, exact in ((cc._mul, xa * xb), (cc._add, xa + xb), (cc._div, xa / xb)):
        lo, hi = cc._frac(op(a, b, False)), cc._frac(op(a, b, True))
        assert lo <= exact <= hi and hi - lo <= hi / 2**159, op.__name__
    for x in (third, five_sevenths, Fraction(3**200 + 1, 7**90)):
        lo, hi = cc._frac(cc._from_fraction(x, False)), cc._frac(cc._from_fraction(x, True))
        assert lo <= x <= hi and hi - lo <= hi / 2**159


def test_t7_the_binomial_coefficient_is_enclosed_where_the_factorials_are_rounded():
    """From n = 70 on, n! no longer fits 160 bits, so the factorial table carries rounded bounds; the pmf must
    still enclose the exact value for every x."""
    for n in (70, 97, 150):
        for x in range(0, n + 1, 7):
            p = Fraction(2, 5)
            exact = math.comb(n, x) * p**x * (1 - p) ** (n - x)
            lo, hi = cc._pmf(n, p, x).bounds()
            assert lo <= exact <= hi, (n, x)


@pytest.mark.parametrize(
    "x, down, up",
    [
        # just above 10^-6, where the digit counts of the parts put the exponent a step high
        (Fraction(7939260503993197, 7939260503993196500000), "1.00000e-06", "1.00001e-06"),
        (Fraction(1, 5), "2.00000e-01", "2.00000e-01"),  # the digit counts give 10^0 > 1/5: one step down
        (Fraction(1), "1.00000e+00", "1.00000e+00"),  # exactly a power of ten
        (Fraction(10**30), "1.00000e+30", "1.00000e+30"),
        (Fraction(1, 10**30), "1.00000e-30", "1.00000e-30"),
        (Fraction(999999999999999, 10**14), "9.99999e+00", "1.00000e+01"),
    ],
)
def test_t7_the_decimal_exponent_of_a_bound_is_exact(x, down, up):
    for text, rounded_up in ((down, False), (up, True)):
        mantissa, exponent = text.split("e")
        shown = Fraction(mantissa) * Fraction(10) ** int(exponent)
        assert (shown >= x) if rounded_up else (shown <= x)  # the expected strings are directed bounds of x
        assert Fraction(1) <= Fraction(mantissa) < 10
        assert cc.sig_digits(x, 6, rounded_up) == text


def test_t7_the_exponent_rule_holds_for_the_digit_counts_of_any_fraction():
    # 10^(a-1) <= n < 10^a and 10^(b-1) <= d < 10^b give 10^(a-b-1) < n/d < 10^(a-b+1)
    for n, d in ((1, 9), (9, 1), (10, 99), (99, 10), (123, 1000), (1000, 123), (5, 7), (7, 5)):
        x = Fraction(n, d)
        e = cc.sig_digits(x, 1, False).split("e")[1]
        assert Fraction(10) ** int(e) <= x < Fraction(10) ** (int(e) + 1)


@pytest.mark.parametrize(
    "x, places, down, up",
    [
        (1 + Fraction(1, 10**100), 3, "1.000", "1.001"),  # beyond any fixed working precision
        (Fraction(1, 2), 3, "0.500", "0.500"),  # carried exactly: no rounding either way
        (Fraction(2, 3), 3, "0.666", "0.667"),
        (Fraction(0), 2, "0.00", "0.00"),
        (Fraction(123456789, 1000), 0, "123456", "123457"),
    ],
)
def test_t3_outward_rounding_is_exact(x, places, down, up):
    assert cc.outward(x, places, False) == down and Fraction(down) <= x
    assert cc.outward(x, places, True) == up and Fraction(up) >= x
    with pytest.raises(cc.DisplayDomainError, match="^x must be an int or a Fraction >= 0$"):
        cc.outward(-x - 1, places, True)


@pytest.mark.parametrize(
    "x, places, want",
    [
        (Fraction(1, 8), 2, "0.12"),  # 12.5 hundredths: a tie, to even
        (Fraction(3, 8), 2, "0.38"),  # 37.5: a tie, to even
        (Fraction(5, 2), 0, "2"),
        (1 + Fraction(1, 2 * 10**9) + Fraction(1, 10**100), 9, "1.000000001"),  # just past a tie
        (Fraction(1, 2 * 10**9), 9, "0.000000000"),
        (Fraction(2, 3), 6, "0.666667"),
    ],
)
def test_t8_fixed_rounds_half_even_exactly(x, places, want):
    assert cc.fixed(x, places) == want
    assert abs(Fraction(want) - x) <= Fraction(1, 2 * 10**places)


@pytest.mark.parametrize(
    "model, measured, sigma2, want",
    [
        (Fraction(9, 8), Fraction(0), Fraction(1), "+1.12"),  # 112.5 hundredths: a tie, to even
        (Fraction(0), Fraction(227, 200), Fraction(1), "-1.14"),  # 113.5: a tie, to even
        (Fraction(1), Fraction(0), Fraction(4, 9), "+1.50"),
        (Fraction(0), Fraction(1, 1000), Fraction(1), "-0.00"),  # a negative pull that rounds to zero
        (Fraction(1), Fraction(1), Fraction(1), "+0.00"),
        (Fraction(1), Fraction(0), Fraction(2), "+0.71"),  # 1/sqrt(2) = 0.7071...
    ],
)
def test_t8_the_pull_rounds_half_even_exactly(model, measured, sigma2, want):
    assert cc.pull(model, measured, sigma2) == want
    with pytest.raises(cc.DisplayDomainError, match="^sigma2 must be an int or a Fraction > 0$"):
        cc.pull(model, measured, Fraction(0))


@pytest.mark.parametrize(
    "sigma2, want",
    [
        (Fraction(1, 4 * 10**18), "0.000000000"),  # sqrt = 0.5e-9: a tie, to even
        (Fraction(9, 4 * 10**18), "0.000000002"),  # 1.5e-9: a tie, to even
        (Fraction(2), "1.414213562"),
        (Fraction(1, 9), "0.333333333"),
    ],
)
def test_t8_sigma_rounds_half_even_exactly(sigma2, want):
    assert cc.sigma_decimal(sigma2) == want
    shown = Fraction(want)
    assert (shown - Fraction(1, 2 * 10**9)) ** 2 <= sigma2 <= (shown + Fraction(1, 2 * 10**9)) ** 2
    with pytest.raises(cc.DisplayDomainError, match="^sigma2 must be an int or a Fraction >= 0$"):
        cc.sigma_decimal(Fraction(-1))


def test_t5_a_step_without_secondaries_after_the_last_e_line_is_refused():
    lines = event(0, legal())
    with pytest.raises(ValueError, match="^record lines after the last E line$"):
        cc.reconstruct([*lines, f"S 1 2 1 muIoni {h(0.0)} {h(0.0)} 0\n"], L, "bound_decay", 13)
    with pytest.raises(ValueError, match="^record lines after the last E line$"):
        cc.reconstruct([*lines, f"T 1 1 0 13 - - {h(0.0)} {h(0.0)}\n"], L, "bound_decay", 13)


def test_t10_a_harvest_that_names_other_cells_on_the_two_revisions_raises():
    inp = fresh_inputs()
    cells = inp.cascade.rule["harvest"]["cells"]
    v2_cells = [c for c in cells if c["version"] == V2]
    v2_cells[0]["target"], v2_cells[1]["target"] = v2_cells[1]["target"], v2_cells[0]["target"]
    with pytest.raises(
        ValueError, match="^the harvest does not name the same cells, in the same order, on both"
    ):
        gen.cascade_harvest(inp)


def test_t10_route_pairs_that_differ_are_listed_in_revision_order():
    inp = fresh_inputs()
    for r in inp.cascade.harvest:
        if r["config"] == "precompound":
            r["sequences_sha256"] = "f" * 64 if r["version"] == V1 else "e" * 64
    out = section(gen.render(inp).text)
    assert f"The routes rebuilt different cascades on {V1} Al-27 and {V2} Al-27." in out


def test_t10_no_terminating_decimal_equals_the_largest_k_alpha_ratio_of_the_domain():
    """The chapter compares measured K alpha / sum (decimal strings) with the domain maximum by '>': an
    equality would need the maximum to be a terminating decimal, which it is not."""
    top = max(cc.ka_over_ktot(z) for z in range(1, cc.Z_MAX + 1))
    d = top.denominator
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    assert d > 1


def test_t7_interval_division_rounds_its_lower_end_down_and_its_upper_end_up():
    one, three = cc.Interval.of(Fraction(1)), cc.Interval.of(Fraction(3))
    lo, hi = (one / three).bounds()
    assert lo < Fraction(1, 3) < hi and hi - lo <= hi / 2**159
    lo, hi = (cc.Interval.of(Fraction(6)) / three).bounds()
    assert lo == hi == 2  # an exact quotient is carried exactly
    third = cc.Interval.of(Fraction(1, 3))  # an interval of width one unit: the quotient spans both ends
    lo, hi = (one / third).bounds()
    tl, th = third.bounds()
    assert lo <= 1 / th and 1 / tl <= hi


def test_t5_a_c_line_after_a_step_without_secondaries_is_refused_as_out_of_place():
    lines = event(0, legal())
    stream = [
        *lines[:-1],
        f"S 0 4 1 hIoni {h(0.0)} {h(0.0)} 0\n",
        tsm.c_line(0, 4, 0, 11, "-", 0.1, 0.0),
        lines[-1],
    ]
    with pytest.raises(ValueError, match="^C-line not after an at-rest S-line of the muon$"):
        cc.reconstruct(stream, L, "bound_decay", 13)


def test_t1_the_rule_statements_cannot_share_a_line():
    """Each statement pattern opens with a fixed literal, and no two of these literals agree on their common
    length, so no line matches two statements: the order check compares distinct lines."""
    heads = []
    for _, pattern in cc.STATEMENTS:
        head = re.match(r"(?:\\.|[^\\(\[{?*+|])*", pattern).group(0)
        heads.append(head.replace("\\", ""))
    assert all(heads)
    for i, a in enumerate(heads):
        for b in heads[i + 1 :]:
            k = min(len(a), len(b))
            assert a[:k] != b[:k], (a, b)


def test_t7_the_walk_stops_once_the_remainder_is_below_its_tolerance():
    """The documented stop: once the remainder bound is below 2^-120 of lo. Each step multiplies the remainder
    bound by about the ratio, which stays above 3/10 for 500 steps past x = mode + 60 (n = 2000, p = 1/3), so
    the remainder the walk stops with, part of the returned width, is above 2^-130 lo: the walk does not
    run on to a width set by rounding alone (about 2^-150 lo), and it stays below the documented 2^-119 lo."""
    n, p = 2000, Fraction(1, 3)
    x = math.floor((n + 1) * p) + 60
    lo, hi = cc.binomial_tail(n, p, x, "hi")
    assert Fraction(n - x - 500, x + 501) * p / (1 - p) > Fraction(3, 10)  # the ratio 500 steps on
    assert lo / 2**130 < hi - lo < lo / 2**119


def test_t5_an_empty_record_stream_is_read_as_no_events():
    hv = cc.reconstruct([], L, "bound_decay", 13)
    assert (hv.events, hv.stops, hv.chains, dict(hv.refused)) == (0, 0, 0, {})


def test_t8_the_decimal_helpers_admit_zero_and_refuse_a_negative_value():
    assert cc.fixed(Fraction(0), 2) == "0.00"
    assert cc.sigma_decimal(Fraction(0)) == "0.000000000"
    with pytest.raises(cc.DisplayDomainError, match="^x must be an int or a Fraction >= 0$"):
        cc.fixed(Fraction(-1, 3), 2)


# ============================================================================================
# The display helpers over their whole stated domain: any size, any integer number of places
# ============================================================================================
def value(text: str) -> Fraction:
    """The exact value of a plain decimal string of any length, read in 500-digit chunks (int() of a longer
    string is refused by the interpreter's int-string limit)."""
    sign = -1 if text.startswith("-") else 1
    whole, _, frac = text.lstrip("+-").partition(".")
    digits = whole + frac
    n = 0
    for i in range(0, len(digits), 500):
        chunk = digits[i : i + 500]
        n = n * 10 ** len(chunk) + int(chunk)
    return sign * Fraction(n, 10 ** len(frac))


def directed_ok(text: str, x: Fraction, step: Fraction, up: bool) -> bool:
    """``text`` is the multiple of ``step`` just at or above (up) or at or below x."""
    v = value(text)
    return v % step == 0 and ((v >= x and v - step < x) if up else (v <= x and v + step > x))


def nearest_even_ok(text: str, x: Fraction, step: Fraction) -> bool:
    """``text`` is the multiple of ``step`` nearest x, a tie going to the even multiple."""
    v = value(text)
    if v % step or abs(v - x) > step / 2:
        return False
    return abs(v - x) < step / 2 or (v / step) % 2 == 0


EPS = Fraction(1, 10**60)


@pytest.mark.parametrize(
    "x, places, down, up",
    [
        (Fraction(123), -1, "120", "130"),
        (Fraction(123), -2, "100", "200"),
        (Fraction(120), -1, "120", "120"),  # a multiple: no rounding either way
        (Fraction(200), -2, "200", "200"),
        (Fraction(125), -1, "120", "130"),
        (120 + EPS, -1, "120", "130"),
        (130 - EPS, -1, "120", "130"),
        (Fraction(7), -1, "0", "10"),
        (Fraction(0), -3, "0", "0"),
        (Fraction(10**5000 + 1), -5000, "1" + "0" * 5000, "2" + "0" * 5000),  # beyond the int-string limit
    ],
)
def test_t3_outward_rounds_to_tens_and_hundreds_exactly(x, places, down, up):
    step = Fraction(10) ** -places
    assert directed_ok(down, x, step, False) and directed_ok(up, x, step, True)  # the expectations are right
    assert cc.outward(x, places, False) == down
    assert cc.outward(x, places, True) == up


@pytest.mark.parametrize(
    "x, places, want",
    [
        (Fraction(123), -1, "120"),
        (Fraction(123), -2, "100"),
        (Fraction(125), -1, "120"),  # 12.5 tens: a tie, to even
        (Fraction(135), -1, "140"),  # 13.5 tens: a tie, to even
        (125 + EPS, -1, "130"),  # just above the tie
        (125 - EPS, -1, "120"),  # just below the tie
        (Fraction(150), -2, "200"),  # 1.5 hundreds: a tie, to even
        (Fraction(250), -2, "200"),  # 2.5 hundreds: a tie, to even
        (250 + EPS, -2, "300"),
        (150 - EPS, -2, "100"),
        (Fraction(10**5000) + Fraction(1, 3), 0, "1" + "0" * 5000),  # beyond the int-string limit
        (Fraction(10**5000, 3), 2, "3" * 5000 + ".33"),
    ],
)
def test_t8_fixed_rounds_half_even_at_any_integer_places(x, places, want):
    assert nearest_even_ok(want, x, Fraction(10) ** -places)
    assert cc.fixed(x, places) == want


@pytest.mark.parametrize(
    "root, places, want",
    [
        (Fraction(123), -1, "120"),
        (Fraction(123), -2, "100"),
        (Fraction(125), -1, "120"),  # sqrt exactly 125: 12.5 tens, to even
        (Fraction(135), -1, "140"),
        (Fraction(250), -2, "200"),
        (Fraction(350), -2, "400"),
    ],
)
def test_t8_sigma_rounds_half_even_to_tens_and_hundreds(root, places, want):
    assert nearest_even_ok(want, root, Fraction(10) ** -places)
    assert cc.sigma_decimal(root * root, places) == want


def test_t8_sigma_just_off_a_tie_at_negative_places():
    # sqrt(15625 + d) is just above 125 and sqrt(15625 - d) just below: 130 and 120 at places -1
    assert cc.sigma_decimal(Fraction(15625) + EPS, -1) == "130"
    assert cc.sigma_decimal(Fraction(15625) - EPS, -1) == "120"


@pytest.mark.parametrize("bad", [True, 1.0, "2", None])
def test_t8_the_display_helpers_refuse_places_that_are_not_ints(bad):
    for call in (
        lambda: cc.fixed(Fraction(1), bad),
        lambda: cc.outward(Fraction(1), bad, True),
        lambda: cc.sigma_decimal(Fraction(1), bad),
    ):
        with pytest.raises(cc.DisplayDomainError, match="^places must be an int"):
            call()
    with pytest.raises(cc.DisplayDomainError, match="^digits must be an int"):
        cc.sig_digits(Fraction(1), bad, True)
    with pytest.raises(
        cc.DisplayDomainError, match=re.escape(f"digits must be an int in [1, {cc.PRECISION_MAX}]")
    ):
        cc.sig_digits(Fraction(1), 0, True)


def test_t7_the_exponent_of_a_rational_with_5001_digit_parts():
    x = Fraction(10**5000 + 1, 3 * 10**5000)  # just above 1/3, beyond the int-string limit
    assert Fraction(333333, 10**6) <= x < Fraction(333334, 10**6)
    assert cc.sig_digits(x, 6, False) == "3.33333e-01"
    assert cc.sig_digits(x, 6, True) == "3.33334e-01"
    tiny = Fraction(1, 10**35775) * Fraction(42021, 10**4)  # the scale of an extreme certified tail
    assert cc.sig_digits(tiny, 5, False) == cc.sig_digits(tiny, 5, True) == "4.2021e-35775"
    big = Fraction(7 * 10**6000 + 1)
    assert cc.sig_digits(big, 3, False) == "7.00e+6000" and cc.sig_digits(big, 3, True) == "7.01e+6000"


def test_t7_a_full_consumer_call_on_an_extreme_count_set_serialises_its_rejects():
    """N = 100000 stops of Al-27 (Z = 13) all taking the photon 14 -> 1 (a legal chain): the counts conserve
    the stops, most tails are below 10^-30000, and every one of the 107 rows must come back, each reject
    carrying an upper bound that is at least its certified bound and at most the budget."""
    n, z = 100000, 13
    counts = {k: 0 for k in cc.transition_keys()}
    counts["g14->1"] = n
    obs = cc.observed_counts(counts)
    assert sum(obs[f"g{k}->1"] for k in range(2, cc.LEVELS + 1)) + obs["e2->1"] == n
    rule = json.loads((DATA / "cascade_rule.json").read_text(encoding="utf-8"))["harvest"]
    probs = cc.retained_counts(cc.transitions(z))
    a_side = cc.alpha_side(rule, len(probs))
    rows = cc.consumer_rows({"version": "v", "config": "c", "target": "t"}, obs, z, n, rule)
    assert [r["count"] for r in rows] == list(probs)
    rejects = [r for r in rows if r["decision"] == "reject"]
    assert rejects
    for r in rows:
        d = cc.decide(n, probs[r["count"]], obs[r["count"]], a_side)
        assert (r["side"], r["bound_kind"], r["decision"]) == (d.side, d.bound_kind, d.decision)
        assert parse_exact(r, "bound") == d.bound and parse_exact(r, "budget") == a_side, r["count"]
    g21 = next(r for r in rows if r["count"] == "g2->1")
    assert g21["decision"] == "reject" and parse_exact(g21, "bound") < Fraction(1, 10**30000)


def test_t8_digit_strings_of_any_size_and_their_refusals():
    assert cc._digits(0) == "0" and cc._digits(10**500) == "1" + "0" * 500
    assert cc._digits(10**1200 + 7) == "1" + "0" * 1199 + "7"
    with pytest.raises(cc.DisplayDomainError, match="^n must be an int >= 0$"):
        cc._digits(-1)
    with pytest.raises(cc.DisplayDomainError, match="^places must be an int in .*, not bool$"):
        cc.fixed(Fraction(1), True)


# ============================================================================================
# The one entry contract of the display helpers: the refusal matrix
# ============================================================================================
class _IntSub(int):
    pass


class _FractionSub(Fraction):
    pass


#: Every helper with an admitted call and the kind of each argument.
HELPERS = {
    "outward": (cc.outward, {"x": Fraction(3, 7), "places": 2, "up": True}),
    "fixed": (cc.fixed, {"x": Fraction(3, 7), "places": 2}),
    "sigma_decimal": (cc.sigma_decimal, {"sigma2": Fraction(3, 7), "places": 2}),
    "pull": (cc.pull, {"model": Fraction(1, 3), "measured": Fraction(1, 7), "sigma2": Fraction(3, 7)}),
    "sig_digits": (cc.sig_digits, {"x": Fraction(3, 7), "digits": 6, "up": True}),
    "_decimal": (cc._decimal, {"n": 5, "places": 2, "negative": False}),
    "_digits": (cc._digits, {"n": 5}),
}
KIND = {
    "x": "rational",
    "model": "rational",
    "measured": "rational",
    "sigma2": "rational",
    "places": "int",
    "digits": "int",
    "n": "int",
    "up": "flag",
    "negative": "flag",
}
#: Inputs no argument admits: floats (with NaN and the infinities), Decimals, strings, None, complex numbers
#: and subclasses of the admitted types.
NEVER = [
    0.5,
    0.1,
    2.0,
    float("nan"),
    float("inf"),
    float("-inf"),
    Decimal("0.5"),
    Decimal("NaN"),
    Decimal("Infinity"),
    "1",
    "1/2",
    None,
    1j,
    _IntSub(1),
    _FractionSub(1, 2),
]
#: What each kind refuses on top of NEVER, by type.
REFUSED = {
    "rational": [True, False],
    "int": [True, False, Fraction(1, 2), Fraction(2)],
    "flag": [1, 0, Fraction(1), "True"],
}


def _refusal_matrix():
    for helper, (_, good) in HELPERS.items():
        for arg in good:
            for bad in NEVER + REFUSED[KIND[arg]]:
                yield helper, arg, bad


@pytest.mark.parametrize("helper, arg, bad", list(_refusal_matrix()))
def test_t9_every_helper_refuses_every_forbidden_type_in_every_argument(helper, arg, bad):
    call, good = HELPERS[helper]
    call(**good)  # the admitted call returns
    with pytest.raises((TypeError, ValueError)) as info:
        call(**{**good, arg: bad})
    assert isinstance(info.value, cc.DisplayDomainError)
    assert str(info.value).startswith(f"{arg} must be ") and str(info.value).endswith(
        f", not {type(bad).__name__}"
    )


@pytest.mark.parametrize(
    "helper, arg, bad",
    [
        ("outward", "x", -1),
        ("outward", "x", -EPS),
        ("fixed", "x", Fraction(-1, 3)),
        ("sigma_decimal", "sigma2", -EPS),
        ("pull", "sigma2", 0),
        ("pull", "sigma2", Fraction(-1)),
        ("sig_digits", "x", 0),
        ("sig_digits", "x", -EPS),
        ("sig_digits", "digits", 0),
        ("sig_digits", "digits", -1),
        ("_digits", "n", -1),
        *[
            (h, "places", edge)
            for h in ("outward", "fixed", "sigma_decimal")
            for edge in ("max+1", "-(max+1)")
        ],
        ("sig_digits", "digits", "max+1"),
    ],
)
def test_t9_every_helper_refuses_a_value_outside_its_range(helper, arg, bad):
    call, good = HELPERS[helper]
    if isinstance(bad, str):  # one past the stated precision bound, read when the test runs
        bad = {"max+1": 1, "-(max+1)": -1}[bad] * (cc.PRECISION_MAX + 1)
    with pytest.raises((TypeError, ValueError)) as info:
        call(**{**good, arg: bad})
    assert isinstance(info.value, cc.DisplayDomainError)
    assert str(info.value).startswith(f"{arg} must be ")
    assert ", not " not in str(info.value)  # refused for its value, its type being admitted


def test_t9_the_range_edges_are_admitted():
    assert cc.outward(0, 3, True) == cc.fixed(0, 3) == cc.sigma_decimal(0, 3) == "0.000"
    assert cc.fixed(Fraction(1, 3), cc.PRECISION_MAX) == "0." + "3" * cc.PRECISION_MAX
    assert cc.outward(Fraction(1), -cc.PRECISION_MAX, True) == "1" + "0" * cc.PRECISION_MAX
    assert cc.sigma_decimal(Fraction(1), -cc.PRECISION_MAX) == "0"
    assert (
        cc.sig_digits(Fraction(1, 3), cc.PRECISION_MAX, False) == "3." + "3" * (cc.PRECISION_MAX - 1) + "e-01"
    )
    assert cc.sig_digits(2, 1, True) == "2e+00" and cc.pull(-1, 2, 9) == "-1.00"  # ints are admitted values
    assert cc._decimal(-7, -2, True) == "-700" and cc._digits(0) == "0"


def test_t9_a_refusal_is_both_a_type_error_and_a_value_error():
    """Code that caught the TypeError of a wrong type, or the ValueError of a negative value, still does."""
    assert issubclass(cc.DisplayDomainError, TypeError) and issubclass(cc.DisplayDomainError, ValueError)
    for bad in ((0.5, 2), (Fraction(-1), 2), (Fraction(1), True), (Fraction(1), cc.PRECISION_MAX + 1)):
        for family in (TypeError, ValueError):
            with pytest.raises(family):
                cc.fixed(*bad)


def test_t9_the_inputs_that_slipped_through_are_refused():
    """Floats used to pass and round at their decimal look-alike, not at their exact binary value (0.1 is
    above 1/10, 0.3 below 3/10, 0.15 below 15/100), and a fractional coefficient was written as digits."""
    for call in (
        lambda: cc.outward(0.1, 1, True),
        lambda: cc.outward(0.3, 1, False),
        lambda: cc.fixed(0.15, 1),
        lambda: cc.sigma_decimal(0.25, 1),
        lambda: cc.pull(0.5, 0, 1),
        lambda: cc.sig_digits(0.1, 3, True),
        lambda: cc._digits(0.5),
        lambda: cc._decimal(0.5, 0),
    ):
        with pytest.raises((TypeError, ValueError)) as info:
            call()
        assert isinstance(info.value, cc.DisplayDomainError)
    # the exact binary values, supplied as Fractions, round as the oracle says
    assert (
        cc.outward(Fraction.from_float(0.1), 1, True)
        == oracle_directed(Fraction.from_float(0.1), 1, True)
        == "0.2"
    )
    assert (
        cc.outward(Fraction.from_float(0.3), 1, False)
        == oracle_directed(Fraction.from_float(0.3), 1, False)
        == "0.2"
    )
    assert (
        cc.fixed(Fraction.from_float(0.15), 1) == oracle_nearest_even(Fraction.from_float(0.15), 1) == "0.1"
    )


def test_t9_the_consumer_path_refuses_counts_that_are_not_ints():
    counts = dict.fromkeys(cc.retained_counts(cc.transitions(13)), 1)
    cell, policy = {"version": "v", "config": "c", "target": "t"}, {"budget_per_cell": "1/10000", "sides": 2}
    for n, bad in ((100.0, 1), (True, 1), (-1, 1), (100, 1.0), (100, True), (100, Decimal(1)), (100, -1)):
        with pytest.raises(cc.DisplayDomainError):
            cc.consumer_rows(cell, {**counts, "K>=6": bad}, 13, n, policy)
    assert len(cc.consumer_rows(cell, counts, 13, 100, policy)) == len(counts)


@pytest.mark.parametrize("places, text", [(-2, "12300"), (-1, "1230"), (0, "123"), (2, "1.23")])
def test_t9_decimal_writes_the_magnitude_and_takes_its_sign_from_negative(places, text):
    for n in (-123, 123):
        shown = cc._decimal(n, places)
        assert shown == text and value(shown) == abs(n) * Fraction(10) ** -places
        assert (
            cc._decimal(n, places, True) == "-" + text
            and value("-" + text) == -abs(n) * Fraction(10) ** -places
        )


# ============================================================================================
# An independent exact oracle: integer division and Python's own str(), not the module
# ============================================================================================
@contextlib.contextmanager
def _unlimited_str():
    """The oracle writes its digits with str(), the int-string limit lifted for that call only."""
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(old)


def _scaled(x: Fraction, places: int) -> tuple[int, int]:
    """x 10^places as integers (a, b) with b > 0."""
    if places >= 0:
        return x.numerator * 10**places, x.denominator
    return x.numerator, x.denominator * 10**-places


def _show(m: int, places: int) -> str:
    """The plain decimal text of m 10^-places for an int m >= 0."""
    with _unlimited_str():
        if places <= 0:
            return str(m) + "0" * -places if m else "0"
        whole, frac = divmod(m, 10**places)
        return f"{whole}.{frac:0{places}d}"


def oracle_directed(x: Fraction, places: int, up: bool) -> str:
    a, b = _scaled(x, places)
    return _show(-(-a // b) if up else a // b, places)


def oracle_nearest_even(x: Fraction, places: int) -> str:
    a, b = _scaled(x, places)
    k, r = divmod(a, b)
    return _show(k + 1 if 2 * r > b or (2 * r == b and k % 2) else k, places)


def _isqrt_nearest_even(a: int, b: int) -> int:
    """The integer nearest sqrt(a / b), ties to even: f = floor(sqrt(a / b)) = isqrt(a b) // b, and the root
    passes f + 1/2 exactly when 4 a > (2 f + 1)^2 b."""
    f = math.isqrt(a * b) // b
    left, right = 4 * a, (2 * f + 1) ** 2 * b
    return f + 1 if left > right or (left == right and f % 2) else f


def oracle_sigma(sigma2: Fraction, places: int) -> str:
    return _show(_isqrt_nearest_even(*_scaled(sigma2, 2 * places)), places)


def oracle_pull(model: Fraction, measured: Fraction, sigma2: Fraction) -> str:
    d = Fraction(model) - measured
    return ("-" if d < 0 else "+") + _show(_isqrt_nearest_even(*_scaled(d * d / sigma2, 4)), 2)


def oracle_sig(x: Fraction, digits: int, up: bool) -> str:
    x = Fraction(x)
    with _unlimited_str():
        e = len(str(x.numerator)) - len(str(x.denominator))
    while Fraction(10) ** e > x:
        e -= 1
    while Fraction(10) ** (e + 1) <= x:
        e += 1
    a, b = _scaled(x, digits - 1 - e)
    q = -(-a // b) if up else a // b
    if q == 10**digits:
        q, e = q // 10, e + 1
    with _unlimited_str():
        s = str(q)
    return s[0] + ("." + s[1:] if digits > 1 else "") + ("e-" if e < 0 else "e+") + f"{abs(e):02d}"


def test_t9_the_oracle_on_values_derived_by_hand():
    assert (oracle_directed(Fraction(2, 3), 3, False), oracle_directed(Fraction(2, 3), 3, True)) == (
        "0.666",
        "0.667",
    )
    assert (oracle_directed(Fraction(123), -1, False), oracle_directed(Fraction(123), -1, True)) == (
        "120",
        "130",
    )
    assert (oracle_directed(Fraction(123), -2, False), oracle_directed(Fraction(123), -2, True)) == (
        "100",
        "200",
    )
    assert (
        oracle_directed(Fraction(0), -3, True) == "0" and oracle_directed(Fraction(1, 2), 3, True) == "0.500"
    )
    assert (
        oracle_nearest_even(Fraction(1, 8), 2) == "0.12" and oracle_nearest_even(Fraction(3, 8), 2) == "0.38"
    )
    assert oracle_nearest_even(Fraction(125), -1) == "120" and oracle_nearest_even(Fraction(135), -1) == "140"
    assert oracle_nearest_even(Fraction(5, 2), 0) == "2" and oracle_nearest_even(Fraction(7, 2), 0) == "4"
    assert oracle_sigma(Fraction(2), 9) == "1.414213562" and oracle_sigma(Fraction(15129), -1) == "120"
    assert oracle_sigma(Fraction(1, 4 * 10**18), 9) == "0.000000000"  # root 0.5e-9: a tie, to even
    assert oracle_sigma(Fraction(9, 4 * 10**18), 9) == "0.000000002"  # root 1.5e-9: a tie, to even
    assert oracle_pull(Fraction(9, 8), Fraction(0), Fraction(1)) == "+1.12"
    assert oracle_pull(Fraction(0), Fraction(227, 200), Fraction(1)) == "-1.14"
    assert oracle_pull(Fraction(0), Fraction(1, 1000), Fraction(1)) == "-0.00"
    assert oracle_pull(Fraction(1), Fraction(0), Fraction(2)) == "+0.71"
    assert oracle_sig(Fraction(123456789, 10**12), 6, False) == "1.23456e-04"
    assert oracle_sig(Fraction(123456789, 10**12), 6, True) == "1.23457e-04"
    assert oracle_sig(Fraction(9999999, 10**11), 6, True) == "1.00000e-04"  # the carry
    assert (
        oracle_sig(Fraction(10), 1, True) == "1e+01"
        and oracle_sig(Fraction(7 * 10**6000 + 1), 3, True) == "7.01e+6000"
    )
    assert oracle_sig(Fraction(1, 3), 2, False) == "3.3e-01"


@pytest.mark.parametrize("estimate", [Fraction(0), Fraction(1, 10), Fraction(1, 2), Fraction(1)])
def test_t9_the_exponent_is_exact_whatever_its_estimate(monkeypatch, estimate):
    """The bit-length estimate only sets where the exact comparisons start: a start far below (log10(2)
    taken as 0 or 1/10) or far above (1/2 or 1) the exponent gives the same digits as the oracle."""
    monkeypatch.setattr(cc, "_LOG10_2", estimate)
    for x in (
        Fraction(1, 3),
        Fraction(10**200 + 1, 7),
        Fraction(7, 10**200),
        Fraction(10**300),
        Fraction(99999, 10**5),
    ):
        for digits in (1, 6):
            for up in (False, True):
                assert cc.sig_digits(x, digits, up) == oracle_sig(x, digits, up), (x, digits, up)


# ============================================================================================
# The differential test: seeded cases of every helper and direction against the oracle
# ============================================================================================
SEED = 20261022
CASES_PER_STREAM = 3000
PLACES = range(-3, 7)
DIGITS = range(1, 21)
FAMILIES = ("multiple", "tie", "tie+eps", "tie-eps", "rational", "integer")
STREAMS = ("outward down", "outward up", "fixed", "sigma", "pull", "sig down", "sig up")


def _integer(rng: random.Random) -> int:
    """A random integer of 1 to 6000 digits."""
    d = rng.randint(1, 6000)
    return rng.randrange(10 ** (d - 1), 10**d)


def _rational(rng: random.Random, low: int = 0) -> Fraction:
    return Fraction(rng.randrange(low, 10 ** rng.randint(1, 40)), rng.randrange(1, 10 ** rng.randint(1, 40)))


def _on_grid(rng: random.Random, family: str, step: Fraction, k: int | None = None) -> Fraction:
    """A value >= 0 of ``family`` against the grid of multiples k step of ``step`` (k random unless given);
    the near-ties lie EPS step from the tie."""
    if family == "rational":
        return _rational(rng)
    if family == "integer":
        return Fraction(_integer(rng))
    if k is None:
        k = rng.randrange(0, 10 ** rng.randint(1, 30))
    if family == "multiple":
        return k * step
    tie = (k + Fraction(1, 2)) * step
    return {"tie": tie, "tie+eps": tie + EPS * step, "tie-eps": tie - EPS * step}[family]


def differential_cases(stream: str):
    """CASES_PER_STREAM seeded cases of ``stream``: (family, precision, call, expected). Families and
    precisions cycle, so every family meets every precision."""
    rng = random.Random(f"{SEED}/{stream}")
    for i in range(CASES_PER_STREAM):
        family = FAMILIES[i % len(FAMILIES)]
        if stream in ("sig down", "sig up"):
            digits, up = DIGITS[(i // len(FAMILIES)) % len(DIGITS)], stream == "sig up"
            # the grid of ``digits`` significant digits at a random decade; every seventh grid case sits at
            # the top of its decade, where rounding up carries into a new digit
            k = 10**digits - 1 if i % 7 == 0 else rng.randrange(10 ** (digits - 1), 10**digits)
            x = _on_grid(rng, family, Fraction(10) ** rng.randint(-80, 80), k)
            if x == 0:
                x = Fraction(1)
            yield family, digits, (cc.sig_digits, (x, digits, up)), oracle_sig(x, digits, up)
            continue
        places = PLACES[(i // len(FAMILIES)) % len(PLACES)]
        step = Fraction(10) ** -places
        if stream == "pull":  # ties and near-ties of |d| / s on the grid of hundredths
            s = _rational(rng, 1)
            ratio = (
                _on_grid(rng, family, Fraction(1, 100)) if family != "integer" else Fraction(_integer(rng))
            )
            measured = _rational(rng) * rng.choice((1, -1))
            model = measured + rng.choice((1, -1)) * ratio * s
            sigma2 = s * s if family != "rational" else _rational(rng, 1)
            if family == "rational":
                model = _rational(rng) * rng.choice((1, -1))
            yield family, places, (cc.pull, (model, measured, sigma2)), oracle_pull(model, measured, sigma2)
            continue
        x = _on_grid(rng, family, step)
        if (
            stream == "sigma"
        ):  # the root lies on, at or just off the grid; a random rational is rarely a square
            sigma2 = x if family == "rational" else x * x
            yield family, places, (cc.sigma_decimal, (sigma2, places)), oracle_sigma(sigma2, places)
        elif stream == "fixed":
            yield family, places, (cc.fixed, (x, places)), oracle_nearest_even(x, places)
        else:
            up = stream == "outward up"
            yield family, places, (cc.outward, (x, places, up)), oracle_directed(x, places, up)


def test_t9_the_differential_test_covers_twenty_thousand_cases_of_every_family_and_precision():
    helper = {
        "outward down": (cc.outward, False),
        "outward up": (cc.outward, True),
        "fixed": (cc.fixed, None),
        "sigma": (cc.sigma_decimal, None),
        "pull": (cc.pull, None),
        "sig down": (cc.sig_digits, False),
        "sig up": (cc.sig_digits, True),
    }
    assert set(helper) == set(STREAMS)
    total = 0
    for stream in STREAMS:
        cases = list(differential_cases(stream))
        # each stream calls its own helper in its own direction
        assert {
            (call, args[2] if len(args) == 3 and call is not cc.pull else None)
            for _, _, (call, args), _ in cases
        } == {helper[stream]}, stream
        seen = {(family, precision) for family, precision, _, _ in cases}
        precisions = DIGITS if stream in ("sig down", "sig up") else PLACES
        if stream == "pull":
            assert {f for f, _ in seen} == set(FAMILIES)
        else:
            assert seen == {(f, q) for f in FAMILIES for q in precisions}, stream
        total += len(cases)
    assert total == len(STREAMS) * CASES_PER_STREAM >= 20000


@pytest.mark.parametrize("stream", STREAMS)
def test_t9_every_display_helper_agrees_with_the_exact_oracle(stream):
    checked = as_int = 0
    for family, precision, (call, args), want in differential_cases(stream):
        assert call(*args) == want, (stream, family, precision)
        checked += 1
        # the same case with every integral-valued Fraction argument given as an int
        ints = tuple(int(a) if type(a) is Fraction and a.denominator == 1 else a for a in args)
        if ints != args or any(type(x) is not type(y) for x, y in zip(ints, args, strict=True)):
            assert call(*ints) == want, (stream, family, precision, "as ints")
            as_int += 1
    assert checked == CASES_PER_STREAM and as_int > 0, (
        as_int
    )  # pull: also test_t11_every_integral_argument_...


# ============================================================================================
# The full consumer path, decide -> consumer_rows -> csv_text -> parsed CSV, against exact binomial sums
# ============================================================================================
def _exact_tail(n: int, p: Fraction, x: int, side: str) -> tuple[int, int]:
    """P(X >= x) (side hi) or P(X <= x) (side lo) of Binomial(n, p) as an unreduced integer ratio, summed
    term by term from math.comb."""
    a, b = p.numerator, p.denominator
    ks = range(x, n + 1) if side == "hi" else range(0, x + 1)
    return sum(math.comb(n, k) * a**k * (b - a) ** (n - k) for k in ks), b**n


def _grid(shown: str) -> tuple[Fraction, Fraction, Fraction]:
    """A six-digit scientific string as (value, the grid point below it, the grid point above it)."""
    mantissa, exponent = shown.split("e")
    m, e = int(mantissa.replace(".", "")), int(exponent) - 5
    v = Fraction(m) * Fraction(10) ** e
    below = v - Fraction(10) ** (e if m > 10**5 else e - 1)
    return v, below, v + Fraction(10) ** e


def _check_consumer_path(n: int, observed: dict[str, int]) -> list[dict[str, str]]:
    rule = json.loads((DATA / "cascade_rule.json").read_text(encoding="utf-8"))["harvest"]
    probs = cc.retained_counts(cc.transitions(13))
    budget = Fraction(rule["budget_per_cell"]) / (len(probs) * int(rule["sides"]))
    rows = cc.consumer_rows({"version": "v", "config": "c", "target": "t"}, observed, 13, n, rule)
    text = cc.csv_text(cc.CONSUMER_COLUMNS, rows)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert text.splitlines()[0] == ",".join(cc.CONSUMER_COLUMNS) and parsed == rows
    assert [r["count"] for r in parsed] == list(probs)
    assert cc.consumer_problems(parsed, budget) == []
    for r in parsed:
        p, x = probs[r["count"]], observed[r["count"]]
        # the certified columns, read by the test's own parser
        assert (parse_int_text(r["n"]), parse_int_text(r["observed"])) == (n, x)
        assert (parse_exact(r, "p"), parse_exact(r, "budget")) == (p, budget)
        assert r["expected_display"] == oracle_nearest_even(n * p, 6)
        mode = (n + 1) * p.numerator // p.denominator
        assert parse_int_text(r["mode"]) == mode
        if x == mode:
            assert (
                r["side"],
                r["bound_kind"],
                r["bound_num"],
                r["bound_den"],
                r["bound_display"],
                r["decision"],
            ) == (
                "mode",
                "none",
                "",
                "",
                "",
                "pass",
            )
            continue
        side = "hi" if x > mode else "lo"
        num, den = _exact_tail(n, p, x, side)
        reject = num * budget.denominator <= budget.numerator * den
        kind = "upper" if reject else "lower"
        assert (r["side"], r["bound_kind"], r["decision"]) == (side, kind, "reject" if reject else "pass")
        bound = parse_exact(r, "bound")
        assert r["bound_display"] == oracle_sig(
            bound, 6, kind == "upper"
        )  # the display is the exact bound shown
        if kind == "upper":
            # the certified bound of the whole tail: at least the tail and within 2^-119 of it
            assert num <= bound * den <= num * (1 + Fraction(1, 2**119)) + den * Fraction(1, 2**599), r[
                "count"
            ]
        else:
            # a certified lower bound of a partial sum (the walk stops once it passes the budget): at most the
            # tail and at least the first term, the probability of x, up to its rounding
            term = math.comb(n, x) * p.numerator**x * (p.denominator - p.numerator) ** (n - x)
            assert bound * den <= num and term * (1 - Fraction(1, 2**100)) <= bound * den, r["count"]
    return parsed


def test_t9_the_consumer_path_rejects_every_count_of_an_extreme_set_exactly():
    n = 2000
    counts = {k: 0 for k in cc.transition_keys()}
    counts["g14->1"] = n
    rows = _check_consumer_path(n, cc.observed_counts(counts))
    assert {r["decision"] for r in rows} == {"reject"} and len(rows) == len(
        cc.retained_counts(cc.transitions(13))
    )


def test_t9_the_consumer_path_passes_every_count_at_and_next_to_its_mode_exactly():
    n = 200
    probs = cc.retained_counts(cc.transitions(13))
    modes = {k: (n + 1) * p.numerator // p.denominator for k, p in probs.items()}
    rows = _check_consumer_path(n, modes)
    assert {(r["bound_kind"], r["decision"]) for r in rows} == {("none", "pass")}
    rows = _check_consumer_path(n, {k: m + 1 for k, m in modes.items()})
    assert {(r["bound_kind"], r["decision"]) for r in rows} == {("lower", "pass")}


def test_t9_the_consumer_path_decides_a_mixed_set_exactly():
    n = 200
    probs = cc.retained_counts(cc.transitions(13))
    observed = {k: (n + 1) * p.numerator // p.denominator + i % 2 for i, (k, p) in enumerate(probs.items())}
    observed[max(probs, key=lambda k: probs[k])] = 0  # far below the mode of the largest probability
    observed[min(probs, key=lambda k: probs[k])] = n  # far above the mode of the smallest
    rows = _check_consumer_path(n, observed)
    kinds = {(r["bound_kind"], r["decision"]) for r in rows}
    assert kinds == {("none", "pass"), ("lower", "pass"), ("upper", "reject")}


# ============================================================================================
# The same entry contract on the decision path: the model, the binomial test, its budget, the comparison
# ============================================================================================
ALL_KEYS = cc.transition_keys()
#: Every public function of the certified path that takes numbers, with an admitted call: (call, arguments,
#: the name each argument has in the refusal). Rule numbers ("rule") may also be exact literals in a str.
DECISIONS = {
    "decide": (cc.decide, {"n": 30, "p": Fraction(1, 3), "x": 3, "alpha_side": Fraction(1, 1000)}, {}),
    "binomial_tail": (cc.binomial_tail, {"n": 30, "p": Fraction(1, 3), "x": 15, "side": "hi"}, {}),
    "alpha_side": (
        lambda counts, budget_per_cell, sides: cc.alpha_side(
            {"budget_per_cell": budget_per_cell, "sides": sides}, counts
        ),
        {"counts": 107, "budget_per_cell": "1/10000", "sides": 2},
        {},
    ),
    "verdict": (
        lambda model, measured, sigma2, holds_within, tension_within: cc.verdict(
            model, measured, sigma2, {"holds_within": holds_within, "tension_within": tension_within}
        ),
        {
            "model": Fraction(1, 3),
            "measured": Fraction(1, 4),
            "sigma2": Fraction(1, 100),
            "holds_within": "2",
            "tension_within": "3",
        },
        {},
    ),
    "ratio_sigma2": (
        cc.ratio_sigma2,
        {"a": Fraction(1, 2), "sa": Fraction(1, 100), "b": Fraction(1, 3), "sb": Fraction(1, 100), "rho": 0},
        {},
    ),
    "knorm_sigma2": (
        lambda v0, s0, v1, s1: cc.knorm_sigma2([(v0, s0), (v1, s1)]),
        {"v0": Fraction(2, 3), "s0": Fraction(1, 100), "v1": Fraction(1, 3), "s1": Fraction(1, 50)},
        {
            "v0": "members[0].value",
            "s0": "members[0].sigma",
            "v1": "members[1].value",
            "s1": "members[1].sigma",
        },
    ),
    "transitions": (
        lambda z, power: cc.transitions(z, {**RULES, "power": power}),
        {"z": 13, "power": 4},
        {"power": "rules[power]"},
    ),
    "k_line": (cc.k_line, {"z": 13, "n": 2}, {}),
    "high_over_ka": (cc.high_over_ka, {"z": 13, "k": 3}, {}),
    "transition_keys": (cc.transition_keys, {"top": cc.LEVELS}, {}),
    "union_key": (cc.union_key, {"k": 6}, {}),
    "key": (
        lambda upper, lower, kind: cc.key((upper, lower, kind)),
        {"upper": 2, "lower": 1, "kind": "photon"},
        {},
    ),
    "retained_counts": (
        lambda flow: cc.retained_counts({**cc.transitions(13), (2, 1, "photon"): flow}),
        {"flow": Fraction(1, 2)},
        {"flow": "flows[g2->1]"},
    ),
    "observed_counts": (
        lambda count: cc.observed_counts({**dict.fromkeys(ALL_KEYS, 0), "g2->1": count}),
        {"count": 5},
        {"count": "transition_counts[g2->1]"},
    ),
    "chain": (lambda z: cc.chain([], L, z), {"z": 13}, {}),
}
DECISION_KIND = {
    "n": "int",
    "p": "rational",
    "x": "int",
    "alpha_side": "rational",
    "side": "text",
    "counts": "int",
    "budget_per_cell": "rule",
    "sides": "int",
    "model": "rational",
    "measured": "rational",
    "sigma2": "rational",
    "holds_within": "rule",
    "tension_within": "rule",
    "a": "rational",
    "sa": "rational",
    "b": "rational",
    "sb": "rational",
    "rho": "rational",
    "v0": "rational",
    "s0": "rational",
    "v1": "rational",
    "s1": "rational",
    "z": "int",
    "power": "int",
    "k": "int",
    "top": "int",
    "upper": "int",
    "lower": "int",
    "kind": "text",
    "flow": "rational",
    "count": "int",
}
DECISION_REFUSED = {
    "rational": NEVER + [True, False],
    "int": NEVER + [True, False, Fraction(1, 2), Fraction(2)],
    "rule": [v for v in NEVER if not isinstance(v, str)] + [True, False],
    "text": [0.5, 1, None, True, Fraction(1, 2), Decimal(1), b"hi", _IntSub(1)],
}


def _decision_matrix():
    for fn, (_, good, _) in DECISIONS.items():
        for arg in good:
            for bad in DECISION_REFUSED[DECISION_KIND[arg]]:
                yield fn, arg, bad


def _call(fn: str, **override):
    call, good, _ = DECISIONS[fn]
    return call(**{**good, **override})


@pytest.mark.parametrize("fn", list(DECISIONS))
def test_t9_every_decision_function_admits_its_exact_call(fn):
    if fn == "chain":
        with pytest.raises(ValueError, match="^no cascade$"):  # admitted: refused only for the empty stop
            _call(fn)
    else:
        _call(fn)


@pytest.mark.parametrize("fn, arg, bad", list(_decision_matrix()))
def test_t9_every_decision_function_refuses_every_forbidden_type_in_every_argument(fn, arg, bad):
    with pytest.raises((TypeError, ValueError)) as info:
        _call(fn, **{arg: bad})
    assert isinstance(info.value, cc.DisplayDomainError)
    name = DECISIONS[fn][2].get(arg, arg)
    assert str(info.value).startswith(f"{name} must be ") and str(info.value).endswith(
        f", not {type(bad).__name__}"
    )


@pytest.mark.parametrize(
    "fn, arg, bad",
    [
        ("decide", "p", 0),
        ("decide", "p", 1),
        ("decide", "p", Fraction(-1, 3)),
        ("decide", "n", -1),
        ("decide", "x", -1),
        ("decide", "x", 31),
        ("decide", "alpha_side", 0),
        ("decide", "alpha_side", 1),
        ("binomial_tail", "p", Fraction(3, 2)),
        ("binomial_tail", "x", 31),
        ("alpha_side", "counts", 0),
        ("alpha_side", "sides", 0),
        ("alpha_side", "budget_per_cell", "1"),
        ("alpha_side", "budget_per_cell", Fraction(0)),
        ("alpha_side", "budget_per_cell", "-1/2"),
        ("verdict", "sigma2", Fraction(-1)),
        ("verdict", "holds_within", "-2"),
        ("verdict", "tension_within", -3),
        ("ratio_sigma2", "a", 0),
        ("ratio_sigma2", "b", 0),
        ("ratio_sigma2", "sa", Fraction(-1)),
        ("ratio_sigma2", "sb", -1),
        ("ratio_sigma2", "rho", Fraction(3, 2)),
        ("ratio_sigma2", "rho", -2),
        ("knorm_sigma2", "s0", Fraction(-1, 100)),
        ("knorm_sigma2", "s1", -1),
        ("transitions", "z", 0),
        ("transitions", "z", cc.Z_MAX + 1),
        ("transitions", "power", -1),
        ("k_line", "n", 1),
        ("k_line", "n", cc.LEVELS + 1),
        ("high_over_ka", "k", 2),
        ("transition_keys", "top", 1),
        ("union_key", "k", 1),
        ("key", "upper", 0),
        ("key", "lower", -1),
        ("retained_counts", "flow", Fraction(-1)),
        ("observed_counts", "count", -1),
        ("chain", "z", 0),
    ],
)
def test_t9_every_decision_function_refuses_a_value_outside_its_range(fn, arg, bad):
    with pytest.raises((TypeError, ValueError)) as info:
        _call(fn, **{arg: bad})
    assert isinstance(info.value, cc.DisplayDomainError)
    assert str(info.value).startswith(f"{DECISIONS[fn][2].get(arg, arg)} must be ")
    assert ", not " not in str(info.value)  # refused for its value, its type being admitted


def test_t9_the_decision_path_admits_the_edges_of_its_ranges():
    """Zero is in the range of the verdict's thresholds and of a member's error; the budget's interval is open
    and the counts' bounds are closed."""
    assert (
        cc.verdict(Fraction(1), Fraction(1), Fraction(0), {"holds_within": "0", "tension_within": 0})
        == "HOLDS"
    )
    assert (
        cc.verdict(Fraction(2), Fraction(1), Fraction(1), {"holds_within": 0, "tension_within": "0"})
        == "FAILS"
    )
    assert cc.ratio_sigma2(Fraction(1, 2), 0, Fraction(1, 3), 0) == 0
    assert cc.ratio_sigma2(Fraction(1, 2), Fraction(1, 100), Fraction(1, 3), 0, -1) > 0
    assert cc.ratio_sigma2(Fraction(1, 2), 0, Fraction(1, 3), Fraction(1, 100), 1) > 0
    assert cc.knorm_sigma2([(Fraction(1), 0), (Fraction(1), 0)]) == (Fraction(1, 2), 0, 0)
    assert cc.alpha_side({"budget_per_cell": "9999/10000", "sides": 1}, 1) == Fraction(9999, 10000)
    assert cc.decide(30, Fraction(1, 3), 0, Fraction(1, 1000)).side == "lo"
    assert cc.decide(30, Fraction(1, 3), 30, Fraction(1, 1000)).side == "hi"
    assert cc.transitions(1) and cc.transitions(cc.Z_MAX) and cc.observed_counts(dict.fromkeys(ALL_KEYS, 0))


@pytest.mark.parametrize(
    "bad", ["abc", "nan", "inf", "1/0", "", "0x10", "1e-100001", "1e100001", "1e+0100001"]
)
def test_t9_a_rule_number_that_is_no_exact_literal_is_refused(bad):
    for fn, arg in (
        ("alpha_side", "budget_per_cell"),
        ("verdict", "holds_within"),
        ("verdict", "tension_within"),
    ):
        with pytest.raises(
            cc.DisplayDomainError, match=f"^{arg} must be an exact decimal or ratio literal, not "
        ):
            _call(fn, **{arg: bad})


def test_t9_the_structural_refusals_of_the_decision_path():
    with pytest.raises(cc.DisplayDomainError, match="^side is"):
        cc.binomial_tail(30, Fraction(1, 3), 15, "up")
    with pytest.raises(cc.DisplayDomainError, match="^kind must be one of"):
        cc.key((2, 1, "gamma"))
    with pytest.raises(cc.DisplayDomainError, match="^members must not be empty$"):
        cc.knorm_sigma2([])
    with pytest.raises(cc.DisplayDomainError, match="^the member values must not sum to 0$"):
        cc.knorm_sigma2([(Fraction(1), Fraction(0)), (Fraction(-1), Fraction(0))])


def test_t9_a_reference_cell_that_is_not_text_is_refused():
    """comparison_rows reads exact values from the text a CSV reader gives; a number in a cell (a float would
    be read at its binary value, an int(13.7) would be 13) is refused."""
    reference = list(
        csv.DictReader(io.StringIO((DATA / "cascade_reference.csv").read_text(encoding="utf-8")))
    )
    rule = json.loads((DATA / "cascade_rule.json").read_text(encoding="utf-8"))
    assert cc.comparison_rows(reference, rule)
    for column, bad in (("value", 0.0526), ("Z", 13.7), ("sigma", Fraction(1, 100)), ("Z", 13)):
        rows = [dict(r) for r in reference]
        rows[0][column] = bad
        with pytest.raises(
            cc.DisplayDomainError, match=re.escape(f"reference[0][{column}] must be a str, not ")
        ):
            cc.comparison_rows(rows, rule)


def test_t9_a_float_probability_is_refused_by_the_decision_path():
    """The numeric boundary refuses each float witness with DisplayDomainError."""
    for call in (
        lambda: cc.decide(30, 1 / 3, 3, Fraction(1, 1000)),  # a float p
        lambda: cc.decide(30, 1 / 3, 10, Fraction(1, 1000)),  # a float p at its float mode: passed untested
        lambda: cc.decide(30, Fraction(1, 3), 3, 0.001),  # a float budget
        lambda: cc.decide(30.0, Fraction(1, 3), 3, Fraction(1, 1000)),  # a float N
        lambda: cc.binomial_tail(30, 0.5, 15, "hi"),
        lambda: cc.alpha_side({"budget_per_cell": 0.0001, "sides": 2}, 107),
        lambda: cc.alpha_side({"budget_per_cell": "1/10000", "sides": 2.0}, 107),
        lambda: cc.verdict(0.3, Fraction(1, 4), Fraction(1, 100), RULE),
        lambda: cc.ratio_sigma2(0.5, Fraction(1, 100), Fraction(1, 3), Fraction(1, 100)),
        lambda: cc.observed_counts({**dict.fromkeys(ALL_KEYS, 0), "g2->1": 3.7}),  # int() would make it 3
    ):
        with pytest.raises((TypeError, ValueError)) as info:
            call()
        assert isinstance(info.value, cc.DisplayDomainError)


# ============================================================================================
# Exact interchange: an independent chunk-reading parser, the int-string limits, tiny budgets
# ============================================================================================
def parse_int_text(text: str) -> int:
    """The test's own reader of an int column, independent of the module: an optional '-' and ASCII digits
    without a leading zero, read 400 digits at a time (never int() of a longer string)."""
    assert re.fullmatch(r"-?(0|[1-9][0-9]*)", text) and text != "-0", text[:40]
    sign, body = (-1, text[1:]) if text.startswith("-") else (1, text)
    n = 0
    for i in range(0, len(body), 400):
        piece = body[i : i + 400]
        n = n * 10 ** len(piece) + int(piece)
    return sign * n


def int_text_of(n: int) -> str:
    """The test's own writer of an int, independent of the module, 400 digits at a time."""
    if n < 0:
        return "-" + int_text_of(-n)
    pieces = []
    while True:
        n, r = divmod(n, 10**400)
        pieces.append(r)
        if not n:
            break
    return str(pieces[-1]) + "".join(str(c).zfill(400) for c in reversed(pieces[:-1]))


def parse_exact(row: dict[str, str], name: str) -> Fraction:
    """The exact rational of a row's name_num / name_den columns, read by parse_int_text; den > 0, lowest
    terms."""
    num, den = parse_int_text(row[name + "_num"]), parse_int_text(row[name + "_den"])
    assert den > 0 and math.gcd(num, den) == 1, name
    return Fraction(num, den)


@contextlib.contextmanager
def int_string_limit(limit: int):
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(limit)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(old)


CELL = {"version": "v", "config": "c", "target": "t"}


def conserved(n: int) -> dict[str, int]:
    """A conserved sample: every one of n stops takes the photon 14 -> 1 (a legal chain)."""
    return cc.observed_counts({**dict.fromkeys(cc.transition_keys(), 0), "g14->1": n})


def round_trip(columns, rows):
    return list(csv.DictReader(io.StringIO(cc.csv_text(columns, rows))))


ROUND_TRIP_VALUES = [
    Fraction(0),
    Fraction(-7, 3),
    Fraction(1, 10**5000),
    Fraction(10**6000 + 1, 3),
    -Fraction(2**20000 + 1, 3**9000),
    Fraction(123456789, 10**4300),
]


@pytest.mark.parametrize("limit", [640, 4300])
def test_t11_every_exact_field_round_trips_under_the_int_string_limit(limit):
    with int_string_limit(limit):
        rows = [cc.exact_fields("x", v) for v in ROUND_TRIP_VALUES]
        parsed = round_trip(("x_num", "x_den"), rows)
        assert [parse_exact(r, "x") for r in parsed] == ROUND_TRIP_VALUES
        assert [cc.read_fraction(r, "x") for r in parsed] == ROUND_TRIP_VALUES
        for n in (0, -1, 10**4299, -(10**5000) - 7, 2**30000):
            assert cc.read_int(cc.int_text(n)) == n == parse_int_text(cc.int_text(n))
            assert cc.int_text(n) == int_text_of(n)


@pytest.mark.parametrize("limit", [640, 4300])
@pytest.mark.parametrize("budget", ["1e-5000", Fraction(1, 10**5000)], ids=["literal", "Fraction"])
def test_t11_a_tiny_budget_reaches_the_csv_and_round_trips(limit, budget):
    """n = 100, z = 13 and one count per retained channel, under a budget of 10^-5000 per cell: the per-side
    budget is exactly 1/(214 10^5000), every row is written and read back exactly."""
    policy = {"budget_per_cell": budget, "sides": 2}
    observed = dict.fromkeys(cc.retained_counts(cc.transitions(13)), 1)
    with int_string_limit(limit):
        rows = cc.consumer_rows(CELL, observed, 13, 100, policy)
        parsed = round_trip(cc.CONSUMER_COLUMNS, rows)
        assert parsed == rows and len(parsed) == N_COUNTS
        assert {parse_exact(r, "budget") for r in parsed} == {Fraction(1, 214 * 10**5000)}
        assert cc.consumer_problems(parsed, Fraction(1, 214 * 10**5000)) == []


@pytest.mark.parametrize("limit", [640, 4300])
@pytest.mark.parametrize("budget", ["1e-5000", Fraction(1, 10**5000)], ids=["literal", "Fraction"])
@pytest.mark.parametrize("n, decision", [(100, "pass"), (20000, "reject")])
def test_t11_the_conserved_samples_decide_and_round_trip_under_a_tiny_budget(limit, budget, n, decision):
    """Sol's conserved samples: every stop takes g14->1; at N = 100 the g2->1 count is a certified pass, at
    N = 20000 a certified reject, and both reach the CSV."""
    with int_string_limit(limit):
        rows = cc.consumer_rows(CELL, conserved(n), 13, n, {"budget_per_cell": budget, "sides": 2})
        parsed = round_trip(cc.CONSUMER_COLUMNS, rows)
        [g21] = [r for r in parsed if r["count"] == "g2->1"]
        assert g21["decision"] == decision and g21["bound_kind"] == ("lower" if n == 100 else "upper")
        assert parse_exact(g21, "budget") == Fraction(1, 214 * 10**5000)
        assert cc.consumer_problems(parsed, Fraction(1, 214 * 10**5000)) == []


@pytest.mark.parametrize(
    "name, change",
    [
        ("a flipped decision", lambda r: {"decision": "pass" if r["decision"] == "reject" else "reject"}),
        ("a corrupted numerator", lambda r: {"bound_num": _bump(r["bound_num"])}),
        ("a corrupted denominator", lambda r: {"bound_den": _bump(r["bound_den"])}),
        ("a zero denominator", lambda r: {"budget_den": "0"}),
        ("a negative denominator", lambda r: {"budget_num": "-1", "budget_den": "-" + r["budget_den"]}),
        (
            "an unreduced fraction",
            lambda r: {"budget_num": "2", "budget_den": int_text_of(2 * parse_int_text(r["budget_den"]))},
        ),
    ],
)
@pytest.mark.parametrize("n", [100, 20000])
def test_t11_adverse_edits_of_the_exact_columns_fail_the_check(name, change, n):
    with int_string_limit(640):
        rows = round_trip(
            cc.CONSUMER_COLUMNS,
            cc.consumer_rows(CELL, conserved(n), 13, n, {"budget_per_cell": "1e-5000", "sides": 2}),
        )
        [g21] = [r for r in rows if r["count"] == "g2->1"]
        assert cc.consumer_problems([g21]) == []
        assert cc.consumer_problems([{**g21, **change(g21)}])


COMPARISON_FAULTS = [
    ("a flipped verdict", lambda r: {"verdict": "HOLDS" if r["verdict"] != "HOLDS" else "FAILS"}),
    # a model 10^-60 off the module's value, reduced and with the same verdict: only the model check sees it
    ("a corrupted model", lambda r: cc.exact_fields("model", parse_exact(r, "model") + Fraction(1, 10**60))),
    ("a corrupted measured denominator", lambda r: {"measured_den": _bump(r["measured_den"])}),
    ("a zero sigma2 denominator", lambda r: {"sigma2_den": "0"}),
    (
        "an unreduced sigma2",
        lambda r: {"sigma2_num": r["sigma2_num"] + "0", "sigma2_den": r["sigma2_den"] + "0"},
    ),
    ("a verdict without its variance", lambda r: {"verdict_rho_minus1": "HOLDS"}),
]


@pytest.mark.parametrize("name, change", COMPARISON_FAULTS, ids=[f[0] for f in COMPARISON_FAULTS])
def test_t11_adverse_edits_of_the_comparison_columns_fail_the_check(name, change):
    rows = gen.read_csv(DATA / "cascade_comparison.csv")
    inp = fresh_inputs()
    assert gen.comparison_problems(rows, inp) == []
    row = next(r for r in rows if r["observable"] == "ka_over_ktot")
    assert gen.comparison_problems([{**row, **change(row)}], inp)


def test_t11_the_committed_comparison_is_recomputed_from_its_exact_columns():
    rows = gen.read_csv(DATA / "cascade_comparison.csv")
    rule = json.loads((DATA / "cascade_rule.json").read_text(encoding="utf-8"))
    assert tuple(rows[0]) == cc.COMPARISON_COLUMNS and len(rows) == len(
        gen.cascade_comparison(fresh_inputs())
    )
    for r in rows:  # the test's own reading of the exact columns and the rule's squares
        d2 = (parse_exact(r, "model") - parse_exact(r, "measured")) ** 2
        s2 = parse_exact(r, "sigma2")
        h, t = Fraction(rule["ratios"]["holds_within"]), Fraction(rule["ratios"]["tension_within"])
        assert r["verdict"] == ("HOLDS" if d2 <= h * h * s2 else "TENSION" if d2 <= t * t * s2 else "FAILS")
        assert r["model"] == oracle_nearest_even(
            parse_exact(r, "model"), 9
        )  # the display shows the exact value
    displayed = [{**r, "model": "x", "measured": "x", "sigma": "x", "pull": "x"} for r in rows]
    assert gen.comparison_problems(displayed, fresh_inputs()) == []  # the display columns are never read


# ============================================================================================
# Representation invariance: every integral-valued argument as an int and as a Fraction
# ============================================================================================
def oracle_ratio_sigma2(a, sa, b, sb, rho=0):
    """sigma^2 of b / a, expanded: b^2 sa^2 / a^4 + sb^2 / a^2 - 2 rho b sa sb / a^3."""
    a, sa, b, sb, rho = (Fraction(v) for v in (a, sa, b, sb, rho))
    return b * b * sa * sa / a**4 + sb * sb / (a * a) - 2 * rho * b * sa * sb / a**3


def oracle_verdict(model, measured, sigma2, h=2, t=3):
    d2, s2 = (Fraction(model) - Fraction(measured)) ** 2, Fraction(sigma2)
    return "HOLDS" if d2 <= h * h * s2 else "TENSION" if d2 <= t * t * s2 else "FAILS"


def oracle_knorm(members):
    """(R, conventional sigma^2, independent sigma^2) of R = members[0] / sum, by partial derivatives."""
    values = [Fraction(v) for v, _ in members]
    errors = [Fraction(e) for _, e in members]
    s, a = sum(values), values[0]
    grads = [(s - a) / (s * s)] + [-a / (s * s)] * (len(values) - 1)
    return a / s, errors[0] ** 2 / (s * s), sum(g * g * e * e for g, e in zip(grads, errors, strict=True))


def _representations(args: tuple, integral: list[int]):
    """Every int / Fraction choice for the integral-valued positions: 2^len(integral) argument tuples."""
    for mask in range(1 << len(integral)):
        out = list(args)
        for bit, i in enumerate(integral):
            out[i] = int(args[i]) if mask >> bit & 1 else Fraction(args[i])
        yield tuple(out)


def _integral(rng: random.Random, sign: bool = True) -> int:
    v = rng.choice([rng.randint(1, 9), rng.randint(10**5, 10**9), rng.randrange(10**200, 10**201)])
    return -v if sign and rng.random() < 0.3 else v


def representation_cases(rng: random.Random):
    """Seeded base cases with integral-valued arguments (small, 10^5..10^9 and 201-digit ints, of either sign
    where the domain allows), and a non-integral one beside them: (function name, call, args, integral
    positions, oracle)."""
    for _ in range(100):
        m, d, s2 = _integral(rng), _integral(rng), abs(_integral(rng, False))
        yield "pull", cc.pull, (m, d, s2), [0, 1, 2], oracle_pull
        yield "pull", cc.pull, (m, Fraction(1, 3), s2), [0, 2], oracle_pull
        yield (
            "verdict",
            lambda *a: cc.verdict(*a, {"holds_within": 2, "tension_within": 3}),
            (m, d, s2),
            [0, 1, 2],
            oracle_verdict,
        )
        a, sa, b, sb = _integral(rng), abs(_integral(rng, False)), _integral(rng), abs(_integral(rng, False))
        yield "ratio_sigma2 (default rho)", cc.ratio_sigma2, (a, sa, b, sb), [0, 1, 2, 3], oracle_ratio_sigma2
        rho = rng.choice([-1, 0, 1])
        yield (
            "ratio_sigma2 (explicit rho)",
            cc.ratio_sigma2,
            (a, sa, b, sb, rho),
            [0, 1, 2, 3, 4],
            oracle_ratio_sigma2,
        )
        yield (
            "ratio_sigma2 (rational rho)",
            cc.ratio_sigma2,
            (a, sa, b, sb, Fraction(1, 3)),
            [0, 1, 2, 3],
            oracle_ratio_sigma2,
        )
        v0, e0, v1, e1 = (
            abs(_integral(rng, False)),
            abs(_integral(rng, False)),
            abs(_integral(rng, False)),
            rng.randint(0, 9),
        )
        yield (
            "knorm_sigma2",
            lambda *x: cc.knorm_sigma2([(x[0], x[1]), (x[2], x[3])]),
            (v0, e0, v1, e1),
            [0, 1, 2, 3],
            (lambda *x: oracle_knorm([(x[0], x[1]), (x[2], x[3])])),
        )
        x, places = abs(_integral(rng, False)), rng.randint(-3, 6)
        yield "fixed", cc.fixed, (x, places), [0], oracle_nearest_even
        yield (
            "outward up",
            lambda x, pl: cc.outward(x, pl, True),
            (x, places),
            [0],
            lambda x, pl: oracle_directed(Fraction(x), pl, True),
        )
        yield "sigma_decimal", cc.sigma_decimal, (x, places), [0], lambda x, pl: oracle_sigma(Fraction(x), pl)
        yield (
            "sig_digits",
            lambda x, dg: cc.sig_digits(x, dg, False),
            (x, rng.randint(1, 20)),
            [0],
            lambda x, dg: oracle_sig(Fraction(x), dg, False),
        )


def test_t11_every_integral_argument_gives_the_same_exact_result_as_an_int_or_a_fraction():
    rng = random.Random(f"{SEED}/representations")
    calls = Counter()
    for name, call, args, integral, oracle in representation_cases(rng):
        want = oracle(*[Fraction(v) if i in integral else v for i, v in enumerate(args)])
        for variant in _representations(args, integral):
            got = call(*variant)
            assert got == want and type(got) is type(want), (name, variant)
            if isinstance(got, tuple):
                assert all(type(g) is Fraction for g in got)
            calls[name] += 1
    assert set(calls) == {
        "pull",
        "verdict",
        "ratio_sigma2 (default rho)",
        "ratio_sigma2 (explicit rho)",
        "ratio_sigma2 (rational rho)",
        "knorm_sigma2",
        "fixed",
        "outward up",
        "sigma_decimal",
        "sig_digits",
    }
    assert sum(calls.values()) >= 8000, calls


def test_t11_the_round_four_witnesses_are_exact():
    assert cc.pull(200000003, 0, 40000) == "+1000000.02" and cc.pull(0, 200000003, 40000) == "-1000000.02"
    assert cc.pull(Fraction(200000003), Fraction(0), Fraction(40000)) == "+1000000.02"
    assert cc.pull(10**200, 0, 1) == "+" + "1" + "0" * 200 + ".00"
    assert cc.ratio_sigma2(3, 1, 2, 1) == Fraction(13, 81) and type(cc.ratio_sigma2(3, 1, 2, 1)) is Fraction
    assert cc.ratio_sigma2(10**200, 1, 1, 1) == Fraction(1, 10**400) + Fraction(1, 10**800)
    assert (
        cc.verdict(1, 0, cc.ratio_sigma2(3, 1, 2, 1), {"holds_within": "2", "tension_within": "3"})
        == "TENSION"  # 52/81 < 1 <= 117/81
    )


def test_t11_the_validator_returns_exact_fractions_for_every_rational_kind():
    for kind in ("rational", "rational >= 0", "rational > 0", "rational != 0", "correlation"):
        (v,) = cc._admit(x=(1, kind))
        assert type(v) is Fraction and v == 1
    assert cc._admit(x=(Fraction(1, 3), "probability")) == (Fraction(1, 3),)
    assert [type(v) for v in cc._admit(a=(3, "int >= 0"), b=(True, "flag"), c=("s", "text"))] == [
        int,
        bool,
        str,
    ]


# ============================================================================================
# The audit of the certified closure, pinned: every true division is reviewed, and every int reaches text
# through the one writer
# ============================================================================================
#: Every true division in the functions of the certified closure, (function, source), with why it is exact.
REVIEWED_DIVISIONS = {
    ("kb_over_ka", "k_line(z, 3, rules) / k_line(z, 2, rules)"): "Fractions from transitions",
    (
        "high_over_ka",
        "sum((k_line(z, n, rules) for n in range(k, LEVELS + 1)), Fraction(0)) / k_line(z, 2, rules)",
    ): "Fractions",
    ("ka_over_ktot", "k_line(z, 2, rules) / k_total(z, rules)"): "Fractions",
    ("_pmf", "_fact(n) / (_fact(x) * _fact(n - x))"): "Interval division, outward-rounded dyadic",
    ("_walk", "p / (1 - p)"): "p a Fraction from _check_binomial",
    ("_walk", "(1 - p) / p"): "p a Fraction from _check_binomial",
    ("_walk", "_frac(term.hi) * r_hi / (1 - r_hi)"): "Fractions from _frac",
    (
        "alpha_side",
        '_rule_number(policy, "budget_per_cell", "probability") / (counts * sides)',
    ): "Fraction / int",
    ("sig_digits", "x / Fraction(10) ** (e - digits + 1)"): "x a Fraction from _admit",
    ("pull", "d * d * 10**4 / sigma2"): "d and sigma2 Fractions from _admit",
    ("ratio_sigma2", "b / a"): "Fractions from _admit",
    ("ratio_sigma2", "sa / a"): "Fractions from _admit",
    ("ratio_sigma2", "sb / b"): "Fractions from _admit",
    ("knorm_sigma2", "a / s"): "Fractions from _admit",
    ("knorm_sigma2", "sa * sa / (s * s)"): "Fractions from _admit",
    ("knorm_sigma2", "((s - a) ** 2 * sa * sa + a * a * others) / s**4"): "Fractions from _admit",
    ("comparison_rows", "b[0] / a[0]"): "Fractions from read_exact",
    ("comparison_rows", "v / 100"): "a Fraction from read_exact",
    ("comparison_rows", "s / 100"): "a Fraction from read_exact",
    ("comparison_rows", "v / a[0]"): "Fractions from read_exact",
}
CERTIFIED_ENTRIES = (
    "decide",
    "binomial_tail",
    "alpha_side",
    "verdict",
    "ratio_sigma2",
    "knorm_sigma2",
    "pull",
    "consumer_rows",
    "consumer_problems",
    "comparison_rows",
    "comparison_problems",
    "transitions",
    "k_line",
    "auger_into_1s",
    "k_total",
    "kb_over_ka",
    "high_over_ka",
    "ka_over_ktot",
    "retained_counts",
    "observed_counts",
    "key",
    "union_key",
    "transition_keys",
    "exact_fields",
    "int_text",
    "read_int",
    "read_exact",
    "read_fraction",
    "fixed",
    "outward",
    "sigma_decimal",
    "sig_digits",
)


def _module_functions():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs[node.name] = node
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    funcs[node.name + "." + sub.name] = sub
    return source, tree, funcs


def test_t11_every_true_division_of_the_certified_closure_is_reviewed():
    source, _, funcs = _module_functions()
    closure, todo = set(), list(CERTIFIED_ENTRIES)
    while todo:
        f = todo.pop()
        if f in closure or f not in funcs:
            continue
        closure.add(f)
        for node in ast.walk(funcs[f]):
            if isinstance(node, ast.Name) and node.id in funcs:
                todo.append(node.id)
            if isinstance(node, ast.Name) and node.id == "Interval":
                todo += [k for k in funcs if k.startswith("Interval.")]
    found = set()
    for f in closure:
        for node in ast.walk(funcs[f]):
            if isinstance(node, ast.BinOp | ast.AugAssign) and isinstance(node.op, ast.Div):
                found.add((f, ast.get_source_segment(source, node)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
                found.add((f, "float("))
    assert found == set(REVIEWED_DIVISIONS)


#: The expressions an f-string of the module may format, each a str or a small module constant: every int
#: reaches text through int_text (an f-string formats the str it returns).
FSTRING_VALUES = {
    "Z_MAX": "constant 215",
    "PRECISION_MAX": "constant 10^5",
    "name": "str",
    "ident": "str",
    "s": "str",
    "w1": "str",
    "w2": "str",
    "b1": "str",
    "b2": "str",
    "b3": "str",
    "f[3]": "str",
    "line[:40]": "str",
    "sorted(RULES)": "list of str",
    "sorted(CAPTURE_PROCESS)": "list of str",
    "' '.join(seq)": "str",
    "ref.rules": "dict of the source's rule constants, each read by int() of a short literal",
    "RULES": "dict constant",
}
#: The str() / repr() calls of the module: none of them writes an int of unbounded size.
STR_CALLS = {
    "str(n)": "_digits: n < 10^500",
    "str(parts[-1])": "_digits: a chunk < 10^500",
    "str(c)": "_digits: a chunk < 10^500",
    "str(PRECISION_MAX)": "constant",
    "str(e)": "an exception",
    "str(self._hash.hexdigest())": "a hex digest",
    'str(r.get("count", "?"))': "a str cell",
    "repr(text[:40])": "a str",
    "repr(kind)": "a str from the rule file",
}


def test_t11_every_int_the_module_writes_goes_through_the_one_writer():
    source, tree, _ = _module_functions()
    for node in ast.walk(tree):
        if isinstance(node, ast.FormattedValue):
            text = ast.get_source_segment(source, node.value)
            is_writer = isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "int_text"
            assert is_writer or text in FSTRING_VALUES, text
            assert node.format_spec is None, text  # no numeric format specification anywhere
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("str", "repr", "format", "hex"):
            assert ast.get_source_segment(source, node) in STR_CALLS, ast.get_source_segment(source, node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            assert not (isinstance(node.left, ast.Constant) and isinstance(node.left.value, str))


@pytest.mark.parametrize("table", ["cascade_consumer.csv", "cascade_comparison.csv"])
def test_t11_compare_audits_the_written_tables_from_their_exact_columns(tmp_path, monkeypatch, table):
    """compare writes the two cascade tables, reads them back and recomputes every decision and verdict from
    their exact columns before it writes the document: a stored decision the columns do not give stops it."""
    out = {}
    real_read = gen.read_csv

    def capture(path, fields, rows):
        rows = [dict(r) for r in rows]
        if Path(path).name == table:
            rows[0]["decision" if table == "cascade_consumer.csv" else "verdict"] = (
                "reject"
                if table == "cascade_consumer.csv"
                else "HOLDS"
                if rows[0]["verdict"] != "HOLDS"
                else "FAILS"
            )
        out[Path(path).name] = cc.csv_text(fields, rows)

    monkeypatch.setattr(gen, "write_csv", capture)
    monkeypatch.setattr(
        gen,
        "read_csv",
        lambda path: (
            list(csv.DictReader(io.StringIO(out[Path(path).name])))
            if Path(path).name in out
            else real_read(path)
        ),
    )
    monkeypatch.setattr(gen, "DOCUMENT", tmp_path / "BENCHMARK_STOPPED_MU.md")
    with pytest.raises(ValueError, match="^the written cascade tables do not certify their decisions"):
        gen.compare()
    assert not (tmp_path / "BENCHMARK_STOPPED_MU.md").exists()


def test_t11_a_literal_exponent_up_to_the_stated_bound_is_read_exactly():
    assert cc.read_exact("1e-" + str(cc.PRECISION_MAX)) == Fraction(1, 10**cc.PRECISION_MAX)
    assert cc.read_exact("-25.5e+2") == -2550 and cc.read_exact("7/14") == Fraction(1, 2)
