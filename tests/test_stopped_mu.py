"""The stopped-muon benchmark: its classifier, reducer, gamma-line tag and rule (`openmucf/g4/stopped_mu.py`),
its driver (`cpp/stopped_muon/stopped_muon.py`) and the committed files of `data/g4/stopped_mu/`.

Every test here calls the function it checks with the input it names, so no earlier guard pre-empts it. The
synthetic policy is written in this file; each value it carries is one the function under test must read from
it.
"""

from __future__ import annotations

import argparse
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
from collections import Counter
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "openmucf" / "g4" / "stopped_mu.py"
DRIVER_PATH = REPO / "cpp" / "stopped_muon" / "stopped_muon.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sm = _load("stopped_mu_under_test", MODULE_PATH)
drv = _load("stopped_muon_driver_under_test", DRIVER_PATH)


# --------------------------------------------------------------------------------------------
# a synthetic policy
# --------------------------------------------------------------------------------------------
def synthetic_policy(captures: int = 4, rate_limit: str = "0.25") -> dict:
    predicates = [
        {
            "predicate": "alcap.Al.p",
            "family": "alcap",
            "target": "Al-27",
            "pdg": 2212,
            "ke_lo_MeV": "3.5",
            "ke_hi_MeV": "20.0",
            "lower_edge": "inclusive",
            "ref": "0.5",
            "ref_stat": "0.03",
            "ref_syst": "0.04",
        },
        {
            "predicate": "twist.Al.p",
            "family": "twist",
            "target": "Al-27",
            "pdg": 2212,
            "ke_lo_MeV": "3.404349",
            "ke_hi_MeV": "",
            "lower_edge": "exclusive",
            "ref": "0.5",
            "ref_stat": "0.03",
            "ref_syst": "0.04",
        },
        {
            "predicate": "gamma.Al.1808",
            "family": "gamma",
            "target": "Al-27",
            "ref": "0.5",
            "ref_stat": "0.05",
            "ref_syst": "0",
        },
        {
            "predicate": "gamma.Al.1129",
            "family": "gamma",
            "target": "Al-27",
            "ref": "0.1",
            "ref_stat": "0.01",
            "ref_syst": "0",
        },
        {
            "predicate": "nmean.Al",
            "family": "nmean",
            "target": "Al-27",
            "ref": "1",
            "ref_stat": "0.03",
            "ref_syst": "0.04",
        },
        {
            "predicate": "f2.Al",
            "family": "f2",
            "target": "Al-27",
            "ref": "0.5",
            "ref_stat": "0.1",
            "ref_syst": "0",
        },
    ]
    return {
        "setup": {
            "captures_per_configuration": captures,
            "seed_base": 1000,
            "seed_stride": 7,
            "threads": 1,
            "versions": ["A"],
            "configs": ["bertini", "bertini_pc", "precompound"],
            "configurations": {
                "bertini": {"route": "bound_decay", "environment": {}},
                "bertini_pc": {"route": "bound_decay", "environment": {"G4CASCADE_USE_PRECOMPOUND": "1"}},
                "precompound": {"route": "muonic_atom_helper", "environment": {}},
            },
            "harness_options": ["--opt-in", "off", "--threads", "1", "--momenta", "--record-datasets"],
            "targets": [
                {"target": "Al-27", "basis": "single_isotope", "arguments": ["--Z", "13", "--A", "27"]},
                {"target": "natSi", "basis": "natural", "arguments": ["--element", "Si"]},
            ],
            "datasets": {
                "variables": ["G4LEDATA", "G4LEVELGAMMADATA"],
                "binding": {"A": {"G4LEDATA": "EMLOW", "G4LEVELGAMMADATA": "PhotonEvaporation"}},
            },
        },
        "classification": {
            "capture_process": "muMinusCaptureAtRest",
            "product_model_suffix": "_NuclearCapture",
            "decay_model_suffix": "_DIO",
            "helper_decay_pdg": -12,
            "neutron_pdg": 2112,
            "photon_pdg": 22,
            "nucleus_pdg_min": 1000000000,
            "descendant_photon_window_ns": "1000",
        },
        "gamma": {
            "lines_MeV": {"gamma.Al.1808": "1.80874", "gamma.Al.1129": "1.12959"},
            "mev_per_u": "931.494",
            "residual_z_a": [12, 26],
            "tag_half_width_MeV": "0.001",
            "sideband_MeV": ["0.003", "0.013"],
            "sideband_scale": [2, 20],
        },
        "guards": {
            "per_line": {"fraction": "0.01", "beta_max": "0.02"},
            "tag_width": {"inner_MeV": "0.003", "continuum_scale": [4, 20], "k": "3", "offset": "5"},
            "max_extension_chunks": 1,
        },
        "rule": {
            "k_pass": "2",
            "k_fail": "3",
            "rescue_quantum": "1e-12",
            "efficiency_bracket": {
                "eps0": "0.545",
                "eps_lo": "0.40",
                "eps_hi": "0.65",
                "r_lo": "0.8384",
                "r_hi": "1.3625",
            },
            "no_outcome": {"rate_limit": rate_limit},
        },
        "predicates": predicates,
        "rows": [],
    }


def h(x: float) -> str:
    return float(x).hex()


# --------------------------------------------------------------------------------------------
# T5 -- the classifier on hand-written record streams
# --------------------------------------------------------------------------------------------
NC = "model_muMinusCaptureAtRest_NuclearCapture"
EM = "model_muMinusCaptureAtRest_EMCascade"
DIO = "model_muMinusCaptureAtRest_DIO"
P0 = f"{h(0.0)} {h(0.0)} {h(1.0)}"


def c_line(ev, track, k, pdg, model, ke, t, mom=P0):
    return f"C {ev} {track} {k} {pdg} {model} {h(ke)} {h(t)} {mom}\n"


def bound_decay_events() -> tuple[list[str], list[str]]:
    """Five events of the bound_decay route and the compact line each must give, derived by hand."""
    lines: list[str] = []
    want: list[str] = []
    # event 0: a capture with a neutron, a 26Mg residual and a prompt photon; the residual (track 3) emits two
    # photons, one 400 ns after the earliest product (a g token) and one 1500 ns after it (none); a photon of
    # the muon's own (track 1) is no capture-step nucleus's and gives no token either.
    lines += [
        "T 0 1 0 13 - - 0x0p+0 0x0p+0\n",
        "S 0 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 4\n",
        c_line(0, 1, 0, 11, EM, 0.01, 0.0),
        c_line(0, 1, 1, 2112, NC, 2.5, 800.0, f"{h(1.0)} {h(2.0)} {h(3.0)}"),
        c_line(0, 1, 2, 1000120260, NC, 0.25, 700.0, f"{h(-4.0)} {h(0.5)} {h(0.0)}"),
        c_line(0, 1, 3, 22, NC, 1.5, 750.0, f"{h(0.0)} {h(1.5)} {h(0.0)}"),
        f"T 0 5 1 22 muMinusCaptureAtRest {NC} {h(1.5)} {h(750.0)}\n",
        f"T 0 4 1 2112 muMinusCaptureAtRest {NC} {h(2.5)} {h(800.0)}\n",
        f"T 0 3 1 1000120260 muMinusCaptureAtRest {NC} {h(0.25)} {h(700.0)}\n",
        f"T 0 9 3 22 - - {h(1.80874)} {h(1100.0)}\n",
        f"T 0 10 3 22 - - {h(1.12959)} {h(2200.0)}\n",
        f"T 0 2 1 11 muMinusCaptureAtRest {EM} {h(0.01)} {h(0.0)}\n",
        f"T 0 11 1 22 - - {h(0.4)} {h(710.0)}\n",
        "E 0 1000 8\n",
    ]
    want.append(
        f"0 C {h(700.0)} 2112:{h(2.5)}:{h(1.0)}:{h(2.0)}:{h(3.0)} "
        f"1000120260:{h(0.25)}:{h(-4.0)}:{h(0.5)}:{h(0.0)} 22:{h(1.5)}:{h(0.0)}:{h(1.5)}:{h(0.0)} "
        f"g22:{h(1.80874)}:{h(400.0)}:1000120260"
    )
    # event 1: a decay in orbit
    lines += [
        "T 1 1 0 13 - - 0x0p+0 0x0p+0\n",
        "S 1 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 2\n",
        c_line(1, 1, 0, 11, DIO, 40.0, 900.0),
        c_line(1, 1, 1, 14, DIO, 30.0, 900.0),
        f"T 1 3 1 14 muMinusCaptureAtRest {DIO} {h(30.0)} {h(900.0)}\n",
        f"T 1 2 1 11 muMinusCaptureAtRest {DIO} {h(40.0)} {h(900.0)}\n",
        "E 1 1001 3\n",
    ]
    want.append("1 D")
    # event 2: both a capture product and a decay product -> X
    lines += [
        "T 2 1 0 13 - - 0x0p+0 0x0p+0\n",
        "S 2 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 2\n",
        c_line(2, 1, 0, 2112, NC, 1.0, 900.0),
        c_line(2, 1, 1, 11, DIO, 30.0, 900.0),
        f"T 2 3 1 11 muMinusCaptureAtRest {DIO} {h(30.0)} {h(900.0)}\n",
        f"T 2 2 1 2112 muMinusCaptureAtRest {NC} {h(1.0)} {h(900.0)}\n",
        "E 2 1002 3\n",
    ]
    want.append("2 X capture_and_dio")
    # event 3: only the electromagnetic cascade -> no outcome
    lines += [
        "T 3 1 0 13 - - 0x0p+0 0x0p+0\n",
        "S 3 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 1\n",
        c_line(3, 1, 0, 22, EM, 0.3, 0.0),
        f"T 3 2 1 22 muMinusCaptureAtRest {EM} {h(0.3)} {h(0.0)}\n",
        "E 3 1003 2\n",
    ]
    want.append("3 X no_outcome")
    # event 4: the muon never reaches the capture process
    lines += ["T 4 1 0 13 - - 0x0p+0 0x0p+0\n", "S 4 1 1 Decay 0x0p+0 0x0p+0 0\n", "E 4 1004 1\n"]
    want.append("4 X capture_steps=0")
    return lines, want


def helper_events() -> tuple[list[str], list[str]]:
    lines: list[str] = []
    want: list[str] = []
    atom = "MuAtom"
    # event 0: the muonic atom (track 2) captures: products of its LAST step
    lines += [
        "T 0 1 0 13 - - 0x0p+0 0x0p+0\n",
        "S 0 1 1 muMinusAtomicCaptureAtRest 0x0p+0 0x0p+0 1\n",
        c_line(0, 1, 0, 1000130279, atom, 0.0, 0.0),
        f"T 0 2 1 1000130279 muMinusAtomicCaptureAtRest {atom} {h(0.0)} {h(0.0)}\n",
        "S 0 2 1 Transportation 0x0p+0 0x0p+0 1\n",
        c_line(0, 2, 0, 22, "x", 0.1, 5.0),
        "S 0 2 2 muonicAtomDecay 0x0p+0 0x0p+0 2\n",
        c_line(0, 2, 0, 2112, "model_G4MuMinusCapturePrecompound", 3.0, 600.0),
        c_line(0, 2, 1, 14, "model_G4MuMinusCapturePrecompound", 90.0, 500.0),
        f"T 0 5 2 14 muonicAtomDecay model_G4MuMinusCapturePrecompound {h(90.0)} {h(500.0)}\n",
        f"T 0 4 2 2112 muonicAtomDecay model_G4MuMinusCapturePrecompound {h(3.0)} {h(600.0)}\n",
        f"T 0 3 2 22 Transportation x {h(0.1)} {h(5.0)}\n",
        "E 0 1000 5\n",
    ]
    want.append(f"0 C {h(500.0)} 2112:{h(3.0)}:{P0.replace(' ', ':')} 14:{h(90.0)}:{P0.replace(' ', ':')}")
    # event 1: a decay, told by its electron antineutrino (PDG -12)
    lines += [
        "T 1 1 0 13 - - 0x0p+0 0x0p+0\n",
        f"T 1 2 1 1000130279 muMinusAtomicCaptureAtRest {atom} {h(0.0)} {h(0.0)}\n",
        "S 1 2 1 muonicAtomDecay 0x0p+0 0x0p+0 2\n",
        c_line(1, 2, 0, 11, "model_DIO", 40.0, 900.0),
        c_line(1, 2, 1, -12, "model_DIO", 30.0, 900.0),
        f"T 1 4 2 -12 muonicAtomDecay model_DIO {h(30.0)} {h(900.0)}\n",
        f"T 1 3 2 11 muonicAtomDecay model_DIO {h(40.0)} {h(900.0)}\n",
        "E 1 1001 4\n",
    ]
    want.append("1 D")
    # event 2: no products
    lines += [
        "T 2 1 0 13 - - 0x0p+0 0x0p+0\n",
        f"T 2 2 1 1000130279 muMinusAtomicCaptureAtRest {atom} {h(0.0)} {h(0.0)}\n",
        "S 2 2 1 muonicAtomDecay 0x0p+0 0x0p+0 0\n",
        "E 2 1002 2\n",
    ]
    want.append("2 X no_products")
    # event 3: two muonic-atom tracks
    lines += [
        "T 3 1 0 13 - - 0x0p+0 0x0p+0\n",
        f"T 3 2 1 1000130279 muMinusAtomicCaptureAtRest {atom} {h(0.0)} {h(0.0)}\n",
        "S 3 2 1 muonicAtomDecay 0x0p+0 0x0p+0 0\n",
        f"T 3 3 1 1000130279 muMinusAtomicCaptureAtRest {atom} {h(0.0)} {h(0.0)}\n",
        "S 3 3 1 muonicAtomDecay 0x0p+0 0x0p+0 0\n",
        "E 3 1003 3\n",
    ]
    want.append("3 X atom_tracks=2")
    return lines, want


def run_stream(lines: list[str], route: str, events: int, offset: int = 0, policy: dict | None = None):
    out = io.StringIO()
    st = sm.classify_stream(
        iter(lines),
        out,
        route=route,
        version="A",
        config="bertini",
        target="Al-27",
        offset=offset,
        events=events,
        policy=policy or synthetic_policy(),
    )
    return out.getvalue().splitlines(), st


def test_t5_bound_decay_events_classify_as_derived_by_hand():
    lines, want = bound_decay_events()
    got, st = run_stream(lines, "bound_decay", 5)
    assert got[0] == "# A bertini Al-27 0 5 bound_decay"
    assert got[1:] == want
    assert (st["C"], st["D"], st["X"], st["g_tokens"]) == (1, 1, 3, 1)
    assert st["x_reasons"] == {"capture_and_dio": 1, "no_outcome": 1, "capture_steps=0": 1}


def test_t5_event_numbers_are_global():
    lines, want = bound_decay_events()
    got, _ = run_stream(lines, "bound_decay", 5, offset=2000)
    assert got[0] == "# A bertini Al-27 2000 5 bound_decay"
    assert [g.split(" ", 1)[0] for g in got[1:]] == [str(2000 + i) for i in range(5)]


def test_t5_helper_route_events_classify_as_derived_by_hand():
    lines, want = helper_events()
    got, st = run_stream(lines, "muonic_atom_helper", 4)
    assert got[1:] == want
    assert st["x_reasons"] == {"no_products": 1, "atom_tracks=2": 1}


def test_t5_the_descendant_window_is_read_from_the_policy():
    lines, _ = bound_decay_events()
    policy = synthetic_policy()
    policy["classification"]["descendant_photon_window_ns"] = "300"
    got, st = run_stream(lines, "bound_decay", 5, policy=policy)
    assert "g22:" not in got[1] and st["g_tokens"] == 0


@pytest.mark.parametrize(
    "case", ["no momenta", "unknown kind", "out of order", "trailing", "count", "C first"]
)
def test_t5_malformed_streams_are_refused(case):
    lines, _ = bound_decay_events()
    events = 5
    if case == "no momenta":
        # the electromagnetic-cascade line, which no product token reads: only the field count refuses it
        lines[2] = " ".join(lines[2].split()[:8]) + "\n"
    elif case == "unknown kind":
        lines.insert(2, "Q 0 1\n")
    elif case == "out of order":
        lines = [ln.replace("E 1 1001", "E 7 1001") for ln in lines]
    elif case == "trailing":
        lines.append("T 5 1 0 13 - - 0x0p+0 0x0p+0\n")
    elif case == "count":
        events = 6
    elif case == "C first":
        lines.insert(0, c_line(0, 1, 0, 2112, NC, 1.0, 1.0))
    with pytest.raises(sm.Malformed, match="without momenta" if case == "no momenta" else None):
        run_stream(lines, "bound_decay", events)


def record_count_mismatches(lines: list[str]) -> tuple[int, int, int]:
    """(S lines whose C-line count differs from the count they declare, S lines whose C lines are not
    numbered 0, 1, ..., E lines whose track count differs from the T lines of their event), counted
    independently of the classifier: a C line belongs to the last S line of its event, whatever lies between
    them."""
    s_bad = k_bad = e_bad = 0
    t = 0
    step: list[str] | None = None
    ks: list[int] = []

    def end_step() -> None:
        nonlocal s_bad, k_bad
        if step is not None:
            s_bad += int(step[7]) != len(ks)
            k_bad += ks != list(range(len(ks)))

    for line in lines:
        f = line.split()
        if f[0] == "T":
            t += 1
        elif f[0] == "S":
            end_step()
            step, ks = f, []
        elif f[0] == "C":
            ks.append(int(f[3]))
        elif f[0] == "E":
            end_step()
            step, ks = None, []
            e_bad += int(f[3]) != t
            t = 0
    return s_bad, k_bad, e_bad


def test_t5_the_hand_fixtures_follow_the_record_counts():
    for lines, _ in (bound_decay_events(), helper_events()):
        assert record_count_mismatches(lines) == (0, 0, 0)


def edited(lines: list[str], prefix: str, field: int, value: str) -> list[str]:
    """``lines`` with field ``field`` of the one line starting with ``prefix`` set to ``value``."""
    [i] = [j for j, ln in enumerate(lines) if ln.startswith(prefix)]
    f = lines[i].split()
    f[field] = value
    return [*lines[:i], " ".join(f) + "\n", *lines[i + 1 :]]


def without(lines: list[str], prefix: str) -> list[str]:
    [i] = [j for j, ln in enumerate(lines) if ln.startswith(prefix)]
    return lines[:i] + lines[i + 1 :]


def missing_neutron_renumbered(lines: list[str]) -> list[str]:
    """Event 0 without its neutron C line, the C lines after it renumbered so the ordinals stay 0, 1, 2."""
    out = without(lines, "C 0 1 1 2112 ")
    out = edited(out, "C 0 1 2 1000120260 ", 3, "1")
    return edited(out, "C 0 1 3 22 ", 3, "2")


def duplicate_neutron(lines: list[str]) -> list[str]:
    """Event 0 with its photon C line replaced by a copy of its neutron C line: the S line's count holds."""
    [n] = [ln for ln in lines if ln.startswith("C 0 1 1 2112 ")]
    [i] = [j for j, ln in enumerate(lines) if ln.startswith("C 0 1 3 22 ")]
    return [*lines[:i], n, *lines[i + 1 :]]


def t_line_between_s_and_c(lines: list[str]) -> list[str]:
    """Event 1 with the muon's T line moved from before its S line to between the S line and its C lines."""
    out = without(lines, "T 1 1 0 13 ")
    i = out.index("S 1 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 2\n")
    return [*out[: i + 1], "T 1 1 0 13 - - 0x0p+0 0x0p+0\n", *out[i + 1 :]]


def renumbered_event(lines: list[str], old: str, new: str) -> list[str]:
    """Every line of event ``old``, its E line included, renumbered to ``new``."""
    out = []
    for ln in lines:
        f = ln.split()
        out.append(" ".join([f[0], new, *f[2:]]) + "\n" if f[1] == old else ln)
    return out


#: (name, the corrupted bound_decay stream, the clause that must refuse it, the independent record counts of
#: the corrupted stream or None). Each corruption keeps every other count, ordinal and field of the stream
#: consistent, so only its own clause can refuse it.
RECORD_FAULTS = [
    (
        "missing C, ordinals renumbered",
        missing_neutron_renumbered,
        "declares 4 secondaries, 3 C-lines",
        (1, 0, 0),
    ),
    ("duplicate C, S count unchanged", duplicate_neutron, "C-line numbered 1 where 3 is next", (0, 1, 0)),
    (
        "missing T, E count unchanged",
        lambda ls: without(ls, "T 0 9 3 22 "),
        "counts 8 tracks, 7 T-lines",
        (0, 0, 1),
    ),
    (
        "photon PDG +22",
        lambda ls: edited(ls, "T 0 9 3 22 ", 4, "+22"),
        r"'\+22' is not an integer",
        (0, 0, 0),
    ),
    (
        "nucleus track ID +3",
        lambda ls: edited(ls, "T 0 3 1 1000120260 ", 2, "+3"),
        r"'\+3' is not an integer",
        (0, 0, 0),
    ),
    ("parent ID +3", lambda ls: edited(ls, "T 0 9 3 22 ", 3, "+3"), r"'\+3' is not an integer", (0, 0, 0)),
    ("leading zero", lambda ls: edited(ls, "T 0 9 3 22 ", 3, "03"), "'03' is not an integer", (0, 0, 0)),
    (
        "decimal float",
        lambda ls: edited(ls, "C 0 1 1 2112 ", 6, "2.5"),
        "'2.5' is not a C99 hexadecimal",
        (0, 0, 0),
    ),
    (
        "T with 10 fields",
        lambda ls: edited(ls, "T 0 9 3 22 ", 8, f"{h(1100.0)} 0"),
        "T-line with 10 fields",
        None,
    ),
    ("S with 7 fields", lambda ls: edited(ls, "S 4 1 1 Decay ", 7, ""), "S-line with 7 fields", None),
    ("E with 3 fields", lambda ls: edited(ls, "E 4 1004 ", 3, ""), "E-line with 3 fields", None),
    ("more C than declared", lambda ls: edited(ls, "S 1 1 1 ", 7, "1"), "more C-lines than the 1", (1, 0, 0)),
    (
        "C of another track",
        lambda ls: edited(ls, "C 0 1 0 11 ", 2, "3"),
        "C-line not after its S-line",
        (0, 0, 0),
    ),
    # the T line closes the S line's C lines, whose count then falls short of the declared 2
    ("C after a T line", t_line_between_s_and_c, "declares 2 secondaries, 0 C-lines", (0, 0, 0)),
    (
        "T of another event",
        lambda ls: edited(ls, "T 0 11 1 22 ", 1, "4"),
        "line of event 4 inside event 0",
        (0, 0, 0),
    ),
    (
        "E out of order, body consistent",
        lambda ls: renumbered_event(ls, "1", "7"),
        "event 7 where 1 expected",
        (0, 0, 0),
    ),
]


@pytest.mark.parametrize("name, corrupt, match, counts", RECORD_FAULTS, ids=[f[0] for f in RECORD_FAULTS])
def test_t5_a_record_that_departs_from_the_format_is_refused_by_its_own_clause(name, corrupt, match, counts):
    lines, _ = bound_decay_events()
    run_stream(lines, "bound_decay", 5)
    bad = corrupt(list(lines))
    assert bad != lines
    if counts is not None:
        assert record_count_mismatches(bad) == counts, name
    with pytest.raises(sm.Malformed, match=match):
        run_stream(bad, "bound_decay", 5)


def test_t5_the_c_after_t_fixture_moves_only_the_muon_track_line():
    lines, _ = bound_decay_events()
    bad = t_line_between_s_and_c(lines)
    assert sorted(bad) == sorted(lines)
    i = bad.index("S 1 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 2\n")
    assert bad[i + 1].startswith("T 1 1 0 13") and bad[i + 2].startswith("C 1 1 0 11 ")


def test_t5_a_helper_decay_pdg_written_with_a_leading_zero_is_refused():
    lines, want = helper_events()
    got, _ = run_stream(lines, "muonic_atom_helper", 4)
    assert got[2] == want[1] == "1 D"
    bad = edited(lines, "C 1 2 1 -12 ", 4, "-012")
    assert record_count_mismatches(bad) == (0, 0, 0)
    with pytest.raises(sm.Malformed, match="'-012' is not an integer"):
        run_stream(bad, "muonic_atom_helper", 4)


def al27_capture_stream() -> list[str]:
    """Event 0 of the bound_decay fixture (a capture whose 26Mg residual emits a 1808.74 keV photon 400 ns
    after it) and three captures without photons, all with consistent counts."""
    lines, _ = bound_decay_events()
    out = lines[: lines.index("E 0 1000 8\n") + 1]
    for ev in (1, 2, 3):
        out += [
            f"T {ev} 1 0 13 - - 0x0p+0 0x0p+0\n",
            f"S {ev} 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 2\n",
            c_line(ev, 1, 0, 2112, NC, 1.0, 900.0),
            c_line(ev, 1, 1, 1000120260, NC, 0.25, 900.0),
            f"T {ev} 3 1 1000120260 muMinusCaptureAtRest {NC} {h(0.25)} {h(900.0)}\n",
            f"T {ev} 2 1 2112 muMinusCaptureAtRest {NC} {h(1.0)} {h(900.0)}\n",
            f"E {ev} {1000 + ev} 3\n",
        ]
    return out


def classify_and_reduce(lines: list[str]):
    pol = synthetic_policy(captures=4)
    out = io.StringIO()
    sm.classify_stream(
        iter(lines),
        out,
        route="bound_decay",
        version="A",
        config="bertini",
        target="Al-27",
        offset=0,
        events=4,
        policy=pol,
    )
    compact = out.getvalue().splitlines()
    chunk0 = sm.Chunk(chunk="chunk0", offset=0, events=4, extension=False, lines=compact)
    return sm.reduce_cell([chunk0], version="A", config="bertini", target="Al-27", policy=pol)


@pytest.mark.parametrize(
    "name", ["missing T, E count unchanged", "photon PDG +22", "nucleus track ID +3", "parent ID +3"]
)
def test_t5_a_descendant_that_refuses_its_line_is_never_hidden_by_an_altered_record(name):
    stream = al27_capture_stream()
    assert record_count_mismatches(stream) == (0, 0, 0)
    cell, status = classify_and_reduce(stream)
    assert cell["line_guard"]["gamma.Al.1808"] == {"ok": False, "C_line": 0, "D_line": 1, "beta_max": 0.0}
    assert status == "refused" and cell["problems"] == [
        "line guard refused gamma.Al.1808: C=0 D=1 beta_max=0.0"
    ]
    # The same capture without the descendant, its E line counting one track fewer, is accepted: the line
    # guard alone refused the stream above.
    clean = without(stream, "T 0 9 3 22 ")
    clean = edited(clean, "E 0 1000 ", 3, "7")
    assert classify_and_reduce(clean)[1] == "accepted"
    [(_, corrupt, match, _)] = [f for f in RECORD_FAULTS if f[0] == name]
    with pytest.raises(sm.Malformed, match=match):
        classify_and_reduce(corrupt(stream))


# --------------------------------------------------------------------------------------------
# T6 -- the reducer on hand-written chunks
# --------------------------------------------------------------------------------------------
def cap(ev: int, toks: list[tuple[int, float]]) -> str:
    return f"{ev} C {h(0.0)} " + " ".join(f"{pdg}:{h(ke)}:{P0.replace(' ', ':')}" for pdg, ke in toks)


def chunk(
    name: str,
    offset: int,
    body: list[str],
    route: str = "bound_decay",
    target: str = "natSi",
    extension: bool = False,
) -> sm.Chunk:
    head = f"# A bertini {target} {offset} {len(body)} {route}"
    return sm.Chunk(chunk=name, offset=offset, events=len(body), extension=extension, lines=[head, *body])


def reduce(chunks, policy=None, no_outcome="exclude", target="natSi"):
    return sm.reduce_cell(
        chunks,
        version="A",
        config="bertini",
        target=target,
        policy=policy or synthetic_policy(),
        no_outcome=no_outcome,
    )


# Protons at the window edges: 3.5 is inside [3.5, 20) and 20.0 outside; 3.404349 is outside the strict TWIST
# cut.
CAPTURES = [
    [(2212, 3.5), (2112, 1.0), (2112, 2.0)],
    [(2212, 20.0), (2212, 3.404349), (1000140280, 0.1)],
    [(2212, 10.0), (2212, 3.4043491), (2112, 1.0)],
    [(2112, 1.0), (2112, 1.0), (2112, 1.0), (1000130270, 0.1)],
    [(2212, 5.0), (2112, 1.0)],
]


def fraction_se(values: list[int]) -> str:
    n = len(values)
    mean = Fraction(sum(values), n)
    var = sum((Fraction(v) - mean) ** 2 for v in values) / (n - 1)
    with localcontext() as ctx:
        ctx.prec = 60
        se = (Decimal(var.numerator) / Decimal(var.denominator) / n).sqrt()
        q = Decimal(1).scaleb(se.adjusted() - 11)
        return format(se.quantize(q, rounding=ROUND_HALF_EVEN), "f")


def test_t6_counts_windows_and_standard_errors():
    body = [
        cap(0, CAPTURES[0]),
        "1 D",
        cap(2, CAPTURES[1]),
        cap(3, CAPTURES[2]),
        "4 D",
        cap(5, CAPTURES[3]),
        cap(6, CAPTURES[4]),
    ]
    cell, status = reduce([chunk("chunk0", 0, body)])
    assert status == "accepted"
    assert (cell["events_total"], cell["captures_total"], cell["decays_total"], cell["N_capt"]) == (
        7,
        5,
        2,
        4,
    )
    assert cell["events_scanned"] == 6  # the fourth capture is event 5, the sixth event read
    q = cell["quantities"]
    alcap = [1, 0, 1, 0]  # 3.5 in, 20.0 out, 10.0 in, none
    twist = [1, 1, 2, 0]  # 3.404349 out (strict), 3.4043491 in; the fifth capture is not used
    neutrons = [2, 0, 1, 3]
    assert q["alcap.Al.p"] == {"count": 2, "x": "0.5", "x_se": fraction_se(alcap)}
    assert q["twist.Al.p"] == {"count": 4, "x": "1", "x_se": fraction_se(twist)}
    assert q["nmean.Al"] == {"count": 6, "x": "1.5", "x_se": fraction_se(neutrons)}
    assert cell["neutron_hist"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert cell["residuals_top"] == {"1000140280": 1, "1000130270": 1, "none": 2}


def independent_f2(values: list[int]) -> tuple[Fraction, Fraction]:
    """f2 and the variance of its delta-method estimate, from the multiplicities themselves."""
    n = len(values)
    m = [v * (v - 1) for v in values]
    mean_n, mean_m = Fraction(sum(values), n), Fraction(sum(m), n)
    var_n = sum((v - mean_n) ** 2 for v in values) / (n - 1)
    var_m = sum((v - mean_m) ** 2 for v in m) / (n - 1)
    cov = sum((a - mean_m) * (b - mean_n) for a, b in zip(m, values, strict=True)) / (n - 1)
    f2 = mean_m / mean_n**2
    da, db = 1 / mean_n**2, -2 * mean_m / mean_n**3
    return f2, (da * da * var_m + db * db * var_n + 2 * da * db * cov) / n


def test_t6_f2_and_its_delta_method_error_match_an_independent_computation():
    values = [2, 0, 1, 3, 5, 1, 1, 0]
    hist = Counter(values)
    f2, se = sm.f2_stats(dict(hist))
    want_f2, want_var = independent_f2(values)
    assert f2 == want_f2
    assert math.isclose(se, math.sqrt(want_var), rel_tol=1e-15)


def test_t6_a_short_cell_is_short_and_carries_no_quantities():
    cell, status = reduce([chunk("chunk0", 0, [cap(0, CAPTURES[0]), "1 D", cap(2, CAPTURES[1])])])
    assert status == "short" and "quantities" not in cell and cell["N_capt"] == 2


def test_t6_the_capture_count_is_read_from_the_policy():
    body = [cap(i, CAPTURES[i % 5]) for i in range(5)]
    _, status = reduce([chunk("chunk0", 0, body)], policy=synthetic_policy(captures=5))
    assert status == "accepted"
    _, status = reduce([chunk("chunk0", 0, body)], policy=synthetic_policy(captures=6))
    assert status == "short"


def test_t6_a_ratio_that_does_not_terminate_is_refused():
    body = [cap(i, CAPTURES[i % 5]) for i in range(3)]
    with pytest.raises(sm.Refused, match="does not terminate"):
        reduce([chunk("chunk0", 0, body)], policy=synthetic_policy(captures=3))


def test_t6_an_offset_gap_and_a_wrong_header_are_refused():
    a = chunk("chunk0", 0, [cap(0, CAPTURES[0]), cap(1, CAPTURES[1])])
    b = chunk("chunk1", 3, [cap(3, CAPTURES[2]), cap(4, CAPTURES[3])])
    with pytest.raises(sm.Refused, match="offset 3 != 2"):
        reduce([a, b])
    bad = sm.Chunk(
        chunk="chunk0",
        offset=0,
        events=1,
        extension=False,
        lines=["# B bertini natSi 0 1 bound_decay", cap(0, CAPTURES[0])],
    )
    with pytest.raises(sm.Refused, match="header"):
        reduce([bad])
    with pytest.raises(sm.Refused, match="event 5 where 1"):
        reduce([chunk("chunk0", 0, [cap(0, CAPTURES[0]), cap(5, CAPTURES[1])])])


def test_t6_two_extension_chunks_and_an_x_event_refuse_the_cell():
    body = [cap(i, CAPTURES[i % 5]) for i in range(2)]
    parts = [
        chunk("chunk0", 0, body),
        chunk("chunk1", 2, [cap(2, CAPTURES[2])], extension=True),
        chunk("chunk2", 3, [cap(3, CAPTURES[3])], extension=True),
    ]
    cell, status = reduce(parts)
    assert status == "refused" and cell["problems"] == ["extension count 2 > 1"]
    cell, status = reduce(
        [chunk("chunk0", 0, body + ["2 X capture_and_dio", cap(3, CAPTURES[2]), cap(4, CAPTURES[3])])]
    )
    assert status == "refused" and cell["X"] == 1


def test_t6_no_outcome_events_leave_both_denominators_and_the_rate_guard_holds_at_its_limit():
    body = [
        cap(0, CAPTURES[0]),
        "1 X no_outcome",
        cap(2, CAPTURES[1]),
        "3 D",
        cap(4, CAPTURES[2]),
        "5 X no_outcome",
        cap(6, CAPTURES[3]),
        "7 D",
    ]
    cell, status = reduce([chunk("chunk0", 0, body)])  # 2 of 8 stops = the synthetic limit 0.25 exactly
    assert status == "accepted"
    assert (cell["events_total"], cell["captures_total"], cell["stops_total"]) == (6, 4, 8)
    assert (cell["n_no_outcome"], cell["no_outcome_events"], cell["no_outcome_guard_ok"]) == (2, [1, 5], True)
    cell, status = reduce([chunk("chunk0", 0, body)], policy=synthetic_policy(rate_limit="0.2499"))
    assert status == "refused" and cell["no_outcome_guard_ok"] is False


def test_t6_the_alternative_treatment_displaces_the_last_capture():
    body = [
        cap(0, CAPTURES[0]),
        "1 X no_outcome",
        cap(2, CAPTURES[1]),
        cap(3, CAPTURES[2]),
        cap(4, CAPTURES[3]),
    ]
    actual, _ = reduce([chunk("chunk0", 0, body)])
    alt, status = reduce([chunk("chunk0", 0, body)], no_outcome="as_capture")
    assert status == "accepted"
    assert actual["quantities"]["nmean.Al"]["x"] == "1.5"  # 2, 0, 1, 3
    assert alt["quantities"]["nmean.Al"]["x"] == "0.75"  # 2, 0 (the event), 0, 1: the 3 leaves
    assert alt["neutron_hist"] == {"0": 2, "1": 1, "2": 1}
    with pytest.raises(sm.Refused):
        reduce([chunk("chunk0", 0, body)], no_outcome="count")


# --------------------------------------------------------------------------------------------
# T7 -- the gamma-line tag, its guard and its tag-width gate
# --------------------------------------------------------------------------------------------
GP = sm.gamma_params(synthetic_policy())


@pytest.mark.parametrize("shape", sm.SYNTH_SHAPES)
def test_t7_the_tag_recovers_the_planted_yield_exactly_for_every_line_shape(shape):
    lines, truth, maxerr = sm.synth(shape, 4000, GP)
    assert maxerr < 0.7
    for key, de in GP.lines.items():
        r = sm.line_yield(lines, de, GP)
        assert truth[key] > 0
        assert r["numerator_x20"] == GP.sb_den * truth[key], (shape, key)


def guard_population(c_line: int, d_line: int, n_other: int) -> list[str]:
    t = sm.e_t(GP.lines["gamma.Al.1808"], GP)
    res = sm._nucleus_tok(1000120260, 26, [0.0, 0.0, 100.0], GP)
    b = sm.beta_vec((0, *[float.fromhex(x) for x in res.split(":")[1:]]))
    out = []
    for i in range(max(c_line, 1, n_other)):
        toks = [res]
        if i < c_line:
            e, pg = sm._boost(t, [1.0, 0.0, 0.0], list(b))
            toks.append(sm._tok(22, e, pg))
        if i < n_other:
            toks.append(sm._tok(22, 0.5, [0.0, 0.5, 0.0]))
        out.append(f"{i} C {(0.0).hex()} " + " ".join(toks))
    desc = " ".join(f"g22:{t.hex()}:{(10.0).hex()}:1000120260" for _ in range(d_line))
    if desc:
        out[0] += " " + desc
    return out


@pytest.mark.parametrize(
    "c, d, other, ok",
    [(1, 90, 10000, False), (1, 0, 10000, True), (1000, 5, 10000, True), (100, 2, 10000, False)],
)
def test_t7_the_per_line_guard_refuses_line_dominant_descendants(c, d, other, ok):
    good, tagged, descendants, _ = sm.line_guard(guard_population(c, d, other), GP.lines["gamma.Al.1808"], GP)
    assert (tagged, descendants) == (c, d)
    assert good is ok


def test_t7_the_per_line_guard_refuses_a_fast_residual():
    t = sm.e_t(GP.lines["gamma.Al.1808"], GP)
    res = sm._nucleus_tok(1000120260, 26, [0.0, 0.0, 0.026 * 26 * GP.mev_per_u], GP)
    b = sm.beta_vec((0, *[float.fromhex(x) for x in res.split(":")[1:]]))
    e, pg = sm._boost(t, [1.0, 0.0, 0.0], list(b))
    good, _, _, bmax = sm.line_guard(
        [f"0 C {(0.0).hex()} {res} {sm._tok(22, e, pg)}"], GP.lines["gamma.Al.1808"], GP
    )
    assert bmax >= GP.guard_beta_max and good is False


def test_t7_the_tag_width_gate_passes_a_narrow_line_and_refuses_a_displaced_one():
    lines, _, _ = sm.synth("uniform_0_300", 4000, GP)
    assert sm.tag_width_gate(lines, GP.lines["gamma.Al.1808"], GP)[0] is True
    t = sm.e_t(GP.lines["gamma.Al.1808"], GP)
    res = sm._nucleus_tok(1000120260, 26, [0.0, 0.0, 100.0], GP)
    b = sm.beta_vec((0, *[float.fromhex(x) for x in res.split(":")[1:]]))
    shifted = []
    for i in range(200):
        e, pg = sm._boost(t + 2.0e-3, sm._iso(__import__("random").Random(i)), list(b))
        shifted.append(f"{i} C {(0.0).hex()} {res} {sm._tok(22, e, pg)}")
    good, n1, n3, _ = sm.tag_width_gate(shifted, GP.lines["gamma.Al.1808"], GP)
    assert good is False and n1 == 0 and n3 == 200


def test_t7_an_al27_cell_runs_the_line_guards_and_reports_its_tallies():
    lines, truth, _ = sm.synth("box_100", 40, GP)
    cell, status = reduce(
        [chunk("chunk0", 0, lines, target="Al-27")], policy=synthetic_policy(captures=40), target="Al-27"
    )
    assert status == "accepted"
    assert (
        cell["gamma"]["gamma.Al.1808"]["w"] - Fraction(2, 20) * cell["gamma"]["gamma.Al.1808"]["s"]
        == truth["gamma.Al.1808"]
    )
    assert cell["quantities"]["gamma.Al.1808"]["x"] == sm.exact_ratio(20 * truth["gamma.Al.1808"], 20 * 40)
    assert set(cell["line_guard"]) == set(cell["tag_width"]) == set(GP.lines)


# --------------------------------------------------------------------------------------------
# T13 -- the module stands alone
# --------------------------------------------------------------------------------------------
def test_t13_the_module_imports_the_standard_library_alone():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import"
            names.add((node.module or "").split(".")[0])
    names.discard("__future__")
    assert names, "no import found: the walk is broken"
    assert names <= set(sys.stdlib_module_names), sorted(names - set(sys.stdlib_module_names))


def test_t13_the_module_and_the_driver_load_by_path_in_an_isolated_interpreter():
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('d', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['d'] = m; spec.loader.exec_module(m)\n"
        "print('LOADED', 'openmucf' in sys.modules, m.sm.__name__)\n"
    )
    p = subprocess.run(
        [sys.executable, "-I", "-c", code, str(DRIVER_PATH)],
        capture_output=True,
        text=True,
        cwd=str(REPO / "cpp"),
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["LOADED", "False", "stopped_mu"]


# --------------------------------------------------------------------------------------------
# T14 -- the driver
# --------------------------------------------------------------------------------------------
def test_t14_target_arguments_and_seeds_come_from_the_policy():
    p = synthetic_policy()
    assert drv.target_arguments(p, "Al-27") == ["--Z", "13", "--A", "27"]
    assert drv.target_arguments(p, "natSi") == ["--element", "Si"]
    with pytest.raises(drv.Refusal):
        drv.target_arguments(p, "natXx")
    assert drv.seed_base(p, 0) == 1000 and drv.seed_base(p, 2000) == 3000
    argv = drv.harness_argv("exe", p, config="precompound", target="natSi", offset=2000, events=5, out="w")
    assert argv[:4] == ["exe", "--route", "muonic_atom_helper", "--element"]
    assert argv[argv.index("--seed-base") + 1] == "3000" and argv[argv.index("--seed-stride") + 1] == "7"
    assert argv[-6:] == ["--opt-in", "off", "--threads", "1", "--momenta", "--record-datasets"]
    pre = drv.harness_argv(
        "exe", p, config="bertini", target="Al-27", offset=0, events=5, out="w", print_datasets=True
    )
    assert pre[-1] == "--print-datasets" and "--momenta" not in pre


def test_t14_the_environment_unsets_binds_and_switches_precompound_for_bertini_pc_only():
    p = synthetic_policy()
    base = {
        "PATH": "/bin",
        "GEANT4_DATA_DIR": "/x",
        "G4LEDATA": "/old",
        "G4FOODATA": "/y",
        "G4CASCADE_USE_PRECOMPOUND": "1",
        "G4NUCMODEL_X": "1",
        "G4CASCADE_OTHER": "2",
        "LD_LIBRARY_PATH": "/z",
    }
    env = drv.environment(p, base, version="A", config="bertini", install="/inst", data="/data")
    assert env == {
        "PATH": "/bin",
        "G4LEDATA": "/data/EMLOW",
        "G4LEVELGAMMADATA": "/data/PhotonEvaporation",
        "LD_LIBRARY_PATH": "/inst/lib",
    }
    env_pc = drv.environment(p, base, version="A", config="bertini_pc", install="/inst", data="/data")
    assert env_pc == dict(env, G4CASCADE_USE_PRECOMPOUND="1")
    env_pre = drv.environment(p, base, version="A", config="precompound", install="/inst", data="/data")
    assert "G4CASCADE_USE_PRECOMPOUND" not in env_pre
    with pytest.raises(drv.Refusal):
        drv.environment(p, base, version="A", config="geant", install="/inst", data="/data")
    with pytest.raises(drv.Refusal):
        drv.environment(p, base, version="B", config="bertini", install="/inst", data="/data")


def test_t14_the_dataset_lines_must_equal_the_binding():
    want = drv.binding(synthetic_policy(), "A", "/data")
    good = "DATASET G4LEDATA /data/EMLOW\nDATASET G4LEVELGAMMADATA /data/PhotonEvaporation\n"
    drv.check_datasets(good, want, "before the run")
    for bad in (
        good.replace("/data/EMLOW", "/data/EMLOW2"),
        good.splitlines()[0],
        "\n".join(good.splitlines()[::-1]),
        good + "DATASET G4LEDATA /data/EMLOW\n",
    ):
        with pytest.raises(drv.Refusal):
            drv.check_datasets(bad, want, "after the run")


#: The twelve dataset variables, held here independently of the pre-registration and of the driver.
DATASET_VARIABLES = (
    "G4NEUTRONHPDATA",
    "G4LEDATA",
    "G4LEVELGAMMADATA",
    "G4RADIOACTIVEDATA",
    "G4PARTICLEXSDATA",
    "G4PIIDATA",
    "G4REALSURFACEDATA",
    "G4SAIDXSDATA",
    "G4ABLADATA",
    "G4INCLDATA",
    "G4ENSDFSTATEDATA",
    "G4CHANNELINGDATA",
)


def dataset_text(binding: dict[str, str]) -> str:
    """The DATASET lines the harness writes for ``binding``: ``DATASET <variable> <directory>``, one space."""
    return "".join(f"DATASET {var} {path}\n" for var, path in binding.items())


@pytest.mark.parametrize("when", ["before the run", "after the run"])
def test_t14_a_data_root_with_a_space_is_read_whole_and_still_compared_exactly(when):
    pol = policy()
    assert tuple(pol["setup"]["datasets"]["variables"]) == DATASET_VARIABLES
    want = drv.binding(pol, "11.4.2", "/data root")
    assert all(
        path == f"/data root/{pol['setup']['datasets']['binding']['11.4.2'][v]}" for v, path in want.items()
    )
    good = dataset_text(want)
    assert drv.dataset_lines(good) == want
    drv.check_datasets(good, want, when)
    first, second = DATASET_VARIABLES[:2]
    faults = {
        "differing suffix": good.replace(f" {want[first]}\n", f" {want[first]}2\n"),
        "missing variable": "".join(
            ln + "\n" for ln in good.splitlines() if not ln.startswith(f"DATASET {first} ")
        ),
        "wrong order": dataset_text({second: want[second], first: want[first], **want}),
        "duplicate variable": good + f"DATASET {first} {want[first]}\n",
        "no directory": good.replace(f"DATASET {first} {want[first]}\n", f"DATASET {first}\n"),
    }
    for name, text in faults.items():
        assert text != good, name
        with pytest.raises(drv.Refusal):
            drv.check_datasets(text, want, when)


@pytest.mark.parametrize("variable", DATASET_VARIABLES)
def test_t14_each_dataset_directory_is_compared_with_its_binding(variable):
    want = drv.binding(policy(), "11.4.2", "/data root")
    got = dict(want, **{variable: want[variable] + "-other"})
    with pytest.raises(drv.Refusal, match=f"after the run: {variable} "):
        drv.check_datasets(dataset_text(got), want, "after the run")


def test_t14_a_binding_whose_variables_differ_from_the_list_is_refused():
    pol = copy.deepcopy(policy())
    dirs = pol["setup"]["datasets"]["binding"]["11.4.2"]
    first = next(iter(dirs))
    pol["setup"]["datasets"]["binding"]["11.4.2"] = {k: dirs[k] for k in [*list(dirs)[1:], first]}
    with pytest.raises(drv.Refusal, match="binding's variables"):
        drv.binding(pol, "11.4.2", "/data")


def test_t14_the_parser_offers_no_capture_count(capsys):
    argv = [
        "run",
        "--binary",
        "b",
        "--install",
        "i",
        "--data",
        "d",
        "--work",
        "w",
        "--version",
        "A",
        "--config",
        "bertini",
        "--target",
        "Al-27",
    ]
    assert drv.parser().parse_args(argv).cmd == "run"
    with pytest.raises(SystemExit) as stop:
        drv.parser().parse_args([*argv, "--captures", "3"])
    assert stop.value.code != 0
    assert "unrecognized arguments: --captures 3" in capsys.readouterr().err


def test_t14_the_chunk_plan_reads_the_capture_count_from_the_policy():
    # 1200 of the 2000 pilot stops captured: ceil(captures x 1.03 / 0.6) - 2000 stops follow
    assert drv.second_chunk_events(synthetic_policy(captures=100000), 1200) == math.ceil(103000 / 0.6) - 2000
    assert drv.second_chunk_events(synthetic_policy(captures=50000), 1200) == math.ceil(51500 / 0.6) - 2000
    with pytest.raises(drv.Refusal):
        drv.second_chunk_events(synthetic_policy(), 0)
    assert drv.extension_events(synthetic_policy(captures=100000), 99000, 165000) == math.ceil(
        1000 / 0.6 * 1.1
    )


def test_t14_a_rerun_chunk_whose_compact_text_differs_by_one_byte_is_named():
    text = b"# A bertini Al-27 0 1 bound_decay\n0 D\n"
    sha, _ = sm.compact_digests(text)
    committed = {"chunk": "chunk0", "compact_text_sha256": sha}
    assert drv.compare_chunk(committed, sha) is None
    altered, _ = sm.compact_digests(text[:-2] + b"X\n")
    assert drv.compare_chunk(committed, altered).startswith("chunk0 compact_text_sha256")
    flipped = dict(committed, compact_text_sha256=sha[:-1] + ("0" if sha[-1] != "0" else "1"))
    assert drv.compare_chunk(flipped, sha) is not None


def test_t14_a_cell_field_that_differs_is_named():
    a = {"quantities": {"x": {"x": "1"}}, "neutron_hist": {"0": 1}, "f2": {"x": "1"}, "N_capt": 4}
    assert drv.compare_cell(a, dict(a)) == []
    assert drv.compare_cell(dict(a, neutron_hist={"0": 2}), a) == ["neutron_hist"]


# --------------------------------------------------------------------------------------------
# the committed files
# --------------------------------------------------------------------------------------------
DATA = REPO / "data" / "g4" / "stopped_mu"
PREREG = DATA / "preregistration.json"
#: The committed pre-registration, pinned: any later edit of it is a visible diff of this line.
PREREG_SHA256 = "44a3944f4b4b2eea5f9d5326db141534b92dd8e8a3f178ef42d79dec773fa200"
gen = _load("generate_stopped_mu_under_test", REPO / "scripts" / "generate_stopped_mu.py")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@functools.lru_cache(maxsize=1)
def policy() -> dict:
    return sm.load_policy(PREREG)


@functools.lru_cache(maxsize=1)
def cells() -> dict:
    return sm.load_cells(DATA / "cells.jsonl", policy())


def cells_copy() -> list[dict]:
    return [json.loads(line) for line in (DATA / "cells.jsonl").read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------------------------
# T1, T2 -- the sources and every chunk are bound to the builds
# --------------------------------------------------------------------------------------------
def source_mismatches(builds: list[dict[str, str]], source: bytes, cmakelists: bytes) -> list[str]:
    s, c = hashlib.sha256(source).hexdigest(), hashlib.sha256(cmakelists).hexdigest()
    return [b["tag"] for b in builds if b["source_sha256"] != s or b["cmakelists_sha256"] != c]


def test_t1_the_committed_sources_are_the_sources_of_every_build():
    builds = read_csv(DATA / "builds.csv")
    assert [b["tag"] for b in builds] == [policy()["setup"]["tags"][v] for v in policy()["setup"]["versions"]]
    source = (REPO / "cpp" / "stopped_muon" / "g4muonic_transport.cc").read_bytes()
    cmakelists = (REPO / "cpp" / "stopped_muon" / "CMakeLists.txt").read_bytes()
    assert source_mismatches(builds, source, cmakelists) == []
    changed = source[:100] + bytes([source[100] ^ 1]) + source[101:]
    assert source_mismatches(builds, changed, cmakelists) == [b["tag"] for b in builds]


def chunk_mismatches(cell_list: list[dict], builds: list[dict[str, str]], tags: dict[str, str]) -> list[str]:
    ref = {b["tag"]: b["reference_binary_sha256"] for b in builds}
    return [
        f"{c['version']}/{c['config']}/{c['target']}/{ch['chunk']}"
        for c in cell_list
        for ch in c["chunks"]
        if ch["binary_sha256"] != ref[tags[c["version"]]]
    ]


def test_t2_every_chunk_names_its_revisions_reference_binary():
    builds = read_csv(DATA / "builds.csv")
    tags = policy()["setup"]["tags"]
    listed = cells_copy()
    assert chunk_mismatches(listed, builds, tags) == []
    assert sum(len(c["chunks"]) for c in listed) == sum(len(c["chunks"]) for c in cells().values())
    listed[5]["chunks"][1]["binary_sha256"] = "0" * 64
    assert chunk_mismatches(listed, builds, tags) == [
        f"{listed[5]['version']}/{listed[5]['config']}/{listed[5]['target']}/chunk1"
    ]


def test_t2_the_reproduced_chunks_are_committed_chunks_of_the_rebuilt_binaries():
    builds = {b["tag"]: b for b in read_csv(DATA / "builds.csv")}
    tags = policy()["setup"]["tags"]
    rows = read_csv(DATA / "reproduced.csv")
    assert rows and tuple(rows[0]) == drv.REPRODUCED_COLUMNS
    for r in rows:
        cell = cells()[(r["version"], r["config"], r["target"])]
        [ch] = [c for c in cell["chunks"] if c["chunk"] == r["chunk"]]
        assert (int(r["offset"]), int(r["events"])) == (ch["offset"], ch["events"])
        assert r["binary_sha256"] == builds[tags[r["version"]]]["rebuilt_binary_sha256"]


# --------------------------------------------------------------------------------------------
# T3 -- cell admission
# --------------------------------------------------------------------------------------------
def test_t3_the_committed_cells_are_the_policys_matrix_and_every_one_is_admitted():
    setup = policy()["setup"]
    want = {
        (v, c, t["target"]) for v in setup["versions"] for c in setup["configs"] for t in setup["targets"]
    }
    assert set(cells()) == want
    assert len(cells()) == len(setup["versions"]) * len(setup["configs"]) * len(setup["targets"])


def _a_cell(key=("11.4.2", "bertini_pc", "natPb")) -> dict:
    return copy.deepcopy(cells()[key])


#: One fault per admission clause, each built so that only its own clause can refuse it.
ADMISSION_FAULTS = [
    ("captures", lambda c: c.update(N_capt=c["N_capt"] - 1), "N_capt"),
    ("X", lambda c: c.update(X=1), "X 1"),
    ("problems", lambda c: c.update(problems=["a refusal"]), "problems"),
    ("extensions", lambda c: c.update(extension_chunks=2), "extension chunks"),
    ("contiguity", lambda c: c["chunks"][1].update(offset=c["chunks"][1]["offset"] + 1), "offset"),
    ("stops", lambda c: c.update(stops_total=c["stops_total"] + 1), r"stops_total \d+ != chunk events"),
    (
        "events",
        lambda c: c.update(events_total=c["events_total"] + 1, decays_total=c["decays_total"] + 1),
        r"events_total \+ n_no_outcome",
    ),
    ("classes", lambda c: c.update(decays_total=c["decays_total"] + 1), r"captures \+ decays"),
    ("no-outcome list", lambda c: c.update(no_outcome_events=[]), "event list"),
    (
        "short",
        lambda c: c.update(
            captures_total=c["N_capt"] - 1,
            decays_total=c["decays_total"] + c["captures_total"] - c["N_capt"] + 1,
        ),
        "captures_total",
    ),
]


@pytest.mark.parametrize("clause, change, match", ADMISSION_FAULTS, ids=[f[0] for f in ADMISSION_FAULTS])
def test_t3_each_admission_clause_refuses_its_fault(clause, change, match):
    cell = _a_cell()
    sm.admit_cell(cell, policy())
    change(cell)
    with pytest.raises(sm.Refused, match=match):
        sm.admit_cell(cell, policy())


def _rate_cell(n: int) -> dict:
    """The natPb cell recut to 106000 stops with ``n`` events without an outcome, every other clause kept."""
    cell = _a_cell()
    cell["chunks"][0]["events"] = 2000
    cell["chunks"][1].update(offset=2000, events=104000)
    cell["stops_total"] = 106000
    cell["n_no_outcome"], cell["no_outcome_events"] = n, list(range(n))
    cell["events_total"] = 106000 - n
    cell["decays_total"] = cell["events_total"] - cell["captures_total"]
    return cell


def test_t3_the_no_outcome_rate_is_admitted_at_its_limit_and_refused_one_event_above():
    limit = Fraction(policy()["rule"]["no_outcome"]["rate_limit"])
    assert Fraction(106, 106000) == limit
    sm.admit_cell(_rate_cell(106), policy())
    with pytest.raises(sm.Refused, match="no-outcome rate"):
        sm.admit_cell(_rate_cell(107), policy())


@pytest.mark.parametrize("gate", ["line_guard", "tag_width"])
def test_t3_an_al27_cell_whose_line_gate_refused_is_not_admitted(gate):
    cell = _a_cell(("11.4.2", "bertini", "Al-27"))
    sm.admit_cell(cell, policy())
    cell[gate]["gamma.Al.1129"]["ok"] = False
    with pytest.raises(sm.Refused, match=gate.replace("_", " ")):
        sm.admit_cell(cell, policy())


def test_t3_a_missing_or_repeated_cell_is_refused(tmp_path):
    lines = (DATA / "cells.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "missing.jsonl").write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(sm.Refused, match="missing"):
        sm.load_cells(tmp_path / "missing.jsonl", policy())
    (tmp_path / "twice.jsonl").write_text("\n".join([lines[0], *lines]) + "\n", encoding="utf-8")
    with pytest.raises(sm.Refused, match="twice"):
        sm.load_cells(tmp_path / "twice.jsonl", policy())


# --------------------------------------------------------------------------------------------
# T4 -- the pre-registration
# --------------------------------------------------------------------------------------------
def q6(x: Fraction) -> str:
    with localcontext() as ctx:
        ctx.prec = 60
        return str(
            (Decimal(x.numerator) / Decimal(x.denominator)).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_EVEN
            )
        )


def derived_references(pol: dict) -> dict[str, tuple[str, str, str]]:
    """Every predicate's (ref, ref_stat, ref_syst) derived from its anchors: AlCap yields per thousand
    captures divided by 1000, gamma yields in percent divided by 100, neutron means and TWIST yields as
    printed, f2 from the printed distribution as S*F2/m1^2 with sigma linearised in independent errors on each
    printed cell."""
    anchors = {a["anchor_id"]: a for a in pol["anchors"]}
    out = {}
    for p in pol["predicates"]:
        fam = p["family"]
        if fam in ("alcap", "gamma", "nmean", "twist"):
            [aid] = p["anchor_ids"]
            a = anchors[aid]
            div = {"alcap": 1000, "gamma": 100}.get(fam, 1)
            ref, st = Fraction(a["value"]) / div, Fraction(a["sigma_stat"]) / div
            sy = Fraction(a["sigma_syst"] or "0") / div
            out[p["predicate"]] = (ref, st, sy)
        else:
            cells_ = [
                (
                    int(aid.rsplit(".", 1)[1]),
                    Fraction(anchors[aid]["value"]),
                    Fraction(anchors[aid]["sigma_stat"]),
                )
                for aid in p["anchor_ids"]
            ]
            S = sum(v for _, v, _ in cells_)
            m1 = sum(k * v for k, v, _ in cells_)
            F2 = sum(k * (k - 1) * v for k, v, _ in cells_)
            var = sum(
                (F2 / m1**2 + S * k * (k - 1) / m1**2 - 2 * S * F2 * k / m1**3) ** 2 * s**2
                for k, _, s in cells_
            )
            with localcontext() as ctx:
                ctx.prec = 60
                sigma = (Decimal(var.numerator) / Decimal(var.denominator)).sqrt()
            out[p["predicate"]] = (S * F2 / m1**2, sigma, Fraction(0))
    return out


def reference_mismatches(pol: dict) -> list[str]:
    bad = []
    for p in pol["predicates"]:
        ref, st, sy = derived_references(pol)[p["predicate"]]
        if p["family"] == "f2":
            with localcontext() as ctx:
                ctx.prec = 60
                sig6 = str(st.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN))
            if (p["ref"], p["ref_stat"], p["ref_syst"]) != (q6(ref), sig6, "0"):
                bad.append(p["predicate"])
        elif (Fraction(p["ref"]), Fraction(p["ref_stat"]), Fraction(p["ref_syst"])) != (ref, st, sy):
            bad.append(p["predicate"])
    return bad


def test_t4_the_committed_pre_registration_is_the_pinned_file():
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == PREREG_SHA256


def test_t4_the_rows_are_every_predicate_on_every_revision_and_configuration():
    pol = policy()
    setup = pol["setup"]
    want = [
        f"{p['predicate']}|{v}|{c}"
        for p in pol["predicates"]
        for v in setup["versions"]
        for c in setup["configs"]
    ]
    assert sorted(r["row_id"] for r in pol["rows"]) == sorted(want)
    by = {p["predicate"]: p for p in pol["predicates"]}
    for r in pol["rows"]:
        p = by[r["predicate"]]
        assert r["row_id"] == f"{r['predicate']}|{r['version']}|{r['config']}"
        assert (r["family"], r["target"], r["ref"], r["ref_stat"], r["ref_syst"]) == (
            p["family"],
            p["target"],
            p["ref"],
            p["ref_stat"],
            p["ref_syst"],
        )
        assert r["rescue"] is (p["family"] == "nmean")
    targets = {t["target"]: t["basis"] for t in setup["targets"]}
    assert all(p["basis"] == targets[p["target"]] for p in pol["predicates"])


def test_t4_every_reference_derives_from_its_anchors_and_a_perturbed_anchor_is_caught():
    pol = policy()
    assert reference_mismatches(pol) == []
    for aid, pred in (("alcap.Ti.d", "alcap.Ti.d"), ("pn545.Ag.3", "f2.Ag"), ("nmean.Pb", "nmean.Pb")):
        bad = copy.deepcopy(pol)
        [a] = [x for x in bad["anchors"] if x["anchor_id"] == aid]
        a["value"] = str(Decimal(a["value"]) + Decimal("0.001"))
        assert reference_mismatches(bad) == [pred]


def test_t4_the_twist_thresholds_follow_from_the_momentum_cuts_and_masses():
    pol = policy()
    tw = pol["twist"]
    particle = {"twist.Al.p": "proton", "twist.Al.d": "deuteron"}
    for p in pol["predicates"]:
        if p["family"] != "twist":
            continue
        with localcontext() as ctx:
            ctx.prec = 60
            pc = Decimal(tw["momentum_cut_MeV_per_c"][p["predicate"]])
            m = Decimal(tw["masses_MeV"][particle[p["predicate"]]])
            ke = ((pc * pc + m * m).sqrt() - m).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
        assert p["ke_lo_MeV"] == str(ke) and p["lower_edge"] == "exclusive" and p["ke_hi_MeV"] == ""


# --------------------------------------------------------------------------------------------
# T8 -- the rule
# --------------------------------------------------------------------------------------------
def sigma2(r: dict) -> Fraction:
    return Fraction(r["ref_stat"]) ** 2 + Fraction(r["ref_syst"]) ** 2


def placed(
    x: Fraction,
    r: dict,
    *,
    beyond: int | None = None,
    within: int | None = None,
    inside: bool | None = None,
    edges: tuple[str, str] | None = None,
) -> None:
    """Assert, before any verdict, that a fixture sits where it claims (raises AssertionError as a fixture
    error)."""
    d2 = (x - Fraction(r["ref"])) ** 2
    if beyond is not None:
        assert d2 > beyond * beyond * sigma2(r), ("fixture inside the band", r["row_id"], beyond)
    if within is not None:
        assert d2 <= within * within * sigma2(r), ("fixture outside the band", r["row_id"], within)
    if inside is not None:
        assert edges is not None
        assert (Fraction(edges[0]) <= x <= Fraction(edges[1])) is inside, ("fixture bracket", r["row_id"])


def as_decimal(x: Fraction) -> str:
    """A terminating fraction as its exact decimal string."""
    with localcontext() as ctx:
        ctx.prec = 80
        d = Decimal(x.numerator) / Decimal(x.denominator)
    assert Fraction(d) == x, "not a terminating decimal"
    return format(d.normalize(), "f")


def comparison_of(pol: dict, xs: dict[str, str]) -> list[dict[str, str]]:
    out = []
    for r in pol["rows"]:
        lo, hi = sm.edges(r["ref"], r["ref_stat"], r["ref_syst"], "0", pol) if r["rescue"] else ("", "")
        out.append(
            {
                "row_id": r["row_id"],
                "predicate": r["predicate"],
                "family": r["family"],
                "target": r["target"],
                "version": r["version"],
                "config": r["config"],
                "n_capt": "100000",
                "x": xs.get(r["row_id"], r["ref"]),
                "x_se": "0",
                "ref": r["ref"],
                "ref_stat": r["ref_stat"],
                "ref_syst": r["ref_syst"],
                "rescue_lo": lo,
                "rescue_hi": hi,
            }
        )
    return out


def verdicts_of(pol: dict, xs: dict[str, str]) -> dict[str, tuple[str, str]]:
    return {v.row_id: (v.verdict, v.kind) for v in sm.judge(pol, comparison_of(pol, xs))}


def sigma_tot(r: dict) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 60
        return (Decimal(r["ref_stat"]) ** 2 + Decimal(r["ref_syst"]) ** 2).sqrt()


def q12(x: Decimal) -> str:
    with localcontext() as ctx:
        ctx.prec = 60
        return str(x.quantize(Decimal("1e-12")))


def test_t8_every_row_passes_at_its_reference_and_is_tension_at_two_and_a_half_sigma():
    pol = policy()
    assert {v for v, _ in verdicts_of(pol, {}).values()} == {"PASS"}
    xs = {}
    for r in pol["rows"]:
        with localcontext() as ctx:
            ctx.prec = 60
            x = q12(Decimal(r["ref"]) + Decimal("2.5") * sigma_tot(r))
        placed(Fraction(x), r, beyond=2, within=3)
        xs[r["row_id"]] = x
    got = verdicts_of(pol, xs)
    assert set(got.values()) == {("UNQUALIFIED", "tension")}
    assert len(got) == len(pol["rows"])


def test_t8_rows_without_a_bracket_fail_four_sigma_away_on_either_side():
    pol = policy()
    rows = [r for r in pol["rows"] if not r["rescue"]]
    assert rows
    for sign in (1, -1):
        xs = {}
        for r in rows:
            x = Fraction(r["ref"]) + sign * 4 * (Fraction(r["ref_stat"]) + Fraction(r["ref_syst"]))
            placed(x, r, beyond=3)
            xs[r["row_id"]] = as_decimal(x)
        got = verdicts_of(pol, xs)
        assert {got[r["row_id"]] for r in rows} == {("FAIL", "fail")}


def test_t8_rows_with_a_bracket_are_efficiency_limited_inside_it_and_fail_beyond_it():
    pol = policy()
    rows = [r for r in pol["rows"] if r["rescue"]]
    assert rows and all(r["family"] == "nmean" for r in rows)
    cases = {"efficiency_limited": [], "fail": []}
    for r in rows:
        lo, hi = sm.edges(r["ref"], r["ref_stat"], r["ref_syst"], "0", pol)
        s = Fraction(r["ref_stat"]) + Fraction(r["ref_syst"])
        ref = Fraction(r["ref"])
        for sign in (1, -1):  # ref +- 4 sigma_tot lies inside the bracket on these rows
            with localcontext() as ctx:
                ctx.prec = 60
                x = Fraction(q12(Decimal(r["ref"]) + sign * 4 * sigma_tot(r)))
            placed(x, r, beyond=3, inside=True, edges=(lo, hi))
            cases["efficiency_limited"].append((r["row_id"], x))
        for x in ((ref + 3 * s + Fraction(hi)) / 2, (ref - 3 * s + Fraction(lo)) / 2):
            placed(x, r, beyond=3, inside=True, edges=(lo, hi))
            cases["efficiency_limited"].append((r["row_id"], x))
        for x in (Fraction(hi) + s, Fraction(lo) - s):
            placed(x, r, beyond=3, inside=False, edges=(lo, hi))
            cases["fail"].append((r["row_id"], x))
    for kind, fixtures in cases.items():
        for i in range(4 if kind == "efficiency_limited" else 2):
            xs = {rid: as_decimal(x) for rid, x in fixtures[i :: 4 if kind == "efficiency_limited" else 2]}
            got = verdicts_of(pol, xs)
            assert {got[rid] for rid in xs} == {("UNQUALIFIED" if kind != "fail" else "FAIL", kind)}, (
                kind,
                i,
            )


def two_row_policy() -> dict:
    pol = copy.deepcopy(policy())
    base = {
        "family": "x",
        "target": "T",
        "version": "V",
        "config": "c",
        "ref": "1",
        "ref_stat": "0.03",
        "ref_syst": "0.04",
    }
    pol["rows"] = [
        dict(base, row_id="plain", predicate="plain", rescue=False),
        dict(base, row_id="bracket", predicate="bracket", rescue=True),
    ]
    return pol


def test_t8_the_band_and_bracket_edges_are_inside_and_one_quantum_beyond_them_is_not():
    pol = two_row_policy()
    assert sm.edges("1", "0.03", "0.04", "0", pol) == ("0.712640000000", "1.566875000000")

    def verdict(row, x):
        return verdicts_of(pol, {row: x})[row]

    assert verdict("plain", "1.10") == ("PASS", "pass") and verdict("plain", "0.90") == ("PASS", "pass")
    assert verdict("plain", "1.100000000001") == ("UNQUALIFIED", "tension")
    assert verdict("plain", "1.15") == ("UNQUALIFIED", "tension")
    assert verdict("plain", "1.150000000001") == ("FAIL", "fail")
    assert verdict("bracket", "1.566875") == ("UNQUALIFIED", "efficiency_limited")
    assert verdict("bracket", "0.71264") == ("UNQUALIFIED", "efficiency_limited")
    assert verdict("bracket", "1.566875000001") == ("FAIL", "fail")
    assert verdict("bracket", "0.712639999999") == ("FAIL", "fail")


@pytest.mark.parametrize("fault", ["missing", "extra", "repeated", "ref", "edge", "edge on plain row"])
def test_t8_a_roster_fault_is_refused_never_judged(fault):
    pol = two_row_policy()
    rows = comparison_of(pol, {})
    if fault == "missing":
        rows = rows[1:]
    elif fault == "extra":
        rows.append(dict(rows[0], row_id="other"))
    elif fault == "repeated":
        rows.append(dict(rows[0]))
    elif fault == "ref":
        rows[0]["ref"] = "1.1"
    elif fault == "edge":
        rows[1]["rescue_hi"] = "1.566875000002"
    elif fault == "edge on plain row":
        rows[0]["rescue_lo"] = "0.5"
    with pytest.raises(sm.Refused):
        sm.judge(pol, rows)


def test_t8_the_regenerated_comparison_and_verdicts_are_the_committed_files(tmp_path):
    _, _, rows, verdicts = gen.comparison()
    gen.write_csv(tmp_path / "comparison.csv", sm.COMPARISON_COLUMNS, rows)
    gen.write_csv(
        tmp_path / "verdicts.csv",
        sm.VERDICT_COLUMNS,
        [{"row_id": v.row_id, "verdict": v.verdict, "kind": v.kind, "pull": v.pull} for v in verdicts],
    )
    assert (tmp_path / "comparison.csv").read_bytes() == (DATA / "comparison.csv").read_bytes()
    assert (tmp_path / "verdicts.csv").read_bytes() == (DATA / "verdicts.csv").read_bytes()


# --------------------------------------------------------------------------------------------
# T9 - T12 -- the generated document
# --------------------------------------------------------------------------------------------
DOCUMENT = "BENCHMARK_STOPPED_MU.md"


@functools.lru_cache(maxsize=1)
def rendered():
    return gen.document()


def fresh_inputs():
    return copy.deepcopy(gen.load_inputs())


def test_t9_the_committed_document_is_the_rendered_document():
    assert (REPO / DOCUMENT).read_text(encoding="utf-8") == rendered().text


def test_t9_the_generator_writes_the_three_files_byte_for_byte(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "DATA", DATA)
    out = {}

    def capture(path, fields, rows):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        out[Path(path).name] = stream.getvalue().encode("utf-8")

    monkeypatch.setattr(gen, "write_csv", capture)
    monkeypatch.setattr(gen, "DOCUMENT", tmp_path / DOCUMENT)
    gen.compare()
    assert out["comparison.csv"] == (DATA / "comparison.csv").read_bytes()
    assert out["verdicts.csv"] == (DATA / "verdicts.csv").read_bytes()
    # The generator writes LF; the committed file is read with universal newlines, so a checkout that writes
    # CRLF compares equal while any other byte of difference does not (the CSV files are -text: bytes as
    # committed).
    assert (tmp_path / DOCUMENT).read_bytes() == (REPO / DOCUMENT).read_text(encoding="utf-8").encode("utf-8")


def rejudge(inp) -> None:
    inp.verdicts = sm.judge(inp.policy, inp.rows)


def test_t10_a_revision_that_differs_lists_the_pair_and_every_row():
    base = rendered().text
    assert "so the tables below list the" in base
    assert "| predicate | target | revision | configuration |" not in base
    inp = fresh_inputs()
    v2 = inp.policy["setup"]["versions"][1]
    [row] = [r for r in inp.rows if r["row_id"] == f"alcap.Al.p|{v2}|bertini"]
    row["x"] = "0.02664"
    rejudge(inp)
    text = gen.render(inp).text
    assert "These " in text and "row pairs differ between the revisions" in text
    assert "| predicate | target | revision | configuration |" in text
    setup = inp.policy["setup"]
    for v in setup["versions"]:
        assert text.count(f"| `alcap.Al.p` | Al-27 | {v} |") == len(setup["configs"])


def test_t10_a_line_yield_beyond_its_error_is_stated_with_its_error():
    base = rendered().text
    zero = base.count("the net line yield consistent with zero")
    assert zero >= 1
    inp = fresh_inputs()
    for r in inp.rows:
        if r["predicate"] == "gamma.Al.1808" and r["config"] == "bertini":
            r["x"] = str(5 * Decimal(r["x_se"]))
    rejudge(inp)
    text = gen.render(inp).text
    assert text.count("the net line yield consistent with zero") == zero - 1
    x = next(r["x"] for r in inp.rows if r["predicate"] == "gamma.Al.1808" and r["config"] == "bertini")
    assert f"x = {x} ± " in text


def test_t10_an_alternative_value_that_crosses_a_band_is_listed():
    assert "changes 0 of their verdicts." in rendered().text
    inp = fresh_inputs()
    for cell in inp.cells.values():
        if "alternative" in cell:
            cell["alternative"]["quantities"]["nmean.Pb"]["x"] = "1.71"
    text = gen.render(inp).text
    changed = sum(1 for c in inp.cells.values() if "alternative" in c)
    assert f"changes {changed} of their verdicts:" in text
    assert "FAIL under the rule, PASS under the alternative." in text


def test_t10_a_reproduced_chunk_that_differs_is_named():
    assert "gave compact text with the sha256 the committed cell records" in rendered().text
    inp = fresh_inputs()
    inp.reproduced[0]["compact_text_sha256"] = "0" * 64
    text = gen.render(inp).text
    assert "which the committed cell does not record" in text
    assert "gave compact text with the sha256 the committed cell records" not in text
    assert "was run with binary sha256" not in text


def test_t10_the_committed_reproduction_is_admitted_with_no_fault():
    inp = fresh_inputs()
    cells_, faults = gen.reproduction(inp)
    assert faults == []
    assert cells_ == list(dict.fromkeys((r["version"], r["config"], r["target"]) for r in inp.reproduced))
    assert len(inp.reproduced) == sum(len(inp.cells[k]["chunks"]) for k in cells_)


def test_t10_a_chunk_run_with_another_binary_is_named_without_denying_its_compact_text():
    inp = fresh_inputs()
    r = inp.reproduced[0]
    want = next(b for b in inp.builds if b["tag"] == inp.policy["setup"]["tags"][r["version"]])
    assert r["binary_sha256"] == want["rebuilt_binary_sha256"]
    r["binary_sha256"] = "e" * 64
    text = gen.render(inp).text
    line = (
        f"- {r['version']} {r['config']} {r['target']} {r['chunk']} was run with binary sha256 {'e' * 64}, "
        f"which differs from the recorded rebuilt binary sha256 {want['rebuilt_binary_sha256']}."
    )
    assert line in text.splitlines()
    assert "which the committed cell does not record" not in text
    assert "gave compact text with the sha256 the committed cell records" not in text


def _reproduction_fault(inp, fault: str) -> None:
    rows = inp.reproduced
    cell_of = {(r["version"], r["config"], r["target"]) for r in rows}
    if fault == "empty":
        rows.clear()
    elif fault == "missing row":
        del rows[1]
    elif fault == "duplicate row":
        rows[1] = dict(rows[0])
    elif fault == "repeated row":
        rows.insert(1, dict(rows[0]))
    elif fault == "unknown cell":
        rows.append(dict(rows[0], target="natXx"))
    elif fault == "unknown chunk":
        rows[1] = dict(rows[1], chunk="chunk7")
    elif fault == "offset":
        rows[1]["offset"] = str(int(rows[1]["offset"]) + 1)
    elif fault == "events":
        rows[1]["events"] = str(int(rows[1]["events"]) + 1)
    elif fault == "coverage":
        # every row of the cells of the second revision dropped: that revision is not covered
        v2 = inp.policy["setup"]["versions"][1]
        assert any(k[0] == v2 for k in cell_of)
        rows[:] = [r for r in rows if r["version"] != v2]
    elif fault == "no-outcome coverage":
        # every row of the cells holding an event without an outcome replaced by the rows of a cell of the
        # same revision, configuration and target basis that holds none, so only that coverage is lost
        basis = {t["target"]: t["basis"] for t in inp.policy["setup"]["targets"]}
        held = {k for k in cell_of if inp.cells[k]["n_no_outcome"]}
        assert held
        rows[:] = [r for r in rows if (r["version"], r["config"], r["target"]) not in held]
        for k in held:
            other = next(
                key
                for key, c in sorted(inp.cells.items())
                if key[:2] == k[:2] and not c["n_no_outcome"] and basis[key[2]] == basis[k[2]]
            )
            rows += [
                {
                    "version": other[0],
                    "config": other[1],
                    "target": other[2],
                    "chunk": ch["chunk"],
                    "offset": str(ch["offset"]),
                    "events": str(ch["events"]),
                    "compact_text_sha256": ch["compact_text_sha256"],
                    "binary_sha256": next(
                        b["rebuilt_binary_sha256"]
                        for b in inp.builds
                        if b["tag"] == inp.policy["setup"]["tags"][other[0]]
                    ),
                }
                for ch in inp.cells[other]["chunks"]
            ]


REPRODUCTION_FAULTS = [
    ("empty", "vacuous"),
    ("missing row", "lists the chunks"),
    ("duplicate row", "lists the chunks"),
    ("repeated row", "lists the chunks"),
    ("unknown cell", "which the committed cells do not hold"),
    ("unknown chunk", "lists the chunks"),
    ("offset", r"chunk1 offset \d+, the committed chunk"),
    ("events", r"chunk1 events \d+, the committed chunk"),
    ("coverage", "do not cover the revisions"),
    ("no-outcome coverage", "do not cover the events without an outcome of"),
]


@pytest.mark.parametrize("fault, match", REPRODUCTION_FAULTS, ids=[f[0] for f in REPRODUCTION_FAULTS])
def test_t10_a_reproduction_record_the_screen_cannot_judge_is_refused_before_the_document(fault, match):
    inp = fresh_inputs()
    gen.reproduction(inp)
    _reproduction_fault(inp, fault)
    with pytest.raises(ValueError, match=match):
        gen.reproduction(inp)
    with pytest.raises(ValueError, match=match):
        gen.render(inp)


@pytest.mark.parametrize(
    "old, replacement, configurations, bases, clause",
    [
        (
            ("11.5.0.beta", "precompound", "natSi"),
            ("11.5.0.beta", "bertini_pc", "natPb"),
            {"bertini", "bertini_pc"},
            {"natural", "single_isotope"},
            "configurations",
        ),
        (
            ("11.4.2", "bertini", "Al-27"),
            ("11.4.2", "bertini", "natSi"),
            {"bertini", "bertini_pc", "precompound"},
            {"natural"},
            "target bases",
        ),
    ],
    ids=["configuration coverage", "target basis coverage"],
)
def test_t10_a_complete_reproduction_missing_only_one_coverage_dimension_is_refused(
    old, replacement, configurations, bases, clause
):
    inp = fresh_inputs()
    original = copy.deepcopy(inp.reproduced)
    retained = [r for r in original if (r["version"], r["config"], r["target"]) != old]
    assert len(retained) < len(original)
    version, config, target = replacement
    binary = next(
        b["rebuilt_binary_sha256"] for b in inp.builds if b["tag"] == inp.policy["setup"]["tags"][version]
    )
    # These complete synthetic records exercise coverage; they are not additional native executions.
    inp.reproduced = retained + [
        {
            "version": version,
            "config": config,
            "target": target,
            "chunk": ch["chunk"],
            "offset": str(ch["offset"]),
            "events": str(ch["events"]),
            "compact_text_sha256": ch["compact_text_sha256"],
            "binary_sha256": binary,
        }
        for ch in inp.cells[replacement]["chunks"]
    ]
    assert inp.reproduced[: len(retained)] == retained
    keys = {(r["version"], r["config"], r["target"]) for r in inp.reproduced}
    assert keys == {(r["version"], r["config"], r["target"]) for r in retained} | {replacement}
    assert len(keys) == 3 and old not in keys
    # Independently check complete, distinct, hash-consistent chunks before exercising the coverage guard.
    for key in keys:
        rows = [r for r in inp.reproduced if (r["version"], r["config"], r["target"]) == key]
        chunks = inp.cells[key]["chunks"]
        assert Counter(r["chunk"] for r in rows) == Counter(ch["chunk"] for ch in chunks)
        assert all(n == 1 for n in Counter(r["chunk"] for r in rows).values())
        for r in rows:
            [ch] = [ch for ch in chunks if ch["chunk"] == r["chunk"]]
            assert (r["offset"], r["events"], r["compact_text_sha256"]) == (
                str(ch["offset"]), str(ch["events"]), ch["compact_text_sha256"]
            )
            assert r["binary_sha256"] == next(
                b["rebuilt_binary_sha256"]
                for b in inp.builds
                if b["tag"] == inp.policy["setup"]["tags"][key[0]]
            )
    assert {k[0] for k in keys} == {"11.4.2", "11.5.0.beta"}
    assert {k[1] for k in keys} == configurations
    target_basis = {t["target"]: t["basis"] for t in inp.policy["setup"]["targets"]}
    assert {target_basis[k[2]] for k in keys} == bases
    assert any(inp.cells[k]["n_no_outcome"] for k in keys)
    message = f"the reproduced cells do not cover the {clause} of the committed cells"
    for consumer in (gen.reproduction, gen.render):
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            consumer(inp)


def test_t10_the_chunk_body_statement_follows_the_bodies():
    inp = fresh_inputs()
    same, diff = gen.body_pairs(inp)
    assert diff and f"{same} have identical compact bodies" in rendered().text
    v1, v2 = inp.policy["setup"]["versions"]
    for (v, c, t), cell in inp.cells.items():
        if v == v2:
            for a, b in zip(cell["chunks"], inp.cells[(v1, c, t)]["chunks"], strict=True):
                a["compact_body_sha256"] = b["compact_body_sha256"]
    text = gen.render(inp).text
    assert f"All {same + diff} pairs of chunks" in text


def test_t10_a_rebuild_that_differs_is_stated():
    assert "the rebuild of the committed sources is byte-identical" in rendered().text
    inp = fresh_inputs()
    inp.builds[1]["rebuilt_binary_sha256"] = "f" * 64
    text = gen.render(inp).text
    assert "differs from the binary that made the committed cells" in text


def document_pins(numeric):
    """`(what, path, pattern, groups, expected)` for every line of the document that carries a slot: the
    line, whitespace-collapsed and escaped, with each span the data filled as a captured group. The values
    themselves are held by the byte-for-byte regeneration of the document (T9); ``numeric`` is the prose
    check's tokenizer, kept in the signature the other document modules share."""
    r = rendered()
    lines = r.text.splitlines()
    by_line: dict[int, list[tuple[int, int]]] = {}
    for lineno, start, end in r.slots:
        by_line.setdefault(lineno, []).append((start, end))
    pins = []
    for lineno, spans in sorted(by_line.items()):
        line = lines[lineno - 1]
        assert line == " ".join(line.split()), "a generated line carries whitespace the collapse would change"
        parts, at = [], 0
        for start, end in sorted(spans):
            parts.append(re.escape(line[at:start]))
            parts.append(f"({re.escape(line[start:end])})")
            at = end
        parts.append(re.escape(line[at:]))
        pins.append(
            (f"{DOCUMENT} line {lineno}", DOCUMENT, "".join(parts), tuple(range(1, len(spans) + 1)), None)
        )
    return pins


def test_t11_every_slot_line_has_a_pin_whose_groups_are_its_slots():
    r = rendered()
    lines = r.text.splitlines()
    pins = document_pins(None)
    assert len(pins) == len({lineno for lineno, _, _ in r.slots})
    for what, _, pattern, groups, _ in pins:
        lineno = int(what.rsplit(" ", 1)[1])
        hit = re.fullmatch(pattern, lines[lineno - 1])
        assert hit is not None, what
        assert [hit.span(g) for g in groups] == sorted((s, e) for n, s, e in r.slots if n == lineno), what


def test_t11_drill_a_digit_in_a_template_literal_is_left_to_the_census(monkeypatch):
    import test_g4prose as prose

    original = gen.Writer.line

    def planted(self, *parts):
        if parts and parts[0] == "**The verdicts.** Of the ":
            parts = (*parts, " (7 in all)")
        original(self, *parts)

    monkeypatch.setattr(gen.Writer, "line", planted)
    rendered.cache_clear()
    try:
        r = gen.document()
        monkeypatch.setattr(gen, "document", lambda: r)
        pins = [prose.Pin(w, p, pat, g, e) for w, p, pat, g, e in document_pins(prose.NUMERIC)]
        open_tokens, problems = prose.unadmitted_by_pins_and_classes(
            {DOCUMENT: r.text}, pins, prose.read_classes()
        )
        assert not problems
        lineno = next(i for i, line in enumerate(r.text.splitlines(), 1) if "(7 in all)" in line)
        assert [t.text for t in open_tokens[(DOCUMENT, lineno)]] == ["7"]
    finally:
        rendered.cache_clear()


@pytest.mark.parametrize("path", [DOCUMENT, "cpp/stopped_muon/README.md"])
def test_t12_the_document_and_the_application_readme_carry_none_of_the_excluded_words(path):
    text = (REPO / path).read_text(encoding="utf-8")
    assert gen.forbidden_hits(text) == []
    assert gen.forbidden_hits(text + "\nthe first capture") == ["first"]


def test_t10_a_cell_run_with_the_dataset_opt_in_on_is_refused():
    inp = fresh_inputs()
    opt = inp.policy["setup"]["harness_options"]
    opt[opt.index("--opt-in") + 1] = "on"
    with pytest.raises(ValueError, match="opt-in on"):
        gen.render(inp)


def test_t10_a_rescue_bracket_on_a_row_that_is_not_a_neutron_mean_is_refused():
    inp = fresh_inputs()
    [row] = [r for r in inp.policy["rows"] if r["row_id"].startswith("alcap.Al.p|")][:1]
    row["rescue"] = True
    with pytest.raises(ValueError, match="not the neutron-mean rows"):
        gen.render(inp)


def test_t10_cells_without_an_event_lacking_an_outcome_say_none():
    assert "\n- none.\n" not in rendered().text
    inp = fresh_inputs()
    for cell in inp.cells.values():
        cell["n_no_outcome"] = 0
        cell["no_outcome_events"] = []
        cell.pop("alternative", None)
    text = gen.render(inp).text
    assert "\n- none.\n" in text
    assert "the alternative values of the 0 rows of these cells changes 0 of their verdicts." in text


def test_t10_the_single_isotope_is_called_natural_only_when_the_natural_element_has_that_isotope_alone():
    sentence = "is also the element of natural composition"
    assert rendered().text.count(sentence) == 1
    inp = fresh_inputs()
    [al] = [r for r in inp.natural if r["z"] == "13"]
    inp.natural.append(dict(al, a="26", abundance="0.0"))
    assert sentence not in gen.render(inp).text


# --------------------------------------------------------------------------------------------
# T15 -- every comparator at its equality edge, and every member of a binding on its own
# --------------------------------------------------------------------------------------------
def binary_policy(**gamma_overrides: str) -> dict:
    """The synthetic policy with a gamma part whose values are exact binary fractions, so that every distance
    a comparator sees below is computed without rounding: A = 32 and 1 MeV per u make the recoil of a 2 MeV
    transition 4 / 64 = 0.0625 MeV, so E_t = 1.9375 MeV."""
    pol = synthetic_policy(captures=4)
    pol["gamma"].update(
        {
            "lines_MeV": {"gamma.Al.1808": "2", "gamma.Al.1129": "1"},
            "mev_per_u": "1",
            "residual_z_a": [12, 32],
            "tag_half_width_MeV": "0.0625",
            "sideband_MeV": ["0.125", "0.5"],
        }
    )
    pol["guards"]["per_line"].update({"fraction": "0.5", "beta_max": "0.0625"})
    pol["guards"]["tag_width"].update({"inner_MeV": "0.125"})
    for key, value in gamma_overrides.items():
        pol["guards"]["tag_width" if key == "inner_MeV" else "per_line"][key] = value
    return pol


RES32 = 1000120320  # the (Z, A) = (12, 32) residual of the binary policy


def at_rest_capture(*photons: float, residual: str | None = None, desc: str = "") -> str:
    """A compact capture line: a residual at rest (beta 0, so a photon's rest-frame energy is its energy) and
    photons of the given energies."""
    res = residual or f"{RES32}:{h(0.0)}:{h(0.0)}:{h(0.0)}:{h(0.0)}"
    toks = [res] + [f"22:{h(e)}:{h(e)}:{h(0.0)}:{h(0.0)}" for e in photons]
    return f"0 C {h(0.0)} " + " ".join(toks) + desc


def test_t15_the_binary_policy_is_exact():
    gp = sm.gamma_params(binary_policy())
    assert sm.e_t(gp.lines["gamma.Al.1808"], gp) == 1.9375
    assert (gp.tag_half, gp.sb_lo, gp.sb_hi, gp.inner) == (0.0625, 0.125, 0.5, 0.125)
    assert 1.9375 * gp.guard_beta_max + gp.tag_half == 0.18359375


UP = math.inf


@pytest.mark.parametrize(
    "energy, w, s",
    [
        (2.0, 1, 0),  # |E* - E_t| = the tag half-width: in the tag
        (math.nextafter(2.0, UP), 0, 0),
        (2.0625, 0, 0),  # = the lower sideband edge, which the sideband excludes
        (math.nextafter(2.0625, UP), 0, 1),
        (2.4375, 0, 1),  # = the upper sideband edge, which the sideband includes
        (math.nextafter(2.4375, UP), 0, 0),
    ],
)
def test_t15_the_tag_and_sideband_edges_of_a_capture(energy, w, s):
    gp = sm.gamma_params(binary_policy())
    prods, _ = sm.parse_line(at_rest_capture(energy))
    assert sm.capture_counts(prods, 1.9375, gp) == (w, s, 0.0)


@pytest.mark.parametrize(
    "energy, n1, n3, s",
    [
        (2.0, 1, 0, 0),  # = the tag half-width: in the tag
        (math.nextafter(2.0, UP), 0, 1, 0),
        (2.0625, 0, 1, 0),  # = the inner edge, which the inner band includes
        (math.nextafter(2.0625, UP), 0, 0, 1),
        (2.4375, 0, 0, 1),  # = the upper sideband edge
        (math.nextafter(2.4375, UP), 0, 0, 0),
    ],
)
def test_t15_the_bands_of_the_tag_width_gate(energy, n1, n3, s):
    gp = sm.gamma_params(binary_policy())
    ok, got_n1, got_n3, expect = sm.tag_width_gate([at_rest_capture(energy)], gp.lines["gamma.Al.1808"], gp)
    assert (got_n1, got_n3, expect) == (n1, n3, gp.cont_num * s / gp.cont_den)
    assert ok


def test_t15_the_lower_sideband_edge_of_the_tag_width_gate_with_an_inner_band_below_it():
    # With the committed inner edge equal to the lower sideband edge, |E* - E_t| = sb_lo falls in the inner
    # band, and the strict lower sideband edge cannot change a count; an inner edge below the sideband
    # exposes it.
    gp = sm.gamma_params(binary_policy(inner_MeV="0.09375"))
    for energy, s in ((2.0625, 0), (math.nextafter(2.0625, UP), 1)):
        _, n1, n3, expect = sm.tag_width_gate([at_rest_capture(energy)], gp.lines["gamma.Al.1808"], gp)
        assert (n1, n3, expect) == (0, 0, gp.cont_num * s / gp.cont_den)


@pytest.mark.parametrize("n3, ok", [(15, True), (16, False)])
def test_t15_the_tag_width_gate_admits_its_threshold_and_refuses_one_count_more(n3, ok):
    gp = sm.gamma_params(binary_policy())
    # 20 sideband photons predict a continuum of 4 x 20 / 20 = 4 counts: the threshold is
    # 4 + 3 sqrt(4) + 5 = 15
    lines = [at_rest_capture(*([2.25] * 20 + [2.03125] * n3))]
    good, n1, got_n3, expect = sm.tag_width_gate(lines, gp.lines["gamma.Al.1808"], gp)
    assert (n1, got_n3, expect) == (0, n3, 4.0)
    assert expect + gp.k * math.sqrt(expect) + gp.offset == 15
    assert good is ok


def descendant(ke: float, dt: float) -> str:
    return f" g22:{h(ke)}:{h(dt)}:{RES32}"


@pytest.mark.parametrize(
    "ke, dt, counted",
    [
        (1.9375, 1000.0, False),  # emitted exactly the window after the capture: outside it
        (1.9375, math.nextafter(1000.0, 0.0), True),
        (2.12109375, 10.0, True),  # |E - E_t| = E_t beta_max + the tag half-width: inside the energy window
        (math.nextafter(2.12109375, UP), 10.0, False),
    ],
)
def test_t15_the_time_and_energy_edges_of_a_descendant(ke, dt, counted):
    gp = sm.gamma_params(binary_policy())
    lines = [at_rest_capture(2.0, desc=descendant(ke, dt))]
    ok, c, d, bmax = sm.line_guard(lines, gp.lines["gamma.Al.1808"], gp)
    assert (c, d, bmax) == (1, int(counted), 0.0)
    assert ok  # one descendant against one tagged photon is exactly the fraction 0.5


@pytest.mark.parametrize("d_line, ok", [(1, True), (2, False)])
def test_t15_the_descendant_fraction_is_admitted_at_its_limit(d_line, ok):
    gp = sm.gamma_params(binary_policy())
    lines = [at_rest_capture(2.0, desc=descendant(1.9375, 10.0) * d_line)]
    good, c, d, bmax = sm.line_guard(lines, gp.lines["gamma.Al.1808"], gp)
    assert (c, d, bmax) == (1, d_line, 0.0)
    assert Fraction(d, c + d) == (Fraction(1, 2) if ok else Fraction(2, 3))
    assert good is ok


def test_t15_a_tagged_residual_at_exactly_the_beta_limit_is_refused():
    # A residual of KE 0.5 MeV and momentum 4 MeV/c along z: m = (16 - 0.25) / 1 = 15.75, E = 16.25 and
    # beta = 4 / 16.25, its velocity a single component, so |beta| is that quotient exactly.
    beta = 4.0 / 16.25
    g = 1 / math.sqrt(1 - beta * beta)
    e = 1.9375 / g  # along x, perpendicular to the residual's velocity: rest-frame energy g e = E_t
    res = f"{RES32}:{h(0.5)}:{h(0.0)}:{h(0.0)}:{h(4.0)}"
    line = f"0 C {h(0.0)} {res} 22:{h(e)}:{h(e)}:{h(0.0)}:{h(0.0)}"
    for beta_max, ok in ((repr(beta), False), (repr(math.nextafter(beta, UP)), True)):
        gp = sm.gamma_params(binary_policy(beta_max=beta_max))
        good, c, d, bmax = sm.line_guard([line], gp.lines["gamma.Al.1808"], gp)
        assert (c, d, bmax) == (1, 0, beta)
        assert (bmax == float(beta_max)) is (not ok)
        assert good is ok


@pytest.mark.parametrize("offset, kept", [(1000.0, False), (math.nextafter(1000.0, 0.0), True)])
def test_t15_the_classifier_keeps_a_descendant_photon_only_inside_the_window(offset, kept):
    lines = [
        "T 0 1 0 13 - - 0x0p+0 0x0p+0\n",
        "S 0 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 1\n",
        c_line(0, 1, 0, 1000120260, NC, 0.25, 0.0),
        f"T 0 2 1 1000120260 muMinusCaptureAtRest {NC} {h(0.25)} {h(0.0)}\n",
        f"T 0 3 2 22 - - {h(1.80874)} {h(offset)}\n",
        "E 0 1000 3\n",
    ]
    got, st = run_stream(lines, "bound_decay", 1)
    assert float(synthetic_policy()["classification"]["descendant_photon_window_ns"]) == 1000.0
    assert (f"g22:{h(1.80874)}:{h(offset)}:1000120260" in got[1]) is kept
    assert st["g_tokens"] == int(kept)


def test_t15_one_extension_chunk_is_admitted_by_the_reducer():
    parts = [
        chunk("chunk0", 0, [cap(0, CAPTURES[0]), cap(1, CAPTURES[1])]),
        chunk("chunk1", 2, [cap(2, CAPTURES[2]), cap(3, CAPTURES[3])], extension=True),
    ]
    cell, status = reduce(parts)
    assert synthetic_policy()["guards"]["max_extension_chunks"] == 1
    assert (status, cell["problems"], cell["extension_chunks"]) == ("accepted", [], 1)


def _extensions(cell: dict, flags: tuple[bool, ...], count: int) -> dict:
    for c, flag in zip(cell["chunks"], flags, strict=True):
        c["extension"] = flag
    cell["extension_chunks"] = count
    return cell


def test_t15_admission_takes_one_extension_chunk_and_refuses_two_and_a_wrong_tally():
    assert policy()["guards"]["max_extension_chunks"] == 1
    assert len(_a_cell()["chunks"]) == 2
    sm.admit_cell(_extensions(_a_cell(), (False, True), 1), policy())
    with pytest.raises(sm.Refused, match="extension chunks 2"):
        sm.admit_cell(_extensions(_a_cell(), (True, True), 2), policy())  # the tally holds: the limit refuses
    with pytest.raises(sm.Refused, match="extension chunks 0"):
        sm.admit_cell(
            _extensions(_a_cell(), (False, True), 0), policy()
        )  # within the limit: the tally refuses


def test_t15_a_cell_with_exactly_the_policys_captures_is_admitted():
    cell = _a_cell()
    n_sel = int(policy()["setup"]["captures_per_configuration"])
    surplus = cell["captures_total"] - n_sel
    assert surplus > 0
    cell.update(captures_total=n_sel, decays_total=cell["decays_total"] + surplus)
    sm.admit_cell(cell, policy())


@pytest.mark.parametrize(
    "d, pull",
    [("1.005", "1.00"), ("-1.005", "-1.00"), ("1.015", "1.02"), ("-1.015", "-1.02"), ("1.025", "1.02")],
)
def test_t15_a_pull_exactly_at_a_midpoint_rounds_half_even(d, pull):
    assert sm._pull(Fraction(d), Fraction(1)) == pull


def test_t15_a_line_yield_exactly_at_its_band_is_called_consistent_with_zero_and_one_step_beyond_is_not():
    phrase = "the net line yield consistent with zero"
    base = rendered().text.count(phrase)
    k = Decimal(policy()["rule"]["k_pass"])
    for step, change in ((Decimal(0), 0), (Decimal("1e-15"), -1)):
        inp = fresh_inputs()
        rows = [r for r in inp.rows if r["predicate"] == "gamma.Al.1808" and r["config"] == "bertini"]
        assert rows and all(abs(Fraction(r["x"])) <= Fraction(k) * Fraction(r["x_se"]) for r in rows)
        for r in rows:
            r["x"] = str(k * Decimal(r["x_se"]) + step)
            assert (Fraction(r["x"]) == Fraction(k) * Fraction(r["x_se"])) is (step == 0)
        rejudge(inp)
        assert gen.render(inp).text.count(phrase) == base + change


def test_t15_a_changed_cmakelists_is_caught_by_its_own_pin():
    builds = read_csv(DATA / "builds.csv")
    source = (REPO / "cpp" / "stopped_muon" / "g4muonic_transport.cc").read_bytes()
    cmakelists = (REPO / "cpp" / "stopped_muon" / "CMakeLists.txt").read_bytes()
    assert source_mismatches(builds, source, cmakelists) == []
    changed = cmakelists[:10] + bytes([cmakelists[10] ^ 1]) + cmakelists[11:]
    assert source_mismatches(builds, source, changed) == [b["tag"] for b in builds]


#: The seven fields of a compact header, held here: (position, a value differing from the reference's).
HEADER_FIELDS = [
    ("#", 0, "%"),
    ("version", 1, "B"),
    ("config", 2, "bertini_pc"),
    ("target", 3, "Al-27"),
    ("offset", 4, "1"),
    ("events", 5, "4"),
    ("route", 6, "muonic_atom_helper"),
]


@pytest.mark.parametrize("field, at, value", HEADER_FIELDS, ids=[f[0] for f in HEADER_FIELDS])
def test_t15_each_field_of_a_chunk_header_is_bound(field, at, value):
    body = [cap(i, CAPTURES[i]) for i in range(5)]
    good = chunk("chunk0", 0, body)
    assert reduce([good])[1] == "accepted"
    head = good.lines[0].split()
    assert head[at] != value
    head[at] = value
    bad = sm.Chunk(chunk="chunk0", offset=0, events=5, extension=False, lines=[" ".join(head), *body])
    with pytest.raises(sm.Refused, match="header"):
        reduce([bad])


def test_t15_a_chunk_with_fewer_events_than_its_header_is_refused():
    body = [cap(i, CAPTURES[i]) for i in range(5)]
    short = sm.Chunk(
        chunk="chunk0",
        offset=0,
        events=6,
        extension=False,
        lines=["# A bertini natSi 0 6 bound_decay", *body],
    )
    with pytest.raises(sm.Refused, match="5 events, 6 in meta"):
        reduce([short])


#: The identity and reference fields of a comparison row that the rule binds to the pre-registration, held
#: here, each with a value differing from the two-row policy's.
ROSTER_FIELDS = [
    ("predicate", "other"),
    ("family", "y"),
    ("target", "U"),
    ("version", "W"),
    ("config", "d"),
    ("ref", "1.1"),
    ("ref_stat", "0.031"),
    ("ref_syst", "0.041"),
]


@pytest.mark.parametrize("field, value", ROSTER_FIELDS, ids=[f[0] for f in ROSTER_FIELDS])
def test_t15_each_roster_field_of_a_row_is_bound_to_the_pre_registration(field, value):
    pol = two_row_policy()
    rows = comparison_of(pol, {})
    sm.judge(pol, rows)
    assert rows[0]["row_id"] == "plain" and rows[0][field] != value
    rows[0][field] = value
    with pytest.raises(sm.Refused, match=f"plain: {field} "):
        sm.judge(pol, rows)


@pytest.mark.parametrize("fault", ["extra column", "columns reordered", "upper edge on a plain row"])
def test_t15_a_comparison_row_of_another_shape_is_refused(fault):
    pol = two_row_policy()
    rows = comparison_of(pol, {})
    if fault == "extra column":
        rows[0]["note"] = ""
    elif fault == "columns reordered":
        rows[0] = dict(reversed(list(rows[0].items())))
    else:
        rows[0]["rescue_hi"] = "1.5"
    with pytest.raises(sm.Refused, match="rescue edges on a row" if fault.startswith("upper") else "columns"):
        sm.judge(pol, rows)


def test_t15_a_cell_outside_the_policys_matrix_is_refused(tmp_path):
    lines = (DATA / "cells.jsonl").read_text(encoding="utf-8").splitlines()
    extra = json.loads(lines[0])
    extra["target"] = "natZz"
    sm.admit_cell(extra, policy())
    (tmp_path / "extra.jsonl").write_text("\n".join([*lines, json.dumps(extra)]) + "\n", encoding="utf-8")
    with pytest.raises(sm.Refused, match=r"extra \[\('[^']+', '[^']+', 'natZz'\)\]"):
        sm.load_cells(tmp_path / "extra.jsonl", policy())


#: The phrases the document never uses, held here independently of the generator.
FORBIDDEN_PHRASES = (
    "first",
    "nobody noticed",
    "no published comparison",
    "still unfixed",
    "caused",
    "introduced",
    "regression",
    "absent",
    "every physics list",
    "all physics lists",
)


@pytest.mark.parametrize("phrase", FORBIDDEN_PHRASES)
def test_t15_each_excluded_phrase_is_found_on_its_own(phrase):
    assert gen.forbidden_hits(f"A sentence with {phrase.upper()} in it.") == [phrase]
    assert gen.forbidden_hits(f"A sentence with x{phrase}x in it.") == []


@pytest.mark.parametrize("field", ["x", "x_se", "ref", "ref_stat", "ref_syst"])
def test_t15_a_revision_pair_that_differs_in_one_field_alone_is_listed(field):
    inp = fresh_inputs()
    v1, v2 = inp.policy["setup"]["versions"]
    rid1, rid2 = f"alcap.Al.p|{v1}|bertini", f"alcap.Al.p|{v2}|bertini"
    assert gen.revision_pairs(inp)[1] == []
    before = {v.row_id: v.verdict for v in inp.verdicts}[rid2]
    [row] = [r for r in inp.rows if r["row_id"] == rid2]
    assert not row["rescue_lo"]
    new = str(Decimal(row[field]) + Decimal("1e-9"))
    row[field] = new
    if field.startswith("ref"):
        [fr] = [r for r in inp.policy["rows"] if r["row_id"] == rid2]
        fr[field] = new  # the reference cell of this row's own pre-registration row, so the rule judges it
    rejudge(inp)
    assert {v.row_id: v.verdict for v in inp.verdicts}[rid2] == before  # the verdict does not change
    assert gen.revision_pairs(inp)[1] == [rid1]


def al27_width_chunk() -> sm.Chunk:
    """Four Al-27 captures, each with a neutron and two photons 2 keV from E_t in the rest frame of a residual
    at rest: 8 counts between the tag and the inner edge against an expected continuum of 0 and an offset of
    5."""
    gp = sm.gamma_params(synthetic_policy())
    e = sm.e_t(gp.lines["gamma.Al.1808"], gp) + 0.002
    res = f"1000120260:{h(0.0)}:{h(0.0)}:{h(0.0)}:{h(0.0)} 2112:{h(1.0)}:{P0.replace(' ', ':')}"
    body = [
        f"{i} C {h(0.0)} {res} " + " ".join([f"22:{h(e)}:{h(e)}:{h(0.0)}:{h(0.0)}"] * 2) for i in range(4)
    ]
    return chunk("chunk0", 0, body, target="Al-27")


def test_t15_a_refused_tag_width_gate_refuses_the_reduced_cell():
    cell, status = reduce([al27_width_chunk()], policy=synthetic_policy(captures=4), target="Al-27")
    assert cell["line_guard"]["gamma.Al.1808"]["ok"] is True
    assert cell["tag_width"]["gamma.Al.1808"] == {
        "ok": False,
        "n_1keV": 0,
        "n_1to3keV": 8,
        "continuum_expected": 0.0,
    }
    assert status == "refused"
    assert cell["problems"] == ["tag-width gate refused gamma.Al.1808: n1=0 n3=8 expected 0.0"]


# ---- the driver, its harness replaced by a stand-in that writes what the harness writes
class StandIn:
    """A stand-in for the harness: the DATASET lines of the binding, and on a run a records file of
    ``records`` (one file per name in ``files``), config.txt and the return code ``rc``."""

    def __init__(
        self, records: list[str], datasets: str, rc: int = 0, files: tuple[str, ...] = ("records-0.txt",)
    ):
        self.records, self.datasets, self.rc, self.files = records, datasets, rc, files

    def run(self, argv, **kwargs):
        if "--print-datasets" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=self.datasets, stderr="")
        out = Path(argv[argv.index("--out") + 1])
        for name in self.files:
            (out / name).write_text("".join(self.records), encoding="ascii", newline="\n")
        (out / "config.txt").write_text(self.datasets, encoding="utf-8")
        return subprocess.CompletedProcess(argv, self.rc)


def stand_in(monkeypatch, records, **kwargs) -> StandIn:
    pol = synthetic_policy(captures=4)
    fake = StandIn(records, dataset_text(drv.binding(pol, "A", "/data")), **kwargs)
    monkeypatch.setattr(drv, "subprocess", SimpleNamespace(run=fake.run, STDOUT=subprocess.STDOUT))
    return fake


def chunk_kwargs(tmp_path: Path) -> dict:
    binary = tmp_path / "g4muonic_transport"
    binary.write_bytes(b"a stand-in binary")
    return dict(
        binary=str(binary),
        install="/inst",
        data="/data",
        work=tmp_path / "work",
        version="A",
        config="bertini",
        target="Al-27",
        offset=0,
        events=4,
        chunk=0,
    )


def test_t15_the_driver_runs_a_chunk_through_the_stand_in(tmp_path, monkeypatch):
    stand_in(monkeypatch, al27_capture_stream())
    meta = drv.run_chunk(synthetic_policy(captures=4), **chunk_kwargs(tmp_path))
    assert (meta["stats"]["C"], meta["stats"]["D"], meta["stats"]["X"]) == (4, 0, 0)
    with pytest.raises(drv.Refusal, match="chunk directory exists"):
        drv.run_chunk(synthetic_policy(captures=4), **chunk_kwargs(tmp_path))


@pytest.mark.parametrize(
    "fault, kwargs, match",
    [
        ("harness rc", {"rc": 1}, "harness rc 1"),
        ("no record file", {"files": ()}, "expected one worker record file"),
        (
            "two record files",
            {"files": ("records-0.txt", "records-1.txt")},
            "expected one worker record file",
        ),
    ],
    ids=["harness rc", "no record file", "two record files"],
)
def test_t15_the_driver_refuses_a_failed_run_or_an_unexpected_record_file(
    tmp_path, monkeypatch, fault, kwargs, match
):
    stand_in(monkeypatch, al27_capture_stream(), **kwargs)
    with pytest.raises(drv.Refusal, match=match):
        drv.run_chunk(synthetic_policy(captures=4), **chunk_kwargs(tmp_path))


def committed_record(tmp_path: Path, monkeypatch, records: list[str]) -> tuple[Path, Path, str]:
    """(pre-registration file, cells file, status): the cell the stand-in's records reduce to, written as the
    committed record a reproduction compares with."""
    pol = synthetic_policy(captures=4)
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps(pol), encoding="utf-8")
    stand_in(monkeypatch, records)
    kw = chunk_kwargs(tmp_path)
    kw["work"] = tmp_path / "first"
    meta = drv.run_chunk(pol, **kw)
    cell, status = drv.reduce_work(pol, kw["work"], "A", "bertini", "Al-27", [meta])
    cell["chunks"][0]["compact_text_sha256"] = meta["compact_text_sha256"]
    cells_file = tmp_path / "cells.jsonl"
    cells_file.write_text(json.dumps(cell) + "\n", encoding="utf-8")
    return prereg, cells_file, status


def reproduce_args(tmp_path: Path, prereg: Path, cells_file: Path, cell: str = "A/bertini/Al-27"):
    return argparse.Namespace(
        prereg=str(prereg),
        cell=cell,
        cells=str(cells_file),
        binary=str(tmp_path / "g4muonic_transport"),
        install="/inst",
        data="/data",
        work=str(tmp_path / "again"),
        out=str(tmp_path / "reproduced.csv"),
    )


def test_t15_a_reproduction_equal_field_by_field_but_refused_reports_its_status(
    tmp_path, monkeypatch, capsys
):
    prereg, cells_file, status = committed_record(tmp_path, monkeypatch, al27_capture_stream())
    assert status == "refused"  # the line guard refuses the descendant of event 0
    assert drv.reproduce(reproduce_args(tmp_path, prereg, cells_file)) == 1
    out = capsys.readouterr().out
    assert [ln for ln in out.splitlines() if ln.startswith("DIFFERS")] == [
        "DIFFERS A/bertini/Al-27 status refused"
    ]


def test_t15_a_reproduction_of_an_accepted_cell_is_reproduced(tmp_path, monkeypatch, capsys):
    clean = edited(without(al27_capture_stream(), "T 0 9 3 22 "), "E 0 1000 ", 3, "7")
    prereg, cells_file, status = committed_record(tmp_path, monkeypatch, clean)
    assert status == "accepted"
    assert drv.reproduce(reproduce_args(tmp_path, prereg, cells_file)) == 0
    assert "REPRODUCED A/bertini/Al-27: 1 chunks" in capsys.readouterr().out


def test_t15_the_driver_refuses_a_cell_it_cannot_name_or_find(tmp_path, monkeypatch):
    prereg, cells_file, _ = committed_record(tmp_path, monkeypatch, al27_capture_stream())
    with pytest.raises(drv.Refusal, match="--cell is"):
        drv.reproduce(reproduce_args(tmp_path, prereg, cells_file, cell="A/bertini"))
    with pytest.raises(drv.Refusal, match="no committed cell A/bertini/natSi"):
        drv.reproduce(reproduce_args(tmp_path, prereg, cells_file, cell="A/bertini/natSi"))


def test_t15_the_refusals_that_need_no_records(tmp_path):
    with pytest.raises(sm.Refused, match="route must be one of"):
        sm.classify_stream(
            iter([]),
            io.StringIO(),
            route="bound",
            version="A",
            config="bertini",
            target="Al-27",
            offset=0,
            events=0,
            policy=synthetic_policy(),
        )
    with pytest.raises(ValueError, match="not a capture line"):
        sm.parse_line("0 D")
    with pytest.raises(drv.Refusal, match="capture fraction 0"):
        drv.extension_events(synthetic_policy(), 0, 100)
    with pytest.raises(drv.Refusal, match="no Geant4 CMake package"):
        drv.build(argparse.Namespace(install=str(tmp_path), build=str(tmp_path / "b"), jobs=1))
