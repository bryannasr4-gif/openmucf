"""The stopped-muon benchmark: its classifier, reducer, gamma-line tag and rule (`openmucf/g4/stopped_mu.py`),
its driver (`cpp/stopped_muon/stopped_muon.py`) and the committed files of `data/g4/stopped_mu/`.

Every test here calls the function it checks with the input it names, so no earlier guard pre-empts it. The
synthetic policy is written in this file; each value it carries is one the function under test must read from
it.
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
import subprocess
import sys
from collections import Counter
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

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
        f"T 0 3 1 1000120260 muMinusCaptureAtRest {NC} {h(0.25)} {h(700.0)}\n",
        f"T 0 9 3 22 - - {h(1.80874)} {h(1100.0)}\n",
        f"T 0 10 3 22 - - {h(1.12959)} {h(2200.0)}\n",
        f"T 0 11 1 22 - - {h(0.4)} {h(710.0)}\n",
        "E 0 1000 6\n",
    ]
    want.append(
        f"0 C {h(700.0)} 2112:{h(2.5)}:{h(1.0)}:{h(2.0)}:{h(3.0)} "
        f"1000120260:{h(0.25)}:{h(-4.0)}:{h(0.5)}:{h(0.0)} 22:{h(1.5)}:{h(0.0)}:{h(1.5)}:{h(0.0)} "
        f"g22:{h(1.80874)}:{h(400.0)}:1000120260"
    )
    # event 1: a decay in orbit
    lines += [
        "S 1 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 2\n",
        c_line(1, 1, 0, 11, DIO, 40.0, 900.0),
        c_line(1, 1, 1, 14, DIO, 30.0, 900.0),
        "E 1 1001 3\n",
    ]
    want.append("1 D")
    # event 2: both a capture product and a decay product -> X
    lines += [
        "S 2 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 2\n",
        c_line(2, 1, 0, 2112, NC, 1.0, 900.0),
        c_line(2, 1, 1, 11, DIO, 30.0, 900.0),
        "E 2 1002 3\n",
    ]
    want.append("2 X capture_and_dio")
    # event 3: only the electromagnetic cascade -> no outcome
    lines += [
        "S 3 1 1 muMinusCaptureAtRest 0x0p+0 0x0p+0 1\n",
        c_line(3, 1, 0, 22, EM, 0.3, 0.0),
        "E 3 1003 2\n",
    ]
    want.append("3 X no_outcome")
    # event 4: the muon never reaches the capture process
    lines += ["S 4 1 1 Decay 0x0p+0 0x0p+0 0\n", "E 4 1004 1\n"]
    want.append("4 X capture_steps=0")
    return lines, want


def helper_events() -> tuple[list[str], list[str]]:
    lines: list[str] = []
    want: list[str] = []
    atom = "MuAtom"
    # event 0: the muonic atom (track 2) captures: products of its LAST step
    lines += [
        "S 0 1 1 muMinusAtomicCaptureAtRest 0x0p+0 0x0p+0 1\n",
        c_line(0, 1, 0, 1000130279, atom, 0.0, 0.0),
        "S 0 2 1 Transportation 0x0p+0 0x0p+0 1\n",
        c_line(0, 2, 0, 22, "x", 0.1, 5.0),
        "S 0 2 2 muonicAtomDecay 0x0p+0 0x0p+0 2\n",
        c_line(0, 2, 0, 2112, "model_G4MuMinusCapturePrecompound", 3.0, 600.0),
        c_line(0, 2, 1, 14, "model_G4MuMinusCapturePrecompound", 90.0, 500.0),
        "E 0 1000 4\n",
    ]
    want.append(f"0 C {h(500.0)} 2112:{h(3.0)}:{P0.replace(' ', ':')} 14:{h(90.0)}:{P0.replace(' ', ':')}")
    # event 1: a decay, told by its electron antineutrino (PDG -12)
    lines += [
        "S 1 2 1 muonicAtomDecay 0x0p+0 0x0p+0 2\n",
        c_line(1, 2, 0, 11, "model_DIO", 40.0, 900.0),
        c_line(1, 2, 1, -12, "model_DIO", 30.0, 900.0),
        "E 1 1001 3\n",
    ]
    want.append("1 D")
    # event 2: no products
    lines += ["S 2 2 1 muonicAtomDecay 0x0p+0 0x0p+0 0\n", "E 2 1002 1\n"]
    want.append("2 X no_products")
    # event 3: two muonic-atom tracks
    lines += [
        "S 3 2 1 muonicAtomDecay 0x0p+0 0x0p+0 0\n",
        "S 3 3 1 muonicAtomDecay 0x0p+0 0x0p+0 0\n",
        "E 3 1003 2\n",
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
