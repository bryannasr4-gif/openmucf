"""Binding a measured vector to a material, and the selection Geant4 then makes.

`openmucf/g4/d2_material.py` states the binding rules and the selection arithmetic that the C++ provider in
`cpp/material_selector/` applies. Every refusal is required by type (`BindRefused`), never as an arbitrary
exception, and every positive control binds. The ladder counts are checked against a brute-force count over
every draw of the ladder.
"""

from __future__ import annotations

import math
import pathlib
from fractions import Fraction

import numpy as np
import pytest

from openmucf.g4 import d2 as selector
from openmucf.g4 import d2_material as dmat
from openmucf.g4 import d2_measured as dm

REPO = pathlib.Path(__file__).resolve().parents[1]
DIRECTORY = REPO / "data" / "g4" / "d2" / "measured"
AL = "stanislaus1987-al2o3"
N = selector.DRAW_COUNT


def entry(profile: str) -> dmat.VectorEntry:
    return dmat.load_entry((DIRECTORY / f"{profile}.g4dat").read_text(encoding="ascii"))


def request(
    elements=((13, 2.0e22), (8, 3.0e22)),
    formula="Al2O3",
    counts=((13, 2), (8, 3)),
    ack=True,
    profile=AL,
    name="Al2O3",
    metadata=None,
) -> dmat.MaterialRequest:
    return dmat.MaterialRequest(name, elements, formula, counts, ack, profile, metadata or {})


def brute_counts(weights) -> tuple[int, ...]:
    """Every draw of the ladder through the cumulative search, vectorized."""
    cum = np.cumsum(np.array(weights, dtype=np.float64))
    # np.cumsum adds left to right in binary64, as the selector does; check it against the scalar sums
    assert list(cum) == dmat.cumulative(weights)
    u = (np.arange(N, dtype=np.float64) + 0.5) / N
    picks = np.minimum(np.searchsorted(cum, cum[-1] * u, side="left"), len(cum) - 1)
    return tuple(int(c) for c in np.bincount(picks, minlength=len(cum)))


def test_every_committed_table_loads_as_a_vector_entry():
    tables = sorted(DIRECTORY.glob("*.g4dat"))
    assert tables
    for path in tables:
        e = dmat.load_entry(path.read_text(encoding="ascii"))
        assert e.profile == path.stem
        assert dmat.formula_counts(e.formula) == dict(zip(e.z, e.n, strict=True))
        assert abs(math.fsum(e.p) - 1.0) <= dmat.SUM_TOL


def test_the_aluminium_oxide_witness_at_the_ladder():
    """Material order Al, O: the measured entry gives p = 47/122 and the stock selector 325/493, whose total
    variation is 16479/60146; at u = 1/2 they select different elements, and the 2^20 midpoint ladder counts
    of the two for aluminium are those a brute-force count over every draw gives."""
    e = entry(AL)
    b = dmat.bind(e, request())
    assert b.probabilities == (47 / 122, 1 - 47 / 122)
    stock = dmat.stock_weights(((13, 2.0e22), (8, 3.0e22)))
    assert Fraction(stock[0]) / Fraction(sum(stock)) == pytest.approx(Fraction(325, 493), rel=1e-15)
    assert abs(Fraction(325, 493) - Fraction(47, 122)) == Fraction(16479, 60146)
    assert dmat.select(b.probabilities, 0.5) == 1 and dmat.select(stock, 0.5) == 0
    measured_counts, stock_counts = dmat.ladder_counts(b.probabilities), dmat.ladder_counts(stock)
    assert measured_counts == brute_counts(b.probabilities)
    assert stock_counts == brute_counts(stock)
    assert measured_counts[0] != stock_counts[0]


def test_the_cumulative_search_at_each_boundary():
    """For every table in either element order: the largest draw whose scaled value does not pass the first
    running sum selects the first element, and the next representable draw selects the second; the counts of
    the ladder are those of a brute-force count."""
    for e in entries_all():
        for order in (e.p, tuple(reversed(e.p))):
            cum = dmat.cumulative(order)
            lo, hi = 0.0, 1.0
            for _ in range(200):  # bisection on the float line
                mid = 0.5 * (lo + hi)
                if mid in (lo, hi):
                    break
                lo, hi = (mid, hi) if cum[-1] * mid <= cum[0] else (lo, mid)
            u = lo if cum[-1] * hi > cum[0] else hi
            assert cum[-1] * u <= cum[0] < cum[-1] * math.nextafter(u, 1.0)
            assert dmat.select(order, u) == 0 and dmat.select(order, math.nextafter(u, 1.0)) == 1
            assert dmat.ladder_counts(order) == brute_counts(order)


def entries_all() -> list[dmat.VectorEntry]:
    return [dmat.load_entry(p.read_text(encoding="ascii")) for p in sorted(DIRECTORY.glob("*.g4dat"))]


def test_reversed_element_order_binds_by_atomic_number():
    e = entry(AL)
    forward = dmat.bind(e, request())
    reverse = dmat.bind(e, request(elements=((8, 3.0e22), (13, 2.0e22))))
    assert reverse.probabilities == tuple(reversed(forward.probabilities))
    assert dmat.ladder_counts(reverse.probabilities) == brute_counts(reverse.probabilities)
    # a binding by position would give oxygen the aluminium probability
    assert reverse.probabilities[0] == forward.probabilities[1] != forward.probabilities[0]


@pytest.mark.parametrize("mutant", ["omitted_mn", "swapped", "double_stoichiometry", "inverted_a"])
def test_consumer_defects_change_the_witness(mutant):
    """Each defect of the transform from A to the vector -- the stoichiometry omitted, the elements swapped,
    the counts applied a second time, A inverted -- moves the aluminium probability and its ladder count."""
    a, m, n = Fraction("0.94"), 2, 3
    p = m * a / (n + m * a)
    q = {
        "omitted_mn": a / (1 + a),
        "swapped": 1 - p,
        "double_stoichiometry": (p * m) / (p * m + (1 - p) * n),
        "inverted_a": m / a / (n + m / a),
    }[mutant]
    assert abs(q - p) > Fraction(1, 100)
    good = dmat.ladder_counts((float(p), float(1 - p)))
    bad = dmat.ladder_counts((float(q), float(1 - q)))
    assert good != bad


# --------------------------------------------------------------------------------------------
# the named NIST oxides
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "formula"), sorted(dmat.NIST_POLICY.items()))
def test_each_named_nist_oxide_binds_its_own_entry_within_the_policy(name, formula):
    """A weight-fraction definition rounds the atom ratio; 8.07e-5 (the size seen for lead oxide) binds under
    the NIST name and its own formula, in either element order, and is refused under any other name."""
    e = entry(dm.PROFILE_PREFIX + formula.lower())
    counts = dict(zip(e.z, e.n, strict=True))
    z_metal = max(e.z)
    density = {8: counts[8] * 1.0e22 * (1 + 8.07e-5), z_metal: counts[z_metal] * 1.0e22}
    pairs = tuple(sorted(counts.items()))
    for order in ((8, z_metal), (z_metal, 8)):
        elements = tuple((z, density[z]) for z in order)
        b = dmat.bind(e, request(elements, formula, pairs, profile=e.profile, name=name))
        assert b.tolerance == dmat.NIST_TOL and b.allowance == dmat.representation_allowance(dmat.NIST_TOL)
        with pytest.raises(dmat.BindRefused):
            dmat.bind(e, request(elements, formula, pairs, profile=e.profile, name=name + "_COPY"))
    far = {8: counts[8] * 1.0e22 * (1 + 2e-4), z_metal: counts[z_metal] * 1.0e22}
    with pytest.raises(dmat.BindRefused):
        dmat.bind(e, request(tuple(far.items()), formula, pairs, profile=e.profile, name=name))


def test_a_nist_name_is_admitted_only_for_its_own_formula():
    e = entry(dm.PROFILE_PREFIX + "pbo2")
    near = ((8, 2.0e22 * (1 + 8.07e-5)), (82, 1.0e22))
    with pytest.raises(dmat.BindRefused):
        dmat.bind(e, request(near, "PbO2", ((8, 2), (82, 1)), profile=e.profile, name="G4_LEAD_OXIDE"))


def test_the_representation_allowances():
    assert dmat.representation_allowance(1e-9) == pytest.approx(2.5000000025e-10, rel=1e-12)
    assert dmat.representation_allowance(1e-4) == pytest.approx(2.5002500250e-5, rel=1e-10)
    for bad in (0.0, 1.0, -1e-9, math.nan):
        with pytest.raises(ValueError):
            dmat.representation_allowance(bad)


# --------------------------------------------------------------------------------------------
# refusals, by type
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"elements": ((13, 2.0e22),)},  # short: oxygen missing
        {"elements": ((13, 2.0e22), (8, 3.0e22), (1, 1.0e22))},  # long: hydrogen added
        {"elements": ((13, 2.0e22), (13, 2.0e22))},  # a repeated element
        {"elements": ((13, 2.0e22), (14, 3.0e22))},  # the wrong element
        {"elements": ((13, 2.0e22), (8, 3.1e22))},  # the wrong atom ratio
        {"elements": ((13, 2.0e22), (8, 3.0e22 * (1 + 2e-9)))},  # just outside 1e-9
        {"elements": ((13, math.nan), (8, 3.0e22))},
        {"elements": ((13, math.inf), (8, 3.0e22))},
        {"elements": ((13, -math.inf), (8, 3.0e22))},
        {"elements": ((13, 0.0), (8, 3.0e22))},
        {"elements": ((13, -2.0e22), (8, 3.0e22))},
        {"elements": ((13, True), (8, 3.0e22))},
        {"elements": ((13, "2e22"), (8, 3.0e22))},
        {"elements": ((13.0, 2.0e22), (8, 3.0e22))},
        {"elements": ()},
        {"elements": [(13, 2.0e22), (8, 3.0e22)]},
        {"formula": "AlO"},
        {"formula": "Al2O3H"},
        {"formula": ""},
        {"formula": "Al02O3"},
        {"counts": ((13, 2), (8, 2))},
        {"counts": ((13, 2),)},
        {"counts": ((13, 2), (8, 3), (8, 3))},
        {"counts": ((13, 2.0), (8, 3))},
        {"counts": ((13, True), (8, 3))},
        {"ack": False},
        {"ack": 1},
        {"ack": "yes"},
        {"profile": "stanislaus1987-sio2"},
        {"metadata": {"hydration": "hydrated"}},
        {"metadata": {"source_fingerprint": "x"}},
    ],
)
def test_every_refusal_is_a_bind_refused(kwargs):
    with pytest.raises(dmat.BindRefused) as exc:
        dmat.bind(entry(AL), request(**kwargs))
    assert type(exc.value) is dmat.BindRefused and exc.value.code == "ENTRY_REFUSED"


def test_optional_metadata_does_not_block_a_valid_binding():
    meta = {
        "polymorph": "corundum",
        "hydration": "anhydrous",
        "bulk_density_g_cm3": "3.97",
        "temperature_K": "293",
    }
    assert dmat.bind(entry(AL), request(metadata=meta)).probabilities == (47 / 122, 1 - 47 / 122)


def test_just_inside_the_atom_tolerance_binds():
    b = dmat.bind(entry(AL), request(elements=((13, 2.0e22), (8, 3.0e22 * (1 + 5e-10)))))
    assert b.tolerance == dmat.ATOM_TOL


@pytest.mark.parametrize(
    "edit",
    [
        lambda t: t.replace("#SEAM         d2_atomic_capture", "#SEAM         d1_nuclear_capture"),
        lambda t: t.replace("form:vector", "form:law"),
        lambda t: t.replace("material:Al2O3", "material:AlO"),
        lambda t: t.replace(" status:SOURCE_REPORTED_PROVISIONAL", ""),
        lambda t: t.replace("13 0.38524590163934425 2", "13 0.48524590163934425 2"),
        lambda t: t.replace("13 0.38524590163934425 2", "13 0.38524590163934425 2.5"),
    ],
)
def test_a_malformed_table_is_refused(edit):
    text = (DIRECTORY / f"{AL}.g4dat").read_text(encoding="ascii")
    bad = edit(text)
    assert bad != text
    with pytest.raises(dmat.BindRefused):
        dmat.load_entry(bad)
