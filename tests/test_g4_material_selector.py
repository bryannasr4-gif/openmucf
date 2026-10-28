"""What the material selector delivered in real Geant4 builds, checked against the reference arithmetic.

`cpp/material_selector/evidence/` holds the outputs of `cpp/material_selector/native.sh` run against Geant4
11.4.2 and 11.5.0.beta: every measured table read back and bound by the C++ provider, the named NIST oxides,
the binding refusals, the installation order, the capture ladders on both routes and both run managers, the
default-off traces and the teardown. Each check here is a function that returns the problems it finds; it must
return none for the builds that deliver physics, and for each mutation-drill build it must return the problem
the drill was built to cause.
"""

from __future__ import annotations

import csv
import hashlib
import pathlib
from collections import defaultdict

import pytest

from openmucf.g4 import d2 as selector
from openmucf.g4 import d2_material as dmat

REPO = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "cpp" / "material_selector" / "evidence"
MEASURED = REPO / "data" / "g4" / "d2" / "measured"
SOURCES = (
    "G4MuonicMaterialSelector.hh",
    "G4MuonicMaterialSelector.cc",
    "harvest_material.cc",
    "CMakeLists.txt",
)
BASE = {"v1142": "v11.4.2", "v1150": "v11.5.0.beta"}
N = selector.DRAW_COUNT
AL, PB = "stanislaus1987-al2o3", "stanislaus1987-pbo"


def lf_sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def builds() -> dict[str, dict[str, str]]:
    with (EVIDENCE / "builds.csv").open(newline="", encoding="utf-8") as f:
        return {r["tag"]: r for r in csv.DictReader(f)}


def blocks(kind: str, tag: str) -> dict[str, list[str]]:
    """The `=== RUN <name>` blocks of one evidence file: one run's output each."""
    out: dict[str, list[str]] = {}
    current = None
    for line in (EVIDENCE / f"{kind}_{tag}.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("=== RUN "):
            current = line[8:]
            out[current] = []
        elif current is not None:
            out[current].append(line)
    return out


def entries() -> dict[str, dmat.VectorEntry]:
    return {p.stem: dmat.load_entry(p.read_text(encoding="ascii")) for p in sorted(MEASURED.glob("*.g4dat"))}


def hexf(text: str) -> float:
    return float.fromhex(text)


# --------------------------------------------------------------------------------------------
# builds
# --------------------------------------------------------------------------------------------


def test_each_revision_was_built_from_the_committed_sources_with_its_recorded_header():
    """Both pinned revisions, multithreaded; the atomic setter did not link, so the shim was compiled, and
    only because the installed header is one `supported_headers.txt` lists for that revision; the provider and
    harness bytes are the committed ones."""
    b = builds()
    supported = {}
    for line in (
        (REPO / "cpp/material_selector/supported_headers.txt").read_text(encoding="utf-8").splitlines()
    ):
        revision, commit, name, sha = line.split()
        supported[revision] = (commit, sha)
    for tag, row in b.items():
        assert row["multithreading"] == "yes", tag
        assert (row["source_commit"], row["capture_header"]) == supported[row["revision"]], tag
        assert (row["setter_linked"], row["shim"]) == ("FALSE", "ON"), tag
        recorded = dict(s.split(":") for s in row["sources"].split(";"))
        for name in SOURCES:
            assert recorded[name] == lf_sha(REPO / "cpp/material_selector" / name), (tag, name)
    assert {b[t]["revision"] for t in BASE} == set(BASE.values())
    assert all(b[t]["drills"] == "none" and b[t]["asan"] == "OFF" for t in BASE)


# --------------------------------------------------------------------------------------------
# every table, read back by the provider
# --------------------------------------------------------------------------------------------


def probe_problems(tag: str) -> list[str]:
    lines = blocks("probe", tag)["probe"]
    problems = []
    seen = set()
    ents = entries()
    weights: dict[tuple[str, str], list[tuple[int, float, float, float]]] = defaultdict(list)
    for line in lines:
        f = line.split()
        if f[0] == "ENTRY":
            seen.add((f[1], f[2]))
            if f[3] != "BOUND" or hexf(f[5]) != dmat.ATOM_TOL:
                problems.append(f"{f[1]} {f[2]} not bound at the atom tolerance")
        elif f[0] == "P":
            weights[(f[1], f[2])].append((int(f[4]), hexf(f[5]), hexf(f[6]), hexf(f[7])))
    for (profile, order), rows in weights.items():
        e = ents[profile]
        p_by_z = dict(zip(e.z, e.p, strict=True))
        w = [r[2] for r in rows]
        if w != [p_by_z[r[0]] for r in rows]:
            problems.append(f"{profile} {order}: weights are not the entry's by Z")
        if [r[3] for r in rows] != dmat.cumulative(w):
            problems.append(f"{profile} {order}: running sums differ")
        line = next(x for x in lines if x.startswith(f"ENTRY {profile} {order} "))
        counts = tuple(int(c) for c in line.split("counts ")[1].split())
        if counts != dmat.ladder_counts(w):
            problems.append(f"{profile} {order}: ladder counts differ")
        zs = [r[0] for r in rows]
        if zs != (sorted(zs, reverse=(order == "desc"))):
            problems.append(f"{profile} {order}: element order")
    if seen != {(p, o) for p in ents for o in ("asc", "desc")}:
        problems.append("not every table was read back in both orders")
    return problems


@pytest.mark.parametrize("tag", sorted(BASE))
def test_every_table_reads_back_and_binds_in_both_orders(tag):
    assert probe_problems(tag) == []


@pytest.mark.parametrize("tag", sorted(BASE))
def test_the_named_nist_oxides_bind_within_the_policy(tag):
    """Each named NIST oxide, defined by weight fraction, binds the entry of its formula at 1e-4; its atom
    ratio, recomputed from the harvested densities, is off the formula's by less than that, and for lead oxide
    by more than the 1e-9 an atom-count material must meet."""
    ents = entries()
    worst = {}
    for line in blocks("probe", tag)["probe"]:
        f = line.split()
        if f[0] != "NIST":
            continue
        name, profile, result = f[1], f[2], f[3]
        assert result == "BOUND" and hexf(f[6]) == dmat.NIST_TOL, name
        dens = [(int(zd.split(":")[0]), hexf(zd.split(":")[1])) for zd in f[7:]]
        e = ents[profile]
        b = dmat.bind(
            e,
            dmat.MaterialRequest(
                name, tuple(dens), e.formula, tuple(zip(e.z, e.n, strict=True)), True, profile
            ),
        )
        assert b.tolerance == dmat.NIST_TOL
        counts = dict(zip(e.z, e.n, strict=True))
        z0, d0 = dens[0]
        worst[name] = max(abs((d / d0) / (counts[z] / counts[z0]) - 1) for z, d in dens[1:])
    assert set(worst) == set(dmat.NIST_POLICY)
    assert all(v < dmat.NIST_TOL for v in worst.values())
    assert worst["G4_LEAD_OXIDE"] > dmat.ATOM_TOL


def bind_expected() -> dict[str, str]:
    refused = (
        "short_vector",
        "long_vector",
        "mismatched_formula",
        "wrong_atom_ratio",
        "weight_fraction_not_on_policy",
        "nist_on_another_entry",
        "not_acknowledged",
        "unknown_profile",
        "frozen",
    )
    return {
        **dict.fromkeys(
            ("control", "control_reversed_order", "control_nist_lead_oxide"),
            "result=BOUND exception=- bound_count=1",
        ),
        **dict.fromkeys(refused, "result=BIND_REFUSED exception=G4MuonicData101 bound_count=0"),
    }


@pytest.mark.parametrize("tag", sorted(BASE))
def test_the_binding_refuses_exactly_the_hostile_requests(tag):
    got = {}
    for line in blocks("bind", tag)["bind"]:
        if line.startswith("CASE "):
            name, rest = line[5:].split(" ", 1)
            got[name] = rest
    assert got == bind_expected()


@pytest.mark.parametrize("tag", sorted(BASE))
def test_installation_before_the_stopping_physics_fails(tag):
    b = blocks("order", tag)
    assert [x for x in b["order_after"] if x.startswith("INSTALL_ORDER")] == [
        "INSTALL_ORDER after exceptions none"
    ]
    assert [x for x in b["order_before"] if x.startswith("INSTALL_ORDER")] == [
        "INSTALL_ORDER before exceptions G4MuonicData103"
    ]


# --------------------------------------------------------------------------------------------
# the capture ladders
# --------------------------------------------------------------------------------------------


def isotopes_at_half(tag: str) -> dict[int, int]:
    out = {}
    for line in blocks("probe", tag)["probe"]:
        if line.startswith("ISOTOPES "):
            f = line.split()
            ns = tuple(int(x.split(":")[0]) for x in f[2:])
            ab = tuple(hexf(x.split(":")[1]) for x in f[2:])
            out[int(f[1])] = selector.isotope_at_half(selector.Element(int(f[1]), 1.0, ns, ab))
    return out


def parse_ladder(lines: list[str]):
    config = dict(
        line.split(" ", 1)
        for line in lines
        if line.split(" ", 1)[0].isupper()
        and " " in line
        and not line.startswith(("ELEMENT", "WORKER", "SELECTED", "ONWARD", "SELECTORS"))
    )
    elements = [(int(x.split()[2]), hexf(x.split()[3])) for x in lines if x.startswith("ELEMENT ")]
    workers = {}
    selected: dict[tuple[int, int], int] = defaultdict(int)
    onward: dict[tuple[int, int], int] = defaultdict(int)
    for x in lines:
        f = x.split()
        if f[0] == "WORKER":
            workers[int(f[1])] = {k: int(v) for k, v in zip(f[2::2], f[3::2], strict=True)}
        elif f[0] in ("SELECTED", "ONWARD"):
            (selected if f[0] == "SELECTED" else onward)[(int(f[2]), int(f[3]))] += int(f[4])
    return config, elements, workers, dict(selected), dict(onward)


def ladder_problems(
    lines: list[str], weights_kind: str, at_half: dict[int, int], vector: tuple[float, ...] | None = None
) -> list[str]:
    """`weights_kind` is "measured" (the entry's vector in material order, or `vector`) or "stock"."""
    config, elements, workers, selected, onward = parse_ladder(lines)
    problems = []
    if weights_kind == "measured":
        profile = AL if config["MATERIAL"].startswith("al2o3") else PB
        e = entries()[profile]
        p_by_z = dict(zip(e.z, e.p, strict=True))
        w = vector if vector is not None else tuple(p_by_z[z] for z, _ in elements)
    else:
        w = dmat.stock_weights(elements)
    want = {(z, at_half[z]): c for (z, _), c in zip(elements, dmat.ladder_counts(w), strict=True)}
    if selected != want:
        problems.append(f"selected {selected} != {want}")
    if onward != selected:
        problems.append("the nucleus handed onward differs from the selected one")
    if sum(w_["events"] for w_ in workers.values()) != N or sum(w_["calls"] for w_ in workers.values()) != N:
        problems.append("events or selector calls do not add up to the ladder")
    if any(w_["draw_mismatch"] for w_ in workers.values()):
        problems.append("a selector call consumed a number of draws other than the stock rule's")
    if any(w_["onward_mismatch"] for w_ in workers.values()):
        problems.append("an event's onward nucleus or call count was wrong")
    if (
        config["MANAGER"] == "tasking"
        and len([k for k, v in workers.items() if k >= 0 and v["calls"] > 0]) < 2
    ):
        problems.append("fewer than two workers made selector calls")
    return problems


def expected_kind(case: str) -> str:
    return "measured" if case.endswith(("_on", "_on_rev", "_pbo_nist", "_pbo_rev")) else "stock"


@pytest.mark.parametrize("tag", sorted(BASE))
def test_the_aluminium_and_lead_oxide_ladders(tag):
    """On both routes and both run managers: enabled, the measured vector reaches the onward nucleus (for
    aluminium oxide in either element order, for the NIST lead oxide and its reversed order); absent,
    disabled, disabled with bindings in place, and enabled with nothing bound, the stock selector's counts."""
    at_half = isotopes_at_half(tag)
    ladders = blocks("ladder", tag)
    names = {
        f"ladder_{r}_{m}_{e}"
        for r in ("stopping", "atomic")
        for m in ("serial", "tasking")
        for e in ("absent", "off", "on", "off_bound", "on_unbound", "on_rev", "pbo_nist", "pbo_rev")
    }
    assert names <= set(ladders)
    for name in sorted(names):
        assert ladder_problems(ladders[name], expected_kind(name), at_half) == [], name


def test_the_witness_separates_entry_and_stock():
    lines = blocks("ladder", "v1142")
    on = parse_ladder(lines["ladder_stopping_serial_on"])[3]
    stock = parse_ladder(lines["ladder_stopping_serial_absent"])[3]
    assert on[(13, 27)] != stock[(13, 27)]


MUTANT_VECTORS = {
    "inverted_a": lambda a, m, n: m / a / (n + m / a),
    "omitted_mn": lambda a, m, n: a / (1 + a),
    "swapped": lambda a, m, n: 1 - m * a / (n + m * a),
    "double_stoichiometry": lambda a, m, n: (
        (m * a / (n + m * a)) * m / ((m * a / (n + m * a)) * m + (1 - m * a / (n + m * a)) * n)
    ),
}


@pytest.mark.parametrize("mutant", sorted(MUTANT_VECTORS))
def test_a_defective_table_fails_the_witness_it_is_run_against(mutant):
    """A table carrying one consumer defect, run through the same provider: the witness check (counts of the
    committed vector) fails, and the counts are those of the defective vector."""
    from fractions import Fraction

    tag = f"v1142-mutant-{mutant}"
    at_half = isotopes_at_half("v1142")
    lines = blocks("ladder", tag)
    q = float(MUTANT_VECTORS[mutant](Fraction("0.94"), 2, 3))
    for route in ("stopping", "atomic"):
        on = lines[f"ladder_{route}_serial_on"]
        elements = parse_ladder(on)[1]
        assert ladder_problems(on, "measured", at_half) != []
        vector = tuple(q if z == 13 else 1 - q for z, _ in elements)
        assert ladder_problems(on, "measured", at_half, vector) == []
        assert ladder_problems(lines[f"ladder_{route}_serial_off_bound"], "stock", at_half) == []


DRILL_LADDERS = {
    # drill macro: (case, the check that must fail, the problem it must report)
    "enabled_delegates": ("ladder_stopping_serial_on", "measured", "selected"),
    "force_active": ("ladder_stopping_serial_off_bound", "stock", "selected"),
    "extra_enabled_draw": ("ladder_stopping_serial_on", "measured", "number of draws"),
    "master_only": ("ladder_stopping_tasking_on", "measured", "selected"),
    # the entry lists O first; the material Al first, so a binding by position swaps the two
    "positional": ("ladder_stopping_serial_on", "measured", "selected"),
}


@pytest.mark.parametrize("drill", sorted(DRILL_LADDERS))
def test_each_mutation_drill_fails_its_ladder_check(drill):
    case, kind, problem = DRILL_LADDERS[drill]
    tag = f"v1142-drill-{drill}"
    assert builds()[tag]["drills"] == f"G4MUONIC_DRILL_{drill.upper()}"
    found = ladder_problems(blocks("ladder", tag)[case], kind, isotopes_at_half("v1142"))
    assert any(problem in p for p in found), found


# --------------------------------------------------------------------------------------------
# default off, and teardown
# --------------------------------------------------------------------------------------------


def identity(tag: str) -> dict[str, str]:
    with (EVIDENCE / f"identity_{tag}.csv").open(newline="", encoding="utf-8") as f:
        return {r["case"]: r["sha256"] for r in csv.DictReader(f)}


def identity_problems(
    tag: str, modes=("absent", "off", "off_bound", "on_unbound"), prefix="identity"
) -> list[str]:
    d = identity(tag)
    problems = []
    groups = 0
    for route in ("stopping", "atomic"):
        for manager in ("serial", "tasking"):
            for seed in (17, 101, 1009):
                digests = {m: d.get(f"{prefix}_{route}_{manager}_{seed}_{m}") for m in modes}
                groups += 1
                if None in digests.values() or len(set(digests.values())) != 1:
                    problems.append(f"{route} {manager} {seed}: {digests}")
    return problems if groups == 12 else [*problems, "groups"]


@pytest.mark.parametrize("tag", sorted(BASE))
def test_the_default_off_traces_are_identical(tag):
    """Seeds 17, 101 and 1009, 100 events each, both routes and run managers: the event-seeded numerical
    traces with the engine state folded in at every point are identical with the provider absent, disabled,
    disabled with bindings in place, and enabled with nothing bound."""
    assert identity_problems(tag) == []


def cases(tag: str) -> dict[str, dict[str, str]]:
    with (EVIDENCE / f"cases_{tag}.csv").open(newline="", encoding="utf-8") as f:
        return {r["case"]: r for r in csv.DictReader(f)}


@pytest.mark.parametrize("tag", sorted(BASE))
def test_a_disabled_provider_reads_no_file(tag):
    """Disabled, with the catalogue path missing or holding a malformed table, the traces are those of a run
    without the provider; enabled, the same paths stop the run."""
    d, c = identity(tag), cases(tag)
    for kind in ("missing", "malformed"):
        assert d[f"noio_{kind}_off"] == d["identity_stopping_serial_17_absent"], kind
        assert c[f"noio_{kind}_off"]["rc"] == "0" and c[f"noio_{kind}_on"]["rc"] != "0", kind


def test_forced_activation_with_bindings_in_place_changes_the_traces():
    problems = identity_problems("v1142-drill-force_active")
    assert problems and all("off_bound" in p for p in problems)


@pytest.mark.parametrize("tag", sorted(BASE))
def test_the_data_switch_and_the_selector_are_independent(tag):
    """In a build whose muonic-data switch is on, with non-parity capture and transition profiles selected,
    the traces differ from the build without the switch (the switch acts), the provider disabled leaves them
    unchanged, and enabled gives the same ladder as in the build without the switch."""
    ptag = tag + "-data"
    assert builds()[ptag]["geant4_version"].endswith("patched")
    assert identity_problems(ptag, ("absent", "off"), "identity_data") == []
    stock, switched = identity(tag), identity(ptag)
    pairs = [(k, "identity_data_" + k[len("identity_"):]) for k in stock if k.endswith("_absent")]
    assert pairs and all(stock[a] != switched[b] for a, b in pairs)
    at_half = isotopes_at_half(tag)
    for route in ("stopping", "atomic"):
        assert (
            ladder_problems(blocks("ladder", ptag)[f"ladder_data_{route}_serial_on"], "measured", at_half)
            == []
        )


def teardown_lines(tag: str) -> dict[str, str]:
    out = {}
    for case, lines in blocks("teardown", tag).items():
        out[case] = next(x for x in lines if x.startswith("SELECTORS"))
    return out


@pytest.mark.parametrize("tag", sorted(BASE))
def test_repeated_runs_and_teardown(tag):
    """Tasking, three runs, then the run manager deleted. On the stopping route the processes delete every
    selector they were given; on the atomic route the stock G4MuonMinusAtomicCapture deletes none, as it
    deletes none of its own. An instrumented build runs the same cases without a sanitizer report."""
    t = teardown_lines(tag)
    stop = t["teardown_stopping"].split()
    atom = t["teardown_atomic"].split()
    assert int(stop[2]) >= 3 and stop[2] == stop[4]
    assert int(atom[2]) >= 3 and atom[4] == "0"
    with (EVIDENCE / f"cases_{tag}-asan.csv").open(newline="", encoding="utf-8") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    for case in ("teardown_stopping", "teardown_atomic"):
        assert rows[case]["rc"] == "0" and "asan" not in rows[case]["flags"], case


def test_an_installer_that_also_deletes_is_caught_by_the_sanitizer():
    with (EVIDENCE / "cases_v1142-drill-installer_deletes.csv").open(newline="", encoding="utf-8") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    assert builds()["v1142-drill-installer_deletes"]["asan"] == "ON"
    assert rows["teardown_stopping"]["rc"] != "0" and "asan" in rows["teardown_stopping"]["flags"]
