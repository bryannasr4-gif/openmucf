"""The natural-target answer classes and the mixture Geant4 makes of a natural target.

The classes and answers tests regenerate ``record_classes.csv`` and ``natural_target_answers.csv`` byte for
byte, check that every evaluated-profile row's printed label agrees with its key (A-lab), and that the class
shares partition each element's abundance and aggregate to the route shares ``natural_target_capture.csv``
states (A-share). Each gate has a drill that must turn it red.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import math
import re
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from openmucf.g4 import d1_lifetime as lifetime
from openmucf.g4 import d1_mixture as mixture
from openmucf.g4 import d1_natural as natural
from openmucf.g4 import spec
from openmucf.g4.sources import d1_nuclear_capture as d1

ROOT = Path(__file__).resolve().parents[1]
D1 = ROOT / "data" / "g4" / "d1"


def _rows(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((D1 / name).read_bytes().decode("ascii"))))


# --------------------------------------------------------------------------------------------
# classes and answers: the committed files
# --------------------------------------------------------------------------------------------


def test_classes_regenerate_byte_for_byte() -> None:
    classes = mixture.record_classes(ROOT)
    assert mixture.render_csv(classes, mixture.CLASS_COLUMNS) == (ROOT / mixture.CLASSES_RELPATH).read_bytes()


def test_answers_regenerate_byte_for_byte() -> None:
    rows = mixture.answers(ROOT)
    assert mixture.render_csv(rows, mixture.ANSWER_COLUMNS) == (ROOT / mixture.ANSWERS_RELPATH).read_bytes()


def test_classes_cover_every_record_of_every_profile() -> None:
    classes = _rows("record_classes.csv")
    tables = natural.capture_profiles(D1)
    for name, table in tables.items():
        keys = {(int(z), int(a)) for z, a, _, _ in table.records}
        assert keys == {(int(r["z"]), int(r["a"])) for r in classes if r["profile"] == name}, name
    assert {r["value_lineage"] for r in classes} <= set(mixture.LINEAGES)
    assert {r["target_kind"] for r in classes} <= set(mixture.TARGET_KINDS)


# --------------------------------------------------------------------------------------------
# A-lab: an evaluated row's printed label agrees with its key
# --------------------------------------------------------------------------------------------


def a_lab_failures(profiles: dict[str, list[tuple[int, int, float, float]]]) -> list[str]:
    """Every evaluated-profile row whose printed label, found by value, does not admit its key: a mass-number
    label keys that mass number, ``nat`` keys ``A = 0``, a bare symbol keys the single natural isotope of its
    element and
    a bare symbol of an element with more than one is not admitted."""
    single = {
        z: m[0].a for z, m in natural.load_isotopes(D1 / "geant4_natural_isotopes.csv").items() if len(m) == 1
    }
    mizuno = _rows("mizuno2025_table3.csv")
    quantity = [
        r for r in _rows("suzuki1987_quantity_rows.csv") if r["own_measurement"] == "true" and r["rate_unc"]
    ]
    from openmucf.g4.sources import suzuki1987

    failures: list[str] = []
    for name, records in profiles.items():
        for z, a, value, unc in records:
            if name == "mizuno2025":
                labels = [
                    r["nuclide"]
                    for r in mizuno
                    if int(r["Z"]) == z and float(r["rate"]) == value and float(r["rate_unc"]) == unc
                ]
            else:
                labels = [
                    r["target_label"]
                    for r in quantity
                    if int(r["Z"]) == z
                    and r["rate_parenthesized"] == "false"
                    and float(suzuki1987.scaled(r, "rate")) == value
                    and float(suzuki1987.scaled(r, "rate_unc")) == unc
                ]
            if len(labels) != 1:
                failures.append(f"{name} ({z}, {a}): {len(labels)} printed labels")
                continue
            label = labels[0]
            if label.startswith("nat"):
                ok = a == 0
            elif label[0].isdigit():
                ok = int("".join(ch for ch in label if ch.isdigit())) == a
            else:
                ok = single.get(z) == a
            if not ok:
                failures.append(f"{name} ({z}, {a}): printed label {label!r} does not admit the key")
    return failures


def _evaluated_records() -> dict[str, list[tuple[int, int, float, float]]]:
    tables = natural.capture_profiles(D1)
    return {
        name: [(int(z), int(a), float(v), float(u)) for z, a, v, u in t.records]
        for name, t in tables.items()
        if name != spec.PARITY_PROFILE
    }


def test_classes_a_lab_every_evaluated_row() -> None:
    assert a_lab_failures(_evaluated_records()) == []


def test_classes_a_lab_drill_natural_row_rekeyed() -> None:
    """Rekey mizuno2025's natural-silicon row `14 0` to `14 28` and drop the enriched 28Si row: the format
    layer and
    the lookup accept the table, and A-lab must not."""
    profiles = _evaluated_records()
    records = [r for r in profiles["mizuno2025"] if (r[0], r[1]) != (14, 28)]
    records = [(14, 28, v, u) if (z, a) == (14, 0) else (z, a, v, u) for z, a, v, u in records]
    table = natural.capture_profiles(D1)["mizuno2025"]
    rekeyed = spec.G4DatTable(directives=table.directives, records=tuple(sorted(records)))
    reparsed = spec.parse(spec.render(rekeyed))
    assert (14, 28) in natural.profile_rows(reparsed)
    failures = a_lab_failures({**profiles, "mizuno2025": records})
    assert failures == ["mizuno2025 (14, 28): printed label 'natSi' does not admit the key"], failures


# --------------------------------------------------------------------------------------------
# A-share: the class shares partition each element and aggregate to the route shares
# --------------------------------------------------------------------------------------------

ROUTE_OF_CLASS = {mixture.EVALUATED_ISOTOPE: natural.PROFILE_ROUTE, mixture.FORMULA: natural.FORMULA_ROUTE}


def a_share_failures(answers: list[dict[str, str]]) -> list[str]:
    isotopes = natural.load_isotopes(D1 / "geant4_natural_isotopes.csv")
    study = {(r["profile"], int(r["z"])): r for r in _rows("natural_target_capture.csv")}
    failures: list[str] = []
    seen: dict[tuple[str, int], list[int]] = {}
    for row in answers:
        seen.setdefault((row["profile"], int(row["z"])), []).append(int(row["a"]))
    for key, members in seen.items():
        if sorted(members) != sorted(i.a for i in isotopes[key[1]]):
            failures.append(f"{key}: the rows do not partition the element's isotopes")
    shares = mixture.class_shares(answers, isotopes)
    routes = (natural.PROFILE_ROUTE, natural.TABLE_ROUTE, natural.FORMULA_ROUTE)
    for key, by_class in shares.items():
        # each class share is the correctly rounded sum of its isotopes' abundances, so the shares of an
        # element differ from its total abundance by at most one rounding per class
        total = math.fsum(i.abundance for i in isotopes[key[1]])
        if abs(math.fsum(by_class.values()) - total) > len(by_class) * math.ulp(total):
            failures.append(f"{key}: the class shares do not sum to the element's abundance")
        abundance = {i.a: i.abundance for i in isotopes[key[1]]}
        exact = {
            route: math.fsum(
                abundance[int(r["a"])]
                for r in answers
                if (r["profile"], int(r["z"])) == key
                and ROUTE_OF_CLASS.get(r["answer_class"], natural.TABLE_ROUTE) == route
            )
            for route in routes
        }
        for route, column in (
            (natural.PROFILE_ROUTE, "profile_share"),
            (natural.TABLE_ROUTE, "table_share"),
            (natural.FORMULA_ROUTE, "formula_share"),
        ):
            if repr(exact[route]) != study[key][column]:
                failures.append(f"{key}: {route} share {exact[route]!r} != {study[key][column]}")
    return failures


def test_answers_a_share_partition_and_aggregate() -> None:
    assert a_share_failures(_rows("natural_target_answers.csv")) == []


def test_answers_a_share_drill_moved_abundance() -> None:
    """Move iron-56 from its natural-composition record to the formula class without changing the route column
    or any total's source: A-share must fail on that element and on no other."""
    answers = _rows("natural_target_answers.csv")
    for row in answers:
        if (row["profile"], row["z"], row["a"]) == ("parity", "26", "56"):
            row["answer_class"] = mixture.FORMULA
    failures = a_share_failures(answers)
    assert failures and all(f.startswith("('parity', 26)") for f in failures), failures


def test_answers_route_and_class_agree() -> None:
    for row in _rows("natural_target_answers.csv"):
        assert ROUTE_OF_CLASS.get(row["answer_class"], natural.TABLE_ROUTE) == row["route"], row


# --------------------------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle() -> mixture.Oracle:
    return mixture.Oracle(ROOT)


def test_oracle_single_isotope_rates_equal_the_mean_life_comparison(oracle: mixture.Oracle) -> None:
    """(i) For an element with a single natural isotope, each route's rate equals the rate the mean-life
    comparison gives its row, bit for bit, under every profile."""
    checked = 0
    for row in lifetime.read_comparison(ROOT / lifetime.COMPARISON_RELPATH):
        z = int(row["z"])
        if not row["key_a"] or len(oracle.isotopes[z]) != 1:
            continue
        stopping = oracle.components(row["profile"], mixture.STOPPING, z)[0]
        atomic = oracle.components(row["profile"], mixture.ATOMIC, z)[0]
        assert repr(stopping.total * d1.MICROSECOND) == row["rate_bound"], row["locator"]
        assert repr((1.0 / atomic.tau) * d1.MICROSECOND) == row["rate_helper"], row["locator"]
        checked += 1
    assert checked > 0


def test_oracle_weighted_capture_rates_equal_the_study(oracle: mixture.Oracle) -> None:
    """(ii) The abundance-weighted capture rate of the oracle's components equals natural_target_capture.csv's
    weighted_rate for every profile and element, on both routes."""
    for row in _rows("natural_target_capture.csv"):
        z = int(row["z"])
        for route in mixture.ROUTES:
            comps = oracle.components(row["profile"], route, z)
            weighted = math.fsum(
                i.abundance * c.capture for i, c in zip(oracle.isotopes[z], comps, strict=True)
            )
            assert repr(weighted * d1.MICROSECOND) == row["weighted_rate"], (row["profile"], z, route)


ESTIMATOR_FAMILIES = {
    "suzuki-ll-constant": mixture.FitSpec(
        "ll", 0.0, 9000.0, True, ((1.0, 0.0),), ((1 / 2026.3, 0.02, True), (1 / 1795.4, 0.05, True))
    ),
    "suzuki-ls-fixed-carbon": mixture.FitSpec(
        "ls", 1500.0, 9000.0, False, ((1.0, 0.0),), ((1 / 2026.3, 0.01, False),)
    ),
    "mizuno-ll-double-pulse": mixture.FitSpec(
        "ll",
        0.0,
        10000.0,
        False,
        ((0.5, 40.0), (0.5, 640.0)),
        ((1 / 87.0, 0.05, True), (1 / 1940.0, 0.02, True), (1 / 25800.0, 0.05, True)),
    ),
}


@pytest.mark.parametrize("family", sorted(ESTIMATOR_FAMILIES))
def test_estimator_returns_a_single_exponential_target_rate(oracle: mixture.Oracle, family: str) -> None:
    """(iii) Every estimator family returns the rate of a pure single-exponential target, with its non-target
    components present, to 1e-25, in 45-digit decimals from a start 1e-3 away."""
    comps = [oracle.components(spec.PARITY_PROFILE, mixture.STOPPING, 13)[0]]
    with localcontext() as context:
        context.prec = 45
        fitted = mixture.fit_rate(comps, ESTIMATOR_FAMILIES[family], conv=Decimal, start=1.001)
        truth = Decimal(comps[0].rate.numerator) / Decimal(comps[0].rate.denominator)
        assert abs(fitted / truth - 1) < Decimal("1e-25")


def test_estimator_two_component_target_matches_an_independent_quadrature(oracle: mixture.Oracle) -> None:
    """(iv) On a synthetic two-component target, the fitted rate equals an independent computation (adaptive
    quadrature and a root of the stationarity equations) to 1e-10, for both methods."""
    integrate = pytest.importorskip("scipy.integrate")
    optimize = pytest.importorskip("scipy.optimize")
    base = oracle.components(spec.PARITY_PROFILE, mixture.STOPPING, 32)
    comps = [
        mixture.Component(
            c.a,
            Fraction(1, 2),
            c.capture,
            c.decay,
            c.total,
            c.tau,
            c.tau_dio,
            c.tau_nc,
            c.rate,
            c.p_dio,
            c.lookup,
            c.answer_class,
            c.record,
            c.record_unc,
        )
        for c in (base[1], base[3])
    ]
    rates = [float(c.rate) for c in comps]
    weights = [float(c.abundance * c.p_dio * c.rate) for c in comps]
    mu, share = 1 / 2026.3, 0.03
    count = sum(w / r for w, r in zip(weights, rates, strict=True))
    amps = [w * (1 - share) / count for w in weights]

    def g(t: float) -> float:
        return sum(a * math.exp(-r * t) for a, r in zip(amps, rates, strict=True)) + share * mu * math.exp(
            -mu * t
        )

    t1, t2 = 0.0, 12 / float(mixture.mean_rate(comps))
    for method in ("ll", "ls"):
        ours = mixture.fit_rate(
            comps, mixture.FitSpec(method, t1, t2, False, ((1.0, 0.0),), ((mu, share, True),))
        )

        def equations(p: list[float], method: str = method) -> list[float]:
            lam, a0, b = p

            def m(t: float) -> float:
                return a0 * math.exp(-lam * t) + b * math.exp(-mu * t)

            def residual(t: float) -> float:
                return g(t) / m(t) - 1 if method == "ll" else (g(t) - m(t)) / g(t)

            parts = (
                lambda t: -a0 * t * math.exp(-lam * t),
                lambda t: math.exp(-lam * t),
                lambda t: math.exp(-mu * t),
            )
            return [
                integrate.quad(lambda t, d=d: residual(t) * d(t), t1, t2, epsabs=0, epsrel=1e-13, limit=500)[
                    0
                ]
                for d in parts
            ]

        start = [float(mixture.mean_rate(comps)), sum(amps), share * mu]
        theirs = optimize.root(equations, start, method="hybr", options={"xtol": 1e-14}).x[0]
        assert abs(ours / theirs - 1) < 1e-10, (method, ours, theirs)


def test_oracle_atomic_branch_is_built_as_the_decay_builds_it(oracle: mixture.Oracle) -> None:
    """(v) The atomic route's branch expectation is Fraction(l1) / Fraction(l1 + l2) from the stored
    lifetimes, in the order G4MuonicAtomDecay::DecayIt forms them; lambda_d / (lambda_c + lambda_d) differs
    from it by less than
    1e-14 here (information: the drill that substitutes it can change nothing measurable)."""
    for revision in ("v11.4.2", "v11.5.0.beta"):
        text = (ROOT / "third_party" / "geant4" / revision / "G4MuonicAtomDecay.cc").read_text(
            encoding="ascii"
        )
        squeezed = "".join(text.split())
        for statement in (
            "G4double lambdac = 1./muatom->GetDIOLifeTime();",
            "G4double lambdad = 1./muatom->GetNCLifeTime();",
            "G4double lambda = lambdac + lambdad;",
            "if (G4UniformRand()*lambda < lambdac)",
        ):
            assert "".join(statement.split()) in squeezed, (revision, statement)
    largest = Fraction(0)
    for z in (3, 17, 32, 47, 82):
        for c in oracle.components(spec.PARITY_PROFILE, mixture.ATOMIC, z):
            l1, l2 = 1.0 / c.tau_dio, 1.0 / c.tau_nc
            assert c.p_dio == Fraction(l1) / Fraction(l1 + l2)
            largest = max(
                largest, abs(c.p_dio - Fraction(c.decay) / (Fraction(c.capture) + Fraction(c.decay)))
            )
    assert largest < Fraction(1, 10**14)


def test_oracle_chi_square_tails_match_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    for dof in (1, 2, 5, 19, 20):
        for x in ("0.5", "7.25", "33.0", "71.5"):
            ours = float(mixture.chi2_sf(Decimal(x), dof))
            assert abs(ours / stats.chi2.sf(float(x), dof) - 1) < 1e-12, (dof, x)
        crit = mixture.critical(dof, Decimal("1e-6"))
        assert crit.lo < Decimal(repr(float(stats.chi2.isf(1e-6, dof)))) * (1 + Decimal("1e-12"))
        assert crit.hi > Decimal(repr(float(stats.chi2.isf(1e-6, dof)))) * (1 - Decimal("1e-12"))
    assert abs(mixture.noncentral_sf(40.0, 19, 30.0) / stats.ncx2.sf(40.0, 19, 30.0) - 1) < 1e-9


# --------------------------------------------------------------------------------------------
# the estimators and the policy
# --------------------------------------------------------------------------------------------

#: The SHA-256 of the committed policy, fixed before any harvest output or comparison existed.
POLICY_SHA256 = "04bd3151358ca0d73af3755d761a991651b1fe57dfc87e7247e7ffcbccd44884"


def test_estimator_labels_follow_the_transcription_mechanically() -> None:
    facts = mixture.load_estimators(ROOT)
    refs = mixture.references(ROOT)
    assert len(refs) == 42 and len({r.z for r in refs}) == 38
    labels = mixture.load_policy(ROOT)["estimators"]["labels"]  # type: ignore[index]
    for ref, recorded in zip(refs, labels, strict=True):
        label, missing = mixture.estimator_label(mixture.row_facts(ROOT, ref, facts))
        assert (ref.locator, label, missing) == (
            recorded["locator"],
            recorded["estimator"],
            recorded["missing"],
        )


def test_estimator_label_drill_a_fully_printed_row_is_established() -> None:
    """A row whose every establishing fact were printed would be ESTABLISHED: the rule can return it."""
    full = {fact: {"status": "printed", "value": "printed value"} for fact in mixture.ESTABLISHING_FACTS}
    assert mixture.estimator_label(full) == (mixture.ESTABLISHED, [])
    missing_range = {**full, "fit_range": {"status": "not_stated", "value": ""}}
    assert mixture.estimator_label(missing_range) == (mixture.SURROGATE, ["fit_range"])


def test_policy_is_pinned_and_regenerates() -> None:
    data = (ROOT / mixture.POLICY_RELPATH).read_bytes()
    assert hashlib.sha256(data).hexdigest() == POLICY_SHA256
    policy = mixture.load_policy(ROOT)
    rebuilt, _ = mixture.build_policy(ROOT, policy["compute"]["ms_per_stop"])  # type: ignore[index]
    assert mixture.render_policy(rebuilt) == data


def test_policy_compute_is_below_the_threshold() -> None:
    compute = mixture.load_policy(ROOT)["compute"]
    assert compute["core_hours"] < compute["stop_threshold_core_hours"]  # type: ignore[index]


def test_policy_brackets_cover_every_reference_row(oracle: mixture.Oracle) -> None:
    for ref in mixture.references(ROOT):
        comps = oracle.components(spec.PARITY_PROFILE, mixture.STOPPING, ref.z)
        fits, nominal = mixture.bracket(ROOT, ref, float(1 / mixture.mean_rate(comps)))
        labels = [label for label, _ in fits]
        assert nominal in labels and len(labels) == len(set(labels)), ref.locator


# --------------------------------------------------------------------------------------------
# the harvest at the call sites: gates recomputed from the committed summary
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harvest() -> dict[str, dict[str, dict[str, str]]]:
    return mixture.load_harvest(ROOT)


@pytest.fixture(scope="module")
def policy() -> dict[str, object]:
    return mixture.load_policy(ROOT)


def _cells(policy: dict[str, object]) -> list[dict[str, object]]:
    return mixture.cells(ROOT, policy)


def _profile_of(cell: dict[str, object]) -> str:
    return str(cell["profile"])


def test_harvest_covers_every_cell_of_the_policy(harvest: dict, policy: dict) -> None:
    expected = {mixture.cell_name(c): c for c in _cells(policy)}
    controls = policy["pristine_controls"]
    for z in controls["elements"]:
        for route in mixture.ROUTES:
            for revision in mixture.REVISIONS:
                expected[f"pristine:{route}:{revision}:{z}"] = {"events": controls["events"]}
    production = {k for k in harvest if not k.startswith("identity")}
    assert production == set(expected)
    for name, cell in expected.items():
        assert int(harvest[name]["meta"]["events"]) == cell["events"], name


def test_harvest_call_sites_agree_with_each_other_and_the_transport(harvest: dict) -> None:
    """In every cell the selector's nucleus is the decay's, the decay's branch is the transported outcome, and
    the
    transported decay electron leaves at the delay the call site drew, on every stop; no time falls on a bin
    edge."""
    for name, stats in harvest.items():
        if name.startswith("identity"):
            continue
        meta = stats["meta"]
        for key in (
            "mismatch_selector_vs_decay",
            "mismatch_branch_vs_outcome",
            "mismatch_delay_vs_electron",
            "time_ties",
        ):
            assert meta[key] == "0", (name, key)


def test_harvest_builds_record_the_committed_harness() -> None:
    builds = mixture.read_csv(ROOT / mixture.BUILDS_RELPATH, mixture.BUILD_COLUMNS)
    source = hashlib.sha256((ROOT / mixture.HARNESS_RELPATH).read_bytes()).hexdigest()
    assert {b["source_sha256"] for b in builds} == {source}
    assert {(b["revision"], b["kind"], b["drill"]) for b in builds} == {
        (r, k, "none") for r in mixture.REVISIONS for k in ("pristine", "patched")
    } | {(r, "patched", d) for r in mixture.REVISIONS for d in ("extra_draw", "early_return")}
    for b in builds:
        assert "-ffp-contract=off" in b["compile_flags"]
        assert ("-DHARVEST_MIXTURE_PATCHED" in b["compile_flags"]) == (b["kind"] == "patched")


def _components(oracle: mixture.Oracle, name: str) -> tuple[str, list[mixture.Component]]:
    profile, route, _, z = name.split(":")
    profile = spec.PARITY_PROFILE if profile == "pristine" else profile
    return route, list(oracle.components(profile, route, int(z)))


def test_gate_every_cell_recomputes_to_its_committed_verdict(harvest: dict, policy: dict, oracle) -> None:
    """Every gate of every cell, recomputed exactly from the committed summary, the policy and the oracle,
    gives
    the verdict the summary commits; G-rate and G-set pass on every cell."""
    for name, stats in harvest.items():
        if name.startswith("identity"):
            continue
        route, comps = _components(oracle, name)
        verdicts = mixture.evaluate_cell(stats, route, comps, policy)
        assert {g: v["verdict"] for g, v in verdicts.items()} == stats["verdict"], name
        assert verdicts["G-rate"]["verdict"] == "pass" and verdicts["G-set"]["verdict"] == "pass", name


def test_gate_identity_controls(harvest: dict) -> None:
    """The random-engine streams, the base delegation and the transported records of every revision, route
    and element of the identity runs; the untouched evaluated cells repeat the parity cells; the pristine
    controls equal the first stops of the patched parity cells."""
    entries = [v["identity"] for k, v in harvest.items() if k.startswith("identity:")]
    assert len(entries) == 16
    for e in entries:
        assert e["rng_identical"] == e["base_delegation"] == e["transport_identical"] == "True"
        assert e["rng_patched_hooked"] == e["rng_patched_unhooked"] == e["rng_pristine_hooked"]
        assert (
            e["transport_patched_hooked"] == e["transport_patched_unhooked"] == e["transport_pristine_hooked"]
        )
        assert e["entry_mismatch"] == e["exit_mismatch"] == "0" and e["hook_calls"] == e["base_calls"]
    for profile, z in (("suzuki1987", 82), ("mizuno2025", 12)):
        for route in mixture.ROUTES:
            for revision in mixture.REVISIONS:
                a = harvest[f"{profile}:{route}:{revision}:{z}"]["meta"]["stops_sha256"]
                b = harvest[f"parity:{route}:{revision}:{z}"]["meta"]["stops_sha256"]
                assert a == b, (profile, route, revision)
    for name, stats in harvest.items():
        if name.startswith("pristine:"):
            parity = harvest["parity:" + name.split(":", 1)[1]]
            assert stats["meta"]["stops_sha256"] == parity["meta"]["first_lines_sha256"], name


def test_drill_identity_hooks(harvest: dict) -> None:
    """A hook that draws one extra random number after the base call turns the engine-stream identity red; one
    that returns before the base call on germanium-74 turns the base-delegation count red."""
    drills = [v["identity"] for k, v in harvest.items() if k.startswith("identity_drill:")]
    assert len(drills) == 4
    for d in drills:
        assert d["extra_draw_rng_red"] == "True" and d["extra_draw_rng"] != d["reference_rng"]
        assert d["early_return_delegation_red"] == "True"
        assert int(d["early_return_hook_calls"]) > int(d["early_return_base_calls"])


def test_gate_hooks_draw_no_random_number() -> None:
    """The hook classes call no random-number function (static check of the region between SPY-BEGIN and
    SPY-END, the drill macros excluded)."""
    text = (ROOT / mixture.HARNESS_RELPATH).read_text(encoding="ascii")
    region = text[text.index("// SPY-BEGIN") : text.index("// SPY-END")]
    kept, skip = [], False
    for line in region.splitlines():
        if line.startswith("#ifdef HARVEST_MIXTURE_DRILL"):
            skip = True
        elif line.startswith("#endif"):
            skip = False
        elif not skip:
            kept.append(line)
    for token in ("G4UniformRand", "G4Random", "CLHEP::Rand", "flat(", "G4RandFlat", "G4Log(G4Uniform"):
        assert token not in "\n".join(kept), token
    assert text.count("G4ElementSelector::SelectZandA(track, target)") == 1
    assert text.count("G4MuonMinusBoundDecay::ApplyYourself(projectile, nucleus)") == 1


# --------------------------------------------------------------------------------------------
# drills of the harvest gates: each planted defect turns its gate red, alone and through the chain
# --------------------------------------------------------------------------------------------


def _replace_capture(components: list[mixture.Component], a: int, capture: float, route: str) -> list:
    out = []
    for c in components:
        if c.a != a:
            out.append(c)
            continue
        total = capture + c.decay
        if route == mixture.STOPPING:
            out.append(
                mixture.Component(
                    c.a,
                    c.abundance,
                    capture,
                    c.decay,
                    total,
                    c.tau,
                    c.tau_dio,
                    c.tau_nc,
                    Fraction(total),
                    mixture.stopping_branch(capture, total),
                    c.lookup,
                    c.answer_class,
                    c.record,
                    c.record_unc,
                )
            )
        else:
            tau = 1.0 / total
            out.append(
                mixture.Component(
                    c.a,
                    c.abundance,
                    capture,
                    c.decay,
                    total,
                    tau,
                    c.tau_dio,
                    1.0 / capture,
                    1 / Fraction(tau),
                    mixture.atomic_branch(c.tau_dio, 1.0 / capture),
                    c.lookup,
                    c.answer_class,
                    c.record,
                    c.record_unc,
                )
            )
    return out


def _copy(stats: dict) -> dict:
    return {k: dict(v) for k, v in stats.items()}


def _time_resampled(stats: dict, components: list[mixture.Component], seed: int) -> dict:
    """The G-time bins when every stop's delay is drawn again from Exp(lambda_bar), the statistic still
    scaling
    by the drawn isotope's rate."""
    import random

    rng = random.Random(seed)
    lam = float(mixture.mean_rate(components))
    bins = [0] * mixture.TIME_BINS
    for c in components:
        rate = float(c.rate)
        for _ in range(int(stats["count"].get(str(c.a), "0"))):
            u = rate * rng.expovariate(lam)
            bins[min(int((1.0 - math.exp(-u)) * mixture.TIME_BINS), mixture.TIME_BINS - 1)] += 1
    out = _copy(stats)
    out["time"] = {str(j): str(n) for j, n in enumerate(bins)}
    return out


def drill_cases(harvest: dict, policy: dict, oracle: mixture.Oracle) -> list[dict[str, object]]:
    """Every drill of the policy's plan on each cell it names (both revisions and both routes), as the planted
    summary and oracle, the target gate and the chain's first red gate."""
    cases: list[dict[str, object]] = []
    alpha = str(policy["statistics"]["per_test_alpha"])
    for entry in policy["drills"]["plan"]:
        for revision in mixture.REVISIONS:
            for route in mixture.ROUTES:
                name = f"parity:{route}:{revision}:{entry['z']}"
                stats = harvest[name]
                comps = list(oracle.components(spec.PARITY_PROFILE, route, int(entry["z"])))
                drill, gate = entry["drill"], entry["gate"]
                planted_stats, planted = _copy(stats), comps
                if drill == "rate_substitution":
                    natural_rate = next(c.capture for c in comps if c.a == 56)
                    planted = _replace_capture(comps, 54, natural_rate, route)
                elif drill == "fallback_dropped":
                    planted = mixture._without_fallback(comps)
                elif drill == "formula_stops_removed":
                    for c in comps:
                        if c.answer_class == mixture.FORMULA:
                            planted_stats["count"].pop(str(c.a), None)
                elif drill == "branch_swap":
                    top = max(comps, key=lambda c: int(stats["count"][str(c.a)]))
                    cap = int(stats["branch"][f"{top.a}:N"])
                    moved = int(Fraction(policy["drills"]["branch_swap_fraction"]) * cap)
                    planted_stats["branch"][f"{top.a}:N"] = str(cap - moved)
                    planted_stats["branch"][f"{top.a}:D"] = str(int(stats["branch"][f"{top.a}:D"]) + moved)
                elif drill == "exp_lambda_bar":
                    planted_stats = _time_resampled(stats, comps, int(policy["drills"]["seed"]))
                elif drill == "single_exponential":
                    planted = mixture._single_exponential(comps)
                elif drill == "omitted_fallback":
                    planted = mixture._without_fallback(comps)
                elif drill == "single_exponential_window":
                    pass  # the window's oracle is replaced below
                elif drill == "atomic_expectation":
                    if route != mixture.ATOMIC:
                        continue
                    alt = mixture.gate_branch(
                        stats,
                        comps,
                        alpha,
                        branch_expectation=lambda c: (
                            Fraction(c.decay) / (Fraction(c.capture) + Fraction(c.decay))
                        ),
                    )
                    base = mixture.gate_branch(stats, comps, alpha)
                    largest = max(
                        abs(c.p_dio - Fraction(c.decay) / (Fraction(c.capture) + Fraction(c.decay)))
                        for c in comps
                    )
                    cases.append(
                        {
                            "drill": drill,
                            "cell": name,
                            "gate": gate,
                            "information": True,
                            "alone": alt["verdict"],
                            "unchanged": alt["verdict"] == base["verdict"],
                            "largest_difference": float(largest),
                        }
                    )
                    continue
                if drill == "single_exponential_window":
                    wedges = mixture.window_edges(comps)
                    probs = mixture.window_probabilities(mixture._single_exponential(comps), wedges)
                    counts = [int(stats["win"][str(j)]) for j in range(len(wedges) - 1)]
                    alone = mixture._x2_verdict(counts, probs, alpha)
                    chain = {
                        g: v["verdict"] for g, v in mixture.evaluate_cell(stats, route, comps, policy).items()
                    }
                    chain["G-win"] = alone["verdict"]
                    first = next((g for g in mixture.GATES if chain.get(g, "pass") != "pass"), None)
                else:
                    verdicts = mixture.evaluate_cell(planted_stats, route, planted, policy)
                    alone = verdicts[gate]
                    first = mixture.first_red(verdicts)
                    if gate == "G-DIO":
                        # the harvested counts in the harvest's own bins against the drill oracle's
                        # probabilities
                        # there, so the statistic, not the bin check, has to turn red
                        dedges = [
                            float.fromhex(stats["dio_edge"][str(j)]) for j in range(len(stats["dio_edge"]))
                        ]
                        counts = [int(stats["dio"][str(j)]) for j in range(len(dedges))] + [
                            int(stats["dio"]["capture"])
                        ]
                        alone = mixture._x2_verdict(counts, mixture.dio_probabilities(planted, dedges), alpha)
                cases.append(
                    {
                        "drill": drill,
                        "cell": name,
                        "gate": gate,
                        "alone": alone["verdict"],
                        "first_red": first,
                        "x2": alone.get("x2", ""),
                        "information": False,
                    }
                )
    return cases


def test_drill_every_harvest_gate_turns_red(harvest: dict, policy: dict, oracle) -> None:
    cases = drill_cases(harvest, policy, oracle)
    graded = [c for c in cases if not c["information"]]
    assert graded and all(c["alone"] == "fail" for c in graded), [c for c in graded if c["alone"] != "fail"]
    assert all(c["first_red"] is not None for c in graded)
    info = [c for c in cases if c["information"]]
    assert info and all(c["unchanged"] for c in info)


# --------------------------------------------------------------------------------------------
# the comparison with the measured mean lives
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison() -> list[dict[str, str]]:
    return mixture.with_measured(
        ROOT, mixture.read_csv(ROOT / mixture.COMPARISON_RELPATH, mixture.COMPARISON_COLUMNS)
    )


NUMERIC_COLUMNS = {
    "fit_rate",
    "fit_rate_min",
    "fit_rate_max",
    "sigma_tab",
    "sigma_win",
    "mean_rate",
    "alias_rate",
    "difference",
    "difference_sigma",
    "part_estimator",
    "part_composition",
    "part_single_nucleus",
    "part_merged",
    "formula_share",
    "same_source_share",
}


def test_compare_regenerates_within_tolerance() -> None:
    """The comparison regenerates with every number within a relative 1e-9 (the solver's last digits may
    differ
    between platforms) and every other field, the verdicts among them, exactly."""
    committed = mixture.read_csv(ROOT / mixture.COMPARISON_RELPATH, mixture.COMPARISON_COLUMNS)
    rows = mixture.comparison_rows(ROOT)
    assert len(rows) == len(committed)
    for new, old in zip(rows, committed, strict=True):
        for column in mixture.COMPARISON_COLUMNS:
            if column in NUMERIC_COLUMNS and old[column]:
                assert float(new[column]) == pytest.approx(float(old[column]), rel=1e-9, abs=1e-15), column
            else:
                assert new[column] == old[column], (old["locator"], column)


def _sigma(row: dict[str, str]) -> float:
    return math.sqrt(sum(float(row[k]) ** 2 for k in ("measured_rate_unc", "sigma_tab", "sigma_win")))


def test_compare_verdicts_follow_the_band(comparison: list[dict[str, str]]) -> None:
    for row in comparison:
        inside = mixture._within(
            row["fit_rate"],
            row["measured_rate"],
            (row["measured_rate_unc"], row["sigma_tab"], row["sigma_win"]),
        )
        assert row["verdict"] == ("WITHIN" if inside else "OUTSIDE"), row["locator"]


def test_compare_parts_add_up_to_the_difference(comparison: list[dict[str, str]]) -> None:
    for row in comparison:
        parts = [row["part_estimator"]] + (
            [row["part_merged"]]
            if row["part_merged"]
            else [row["part_composition"], row["part_single_nucleus"]]
        )
        total = math.fsum(float(p) for p in parts)
        assert abs(total - float(row["difference"])) <= 1e-12 * max(1.0, abs(float(row["measured_rate"]))), (
            row
        )


def test_compare_counts_and_qualification(comparison: list[dict[str, str]]) -> None:
    """42 measurements of 38 elements under three profiles and two routes; every estimator a surrogate; no row
    qualifies, so the qualified family is vacuous, reported as such."""
    assert len({(r["source"], r["locator"]) for r in comparison}) == 42
    assert len({r["z"] for r in comparison}) == 38
    assert {r["estimator"] for r in comparison} == {mixture.SURROGATE}
    assert all(r["qualified"] == "false" for r in comparison)
    for verdicts in mixture.science(comparison).values():
        assert verdicts["qualified"] == "SCREEN_VACUOUS"
        assert verdicts["all"] == ("SCIENCE_FAIL" if verdicts["outside"] else "UNQUALIFIED")


def test_drill_band_shift_both_ways(comparison: list[dict[str, str]]) -> None:
    """A measured value moved to the band's edge plus a hair, either way, is OUTSIDE; minus a hair, WITHIN."""
    for row in comparison[:12]:
        fit, sd = float(row["fit_rate"]), _sigma(row)
        sig = (row["measured_rate_unc"], row["sigma_tab"], row["sigma_win"])
        for sign in (1, -1):
            assert not mixture._within(row["fit_rate"], repr(fit + sign * 3 * sd * (1 + 1e-9)), sig)
            assert mixture._within(row["fit_rate"], repr(fit + sign * 3 * sd * (1 - 1e-9)), sig)


def test_drill_initial_rate_in_place_of_the_fit(comparison: list[dict[str, str]]) -> None:
    """Replacing the estimator's rate by the mixture's initial rate changes the published file wherever the
    two differ by more than a tenth of the band's standard deviation; there are such rows, so the estimator is
    not a
    no-op in this comparison."""
    moved = [r for r in comparison if abs(float(r["fit_rate"]) - float(r["mean_rate"])) > 0.1 * _sigma(r)]
    assert moved
    for row in moved:
        old = float(row["difference_sigma"])
        new = (float(row["mean_rate"]) - float(row["measured_rate"])) / _sigma(row)
        assert abs(new - old) > 0.1


def test_compare_same_source_shares(comparison: list[dict[str, str]]) -> None:
    """A Suzuki row answered by the compiled-in table, or any row answered by its own source's profile, is
    same-source on that share; a Mizuno row under parity is answered by another source's records."""
    for row in comparison:
        share = float(row["same_source_share"])
        if row["source"] == "mizuno2025" and row["profile"] == "parity":
            assert share == 0.0
        if row["source"] == "suzuki1987" and row["profile"] == "parity":
            assert share == pytest.approx(1.0 - float(row["formula_share"]))


# --------------------------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------------------------

DOCUMENT = "DATASET_D1.md"
_SPELLED = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million)"
    r"(?:-(?:one|two|three|four|five|six|seven|eight|nine))*\b",
    re.IGNORECASE,
)


def _line_pattern(line: str, numeric: re.Pattern[str]) -> tuple[str, tuple[int, ...]]:
    """A pattern matching ``line`` whitespace-collapsed, with every numeric token and spelled number a
    group."""
    collapsed = " ".join(line.split())
    spans = sorted(
        {m.span() for m in numeric.finditer(collapsed)} | {m.span() for m in _SPELLED.finditer(collapsed)}
    )
    parts, groups, at = [], [], 0
    for index, (start, end) in enumerate(spans, start=1):
        parts.append(re.escape(collapsed[at:start]))
        parts.append(f"({re.escape(collapsed[start:end])})")
        groups.append(index)
        at = end
    parts.append(re.escape(collapsed[at:]))
    return "".join(parts), tuple(groups)


@functools.cache
def _section() -> str:
    return mixture.render_section(ROOT)


def document_pins(numeric: re.Pattern[str]) -> list[tuple[str, str, str, tuple[int, ...], object]]:
    """``(what, path, pattern, groups, expected)`` for every number section 13 of DATASET_D1.md states: one
    row per line of the generated section that carries a number, every number a group (the section is required
    to be the
    generated text, so each number is the generator's), and one per pointer sentence of sections 11 and 12."""
    number = re.match(r"## (\d+)\. ", mixture.SECTION_HEADING)
    assert number is not None
    pins: list[tuple[str, str, str, tuple[int, ...], object]] = [
        (
            "section 13 pointer, section 11",
            DOCUMENT,
            r"Section (\d+) gives the class of record",
            (1,),
            number[1],
        ),
        (
            "section 13 pointer, section 12",
            DOCUMENT,
            r"Section (\d+) compares the transported mixture",
            (1,),
            number[1],
        ),
    ]
    for index, line in enumerate(_section().splitlines(), start=1):
        pattern, groups = _line_pattern(line, numeric)
        if groups:
            pins.append((f"section 13 line {index}", DOCUMENT, pattern, groups, None))
    return pins


def test_document_section_13_is_the_generated_text() -> None:
    text = (ROOT / DOCUMENT).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.count("\n" + mixture.SECTION_HEADING + "\n") == 1
    assert "## 13. " + text.split("\n## 13. ", 1)[1] == _section()


def test_document_pointer_sentences_one_each() -> None:
    text = (ROOT / DOCUMENT).read_text(encoding="utf-8").replace("\r\n", "\n")
    eleven = text.split("\n## 11. ", 1)[1].split("\n## 12. ", 1)[0]
    twelve = text.split("\n## 12. ", 1)[1].split("\n## 13. ", 1)[0]
    assert eleven.count("Section 13") == 1 and twelve.count("Section 13") == 1


def test_document_pins_match_once() -> None:
    collapsed = " ".join((ROOT / DOCUMENT).read_text(encoding="utf-8").split())
    for what, _, pattern, _, expected in document_pins(re.compile(r"[0-9][0-9,.]*[0-9]|[0-9]|[⁰¹²³⁴⁵⁶⁷⁸⁹]+")):
        hits = list(re.finditer(pattern, collapsed))
        assert len(hits) == 1, what
        if expected is not None:
            assert hits[0].group(1) == expected, what
