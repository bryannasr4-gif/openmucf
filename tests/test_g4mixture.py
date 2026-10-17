"""The natural-target answer classes and the mixture Geant4 makes of a natural target.

The classes and answers tests regenerate ``record_classes.csv`` and ``natural_target_answers.csv`` byte for
byte, check that every evaluated-profile row's printed label agrees with its key (A-lab), and that the class
shares partition each element's abundance and aggregate to the route shares ``natural_target_capture.csv``
states (A-share). Each gate has a drill that must turn it red.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
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
