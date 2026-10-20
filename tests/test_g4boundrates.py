"""The common bound-rate pair of the two stopped-muon routes: the reference evaluator, the verification policy
and the stock reproducer (``openmucf/g4/d1_boundrates.py``, ``data/g4/d1/bound_rates_policy.json``,
``cpp/tools/repro_bound_rates.cc``)."""

from __future__ import annotations

import importlib.util
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import pytest

from openmucf.g4 import d1_boundrates as bound
from openmucf.g4 import d1_lifetime as lifetime
from openmucf.g4.sources import d1_nuclear_capture as d1

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "cpp" / "tools" / "repro_bound_rates.cc"


def _generator():  # type: ignore[no-untyped-def]
    """``scripts/generate_g4boundrates.py``, loaded by path: ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("generate_g4boundrates_test", ROOT / "scripts" /
                                                  "generate_g4boundrates.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def policy() -> dict:
    return bound.load_policy(ROOT)


@pytest.fixture(scope="module")
def ref() -> bound.Reference:
    return bound.Reference(ROOT)


# --------------------------------------------------------------------------------------------
# the reference evaluator
# --------------------------------------------------------------------------------------------


def test_reference_decay_reproduces_every_compiled_rate_of_the_domain(ref: bound.Reference) -> None:
    """Over every oracle key of the domain, the reference decay rate is the rate a compiled build's
    ``G4MuonMinusBoundDecay::GetMuonDecayRate`` returned, bit for bit."""
    keys = [k for k in ref.oracle.light if bound.in_domain(*k)]
    assert keys and len(keys) == sum(1 for z, a in ref.oracle.light if z >= 2)
    for z, a in keys:
        assert ref.pair(z, a).decay == ref.oracle.light[(z, a)][1], (z, a)


def test_reference_capture_is_the_helpers_compiled_resolver(ref: bound.Reference) -> None:
    """The reference capture rate is the helper's table value divided by the microsecond where the table has
    the key, and the helper's Goulard-Primakoff fallback otherwise; the stopping route's own table gives the
    same bits over the whole domain (the two compiled copies of the table and the fallback are the same)."""
    stopping = d1.load(ROOT / d1.VENDORED_RELPATH)
    table = {(z, a): r for z, a, r, _ in ref.helper.capture_records}
    from openmucf.g4 import d1_natural as natural

    stop_model = natural.reference_model(stopping)
    for z, a in (k for k in ref.oracle.light if bound.in_domain(*k)):
        c = ref.pair(z, a).capture
        if (z, a) in table:
            assert c == table[(z, a)] / d1.MICROSECOND, (z, a)
        assert c == d1.capture_rate(z, a, stopping.capture_records, stop_model), (z, a)


def test_reference_refuses_outside_the_domain(ref: bound.Reference) -> None:
    for z, a in ((1, 1), (14, 28), (6, 0), (6, 5), (82, 208)):
        assert not bound.in_domain(z, a)
        with pytest.raises(bound.BoundRateError):
            ref.pair(z, a)


def test_reference_lifetimes_and_branch_follow_the_pair(ref: bound.Reference) -> None:
    p = ref.pair(6, 12)
    tau, tau_dio, tau_nc = p.lifetimes
    assert (tau, tau_dio, tau_nc) == (1.0 / (p.capture + p.decay), 1.0 / p.decay, 1.0 / p.capture)
    assert p.p_capture == Fraction(p.capture) / (Fraction(p.capture) + Fraction(p.decay))
    zero = bound.Pair(6, 12, p.mass, 0.0, p.decay)
    assert zero.lifetimes[2] == math.inf and zero.p_capture == 0


# --------------------------------------------------------------------------------------------
# the policy
# --------------------------------------------------------------------------------------------


def test_policy_regenerates_from_the_committed_inputs(policy: dict) -> None:
    """The committed policy is what the generator gives from the committed files: integers and strings
    exactly,
    the scipy-computed quantiles and powers to a relative 1e-9."""
    fresh = json.loads(bound.render_policy(_generator().build_policy(ROOT)))

    def compare(a: object, b: object, where: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            assert set(a) == set(b), where
            for key in a:
                compare(a[key], b[key], f"{where}.{key}")
        elif isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b), where
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                compare(x, y, f"{where}[{i}]")
        elif isinstance(a, str) and isinstance(b, str) and re.fullmatch(r"-?[0-9.]+(e[-+]?[0-9]+)?", a):
            x, y = float(a), float(b)
            assert x == y or abs(x - y) <= 1e-9 * max(abs(x), abs(y)), (where, a, b)
        else:
            assert a == b, (where, a, b)

    compare(policy, fresh, "policy")


def test_policy_groups_seeds_strata_and_chunks(policy: dict) -> None:
    gs = policy["groups"]
    assert len(gs) == len(bound.ISOTOPES) * len(bound.REVISIONS) * len(bound.ROUTES)
    assert [g["group"] for g in gs] == list(range(len(gs)))
    total = 0
    seeds: list[tuple[int, int]] = []
    for g in gs:
        z, a, events = bound.isotope(g["isotope"])
        assert (g["z"], g["a"], g["events"]) == (z, a, events)
        assert g["seed_base"] == bound.SEED_FIRST + bound.SEED_STRIDE * g["group"]
        assert [s["t0_ns"] for s in g["strata"]] == list(bound.T0_NS)
        assert sum(s["count"] for s in g["strata"]) == events
        firsts = [sum(x["count"] for x in g["strata"][:i]) for i in range(len(bound.T0_NS))]
        assert [s["first"] for s in g["strata"]] == firsts
        total += events
        seeds.append((g["seed_base"], g["seed_base"] + events - 1))
        assert sum(c["count"] for c in bound.chunks(g)) == events
    assert total == sum(e for *_, e in bound.ISOTOPES) * len(bound.REVISIONS) * len(bound.ROUTES)
    seeds.sort()
    assert all(a[1] < b[0] for a, b in zip(seeds, seeds[1:], strict=False)), "group seed ranges overlap"
    identity_lo = bound.IDENTITY_SEED_FIRST
    identity_hi = bound.identity_seed(bound.REVISIONS[-1], bound.ROUTES[-1], bound.IDENTITY_ROSTER[-1],
                                      bound.T0_NS[-1]) + bound.IDENTITY_EVENTS
    assert seeds[-1][1] < identity_lo < identity_hi < bound.DRILL_SEED_FIRST < 2**31


def test_policy_reference_pairs_are_the_evaluator_s(policy: dict, ref: bound.Reference) -> None:
    for label, z, a, _ in bound.ISOTOPES:
        assert bound.frozen_pair(policy, label) == ref.pair(z, a), label


def test_policy_capture_intervals_have_the_defining_tail_property(policy: dict) -> None:
    """Each capture-count interval is the inclusive binomial quantile interval at half the per-test alpha in
    each tail: checked from its defining property with scipy's binomial distribution, not by repeating the
    generator's quantile call."""
    from scipy import stats

    alpha = float(policy["statistics"]["primary"]["per_test_alpha"])
    assert alpha == pytest.approx(0.001 / 48, rel=1e-15)
    for label, *_ in bound.ISOTOPES:
        lo, hi = policy["statistics"]["primary"]["limits"][label]["capture_count"]
        _, _, events = bound.isotope(label)
        p = float(bound.frozen_pair(policy, label).p_capture)
        assert stats.binom.cdf(lo - 1, events, p) < alpha / 2 <= stats.binom.cdf(lo, events, p), label
        assert stats.binom.sf(hi, events, p) <= alpha / 2 < stats.binom.sf(hi - 1, events, p), label
        q_lo, q_hi = (float(x) for x in policy["statistics"]["primary"]["limits"][label]["sum_all"])
        assert stats.gamma.cdf(q_lo, events) == pytest.approx(alpha / 2, rel=1e-6), label
        assert stats.gamma.sf(q_hi, events) == pytest.approx(alpha / 2, rel=1e-6), label


def test_policy_power_meets_its_target_against_the_stated_alternative(policy: dict) -> None:
    for label, *_ in bound.ISOTOPES:
        pw = policy["statistics"]["power"]["values"][label]
        for test in ("branch", "sum_all", "sum_capture", "sum_decay"):
            assert 0.99 <= float(pw[test]) <= 1.0, (label, test, pw[test])
        assert float(pw["omitted_count_probability"]) < 1e-12


def test_policy_distribution_family(policy: dict) -> None:
    dist = policy["statistics"]["distribution"]
    assert dist["count"] == len(policy["groups"]) * len(bound.STREAMS)
    assert float(dist["per_test_alpha"]) == pytest.approx(0.001 / dist["count"], rel=1e-15)
    from scipy import stats

    lo, hi = (float(x) for x in dist["critical"])
    alpha = float(dist["per_test_alpha"])
    # the enclosure is 1e-20 wide; scipy resolves the tail to about 1e-15 relative
    assert stats.chi2.sf(lo, dist["dof"]) >= alpha * (1 - 1e-12)
    assert stats.chi2.sf(hi, dist["dof"]) <= alpha * (1 + 1e-12)
    assert stats.chi2.sf(lo * (1 - 1e-6), dist["dof"]) > alpha > stats.chi2.sf(hi * (1 + 1e-6), dist["dof"])


# --------------------------------------------------------------------------------------------
# the stock reproducer
# --------------------------------------------------------------------------------------------


def test_stock_reproducer_reads_every_rate_from_geant4() -> None:
    """The reproducer carries no capture or decay formula: none of the literals of the two decay expressions,
    of the Goulard-Primakoff fallback or of the capture table, and no use of the fine-structure constant or of
    an effective charge; every rate comes from a public Geant4 function it calls."""
    text = REPRO.read_text(encoding="ascii")
    code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
    for literal in ("0.45517005", "0.4558514", "fine_structure_const", "2.5", "0.06", "0.868699", "0.708985",
                    "875.e-9", "2.663e-5", "0.75704", "3.24", "Zeff", "zeff", "0.03807"):
        assert literal not in code, literal
    for call in ("G4MuonMinusBoundDecay::GetMuonCaptureRate(Z, A)",
                 "G4MuonMinusBoundDecay::GetMuonDecayRate(Z, A, muMass, massAtomic)",
                 "G4MuonicAtomHelper::GetMuonCaptureRate(Z, A)", "G4MuonicAtomHelper::GetMuonDecayRate(Z)",
                 "GetMuonicAtom(Z, A)", "G4NucleiProperties::GetNuclearMass(A, Z)",
                 "nucleus.AtomicMass(A, Z)"):
        assert call in code, call


def test_oracle_is_the_one_the_policy_names(policy: dict) -> None:
    import hashlib

    assert policy["inputs"]["oracle"] == lifetime.ORACLE_RELPATH
    digest = hashlib.sha256((ROOT / lifetime.ORACLE_RELPATH).read_bytes()).hexdigest()
    assert policy["inputs"]["oracle_sha256"] == digest


# --------------------------------------------------------------------------------------------
# the theory admission gate
# --------------------------------------------------------------------------------------------


def _complete_manifest() -> dict:
    terms = {name: {"value": "1e-6", "source": "a stated derivation"} for name in bound.REQUIRED_TERMS}
    return {"terms": terms, "qualified_roster": ["6-12"],
            "references": [{"id": "independent width", "independent": True, "exposed": False}]}


def test_the_committed_theory_status_is_what_the_gate_gives() -> None:
    manifest = json.loads((ROOT / bound.THEORY_RELPATH).read_text(encoding="utf-8"))
    verdict = bound.theory_admission(manifest)
    assert verdict["status"] == manifest["status"] == "NOT_ESTABLISHED"
    assert manifest["screen"] == "SCREEN_VACUOUS" and manifest["qualified_roster"] == []
    assert len(verdict["problems"]) == len(bound.REQUIRED_TERMS)


def _without(m: dict, term: str) -> dict:
    return {**m, "terms": {k: v for k, v in m["terms"].items() if k != term}}


@pytest.mark.parametrize("label, change, status", [
    ("complete", lambda m: m, "ADMITTED"),
    ("complete, empty roster", lambda m: {**m, "qualified_roster": []}, "SCREEN_VACUOUS"),
    ("a required term removed", lambda m: _without(m, "radiative"), "NOT_ESTABLISHED"),
    ("an unknown term replaced by zero",
     lambda m: {**m, "terms": {**m["terms"], "recoil": {"value": "0", "source": None}}}, "NOT_ESTABLISHED"),
    ("an unsupported nuclide admitted", lambda m: {**m, "qualified_roster": ["14-28"]}, "NOT_ESTABLISHED"),
    ("a dependent reference",
     lambda m: {**m, "references": [{"id": "same formula", "independent": False, "exposed": False}]},
     "NOT_ESTABLISHED"),
    ("an exposed reference as a holdout",
     lambda m: {**m, "references": [{"id": "residuals", "independent": True, "exposed": True}]},
     "NOT_ESTABLISHED"),
    ("a sourced term with no value",
     lambda m: {**m, "terms": {**m["terms"], "recoil": {"value": None, "source": "a stated derivation"}}},
     "NOT_ESTABLISHED"),
    ("a reference that does not state its independence",
     lambda m: {**m, "references": [{"id": "unstated", "exposed": False}]}, "NOT_ESTABLISHED"),
    ("a reference that does not state its exposure",
     lambda m: {**m, "references": [{"id": "unstated", "independent": True}]}, "NOT_ESTABLISHED"),
    ("an admitted nuclide with no reference", lambda m: {**m, "references": []}, "NOT_ESTABLISHED"),
])
def test_drill_the_theory_gate_refuses_each_defect(label: str, change, status: str) -> None:  # type: ignore[no-untyped-def]
    assert bound.theory_admission(change(_complete_manifest()))["status"] == status, label
