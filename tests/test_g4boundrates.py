"""The common bound-rate pair of the two stopped-muon routes: the reference evaluator, the verification policy
and the stock reproducer (``openmucf/g4/d1_boundrates.py``, ``data/g4/d1/bound_rates_policy.json``,
``cpp/tools/repro_bound_rates.cc``)."""

from __future__ import annotations

import csv
import dataclasses
import gzip
import hashlib
import importlib.util
import json
import math
import re
import shutil
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
# the summarizer and the gates, on synthetic records with known answers
# --------------------------------------------------------------------------------------------

#: The capture model's local time in the synthetic records: negative, like the cascade model's default, so the
#: clamp of the stopping caller cannot hide a fixture's clock defect.
MODEL_TAU = -1.0
#: The same on the atomic route: not negative, the case the frozen clock predictions assume there (the
#: physically correct time and the delivered time then agree at t0 = 0).
ATOM_MODEL_TAU = 0.25


def write_stream(directory: Path, route: str, pair: bound.Pair, n: int, t0: float, seed: int = 7, *,  # noqa: C901
                 first: int = 0, mutate=None, constant_delay: bool = False,  # type: ignore[no-untyped-def]
                 physical_atom_clock: bool = False, shift: float = 0.0,
                 consumed: tuple[float, float] | None = None) -> Path:
    """The stops.txt, bound.txt and bound_summary.txt a trace build writes for ``n`` events with the switch
    on, drawn here from the competing exponentials of ``pair`` (or of ``consumed``, the pair the consumer is
    made to use). The atomic records carry the implementation's clock (atom created at 2 t0) unless
    ``physical_atom_clock``; ``shift`` moves every atomic product time; ``constant_delay`` makes every delay
    1/Lambda. ``mutate(lines)`` may edit the bound lines before they are written."""
    import random

    rng = random.Random(seed)
    c, d = consumed if consumed is not None else (pair.capture, pair.decay)
    lam = c + d
    tau, tau_dio, tau_nc = 1.0 / lam, 1.0 / d, (1.0 / c if c > 0.0 else math.inf)
    raw = directory / "raw"
    raw.mkdir(parents=True)
    stops, lines = [], []
    za = f"{pair.z} {pair.a}"
    sums: dict[str, list[float]] = {"all": [], "capture": [], "decay": []}
    captures = 0
    for k in range(n):
        i = first + k
        elapsed = 1.0 / lam if constant_delay else max(rng.expovariate(lam), 1.5)
        captured = rng.random() * lam < c
        captures += captured
        taus = [MODEL_TAU if route == bound.STOPPING else ATOM_MODEL_TAU] * 2 if captured else []
        branch = "N" if captured else "D"
        if route == bound.STOPPING:
            products = [t0 + max(elapsed + t, 0.0) for t in taus]
            electron = None if captured else t0 + elapsed
            e = electron.hex() if electron is not None else "-"
            stops.append(f"{i} {za} {za} {c.hex()} {d.hex()} {0.0.hex()} {elapsed.hex()} "
                         f"{1 if captured else 0} {branch} {e}")
            fields = ["S", str(i), str(pair.z), str(pair.a), 0.0.hex(), elapsed.hex(), branch,
                      "P", "1", "0", "0", c.hex(), d.hex()]
        else:
            counter = elapsed / tau
            elapsed = counter * tau
            create = (t0 if physical_atom_clock else 2.0 * t0) + shift
            if physical_atom_clock:
                products = [t0 + elapsed + max(t, 0.0) + shift for t in taus]
                electron = None if captured else t0 + elapsed + shift
            else:
                products = [create + max(elapsed + t, 0.0) for t in taus]
                electron = None if captured else create + elapsed
            e = electron.hex() if electron is not None else "-"
            mid = [tau.hex(), tau_dio.hex(), tau_nc.hex(), counter.hex(), elapsed.hex(), create.hex(), branch]
            stops.append(" ".join([str(i), za, za, *mid, branch, e]))
            if k == 0:
                lines.append(f"C {i} helper {pair.z} {pair.a} 0 {c.hex()} {d.hex()}")
            fields = ["A", str(i), str(pair.z), str(pair.a), create.hex(), *mid, branch]
        sums["all"].append(elapsed)
        sums["capture" if captured else "decay"].append(elapsed)
        fields += ["M", "1" if captured else "0", str(len(taus)), *(t.hex() for t in taus),
                   "N", str(len(products)), *(t.hex() for t in products), "E", e]
        lines.append(" ".join(fields))
    if mutate is not None:
        lines = mutate(lines)
    (raw / "stops.txt").write_text("\n".join(stops) + "\n", encoding="ascii")
    (raw / "bound.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    (raw / "bound_summary.txt").write_text(
        f"events {n}\ncaptures {captures}\nsum_all {math.fsum(sums['all']).hex()}\n"
        f"sum_capture {math.fsum(sums['capture']).hex()}\nsum_decay {math.fsum(sums['decay']).hex()}\n",
        encoding="ascii")
    return raw


@pytest.fixture(scope="module")
def c12(policy: dict) -> bound.Pair:
    return bound.frozen_pair(policy, "C-12")


def _summary(tmp_path: Path, route: str, pair: bound.Pair, n: int = 4000, t0: float = 250.0,
             **kw) -> dict:  # type: ignore[no-untyped-def]
    raw = write_stream(tmp_path / "chunk", route, pair, n, t0, **kw)
    return bound.merge([bound.summarize_chunk(raw, route, pair, t0, kw.get("first", 0), n)])


def _section(line: str, tag: str) -> tuple[list[str], int]:
    t = line.split()
    return t, t.index(tag)


@pytest.mark.parametrize("route", bound.ROUTES)
def test_a_correct_stream_passes_every_gate_and_only_the_predicted_clocks_fail(
        tmp_path: Path, c12: bound.Pair, route: str) -> None:
    s = _summary(tmp_path, route, c12, t0=250.0)
    assert bound.gate_complete(s, 4000)["verdict"] == "pass", bound.gate_complete(s, 4000)
    assert bound.gate_rate(s, route)["verdict"] == "pass"
    assert int(s["rate_exact"]) == 4000
    dist = bound.gate_distribution(s, 36)
    assert {v["verdict"] for v in dist.values()} == {"pass"}, dist
    clocks = bound.gate_clocks({250: s}, route, (250,))
    assert clocks["delivery"]["verdict"] == "pass", clocks["delivery"]
    for key, v in clocks["physical"].items():
        assert v["as_predicted"], (key, v)


def test_the_physical_clock_passes_a_correct_atomic_clock_and_fails_a_shifted_one(tmp_path: Path,
                                                                                   c12: bound.Pair) -> None:
    """A synthetic atomic stream with the physically correct clock (atom at t0, decay at t0 + T, capture
    products at t0 + T + max(tau, 0)) passes the physical gate; the same stream with every atomic product
    moved by 5 ns fails it. The reference origin is t0, never a product's own time."""
    good = _summary(tmp_path / "good", bound.ATOMIC, c12, physical_atom_clock=True)
    phys = bound.gate_clocks({250: good}, bound.ATOMIC, (250,))["physical"]
    assert all(v["verdict"] == "pass" for v in phys.values()), phys
    bad = _summary(tmp_path / "bad", bound.ATOMIC, c12, physical_atom_clock=True, shift=5.0)
    phys = bound.gate_clocks({250: bad}, bound.ATOMIC, (250,))["physical"]
    assert all(v["verdict"] == "fail" for v in phys.values()), phys


def _move_electron(change):  # type: ignore[no-untyped-def]
    def mutate(lines: list[str]) -> list[str]:
        out = []
        for line in lines:
            t = line.split()
            if t[0] == "S" and t[-1] != "-":
                t[-1] = change(float.fromhex(t[-1]), float.fromhex(t[5])).hex()
            out.append(" ".join(t))
        return out
    return mutate


def _move_products(change):  # type: ignore[no-untyped-def]
    def mutate(lines: list[str]) -> list[str]:
        out = []
        for line in lines:
            t = line.split()
            if t[0] == "S":
                n = t.index("N", t.index("M"))
                for j in range(n + 2, n + 2 + int(t[n + 1])):
                    t[j] = change(float.fromhex(t[j]), float.fromhex(t[5])).hex()
            out.append(" ".join(t))
        return out
    return mutate


@pytest.mark.parametrize("label, mutate", [
    ("t0 omitted", _move_electron(lambda t, T: t - 250.0)),
    ("t0 doubled", _move_electron(lambda t, T: t + 250.0)),
    ("sampled delay omitted", _move_electron(lambda t, T: t - T)),
    ("sampled delay doubled", _move_electron(lambda t, T: t + T)),
    ("t0 omitted from the capture products", _move_products(lambda t, T: t - 250.0)),
])
def test_drill_each_clock_defect_turns_the_delivery_gate_red(tmp_path: Path, c12: bound.Pair, label: str,
                                                             mutate) -> None:  # type: ignore[no-untyped-def]
    s = _summary(tmp_path, bound.STOPPING, c12, mutate=mutate)
    assert bound.gate_clocks({250: s}, bound.STOPPING, (250,))["delivery"]["verdict"] == "fail", label


def test_drill_the_constant_delay_stream_passes_the_mean_test_and_fails_the_distribution(
        tmp_path: Path, c12: bound.Pair) -> None:
    """Every delay 1/Lambda, the branch counts and the clocks correct: the scaled sum of all delays sits at
    its mean, inside its gamma interval, and the distribution family fails."""
    from scipy import stats

    s = _summary(tmp_path, bound.STOPPING, c12, n=20000, constant_delay=True)
    n = int(s["events"])
    u = c12.total * float.fromhex(s["sums"]["all"])
    alpha = 0.001 / 48
    assert stats.gamma.ppf(alpha / 2, n) <= u <= stats.gamma.isf(alpha / 2, n)
    assert bound.gate_complete(s, n)["verdict"] == "pass"
    assert bound.gate_rate(s, bound.STOPPING)["verdict"] == "pass"
    assert bound.gate_distribution(s, 36)["all"]["verdict"] == "fail"


def _unobserve_rates(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        t, p = _section(line, "P") if line.startswith("S") else (line.split(), -1)
        out.append(" ".join(t[:p + 1] + ["x"] + t[p + 6:]) if p >= 0 else line)
    return out


def _erase_outcome(lines: list[str]) -> list[str]:
    out = []
    for i, line in enumerate(lines):
        t = line.split()
        if i == 3:
            n = t.index("N", t.index("M"))
            t = t[:n] + ["N", "0", "E", "-"]
        out.append(" ".join(t))
    return out


@pytest.mark.parametrize("label, mutate, problem", [
    ("every rate record removed", _unobserve_rates, "wiring_unobserved"),
    ("one event's record removed", lambda ls: ls[:10] + ls[11:], "missing"),
    ("an event duplicated", lambda ls: ls + ls[5:6], "duplicates"),
    ("an outcome erased, never counted as capture", _erase_outcome, "outcome_none"),
])
def test_drill_missing_duplicate_or_unobserved_records_fail_completeness(tmp_path: Path, c12: bound.Pair,
                                                                          label: str, mutate,  # type: ignore[no-untyped-def]
                                                                          problem: str) -> None:
    s = _summary(tmp_path, bound.STOPPING, c12, mutate=mutate)
    verdict = bound.gate_complete(s, 4000)
    assert verdict["verdict"] == "fail", label
    assert any(problem in p for p in verdict["problems"]), (label, verdict)


def test_drill_a_lost_branch_count_or_an_overlapping_range_fails_completeness(tmp_path: Path,
                                                                              c12: bound.Pair) -> None:
    raw = write_stream(tmp_path / "a", bound.STOPPING, c12, 2000, 0.0)
    text = (raw / "bound_summary.txt").read_text().splitlines()
    (raw / "bound_summary.txt").write_text("\n".join(x for x in text if not x.startswith("captures")) + "\n")
    s = bound.merge([bound.summarize_chunk(raw, bound.STOPPING, c12, 0.0, 0, 2000)])
    assert "the in-process sums disagree with the records" in bound.gate_complete(s, 2000)["problems"]
    a = bound.summarize_chunk(write_stream(tmp_path / "b", bound.STOPPING, c12, 2000, 0.0), bound.STOPPING,
                              c12, 0.0, 0, 2000)
    b = bound.summarize_chunk(write_stream(tmp_path / "c", bound.STOPPING, c12, 2000, 0.0, first=1000),
                              bound.STOPPING, c12, 0.0, 1000, 2000)
    merged = bound.merge([a, b])
    assert merged["chunk_overlap"] == 1000
    assert any("more than one chunk" in p for p in bound.gate_complete(merged, 4000)["problems"])


def test_drill_compensation_keeps_the_lifetime_and_fails_the_rates(tmp_path: Path, c12: bound.Pair) -> None:
    """The consumer given the atomic route's former decay rate, the difference moved into capture: the total
    agrees with the reference, the pair (and the two partial lifetimes) do not."""
    from openmucf.g4 import d1_lifetime as lt

    helper = d1.load_helper(ROOT / d1.HELPER_RELPATH)
    zmin, zmax = lt.helper_clamp(helper)
    d_bad = lt.helper_decay_rate(6, helper.zeff, zmin, zmax)
    c_bad = c12.capture + (c12.decay - d_bad)
    for route in bound.ROUTES:
        s = _summary(tmp_path / route, route, c12, consumed=(c_bad, d_bad))
        assert bound.gate_rate(s, route)["verdict"] == "fail", route
        comps = [float.fromhex(x) for x in s["rate_components"]]
        total = comps[2] if route == bound.STOPPING else comps[0]
        assert total <= bound.RATE_RTOL < max(comps), (route, comps)


def test_drill_a_stopping_stream_without_the_provider_call_fails_the_wiring_alone(tmp_path: Path,
                                                                                  c12: bound.Pair) -> None:
    """The stopping consumer that never asks the provider, with the switch on: completeness and the
    distribution pass (its own pair is the stopping prescription), and the wiring gate fails."""
    s = _summary(tmp_path, bound.STOPPING, c12, mutate=_rewrite_rate_call(["0", "0", "-1", "x", "x"]))
    verdict = bound.gate_rate(s, bound.STOPPING)
    assert verdict["verdict"] == "fail" and any("wiring" in p for p in verdict["problems"]), verdict
    assert bound.gate_complete(s, 4000)["verdict"] == "pass"
    assert {v["verdict"] for v in bound.gate_distribution(s, 36).values()} == {"pass"}


def _rewrite_rate_call(tokens: list[str]):  # type: ignore[no-untyped-def]
    """A ``mutate`` that replaces the first ``len(tokens)`` fields after ``P`` on every stopping line."""
    def mutate(lines: list[str]) -> list[str]:
        out = []
        for line in lines:
            t = line.split()
            if t[0] == "S":
                p = t.index("P")
                t[p + 1:p + 1 + len(tokens)] = tokens
            out.append(" ".join(t))
        return out
    return mutate


def test_drill_a_pair_written_beside_a_status_other_than_common_is_malformed(tmp_path: Path,
                                                                            c12: bound.Pair) -> None:
    """The harness writes the pair only with the common status (``x x`` otherwise): a record that carries a
    pair beside the no-call status is refused at the read, so no event is counted and neither gate passes."""
    s = _summary(tmp_path, bound.STOPPING, c12, n=400, mutate=_rewrite_rate_call(["0", "0", "-1"]))
    assert s["malformed"] == 400 and s["events"] == 0
    assert "malformed" in bound.gate_complete(s, 400)["problems"]
    assert "no event whose rates could be checked" in bound.gate_rate(s, bound.STOPPING)["problems"]


# --------------------------------------------------------------------------------------------
# the typed reader: a nonfinite or unobserved value is refused before any statistic sees it
# --------------------------------------------------------------------------------------------

NONFINITE = ("nan", "inf", "-inf")


def _helper(edit):  # type: ignore[no-untyped-def]
    """A ``mutate`` that rewrites the helper's provider-call line with ``edit`` (tokens -> line)."""
    return lambda lines: [edit(x.split()) if x.startswith("C ") else x for x in lines]


def _every(t: list[str]) -> bool:
    return True


def _model_time(t: list[str]) -> int:
    return t.index("M") + 3


def _product_time(t: list[str]) -> int:
    return t.index("N", t.index("M")) + 2


def _electron_time(t: list[str]) -> int:
    return len(t) - 1


#: (label, route, which line, the field) for every binary64 field of the two record kinds
NONFINITE_FIELDS = [
    ("stopping time before the call", bound.STOPPING, _every, lambda t: 4),
    ("stopping time after the call", bound.STOPPING, _every, lambda t: 5),
    ("stopping consumed capture rate", bound.STOPPING, _every, lambda t: 11),
    ("stopping consumed decay rate", bound.STOPPING, _every, lambda t: 12),
    ("stopping capture-model time", bound.STOPPING, lambda t: _captured(t), _model_time),
    ("stopping capture product time", bound.STOPPING, lambda t: _captured(t), _product_time),
    ("stopping decay-electron time", bound.STOPPING, lambda t: _decayed(t), _electron_time),
    ("atom-creation time", bound.ATOMIC, _every, lambda t: 4),
    ("atomic lifetime", bound.ATOMIC, _every, lambda t: 5),
    ("atomic DIO lifetime", bound.ATOMIC, _every, lambda t: 6),
    ("atomic nuclear-capture lifetime", bound.ATOMIC, _every, lambda t: 7),
    ("atomic interaction-length counter", bound.ATOMIC, _every, lambda t: 8),
    ("atomic delay", bound.ATOMIC, _every, lambda t: 9),
    ("atomic atom time", bound.ATOMIC, _every, lambda t: 10),
    ("atomic capture-model time", bound.ATOMIC, lambda t: _captured(t), _model_time),
    ("atomic capture product time", bound.ATOMIC, lambda t: _captured(t), _product_time),
    ("atomic decay-electron time", bound.ATOMIC, lambda t: _decayed(t), _electron_time),
]


@pytest.mark.parametrize("value", NONFINITE)
@pytest.mark.parametrize("label, route, which, index_of", NONFINITE_FIELDS,
                         ids=[c[0] for c in NONFINITE_FIELDS])
def test_drill_a_nonfinite_field_is_refused_at_the_read(  # type: ignore[no-untyped-def]
        tmp_path: Path, c12: bound.Pair, label: str, route: str, which, index_of, value: str) -> None:
    """One field of one record set to NaN, +inf or -inf: the record is malformed, its event is missing and
    completeness fails; the same stream restored passes completeness and the rates."""
    n = 400
    clean = _summary(tmp_path / "clean", route, c12, n=n)
    assert bound.gate_complete(clean, n)["verdict"] == "pass"
    assert bound.gate_rate(clean, route)["verdict"] == "pass"
    tag = "S" if route == bound.STOPPING else "A"
    s = _summary(tmp_path / "defect", route, c12, n=n, mutate=_first_line(tag, which, _set(index_of, value)))
    assert (s["malformed"], s["events"], s["missing"]) == (1, n - 1, 1), label
    problems = bound.gate_complete(s, n)["problems"]
    assert "malformed" in problems and "missing" in problems, (label, problems)


#: (label, route, which line, the field, the value) -- finite values a lifetime or a rate may not take
NONPOSITIVE_FIELDS = [
    ("a zero atomic lifetime", bound.ATOMIC, _every, lambda t: 5, "0x0p+0"),
    ("a negative atomic DIO lifetime", bound.ATOMIC, _every, lambda t: 6, "-0x1p+0"),
    ("a zero nuclear-capture lifetime", bound.ATOMIC, _every, lambda t: 7, "0x0p+0"),
    ("a zero consumed decay rate", bound.STOPPING, _every, lambda t: 12, "0x0p+0"),
    ("a negative consumed capture rate", bound.STOPPING, _every, lambda t: 11, "-0x1p-5"),
]


@pytest.mark.parametrize("label, route, which, index_of, value", NONPOSITIVE_FIELDS,
                         ids=[c[0] for c in NONPOSITIVE_FIELDS])
def test_drill_a_nonpositive_lifetime_or_rate_is_refused_at_the_read(  # type: ignore[no-untyped-def]
        tmp_path: Path, c12: bound.Pair, label: str, route: str, which, index_of, value: str) -> None:
    tag = "S" if route == bound.STOPPING else "A"
    s = _summary(tmp_path, route, c12, n=400, mutate=_first_line(tag, which, _set(index_of, value)))
    assert (s["malformed"], s["events"]) == (1, 399), label


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_clock_tally_refuses_a_nonfinite_time(value: float) -> None:
    for args in ((value, 1.0, 1.0), (1.0, value, 1.0), (1.0, 1.0, value), (1.0, 1.0, 1.0, value)):
        tally = bound.ClockTally()
        with pytest.raises(bound.BoundRateError):
            tally.add(*args)
        assert tally == bound.ClockTally(), args


def test_the_zero_capture_lifetime_is_typed_and_accepted_only_for_a_zero_capture_rate(tmp_path: Path,
                                                                                    c12: bound.Pair) -> None:
    """A registered capture rate of exactly zero: the helper stores 1/0 = +inf as the nuclear-capture lifetime
    and G4MuonicAtomDecay recovers an exact zero. That +inf is read only against a zero registered capture
    rate; against a positive one, and NaN or -inf ever, it is refused."""
    zero = dataclasses.replace(c12, capture=0.0)
    assert bound.pair_lifetimes(0.0, zero.decay) == zero.lifetimes == (1.0 / zero.decay, 1.0 / zero.decay,
                                                                         math.inf)
    assert bound.recovered_rates(zero.lifetimes) == (zero.decay, 0.0)
    tau = 1.0 / zero.decay
    line = (f"A 0 6 12 {(500.0).hex()} {tau.hex()} {tau.hex()} inf {(1.0).hex()} {tau.hex()} {(500.0).hex()} "
            f"D D M 0 0 N 0 E {(500.0 + tau).hex()}").split()
    record = bound.parse_record(line, bound.ATOMIC, zero)
    assert isinstance(record, bound.EventRecord) and record.lifetimes == zero.lifetimes
    with pytest.raises(bound.BoundRateError):
        bound.parse_record(line, bound.ATOMIC, c12)
    for bad in ("nan", "-inf", "0x0p+0", "x"):
        with pytest.raises(bound.BoundRateError):
            bound.parse_record(line[:7] + [bad] + line[8:], bound.ATOMIC, zero)
    assert (bound._rel(0.0, 0.0), bound._rel(2.0 ** -1000, 0.0)) == (0.0, math.inf)
    assert (bound._rel(math.inf, math.inf), bound._rel(1.0, math.inf)) == (0.0, math.inf)
    # end to end: a stream drawn from the zero-capture pair passes completeness and the rates, bit for bit
    s = _summary(tmp_path, bound.ATOMIC, zero, n=400)
    assert bound.gate_complete(s, 400)["verdict"] == "pass", bound.gate_complete(s, 400)
    assert bound.gate_rate(s, bound.ATOMIC)["verdict"] == "pass"
    assert (s["rate_exact"], s["recovered_exact"], s["captures"]) == (400, 400, 0)


#: (label, mutate, the problems the rate gate must name)
HELPER_DEFECTS = [
    ("a refused call for another nucleus", _helper(lambda t: "C 0 helper 8 18 3 x x"),
     ["0 valid helper calls in 1 processes", "1 helper calls with another nucleus"]),
    ("a common call for another nucleus", _helper(lambda t: " ".join(t[:3] + ["8", "16"] + t[5:])),
     ["0 valid helper calls in 1 processes", "1 helper calls with another nucleus"]),
    ("a disabled call", _helper(lambda t: " ".join(t[:5] + ["1", "x", "x"])),
     ["0 valid helper calls in 1 processes", "1 helper calls with another nucleus"]),
    ("an unsupported call", _helper(lambda t: " ".join(t[:5] + ["2", "x", "x"])),
     ["0 valid helper calls in 1 processes", "1 helper calls with another nucleus"]),
    ("a refused call", _helper(lambda t: " ".join(t[:5] + ["3", "x", "x"])),
     ["0 valid helper calls in 1 processes", "1 helper calls with another nucleus"]),
    ("a common call with a NaN capture rate", _helper(lambda t: " ".join(t[:6] + ["nan", t[7]])),
     ["0 valid helper calls in 1 processes"]),
    ("a common call with an infinite decay rate", _helper(lambda t: " ".join(t[:7] + ["inf"])),
     ["0 valid helper calls in 1 processes"]),
    ("a common call with no pair", _helper(lambda t: " ".join(t[:6] + ["x", "x"])),
     ["0 valid helper calls in 1 processes"]),
    ("a second call", lambda ls: ls[:1] + ls,
     ["1 helper calls with another nucleus"]),
]


@pytest.mark.parametrize("label, mutate, problems", HELPER_DEFECTS, ids=[c[0] for c in HELPER_DEFECTS])
def test_drill_each_helper_call_defect_fails_the_wiring(  # type: ignore[no-untyped-def]
        tmp_path: Path, c12: bound.Pair, label: str, mutate, problems: list[str]) -> None:
    """A helper call counts only when it names the chunk's nucleus, returns the common status and a finite
    supported pair, and is the first; every atomic event is then checked against it, and without one every
    atomic event fails that check. The rate gate and the drill wiring family agree."""
    from scipy import stats

    n = 400
    s = _summary(tmp_path, bound.ATOMIC, c12, n=n, mutate=mutate)
    verdict = bound.gate_rate(s, bound.ATOMIC)
    assert verdict["verdict"] == "fail", (label, verdict)
    for problem in problems:
        assert any(p.startswith(problem) for p in verdict["problems"]), (label, problem, verdict)
    if s["helper_calls_valid"] == 0:
        assert s["lifetimes_from_call_fail"] == n, label
        assert "stored lifetimes differ from the helper call's pair, or no valid call" in verdict["problems"]
    meta = {"route": bound.ATOMIC, "events": str(n), "switch_on": "True"}
    assert bound.drill_families(s, meta, c12, stats)["wiring"] == "fail", label


@pytest.mark.parametrize("label", ["N-14", "O-16"])
def test_the_recovered_atomic_rates_meet_the_tolerance_where_they_are_not_bit_identical(
        tmp_path: Path, policy: dict, label: str) -> None:
    """For N-14 and O-16 the stored reciprocal lifetimes are the registered ones bit for bit, and the decay
    rate G4MuonicAtomDecay recovers from them (1 / DIO lifetime) is one rounding away from the registered
    one: not bit-identical, inside the registered relative tolerance."""
    pair = bound.frozen_pair(policy, label)
    back = bound.recovered_rates(pair.lifetimes)
    assert back != (pair.decay, pair.capture)
    worst = max(bound._rel(back[0], pair.decay), bound._rel(back[1], pair.capture))
    assert 0.0 < worst <= bound.RATE_RTOL
    s = _summary(tmp_path, bound.ATOMIC, pair, n=400)
    assert (s["rate_exact"], s["recovered_exact"], s["recovered_fail"]) == (400, 0, 0)
    assert float.fromhex(s["recovered_max_rel"]) == worst
    assert bound.gate_rate(s, bound.ATOMIC)["verdict"] == "pass"


def test_drill_an_atomic_pair_off_fails_the_stored_and_the_recovered_rates(tmp_path: Path,
                                                                         c12: bound.Pair) -> None:
    """An atomic consumer given a capture rate 1e-8 off: the stored lifetimes and the recovered rates both
    fail, each named; a recovered-rate failure alone fails the drill rate family, and a helper call on the
    stopping route fails its wiring family."""
    from scipy import stats

    s = _rate_summary(tmp_path / "a", bound.ATOMIC, c12, {"consumed_scale": 1 + 1e-8})
    assert (s["rate_fail"], s["recovered_fail"]) == (4000, 4000)
    problems = bound.gate_rate(s, bound.ATOMIC)["problems"]
    assert "4000 events outside the rate tolerance" in problems
    assert "4000 events whose recovered rates are outside the rate tolerance" in problems
    good = _summary(tmp_path / "b", bound.ATOMIC, c12, n=400)
    meta = {"route": bound.ATOMIC, "events": "400", "switch_on": "True"}
    assert bound.drill_families(good, meta, c12, stats)["rate"] == "pass"
    assert bound.drill_families({**good, "recovered_fail": 1}, meta, c12, stats)["rate"] == "fail"
    stop = _summary(tmp_path / "c", bound.STOPPING, c12, n=400, mutate=_append(_helper_line))
    stop_meta = {"route": bound.STOPPING, "events": "400", "switch_on": "True"}
    assert bound.drill_families(stop, stop_meta, c12, stats)["wiring"] == "fail"


def test_the_recovered_c12_rates_are_bit_identical(tmp_path: Path, c12: bound.Pair) -> None:
    assert bound.recovered_rates(c12.lifetimes) == (c12.decay, c12.capture)
    s = _summary(tmp_path, bound.ATOMIC, c12, n=400)
    assert (s["recovered_exact"], float.fromhex(s["recovered_max_rel"])) == (400, 0.0)


# --------------------------------------------------------------------------------------------
# one-defect fixtures: each guard of the summarizer and the gates names its own problem
# --------------------------------------------------------------------------------------------


def _first_line(tag: str, which, edit):  # type: ignore[no-untyped-def]
    """A ``mutate`` for :func:`write_stream`: ``edit`` (tokens -> tokens) applied to the first bound line of
    kind ``tag`` that ``which`` (tokens -> bool) selects."""
    def mutate(lines: list[str]) -> list[str]:
        out, done = [], False
        for line in lines:
            t = line.split()
            if not done and t[0] == tag and which(t):
                t, done = edit(t), True
            out.append(" ".join(t))
        assert done, tag
        return out
    return mutate


def _captured(t: list[str]) -> bool:
    return t[t.index("N", t.index("M")) + 1] != "0"


def _decayed(t: list[str]) -> bool:
    return not _captured(t)


def _set(index_of, value: str):  # type: ignore[no-untyped-def]
    def edit(t: list[str]) -> list[str]:
        t = list(t)
        t[index_of(t)] = value
        return t
    return edit


def _drop_one_tau(t: list[str]) -> list[str]:
    m = t.index("M")
    return t[:m + 2] + ["1", t[m + 3]] + t[m + 5:]


def _unobserve_model(t: list[str]) -> list[str]:
    m = t.index("M")
    return t[:m + 1] + ["x"] + t[m + 3:]


def _append(make):  # type: ignore[no-untyped-def]
    return lambda lines: lines + [make(lines)]


def _helper_line(lines: list[str]) -> str:
    t = lines[0].split()
    return f"C {t[1]} helper 6 12 0 0x1p-5 0x1p-1"


def _leftover(t: list[str]) -> list[str]:
    return t + ["0x1p+0"]


def _no_e_section(t: list[str]) -> list[str]:
    return t[:-2]


#: (label, route, mutate) -- records outside the grammar the harness writes, each refused as malformed
GRAMMAR_DEFECTS = [
    ("an outcome that is neither decay nor capture", bound.STOPPING,
     _first_line("S", _every, _set(lambda t: 6, "X"))),
    ("an atomic branch that is neither", bound.ATOMIC, _first_line("A", _every, _set(lambda t: 11, "X"))),
    ("a negative interaction-length counter", bound.ATOMIC,
     _first_line("A", _every, _set(lambda t: 8, "-0x1p+0"))),
    ("a stopping time after that precedes the time before", bound.STOPPING,
     _first_line("S", _every, _set(lambda t: 5, "-0x1p+0"))),
    ("a token left over after the E section", bound.STOPPING, _first_line("S", _every, _leftover)),
    ("no E section", bound.STOPPING, _first_line("S", _every, _no_e_section)),
    ("a provider status the harness never writes", bound.STOPPING,
     _first_line("S", _every, _set(lambda t: 10, "7"))),
    ("a product count that is not an integer", bound.STOPPING,
     _first_line("S", _every, _set(lambda t: t.index("N", t.index("M")) + 1, "1.5"))),
    ("a provider-call line with a token left over", bound.ATOMIC, _helper(lambda t: " ".join(t + ["0"]))),
]


@pytest.mark.parametrize("label, route, mutate", GRAMMAR_DEFECTS, ids=[c[0] for c in GRAMMAR_DEFECTS])
def test_drill_a_record_outside_the_grammar_is_malformed(  # type: ignore[no-untyped-def]
        tmp_path: Path, c12: bound.Pair, label: str, route: str, mutate) -> None:
    s = _summary(tmp_path, route, c12, n=400, mutate=mutate)
    assert s["malformed"] == 1, (label, s["malformed"])
    assert "malformed" in bound.gate_complete(s, 400)["problems"], label


#: (label, route, mutate, the problem the completeness gate must name)
COMPLETENESS_DEFECTS = [
    ("an event outside the chunk's range", bound.STOPPING,
     _append(lambda ls: " ".join(["S", "4005", *ls[-1].split()[2:]])), "out_of_range"),
    ("an unreadable record", bound.STOPPING, _append(lambda ls: "S x"), "malformed"),
    ("a capture that also delivers a decay electron", bound.STOPPING,
     _first_line("S", _captured, _set(lambda t: len(t) - 1, (260.0).hex())), "outcome_both"),
    ("a decay whose branch says capture", bound.STOPPING, _first_line("S", _decayed, _set(lambda t: 6, "N")),
     "branch_outcome_mismatch"),
    ("a capture with no capture-model call", bound.STOPPING,
     _first_line("S", _captured, _set(lambda t: t.index("M") + 1, "0")), "model_missing"),
    ("a capture whose model times and products differ in number", bound.STOPPING,
     _first_line("S", _captured, _drop_one_tau), "model_product_mismatch"),
    ("an event whose capture model was not observed", bound.STOPPING,
     _first_line("S", _decayed, _unobserve_model), "model_unobserved"),
    ("a record of another nuclide", bound.STOPPING, _first_line("S", lambda t: True, _set(lambda t: 3, "13")),
     "branch counts do not add up to the events"),
    ("an atomic record that disagrees with its stops record", bound.ATOMIC,
     _first_line("A", _decayed, _set(lambda t: 11, "N")), "stops_bound_mismatch"),
]


@pytest.mark.parametrize("label, route, mutate, problem", COMPLETENESS_DEFECTS,
                         ids=[c[0] for c in COMPLETENESS_DEFECTS])
def test_drill_each_record_defect_is_named_by_the_completeness_gate(  # type: ignore[no-untyped-def]
        tmp_path: Path, c12: bound.Pair, label: str, route: str, mutate, problem: str) -> None:
    clean = _summary(tmp_path / "clean", route, c12)
    assert bound.gate_complete(clean, 4000)["verdict"] == "pass"
    s = _summary(tmp_path / "defect", route, c12, mutate=mutate)
    verdict = bound.gate_complete(s, 4000)
    assert verdict["verdict"] == "fail" and any(problem in p for p in verdict["problems"]), (label, verdict)


def test_drill_a_stops_record_that_disagrees_with_the_bound_record_is_named(tmp_path: Path,
                                                                            c12: bound.Pair) -> None:
    raw = write_stream(tmp_path / "a", bound.STOPPING, c12, 2000, 0.0)
    lines = (raw / "stops.txt").read_text(encoding="ascii").splitlines()
    t = lines[0].split()
    t[7] = (2.0 ** -30).hex()  # the time before ApplyYourself, as the stops file records it
    lines[0] = " ".join(t)
    (raw / "stops.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    s = bound.merge([bound.summarize_chunk(raw, bound.STOPPING, c12, 0.0, 0, 2000)])
    assert any("stops_bound_mismatch" in p for p in bound.gate_complete(s, 2000)["problems"])


def test_drill_an_extra_stops_record_or_a_missing_chunk_is_named(tmp_path: Path, c12: bound.Pair) -> None:
    raw = write_stream(tmp_path / "a", bound.STOPPING, c12, 2000, 0.0)
    with (raw / "stops.txt").open("a", encoding="ascii") as handle:
        handle.write("2000 6 12 6 12 0x0p+0 0x0p+0 0x0p+0 0x1p+1 0 D 0x1p+1\n")
    s = bound.merge([bound.summarize_chunk(raw, bound.STOPPING, c12, 0.0, 0, 2000)])
    assert "stops lines 2001 != 2000" in bound.gate_complete(s, 2000)["problems"]
    one = bound.summarize_chunk(write_stream(tmp_path / "b", bound.STOPPING, c12, 2000, 0.0), bound.STOPPING,
                                c12, 0.0, 0, 2000)
    assert "events 2000 != 4000" in bound.gate_complete(bound.merge([one]), 4000)["problems"]


#: (label, route, how the stream is made, the problem the rate gate must name)
RATE_DEFECTS = [
    ("a provider call from another caller", bound.STOPPING,
     {"mutate": _append(lambda ls: "C 3 other 6 12 0 0x1p-5 0x1p-1")}, "unexpected caller"),
    ("a helper call on the stopping route", bound.STOPPING, {"mutate": _append(_helper_line)},
     "helper calls on the stopping route"),
    ("the atomic route without its helper call", bound.ATOMIC,
     {"mutate": lambda ls: [x for x in ls if not x.startswith("C ")]}, "0 valid helper calls in 1 processes"),
    ("atoms whose lifetimes are not the helper call's pair", bound.ATOMIC,
     {"mutate": lambda ls: [_bump_helper_capture(x) if x.startswith("C ") else x for x in ls]},
     "stored lifetimes differ from the helper call's pair"),
    ("a stopping pair 1e-8 off", bound.STOPPING, {"consumed_scale": 1 + 1e-8}, "outside the rate tolerance"),
    ("an atomic pair 1e-8 off", bound.ATOMIC, {"consumed_scale": 1 + 1e-8}, "outside the rate tolerance"),
]


def _bump_helper_capture(line: str) -> str:
    t = line.split()
    t[6] = (float.fromhex(t[6]) * (1 + 1e-12)).hex()
    return " ".join(t)


def _rate_summary(tmp_path: Path, route: str, pair: bound.Pair, how: dict) -> dict:
    scale = how.get("consumed_scale")
    consumed = (pair.capture * scale, pair.decay) if scale is not None else None
    return _summary(tmp_path, route, pair, mutate=how.get("mutate"), consumed=consumed)


@pytest.mark.parametrize("label, route, how, problem", RATE_DEFECTS, ids=[c[0] for c in RATE_DEFECTS])
def test_drill_each_rate_or_wiring_defect_is_named_by_the_rate_gate(
        tmp_path: Path, c12: bound.Pair, label: str, route: str, how: dict, problem: str) -> None:
    verdict = bound.gate_rate(_rate_summary(tmp_path, route, c12, how), route)
    assert verdict["verdict"] == "fail" and any(problem in p for p in verdict["problems"]), (label, verdict)


@pytest.mark.parametrize("route", bound.ROUTES)
def test_a_pair_inside_the_rate_tolerance_passes(tmp_path: Path, c12: bound.Pair, route: str) -> None:
    s = _rate_summary(tmp_path, route, c12, {"consumed_scale": 1 + 1e-10})
    assert bound.gate_rate(s, route)["verdict"] == "pass"
    assert int(s["rate_exact"]) == 0


def test_drill_an_empty_stream_or_a_bin_tie_never_passes_the_distribution(tmp_path: Path,
                                                                          c12: bound.Pair) -> None:
    s = _summary(tmp_path, bound.STOPPING, c12)
    empty = {**s, "bins": {**s["bins"], "capture": [0] * bound.TIME_BINS}}
    out = bound.gate_distribution(empty, 36)
    assert out["capture"] == {"verdict": "fail", "reason": "empty stream"}
    assert {v["verdict"] for v in bound.gate_distribution({**s, "bin_ties": 1}, 36).values()} == {"refuse"}


def test_drill_a_clock_category_with_no_item_fails_delivery(tmp_path: Path, c12: bound.Pair) -> None:
    s = _summary(tmp_path, bound.STOPPING, c12)
    none = {**s, "clocks": {**s["clocks"], "stop_capture": bound.ClockTally().as_dict()}}
    clocks = bound.gate_clocks({250: none}, bound.STOPPING, (250,))
    assert "t0=250 stop_capture: no item" in clocks["delivery"]["problems"]
    assert clocks["physical"]["stop_capture@250"]["verdict"] == "no_item"

# --------------------------------------------------------------------------------------------
# the results generator, end to end on a synthetic harvest of every group at a reduced count
# --------------------------------------------------------------------------------------------

#: Events per stratum of the synthetic harvest; the frozen limits are recomputed for it the way the policy
#: step computes them, so that only the count differs from the registered plan.
SYNTHETIC_STRATUM = 3000


def _generator():  # type: ignore[no-untyped-def]
    path = ROOT / "scripts/generate_g4boundrates.py"
    spec = importlib.util.spec_from_file_location("generate_g4boundrates_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_record(group: dict, first: int, count: int, t0: int, cell: Path) -> dict:
    """The run record ``cpp/tools/mixture_harvest.py run`` writes for a production chunk of ``group``."""
    argv = [f"/apps/builds/{group['revision']}-bound-trace/harvest_mixture", "--route", group["route"],
            "--symbol", group["isotope"].split("-")[0], "--Z", str(group["z"]), "--events", str(count),
            "--seed-base", str(group["seed_base"]), "--opt-in", "off", "--hooks", "on", "--identity", "off",
            "--out", str(cell / "raw"), "--isotope", str(group["a"])]
    argv += ["--t0", str(t0)] if t0 else []
    argv += ["--event-offset", str(first)] if first else []
    argv += ["--bound-rates", "on", "--bound-record", "on"]
    return {"argv": argv, "build_binary_sha256": "synthetic", "files": {}, "rc": 0}


def _synthetic_harvest(tmp_path: Path, policy: dict, *, skip_group: int | None = None,
                       registered_counts: bool = False, edit_run=None) -> Path:  # type: ignore[no-untyped-def]
    """A repository root holding the theory status, a policy whose groups have ``SYNTHETIC_STRATUM`` events
    per stratum (unless ``registered_counts``) and the harvest summary of correct synthetic chunks of every
    group but ``skip_group``, each with the run record the harness writes (``edit_run`` changes the first
    chunk's)."""
    from scipy import stats

    root = tmp_path / "root"
    (root / "data/g4/d1").mkdir(parents=True)
    (root / bound.THEORY_RELPATH).write_bytes((ROOT / bound.THEORY_RELPATH).read_bytes())
    digests: dict[tuple[str, int], dict[str, str]] = {}
    small = json.loads(json.dumps(policy))
    n = 3 * SYNTHETIC_STRATUM
    if not registered_counts:
        alpha = float(small["statistics"]["primary"]["per_test_alpha"])
        small["chunk_events"] = SYNTHETIC_STRATUM
        for g in small["groups"]:
            g["events"] = n
            for st in g["strata"]:
                st["count"] = SYNTHETIC_STRATUM
        for label, limits in small["statistics"]["primary"]["limits"].items():
            p = float(bound.frozen_pair(small, label).p_capture)
            limits["capture_count"] = [int(stats.binom.ppf(alpha / 2, n, p)),
                                       int(stats.binom.isf(alpha / 2, n, p))]
            limits["sum_all"] = [repr(float(stats.gamma.ppf(alpha / 2, n))),
                                 repr(float(stats.gamma.isf(alpha / 2, n)))]
    (root / bound.POLICY_RELPATH).write_bytes(bound.render_policy(small))
    chunks = []
    for g in small["groups"]:
        if g["group"] == skip_group:
            continue
        pair = bound.frozen_pair(small, g["isotope"])
        for k, st in enumerate(g["strata"]):
            cell = tmp_path / "cells" / f"g{g['group']:02d}_{k}"
            write_stream(cell, g["route"], pair, SYNTHETIC_STRATUM, float(st["t0_ns"]),
                         seed=100 * g["group"] + k, first=st["first"])
            run = _run_record(g, st["first"], SYNTHETIC_STRATUM, st["t0_ns"], cell)
            run["files"] = _raw_files(cell)
            digests[(bound.group_name(g), int(st["first"]))] = dict(run["files"])
            if edit_run is not None and not chunks:
                run = edit_run(run)
            (cell / "run.json").write_text(json.dumps(run), encoding="utf-8")
            chunks.append({"group": g["group"], "first": st["first"], "count": SYNTHETIC_STRATUM,
                           "t0_ns": st["t0_ns"], "cell": str(cell)})
    (root / bound.RAW_DIGESTS_RELPATH).write_bytes(
        bound.render_csv(bound.raw_digest_rows(digests), bound.RAW_DIGEST_COLUMNS))
    plan = {"switch_on": True, "chunks": chunks}
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    bound.summarize_plan(root, plan, root / "data/g4/d1")
    return root


def _raw_files(cell: Path) -> dict[str, str]:
    """The digests the harness records for a chunk: the SHA-256 of each raw file's uncompressed bytes."""
    return {(x.name[:-3] if x.name.endswith(".gz") else x.name): bound.file_digest(x)
            for x in sorted((cell / "raw").iterdir())}


def test_a_correct_synthetic_harvest_gives_four_separate_statuses_and_only_the_predicted_clock_failures(
        tmp_path: Path, policy: dict) -> None:
    results = _generator().build_results(_synthetic_harvest(tmp_path, policy))
    assert results["RATE_RESULTS"]["status"] == "PASS", results["RATE_RESULTS"]
    assert results["TRANSPORT_RESULTS"]["status"] == "PASS"
    assert results["TRANSPORT_RESULTS"]["primary_rejections"] == []
    assert results["TRANSPORT_RESULTS"]["distribution_rejections"] == []
    clocks = results["CLOCK_RESULTS"]
    assert clocks["delivery_status"] == "PASS"
    assert clocks["not_as_predicted"] == []
    # the frozen predictions make the physical status FAIL however correct the delivery is
    assert clocks["physical_status"] == "FAIL" and clocks["physical_failures"]
    expected_fail = {(cat, t0) for cat, by_t0 in bound.PREDICTED_PHYSICAL.items() for t0, v in by_t0.items()
                     if v == "fail"}
    assert {tuple(f.rsplit(":", 1)[1].split("@")) for f in clocks["physical_failures"]} == {
        (cat, str(t0)) for cat, t0 in expected_fail}
    assert results["THEORY_STATUS"]["status"] == "NOT_ESTABLISHED"
    assert results["THEORY_STATUS"]["screen"] == "SCREEN_VACUOUS"


def test_drill_a_missing_group_or_the_registered_count_unmet_fails_rates_and_transport(
        tmp_path: Path, policy: dict) -> None:
    gen = _generator()
    missing = gen.build_results(_synthetic_harvest(tmp_path / "a", policy, skip_group=5))
    name = bound.group_name(policy["groups"][5])
    for family in ("RATE_RESULTS", "TRANSPORT_RESULTS"):
        assert missing[family]["status"] == "INCOMPLETE", family
        assert missing[family]["roster_problems"] == [f"{name}: missing"], family
    assert missing["CLOCK_RESULTS"]["delivery_status"] == "INCOMPLETE"
    assert missing["CLOCK_RESULTS"]["physical_status"] == "INCOMPLETE"
    assert missing["RATE_RESULTS"]["groups"][name]["complete"]["problems"] == [
        "the summary has no complete cell for this group"]
    assert (missing["RATE_RESULTS"]["groups"][name]["complete"]["verdict"],
            missing["RATE_RESULTS"]["groups"][name]["rate"]["verdict"]) == ("fail", "fail")
    rejected = missing["TRANSPORT_RESULTS"]["primary_rejections"]
    assert {r.split(":", 3)[3] for r in rejected} == set(bound.PRIMARY_TESTS)
    assert {r.rsplit(":", 1)[0] for r in rejected} == {name}
    physical = missing["CLOCK_RESULTS"]["groups"][name]["physical"]
    assert all(v["verdict"] == "missing" for v in physical.values())
    # the same correct chunks graded against the registered counts: never complete, never inside the limits
    short = gen.build_results(_synthetic_harvest(tmp_path / "b", policy, registered_counts=True))
    assert short["RATE_RESULTS"]["status"] == "INCOMPLETE"
    for group in policy["groups"]:
        name_ = bound.group_name(group)
        problems = short["RATE_RESULTS"]["groups"][name_]["complete"]["problems"]
        assert f"events {3 * SYNTHETIC_STRATUM} != {group['events']}" in problems
        assert f"{name_}: {3 * SYNTHETIC_STRATUM} events, {group['events']} registered" in \
            short["RATE_RESULTS"]["roster_problems"]
    assert short["TRANSPORT_RESULTS"]["status"] == "INCOMPLETE"


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory, policy: dict) -> Path:
    return _synthetic_harvest(tmp_path_factory.mktemp("synthetic"), policy)


def _edited_results(root: Path, tmp_path: Path, edit) -> dict:  # type: ignore[no-untyped-def]
    """The results of a copy of ``root`` whose harvest rows ``edit`` (rows -> rows) has changed."""
    copy = tmp_path / "root"
    shutil.copytree(root, copy)
    path = copy / bound.HARVEST_RELPATH
    rows = edit(bound.read_csv(path, bound.HARVEST_COLUMNS))
    path.write_bytes(bound.render_csv(rows, bound.HARVEST_COLUMNS))
    return _generator().build_results(copy)


def _drop(cell: str, stat_prefix: str, key_prefix: str = ""):  # type: ignore[no-untyped-def]
    return lambda rows: [r for r in rows if not (r["cell"] == cell and r["stat"].startswith(stat_prefix)
                                                  and r["key"].startswith(key_prefix))]


def _set_row(cell: str, stat: str, key: str, value: str):  # type: ignore[no-untyped-def]
    def edit(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        hits = [r for r in rows if (r["cell"], r["stat"], r["key"]) == (cell, stat, key)]
        assert len(hits) == 1, (cell, stat, key)
        hits[0]["value"] = value
        return rows
    return edit


def test_drill_a_missing_stratum_is_named_and_incomplete(synthetic_root: Path, tmp_path: Path,
                                                         policy: dict) -> None:
    name = bound.group_name(policy["groups"][0])
    out = _edited_results(synthetic_root, tmp_path, _drop(name, "stratum_250"))
    for family in ("RATE_RESULTS", "TRANSPORT_RESULTS", "CLOCK_RESULTS"):
        assert out[family]["roster_problems"] == [f"{name}: stratum_250 missing"], family
    assert out["RATE_RESULTS"]["status"] == out["TRANSPORT_RESULTS"]["status"] == "INCOMPLETE"
    assert out["CLOCK_RESULTS"]["delivery_status"] == out["CLOCK_RESULTS"]["physical_status"] == "INCOMPLETE"
    stratum = out["RATE_RESULTS"]["groups"][name]["strata"][250]
    assert stratum["complete"]["problems"] == ["stratum 250 missing"]
    assert (stratum["complete"]["verdict"], stratum["rate"]["verdict"]) == ("fail", "fail")
    physical = out["CLOCK_RESULTS"]["groups"][name]["physical"]
    missing = {k for k, v in physical.items() if v["verdict"] == "missing"}
    assert missing == {"stop_dio@250", "stop_capture@250"}


def test_drill_every_stratum_missing_is_named_and_incomplete(synthetic_root: Path, tmp_path: Path,
                                                            policy: dict) -> None:
    name = bound.group_name(policy["groups"][1])
    out = _edited_results(synthetic_root, tmp_path, _drop(name, "stratum_"))
    assert set(out["RATE_RESULTS"]["roster_problems"]) == {f"{name}: stratum_{t0} missing"
                                                            for t0 in bound.T0_NS}
    assert out["RATE_RESULTS"]["status"] == "INCOMPLETE"
    assert out["CLOCK_RESULTS"]["physical_status"] == "INCOMPLETE"
    assert all(v["verdict"] == "missing" for v in out["CLOCK_RESULTS"]["groups"][name]["physical"].values())


def test_drill_a_bad_stratum_under_an_unchanged_group_fails_its_family(synthetic_root: Path, tmp_path: Path,
                                                                      policy: dict) -> None:
    """A stratum whose rates fail while the group's merged statistics are untouched: the rate family fails on
    the stratum; a stratum whose event count is not the registered one is named and incomplete."""
    name = bound.group_name(policy["groups"][2])
    out = _edited_results(synthetic_root, tmp_path / "a", _set_row(name, "stratum_1000", "rate_fail", "1"))
    assert out["RATE_RESULTS"]["roster_problems"] == []
    assert out["RATE_RESULTS"]["status"] == "FAIL"
    assert out["RATE_RESULTS"]["groups"][name]["rate"]["verdict"] == "pass"
    assert out["RATE_RESULTS"]["groups"][name]["strata"][1000]["rate"]["verdict"] == "fail"
    short = _edited_results(synthetic_root, tmp_path / "b",
                            _set_row(name, "stratum_0", "events", str(SYNTHETIC_STRATUM - 1)))
    assert short["RATE_RESULTS"]["roster_problems"] == [
        f"{name}: stratum 0 has {SYNTHETIC_STRATUM - 1} events, {SYNTHETIC_STRATUM} registered"]
    assert short["RATE_RESULTS"]["status"] == "INCOMPLETE"
    extra = _edited_results(synthetic_root, tmp_path / "c", _set_row(name, "group", "chunks", "4"))
    assert extra["RATE_RESULTS"]["roster_problems"] == [f"{name}: 4 chunks, 3 registered"]
    # a defect in the group's merged statistics alone, its strata untouched: the rate family fails on it
    group_only = _edited_results(synthetic_root, tmp_path / "d", _set_row(name, "group", "malformed", "1"))
    assert group_only["RATE_RESULTS"]["roster_problems"] == []
    assert group_only["RATE_RESULTS"]["groups"][name]["complete"]["verdict"] == "fail"
    strata = group_only["RATE_RESULTS"]["groups"][name]["strata"]
    assert all(x["complete"]["verdict"] == "pass" for x in strata.values())
    assert group_only["RATE_RESULTS"]["status"] == "FAIL"


def test_drill_a_stratum_outside_the_registered_ones_is_named(synthetic_root: Path, tmp_path: Path,
                                                              policy: dict) -> None:
    name = bound.group_name(policy["groups"][0])

    def move(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        for r in rows:
            if r["cell"] == name and r["stat"] == "stratum_250":
                r["stat"] = "stratum_500"
        return rows

    out = _edited_results(synthetic_root, tmp_path, move)
    assert out["RATE_RESULTS"]["roster_problems"] == [f"{name}: stratum_250 missing",
                                                      f"{name}: stratum_500 is not registered"]
    assert out["RATE_RESULTS"]["status"] == "INCOMPLETE"
    assert "t0=500: not a registered stratum" in out["CLOCK_RESULTS"]["groups"][name]["delivery"]["problems"]


def test_drill_a_cell_outside_the_roster_or_a_missing_clock_category_is_named(synthetic_root: Path,
                                                                             tmp_path: Path,
                                                                             policy: dict) -> None:
    name = bound.group_name(policy["groups"][0])

    def extra(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return rows + [{**r, "cell": "X-99:v0:stopping"} for r in rows if r["cell"] == name]

    out = _edited_results(synthetic_root, tmp_path / "a", extra)
    assert out["RATE_RESULTS"]["roster_problems"] == ["X-99:v0:stopping: not a registered group"]
    assert out["RATE_RESULTS"]["status"] == "INCOMPLETE"
    gone = _edited_results(synthetic_root, tmp_path / "b", _drop(name, "stratum_0", "clock.stop_dio."))
    clocks = gone["CLOCK_RESULTS"]
    assert "t0=0 stop_dio: missing" in clocks["groups"][name]["delivery"]["problems"]
    assert clocks["delivery_status"] == "FAIL" and clocks["physical_status"] == "INCOMPLETE"
    half = _edited_results(synthetic_root, tmp_path / "c", _drop(name, "stratum_0", "clock.stop_dio.n"))
    assert "t0=0 stop_dio: missing" in half["CLOCK_RESULTS"]["groups"][name]["delivery"]["problems"]


def test_drill_an_empty_roster_or_a_duplicated_row_never_passes(synthetic_root: Path, policy: dict) -> None:
    harvest = bound.load_harvest(synthetic_root)
    problems = bound.roster_problems({**policy, "groups": []}, harvest)
    assert problems[0] == "the policy registers no group"
    assert set(problems[1:]) == {f"{name}: not a registered group" for name in harvest}
    assert bound.roster_problems(policy, {}) == [f"{bound.group_name(g)}: missing" for g in policy["groups"]]
    twice = bound.roster_problems({**policy, "groups": policy["groups"] + policy["groups"][:1]}, harvest)
    assert f"{bound.group_name(policy['groups'][0])}: registered more than once" in twice
    for route in bound.ROUTES:
        none = bound.gate_clocks({}, route, ())
        assert none["delivery"]["problems"] == ["no registered stratum"]
        assert none["physical_verdict"] == "incomplete"
        stray = bound.gate_clocks({250: harvest[bound.group_name(policy["groups"][0])]["stratum_250"]},
                                  route, (0,))
        assert "t0=250: not a registered stratum" in stray["delivery"]["problems"]


def test_drill_a_duplicated_summary_row_is_refused(synthetic_root: Path, tmp_path: Path) -> None:
    with pytest.raises(bound.BoundRateError, match="appears twice"):
        _edited_results(synthetic_root, tmp_path, lambda rows: rows + rows[-1:])


def test_drill_a_distribution_only_failure_fails_transport_with_no_primary_rejection(
        synthetic_root: Path, tmp_path: Path, policy: dict) -> None:
    """Every delay of one group moved into the first bin, its sums and branch counts untouched: the four
    registered tests pass, the distribution family fails, and the transport family fails on it alone."""
    name = bound.group_name(policy["groups"][3])

    def pile(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        cells = [r for r in rows
                 if (r["cell"], r["stat"]) == (name, "group") and r["key"].startswith("bin.all.")]
        total = sum(int(r["value"]) for r in cells)
        for r in cells:
            r["value"] = str(total if r["key"] == "bin.all.0" else 0)
        return rows

    out = _edited_results(synthetic_root, tmp_path, pile)
    transport = out["TRANSPORT_RESULTS"]
    assert transport["status"] == "FAIL"
    assert transport["primary_rejections"] == []
    assert transport["distribution_rejections"] == [f"{name}:all"]
    assert out["RATE_RESULTS"]["status"] == "PASS"


def test_drill_an_unexpected_clock_outcome_is_listed_exactly(synthetic_root: Path, tmp_path: Path,
                                                             policy: dict) -> None:
    """A category predicted to fail its physical check observed to pass: the outcome is listed, alone, as not
    as predicted, and the delivered clock still passes."""
    group = next(g for g in policy["groups"] if g["route"] == bound.STOPPING)
    name = bound.group_name(group)
    assert bound.PREDICTED_PHYSICAL["stop_capture"][250] == "fail"
    out = _edited_results(synthetic_root, tmp_path,
                          _set_row(name, "stratum_250", "clock.stop_capture.fail_physical", "0"))
    clocks = out["CLOCK_RESULTS"]
    assert clocks["not_as_predicted"] == [f"{name}:stop_capture@250"]
    assert f"{name}:stop_capture@250" not in clocks["physical_failures"]
    assert clocks["delivery_status"] == "PASS"


def _argv_set(flag: str, value: str):  # type: ignore[no-untyped-def]
    def edit(run: dict) -> dict:
        argv = list(run["argv"])
        argv[argv.index(flag) + 1] = value
        return {**run, "argv": argv}
    return edit


def _argv_drop(flag: str):  # type: ignore[no-untyped-def]
    def edit(run: dict) -> dict:
        argv = list(run["argv"])
        i = argv.index(flag)
        return {**run, "argv": argv[:i] + argv[i + 2:]}
    return edit


#: (label, how the run record of a registered chunk is changed, the problem named)
PROVENANCE_DEFECTS = [
    ("another seed base", _argv_set("--seed-base", "1"), "--seed-base '1'"),
    ("another start time", _argv_set("--t0", "1000"), "--t0 '1000'"),
    ("no event offset", _argv_drop("--event-offset"), "--event-offset None"),
    ("another element", _argv_set("--symbol", "O"), "--symbol 'O'"),
    ("another Z", _argv_set("--Z", "8"), "--Z '8'"),
    ("another mass number", _argv_set("--isotope", "13"), "--isotope '13'"),
    ("another event count", _argv_set("--events", "2999"), "--events '2999'"),
    ("the switch off", _argv_set("--bound-rates", "off"), "--bound-rates 'off'"),
    ("the dataset opt-in on", _argv_set("--opt-in", "on"), "--opt-in 'on'"),
    ("another route", _argv_set("--route", "stopping"), "--route 'stopping'"),
    ("an option the policy does not register", lambda r: {**r, "argv": r["argv"] + ["--drill-a", "13"]},
     "options the policy does not register: --drill-a"),
    ("an option given twice", lambda r: {**r, "argv": r["argv"] + ["--hooks", "on"]},
     "an option given twice"),
    ("another chunk's output", _argv_set("--out", "/elsewhere/raw"), "is not the chunk's own"),
    ("another revision's binary",
     lambda r: {**r, "argv": ["/apps/builds/v0-bound-trace/harvest_mixture"] + r["argv"][1:]},
     "is not the v11.5.0.beta traced bound-rate build"),
    ("a failed run", lambda r: {**r, "rc": 1}, "exit code 1"),
    ("no command line", lambda r: {k: v for k, v in r.items() if k != "argv"}, "no readable command line"),
]


@pytest.mark.parametrize("label, change, problem", PROVENANCE_DEFECTS, ids=[c[0] for c in PROVENANCE_DEFECTS])
def test_drill_a_run_record_that_disagrees_with_the_policy_is_named(  # type: ignore[no-untyped-def]
        tmp_path: Path, policy: dict, label: str, change, problem: str) -> None:
    """A registered chunk (group 3, the second stratum's second chunk) whose run record departs from the
    policy in one respect: the disagreement is named; the record as the harness writes it has none."""
    group = policy["groups"][3]
    reg = bound.chunks(group)[len(bound.chunks(group)) // 3 + 1]
    chunk = {"group": 3, "first": reg["first"], "count": reg["count"], "t0_ns": reg["t0_ns"],
             "cell": str(tmp_path / "c")}
    good = _run_record(group, reg["first"], reg["count"], reg["t0_ns"], tmp_path / "c")
    assert reg["t0_ns"] and reg["first"]
    assert bound.run_provenance_problems(policy, group, chunk, good) == []
    problems = bound.run_provenance_problems(policy, group, chunk, change(good))
    assert any(problem in p for p in problems), (label, problems)


def test_drill_a_plan_chunk_the_policy_does_not_register_is_named(tmp_path: Path, policy: dict) -> None:
    group = policy["groups"][0]
    reg = bound.chunks(group)[1]
    good = _run_record(group, reg["first"], reg["count"], reg["t0_ns"], tmp_path / "c")
    for chunk in ({**reg, "first": reg["first"] + 1}, {**reg, "count": reg["count"] - 1},
                  {**reg, "t0_ns": 1000}):
        problems = bound.run_provenance_problems(policy, group, {**chunk, "cell": str(tmp_path / "c")}, good)
        assert problems == [f"chunk at {chunk['first']}: not a chunk the policy registers"], (chunk, problems)


def test_drill_a_disagreeing_run_record_makes_the_harvest_incomplete(tmp_path: Path, policy: dict) -> None:
    root = _synthetic_harvest(tmp_path, policy, edit_run=_argv_set("--seed-base", "1"))
    out = _generator().build_results(root)
    name = bound.group_name(policy["groups"][0])
    seed = policy["groups"][0]["seed_base"]
    assert out["RATE_RESULTS"]["roster_problems"] == [
        f"{name}: 1 run records disagree with the policy (chunk at 0: --seed-base '1', registered '{seed}')"]
    for family in ("RATE_RESULTS", "TRANSPORT_RESULTS"):
        assert out[family]["status"] == "INCOMPLETE", family
    harvest = bound.load_harvest(root)
    harvest[name]["meta"].pop("provenance_problems")
    assert f"{name}: no provenance check of its run records" in bound.roster_problems(
        bound.load_policy(root), harvest)


def _resummarize_group0(synthetic_root: Path, tmp_path: Path, alter) -> tuple[dict, list[str]]:  # type: ignore[no-untyped-def]
    """Group 0 of the synthetic harvest re-summarized from copies of its three chunks after ``alter(cells)``
    (the copied chunk directories) has changed them; the committed digests are the synthetic harvest's.
    Returns the group's meta rows and its named roster problems."""
    plan = json.loads((synthetic_root.parent / "plan.json").read_text(encoding="utf-8"))
    root = tmp_path / "root"
    shutil.copytree(synthetic_root, root)
    chunks = []
    for c in (c for c in plan["chunks"] if c["group"] == 0):
        cell = tmp_path / "cells" / Path(c["cell"]).name
        shutil.copytree(c["cell"], cell)
        run = json.loads((cell / "run.json").read_text(encoding="utf-8"))
        argv = list(run["argv"])
        argv[argv.index("--out") + 1] = str(cell / "raw")
        (cell / "run.json").write_text(json.dumps({**run, "argv": argv}), encoding="utf-8")
        chunks.append({**c, "cell": str(cell)})
    alter([Path(c["cell"]) for c in chunks])
    out = tmp_path / "out"
    bound.summarize_plan(root, {"switch_on": True, "chunks": chunks}, out)
    (root / bound.HARVEST_RELPATH).write_bytes((out / Path(bound.HARVEST_RELPATH).name).read_bytes())
    harvest = bound.load_harvest(root)
    name = bound.group_name(bound.load_policy(root)["groups"][0])
    problems = [x for x in bound.roster_problems(bound.load_policy(root), harvest)
                if x.startswith(name + ":")]
    return harvest[name]["meta"], problems


def _append_blank_line(cell: Path) -> None:
    """A raw bound record changed in a way the statistics cannot see: a blank line appended."""
    path = cell / "raw" / "bound.txt"
    path.write_bytes(path.read_bytes() + b"\n")


def _record(cell: Path, edit) -> None:  # type: ignore[no-untyped-def]
    run = json.loads((cell / "run.json").read_text(encoding="utf-8"))
    (cell / "run.json").write_text(json.dumps(edit(run)), encoding="utf-8")


def test_the_untouched_chunks_rehash_clean(synthetic_root: Path, tmp_path: Path) -> None:
    meta, problems = _resummarize_group0(synthetic_root, tmp_path, lambda cells: None)
    assert (meta["raw_digest_problems"], meta["provenance_problems"], problems) == ("0", "0", [])


def test_drill_a_raw_file_swapped_under_an_untouched_run_record_is_named(synthetic_root: Path,
                                                                         tmp_path: Path) -> None:
    """A blank line appended to one chunk's bound record after the run: the statistics are those of the
    original, the re-hashed file differs from the run's record and from the committed digest."""
    meta, problems = _resummarize_group0(synthetic_root, tmp_path, lambda cells: _append_blank_line(cells[1]))
    first = json.loads((synthetic_root.parent / "plan.json").read_text())["chunks"][1]["first"]
    assert meta["raw_digest_problems"] == "2"
    assert meta["raw_digest_first"] == f"chunk at {first}: bound.txt differs from its run record"
    name = bound.group_name(bound.load_policy(synthetic_root)["groups"][0])
    assert problems == [f"{name}: 2 raw-file digests disagree (chunk at {first}: bound.txt differs from "
                        "its run record)"]


def test_drill_a_run_record_edited_to_match_a_swapped_file_still_fails(synthetic_root: Path,
                                                                       tmp_path: Path) -> None:
    """The swapped file of the drill above with its run record's digest edited to the swapped bytes: the
    record agrees, and the committed digest -- the identity registered for the chunk -- does not."""
    def alter(cells: list[Path]) -> None:
        _append_blank_line(cells[1])
        digest = bound.file_digest(cells[1] / "raw" / "bound.txt")
        _record(cells[1], lambda run: {**run, "files": {**run["files"], "bound.txt": digest}})

    meta, problems = _resummarize_group0(synthetic_root, tmp_path, alter)
    assert meta["raw_digest_problems"] == "1"
    assert meta["raw_digest_first"].endswith(": bound.txt differs from its committed digest")
    assert len(problems) == 1 and "1 raw-file digests disagree" in problems[0]


def test_drill_a_run_record_whose_digest_was_edited_alone_is_named(synthetic_root: Path,
                                                                   tmp_path: Path) -> None:
    meta, _ = _resummarize_group0(synthetic_root, tmp_path, lambda cells: _record(
        cells[0], lambda run: {**run, "files": {**run["files"], "stops.txt": "0" * 64}}))
    assert (meta["raw_digest_problems"], meta["raw_digest_first"]) == (
        "1", "chunk at 0: stops.txt differs from its run record")


def test_drill_a_planted_or_missing_raw_file_is_named(synthetic_root: Path, tmp_path: Path) -> None:
    planted, _ = _resummarize_group0(synthetic_root, tmp_path / "a", lambda cells: (
        cells[2] / "raw" / "records.txt").write_text("x\n", encoding="ascii"))
    first = json.loads((synthetic_root.parent / "plan.json").read_text())["chunks"][2]["first"]
    assert planted["raw_digest_problems"] == "2"
    assert planted["raw_digest_first"] == f"chunk at {first}: records.txt is not in its run record"
    gone, _ = _resummarize_group0(synthetic_root, tmp_path / "b", lambda cells: (
        cells[0] / "raw" / "bound_summary.txt").unlink())
    assert gone["raw_digest_first"] == ("chunk at 0: bound_summary.txt, which the analysis consumes, has no "
                                        "digest: it is not in the raw directory")
    # the run record and the committed digests still name it: it is recorded but not in the raw directory
    assert gone["raw_digest_problems"] == "2"


def test_drill_a_tampered_gzipped_record_beside_an_intact_plain_copy_is_named(synthetic_root: Path,
                                                                             tmp_path: Path) -> None:
    """The reader consumes bound.txt.gz whenever it exists. A gzipped copy with a blank line appended, put
    beside the intact plain bound.txt: the digest is of the bytes the reader consumed, so it differs from the
    run record and from the committed digest; a digest re-taken from the plain copy would agree with both."""
    def alter(cells: list[Path]) -> None:
        raw = cells[1] / "raw"
        (raw / "bound.txt.gz").write_bytes(gzip.compress((raw / "bound.txt").read_bytes() + b"\n", mtime=0))

    meta, problems = _resummarize_group0(synthetic_root, tmp_path, alter)
    first = json.loads((synthetic_root.parent / "plan.json").read_text())["chunks"][1]["first"]
    # the consumed copy differs from the run record and from the committed digest; the name held twice is
    # named
    assert meta["raw_digest_problems"] == "3"
    assert meta["raw_digest_first"] == f"chunk at {first}: bound.txt differs from its run record"
    name = bound.group_name(bound.load_policy(synthetic_root)["groups"][0])
    assert problems == [f"{name}: 3 raw-file digests disagree (chunk at {first}: bound.txt differs from "
                        "its run record)"]


BOTH_FORMS = "is held both plain and gzipped; one copy was hashed and the other was not"


def test_drill_a_tampered_plain_copy_beside_an_intact_consumed_gzipped_record_is_named(synthetic_root: Path,
                                                                                    tmp_path: Path) -> None:
    """The intact record gzipped as the reader consumes it, and a plain copy with a blank line appended beside
    it: the consumed bytes agree with the run record and the committed digest, and the plain copy, which is
    never read or hashed, is named through the name it shares."""
    def alter(cells: list[Path]) -> None:
        raw = cells[1] / "raw"
        body = (raw / "bound.txt").read_bytes()
        (raw / "bound.txt.gz").write_bytes(gzip.compress(body, mtime=0))
        (raw / "bound.txt").write_bytes(body + b"\n")

    meta, problems = _resummarize_group0(synthetic_root, tmp_path, alter)
    first = json.loads((synthetic_root.parent / "plan.json").read_text())["chunks"][1]["first"]
    assert (meta["raw_digest_problems"], meta["raw_digest_first"]) == (
        "1", f"chunk at {first}: bound.txt {BOTH_FORMS}")
    name = bound.group_name(bound.load_policy(synthetic_root)["groups"][0])
    assert problems == [f"{name}: 1 raw-file digests disagree (chunk at {first}: bound.txt {BOTH_FORMS})"]


def test_drill_an_unconsumed_raw_file_held_both_plain_and_gzipped_is_named(tmp_path: Path,
                                                                          c12: bound.Pair) -> None:
    """A raw file the analysis does not consume (a config record, as the harness writes one), recorded in the
    run record and the committed digests, then a changed gzipped copy put beside it: only the plain copy is
    hashed, it agrees with both, and the name held twice is named."""
    job, cell, run = _drill_job(tmp_path, c12, "a", 1900000000)
    (cell / "raw" / "config.txt").write_text("isotope 12\n", encoding="ascii")
    files = _raw_files(cell)
    clean = bound.summarize_chunk(cell / "raw", bound.ATOMIC, c12, 0.0, 0, 400)
    assert bound.raw_digest_problems("run at 0", clean, files, dict(files)) == []
    (cell / "raw" / "config.txt.gz").write_bytes(gzip.compress(b"isotope 13\n", mtime=0))
    stats = bound.summarize_chunk(cell / "raw", bound.ATOMIC, c12, 0.0, 0, 400)
    assert stats["raw_sha256"] == clean["raw_sha256"] and stats["raw_both_forms"] == ["config.txt"]
    assert bound.raw_digest_problems("run at 0", stats, files, dict(files)) == [
        f"run at 0: config.txt {BOTH_FORMS}"]


def test_a_summary_that_does_not_list_the_names_held_twice_is_named(tmp_path: Path, c12: bound.Pair) -> None:
    job, cell, run = _drill_job(tmp_path, c12, "a", 1900000000)
    stats = bound.summarize_chunk(cell / "raw", bound.ATOMIC, c12, 0.0, 0, 400)
    assert stats["raw_both_forms"] == []
    assert bound.raw_digest_problems("run at 0", stats, run["files"], dict(run["files"])) == []
    without = {k: v for k, v in stats.items() if k != "raw_both_forms"}
    for broken in (without, {**stats, "raw_both_forms": None}):
        assert bound.raw_digest_problems("run at 0", broken, run["files"], dict(run["files"])) == [
            "run at 0: the summary does not list the raw names held both plain and gzipped"]


def test_drill_no_committed_digests_is_named(synthetic_root: Path, tmp_path: Path) -> None:
    def alter(cells: list[Path]) -> None:
        (tmp_path / "root" / bound.RAW_DIGESTS_RELPATH).unlink()

    meta, problems = _resummarize_group0(synthetic_root, tmp_path, alter)
    assert meta["raw_digest_first"] == "chunk at 0: bound.txt has no committed digest"
    assert problems and "raw-file digests disagree" in problems[0]
    harvest = bound.load_harvest(synthetic_root)
    name = bound.group_name(bound.load_policy(synthetic_root)["groups"][0])
    harvest[name]["meta"].pop("raw_digest_problems")
    assert f"{name}: no re-hash of its raw files" in bound.roster_problems(bound.load_policy(synthetic_root),
                                                                           harvest)


def _drill_job(tmp_path: Path, c12: bound.Pair, name: str, seed: int) -> tuple[dict, Path, dict]:
    """A synthetic drill run of a registered job: its directory, the job and the run record the harness
    writes for it."""
    chunk = tmp_path / name
    write_stream(chunk, bound.ATOMIC, c12, 400, 0.0, seed=seed)
    job = {"build": "drilltree/v11.4.2/builds/total_only-trace", "route": bound.ATOMIC, "symbol": "C", "z": 6,
           "a": 12, "events": 400, "seed_base": seed, "event_offset": 0, "extra": []}
    argv = [f"/apps/{job['build']}/harvest_mixture", "--route", "atomic", "--symbol", "C", "--Z", "6",
            "--events", "400", "--seed-base", str(seed), "--opt-in", "off", "--hooks", "on",
            "--identity", "off",
            "--out", str(chunk / "raw"), "--isotope", "12", "--bound-rates", "on", "--bound-record", "on"]
    run = {"argv": argv, "rc": 0, "build_binary_sha256": "synthetic", "files": _raw_files(chunk)}
    return job, chunk, run


def test_drill_a_drill_row_from_a_different_chunk_is_named(tmp_path: Path, c12: bound.Pair) -> None:
    """Two registered drill runs A and B. A's job read against its own run is clean. A's directory holding B's
    run record names B's seed base and output; A's directory holding B's raw files under A's untouched record
    differs from A's record and from A's committed digests."""
    job_a, dir_a, run_a = _drill_job(tmp_path, c12, "a", 1900000000)
    _, dir_b, run_b = _drill_job(tmp_path, c12, "b", 1901000000)
    registered_a = dict(run_a["files"])
    stats_a = bound.summarize_chunk(dir_a / "raw", bound.ATOMIC, c12, 0.0, 0, 400)
    assert bound.drill_run_problems(job_a, dir_a, run_a) == []
    assert bound.raw_digest_problems("run at 0", stats_a, run_a["files"], registered_a) == []
    wrong_run = bound.drill_run_problems(job_a, dir_a, run_b)
    assert "run at 0: --seed-base '1901000000', registered '1900000000'" in wrong_run
    assert any("is not the chunk's own" in x for x in wrong_run)
    shutil.rmtree(dir_a / "raw")
    shutil.copytree(dir_b / "raw", dir_a / "raw")
    stats_swapped = bound.summarize_chunk(dir_a / "raw", bound.ATOMIC, c12, 0.0, 0, 400)
    digests = bound.raw_digest_problems("run at 0", stats_swapped, run_a["files"], registered_a)
    for f in bound.CONSUMED_RAW:
        assert f"run at 0: {f} differs from its run record" in digests, f
        assert f"run at 0: {f} differs from its committed digest" in digests, f
    extra = bound.drill_run_problems({**job_a, "extra": ["--synthetic-capture-scale", "4"]}, dir_a, run_a)
    assert extra == ["run at 0: --synthetic-capture-scale None, registered '4'"]


def test_drill_an_empty_branch_fails_its_conditional_test(policy: dict) -> None:
    from scipy import stats

    gen = _generator()
    pair = bound.frozen_pair(policy, "C-12")
    limits = policy["statistics"]["primary"]["limits"]["C-12"]
    alpha = float(policy["statistics"]["primary"]["per_test_alpha"])
    g = {"events": 10, "captures": 0, "sums": {"all": (10 / pair.total).hex(), "capture": (0.0).hex(),
                                                 "decay": (10 / pair.total).hex()}}
    out = gen.primary_tests(stats, g, pair, limits, alpha)
    assert out["sum_capture"]["verdict"] == "fail" and out["sum_capture"]["reason"] == "empty branch"

@pytest.mark.parametrize("label, change, test", [
    ("one capture above the interval", lambda g, lim: {**g, "captures": lim["capture_count"][1] + 1},
     "capture_count"),
    ("all delays 10% long", lambda g, lim: _scale_sum(g, "all", 1.1), "sum_all"),
    ("capture delays 50% long", lambda g, lim: _scale_sum(g, "capture", 1.5), "sum_capture"),
    ("decay delays 10% long", lambda g, lim: _scale_sum(g, "decay", 1.1), "sum_decay"),
])
def test_drill_each_registered_test_rejects_its_statistic_outside_the_interval(  # type: ignore[no-untyped-def]
        tmp_path: Path, policy: dict, c12: bound.Pair, label: str, change, test: str) -> None:
    from scipy import stats

    gen = _generator()
    n = 3 * SYNTHETIC_STRATUM
    alpha = float(policy["statistics"]["primary"]["per_test_alpha"])
    p = float(c12.p_capture)
    limits = {"capture_count": [int(stats.binom.ppf(alpha / 2, n, p)), int(stats.binom.isf(alpha / 2, n, p))],
              "sum_all": [repr(float(stats.gamma.ppf(alpha / 2, n))),
                          repr(float(stats.gamma.isf(alpha / 2, n)))]}
    g = _summary(tmp_path, bound.STOPPING, c12, n=n)
    assert {v["verdict"] for v in gen.primary_tests(stats, g, c12, limits, alpha).values()} == {"pass"}
    out = gen.primary_tests(stats, change(g, limits), c12, limits, alpha)
    assert out[test]["verdict"] == "fail", (label, out[test])


def _scale_sum(g: dict, stream: str, factor: float) -> dict:
    sums = dict(g["sums"])
    sums[stream] = (float.fromhex(sums[stream]) * factor).hex()
    if stream != "all":
        sums["all"] = math.fsum(float.fromhex(sums[k]) for k in ("capture", "decay")).hex()
    return {**g, "sums": sums}

# --------------------------------------------------------------------------------------------
# the theory admission gate
# --------------------------------------------------------------------------------------------


def _complete_manifest() -> dict:
    bounded = {"status": "BOUNDED", "bound": 1e-6, "source": "a stated derivation"}
    terms: dict[str, dict[str, object]] = {name: dict(bounded) for name in bound.ERROR_TERMS}
    terms[bound.EQUATION_TERM] = {"status": "VERIFIED", "statement": "the primary equation, read",
                                  "source": "the primary paper"}
    return {"terms": terms, "qualified_roster": ["6-12"],
            "references": [{"id": "independent width", "independent": True, "exposed": False}]}


def _term(name: str, **fields: object):  # type: ignore[no-untyped-def]
    return lambda m: {**m, "terms": {**m["terms"], name: {**m["terms"][name], **fields}}}


def test_the_committed_theory_status_is_what_the_gate_gives() -> None:
    manifest = json.loads((ROOT / bound.THEORY_RELPATH).read_text(encoding="utf-8"))
    verdict = bound.theory_admission(manifest)
    assert verdict["status"] == manifest["status"] == "NOT_ESTABLISHED"
    assert manifest["screen"] == "SCREEN_VACUOUS" and manifest["qualified_roster"] == []
    for name in bound.REQUIRED_TERMS:
        assert any(p.startswith(f"{name}: ") for p in verdict["problems"]), name


def _without(m: dict, term: str) -> dict:
    return {**m, "terms": {k: v for k, v in m["terms"].items() if k != term}}


@pytest.mark.parametrize("label, change, status", [
    ("complete", lambda m: m, "ADMITTED"),
    ("complete, empty roster", lambda m: {**m, "qualified_roster": []}, "SCREEN_VACUOUS"),
    ("a required term removed", lambda m: _without(m, "radiative"), "NOT_ESTABLISHED"),
    ("an unknown term replaced by zero",
     lambda m: {**m, "terms": {**m["terms"], "recoil": {"status": "UNKNOWN", "bound": 0, "source": None}}},
     "NOT_ESTABLISHED"),
    ("an unbounded term written as a zero bound", _term("recoil", status="UNBOUNDED", bound=0),
     "NOT_ESTABLISHED"),
    ("a NaN bound", _term("finite_size", bound=float("nan")), "NOT_ESTABLISHED"),
    ("a +inf bound", _term("finite_size", bound=float("inf")), "NOT_ESTABLISHED"),
    ("a -inf bound", _term("finite_size", bound=float("-inf")), "NOT_ESTABLISHED"),
    ("a bound written as text", _term("finite_size", bound="not a bound"), "NOT_ESTABLISHED"),
    ("a bound written as a numeric string", _term("finite_size", bound="1e-6"), "NOT_ESTABLISHED"),
    ("a boolean bound", _term("finite_size", bound=True), "NOT_ESTABLISHED"),
    ("a negative bound", _term("finite_size", bound=-1e-6), "NOT_ESTABLISHED"),
    ("an error term with no status", _term("radiative", status=None), "NOT_ESTABLISHED"),
    ("an error term with an empty source", _term("radiative", source=" "), "NOT_ESTABLISHED"),
    ("an unverified equation", _term("equation", status="UNKNOWN"), "NOT_ESTABLISHED"),
    ("an equation with no statement", _term("equation", statement=None), "NOT_ESTABLISHED"),
    ("an unsupported nuclide admitted", lambda m: {**m, "qualified_roster": ["14-28"]}, "NOT_ESTABLISHED"),
    ("a dependent reference",
     lambda m: {**m, "references": [{"id": "same formula", "independent": False, "exposed": False}]},
     "NOT_ESTABLISHED"),
    ("an exposed reference as a holdout",
     lambda m: {**m, "references": [{"id": "residuals", "independent": True, "exposed": True}]},
     "NOT_ESTABLISHED"),
    ("a sourced term with no bound", _term("recoil", bound=None), "NOT_ESTABLISHED"),
    ("a reference that does not state its independence",
     lambda m: {**m, "references": [{"id": "unstated", "exposed": False}]}, "NOT_ESTABLISHED"),
    ("a reference that does not state its exposure",
     lambda m: {**m, "references": [{"id": "unstated", "independent": True}]}, "NOT_ESTABLISHED"),
    ("an admitted nuclide with no reference", lambda m: {**m, "references": []}, "NOT_ESTABLISHED"),
])
def test_drill_the_theory_gate_refuses_each_defect(label: str, change, status: str) -> None:  # type: ignore[no-untyped-def]
    assert bound.theory_admission(change(_complete_manifest()))["status"] == status, label


def test_the_running_sum_is_exactly_rounded() -> None:
    import random

    rng = random.Random(11)
    values = [rng.expovariate(0.0005) * (1e6 if i % 97 == 0 else 1.0) for i in range(20000)]
    values += [1e20, 1.0, -1e20]
    acc = bound.ExactSum()
    for v in values:
        acc.add(v)
    assert acc.value() == math.fsum(values)


# --------------------------------------------------------------------------------------------
# the committed harvest, drills, identity runs, workers, sweep and results
# --------------------------------------------------------------------------------------------

#: The drills the campaign ran, by the defect each plants (library drills, the synthetic-pair witness), the
#: reruns at a size where the transport tests have power, and the matched controls on the drill base
#: library.
DRILL_IDS = {"old_helper_atomic", "disconnect_stopping", "units_1000", "omit_recoil", "neighbour_mass",
             "drop_a", "total_only", "swap_partials", "compensation", "constant_delay",
             "synthetic_capture_x4", "control", "old_helper_atomic_n10m", "total_only_n60m"}
#: What each drill result was ruled to show where the drill plan asked more of it than the run could show; the
#: recorded outcomes stay as they are.
RULINGS_RELPATH = "data/g4/d1/bound_rates_drill_rulings.csv"
RULING_COLUMNS = ("kind", "revision", "route", "item", "ruled_intended", "rerun", "reason")
#: The worker check: the patched build with the switch on and off and the unpatched build, at 1, 4 and again 4
#: threads, on both routes, for C-12 and Al-27.
WORKERS_RELPATH = "data/g4/d1/bound_rates_workers.csv"


@pytest.fixture(scope="module")
def harvest() -> dict:
    return bound.load_harvest(ROOT)


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads((ROOT / bound.RESULTS_RELPATH).read_text(encoding="utf-8"))


def test_the_harvest_covers_every_group_chunk_and_stratum_of_the_policy(policy: dict, harvest: dict) -> None:
    builds = bound.read_csv(ROOT / bound.BUILDS_RELPATH, bound.BUILD_COLUMNS)
    harness = hashlib.sha256((ROOT / "cpp/tools/harvest_mixture.cc").read_bytes()).hexdigest()
    traced = {b["binary_sha256"] for b in builds if b["trace"] == "True" and b["kind"] == "bound"}
    assert {b["source_sha256"] for b in builds} == {harness}
    assert {b["revision"] for b in builds} == set(bound.REVISIONS)
    assert set(harvest) == {bound.group_name(g) for g in policy["groups"]}
    for g in policy["groups"]:
        cell = harvest[bound.group_name(g)]
        assert cell["meta"]["switch_on"] == "True" and cell["meta"]["binary_sha256"] in traced
        assert cell["group"]["events"] == g["events"] and cell["group"]["chunks"] == len(bound.chunks(g))
        assert cell["group"]["chunk_overlap"] == 0
        for stratum in g["strata"]:
            s = cell[f"stratum_{stratum['t0_ns']}"]
            assert s["events"] == stratum["count"], (g["group"], stratum)
        assert sum(cell[f"stratum_{t}"]["events"] for t in bound.T0_NS) == g["events"]


def test_every_group_is_complete_and_its_rates_are_the_registered_ones(policy: dict, harvest: dict) -> None:
    """Every group complete, read by the current analysis. Every stopping event received the registered pair
    bit for bit from one common-status call; every atomic event stored the registered reciprocal lifetimes bit
    for bit, from the one valid helper call of its process, and the rates G4MuonicAtomDecay recovers from them
    are inside the rate tolerance -- bit for bit exactly where binary64 gives the registered pair back."""
    for g in policy["groups"]:
        name = bound.group_name(g)
        cell = harvest[name]
        s = cell["group"]
        assert cell["meta"]["analysis_version"] == str(bound.ANALYSIS_VERSION), name
        assert bound.gate_complete(s, g["events"])["verdict"] == "pass", name
        assert bound.gate_rate(s, g["route"])["verdict"] == "pass", name
        assert s["rate_exact"] == g["events"] and s["malformed"] == 0, name
        if g["route"] == bound.STOPPING:
            assert s["wiring_fail"] == 0 and s["provider_calls_helper"] == 0, name
        else:
            pair = bound.frozen_pair(policy, g["isotope"])
            exact = bound.recovered_rates(pair.lifetimes) == (pair.decay, pair.capture)
            assert s["recovered_fail"] == 0, name
            assert s["recovered_exact"] == (g["events"] if exact else 0), name
            assert (s["helper_calls_valid"], s["helper_calls_invalid"]) == (s["chunks"], 0), name
            assert s["lifetimes_from_call_fail"] == 0, name


def test_the_reanalysis_keeps_every_registered_outcome_of_the_first_summary(results: dict) -> None:
    """The summary recomputed from the same raw records by the current reader: every registered outcome of
    the first summary's results, kept as committed, is unchanged -- statuses, verdicts, statistics,
    intervals, rejections, clock counts and residuals -- and the new results name no roster problem."""
    history = ROOT / "data/g4/d1/history/bound_rates_results.analysis1.json"
    # the first results as they were committed, byte for byte
    assert hashlib.sha256(history.read_bytes()).hexdigest() == (
        "ff3b4df548c9f0e73b6be4118875f5f5e4418bf43cad26c74f074e3600ca0944")
    first = json.loads(history.read_text(encoding="utf-8"))

    def strip(x: object) -> object:
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items()
                    if k not in ("problems", "roster_problems", "strata", "recovered_exact_events",
                                 "recovered_max_relative_deviation", "qualified_roster")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    for family in ("RATE_RESULTS", "TRANSPORT_RESULTS", "CLOCK_RESULTS"):
        assert results[family]["roster_problems"] == [], family
        assert strip(results[family]) == strip(first[family]), family
    assert results["THEORY_STATUS"]["status"] == first["THEORY_STATUS"]["status"]
    # the strata the first results kept under transport are the counts the new ones keep
    for name, g in first["TRANSPORT_RESULTS"]["groups"].items():
        assert g["strata"] == results["TRANSPORT_RESULTS"]["groups"][name]["strata"], name


def test_the_four_registered_tests_recomputed_independently(policy: dict, harvest: dict,
                                                            results: dict) -> None:
    """Each group's four tests from its committed sums with scipy's distributions called here: the capture
    count against the policy's interval, and each scaled sum against its gamma quantiles at half the
    per-test alpha. The verdicts are the committed ones."""
    from scipy import stats

    alpha = float(policy["statistics"]["primary"]["per_test_alpha"])
    for g in policy["groups"]:
        name = bound.group_name(g)
        s = harvest[name]["group"]
        pair = bound.frozen_pair(policy, g["isotope"])
        n, k = s["events"], s["captures"]
        lo, hi = policy["statistics"]["primary"]["limits"][g["isotope"]]["capture_count"]
        mine = {"capture_count": lo <= k <= hi}
        for test, stream, shape in (("sum_all", "all", n), ("sum_capture", "capture", k),
                                    ("sum_decay", "decay", n - k)):
            u = pair.total * float.fromhex(s["sums"][stream])
            mine[test] = bool(stats.gamma.ppf(alpha / 2, shape) <= u <= stats.gamma.isf(alpha / 2, shape))
        committed = results["TRANSPORT_RESULTS"]["groups"][name]["primary"]
        assert {t: v["verdict"] == "pass" for t, v in committed.items()} == mine, name


def test_the_distribution_family_recomputed_independently(policy: dict, harvest: dict, results: dict) -> None:
    from scipy import stats

    dist = policy["statistics"]["distribution"]
    crit = stats.chi2.isf(float(dist["per_test_alpha"]), dist["dof"])
    for g in policy["groups"]:
        name = bound.group_name(g)
        s = harvest[name]["group"]
        for stream in bound.STREAMS:
            counts = s["bins"][stream]
            n = sum(counts)
            x2 = sum((c - n / bound.TIME_BINS) ** 2 / (n / bound.TIME_BINS) for c in counts)
            committed = results["TRANSPORT_RESULTS"]["groups"][name]["distribution"][stream]["verdict"]
            assert (x2 < crit) == (committed == "pass"), (name, stream, x2, crit)


def test_the_clock_results_follow_the_frozen_predictions_and_delivery_passes(results: dict) -> None:
    clocks = results["CLOCK_RESULTS"]
    assert clocks["delivery_status"] == "PASS"
    for name, v in clocks["groups"].items():
        assert v["delivery"]["verdict"] == "pass", name
        for key, r in v["physical"].items():
            assert r["n"] > 0, (name, key)
    assert sorted(clocks["not_as_predicted"]) == sorted(
        f"{name}:{key}" for name, v in clocks["groups"].items() for key, r in v["physical"].items()
        if not r["as_predicted"])


def test_the_results_regenerate_from_the_committed_files(results: dict) -> None:
    fresh = _generator().build_results(ROOT)

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

    compare(json.loads(json.dumps(fresh)), results, "results")


def test_the_theory_family_stays_separate_from_the_engineering_families(results: dict) -> None:
    assert results["THEORY_STATUS"]["status"] == "NOT_ESTABLISHED"
    assert results["THEORY_STATUS"]["screen"] == "SCREEN_VACUOUS"
    assert results["THEORY_STATUS"]["qualified_roster"] == []


def _rulings(kind: str) -> list[dict[str, str]]:
    return [r for r in bound.read_csv(ROOT / RULINGS_RELPATH, RULING_COLUMNS) if r["kind"] == kind]


def _transport_power(policy: dict, n: int, drill: str) -> float:
    """The largest power of the four registered tests at size ``n`` (alpha the per-test alpha, two-sided)
    against the defect ``drill`` plants on the atomic route of C-12, with the policy's frozen alternative
    decay rate: old_helper_atomic gives the consumer (c, d_old), so the delay and the branch move;
    total_only keeps the total lifetime and leaves the DIO lifetime stale, so only the branch moves. The
    conditional tests are evaluated at their expected count."""
    from scipy import stats

    pair = bound.frozen_pair(policy, "C-12")
    c, d = pair.capture, pair.decay
    d_old = float.fromhex(policy["statistics"]["power"]["values"]["C-12"]["alternative_decay"])
    a = float(policy["statistics"]["primary"]["per_test_alpha"])
    p0, p1 = c / (c + d), c / (c + d_old)
    scale = (c + d) / (c + d_old) if drill.startswith("old_helper_atomic") else 1.0
    lo, hi = stats.binom.ppf(a / 2, n, p0), stats.binom.isf(a / 2, n, p0)
    powers = [float(stats.binom.cdf(lo - 1, n, p1) + stats.binom.sf(hi, n, p1))]
    for shape in (n, n * p1, n * (1 - p1)):
        g_lo, g_hi = stats.gamma.ppf(a / 2, shape), stats.gamma.isf(a / 2, shape)
        powers.append(float(stats.gamma.cdf(g_lo, shape, scale=scale)
                            + stats.gamma.sf(g_hi, shape, scale=scale)))
    return max(powers)


def test_every_drill_turns_its_intended_family_red_and_every_control_stays_green(policy: dict) -> None:
    """Each drill red on its intended families and each control green; where a ruling narrows a drill's
    intended families (the registered transport tests unpowered against the defect at the run's size),
    the narrowed set is red, the transport power at that size is below 0.99, and the ruling's rerun is
    powered (some registered test at 0.99 or more) and red on rate, wiring and transport."""
    from scipy import stats

    rows = bound.read_csv(ROOT / bound.DRILLS_RELPATH, bound.HARVEST_COLUMNS)
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(r["cell"], []).append(r)
    ruled = {r["item"]: r for r in _rulings("statistical")}
    ref = bound.Reference(ROOT)
    seen, outcomes = set(), {}
    for cell, cell_rows in grouped.items():
        meta = {r["key"]: r["value"] for r in cell_rows if r["stat"] == "meta"}
        s = bound.rows_stats([r for r in cell_rows if r["stat"] == "group"])
        pair = ref.pair(int(meta["z"]), int(meta["a"]))
        families = bound.drill_families(s, meta, pair, stats)
        intended = ruled[cell]["ruled_intended"] if cell in ruled else meta["intended"]
        outcomes[cell] = bound.drill_outcome(meta["kind"], intended, families)
        expected = "green" if meta["kind"] == "control" or intended == "none" else "red"
        assert outcomes[cell] == expected, (cell, intended, families)
        if cell in ruled:
            assert _transport_power(policy, int(meta["events"]), meta["drill"]) < 0.99, cell
        seen.add(meta["drill"])
    assert seen == DRILL_IDS
    for item, r in ruled.items():
        assert item in grouped and r["rerun"] in grouped, r
        rerun_meta = {x["key"]: x["value"] for x in grouped[r["rerun"]] if x["stat"] == "meta"}
        assert rerun_meta["intended"] == "rate,wiring,transport" and outcomes[r["rerun"]] == "red", r
        assert _transport_power(policy, int(rerun_meta["events"]), rerun_meta["drill"]) >= 0.99, r
    for rev in bound.REVISIONS:
        assert any(c.startswith(f"drill:control:control:{rev}:") for c in grouped), rev


#: The chunk_files_sha256 of every group as the first summary recorded it, from the run records.
CHUNK_FILES_HISTORY = "data/g4/d1/history/bound_rates_chunk_files.analysis1.csv"


def test_the_committed_digests_reproduce_the_first_summary_s_record_of_every_chunk(policy: dict,
                                                                                  harvest: dict) -> None:
    """Every production chunk and drill run has committed digests of its raw files, re-hashed by the analysis.
    For each group, the digest over its chunks' files recomputed from the committed digests equals the
    chunk_files_sha256 the first summary computed from the run records when the harvest ended
    (``history/bound_rates_chunk_files.analysis1.csv``), and the summary's own; every group's re-hash and
    run-record checks found nothing."""
    digests = bound.load_raw_digests(ROOT)
    first = {r["cell"]: r["chunk_files_sha256"]
             for r in bound.read_csv(ROOT / CHUNK_FILES_HISTORY, ("cell", "chunk_files_sha256"))}
    assert set(first) == {bound.group_name(g) for g in policy["groups"]}
    for g in policy["groups"]:
        name = bound.group_name(g)
        text = "".join(f"{f}={sha};" for c in bound.chunks(g)
                       for f, sha in sorted(digests[(name, c["first"])].items()))
        recomputed = hashlib.sha256(text.encode("ascii")).hexdigest()
        assert recomputed == first[name] == harvest[name]["meta"]["chunk_files_sha256"], name
        assert set(digests[(name, bound.chunks(g)[0]["first"])]) >= set(bound.CONSUMED_RAW), name
        meta = harvest[name]["meta"]
        assert (meta["raw_digest_problems"], meta["provenance_problems"]) == ("0", "0"), name
    production = {(bound.group_name(g), c["first"]) for g in policy["groups"] for c in bound.chunks(g)}
    drills = {k for k in digests if k not in production}
    metas: dict[str, dict[str, str]] = {}
    for r in bound.read_csv(ROOT / bound.DRILLS_RELPATH, bound.HARVEST_COLUMNS):
        if r["stat"] == "meta":
            metas.setdefault(r["cell"], {})[r["key"]] = r["value"]
    assert {c for c, _ in drills} == set(metas)
    for cell, m in metas.items():
        assert (m["provenance_problems"], m["raw_digest_problems"]) == ("0", "0"), cell
        assert len([k for k in drills if k[0] == cell]) == int(m["chunks"]), cell


def test_every_drill_run_has_its_own_registered_seed_range(policy: dict) -> None:
    """The seed range [seed base, seed base + events) of every drill cell, production group and rerun is
    disjoint from every other; a single-chunk drill's seed base lies on the policy's drill grid."""
    metas: dict[str, dict[str, str]] = {}
    for r in bound.read_csv(ROOT / bound.DRILLS_RELPATH, bound.HARVEST_COLUMNS):
        if r["stat"] == "meta":
            metas.setdefault(r["cell"], {})[r["key"]] = r["value"]
    groups = [(int(g["seed_base"]), int(g["events"]), bound.group_name(g)) for g in policy["groups"]]
    ranges = sorted([(int(m["seed_base"]), int(m["events"]), c) for c, m in metas.items()] + groups)
    assert all(a[0] + a[1] <= b[0] for a, b in zip(ranges, ranges[1:], strict=False)), ranges
    grid = policy["drills"]
    for c, m in metas.items():
        if m["chunks"] == "1":
            assert (int(m["seed_base"]) - grid["seed_first"]) % (grid["seed_stride"] * 100) == 0, c


def test_the_drill_sentence_counts_are_recounted_independently() -> None:
    """The three counts of section 14's drill sentence, recounted from the committed drill summary and
    rulings by kind and intended set alone (each run's red or green outcome is the drill test's): the runs
    with a nonempty intended set outside the rulings, the drill runs whose intended set is empty (an
    unaffected route or isotope) and the controls -- with the ruled runs, every run once."""
    metas: dict[str, dict[str, str]] = {}
    for r in bound.read_csv(ROOT / bound.DRILLS_RELPATH, bound.HARVEST_COLUMNS):
        if r["stat"] == "meta":
            metas.setdefault(r["cell"], {})[r["key"]] = r["value"]
    ruled = {r["item"] for r in _rulings("statistical")}
    drills = {c: m for c, m in metas.items() if m["kind"] != "control" and c not in ruled}
    red = sum(1 for m in drills.values() if m["intended"] != "none")
    unaffected = [m for m in drills.values() if m["intended"] == "none"]
    controls = sum(1 for m in metas.values() if m["kind"] == "control")
    assert red + len(unaffected) + controls + len(ruled) == len(metas) and ruled <= set(metas)
    # an empty intended set only where the drill's defect does not reach: another route or another isotope
    for m in unaffected:
        assert (m["drill"], m["route"], m["nuclide"]) in {("drop_a", "atomic", "C-12"),
                                                         ("old_helper_atomic", "stopping", "C-12")}, m
    text = " ".join((ROOT / "DATASET_D1.md").read_text(encoding="utf-8").split())
    sentence = (r"(\d+) runs with nonempty intended failure sets turned every intended family red; "
                r"(\d+) runs on unaffected routes or isotopes and all (\d+) matched controls stayed green")
    found = re.findall(sentence, text)
    assert found == [(str(red), str(len(unaffected)), str(controls))], found


def test_every_identity_drill_is_detected() -> None:
    """Every identity drill observed as expected, except where a ruling finds the check not applicable (the
    drill never runs on that route); there the ruling's replacement drill, on the same revision and
    route, is observed as expected."""
    rows = bound.read_csv(ROOT / bound.IDENTITY_DRILLS_RELPATH,
                          ("revision", "route", "check", "expect", "observed"))
    ruled = {(r["revision"], r["route"], r["item"]): r for r in _rulings("identity")}
    by_key = {(r["revision"], r["route"], r["check"]): r for r in rows}
    for key, r in by_key.items():
        if key in ruled:
            assert ruled[key]["ruled_intended"] == "not applicable"
            replacement = by_key[(key[0], key[1], ruled[key]["rerun"])]
            assert replacement["observed"] == "True" and replacement["expect"] == "differ", replacement
        else:
            assert r["observed"] == "True", r
    assert set(ruled) <= set(by_key)
    assert {(r["revision"], r["route"]) for r in rows} == {
        (v, r) for v in bound.REVISIONS for r in bound.ROUTES}


def test_the_switch_adds_no_thread_dependence() -> None:
    """With the dataset opt-in off: the patched build with the switch off equals the unpatched build at 1
    and at 4 threads; every 4-thread configuration repeats; the switch changes the atomic route's records
    and not the stopping route's; the events that differ between 1 and 4 threads with the switch on are
    exactly those that differ in the unpatched build, and all are first events of a 4-thread worker."""
    rows = list(csv.DictReader((ROOT / WORKERS_RELPATH).open(encoding="ascii")))
    assert {(r["revision"], r["route"], r["target"]) for r in rows} == {
        (v, route, t) for v in bound.REVISIONS for route in ("bound_decay", "muonic_atom_helper")
        for t in ("6-12", "13-27")}
    for r in rows:
        checks = ("a_off_equals_pristine", "b_four_threads_repeat", "c_switch_effect",
                  "d_same_thread_dependence", "bound_rates_line")
        assert all(r[k] == "True" for k in checks) and r["verdict"] == "pass", r
        assert r["on1_vs_on4_events"] == r["pristine1_vs_pristine4_events"], r
        assert set(r["on1_vs_on4_events"].split()) <= set(r["worker_first_events"].split()), r


IDENTITY_DIGESTS = ("stops", "transport", "engine_hashes", "engine_states", "bound")


def test_the_identity_verdicts_recomputed_from_the_digests() -> None:
    """Pristine against the new build with every option off; hooked against unhooked; traced against
    untraced with the switch off and on; the previous overlay against the new one under each capture
    profile; and, outside the provider's domain and on the stopping route everywhere, the switch on
    against the switch off -- each a byte comparison of the committed digests, on every roster nuclide,
    route, revision and start time."""
    rows = bound.read_csv(ROOT / bound.IDENTITY_RELPATH, ("revision", "route", "nuclide", "t0_ns", "config",
                                                          *IDENTITY_DIGESTS))
    cells: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for r in rows:
        cells.setdefault((r["revision"], r["route"], r["nuclide"], r["t0_ns"]), {})[r["config"]] = r
    assert len(cells) == (len(bound.REVISIONS) * len(bound.ROUTES) * len(bound.IDENTITY_ROSTER)
                          * len(bound.T0_NS))
    for key, c in cells.items():
        def same(a: str, b: str, fields: tuple[str, ...] = IDENTITY_DIGESTS,
                 c: dict[str, dict[str, str]] = c) -> bool:
            return all(c[a][f] and c[a][f] == c[b][f] for f in fields)

        assert same("P", "B0"), key
        assert same("B0", "B0u", ("transport", "engine_hashes", "engine_states")), key
        assert same("B0", "T0") and same("B1", "T1"), key
        for p in ("parity", "mizuno2025", "suzuki1987", "compiled"):
            assert same(f"O_{p}", f"N_{p}"), (key, p)
        z, a = _roster_za(key[2])
        if key[1] == bound.STOPPING or not bound.in_domain(z, a):
            assert same("B1", "B0"), key
        else:
            assert c["B1"]["stops"] != c["B0"]["stops"], key
    verdicts = bound.read_csv(ROOT / bound.IDENTITY_VERDICTS_RELPATH,
                              ("revision", "route", "nuclide", "t0_ns", "check", "verdict"))
    assert verdicts and all(v["verdict"] == "True" for v in verdicts)


def _roster_za(label: str) -> tuple[int, int]:
    sym, a = label.split("-")
    (z,) = [z for s, z, aa in bound.IDENTITY_ROSTER if s == sym and aa == int(a)]
    return z, int(a)


def test_the_sweep_holds_on_every_key_and_refuses_what_the_reference_predicts(ref: bound.Reference) -> None:
    rows = bound.read_csv(ROOT / bound.SWEEP_RELPATH, tuple(_sweep_columns()))
    assert rows and all(r["verdict"] == "pass" for r in rows), [r for r in rows if r["verdict"] != "pass"]
    predicted = sorted(f"{z}-{a}" for z, a in ref.oracle.light if bound.in_domain(z, a)
                       and d1.capture_rate(z, a, ref.helper.capture_records, ref.model) < 0)
    for r in rows:
        assert r["lifecycle"].startswith("kept"), r
        if r["switch"] == "on" and r["profile"] in ("none", "compiled"):
            assert sorted(r["refused_keys"].split()) == predicted, r
        if r["switch"] == "off":
            assert int(r["disabled"]) == int(r["keys"]) and not r["refused_keys"], r


def test_document_section_14_is_the_generated_text() -> None:
    """Section 14 of DATASET_D1.md, the document's last, is the text the generator renders from the committed
    files, so every number in it is the generator's."""
    gen = _generator()
    text = (ROOT / gen.DOCUMENT).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.count("\n" + gen.SECTION_HEADING + "\n") == 1
    assert text.split("\n## 14. ", 1)[1] == gen.render_section(ROOT)[len("## 14. "):]


def _render_root(tmp_path: Path) -> Path:
    """A copy of the files section 14 is rendered from: the committed bound-rate files and their history, the
    decay-rate oracle and the helper source the reference reads."""
    root = tmp_path / "root"
    for rel in [str(x.relative_to(ROOT)) for x in (ROOT / "data/g4/d1").glob("bound_rates_*")] + [
            "data/g4/d1/history/bound_rates_results.analysis1.json", "data/g4/d1/d1_decay_rate.oracle",
            "third_party/geant4/v11.4.2/G4MuonicAtomHelper.cc"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, root / rel)
    return root


def _edit_csv(path: Path, cell: str, stat: str, key: str, value: str) -> None:
    rows = bound.read_csv(path, bound.HARVEST_COLUMNS)
    hits = [r for r in rows if (r["cell"], r["stat"], r["key"]) == (cell, stat, key)]
    assert len(hits) == 1, (cell, stat, key)
    hits[0]["value"] = value
    path.write_bytes(bound.render_csv(rows, bound.HARVEST_COLUMNS))


def _edit_json(path: Path, edit) -> None:  # type: ignore[no-untyped-def]
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="ascii", newline="\n")


def _first_cell(policy: dict, route: str) -> str:
    return bound.group_name(next(g for g in policy["groups"] if g["route"] == route))


def _first_control(rows_path: Path) -> str:
    return next(r["cell"] for r in bound.read_csv(rows_path, bound.HARVEST_COLUMNS)
                if r["stat"] == "meta" and r["key"] == "kind" and r["value"] == "control")


HISTORY_RELPATH = "data/g4/d1/history/bound_rates_results.analysis1.json"


def _stopping_pair_short(root: Path, policy: dict) -> None:
    group = next(g for g in policy["groups"] if g["route"] == bound.STOPPING)
    _edit_csv(root / bound.HARVEST_RELPATH, bound.group_name(group), "group", "rate_exact",
              str(group["events"] - 1))


def _atomic_recovered_fail(root: Path, policy: dict) -> None:
    _edit_csv(root / bound.HARVEST_RELPATH, _first_cell(policy, bound.ATOMIC), "group", "recovered_fail", "1")


def _old_analysis_version(root: Path, policy: dict) -> None:
    _edit_csv(root / bound.HARVEST_RELPATH, _first_cell(policy, bound.ATOMIC), "meta", "analysis_version",
              "1")


def _roster_problem(root: Path, policy: dict) -> None:
    _edit_json(root / bound.RESULTS_RELPATH,
               lambda d: d["RATE_RESULTS"].__setitem__("roster_problems", ["x: missing"]))


def _first_outcome_differs(root: Path, policy: dict) -> None:
    _edit_json(root / HISTORY_RELPATH,
               lambda d: d["TRANSPORT_RESULTS"].__setitem__("primary_rejections", ["x:y"]))


def _control_red(root: Path, policy: dict) -> None:
    path = root / bound.DRILLS_RELPATH
    _edit_csv(path, _first_control(path), "group", "rate_fail", "1")


def _ruled_run_absent(root: Path, policy: dict) -> None:
    path = root / bound.DRILLS_RELPATH
    item = bound.read_csv(root / RULINGS_RELPATH, RULING_COLUMNS)[0]["item"]
    rows = [r for r in bound.read_csv(path, bound.HARVEST_COLUMNS) if r["cell"] != item]
    path.write_bytes(bound.render_csv(rows, bound.HARVEST_COLUMNS))


def _drill_run_unchecked(root: Path, policy: dict) -> None:
    path = root / bound.DRILLS_RELPATH
    _edit_csv(path, _first_control(path), "meta", "raw_digest_problems", "1")


def _clock_not_as_predicted(root: Path, policy: dict) -> None:
    for path in (root / bound.RESULTS_RELPATH, root / HISTORY_RELPATH):
        _edit_json(path, lambda d: d["CLOCK_RESULTS"].__setitem__("not_as_predicted", ["x:y@0"]))


#: (label, how the copied files are changed, the refusal section 14's renderer must give)
RENDER_DEFECTS = [
    ("a stopping event without the exact pair", _stopping_pair_short,
     "not every stopping event received the pair exactly"),
    ("an atomic event whose recovered rates fail", _atomic_recovered_fail,
     "an atomic lifetime or recovered rate disagrees"),
    ("a harvest cell from another analysis version", _old_analysis_version,
     "a harvest cell from another analysis version"),
    ("a results file naming a roster problem", _roster_problem,
     "the summary does not match the registered roster"),
    ("a first-summary outcome that differs", _first_outcome_differs,
     "a registered outcome differs from the first summary's"),
    ("a control that turned red", _control_red, "the section does not state it"),
    ("a drill run whose raw files were not checked clean", _drill_run_unchecked,
     "a drill run whose run record or raw files were not checked clean"),
    ("a ruled run missing from the drill summary", _ruled_run_absent,
     "a ruled drill run or its rerun is not in the drill summary"),
    ("a clock outcome not as predicted, in both summaries", _clock_not_as_predicted,
     "a clock result not as predicted"),
]


def test_section_14_renders_the_same_from_a_copy_of_its_files(tmp_path: Path) -> None:
    gen = _generator()
    assert gen.render_section(_render_root(tmp_path)) == gen.render_section(ROOT)


@pytest.mark.parametrize("label, change, refusal", RENDER_DEFECTS, ids=[c[0] for c in RENDER_DEFECTS])
def test_drill_section_14_refuses_what_it_does_not_state(  # type: ignore[no-untyped-def]
        tmp_path: Path, policy: dict, label: str, change, refusal: str) -> None:
    root = _render_root(tmp_path)
    change(root, policy)
    with pytest.raises(SystemExit, match=re.escape(refusal)):
        _generator().render_section(root)


def document_pins(numeric: re.Pattern[str]) -> list[tuple[str, str, str, tuple[int, ...], object]]:
    """``(what, path, pattern, groups, expected)`` for every line of section 14 that carries a number: each
    line whitespace-collapsed with every number a group (the section is required to be the generated text)."""
    from test_g4mixture import _line_pattern

    gen = _generator()
    pins: list[tuple[str, str, str, tuple[int, ...], object]] = []
    for index, line in enumerate(gen.render_section(ROOT).splitlines(), start=1):
        pattern, groups = _line_pattern(line, numeric)
        if groups:
            pins.append((f"section 14 line {index}", gen.DOCUMENT, pattern, groups, None))
    return pins


def test_document_section_14_pins_match_once() -> None:
    collapsed = " ".join((ROOT / "DATASET_D1.md").read_text(encoding="utf-8").split())
    for what, _, pattern, _, _ in document_pins(re.compile(r"[0-9][0-9,.]*[0-9]|[0-9]|[⁰¹²³⁴⁵⁶⁷⁸⁹]+")):
        assert len(list(re.finditer(pattern, collapsed))) == 1, what


def _sweep_columns() -> list[str]:
    with (ROOT / bound.SWEEP_RELPATH).open(encoding="ascii") as f:
        return f.readline().strip().split(",")
