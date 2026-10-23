"""The common bound-rate pair of the two stopped-muon routes: the reference evaluator, the verification policy
and the stock reproducer (``openmucf/g4/d1_boundrates.py``, ``data/g4/d1/bound_rates_policy.json``,
``cpp/tools/repro_bound_rates.cc``)."""

from __future__ import annotations

import csv
import hashlib
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
    tau, tau_dio, tau_nc = 1.0 / lam, 1.0 / d, 1.0 / c
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
    clocks = bound.gate_clocks({250: s}, route)
    assert clocks["delivery"]["verdict"] == "pass", clocks["delivery"]
    for key, v in clocks["physical"].items():
        assert v["as_predicted"], (key, v)


def test_the_physical_clock_passes_a_correct_atomic_clock_and_fails_a_shifted_one(tmp_path: Path,
                                                                                   c12: bound.Pair) -> None:
    """A synthetic atomic stream with the physically correct clock (atom at t0, decay at t0 + T, capture
    products at t0 + T + max(tau, 0)) passes the physical gate; the same stream with every atomic product
    moved by 5 ns fails it. The reference origin is t0, never a product's own time."""
    good = _summary(tmp_path / "good", bound.ATOMIC, c12, physical_atom_clock=True)
    phys = bound.gate_clocks({250: good}, bound.ATOMIC)["physical"]
    assert all(v["verdict"] == "pass" for v in phys.values()), phys
    bad = _summary(tmp_path / "bad", bound.ATOMIC, c12, physical_atom_clock=True, shift=5.0)
    phys = bound.gate_clocks({250: bad}, bound.ATOMIC)["physical"]
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
    assert bound.gate_clocks({250: s}, bound.STOPPING)["delivery"]["verdict"] == "fail", label


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
    s = _summary(tmp_path, bound.STOPPING, c12, mutate=lambda ls: [line.replace(" P 1 0 0 ", " P 0 0 -1 ")
                                                                   for line in ls])
    verdict = bound.gate_rate(s, bound.STOPPING)
    assert verdict["verdict"] == "fail" and any("wiring" in p for p in verdict["problems"]), verdict
    assert bound.gate_complete(s, 4000)["verdict"] == "pass"
    assert {v["verdict"] for v in bound.gate_distribution(s, 36).values()} == {"pass"}


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
     {"mutate": lambda ls: [x for x in ls if not x.startswith("C ")]}, "0 helper calls in 1 processes"),
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
    clocks = bound.gate_clocks({250: none}, bound.STOPPING)
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


def _synthetic_harvest(tmp_path: Path, policy: dict, *, skip_group: int | None = None,
                       registered_counts: bool = False) -> Path:
    """A repository root holding the theory status, a policy whose groups have ``SYNTHETIC_STRATUM`` events
    per stratum (unless ``registered_counts``) and the harvest summary of correct synthetic chunks of every
    group but ``skip_group``."""
    from scipy import stats

    root = tmp_path / "root"
    (root / "data/g4/d1").mkdir(parents=True)
    (root / bound.THEORY_RELPATH).write_bytes((ROOT / bound.THEORY_RELPATH).read_bytes())
    small = json.loads(json.dumps(policy))
    n = 3 * SYNTHETIC_STRATUM
    if not registered_counts:
        alpha = float(small["statistics"]["primary"]["per_test_alpha"])
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
            (cell / "run.json").write_text(json.dumps({"build_binary_sha256": "synthetic", "files": {}}),
                                           encoding="utf-8")
            chunks.append({"group": g["group"], "first": st["first"], "count": SYNTHETIC_STRATUM,
                           "t0_ns": st["t0_ns"], "cell": str(cell)})
    bound.summarize_plan(root, {"switch_on": True, "chunks": chunks}, root / "data/g4/d1")
    return root


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
    assert missing["RATE_RESULTS"]["status"] == "FAIL"
    assert missing["RATE_RESULTS"]["groups"][name]["complete"]["problems"] == [
        "the summary has no cell for this group"]
    assert missing["TRANSPORT_RESULTS"]["status"] == "FAIL"
    rejected = missing["TRANSPORT_RESULTS"]["primary_rejections"]
    assert {r.split(":", 3)[3] for r in rejected} == set(bound.PRIMARY_TESTS)
    assert {r.rsplit(":", 1)[0] for r in rejected} == {name}
    assert missing["CLOCK_RESULTS"]["delivery_status"] == "FAIL"
    # the same correct chunks graded against the registered counts: never complete, never inside the limits
    short = gen.build_results(_synthetic_harvest(tmp_path / "b", policy, registered_counts=True))
    assert short["RATE_RESULTS"]["status"] == "FAIL"
    for group in policy["groups"]:
        problems = short["RATE_RESULTS"]["groups"][bound.group_name(group)]["complete"]["problems"]
        assert f"events {3 * SYNTHETIC_STRATUM} != {group['events']}" in problems
    assert short["TRANSPORT_RESULTS"]["status"] == "FAIL"


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


def test_every_group_is_complete_and_every_consumed_rate_is_the_registered_one(policy: dict,
                                                                             harvest: dict) -> None:
    for g in policy["groups"]:
        s = harvest[bound.group_name(g)]["group"]
        assert bound.gate_complete(s, g["events"])["verdict"] == "pass", bound.group_name(g)
        assert bound.gate_rate(s, g["route"])["verdict"] == "pass", bound.group_name(g)
        assert s["rate_exact"] == g["events"], bound.group_name(g)


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
