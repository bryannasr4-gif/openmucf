"""The evaluated bound-muon rate pair: independent re-derivations, the decision rules, and their refusals.

`openmucf/g4/d1_evaluated.py` forms one capture/decay pair from one primitive tuple ``(tau, q, free_tau)``,
assembles the decay factor and its evaluator region, encloses the late-slope mapping of a scalar fit to a
two-state
hyperfine spectrum, and decides fixed budgets over the frozen product of regions. These tests re-derive what
it
computes by other means (scipy quadrature and optimizers, the matrix exponential of the two-state generator, a
dense grid), and drill every rule with input it must refuse or decide adversely. They never compare a number
with
a literal typed from a report: expected values come from the committed primitives, the oracle or an
independent
computation.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import pytest
from scipy import integrate, linalg, optimize

from openmucf.g4 import d1_evaluated as ev
from openmucf.g4 import d1_lifetime as lifetime

REPO = pathlib.Path(__file__).resolve().parents[1]
INPUTS = REPO / lifetime.EVALUATED_INPUTS_RELPATH


@pytest.fixture(scope="module")
def inputs() -> ev.Inputs:
    return lifetime.evaluated_inputs(REPO)


@pytest.fixture(scope="module")
def oracle(inputs: ev.Inputs) -> ev.OracleInputs:
    return lifetime.oracle_inputs(REPO, inputs)


@pytest.fixture(scope="module")
def record() -> dict:
    return lifetime.evaluation(REPO)


def raw_inputs() -> dict:
    return json.loads(INPUTS.read_bytes())


def dumped(obj: object) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _generator():
    path = REPO / "scripts" / "generate_g4evaluated_rates.py"
    spec = importlib.util.spec_from_file_location("generate_g4evaluated_rates", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------------
# the committed record is its recomputation
# --------------------------------------------------------------------------------------------


def test_the_committed_evaluation_and_convention_record_are_their_recomputation():
    _generator().check(REPO)


def test_the_registered_decisions_hold(record):
    """The frozen rules give, for the admitted Al27 row: late lifetime and decay rate MEETS; stopped mean and
    capture probability NOT_ESTABLISHED with L = 0; atomic-route late-lifetime improvement over stock
    ESTABLISHED and every other improvement NOT_ESTABLISHED."""
    d = record["decisions"]
    assert {k: v["verdict"] for k, v in d.items()} == {
        "late_lifetime": ev.MEETS,
        "decay_rate": ev.MEETS,
        "stopped_mean": ev.NOT_ESTABLISHED,
        "capture_probability": ev.NOT_ESTABLISHED,
    }
    assert d["stopped_mean"]["L"] == 0.0 and d["capture_probability"]["L"] == 0.0
    assert d["late_lifetime"]["label"] == "source-precision conditional (no validation power)"
    improvements = {
        (route, name): v["verdict"]
        for route, s in record["stock"].items()
        for name, v in s["improvement"].items()
    }
    assert [k for k, v in improvements.items() if v == ev.ESTABLISHED] == [("atomic", "late_lifetime")]
    assert record["roster"] == {
        "Al27": "IN",
        "O16": "DEFERRED_RUNTIME_SOURCE_MAPPING",
        "C12": "DEFERRED_RUNTIME_MIXTURE_ESTIMAND",
    }
    assert record["admitted_rows"] == [[13, 27]]


def test_power_on_the_actual_region(record):
    """At exact tau a wrong partition (q mutant) fails the decay-rate and capture-probability decisions, and a
    stopped mean three per cent high fails its decision, on the same finite region the production row
    meets."""
    p = record["power"]
    assert p["q_mutant"]["decay_rate"]["verdict"] == ev.DOES_NOT_MEET
    assert p["q_mutant"]["capture_probability"]["verdict"] == ev.DOES_NOT_MEET
    assert p["mean_mutant"]["verdict"] == ev.DOES_NOT_MEET
    for v in (*p["q_mutant"].values(), p["mean_mutant"]):
        assert v["L"] > 0


# --------------------------------------------------------------------------------------------
# independent re-derivations
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("z", [6, 8, 13])
def test_the_uehling_term_matches_adaptive_quadrature_in_another_coordinate(z):
    """The kinetic Uehling integral in t = cosh(u) form (an independent coordinate), by scipy quad."""
    alpha = 1 / 137.035999173
    zz = z * alpha
    kappa = 0.0048363 / zz

    def integrand(u: float) -> float:
        c = math.cosh(u)
        return math.tanh(u) ** 2 * (1 + 0.5 / c**2) * (1 + 3 * kappa * c) / (1 + kappa * c) ** 3

    iq = integrate.quad(integrand, 0, 40, epsabs=0, epsrel=1e-13, limit=500)[0]
    expected = -2 * alpha * zz**2 / (3 * math.pi) * iq
    assert ev.uehling_kinetic(z, 137.035999173, 0.0048363) == pytest.approx(expected, rel=1e-11)


def test_the_q_assembly_and_region_follow_the_frozen_rule(inputs, oracle, record):
    """Q = Q0 + (Z alpha)^2 eta + dQ_U; W = max |Q - Q_k| + sum |ledger|, recomputed here term by term."""
    al = record["q"]["Al27"]
    alpha, mu = oracle.alpha, oracle.mu_mass
    z = 13 * alpha
    eta = mu / oracle.masses[(13, 27)]
    dq = ev.uehling_kinetic(13, 137.035999173, 0.0048363)
    assembly = 0.9936 + z * z * eta + dq
    ledger = [
        5e-5 / math.sqrt(3),
        2e-5 * 0.9936,
        z**3 * eta,
        z * z * eta**2,
        1.5 * z * z * eta**2,
        1.06 * z * z * eta,
        abs(dq) * z * z,
        alpha / math.pi * z * z,
        (alpha / math.pi) ** 2 * z * z,
    ]
    members = [0.9936, 0.9934, 0.993, 1 - z * z * (0.5 - eta)]
    w = max(abs(assembly - m) for m in members) + sum(ledger)
    assert al["assembly"] == pytest.approx(assembly, rel=1e-15)
    assert al["half_width"] == pytest.approx(w, rel=1e-12)
    assert al["region"] == pytest.approx([assembly - w, assembly + w], rel=1e-14)
    assert record["tuple"]["q"] == str(round(assembly, inputs.q_decimals))


def _two_state(b: float, d: float, cp: float, h: float, f: float):
    cm = b - d
    gen = np.array([[-(cp + d + h), 0.0], [h, -(cm + d)]])
    p0 = np.array([f, 1 - f])
    return gen, p0


@pytest.mark.parametrize("seed", range(6))
def test_mean_and_capture_probability_equal_the_two_state_resolvent(seed, record):
    """M = -1^T G^-1 p0 and Pcap = 1 - d M from the exact generator, at random points of the frozen box."""
    rng = np.random.default_rng(seed)
    box = [record["box"][c] for c in ev.COMPONENTS]
    x = [lo + (hi - lo) * rng.random() for lo, hi in box]
    r = ev.reference(x, 7 / 12)
    gen, p0 = _two_state(1 / r.tau_late, r.decay, r.c_plus, r.transition, 7 / 12)
    mean = float(-np.ones(2) @ np.linalg.solve(gen, p0))
    assert r.mean == pytest.approx(mean, rel=1e-12)
    assert r.capture_probability == pytest.approx(1 - r.decay * mean, abs=1e-14)
    # DIO-conditional mean: int t d S(t) dt / int d S(t) dt, by quadrature of the matrix exponential
    s = lambda t: float(np.ones(2) @ linalg.expm(gen * t) @ p0)  # noqa: E731
    num = integrate.quad(lambda t: t * s(t), 0, 60000, limit=400, epsrel=1e-12, points=[100, 1000, 5000])[0]
    den = integrate.quad(s, 0, 60000, limit=400, epsrel=1e-12, points=[100, 1000, 5000])[0]
    assert ev.dio_conditional_mean(r, 7 / 12) == pytest.approx(num / den, rel=1e-9)


LOW, HIGH = 40.0, 57000.0


def _window_mean(rate: float) -> float:
    """The mean of a truncated exponential on [LOW, HIGH]."""
    e0, e1 = math.exp(-rate * LOW), math.exp(-rate * HIGH)
    return ((LOW * e0 - HIGH * e1) / rate + (e0 - e1) / rate**2) / ((e0 - e1) / rate)


def test_the_mapping_projection_matches_quadrature_and_the_matrix_exponential():
    """beta from the closed form equals the finite-window maximum-likelihood fit by quadrature of the exact
    survival (expm), at the brief's example point and at the self-consistent worst point."""
    f, t0 = 7 / 12, 40.0
    for tau_late, ratio, h in ((863.7, 18.4228, 0.023), (860.0, 2.0, 0.05), (866.0, 400.0, 0.03)):
        b = 1 / tau_late
        d = 0.993598 / 2196.9811
        cm = b - d
        cp = ratio * cm
        delta = cm - cp
        a = cp + d + h
        gen, p0 = _two_state(b, d, cp, h, f)
        s = lambda t, gen=gen, p0=p0: float(np.ones(2) @ linalg.expm(gen * t) @ p0)  # noqa: E731
        num = integrate.quad(
            lambda t, s=s: t * s(t), LOW, HIGH, limit=500, epsrel=1e-13, points=[100, 400, 3000]
        )[0]
        den = integrate.quad(s, LOW, HIGH, limit=500, epsrel=1e-13, points=[100, 400, 3000])[0]
        target = num / den
        rate = optimize.brentq(
            lambda r, target=target: _window_mean(r) - target, 0.3 * b, 3 * b, xtol=1e-20, rtol=1e-15
        )
        fitted = b / rate - 1
        assert ev.window_bias(b, a, delta, h, f, t0, HIGH) == pytest.approx(fitted, rel=1e-6, abs=1e-12)
        assert ev.mapping_bias(b, a, delta, h, f, t0) == pytest.approx(fitted, rel=1e-4, abs=1e-9)


def test_the_upper_function_bounds_the_projection_and_its_maximum_is_the_stationary_root(record):
    """For u >= h, |beta| <= F; F's maximum over a dense grid equals the admissible stationary root's value;
    F increases with B and decreases with h, as the self-consistent construction uses."""
    m = record["mapping"]
    f, t0, h = 7 / 12, 40.0, m["worst_transition_per_ns"]
    b = (1 + m["self_consistent_eps"]) / m["fit_low_ns"]
    grid = np.concatenate([np.linspace(h, 0.2, 200001), np.linspace(0.2, 5, 2001)])
    values = [ev.upper_function(float(u), b, h, f, t0) for u in grid]
    assert max(values) == pytest.approx(m["self_consistent_eps"], rel=1e-9)
    assert grid[int(np.argmax(values))] == pytest.approx(m["admissible_stationary_roots_per_ns"][0], rel=1e-4)
    for u in np.linspace(h * 1.0001, 1.0, 400):
        delta = h - float(u)
        a = float(u) + b
        assert abs(ev.mapping_bias(b, a, delta, h, f, t0)) <= ev.upper_function(float(u), b, h, f, t0) * (
            1 + 1e-12
        )
        assert ev.upper_function(float(u), b * 1.01, h, f, t0) > ev.upper_function(float(u), b, h, f, t0)
        assert ev.upper_function(float(u), b, h * 1.01, f, t0) <= ev.upper_function(float(u), b, h, f, t0)
    assert m["eps_map"] == "0.001919"
    assert m["positive_branch_sup"] < float(m["eps_map"])


@pytest.mark.parametrize("name", [n for n, _ in ev.DECISIONS])
def test_no_bounded_optimizer_finds_an_extreme_beyond_the_reported_range(name, record):
    """scipy L-BFGS-B from 40 starts never leaves the reported range (which carries the outward allowance)."""
    box = [tuple(record["box"][c]) for c in ev.COMPONENTS]
    lo = np.array([b[0] for b in box])
    hi = np.array([b[1] for b in box])
    rng = np.random.default_rng(7)

    def fn(y: np.ndarray, sign: float) -> float:
        x = lo + (hi - lo) * np.clip(y, 0, 1)
        return sign * float(ev.reference_quantity(name, ev.reference(list(x), 7 / 12)))

    reported = record["ranges"][name]
    for _ in range(40):
        y0 = rng.random(5)
        for sign in (1.0, -1.0):
            r = optimize.minimize(fn, y0, args=(sign,), bounds=[(0, 1)] * 5, method="L-BFGS-B")
            v = sign * r.fun
            assert reported["low"] <= v <= reported["high"]


# --------------------------------------------------------------------------------------------
# the decision rule, its adverse outcomes and its non-informative cases
# --------------------------------------------------------------------------------------------


def test_drill_the_decision_returns_each_outcome_and_holds_its_edges():
    assert ev.decision(1.0, 0.995, 1.005, 0.01, "relative")["verdict"] == ev.MEETS
    assert ev.decision(1.0, 1.02, 1.03, 0.01, "relative")["verdict"] == ev.DOES_NOT_MEET
    assert ev.decision(1.0, 0.98, 1.02, 0.01, "relative")["verdict"] == ev.NOT_ESTABLISHED
    # binary-exact values: U exactly on the budget meets; L exactly on it does not fail
    assert ev.decision(0.5, 0.5, 0.625, 0.125, "absolute")["verdict"] == ev.MEETS
    assert ev.decision(0.5, 0.625, 0.75, 0.125, "absolute")["verdict"] == ev.NOT_ESTABLISHED


def test_drill_an_empty_or_full_domain_reference_is_never_a_pass():
    assert ev.decision(1.0, 2.0, 1.0, 0.01, "relative")["verdict"] == ev.VACUOUS
    assert ev.decision(1.0, 0.0, 10.0, 0.01, "relative")["verdict"] == ev.NOT_ESTABLISHED
    full = ev.decision(0.6, 0.0, 1.0, 0.005, "absolute")
    assert full["verdict"] == ev.NOT_ESTABLISHED and full["L"] == 0.0


def test_drill_large_bias_with_tiny_numerical_error_fails_and_inflated_uncertainty_cannot_rescue_it():
    """The rule reads only the central prediction: there is no argument through which a prediction's own
    uncertainty could widen the reference."""
    assert ev.decision(1.10, 1.0, 1.0 + 1e-15, 0.01, "relative")["verdict"] == ev.DOES_NOT_MEET
    with pytest.raises(TypeError):
        ev.decision(1.10, 1.0, 1.0, 0.01, "relative", prediction_uncertainty=1.0)  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "1.0", None, [1.0]])
def test_drill_the_decision_refuses_a_non_finite_or_non_real_value(bad):
    for args in ((bad, 0.9, 1.1, 0.01), (1.0, bad, 1.1, 0.01), (1.0, 0.9, bad, 0.01), (1.0, 0.9, 1.1, bad)):
        with pytest.raises(ev.EvaluationError):
            ev.decision(*args, "relative")


def test_drill_improvement_holds_positive_zero_and_negative_margins():
    """A strict comparison: a reference interval reaching the midpoint is not an improvement (the `>` to `>=`
    mutant would pass it)."""
    assert ev.improvement(1.0, 2.0, 0.5, 1.4)["verdict"] == ev.ESTABLISHED
    assert ev.improvement(1.0, 2.0, 0.5, 1.5)["verdict"] == ev.NOT_ESTABLISHED
    assert ev.improvement(1.0, 2.0, 1.6, 3.0)["verdict"] == ev.NOT_ESTABLISHED
    assert ev.improvement(1.0, 2.0, 1.5, 1.4)["verdict"] == ev.VACUOUS


def test_drill_a_synthetic_large_contrast_fixture_fails_the_scalar_mean_and_probability(record):
    """Excluded by the frozen physical region, but supported by positivity: c_plus = 0 at the slowest
    transition. A scalar pair at the late slope then misses the stopped mean and the capture probability."""
    t = record["tuple"]
    p = ev.pair(float(t["tau_ns"]), float(t["q"]), float(t["free_tau_ns"]))
    r = ev.reference((p.tau, p.q, p.free_tau, 1.0, 0.023), 7 / 12)
    assert r.c_plus == pytest.approx(0, abs=1e-18)
    assert ev.decision(p.lifetime, r.mean, r.mean, 0.01, "relative")["verdict"] == ev.DOES_NOT_MEET
    assert (
        ev.decision(p.capture_probability, r.capture_probability, r.capture_probability, 0.005, "absolute")[
            "verdict"
        ]
        == ev.DOES_NOT_MEET
    )
    assert not (record["box"]["contrast"][0] <= 1.0 <= record["box"]["contrast"][1])


def test_drill_mean_used_as_late_slope_fails_the_estimand_gate(inputs, record):
    """The tuple's lifetime is the source's late-slope fit, exactly; the mean-matched alternative is a
    different
    estimand and is never the production tuple."""
    assert record["tuple"]["tau_ns"] == str(inputs.tau)
    mm = record["mean_matched"]
    assert mm["selected_for_production"] is False
    with pytest.raises(ev.EvaluationError, match="estimand"):
        ev.check_estimand(repr(mm["mean_ns"]), inputs)
    ev.check_estimand(record["tuple"]["tau_ns"], inputs)


def test_drill_old_capture_with_new_q_fails_the_convention_gate(inputs, record):
    """A source capture rate formed with the source's own Q, kept while Q changes, shifts the lifetime by a
    fraction of a nanosecond: far inside the loose budget, refused by the convention gate."""
    t = record["tuple"]
    tau, q, ft = float(t["tau_ns"]), float(t["q"]), float(t["free_tau_ns"])
    c_source = 1 / tau - 0.993 / ft
    d_new = q / ft
    shifted = 1 / (c_source + d_new)
    assert 0.1 < abs(shifted - tau) < 0.01 * tau
    with pytest.raises(ev.EvaluationError, match="convention"):
        ev.check_delivered(c_source, d_new, t)
    ev.check_delivered(1 / tau - d_new, d_new, t)


# --------------------------------------------------------------------------------------------
# the structural covariance gate
# --------------------------------------------------------------------------------------------


def _primitive_draws(n: int, u_q: float, seed: int) -> list[tuple[str, float, float, float]]:
    rng = np.random.default_rng(seed)
    return [
        (f"draw-{i}", 863.7 + rng.normal(), 0.993598 + u_q * rng.normal(), 2196.9811 + 0.0022 * rng.normal())
        for i in range(n)
    ]


@pytest.mark.parametrize("u_q", [0.0, 0.001], ids=["real-record", "synthetic-larger-variance"])
def test_drill_independently_sampled_marginals_fail_the_structural_gate(u_q):
    draws = ev.draws_from_primitives(_primitive_draws(200, u_q, 1))
    ev.check_draws(draws)
    ids = ev.exact_identities(draws)
    assert ids["var_total"] == pytest.approx(ids["var_lambda"], rel=1e-9)
    assert ids["cov_cd"] == pytest.approx(ids["cov_lambda_d_minus_var_d"], rel=1e-9, abs=1e-30)
    # the mutant: c and d drawn from their marginals independently, each paired with some draw's primitives
    rng = np.random.default_rng(2)
    cs = rng.permutation([d.capture for d in draws])
    mutant = [
        ev.Draw(d.id, d.tau, d.q, d.free_tau, float(c), d.decay) for d, c in zip(draws, cs, strict=True)
    ]
    with pytest.raises(ev.EvaluationError, match="c \\+ d"):
        ev.check_draws(mutant)


def test_drill_draws_without_ids_or_with_invalid_values_are_refused():
    good = ev.draws_from_primitives(_primitive_draws(3, 0.0, 3))
    with pytest.raises(ev.EvaluationError):
        ev.draws_from_primitives([("a", 863.7, 0.99, 2196.98), ("a", 863.7, 0.99, 2196.98)])
    for change in ({"id": ""}, {"capture": math.nan}, {"decay": -1.0}, {"capture": -1e-6}):
        bad = [ev.Draw(**{**good[0].__dict__, **change})]
        with pytest.raises(ev.EvaluationError):
            ev.check_draws(bad)
    with pytest.raises(ev.EvaluationError):
        ev.check_draws([])


def test_the_first_order_view_keeps_nulls_and_differs_from_the_exact_fixture(record):
    fo = record["covariance"]["first_order"]
    assert fo["label"] == "FIRST_ORDER" and fo["null_components"] == ["q"]
    fx = record["covariance"]["fixture"]
    assert fx["exact"]["var_lambda"] == fx["exact"]["var_total"]
    assert 1e-7 < fx["relative_gap"] < 1e-5


@pytest.mark.parametrize(
    "tau,q,ft",
    [
        (math.nan, 0.99, 2197.0),
        (863.7, math.inf, 2197.0),
        (863.7, 0.99, -2197.0),
        (863.7, 1.0000001, 2197.0),
        (863.7, 0.0, 2197.0),
        (3000.0, 0.999, 2197.0),
        (True, 0.99, 2197.0),
        ("863.7", 0.99, 2197.0),
    ],
)
def test_drill_the_pair_refuses_invalid_primitives(tau, q, ft):
    with pytest.raises(ev.EvaluationError):
        ev.pair(tau, q, ft)


# --------------------------------------------------------------------------------------------
# typed inputs: every hostile input is refused
# --------------------------------------------------------------------------------------------


def _mutated(path: list, value: object) -> bytes:
    obj = raw_inputs()
    node = obj
    for key in path[:-1]:
        node = node[key]
    if value is KeyError:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return dumped(obj)


@pytest.mark.parametrize(
    "path,value",
    [
        (["lifetime", "tau_ns"], "NaN"),
        (["lifetime", "tau_ns"], "Infinity"),
        (["lifetime", "tau_ns"], "-inf"),
        (["lifetime", "tau_ns"], 863.7),
        (["lifetime", "tau_ns"], True),
        (["lifetime", "tau_ns"], ""),
        (["lifetime", "tau_ns"], "abc"),
        (["lifetime", "tau_ns"], KeyError),
        (["free_lifetime", "tau_ns"], "-2196.9811"),
        (["nuclei", "Al27", "Z"], 13.0),
        (["nuclei", "Al27", "Z"], "13"),
        (["nuclei", "Al27", "static_q"], "nan"),
        (["nuclei", "O16", "roster"], "IN"),
        (["nuclei", "C12", "roster"], "IN"),
        (["nuclei", "Al27", "roster"], "DEFERRED_RUNTIME_SOURCE_MAPPING"),
        (["nuclei", "Al27", "roster"], "ACCEPTED"),
        (["nuclei", "Al27", "A"], 28),
        (["nuclei", "Al27", "A"], 0),
        (["budgets", "stopped_mean"], "0"),
        (["hyperfine", "transition_center_per_us"], "9"),
        (["lifetime", "standard_uncertainty_ns"], "0.94"),
        (["lifetime", "unexpected"], "1"),
        (["schema"], "openmucf.bound-rate-inputs/2"),
        (["pair_profile"], "parity"),
    ],
)
def test_drill_the_inputs_loader_refuses_hostile_input(path, value):
    with pytest.raises(ev.EvaluationError):
        ev.load_inputs(_mutated(path, value))


def test_drill_the_inputs_loader_refuses_duplicate_keys_json_constants_and_cr():
    text = INPUTS.read_text(encoding="utf-8")
    with pytest.raises(ev.EvaluationError, match="duplicate"):
        ev.load_inputs(
            text.replace('"tau_ns": "863.7",', '"tau_ns": "863.7", "tau_ns": "863.7",', 1).encode()
        )
    with pytest.raises(ev.EvaluationError):
        ev.load_inputs(text.replace('"coverage_k": "2",', '"coverage_k": NaN,', 1).encode())
    with pytest.raises(ev.EvaluationError, match="CR"):
        ev.load_inputs(text.replace("\n", "\r\n").encode())
    prescriptions = raw_inputs()
    prescriptions["nuclei"]["Al27"]["prescriptions"].append(
        prescriptions["nuclei"]["Al27"]["prescriptions"][0]
    )
    with pytest.raises(ev.EvaluationError, match="duplicate prescription"):
        ev.load_inputs(dumped(prescriptions))


def test_drill_omitting_a_stated_calibration_component_is_refused():
    """The lifetime's stated standard uncertainty must cover the quadrature of its stated components:
    dropping the
    calibration/background terms (keeping only the fit) is refused even though every central value is
    exact."""
    with pytest.raises(ev.EvaluationError, match="components"):
        ev.load_inputs(_mutated(["lifetime", "standard_uncertainty_ns"], "0.94"))


def test_drill_a_deferred_row_cannot_reach_the_pair_table_by_relabelling():
    """Changing O16's status string is refused by the loader, and only admitted nuclei produce a tuple: no O
    or
    C row and no natural-composition alias reaches the table."""
    assert [(z, a) for z, a, *_ in lifetime.evaluated_tuples(REPO)] == [(13, 27)]


# --------------------------------------------------------------------------------------------
# committed records: derived fields are recomputed, never trusted
# --------------------------------------------------------------------------------------------


def _evaluation_obj() -> dict:
    return json.loads((REPO / lifetime.EVALUATION_RELPATH).read_bytes())


@pytest.mark.parametrize(
    "change",
    [
        lambda o: o["decisions"]["stopped_mean"].__setitem__("verdict", "MEETS"),
        lambda o: o["decisions"]["capture_probability"].__setitem__("U", 0.004),
        lambda o: o["pair"].__setitem__("capture_per_ns", o["pair"]["capture_per_ns"] * (1 + 1e-9)),
        lambda o: o["q"]["Al27"].__setitem__("half_width", 0.001),
        lambda o: o["ranges"]["stopped_mean"].__setitem__("high", 870.0),
        lambda o: o["stock"]["stopping"]["improvement"]["late_lifetime"].__setitem__(
            "verdict", "ESTABLISHED"
        ),
        lambda o: o["covariance"]["first_order"].__setitem__("null_components", []),
        lambda o: o.__setitem__("unknown", 1),
        lambda o: o.pop("power"),
        lambda o: o["roster"].__setitem__("O16", "IN"),
    ],
    ids=[
        "verdict",
        "U",
        "pair",
        "half-width",
        "range",
        "improvement",
        "nulls",
        "unknown-key",
        "missing-key",
        "roster",
    ],
)
def test_drill_a_resealed_record_with_a_false_derived_field_is_refused(change, record):
    obj = _evaluation_obj()
    change(obj)
    with pytest.raises(ev.EvaluationError):
        ev.check_evaluation(dumped(obj), record)


def test_a_reordered_record_keeps_its_identity_and_metadata_wrapping_keeps_the_payload_digest(record):
    obj = _evaluation_obj()
    reordered = json.dumps(obj, sort_keys=False, indent=3).encode()
    ev.check_evaluation(reordered, record)
    convention = json.loads((REPO / lifetime.CONVENTION_RELPATH).read_bytes())
    wrapped = copy.deepcopy(convention)
    wrapped["note"] = "a harmless metadata field"
    assert ev.payload_digest(wrapped["payload"]) == convention["payload_sha256"]
    tampered = copy.deepcopy(convention["payload"])
    tampered["evaluated_physics"]["Al27"]["serialized_q"] = "0.993599"
    assert ev.payload_digest(tampered) != convention["payload_sha256"]


def test_the_shared_record_keeps_the_two_source_response_identities_apart():
    payload = json.loads((REPO / lifetime.CONVENTION_RELPATH).read_bytes())["payload"]
    response = payload["source_response"]
    assert response["stanislaus1983_thesis"]["q_formula"] == "1 - 0.5 (Z/137)^2"
    assert response["stanislaus1987_paper"]["status"] == ev.UNRESOLVED
    d2 = response["d2_response"]
    assert d2["delivery_mode"] == "SOURCE_REPORTED_FROZEN"
    assert d2["conversion_refusal"] == "CONVERSION_NOT_ESTABLISHED"
    assert d2["spectrum_id"] == ev.UNRESOLVED
    assert d2["k_definition"] == "K_ij = Q_i*E2_ij = Gamma_acc,ij/Gamma_free"
    assert {k: v["runtime_status"] for k, v in payload["evaluated_physics"].items()} == {
        "Al27": "IN",
        "O16": "DEFERRED_RUNTIME_SOURCE_MAPPING",
        "C12": "DEFERRED_RUNTIME_MIXTURE_ESTIMAND",
    }


def test_no_gate_reads_raw_json_outside_the_typed_loaders():
    """The evaluation's gates parse JSON only through d1_evaluated's typed loaders: an AST scan of the gate
    code
    finds no other json.load/json.loads call."""
    for rel in ("scripts/generate_g4evaluated_rates.py", "openmucf/g4/d1_lifetime.py"):
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("load", "loads")
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "json"
        ]
        assert not calls, (rel, [c.lineno for c in calls])


# --------------------------------------------------------------------------------------------
# the descriptive census and the source findings
# --------------------------------------------------------------------------------------------


def test_the_census_counts_come_from_the_committed_compendium(record):
    rows = lifetime.compendium_rows(REPO)
    for name, z, label in (("O16", 8, "O"), ("C12", 6, "C")):
        block = [r for r in rows if r["Z"] == str(z) and r["target_label"] == label]
        with_tau = [r for r in block if r["mean_life_ns"]]
        assert record["census"][name]["lifetime_records"] == len(with_tau)
        assert record["census"][name]["rate_only_records"] == len(block) - len(with_tau)
        assert record["census"][name]["status"].startswith("DESCRIPTIVE_ONLY")


def test_the_stock_table_finding_reproduces_the_compiled_entries(record):
    for label, v in record["stock_table_finding"].items():
        assert v["matches_stock_at_4_decimals"], label
    assert record["stock_table_finding"]["Al"]["difference_in_combined_printed_units"] > 10


def test_the_comparison_identifies_the_pair_rows_and_both_routes_take_the_pair():
    rows = [r for r in lifetime.comparison_rows(REPO) if r["profile"] == ev.PAIR_PROFILE]
    assert rows and all(r["lookup"] == lifetime.PAIR_ROUTE for r in rows)
    for r in rows:
        assert r["rate_bound"] == r["rate_helper"] and r["decay_rate_bound"] == r["decay_rate_helper"]
    assert {r["key_a"] for r in rows} == {"27"}


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))


# --------------------------------------------------------------------------------------------
# the written artifacts: tampered bytes, a lost sibling, reordered columns or rows
# --------------------------------------------------------------------------------------------


def _check_with(monkeypatch, relpath: str, change) -> None:
    """Run the generator's check with one committed file's bytes changed (``change`` returns the new bytes, or
    raises FileNotFoundError for a missing file)."""
    target = (REPO / relpath).resolve()
    original = pathlib.Path.read_bytes

    def read_bytes(self: pathlib.Path) -> bytes:
        if self.resolve() == target:
            return change(original(self))
        return original(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
    _generator().check(REPO)


def _missing(_: bytes) -> bytes:
    raise FileNotFoundError("missing provenance sibling")


@pytest.mark.parametrize("relpath,change", [
    (lifetime.PAIR_LAYER2_RELPATH, lambda b: b.replace(b"Mizuno2025", b"Mizuno2024", 1)),
    (lifetime.PAIR_LAYER2_RELPATH, _missing),
    (lifetime.PAIR_LAYER1_RELPATH, lambda b: b.replace(b"0.99359799999999998", b"0.99359899999999998")),
    (lifetime.PAIR_LAYER1_RELPATH, lambda b: b.replace(b"Z A tau q free_tau", b"Z A q tau free_tau")),
    (lifetime.CONVENTION_RELPATH, lambda b: b.replace(b'"SOURCE_REPORTED_FROZEN"', b'"SOURCE_CONVERTED"')),
    ("docs/evaluated-bound-rates.md", lambda b: b.replace(b"NOT_ESTABLISHED", b"MEETS", 1)),
], ids=["tampered-provenance", "missing-provenance", "tampered-tuple", "reordered-columns",
        "tampered-convention", "tampered-page"])
def test_drill_the_check_refuses_a_tampered_or_missing_artifact(monkeypatch, relpath, change):
    from openmucf.g4 import spec

    refusals = (SystemExit, ValueError, FileNotFoundError, ev.EvaluationError, spec.G4DatFormatError)
    with pytest.raises(refusals):
        _check_with(monkeypatch, relpath, change)


def test_drill_a_pair_table_with_rows_out_of_order_is_refused():
    from openmucf.g4 import spec

    text = (REPO / lifetime.PAIR_LAYER1_RELPATH).read_text(encoding="ascii")
    row = next(line for line in text.splitlines() if line and not line.startswith("#"))
    later = row.replace("13 27 ", "13 28 ", 1)
    swapped = text.replace(row + "\n", later + "\n" + row + "\n").replace("Z:13-13", "Z:13-13")
    with pytest.raises(spec.G4DatFormatError):
        spec.parse(swapped)
