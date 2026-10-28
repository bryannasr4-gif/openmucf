"""The shared precision and improvement decisions: the rules, the typed partial evidence, and the D1 record.

`openmucf/g4/science_decisions.py` takes every verdict; `openmucf/g4/d1_evaluated.py` and
`openmucf/g4/d2_measured.py` prepare their regions and call it. Each rule is drilled with a complete finite
positive control and the input that must flip it, and the committed D1 evaluation is re-decided through the
shared functions from its own committed numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib

import pytest

from openmucf.g4 import d1_evaluated as ev
from openmucf.g4 import science_decisions as sd

REPO = pathlib.Path(__file__).resolve().parents[1]
D1_EVALUATION = REPO / "data" / "g4" / "d1" / "bound_rate_evaluation.json"
SHARED_RECORD = REPO / "data" / "g4" / "d1" / "bound_decay_response_v1.json"

#: The aluminium oxide witness: the measured entry, the stock selector's value for the same material, and
#: their midpoint, where robust improvement changes sign.
AL_ENTRY = 47 / 122
AL_STOCK = 325 / 493
AL_MID = (AL_ENTRY + AL_STOCK) / 2


# --------------------------------------------------------------------------------------------
# the L/U rule on complete regions
# --------------------------------------------------------------------------------------------


def test_complete_region_verdicts_follow_l_and_u():
    """U = budget meets; L above the budget does not meet; a region containing the delivered value whose U
    exceeds the budget is not established -- never DOES_NOT_MEET, which needs L above the budget."""
    assert sd.precision(sd.CompleteLoss(0.0, 0.01), 0.01)["verdict"] == sd.MEETS
    assert sd.precision(sd.CompleteLoss(0.011, 0.02), 0.01)["verdict"] == sd.DOES_NOT_MEET
    centre = sd.precision(sd.CompleteLoss(0.0, 0.02), 0.01)
    assert centre["verdict"] == sd.NOT_ESTABLISHED
    assert (centre["L"], centre["U"]) == (0.0, 0.02)


def test_interval_decision_moves_with_the_region():
    """The same delivered value against a region whose U is the budget, then the region moved so that L is
    0.011, then a region around the value with U = 0.02."""
    assert sd.interval_decision(0.0, -0.01, 0.01, 0.01, "absolute")["verdict"] == sd.MEETS
    moved = sd.interval_decision(0.0, 0.011, 0.02, 0.01, "absolute")
    assert moved["verdict"] == sd.DOES_NOT_MEET and moved["L"] == 0.011
    wide = sd.interval_decision(0.0, -0.02, 0.02, 0.01, "absolute")
    assert wide["verdict"] == sd.NOT_ESTABLISHED and wide["L"] == 0.0


def test_bounded_measurement_model_has_power():
    """Y = t + E with E in [-0.002, 0.002] and C = [Y - 0.002, Y + 0.002] contains t. Over a deterministic
    grid: d = t meets 0.01; d = t + 0.05 has L >= 0.046 and does not meet; an entry at t beats a stock value
    at t + 0.10 by at least 0.092 throughout C."""
    for i in range(41):
        t = 0.05 + 0.02 * i
        for j in range(9):
            e = -0.002 + 0.0005 * j
            y = t + e
            lo, hi = y - 0.002, y + 0.002
            assert lo <= t <= hi
            assert sd.interval_decision(t, lo, hi, 0.01, "absolute")["verdict"] == sd.MEETS
            far = sd.interval_decision(t + 0.05, lo, hi, 0.01, "absolute")
            assert far["verdict"] == sd.DOES_NOT_MEET and far["L"] >= 0.046 - 1e-12
            better = sd.interval_improvement(t, t + 0.10, lo, hi)
            assert better["verdict"] == sd.ESTABLISHED and better["min_advantage"] >= 0.092 - 1e-12


# --------------------------------------------------------------------------------------------
# strict improvement
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shift", "verdict"),
    [(-1e-6, sd.ESTABLISHED), (0.0, sd.NOT_ESTABLISHED), (1e-6, sd.NOT_ESTABLISHED)],
)
def test_improvement_is_strict_at_the_aluminium_midpoint(shift, verdict):
    """Complete singleton references below, at and above the midpoint of the aluminium entry and the stock
    value give margins +2e-6, 0 and -2e-6; only the positive margin establishes improvement."""
    q = AL_MID + shift
    out = sd.interval_improvement(AL_ENTRY, AL_STOCK, q, q)
    assert out["verdict"] == verdict
    assert math.isclose(out["min_advantage"], -2 * shift, abs_tol=1e-15)
    assert sd.robust_improvement(sd.CompleteMargin(out["min_advantage"]))["verdict"] == verdict


def test_zero_margin_does_not_establish():
    assert sd.robust_improvement(sd.CompleteMargin(0.0))["verdict"] == sd.NOT_ESTABLISHED
    assert sd.robust_improvement(sd.CompleteMargin(-0.0))["verdict"] == sd.NOT_ESTABLISHED
    assert sd.robust_improvement(sd.CompleteMargin(5e-324))["verdict"] == sd.ESTABLISHED


# --------------------------------------------------------------------------------------------
# partial evidence
# --------------------------------------------------------------------------------------------


def test_partial_evidence_excludes_meets_without_completing_the_region():
    """An admissible point at zero loss and a nested sub-region whose supremum exceeds the budget exclude both
    decided verdicts; the result is NOT_ESTABLISHED, carried with what it excluded."""
    out = sd.precision(sd.PartialLoss(point_loss=0.0, nested_upper=0.0144), 0.01)
    assert out["verdict"] == sd.NOT_ESTABLISHED
    assert out["excluded"] == [sd.MEETS_EXCLUDED, sd.DOES_NOT_MEET_EXCLUDED]
    assert (out["L_at_most"], out["U_at_least"]) == (0.0, 0.0144)


def test_partial_evidence_never_decides():
    """A nested supremum within the budget is a lower bound on U, not an upper one: no MEETS. A point loss
    above the budget is an upper bound on L, not a lower one: no DOES_NOT_MEET."""
    small = sd.precision(sd.PartialLoss(point_loss=0.0, nested_upper=0.005), 0.01)
    assert small["verdict"] == sd.NOT_ESTABLISHED and small["excluded"] == [sd.DOES_NOT_MEET_EXCLUDED]
    far = sd.precision(sd.PartialLoss(point_loss=0.5, nested_upper=0.6), 0.01)
    assert far["verdict"] == sd.NOT_ESTABLISHED and far["excluded"] == [sd.MEETS_EXCLUDED]
    at_budget = sd.precision(sd.PartialLoss(point_loss=None, nested_upper=0.01), 0.01)
    assert at_budget["excluded"] == []


def test_an_adverse_witness_settles_improvement_before_the_region_is_complete():
    out = sd.robust_improvement(sd.AdverseWitness(-0.004, "core upper end, run band, origin"))
    assert out["verdict"] == sd.NOT_ESTABLISHED and out["evidence"] == sd.ADVERSE_WITNESS
    assert sd.robust_improvement(sd.AdverseWitness(0.0, "midpoint"))["verdict"] == sd.NOT_ESTABLISHED
    with pytest.raises(sd.DecisionError, match="not an adverse witness"):
        sd.AdverseWitness(1e-9, "a favourable point")


def test_an_incomplete_region_is_not_established_whatever_its_nested_margin():
    out = sd.robust_improvement(sd.IncompleteMargin(("E2 response not validated",), nested_lower=0.3))
    assert out["verdict"] == sd.NOT_ESTABLISHED and out["evidence"] == sd.INCOMPLETE
    assert out["missing"] == ["E2 response not validated"]


@pytest.mark.parametrize(
    "build",
    [
        lambda: sd.CompleteLoss(math.nan, 0.01),
        lambda: sd.CompleteLoss(0.0, math.inf),
        lambda: sd.CompleteLoss(0.0, -math.inf),
        lambda: sd.CompleteLoss(True, 0.01),
        lambda: sd.CompleteLoss("0", 0.01),
        lambda: sd.CompleteLoss(0.02, 0.01),
        lambda: sd.CompleteLoss(-0.01, 0.01),
        lambda: sd.PartialLoss(None, None),
        lambda: sd.PartialLoss(math.nan, 0.02),
        lambda: sd.PartialLoss(0.0, -0.02),
        lambda: sd.CompleteMargin(math.nan),
        lambda: sd.CompleteMargin(math.inf),
        lambda: sd.CompleteMargin(None),
        lambda: sd.AdverseWitness(-0.1, ""),
        lambda: sd.AdverseWitness(math.nan, "x"),
        lambda: sd.IncompleteMargin(()),
        lambda: sd.IncompleteMargin(("E2", "E2")),
        lambda: sd.IncompleteMargin(["E2"]),  # type: ignore[arg-type]
        lambda: sd.IncompleteMargin(("",)),
        lambda: sd.IncompleteMargin(("E2",), nested_lower=math.nan),
    ],
)
def test_malformed_evidence_is_refused(build):
    with pytest.raises(sd.DecisionError):
        build()


@pytest.mark.parametrize("evidence", [{"lower": 1.0}, (0.0, 0.01), 0.5, None])
def test_untyped_evidence_is_refused(evidence):
    with pytest.raises(sd.DecisionError):
        sd.precision(evidence, 0.01)  # type: ignore[arg-type]
    with pytest.raises(sd.DecisionError):
        sd.robust_improvement(evidence)  # type: ignore[arg-type]


@pytest.mark.parametrize("budget", [0.0, -0.01, math.nan, math.inf, True, "0.01", None])
def test_a_budget_must_be_finite_and_positive(budget):
    with pytest.raises(sd.DecisionError):
        sd.precision(sd.CompleteLoss(0.0, 0.001), budget)


# --------------------------------------------------------------------------------------------
# the committed D1 record through the shared functions
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def d1_record() -> dict:
    return json.loads(D1_EVALUATION.read_bytes())


def test_d1_delegates_to_the_shared_functions(monkeypatch):
    """`d1_evaluated.decision` and `.improvement` call the shared functions: replacing them changes D1."""
    calls = []

    def spy_decision(*args):
        calls.append(("decision", args))
        return {"verdict": "SPY"}

    def spy_improvement(*args):
        calls.append(("improvement", args))
        return {"verdict": "SPY"}

    monkeypatch.setattr(sd, "interval_decision", spy_decision)
    monkeypatch.setattr(sd, "interval_improvement", spy_improvement)
    assert ev.decision(1.0, 0.99, 1.01, 0.01, "relative") == {"verdict": "SPY"}
    assert ev.improvement(1.0, 2.0, 0.5, 1.4) == {"verdict": "SPY"}
    assert [c[0] for c in calls] == ["decision", "improvement"]


def test_d1_refusals_keep_their_type():
    with pytest.raises(ev.EvaluationError):
        ev.decision(math.nan, 0.99, 1.01, 0.01, "relative")
    with pytest.raises(ev.EvaluationError):
        ev.improvement(1.0, 2.0, math.inf, 1.4)


def test_the_committed_d1_decisions_are_the_shared_rule_of_their_committed_numbers(d1_record):
    """Every committed D1 decision, improvement and power verdict, re-decided by the shared functions from
    the committed prediction and reference range, is the committed field exactly: late lifetime and decay
    rate MEETS, stopped mean and capture probability NOT_ESTABLISHED, atomic-route late-lifetime improvement
    ESTABLISHED and no other improvement."""
    ranges = d1_record["ranges"]
    for name, committed in d1_record["decisions"].items():
        low, high = ranges[name]["low"], ranges[name]["high"]
        again = sd.interval_decision(
            committed["prediction"], low, high, committed["budget"], committed["norm"]
        )
        assert again == {k: committed[k] for k in ("verdict", "L", "U")}, name
    verdicts = {k: v["verdict"] for k, v in d1_record["decisions"].items()}
    assert verdicts == {
        "late_lifetime": sd.MEETS,
        "decay_rate": sd.MEETS,
        "stopped_mean": sd.NOT_ESTABLISHED,
        "capture_probability": sd.NOT_ESTABLISHED,
    }
    predictions = {k: v["prediction"] for k, v in d1_record["decisions"].items()}
    established = []
    for route, stock in d1_record["stock"].items():
        stock_values = {
            "late_lifetime": stock["lifetime_ns"],
            "stopped_mean": stock["lifetime_ns"],
            "capture_probability": stock["capture_probability"],
        }
        for name, committed in stock["improvement"].items():
            again = sd.interval_improvement(
                predictions[name], stock_values[name], ranges[name]["low"], ranges[name]["high"]
            )
            assert again == committed, (route, name)
            if again["verdict"] == sd.ESTABLISHED:
                established.append((route, name))
    assert established == [("atomic", "late_lifetime")]


def test_the_d1_record_and_the_shared_record_are_the_ones_d2_consumes():
    """The D1 evaluation and the shared convention record are byte-identical to the committed outputs this
    package's D2 evaluation was decided against; the record's payload digest is its recomputation."""
    assert hashlib.sha256(D1_EVALUATION.read_bytes()).hexdigest() == (
        "8e2e892e478f0b97515688f5e8689efd6c5bc93295c09b54aaf682d9bb4e00b9"
    )
    record = json.loads(SHARED_RECORD.read_bytes())
    assert (
        ev.payload_digest(record["payload"])
        == record["payload_sha256"]
        == ("0bb824713a28b1a5e28062d9c366ef8f0c11ade8267235762ca43d48be5ddef8")
    )
