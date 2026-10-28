"""The precision and improvement decisions every evaluated muonic quantity in this package takes.

One implementation serves every dataset that decides a fixed delivered value against a reference region:
the evaluated bound-muon rate pair (:mod:`openmucf.g4.d1_evaluated`) and the measured oxide capture vectors
(:mod:`openmucf.g4.d2_measured`). A dataset prepares its own reference region; the verdict is taken here.

**Precision.** For a fixed delivered value ``d``, a reference region ``C`` and a loss ``loss(d, q)``, with
``L = inf_C loss`` and ``U = sup_C loss``:

* ``MEETS`` iff ``U <= budget``;
* ``DOES_NOT_MEET`` iff ``L > budget``;
* ``NOT_ESTABLISHED`` otherwise.

**Robust improvement** over a stock value ``s``: ``I = inf_C [loss(s, q) - loss(d, q)]``, the same ``q`` in
both losses. ``ESTABLISHED`` requires a strictly positive lower bound on ``I`` over the complete region;
otherwise ``NOT_ESTABLISHED``. Equality defeats strict improvement.

**Partial evidence is typed.** A region is often known only in part: some components are supported, one is
not. What such evidence can and cannot decide is fixed by the type the caller passes, never by a number:

* :class:`CompleteLoss` -- certified ``L`` and ``U`` over the complete region; the only precision evidence
  that can give ``MEETS`` or ``DOES_NOT_MEET``.
* :class:`PartialLoss` -- the loss at one admissible reference point (so ``L`` is at most that loss) and the
  supremum over a nested sub-region (so ``U`` is at least that supremum). A lower bound on ``U`` is never
  read as an upper bound on ``U``. When the nested supremum exceeds the budget, ``MEETS`` is excluded under
  every monotone completion of the region; when the point loss is within the budget, ``DOES_NOT_MEET`` is
  excluded. Partial evidence alone always gives ``NOT_ESTABLISHED``, with what it excluded recorded.
* :class:`CompleteMargin` -- a certified lower bound on ``I`` over the complete region.
* :class:`AdverseWitness` -- the improvement margin at one admissible reference point, nonpositive. It proves
  that robust improvement is not established before the rest of the region is known.
* :class:`IncompleteMargin` -- a region with named missing components: ``NOT_ESTABLISHED``.

**Numerical enclosure policy.** The comparisons here are exact on the numbers given. A caller hands over
bounds already moved outward (for a complete region) or inward (for a nested sub-region and an admissible
point) by its own numerical allowance; this module adds no tolerance and widens nothing, so a margin of
exactly zero, or a ``U`` exactly at the budget, is decided as the rule states. The interval functions
:func:`interval_decision` and :func:`interval_improvement` take the extremes of a monotone scalar map at the
interval ends, which is exact for the absolute and relative norms they offer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ADVERSE_WITNESS",
    "DOES_NOT_MEET",
    "DOES_NOT_MEET_EXCLUDED",
    "ESTABLISHED",
    "INCOMPLETE",
    "MEETS",
    "MEETS_EXCLUDED",
    "NOT_ESTABLISHED",
    "VACUOUS",
    "AdverseWitness",
    "CompleteLoss",
    "CompleteMargin",
    "DecisionError",
    "IncompleteMargin",
    "PartialLoss",
    "interval_decision",
    "interval_improvement",
    "precision",
    "robust_improvement",
]

#: The verdicts.
MEETS, DOES_NOT_MEET, NOT_ESTABLISHED, VACUOUS = "MEETS", "DOES_NOT_MEET", "NOT_ESTABLISHED", "VACUOUS"
ESTABLISHED = "ESTABLISHED"
#: What partial evidence can exclude, and why an improvement is not established.
MEETS_EXCLUDED = "MEETS_EXCLUDED_UNDER_MONOTONE_COMPLETION"
DOES_NOT_MEET_EXCLUDED = "DOES_NOT_MEET_EXCLUDED_BY_ADMISSIBLE_POINT"
ADVERSE_WITNESS = "NONPOSITIVE_MARGIN_AT_ADMISSIBLE_POINT"
INCOMPLETE = "REGION_INCOMPLETE"


class DecisionError(ValueError):
    """Evidence or an argument the decision rules refuse."""


def _real(x: Any, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise DecisionError(f"{where}: {x!r} is not a real number")
    value = float(x)
    if not math.isfinite(value):
        raise DecisionError(f"{where}: {x!r} is not finite")
    return value


def _budget(budget: Any) -> float:
    bud = _real(budget, "budget")
    if not bud > 0:
        raise DecisionError("the budget must be positive")
    return bud


# --------------------------------------------------------------------------------------------
# a scalar reference interval
# --------------------------------------------------------------------------------------------


def interval_decision(predicted: Any, low: Any, high: Any, budget: Any, norm: str) -> dict[str, Any]:
    """The precision decision of a delivered scalar against the reference interval ``[low, high]``.

    ``norm`` is ``"absolute"`` (``|d - q|``) or ``"relative"`` (``|d - q| / q``); both are monotone in ``q``
    on either side of ``d``, so ``U`` is at an end of the interval and ``L`` is zero when ``d`` lies inside it
    and at the nearer end otherwise. Only the central prediction enters: a prediction uncertainty cannot move
    a decision. An empty interval is ``VACUOUS``; a relative reference that reaches zero is non-informative.
    """
    y = _real(predicted, "prediction")
    lo, hi = _real(low, "reference"), _real(high, "reference")
    bud = _budget(budget)
    if not lo <= hi:
        return {"verdict": VACUOUS, "L": None, "U": None}
    if norm == "relative":
        if not lo > 0:
            return {"verdict": NOT_ESTABLISHED, "L": None, "U": None, "note": "non-informative reference"}

        def err(q: float) -> float:
            return abs(y - q) / q
    elif norm == "absolute":

        def err(q: float) -> float:
            return abs(y - q)
    else:
        raise DecisionError(f"norm {norm!r}")
    upper = max(err(lo), err(hi))
    lower = 0.0 if lo <= y <= hi else min(err(lo), err(hi))
    return {"verdict": precision(CompleteLoss(lower, upper), bud)["verdict"], "L": lower, "U": upper}


def interval_improvement(entry: Any, stock: Any, low: Any, high: Any) -> dict[str, Any]:
    """Robust improvement of an entry over a stock value for every ``q`` in ``[low, high]``, absolute norm.

    The advantage ``|stock - q| - |entry - q|`` is monotone between the two predictions and constant outside
    them, so its minimum over an interval is at an end."""
    e, s = _real(entry, "improvement"), _real(stock, "improvement")
    lo, hi = _real(low, "improvement"), _real(high, "improvement")
    if not lo <= hi:
        return {"verdict": VACUOUS, "min_advantage": None, "midpoint": None}
    adv = min(abs(s - q) - abs(e - q) for q in (lo, hi))
    return {
        "verdict": robust_improvement(CompleteMargin(adv))["verdict"],
        "min_advantage": adv,
        "midpoint": 0.5 * (e + s),
    }


# --------------------------------------------------------------------------------------------
# typed evidence
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CompleteLoss:
    """Certified extremes of the loss over the complete reference region: ``lower <= L``, ``U <= upper``."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lo, hi = _real(self.lower, "lower loss"), _real(self.upper, "upper loss")
        if lo < 0 or not lo <= hi:
            raise DecisionError(f"complete loss bounds [{lo!r}, {hi!r}] are not an ordered nonnegative pair")


@dataclass(frozen=True)
class PartialLoss:
    """Evidence about a region not known in full.

    ``point_loss``: the loss at one admissible reference point, an upper bound on ``L``; ``nested_upper``:
    the supremum of the loss over a nested sub-region, a lower bound on ``U``. Either may be ``None``."""

    point_loss: float | None
    nested_upper: float | None

    def __post_init__(self) -> None:
        for name in ("point_loss", "nested_upper"):
            v = getattr(self, name)
            if v is not None and _real(v, name) < 0:
                raise DecisionError(f"{name} {v!r} is negative")
        if self.point_loss is None and self.nested_upper is None:
            raise DecisionError("partial loss evidence carries no bound")


@dataclass(frozen=True)
class CompleteMargin:
    """A certified lower bound on the improvement margin over the complete region."""

    lower: float

    def __post_init__(self) -> None:
        _real(self.lower, "margin")


@dataclass(frozen=True)
class AdverseWitness:
    """The improvement margin at one admissible reference point; it must be nonpositive."""

    margin: float
    point: str

    def __post_init__(self) -> None:
        if _real(self.margin, "witness margin") > 0:
            raise DecisionError("a positive margin at one point is not an adverse witness")
        if not isinstance(self.point, str) or not self.point:
            raise DecisionError("an adverse witness names its point")


@dataclass(frozen=True)
class IncompleteMargin:
    """A region with named components that are not supported. ``nested_lower``, when given, is a lower bound
    on the margin over the supported sub-region only; it bounds nothing about the complete region."""

    missing: tuple[str, ...]
    nested_lower: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.missing, tuple)
            or not self.missing
            or not all(isinstance(m, str) and m for m in self.missing)
            or len(set(self.missing)) != len(self.missing)
        ):
            raise DecisionError("an incomplete region names each missing component once")
        if self.nested_lower is not None:
            _real(self.nested_lower, "nested margin")


# --------------------------------------------------------------------------------------------
# the verdicts
# --------------------------------------------------------------------------------------------


def precision(evidence: CompleteLoss | PartialLoss, budget: Any) -> dict[str, Any]:
    """The precision verdict from typed evidence. Only :class:`CompleteLoss` can give ``MEETS`` or
    ``DOES_NOT_MEET``; :class:`PartialLoss` gives ``NOT_ESTABLISHED`` with the verdicts it excludes."""
    bud = _budget(budget)
    if type(evidence) is CompleteLoss:
        if evidence.upper <= bud:
            verdict = MEETS
        elif evidence.lower > bud:
            verdict = DOES_NOT_MEET
        else:
            verdict = NOT_ESTABLISHED
        return {"verdict": verdict, "evidence": "complete", "L": evidence.lower, "U": evidence.upper}
    if type(evidence) is PartialLoss:
        excluded = []
        if evidence.nested_upper is not None and evidence.nested_upper > bud:
            excluded.append(MEETS_EXCLUDED)
        if evidence.point_loss is not None and evidence.point_loss <= bud:
            excluded.append(DOES_NOT_MEET_EXCLUDED)
        return {
            "verdict": NOT_ESTABLISHED,
            "evidence": "partial",
            "L_at_most": evidence.point_loss,
            "U_at_least": evidence.nested_upper,
            "excluded": excluded,
        }
    raise DecisionError(f"precision evidence of type {type(evidence).__name__}")


def robust_improvement(evidence: CompleteMargin | AdverseWitness | IncompleteMargin) -> dict[str, Any]:
    """The improvement verdict from typed evidence: ``ESTABLISHED`` only from a strictly positive
    :class:`CompleteMargin`."""
    if type(evidence) is CompleteMargin:
        return {
            "verdict": ESTABLISHED if evidence.lower > 0 else NOT_ESTABLISHED,
            "evidence": "complete",
            "margin_lower": evidence.lower,
        }
    if type(evidence) is AdverseWitness:
        return {
            "verdict": NOT_ESTABLISHED,
            "evidence": ADVERSE_WITNESS,
            "margin_at_point": evidence.margin,
            "point": evidence.point,
        }
    if type(evidence) is IncompleteMargin:
        return {
            "verdict": NOT_ESTABLISHED,
            "evidence": INCOMPLETE,
            "missing": list(evidence.missing),
            "nested_margin_lower": evidence.nested_lower,
        }
    raise DecisionError(f"improvement evidence of type {type(evidence).__name__}")
