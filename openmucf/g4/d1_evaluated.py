"""An evaluated bound-muon rate pair: one lifetime primitive and one decay factor, given to Geant4 together.

A negative muon in the 1s state of a muonic atom disappears by nuclear capture (rate ``c``) or by decay in
orbit (rate ``d``). The pair delivered here is formed from one primitive tuple ``(tau, q, free_tau)``:

    d = q / free_tau,   c = 1 / tau - d,   Pcap = c / (c + d) = 1 - tau * d,

so the two rates come from one realization of the primitives and ``c + d = 1 / tau`` holds exactly. ``tau``
is a measured late-time lifetime, ``q`` the bound-to-free decay factor and ``free_tau`` the empirical
free-muon lifetime. This module is a leaf: it reads no file itself. :mod:`openmucf.g4.d1_lifetime` loads
the committed inputs and the harvested oracle and delegates the evaluation to it.

What is computed, and what it is not:

* the decay factor assembly ``Q = Q0 + (Z alpha)^2 eta + dQ_U`` (finite-charge static value, adopted
  linear recoil, signed point-Uehling kinetic term), a nine-item diagnostic ledger and the frozen
  evaluator region ``Q_a +/- W_Q`` with ``W_Q = max_k |Q_a - Q_k| + sum |ledger|``;
* the late-slope mapping enclosure of a scalar exponential fitted from ``t0`` to a two-state hyperfine
  spectrum, and the late-lifetime reference interval it gives;
* the stopped mean and capture probability of the exact two-state model over the frozen product of
  regions, and the MEETS / DOES_NOT_MEET / NOT_ESTABLISHED decisions of fixed budgets against it;
* robust improvement over Geant4's stock pair, per route;
* first-order and exact covariance views, the per-draw closure invariant, consequential sensitivities;
* a descriptive census of the deferred oxygen and carbon lifetime records.

The regions are evaluator-assessed (EVALUATOR_TYPE_B): they are not confidence intervals, and a MEETS
decision is conditional on them. The late-lifetime decision has no validation power: the delivered
lifetime is the source's own value. Nothing here is an independent measurement of the capture branching.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

import numpy as np

INPUTS_SCHEMA = "openmucf.bound-rate-inputs/1"
EVALUATION_SCHEMA = "openmucf.bound-rate-evaluation/1"
CONVENTION_SCHEMA = "openmucf.bound-decay-response/1"
CONVENTION_ID = "bound-decay-response-v1"
PAIR_PROFILE = "evaluated-bound"
PAIR_TABLE = "bound_rate_primitives"
PAIR_COLUMNS = ("Z", "A", "tau", "q", "free_tau")
PAIR_UNITS = "tau=ns q=1 free_tau=ns"
#: The decision labels and the improvement labels.
MEETS, DOES_NOT_MEET, NOT_ESTABLISHED, VACUOUS = "MEETS", "DOES_NOT_MEET", "NOT_ESTABLISHED", "VACUOUS"
ESTABLISHED = "ESTABLISHED"
REGION_LABEL = "EVALUATOR_TYPE_B"
#: The roster statuses the evaluation knows.
ROSTER_IN = "IN"
ROSTER_DEFERRED = ("DEFERRED_RUNTIME_SOURCE_MAPPING", "DEFERRED_RUNTIME_MIXTURE_ESTIMAND")
#: Implementation tolerances: rates and probabilities, and deterministic lifetimes (ns).
REL_TOL = 1e-12
NS_TOL = 1e-9
#: The decisions, in order, with their norm.
DECISIONS = (
    ("late_lifetime", "relative"),
    ("stopped_mean", "relative"),
    ("capture_probability", "absolute"),
    ("decay_rate", "relative"),
)
#: The components of the frozen reference product, in the order the reference point takes them.
COMPONENTS = ("tau_late", "q", "free_tau", "contrast", "transition")


class EvaluationError(ValueError):
    """An input, a record or a request this evaluation refuses."""


# --------------------------------------------------------------------------------------------
# typed inputs
# --------------------------------------------------------------------------------------------


def _decimal(value: Any, where: str) -> Decimal:
    """A decimal string: finite, non-empty, not a JSON number (whose binary value is not the printed one)."""
    if type(value) is not str or not value.strip() or value != value.strip():
        raise EvaluationError(f"{where}: expected a decimal string, got {value!r}")
    try:
        out = Decimal(value)
    except InvalidOperation:
        raise EvaluationError(f"{where}: {value!r} is not a decimal") from None
    if not out.is_finite():
        raise EvaluationError(f"{where}: {value!r} is not finite")
    return out


def _fraction(value: Any, where: str) -> Fraction:
    """A decimal or ``p/q`` string as an exact rational."""
    if type(value) is str and "/" in value:
        parts = value.split("/")
        if len(parts) != 2:
            raise EvaluationError(f"{where}: {value!r} is not p/q")
        num, den = (_decimal(p, where) for p in parts)
        if den == 0 or num != num.to_integral_value() or den != den.to_integral_value():
            raise EvaluationError(f"{where}: {value!r} is not a ratio of integers")
        return Fraction(int(num), int(den))
    return Fraction(_decimal(value, where))


def _int(value: Any, where: str) -> int:
    if type(value) is not int:
        raise EvaluationError(f"{where}: expected an integer, got {value!r}")
    return value


def _text(value: Any, where: str, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise EvaluationError(f"{where}: expected text, got {value!r}")
    return value


def _keys(obj: Any, required: Sequence[str], where: str) -> Mapping[str, Any]:
    """An object with exactly these keys: an unknown or missing key is refused."""
    if not isinstance(obj, dict):
        raise EvaluationError(f"{where}: expected an object")
    have, want = set(obj), set(required)
    if have != want:
        raise EvaluationError(
            f"{where}: keys differ: missing {sorted(want - have)}, unknown {sorted(have - want)}"
        )
    return obj


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise EvaluationError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise EvaluationError(f"non-finite JSON constant {name}")


def parse_json(raw: bytes) -> Any:
    """JSON with duplicate keys and NaN/Infinity refused; CR refused."""
    if b"\r" in raw:
        raise EvaluationError("contains CR")
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"not JSON: {exc}") from None


@dataclass(frozen=True)
class Prescription:
    id: str
    kind: str
    value: Decimal | None
    locator: str


@dataclass(frozen=True)
class Nucleus:
    name: str
    z: int
    a: int
    roster: str
    static_q: Decimal
    prescriptions: tuple[Prescription, ...]
    excluded: tuple[tuple[str, Decimal, str], ...]


@dataclass(frozen=True)
class Inputs:
    """The committed primitives, typed. Nothing derived is stored here."""

    raw: Mapping[str, Any]
    evaluation_id: str
    alpha_inverse_uehling: Decimal
    mass_ratio: Decimal
    upper_weight: Fraction
    tau: Decimal
    tau_u: Decimal
    tau_k: Decimal
    components: Mapping[str, Decimal]
    t0: Decimal
    window_upper: Decimal
    free_tau: Decimal
    free_tau_u: Decimal
    free_tau_k: Decimal
    nuclei: Mapping[str, Nucleus]
    ledger: Mapping[str, Decimal]
    q_decimals: int
    amplitude_center: Decimal
    amplitude_u: Decimal
    amplitude_k: Decimal
    transition_center: Decimal
    transition_u: Decimal
    transition_k: Decimal
    transition_crosscheck: tuple[tuple[Decimal, Decimal], ...]
    mean_matched_amplitude: Decimal
    mean_matched_transition: Decimal
    budgets: Mapping[str, Decimal]
    consequential_fraction: Decimal
    mapping_step: Decimal
    q_mutant: Decimal
    mean_mutant_factor: Decimal
    census: Mapping[str, Any]


_TOP = (
    "schema",
    "evaluation_id",
    "pair_profile",
    "constants",
    "lifetime",
    "free_lifetime",
    "nuclei",
    "q_ledger",
    "q_serialization_decimals",
    "hyperfine",
    "budgets",
    "consequential_fraction",
    "mapping_outward_step",
    "power",
    "census",
)
_CONSTANTS = (
    "oracle",
    "uehling_alpha_inverse",
    "uehling_alpha_locator",
    "electron_muon_mass_ratio",
    "electron_muon_mass_ratio_locator",
    "upper_state_weight",
    "upper_state_weight_locator",
)
_LIFETIME = (
    "nucleus",
    "tau_ns",
    "standard_uncertainty_ns",
    "coverage_k",
    "components_ns",
    "components_note",
    "source",
    "copy_read",
    "locator",
    "time_origin_ns",
    "time_origin_locator",
    "window_upper_ns",
    "development_comparisons",
)
_FREE = ("tau_ns", "standard_uncertainty_ns", "coverage_k", "locator")
_NUCLEUS = ("Z", "A", "roster", "static_q", "static_q_locator", "prescriptions", "excluded_prescriptions")
_PRESCRIPTION = ("id", "kind", "value", "locator")
_PRESCRIPTION_KINDS = ("value", "divide_by_f", "leading_formula", "recovered_natc")
_EXCLUDED = ("id", "value", "reason")
LEDGER_ITEMS = (
    "printing_half_unit",
    "termination_fraction",
    "binding_recoil_cubic",
    "recoil_quadratic",
    "kinetic_quadratic",
    "historical_recoil_contrast",
    "relativistic_uehling",
    "one_loop",
    "two_loop",
)
_HYPERFINE = (
    "nucleus",
    "amplitude_center",
    "amplitude_standard_uncertainty",
    "amplitude_coverage_k",
    "amplitude_definition",
    "amplitude_locator",
    "transition_center_per_us",
    "transition_standard_uncertainty_per_us",
    "transition_coverage_k",
    "transition_locator",
    "transition_crosscheck_per_us",
    "transition_crosscheck_locator",
    "registered_discrepancy",
    "mean_matched_amplitude",
    "mean_matched_transition_per_us",
)
_BUDGETS = tuple(name for name, _ in DECISIONS)
_POWER = ("q_mutant", "mean_mutant_factor")
_CENSUS = ("source", "O16", "C12", "minor_isotope")
_NUCLEUS_NAMES = ("Al27", "O16", "C12")


def _positive(value: Decimal, where: str) -> Decimal:
    if not value > 0:
        raise EvaluationError(f"{where}: {value} is not positive")
    return value


def load_inputs(raw: bytes) -> Inputs:
    """The committed inputs, with every key, type and domain checked."""
    obj = _keys(parse_json(raw), _TOP, "inputs")
    if obj["schema"] != INPUTS_SCHEMA:
        raise EvaluationError(f"inputs: schema {obj['schema']!r}")
    if obj["pair_profile"] != PAIR_PROFILE:
        raise EvaluationError(f"inputs: pair profile {obj['pair_profile']!r}")
    constants = _keys(obj["constants"], _CONSTANTS, "constants")
    for key in (
        "oracle",
        "uehling_alpha_locator",
        "electron_muon_mass_ratio_locator",
        "upper_state_weight_locator",
    ):
        _text(constants[key], f"constants.{key}")
    weight = _fraction(constants["upper_state_weight"], "constants.upper_state_weight")
    if not 0 < weight < 1:
        raise EvaluationError("constants.upper_state_weight is not in (0, 1)")
    life = _keys(obj["lifetime"], _LIFETIME, "lifetime")
    if life["nucleus"] != "Al27":
        raise EvaluationError(f"lifetime: nucleus {life['nucleus']!r} is not the admitted row")
    for key in (
        "components_note",
        "source",
        "copy_read",
        "locator",
        "time_origin_locator",
        "development_comparisons",
    ):
        _text(life[key], f"lifetime.{key}")
    components = _keys(life["components_ns"], ("fit", "cno", "background", "range"), "lifetime.components_ns")
    free = _keys(obj["free_lifetime"], _FREE, "free_lifetime")
    _text(free["locator"], "free_lifetime.locator")
    nuclei: dict[str, Nucleus] = {}
    raw_nuclei = _keys(obj["nuclei"], _NUCLEUS_NAMES, "nuclei")
    for name in _NUCLEUS_NAMES:
        n = _keys(raw_nuclei[name], _NUCLEUS, f"nuclei.{name}")
        z, a = _int(n["Z"], f"{name}.Z"), _int(n["A"], f"{name}.A")
        if not 1 <= z <= a:
            raise EvaluationError(f"{name}: Z={z} A={a}")
        roster = _text(n["roster"], f"{name}.roster")
        if roster != ROSTER_IN and roster not in ROSTER_DEFERRED:
            raise EvaluationError(f"{name}: roster status {roster!r} is not one this evaluation knows")
        _text(n["static_q_locator"], f"{name}.static_q_locator")
        if not isinstance(n["prescriptions"], list) or not n["prescriptions"]:
            raise EvaluationError(f"{name}: no prescription")
        members = []
        for i, item in enumerate(n["prescriptions"]):
            p = _keys(item, _PRESCRIPTION, f"{name}.prescriptions[{i}]")
            kind = _text(p["kind"], f"{name}.prescriptions[{i}].kind")
            if kind not in _PRESCRIPTION_KINDS:
                raise EvaluationError(f"{name}: prescription kind {kind!r}")
            value = (
                None
                if kind == "leading_formula"
                else _decimal(p["value"], f"{name}.prescriptions[{i}].value")
            )
            if kind == "leading_formula" and p["value"] != "":
                raise EvaluationError(f"{name}: the leading formula carries no value")
            members.append(
                Prescription(
                    _text(p["id"], f"{name}.id"), kind, value, _text(p["locator"], f"{name}.locator")
                )
            )
        if len({m.id for m in members}) != len(members):
            raise EvaluationError(f"{name}: duplicate prescription id")
        excluded = []
        if not isinstance(n["excluded_prescriptions"], list):
            raise EvaluationError(f"{name}: excluded prescriptions must be a list")
        for i, item in enumerate(n["excluded_prescriptions"]):
            e = _keys(item, _EXCLUDED, f"{name}.excluded[{i}]")
            excluded.append(
                (
                    _text(e["id"], f"{name}.excluded.id"),
                    _decimal(e["value"], f"{name}.excluded.value"),
                    _text(e["reason"], f"{name}.excluded.reason"),
                )
            )
        nuclei[name] = Nucleus(
            name,
            z,
            a,
            roster,
            _positive(_decimal(n["static_q"], f"{name}.static_q"), name),
            tuple(members),
            tuple(excluded),
        )
    if nuclei["Al27"].roster != ROSTER_IN or (nuclei["Al27"].z, nuclei["Al27"].a) != (13, 27):
        raise EvaluationError("Al27 must be the admitted (13, 27) row")
    for name in ("O16", "C12"):
        if nuclei[name].roster == ROSTER_IN:
            raise EvaluationError(
                f"{name}: a deferred candidate cannot be admitted by changing its status string"
            )
    ledger = _keys(obj["q_ledger"], LEDGER_ITEMS, "q_ledger")
    decimals = _int(obj["q_serialization_decimals"], "q_serialization_decimals")
    if not 1 <= decimals <= 12:
        raise EvaluationError("q_serialization_decimals out of range")
    hf = _keys(obj["hyperfine"], _HYPERFINE, "hyperfine")
    for key in (
        "amplitude_definition",
        "amplitude_locator",
        "transition_locator",
        "transition_crosscheck_locator",
        "registered_discrepancy",
    ):
        _text(hf[key], f"hyperfine.{key}")
    if hf["nucleus"] != "Al27":
        raise EvaluationError("hyperfine: nucleus")
    crosscheck = hf["transition_crosscheck_per_us"]
    if not isinstance(crosscheck, list) or not crosscheck:
        raise EvaluationError("hyperfine: transition cross-check")
    pairs = []
    for i, item in enumerate(crosscheck):
        if not isinstance(item, list) or len(item) != 2:
            raise EvaluationError(f"hyperfine: cross-check {i}")
        pairs.append(
            (
                _positive(_decimal(item[0], "crosscheck"), "crosscheck"),
                _positive(_decimal(item[1], "crosscheck"), "crosscheck"),
            )
        )
    budgets = _keys(obj["budgets"], _BUDGETS, "budgets")
    power = _keys(obj["power"], _POWER, "power")
    # The stated standard uncertainty must cover the quadrature of its stated components: a total that omits a
    # stated calibration or background term is refused even when every central value is exact.
    u_total = _positive(_decimal(life["standard_uncertainty_ns"], "lifetime.u"), "tau u")
    quadrature = sum(
        (_positive(_decimal(v, f"component {k}"), k) ** 2 for k, v in components.items()), Decimal(0)
    ).sqrt()
    if u_total < quadrature:
        raise EvaluationError(
            f"lifetime: the standard uncertainty {u_total} ns is below the quadrature {quadrature:.6f} ns "
            "of its "
            "stated components"
        )
    # The frozen regions must stay inside the positive two-state model.
    h_low = _decimal(hf["transition_center_per_us"], "h") - _decimal(
        hf["transition_coverage_k"], "h k"
    ) * _decimal(hf["transition_standard_uncertainty_per_us"], "h u")
    if not h_low > 0:
        raise EvaluationError("hyperfine: the transition-rate region reaches zero")
    a_high = _decimal(hf["amplitude_center"], "a") + _decimal(hf["amplitude_coverage_k"], "a k") * _decimal(
        hf["amplitude_standard_uncertainty"], "a u"
    )
    if not Fraction(a_high) / weight < 1:
        raise EvaluationError("hyperfine: the contrast region reaches a negative upper-state capture rate")
    census = _keys(obj["census"], _CENSUS, "census")
    _text(census["source"], "census.source")
    return Inputs(
        raw=obj,
        evaluation_id=_text(obj["evaluation_id"], "evaluation_id"),
        alpha_inverse_uehling=_positive(_decimal(constants["uehling_alpha_inverse"], "alpha"), "alpha"),
        mass_ratio=_positive(_decimal(constants["electron_muon_mass_ratio"], "mass ratio"), "mass ratio"),
        upper_weight=weight,
        tau=_positive(_decimal(life["tau_ns"], "lifetime.tau_ns"), "tau"),
        tau_u=_positive(_decimal(life["standard_uncertainty_ns"], "lifetime.u"), "tau u"),
        tau_k=_positive(_decimal(life["coverage_k"], "lifetime.k"), "tau k"),
        components={k: _positive(_decimal(v, f"component {k}"), k) for k, v in components.items()},
        t0=_positive(_decimal(life["time_origin_ns"], "t0"), "t0"),
        window_upper=_positive(_decimal(life["window_upper_ns"], "window"), "window"),
        free_tau=_positive(_decimal(free["tau_ns"], "free_tau"), "free_tau"),
        free_tau_u=_positive(_decimal(free["standard_uncertainty_ns"], "free_tau u"), "free_tau u"),
        free_tau_k=_positive(_decimal(free["coverage_k"], "free_tau k"), "free_tau k"),
        nuclei=nuclei,
        ledger={k: _positive(_decimal(v, f"ledger {k}"), k) for k, v in ledger.items()},
        q_decimals=decimals,
        amplitude_center=_decimal(hf["amplitude_center"], "amplitude_center"),
        amplitude_u=_positive(_decimal(hf["amplitude_standard_uncertainty"], "amplitude u"), "amplitude u"),
        amplitude_k=_positive(_decimal(hf["amplitude_coverage_k"], "amplitude k"), "amplitude k"),
        transition_center=_positive(_decimal(hf["transition_center_per_us"], "h"), "h"),
        transition_u=_positive(_decimal(hf["transition_standard_uncertainty_per_us"], "h u"), "h u"),
        transition_k=_positive(_decimal(hf["transition_coverage_k"], "h k"), "h k"),
        transition_crosscheck=tuple(pairs),
        mean_matched_amplitude=_decimal(hf["mean_matched_amplitude"], "mean matched amplitude"),
        mean_matched_transition=_positive(_decimal(hf["mean_matched_transition_per_us"], "mm h"), "mm h"),
        budgets={k: _positive(_decimal(v, f"budget {k}"), k) for k, v in budgets.items()},
        consequential_fraction=_positive(
            _decimal(obj["consequential_fraction"], "consequential"), "consequential"
        ),
        mapping_step=_positive(_decimal(obj["mapping_outward_step"], "mapping step"), "mapping step"),
        q_mutant=_positive(_decimal(power["q_mutant"], "q mutant"), "q mutant"),
        mean_mutant_factor=_positive(_decimal(power["mean_mutant_factor"], "mean mutant"), "mean mutant"),
        census=census,
    )


@dataclass(frozen=True)
class OracleInputs:
    """What the harvested Geant4 oracle supplies: constants and nuclear masses, MeV."""

    alpha: float
    mu_mass: float
    masses: Mapping[tuple[int, int], float]
    stock_capture: float  # ns^-1, the compiled capture table for (13, 27) with the data option off
    stock_decay_atomic: float  # ns^-1, G4MuonicAtomHelper::GetMuonDecayRate(13)
    stock_decay_stopping: float  # ns^-1, G4MuonMinusBoundDecay::GetMuonDecayRate(13, 27, ...)


# --------------------------------------------------------------------------------------------
# the decay factor
# --------------------------------------------------------------------------------------------


def _legendre_nodes(n: int) -> tuple[tuple[float, float], ...]:
    """Gauss-Legendre nodes and weights on [-1, 1], by Newton iteration on P_n."""
    out = []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1.0)
            step = p1 / dp
            x -= step
            if abs(step) < 1e-16:
                break
        p0, p1 = 1.0, x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1.0)
        out.append((x, 2.0 / ((1.0 - x * x) * dp * dp)))
    return tuple(out)


_NODES = _legendre_nodes(48)


def _integrate(f: Callable[[float], float], breaks: Sequence[float]) -> float:
    total = 0.0
    for lo, hi in itertools.pairwise(breaks):
        half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
        total += half * sum(w * f(mid + half * x) for x, w in _NODES)
    return total


def uehling_kinetic(z: int, alpha_inverse: float, mass_ratio: float) -> float:
    """The signed point-Uehling kinetic contribution to Q for a nonrelativistic Coulomb 1s muon:
    ``-2 alpha (Z alpha)^2 / (3 pi) * IQ(kappa)``, ``kappa = (m_e/m_mu) / (Z alpha)``, with
    ``IQ = int_0^{pi/2} cos^2 t (1 + sin^2 t / 2) sin t (sin t + 3 kappa) / (sin t + kappa)^3 dt``.
    The potential is that of Szafron and Czarnecki, Phys. Rev. D 94, 051301 (2016), section II.2."""
    alpha = 1.0 / alpha_inverse
    zz = z * alpha
    kappa = mass_ratio / zz

    def integrand(t: float) -> float:
        s, c = math.sin(t), math.cos(t)
        return c * c * (1.0 + 0.5 * s * s) * s * (s + 3.0 * kappa) / (s + kappa) ** 3

    # graded breaks resolve the scale kappa near t = 0
    breaks = [0.0] + [kappa * 2.0**k for k in range(-4, 6) if kappa * 2.0**k < math.pi / 2] + [math.pi / 2]
    return -2.0 * alpha * zz * zz / (3.0 * math.pi) * _integrate(integrand, breaks)


def f_factor(x: float) -> float:
    """The finite-electron-mass free tree width factor ``1 - 8x^2 - 24x^4 ln x + 8x^6 - x^8``."""
    return 1.0 - 8.0 * x**2 - 24.0 * x**4 * math.log(x) + 8.0 * x**6 - x**8


@dataclass(frozen=True)
class QRecord:
    nucleus: str
    z: int
    a: int
    roster: str
    static_q: float
    eta: float
    recoil: float
    uehling: float
    assembly: float
    leading_formula: float
    ledger: Mapping[str, float]
    ledger_sum: float
    members: Mapping[str, float]
    spread: float
    half_width: float
    region: tuple[float, float]


def q_record(inputs: Inputs, oracle: OracleInputs, name: str, natc: CensusRow | None = None) -> QRecord:
    """The decay-factor assembly, ledger, prescription members and frozen region of one nucleus.

    ``natc`` is the natural-carbon primary row a ``recovered_natc`` member is recovered from."""
    n = inputs.nuclei[name]
    alpha = oracle.alpha
    if (n.z, n.a) not in oracle.masses:
        raise EvaluationError(f"{name}: no archived nuclear mass for ({n.z}, {n.a})")
    mass = oracle.masses[(n.z, n.a)]
    z = n.z * alpha
    z2 = z * z
    eta = oracle.mu_mass / mass
    recoil = z2 * eta
    uehling = uehling_kinetic(n.z, float(inputs.alpha_inverse_uehling), float(inputs.mass_ratio))
    q0 = float(n.static_q)
    assembly = q0 + recoil + uehling
    leading = 1.0 - z2 * (0.5 - eta)
    led = inputs.ledger
    a_pi = alpha / math.pi
    ledger = {
        "printing_half_unit": float(led["printing_half_unit"]) / math.sqrt(3.0),
        "termination_fraction": float(led["termination_fraction"]) * q0,
        "binding_recoil_cubic": float(led["binding_recoil_cubic"]) * z**3 * eta,
        "recoil_quadratic": float(led["recoil_quadratic"]) * z2 * eta**2,
        "kinetic_quadratic": float(led["kinetic_quadratic"]) * z2 * eta**2,
        "historical_recoil_contrast": float(led["historical_recoil_contrast"]) * z2 * eta,
        "relativistic_uehling": float(led["relativistic_uehling"]) * abs(uehling) * z2,
        "one_loop": float(led["one_loop"]) * a_pi * z2,
        "two_loop": float(led["two_loop"]) * a_pi**2 * z2,
    }
    ledger_sum = sum(abs(v) for v in ledger.values())
    f = f_factor(float(inputs.mass_ratio))
    members: dict[str, float] = {}
    for p in n.prescriptions:
        if p.kind == "value":
            assert p.value is not None
            members[p.id] = float(p.value)
        elif p.kind == "divide_by_f":
            assert p.value is not None
            members[p.id] = float(p.value) / f
        elif p.kind == "leading_formula":
            members[p.id] = leading
        else:
            assert p.value is not None
            if natc is None:
                raise EvaluationError(f"{name}: {p.id} needs the natural-carbon primary row")
            members[p.id] = recovered_q(natc.tau, natc.rate, float(p.value))
    spread = max(abs(assembly - v) for v in members.values())
    half_width = spread + ledger_sum
    return QRecord(
        name,
        n.z,
        n.a,
        n.roster,
        q0,
        eta,
        recoil,
        uehling,
        assembly,
        leading,
        ledger,
        ledger_sum,
        members,
        spread,
        half_width,
        (assembly - half_width, assembly + half_width),
    )


# --------------------------------------------------------------------------------------------
# the source census (descriptive) and the historical-convention arithmetic
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CensusRow:
    """One printed bare-element row of the Suzuki, Measday and Roalsvig compendium."""

    locator: str
    label: str
    z: int
    ref: str
    tau: float  # ns, nan when the row prints no mean life
    tau_u: float
    rate: float  # ns^-1, nan when the row prints no capture rate
    rate_u: float
    own: bool


def census_rows(rows: Sequence[Mapping[str, str]], z: int, label: str) -> tuple[CensusRow, ...]:
    """The rows of the bare-element block ``label`` at ``z``, in printed order."""
    out = []
    for row in rows:
        if row["Z"] != str(z) or row["target_label"] != label:
            continue
        scale = 10.0 ** int(row["rate_exp10"]) if row["rate_exp10"] else math.nan
        unit = row["rate_unit"]
        if unit == "s^-1":
            to_ns = 1e-9
        elif unit == "10^6/s":
            to_ns = 1e-3
        else:
            raise EvaluationError(f"census: rate unit {unit!r}")
        tau = float(row["mean_life_ns"]) if row["mean_life_ns"] else math.nan
        tau_u = float(row["mean_life_unc_ns"]) if row["mean_life_unc_ns"] else math.nan
        rate = float(row["rate"]) * scale * to_ns if row["rate"] else math.nan
        rate_u = float(row["rate_unc"]) * scale * to_ns if row["rate_unc"] else math.nan
        out.append(
            CensusRow(
                f"Table {row['table']} p.{row['printed_page']} row {row['page_row_ordinal']}",
                row["target_label"],
                z,
                row["refs"],
                tau,
                tau_u,
                rate,
                rate_u,
                row["own_measurement"] == "true",
            )
        )
    if not out:
        raise EvaluationError(f"census: no row for {label} at Z={z}")
    return tuple(out)


def census_summary(rows: Sequence[CensusRow], primary_ref: str) -> dict[str, Any]:
    """A descriptive census: lifetime rows, observable-type exclusions, and the coordinate hull of
    ``tau_j +/- 2 u_j`` about the primary. The hull is a descriptive statistic only; no successor's
    between-experiment term is frozen here."""
    lifetimes = [r for r in rows if math.isfinite(r.tau)]
    rate_only = [r for r in rows if not math.isfinite(r.tau)]
    primaries = [r for r in lifetimes if r.ref == primary_ref]
    if len(primaries) != 1:
        raise EvaluationError(f"census: {len(primaries)} primary rows with ref {primary_ref!r}")
    p = primaries[0]
    low = min(r.tau - 2 * r.tau_u for r in lifetimes)
    high = max(r.tau + 2 * r.tau_u for r in lifetimes)
    radius = max(p.tau - low, high - p.tau)
    return {
        "lifetime_records": len(lifetimes),
        "rate_only_records": len(rate_only),
        "refs": [r.ref for r in lifetimes],
        "primary": {"locator": p.locator, "tau_ns": p.tau, "u_ns": p.tau_u},
        "central_spread_ns": max(abs(r.tau - p.tau) for r in lifetimes),
        "descriptive_hull_ns": [low, high],
        "descriptive_radius_ns": radius,
        "descriptive_extra_radius_ns": max(0.0, radius - 2 * p.tau_u),
        "status": "DESCRIPTIVE_ONLY: the successor freezes its own between-experiment term before it scores",
    }


def recovered_q(tau: float, capture: float, free_tau: float) -> float:
    """The decay factor a source's subtraction implies: ``(1/tau - capture) * free_tau``."""
    return (1.0 / tau - capture) * free_tau


def natc_recovery(row: CensusRow, alternatives: Sequence[float]) -> dict[str, Any]:
    """The natural-carbon source decay factor for each free-lifetime alternative, and its range when the
    printed mean life (last digit 0.1 ns) and capture rate (last digit 0.1e3 /s) move by half a unit."""
    if not (math.isfinite(row.tau) and math.isfinite(row.rate)):
        raise EvaluationError("natC recovery needs a mean life and a capture rate")
    out = {}
    for alt in alternatives:
        corners = [
            recovered_q(row.tau + s1 * 0.05, row.rate + s2 * 0.05e-6, alt) for s1 in (-1, 1) for s2 in (-1, 1)
        ]
        out[repr(alt)] = {
            "central": recovered_q(row.tau, row.rate, alt),
            "printed_rounding": [min(corners), max(corners)],
        }
    return out


def stock_table_finding(
    rows: Sequence[Mapping[str, str]], comparison_ref: str, stock: Mapping[int, float]
) -> dict[str, Any]:
    """The compiled capture entries for Mg, Al, Si and P against the unscaled inverse-variance average of the
    compendium's own measurement and the row of reference ``comparison_ref`` (10^6/s)."""
    out: dict[str, Any] = {}
    for z, label in ((12, "Mg"), (13, "Al"), (14, "Si"), (15, "P")):
        block = census_rows(rows, z, label)
        own = [r for r in block if r.own]
        other = [r for r in block if r.ref == comparison_ref]
        if len(own) != 1 or len(other) != 1:
            raise EvaluationError(f"stock finding: {label} rows")
        a, sa = own[0].rate * 1e3, own[0].rate_u * 1e3
        b, sb = other[0].rate * 1e3, other[0].rate_u * 1e3
        wa, wb = 1 / sa**2, 1 / sb**2
        mean = (a * wa + b * wb) / (wa + wb)
        units = abs(a - b) / math.hypot(sa, sb)
        out[label] = {
            "own": [a, sa],
            "comparison": [b, sb],
            "unscaled_average": mean,
            "average_u": (wa + wb) ** -0.5,
            "stock": stock[z],
            "matches_stock_at_4_decimals": round(mean, 4) == stock[z],
            "difference_in_combined_printed_units": units,
            "chi2_one_dof": units**2,
        }
    return out


# --------------------------------------------------------------------------------------------
# the pair
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """The rates of one primitive realization, ns^-1."""

    tau: float
    q: float
    free_tau: float
    capture: float
    decay: float

    @property
    def lifetime(self) -> float:
        return 1.0 / (self.capture + self.decay)

    @property
    def capture_probability(self) -> float:
        return self.capture / (self.capture + self.decay)


def _finite_real(x: Any, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise EvaluationError(f"{where}: {x!r} is not a real number")
    value = float(x)
    if not math.isfinite(value):
        raise EvaluationError(f"{where}: {x!r} is not finite")
    return value


#: An upper bound on the exponential variate ``-ln(u)`` a consumer scales a lifetime by when it samples a
#: time: no binary64 ``u`` in (0, 1) gives more than ``-ln(2**-1074) = 744.44...``.
EXPONENTIAL_VARIATE_MAX = 745.0


def pair(tau: Any, q: Any, free_tau: Any) -> Pair:
    """``d = q / free_tau``, ``c = 1/tau - d`` from one primitive tuple (ns, 1, ns), admitted only if every
    quantity a consumer forms from it is finite and in its domain; nothing is clipped. The consumer transform,
    in the order the consumers evaluate it:

    * the primitives: ``tau`` and ``free_tau`` finite and positive, ``q`` finite in (0, 1];
    * the decay rate ``d = q/free_tau``: finite and in ``(0, 1/free_tau]`` (a quotient that underflows to
      zero is refused);
    * ``1/tau``: finite; the capture rate ``c = 1/tau - d``: finite and not negative (``c = 0`` is admitted);
    * the total rate ``c + d``: finite and positive. The consumer lifetime `1/(capture + decay)` must be
      finite and positive;
    * the capture probability ``c/(c + d)``: in [0, 1];
    * the partial lifetimes the atomic route stores: ``1/d`` finite; ``1/c`` finite when ``c > 0`` (``c = 0``
      stores an infinite capture lifetime, whose reciprocal is a zero capture rate, so capture is never
      chosen); the rates it reads back from them, ``1/(1/c) + 1/(1/d)``, finite and positive;
    * a sampled waiting time: ``EXPONENTIAL_VARIATE_MAX`` times the lifetime, and divided by the total rate,
      finite.

    A non-real primitive is refused too. The same transform is checked by the semantic layer and the provider
    of the Geant4 glue."""
    t, x, f = _finite_real(tau, "tau"), _finite_real(q, "q"), _finite_real(free_tau, "free_tau")
    if not (t > 0 and f > 0):
        raise EvaluationError("tau and free_tau must be positive")
    if not 0 < x <= 1:
        raise EvaluationError(f"q={x!r} is outside (0, 1]")
    decay = x / f
    if not (math.isfinite(decay) and decay > 0 and decay <= 1.0 / f):
        raise EvaluationError("the decay rate is outside 0 < d <= 1/free_tau")
    total = 1.0 / t
    if not math.isfinite(total):
        raise EvaluationError("1/tau is not finite")
    capture = total - decay
    if not (math.isfinite(capture) and capture >= 0):
        raise EvaluationError(f"the capture rate {capture!r} is negative or not finite")
    rate = capture + decay
    if not (math.isfinite(rate) and rate > 0):
        raise EvaluationError("the total rate c + d is not finite and positive")
    lifetime = 1.0 / rate
    if not (math.isfinite(lifetime) and lifetime > 0):
        raise EvaluationError("the consumer lifetime 1/(capture + decay) is not finite and positive")
    probability = capture / rate
    if not 0 <= probability <= 1:
        raise EvaluationError("the capture probability c/(c + d) is outside [0, 1]")
    decay_lifetime = 1.0 / decay
    if not math.isfinite(decay_lifetime):
        raise EvaluationError("the decay partial lifetime 1/d is not finite")
    capture_lifetime = 1.0 / capture if capture > 0 else math.inf
    if capture > 0 and not math.isfinite(capture_lifetime):
        raise EvaluationError("the capture partial lifetime 1/c of a positive capture rate is not finite")
    read_back = 1.0 / capture_lifetime + 1.0 / decay_lifetime
    if not (math.isfinite(read_back) and read_back > 0):
        raise EvaluationError("the rates read back from the partial lifetimes are not finite and positive")
    longest = EXPONENTIAL_VARIATE_MAX
    if not (math.isfinite(longest * lifetime) and math.isfinite(longest / rate)):
        raise EvaluationError("a sampled waiting time could overflow")
    return Pair(t, x, f, capture, decay)


def serialize_q(assembly: float, decimals: int) -> Decimal:
    """The production decay factor: the assembly rounded half-even to ``decimals`` places."""
    return Decimal(repr(assembly)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)


# --------------------------------------------------------------------------------------------
# the two-state hyperfine model and the late-slope mapping
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """One point of the frozen reference product and what the exact two-state model gives there."""

    tau_late: Any
    q: Any
    free_tau: Any
    contrast: Any  # Delta / c_minus
    transition: Any  # h, ns^-1
    decay: Any
    c_minus: Any
    c_plus: Any
    delta: Any
    mean: Any
    capture_probability: Any


def reference(point: Sequence[Any], weight: float) -> Reference:
    """``(tau_late, Q, free_tau, Delta/c_minus, h)`` -> ``d = Q/free_tau``, ``c_minus = 1/tau_late - d``,
    ``Delta = x c_minus``, ``c_plus = c_minus - Delta``, ``A = c_plus + d + h``,
    ``M = tau_late (1 + f Delta / A)``, ``Pcap = 1 - d M``. Derived rates never vary independently.
    Complex arguments are accepted for complex-step derivatives."""
    tl, q, ft, x, h = point
    d = q / ft
    cm = 1.0 / tl - d
    delta = x * cm
    cp = cm - delta
    big_a = cp + d + h
    mean = tl * (1.0 + weight * delta / big_a)
    return Reference(tl, q, ft, x, h, d, cm, cp, delta, mean, 1.0 - d * mean)


def check_reference(r: Reference) -> None:
    """A reference point outside the positive two-state model is refused, never clipped."""
    for name in ("tau_late", "q", "free_tau", "transition", "decay", "mean"):
        value = getattr(r, name)
        if not (math.isfinite(value) and value > 0):
            raise EvaluationError(f"reference {name}={value!r}")
    if not (r.c_minus >= 0 and r.c_plus >= 0):
        raise EvaluationError("reference capture rate is negative")


def dio_conditional_mean(r: Reference, weight: float) -> float:
    """``E[T | DIO] = [(1-f)/B^2 + f/A^2 + h f (A+B)/(A^2 B^2)] / M``."""
    b = r.c_minus + r.decay
    a = r.c_plus + r.decay + r.transition
    f = weight
    return float(((1 - f) / b**2 + f / a**2 + r.transition * f * (a + b) / (a**2 * b**2)) / r.mean)


def mapping_bias(b: float, a: float, delta: float, transition: float, weight: float, t0: float) -> float:
    """Relative bias ``tau_fit/tau_late - 1`` of a free-amplitude single exponential fitted to the
    decay-electron
    time spectrum from ``t0`` to infinity: ``beta = -k u B / [A (A + k B)]``, ``u = A - B``,
    ``k = -f Delta exp(-u t0) / ((1-f) u + f h)``."""
    u = a - b
    k = -weight * delta * math.exp(-u * t0) / ((1 - weight) * u + weight * transition)
    return -k * u * b / (a * (a + k * b))


def window_bias(
    b: float, a: float, delta: float, transition: float, weight: float, t0: float, upper: float
) -> float:
    """The same projection on the finite window ``[t0, upper]``: the truncated-exponential maximum-likelihood
    rate whose window mean equals the spectrum's window mean, solved by bisection."""
    u = a - b
    kk = -weight * delta / ((1 - weight) * u + weight * transition)  # fast/slow amplitude ratio at t = 0

    def moments(rate: float) -> tuple[float, float]:
        e0, e1 = math.exp(-rate * t0), math.exp(-rate * upper)
        n0 = (e0 - e1) / rate
        n1 = (t0 * e0 - upper * e1) / rate + (e0 - e1) / rate**2
        return n0, n1

    s0, s1 = moments(b)
    f0, f1 = moments(a)
    target = (s1 + kk * f1) / (s0 + kk * f0)
    lo, hi = 0.25 * b, 4.0 * b
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        n0, n1 = moments(mid)
        if n1 / n0 > target:
            lo = mid
        else:
            hi = mid
    return b / (0.5 * (lo + hi)) - 1.0


def upper_function(u: float, b: float, h: float, weight: float, t0: float) -> float:
    """``F(u; B, h) = f (u - h) u B exp(-t0 u) / {[(1-f) u + f h] (B + u)^2}``: for ``u >= h`` the
    magnitude of
    the negative mapping bias is at most F (the positive ``k B`` term of the denominator is dropped)."""
    return weight * (u - h) * u * b * math.exp(-t0 * u) / (((1 - weight) * u + weight * h) * (b + u) ** 2)


def stationary_roots(b: float, h: float, weight: float, t0: float) -> list[float]:
    """Every real root of ``d log F / du = 0`` with the denominators cleared, a quartic in ``u``; the roots
    with
    ``u > h`` are returned, polished by Newton's method on the quartic."""
    P = np.polynomial.polynomial
    lin_a = [weight * h, 1 - weight]  # (1-f) u + f h
    u_minus_h = [-h, 1.0]
    u_ = [0.0, 1.0]
    b_plus_u = [b, 1.0]
    terms = [
        P.polymul(P.polymul(u_, lin_a), b_plus_u),
        P.polymul(P.polymul(u_minus_h, lin_a), b_plus_u),
        -t0 * P.polymul(P.polymul(P.polymul(u_minus_h, u_), lin_a), b_plus_u),
        -(1 - weight) * P.polymul(P.polymul(u_minus_h, u_), b_plus_u),
        -2.0 * P.polymul(P.polymul(u_minus_h, u_), lin_a),
    ]
    total = np.zeros(5)
    for t in terms:
        total[: len(t)] += t
    deriv = P.polyder(total)
    out = []
    for r in P.polyroots(total):
        if abs(r.imag) <= 1e-9 * max(1e-3, abs(r.real)) and r.real > h:
            x = float(r.real)
            for _ in range(60):
                step = float(P.polyval(x, total) / P.polyval(x, deriv))
                x -= step
                if abs(step) <= 1e-18:
                    break
            out.append(x)
    return sorted(out)


def mapping_enclosure(inputs: Inputs, h_worst: float) -> dict[str, Any]:
    """The self-consistent conservative negative-bias bound ``eps = max_u F(u; (1+eps)/tau_low, h)`` with
    ``tau_low = tau - k u_tau``, its outward rounding, and the late-lifetime reference interval
    ``[(tau - k u)/(1 + eps_map), (tau + k u)/(1 - eps_map)]``. The boundary ``u = h`` and the limit
    ``u -> infinity`` both give F = 0."""
    weight = float(inputs.upper_weight)
    t0 = float(inputs.t0)
    tau, half = float(inputs.tau), float(inputs.tau_k * inputs.tau_u)
    tau_low = tau - half
    eps = 0.0
    roots: list[float] = []
    iterations = 0
    for iterations in range(1, 201):  # noqa: B007 -- the count is reported
        b = (1.0 + eps) / tau_low
        roots = stationary_roots(b, h_worst, weight, t0)
        new = max([0.0] + [upper_function(u, b, h_worst, weight, t0) for u in roots])
        if abs(new - eps) <= 1e-18:
            eps = new
            break
        eps = new
    step = inputs.mapping_step
    eps_map = (Decimal(repr(eps)) / step).to_integral_value(rounding=ROUND_CEILING) * step
    em = float(eps_map)
    return {
        "worst_transition_per_ns": h_worst,
        "fit_low_ns": tau_low,
        "admissible_stationary_roots_per_ns": roots,
        "iterations": iterations,
        "self_consistent_eps": eps,
        "eps_map": str(eps_map),
        "late_interval_ns": [(tau - half) / (1 + em), (tau + half) / (1 - em)],
    }


def positive_branch_sup(
    weight: float,
    t0: float,
    late: Sequence[float],
    decay: Sequence[float],
    transition: Sequence[float],
    n: int = 400,
) -> float:
    """The largest positive mapping bias over the closure ``0 <= c_plus <= c_minus`` (``Delta > 0``), on a
    dense deterministic grid. A diagnostic: the frozen interval applies ``eps_map`` on both sides."""
    best = 0.0
    for tl in late:
        for d in decay:
            b = 1.0 / tl
            cm = b - d
            for h in np.linspace(transition[0], transition[1], 7):
                for frac in np.linspace(0.0, 1.0, n + 1):
                    delta = float(frac) * cm
                    a = (cm - delta) + d + float(h)
                    best = max(best, mapping_bias(b, a, delta, float(h), weight, t0))
    return best


# --------------------------------------------------------------------------------------------
# reference ranges and decisions
# --------------------------------------------------------------------------------------------


Box = tuple[tuple[float, float], ...]


def reference_quantity(name: str, r: Reference) -> Any:
    return {
        "late_lifetime": r.tau_late,
        "stopped_mean": r.mean,
        "capture_probability": r.capture_probability,
        "decay_rate": r.decay,
    }[name]


def gradient(fn: Callable[[Sequence[Any]], Any], x: Sequence[float]) -> list[float]:
    """Complex-step derivative of a real-analytic function: exact to rounding, no subtraction."""
    out = []
    for j in range(len(x)):
        step = 1e-30 * max(1.0, abs(x[j]))
        z = [complex(v) for v in x]
        z[j] += 1j * step
        out.append(fn(z).imag / step)
    return out


def quantity_range(
    name: str, box: Box, weight: float, starts: int = 48, seed: int = 20261027
) -> tuple[float, float, dict[str, Any]]:
    """The minimum and maximum of a reference quantity over the box: every corner, then projected
    sign-gradient
    ascent and descent from every corner and from deterministic interior starts. The extremes are widened
    outward by the numerical allowance ``1e-12`` relative."""

    def fn(x: Sequence[Any]) -> Any:
        return reference_quantity(name, reference(x, weight))

    lo_b = np.array([b[0] for b in box])
    hi_b = np.array([b[1] for b in box])
    scale = hi_b - lo_b
    corners = [list(c) for c in itertools.product(*box)]
    values = [float(fn(c)) for c in corners]
    vmin, vmax = min(values), max(values)
    rng = np.random.default_rng(seed)
    interior = [list(lo_b + scale * rng.random(len(box))) for _ in range(starts)]
    beyond = 0
    for sign in (1.0, -1.0):
        for start in corners + interior:
            x = np.array(start, dtype=float)
            step = 0.25
            for _ in range(80):
                g = np.array(gradient(fn, list(x))) * sign
                if not np.any(g):
                    break
                trial = np.clip(x + step * scale * np.sign(g), lo_b, hi_b)
                if sign * float(fn(list(trial))) > sign * float(fn(list(x))):
                    x = trial
                else:
                    step *= 0.5
                    if step < 1e-12:
                        break
            v = float(fn(list(x)))
            if v < vmin or v > vmax:
                beyond += 1
            vmin, vmax = min(vmin, v), max(vmax, v)
    allowance = 1e-12 * max(abs(vmin), abs(vmax))
    record = {
        "corners": len(corners),
        "interior_starts": starts,
        "searches_beyond_corners": beyond,
        "outward_allowance": allowance,
    }
    return vmin - allowance, vmax + allowance, record


def decision(predicted: Any, low: Any, high: Any, budget: Any, norm: str) -> dict[str, Any]:
    """MEETS iff the largest error over the reference interval is within the budget, DOES_NOT_MEET iff the
    smallest exceeds it, NOT_ESTABLISHED otherwise. Only the central prediction enters: a prediction
    uncertainty cannot move a decision. An empty interval is VACUOUS; a relative reference that reaches zero
    is non-informative."""
    y = _finite_real(predicted, "prediction")
    lo, hi, bud = (_finite_real(v, "reference") for v in (low, high, budget))
    if not bud > 0:
        raise EvaluationError("the budget must be positive")
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
        raise EvaluationError(f"norm {norm!r}")
    upper = max(err(lo), err(hi))
    lower = 0.0 if lo <= y <= hi else min(err(lo), err(hi))
    if upper <= bud:
        verdict = MEETS
    elif lower > bud:
        verdict = DOES_NOT_MEET
    else:
        verdict = NOT_ESTABLISHED
    return {"verdict": verdict, "L": lower, "U": upper}


def improvement(entry: Any, stock: Any, low: Any, high: Any) -> dict[str, Any]:
    """Robust improvement: ``|entry - q| < |stock - q|`` for every q in the interval. The advantage
    ``|stock - q| - |entry - q|`` is monotone between the two predictions and constant outside them, so its
    minimum over an interval is at an endpoint."""
    e, s, lo, hi = (_finite_real(v, "improvement") for v in (entry, stock, low, high))
    if not lo <= hi:
        return {"verdict": VACUOUS, "min_advantage": None, "midpoint": None}
    adv = min(abs(s - q) - abs(e - q) for q in (lo, hi))
    return {
        "verdict": ESTABLISHED if adv > 0 else NOT_ESTABLISHED,
        "min_advantage": adv,
        "midpoint": 0.5 * (e + s),
    }


# --------------------------------------------------------------------------------------------
# covariance views and the structural invariant
# --------------------------------------------------------------------------------------------


def first_order_covariance(
    tau: float, q: float, free_tau: float, u_tau: float | None, u_q: float | None, u_free: float | None
) -> dict[str, Any]:
    """``J V J^T`` for ``(c, d, Pcap)`` at ``x = (tau, Q, free_tau)``, FIRST_ORDER. A null standard
    uncertainty stays null: its column is excluded, never set to zero, and the excluded components are
    named."""
    jac = {
        "capture": (-1 / tau**2, -1 / free_tau, q / free_tau**2),
        "decay": (0.0, 1 / free_tau, -q / free_tau**2),
        "capture_probability": (-q / free_tau, -tau / free_tau, tau * q / free_tau**2),
    }
    us = (u_tau, u_q, u_free)
    names = ("tau", "q", "free_tau")
    assigned = [i for i, u in enumerate(us) if u is not None]
    keys = list(jac)
    variances = {i: float(u) ** 2 for i, u in enumerate(us) if u is not None}
    cov = {a: {b: sum(jac[a][i] * jac[b][i] * variances[i] for i in assigned) for b in keys} for a in keys}
    return {
        "label": "FIRST_ORDER",
        "jacobian": {k: list(v) for k, v in jac.items()},
        "assigned_components": [names[i] for i in assigned],
        "null_components": [names[i] for i in range(3) if us[i] is None],
        "covariance": cov,
    }


@dataclass(frozen=True)
class Draw:
    """One shared primitive realization and the pair it generates."""

    id: str
    tau: float
    q: float
    free_tau: float
    capture: float
    decay: float


def draws_from_primitives(primitives: Sequence[tuple[str, float, float, float]]) -> list[Draw]:
    """One pair per primitive realization, keyed by the realization's id."""
    out = []
    seen = set()
    for ident, tau, q, free_tau in primitives:
        if ident in seen:
            raise EvaluationError(f"duplicate primitive id {ident!r}")
        seen.add(ident)
        p = pair(tau, q, free_tau)
        out.append(Draw(ident, tau, q, free_tau, p.capture, p.decay))
    return out


def check_draws(draws: Sequence[Draw]) -> None:
    """The structural gate: every draw carries a primitive id and satisfies ``c + d = 1/tau`` and
    ``d = q / free_tau`` to 1e-12 relative. Independently sampled marginals fail it."""
    if not draws:
        raise EvaluationError("no draw")
    ids = [d.id for d in draws]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise EvaluationError("draws without unique primitive ids")
    for d in draws:
        values = (d.tau, d.q, d.free_tau, d.capture, d.decay)
        if not all(isinstance(v, float) and math.isfinite(v) for v in values):
            raise EvaluationError(f"draw {d.id}: non-finite or non-float value")
        if d.capture < 0 or d.decay <= 0:
            raise EvaluationError(f"draw {d.id}: negative capture or non-positive decay")
        if abs((d.capture + d.decay) * d.tau - 1.0) > REL_TOL:
            raise EvaluationError(f"draw {d.id}: c + d != 1/tau")
        if abs(d.decay * d.free_tau / d.q - 1.0) > REL_TOL:
            raise EvaluationError(f"draw {d.id}: d != q/free_tau")


def exact_identities(draws: Sequence[Draw]) -> dict[str, float]:
    """Exact (population) identities over equal-weight draws of the actual transformed variables:
    ``Var(c + d) = Var(lambda)`` and ``Cov(c, d) = Cov(lambda, d) - Var(d)``."""
    n = len(draws)
    lam = [1 / d.tau for d in draws]
    cs = [d.capture for d in draws]
    ds = [d.decay for d in draws]

    def mean(v: Sequence[float]) -> float:
        return sum(v) / n

    def cov(a: Sequence[float], b: Sequence[float]) -> float:
        ma, mb = mean(a), mean(b)
        return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / n

    total = [c + d for c, d in zip(cs, ds, strict=True)]
    return {
        "var_lambda": cov(lam, lam),
        "var_total": cov(total, total),
        "cov_cd": cov(cs, ds),
        "cov_lambda_d_minus_var_d": cov(lam, ds) - cov(ds, ds),
    }


# --------------------------------------------------------------------------------------------
# canonical numerical identity
# --------------------------------------------------------------------------------------------


def canonical_bytes(payload: Any) -> bytes:
    """Sorted-key, compact, ASCII JSON of a numerical payload; floats by their shortest round-trip repr."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def payload_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


# --------------------------------------------------------------------------------------------
# the evaluation
# --------------------------------------------------------------------------------------------


def frozen_box(inputs: Inputs, al: QRecord, late: Sequence[float]) -> Box:
    """The frozen reference product: late lifetime, Q region, free lifetime +/- k u, the contrast
    ``Delta/c_minus = a/f`` over ``a = a0 +/- k u_a`` and the transition rate ``h0 +/- k u_h`` (ns^-1)."""
    f = float(inputs.upper_weight)
    a_lo = float(inputs.amplitude_center - inputs.amplitude_k * inputs.amplitude_u)
    a_hi = float(inputs.amplitude_center + inputs.amplitude_k * inputs.amplitude_u)
    h_lo = float((inputs.transition_center - inputs.transition_k * inputs.transition_u) / 1000)
    h_hi = float((inputs.transition_center + inputs.transition_k * inputs.transition_u) / 1000)
    ft, fk = float(inputs.free_tau), float(inputs.free_tau_k * inputs.free_tau_u)
    if not h_lo > 0:
        raise EvaluationError("the transition-rate region reaches zero")
    return ((late[0], late[1]), al.region, (ft - fk, ft + fk), (a_lo / f, a_hi / f), (h_lo, h_hi))


def delivered(inputs: Inputs, al: QRecord) -> tuple[Decimal, Pair]:
    """The production tuple and the pair a consumer forms from its binary64 values."""
    q = serialize_q(al.assembly, inputs.q_decimals)
    return q, pair(float(inputs.tau), float(q), float(inputs.free_tau))


def check_estimand(tau: str, inputs: Inputs) -> None:
    """The production lifetime is the source's late-slope fit, exactly as printed: any other lifetime (a
    stopped
    mean, a mean-matched alternative) is a different estimand and is refused."""
    if tau != str(inputs.tau):
        raise EvaluationError(
            f"estimand: the tuple's lifetime {tau!r} is not the source's late-slope {inputs.tau}"
        )


def check_delivered(capture: Any, decay: Any, row: Mapping[str, Any]) -> None:
    """The convention gate: a delivered pair must be the one its tuple gives, ``c + d = 1/tau`` within the
    lifetime tolerance and ``d = q/free_tau`` to 1e-12 relative. A capture rate formed with another decay
    convention is refused, however small the lifetime shift it makes."""
    c, d = _finite_real(capture, "capture"), _finite_real(decay, "decay")
    tau, q, ft = float(row["tau_ns"]), float(row["q"]), float(row["free_tau_ns"])
    if not (c >= 0 and d > 0):
        raise EvaluationError("convention: negative capture or non-positive decay")
    if abs(d - q / ft) > REL_TOL * (q / ft):
        raise EvaluationError("convention: the decay rate is not q/free_tau")
    if abs(1.0 / (c + d) - tau) > NS_TOL:
        raise EvaluationError("convention: the pair's lifetime is not the tuple's tau")


def predictions(p: Pair) -> dict[str, float]:
    """What a scalar consumer receives: one lifetime that is both its late slope and its stopped mean."""
    return {
        "late_lifetime": p.lifetime,
        "stopped_mean": p.lifetime,
        "capture_probability": p.capture_probability,
        "decay_rate": p.decay,
    }


def sensitivities(box: Box, weight: float, budgets: Mapping[str, float], fraction: float) -> dict[str, Any]:
    """For each decision and component: the largest |d q_ref / d x_j| over corners and an interior grid (in
    the decision's norm), times the component's half-width; consequential iff >= fraction * budget. The
    complex-step derivative is compared with central finite differences, and the second difference is
    reported as curvature."""
    out: dict[str, Any] = {}
    grid = [list(c) for c in itertools.product(*[(b[0], 0.5 * (b[0] + b[1]), b[1]) for b in box])]
    for name, norm in DECISIONS:

        def fn(x: Sequence[Any], name: str = name) -> Any:
            return reference_quantity(name, reference(x, weight))

        rows = {}
        for j, comp in enumerate(COMPONENTS):
            half = 0.5 * (box[j][1] - box[j][0])
            best, worst_fd, worst_curv = 0.0, 0.0, 0.0
            for x in grid:
                g = gradient(fn, x)[j]
                value = float(fn(x))
                s = abs(g) / (abs(value) if norm == "relative" else 1.0)
                best = max(best, s)
                hstep = 1e-4 * (box[j][1] - box[j][0])
                xp, xm = list(x), list(x)
                xp[j] += hstep
                xm[j] -= hstep
                fp, fm = float(fn(xp)), float(fn(xm))
                fd = (fp - fm) / (2 * hstep)
                if g != 0:
                    worst_fd = max(worst_fd, abs(fd - g) / abs(g))
                worst_curv = max(
                    worst_curv,
                    abs(fp - 2 * value + fm)
                    * (half / hstep) ** 2
                    / (abs(value) if norm == "relative" else 1.0),
                )
            impact = best * half
            rows[comp] = {
                "max_sensitivity": best,
                "half_width": half,
                "impact": impact,
                "threshold": fraction * budgets[name],
                "consequential": impact >= fraction * budgets[name],
                "finite_difference_relative_disagreement": worst_fd,
                "second_order_over_half_width": worst_curv,
            }
        out[name] = rows
    return out


def evaluate(
    inputs: Inputs,
    oracle: OracleInputs,
    compendium: Sequence[Mapping[str, str]],
    stock_capture_table: Mapping[int, float],
) -> dict[str, Any]:
    """Every derived quantity and decision, from the typed inputs, the oracle and the compendium rows."""
    weight = float(inputs.upper_weight)
    census = inputs.census
    o_rows = census_rows(compendium, census["O16"]["Z"], census["O16"]["block_label"])
    c_rows = census_rows(compendium, census["C12"]["Z"], census["C12"]["block_label"])
    natc_primary = [r for r in c_rows if r.ref == census["C12"]["primary_ref"]]
    if len(natc_primary) != 1:
        raise EvaluationError("natC primary row")
    q_records = {
        "Al27": q_record(inputs, oracle, "Al27"),
        "O16": q_record(inputs, oracle, "O16"),
        "C12": q_record(inputs, oracle, "C12", natc_primary[0]),
    }
    al = q_records["Al27"]
    q_ser, delivered_pair = delivered(inputs, al)
    h_lo = float((inputs.transition_center - inputs.transition_k * inputs.transition_u) / 1000)
    mapping = mapping_enclosure(inputs, h_lo)
    late = (mapping["late_interval_ns"][0], mapping["late_interval_ns"][1])
    box = frozen_box(inputs, al, late)
    for corner in itertools.product(*box):
        check_reference(reference(corner, weight))
    # the mapping is rechecked on the finite window at the worst point, and against exact projections
    b_worst = (1 + mapping["self_consistent_eps"]) / mapping["fit_low_ns"]
    u_star = mapping["admissible_stationary_roots_per_ns"][0]
    delta_star = h_lo - u_star
    a_star = u_star + b_worst
    exact_star = mapping_bias(b_worst, a_star, delta_star, h_lo, weight, float(inputs.t0))
    window_star = window_bias(
        b_worst, a_star, delta_star, h_lo, weight, float(inputs.t0), float(inputs.window_upper)
    )
    d_range = (box[1][0] / box[2][1], box[1][1] / box[2][0])
    positive = positive_branch_sup(weight, float(inputs.t0), late, d_range, (box[4][0], box[4][1]))
    ranges = {}
    for name, _ in DECISIONS:
        lo, hi, record = quantity_range(name, box, weight)
        ranges[name] = {"low": lo, "high": hi, "search": record}
    preds = predictions(delivered_pair)
    budgets = {k: float(v) for k, v in inputs.budgets.items()}
    decisions = {}
    for name, norm in DECISIONS:
        decisions[name] = dict(
            decision(preds[name], ranges[name]["low"], ranges[name]["high"], budgets[name], norm),
            norm=norm,
            budget=budgets[name],
            prediction=preds[name],
        )
    decisions["late_lifetime"]["label"] = "source-precision conditional (no validation power)"
    for name in decisions:
        decisions[name]["conditional_on"] = REGION_LABEL + " regions and the two-state observation model"
    # stock comparators, both options off
    stock = {}
    for route, decay in (("atomic", oracle.stock_decay_atomic), ("stopping", oracle.stock_decay_stopping)):
        sp = Pair(math.nan, math.nan, math.nan, oracle.stock_capture, decay)
        sp_pred = predictions(sp)
        stock[route] = {
            "capture_per_ns": oracle.stock_capture,
            "decay_per_ns": decay,
            "lifetime_ns": sp.lifetime,
            "capture_probability": sp.capture_probability,
            "improvement": {
                name: improvement(preds[name], sp_pred[name], ranges[name]["low"], ranges[name]["high"])
                for name in ("late_lifetime", "stopped_mean", "capture_probability")
            },
        }
    # power on the actual region
    q_mut = pair(float(inputs.tau), float(inputs.q_mutant), float(inputs.free_tau))
    mut_preds = predictions(q_mut)
    power = {
        "q_mutant": {
            name: decision(mut_preds[name], ranges[name]["low"], ranges[name]["high"], budgets[name], norm)
            for name, norm in DECISIONS
            if name in ("decay_rate", "capture_probability")
        },
        "mean_mutant": decision(
            preds["stopped_mean"] * float(inputs.mean_mutant_factor),
            ranges["stopped_mean"]["low"],
            ranges["stopped_mean"]["high"],
            budgets["stopped_mean"],
            "relative",
        ),
    }
    # the mean-matched alternative, diagnostic only
    mm = reference(
        (
            float(inputs.tau),
            float(q_ser),
            float(inputs.free_tau),
            float(inputs.mean_matched_amplitude) / weight,
            float(inputs.mean_matched_transition) / 1000,
        ),
        weight,
    )
    mean_matched = {
        "contrast": mm.contrast,
        "transition_per_ns": mm.transition,
        "mean_ns": mm.mean,
        "capture_rate_per_ns": 1 / mm.mean - mm.decay,
        "capture_probability": mm.capture_probability,
        "late_lifetime_cost": mm.mean / float(inputs.tau) - 1,
        "selected_for_production": False,
        "dio_conditional_mean_ns": dio_conditional_mean(mm, weight),
    }
    # covariance views
    fo = first_order_covariance(
        delivered_pair.tau,
        delivered_pair.q,
        delivered_pair.free_tau,
        float(inputs.tau_u),
        None,
        float(inputs.free_tau_u),
    )
    fixture = draws_from_primitives(
        [
            ("fixture-low", float(inputs.tau) - 1.0, float(q_ser), float(inputs.free_tau)),
            ("fixture-high", float(inputs.tau) + 1.0, float(q_ser), float(inputs.free_tau)),
        ]
    )
    check_draws(fixture)
    fixture_ids = exact_identities(fixture)
    fixture_first = first_order_covariance(
        float(inputs.tau), float(q_ser), float(inputs.free_tau), 1.0, None, None
    )
    first_var_total = (1 / float(inputs.tau) ** 2) ** 2
    covariance = {
        "first_order": fo,
        "exact_identities_note": "Var(c+d) = Var(lambda) and Cov(c,d) = Cov(lambda,d) - Var(d) hold for "
        "actual "
        "shared draws; a nonlinear draw need not equal the linearized matrix",
        "fixture": {
            "tau_ns": [float(inputs.tau) - 1.0, float(inputs.tau) + 1.0],
            "exact": fixture_ids,
            "first_order_var_lambda": first_var_total,
            "relative_gap": fixture_ids["var_lambda"] / first_var_total - 1,
            "first_order_view": fixture_first["covariance"],
        },
    }
    sens = sensitivities(box, weight, budgets, float(inputs.consequential_fraction))
    # descriptive census and source findings
    o_census = census_summary(o_rows, census["O16"]["primary_ref"])
    c_census = census_summary(c_rows, census["C12"]["primary_ref"])
    alternatives = [
        float(p.value)
        for p in inputs.nuclei["C12"].prescriptions
        if p.kind == "recovered_natc" and p.value is not None
    ]
    natc = natc_recovery(natc_primary[0], alternatives)
    finding = stock_table_finding(compendium, "42", stock_capture_table)
    h_inv = [(float(c), float(u)) for c, u in inputs.transition_crosscheck]
    w = [1 / u**2 for _, u in h_inv]
    h_mean = sum(c * wi for (c, _), wi in zip(h_inv, w, strict=True)) / sum(w)
    h_u = sum(w) ** -0.5
    return {
        "schema": EVALUATION_SCHEMA,
        "evaluation_id": inputs.evaluation_id,
        "roster": {name: q_records[name].roster for name in _NUCLEUS_NAMES},
        "admitted_rows": [[al.z, al.a]],
        "q": {
            name: {
                "Z": r.z,
                "A": r.a,
                "static_q": r.static_q,
                "eta": r.eta,
                "recoil": r.recoil,
                "uehling_kinetic": r.uehling,
                "assembly": r.assembly,
                "leading_formula": r.leading_formula,
                "ledger": dict(r.ledger),
                "ledger_sum": r.ledger_sum,
                "members": dict(r.members),
                "spread": r.spread,
                "half_width": r.half_width,
                "region": list(r.region),
                "region_label": REGION_LABEL,
                "half_width_over_ledger": r.half_width / r.ledger_sum,
                "half_width_over_largest_coefficient_one_diagnostic": r.half_width
                / max(
                    r.ledger["binding_recoil_cubic"],
                    r.ledger["recoil_quadratic"],
                    r.ledger["relativistic_uehling"],
                    r.ledger["one_loop"],
                    r.ledger["two_loop"],
                ),
            }
            for name, r in q_records.items()
        },
        "f_factor": f_factor(float(inputs.mass_ratio)),
        "tuple": {
            "Z": al.z,
            "A": al.a,
            "tau_ns": str(inputs.tau),
            "q": str(q_ser),
            "free_tau_ns": str(inputs.free_tau),
            "rounding_displacement": al.assembly - float(q_ser),
        },
        "pair": {
            "capture_per_ns": delivered_pair.capture,
            "decay_per_ns": delivered_pair.decay,
            "lifetime_ns": delivered_pair.lifetime,
            "capture_probability": delivered_pair.capture_probability,
        },
        "mapping": dict(
            mapping,
            exact_bias_at_worst=exact_star,
            window_bias_at_worst=window_star,
            positive_branch_sup=positive,
        ),
        "box": {comp: list(b) for comp, b in zip(COMPONENTS, box, strict=True)},
        "ranges": ranges,
        "decisions": decisions,
        "stock": stock,
        "power": power,
        "mean_matched": mean_matched,
        "covariance": covariance,
        "sensitivities": sens,
        "transition_crosscheck": {
            "mean_per_us": h_mean,
            "u_per_us": h_u,
            "k2": [h_mean - 2 * h_u, h_mean + 2 * h_u],
        },
        "census": {
            "O16": o_census,
            "C12": c_census,
            "natc_source_q": natc,
            "minor_isotope": census["minor_isotope"],
        },
        "stock_table_finding": finding,
    }


# --------------------------------------------------------------------------------------------
# the shared D1/D2 convention record
# --------------------------------------------------------------------------------------------

UNRESOLVED = "UNRESOLVED"
#: Significant digits of a number in the shared record's payload: its digest must not depend on the last
#: bits a platform's libm gives a quadrature.
PAYLOAD_DIGITS = 12


def significant(x: float) -> str:
    """``x`` as a decimal string of :data:`PAYLOAD_DIGITS` significant digits."""
    return format(_finite_real(x, "payload"), f".{PAYLOAD_DIGITS}g")


def convention_record(inputs: Inputs, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """The shared ``bound-decay-response-v1`` record: historical conversion, evaluated physics, the two
    source-response identities, and D2's source-response role (convention-only: delivery mode, conversion
    refusal and width definitions; the spectrum identity stays UNRESOLVED). No D1 decision reads this record.
    Its numerical payload has one canonical digest."""
    q = evaluation["q"]
    natc = evaluation["census"]["natc_source_q"]
    payload = {
        "historical_conversion": {
            "identity": "c_cons = c_src + Q_src g_src - Q_cons g_cons, only when the source normalization is "
            "established; a tau primitive is preferred whenever available",
            "suzuki1987_al_q": "0.993",
            "mizuno2025_al_q": "0.993",
            "free_tau_alternatives_ns": ["2197.03", "2197.0", str(inputs.free_tau)],
            "natc_recovered_q": {k: significant(v["central"]) for k, v in natc.items()},
            "status": "source normalizations carried as alternatives; none chosen from rounded capture rates",
        },
        "source_response": {
            "stanislaus1983_thesis": {
                "q_formula": "1 - 0.5 (Z/137)^2",
                "locator": "MSc thesis, UBC (1983), Eq. (4) p.56",
                "status": "RESOLVED_FOR_THESIS",
            },
            "stanislaus1987_paper": {
                "q_formula": UNRESOLVED,
                "status": UNRESOLVED,
                "note": "the paper prints no Q formula and also cites another evaluation; "
                "equal published results do not establish equal conventions",
            },
            "d2_response": {
                "delivery_mode": "SOURCE_REPORTED_FROZEN",
                "conversion_refusal": "CONVERSION_NOT_ESTABLISHED",
                "q_definition": "Q_i = Gamma_dec,i/Gamma_free",
                "e2_definition": "E2_ij = Gamma_acc,ij/Gamma_dec,i",
                "k_definition": "K_ij = Q_i*E2_ij = Gamma_acc,ij/Gamma_free",
                "accepted_width_definition": (
                    "Gamma_acc,ij = sum over channels of the integral of R_ij dGamma_i; "
                    "invisible channels have zero response"
                ),
                "spectrum_id": UNRESOLVED,
            },
        },
        "evaluated_physics": {
            name: {
                "Z": q[name]["Z"],
                "A": q[name]["A"],
                "assembly": significant(q[name]["assembly"]),
                "region": [significant(x) for x in q[name]["region"]],
                "region_label": REGION_LABEL,
                "runtime_status": evaluation["roster"][name],
                "serialized_q": evaluation["tuple"]["q"] if name == "Al27" else None,
            }
            for name in _NUCLEUS_NAMES
        },
        "definitions": {
            "Q": "Gamma_decay_bound / Gamma_free; d = Q / free_tau with the empirical free lifetime",
            "free_tau_ns": str(inputs.free_tau),
            "units": "ns, ns^-1, dimensionless Q",
            "d2_ratio": "A = (n/m)(a_M/a_O)(Q_O/Q_M)(E2_O/E2_M); an evaluated update changes Q and the "
            "E2/amplitude "
            "response together; absolute free normalization cancels",
        },
    }
    return {
        "schema": CONVENTION_SCHEMA,
        "id": CONVENTION_ID,
        "status": "DRAFT_D1_ROLES",
        "payload": payload,
        "payload_sha256": payload_digest(payload),
    }


# --------------------------------------------------------------------------------------------
# reading a committed evaluation: derived fields are never trusted
# --------------------------------------------------------------------------------------------

_EVALUATION_KEYS = (
    "schema",
    "evaluation_id",
    "roster",
    "admitted_rows",
    "q",
    "f_factor",
    "tuple",
    "pair",
    "mapping",
    "box",
    "ranges",
    "decisions",
    "stock",
    "power",
    "mean_matched",
    "covariance",
    "sensitivities",
    "transition_crosscheck",
    "census",
    "stock_table_finding",
)


def _close(a: Any, b: Any, where: str) -> None:
    """Structural equality with floats compared to 1e-12 relative (numerical search is not byte-pinned)."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        if a != b:
            raise EvaluationError(f"{where}: committed {a!r} differs from recomputed {b!r}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (math.isfinite(a) and math.isfinite(b)):
            raise EvaluationError(f"{where}: non-finite")
        if abs(a - b) > 1e-12 * max(abs(a), abs(b), 1e-300) and a != b:
            raise EvaluationError(f"{where}: committed {a!r} differs from recomputed {b!r}")
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise EvaluationError(f"{where}: length differs")
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _close(x, y, f"{where}[{i}]")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise EvaluationError(f"{where}: keys differ: {sorted(set(a) ^ set(b))}")
        for k in a:
            _close(a[k], b[k], f"{where}.{k}")
        return
    raise EvaluationError(f"{where}: type {type(a).__name__} differs from {type(b).__name__}")


def check_evaluation(raw: bytes, recomputed: Mapping[str, Any]) -> dict[str, Any]:
    """A committed evaluation record is accepted only if every field equals its recomputation from the
    primitives. Unknown keys, missing keys and any contradiction are refused; the recomputed record is what
    a caller receives, never the committed derived fields."""
    obj = _keys(parse_json(raw), _EVALUATION_KEYS, "evaluation")
    normalized = json.loads(canonical_bytes(recomputed))
    _close(obj, normalized, "evaluation")
    return dict(recomputed)
