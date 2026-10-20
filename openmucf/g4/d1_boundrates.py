"""Verification of the common bound-rate pair of the two stopped-muon routes.

With ``G4HadronicParameters::SetEnableMuonicBoundRates(true)``, a build carrying the patches under
``cpp/patches/`` gives ``G4MuonMinusBoundDecay::ApplyYourself`` (the stopping route) and
``G4MuonicAtomHelper::ConstructMuonicAtom`` (the atomic route) one rate pair for a negative muon in the 1s
state of a nucleus with ``2 <= Z <= 13`` and ``A >= Z``: the capture rate of the helper's resolver and the
stopping route's decay rate below Z = 14 (``cpp/include/G4MuonicBoundRateProvider.hh``). This module holds
what the verification of that pair needs without Geant4:

* :func:`reference` -- the registered pair for one nuclide, evaluated here: the capture rate the helper's
  compiled resolver returns (:func:`openmucf.g4.sources.d1_nuclear_capture.capture_rate` with the helper's
  table and fallback) and the decay rate of the stopping expression
  (:func:`openmucf.g4.d1_lifetime.bound_decay_rate`), with the muon mass, fine-structure constant, microsecond
  and nuclear masses of the committed decay-rate oracle. Agreement with it is agreement with the prescription
  given those inputs; it says nothing about the physical accuracy of the prescription or of the inputs.
* :func:`groups` and :data:`POLICY_RELPATH` -- the verification policy: the 12 groups (isotope x revision x
  route), their events, seeds and start-time strata, the frozen reference pairs, the statistical limits and
  the tolerances. ``scripts/generate_g4boundrates.py`` generates it; it is committed before the production
  harvest runs.

Rates are in ns^-1, Geant4's unit; times in ns.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from openmucf.g4 import d1_lifetime as lifetime
from openmucf.g4 import d1_mixture as mixture
from openmucf.g4 import d1_natural as natural
from openmucf.g4.sources import d1_nuclear_capture as d1

#: The committed policy, relative to the repository root.
POLICY_RELPATH = "data/g4/d1/bound_rates_policy.json"
#: The status of the theory budget the physical-accuracy question needs.
THEORY_RELPATH = "data/g4/d1/bound_rates_theory.json"

STOPPING, ATOMIC = mixture.STOPPING, mixture.ATOMIC
ROUTES = (STOPPING, ATOMIC)
REVISIONS = mixture.REVISIONS

#: The production isotopes and the events of each group.
ISOTOPES: tuple[tuple[str, int, int, int], ...] = (
    ("C-12", 6, 12, 60_000_000),
    ("N-14", 7, 14, 24_000_000),
    ("O-16", 8, 16, 12_000_000),
)
#: The start times of the three strata of every group, in event order.
T0_NS = (0, 250, 1000)
#: Group g is seeded from SEED_FIRST + SEED_STRIDE * g; event i of it from that plus i.
SEED_FIRST, SEED_STRIDE = 100_000_000, 100_000_000
#: The events of one production chunk; every stratum is a whole number of chunks.
CHUNK_EVENTS = 2_000_000

#: The two statistical families: four tests per group (capture count; the scaled sums of all, capture and
#: decay delays), and three Pearson tests per group of the scaled delays in equal-probability bins of Exp(1).
FAMILY_ALPHA = Decimal("0.001")
PRIMARY_TESTS = ("capture_count", "sum_all", "sum_capture", "sum_decay")
STREAMS = ("all", "capture", "decay")
TIME_BINS = mixture.TIME_BINS

#: Numerical tolerances.
RATE_RTOL = 1e-9
CLOCK_ABS_NS = 1e-6
CLOCK_EPS_FACTOR = 64
EPS = 2.0**-52

#: The identity roster, its seeds and events.
IDENTITY_ROSTER: tuple[tuple[str, int, int], ...] = (
    ("H", 1, 1), ("He", 2, 4), ("C", 6, 12), ("C", 6, 13), ("N", 7, 14), ("O", 8, 16), ("O", 8, 18),
    ("Al", 13, 27), ("Si", 14, 28), ("Pb", 82, 208),
)
IDENTITY_SEED_FIRST, IDENTITY_SEED_STRIDE, IDENTITY_EVENTS = 1_700_000_000, 10_000, 2000
DRILL_SEED_FIRST, DRILL_SEED_STRIDE = 1_900_000_000, 10_000

#: The clock categories, and the physical-clock outcome predicted for each before the harvest, by stratum.
CLOCK_CATEGORIES = ("stop_dio", "stop_capture", "atom_creation", "atom_dio", "atom_capture")
PREDICTED_PHYSICAL = {
    "stop_dio": {0: "pass", 250: "pass", 1000: "pass"},
    "stop_capture": {0: "fail", 250: "fail", 1000: "fail"},
    "atom_creation": {0: "pass", 250: "fail", 1000: "fail"},
    "atom_dio": {0: "pass", 250: "fail", 1000: "fail"},
    "atom_capture": {0: "pass", 250: "fail", 1000: "fail"},
}


class BoundRateError(RuntimeError):
    """A record, a committed file or an input this verification cannot use."""


# --------------------------------------------------------------------------------------------
# the registered pair
# --------------------------------------------------------------------------------------------


def in_domain(z: int, a: int) -> bool:
    """The provider's implementation domain: ``2 <= Z <= 13`` and ``A >= Z``."""
    return 2 <= z <= 13 and a >= z


@dataclass(frozen=True)
class Pair:
    """The registered pair of one nuclide, binary64 in ns^-1, with the nuclear mass (MeV) it was evaluated
    with."""

    z: int
    a: int
    mass: float
    capture: float
    decay: float

    @property
    def total(self) -> float:
        """``capture + decay`` in binary64, the sum ``G4MuonMinusBoundDecay::ApplyYourself`` forms."""
        return self.capture + self.decay

    @property
    def lifetimes(self) -> tuple[float, float, float]:
        """``(1 / (c + d), 1 / d, 1 / c)`` in binary64: the three lifetimes the helper stores from the
        pair."""
        nc = 1.0 / self.capture if self.capture > 0.0 else math.inf
        return 1.0 / self.total, 1.0 / self.decay, nc

    @property
    def p_capture(self) -> Fraction:
        """``c / (c + d)``, exactly, from the two binary64 rates."""
        return Fraction(self.capture) / (Fraction(self.capture) + Fraction(self.decay))


class Reference:
    """The registered pair from the committed inputs: never from a harvest."""

    def __init__(self, root: Path) -> None:
        self.oracle = lifetime.load_oracle(root / lifetime.ORACLE_RELPATH)
        self.helper = d1.load_helper(root / d1.HELPER_RELPATH)
        self.model = natural.reference_model(self.helper)
        self.mu_mass = self.oracle.constants["mu_mass"]

    def pair(self, z: int, a: int) -> Pair:
        if not in_domain(z, a):
            raise BoundRateError(f"Z={z} A={a} lies outside the domain")
        mass = self.oracle.light[(z, a)][0]
        capture = d1.capture_rate(z, a, self.helper.capture_records, self.model)
        decay = lifetime.bound_decay_rate(z, a, self.mu_mass, mass)
        return Pair(z, a, mass, capture, decay)


def reference(root: Path, z: int, a: int) -> Pair:
    """The registered pair of ``(z, a)``."""
    return Reference(root).pair(z, a)


def isotope(label: str) -> tuple[int, int, int]:
    """``(Z, A, events per group)`` of a production isotope."""
    for name, z, a, events in ISOTOPES:
        if name == label:
            return z, a, events
    raise BoundRateError(f"no production isotope {label}")


# --------------------------------------------------------------------------------------------
# groups, strata, chunks
# --------------------------------------------------------------------------------------------


def groups() -> list[dict[str, Any]]:
    """The 12 primary groups in registration order (isotope, then revision, then route), each with its seed
    base and its three strata of contiguous events."""
    out: list[dict[str, Any]] = []
    g = 0
    for label, z, a, events in ISOTOPES:
        for revision in REVISIONS:
            for route in ROUTES:
                per = events // len(T0_NS)
                strata = [{"t0_ns": t0, "first": k * per, "count": per} for k, t0 in enumerate(T0_NS)]
                out.append({"group": g, "isotope": label, "z": z, "a": a, "revision": revision,
                            "route": route, "events": events, "seed_base": SEED_FIRST + SEED_STRIDE * g,
                            "strata": strata})
                g += 1
    return out


def group_name(group: Mapping[str, Any]) -> str:
    return f"{group['isotope']}:{group['revision']}:{group['route']}"


def chunks(group: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The production chunks of a group: ``CHUNK_EVENTS`` contiguous events each, inside one stratum."""
    out = []
    for stratum in group["strata"]:
        if stratum["count"] % CHUNK_EVENTS:
            raise BoundRateError(f"{group_name(group)}: a stratum is not a whole number of chunks")
        for k in range(stratum["count"] // CHUNK_EVENTS):
            out.append({"t0_ns": stratum["t0_ns"], "first": stratum["first"] + k * CHUNK_EVENTS,
                        "count": CHUNK_EVENTS})
    return out


def identity_seed(revision: str, route: str, label: tuple[str, int, int], t0: int) -> int:
    """The seed base of one identity configuration: its index in the order revision, route, roster, start
    time."""
    k = (((REVISIONS.index(revision) * len(ROUTES) + ROUTES.index(route)) * len(IDENTITY_ROSTER)
          + IDENTITY_ROSTER.index(label)) * len(T0_NS) + T0_NS.index(t0))
    return IDENTITY_SEED_FIRST + IDENTITY_SEED_STRIDE * k


def per_test_alpha(n_tests: int) -> Decimal:
    return FAMILY_ALPHA / n_tests


# --------------------------------------------------------------------------------------------
# the policy file
# --------------------------------------------------------------------------------------------


def policy_core(root: Path) -> dict[str, Any]:
    """Everything of the policy that needs no statistical function: the prescription, its inputs, the frozen
    reference pairs, the groups and the tolerances. The generator adds the limits."""
    ref = Reference(root)
    oracle_path = root / lifetime.ORACLE_RELPATH
    pairs = {}
    for label, z, a, _ in ISOTOPES:
        p = ref.pair(z, a)
        tau, tau_dio, tau_nc = p.lifetimes
        pairs[label] = {"z": z, "a": a, "mass_mev": p.mass.hex(), "capture": p.capture.hex(),
                        "decay": p.decay.hex(), "total": p.total.hex(), "lifetime": tau.hex(),
                        "dio_lifetime": tau_dio.hex(), "nc_lifetime": tau_nc.hex(),
                        "p_capture": repr(float(p.p_capture))}
    return {
        "prescription": {
            "capture": "G4MuonicAtomHelper::GetMuonCaptureRate(Z, A), the helper's resolver as it stands",
            "decay": "(0.45517005 / microsecond) * [1 - (Z alpha)^2 (0.5 + 0.06 m_mu / M)], grouped as "
                     "G4MuonMinusBoundDecay::GetMuonDecayRate groups it below Z = 14",
            "domain": "2 <= Z <= 13 and A >= Z",
            "physical_accuracy": "not established: the policy verifies the implementation of the "
                                 "prescription",
        },
        "inputs": {"oracle": lifetime.ORACLE_RELPATH,
                   "oracle_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
                   "mu_mass_mev": ref.mu_mass.hex(), "alpha": ref.oracle.constants["alpha"].hex(),
                   "microsecond_ns": ref.oracle.constants["microsecond"].hex()},
        "references": pairs,
        "groups": groups(),
        "chunk_events": CHUNK_EVENTS,
        "tolerances": {"rate_relative": repr(RATE_RTOL), "clock_absolute_ns": repr(CLOCK_ABS_NS),
                       "clock_eps_factor": CLOCK_EPS_FACTOR, "eps": EPS.hex()},
        "identity": {"seed_first": IDENTITY_SEED_FIRST, "seed_stride": IDENTITY_SEED_STRIDE,
                     "events": IDENTITY_EVENTS, "t0_ns": list(T0_NS),
                     "roster": [f"{s}-{a}" for s, _, a in IDENTITY_ROSTER]},
        "drills": {"seed_first": DRILL_SEED_FIRST, "seed_stride": DRILL_SEED_STRIDE},
        "clock_predictions": PREDICTED_PHYSICAL,
    }


def render_policy(policy: Mapping[str, Any]) -> bytes:
    return (json.dumps(policy, indent=1, sort_keys=True) + "\n").encode("ascii")


def load_policy(root: Path) -> dict[str, Any]:
    raw = (root / POLICY_RELPATH).read_bytes()
    if b"\r" in raw:
        raise BoundRateError(f"{POLICY_RELPATH} contains CR")
    policy: dict[str, Any] = json.loads(raw.decode("ascii"))
    return policy


def frozen_pair(policy: Mapping[str, Any], label: str) -> Pair:
    """The reference pair the policy froze for a production isotope."""
    r = policy["references"][label]
    return Pair(int(r["z"]), int(r["a"]), float.fromhex(r["mass_mev"]), float.fromhex(r["capture"]),
                float.fromhex(r["decay"]))


# --------------------------------------------------------------------------------------------
# the theory admission gate
# --------------------------------------------------------------------------------------------

#: The terms a physical uncertainty budget of the prescription must bound before any nuclide is admitted.
REQUIRED_TERMS = ("equation", "binding_expansion", "recoil", "nuclear_mass", "finite_size", "radiative",
                  "numerical")


def theory_admission(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Whether a theory-budget manifest admits any nuclide to a physical-accuracy comparison.

    Every term of :data:`REQUIRED_TERMS` must be present with a value and a source (a term with no value is
    unbounded; a value with no source, a zero included, is refused, because an unknown term cannot be zero);
    every admitted nuclide must lie in the provider's domain; every reference must be independent of the
    evaluated formula and not exposed before the manifest was frozen. A manifest that fails any of these
    admits nothing (``NOT_ESTABLISHED``); one that passes them with an empty roster is ``SCREEN_VACUOUS``;
    otherwise ``ADMITTED``."""
    problems: list[str] = []
    terms = manifest.get("terms", {})
    for name in REQUIRED_TERMS:
        term = terms.get(name)
        if term is None:
            problems.append(f"{name}: missing")
        elif term.get("value") in (None, "", "unbounded"):
            problems.append(f"{name}: unbounded")
        elif not term.get("source"):
            problems.append(f"{name}: a value with no source")
    roster = list(manifest.get("qualified_roster", []))
    for label in roster:
        z, a = (int(x) for x in str(label).split("-"))
        if not in_domain(z, a):
            problems.append(f"{label}: outside the provider's domain")
    for ref in manifest.get("references", []):
        if not ref.get("independent", False):
            problems.append(f"reference {ref.get('id')}: shares the evaluated formula or its inputs")
        if ref.get("exposed", True):
            problems.append(f"reference {ref.get('id')}: exposed before the manifest was frozen")
    if roster and not manifest.get("references"):
        problems.append("an admitted nuclide with no reference")
    if problems:
        status = "NOT_ESTABLISHED"
    elif not roster:
        status = "SCREEN_VACUOUS"
    else:
        status = "ADMITTED"
    return {"status": status, "problems": problems, "roster": roster}

