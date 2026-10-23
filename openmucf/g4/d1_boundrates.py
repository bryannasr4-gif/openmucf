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
  the tolerances. ``scripts/generate_g4boundrates.py`` generates it; it was committed before the production
  harvest ran.
* :func:`parse_record`, :func:`summarize_chunk` and :func:`merge` -- every record line parsed once into a
  typed record (a nonfinite, unobserved or malformed field refuses the line before any statistic sees it),
  and the sufficient statistics of a harvest chunk and of a group, from the files
  ``cpp/tools/harvest_mixture.cc`` writes with ``--bound-record on``. Standard library only, so they run on
  a Geant4 host. :data:`ANALYSIS_VERSION` names this implementation in every summary it writes.
* the gates, from the committed summary: rates, completeness, the waiting-time distribution family and the two
  clock questions (the time the observed implementation delivers, and the physically correct time).

Rates are in ns^-1, Geant4's unit; times in ns."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

from openmucf.g4 import d1_lifetime as lifetime
from openmucf.g4 import d1_mixture as mixture
from openmucf.g4 import d1_natural as natural
from openmucf.g4.sources import d1_nuclear_capture as d1

#: The committed policy, relative to the repository root.
POLICY_RELPATH = "data/g4/d1/bound_rates_policy.json"
#: The committed summary of the harvest and the builds it ran on.
HARVEST_RELPATH = "data/g4/d1/bound_rates_harvest.csv"
BUILDS_RELPATH = "data/g4/d1/bound_rates_builds.csv"
#: The committed results: four families with separate status fields.
RESULTS_RELPATH = "data/g4/d1/bound_rates_results.json"
#: The status of the theory budget the physical-accuracy question needs.
THEORY_RELPATH = "data/g4/d1/bound_rates_theory.json"
#: The analysis implementation: 1 summarized the production harvest first; 2 parses every record into a typed,
#: validated record before any statistic, validates the helper's provider call by nucleus, status and pair,
#: compares the rates the atomic consumer recovers from the stored lifetimes, and authenticates every summary
#: against the policy's registered roster. The same raw records summarized by both give the same registered
#: outcomes (the production re-summary is committed beside the first summary).
ANALYSIS_VERSION = 3

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


def chunks(group: Mapping[str, Any], chunk_events: int = CHUNK_EVENTS) -> list[dict[str, Any]]:
    """The production chunks of a group: ``chunk_events`` (the policy's, :data:`CHUNK_EVENTS`) contiguous
    events each, inside one stratum."""
    out = []
    for stratum in group["strata"]:
        if stratum["count"] % chunk_events:
            raise BoundRateError(f"{group_name(group)}: a stratum is not a whole number of chunks")
        for k in range(stratum["count"] // chunk_events):
            out.append({"t0_ns": stratum["t0_ns"], "first": stratum["first"] + k * chunk_events,
                        "count": chunk_events})
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
# the raw records of a chunk
# --------------------------------------------------------------------------------------------


def _open_text(path: Path, digests: dict[str, str] | None = None) -> Iterator[str]:
    """The lines of a raw file, ``path`` or its gzipped copy, decoded as ASCII. With ``digests``, the SHA-256
    of every uncompressed byte read is recorded under the file's name once the file has been read to its end
    (the digest ``cpp/tools/mixture_harvest.py run`` records before it compresses), so the digest is of the
    bytes the analysis consumed, recomputed, never read from a record."""
    gz = path.with_name(path.name + ".gz")
    if gz.is_file():
        source = gz
    elif path.is_file():
        source = path
    else:
        raise BoundRateError(f"missing {path}")
    h = hashlib.sha256()
    with _open_raw(source) as handle:
        for raw in handle:
            h.update(raw)
            yield raw.decode("ascii")
    if digests is not None:
        digests[path.name] = h.hexdigest()


def _open_raw(path: Path) -> BinaryIO:
    """``path`` opened for its uncompressed bytes: decompressed when its name ends in ``.gz``."""
    if path.name.endswith(".gz"):
        return cast(BinaryIO, gzip.open(path, "rb"))
    return path.open("rb")


def file_digest(path: Path) -> str:
    """The SHA-256 of a raw file's uncompressed bytes (``path``, decompressed when it ends in ``.gz``)."""
    h = hashlib.sha256()
    with _open_raw(path) as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _hex(token: str) -> float:
    """A required binary64 field (``%a``): refused when unobserved (``x`` or ``-``) or not finite."""
    if token in ("x", "-"):
        raise BoundRateError(f"unobserved field {token!r} where a value is required")
    value = float.fromhex(token)
    if not math.isfinite(value):
        raise BoundRateError(f"nonfinite field {token!r}")
    return value


def _positive(token: str) -> float:
    """A required finite, positive binary64 field (a lifetime)."""
    value = _hex(token)
    if not value > 0.0:
        raise BoundRateError(f"field {token!r} is not positive")
    return value


def _nc_lifetime(token: str, ref: Pair) -> float:
    """The stored nuclear-capture lifetime: finite and positive, or, only when the registered capture rate is
    exactly zero, the documented representation of 1/0, +inf. Any other nonfinite value is refused."""
    if token in ("x", "-"):
        raise BoundRateError(f"unobserved field {token!r} where a value is required")
    value = float.fromhex(token)
    if math.isfinite(value) and value > 0.0:
        return value
    if value == math.inf and ref.capture == 0.0:
        return value
    raise BoundRateError(f"nuclear-capture lifetime {token!r} with a registered capture rate {ref.capture!r}")


def _count(token: str) -> int:
    """A required count: a non-negative decimal integer."""
    if not token.isdigit():
        raise BoundRateError(f"count field {token!r} is not a non-negative integer")
    return int(token)


#: The provider's statuses as the harness writes them: -1 no call yet, 0 kCommon, 1 kDisabled, 2 kUnsupported,
#: 3 kRefused.
PROVIDER_STATUSES = (-1, 0, 1, 2, 3)


def _status(token: str) -> int:
    if token not in {str(v) for v in PROVIDER_STATUSES}:
        raise BoundRateError(f"provider status {token!r}")
    return int(token)


def _pair(status: int, capture: str, decay: str) -> tuple[float, float] | None:
    """The pair a provider call returned: a finite capture rate >= 0 and a finite decay rate > 0 when the
    status is common, both fields ``x`` otherwise."""
    if status == 0:
        c, d = _hex(capture), _positive(decay)
        if c < 0.0:
            raise BoundRateError(f"negative capture rate {capture!r} in a common pair")
        return c, d
    if (capture, decay) != ("x", "x"):
        raise BoundRateError(f"a pair ({capture!r}, {decay!r}) with status {status}")
    return None


def _rel(got: float, want: float) -> float:
    """The relative deviation of ``got`` from ``want``, typed for the two references that are not finite and
    positive: an exact zero (a zero capture rate) and +inf (its reciprocal lifetime) match only exactly (0
    when equal, +inf otherwise)."""
    if want == 0.0 or math.isinf(want):
        return 0.0 if got == want else math.inf
    return abs(got - want) / abs(want)


def pair_lifetimes(capture: float, decay: float) -> tuple[float, float, float]:
    """``(1 / (c + d), 1 / d, 1 / c)`` in binary64, as the helper stores them; a zero capture rate gives the
    documented +inf nuclear-capture lifetime."""
    return 1.0 / (capture + decay), 1.0 / decay, (1.0 / capture if capture > 0.0 else math.inf)


def recovered_rates(lifetimes: Sequence[float]) -> tuple[float, float]:
    """``(decay, capture)`` as G4MuonicAtomDecay recovers them from the stored DIO and nuclear-capture
    lifetimes (``1 / lifetime``; an infinite lifetime gives an exact zero)."""
    return 1.0 / lifetimes[1], (0.0 if math.isinf(lifetimes[2]) else 1.0 / lifetimes[2])


def clock_tolerance(*times: float) -> float:
    """``1e-6 ns + 64 eps max |t|`` over the times a comparison involves."""
    return CLOCK_ABS_NS + CLOCK_EPS_FACTOR * EPS * max(abs(t) for t in times)


@dataclass
class ClockTally:
    """Delivery and physical-clock residuals of one category."""

    n: int = 0
    fail_delivery: int = 0
    fail_physical: int = 0
    max_ratio_delivery: float = 0.0
    max_ratio_physical: float = 0.0
    max_abs_physical: float = 0.0

    def add(self, observed: float, delivery: float, physical: float, *inputs: float) -> None:
        if not all(math.isfinite(v) for v in (observed, delivery, physical, *inputs)):
            raise BoundRateError("a clock comparison with a nonfinite time")
        self.n += 1
        tol = clock_tolerance(observed, delivery, physical, *inputs)
        rd = abs(observed - delivery)
        rp = abs(observed - physical)
        self.fail_delivery += rd > tol
        self.fail_physical += rp > tol
        self.max_ratio_delivery = max(self.max_ratio_delivery, rd / tol)
        self.max_ratio_physical = max(self.max_ratio_physical, rp / tol)
        self.max_abs_physical = max(self.max_abs_physical, rp)

    def merge(self, other: ClockTally) -> None:
        self.n += other.n
        self.fail_delivery += other.fail_delivery
        self.fail_physical += other.fail_physical
        self.max_ratio_delivery = max(self.max_ratio_delivery, other.max_ratio_delivery)
        self.max_ratio_physical = max(self.max_ratio_physical, other.max_ratio_physical)
        self.max_abs_physical = max(self.max_abs_physical, other.max_abs_physical)

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "fail_delivery": self.fail_delivery, "fail_physical": self.fail_physical,
                "max_ratio_delivery": self.max_ratio_delivery.hex(),
                "max_ratio_physical": self.max_ratio_physical.hex(),
                "max_abs_physical_ns": self.max_abs_physical.hex()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ClockTally:
        return cls(int(d["n"]), int(d["fail_delivery"]), int(d["fail_physical"]),
                   float.fromhex(d["max_ratio_delivery"]), float.fromhex(d["max_ratio_physical"]),
                   float.fromhex(d["max_abs_physical_ns"]))


#: The fields of one clock category's record (:meth:`ClockTally.as_dict`); a category with any other field set
#: is missing.
CLOCK_FIELDS = frozenset(ClockTally().as_dict())


class ExactSum:
    """An exactly rounded running sum (Shewchuk's partials, as ``math.fsum`` keeps them), so a chunk's delays
    need not be held in memory."""

    def __init__(self) -> None:
        self.partials: list[float] = []

    def add(self, x: float) -> None:
        i = 0
        for y in self.partials:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                self.partials[i] = lo
                i += 1
            x = hi
        self.partials[i:] = [x]

    def value(self) -> float:
        return math.fsum(self.partials)


class _StopsCursor:
    """The stops file read in step with the bound record: both list events in increasing order, so the line of
    an event is found by advancing; a stops line no bound line asks for is passed over and still counted."""

    def __init__(self, path: Path, digests: dict[str, str] | None = None) -> None:
        self._lines = _open_text(path, digests)
        self._pending: list[str] | None = None
        self.count = 0

    def _next(self) -> list[str] | None:
        for line in self._lines:
            f = line.split()
            if f:
                self.count += 1
                return f
        return None

    def find(self, event: int) -> list[str] | None:
        while True:
            if self._pending is None:
                self._pending = self._next()
                if self._pending is None:
                    return None
            at = int(self._pending[0])
            if at == event:
                found, self._pending = self._pending, None
                return found
            if at > event:
                return None
            self._pending = None

    def drain(self) -> int:
        while self._next() is not None:
            pass
        return self.count


def _bin(u: float) -> int | None:
    """The equal-probability Exp(1) bin of ``u``, with the inherited exact rule near an edge."""
    return mixture._bin_of_time(u, lambda: Fraction(u))


def pair_digest(values: Sequence[float]) -> str:
    """SHA-256 over the big-endian binary64 bytes of ``values``."""
    h = hashlib.sha256()
    for v in values:
        h.update(struct.pack(">d", v))
    return h.hexdigest()


@dataclass(frozen=True)
class ProviderCall:
    """A ``C`` line: a call the provider received from the muonic-atom helper or from another caller."""

    event: int
    caller: str
    z: int
    a: int
    status: int
    pair: tuple[float, float] | None


@dataclass(frozen=True)
class EventRecord:
    """An ``S`` (stopping) or ``A`` (atomic) line, every field parsed and validated."""

    kind: str
    event: int
    z: int
    a: int
    elapsed: float
    outcome: str
    branch: str
    rate_call: tuple[int, int, int] | None
    pair: tuple[float, float] | None
    t_create: float
    lifetimes: tuple[float, float, float] | None
    counter: float
    t_atom: float
    model_calls: int | None
    taus: tuple[float, ...]
    products: tuple[float, ...]
    electron: float | None


def parse_record(f: Sequence[str], route: str, ref: Pair) -> ProviderCall | EventRecord:
    """One line of ``bound.txt`` as a typed record, or :class:`BoundRateError` (or
    ``ValueError``/``IndexError`` from a token that does not parse): an unobserved required field, a
    nonfinite value (bar the documented zero-capture lifetime), a negative elapsed time, a count that is not
    a non-negative integer, a section out of place or a token left over. Nothing is coerced."""
    tag = f[0]
    if tag == "C":
        if len(f) != 8:
            raise BoundRateError(f"a provider-call line of {len(f)} fields")
        status = _status(f[5])
        return ProviderCall(_count(f[1]), f[2], _count(f[3]), _count(f[4]), status, _pair(status, f[6], f[7]))
    if tag != ("S" if route == STOPPING else "A"):
        raise BoundRateError(f"a {tag!r} record on the {route} route")
    event, z, a = _count(f[1]), _count(f[2]), _count(f[3])
    rate_call: tuple[int, int, int] | None = None
    pair: tuple[float, float] | None = None
    lifetimes: tuple[float, float, float] | None = None
    t_create = counter = t_atom = 0.0
    branch = ""
    if tag == "S":
        t_before, t_after = _hex(f[4]), _hex(f[5])
        elapsed = t_after - t_before
        outcome = f[6]
        if f[7] != "P":
            raise BoundRateError("no P section")
        if f[8] == "x":
            m = 9
        else:
            rate_call = (_count(f[8]), _count(f[9]), _status(f[10]))
            pair = _pair(rate_call[2], f[11], f[12])
            m = 13
    else:
        t_create = _hex(f[4])
        lifetimes = (_positive(f[5]), _positive(f[6]), _nc_lifetime(f[7], ref))
        counter, elapsed, t_atom = _hex(f[8]), _hex(f[9]), _hex(f[10])
        branch, outcome = f[11], f[12]
        if branch not in ("D", "N"):
            raise BoundRateError(f"branch {branch!r}")
        if counter < 0.0:
            raise BoundRateError("a negative interaction-length counter")
        m = 13
    if outcome not in ("D", "N"):
        raise BoundRateError(f"outcome {outcome!r}")
    if not (math.isfinite(elapsed) and elapsed >= 0.0):
        raise BoundRateError(f"elapsed time {elapsed!r}")
    if f[m] != "M":
        raise BoundRateError("no M section")
    model_calls: int | None
    if f[m + 1] == "x":
        model_calls, taus, n = None, tuple[float, ...](), m + 2
    else:
        model_calls, k = _count(f[m + 1]), _count(f[m + 2])
        taus = tuple(_hex(t) for t in f[m + 3:m + 3 + k])
        if len(taus) != k:
            raise BoundRateError("fewer model-local times than the count")
        n = m + 3 + k
    if f[n] != "N":
        raise BoundRateError("no N section")
    n_products = _count(f[n + 1])
    products = tuple(_hex(t) for t in f[n + 2:n + 2 + n_products])
    if len(products) != n_products:
        raise BoundRateError("fewer product times than the count")
    e = n + 2 + n_products
    if f[e] != "E" or len(f) != e + 2:
        raise BoundRateError("the E section is not the last field")
    electron = None if f[e + 1] == "-" else _hex(f[e + 1])
    return EventRecord(tag, event, z, a, elapsed, outcome, branch, rate_call, pair, t_create, lifetimes,
                       counter, t_atom, model_calls, taus, products, electron)


def summarize_chunk(raw: Path, route: str, ref: Pair, t0_ns: float, first: int, count: int,
                    switch_on: bool = True) -> dict[str, Any]:
    """The sufficient statistics of one chunk: ``raw`` holds ``stops.txt``, ``bound.txt`` and
    ``bound_summary.txt`` (each possibly gzipped). ``ref`` is the frozen pair, ``t0_ns`` the stratum's start
    time, ``first`` and ``count`` the chunk's absolute events. Every line is parsed by :func:`parse_record`
    before it is counted; every malformed, missing, duplicate or unexpected record is counted, never
    coerced. On the atomic route a helper call counts only if it names the chunk's nucleus, returns the
    common status and a finite supported pair, and is the process's first such call; every atomic event is
    checked against it, and without it every atomic event fails that check."""
    lam = ref.total
    want = ((ref.capture, ref.decay) if route == STOPPING else ref.lifetimes)
    s: dict[str, Any] = {"route": route, "t0_ns": t0_ns, "first": first, "count": count,
                         "events": 0, "duplicates": 0, "out_of_range": 0, "malformed": 0,
                         "captures": 0, "decays": 0, "outcome_none": 0, "outcome_both": 0,
                         "branch_outcome_mismatch": 0, "stops_bound_mismatch": 0,
                         "rate_fail": 0, "rate_exact": 0, "rate_max_rel": 0.0, "rate_components": [],
                         "recovered_fail": 0, "recovered_exact": 0, "recovered_max_rel": 0.0,
                         "wiring_fail": 0, "provider_calls_helper": 0, "provider_calls_other": 0,
                         "helper_calls_valid": 0, "helper_calls_invalid": 0,
                         "helper_pair_exact": 0, "lifetimes_from_call_fail": 0,
                         "model_missing": 0, "model_product_mismatch": 0, "bin_ties": 0,
                         "wiring_unobserved": 0, "model_unobserved": 0}
    bins = {k: [0] * TIME_BINS for k in STREAMS}
    sums = {k: ExactSum() for k in STREAMS}
    clocks = {c: ClockTally() for c in CLOCK_CATEGORIES}
    seen = bytearray(count)  # the chunk's own events, one byte each
    seen_outside: set[int] = set()
    digests: dict[str, str] = {}
    stops = _StopsCursor(raw / "stops.txt", digests)
    helper_pair: tuple[float, float] | None = None
    for line in _open_text(raw / "bound.txt", digests):
        f = line.split()
        if not f:
            continue
        try:
            rec = parse_record(f, route, ref)
        except (ValueError, IndexError, BoundRateError):
            s["malformed"] += 1
            continue
        if isinstance(rec, ProviderCall):
            if rec.caller != "helper":
                s["provider_calls_other"] += 1
                continue
            s["provider_calls_helper"] += 1
            valid = (route == ATOMIC and (rec.z, rec.a) == (ref.z, ref.a) and rec.status == 0
                     and rec.pair is not None and helper_pair is None)
            if valid and rec.pair is not None:
                helper_pair = rec.pair
                s["helper_calls_valid"] += 1
                s["helper_pair_exact"] += rec.pair == (ref.capture, ref.decay)
            else:
                s["helper_calls_invalid"] += 1
            continue
        event = rec.event
        inside = first <= event < first + count
        if (seen[event - first] if inside else event in seen_outside):
            s["duplicates"] += 1
            continue
        if inside:
            seen[event - first] = 1
        else:
            seen_outside.add(event)
            s["out_of_range"] += 1
        s["events"] += 1
        model_seen = rec.model_calls is not None
        k, np_ = len(rec.taus), len(rec.products)
        s["model_unobserved"] += not model_seen
        stop_line = stops.find(event)
        if (rec.z, rec.a) != (ref.z, ref.a) or stop_line is None:
            s["stops_bound_mismatch"] += 1
            continue
        elapsed = rec.elapsed
        if route == STOPPING:
            captured = stop_line[9] == "1"
            if [stop_line[3], stop_line[4], stop_line[7], stop_line[8]] != list(f[2:6]):
                s["stops_bound_mismatch"] += 1
            if rec.rate_call is None:
                s["wiring_unobserved"] += 1
            elif switch_on:
                calls, other, status = rec.rate_call
                ok = calls == 1 and other == 0 and status == 0 and rec.pair is not None
                s["wiring_fail"] += not ok
                if ok and rec.pair is not None:
                    got = rec.pair
                    s["rate_exact"] += got == want
                    rels = [_rel(g, w) for g, w in zip(got + (got[0] + got[1],), want + (ref.total,),
                                                       strict=True)]
                    _component_max(s, rels)
                    rel = max(rels[:2])
                    s["rate_max_rel"] = max(s["rate_max_rel"], rel)
                    s["rate_fail"] += not rel <= RATE_RTOL
            else:
                s["wiring_fail"] += rec.rate_call[0] != 0 or rec.rate_call[1] != 0
        else:
            assert rec.lifetimes is not None
            captured = rec.branch == "N"
            if stop_line[5:12] != list(f[5:12]):
                s["stops_bound_mismatch"] += 1
            if elapsed != rec.counter * rec.lifetimes[0]:
                s["stops_bound_mismatch"] += 1
            s["rate_exact"] += rec.lifetimes == want
            rels = [_rel(g, w) for g, w in zip(rec.lifetimes, want, strict=True)]
            _component_max(s, rels)
            rel = max(rels)
            s["rate_max_rel"] = max(s["rate_max_rel"], rel)
            s["rate_fail"] += not rel <= RATE_RTOL
            back = recovered_rates(rec.lifetimes)
            s["recovered_exact"] += back == (ref.decay, ref.capture)
            back_rel = max(_rel(back[0], ref.decay), _rel(back[1], ref.capture))
            s["recovered_max_rel"] = max(s["recovered_max_rel"], back_rel)
            s["recovered_fail"] += not back_rel <= RATE_RTOL
            if switch_on:
                s["lifetimes_from_call_fail"] += (helper_pair is None
                                                  or rec.lifetimes != pair_lifetimes(*helper_pair))
        has_dio = rec.electron is not None
        has_capture = np_ > 0
        if has_dio and has_capture:
            s["outcome_both"] += 1
        elif not has_dio and not has_capture:
            s["outcome_none"] += 1
        if captured != has_capture or captured == has_dio or rec.outcome != ("N" if captured else "D"):
            s["branch_outcome_mismatch"] += 1
        if captured:
            s["captures"] += 1
            if model_seen and (rec.model_calls or 0) < 1:
                s["model_missing"] += 1
            if model_seen and k != np_:
                s["model_product_mismatch"] += 1
        else:
            s["decays"] += 1
        stream = "capture" if captured else "decay"
        sums["all"].add(elapsed)
        sums[stream].add(elapsed)
        j = _bin(lam * elapsed)
        if j is None:
            s["bin_ties"] += 1
        else:
            bins["all"][j] += 1
            bins[stream][j] += 1
        # clocks
        if route == STOPPING:
            if rec.electron is not None:
                clocks["stop_dio"].add(rec.electron, t0_ns + elapsed, t0_ns + elapsed, elapsed)
            if has_capture and model_seen and k == np_:
                for tau, t in zip(rec.taus, rec.products, strict=True):
                    clocks["stop_capture"].add(t, t0_ns + max(elapsed + tau, 0.0),
                                               t0_ns + elapsed + max(tau, 0.0), elapsed, tau)
        else:
            clocks["atom_creation"].add(rec.t_create, 2.0 * t0_ns, t0_ns)
            if rec.electron is not None:
                clocks["atom_dio"].add(rec.electron, rec.t_atom + elapsed, t0_ns + elapsed, elapsed,
                                       rec.t_atom)
            if has_capture and model_seen and k == np_:
                for tau, t in zip(rec.taus, rec.products, strict=True):
                    clocks["atom_capture"].add(t, rec.t_atom + max(elapsed + tau, 0.0),
                                               t0_ns + elapsed + max(tau, 0.0), elapsed, tau, rec.t_atom)
    s["missing"] = count - sum(seen)
    s["stops_lines"] = stops.drain()
    s["sums"] = {k: v.value().hex() for k, v in sums.items()}
    s["bins"] = bins
    s["clocks"] = {c: t.as_dict() for c, t in clocks.items()}
    s["rate_max_rel"] = float(s["rate_max_rel"]).hex()
    s["recovered_max_rel"] = float(s["recovered_max_rel"]).hex()
    s["rate_components"] = [float(x).hex() for x in s["rate_components"]]
    totals: dict[str, str] = {}
    try:
        summary_lines = list(_open_text(raw / "bound_summary.txt", digests))
    except BoundRateError:
        summary_lines = []
    try:
        for line in summary_lines:
            key, value = line.split()
            totals[key] = value
    except ValueError:
        totals = {}
    s["process"] = totals
    # Every raw file under a name the reader did not hash is hashed here, in sorted order; the run record
    # and the committed digests are checked against these. One digest is kept per name, so when a name is
    # held both plain and gzipped only one copy is hashed (the copy the reader consumed, else the plain one,
    # first in sorted order); every such name is listed in raw_both_forms, which raw_digest_problems names.
    forms: dict[str, int] = {}
    for path in sorted(raw.iterdir()):
        if not path.is_file():
            continue
        name = path.name[:-3] if path.name.endswith(".gz") else path.name
        forms[name] = forms.get(name, 0) + 1
        if name not in digests:
            digests[name] = file_digest(path)
    s["raw_sha256"] = dict(sorted(digests.items()))
    s["raw_both_forms"] = sorted(name for name, n in forms.items() if n > 1)
    wanted = {"events", "captures"} | {f"sum_{k}" for k in STREAMS}
    try:
        s["process_agrees"] = wanted <= set(totals) and (
            int(totals["events"]) == s["events"] and int(totals["captures"]) == s["captures"]
            and all(_close(_hex(totals[f"sum_{k}"]), float.fromhex(s["sums"][k])) for k in STREAMS))
    except (ValueError, BoundRateError):
        s["process_agrees"] = False
    return s


#: The components :func:`summarize_chunk` compares, by route: the stopping route's pair and their sum; the
#: atomic route's lifetime, DIO lifetime and nuclear-capture lifetime.
COMPONENTS = {STOPPING: ("capture", "decay", "total"), ATOMIC: ("lifetime", "dio_lifetime", "nc_lifetime")}


def _component_max(s: dict[str, Any], rels: Sequence[float]) -> None:
    """The largest relative deviation seen so far of each compared component."""
    cur = s["rate_components"]
    s["rate_components"] = [max(a, b) for a, b in zip(cur, rels, strict=True)] if cur else list(rels)


def _close(a: float, b: float) -> bool:
    """Two compensated sums of the same values agree to 1e-12 relative (or both are zero)."""
    return a == b or abs(a - b) <= 1e-12 * max(abs(a), abs(b))


# --------------------------------------------------------------------------------------------
# merging chunks into a group
# --------------------------------------------------------------------------------------------

_COUNTERS = ("events", "duplicates", "out_of_range", "malformed", "captures", "decays", "outcome_none",
             "outcome_both", "branch_outcome_mismatch", "stops_bound_mismatch", "rate_fail", "rate_exact",
             "recovered_fail", "recovered_exact", "wiring_fail", "provider_calls_helper",
             "provider_calls_other", "helper_calls_valid", "helper_calls_invalid", "helper_pair_exact",
             "lifetimes_from_call_fail", "model_missing", "model_product_mismatch", "bin_ties", "missing",
             "stops_lines", "wiring_unobserved", "model_unobserved")


def merge(chunk_stats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The statistics of a set of chunks (a stratum or a group): counters added, sums added with ``fsum`` over
    the chunk sums (each exactly rounded within its chunk), bins added, clock maxima taken."""
    if not chunk_stats:
        raise BoundRateError("no chunk to merge")
    out: dict[str, Any] = {k: sum(int(c[k]) for c in chunk_stats) for k in _COUNTERS}
    out["chunks"] = len(chunk_stats)
    # the chunks' event ranges must not overlap: a range harvested twice is a duplicated seed range
    spans = sorted((int(c["first"]), int(c["first"]) + int(c["count"])) for c in chunk_stats)
    out["chunk_overlap"] = sum(max(0, prev[1] - cur[0]) for prev, cur in zip(spans, spans[1:], strict=False))
    out["process_agrees"] = all(bool(c["process_agrees"]) for c in chunk_stats)
    out["rate_max_rel"] = max(float.fromhex(c["rate_max_rel"]) for c in chunk_stats).hex()
    out["recovered_max_rel"] = max(float.fromhex(c["recovered_max_rel"]) for c in chunk_stats).hex()
    comps = [c["rate_components"] for c in chunk_stats if c["rate_components"]]
    out["rate_components"] = ([max(float.fromhex(v[i]) for v in comps).hex() for i in range(len(comps[0]))]
                              if comps else [])
    out["sums"] = {k: math.fsum(float.fromhex(c["sums"][k]) for c in chunk_stats).hex() for k in STREAMS}
    out["bins"] = {k: [sum(c["bins"][k][j] for c in chunk_stats) for j in range(TIME_BINS)] for k in STREAMS}
    clocks = {}
    for cat in CLOCK_CATEGORIES:
        tally = ClockTally()
        for c in chunk_stats:
            tally.merge(ClockTally.from_dict(c["clocks"][cat]))
        clocks[cat] = tally.as_dict()
    out["clocks"] = clocks
    return out


# --------------------------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------------------------


def gate_complete(stats: Mapping[str, Any], events: int) -> dict[str, Any]:
    """Every event present once, with one terminal outcome observed in the transported products, its branch
    reconciled with that outcome, the stops and bound records consistent, the capture model observed for every
    capture, and the in-process sums equal to the ones recomputed from the records."""
    problems = [k for k in ("duplicates", "out_of_range", "malformed", "outcome_none", "outcome_both",
                            "branch_outcome_mismatch", "stops_bound_mismatch", "model_missing",
                            "model_product_mismatch", "missing", "wiring_unobserved", "model_unobserved")
                if int(stats[k])]
    if int(stats["events"]) != events:
        problems.append(f"events {stats['events']} != {events}")
    if int(stats["captures"]) + int(stats["decays"]) != int(stats["events"]):
        problems.append("branch counts do not add up to the events")
    if int(stats["stops_lines"]) != events:
        problems.append(f"stops lines {stats['stops_lines']} != {events}")
    if not stats["process_agrees"]:
        problems.append("the in-process sums disagree with the records")
    if int(stats.get("chunk_overlap", 0)):
        problems.append(f"{stats['chunk_overlap']} events harvested by more than one chunk")
    return {"verdict": "pass" if not problems else "fail", "problems": problems}


def gate_rate(stats: Mapping[str, Any], route: str, switch_on: bool = True) -> dict[str, Any]:
    """Every event's consumed rates within relative 1e-9 of the reference -- on the stopping route the pair
    returned to ``ApplyYourself``; on the atomic route the three stored lifetimes, and the decay and capture
    rates ``G4MuonicAtomDecay`` recovers from them -- and the wiring: on the stopping route one
    common-status call per event from bound decay and none from elsewhere; on the atomic route, with the
    switch on, exactly one valid helper call per process (the chunk's nucleus, the common status, a finite
    supported pair), no invalid one, and every stored lifetime equal to that call's pair; with the switch
    off, no helper call."""
    problems = []
    n = int(stats["events"])
    if n == 0:
        problems.append("no event whose rates could be checked")
    if int(stats["rate_fail"]):
        problems.append(f"{stats['rate_fail']} events outside the rate tolerance")
    if route == ATOMIC and int(stats["recovered_fail"]):
        problems.append(f"{stats['recovered_fail']} events whose recovered rates are outside the rate "
                        "tolerance")
    if int(stats["wiring_fail"]):
        problems.append(f"{stats['wiring_fail']} events with a wiring fault")
    if int(stats["provider_calls_other"]):
        problems.append(f"{stats['provider_calls_other']} provider calls from an unexpected caller")
    if route == ATOMIC:
        if switch_on:
            if int(stats["helper_calls_valid"]) != int(stats["chunks"]):
                problems.append(f"{stats['helper_calls_valid']} valid helper calls in {stats['chunks']} "
                                "processes")
            if int(stats["helper_calls_invalid"]):
                problems.append(f"{stats['helper_calls_invalid']} helper calls with another nucleus, a "
                                "status other than common, no usable pair, or a second call")
            if int(stats["lifetimes_from_call_fail"]):
                problems.append("stored lifetimes differ from the helper call's pair, or no valid call")
        elif int(stats["provider_calls_helper"]):
            problems.append("helper calls with the switch off")
    elif int(stats["provider_calls_helper"]):
        problems.append("helper calls on the stopping route")
    return {"verdict": "pass" if not problems else "fail", "problems": problems,
            "exact_events": int(stats["rate_exact"]), "events": n,
            "max_relative_deviation": stats["rate_max_rel"],
            "recovered_exact_events": int(stats["recovered_exact"]),
            "recovered_max_relative_deviation": stats["recovered_max_rel"]}


def gate_distribution(stats: Mapping[str, Any], n_tests: int) -> dict[str, dict[str, str]]:
    """The three Pearson tests of the scaled delays in equal-probability bins of Exp(1)."""
    alpha = per_test_alpha(n_tests)
    crit = mixture.critical(TIME_BINS - 1, alpha)
    out = {}
    for stream in STREAMS:
        counts = [int(c) for c in stats["bins"][stream]]
        if sum(counts) == 0:
            out[stream] = {"verdict": "fail", "reason": "empty stream"}
            continue
        x2 = mixture.pearson(counts, [Fraction(1, TIME_BINS)] * TIME_BINS)
        verdict = mixture.verdict(x2, crit)
        if int(stats["bin_ties"]):
            verdict = "refuse"
        out[stream] = {"verdict": verdict, "x2": str(x2[0].quantize(Decimal("1e-6"))),
                       "critical_lo": str(crit.lo.quantize(Decimal("1e-6"))), "n": str(sum(counts))}
    return out


def gate_clocks(strata: Mapping[int, Mapping[str, Any]], route: str,
                registered: Sequence[int]) -> dict[str, Any]:
    """Delivery (every category of every registered stratum: at least one item and zero failures) and physical
    correctness (each category and stratum's verdict beside the one predicted before the harvest). A
    registered stratum or category that is absent, or one with no item, is named, fails delivery and makes
    the physical verdict ``incomplete``; a stratum outside ``registered`` is named and fails delivery."""
    cats = (("stop_dio", "stop_capture") if route == STOPPING
            else ("atom_creation", "atom_dio", "atom_capture"))
    delivery_problems = []
    physical: dict[str, dict[str, Any]] = {}
    by_t0 = {int(t0): v for t0, v in strata.items()}
    if not registered:
        delivery_problems.append("no registered stratum")
    for t0 in sorted(set(by_t0) - {int(t) for t in registered}):
        delivery_problems.append(f"t0={t0}: not a registered stratum")
    for t0 in sorted(int(t) for t in registered):
        stats = by_t0.get(t0)
        for cat in cats:
            key = f"{cat}@{t0}"
            predicted = PREDICTED_PHYSICAL[cat][t0]
            row = stats["clocks"].get(cat) if stats is not None else None
            if row is None or set(row) != CLOCK_FIELDS:
                delivery_problems.append(f"t0={t0} {cat}: missing")
                physical[key] = {"verdict": "missing", "predicted": predicted, "as_predicted": False, "n": 0,
                                 "fail": 0, "max_abs_residual_ns": (0.0).hex()}
                continue
            tally = ClockTally.from_dict(row)
            if tally.n == 0:
                delivery_problems.append(f"t0={t0} {cat}: no item")
            if tally.fail_delivery:
                delivery_problems.append(f"t0={t0} {cat}: {tally.fail_delivery} of {tally.n}")
            observed = "no_item" if tally.n == 0 else ("fail" if tally.fail_physical else "pass")
            physical[key] = {"verdict": observed, "predicted": predicted,
                             "as_predicted": observed == predicted, "n": tally.n,
                             "fail": tally.fail_physical,
                             "max_abs_residual_ns": float.fromhex(row["max_abs_physical_ns"]).hex()}
    delivery = {"verdict": "pass" if not delivery_problems else "fail", "problems": delivery_problems}
    verdicts = {v["verdict"] for v in physical.values()}
    if not physical or verdicts & {"missing", "no_item"}:
        physical_verdict = "incomplete"
    else:
        physical_verdict = "fail" if verdicts != {"pass"} else "pass"
    return {"delivery": delivery, "physical": physical, "physical_verdict": physical_verdict}


# --------------------------------------------------------------------------------------------
# the committed summary: rows of (cell, stat, key, value)
# --------------------------------------------------------------------------------------------

HARVEST_COLUMNS = ("cell", "stat", "key", "value")
BUILD_COLUMNS = ("build", "revision", "kind", "trace", "drill", "install_kind", "geant4_version", "gcc",
                 "compile_flags", "source_sha256", "binary_sha256", "particles_sha256", "hadronic_sha256")


def stats_rows(cell: str, stat: str, stats: Mapping[str, Any]) -> list[dict[str, str]]:
    """One block of statistics (a stratum or a group) as rows; nested fields are dotted keys."""
    rows = []
    for key in _COUNTERS + ("chunks", "chunk_overlap"):
        rows.append({"cell": cell, "stat": stat, "key": key, "value": str(int(stats[key]))})
    rows.append({"cell": cell, "stat": stat, "key": "process_agrees",
                 "value": str(bool(stats["process_agrees"]))})
    rows.append({"cell": cell, "stat": stat, "key": "rate_max_rel", "value": str(stats["rate_max_rel"])})
    rows.append({"cell": cell, "stat": stat, "key": "recovered_max_rel",
                 "value": str(stats["recovered_max_rel"])})
    rows.append({"cell": cell, "stat": stat, "key": "rate_components",
                 "value": " ".join(stats["rate_components"])})
    for k in STREAMS:
        rows.append({"cell": cell, "stat": stat, "key": f"sum.{k}", "value": str(stats["sums"][k])})
        for j, count in enumerate(stats["bins"][k]):
            rows.append({"cell": cell, "stat": stat, "key": f"bin.{k}.{j}", "value": str(int(count))})
    for cat in CLOCK_CATEGORIES:
        for field, value in stats["clocks"][cat].items():
            rows.append({"cell": cell, "stat": stat, "key": f"clock.{cat}.{field}", "value": str(value)})
    return rows


def rows_stats(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """The inverse of :func:`stats_rows` for the rows of one (cell, stat)."""
    values = {r["key"]: r["value"] for r in rows}
    out: dict[str, Any] = {key: int(values[key]) for key in _COUNTERS + ("chunks", "chunk_overlap")}
    out["process_agrees"] = values["process_agrees"] == "True"
    out["rate_max_rel"] = values["rate_max_rel"]
    out["recovered_max_rel"] = values["recovered_max_rel"]
    out["rate_components"] = values["rate_components"].split()
    out["sums"] = {k: values[f"sum.{k}"] for k in STREAMS}
    out["bins"] = {k: [int(values[f"bin.{k}.{j}"]) for j in range(TIME_BINS)] for k in STREAMS}
    clocks: dict[str, dict[str, Any]] = {}
    for cat in CLOCK_CATEGORIES:
        fields = {key.split(".", 2)[2]: value for key, value in values.items()
                  if key.startswith(f"clock.{cat}.")}
        clocks[cat] = {f: (int(v) if f in ("n", "fail_delivery", "fail_physical") else v)
                       for f, v in fields.items()}
    out["clocks"] = clocks
    return out


def render_csv(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> bytes:
    return mixture.render_csv(rows, columns)


def read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    return mixture.read_csv(path, columns)


def load_harvest(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """``{cell: {stat: statistics}}`` from the committed summary; the ``meta`` stat stays a flat dict."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in read_csv(root / HARVEST_RELPATH, HARVEST_COLUMNS):
        key = (row["cell"], row["stat"], row["key"])
        if key in seen:
            raise BoundRateError(f"{HARVEST_RELPATH}: {key} appears twice")
        seen.add(key)
        grouped.setdefault((row["cell"], row["stat"]), []).append(row)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (cell, stat), rows in grouped.items():
        out.setdefault(cell, {})[stat] = ({r["key"]: r["value"] for r in rows} if stat == "meta"
                                          else rows_stats(rows))
    return out


def run_provenance_problems(policy: Mapping[str, Any], group: Mapping[str, Any], chunk: Mapping[str, Any],
                            run: Mapping[str, Any], switch_on: bool = True) -> list[str]:
    """The run record of one chunk (``run.json`` of ``cpp/tools/mixture_harvest.py run``) against the chunk
    the policy registers at the chunk's first event: the plan's count and start time must be that chunk's, and
    the recorded command line must name the group's route, element, Z, A and seed base, the chunk's event
    count, event offset and start time (the last two absent at their defaults of 0), the bound-rate switch and
    record, hooks on, dataset opt-in and identity mode off, the chunk's own output directory and no other
    option, and a binary of the group's revision's traced bound-rate build that exited 0. Each disagreement is
    named."""
    first = int(chunk["first"])
    where = f"chunk at {first}"
    registered = {int(c["first"]): c for c in chunks(group, int(policy.get("chunk_events", CHUNK_EVENTS)))}
    reg = registered.get(first)
    if (reg is None or int(chunk["count"]) != int(reg["count"])
            or int(float(chunk["t0_ns"])) != int(reg["t0_ns"])):
        return [f"{where}: not a chunk the policy registers"]
    t0 = int(reg["t0_ns"])
    want: dict[str, str | None] = {
        "--route": str(group["route"]), "--symbol": str(group["isotope"]).split("-")[0],
        "--Z": str(group["z"]), "--isotope": str(group["a"]), "--events": str(reg["count"]),
        "--seed-base": str(group["seed_base"]),
        "--opt-in": "off", "--hooks": "on", "--identity": "off",
        "--bound-rates": "on" if switch_on else "off", "--bound-record": "on",
        "--t0": str(t0) if t0 else None, "--event-offset": str(first) if first else None}
    revision = group["revision"]
    return _command_problems(where, run, want, Path(chunk["cell"]) / "raw", f"{revision}-bound-trace",
                             f"the {revision} traced bound-rate build")


def _command_problems(where: str, run: Mapping[str, Any], want: Mapping[str, str | None], out: Path,
                      build: str, build_words: str) -> list[str]:
    """A run record's command line against the options registered for it (``want``; ``None``: absent), its
    own output directory ``out`` and a binary in a build directory whose path ends in ``build``; and its exit
    code. Each disagreement is named."""
    argv = run.get("argv")
    if (not isinstance(argv, list) or len(argv) < 3 or len(argv) % 2 == 0
            or not all(isinstance(a, str) for a in argv)):
        return [f"{where}: no readable command line"]
    pairs = list(zip(argv[1::2], argv[2::2], strict=True))
    flags = dict(pairs)
    problems = []
    if len(flags) != len(pairs):
        problems.append(f"{where}: an option given twice")
    for flag, value in want.items():
        if flags.get(flag) != value:
            problems.append(f"{where}: {flag} {flags.get(flag)!r}, registered {value!r}")
    if "--out" not in flags or Path(flags["--out"]) != out:
        problems.append(f"{where}: output {flags.get('--out')!r} is not the chunk's own")
    extra = sorted(set(flags) - set(want) - {"--out"})
    if extra:
        problems.append(f"{where}: options the policy does not register: {' '.join(extra)}")
    binary = PurePosixPath(argv[0].replace("\\", "/")).parent.as_posix()
    if not (binary == build or binary.endswith("/" + build)):
        problems.append(f"{where}: binary {argv[0]!r} is not {build_words}")
    if run.get("rc") != 0:
        problems.append(f"{where}: exit code {run.get('rc')!r}")
    return problems


#: The raw files the analysis consumes; the reader hashes each as it reads it to its end.
CONSUMED_RAW = ("bound.txt", "bound_summary.txt", "stops.txt")
#: The committed SHA-256 of every raw file of every production chunk and drill run, by summary cell and first
#: event: the identity a re-summary is checked against, beside each run's own record.
RAW_DIGESTS_RELPATH = "data/g4/d1/bound_rates_raw_digests.csv"
RAW_DIGEST_COLUMNS = ("cell", "first", "file", "sha256")


def raw_digest_problems(where: str, stats: Mapping[str, Any], recorded: Any,
                        registered: Mapping[str, str] | None) -> list[str]:
    """The raw files of one chunk as :func:`summarize_chunk` hashed them (``stats["raw_sha256"]``: file
    name -> SHA-256 of the uncompressed bytes; ``stats["raw_both_forms"]``: the names held both plain and
    gzipped), against the run's own record (``run.json`` ``files``) and against the committed digests
    (``registered``, from :data:`RAW_DIGESTS_RELPATH`). A consumed file without a digest is named
    (``summarize_chunk`` hashes every file present, so it is absent from the raw directory); every hashed
    file must be in the record and in the committed set with the same digest; neither may name a file the
    raw directory lacks; a name held both plain and gzipped is named, since only one of its copies was
    hashed; a summary that does not list those names is named. A raw file swapped after the run differs
    from its record; a record edited to match the swapped file still differs from the committed digest.
    Each disagreement is named."""
    recomputed = stats.get("raw_sha256")
    recomputed = recomputed if isinstance(recomputed, Mapping) else {}
    recorded = recorded if isinstance(recorded, Mapping) else {}
    registered_: Mapping[str, str] = registered if registered is not None else {}
    problems = [f"{where}: {f}, which the analysis consumes, has no digest: it is not in the raw directory"
                for f in CONSUMED_RAW if f not in recomputed]
    for f in sorted(set(recomputed) | set(recorded) | set(registered_)):
        got, rec, reg = recomputed.get(f), recorded.get(f), registered_.get(f)
        if got is None:
            problems.append(f"{where}: {f} is recorded but not in the raw directory")
            continue
        if rec is None:
            problems.append(f"{where}: {f} is not in its run record")
        elif got != rec:
            problems.append(f"{where}: {f} differs from its run record")
        if reg is None:
            problems.append(f"{where}: {f} has no committed digest")
        elif got != reg:
            problems.append(f"{where}: {f} differs from its committed digest")
    both = stats.get("raw_both_forms")
    if not isinstance(both, list):
        problems.append(f"{where}: the summary does not list the raw names held both plain and gzipped")
    else:
        problems += [f"{where}: {f} is held both plain and gzipped; one copy was hashed and the other was not"
                     for f in both]
    return problems


def load_raw_digests(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    """``{(cell, first): {file: sha256}}`` from :data:`RAW_DIGESTS_RELPATH`; empty when it is absent. A row
    given twice is refused."""
    path = root / RAW_DIGESTS_RELPATH
    out: dict[tuple[str, int], dict[str, str]] = {}
    if not path.is_file():
        return out
    for row in read_csv(path, RAW_DIGEST_COLUMNS):
        files = out.setdefault((row["cell"], int(row["first"])), {})
        if row["file"] in files:
            raise BoundRateError(f"{RAW_DIGESTS_RELPATH}: {row['cell']} {row['first']} {row['file']} twice")
        files[row["file"]] = row["sha256"]
    return out


def raw_digest_rows(digests: Mapping[tuple[str, int], Mapping[str, str]]) -> list[dict[str, str]]:
    return [{"cell": cell, "first": str(first), "file": f, "sha256": sha}
            for (cell, first), files in sorted(digests.items()) for f, sha in sorted(files.items())]


def drill_run_problems(job: Mapping[str, Any], chunk_dir: Path, run: Mapping[str, Any]) -> list[str]:
    """One drill or rerun chunk's run record against its registered job: route, element, Z, A, event count,
    seed base and event offset (absent at 0), the switch and record on, hooks on, dataset opt-in and identity
    mode off, the job's extra options and no other, its own output directory, a binary of the job's build, and
    exit code 0. A job directory holding another chunk's run differs in its seed base, offset or output."""
    first = int(job.get("event_offset", 0))
    where = f"run at {first}"
    want: dict[str, str | None] = {
        "--route": str(job["route"]), "--symbol": str(job["symbol"]), "--Z": str(job["z"]),
        "--isotope": str(job["a"]), "--events": str(job["events"]), "--seed-base": str(job["seed_base"]),
        "--opt-in": "off", "--hooks": "on", "--identity": "off",
        "--bound-rates": "on" if job.get("switch_on", True) else "off", "--bound-record": "on",
        "--event-offset": str(first) if first else None}
    extra = list(job.get("extra", []))
    if len(extra) % 2:
        return [f"{where}: the job's extra options are not option-value pairs"]
    want.update(dict(zip(extra[0::2], extra[1::2], strict=True)))
    return _command_problems(where, run, want, chunk_dir / "raw", str(job["build"]),
                             f"the build {job['build']}")


def summarize_plan(root: Path, plan: Mapping[str, Any], out_dir: Path,
                   stats_name: str = "bound_stats.json",
                   registered_digests: Mapping[tuple[str, int], Mapping[str, str]] | None = None) -> None:
    """Summarize every chunk a plan names and write the summary and builds files into ``out_dir``. A plan is
    ``{"switch_on": bool, "chunks": [{"group": g, "first": i, "count": n, "t0_ns": t, "cell": dir}], "builds":
    {revision: build dir}}``; a chunk's ``cell`` holds the ``run.json`` and ``raw/`` of
    ``cpp/tools/mixture_harvest.py run``. Each chunk's statistics are also written as JSON beside it, and with
    ``"reuse": true`` a chunk whose JSON exists is read back rather than summarized again. Every chunk's run
    record is checked against the policy (:func:`run_provenance_problems`); the group's meta rows carry the
    count of disagreeing chunks and the first disagreement."""
    policy = load_policy(root)
    by_group: dict[int, list[dict[str, Any]]] = {}
    meta: dict[int, dict[str, str]] = {}
    provenance: dict[int, list[str]] = {}
    digest_problems: dict[int, list[str]] = {}
    registered = load_raw_digests(root) if registered_digests is None else registered_digests
    for chunk in plan["chunks"]:
        g = int(chunk["group"])
        group = policy["groups"][g]
        cell = Path(chunk["cell"])
        saved = cell / stats_name
        if plan.get("reuse") and saved.is_file():
            stats = json.loads(saved.read_text(encoding="ascii"))
        else:
            stats = summarize_chunk(cell / "raw", group["route"], frozen_pair(policy, group["isotope"]),
                                    float(chunk["t0_ns"]), int(chunk["first"]), int(chunk["count"]),
                                    bool(plan["switch_on"]))
            saved.write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="ascii")
        run = json.loads((cell / "run.json").read_text(encoding="utf-8"))
        provenance.setdefault(g, []).extend(run_provenance_problems(policy, group, chunk, run,
                                                                    bool(plan["switch_on"])))
        recomputed = stats.get("raw_sha256", {})
        digest_problems.setdefault(g, []).extend(raw_digest_problems(
            f"chunk at {int(chunk['first'])}", stats, run.get("files"),
            registered.get((group_name(group), int(chunk["first"])))))
        by_group.setdefault(g, []).append(stats)
        m = meta.setdefault(g, {"binary_sha256": run["build_binary_sha256"], "files": ""})
        if m["binary_sha256"] != run["build_binary_sha256"]:
            raise BoundRateError(f"group {g}: chunks from different binaries")
        m["files"] += "".join(f"{name}={digest};" for name, digest in sorted(recomputed.items()))
    rows: list[dict[str, str]] = []
    for g in sorted(by_group):
        group = policy["groups"][g]
        name = group_name(group)
        chunk_stats = sorted(by_group[g], key=lambda s: int(s["first"]))
        rows.append({"cell": name, "stat": "meta", "key": "binary_sha256", "value": meta[g]["binary_sha256"]})
        rows.append({"cell": name, "stat": "meta", "key": "chunk_files_sha256",
                     "value": hashlib.sha256(meta[g]["files"].encode("ascii")).hexdigest()})
        rows.append({"cell": name, "stat": "meta", "key": "switch_on", "value": str(bool(plan["switch_on"]))})
        rows.append({"cell": name, "stat": "meta", "key": "analysis_version", "value": str(ANALYSIS_VERSION)})
        rows.append({"cell": name, "stat": "meta", "key": "provenance_problems",
                     "value": str(len(provenance[g]))})
        rows.append({"cell": name, "stat": "meta", "key": "provenance_first",
                     "value": provenance[g][0] if provenance[g] else "-"})
        rows.append({"cell": name, "stat": "meta", "key": "raw_digest_problems",
                     "value": str(len(digest_problems[g]))})
        rows.append({"cell": name, "stat": "meta", "key": "raw_digest_first",
                     "value": digest_problems[g][0] if digest_problems[g] else "-"})
        rows += stats_rows(name, "group", merge(chunk_stats))
        for t0 in T0_NS:
            part = [s for s in chunk_stats if int(float(s["t0_ns"])) == t0]
            if part:
                rows += stats_rows(name, f"stratum_{t0}", merge(part))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / Path(HARVEST_RELPATH).name).write_bytes(render_csv(rows, HARVEST_COLUMNS))
    build_rows = []
    for revision, build_dir in sorted(plan.get("builds", {}).items()):
        info = json.loads((Path(build_dir) / "build.json").read_text(encoding="utf-8"))
        libs = {Path(lib["path"]).name: lib["sha256"] for lib in info["libraries"]}
        build_rows.append({
            "build": Path(build_dir).name, "revision": revision, "kind": info["kind"],
            "trace": str(bool(info.get("trace", False))), "drill": info["drill"],
            "install_kind": "bound-rate patched", "geant4_version": info["geant4_version"],
            "gcc": info["gcc"],
            "compile_flags": info["compile_flags"], "source_sha256": info["source_sha256"],
            "binary_sha256": info["binary_sha256"], "particles_sha256": libs.get("libG4particles.so", ""),
            "hadronic_sha256": libs.get("libG4processes_hadronic.so", "")})
    if build_rows:
        (out_dir / Path(BUILDS_RELPATH).name).write_bytes(render_csv(build_rows, BUILD_COLUMNS))


def roster_problems(policy: Mapping[str, Any], harvest: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The summary against the policy's registered roster, each problem named: no registered group; a group
    registered twice; a summary cell that is not a registered group; a registered group without its cell; a
    group whose chunk run records were not checked or disagree with the policy
    (:func:`run_provenance_problems`); a cell without its meta or group statistics or without a registered
    stratum, or with a stratum that is not registered; and a group or stratum whose event count or chunk count
    is not the registered one."""
    problems: list[str] = []
    groups_ = list(policy.get("groups", []))
    if not groups_:
        problems.append("the policy registers no group")
    names = [group_name(g) for g in groups_]
    for name in sorted({x for x in names if names.count(x) > 1}):
        problems.append(f"{name}: registered more than once")
    for name in sorted(set(harvest) - set(names)):
        problems.append(f"{name}: not a registered group")
    for g in groups_:
        name = group_name(g)
        cell = harvest.get(name)
        if cell is None:
            problems.append(f"{name}: missing")
            continue
        meta = cell.get("meta", {})
        if meta.get("raw_digest_problems") is None:
            problems.append(f"{name}: no re-hash of its raw files")
        elif meta["raw_digest_problems"] != "0":
            problems.append(f"{name}: {meta['raw_digest_problems']} raw-file digests disagree "
                            f"({meta.get('raw_digest_first')})")
        if meta.get("provenance_problems") is None:
            problems.append(f"{name}: no provenance check of its run records")
        elif meta["provenance_problems"] != "0":
            problems.append(f"{name}: {meta['provenance_problems']} run records disagree with the policy "
                            f"({meta.get('provenance_first')})")
        wanted = {"meta", "group"} | {f"stratum_{st['t0_ns']}" for st in g["strata"]}
        problems += [f"{name}: {stat} missing" for stat in sorted(wanted - set(cell))]
        problems += [f"{name}: {stat} is not registered" for stat in sorted(set(cell) - wanted)]
        if "group" in cell:
            if int(cell["group"]["events"]) != int(g["events"]):
                problems.append(f"{name}: {cell['group']['events']} events, {g['events']} registered")
            try:
                registered = len(chunks(g, int(policy.get("chunk_events", CHUNK_EVENTS))))
            except BoundRateError as exc:
                problems.append(str(exc))
            else:
                if int(cell["group"]["chunks"]) != registered:
                    problems.append(f"{name}: {cell['group']['chunks']} chunks, {registered} registered")
        for st in g["strata"]:
            part = cell.get(f"stratum_{st['t0_ns']}")
            if part is not None and int(part["events"]) != int(st["count"]):
                problems.append(f"{name}: stratum {st['t0_ns']} has {part['events']} events, {st['count']} "
                                "registered")
    return problems


# --------------------------------------------------------------------------------------------
# the theory admission gate
# --------------------------------------------------------------------------------------------

#: The terms a physical uncertainty budget of the prescription must carry before any nuclide is admitted:
#: the equation (a verified statement of the primary equation, its assumptions and domain) and the error
#: terms (each a bound on the term's contribution).
EQUATION_TERM = "equation"
ERROR_TERMS = ("binding_expansion", "recoil", "nuclear_mass", "finite_size", "radiative", "numerical")
REQUIRED_TERMS = (EQUATION_TERM, *ERROR_TERMS)


def _term_problems(name: str, term: object) -> list[str]:
    """A term of the budget against its schema. The equation term: ``status`` "VERIFIED", a non-empty
    ``statement`` and ``source``. An error term: ``status`` "BOUNDED", a ``bound`` that is a JSON number
    (not a string, not a boolean), finite and non-negative, and a non-empty ``source``. Any other status --
    UNKNOWN, UNBOUNDED, IMPLEMENTATION_ONLY or none -- admits nothing, whatever bound is written beside it."""
    if not isinstance(term, Mapping):
        return [f"{name}: missing"]
    problems = []
    source = term.get("source")
    if not isinstance(source, str) or not source.strip():
        problems.append(f"{name}: no source")
    status = term.get("status")
    if name == EQUATION_TERM:
        if status != "VERIFIED":
            problems.append(f"{name}: status {status!r}, not VERIFIED")
        statement = term.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            problems.append(f"{name}: no statement")
        return problems
    if status != "BOUNDED":
        problems.append(f"{name}: status {status!r}, not BOUNDED")
    bound = term.get("bound")
    if isinstance(bound, bool) or not isinstance(bound, (int, float)):
        problems.append(f"{name}: bound {bound!r} is not a number")
    elif not (math.isfinite(bound) and bound >= 0):
        problems.append(f"{name}: bound {bound!r} is not finite and non-negative")
    return problems


def theory_admission(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Whether a theory-budget manifest admits any nuclide to a physical-accuracy comparison.

    Every term of :data:`REQUIRED_TERMS` must satisfy its schema (:func:`_term_problems`); every admitted
    nuclide must lie in the provider's domain; every reference must state that it is independent of the
    evaluated formula and was not exposed before the manifest was frozen; an admitted nuclide needs a
    reference. A manifest that fails any of these admits nothing (``NOT_ESTABLISHED``); one that passes them
    with an empty roster is ``SCREEN_VACUOUS``; otherwise ``ADMITTED``."""
    problems: list[str] = []
    terms = manifest.get("terms", {})
    if not isinstance(terms, Mapping):
        terms = {}
    for name in REQUIRED_TERMS:
        problems += _term_problems(name, terms.get(name))
    roster = list(manifest.get("qualified_roster", []))
    for label in roster:
        z, a = (int(x) for x in str(label).split("-"))
        if not in_domain(z, a):
            problems.append(f"{label}: outside the provider's domain")
    for ref in manifest.get("references", []):
        if ref.get("independent") is not True:
            problems.append(f"reference {ref.get('id')}: not stated independent of the evaluated formula")
        if ref.get("exposed") is not False:
            problems.append(f"reference {ref.get('id')}: not stated unexposed before the manifest was frozen")
    if roster and not manifest.get("references"):
        problems.append("an admitted nuclide with no reference")
    if problems:
        status = "NOT_ESTABLISHED"
    elif not roster:
        status = "SCREEN_VACUOUS"
    else:
        status = "ADMITTED"
    return {"status": status, "problems": problems, "roster": roster}


# --------------------------------------------------------------------------------------------
# drills
# --------------------------------------------------------------------------------------------

#: The committed drill summaries (same rows as the harvest summary) and the identity verdicts.
DRILLS_RELPATH = "data/g4/d1/bound_rates_drills.csv"
IDENTITY_RELPATH = "data/g4/d1/bound_rates_identity.csv"
IDENTITY_VERDICTS_RELPATH = "data/g4/d1/bound_rates_identity_verdicts.csv"
IDENTITY_DRILLS_RELPATH = "data/g4/d1/bound_rates_identity_drills.csv"
SWEEP_RELPATH = "data/g4/d1/bound_rates_sweep.csv"
DEFAULTS_RELPATH = "data/g4/d1/bound_rates_harness_defaults.csv"
#: The families a drill can be intended to turn red.
FAMILIES = ("complete", "rate", "wiring", "transport", "distribution", "delivery")


def drill_families(stats: Mapping[str, Any], meta: Mapping[str, str], pair: Pair, stats_mod: Any,
                   n_primary: int = 48, n_distribution: int = 36) -> dict[str, str]:
    """Every family's verdict for one drill or control run: completeness, the rate values, the wiring, the
    four registered tests at the run's own N (``stats_mod`` supplies the binomial and gamma functions), the
    distribution family and the delivered clock."""
    route = meta["route"]
    n = int(meta["events"])
    switch_on = meta["switch_on"] == "True"
    out = {"complete": gate_complete(stats, n)["verdict"]}
    rate_bad = int(stats["rate_fail"]) or (route == ATOMIC and int(stats["recovered_fail"]))
    out["rate"] = "fail" if rate_bad or int(stats["events"]) == 0 else "pass"
    # the wiring as :func:`gate_rate` grades it: on the atomic route with the switch on, one valid helper call
    # per process and no invalid one, and every stored lifetime from that call
    wiring = bool(int(stats["wiring_fail"]) or int(stats["provider_calls_other"])
                  or int(stats["wiring_unobserved"]))
    if switch_on and route == ATOMIC:
        wiring = wiring or int(stats["helper_calls_valid"]) != int(stats["chunks"]) or bool(
            int(stats["helper_calls_invalid"]) or int(stats["lifetimes_from_call_fail"]))
    else:
        wiring = wiring or bool(int(stats["provider_calls_helper"]))
    out["wiring"] = "fail" if wiring else "pass"
    alpha = float(per_test_alpha(n_primary))
    k = int(stats["captures"])
    p = float(pair.p_capture)
    lo = int(stats_mod.binom.ppf(alpha / 2, n, p))
    hi = int(stats_mod.binom.isf(alpha / 2, n, p))
    verdicts = [lo <= k <= hi]
    for stream, shape in (("all", n), ("capture", k), ("decay", n - k)):
        if shape <= 0:
            verdicts.append(False)
            continue
        u = pair.total * float.fromhex(stats["sums"][stream])
        verdicts.append(float(stats_mod.gamma.ppf(alpha / 2, shape)) <= u
                        <= float(stats_mod.gamma.isf(alpha / 2, shape)))
    dist = gate_distribution(stats, n_distribution)
    out["distribution"] = "pass" if all(v["verdict"] == "pass" for v in dist.values()) else "fail"
    out["transport"] = "pass" if all(verdicts) and out["distribution"] == "pass" else "fail"
    clocks = gate_clocks({0: stats}, route, (0,))
    out["delivery"] = clocks["delivery"]["verdict"]
    return out


def drill_outcome(kind: str, intended: str, families: Mapping[str, str]) -> str:
    """``red`` when every intended family failed (a drill) or ``green`` when every family passed (a control,
    or a drill whose intended set is ``none``); anything else is a problem, named."""
    wanted = [] if intended == "none" else intended.split(",")
    if kind == "control" or not wanted:
        return "green" if all(v == "pass" for v in families.values()) else "unexpected red"
    return "red" if all(families[f] == "fail" for f in wanted) else "not red"
