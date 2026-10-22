"""The stock muonic cascade of Geant4 (``G4EmCaptureCascade``), exactly, and the tests that hold it to the
compiled code and to measurement, with the standard library alone.

``G4EmCaptureCascade::ApplyYourself`` (vendored under ``third_party/geant4/<tag>/``) captures the muon on the
14th level, emits that level's binding energy as an electron and walks the muon down to the K shell with two
branching rules. With Z the atomic number and U a uniform draw: while fewer than Z electrons have been
emitted (the counter starts at 1 for the capture electron), an Auger electron takes the muon one level
down when (Z^4 + 10000) U < 10000; otherwise a photon takes it from level index L to L - 1 when
(10 + L - 1) U <= 10 and to each lower level with equal weight 1 otherwise. Which transition the cascade
takes depends on Z alone; the mass number enters only the level energies.

"Exact" here means the expectation of those coded rules with ``G4UniformRand`` taken as a continuous uniform
draw on (0, 1): ``transitions(z)`` propagates the probability of every state (level, Auger count) forward in
``fractions.Fraction`` and returns the expected number of each transition per stopped muon. The domain is
1 <= z <= 215, over which the code's 32-bit integer product ``Z*Z*Z*Z`` does not overflow.

The module also holds what tests the model against the compiled code and against measurement:

- ``bind_source``: the rule constants read back from the whitespace-free rule statements of the vendored
  source;
- ``reconstruct``: the cascade of each stop rebuilt from the records of the stopped-muon application
  (``cpp/stopped_muon/``), energy for energy, with the Auger cap the source applies;
- ``decide``: an exact binomial test of each retained count with certified tails (outward-rounded dyadic
  interval arithmetic); ``retained_counts`` names the counts;
- the comparison rule for measured K-series intensity ratios (``comparison_rows``), with the policy passed in.

It reads no path of its own and imports nothing from ``openmucf``, so it can be loaded by path where the
package is not installed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Any

#: The coded rules as data; ``bind_source`` reads each back from the vendored source.
RULES: dict[str, int] = {
    "start_level": 13,  # nLevel at capture: level index 13, n = 14
    "auger_count0": 1,  # nAuger before the first cascade Auger (the capture electron counts)
    "auger_weight": 10000,  # Auger iff (Z^4 + w) U < w
    "photon_weight": 10,  # photon weight of the step to the next level down
    "power": 4,  # pGamma = Z*Z*Z*Z
}
Z_MAX = 215  # 215**4 <= 2**31 - 1 < 216**4
LEVELS = RULES["start_level"] + 1
UNIONS = (6, 7, 8)
KINDS = ("photon", "auger")

Transition = tuple[int, int, str]  # (upper n, lower n, kind), n the principal quantum number


# ------------------------------------------------------------------------------------------------ the model
def _check_z(z: int) -> None:
    if isinstance(z, bool) or not isinstance(z, int):
        raise TypeError(f"z must be an int, not {type(z).__name__}")
    if not 1 <= z <= Z_MAX:
        raise ValueError(
            f"z = {z} is outside 1..{Z_MAX}: the source forms Z*Z*Z*Z in a 32-bit G4int, which overflows "
            f"for Z > {Z_MAX}, and Z >= 1 is a nucleus"
        )


def _rules_key(rules: Mapping[str, int] | None) -> tuple[tuple[str, int], ...]:
    r = dict(RULES if rules is None else rules)
    if set(r) != set(RULES):
        raise ValueError(f"rules must name exactly {sorted(RULES)}")
    return tuple(sorted(r.items()))


@lru_cache(maxsize=512)
def _transitions(z: int, key: tuple[tuple[str, int], ...]) -> tuple[tuple[Transition, Fraction], ...]:
    r = dict(key)
    top, w, base = r["start_level"], r["auger_weight"], r["photon_weight"]
    a = Fraction(w, z ** r["power"] + w)
    pop: dict[tuple[int, int], Fraction] = {(top, r["auger_count0"]): Fraction(1)}
    flow: dict[Transition, Fraction] = {}
    for lev in range(top, 0, -1):
        for n_aug in sorted(k[1] for k in list(pop) if k[0] == lev):
            p = pop.pop((lev, n_aug))
            pa = a if n_aug < z else Fraction(0)
            if pa:
                q = p * pa
                pop[(lev - 1, n_aug + 1)] = pop.get((lev - 1, n_aug + 1), Fraction(0)) + q
                t = (lev + 1, lev, "auger")
                flow[t] = flow.get(t, Fraction(0)) + q
            pg = p * (1 - pa)
            tot = base + lev - 1
            for low in range(lev - 1, -1, -1):
                q = pg * Fraction(base if low == lev - 1 else 1, tot)
                pop[(low, n_aug)] = pop.get((low, n_aug), Fraction(0)) + q
                t = (lev + 1, low + 1, "photon")
                flow[t] = flow.get(t, Fraction(0)) + q
    return tuple(sorted(flow.items()))


def transitions(z: int, rules: Mapping[str, int] | None = None) -> dict[Transition, Fraction]:
    """The expected number of each transition per stopped muon, keyed (upper n, lower n, kind); every
    transition the rules allow is present (a transition the Auger cap forbids at this z has value 0)."""
    _check_z(z)
    flow = dict(_transitions(z, _rules_key(rules)))
    top = dict(_rules_key(rules))["start_level"] + 1
    for u in range(2, top + 1):
        flow.setdefault((u, u - 1, "auger"), Fraction(0))
        for low in range(1, u):
            flow.setdefault((u, low, "photon"), Fraction(0))
    return flow


def k_line(z: int, n: int, rules: Mapping[str, int] | None = None) -> Fraction:
    """Photons n -> 1 per stopped muon (n = 2..14)."""
    if not 2 <= n <= LEVELS:
        raise ValueError(f"n = {n} is outside 2..{LEVELS}")
    return transitions(z, rules)[(n, 1, "photon")]


def auger_into_1s(z: int, rules: Mapping[str, int] | None = None) -> Fraction:
    return transitions(z, rules)[(2, 1, "auger")]


def k_total(z: int, rules: Mapping[str, int] | None = None) -> Fraction:
    return sum((k_line(z, n, rules) for n in range(2, LEVELS + 1)), Fraction(0))


def kb_over_ka(z: int, rules: Mapping[str, int] | None = None) -> Fraction:
    return k_line(z, 3, rules) / k_line(z, 2, rules)


def high_over_ka(z: int, k: int, rules: Mapping[str, int] | None = None) -> Fraction:
    """(sum over n >= k of the photons n -> 1) / K alpha."""
    if not 3 <= k <= LEVELS:
        raise ValueError(f"k = {k} is outside 3..{LEVELS}")
    return sum((k_line(z, n, rules) for n in range(k, LEVELS + 1)), Fraction(0)) / k_line(z, 2, rules)


def ka_over_ktot(z: int, rules: Mapping[str, int] | None = None) -> Fraction:
    return k_line(z, 2, rules) / k_total(z, rules)


def key(t: Transition) -> str:
    """The name of a transition in the harvest files: ``g<upper>-><lower>`` (photon) or ``e...`` (Auger)."""
    return f"{'g' if t[2] == 'photon' else 'e'}{t[0]}->{t[1]}"


def union_key(k: int) -> str:
    return f"K>={k}"


def transition_keys(top: int = LEVELS) -> list[str]:
    """The 104 transitions the rules allow, photons by (upper, lower) then Augers by upper."""
    out = [key((u, low, "photon")) for u in range(2, top + 1) for low in range(1, u)]
    return out + [key((u, u - 1, "auger")) for u in range(2, top + 1)]


def retained_counts(flows: Mapping[Transition, Fraction]) -> dict[str, Fraction]:
    """The model probability of every retained count: each of the transitions (a stop takes each at most once,
    so the count of stops taking it is binomial) and the stops entering 1s by a photon from n >= k, k in
    ``UNIONS``."""
    out = {key(t): flows[t] for t in sorted(flows, key=lambda t: (KINDS.index(t[2]), t[0], t[1]))}
    for k in UNIONS:
        out[union_key(k)] = sum((flows[(n, 1, "photon")] for n in range(k, LEVELS + 1)), Fraction(0))
    return out


def observed_counts(transition_counts: Mapping[str, int]) -> dict[str, int]:
    """The retained counts of a harvested cell from its per-transition counts (the unions are sums)."""
    out = {k: int(transition_counts[k]) for k in transition_keys()}
    for k in UNIONS:
        out[union_key(k)] = sum(out[f"g{n}->1"] for n in range(k, LEVELS + 1))
    return out


# --------------------------------------------------------------------------------------------- source binding
#: The rule statements, whitespace removed, each matched by exactly one line of the rule block, in this order.
STATEMENTS: tuple[tuple[str, str], ...] = (
    ("n_elec", r"G4intnElec=Z;"),
    ("auger_count0", r"G4intnAuger=(\d+);"),
    ("start_level", r"G4intnLevel=(\d+);"),
    ("power", r"G4doublepGamma=\((Z(?:\*Z)*)\);"),
    ("capture_electron", r"G4doubleedep=fLevelEnergy\[(\d+)\];"),
    ("auger_test", r"if\(\(nAuger<nElec\)&&\(\(pGamma\+(\d+\.\d*)\)\*G4UniformRand\(\)<(\d+\.\d*)\)\)\{?"),
    ("auger_count", r"\+\+nAuger;"),
    ("auger_energy", r"deltaE=fLevelEnergy\[nLevel-1\]-fLevelEnergy\[nLevel\];"),
    ("auger_step", r"--nLevel;"),
    ("photon_draw", r"G4doublevar=\((\d+\.\d*)\+G4double\(nLevel-1\)\)\*G4UniformRand\(\);"),
    ("photon_next", r"G4intiLevel=nLevel-1;"),
    ("photon_jump", r"if\(var>(\d+\.\d*)\)iLevel-=G4int\(var-(\d+\.\d*)\)\+1;"),
    ("photon_clamp", r"if\(iLevel<0\)iLevel=0;"),
    ("photon_energy", r"deltaE=fLevelEnergy\[iLevel\]-fLevelEnergy\[nLevel\];"),
    ("photon_step", r"nLevel=iLevel;"),
    ("loop", r"\}while\(nLevel>0\);"),
)
MASS_IDENTIFIERS = ("A", "massA")


class SourceMismatch(Exception):
    """The vendored source does not state the rules this module computes."""


@dataclass(frozen=True)
class Binding:
    #: line number (from 1) of each statement in the source that was bound
    lines: dict[str, int]
    #: the rule constants read from the statements
    rules: dict[str, int]


def _bare(line: str) -> str:
    return "".join(line.split("//", 1)[0].split())


def read_rules(text: str) -> Binding:
    """The rule constants and their line numbers from one copy of ``G4EmCaptureCascade.cc``."""
    lines = text.splitlines()
    bare = [_bare(s) for s in lines]
    found: dict[str, int] = {}
    groups: dict[str, tuple[str, ...]] = {}
    start = 0
    for name, pattern in STATEMENTS:
        hits = [i for i, s in enumerate(bare) if re.fullmatch(pattern, s)]
        if len(hits) != 1:
            raise SourceMismatch(f"statement {name} matches {len(hits)} lines, not one")
        if hits[0] < start:
            raise SourceMismatch(f"statement {name} is out of order")
        start = hits[0]
        found[name] = hits[0] + 1
        groups[name] = re.fullmatch(pattern, bare[hits[0]]).groups()  # type: ignore[union-attr]
    block = "\n".join(s.split("//", 1)[0] for s in lines[found["n_elec"] - 1 : found["loop"]])
    for ident in MASS_IDENTIFIERS:
        if re.search(rf"\b{ident}\b", block):
            raise SourceMismatch(f"the rule block names the mass-number identifier {ident}")

    def number(s: str) -> int:
        f = Fraction(s)
        if f.denominator != 1:
            raise SourceMismatch(f"constant {s} is not an integer")
        return int(f)

    w1, w2 = groups["auger_test"]
    b1, b2, b3 = groups["photon_draw"][0], *groups["photon_jump"]
    if number(w1) != number(w2):
        raise SourceMismatch(f"the Auger test compares with {w2} after adding {w1}")
    if not number(b1) == number(b2) == number(b3):
        raise SourceMismatch(f"the photon draw uses {b1}, the jump {b2} and {b3}")
    rules = {
        "start_level": int(groups["start_level"][0]),
        "auger_count0": int(groups["auger_count0"][0]),
        "auger_weight": number(w1),
        "photon_weight": number(b1),
        "power": groups["power"][0].count("Z"),
    }
    if int(groups["capture_electron"][0]) != rules["start_level"]:
        raise SourceMismatch("the capture electron is not emitted at the start level")
    return Binding(found, rules)


def bind_source(reference: str, *others: str) -> Binding:
    """Bind the module to the vendored sources: the rules of ``reference`` (whose line numbers are returned)
    must equal ``RULES``, and every other copy must state the same statements once whitespace (and a block's
    opening brace, which a reformatted copy puts on its own line) is removed."""
    ref = read_rules(reference)
    if ref.rules != RULES:
        raise SourceMismatch(f"the source states {ref.rules}, the module computes {RULES}")
    ref_bare = [_bare(s) for s in reference.splitlines()]
    for i, other in enumerate(others):
        b = read_rules(other)
        obare = [_bare(s) for s in other.splitlines()]
        for name in ref.lines:
            if ref_bare[ref.lines[name] - 1].rstrip("{") != obare[b.lines[name] - 1].rstrip("{"):
                raise SourceMismatch(f"copy {i + 1}: statement {name} differs once whitespace is removed")
    return ref  # equal statements state equal rules: the constants are read from them


# --------------------------------------------------------------------------------------------- reconstruction
#: The muon's at-rest process on each route, and how its cascade secondaries are recognised.
CAPTURE_PROCESS = {"bound_decay": "muMinusCaptureAtRest", "muonic_atom_helper": "muMinusAtomicCaptureAtRest"}
CASCADE_MODEL_SUFFIX = "_EMCascade"  # bound_decay: the creator model of the cascade secondaries
HELPER_MODEL = "-"  # muonic_atom_helper: G4MuonMinusAtomicCapture's secondaries carry no creator model
ELECTRON, PHOTON = 11, 22
NUCLEUS_PDG_MIN = 1000000000


def levels_from_config(text: str) -> list[float]:
    """The 14 level energies (MeV) of the ``L`` line the application writes to ``config.txt``."""
    rows = [line.split()[1:] for line in text.splitlines() if line.startswith("L ")]
    if len(rows) != 1 or len(rows[0]) != LEVELS:
        raise ValueError(f"config.txt must hold one L line of {LEVELS} values")
    return [float.fromhex(x) for x in rows[0]]


@dataclass
class Harvest:
    """What ``reconstruct`` read from one record stream."""

    events: int = 0
    stops: int = 0
    chains: int = 0
    refused: Counter[str] = field(default_factory=Counter)
    transitions: Counter[str] = field(default_factory=Counter)
    augers: int = 0
    photons: int = 0
    seeds: list[tuple[int, int]] = field(default_factory=list)
    _hash: Any = field(default_factory=hashlib.sha256)

    @property
    def sequences_sha256(self) -> str:
        """sha256 of every reconstructed stop, one line ``<event> <transition> ...`` each, in event order."""
        return str(self._hash.hexdigest())


def chain(secondaries: Sequence[tuple[int, float]], levels: Sequence[float], z: int) -> list[str]:
    """The transitions of one stop from its cascade secondaries (pdg, kinetic energy in MeV) in emission
    order.

    Raises ``ValueError`` naming the reason when the secondaries are not a cascade the rules allow: the first
    must be the capture electron (energy ``levels[13]``); each later electron an Auger from the current level
    index n (energy ``levels[n-1] - levels[n]``), admitted only while the Auger counter, which starts at 1, is
    below z; each photon the difference ``levels[i] - levels[n]`` for exactly one i < n; the chain must end at
    the K shell."""
    _check_z(z)
    top = RULES["start_level"]
    if not secondaries:
        raise ValueError("no cascade")
    pdg0, e0 = secondaries[0]
    if pdg0 != ELECTRON or e0 != levels[top]:
        raise ValueError("no capture electron")
    n, n_aug, out = top, RULES["auger_count0"], []
    for pdg, e in secondaries[1:]:
        if n == 0:
            raise ValueError("secondary after the K shell")
        if pdg == ELECTRON:
            if e != levels[n - 1] - levels[n]:
                raise ValueError("electron energy is no Auger energy")
            if not n_aug < z:
                raise ValueError("over the Auger cap")
            n_aug += 1
            out.append(key((n + 1, n, "auger")))
            n -= 1
        elif pdg == PHOTON:
            hits = [i for i in range(n) if levels[i] - levels[n] == e]
            if len(hits) != 1:
                raise ValueError("photon energy matches no single transition")
            out.append(key((n + 1, hits[0] + 1, "photon")))
            n = hits[0]
        else:
            raise ValueError(f"cascade secondary of PDG {pdg}")
    if n != 0:
        raise ValueError("chain does not reach the K shell")
    return out


def reconstruct(stream: Iterable[str], levels: Sequence[float], route: str, z: int) -> Harvest:
    """The cascade of every stop of a record stream of the stopped-muon application (one run, one thread; the
    record format of ``openmucf/g4/stopped_mu.py``). A stop is the muon's (track 1) at-rest step on ``route``;
    its cascade is, on ``bound_decay``, the secondaries of that step whose creator model ends ``_EMCascade``
    and, on ``muonic_atom_helper``, its electron and photon secondaries (which carry no creator model) beside
    the one muonic atom. Each event must hold exactly one stop. A stop whose cascade ``chain`` refuses, or an
    event without one stop, is counted in ``refused`` by reason and never in the transition counts."""
    if route not in CAPTURE_PROCESS:
        raise ValueError(f"route must be one of {sorted(CAPTURE_PROCESS)}")
    _check_z(z)
    if len(levels) != LEVELS:
        raise ValueError(f"{LEVELS} levels expected")
    h = Harvest()
    process = CAPTURE_PROCESS[route]
    steps: list[list[tuple[int, str, float]]] = []
    current: list[tuple[int, str, float]] | None = None
    declared = 0
    expect = 0
    open_event = False  # a record line read since the last E line

    def close() -> None:
        if current is not None and len(current) != declared:
            raise ValueError(f"S-line declares {declared} secondaries, {len(current)} C-lines follow it")

    for line in stream:
        f = line.split()
        if not f:
            raise ValueError("empty record line")
        kind = f[0]
        open_event = kind != "E"
        if kind == "C":
            if current is None:
                raise ValueError("C-line not after an at-rest S-line of the muon")
            if int(f[3]) != len(current):
                raise ValueError(f"C-line numbered {f[3]} where {len(current)} is next")
            current.append((int(f[4]), f[5], float.fromhex(f[6])))
            continue
        close()
        current = None
        if kind == "S":
            if f[2] == "1" and f[4] == process:
                current = []
                declared = int(f[7])
                steps.append(current)
            elif int(f[7]) != 0:
                current = []  # secondaries of another step: read and checked, never part of a cascade
                declared = int(f[7])
        elif kind == "E":
            ev = int(f[1])
            if ev != expect:
                raise ValueError(f"event {ev} where {expect} expected")
            expect += 1
            h.events += 1
            h.seeds.append((ev, int(f[2])))
            _stop(h, ev, steps, levels, route, z)
            steps = []
        elif kind != "T":
            raise ValueError(f"unknown record line {line[:40]!r}")
    close()
    if open_event:
        raise ValueError("record lines after the last E line")
    return h


def _stop(
    h: Harvest,
    ev: int,
    steps: list[list[tuple[int, str, float]]],
    levels: Sequence[float],
    route: str,
    z: int,
) -> None:
    if len(steps) != 1:
        h.refused[f"at-rest steps of the muon: {len(steps)}"] += 1
        return
    h.stops += 1
    secs = steps[0]
    if route == "bound_decay":
        cascade = [(p, e) for p, m, e in secs if m.endswith(CASCADE_MODEL_SUFFIX)]
    else:
        atoms = [p for p, m, e in secs if p >= NUCLEUS_PDG_MIN]
        rest = [(p, m) for p, m, e in secs if p < NUCLEUS_PDG_MIN]
        if len(atoms) != 1 or any(p not in (ELECTRON, PHOTON) or m != HELPER_MODEL for p, m in rest):
            h.refused["at-rest step is not one muonic atom and its cascade"] += 1
            return
        cascade = [(p, e) for p, m, e in secs if p < NUCLEUS_PDG_MIN]
    try:
        seq = chain(cascade, levels, z)
    except ValueError as e:
        h.refused[str(e)] += 1
        return
    h.chains += 1
    for t in seq:
        h.transitions[t] += 1
        if t[0] == "e":
            h.augers += 1
        else:
            h.photons += 1
    if len(set(seq)) != len(seq):
        raise AssertionError("a stop took one transition twice")
    h._hash.update(f"{ev} {' '.join(seq)}\n".encode("ascii"))


# ------------------------------------------------------------------------------- certified binomial tails
_P = 160  # mantissa bits of the dyadic bounds
Dyadic = tuple[int, int]  # m * 2**e


def _norm(m: int, e: int, up: bool) -> Dyadic:
    if m == 0:
        return (0, 0)
    extra = m.bit_length() - _P
    if extra > 0:
        q, r = divmod(m, 1 << extra)
        if up and r:
            q += 1
        return (q, e + extra)
    return (m, e)


def _from_fraction(x: Fraction, up: bool) -> Dyadic:
    n, d = x.numerator, x.denominator
    if n < 0:
        raise ValueError("non-negative values only")
    if n == 0:
        return (0, 0)
    k = _P + d.bit_length() - n.bit_length() + 1
    q, r = divmod(n << k, d) if k >= 0 else divmod(n, d << (-k))
    if up and r:
        q += 1
    return _norm(q, -k, up)


def _mul(a: Dyadic, b: Dyadic, up: bool) -> Dyadic:
    return _norm(a[0] * b[0], a[1] + b[1], up)


def _add(a: Dyadic, b: Dyadic, up: bool) -> Dyadic:
    if a[0] == 0:
        return b
    if b[0] == 0:
        return a
    e = min(a[1], b[1])
    return _norm((a[0] << (a[1] - e)) + (b[0] << (b[1] - e)), e, up)


def _div(a: Dyadic, b: Dyadic, up: bool) -> Dyadic:
    if a[0] == 0:
        return (0, 0)
    shift = _P + b[0].bit_length() - a[0].bit_length() + 1
    q, r = divmod(a[0] << shift, b[0])
    if up and r:
        q += 1
    return _norm(q, a[1] - b[1] - shift, up)


def _frac(a: Dyadic) -> Fraction:
    m, e = a
    return Fraction(m << e) if e >= 0 else Fraction(m, 1 << (-e))


@dataclass(frozen=True)
class Interval:
    """[lo, hi] with dyadic ends, every operation rounding lo down and hi up (non-negative values only)."""

    lo: Dyadic
    hi: Dyadic

    @staticmethod
    def of(x: Fraction) -> Interval:
        return Interval(_from_fraction(x, False), _from_fraction(x, True))

    def __mul__(self, o: Interval) -> Interval:
        return Interval(_mul(self.lo, o.lo, False), _mul(self.hi, o.hi, True))

    def __add__(self, o: Interval) -> Interval:
        return Interval(_add(self.lo, o.lo, False), _add(self.hi, o.hi, True))

    def __truediv__(self, o: Interval) -> Interval:  # o.lo > 0
        return Interval(_div(self.lo, o.hi, False), _div(self.hi, o.lo, True))

    def pow(self, k: int) -> Interval:
        res, base = Interval((1, 0), (1, 0)), self
        while k:
            if k & 1:
                res = res * base
            base = base * base
            k >>= 1
        return res

    def bounds(self) -> tuple[Fraction, Fraction]:
        return _frac(self.lo), _frac(self.hi)


_FACT: list[Interval] = [Interval((1, 0), (1, 0))]


def _fact(k: int) -> Interval:
    while len(_FACT) <= k:
        j = len(_FACT)
        _FACT.append(_FACT[-1] * Interval((j, 0), (j, 0)))
    return _FACT[k]


def _pmf(n: int, p: Fraction, x: int) -> Interval:
    coef = _fact(n) / (_fact(x) * _fact(n - x))
    return coef * Interval.of(p).pow(x) * Interval.of(1 - p).pow(n - x)


def _check_binomial(n: int, p: Fraction, x: int, side: str) -> None:
    if not 0 < p < 1:
        raise ValueError("0 < p < 1 required")
    if not 0 <= x <= n:
        raise ValueError("0 <= x <= n required")
    if side not in ("hi", "lo"):
        raise ValueError("side is 'hi' (P(X >= x)) or 'lo' (P(X <= x))")


def _walk(
    n: int, p: Fraction, x: int, side: str, accept_above: Fraction | None, tol_bits: int = 120
) -> tuple[Fraction, Fraction]:
    """Certified [lo, hi] of the tail from x away from the mode. With ``accept_above`` the walk stops as soon
    as the lower bound of the partial sum exceeds it (hi is then returned as 1, unused) or the upper bound of
    the whole tail is at most it."""
    term = _pmf(n, p, x)
    acc = term
    k = x
    ratio = Interval.of(p / (1 - p)) if side == "hi" else Interval.of((1 - p) / p)
    while True:
        lo = _frac(acc.lo)
        if accept_above is not None and lo > accept_above:
            return lo, Fraction(1)
        if (side == "hi" and k >= n) or (side == "lo" and k <= 0):
            return acc.bounds()
        if side == "hi":
            r = Interval.of(Fraction(n - k, k + 1)) * ratio
            k += 1
        else:
            r = Interval.of(Fraction(k, n - k + 1)) * ratio
            k -= 1
        term = term * r
        acc = acc + term
        # past the mode the ratios fall monotonically in the direction of travel, so once the ratio's upper
        # bound r is < 1 every remaining term is <= term r^j and the rest of the tail is <= term r / (1 - r)
        r_hi = _frac(r.hi)
        if r_hi < 1:
            rest = _frac(term.hi) * r_hi / (1 - r_hi)
            lo, hi = _frac(acc.lo), _frac(acc.hi) + rest
            if accept_above is not None and hi <= accept_above:
                return lo, hi
            if accept_above is None and (
                (lo > 0 and rest * (1 << tol_bits) < lo) or rest < Fraction(1, 1 << 600)
            ):
                return lo, hi


def binomial_tail(n: int, p: Fraction, x: int, side: str) -> tuple[Fraction, Fraction]:
    """Certified bounds [lo, hi] of P(X >= x) (``side`` 'hi') or P(X <= x) ('lo'), X ~ Binomial(n, p), for
    an x on that side of the mode (or at it): lo <= tail <= hi. The walk stops at the end of the support or
    once the remainder bound is below 2^-120 of lo or below 2^-600, so for n < 2^30 the width hi - lo is
    below 2^-119 lo + 2^-599."""
    _check_binomial(n, p, x, side)
    mode = math.floor((n + 1) * p)
    if (side == "hi" and x < mode) or (side == "lo" and x > mode):
        raise ValueError("x must lie on the tested side of the mode")
    return _walk(n, p, x, side, None)


@dataclass(frozen=True)
class Decision:
    side: str  # hi (P(X >= x) tested), lo (P(X <= x) tested) or mode (accepted without a tail)
    bound_kind: str  # lower (a pass decided by a tail), upper (a reject) or none (side mode)
    bound: Fraction | None
    decision: str  # pass or reject


def decide(n: int, p: Fraction, x: int, alpha_side: Fraction) -> Decision:
    """The exact binomial test of one count: take the tail on x's side of the mode floor((n + 1) p) and
    reject iff a certified upper bound of it is <= ``alpha_side``; a pass is decided by a certified lower
    bound of the tail above ``alpha_side`` (usually the first term); x at the mode passes without a tail."""
    if not 0 < alpha_side < 1:
        raise ValueError("0 < alpha_side < 1 required")
    _check_binomial(n, p, x, "hi")
    mode = math.floor((n + 1) * p)
    if x == mode:
        return Decision("mode", "none", None, "pass")
    side = "hi" if x > mode else "lo"
    lo, hi = _walk(n, p, x, side, alpha_side)
    if lo > alpha_side:
        return Decision(side, "lower", lo, "pass")
    if hi <= alpha_side:
        return Decision(side, "upper", hi, "reject")
    # the enclosure straddles the budget: the tail is within the bounds' width of alpha_side and no
    # certified decision exists at this precision (for example n = 1, p = alpha_side, x = 1, whose tail is
    # exactly alpha_side)
    raise ArithmeticError("the certified tail does not separate from the budget")


def alpha_side(policy: Mapping[str, Any], counts: int) -> Fraction:
    """The per-side budget: the per-cell budget split over the retained counts and the two sides."""
    return Fraction(policy["budget_per_cell"]) / (counts * int(policy["sides"]))


# ------------------------------------------------------------------------------- the decimal display helpers
class DisplayDomainError(TypeError, ValueError):
    """The one refusal of the decimal display helpers: an argument outside the helper's stated domain. It is
    both a TypeError and a ValueError, so a caller may catch it as either."""


#: The bound every display helper states on its number of places (as abs(places)) and of significant digits.
PRECISION_MAX = 10**5

#: The kinds of argument the display helpers admit: the exact types, the range, and the words of the refusal.
_KINDS: dict[str, tuple[tuple[type, ...], Callable[[Any], bool], str]] = {
    "rational": ((int, Fraction), lambda v: True, "an int or a Fraction"),
    "rational >= 0": ((int, Fraction), lambda v: v >= 0, "an int or a Fraction >= 0"),
    "rational > 0": ((int, Fraction), lambda v: v > 0, "an int or a Fraction > 0"),
    "int": ((int,), lambda v: True, "an int"),
    "int >= 0": ((int,), lambda v: v >= 0, "an int >= 0"),
    "places": (
        (int,),
        lambda v: -PRECISION_MAX <= v <= PRECISION_MAX,
        f"an int in [-{PRECISION_MAX}, {PRECISION_MAX}]",
    ),
    "digits": ((int,), lambda v: 1 <= v <= PRECISION_MAX, f"an int in [1, {PRECISION_MAX}]"),
    "flag": ((bool,), lambda v: True, "a bool"),
}


def _admit(**args: tuple[object, str]) -> None:
    """The one entry check of the display helpers, called first, before any arithmetic. Each argument is
    given as name=(value, kind). Its type must be exactly one of the kind's types, so a bool is not an int
    and subclasses, floats (NaN and the infinities among them), Decimals and strings are refused, and its
    value must lie in the kind's range. Anything else raises DisplayDomainError."""
    for name, (value, kind) in args.items():
        types, admits, words = _KINDS[kind]
        if type(value) not in types:
            raise DisplayDomainError(f"{name} must be {words}, not {type(value).__name__}")
        if not admits(value):
            raise DisplayDomainError(f"{name} must be {words}")


#: log10(2) rounded down to 15 decimals, as an exact rational (the integer exponent estimate below).
_LOG10_2 = Fraction(301029995663981, 10**15)
#: Every decimal string is assembled from chunks of this many digits: str() of an int of at most 500 digits is
#: allowed under every setting of the interpreter's int-to-string limit (whose smallest allowed value is 640).
_CHUNK = 500


def _digits(n: int) -> str:
    """The decimal digits of n, without str() of an int above _CHUNK digits.

    Domain: n an int >= 0 of any size. Anything else raises DisplayDomainError."""
    _admit(n=(n, "int >= 0"))
    base = 10**_CHUNK
    if n < base:
        return str(n)
    parts = []
    while n:
        n, r = divmod(n, base)
        parts.append(r)
    return str(parts[-1]) + "".join(str(c).zfill(_CHUNK) for c in reversed(parts[:-1]))


def _exponent(x: Fraction) -> int:
    """The integer e with 10^e <= x < 10^(e + 1), for any rational x > 0, in integer arithmetic.

    With b = bit_length(numerator) - bit_length(denominator), 2^(b - 1) < x < 2^(b + 1). _LOG10_2 differs from
    log10(2) by less than 10^-15, so for |b| < 10^15 (every rational that fits in memory)
    t = (b - 1) _LOG10_2 differs from (b - 1) log10(2) by less than 1, and e0 = floor(t) - 2 satisfies
    e0 + 1 <= t - 1 < (b - 1) log10(2), so 10^(e0 + 1) < 2^(b - 1) < x: e0 is below the exponent. The exponent
    is below (b + 1) log10(2) < e0 + 4, so at most four exact comparisons reach it."""
    b = x.numerator.bit_length() - x.denominator.bit_length()
    e = math.floor((b - 1) * _LOG10_2) - 2
    while Fraction(10) ** (e + 1) <= x:
        e += 1
    return e


def sig_digits(x: Fraction, digits: int, up: bool) -> str:
    """``x`` to ``digits`` significant digits, rounded up (``up``) or down, in scientific notation, exactly.

    Domain: x an int or a Fraction > 0 of any size; digits an int in [1, PRECISION_MAX]; up a bool. Anything
    else raises DisplayDomainError."""
    _admit(x=(x, "rational > 0"), digits=(digits, "digits"), up=(up, "flag"))
    e = _exponent(x)
    m = x / Fraction(10) ** (e - digits + 1)  # 10^(digits - 1) <= m < 10^digits
    q = math.ceil(m) if up else math.floor(m)
    if q == 10**digits:  # rounding up carried into a new digit
        q, e = q // 10, e + 1
    s = _digits(q)
    return f"{s[0]}.{s[1:]}e{e:+03d}" if digits > 1 else f"{s}e{e:+03d}"


def _half_even(q: Fraction) -> int:
    """The integer nearest to q, ties to even."""
    f = math.floor(q)
    r = q - f
    return f + 1 if r > Fraction(1, 2) or (r == Fraction(1, 2) and f % 2) else f


def _sqrt_half_even(y: Fraction) -> int:
    """The integer nearest to sqrt(y), y >= 0, ties to even, in exact integer arithmetic: floor(sqrt(y)) is
    isqrt(floor(y)), and sqrt(y) lies above f + 1/2 exactly when y > (f + 1/2)^2."""
    f = math.isqrt(math.floor(y))
    half = (Fraction(2 * f + 1, 2)) ** 2
    return f + 1 if y > half or (y == half and f % 2) else f


def _decimal(n: int, places: int, negative: bool = False) -> str:
    """Format the magnitude abs(n) * 10^-places as plain decimal text, prefixing '-' when ``negative``.
    Domain: n and places any ints. Positive places give exactly that many fractional digits; nonpositive
    places give an integer string."""
    _admit(n=(n, "int"), places=(places, "int"), negative=(negative, "flag"))
    if places <= 0:
        body = _digits(abs(n) * 10 ** (-places))
    else:
        digits = _digits(abs(n)).rjust(places + 1, "0")
        body = f"{digits[:-places]}.{digits[-places:]}"
    return ("-" if negative else "") + body


def fixed(x: Fraction, places: int) -> str:
    """``x`` rounded half-even to a multiple of 10^-places, exactly.

    Domain: x an int or a Fraction >= 0 of any size; places an int in [-PRECISION_MAX, PRECISION_MAX] (a
    negative ``places`` rounds to tens, hundreds, ...). Anything else raises DisplayDomainError."""
    _admit(x=(x, "rational >= 0"), places=(places, "places"))
    return _decimal(_half_even(x * Fraction(10) ** places), places)


CONSUMER_COLUMNS = (
    "version",
    "config",
    "target",
    "count",
    "observed",
    "expected",
    "side",
    "budget",
    "bound_kind",
    "bound_value",
    "decision",
)


def consumer_rows(
    cell: Mapping[str, str],
    observed: Mapping[str, int],
    z: int,
    n: int,
    policy: Mapping[str, Any],
    rules: Mapping[str, int] | None = None,
) -> list[dict[str, str]]:
    """One row per retained count of a harvested cell, in the order of ``retained_counts``, each decided by
    ``decide`` against the model of ``rules`` (default the stock rules). ``n`` and every observed count must
    be ints >= 0 (otherwise DisplayDomainError), so the displayed counts are the counts tested."""
    _admit(n=(n, "int >= 0"), **{f"observed[{k}]": (v, "int >= 0") for k, v in observed.items()})
    probs = retained_counts(transitions(z, rules))
    if set(observed) != set(probs):
        raise ValueError("the observed counts are not the retained counts")
    a = alpha_side(policy, len(probs))
    rows = []
    for name, p in probs.items():
        d = decide(n, p, observed[name], a)
        value = "" if d.bound is None else sig_digits(d.bound, 6, up=d.bound_kind == "upper")
        rows.append(
            {
                "version": cell["version"],
                "config": cell["config"],
                "target": cell["target"],
                "count": name,
                "observed": str(observed[name]),
                "expected": fixed(n * p, 6),
                "side": d.side,
                "budget": f"{a.numerator}/{a.denominator}",
                "bound_kind": d.bound_kind,
                "bound_value": value,
                "decision": d.decision,
            }
        )
    return rows


# ------------------------------------------------------------------------------------- the comparison rule
OBSERVABLES = ("kb_over_ka", "high_over_ka", "ka_over_ktot")
VERDICTS = ("HOLDS", "TENSION", "FAILS")
COMPARISON_COLUMNS = (
    "source",
    "target",
    "Z",
    "phase",
    "set",
    "observable",
    "k",
    "model",
    "measured",
    "sigma",
    "pull",
    "verdict",
    "sigma_independent",
    "pull_independent",
    "verdict_independent",
    "verdict_rho_minus1",
    "verdict_rho_plus1",
)


def verdict(model: Fraction, measured: Fraction, sigma2: Fraction, policy: Mapping[str, Any]) -> str:
    """HOLDS iff (m - d)^2 <= h^2 s^2, TENSION iff h^2 s^2 < (m - d)^2 <= t^2 s^2, else FAILS; decided on
    squares in exact arithmetic (h, t: the policy's holds_within and tension_within)."""
    if sigma2 < 0:
        raise ValueError("sigma^2 < 0")
    d2 = (model - measured) ** 2
    h, t = Fraction(policy["holds_within"]), Fraction(policy["tension_within"])
    if d2 <= h * h * sigma2:
        return "HOLDS"
    if d2 <= t * t * sigma2:
        return "TENSION"
    return "FAILS"


def pull(model: Fraction, measured: Fraction, sigma2: Fraction) -> str:
    """(m - d) / s to 2 decimals, half-even, signed, in exact arithmetic (s = sqrt(sigma2)); it decides
    nothing.

    Domain: model and measured ints or Fractions of any sign and size; sigma2 an int or a Fraction > 0.
    Anything else raises DisplayDomainError."""
    _admit(model=(model, "rational"), measured=(measured, "rational"), sigma2=(sigma2, "rational > 0"))
    d = model - measured
    n = _sqrt_half_even(d * d * 10**4 / sigma2)
    return ("-" if d < 0 else "+") + _decimal(n, 2)


def sigma_decimal(sigma2: Fraction, places: int = 9) -> str:
    """sqrt(sigma2) rounded half-even to a multiple of 10^-places, in exact arithmetic.

    Domain: sigma2 an int or a Fraction >= 0 of any size; places an int in [-PRECISION_MAX, PRECISION_MAX].
    Anything else raises DisplayDomainError."""
    _admit(sigma2=(sigma2, "rational >= 0"), places=(places, "places"))
    return _decimal(_sqrt_half_even(sigma2 * Fraction(10) ** (2 * places)), places)


def ratio_sigma2(
    a: Fraction, sa: Fraction, b: Fraction, sb: Fraction, rho: Fraction = Fraction(0)
) -> Fraction:
    """sigma^2 of r = b / a with correlation rho between the two errors: r^2 (ea^2 + eb^2 - 2 rho ea eb)."""
    r, ea, eb = b / a, sa / a, sb / b
    return r * r * (ea * ea + eb * eb - 2 * rho * ea * eb)


def knorm_sigma2(members: Sequence[tuple[Fraction, Fraction]]) -> tuple[Fraction, Fraction, Fraction]:
    """(R, sigma^2 by the convention, sigma^2 by independent propagation) of R = K alpha / sum of the members,
    members[0] being K alpha: the convention holds the sum fixed, sigma = sigma(K alpha) / sum; independent
    propagation gives ((S - a)^2 sa^2 + a^2 sum of the other members' sigma^2) / S^4."""
    (a, sa), s = members[0], sum((v for v, _ in members), Fraction(0))
    others = sum((e * e for _, e in members[1:]), Fraction(0))
    return a / s, sa * sa / (s * s), ((s - a) ** 2 * sa * sa + a * a * others) / s**4


def _members(
    rows: list[dict[str, str]],
) -> tuple[dict[int, tuple[Fraction, Fraction]], tuple[int, Fraction, Fraction] | None]:
    single = {int(r["n_upper"]): (Fraction(r["value"]), Fraction(r["sigma"])) for r in rows if r["n_upper"]}
    grouped = [(int(r["n_from"]), Fraction(r["value"]), Fraction(r["sigma"])) for r in rows if r["n_from"]]
    if len(grouped) > 1:
        raise ValueError("more than one grouped member")
    return single, (grouped[0] if grouped else None)


def comparison_rows(reference: list[dict[str, str]], policy: Mapping[str, Any]) -> list[dict[str, str]]:
    """The comparison of the stock cascade with every measured ratio of ``reference`` (the rows of
    ``cascade_reference.csv``) under ``policy`` (``cascade_rule.json``; its ``ratios`` part is read), one row
    per (source, target, observable, k): K beta / K alpha and (n >= k) / K alpha of every source, and
    K alpha / the sum of the printed members for a source of absolute members."""
    order: list[tuple[str, str]] = []
    by: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in reference:
        k = (r["source"], r["target"])
        if k not in by:
            order.append(k)
            by[k] = []
        by[k].append(r)
    ratios = policy["ratios"]
    out: list[dict[str, str]] = []
    for source, target in order:
        rows = by[(source, target)]
        z = int(rows[0]["Z"])
        if any(int(r["Z"]) != z or r["phase"] != rows[0]["phase"] for r in rows):
            raise ValueError(f"{source} {target}: rows disagree on Z or phase")
        single, grouped = _members(rows)
        kind = ratios["sources"][source]
        base = {
            "source": source,
            "target": target,
            "Z": str(z),
            "phase": rows[0]["phase"],
            "set": ratios["set_of_phase"][rows[0]["phase"]],
        }

        def emit(
            obs: str,
            k: str,
            model: Fraction,
            measured: Fraction,
            s2: Fraction,
            extra: dict[str, str],
            base: dict[str, str] = base,
        ) -> None:
            row = dict(
                base,
                observable=obs,
                k=k,
                model=fixed(model, 9),
                measured=fixed(measured, 9),
                sigma=sigma_decimal(s2),
                pull=pull(model, measured, s2),
                verdict=verdict(model, measured, s2, ratios),
            )
            row.update({c: "" for c in COMPARISON_COLUMNS if c not in row})
            row.update(extra)
            out.append(row)

        def scan(
            a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction], model: Fraction
        ) -> dict[str, str]:
            m = b[0] / a[0]
            return {
                "verdict_rho_minus1": verdict(
                    model, m, ratio_sigma2(a[0], a[1], b[0], b[1], Fraction(-1)), ratios
                ),
                "verdict_rho_plus1": verdict(
                    model, m, ratio_sigma2(a[0], a[1], b[0], b[1], Fraction(1)), ratios
                ),
            }

        if kind == "printed_ratio_percent":  # Orth: n->1 relative to 2->1, in percent
            if 3 in single:
                v, s = single[3]
                emit("kb_over_ka", "", kb_over_ka(z), v / 100, (s / 100) ** 2, {})
            if grouped is not None:
                kk, v, s = grouped
                emit("high_over_ka", str(kk), high_over_ka(z, kk), v / 100, (s / 100) ** 2, {})
        elif kind == "absolute_members":  # ENSDF, Kirch: members of the K series
            a = single[2]
            members = [single[n] for n in sorted(single)] + ([(grouped[1], grouped[2])] if grouped else [])
            rr, sig2_conv, sig2_ind = knorm_sigma2(members)
            model = ka_over_ktot(z)
            emit(
                "ka_over_ktot",
                "",
                model,
                rr,
                sig2_conv,
                {
                    "sigma_independent": sigma_decimal(sig2_ind),
                    "pull_independent": pull(model, rr, sig2_ind),
                    "verdict_independent": verdict(model, rr, sig2_ind, ratios),
                },
            )
            if 3 in single:
                b = single[3]
                model = kb_over_ka(z)
                emit(
                    "kb_over_ka",
                    "",
                    model,
                    b[0] / a[0],
                    ratio_sigma2(a[0], a[1], b[0], b[1]),
                    scan(a, b, model),
                )
            if grouped is not None:
                kk, v, s = grouped
                model = high_over_ka(z, kk)
                emit(
                    "high_over_ka",
                    str(kk),
                    model,
                    v / a[0],
                    ratio_sigma2(a[0], a[1], v, s),
                    scan(a, (v, s), model),
                )
        else:
            raise ValueError(f"unknown source kind {kind!r}")
    return out


def csv_text(columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def outward(x: Fraction, places: int, up: bool) -> str:
    """``x`` rounded up (``up``) or down to a multiple of 10^-places, exactly.

    Domain: x an int or a Fraction >= 0 of any size; places an int in [-PRECISION_MAX, PRECISION_MAX] (a
    negative ``places`` rounds to tens, hundreds, ...); up a bool. Anything else raises DisplayDomainError."""
    _admit(x=(x, "rational >= 0"), places=(places, "places"), up=(up, "flag"))
    scaled = x * Fraction(10) ** places
    return _decimal(math.ceil(scaled) if up else math.floor(scaled), places)
