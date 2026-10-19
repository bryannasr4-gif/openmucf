"""Classify, reduce and judge the records of the stopped-muon benchmark, with the standard library alone.

The benchmark stops a negative muon at the centre of a single-element box, lets Geant4 transport it with the
transport harness of ``cpp/stopped_muon/`` and scores the secondaries Geant4 creates at the capture step
against measured yields. Every window, PDG code, line energy, guard, the capture count and every constant of
the rule are read from the pre-registration, a JSON file the caller names (``load_policy(path)``); this module
reads no path of its own and imports nothing from ``openmucf``, so it runs under any Python 3.11+ and can be
loaded by path where the package is not installed.

Harness records (one run, one thread; KE in MeV, times in ns, momenta in MeV/c, every float a C99 hexadecimal
float)::

    T <event> <track> <parent> <pdg> <creator process> <creator model> <KE> <global time>
    S <event> <track> <step> <process> <post-step KE> <post-step time> <n secondaries>
    C <event> <track> <k> <pdg> <creator model> <KE> <global time> <px> <py> <pz>
    E <event> <seed> <tracks>

S lines are written for the muon and for muonic atoms only, each followed by the C lines of its secondaries;
the E line closes an event. Every integer field is written in canonical decimal (no sign on a positive value,
no leading zero) and every float as C99 ``%a`` output. The classifier refuses (``Malformed``) a stream that
departs from this: a field count or a field lexeme other than these, an S line not followed by exactly its
``<n secondaries>`` C lines of its own track numbered 0, 1, ..., or an E line whose ``<tracks>`` differs from
the number of T lines of its event.

Compact format (one line per event, the event number global: chunk offset + local event id), opened by the
header ``# <version> <config> <target> <offset> <events> <route>``::

    <ev> C <t_capture> <pdg>:<KE>:<px>:<py>:<pz> ... g22:<KE>:<dt>:<parent pdg> ...
    <ev> D
    <ev> X <reason>

A ``C`` line is a nuclear capture: its plain tokens are the capture-step products and its ``g`` tokens the
photons whose parent is a nucleus the capturing track created, emitted less than the descendant window after
the capture (``dt`` from ``t_capture``, the smallest product time). ``D`` is a decay in orbit. ``X`` is an
event that is neither one capture nor one decay; the reducer refuses a cell holding one, except ``X
no_outcome`` (the capture step returned neither products nor a decay), which the pre-registration's
``rule.no_outcome`` excludes from both denominators.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, TextIO

ROUTES = ("bound_decay", "muonic_atom_helper")
NO_OUTCOME_MODES = ("exclude", "as_capture")
COMPARISON_COLUMNS = (
    "row_id",
    "predicate",
    "family",
    "target",
    "version",
    "config",
    "n_capt",
    "x",
    "x_se",
    "ref",
    "ref_stat",
    "ref_syst",
    "rescue_lo",
    "rescue_hi",
)
VERDICT_COLUMNS = ("row_id", "verdict", "kind", "pull")


class Malformed(Exception):
    """A record stream that does not follow the harness's record format."""


class Refused(Exception):
    """A cell, a comparison or a policy that the benchmark refuses: never a verdict."""


Policy = dict[str, Any]


def load_policy(path: str | Path) -> Policy:
    """The parsed pre-registration at ``path``. Nothing here has a default location."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def route_of(policy: Policy, config: str) -> str:
    try:
        return str(policy["setup"]["configurations"][config]["route"])
    except KeyError:
        raise Refused(f"unknown configuration {config!r}") from None


# --------------------------------------------------------------------------------------------- classification
@dataclass(frozen=True)
class _Classes:
    capture_process: str
    product_suffix: str
    decay_suffix: str
    helper_decay_pdg: int
    photon_pdg: int
    nucleus_pdg_min: int
    window_ns: float


def _classes(policy: Policy) -> _Classes:
    c = policy["classification"]
    return _Classes(
        c["capture_process"],
        c["product_model_suffix"],
        c["decay_model_suffix"],
        int(c["helper_decay_pdg"]),
        int(c["photon_pdg"]),
        int(c["nucleus_pdg_min"]),
        float(c["descendant_photon_window_ns"]),
    )


#: The fields after the kind of each record line: i an integer in canonical decimal, h a C99 hexadecimal float
#: as ``%a`` writes it, w a word.
RECORD_FIELDS = {"T": "iiiiwwhh", "S": "iiiwhhi", "C": "iiiiwhhhhh", "E": "iii"}
_INT = re.compile(r"0|-?[1-9][0-9]*")
_HEX = re.compile(r"-?0x[0-9a-f]+(?:\.[0-9a-f]+)?p[+-][0-9]+")


def record_fields(line: str) -> list[str]:
    """The fields of one record line, its kind first, each checked against ``RECORD_FIELDS``."""
    p = line.split()
    layout = RECORD_FIELDS.get(p[0]) if p else None
    if layout is None:
        raise Malformed(f"unknown record line {line[:60]!r}")
    if len(p) != len(layout) + 1:
        if p[0] == "C" and len(p) == 8:
            raise Malformed(f"C-line without momenta ({len(p)} fields)")
        raise Malformed(f"{p[0]}-line with {len(p)} fields")
    for i, (tok, kind) in enumerate(zip(p[1:], layout, strict=True), 1):
        if kind == "i" and not _INT.fullmatch(tok):
            raise Malformed(f"{p[0]}-line field {i} {tok!r} is not an integer in canonical decimal")
        if kind == "h" and not _HEX.fullmatch(tok):
            raise Malformed(f"{p[0]}-line field {i} {tok!r} is not a C99 hexadecimal float")
    return p


def new_stats() -> dict[str, Any]:
    return {
        "events": 0,
        "C": 0,
        "D": 0,
        "X": 0,
        "bytes": 0,
        "lines": Counter(),
        "capture_step_models": Counter(),
        "pdg_C": Counter(),
        "pdg_D": Counter(),
        "x_reasons": Counter(),
        "g_tokens": 0,
    }


def _x(gev: int, reason: str, st: dict[str, Any]) -> str:
    st["X"] += 1
    st["x_reasons"][reason] += 1
    return f"{gev} X {reason}"


def classify_event(
    lines: list[str],
    ev: int,
    offset: int,
    route: str,
    st: dict[str, Any],
    policy: Policy,
    classes: _Classes | None = None,
    tracks: int | None = None,
) -> str:
    """The compact line of one event from its T/S/C record lines (``ev`` is the run's local event id;
    ``tracks``, the count of its E line, must equal the number of its T lines when given)."""
    k = classes or _classes(policy)
    gev = offset + ev
    steps: list[list[Any]] = []  # [track, process, [C fields]]
    tlines: list[list[str]] = []
    declared: int | None = None  # the secondaries of the S line whose C lines may follow; None after a T line

    def close_step() -> None:
        if declared is not None and len(steps[-1][2]) != declared:
            raise Malformed(f"S-line declares {declared} secondaries, {len(steps[-1][2])} C-lines follow it")

    for line in lines:
        st["lines"][line[0]] += 1
        p = record_fields(line)
        if p[0] == "E":
            raise Malformed("E-line inside an event")
        if int(p[1]) != ev:
            raise Malformed(f"line of event {p[1]} inside event {ev}")
        if p[0] == "C":
            if declared is None or steps[-1][0] != int(p[2]):
                raise Malformed("C-line not after its S-line")
            cs = steps[-1][2]
            if int(p[3]) != len(cs):
                raise Malformed(f"C-line numbered {p[3]} where {len(cs)} is next")
            if len(cs) == declared:
                raise Malformed(f"more C-lines than the {declared} secondaries the S-line declares")
            cs.append(p)
            continue
        close_step()
        if p[0] == "S":
            steps.append([int(p[2]), p[4], []])
            declared = int(p[7])
        else:
            declared = None
            tlines.append(p)
    close_step()
    if tracks is not None and len(tlines) != tracks:
        raise Malformed(f"E-line of event {ev} counts {tracks} tracks, {len(tlines)} T-lines read")
    if route == "bound_decay":
        caps = [s for s in steps if s[0] == 1 and s[1] == k.capture_process]
        if len(caps) != 1:
            return _x(gev, f"capture_steps={len(caps)}", st)
        cs = caps[0][2]
        for p in cs:
            st["capture_step_models"][p[5]] += 1
        prods = [p for p in cs if p[5].endswith(k.product_suffix)]
        dio = [p for p in cs if p[5].endswith(k.decay_suffix)]
        if prods and dio:
            return _x(gev, "capture_and_dio", st)
        if not prods and not dio:
            return _x(gev, "no_outcome", st)
        cap_track = 1
        if dio:
            for p in dio:
                st["pdg_D"][int(p[4])] += 1
            st["D"] += 1
            return f"{gev} D"
    elif route == "muonic_atom_helper":
        atoms = sorted({s[0] for s in steps if s[0] != 1})
        if len(atoms) != 1:
            return _x(gev, f"atom_tracks={len(atoms)}", st)
        cap_track = atoms[0]
        prods = [s for s in steps if s[0] == cap_track][-1][2]
        for p in prods:
            st["capture_step_models"][p[5]] += 1
        if not prods:
            return _x(gev, "no_products", st)
        if any(int(p[4]) == k.helper_decay_pdg for p in prods):
            for p in prods:
                st["pdg_D"][int(p[4])] += 1
            st["D"] += 1
            return f"{gev} D"
    else:
        raise Refused(f"unknown route {route!r}")
    t_cap = min(float.fromhex(p[7]) for p in prods)
    toks = []
    for p in prods:
        st["pdg_C"][int(p[4])] += 1
        toks.append(f"{p[4]}:{p[6]}:{p[8]}:{p[9]}:{p[10]}")
    nuclei: dict[int, str] = {}
    for t in tlines:
        if int(t[3]) == cap_track and int(t[4]) >= k.nucleus_pdg_min:
            nuclei[int(t[2])] = t[4]
    if nuclei:
        for t in tlines:
            if int(t[4]) == k.photon_pdg and int(t[3]) in nuclei:
                dt = float.fromhex(t[8]) - t_cap
                if dt < k.window_ns:
                    toks.append(f"g22:{t[7]}:{dt.hex()}:{nuclei[int(t[3])]}")
                    st["g_tokens"] += 1
    st["C"] += 1
    return f"{gev} C {t_cap.hex()} " + " ".join(toks)


def header(version: str, config: str, target: str, offset: int, events: int, route: str) -> str:
    return f"# {version} {config} {target} {offset} {events} {route}"


def classify_stream(
    records: Iterable[str],
    out: TextIO,
    *,
    route: str,
    version: str,
    config: str,
    target: str,
    offset: int,
    events: int,
    policy: Policy,
) -> dict[str, Any]:
    """Classify a record stream into compact lines written to ``out`` (header first); return the tallies.

    Raises ``Malformed`` for a line of an unknown kind or that does not follow the record format (the field
    counts and lexemes, the C lines of each S line and the track count of each E line; see the module
    docstring), an event out of order, record lines after the last E line or a count of events other than
    ``events``. A malformed stream stops the classification: no compact line is written for its event."""
    if route not in ROUTES:
        raise Refused(f"route must be one of {ROUTES}")
    k = _classes(policy)
    st = new_stats()
    out.write(header(version, config, target, offset, events, route) + "\n")
    buf: list[str] = []
    expect_ev = 0
    try:
        for line in records:
            st["bytes"] += len(line)
            c = line[0]
            if c == "E":
                p = record_fields(line)
                ev = int(p[1])
                if ev != expect_ev:
                    raise Malformed(f"event {ev} where {expect_ev} expected")
                out.write(classify_event(buf, ev, offset, route, st, policy, k, tracks=int(p[3])) + "\n")
                st["events"] += 1
                expect_ev += 1
                buf = []
            elif c in "TSC":
                buf.append(line)
            else:
                raise Malformed(f"unknown record line {line[:60]!r}")
        if buf:
            raise Malformed(f"{len(buf)} record lines after the last E line")
        if st["events"] != events:
            raise Malformed(f"{st['events']} events read, {events} expected")
    except (ValueError, IndexError) as e:
        raise Malformed(str(e)) from e
    return {key: (dict(v) if isinstance(v, Counter) else v) for key, v in st.items()}


def compact_digests(text: bytes) -> tuple[str, str]:
    """(sha256 of the compact text, sha256 of the text without its header line, which names the revision)."""
    body = text.split(b"\n", 1)[1] if b"\n" in text else b""
    return hashlib.sha256(text).hexdigest(), hashlib.sha256(body).hexdigest()


# ------------------------------------------------------------------------------------------------- numbers
def sig(v: float | Decimal, n: int = 12) -> str:
    """Plain decimal string with ``n`` significant digits (no exponent), half-even."""
    with localcontext() as ctx:
        ctx.prec = 50
        d = Decimal(repr(float(v))) if not isinstance(v, Decimal) else v
        if d == 0:
            return "0"
        q = Decimal(1).scaleb(d.adjusted() - (n - 1))
        return format(d.quantize(q, rounding=ROUND_HALF_EVEN), "f")


def exact_ratio(num: int, den: int) -> str:
    """``num/den`` as an exact plain decimal string; a ratio whose expansion does not terminate is refused."""
    with localcontext() as ctx:
        ctx.prec = 50
        d = Decimal(num) / Decimal(den)
        s = format(d.normalize(), "f")
    if Fraction(s) != Fraction(num, den):
        raise Refused(f"ratio {num}/{den} does not terminate")
    return s


def _decimal_of(f: Fraction) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 50
        return Decimal(f.numerator) / Decimal(f.denominator)


class Acc:
    """Per-capture count accumulator: exact integer sums."""

    def __init__(self) -> None:
        self.n = self.s = self.s2 = 0

    def add(self, k: int) -> None:
        self.n += 1
        self.s += k
        self.s2 += k * k

    def mean_se(self) -> tuple[Fraction, float]:
        mean = Fraction(self.s, self.n)
        var = (Fraction(self.s2) - Fraction(self.s * self.s, self.n)) / (self.n - 1)
        return mean, math.sqrt(var / self.n)


def f2_stats(hist: dict[int, int]) -> tuple[Fraction, float]:
    """f2 = E[n(n-1)]/E[n]^2 (no clipping) and its delta-method standard error on (m = n(n-1), n)."""
    N = sum(hist.values())
    S1 = sum(k * v for k, v in hist.items())
    Sm = sum(k * (k - 1) * v for k, v in hist.items())
    if S1 == 0:
        raise Refused("f2 undefined: no neutron in the selected population")
    f2 = Fraction(N * Sm, S1 * S1)
    a, b = Fraction(Sm, N), Fraction(S1, N)
    vn = (sum(k * k * v for k, v in hist.items()) - Fraction(S1 * S1, N)) / (N - 1)
    vm = (sum((k * (k - 1)) ** 2 * v for k, v in hist.items()) - Fraction(Sm * Sm, N)) / (N - 1)
    cmn = (sum(k * (k - 1) * k * v for k, v in hist.items()) - Fraction(Sm * S1, N)) / (N - 1)
    ga, gb = 1 / b**2, -2 * a / b**3
    var = (ga * ga * vm + gb * gb * vn + 2 * ga * gb * cmn) / N
    return f2, math.sqrt(float(var))


def p_obs_ratio(hist: dict[int, int], eps: str) -> float | None:
    """P_obs(n>=3)/P_obs(n=1) after binomial thinning at efficiency ``eps`` (descriptive)."""
    e = Fraction(eps)
    N = sum(hist.values())
    kmax = max(hist)
    pobs = [
        sum(
            Fraction(hist.get(k, 0), N) * math.comb(k, j) * e**j * (1 - e) ** (k - j)
            for k in range(j, kmax + 1)
        )
        for j in range(kmax + 1)
    ]
    p1 = pobs[1] if len(pobs) > 1 else 0
    p3 = sum(pobs[3:])
    return float(p3 / p1) if p1 else None


# ------------------------------------------------------------------------------------------------ gamma lines
@dataclass(frozen=True)
class GammaParams:
    lines: dict[str, float]
    mev_per_u: float
    residual: tuple[int, int]
    tag_half: float
    sb_lo: float
    sb_hi: float
    sb_num: int
    sb_den: int
    guard_fraction: float
    guard_beta_max: float
    window_ns: float
    inner: float
    cont_num: float
    cont_den: float
    k: int
    offset: int
    photon_pdg: int
    nucleus_pdg_min: int


def gamma_params(policy: Policy) -> GammaParams:
    g, gl, tw = policy["gamma"], policy["guards"]["per_line"], policy["guards"]["tag_width"]
    return GammaParams(
        lines={key: float(v) for key, v in g["lines_MeV"].items()},
        mev_per_u=float(g["mev_per_u"]),
        residual=(int(g["residual_z_a"][0]), int(g["residual_z_a"][1])),
        tag_half=float(g["tag_half_width_MeV"]),
        sb_lo=float(g["sideband_MeV"][0]),
        sb_hi=float(g["sideband_MeV"][1]),
        sb_num=int(g["sideband_scale"][0]),
        sb_den=int(g["sideband_scale"][1]),
        guard_fraction=float(gl["fraction"]),
        guard_beta_max=float(gl["beta_max"]),
        window_ns=float(policy["classification"]["descendant_photon_window_ns"]),
        inner=float(tw["inner_MeV"]),
        cont_num=float(tw["continuum_scale"][0]),
        cont_den=float(tw["continuum_scale"][1]),
        k=int(tw["k"]),
        offset=int(tw["offset"]),
        photon_pdg=int(policy["classification"]["photon_pdg"]),
        nucleus_pdg_min=int(policy["classification"]["nucleus_pdg_min"]),
    )


def e_t(de: float, gp: GammaParams) -> float:
    """Photon energy of a transition of level difference ``de`` after the recoil of the residual."""
    a = gp.residual[1]
    return de - de * de / (2 * a * gp.mev_per_u)


def za(pdg: int) -> tuple[int, int]:
    return (pdg // 10000) % 1000, (pdg // 10) % 1000


Product = tuple[int, float, float, float, float]
Descendant = tuple[int, float, float, int]


def parse_line(line: str) -> tuple[list[Product], list[Descendant]]:
    f = line.split()
    if len(f) < 3 or f[1] != "C":
        raise ValueError("not a capture line")
    prods: list[Product] = []
    desc: list[Descendant] = []
    for tok in f[3:]:
        if tok.startswith("g"):
            pdg, ke, dt, par = tok[1:].split(":")
            desc.append((int(pdg), float.fromhex(ke), float.fromhex(dt), int(par)))
        else:
            pdg, ke, px, py, pz = tok.split(":")
            prods.append(
                (int(pdg), float.fromhex(ke), float.fromhex(px), float.fromhex(py), float.fromhex(pz))
            )
    return prods, desc


def residual(prods: list[Product], gp: GammaParams) -> Product | None:
    nuclei = [p for p in prods if p[0] >= gp.nucleus_pdg_min]
    if not nuclei:
        return None
    return max(nuclei, key=lambda p: (za(p[0])[1], za(p[0])[0]))


def beta_vec(nuc: Product) -> tuple[float, float, float]:
    _, ke, px, py, pz = nuc
    if ke <= 0:
        return (0.0, 0.0, 0.0)
    p2 = px * px + py * py + pz * pz
    m = (p2 - ke * ke) / (2 * ke)
    e = ke + m
    return (px / e, py / e, pz / e)


def rest_energy(ph: Product, b: tuple[float, float, float]) -> float:
    _, e, px, py, pz = ph
    b2 = b[0] ** 2 + b[1] ** 2 + b[2] ** 2
    g = 1 / math.sqrt(1 - b2)
    return g * (e - (b[0] * px + b[1] * py + b[2] * pz))


def capture_counts(prods: list[Product], et: float, gp: GammaParams) -> tuple[int, int, float]:
    """(w, s, beta) for one capture: photons in the tag window and in the sidebands of the residual's rest
    frame."""
    r = residual(prods, gp)
    if r is None or za(r[0]) != gp.residual:
        return 0, 0, 0.0
    b = beta_vec(r)
    w = s = 0
    for ph in prods:
        if ph[0] != gp.photon_pdg:
            continue
        d = abs(rest_energy(ph, b) - et)
        if d <= gp.tag_half:
            w += 1
        elif gp.sb_lo < d <= gp.sb_hi:
            s += 1
    return w, s, math.sqrt(sum(x * x for x in b))


def line_yield(lines: list[str], de: float, gp: GammaParams) -> dict[str, Any]:
    """Per-capture yield y = w - (sb_num/sb_den) s, its standard error, and the integer tallies."""
    et = e_t(de, gp)
    n = 0
    sy = 0.0
    sy2 = 0.0
    W = S = 0
    bmax = 0.0
    for ln in lines:
        prods, _ = parse_line(ln)
        w, s, b = capture_counts(prods, et, gp)
        y = w - gp.sb_num * s / gp.sb_den
        n += 1
        sy += y
        sy2 += y * y
        W += w
        S += s
        if w:
            bmax = max(bmax, b)
    mean = sy / n
    se = math.sqrt(max(sy2 / n - mean * mean, 0.0) * n / (n - 1) / n) if n > 1 else float("nan")
    return {
        "n": n,
        "w": W,
        "s": S,
        "numerator_x20": gp.sb_den * W - gp.sb_num * S,
        "mean": mean,
        "se": se,
        "beta_max_tagged": bmax,
    }


def line_guard(lines: list[str], de: float, gp: GammaParams) -> tuple[bool, int, int, float]:
    """Per-line descendant guard on the selected capture population: (ok, C_line, D_line, beta_max)."""
    et = e_t(de, gp)
    c_line = d_line = 0
    bmax = 0.0
    for ln in lines:
        prods, desc = parse_line(ln)
        w, _, b = capture_counts(prods, et, gp)
        c_line += w
        if w:
            bmax = max(bmax, b)
        for pdg, ke, dt, par in desc:
            if (
                pdg == gp.photon_pdg
                and za(par) == gp.residual
                and dt < gp.window_ns
                and abs(ke - et) <= et * gp.guard_beta_max + gp.tag_half
            ):
                d_line += 1
    ok = (d_line == 0 or d_line <= gp.guard_fraction * (c_line + d_line)) and bmax < gp.guard_beta_max
    return ok, c_line, d_line, bmax


def tag_width_gate(lines: list[str], de: float, gp: GammaParams) -> tuple[bool, int, int, float]:
    """The rest-frame line must be narrower than the tag window: counts with tag_half < |E*-E_t| <= inner may
    not exceed the continuum the sidebands predict there by more than k sigma + offset. (ok, n_in_tag,
    n_inner, expected)."""
    et = e_t(de, gp)
    n1 = n3 = s = 0
    for ln in lines:
        prods, _ = parse_line(ln)
        r = residual(prods, gp)
        if r is None or za(r[0]) != gp.residual:
            continue
        b = beta_vec(r)
        for ph in prods:
            if ph[0] != gp.photon_pdg:
                continue
            d = abs(rest_energy(ph, b) - et)
            if d <= gp.tag_half:
                n1 += 1
            elif d <= gp.inner:
                n3 += 1
            elif gp.sb_lo < d <= gp.sb_hi:
                s += 1
    expect = gp.cont_num * s / gp.cont_den
    return n3 <= expect + gp.k * math.sqrt(expect) + gp.offset, n1, n3, expect


def _boost(e_rest: float, n: list[float], b: list[float]) -> tuple[float, list[float]]:
    """Photon of rest-frame energy e_rest along unit n (rest frame) boosted by velocity b: lab (E, p)."""
    b2 = sum(x * x for x in b)
    if b2 == 0:
        return e_rest, [e_rest * x for x in n]
    g = 1 / math.sqrt(1 - b2)
    bn = sum(bi * ni for bi, ni in zip(b, n, strict=True))
    e = g * e_rest * (1 + bn)
    p = [e_rest * ni + ((g - 1) * e_rest * bn / b2 + g * e_rest) * bi for ni, bi in zip(n, b, strict=True)]
    return e, p


def _iso(rng: random.Random) -> list[float]:
    c = 2 * rng.random() - 1
    ph = 2 * math.pi * rng.random()
    s = math.sqrt(1 - c * c)
    return [s * math.cos(ph), s * math.sin(ph), c]


def _tok(pdg: int, e: float, p: list[float]) -> str:
    return f"{pdg}:{float(e).hex()}:{float(p[0]).hex()}:{float(p[1]).hex()}:{float(p[2]).hex()}"


def _nucleus_tok(pdg: int, a: int, p: list[float], gp: GammaParams) -> str:
    m = a * gp.mev_per_u
    pp = math.sqrt(sum(x * x for x in p))
    ke = math.sqrt(pp * pp + m * m) - m
    return _tok(pdg, ke, p)


SYNTH_SHAPES = ("at_rest", "box_100", "uniform_0_300", "fast_300")


def synth(
    shape: str, n_capt: int, gp: GammaParams, seed: int = 20261013
) -> tuple[list[str], dict[str, int], float]:
    """The line-shape drill: captures with a known number of photons of the two lines (keys of ``gp.lines``,
    the upper line first), Doppler-shifted per ``shape``, plus an exactly subtractable continuum and decoys.
    Returns (compact capture lines, true counts per line, max frame error in keV)."""
    rng = random.Random(seed)
    z, a = gp.residual
    M = a * gp.mev_per_u
    (k_hi, de_hi), (k_lo, de_lo) = list(gp.lines.items())
    t_hi, t_lo = e_t(de_hi, gp), e_t(de_lo, gp)
    truth = {k_hi: 0, k_lo: 0}
    pdg_res = 1000000000 + z * 10000 + a * 10
    pdg_decoy = pdg_res - 10
    lines, maxerr = [], 0.0
    for i in range(n_capt):
        pmag = {"at_rest": 0.0, "box_100": 100.0, "uniform_0_300": 300.0 * rng.random(), "fast_300": 300.0}[
            shape
        ]
        p = [pmag * x for x in _iso(rng)]
        toks, photons = [], []
        kind = i % 10
        if kind in (0, 1, 2, 3, 4):  # the upper line, direct
            seq = [t_hi]
        elif kind in (5, 6):  # the cascade through the lower line into the upper line's level
            seq = [t_lo, t_hi]
        else:
            seq = []
        for et in seq:  # sequential emission with the nucleus recoiling after each photon
            e_nuc = math.sqrt(sum(x * x for x in p) + M * M)
            b = [x / e_nuc for x in p]
            e, pg = _boost(et, _iso(rng), b)
            photons.append((et, e, pg))
            p = [pi - gi for pi, gi in zip(p, pg, strict=True)]
            truth[k_hi if et == t_hi else k_lo] += 1
        res_pdg = pdg_res
        if kind == 9:  # decoy: an A-1 residual with a photon at the upper line's rest energy
            res_pdg = pdg_decoy
            e_nuc = math.sqrt(sum(x * x for x in p) + ((a - 1) * gp.mev_per_u) ** 2)
            e, pg = _boost(t_hi, _iso(rng), [x / e_nuc for x in p])
            toks.append(_tok(22, e, pg))
        res_tok = _nucleus_tok(res_pdg, a - 1 if kind == 9 else a, p, gp)
        e_fin = math.sqrt(sum(x * x for x in p) + M * M)
        bfin = [x / e_fin for x in p]
        for et, e, pg in photons:
            toks.append(_tok(22, e, pg))
            ph: Product = (22, e, pg[0], pg[1], pg[2])
            maxerr = max(maxerr, abs(rest_energy(ph, (bfin[0], bfin[1], bfin[2])) - et) * 1e3)
        if kind == 8:  # continuum on a grid in the final rest frame around both lines
            for et in (t_hi, t_lo):
                for j in range(-130, 130):
                    d = (j + 0.5) * 1e-4  # -12.95 .. +12.95 keV, step 0.1 keV, symmetric
                    e, pg = _boost(et + d, _iso(rng), bfin)
                    toks.append(_tok(22, e, pg))
            e, pg = _boost(t_hi + 0.050, _iso(rng), bfin)  # far continuum that may sit at the lab line energy
            toks.append(_tok(22, e, pg))
        toks.append(_tok(2112, 2.0, [0.0, 0.0, 61.3]))
        toks.append(res_tok)
        rng.shuffle(toks)
        lines.append(f"{i} C {(0.0).hex()} " + " ".join(toks))
    return lines, truth, maxerr


# ------------------------------------------------------------------------------------------------ the reducer
@dataclass
class Chunk:
    """One chunk of a cell: its metadata and its compact lines (header first)."""

    chunk: str
    offset: int
    events: int
    extension: bool
    lines: Iterable[str]
    compact_sha256: str | None = None
    wall_s: float | None = None
    binary_sha256: str | None = None


def _window(p: dict[str, Any]) -> tuple[int, float, float | None, bool]:
    lo = float(p["ke_lo_MeV"])
    hi = float(p["ke_hi_MeV"]) if p.get("ke_hi_MeV") else None
    return int(p["pdg"]), lo, hi, p["lower_edge"] == "exclusive"


def in_window(ke: float, lo: float, hi: float | None, strict: bool) -> bool:
    if strict:
        return ke > lo and (hi is None or ke < hi)
    return lo <= ke and (hi is None or ke < hi)


def _compact(c: Chunk, version: str, config: str, target: str, route: str) -> Iterator[tuple[int, str]]:
    it = iter(c.lines)
    head = next(it, "").split()
    want = ["#", version, config, target, str(c.offset), str(c.events), route]
    if head != want:
        raise Refused(f"malformed chunk {c.chunk}: header {head} != {want}")
    ev = c.offset
    for line in it:
        g = int(line.split(" ", 1)[0])
        if g != ev:
            raise Refused(f"malformed chunk {c.chunk}: event {g} where {ev} expected")
        ev += 1
        yield g, line.rstrip("\n")
    if ev != c.offset + c.events:
        raise Refused(f"malformed chunk {c.chunk}: {ev - c.offset} events, {c.events} in meta")


def gamma_targets(policy: Policy) -> set[str]:
    return {p["target"] for p in policy["predicates"] if p["family"] == "gamma"}


def reduce_cell(
    chunks: list[Chunk],
    *,
    version: str,
    config: str,
    target: str,
    policy: Policy,
    no_outcome: str = "exclude",
) -> tuple[dict[str, Any], str]:
    """Reduce one cell: (record, status) with status ``accepted``, ``short`` (fewer captures than the policy's
    captures_per_configuration and nothing else wrong) or ``refused`` (the record's ``problems`` say why).

    The population is the earliest ``captures_per_configuration`` captures in global event order; quantities
    are computed only when the cell holds exactly that many. ``no_outcome="exclude"`` applies the
    pre-registration's rule (the event leaves both denominators); ``"as_capture"`` is the alternative
    treatment (the event becomes a capture with no products, so it enters the population and the last capture
    leaves)."""
    if no_outcome not in NO_OUTCOME_MODES:
        raise Refused(f"no_outcome must be one of {NO_OUTCOME_MODES}")
    route = route_of(policy, config)
    n_sel = int(policy["setup"]["captures_per_configuration"])
    neutron = int(policy["classification"]["neutron_pdg"])
    nucleus_min = int(policy["classification"]["nucleus_pdg_min"])
    max_ext = int(policy["guards"]["max_extension_chunks"])
    gp = gamma_params(policy)
    chunks = sorted(chunks, key=lambda c: c.offset)
    if not chunks:
        raise Refused(f"malformed cell {version} {config} {target}: no chunk")
    nxt = 0
    for c in chunks:
        if c.offset != nxt:
            raise Refused(f"malformed cell: chunk {c.chunk} offset {c.offset} != {nxt}")
        nxt += c.events
    problems: list[str] = []
    n_ext = sum(c.extension for c in chunks)
    if n_ext > max_ext:
        problems.append(f"extension count {n_ext} > {max_ext}")
    preds = [p for p in policy["predicates"] if p["family"] in ("alcap", "twist")]
    wins = {p["predicate"]: _window(p) for p in preds}
    accs = {key: Acc() for key in wins}
    n_acc = Acc()
    hist: Counter[int] = Counter()
    residuals: Counter[Any] = Counter()
    sel_lines: list[str] | None = [] if target in gamma_targets(policy) else None
    events = caps = decays = xs = 0
    events_scanned = None
    x_examples: list[str] = []
    no_out: list[int] = []
    for c in chunks:
        for g, line in _compact(c, version, config, target, route):
            f = line.split(" ")
            if f[1] == "X" and f[2:] == ["no_outcome"]:
                no_out.append(g)
                if no_outcome == "exclude":
                    continue
                line = f"{g} C {(0.0).hex()}"
                f = line.split(" ")
            events += 1
            kind = f[1]
            if kind == "D":
                decays += 1
                continue
            if kind == "X":
                xs += 1
                if len(x_examples) < 5:
                    x_examples.append(line)
                continue
            if kind != "C":
                raise Refused(f"malformed compact line {line[:60]!r}")
            caps += 1
            if caps > n_sel:
                continue
            per: Counter[str] = Counter()
            nn = 0
            nuclei = []
            for tok in f[3:]:
                if tok[0] == "g":
                    continue
                q = tok.split(":")
                pdg = int(q[0])
                if pdg == neutron:
                    nn += 1
                    continue
                if pdg >= nucleus_min:
                    nuclei.append(pdg)
                ke = float.fromhex(q[1])
                for key, (wp, lo, hi, strict) in wins.items():
                    if pdg == wp and in_window(ke, lo, hi, strict):
                        per[key] += 1
            for key in wins:
                accs[key].add(per[key])
            n_acc.add(nn)
            hist[nn] += 1
            if nuclei:
                residuals[max(nuclei, key=lambda p: (za(p)[1], za(p)[0]))] += 1
            else:
                residuals["none"] += 1
            if sel_lines is not None:
                sel_lines.append(line)
            if caps == n_sel:
                events_scanned = events
    if xs:
        problems.append(f"X={xs} ({x_examples[:2]})")
    n_capt = min(caps, n_sel)
    out: dict[str, Any] = {
        "version": version,
        "config": config,
        "target": target,
        "route": route,
        "chunks": [
            {
                "chunk": c.chunk,
                "offset": c.offset,
                "events": c.events,
                "extension": c.extension,
                "compact_sha256": c.compact_sha256,
                "wall_s": c.wall_s,
                "binary_sha256": c.binary_sha256,
            }
            for c in chunks
        ],
        "events_total": events,
        "captures_total": caps,
        "decays_total": decays,
        "X": xs,
        "events_scanned": events_scanned,
        "N_capt": n_capt,
        "extension_chunks": n_ext,
    }
    if n_capt and n_capt == n_sel:
        quantities: dict[str, dict[str, Any]] = {}
        for key, a in accs.items():
            _, se = a.mean_se()
            quantities[key] = {"count": a.s, "x": exact_ratio(a.s, a.n), "x_se": sig(se)}
        _, se = n_acc.mean_se()
        for p in policy["predicates"]:
            if p["family"] == "nmean":
                quantities[p["predicate"]] = {
                    "count": n_acc.s,
                    "x": exact_ratio(n_acc.s, n_acc.n),
                    "x_se": sig(se),
                }
        f2, f2se = f2_stats(hist)
        f2s = sig(_decimal_of(f2), 15)
        for p in policy["predicates"]:
            if p["family"] == "f2":
                quantities[p["predicate"]] = {"x": f2s, "x_se": sig(f2se)}
        out["quantities"] = quantities
        out["neutron_hist"] = {str(k): hist[k] for k in sorted(hist)}
        out["f2"] = {"x": f2s, "x_se": sig(f2se)}
        eb = policy["rule"]["efficiency_bracket"]
        out["p_obs_ratio_3plus_over_1"] = {
            e: p_obs_ratio(hist, e) for e in (eb["eps_lo"], eb["eps0"], eb["eps_hi"])
        }
        out["residuals_top"] = {str(k): v for k, v in residuals.most_common(40)}
        out["residuals_distinct"] = len(residuals)
        if sel_lines is not None:
            gam: dict[str, Any] = {}
            guards: dict[str, Any] = {}
            tags: dict[str, Any] = {}
            for key, de in gp.lines.items():
                r = line_yield(sel_lines, de, gp)
                gam[key] = {
                    "x": exact_ratio(r["numerator_x20"], gp.sb_den * r["n"]),
                    "x_se": sig(r["se"]),
                    "w": r["w"],
                    "s": r["s"],
                    "beta_max_tagged": r["beta_max_tagged"],
                }
                ok, C, D, bmax = line_guard(sel_lines, de, gp)
                guards[key] = {"ok": ok, "C_line": C, "D_line": D, "beta_max": bmax}
                wide_ok, n1, n3, ex = tag_width_gate(sel_lines, de, gp)
                tags[key] = {"ok": wide_ok, "n_1keV": n1, "n_1to3keV": n3, "continuum_expected": ex}
                if not ok:
                    problems.append(f"line guard refused {key}: C={C} D={D} beta_max={bmax}")
                if not wide_ok:
                    problems.append(f"tag-width gate refused {key}: n1={n1} n3={n3} expected {ex}")
                quantities[key] = {"x": gam[key]["x"], "x_se": gam[key]["x_se"]}
            out["gamma"], out["line_guard"], out["tag_width"] = gam, guards, tags
    out["problems"] = problems
    stops = sum(c.events for c in chunks)
    n = len(no_out)
    out.update(
        {
            "stops_total": stops,
            "n_no_outcome": n,
            "no_outcome_events": no_out,
            "no_outcome_rate": n / stops if stops else None,
        }
    )
    guard_ok = Fraction(n, stops) <= Fraction(policy["rule"]["no_outcome"]["rate_limit"]) if stops else True
    out["no_outcome_guard_ok"] = guard_ok
    if problems or not guard_ok:
        status = "refused"
    elif caps < n_sel:
        status = "short" if caps else "refused"
    else:
        status = "accepted"
    return out, status


ALTERNATIVE_FIELDS = ("quantities", "neutron_hist", "f2")


def differing_fields(a: dict[str, Any], b: dict[str, Any], keys: Iterable[str]) -> list[str]:
    """The keys among ``keys`` whose values differ between the two records (a missing key differs)."""
    marker = object()
    return [k for k in keys if a.get(k, marker) != b.get(k, marker)]


# ------------------------------------------------------------------------------------------------- cells
def cell_key(cell: dict[str, Any]) -> tuple[str, str, str]:
    return cell["version"], cell["config"], cell["target"]


def admit_cell(cell: dict[str, Any], policy: Policy) -> None:
    """Refuse a reduced cell the comparison may not use, naming the clause."""
    key = " ".join(cell_key(cell))
    n_sel = int(policy["setup"]["captures_per_configuration"])
    if cell["N_capt"] != n_sel:
        raise Refused(f"{key}: N_capt {cell['N_capt']} != {n_sel}")
    if cell["captures_total"] < n_sel:
        raise Refused(f"{key}: captures_total {cell['captures_total']} < {n_sel}")
    if cell["X"] != 0:
        raise Refused(f"{key}: X {cell['X']}")
    if cell["problems"]:
        raise Refused(f"{key}: problems {cell['problems']}")
    max_ext = int(policy["guards"]["max_extension_chunks"])
    if (
        cell["extension_chunks"] > max_ext
        or sum(c["extension"] for c in cell["chunks"]) != cell["extension_chunks"]
    ):
        raise Refused(f"{key}: extension chunks {cell['extension_chunks']}")
    nxt = 0
    for c in cell["chunks"]:
        if c["offset"] != nxt:
            raise Refused(f"{key}: chunk {c['chunk']} offset {c['offset']} != {nxt}")
        nxt += c["events"]
    if cell["stops_total"] != nxt:
        raise Refused(f"{key}: stops_total {cell['stops_total']} != chunk events {nxt}")
    if cell["events_total"] + cell["n_no_outcome"] != cell["stops_total"]:
        raise Refused(f"{key}: events_total + n_no_outcome != stops_total")
    if cell["captures_total"] + cell["decays_total"] + cell["X"] != cell["events_total"]:
        raise Refused(f"{key}: captures + decays + X != events_total")
    if cell["n_no_outcome"] != len(cell["no_outcome_events"]):
        raise Refused(f"{key}: n_no_outcome != its event list")
    if Fraction(cell["n_no_outcome"], cell["stops_total"]) > Fraction(
        policy["rule"]["no_outcome"]["rate_limit"]
    ):
        raise Refused(f"{key}: no-outcome rate above the limit")
    if cell["target"] in gamma_targets(policy):
        for line in policy["gamma"]["lines_MeV"]:
            if not cell["line_guard"][line]["ok"]:
                raise Refused(f"{key}: line guard {line}")
            if not cell["tag_width"][line]["ok"]:
                raise Refused(f"{key}: tag width {line}")


def load_cells(path: str | Path, policy: Policy) -> dict[tuple[str, str, str], dict[str, Any]]:
    """The reduced cells of ``path`` (one JSON object per line), each admitted, exactly one per version x
    configuration x target of the policy."""
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cell = json.loads(line)
        key = cell_key(cell)
        if key in cells:
            raise Refused(f"cell {key} appears twice")
        admit_cell(cell, policy)
        cells[key] = cell
    setup = policy["setup"]
    want = {
        (v, c, t["target"]) for v in setup["versions"] for c in setup["configs"] for t in setup["targets"]
    }
    if set(cells) != want:
        missing, extra = sorted(want - set(cells)), sorted(set(cells) - want)
        raise Refused(f"cells differ from the policy's matrix: missing {missing[:3]} extra {extra[:3]}")
    return cells


# ------------------------------------------------------------------------------------------------- the rule
def edges(ref: str, stat: str, syst: str, xse: str, policy: Policy) -> tuple[str, str]:
    """The rescue bracket [L(r_lo), U(r_hi)], L rounded down and U up at the policy's quantum."""
    rule = policy["rule"]
    with localcontext() as ctx:
        ctx.prec = 60
        k = Decimal(rule["k_fail"])
        rlo, rhi = Decimal(rule["efficiency_bracket"]["r_lo"]), Decimal(rule["efficiency_bracket"]["r_hi"])
        q = Decimal(rule["rescue_quantum"])
        r, st, sy, xs = (Decimal(v) for v in (ref, stat, syst, xse))
        lo = rlo * r - k * ((rlo * st) ** 2 + (rlo * sy) ** 2 + xs**2).sqrt()
        hi = rhi * r + k * ((rhi * st) ** 2 + (rhi * sy) ** 2 + xs**2).sqrt()
        return str(lo.quantize(q, rounding=ROUND_FLOOR)), str(hi.quantize(q, rounding=ROUND_CEILING))


def comparison_rows(
    policy: Policy, cells: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, str]]:
    """One comparison row per row of the policy, in its order: the model x and x_se of the row's cell, the
    reference cells of the policy and, for a rescue row, the rescue edges."""
    out = []
    for fr in policy["rows"]:
        key = (fr["version"], fr["config"], fr["target"])
        cell = cells.get(key)
        if cell is None:
            raise Refused(f"{fr['row_id']}: cell {key} missing")
        admit_cell(cell, policy)
        q = cell["quantities"].get(fr["predicate"])
        if q is None:
            raise Refused(f"{fr['row_id']}: quantity {fr['predicate']} absent from the cell")
        lo, hi = (
            edges(fr["ref"], fr["ref_stat"], fr["ref_syst"], q["x_se"], policy) if fr["rescue"] else ("", "")
        )
        out.append(
            {
                "row_id": fr["row_id"],
                "predicate": fr["predicate"],
                "family": fr["family"],
                "target": fr["target"],
                "version": fr["version"],
                "config": fr["config"],
                "n_capt": str(cell["N_capt"]),
                "x": q["x"],
                "x_se": q["x_se"],
                "ref": str(fr["ref"]),
                "ref_stat": str(fr["ref_stat"]),
                "ref_syst": str(fr["ref_syst"]),
                "rescue_lo": lo,
                "rescue_hi": hi,
            }
        )
    return out


@dataclass(frozen=True)
class Verdict:
    row_id: str
    verdict: str
    kind: str
    pull: str


def _pull(d: Fraction, s2: Fraction) -> str:
    """(x - ref)/sigma_tot to 2 decimals, half-even, decided exactly: with t = d^2/s2 the square of the pull,
    the hundredths n = floor(100 |pull|) = isqrt(floor(10000 t)), and |pull| is compared with the midpoint
    (2n + 1)/200 through squares, so no rounding of a square root decides a digit."""
    if s2 == 0:
        raise Refused("sigma_tot is zero")
    t = d * d / s2
    n = math.isqrt(math.floor(t * 10000))
    mid = Fraction(2 * n + 1, 200) ** 2
    if t > mid or (t == mid and n % 2 == 1):
        n += 1
    return f"{'-' if d < 0 else ''}{n // 100}.{n % 100:02d}"


def judge_row(row: dict[str, str], k_pass: Fraction, k_fail: Fraction) -> Verdict:
    d = Fraction(row["x"]) - Fraction(row["ref"])
    s2 = Fraction(row["ref_stat"]) ** 2 + Fraction(row["ref_syst"]) ** 2 + Fraction(row["x_se"]) ** 2
    within_pass = d * d <= k_pass * k_pass * s2
    within_fail = d * d <= k_fail * k_fail * s2
    rescue = row["rescue_lo"] != ""
    inside_bracket = rescue and Fraction(row["rescue_lo"]) <= Fraction(row["x"]) <= Fraction(row["rescue_hi"])
    if within_pass:
        verdict, kind = "PASS", "pass"
    elif not within_fail and not inside_bracket:
        verdict, kind = "FAIL", "fail"
    else:
        verdict, kind = "UNQUALIFIED", "tension" if within_fail else "efficiency_limited"
    return Verdict(row["row_id"], verdict, kind, _pull(d, s2))


def judge(policy: Policy, comparison: list[dict[str, str]]) -> list[Verdict]:
    """The verdict of every row, in the policy's order. The roster is the policy's rows: a missing, extra or
    repeated row, a reference cell or identity field that differs from the policy, or a rescue edge that
    differs from its recomputation is refused (``Refused``), never judged."""
    got: dict[str, dict[str, str]] = {}
    for r in comparison:
        if tuple(r) != COMPARISON_COLUMNS:
            raise Refused(f"comparison columns are not {COMPARISON_COLUMNS}")
        if r["row_id"] in got:
            raise Refused(f"row {r['row_id']} appears twice")
        got[r["row_id"]] = r
    roster = {fr["row_id"]: fr for fr in policy["rows"]}
    extra = sorted(set(got) - set(roster))
    if extra:
        raise Refused(f"rows not in the pre-registration: {extra[:3]}")
    k_pass, k_fail = Fraction(policy["rule"]["k_pass"]), Fraction(policy["rule"]["k_fail"])
    out = []
    for fr in policy["rows"]:
        row = got.get(fr["row_id"])
        if row is None:
            raise Refused(f"row {fr['row_id']} missing")
        for k in ("predicate", "family", "target", "version", "config", "ref", "ref_stat", "ref_syst"):
            if row[k] != str(fr[k]):
                raise Refused(f"{fr['row_id']}: {k} {row[k]!r} differs from the pre-registration's {fr[k]!r}")
        if fr["rescue"]:
            want = edges(fr["ref"], fr["ref_stat"], fr["ref_syst"], row["x_se"], policy)
            got_edges = (row["rescue_lo"], row["rescue_hi"])
            if got_edges != want:
                raise Refused(f"{fr['row_id']}: rescue edges {got_edges} differ from {want}")
        elif row["rescue_lo"] or row["rescue_hi"]:
            raise Refused(f"{fr['row_id']}: rescue edges on a row without a bracket")
        out.append(judge_row(row, k_pass, k_fail))
    return out
