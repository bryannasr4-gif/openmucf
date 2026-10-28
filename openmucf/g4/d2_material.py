"""Binding a measured capture vector to a material, and the selection Geant4 then makes.

This is the reference implementation of the binding rules and the selection arithmetic that
``cpp/material_selector/G4MuonicMaterialSelector.cc`` applies, so that the C++ provider can be checked against
it without Geant4 and against the measured tables of :mod:`openmucf.g4.d2_measured`.

**Binding.** A caller binds a material to a vector entry explicitly, naming the entry's profile, the
material's formula and its element counts, and acknowledging that the entry is a provisional, source-reported
value. The binding is refused, by :class:`BindRefused` and nothing else, unless:

* the acknowledgement is ``True`` (the boolean, not a truthy value);
* the formula parses, names each element once, and gives the counts the request states and the entry carries;
* the material's elements are exactly the entry's, each once, matched by atomic number and not by position
  (so a material that lists oxygen first binds as well as one that lists it last), with no extra element --
  hydrogen included, since the entries are for hydrogen-free solids -- and none missing;
* every atom number density is finite and positive;
* the ratio of each element's atom number density to the first element's equals the ratio of their counts
  within a relative ``1e-9`` -- or within ``1e-4`` for exactly the named Geant4 NIST oxides of
  :data:`NIST_POLICY` that are defined by weight fraction, bound to the entry of their own formula;
* any optional metadata uses only the keys :data:`OPTIONAL_METADATA`, and a stated hydration is anhydrous.

A table is read only if its units are the dimensionless ``P=1 n=1``, its scope the hydrogen-free nominal solid
and its status the provisional, source-reported value; any other table is refused before it can be bound.

**Selection.** A bound material is sampled with Geant4's own cumulative search, over the entry's probabilities
in the material's element order: one uniform draw for the element, then the stock isotope draw, one draw iff
the selected element has several isotopes. The element is never redrawn through the stock selector. The
binary64 cumulative sums of the vector are those of the written table; the ladder of :func:`ladder_counts`
reproduces the counts of the midpoint draws ``(k + 1/2) / N`` exactly.

**Allowance.** A bound ratio within a relative ``epsilon`` of the counts' ratio moves a binary probability by
at most ``epsilon / (4 (1 - epsilon))`` (:func:`representation_allowance`); a comparison of the delivered
vector with a reference carries that term separately. It is not a chemical uncertainty.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from openmucf.g4 import d2 as selector
from openmucf.g4 import d2_measured as measured
from openmucf.g4 import spec

ATOM_TOL = 1e-9
NIST_TOL = 1e-4
#: The Geant4 NIST materials, defined by weight fraction, admitted under :data:`NIST_TOL`: each only for the
#: entry of its own formula.
NIST_POLICY = {
    "G4_ALUMINUM_OXIDE": "Al2O3",
    "G4_CALCIUM_OXIDE": "CaO",
    "G4_FERRIC_OXIDE": "Fe2O3",
    "G4_LEAD_OXIDE": "PbO",
    "G4_MAGNESIUM_OXIDE": "MgO",
    "G4_SILICON_DIOXIDE": "SiO2",
    "G4_TITANIUM_DIOXIDE": "TiO2",
}
OPTIONAL_METADATA = ("polymorph", "hydration", "bulk_density_g_cm3", "temperature_K")
ANHYDROUS = ("anhydrous", "none")
REFUSED = "ENTRY_REFUSED"
_VALIDITY_TOKENS = ("form", "basis", "scope", "status", "material")
#: The only meaning a reader accepts: dimensionless probabilities and counts, for the hydrogen-free nominal
#: solid, carried as reported and provisional.
SUPPORTED_UNITS = {"P": "1", "n": "1"}
SUPPORTED_SCOPE = "hydrogen_free_nominal_solid"
SUPPORTED_STATUS = "SOURCE_REPORTED_PROVISIONAL"
#: The sum of a written vector may differ from one by rounding only.
SUM_TOL = 4 * 2.0**-52


class BindRefused(ValueError):
    """A binding request the rules refuse. Nothing is bound; the stock selector is not substituted."""

    code = REFUSED

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{REFUSED}: {reason}")


@dataclass(frozen=True)
class VectorEntry:
    """A measured table: its profile, its formula, and per element (ascending Z) the probability and count."""

    profile: str
    formula: str
    z: tuple[int, ...]
    p: tuple[float, ...]
    n: tuple[int, ...]


def load_entry(text: str) -> VectorEntry:
    """A ``material_vector`` table of seam ``d2_atomic_capture``, parsed by the Layer-1 reader and checked
    against its own ``material`` token."""
    table = spec.parse(text)
    d = table.directives
    if d["SEAM"] != measured.SEAM or d["TABLE"] != measured.TABLE or tuple(d["COLUMNS"].split()) != (
        measured.COLUMNS
    ):
        raise BindRefused("not a measured vector table")
    tokens = spec.validity_assignments(table)
    if tuple(sorted(tokens)) != tuple(sorted(_VALIDITY_TOKENS)) or tokens["form"] != "vector" or tokens[
        "basis"
    ] != "atom":
        raise BindRefused("the table's validity tokens")
    units = dict(item.split("=", 1) for item in d["UNITS"].split() if "=" in item)
    if units != SUPPORTED_UNITS or len(d["UNITS"].split()) != len(SUPPORTED_UNITS):
        raise BindRefused(f"the table's units {d['UNITS']!r} are not the dimensionless P and n")
    if tokens["scope"] != SUPPORTED_SCOPE:
        raise BindRefused(f"the table's scope {tokens['scope']!r} is not {SUPPORTED_SCOPE}")
    if tokens["status"] != SUPPORTED_STATUS:
        raise BindRefused(f"the table's status {tokens['status']!r} is not {SUPPORTED_STATUS}")
    zs = tuple(int(r[0]) for r in table.records)
    ps = tuple(float(r[1]) for r in table.records)
    counts = []
    for r in table.records:
        c = float(r[2])
        if not (c >= 1 and c == int(c)):
            raise BindRefused("a count is not a positive integer")
        counts.append(int(c))
    if len(zs) < 2 or len(set(zs)) != len(zs):
        raise BindRefused("a vector names at least two distinct elements")
    if not all(0.0 < p < 1.0 for p in ps) or abs(math.fsum(ps) - 1.0) > SUM_TOL:
        raise BindRefused("the probabilities are not a distribution")
    if formula_counts(tokens["material"]) != dict(zip(zs, counts, strict=True)):
        raise BindRefused("the table's counts differ from its formula")
    return VectorEntry(d["PROFILE"], tokens["material"], zs, ps, tuple(counts))


def formula_counts(formula: Any) -> dict[int, int]:
    """Atoms per formula unit by Z, each element named once."""
    if not isinstance(formula, str) or not formula:
        raise BindRefused("a formula is required")
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
    if "".join(a + b for a, b in parts) != formula:
        raise BindRefused(f"formula {formula!r} does not parse")
    out: dict[int, int] = {}
    for symbol, digits in parts:
        z = selector._SYMBOLS.get(symbol)
        if z is None or z in out or digits.startswith("0"):
            raise BindRefused(f"formula {formula!r}: element {symbol!r}")
        out[z] = int(digits) if digits else 1
    return out


@dataclass(frozen=True)
class MaterialRequest:
    """What a caller states. ``elements`` is in the material's own element order: ``(Z, atom density)``."""

    name: str
    elements: tuple[tuple[int, float], ...]
    formula: str
    counts: tuple[tuple[int, int], ...]
    provisional_source_acknowledged: bool
    profile: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Binding:
    profile: str
    material: str
    probabilities: tuple[float, ...]
    tolerance: float
    allowance: float


def representation_allowance(epsilon: float) -> float:
    """The largest change of a binary probability that a relative ratio error ``epsilon`` allows."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon in (0, 1)")
    return epsilon / (4 * (1 - epsilon))


def _finite_positive(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def bind(entry: VectorEntry, request: MaterialRequest) -> Binding:
    """Bind or refuse; see the module docstring for the rules."""
    if not isinstance(request, MaterialRequest) or not isinstance(entry, VectorEntry):
        raise BindRefused("typed entry and request required")
    if request.provisional_source_acknowledged is not True:
        raise BindRefused("the provisional, source-reported status was not acknowledged")
    if request.profile != entry.profile:
        raise BindRefused(f"the request names {request.profile!r}, not {entry.profile!r}")
    if not isinstance(request.metadata, Mapping) or set(request.metadata) - set(OPTIONAL_METADATA):
        raise BindRefused("unknown metadata")
    hydration = request.metadata.get("hydration")
    if hydration is not None and hydration not in ANHYDROUS:
        raise BindRefused("a hydrated material is outside the hydrogen-free scope")
    want = dict(zip(entry.z, entry.n, strict=True))
    if formula_counts(request.formula) != want:
        raise BindRefused("the formula differs from the entry's")
    if not isinstance(request.counts, tuple) or any(
        not isinstance(c, tuple) or len(c) != 2 for c in request.counts
    ):
        raise BindRefused("counts are (Z, count) pairs")
    counts: dict[int, int] = {}
    for z, c in request.counts:
        if isinstance(z, bool) or isinstance(c, bool) or not isinstance(z, int) or not isinstance(c, int):
            raise BindRefused("integer Z and count required")
        if z in counts:
            raise BindRefused("a count repeats an element")
        counts[z] = c
    if counts != want:
        raise BindRefused("the stated counts differ from the formula's")
    if not isinstance(request.elements, tuple) or not request.elements:
        raise BindRefused("the material has no elements")
    zs = []
    for item in request.elements:
        if not isinstance(item, tuple) or len(item) != 2:
            raise BindRefused("an element is a (Z, density) pair")
        z, density = item
        if isinstance(z, bool) or not isinstance(z, int):
            raise BindRefused("an element's Z is an integer")
        if not _finite_positive(density):
            raise BindRefused(f"element Z={z}: atom number density {density!r} is not finite and positive")
        zs.append(z)
    if len(set(zs)) != len(zs):
        raise BindRefused("an element repeats")
    if set(zs) != set(want):
        raise BindRefused(f"the material's elements {sorted(zs)} are not the entry's {sorted(want)}")
    nist = NIST_POLICY.get(request.name)
    tol = NIST_TOL if nist is not None and nist == entry.formula else ATOM_TOL
    z0, d0 = request.elements[0]
    for z, density in request.elements[1:]:
        ratio = (density / d0) / (want[z] / want[z0])
        if not abs(ratio - 1.0) <= tol:
            raise BindRefused(f"the atom ratio of Z={z} to Z={z0} is off the counts by {ratio - 1.0:.3e}")
    p_by_z = dict(zip(entry.z, entry.p, strict=True))
    probabilities = tuple(p_by_z[z] for z in zs)
    return Binding(entry.profile, request.name, probabilities, tol, representation_allowance(tol))


def cumulative(weights: Sequence[float]) -> list[float]:
    """Geant4's running sums, in binary64, in the given order."""
    out, total = [], 0.0
    for w in weights:
        total += w
        out.append(total)
    return out


def select(weights: Sequence[float], u: float) -> int:
    """The element index Geant4's cumulative search returns for the uniform draw ``u``."""
    cum = cumulative(weights)
    if len(cum) == 1:
        return 0
    return min(bisect_left(cum, cum[-1] * u), len(cum) - 1)


def ladder_counts(weights: Sequence[float], draws: int = selector.DRAW_COUNT) -> tuple[int, ...]:
    """How many midpoint draws ``(k + 1/2) / draws`` select each index, by bisection on each boundary."""
    if draws < 1 or not weights:
        raise ValueError("invalid ladder")
    bounds = [0]
    for index in range(len(weights) - 1):
        lo, hi = 0, draws
        while lo < hi:
            mid = (lo + hi) // 2
            if select(weights, (mid + 0.5) / draws) <= index:
                lo = mid + 1
            else:
                hi = mid
        bounds.append(lo)
    bounds.append(draws)
    return tuple(b - a for a, b in zip(bounds, bounds[1:], strict=False))


def stock_weights(elements: Sequence[tuple[int, float]]) -> tuple[float, ...]:
    """The stock selector's weights ``factor(Z) * Z * n``, in the material's order, as Geant4 forms them."""
    return tuple(selector.factor(z) * z * density for z, density in elements)
