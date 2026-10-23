"""The rate at which a stopped negative muon disappears from a muonic atom, as Geant4 predicts it.

A negative muon in the 1s state of a muonic atom disappears by nuclear capture or by decay, and the inverse
of its measured mean life is the sum of the two rates. Geant4 computes that sum on two routes, each with
its own decay rate:

* ``G4MuonMinusBoundDecay::ApplyYourself`` draws the time of a stop from
  ``GetMuonCaptureRate(Z, A) + GetMuonDecayRate(Z, A, muMass, nuclMass)``;
* ``G4MuonicAtomHelper::ConstructMuonicAtom`` gives a muonic atom the lifetime
  ``1. / (GetMuonCaptureRate(Z, A) + GetMuonDecayRate(Z))``, which ``G4MuonicAtomDecay`` reads.

:func:`bound_decay_rate` and :func:`helper_decay_rate` evaluate the two decay rates with each product and
quotient grouped as the compiled expressions group them, and ``data/g4/d1/d1_decay_rate.oracle`` holds what
compiled builds returned, so a test compares the two with no Geant4 present. :func:`comparison_rows` then
gives, for each profile of the capture table and each mean life a source printed for its own measurement,
the measured rate and, when the measurement names a nucleus, each route's rate for it. The capture rate
follows the lookup a patched build runs (:func:`openmucf.g4.d1_natural.lookup`).

Rates are in ns^-1 internally, Geant4's unit, and in 1e6/s in the files.
"""

from __future__ import annotations

import csv
import decimal
import hashlib
import io
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from openmucf.g4 import d1_natural as natural
from openmucf.g4 import spec
from openmucf.g4.sources import d1_nuclear_capture as d1
from openmucf.g4.sources import mizuno2025, suzuki1987

#: The committed oracle, relative to the repository root.
ORACLE_RELPATH = "data/g4/d1/d1_decay_rate.oracle"
#: The generated comparison.
COMPARISON_RELPATH = "data/g4/d1/mean_life_comparison.csv"
#: The harvest driver, whose SHA-256 the oracle carries.
HARNESS_RELPATH = "cpp/tools/harvest_decay.cc"
#: The committed transcription of Table 1 of Mizuno et al.
MIZUNO_TABLE1_RELPATH = "data/g4/d1/mizuno2025_table1.csv"
#: The builds the harvest runs on: one unpatched install per Geant4 revision the patches serve.
BUILDS = (
    ("unpatched-v11.4.2", "v11.4.2"),
    ("unpatched-v11.5.0.beta", "v11.5.0.beta"),
)
#: A recorded compile flag: a plain compiler option, carrying neither whitespace, ``#`` nor ``;``, so the
#: flags cannot break the oracle's line-oriented header or its ``;`` delimiter.
FLAG_TOKEN = re.compile(r"[A-Za-z0-9_+=.,/-]+")

#: The literals of ``G4MuonMinusBoundDecay::GetMuonDecayRate(Z, A, muMass, nuclMass)``.
FREE_DECAY_RATE = 0.45517005
BOUND_THRESHOLD_Z = 14
LIGHT_SCALE, LIGHT_MASS_SCALE = 0.5, 0.06
HEAVY_SCALE, HEAVY_SLOPE = 0.868699, 0.708985
#: The literals of ``G4MuonicAtomHelper::GetMuonDecayRate(Z)``: the rate it tabulates for Z = 1, and the
#: coefficient of its formula.
HELPER_HYDROGEN_RATE = 0.4558514
HELPER_SCALE = 2.5
#: ``CLHEP::fine_structure_const`` as the builds the oracle records compiled it (its ``K alpha`` line).
ALPHA = float.fromhex("0x1.de3d429d88bb3p-8")

#: The sweep of the oracle: every A from Z to 3Z below the threshold, where the decay rate reads the
#: nuclear mass, and every A from Z to 300 from the threshold on, where it does not.
LOW_A_FACTOR = 3
HIGH_Z_MAX, HIGH_A_MAX = 120, 300
HELPER_Z = range(-1, 122)

#: The measurement sources, named as the profiles their rows make.
SUZUKI, MIZUNO = suzuki1987.PROFILE, mizuno2025.PROFILE
#: The compiled-in capture table's source comment credits its data to Suzuki, Measday and Roalsvig, with
#: hydrogen and helium carved out to other papers.
TABLE_SOURCE, TABLE_SOURCE_MIN_Z = SUZUKI, 3

#: The routes a capture lookup can take, and the classes of a comparison row.
PROFILE_ROUTE, TABLE_ROUTE, FORMULA_ROUTE = natural.PROFILE_ROUTE, natural.TABLE_ROUTE, natural.FORMULA_ROUTE
COMPARED, OTHER_SOURCE, OTHER_BASIS = "compared", "other-source", "other-basis"
FORMULA, NO_KEY = "formula", "no-key"
POPULATIONS = (COMPARED, OTHER_SOURCE, OTHER_BASIS, FORMULA, NO_KEY)
#: The two routes, by the column suffix the comparison gives each and the class whose decay rate it uses.
ROUTES = (("bound", "G4MuonMinusBoundDecay"), ("helper", "G4MuonicAtomHelper"))

COLUMNS = (
    "source", "locator", "label", "z", "basis", "measured_state", "key_a", "profile", "lookup", "population",
    "mean_life_ns", "mean_life_unc_ns", "measured_rate", "measured_rate_unc", "printed_capture_rate",
    "capture_rate", "capture_rate_unc", "decay_rate_bound", "decay_rate_helper", "rate_bound", "rate_helper",
    "within_bound", "within_helper",
)
#: The band a route's rate is held to: ``|rate - measured| <= K * sqrt(measured_unc^2 + capture_unc^2)``.
K = 3

_RAW_LINE = re.compile(r"(K) (\S+) (\S+)|(H) (-?[0-9]+) (\S+)|(B) ([0-9]+) ([0-9]+) (\S+) (\S+)")


class LifetimeError(RuntimeError):
    """A harvest, a committed file or an input this comparison cannot be computed from."""


# --------------------------------------------------------------------------------------------
# the two decay rates
# --------------------------------------------------------------------------------------------


def bound_decay_rate(z: int, a: int, mu_mass: float, nucl_mass: float) -> float:
    """``G4MuonMinusBoundDecay::GetMuonDecayRate(Z, A, muMass, nuclMass)`` in ns^-1. The masses are read only
    below :data:`BOUND_THRESHOLD_Z`; each product and quotient groups as the C++ expression does."""
    if z == 0 and a == 0:
        return FREE_DECAY_RATE / d1.MICROSECOND
    free = FREE_DECAY_RATE / d1.MICROSECOND
    rate = 1.0
    x = z * ALPHA
    if z < BOUND_THRESHOLD_Z:
        rate -= x * x * (LIGHT_SCALE + LIGHT_MASS_SCALE * mu_mass / nucl_mass)
    else:
        rate -= x * x * (HEAVY_SCALE - x * HEAVY_SLOPE)
    rate *= free
    return rate


def helper_decay_rate(z: int, zeff: Sequence[float], zmin: int, zmax: int) -> float:
    """``G4MuonicAtomHelper::GetMuonDecayRate(Z)`` in ns^-1, with the helper's effective charges and the
    clamp it applies before indexing them."""
    if z == 1:
        return HELPER_HYDROGEN_RATE / d1.MICROSECOND
    free = FREE_DECAY_RATE / d1.MICROSECOND
    rate = 1.0
    x = zeff[max(min(z, zmax), zmin)] * ALPHA
    rate -= HELPER_SCALE * x * x
    rate *= free
    return rate


def helper_clamp(found: d1.D1Extraction) -> tuple[int, int]:
    """The clamp the helper applies before indexing its effective charges, read from its source."""
    coefficients = found.coefficients
    return int(coefficients["zmin"]), int(coefficients["zmax"])


# --------------------------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Oracle:
    """What compiled builds returned: the constants, the helper's rate for each Z, the bound-decay rate with
    the nuclear mass it was passed below the threshold, and from the threshold on its rate at A = Z."""

    header: dict[str, str]
    constants: dict[str, float]
    helper: dict[int, float]
    light: dict[tuple[int, int], tuple[float, float]]
    heavy: dict[int, float]


def light_keys() -> list[tuple[int, int]]:
    """The keys below the threshold, Z ascending outermost."""
    return [(z, a) for z in range(1, BOUND_THRESHOLD_Z) for a in range(z, LOW_A_FACTOR * z + 1)]


def heavy_keys() -> list[tuple[int, int]]:
    """The keys from the threshold on, Z ascending outermost: the keys the oracle's digest is over."""
    return [(z, a) for z in range(BOUND_THRESHOLD_Z, HIGH_Z_MAX + 1) for a in range(z, HIGH_A_MAX + 1)]


def digest(rates: Sequence[float]) -> str:
    """SHA-256 over the big-endian IEEE-754 binary64 bytes of ``rates``, in order."""
    running = hashlib.sha256()
    for rate in rates:
        running.update(struct.pack(">d", rate))
    return running.hexdigest()


Raw = tuple[dict[str, float], dict[int, float], dict[tuple[int, int], tuple[float, float]]]


def parse_raw(text: str) -> Raw:
    """The lines ``harvest_decay.cc`` prints, refusing any other line, a repeated key and a missing one."""
    constants: dict[str, float] = {}
    helper: dict[int, float] = {}
    bound: dict[tuple[int, int], tuple[float, float]] = {}
    for number, line in enumerate(text.split("\n"), start=1):
        if not line:
            continue
        match = _RAW_LINE.fullmatch(line)
        if match is None:
            raise LifetimeError(f"line {number}: not a harvest line: {line!r}")
        if match[1]:
            if match[2] in constants:
                raise LifetimeError(f"line {number}: constant {match[2]} repeated")
            constants[match[2]] = float.fromhex(match[3])
        elif match[4]:
            z = int(match[5])
            if z in helper:
                raise LifetimeError(f"line {number}: helper Z={z} repeated")
            helper[z] = float.fromhex(match[6])
        else:
            key = (int(match[8]), int(match[9]))
            if key in bound:
                raise LifetimeError(f"line {number}: bound-decay key {key} repeated")
            bound[key] = (float.fromhex(match[10]), float.fromhex(match[11]))
    if set(constants) != {"mu_mass", "alpha", "microsecond"}:
        raise LifetimeError(f"the constants are {sorted(constants)}")
    if sorted(helper) != list(HELPER_Z):
        raise LifetimeError("the helper's rates do not cover its Z range")
    if sorted(bound) != sorted(light_keys() + heavy_keys()):
        raise LifetimeError("the bound-decay rates do not cover the sweep")
    return constants, helper, bound


def recorded_flags(folder: Path, build: str) -> str:
    """The compile flags the harvest of ``build`` wrote beside its output (``flags.txt``), stripped: one line
    of :data:`FLAG_TOKEN` options separated by single spaces. A missing, empty or whitespace-only record, a
    byte outside ASCII, a CR and anything else are refused, naming the build."""
    path = folder / "flags.txt"
    raw = path.read_bytes() if path.is_file() else b""
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise LifetimeError(f"{build}: the recorded compile flags (flags.txt) are not ASCII") from None
    flags = text.strip()
    if not flags:
        raise LifetimeError(f"{build}: the harvest recorded no compile flags (flags.txt)")
    if b"\r" in raw or not all(FLAG_TOKEN.fullmatch(token) for token in flags.split(" ")):
        raise LifetimeError(
            f"{build}: the recorded compile flags (flags.txt) are not one line of plain compiler options: "
            f"{flags!r}")
    return flags


def harvest(raw: Path, root: Path) -> bytes:
    """The committed oracle, from one raw directory per build of :data:`BUILDS`. Every build must have
    compiled the committed driver with one compiler and the same flags, which it recorded beside its output
    (:func:`recorded_flags`), written nothing to stderr and printed the same bytes, and from the threshold on
    each Z's rate must be the same at every A. The oracle names the flags the builds recorded."""
    harness = hashlib.sha256((root / HARNESS_RELPATH).read_bytes()).hexdigest()
    outputs: list[bytes] = []
    versions: list[str] = []
    compilers: set[str] = set()
    flags: set[str] = set()
    for build, _ in BUILDS:
        folder = raw / build
        if not folder.is_dir():
            raise LifetimeError(f"{build}: the raw tree has no directory for this build ({folder})")
        if (folder / "sha256.txt").read_text(encoding="ascii").split()[0] != harness:
            raise LifetimeError(f"{build}: the compiled driver is not {HARNESS_RELPATH}")
        if (folder / "stderr.txt").read_bytes():
            raise LifetimeError(f"{build}: the driver wrote to stderr")
        outputs.append((folder / "decay.txt").read_bytes())
        versions.append((folder / "version.txt").read_text(encoding="ascii").strip())
        compilers.add((folder / "gcc.txt").read_text(encoding="ascii").strip())
        flags.add(recorded_flags(folder, build))
    if any(output != outputs[0] for output in outputs):
        raise LifetimeError("the builds printed different rates")
    if len(compilers) != 1:
        raise LifetimeError(f"the builds used different compilers: {sorted(compilers)}")
    if len(flags) != 1:
        raise LifetimeError(f"the builds recorded different flags: {sorted(flags)}")
    constants, helper, bound = parse_raw(outputs[0].decode("ascii"))
    for z in range(BOUND_THRESHOLD_Z, HIGH_Z_MAX + 1):
        if len({bound[(z, a)][1] for a in range(z, HIGH_A_MAX + 1)}) != 1:
            raise LifetimeError(f"Z={z}: the bound-decay rate depends on the mass number")
    heavy = [bound[key][1] for key in heavy_keys()]
    built = ", ".join(f"{build} ({version})" for (build, _), version in zip(BUILDS, versions, strict=True))
    lines = [
        "# G4MuonicData D1 -- compiled-Geant4 oracle for the decay rate of a negative muon in a muonic atom",
        "#",
        "# HARVESTED, not generated: these values came out of Geant4-linked binaries, and make audit does",
        "# not rewrite this file. tests/test_g4lifetime.py recomputes in Python every decay rate, evaluated",
        "# with the oracle's constants and nuclear-mass inputs, and compares, bit for bit.",
        "#",
        f"# driver            {HARNESS_RELPATH}",
        f"# driver_sha256     {harness}",
        f"# builds            {built}, the version each geant4-config printed;",
        "#                   both printed the same bytes",
        f"# compiler          {compilers.pop()}",
        f"# driver_flags      {flags.pop()}; the rates are computed inside the Geant4 libraries",
        f"# raw_sha256        {hashlib.sha256(outputs[0]).hexdigest()}",
        f"# heavy_sweep       Z {BOUND_THRESHOLD_Z}..{HIGH_Z_MAX} x A Z..{HIGH_A_MAX} = {len(heavy)} points,",
        "#                   Z ascending outermost; each Z's rate is the same at every A, and its B line",
        "#                   gives it at A = Z",
        "# heavy_rule        sha256 over the concatenated big-endian IEEE-754 binary64 bytes of the rates",
        f"# heavy_sha256      {digest(heavy)}",
        "#",
        "# K <name> <value>           mu_mass (MeV), alpha, microsecond (ns)",
        "# H <Z> <rate>               G4MuonicAtomHelper::GetMuonDecayRate(Z), ns^-1",
        "# B <Z> <A> <mass> <rate>    G4Nucleus::AtomicMass(A, Z) (MeV) and",
        "#                            G4MuonMinusBoundDecay::GetMuonDecayRate(Z, A, mu_mass, mass), ns^-1;",
        "#                            mass - from the threshold on, where it is not read",
    ]
    lines += [f"K {name} {constants[name].hex()}" for name in ("mu_mass", "alpha", "microsecond")]
    lines += [f"H {z} {helper[z].hex()}" for z in HELPER_Z]
    for z, a in light_keys():
        mass, rate = bound[(z, a)]
        lines.append(f"B {z} {a} {mass.hex()} {rate.hex()}")
    lines += [f"B {z} {z} - {bound[(z, z)][1].hex()}" for z in range(BOUND_THRESHOLD_Z, HIGH_Z_MAX + 1)]
    return ("\n".join(lines) + "\n").encode("ascii")


def load_oracle(path: Path) -> Oracle:
    """The committed oracle."""
    raw = path.read_bytes()
    if b"\r" in raw:
        raise LifetimeError(f"{path.name} contains CR")
    header: dict[str, str] = {}
    constants: dict[str, float] = {}
    helper: dict[int, float] = {}
    light: dict[tuple[int, int], tuple[float, float]] = {}
    heavy: dict[int, float] = {}
    for line in raw.decode("ascii").splitlines():
        if line.startswith("#"):
            field = re.fullmatch(r"# ([a-z0-9_]+) +(.*)", line)
            if field:
                header[field[1]] = field[2]
            continue
        parts = line.split(" ")
        if parts[0] == "K" and len(parts) == 3:
            constants[parts[1]] = float.fromhex(parts[2])
        elif parts[0] == "H" and len(parts) == 3:
            helper[int(parts[1])] = float.fromhex(parts[2])
        elif parts[0] == "B" and len(parts) == 5 and parts[3] != "-":
            light[(int(parts[1]), int(parts[2]))] = (float.fromhex(parts[3]), float.fromhex(parts[4]))
        elif parts[0] == "B" and len(parts) == 5 and parts[1] == parts[2]:
            heavy[int(parts[1])] = float.fromhex(parts[4])
        else:
            raise LifetimeError(f"{path.name}: not an oracle line: {line!r}")
    return Oracle(header, constants, helper, light, heavy)


# --------------------------------------------------------------------------------------------
# the measurements
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """A mean life a source printed for its own measurement, with the capture rate it printed beside it
    (1e6/s)."""

    source: str
    locator: str
    label: str
    z: int
    basis: str
    mass_number: int | None
    mean_life: str
    mean_life_unc: str
    printed_capture_rate: decimal.Decimal
    #: The state of the muonic atom a note of the source says the mean life describes, when it names one
    #: (:func:`openmucf.g4.sources.suzuki1987.measured_states`); empty when it names none.
    state: str = ""


def suzuki_measurements(path: Path, states: Mapping[str, str] | None = None) -> list[Measurement]:
    """The rows of Tables III and IV marked as the authors' own measurements that print a mean life, its
    uncertainty and a capture rate, in the transcription's order, each with the state ``states`` (locator to
    state) gives its row."""
    states = states or {}
    out: list[Measurement] = []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        printed = row["mean_life_ns"] and row["mean_life_unc_ns"] and row["rate"]
        if row["own_measurement"] != "true" or not printed:
            continue
        if row["target_basis"] not in ("isotope", "natural", "unspecified"):
            raise LifetimeError(f"{row['target_label']!r}: basis {row['target_basis']!r} names no nucleus")
        prefix = re.match(r"[0-9]+", row["target_label"])
        mass_number = None
        if row["target_basis"] == "isotope":
            if prefix is None:
                raise LifetimeError(f"{row['target_label']!r}: an isotope row without a mass number")
            mass_number = int(prefix.group())
        locator = f"Table {row['table']} p.{row['printed_page']} row {row['page_row_ordinal']}"
        out.append(Measurement(
            SUZUKI, locator, row["target_label"], int(row["Z"]), row["target_basis"], mass_number,
            row["mean_life_ns"], row["mean_life_unc_ns"], suzuki1987.scaled(row, "rate"),
            states.get(locator, ""),
        ))
    return out


def mizuno_measurements(path: Path, symbols: Mapping[str, int]) -> list[Measurement]:
    """Every row of Table 1, in printed order."""
    out: list[Measurement] = []
    for number, row in enumerate(mizuno2025.load_table1(path), start=1):
        label = re.fullmatch(r"(nat|[0-9]+)([A-Z][a-z]?)", row.nuclide)
        if label is None or label[2] not in symbols:
            raise LifetimeError(f"Table 1 row {number}: {row.nuclide!r} is not a mass number and a symbol")
        natural_row = label[1] == "nat"
        out.append(Measurement(
            MIZUNO, f"{row.locator} row {number}", row.nuclide, symbols[label[2]],
            "natural" if natural_row else "isotope", None if natural_row else int(label[1]),
            row.lifetime_ns, row.lifetime_unc_ns, decimal.Decimal(row.rate),
        ))
    return out


def key_mass_number(
    measurement: Measurement, references: Mapping[int, int], isotopes: Mapping[int, Sequence[natural.Isotope]]
) -> int | None:
    """The mass number Geant4 is asked about for the nucleus a measurement names: the printed one; for a
    target printed with no mass number or as natural, the record the isotope audit establishes rests on a
    measurement of the element printed with no mass number, or else the single isotope of a natural
    composition that has one; otherwise none."""
    if measurement.mass_number is not None:
        return measurement.mass_number
    if measurement.z in references:
        return references[measurement.z]
    members = isotopes.get(measurement.z, ())
    if len(members) == 1:
        return members[0].a
    return None


def population(
    measurement: Measurement, route: str, value_source: str, value_basis: str, single: bool
) -> str:
    """The class of a row with a key: the formula answers it; a value from another source than the
    measurement's answers it; a value measured on another kind of target answers it (natural composition
    against nuclide, one and the same for an element with a single isotope); or it is compared."""
    if route == FORMULA_ROUTE:
        return FORMULA
    if value_source != measurement.source:
        return OTHER_SOURCE
    measured_basis = "isotope" if measurement.basis == "isotope" else "natural"
    if measured_basis != value_basis and not single:
        return OTHER_BASIS
    return COMPARED


# --------------------------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------------------------


def within(rate: str, measured: str, sigmas: Sequence[str]) -> bool:
    """``|rate - measured| <= K * sqrt(sum sigma^2)`` for the values as the comparison prints them, decided
    on squares in exact decimal arithmetic, so a row on the band's edge is inside it."""
    with decimal.localcontext() as context:
        context.prec = 400
        context.traps[decimal.Inexact] = True
        difference = decimal.Decimal(rate) - decimal.Decimal(measured)
        bound = sum((decimal.Decimal(sigma) ** 2 for sigma in sigmas), decimal.Decimal(0))
        return difference * difference <= K * K * bound


def _profile_records(table: spec.G4DatTable) -> dict[tuple[int, int], tuple[float, float]]:
    """A profile's rows a transported nucleus can reach (:func:`openmucf.g4.d1_natural.profile_rows`), with
    their uncertainties (1e6/s)."""
    reach = natural.profile_rows(table)
    return {(int(z), int(a)): (float(value), float(unc)) for z, a, value, unc in table.records
            if (int(z), int(a)) in reach}


def comparison_rows(root: Path) -> list[dict[str, str]]:
    """One row per profile of the capture table and measurement: each route's rate, capture plus decay, the
    measured rate, and whether each route's rate lies in the band."""
    found = d1.load(root / d1.VENDORED_RELPATH)
    helper_found = d1.load_helper(root / d1.HELPER_RELPATH)
    model = natural.reference_model(found)
    helper_model = natural.reference_model(helper_found)
    zmin, zmax = helper_clamp(helper_found)
    oracle = load_oracle(root / ORACLE_RELPATH)
    if oracle.constants.get("alpha") != ALPHA or oracle.constants.get("microsecond") != d1.MICROSECOND:
        raise LifetimeError("the oracle's alpha or microsecond is not the one this module computes with")
    mu_mass = oracle.constants["mu_mass"]
    isotopes = natural.load_isotopes(root / natural.ISOTOPES_RELPATH)
    symbols = {members[0].symbol: z for z, members in isotopes.items()}
    references = natural.element_label_records(d1.load_isotope_audit(root / d1.AUDIT_RELPATH))
    tables = natural.capture_profiles(root / "data" / "g4" / "d1")
    zeff_table = {int(z): float(value) for z, value in spec.parse(
        (root / "data" / "g4" / "d1" / "d1_zeff.g4dat").read_bytes().decode("ascii")).records}
    for name in tables:
        if name not in (spec.PARITY_PROFILE, SUZUKI, MIZUNO):
            raise LifetimeError(f"profile {name!r}: no measurement source is known for its rows")
    states = suzuki1987.measured_states(suzuki1987.load_printed_rows(root / suzuki1987.PRINTED_ROWS_RELPATH))
    measurements = suzuki_measurements(root / suzuki1987.QUANTITY_ROWS_RELPATH, states) + mizuno_measurements(
        root / MIZUNO_TABLE1_RELPATH, symbols)
    compiled = {(z, a): unc for z, a, _, unc in found.capture_records}
    rows: list[dict[str, str]] = []
    for name, table in tables.items():
        profile = None if name == spec.PARITY_PROFILE else _profile_records(table)
        reach = None if profile is None else {key: value for key, (value, _) in profile.items()}
        for m in measurements:
            tau, tau_unc = float(m.mean_life), float(m.mean_life_unc)
            measured, measured_unc = repr(d1.MICROSECOND / tau), repr(d1.MICROSECOND * tau_unc / (tau * tau))
            row = dict.fromkeys(COLUMNS, "")
            row.update({
                "source": m.source, "locator": m.locator, "label": m.label, "z": str(m.z), "basis": m.basis,
                "measured_state": m.state,
                "profile": name, "mean_life_ns": m.mean_life, "mean_life_unc_ns": m.mean_life_unc,
                "measured_rate": measured, "measured_rate_unc": measured_unc,
                "printed_capture_rate": format(m.printed_capture_rate, "f"),
            })
            a = key_mass_number(m, references, isotopes)
            if a is None:
                row["population"] = NO_KEY
                rows.append(row)
                continue
            route = natural.lookup(m.z, a, reach, found.capture_records, model)
            if natural.lookup(m.z, a, reach, helper_found.capture_records, helper_model) != route:
                raise LifetimeError(f"({m.z}, {a}) under {name}: the compiled copies return different rates")
            if zeff_table.get(m.z) != helper_found.zeff[max(min(m.z, zmax), zmin)]:
                raise LifetimeError(f"Z={m.z}: the muon_zeff table and the helper's compiled charge differ")
            if m.z < BOUND_THRESHOLD_Z:
                if (m.z, a) not in oracle.light:
                    raise LifetimeError(f"({m.z}, {a}): the oracle carries no nuclear mass for it")
                mass = oracle.light[(m.z, a)][0]
            else:
                mass = math.nan
            decay_bound = bound_decay_rate(m.z, a, mu_mass, mass)
            decay_helper = helper_decay_rate(m.z, helper_found.zeff, zmin, zmax)
            rate_bound = route.rate + decay_bound
            rate_helper = 1.0 / (1.0 / (route.rate + decay_helper))
            unc: float | None = None
            value_source = value_basis = ""
            if route.route == PROFILE_ROUTE and profile is not None:
                unc, value_source, value_basis = profile[(m.z, a)][1], name, "isotope"
            elif route.route == TABLE_ROUTE:
                unc = compiled[(m.z, a)]
                value_source = TABLE_SOURCE if m.z >= TABLE_SOURCE_MIN_Z else ""
                # A record resting on a measurement of the element printed with no mass number counts as a
                # natural composition; lead's matches a reconstructed combination with a radiogenic-lead row
                # (DATASET_D1.md section 11).
                value_basis = "natural" if references.get(m.z) == a else "isotope"
            row.update({
                "key_a": str(a), "lookup": route.route,
                "population": population(
                    m, route.route, value_source, value_basis, len(isotopes.get(m.z, ())) == 1),
                "capture_rate": repr(route.rate * d1.MICROSECOND),
                "decay_rate_bound": repr(decay_bound * d1.MICROSECOND),
                "decay_rate_helper": repr(decay_helper * d1.MICROSECOND),
                "rate_bound": repr(rate_bound * d1.MICROSECOND),
                "rate_helper": repr(rate_helper * d1.MICROSECOND),
            })
            if unc is not None:
                row["capture_rate_unc"] = repr(unc)
                for suffix, _ in ROUTES:
                    inside = within(row[f"rate_{suffix}"], measured, (measured_unc, row["capture_rate_unc"]))
                    row[f"within_{suffix}"] = "true" if inside else "false"
            rows.append(row)
    return rows


def render_csv(rows: Sequence[Mapping[str, str]]) -> bytes:
    """``rows`` as LF-terminated CSV bytes."""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("ascii")


def read_comparison(path: Path) -> list[dict[str, str]]:
    """The committed comparison."""
    raw = path.read_bytes()
    if b"\r" in raw:
        raise LifetimeError(f"{path.name} contains CR")
    reader = csv.DictReader(io.StringIO(raw.decode("ascii")))
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise LifetimeError(f"{path.name} header is {reader.fieldnames!r}, expected {COLUMNS!r}")
    return list(reader)


# --------------------------------------------------------------------------------------------
# the document's tables and counts
# --------------------------------------------------------------------------------------------


def compared(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return [row for row in rows if row["population"] == COMPARED]


def measurements_compared(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Mapping[str, str]]:
    """The measurements compared under at least one profile, each with its first compared row. The decay
    rates, the measured rate and the printed capture rate of a measurement do not depend on the profile."""
    out: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in compared(rows):
        out.setdefault((row["source"], row["locator"]), row)
    return out


def sigma(row: Mapping[str, str]) -> float:
    """The band's standard deviation for ``row``, 1e6/s."""
    return math.hypot(float(row["measured_rate_unc"]), float(row["capture_rate_unc"]))


def decay_below(rows: Sequence[Mapping[str, str]], suffix: str) -> int:
    """The compared measurements whose decay rate on route ``suffix`` lies below the measured rate less the
    capture rate the source prints beside the mean life."""
    return sum(
        1 for row in measurements_compared(rows).values()
        if float(row[f"decay_rate_{suffix}"])
        < float(row["measured_rate"]) - float(row["printed_capture_rate"])
    )


def document_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    """The counts the document states: the measurements of each source, those compared under a profile, and
    those whose decay rate on each route lies below the measured rate less the printed capture rate."""
    return {
        "suzuki": len({row["locator"] for row in rows if row["source"] == SUZUKI}),
        "mizuno": len({row["locator"] for row in rows if row["source"] == MIZUNO}),
        "measured": len(measurements_compared(rows)),
        "helper_below": decay_below(rows, "helper"),
        "bound_below": decay_below(rows, "bound"),
    }


SUMMARY_HEADER = (
    "| profile | rows compared | within the band, `G4MuonMinusBoundDecay` | "
    "within the band, `G4MuonicAtomHelper` |\n"
    "|---|---|---|---|"
)


def summary(rows: Sequence[Mapping[str, str]]) -> list[tuple[str, int, int, int]]:
    """``(profile, compared, within on the first route, within on the second)`` per profile, in the
    comparison's order."""
    out: list[tuple[str, int, int, int]] = []
    for name in dict.fromkeys(row["profile"] for row in rows):
        mine = [row for row in compared(rows) if row["profile"] == name]
        first, second = (sum(row[f"within_{suffix}"] == "true" for row in mine) for suffix, _ in ROUTES)
        out.append((name, len(mine), first, second))
    return out


def render_summary_table(rows: Sequence[Mapping[str, str]]) -> str:
    lines = [SUMMARY_HEADER]
    lines += [f"| `{name}` | {count} | {first} | {second} |" for name, count, first, second in summary(rows)]
    return "\n".join(lines)


OUTSIDE_HEADER = (
    "| route | measurement | target | profiles | difference (σ) | capture part (σ) |\n"
    "|---|---|---|---|---|---|"
)


def _signed(value: float) -> str:
    """``value`` to a tenth, signed; a value that rounds to zero is ``0.0``."""
    text = format(value, "+.1f")
    return "0.0" if text in ("+0.0", "-0.0") else text


def outside(rows: Sequence[Mapping[str, str]]) -> list[tuple[str, str, str, list[str], float, float]]:
    """``(class, measurement, target, profiles, difference, capture part)`` for every compared row outside the
    band on a route, the profiles of one measurement merged where its figures are the same; the difference
    and its capture part are in units of the band's standard deviation."""
    groups: dict[tuple[str, str, str, str, str], list[str]] = {}
    values: dict[tuple[str, str, str, str, str], tuple[float, float]] = {}
    for suffix, name in ROUTES:
        for row in compared(rows):
            if row[f"within_{suffix}"] != "false":
                continue
            s = sigma(row)
            difference = (float(row[f"rate_{suffix}"]) - float(row["measured_rate"])) / s
            capture = (float(row["capture_rate"]) - float(row["printed_capture_rate"])) / s
            measurement = f"`{row['source']}` {row['locator']}"
            key = (name, measurement, row["label"], _signed(difference), _signed(capture))
            groups.setdefault(key, []).append(f"`{row['profile']}`")
            values[key] = (difference, capture)
    return [(k[0], k[1], k[2], profiles, *values[k]) for k, profiles in groups.items()]


def render_outside_table(rows: Sequence[Mapping[str, str]]) -> str:
    lines = [OUTSIDE_HEADER]
    lines += [
        f"| `{name}` | {measurement} | {target} | {', '.join(profiles)} | "
        f"{_signed(difference)} | {_signed(capture)} |"
        for name, measurement, target, profiles, difference, capture in outside(rows)
    ]
    return "\n".join(lines)
