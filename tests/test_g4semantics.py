"""T-112 .. T-119 -- the semantic layer: what a dataset must mean, beyond what its grammar allows.

`cpp/src/G4MuonicDataSemantics.cc` is the production copy: the standalone validator runs it as
`V-17` and the Geant4-facing glue runs it inside `Load()`, so the same six rule families decide
whether a patched build may read a directory at all. That copy is exercised by the `ctest` cases
named `validate_semantics_*`, which need a C++ toolchain.

What is here is a second, independent statement of the same six rules, written against
`openmucf.g4.spec` rather than against the C++ reader, so that the shipped bytes are held to them
on every platform in ordinary CI -- including the ones with no compiler. Two implementations that
must agree is the point: a rule that is wrong in one of them shows up as a disagreement rather than
as a green run.

* **T-112** -- the shipped dataset yields no issue under any of the six rules.
* **T-113 .. T-118** -- one drill per rule family: one stated edit of one shipped member, each
  derived from that member's own bytes, and the code that edit must raise.
* **T-119** -- the two implementations name the same codes and the same key-domain bounds, so
  neither can drift from the other in silence.

Every key, column, unit and bound below is either read from the shipped files at run time or held
against the production source by T-119; the drills state no dataset value.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re
import shutil

import pytest

from openmucf.g4 import spec

REPO = pathlib.Path(__file__).resolve().parents[1]
#: The assembled dataset the patched build reads: both seams' members in one directory.
DATASET_DIRS = (REPO / "data" / "g4" / "d1", REPO / "data" / "g4" / "d3")
#: The production copy of the rules, which T-119 holds this module against.
SOURCE = REPO / "cpp" / "src" / "G4MuonicDataSemantics.cc"

DATASET_NAME = "G4MuonicData"
CAPTURE, ZEFF, KSHELL, LEVELS = "nuclear_capture_rate", "muon_zeff", "k_shell_energy", "level_energy"
D1_SEAM, D3_SEAM = "d1_nuclear_capture", "d3_transitions"
RATE_UNIT, ZEFF_UNIT, ENERGY_UNIT = "1e6/s", "dimensionless", "keV"
LISTED, NATURAL, REPRESENTATIVE = "listed", "natural_and_listed", "most_abundant_and_listed"
#: The key domain and the level-index range, named as the production source names them.
BOUNDS = {
    "kMinZ": 1,
    "kMaxZ": 120,
    "kMaxA": 300,
    "kMaxZeffZ": 100,
    "kMinLevelIndex": 2,
    "kMaxLevelIndex": 14,
}
CODES = ("S001", "S002", "S003", "S004", "S005", "S006")


@dataclasses.dataclass(frozen=True)
class Issue:
    code: str
    file: str
    profile: str
    table: str
    text: str


@dataclasses.dataclass(frozen=True)
class Loaded:
    """One parsed member, with the name it was read from."""

    file: str
    profile: str
    table: str
    directives: dict[str, str]
    columns: tuple[str, ...]
    records: tuple[tuple[float, ...], ...]


def load(*directories: pathlib.Path) -> list[Loaded]:
    """Every `*.g4dat` member of these directories, in bytewise-sorted name order."""
    files = sorted(
        (path for directory in directories for path in directory.iterdir() if path.name.endswith(".g4dat")),
        key=lambda path: path.name.encode("utf-8"),
    )
    out = []
    for path in files:
        table = spec.parse(path.read_bytes().decode("ascii"))
        out.append(
            Loaded(
                file=path.name,
                profile=table.directives.get("PROFILE", ""),
                table=table.directives.get("TABLE", ""),
                directives=dict(table.directives),
                columns=tuple(table.directives.get("COLUMNS", "").split()),
                records=tuple(table.records),
            )
        )
    return out


def assignments(value: str, delimiter: str) -> dict[str, str]:
    """`NAME<delimiter>VALUE` tokens. Raises `ValueError` on a malformed or repeated one."""
    out: dict[str, str] = {}
    for token in value.split():
        name, found, rest = token.partition(delimiter)
        if not found or not name or not rest:
            raise ValueError(f"assignment {token!r} is not NAME{delimiter}VALUE")
        if name in out:
            raise ValueError(f"assigns {name!r} twice")
        out[name] = rest
    return out


def level_index(columns: tuple[str, ...]) -> int | None:
    """The N of a `Z A e2 .. eN u2 .. uN` sequence, or None when the sequence is not one."""
    if len(columns) < 4 or len(columns) % 2:
        return None
    n = (len(columns) - 2) // 2 + 1
    if not BOUNDS["kMinLevelIndex"] <= n <= BOUNDS["kMaxLevelIndex"]:
        return None
    expected = ["Z", "A"] + [f"e{i}" for i in range(2, n + 1)] + [f"u{i}" for i in range(2, n + 1)]
    return n if list(columns) == expected else None


def fixed_columns(name: str) -> list[str] | None:
    if name in (CAPTURE, KSHELL):
        return ["Z", "A", "value", "unc"]
    if name == ZEFF:
        return ["Z", "value"]
    return None


SEAM_OF = {CAPTURE: D1_SEAM, ZEFF: D1_SEAM, KSHELL: D3_SEAM, LEVELS: D3_SEAM}
UNIT_OF = {CAPTURE: RATE_UNIT, ZEFF: ZEFF_UNIT, KSHELL: ENERGY_UNIT, LEVELS: ENERGY_UNIT}


def _table_issues(member: Loaded, version: str) -> list[Issue]:
    out: list[Issue] = []

    def add(code: str, text: str) -> None:
        out.append(Issue(code, member.file, member.profile, member.table, text))

    if member.directives.get("DATASET", "") != DATASET_NAME:
        add("S001", "#DATASET")
    declared = member.directives.get("VERSION", "")
    if not declared:
        add("S001", "#VERSION is empty")
    elif declared != version:
        add("S001", "#VERSION differs from the first loaded file")

    if member.table not in SEAM_OF:
        add("S002", f"'#TABLE {member.table}' is not a table this dataset defines")
        return out
    expected = fixed_columns(member.table)
    if expected is None:
        n = level_index(member.columns)
        if n is None:
            add("S002", "#COLUMNS is not Z A e2 .. eN u2 .. uN")
            return out
        expected = ["Z", "A"] + [f"e{i}" for i in range(2, n + 1)] + [f"u{i}" for i in range(2, n + 1)]
    if member.directives.get("SEAM", "") != SEAM_OF[member.table]:
        add("S002", "#SEAM")
    if list(member.columns) != expected:
        add("S002", "#COLUMNS")
        return out
    if not member.records:
        add("S002", "declares no record")

    two_key = member.table != ZEFF
    values = expected[2:] if two_key else expected[1:]
    unit = UNIT_OF[member.table]

    try:
        units = assignments(member.directives.get("UNITS", ""), "=")
    except ValueError as error:
        add("S004", f"#UNITS {error}")
    else:
        for column in values:
            if units.get(column, "") != unit:
                add("S004", f"#UNITS assigns {column!r} the unit {units.get(column, '')!r}")
                break

    convention = ""
    z_range: tuple[int, int] | None = None
    try:
        validity = assignments(member.directives.get("VALIDITY", ""), ":")
    except ValueError as error:
        add("S003", f"#VALIDITY {error}")
    else:
        for name in sorted(validity):
            if name not in ("Z", "A"):
                add("S003", f"#VALIDITY assigns {name!r}, which this layer has no convention for")
        if "Z" not in validity:
            add("S003", "#VALIDITY assigns no 'Z'")
        elif validity["Z"] != LISTED:
            match = re.fullmatch(r"([0-9]+)-([0-9]+)", validity["Z"])
            if match is None or int(match.group(1)) > int(match.group(2)):
                add("S003", f"#VALIDITY assigns 'Z:{validity['Z']}', which is not an inclusive MIN-MAX")
            else:
                z_range = (int(match.group(1)), int(match.group(2)))
        if not two_key:
            if "A" in validity:
                add("S003", f"{ZEFF} has no mass-number column, and #VALIDITY assigns 'A'")
        elif "A" not in validity:
            add("S003", "#VALIDITY assigns no 'A'")
        elif validity["A"] not in (LISTED, NATURAL, REPRESENTATIVE):
            add("S003", f"#VALIDITY assigns 'A:{validity['A']}', which is no known convention")
        else:
            convention = validity["A"]

    reported: set[str] = set()

    def once(rule: str, text: str) -> None:
        if rule not in reported:
            reported.add(rule)
            add("S003" if rule.startswith("key") else "S005", text)

    for record in member.records:
        z = int(record[0])
        key = "-".join(str(int(part)) for part in record[: 2 if two_key else 1])
        if two_key and not BOUNDS["kMinZ"] <= z <= BOUNDS["kMaxZ"]:
            once("key_z", f"row {z} has a Z outside the key domain")
        if not two_key and not 0 <= z <= BOUNDS["kMaxZeffZ"]:
            once("key_z", f"row {z} has a Z outside the key domain")
        if z_range is not None and not z_range[0] <= z <= z_range[1]:
            once("key_range", f"row {key} has a Z outside the declared range")
        if two_key:
            a = int(record[1])
            if a != 0 and not z <= a <= BOUNDS["kMaxA"]:
                once("key_a", f"row {z}-{a} has a mass number outside the key domain")
            if a == 0 and convention == LISTED:
                once("key_natural", f"row {z}-{a} carries A = 0 under 'A:{LISTED}'")

    level_values = len(values) // 2 if member.table == LEVELS else 0
    for record in member.records:
        floats = list(record[2:]) if two_key else list(record[1:])
        z = int(record[0])
        for index, value in enumerate(floats):
            if not math.isfinite(value):
                once("finite", f"column {values[index]!r} is not finite")
            uncertainty = index >= level_values if member.table == LEVELS else values[index] == "unc"
            if uncertainty:
                if not value >= 0:
                    once("uncertainty", f"column {values[index]!r} is a negative uncertainty")
                continue
            if member.table == ZEFF:
                sentinel = z == 0 and member.profile == "parity"
                if not (value == 0 if sentinel else 0 < value <= z):
                    once("zeff", f"row {z} has an effective charge outside (0, Z]")
                continue
            if not value > 0:
                once("positive", f"row {z} column {values[index]!r} is not positive")
    return out


def _pair_issues(members: list[Loaded]) -> list[Issue]:
    out: list[Issue] = []
    by_profile: dict[str, dict[str, Loaded]] = {}
    for member in members:
        by_profile.setdefault(member.profile, {})[member.table] = member
    for profile in sorted(by_profile):
        kshell, levels = by_profile[profile].get(KSHELL), by_profile[profile].get(LEVELS)
        if kshell is None and levels is None:
            continue
        if kshell is None or levels is None:
            out.append(Issue("S006", "", profile, "", "carries one energy table but not the other"))
            continue
        n = level_index(levels.columns)
        if list(kshell.columns) != fixed_columns(KSHELL) or n is None:
            continue
        keys = {tuple(record[:2]) for record in kshell.records}
        other = {tuple(record[:2]) for record in levels.records}
        if keys != other:
            out.append(Issue("S006", "", profile, "", "the two energy tables carry different key sets"))
            continue
        def convention_of(member: Loaded) -> str:
            table = spec.G4DatTable(member.directives, member.records)
            return spec.validity_assignments(table).get("A", "")

        left, right = convention_of(kshell), convention_of(levels)
        if left != right:
            differ = "the two energy tables assign different 'A' conventions"
            out.append(Issue("S006", "", profile, "", differ))
            continue
        rows = {tuple(record[:2]): record for record in levels.records}
        for record in kshell.records:
            energies = [record[2]] + list(rows[tuple(record[:2])][2 : 2 + n - 1])
            falling = all(upper > lower for upper, lower in zip(energies, energies[1:], strict=False))
            if not falling or not energies[-1] > 0:
                key = "-".join(str(int(part)) for part in record[:2])
                out.append(Issue("S006", "", profile, "", f"at key {key} the energies do not fall"))
                break
    return out


def semantic_issues(members: list[Loaded]) -> list[Issue]:
    """Every issue, in loaded-file order and then in bytewise profile order."""
    declared = (member.directives.get("VERSION", "") for member in members)
    version = next((value for value in declared if value), "")
    out: list[Issue] = []
    for member in members:
        out += _table_issues(member, version)
    return out + _pair_issues(members)


# T-112 -- the shipped dataset means what a consumer may read
# --------------------------------------------------------------------------------------------


def test_t112_the_shipped_dataset_raises_no_semantic_issue():
    members = load(*DATASET_DIRS)
    assert members, DATASET_DIRS
    issues = semantic_issues(members)
    assert issues == [], issues


# T-113 .. T-118 -- one drill per rule family
# --------------------------------------------------------------------------------------------


@pytest.fixture
def mutable(tmp_path: pathlib.Path) -> pathlib.Path:
    """A copy of every shipped member, as one directory, for a drill to edit."""
    for directory in DATASET_DIRS:
        for path in directory.iterdir():
            if path.name.endswith(".g4dat"):
                shutil.copy(path, tmp_path / path.name)
    return tmp_path


def kshell_member(directory: pathlib.Path) -> pathlib.Path:
    (path,) = [p for p in sorted(directory.iterdir()) if p.name.endswith(".g4dat") and "kshell" in p.name]
    return path


def zeff_member(directory: pathlib.Path) -> pathlib.Path:
    (path,) = [p for p in sorted(directory.iterdir()) if p.name.endswith(".g4dat") and "zeff" in p.name]
    return path


def d1_member(directory: pathlib.Path, table: str) -> pathlib.Path:
    (path,) = [p for p in sorted(directory.glob("*.g4dat"))
               if directive_line(p, "PROFILE").split()[-1] == "parity"
               and directive_line(p, "TABLE").split()[-1] == table]
    return path


def replace_once(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text(encoding="ascii")
    assert text.count(old) == 1, (path.name, old)
    path.write_text(text.replace(old, new), encoding="ascii", newline="")


def codes_of(directory: pathlib.Path) -> list[str]:
    return [issue.code for issue in semantic_issues(load(directory))]


def directive_line(path: pathlib.Path, keyword: str) -> str:
    lines = path.read_text(encoding="ascii").splitlines()
    (line,) = [text for text in lines if text.startswith(f"#{keyword} ")]
    return line


def first_record_line(path: pathlib.Path) -> str:
    lines = path.read_text(encoding="ascii").splitlines()
    return next(line for line in lines if not line.startswith("#") and line.strip())


def negative_value(line: str, index: int) -> str:
    matches = list(re.finditer(r"\S+", line))
    value = matches[index]
    assert float(value.group()) > 0
    return line[:value.start()] + "-" + line[value.start():]


def test_t113_a_foreign_dataset_name_raises_s001(mutable: pathlib.Path):
    member = kshell_member(mutable)
    line = directive_line(member, "DATASET")
    replace_once(member, line, line.replace(DATASET_NAME, "other_" + DATASET_NAME))
    assert codes_of(mutable)[0] == "S001"


def test_t114_a_table_in_the_wrong_seam_raises_s002(mutable: pathlib.Path):
    member = kshell_member(mutable)
    line = directive_line(member, "SEAM")
    replace_once(member, line, line.replace(D3_SEAM, D1_SEAM))
    assert codes_of(mutable)[0] == "S002"


@pytest.mark.parametrize("table", (CAPTURE, ZEFF))
def test_t114_drill_d1_member_in_the_wrong_seam_raises_s002(mutable: pathlib.Path, table: str):
    member = d1_member(mutable, table)
    own = directive_line(member, "SEAM")
    other = directive_line(kshell_member(mutable), "SEAM")
    replace_once(member, own, other)
    assert codes_of(mutable)[0] == "S002"


def test_t115_a_z_range_that_is_not_inclusive_raises_s003(mutable: pathlib.Path):
    member = kshell_member(mutable)
    line = directive_line(member, "VALIDITY")
    swapped = re.sub(r"Z:([0-9]+)-([0-9]+)", lambda m: f"Z:{m.group(2)}-{m.group(1)}", line)
    assert swapped != line, line
    replace_once(member, line, swapped)
    assert codes_of(mutable)[0] == "S003"


def test_t115_drill_capture_range_not_inclusive_raises_s003(mutable: pathlib.Path):
    member = d1_member(mutable, CAPTURE)
    line = directive_line(member, "VALIDITY")
    swapped = re.sub(r"Z:([0-9]+)-([0-9]+)", lambda m: f"Z:{m.group(2)}-{m.group(1)}", line)
    assert swapped != line
    replace_once(member, line, swapped)
    assert codes_of(mutable)[0] == "S003"


def test_t116_a_unit_the_lookup_does_not_read_raises_s004(mutable: pathlib.Path):
    member = kshell_member(mutable)
    line = directive_line(member, "UNITS")
    replace_once(member, line, line.replace(" value=", " value=x"))
    assert codes_of(mutable)[0] == "S004"


@pytest.mark.parametrize("table", (CAPTURE, ZEFF))
def test_t116_drill_d1_value_unit_the_lookup_does_not_read_raises_s004(mutable: pathlib.Path, table: str):
    member = d1_member(mutable, table)
    line = directive_line(member, "UNITS")
    replace_once(member, line, line.replace(" value=", " value=x", 1))
    assert codes_of(mutable)[0] == "S004"


def test_t117_a_negative_binding_energy_raises_s005(mutable: pathlib.Path):
    member = kshell_member(mutable)
    line = first_record_line(member)
    head, _, rest = line.rpartition(" ")
    keys, gap, value = head.rpartition(" ")
    assert value and gap, line
    replace_once(member, line, f"{keys}{gap[:-1]}-{value} {rest}")
    assert codes_of(mutable)[0] == "S005"


@pytest.mark.parametrize("table", (CAPTURE, ZEFF))
def test_t117_drill_d1_negative_value_raises_s005(mutable: pathlib.Path, table: str):
    member = d1_member(mutable, table)
    lines = member.read_text(encoding="ascii").splitlines()
    line = next(text for text in lines if not text.startswith("#") and text.strip()
                and (table != ZEFF or int(text.split()[0]) >= BOUNDS["kMinZ"]))
    index = 2 if table == CAPTURE else 1
    replace_once(member, line, negative_value(line, index))
    assert codes_of(mutable)[0] == "S005"


def test_t118_a_k_energy_below_its_next_level_raises_s006(mutable: pathlib.Path):
    member = kshell_member(mutable)
    line = first_record_line(member)
    head, _, unc = line.rpartition(" ")
    keys, gap, _value = head.rpartition(" ")
    replace_once(member, line, f"{keys}{gap}{unc} {unc}")
    codes = codes_of(mutable)
    assert codes and codes[0] == "S006", codes


def test_t120_a_z_outside_the_one_key_tables_declared_range_raises_s003(mutable: pathlib.Path):
    member = zeff_member(mutable)
    line = directive_line(member, "VALIDITY")
    narrowed = re.sub(r"Z:([0-9]+)-([0-9]+)", lambda m: f"Z:{m.group(1)}-{m.group(1)}", line)
    assert narrowed != line, line
    replace_once(member, line, narrowed)
    assert codes_of(mutable)[0] == "S003"


# T-119 -- the two statements of the rules name the same things
# --------------------------------------------------------------------------------------------


def test_t119_the_production_source_declares_the_same_codes_and_bounds():
    source = SOURCE.read_text(encoding="utf-8")
    assert set(re.findall(r'"(S00[0-9])"', source)) == set(CODES)
    for name, value in BOUNDS.items():
        match = re.search(rf"^const (?:long|std::size_t) {name} = ([0-9]+);", source, re.MULTILINE)
        assert match is not None, name
        assert int(match.group(1)) == value, (name, match.group(1))
    for token in (DATASET_NAME, CAPTURE, ZEFF, KSHELL, LEVELS, D1_SEAM, D3_SEAM, RATE_UNIT, ZEFF_UNIT,
                  ENERGY_UNIT, LISTED, NATURAL, REPRESENTATIVE):
        assert f'"{token}"' in source, token
