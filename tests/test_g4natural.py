"""The natural-target capture study: the harvested compositions, the lookup, the generated study and the
document's restatement of it.

`data/g4/d1/geant4_natural_isotopes.csv` is what `cpp/tools/harvest_natural.cc` printed on Geant4 builds, and
`data/g4/d1/natural_target_capture.csv` is generated from it by `scripts/generate_g4natural.py compare`. These
tests regenerate the study and compare its bytes, tie the harvest to the driver that printed it, drill each
refusal of the harvest and each outcome the study can return, and bind section 11 of `DATASET_D1.md` to the
study.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from openmucf.g4 import d1_natural as natural
from openmucf.g4 import spec
from openmucf.g4.sources import d1_nuclear_capture as d1
from openmucf.g4.sources import suzuki1987

REPO = Path(__file__).resolve().parents[1]
DOCUMENT = "DATASET_D1.md"
BOUND_DECAY = ("third_party/geant4/v11.4.2/G4MuonMinusBoundDecay.cc",
               "third_party/geant4/v11.5.0.beta/G4MuonMinusBoundDecay.cc")
HELPER = ("third_party/geant4/v11.4.2/G4MuonicAtomHelper.cc",
          "third_party/geant4/v11.5.0.beta/G4MuonicAtomHelper.cc")


def study() -> list[dict[str, str]]:
    """The committed study."""
    return natural._read_csv(REPO / natural.STUDY_RELPATH, natural.STUDY_COLUMNS)


def compared(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The `parity` rows that carry a natural-composition record."""
    return [row for row in rows if row["profile"] == spec.PARITY_PROFILE and row["reference_a"]]


def beyond_uncertainty(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows of :func:`compared` whose relative difference exceeds the record's tabulated relative
    uncertainty."""
    return [
        row for row in compared(rows)
        if abs(float(row["relative_difference"])) > float(row["reference_unc"]) / float(row["reference_rate"])
    ]


def found() -> d1.D1Extraction:
    return d1.load(REPO / d1.VENDORED_RELPATH)


# --------------------------------------------------------------------------------------------
# the harvest
# --------------------------------------------------------------------------------------------


def test_the_harvest_records_the_committed_driver_on_each_revision():
    """Each build record names a revision the patches serve, the version its `geant4-config` printed (the
    beta's prints its release number without the suffix) and the SHA-256 of the committed driver."""
    builds = natural.load_builds(REPO / natural.BUILDS_RELPATH)
    assert [(row["build"], row["revision"]) for row in builds] == list(natural.BUILDS)
    harness = hashlib.sha256((REPO / natural.HARNESS_RELPATH).read_bytes()).hexdigest()
    for row in builds:
        assert row["harness_sha256"] == harness, row["build"]
        assert row["geant4_version"] == row["revision"].removeprefix("v").removesuffix(".beta"), row
        assert row["compile_flags"] == natural.COMPILE_FLAGS


def test_the_compositions_cover_the_parity_validity_range_each_normalized():
    """The compositions cover exactly the Z range the `parity` capture table declares valid; each element's
    isotopes ascend in mass number, carry one symbol, and their abundances sum to one within one unit in the
    last place, as `G4Element` renormalizes them."""
    isotopes = natural.load_isotopes(REPO / natural.ISOTOPES_RELPATH)
    tables = natural.capture_profiles(REPO / "data" / "g4" / "d1")
    validity = tables[spec.PARITY_PROFILE].directives["VALIDITY"]
    span = re.match(r"Z:([0-9]+)-([0-9]+) ", validity)
    assert span is not None
    assert sorted(isotopes) == list(range(int(span[1]), int(span[2]) + 1))
    for z, members in isotopes.items():
        assert [i.a for i in members] == sorted({i.a for i in members}), z
        assert len({i.symbol for i in members}) == 1, z
        assert abs(math.fsum(i.abundance for i in members) - 1.0) <= math.ulp(1.0), z


def test_every_parity_rate_the_study_reads_is_a_point_of_the_oracle_sweep():
    """Every isotope of every composition lies in the box `d1_gp_sweep.oracle` pins, so the Python reference
    reproduces, bit for bit, the rate the compiled library returns for it (test_g4parity.py T-48)."""
    for z, members in natural.load_isotopes(REPO / natural.ISOTOPES_RELPATH).items():
        assert d1.SWEEP_Z_MIN <= z <= d1.SWEEP_Z_MAX
        for isotope in members:
            assert d1.SWEEP_A_MIN <= isotope.a <= d1.SWEEP_A_MAX, (z, isotope.a)


_RAW = "E\t1\tH\t7\t1\t2\nI\t1\t1\t0x1.fffp-1\nI\t1\t2\t0x1p-12\nE\t2\tHe\t8\t3\t1\nI\t2\t4\t0x1p+0\n"


@pytest.mark.parametrize(
    "text",
    [
        _RAW.replace("E\t2\t", "E\t3\t"),  # an element out of order
        "I\t1\t1\t0x1p+0\n" + _RAW,  # an isotope before its element
        _RAW.replace("0x1p-12", "0x0p+0"),  # an abundance that is not positive
        _RAW.replace("\t2\nI\t1\t1", "\t3\nI\t1\t1"),  # fewer isotope lines than the element line states
        _RAW + "the driver writes nothing else\n",  # a line the driver does not write
        # an element out of order whose isotope lines all follow their own element
        _RAW.replace("E\t2\t", "E\t3\t").replace("I\t2\t4", "I\t3\t4"),
        # an isotope line after a later element's line, the earlier element's count still met
        "E\t1\tH\t7\t1\t2\nI\t1\t1\t0x1.fffp-1\nE\t2\tHe\t8\t3\t1\nI\t2\t4\t0x1p+0\nI\t1\t2\t0x1p-12\n",
    ],
    ids=["order", "orphan", "zero", "count", "foreign", "gap", "late"],
)
def test_drill_the_harvest_parser_refuses_what_the_driver_does_not_print(text):
    with pytest.raises(natural.NaturalTargetError):
        natural.parse_harvest(text)


def _raw_tree(root: Path, outputs: tuple[str, str], digest: str, stderr: bytes = b"") -> Path:
    raw = root / "raw"
    for (build, _), output in zip(natural.BUILDS, outputs, strict=True):
        folder = raw / build
        folder.mkdir(parents=True)
        (folder / "natural.txt").write_bytes(output.encode("ascii"))
        (folder / "sha256.txt").write_text(digest + "\n", encoding="ascii")
        (folder / "version.txt").write_text("11.4.2\n", encoding="ascii")
        (folder / "gcc.txt").write_text("g++\n", encoding="ascii")
        (folder / "stderr.txt").write_bytes(stderr)
    return raw


def test_drill_the_harvest_conversion_refuses_differing_builds_a_foreign_driver_and_stderr(tmp_path):
    """The conversion writes rows only when every build compiled the committed driver, wrote nothing to
    stderr and printed the same bytes; each of the three ways to break that is refused."""
    harness = hashlib.sha256((REPO / natural.HARNESS_RELPATH).read_bytes()).hexdigest()
    rows, builds = natural.harvest(_raw_tree(tmp_path / "ok", (_RAW, _RAW), harness), REPO)
    assert [row["a"] for row in rows] == ["1", "2", "4"] and len(builds) == len(natural.BUILDS)
    assert rows[1]["abundance"] == repr(float.fromhex("0x1p-12"))
    for name, raw in (
        ("differ", _raw_tree(tmp_path / "differ", (_RAW, _RAW.replace("0x1p-12", "0x1p-11")), harness)),
        ("driver", _raw_tree(tmp_path / "driver", (_RAW, _RAW), "0" * 64)),
        ("stderr", _raw_tree(tmp_path / "stderr", (_RAW, _RAW), harness, b"warning\n")),
    ):
        with pytest.raises(natural.NaturalTargetError):
            natural.harvest(raw, REPO)
        assert name


# --------------------------------------------------------------------------------------------
# the lookup and the study
# --------------------------------------------------------------------------------------------


def test_the_committed_study_is_the_generated_one():
    """Regenerated from the committed compositions, the vendored source, the shipped capture profiles and the
    isotope audit, the study is byte for byte the committed file."""
    regenerated = natural.render_csv(natural.compute(REPO), natural.STUDY_COLUMNS)
    assert regenerated == (REPO / natural.STUDY_RELPATH).read_bytes()


def test_drill_the_study_refuses_a_natural_row_in_parity_and_a_composition_short_of_the_range(monkeypatch):
    """The study is computed only when the `parity` capture table carries no natural-composition row, whose
    presence would put a rung before the compiled-in chain, and only when the compositions cover the
    table's validity range."""
    tables = natural.capture_profiles(REPO / "data" / "g4" / "d1")
    parity = tables[spec.PARITY_PROFILE]
    planted = dict(tables)
    planted[spec.PARITY_PROFILE] = replace(parity, records=(*parity.records, (12, 0, 0.5, 0.01)))
    with monkeypatch.context() as patch:
        patch.setattr(natural, "capture_profiles", lambda directory: planted)
        with pytest.raises(natural.NaturalTargetError):
            natural.compute(REPO)
    isotopes = natural.load_isotopes(REPO / natural.ISOTOPES_RELPATH)
    short = {z: members for z, members in isotopes.items() if z != max(isotopes)}
    with monkeypatch.context() as patch:
        patch.setattr(natural, "load_isotopes", lambda path: short)
        with pytest.raises(natural.NaturalTargetError):
            natural.compute(REPO)


def test_the_lookup_takes_each_route():
    """A profile's row keyed by the nuclide answers first; without one, the compiled-in record answers its
    own key and the formula every other, exactly as the reference chain computes them."""
    extraction = found()
    model = natural.reference_model(extraction)
    records = extraction.capture_records
    z, a, value, _ = records[0]
    hit_table = natural.lookup(z, a, None, records, model)
    assert hit_table == natural.Route(natural.TABLE_ROUTE, value / d1.MICROSECOND)
    miss = natural.lookup(z, a + 1, None, records, model)
    assert miss == natural.Route(natural.FORMULA_ROUTE, model.rate(z, a + 1))
    hit = natural.lookup(z, a + 1, {(z, a + 1): 2.5}, records, model)
    assert hit == natural.Route(natural.PROFILE_ROUTE, 2.5 / d1.MICROSECOND)
    assert natural.lookup(z, a, {(z, a + 1): 2.5}, records, model).route == natural.TABLE_ROUTE


def test_a_profiles_natural_composition_row_answers_no_transported_nucleus():
    """A capture profile's `A = 0` row is not among the rows the study looks a nucleus up in, so an element a
    profile carries only as a natural-composition row gets the `parity` row's weighted rate."""
    tables = natural.capture_profiles(REPO / "data" / "g4" / "d1")
    rows = study()
    parity = {row["z"]: row for row in rows if row["profile"] == spec.PARITY_PROFILE}
    seen = 0
    for name, table in tables.items():
        reachable = natural.profile_rows(table)
        assert all(a != 0 for _, a in reachable)
        natural_only = {int(z) for z, a, _, _ in table.records if int(a) == 0} - {z for z, _ in reachable}
        for row in rows:
            if row["profile"] == name and int(row["z"]) in natural_only:
                assert row["weighted_rate"] == parity[row["z"]]["weighted_rate"], (name, row["z"])
                seen += 1
    assert seen, "no profile carries an element only as a natural-composition row; the drill tests nothing"


def test_drill_the_study_returns_no_difference_a_difference_and_a_nonpositive_share():
    """The study can return each outcome it reports: a composition entirely on the record gives no
    difference; one with a second isotope gives the formula's share and a difference; and a nuclide the
    formula rates below zero is counted in `nonpositive_share` and kept in the weighted rate, not clamped,
    under `parity` and under another profile alike."""
    extraction = found()
    model = natural.reference_model(extraction)
    z, a, value, _ = next(record for record in extraction.capture_records if record[0] == 1)
    profiles: dict[str, dict[tuple[int, int], float] | None] = {spec.PARITY_PROFILE: None, "drill": {}}

    def rows(members: list[tuple[int, float]]) -> list[dict[str, str]]:
        isotopes = {z: tuple(natural.Isotope(z, "H", mass, share) for mass, share in members)}
        return natural.study_rows(isotopes, profiles, extraction, model, {z: a})

    whole = rows([(a, 1.0)])[0]
    assert (float(whole["table_share"]), float(whole["relative_difference"])) == (1.0, 0.0)
    assert float(whole["weighted_rate"]) == value / d1.MICROSECOND * d1.MICROSECOND
    mixed = rows([(a, 0.5), (a + 1, 0.5)])[0]
    assert float(mixed["formula_share"]) == 0.5 and float(mixed["relative_difference"]) != 0.0
    negative_a = a + 2
    assert model.rate(z, negative_a) < 0.0, "the drill needs a nuclide the formula rates below zero"
    for row in rows([(a, 0.5), (negative_a, 0.5)]):
        assert float(row["nonpositive_share"]) == 0.5, row["profile"]
        expected = math.fsum((0.5 * value / d1.MICROSECOND, 0.5 * model.rate(z, negative_a))) * d1.MICROSECOND
        assert float(row["weighted_rate"]) == expected


def test_no_composition_reaches_a_rate_that_is_not_positive():
    """No row of the study has a positive `nonpositive_share`: no isotope of a composition the harvest
    recorded gets a capture rate that is not a positive finite rate, under any profile."""
    rows = study()
    assert rows
    assert all(float(row["nonpositive_share"]) == 0.0 for row in rows)


def test_the_route_shares_partition_every_composition():
    rows = study()
    for row in rows:
        shares = [float(row[key]) for key in ("profile_share", "table_share", "formula_share")]
        assert abs(math.fsum(shares) - 1.0) <= 4 * math.ulp(1.0), row
        if row["profile"] == spec.PARITY_PROFILE:
            assert float(row["profile_share"]) == 0.0


# --------------------------------------------------------------------------------------------
# the decay rate, which the document says depends on the mass number only below a threshold
# --------------------------------------------------------------------------------------------


def _code(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def decay_threshold() -> int:
    """The Z below which `G4MuonMinusBoundDecay::GetMuonDecayRate(Z, A, muMass, nuclMass)` reads the nuclear
    mass, the same in both vendored revisions: the branch below it reads the muon and nuclear masses, the
    branch above it reads neither and no mass number."""
    thresholds = set()
    for relpath in BOUND_DECAY:
        text = _code((REPO / relpath).read_text(encoding="ascii"))
        body = text.split("G4MuonMinusBoundDecay::GetMuonDecayRate(G4int Z, G4int A", 1)[1]
        branch = re.search(r"if\s*\(\s*Z\s*<\s*([0-9]+)\s*\)\s*\{(.*?)\}\s*else\s*\{(.*?)\}", body, re.S)
        assert branch is not None, relpath
        assert "muMass" in branch[2] and "nuclMass" in branch[2], relpath
        assert not re.search(r"\b(?:A|muMass|nuclMass)\b", branch[3]), relpath
        thresholds.add(int(branch[1]))
        call = r"GetMuonDecayRate\(Z,\s*A,\s*fMuMass,\s*targetNucleus\.AtomicMass\(A,\s*Z\)\)"
        assert re.search(call, text), relpath
    assert len(thresholds) == 1
    return thresholds.pop()


def test_the_decay_rate_depends_on_the_mass_number_only_below_the_threshold():
    """The BoundDecay route passes the sampled nucleus's mass to a decay rate that reads it only below the
    threshold; the helper route's decay rate takes Z alone."""
    assert decay_threshold() > 1
    for relpath in HELPER:
        text = _code((REPO / relpath).read_text(encoding="ascii"))
        assert re.search(r"G4double G4MuonicAtomHelper::GetMuonDecayRate\(G4int Z\)\s*\{", text), relpath
        assert re.search(r"G4double lambdad = GetMuonDecayRate\(Z\);", text), relpath


# --------------------------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------------------------


def _row_pattern(line: str, numeric: re.Pattern[str]) -> tuple[str, tuple[int, ...]]:
    """A pattern matching ``line`` whitespace-collapsed, with every numeric token a group."""
    collapsed = " ".join(line.split())
    parts, groups, at = [], [], 0
    for index, match in enumerate(numeric.finditer(collapsed), start=1):
        parts.append(re.escape(collapsed[at:match.start()]))
        parts.append(f"({re.escape(match.group())})")
        groups.append(index)
        at = match.end()
    parts.append(re.escape(collapsed[at:]))
    return "".join(parts), tuple(groups)


def document_pins(numeric: re.Pattern[str]) -> list[tuple[str, str, str, tuple[int, ...], object]]:
    """``(what, path, pattern, groups, expected)`` for every number section 11 of `DATASET_D1.md` states: one
    row per line of its two tables, every numeric token a group, and one per count of the prose, ``expected``
    computed from the committed study or the vendored source. ``numeric`` is the prose check's tokenizer."""
    rows = study()
    pins: list[tuple[str, str, str, tuple[int, ...], object]] = [
        ("elements with a natural-composition record", DOCUMENT,
         r"Of the (\d+) elements with a natural-composition record", (1,), len(compared(rows))),
        ("rows beyond the tabulated relative uncertainty", DOCUMENT,
         r"tabulated relative uncertainty in (\d+) of them", (1,), len(beyond_uncertainty(rows))),
        ("the decay-rate threshold", DOCUMENT, r"wherever Z ≥ (\d+)", (1,), decay_threshold()),
    ]
    reach = natural.render_reach_sentence(natural.capture_profiles(REPO / "data" / "g4" / "d1"))
    pattern, groups = _row_pattern(reach, numeric)
    pins.append(("reachable rows", DOCUMENT, pattern, groups, None))
    for name, block in (("natural-target table", natural.render_table(rows)),
                        ("profile table", natural.render_profile_table(rows))):
        for number, line in enumerate(block.splitlines()[2:], start=1):
            pattern, groups = _row_pattern(line, numeric)
            pins.append((f"{name} row {number}", DOCUMENT, pattern, groups, None))
    return pins


_NUMERIC = re.compile(r"[0-9][0-9,.]*[0-9]|[0-9]|[⁰¹²³⁴⁵⁶⁷⁸⁹]+")


def test_the_document_carries_both_tables_as_generated():
    """Section 11 of `DATASET_D1.md` carries each table exactly as the study renders it."""
    text = (REPO / DOCUMENT).read_text(encoding="utf-8")
    section = text.split("\n## 11. ", 1)[1]
    rows = study()
    for block in (natural.render_table(rows), natural.render_profile_table(rows)):
        assert section.count(block + "\n") == 1


def test_the_document_states_the_reachable_rows_of_each_profile_once():
    """Section 11 carries the sentence `render_reach_sentence` renders, once; each count it states is the
    number of the profile's records keyed by a mass number, and a count moved by one is no longer the
    rendered sentence."""
    tables = natural.capture_profiles(REPO / "data" / "g4" / "d1")
    sentence = natural.render_reach_sentence(tables)
    collapsed = " ".join((REPO / DOCUMENT).read_text(encoding="utf-8").split())
    section = collapsed.split(" ## 11. ", 1)[1].split(" ## 12. ", 1)[0]
    assert section.count(sentence) == 1 and collapsed.count(sentence) == 1
    for name, table in tables.items():
        keyed = sum(1 for _, a, _, _ in table.records if int(a) != 0)
        assert f"{keyed} of the {len(table.records)} rows of `{name}`" in sentence
    count = len(natural.profile_rows(tables["parity"]))
    moved = sentence.replace(f"{count} of the", f"{count + 1} of the", 1)
    assert moved != sentence and collapsed.replace(sentence, moved).count(sentence) == 0


def test_every_pin_of_the_section_states_its_value():
    collapsed = " ".join((REPO / DOCUMENT).read_text(encoding="utf-8").split())
    for what, _, pattern, _, expected in document_pins(_NUMERIC):
        hits = re.findall(pattern, collapsed)
        assert len(hits) == 1, what
        if expected is not None:
            assert int(hits[0]) == expected, what


def test_drill_a_changed_cell_or_count_is_refused():
    """A table cell moved by a tenth and a count moved by one each leave their pin without a match."""
    collapsed = " ".join((REPO / DOCUMENT).read_text(encoding="utf-8").split())
    pins = document_pins(_NUMERIC)
    row = next(pin for pin in pins if pin[0] == "natural-target table row 1")
    line = " ".join(natural.render_table(study()).splitlines()[2].split())
    head, last = line.rsplit(" | ", 1)
    assert collapsed.count(line) == 1 and re.search(row[2], collapsed)
    assert not re.search(row[2], collapsed.replace(line, f"{head} | 9{last}"))
    count = pins[0]
    moved = re.sub(count[2], lambda m: m[0].replace(m[1], str(int(m[1]) + 1)), collapsed)
    assert int(re.findall(count[2], moved)[0]) != count[4]


#: The words of section 11 that say what the `parity` lead record is measured against.
LEAD_WORDS = (
    "the inverse-variance weighted mean of the row of lead printed with no mass number and the row of what a "
    "footnote of that table labels radiogenic lead",
    "matches a reconstructed combination of a measurement on lead and a measurement on radiogenic lead",
)


def lead_rows() -> tuple[dict[str, str], dict[str, str]]:
    """The rows of Table IV at the lead record's Z that share a reference with the row the table labels with
    an `R` prefix: the row on lead printed with no mass number, then that row."""
    rows = suzuki1987.normalize(suzuki1987.load_printed_rows(REPO / suzuki1987.PRINTED_ROWS_RELPATH))
    (z,) = {int(row["z"]) for row in compared(study()) if row["symbol"] == "Pb"}
    (radiogenic,) = [row for row in rows if int(row["Z"]) == z and row["basis_rule"] == "R-prefix"]
    (plain,) = [row for row in rows if int(row["Z"]) == z and row["refs"] == radiogenic["refs"]
                and row is not radiogenic]
    assert plain["target_basis"] == "unspecified" and plain["target_label"] == "Pb", plain
    return plain, radiogenic


def lead_problems(pair: tuple[dict[str, str], dict[str, str]]) -> list[str]:
    """What would make the section's statement false: the pair's inverse-variance weighted mean and its
    uncertainty, at the record's printed precision, are not the `parity` lead record."""
    import decimal

    data = found()
    (index,) = [i for i, record in enumerate(data.capture_records) if record[0] == int(pair[0]["Z"])]
    value, unc = (decimal.Decimal(cell) for cell in data.capture_literals[index])
    weights = [1 / suzuki1987.scaled(row, "rate_unc") ** 2 for row in pair]
    total = sum(w * suzuki1987.scaled(row, "rate") for w, row in zip(weights, pair, strict=True))
    mean = total / sum(weights)
    sigma = 1 / sum(weights).sqrt()
    if (mean.quantize(value), sigma.quantize(unc)) != (value, unc):
        return [f"the weighted mean is {mean} +- {sigma}; the record is {value} +- {unc}"]
    return []


def test_the_lead_record_equals_the_weighted_mean_of_a_lead_row_and_a_radiogenic_lead_row():
    """Section 11 says, after its table, that the `parity` lead record is the inverse-variance weighted mean,
    at its printed precision, of a row of Table IV on lead printed with no mass number and a row the table's
    footnote labels radiogenic lead, from the same reference; recomputed here from the committed transcription
    and the vendored record."""
    plain, radiogenic = lead_rows()
    assert lead_problems((plain, radiogenic)) == []
    printed = suzuki1987.load_printed_rows(REPO / suzuki1987.PRINTED_ROWS_RELPATH)
    marker = radiogenic["target_label"].rpartition("^")[2]
    (footnote,) = [row.note for row in printed
                   if row.row_kind == "footnote" and row.note.startswith(f"^{marker}R")]
    assert "radiogenic lead" in footnote, footnote
    text = (REPO / DOCUMENT).read_text(encoding="utf-8")
    section = " ".join(text.split("\n## 11. ", 1)[1].split("\n## 12. ", 1)[0].split())
    table = " ".join(natural.render_table(study()).split())
    for words in LEAD_WORDS:
        assert section.count(words) == 1 and section.index(words) > section.index(table), words


def test_drill_another_lead_row_in_place_of_the_radiogenic_row_is_refused():
    """Taking the lead row of another reference in place of the radiogenic-lead row moves the weighted mean
    off the record."""
    plain, radiogenic = lead_rows()
    rows = suzuki1987.normalize(suzuki1987.load_printed_rows(REPO / suzuki1987.PRINTED_ROWS_RELPATH))
    others = [row for row in rows if row["Z"] == plain["Z"] and row["refs"] != plain["refs"]
              and row["target_basis"] == "unspecified" and row["rate_unc"]]
    assert others
    for other in others:
        assert lead_problems((plain, other)), other["refs"]


#: The two records the isotope audit leaves open because each value agrees with a combination of printed rows:
#: the Table IV labels of the rows its evidence names, the element's row being the one of this experiment.
OPEN_COMBINATIONS = {(29, 63): ("Cu", "63Cu"), (35, 79): ("79Br", "81Br")}


def combination_rows(z: int, labels: tuple[str, ...]) -> list[dict[str, str]]:
    """The Table IV row of each label at ``z`` that prints a rate and its uncertainty; where a label
    prints more than one, the row of this experiment."""
    rows = suzuki1987.normalize(suzuki1987.load_printed_rows(REPO / suzuki1987.PRINTED_ROWS_RELPATH))
    out = []
    for label in labels:
        hits = [row for row in rows if row["table"] == "IV" and int(row["Z"]) == z
                and row["target_label"] == label and row["rate_unc"]]
        if len(hits) > 1:
            hits = [row for row in hits if row["own_measurement"] == "true"]
        (row,) = hits
        out.append(row)
    return out


def combination_problems(key: tuple[int, int], rows: list[dict[str, str]]) -> list[str]:
    """What would make the evidence false: the rows' inverse-variance weighted mean or its uncertainty differs
    from the record by a unit of the record's last printed digit or more."""
    import decimal

    data = found()
    (index,) = [i for i, record in enumerate(data.capture_records) if (record[0], record[1]) == key]
    value, unc = (decimal.Decimal(cell) for cell in data.capture_literals[index])
    weights = [1 / suzuki1987.scaled(row, "rate_unc") ** 2 for row in rows]
    total = sum(w * suzuki1987.scaled(row, "rate") for w, row in zip(weights, rows, strict=True))
    mean = total / sum(weights)
    sigma = 1 / sum(weights).sqrt()
    unit_v = decimal.Decimal(1).scaleb(value.as_tuple().exponent)
    unit_u = decimal.Decimal(1).scaleb(unc.as_tuple().exponent)
    if abs(mean - value) >= unit_v or abs(sigma - unc) >= unit_u:
        return [f"{key}: the weighted mean is {mean} +- {sigma}; the record is {value} +- {unc}"]
    return []


def test_the_two_open_records_agree_with_the_weighted_means_their_evidence_names():
    """The `(29, 63)` and `(35, 79)` records are open, not isotope-resolved, and name their value a
    reconstructed combination; each agrees, within a unit of its last printed digit, with the inverse-variance
    weighted mean of the two Table IV rows its evidence names. The lead record, settled and not
    isotope-resolved, names its value a reconstructed combination too."""
    audit = d1.load_isotope_audit(REPO / d1.AUDIT_RELPATH)
    for key, labels in OPEN_COMBINATIONS.items():
        finding = audit[key]
        assert not finding.isotope_resolved and not finding.settled, key
        assert d1.RECONSTRUCTED_CLAUSE in finding.evidence, key
        assert combination_problems(key, combination_rows(key[0], labels)) == [], key
    plain, _ = lead_rows()
    lead = audit[(int(plain["Z"]), 207)]
    assert lead.settled and not lead.isotope_resolved and d1.RECONSTRUCTED_CLAUSE in lead.evidence


def test_drill_a_copper_row_of_another_reference_in_the_combination_is_refused():
    """Each Table IV row of Cu printed with no mass number from another reference, taken in place of the
    row of this experiment, moves the weighted mean off the `(29, 63)` record."""
    rows = suzuki1987.normalize(suzuki1987.load_printed_rows(REPO / suzuki1987.PRINTED_ROWS_RELPATH))
    others = [row for row in rows if row["table"] == "IV" and row["Z"] == "29" and row["target_label"] == "Cu"
              and row["own_measurement"] != "true" and row["rate_unc"]]
    assert others
    (isotope,) = combination_rows(29, ("63Cu",))
    for other in others:
        assert combination_problems((29, 63), [other, isotope]), other["refs"]
