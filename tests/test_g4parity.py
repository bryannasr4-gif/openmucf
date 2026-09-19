"""D1 in parity mode: the vendored upstream, the extraction, and the bit-parity proof.

``tests/test_g4spec.py`` tests the *format*. This file tests the one **dataset** that claims to
reproduce something: `data/g4/d1/`, which asserts that every muon-capture record and every effective
charge its `parity` tables ship is bit-for-bit what Geant4 v11.4.2 compiles in, and that the
Goulard-Primakoff fallback it declares evaluates to the same doubles the compiled library returns.

Three disciplines run through every test here, because the claim is only as good as they are:

* **no count is written down.** Every record count is ``len()`` of something parsed out of
  ``third_party/geant4/v11.4.2/G4MuonMinusBoundDecay.cc``; T-42 walks the extractor's AST and fails
  if a literal count appears in it. A test asserting a literal would re-create the bug that put
  "94 entries" in an earlier design document -- the number was the *maximum Z*, not the record count.
* **nothing is compared against itself.** The parity tests compare two independently derived things:
  the vendored upstream source on one side, the generated dataset on the other. A test that read the
  count out of the generated file and then checked the generated file against it would pass forever
  and prove nothing.
* **the oracle is harvested, not regenerated.** ``data/g4/d1/d1_gp_sweep.oracle`` came out of a
  Geant4-linked binary. No Python in this repository can produce it, which is exactly why comparing
  the Python reference implementation against it is evidence rather than a tautology.
"""

import ast
import dataclasses
import decimal
import hashlib
import importlib.util
import io
import json
import math
import pathlib
import re
import struct
import subprocess
import sys
import tarfile
from collections.abc import Sequence

import pytest

import openmucf
from openmucf import rates
from openmucf.g4 import emit, provenance, sources, spec
from openmucf.g4.sources import d1_nuclear_capture as d1
from openmucf.g4.sources import mizuno2025

REPO = pathlib.Path(__file__).resolve().parents[1]
VENDORED = REPO / "third_party" / "geant4" / "v11.4.2" / "G4MuonMinusBoundDecay.cc"
#: The second compiled-in copy of the same two tables, vendored beside the first.
HELPER = REPO / "third_party" / "geant4" / "v11.4.2" / "G4MuonicAtomHelper.cc"
VENDORED_README = REPO / "third_party" / "geant4" / "README.md"
D1DIR = REPO / "data" / "g4" / "d1"
ORACLE = D1DIR / "d1_gp_sweep.oracle"

#: Upstream's own object name for the vendored bytes, at tag v11.4.2
#: (commit 8cc04f65977807f1848da7b958c421cd5e162f26). This is a *pin*, not a measurement: it is the
#: pre-registered identity of the file the whole parity chain is derived from, and it is verifiable
#: against github.com/Geant4/geant4 by anyone, with no Geant4 checkout and no `git` binary.
UPSTREAM_BLOB_ID = "29bd73719cd619de34ef83ca5ca076ceadf1cc5a"
UPSTREAM_SHA256 = "860dcdb53167c6437484b12c05ac1ab2eae4a6a52886af83fcf4394611882813"

#: The same two files at the later tag, vendored beside the v11.4.2
#: copies as evidence and never as a source: nothing D1 ships is generated from them. Each pin is
#: upstream's own object name for the bytes at that tag's commit, verifiable the same way.
BETA_TAG = "v11.5.0.beta"
BETA_COMMIT = "f3d5293d384757b8a228a099898b2b87cfa4023c"
BETA_DIR = REPO / "third_party" / "geant4" / BETA_TAG
BETA_BOUND_DECAY = BETA_DIR / VENDORED.name
BETA_HELPER = BETA_DIR / HELPER.name
BETA_BLOB_IDS = {
    VENDORED.name: "ff95c000f2cc3f6cfd6b835bade304e05af9feb5",
    HELPER.name: "8c2c37a99cdb3effd3ce7f1898488ad75f82b3fd",
}
BETA_SHA256S = {
    VENDORED.name: "bb925829e0acaa7fa3efd4560d954dd2f58f344fd155cdb3ce2554bd77539288",
    HELPER.name: "d020924b759ad1149cf74e2955eeffede84bc55c8be4ffb364385cc803720e2a",
}


def git_blob_id(data: bytes) -> str:
    """Git's object name for ``data`` as a blob: ``sha1("blob <len>\\0" + data)``.

    Three lines of `hashlib` rather than a `git` call, deliberately: this must work in an unpacked
    sdist, in a container with no git, and for a reader who is checking our work against upstream
    without cloning Geant4.
    """
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()

def _ulp_distance(x: float, y: float) -> int:
    """How many representable doubles lie between `x` and `y`: the binary64 bit patterns, with the
    negative half reflected so they order as numbers, differenced. `-0.0` and `0.0` are nowhere
    apart, and a negative rate sits the right distance from its neighbours."""

    def ordered(v: float) -> int:
        bits = struct.unpack(">q", struct.pack(">d", v))[0]
        return bits if bits >= 0 else -0x8000000000000000 - bits

    return abs(ordered(x) - ordered(y))


def subset_max_ulp(found: d1.D1Extraction, model: d1.GoulardPrimakoff, oracle: dict) -> int:
    """The largest ulp distance between the reference evaluation and the compiled oracle over the
    oracle's diagnostic subset -- the loop T-49 runs, reduced to the figure the documents state."""
    assert oracle["subset"], "the oracle carries no diagnostic subset"
    return max(
        _ulp_distance(d1.capture_rate(z, a, found.capture_records, model), expected)
        for (z, a), expected in oracle["subset"].items()
    )


# --------------------------------------------------------------------------------------------
# T-40..T-41 -- the vendored upstream is the pinned upstream, and its bytes survived the checkout
# --------------------------------------------------------------------------------------------


def test_t40_vendored_source_matches_the_upstream_pins():
    """The vendored file is upstream's file, proven by upstream's own object name.

    The blob id is the load-bearing pin: it is what `github.com/Geant4/geant4` calls these bytes, so
    a third party can verify this copy without trusting us and without installing anything. The
    sha256 is recorded alongside because SHA-1 is a provenance pin here and not a security control --
    a distinction worth stating in the test rather than defending later.
    """
    data = VENDORED.read_bytes()
    assert git_blob_id(data) == UPSTREAM_BLOB_ID, (
        "the vendored source is not the pinned upstream blob; if this is a deliberate re-pin it "
        "belongs in a NEW third_party/geant4/<tag>/ directory, never as an overwrite -- overwriting "
        "destroys the evidence that the previously published dataset was faithful to the version it "
        "claimed"
    )
    assert hashlib.sha256(data).hexdigest() == UPSTREAM_SHA256


def test_t41_vendored_source_has_no_carriage_returns():
    """`.gitattributes` marks `third_party/geant4/** -text`, and that line is load-bearing.

    The file's identity IS its bytes, and this repository is developed on a checkout with
    `core.autocrlf=true`. Without the attribute, a Windows clone rewrites every LF to CRLF, the blob
    id and the sha256 both stop matching, and T-40 fails with a hash mismatch that names no cause.
    Asserting the byte directly is what turns that into a message a maintainer can act on.
    """
    data = VENDORED.read_bytes()
    assert b"\r" not in data, (
        "the checkout rewrote the vendored source's line endings: check that .gitattributes still "
        "carries `third_party/geant4/** -text`"
    )
    # The `**` form is required, not decoration: a gitattributes `*` does not cross a `/`, so a
    # `third_party/geant4/*` line would leave the versioned subdirectory -- the file that matters --
    # unprotected. Pinned here because the failure it prevents is invisible on Linux.
    attributes = (REPO / ".gitattributes").read_text("utf-8")
    assert "third_party/geant4/** -text" in attributes
    assert "data/g4/d1/* -text" in attributes


# --------------------------------------------------------------------------------------------
# T-80 -- the vendored README's pins are the values computed from the vendored bytes
# --------------------------------------------------------------------------------------------


def test_t80_the_vendored_readme_pins_are_computed_from_the_vendored_bytes():
    """Every pin cell of `third_party/geant4/README.md`, read from its table by row label, equals
    the value computed from the vendored file it describes: the BoundDecay blob id, sha256 and
    size, the helper's size, the blob id the fenced example prints as its own comment, and, in the
    second table, the beta tag's commit and each beta copy's blob id, sha256 and size. T-40 and
    T-69 hold the module's pins to the bytes; this test holds the README's copies and the
    example's comment to the same bytes. The size cell is compared the way T-69 already compares
    the helper's: bytes, and newline count as the line count.
    """
    readme = VENDORED_README.read_text("utf-8")

    def cell(label: str) -> str:
        hits = re.findall(rf"^\| {label} \| `?([^`|]+?)`? \|$", readme, re.M)
        assert len(hits) == 1, f"row {label!r}: {hits}"
        return hits[0]

    assert cell(re.escape(f"`{BETA_TAG}` commit")) == BETA_COMMIT, (
        "the beta `commit` cell is not the pinned beta commit"
    )
    for name, beta_path in ((VENDORED.name, BETA_BOUND_DECAY), (HELPER.name, BETA_HELPER)):
        beta = beta_path.read_bytes()
        beta_lines = beta.count(b"\n")
        label = re.escape(f"`{BETA_TAG}/{name}`")
        assert cell(label + r" \*\*git blob id\*\*") == git_blob_id(beta), (
            f"the beta {name} `git blob id` cell is not the blob id of the vendored beta bytes"
        )
        assert cell(label + " sha256") == hashlib.sha256(beta).hexdigest(), (
            f"the beta {name} `sha256` cell is not the sha256 of the vendored beta bytes"
        )
        assert cell(label + " size") == f"{len(beta)} bytes, {beta_lines} lines", (
            f"the beta {name} `size` cell is not the vendored beta file's byte and newline count"
        )

    data = VENDORED.read_bytes()
    assert cell(r"\*\*git blob id\*\*") == d1.UPSTREAM_BLOB_ID, (
        "the BoundDecay `git blob id` cell is not the blob id of the vendored bytes"
    )
    assert cell("sha256") == d1.UPSTREAM_SHA256, (
        "the BoundDecay `sha256` cell is not the sha256 of the vendored bytes"
    )
    line_count = data.count(b"\n")
    assert cell("size") == f"{len(data)} bytes, {line_count} lines", (
        "the BoundDecay `size` cell is not the vendored file's byte and newline count"
    )
    helper = HELPER.read_bytes()
    helper_lines = helper.count(b"\n")
    assert cell(r"`G4MuonicAtomHelper\.cc` size") == f"{len(helper)} bytes, {helper_lines} lines", (
        "the helper `size` cell is not the vendored helper's byte and newline count"
    )
    comments = re.findall(r"^# ([0-9a-f]{40})$", readme, re.M)
    assert comments == [d1.UPSTREAM_BLOB_ID], (
        f"the fenced example's `# <hex>` comment line is not the vendored blob id: {comments}"
    )


# --------------------------------------------------------------------------------------------
# T-42, T-50, T-51 -- the extraction: derived counts, verbatim coefficients, a live directive
# --------------------------------------------------------------------------------------------


def extraction() -> d1.D1Extraction:
    """The one extraction under test, always straight from the vendored source."""
    return d1.load(VENDORED)


def independently_counted_records(
    text: str, declaration: str = "static const capRate capRates"
) -> int:
    """Count `{...}` groups in the `capRates[]` body by brace depth alone -- no record pattern.

    Deliberately a *different* method from the extractor's: the extractor matches records with a
    regex and proves completeness by residue, this walks the body character by character and counts
    depth transitions. If a regex ever stopped early, these two would disagree, which is the whole
    point of not reusing the extractor's own machinery here.

    `declaration` is the text the scan anchors on, because the two compiled-in copies spell the same
    array two ways (`static const` and `constexpr`); the counting method itself is identical, which
    is what makes the two counts comparable.
    """
    start = text.index(declaration)
    opening = text.index("{", text.index("=", start))
    depth, count, index = 0, 0, opening
    while index < len(text):
        if text[index] == "{":
            depth += 1
            if depth == 2:
                count += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return count
        index += 1
    raise AssertionError("the capRates[] initialiser never closes")


def test_t42_every_count_is_derived_from_the_vendored_source():
    """No count is written down anywhere in the extraction chain, and the parse is not short.

    Two independent halves. First, the extractor's record count is checked against a brace-depth
    recount of the same file -- if the record regex ever matched a subset, the two disagree here
    rather than agreeing forever on a short list. Second, an AST walk of the extraction module
    forbids the three counts (90 records, 101 effective charges, 74 distinct Z) from appearing as
    integer literals in it at all.

    The second half is a source-level rule in the T-34/T-39 family, and it uses the AST rather than
    a text grep on purpose: `zmax=100` and `maxZ` are legitimately present in that file, and a grep
    would either flag them or be loosened until it caught nothing.
    """
    found = extraction()
    text = VENDORED.read_text("ascii")

    assert len(found.capture_records) == independently_counted_records(text)
    assert len(found.capture_literals) == len(found.capture_records)
    assert len(found.capture_lines) == len(found.capture_records)
    assert len(found.zeff) == len(found.zeff_literals) == len(found.zeff_lines)
    # The zeff table is indexed by Z after clamping to [1, maxZ], so it must hold maxZ + 1 entries;
    # both sides of this come from the parse, neither is a number anyone chose.
    assert len(found.zeff) == found.zeff_max_z + 1

    # And what SHIPPED carries those counts too -- checked against the source, never against itself.
    # This is the direction that matters: a count taken from the generated file and then compared to
    # the generated file is a tautology wearing a derivation's clothes.
    for layer1_path, layer2_path, expected in (
        (D1DIR / "d1_capture.g4dat", D1DIR / "d1_capture.prov.json", len(found.capture_records)),
        (D1DIR / "d1_zeff.g4dat", D1DIR / "d1_zeff.prov.json", len(found.zeff)),
    ):
        table = spec.parse(layer1_path.read_bytes().decode("ascii"))
        document = provenance.from_json_obj(json.loads(layer2_path.read_bytes().decode("ascii")))
        assert len(table.records) == expected, layer1_path.name
        assert len(document.rows) == expected, layer2_path.name

    module = pathlib.Path(d1.__file__)
    banned = {
        len(found.capture_records),
        len(found.zeff),
        len(found.distinct_capture_z),
    }
    tree = ast.parse(module.read_text("utf-8"), filename=str(module))
    offenders = [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value in banned
    ]
    assert not offenders, (
        f"{module.name} hard-codes a count that must be derived from the pinned source: {offenders}. "
        "A test or a module asserting a literal re-creates the bug that recorded this table's "
        "record count as its maximum Z."
    )


# --------------------------------------------------------------------------------------------
# T-69, T-70 -- the two compiled-in copies: identical tables, one deliberately different clamp
# --------------------------------------------------------------------------------------------


def compare_copies(a: d1.D1Extraction, b: d1.D1Extraction) -> list[str]:
    """Every way two extractions of the same compiled-in tables disagree, named one at a time.

    A boolean would say "the copies differ" and leave a maintainer to find where. The thing this
    comparison exists to catch -- one of Geant4's two copies of the table updated and the other not
    -- shows up as a handful of rows out of hundreds, so every message names the record's `(Z, A)`
    or the array index rather than reporting a count.

    The clamp coefficient is deliberately *not* excluded here: it is a real difference between
    the two copies, and a comparison that swallowed it would be asserting something false. The
    caller states which coefficients it expects to differ.
    """
    # `strict=False` throughout: a length difference is REPORTED above rather than raised, so
    # the caller gets every disagreeing row and the count, not one exception on the first.
    problems: list[str] = []

    if len(a.capture_records) != len(b.capture_records):
        problems.append(
            f"capture record count: {len(a.capture_records)} vs {len(b.capture_records)}"
        )
    for index, (left, right) in enumerate(zip(a.capture_records, b.capture_records, strict=False)):
        if left != right:
            problems.append(f"capture record {index} (Z, A)={left[:2]}: {left} vs {right}")
    literals = zip(a.capture_literals, b.capture_literals, strict=False)
    for index, (left_text, right_text) in enumerate(literals):
        if left_text != right_text:
            key = a.capture_records[index][:2]
            problems.append(f"capture literal {index} (Z, A)={key}: {left_text} vs {right_text}")

    if len(a.zeff) != len(b.zeff):
        problems.append(f"effective-charge count: {len(a.zeff)} vs {len(b.zeff)}")
    for index, (left_value, right_value) in enumerate(zip(a.zeff, b.zeff, strict=False)):
        if left_value != right_value:
            problems.append(f"zeff[{index}]: {left_value} vs {right_value}")
    for index, (left_text, right_text) in enumerate(zip(a.zeff_literals, b.zeff_literals, strict=False)):
        if left_text != right_text:
            problems.append(f"zeff literal [{index}]: {left_text!r} vs {right_text!r}")

    if a.zeff_max_z != b.zeff_max_z:
        problems.append(f"maxZ: {a.zeff_max_z} vs {b.zeff_max_z}")

    return problems


def test_t69_the_two_compiled_in_copies_hold_the_same_tables_and_differ_in_the_clamp():
    """Geant4 compiles the capture and effective-charge tables in twice; here is what that costs.

    `G4MuonMinusBoundDecay.cc` and `G4MuonicAtomHelper.cc` carry the same `capRates[]` and the same
    `zeff[]`, written in two dialects. This test reads both with the same extractor
    -- parameterised only by declaration shape -- and requires them to agree element for element.
    That is what makes the dataset's provenance claim about "Geant4's compiled-in table" well
    defined: there are two copies, and they hold the same data.

    Where they do *not* agree is the clamp applied before indexing `zeff`, and that difference is
    pinned here rather than smoothed over: BoundDecay clamps `Z` into `[1, maxZ]`, the helper into
    `[0, maxZ]`, so for `Z <= 0` one returns `zeff[1]` and the other `zeff[0]`. The shipped dataset
    reproduces BoundDecay's behaviour; a reader is entitled to know the other copy exists and does
    something else at the edge.
    """
    helper = d1.load_helper(HELPER)
    bd = extraction()

    problems = compare_copies(helper, bd)
    assert not problems, (
        "the two compiled-in copies of the capture/effective-charge tables no longer agree; "
        "one of them was updated upstream and the other was not: " + "; ".join(problems)
    )
    # The comparison is only worth anything if it walked a non-empty table.
    assert helper.capture_records and helper.zeff

    # Every fallback coefficient is the same source text in both copies -- except the clamp. Both
    # sides are read from the sources.
    for name in d1.FALLBACK_NAMES:
        if name == "zmin":
            continue
        assert helper.coefficients[name] == bd.coefficients[name], name
    assert helper.coefficients["zmin"] != bd.coefficients["zmin"], (
        "the two copies now clamp Z the same way"
    )

    # And the clamp difference as source text.
    helper_text = re.sub(r"\s+", " ", HELPER.read_text("ascii"))
    bd_text = re.sub(r"\s+", " ", VENDORED.read_text("ascii"))
    assert "std::max(std::min(ZZ, maxZ), 1)" in bd_text
    assert "if (Z < 0)" not in bd_text
    assert "if (Z < 0) { Z = 0; }" in helper_text
    assert "if (Z > G4int(maxZ)) { Z = maxZ; }" in helper_text
    assert "std::max(std::min(" not in helper_text

    # T-42's brace-depth recount, on the second copy's own declaration spelling: the helper's parse
    # is proved complete the same independent way the first copy's is.
    assert len(helper.capture_records) == independently_counted_records(
        HELPER.read_text("ascii"), "constexpr capRate capRates"
    )

    # The pins the README publishes for the second file are computed from the file, never typed.
    data = HELPER.read_bytes()
    assert b"\r" not in data, (
        "the checkout rewrote the vendored helper's line endings: check that .gitattributes still "
        "carries `third_party/geant4/** -text`"
    )
    blob = git_blob_id(data)
    digest = hashlib.sha256(data).hexdigest()
    assert blob == d1.HELPER_BLOB_ID
    assert digest == d1.HELPER_SHA256
    readme = VENDORED_README.read_text("utf-8")
    assert "v11.4.2/G4MuonicAtomHelper.cc" in readme
    assert blob in readme
    assert digest in readme
    # Computed outside the f-string on purpose: a backslash inside an f-string expression is a
    # syntax error before Python 3.12, and the CI matrix still runs 3.11.
    line_count = data.count(b"\n")
    assert f"{len(data)} bytes, {line_count} lines" in readme
    assert {p.name for p in HELPER.parent.iterdir() if p.is_file()} == {VENDORED.name, HELPER.name}


def test_t70_mutation_drill_a_moved_digit_in_the_second_copy_is_named(tmp_path):
    """Drill for T-69: change one rate in the helper and the comparison must say which record.

    Two halves, because two guards have to hold. The comparison must *find* the change and name the
    `(Z, A)` it belongs to -- a comparison that reported "the copies differ" would have passed a
    truncated parse just as happily. And `load_helper` must refuse the mutated bytes outright,
    naming the pin they no longer match, so the failure a maintainer sees is "this is not upstream's
    file" rather than an unexplained table mismatch.
    """
    bd = extraction()
    original = HELPER.read_text("ascii")
    rate_literal = bd.capture_literals[0][0]
    z, a = bd.capture_records[0][:2]

    mutated = original.replace(rate_literal, rate_literal + "1", 1)
    assert mutated != original, "the drill did not change the source it was pointed at"

    problems = compare_copies(d1.extract(mutated, d1.HELPER), bd)
    assert problems, "a moved digit in the second copy went unnoticed"
    assert any(f"(Z, A)={(z, a)}" in problem for problem in problems), problems

    corrupted = tmp_path / "G4MuonicAtomHelper.cc"
    corrupted.write_bytes(mutated.encode("ascii"))
    with pytest.raises(d1.SourceExtractionError) as raised:
        d1.load_helper(corrupted)
    assert d1.HELPER_BLOB_ID in str(raised.value)


def test_t50_every_fallback_coefficient_occurs_verbatim_in_the_source():
    """`#FALLBACK` ships each constant in the source's own spelling, and here is the proof.

    A directive value is one opaque string to the reader, so the faithful spelling is available --
    `875.e-9` rather than a re-rendered `8.75e-07`. Faithful is only worth anything if it is checked:
    each coefficient string must occur, character for character, in the vendored file.
    """
    found = extraction()
    text = VENDORED.read_text("ascii")
    coefficients = found.coefficients

    assert tuple(coefficients) == d1.FALLBACK_NAMES, "the fallback must declare all eight inputs"
    for name, literal in coefficients.items():
        assert literal in text, f"{name}={literal!r} is not the text of any constant in the source"

    directive = d1.render_fallback_directive(d1.FALLBACK_MODEL, coefficients)
    model, parsed = d1.parse_fallback_directive(directive)
    assert model == d1.FALLBACK_MODEL and parsed == coefficients  # the directive round-trips


def test_t51_the_reference_implementation_reads_its_constants_from_the_directive():
    """Mutate the parsed `#FALLBACK` value and the model must move; otherwise it is decorative.

    The declared-as-data claim is that the directive *is* the model, not a comment beside a hard-coded
    one. The test that settles it is a mutation: perturb one coefficient in the directive string, and
    a rate that does not change means the constant is really coming from somewhere else.
    """
    found = extraction()
    directive = d1.render_fallback_directive(d1.FALLBACK_MODEL, found.coefficients)
    model = d1.GoulardPrimakoff.from_directive(directive, found.zeff)

    # Three points the table does not list, so the fallback is what answers at each. More than one
    # is needed because two of the eight constants are clamp bounds: `zmin` only bites below the
    # clamp and `zmax` only above it, so a single mid-range probe would report them as dead when
    # they are simply not in play there. The claim under test is that no constant is inert.
    probes = ((1, 3), (26, 77), (110, 250))
    baselines = [model.rate(*probe) for probe in probes]
    assert baselines[1] == d1.capture_rate(*probes[1], found.capture_records, model)

    for name in d1.FALLBACK_NAMES:
        original = found.coefficients[name]
        # The clamp bounds are integers; the coefficients are floats. Double either one and any
        # evaluation that really uses it has to move.
        mutated = dict(found.coefficients)
        # Clamp bounds move INWARD (zmin up, zmax down): outward is not a perturbation but an
        # invalid directive, and the model now rejects it -- asserted separately below.
        if name == "zmin":
            mutated[name] = str(int(original) + 1)
        elif name == "zmax":
            mutated[name] = str(int(original) - 1)
        else:
            mutated[name] = f"{float(original) * 2}"
        moved = d1.GoulardPrimakoff.from_directive(
            d1.render_fallback_directive(d1.FALLBACK_MODEL, mutated), found.zeff
        )
        assert any(
            moved.rate(*probe) != baseline for probe, baseline in zip(probes, baselines, strict=True)
        ), (
            f"mutating {name} in the '#FALLBACK' string changed no rate anywhere in the probe set, "
            "so the reference implementation is not reading that coefficient from the directive"
        )

    # And a directive missing a coefficient is rejected rather than silently defaulted.
    incomplete = {k: v for k, v in found.coefficients.items() if k != "b0c"}
    with pytest.raises(ValueError, match="b0c"):
        d1.render_fallback_directive(d1.FALLBACK_MODEL, incomplete)
    with pytest.raises(ValueError, match="b0c"):
        d1.GoulardPrimakoff.from_directive("goulard_primakoff b0a=-0.03", found.zeff)

    # A clamp that does not index the table it is declared against is a diagnosis, not an IndexError
    # thrown from inside the model at whichever consumer happened to evaluate it first.
    outward = dict(found.coefficients)
    outward["zmax"] = str(len(found.zeff))
    with pytest.raises(ValueError, match="muon_zeff table"):
        d1.GoulardPrimakoff.from_directive(
            d1.render_fallback_directive(d1.FALLBACK_MODEL, outward), found.zeff
        )


# --------------------------------------------------------------------------------------------
# T-56 -- Layer-2 row keys for a table whose primary key is a single column (FORMAT_SPEC.md 3)
# --------------------------------------------------------------------------------------------

SINGLE_KEY_ROW = {
    "source_bibkey": "geant4_v11_4_2",
    "source_locator": "third_party/geant4/v11.4.2/G4MuonMinusBoundDecay.cc",
    "unc_type": "table",
    "conditions": "none",
    "validity_range": "Z=1",
    "evaluation_method": "compiled-in constant table",
    "single_source": False,
    "needs_verification": True,
    "recommendation": "",
    "evaluation_id": "single-key-fixture",
    "source_library": "geant4-compiled-in",
    "isotope_resolved": False,
}


def single_key_pair(records=((1, 1.0), (2, 1.98))):
    """A one-key Layer-1 table and the Layer-2 document that describes it."""
    document = provenance.ProvDocument(
        dataset="G4MuonicData",
        version="0.1.0",
        profile="parity",
        seam="d1_nuclear_capture",
        precedence=("geant4-compiled-in",),
        rows={str(z): provenance.ProvRow(**SINGLE_KEY_ROW) for z, _ in records},
    )
    raw = provenance.document_bytes(document)
    table = spec.G4DatTable(
        directives={
            "GRAMMAR": spec.GRAMMAR_VERSION,
            "DATASET": document.dataset,
            "VERSION": document.version,
            "PROFILE": document.profile,
            "SEAM": document.seam,
            "TABLE": "muon_zeff",
            "GENERATOR": "openmucf-g4 test",
            "SOURCEDIGEST": provenance.source_digest(raw),
            "SOURCESHA": d1.UPSTREAM_COMMIT,
            "UNITS": "value=dimensionless",
            "COLUMNS": "Z value",
            "VALIDITY": "Z:0-100",
        },
        records=tuple(records),
    )
    return table, document


def test_t56_single_key_tables_have_a_defined_row_key():
    """A table keyed by one column keys its rows by that column's unpadded integer.

    Until `muon_zeff` needed it this was an explicitly *registered* undefined case: the checker
    refused such a table outright and said in a comment that the `"Z-A"` key "is only defined for a
    table declaring both". Defining it now is the decision that was deferred, not a re-opening --
    and Layer 2 is never read by Geant4, so nothing in the C++ contract moves.

    Both directions of mismatch must still raise. That is the half most worth testing: a rule that
    only catches unkeyable rows lets records ship with no provenance at all.
    """
    table, document = single_key_pair()
    spec.validate(table)
    assert provenance.check_against_table(table, document) is None

    # It also has to survive the JSON round trip, since the key pattern is checked on decode.
    assert provenance.from_json_obj(provenance.to_json_obj(document)) == document

    # A record with no row.
    extra_record = dataclasses.replace(table, records=(*table.records, (3, 2.94)))
    with pytest.raises(ValueError, match="1 record"):
        provenance.check_against_table(extra_record, document)

    # A row with no record -- including one wearing the two-column spelling, which is not this
    # table's key form and must therefore read as an unmatched row rather than as a near-miss.
    for stray in ("3", "1-1"):
        extra_row = dataclasses.replace(
            document, rows={**document.rows, stray: provenance.ProvRow(**SINGLE_KEY_ROW)}
        )
        with pytest.raises(ValueError, match="1 row"):
            provenance.check_against_table(table, extra_row)

    # Zero padding stays rejected in the new form, exactly as in the old one.
    for bad in ("029", "01-1", "+1", "1 "):
        with pytest.raises(ValueError, match="row key"):
            provenance.validate_document(
                {**provenance.to_json_obj(document), "rows": {bad: dict(SINGLE_KEY_ROW)}}
            )


# --------------------------------------------------------------------------------------------
# T-48, T-49, T-52 -- the compiled oracle: the sweep digest, the diagnostic subset, the edges
# --------------------------------------------------------------------------------------------


#: Every field the oracle's header declares. The set is closed on purpose: a missing field and an
#: invented one are both defects, and the parser below treats anything else on a `#` line as prose.
ORACLE_FIELDS = frozenset(
    {
        "upstream_commit",
        "upstream_path",
        "upstream_blob",
        "driver",
        "driver_degenerate",
        "build",
        "sweep",
        "digest_rule",
        "fullsweep_sha256",
        "subset",
        "columns",
        "zeff",
        "degenerate",
    }
)


def read_oracle() -> dict:
    """Parse the harvested oracle into its header fields, subset, zeff rows and degenerate rows.

    Values are converted with `float.fromhex` and compared as **numbers**. Comparing the printed
    strings instead would turn a C `%a` versus Python `float.hex()` formatting difference into a
    parity failure, which is a question about printf and not about physics.
    """
    header: dict[str, str] = {}
    subset: dict[tuple[int, int], float] = {}
    zeff: dict[int, float] = {}
    degenerate_rates: list[tuple[int, int, str, float]] = []
    degenerate_zeff: dict[int, float] = {}
    # The rows as WRITTEN, kept beside the parsed values so the file can be handed back to its own
    # producer verbatim. The hexfloat spellings are the harvest's own bytes and nothing here may
    # re-render them: C's "%a" prints the shortest form, Python's float.hex() pads to 13 digits.
    raw_subset: list[tuple[int, int, str]] = []
    raw_zeff: list[tuple[int, str]] = []
    raw_degenerate: list[str] = []
    build: list[str] = []
    terminated = False
    last_field: str | None = None
    for line in ORACLE.read_text("ascii").splitlines():
        if line == "#END":
            # A SECOND `#END` is content after `#END`. Tolerating it would let a whole second body
            # be appended below the terminator and read by nobody.
            assert not terminated, "the oracle carries more than one #END"
            terminated = True
            continue
        assert not terminated, f"content after #END: {line!r}"
        if line.startswith("#"):
            # A header FIELD is one of the declared names on the column the header is aligned to;
            # everything else on a `#` line is prose. Keying on "first word of any comment" instead
            # turned every prose word into a header key, which is how a field could be duplicated
            # (first-wins) or shadowed by a sentence that happened to start with its name.
            body = line[1:].strip()
            name = body.split(None, 1)[0] if body else ""
            if name in ORACLE_FIELDS and line.startswith(f"# {name} "):
                assert name not in header, f"duplicate header field {name!r}"
                header[name] = body[len(name) :].strip()
                last_field = name
                if name == "build":
                    build.append(header[name])
            elif line.startswith("#" + " " * 19) and last_field:
                header[last_field] += " " + body
                if last_field == "build":
                    build.append(body)
            else:
                last_field = None  # a prose line ends the field it followed
            continue
        # A data row ends the header field above it too. Without this a comment line dropped in
        # among the rows attaches to a field sixty lines further up, which is where an unchecked
        # sentence could be smuggled into a checked one.
        last_field = None
        # Every keyed section rejects a repeat. Assigning into a dict is last-wins, so a duplicated
        # row SHADOWS the one before it: a wrong value could sit in the file, be read, be discarded
        # in favour of the correct copy underneath, and be certified by a test that never saw it.
        # The oracle is deliberately outside the byte-diff audit -- re-derivation is its whole
        # protection -- so a hand-edit is precisely the threat this parser has to survive.
        fields = line.split()
        # Name a malformed row rather than dying on an index further down: an empty line, a stray
        # token, or a spelling no driver emits should say so, not surface as an IndexError from a
        # parser three assertions away from the cause.
        spelling = fields[-1] if fields else ""
        assert fields and spelling == spelling.lower() and not spelling.startswith("+"), (
            f"not a row this oracle's producer emits: {line!r}"
        )
        if fields[0] == "ZEFF":
            key = int(fields[1])
            assert key not in zeff, f"duplicate ZEFF row for Z={key}"
            zeff[key] = float.fromhex(fields[2])
            raw_zeff.append((key, fields[2]))
        elif fields[0] == "RATE":
            degenerate_rates.append(
                (int(fields[1]), int(fields[2]), fields[3], float.fromhex(fields[4]))
            )
            raw_degenerate.append(line)
        elif fields[0] == "ZEFFCLAMP":
            key = int(fields[1])
            assert key not in degenerate_zeff, f"duplicate ZEFFCLAMP row for Z={key}"
            degenerate_zeff[key] = float.fromhex(fields[2])
            raw_degenerate.append(line)
        else:
            pair = (int(fields[0]), int(fields[1]))
            assert pair not in subset, f"duplicate subset row for {pair}"
            subset[pair] = float.fromhex(fields[2])
            raw_subset.append((*pair, fields[2]))
    assert terminated, "the oracle is not #END-terminated"
    # Ascending by key, like every other committed table here (E015's discipline). The producer
    # emits them sorted, so a file in any other order is one no rebuild reproduces -- and a
    # reordered file that still passed would be a certified artifact drifting from its own producer.
    assert list(subset) == sorted(subset), "the oracle's subset rows are not ascending by (Z, A)"
    assert list(zeff) == sorted(zeff), "the oracle's ZEFF rows are not ascending by Z"
    return {
        "header": header,
        "subset": subset,
        "zeff": zeff,
        "degenerate_rates": degenerate_rates,
        "degenerate_zeff": degenerate_zeff,
        "raw": {
            "subset": raw_subset,
            "zeff": raw_zeff,
            "degenerate": raw_degenerate,
            "build": build,
        },
    }


def oracle_producer():
    """`cpp/tools/build_oracle.py`, loaded by path -- `cpp/tools/` is a directory, not a package."""
    path = REPO / "cpp" / "tools" / "build_oracle.py"
    spec = importlib.util.spec_from_file_location("build_oracle", path)
    assert spec and spec.loader, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RATE_PROBE_DECL = re.compile(r"probes\[\]\[2\]\s*=\s*\{(.*?)\}\s*;", re.DOTALL)
_RATE_PROBE_PAIR = re.compile(r"\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}")
_CLAMP_PROBE_DECL = re.compile(r"for\s*\(\s*int\s+Z\s*:\s*\{([^}]*)\}\s*\)")


def declared_degenerate_probes() -> tuple[list[tuple[int, int]], list[int]]:
    """The degenerate probe set, read out of the driver that is the only place it exists.

    `harvest_d1_degenerate.cc` decides which inputs the oracle's degenerate block records; nothing
    else in the repository states them. Reading them back out of the driver is what makes the block's
    composition checkable at all -- writing "four rate probes and four clamp probes" into this file
    would be the written-down count the parity discipline forbids, and would agree with a truncated
    harvest as happily as with a whole one.
    """
    source = (REPO / "cpp" / "tools" / "harvest_d1_degenerate.cc").read_text("ascii")
    rate_block = _RATE_PROBE_DECL.search(source)
    clamp_block = _CLAMP_PROBE_DECL.search(source)
    assert rate_block and clamp_block, "the degenerate driver no longer declares its probes as lists"
    rates = [(int(z), int(a)) for z, a in _RATE_PROBE_PAIR.findall(rate_block.group(1))]
    clamps = [int(z) for z in clamp_block.group(1).split(",")]
    assert rates and clamps, "the degenerate driver declares an empty probe set"
    return rates, clamps


def reference_model(found: d1.D1Extraction) -> d1.GoulardPrimakoff:
    """The declared model, built the way a consumer would: from the `#FALLBACK` string."""
    return d1.GoulardPrimakoff.from_directive(
        d1.render_fallback_directive(d1.FALLBACK_MODEL, found.coefficients), found.zeff
    )


def test_t48_the_full_sweep_digest_matches_the_compiled_library():
    """36000 points of bit-parity against a real Geant4 binary, checked with no Geant4 present.

    This is the whole parity claim in one assertion, and it is worth being precise about why it is
    not circular. The digest on the right came out of a Geant4-linked binary and is committed; the
    digest on the left is computed here, now, by evaluating the reference implementation over the
    same box. Nothing in this repository can regenerate the oracle -- it is not written by
    `make g4data` and not byte-diffed by `make audit` -- so the two sides have genuinely independent
    origins, which is what makes a 64-byte committed file worth an exhaustive parity proof.

    It also runs on every CI platform, which is the point of the reference implementation being pure
    Python: the arm64 job proves the same digest cross-architecture, and needs no Geant4 to do it.
    """
    found = extraction()
    oracle = read_oracle()
    computed = d1.sweep_digest(found.capture_records, reference_model(found))
    assert computed == oracle["header"]["fullsweep_sha256"], (
        "the Python reference implementation no longer reproduces the compiled Geant4 sweep. Do NOT "
        "re-pin the oracle and do NOT adjust the reference implementation to match: on the build "
        "recorded in the oracle header the value is determined, so a disagreement means something "
        "about the extraction, the association order or the model has changed."
    )
    # The digest is over the SWEEP, so a test that never evaluated the box could still pass the line
    # above if `sweep_digest` were gutted. Pin the shape it hashes, from the module's own bounds --
    # the WHOLE sentence, including the traversal order, because that clause is the digest's
    # definition and Stage 3's C++ validator implements the digest from this file. A pinned prefix
    # left "row-major, Z ascending outermost" free to be edited to its own opposite.
    swept = (d1.SWEEP_Z_MAX - d1.SWEEP_Z_MIN + 1) * (d1.SWEEP_A_MAX - d1.SWEEP_A_MIN + 1)
    assert oracle["header"]["sweep"] == (
        f"Z {d1.SWEEP_Z_MIN}..{d1.SWEEP_Z_MAX} x A {d1.SWEEP_A_MIN}..{d1.SWEEP_A_MAX} = {swept} "
        f"points, row-major, Z ascending outermost"
    )

    # Everything in this file that is not a measured number is emitted by its producer, so hand the
    # file back to the producer and require the same bytes out. That closes the whole class at once
    # -- prose, the field set and its order, every field value, the row order, the spacing, stray
    # lines, invented fields -- instead of asserting one clause at a time and stopping one clause
    # short, which is what the four rounds before this one each did.
    #
    # It is not circular for the NUMBERS. The rows go back in as the harvest's own `%a` strings
    # (they must: C prints the shortest form and `float.hex()` pads to 13 digits), and every one of
    # them is re-derived against the Python reimplementation elsewhere in this file, at zero ulp.
    # What this pins is that the committed artifact is still the artifact the producer emits, which
    # is exactly what a hand-edit breaks -- and the oracle is deliberately outside `make audit`'s
    # byte-diff, so nothing else would notice.
    #
    # `build` is the one input with no derivation: it records the environment the harvest ran in,
    # and nothing here can confirm it. It is fed back in as found, and checked only for presence.
    #
    # The trust boundary, stated rather than implied: this pins the file to its PRODUCER, so the
    # prose is only as true as the producer's constants. That is the right boundary for a hand-edit
    # -- the threat this file has, being outside the byte-diff audit -- but it is not a check on the
    # code. One clause is exempted from that limit because Stage 3's C++ validator implements the
    # digest from it: "big-endian" is pinned to the arithmetic instead, by requiring the committed
    # digest to reproduce under big-endian packing and NOT under little-endian.
    # Be exact about which half of that sentence this pins. The ENCODING is pinned to arithmetic:
    # the committed digest reproduces under big-endian binary64 and under nothing else tried, so a
    # validator packing little-endian or single precision cannot agree with this dataset and think
    # it has. The WORDING is pinned to the producer, like the rest of the header -- a matching edit
    # to both the artifact and the producer's constant would keep a wrong sentence, and that is a
    # code change inside the reviewable diff, not the hand-edit this file is exposed to.
    alternative = reference_model(found)
    for code, label in (("<d", "little-endian"), (">f", "single-precision")):
        digest = hashlib.sha256()
        for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1):
            for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1):
                digest.update(
                    struct.pack(code, d1.capture_rate(z, a, found.capture_records, alternative))
                )
        assert computed != digest.hexdigest(), f"the digest does not distinguish {label}"
    assert "big-endian" in oracle["header"]["digest_rule"]
    assert "binary64" in oracle["header"]["digest_rule"]
    header = oracle["header"]
    assert set(header) == set(ORACLE_FIELDS), (
        f"the oracle's header fields are not the declared set: missing "
        f"{sorted(ORACLE_FIELDS - set(header))}, unexpected {sorted(set(header) - ORACLE_FIELDS)}"
    )
    assert header["upstream_commit"] == d1.UPSTREAM_COMMIT
    assert header["upstream_path"] == d1.UPSTREAM_PATH
    assert header["upstream_blob"] == d1.UPSTREAM_BLOB_ID
    assert oracle["raw"]["build"], "the oracle does not name the build that produced it"

    raw = oracle["raw"]
    hits = {(z, a) for z, a, _, _ in found.capture_records}
    corners = {
        (z, a)
        for z in (d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX)
        for a in (d1.SWEEP_A_MIN, d1.SWEEP_A_MAX)
    }
    negatives = set(oracle["subset"]) - hits - corners
    rendered = oracle_producer().render_oracle(
        subset=raw["subset"],
        zeff=raw["zeff"],
        degenerate_lines=raw["degenerate"],
        build=raw["build"],
        digest=computed,
        swept=swept,
        tallies=(len(hits), len(negatives), len(corners)),
    )
    # BYTES, not text. `read_text` applies universal newlines, so comparing against it accepted a
    # CRLF rewrite of the whole file -- the comparison said "same text" while the commit that
    # introduced it said "same bytes out". `.gitattributes` keeps checkout from doing that rewrite,
    # but nothing kept a hand-edit from doing it, and this file is outside the byte-diff audit.
    assert rendered.encode("ascii") == ORACLE.read_bytes(), (
        "the committed oracle is no longer what cpp/tools/build_oracle.py emits from its own rows. "
        "Rebuild it with that script rather than editing it by hand."
    )


def test_t49_the_diagnostic_subset_agrees_to_zero_ulp():
    """Every point the oracle spells out, re-derived and compared bit-for-bit.

    The gate asks for <= 1 ulp; this asserts the measured **0**, deliberately. A drift to one ulp
    would be a finding -- something in the evaluation order or the constants moved -- and a test
    with a 1-ulp tolerance would absorb it silently. The subset is what makes a digest mismatch
    diagnosable: a bare hash says only that something moved, these rows say which points.
    """
    found = extraction()
    model = reference_model(found)
    oracle = read_oracle()

    assert oracle["subset"], "the oracle carries no diagnostic subset"
    for (z, a), expected in oracle["subset"].items():
        actual = d1.capture_rate(z, a, found.capture_records, model)
        assert struct.pack(">d", actual) == struct.pack(">d", expected), (
            f"({z}, {a}): reference {actual!r} is not bit-identical to compiled Geant4 {expected!r}"
        )

    # Every table hit is in the subset by the stated rule, so the fallback is not the only path
    # covered: these are the points where the compiled function returns `cRate / microsecond`.
    for z, a, _, _ in found.capture_records:
        assert (z, a) in oracle["subset"]

    # The subset's COMPOSITION is part of the fixture, not decoration. The value loop above checks
    # what is present and says nothing about what is absent, so a subset that had quietly lost its
    # first-negative rows or a corner would still pass while the negative-rate finding lost its
    # evidence. The selection rule is fully derivable, so derive it and require set equality.
    hits = {(z, a) for z, a, _, _ in found.capture_records}
    first_negative = set()
    for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1):
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1):
            if d1.capture_rate(z, a, found.capture_records, model) < 0.0:
                first_negative.add((z, a))
                break
    corners = {
        (z, a)
        for z in (d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX)
        for a in (d1.SWEEP_A_MIN, d1.SWEEP_A_MAX)
    }
    derived = hits | first_negative | corners
    assert set(oracle["subset"]) == derived, (
        f"the oracle's subset is not what its own rule selects: "
        f"{sorted(derived - set(oracle['subset']))} missing, "
        f"{sorted(set(oracle['subset']) - derived)} unexpected. Rebuild it with "
        f"cpp/tools/build_oracle.py rather than editing it by hand."
    )
    # ...and the header states the same composition in words, where a reader meets it first. Pin the
    # sentence to the derivation so the prose cannot drift away from the rows underneath it.
    assert oracle["header"]["subset"] == (
        f"{len(derived)} points = {len(hits)} table hits + {len(first_negative)} first-negative + "
        f"{len(corners)} corners, deduplicated"
    )

    # The zeff rows are what `GetMuonZeff(Z)` RETURNS, which is not the same thing as the array's
    # entries -- the function clamps its argument into [1, maxZ] first. Comparing them to raw array
    # elements is what surfaced the point: at Z = 0 the array holds 0.0 while the function returns
    # zeff[1] = 1.0, so element 0 can never be observed through the accessor.
    # Coverage before values, for the same reason the subset's composition is checked above: the
    # loop below only visits rows the file happens to carry, so a harvest interrupted after the
    # sweep -- which is exactly when the driver is still printing these -- would leave a short tail
    # whose every remaining value is correct. The range is derived from the vendored source: every
    # entry of the table, plus one probe past its last index, which is what makes the clamp
    # observable at the top end at all.
    covered = set(range(len(found.zeff) + 1))
    assert set(oracle["zeff"]) == covered, (
        f"the oracle's zeff rows do not cover Z 0..{len(found.zeff)} as its header claims: missing "
        f"{sorted(covered - set(oracle['zeff']))}, unexpected "
        f"{sorted(set(oracle['zeff']) - covered)}"
    )
    # ...and the sentence that makes that claim is pinned to the same derivation, exactly as the
    # subset's tally is. It was this header line the coverage guard above was written to keep true,
    # and it could be edited into a falsehood while every row underneath stayed correct.
    assert oracle["header"]["zeff"] == (
        f"Z 0..{len(found.zeff)}, pinning the clamp at both ends"
    )
    for z, expected in oracle["zeff"].items():
        assert model.muon_zeff(z) == expected

    # The dataset ships element 0 anyway, and says why: "101/101 bit-identical" means the array as
    # declared, and silently dropping an element the dataset claims to reproduce would be a worse
    # artifact. Its unreachability is a disclosure, so assert the unreachability itself.
    unreachable = found.zeff[0]
    assert unreachable != oracle["zeff"][0] == found.zeff[model.zmin]
    assert unreachable not in {model.muon_zeff(z) for z in range(-5, len(found.zeff) + 20)}


def test_t77_ulp_distance_counts_representable_doubles_and_the_subset_maximum_moves_with_a_perturbed_row():
    """The figure the documents state as "maximum N ulp" is computed, not copied, so the computation
    is drilled: the distance counts neighbouring doubles on either side of one, across the sign at
    zero, and the subset maximum moves the moment one oracle row is nudged to its neighbouring
    double."""
    assert _ulp_distance(1.0, math.nextafter(1.0, 2.0)) == 1
    assert _ulp_distance(-1.0, math.nextafter(-1.0, 0.0)) == 1
    assert _ulp_distance(0.0, -0.0) == 0
    smallest = math.nextafter(0.0, 1.0)
    assert _ulp_distance(-smallest, smallest) == 2

    found = extraction()
    model = reference_model(found)
    oracle = read_oracle()
    assert subset_max_ulp(found, model, oracle) == 0

    perturbed = dict(oracle)
    perturbed["subset"] = dict(oracle["subset"])
    key = min(perturbed["subset"])
    perturbed["subset"][key] = math.nextafter(perturbed["subset"][key], math.inf)
    assert subset_max_ulp(found, model, perturbed) == 1


def test_t52_degenerate_inputs_reproduce_the_recorded_classification():
    """What Geant4 does at Z = 0, A = 0 and Z < 0 -- registered as a finding, reproduced, not fixed.

    A parity dataset reproduces the library including its edges, so these are compared by
    classification (`nan` / `+inf` / sign) rather than by value: a NaN has no single bit pattern, so
    there is nothing here to hash and nothing to assert equal. The finding is that Geant4 returns
    non-finite rates with no coded rejection at all -- and our own format rejects non-finite floats,
    which is why the declared model carries a domain contract instead.
    """
    found = extraction()
    model = reference_model(found)
    oracle = read_oracle()

    # Composition before classification, for the third and last harvested section: this block is
    # spliced in from its own driver, which prints the clamp probes LAST, so an interrupted run --
    # or a one-line hand-edit -- drops rows whose absence every assertion below survives. The probe
    # set is derived from the driver that produced it, the only place it is stated.
    rate_probes, clamp_probes = declared_degenerate_probes()
    assert [(z, a) for z, a, _, _ in oracle["degenerate_rates"]] == rate_probes, (
        f"the oracle's degenerate rate probes are not the ones cpp/tools/harvest_d1_degenerate.cc "
        f"harvests: file has {[(z, a) for z, a, _, _ in oracle['degenerate_rates']]}, driver "
        f"declares {rate_probes}"
    )
    # Exact order, not sorted: the producer compares the harvest's clamp probes as a LIST, so a
    # sorted comparison here would accept a committed file in an order no rebuild can produce.
    assert list(oracle["degenerate_zeff"]) == clamp_probes, (
        f"the oracle's zeff clamp probes are not the driver's, in order: file has "
        f"{list(oracle['degenerate_zeff'])}, driver declares {clamp_probes}"
    )

    assert oracle["degenerate_rates"], "the oracle records no degenerate inputs"
    assert {row[2] for row in oracle["degenerate_rates"]} == {"nan", "+inf", "negative"}
    for z, a, classification, recorded in oracle["degenerate_rates"]:
        # Every recorded probe is outside the declared domain, and every one of them gets a value
        # out of Geant4 rather than a rejection. That gap is `DATASET_D1.md`'s finding F-2, so state
        # it as an assertion: the declared model refuses where the library answers.
        assert z < 1 or a < 1
        with pytest.raises(ValueError, match="outside its domain"):
            model.rate(z, a)

        # ...and the reproduction is still checked, on the ungated evaluation, wherever the
        # arithmetic has a value at all.
        #
        # At Z = 0 and at A = 0 it does not. Both divide by zero -- `a2ze = 0.5*A/Z` for the first,
        # `... / G4double(A*4)` for the second -- and there the two languages part company by
        # design: IEEE-754 hands C++ a NaN and a +inf and lets them propagate into a lifetime, while
        # CPython raises. The recorded classification is what Geant4 does; the exception is what
        # Python does; and the fact that one of them silently produces a number is the finding.
        if z == 0 or a == 0:
            assert not math.isfinite(recorded)
            assert classification == ("nan" if z == 0 else "+inf")
            assert math.isnan(recorded) if z == 0 else (math.isinf(recorded) and recorded > 0)
            # The SIGN of the NaN is measured too, and the classification column does not carry it:
            # the driver's `classify()` collapses both signs to "nan", so only the hexfloat records
            # that this build returns a negative NaN. Pinned as the measurement it is -- if a
            # re-harvest ever prints a positive one, that is a finding about the seam and should
            # arrive as a failure here rather than as a silent edit nobody sees.
            if z == 0:
                assert math.copysign(1.0, recorded) == -1.0, "the recorded NaN lost its sign"
            with pytest.raises(ZeroDivisionError):
                model.evaluate_unchecked(z, a)
        else:
            # Z < 0 is the dangerous one: the zeff clamp pulls the lookup back to Z=1 and the
            # arithmetic completes, so Geant4 hands back a finite, negative, entirely
            # plausible-looking rate. Reproduced bit-for-bit -- that is what makes it evidence.
            assert classification == "negative"
            actual = model.evaluate_unchecked(z, a)
            assert math.isfinite(actual) and actual < 0
            assert struct.pack(">d", actual) == struct.pack(">d", recorded)

    # The clamp holds at both ends, which is what makes `zeff` evaluable for any Z at all.
    assert oracle["degenerate_zeff"], "the oracle records no zeff clamp probes"
    for z, expected in oracle["degenerate_zeff"].items():
        assert model.muon_zeff(z) == expected
    below = [z for z in oracle["degenerate_zeff"] if z < model.zmin]
    above = [z for z in oracle["degenerate_zeff"] if z > model.zmax]
    assert below and above, "the clamp probes must cover both ends"
    assert {model.muon_zeff(z) for z in below} == {found.zeff[model.zmin]}
    assert {model.muon_zeff(z) for z in above} == {found.zeff[model.zmax]}


# --------------------------------------------------------------------------------------------
# T-76 -- every shipped table sits beside its provenance file
# --------------------------------------------------------------------------------------------


def test_t76_every_shipped_table_ships_beside_its_provenance_file():
    """A `.g4dat` under `data/` without its `.prov.json`, or the reverse, is a half-shipped dataset.

    E009 binds Layer 1 to Layer 2 by a digest over the Layer-2 bytes, so a `.g4dat` published
    without its sibling carries a `#SOURCEDIGEST` nothing can check -- the one rejection that needs
    both files becomes unreachable, and the dataset's provenance claim goes with it. The reverse is
    just as bad in a different way: a Layer-2 document describing a table that is not there.

    Both directions, over `data/**` by rglob rather than over a list, because a list is the thing
    that goes stale when a seam is added. The malformed fixtures live outside `data/` by
    construction, so they are not swept up by this.
    """
    data = REPO / "data"
    tables = sorted(data.rglob("*.g4dat"))
    documents = sorted(data.rglob("*.prov.json"))
    assert tables and documents, "no shipped dataset files were found at all"

    # `.prov.json` carries TWO suffixes, so `Path.stem` leaves a trailing `.prov` on it and pairing
    # on `stem` silently matches nothing. Strip the whole extension by name instead.
    def paired(path: pathlib.Path, extension: str) -> pathlib.Path:
        base = path.name.removesuffix(".prov.json").removesuffix(".g4dat")
        return path.with_name(base + extension)

    orphan_tables = [p for p in tables if not paired(p, ".prov.json").exists()]
    orphan_documents = [p for p in documents if not paired(p, ".g4dat").exists()]
    assert not orphan_tables, (
        "shipped table(s) with no provenance file beside them, so their '#SOURCEDIGEST' can never "
        f"be checked: {[p.relative_to(REPO).as_posix() for p in orphan_tables]}"
    )
    assert not orphan_documents, (
        "provenance file(s) describing a table that is not shipped: "
        f"{[p.relative_to(REPO).as_posix() for p in orphan_documents]}"
    )
    # The pairing is a bijection on base names, so the two sweeps above cannot both pass on a
    # directory where one name is doing double duty.
    assert {paired(p, "") for p in tables} == {paired(p, "") for p in documents}


# --------------------------------------------------------------------------------------------
# T-67, T-68 -- the oracle's hexfloat grammar, and the two digest implementations
# --------------------------------------------------------------------------------------------


def oracle_hex_fields() -> list[str]:
    """Every value field of the committed oracle that is written as a hexfloat, as WRITTEN.

    The four non-finite spellings are excluded because they are not hexfloats: the degenerate block
    compares them by classification and never parses them as hex. Everything else in the file --
    subset rows, `ZEFF` rows, `ZEFFCLAMP` rows, and the `RATE` rows whose value is finite -- is a
    hexfloat and is subject to the grammar.
    """
    producer = oracle_producer()
    raw = read_oracle()["raw"]
    fields = [value for _, _, value in raw["subset"]]
    fields += [value for _, value in raw["zeff"]]
    fields += [
        line.split()[-1]
        for line in raw["degenerate"]
        if line.split()[-1] not in producer.NON_FINITE_FIELDS
    ]
    return fields


def test_t67_every_oracle_hexfloat_obeys_the_grammar_and_re_renders_to_its_own_bytes(tmp_path):
    """The spelling of a value is part of the artifact, and here is the rule it obeys.

    A hex reader is far more permissive than the producer that wrote these bytes. `float.fromhex`
    accepts `infinity`, `1.5p+3`, `0x1.5p3`, `0x1.5p+03`, `0x0.3p+5` and `0x1.50p+3` -- six
    spellings a C `%a` never prints. Three of them are caught by the grammar and three only by
    requiring the field to re-render to itself, so a validator implementing one half accepts files
    this producer cannot emit. Both halves are asserted here over every committed field, and both
    halves are then shown to be load-bearing on the six.

    The second half of the test is `read_sweep`'s side of the same rule: a harvest carrying a
    duplicate row, an upper-case field, a blank line, `infinity` or a duplicate `ZEFF` row must be
    rejected by a NAMED error that says which line. Each of the first four was once silent --
    skipped, or accepted last-wins, or handed to a parser that took it. The third half is
    `check_degenerate`'s: the committed degenerate block with one value respelled must be rejected
    by the re-render rule, naming the line.

    The respellings are DERIVED from one committed field, not typed: a typed sextet pins six
    strings nobody re-reads, while spellings built from a field the oracle actually carries stay
    tied to the grammar they exercise.
    """
    producer = oracle_producer()
    fields = oracle_hex_fields()
    assert fields, "the oracle carries no hexfloat fields"
    for value in fields:
        assert producer.hexfloat_problem(value) is None, value
        assert producer.HEXFLOAT.match(value), value
        assert producer.canonical_hex(float.fromhex(value)) == value, value

    def trailing_zero(spelling: str) -> str:
        """`0x1p+e` -> `0x1.0p+e`, `0x1.<m>p+e` -> `0x1.<m>0p+e`: same value, a spelling %a never prints."""
        mantissa, exponent = spelling[2:].split("p")
        digits = mantissa[2:] if "." in mantissa else ""
        return f"0x1.{digits}0p{exponent}"

    # The six respellings, derived from the first `ZEFF` field with room for one more mantissa digit,
    # split by which half rejects them -- named, so a change that quietly widened the grammar to
    # cover the re-render three would fail here rather than pass more.
    raw = read_oracle()["raw"]
    derivable = re.compile(r"^0x1(\.[0-9a-f]{1,12})?p\+[0-9]+$")
    field = next(value for _, value in raw["zeff"] if derivable.match(value))
    mantissa = field[2 : field.index("p")]
    mantissa_digits = mantissa[2:] if "." in mantissa else ""
    exponent = int(field[field.index("p+") + 2 :])
    by_grammar = ("infinity", field[2:], field.replace("p+", "p"))
    by_re_render = (
        field.replace("p+", "p+0"),
        trailing_zero(field),
        f"0x0.1{mantissa_digits}p+{exponent + 4}",
    )
    for spelling in by_grammar[1:] + by_re_render:
        assert float.fromhex(spelling) == float.fromhex(field), (spelling, field)
    for spelling in by_grammar:
        assert producer.HEXFLOAT.match(spelling) is None, spelling
        assert producer.hexfloat_problem(spelling) is not None, spelling
    for spelling in by_re_render:
        assert producer.HEXFLOAT.match(spelling), f"{spelling} should pass the grammar"
        assert producer.canonical_hex(float.fromhex(spelling)) != spelling, spelling
        assert producer.hexfloat_problem(spelling) is not None, spelling

    # `read_sweep`'s side of the same rule, on crafted harvests. Each rejection must be the NAMED
    # error and must carry `path:line`: "somewhere in this file" is what these replaced.
    size = len(extraction().zeff) - 1
    tail = [f"ZEFF {z} 0x1p+0" for z in range(size + 1)]
    good = "1 1 0x1p+0"
    for label, rows in {
        "duplicate_row": [good, good],
        "upper_case": ["1 1 0X1P+0"],
        "blank_line": [good, ""],
        "infinity": ["1 1 infinity"],
        "duplicate_zeff": [good, "ZEFF 0 0x1p+0", "ZEFF 0 0x1p+0"],
    }.items():
        harvest = tmp_path / f"{label}.txt"
        harvest.write_text("\n".join(rows + tail) + "\n", encoding="ascii", newline="\n")
        with pytest.raises(producer.SweepFormatError) as raised:
            producer.read_sweep(harvest, size)
        # The offending row is the last one given in each case, so its line number is len(rows).
        assert f"{harvest}:{len(rows)}:" in str(raised.value), (label, str(raised.value))

    # `check_degenerate`'s side: the driver's declared probes verbatim, except one `ZEFFCLAMP` value
    # respelled with a trailing mantissa zero -- grammar-valid and non-canonical, so it reaches the
    # re-render loop rather than the per-line shape rule -- rejected by name, `path:line`.
    block = list(raw["degenerate"])
    index = next(
        i
        for i, line in enumerate(block)
        if line.startswith("ZEFFCLAMP ") and derivable.match(line.split()[-1])
    )
    original = block[index].split()[-1]
    respelled = trailing_zero(original)
    assert float.fromhex(respelled) == float.fromhex(original)
    block[index] = block[index][: -len(original)] + respelled
    degenerate = tmp_path / "degenerate.txt"
    degenerate.write_text("\n".join(block) + "\n", encoding="ascii", newline="\n")
    with pytest.raises(producer.SweepFormatError) as raised:
        producer.check_degenerate(degenerate, REPO / "cpp" / "tools" / "harvest_d1_degenerate.cc")
    assert f"{degenerate}:{index + 1}:" in str(raised.value), str(raised.value)


def test_t68_the_two_digest_implementations_agree_with_the_compiled_oracle():
    """Three independent computations of one 64-hex number, required to be the same number.

    `d1.sweep_digest` walks the box and hashes the doubles it evaluates. `build_oracle.sweep_digest`
    hashes the hexfloat STRINGS a harvest carries. The oracle's `fullsweep_sha256` came out of a
    Geant4-linked binary. This test compares the first two and pins both to the
    third, so the C++ validator's digest has a value to reproduce rather than a description.

    The strings on the Python side are rendered by `canonical_hex`, which makes this a test of the
    renderer as well: a renderer that lost a digit would produce a different double and a different
    digest here, not a cosmetic difference.
    """
    producer = oracle_producer()
    found = extraction()
    model = reference_model(found)
    oracle = read_oracle()

    rates = {
        (z, a): producer.canonical_hex(d1.capture_rate(z, a, found.capture_records, model))
        for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1)
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1)
    }
    from_strings = producer.sweep_digest(rates)
    from_values = d1.sweep_digest(found.capture_records, model)
    assert from_strings == from_values, (
        "the two digest implementations disagree; one of them is not hashing what it claims to hash"
    )
    assert from_strings == oracle["header"]["fullsweep_sha256"]

    # Drill: the traversal order is part of the digest's definition, not a detail. Hashing exactly
    # the same doubles with A outermost must give a different answer, or "row-major, Z ascending
    # outermost" would be an unenforced sentence in the header and a validator could pick either.
    transposed = hashlib.sha256()
    for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1):
        for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1):
            transposed.update(struct.pack(">d", float.fromhex(rates[z, a])))
    assert transposed.hexdigest() != from_strings, (
        "the digest does not depend on the traversal order, so the header's 'Z ascending outermost' "
        "clause is not being enforced by anything"
    )


# --------------------------------------------------------------------------------------------
# T-43..T-47, T-53..T-55, T-57, T-58 -- the shipped dataset against the source it claims
# --------------------------------------------------------------------------------------------

CAPTURE_LAYER1 = D1DIR / "d1_capture.g4dat"
CAPTURE_LAYER2 = D1DIR / "d1_capture.prov.json"
ZEFF_LAYER1 = D1DIR / "d1_zeff.g4dat"
ZEFF_LAYER2 = D1DIR / "d1_zeff.prov.json"
#: The capture table's second profile, named by the profile.
MIZUNO_LAYER1 = D1DIR / f"d1_capture.{mizuno2025.PROFILE}.g4dat"
MIZUNO_LAYER2 = D1DIR / f"d1_capture.{mizuno2025.PROFILE}.prov.json"
GENERATOR = REPO / "scripts" / "generate_g4data.py"


def committed(layer1_path: pathlib.Path, layer2_path: pathlib.Path):
    """The committed pair, parsed: the Layer-1 table and its Layer-2 document."""
    table = spec.parse(layer1_path.read_bytes().decode("ascii"))
    raw = layer2_path.read_bytes()
    provenance.check_canonical_bytes(raw)
    document = provenance.from_json_obj(json.loads(raw.decode("ascii")))
    return table, document


def shipped_layer2_files() -> list[pathlib.Path]:
    """Every Layer-2 file this repository ships, example included."""
    return sorted((REPO / "data" / "g4").rglob("*.prov.json"))


def test_t43_every_capture_value_is_the_source_literal():
    """Bit-for-bit, both columns, against the float LITERALS in the vendored source.

    Not "close", and not "equal after re-parsing our own output": each committed double is compared
    to `float(<the exact text upstream wrote>)`, byte pattern against byte pattern. That is what
    rules out a transcription that happens to round to the same displayed digits, and it is why the
    extractor carries the literals alongside the parsed values.
    """
    found = extraction()
    table, _ = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)

    assert len(table.records) == len(found.capture_records)
    by_key = {(int(z), int(a)): (value, unc) for z, a, value, unc in table.records}
    assert len(by_key) == len(table.records), "the committed table has a duplicate key"

    for (z, a, _, _), (rate_text, error_text) in zip(
        found.capture_records, found.capture_literals, strict=True
    ):
        value, unc = by_key[(z, a)]
        assert struct.pack("<d", value) == struct.pack("<d", float(rate_text)), f"value at ({z}, {a})"
        assert struct.pack("<d", unc) == struct.pack("<d", float(error_text)), f"unc at ({z}, {a})"


def test_t44_every_effective_charge_is_the_source_literal():
    """The same, for all 101 entries -- including index 0, which the accessor can never return."""
    found = extraction()
    table, _ = committed(ZEFF_LAYER1, ZEFF_LAYER2)

    assert len(table.records) == len(found.zeff)
    by_z = {int(z): value for z, value in table.records}
    for z, literal in enumerate(found.zeff_literals):
        assert struct.pack("<d", by_z[z]) == struct.pack("<d", float(literal)), f"zeff[{z}]"
    # "101/101 bit-identical" means the array AS DECLARED, so the unreachable element ships too.
    assert 0 in by_z and by_z[0] == found.zeff[0]


def test_t45_row_sets_agree_three_ways():
    """extraction <-> Layer 1 <-> Layer 2, for both tables. Any two agreeing is not enough.

    The failure this rules out is a generated file that is internally consistent and wrong: Layer 1
    and Layer 2 are both produced by the same script, so checking them against each other proves
    only that the script is self-consistent. The third leg -- the vendored source -- is the one that
    makes it a parity check.
    """
    found = extraction()

    capture_table, capture_document = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)
    from_source = {f"{z}-{a}" for z, a, _, _ in found.capture_records}
    from_layer1 = {f"{int(z)}-{int(a)}" for z, a, _, _ in capture_table.records}
    assert from_source == from_layer1 == set(capture_document.rows)

    zeff_table, zeff_document = committed(ZEFF_LAYER1, ZEFF_LAYER2)
    zeff_source = {str(z) for z in range(len(found.zeff))}
    zeff_layer1 = {str(int(z)) for z, _ in zeff_table.records}
    assert zeff_source == zeff_layer1 == set(zeff_document.rows)

    # And the checker agrees, which is the rule a consumer would actually apply.
    assert provenance.check_against_table(capture_table, capture_document) is None
    assert provenance.check_against_table(zeff_table, zeff_document) is None


def test_t46_the_reorder_moved_nothing():
    """The shipped file is canonically sorted; the upstream array is not. Prove nothing was lost.

    Geant4's array is sorted by Z alone and contains exactly one `(Z, A)` inversion, so a format
    requiring ascending `(Z, A)` cannot preserve the source order. Re-ordering records is safe only
    if the multiset is unchanged -- so check the multiset, not the count, because a swap that
    duplicated one record and dropped another keeps the count identical.
    """
    found = extraction()
    capture_table, _ = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)

    source_multiset = sorted(found.capture_records)
    shipped_multiset = sorted((int(z), int(a), value, unc) for z, a, value, unc in capture_table.records)
    assert shipped_multiset == source_multiset

    # The source really is out of order, or this test is guarding nothing.
    source_keys = [(z, a) for z, a, _, _ in found.capture_records]
    assert source_keys != sorted(source_keys), "upstream is already sorted; T-46 no longer has a job"
    assert [z for z, _, _, _ in found.capture_records] == sorted(
        z for z, _, _, _ in found.capture_records
    ), "upstream is not even sorted by Z, which the early-exit scan depends on"

    # Both committed tables are ascending by their own declared key (E015).
    for table in (capture_table, committed(ZEFF_LAYER1, ZEFF_LAYER2)[0]):
        columns = table.directives["COLUMNS"].split()
        indices = [columns.index(name) for name in ("Z", "A") if name in columns]
        keys = [tuple(int(record[i]) for i in indices) for record in table.records]
        assert keys == sorted(keys)


def test_t47_the_reorder_preserves_geant4s_own_lookup():
    """Geant4's early-exit scan over the SOURCE order agrees with a keyed lookup over ours.

    This is the argument that the reorder is behaviour-preserving, rather than the hope. Geant4 does
    not do a dictionary lookup: it walks the array and gives up the moment it sees a Z greater than
    the one asked for (`if (capRates[j].Z > Z) break;`). That is only sound because the array is
    sorted by Z -- and our canonical `(Z, A)` order is a refinement of "sorted by Z", so the early
    exit fires at the same Z. Checking it over the whole sweep box is what turns that sentence into
    evidence.
    """
    found = extraction()
    capture_table, _ = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)
    shipped = {(int(z), int(a)): value for z, a, value, unc in capture_table.records}

    def geant4_scan(z: int, a: int) -> float | None:
        """Upstream's loop, verbatim, over the records in the order upstream declares them."""
        for record_z, record_a, value, _ in found.capture_records:
            if record_z == z and record_a == a:
                return value
            if record_z > z:
                return None
        return None

    hits = 0
    for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1):
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1):
            scanned = geant4_scan(z, a)
            keyed = shipped.get((z, a))
            assert scanned == keyed, f"({z}, {a}): source scan {scanned!r}, sorted lookup {keyed!r}"
            hits += scanned is not None
    assert hits == len(found.capture_records), "the scan did not reach every table row"


QUOTE_PREFIX = 'quoted from the upstream source comment: "'


def quoted_attribution(conditions: str) -> str:
    """The whole upstream quotation inside a `conditions` string, quote marks of its own included.

    Splitting on `"` and taking field 1 is the obvious way to do this and it is wrong: the zeff
    attribution quotes upstream quoting a paper *title*, so the field-1 slice stops at upstream's
    own quote mark after 27 of its 218 characters and the rest goes unchecked.
    """
    assert conditions.startswith(QUOTE_PREFIX), conditions
    closing = conditions.rindex('". ')
    # `rindex` is only unambiguous while the prose after the quotation carries no quote mark of its
    # own, so check that rather than assume it. Upstream's title quotes live INSIDE the quotation and
    # are the reason the naive `split('"')[1]` failed here in the first place.
    assert '"' not in conditions[closing + 1 :], conditions
    return conditions[len(QUOTE_PREFIX) : closing]


def conditions_tail(conditions: str) -> str:
    """Everything after the upstream quotation closes -- this project's own prose, not upstream's."""
    return conditions[conditions.rindex('". ') + 3 :]


#: Every sentence this repository is allowed to append after an upstream quotation, verbatim. Three
#: strings for three situations, and nothing else may ship.
EXPECTED_CONDITION_TAILS = frozenset({
    "Upstream does not state what kind of uncertainty cRErr is, so unc_type is table; Geant4 itself "
    "never reads cRErr.",
    "No uncertainty is published upstream and this table carries no unc column, so unc_type is "
    "table.",
    "No uncertainty is published upstream and this table carries no unc column.",
    "No uncertainty is published upstream and this table carries no unc column, so unc_type is "
    "table. This entry is UNREACHABLE through GetMuonZeff, which clamps Z into [1, 100] before "
    "indexing. It ships because the dataset reproduces the array as declared, and silently dropping "
    "an element it claims to reproduce would be a worse artifact than shipping one with a "
    "disclosure.",
})


def is_upstream_verbatim(text: str, comment_lines: tuple[str, ...]) -> bool:
    """Is `text` exactly a space-join of whole `comment_lines` entries, in source order?

    That is how an attribution is built: the generator SELECTS comment lines and joins them, and
    never retypes, trims or reorders one. Checking the join rather than checking substrings is what
    makes a fabricated word impossible rather than merely awkward -- there is nowhere in the string
    left for one to sit. If a future selector legitimately needs part of a line, this is the
    assertion to revisit deliberately, not the one to loosen.

    What this does NOT certify, stated so nobody reads more into it: the selection is a subsequence,
    so a quotation may omit an upstream line and still pass. Every word is upstream's, whole lines,
    in order -- completeness of the attribution is a maintainer's judgement, not a checkable
    property, and `_quote_upstream`'s needles are where that judgement lives.

    The base case fires only immediately after a whole line, never after a separator, and the empty
    text is not a join of anything. Written the other way round -- `if position == len(text): return
    True` at the top -- it accepts a trailing space and accepts a quotation of nothing at all, both
    of which are exactly the "certified as upstream's, checked by nobody" failure this guard exists
    to remove. Checked against a brute-force enumeration of every in-order subsequence.
    """
    memo: dict[tuple[int, int], bool] = {}

    def walk(position: int, first: int) -> bool:
        if (position, first) not in memo:
            result = False
            for index in range(first, len(comment_lines)):
                line = comment_lines[index]
                # An empty comment line is not text and cannot carry an attribution; the extractor
                # drops them (measured: none in the vendored source), so skipping is defensive.
                if not line or not text.startswith(line, position):
                    continue
                end = position + len(line)
                if end == len(text) or (text[end] == " " and walk(end + 1, index + 1)):
                    result = True
                    break
            memo[position, first] = result
        return memo[position, first]

    return bool(text) and walk(0, 0)


def test_t53_parity_profile_layer2_invariants_hold_on_every_row():
    """What a `parity` profile is allowed to claim, asserted row by row on both tables."""
    found = extraction()
    zeff_audit = d1.load_zeff_audit(REPO / d1.ZEFF_AUDIT_RELPATH)
    tables = (
        (CAPTURE_LAYER1, CAPTURE_LAYER2, found.capture_comment_lines),
        (ZEFF_LAYER1, ZEFF_LAYER2, found.zeff_comment_lines),
    )
    for layer1_path, layer2_path, comment_lines in tables:
        table, document = committed(layer1_path, layer2_path)
        assert table.directives["PROFILE"] == spec.PARITY_PROFILE
        # A parity file must name the revision it reproduces, and it must be the one we vendored.
        assert table.directives["SOURCESHA"] == d1.UPSTREAM_COMMIT
        assert document.precedence == ("geant4-compiled-in",)
        assert document.version == table.directives["VERSION"]
        for key, row in document.rows.items():
            # The value came from the library, and the bibkey names the library -- not the papers
            # the library cites, which no one here has read. Those travel as quoted upstream text.
            assert row.source_library == "geant4-compiled-in", key
            assert row.source_bibkey == "geant4_v11_4_2", key
            # A parity profile reproduces; it does not recommend. That much is unconditional.
            assert row.recommendation == "", key
            # `needs_verification` is no longer unconditional on either table: a row settled
            # by a primary read carries false, and the locator's second clause is what says so.
            # The two must agree on every row, or one of them is decoration.
            established = "; isotope_resolved established by " in row.source_locator
            if layer2_path == ZEFF_LAYER2:
                read = "; printed cell read in " in row.source_locator
                assert row.needs_verification is not read, key
                assert not established, key
                z = int(key)
                assert read is (z in zeff_audit), key
                assert row.unc_type == (
                    "estimate" if z in zeff_audit and zeff_audit[z].underlined else "table"
                ), key
            else:
                assert row.needs_verification is not established, key
                assert row.unc_type == "table", key
            # Upstream says "weighted average of the two most precise measurements".
            assert row.single_source is False, key
            # The locator must resolve in THIS repository, at a real line of the vendored file.
            assert row.source_locator.startswith(d1.VENDORED_RELPATH + ":"), key
            line = int(row.source_locator.split(":")[1].split()[0])
            assert 1 <= line <= VENDORED.read_text("ascii").count("\n") + 1, key
            assert d1.UPSTREAM_BLOB_ID in row.source_locator, key
            # `conditions` carries UPSTREAM's words, marked as upstream's, never as our finding.
            assert row.conditions.startswith(QUOTE_PREFIX), key
        # Every quoted attribution is upstream's own words, WHOLE, and from this table's own comment
        # block. `conditions` is the one field a parity profile certifies as upstream's, so the
        # check has to leave no room at all: the full quoted span must be a space-join of complete
        # comment lines, in source order. What this replaced compared `". "`-split pieces, matching
        # each against any single comment line OR its first 40 characters against all of them --
        # which passed a fragment whose tail was fabricated past character 40, and never saw the
        # 191 characters of the zeff attribution that sit beyond upstream's own quote mark.
        for text in {quoted_attribution(row.conditions) for row in document.rows.values()}:
            assert is_upstream_verbatim(text, comment_lines), (
                f"{text!r} is not a verbatim join of {layer1_path.name}'s upstream comment lines: "
                f"{comment_lines}"
            )
        # ...and the PROSE AFTER the quotation is pinned too. Until now it was constrained only to
        # contain no `"`, which left the majority of several `conditions` strings -- including the
        # sentence that tells a consumer this zeff entry is unreachable -- free text that any
        # rewrite could alter with nothing failing. `conditions` is generated from a fixed template
        # with exactly one variable part, so the tail is a pure function of the producer and there
        # is no reason for it to be open. Enumerated, not pattern-matched: a regex here would be a
        # second place for the wording to live.
        tails = {conditions_tail(row.conditions) for row in document.rows.values()}
        assert tails <= EXPECTED_CONDITION_TAILS, (
            f"{layer2_path.name} carries unpinned prose after the upstream quotation: "
            f"{sorted(tails - EXPECTED_CONDITION_TAILS)!r}. If this wording changed deliberately, "
            "the ruling is to update EXPECTED_CONDITION_TAILS in the same commit -- not to relax "
            "this assertion."
        )


def test_t54_isotope_resolved_is_the_audit_and_says_which_rule_produced_it():
    """The one non-obvious boolean, now carried by the audit rather than derived from this table.

    The rule this replaced was mechanical: true if and only if the Z carried more than one capture
    record. Its soundness argument was about the Z -- two differing rates at one Z do show the
    underlying data distinguishes isotopes -- and it was applied to each ROW of that Z, which does
    not follow: one of those rows can still be the natural-composition entry. So the flag is no
    longer derivable from this file at all, and the test's job is to check that every row's flag is
    the audited one and that the row SAYS which rule produced it.
    """
    audit = d1.load_isotope_audit(REPO / d1.AUDIT_RELPATH)
    _, capture_document = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)

    assert {f"{z}-{a}" for z, a in audit} == set(capture_document.rows)
    for key, row in capture_document.rows.items():
        z, a = (int(part) for part in key.split("-"))
        finding = audit[(z, a)]
        assert row.isotope_resolved is finding.isotope_resolved, key
        # The evidence itself is carried into the shipped file, so a consumer never has to fetch
        # the audit to see why a flag says what it says.
        assert finding.evidence in row.evaluation_method, key
        marker = (
            "isotope_resolved was established by reading the primary literature"
            if finding.settled
            else "isotope_resolved is NOT established for this record"
        )
        assert marker in row.evaluation_method, key

    # The rule really did change something: if the audit agreed with the old mechanical derivation
    # on every row, this whole layer would be ceremony. The COUNT is a measurement and belongs in
    # the ledger, not in an assertion -- what is checked here is that the disagreement is non-empty
    # in both directions, which is the structural claim.
    per_z = extraction().capture_rows_per_z()
    mechanical = {key: per_z[int(key.split("-")[0])] > 1 for key in capture_document.rows}
    rows = capture_document.rows
    assert any(rows[k].isotope_resolved and not mechanical[k] for k in rows), (
        "the audit found no row the mechanical rule under-called"
    )
    assert any(mechanical[k] and not rows[k].isotope_resolved for k in rows), (
        "the audit found no row the mechanical rule over-called"
    )

    # An effective charge is per-Z: there is no isotope for it to be resolved to, on any row.
    _, zeff_document = committed(ZEFF_LAYER1, ZEFF_LAYER2)
    assert not any(row.isotope_resolved for row in zeff_document.rows.values())
    assert all("per-Z quantity" in row.evaluation_method for row in zeff_document.rows.values())


def test_t55_every_shipped_bibkey_resolves():
    """No Layer-2 row may cite a key the bibliography does not define -- in ANY shipped dataset.

    This bites immediately rather than theoretically: the format example's rows cite
    `openmucf-format-spec`, which resolved nowhere at all until this stage added it.
    """
    bib = REPO / "openmucf" / "data" / "references.bib"
    known = sources.bibkeys(bib)
    files = shipped_layer2_files()
    assert len(files) >= 3, "expected the example plus both D1 Layer-2 files"
    for path in files:
        document = provenance.from_json_obj(json.loads(path.read_bytes().decode("ascii")))
        for key, row in document.rows.items():
            assert row.source_bibkey in known, f"{path.name} row {key}: {row.source_bibkey!r}"

    # The fenced copy of `bibkeys` must not drift from the one in the rest of the package. The g4
    # subpackage keeps its own three-line regex so the data layer stays liftable; this is the check
    # that keeps the duplication honest rather than merely admitted.
    assert known == rates.bibkeys(bib)


def test_t57_mutation_drill_every_generated_artifact_is_actually_guarded():
    """Corrupt one digit in each generated artifact in turn; `--audit` must fail and name it.

    A byte-diff list is a claim that every file on it is watched. The only way to know is to break
    each one and check the alarm sounds -- an artifact accidentally left off the list, or one whose
    regeneration silently reproduces the corruption, passes every other test in this file.
    """
    d3dir = REPO / "data" / "g4" / "d3"
    artifacts = sorted(D1DIR.glob("d1_*.g4dat")) + sorted(D1DIR.glob("*.prov.json")) + [
        D1DIR / "geant4_add_dataset.snippet"
    ] + sorted(d3dir.glob("d3_*.g4dat")) + sorted(d3dir.glob("*.prov.json")) + [d3dir / "validation.csv"]
    # The files found on disk are exactly the ones the generator writes: a generated file the
    # globs miss, or a stray file they catch, would make this drill prove less than it claims.
    assert set(artifacts) == set(generator_module().build_dataset_artifacts()[0]), [p.name for p in artifacts]

    def audit() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(GENERATOR), "--audit"], capture_output=True, text=True, cwd=REPO
        )

    assert audit().returncode == 0, "the drill cannot start from a dirty tree"
    for path in artifacts:
        original = path.read_bytes()
        try:
            path.write_bytes(_flip_one_digit(original))
            result = audit()
            assert result.returncode != 0, (
                f"corrupting {path.name} did not fail the audit; the Layer-2 file is rebuilt from "
                f"{d1.AUDIT_RELPATH}: check that file first"
            )
            assert path.name in result.stdout + result.stderr, (
                f"the audit failed but never named {path.name}: {result.stdout}{result.stderr}; "
                f"the Layer-2 file is rebuilt from {d1.AUDIT_RELPATH}: check that file first"
            )
        finally:
            path.write_bytes(original)
    assert audit().returncode == 0, "the drill did not restore the tree"


def _flip_one_digit(payload: bytes) -> bytes:
    """Change exactly one decimal digit, leaving the file the same length and still well-formed."""
    for index, byte in enumerate(payload):
        if 0x30 <= byte <= 0x38 and payload[index - 1 : index] not in (b"\n", b"#"):
            return payload[:index] + bytes([byte + 1]) + payload[index + 1 :]
    raise AssertionError("no digit to corrupt")


def test_t58_the_generator_version_is_coupled_to_every_dataset_it_stamped():
    """`#GENERATOR` embeds `openmucf.__version__`, in all three shipped Layer-1 files.

    That is deliberate -- a consumer holding a broken file needs to know which tool made it -- and
    the cost is that a version bump moves these bytes and both archive MD5s. The coupling is made
    loud HERE, with the remedy in the message, rather than discovered as a red audit at tag time.
    """
    stamped = f"openmucf-g4 {openmucf.__version__}"
    files = [CAPTURE_LAYER1, ZEFF_LAYER1, REPO / "data" / "g4" / "example.g4dat"]
    for path in files:
        table = spec.parse(path.read_bytes().decode("ascii"))
        assert table.directives["GENERATOR"] == stamped, (
            f"openmucf.__version__ has moved but {path.relative_to(REPO).as_posix()} was not "
            "regenerated: run `python scripts/generate_g4data.py` and commit data/g4/example.g4dat, "
            "data/g4/d1/*.g4dat AND both geant4_add_dataset.snippet files (their MD5SUMs change too)"
        )
    # The D1 files also carry no CR, for the same reason the vendored source does not: `data/g4/d1/`
    # needs its own `-text` line, because a gitattributes `*` does not cross a `/`.
    for path in files[:2] + [CAPTURE_LAYER2, ZEFF_LAYER2]:
        assert b"\r" not in path.read_bytes(), f"the checkout rewrote {path.name}"


# --------------------------------------------------------------------------------------------
# T-59..T-62 -- the isotope audit: what it is allowed to claim
# --------------------------------------------------------------------------------------------

#: The copies of a paper this project distinguishes. A locator that does not say WHICH copy was
#: read is a locator that cannot be re-checked: the scanned preprint and the published article are
#: different documents with different pagination, and this dataset was built from the preprint.
KNOWN_COPIES = frozenset({"preprint-scan", "arxiv-preprint", "published-pdf"})


def audit_rows():
    return d1.load_isotope_audit(REPO / d1.AUDIT_RELPATH)


def test_t59_the_audit_covers_every_capture_record_exactly_once():
    """Coverage derived from the vendored source, never from the audit's own length.

    Both directions matter. A missing key would leave a record with a flag nobody derived; an extra
    key is a row somebody audited that this dataset does not ship, which means the audit was
    written against a different table than the one in the repository.
    """
    found = extraction()
    keys = [(z, a) for z, a, _, _ in found.capture_records]
    audit = audit_rows()

    assert len(keys) == len(set(keys)), "the vendored source has a duplicate (Z, A)"
    assert set(audit) == set(keys)
    assert len(audit) == len(keys)

    table, _ = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)
    assert set(audit) == {(int(z), int(a)) for z, a, _, _ in table.records}


def test_t60_a_resolved_row_carries_a_locator_and_names_the_copy_that_was_read():
    """`isotope_resolved: true` with nothing behind it is the failure this column pair prevents."""
    audit = audit_rows()
    resolved = [row for row in audit.values() if row.isotope_resolved]
    assert resolved, "an audit that resolves nothing would make every check below vacuous"

    for row in audit.values():
        where = f"({row.z}, {row.a})"
        assert row.evidence, where
        if row.isotope_resolved:
            assert row.locator, where
            assert row.copy_read, where
        # A locator and a copy travel together in both directions -- see the loader, which refuses
        # the file outright if they do not. Asserted here as well because this is the property a
        # reader of the audit is entitled to rely on, not an implementation detail of the loader.
        assert bool(row.locator) == bool(row.copy_read), where
        if row.copy_read:
            assert row.copy_read in KNOWN_COPIES, f"{where}: unknown copy {row.copy_read!r}"
        # Every locator names a table or a section AND a page: "the paper" is not a locator.
        if row.locator:
            assert re.search(r"\b(Table|abstract)\b", row.locator), f"{where}: {row.locator!r}"
            assert re.search(r"\bp\.\d+", row.locator), f"{where}: {row.locator!r}"


def test_t61_settled_rows_say_settled_and_unsettled_rows_still_say_open():
    """The audit and the shipped Layer 2 must agree about which questions are still open."""
    audit = audit_rows()
    _, document = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)

    unsettled = {key for key, row in audit.items() if not row.settled}
    assert unsettled, (
        "every row settled would be a stronger claim than this primary supports; if that is "
        "genuinely the finding, it is a ruling to record, not a test to delete"
    )
    for (z, a), finding in audit.items():
        row = document.rows[f"{z}-{a}"]
        assert row.needs_verification is not finding.settled, (z, a)
        # An unsettled row may not claim resolution: "not established" is the whole point.
        if not finding.settled:
            assert finding.isotope_resolved is False, (z, a)
            assert row.isotope_resolved is False, (z, a)


def test_t62_every_primary_the_audit_cites_resolves_in_the_bibliography():
    """Nothing enters the audit that is not in `references.bib` with a DOI or a URL to reach it.

    The match is on the bibkey the audit's locator names, so a locator citing a paper
    nobody added to the bibliography fails rather than passing as free text.
    """
    bib_text = (REPO / "openmucf" / "data" / "references.bib").read_text("utf-8")
    entries = {
        key: body
        for key, body in re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib_text, re.S)
    }
    assert entries, "the bibliography parsed to nothing; the regex, not the data, is wrong"

    # A prefix match is not enough: any `Suzuki*` entry would satisfy it regardless of which paper
    # it is. The audit's locator names an author and a year, so require BOTH, and require the
    # matched entry to carry the year in its own `year =` field -- that is what ties the citation
    # to the paper rather than to a surname.
    cited = {
        (match.group(1).lower(), match.group(2))
        for match in (re.match(r"([A-Za-z]+) (\d{4})", row.locator)
                      for row in audit_rows().values() if row.locator)
        if match
    }
    settled = [row for row in audit_rows().values() if row.locator]
    assert len(cited) >= 1 and cited, "no locator names a primary at all"
    assert all(re.match(r"[A-Za-z]+ \d{4}", row.locator) for row in settled), (
        "every locator must open with an author and a year so it can be tied to a bibliography entry"
    )
    for surname, year in sorted(cited):
        matches = [
            key for key, body in entries.items()
            if key.lower().startswith(surname) and re.search(rf"year\s*=\s*\{{?{year}\}}?", body)
        ]
        assert matches, f"the audit cites {surname!r} ({year}) but no bibliography entry matches both"
        for key in matches:
            body = entries[key]
            assert re.search(r"\b(doi|url)\s*=", body, re.I), f"{key} has neither a DOI nor a URL"


# --------------------------------------------------------------------------------------------
# T-63 -- the document's published counts ARE the shipped data's counts
# --------------------------------------------------------------------------------------------

#: Atomic numbers for the elements the primary's quoted sentence names. Reference data, not a
#: count: WHICH elements the sentence names is read out of the quotation in the document itself,
#: so the "nine" and the "seven" below are both derived rather than asserted here.
SYMBOL_Z = {
    "Ca": 20, "Cr": 24, "Ni": 28, "U": 92, "Pu": 94, "Cu": 29, "Sr": 38, "Br": 35, "Cl": 17,
}

#: Counts the document spells in words rather than digits.
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def _stated(text: str) -> int:
    """The integer a document fragment states, in digits or in words."""
    cleaned = text.strip().lower()
    if cleaned in NUMBER_WORDS:
        return NUMBER_WORDS[cleaned]
    return int(cleaned.replace(",", "").replace(" ", "").replace(" ", ""))


@dataclasses.dataclass(frozen=True)
class DocumentPins:
    """The collapsed documents and the pin tables checked against them, as `document_pins` built
    them -- so the prose coverage check can reuse every pattern without restating one."""

    doc: str
    changelog: str
    readme: str
    tools_readme: str
    claims: list
    rounded: list
    changelog_claims: list
    readme_claims: list
    tools_readme_claims: list
    changelog_rounded: list
    #: Section 9's cross-check table, one `(what, pattern, expected_row)` per row the two capture
    #: profiles disagree on; the whole row is the pinned span.
    crosscheck_rows: list
    #: Section 9's account of the keys the cross-check never compares: one `(what, pattern,
    #: expected_string)` row when the `mizuno2025` profile carries a key no `parity` record
    #: partners, its expected string the sorted keys as the document lists them; empty otherwise.
    crosscheck_unpartnered: list
    #: `(what, pattern, expected_string)` rows of `CHANGELOG.md` whose value is a string read from
    #: a shipped file rather than a count.
    string_claims: list
    #: Section 6's comparison table, one `(what, pattern, expected_row)` per capture row the audit
    #: leaves open, the compiled-in literals against every printed cell at its Z; the whole row is
    #: the pinned span.
    open_row_comparisons: list
    #: Section 6's two sentences naming the one record the value comparison settled, `(what,
    #: pattern, expected_key)` rows whose expected string is that record's key as the document
    #: writes it, derived from the audit and the cells.
    settled_by_value: list


def zeff_covered_split(zs, zeff_table) -> tuple[set[int], set[int], set[int]]:
    """The effective-charge entries at the capture table's Z, and their split between the
    primary's Table III and Table IV: `(covered, in Table III, in Table IV)`. One expression,
    called by `document_pins` and by T-94, so the two derivations cannot drift apart."""
    covered = {int(z) for z, _ in zeff_table.records if int(z) in zs}
    covered_iv = {z for z in covered if z >= 10}
    return covered, covered - covered_iv, covered_iv


def document_pins() -> DocumentPins:
    """This is the guard the D1 chain was missing, and its absence was measured rather than supposed:
    nothing in this repository read `DATASET_D1.md`, so a falsified count in it passed the entire
    battery, the byte-diff audit included. Counting claims in that document have shipped wrong --
    each a number updated to match a rewrite instead of re-derived from the data it describes -- and
    they would have failed here.

    **No expected value is written down.** Each is computed -- most from
    `data/g4/d1/isotope_audit.csv`, the committed `.g4dat` tables or the vendored source, and a few
    from the documents' own structure, which are named below -- and the document is then required to
    state that computed value. It is the discipline the extraction already obeys (T-42), applied to
    the prose that describes it: a count written down is a count that drifts.

    **What is hard-coded here is selectors, never expected values.** Several string and numeric
    constants pick out subsets of the audit -- the locator substrings that name the primary, the
    evidence openings that mark each isotope-resolution route, the atomic number where the primary's
    second table starts -- and they are visible in the code below rather than inventoried here,
    because two attempts at an exact inventory in this docstring have each been wrong. `SYMBOL_Z` is
    atomic numbers, reference data. What matters is the invariant: every expected value in `claims`,
    `rounded`, `changelog_claims` and `readme_claims` is a computed expression, never typed in.

    **This is not a census of these documents' numbers, and does not claim to be.** Two earlier
    revisions did claim it, in two different forms -- "everything countable from what ships is here",
    then a four-group exclusion list -- and adversarial passes falsified both. The lesson taken is
    that a completeness claim over prose is unbounded and will keep being wrong, so this test no
    longer makes one. What is pinned below is pinned; other numbers in those documents are not, and
    adding a pin is always in order. Some are structurally out of reach whatever the effort -- F-3's
    contraction figures need a Geant4 build and two compiler configurations, F-2's `NaN` and `+inf`
    need C++ division semantics that CPython raises on, every claim about what the primary *prints*
    rests on a paper this repository does not redistribute, and the dysprosium rounding tie has no
    shipped source at all -- but that list is an illustration, not an inventory.

    **Some pins here are not independent of what they check, and are named so nobody counts them as
    more than they are.** They are named, deliberately, without a running total: every enumeration
    of this docstring's own contents has eventually gone stale, each time because a later edit was
    correct and an earlier sentence counting it was left behind. The
    effective-charge coverage restates the attribution finding's set equality rather than re-deriving
    it, because the primary does not ship. The abundance figures are read from the audit's own
    hand-authored `evidence` strings, so they hold the prose to the CSV and check neither against an
    external table. Section 5's finding partition is counted from the document's own `F-n` headings,
    so it checks that the summary agrees with the section rather than with anything that ships. And
    the count of elements the primary's sentence names is read out of the quotation in that same
    document: only the *seven of them carrying a separated-isotope record* reaches the audit, while
    the *nine* is the document checked against itself.

    The ruling if this fails after a deliberate rewording: move the anchor in the same commit, never
    the expected value, and never delete a row.
    """
    doc = " ".join((REPO / "DATASET_D1.md").read_text(encoding="utf-8").split())

    found = extraction()
    audit = audit_rows()
    table, _ = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)
    zeff_table, _ = committed(ZEFF_LAYER1, ZEFF_LAYER2)

    keys = {(z, a) for z, a, _, _ in found.capture_records}
    zs = sorted({z for z, _ in keys})
    trues = {k for k, r in audit.items() if r.isotope_resolved}
    settled = {k for k, r in audit.items() if r.settled}
    unsettled = set(audit) - settled
    separated = {k for k, r in audit.items()
                 if r.evidence.startswith("the primary lists the separated isotope")}
    mononuclidic = {k for k, r in audit.items() if "is mononuclidic" in r.evidence}
    # Derived from which paper settled the row, NOT from `trues`. Defining it as "the resolved rows
    # at Z=1 and Z=2" made the partition assertion below a tautology exactly where the document
    # counts three: no resolved row there could fall outside the union however its evidence read,
    # and stripping both helium rows of any recognisable route still passed. The carve-out is a
    # claim about the SOURCE -- the two papers Geant4's comment carves hydrogen and helium out to --
    # so it is selected on the locator, which is the same evidence the document cites for it.
    carve_out = {k for k, r in audit.items() if r.settled and "Suzuki" not in r.locator}
    natural = (set(audit) - trues) & settled
    not_most_abundant = {k for k, r in audit.items() if "most abundant nuclide" in r.evidence}

    # The three routes are counted separately below, so assert here that they are what the document
    # calls them -- a partition of the true rows. Counting them as three independent numbers would
    # let a row with separated-isotope evidence and a false flag inflate the 23 without touching the
    # 45, and every count would still agree.
    assert separated | mononuclidic | carve_out == trues, (
        "the three isotope-resolution routes no longer cover exactly the resolved rows: "
        f"uncovered {sorted(trues - (separated | mononuclidic | carve_out))}, "
        f"outside the resolved set {sorted((separated | mononuclidic | carve_out) - trues)}"
    )
    assert len(separated) + len(mononuclidic) + len(carve_out) == len(trues), (
        "the three isotope-resolution routes overlap; the document presents them as a partition"
    )

    per_z = {z: sum(1 for k in keys if k[0] == z) for z in zs}
    old_rule = {k: per_z[k[0]] > 1 for k in keys}
    disagree = {k for k in keys if old_rule[k] != audit[k].isotope_resolved}
    under_called = {k for k in disagree if not old_rule[k]}
    contradicted = {k for k in disagree if old_rule[k] and k in settled}
    unestablished = {k for k in disagree if old_rule[k] and k in unsettled}

    # The Z whose `locator` column -- not merely whose evidence prose -- names a table of the
    # primary the set equality is against. F-4 claims the comparison can be repeated from the
    # shipped files, and this is what "repeated" costs.
    located_in_primary = {
        z for z in zs
        if any("Suzuki" in r.locator and ("Table III" in r.locator or "Table IV" in r.locator)
               for k, r in audit.items() if k[0] == z)
    }
    # The primary tabulates an effective charge in BOTH of its per-nuclide tables. Table III's
    # first column is headed Z(Zeff) for the light nuclides Z = 1-9 exactly as Table IV's is from
    # Z = 10 up, under the same "Zeff is taken from ref. 77" footnote. An earlier revision counted
    # only the Z >= 10 half here and the document then told the reader the other nine fell to the
    # fallback branch -- which the vendored comment's own "and if not present from" makes false.
    # So the covered set is the whole Z set F-4's equality is against, and because the document
    # now states the split between the two tables, both halves are pinned as well.
    zeff_covered, zeff_covered_iii, zeff_covered_iv = zeff_covered_split(zs, zeff_table)
    zeff_uncovered = len(zeff_table.records) - len(zeff_covered)

    # Section 3's re-ordering disclosure, both halves. "Misplaced record" is an adjacent descent;
    # "inverted pair" is the standard combinatorial sense. They are different numbers, which is what
    # the disclosure got wrong before, so both are pinned.
    source_order = [(z, a) for z, a, _, _ in found.capture_records]
    misplaced = sum(1 for i in range(1, len(source_order))
                    if source_order[i] < source_order[i - 1])
    inverted_pairs = sum(
        1
        for i in range(len(source_order))
        for j in range(i + 1, len(source_order))
        if source_order[i] > source_order[j]
    )
    inverted_z = {z for z, _ in source_order}.intersection(
        {source_order[i][0] for i in range(1, len(source_order))
         if source_order[i] < source_order[i - 1]}
    )

    # F-5's shape claims, from the committed effective-charge table.
    zeff = {int(z): float(v) for z, v in zeff_table.records}
    descents = [z for z in range(1, len(zeff)) if zeff[z] < zeff[z - 1]]
    step_into_81 = round(zeff[81] - zeff[80], 2)
    preceding_steps = [round(zeff[z] - zeff[z - 1], 2) for z in (78, 79, 80)]

    # Which elements the primary's sentence names is read out of the document's own quotation. Be
    # precise about what that buys: the SEVEN reaches the shipped audit, because carrying a
    # separated-isotope record is a fact about the CSV. The NINE does not -- it is the document
    # checked against itself, and is listed among this test's non-independences for that reason.
    quotation = re.search(r"> Now for muon capture (.+?) Read it precisely", doc)
    assert quotation, "the quoted section-IV sentence is no longer where this test reads it"
    # Every one- or two-letter capitalised token in that quotation is an element symbol -- the
    # longer words in it ("Primakoff", "Goulard") cannot match. Requiring set equality with
    # SYMBOL_Z means an element ADDED to the quotation fails here rather than passing unnoticed,
    # which a lookup keyed only on the known symbols would have allowed.
    quoted_symbols = set(re.findall(r"\b[A-Z][a-z]?\b", quotation.group(1)))
    assert quoted_symbols == set(SYMBOL_Z), (
        f"the quoted sentence names {sorted(quoted_symbols)}; SYMBOL_Z knows "
        f"{sorted(SYMBOL_Z)}. Extend SYMBOL_Z in the same commit that changes the quotation."
    )
    named = {sym: z for sym, z in SYMBOL_Z.items()
             if re.search(rf"\b{sym}\b", quotation.group(1))}
    carrying = {sym for sym, z in named.items() if any(k[0] == z for k in separated)}

    # F-6 does not only count the unsettled rows, it says where they sit. Counting alone would
    # leave the load-bearing half of that sentence unchecked.
    assert {k[0] for k in unsettled} <= set(named.values()), (
        "DATASET_D1.md F-6 says every record this dataset cannot settle sits at an element the "
        f"primary's sentence names; the audit has open rows at Z="
        f"{sorted({k[0] for k in unsettled} - set(named.values()))}, which it does not."
    )

    # The section-5 partition, derived from the section's own structure.
    findings = len(re.findall(r"\*\*F-\d+ ", doc))
    # Counted as F-n headings where the marker OPENS ITS OWN SENTENCE, which is load-bearing twice
    # over. A revision that matched any heading ending in the marker counted "NOT SETTLED against
    # the primary." as settled; a revision that excluded only the literal "NOT " still counted
    # "UNSETTLED", "NOT YET SETTLED", "Not SETTLED" and "NEVER SETTLED" -- so the document could
    # declare a finding open while its own summary counted it closed. Requiring the preceding
    # sentence to have ended closes the whole negation family rather than the spellings thought of.
    settled_findings = len(
        re.findall(r"\*\*F-\d+ [^*]*\. SETTLED against the primary\.\*\*", doc)
    )

    # The free-muon decay rate comes from the VENDORED SOURCE, never from the prose under test.
    # Reading it out of the document would let a self-consistent falsification -- move the constant
    # and the count together -- pass, which is the "recomputing the header value it just read"
    # failure this file's own preamble warns about.
    free_muon = re.search(
        r"\{\s*0,\s*0,\s*([0-9.]+),\s*[0-9.]+\s*\}\s*//\s*free muon", VENDORED.read_text()
    )
    assert free_muon, "the free-muon decay rate is no longer where this test reads it in the source"
    # us^-1 -> ns^-1 through the module's own constant, which it derives from CLHEP's chain rather
    # than asserting; a second literal here would be a second place the unit convention lives.
    decay_rate = float(free_muon.group(1)) / d1.MICROSECOND

    coefficients = dict(found.fallback_coefficients)
    model = d1.GoulardPrimakoff(
        b0a=float(coefficients["b0a"]), b0b=float(coefficients["b0b"]),
        b0c=float(coefficients["b0c"]), t1=float(coefficients["t1"]),
        xmu_coeff=float(coefficients["xmu_coeff"]), mix=float(coefficients["mix"]),
        zmin=int(coefficients["zmin"]), zmax=int(coefficients["zmax"]),
        zeff=tuple(found.zeff),
    )
    oracle = read_oracle()
    max_ulp_subset = subset_max_ulp(found, model, oracle)
    swept = negative = negative_past_decay = table_hits = 0
    first_negative: dict[int, int] = {}
    for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1):
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1):
            swept += 1
            # Counted inside the sweep, not taken as `len(records)`. Section 4's claim is that the
            # sweep COVERS the table, which is only the record count if every key is distinct and
            # every one lands inside the box -- two preconditions nothing else here asserts.
            table_hits += (z, a) in keys
            value = d1.capture_rate(z, a, found.capture_records, model)
            if value < 0:
                negative += 1
                first_negative.setdefault(z, a)
                if abs(value) > decay_rate:
                    negative_past_decay += 1

    claims = [
        ("capture record count, file table",
         r"\| `d1_capture\.g4dat` \| (\d+) records", len(table.records)),
        ("effective-charge record count, file table",
         r"\| `d1_zeff\.g4dat` \| (\d+) records", len(zeff_table.records)),
        ("capture record count, section 1",
         r"The table has (\d+) records spanning", len(found.capture_records)),
        ("distinct Z, section 1", r"records spanning (\d+) distinct Z", len(zs)),
        ("distinct Z, F-4 headline", r"this table spans \*\*(\d+) distinct Z\*\*", len(zs)),
        ("distinct Z, F-4 set equality",
         r"span \*\*exactly the same (\d+) distinct Z\*\*", len(zs)),
        ("Z whose locator names a table of the primary",
         r"\*\*(\d+) of them carry that table and page", len(located_in_primary)),
        ("effective-charge entries the primary covers",
         r"\*\*(\d+) of the \d+ `zeff` entries", len(zeff_covered)),
        ("effective-charge entries in total, F-5",
         r"\*\*\d+ of the (\d+) `zeff` entries", len(zeff_table.records)),
        ("effective-charge entries the primary's Table IV covers",
         r"(\d+) in Table IV", len(zeff_covered_iv)),
        ("effective-charge entries the primary's Table III covers",
         r"(\d+) in Table III", len(zeff_covered_iii)),
        ("effective-charge entries not covered", r"the remaining (\d+) \(Z = 0", zeff_uncovered),
        ("settled rows", r"\*\*(\d+) of the \d+ are settled", len(settled)),
        ("records checked, section 6", r"\*\*\d+ of the (\d+) are settled", len(audit)),
        ("open rows", r"`needs_verification: false`; (\d+) are not", len(unsettled)),
        ("isotope_resolved true", r"\*\*(\d+) are `isotope_resolved: true`\*\*", len(trues)),
        ("separated-isotope route", r"(\d+) because the primary lists a separated isotope",
         len(separated)),
        ("mononuclidic route", r"(\d+) because the element is mononuclidic", len(mononuclidic)),
        ("hydrogen and helium carve-out route",
         r"and (\d+) — the hydrogen and helium records", len(carve_out)),
        ("natural-composition rows, section 6",
         r"The remaining \*\*(\d+) are `false` as an established finding\*\*", len(natural)),
        ("natural-composition rows, F-7", r"For \*\*(\d+) of the \d+ records\*\* the primary shows",
         len(natural)),
        ("rows whose A is not the element's most abundant nuclide",
         r"In (\d+) of those, `A` is not even", len(not_most_abundant)),
        ("rows the old rule disagrees with",
         r"\*\*(\d+) of the \d+ records\*\*, and the three ways", len(disagree)),
        ("records the old rule was applied to",
         r"\*\*\d+ of the (\d+) records\*\*, and the three ways", len(audit)),
        ("rows the old rule under-called",
         r"\* \*\*(\d+)\*\* the rule called unresolved", len(under_called)),
        ("rows the primary flatly contradicts",
         r"\* \*\*(\d+)\*\* the primary flatly contradicts", len(contradicted)),
        ("rows the primary fails to establish",
         r"\* \*\*(\d+)\*\* — (?:`\(\d+, \d+\)`,? )+— where the primary",
         len(unestablished)),
        ("elements the primary's sentence names", r"Of the (\w+) the sentence names", len(named)),
        ("named elements carrying a separated-isotope record",
         r"the sentence names, \*\*(\w+) carry at", len(carrying)),
        ("unsettled records sitting at a named element",
         r"\*\*all (\w+) of the records this dataset cannot settle", len(unsettled)),
        ("findings in section 5", r"Section 5 carries (\w+) findings", findings),
        ("findings that are defects", r"findings: \*\*(\w+)\*\* defects",
         findings - settled_findings),
        ("findings the primary settled", r"corrects, and \*\*(\w+)\*\* questions",
         settled_findings),
        ("swept points, section 4", r"= (\d+) points\*\*", swept),
        ("swept points returning a negative rate, section 2",
         r"returns λ_c < 0 on (\d+) of the \d+ swept points", negative),
        ("swept points in total, section 2",
         r"returns λ_c < 0 on \d+ of the (\d+) swept points", swept),
        ("swept points returning a negative rate, F-1",
         r"capture rates\.\*\* (\d+) of the \d+ swept points", negative),
        ("swept points in total, F-1",
         r"capture rates\.\*\* \d+ of the (\d+) swept points", swept),
        ("negative points past the free-muon decay rate",
         r"For the \*\*(\d+)\*\* swept points where λ_c is negative", negative_past_decay),
        ("records out of place in the upstream declaration order",
         r"— \*\*(\w+) misplaced record", misplaced),
        ("inverted pairs in the upstream declaration order",
         r"misplaced record, (\w+) inverted pairs\*\*", inverted_pairs),
        ("distinct Z whose locator does not name the primary",
         r"The (\w+) that do not are", len(zs) - len(located_in_primary)),
        ("effective charges below their predecessor",
         r"rises monotonically except at exactly (\w+) steps", len(descents)),
        ("declared fallback inputs", r"All (\w+) inputs are declared",
         len(found.fallback_coefficients)),
        ("effective-charge array length, section 3", r"The array holds (\d+) entries",
         len(zeff_table.records)),
        ("the maximum Z, section 1", r"\"94 entries\"; (\d+) is the maximum", max(zs)),
        ("table hits inside the sweep, section 4", r"The (\d+) table hits are included",
         table_hits),
    ]

    #: Figures the document rounds. `(what, pattern, computed value, decimal places)`.
    def _pct(key: tuple[int, int], nuclide: str) -> float:
        match = re.search(rf"{nuclide} is ([\d.]+)% of natural", audit[key].evidence)
        assert match, f"the audit row {key} no longer states {nuclide}'s natural abundance"
        return float(match.group(1))

    lead = re.search(r"most abundant nuclide \(Pb-208, ([\d.]+)%\)", audit[(82, 207)].evidence)
    assert lead, "the audit row (82, 207) no longer states Pb-208's natural abundance"

    rounded = [
        ("Sm-150's share of natural samarium",
         r"Sm-150 is \*\*([\d.]+) %\*\*", _pct((62, 150), "Sm-150"), 1),
        ("Sn-119's share of natural tin",
         r"Sn-119 is \*\*([\d.]+) %\*\*", _pct((50, 119), "Sn-119"), 1),
        ("Pb-208's share of natural lead",
         r"which is (\d+) % of natural lead", float(lead.group(1)), 0),
        ("the effective charge at Z=81", r"Z=81→82 \(([\d.]+) →", zeff[81], 2),
        ("the effective charge at Z=82", r"Z=81→82 \([\d.]+ → ([\d.]+)\)", zeff[82], 2),
        ("the effective charge at Z=83", r"Z=82→83 \([\d.]+ → ([\d.]+)\)", zeff[83], 2),
        ("the step into Z=81", r"anomalously large, \+([\d.]+) against", step_into_81, 2),
        ("the smallest neighbouring step",
         r"neighbours of \+(\d+\.\d+) to", min(preceding_steps), 2),
        ("the largest neighbouring step",
         r"neighbours of \+\d+\.\d+ to \+(\d+\.\d+)", max(preceding_steps), 2),
        ("the effective charge at Z=56", r"`zeff\[56\] = ([\d.]+)`", zeff[56], 2),
        ("caesium's effective charge", r"caesium's ([\d.]+) \(Z = 55\)", zeff[55], 2),
        ("lanthanum's effective charge", r"lanthanum's ([\d.]+)", zeff[57], 2),
    ]

    for what, pattern, expected in claims:
        hits = re.findall(pattern, doc)
        assert len(hits) == 1, (
            f"DATASET_D1.md: the anchor for {what} matched {len(hits)} times, expected exactly one. "
            f"The wording moved; move the anchor in the same commit rather than deleting this row. "
            f"Pattern: {pattern!r}"
        )
        assert _stated(hits[0]) == expected, (
            f"DATASET_D1.md states {hits[0]!r} for {what}; the shipped data says {expected}. "
            "The document is wrong, not this test -- every expected value here is computed, most "
            "from isotope_audit.csv, the committed .g4dat tables or the vendored source and a few "
            "from the documents own structure, and none is written down."
        )

    # `CHANGELOG.md` restates the same counts for a reader who never opens the dataset document --
    # a second copy of every number, on a file nothing else in this repository reads, and one where
    # a count has already shipped wrong once: its account of the old rule's failures survived a
    # round after the dataset document's had been corrected.
    changelog = " ".join((REPO / "CHANGELOG.md").read_text(encoding="utf-8").split())
    # The cross-check between the two capture profiles (T-91), derived from the two shipped files.
    crosscheck_pairs = mizuno_parity_pairs()
    crosscheck_disagreements = [pair for pair in crosscheck_pairs if not pair["agrees"]]
    # Released lines are held by dated registry rows; only `[Unreleased]` lines may be pinned live.
    changelog_claims: list = []
    # A string the changelog states about a shipped file, read from that file: the dataset version
    # the entry names is the `#VERSION` the committed capture table carries.
    shipped_version = re.search(r"^#VERSION\s+(\S+)$", CAPTURE_LAYER1.read_text("ascii"), re.M)
    assert shipped_version, "the committed capture table declares no #VERSION"
    # The one record the value comparison settled: the settled audit row `decided_by_value` decides,
    # derived from the audit and the printed cells. Every sentence naming it carries this key.
    blocks = d1.cells_by_z(capture_cells())
    settled_by_value_keys = sorted(
        key for key, finding in audit.items()
        if finding.settled and key[0] in blocks
        and d1.decided_by_value(key, finding.evidence, blocks[key[0]])
    )
    assert len(settled_by_value_keys) == 1, (
        "the sentences below name one record settled by the comparison; if this count moves, the "
        f"document is rewritten, not this test: {settled_by_value_keys}"
    )
    ((settled_z, settled_a),) = settled_by_value_keys
    key_text = f"`({settled_z}, {settled_a})`"
    string_claims = [
        ("the dataset version the entry names",
         r"moves the dataset's `#VERSION` to (\d+\.\d+\.\d+)", shipped_version.group(1)),
        ("the record the value comparison settled",
         r"the (`\(\d+, \d+\)`) record is settled as the separated isotope the primary lists",
         key_text),
    ]

    # `README.md` is the third copy of these numbers and the one a reader meets first. Its G4
    # section restates the table sizes, the sweep and the finding count for a reader who never opens
    # either document, and until this test grew to reach it nothing in the repository read it
    # either -- the same condition that let the counting claims ship wrong in the first place.
    readme = " ".join((REPO / "README.md").read_text(encoding="utf-8").split())
    readme_claims = [
        ("capture record count",
         r"compiled in\*\*: (\d+) `\{Z, A, rate, error\}` records", len(found.capture_records)),
        ("effective-charge record count",
         r"(\d+)-value effective-charge table", len(zeff_table.records)),
        ("swept points at zero ulp",
         r"over \*\*(\d+) \(Z, A\) points at zero ulp\*\*", swept),
        ("findings that are defects",
         r"surfaced (\w+) defects in the upstream seam", findings - settled_findings),
        ("swept points returning a negative rate",
         r"negative capture rates on (\d+) of those \d+ points", negative),
        ("swept points in total, restated",
         r"negative capture rates on \d+ of those (\d+) points", swept),
        ("maximum ulp over the diagnostic subset", r"points at (zero) ulp", max_ulp_subset),
    ]

    # `cpp/tools/README.md` restates the sweep size beside the contraction figures that only a
    # compiled producer can check (cpp/test/check_f3.py); the size is the one number there this
    # file computes.
    tools_readme = " ".join(
        (REPO / "cpp" / "tools" / "README.md").read_text(encoding="utf-8").split()
    )
    tools_readme_claims = [
        ("swept points, harvest tooling", r"with \d+ of the (\d+) swept points", swept),
    ]

    for what, pattern, value, places in rounded:
        hits = re.findall(pattern, doc)
        assert len(hits) == 1, (
            f"DATASET_D1.md: the anchor for {what} matched {len(hits)} times, expected exactly one. "
            f"Pattern: {pattern!r}"
        )
        expected = round(value, places) if places else float(round(value))
        assert float(hits[0]) == expected, (
            f"DATASET_D1.md states {hits[0]!r} for {what}; the audit says {value}, which rounds to "
            f"{expected} at {places} decimal place(s)."
        )

    for where, text, rows in (
        ("CHANGELOG.md", changelog, changelog_claims),
        ("README.md", readme, readme_claims),
        ("cpp/tools/README.md", tools_readme, tools_readme_claims),
    ):
        for what, pattern, expected in rows:
            hits = re.findall(pattern, text)
            assert len(hits) == 1, (
                f"{where}: the anchor for {what} matched {len(hits)} times, expected exactly one. "
                f"Move the anchor in the same commit rather than deleting this row. "
                f"Pattern: {pattern!r}"
            )
            assert _stated(hits[0]) == expected, (
                f"{where} states {hits[0]!r} for {what}; the shipped data says {expected}. "
                f"{where} is wrong, not this test."
            )

    for what, pattern, expected_string in string_claims:
        assert re.findall(pattern, changelog) == [expected_string], (
            f"CHANGELOG.md: {what}: expected exactly one match of {pattern!r} stating "
            f"{expected_string!r}, found {re.findall(pattern, changelog)}"
        )

    # Section 9's cross-check table: one row per pair on which the two capture profiles disagree
    # at the primary's printed precision, each row pinned whole -- the parity cells as the vendored
    # source prints them, the other profile's cells as its transcription prints them -- so every
    # digit in the table is the derived one and a row the derivation does not produce has no pin.
    crosscheck_rows = []
    for pair in crosscheck_disagreements:
        z, a_mizuno = pair["mizuno"]
        _, a_parity = pair["parity"]
        parity_value, parity_unc = pair["parity_literal"]
        printed_value, printed_unc = pair["printed"]
        row_text = (
            f"| {z} | {a_mizuno} | {a_parity} | {parity_value} ± {parity_unc} | "
            f"{printed_value} ± {printed_unc} | {pair['locator']} |"
        )
        crosscheck_rows.append((
            f"cross-check row: {mizuno2025.PROFILE} ({z}, {a_mizuno}) against parity ({z}, {a_parity})",
            "(" + re.escape(row_text) + ")",
            row_text,
        ))
    for what, pattern, expected_row in crosscheck_rows:
        assert re.findall(pattern, doc) == [expected_row], (
            f"DATASET_D1.md: {what}: the derived row {expected_row!r} must appear exactly once in "
            "section 9's table; the document is wrong, not this test"
        )

    # The keys the cross-check never compares: every `mizuno2025` key the partner map covers no
    # pair for, derived from the same two shipped files, and section 9 must name exactly those.
    unpartnered = sorted(set(mizuno_extraction().keys) - {pair["mizuno"] for pair in crosscheck_pairs})
    text = unpartnered_keys_text(unpartnered)
    crosscheck_unpartnered = [(
        "cross-check: the keys no parity record partners",
        "partner nothing and are not compared are (" + re.escape(text) + r")\.",
        text,
    )] if unpartnered else []
    for what, pattern, expected_row in crosscheck_unpartnered:
        assert re.findall(pattern, doc) == [expected_row], (
            f"DATASET_D1.md: {what}: the derived row {expected_row!r} must appear exactly once in "
            "section 9; the document is wrong, not this test"
        )

    # Section 6's comparison table: one row per capture row the audit leaves open, pinned whole --
    # the compiled-in literals as the vendored source prints them, every cell the primary prints at
    # that Z as the committed transcription prints it, and the outcome the comparison derives.
    literals = {
        (z, a): literal
        for (z, a, _, _), literal in zip(found.capture_records, found.capture_literals, strict=True)
    }
    values = {(z, a): (value, unc) for z, a, value, unc in found.capture_records}
    open_row_comparisons = []
    for z, a in sorted(unsettled):
        matches = d1.printed_matches(*values[(z, a)], blocks[z])
        literal_value, literal_unc = literals[(z, a)]
        row_text = (
            f"| ({z}, {a}) | {literal_value} +- {literal_unc} | {d1.render_cells(blocks[z])} | "
            f"{d1.render_outcome(matches)} |"
        )
        open_row_comparisons.append((
            f"comparison row: the open record ({z}, {a}) against the printed cells at Z={z}",
            "(" + re.escape(row_text) + ")",
            row_text,
        ))
    for what, pattern, expected_row in open_row_comparisons:
        assert re.findall(pattern, doc) == [expected_row], (
            f"DATASET_D1.md: {what}: the derived row {expected_row!r} must appear exactly once in "
            "section 6's table; the document is wrong, not this test"
        )
    # Section 6's two sentences that name the record the comparison settled, each exactly once.
    settled_by_value = [
        ("the record the value comparison settled, among the separated-isotope route",
         r"\(the (`\(\d+, \d+\)`) record among them settled to that entry by the comparison below\)",
         key_text),
        ("the record the value comparison settled, the comparison's own sentence",
         r"which is how the (`\(\d+, \d+\)`) record became `isotope_resolved`", key_text),
    ]
    for what, pattern, expected_key in settled_by_value:
        assert re.findall(pattern, doc) == [expected_key], (
            f"DATASET_D1.md: {what}: {expected_key} must appear exactly once as "
            f"{pattern!r}; the document is wrong, not this test"
        )
    # The bullet that lists the rows the primary fails to establish names exactly those keys, in
    # ascending order: the count above is pinned, and so is the list.
    listed = re.search(r"\* \*\*\d+\*\* — ((?:`\(\d+, \d+\)`,? )+)— where the primary", doc)
    assert listed, "DATASET_D1.md no longer lists the rows the primary fails to establish"
    listed_keys = [(int(z), int(a)) for z, a in re.findall(r"\((\d+), (\d+)\)", listed.group(1))]
    assert listed_keys == sorted(unestablished), (
        f"DATASET_D1.md lists {listed_keys} as the rows the primary fails to establish; the audit "
        f"derives {sorted(unestablished)}"
    )

    changelog_rounded = [
        ("extreme natural abundance", r"is ([\d.]+) % of the natural element",
         _pct((62, 150), "Sm-150"), 1),
    ]
    for what, pattern, value, places in changelog_rounded:
        hits = re.findall(pattern, changelog)
        assert len(hits) == 1, (
            f"CHANGELOG.md: the anchor for {what} matched {len(hits)} times, expected exactly one. "
            f"Pattern: {pattern!r}"
        )
        expected = round(value, places) if places else float(round(value))
        assert float(hits[0]) == expected, (
            f"CHANGELOG.md states {hits[0]!r} for {what}; the audit says {value}, which rounds to "
            f"{expected} at {places} decimal place(s)."
        )

    # The one physical constant the document prints. Derived from the vendored source above, so
    # this pins the printed value rather than trusting it.
    printed = re.search(r"free-muon decay rate \(([\d.e−+-]+) ns", doc)
    assert printed, "the free-muon decay rate is no longer printed where this test reads it"
    stated_rate = float(printed.group(1).replace("−", "-"))
    assert stated_rate == float(f"{decay_rate:.5g}"), (
        f"DATASET_D1.md prints the free-muon decay rate as {stated_rate}; the vendored source "
        f"declares {free_muon.group(1)} per microsecond, which is {decay_rate} per nanosecond."
    )

    # Section 3 names the upstream order at the misplaced record's Z; the numbers above say how many
    # pairs it inverts, this says which they are.
    stated_order = re.search(
        r"at Z=(\d+) the source declares A in the order (\d+), (\d+), (\d+), (\d+)", doc
    )
    assert stated_order, "section 3 no longer states the upstream order where this test reads it"
    inverted_at = int(stated_order.group(1))
    assert {inverted_at} == inverted_z, (
        f"DATASET_D1.md says the misplaced record sits at Z={inverted_at}; the vendored source puts "
        f"it at Z={sorted(inverted_z)}."
    )
    assert [int(g) for g in stated_order.groups()[1:]] == [
        a for z, a in source_order if z == inverted_at
    ], (
        f"DATASET_D1.md states the Z={inverted_at} order as {stated_order.groups()[1:]}; the "
        f"vendored source declares {[a for z, a in source_order if z == inverted_at]}."
    )

    # F-1's per-Z thresholds, and the one rate it prints in full.
    thresholds = re.search(
        r"the first negative A is (\d+) for Z=(\d+), (\d+) for Z=(\d+), (\d+) for Z=(\d+), "
        r"(\d+) for Z=(\d+), (\d+) for Z=(\d+), (\d+) for Z=(\d+) and (\d+) for Z=(\d+)",
        doc,
    )
    assert thresholds, "F-1 no longer lists its per-Z thresholds where this test reads them"
    pairs = [int(g) for g in thresholds.groups()]
    for stated_a, stated_z in zip(pairs[0::2], pairs[1::2], strict=True):
        assert first_negative.get(stated_z) == stated_a, (
            f"DATASET_D1.md says the first negative A at Z={stated_z} is {stated_a}; the sweep "
            f"says {first_negative.get(stated_z)}."
        )

    tritium = re.search(r"λ_c = (−[\d.]+e−\d+) ns", doc)
    assert tritium, "F-1 no longer prints the tritium rate where this test reads it"
    assert tritium.group(1).replace("−", "-") == (
        f"{d1.capture_rate(1, 3, found.capture_records, model):.6e}"
    ), "DATASET_D1.md's tritium rate is not what the reference implementation returns"

    degenerate = re.search(r"`Z = -1, A = 12` returns (−[\d.]+e−\d+)", doc)
    assert degenerate, "F-2 no longer prints the reachable degenerate value where this test reads it"
    assert degenerate.group(1).replace("−", "-") == (
        f"{model.evaluate_unchecked(-1, 12):.6e}"
    ), "DATASET_D1.md's Z=-1 value is not what the reference implementation returns"

    # The two provenance identities the document tells a reader to check the vendored copy against.
    # A wrong digit here does not merely misinform, it sends someone to the wrong upstream object.
    quoted_blob = re.search(r"\*\*git blob id\*\* `([0-9a-f]{40})`", doc)
    assert quoted_blob, "section 1 no longer prints the upstream blob id where this test reads it"
    assert quoted_blob.group(1) == d1.UPSTREAM_BLOB_ID, (
        f"DATASET_D1.md prints the blob id {quoted_blob.group(1)}; the pin is {d1.UPSTREAM_BLOB_ID}"
    )
    shipped_sha = re.search(r"#SOURCESHA\s+(\S+)", CAPTURE_LAYER1.read_text(encoding="ascii"))
    quoted_sha = re.search(r"#SOURCESHA `?([0-9a-f]{40})", doc)
    assert quoted_sha, "section 1 no longer prints the source revision where this test reads it"
    assert quoted_sha.group(1) == shipped_sha.group(1), (
        f"DATASET_D1.md prints #SOURCESHA {quoted_sha.group(1)}; the shipped table declares "
        f"{shipped_sha.group(1)}"
    )

    # The model contract quotes the shipped directive as a code block. Quoting it wrongly would
    # hand a consumer coefficients the dataset does not declare.
    quoted_fallback = re.search(r"```\s*(#FALLBACK goulard_primakoff [^`]+?)\s*```", doc)
    assert quoted_fallback, "section 2 no longer quotes the fallback directive where this test reads it"
    shipped_fallback = re.search(
        r"(#FALLBACK\s+\S+.*)", CAPTURE_LAYER1.read_text(encoding="ascii")
    )
    assert quoted_fallback.group(1).split() == shipped_fallback.group(1).split(), (
        f"DATASET_D1.md quotes {quoted_fallback.group(1)!r}; the shipped table declares "
        f"{shipped_fallback.group(1)!r}"
    )

    # `zeff[0] ships and is unreachable` names the clamp the model applies.
    clamp = re.search(r"clamps its argument into `\[(\d+), (\d+)\]`", doc)
    assert clamp, "section 3 no longer states the clamp range where this test reads it"
    assert (int(clamp.group(1)), int(clamp.group(2))) == (model.zmin, model.zmax), (
        f"DATASET_D1.md says GetMuonZeff clamps into [{clamp.group(1)}, {clamp.group(2)}]; the "
        f"declared directive says [{model.zmin}, {model.zmax}]"
    )

    # F-1 names its two call sites by the vendored files' own identifiers; both must still exist
    # in the copies the dataset is measured against.
    assert "ApplyYourself(" in VENDORED.read_text("ascii")
    assert "ConstructMuonicAtom(" in HELPER.read_text("ascii")

    # F-4's gap list is the evidence for its set equality, so it is checked as a set rather than as
    # a string: the document writes runs as ranges, and how it spells them is not the claim.
    gap_text = re.search(r"the same gaps at Z = (.+?)\. That is a set equality", doc)
    assert gap_text, "F-4 no longer lists its gaps where this test reads them"
    quoted_gaps: set[int] = set()
    for piece in re.split(r",| and ", gap_text.group(1)):
        piece = piece.strip()
        if not piece:
            continue
        run = re.fullmatch(r"(\d+)[–-](\d+)", piece)
        quoted_gaps.update(range(int(run.group(1)), int(run.group(2)) + 1) if run else [int(piece)])
    computed_gaps = {z for z in range(min(zs), max(zs) + 1) if z not in zs}
    assert quoted_gaps == computed_gaps, (
        f"DATASET_D1.md lists gaps {sorted(quoted_gaps)}; the shipped table has "
        f"{sorted(computed_gaps)}"
    )

    # F-7 restates three of F-1's thresholds. Pinned separately so the two statements cannot drift
    # apart while each looks right on its own.
    restated = re.search(
        r"thresholds are small: A=(\d+) at Z=(\d+), A=(\d+) at Z=(\d+), A=(\d+) at Z=(\d+)", doc
    )
    assert restated, "F-7 no longer restates F-1's low-Z thresholds where this test reads them"
    restated_pairs = [int(g) for g in restated.groups()]
    for stated_a, stated_z in zip(restated_pairs[0::2], restated_pairs[1::2], strict=True):
        assert first_negative.get(stated_z) == stated_a, (
            f"DATASET_D1.md F-7 says the first negative A at Z={stated_z} is {stated_a}; the sweep "
            f"says {first_negative.get(stated_z)}"
        )

    return DocumentPins(
        doc, changelog, readme, tools_readme, claims, rounded, changelog_claims, readme_claims,
        tools_readme_claims, changelog_rounded, crosscheck_rows, crosscheck_unpartnered,
        string_claims, open_row_comparisons, settled_by_value,
    )


def test_t63_the_documents_published_counts_are_the_shipped_datas_counts():
    """Every pin table `document_pins` builds is checked as it is built; this test is that run."""
    document_pins()


# --------------------------------------------------------------------------------------------
# T-82 -- the shipped D1 archive unpacks to the dataset directory, with README and History
# --------------------------------------------------------------------------------------------


def generator_module():
    """`scripts/generate_g4data.py`, loaded by path -- `scripts/` is a directory, not a package."""
    module_spec = importlib.util.spec_from_file_location("generate_g4data", GENERATOR)
    assert module_spec and module_spec.loader, GENERATOR
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_t82_the_d1_archive_unpacks_to_the_dataset_directory_with_readme_and_history():
    """What the archive the snippet checksums actually holds, opened rather than described.

    Every member sits under the one directory Geant4's dataset machinery expects after unpacking,
    no member is a directory entry, every stored name fits the ustar name field, the README carries
    the attribution notice DATASET_D1.md states and each table's record count as the committed
    Layer-2 file holds it, the History names the dataset and its version, and the archive's MD5
    is the MD5SUM the committed snippet declares.
    """
    generator = generator_module()
    _, archive = generator.build_dataset_artifacts()
    directory = emit.dataset_directory(generator.DATASET_NAME, generator.DATASET_VERSION)
    pairs = (
        (CAPTURE_LAYER1, CAPTURE_LAYER2), (ZEFF_LAYER1, ZEFF_LAYER2), (MIZUNO_LAYER1, MIZUNO_LAYER2),
        (generator.D3_KSHELL_LAYER1, generator.D3_KSHELL_LAYER2),
        (generator.D3_LEVELS_LAYER1, generator.D3_LEVELS_LAYER2),
    )
    committed = [path.name for pair in pairs for path in pair]
    with tarfile.open(fileobj=io.BytesIO(archive)) as opened:
        entries = opened.getmembers()
        names = [entry.name for entry in entries]
        assert names == sorted(f"{directory}/{x}" for x in committed + ["README", "History"])
        assert not any(entry.isdir() for entry in entries), names
        assert all(len(name.encode("ascii")) <= 100 for name in names), names  # ustar name field
        readme = opened.extractfile(f"{directory}/README").read().decode("ascii")
        history = opened.extractfile(f"{directory}/History").read().decode("ascii")

    notice = [
        line[2:] for line in (REPO / "DATASET_D1.md").read_text("utf-8").splitlines()[:8]
        if line.startswith("> ")
    ]
    assert len(notice) == 2, notice
    for line in notice:
        assert line in readme.splitlines(), line
    for layer1, layer2 in pairs:
        document = provenance.from_json_obj(json.loads(layer2.read_bytes().decode("ascii")))
        # On the README line that names THIS table's file, not anywhere in the document: two
        # tables with the same count would otherwise vouch for each other.
        (line,) = [text for text in readme.splitlines() if text.startswith(f"  - {layer1.name}:")]
        assert line.endswith(f", {len(document.rows)} records"), line
    assert history.splitlines()[0] == f"History for {generator.DATASET_NAME} files:"
    assert generator.DATASET_VERSION in history

    snippet = (D1DIR / "geant4_add_dataset.snippet").read_text("ascii")
    declared = re.search(r"^\s*MD5SUM\s+([0-9a-f]{32})$", snippet, re.MULTILINE)
    assert declared, snippet
    assert emit.tarball_md5(archive) == declared.group(1)


# ---------------------------------------------------------------------------------------
# T-85..T-86 -- the natural-row corpus members, and the zero-rows oracle fixtures
# ---------------------------------------------------------------------------------------


def test_t85_natural_row_corpus_members_and_the_shipped_tables():
    """The three corpus members that carry an `A = 0` record are OK to Layer 1 (their expected.tsv
    rows, read here), and the consumer's rule sorts them: admissible under `A:natural_and_listed`,
    refused under `A:listed`; the shipped parity tables carry no natural row at all."""
    corpus = REPO / "tests" / "fixtures" / "g4dat_conformance"
    expected = {}
    for line in (corpus / "expected.tsv").read_bytes().decode("ascii").splitlines():
        member, code, _line = line.split("\t")
        expected[member] = code
    members = {
        suffix: corpus / f"ok_natural_row_under_{suffix}.g4dat"
        for suffix in ("natural_and_listed", "listed", "parity")
    }
    for path in members.values():
        assert path.is_file(), path
        assert expected[path.name] == "OK", (path.name, expected[path.name])

    def table(path):
        return spec.parse(path.read_bytes().decode("ascii"))

    assert spec.natural_rows(table(members["natural_and_listed"])) == 1
    assert spec.natural_rows(table(members["parity"])) == 1
    with pytest.raises(ValueError, match="natural_and_listed"):
        spec.natural_rows(table(members["listed"]))

    for layer1, layer2 in ((CAPTURE_LAYER1, CAPTURE_LAYER2), (ZEFF_LAYER1, ZEFF_LAYER2)):
        shipped, _document = committed(layer1, layer2)
        assert spec.natural_rows(shipped) == 0, layer1.name


def test_t86_zero_rows_fixtures_are_copies_of_shipped_oracle_lines():
    """Every line of every zero-rows fixture is `#END`, the shipped oracle's `# sweep` or
    `# fullsweep_sha256` line byte for byte, or a row present verbatim in the shipped oracle; and
    the per-section census is what each fixture's name says."""
    shipped = (D1DIR / "d1_gp_sweep.oracle").read_bytes().split(b"\n")
    header = {line for line in shipped if line.startswith((b"# sweep ", b"# fullsweep_sha256 "))}
    assert len(header) == 2, header
    rows = {line for line in shipped if line and not line.startswith(b"#")}
    assert rows

    def section(line: bytes) -> str:
        for prefix, name in ((b"ZEFFCLAMP ", "ZEFFCLAMP"), (b"ZEFF ", "ZEFF"), (b"RATE ", "RATE")):
            if line.startswith(prefix):
                return name
        return "subset"

    by_fixture = {"subset": "subset", "zeff": "ZEFF", "rate": "RATE", "clamp": "ZEFFCLAMP"}
    fixtures = REPO / "tests" / "fixtures" / "g4dat_zero_rows"
    cases = [("d1_gp_sweep", None), *((f"zero_{key}", name) for key, name in by_fixture.items())]
    for stem, empty in cases:
        raw = (fixtures / f"{stem}.oracle").read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n"), stem
        census = dict.fromkeys(by_fixture.values(), 0)
        for line in raw.split(b"\n")[:-1]:
            assert line == b"#END" or line in header or line in rows, (stem, line)
            if line in rows:
                census[section(line)] += 1
        wanted = {name: (0 if empty is None or name == empty else 1) for name in census}
        assert census == wanted, (stem, census)


# --------------------------------------------------------------------------------------------
# T-87, T-88 -- the v11.5.0.beta copies: upstream's bytes, carrying the same tables as v11.4.2
# --------------------------------------------------------------------------------------------

#: The `D1Extraction` fields that are source positions, not content. The beta re-sorted its
#: include blocks, so every position is free to move while every table stays where it is.
POSITION_FIELDS = frozenset({"capture_lines", "zeff_lines"})


def assert_same_tables(found: d1.D1Extraction, reference: d1.D1Extraction) -> None:
    """Every `D1Extraction` field but the source positions, compared one at a time and named.

    Field by field, so a failure says which table moved -- `capture_records`, `zeff`, a comment
    line, a fallback coefficient -- rather than that two large records differ somewhere.
    """
    for field in dataclasses.fields(d1.D1Extraction):
        if field.name in POSITION_FIELDS:
            continue
        left, right = getattr(found, field.name), getattr(reference, field.name)
        assert left == right, f"{field.name} differs between the two vendored copies"


def beta_pair(copy: d1.SourceCopy) -> tuple[d1.D1Extraction, d1.D1Extraction]:
    """`(beta, v11.4.2)` extractions of one compiled-in copy, each straight from its vendored file."""
    paths = {d1.BOUND_DECAY.name: (BETA_BOUND_DECAY, VENDORED), d1.HELPER.name: (BETA_HELPER, HELPER)}
    beta_path, reference_path = paths[copy.name]
    return (
        d1.extract(beta_path.read_text("ascii"), copy),
        d1.extract(reference_path.read_text("ascii"), copy),
    )


@pytest.mark.parametrize("path", [BETA_BOUND_DECAY, BETA_HELPER], ids=lambda p: p.name)
def test_t87_the_beta_copies_are_the_pinned_upstream_blobs(path: pathlib.Path):
    """Each beta copy is upstream's file at the beta commit, proven by upstream's own object name,
    with the sha256 recorded alongside and no CR byte -- the same three guards T-40 and T-41 put
    on the v11.4.2 BoundDecay copy and T-69 puts on the v11.4.2 helper copy, so the beta directory
    holds exactly the two pinned files."""
    data = path.read_bytes()
    assert b"\r" not in data, (
        "the checkout rewrote the vendored beta file's line endings: check that .gitattributes "
        "still carries `third_party/geant4/** -text`"
    )
    assert git_blob_id(data) == BETA_BLOB_IDS[path.name], (
        "the vendored beta file is not the pinned upstream blob at the beta commit"
    )
    assert hashlib.sha256(data).hexdigest() == BETA_SHA256S[path.name]
    assert {p.name for p in BETA_DIR.iterdir() if p.is_file()} == set(BETA_BLOB_IDS)


@pytest.mark.parametrize("copy", [d1.BOUND_DECAY, d1.HELPER], ids=lambda c: c.name)
def test_t87_the_beta_tables_equal_the_v11_4_2_tables_field_by_field(copy: d1.SourceCopy):
    """The tables compiled into the beta are the tables compiled into v11.4.2: every extracted
    field but the source positions is equal, for both compiled-in copies. This is what lets the
    dataset's `#SOURCESHA` stay at the revision it was generated from while the overlay targets
    both revisions."""
    beta, reference = beta_pair(copy)
    assert_same_tables(beta, reference)


def test_t87_the_reference_model_fed_the_beta_values_reproduces_the_oracle_digest():
    """The reference implementation, fed the beta BoundDecay's records, effective charges and
    fallback coefficients, reproduces the full-sweep digest the oracle harvested from the v11.4.2
    build."""
    beta, _ = beta_pair(d1.BOUND_DECAY)
    coefficients = beta.coefficients
    model = d1.GoulardPrimakoff(
        b0a=float(coefficients["b0a"]), b0b=float(coefficients["b0b"]),
        b0c=float(coefficients["b0c"]), t1=float(coefficients["t1"]),
        xmu_coeff=float(coefficients["xmu_coeff"]), mix=float(coefficients["mix"]),
        zmin=int(coefficients["zmin"]), zmax=int(coefficients["zmax"]),
        zeff=tuple(beta.zeff),
    )
    expected = read_oracle()["header"]["fullsweep_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", expected), expected
    assert d1.sweep_digest(beta.capture_records, model) == expected


def _mutated_on_line(text: str, lineno: int, literal: str) -> str:
    """`text` with the first occurrence of `literal` on line `lineno` given one more digit."""
    lines = text.split("\n")
    assert literal in lines[lineno - 1], (lineno, literal)
    lines[lineno - 1] = lines[lineno - 1].replace(literal, literal + "1", 1)
    return "\n".join(lines)


def test_t88_drill_a_changed_beta_capture_rate_is_named_as_capture_records():
    """Alter one `capRates` literal of the beta text in memory: the field comparison must fail,
    and name `capture_records` -- the table the change belongs to, not a later field."""
    beta, reference = beta_pair(d1.BOUND_DECAY)
    text = BETA_BOUND_DECAY.read_text("ascii")
    mutated = _mutated_on_line(text, beta.capture_lines[0], beta.capture_literals[0][0])
    assert mutated != text
    with pytest.raises(AssertionError, match=r"\Acapture_records differs"):
        assert_same_tables(d1.extract(mutated, d1.BOUND_DECAY), reference)


def test_t88_drill_a_changed_beta_zeff_is_named_as_zeff():
    """Alter one `zeff` literal of the beta text in memory: the field comparison must fail, and
    name `zeff` -- with every capture field still equal, so the name is the changed table's."""
    beta, reference = beta_pair(d1.BOUND_DECAY)
    text = BETA_BOUND_DECAY.read_text("ascii")
    mutated = _mutated_on_line(text, beta.zeff_lines[0], beta.zeff_literals[0])
    assert mutated != text
    with pytest.raises(AssertionError, match=r"\Azeff differs"):
        assert_same_tables(d1.extract(mutated, d1.BOUND_DECAY), reference)


# --------------------------------------------------------------------------------------------
# T-98 -- the fallback expression compiled into the beta is token-identical to the v11.4.2 one
# --------------------------------------------------------------------------------------------

#: The identifiers the Goulard-Primakoff block must carry: the coefficients T-87 extracts, and
#: the rate the block assigns. Their presence is what proves the regex found the block and not a
#: fragment of it.
FALLBACK_IDENTIFIERS = ("b0a", "b0b", "b0c", "t1", "lambda")
#: The v11.4.2 copy and the beta copy of each compiled-in file, by file name.
FALLBACK_COPIES: dict[str, tuple[pathlib.Path, pathlib.Path]] = {
    VENDORED.name: (VENDORED, BETA_BOUND_DECAY),
    HELPER.name: (HELPER, BETA_HELPER),
}


def fallback_expression_tokens(text: str) -> list[str]:
    """The tokens of the Goulard-Primakoff block of one source text: from the `G4double b0a`
    declaration through the `lambda = t1 ...;` statement, comments stripped, tokenised as
    identifiers, numeric literals and single punctuation characters. Whitespace and comments are
    the only things the tokenisation forgets, so two copies with equal token lists compile the
    same expression tree and the same association -- what T-87's coefficient extraction does not
    reach."""
    stripped = re.sub(r"//[^\n]*", " ", re.sub(r"/\*.*?\*/", " ", text, flags=re.S))
    match = re.search(r"G4double\s+b0a\b.*?lambda\s*=\s*t1\b.*?;", stripped, flags=re.S)
    assert match is not None, "no `G4double b0a` ... `lambda = t1 ...;` block in the text"
    return re.findall(r"[A-Za-z_]\w*|\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\S", match.group(0))


def assert_same_tokens(found: list[str], expected: list[str]) -> None:
    """Equal token lists, or a message with the first differing index and the tokens around it."""
    for index, (left, right) in enumerate(zip(found, expected, strict=False)):
        if left != right:
            lo, hi = max(index - 3, 0), index + 4
            raise AssertionError(
                f"token {index} differs: found {left!r}, expected {right!r}; "
                f"found {found[lo:hi]}, expected {expected[lo:hi]}"
            )
    assert len(found) == len(expected), (
        f"token lists differ in length at index {min(len(found), len(expected))}: "
        f"{len(found)} found, {len(expected)} expected"
    )


@pytest.mark.parametrize("name", sorted(FALLBACK_COPIES))
def test_t98_the_fallback_expression_is_token_identical_across_the_vendored_revisions(name: str):
    """For each compiled-in copy, the beta's Goulard-Primakoff block tokenises to exactly the
    v11.4.2 block's tokens, and both carry the coefficient and rate identifiers."""
    reference, beta = FALLBACK_COPIES[name]
    expected = fallback_expression_tokens(reference.read_text("ascii"))
    found = fallback_expression_tokens(beta.read_text("ascii"))
    for identifier in FALLBACK_IDENTIFIERS:
        assert identifier in expected, (name, identifier)
        assert identifier in found, (name, identifier)
    assert_same_tokens(found, expected)


#: (label, old, new): one edit each to the beta text, in memory, that changes the expression
#: tree, the association or a literal factor while leaving every extracted coefficient in place.
T98_PROBES = [
    ("association", "t1 * zeff2 * zeff2", "t1 * (zeff2 * zeff2)"),
    ("2 * (A - Z) -> 3 * (A - Z)", "2 * (A - Z)", "3 * (A - Z)"),
    ("A * 4 -> A * 5", "G4double(A * 4)", "G4double(A * 5)"),
    ("(r2 * r2) -> (r2 * r2 * r2)", "(r2 * r2)", "(r2 * r2 * r2)"),
]


@pytest.mark.parametrize("name", sorted(FALLBACK_COPIES))
def test_t98_drill_each_expression_edit_is_named(name: str):
    """Each probe applied to the beta text one at a time fails the token comparison with the
    first differing token named; the unedited beta text passes."""
    reference, beta = FALLBACK_COPIES[name]
    expected = fallback_expression_tokens(reference.read_text("ascii"))
    text = beta.read_text("ascii")
    assert_same_tokens(fallback_expression_tokens(text), expected)
    for label, old, new in T98_PROBES:
        assert text.count(old) == 1, (name, label, text.count(old))
        with pytest.raises(AssertionError, match=r"token \d+ differs|differ in length") as raised:
            assert_same_tokens(fallback_expression_tokens(text.replace(old, new, 1)), expected)
        print(f"{name} {label}: {raised.value}")


# --------------------------------------------------------------------------------------------
# T-89 -- the Layer-2 vocabularies have one home: the specification's cells restate the package's
# --------------------------------------------------------------------------------------------

FORMAT_SPEC = REPO / "FORMAT_SPEC.md"


def layer2_vocabulary_cells(text: str) -> dict[str, tuple[str, ...]]:
    """The backticked tokens in the value cell of each per-row field row of `FORMAT_SPEC.md`
    section 3, keyed by field name -- read from the section's own table, never from memory."""
    start = text.index("\n## 3. ")
    end = text.index("\n## 4. ", start)
    section = text[start:end]
    cells: dict[str, tuple[str, ...]] = {}
    for line in section.splitlines():
        match = re.match(r"^\| `(\w+)` \| (?:string|bool) \| (.*) \|$", line)
        if match:
            cells[match.group(1)] = tuple(re.findall(r"`([^`]*)`", match.group(2)))
    return cells


def test_t89_the_specifications_vocabulary_cells_are_exactly_the_packages_tuples():
    """`source_library` and `unc_type` each have two homes -- `provenance.py`'s tuple and the
    specification's table cell -- and a token added to one and not the other is a value the
    reference implementation accepts and the specification does not admit, or the reverse. The
    cell is held to the tuple, in order, so neither home can drift."""
    cells = layer2_vocabulary_cells(FORMAT_SPEC.read_text("utf-8"))
    assert cells["source_library"] == provenance.SOURCE_LIBRARIES
    assert cells["unc_type"] == provenance.UNC_TYPES


def test_t89_drill_a_token_dropped_from_either_cell_is_refused():
    """Drop the last token from each cell of an in-memory copy of the specification: the cell no
    longer equals the tuple, and the check names the field by failing on it."""
    text = FORMAT_SPEC.read_text("utf-8")
    for field, vocabulary in (
        ("source_library", provenance.SOURCE_LIBRARIES),
        ("unc_type", provenance.UNC_TYPES),
    ):
        last = f", `{vocabulary[-1]}`"
        row = next(line for line in text.splitlines() if line.startswith(f"| `{field}` |"))
        assert row.count(last) == 1, (field, row)
        mutated = text.replace(row, row.replace(last, ""), 1)
        assert mutated != text
        assert layer2_vocabulary_cells(mutated)[field] == vocabulary[:-1]
        assert layer2_vocabulary_cells(mutated)[field] != vocabulary


# --------------------------------------------------------------------------------------------
# T-90 -- the mizuno2025 transcriptions: every structural rule of the loader fires on a fixture
# --------------------------------------------------------------------------------------------

MIZUNO_TABLE1 = REPO / mizuno2025.TABLE1_RELPATH
MIZUNO_TABLE3 = REPO / mizuno2025.TABLE3_RELPATH


def mizuno_extraction() -> mizuno2025.Mizuno2025Extraction:
    return mizuno2025.load(MIZUNO_TABLE1, MIZUNO_TABLE3)


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)


def _first_data_line(text: str) -> str:
    return text.split("\n")[1]


#: (label, which file, mutation of that file's text, message the loader must give). Each row is
#: one rule of `mizuno2025.load`; the shipped files pass every one (the first test below).
MIZUNO_DRILLS = [
    ("table 1 header renamed", 1, lambda t: _replace_once(t, "huff_factor", "huff"), "header is"),
    ("table 3 header reordered", 3, lambda t: _replace_once(t, "Z,A,", "A,Z,"), "header is"),
    ("a CR byte", 1, lambda t: t.replace("\n", "\r\n", 1), "contains CR"),
    ("a non-ASCII cell outside size_mm", 1,
     lambda t: _replace_once(t, "Powder in case,\u03d515.0\u00d72.8,0.500,2.02",
                             "Powder in c\u00e2se,\u03d515.0\u00d72.8,0.500,2.02"),
     "must be ASCII"),
    ("a non-decimal rate", 3, lambda t: _replace_once(t, ",0.893,", ",0.893x,"),
     "digits, a point and digits"),
    ("a non-decimal weight", 1, lambda t: _replace_once(t, ",9.00,", ",9.00g,"),
     "digits, a point and digits"),
    ("a minus sign before a rate", 3, lambda t: _replace_once(t, ",0.893,", ",-0.893,"),
     "digits, a point and digits"),
    ("a plus sign before a rate", 3, lambda t: _replace_once(t, ",0.893,", ",+0.893,"),
     "digits, a point and digits"),
    ("an underscore inside a rate", 3, lambda t: _replace_once(t, ",0.8794,", ",0.87_94,"),
     "digits, a point and digits"),
    ("a leading space before a rate", 3, lambda t: _replace_once(t, ",0.808,", ", 0.808,"),
     "digits, a point and digits"),
    ("an exponent in a rate", 3, lambda t: _replace_once(t, ",0.713,", ",7.13e-1,"),
     "digits, a point and digits"),
    ("a weight without a point", 1, lambda t: _replace_once(t, ",9.00,", ",9,"),
     "digits, a point and digits"),
    ("NaN as a lifetime", 1, lambda t: _replace_once(t, ",743.9,5.0,", ",NaN,5.0,"),
     "digits, a point and digits"),
    ("an exponent in a Suzuki cell", 1,
     lambda t: _replace_once(t, ",756.0,1.0,0.882,", ",756.0,1e0,0.882,"), "digits, a point and digits"),
    ("a zero uncertainty", 3, lambda t: _replace_once(t, ",0.009,", ",0.000,"), "greater than zero"),
    ("a non-integer Z", 3, lambda t: _replace_once(t, "\n47,0,", "\nAg,0,"), "must be integers"),
    ("a signed Z", 3, lambda t: _replace_once(t, "\n47,0,", "\n+47,0,"), "must be integers"),
    ("a leading zero in A", 3, lambda t: _replace_once(t, "\n14,28,", "\n14,028,"), "must be integers"),
    ("a duplicate key", 3, lambda t: _replace_once(t, "\n14,29,", "\n14,28,"), "duplicate key"),
    ("a label only in table 1", 1, lambda t: _replace_once(t, "\n55Mn,", "\n55mn,"), "labels differ"),
    ("a label only in table 3", 3, lambda t: _replace_once(t, ",natAg,", ",natag,"), "labels differ"),
    ("an averaged row without its footnote", 3,
     lambda t: _replace_once(t, ",natSi,0.8794,0.0018,Average of two experimental data in Table 1.,",
                             ",natSi,0.8794,0.0018,,"),
     "note is empty"),
    ("a single-row value that differs from table 1", 3,
     lambda t: _replace_once(t, ",28Si,0.893,0.009,", ",28Si,0.894,0.009,"), "string for string"),
    ("a single-row uncertainty that differs from table 1", 3,
     lambda t: _replace_once(t, ",55Mn,3.90,0.08,", ",55Mn,3.90,0.09,"), "string for string"),
    ("a footnote on a single-row nuclide", 3,
     lambda t: _replace_once(t, ",natMg,0.4856,0.0018,,", ",natMg,0.4856,0.0018,averaged,"),
     "note is set"),
    ("an empty locator", 3,
     lambda t: _replace_once(t, ',"Table 3, Exp. column",arxiv-html\n12,', ",,arxiv-html\n12,"),
     "carry a locator"),
    ("an empty copy_read", 1,
     lambda t: _replace_once(t, "0.893,0.009,Table 1,arxiv-html", "0.893,0.009,Table 1,"),
     "carry a copy_read"),
    ("one Suzuki cell without the other", 1,
     lambda t: _replace_once(t, ",756.0,1.0,0.882,", ",756.0,,0.882,"), "present or absent together"),
    ("no rows", 3, lambda t: t.split("\n")[0] + "\n", "carries no rows"),
]


def test_t90_the_shipped_transcriptions_load_and_the_two_tables_name_one_set_of_nuclides():
    """The committed files pass every rule; Table 3 has one row per nuclide label of Table 1 and
    every Table-1 row of a nuclide is reachable from its Table-3 row -- counts derived, not typed."""
    found = mizuno_extraction()
    labels = {row.nuclide for row in found.table1}
    order = [r.nuclide for r in found.table1].index
    assert [row.nuclide for row in found.table3] == sorted(labels, key=order)
    assert sum(len(found.table1_rows(row.nuclide)) for row in found.table3) == len(found.table1)
    assert len(set(found.keys)) == len(found.table3)
    for row in found.table3:
        printed = found.table1_rows(row.nuclide)
        assert (len(printed) >= 2) == bool(row.note), row.nuclide
        assert all(r.locator and r.copy_read for r in printed)
        assert row.locator and row.copy_read


@pytest.mark.parametrize("label, which, mutate, message", MIZUNO_DRILLS, ids=[d[0] for d in MIZUNO_DRILLS])
def test_t90_drill_each_loader_rule_refuses_its_fixture(tmp_path, label, which, mutate, message):
    """Corrupt one transcription in one way, in a temporary copy of both files; the loader must
    refuse it with the rule's own message. The unmutated copy loads, so the message is the rule's."""
    texts = {1: MIZUNO_TABLE1.read_bytes().decode("utf-8"), 3: MIZUNO_TABLE3.read_bytes().decode("utf-8")}
    paths = {n: tmp_path / p.name for n, p in ((1, MIZUNO_TABLE1), (3, MIZUNO_TABLE3))}
    for n, text in texts.items():
        paths[n].write_bytes(text.encode("utf-8"))
    mizuno2025.load(paths[1], paths[3])  # the copies load before the mutation
    mutated = mutate(texts[which])
    assert mutated != texts[which], label
    paths[which].write_bytes(mutated.encode("utf-8"))
    with pytest.raises(mizuno2025.Mizuno2025Error, match=re.escape(message)):
        mizuno2025.load(paths[1], paths[3])


# --------------------------------------------------------------------------------------------
# T-92 -- what the mizuno2025 profile is allowed to claim, asserted row by row on the shipped pair
# --------------------------------------------------------------------------------------------


def test_t92_mizuno2025_profile_layer2_invariants_hold_on_every_row():
    """The second capture profile's Layer 1 and Layer 2, against the transcriptions they were built
    from: the printed decimals are the records, and every provenance field is what the profile's
    rules say -- no upstream revision claimed, no fallback declared, every value read from the
    primary itself, and the key scheme carrying the isotope disclosure."""
    table, document = committed(MIZUNO_LAYER1, MIZUNO_LAYER2)
    found = mizuno_extraction()
    assert table.directives["PROFILE"] == mizuno2025.PROFILE == document.profile
    assert "SOURCESHA" not in table.directives
    assert "FALLBACK" not in table.directives
    assert document.precedence == (mizuno2025.PROFILE,)
    assert document.version == table.directives["VERSION"]
    assert spec.validity_assignments(table)["A"] == spec.A_NATURAL_AND_LISTED
    assert spec.natural_rows(table) == sum(1 for row in found.table3 if row.a == 0)
    assert spec.natural_rows(table) > 0

    by_key = {row.key: row for row in found.table3}
    assert len(table.records) == len(by_key)
    for z, a, value, unc in table.records:
        printed = by_key[(int(z), int(a))]
        # The record is the printed decimal, round-tripped through %.17g and nothing else.
        assert value == float(printed.rate) and unc == float(printed.rate_unc), (z, a)

    assert set(document.rows) == {f"{z}-{a}" for z, a in by_key}
    for key, row in document.rows.items():
        z, a = (int(part) for part in key.split("-"))
        printed = by_key[(z, a)]
        targets = found.table1_rows(printed.nuclide)
        assert row.source_library == mizuno2025.PROFILE, key
        assert row.source_bibkey == mizuno2025.BIBKEY, key
        assert row.unc_type == "exp", key
        assert row.evaluation_id == f"{mizuno2025.PROFILE}-table3", key
        assert row.recommendation == "", key
        assert row.needs_verification is False, key
        assert row.isotope_resolved is (a != 0), key
        assert "Table 3" in row.source_locator and "arxiv-html" in row.source_locator, key
        assert row.source_locator == f"{printed.locator} [copy read: {printed.copy_read}]", key
        assert row.single_source is (not any(r.has_suzuki_value for r in targets)), key
        assert ("natural composition" in row.validity_range) is (a == 0), key
        assert row.validity_range.startswith(f"Z={z} "), key
        assert (f'Table 3 footnote: "{printed.note}"' in row.conditions) is bool(printed.note), key
        # The averaged rows' method names the primary's footnote; no other row's does.
        assert ("footnote" in row.evaluation_method) is bool(printed.note), key
        if printed.note:
            assert printed.note in row.evaluation_method, key
        for target in targets:
            assert (
                f'"{target.form}", lifetime {target.lifetime_ns} ns, '
                f"uncertainty {target.lifetime_unc_ns} ns"
            ) in row.conditions, key
            # No parenthesised uncertainty survives: the notation is this profile's, not the primary's.
            assert f"({target.lifetime_unc_ns})" not in row.conditions, key
        assert row.evaluation_method == (
            mizuno2025.METHOD_AVERAGED.format(note=printed.note) if printed.note else mizuno2025.METHOD
        ), key
        assert mizuno2025.TABLE1_CAPTION in row.conditions and mizuno2025.COPY in row.conditions, key


def test_t92_the_shipped_directory_keys_one_pair_per_file_and_parity_carries_every_table():
    """The Python mirror of the reader's directory rule over the shipped dataset: every file is
    its own (profile, table) pair, `parity` carries every table any profile carries, and only
    the second capture profile carries natural-composition rows."""
    tables = spec.load_directory(D1DIR)
    files = sorted(path.name for path in D1DIR.glob("*.g4dat"))
    assert len(tables) == len(files), (sorted(tables), files)
    for profile, name in tables:
        assert (spec.PARITY_PROFILE, name) in tables, (profile, name)
    assert {profile for profile, _ in tables} == {spec.PARITY_PROFILE, mizuno2025.PROFILE}
    for (profile, _name), table in tables.items():
        natural = spec.natural_rows(table)
        assert (natural > 0) is (profile == mizuno2025.PROFILE), (profile, natural)


# --------------------------------------------------------------------------------------------
# T-91 -- the two capture profiles compared key by key, at the primary's printed precision
# --------------------------------------------------------------------------------------------


agrees_at_printed_precision = d1.agrees_at_printed_precision


def unpartnered_keys_text(keys: Sequence[tuple[int, int]]) -> str:
    """The keys no `parity` record partners, as section 9 lists them: each `Z-A` in backticks, a
    comma-separated run with `and` before the last. The one home of that rendering."""
    texts = [f"`{z}-{a}`" for z, a in keys]
    if len(texts) > 1:
        return ", ".join(texts[:-1]) + " and " + texts[-1]
    return "".join(texts)


def mizuno_parity_pairs() -> list[dict]:
    """The partner map between the `mizuno2025` keys and the `parity` records, and how each pair
    compares. A `(Z, A != 0)` key partners the parity record at `(Z, A)` when there is one; a
    `(Z, 0)` key partners every parity record at that Z whose Layer-2 row is not isotope-resolved,
    the compiled-in row that carries the element's natural-composition value under an isotope
    label. Both sides are read from the shipped files; nothing here is typed."""
    found = extraction()
    _, parity_document = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)
    mizuno = mizuno_extraction()
    values = {(z, a): (value, unc) for z, a, value, unc in found.capture_records}
    literals = {
        (z, a): literal
        for (z, a, _, _), literal in zip(found.capture_records, found.capture_literals, strict=True)
    }
    pairs = []
    for row in mizuno.table3:
        if row.a != 0:
            partners = [(row.z, row.a)] if (row.z, row.a) in values else []
        else:
            partners = sorted(
                (z, a) for (z, a) in values
                if z == row.z and not parity_document.rows[f"{z}-{a}"].isotope_resolved
            )
        for z, a in partners:
            value, unc = values[(z, a)]
            value_agrees = agrees_at_printed_precision(value, row.rate)
            unc_agrees = agrees_at_printed_precision(unc, row.rate_unc)
            pairs.append({
                "mizuno": (row.z, row.a),
                "parity": (z, a),
                "parity_literal": literals[(z, a)],
                "printed": (row.rate, row.rate_unc),
                "locator": row.locator,
                "value_agrees": value_agrees,
                "unc_agrees": unc_agrees,
                "agrees": value_agrees and unc_agrees,
            })
    # Ascending by the profile's own key, then by the partner's: the order the document tabulates.
    return sorted(pairs, key=lambda pair: (pair["mizuno"], pair["parity"]))


def test_t91_the_two_capture_profiles_are_compared_key_by_key_and_the_document_lists_the_disagreements():
    """The partner map has the shape the key scheme implies -- every natural-composition key
    partners exactly one compiled-in row, the enriched silicon isotopes partner the one compiled-in
    silicon row or nothing -- and the pairs whose value or uncertainty differ at the printed
    precision are printed here and are exactly the rows section 9 of DATASET_D1.md tabulates."""
    pairs = mizuno_parity_pairs()
    mizuno = mizuno_extraction()
    partners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pair in pairs:
        partners.setdefault(pair["mizuno"], []).append(pair["parity"])
    natural = [row.key for row in mizuno.table3 if row.a == 0]
    assert natural, "the profile carries no natural-composition key"
    for key in natural:
        assert len(partners.get(key, [])) == 1, (key, partners.get(key))
    # The primary's natural silicon and its enriched Si-28 both meet the one compiled-in silicon
    # row, which carries the isotope label; Si-29 and Si-30 meet nothing.
    assert partners[(14, 0)] == [(14, 28)]
    assert partners[(14, 28)] == [(14, 28)]
    assert partners.get((14, 29), []) == [] and partners.get((14, 30), []) == []
    assert {pair["mizuno"] for pair in pairs} | {(14, 29), (14, 30)} == set(mizuno.keys)

    disagreements = [pair for pair in pairs if not pair["agrees"]]
    unpartnered = sorted(set(mizuno.keys) - {p["mizuno"] for p in pairs})
    pins = document_pins()
    assert pins.crosscheck_unpartnered and pins.crosscheck_unpartnered[0][2] == unpartnered_keys_text(
        unpartnered
    )
    print(f"\ncross-check: {len(disagreements)} of {len(pairs)} pair(s) differ at the printed precision")
    print(f"  {mizuno2025.PROFILE} keys no parity record partners (not compared): {unpartnered}")
    for pair in pairs:
        parity_value, parity_unc = pair["parity_literal"]
        printed_value, printed_unc = pair["printed"]
        print(
            f"  {mizuno2025.PROFILE} {pair['mizuno']} vs parity {pair['parity']}: "
            f"parity {parity_value} +- {parity_unc}, printed {printed_value} +- {printed_unc}; "
            f"value {'equal' if pair['value_agrees'] else 'differs'}, "
            f"unc {'equal' if pair['unc_agrees'] else 'differs'}"
        )
    assert len(pins.crosscheck_rows) == len(disagreements)


def test_t91_drill_agreement_is_decided_at_the_printed_digits():
    """A double that rounds to the printed decimal agrees; one off in the last printed digit, or
    printed to fewer digits than it differs at, does not -- for values and uncertainties alike."""
    assert agrees_at_printed_precision(0.4823, "0.4823")
    assert agrees_at_printed_precision(3.8999999999999999, "3.90")
    assert agrees_at_printed_precision(0.48234, "0.4823")
    assert not agrees_at_printed_precision(0.48236, "0.4823")
    assert not agrees_at_printed_precision(0.4823, "0.4856")
    assert not agrees_at_printed_precision(0.03, "0.08")
    assert not agrees_at_printed_precision(0.0015, "0.009")


# --------------------------------------------------------------------------------------------
# T-93 -- the effective-charge audit: the shipped file loads, and each loader rule refuses its fixture
# --------------------------------------------------------------------------------------------

ZEFF_AUDIT = REPO / d1.ZEFF_AUDIT_RELPATH


def zeff_audit_rows() -> dict[int, d1.ZeffAuditRow]:
    return d1.load_zeff_audit(ZEFF_AUDIT)


def _line(text: str, index: int, mutate) -> str:
    """The text with line `index` (0 = the header, 1 = the first data line) replaced."""
    lines = text.split("\n")
    lines[index] = mutate(lines[index])
    return "\n".join(lines)


def _cell(index: int, value: str):
    """A line mutation that sets column `index` to `value`."""
    def mutate(line: str) -> str:
        cells = line.split(",")
        cells[index] = value
        return ",".join(cells)
    return mutate


def _swap_first_two_data_lines(text: str) -> str:
    lines = text.split("\n")
    lines[1], lines[2] = lines[2], lines[1]
    return "\n".join(lines)


#: (label, mutation of the file's text, message the loader must give). Each row is one rule of
#: `load_zeff_audit`; fixtures are cut from the shipped file's own lines.
ZEFF_AUDIT_DRILLS = [
    ("a CR byte", lambda t: t.replace("\n", "\r\n", 1), "contains CR"),
    ("a non-ASCII byte", lambda t: _line(t, 1, lambda l: l.replace("preprint", "préprint")),
     "is not ASCII"),
    ("the header renamed", lambda t: _line(t, 0, lambda l: l.replace("printed_z,", "printed,", 1)),
     "header is"),
    ("a non-integer Z", lambda t: _line(t, 1, _cell(0, "H")), "must be integers"),
    ("a non-integer printed_z", lambda t: _line(t, 1, _cell(1, "H")), "must be integers"),
    ("a printed_zeff without a point", lambda t: _line(t, 1, _cell(2, "1")),
     "digits, a point and digits"),
    ("an underlined outside true/false", lambda t: _line(t, 1, _cell(3, "yes")),
     "must be 'true' or 'false'"),
    ("an empty locator", lambda t: _line(t, 1, _cell(4, "")), "locator and a copy_read"),
    ("an empty copy_read", lambda t: _line(t, 1, _cell(5, "")), "locator and a copy_read"),
    ("a duplicate Z", lambda t: _line(t, 2, _cell(0, t.split("\n")[1].split(",")[0])), "duplicate Z"),
    ("rows not ascending in Z", _swap_first_two_data_lines, "strictly ascending in Z"),
    ("no rows", lambda t: t.split("\n")[0] + "\n", "carries no rows"),
]


def test_t93_the_shipped_effective_charge_audit_loads_and_every_row_names_a_table_page_and_copy():
    """The committed file passes every rule; each row is keyed by its own Z, names one of the
    primary's two tables and a page, and names a copy this project distinguishes."""
    audit = zeff_audit_rows()
    assert audit, "an empty audit would make every check below vacuous"
    assert list(audit) == sorted(audit)
    for z, row in audit.items():
        assert row.z == z
        assert re.search(r"\bTable (III|IV)\b", row.locator), (z, row.locator)
        assert re.search(r"\bp\.\d+", row.locator), (z, row.locator)
        assert row.copy_read in KNOWN_COPIES, (z, row.copy_read)


@pytest.mark.parametrize("label, mutate, message", ZEFF_AUDIT_DRILLS, ids=[d[0] for d in ZEFF_AUDIT_DRILLS])
def test_t93_drill_each_loader_rule_refuses_its_fixture(tmp_path, label, mutate, message):
    """Corrupt the audit in one way, in a temporary copy; the loader must refuse it with the rule's
    own message. The unmutated copy loads, so the message is the rule's."""
    text = ZEFF_AUDIT.read_bytes().decode("ascii")
    path = tmp_path / ZEFF_AUDIT.name
    path.write_bytes(text.encode("ascii"))
    d1.load_zeff_audit(path)  # the copy loads before the mutation
    mutated = mutate(text)
    assert mutated != text, label
    path.write_bytes(mutated.encode("utf-8"))
    with pytest.raises(d1.ZeffAuditError, match=re.escape(message)):
        d1.load_zeff_audit(path)


# --------------------------------------------------------------------------------------------
# T-94 -- every printed cell equals the shipped value, a misprinted Z is localized by its duplicate,
#         and the audit's coverage is the document's coverage, derived twice
# --------------------------------------------------------------------------------------------


def test_t94_every_printed_effective_charge_cell_equals_the_shipped_value_and_covers_the_documented_set():
    """(a) The printed cell is the shipped double at the primary's own precision -- exact decimal
    equality, since the compiled-in table was transcribed from these cells. (b) The rows whose printed
    Z is not their element's Z are exactly the misprint the document names: its Z, printed Z and
    printed value are read from `DATASET_D1.md`'s sentence, never typed here. (c) The set of Z the
    audit covers is computed a second way -- by `zeff_covered_split`, the helper `document_pins`
    calls, from the capture table's Z set -- and the two derivations are compared, never
    restated; the per-table split is compared the same way. (d) The counts are printed, not asserted."""
    audit = zeff_audit_rows()
    found = extraction()
    zeff_table, _ = committed(ZEFF_LAYER1, ZEFF_LAYER2)
    shipped = {int(z): value for z, value in zeff_table.records}

    for z, row in audit.items():
        assert z in shipped, z
        assert decimal.Decimal(row.printed_zeff) == decimal.Decimal(repr(shipped[z])), (
            z, row.printed_zeff, shipped[z]
        )

    misprinted = {z: row.printed_z for z, row in audit.items() if row.printed_z != z}
    document = (REPO / "DATASET_D1.md").read_text("utf-8")
    ((printed_z, printed_zeff),) = re.findall(
        r'prints the barium row as \*\*"(\d+)\(([\d.]+)\)"\*\*', document
    )
    (barium,) = re.findall(r"barium is Z = (\d+)", document)
    assert misprinted == {int(barium): int(printed_z)}, (misprinted, barium, printed_z)
    assert audit[int(barium)].printed_zeff == printed_zeff, (barium, printed_zeff)

    zs = sorted({z for z, _ in {(z, a) for z, a, _, _ in found.capture_records}})
    zeff_covered, zeff_covered_iii, zeff_covered_iv = zeff_covered_split(zs, zeff_table)
    assert set(audit) == zeff_covered
    in_iii = {z for z, row in audit.items() if "Table III" in row.locator}
    in_iv = {z for z, row in audit.items() if "Table IV" in row.locator}
    assert in_iii == zeff_covered_iii
    assert in_iv == zeff_covered_iv
    assert in_iii.isdisjoint(in_iv)
    print(
        f"\nzeff audit: covered {len(audit)} split {len(in_iii)} (Table III) / {len(in_iv)} "
        f"(Table IV); misprinted Z {sorted(misprinted.items())}"
    )


# --------------------------------------------------------------------------------------------
# T-95 -- the estimate marks in the shipped Layer 2 are the underlined cells of the audit, counted
#         on both sides; a copy with one mark flipped is caught
# --------------------------------------------------------------------------------------------


def estimate_rows(layer2_path: pathlib.Path) -> set[str]:
    """The keys of the rows a Layer-2 file marks `unc_type: estimate`, read from its bytes."""
    document = provenance.from_json_obj(json.loads(layer2_path.read_bytes().decode("ascii")))
    return {key for key, row in document.rows.items() if row.unc_type == "estimate"}


def test_t95_the_estimate_marks_are_the_underlined_cells_counted_on_both_sides():
    audit = zeff_audit_rows()
    underlined = {str(z) for z, row in audit.items() if row.underlined}
    estimates = estimate_rows(ZEFF_LAYER2)
    print(f"\nestimate {len(estimates)} underlined {len(underlined)}")
    assert len(estimates) == len(underlined)
    assert estimates == underlined


def test_t95_drill_a_copy_with_one_estimate_flipped_is_caught(tmp_path):
    """Flip exactly one `estimate` to `table` in a copy of the shipped file: the count on the
    Layer-2 side drops by one and the identity above fails on it."""
    audit = zeff_audit_rows()
    underlined = {str(z) for z, row in audit.items() if row.underlined}
    text = ZEFF_LAYER2.read_bytes().decode("ascii")
    marker = '"unc_type": "estimate"'
    assert text.count(marker) == len(underlined)
    flipped = text.replace(marker, '"unc_type": "table"', 1)
    assert flipped != text
    copy = tmp_path / ZEFF_LAYER2.name
    copy.write_bytes(flipped.encode("ascii"))
    estimates = estimate_rows(copy)
    assert len(estimates) == len(underlined) - 1
    assert estimates != underlined


# --------------------------------------------------------------------------------------------
# T-96 -- the printed capture cells: the shipped file loads, and each loader rule refuses its fixture
# --------------------------------------------------------------------------------------------

CAPTURE_CELLS = REPO / d1.CAPTURE_CELLS_RELPATH


def capture_cells() -> tuple[d1.CaptureCellRow, ...]:
    return d1.load_capture_cells(CAPTURE_CELLS)


def _first_data_cell(text: str, index: int) -> str:
    return text.split("\n")[1].split(",")[index]


#: (label, mutation of the file's text, message the loader must give). Each row is one rule of
#: `load_capture_cells`; fixtures are cut from the shipped file's own lines.
CAPTURE_CELLS_DRILLS = [
    ("a CR byte", lambda t: t.replace("\n", "\r\n", 1), "contains CR"),
    ("a non-ASCII byte", lambda t: _line(t, 1, lambda l: l.replace("preprint", "préprint")),
     "is not ASCII"),
    ("a quote byte", lambda t: _line(t, 1, lambda l: l.replace("preprint", '"preprint"')),
     "contains a quote byte"),
    ("the header renamed", lambda t: _line(t, 0, lambda l: l.replace("rate_unc,", "unc,", 1)),
     "header is"),
    ("a line with a tenth cell", lambda t: _line(t, 1, lambda l: l + ","), "exactly 9 cells"),
    ("a line with eight cells", lambda t: _line(t, 1, lambda l: l.rsplit(",", 1)[0]),
     "exactly 9 cells"),
    ("a non-integer Z", lambda t: _line(t, 1, _cell(0, _first_data_cell(t, 1))),
     "must be an integer"),
    ("a label outside its grammar", lambda t: _line(t, 1, _cell(1, _first_data_cell(t, 7))),
     "label must be"),
    ("a rate without a point", lambda t: _line(t, 1, _cell(3, _first_data_cell(t, 0))),
     "digits, a point and digits"),
    ("a rate_unc without a point", lambda t: _line(t, 1, _cell(4, _first_data_cell(t, 0))),
     "digits, a point and digits"),
    ("a refs outside its grammar", lambda t: _line(t, 1, _cell(6, _first_data_cell(t, 1))),
     "refs must be"),
    ("a dagger outside true/false", lambda t: _line(t, 1, _cell(2, _first_data_cell(t, 1))),
     "dagger must be 'true' or 'false'"),
    ("a bracketed outside true/false", lambda t: _line(t, 1, _cell(5, _first_data_cell(t, 1))),
     "bracketed must be 'true' or 'false'"),
    ("an empty locator", lambda t: _line(t, 1, _cell(7, "")), "locator and a copy_read"),
    ("an empty copy_read", lambda t: _line(t, 1, _cell(8, "")), "locator and a copy_read"),
    ("a block split in two", lambda t: _line(t, 2, _cell(0, t.split("\n")[-2].split(",")[0])),
     "not contiguous"),
    ("blocks not ascending in Z",
     lambda t: "\n".join([t.split("\n")[0], t.split("\n")[-2]] + t.split("\n")[1:-2] + [""]),
     "strictly ascending in Z"),
    ("a duplicate cell", lambda t: _line(t, 2, lambda l: t.split("\n")[1]), "duplicate cell"),
    ("no rows", lambda t: t.split("\n")[0] + "\n", "carries no rows"),
]


def test_t96_the_shipped_capture_cells_load_and_every_row_names_its_element_table_page_and_copy():
    """The committed file passes every rule; each row's Z is the atomic number of the element its
    label names, its locator names the primary's Table IV and a page, and it names a copy this
    project distinguishes."""
    cells = capture_cells()
    assert cells, "an empty cells file would make every check below vacuous"
    for cell in cells:
        assert cell.z == SYMBOL_Z[cell.label.split("-")[0]], (cell.z, cell.label)
        assert re.search(r"\bTable IV\b", cell.locator), (cell.z, cell.locator)
        assert re.search(r"\bp\.\d+", cell.locator), (cell.z, cell.locator)
        assert cell.copy_read in KNOWN_COPIES, (cell.z, cell.copy_read)


@pytest.mark.parametrize(
    "label, mutate, message", CAPTURE_CELLS_DRILLS, ids=[d[0] for d in CAPTURE_CELLS_DRILLS]
)
def test_t96_drill_each_loader_rule_refuses_its_fixture(tmp_path, label, mutate, message):
    """Corrupt the cells file in one way, in a temporary copy; the loader must refuse it with the
    rule's own message. The unmutated copy loads, so the message is the rule's."""
    text = CAPTURE_CELLS.read_bytes().decode("ascii")
    path = tmp_path / CAPTURE_CELLS.name
    path.write_bytes(text.encode("ascii"))
    d1.load_capture_cells(path)  # the copy loads before the mutation
    mutated = mutate(text)
    assert mutated != text, label
    path.write_bytes(mutated.encode("utf-8"))
    with pytest.raises(d1.CaptureCellsError, match=re.escape(message)):
        d1.load_capture_cells(path)


# --------------------------------------------------------------------------------------------
# T-97 -- the capture rows the audit had left open, decided by comparison with the printed cells
# --------------------------------------------------------------------------------------------


def check_open_row_verdicts(
    audit: dict[tuple[int, int], d1.IsotopeAuditRow],
    cells: tuple[d1.CaptureCellRow, ...],
    document: provenance.ProvDocument,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The comparison rule as a check: every capture row whose flag the listing alone cannot settle
    is compared, value and uncertainty, with every cell the primary prints at its Z at the primary's
    printed precision; one equal cell settles it to that entry, any other count leaves it open, and
    the audit, the shipped Layer 2 and the block of cells must all say so. Returns
    `(by_value, unsettled)`; raises `AssertionError` where they disagree."""
    found = extraction()
    blocks = d1.cells_by_z(cells)
    values = {(z, a): (value, unc) for z, a, value, unc in found.capture_records}
    literals = {
        (z, a): literal
        for (z, a, _, _), literal in zip(found.capture_records, found.capture_literals, strict=True)
    }
    by_value = sorted(
        key for key, finding in audit.items()
        if key[0] in blocks and d1.decided_by_value(key, finding.evidence, blocks[key[0]])
    )
    unsettled = sorted(key for key, finding in audit.items() if not finding.settled)
    # (a) every unsettled row is decided here, and every block of cells decides at least one row.
    assert set(unsettled) <= set(by_value), sorted(set(unsettled) - set(by_value))
    for z in blocks:
        assert any(key[0] == z for key in by_value), f"the cells at Z={z} decide no row"
    print(f"\ndecided by value: {by_value}")
    open_by_value = set()
    for key in by_value:
        z, a = key
        finding = audit[key]
        block = blocks[z]
        matches = d1.printed_matches(*values[key], block)
        literal_value, literal_unc = literals[key]
        # (b) one equal cell is a settled row, to exactly that cell's entry; otherwise open.
        assert (len(matches) == 1) is finding.settled, (key, d1.render_outcome(matches))
        if finding.settled:
            (match,) = matches
            assert finding.locator == match.locator, key
            assert finding.copy_read == match.copy_read, key
            assert finding.isotope_resolved is d1.is_separated_label(match.label), key
            if d1.is_separated_label(match.label):
                assert finding.evidence.startswith(
                    "the primary lists the separated isotope " + match.label
                ), key
        row = document.rows[f"{z}-{a}"]
        assert row.needs_verification is not finding.settled, key
        assert row.evaluation_method.endswith(
            d1.render_comparison(z, literal_value, literal_unc, block, matches)
        ), key
        # (c) a record whose A no printed label carries can equal a cell of another nuclide; that
        # is a finding to register under its own name, never a row to settle here.
        printed_as = {
            int(cell.label.split("-")[1]) for cell in block if d1.is_separated_label(cell.label)
        }
        if a not in printed_as:
            assert len(matches) != 1, (
                f"{key} carries no printed label of its A yet equals exactly one cell, "
                f"{d1.render_cell(matches[0]) if matches else ''}: a registration is needed"
            )
        if len(matches) != 1:
            open_by_value.add(key)
        print(
            f"  {key}: parity {literal_value} +- {literal_unc}, matches "
            f"{[d1.render_cell(match) for match in matches]}, verdict {d1.render_outcome(matches)}"
        )
    # (d) the open rows are exactly the decided-by-value rows no single cell equals.
    assert set(unsettled) == open_by_value, (sorted(unsettled), sorted(open_by_value))
    return by_value, unsettled


def test_t97_every_open_capture_row_is_decided_by_comparison_with_the_printed_cells():
    """The rule the chapter states as a command over the shipped audit, cells and Layer 2; and the
    document tabulates exactly the rows that stay open."""
    _, document = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)
    by_value, unsettled = check_open_row_verdicts(audit_rows(), capture_cells(), document)
    assert by_value, "no row is decided by value; the check above was vacuous"
    # (e) the rows section 6's table carries, read from the document itself -- the only table whose
    # rows open with a parenthesised key -- are exactly the rows the audit leaves open.
    tabulated = sorted(
        (int(z), int(a))
        for z, a in re.findall(r"\| \((\d+), (\d+)\) \| \S+ \+- \S+ \| ", document_pins().doc)
    )
    assert tabulated == unsettled, (tabulated, unsettled)


def test_t97_drill_a_settled_row_whose_cell_moves_in_its_last_digit_is_caught(tmp_path):
    """Take the one cell a settled decided-by-value row equals -- found by label from the shipped
    files, not typed -- and alter its last printed digit in a temporary copy of the cells file: the
    row then equals no cell while the audit still says settled, and the check fails on it."""
    audit = audit_rows()
    cells = capture_cells()
    _, document = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)
    by_value, _ = check_open_row_verdicts(audit, cells, document)
    settled = [key for key in by_value if audit[key].settled]
    assert settled, "no decided-by-value row is settled; the drill has nothing to alter"
    z, a = settled[0]
    found = extraction()
    values = {(z_, a_): (value, unc) for z_, a_, value, unc in found.capture_records}
    (match,) = d1.printed_matches(*values[(z, a)], d1.cells_by_z(cells)[z])
    text = CAPTURE_CELLS.read_bytes().decode("ascii")
    lines = text.split("\n")
    hits = [
        index for index, line in enumerate(lines[1:], start=1)
        if line.split(",")[:2] == [str(match.z), match.label]
        and line.split(",")[3:5] == [match.rate, match.rate_unc]
    ]
    assert len(hits) == 1, hits
    cells_ = lines[hits[0]].split(",")
    cells_[3] = match.rate[:-1] + str((int(match.rate[-1]) + 1) % 10)
    lines[hits[0]] = ",".join(cells_)
    mutated = "\n".join(lines)
    assert mutated != text
    path = tmp_path / CAPTURE_CELLS.name
    path.write_bytes(mutated.encode("ascii"))
    with pytest.raises(AssertionError):
        check_open_row_verdicts(audit, d1.load_capture_cells(path), document)


# --------------------------------------------------------------------------------------------
# T-101 -- each conjunct of the comparison rule decides a block derived from the shipped files
# --------------------------------------------------------------------------------------------


def test_t101_each_conjunct_of_the_comparison_rule_decides_a_derived_block():
    """Every input is the shipped audit, cells and extraction; every block below is derived from
    them by one stated change. (a) the uncertainty conjunct of `printed_matches`: the one cell the
    settled decided-by-value row equals, its uncertainty moved by one unit in its last printed digit,
    is no longer a match. (b) the collision shape of `decided_by_value`: an open key whose block
    carries a natural label and a separated label of its A, and whose evidence states `round(Ar)`
    equal to A, is decided; with the stated `round(Ar)` moved off A, or with every natural label
    removed from the block, it is not. (c) the absent shape: an open key whose A no separated label
    carries is decided, with or without a `round(Ar)` clause in its evidence."""
    audit = audit_rows()
    blocks = d1.cells_by_z(capture_cells())
    found = extraction()
    values = {(z, a): (value, unc) for z, a, value, unc in found.capture_records}
    by_value = sorted(
        key for key, finding in audit.items()
        if key[0] in blocks and d1.decided_by_value(key, finding.evidence, blocks[key[0]])
    )

    def separated_labels(block):
        return {int(cell.label.split("-")[1]) for cell in block if d1.is_separated_label(cell.label)}

    # (a)
    settled = [key for key in by_value if audit[key].settled]
    assert len(settled) == 1, settled
    (key,) = settled
    z, a = key
    (match,) = d1.printed_matches(*values[key], blocks[z])
    printed_unc = decimal.Decimal(match.rate_unc)
    one_unit = decimal.Decimal(1).scaleb(printed_unc.as_tuple().exponent)
    bumped = dataclasses.replace(match, rate_unc=str(printed_unc + one_unit))
    assert bumped.rate == match.rate and bumped.rate_unc != match.rate_unc
    block = tuple(bumped if cell is match else cell for cell in blocks[z])
    assert d1.printed_matches(*values[key], block) == ()

    # (b)
    collisions = [
        key for key, finding in audit.items()
        if not finding.settled and key[0] in blocks
        and any(not d1.is_separated_label(cell.label) for cell in blocks[key[0]])
        and key[1] in separated_labels(blocks[key[0]])
        and f"round(Ar)={key[1]}" in finding.evidence
    ]
    assert collisions, "no open key carries a natural label, a separated label of its A and round(Ar)=A"
    for key in collisions:
        z, a = key
        evidence = audit[key].evidence
        assert d1.decided_by_value(key, evidence, blocks[z]), key
        moved = evidence.replace(f"round(Ar)={a}", f"round(Ar)={a + 1}")
        assert moved != evidence
        assert not d1.decided_by_value(key, moved, blocks[z]), key
        without_natural = tuple(cell for cell in blocks[z] if d1.is_separated_label(cell.label))
        assert without_natural != blocks[z]
        assert not d1.decided_by_value(key, evidence, without_natural), key

    # (c)
    absent = [
        key for key, finding in audit.items()
        if not finding.settled and key[0] in blocks and key[1] not in separated_labels(blocks[key[0]])
    ]
    assert absent, "no open key has an A no separated label carries"
    for key in absent:
        evidence = audit[key].evidence
        assert d1.decided_by_value(key, evidence, blocks[key[0]]), key
        stripped = re.sub(r"round\(Ar\)=\d+", "", evidence)
        assert d1.decided_by_value(key, stripped, blocks[key[0]]), key


# --------------------------------------------------------------------------------------------
# T-102 -- the unpartnered keys render from one home
# --------------------------------------------------------------------------------------------


def test_t102_the_unpartnered_keys_render_from_one_home():
    """`unpartnered_keys_text` is the one rendering both `document_pins` and T-91 compare the document
    against; drill keys, at one, two and three keys."""
    assert unpartnered_keys_text([(1, 2)]) == "`1-2`"
    assert unpartnered_keys_text([(1, 2), (3, 4)]) == "`1-2` and `3-4`"
    assert unpartnered_keys_text([(1, 2), (3, 4), (5, 6)]) == "`1-2`, `3-4` and `5-6`"
    assert unpartnered_keys_text([]) == ""


# --------------------------------------------------------------------------------------------
# T-99 -- the profile sweep predicted: which keys a second profile moves, and to what
# --------------------------------------------------------------------------------------------


def profile_sweep_diff(
    profile_values: dict[tuple[int, int], float],
    parity_records: Sequence[tuple[int, int, float, float]],
    model: d1.GoulardPrimakoff,
) -> list[tuple[int, int, float, float]]:
    """The rows of the sweep box where a patched build reading through a profile returns a
    different double from the `parity` sweep: `(Z, A, parity value, profile value)`.

    The profile resolves a key the way the reader's `LookupNatural` does -- the exact `(Z, A)`
    record, else the `(Z, 0)` record, else nothing -- and a resolved value goes through the same
    `value / microsecond` expression the seam runs, here as the reference model's table branch
    fed one record. An unresolved key falls through to the compiled-in code, which is the parity
    value, so it never differs.
    """
    rows: list[tuple[int, int, float, float]] = []
    for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1):
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1):
            parity_value = d1.capture_rate(z, a, parity_records, model)
            if (z, a) in profile_values:
                value = profile_values[(z, a)]
            elif (z, 0) in profile_values:
                value = profile_values[(z, 0)]
            else:
                continue
            profile_value = d1.capture_rate(z, a, [(z, a, value, 0.0)], model)
            if struct.pack(">d", parity_value) != struct.pack(">d", profile_value):
                rows.append((z, a, parity_value, profile_value))
    return rows


def profile_capture_values(layer1_path: pathlib.Path) -> dict[tuple[int, int], float]:
    """`{(Z, A): value}` of one shipped capture file, parsed by the format module."""
    table = spec.parse(layer1_path.read_bytes().decode("ascii"))
    columns = table.directives["COLUMNS"].split()
    z_index, a_index, value_index = columns.index("Z"), columns.index("A"), columns.index("value")
    return {(int(r[z_index]), int(r[a_index])): float(r[value_index]) for r in table.records}


def predicted_profile_sweep(profile_layer1: pathlib.Path) -> list[tuple[int, int, float, float]]:
    """The predicted rows for one profile file against the shipped `parity` side."""
    found = extraction()
    return profile_sweep_diff(
        profile_capture_values(profile_layer1), found.capture_records, reference_model(found)
    )


def test_t99_the_second_profile_moves_exactly_the_keys_it_resolves():
    """Over the shipped `mizuno2025` file: every differing key is one the profile resolves; every
    T-91 pair whose value differs at the printed precision has its parity key among them; every
    swept A of a Z with a `(Z, 0)` row is resolved; and the directory holds no `muon_zeff` under
    that profile, so its effective charges fall through unchanged."""
    profile_values = profile_capture_values(MIZUNO_LAYER1)
    rows = predicted_profile_sweep(MIZUNO_LAYER1)
    assert rows, "the second profile moves no key"
    differing = {(z, a) for z, a, _, _ in rows}
    resolved = {
        (z, a)
        for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1)
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1)
        if (z, a) in profile_values or (z, 0) in profile_values
    }
    assert differing <= resolved, sorted(differing - resolved)
    for pair in mizuno_parity_pairs():
        if not pair["value_agrees"]:
            assert pair["parity"] in differing, pair
    natural = sorted(z for z, a in profile_values if a == 0)
    assert natural, "the profile carries no natural-composition row"
    for z in natural:
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1):
            assert (z, a) in resolved, (z, a)
    tables = spec.load_directory(D1DIR)
    assert (mizuno2025.PROFILE, "muon_zeff") not in tables, sorted(tables)
    assert (spec.PARITY_PROFILE, "muon_zeff") in tables, sorted(tables)
    print(f"\n{mizuno2025.PROFILE}: {len(rows)} differing key(s); no muon_zeff table under that profile")
    for z, a, parity_value, profile_value in rows[:3] + rows[-3:]:
        print(f"  {z} {a} {parity_value.hex()} {profile_value.hex()}")


def test_t99_drill_a_dropped_natural_row_loses_exactly_that_elements_rows():
    """The profile map without its `(12, 0)` entry -- found from the shipped file, not typed --
    loses exactly the rows at Z = 12, which are non-empty, and nothing else moves."""
    profile_values = profile_capture_values(MIZUNO_LAYER1)
    natural_z = 12
    assert (natural_z, 0) in profile_values, sorted(profile_values)
    assert [a for z, a in profile_values if z == natural_z] == [0]
    found = extraction()
    model = reference_model(found)
    full = profile_sweep_diff(profile_values, found.capture_records, model)
    without = profile_sweep_diff(
        {key: value for key, value in profile_values.items() if key != (natural_z, 0)},
        found.capture_records,
        model,
    )
    lost = [row for row in full if row[0] == natural_z]
    assert lost, f"no row at Z = {natural_z} to lose"
    assert without == [row for row in full if row[0] != natural_z]


# --------------------------------------------------------------------------------------------
# T-100 -- the revisions the dataset's headline names are builds its section 4 names
# --------------------------------------------------------------------------------------------

DATASET_DOCUMENT = REPO / "DATASET_D1.md"
_REVISION = re.compile(r"\bv?(11\.\d+\.\d+(?:\.beta)?)\b")


def revision_tokens(text: str) -> set[str]:
    """Every Geant4 revision token in `text`, without its `v`: `11.4.2`, `11.5.0.beta`."""
    return {m.group(1) for m in _REVISION.finditer(text)}


def headline_and_section4(text: str) -> tuple[str, str]:
    """The document's headline (everything before its first section) and its section 4."""
    headline = text.split("\n## ", 1)[0]
    section4 = text.split("\n## 4.", 1)[1].split("\n## 5.", 1)[0]
    return headline, section4


def check_headline_revisions(text: str) -> None:
    """The headline names exactly the two vendored revisions, and section 4 names every one of
    them -- a parity claim is a claim about a named build, so a revision the headline claims
    parity with must be a build section 4 describes."""
    headline, section4 = headline_and_section4(text)
    assert revision_tokens(headline) == {d1.UPSTREAM_TAG[1:], BETA_TAG[1:]}, revision_tokens(headline)
    assert revision_tokens(headline) <= revision_tokens(section4), (
        revision_tokens(headline) - revision_tokens(section4)
    )


def test_t100_the_revisions_the_headline_names_are_builds_section_4_names():
    check_headline_revisions(DATASET_DOCUMENT.read_bytes().replace(b"\r\n", b"\n").decode("utf-8"))


def test_t100_drill_a_section_4_that_names_no_beta_build_is_caught():
    """Section 4 with every line naming the beta removed: the headline still claims parity with
    it, and the check fails on exactly that revision."""
    text = DATASET_DOCUMENT.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    check_headline_revisions(text)
    head, rest = text.split("\n## 4.", 1)
    section4, tail = rest.split("\n## 5.", 1)
    kept = [line for line in section4.split("\n") if BETA_TAG[1:] not in line]
    assert len(kept) < len(section4.split("\n")), "section 4 names the beta on no line"
    mutated = head + "\n## 4." + "\n".join(kept) + "\n## 5." + tail
    with pytest.raises(AssertionError, match=re.escape(BETA_TAG[1:])):
        check_headline_revisions(mutated)
