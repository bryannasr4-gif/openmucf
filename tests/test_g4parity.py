"""D1 in parity mode: the vendored upstream, the extraction, and the bit-parity proof.

``tests/test_g4spec.py`` tests the *format*. This file tests the one **dataset** that claims to
reproduce something: `data/g4/d1/`, which asserts that every muon-capture record and every effective
charge it ships is bit-for-bit what Geant4 v11.4.2 compiles in, and that the Goulard-Primakoff
fallback it declares evaluates to the same doubles the compiled library returns.

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
import hashlib
import pathlib

import pytest

from openmucf.g4.sources import d1_nuclear_capture as d1

REPO = pathlib.Path(__file__).resolve().parents[1]
VENDORED = REPO / "third_party" / "geant4" / "v11.4.2" / "G4MuonMinusBoundDecay.cc"

#: Upstream's own object name for the vendored bytes, at tag v11.4.2
#: (commit 8cc04f65977807f1848da7b958c421cd5e162f26). This is a *pin*, not a measurement: it is the
#: pre-registered identity of the file the whole parity chain is derived from, and it is verifiable
#: against github.com/Geant4/geant4 by anyone, with no Geant4 checkout and no `git` binary.
UPSTREAM_BLOB_ID = "29bd73719cd619de34ef83ca5ca076ceadf1cc5a"
UPSTREAM_SHA256 = "860dcdb53167c6437484b12c05ac1ab2eae4a6a52886af83fcf4394611882813"


def git_blob_id(data: bytes) -> str:
    """Git's object name for ``data`` as a blob: ``sha1("blob <len>\\0" + data)``.

    Three lines of `hashlib` rather than a `git` call, deliberately: this must work in an unpacked
    sdist, in a container with no git, and for a reader who is checking our work against upstream
    without cloning Geant4.
    """
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


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
# T-42, T-50, T-51 -- the extraction: derived counts, verbatim coefficients, a live directive
# --------------------------------------------------------------------------------------------


def extraction() -> d1.D1Extraction:
    """The one extraction under test, always straight from the vendored source."""
    return d1.load(VENDORED)


def independently_counted_records(text: str) -> int:
    """Count `{...}` groups in the `capRates[]` body by brace depth alone -- no record pattern.

    Deliberately a *different* method from the extractor's: the extractor matches records with a
    regex and proves completeness by residue, this walks the body character by character and counts
    depth transitions. If a regex ever stopped early, these two would disagree, which is the whole
    point of not reusing the extractor's own machinery here.
    """
    start = text.index("static const capRate capRates")
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
