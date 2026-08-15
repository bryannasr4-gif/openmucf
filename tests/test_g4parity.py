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
import dataclasses
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import struct
import subprocess
import sys

import pytest

import openmucf
from openmucf import rates
from openmucf.g4 import provenance, sources, spec
from openmucf.g4.sources import d1_nuclear_capture as d1

REPO = pathlib.Path(__file__).resolve().parents[1]
VENDORED = REPO / "third_party" / "geant4" / "v11.4.2" / "G4MuonMinusBoundDecay.cc"
D1DIR = REPO / "data" / "g4" / "d1"
ORACLE = D1DIR / "d1_gp_sweep.oracle"

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
    byte_swapped = reference_model(found)
    little = hashlib.sha256()
    for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1):
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1):
            little.update(
                struct.pack("<d", d1.capture_rate(z, a, found.capture_records, byte_swapped))
            )
    assert "big-endian" in oracle["header"]["digest_rule"]
    assert computed != little.hexdigest(), "the sweep is byte-order blind; the digest rule proves nothing"
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
    assert rendered == ORACLE.read_text("ascii"), (
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
# T-43..T-47, T-53..T-55, T-57, T-58 -- the shipped dataset against the source it claims
# --------------------------------------------------------------------------------------------

CAPTURE_LAYER1 = D1DIR / "d1_capture.g4dat"
CAPTURE_LAYER2 = D1DIR / "d1_capture.prov.json"
ZEFF_LAYER1 = D1DIR / "d1_zeff.g4dat"
ZEFF_LAYER2 = D1DIR / "d1_zeff.prov.json"
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
            # A parity profile reproduces; it does not recommend, and nothing is verified yet.
            assert row.recommendation == "", key
            assert row.needs_verification is True, key
            # Upstream says "weighted average of the two most precise measurements".
            assert row.single_source is False, key
            assert row.unc_type == "table", key
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


def test_t54_isotope_resolved_obeys_the_stated_derivation():
    """The one non-obvious boolean, checked against the rule the file itself states.

    This release's rule is mechanical and sound in exactly one direction: two rates at two A for
    one Z prove the underlying data distinguishes isotopes; one row proves nothing either way. So
    `true` must mean "this Z carries more than one row" and nothing else -- and `needs_verification`
    stays true everywhere, because "not established" is not the same claim as "false".
    """
    found = extraction()
    per_z = found.capture_rows_per_z()
    _, capture_document = committed(CAPTURE_LAYER1, CAPTURE_LAYER2)

    resolved = set()
    for key, row in capture_document.rows.items():
        z = int(key.split("-")[0])
        assert row.isotope_resolved is (per_z[z] > 1), key
        if row.isotope_resolved:
            resolved.add(z)
        # The derivation is stated in the row, so a reader sees how the flag was obtained.
        assert "isotope_resolved is derived mechanically" in row.evaluation_method, key
    assert resolved == {z for z, count in per_z.items() if count > 1}
    assert resolved, "no multi-isotope Z at all would make this rule vacuous"

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
    artifacts = sorted(D1DIR.glob("d1_*.g4dat")) + sorted(D1DIR.glob("*.prov.json")) + [
        D1DIR / "geant4_add_dataset.snippet"
    ]
    assert len(artifacts) == 5, [p.name for p in artifacts]

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
            assert result.returncode != 0, f"corrupting {path.name} did not fail the audit"
            assert path.name in result.stdout + result.stderr, (
                f"the audit failed but never named {path.name}: {result.stdout}{result.stderr}"
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
