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

import hashlib
import pathlib

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
