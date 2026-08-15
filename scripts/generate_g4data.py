"""Generate the ``G4MuonicData`` datasets and their registration snippets.

    python scripts/generate_g4data.py           # regenerate data/g4/ and print the archive MD5s
    python scripts/generate_g4data.py --audit   # rebuild into a temp dir and byte-compare

Two builds ship, and they are different KINDS of artifact.

**The example** (``data/g4/example.*``) carries no physics. Its rows are format examples with round,
obviously synthetic numbers, and every Layer-2 field says so in words. It exists because the
byte-diff audit needs a committed, regenerable artifact to guard, and because a format with no
example is a format nobody can check their reader against. Its Layer-2 file is hand-authored --
three rows of invented numbers have no other origin -- and everything Layer 1 declares about it is
read back out of that file.

**The D1 build** (``data/g4/d1/``) carries real data and claims something falsifiable: that every
muon-capture record and every effective charge in it is bit-for-bit what Geant4 v11.4.2 compiles in.
Nothing about it is hand-authored. Both layers are generated from the vendored upstream source, at
build time, every time -- so "bit-for-bit" is a property of this script rather than a claim about
what somebody typed once. Layer 2 is still the source of truth in the sense that matters: it is the
byte range ``#SOURCEDIGEST`` is taken over, and ``--audit`` verifies that on the COMMITTED pair.

Audit wiring: every generated artifact below joins ``make audit``'s ``git diff --exit-code`` list.
The ``.tar.gz`` archives are **not** committed -- they are build products whose determinism is
proven by test rather than by a stored copy -- but their MD5s are written into the snippets, because
a ``geant4_add_dataset`` block without a real ``MD5SUM`` is not a usable block.

Deliberately NOT regenerated here: ``data/g4/d1/d1_gp_sweep.oracle``. It was harvested from a
Geant4-linked binary and no code in this repository can produce it, which is exactly what makes it
evidence; it is guarded by re-derivation in ``tests/test_g4parity.py`` instead of by a byte-diff.
"""

from __future__ import annotations

import filecmp
import json
import sys
import tempfile
from pathlib import Path

import openmucf
from openmucf.g4 import emit, provenance, spec
from openmucf.g4.sources import d1_nuclear_capture as d1src

ROOT = Path(__file__).resolve().parents[1]
G4DIR = ROOT / "data" / "g4"
LAYER2_PATH = G4DIR / "example.prov.json"
LAYER1_PATH = G4DIR / "example.g4dat"
SNIPPET_PATH = G4DIR / "geant4_add_dataset.snippet"

#: The environment variable a "mode 2" user exports (``FORMAT_SPEC.md`` section 5). Provisional.
DATASET_ENVVAR = "G4MUONICDATA"
DATASET_NAME = "G4MuonicData"
TABLE_NAME = "format_example"
UNITS = "value=arbitrary unc=arbitrary"
COLUMNS = "Z A value unc"
VALIDITY = "Z:listed A:listed"

#: The example rows: round numbers, chosen so that no reader can mistake them for measurements.
#: Keys must match ``example.prov.json`` exactly -- see the module docstring.
EXAMPLE_RECORDS: tuple[tuple[int, int, float, float], ...] = (
    (1, 1, 1.0, 0.1),
    (6, 12, 2.0, 0.2),
    (29, 63, 3.0, 0.3),
)

# --------------------------------------------------------------------------------------------
# D1 -- the parity build
# --------------------------------------------------------------------------------------------

D1DIR = G4DIR / "d1"
D1_CAPTURE_LAYER1 = D1DIR / "d1_capture.g4dat"
D1_CAPTURE_LAYER2 = D1DIR / "d1_capture.prov.json"
D1_ZEFF_LAYER1 = D1DIR / "d1_zeff.g4dat"
D1_ZEFF_LAYER2 = D1DIR / "d1_zeff.prov.json"
D1_SNIPPET_PATH = D1DIR / "geant4_add_dataset.snippet"
VENDORED_PATH = ROOT / d1src.VENDORED_RELPATH

#: First build carrying content: plainly distinct from the example's `0.0.0-example`, and below
#: 1.0.0 because D1 alone is not the dataset.
D1_VERSION = "0.1.0"
D1_SEAM = "d1_nuclear_capture"
#: The release we actually read -- we vendored it. NOT the papers Geant4 cites: those are carried as
#: quoted upstream text in `conditions`, because citing a paper this project has not opened would be
#: exactly the ground-truth violation the two-layer design exists to prevent.
D1_BIBKEY = "geant4_v11_4_2"
D1_SOURCE_LIBRARY = "geant4-compiled-in"
D1_CAPTURE_TABLE = "nuclear_capture_rate"
D1_ZEFF_TABLE = "muon_zeff"

#: The parity profile's whole claim, in one sentence of two clauses, followed by the derivation
#: behind the row's one non-obvious boolean. The two live in one field deliberately: a reader who
#: sees the flag must see how it was obtained, and `conditions` is reserved for upstream's own words.
CAPTURE_METHOD = (
    "compiled-in constant table transcribed by Geant4; reproduced here bit-for-bit, not "
    "re-evaluated. isotope_resolved is derived mechanically: true if and only if this Z carries "
    "more than one capture row. That is sound in one direction and only that direction -- differing "
    "rates for two A at one Z establish that the underlying data distinguishes isotopes, while a "
    "single row establishes nothing either way, and needs_verification carries that state."
)
ZEFF_METHOD = (
    "compiled-in constant table transcribed by Geant4; reproduced here bit-for-bit, not "
    "re-evaluated. isotope_resolved is false on every row as a fact rather than a default: an "
    "effective charge is a per-Z quantity, so there is no isotope for it to be resolved to."
)


def _quote_upstream(lines: tuple[str, ...], *needles: str) -> str:
    """The upstream comment lines matching ``needles``, joined, in source order.

    The SELECTOR is written here; the TEXT is whatever the source says. That split is the point --
    a maintainer choosing which comment governs a row is a judgement, but the words that end up in
    the shipped file are copied out of the vendored bytes and never retyped.
    """
    picked = [line for line in lines if any(needle in line for needle in needles)]
    if not picked:
        raise SystemExit(
            f"the upstream comment block no longer contains any of {needles!r}; the attribution "
            "these rows quote cannot be located, and guessing at it is not an option"
        )
    return " ".join(picked)


def _locator(line: int) -> str:
    """A Layer-2 ``source_locator`` that resolves in THIS repository, not in someone's ~/geant4."""
    return f"{d1src.VENDORED_RELPATH}:{line} (upstream git blob {d1src.UPSTREAM_BLOB_ID})"


def _evaluation_id(table: str) -> str:
    return f"g4-{d1src.UPSTREAM_TAG.lstrip('v')}-boundDecay-{table}"


def build_capture_document(found: d1src.D1Extraction) -> provenance.ProvDocument:
    """Layer 2 for the capture table: one row per record, every field decided by rule."""
    per_z = found.capture_rows_per_z()
    general = _quote_upstream(
        found.capture_comment_lines, "capture data from", "Suzuki", "weighted average"
    )
    hydrogen = _quote_upstream(found.capture_comment_lines, "Hydrogen")
    helium = _quote_upstream(found.capture_comment_lines, "Helium")

    rows = {}
    for (z, a, _, _), line in zip(found.capture_records, found.capture_lines, strict=True):
        upstream = {1: hydrogen, 2: helium}.get(z, general)
        rows[f"{z}-{a}"] = provenance.ProvRow(
            source_bibkey=D1_BIBKEY,
            source_locator=_locator(line),
            # The enum has no "unstated", and cRErr is an uncertainty as TABULATED upstream --
            # upstream does not say what kind. `conditions` says so rather than letting the closest
            # available label imply a claim nobody made.
            unc_type="table",
            conditions=(
                f'quoted from the upstream source comment: "{upstream}". Upstream does not state '
                "what kind of uncertainty cRErr is, so unc_type is table; Geant4 itself never reads "
                "cRErr."
            ),
            validity_range=(
                f"Z={z} A={a}; outside the listed keys the {d1src.FALLBACK_MODEL} fallback applies"
            ),
            evaluation_method=CAPTURE_METHOD,
            # Upstream says "weighted average of the two most precise measurements"; asserting a
            # single source would be a claim this project cannot make.
            single_source=False,
            # Nothing here has been checked against a primary. That is a later stage's job, and
            # saying so is what keeps this profile honest about what it is.
            needs_verification=True,
            # A parity profile reproduces; it does not recommend.
            recommendation="",
            evaluation_id=_evaluation_id("capRates"),
            source_library=D1_SOURCE_LIBRARY,
            isotope_resolved=per_z[z] > 1,
        )
    return provenance.ProvDocument(
        dataset=DATASET_NAME,
        version=D1_VERSION,
        profile=spec.PARITY_PROFILE,
        seam=D1_SEAM,
        # A one-entry ordering, which is the honest ranking of a file carrying exactly one library.
        precedence=(D1_SOURCE_LIBRARY,),
        rows=rows,
    )


def build_zeff_document(found: d1src.D1Extraction) -> provenance.ProvDocument:
    """Layer 2 for the effective-charge table, keyed by Z alone."""
    upstream = _quote_upstream(
        found.zeff_comment_lines, "Effective charges", "Total Nuclear", "Suzuki", "Ford and Wills",
        "not present",
    )
    coefficients = found.coefficients
    zmin, zmax = int(coefficients["zmin"]), int(coefficients["zmax"])

    rows = {}
    for z, line in enumerate(found.zeff_lines):
        unreachable = z < zmin or z > zmax
        conditions = (
            f'quoted from the upstream source comment: "{upstream}". No uncertainty is published '
            "upstream and this table carries no unc column, so unc_type is table."
        )
        if unreachable:
            conditions += (
                f" This entry is UNREACHABLE through GetMuonZeff, which clamps Z into [{zmin}, "
                f"{zmax}] before indexing. It ships because the dataset reproduces the array as "
                "declared, and silently dropping an element it claims to reproduce would be a worse "
                "artifact than shipping one with a disclosure."
            )
        rows[str(z)] = provenance.ProvRow(
            source_bibkey=D1_BIBKEY,
            source_locator=_locator(line),
            unc_type="table",
            conditions=conditions,
            validity_range=(
                f"Z={z}; unreachable, the clamp maps it to Z={zmin}"
                if unreachable
                else f"Z={z}; GetMuonZeff clamps its argument into [{zmin}, {zmax}] before indexing"
            ),
            evaluation_method=ZEFF_METHOD,
            single_source=False,
            needs_verification=True,
            recommendation="",
            evaluation_id=_evaluation_id("zeff"),
            source_library=D1_SOURCE_LIBRARY,
            # A fact, not a default: an effective charge is per-Z and has no isotope.
            isotope_resolved=False,
        )
    return provenance.ProvDocument(
        dataset=DATASET_NAME,
        version=D1_VERSION,
        profile=spec.PARITY_PROFILE,
        seam=D1_SEAM,
        precedence=(D1_SOURCE_LIBRARY,),
        rows=rows,
    )


def build_capture_table(found: d1src.D1Extraction, digest: str) -> spec.G4DatTable:
    """Layer 1 for the capture table, records ascending by ``(Z, A)``."""
    z_values = found.distinct_capture_z
    directives = {
        "GRAMMAR": spec.GRAMMAR_VERSION,
        "DATASET": DATASET_NAME,
        "VERSION": D1_VERSION,
        "PROFILE": spec.PARITY_PROFILE,
        "SEAM": D1_SEAM,
        "TABLE": D1_CAPTURE_TABLE,
        "GENERATOR": f"openmucf-g4 {openmucf.__version__}",
        "SOURCEDIGEST": digest,
        "SOURCESHA": d1src.UPSTREAM_COMMIT,
        # `value`/`unc` are the COLUMN names. Naming the quantity instead would break section 2.2's
        # own rule that every `#UNITS` name is a `#COLUMNS` name.
        "UNITS": "value=1e6/s unc=1e6/s",
        "COLUMNS": "Z A value unc",
        # `A:listed`, not `A:natural_and_listed`: these are an enumerated set of specific isotopes,
        # not a natural-abundance rule, and `listed` is what is true.
        "VALIDITY": f"Z:{z_values[0]}-{z_values[-1]} A:listed",
        "FALLBACK": d1src.render_fallback_directive(d1src.FALLBACK_MODEL, found.coefficients),
    }
    records = tuple(sorted(found.capture_records, key=lambda record: (record[0], record[1])))
    return spec.G4DatTable(directives=directives, records=records)


def build_zeff_table(found: d1src.D1Extraction, digest: str) -> spec.G4DatTable:
    """Layer 1 for the effective-charge table. No ``#FALLBACK``: the clamp IS the model."""
    directives = {
        "GRAMMAR": spec.GRAMMAR_VERSION,
        "DATASET": DATASET_NAME,
        "VERSION": D1_VERSION,
        "PROFILE": spec.PARITY_PROFILE,
        "SEAM": D1_SEAM,
        "TABLE": D1_ZEFF_TABLE,
        "GENERATOR": f"openmucf-g4 {openmucf.__version__}",
        "SOURCEDIGEST": digest,
        "SOURCESHA": d1src.UPSTREAM_COMMIT,
        "UNITS": "value=dimensionless",
        "COLUMNS": "Z value",
        "VALIDITY": f"Z:0-{len(found.zeff) - 1}",
    }
    records = tuple((z, value) for z, value in enumerate(found.zeff))
    return spec.G4DatTable(directives=directives, records=records)


def build_d1_artifacts() -> tuple[dict[Path, bytes], bytes]:
    """The committed D1 artifacts keyed by path, plus the archive they describe (not committed)."""
    found = d1src.load(VENDORED_PATH)  # checks the upstream pins before anything is generated

    members: dict[str, bytes] = {}
    artifacts: dict[Path, bytes] = {}
    for layer1_path, layer2_path, document, build in (
        (D1_CAPTURE_LAYER1, D1_CAPTURE_LAYER2, build_capture_document(found), build_capture_table),
        (D1_ZEFF_LAYER1, D1_ZEFF_LAYER2, build_zeff_document(found), build_zeff_table),
    ):
        raw = provenance.document_bytes(document)
        table = build(found, provenance.source_digest(raw))
        spec.validate(table)
        provenance.check_against_table(table, document)
        provenance.check_source_digest(table, raw)
        layer1 = spec.render(table).encode("ascii")
        artifacts[layer1_path] = layer1
        artifacts[layer2_path] = raw
        members[layer1_path.name] = layer1
        members[layer2_path.name] = raw

    archive = emit.build_tarball(members)
    snippet = emit.add_dataset_snippet(
        name=DATASET_NAME,
        version=D1_VERSION,
        filename=DATASET_NAME,
        envvar=DATASET_ENVVAR,
        md5=emit.tarball_md5(archive),
    )
    artifacts[D1_SNIPPET_PATH] = snippet.encode("ascii")
    return artifacts, archive


# --------------------------------------------------------------------------------------------
# the example build
# --------------------------------------------------------------------------------------------


def load_layer2() -> tuple[bytes, provenance.ProvDocument]:
    """Read the example's Layer-2 file as BYTES and check it is canonical.

    Binary, never text mode: the digest is taken over these exact bytes, and a text-mode read on
    Windows would silently hand back a different byte string than the file contains. The canonical
    check makes a hand edit that reflows the JSON a loud failure rather than a digest that quietly
    stops matching what a reader downloads.
    """
    raw = LAYER2_PATH.read_bytes()
    try:
        provenance.check_canonical_bytes(raw)
    except ValueError as exc:
        raise SystemExit(f"{LAYER2_PATH.relative_to(ROOT)}: {exc}") from None
    return raw, provenance.from_json_obj(json.loads(raw.decode("ascii")))


def build_table(raw: bytes, document: provenance.ProvDocument) -> spec.G4DatTable:
    """Render the example's Layer-1 table from its Layer-2 document plus the example numbers."""
    declared = {f"{z}-{a}" for z, a, *_ in EXAMPLE_RECORDS}
    if declared != set(document.rows):
        raise SystemExit(
            f"row sets disagree: {LAYER2_PATH.name} has {sorted(document.rows)}, this script has "
            f"{sorted(declared)}. Every Layer-1 record needs a Layer-2 row and vice versa."
        )
    directives = {
        "GRAMMAR": spec.GRAMMAR_VERSION,
        "DATASET": document.dataset,
        "VERSION": document.version,
        "PROFILE": document.profile,
        "SEAM": document.seam,
        "TABLE": TABLE_NAME,
        "GENERATOR": f"openmucf-g4 {openmucf.__version__}",
        "SOURCEDIGEST": provenance.source_digest(raw),
        "UNITS": UNITS,
        "COLUMNS": COLUMNS,
        "VALIDITY": VALIDITY,
    }
    table = spec.G4DatTable(directives=directives, records=EXAMPLE_RECORDS)
    spec.validate(table)
    provenance.check_against_table(table, document)
    # A wiring assertion, not an integrity check: the digest three lines up was just derived from
    # `raw`, so this can only fire if a caller hands build_table() two different byte strings. The
    # integrity check that has teeth is in audit(), against the two COMMITTED files.
    provenance.check_source_digest(table, raw)
    return table


def build_example_artifacts() -> tuple[dict[Path, bytes], bytes]:
    """The committed example artifacts keyed by path, plus the archive they describe."""
    raw, document = load_layer2()
    table = build_table(raw, document)
    layer1 = spec.render(table).encode("ascii")
    archive = emit.build_tarball({LAYER1_PATH.name: layer1, LAYER2_PATH.name: raw})
    snippet = emit.add_dataset_snippet(
        name=document.dataset,
        version=document.version,
        filename=document.dataset,
        envvar=DATASET_ENVVAR,
        md5=emit.tarball_md5(archive),
    )
    return {LAYER1_PATH: layer1, SNIPPET_PATH: snippet.encode("ascii")}, archive


# --------------------------------------------------------------------------------------------
# both builds
# --------------------------------------------------------------------------------------------


def build_artifacts() -> tuple[dict[Path, bytes], dict[str, bytes]]:
    """Every committed artifact of both builds, plus each build's archive keyed by name."""
    example, example_archive = build_example_artifacts()
    d1, d1_archive = build_d1_artifacts()
    return {**example, **d1}, {"example": example_archive, "d1": d1_archive}


def _write(artifacts: dict[Path, bytes], directory: Path) -> None:
    """Write every artifact under ``directory``, preserving its path RELATIVE to ``data/g4``.

    Relative paths, not bare names: both builds emit a `geant4_add_dataset.snippet`, and flattening
    them would make one silently overwrite the other -- in the audit's temp directory, where the
    consequence is a byte-comparison that passes against the wrong file.
    """
    for path, payload in artifacts.items():
        target = directory / path.relative_to(G4DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)  # binary: LF stays LF on every platform


def regenerate() -> None:
    artifacts, archives = build_artifacts()
    _write(artifacts, G4DIR)
    for path in artifacts:
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    for name, archive in archives.items():
        print(
            f"archive {name}.{emit.ARCHIVE_EXTENSION} (not committed): "
            f"{len(archive)} bytes, md5={emit.tarball_md5(archive)}"
        )


def audit() -> None:
    """Rebuild into a temp dir and byte-compare against what is committed."""
    # The cross-layer invariant on every COMMITTED pair, checked FIRST -- both files read from disk,
    # neither derived from the other. It has to run before the regenerate-and-compare below or it can
    # never fire: Layer 1 embeds sha256(Layer 2), so any drift between the two also changes the
    # regenerated bytes, and the byte-diff would report "an artifact differs" while this line, which
    # names both layers and gives the consumer-facing code, never executed.
    pairs = (
        (LAYER1_PATH, LAYER2_PATH),
        (D1_CAPTURE_LAYER1, D1_CAPTURE_LAYER2),
        (D1_ZEFF_LAYER1, D1_ZEFF_LAYER2),
    )
    for layer1_path, layer2_path in pairs:
        try:
            committed = spec.parse(layer1_path.read_bytes().decode("ascii"))
            provenance.check_source_digest(committed, layer2_path.read_bytes())
        except spec.G4DatFormatError as exc:
            # A coded, located message and a clean exit, like every other failure here -- not the
            # stack trace a bare raise would print at whoever is running `make audit`.
            #
            # BOTH file names, not just Layer 1's. This check fires when the two disagree, and the
            # side that moved is at least as often Layer 2 -- naming only the file carrying the
            # digest sends whoever is reading to edit the wrong one. Caught by the mutation drill,
            # which corrupts each artifact in turn and requires the audit to name it.
            raise SystemExit(
                f"g4data audit FAILED: the committed pair "
                f"{layer1_path.relative_to(ROOT).as_posix()} and "
                f"{layer2_path.relative_to(ROOT).as_posix()} do not agree: {exc}"
            ) from None

    artifacts, archives = build_artifacts()
    with tempfile.TemporaryDirectory() as scratch:
        fresh = Path(scratch)
        _write(artifacts, fresh)
        drifted = [
            path.relative_to(G4DIR).as_posix()
            for path in artifacts
            if not (
                path.exists()
                and filecmp.cmp(fresh / path.relative_to(G4DIR), path, shallow=False)
            )
        ]
    if drifted:
        raise SystemExit(
            "g4data audit FAILED: regenerated artifact(s) differ from the committed copy: "
            + ", ".join(sorted(drifted))
        )
    # Determinism is a property of the builder, not of one run: build again and compare.
    _, rebuilt = build_artifacts()
    for name, archive in archives.items():
        if rebuilt[name] != archive:
            raise SystemExit(f"g4data audit FAILED: two {name} archive builds in one process differ")
    print(
        f"g4data audit OK: {len(artifacts)} artifact(s) byte-identical to the committed copy across "
        f"{len(archives)} build(s); committed cross-layer digests verified (E009 clean on "
        f"{len(pairs)} pair(s)); archives reproducible in-process ("
        + ", ".join(f"{name} md5={emit.tarball_md5(archive)}" for name, archive in archives.items())
        + ")"
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--audit" in argv:
        audit()
    else:
        regenerate()


if __name__ == "__main__":
    main()
