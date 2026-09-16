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
muon-capture record and every effective charge in its `parity` tables is bit-for-bit what Geant4
v11.4.2 compiles in.
Both layers are generated from the vendored upstream source, at
build time, every time -- so "bit-for-bit" is a property of this script rather than a claim about
what somebody typed once. Layer 2 is still the source of truth in the sense that matters: it is the
byte range ``#SOURCEDIGEST`` is taken over, and ``--audit`` verifies that on the COMMITTED pair.

The D1 directory also carries a **second profile of the capture table**, ``mizuno2025``, generated
the same way from two committed transcriptions of a primary's printed tables
(``openmucf.g4.sources.mizuno2025`` states what they hold and refuses what they must not).

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
import functools
import json
import sys
import tempfile
from pathlib import Path

import openmucf
from openmucf.g4 import emit, provenance, spec
from openmucf.g4.sources import d1_nuclear_capture as d1src
from openmucf.g4.sources import mizuno2025 as mizsrc

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
#: The second capture profile: a table generated from two committed transcriptions of a primary
#: rather than from a vendored source, shipped beside the parity pair under the profile's name.
D1_MIZUNO_LAYER1 = D1DIR / f"d1_capture.{mizsrc.PROFILE}.g4dat"
D1_MIZUNO_LAYER2 = D1DIR / f"d1_capture.{mizsrc.PROFILE}.prov.json"
MIZUNO_TABLE1_PATH = ROOT / mizsrc.TABLE1_RELPATH
MIZUNO_TABLE3_PATH = ROOT / mizsrc.TABLE3_RELPATH

#: The version moves with the archive: this one adds the capture table's second profile as a pair
#: of members beside the parity pair, and the previous one packed the members under the dataset
#: directory Geant4 unpacks to with the generated `README` and `History`. Plainly distinct from the
#: example's `0.0.0-example`, and below 1.0.0 because D1 alone is not the dataset.
D1_VERSION = "0.3.0"
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
#: The parity clause, shared by every capture row. What follows it differs per row, because
#: `isotope_resolved` no longer has one derivation: most rows are now settled by a primary read and
#: a few are not, and a reader must be able to tell which from the row alone.
CAPTURE_PARITY_CLAUSE = (
    "compiled-in constant table transcribed by Geant4; reproduced here bit-for-bit, not "
    "re-evaluated."
)
#: Settled by a primary -- in EITHER direction. `false` here is a finding ("the primary shows this
#: value to rest on a natural-composition target"), not the absence of one, which is exactly the
#: distinction `needs_verification` exists to carry.
CAPTURE_METHOD_SETTLED = (
    CAPTURE_PARITY_CLAUSE + " isotope_resolved was established by reading the primary literature "
    "this record's value is attributed to, not derived from the shape of this table: {evidence}."
)
#: Not settled: the primary was read and does not decide THIS RECORD. Note what that exposes about
#: the rule this replaced. "More than one rate at one Z" is evidence about the Z -- it shows the
#: underlying data distinguishes isotopes -- and it was read as evidence about each row of that Z,
#: which does not follow: one of those rows can still be the natural-composition entry. These
#: records sit in exactly that gap, so the flag is false in the field's stated sense, "not
#: established", and needs_verification says the question is open rather than answered.
CAPTURE_METHOD_UNSETTLED = (
    CAPTURE_PARITY_CLAUSE + " isotope_resolved is NOT established for this record: {evidence}. It "
    "is therefore false in this field's stated sense -- not established, which is not the same "
    "claim as established-to-be-unresolved -- and needs_verification is true."
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


def _isotope_locator(line: int, audit: d1src.IsotopeAuditRow) -> str:
    """The vendored-source locator, plus the primary locator that establishes the flag.

    Two clauses, labelled, because they answer two different questions and conflating them would be
    a provenance error. The first says where the VALUE came from and must keep resolving inside this
    repository (a `parity` row reproduces Geant4, not a paper). The second says what established the
    row's `isotope_resolved` flag, and is absent on exactly the rows no primary settles -- so the
    presence of the second clause is itself the machine-checkable statement that the row is settled.
    """
    base = _locator(line)
    if not audit.settled:
        return base
    return (
        f"{base}; isotope_resolved established by {audit.locator} "
        f"[copy read: {audit.copy_read}]"
    )


def build_capture_document(found: d1src.D1Extraction) -> provenance.ProvDocument:
    """Layer 2 for the capture table: one row per record, every field decided by rule."""
    audit = d1src.load_isotope_audit(ROOT / d1src.AUDIT_RELPATH)
    keys = {(z, a) for z, a, _, _ in found.capture_records}
    if set(audit) != keys:
        # Never build from a partial audit. A missing key would otherwise fall back to some default
        # and ship a flag nobody derived, which is the failure this whole layer exists to prevent.
        missing = sorted(keys - set(audit))
        extra = sorted(set(audit) - keys)
        raise SystemExit(
            "g4data build FAILED: the isotope audit does not cover the extracted records exactly; "
            f"missing {missing}, unexpected {extra}"
        )
    general = _quote_upstream(
        found.capture_comment_lines, "capture data from", "Suzuki", "weighted average"
    )
    hydrogen = _quote_upstream(found.capture_comment_lines, "Hydrogen")
    helium = _quote_upstream(found.capture_comment_lines, "Helium")

    # The primary's printed cells at the Z of every row the audit had left open. A row the listing
    # alone cannot settle is decided by comparing its compiled-in value with those cells, and the
    # comparison travels in the row whichever way it came out.
    cells = d1src.load_capture_cells(ROOT / d1src.CAPTURE_CELLS_RELPATH)
    blocks = d1src.cells_by_z(cells)
    orphan_blocks = sorted(set(blocks) - {z for z, _ in audit})
    if orphan_blocks:
        raise SystemExit(
            f"g4data build FAILED: the printed cells carry a block at Z {orphan_blocks} with no "
            "audit key; a block nothing is compared with has no reason to be shipped"
        )
    uncovered = sorted({z for (z, _), finding in audit.items() if not finding.settled} - set(blocks))
    if uncovered:
        raise SystemExit(
            f"g4data build FAILED: an unsettled audit row at Z {uncovered} has no block of printed "
            "cells to be compared with"
        )
    values = {(z, a): (value, unc) for z, a, value, unc in found.capture_records}
    literals = {
        (z, a): literal
        for (z, a, _, _), literal in zip(found.capture_records, found.capture_literals, strict=True)
    }

    rows = {}
    for (z, a, _, _), line in zip(found.capture_records, found.capture_lines, strict=True):
        upstream = {1: hydrogen, 2: helium}.get(z, general)
        finding = audit[(z, a)]
        template = CAPTURE_METHOD_SETTLED if finding.settled else CAPTURE_METHOD_UNSETTLED
        method = template.format(evidence=finding.evidence)
        if z in blocks and d1src.decided_by_value((z, a), finding.evidence, blocks[z]):
            matches = d1src.printed_matches(*values[(z, a)], blocks[z])
            # The audit's finding must be the comparison's outcome, never a flag beside it: one
            # equal cell settles the row to that entry, any other count leaves it open.
            if (len(matches) == 1) is not finding.settled:
                raise SystemExit(
                    f"g4data build FAILED: the audit row ({z}, {a}) is "
                    f"{'settled' if finding.settled else 'open'} but the compiled-in value equals "
                    f"{d1src.render_outcome(matches)} of the primary's printed cells at Z={z}"
                )
            if finding.settled and (
                finding.locator != matches[0].locator
                or finding.copy_read != matches[0].copy_read
                or finding.isotope_resolved is not d1src.is_separated_label(matches[0].label)
            ):
                raise SystemExit(
                    f"g4data build FAILED: the audit row ({z}, {a}) is settled to a locator, copy "
                    f"or flag other than the one printed cell it equals, {matches[0]!r}"
                )
            method = method + " " + d1src.render_comparison(
                z, *literals[(z, a)], blocks[z], matches
            )
        rows[f"{z}-{a}"] = provenance.ProvRow(
            source_bibkey=D1_BIBKEY,
            source_locator=_isotope_locator(line, finding),
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
            evaluation_method=method,
            # Upstream says "weighted average of the two most precise measurements"; asserting a
            # single source would be a claim this project cannot make.
            single_source=False,
            # False once a primary settles the row -- in either direction. It stays true only where
            # the primary was read and does not decide, which is a narrower and more useful claim
            # than the blanket "nothing here has been checked" this shipped before.
            needs_verification=not finding.settled,
            # A parity profile reproduces; it does not recommend.
            recommendation="",
            evaluation_id=_evaluation_id("capRates"),
            source_library=D1_SOURCE_LIBRARY,
            isotope_resolved=finding.isotope_resolved,
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
    # The printed cells of the primary's two tables, read off the page images and committed. A row
    # the audit covers was read against the primary; a row it does not cover is left as it was.
    audit = d1src.load_zeff_audit(ROOT / d1src.ZEFF_AUDIT_RELPATH)
    for z in audit:
        if not 0 <= z < len(found.zeff):
            raise SystemExit(
                f"zeff audit row Z={z} is not an index of the upstream zeff array; the audit "
                "names a cell this table does not carry"
            )

    rows = {}
    for z, line in enumerate(found.zeff_lines):
        unreachable = z < zmin or z > zmax
        cell = audit.get(z)
        underlined = cell is not None and cell.underlined
        # An underlined cell is the primary's own estimate mark. The tail after the quotation then
        # stops before the clause that names the uncertainty type: `unc_type` carries the fact, the
        # audit row carries the mark, and the locator's second clause says where it was read.
        conditions = (
            f'quoted from the upstream source comment: "{upstream}". No uncertainty is published '
            "upstream and this table carries no unc column"
            + ("." if underlined else ", so unc_type is table.")
        )
        locator = _locator(line)
        method = ZEFF_METHOD
        if cell is not None:
            locator += f"; printed cell read in {cell.locator} [copy read: {cell.copy_read}]"
            method += (
                f" The printed cell {cell.printed_z}({cell.printed_zeff}) was read from the "
                "primary and equals this value."
            )
            if cell.printed_z != z:
                method += (
                    f" The primary prints it against Z {cell.printed_z}, a Z it prints a second "
                    "time on that element's own row; the element column decides."
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
            source_locator=locator,
            unc_type="estimate" if underlined else "table",
            conditions=conditions,
            validity_range=(
                f"Z={z}; unreachable, the clamp maps it to Z={zmin}"
                if unreachable
                else f"Z={z}; GetMuonZeff clamps its argument into [{zmin}, {zmax}] before indexing"
            ),
            evaluation_method=method,
            single_source=False,
            # False on exactly the rows the audit read against the primary; the locator's second
            # clause is the machine-checkable statement of which rows those are.
            needs_verification=cell is None,
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


# --------------------------------------------------------------------------------------------
# D1 -- the mizuno2025 capture profile, from two committed transcriptions of the primary
# --------------------------------------------------------------------------------------------

# The provenance text every row quotes -- the primary's account of its printed value, its Table 1
# caption and the copy read -- lives beside the transcription it describes (`mizsrc.METHOD`,
# `mizsrc.METHOD_AVERAGED`, `mizsrc.TABLE1_CAPTION`, `mizsrc.COPY`).


def _mizuno_conditions(row: mizsrc.Table3Row, printed: tuple[mizsrc.Table1Row, ...]) -> str:
    """Quoted fragments of the primary and its printed cells for this nuclide, by rule per row."""
    huff = ", ".join(sorted({r.huff_factor for r in printed}))
    parts = [
        mizsrc.TABLE1_CAPTION,
        f"printed Q for this target: {huff}",
    ]
    for r in printed:
        parts.append(
            f'Table 1 row: "{r.form}", lifetime {r.lifetime_ns} ns, uncertainty {r.lifetime_unc_ns} ns'
        )
    if row.note:
        parts.append(f'Table 3 footnote: "{row.note}"')
    parts.append(mizsrc.COPY)
    return "; ".join(parts)


def build_mizuno_capture_document(found: mizsrc.Mizuno2025Extraction) -> provenance.ProvDocument:
    """Layer 2 for the ``mizuno2025`` capture table: one row per Table-3 record, every field by rule."""
    rows = {}
    for row in found.table3:
        printed = found.table1_rows(row.nuclide)
        where = f"Z={row.z} natural composition" if row.a == 0 else f"Z={row.z} A={row.a}"
        rows[f"{row.z}-{row.a}"] = provenance.ProvRow(
            source_bibkey=mizsrc.BIBKEY,
            source_locator=f"{row.locator} [copy read: {row.copy_read}]",
            unc_type="exp",
            conditions=_mizuno_conditions(row, printed),
            validity_range=f"{where}",
            evaluation_method=(
                mizsrc.METHOD_AVERAGED.format(note=row.note) if row.note else mizsrc.METHOD
            ),
            # Derived from the primary's own comparison column: a target it prints no earlier
            # lifetime for is one it alone has measured.
            single_source=not any(r.has_suzuki_value for r in printed),
            # Every value is read from the primary itself, and its locator names the table.
            needs_verification=False,
            recommendation="",
            evaluation_id=f"{mizsrc.PROFILE}-table3",
            source_library=mizsrc.PROFILE,
            # The key scheme IS the disclosure: an enriched or mononuclidic nuclide by its mass
            # number, a natural-composition target by A = 0.
            isotope_resolved=row.a != 0,
        )
    return provenance.ProvDocument(
        dataset=DATASET_NAME,
        version=D1_VERSION,
        profile=mizsrc.PROFILE,
        seam=D1_SEAM,
        precedence=(mizsrc.PROFILE,),
        rows=rows,
    )


def build_mizuno_capture_table(found: mizsrc.Mizuno2025Extraction, digest: str) -> spec.G4DatTable:
    """Layer 1 for the ``mizuno2025`` capture table: the printed decimals, records ascending by
    ``(Z, A)``. No ``#SOURCESHA`` (the file reproduces no upstream revision) and no ``#FALLBACK``."""
    z_values = sorted({row.z for row in found.table3})
    directives = {
        "GRAMMAR": spec.GRAMMAR_VERSION,
        "DATASET": DATASET_NAME,
        "VERSION": D1_VERSION,
        "PROFILE": mizsrc.PROFILE,
        "SEAM": D1_SEAM,
        "TABLE": D1_CAPTURE_TABLE,
        "GENERATOR": f"openmucf-g4 {openmucf.__version__}",
        "SOURCEDIGEST": digest,
        "UNITS": "value=1e6/s unc=1e6/s",
        "COLUMNS": "Z A value unc",
        # `natural_and_listed`: the enriched and mononuclidic nuclides by their mass number, the
        # natural-composition targets by A = 0.
        "VALIDITY": f"Z:{z_values[0]}-{z_values[-1]} A:{spec.A_NATURAL_AND_LISTED}",
    }
    records = tuple(
        sorted(
            ((row.z, row.a, float(row.rate), float(row.rate_unc)) for row in found.table3),
            key=lambda record: (record[0], record[1]),
        )
    )
    return spec.G4DatTable(directives=directives, records=records)


def build_d1_artifacts() -> tuple[dict[Path, bytes], bytes]:
    """The committed D1 artifacts keyed by path, plus the archive they describe (not committed)."""
    found = d1src.load(VENDORED_PATH)  # checks the upstream pins before anything is generated
    mizuno = mizsrc.load(MIZUNO_TABLE1_PATH, MIZUNO_TABLE3_PATH)

    members: dict[str, bytes] = {}
    artifacts: dict[Path, bytes] = {}
    files: list[emit.TableEntry] = []
    for layer1_path, layer2_path, document, build in (
        (
            D1_CAPTURE_LAYER1, D1_CAPTURE_LAYER2, build_capture_document(found),
            functools.partial(build_capture_table, found),
        ),
        (
            D1_ZEFF_LAYER1, D1_ZEFF_LAYER2, build_zeff_document(found),
            functools.partial(build_zeff_table, found),
        ),
        (
            D1_MIZUNO_LAYER1, D1_MIZUNO_LAYER2, build_mizuno_capture_document(mizuno),
            functools.partial(build_mizuno_capture_table, mizuno),
        ),
    ):
        raw = provenance.document_bytes(document)
        table = build(provenance.source_digest(raw))
        spec.validate(table)
        provenance.check_against_table(table, document)
        provenance.check_source_digest(table, raw)
        layer1 = spec.render(table).encode("ascii")
        artifacts[layer1_path] = layer1
        artifacts[layer2_path] = raw
        members[layer1_path.name] = layer1
        members[layer2_path.name] = raw
        files.append(
            (
                layer1_path.name,
                layer2_path.name,
                table.directives["TABLE"],
                document.profile,
                document.seam,
                len(document.rows),
            )
        )

    # README and History are archive members only, like the archive itself: pure functions of the
    # Layer-2 documents and the member names, so nothing about them needs a committed copy.
    members["README"] = emit.readme_member(name=DATASET_NAME, version=D1_VERSION, files=files)
    members["History"] = emit.history_member(name=DATASET_NAME, version=D1_VERSION, files=files)
    archive = emit.build_tarball(
        members, directory=emit.dataset_directory(DATASET_NAME, D1_VERSION)
    )
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
    files: list[emit.TableEntry] = [
        (
            LAYER1_PATH.name,
            LAYER2_PATH.name,
            TABLE_NAME,
            document.profile,
            document.seam,
            len(document.rows),
        )
    ]
    members = {
        LAYER1_PATH.name: layer1,
        LAYER2_PATH.name: raw,
        "README": emit.readme_member(
            name=document.dataset, version=document.version, files=files
        ),
        "History": emit.history_member(
            name=document.dataset, version=document.version, files=files
        ),
    }
    archive = emit.build_tarball(
        members, directory=emit.dataset_directory(document.dataset, document.version)
    )
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
    """Every committed artifact of both builds, plus each build's archive keyed by its file name."""
    example, example_archive = build_example_artifacts()
    d1, d1_archive = build_d1_artifacts()
    _, example_document = load_layer2()
    return {**example, **d1}, {
        emit.archive_name(example_document.dataset, example_document.version): example_archive,
        emit.archive_name(DATASET_NAME, D1_VERSION): d1_archive,
    }


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
        print(f"archive {name} (not committed): {len(archive)} bytes, md5={emit.tarball_md5(archive)}")


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
        (D1_MIZUNO_LAYER1, D1_MIZUNO_LAYER2),
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
