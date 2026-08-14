"""openmucf.g4.provenance -- Layer 2 (``*.prov.json``): the schema, its canonical bytes, and the
digest that binds it to Layer 1.

Layer 2 is never read by the transport code. It carries what Layer 1 deliberately leaves out: where
each number came from, what kind of uncertainty it is, **which evaluation it belongs to**, and the
disclosures a consumer cannot reconstruct from the numbers themselves. Field names match this
project's rate ledger (``openmucf/data/rates.schema.json``) wherever the concept already exists, so
there is one provenance vocabulary rather than one per dataset.

Two rules here do real work:

* ``precedence`` is an **ordered list declared as data**, not a rule compiled into a reader -- the
  same discipline that turns an analytic fallback into a ``#FALLBACK`` directive rather than an
  ``else`` branch;
* ``isotope_resolved`` is **required on every row**, so the disclosure of whether a row is an
  isotope-resolved measurement or an element value wearing an isotope label cannot be skipped by
  omission.

The digest invariant (``FORMAT_SPEC.md`` section 3): Layer 1's ``#SOURCEDIGEST`` is the SHA-256 of
the **exact bytes** of the Layer-2 file, which are :func:`document_bytes`. Not an LF-normalized
copy, not a re-indented one, not an in-memory object -- one specific byte string, so that
:func:`check_source_digest` cannot pass on a file that differs from the one that was hashed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from typing import Any

from .spec import (
    ALLOWED_SEAMS,
    PROFILE_PATTERN,
    G4DatFormatError,
    G4DatTable,
    _canonical_lines,  # the one line map, shared rather than reimplemented here
    _split_fields,  # and the one field splitter, for the same reason
)

__all__ = [
    "ProvDocument",
    "ProvRow",
    "check_against_table",
    "check_canonical_bytes",
    "check_source_digest",
    "document_bytes",
    "from_json_obj",
    "render_json",
    "source_digest",
    "to_json_obj",
    "validate_document",
]

#: Where a row's value came from. ``geant4-compiled-in`` is a first-class library: reproducing what
#: a transport code does today is an evaluation like any other, and saying so is the honest label.
SOURCE_LIBRARIES = ("geant4-compiled-in", "suzuki1987", "iwamoto2025", "jendl-mund", "openmucf")
#: Same vocabulary as the rate ledger's schema.
UNC_TYPES = ("stat", "exp", "theory", "theory-spread", "model", "table", "estimate", "exact")
RECOMMENDATIONS = ("recommended", "superseded", "")
FILE_FIELDS = ("dataset", "version", "profile", "seam", "precedence", "rows")
#: ``"Z-A"`` in decimal with **no zero padding**. The looser ``^[0-9]+-[0-9]+$`` implemented only the
#: parenthesized half of section 3's rule and let ``"1-1"``, ``"01-1"`` and ``"001-001"`` be three
#: distinct JSON keys for one record -- the exact collision that paragraph exists to prevent, since
#: JSON object keys are strings and nothing normalizes them.
_ROW_KEY_PATTERN = re.compile(r"^(0|[1-9][0-9]*)-(0|[1-9][0-9]*)$")


@dataclass(frozen=True)
class ProvRow:
    """Provenance for one Layer-1 record. Every field is required on every row."""

    source_bibkey: str
    source_locator: str
    unc_type: str
    conditions: str
    validity_range: str
    evaluation_method: str
    single_source: bool
    needs_verification: bool
    recommendation: str
    evaluation_id: str
    source_library: str
    isotope_resolved: bool


@dataclass(frozen=True)
class ProvDocument:
    """A Layer-2 file: the file-level fields plus one :class:`ProvRow` per record, keyed ``"Z-A"``."""

    dataset: str
    version: str
    profile: str
    seam: str
    precedence: tuple[str, ...]
    rows: dict[str, ProvRow]


ROW_FIELDS = tuple(f.name for f in fields(ProvRow))
_BOOL_ROW_FIELDS = ("single_source", "needs_verification", "isotope_resolved")


# --------------------------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------------------------


def _require_string(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string, got {value!r}")
    if not allow_empty and not value:
        raise ValueError(f"{where} must not be empty")
    return value


def _validate_row(key: str, row: Any) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"row {key!r} must be an object, got {row!r}")
    present = set(row)
    missing = [name for name in ROW_FIELDS if name not in present]
    if missing:
        raise ValueError(f"row {key!r} is missing required field(s): {', '.join(missing)}")
    unknown = sorted(present - set(ROW_FIELDS))
    if unknown:
        raise ValueError(f"row {key!r} carries unknown field(s): {', '.join(unknown)}")

    for name in ROW_FIELDS:
        value = row[name]
        if name in _BOOL_ROW_FIELDS:
            if not isinstance(value, bool):
                raise ValueError(f"row {key!r} field {name!r} must be a boolean, got {value!r}")
            continue
        _require_string(value, f"row {key!r} field {name!r}", allow_empty=name == "recommendation")

    if row["unc_type"] not in UNC_TYPES:
        raise ValueError(f"row {key!r} unc_type {row['unc_type']!r} is not one of {', '.join(UNC_TYPES)}")
    if row["recommendation"] not in RECOMMENDATIONS:
        raise ValueError(f"row {key!r} recommendation {row['recommendation']!r} is not a known value")
    if row["source_library"] not in SOURCE_LIBRARIES:
        raise ValueError(
            f"row {key!r} source_library {row['source_library']!r} is not one of "
            f"{', '.join(SOURCE_LIBRARIES)}"
        )


def _validate_precedence(precedence: Any) -> None:
    """An ordered list of known ``source_library`` values -- the precedence rule, declared as data."""
    if not isinstance(precedence, list):
        raise ValueError(f"precedence must be an ordered list, got {precedence!r}")
    if not precedence:
        raise ValueError("precedence must name at least one source_library")
    seen: set[str] = set()
    for entry in precedence:
        if entry not in SOURCE_LIBRARIES:
            raise ValueError(f"precedence entry {entry!r} is not one of {', '.join(SOURCE_LIBRARIES)}")
        if entry in seen:
            raise ValueError(f"precedence entry {entry!r} is repeated; an ordering cannot rank it twice")
        seen.add(entry)


def validate_document(obj: Any) -> None:
    """Check a decoded Layer-2 object against the schema, raising ``ValueError`` on the first problem.

    Layer 2 is JSON and has no line numbers, so it does not use the Layer-1 error codes; the one
    exception is the cross-layer digest, which is E009 (:func:`check_source_digest`).
    """
    if not isinstance(obj, dict):
        raise ValueError(f"a Layer-2 document must be a JSON object, got {obj!r}")
    missing = [name for name in FILE_FIELDS if name not in obj]
    if missing:
        raise ValueError(f"document is missing required field(s): {', '.join(missing)}")
    unknown = sorted(set(obj) - set(FILE_FIELDS))
    if unknown:
        raise ValueError(f"document carries unknown field(s): {', '.join(unknown)}")

    _require_string(obj["dataset"], "dataset")
    _require_string(obj["version"], "version")
    profile = _require_string(obj["profile"], "profile")
    if PROFILE_PATTERN.match(profile) is None:
        raise ValueError(f"profile {profile!r} is not a {PROFILE_PATTERN.pattern} token")
    seam = _require_string(obj["seam"], "seam")
    if seam not in ALLOWED_SEAMS:
        raise ValueError(f"seam {seam!r} is not one of {', '.join(ALLOWED_SEAMS)}")
    _validate_precedence(obj["precedence"])

    rows = obj["rows"]
    if not isinstance(rows, dict):
        raise ValueError(f"rows must be an object keyed 'Z-A', got {rows!r}")
    for key, row in rows.items():
        if _ROW_KEY_PATTERN.match(key) is None:
            raise ValueError(f"row key {key!r} is not of the form 'Z-A'")
        _validate_row(key, row)


def to_json_obj(document: ProvDocument) -> dict[str, Any]:
    """The document as plain JSON types (``precedence`` becomes a list, rows become objects)."""
    return {
        "dataset": document.dataset,
        "version": document.version,
        "profile": document.profile,
        "seam": document.seam,
        "precedence": list(document.precedence),
        "rows": {
            key: {name: getattr(row, name) for name in ROW_FIELDS} for key, row in document.rows.items()
        },
    }


def from_json_obj(obj: Any) -> ProvDocument:
    """Validate a decoded Layer-2 object and build a :class:`ProvDocument` from it."""
    validate_document(obj)
    return ProvDocument(
        dataset=obj["dataset"],
        version=obj["version"],
        profile=obj["profile"],
        seam=obj["seam"],
        precedence=tuple(obj["precedence"]),
        rows={key: ProvRow(**row) for key, row in obj["rows"].items()},
    )


# --------------------------------------------------------------------------------------------
# canonical bytes + the digest that binds the two layers
# --------------------------------------------------------------------------------------------


def render_json(document: ProvDocument) -> str:
    """Canonical serialization: sorted keys, two-space indent, ASCII-only, one trailing newline.

    Deterministic by construction, so a regenerated Layer-2 file byte-diffs cleanly and its digest
    is reproducible on any platform.
    """
    validate_document(to_json_obj(document))
    return json.dumps(to_json_obj(document), sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def document_bytes(document: ProvDocument) -> bytes:
    """The exact bytes of the Layer-2 file. Write these; do not re-encode them in text mode, or a
    CRLF translation will change the file the digest was taken over."""
    return render_json(document).encode("ascii")


def check_canonical_bytes(raw: bytes) -> None:
    """Reject Layer-2 bytes that are valid JSON but not in the canonical form of section 3.

    The canonical form is normative, for a mechanical reason: the digest is taken over the file's
    bytes, so a re-indented or key-reordered file is a *different* Layer-2 file with a different
    digest, and nothing downstream would say why. This check is here, in the shipped package, rather
    than in the generator script -- ``packages`` ships ``openmucf`` and ``openmucf.g4`` and not
    ``scripts/``, so a rule enforced only there is a rule no installed consumer can apply.
    """
    try:
        obj = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Layer-2 bytes are not ASCII JSON: {exc}") from None
    canonical = document_bytes(from_json_obj(obj))
    if raw != canonical:
        raise ValueError(
            "Layer-2 bytes are valid but NOT in canonical form (sorted keys, two-space indent, "
            "ASCII-escaped, LF endings, exactly one trailing newline). The digest is taken over "
            "these bytes, so the file must be written with document_bytes(): "
            f"{len(raw)} bytes on disk, {len(canonical)} bytes canonical."
        )


def source_digest(data: bytes | ProvDocument) -> str:
    """SHA-256 of the Layer-2 file bytes, as ``#SOURCEDIGEST`` carries it."""
    payload = data if isinstance(data, bytes) else document_bytes(data)
    return hashlib.sha256(payload).hexdigest()


def check_source_digest(table: G4DatTable, layer2: bytes) -> None:
    """E009 -- ``#SOURCEDIGEST`` must equal the SHA-256 of ``layer2``, the Layer-2 file's bytes.

    This is the only rejection that needs both layers: a ``.g4dat`` is fully verified only together
    with the Layer-2 file it was generated from.
    """
    dline, _ = _canonical_lines(table.directives)
    if "SOURCEDIGEST" not in table.directives:
        raise G4DatFormatError("E002", dline("SOURCEDIGEST"), "missing required directive '#SOURCEDIGEST'")
    declared = table.directives["SOURCEDIGEST"]
    actual = source_digest(layer2)
    if declared != actual:
        raise G4DatFormatError(
            "E009",
            dline("SOURCEDIGEST"),
            f"'#SOURCEDIGEST' is {declared!r} but the Layer-2 file hashes to {actual!r}",
        )


def check_against_table(table: G4DatTable, document: ProvDocument) -> None:
    """The file-level fields must agree with the Layer-1 directives they mirror.

    Raises ``ValueError``: a disagreement means the two layers describe different datasets, which is
    a generation bug rather than a malformed file.
    """
    for field_name, directive in (
        ("dataset", "DATASET"),
        ("version", "VERSION"),
        ("profile", "PROFILE"),
        ("seam", "SEAM"),
    ):
        theirs = table.directives.get(directive)
        ours = getattr(document, field_name)
        if theirs != ours:
            raise ValueError(
                f"Layer-2 {field_name} {ours!r} does not match Layer-1 '#{directive}' {theirs!r}"
            )

    # Section 3's "one object per Layer-1 record", enforced in the shipped package rather than in a
    # build script -- the same reason check_canonical_bytes() lives here. A row set that has drifted
    # from the table is a dataset whose provenance does not describe what it ships.
    columns = _split_fields(table.directives.get("COLUMNS", ""))
    key_indices = [columns.index(name) for name in ("Z", "A") if name in columns]
    if len(key_indices) != 2:
        # Section 3's row key is `"Z-A"`, so it is only defined for a table declaring both. Rather
        # than silently skipping the rule -- which let a single-key table carry any rows at all --
        # say so. BOTH directions are a violation: rows that cannot be keyed against the table, and
        # records that no row can describe. Checking only the first left records shipping with no
        # provenance at all, which is the half section 3 most cares about.
        if document.rows or table.records:
            raise ValueError(
                "Layer-2 rows are keyed 'Z-A', which is defined only for a table declaring both "
                f"'Z' and 'A'; this table declares {' '.join(columns) or 'no columns'} and carries "
                f"{len(table.records)} record(s) against {len(document.rows)} row(s)"
            )
        return
    expected = {"-".join(str(int(record[i])) for i in key_indices) for record in table.records}
    missing = sorted(expected - set(document.rows))
    extra = sorted(set(document.rows) - expected)
    if missing or extra:
        raise ValueError(
            "Layer-2 rows do not match the Layer-1 records one-for-one: "
            f"{len(missing)} record(s) with no row ({', '.join(missing[:5]) or 'none'}), "
            f"{len(extra)} row(s) with no record ({', '.join(extra[:5]) or 'none'})"
        )
