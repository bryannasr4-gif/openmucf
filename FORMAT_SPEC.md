# FORMAT_SPEC.md -- the `G4MuonicData` external-data format

Version of this document: **grammar 1.0**.

This document specifies a small, dependency-free file format for shipping muonic-atom physics data
to Geant4 (and to any other transport code) **with its provenance and its uncertainty attached**.
It is normative: an implementation that disagrees with this document is wrong, and the reference
implementation (`openmucf/g4/spec.py`) is tested against every rule stated here.

Names are provisional. The dataset name `G4MuonicData` and the environment variable `G4MUONICDATA`
are placeholders chosen so that tooling can be built and tested; they may change before any upstream
registration.

---

## 1. Why two layers

Geant4's core must be able to read a dataset **with no third-party dependency** -- no JSON, XML or
HDF5 library. That constraint rules out the obvious answer ("ship JSON") for the file Geant4 reads.
Research-grade provenance and uncertainty, on the other hand, need a rich, nested representation.

So the dataset has two layers, shipped together:

| Layer | File | Read by Geant4 | Content |
|---|---|---|---|
| 1 | `*.g4dat` | **yes** | plain US-ASCII directives + numeric records; a C++ reader needs only the standard library |
| 2 | `*.prov.json` | never | one JSON object per Layer-1 record: bibliographic source, uncertainty type, evaluation identity, disclosure flags |

**Binding invariant.** Layer 1 is *generated from* Layer 2 by one tool, and Layer 1's
`#SOURCEDIGEST` directive is the SHA-256 of the Layer-2 file's bytes. The two cannot drift without a
checkable error (`E009`).

---

## 2. Layer 1 -- the `.g4dat` grammar

### 2.1 Lexical rules

1. Encoding is **US-ASCII only**. Any byte outside `0x00`-`0x7F` is an error (`E005`).
2. Line ending is **LF (`\n`) only**. A carriage return anywhere is an error (`E006`), whether it
   appears as CRLF or as a lone CR.
3. There is **no byte-order mark**. A BOM is a non-ASCII byte and is rejected by rule 1.
4. The file **ends with a newline-terminated `#END` line** (`E012`).
5. There are **no free-form comments**. Every line is a directive, a record, or the terminator.
   Commentary belongs in Layer 2. An unrecognised `#` keyword is a hard error (`E001`), not a
   skipped comment: silently ignoring `#FOO` is how format drift starts and how a typo becomes
   data loss.
6. There is **no wall-clock timestamp anywhere** in the format. A file's identity is `#VERSION`
   plus `#SOURCEDIGEST`. This is what makes byte-for-byte regeneration audits possible.

### 2.2 Directives

A directive line has `#` in column 1, an uppercase keyword, whitespace, then the value. **One
directive per line.** The value is everything after the separating whitespace, with leading and
trailing whitespace removed; internal whitespace is preserved verbatim.

Directives appear in **exactly this order**. A directive that repeats, or that appears after a
directive that should follow it, is an error (`E003`).

| # | Directive | Required | Value |
|---|---|---|---|
| 1 | `#GRAMMAR` | yes | version of *this format*, `MAJOR.MINOR`; currently `1.0` |
| 2 | `#DATASET` | yes | dataset name, e.g. `G4MuonicData` |
| 3 | `#VERSION` | yes | version of the *dataset content* (independent of `#GRAMMAR`) |
| 4 | `#PROFILE` | yes | which evaluation this file carries (section 2.5) |
| 5 | `#SEAM` | yes | which physics seam the table serves (section 2.5) |
| 6 | `#TABLE` | yes | table name within the seam, e.g. `nuclear_capture_rate` |
| 7 | `#GENERATOR` | yes | producing tool and its version |
| 8 | `#SOURCEDIGEST` | yes | SHA-256 of the Layer-2 file, 64 lowercase hex characters |
| 9 | `#SOURCESHA` | iff `#PROFILE parity` | revision of the upstream source the parity profile reproduces |
| 10 | `#UNITS` | yes | `name=unit` assignments, e.g. `rate=1e6/s` |
| 11 | `#COLUMNS` | yes | whitespace-separated column names, e.g. `Z A value unc` |
| 12 | `#VALIDITY` | yes | where the table applies, e.g. `Z:1-94 A:natural_and_listed` |
| 13 | `#FALLBACK` | optional | analytic fallback declared **as data**, e.g. `goulard_primakoff b0a=-0.03 b0b=-0.25 b0c=3.24 t1=875e-9` |

`#GRAMMAR` is deliberately **separate from `#VERSION`**: one is the version of the format, the other
the version of the data. Without that separation there is no backward-compatibility story at all.

`#SOURCESHA` is required **if and only if** `#PROFILE` is `parity`: a parity file exists to reproduce
a specific upstream revision bit-for-bit, so it must name that revision; a non-parity file is not
reproducing anything and must not claim to (`E013` in both directions).

An analytic fallback is carried as declared data rather than as compiled-in behaviour so that it is
versioned, testable, swappable and citable like any other row.

Example header:

```
#GRAMMAR      1.0
#DATASET      G4MuonicData
#VERSION      1.0.0
#PROFILE      parity
#SEAM         d1_nuclear_capture
#TABLE        nuclear_capture_rate
#GENERATOR    openmucf-g4 1.1.0
#SOURCEDIGEST 3b1f...64 lowercase hex total...
#SOURCESHA    8cc04f65977807f1848da7b958c421cd5e162f26
#UNITS        rate=1e6/s
#COLUMNS      Z A value unc
#VALIDITY     Z:1-94 A:natural_and_listed
```

### 2.3 Records

After the directives come **zero or more records**, one per line.

1. Fields are whitespace-delimited. Leading whitespace is permitted, so a writer may align columns.
   A reader must treat any run of spaces or tabs as one separator and must not depend on alignment.
2. The number of fields must equal the arity of `#COLUMNS` (`E004`).
3. Columns named `Z` and `A` are **integer columns**: the field must match `^[0-9]+$`.
4. Every other column is a **float column**: the field must match the strict C-locale float
   `^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$`. **A comma decimal separator is a syntax error
   (`E007`), never a silent truncation** -- see section 6.
5. A field that denotes a non-finite value (`inf`, `+inf`, `-inf`, `infinity`, `nan`, in any letter
   case), or a lexically valid field that overflows to infinity, is rejected (`E014`).
6. `(Z, A)` is the **primary key**. A repeated key is an error naming the first-seen line (`E008`).
7. Records are sorted **ascending by `(Z, A)`** (`E015`). Deterministic output requires it and it
   makes diffs readable.

A table whose `#COLUMNS` contains neither `Z` nor `A` has no primary key; rules 6 and 7 then have
nothing to check and are not enforced. Every table defined for this format so far carries `Z` and
`A` as its first two columns.

### 2.4 Terminator

The record block ends with a line containing exactly `#END`, terminated by a newline. Any further
line -- including an empty one -- is content after the terminator (`E011`). A file with no
newline-terminated `#END` line is incomplete (`E012`).

### 2.5 Allowed values

`#PROFILE` is a token matching `^[a-z][a-z0-9_-]{2,31}$` (`E016` otherwise). `parity` and
`evaluated` are the conventional names; the token set is deliberately open so that **N competing
evaluations can coexist** (`iwamoto2025`, `jendl-mund`, ...), each as its own file in the same
archive, rather than as extra columns in one file. Keeping competing evaluations in separate files
is what allows the C++ reader to stay trivial: per-row evaluation bookkeeping is Layer-2 business.

`#SEAM` is one of `d1_nuclear_capture`, `d2_atomic_capture`, `d3_transitions`, `d4_mucf_cycle`
(`E016` otherwise).

### 2.6 Floats

Floats are written with `%.17g` and **must round-trip exactly**: reading the emitted text back gives
the identical IEEE-754 double. Seventeen significant decimal digits are sufficient for that
guarantee for every finite `binary64` value, including subnormals. The consequence is that a value
entered as `0.000725` is emitted as `0.00072499999999999995`: the file records the double that is
actually used, not a prettier decimal that is a different number.

Python's float formatting is locale-independent, so the emitter is unaffected by `LC_NUMERIC`; the
reference implementation proves that with a test that renders under a comma-decimal locale and
compares bytes. C and C++ readers are **not** automatically safe -- see section 6.

### 2.7 Versioning and backward compatibility

`#GRAMMAR` is `MAJOR.MINOR`.

- A reader **must reject a `MAJOR` it does not know** (`E010`). A malformed `#GRAMMAR` value is
  likewise rejected with `E010`, because no supported major can be established from it.
- A reader accepts any `MINOR` of a `MAJOR` it knows.
- Because unknown directives are a hard error, a `MINOR` increment that introduces a new directive
  will be rejected by an older reader **as soon as a file actually uses that directive**. A `MINOR`
  increment is therefore transparently backward-compatible only when it widens the allowed values of
  an existing directive (a new `#SEAM` token, say). Producers that must serve older readers emit no
  directive above the reader's minor.
- A change to the meaning of an existing directive, to the record grammar, or to the terminator is a
  `MAJOR` increment.

---

## 3. Layer 2 -- `*.prov.json`

Layer 2 is never read by Geant4. It is a single JSON object with the file-level fields below and one
object per Layer-1 record under `rows`, keyed `"Z-A"`.

**File-level fields**

| Field | Meaning |
|---|---|
| `dataset` | must equal Layer 1's `#DATASET` |
| `version` | must equal Layer 1's `#VERSION` |
| `profile` | must equal Layer 1's `#PROFILE` |
| `seam` | must equal Layer 1's `#SEAM` |
| `precedence` | **ordered** list of `source_library` values, most-preferred first |

`precedence` is the precedence rule **declared as data**, for the same reason the analytic fallback
is: a rule that lives in a file can be versioned, diffed, cited and disagreed with. A rule compiled
into a reader cannot.

**Per-row fields** (all required on every row)

| Field | Type | Meaning |
|---|---|---|
| `source_bibkey` | string | key resolvable in the accompanying bibliography |
| `source_locator` | string | table/equation/page within that source |
| `unc_type` | string | `stat`, `exp`, `theory`, `theory-spread`, `model`, `table`, `estimate`, `exact` |
| `conditions` | string | conditions under which the value applies |
| `validity_range` | string | range over which the row is claimed valid |
| `evaluation_method` | string | how the recommended value was arrived at |
| `single_source` | bool | true if the row traces to one never-independently-rechecked measurement |
| `needs_verification` | bool | true if the digit or locator is not yet pinned from the primary text |
| `recommendation` | string | `recommended`, `superseded`, or empty |
| `evaluation_id` | string | identifies *which* evaluation this row belongs to |
| `source_library` | string | `geant4-compiled-in`, `suzuki1987`, `iwamoto2025`, `jendl-mund`, `openmucf` |
| `isotope_resolved` | bool | **disclosure**: is this row an isotope-resolved value, or an element value carrying an isotope label? |

The first nine field names are identical to those used by this project's rate ledger
(`openmucf/data/rates.schema.json`), so there is one provenance vocabulary across the project rather
than one per dataset.

`evaluation_id` and `source_library` exist so that **two evaluations of the same `(Z, A)` can
coexist** in the corpus without either being silently averaged away or dropped.

`isotope_resolved` is mandatory on every row and may not be omitted. Isotope dependence of muon
capture rates is an open question in the literature, while compiled-in tables are keyed `(Z, A)`
throughout; which rows carry a genuinely isotope-resolved measurement is invisible to a consumer
unless the dataset says so. Making the flag required means the disclosure cannot be quietly skipped.

**The digest invariant.** Layer 1's `#SOURCEDIGEST` is `sha256` over the **exact bytes of the
Layer-2 file**, which are `json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True)` followed by
a single `\n`, encoded as ASCII. Any other byte range is a different number: the digest is over the
serialized file, not over an in-memory object, and not over an LF-normalized or re-indented copy.

---

## 4. Error codes

Every rejection carries an exact code and a **1-based line number**. The message shape is exactly:

```
{code}: {human text} (line {n})
```

| Code | Condition |
|---|---|
| `E001` | unknown directive |
| `E002` | missing required directive |
| `E003` | directive out of order (including a repeated directive) |
| `E004` | record field count differs from the `#COLUMNS` arity |
| `E005` | non-ASCII byte |
| `E006` | CRLF or CR line ending |
| `E007` | field does not match its column's lexical class (strict C-locale float, or `^[0-9]+$` for `Z` and `A`) |
| `E008` | duplicate `(Z, A)` key; the message names the first-seen line |
| `E009` | `#SOURCEDIGEST` differs from the SHA-256 of the Layer-2 file |
| `E010` | unsupported `#GRAMMAR` major version |
| `E011` | content after `#END` |
| `E012` | missing newline-terminated `#END` |
| `E013` | `#PROFILE parity` without `#SOURCESHA`, or `#SOURCESHA` under a non-parity profile |
| `E014` | non-finite float (`inf` / `nan`) |
| `E015` | records not sorted ascending by `(Z, A)` |
| `E016` | `#PROFILE` or `#SEAM` value outside its allowed set |

**Reporting order.** Whole-file lexical checks run first, encoding (`E005`) before line structure
(`E006`), because decoding precedes splitting into lines. Then the file is read line by line, so the
first structural or record error in file order is the one reported. Within one record, field count
(`E004`) is checked before field lexis (`E007`, `E014`), and a duplicate key (`E008`) is reported
before an ordering violation (`E015`) -- otherwise a duplicate would always surface as "not
ascending" and `E008` would be unreachable.

`E009` is the only code that cannot be raised by reading the Layer-1 file alone: it requires the
Layer-2 file as well. A `.g4dat` is only fully verified together with the Layer-2 file it was
generated from.

---

## 5. Finding the dataset at run time -- **both** discovery modes

A consumer inside Geant4 resolves data directories with `G4FindDataDir("G4MUONICDATA")` from
`G4EnvironmentUtils.hh` (the sanctioned lookup since Geant4 11.1). That is necessary but **not
sufficient to describe what users will experience**, because there are two distinct modes and they
have very different prerequisites. Verified against Geant4 v11.4.2:

**Mode 1 -- registered.** The dataset has an entry in `cmake/Modules/G4DatasetDefinitions.cmake`
(`geant4_add_dataset(NAME ... VERSION ... FILENAME ... EXTENSION tar.gz ... ENVVAR ... MD5SUM ...)`),
which generates a compiled-in `dataset_definitions[]` table. `G4FindDataDir` resolves the dataset
underneath `GEANT4_DATA_DIR` with no per-dataset environment variable set. This mode requires an
upstream change to Geant4 itself.

**Mode 2 -- unregistered, explicit environment variable.** For a dataset that is not in
`dataset_definitions[]`, an explicitly exported `G4MUONICDATA=/path/to/dataset` is the **only** way
to find it. At v11.4.2 `geant4.sh` exports no per-dataset `G4*DATA` variables at all -- only
`GEANT4_DATA_DIR` -- so nothing else will resolve it.

**Every early adopter hits mode 2**, because mode 1 does not exist until an upstream merge. Anything
that documents only mode 1 is documenting a configuration that no external user can have yet.

A reader must therefore distinguish three outcomes and never confuse them:

1. dataset found (either mode) -- use it;
2. variable unset **and** dataset unregistered -- report a **precise, actionable error** naming the
   variable it looked for; do **not** silently fall through to compiled-in values;
3. dataset found but unreadable or failing validation -- report the error code and line from
   section 4, not a generic failure.

---

## 6. Rules for C and C++ readers

**Do not use `std::strtod`, `atof`, `sscanf("%lf")`, or an unqualified `std::istream >> double`.**
All of them honour the process's `LC_NUMERIC`. Under a comma-decimal locale (`de_DE`, `fr_FR`,
`pt_BR`, ...) `strtod("0.00072499999999999995")` stops at the `.` and returns `0` -- silently, with
no error flag a caller usually checks. A dataset of rates read that way is a dataset of zeros and
the job still finishes.

Use one of:

- `std::from_chars(first, last, value)` (C++17; locale-independent by construction), or
- a stream explicitly imbued with the classic locale: `stream.imbue(std::locale::classic())`.

Two further requirements follow from section 2:

- The reader must reject a `#GRAMMAR` major it does not know rather than guess.
- The reader must not accept a comma decimal separator under any locale. The grammar has exactly one
  numeric syntax and it is the C-locale one.

---

## 7. The reference implementation

`openmucf/g4/spec.py` is the reference implementation of section 2 and section 4:

- `parse(text: str) -> G4DatTable` -- parse a Layer-1 file, raising `G4DatFormatError(code, line, message)`.
- `render(table: G4DatTable) -> str` -- emit a Layer-1 file.
- `validate(table: G4DatTable) -> None` -- check an in-memory table against section 2.
- `format_float(x: float) -> str` -- the `%.17g` float syntax of section 2.6.

`openmucf/g4/provenance.py` is the reference implementation of section 3, including the
`#SOURCEDIGEST` check (`E009`).

Guarantees, each covered by a test:

1. **Round-trip.** For every table that `validate()` accepts, `parse(render(t)) == t`.
2. **Determinism.** `render(t)` is a pure function of `t`: two calls produce identical bytes, the
   output contains no timestamp, and the bytes do not change under a comma-decimal `LC_NUMERIC`.
3. **Only conforming output.** `render()` first sorts the records ascending by `(Z, A)` and then
   validates; it therefore accepts a table whose records are not yet sorted, and everything it emits
   is accepted by `parse()`. `validate()` itself reports an unsorted table as `E015`.
4. **Exact floats.** `float(format_float(x)) == x` for every finite double.

`validate()` reports the line number the offending item **would** occupy in the emitted file, so an
in-memory table and a parsed file report errors the same way.

A table that no file could ever produce -- a directive value carrying leading or trailing
whitespace, or an embedded newline -- is a programming error rather than a format error: `validate()`
raises `ValueError` for it, and the section-4 codes remain exactly the sixteen file-level conditions.
`format_float()` likewise raises `ValueError` on a non-finite input, since at that layer there is no
line number to report; `E014` is raised by `validate()` and `parse()`, which have one.

---

## 8. Not included in this release

The archive packaging (deterministic tarball plus checksum), the dataset generator, and the C++
reader and its standalone validation application are specified here but are not part of this
release. Section 5 and section 6 are stated now precisely so that the reader, when written, cannot
get them wrong by accident.
