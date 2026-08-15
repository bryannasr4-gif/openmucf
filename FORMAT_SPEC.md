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

1. Encoding is **US-ASCII only**, and narrower than that: the only permitted bytes are
   **`0x09` (TAB), `0x0A` (LF), `0x0D` (CR), and `0x20`-`0x7E`**. Any other byte -- non-ASCII, or an
   ASCII control character such as VT (`0x0B`) or FF (`0x0C`) -- is an error (`E005`).
   The narrow set is what makes "whitespace" mean exactly space and tab (rule 2.3.1): a VT is a
   separator to some standard-library split functions and an ordinary field character to a
   space/tab reader, so a file containing one would have two different field counts depending on
   who read it. CR is inside the permitted set only so that it receives the more specific `E006`.
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

Directives appear in **exactly this order**. A directive that repeats, that appears after a
directive that should follow it, or that appears after the record block has begun, is an error
(`E003`).

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

**A directive present with an empty value carries no information, and is treated as absent.** For a
**required** directive that is `E002`, reported at that directive's own line: a `#PROFILE` with
nothing after it has not told the reader which evaluation the file carries, and saying so is more
useful than reporting the empty string as a malformed token. `E002` is the code even where another
would otherwise apply — an empty `#GRAMMAR` declares no version at all, so it is `E002`, not `E010`.
For `#SOURCESHA` the same principle drives `E013` (below).

Where a directive is **not required**, the rule is about meaning rather than validity: the file is
still conforming, and it is not byte-identical to one that omits the line — two distinct files, one
meaning. That applies to an empty `#FALLBACK` (which declares no fallback) and to an empty
`#SOURCESHA` under a non-parity profile (which claims no upstream revision, and so does not trip
`E013`'s second direction).

`#SOURCESHA` is required **if and only if** `#PROFILE` is `parity`: a parity file exists to reproduce
a specific upstream revision bit-for-bit, so it must name that revision; a non-parity file is not
reproducing anything and must not claim to (`E013` in both directions). An empty `#SOURCESHA` is a
parity file claiming to reproduce nothing, which is exactly the claim `E013` exists to stop — and it
is **`E013`, not `E002`**, even though the empty-counts-as-absent rule is what makes it a violation:
`#SOURCESHA` is never in the always-required set, so its absence is only ever a `#PROFILE` problem.

An analytic fallback is carried as declared data rather than as compiled-in behaviour so that it is
versioned, testable, swappable and citable like any other row.

#### Which values the reader itself checks

Layer 1 enforces a lexical form on exactly those values that are **load-bearing for the reader**;
every other value is one opaque string to it. This is a decision, not an omission, and the boundary
is where it is for a reason:

| Directive | Enforced form | Why the reader must check it |
|---|---|---|
| `#GRAMMAR` | `^(0\|[1-9]\d*)\.(0\|[1-9]\d*)$` (`E010`) | it decides whether to read the file at all |
| `#SOURCEDIGEST` | exactly 64 lowercase hex characters, `^[0-9a-f]{64}$` (`E016`) | it carries section 1's binding invariant |
| `#COLUMNS` | one or more names matching `^[A-Za-z_][A-Za-z0-9_]*$`, all **distinct** (`E016`) | it determines record arity and the primary key |
| `#PROFILE`, `#SEAM` | the value sets of section 2.5 (`E016`) | they select the evaluation and the seam |

`#SOURCEDIGEST` is checked lexically **as well as** against Layer 2 (`E009`) because a standalone
Layer-1 validator has no Layer-2 file and could otherwise say nothing at all about the field the
whole two-layer design rests on. Distinct `#COLUMNS` names matter for the same kind of reason:
"the primary key is whichever of `Z` and `A` the table declares" (section 2.3 rule 6) means nothing
on a table that declares `Z` twice, and the second column would be silently unreachable.

`#DATASET`, `#VERSION`, `#TABLE`, `#GENERATOR` and `#SOURCESHA` are constrained **only** to being
non-empty; grammar 1.0 pins no internal syntax for them. `#UNITS`, `#VALIDITY` and `#FALLBACK` are
likewise one string to the reader — their sub-grammars, below, bind the **consumer**.

#### Value sub-grammars

To Layer 1 the values of `#UNITS`, `#VALIDITY` and `#FALLBACK` are **one string** each -- the parser
does not decompose them, and no error code concerns their internal structure (unlike the four
directives in the table above, which the reader does check). The sub-grammars here therefore bind
the **consumer**, not the reader: they are what a C++ implementation must be able to parse out of
those three values once it has the string, and stating them now is what stops three implementations
from inventing three different splittings of the same bytes.

| Directive | Value | Element |
|---|---|---|
| `#UNITS` | one or more space/tab-separated assignments | `NAME=UNIT`, `NAME` matching `^[A-Za-z_][A-Za-z0-9_]*$`, `UNIT` a non-empty run of printable non-whitespace bytes |
| `#VALIDITY` | one or more space/tab-separated assignments | `NAME:RANGE`, `NAME` as above, `RANGE` a non-empty run of printable non-whitespace bytes |
| `#FALLBACK` | a model name, then zero or more assignments | `MODEL` matching `^[a-z][a-z0-9_]*$`, then `NAME=VALUE` with `NAME` as above and `VALUE` a float in the section-2.3 rule-4 syntax |

Every `NAME` used in `#UNITS` should be a `#COLUMNS` name; a unit for a column that does not exist
is a producer bug, and a consumer may report it as one. `#VALIDITY` names need not be columns
(`A:natural_and_listed` describes a selection rule, not a column range).

**A `#FALLBACK` model name means whatever the dataset's own documentation says it means**, and that
documentation must state the formula **and its evaluation order**. This is not pedantry: floating
point addition and multiplication are not associative, so two consumers that re-group the same
expression compute different numbers from the same file, and a dataset shipping a fallback without a
pinned association order cannot be reproduced bit-for-bit by anyone. A model definition must
therefore also say whether a conforming evaluation may contract operations (fuse a multiply and an
add) — for the `goulard_primakoff` model of the D1 dataset, see `DATASET_D1.md`, which says it may
not, and why.

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
#UNITS        value=1e6/s unc=1e6/s
#COLUMNS      Z A value unc
#VALIDITY     Z:1-94 A:natural_and_listed
```

(The `#UNITS` line names `value` and `unc` — the columns — rather than the quantity. An earlier
revision of this document showed `rate=1e6/s` here, against the paragraph directly above requiring
every `#UNITS` name to be a `#COLUMNS` name: the example violated its own advisory, and the advisory
is the part that is right.)

### 2.3 Records

After the directives come **zero or more records**, one per line.

1. Fields are separated by **space (`0x20`) and tab (`0x09`) only** -- no other byte is a separator,
   because no other whitespace byte may appear at all (rule 2.1.1). Any run of them is one
   separator, and **leading and trailing whitespace are both permitted and are not fields**: strip
   the line of spaces and tabs at both ends before splitting it. A writer may therefore align
   columns, and a reader must not depend on the alignment. Stating the trailing case matters as much
   as the leading one: a reader that splits without stripping the end sees a trailing separator run
   as one more (empty) field and reports `E004` on a file every other reader accepts.
2. The number of fields must equal the arity of `#COLUMNS` (`E004`). A blank line is a record with
   zero fields and is rejected the same way: the format has no blank lines and no comments. This is
   also why `#COLUMNS` may not be empty (section 2.2): at arity zero a blank line *would* be a
   conforming record, and a file of blank lines would round-trip.
3. Columns named `Z` and `A` are **integer columns**: the field must match `^[0-9]+$` and its value
   must lie in **`0`-`9999` inclusive** (`E007` otherwise). The bound is what keeps a reader's
   integer width from being implementation-defined -- without it, whether a file is readable
   depends on whether the reader chose `int`, `long` or `int64_t`. It is physically generous:
   `Z <= 118` and `A` does not reach 300. An integer column has no other lexical class, so `nan` or
   `1e3` there is `E007` (it is not a number of the kind that column accepts), never `E014`.
4. Every other column is a **float column**: the field must match the strict C-locale float
   `^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$`. **A comma decimal separator is a syntax error
   (`E007`), never a silent truncation** -- see section 6. **Test rule 5's non-finite literals
   first**: `inf`, `nan` and friends do not match this pattern, so a reader that applies the pattern
   before checking for them reports `E007` where this format requires `E014`.
5. **In a float column**, a field that denotes a non-finite value (`inf`, `+inf`, `-inf`,
   `infinity`, `nan`, in any letter case), or a lexically valid field that leaves the representable
   range in either direction --
   **overflowing to infinity or underflowing to zero** -- is rejected (`E014`). Underflow is
   included for the same reason as overflow and matters just as much in practice: `1e-999` converts
   to `0.0` silently in some languages, while a C++ `std::from_chars` reports
   `result_out_of_range` for it, so accepting the field would mean two conforming readers disagree
   about the same file. **"Underflow" here means the result is *exactly zero*, not merely
   subnormal**: `4.9406564584124654e-324` is representable, is required to round-trip (section 2.6),
   and is accepted. A field that is **lexically zero** (no digit `1`-`9` before the exponent, e.g.
   `0`, `0.0`, `-0.0`, `0e-999`) is a genuine zero and is likewise accepted.
6. The **primary key** is whichever of `Z` and `A` the table declares, in that order -- `(Z, A)` for
   the usual table, `(Z)` or `(A)` for one that declares only one of them. A repeated key is an
   error naming the first-seen line (`E008`), and the message names the key columns that table
   actually declares.
7. Records are sorted **ascending by the primary key**, compared as **integers, most significant
   key column first** -- never as text (`E015`). Numerically `2` precedes `10`; lexicographically
   `"10"` precedes `"2"`, so a reader that compares the field text sorts the same file differently.
   Deterministic output requires the order and it makes diffs readable.

A table whose `#COLUMNS` contains neither `Z` nor `A` has no primary key; rules 6 and 7 then have
nothing to check and are not enforced. Every table defined for this format so far carries `Z` and
`A` as its first two columns.

### 2.4 Terminator

The record block ends with a line containing `#END`, terminated by a newline. **`#END` starts in
column 1**, like every other directive; trailing spaces and tabs after it are ignored, leading ones
are not (a line beginning with whitespace is a record, and is diagnosed as one). Any further line --
including an empty one -- is content after the terminator (`E011`). A file with no
newline-terminated `#END` line is incomplete (`E012`), which covers both a missing `#END` and a file
whose last line has no newline.

**`#END` is not a directive.** It is not part of the directive order of section 2.2, and a reader
must not diagnose it through the directive machinery. A terminator line that carries anything beyond
spaces and tabs -- `#END x` -- is **`E011` at that line**, in every position, because the content is
*on* the terminator. Diagnosing it as an unknown or out-of-order directive produces a message that
is false about the terminator, and produces a *different* false message depending on whether records
happened to precede it. A line such as `#ENDX` is a different thing: its keyword is `ENDX`, and it is
diagnosed as the directive it claims to be.

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

`#GRAMMAR`'s value is exactly `MAJOR.MINOR`, lexically `^(0|[1-9]\d*)\.(0|[1-9]\d*)$` -- two runs of
decimal digits separated by one `.`, with no sign, no third component, no pre-release suffix, no
leading `v`, and **no leading zeros**. `01.0` is not a spelling of `1.0`; it is rejected (`E010`).
One version has one spelling, in a format whose identity discipline is byte-exactness throughout.

- A reader **must reject a `MAJOR` it does not know** (`E010`). A `#GRAMMAR` value that does not
  match the lexical form above is likewise rejected with `E010`, because no major can be
  established from it at all — **except an empty value, which is `E002`** and not `E010`: a
  directive with nothing after it has not declared a version rather than declared a bad one, and
  section 2.2's empty-counts-as-absent rule governs it. This carve-out is stated because the two
  rules otherwise both apply to the same file and a reader would have to guess which wins.
- A reader **accepts any `MINOR` of a `MAJOR` it knows** -- that is, it accepts the version
  *declaration*. Accepting the declaration is not the same as being able to read the file: a
  higher-minor file may use a directive this reader does not have, and it will then be rejected
  (`E001`) on that directive's line.
- `#GRAMMAR` is validated **eagerly, at its own line**, before any later line is diagnosed. Every
  other diagnosis is only meaningful under a grammar the reader implements, so `E010` preempts what
  follows it; without that rule, a file written to a future grammar reports its new directive as
  "unknown", which is a true statement about the wrong problem.
- Because unknown directives are a hard error, a `MINOR` increment that introduces a new directive
  is rejected by an older reader **as soon as a file actually uses that directive**. A `MINOR`
  increment is therefore transparently backward-compatible only when it widens the allowed values
  of an **open** value set -- a new `#PROFILE` token, say, which matches an existing pattern rather
  than joining a fixed list. Widening a **closed** set is not backward-compatible: `#SEAM`'s values
  are enumerated (section 2.5), so an older reader rejects a new seam token with `E016`.
  Producers that must serve older readers emit no directive above the reader's minor.
- A change to the meaning of an existing directive, to the record grammar, or to the terminator is a
  `MAJOR` increment.

---

## 3. Layer 2 -- `*.prov.json`

Layer 2 is never read by Geant4. It is a single JSON object with the file-level fields below and one
object per Layer-1 record under `rows`, keyed `"Z-A"`.

**Row-key format, exactly.** A key is the record's primary key, written in **decimal with no zero
padding, no sign and no whitespace**. There are two forms, and which one a file uses is decided by
the Layer-1 table's `#COLUMNS`, never by the Layer-2 file:

| The table declares | Key form | Examples |
|---|---|---|
| both `Z` and `A` | `Z`, a single `-`, then `A` — `^(0\|[1-9][0-9]*)-(0\|[1-9][0-9]*)$` | `"1-1"`, `"29-63"`, `"94-242"` |
| exactly one of them | that column's integer — `^(0\|[1-9][0-9]*)$` | `"0"`, `"29"`, `"100"` |

Never `"001-001"`, `"+1-1"`, `"1 - 1"` or `"029"`. JSON object keys are strings, so without a pinned
spelling `"1-1"` and `"01-1"` would be two different keys for one record. (Layer 1's integer *fields*
are laxer — `^[0-9]+$`, section 2.3 rule 3 — because a reader converts them to integers and the
writer re-emits them canonically, so no two spellings survive. A JSON key is never normalized by
anything, which is why this one is strict.)

The single-key form exists because not every table has both coordinates: an effective-charge table
is a per-`Z` quantity, and a mass number is not a thing it has. The alternative — a sentinel `A`
column of zeros — was rejected: it would put a column in every file that a C++ reader has to skip,
and it would state something false, since `0` is not a mass number.
`openmucf.g4.provenance.check_against_table()` decides which form applies from the table and rejects
both directions of mismatch — a row that keys nothing, and a record that no row describes.

**One row per record.** `rows` has exactly one object per Layer-1 record and no others;
`openmucf.g4.provenance.check_against_table()` enforces it alongside the file-level fields.

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

**Canonical serialization (normative).** A Layer-2 file is written **only** in this form: keys
sorted at every level, two-space indentation, ASCII-escaped (`ensure_ascii`), LF line endings, and
exactly one trailing newline. This is not a formatting preference -- it is what makes the digest
below reproducible, so a Layer-2 file that is valid JSON but not in canonical form is not a valid
Layer-2 file. `openmucf.g4.provenance.check_canonical_bytes()` is the check, and it lives in the
shipped package rather than in a build script so that any consumer can apply it: a re-indented file
is a *different* file with a *different* digest, and nothing downstream would say why.

**The digest invariant.** Layer 1's `#SOURCEDIGEST` is `sha256` over the **exact bytes of the
Layer-2 file**, which are `json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True)` followed by
a single `\n`, encoded as ASCII. Any other byte range is a different number: the digest is over the
serialized file, not over an in-memory object, and not over an LF-normalized or re-indented copy.
In particular, writing those bytes through a text-mode stream that translates LF to CRLF produces a
file whose digest no longer matches, and `E009` is what catches it.

---

## 4. Error codes

Every rejection carries an exact code and a **1-based line number**. The message shape is exactly:

```
{code}: {human text} (line {n})
```

| Code | Condition |
|---|---|
| `E001` | unknown directive |
| `E002` | missing required directive, or one present with an empty value |
| `E003` | directive out of order (including a repeated directive) |
| `E004` | record field count differs from the `#COLUMNS` arity |
| `E005` | byte outside `{TAB, LF, CR, 0x20-0x7E}` (non-ASCII, or an ASCII control character) |
| `E006` | CRLF or CR line ending |
| `E007` | field does not match its column's lexical class (strict C-locale float, or `^[0-9]+$` within `0`-`9999` for `Z` and `A`) |
| `E008` | duplicate primary key; the message names the first-seen line |
| `E009` | `#SOURCEDIGEST` differs from the SHA-256 of the Layer-2 file |
| `E010` | unsupported or malformed `#GRAMMAR` version |
| `E011` | content after the `#END` terminator, or on the terminator line itself |
| `E012` | missing newline-terminated `#END` |
| `E013` | `#PROFILE parity` without a non-empty `#SOURCESHA`, or a **non-empty** `#SOURCESHA` under a non-parity profile |
| `E014` | float outside the representable range: non-finite (`inf` / `nan`), overflow to infinity, or underflow to zero |
| `E015` | records not sorted ascending by the primary key |
| `E016` | a directive value outside the set or lexical form section 2.2 requires of it -- `#PROFILE`, `#SEAM`, `#SOURCEDIGEST`, `#COLUMNS`; the message names the offending directive |

**Reporting order.** A file usually has one defect, but when it has several the reader must be
predictable about which one it names, or two conforming implementations will disagree on a file
they both correctly reject. The order is **three phases**, and it is not simply "first error in file
order":

1. **Whole-file lexical.** `E005`, then `E006`. Both are properties of the byte stream rather than
   of any one line, and decoding precedes splitting into lines, so these run before anything else
   -- **including the eager `#GRAMMAR` check of phase 2** -- and `E005` runs before `E006`. A file
   with a forbidden byte on line 40 and an unreadable `#GRAMMAR` on line 1 reports `E005`.
2. **Line scan, in file order.** Each line is diagnosed as it is reached: `E001`, `E003` and `E011`
   for header and block structure, `E004` then `E007`/`E014` within a record (field count before
   field lexis, since the lexis of a field the record should not have is not interesting). The
   `#GRAMMAR` line is checked **eagerly, as it is reached**, which makes its verdict the one that
   preempts everything after it -- see section 2.7 for why. That verdict is `E010` for a version
   this reader cannot read, and `E002` for an empty value (section 2.2); **the preemption belongs to
   the position, not to the code**, so an empty `#GRAMMAR` preempts a later defect exactly as an
   unsupported one does.
3. **Block-close and post-scan.** Checks that cannot be decided from one line: `E002`, `E013` and
   `E016` **when the directive block closes**, then `E012`, then `E008`, then `E015` once all records
   are in hand. These are **reported at the line of the directive or record at fault**, which may
   be an *earlier* line than a phase-2 defect that preempted them. A directive that is **missing**
   has no line of its own, so it is reported at the line it **would have occupied** — count the
   directives that precede it in the section-2.2 order and are present, and add one.

   Within the block-close group the order is fixed, because several of these can be true at once:
   **all `E002` first**, in section-2.2 directive order; then `E016` for `#PROFILE`, `#SEAM`,
   `#SOURCEDIGEST`, `#COLUMNS`, in that order; then `E013`. This is a priority order, not a
   line order — an `E016` on line 8 is reported ahead of an `E013` whose fault line is 4.

**The directive block closes at the first record line, or at `#END`, whichever comes first** -- and
this is a rule about the reader's control flow, not a footnote. The header checks run *there*, so:

- a file that has records but no `#END`, and is missing `#VALIDITY`, reports **`E002`**: the block
  closed at the first record line, long before the missing terminator was noticed;
- a file with **no records and no `#END`** reports **`E012`** in place of *any block-close* defect
  (`E002`, `E013`, `E016`), because the block never closes and those checks never run.

Phase 2 is unaffected by all of this: a defect found during the line scan -- `E001`, `E003`, `E004`,
`E007`, `E014`, and `E010` at the `#GRAMMAR` line -- is reported when the scan reaches it, whether or
not the block ever closes. A header-only file carrying an unknown directive reports `E001` at that
directive's line, not `E012`.

That second case is the ordinary shape of a truncated download, and `E012` is both the true and the
more useful diagnosis for it: telling someone whose file was cut off that their `#VALIDITY` is
missing sends them to fix a header that is probably fine.

Two same-line tie-breaks, so no reader has to guess them. **The line that closes the block is
diagnosed as a block close first**: a file missing `#VALIDITY` whose first record is also malformed
reports `E002`, not the record's `E004`/`E007`, because the header rules run at that line before it
is read as a record. And **the directive order is checked before the eager `#GRAMMAR` rule**: a
`#GRAMMAR` that arrives out of order is `E003` at its own line, not `E010`, because it has not been
accepted as the grammar declaration yet.

The consequence worth stating plainly: **a per-line defect, or an unclosed block, can preempt a
block-level defect on an earlier line.** A file whose `#PROFILE` on line 4 lacks its `#SOURCESHA`
*and* whose line 12 repeats a directive reports `E003` on line 12, not `E013` on line 4, because the
block does not close until line 12 is passed. `E010` is the deliberate exception -- it is raised
eagerly at the `#GRAMMAR` line and preempts everything after it. Fix the reported defect and re-run:
the reader is a decision procedure for "is this file conforming", not a defect enumerator.

Within phase 3, a duplicate key (`E008`) is reported before an ordering violation (`E015`) --
otherwise a duplicate would always surface as "not ascending" and `E008` would be unreachable.

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

- `std::from_chars(first, last, value)` -- **C++17 or later**, which is also the standard Geant4
  itself is built with; locale-independent by construction, and the recommended choice; or
- a stream explicitly imbued with the classic locale: `stream.imbue(std::locale::classic())`.

Three mechanical details that will otherwise be got wrong:

- **Strip a leading `+` before calling `std::from_chars`.** The grammar of section 2.3 rule 4 allows
  `+1.5`; `std::from_chars` does **not** accept a leading `+` and returns
  `std::errc::invalid_argument` for it. A reader that passes the field through unmodified rejects a
  conforming file.
- **Map `std::errc::result_out_of_range` to `E014`, not to `E007`** -- but **only** when the value
  really is out of range. `from_chars` returns it for both ends -- a value too large to represent and
  a value that underflows to zero -- and section 2.3 rule 5 makes both `E014`; treating it as a
  lexical failure would report the wrong code for a well-formed number. **Subnormals are not an
  error.** Section 2.6 guarantees that every finite `binary64` value round-trips, subnormals
  included, and the emitter writes them (`5e-324` is emitted as `4.9406564584124654e-324`); an
  implementation that forwards to `strtod` may raise `ERANGE` for a subnormal *result*, and a reader
  that maps that straight to `E014` rejects a file this document guarantees. Test the returned value,
  not just the error code: report `E014` when the result is ±infinity, or when the field is
  lexically nonzero (a digit `1`-`9` before the exponent) and the result is exactly zero. Otherwise
  accept it.
- **Check `ptr == last`.** `from_chars` succeeds on a valid prefix, so `1.5x` parses as `1.5` with
  `ptr` left at the `x`. The field must be consumed in full or it is `E007`.

**Open the file in binary mode.** `std::ifstream(path, std::ios::binary)`, and read the bytes as
they are. This is not a portability nicety: on Windows the default is *text* mode, which strips `CR`
before your code ever sees it — so a CRLF-corrupted dataset reads as if it were clean, `E006` can
never fire, and if the same mistake is made on the Layer-2 file its digest still matches because the
bytes you hashed are not the bytes on disk. A reader that omits `std::ios::binary` silently accepts
exactly the corruption sections 2.1 and 3 exist to catch. The same applies to every other language:
read bytes, decode yourself, and never let a runtime's universal-newline translation sit between the
file and the check.

Two further requirements follow from section 2:

- The reader must reject a `#GRAMMAR` major it does not know rather than guess, and must do so
  before diagnosing anything later in the file (section 2.7).
- The reader must not accept a comma decimal separator under any locale. The grammar has exactly one
  numeric syntax and it is the C-locale one.

**Line length is not bounded by this format**, and deliberately carries no error code: a dataset is
a generated, audited artifact of at most a few hundred rows, not untrusted network input, so a
resource cap would buy nothing and add a seventeenth code. A reader that must run in a fixed memory
budget should impose its own limit and report it as its own error, not as one of these.

---

## 7. The reference implementation

`openmucf/g4/spec.py` is the reference implementation of section 2 and section 4:

- `parse(text: str) -> G4DatTable` -- parse a Layer-1 file, raising `G4DatFormatError(code, line, message)`.
- `render(table: G4DatTable) -> str` -- emit a Layer-1 file.
- `validate(table: G4DatTable) -> None` -- check an in-memory table against section 2.
- `format_float(x: float) -> str` -- the `%.17g` float syntax of section 2.6.

`openmucf/g4/provenance.py` is the reference implementation of section 3: `validate_document()` for
the schema, `document_bytes()` for the exact bytes the digest is taken over,
`check_canonical_bytes()` for the canonical-form rule, `check_source_digest()` for the cross-layer
digest (`E009`), and `check_against_table()` for the requirement that the file-level fields equal the
Layer-1 directives they mirror. Layer 2 has no line numbers, so its schema violations raise
`ValueError`; `E009` is the one code that spans both layers.

`openmucf/g4/emit.py` is the reference implementation of section 8: `build_tarball()`,
`gzip_header()`, `tarball_md5()` and `add_dataset_snippet()`.

Guarantees, each covered by a test:

1. **Round-trip.** For every table that `validate()` accepts, `parse(render(t)) == t`.
2. **Determinism.** `render(t)` is a pure function of `t`: two calls produce identical bytes, the
   output contains no timestamp, and the bytes do not change under a comma-decimal `LC_NUMERIC`.
3. **Only conforming output.** `render()` first sorts the records ascending by the primary key and then
   validates; it therefore accepts a table whose records are not yet sorted, and everything it emits
   is accepted by `parse()`. `validate()` itself reports an unsorted table as `E015`.
4. **Exact floats.** `float(format_float(x)) == x` for every finite double.

`validate()` reports the line number the offending item **would** occupy in the emitted file, so an
in-memory table and a parsed file report errors the same way. It scans in that same canonical order,
so its diagnosis is a function of the table's *content*: two callers who insert the same directives
in different orders get the identical error, exactly as they get identical bytes from `render()`.

A table that no file could ever produce -- a directive value carrying leading or trailing
whitespace, or an embedded newline -- is a programming error rather than a format error: `validate()`
raises `ValueError` for it, and the section-4 codes remain exactly the sixteen file-level conditions.
`format_float()` likewise raises `ValueError` on a non-finite input, since at that layer there is no
line number to report; `E014` is raised by `validate()` and `parse()`, which have one.

---

## 8. The archive

A dataset ships as **one gzipped tar archive** holding the Layer-1 `.g4dat` files and the Layer-2
`*.prov.json` files they were generated from -- the extension Geant4's dataset machinery expects
(`EXTENSION tar.gz`). The archive is **a pure function of its members**: nothing about the machine
that built it may appear in its bytes, or the artifact cannot be checksummed once and shipped, and a
reader cannot reproduce it to check our work.

Every field that would otherwise leak the builder is pinned:

| Layer | Field | Value |
|---|---|---|
| tar | format | `ustar` **explicitly** -- the default format has changed between writer versions |
| tar | member order | ascending by name |
| tar | `mtime` | `0` |
| tar | `uid`, `gid` | `0`, `0` |
| tar | `uname`, `gname` | empty, empty |
| tar | `mode` | `0644` |
| tar | typeflag | the byte `'0'` (`0x30`), not NUL — both spell "regular file" and readers accept either, but they are different bytes and change the header checksum |
| tar | member name | a **flat US-ASCII name** -- no path separator, no `./` prefix, no directory component -- at most **100 bytes** (a longer name forces a GNU/PAX extension header whose bytes are not writer-stable) |
| tar | magic + version | `ustar\0` then `00` (bytes 257-264 of each header block) |
| tar | numeric field encoding | zero-padded octal, NUL-terminated, filling the field: `mode`/`uid`/`gid` as 7 digits + NUL (`0000644`, `0000000`), `size`/`mtime` as 11 digits + NUL |
| tar | header checksum | **six octal digits, then NUL, then space** — not seven digits, and not digits + space + NUL. Computed per POSIX: the unsigned sum of all 512 header bytes **with the checksum field itself taken as eight spaces** |
| tar | `devmajor`, `devminor` | **16 NUL bytes** (offsets 329-344), *not* octal zeros |
| tar | end of archive | two 512-byte zero blocks, then zero padding to a multiple of **10240** bytes |
| gzip | `mtime` | `0` |
| gzip | `FNAME` | absent (flag bit `0x08` clear) |
| gzip | compression level | `9` with the **default strategy** and the default memory level, and therefore `XFL` = `2` (a different strategy can change `XFL`, and does change the stream) |
| gzip | `OS` byte | **255** (unknown) — *not* `3`, which is what a Unix `gzip(1)` writes |

The members sit at the **archive root** (a flat archive), and the checksum in the registration
snippet is the **MD5 of the archive's bytes** -- MD5 because that is what `geant4_add_dataset`'s
`MD5SUM` field is: a download-integrity check against corruption, not a security boundary.

**The encoding rows matter as much as the value rows**, and they are where a reimplementation goes
wrong: the numeric-field encoding, the checksum encoding, `devmajor`/`devminor`, the 10240-byte
padding, `XFL` and the `OS` byte each change the archive's MD5, and each has a plausible alternative
that a conforming writer picks by default. `bsdtar --format=ustar`, for instance, writes
`devmajor`/`devminor` as `000000 \0` and `mode` as `000666 \0`, both legal ustar and both a
different archive. These rows are listed for that reason, not because they matter to a consumer,
who only unpacks the archive.

**One honest limit.** The table above fixes the *container*; the DEFLATE stream inside it comes from
zlib, and two zlib builds -- or two compression settings that this document does not pin, such as
the memory level or the strategy -- are not guaranteed to emit byte-identical compressed output for
the same input. So the container fields are the normative, reproducible part, **rebuilding the exact
bytes additionally requires a compatible zlib**, and **the `MD5SUM` identifies one built artifact
rather than being a portable identity of the dataset**. The dataset's portable identity is
`#VERSION` plus `#SOURCEDIGEST` (section 2.1 rule 6), which is why those exist. In practice the
reference implementation's archive has been byte-identical on Windows/x86-64, Linux/x86-64 and
macOS/arm64; a consumer that needs to prove two archives carry the same data should compare the
members, not the compressed bytes.

## 9. Not included in this release

The **C++ reader and its standalone validation application** are specified here (sections 5 and 6)
but are not part of this release; they are stated now precisely so that the reader, when written,
cannot get them wrong by accident. Everything else this document specifies -- the grammar, the
Layer-2 schema, the error codes, the archive, and the generator that produces all of them -- ships.

### Attribution

> This product includes software developed by Members of the Geant4 Collaboration
> ( http://cern.ch/geant4 ).

The `parity` datasets described by this format reproduce values compiled into Geant4, and are
generated from Geant4 source vendored in `third_party/geant4/` under the Geant4 Software License
v1.0. Those terms apply to that directory; the rest of this repository is Apache-2.0 (code) and
CC-BY-4.0 (data).
