# Vendored Geant4 source

This directory holds **unmodified files** from the Geant4 toolkit, together with the Geant4
Software License they are distributed under.

> This product includes software developed by Members of the Geant4 Collaboration
> ( http://cern.ch/geant4 ).

## What is here, and why

| Path | What |
|---|---|
| `LICENSE` | Geant4 Software License v1.0, verbatim |
| `v11.4.2/G4MuonMinusBoundDecay.cc` | the upstream source file, **byte-for-byte unmodified** |
| `v11.4.2/G4MuonicAtomHelper.cc` | the upstream source file carrying the second compiled-in copy of the same capture and effective-charge tables, **byte-for-byte unmodified** |

`G4MuonMinusBoundDecay.cc` carries Geant4's compiled-in muon-capture data: a 90-record
`{Z, A, cRate, cRErr}` table, a 101-value effective-charge (`zeff`) table, and the
Goulard–Primakoff analytic fallback used for every `(Z, A)` the table does not list. The
`G4MuonicData` D1 dataset in `data/g4/d1/` is generated **from this file** — the record counts, the
values and the fallback coefficients are all parsed out of it at build time by
`openmucf/g4/sources/d1_nuclear_capture.py`, and nothing in that chain is transcribed by hand.

Vendoring it is what makes the parity claim checkable by someone who has neither a Geant4 checkout
nor a Geant4 build: `make g4data` regenerates the dataset from these bytes, `make audit` byte-diffs
the result, and the tests in `tests/test_g4parity.py` re-derive every count and every value from
here rather than from the generated file.

## The pins

| Fact | Value |
|---|---|
| upstream repository | https://github.com/Geant4/geant4 |
| upstream tag | `v11.4.2` |
| upstream commit | `8cc04f65977807f1848da7b958c421cd5e162f26` |
| upstream path | `source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc` |
| **git blob id** | `29bd73719cd619de34ef83ca5ca076ceadf1cc5a` |
| sha256 | `860dcdb53167c6437484b12c05ac1ab2eae4a6a52886af83fcf4394611882813` |
| size | 16312 bytes, 451 lines |
| `G4MuonicAtomHelper.cc` upstream path | `source/particles/management/src/G4MuonicAtomHelper.cc` |
| `G4MuonicAtomHelper.cc` **git blob id** | `98935195538c67c24ad1229c6064c8da05c9e7e2` |
| `G4MuonicAtomHelper.cc` sha256 | `038a13afafdb23a7a34648659a066359fbaec13c73d0ab31954512c3702ff463` |
| `G4MuonicAtomHelper.cc` size | 13669 bytes, 387 lines |

**The blob ids are the load-bearing pins.** Each is upstream's own object name for those exact
bytes, so a third party can verify each copy against the Geant4 repository without cloning Geant4
and without trusting us — and each is computable in three lines of `hashlib`, with no `git` binary:

```python
import hashlib, pathlib
data = pathlib.Path("third_party/geant4/v11.4.2/G4MuonMinusBoundDecay.cc").read_bytes()
print(hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest())
# 29bd73719cd619de34ef83ca5ca076ceadf1cc5a
```

A sha256 is recorded alongside each because SHA-1 is a **provenance pin** here, not a security
control, and saying so is cheaper than defending it later.

`.gitattributes` marks `third_party/geant4/** -text`. That line is load-bearing: each file's
identity *is* its bytes, so a checkout with `core.autocrlf` set would rewrite them and break the
pins. `tests/test_g4parity.py` asserts the vendored bytes contain no `\r`, so a deleted attribute
names its own cause instead of surfacing as an unexplained hash mismatch.

## Re-pinning a future release

Overwriting these files in place is **forbidden**: it would destroy the evidence that the previously
published dataset was faithful to the version it claimed. A new upstream revision gets a new
`third_party/geant4/<tag>/` directory, a new `#SOURCESHA` in the generated dataset, and a written
record of what moved.

## Licensing

The terms in `LICENSE` apply to **this directory only**. The rest of this repository is
Apache-2.0 (code) and CC-BY-4.0 (data) — see `../../LICENSE` and `../../LICENSE-DATA`.
