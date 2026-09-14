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
| `v11.5.0.beta/G4MuonMinusBoundDecay.cc` | the same file at tag `v11.5.0.beta`, **byte-for-byte unmodified**; its tables are proved equal to the `v11.4.2` copy's by `tests/test_g4parity.py` |
| `v11.5.0.beta/G4MuonicAtomHelper.cc` | the same file at tag `v11.5.0.beta`, **byte-for-byte unmodified**; its tables are proved equal to the `v11.4.2` copy's by `tests/test_g4parity.py` |

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

## The pins — v11.5.0.beta

| Fact | Value |
|---|---|
| `v11.5.0.beta` commit | `f3d5293d384757b8a228a099898b2b87cfa4023c` |
| `v11.5.0.beta/G4MuonMinusBoundDecay.cc` **git blob id** | `ff95c000f2cc3f6cfd6b835bade304e05af9feb5` |
| `v11.5.0.beta/G4MuonMinusBoundDecay.cc` sha256 | `bb925829e0acaa7fa3efd4560d954dd2f58f344fd155cdb3ce2554bd77539288` |
| `v11.5.0.beta/G4MuonMinusBoundDecay.cc` size | 15642 bytes, 396 lines |
| `v11.5.0.beta/G4MuonicAtomHelper.cc` **git blob id** | `8c2c37a99cdb3effd3ce7f1898488ad75f82b3fd` |
| `v11.5.0.beta/G4MuonicAtomHelper.cc` sha256 | `d020924b759ad1149cf74e2955eeffede84bc55c8be4ffb364385cc803720e2a` |
| `v11.5.0.beta/G4MuonicAtomHelper.cc` size | 13695 bytes, 397 lines |

`.gitattributes` marks `third_party/geant4/** -text`. That line is load-bearing: each file's
identity *is* its bytes, so a checkout with `core.autocrlf` set would rewrite them and break the
pins. `tests/test_g4parity.py` asserts the vendored bytes contain no `\r`, so a deleted attribute
names its own cause instead of surfacing as an unexplained hash mismatch.

## Re-pinning a future release

Overwriting these files in place is **forbidden**: it would destroy the evidence that the previously
published dataset was faithful to the version it claimed. A new upstream revision gets a new
`third_party/geant4/<tag>/` directory, a new `#SOURCESHA` in the generated dataset, and a written
record of what moved. `v11.5.0.beta/` is such a directory whose tables a test proves equal to
`v11.4.2/`'s, so the dataset's `#SOURCESHA` stays at the revision it was generated from.

## Licensing

The terms in `LICENSE` apply to **this directory only**. The rest of this repository is
Apache-2.0 (code) and CC-BY-4.0 (data) — see `../../LICENSE` and `../../LICENSE-DATA`.
