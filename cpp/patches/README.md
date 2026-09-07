# Geant4 patches — reading the dataset from inside Geant4

`g4-v11.4.2-muonicdata.patch` lets Geant4 read the `G4MuonicData` dataset: it adds the repository's
reader (`G4MuonicDataTable`, byte for byte the files under `cpp/include` and `cpp/src`) and one glue
file (`G4MuonicDataOverlay`) to the `G4globman` module, lists them in that module's `sources.cmake`,
and inserts a table lookup into the four functions, in two files, that carry the compiled-in
muon-capture tables: `G4MuonMinusBoundDecay::GetMuonCaptureRate` and `::GetMuonZeff`, and
`G4MuonicAtomHelper::GetMuonCaptureRate` and `::GetMuonZeff`. It applies with `git apply` (the patch
carries the usual `a/` and `b/` prefixes) to Geant4 `v11.4.2` at the revision the dataset's
`#SOURCESHA` directive names, and it changes no existing line of either seam file — every hunk there
only inserts.

The lookup is off by default: nothing in the patch calls `G4MuonicDataTable::Enable()`, and until an
application does so before its first capture-rate call, the four functions run exactly their
unpatched code after one boolean test — no lookup, no file access, no message. With the opt-in on,
each function first applies its own clamp, as before, then consults the table; a key the table lacks
falls through to that function's compiled-in code, so the fallback formula is reproduced as it is,
including the negative rates the dataset's documentation registers.

Discovery follows the `G4FindDataDir` lookup Geant4 provides: an exported `G4MUONICDATA`, or, once the
dataset is registered, the entry under `GEANT4_DATA_DIR`. If neither resolves a directory, the first lookup
raises a fatal `G4Exception` naming both `G4MUONICDATA` and `GEANT4_DATA_DIR`; if a directory is
found but a file in it fails validation, the exception carries the reader's error code and line.
`g4-v11.4.2-register-dataset.patch` is separate and serves the registered mode only: it appends the
dataset's `geant4_add_dataset` entry to `G4DatasetDefinitions.cmake`, so a build carrying it resolves
the dataset under `GEANT4_DATA_DIR` with no variable exported.

The evidence that a patched build behaves as stated — with the opt-in off, application runs and a
harvest whose whole output is bit-identical to an unpatched build's; with the opt-in on, the
harvested sweep reproducing the digest recorded by the oracle file beside the dataset
(`cpp/tools/README.md` describes it), through both compiled-in copies and in both discovery modes —
is kept outside this repository. That digest is stated here by reference to the oracle file, never
as a literal, and a test holds this file to that.
