# Geant4 patches — reading the dataset from inside Geant4

`g4-v11.4.2-muonicdata.patch` and `g4-v11.5.0.beta-muonicdata.patch` let Geant4 read the
`G4MuonicData` dataset, each cut against the pristine tree of the revision its name carries: the
patch adds the repository's reader (`G4MuonicDataTable`) and its glue (`G4MuonicDataOverlay`),
byte for byte the files under `cpp/include` and `cpp/src`, to the `G4partman` module,
lists them in that module's `sources.cmake`, adds one boolean to `G4HadronicParameters`, and
inserts a table lookup into the four functions, in two files, that carry the compiled-in
muon-capture tables: `G4MuonMinusBoundDecay::GetMuonCaptureRate` and `::GetMuonZeff`, and
`G4MuonicAtomHelper::GetMuonCaptureRate` and `::GetMuonZeff`, and into
`G4EmCaptureCascade::ApplyYourself` and `G4MuonicAtomHelper::GetKShellEnergy`, and adds a form of
`GetKShellEnergy` that also takes the mass number. It applies with `git apply` (the patch
carries the usual `a/` and `b/` prefixes) to the tag its name carries — the dataset's `#SOURCESHA`
names the `v11.4.2` revision, and a test over the vendored copies under `third_party/` proves that
the tables compiled into `v11.5.0.beta` are the same — and the only existing lines it changes are
the calls to `GetKShellEnergy` in `G4MuonicAtomHelper::ConstructMuonicAtom` and
`G4MuonicAtomDecay::DecayIt`, which now pass the mass number: every other hunk in an existing file
only inserts.

The lookup is off by default: the boolean the patch adds to `G4HadronicParameters` starts `false`,
its setter is the only caller of `G4MuonicDataTable::Enable()`, and until an application sets it —
one line before its physics list is built: `SetEnableMuonicData(true)` on the `G4HadronicParameters`
singleton — every function it inserts a lookup into runs exactly its unpatched code after one
boolean test — no lookup, no file access, no message. With the opt-in on, each function consults the
table; a key the table lacks falls through to that function's compiled-in code, so the fallback
formula is reproduced as it is, including the negative rates the dataset's documentation registers.

Discovery follows the `G4FindDataDir` lookup Geant4 provides: an exported `G4MUONICDATA`, or, once
the dataset is registered, the entry under `GEANT4_DATA_DIR` or, when that variable is unset or
names no directory, under the default system paths `G4FindDataDir` searches next. If none resolves a
directory, the first lookup raises a fatal `G4Exception` naming both `G4MUONICDATA` and
`GEANT4_DATA_DIR`; if a directory is found but a file in it fails validation, the exception carries
the reader's error code and line. A patched build reads each seam through a profile of its own:
`G4MUONICDATA_D1_PROFILE` for nuclear capture and `G4MUONICDATA_D3_PROFILE` for the muonic
transitions, `G4MUONICDATA_PROFILE` for both where neither is set, and, where none is set,
`parity` for capture and the compiled-in code for the transitions; a variable set to nothing is
unset, and `compiled` is a reserved token naming a seam with no tables, whose selection reads no
directory at all. The environment is read where the opt-in is set and not again, so a variable
exported after that selects nothing. A variable naming a profile that carries no table of its own
seam raises a fatal `G4Exception` naming that variable, and so does a token no file in the
directory carries.
A table the named profile carries no file for is treated like a key it lacks: the function's
compiled-in code runs. So is a nuclide the selected profile has no row for: under every profile
but `parity` a lookup reads the row keyed by exactly that nuclide, and an element's
natural-composition row answers only a request that asks for it by name.
Before any table is selected, the glue checks what every loaded table means — its identity and
version, its shape, its keys and declared validity, the units of the columns a lookup reads, the
domain of its values, and, for the two energy tables of one profile, that they cover one set of
nuclides in one falling order — and refuses the whole dataset with a fatal `G4Exception` naming
the rule, the file and the table where any of that does not hold.
`g4-v11.4.2-register-dataset.patch` and
`g4-v11.5.0.beta-register-dataset.patch` are separate and serve the registered mode only: each
appends the dataset's `geant4_add_dataset` entry to `G4DatasetDefinitions.cmake`, so a build
carrying it resolves the dataset under `GEANT4_DATA_DIR` with no variable exported. That mode looks
for the dataset under a `<NAME><VERSION>` directory, which is the directory the archive unpacks to.

The evidence that a patched build behaves as stated — with the opt-in off, application runs and a
harvest whose whole output is bit-identical to an unpatched build's; with the opt-in on, the
harvested sweep reproducing the digest recorded by the oracle file beside the dataset
(`cpp/tools/README.md` describes it), through both compiled-in copies and in both discovery modes —
is kept outside this repository. That digest is stated here by reference to the oracle file, never
as a literal, and a test holds this file to that.
A patched build was also measured with the second profile the dataset ships named: its harvest
differed from the `parity` harvest on exactly the keys that profile resolves to a different value.
The cascade and K-energy harvests of `v11.4.2` and `v11.5.0.beta` were measured the same way: with
the opt-in off, or under a profile carrying no D3 table, they are bit-identical to an unpatched
build's.
