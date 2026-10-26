# Geant4 patches — reading the dataset from inside Geant4

`g4-v11.4.2-muonicdata.patch` and `g4-v11.5.0.beta-muonicdata.patch` let Geant4 read the
`G4MuonicData` dataset, each cut against the pristine tree of the revision its name carries: the
patch adds the repository's reader (`G4MuonicDataTable`), its semantic layer
(`G4MuonicDataSemantics`), its glue (`G4MuonicDataOverlay`) and the bound-rate provider
(`G4MuonicBoundRateProvider`), byte for byte the files under `cpp/include` and `cpp/src`, to the
`G4partman` module, lists them in that module's `sources.cmake`, adds a boolean to
`G4HadronicParameters` for each of its opt-ins, and
inserts a table lookup into the four functions, in two files, that carry the compiled-in
muon-capture tables: `G4MuonMinusBoundDecay::GetMuonCaptureRate` and `::GetMuonZeff`, and
`G4MuonicAtomHelper::GetMuonCaptureRate` and `::GetMuonZeff`, and into
`G4EmCaptureCascade::ApplyYourself` and `G4MuonicAtomHelper::GetKShellEnergy`, and adds a form of
`GetKShellEnergy` that also takes the mass number. It applies with `git apply` (the patch
carries the usual `a/` and `b/` prefixes) to the tag its name carries — the dataset's `#SOURCESHA`
names the `v11.4.2` revision, and a test over the vendored copies under `third_party/` proves that
the tables compiled into `v11.5.0.beta` are the same — and the only existing lines it changes are
the calls to `GetKShellEnergy` in `G4MuonicAtomHelper::ConstructMuonicAtom` and
`G4MuonicAtomDecay::DecayIt`, which now pass the mass number, and the first line of the compiled-in
`GetKShellEnergy(G4double Z)` definition, which renames it to the private
`GetCompiledKShellEnergy` that both public forms fall to: every other hunk in an existing file
only inserts. The bound-rate opt-in described below inserts a block into
`G4MuonMinusBoundDecay::ApplyYourself` and a block into `G4MuonicAtomHelper::ConstructMuonicAtom`.

The lookup is off by default: the dataset boolean the patch adds to `G4HadronicParameters` starts `false`,
its setter is the only caller of `G4MuonicDataTable::Enable()`, and until an application sets it —
one line before its physics list is built: `SetEnableMuonicData(true)` on the `G4HadronicParameters`
singleton — every function it inserts a lookup into runs its unpatched code and one boolean test,
and both capture-rate functions then pass the rate they computed to
`G4MuonicDataOverlay::CheckComputedCaptureRate`, which makes the same test and returns — no
lookup, no file access, no message. With the opt-in on, each function looks its key up in the
table its seam selects, if it selects any, and a key it finds no row for falls through to that
function's compiled-in code. Under `parity`, and where the nuclear-capture seam has no table —
under `compiled`, or under a profile that carries none of its tables — the fallback formula is
reproduced as it is, including the negative rates the dataset's documentation registers; under an
evaluated capture profile, a profile other than `parity` that gives the seam a table,
`CheckComputedCaptureRate` raises the fatal `G4Exception` `G4MuonicData005` when the rate a
capture-rate function computed is not finite, and leaves a finite rate that is not positive, such as
the fallback formula's rate for tritium, as computed, raising the warning `G4Exception`
`G4MuonicData006` the first time in a process that it meets such a rate.

The bound-rate opt-in is separate from the dataset opt-in, and off by default. With
`SetEnableMuonicBoundRates(true)` on the `G4HadronicParameters` singleton, set before the physics list
is built, the stopping route (`G4MuonMinusBoundDecay::ApplyYourself`, which draws the delay and the
branch of a stopped negative muon) and the atomic route (`G4MuonicAtomHelper::ConstructMuonicAtom`,
which stores the muonic atom's lifetime, DIO lifetime and nuclear-capture lifetime) take one capture
and decay rate pair from `G4MuonicBoundRateProvider`: the capture rate
`G4MuonicAtomHelper::GetMuonCaptureRate` returns, called as it stands, with whatever the dataset
opt-in gives it, and the current leading inclusive tree decay rate
`free * [1 - (Z alpha)^2 * (1/2 - eta)]`, with `eta = muonMass/nuclearMass`,
evaluated using each consumer's nuclear mass. Its stable identifier is
`low-z-inclusive-tree-linear-recoil-v1`; the [derivation](../../docs/low-z-bound-muon-decay.md)
states its point-scalar, unpolarized 1s, fully inclusive scope. This is a linear-eta,
leading-binding prescription with unassigned higher-order and nuclear/radiative
remainders. The consistency guard `eta < 1/2` supplies no accuracy bound; the
physical expansion needs `eta << 1`. The provider
answers for `2 <= Z <= 13` and `A >= Z`; for hydrogen, for higher `Z` and for a natural-composition
request it answers unsupported and each consumer keeps the pair it formed itself, and inside that
domain nonpositive/nonfinite masses, nonfinite eta or eta >= 1/2, a negative/nonfinite
capture rate, or decay outside `0 < decay <= free` raises the fatal `G4Exception`
`G4MuonicBoundRates001` rather than being clipped. The boolean starts `false`, its setter is the
only caller of `G4MuonicBoundRateProvider::Enable()`, and `SetEnableMuonicData` does not set it;
with it off, each consumer runs its unpatched code and a test of the switch. The first time a
consumer forms a pair it freezes the switch, and a later change raises the fatal `G4Exception`
`G4MuonicBoundRates002`, because muonic atoms already built keep the lifetimes they were built with.
The getters `G4MuonMinusBoundDecay::GetMuonDecayRate` and `G4MuonicAtomHelper::GetMuonDecayRate`
are unchanged and keep reporting each route's own prescription, not the pair a consumer receives with
the opt-in on. The former oracle equality and `bound_rates_*` harvests describe the historical
legacy-getter model; the current deterministic checks are separate. The pair supplies one prescription
on both routes; it is not a statement of physical
accuracy, and it leaves the timing of the products both routes emit as it was.

A [separate clock correction](../clock/README.md) is available for explicitly patched
builds. It changes the applied build's default product times independently of these
dataset and bound-rate options; installing or enabling those options does not apply it.

Discovery follows the `G4FindDataDir` lookup Geant4 provides: an exported `G4MUONICDATA`, or, once
the dataset is registered, the entry under `GEANT4_DATA_DIR` or, when that variable is unset or
names no directory, under the default system paths `G4FindDataDir` searches next. Setting the
opt-in calls `G4MuonicDataOverlay::Initialize()`, which discovers and reads the dataset once,
unless both seams select the reserved token `compiled` described below: if none resolves a
directory, it raises the fatal `G4Exception` `G4MuonicData001` naming both `G4MUONICDATA` and
`GEANT4_DATA_DIR`, and if the reader rejects the directory, it raises `G4MuonicData002`, whose
message carries the reader's error code and line when a file breaks the grammar. A patched build reads each seam through a profile of its own:
`G4MUONICDATA_D1_PROFILE` for nuclear capture and `G4MUONICDATA_D3_PROFILE` for the muonic
transitions, `G4MUONICDATA_PROFILE` for both where neither is set, and, where none is set,
`parity` for capture and the compiled-in code for the transitions; a variable set to nothing is
unset, and `compiled` is a reserved token naming a seam with no tables; a configuration in which both
seams select it reads no directory at all. The environment is read where the opt-in is set and not again, so a variable
exported after that selects nothing. `G4MUONICDATA_D1_PROFILE` or `G4MUONICDATA_D3_PROFILE` naming a profile that carries no table of its own seam raises the fatal `G4Exception` `G4MuonicData004` naming that variable, and so does a variable a seam reads through naming a token other than `compiled` that no file in the directory carries.
When the nuclear-capture seam reads `parity`, a dataset whose `parity` profile carries a
capture-rate table but no effective-charge table raises the fatal `G4Exception` `G4MuonicData003`.
Under every profile but `parity` a lookup reads the row keyed by exactly the nuclide asked for, and
an element's natural-composition row answers only a request that asks for it by name.
Before any table is selected, the glue checks what every loaded table means — its identity and
version, its shape, its keys and declared validity, the units of the columns a lookup reads, the
domain of its values, and, for the two energy tables of one profile, that they cover one set of
nuclides in one falling order — and refuses the whole dataset with the fatal `G4Exception`
`G4MuonicData005`, naming the rule and either the file and table or, for the pair of energy
tables, the profile where any of that does not hold; only a dataset these checks accept reaches
the `G4MuonicData004` and `G4MuonicData003` refusals above.
With the opt-in on, once a lookup in the transition seam's tables has given
`G4EmCaptureCascade::ApplyYourself` the K energy or a shell energy of the nuclide being transported, the
glue's `CheckCascadeLevels` checks the whole array of level energies the cascade then holds, the tables'
values and the hydrogen-like ones above them, and raises the fatal `G4Exception` `G4MuonicData005` when a
level is not a positive finite energy or the levels do not fall from each shell to the next. A dataset
the checks above accept can reach this refusal during a run: the transport workflow's refusal cases
reach it on both revisions (`cpp/transport/README.md`).
`g4-v11.4.2-register-dataset.patch` and
`g4-v11.5.0.beta-register-dataset.patch` are separate and serve the registered mode only: each
appends the dataset's `geant4_add_dataset` entry to `G4DatasetDefinitions.cmake`, so a build
carrying it resolves the dataset under `GEANT4_DATA_DIR` with no variable exported. That mode looks
for the dataset under a `<NAME><VERSION>` directory, which is the directory the archive unpacks to.

The evidence that a patched build behaves as stated — with the opt-in off, application runs and a
harvest whose whole output is bit-identical to an unpatched build's; with the opt-in on, the
harvested sweep reproducing the digest recorded by the oracle file beside the dataset
(`cpp/tools/README.md` describes it), through both compiled-in copies and in both discovery modes —
is kept outside this repository, except the transport runs `cpp/transport/README.md` describes, whose manifest and check results `cpp/transport/evidence/` holds. Because the `parity` tables hold the values compiled into Geant4, that sweep shows that enabling the dataset with them changes no rate it covers, and it cannot tell a rate the table answered from a rate the compiled-in code computed. That digest is stated here by reference to the oracle file, never
as a literal, and a test holds this file to that.
The cascade and K-energy harvests of `v11.4.2` and `v11.5.0.beta` were measured the same way: with
the opt-in off they are bit-identical to an unpatched build's.
The patched trees were built and tested on Linux only: continuous integration builds the reader and its
validator, not a patched Geant4, on Linux, macOS and Windows (`cpp/README.md`).

The patch files in this directory, the Geant4 source lines they carry as context and the lines
they add alike, are offered under the Geant4 Software License, so that they can be taken into
Geant4 under the licence Geant4 itself carries. `LICENSE` in this directory is a verbatim copy of
that licence, and a test holds it byte for byte to the copy vendored under `third_party/`. The
files the patches add keep, under `cpp/include` and `cpp/src`, the licence the rest of this
repository's code carries.

> This product includes software developed by Members of the Geant4 Collaboration
> ( http://cern.ch/geant4 ).
