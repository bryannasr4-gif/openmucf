# The measured-vector material selector

`G4MuonicMaterialSelector` lets a Geant4 application replace the capturing-element weights of the stock
`G4ElementSelector`, for materials it binds explicitly, by a measured probability vector from
`data/g4/d2/measured/`. `openmucf/g4/d2_material.py` is the reference implementation of the binding rules and of
the selection arithmetic; the header states them for the C++ side.

## Using it

```cpp
#include "G4MuonicMaterialSelector.hh"

auto catalogue = G4MuonicMaterialCatalogue::Load("<path>/data/g4/d2/measured");  // the only file access
auto* bindings = new G4MuonicMaterialBindings(&catalogue);
bindings->Bind(aluminiumOxide, "stanislaus1987-al2o3", "Al2O3", /*acknowledgeProvisional=*/true);
bindings->Freeze();
physicsList->RegisterPhysics(new G4MuonicMaterialSelectorPhysics(bindings, /*enabled=*/true));  // after stopping
```

The catalogue and the bindings must outlive every run. Registered after the stopping physics, the constructor
installs a new selector into the negative muon's `muMinusCaptureAtRest` and into any `G4MuonMinusAtomicCapture`
of every thread that builds processes; each process owns the selector it receives. Constructed with
`enabled = false`, or for a material that is not bound, the installed selector calls the stock selector once,
unchanged. Registered before the stopping physics, the constructor finds nothing to install into and raises
`G4MuonicData103`. A refused binding raises `G4MuonicData101` with `ENTRY_REFUSED` and binds nothing; the stock
selector is not substituted for a refused binding.

`G4MuonMinusAtomicCapture` declares `SetElementSelector` inline, and neither Geant4 11.4.2 nor 11.5.0.beta
defines it. `CMakeLists.txt` links a call to it first; only when that fails and the installed header's SHA-256
is one `supported_headers.txt` lists does it compile the definition the header gives `G4HadronStoppingProcess`'s
own setter. Stock `G4MuonMinusAtomicCapture` does not delete its selector in its destructor; that holds for the
installed selector as it does for the stock one.

## Building and running the checks

```sh
cmake -S cpp/material_selector -B build -DGeant4_DIR=<install>/lib/cmake/Geant4
cmake --build build
cpp/material_selector/native.sh build <tag> <install> <work>
cpp/material_selector/native.sh run <tag> <work> <catalogue> <physics data> <batch>
```

`harvest_material.cc` runs a negative muon at rest in a box of one material, on the stopping route (QBBC) or the
atomic route (`G4MuonMinusAtomicCapture` with the muonic-atom decay), under the Serial or the Tasking run
manager. Its modes are listed at the top of the file. The batches of `native.sh` run: every table read back and
bound in both element orders, with its running sums and the counts of the 2^20-draw midpoint ladder; the named
NIST oxides; the refusal cases; the installation order; the aluminium-oxide ladder on both routes and both run
managers with the provider absent, disabled, disabled with bindings in place, enabled with nothing bound and
enabled, and in reversed element order; the lead-oxide ladder for the NIST material and for the reversed order;
the default-off traces, with and without the muonic-data switch of an installation built with `cpp/patches/`;
a disabled provider given a missing or malformed catalogue path; and repeated runs with teardown. The mutation
drills are separate builds with one `G4MUONIC_DRILL_*` macro each; no build that delivers physics defines one.

`evidence/` holds the outputs of those batches in builds of Geant4 11.4.2 and 11.5.0.beta, with the identities
of each build, and `tests/test_g4_material_selector.py` checks them against `openmucf/g4/d2_material.py`.
