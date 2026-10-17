# The stopped-muon application

This directory holds the application whose records the stopped-muon benchmark scores, and the driver that builds it,
runs it and reduces its records.

`g4muonic_transport.cc` and `CMakeLists.txt` are the transport harness of `cpp/transport/` with options added for a
target of natural composition, for the momenta of the secondaries and for the dataset directories Geant4 reads. The
executable keeps the harness's name, `g4muonic_transport`. The added options are:

- `--element <symbol>` builds the target from the element `G4NistManager::FindOrBuildElement` returns for the symbol,
  with its natural isotopic composition, in the same box and at the same density as a single-isotope target; it refuses
  `--Z` and `--A` beside it.
- `--momenta` appends the momentum of each secondary, in MeV/c, to its record line.
- `--print-datasets` prints, for each dataset variable the harness lists, the directory `G4FindDataDir` returns, and
  exits before the run.
- `--record-datasets` appends the same lines to `config.txt` after the run.

## How each configuration is reached

- `bertini` runs the `bound_decay` route: the physics an unmodified QBBC application receives, whose stopping
  constructor `G4StoppingPhysics` registers the capture at rest of a negative muon, handed by `G4MuonMinusCapture` to
  the Bertini cascade.
- `bertini_pc` runs the same route with the environment variable `G4CASCADE_USE_PRECOMPOUND` set, which switches on the
  precompound de-excitation of the Bertini cascade.
- `precompound` runs the `muonic_atom_helper` route, which an application receives only by registering constructors:
  the harness replaces QBBC's stopping constructor with a `G4StoppingPhysics` whose last argument switches the capture
  at rest of a negative muon off in that constructor, registers a constructor that adds `G4MuonMinusAtomicCapture` as
  the rest process of the negative muon, and registers `G4MuonicAtomDecayPhysics`. Its rows describe what an
  application receives after those registrations, not what QBBC gives by default.

This route is not the dataset opt-in. The harness option `--opt-in` switches `G4HadronicParameters::SetEnableMuonicData`,
which exists only in a build that carries the patches of `cpp/patches/`; an unpatched build refuses `--opt-in on`, and
the driver passes `--opt-in off` on every route.

The note of `cpp/transport/README.md` on the muonic atom the harness creates before the run, and the limits it states,
apply here unchanged; for a natural target the harness creates a muonic atom for each isotope of the element.

## Building, running and re-running a cell

`stopped_muon.py` uses the standard library alone and loads `openmucf/g4/stopped_mu.py` by path, so it runs where the
package is not installed. `run` and `reproduce` take the pre-registration as `--prereg <file>`.

```sh
python3 cpp/stopped_muon/stopped_muon.py build --install <prefix> --build <dir>
python3 cpp/stopped_muon/stopped_muon.py run --binary <exe> --install <prefix> --data <dir> --work <dir> \
    --version <version> --config <config> --target <target> --prereg <file>
python3 cpp/stopped_muon/stopped_muon.py reproduce --cell <version>/<config>/<target> --binary <exe> \
    --install <prefix> --data <dir> --work <dir> --out <file> --prereg <file> --cells <file>
```

`build` configures the application with CMake in Release mode against an unpatched install
(`-DG4MUONIC_TRANSPORT_PATCHED=OFF`), builds it, and prints the sha256 of the sources and of the executable, the
version `geant4-config` prints, the compiler and CMake versions, and the compiler flags and multithreading setting the
install's `Geant4Config.cmake` records.

`run` runs a cell: a pilot chunk, then a chunk sized from the pilot's share of captures to reach the
pre-registration's number of captures with a margin, then at most a single extension chunk if the cell is still short. A
chunk starting at global event `offset` is run with `--seed-base` equal to the pre-registration's seed base plus
`offset`, and the harness seeds each event with its `--seed-base` plus the event's number in the run, so every event
of a cell is seeded with the seed base plus its global number, however the cell is cut into chunks. The pre-registration's number of captures is the only sample size: the driver has no option
that changes it. Each chunk's records go to a regular file, are classified into its compact file and are then deleted;
`run` reduces the chunks into the cell record `cell.json` under `--work`.

`reproduce` re-runs every chunk of a committed cell with the committed offsets and event counts, compares the sha256 of
each re-run compact file with the committed sha256, reduces the re-run chunks and compares the cell's quantities, neutron
multiplicities, f2, counts and events without an outcome with the committed record. It prints `REPRODUCED` when every
comparison is equal and `DIFFERS` with what differs otherwise, exits nonzero on any difference, and appends a line per
re-run chunk to the file `--out` names.

## The environment

Every Geant4 call runs without `GEANT4_DATA_DIR`, without any variable matching `^G4.*DATA$` and without any
`G4CASCADE_*` or `G4NUCMODEL_*` variable the caller set. The driver then binds each dataset variable to
`<data>/<directory>` with the directory the pre-registration names for the revision, sets `G4CASCADE_USE_PRECOMPOUND`
for `bertini_pc` only and points `LD_LIBRARY_PATH` at `<install>/lib`. Before the run the directories
`--print-datasets` reports must equal that binding, and after the run the `DATASET` lines of `config.txt` must equal it
again; a difference refuses the chunk.

## The records and the compact format

The harness writes, for each event, a `T` line for each track (its parent, PDG code, creator process and model, kinetic
energy and global time), an `S` line for each step of the muon or of a muonic atom (its process and the number of
secondaries it created), followed by a `C` line for each of those secondaries (PDG code, creator model, kinetic energy,
global time and, with `--momenta`, momentum), and an `E` line that closes the event. Energies, times and momenta are
written as exact C99 hexadecimal floating-point values.

The classifier in `openmucf/g4/stopped_mu.py` turns the records of a chunk into a compact line per event, numbered
from the chunk's offset: `<event> C <capture time> <products>` for a nuclear capture, with each capture-step product's
PDG code, kinetic energy and momentum and each photon a capture-step nucleus emits within the pre-registration's window
after the capture; `<event> D` for a decay in orbit; and `<event> X <reason>` for an event that is not a single capture
or a single decay. The module's docstring gives the grammar of both formats.
