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

`stopped_muon.py` uses the standard library alone and loads `openmucf/g4/stopped_mu.py` and
`openmucf/g4/capture_cascade.py` by path, so it runs where the package is not installed. `run` and `reproduce` take the pre-registration as `--prereg <file>`, by default the
committed `data/g4/stopped_mu/preregistration.json`, and `reproduce` reads the committed cells from `--cells <file>`,
by default `data/g4/stopped_mu/cells.jsonl`.

```sh
python3 cpp/stopped_muon/stopped_muon.py build --install <prefix> --build <dir>
python3 cpp/stopped_muon/stopped_muon.py run --binary <exe> --install <prefix> --data <dir> --work <dir> \
    --version <version> --config <config> --target <target> --prereg <file>
python3 cpp/stopped_muon/stopped_muon.py reproduce --cell <version>/<config>/<target> --binary <exe> \
    --install <prefix> --data <dir> --work <dir> --out <file> --prereg <file> --cells <file>
python3 cpp/stopped_muon/stopped_muon.py cascade --binary <exe> --install <prefix> --data <dir> --work <dir> \
    --version <version> --config <config> --target <target> --out <file> --transitions <file> \
    --prereg <file> --rule <file>
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

`cascade` harvests the muonic cascade of a cell that the `harvest` block of `--rule` names, by default the committed
`data/g4/stopped_mu/cascade_rule.json`: a single run of the rule's events from the rule's seed base, with the
environment, the harness options and the dataset checks of `run`, for a single-isotope target. The records go to a
regular file; `openmucf/g4/capture_cascade.py` rebuilds the cascade of every stop from the muon's at-rest step,
matching the energy of each electron and photon to the level energies of the run's `config.txt` and admitting an
Auger electron only while the source's Auger counter allows it, and the records are then deleted. Every event must
be a single stop whose cascade is rebuilt, seeded as the rule says; otherwise the command refuses the cell, keeps the
records and exits nonzero. It appends the cell's row (events, stops, rebuilt cascades, Auger electrons, photons, the
binary's sha256 and the sha256 of the rebuilt cascades) to the file `--out` names and the number of stops taking
each transition to the file `--transitions` names.

## The committed outputs

`data/g4/stopped_mu/` holds what this application made and the comparison made from it:

- `preregistration.json`: the setup, the measured anchors, the predicates, the rows and the rule, fixed before the
  runs apart from the treatment of events without an outcome, which the file marks as fixed after part of the matrix
  had been run;
- `cells.jsonl`: the reduced record of every cell, with the sha256 of each chunk's compact text;
- `builds.csv`: the builds that made the cells and a rebuild of the committed sources;
- `reproduced.csv`: the chunks re-run with the committed application;
- `comparison.csv` and `verdicts.csv`: the cells set beside the measured yields and the verdict of each row under the
  pre-registration's rule, which `scripts/generate_stopped_mu.py compare` writes as a step of `make audit`;
- `cascade_reference.csv` and `cascade_rule.json`: the measured K-series intensities and the rule that compares the
  stock cascade with them and tests the harvested cascades, with when each part was fixed;
- `cascade_harvest.csv` and `cascade_transitions.csv`: the cascades harvested with `cascade`;
- `cascade_comparison.csv` and `cascade_consumer.csv`: the comparison with the measured intensities and the
  decision on every count of the harvested cascades, which the same command writes.

The same command generates `BENCHMARK_STOPPED_MU.md` from these files and the natural compositions of
`data/g4/d1/geant4_natural_isotopes.csv`: the setup, the rule and when each part of it was fixed, the verdicts with
the comparison behind them, and the reproduction.

The compact files themselves are not committed; a re-run is compared with the committed record through the sha256 of
each chunk's compact text.

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
