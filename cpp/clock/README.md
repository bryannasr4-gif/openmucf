# Stopped negative-muon clock correction

`../patches/g4-v11.4.2-stopped-muon-clock-v3.patch` and
`../patches/g4-v11.5.0.beta-stopped-muon-clock-v3.patch` are separate patches against
the named pristine Geant4 tags. Applying a patch explicitly changes that build's
default timing. There is no runtime switch. Neither the dataset patch nor
`SetEnableMuonicBoundRates` applies or enables this correction. The clock and
dataset/rate patches apply in either order. Apply exactly one complete clock patch,
matching the source revision.

The correction changes four consumers:

- `G4MuonMinusAtomicCapture::AtRestDoIt` gives the new muonic atom local time zero.
  Its common secondary loop adds the stopped muon's global time once. Previously,
  the atom already carried that global time and the loop added it again.
- `G4HadronStoppingProcess::AtRestDoIt` maps an unset negative nuclear-model
  secondary time to zero **before** adding the sampled capture waiting time.
  Previously, the default `G4HadSecondary` sentinel of minus one nanosecond was
  added to that waiting time and the composed result was clamped afterward.
  A legitimate positive model delay is preserved.
- `G4MuonicAtomDecay::DecayIt` applies the same ordering to nuclear-model times
  before `FillResult` adds the atom origin. The constructor accepts supplied nuclear
  interactions, including models with negative unset local times. Clamp that local
  time before adding the sampled atom waiting time. The verbose per-secondary
  diagnostic reports the resulting composed local time.
- `G4MuMinusCapturePrecompound::ApplyYourself` consumes
  `max(reaction_product.GetFormationTime(), 0)` before adding its projectile origin.
  It previously read `GetTOF`, whose particle-taking constructors initialize its
  legacy field by PDG sign. The default de-excitation return path writes fragment
  creation times to `FormationTime`; normal `G4PreCompoundModel` uses that field too.
  A valid zero formation time stays zero, with no TOF fallback. The light-target
  bypasses retain their existing origin handling.

Under the existing instantaneous atomic-formation approximation, the atom is
created at `t_stop`. Bound-decay products arrive at `t_stop + waiting_time`;
nuclear-capture products arrive at
`t_stop + waiting_time + max(model_secondary_local_time, 0)`.
The default atomic route's precompound wrapper starts its initial fragment at
creation time zero and receives projectile time zero from the atom-decay consumer.
Formation times are therefore relative to capture; the enclosing consumer adds
the sampled waiting time and atom origin. Positive modeled formation times are
retained. The previous uniform one-nanosecond local values came from the legacy
TOF field rather than formation metadata. Propagating the model's formation times
does not establish their experimental accuracy. Generic `G4PreCompoundModel`
origin handling is unchanged. Supplied dependencies must implement the
FormationTime contract; arbitrary TOF-only custom dependencies are outside this
correction's supported metadata contract.
Positive formation-cascade secondary times are retained too;
this is a check of secondary-time handling, not a model of delayed atom formation
for arbitrary custom cascades.

The stopping-process class is shared by other negatively charged hadrons.
Applying its hunk changes the same negative-sentinel composition for those users
as well. The finite consumer checks here cover stopped negative muons in Pb-208,
not every particle or hadronic model that can use that base class.

## Apply and build

Use a fresh checkout of the appropriate Geant4 tag and a separate build/install
prefix. From that source tree, with `PATCH` naming the matching absolute patch path:

```sh
git apply --check "$PATCH"
git apply "$PATCH"
cmake -S . -B /absolute/path/to/clock-build \
  -DCMAKE_INSTALL_PREFIX=/absolute/path/to/clock-install \
  -DGEANT4_INSTALL_DATA=OFF
cmake --build /absolute/path/to/clock-build --parallel 2
cmake --install /absolute/path/to/clock-build
```

Configure data discovery for the existing Geant4 datasets as usual. The patch
adds no dataset. An installation built without the clock patch keeps its previous
timing. The patch files use the Geant4 Software License in `../patches/LICENSE`.

## Focused regression probes

```sh
cmake -S cpp/clock -B /absolute/path/to/probe-build \
  -DGeant4_DIR=/absolute/path/to/clock-install/lib/cmake/Geant4
cmake --build /absolute/path/to/probe-build --parallel 2
/absolute/path/to/probe-build/clock_probe fixture 250 160
/absolute/path/to/probe-build/clock_probe atomic 250 300
/absolute/path/to/probe-build/clock_probe stopping 250 300
/absolute/path/to/probe-build/clock_probe atom_decay 250 500
/absolute/path/to/probe-build/formation_probe controlled 250 4
/absolute/path/to/probe-build/formation_probe default 250 50
/absolute/path/to/probe-build/formation_probe light 250 4
/absolute/path/to/probe-build/clock_transport atomic spy 250 120 12345
/absolute/path/to/probe-build/clock_transport stopping spy 250 500 12345
```

Repeat at stop times zero, 250 and 1000 ns. `clock_probe` calls the actual Geant4
process before transport scheduling and writes exact hexadecimal product times,
kinematics, rate readouts and the complete random-engine state. Its fixture runs
the actual stopping process with deterministic interactions: waiting times zero,
0.25, one, two and 123 ns; negative, zero and positive model times; both branches;
a positive 0.375 ns formation-cascade secondary time; and generated finite cases.
The fixture intentionally disables energy-conservation checking for its synthetic
interaction and makes no claim about those synthetic particles' nuclear physics.

The `atom_decay` probe uses the actual `G4MuonicAtomDecay` class and a supplied
controlled nuclear interaction. It tests 30 waiting/local-time combinations and
then the requested number of actual GPIL-sampled waiting times. A small subclass
sets the protected interaction-length counter only for the deterministic boundary
cases; `AtRestGetPhysicalInteractionLength` observes each waiting time before the
product-producing call. This independent observation is also used for the sampled
cases. The atom's standard rates, lifetimes and DIO model are retained. Controlled
model-local times are test inputs, not calibrated physical delays.

`clock_transport` uses QBBC and the standard atomic or stopping route. Its `spy`
mode delegates the stock bound-decay/precompound models unchanged and observes
their times; `plain` omits those wrappers. Passive stepping records the delivered
products. On the atomic route the standard decay process's verbose diagnostic
also reports the sampled waiting time. Both DIO and nuclear capture must occur
before claiming branch coverage; extend a fixed-seed prefix if necessary, retain
the shorter record, and make no rate-distribution claim from that extension.

For a small Linux validation without rebuilding all Geant4, CMake also accepts
`CLOCK_PROCESS_SOURCES`, a semicolon-separated list of the four upstream process
`.cc` files. Compile unmodified copies first and compare their records to the
installed-library baseline. Then compile the copies with the clock patch applied.
This is **application object-overlay validation**, not a rebuilt installation:
only the four process definitions linked into these applications are replaced.
The existing Geant4 shared libraries and data remain in use. Check the executable's
exported `AtRestDoIt` symbols and record the source, patch and executable hashes.

`check_clock.py LOG_ROOT` reads
`LOG_ROOT/<revision>/<variant>-<route>-<stop>.log`, with revisions `v11.4.2` and
`v11.5.0.beta`, routes `atomic`, `stopping`, `fixture`, and stops `0`, `250`, `1000`.
Variants are `installed`, `pristine`, `corrected`, `wrong-clamp`, `duplicate-stop`
and `omit-stop`. The last three use corrected copies with, respectively,
`max(0., waiting_time + model_time)`, a second stop-time addition, or no stop-time
addition. The physical-clock gate must reject all three and the old known defects.
It requires installed/pristine-overlay equality and unchanged non-time records
and full RNG state for matched process inputs. `check_transport.py LOG_ROOT`
reads `<revision>/<installed|corrected>-<atomic|stopping>-<stop>-<spy|plain>.log`,
checks downstream absolute times and requires identical observed spy/plain products.
Supply exact invocation counts explicitly, for example
`--atomic-events 120 --stopping-events 120 --stopping-250-events 500` for a run
that extended only the stopping 250 ns cells. A shorter contiguous prefix fails
even when it contains both branches. The process checker also requires exactly
one rate-readout record and the complete declared deterministic fixture grid.
Both checkers reject nonfinite observed times and empty/missing coverage.

`check_atomic_decay.py NEW_LOG_ROOT PRIOR_TRANSPORT_ROOT --sampled-events 500
--transport-events 120` checks the third consumer's controlled and sampled cases,
rejects the old composition, and requires matched rates, branches, kinematics and
complete random-engine state. Its logs use
`<revision>/<installed|clock1|clock2>-atom_decay-<stop>.log` and
`<revision>/<clock1|clock2>-atomic-<stop>-<spy|plain>.log`; `clock1` denotes the
two-consumer comparator with the original atomic-decay class, and `clock2` the
complete correction. It also requires unchanged nonnegative-model/zero-wait/DIO
clocks and unchanged default atomic transport observations, including comparison
to the prior accepted transport records. This inheritance uses exact observed
records and source hashes; it does not relabel earlier clock failures as passes.

`formation_probe` calls the actual `G4MuMinusCapturePrecompound` wrapper. Its
supplied controlled `DeExcite` dependency crosses independent TOF values
`{-1, 0, 1, 7}` with formation times `{-2, 0, 0.125, 3}` ns. Nonzero origins test
that only the local formation value is clamped. The `light` mode covers the four
bypass target branches and verifies that the supplied dependency is not called.
The `default` mode calls the unmodified standard de-excitation dynamics.

The test-only [formation validation support](formation_validation/README.md)
publishes the observer hooks, print-only trace patches, explicit old/zero comparator
statements and the variant CMake build. It produces all inputs required by
`check_formation.py`, including passive `AUDIT_FRAGMENT`, `AUDIT_RP`, `AUDIT_PICK`
and complete `AUDIT_RNG` records. The bare `formation_probe` examples above are
component probes; use the following complete flow for the checker. On Linux with
Python 3.10+, Git, CMake, a C++17 compiler, both pristine source revisions and their
matching existing Geant4 installations, run from the repository root:

```sh
# Configure the matching installed Geant4 datasets before each revision's run.
# Supply fresh output paths: the script refuses to overwrite an earlier run.
source /absolute/path/to/geant4-11.4.2-install/bin/geant4.sh
python cpp/clock/formation_validation/run.py \
  --revision v11.4.2 \
  --geant4-source /absolute/path/to/pristine-geant4-11.4.2 \
  --geant4-install /absolute/path/to/geant4-11.4.2-install \
  --output-root /absolute/path/to/new-formation-validation

source /absolute/path/to/geant4-beta-install/bin/geant4.sh
python cpp/clock/formation_validation/run.py \
  --revision v11.5.0.beta \
  --geant4-source /absolute/path/to/pristine-geant4-beta \
  --geant4-install /absolute/path/to/geant4-beta-install \
  --output-root /absolute/path/to/new-formation-validation

python cpp/clock/check_formation.py \
  /absolute/path/to/new-formation-validation/logs \
  --controlled-calls 4 --default-calls 50 --integrated-events 120 \
  --output /absolute/path/to/new-formation-validation/result.json
```

The source trees and installations are read-only inputs. The script copies four
source files, applies the complete production patch and then its test-only trace
patch to those fresh copies, builds with two cores and runs the finite matrix.
It inherits dataset environment settings. For a nonstandard installation layout,
pass `--geant4-dir /absolute/path/to/Geant4Config-directory`. Source and install
paths may contain spaces. Configure/build logs and exact prepared-model hashes
are recorded under the output root. The generated source hashes must match the
published tested bodies before compilation. No production installation changes.

`check_formation.py` checks the supplied-field grid, real-default wrapper calls and
integrated atomic products. The logs use
`<revision>/<old|new|zero>-<controlled|default|light>-<origin>.log` and
`<revision>/<old|new|zero>-integrated-250.log`. The comparator restores the old TOF
getter or removes every formation delay; both must fail. Corrected products must
follow `t_stop + sampled_wait + max(raw_formation_time, 0)` on the integrated atomic
route. Matched wrapper inputs require identical non-time final-state fields,
metadata and complete RNG states. This getter correction intentionally changes
default atomic product times; earlier default-transport records remain historical
composition evidence, not a reference for the corrected formation contract.

Finite checks on both named revisions cover these clocks and matched-input
invariants. Corrected times can change downstream scheduling and therefore later
random streams; whole-run equality between corrected and uncorrected transport
is not required. The probes run serially and do not repair or test away the
separately recorded atomic-route worker-first-event dependence. This correction
does not establish isotope-rate theory, experimental capture/decay accuracy, or
completion of the other D1 obligations. Earlier bound-rate clock failures remain
historical results for the builds that produced them.
