# Stopped negative-muon clock correction

`../patches/g4-v11.4.2-stopped-muon-clock-v1.patch` and
`../patches/g4-v11.5.0.beta-stopped-muon-clock-v1.patch` are separate patches against
the named pristine Geant4 tags. Applying a patch explicitly changes that build's
default timing. There is no runtime switch. Neither the dataset patch nor
`SetEnableMuonicBoundRates` applies or enables this correction. The clock and
dataset/rate patches apply in either order.

The correction changes two consumers:

- `G4MuonMinusAtomicCapture::AtRestDoIt` gives the new muonic atom local time zero.
  Its common secondary loop adds the stopped muon's global time once. Previously,
  the atom already carried that global time and the loop added it again.
- `G4HadronStoppingProcess::AtRestDoIt` maps an unset negative nuclear-model
  secondary time to zero **before** adding the sampled capture waiting time.
  Previously, the default `G4HadSecondary` sentinel of minus one nanosecond was
  added to that waiting time and the composed result was clamped afterward.
  A legitimate positive model delay is preserved.

Under the existing instantaneous atomic-formation approximation, the atom is
created at `t_stop`. Bound-decay products arrive at `t_stop + waiting_time`;
nuclear-capture products arrive at
`t_stop + waiting_time + max(model_secondary_local_time, 0)`.
The atomic route's subsequent precompound-capture model retains its own positive
secondary delays. Positive formation-cascade secondary times are retained too;
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

`clock_transport` uses QBBC and the standard atomic or stopping route. Its `spy`
mode delegates the stock bound-decay/precompound models unchanged and observes
their times; `plain` omits those wrappers. Passive stepping records the delivered
products. On the atomic route the standard decay process's verbose diagnostic
also reports the sampled waiting time. Both DIO and nuclear capture must occur
before claiming branch coverage; extend a fixed-seed prefix if necessary, retain
the shorter record, and make no rate-distribution claim from that extension.

For a small Linux validation without rebuilding all Geant4, CMake also accepts
`CLOCK_PROCESS_SOURCES`, a semicolon-separated list of the two upstream process
`.cc` files. Compile unmodified copies first and compare their records to the
installed-library baseline. Then compile the copies with the clock patch applied.
This is **application object-overlay validation**, not a rebuilt installation:
only the two process definitions linked into these applications are replaced.
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
Both checkers reject nonfinite observed times and empty/missing coverage.

Finite checks on both named revisions cover these clocks and matched-input
invariants. Corrected times can change downstream scheduling and therefore later
random streams; whole-run equality between corrected and uncorrected transport
is not required. The probes run serially and do not repair or test away the
separately recorded atomic-route worker-first-event dependence. This correction
does not establish isotope-rate theory, experimental capture/decay accuracy, or
completion of the other D1 obligations. Earlier bound-rate clock failures remain
historical results for the builds that produced them.
