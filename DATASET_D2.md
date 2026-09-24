# D2 — the capturing-atom selector and the atomic-capture reference corpus

## The selector

Geant4 chooses the element whose nucleus takes a stopped negative muon in `G4ElementSelector::SelectZandA`, weighting each element of the material by its atomic number times its atom number density, with a reduced factor for the halogens and for oxygen. The weighting runs only when the material holds more than a single element, and the same function then draws the isotope from the element's relative abundances. `G4HadronStoppingProcess::AtRestDoIt` calls the selector before the capture cascade runs, and `G4MuonMinusAtomicCapture` calls it on the muonic-atom route. The stopping constructors register process classes derived from `G4HadronStoppingProcess` for the negative muon, pion and kaon and for the antiproton, so the same selector picks the capturing element for each of them. `SelectZandA` contains no step that moves the muon from the element it picks to another, so the element it picks is the element whose nucleus the capture code then uses. A review of muon capture explains that a muon captured by hydrogen forms a neutral system that can pass to a heavier nucleus nearby, and calls polyethylene effectively a carbon target and water a convenient oxygen target.

## The reference corpus

`selector_harvest.csv` records, for each material the harness `cpp/tools/harvest_d2.cc` builds, the element that each draw of a fixed ladder of uniform draws selects in a compiled Geant4 library, and `openmucf/g4/d2.py` reproduces every count from the harvested atom number densities. `harvest_builds.csv` names the Geant4 builds, `selector_bindings.csv` lists the process each of these particles meets at rest in each reference physics list the harness builds, and `selector_call_sites.csv` lists the calls of the selector and the classes derived from `G4HadronStoppingProcess` in each Geant4 source tree. `reference/d2/sources.csv` lists each publication this corpus considered, the route by which it was reached and whether a copy was read. `printed_rows.csv` transcribes the measured values those copies print, row by row as printed, and `measurements.csv` restates each with the method and corrections its source states. `review_quoted.csv` holds values a review quotes, and no code that computes a comparison reads it. A per-atom capture ratio divides the fraction of muons captured by an element, per atom of that element in the target, by the same quantity for another element. A primary in this corpus contrasts the muonic X-ray and decay-electron lifetime methods: for muonic X-ray intensities the energy dependence of the detector's efficiency is a serious problem, and for decay-electron lifetimes the main correction is for muon absorption, which reduces the number of decay electrons. `selector_vs_primary.csv` sets beside each measurement row the per-atom ratio the selector's weights give for the same pair of elements, and marks a row as gated only when it measures the element that finally takes the muon among all muons stopped in the target, prints an uncertainty, and its source states the method and the corrections behind the ratio. Nothing in these files changes how Geant4 picks the capturing element.

## Building the harvest

The harvest is built and run against each Geant4 install as follows:

```sh
g++ -std=c++17 -O2 -ffp-contract=off cpp/tools/harvest_d2.cc -o harvest_d2 $(<install>/bin/geant4-config --cflags --libs)
env -i PATH=/usr/bin:/bin HOME=$HOME LD_LIBRARY_PATH=<install>/lib GEANT4_DATA_DIR=<data> ./harvest_d2 selector > selector.txt
env -i PATH=/usr/bin:/bin HOME=$HOME LD_LIBRARY_PATH=<install>/lib GEANT4_DATA_DIR=<data> ./harvest_d2 bindings <physics list> > bindings.txt
```
