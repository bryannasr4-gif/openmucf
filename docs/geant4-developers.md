# For Geant4 developers

This repository ships Geant4 muonic-atom data as an external, versioned dataset, with provenance and uncertainty attached, that a patched Geant4 consults only after an explicit opt-in ([`paper.md`](../paper/muonic-data/paper.md)).

## Try it

In an empty Linux directory, with pristine v11.4.2 source at `/path/to/geant4-v11.4.2` (v11.5.0.beta: `g4-v11.5.0.beta-muonicdata.patch`; [`cpp/patches/README.md`](../cpp/patches/README.md)):

```sh
git clone https://github.com/bryannasr4-gif/openmucf.git
git -C /path/to/geant4-v11.4.2 apply "$PWD/openmucf/cpp/patches/g4-v11.4.2-muonicdata.patch"
mkdir G4MuonicData && cp openmucf/data/g4/d1/*.g4dat openmucf/data/g4/d3/*.g4dat G4MuonicData/ && export G4MUONICDATA="$PWD/G4MuonicData"
```

Rebuild Geant4, then opt in before your physics list is built ([`DATASET_D1.md`](../DATASET_D1.md) section 7):

```cpp
G4HadronicParameters::Instance()->SetEnableMuonicData(true);  // #include "G4HadronicParameters.hh"
```

Exporting `G4MUONICDATA_D3_PROFILE=mudirac130` too gives the cascade the MuDirac K and shell energies up to the chain's last ([`DATASET_D3.md`](../DATASET_D3.md) section 7).

## What has been checked

- **Parity.** The `parity` tables reproduce the capture data compiled into Geant4 v11.4.2 and v11.5.0.beta bit-for-bit on the builds [`DATASET_D1.md`](../DATASET_D1.md) section 4 names; a committed digest lets CI check this without Geant4.
- **Default behaviour.** The unpatched build and the patched build with the opt-in off, or on with no profile selected, write identical records, and sorted records match across the thread counts the workflow runs ([`cpp/transport/README.md`](../cpp/transport/README.md)).
- **Energies against measurement.** The MuDirac lines and a patched cascade's shell differences are compared with energies Fricke et al. (1995) and Saito et al. (2025) measured ([`DATASET_D3.md`](../DATASET_D3.md) sections 4 and 5); rows outside tolerance are registered disagreements, and none is fitted away. The shell difference lies closer than the unpatched cascade's energy on 42 of the 67 gated rows, and in a retrospective comparison of the K–L difference that qualifies no accuracy, its largest absolute residual is 4.26 keV where the unpatched cascade's reaches 227 keV.
- **Platforms.** The patched trees were built and tested on Linux only; CI builds the reader and its validator, not a patched Geant4, on Linux, macOS and Windows ([`cpp/patches/README.md`](../cpp/patches/README.md)).

## What building it showed about Geant4

- **Natural targets.** `G4ElementSelector::SelectZandA` draws the capturing isotope from the element's abundances; the capture table answers only a nucleus whose Z and A both match a record, and the Goulard–Primakoff formula answers every other, so a target of natural composition reaches an element's record only on the stops that draw the record's own mass number. As an abundance-weighted first-order estimate against the natural measurement the table itself encodes, the weighted rate differs from the record's by more than the record's tabulated relative uncertainty for 25 of the 41 elements with a natural-composition record; a Geant4 bug report had reported incorrect rates for nuclei the table does not list ([`DATASET_D1.md`](../DATASET_D1.md) section 11).
- **Capturing-atom selector.** `G4ElementSelector::SelectZandA` weights elements by atomic number times atom number density (reduced for halogens and oxygen) and has no step that moves a muon between elements. Against measured per-atom ratios ([`DATASET_D2.md`](../DATASET_D2.md)), a first rule gates no row; a second, adopted after the selector's ratio had already been compared with the rows it gates, gates 62 rows in oxides and the calcium hydroxide row, and the selector's ratio lies above the measured ratio on each of the 63 gated ratios of an element to oxygen, at 2.96 times it at the median. For the calcium hydroxide row, of a material that holds hydrogen, the difference may come from such transfer rather than from the weights, and this comparison does not tell them apart; for the rows whose material holds no hydrogen, it does not tell whether the difference arises in the weights or in the steps by which the muon reaches that element.
- **Shell-level cascade.** The cascade, patched or not, emits a single energy per shell whatever the orbit; for every multi-line group (same source, nuclide and shell pair) in [`DATASET_D3.md`](../DATASET_D3.md) section 5, no single shell energy can lie within the band of every line it stands for.
- **Bound-decay routes.** Geant4 computes a bound muon's disappearance rate in `G4MuonMinusBoundDecay`, which the reference physics lists QBBC and FTFP_BERT reach, and in `G4MuonicAtomHelper`, which a user reaches by registering `G4MuonMinusAtomicCapture` and the `G4MuonicAtomDecayPhysics` constructor, each with its own decay rate. Against measured mean lives under `parity`, `G4MuonMinusBoundDecay`'s rate lies within three times the combined uncertainty on 47 of 51 compared rows and `G4MuonicAtomHelper`'s on 43, and `G4MuonicAtomHelper`'s decay rate lies below the measured rate less the printed capture rate for 54 of the 57 measurements. The comparison is of the rates the code returns for a single nucleus, not of a transported simulation of a target, and the uncertainties its band combines can rest on the same mean life, so the band errs wide ([`DATASET_D1.md`](../DATASET_D1.md) section 12).
- **Threading race.** Multithreaded runs on the `muonic_atom_helper` route aborted with `PART122` in unpatched v11.4.2 and v11.5.0.beta, because `G4IonTable::GetMuonicAtom` constructs a new muonic atom before it takes the ion table's lock; the harness pre-creates it on the master thread ([`cpp/transport/README.md`](../cpp/transport/README.md)).
- **Capture defects.** Building the `parity` profile registered defects in the fallback, disclosed rather than fixed: negative capture rates on part of the swept box, tritium among them, though how many of those points are bound nuclides is not established (F-1); non-finite returns at degenerate inputs that nothing rejects, a missing input validation (F-2); and results that move between conforming compiler configurations, a build-reproducibility matter the documentation calls physically negligible (F-3) ([`DATASET_D1.md`](../DATASET_D1.md) section 5).

## Taking it upstream

- **Reader.** `G4MuonicDataTable`, a standard-library-only C++17 reader for the `.g4dat` grammar of [`FORMAT_SPEC.md`](../FORMAT_SPEC.md), has no Geant4 dependency; CI runs its standalone validator on Linux, macOS and Windows ([`cpp/README.md`](../cpp/README.md)).
- **Opt-in.** It takes the form NuDEX's does: `SetEnableMuonicData(true)` on the `G4HadronicParameters` singleton, `false` by default ([`paper.md`](../paper/muonic-data/paper.md), [`cpp/patches/README.md`](../cpp/patches/README.md)).
- **Uncertainties.** The capture rates and energies carry their uncertainties in the dataset, and the patched lookups read the values and never the uncertainties: they are carried, not consumed ([`cpp/src/G4MuonicDataOverlay.cc`](../cpp/src/G4MuonicDataOverlay.cc)).
- **Names, registration.** `G4MuonicData` and `G4MUONICDATA` are provisional, pending discussion with the Geant4 collaboration ([`README.md`](../README.md)); resolving the dataset under `GEANT4_DATA_DIR` requires an upstream change to Geant4 ([`FORMAT_SPEC.md`](../FORMAT_SPEC.md) section 5), which the separate registration patches make ([`cpp/patches/README.md`](../cpp/patches/README.md)).
