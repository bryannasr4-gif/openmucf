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
- **Energies against measurement.** The MuDirac lines and a patched cascade's shell differences are compared with energies Fricke et al. (1995) and Saito et al. (2025) measured ([`DATASET_D3.md`](../DATASET_D3.md) sections 4 and 5); rows outside tolerance are registered disagreements, and none is fitted away.

## What building it showed about Geant4

- **Capture defects.** Building the `parity` profile surfaced defects, registered and disclosed rather than fixed, among them negative fallback capture rates (F-1), non-finite returns at degenerate inputs (F-2) and fallback results that move between conforming compiler configurations (F-3) ([`DATASET_D1.md`](../DATASET_D1.md) section 5).
- **Shell-level cascade.** The cascade, patched or not, emits a single energy per shell whatever the orbit; for every multi-line group (same source, nuclide and shell pair) in [`DATASET_D3.md`](../DATASET_D3.md) section 5, no single shell energy can lie within the band of every line it stands for.
- **Capturing-atom selector.** `G4ElementSelector::SelectZandA` weights elements by atomic number times atom number density (reduced for halogens and oxygen) and has no step that moves a muon between elements. Against measured per-atom ratios ([`DATASET_D2.md`](../DATASET_D2.md)), a first rule gates no row; a second, adopted after the selector's ratio had already been compared with the rows it gates, gates only lifetime measurements in oxides and other compounds, all `outside`; the difference may come from such transfer rather than from the weights, and this comparison does not tell them apart.
- **Threading race.** Multithreaded runs on the `muonic_atom_helper` route aborted with `PART122` in unpatched v11.4.2 and v11.5.0.beta, because `G4IonTable::GetMuonicAtom` constructs a new muonic atom before it takes the ion table's lock; the harness pre-creates it on the master thread ([`cpp/transport/README.md`](../cpp/transport/README.md)).

## Taking it upstream

- **Reader.** `G4MuonicDataTable`, a standard-library-only C++17 reader for the `.g4dat` grammar of [`FORMAT_SPEC.md`](../FORMAT_SPEC.md), has no Geant4 dependency; CI runs its standalone validator on Linux, macOS and Windows ([`cpp/README.md`](../cpp/README.md)).
- **Opt-in.** It takes the form NuDEX's does: `SetEnableMuonicData(true)` on the `G4HadronicParameters` singleton, `false` by default ([`paper.md`](../paper/muonic-data/paper.md), [`cpp/patches/README.md`](../cpp/patches/README.md)).
- **Names, registration.** `G4MuonicData` and `G4MUONICDATA` are provisional, pending discussion with the Geant4 collaboration ([`README.md`](../README.md)); resolving the dataset under `GEANT4_DATA_DIR` requires an upstream change to Geant4 ([`FORMAT_SPEC.md`](../FORMAT_SPEC.md) section 5), which the separate registration patches make ([`cpp/patches/README.md`](../cpp/patches/README.md)).
