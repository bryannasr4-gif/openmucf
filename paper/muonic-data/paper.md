---
title: 'An external, provenance-carrying data layer for the muonic-atom physics of Geant4'
tags:
  - Python
  - C++
  - Geant4
  - muonic atoms
  - muon capture
  - nuclear data
  - reproducible research
authors:
  - name: Bryan Nasr
    orcid: 0009-0008-2360-7522
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 2026-09-24
bibliography: paper.bib
---

# Summary

A negative muon stopped in matter is captured into an atomic orbit, forming a muonic atom; it
cascades to the ground state while emitting muonic X-rays, and then either decays or is captured
by the nucleus [@Watanabe2026].
The simulation toolkit Geant4 [@Agostinelli2003] carries its muon-capture data compiled in:
90 `{Z, A, rate, error}` records and a 101-value effective-charge table, with a
Goulard–Primakoff analytic fallback for everything else.
This software ships such data as an external, versioned dataset, with its provenance and its
uncertainty attached, that a patched Geant4 consults only after an explicit opt-in.
It carries the capture data in a `parity` profile that reproduces the values compiled into Geant4
bit for bit on the builds its documentation names and in profiles that carry published
capture-rate measurements, and it carries muonic binding energies derived from the output of
MuDirac, a Dirac-equation solver for muonic atoms [@Sturniolo2021; @Liborio2026].
The dataset name `G4MuonicData` and the environment variable `G4MUONICDATA` are provisional
placeholders, pending discussion with the Geant4 collaboration.

# Statement of need

Changing any of the compiled-in capture data means recompiling the toolkit, the per-record
uncertainties it already stores are never used, and nothing tells a user which rows rest on an
isotope-resolved measurement.
A user who simulates muonic X-ray emission or muon capture with Geant4 therefore cannot substitute
a newer measurement without editing and rebuilding the toolkit.
Newer measurements exist: Mizuno et al. measured the lifetimes of the muonic atoms of enriched
silicon isotopes and print capture rates for them [@Mizuno2025], among them rates for isotopes the
compiled-in table has no row for.
The format is not tied to Geant4: its specification addresses Geant4 and any other transport code.

# State of the field

The Muon Nuclear Data Development Project aims to construct a data library for muon capture
reactions whose sub-libraries include muonic X-ray energies and intensities, and lifetimes of
muonic atoms and nuclear capture rates [@Watanabe2026].
Cataldo et al. reported preliminary results of implementing a database of MuDirac transition
energies in Geant4 [@Cataldo2023].
Geant4 already ships optional data behind an opt-in: Geant4 11.3 added the optional NuDEX
de-excitation model [@Mendoza2020] with the optional dataset G4NUDEXLIB, read through the
`G4NUDEXLIBDATA` environment variable and enabled by the line
`G4HadronicParameters::Instance()->SetEnableNUDEX(true);` in the main program, before the physics
list is instantiated [@Geant4ReleaseNotes].
G4CASCADE is a data-driven Geant4 module that simulates (n, γ) de-excitation pathways
[@Weimer2024].
The layer described here extends Geant4's existing muonic-atom code rather than replacing it:
every hunk its patches add to an existing Geant4 file only inserts, except the calls to
`GetKShellEnergy` that now pass the mass number, and its opt-in takes the form NuDEX's does,
`SetEnableMuonicData(true)` on the `G4HadronicParameters` singleton.

# Software design

Geant4's core must be able to read a dataset with no third-party dependency, while provenance and
uncertainty need a nested representation, so every table ships as a pair of files: a US-ASCII
`.g4dat` file that a C++ reader parses with the standard library alone, and a `*.prov.json` file
holding, for each record, its bibliographic source, uncertainty type, evaluation identity and
disclosure flags.
The `.g4dat` file is generated from the provenance file by a single tool and carries the SHA-256 of
that file's bytes, so the pair cannot drift without a checkable error.
The reader, `G4MuonicDataTable`, is standard-library-only C++17 with no Geant4 dependency, and a
standalone validator built on it runs in continuous integration on Linux, macOS and Windows.

Each table names a profile, and a patched build reads the nuclear-capture seam and the
muonic-transition seam each through a profile of its own, chosen by an environment variable;
where none is set, capture reads `parity` and the transitions keep the compiled-in code.
A key the selected table lacks falls through to the compiled-in code, so the fallback formula is
reproduced as it is, including the negative rates the dataset's documentation registers.
Before any table is selected, the glue checks what every loaded table means, including its keys,
the units of the columns a lookup reads and the domain of its values, and refuses the whole dataset
with a fatal exception naming the rule, the file and the table where a check fails.

The `parity` profile is the fixed point against which every other profile is measured, so it
reproduces the compiled-in data including its defects instead of correcting them.
Its tables are generated from the vendored Geant4 source, pinned by its upstream git blob id,
rather than transcribed.
A Geant4-linked driver evaluated the compiled capture-rate function over a 36000-point box, a
pure-Python evaluation reproduced every point bit for bit, and a committed digest of that sweep
lets continuous integration check parity with no Geant4 installed.
The patches are cut against Geant4 v11.4.2 and v11.5.0.beta, and a separate patch for each revision
registers the dataset so that Geant4 resolves it under `GEANT4_DATA_DIR`, from an archive that
unpacks to a `G4MuonicData<version>` directory.
Under the `mudirac130` profile the patched cascade takes the K energy and the energy of every shell
above it up to the chain's last from the tables, and keeps its hydrogen-like formula for the shells
above those and its branching and random draws unchanged.

A transport workflow builds each revision from a clean export of its tag with and without the
patches, transports negative muons stopped in single-isotope targets through both builds, and
requires the unpatched build and the patched build with the opt-in off, or on with no profile
selected, to write identical records.
Under the profiles its matrix selects, it also requires each cascade transition between tabulated
levels to carry the difference of their table values, within the tolerance the script states; it
records the SHA-256 of each patch, executable and loaded library, and a test requires every row of
its committed results to be a pass or an informational record.
Geant4's choice of the element whose nucleus takes a stopped muon in a material of several elements is not made a data seam here; `DATASET_D2.md` documents how Geant4 makes that choice.

# Research impact statement

Building the `parity` profile surfaced defects in the compiled-in seam that are registered and
disclosed rather than fixed: the fallback returns negative capture rates on 6325 of the 36000
swept points, and fusing a multiply and an add in its expression into a single rounded operation
moves its result by up to 2980 ulp.
The MuDirac output behind the binding-energy tables is compared with the measured energies of
Fricke et al. [@Fricke1995] and Saito et al. [@Saito2025], each as printed, at a
tolerance of 3 times the printed standard uncertainty: of the 67 gated rows, the line MuDirac
prints lies within it for 26, the shell difference the patched cascade receives for 1, and the
energy the unpatched cascade emits for 0.
The rows outside tolerance are registered disagreements, and none is fitted away.
Every `.g4dat` table under `data/g4/` is rebuilt from committed inputs and byte-compared in
continuous integration, and the transport workflow's manifest and per-check results are committed.

# AI usage disclosure

<!-- TODO: the author completes this section before any submission. -->

# Acknowledgements

<!-- TODO: the author completes this section before any submission. -->

# References
