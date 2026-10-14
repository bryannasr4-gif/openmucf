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

This paper is a draft and has not been submitted.
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

Changing any of the compiled-in capture data means recompiling the toolkit, and nothing tells a user
which rows rest on an isotope-resolved measurement.
A user who simulates muonic X-ray emission or muon capture with an unpatched Geant4 therefore cannot
substitute a newer measurement without editing and rebuilding the toolkit; with this layer's patches
built in and the opt-in set, a value comes from a dataset directory an environment variable names
only where the selected profile has a row for it.
Newer measurements exist: Mizuno et al. measured the lifetimes of the muonic atoms of enriched
silicon isotopes and print capture rates for them [@Mizuno2025], among them rates for isotopes the
compiled-in table has no row for.
The compiled-in table stores an uncertainty beside each capture rate that a comment in its source says
is not used now; this layer's capture rates and energies carry uncertainties too, and they are
carried, not consumed: the patched lookups read only the values.
The format is not tied to Geant4: its specification addresses Geant4 and any other transport code.

# State of the field

The Muon Nuclear Data Development Project aims to construct a data library for muon capture
reactions whose sub-libraries include muonic X-ray energies and intensities, and lifetimes of
muonic atoms and nuclear capture rates [@Watanabe2026].
Cataldo et al. reported preliminary results of implementing a database of MuDirac transition
energies in Geant4 [@Cataldo2023], taking a transition's energy from the target's MuDirac file wherever
it was found, which, as they note, loses the information from Geant4's own cascade
calculation; here the tables' energies enter Geant4's cascade instead.
Geant4 already ships optional data behind an opt-in: Geant4 11.3 added the optional NuDEX
de-excitation model [@Mendoza2020] with the optional dataset G4NUDEXLIB, read through the
`G4NUDEXLIBDATA` environment variable and enabled by the line
`G4HadronicParameters::Instance()->SetEnableNUDEX(true);` in the main program, before the physics
list is instantiated [@Geant4ReleaseNotes].
G4CASCADE is a data-driven Geant4 module that simulates (n, γ) de-excitation pathways
[@Weimer2024].
Of the formats for evaluated nuclear data [@Schnabel2023], the table of sub-libraries of the ENDF-6
formats manual names none for muons or muonic atoms [@ENDF6Manual]; Generalized Nuclear Data, as
Mattoon et al. describe it, leaves file storage to a meta-language such as XML or HDF5 [@Mattoon2012],
and Geant4's LEND package reads GNDS data through a library and an XML parser it carries.
The layer described here extends Geant4's existing muonic-atom code rather than replacing it:
every hunk its patches add to an existing Geant4 file only inserts, except the calls to
`GetKShellEnergy` that now pass the mass number and the line that renames its compiled-in
definition, and its opt-in takes the form NuDEX's does,
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
A request the selected table finds no row for falls through to the compiled-in code; under
`parity` the fallback formula is reproduced as it is, including the negative rates the dataset's
documentation registers, and under the profiles that carry published capture-rate measurements a
computed rate that is not positive and finite raises a fatal exception.
Before any table is selected, the glue checks what every loaded table means, including its keys,
the units of the columns a lookup reads and the domain of its values, and refuses the whole dataset
with a fatal exception naming the rule that failed and the file and table, or the profile, it
failed in.

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
records the SHA-256 of each patch, executable and loaded library.
Geant4's choice of the element whose nucleus takes a stopped muon in a material of several elements is not made a data seam here; `DATASET_D2.md` documents how Geant4 makes that choice.

# Research impact statement

In a target of natural composition, Geant4's capture-rate table answers only the stops that draw an
isotope it holds a record for, and the Goulard–Primakoff formula answers the rest; as an
abundance-weighted first-order estimate against the natural measurement the table itself encodes, the weighted rate differs from the record's by more than the record's tabulated
relative uncertainty for 25 of the 41 elements with a natural-composition record, by as much as
+12.5 % for Hg (`DATASET_D1.md` section 11); a Geant4 bug report had reported incorrect rates
for nuclei the table does not list [@Geant4Bugzilla].
Under a rule adopted after the comparison had been made, the ratio Geant4's capturing-atom selector
gives lies above the measured per-atom capture ratio on each of the 63 gated ratios of an element to
oxygen, at 2.96 times it at the median; the comparison does not tell whether the difference arises in
the selector's weights or in the steps by which the muon reaches the element, and for the calcium
hydroxide row transfer of the muon from hydrogen may explain it (`DATASET_D2.md`).
The MuDirac output behind the binding-energy tables is compared with the measured energies of
Fricke et al. [@Fricke1995] and Saito et al. [@Saito2025], each as printed, at a
tolerance of 3 times the printed standard uncertainty: of the 67 gated rows, the line MuDirac
prints lies within it for 26, the shell difference the patched cascade receives for 1, and the
energy the unpatched cascade emits for 0.
A cascade that emits a single energy per pair of shells cannot lie within the band of every line it
stands for, as 29 of the 29 multi-line groups show; yet the shell difference lies closer than the
unpatched cascade's energy on 42 of the 67 gated rows, and in a retrospective comparison of the K–L
difference that qualifies no accuracy, its largest absolute residual is 4.26 keV where the unpatched
cascade's reaches 227 keV (`DATASET_D3.md` section 5).
The rows outside tolerance are registered disagreements, and none is fitted away.
Geant4 computes a bound muon's disappearance rate in `G4MuonMinusBoundDecay`, which the reference
physics lists reach, and in `G4MuonicAtomHelper`, which needs `G4MuonMinusAtomicCapture` and the
`G4MuonicAtomDecayPhysics` constructor, each with its
own decay rate; against measured mean lives under `parity`, `G4MuonMinusBoundDecay`'s rate lies within
three times the combined uncertainty on 47 of 51 compared rows and `G4MuonicAtomHelper`'s on 43,
and `G4MuonicAtomHelper`'s decay rate lies below the measured rate less the printed capture rate for
54 of the 57 measurements (`DATASET_D1.md` section 12).
Both compare single-nucleus rates, not transported natural targets, and the band errs
wide, as its uncertainties can rest on the same mean life.
Multithreaded runs on the `muonic_atom_helper` route aborted with `PART122` in unpatched v11.4.2 and
v11.5.0.beta, because `G4IonTable::GetMuonicAtom` constructs a new muonic atom before it takes the
ion table's lock (`cpp/transport/README.md`).
Building the `parity` profile also registered defects, disclosed rather than fixed: the fallback
returns negative capture rates on 6325 of the 36000 swept points, tritium among them, though how many
of those points are bound nuclides is not established; it returns non-finite rates at degenerate
inputs, a missing input validation; and, as a build-reproducibility note, fusing a multiply and an add
into a single rounded operation moves its result by up to 2980 ulp, a physically negligible difference.
Every `.g4dat` table under `data/g4/` is rebuilt from committed inputs and byte-compared in
continuous integration.

# AI usage disclosure

<!-- TODO: the author completes this section before any submission. -->

# Acknowledgements

<!-- TODO: the author completes this section before any submission. -->

# References
