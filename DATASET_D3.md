# The D3 energy tables of the `mudirac130p1` profile

## 1. What ships

`data/g4/d3/` ships the tables `k_shell_energy` (`d3_kshell.mudirac130p1.g4dat`) and `level_energy`
(`d3_levels.mudirac130p1.g4dat`) under `#PROFILE mudirac130p1` and `#SEAM d3_transitions`, each with its
Layer-2 file.
Every `value` and `e<n>` cell is a positive binding energy in keV derived from the output of MuDirac 1.3.0 (Sturniolo and
Hillier (2021); Liborio et al. (2026)), a Dirac-equation solver for muonic atoms released under the
MIT License with the copyright held by the Science and Technology Facility Council, as its
`LICENSE` reads, built with the patch `third_party/mudirac/mudirac-1.3.0-p1.patch` applied.
The profile's name says so: MuDirac 1.3.0 with the patch set in `third_party/mudirac/`, `p1` its first
published revision.
The patch is a numerical and Hamiltonian repair of the runs (section 2), not a corrected resolved spectrum: the
physics it leaves out is listed in section 4, and section 6 shows what a cascade that takes one energy per shell
cannot represent.
It replaces the `mudirac130` profile, whose tables came from runs of the unpatched MuDirac 1.3.0; they ship in
release v1.3.0 of this repository (version DOI 10.5281/zenodo.22995079) and no longer in this directory.
A Geant4 configuration that still names `mudirac130` stops with `G4MuonicData004`, because the dataset carries no
table under that `#PROFILE` (`cpp/src/G4MuonicDataOverlay.cc`); it is never given other numbers under that name.
`k_shell_energy` carries the binding energy of the `K1` orbit in its `value` column, with the
magnitude of its radius response in `unc`.
`level_energy` carries, for each shell above `K`, the mean binding energy of both circular states
of the shell weighted by their degeneracy, in the columns `e<n>`, with the magnitudes of their
radius responses in `u<n>`.
A magnitude below the floor stated in the next section ships as that floor.
The chain ends at shell 8; above it a patched cascade keeps its hydrogen-like formula, and section 3
checks the circular states of the chain's last shells against that solution.
Each table has a row for every kept (Z, A) and, for each element whose most abundant isotope in
MuDirac's `abundant.dat` is kept, a natural-composition row that carries that isotope's values
under `#VALIDITY` `A:most_abundant_and_listed`.
Of the 895 members of the input set, 892 are kept, and the tables carry 81
natural-composition rows.

## 2. How the values are made

The input set is every (Z, A) with Z from 1 through 92 that MuDirac's bundled
`nuclear_radii.dat` lists.
Every input starts from the same keywords: `nuclear_model: FERMI2`, `uehling_correction: TRUE`,
`reduced_mass: TRUE`, `optimise_fermi_parameters: FALSE`, `fermi_t: 2.3`; no `electronic_config` is given, so the atom
carries no electrons.
MuDirac's keyword documentation attributes its default radii to Angeli and Marinova (2013).
Every production input then sets the patch's keywords, each described in `third_party/mudirac/README.md`:
`uehling_density: fermi2`, so that the Uehling potential is that of the run's own 2pF charge density, the one
the Coulomb potential uses; `fermi2_rms_exact` with `fermi2_rms_fm`, so that the 2pF parameter c is solved for the
intended rms radius; `nuclear_mass`, so that the reduced mass is taken with the nuclear mass; `ev_codata2022`, so
that energies are printed in eV through the CODATA 2022 Hartree energy; `fermi2_hermite`; and the numerical
settings `loggrid_step: 0.000625`, `energy_tol: 1e-07`, `uehling_steps: 60` and `uehling_shell_nodes: 12`.
Each run prints its lines to 12 decimals and the energy header of each state to 17 significant
digits.
The inputs are committed as `mudirac_inputs.csv`, and what MuDirac printed as `mudirac_runs.csv`,
`mudirac_states.csv` and `mudirac_lines.csv`, for every run of four batches: `prod`, each kind of each member
beside its twin without the Uehling potential (`uehling_correction: FALSE`); `refine`, each member's anchor run
with one numerical setting one step finer (section 5); `rminus`, each member's rms radius moved down by its
uncertainty; and `stages`, each compared nuclide's v1.3.0 input and its input at each step of the repair.
The run table lists 8994, 5370, 892 and 150 runs in those batches.
Of a hydrogen-like run only the circular states of the checked shells are committed, the only values
section 3 reads, and of no run the transition rates MuDirac prints beside its lines.
`scripts/mudirac_d3.py` renders the inputs, runs MuDirac and collects what it prints; the collector
refuses a run that lacks a line or a state header its input implies, exits with a nonzero status or
leaves a non-empty error file.
Every input sets `optimise_fermi_parameters: FALSE` and every run is given its input file as its
only argument, so MuDirac's fitting mode, which reads measured energies, never runs.
The tables are rebuilt from the committed CSV files by `scripts/generate_g4data.py` and
byte-compared by its audit; MuDirac itself is not run in CI, and cross-compiler reproducibility of
its output is not claimed.
The binding energies are derived from the printed strings by exact decimal sums: the outermost
circular orbit of shell 8, `R15`, is the printed binding energy of the hydrogen-like orbit `S17` plus
the printed line `R15-S17`, every other circular orbit follows by adding or subtracting printed
lines, and each derived energy is checked against its own printed header, within the rounding of
the strings summed and the spacing of the doubles in which MuDirac holds a total energy.
`value` and `e<n>` come from the anchor run with the Uehling potential; `unc` and `u<n>` are the
absolute change of the same quantity in the `rsig` run, whose rms radius is moved by its uncertainty
in the IAEA charge-radii table (`charge_radii.csv`), measured through the printed line energies with
the energy of the outermost circular state held fixed, and propagate nothing else.
A smaller change ships as the floor its row's Layer-2 `evaluation_method` states, the rounding of the
printed lines summed.
Where the bundled rms radius and that table's value differ at the fourth decimal, the generator
prints the member; neither value is edited.

## 3. Which nuclides are kept

A member is kept only when the intended rms radius of its anchor and `rsig` runs exceeds 1.882698418777337
fm, the rms radius of the 2pF density of thickness 2.3 fm whose half-density radius c is zero, below which no
2pF density of that thickness reaches;
when each circular state of shells 6 through 8 lies within 1 % of the same state in the run that treats the
atom as hydrogen-like from that shell up (`ideal_atom_minshell`), with the Uehling potential and without it;
and when the derivation holds on its anchor and `rsig` runs, with the Uehling potential and without it.
The collector admits only complete runs.
The generator drops `H1`, `H3` and `He4`, whose intended rms radii lie below that floor, and
prints each with its radii.
An element whose most abundant isotope is dropped has no natural-composition row, so a lookup for
it falls through to Geant4's compiled-in code.

## 4. Comparison with measured energies

`validation.csv` compares the model line, the difference between the energies of its initial and
final states as the anchor run with the Uehling potential prints it, with the energies transcribed
from Fricke et al. (1995), Tables IIIA and IIIB, whose titles call them transition energies, and from
Saito et al. (2025), whose Table III is titled "Observed muonic X-ray energies" and Table IV "Observed
muonic X-ray transition energies", each as printed (`validation_cells.csv`).
Saito et al. write: "The observed X-ray energies in Table III are different from the transition
energies because they are affected by the photon recoil effect"; no recoil correction is applied
to either side of this comparison.
The Fricke values were read from page images of the journal copy.
Each uncertainty carries its source's own label: `statistical` (Table IIIA), `statistical and fit`
(Table IIIB) or `statistical and systematic` (Saito et al.).
The tolerance is 3 times the printed standard uncertainty, and σ_calc, the model's own
uncertainty, is set to 0 by decision.
That is a decision, not a figure taken from either paper: Liborio et al. (2026) write on p. 2 that
"MuDirac can compute muonic atom transition energies up to precisions on the order of the keV",
and Sturniolo and Hillier (2021) state a precision for eigenenergies on p. 2 of their arXiv copy.
No bound on the model's own uncertainty is stated: the tables leave out nuclear polarization, the
self energy, vacuum polarization beyond the Uehling potential, recoil beyond the reduced mass,
electron screening, hyperfine structure and nuclear charge distributions other than 2pF, the
omitted components the header of `components.jsonl` lists.
No value is changed for lying outside the tolerance, and omitted physics is not absorbed into it:
each Table IIIA row carries the nuclear-polarization correction that table prints beside it (`NPol`).
A line's band is the range of energies within that tolerance of its measured value.
`dE_sigma_keV` and `dE_1pct_keV` are the model line's change when the rms radius moves by its
uncertainty and when it is multiplied by 1.01 (the `r101` runs, made for the compared nuclides
only).
A row is size-dominated when the larger of those changes is at least a third of its tolerance,
and weakly sensitive otherwise; agreement on a size-dominated row is a consistency check only,
because the radius passed may itself come from muonic X-ray data (the column `radius_origin`).
`centroid` marks a Table IIIA center of gravity, and `hyperfine` a Table IIIB hyperfine component
or a line Saito et al. describe as showing hyperfine splitting.
Neither is compared in this section, whose model is the orbit-resolved line: the first because the
explanation of Table IIIA names such a value the center of gravity of the 2p → 1s transition and
states no weighting for it, the second because the keywords MuDirac documents include none for a
hyperfine or quadrupole interaction.
Section 6 screens the `centroid` rows separately, against the energy a patched cascade emits, and
that screen qualifies no accuracy.
Of the 67 gated rows, 30 lie within tolerance and 37 outside it; 16 of the gated
rows are weakly sensitive, and 13 of those lie within tolerance.
On 34 of the 34 gated rows that carry an `NPol`, its magnitude exceeds the row's tolerance.
For context only, and never compared, `geant4_keV` is the energy the cascade compiled into Geant4
v11.4.2 and v11.5.0.beta, unpatched, emits between the line's lower and upper shell, a single energy per shell
whatever the orbit (`geant4_cascade_levels.csv`), and `geant4_residual_keV`, the table's Geant4
residual, is that energy minus the measured value.
Every gated row is listed below; the table is generated from `validation.csv`.

| source | Z | A | transition | line | measured (keV) | sigma (keV) | model (keV) | residual (keV) | Geant4 residual (keV) | tol (keV) | label | within | NPol (keV) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fricke1995 | 21 | 45 | 2p1/2-1s1/2 | `K1-L2` | 855.185 | 0.041 | 855.179074376356904 | -0.005925623643096 | -18.432908369 | 0.123 | size-dominated | true | 0.182 |
| Fricke1995 | 21 | 45 | 2p3/2-1s1/2 | `K1-L3` | 857.005 | 0.041 | 856.974807946785237 | -0.030192053214763 | -20.252908369 | 0.123 | size-dominated | true | 0.203 |
| Fricke1995 | 24 | 52 | 2p3/2-1s1/2 | `K1-L3` | 1092.286 | 0.021 | 1092.019374653749866 | -0.266625346250134 | -24.338419151 | 0.063 | size-dominated | false | 0.299 |
| Fricke1995 | 26 | 56 | 2p1/2-1s1/2 | `K1-L2` | 1252.919 | 0.058 | 1252.772586157402024 | -0.146413842597976 | -19.193228401 | 0.174 | size-dominated | true | 0.403 |
| Fricke1995 | 26 | 56 | 2p3/2-1s1/2 | `K1-L3` | 1257.047 | 0.044 | 1256.919597153596813 | -0.127402846403187 | -23.321228401 | 0.132 | size-dominated | true | 0.403 |
| Fricke1995 | 29 | 63 | 2p1/2-1s1/2 | `K1-L2` | 1508.052 | 0.060 | 1507.991624567749444 | -0.060375432250556 | -17.221219739 | 0.180 | size-dominated | true | 0.467 |
| Fricke1995 | 29 | 63 | 2p3/2-1s1/2 | `K1-L3` | 1514.452 | 0.047 | 1514.3109769310134 | -0.1410230689866 | -23.621219739 | 0.141 | size-dominated | false | 0.538 |
| Fricke1995 | 32 | 74 | 2p1/2-1s1/2 | `K1-L2` | 1765.729 | 0.026 | 1765.098639619429829 | -0.630360380570171 | -9.510282508 | 0.078 | size-dominated | false | 0.836 |
| Fricke1995 | 32 | 74 | 2p3/2-1s1/2 | `K1-L3` | 1774.881 | 0.017 | 1774.265416417552857 | -0.615583582447143 | -18.662282508 | 0.051 | size-dominated | false | 0.839 |
| Fricke1995 | 38 | 88 | 2p1/2-1s1/2 | `K1-L2` | 2324.673 | 0.010 | 2324.41085138881579 | -0.26214861118421 | -15.514085307 | 0.030 | size-dominated | false | 0.929 |
| Fricke1995 | 38 | 88 | 2p3/2-1s1/2 | `K1-L3` | 2342.192 | 0.008 | 2341.878577542542946 | -0.313422457457054 | -33.033085307 | 0.024 | size-dominated | false | 0.937 |
| Fricke1995 | 40 | 90 | 2p1/2-1s1/2 | `K1-L2` | 2515.368 | 0.011 | 2515.408810876565054 | 0.040810876565054 | -11.769812304 | 0.033 | size-dominated | false | 0.968 |
| Fricke1995 | 40 | 90 | 2p3/2-1s1/2 | `K1-L3` | 2536.500 | 0.010 | 2536.502239599985536 | 0.002239599985536 | -32.901812304 | 0.030 | size-dominated | true | 0.975 |
| Fricke1995 | 40 | 90 | 2p1/2-1s1/2 | `K1-L2` | 2515.122 | 0.023 | 2515.408810876565054 | 0.286810876565054 | -11.523812304 | 0.069 | size-dominated | false | 1.083 |
| Fricke1995 | 40 | 90 | 2p3/2-1s1/2 | `K1-L3` | 2536.237 | 0.022 | 2536.502239599985536 | 0.265239599985536 | -32.638812304 | 0.066 | size-dominated | false | 0.964 |
| Fricke1995 | 41 | 93 | 2p1/2-1s1/2 | `K1-L2` | 2603.418 | 0.020 | 2603.334445917394012 | -0.083554082605988 | -4.740095737 | 0.060 | size-dominated | false | 0.991 |
| Fricke1995 | 41 | 93 | 2p3/2-1s1/2 | `K1-L3` | 2626.680 | 0.016 | 2626.353202387153637 | -0.326797612846363 | -28.002095737 | 0.048 | size-dominated | false | 1.060 |
| Fricke1995 | 44 | 102 | 2p1/2-1s1/2 | `K1-L2` | 2864.404 | 0.035 | 2863.787876826139633 | -0.616123173860367 | 13.078024463 | 0.105 | size-dominated | false | 1.547 |
| Fricke1995 | 44 | 102 | 2p3/2-1s1/2 | `K1-L3` | 2893.906 | 0.029 | 2893.20611692710733 | -0.69988307289267 | -16.423975537 | 0.087 | size-dominated | false | 1.557 |
| Fricke1995 | 47 | 107 | 2p1/2-1s1/2 | `K1-L2` | 3147.135 | 0.028 | 3146.064993581538554 | -1.070006418461446 | 4.277529299 | 0.084 | size-dominated | false | 1.487 |
| Fricke1995 | 47 | 107 | 2p3/2-1s1/2 | `K1-L3` | 3184.302 | 0.021 | 3183.144663038369268 | -1.157336961630732 | -32.889470701 | 0.063 | size-dominated | false | 1.485 |
| Fricke1995 | 49 | 115 | 2p1/2-1s1/2 | `K1-L2` | 3322.991 | 0.032 | 3322.770628004561644 | -0.220371995438356 | 6.741739173 | 0.096 | size-dominated | false | 0.915 |
| Fricke1995 | 49 | 115 | 2p3/2-1s1/2 | `K1-L3` | 3366.759 | 0.021 | 3365.435484333898872 | -1.323515666101128 | -37.026260827 | 0.063 | size-dominated | false | 1.933 |
| Fricke1995 | 53 | 127 | 2p1/2-1s1/2 | `K1-L2` | 3667.361 | 0.035 | 3667.875272686110809 | 0.514272686110809 | 6.641649588 | 0.105 | size-dominated | false | 0.532 |
| Fricke1995 | 53 | 127 | 2p3/2-1s1/2 | `K1-L3` | 3723.742 | 0.033 | 3723.022797380720731 | -0.719202619279269 | -49.739350412 | 0.099 | size-dominated | false | 1.454 |
| Fricke1995 | 55 | 133 | 2p1/2-1s1/2 | `K1-L2` | 3840.702 | 0.039 | 3840.596965055105742 | -0.105034944894258 | 0.499953431 | 0.117 | size-dominated | true | 1.531 |
| Fricke1995 | 55 | 133 | 2p3/2-1s1/2 | `K1-L3` | 3902.636 | 0.031 | 3902.706495380249806 | 0.070495380249806 | -61.434046569 | 0.093 | size-dominated | true | 1.289 |
| Fricke1995 | 60 | 142 | 2p3/2-1s1/2 | `K1-L3` | 4352.354 | 0.051 | 4352.649078586502001 | 0.295078586502001 | -88.176096506 | 0.153 | size-dominated | false | 1.957 |
| Fricke1995 | 79 | 197 | 2p1/2-1s1/2 | `K1-L2` | 5591.710 | 0.146 | 5594.145673640200868 | 2.435673640200868 | -73.765993793 | 0.438 | size-dominated | false | -0.538 |
| Fricke1995 | 79 | 197 | 2p3/2-1s1/2 | `K1-L3` | 5760.792 | 0.153 | 5763.507347457136028 | 2.715347457136028 | -242.847993793 | 0.459 | size-dominated | false | -1.305 |
| Fricke1995 | 81 | 205 | 2p1/2-1s1/2 | `K1-L2` | 5717.210 | 0.650 | 5722.24183306656871 | 5.03183306656871 | -105.121399721 | 1.950 | size-dominated | false | 3.737 |
| Fricke1995 | 81 | 205 | 2p3/2-1s1/2 | `K1-L3` | 5897.290 | 0.670 | 5902.032599231299013 | 4.742599231299013 | -285.201399721 | 2.010 | size-dominated | false | 3.737 |
| Fricke1995 | 82 | 208 | 2p1/2-1s1/2 | `K1-L2` | 5778.058 | 0.100 | 5779.644122001012787 | 1.586122001012787 | -104.177130102 | 0.300 | size-dominated | false | 2.945 |
| Fricke1995 | 82 | 208 | 2p3/2-1s1/2 | `K1-L3` | 5962.854 | 0.090 | 5964.415968516997062 | 1.561968516997062 | -288.973130102 | 0.270 | size-dominated | false | 2.718 |
| Saito2025 | 46 | 104 | 2p1/2-1s1/2 | `K1-L2` | 3057.8 | 0.3 | 3057.390292917396408 | -0.409707082603592 | 3.049569046 | 0.9 | size-dominated | true |  |
| Saito2025 | 46 | 104 | 2p3/2-1s1/2 | `K1-L3` | 3091.7 | 0.2 | 3091.851641079753637 | 0.151641079753637 | -30.850430954 | 0.6 | size-dominated | true |  |
| Saito2025 | 46 | 104 | 3d5/2-2p3/2 | `L3-M5` | 833.4 | 0.1 | 832.839352577401791 | -0.560647422598209 | -7.871536192 | 0.3 | size-dominated | false |  |
| Saito2025 | 46 | 104 | 3d3/2-2p1/2 | `L2-M4` | 863.6 | 0.3 | 863.045910569646512 | -0.554089430353488 | -38.071536192 | 0.9 | size-dominated | true |  |
| Saito2025 | 46 | 104 | 4d5/2-2p3/2 | `L3-N5` | 1122.5 | 0.6 | 1123.660132803065935 | 1.160132803065935 | -8.036573859 | 1.8 | weakly sensitive | true |  |
| Saito2025 | 46 | 104 | 4f7/2-3d5/2 | `M5-N7` | 291.6 | 0.1 | 291.859981443115743 | 0.259981443115743 | -2.665037667 | 0.3 | weakly sensitive | true |  |
| Saito2025 | 46 | 104 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.2 | 295.227212633686257 | 0.327212633686257 | -5.965037667 | 0.6 | weakly sensitive | true |  |
| Saito2025 | 46 | 105 | 2p1/2-1s1/2 | `K1-L2` | 3054.9 | 0.5 | 3055.072524188620038 | 0.172524188620038 | 5.934098909 | 1.5 | size-dominated | true |  |
| Saito2025 | 46 | 105 | 3d5/2-2p3/2 | `L3-M5` | 832.3 | 0.3 | 832.810797961937613 | 0.510797961937613 | -6.762941671 | 0.9 | weakly sensitive | true |  |
| Saito2025 | 46 | 105 | 3d3/2-2p1/2 | `L2-M4` | 864.5 | 0.3 | 862.991720146174077 | -1.508279853825923 | -38.962941671 | 0.9 | size-dominated | false |  |
| Saito2025 | 46 | 105 | 4f7/2-3d5/2 | `M5-N7` | 291.4 | 0.2 | 291.862972288814606 | 0.462972288814606 | -2.462029585 | 0.6 | weakly sensitive | true |  |
| Saito2025 | 46 | 105 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.6 | 295.230155805940507 | 0.330155805940507 | -5.962029585 | 1.8 | weakly sensitive | true |  |
| Saito2025 | 46 | 106 | 2p1/2-1s1/2 | `K1-L2` | 3050.3 | 0.5 | 3049.650495776996948 | -0.649504223003052 | 10.518960758 | 1.5 | size-dominated | true |  |
| Saito2025 | 46 | 106 | 2p3/2-1s1/2 | `K1-L3` | 3084.0 | 0.3 | 3084.025997042023111 | 0.025997042023111 | -23.181039242 | 0.9 | size-dominated | true |  |
| Saito2025 | 46 | 106 | 3d5/2-2p3/2 | `L3-M5` | 833.4 | 0.3 | 832.732278216345119 | -0.667721783654881 | -7.854531587 | 0.9 | weakly sensitive | true |  |
| Saito2025 | 46 | 106 | 3d3/2-2p1/2 | `L2-M4` | 863.8 | 0.3 | 862.853178162999335 | -0.946821837000665 | -38.254531587 | 0.9 | size-dominated | false |  |
| Saito2025 | 46 | 106 | 4d5/2-2p3/2 | `L3-N5` | 1124.8 | 0.4 | 1123.559003606008366 | -1.240996393991634 | -10.313617643 | 1.2 | weakly sensitive | false |  |
| Saito2025 | 46 | 106 | 4f7/2-3d5/2 | `M5-N7` | 291.7 | 0.2 | 291.865821233231865 | 0.165821233231865 | -2.759086056 | 0.6 | weakly sensitive | true |  |
| Saito2025 | 46 | 106 | 4f5/2-3d3/2 | `M4-N6` | 295.1 | 0.3 | 295.232845260959468 | 0.132845260959468 | -6.159086056 | 0.9 | weakly sensitive | true |  |
| Saito2025 | 46 | 108 | 2p1/2-1s1/2 | `K1-L2` | 3042.9 | 0.2 | 3041.774121005286928 | -1.125878994713072 | 17.889471981 | 0.6 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 2p3/2-1s1/2 | `K1-L3` | 3076.6 | 0.2 | 3076.061951895696111 | -0.538048104303889 | -15.810528019 | 0.6 | size-dominated | true |  |
| Saito2025 | 46 | 108 | 3d5/2-2p3/2 | `L3-M5` | 833.3 | 0.1 | 832.620682761723176 | -0.679317238276824 | -7.738148934 | 0.3 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 3d3/2-2p1/2 | `L2-M4` | 863.5 | 0.1 | 862.654113812212367 | -0.845886187787633 | -37.938148934 | 0.3 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 4d5/2-2p3/2 | `L3-N5` | 1124.4 | 0.3 | 1123.453130366751924 | -0.946869633248076 | -9.891501060 | 0.9 | weakly sensitive | false |  |
| Saito2025 | 46 | 108 | 4f7/2-3d5/2 | `M5-N7` | 291.6 | 0.1 | 291.871431552700582 | 0.271431552700582 | -2.653352127 | 0.3 | weakly sensitive | true |  |
| Saito2025 | 46 | 108 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.1 | 295.238236475085258 | 0.338236475085258 | -5.953352127 | 0.3 | weakly sensitive | false |  |
| Saito2025 | 46 | 110 | 2p1/2-1s1/2 | `K1-L2` | 3036.2 | 0.7 | 3034.758523361260071 | -1.441476638739929 | 24.561043691 | 2.1 | size-dominated | true |  |
| Saito2025 | 46 | 110 | 2p3/2-1s1/2 | `K1-L3` | 3068.8 | 0.4 | 3068.968004062383436 | 0.168004062383436 | -8.038956309 | 1.2 | size-dominated | true |  |
| Saito2025 | 46 | 110 | 3d5/2-2p3/2 | `L3-M5` | 833.4 | 0.2 | 832.520916676175664 | -0.879083323824336 | -7.822355439 | 0.6 | size-dominated | false |  |
| Saito2025 | 46 | 110 | 3d3/2-2p1/2 | `L2-M4` | 863.4 | 0.3 | 862.476174952254863 | -0.923825047745137 | -37.822355439 | 0.9 | size-dominated | false |  |
| Saito2025 | 46 | 110 | 4d5/2-2p3/2 | `L3-N5` | 1124.2 | 0.5 | 1123.358883610287216 | -0.841116389712784 | -9.670179843 | 1.5 | weakly sensitive | true |  |
| Saito2025 | 46 | 110 | 4f7/2-3d5/2 | `M5-N7` | 291.6 | 0.1 | 291.876849593466031 | 0.276849593466031 | -2.647824404 | 0.3 | weakly sensitive | true |  |
| Saito2025 | 46 | 110 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.2 | 295.24346010610275 | 0.34346010610275 | -5.947824404 | 0.6 | weakly sensitive | true |  |


## 5. The numerical checks

Two checks bound how closely the runs solve the equations their keywords state; neither says how close those
equations are to the atom, which section 4 leaves open.
The first repeats each member's anchor run with one numerical setting one step finer at a time: the grid step
halved (`k4`), the energy tolerance tightened tenfold (`tol8`), the Uehling steps raised from 60 to 90 (`nu90`)
and the shell nodes from 12 to 16 (`ns16`); the twin without the Uehling potential is repeated with the first two.
A value is `ONE_STEP_STABLE` when each change lies within a tenth of the value's resolution target, a hundredth
for the energy tolerance (`openmucf/g4/d3_contract.py`): for a printed line, a tenth of the smallest printed
uncertainty among the nuclide's compared cells; for `K`, `e<n>` or a shell difference, the smaller of that and
the larger of 0.000001 keV and 0.000001 times the value.
It is a check of one step, not an error bound on the table: what one step finer changes bounds nothing about
the steps beyond it.
All 7136 values with the Uehling potential, their 7136 twins without it, and all 67 gated solver lines are
`ONE_STEP_STABLE`; the largest change is 0.656 of its limit, that of the grid step on `K` of Sr-88.
The second sets values of the production anchor runs beside the values an independent radial Dirac solver
computed for the same equations (`numerical_check.csv`); that solver is to be released, and the check cannot be
re-run from this repository today.
The file carries 5900 rows over 59 nuclides: 20 with which the patched recipe was developed and checked (sets
`pilot_D`, `pilot_RA` and `pilot_RB`) and a sample of 39 (`audit`) that holds each compared nuclide the
pilot sets lack and one member for each Z above 82, the highest Z of the compared nuclides, that the input
set carries.
Each row gives the production value `Q_eV`, which the loader holds to the committed run at the decimals it
prints, the independent value `Q_ref_eV`, the bound `b_num_eV` on the production value's numerical error, the
larger of two estimates, and the row's target.
Every row is `QUALIFIED`, with `b_num_eV` at most 0.194 of the target; the largest difference between a
production value and the independent one is 0.429360 eV, on the K level of U-238 with the Uehling potential,
against a target of 12.13 eV.
The first-order Uehling expectation, `bue_1_eV`, comes with the independent solution and is carried only on
its rows with the Uehling potential; the Uehling contribution under other nuclear charge shapes is not computed, because no table and
no check reads it and alone it would leave out the larger sensitivity of the Coulomb potential to the shape
(the header of `components.jsonl`, `not_computed`).

## 6. The energy the cascade receives

The comparison above scores the orbit-resolved line MuDirac prints; a patched cascade receives, for
each shell above K, the degeneracy-weighted mean of its two circular states, so the energy it emits
between two shells is one number even where the sources print two.
`shell_projection.csv` joins each gated row to that shell difference (`consumer_quantity`), beside
the solver line and the energy the unpatched cascade emits, and never relabels a measured line as a
centroid.
Of the 67 gated rows, the shell difference lies within the band for 1, the solver line for 30,
the unpatched cascade for 0, and the shell difference is closer than the unpatched cascade for 42.
`incompatible_groups.csv` takes every (source, nuclide, shell pair) with more than one distinct gated line and
intersects the lines' bands: 29 of the 29 such groups have an empty intersection, the widest,
Pb-208 K–L, by 184.226 keV, so no single shell energy can lie within the band of every line it stands
for.
Every such group is listed below; the table is generated from `incompatible_groups.csv`.

| source | Z | A | initial n | final n | lines | quantities | lower (keV) | upper (keV) | gap (keV) | empty |
|---|---|---|---|---|---|---|---|---|---|---|
| Fricke1995 | 21 | 45 | 2 | 1 | 2 | `K1-L2;K1-L3` | 856.882 | 855.308 | 1.574 | true |
| Fricke1995 | 26 | 56 | 2 | 1 | 2 | `K1-L2;K1-L3` | 1256.915 | 1253.093 | 3.822 | true |
| Fricke1995 | 29 | 63 | 2 | 1 | 2 | `K1-L2;K1-L3` | 1514.311 | 1508.232 | 6.079 | true |
| Fricke1995 | 32 | 74 | 2 | 1 | 2 | `K1-L2;K1-L3` | 1774.83 | 1765.807 | 9.023 | true |
| Fricke1995 | 38 | 88 | 2 | 1 | 2 | `K1-L2;K1-L3` | 2342.168 | 2324.703 | 17.465 | true |
| Fricke1995 | 40 | 90 | 2 | 1 | 4 | `K1-L2;K1-L3;K1-L2;K1-L3` | 2536.47 | 2515.191 | 21.279 | true |
| Fricke1995 | 41 | 93 | 2 | 1 | 2 | `K1-L2;K1-L3` | 2626.632 | 2603.478 | 23.154 | true |
| Fricke1995 | 44 | 102 | 2 | 1 | 2 | `K1-L2;K1-L3` | 2893.819 | 2864.509 | 29.31 | true |
| Fricke1995 | 47 | 107 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3184.239 | 3147.219 | 37.02 | true |
| Fricke1995 | 49 | 115 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3366.696 | 3323.087 | 43.609 | true |
| Fricke1995 | 53 | 127 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3723.643 | 3667.466 | 56.177 | true |
| Fricke1995 | 55 | 133 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3902.543 | 3840.819 | 61.724 | true |
| Fricke1995 | 79 | 197 | 2 | 1 | 2 | `K1-L2;K1-L3` | 5760.333 | 5592.148 | 168.185 | true |
| Fricke1995 | 81 | 205 | 2 | 1 | 2 | `K1-L2;K1-L3` | 5895.28 | 5719.16 | 176.12 | true |
| Fricke1995 | 82 | 208 | 2 | 1 | 2 | `K1-L2;K1-L3` | 5962.584 | 5778.358 | 184.226 | true |
| Saito2025 | 46 | 104 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3091.1 | 3058.7 | 32.4 | true |
| Saito2025 | 46 | 104 | 3 | 2 | 2 | `L3-M5;L2-M4` | 862.7 | 833.7 | 29 | true |
| Saito2025 | 46 | 104 | 4 | 3 | 2 | `M5-N7;M4-N6` | 294.3 | 291.9 | 2.4 | true |
| Saito2025 | 46 | 105 | 3 | 2 | 2 | `L3-M5;L2-M4` | 863.6 | 833.2 | 30.4 | true |
| Saito2025 | 46 | 105 | 4 | 3 | 2 | `M5-N7;M4-N6` | 293.1 | 292 | 1.1 | true |
| Saito2025 | 46 | 106 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3083.1 | 3051.8 | 31.3 | true |
| Saito2025 | 46 | 106 | 3 | 2 | 2 | `L3-M5;L2-M4` | 862.9 | 834.3 | 28.6 | true |
| Saito2025 | 46 | 106 | 4 | 3 | 2 | `M5-N7;M4-N6` | 294.2 | 292.3 | 1.9 | true |
| Saito2025 | 46 | 108 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3076 | 3043.5 | 32.5 | true |
| Saito2025 | 46 | 108 | 3 | 2 | 2 | `L3-M5;L2-M4` | 863.2 | 833.6 | 29.6 | true |
| Saito2025 | 46 | 108 | 4 | 3 | 2 | `M5-N7;M4-N6` | 294.6 | 291.9 | 2.7 | true |
| Saito2025 | 46 | 110 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3067.6 | 3038.3 | 29.3 | true |
| Saito2025 | 46 | 110 | 3 | 2 | 2 | `L3-M5;L2-M4` | 862.5 | 834 | 28.5 | true |
| Saito2025 | 46 | 110 | 4 | 3 | 2 | `M5-N7;M4-N6` | 294.3 | 291.9 | 2.4 | true |


`unc` and `u<n>` are the magnitude of one signed radius response with a floor, not a predictive
uncertainty; `components.jsonl` carries, for every value, the signed response in both directions (the
`rminus` run is made where the lowered rms radius keeps the 2pF density), the rounding bounds of the
anchor and of the summed lines, `bue_np_keV`, the whole Uehling contribution between the anchor run and
its twin, the one-step changes with their verdict and, where `numerical_check.csv` carries the value, the
independent check (section 5), and no total uncertainty.
Every value is solved with the Uehling potential in its Hamiltonian, to all orders, so no further Uehling term
is to be added to it (the header's `assembly_rule`).
`radius_lineage.csv` records, per compared nuclide, what is known of the experiments behind the
radius passed to MuDirac and behind the compared line; its state is `UNKNOWN` on every row, because
the evaluation the radii come from is not in hand.

### The K–L shell difference beside a single measured value

This part follows a rule fixed after the residuals above had been published and after this shell
difference had already been set beside degeneracy-convention centroids and beside the unpatched
cascade, so it is a retrospective analysis and qualifies no accuracy.
`consumer_centroids.csv` sets the energy a patched cascade emits between the K and L shells, the K
energy minus the L-shell mean of these tables, beside a single value for a nuclide's 2p → 1s
transition and beside the energy the unpatched cascade emits between the same shells, for which
`geant4_cascade_levels.csv` is now also cut to the nuclides of the `centroid` rows.
For a `centroid` row that value is what the explanation of Table IIIA of Fricke et al. (1995) calls
"the center of gravity of the 2p → 1s transition", for which it states no weighting, with its
printed statistical uncertainty.
Where both lines of a 2p → 1s doublet are gated rows above, the value is their degeneracy-convention
centroid, the lines weighted by the degeneracies of their upper states as the L-shell mean of these
tables is, a convention that implies nothing about the lines' relative intensities; a nuclide with a
gated line whose partner is missing or ungated is listed as `excluded`.
`validation_cells.csv` carries no correlation between the lines, so that centroid's uncertainty is
given as its smallest and largest values over every correlation, and each row is screened against
the band of the comparison above as `inside`, `outside` or, where the bounds disagree,
`correlation-dependent`, a label of a discrepancy and not of accuracy.
Fricke et al. (1995) title their Table IIIA "Muonic 2p → 1s Transition Energies" and call each
value in its column `E_exp` an "Experimental energy"; neither the explanation of that table nor their section 4
states whether the photon recoil has been taken out of a value, and the screen subtracts no recoil
from either side.
Each row also carries `b_num_keV`, the bound `numerical_check.csv` gives on the numerical error of the
nuclide's K–L shell difference with the Uehling potential, and its verdict, never added to an uncertainty.
That bound exceeds a tenth of the row's uncertainty, taken at its largest over every correlation, on 0
of the 27 rows.
`centroid_margins.csv` gives, per cohort and source, the smallest, median and largest margin, the
unpatched cascade's absolute residual minus the patched one's, over the isotopes, each counted once,
and the patched cascade's largest absolute residual with its nuclide, as a distribution only; both
files are tabulated below.
For Pb-208 the row also lists the ratio of the doublet's intensities that Jenkins et al. (1971)
measured, from their Table 9 with the target of their Table 3, and the centroid that ratio would
give, for illustration only and never screened.
Neither the nuclear-polarization corrections Fricke et al. print, listed beside each of their rows,
nor electron screening enters the shell difference, a centroid or a band.

| cohort | source | Z | A | value (keV) | sigma (keV) | sigma min (keV) | sigma max (keV) | shell difference (keV) | residual (keV) | unpatched residual (keV) | margin (keV) | screen | b_num (keV) | numerical check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| labelled | Fricke1995 | 4 | 9 | 33.402000000 | 0.010000000 |  |  | 33.392643350 | -0.009356650 | -1.510349355 | 1.500992705 | inside | 0.000000051 | QUALIFIED |
| labelled | Fricke1995 | 6 | 12 | 75.258200000 | 0.000500000 |  |  | 75.261104401 | 0.002904401 | -2.329588794 | 2.326684393 | outside | 0.000000465 | QUALIFIED |
| labelled | Fricke1995 | 8 | 16 | 133.535000000 | 0.002000000 |  |  | 133.532419742 | -0.002580258 | -5.210871570 | 5.208291312 | inside | 0.000001807 | QUALIFIED |
| labelled | Fricke1995 | 11 | 23 | 250.229000000 | 0.002000000 |  |  | 250.223935982 | -0.005064018 | -8.875728079 | 8.870664061 | inside | 0.000000974 | QUALIFIED |
| labelled | Fricke1995 | 13 | 27 | 346.828000000 | 0.002000000 |  |  | 346.835434161 | 0.007434161 | -12.154564424 | 12.147130263 | outside | 0.000001826 | QUALIFIED |
| labelled | Fricke1995 | 14 | 28 | 400.173000000 | 0.005000000 |  |  | 400.166459985 | -0.006540015 | -13.406839395 | 13.400299380 | inside | 0.000002439 | QUALIFIED |
| labelled | Fricke1995 | 18 | 40 | 644.004000000 | 0.025000000 |  |  | 643.631887251 | -0.372112749 | -18.135103427 | 17.762990678 | outside | 0.000006502 | QUALIFIED |
| constructed | Fricke1995 | 21 | 45 | 856.398333333 |  | 0.013666667 | 0.041000000 | 856.376230090 | -0.022103243 | -19.646241702 | 19.624138459 | inside | 0.000011120 | QUALIFIED |
| constructed | Fricke1995 | 26 | 56 | 1255.671000000 |  | 0.010000000 | 0.048666667 | 1255.537260155 | -0.133739845 | -21.945228401 | 21.811488556 | correlation-dependent | 0.000022530 | QUALIFIED |
| constructed | Fricke1995 | 29 | 63 | 1512.318666667 |  | 0.011333333 | 0.051333333 | 1512.204526143 | -0.114140523 | -21.487886406 | 21.373745882 | correlation-dependent | 0.000031860 | QUALIFIED |
| constructed | Fricke1995 | 32 | 74 | 1771.830333333 |  | 0.002666667 | 0.020000000 | 1771.209824152 | -0.620509182 | -15.611615841 | 14.991106660 | outside | 0.000043310 | QUALIFIED |
| constructed | Fricke1995 | 38 | 88 | 2336.352333333 |  | 0.002000000 | 0.008666667 | 2336.056002158 | -0.296331175 | -27.193418640 | 26.897087465 | outside | 0.000068970 | QUALIFIED |
| constructed | Fricke1995 | 40 | 90 | 2529.456000000 |  | 0.003000000 | 0.010333333 | 2529.471096692 | 0.015096692 | -25.857812304 | 25.842715612 | correlation-dependent | 0.000078590 | QUALIFIED |
| constructed | Fricke1995 | 40 | 90 | 2529.198666667 |  | 0.007000000 | 0.022333333 | 2529.471096692 | 0.272430026 | -25.600478971 | 25.328048945 | outside | 0.000078590 | QUALIFIED |
| constructed | Fricke1995 | 41 | 93 | 2618.926000000 |  | 0.004000000 | 0.017333333 | 2618.680283564 | -0.245716436 | -20.248095737 | 20.002379301 | outside | 0.000083830 | QUALIFIED |
| constructed | Fricke1995 | 44 | 102 | 2884.072000000 |  | 0.007666667 | 0.031000000 | 2883.400036893 | -0.671963107 | -6.589975537 | 5.918012430 | outside | 0.000100000 | QUALIFIED |
| constructed | Fricke1995 | 47 | 107 | 3171.913000000 |  | 0.004666667 | 0.023333333 | 3170.784773219 | -1.128226781 | -20.500470701 | 19.372243920 | outside | 0.000116200 | QUALIFIED |
| constructed | Fricke1995 | 49 | 115 | 3352.169666667 |  | 0.003333333 | 0.024666667 | 3351.213865557 | -0.955801109 | -22.436927494 | 21.481126384 | outside | 0.000127500 | QUALIFIED |
| constructed | Fricke1995 | 53 | 127 | 3704.948333333 |  | 0.010333333 | 0.033666667 | 3704.640289149 | -0.308044184 | -30.945683745 | 30.637639561 | outside | 0.000150200 | QUALIFIED |
| constructed | Fricke1995 | 55 | 133 | 3881.991333333 |  | 0.007666667 | 0.033666667 | 3882.003318605 | 0.011985272 | -40.789379902 | 40.777394630 | inside | 0.000161500 | QUALIFIED |
| constructed | Fricke1995 | 79 | 197 | 5704.431333333 |  | 0.053333333 | 0.150666667 | 5707.053456185 | 2.622122851 | -186.487327126 | 183.865204275 | outside | 0.000278300 | QUALIFIED |
| constructed | Fricke1995 | 81 | 205 | 5837.263333333 |  | 0.230000000 | 0.663333333 | 5842.102343843 | 4.839010510 | -225.174733054 | 220.335722545 | outside | 0.000285700 | QUALIFIED |
| constructed | Fricke1995 | 82 | 208 | 5901.255333333 |  | 0.026666667 | 0.093333333 | 5902.825353012 | 1.570019678 | -227.374463435 | 225.804443757 | outside | 0.000289000 | QUALIFIED |
| constructed | Saito2025 | 46 | 104 | 3080.400000000 |  | 0.033333333 | 0.233333333 | 3080.364525026 | -0.035474974 | -19.550430954 | 19.514955980 | inside | 0.000110700 | QUALIFIED |
| constructed | Saito2025 | 46 | 106 | 3072.766666667 |  | 0.033333333 | 0.366666667 | 3072.567496620 | -0.199170046 | -11.947705909 | 11.748535862 | correlation-dependent | 0.000110800 | QUALIFIED |
| constructed | Saito2025 | 46 | 108 | 3065.366666667 |  | 0.066666667 | 0.200000000 | 3064.632674932 | -0.733991734 | -4.577194686 | 3.843202951 | outside | 0.000110900 | QUALIFIED |
| constructed | Saito2025 | 46 | 110 | 3057.933333333 |  | 0.033333333 | 0.500000000 | 3057.564843829 | -0.368489505 | 2.827710358 | 2.459220853 | correlation-dependent | 0.000111100 | QUALIFIED |


| cohort | source | isotopes | margin min (keV) | margin median (keV) | margin max (keV) | largest residual (keV) | Z | A |
|---|---|---|---|---|---|---|---|---|
| labelled | Fricke1995 | 7 | 1.500992705 | 8.870664061 | 17.762990678 | 0.372112749 | 18 | 40 |
| constructed | Fricke1995 | 15 | 5.918012430 | 21.811488556 | 225.804443757 | 4.839010510 | 81 | 205 |
| constructed | Saito2025 | 4 | 2.459220853 | 3.843202951..11.748535862 | 19.514955980 | 0.733991734 | 46 | 108 |


As part of this retrospective analysis, and qualifying no accuracy, the margins over the 26 isotopes of `centroid_margins.csv` run from 1.500992705 keV to 225.804443757 keV, and the patched cascade's largest absolute residual is 4.839010510 keV, for Tl-205, where the unpatched cascade's reaches 227.374463435 keV, for Pb-208.
Moving the value anywhere within its band changes a margin by at most twice the tolerance, and moving the shell difference by `b_num_keV` changes it by at most that bound; with the tolerance taken at its largest over every correlation, the margin exceeds the sum of both on 26 of the 27 rows, and not on the row for Pd-110.

## 7. Findings

- The rows outside tolerance are this comparison's registered disagreements; none is fitted away.
- A single energy per shell pair cannot stand for the lines between the two shells (section 6): the shell
  difference lies within fewer bands than the solver line does, and no shell energy lies within the band of
  every line of a group. This failure of the cascade's representation is kept as a finding, regenerated
  from these tables, and is not repaired by the patch.
- The members whose bundled rms radius differs from the IAEA table are printed by the generator
  and passed as bundled.

## 8. In Geant4

With the opt-in on and the muonic-transition seam reading through `mudirac130p1` — named by
`G4MUONICDATA_D3_PROFILE`, or by `G4MUONICDATA_PROFILE`, which names both seams — a patched
Geant4's muonic cascade
(`G4EmCaptureCascade`) takes the K energy, and the energy of every shell above it up to the chain's
last, from these tables, from the row keyed by exactly the nuclide it is transporting, and keeps
its hydrogen-like formula for the shells above those, and its branching and random draws, unchanged.
`G4MuonicAtomHelper::GetKShellEnergy` reads the element's natural-composition row, which it asks
for by name, and the form of it that also takes the mass number reads the row of that nuclide, for the muonic atom's mass and for the bound
energy its decay passes on; where these tables carry no such row, the compiled-in code runs.
The energy the cascade deposits, which Geant4 passes on as the muon's binding energy to decay in
orbit and to nuclear capture, is the sum of the energies it emitted, and so moves with the table's
K energy.
With the opt-in off, the cascade and K-energy harvests of Geant4 v11.4.2 and v11.5.0.beta are
bit-identical to those of unpatched builds.

By what they represent, the tables suit these uses: the muonic K binding energy, the single energy a patched cascade emits between a pair of shells for all the lines between them, and the energy the cascade deposits; section 6 sets the energy the cascade receives beside measured energies.
That energy rests on shell energies that weight the states of a shell by their degeneracies, a convention that implies nothing about the lines' relative intensities (section 6), so a detector that does not separate those lines does not by that alone make it the centroid of the lines it records, whose weights their intensities set.
They do not suit a spectrum that separates those lines, or the intensities of lines, because the cascade emits a single energy per shell pair and keeps its branching unchanged.
For separated lines, `mudirac_lines.csv` carries the line energies MuDirac printed (section 2).

## 9. Licensing

`data/g4/d3/mudirac_inputs.csv` repeats the nuclear radii MuDirac bundles in `nuclear_radii.dat` and
carries them under the MIT License, whose text `data/g4/d3/LICENSE-MUDIRAC` copies from the release of
MuDirac that section 1 names.
The Layer-2 files of the `k_shell_energy` and `level_energy` tables repeat those radii, as each run's
input radius, under the same licence.
The files that restate cells of the comparison sources' tables, or carry them beside the model's values,
`data/g4/d3/validation_cells.csv`, `data/g4/d3/intensity_ratios.csv`, `data/g4/d3/validation.csv`,
`data/g4/d3/shell_projection.csv` and `data/g4/d3/consumer_centroids.csv`, are offered under no licence
of this repository.
The other files of `data/g4/d3/` are offered under CC-BY-4.0.
The values compiled into Geant4 that `geant4_cascade_levels.csv` repeats are offered under the Geant4
Software License.
