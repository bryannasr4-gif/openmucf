# The D3 energy tables of the `mudirac130` profile

## 1. What ships

`data/g4/d3/` ships the tables `k_shell_energy` (`d3_kshell.mudirac130.g4dat`) and `level_energy`
(`d3_levels.mudirac130.g4dat`) under `#PROFILE mudirac130` and `#SEAM d3_transitions`, each with its
Layer-2 file.
Every `value` and `e<n>` cell is a positive binding energy in keV derived from the output of MuDirac 1.3.0 (Sturniolo and
Hillier (2021); Liborio et al. (2026)), a Dirac-equation solver for muonic atoms released under the
MIT License with the copyright held by the Science and Technology Facility Council, as its
`LICENSE` reads.
`k_shell_energy` carries the binding energy of the `K1` orbit in its `value` column, with the
magnitude of its radius response in `unc`.
`level_energy` carries, for each shell above `K`, the mean binding energy of both circular states
of the shell weighted by their degeneracy, in the columns `e<n>`, with the magnitudes of their
radius responses in `u<n>`.
The chain ends at shell 8, the highest shell through which every circular state converged on
the nuclides checked when the settings were fixed.
Each table has a row for every kept (Z, A) and, for each element whose most abundant isotope in
MuDirac's `abundant.dat` is kept, a natural-composition row that carries that isotope's values
under `#VALIDITY` `A:most_abundant_and_listed`.
Of the 895 members of the input set, 889 are kept, and the tables carry 81
natural-composition rows.

## 2. How the values are made

The input set is every (Z, A) with Z from 1 through 92 that MuDirac's bundled
`nuclear_radii.dat` lists.
Every run uses the same keywords: `nuclear_model: FERMI2`, `uehling_correction: TRUE`,
`reduced_mass: TRUE`, `optimise_fermi_parameters: FALSE`, `fermi_t: 2.3`; no `electronic_config` is given, so the atom
carries no electrons.
MuDirac's keyword documentation attributes its default radii to Angeli and Marinova (2013).
The inputs are committed as `mudirac_inputs.csv`, and what MuDirac printed as `mudirac_runs.csv`,
`mudirac_states.csv`, `mudirac_lines.csv` and `mudirac_nmax_check.csv`; `scripts/mudirac_d3.py`
renders the inputs, runs MuDirac and collects what it prints.
Every input sets `optimise_fermi_parameters: FALSE` and every run is given its input file as its
only argument, so MuDirac's fitting mode, which reads measured energies, never runs.
The tables are rebuilt from the committed CSV files by `scripts/generate_g4data.py` and
byte-compared by its audit; MuDirac itself is not run in CI, and cross-compiler reproducibility of
its output is not claimed.
The tables are derived from the printed line energies by exact decimal sums anchored on the printed
energy of the outermost circular state.
`value` and `e<n>` come from the `base` run; `unc` and `u<n>` are the absolute change of the same
quantity in the `rsig` run, whose rms radius is moved by its uncertainty in the IAEA charge-radii
table (`charge_radii.csv`), measured through the printed line energies with the energy of the
outermost circular state held fixed, and propagate nothing else.
No `unc` or `u<n>` cell is below 0.000000014 keV; a smaller change ships as that floor.
Where the bundled rms radius and that table's value differ at the fourth decimal, the generator
prints the member; neither value is edited.

## 3. Which nuclides are kept

A member is kept only when its `base`, `rsig` and hydrogen-like runs exit cleanly with an empty error
file, when each circular state of shells 6 through 8 lies within 1 % of the
same state in the run that treats the atom as hydrogen-like from that shell up
(`ideal_atom_minshell`), and when the derivation holds on its `base` and `rsig` runs.
The generator drops `He8`, the only member whose sphere radius gives no real value of the
Fermi parameter c that MuDirac computes by default, and prints its radius beside the radius below
which that value is not real.
It also drops every member whose mass number is below the one from which that default depends on
the radius: MuDirac then sets c from the mass number alone.
An element whose most abundant isotope is dropped has no natural-composition row, so a lookup for
it falls through to Geant4's compiled-in code.

## 4. Comparison with measured energies

`validation.csv` compares the model line, the difference between the energies of its initial and
final states as MuDirac prints it, with the energies transcribed from Fricke et al. (1995), Tables
IIIA and IIIB, whose titles call them transition energies, and from Saito et al. (2025), whose
Table III is titled "Observed muonic X-ray energies" and Table IV "Observed muonic X-ray
transition energies", each as printed (`validation_cells.csv`).
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
No value is changed for lying outside the tolerance, and omitted physics is not absorbed into it:
each Table IIIA row carries the nuclear-polarization correction that table prints beside it (`NPol`).
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
Section 5 screens the `centroid` rows separately, against the energy a patched cascade emits, and
that screen qualifies no accuracy.
Of the 67 gated rows, 26 lie within tolerance and 41 outside it; 16 of the gated
rows are weakly sensitive, and 13 of those lie within tolerance.
For context only, and never compared, `geant4_keV` is the energy the cascade compiled into Geant4
v11.4.2 and v11.5.0.beta, unpatched, emits between the line's lower and upper shell, a single energy per shell
whatever the orbit (`geant4_cascade_levels.csv`), and `geant4_residual_keV`, the table's Geant4
residual, is that energy minus the measured value.
Every gated row is listed below; the table is generated from `validation.csv`.

| source | Z | A | transition | line | measured (keV) | sigma (keV) | model (keV) | residual (keV) | Geant4 residual (keV) | tol (keV) | label | within | NPol (keV) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fricke1995 | 21 | 45 | 2p1/2-1s1/2 | `K1-L2` | 855.185 | 0.041 | 855.168632818 | -0.016367182 | -18.432908369 | 0.123 | size-dominated | true | 0.182 |
| Fricke1995 | 21 | 45 | 2p3/2-1s1/2 | `K1-L3` | 857.005 | 0.041 | 856.964379747 | -0.040620253 | -20.252908369 | 0.123 | size-dominated | true | 0.203 |
| Fricke1995 | 24 | 52 | 2p3/2-1s1/2 | `K1-L3` | 1092.286 | 0.021 | 1092.000364461 | -0.285635539 | -24.338419151 | 0.063 | size-dominated | false | 0.299 |
| Fricke1995 | 26 | 56 | 2p1/2-1s1/2 | `K1-L2` | 1252.919 | 0.058 | 1252.746243216 | -0.172756784 | -19.193228401 | 0.174 | size-dominated | true | 0.403 |
| Fricke1995 | 26 | 56 | 2p3/2-1s1/2 | `K1-L3` | 1257.047 | 0.044 | 1256.893284951 | -0.153715049 | -23.321228401 | 0.132 | size-dominated | false | 0.403 |
| Fricke1995 | 29 | 63 | 2p1/2-1s1/2 | `K1-L2` | 1508.052 | 0.060 | 1507.951824562 | -0.100175438 | -17.221219739 | 0.180 | size-dominated | true | 0.467 |
| Fricke1995 | 29 | 63 | 2p3/2-1s1/2 | `K1-L3` | 1514.452 | 0.047 | 1514.271215275 | -0.180784725 | -23.621219739 | 0.141 | size-dominated | false | 0.538 |
| Fricke1995 | 32 | 74 | 2p1/2-1s1/2 | `K1-L2` | 1765.729 | 0.026 | 1765.043144757 | -0.685855243 | -9.510282508 | 0.078 | size-dominated | false | 0.836 |
| Fricke1995 | 32 | 74 | 2p3/2-1s1/2 | `K1-L3` | 1774.881 | 0.017 | 1774.209956155 | -0.671043845 | -18.662282508 | 0.051 | size-dominated | false | 0.839 |
| Fricke1995 | 38 | 88 | 2p1/2-1s1/2 | `K1-L2` | 2324.673 | 0.010 | 2324.311814226 | -0.361185774 | -15.514085307 | 0.030 | size-dominated | false | 0.929 |
| Fricke1995 | 38 | 88 | 2p3/2-1s1/2 | `K1-L3` | 2342.192 | 0.008 | 2341.779467461 | -0.412532539 | -33.033085307 | 0.024 | size-dominated | false | 0.937 |
| Fricke1995 | 40 | 90 | 2p1/2-1s1/2 | `K1-L2` | 2515.368 | 0.011 | 2515.292409825 | -0.075590175 | -11.769812304 | 0.033 | size-dominated | false | 0.968 |
| Fricke1995 | 40 | 90 | 2p3/2-1s1/2 | `K1-L3` | 2536.500 | 0.010 | 2536.385679776 | -0.114320224 | -32.901812304 | 0.030 | size-dominated | false | 0.975 |
| Fricke1995 | 40 | 90 | 2p1/2-1s1/2 | `K1-L2` | 2515.122 | 0.023 | 2515.292409825 | 0.170409825 | -11.523812304 | 0.069 | size-dominated | false | 1.083 |
| Fricke1995 | 40 | 90 | 2p3/2-1s1/2 | `K1-L3` | 2536.237 | 0.022 | 2536.385679776 | 0.148679776 | -32.638812304 | 0.066 | size-dominated | false | 0.964 |
| Fricke1995 | 41 | 93 | 2p1/2-1s1/2 | `K1-L2` | 2603.418 | 0.020 | 2603.210000102 | -0.207999898 | -4.740095737 | 0.060 | size-dominated | false | 0.991 |
| Fricke1995 | 41 | 93 | 2p3/2-1s1/2 | `K1-L3` | 2626.680 | 0.016 | 2626.22854214 | -0.45145786 | -28.002095737 | 0.048 | size-dominated | false | 1.060 |
| Fricke1995 | 44 | 102 | 2p1/2-1s1/2 | `K1-L2` | 2864.404 | 0.035 | 2863.638237678 | -0.765762322 | 13.078024463 | 0.105 | size-dominated | false | 1.547 |
| Fricke1995 | 44 | 102 | 2p3/2-1s1/2 | `K1-L3` | 2893.906 | 0.029 | 2893.056029724 | -0.849970276 | -16.423975537 | 0.087 | size-dominated | false | 1.557 |
| Fricke1995 | 47 | 107 | 2p1/2-1s1/2 | `K1-L2` | 3147.135 | 0.028 | 3145.884473083 | -1.250526917 | 4.277529299 | 0.084 | size-dominated | false | 1.487 |
| Fricke1995 | 47 | 107 | 2p3/2-1s1/2 | `K1-L3` | 3184.302 | 0.021 | 3182.963333628 | -1.338666372 | -32.889470701 | 0.063 | size-dominated | false | 1.485 |
| Fricke1995 | 49 | 115 | 2p1/2-1s1/2 | `K1-L2` | 3322.991 | 0.032 | 3322.569897053 | -0.421102947 | 6.741739173 | 0.096 | size-dominated | false | 0.915 |
| Fricke1995 | 49 | 115 | 2p3/2-1s1/2 | `K1-L3` | 3366.759 | 0.021 | 3365.233620292 | -1.525379708 | -37.026260827 | 0.063 | size-dominated | false | 1.933 |
| Fricke1995 | 53 | 127 | 2p1/2-1s1/2 | `K1-L2` | 3667.361 | 0.035 | 3667.632427035 | 0.271427035 | 6.641649588 | 0.105 | size-dominated | false | 0.532 |
| Fricke1995 | 53 | 127 | 2p3/2-1s1/2 | `K1-L3` | 3723.742 | 0.033 | 3722.777927024 | -0.964072976 | -49.739350412 | 0.099 | size-dominated | false | 1.454 |
| Fricke1995 | 55 | 133 | 2p1/2-1s1/2 | `K1-L2` | 3840.702 | 0.039 | 3840.331520121 | -0.370479879 | 0.499953431 | 0.117 | size-dominated | false | 1.531 |
| Fricke1995 | 55 | 133 | 2p3/2-1s1/2 | `K1-L3` | 3902.636 | 0.031 | 3902.43843788 | -0.19756212 | -61.434046569 | 0.093 | size-dominated | false | 1.289 |
| Fricke1995 | 60 | 142 | 2p3/2-1s1/2 | `K1-L3` | 4352.354 | 0.051 | 4352.31846825 | -0.03553175 | -88.176096506 | 0.153 | size-dominated | true | 1.957 |
| Fricke1995 | 79 | 197 | 2p1/2-1s1/2 | `K1-L2` | 5591.710 | 0.146 | 5593.605121885 | 1.895121885 | -73.765993793 | 0.438 | size-dominated | false | -0.538 |
| Fricke1995 | 79 | 197 | 2p3/2-1s1/2 | `K1-L3` | 5760.792 | 0.153 | 5762.947979365 | 2.155979365 | -242.847993793 | 0.459 | size-dominated | false | -1.305 |
| Fricke1995 | 81 | 205 | 2p1/2-1s1/2 | `K1-L2` | 5717.210 | 0.650 | 5721.677852281 | 4.467852281 | -105.121399721 | 1.950 | size-dominated | false | 3.737 |
| Fricke1995 | 81 | 205 | 2p3/2-1s1/2 | `K1-L3` | 5897.290 | 0.670 | 5901.447639051 | 4.157639051 | -285.201399721 | 2.010 | size-dominated | false | 3.737 |
| Fricke1995 | 82 | 208 | 2p1/2-1s1/2 | `K1-L2` | 5778.058 | 0.100 | 5779.069723885 | 1.011723885 | -104.177130102 | 0.300 | size-dominated | false | 2.945 |
| Fricke1995 | 82 | 208 | 2p3/2-1s1/2 | `K1-L3` | 5962.854 | 0.090 | 5963.819508637 | 0.965508637 | -288.973130102 | 0.270 | size-dominated | false | 2.718 |
| Saito2025 | 46 | 104 | 2p1/2-1s1/2 | `K1-L2` | 3057.8 | 0.3 | 3057.219525585 | -0.580474415 | 3.049569046 | 0.9 | size-dominated | true |  |
| Saito2025 | 46 | 104 | 2p3/2-1s1/2 | `K1-L3` | 3091.7 | 0.2 | 3091.680200251 | -0.019799749 | -30.850430954 | 0.6 | size-dominated | true |  |
| Saito2025 | 46 | 104 | 3d5/2-2p3/2 | `L3-M5` | 833.4 | 0.1 | 832.852073144 | -0.547926856 | -7.871536192 | 0.3 | size-dominated | false |  |
| Saito2025 | 46 | 104 | 3d3/2-2p1/2 | `L2-M4` | 863.6 | 0.3 | 863.057864657 | -0.542135343 | -38.071536192 | 0.9 | size-dominated | true |  |
| Saito2025 | 46 | 104 | 4d5/2-2p3/2 | `L3-N5` | 1122.5 | 0.6 | 1123.67359222 | 1.17359222 | -8.036573859 | 1.8 | weakly sensitive | true |  |
| Saito2025 | 46 | 104 | 4f7/2-3d5/2 | `M5-N7` | 291.6 | 0.1 | 291.86147437 | 0.26147437 | -2.665037667 | 0.3 | weakly sensitive | true |  |
| Saito2025 | 46 | 104 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.2 | 295.228795545 | 0.328795545 | -5.965037667 | 0.6 | weakly sensitive | true |  |
| Saito2025 | 46 | 105 | 2p1/2-1s1/2 | `K1-L2` | 3054.9 | 0.5 | 3054.902109983 | 0.002109983 | 5.934098909 | 1.5 | size-dominated | true |  |
| Saito2025 | 46 | 105 | 3d5/2-2p3/2 | `L3-M5` | 832.3 | 0.3 | 832.823524283 | 0.523524283 | -6.762941671 | 0.9 | weakly sensitive | true |  |
| Saito2025 | 46 | 105 | 3d3/2-2p1/2 | `L2-M4` | 864.5 | 0.3 | 863.003680202 | -1.496319798 | -38.962941671 | 0.9 | size-dominated | false |  |
| Saito2025 | 46 | 105 | 4f7/2-3d5/2 | `M5-N7` | 291.4 | 0.2 | 291.864470739 | 0.464470739 | -2.462029585 | 0.6 | weakly sensitive | true |  |
| Saito2025 | 46 | 105 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.6 | 295.231744311 | 0.331744311 | -5.962029585 | 1.8 | weakly sensitive | true |  |
| Saito2025 | 46 | 106 | 2p1/2-1s1/2 | `K1-L2` | 3050.3 | 0.5 | 3049.480915874 | -0.819084126 | 10.518960758 | 1.5 | size-dominated | true |  |
| Saito2025 | 46 | 106 | 2p3/2-1s1/2 | `K1-L3` | 3084.0 | 0.3 | 3083.8557448 | -0.1442552 | -23.181039242 | 0.9 | size-dominated | true |  |
| Saito2025 | 46 | 106 | 3d5/2-2p3/2 | `L3-M5` | 833.4 | 0.3 | 832.745022949 | -0.654977051 | -7.854531587 | 0.9 | weakly sensitive | true |  |
| Saito2025 | 46 | 106 | 3d3/2-2p1/2 | `L2-M4` | 863.8 | 0.3 | 862.865157067 | -0.934842933 | -38.254531587 | 0.9 | size-dominated | false |  |
| Saito2025 | 46 | 106 | 4d5/2-2p3/2 | `L3-N5` | 1124.8 | 0.4 | 1123.57250738 | -1.22749262 | -10.313617643 | 1.2 | weakly sensitive | false |  |
| Saito2025 | 46 | 106 | 4f7/2-3d5/2 | `M5-N7` | 291.7 | 0.2 | 291.867334494 | 0.167334494 | -2.759086056 | 0.6 | weakly sensitive | true |  |
| Saito2025 | 46 | 106 | 4f5/2-3d3/2 | `M4-N6` | 295.1 | 0.3 | 295.234448879 | 0.134448879 | -6.159086056 | 0.9 | weakly sensitive | true |  |
| Saito2025 | 46 | 108 | 2p1/2-1s1/2 | `K1-L2` | 3042.9 | 0.2 | 3041.605739465 | -1.294260535 | 17.889471981 | 0.6 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 2p3/2-1s1/2 | `K1-L3` | 3076.6 | 0.2 | 3075.892899167 | -0.707100833 | -15.810528019 | 0.6 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 3d5/2-2p3/2 | `L3-M5` | 833.3 | 0.1 | 832.633451977 | -0.666548023 | -7.738148934 | 0.3 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 3d3/2-2p1/2 | `L2-M4` | 863.5 | 0.1 | 862.666117777 | -0.833882223 | -37.938148934 | 0.3 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 4d5/2-2p3/2 | `L3-N5` | 1124.4 | 0.3 | 1123.466679219 | -0.933320781 | -9.891501060 | 0.9 | weakly sensitive | false |  |
| Saito2025 | 46 | 108 | 4f7/2-3d5/2 | `M5-N7` | 291.6 | 0.1 | 291.872965539 | 0.272965539 | -2.653352127 | 0.3 | weakly sensitive | true |  |
| Saito2025 | 46 | 108 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.1 | 295.239861252 | 0.339861252 | -5.953352127 | 0.3 | weakly sensitive | false |  |
| Saito2025 | 46 | 110 | 2p1/2-1s1/2 | `K1-L2` | 3036.2 | 0.7 | 3034.591198567 | -1.608801433 | 24.561043691 | 2.1 | size-dominated | true |  |
| Saito2025 | 46 | 110 | 2p3/2-1s1/2 | `K1-L3` | 3068.8 | 0.4 | 3068.800009018 | 0.000009018 | -8.038956309 | 1.2 | size-dominated | true |  |
| Saito2025 | 46 | 110 | 3d5/2-2p3/2 | `L3-M5` | 833.4 | 0.2 | 832.533706777 | -0.866293223 | -7.822355439 | 0.6 | size-dominated | false |  |
| Saito2025 | 46 | 110 | 3d3/2-2p1/2 | `L2-M4` | 863.4 | 0.3 | 862.488200232 | -0.911799768 | -37.822355439 | 0.9 | size-dominated | false |  |
| Saito2025 | 46 | 110 | 4d5/2-2p3/2 | `L3-N5` | 1124.2 | 0.5 | 1123.372471649 | -0.827528351 | -9.670179843 | 1.5 | weakly sensitive | true |  |
| Saito2025 | 46 | 110 | 4f7/2-3d5/2 | `M5-N7` | 291.6 | 0.1 | 291.878401931 | 0.278401931 | -2.647824404 | 0.3 | weakly sensitive | true |  |
| Saito2025 | 46 | 110 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.2 | 295.245103622 | 0.345103622 | -5.947824404 | 0.6 | weakly sensitive | true |  |

## 5. The energy the cascade receives

The comparison above scores the orbit-resolved line MuDirac prints; a patched cascade receives, for
each shell above K, the degeneracy-weighted mean of its two circular states, so the energy it emits
between two shells is one number where the sources print two.
`shell_projection.csv` joins each gated row to that shell difference (`consumer_quantity`), beside
the solver line and the energy the unpatched cascade emits, and never relabels a measured line as a
centroid.
Of the 67 gated rows, the shell difference lies within the band for 1, the solver line for 26,
the unpatched cascade for 0, and the shell difference is closer than the unpatched cascade for 42.
`incompatible_groups.csv` takes every (source, nuclide, shell pair) with more than one gated line and
intersects the lines' bands: 29 of the 29 such groups have an empty intersection, the widest,
Pb-208 K–L, by 184.226 keV, so no single shell energy can lie within the band of every line it stands
for.
Every such group is listed below; the table is generated from `incompatible_groups.csv`.

| source | Z | A | initial n | final n | lines | quantities | lower (keV) | upper (keV) | gap (keV) | empty |
|---|---|---|---|---|---|---|---|---|---|---|
| Fricke1995 | 21 | 45 | 2 | 1 | 2 | `K1-L2;K1-L3` | 856.882 | 855.308 | 1.574 | true |
| Fricke1995 | 26 | 56 | 2 | 1 | 2 | `K1-L2;K1-L3` | 1256.915 | 1253.093 | 3.822 | true |
| Fricke1995 | 29 | 63 | 2 | 1 | 2 | `K1-L2;K1-L3` | 1514.311 | 1508.232 | 6.079 | true |
| Fricke1995 | 32 | 74 | 2 | 1 | 2 | `K1-L2;K1-L3` | 1774.830 | 1765.807 | 9.023 | true |
| Fricke1995 | 38 | 88 | 2 | 1 | 2 | `K1-L2;K1-L3` | 2342.168 | 2324.703 | 17.465 | true |
| Fricke1995 | 40 | 90 | 2 | 1 | 4 | `K1-L2;K1-L3;K1-L2;K1-L3` | 2536.470 | 2515.191 | 21.279 | true |
| Fricke1995 | 41 | 93 | 2 | 1 | 2 | `K1-L2;K1-L3` | 2626.632 | 2603.478 | 23.154 | true |
| Fricke1995 | 44 | 102 | 2 | 1 | 2 | `K1-L2;K1-L3` | 2893.819 | 2864.509 | 29.310 | true |
| Fricke1995 | 47 | 107 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3184.239 | 3147.219 | 37.020 | true |
| Fricke1995 | 49 | 115 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3366.696 | 3323.087 | 43.609 | true |
| Fricke1995 | 53 | 127 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3723.643 | 3667.466 | 56.177 | true |
| Fricke1995 | 55 | 133 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3902.543 | 3840.819 | 61.724 | true |
| Fricke1995 | 79 | 197 | 2 | 1 | 2 | `K1-L2;K1-L3` | 5760.333 | 5592.148 | 168.185 | true |
| Fricke1995 | 81 | 205 | 2 | 1 | 2 | `K1-L2;K1-L3` | 5895.280 | 5719.160 | 176.120 | true |
| Fricke1995 | 82 | 208 | 2 | 1 | 2 | `K1-L2;K1-L3` | 5962.584 | 5778.358 | 184.226 | true |
| Saito2025 | 46 | 104 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3091.1 | 3058.7 | 32.4 | true |
| Saito2025 | 46 | 104 | 3 | 2 | 2 | `L3-M5;L2-M4` | 862.7 | 833.7 | 29.0 | true |
| Saito2025 | 46 | 104 | 4 | 3 | 2 | `M5-N7;M4-N6` | 294.3 | 291.9 | 2.4 | true |
| Saito2025 | 46 | 105 | 3 | 2 | 2 | `L3-M5;L2-M4` | 863.6 | 833.2 | 30.4 | true |
| Saito2025 | 46 | 105 | 4 | 3 | 2 | `M5-N7;M4-N6` | 293.1 | 292.0 | 1.1 | true |
| Saito2025 | 46 | 106 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3083.1 | 3051.8 | 31.3 | true |
| Saito2025 | 46 | 106 | 3 | 2 | 2 | `L3-M5;L2-M4` | 862.9 | 834.3 | 28.6 | true |
| Saito2025 | 46 | 106 | 4 | 3 | 2 | `M5-N7;M4-N6` | 294.2 | 292.3 | 1.9 | true |
| Saito2025 | 46 | 108 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3076.0 | 3043.5 | 32.5 | true |
| Saito2025 | 46 | 108 | 3 | 2 | 2 | `L3-M5;L2-M4` | 863.2 | 833.6 | 29.6 | true |
| Saito2025 | 46 | 108 | 4 | 3 | 2 | `M5-N7;M4-N6` | 294.6 | 291.9 | 2.7 | true |
| Saito2025 | 46 | 110 | 2 | 1 | 2 | `K1-L2;K1-L3` | 3067.6 | 3038.3 | 29.3 | true |
| Saito2025 | 46 | 110 | 3 | 2 | 2 | `L3-M5;L2-M4` | 862.5 | 834.0 | 28.5 | true |
| Saito2025 | 46 | 110 | 4 | 3 | 2 | `M5-N7;M4-N6` | 294.3 | 291.9 | 2.4 | true |

`unc` and `u<n>` are the magnitude of one signed radius response with a floor, not a predictive
uncertainty; `components.jsonl` carries, for every value, the signed response in both directions,
the rounding bound of the printed anchor, the shift observed when the numerical settings are
refined, and no total uncertainty.
Refining the settings twice (grid step halved, Uehling steps doubled, convergence tolerance
tightened tenfold each time) moves the Pb-208 `K1-L3` line by 0.208069319 keV against a printed
uncertainty of 0.090 keV.
`radius_lineage.csv` records, per compared nuclide, what is known of the experiments behind the
radius passed to MuDirac and behind the compared line; its state is `UNKNOWN` on every row, because
the evaluation the radii come from is not in hand, and the 16 weakly sensitive rows are 5
isotopes of palladium.

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
value it lists an "Experimental energy"; neither the explanation of that table nor their section 4
states whether the photon recoil has been taken out of a value, and the screen subtracts no recoil
from either side.
Each row also carries the shift of its shell difference from the shipped numerical settings to a
finer reference setting (`dU_keV` from the Uehling steps, `dgrid_keV` from the grid step, and their
sum `dnum_keV`), never added to an uncertainty, and `screen_at_reference`, the screen repeated with
that shift applied.
That sum exceeds a tenth of the row's uncertainty, taken at its largest over every correlation, on
26 of the 27 rows and reaches 0.214379930 keV for Pb-208, and `u_certified`, the smallest Uehling
step count at which the rule in `openmucf/g4/d3_centroids.py` certifies the shell difference against
doubling it, is `none` for 6 of the 26 nuclides.
`centroid_margins.csv` gives, per cohort and source, the smallest, median and largest margin, the
unpatched cascade's absolute residual minus the patched one's, over the isotopes, each counted once,
and the patched cascade's largest absolute residual with its nuclide, as a distribution only; both
files are tabulated below.
For Pb-208 the row also lists the ratio of the doublet's intensities that Jenkins et al. (1971)
measured, from their Table 9 with the target of their Table 3, and the centroid that ratio would
give, for illustration only and never screened.
Neither the nuclear-polarization corrections Fricke et al. print, listed beside each of their rows,
nor electron screening enters the shell difference, a centroid or a band.

| cohort | source | Z | A | value (keV) | sigma (keV) | sigma min (keV) | sigma max (keV) | shell difference (keV) | residual (keV) | unpatched residual (keV) | margin (keV) | screen | dnum (keV) | u_certified | screen at reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| labelled | Fricke1995 | 4 | 9 | 33.402000000 | 0.010000000 |  |  | 33.392443192 | -0.009556808 | -1.510349355 | 1.500792547 | inside | 0.000077480 | 100 | inside |
| labelled | Fricke1995 | 6 | 12 | 75.258200000 | 0.000500000 |  |  | 75.261047326 | 0.002847326 | -2.329588794 | 2.326741468 | outside | 0.000281489 | 300 | outside |
| labelled | Fricke1995 | 8 | 16 | 133.535000000 | 0.002000000 |  |  | 133.532449579 | -0.002550421 | -5.210871570 | 5.208321149 | inside | 0.000728762 | 200 | inside |
| labelled | Fricke1995 | 11 | 23 | 250.229000000 | 0.002000000 |  |  | 250.223863606 | -0.005136394 | -8.875728079 | 8.870591685 | inside | 0.002104159 | 300 | inside |
| labelled | Fricke1995 | 13 | 27 | 346.828000000 | 0.002000000 |  |  | 346.834883059 | 0.006883059 | -12.154564424 | 12.147681365 | outside | 0.003681636 | 400 | outside |
| labelled | Fricke1995 | 14 | 28 | 400.173000000 | 0.005000000 |  |  | 400.165477424 | -0.007522576 | -13.406839395 | 13.399316819 | inside | 0.004686204 | 300 | inside |
| labelled | Fricke1995 | 18 | 40 | 644.004000000 | 0.025000000 |  |  | 643.627044234 | -0.376955766 | -18.135103427 | 17.758147661 | outside | 0.010221448 | 200 | outside |
| constructed | Fricke1995 | 21 | 45 | 856.398333333 |  | 0.013666667 | 0.041000000 | 856.365797437 | -0.032535896 | -19.646241702 | 19.613705806 | inside | 0.016292550 | 200 | inside |
| constructed | Fricke1995 | 26 | 56 | 1255.671000000 |  | 0.010000000 | 0.048666667 | 1255.510937706 | -0.160062294 | -21.945228401 | 21.785166107 | outside | 0.029850213 | 300 | correlation-dependent |
| constructed | Fricke1995 | 29 | 63 | 1512.318666667 |  | 0.011333333 | 0.051333333 | 1512.164751704 | -0.153914963 | -21.487886406 | 21.333971443 | correlation-dependent | 0.039430815 | 300 | correlation-dependent |
| constructed | Fricke1995 | 32 | 74 | 1771.830333333 |  | 0.002666667 | 0.020000000 | 1771.154352356 | -0.675980978 | -15.611615841 | 14.935634864 | outside | 0.049111723 | 500 | outside |
| constructed | Fricke1995 | 38 | 88 | 2336.352333333 |  | 0.002000000 | 0.008666667 | 2335.956916383 | -0.395416951 | -27.193418640 | 26.798001690 | outside | 0.073863272 | none | outside |
| constructed | Fricke1995 | 40 | 90 | 2529.456000000 |  | 0.003000000 | 0.010333333 | 2529.354589792 | -0.101410208 | -25.857812304 | 25.756402096 | outside | 0.082833560 | none | correlation-dependent |
| constructed | Fricke1995 | 40 | 90 | 2529.198666667 |  | 0.007000000 | 0.022333333 | 2529.354589792 | 0.155923126 | -25.600478971 | 25.444555845 | outside | 0.082833560 | none | outside |
| constructed | Fricke1995 | 41 | 93 | 2618.926000000 |  | 0.004000000 | 0.017333333 | 2618.555694794 | -0.370305206 | -20.248095737 | 19.877790531 | outside | 0.086443358 | 700 | outside |
| constructed | Fricke1995 | 44 | 102 | 2884.072000000 |  | 0.007666667 | 0.031000000 | 2883.250099042 | -0.821900958 | -6.589975537 | 5.768074579 | outside | 0.097076307 | 600 | outside |
| constructed | Fricke1995 | 47 | 107 | 3171.913000000 |  | 0.004666667 | 0.023333333 | 3170.603713446 | -1.309286554 | -20.500470701 | 19.191184147 | outside | 0.110619061 | none | outside |
| constructed | Fricke1995 | 49 | 115 | 3352.169666667 |  | 0.003333333 | 0.024666667 | 3351.012379212 | -1.157287454 | -22.436927494 | 21.279640039 | outside | 0.118458839 | none | outside |
| constructed | Fricke1995 | 53 | 127 | 3704.948333333 |  | 0.010333333 | 0.033666667 | 3704.396093694 | -0.552239639 | -30.945683745 | 30.393444106 | outside | 0.133614066 | none | outside |
| constructed | Fricke1995 | 55 | 133 | 3881.991333333 |  | 0.007666667 | 0.033666667 | 3881.736131960 | -0.255201373 | -40.789379902 | 40.534178529 | outside | 0.141441215 | none | outside |
| constructed | Fricke1995 | 79 | 197 | 5704.431333333 |  | 0.053333333 | 0.150666667 | 5706.500360205 | 2.069026872 | -186.487327126 | 184.418300255 | outside | 0.208708155 | 400 | outside |
| constructed | Fricke1995 | 81 | 205 | 5837.263333333 |  | 0.230000000 | 0.663333333 | 5841.524376794 | 4.261043461 | -225.174733054 | 220.913689593 | outside | 0.212955462 | 200 | outside |
| constructed | Fricke1995 | 82 | 208 | 5901.255333333 |  | 0.026666667 | 0.093333333 | 5902.236247053 | 0.980913720 | -227.374463435 | 226.393549716 | outside | 0.214379930 | 500 | outside |
| constructed | Saito2025 | 46 | 104 | 3080.400000000 |  | 0.033333333 | 0.233333333 | 3080.193308696 | -0.206691304 | -19.550430954 | 19.343739650 | correlation-dependent | 0.106716959 | 300 | inside |
| constructed | Saito2025 | 46 | 106 | 3072.766666667 |  | 0.033333333 | 0.366666667 | 3072.397468491 | -0.369198175 | -11.947705909 | 11.578507733 | correlation-dependent | 0.105789953 | 200 | correlation-dependent |
| constructed | Saito2025 | 46 | 108 | 3065.366666667 |  | 0.066666667 | 0.200000000 | 3064.463845933 | -0.902820734 | -4.577194686 | 3.674373952 | outside | 0.104855984 | 300 | outside |
| constructed | Saito2025 | 46 | 110 | 3057.933333333 |  | 0.033333333 | 0.500000000 | 3057.397072201 | -0.536261132 | 2.827710358 | 2.291449225 | correlation-dependent | 0.104032117 | 200 | correlation-dependent |

| cohort | source | isotopes | margin min (keV) | margin median (keV) | margin max (keV) | largest residual (keV) | Z | A |
|---|---|---|---|---|---|---|---|---|
| labelled | Fricke1995 | 7 | 1.500792547 | 8.870591685 | 17.758147661 | 0.376955766 | 18 | 40 |
| constructed | Fricke1995 | 15 | 5.768074579 | 21.785166107 | 226.393549716 | 4.261043461 | 81 | 205 |
| constructed | Saito2025 | 4 | 2.291449225 | 3.674373952..11.578507733 | 19.343739650 | 0.902820734 | 46 | 108 |

## 6. Findings

- The rows outside tolerance are this comparison's registered disagreements; none is fitted away.
- The members whose bundled rms radius differs from the IAEA table are printed by the generator
  and passed as bundled.

## 7. In Geant4

With the opt-in on and the muonic-transition seam reading through `mudirac130` — named by
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
With the opt-in off, or under a profile carrying no D3 table, the cascade and K-energy harvests of
Geant4 v11.4.2 and v11.5.0.beta are bit-identical to those of unpatched builds.
