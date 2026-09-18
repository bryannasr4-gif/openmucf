# The D3 energy tables of the `mudirac130` profile

## 1. What ships

`data/g4/d3/` ships the tables `k_shell_energy` (`d3_kshell.mudirac130.g4dat`) and `level_energy`
(`d3_levels.mudirac130.g4dat`) under `#PROFILE mudirac130` and `#SEAM d3_transitions`, each with its
Layer-2 file.
Every value is a positive binding energy in keV computed by MuDirac 1.3.0 (Sturniolo and
Hillier (2021); Liborio et al. (2026)), a Dirac-equation solver for muonic atoms released under the
MIT License with the copyright held by the Science and Technology Facility Council, as its
`LICENSE` reads.
`k_shell_energy` carries the binding energy of the `K1` orbit in its `value` column, with its
uncertainty in `unc`.
`level_energy` carries, for each shell above `K`, the mean binding energy of both circular states
of the shell weighted by their degeneracy, in the columns `e<n>`, with their uncertainties in
`u<n>`.
The chain ends at shell 8, the highest shell through which every circular state converged on
the nuclides checked when the settings were fixed.
Each table has a row for every kept (Z, A) and, for each element whose most abundant isotope in
MuDirac's `abundant.dat` is kept, a natural-composition row that carries that isotope's values
under `#VALIDITY` `A:most_abundant_and_listed`.
Of the 895 members of the input set, 894 are kept, and the tables carry 83
natural-composition rows.

## 2. How the values are made

The input set is every (Z, A) with Z from 1 through 92 that MuDirac's bundled
`nuclear_radii.dat` lists.
Every run uses the same keywords: `nuclear_model: FERMI2`, `uehling_correction: TRUE`,
`reduced_mass: TRUE`, `optimise_fermi_parameters: FALSE`, `fermi_t: 2.3`, and `radius:` set
explicitly to the value `nuclear_radii.dat` bundles; no `electronic_config` is given, so the atom
carries no electrons, and every other keyword is at its documented default.
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
energy of the outermost circular state, and the derivation raises unless every derived energy lies
within the print precision of its own printed state energy and every cross line closes.
`value` and `e<n>` come from the `base` run; `unc` and `u<n>` are the absolute change of the same
quantity in the `rsig` run, whose rms radius is moved by its uncertainty in the IAEA charge-radii
table (`charge_radii.csv`), and propagate nothing else.
Where the bundled rms radius and that table's value differ at the fourth decimal, the generator
prints the member; neither value is edited.

## 3. Which nuclides are kept

A member is kept when its `base`, `rsig` and hydrogen-like runs exit cleanly with an empty error
file, when each circular state of shells 6 through 8 lies within 1 % of the
same state in the run that treats the atom as hydrogen-like from that shell up
(`ideal_atom_minshell`), and when the derivation holds on its `base` and `rsig` runs.
The generator drops `He8`, the only member whose sphere radius gives no real value of the
Fermi parameter c that MuDirac computes by default, and prints its radius beside the radius below
which that value is not real.

## 4. Comparison with measured energies

`validation.csv` compares the model with the transition energies transcribed from Fricke et al.
(1995), Tables IIIA and IIIB, and from Saito et al. (2025), Table III, each as printed
(`validation_cells.csv`).
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
`centroid` marks a Table IIIA centre of gravity, and `hyperfine` a Table IIIB hyperfine component
or a line Saito et al. describe as showing hyperfine splitting.
Both are reported and not compared: the first because no model quantity was fixed for it before
the comparison, the second because the keywords MuDirac documents include none for a hyperfine or
quadrupole interaction.
Of the 67 gated rows, 26 lie within tolerance and 41 outside it; 16 of the gated
rows are weakly sensitive, and 13 of those lie within tolerance.
Every gated row is listed below; the table is generated from `validation.csv`.

| source | Z | A | transition | line | measured (keV) | sigma (keV) | model (keV) | residual (keV) | tol (keV) | label | within | NPol (keV) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fricke1995 | 21 | 45 | 2p1/2-1s1/2 | `K1-L2` | 855.185 | 0.041 | 855.168632818 | -0.016367182 | 0.123 | size-dominated | true | 0.182 |
| Fricke1995 | 21 | 45 | 2p3/2-1s1/2 | `K1-L3` | 857.005 | 0.041 | 856.964379747 | -0.040620253 | 0.123 | size-dominated | true | 0.203 |
| Fricke1995 | 24 | 52 | 2p3/2-1s1/2 | `K1-L3` | 1092.286 | 0.021 | 1092.000364461 | -0.285635539 | 0.063 | size-dominated | false | 0.299 |
| Fricke1995 | 26 | 56 | 2p1/2-1s1/2 | `K1-L2` | 1252.919 | 0.058 | 1252.746243216 | -0.172756784 | 0.174 | size-dominated | true | 0.403 |
| Fricke1995 | 26 | 56 | 2p3/2-1s1/2 | `K1-L3` | 1257.047 | 0.044 | 1256.893284951 | -0.153715049 | 0.132 | size-dominated | false | 0.403 |
| Fricke1995 | 29 | 63 | 2p1/2-1s1/2 | `K1-L2` | 1508.052 | 0.060 | 1507.951824562 | -0.100175438 | 0.180 | size-dominated | true | 0.467 |
| Fricke1995 | 29 | 63 | 2p3/2-1s1/2 | `K1-L3` | 1514.452 | 0.047 | 1514.271215275 | -0.180784725 | 0.141 | size-dominated | false | 0.538 |
| Fricke1995 | 32 | 74 | 2p1/2-1s1/2 | `K1-L2` | 1765.729 | 0.026 | 1765.043144757 | -0.685855243 | 0.078 | size-dominated | false | 0.836 |
| Fricke1995 | 32 | 74 | 2p3/2-1s1/2 | `K1-L3` | 1774.881 | 0.017 | 1774.209956155 | -0.671043845 | 0.051 | size-dominated | false | 0.839 |
| Fricke1995 | 38 | 88 | 2p1/2-1s1/2 | `K1-L2` | 2324.673 | 0.010 | 2324.311814226 | -0.361185774 | 0.030 | size-dominated | false | 0.929 |
| Fricke1995 | 38 | 88 | 2p3/2-1s1/2 | `K1-L3` | 2342.192 | 0.008 | 2341.779467461 | -0.412532539 | 0.024 | size-dominated | false | 0.937 |
| Fricke1995 | 40 | 90 | 2p1/2-1s1/2 | `K1-L2` | 2515.368 | 0.011 | 2515.292409825 | -0.075590175 | 0.033 | size-dominated | false | 0.968 |
| Fricke1995 | 40 | 90 | 2p3/2-1s1/2 | `K1-L3` | 2536.500 | 0.010 | 2536.385679776 | -0.114320224 | 0.030 | size-dominated | false | 0.975 |
| Fricke1995 | 40 | 90 | 2p1/2-1s1/2 | `K1-L2` | 2515.122 | 0.023 | 2515.292409825 | 0.170409825 | 0.069 | size-dominated | false | 1.083 |
| Fricke1995 | 40 | 90 | 2p3/2-1s1/2 | `K1-L3` | 2536.237 | 0.022 | 2536.385679776 | 0.148679776 | 0.066 | size-dominated | false | 0.964 |
| Fricke1995 | 41 | 93 | 2p1/2-1s1/2 | `K1-L2` | 2603.418 | 0.020 | 2603.210000102 | -0.207999898 | 0.060 | size-dominated | false | 0.991 |
| Fricke1995 | 41 | 93 | 2p3/2-1s1/2 | `K1-L3` | 2626.680 | 0.016 | 2626.22854214 | -0.45145786 | 0.048 | size-dominated | false | 1.060 |
| Fricke1995 | 44 | 102 | 2p1/2-1s1/2 | `K1-L2` | 2864.404 | 0.035 | 2863.638237678 | -0.765762322 | 0.105 | size-dominated | false | 1.547 |
| Fricke1995 | 44 | 102 | 2p3/2-1s1/2 | `K1-L3` | 2893.906 | 0.029 | 2893.056029724 | -0.849970276 | 0.087 | size-dominated | false | 1.557 |
| Fricke1995 | 47 | 107 | 2p1/2-1s1/2 | `K1-L2` | 3147.135 | 0.028 | 3145.884473083 | -1.250526917 | 0.084 | size-dominated | false | 1.487 |
| Fricke1995 | 47 | 107 | 2p3/2-1s1/2 | `K1-L3` | 3184.302 | 0.021 | 3182.963333628 | -1.338666372 | 0.063 | size-dominated | false | 1.485 |
| Fricke1995 | 49 | 115 | 2p1/2-1s1/2 | `K1-L2` | 3322.991 | 0.032 | 3322.569897053 | -0.421102947 | 0.096 | size-dominated | false | 0.915 |
| Fricke1995 | 49 | 115 | 2p3/2-1s1/2 | `K1-L3` | 3366.759 | 0.021 | 3365.233620292 | -1.525379708 | 0.063 | size-dominated | false | 1.933 |
| Fricke1995 | 53 | 127 | 2p1/2-1s1/2 | `K1-L2` | 3667.361 | 0.035 | 3667.632427035 | 0.271427035 | 0.105 | size-dominated | false | 0.532 |
| Fricke1995 | 53 | 127 | 2p3/2-1s1/2 | `K1-L3` | 3723.742 | 0.033 | 3722.777927024 | -0.964072976 | 0.099 | size-dominated | false | 1.454 |
| Fricke1995 | 55 | 133 | 2p1/2-1s1/2 | `K1-L2` | 3840.702 | 0.039 | 3840.331520121 | -0.370479879 | 0.117 | size-dominated | false | 1.531 |
| Fricke1995 | 55 | 133 | 2p3/2-1s1/2 | `K1-L3` | 3902.636 | 0.031 | 3902.43843788 | -0.19756212 | 0.093 | size-dominated | false | 1.289 |
| Fricke1995 | 60 | 142 | 2p3/2-1s1/2 | `K1-L3` | 4352.354 | 0.051 | 4352.31846825 | -0.03553175 | 0.153 | size-dominated | true | 1.957 |
| Fricke1995 | 79 | 197 | 2p1/2-1s1/2 | `K1-L2` | 5591.710 | 0.146 | 5593.605121885 | 1.895121885 | 0.438 | size-dominated | false | -0.538 |
| Fricke1995 | 79 | 197 | 2p3/2-1s1/2 | `K1-L3` | 5760.792 | 0.153 | 5762.947979365 | 2.155979365 | 0.459 | size-dominated | false | -1.305 |
| Fricke1995 | 81 | 205 | 2p1/2-1s1/2 | `K1-L2` | 5717.210 | 0.650 | 5721.677852281 | 4.467852281 | 1.950 | size-dominated | false | 3.737 |
| Fricke1995 | 81 | 205 | 2p3/2-1s1/2 | `K1-L3` | 5897.290 | 0.670 | 5901.447639051 | 4.157639051 | 2.010 | size-dominated | false | 3.737 |
| Fricke1995 | 82 | 208 | 2p1/2-1s1/2 | `K1-L2` | 5778.058 | 0.100 | 5779.069723885 | 1.011723885 | 0.300 | size-dominated | false | 2.945 |
| Fricke1995 | 82 | 208 | 2p3/2-1s1/2 | `K1-L3` | 5962.854 | 0.090 | 5963.819508637 | 0.965508637 | 0.270 | size-dominated | false | 2.718 |
| Saito2025 | 46 | 104 | 2p1/2-1s1/2 | `K1-L2` | 3057.8 | 0.3 | 3057.219525585 | -0.580474415 | 0.9 | size-dominated | true |  |
| Saito2025 | 46 | 104 | 2p3/2-1s1/2 | `K1-L3` | 3091.7 | 0.2 | 3091.680200251 | -0.019799749 | 0.6 | size-dominated | true |  |
| Saito2025 | 46 | 104 | 3d5/2-2p3/2 | `L3-M5` | 833.4 | 0.1 | 832.852073144 | -0.547926856 | 0.3 | size-dominated | false |  |
| Saito2025 | 46 | 104 | 3d3/2-2p1/2 | `L2-M4` | 863.6 | 0.3 | 863.057864657 | -0.542135343 | 0.9 | size-dominated | true |  |
| Saito2025 | 46 | 104 | 4d5/2-2p3/2 | `L3-N5` | 1122.5 | 0.6 | 1123.67359222 | 1.17359222 | 1.8 | weakly sensitive | true |  |
| Saito2025 | 46 | 104 | 4f7/2-3d5/2 | `M5-N7` | 291.6 | 0.1 | 291.86147437 | 0.26147437 | 0.3 | weakly sensitive | true |  |
| Saito2025 | 46 | 104 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.2 | 295.228795545 | 0.328795545 | 0.6 | weakly sensitive | true |  |
| Saito2025 | 46 | 105 | 2p1/2-1s1/2 | `K1-L2` | 3054.9 | 0.5 | 3054.902109983 | 0.002109983 | 1.5 | size-dominated | true |  |
| Saito2025 | 46 | 105 | 3d5/2-2p3/2 | `L3-M5` | 832.3 | 0.3 | 832.823524283 | 0.523524283 | 0.9 | weakly sensitive | true |  |
| Saito2025 | 46 | 105 | 3d3/2-2p1/2 | `L2-M4` | 864.5 | 0.3 | 863.003680202 | -1.496319798 | 0.9 | size-dominated | false |  |
| Saito2025 | 46 | 105 | 4f7/2-3d5/2 | `M5-N7` | 291.4 | 0.2 | 291.864470739 | 0.464470739 | 0.6 | weakly sensitive | true |  |
| Saito2025 | 46 | 105 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.6 | 295.231744311 | 0.331744311 | 1.8 | weakly sensitive | true |  |
| Saito2025 | 46 | 106 | 2p1/2-1s1/2 | `K1-L2` | 3050.3 | 0.5 | 3049.480915874 | -0.819084126 | 1.5 | size-dominated | true |  |
| Saito2025 | 46 | 106 | 2p3/2-1s1/2 | `K1-L3` | 3084.0 | 0.3 | 3083.8557448 | -0.1442552 | 0.9 | size-dominated | true |  |
| Saito2025 | 46 | 106 | 3d5/2-2p3/2 | `L3-M5` | 833.4 | 0.3 | 832.745022949 | -0.654977051 | 0.9 | weakly sensitive | true |  |
| Saito2025 | 46 | 106 | 3d3/2-2p1/2 | `L2-M4` | 863.8 | 0.3 | 862.865157067 | -0.934842933 | 0.9 | size-dominated | false |  |
| Saito2025 | 46 | 106 | 4d5/2-2p3/2 | `L3-N5` | 1124.8 | 0.4 | 1123.57250738 | -1.22749262 | 1.2 | weakly sensitive | false |  |
| Saito2025 | 46 | 106 | 4f7/2-3d5/2 | `M5-N7` | 291.7 | 0.2 | 291.867334494 | 0.167334494 | 0.6 | weakly sensitive | true |  |
| Saito2025 | 46 | 106 | 4f5/2-3d3/2 | `M4-N6` | 295.1 | 0.3 | 295.234448879 | 0.134448879 | 0.9 | weakly sensitive | true |  |
| Saito2025 | 46 | 108 | 2p1/2-1s1/2 | `K1-L2` | 3042.9 | 0.2 | 3041.605739465 | -1.294260535 | 0.6 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 2p3/2-1s1/2 | `K1-L3` | 3076.6 | 0.2 | 3075.892899167 | -0.707100833 | 0.6 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 3d5/2-2p3/2 | `L3-M5` | 833.3 | 0.1 | 832.633451977 | -0.666548023 | 0.3 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 3d3/2-2p1/2 | `L2-M4` | 863.5 | 0.1 | 862.666117777 | -0.833882223 | 0.3 | size-dominated | false |  |
| Saito2025 | 46 | 108 | 4d5/2-2p3/2 | `L3-N5` | 1124.4 | 0.3 | 1123.466679219 | -0.933320781 | 0.9 | weakly sensitive | false |  |
| Saito2025 | 46 | 108 | 4f7/2-3d5/2 | `M5-N7` | 291.6 | 0.1 | 291.872965539 | 0.272965539 | 0.3 | weakly sensitive | true |  |
| Saito2025 | 46 | 108 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.1 | 295.239861252 | 0.339861252 | 0.3 | weakly sensitive | false |  |
| Saito2025 | 46 | 110 | 2p1/2-1s1/2 | `K1-L2` | 3036.2 | 0.7 | 3034.591198567 | -1.608801433 | 2.1 | size-dominated | true |  |
| Saito2025 | 46 | 110 | 2p3/2-1s1/2 | `K1-L3` | 3068.8 | 0.4 | 3068.800009018 | 0.000009018 | 1.2 | size-dominated | true |  |
| Saito2025 | 46 | 110 | 3d5/2-2p3/2 | `L3-M5` | 833.4 | 0.2 | 832.533706777 | -0.866293223 | 0.6 | size-dominated | false |  |
| Saito2025 | 46 | 110 | 3d3/2-2p1/2 | `L2-M4` | 863.4 | 0.3 | 862.488200232 | -0.911799768 | 0.9 | size-dominated | false |  |
| Saito2025 | 46 | 110 | 4d5/2-2p3/2 | `L3-N5` | 1124.2 | 0.5 | 1123.372471649 | -0.827528351 | 1.5 | weakly sensitive | true |  |
| Saito2025 | 46 | 110 | 4f7/2-3d5/2 | `M5-N7` | 291.6 | 0.1 | 291.878401931 | 0.278401931 | 0.3 | weakly sensitive | true |  |
| Saito2025 | 46 | 110 | 4f5/2-3d3/2 | `M4-N6` | 294.9 | 0.2 | 295.245103622 | 0.345103622 | 0.6 | weakly sensitive | true |  |

## 5. Findings

- The rows outside tolerance are this comparison's registered disagreements; none is fitted away.
- The dropped member of section 3 is a limit of the fixed settings, not of the input data.
- The members whose bundled rms radius differs from the IAEA table are printed by the generator
  and passed as bundled.
