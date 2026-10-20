# Historical numerical certificate and reference evidence

This directory publishes the primitive calculation records behind the numerical check in `numerical_check.csv`.
It describes an already exposed historical calculation. This public description was written after those results;
its cohorts are not prospective holdouts for reproduction. The numerical rule concerns the specified Hamiltonian.
It does not measure physical accuracy or bound omitted physics.

The standard-library evaluator is `openmucf/g4/d3_numerical.py`; the loader in `openmucf/g4/d3_contract.py` checks
the historical CSV against its computed values.

## Files and units

`protocol.json` identifies the schema, fixed producer recipe and SHA-256 digests of the evidence files.
`cases.json` holds the physical keys, historical cohort, radius and surface thickness in fm, explicit nuclear
reduced mass in electron-mass units, on/off production mappings and probed density parameters in Bohr.
Radii reuse `data/g4/d3/mudirac_inputs.csv`; the reduced masses are calculation inputs derived with the original
embedded isotope masses and electron-binding correction. No isotope mass table is redistributed here.

`payloads.jsonl` holds both potential switches for each case. Each payload contains Method A production and
refinement energies, state node/kappa identities, both Method B bases, domain and Coulomb-discretization shifts,
and the first-order expectation data on the Uehling-free member. Eigenvalues and level shifts are in Hartree;
the payload's `e_h_eV` gives the output conversion. JSON numbers retain round-trip binary precision.

`runs.jsonl` records the required producer runs, their physical keys, numerical settings, input text and digest,
printed line values, state headers and node counts. A mapping identifies each reused production/refinement run
already present in the public output tables. The remaining records supply the additional ladders, tolerance
and phase runs. Input digests prove lineage; the evaluator also reconstructs quantities and tests the rule.
`expected_keys.json` fixes the complete case/switch/quantity cross-product independently of the CSV being checked.
Missing, duplicate or extra keys cannot reduce the population being tested.

## Fixed Hamiltonian and quantity transform

The radial Dirac equation uses `x = ln(r)`, binding eigenvalue `B = E - mu*c*c`, and

```
dP/dx = -kappa*P + r*(alpha*(B - V) + 2*mu*c)*Q
dQ/dx =  kappa*Q - r*alpha*(B - V)*P
```

The original mixed constant conventions are preserved: `alpha = 7.2973525664e-3`, `c = 137.035999139`,
muon mass `1/4.83633170e-3` electron masses, and atomic-mass unit `1822.888486192` electron masses.
The output Hartree is `27.211386245981` eV; the nuclear-mass solve precedes this output conversion.
The nuclear mass is the embedded atomic mass times that atomic-mass unit, minus Z electron masses, plus
the total electron binding divided by the squared speed constant. The electron binding in eV is
`14.4381*Z**2.39 + 1.55468e-6*Z**5.35`, converted to Hartree with the output Hartree before division by `c*c`.
The explicit reduced mass is `1/(1/m_mu + 1/M_nuclear)`. These conventions are preserved for reproduction;
changing them defines a different numerical problem.

Both Coulomb and Uehling potentials use the same normalized two-parameter Fermi density
`rho(r) proportional to 1/(1 + exp(4*ln(3)*(r-c)/t))`. The solver determined c by exact rms inversion at the
case's fixed rms and thickness. Method A is DOP853 shooting/matching; Method B uses dual-kinetic-balance
B-splines. They share this Hamiltonian and potential machinery and are not independent physical models.

Producer bindings telescope from the printed `R15-S17` line and the analytic hydrogenic S17 anchor, using
the archived nuclear reduced mass and the original binary arithmetic. State bindings give K and the
degeneracy-weighted circular-state means e2 through e8; differences of these means give shell energies.
Resolved lines use the producer's own printed line. Each reference variant undergoes the same quantity
transform before the maximum reference displacement is taken. Summing separate per-level maxima would
give a different bound and is not the rule.

## Historical rule

The fixed recipe is k=3, N_u=60, N_s=12 and energy tolerance `1e-7`, without extrapolation.
The dx ladder has k=1,2,3,4. Each printed line allowance is half its last printed unit plus
`ulp(mu*c*c)/eV_factor`; telescoped levels and shell means retain the historical accumulated allowances.
For the floor predicate, every successive difference must satisfy `abs(d) <= 2*unit` or
`abs(d) < 1e-12 eV`. Both estimators then use the unsigned term
`2*(abs(q[-2]-q[-1]) + rho[-2] + rho[-1])`, with no order estimated.

Otherwise the original and rounding-aware Richardson estimators are evaluated separately. Their last
two estimated orders must both be finite and agree within 0.3. The rounding-aware estimator also requires same-sign differences
resolved beyond their rounding allowances and an order interval no wider than 0.1. Stable orders give
the registered signed remainder; unstable or unresolved orders use their unsigned fallbacks. A zero
order with divergent remainder fails. The record qualifies only if both estimator evaluations qualify.

The signed on-only N_u and N_s terms use the next required refinement. Domain and csteps terms come from
the reference payload after the quantity transform. Tolerance and phase terms use their required producer
runs. Required terms are never replaced by zero when missing. These recipe-point runs have zero separability
residual. The comparator computes

```
b_ladder = sum(abs(terms)) + abs(separability) + rounding
D_obs = Q_MD - Q_ref
D_pred = sum(signed terms)
allowance = sum(abs(unsigned terms)) + b_ref + 0.1*target + rounding + abs(separability)
b_num = max(b_ladder, abs(D_obs) + b_ref)
```

The reference bound b_ref is the maximum per-quantity displacement from Method A production among
its refinement variants and the Method B bases. G1 checks producer state/header consistency and reference
node/kappa identities; G2 requires `b_ref <= 0.2*target`; G3 requires `b_num <= target`; G4 requires
`abs(D_obs-D_pred) <= allowance`. G5 is `NOT_APPLICABLE` for this non-extrapolated recipe, not PASS.
All failures are retained; the first-failure reason follows G1 through G4 in order.

Targets derive from the existing measured uncertainty inputs: the nuclide target is a tenth of the
smallest eligible experimental uncertainty. The engineering target is `max(0.001 eV, 1e-6*abs(Q_ref))`.
Levels and resolved lines use the nuclide target when available. Shell quantities use the smaller
of the nuclide and engineering targets, except the explicit labelled-centroid K-e2 cases enumerated
in the evaluator. Nuclides without a measured target use the engineering target.

## Historical summaries

`numerical_check.csv` retains the historical rounded serialization: Q and Q_ref use six decimal places,
b_ref and each estimator's bound use `.3e`, and target uses `.4g`. The summary b_num uses the larger
serialized estimator bound. Nearest-rounded summary cells are not exact bounds or full-precision
certificate inputs. The loader reconstructs and verifies every serialized field and the verdict;
correctly encoded adverse evaluations remain adverse.

First-party calculation records and this description are CC-BY-4.0 under `LICENSE-DATA`.
Reused upstream radii retain the notices and source limits stated in `README.md` and `LICENSE-MUDIRAC`.
