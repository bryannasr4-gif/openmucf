# D1 — nuclear capture, `parity` profile

> This product includes software developed by Members of the Geant4 Collaboration
> ( http://cern.ch/geant4 ).

`data/g4/d1/` is the first `G4MuonicData` dataset carrying real content. It is a **parity** dataset:
its only claim is that it reproduces the muon-capture data compiled into Geant4 v11.4.2
bit-for-bit. It evaluates nothing, recommends nothing, and corrects nothing — including where the
upstream data looks wrong. Section 5 lists five places where it does.

That restraint is the design. A parity profile is the fixed point everything else is measured
against: if the dataset and the transport code disagree, exactly one of them has changed, and you
can tell which. An *evaluated* profile — one that carries a published evaluation with its own
uncertainties — is a separate file with its own `#PROFILE`, so both can ship side by side and a
consumer chooses by name rather than by hoping.

| File | What |
|---|---|
| `d1_capture.g4dat` | 90 records `Z A value unc` — the nuclear capture rate table |
| `d1_zeff.g4dat` | 101 records `Z value` — the effective-charge table the fallback needs |
| `d1_capture.prov.json`, `d1_zeff.prov.json` | Layer 2: per-row provenance for every record |
| `d1_gp_sweep.oracle` | a harvested bit-parity fixture (section 4) |
| `geant4_add_dataset.snippet` | the registration block, with the archive's MD5 |

Both tables carry `#PROFILE parity` and `#SOURCESHA 8cc04f65977807f1848da7b958c421cd5e162f26`,
which is the Geant4 revision they reproduce.

## 1. Where the numbers come from

From one file: `source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc`, at that revision.
It is vendored into this repository at `third_party/geant4/v11.4.2/`, unmodified, and pinned by its
upstream **git blob id** `29bd73719cd619de34ef83ca5ca076ceadf1cc5a` — upstream's own object name for
those exact bytes, so you can check the copy against `github.com/Geant4/geant4` without cloning
Geant4 and without trusting us.

Every number in this dataset is **parsed out of that file at build time**. No count and no value is
transcribed: `make g4data` re-derives all of them, `make audit` byte-diffs the result, and the test
suite forbids the record counts from appearing as literals anywhere in the extraction code. This
matters more than it sounds. An earlier design note for this project recorded the capture table as
having "94 entries"; 94 is the maximum *Z*. The table has 90 records spanning 74 distinct Z. A
number written down once is a number that drifts.

**What the bibliography says, and what it deliberately does not.** Every Layer-2 row cites
`geant4_v11_4_2` — the software release, which this project has read, because it vendored it. Geant4's
own source comments attribute the data to Suzuki, Measday & Roalsvig (1987), to Phys. Rev. Lett. 99
(2007) 032002 for hydrogen, and to Measday's 2001 review for helium. Those attributions are carried
in each row's `conditions` field as **quoted upstream text, marked as upstream's words**. They are
not this dataset's citations, because citing a paper nobody here has opened would be a provenance
claim this project cannot make. Every row therefore also carries `needs_verification: true`.

## 2. The `goulard_primakoff` model contract

`d1_capture.g4dat` declares a fallback for every `(Z, A)` the table does not list:

```
#FALLBACK goulard_primakoff b0a=-0.03 b0b=-0.25 b0c=3.24 t1=875.e-9 xmu_coeff=2.663e-5 mix=.75704 zmin=1 zmax=100
```

All eight inputs are declared, not four: a consumer handed only the coefficients cannot evaluate
the formula, and a fallback that cannot be evaluated is not a fallback. Each value is the **source
text** of the corresponding constant, spelled as upstream spells it.

**The model.** With `zeff` the `muon_zeff` table of this same dataset and profile:

```
r1     = zeff[max(min(Z, zmax), zmin)]
zeff2  = r1 * r1
xmu    = zeff2 * xmu_coeff
a2ze   = 0.5 * (double)A / (double)Z
r2     = 1.0 - xmu
lambda = t1 * zeff2 * zeff2 * (r2 * r2) * (1.0 - (1.0 - xmu) * mix) *
         (a2ze * b0a + 1.0 - (a2ze - 1.0) * b0b -
          (double)(2 * (A - Z) + fabs(a2ze - 1.0)) * b0c / (double)(A * 4))
```

in units of ns⁻¹. For a table hit the rate is instead `value / 1000`, since `value` is in µs⁻¹
(`1e6/s`) and Geant4's internal time unit is the nanosecond.

**The evaluation order is normative, not stylistic.** Multiplication and addition associate **left
to right**, `2 * (A - Z)` is *integer* arithmetic before it meets the double, and the bracket groups
as `((a2ze*b0a + 1.0) - (a2ze-1.0)*b0b) - ((X*b0c)/(4A))`. Floating-point addition and multiplication
are not associative, so a re-grouped evaluation is a *different function*, and the bit-parity claim
below is a claim about this one.

**A conforming evaluation performs no floating-point contraction.** Fusing a multiply and an add
into a single rounded operation changes the result — see section 5, F-3, where it changes it by up
to 2980 ulp. A C or C++ implementation must therefore compile this expression with
`-ffp-contract=off` (or the equivalent), and must not enable fast-math. Python needs no flag:
CPython rounds every operation separately, which is why the reference implementation in
`openmucf/g4/sources/d1_nuclear_capture.py` *is* the contract rather than merely obeying it.

**Domain.** The model is declared valid for **Z ≥ 1 and A ≥ 1**. Outside that range a conforming
consumer must report a domain error. Geant4 does not — see F-2.

## 3. Two disclosures about the shipped tables

**The capture records are re-ordered, and nothing moved.** The `.g4dat` grammar requires records
ascending by `(Z, A)`. Geant4's array is sorted by Z *alone*, and contains exactly one inversion:
`{92, 238, 12.592, 0.035}` is declared before `{92, 233, 14.27, 0.15}`. This dataset is canonically
sorted, so its record order is not upstream's. Two tests hold that equivalence up: one compares the
record **multiset**, and one reimplements Geant4's actual lookup — a linear scan with the early exit
`if (capRates[j].Z > Z) break;` — over the source order and requires it to agree with a keyed lookup
over the sorted order at every point of a 36000-point box. The canonical order is a refinement of
"sorted by Z", so the early exit fires at the same Z; the test is what makes that an argument rather
than a hope.

**`zeff[0]` ships and is unreachable.** The array holds 101 entries and its first is `0.`, but
`GetMuonZeff` clamps its argument into `[1, 100]` before indexing, so element 0 can never be
returned. It is shipped anyway, because "101/101 bit-identical" means the array *as declared*, and a
dataset that silently dropped an element it claims to reproduce would be a worse artifact than one
that ships it with a disclosure. Its Layer-2 row says so.

## 4. How the parity claim was checked

A Geant4-linked driver (`cpp/tools/harvest_d1.cc`) evaluated
`G4MuonMinusBoundDecay::GetMuonCaptureRate(Z, A)` over **Z ∈ [1,120] × A ∈ [1,300] = 36000 points**
against the built library, and a pure-Python evaluation of the model above — with the association
order preserved — reproduced **every one of them bit-for-bit: 0 mismatches, maximum 0 ulp**. The 90
table hits are included, so both branches of the compiled function are covered.

`d1_gp_sweep.oracle` commits that measurement: a SHA-256 over the whole sweep (big-endian IEEE-754
binary64 bytes, Z ascending outermost), plus a diagnostic subset — every table hit, every Z's first
negative A, and the corners of the box — so a mismatch says *which points* moved rather than only
that something did. Because Python reproduces the sweep exactly, the digest is verifiable **with no
Geant4 present**, on every platform, in ordinary CI.

**A parity claim is a claim about a named build**, and this one names it: Ubuntu 26.04 (WSL2),
x86_64, `g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0`, Geant4 11.4.2 `RelWithDebInfo` (`-O2 -g -DNDEBUG`),
no `-ffp-contract` setting, FMA absent from the baseline ISA. F-3 explains why that qualifier is not
decoration.

## 5. Findings — registered, disclosed, and deliberately not fixed

These are outputs of building this dataset, not obstacles to it. A parity profile's contract is to
reproduce Geant4 *including* its defects; fixing anything here would break the one property the
dataset exists to have. Everything below was measured against the pinned revision.

**F-1 — the Goulard–Primakoff fallback returns negative capture rates.** 6325 of the 36000 swept
points return λ_c < 0. The boundary tracks neutron excess: the first negative A is 3 for Z=1, 6 for
Z=2, 8 for Z=3, 17 for Z=6, 23 for Z=8, 77 for Z=26 and 245 for Z=82 — so for Z ≳ 6 the region is
beyond the neutron drip line and unreachable in practice. For hydrogen it is **³H**, a legal Geant4
target and one of direct interest to muon-catalyzed fusion, at λ_c = −2.870050e−08 ns⁻¹.

Read at the two call sites, the consequences are not cosmetic. Where `lambda = lambdac + lambdad`
and the capture branch is `G4UniformRand()*lambda < lambdac`, a small negative λ_c means the capture
branch can **never** be taken — capture is silently disabled rather than made rare. For the **5407**
swept points where |λ_c| exceeds the free-muon decay rate (4.5517e−04 ns⁻¹), the total λ goes
negative and `time = t − log(U)/λ` moves the muon's global time **backwards**. And a muonic-atom
lifetime computed as `1/(lambdac + lambdad)` becomes negative.

**F-2 — degenerate inputs return non-finite rates, with no coded rejection.** `Z = 0` returns NaN
for any A; `A = 0` returns +inf; `Z = -1, A = 12` returns −5.947382e−07 — finite, negative, and
entirely plausible-looking. Nothing in the source rejects any of these. A value that looks like a
rate but is not one is worse than an error, because it propagates. This dataset's model therefore
declares its domain (Z ≥ 1, A ≥ 1) and requires a conforming consumer to report a domain error
there; the difference between that and what Geant4 does is the finding.

**F-3 — the fallback is not reproducible across builds, and this is the most consequential finding
here.** Compiling the identical expression twice on one machine, in one translation unit:

```
g++ -O2 -ffp-contract=off   vs   g++ -O2 -mfma
36000 points | 14668 differ | 5547 differ by more than 1 ulp | max 2980 ulp at (Z=23, A=118)
maximum relative difference 3.5e-13
```

Geant4's own build sets **no** `-ffp-contract` flag, so contraction is whatever the compiler
defaults to wherever FMA is available — which is the *baseline* ISA on aarch64, and any
`-march=native` x86-64 build. Two conforming builds of one source therefore compute different muon
capture rates. Physically the difference is negligible; formally it is fatal to any unqualified
"bit-identical" claim, which is why section 4 names its build and section 2 forbids contraction.

Worth stating precisely, because it decides who has to care: compiling a *caller* with FMA enabled
changes nothing, since the arithmetic happens inside the prebuilt Geant4 library. The hazard lands
on whoever **compiles the expression** — a standalone validator, or a reimplementation of this
model. Measured here: a caller built `-mfma` against the library gives byte-identical results to one
built `-ffp-contract=off`.

**F-4 — the attribution does not obviously reconcile.** The source comment credits the capture table
to Suzuki, Measday & Roalsvig (1987), with hydrogen and helium carved out. That paper's own abstract
describes lifetimes measured "in 50 elements plus 8 isotopes", while this table spans **74 distinct
Z**. Either the paper compiles world data beyond its own measurements, or the table draws on sources
it does not name. Registered as open; it will be settled against the primary text, and reported
either way. (The abstract is a secondary source and decides nothing on its own.)

**F-5 — `zeff[]` is non-monotonic in the lead region.** The array rises monotonically except at
exactly two steps: Z=81→82 (34.21 → 34.18) and Z=82→83 (34.18 → 34.00), after which it resumes
rising. The step *into* Z=81 is also anomalously large — +0.40, against neighbours of +0.17 to
+0.18. This may be physical structure near the Z=82 shell closure, or a transcription artifact.
Registered; to be checked against the primary; not altered.

## 6. Isotope resolution — what the flag means here

Every Layer-2 row carries a required `isotope_resolved` boolean. **`true` means the row's value is
isotope-resolved; `false` means it is not so established** — which includes "not yet checked".
**The companion field says which kind of `true` a row carries:** while `needs_verification` is
`true` the flag is *derived*, and it is *established* — resting on an isotopically resolved
measurement, with a locator naming where — only once `needs_verification` is `false`.

In this release **every row carries `needs_verification: true`**, so every `true` here is the
derived kind, and the derivation is mechanical: **`true` if and only if the row's Z carries more
than one capture record.** That is sound in one direction and only one — if Geant4 gives different
rates for two A at the same Z, the underlying data distinguishes isotopes; a single row establishes
nothing either way. 27 of the 90 rows, over 11 Z values, are `true`.

**So be precise about what those 27 rows assert.** They assert a fact about *upstream's table* —
that it holds distinct values for distinct A at that Z. They do not assert that the value rests on
an isotopically resolved *measurement*, because nothing here has been checked against a primary.
Every row's `evaluation_method` states the derivation, so a reader meets it together with the flag
rather than being handed a bare boolean. The `zeff` rows are `false` throughout, as a fact rather
than a default: an effective charge is a per-Z quantity, so there is no isotope for it to be
resolved to.

Refining this against the primary literature — which rows are genuinely isotope-resolved and which
are element values wearing an isotope label — is the next step for this dataset, and is the part of
it that is a contribution rather than a reproduction.

## 7. Licensing

The values are derived from Geant4 source redistributed under the Geant4 Software License v1.0; see
`third_party/geant4/`, whose terms apply to that directory. The dataset files themselves are
CC-BY-4.0 and the toolchain is Apache-2.0, as for the rest of this repository.
