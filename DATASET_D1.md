# D1 — nuclear capture, `parity` profile

> This product includes software developed by Members of the Geant4 Collaboration
> ( http://cern.ch/geant4 ).

`data/g4/d1/` is the first `G4MuonicData` dataset carrying real content. It is a **parity** dataset:
its only claim is that it reproduces the muon-capture data compiled into Geant4 v11.4.2
bit-for-bit. It evaluates nothing, recommends nothing, and corrects nothing — including where the
upstream data looks wrong. Section 5 lists seven places where it does.

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
| `isotope_audit.csv` | the isotope-resolution audit against the primary literature (section 6) |
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

**What the bibliography says, and what it deliberately does not.** Every Layer-2 row's
`source_bibkey` is `geant4_v11_4_2` — the software release, which this project has read, because it
vendored it. That does not change: for a parity dataset the value's source *is* the library.
Geant4's own source comments attribute the data to Suzuki, Measday & Roalsvig (1987), to
Phys. Rev. Lett. 99 (2007) 032002 for hydrogen, and to Measday's 2001 review for helium, and those
attributions travel in each row's `conditions` field as **quoted upstream text, marked as upstream's
words** rather than adopted as ours.

Those three papers have since been read, for the isotope audit of section 6 and nothing else, and
they are now in the bibliography. Where a row's `isotope_resolved` flag rests on one of them, its
`source_locator` names the paper, the table and the page in a second, clearly labelled clause, and
records which copy was read. So the two provenance questions stay separate: the first clause of a
locator says where the **value** came from, the second says what established the **flag**. Rows the
primaries do not settle carry no second clause and keep `needs_verification: true`.

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

**The primary states that this formula gets isotopic effects wrong.** Suzuki, Measday & Roalsvig —
the paper the capture table is attributed to — write, in their section IV:

> Now for muon capture the Primakoff and Goulard-Primakoff formulae do not account correctly for
> isotopic effects; thus these formulae predict a larger spread between the isotopes than is
> observed experimentally in Ca, Cr, Ni, U, and Pu. (For Cu, Sr, and Br the experiments are not
> sufficiently precise; for Cl the experiment seems questionable.)

That is a published limitation of this exact model, stated by the authors of the data it falls back
from, and it lands precisely where this dataset's `(Z, A)` keys are weakest — see F-6 and F-7. The
same paper reports its own fit quality for the formula: a mean `(Exp−Fit)/Exp` of **4.1 %** over 30
of its own data points and **5.6 %** over 58 past results. The dataset declares no uncertainty on
fallback values, so those figures are the only published indication of how far off one may be.

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

**F-4 — the attribution reconciles; the paper is a compilation. SETTLED against the primary.** The
source comment credits the capture table to Suzuki, Measday & Roalsvig (1987), with hydrogen and
helium carved out, while that paper's abstract describes lifetimes measured "in 50 elements plus 8
isotopes" and this table spans **74 distinct Z**. The two are not in conflict. The paper's Table III
(light nuclides) and Table IV (Z ≥ 10) are a **compilation of world data**, each row carrying its
own reference and the authors' own measurements marked with an asterisk; the abstract's count
describes what *they* measured, a subset. Between them those two tables span **exactly the same 74
distinct Z** this table carries — the same set, with the same gaps at Z = 36, 43, 44, 54, 61, 63,
69–71, 75–78, 84–89 and 91. Nothing here draws on an unnamed source.

**F-5 — `zeff[]`'s non-monotonicity is the primary's own. SETTLED against the primary.** The array
rises monotonically except at exactly two steps: Z=81→82 (34.21 → 34.18) and Z=82→83 (34.18 →
34.00), after which it resumes rising; the step *into* Z=81 is also anomalously large, +0.40 against
neighbours of +0.17 to +0.18. The primary's Table IV prints **81(34.21), 82(34.18), 83(34.0)** — so
the pattern is not introduced by Geant4's transcription; it is reproduced faithfully from the
source. More generally, **65 of the 101 `zeff` entries appear verbatim in that table and every one
of them matches**; the remaining 36 (Z = 0, Z = 1–9, the gaps above, and Z > 94) are not in it, and
fall to the "Ford and Wills … or interpolated" branch the source comment names. This dataset does
not decide whether the lead-region step is physical structure near the Z = 82 shell closure or an
artifact inherited from further upstream — the primary attributes its Zeff column to Ford & Wills
(1962), which is also the fallback the Geant4 comment names, so the two agree on their source and
neither resolves the question. Not altered.

One measured fact bears on it. Table IV of that same paper prints the barium row as **"59(29.99)"**
— barium is Z = 56, and 29.99 sits correctly between caesium's 29.75 (Z = 55) and lanthanum's 30.22
(Z = 57). So the table demonstrably contains at least one typographical error in its Z column, which
raises rather than settles the prior that the lead-region step is also one. Geant4 read that row
correctly: `zeff[56] = 29.99`.

**F-6 — the declared fallback is documented, by the primary, to mispredict isotopic effects.** See
section 2 for the quotation. It matters here because of what it names: the authors say the formula
over-predicts the spread between isotopes for Ca, Cr, Ni, U and Pu, and that for **Cu, Sr and Br**
the experiments are not precise enough to tell, while **for Cl the experiment "seems questionable"**.
Four of those elements — Cl, Cr, Ni and U — are exactly the ones this table keys by separated
isotope, so a consumer relying on the fallback to interpolate between isotopes is relying on a
formula its own source says does not do that well, in the region where it was checked and found
wanting.

**F-7 — `(Z, A)` is a label on most rows, not a target specification.** For **41 of the 90 records**
the primary shows the measurement was made on a **natural-composition element**, not on the nuclide
the record's `A` names. In 15 of those, `A` is not even the element's most abundant nuclide: the
extremes are `(62, 150)`, where Sm-150 is **7.4 %** of natural samarium, and `(50, 119)`, where
Sn-119 is **8.6 %** of natural tin. For `(30, 66)` and `(32, 72)` the `A` is neither the rounded
standard atomic weight nor the most abundant nuclide.

The consequence is a trap with three walls. A consumer asking for the *actual* dominant isotope —
Pb-208, say, which is 52 % of natural lead where this table keys Pb at A=207 — **misses the table
entirely**, falls through to `goulard_primakoff` (F-6: documented to mispredict isotopic effects,
with no declared uncertainty), and in the neutron-rich direction may land in the region where that
fallback returns a negative rate (F-1). Each of the three is individually minor; together they turn
a reasonable-looking call into a silent wrong answer. This dataset does not change any key — it
reproduces the table — but it now says, per row, which reading is which.

## 6. Isotope resolution — what the flag means here

Every Layer-2 row carries a required `isotope_resolved` boolean, and a companion `needs_verification`
that says how much weight it can bear. **`true` means the row's value is established to rest on an
isotopically resolved measurement; `false` means it is not so established.** `needs_verification`
is what separates the two ways a row can be `false`. While it is `true`, the question is *open* —
the flag has not been established either way and `false` is only "not shown to be resolved". Once
it is `false`, the row has been **checked against a primary** and the flag is a finding in whichever
direction it points: `true` means shown resolved, `false` means shown *un*resolved. Reading a
`false` without its companion field therefore loses exactly the distinction this section exists to
draw.

**In this release the capture flags are established from the primary literature, row by row.** Each
of the 90 records was checked against the paper its own value is attributed to — the capture table
to Suzuki, Measday & Roalsvig (1987), hydrogen and helium to the two sources the source comment
carves them out to. **86 of the 90 are settled and carry `needs_verification: false`; 4 are not and
still carry `true`.** Of the settled rows, **45 are `isotope_resolved: true`**: 23 because the
primary lists a separated isotope with that mass number, 19 because the element is mononuclidic and
so a natural-composition target *is* a single nuclide, and 3 — the hydrogen and helium records —
because the sources the table carves them out to describe an isotopically distinct target
(deuterium-depleted protium; ³He and ⁴He tabulated as separate nuclides). The remaining **41 are
`false` as an established finding**, because the primary shows the value rests on a
natural-composition element (F-7).

Every row's `evaluation_method` now states which of those routes produced its flag and carries the
evidence itself, so a reader meets the reasoning together with the boolean rather than a bare one.
Its `source_locator` gains a second, labelled clause naming the table and page that established the
flag and which copy of the paper was read; the first clause still points at the vendored source
line, because that is where the *value* comes from and the two questions must not be conflated. The
audit is shipped as `data/g4/d1/isotope_audit.csv` — one row per record, hand-authored, reviewable
and diffable.

**The four unsettled rows say so with an empty locator, and they are worth naming.** Three —
`(17, 35)`, `(24, 52)` and `(38, 88)` — sit where the primary lists *both* a natural-composition
entry and a separated isotope of that mass number, and where the mass number is also the rounded
standard atomic weight, so the key alone cannot say which entry the record reproduces. The fourth,
`(92, 236)`, is a record whose nuclide the primary's table does not contain at all: it carries
U-233, U-235 and U-238 and no U-236. Deciding these needs a value-level comparison against the
primary, which is a later stage's work; this dataset reports them open rather than guessing.

**What changed, and why the previous rule was not enough.** The earlier release derived the flag
mechanically: `true` if and only if the row's Z carried more than one capture record. Its soundness
argument was about the *Z* — two differing rates at one Z do show the underlying data distinguishes
isotopes — and it was then applied to each *row* of that Z, which does not follow, since one of
those rows can still be the natural-composition entry. The primary disagrees with that rule on **28
of the 90 records**: 23 it called unresolved that are resolved, and 5 it called resolved that are
not. Two of the five are plain: the primary lists "C" and "C-13", and "O" and "O-18", so `(6, 12)`
and `(8, 16)` are natural carbon and natural oxygen, not C-12 and O-16.

The `zeff` rows remain `false` throughout, as a fact rather than a default: an effective charge is a
per-Z quantity, so there is no isotope for it to be resolved to. Their `needs_verification` stays
`true`.

**Two limits on all of the above, stated plainly.** First, what was read is the TRIUMF preprint
TRI-PP-87-5 (January 1987), a scanned copy, not the published article; every locator cites the
preprint's pagination and every row records `copy_read`, and if a value's evidence differs between
the two copies this audit is the one that must move. Second, this is a resolution audit, not an
evaluation: no shipped value was compared to the primary for correctness, and none was altered.

## 7. Licensing

The values are derived from Geant4 source redistributed under the Geant4 Software License v1.0; see
`third_party/geant4/`, whose terms apply to that directory. The dataset files themselves are
CC-BY-4.0 and the toolchain is Apache-2.0, as for the rest of this repository.
