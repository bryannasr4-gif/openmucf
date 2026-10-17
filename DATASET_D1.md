# D1 — nuclear capture

> This product includes software developed by Members of the Geant4 Collaboration
> ( http://cern.ch/geant4 ).

`data/g4/d1/` is the first `G4MuonicData` dataset carrying real content. Its `parity` tables make
one claim: that they reproduce the muon-capture data compiled into Geant4 `v11.4.2` and
`v11.5.0.beta` bit-for-bit, on the builds section 4 names. They evaluate nothing, recommend nothing,
and correct nothing — including where the upstream data looks wrong. Section 5 carries eight
findings: **six** defects it reproduces rather than corrects, and **two** questions about the
upstream data that reading the primary literature has since settled.

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
| `zeff_audit.csv` | every `Z(Z_eff)` cell of the primary's Tables III and IV as printed, with the cells it underlines (section 5, F-5) |
| `capture_rate_cells.csv` | every Total Capture Rate cell the primary prints at the Z of a capture row whose listing alone does not settle it, as printed (section 6) |
| `geant4_add_dataset.snippet` | the registration block, with the archive's MD5 |

Both tables carry `#PROFILE parity` and `#SOURCESHA 8cc04f65977807f1848da7b958c421cd5e162f26`,
which is the Geant4 revision they reproduce.

## 1. Where the numbers come from

From one file: `source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc`, at that revision.
It is vendored into this repository at `third_party/geant4/v11.4.2/`, unmodified, and pinned by its
upstream **git blob id** `29bd73719cd619de34ef83ca5ca076ceadf1cc5a` — upstream's own object name for
those exact bytes, so you can check the copy against `github.com/Geant4/geant4` without cloning
Geant4 and without trusting us.

An earlier design note for this project recorded the capture table as
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
locator says where the **value** came from, the second says what established the **flag**.

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

**The evaluation order is normative, not stylistic.** `2 * (A - Z)` is *integer* arithmetic before
it meets the double, and the bracket groups
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

Read it precisely, because the sentence makes two different statements. For **Ca, Cr, Ni, U and Pu**
it reports a *model* failure — the formula over-predicts the spread between isotopes. For **Cu, Sr,
Br and Cl** it reports a *data* limitation — the experiments are imprecise, and in chlorine's case
questionable — which says nothing about the model. F-6 keeps the two apart.

The same paper reports its own fit quality for the formula, in its Table V (printed p. 29): the row
it labels `(Exp-Fit)/Exp` reads **4.1 %** against 30 of its own data points and **5.6 %** against 58
past results. Read those with two qualifications.
They are **fit residuals over the points the fit was made against**, not a validated uncertainty for
an arbitrary `(Z, A)`, and in particular not for the neutron-rich region where F-1 shows the formula
fails outright. And they belong to **that paper's own fit**, whereas the coefficients this dataset
declares are the ones the upstream source attributes to Goulard and Primakoff, PRC 10 (1974) 2034 —
the same functional form. The dataset declares no uncertainty on fallback values at all, so these remain
the only published indication of scale there is, and they should be read as a floor rather than as
an error bar.

**A negative result is a model failure, not a domain error.** The domain above constrains the
*input*. Inside it the formula still returns λ_c < 0 on 6325 of the 36000 swept points (F-1). This
dataset's contract is that a conforming evaluation **reproduces** that — a parity consumer must get
the same negative number, not a clamped or corrected one — while a consumer *using* the value should
treat λ_c ≤ 0 as unusable rather than as a rate. Geant4 does neither: see F-1 for what it does
instead.

## 3. Two disclosures about the shipped tables

**The capture records are re-ordered, and nothing moved.** The `.g4dat` grammar requires records
ascending by `(Z, A)`. Geant4's array is sorted by Z *alone*, and exactly one record sits out of
place: at Z=92 the source declares A in the order 238, 233, 235, 236, so
`{92, 238, 12.592, 0.035}` is declared before `{92, 233, 14.27, 0.15}` and before the two records
that follow it — **one misplaced record, three inverted pairs**. This dataset is canonically
sorted, so its record order is not upstream's. Two tests hold that equivalence up: one compares the
record **multiset**, and one reimplements Geant4's actual lookup — a linear scan with the early exit
`if (capRates[j].Z > Z) break;` — over the source order and requires it to agree with a keyed lookup
over the sorted order at every point of a 36000-point box. The canonical order is a refinement of
"sorted by Z", so the early exit fires at the same Z; the test is what makes that an argument rather
than a hope.

**`zeff[0]` ships and is unreachable through `G4MuonMinusBoundDecay`.** The array holds 101 entries
and its first is `0.`, but `G4MuonMinusBoundDecay::GetMuonZeff` clamps its argument into `[1, 100]`
before indexing, so element 0 can never be returned. The second compiled-in copy,
`G4MuonicAtomHelper::GetMuonZeff`, clamps differently and can return it.
It is shipped anyway, because "101/101 bit-identical" means the array *as declared*, and a
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

**A parity claim is a claim about a named build**, and this one names it: Ubuntu 26.04 LTS (WSL2),
x86_64, `g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0`, Geant4 11.4.2 `RelWithDebInfo` (`-O2 -g -DNDEBUG`),
no `-ffp-contract` setting, FMA absent from the baseline ISA. F-3 explains why that qualifier is not
decoration.

The second build the patches serve is named the same way: Ubuntu 26.04 LTS (WSL2), x86_64,
`g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0`, Geant4 `v11.5.0.beta` `RelWithDebInfo` (`-O2 -g -DNDEBUG`),
no `-ffp-contract` setting, FMA absent from the baseline ISA; the full sweep through its patched
library, with the opt-in on and `G4MUONICDATA_PROFILE` unset, reproduced the digest
`d1_gp_sweep.oracle` records, and a test holds the fallback expression compiled into that revision
token-identical to the one the model above reproduces.

## 5. Findings — registered, disclosed, and deliberately not fixed

These are outputs of building this dataset, not obstacles to it. A parity profile's contract is to
reproduce Geant4 *including* its defects; fixing anything here would break the one property the
dataset exists to have. Everything below was measured against the pinned revision.

**F-1 — the Goulard–Primakoff fallback returns negative capture rates.** 6325 of the 36000 swept
points return λ_c < 0. The boundary tracks neutron excess: the first negative A is 3 for Z=1, 6 for
Z=2, 8 for Z=3, 17 for Z=6, 23 for Z=8, 77 for Z=26 and 245 for Z=82. For hydrogen it is **³H**, a
legal Geant4 target and one of direct interest to muon-catalyzed fusion, at
λ_c = −2.870050e−08 ns⁻¹.

**How much of that region holds bound nuclides is not established here, and an earlier revision of
this section overstated the answer.** It said that for Z ≳ 6 the negative region lies beyond the
neutron drip line and is unreachable in practice. That claim carried no source, and this dataset
still cannot supply one: deciding which `(Z, A)` in the negative region are bound nuclides needs
evaluated nuclear masses across that region — separation energies, or a drip-line systematic. The
composition table this dataset does cite, for isotopic abundances (section 6), is not that: it
tabulates relative atomic masses but no neutron mass, and its nuclide list does not reach the
thresholds reported above for iron or lead, so it cannot settle bound-ness across the region. The
measurement — the per-Z threshold — stands; the reachability claim is withdrawn as **unsupported**,
rather than answered in either direction, and is
registered as a question for whichever layer of this program next carries evaluated nuclear data.

Read at the two call sites, `G4MuonMinusBoundDecay::ApplyYourself` and
`G4MuonicAtomHelper::ConstructMuonicAtom`, the consequences are not cosmetic. Where `lambda = lambdac + lambdad`
and the capture branch is `G4UniformRand()*lambda < lambdac`, a small negative λ_c means the capture
branch can **never** be taken — capture is silently disabled rather than made rare. For the **5407**
swept points where λ_c is negative *and* exceeds the free-muon decay rate (4.5517e−04 ns⁻¹) in
magnitude, the total λ goes negative and `time = t − log(U)/λ` moves the muon's global time
**backwards**. And a muonic-atom
lifetime computed as `1/(lambdac + lambdad)` becomes negative.

**F-2 — degenerate inputs return non-finite rates, with no coded rejection.** `Z = 0` returns NaN
for any A; `A = 0` returns +inf; `Z = -1, A = 12` returns −5.947382e−07 — finite, negative, and
entirely plausible-looking. Nothing in the source rejects any of these. A value that looks like a
rate but is not one is worse than an error, because it propagates. This dataset's model therefore
declares its domain (Z ≥ 1, A ≥ 1) and requires a conforming consumer to report a domain error
there; the difference between that and what `G4MuonMinusBoundDecay` does is the finding.

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
69–71, 75–78, 84–89 and 91. That is a set equality, not an impression: every one of the 74 was read
off the page images, and **70 of them carry that table and page in `isotope_audit.csv`'s `locator`
column**, so the comparison can be repeated from the shipped files. The four that do not are
visible in the file rather than hidden by it: hydrogen and helium carry instead the locator of the
source that established *their* flag, because Geant4 carves them out to other papers, and the single
records of copper and strontium are unsettled, so they carry no locator at all and name their table
and page in their `evidence` field instead. Nothing here draws on an unnamed source.

**F-5 — `zeff[]`'s non-monotonicity is the primary's own. SETTLED against the primary.** The array
rises monotonically except at exactly two steps: Z=81→82 (34.21 → 34.18) and Z=82→83 (34.18 →
34.00), after which it resumes rising; the step *into* Z=81 is also anomalously large, +0.40 against
neighbours of +0.17 to +0.18. The primary's Table IV prints **81(34.21), 82(34.18), 83(34.0)** — so
the pattern is not introduced by Geant4's transcription; it is reproduced faithfully from the
source. More generally, **74 of the 101 `zeff` entries appear in the primary and every one of them
matches** — 65 in Table IV, and 9 in Table III (printed p. 24), whose first column is headed
`Z(Z_eff)` under the same footnote, *"Z_eff is taken from ref. 77"*; the remaining 27 (Z = 0, the
gaps above, and Z > 94) are in neither table, and the source comment attributes whatever it does not
take from this paper to "Ford and Wills … or interpolated". This dataset does
not decide whether the lead-region step is physical structure near the Z = 82 shell closure or an
artifact inherited from further upstream — the primary attributes its Zeff column to its reference
77, which it prints as *K.W. Ford and J.G. Wills, Nucl. Phys. 35, 1397 (1968)*, while the Geant4
comment names `Ford and Wills Nucl Phys 35(1962)295`: the same authors, journal and volume, cited
with a different year and page, which this dataset records rather than adjudicates. Neither citation
resolves the question. Not altered.

One measured fact bears on it. Table IV of that same paper prints the barium row as **"59(29.99)"**
— barium is Z = 56, and 29.99 sits correctly between caesium's 29.75 (Z = 55) and lanthanum's 30.22
(Z = 57). The argument does not rest on that ordering, which is the very regularity this finding
shows the table can break: **`59` is printed a second time three rows further down, as `59(30.53)`
against praseodymium, where Z = 59 belongs.** The same Z is printed against two elements, so one of
the two is a misprint, and the element column settles which: barium *is* Z = 56 and praseodymium *is*
Z = 59, so it is the barium row's printed Z that is wrong — a fact about the periodic table, not
about how Geant4 read the row. So
the table demonstrably contains at least one typographical error in its Z column, which raises rather
than settles the prior that the lead-region step is also one. Geant4 read that row correctly:
`zeff[56] = 29.99`.

The per-cell match is committed: `data/g4/d1/zeff_audit.csv` carries every `Z(Z_eff)` cell of both
tables as printed, with the barium row under the Z its element column gives it, a test holds each
cell to the shipped value, and the cells the primary underlines — its table note reads
*"When underlined it is an estimate"* — carry `unc_type: estimate` in `d1_zeff.prov.json`.

**F-6 — the declared fallback is documented, by the primary, to mispredict isotopic effects.** See
section 2 for the quotation and for the distinction it draws: **Ca, Cr, Ni, U and Pu** are where the
formula is reported to over-predict the isotopic spread, while **Cu, Sr, Br and Cl** are where the
authors say the *experiments* are too imprecise — or, for chlorine, questionable — to judge. Only
the first group is evidence about the model.

What makes it land here is which elements those are. Of the nine the sentence names, **six carry at
least one record this audit establishes as a separated isotope** — Cl, Cr, Ni, Br, U and Pu. The
three exceptions are instructive rather than incidental. **Ca** appears once, as a record the primary
prints with no mass number. **Sr** appears once too, and it is one of the records this dataset **cannot settle** —
`(38, 88)`, where the primary prints both Sr with no mass number and Sr-88 and the key matches either.
**Cu** appears once as well, and its record `(29, 63)` cannot be settled either: its value equals none
of the Total Capture Rate cells Table IV prints at Z=29 (section 6).

Sharper still: **the four records this dataset cannot settle sit at elements this sentence names.**

**Those are not the same uncertainty, and saying so would overstate it.** This dataset cannot settle
a row because the *key* fails to distinguish two printed entries, or because its value equals none of
the Total Capture Rate cells printed at its Z; the primary doubts the *measurements* themselves. The two are independent,
and the shipped audit shows it in both directions: **Cr** carries an open record yet sits in the half
of the sentence that uses its experiments as evidence *against* the formula, while **Cl** carries the
primary's doubt and both its records are settled here as separated isotopes. What is true is narrower
and still worth saying: the two overlap at Sr, Cu and Br, and they overlap because each open record
there sits beside a separated-isotope entry, which can only be there where separated-isotope work was
done — which is also where these authors had something to say about isotopic effects.

**F-7 — `(Z, A)` is a label on most rows, not a target specification.** For **41 of the 90 records**
the primary shows the measurement was made on an **element printed with no mass number**, not on the nuclide
the record's `A` names. In 15 of those, `A` is not even the element's most abundant nuclide: the
extremes are `(62, 150)`, where Sm-150 is **7.4 %** of natural samarium, and `(50, 119)`, where
Sn-119 is **8.6 %** of natural tin. For `(30, 66)` and `(32, 72)` the `A` is neither the rounded
standard atomic weight nor the most abundant nuclide — rounding **half up**, a convention that
matters exactly once here: dysprosium's standard atomic weight is **162.500**, an exact tie, so
`(66, 163)` counts as matching its rounded weight, and would not under round-half-to-even.

The consequence is that a reasonable-looking call silently changes branch. A consumer asking for the
*actual* dominant isotope — Pb-208, say, which is 52 % of natural lead where this table keys Pb at
A=207 — **misses the table entirely** and falls through to `goulard_primakoff`: a formula the
primary itself says does not account correctly for isotopic effects (F-6), carrying no declared
uncertainty, in place of a measured value that was sitting one mass number away.

**Be precise about how bad that is, because it varies.** For the Pb-208 case the fallback is merely
*inaccurate*: F-1's first negative A at Z=82 is 245, far above 208, so the returned rate is a
plausible number of the right sign. The negative-rate region is a hazard for **light, neutron-rich**
requests, not for this one — and conflating the two would overstate the finding. The genuine
overlap is at low Z, where F-1's thresholds are small: A=3 at Z=1, A=6 at Z=2, A=8 at Z=3.

This dataset changes no key — it reproduces the table — but it now says, per row, which reading is
which, so a consumer can tell a measured isotope from an element wearing an isotope's label before
deciding whether a miss matters.

**F-8 — the aluminium record averages a pair of rows that print the same mean life and different rates.** The
comment above the compiled-in capture table describes its records as the *"weighted average of the
two most precise measurements"*. For aluminium, Table IV of the primary (p.2218) prints four rows;
the pair with the smallest rate uncertainties is the authors' own, 0.7054 ± 0.0013, and that of the
primary's ref. 42, 0.662 ± 0.003, and their inverse-variance weighted mean, at the record's printed
precision, is the record `(13, 27)`, 0.6985 ± 0.0012.

Both rows of that pair print a mean life of 864 ns. The primary's Eq. (4) sets the inverse of the
negative muon's mean life equal to the capture rate plus the Huff factor Q times the inverse of the
positive muon's mean life τ_μ⁺, and its aluminium rows carry a single Q, 0.993, so a single τ_μ⁺
gives both rows of the pair a single rate. Solved for τ_μ⁺ with that Q, the authors' row gives
2197 ns and the ref. 42 row 2004 ns; the primary gives each rate as the authors who
measured it calculated it, most often with their own value for the μ⁺ lifetime. The
`mizuno2025` value for the same key (section 9) differs from the authors' own value by less than its
printed uncertainty and from the record by more. The `parity` profile reproduces the record, as it
reproduces every compiled-in value; this dataset does not decide how the ref. 42 row's rate was
obtained.

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
carry `true`.** Of the settled rows, **45 are `isotope_resolved: true`**: 23 because the
primary lists a separated isotope with that mass number (the `(17, 35)` record among them settled to
that entry by the comparison below), 19 because the element is mononuclidic and so a
target of that element *is* a single nuclide, and 3 — the hydrogen and helium records — because
the sources the table carves them out to describe an isotopically distinct target
(deuterium-depleted protium; ³He and ⁴He tabulated as separate nuclides). The remaining **41 are
`false` as an established finding**, because the primary shows the value rests on an element
printed with no mass number (F-7).
For `(82, 207)` the value equals, at the precision it is printed with, the inverse-variance weighted mean
of the row of lead printed with no mass number and a row of radiogenic lead (section 11): a reconstructed
combination, not a demonstrated history.

**Two different sources are doing work here, and only one of them is the primary.** Whether a
measurement was made on a separated isotope or on an element printed with no mass number is read from the primary. But
whether an element *is* a single nuclide — the 19 mononuclidic calls above — and every
isotopic abundance this document quotes, F-7's two extremes and its comparisons against the rounded
standard atomic weight among them, come from the NIST *Atomic Weights and
Isotopic Compositions* table (`NISTIsotopicCompositions` in `references.bib`). The primary states no
natural isotopic abundance: its tables carry no abundance column, and the composition it prints in a
Table IV footnote (printed p. 28) is what that footnote labels `RPb`, radiogenic lead — not an
element's standard isotopic composition. So nothing in it
establishes a natural abundance or a mononuclidic element, and these calls cannot rest on it.

Every row's `evaluation_method` now states which of those routes produced its flag and carries the
evidence itself, so a reader meets the reasoning together with the boolean rather than a bare one.
Its `source_locator` gains a second, labelled clause naming the table and page that established the
flag and which copy of the paper was read; the first clause still points at the vendored source
line, because that is where the *value* comes from and the two questions must not be conflated. The
audit is shipped as `data/g4/d1/isotope_audit.csv` — one row per record, hand-authored, reviewable
and diffable.

**The unsettled rows say so with an empty locator, and they are worth naming.**
The rows the audit had left open were compared with every Total Capture Rate cell the primary
prints at their Z in Table IV, at the primary's printed precision, from a committed transcription
of those cells (`data/g4/d1/capture_rate_cells.csv` — with the primary's parentheses, whose
footnote reads *"Numbers are not given in original reference"*, and its `Refs.` column); a row
whose value and uncertainty equal a printed cell and no other is settled to that entry, which is
how the `(17, 35)` record became `isotope_resolved`, and the rows no cell equals stay open, unless the
published article settles them, with the comparison recorded in their `evaluation_method`:

| record | compiled-in value +- unc | Total Capture Rate cells the primary prints at this Z | equal at the printed precision |
|---|---|---|---|
| (24, 52) | 3.465 +- 0.026 | Cr 3.24 +- 0.08 [39]; Cr 3.33 +- 0.06 [48]; Cr 3.472 +- 0.031 [*]; Cr-50 3.825 +- 0.05 [72]; Cr-52 3.452 +- 0.047 [72]; Cr-53 3.297 +- 0.045 [72]; Cr-54 3.057 +- 0.042 [72]; Cr-nat 3.444 +- 0.047 [72] | no cell |
| (29, 63) | 5.691 +- 0.030 | Cu 5.79 +- 0.16 [39]; Cu 5.47 +- 0.20 [44]; Cu 5.66 +- 0.09 [53]; Cu 5.67 +- 0.09 [48]; Cu 5.676 +- 0.037 [*]; Cu-63 5.72 +- 0.05 [48] | no cell |
| (35, 79) | 7.223 +- 0.28 | Br-79 7.05 +- 0.35 [54]; Br-81 7.55 +- 0.48 [54]; Br 7.07 +- 0.06 [*] | no cell |
| (38, 88) | 6.93 +- 0.12 | Sr 7.25 +- 0.14 [48]; Sr 7.02 +- 0.14 [*]; Sr-88 6.61 +- 0.27 [48] | no cell |

The `(29, 63)` and `(35, 79)` records are compared too, although the listing alone would settle each to
the cell of its separated isotope: the value and uncertainty of each agree, to within a unit of the last
printed digit, with the inverse-variance weighted mean of two printed rows, Cu printed with no mass number
and Cu-63, and Br-79 and Br-81, a reconstructed combination, not a demonstrated history
(`isotope_audit.csv`), and no cell equals either, so both stay open.

The preprint's Table IV prints U-233, U-235 and U-238 cells and no U-236 cell; the published article's Table IV prints a 236U cell, and the compiled-in `(92, 236)` pair equals it alone at the printed precision (`suzuki1987_parity_cells.csv`), so that row is settled to it and its locator cites the published copy.

**What changed, and why the previous rule was not enough.** The earlier release derived the flag
mechanically: `true` if and only if the row's Z carried more than one capture record. Its soundness
argument was about the *Z* — two differing rates at one Z do show the underlying data distinguishes
isotopes — and it was then applied to each *row* of that Z, which does not follow, since one of
those rows can still be the natural-composition entry. The audited flag differs from that rule on
**26 of the 90 records**, and the three ways it differs are not the same claim:

* **22** the rule called unresolved that the primary establishes as resolved.
* **2** the primary flatly contradicts: it lists "C" and "C-13", and "O" and "O-18", so `(6, 12)`
  and `(8, 16)` are carbon and oxygen printed with no mass number, not C-12 and O-16.
* **2** — `(24, 52)`, `(35, 79)` — where the primary does not contradict the rule but
  **fails to establish the question it was answering**, so the flag falls back to "not established"
  and `needs_verification` is `true`. Counting these as "the rule was wrong" would overstate it;
  counting them as agreement would hide an open question.

Not every unsettled row appears in that list, which is not an oversight:
`(29, 63)` and `(38, 88)` are the only capture records at Z=29 and Z=38, so the old rule called them
unresolved as well, and rule and audited flag agree. The open rows and the disagreements are different sets and do
not line up row for row.

The `zeff` rows remain `false` throughout, as a fact rather than a default: an effective charge is a
per-Z quantity, so there is no isotope for it to be resolved to.

**Two limits on all of the above, stated plainly.** First, what was read is the TRIUMF preprint
TRI-PP-87-5 (January 1987), a scanned copy, except where the published article settles a row the preprint leaves open; every locator for a row
attributed to Suzuki cites the pagination of the copy its `copy_read` names, and if a value's
evidence differs between the two copies this audit is the one that must move. A row attributed to
another primary carries that primary's locator and copy, and a row the audit leaves unsettled
carries its evidence in place of a locator. Second, this is a resolution audit, not an
evaluation: no shipped value was compared to the primary for correctness, and none was altered.

## 7. Consuming the dataset from Geant4

`cpp/patches/` carries a patch against each of the two Geant4 revisions, `v11.4.2` and
`v11.5.0.beta`, that adds the reader to Geant4
and lets both compiled-in copies of the capture tables consult this dataset after an explicit opt-in,
with the default behaviour left untouched. A separate registration patch for each revision registers the dataset so that
Geant4 resolves it under `GEANT4_DATA_DIR`; without it, an exported `G4MUONICDATA` is the only route.
The generator builds `G4MuonicData.<version>.tar.gz` in memory from every table file, their provenance
files, a `README` and a `History`, and the registration snippet carries that archive's MD5; the archive
unpacks to the `G4MuonicData<version>` directory the registered mode looks for, and `<version>` is the
`#VERSION` every table carries. The patched build reads each seam through a profile of its own —
`G4MUONICDATA_D1_PROFILE` for nuclear capture and `G4MUONICDATA_D3_PROFILE` for the muonic
transitions, `G4MUONICDATA_PROFILE` for both where neither is set, and `parity` for capture where
none is — and never through another; `cpp/patches/README.md` states what a token no file carries
does, and what the reserved token `compiled` does.
An application opts in with one line before its physics list is built,
`G4HadronicParameters::Instance()->SetEnableMuonicData(true);`, as `cpp/patches/README.md` states.
`cpp/patches/README.md` states what the patches change, how the dataset is found at run time, and
what a patched build was measured to do.

## 8. Licensing

The `parity` tables `data/g4/d1/d1_capture.g4dat` and `data/g4/d1/d1_zeff.g4dat`, which reproduce
values compiled into Geant4, and their Layer-2 files `data/g4/d1/d1_capture.prov.json` and
`data/g4/d1/d1_zeff.prov.json` are offered under the Geant4 Software License v1.0, the licence of the
Geant4 source in `third_party/geant4/` they are generated from.
`data/g4/d1/geant4_natural_isotopes.csv`, the natural isotopic compositions read from Geant4 builds, is
offered under the Geant4 Software License v1.0.
`data/g4/d1/d1_decay_rate.oracle`, the decay rates and nuclear masses read from Geant4 builds, is offered
under the Geant4 Software License v1.0.
The files that restate cells of the primaries' tables, `data/g4/d1/capture_rate_cells.csv`,
`data/g4/d1/zeff_audit.csv`, `data/g4/d1/suzuki1987_*.csv` and `data/g4/d1/mizuno2025_table*.csv`,
and the `suzuki1987` and `mizuno2025` profiles generated from them,
`data/g4/d1/d1_capture.suzuki1987.*` and `data/g4/d1/d1_capture.mizuno2025.*`, and the comparison that
restates their mean lives, `data/g4/d1/mean_life_comparison.csv`, are offered under no licence of this
repository.
The other files in `data/g4/d1/` are offered under CC-BY-4.0.
Values compiled into Geant4 that another file of the dataset repeats are offered under the Geant4
Software License.
Values from third-party sources are facts: the project asserts no rights over them and takes no
position on the sources' rights.
The toolchain is Apache-2.0; the patch files under `cpp/patches/` are offered under the Geant4
Software License, as `cpp/patches/README.md` states.

## 9. A second profile of the capture table — `mizuno2025`

Sections 1–8 describe the `parity` tables; this section describes a third file in the same
directory, `d1_capture.mizuno2025.g4dat` with its `d1_capture.mizuno2025.prov.json`, which carries
the capture table under `#PROFILE mizuno2025`. Its values are the muon capture rates that Mizuno et
al., Phys. Rev. C **112**, 024307 (2025), doi 10.1103/vl7z-rzp8, arXiv:2501.05897 print in the
`Exp.` column of their Table 3, read from committed transcriptions of that paper's Table 1 and
Table 3 that ship in the same directory. Every value is carried as the primary prints it: the
primary computes each rate from its measured lifetime by its Eq. (3) with the Huff factors it takes
from Suzuki, Measday & Roalsvig (1987), and nothing is re-derived here. Where the primary measured
a nuclide on more than a single target, its Table 3 value is the primary's own average of its
Table 1 rows, marked there by a footnote that the row's provenance quotes.

The keys follow the primary's labels. An enriched isotope or a mononuclidic element is keyed by its
mass number and is `isotope_resolved`; a natural-composition target is keyed by `A = 0`, which the
table admits with `A:natural_and_listed`, and is not.
The file declares no `#FALLBACK`, and, because it reproduces no
upstream revision, no `#SOURCESHA`.

**Where the profiles disagree.** The value and its uncertainty
were compared at the primary's printed precision, a natural-composition key against the compiled-in
row of that element.
The keys that
partner nothing and are not compared are `14-29` and `14-30`.
The test suite derives from the shipped files the pairs whose value or
uncertainty differs and requires this table to list exactly those pairs, the `parity` cells as the
vendored source prints them and the `mizuno2025` cells as the transcription prints them:

| Z | A (mizuno2025) | A (parity) | parity value ± unc | mizuno2025 value ± unc | locator |
|---|---|---|---|---|---|
| 12 | 0 | 24 | 0.4823 ± 0.0013 | 0.4856 ± 0.0018 | Table 3, Exp. column |
| 13 | 27 | 27 | 0.6985 ± 0.0012 | 0.7059 ± 0.0013 | Table 3, Exp. column |
| 14 | 0 | 28 | 0.8656 ± 0.0015 | 0.8794 ± 0.0018 | Table 3, Exp. column |
| 14 | 28 | 28 | 0.8656 ± 0.0015 | 0.893 ± 0.009 | Table 3, Exp. column |
| 25 | 55 | 55 | 3.900 ± 0.030 | 3.90 ± 0.08 | Table 3, Exp. column |
| 47 | 0 | 107 | 10.869 ± 0.095 | 11.07 ± 0.07 | Table 3, Exp. column |

No value in either profile is altered by the comparison. The copy read is the arXiv v2 HTML; the
journal version is unread.

## 10. Published capture-rate tables — `suzuki1987`

`data/g4/d1/suzuki1987_printed_rows.csv` records each printed line of Tables III and IV of the published Suzuki, Measday and Roalsvig article from its page images, with table, printed page and row locators.
Of a caption, the ledger keeps the table's title; of a note or footnote of more than a sentence, only its footnote marker, if any; in place of the text it leaves out, it gives the SHA-256 of that text.
The generated `suzuki1987_quantity_rows.csv` normalizes the data rows and carries the Huff factor column as printed.
The `suzuki1987` capture profile selects this experiment's own rows when the printed target label identifies an isotope or natural composition, or is the bare symbol of an element with a single natural isotope, the rate has a symmetric uncertainty, and the rate is not parenthesized.
A bare element symbol stays `unspecified` under this project's source-preserving convention, which records natural composition only where the printed label carries the `nat` prefix; the format itself makes an `A = 0` record the natural-composition row only under `A:natural_and_listed`.
A row of an element with a single natural isotope is keyed by that isotope's mass number, which the isotope audit of section 6 gives the element from the NIST table and which is the single isotope `geant4_natural_isotopes.csv` records for it, as `mizuno2025` keys its aluminium (section 9).
For F-19 the tabulated mean life and capture rate describe the lower hyperfine state (Table III p.2217 note); both Geant4 routes use that rate as a single-state approximation and do not model the hyperfine transition.
`suzuki1987_parity_cells.csv` lists every compiled-in capture record at a Z the published tables print and the published cells equal at their printed precision.
`suzuki1987_preprint_differences.csv` lists differences against the committed preprint capture and effective-charge cells.
Its `not_in_committed_preprint_cells` kind means a published capture cell at a covered Z has no paired committed preprint cell.
Neither comparison changes a profile value.

The table lists the paired keys whose value or uncertainty differs at the published printed precision:

| Z | A (suzuki1987) | A (parity) | parity value ± unc | suzuki1987 value ± unc | locator |
|---|---|---|---|---|---|
| 3 | 6 | 6 | 0.004647 ± 0.00012 | 0.004180 ± 0.000450 | Table III p.2217 row 14 |
| 3 | 7 | 7 | 0.002229 ± 0.00012 | 0.001810 ± 0.000440 | Table III p.2217 row 17 |
| 4 | 9 | 9 | 0.006107 ± 0.00019 | 0.0074 ± 0.0005 | Table III p.2217 row 21 |
| 5 | 10 | 10 | 0.02757 ± 0.00063 | 0.0278 ± 0.0007 | Table III p.2217 row 23 |
| 5 | 11 | 11 | 0.02188 ± 0.00064 | 0.0219 ± 0.0007 | Table III p.2217 row 25 |
| 6 | 13 | 13 | 0.03474 ± 0.00034 | 0.0376 ± 0.0007 | Table III p.2217 row 39 |
| 11 | 23 | 23 | 0.3773 ± 0.0014 | 0.3772 ± 0.0014 | Table IV p.2218 row 8 |
| 13 | 27 | 27 | 0.6985 ± 0.0012 | 0.7054 ± 0.0013 | Table IV p.2218 row 16 |
| 15 | 31 | 31 | 1.1681 ± 0.0026 | 1.185 ± 0.003 | Table IV p.2218 row 23 |
| 25 | 55 | 55 | 3.900 ± 0.030 | 3.857 ± 0.037 | Table IV p.2219 row 3 |
| 27 | 59 | 59 | 4.945 ± 0.025 | 4.940 ± 0.029 | Table IV p.2219 row 10 |
| 33 | 75 | 75 | 6.094 ± 0.037 | 6.104 ± 0.043 | Table IV p.2219 row 34 |
| 41 | 93 | 93 | 10.38 ± 0.11 | 10.36 ± 0.17 | Table IV p.2219 row 48 |
| 53 | 127 | 127 | 11.276 ± 0.098 | 11.58 ± 0.22 | Table IV p.2220 row 14 |
| 79 | 197 | 197 | 13.35 ± 0.10 | 13.07 ± 0.28 | Table IV p.2220 row 38 |
| 83 | 209 | 209 | 13.238 ± 0.065 | 13.10 ± 0.18 | Table IV p.2220 row 52 |

## 11. Natural-composition targets

Geant4 picks the nucleus that captures a stopped negative muon by drawing the element, then drawing an
isotope of it with the probabilities of the element's relative abundance vector
(`G4ElementSelector::SelectZandA`), and it asks `GetMuonCaptureRate(Z, A)` about that nucleus alone.
`data/g4/d2/selector_call_sites.csv` lists both callers of the selector in each revision:
`G4HadronStoppingProcess`, from which derives the `G4MuonMinusCapture` that the reference physics lists
QBBC and FTFP_BERT give the negative muon (`data/g4/d2/selector_bindings.csv`), and
`G4MuonMinusAtomicCapture`.
The compiled-in table answers only a nucleus whose Z and A both match a record, and the
Goulard–Primakoff formula answers every other (sections 1 and 2), so a target of natural composition
reaches an element's record only on the stops that draw the record's own mass number.
For an element `G4NistManager` builds, that vector is the natural composition Geant4 stores for it;
Geant4's `G4NistElementBuilder.cc` attributes its element data to the NIST database of atomic weights
and isotope compositions.
An element built from explicit `G4Isotope` objects, an enriched target for instance, carries the
composition its builder gave it instead, and nothing in this section applies to it.

`cpp/tools/harvest_natural.cc` prints the isotopes and relative abundances stored in the `G4Element`
that `G4NistManager` builds for every Z of the `parity` capture table's validity range, which are the
vectors the selector reads.
Unpatched builds of `v11.4.2` and `v11.5.0.beta` printed the same bytes; `geant4_natural_isotopes.csv`
commits the compositions, and `natural_harvest_builds.csv` names the builds and the SHA-256 of the driver.
`scripts/generate_g4natural.py`, a step of `make audit`, writes `natural_target_capture.csv`: for each
profile of the capture table and each element, the shares of stops that the profile's own rows, the
compiled-in table and the formula answer, the share whose rate is not a positive finite rate, the
abundance-weighted capture rate, and the abundance-weighted spread of the isotopes' capture rates
relative to it.
Each isotope's rate follows the lookup a patched build runs: under `parity`, whose table holds the
compiled-in records, the compiled-in chain, which the reference implementation of section 2 reproduces
bit for bit at every isotope of these compositions (section 4); under any other profile, the profile's row
keyed by exactly that nuclide, and the compiled-in chain where the profile has none
(`cpp/patches/README.md`).
Where the isotope audit of section 6 establishes that an element's compiled-in record rests on a
measurement of the element printed with no mass number, the row also gives that record and the relative difference
between the weighted rate and the record's rate.
That difference is an abundance-weighted first-order estimate against the measurement the table itself
encodes; for lead that value matches a reconstructed combination of a measurement on lead and one on
radiogenic lead (see below).

It is first-order because Geant4 samples each stop's time from the drawn nucleus's own capture and decay
rates (`G4MuonMinusBoundDecay::ApplyYourself`, and the lifetime
`G4MuonicAtomHelper::ConstructMuonicAtom` gives the muonic atom), so a natural target decays as a
mixture of exponentials, an exponential per isotope weighted by its abundance.
The rate of an exponential fitted to that mixture differs from the mixture's initial rate, the
abundance-weighted sum of the isotopes' capture and decay rates, only at second order in the spread of
those rates; `relative_spread` gives the spread of the isotopes' capture rates, relative to the weighted
capture rate.
The decay rate that competes with capture depends on Z alone in `G4MuonicAtomHelper::GetMuonDecayRate`,
and in `G4MuonMinusBoundDecay::GetMuonDecayRate` wherever Z ≥ 14; below that, the latter also
reads the drawn nucleus's mass.
Wherever the decay rate does not depend on the isotope drawn, the weighted rate and the record's rate
differ by the same amount as the mixture's initial rate and the disappearance rate, capture plus decay,
of a target the record answered on every stop; as a fraction of those larger rates, the difference is
smaller.

Of the 41 elements with a natural-composition record, the weighted rate differs from the
record's rate by more than the record's tabulated relative uncertainty in 25 of them; the table
gives the `parity` rows, each figure rounded to a tenth of a percent.

| Z | element | record A | stops the table answers (%) | relative difference (%) | tabulated relative uncertainty (%) |
|---|---|---|---|---|---|
| 6 | C | 12 | 100.0 | -0.1 | 0.8 |
| 7 | N | 14 | 99.6 | -0.1 | 0.8 |
| 8 | O | 16 | 100.0 | 0.0 | 0.6 |
| 10 | Ne | 20 | 90.5 | -3.0 | 2.0 |
| 12 | Mg | 24 | 79.0 | -6.0 | 0.3 |
| 14 | Si | 28 | 92.2 | -2.2 | 0.2 |
| 16 | S | 32 | 94.9 | -1.5 | 0.2 |
| 18 | Ar | 40 | 99.6 | +0.1 | 5.1 |
| 19 | K | 39 | 93.3 | -1.5 | 0.3 |
| 20 | Ca | 40 | 96.9 | -1.1 | 0.3 |
| 22 | Ti | 48 | 73.7 | -3.4 | 0.4 |
| 23 | V | 51 | 99.8 | 0.0 | 0.7 |
| 26 | Fe | 56 | 91.8 | -0.6 | 0.5 |
| 30 | Zn | 66 | 27.9 | -5.9 | 0.5 |
| 31 | Ga | 69 | 60.1 | -8.8 | 1.1 |
| 32 | Ge | 72 | 27.5 | -8.2 | 0.6 |
| 34 | Se | 80 | 49.6 | -2.0 | 0.5 |
| 37 | Rb | 85 | 72.2 | -4.1 | 2.0 |
| 40 | Zr | 91 | 11.2 | -2.9 | 0.6 |
| 42 | Mo | 96 | 16.7 | +0.5 | 0.7 |
| 46 | Pd | 106 | 27.3 | +3.2 | 0.7 |
| 47 | Ag | 107 | 51.8 | +0.2 | 0.9 |
| 48 | Cd | 112 | 24.1 | 0.0 | 0.9 |
| 49 | In | 115 | 95.7 | +0.3 | 1.0 |
| 50 | Sn | 119 | 8.6 | -0.4 | 1.0 |
| 51 | Sb | 121 | 57.2 | -2.5 | 1.2 |
| 52 | Te | 128 | 31.7 | -0.4 | 0.8 |
| 56 | Ba | 138 | 71.7 | +2.8 | 0.8 |
| 57 | La | 139 | 99.9 | 0.0 | 0.9 |
| 58 | Ce | 140 | 88.5 | -0.4 | 0.8 |
| 60 | Nd | 144 | 23.8 | +3.5 | 1.1 |
| 62 | Sm | 150 | 7.4 | +4.4 | 1.4 |
| 64 | Gd | 157 | 15.7 | +1.7 | 1.1 |
| 66 | Dy | 163 | 24.9 | +2.0 | 1.5 |
| 68 | Er | 167 | 22.9 | +0.1 | 2.1 |
| 72 | Hf | 178 | 27.3 | +1.2 | 1.6 |
| 73 | Ta | 181 | 100.0 | 0.0 | 1.0 |
| 74 | W | 184 | 30.6 | +5.0 | 1.3 |
| 80 | Hg | 201 | 13.2 | +12.5 | 1.4 |
| 81 | Tl | 205 | 70.5 | +3.1 | 1.2 |
| 82 | Pb | 207 | 22.1 | +5.9 | 0.5 |

The `parity` record for lead equals, at the precision it is printed with, the inverse-variance weighted
mean of the row of lead printed with no mass number and the row of what a footnote of that table labels
radiogenic lead, which Table IV of Suzuki, Measday and Roalsvig gives from the same reference
(`suzuki1987_printed_rows.csv`), so the Pb row's relative difference is measured against a value that
matches a reconstructed combination of a measurement on lead and a measurement on radiogenic lead, not
a demonstrated history of how the record was obtained.

This section does not revisit the rule that, under every profile but `parity`, an element's
natural-composition row answers only a request that names it; the next table gives the rows of the
other profiles with a natural-composition record whose weighted rate differs from the `parity` row's.

| profile | Z | element | stops the profile's rows answer (%) | relative difference (%) |
|---|---|---|---|---|
| `mizuno2025` | 14 | Si | 100.0 | +2.1 |
| `suzuki1987` | 6 | C | 1.1 | 0.0 |

A transported nucleus always carries a mass number, so it can reach the rows of a profile keyed by a mass number: 90 of the 90 rows of `parity`, 5 of the 8 rows of `mizuno2025` and 19 of the 19 rows of `suzuki1987`.

No row of `natural_target_capture.csv` has a positive `nonpositive_share`: under an evaluated profile,
no isotope of an element `G4NistManager` builds in this Z range gets a computed capture rate that is not
a positive finite rate, the rates `G4MuonicDataOverlay::CheckComputedCaptureRate` acts on under such a
profile (`cpp/patches/README.md`).
A Geant4 bug report (2013), <https://bugzilla-geant4.kek.jp/show_bug.cgi?id=1457>, reported incorrect
capture rates for nuclei the lookup table does not list and plotted the rate averaged over the isotopes
with their abundances against Z; it is recorded as resolved and fixed.

## 12. Measured mean lives

A negative muon in the ground state of a muonic atom disappears by nuclear capture or by decay, and the
inverse of the mean life a lifetime measurement gives is the capture rate plus the free muon's decay rate
times the Huff factor, which accounts for the binding (Suzuki, Measday & Roalsvig, Eq. (4)).
Geant4 computes that disappearance rate on two routes, each with its own decay rate.
`G4MuonMinusBoundDecay::ApplyYourself` draws the time of a stop from the sum of `GetMuonCaptureRate(Z, A)`
and `G4MuonMinusBoundDecay::GetMuonDecayRate(Z, A, muMass, nuclMass)`, and runs in `G4MuonMinusCapture`,
the process the reference physics lists QBBC and FTFP_BERT give the negative muon at rest
(`data/g4/d2/selector_bindings.csv`).
`G4MuonicAtomHelper::ConstructMuonicAtom` gives a muonic atom the inverse of the sum of
`GetMuonCaptureRate(Z, A)` and `G4MuonicAtomHelper::GetMuonDecayRate(Z)` as its lifetime, which
`G4MuonicAtomDecay` reads; a user reaches that route by registering `G4MuonMinusAtomicCapture` for the
negative muon and the `G4MuonicAtomDecayPhysics` constructor, as `cpp/tools/harvest_d2.cc` does for its
`QBBC+helper` list.
From the threshold of section 11 on, the first decay rate is a function of Z that its source describes as
based on a fit to the data referenced in Suzuki, Measday & Roalsvig.
For every Z this section compares, the second multiplies the free muon's decay rate by the difference
between unity and 2.5 times the square of the product of the effective charge and the
fine-structure constant, the small-Z approximation its source attributes to Mukhopadhyay.
The patches leave both decay functions as they are; the helper's reads the effective charge through
`GetMuonZeff`, which a patched build with the dataset enabled under `parity` answers from the `muon_zeff`
table, equal to the compiled-in charge at every Z compared here.

`cpp/tools/harvest_decay.cc` prints both decay rates, with the nuclear mass `G4Nucleus::AtomicMass(A, Z)`
that `ApplyYourself` passes the first, from unpatched builds of `v11.4.2` and `v11.5.0.beta`, which printed
the same bytes; `d1_decay_rate.oracle` commits them, and `openmucf/g4/d1_lifetime.py` reproduces every decay
rate it holds bit for bit, evaluated with the oracle's constants and nuclear-mass inputs.
`scripts/generate_g4lifetime.py`, a step of `make audit`, writes `mean_life_comparison.csv` from the mean
lives each source printed for its own measurements: the 57 rows of Tables III and IV of Suzuki,
Measday and Roalsvig marked as the authors' own (section 5, F-4) that print a capture rate beside the mean
life, and the 10 rows of Table 1 of Mizuno et al. (section 9).
The other rows of Tables III and IV give each experiment's capture rate as its authors calculated it, most
often with their own value of the positive muon's lifetime (p. 2216), so the comparison does not take them.
For each profile of the capture table and each of those rows, the file gives the measured rate, the inverse
of the mean life, and, when the row names a nucleus, each route's rate, capture plus decay, for it.
A row names the nucleus with the printed mass number, or, for a target printed with no mass number or as
natural, with the mass number of the record the isotope audit of section 6 establishes rests on a
measurement of the element printed with no mass number (for lead, a reconstructed combination with
radiogenic lead: section 11), or else of the single isotope of a natural composition with a single
isotope (section 11); its capture rate follows the lookup a patched build runs (section 11).
A row is compared when a measured value from the row's own source answers it, a profile's row for that
profile's source or the compiled-in table for Suzuki, Measday and Roalsvig, whom its source comment credits
(section 5, F-4), measured on the same kind of target, a nuclide or a natural composition, which are the
same thing for an element with a single isotope; `population` names the reason any other row is not.
`measured_state` names the state of the muonic atom that a note of the source says a row's mean life describes,
where the source names a state.
A route's rate is within the band when it differs from the measured rate by no more than three times the
root sum of squares of the measured rate's uncertainty and the capture rate's tabulated uncertainty; the two
can rest on the same mean life, so the band errs wide.
The comparison is of the rates the code returns for a single nucleus, not of a transported simulation of a
target; a target of natural composition decays as section 11 describes.
For F-19 the tabulated mean life and capture rate describe the lower hyperfine state (Table III p.2217 note); both Geant4 routes do not model the hyperfine transition.

| profile | rows compared | within the band, `G4MuonMinusBoundDecay` | within the band, `G4MuonicAtomHelper` |
|---|---|---|---|
| `parity` | 51 | 47 | 43 |
| `mizuno2025` | 54 | 51 | 48 |
| `suzuki1987` | 51 | 51 | 47 |

Under `parity` and `mizuno2025` each route leaves a compared row outside the band, and under `suzuki1987`
only `G4MuonicAtomHelper` does; where a row of `suzuki1987` answers a compared row, the lookup returns the
capture rate the source prints beside the mean life.
The next table gives those rows: the difference between the route's rate and the measured rate, and the part
of it that comes from the capture rate, the rate the lookup returns less the rate the source prints beside
the mean life, each in units of the band's standard deviation and rounded to a tenth.

| route | measurement | target | profiles | difference (σ) | capture part (σ) |
|---|---|---|---|---|---|
| `G4MuonMinusBoundDecay` | `suzuki1987` Table III p.2217 row 21 | Be | `parity`, `mizuno2025` | -3.1 | -2.8 |
| `G4MuonMinusBoundDecay` | `suzuki1987` Table III p.2217 row 39 | 13C | `parity`, `mizuno2025` | -4.2 | -3.6 |
| `G4MuonMinusBoundDecay` | `suzuki1987` Table IV p.2218 row 16 | Al | `parity` | -3.2 | -3.8 |
| `G4MuonMinusBoundDecay` | `suzuki1987` Table IV p.2218 row 23 | P | `parity`, `mizuno2025` | -4.6 | -4.5 |
| `G4MuonicAtomHelper` | `suzuki1987` Table III p.2217 row 21 | Be | `parity`, `mizuno2025` | -4.6 | -2.8 |
| `G4MuonicAtomHelper` | `suzuki1987` Table III p.2217 row 37 | C | `parity`, `mizuno2025`, `suzuki1987` | -4.7 | -1.5 |
| `G4MuonicAtomHelper` | `suzuki1987` Table III p.2217 row 39 | 13C | `parity`, `mizuno2025` | -6.1 | -3.6 |
| `G4MuonicAtomHelper` | `suzuki1987` Table III p.2217 row 44 | N | `parity`, `mizuno2025`, `suzuki1987` | -3.1 | -0.4 |
| `G4MuonicAtomHelper` | `suzuki1987` Table III p.2217 row 49 | O | `parity`, `mizuno2025`, `suzuki1987` | -3.3 | -0.2 |
| `G4MuonicAtomHelper` | `suzuki1987` Table IV p.2218 row 16 | Al | `parity` | -6.5 | -3.8 |
| `G4MuonicAtomHelper` | `suzuki1987` Table IV p.2218 row 20 | Si | `parity`, `suzuki1987` | -4.8 | -2.4 |
| `G4MuonicAtomHelper` | `suzuki1987` Table IV p.2218 row 23 | P | `parity`, `mizuno2025` | -6.1 | -4.5 |

On each row of the table for `G4MuonMinusBoundDecay`, the capture part is the larger part of the difference.
Of the 57 measurements compared under a profile, the decay rate of `G4MuonicAtomHelper` lies
below the measured rate less the printed capture rate for 54, and that of
`G4MuonMinusBoundDecay` for 29.
No capture rate or profile of this dataset changes with this comparison.
