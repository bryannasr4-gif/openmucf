# Changelog

All notable changes to OpenMuCF are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- Author: Bryan Nasr (ORCID: 0009-0008-2360-7522). -->
<!-- Repository: https://github.com/bryannasr4-gif/openmucf | Zenodo concept DOI 10.5281/zenodo.21251511 (v1.0.0 version DOI 10.5281/zenodo.21251512; v1.1.0 version DOI 10.5281/zenodo.21316574; v1.2.0 version DOI 10.5281/zenodo.22167360). -->

## [Unreleased]

### Added -- the claim guard's sentence layer (G4), first pack (2026-08-30)

The v1.2.0 notes disclosed that a universal whose quantifier and subject fall on different rendered
lines escaped the claim guard. This change closes that hole for eight of the fifteen prose paths a
sentence can wrap in; the second pack, already scoped, extends it to the remaining seven.

- **A wrapped claim sentence is keyed WHOLE.** `test_wrapped_claims_registered` cuts prose into
  units per file type -- string literals and comment runs read by token for `.py` (raw source text,
  so the guard hashes exactly what a reader of the file sees), blank-line-delimited blocks for
  `.md`/`.txt`/`.bib`, table rows standing alone, fenced code excluded -- splits the units into
  sentences, and requires every sentence that touches two or more source lines and matches both G1
  patterns to carry a ruled row in `tests/sentence_claims_registry.tsv`, with an UNREVIEWED cap of
  ZERO. An edit to ANY line of a wrapped claim now re-keys it -- including the line that matches
  nothing on its own, the exact edit that used to change a quantifier with no registry diff.
- **The segmentation is exampled and drilled the way the form tables are.** The 24 protected
  abbreviations (`eq.`, `et al.`, `p.`, ...) and 8 sentence-opener classes live in tables whose
  every member owns a fixture row in `tests/sentence_split_examples.tsv` that flips when the member
  is removed; a universal split across two lines is drilled end-to-end on synthetic files; and a G1
  form deletion goes red through this layer's stale rows too. `.json` and `.csv` claim paths are
  excluded because nothing in them can wrap -- a property `test_unwrappable_paths_cannot_wrap`
  measures on every run rather than assumes.
- **Covered now: 323 wrapped sentences, every one read and ruled, none UNREVIEWED**, across
  CHANGELOG.md, scripts/generate_mucost.py, MUON_COST.md, tests/test_ledger_claims.py, README.md,
  paper/paper.md, ADOPTERS.md and openmucf/data/bib_unresolved.txt.
- **Two forms enter the line guard on this change's own escape.** The coverage bullet above first
  stated its universal -- wrapped sentences, every one read and ruled -- in words no `LEDGER` form
  matched, so that claim was editable with every guard green. `sentence` and `sentences` join
  `LEDGER_FORMS` (a pair, the plural precedent), each with its fixture row, and every line and
  sentence the pair adds is read and ruled with the rest.
- **Wrapped arithmetic is now a named gap, not an unknown.** The G2 arithmetic guard matches one
  line at a time, so a written-out statement that wraps across two rendered lines is outside it.
  Five such statements exist at this head, across four files (MUON_COST.md carries two;
  SYSTEMS.md, openmucf/systems.py and scripts/generate_systems.py one each); each was recomputed
  by hand on 2026-08-30 and all five hold.
  Closing the gap is registered for a later change, not done here.
- **A published count is corrected: 731 -> 592.** The prototype behind the earlier figure walked an
  f-string and its fragments twice, compared f-string text carrying `{}` against source lines
  carrying the field expression (so 79 single-line strings counted as wrapped), and joined whole
  JSON/CSV files and markdown tables into pseudo-sentences. Re-derived on raw source slices with
  line spans, the same tree that the prototype counted at 731 holds 596 wrapped matching sentences
  under the guard as it ships here (592 before `sentence` and `sentences` entered).

### Changed -- the claim guard: forms made deletable-only-in-the-open, three lexical holes closed, and the figures brought inside it (2026-08-29)

The v1.2.0 notes named the claim guard's blind spots as a v1.3.0 deferral. This is the first of the
changes that close them, and it widens the guard rather than loosening it.

- **The two patterns are now built from form tables, and every form is exampled and drilled.**
  `tests/test_ledger_claims.py` used to hold `STRONG` and `LEDGER` as two hand-written alternations,
  and a form that happened to match no line of the tree could be deleted with nothing going red --
  thirteen of them could be, `headlines` among them, the very form a drill had forced in. The patterns
  are now built from `STRONG_FORMS` and `LEDGER_FORMS`; `tests/ledger_claims_examples.tsv` holds one
  sentence per form; `test_guard_forms_are_exampled` requires each form to be atomic, to own an
  example, and to be the only form on its side that matches it; and
  `test_deleting_any_form_breaks_its_example` removes each form in turn and requires its example to
  stop being a claim. A form can still be removed -- in a two-place diff that a reader sees.
- **Three lexical holes closed, every line they add read and ruled.** `exact` joins `STRONG`
  (5 lines; the `datapackage.json` descriptor it hid is true of all 14 rows of that resource). The
  plurals of `ledger`, `manifest`, `quotient` and `denominator` join `LEDGER` (1 line -- the Kovach
  et al. (2017) note in `references.bib`, checked clause by clause against the paper: eq. (1),
  the 10.4 % and 18.3 % denominators, the four subsystem draws). The ledger's own aggregate nouns --
  `ratio`, `spread`, `aggregate`, `median`, `box`, `edge`, with their plurals -- join `LEDGER`
  (71 lines, most of them the edge table's and the prior boxes' own prose). The registry grows
  856 -> 969 rows (144 EXERCISED / 745 REGISTERED / 80 UNREVIEWED); nothing landed UNREVIEWED and
  the cap is unchanged at 80.
- **Figure text is a guarded surface (G3).** A false label once shipped inside a tracked PNG through
  three sweeps because nothing on the chain read a figure. `test_figure_text_registered` now
  enumerates every string literal, f-string template or module constant passed to a matplotlib text
  call (`set_title`, `set_xlabel`, `annotate`, `text`, ... and any `label=`/`title=` keyword) in
  every generator that calls `savefig` under `scripts/` and `openmucf/` -- 46 strings across 5
  generators -- and requires each to carry a ruled row in `tests/figure_text_registry.tsv`, with an
  UNREVIEWED cap of zero. `test_every_shipped_figure_is_named_by_a_generator` ties each
  `figures/*.png` to a generator. Text assembled from data at run time is outside G3 and the test
  says so -- the point labels of `figures/muon_cost_gap.png`, drawn from the ledger's own label
  table, are in that set.

### Fixed
- **`figures/twin_bias.png` renders its disclosure in full.** The right-hand title is set on two
  lines, so "(NOT a detector prediction)" no longer runs past the figure edge. `TWIN_AUDIT.md` and
  `TWIN_MANIFEST.json` are byte-identical; only the PNG changed.

### Still deferred to v1.3.0, re-measured
- **Negation.** Adding `not`, `cannot` and the contraction to `STRONG` enumerates **335** further
  lines at this head (the v1.2.0 notes said "about 223" at theirs; the edge table, the Bertin
  re-typing, and the guard changes' own forms and prose added lines since). It lands as its own
  change, every line read. One correction to the v1.2.0 wording:
  the form `n't` cannot match inside a word-boundary alternation (no boundary precedes the `n` of
  `don't`), so the form that will land is `\w+n't`.
- **A universal split across rendered lines -- the seven paths the sentence layer does not read
  yet.** G4 above keys the whole wrapped sentence, so this deferral now names a smaller hole than
  the one v1.2.0 disclosed: in openmucf/mucost.py, tests/test_mucost.py,
  scripts/generate_findings.py, FINDINGS.md, scripts/generate_neutronomics.py, NEUTRONOMICS.md and
  openmucf/data/references.bib -- 303 wrapped matching sentences at this head -- a wrapped claim
  is still keyed at line granularity only, and an edit to the line of it that matches nothing
  re-keys nothing. The second pack reads and rules them, after which `SENTENCE_PATHS` must equal
  every claim path a sentence can wrap in and a test asserts the equality.

## [1.2.0] - 2026-08-29

**What this release is.** v1.2.0 finishes the muon-cost axis. `openmucf/data/muon_cost.csv` holds
cost *nodes* — each pinned row carrying its value at its own (stage, numeraire, charge-basis)
coordinate, with one row held open and unpinned until its primary is in hand — and
`openmucf/data/muon_cost_chain.csv` now holds the *edges* that join them, so a composed cost carries
the competing published readings of its own conversion factors instead of collapsing them, and
refuses to print a bound as a value. The cycle-closure criterion of Kou & Chen (arXiv:2607.10989) is
registered as sixteen machine-checked reproduction targets. The cross-tier basis correction is
complete: `README.md`, `FINDINGS.md` and `NEUTRONOMICS.md` each state the tier spread as an
order-of-magnitude, mixed-basis observation, and `test_no_basis_class_spans_T1_and_T3` pins the
reason — the design-study and operating-facility tiers share no basis class, so the quantity has
no common denominator to be a ratio *of*. And the claims themselves are now enumerated: a registry
of **856** lines that state a universal about the ledger, **140** of them EXERCISED — a named test
fails when the claim is negated — **636** REGISTERED with the reason each was ruled true, and **80**
still UNREVIEWED under a cap the suite lets fall without argument and lets rise only with one. The
cap moved 92 → 80 across this release, rising once (to 94) when the enumeration was re-keyed. No
registry existed at v1.1.0.

**What is deferred to v1.3.0, named here rather than left for a reader to find.**
- **The claim guard's own blind spots.** `tests/test_ledger_claims.py` enumerates a claim by matching
  a universal quantifier beside a ledger word, one rendered line at a time. It therefore does not see
  a universal stated by *negation* (`not`, `cannot`, `n't` are not in its pattern — about 223 further
  lines), and it cannot see a universal whose quantifier and subject fall on different rendered
  lines. The registry is a **floor** on this repository's quantified-claim surface, not a census of
  it, and widening the pattern is deferred rather than done here because every line the widening adds
  must be read and ruled before it may land.
- **`figures/twin_bias.png` renders its own disclosure truncated.** The panel title
  "(NOT a detector prediction)" runs past the figure edge. The string is correct in
  `scripts/generate_twin_audit.py` and the same disclosure is stated in full in `TWIN_AUDIT.md`, so
  no claim is wrong — but a reader who sees only the figure sees only part of the caveat, and the
  layout is not fixed in this release.
- **The Kamimura calibration prior is unbounded, and two byte-identity assertions are scoped to a
  platform rather than to a toolchain.** `openmucf/calibrate.py`'s informative `omega_s0` prior is a
  Normal with no support bound, so a chain can trap in a zero-prior-mass basin and the audited
  convergence of that cell is a property of one seed rather than of the sampler; separately,
  `_on_recorded_platform` checks machine and platform but not the live `jax`/`jaxlib`/`numpyro`
  versions, so a bit-identity assertion is being made across an arbitrary numerics toolchain. `jax`
  is capped below `0.11.1` in the dev extra for both reasons together. Fixing them moves audited
  `CALIBRATION.md` cells, so it gets its own release rather than riding this one.
- **Three rate-ledger rows carry digits their in-hand primaries could sharpen.** `rates.csv`'s
  `lambda_ttmu` ships the `0.0` placeholder that keeps the tt channel inert and `omega_tt` ships
  `0.14 ± 0.015`, while the JINR preprint's `2.84(32) µs⁻¹` and `0.139(15)` are recorded in those
  rows' notes and in `references.bib` but not in their value cells — because the primary quotes them
  at one condition rather than in this ledger's `× φ × c_t` normalization, which is why both rows
  still say the channel is blocked pending the Matsuzaki/Bom tables. Separately,
  `lambda_dt_transfer` is carried as a commonly-cited value with no locator, where Jones et al.
  (1986) print the same central value with a different uncertainty and an explicit temperature
  dependence. All three are shipped ledger values, so correcting any of them moves numbers the
  manifests pin — which is why they get their own pass rather than this one.

### Fixed — the `bertin_1987` row is counted where its primary counts it (2026-08-29)
`openmucf/data/muon_cost.csv`'s `bertin_1987` row shipped `stage = stopped_useful_in_dt`,
`basis_class = stopped_in_dt` and `useful_fraction_sourced = false` — a typing its primary
(Europhys. Lett. **4**, 875 (1987)) does not support. Its Table II is captioned as the cost of
*producing* muons; the quantity is built from the beam kinetic energy per negative pion produced and
the probability that such a pion decays in flight inside the target; and nothing about the muon after
its birth — range, escape, collection, stopping — is computed anywhere in pp. 875–880. The target
is liquid deuterium, with D–T reached by an argument in a footnote rather than by a calculation, and
the sentence the older typing rested on (p. 876) states the scheme's premise rather than a result.
- The row now carries `stage = produced`, `basis_class = produced` and an empty
  `useful_fraction_sourced`, so it is flagged a **lower bound** on the cost per muon stopped and
  useful in D–T, exactly like every other pinned row of its tier. `needs_verification` stays `false` and is
  not a comment on the typing: that flag records whether the DIGIT is pinned from the primary text,
  and 7.8 ± 1.8 GeV is pinned from Table II.
- **No published number moves.** The value stays 7.80 GeV; the tier medians (4.85, 178 and 4993 GeV)
  and the 1029.5× spread are unchanged; and the design-study and operating-facility tiers still
  share no basis class, so the mixed-basis reading of that spread stands. What changes is what the
  documents say: the T1 tier is now stage-homogeneous within the beam-kinetic numeraire, and
  `MUON_COST.md`, `NEUTRONOMICS.md` and `FINDINGS.md` are regenerated to say so.
- `MUON_COST.md` carries a dated amendment recording what moved and why, and
  `test_bertin_is_counted_at_production` pins the corrected cells together with the derived Headline
  sentence they drive.

### Added — the muon-cost chain gets its edge table, and a composed cost carries its competing readings (2026-08-28)
`muon_cost.csv` holds the cost **nodes**; `openmucf/data/muon_cost_chain.csv` now holds the **edges**
that join them — 8 rows, each moving exactly one axis, each carrying its own bias direction,
evidence status and source locator, and the four that state a factor carrying its operating
conditions with it. Two tables rather than more columns, for one reason: **a single conversion can
carry competing values from two primaries**, and a column cannot hold two.
- **The competing edge, which is the point.** `beam_kinetic -> electrical_minimal` is stated twice for
  the same accelerator: Kelly, Hart & Rose (2021) adopt **0.18**, and Kovach et al. (2017) — the
  primary they cite — state **18.3%**. Both are carried to the end; no mean is formed and neither is
  preferred. Composing every path out of the open-access anchor row that uses **only sourced**
  conversions therefore ends in a **set** of three terminal figures, printed as a set in
  `MUON_COST.md`: **≥ 25.68**, **≥ 26.11** and **≥ 45.19 GeV** per negative muon *produced*. The
  25.68 was previously recorded only as a note about what the unrounded figure would give; it is now
  a terminal figure of the compilation, on the same footing as the 26.11, because the edge table can
  hold both readings of one conversion.
- **What the composition refuses to do.** `ChainValue.compose` / `.to_numeraire`, `compose_path` and
  `enumerate_chain_paths` (`openmucf/mucost.py`) type the result rather than return a float: a path
  through an unsourced or author-declared factor comes back as a **bound**, whose `render_value()`
  raises `BasisError` instead of printing it as a value; and a path through a factor whose own authors
  state that they do not know it is marked *direction unknown* and carries no bound marker at all,
  because it is not one-sided. Each refusal is pinned by a test.
- **Not one of the four stage advances a full chain needs is sourced by any primary read here.**
  `MUON_COST.md`'s coverage table is derived from both tables at generation time rather than typed:
  of the 6 conversions a fully-sourced chain requires — 4 stage advances plus the 2 numeraire
  changes out of `beam_kinetic` — **2 carry a factor from a primary read here, and both of those are
  numeraire changes**. The scope is the compilation's own reading, not a claim about the whole
  literature. The four unsourced links ship as explicit `absent` rows recording what was read and
  found silent — two of them also naming the study type that would source them — rather than as
  silence.
- The stopping fraction is **deliberately not simulated**. A stopping fraction for negative muons in a
  d-t target is a particle-transport result, and producing one here would turn a compilation of
  published numbers into a model of our own; it is recorded as absent and left as a future
  convergence point with simulation work.
- `openmucf/data/muon_cost_chain.schema.json` declares the table, `datapackage.json` declares the
  resource, and `MUON_COST_MANIFEST.json` pins the headlines derived from it. No node value moves.

### Changed — prose and locators re-derived against their sources; no published number moves (2026-08-28)
Wording only: no shipped number or test outcome moves, and every generated document regenerates from
its generator. Three CC-BY data files change text cells only — `muon_cost.csv` and
`validation_targets.csv` one cell each, `benchmarks/jones-1986.json` its `title`, `input_basis` and
`notes` — and the two CSVs' digests in the FINDINGS, MUON_COST and NEUTRONOMICS manifests move, with
MATERIALITY's digest of the FINDINGS manifest following; no manifest VALUE does. Each line below was
checked against the primary or the code it describes and rewritten to what that source says.
- **"record" was this repository's word, not Jones's.** Phys. Rev. Lett. 56, 588 (1986) states on
  p.591 "an average value of 150 ± 4(stat.) ± 20(syst.) fusions per muon" for a liquefied d-t target
  at c_t = 0.3 and does not call it a record. `FINDINGS.md` §2 and its figure legend,
  `references.bib`, the `jones-1986` benchmark case, the quickstart notebook's prose, `LITERATURE.md`,
  `SYSTEMS.md`, `openmucf/systems.py` and the energy tests now say "average"; `MODEL_SPEC.md` and
  the quickstart example name the figure by its source; and the same
  unsourced descriptor ("record-class", "measured record") is dropped from the Petitjean/Breunlich
  113 in `NEUTRONOMICS.md`, its generator and `validation_targets.csv`. The two occurrences inside
  the 1.1.0 and 1.0.0 release notes below are left as those releases wrote them.
- **`docs/getting-started.md` no longer compares the required TOTAL reactivation with the collisional
  `R_col`** ("from ~0.35 toward ~0.94"): the two are successive factors, as `FINDINGS.md` §3 and
  `README.md` already state; the docs line now carries the same R ≥ 0.77 / R_X ≥ 0.64 form.
- **`FINDINGS.md` §2** names the ledger rows its liquid box comes from (`lambda_c_liquid` at
  phi ~ 1.2, and the liquid-scale `R_col`) instead of "(phi ~ 1.2, T ~ 300 K)" — the box has no
  temperature axis. Its header
  blockquote scopes "uniform over each input's own range" to the default box, names the two other
  priors the document uses (§1b's equal-relative box, §2b's tier E_mu boxes), and drops a rhetorical
  clause; §1b now states, read off the box edges at generation time, that the equal-relative box
  runs past three inputs' declared ranges; §2b says the other five inputs are drawn from their
  default boxes, not held fixed.
- **`muon_cost.csv`**: the Kelly–Hart–Rose row's `source_locator` now places each quantity in its
  own section — 4.70 GeV/muon in Sec.5 item (a), Q_elec = 14% in Sec.6 — as its sibling rows already
  did. **`validation_targets.csv`**: one notes cell. **`benchmarks/jones-1986.json`**: `title`,
  `input_basis` and `notes`.
- `paper/paper.md` states the RED-tier warning's scope as `README.md` does (one-shot, concrete calls,
  skipped under jit). `tests/test_uq.py`'s breakeven test is renamed `..._under_prior_uncertainty` —
  only one of the six priors is measured — and this file's §2b bullet below now says the tier panel
  draws the other five inputs from their default boxes rather than holding them fixed.

### Added — the test suite reports its own peak memory (2026-08-27)
The weekly `slow` job runs three long MCMC gates in one process, and when a leg of it was lost on
2026-08-24 — GitHub naming CPU/memory starvation as one thing that can cause that — no run, green or
red, had ever recorded how much memory the suite used. `tests/conftest.py` now reports peak RSS at
the end of every run, on every platform, with the platform-dependent units of `ru_maxrss` (kibibytes
on Linux, **bytes on Darwin**) drilled on both paths in `tests/test_resource_report.py`. It is a
report, not a gate: nothing fails on a memory figure.

### Changed — the ledger's quantified claims are enumerated into a registry, and the false ones are retracted (2026-08-24 … 2026-08-27)
`tests/test_ledger_claims.py` scans the shipped prose, data descriptors and figure text for lines
that state a universal about the ledger, and `tests/ledger_claims_registry.tsv` requires each line it
finds to be either **EXERCISED** — a named test fails when the claim is negated, which is stronger
than a test merely mentioning it — or **REGISTERED** with the reason it was ruled true. What predates
the cost-basis work may sit `UNREVIEWED`, under a ceiling that is monotone non-increasing. The
deferral noted at the top of this release says what the scan does not see.
- The enumeration was extended over the shipped documents, the data descriptors, `references.bib`,
  the schema files and the matplotlib title/label/legend strings, taking the registry from **386 to
  633** lines in this pass, and the sentences it surfaced were read against the tree rather than
  against recollection.
- **The false ones were retracted, not softened.** "measured" was deleted from prior ranges and box
  edges it is false of; "strict" was dropped where the code admits the bound; what the audit, the
  manifest and the trust map are *said* to do was corrected to what they do; the shipped column
  descriptions were rewritten to how the data actually has them; the neutrons-per-joule table was
  given the basis its own tier medians have; and the notes describing primaries were repaired against
  those primaries. Three statements standing in this same section were corrected in place.
- **No published number moved.** Only the generated documents' input digests changed.

### Added — Kou–Chen 2026 cycle-closure criterion registered as machine-checked reproduction targets (2026-08-20)
Sixteen `V_kouchenlawson_` rows in `openmucf/data/validation_targets.csv` register the published
Table I / Sec. IV values of arXiv:2607.10989 (four L_mu anchors, four N_L demands, four omega_crit
boundaries, four G_mu gains) and reproduce all sixteen from the package's existing closed forms in
`tests/test_koucheng2026.py` — 16/16 PASS within the paper's printed precision. The rows are
deliberately NOT scored by the `VALIDATION.md` trust gate: they compare two closed forms and are
blind to the engine, and both generated documents state the exclusion, which two tests pin.

### Changed — a mu⁺-only figure no longer reaches any aggregate; the downstream retraction lands (2026-08-19)
`muon_cost.schema.json` has always said a `mu_plus_only` figure "must never enter a muCF cost
aggregate". It was prose, and two published numbers broke it. **`MuonCostTable.tier_median` filtered
tier, numeraire and pinned status but not charge basis**, so the PSI HIMB figure entered the published
T3 median; and the `FINDINGS.md` §2b T3 prior ran to 1e6 GeV, a support that contained the same
figure. Both are fixed at the aggregate boundary, never at the row: PSI HIMB still ships in the T3
table with its own label, and the exclusion is now applied by `MuonCostTable.aggregate_rows`, which
every median, spread, ratio and prior-box edge reads. `mixed` is deliberately NOT excluded — it counts
μ⁺ and μ⁻ together, biasing an aggregate in this ledger's own declared one-sided direction, so it is
kept on basis grounds rather than on what it does to our own headline, which it raises.

**Numbers moved, and every one of them weakens a claim of ours:**
- `MUON_COST.md` T3 tier median **5497.5 → 4993 GeV**, and the tier ratio **1133.5× → 1029.5×**.
- `FINDINGS.md` §2b T3 prior **Uniform(2.3e3, 1e6) → Uniform(2286, 6002) GeV**, its median Q_net
  **4.39e-07 → 5.25e-05**, and the T1→T3 fall **~5 → ~3 orders of magnitude**. T1 and T2 are unchanged.
- `NEUTRONOMICS.md` T3 row **1.283e+08 → 1.413e+08 n/J**.

**What the documents now say instead.** §2b states that it is a sensitivity-of-Q_net-to-E_μ panel and
publishes the provenance of every box edge, including that its T1 lower edge is the Acceleron company
slide — a disclosed judgment call, since a prior-support edge is not a headline figure. The claim that
the panel's median collapse *is* the muon-cost spread in energy-return form is retracted: the fall is
governed by where the boxes were drawn, and §2b prints the median ratio beside the box-midpoint ratio
so a reader can see that. `NEUTRONOMICS.md` keeps its cross-tier factor but states that it is the cost
ratio itself, inherited by construction along with its mixed basis. The README trust map no longer
sells any cross-tier comparison as citable-as-is: the ledger per row, its basis classification and the
Eq.(15) ceiling stay GREEN, and every cross-tier ratio moves to AMBER with the reason in the table.

**Also fixed:** `SYSTEMS.md` and `openmucf/systems.py` printed a division whose quoted result was not
its quotient. The claim is corrected, not the number — Kelly's quoted 4.70 GeV/μ is load-bearing for
the faithful Table-1 reproduction (it sets the scientific Q; Q_elec carries no E_mu term) and does not move; the document now prints
3606 MeV / 0.77 = 4683 MeV beside it and says the difference is his rounding.

**Guards.** Every aggregate is drilled by recharging a row: the median moves, the box edges pull in,
and the two box rules — no edge read off a barred row, and no box support containing one — each fire
on the fault they exist for, with the historical box as the negative control. The published tier ratio
and the amendment's own before/after arithmetic are recomputed from the ledger rather than read.

### Changed — comments and descriptions now say why, not who; documentation-integrity guard added (2026-08-18)
Prose only: **no published value, test outcome or generated byte moves**, and the audited documents
regenerate byte-identically. Comments across the package explained a design by naming the authority
that settled it, or cited planning documents this repository does not contain; a citation a reader
cannot resolve is not provenance. Each now states the reason itself — the numerical argument, or a
document that ships here. Short internal identifiers were likewise replaced by what they mean, since
they were keys into registers that are not published. Two gate names stay, because a shipped document
defines each with a section heading of its own.

Three strings inside the CC-BY data package changed and are called out for anyone diffing it:
`datapackage.json`'s uncertainty-quantification resource title loses a planning-phase adjective;
three `muon_cost.csv` note fields replace a phrase that dated the check relative to the work with the
date the primary was actually read (2026-07-11), recovered from this repository's own history of those
rows; and `forecasts/forecast.schema.json`'s `source` description drops a clause about what a source
may not be, keeping only what the field is. No row, value, unit, flag or schema type changes.

New: `tests/test_docs_cited_exist.py` requires every `*.md` filename written anywhere in the tree to
name a file the tree contains. It states in its own docstring what it cannot see — references without
the suffix — so it is not read as promising more than it checks.

### Changed — every muon cost now carries its stage and its numeraire (2026-08-16)
`normalized_GeV_per_mu` was a bare number. Each row now states **where on the muCF chain** the muon
has got (`produced -> captured -> transported -> moderated -> stopped_useful_in_dt`) and **what kind
of energy** is being counted (`beam_kinetic`, or electrical on either of the two facility
denominators the same PSI primary states). Wall-plug is treated as a numeraire, not a stage: dividing
by an accelerator efficiency changes the units and applies at any stage. `basis_class` is kept as a
deprecated alias and the loader errors if the two ever disagree. The re-labelling moved **no**
published value, and a test pins that.
- Enforced in code, not prose: `ChainValue` carries a figure with its stage, numeraire and the
  evidence status of every factor composed into it. A figure short of the terminal stage, or
  composing a factor its own source calls arbitrary, is a **bound** — `render()` prefixes `>= ` and
  `render_value()` raises. All aggregation is restricted to a single numeraire, which is why the
  neutrons-per-joule medians did not move when electrical rows joined the table.
- Two derived rows added: Kelly, Hart & Rose's beam figure re-expressed in electrical energy on the
  two denominators the same PSI primary supplies — Kelly's own adopted **18%**, his two-figure
  rounding of that primary's 18.3% minimally-required-subsystem efficiency, and the primary's 10.4%
  site-wide figure, which Kelly does not adopt. The rows divide by 0.18 and 0.104, not by 0.183:
  recomputing on the unrounded 18.3% gives 25.68 GeV rather than the shipped 26.11. Both stay at
  stage `produced`; a numeraire change is not a stage advance.
- The delivery factor whose authors call it an "arbitrary but reasonable assumption" is recorded in
  its own flagged column and composed into nothing that is published.

### Retracted — the "10^3 simulation-to-facility gap" (2026-08-16)
That heading asserted a **same-basis ratio**, which the text beneath it already denied: no basis
class is shared between the design-study and operating-facility tiers, so the quantity has no common
denominator to be a ratio *of*. The spread itself is unchanged and is now stated as what the data
supports — an **order-of-magnitude, mixed-basis observation**, with its basis composition
printed. The test that pinned the old claim was re-specified rather than deleted, so what was
retracted and why is recorded in the suite.
- **It applies to three statements standing in this same section, then unreleased**, not to a
  released one: the muon-cost bullet, which said the gap "is proved from the table itself"; its
  `FINDINGS.md` §2b companion, which called the median collapse "the 10³ gap in energy-return form";
  and the neutronomics bullet, which said the gap "transfers one-for-one to the neutron economy". The
  first is corrected below, because this entry owns it. The other two describe `FINDINGS.md` and
  `NEUTRONOMICS.md`, and were left standing at the time because those are byte-diffed audited
  artifacts regenerated from their own generators.
- **Discharged 2026-08-19** by the downstream basis pass (its own entry at the top of this section):
  both generators were corrected, both documents regenerated, and each now carries a dated amendment
  saying what it retracted. Nothing here is owed before a tag any more.

### Added — the Kou-Chen cycle-closure ceiling (2026-08-16)
A ceiling on what a muon may cost, computed from declared constants on both the 20.4 MeV convention
their paper adopts and the 26.0 MeV a primary derives, with this ledger's sourceable chain points
placed against it and a coverage table showing, per source, which conversions are actually sourced.
`references.bib` gains `Kovach2017` and `KouChenLawson2026` — the latter deliberately NOT
`KouChen2026`, which already exists and is bound to a different paper by the same authors.

### Added — the D1 isotope-resolution audit, established from the primary literature (2026-08-15)
`isotope_resolved` was previously derived from the shape of the compiled-in table: true if and only
if a `Z` carried more than one record. Every one of the 90 records has now been checked against the
paper its own value is attributed to, and the audit is shipped as `data/g4/d1/isotope_audit.csv` —
one row per record, hand-authored, carrying its evidence, the table and page that establish it, and
which copy of the paper was read.
- **The old rule disagrees with the primaries on 28 of the 90 records**, and the disagreement splits
  three ways: **23** it called unresolved that the primary establishes as resolved, **2** it called
  resolved that the primary flatly contradicts, and **3** it called resolved that the primary does
  not settle either way. Its soundness argument was about the `Z`, and was being applied to each row
  of that `Z`, which does not follow.
- **86 records are settled and now carry `needs_verification: false`**, in both directions: 45 are
  established isotope-resolved, 41 are established to rest on a **natural-composition element**. The
  4 that remain open say so with an empty locator rather than guessing.
- **Two registered findings are settled and two are new** (`DATASET_D1.md`). The 74-distinct-Z
  attribution reconciles — the primary's two capture tables span exactly the same 74 Z, gaps
  included, as a compilation of world data. The effective-charge table's non-monotonic step near
  Z=82 is reproduced faithfully from the primary and is not a Geant4 transcription artifact. New:
  the primary states in its own text that the Goulard–Primakoff formula this dataset declares as its
  fallback "do[es] not account correctly for isotopic effects"; and `(Z, A)` is a **label rather
  than a target specification** on 41 records, in the extreme naming a nuclide that is 7.4 % of the
  natural element.
- **No value or uncertainty byte moved.** All 90 Layer-1 record lines are unchanged; the only
  Layer-1 edit is the digest that is supposed to move when Layer 2 does.

### Added — D1 nuclear-capture dataset in `parity` mode (2026-08-14)
The first `G4MuonicData` dataset carrying real content. `data/g4/d1/` reproduces the muon-capture
data compiled into Geant4 v11.4.2 — a 90-record `{Z, A, rate, error}` table and a 101-value
effective-charge table — bit-for-bit, together with the Goulard–Primakoff analytic fallback,
declared as data in a `#FALLBACK` directive carrying all eight of the constants it needs.
- **Generated, never transcribed.** The pinned upstream source is vendored at
  `third_party/geant4/v11.4.2/`, unmodified, and pinned by its upstream **git blob id** — upstream's
  own object name for those bytes, verifiable against `github.com/Geant4/geant4` with no Geant4
  checkout and no `git` binary. Both layers are parsed out of it at build time by
  `openmucf/g4/sources/`, and no record count appears as a literal anywhere in the chain.
- **Parity checked exhaustively.** A Geant4-linked driver harvested 36000 `(Z, A)` points from the
  built library; a pure-Python evaluation preserving the C++ association order reproduced **every
  one bit-for-bit, maximum 0 ulp**. The committed oracle makes that a CI check on every platform
  with **no Geant4 present**.
- **Five upstream findings registered and disclosed, not fixed** (`DATASET_D1.md`): negative capture
  rates on 6325 of 36000 fallback points including ³H; non-finite returns at degenerate inputs with
  no coded rejection; a fallback that moves by up to **2980 ulp** between two conforming compiler
  configurations of the same source, which is why the declared model contract forbids floating-point
  contraction; an attribution that does not reconcile with the table's 74 distinct Z; and a
  non-monotonic step in the effective-charge table near the Z=82 shell closure. **The last two of
  those five were settled against the primary literature by the entry above, inside this same
  unreleased version** — the attribution reconciles and the step is the primary's own — so this
  version ships **five defects and two settled questions**, not seven defects. Both remain
  registered in `DATASET_D1.md`, as F-4 and F-5.
- **The whole harvest chain is committed**, not only its C++ half: `cpp/tools/build_oracle.py` is
  the script that turns a driver's raw `%a` output into the committed oracle, and it reproduces that
  file byte for byte when run on the build named in the oracle's own header. A committed harvested
  artifact whose producing code is missing is the reproducibility hole vendoring the upstream source
  exists to close.
- Layer-2 row keys are now defined for tables whose primary key is a **single column** (a previously
  registered undefined case, whose first consumer is the effective-charge table).
- `FORMAT_SPEC.md` §2.2's example header no longer violates its own advisory that every `#UNITS`
  name be a `#COLUMNS` name, and now states that a `#FALLBACK` model's documentation must pin the
  formula **and its evaluation order**.
- `third_party/geant4/` is the first third-party licence in this repository: Geant4 Software License
  v1.0, applying to that directory only.

### Fixed — cross-architecture reproducibility (2026-08-09)
An independent reproduction of the full audit battery on Apple Silicon (arm64), against an x86-64
reference, found three defects that a single-architecture CI could not have surfaced. All three are fixed
here, and the gap that hid them is closed with a standing arm64 CI job.
- **Silent float32 sampling under a shadowed import (correctness, all platforms).** `jax_enable_x64=True`
  was set only in `openmucf/__init__.py`. Running any script from a directory that also contains a *clone*
  of this repository binds `openmucf` to a namespace package, so `__init__.py` never executes while the
  submodules still import — every NUTS chain then ran in float32, silently, on rates spanning ~7 decades.
  x64 is now enabled by `openmucf/_jaxcfg.py`, imported on every path into the package, and
  `require_x64()` hard-fails at each sampler entry. Pinned by `tests/test_x64_guard.py`, which reproduces
  the shadow and asserts the samplers still run in float64.
- **`DESIGN.md`'s `--audit` tolerance was smaller than the estimator's own Monte-Carlo error.** The
  sd-contraction cells were checked against a fixed 3 pp absolute band, justified by a "±1–3 pp
  Monte-Carlo floor" quoted from `docs/xray_feasibility.md` that had never been measured for these cells;
  the true per-refit spread is 4–18 pp. Such a band can only be met by regenerating the identical
  pseudo-random realization, so it passed on every x86-64 host and failed on 5 of 12 cells the first time
  another architecture ran it. Now: `n_synth` 8 → 64, every cell published as `value ± bootstrap MC
  standard error`, the audit band **derived** as 4σ of those SEs (floored at 0.01 — 4σ, not 3σ, because
  a gate that runs on every push across 12 cells must not cry wolf), and the *structural*
  claims the document makes — C4's lead, the ω_s^eff ranking, class-independence — gated at their own
  separations. The class-contrast prose is now generated from the measured paired difference and its SE,
  so the file cannot again assert a separation the data do not support.
- **The published ± omitted the base chain's own error, and the fix above would have failed cross-platform
  without it.** Every contraction is `1 − s_j/S` with one shared denominator `S = sd(base posterior)`, so a
  bootstrap over the synthetic datasets is blind to `S`'s Monte-Carlo error — the one term that actually
  moves when the run is reproduced on another machine. Measured by batch means on the pinned chain, that
  error is ~1.6% of `sd(ω_s^eff)` and ~1.7% of `sd(R)`, which alone shifts `ose_C1` by 0.011 against a band
  of 0.013. Every cell now publishes the two components combined in quadrature. It cancels to first order
  in the paired class contrast, which is unaffected. Also: the fresh run's SE — half of every band, and
  published nowhere — is now gated against the committed one, and any cell whose band exceeds its own value
  is reported as NOT INFORMATIVE instead of counting as a silent pass.
- **The FC-001 reproduction tests contradicted FC-001's own determinism statement.** `FORECAST_PROTOCOL.md`
  §7 claims bit-identity *under the recorded environment (including platform)* and Monte-Carlo agreement
  cross-platform; two tests asserted bit-identity unconditionally, so a fresh clone failed on any non-x86-64
  host. The strict gate now runs only on the platform the card itself records, and a new always-on gate
  checks every registered number (98 cells) against the pre-registered 2 % Monte-Carlo band and the card's
  structure exactly.
- **Instrumentation.** `generate_calibration.py --audit` now prints the per-class worst margin (as a
  percentage of band) whether it passes or fails, so every CI run on every platform is evidence about how
  the bands are actually sized — previously an "OK" on one architecture proved determinism, not that a
  tolerance was correctly sized. `generate_design.py --audit` prints its worst margin likewise.

### Fixed — the EIG audit band, measured rather than assumed (2026-08-12)
- **`DESIGN.md`'s EIG cells were audited against a 5 % *relative* constant, which is the wrong shape.**
  The sd-contraction cells got a measured, per-cell band on 2026-08-09; the EIG-in-bits cells kept the
  original pre-registered 5 % relative tolerance, annotated "held cross-arch 2026-07-23" — an annotation
  now **deleted**, because the artifacts of that reproduction were never tracked in this repository and
  the claim does not survive measurement. A 200-realization sweep over the base chain's seed (analysis
  seed held fixed, so the nested-Monte-Carlo draws are identical and only the posterior being integrated
  over changes) puts the per-cell realization noise at **0.042–0.068 bits = 1.40 %–6.10 % relative**: the
  noise is *absolute*, its relative size varying 4.4× across cells while its absolute size varies 1.6×.
  Against an independent realization the 5 % band therefore reds **49.5 % of runs**, worst on `eig_C3`,
  whose own noise (6.10 %) exceeds the entire band — and no relative constant fixes it, since the ≥ 34.5 %
  needed to cover `eig_C3` makes `eig_C4`'s band 24.8 σ of its own noise. The one arm64 EIG flag of
  2026-07-23 (|Δ| = 0.088 bits on `eig_C2`) sits at the 92nd percentile of that distribution, z = +1.48:
  an ordinary draw, reproduced on x86-64 by nothing more than a different chain seed.
  The EIG cells now use the same construction as the contraction cells, with the SE **measured in-run**:
  20 replicate base chains per pass (seeds 1–20, the committed seed excluded from its own error bar),
  `sd` with ddof = 1, published as `value ± se` in `DESIGN.md` and tracked in the manifest (31 → 37
  entries). Simulated false alarm **0.555 % per run**, inside the 0.12–0.60 %/run regime already chosen
  for the contraction cells at 4 σ; cost ≈ +20 s per pass. A fixed *absolute* band (0.383 bits) was
  measured and rejected — safe but ~10× more conservative than the project's own 4 σ design point, with
  7.7–18.6 % detection power at a 0.30-bit shift against 22.6–84.3 %, and stale the moment `n_outer`,
  `n_inner` or the chain length changes. **No published value moves and no finding changes**; the SE-ratio
  guard is now enforced only where a band is SE-governed, so the identically-zero `zero_eig` cell keeps
  the 0.01 absolute floor it already had.

### Fixed — the `min ess` audit cell, re-registered as a one-sided floor (2026-08-13)
- **`CALIBRATION.md`'s `min ess` cells were tolerance-audited against the committed realization, which
  is the wrong shape for a convergence diagnostic.** The symmetric 20 % band shared with the `mcse` cells
  reds when a *fresh* realization converges **better** than the committed one — a gate that punishes
  improvement is measuring the sampler's luck, not the artifact's correctness. Measured over a
  100-realization chain-seed sweep of both main chains, that band reds 3.0 % of the weak chain's
  realizations and 5.6 % of the Kamimura chain's converged, physical ones, and it held the worst margin
  of this file's whole battery in the 2026-07-23
  arm64 reproduction — 81.8 % of band, which was `Kamimura.min ess` at 9200 vs 11000: an ordinary draw of
  a diagnostic, not a defect in anything.
  `min ess` therefore leaves the tolerance-audited set. The committed value stays published as a
  *description* of that realization; what `--audit` now gates is the **fresh** run clearing a structural
  floor, `AUDIT_ESS_FLOOR = 2000`. Sizing: the 136 converged-and-physical realizations of that sweep span
  min ess 3885–11398 while the non-convergent mode sits at 2.0, so the floor separates the two regimes by
  three orders of magnitude rather than cutting through either; and because `mcse = sd/√ess`, a chain
  sitting at the floor carries an mcse inflated ×1.39 even against the lowest converged realization
  measured — outside the 20 % `mcse` band, which stays as it was. A run that clears the floor while
  genuinely mis-converged therefore still fails, on its own `mcse` cells. **No published value moves**;
  the `mean`/`sd`/`corr`/`r_hat`/`divergences` classes are untouched, and asking the audit for a
  committed-vs-fresh distance on a one-sided cell is now a hard error rather than a silent number.

### Added
- **`G4MuonicData`: an external-data format for muonic-atom physics, with its reference implementation
  (`FORMAT_SPEC.md` + `openmucf/g4/`).** A dependency-free way to ship muonic-atom data to Geant4 — or to
  any transport code — **with its provenance and its uncertainty attached**, which no dataset of this kind
  currently carries. Two layers, shipped together: Layer 1 (`*.g4dat`) is plain US-ASCII directives and
  numeric records that a C++ reader parses with nothing but its standard library; Layer 2 (`*.prov.json`)
  holds the bibliographic source, uncertainty type, competing-evaluation identity and disclosure flags that
  Layer 1 deliberately leaves out. The two are bound by an invariant: Layer 1 is generated from Layer 2 and
  its `#SOURCEDIGEST` is the SHA-256 of the Layer-2 file's bytes.
  - `FORMAT_SPEC.md` is normative and states the grammar, the Layer-2 schema, **sixteen exact error codes**
    (each carrying a 1-based line number), the reporting order two implementations must agree on, the
    archive format, and the rules a C and C++ reader has to follow — above all *not* `strtod`/`atof`, which
    honour `LC_NUMERIC` and silently return `0` for every value under a comma-decimal locale.
  - `openmucf/g4/{spec,provenance,emit}.py` is the reference implementation: parse/render/validate,
    `%.17g` floats that round-trip every finite double exactly, the Layer-2 schema and digest, and a
    deterministic `.tar.gz` plus the `geant4_add_dataset(...)` registration snippet. Standard library only,
    and fenced by test from importing this project's kinetics modules so the layer stays liftable.
  - `data/g4/` ships a worked example that carries **no physics and says so in the file itself** (every
    Layer-2 row reads "format example, not evaluated physics"), regenerated and byte-diffed by `make
    g4data` inside `make audit`. Float formatting is proven identical on Linux, macOS/arm64 and
    Windows on every CI run. The comma-decimal-locale check is *enforced* on Linux, where CI installs
    `de_DE.UTF-8` and verifies the install, so the test fails rather than skips; on the other two it
    runs only where a comma-decimal locale happens to be present.
  - Names are **provisional**: `G4MuonicData` and `G4MUONICDATA` are placeholders, and the C++ reader and
    its standalone validation application are specified but not yet part of this release.
- **Open muon-cost ledger (`openmucf/data/muon_cost.csv` + `openmucf/mucost.py` + `MUON_COST.md`).** A
  curated compilation-with-provenance of the muon-production energy cost, **each row at its own
  (`stage`, `numeraire`) coordinate rather than on a single common basis** (see the stage/numeraire
  entry above); a recapture credit stays in its own flagged column and is never folded, and wall-plug
  appears as a separate row in an electrical numeraire rather than as an edit to a beam-kinetic one.
  Twelve rows across three tiers — design studies (anchor: Kelly–Hart–Rose 4.70 GeV/μ, open access; corroborated by
  full-text-verified Bertin 1987 and Eliezer–Henis 1994), demonstrated technology, and operating facilities
  (mu2e/COMET/MuSIC/HIMB — original derivations with the arithmetic shown). The tier spread is drawn in
  `figures/muon_cost_gap.png`. **Corrected by the retraction above:** it is an order-of-magnitude,
  mixed-basis observation, not a same-basis ratio proved from the table.
- **`FINDINGS.md` §2b — Q_net by muon-cost tier.** The forward-UQ Q_net is re-run under T1/T2/T3 E_μ priors
  (via `uq.qnet_tier_panel`), with the other five inputs drawn from their default prior boxes; the median Q_net fell ~10⁵× from
  design-study to facility muons — described here at the time as "the 10³ gap in energy-return form"
  (**retracted above**, and corrected in `FINDINGS.md` on 2026-08-19; the T3 prior that produced the
  ~10⁵ figure took in a mu⁺-only row, and both it and that reading are amended in §2b itself).
  The default flat [2, 10] GeV E_μ box
  in §1/§2 is unchanged (the tier panel is an added section, not a replacement).
- `MUON_COST.md` + `MUON_COST_MANIFEST.json` join `make audit` (regenerated + byte-diffed; the PNG is not
  byte-diffed); the provenance manifest check now covers the muon-cost manifest too.
- **The Q Rosetta stone + energy-balance graph (`openmucf/systems.py` + `SYSTEMS.md`).** `SystemChain` is a
  strict superset of the frozen `energy.EnergyChain` — a differentiable `jax.numpy` graph exposing every
  node (wall-plug → muon → fusion(+breeding) → blanket → thermal → electric → recirculation) as a named
  knob, plus two explicit, flagged, default-off factors (a tritium-breeding energy credit and a
  recirculating-power fraction). At the defaults it reproduces the v1 chain to machine precision (the
  G-legacy anchor: scientific breakeven 284.09, net-electrical 2367.42). `rosetta_table` + the `QBasis`
  registry convert v1's scientific/net-electrical gains, Kelly–Hart–Rose's electrical gain, and an
  efficiency-free gain onto one comparable reference basis.
- **η_acc self-correction finding (`SYSTEMS.md`).** Our v1 default η_acc = 0.30 was optimistic; Kelly's
  PSI-measured 0.18 moves the net-electrical breakeven ~2367 → ~3946 fusions/muon (linear in η_acc). The v1
  code default is unchanged this release; the finding carries the correction. The G-Kelly cross-basis check
  reproduces Kelly's electrical-gain chain (Eq. 2 + Table 1) at 15.7% (a documented result vs their 14%
  figure-3-curve headline, not tuned).
- `SYSTEMS.md` + `SYSTEMS_MANIFEST.json` join `make audit` (both regenerated + byte-diffed; closed-form
  algebra, cross-arch stable); the provenance manifest check now covers the systems manifest too.
- **Neutrons-per-joule league table (`NEUTRONOMICS.md` + `scripts/generate_neutronomics.py`).** Places μCF
  as a 14 MeV neutron source against the established incumbents on one basis: neutrons per joule of primary
  beam energy. μCF appears as **three tier-separated rows** — one per muon-cost tier (`MUON_COST.md`),
  never a single blended row — computed as X_μ / (E_μ,tier in J) with the **measured** yield
  X_μ = 113 (`calibrate.OBS['xmu_obs']` / ledger target `V_petitjean_Xmu`, not the forward-UQ median). At
  the design-study muon cost μCF is competitive with a spallation source (~43 MeV of beam per neutron) and
  ~10³× better than a sealed-tube D-T generator; at the operating-facility muon cost the muon-cost
  spread carries into the neutron economy exactly, because n/J goes as 1/E_μ (**retracted above** as a
  same-basis ratio, and requalified in `NEUTRONOMICS.md` on 2026-08-19: the transfer is exact by
  construction and inherits the tier spread's mixed basis along with its size). A short sourced table of alternative 14 MeV/n sources
  (Thermo P385 sealed tube, FNG, RTNS-II, ISIS spallation) is included, each n/J derived from published
  beam parameters. Beam basis only (wall-plug kept separate); neutron-source economics, not breakeven;
  no new physics. `NEUTRONOMICS.md` + `NEUTRONOMICS_MANIFEST.json` join `make audit`.
- **Inverse-design frontiers (`openmucf/frontier.py` + `FRONTIER.md`).** "What would have to be true"
  breakeven frontiers over the energy-balance graph: closed-form requirement curves (`r_required`,
  `lambda_c_required`, `frontier_lambda_c_R`) plus an `optimistix` Newton solver (`solve_inverse`) that
  agree to ~1e-14. Reports the R ≳ 0.77 reactivation a MuFusE-scale programme would need for scientific
  breakeven — bit-identical to the existing forward-UQ audit. Framed strictly as requirements, never
  verdicts (the scenario-verdict registry is deliberately NOT built). `FRONTIER.md` + `FRONTIER_MANIFEST.json`
  join `make audit`; `optimistix` is promoted to an explicit dependency.
- **X-ray/neutron-ratio degeneracy-breaker feasibility scan (`docs/xray_feasibility.md`).** An exploratory
  (not-audited) scan of whether adding an X-ray-per-fusion-neutron observable to the calibration would break
  the `ω_s0`/`R` degeneracy. Best-cell posterior sd(R) contraction 42.95% in the weak-prior chain (≥ a 15%
  feasibility threshold), with its Monte-Carlo noise documented (the "±3 pp" figure originally quoted here
  is scoped to that study's asymptotic setting — see the Fixed section above). Exploratory only; the κ-band
  likelihood term is specced, not built, pending acquisition of a measured κ.
- **²²⁵Ac reproduction notebook (`scripts/parisi_ac225.py` + `notebooks/parisi_ac225.ipynb`).** A forward,
  factor-by-factor reproduction of Parisi & Rutkowski's (arXiv:2511.20951) headline — ~20 mg/yr of ²²⁵Ac from
  a 10 g ²²⁶Ra feedstock at 10¹² muons/s — from their published factors, each cited to its locator; lands at
  20.5 mg/yr (+2.6% vs the 20 mg/yr headline, +0.2% vs their Table-I value), with P_fus ≈ 564 W and ~400× the
  2024 global supply as cross-checks. Explicitly a reproduction of an *external* result — their "viable before
  energy breakeven" framing is quoted as theirs, not an OpenMuCF claim. CI-tested.
- **Bayesian experimental-design ranking (`openmucf/design.py` + `DESIGN.md`).** Ranks which next μCF
  experiment would most sharpen the partly-degenerate `(ω_s^eff, R)` estimand, over the existing calibration
  posterior, by a primary preposterior sd-contraction metric and a secondary nested-Monte-Carlo EIG. The
  X-ray/neutron ratio is the decisive, structural-class-robust R-sharpener; R is reported class-conditionally
  (constant-R vs R(φ)-inflated) because neutron-only observables do not identify R without an assumed
  structural form, and the class **contrast** is reported against its own Monte-Carlo error — on the shipped
  run it is *not* resolved, and the document says so. An internal planning instrument, not a verdict. `DESIGN.md` +
  `DESIGN_MANIFEST.json` carry NUTS-derived numbers, so `make audit` tolerance-checks them
  (`generate_design.py --audit`: every cell, EIG and sd-contraction alike, at 4σ of its own published
  Monte-Carlo SE — both the fixed 3 pp contraction band and the 5 % relative EIG band this release
  originally shipped are superseded, see the Fixed sections above) rather than byte-diffing.

### Changed
- **Class-tiered, falsifiable validation scoreboard.** Every `VALIDATION.md` row now carries a claim
  tier (`self-consistency` / `reproduction (fed input)` / `anchor-consistency` /
  `shape (calibrated model)` / `independent`) in a new `class` column, and the three Yamashita rows count
  as one shape test. Three registered `independent`-tier prediction targets now run and **FAIL by
  design** — `V_petitjean_omega` (derived effective sticking ω_s0·(1−R_col) = 0.557% vs the 0.45% band)
  and `V_faifman_900K` / `V_faifman_lowT` (the placeholder formation model vs the ledger's own
  Faifman1989 rows, ~20×/~17× low) — each a pre-registered, quantified measure of the v1 placeholder's
  distance from the field's rates (`PRE_REGISTRATION.md` amendment). No input, tolerance, or observation
  was changed to make any row pass.
- **Public surface aligned with what the code delivers.** A README trust map ("what you may cite":
  GREEN / AMBER / RED), a reworded status badge and value-prop stating that the Phase-3 surrogate is
  planned (today ω_s0 and R are ledger scalars), and `formation.py` truth-labels for every unsourced
  placeholder resonance plus a RED-tier runtime warning off its 300 K anchor (φ > 1.45 or T < 100 K).
- **Interop thermal export renamed** to `export_lambda_form_eff_thermal` (an effective cycle-scale rate,
  not the bare Faifman λ_dtμ); the old `export_lambda_dtmu_thermal` name and the `lambda_dtmu`
  `geant4_callables` key remain as deprecated aliases (removed in v2.0.0).
- **Sourced temperature-shape comparator (Yamashita–Kino 2022 Fig. 3a, digitized).** A deterministic,
  matplotlib-free digitizer (`scripts/digitize_yamashita_fig3a.py`) extracts λ_c(T) at c_t=0.5
  (`openmucf/data/yamashita_kino_lc_T.csv`, 14 points, CC-BY-4.0). `V_yamashita_ratio` is **re-anchored**
  from the earlier ~1.45 under-read to the full-curve digitized ratio λ_c(800 K)/λ_c(300 K) = 2.358
  (band [2.09, 2.62]; solid-line 2.235); the ±30% tolerance is unchanged, so the strictly-harder target
  flips the engine ratio ~1.31 **PASS→FAIL** and the row is re-tiered `shape (calibrated model)` →
  `independent` (a registered finding). A new `V_yamashita_curve` checks the engine against the digitized
  curve at 200/400/600/800 K (±30% per point; the 800 K point is a registered expected-FAIL). The
  scoreboard is now **6 pass / 5 registered-FAIL findings / 1 deferred**; the three Yamashita rows count
  as one test. No model input, prior, or tolerance was changed (`formation.py` untouched); the FAIL is
  the reported result.

### Planned
- **Phase 3 — compute-trained effective-sticking/reactivation surrogate `ω_s^eff(φ,T,c_t)`.** The one dominant
  rate that every group currently hard-codes, so that the auditor *produces* it instead of importing a
  contested constant. This is the quantitative motivation surfaced by the v1 calibration finding: experiment
  pins `ω_s^eff` and `λ_c` but not the `ω_s0`/`R` split (corr +0.84). Requires HPC/multi-GPU (cross-section
  training set + slowing-down Monte Carlo); a gold-standard close-coupling/R-matrix benchmark is the gating
  acquisition.

## [1.1.0] - 2026-07-11

### Added
- **Machine-checkable provenance (`openmucf/provenance.py` + `FINDINGS_MANIFEST.json`).** Every headline
  number in `FINDINGS.md` that the manifest lists carries a typed entry (formatted value + anchoring
  regex + source type), generated by construction from the same values the document uses; `python -m
  openmucf.provenance --check FINDINGS_MANIFEST.json` fails if a listed value drifts from its doc.
- **Single-sourced physical constants (`openmucf/constants.py`).** `λ₀`, `E_f`, and the muon-cost default
  are read once from the rate ledger and re-exported to the engine modules, so no module forks a literal
  and a broken ledger fails fast at import (zero numeric change to any result).
- **Registered UQ-priors file (`openmucf/data/uq_priors.csv`).** The uncertainty-box priors are now
  machine-sourced from a registered-priors file via `uq.params_from_ledger()` (regression-locked to the
  frozen literals); the box values are unchanged.
- **Typed ledger columns + the liquid cycling-rate row.** `rates.csv` gains
  `distribution`/`dist_lo`/`dist_hi`/`recommendation`/`phase`/`target_molecule` (schema + loader
  validation) and a first-class `lambda_c_liquid` measured-cycling-rate row (closing the long-missing
  cycling-rate row); the `eta_dtmu` row now carries an asymmetric [1, 5] interval rather than a Gaussian ±4.
- **η structural bracket (`FINDINGS.md` §1c).** The epithermal enhancement η (`eta_dtmu`) is threaded
  through the cycle engine and reported as a structural bracket beside the credible interval (X_μ at η=1
  vs η=5), with provenance-manifest entries — deliberately not folded into the UQ box, since the measured
  λ_c band already contains η as it occurred at the anchors.
- **New validation target `V_yamashita_ratio`.** Executes the pre-registered ±30% λ_c(800 K)/λ_c(300 K)
  ratio clause (engine ratio ~1.31 vs ~1.45 digitized = PASS); the validation scoreboard is now
  **7 pass / 1 deferred / 0 fail**.
- **Two absorbing loss channels in the cycle ODE (`cycle.py`) + the accounting table (`docs/accounting.md`).**
  The ttμ side-branch and ³He scavenging are added as explicit, opt-in (`include_loss_channels=True`)
  absorbing channels; the engine default stays channels-OFF and reduces to the v1 network bit-for-bit
  (reduction gate, pure atol 1e-9). `docs/accounting.md` is the single one-channel-one-home table
  recording where each deferred channel lives today and its re-attribution rule. **Framing: loss
  RE-ATTRIBUTION under the constraint that anchor-condition totals still match the measured effective
  sticking — a joint refit, not "more physics moved the numbers."**
- **Three loss-channel ledger rows (`lambda_ttmu`, `omega_tt`, `lambda_dhe3`).** `lambda_dhe3` = 1.92e8 s⁻¹
  from a live open source (Fotev et al., *Search for muon catalyzed d³He fusion*, arXiv:2001.09927);
  `omega_tt` = 0.14 (corroborated ω_tt=13.9%); `lambda_ttmu` ships the documented blocked fallback
  (0.0, `needs_verification`) pending the Matsuzaki/Bom tt-fusion tables (*Muon Catal. Fusion*).
- **Extended closed form (`analytic.fusions_per_muon_v2`).** Adds the ttμ competing-hazard term
  `ω_tt·λ_tt/λ_c` (derived in `MODEL_SPEC.md` §4.1, validated to <1% against the ODE); ³He scavenging is
  intentionally omitted from the closed form (dμ-pool hazard) and documented.
- **Channels-on scoreboard (`VALIDATION_CHANNELS.md`).** The trust gate re-run with channels ON, in the
  `make audit` regenerate+diff list. With the tt channel blocked and the anchors He-purged it reproduces
  the channels-OFF 7/1/0 scoreboard exactly; the channels-OFF `VALIDATION.md` remains the trust gate.
- **muCF-Bench case registry + `openmucf` CLI (`openmucf/bench.py`, `openmucf/cli.py`, `BENCHMARKS.md`).**
  One registry exposes both the pre-registered validation trust gate (the 8 result ids `openmucf.validate`
  emits) and self-contained JSON reproduction cases (`openmucf/data/benchmarks/*.json`, shipped as package
  data) through a single runner and the `openmucf reproduce <case-id>` / `reproduce --all` / `validate`
  console script. `validation_targets.csv` remains the single source of validation truth (the runner
  re-exposes engine results, it does not re-decide any verdict). Two reproduction cases ship: `kou-chen-2026`
  (a friendly reproduction of Kou–Chen's published 112.6/156.5 fusions-per-muon, PASS within ±10%) and
  `jones-1986` (registered PENDING as blocked-acquisition — the record operating point cannot be pinned from
  open sources, so no conditions are guessed). `BENCHMARKS.md` is regenerated and diffed by `make bench` /
  `make audit`.
- **Counts-level twin: neutron time-spectrum forward model + likelihood (`openmucf/twin.py`,
  `openmucf/likelihood.py`, `TWIN_AUDIT.md`).** A fuel-component neutron time-spectrum expectation from the
  v1 cycle (channels-OFF; reduces to the established engine), a Poisson sampler for raw histograms, the
  idealized two-exponential estimator experimenters fit, and a counts-level numpyro likelihood. `TWIN_AUDIT.md`
  (generated, in `make audit`) reports the closed-form disappearance gate (recovers λ_n to <1%), the
  estimator-bias sweep over `t_min × c_t` on synthetic v1 truth, and FC-001 card-interval fuel-component
  disappearance bands. Identifiability is stated honestly: a delta-pulse histogram constrains the muon
  disappearance rate λ_n; ω_s^eff and λ_c are separated only through the informative measured-λ_c prior. A
  200-replica interval-calibration test (`slow`-marked, deselected by default and in CI) checks the λ_n 95%
  credible interval is calibrated. Fenced v0 — no detector response, no real-data fit, no dataset-specific claim.
- **Structural sensitivity brackets (`scripts/generate_materiality.py`, `MATERIALITY.md`).** One-at-a-time
  absorbing-loss-channel toggles at four fixed operating points (OP-A anchor-adjacent/non-headline, plus
  high-T / MuFusE-mid / MuFusE-peak), reported as **one-sided brackets** `X_μ^with − X_μ^without` beside the
  §2 forward-UQ CI width for scale — never convolved into any likelihood or CI (side-by-side combination rule
  only). The ³He scavenging channel is live (`c_He ∈ {1e-4, 1e-3}`; brackets ≤ ~0.18 X_μ units, under ~0.6%
  of the parametric CI width); the ttμ side-branch is rendered **"blocked — pending acquisition of the
  Matsuzaki/Bom tt-fusion tables"** — the generator detects the ledger row's `blocked:` marker and never
  emits a misleading zero bracket. Deterministic (no MCMC), in the `make audit` regenerate+diff list with
  `MATERIALITY_MANIFEST.json`.

### Changed
- **Cross-model review hardening.** Fail-loud capture of the solver's default error norm, the
  breakeven R-requirement computed from the registered omega_s0 nominal (never transcribed), a
  parallel-make-safe audit dependency, and a negative-background guard in the twin estimator — zero
  numeric change to any shipped result.
- **Extended reproducibility audit (`make audit`).** Now also verifies the provenance manifest (across
  `FINDINGS_MANIFEST.json`, `TWIN_MANIFEST.json`, and `MATERIALITY_MANIFEST.json`), exact-diffs
  `FINDINGS_MANIFEST.json`, `VALIDATION_CHANNELS.md`, `TWIN_AUDIT.md`/`TWIN_MANIFEST.json`, and
  `MATERIALITY.md`/`MATERIALITY_MANIFEST.json`, and re-checks the `CALIBRATION.md` MCMC tables (now including
  the channels-on re-attribution section, currently blocked) within a documented tolerance.
- **`slow` pytest marker.** Long-running tests (the twin coverage run) are marked `slow` and deselected from
  the default `pytest` (and CI) via `addopts`; run them with `pytest -m slow`.
- **Formation quadrature grid.** `formation._EGRID` switched from linear to geometric spacing for low-T
  convergence (a grid doubling now moves λ_dtμ(30 K) by <0.5%, previously ~7%); `formation._CALIB` was
  re-anchored so the disclosed 300 K rates are preserved bit-exactly (no 300 K result moved; off-anchor
  temperatures shift slightly, the intended better-quadrature improvement).

## [1.0.0] - 2026-07-07

First public release: the minimum-useful, validated **v1 spine** — FAIR rate ledger → analytic closed form →
differentiable cycle ODE → net-electrical energy balance → global UQ auditor → Bayesian calibration, all
provenance-clean and reproducible.

### Added
- **FAIR rate ledger (`openmucf/data/`).** `rates.csv` with 13 input rates (9 contested, 4 established; each carrying per-row provenance,
  conditions, uncertainty, an established/contested tag, and a validity range), `validation_targets.csv`
  with 10 reproduction anchors, `references.bib`, and `rates.schema.json`. Loaded by `openmucf/rates.py`,
  which enforces schema validation and a provenance cross-check against `references.bib` and returns
  autodiff-friendly float64 rates.
- **`openmucf/analytic.py`** — the closed-form yield `X_μ = 1/(ω_s^eff + λ₀/(φ·λ̃_c))` with
  `ω_s^eff = ω_s0·(1−R)`, plus scientific and net-electrical breakeven. Reproduces the differentiable ODE to
  `rel.diff 0.000%` at the V1 gate.
- **`openmucf/cycle.py`** — the differentiable JAX/diffrax cycle-kinetics ODE network (6 components: 3
  dynamical states + 3 accumulators; Kvaerno5 stiff solver; fast-fusion/adiabatic elimination). Probability
  conserved to `<1e-4`.
- **`openmucf/formation.py`** — a physically-grounded resonance-averaged `λ_dtμ(T,φ,F)`: energy-resolved
  Vesman resonances (peak 7.1e9 s⁻¹ at 0.423 eV, Fujiwara 2000) with a Maxwellian average, thermal scale
  calibrated to the ~1e8 room-temperature anchor.
- **`openmucf/energy.py`** — a transparent scientific and **net-electrical** `Q` chain
  (`η_acc·η_thermal·M`), yielding the energy ladder: record ~150 | scientific breakeven ~284 |
  net-electrical breakeven ~2367.
- **`openmucf/uq.py`** — the uncertainty auditor: autodiff local elasticities, SALib global Sobol indices,
  Monte-Carlo forward UQ, breakeven falsification, and an ODE-vs-analytic gradient cross-check.
- **`openmucf/calibrate.py`** — numpyro (NUTS) Bayesian calibration and the `ω_s0`/`R` identifiability
  analysis.
- **`openmucf/validate.py`** — reproduces the pre-registered literature targets and auto-generates
  `VALIDATION.md` from real engine output.
- **`openmucf/interop.py`** — a GEANT4 / external-tool interop stub (complement, never compete): exports the
  differentiable rates ω_s^eff(φ,T) and λ_dtμ(E,φ,T,F) as CSV/JSON `RateTable`s, a `geant4_callables` API,
  and `ingest_spectrum` for validation data. Honors the pre-registered interop contract.
- **Auto-generated findings docs.**
  - `VALIDATION.md` — **6 pass / 1 deferred / 0 fail** against the pre-registered targets (Kou–Chen baseline
    112.6→114.5, Kou–Chen best 156.5→160.3, Petitjean ~113→130.5, Yamashita λ_c(T) monotone rise,
    Faifman epithermal peak), no input tuned to hit a target.
  - `FINDINGS.md` — sensitivity split (X_μ variance driven by reactivation R, Sobol S_T=0.62), forward-UQ
    credible intervals, and the density-scoped breakeven result `P(X_μ>500)=0` at liquid density (φ≤1.45, unpolarized) — reported as requirements (reaching 500 needs R≥0.77).
  - `CALIBRATION.md` — the `ω_s0`/`R` degeneracy (corr +0.84) that motivates Phase 3.
- **4 figures** — `figures/sobol.png`, `figures/forward_uq.png`, `figures/breakeven.png`, and the
  calibration figure — generated by `scripts/generate_findings.py`.
- **Test suite** — 43 tests across the ledger, analytic, cycle, energy, formation, UQ, calibration,
  validation, and interop modules.
- **Tooling & CI** — `ruff` (clean), GitHub Actions CI (`.github/workflows/ci.yml`), a `Makefile`
  (`make validate` / `make findings` / `make calibration`), a pinned `requirements-lock.txt` for
  reproducible installs, `pyproject.toml` (package `openmucf`, license Apache-2.0), and an expanded
  `README.md`.
- **Positioning docs** — `MODEL_SPEC.md`, `LITERATURE.md`, `PRE_REGISTRATION.md`, `CREDIBILITY_FIREWALL.md`,
  and `ADOPTERS.md`. OpenMuCF introduces **no new fundamental μCF physics**; the cycle is textbook and the
  reactivation transport is Stodden (1990) / Rafelski–Müller (1988/89). The contribution is open, reproducible,
  differentiable, UQ-bearing infrastructure plus honest findings.
- **Forecast registry (`forecasts/`, `openmucf/forecast.py`, `FORECASTS.md`).** Pre-registered, hash-stamped
  probabilistic forecast cards as a pushforward of the calibrated posterior through the analytic map (no new
  physics), scored later by CRPS + interval coverage. First card **FC-001** — effective sticking `ω_s^eff` and
  cycling rate `λ_c` at high density (`φ ∈ {1.2, 2.0, 2.4}`) under a calibrated-model scenario A and an honest
  ignorance-bound scenario B — **registered at this tag** (Zenodo DOI 10.5281/zenodo.21251512). Adds 20 forecast
  tests (**63 total**).

[Unreleased]: https://github.com/bryannasr4-gif/openmucf/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/bryannasr4-gif/openmucf/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/bryannasr4-gif/openmucf/releases/tag/v1.1.0
[1.0.0]: https://github.com/bryannasr4-gif/openmucf/releases/tag/v1.0.0
