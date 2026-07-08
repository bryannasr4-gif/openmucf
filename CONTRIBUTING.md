# Contributing to OpenMuCF

Thanks for helping build shared, trustworthy infrastructure for muon-catalyzed fusion (μCF).

OpenMuCF is **open infrastructure plus an honest finding, not new physics**. The cycle is textbook and
the reactivation transport is Stodden (1990) / Rafelski–Müller (1988/89). The value of this repository is that
every number is *sourced, conditioned, uncertainty-bearing, and reproducible*. Please hold contributions to
that bar: a change that adds a rate without provenance, or that quietly widens what the tool claims, lowers
the trust of everyone who depends on it.

> **Maintainer:** Bryan Nasr (bryannasr4@gmail.com). Repository:
> `https://github.com/bryannasr4-gif/openmucf`.

---

## 1. Development setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"            # editable install + pytest + ruff (all via the [dev] extra)
```

`make install` does the same thing. Python **>= 3.11** is required. The runtime stack is JAX / diffrax /
numpyro / SALib / pandas / matplotlib (declared in `pyproject.toml`); `ruff` is part of the `[dev]` extra
because linting is part of CI.

A pinned environment is available in `requirements-lock.txt` if you need a byte-for-byte reproduction of the
published results.

---

## 2. Running the suite

```bash
pytest                 # 83 tests; the ledger loader raises on any provenance/schema problem
ruff check .           # lint (must be clean)
ruff format .          # auto-format

make validate          # reproduce the pre-registered literature targets -> VALIDATION.md (7 pass / 1 deferred / 0 fail)
make findings          # sensitivity ranking + breakeven falsification -> FINDINGS.md
make calibration       # Bayesian calibration + identifiability -> CALIBRATION.md
make all               # lint + test + findings + calibration
```

CI (`.github/workflows/ci.yml`) runs `ruff check .` and `pytest -q` on every push and pull request. A PR that
is red on either will not be merged.

`load_rates()` validates the entire ledger against `openmucf/data/rates.schema.json` and checks that every
`source_bibkey` resolves in `openmucf/data/references.bib`; a malformed or unsourced row makes `pytest` fail
immediately, so you get fast feedback locally.

---

## 3. The core contribution flow: adding or revising a rate

This is the heart of the project. The ledger lives in `openmucf/data/rates.csv`; each row is one microscopic rate or
parameter and **must** validate against `openmucf/data/rates.schema.json`.

### 3.1 Columns (every column, in order)

Required columns (schema `required`): `symbol`, `description`, `value`, `unit`, `unc_type`, `conditions`,
`source_bibkey`, `status`, `validity_range`. The remaining columns are optional in the schema but expected in
practice — fill them unless they genuinely do not apply.

| column | required | meaning / rules |
|---|---|---|
| `symbol` | yes | Unique machine key, e.g. `omega_s0`. Lowercase-ish, no spaces; this is what code looks up. |
| `description` | yes | One human-readable line. |
| `value` | yes | The number, as a JSON/CSV number (no units in the field). |
| `unit` | yes | e.g. `s^-1`, `percent`, `fraction`, `MeV`, `GeV`, `dimensionless`. |
| `unc` | expected | 1-σ-style uncertainty **in the same unit**. Use `0` only if no uncertainty is quoted by the source — and then set the flags below so the gap is visible, never hidden. |
| `unc_type` | yes | One of `stat`, `exp`, `theory`, `theory-spread`, `model`, `table`, `estimate`, `exact`. Pick the honest provenance of the *uncertainty*, not the value. |
| `conditions` | yes | The physical state the value holds at: temperature `T`, density `phi`, hyperfine `F`, ortho/para, tritium fraction `c_t`, external field. Be structured, e.g. `T=300; phi=1.2; c_t=0.5`. |
| `source_bibkey` | yes | Key(s) resolvable in `openmucf/data/references.bib`; multiple keys are `;`-separable. **No orphan rows.** |
| `source_locator` | expected | The exact equation / table / figure / page inside the source, e.g. `Eq.30`. |
| `status` | yes | `established` or `contested` — see §3.3. |
| `validity_range` | yes | The regime the row is valid over, e.g. `liquid; phi<=1.45`, `epithermal/beam`, `all conditions`. Do not let a number leak outside where its source supports it. |
| `single_source` | expected | `true` if it traces to a single, never-independently-rechecked measurement. Feeds the Phase-4 cross-source audit. |
| `needs_verification` | expected | `true` if the exact digit/locator is not yet pinned from the primary text. |
| `notes` | optional | Context, competing values, and any `[VERIFY]` TODOs. |

### 3.2 Provenance / reference requirement (non-negotiable)

Every rate must cite a real standard-physics source, and that source must exist in `openmucf/data/references.bib`:

1. Add a BibTeX entry to `openmucf/data/references.bib` (see the existing entries for the house style — a terse `note`
   field carrying the specific number and any caveat is expected).
2. Put its key in the row's `source_bibkey`, and the precise location in `source_locator`.
3. If the work is correctly identified but you have not yet pinned the exact volume/page from the primary
   text, mark the bib `note` with `VERIFY-LOCATOR` **and** set `needs_verification: true` on the row. This is
   how the repo distinguishes "sourced but not yet digit-checked" from "settled" — both are honest states;
   silently pretending a number is pinned is not.

The test `test_every_rate_is_sourced_and_in_bib` enforces that every `source_bibkey` resolves; an unsourced or
mistyped key fails CI.

### 3.3 `established` vs `contested` tagging policy

- **`established`** — settled, multiply-confirmed, or definitional numbers that the field does not argue about:
  the muon decay clock (`lambda_mu_decay`), the d+t fusion energetics (`E_fusion`), the intramolecular fusion
  rate being fast enough that it is not the bottleneck. If a serious μCF group would nod and move on, it is
  established.
- **`contested`** — anything that is a single source, a theory-spread, model-dependent, superseded-but-still-cited,
  or the subject of active debate: initial sticking `omega_s0` (Kamimura 2023's 0.857% vs the legacy 0.91–0.93%),
  the reactivation fraction `R_col`, the epithermal enhancement `eta_dtmu` (the η = 1 vs η = 5 debate), muon
  transfer / spin-flip rates quoted from single tables, and the muon-production energy cost.
- **When in doubt, tag `contested`.** Over-claiming settledness is the failure mode this ledger exists to
  prevent. The whole point is to carry the disagreement *with its uncertainty*, not to launder it into a point
  estimate.

### 3.4 Validity range + uncertainty band (both mandatory)

- Every row must state a `validity_range`. A rate measured in an atomic beam is not automatically valid in a
  liquid or a diamond-anvil cell; say where it holds.
- Every row must carry an uncertainty story. Prefer a real `unc` with the honest `unc_type`. If the source
  quotes no uncertainty, set `unc = 0`, choose the `unc_type` that reflects why (`theory`, `table`,
  `estimate`, …), and set `single_source` / `needs_verification` so downstream UQ and the Phase-4 cross-source
  audit can see the gap. A zero uncertainty with no flags reads as "measured to be exactly this," which is
  almost never true.

### 3.5 Credibility-firewall exclusions (hard-excluded, on the record)

Before adding anything, read `CREDIBILITY_FIREWALL.md`. OpenMuCF models the **conventional** μCF cycle using
**standard QM, QED, and nuclear physics** only. The following claim classes are hard-excluded — a number whose
*only* provenance is one of these does **not** go in the ledger, and the exclusion is documented rather than
silently omitted:

- **Holmlid "ultra-dense hydrogen" H(0) / Rydberg-matter** muon and fusion claims — inconsistent with standard
  QM, not independently replicated, widely disputed.
- **LENR / "cold fusion" / electron-screening-as-a-fusion-enhancer** — no standard-model mechanism, not
  reproducible. (Ordinary electron screening of *real* μCF rates is conventional physics and **is** in scope;
  "screening as a cold-fusion mechanism" is not.)
- **Piezonuclear / fracto-fusion** energy claims — unsupported by standard nuclear physics, unreplicated.
- **"Cold/ultradense μCF beats the sticking limit" press claims with no mechanism** — we model the conventional
  reactivation physics that genuinely lowers *effective* sticking at high density (Stodden 1990,
  Rafelski–Müller 1988/89); we do not import claims that bypass the ω_s ceiling without a standard mechanism.

**In scope and welcome:** the genuine, debated, standard-physics questions — why measured high-density ω_s runs
10–50% below theory (Acceleron/MuFusE, arXiv:2606.05333), the epithermal enhancement η, the density/temperature
dependence of reactivation. Quantifying those honestly, *with uncertainty*, is exactly the point. If your rate
sits behind a firewall class, do not add it; if it forces a new exclusion, add a row to the table in
`CREDIBILITY_FIREWALL.md` explaining why.

---

## 4. Pull-request checklist

Before opening a PR, confirm:

- [ ] `pytest` passes (all tests; the ledger loader validates schema + provenance).
- [ ] `ruff check .` is clean (and `ruff format .` applied).
- [ ] Every new/changed rate has a resolvable `source_bibkey` with a matching `openmucf/data/references.bib` entry and a
      `source_locator` (or a documented `VERIFY-LOCATOR` + `needs_verification: true`).
- [ ] `status`, `validity_range`, and the uncertainty (`unc` + `unc_type`, plus `single_source` /
      `needs_verification` where relevant) are all filled per §3.
- [ ] Nothing violates `CREDIBILITY_FIREWALL.md`.
- [ ] **VALIDATION is unaffected, or the discrepancy is documented.** Run `make validate` — the gate must stay
      **7 pass / 1 deferred / 0 fail**. If your change moves a validated target, that is not automatically wrong, but you
      must explain it in the PR (and, if the physics genuinely changed, update `PRE_REGISTRATION.md` and say so
      explicitly — never re-tune inputs to hit a pre-registered target).
- [ ] If your change affects results, regenerate `FINDINGS.md` / `CALIBRATION.md` (`make findings`,
      `make calibration`) and commit the regenerated files rather than hand-editing them (see §6).

Describe *what physical claim changed and why* in the PR body, with the source. "Bumped a number" is not a
review-able change; "replaced the legacy 0.91% sticking with Kamimura 2023 Eq.30 (0.857%), status contested,
because …" is.

---

## 5. Code style

- **Ruff** is the single source of truth (config in `pyproject.toml`): `line-length = 110`, `target-version =
  py311`, rule sets `E`, `F`, `I`, `W`, `UP`, `B`, `SIM`. `E741` is intentionally ignored.
- **Single-letter physics names are allowed** (`E`, `R`, `F`, `T`, `phi`, `c_t`, …) — that is why `E741` is
  off. Use the notation the literature uses; match `MODEL_SPEC.md`.
- **Autodiff-friendly numerics.** The forward map is differentiated end-to-end (JAX / diffrax), and
  `FINDINGS.md` cross-checks the analytic gradient against the ODE gradient to ~0%. Keep it that way:
  - **No Python control flow that branches on a traced value.** Do not write `if x > 0:` where `x` may be a
    traced JAX array — use `jnp.where`, `jax.lax.cond`, `jnp.clip`, etc. Python `if`/`for` on *static* shapes
    and config is fine.
  - Keep functions pure and float64; avoid in-place mutation and NumPy where a JAX op is needed on the traced
    path.
  - Prefer smooth surrogates over hard thresholds on the differentiated path so gradients stay finite.
- Report generators under `scripts/` are allowed long lines (`per-file-ignores` for `E501`) because they embed
  markdown in f-strings.

---

## 6. Findings docs are auto-generated — do not hand-edit

`VALIDATION.md`, `FINDINGS.md`, and `CALIBRATION.md` are **generated artifacts**, each with an
"auto-generated" header:

| file | regenerate with | source |
|---|---|---|
| `VALIDATION.md` | `make validate` | `openmucf.validate` + `openmucf/data/validation_targets.csv` |
| `FINDINGS.md` | `make findings` | `scripts/generate_findings.py` |
| `CALIBRATION.md` | `make calibration` | `scripts/generate_calibration.py` |

Do not edit these by hand — your edit will be silently overwritten on the next `make`, and a hand-tuned
findings file is exactly the kind of untraceable claim this project refuses to ship. To change what they say,
change the ledger, the model code, or the generator, then re-run the target and commit the regenerated output.
The provenance chain is: `openmucf/data/` + `openmucf/` code → generator → doc.

---

Questions, disagreements about a number, or a source we should be citing: open an issue. Honest disagreement,
sourced and uncertainty-bearing, is the contribution.
