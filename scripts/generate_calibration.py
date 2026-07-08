"""Run the Bayesian calibration, write CALIBRATION.md + figure. Run from repo root:

    python scripts/generate_calibration.py            # regenerate CALIBRATION.md + figure
    python scripts/generate_calibration.py --audit    # re-run chains, check committed tables (2% mean / 8% sd tol)

Importable without side effects: all work runs inside functions / ``main()``.
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from openmucf import calibrate  # noqa: E402

CALIBRATION_MD = "CALIBRATION.md"
# --- audit tolerances --------------------------------------------------------------------------------
# AMENDED from the original uniform 2% by the planning model (Fable), 2026-07-07 — see RESULTS.md
# Part B (WS-A) and the WAVE1_EXECUTION_SPEC.md §1.5 amendment note. Why the split:
#   * The committed CALIBRATION.md was generated on arm64 macOS. x86-64 (GitHub CI Linux, the current
#     dev box) is deterministic run-to-run but yields a DIFFERENT realization: last-ulp XLA/libm
#     differences compound chaotically over ~5000 NUTS leapfrog steps, so the two platforms are
#     statistically independent draws of the same posterior.
#   * The MC error of a posterior sd is ~1/sqrt(2*ESS) per realization. The noisiest cell (weak-chain
#     R, which sits on the omega_s0/R degeneracy ridge) has ESS ~ 1.2k of 4000 draws => ~2.0% per
#     side, ~2.8% (1 sigma) for a committed-vs-fresh difference. Measured worst cross-arch delta:
#     2.81% on R.sd (0.112 arm64 vs 0.1088 x86) — exactly a 1-sigma draw. The old uniform 2% sat
#     BELOW the comparison's intrinsic noise floor and would fail CI on sound code.
#   * Mean cells are ~4x tighter (worst measured 0.70%; worst 1-sigma ~1.1%), so the original 2%
#     keeps genuine margin and is retained.
# AUDIT_RTOL_SD = 8% ~ 3 sigma of the sd realization noise + .3g quantization (<=0.5%): wide enough
# that no plausible platform/lockfile realization trips it, tight enough that real regressions still
# fail (prior/observation/model edits move sd cells by >=10-20%; chain settings are pinned by test_d6).
# These are numerical-reproducibility tolerances, NOT tuning of a physics input to a target (I2): no
# prior, observation, seed, or committed number changes. Hard-failing; never widen silently — changing
# these requires a planning-model amendment + a dated RESULTS.md note (WAVE1 spec §1.5). Pinned by
# tests/test_calibration_audit.py.
AUDIT_RTOL_MEAN = 0.02  # mean cells — the original DECIDED 2%, unchanged
AUDIT_RTOL_SD = 0.08  # sd cells — amended 2026-07-07 (arm64-vs-x86 realization noise; see above)


def _run_summaries():
    """Run both calibration chains (seeds fixed at 0) and return (weak, kam, summary_weak, summary_kam)."""
    weak = calibrate.run_mcmc(num_warmup=1000, num_samples=4000, seed=0)
    kam = calibrate.run_mcmc(num_warmup=1000, num_samples=4000, seed=0, omega_s0_prior=("normal", 0.857, 0.03))
    return weak, kam, calibrate.summarize(weak), calibrate.summarize(kam)


def _row(name, s):
    v = s[name]
    return f"| {name} | {v['mean']:.3g} | {v['sd']:.3g} | [{v['lo']:.3g}, {v['hi']:.3g}] |"


def build_md(sw, sk):
    """Render the CALIBRATION.md text from the two chain summaries (deterministic given the summaries)."""
    return f"""# CALIBRATION.md -- Bayesian calibration to experiment (auto-generated)

Calibrated (omega_s0, R, lambda_c) to Petitjean/Breunlich: omega_s_eff = 0.45+-0.05 %, X_mu = 113+-12,
via numpyro NUTS. See `openmucf/calibrate.py`.

## Weak omega_s0 prior (experimental data only) -- exposes the degeneracy
| parameter | mean | sd | 95% CI |
|---|---|---|---|
{_row("omega_s_eff_pct", sw)}
{_row("lambda_c", sw)}
{_row("omega_s0_pct", sw)}
{_row("R", sw)}

**omega_s0 - R correlation = {sw["corr_omega_s0_R"]:.2f}.** The effective sticking (product) and lambda_c
are well constrained, but omega_s0 and R are strongly correlated (a positive ridge along fixed
omega_s0*(1-R)): the yield/sticking data pin the product, NOT the split. (Figure `figures/calibration.png`.)

## Informative omega_s0 prior (Kamimura 0.857+-0.03 %) -- partially resolves it
| parameter | mean | sd | 95% CI |
|---|---|---|---|
{_row("omega_s0_pct", sk)}
{_row("R", sk)}
{_row("omega_s_eff_pct", sk)}

The Kamimura *theory* input tightens omega_s0 (sd {sw["omega_s0_pct"]["sd"]:.3g} -> {sk["omega_s0_pct"]["sd"]:.3g} %)
and hence R -- but R still inherits that uncertainty.

## Finding
Experiment alone determines **effective sticking and the cycling rate**, not the microscopic
sticking/reactivation split. Separating omega_s0 from R -- and predicting how R changes at high density --
requires an independent microscopic calculation. **That is exactly the Phase-3 reactivation surrogate,**
and this degeneracy is the quantitative reason it is needed.
"""


def _write_figure(weak, sw):
    os.makedirs("figures", exist_ok=True)
    # ---- figure: the identifiability ridge + marginals ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    os0 = np.asarray(weak["omega_s0_pct"])
    R = np.asarray(weak["R"])
    ax[0].hexbin(os0, R, gridsize=40, cmap="viridis")
    ax[0].set_xlabel(r"initial sticking $\omega_s^0$ [%]")
    ax[0].set_ylabel(r"reactivation $R$")
    ax[0].set_title(f"Degeneracy ridge (corr={sw['corr_omega_s0_R']:.2f})")
    ax[1].hist(
        np.asarray(weak["omega_s_eff_pct"]),
        bins=60,
        density=True,
        alpha=0.8,
        color="#3388aa",
        label=r"$\omega_s^{eff}$ (constrained)",
    )
    ax[1].hist(R, bins=60, density=True, alpha=0.5, color="#aa6633", label="R (poorly constrained)")
    ax[1].axvline(0.45, color="k", ls="--", lw=1, label="measured $\\omega_s^{eff}$=0.45%")
    ax[1].set_xlabel("value")
    ax[1].set_title("marginals: product pinned, split is not")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/calibration.png", dpi=140)
    plt.close(fig)


def regenerate():
    weak, kam, sw, sk = _run_summaries()
    _write_figure(weak, sw)
    with open(CALIBRATION_MD, "w") as f:
        f.write(build_md(sw, sk))
    print("wrote CALIBRATION.md and figures/calibration.png")
    print(
        f"[weak prior]  omega_s_eff = {sw['omega_s_eff_pct']['mean']:.3f} +- {sw['omega_s_eff_pct']['sd']:.3f} %"
        f" | lambda_c = {sw['lambda_c']['mean']:.3g} | corr(omega_s0,R) = {sw['corr_omega_s0_R']:.2f}"
    )
    print(f"[weak prior]  omega_s0 sd = {sw['omega_s0_pct']['sd']:.3f}, R sd = {sw['R']['sd']:.3f}")
    print(f"[Kamimura]    omega_s0 sd = {sk['omega_s0_pct']['sd']:.3f}, R sd = {sk['R']['sd']:.3f}")


def _parse_tables(md_text):
    """Parse every ``## <title>`` section with a ``| name | mean | sd | CI |`` table.

    Returns ``[(title, {name: (mean, sd)}), ...]``. Generic over the NUMBER of sections/rows, so a chain
    added later (e.g. WS-N's channels-on re-attribution table) is picked up without changing this parser.
    """
    sections = []
    current = None
    for raw in md_text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            current = (line[3:].strip(), {})
            sections.append(current)
        elif line.startswith("|") and current is not None:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 3:
                try:
                    mean, sd = float(cells[1]), float(cells[2])
                except ValueError:
                    continue  # header / separator / non-numeric row
                current[1][cells[0]] = (mean, sd)
    return sections


def _within(a, b, rtol):
    denom = max(abs(a), abs(b))
    return denom == 0.0 or abs(a - b) <= rtol * denom


def audit():
    """Re-run the chains and check EVERY mean/sd cell of the committed CALIBRATION.md within AUDIT_RTOL."""
    _, _, sw, sk = _run_summaries()
    committed = _parse_tables(Path(CALIBRATION_MD).read_text(encoding="utf-8"))
    fresh = _parse_tables(build_md(sw, sk))
    problems = []
    if len(committed) != len(fresh):
        problems.append(f"section count differs: committed {len(committed)} vs fresh {len(fresh)}")
    n_cells = 0
    for (ct, crows), (ft, frows) in zip(committed, fresh, strict=False):
        if ct != ft:
            problems.append(f"section title differs: {ct!r} vs {ft!r}")
        for name, (cmean, csd) in crows.items():
            if name not in frows:
                problems.append(f"[{ct}] {name}: present in committed but missing in fresh run")
                continue
            fmean, fsd = frows[name]
            for label, cval, fval, tol in (
                ("mean", cmean, fmean, AUDIT_RTOL_MEAN),
                ("sd", csd, fsd, AUDIT_RTOL_SD),
            ):
                n_cells += 1
                if not _within(cval, fval, tol):
                    problems.append(
                        f"[{ct}] {name}.{label}: committed {cval:.4g} vs fresh {fval:.4g} (> {tol:.0%} rel. tol)"
                    )
    if problems:
        raise SystemExit("CALIBRATION.md audit FAILED:\n  " + "\n  ".join(problems))
    print(
        f"calibration audit OK: {n_cells} mean/sd cells across {len(committed)} section(s) "
        f"within {AUDIT_RTOL_MEAN:.0%} (mean) / {AUDIT_RTOL_SD:.0%} (sd)"
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--audit" in argv:
        audit()
    else:
        regenerate()


if __name__ == "__main__":
    main()
