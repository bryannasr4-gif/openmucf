"""Run the Bayesian calibration, write CALIBRATION.md + figure. Run from repo root:

    python scripts/generate_calibration.py            # regenerate CALIBRATION.md + figure
    python scripts/generate_calibration.py --audit    # re-run chains, tolerance-check committed tables

Importable without side effects: all work runs inside functions / ``main()``.

CALIBRATION.md is MCMC-derived, so it is NOT byte-diffed (the cross-arch realization differs at last-ulp);
instead ``--audit`` re-runs every chain with the pinned seeds and checks each committed cell within a
tolerance CLASS chosen for that quantity (mean/sd/corr/ess/mcse/r_hat/divergences). See the AUDIT_* block.
"""

import contextlib
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from openmucf import calibrate, uq  # noqa: E402
from openmucf.constants import E_F_MEV, LAMBDA_0  # noqa: E402

CALIBRATION_MD = "CALIBRATION.md"

# --- audit tolerance CLASSES (pinned by tests/test_calibration_audit.py; hard-failing; NEVER widen) ----
# CALIBRATION.md is regenerated on one platform and audited on another (arm64-committed / x86 CI history,
# now x86-committed / x86 CI); last-ulp XLA/libm differences compound over the NUTS leapfrog steps, so the
# two platforms are statistically independent draws of the same posterior. Each committed number is checked
# within a tolerance sized for its quantity's realization noise (I2: these are numerical-reproducibility
# tolerances, NOT tuning of any physics input -- no prior/observation/seed/number changes):
#   * mean cells 2%  -- tightly constrained (worst measured cross-arch ~0.7%; ~1.1% at 1 sigma).
#   * sd cells 8%    -- the noisiest cell (weak-chain R on the degeneracy ridge) drifts ~2.8% (1 sigma)
#                       cross-arch; 8% ~ 3 sigma + .3g quantization, wide enough to never trip on sound
#                       code, tight enough that real edits (prior/observation/model) move sd >=10-20%.
#   * corr cells 8%  -- reported to 2 dp; the linear correlation of the curved ridge is prior-conditional
#                       (see the sweep) so it is bounded, not byte-pinned.
#   * ess + mcse 20% -- effective sample size (and mcse = sd/sqrt(ess)) vary most across realizations.
#   * r_hat cells 2% -- split-Gelman-Rubin sits at ~1.00; 2% is ample and still catches non-convergence.
#   * divergences EXACT == 0 -- a divergence is a correctness signal, never a tolerance question.
# Changing any of these requires a deliberate edit to the pin test + a dated RESULTS.md note.
AUDIT_RTOL_MEAN = 0.02        # mean cells
AUDIT_RTOL_SD = 0.08          # sd cells
AUDIT_RTOL_CORR = 0.08        # correlation cells
AUDIT_RTOL_ESS_MCSE = 0.20    # ess AND mcse cells
AUDIT_RTOL_RHAT = 0.02        # r_hat cells
AUDIT_ATOL_DIVERGENCES = 0    # divergences: EXACT equality (==0)

# --- chain settings (mirrored by forecast.py D6 guard: the two literals below are asserted there) ------
MAIN_WARMUP, MAIN_SAMPLES = 1000, 4000
SWEEP_WARMUP, SWEEP_SAMPLES = 1000, 1000     # 4 chains x 1000 draws per prior-sensitivity config

# --- prior-sensitivity sweep grid (weak-prior mode only) ---------------------------------------------
_R_BOXES = [((0.10, 0.60), "legacy"), ((0.05, 0.70), "wide"), ((0.00, 0.80), "widest")]
_OS0_BOXES = [((0.60, 1.10), "legacy"), ((0.50, 1.20), "wide"), ((0.40, 1.30), "widest")]
_LC_BOXES = [((0.8e8, 1.6e8), "narrow"), ((0.6e8, 1.8e8), "wide")]
_OBS_CORR_ROWS = [0.5, -0.5]                 # off-diagonal covariance-sensitivity rows


# ============================================================================ chain runs + summaries
def _run_summaries():
    """Run both MAIN calibration chains (4 chains, seeds fixed at 0) and return
    (weak, kam, summary_weak, summary_kam) with diagnostics attached."""
    mw, sw = calibrate.run_mcmc_full(num_warmup=MAIN_WARMUP, num_samples=MAIN_SAMPLES, seed=0)
    mk, sk = calibrate.run_mcmc_full(
        num_warmup=MAIN_WARMUP, num_samples=MAIN_SAMPLES, seed=0, omega_s0_prior=("normal", 0.857, 0.03)
    )
    return sw, sk, calibrate.summarize(sw, mcmc=mw), calibrate.summarize(sk, mcmc=mk)


def _rails(summ, boxes):
    """True if any parameter's 95% CI endpoint sits within 1% of its box bound (railing)."""
    for name, (lo, hi) in boxes.items():
        width = hi - lo
        c = summ[name]
        if width > 0 and (abs(c["lo"] - lo) / width < 0.01 or abs(c["hi"] - hi) / width < 0.01):
            return True
    return False


def _sweep_rows():
    """Run the 18-config factorial prior-sensitivity sweep + 2 covariance-sensitivity rows.

    Returns (rows, corr_lo, corr_hi) where rows is a list of dicts with the rendered fields.
    """
    rows = []
    corrs = []
    for (r_lo, r_hi), r_lbl in _R_BOXES:
        for (o_lo, o_hi), o_lbl in _OS0_BOXES:
            for (l_lo, l_hi), l_lbl in _LC_BOXES:
                s = calibrate.run_mcmc(
                    num_warmup=SWEEP_WARMUP, num_samples=SWEEP_SAMPLES, seed=0,
                    omega_s0_prior=("uniform", o_lo, o_hi), R_prior=(r_lo, r_hi),
                    lambda_c_prior=(l_lo, l_hi),
                )
                summ = calibrate.summarize(s)
                boxes = {"omega_s0_pct": (o_lo, o_hi), "R": (r_lo, r_hi), "lambda_c": (l_lo, l_hi)}
                corr = summ["corr_omega_s0_R"]
                corrs.append(corr)
                rows.append({
                    "config": f"R={r_lbl}/os0={o_lbl}/lc={l_lbl}",
                    "boxes": f"R[{r_lo:.2f},{r_hi:.2f}] os0[{o_lo:.2f},{o_hi:.2f}] lc[{l_lo:.1g},{l_hi:.1g}]",
                    "corr": corr,
                    "R_sd": summ["R"]["sd"],
                    "ose_mean": summ["omega_s_eff_pct"]["mean"],
                    "ose_sd": summ["omega_s_eff_pct"]["sd"],
                    "rails": "yes" if _rails(summ, boxes) else "no",
                })
    # covariance-sensitivity rows: DEFAULT boxes, off-diagonal rho_obs (MultivariateNormal likelihood)
    for rho in _OBS_CORR_ROWS:
        s = calibrate.run_mcmc(
            num_warmup=SWEEP_WARMUP, num_samples=SWEEP_SAMPLES, seed=0, obs_corr=rho,
        )
        summ = calibrate.summarize(s)
        boxes = {
            "omega_s0_pct": calibrate.WEAK_OMEGA_S0_PRIOR[1:],
            "R": calibrate.R_PRIOR_DEFAULT, "lambda_c": calibrate.LAMBDA_C_PRIOR_DEFAULT,
        }
        rows.append({
            "config": f"default box; rho_obs={rho:+.1f}",
            "boxes": "default; MultivariateNormal obs likelihood",
            "corr": summ["corr_omega_s0_R"],
            "R_sd": summ["R"]["sd"],
            "ose_mean": summ["omega_s_eff_pct"]["mean"],
            "ose_sd": summ["omega_s_eff_pct"]["sd"],
            "rails": "yes" if _rails(summ, boxes) else "no",
        })
    return rows, min(corrs), max(corrs)


def _pushforward_rows(sw, sk):
    """Posterior pushforward (lives HERE, not FINDINGS): posterior X_mu (weak + Kamimura) and a
    hybrid Q_net (posterior kinetics x ignorance-box economics). Returns list of {quantity, mean, sd, ci}.
    """
    box = {p.name: (p.low, p.high) for p in uq.PARAMS}
    rng = np.random.default_rng(0)

    def xmu_from(summ_samples):
        ose = np.asarray(summ_samples["omega_s_eff_pct"])
        lam = np.asarray(summ_samples["lambda_c"])
        return 1.0 / (ose / 100.0 + LAMBDA_0 / lam)

    x_weak = xmu_from(sw)
    x_kam = xmu_from(sk)
    # hybrid Q_net: weak-box posterior kinetics x seeded-uniform ignorance-box economics
    n = x_weak.size
    e_mu = rng.uniform(*box["E_mu_GeV"], n)
    eta_acc = rng.uniform(*box["eta_acc"], n)
    eta_th = rng.uniform(*box["eta_thermal"], n)
    q_net = x_weak * E_F_MEV * eta_th * eta_acc / (e_mu * 1.0e3)

    def stat(a):
        a = np.asarray(a)
        return {"mean": float(a.mean()), "sd": float(a.std()),
                "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5))}

    return [
        {"quantity": "X_mu (weak-box posterior)", **stat(x_weak)},
        {"quantity": "X_mu (Kamimura posterior)", **stat(x_kam)},
        {"quantity": "Q_net (hybrid: posterior kinetics x ignorance-box economics)", **stat(q_net)},
    ]


# ================================================================================= markdown rendering
def _row(name, s):
    v = s[name]
    d = s["diagnostics"][name]
    return (f"| {name} | {v['mean']:.3g} | {v['sd']:.3g} | {d['mcse']:.3g} | "
            f"[{v['lo']:.3g}, {v['hi']:.3g}] |")


def _convergence_block(sw, sk):
    def cell(summ):
        rhats = [summ["diagnostics"][k]["r_hat"] for k in calibrate._DIAG_SITES]
        esss = [summ["diagnostics"][k]["ess"] for k in calibrate._DIAG_SITES]
        return max(rhats), min(esss), summ["n_divergences"]

    wr, we, wd = cell(sw)
    kr, ke, kd = cell(sk)
    return (
        "## Convergence diagnostics (4 chains, sequential)\n"
        "| chain | max r_hat | min ess | divergences |\n|---|---|---|---|\n"
        f"| weak | {wr:.3f} | {we:.2g} | {wd} |\n"
        f"| Kamimura | {kr:.3f} | {ke:.2g} | {kd} |\n\n"
        "Convergence gate (`tests/test_calibrate.py::test_multichain_diagnostics`): max r_hat < 1.01, "
        "min ess > 400, divergences == 0 on the default (widened-box) chains.\n"
    )


def _sweep_section(rows, corr_lo, corr_hi):
    head = ("## Prior-sensitivity sweep (weak-prior mode; 4 chains x 1000)\n"
            "| config | boxes | corr | R sd | ose mean | ose sd | rails? |\n|---|---|---|---|---|---|---|\n")
    body = "\n".join(
        f"| {r['config']} | {r['boxes']} | {r['corr']:.2f} | {r['R_sd']:.3g} | "
        f"{r['ose_mean']:.3g} | {r['ose_sd']:.3g} | {r['rails']} |"
        for r in rows
    )
    close = (
        f"\n\nThe product omega_s0(1-R) (= ose mean) and lambda_c are box-invariant; corr and R-width are "
        f"support-dependent (corr range [{corr_lo:.2f}, {corr_hi:.2f}] across the sweep) -- the degeneracy "
        f"is the finding, its two-decimal value is not. `rails?` flags a 95% CI endpoint within 1% of its "
        f"OWN box bound; it is `no` for every config here, INCLUDING the legacy boxes -- the old boxes' "
        f"railing was the looser near-bound truncation quoted above (old committed CI R hi 0.588 just below "
        f"0.60; omega_s0 lo 0.608 just above 0.60, the old bound falling inside the posterior), removed by "
        f"the widening; the residual support-dependence now shows as growing R-width, not hard railing. The "
        f"last two rows treat the two observations as correlated (MultivariateNormal, rho_obs=+/-0.5) "
        f"instead of independent Gaussians: the product and lambda_c stay pinned; the corr/width "
        f"shift is the covariance sensitivity.\n"
    )
    return head + body + close


def _pushforward_section(rows):
    head = ("## Posterior pushforward (state-of-knowledge X_mu and Q_net)\n"
            "| quantity | mean | sd | 95% CI |\n|---|---|---|---|\n")
    body = "\n".join(
        f"| {r['quantity']} | {r['mean']:.4g} | {r['sd']:.3g} | [{r['lo']:.4g}, {r['hi']:.4g}] |"
        for r in rows
    )
    note = (
        "\n\nX_mu is the DEFAULT-box weak-chain (and Kamimura-chain) `(omega_s_eff, lambda_c)` joint draws "
        "pushed through `1/(ose/100 + lambda_0/lambda_c)` -- the state-of-knowledge (posterior) interval, "
        "as opposed to the ignorance-box propagation in FINDINGS.md. The two chains agree (the product is "
        "data-pinned). Q_net multiplies the weak-box posterior X_mu by seeded-uniform draws over the FROZEN "
        "uq boxes for E_mu / eta_acc / eta_thermal (posterior kinetics x ignorance-box economics -- hybrid; "
        "the economics is an ignorance box, not a posterior).\n"
    )
    return head + body + note


def build_md(sw, sk, ssw, ssk, sweep, pushforward, corr_lo, corr_hi):
    """Render the full CALIBRATION.md text from the summaries + sweep + pushforward (deterministic)."""
    return (
        _build_md_base(sw, sk, ssw, ssk, corr_lo, corr_hi)
        + _convergence_block(ssw, ssk)
        + "\n"
        + _sweep_section(sweep, corr_lo, corr_hi)
        + "\n"
        + _pushforward_section(pushforward)
        + _tt_refit_section()
    )


def _build_md_base(sw, sk, ssw, ssk, corr_lo, corr_hi):
    return f"""# CALIBRATION.md -- Bayesian calibration to experiment (auto-generated)

Calibrated (omega_s0, R, lambda_c) to Petitjean/Breunlich: omega_s_eff = 0.45+-0.05 %, X_mu = 113+-12,
via numpyro NUTS (4 sequential chains). See `openmucf/calibrate.py`.

The default prior boxes were WIDENED 2026-07-12 (disclosed statistical correction, I2-clean -- no target
involved): the previous boxes provably RAILED -- the old weak-chain 95% CI had R hi 0.588 against the old
R bound 0.60, and omega_s0 lo 0.608 against the old omega_s0 bound 0.60. New boxes: `R ~ Uniform(0.00,
0.80)` (was 0.10-0.60) and weak `omega_s0_pct ~ Uniform(0.50, 1.20)` (was 0.60-1.10); the lambda_c box is
UNCHANGED (0.8-1.6e8, a measured-band prior, not railing).

## Weak omega_s0 prior (experimental data only) -- exposes the degeneracy
| parameter | mean | sd | mcse | 95% CI |
|---|---|---|---|---|
{_row("omega_s_eff_pct", ssw)}
{_row("lambda_c", ssw)}
{_row("omega_s0_pct", ssw)}
{_row("R", ssw)}

**omega_s0 - R correlation = {ssw["corr_omega_s0_R"]:.2f}** (prior-conditional). The posterior concentrates
on the CURVE omega_s0*(1-R) = omega_s_eff (the product is pinned by the data); the linear (Pearson)
correlation is a descriptive statistic of that curved ridge and is prior-support-dependent -- it ranges
[{corr_lo:.2f}, {corr_hi:.2f}] across the prior-sensitivity sweep below. The effective sticking (product)
and lambda_c are well constrained; omega_s0 and R are not separable. (Figure `figures/calibration.png`.)

## Informative omega_s0 prior (Kamimura 0.857+-0.03 %) -- partially resolves it
| parameter | mean | sd | mcse | 95% CI |
|---|---|---|---|---|
{_row("omega_s0_pct", ssk)}
{_row("R", ssk)}
{_row("omega_s_eff_pct", ssk)}

The Kamimura *theory* input tightens omega_s0 (sd {ssw["omega_s0_pct"]["sd"]:.3g} -> {ssk["omega_s0_pct"]["sd"]:.3g} %)
and hence R -- but R still inherits that uncertainty.

## Finding
Experiment alone determines **effective sticking and the cycling rate**, not the microscopic
sticking/reactivation split. Separating omega_s0 from R -- and predicting how R changes at high density --
requires an independent microscopic calculation. **That is exactly the Phase-3 reactivation surrogate,**
and this degeneracy is the quantitative reason it is needed.

"""


def _tt_refit_section():
    """The channels-on re-attribution (3rd chain). SKIPPED-blocked while `lambda_ttmu` is 0.0 (I10, WS-N).

    Deterministic host read of the ledger: if the ttmu formation rate is blocked, emit only the section
    header (no MCMC, no cells); the generic --audit parser handles a table-less section. When the primary
    Matsuzaki/Bom tt tables land and `lambda_ttmu` becomes nonzero, the unblocked path (the actual 3rd
    NUTS chain with the tt channel in BOTH likelihood terms) must be implemented here.
    """
    from openmucf import load_rates

    tt = load_rates().get("lambda_ttmu")
    blocked = tt is None or tt.value == 0.0 or tt.notes.startswith("blocked:")
    if blocked:
        return (
            "\n## Channels-on re-attribution (ttmu) -- blocked\n\n"
            "The joint ttmu loss RE-ATTRIBUTION refit is NOT run: the ttmu formation rate `lambda_ttmu` "
            "is blocked (0.0, needs_verification) -- pending acquisition of the Matsuzaki/Bom tt-fusion "
            "tables (*Muon Catal. Fusion*). When it lands, this chain adds the tt channel to BOTH "
            "likelihood terms (obs_ose observes the TOTAL per-cycle loss ose_pct + tt_pc*100 = 0.45%, and "
            "X_mu carries tt_pc), so the omega_s0(1-R) posterior shifts DOWN by the tt share while X_mu "
            "stays ~113 -- a joint refit under the anchor-total constraint, NOT new physics. See "
            "`docs/accounting.md` and MODEL_SPEC.md sec.4.1.\n"
        )
    raise NotImplementedError(
        "tt re-attribution chain: the unblocked path lands with the Matsuzaki/Bom tt-table acquisition "
        "(lambda_ttmu is no longer 0.0) -- implement the 3rd NUTS chain per WAVE1 spec Sec.3.5 before regen."
    )


def _write_figure(weak, ssw):
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    os0 = np.asarray(weak["omega_s0_pct"])
    R = np.asarray(weak["R"])
    ax[0].hexbin(os0, R, gridsize=40, cmap="viridis")
    ax[0].set_xlabel(r"initial sticking $\omega_s^0$ [%]")
    ax[0].set_ylabel(r"reactivation $R$")
    ax[0].set_title(f"Degeneracy ridge (corr={ssw['corr_omega_s0_R']:.2f})")
    ax[1].hist(
        np.asarray(weak["omega_s_eff_pct"]),
        bins=60, density=True, alpha=0.8, color="#3388aa",
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


def _all():
    """Run everything once (main chains, sweep, pushforward) and return the render inputs."""
    sw, sk, ssw, ssk = _run_summaries()
    sweep, corr_lo, corr_hi = _sweep_rows()
    pushforward = _pushforward_rows(sw, sk)
    return sw, sk, ssw, ssk, sweep, pushforward, corr_lo, corr_hi


def regenerate():
    sw, sk, ssw, ssk, sweep, pushforward, corr_lo, corr_hi = _all()
    _write_figure(sw, ssw)
    with open(CALIBRATION_MD, "w") as f:
        f.write(build_md(sw, sk, ssw, ssk, sweep, pushforward, corr_lo, corr_hi))
    print("wrote CALIBRATION.md and figures/calibration.png")
    print(
        f"[weak prior]  omega_s_eff = {ssw['omega_s_eff_pct']['mean']:.3f} +- {ssw['omega_s_eff_pct']['sd']:.3f} %"
        f" | lambda_c = {ssw['lambda_c']['mean']:.3g} | corr(omega_s0,R) = {ssw['corr_omega_s0_R']:.2f}"
    )
    print(f"[weak prior]  omega_s0 sd = {ssw['omega_s0_pct']['sd']:.3f}, R sd = {ssw['R']['sd']:.3f}")
    print(f"[Kamimura]    omega_s0 sd = {ssk['omega_s0_pct']['sd']:.3f}, R sd = {ssk['R']['sd']:.3f}")
    print(f"[sweep] corr range [{corr_lo:.2f}, {corr_hi:.2f}] ; divergences weak={ssw['n_divergences']}")
    print(f"[pushforward] X_mu(weak) = {pushforward[0]['mean']:.1f} +- {pushforward[0]['sd']:.1f}")


# ============================================================================= audit (tolerance classes)
def _cell_specs(colname):
    """Sub-values to audit in a cell under ``colname``: list of (label, tol_kind); [] => skip the column."""
    c = colname.strip().lower()
    if "diverg" in c:
        return [("divergences", "div")]
    if "r_hat" in c or "rhat" in c:
        return [("r_hat", "rhat")]
    if "mcse" in c:
        return [("mcse", "ess_mcse")]
    if "ess" in c:
        return [("ess", "ess_mcse")]
    if "corr" in c:
        return [("corr", "corr")]
    if "mean" in c and "sd" in c:                      # a combined "mean +- sd" cell
        return [("mean", "mean"), ("sd", "sd")]
    if "sd" in c:                                      # "sd", "R sd", "ose sd"
        return [("sd", "sd")]
    if "mean" in c:                                    # "mean", "ose mean"
        return [("mean", "mean")]
    return []                                          # parameter / config / boxes / 95% CI / rails


_TOL = {
    "mean": AUDIT_RTOL_MEAN, "sd": AUDIT_RTOL_SD, "corr": AUDIT_RTOL_CORR,
    "ess_mcse": AUDIT_RTOL_ESS_MCSE, "rhat": AUDIT_RTOL_RHAT,
}


def _cell_floats(cell):
    """Extract the float(s) from a cell string ('0.46', '0.46 +- 0.04', '2534', '[..]' -> [])."""
    txt = cell.replace("+-", " ").replace("+/-", " ")
    if txt.startswith("["):
        return []
    out = []
    for tok in txt.split():
        with contextlib.suppress(ValueError):
            out.append(float(tok))
    return out


def _parse_tables(md_text):
    """Parse each ``## <title>`` section's markdown table into ``(title, header, rows)``.

    ``header`` = list of column names (or None if the section has no table); ``rows`` = list of cell-lists.
    Generic over the number of sections/columns, so new tables are picked up without editing the parser.
    """
    sections = []
    for raw in md_text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            sections.append([line[3:].strip(), None, []])
        elif line.startswith("|") and sections:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells).replace(" ", "")) <= set("-:"):
                continue  # separator row |---|---|
            if sections[-1][1] is None:
                sections[-1][1] = cells       # header row
            else:
                sections[-1][2].append(cells)  # data row
    return [(t, h, r) for t, h, r in sections]


def _within(a, b, kind):
    if kind == "div":
        return a == b
    denom = max(abs(a), abs(b))
    return denom == 0.0 or abs(a - b) <= _TOL[kind] * denom


def audit():
    """Re-run every chain and tolerance-check each committed CALIBRATION.md cell in its quantity class."""
    sw, sk, ssw, ssk, sweep, pushforward, corr_lo, corr_hi = _all()
    committed = _parse_tables(Path(CALIBRATION_MD).read_text(encoding="utf-8"))
    fresh = _parse_tables(build_md(sw, sk, ssw, ssk, sweep, pushforward, corr_lo, corr_hi))
    problems = []
    if len(committed) != len(fresh):
        problems.append(f"section count differs: committed {len(committed)} vs fresh {len(fresh)}")
    n_cells = 0
    for (ct, chead, crows), (ft, fhead, frows) in zip(committed, fresh, strict=False):
        if ct != ft:
            problems.append(f"section title differs: {ct!r} vs {ft!r}")
        if chead is None or fhead is None:
            continue
        if chead != fhead:
            problems.append(f"[{ct}] header differs: {chead} vs {fhead}")
            continue
        # index committed rows by their row-key (first cell) for order-independent matching
        cby = {r[0]: r for r in crows}
        for frow in frows:
            key = frow[0]
            crow = cby.get(key)
            if crow is None:
                problems.append(f"[{ct}] row {key!r} present in fresh but missing in committed")
                continue
            for j, col in enumerate(fhead[1:], start=1):
                for (label, kind), cval, fval in zip(
                    _cell_specs(col), _cell_floats(crow[j]), _cell_floats(frow[j]), strict=False
                ):
                    n_cells += 1
                    if not _within(cval, fval, kind):
                        tol = "==" if kind == "div" else f"{_TOL[kind]:.0%} rel."
                        problems.append(
                            f"[{ct}] {key}.{col}({label}): committed {cval:.4g} vs fresh {fval:.4g} ({tol} tol)"
                        )
    if problems:
        raise SystemExit("CALIBRATION.md audit FAILED:\n  " + "\n  ".join(problems))
    print(f"calibration audit OK: {n_cells} class-tolerance cells across {len(committed)} section(s)")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--audit" in argv:
        audit()
    else:
        regenerate()


if __name__ == "__main__":
    main()
