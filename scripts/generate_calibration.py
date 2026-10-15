"""Compute the calibration cells and render CALIBRATION.md, the FC-001 dated note and the figure. Run from repo root:

    python scripts/generate_calibration.py                    # compute every cell, run the NUTS cross-check, render
    python scripts/generate_calibration.py --render           # render from the committed cells; nothing computed
    python scripts/generate_calibration.py --audit            # recompute and check within the stated tolerances
    python scripts/generate_calibration.py --cells-out PATH   # write freshly computed quadrature cells to PATH only
    python scripts/generate_calibration.py --converge NAME --out PATH    # posterior NAME at res and 2 res
    python scripts/generate_calibration.py --record-convergence PATH...  # write those runs into the cells file
    python scripts/generate_calibration.py --record-scatter PATH         # scatter against cells from elsewhere

Importable without side effects: all work runs inside functions / ``main()``.

The published cells (`CALIBRATION_CELLS.json`) come from `openmucf.calibrate.posterior_quadrature`, which is
deterministic; `--audit` recomputes each one and accepts it within its class tolerance (QUAD_TOL_*), re-renders
CALIBRATION.md and the FC-001 note and requires byte equality, reruns the NUTS cross-check, and checks that the
input digests the cells record are the live files'.
"""

import contextlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from openmucf import calibrate, forecast, provenance, sidechannels, uq  # noqa: E402
from openmucf.constants import E_F_MEV, LAMBDA_0  # noqa: E402

CALIBRATION_MD = "CALIBRATION.md"
CELLS_JSON = "CALIBRATION_CELLS.json"
SCHEMA = "openmucf-calibration-cells/1"
#: the date of the anchor-basis amendment; names the FC-001 note
AMENDMENT_DATE = "2026-10-14"
NOTE_PATH = f"forecasts/FC-001-note-{AMENDMENT_DATE}.md"
VALIDATION_CHANNELS_MD = "VALIDATION_CHANNELS.md"

# --- NUTS cross-check table: audit tolerance CLASSES (pinned by tests/test_calibration_audit.py; NEVER widen) --
# The NUTS cross-check is a Monte-Carlo realization, so it is not byte-pinned: `--audit` reruns it and checks
# each recorded diagnostic in its class (the classes below), plus the 3-MCSE agreement with the quadrature.
AUDIT_RTOL_MEAN = 0.02        # mean cells
AUDIT_RTOL_SD = 0.08          # sd cells
AUDIT_RTOL_CORR = 0.08        # correlation cells
AUDIT_RTOL_ESS_MCSE = 0.20    # mcse cells (historical name; the ess half left this class 2026-08-13)
AUDIT_RTOL_RHAT = 0.02        # r_hat cells
AUDIT_ATOL_DIVERGENCES = 0    # divergences: EXACT equality (==0)

# --- min ess: a one-sided STRUCTURAL FLOOR, not a tolerance band (re-registered 2026-08-13) -----------
# min ess is a CONVERGENCE DIAGNOSTIC, not a precision claim about a published number, and a symmetric
# relative band on it is the wrong shape: it reds when a FRESH realization converges BETTER than the
# committed one. The committed value is published as a DESCRIPTION of that realization, never as a
# reproduction target; what is audited is the fresh run clearing a floor. NEVER lower this floor to make a
# run pass.
AUDIT_ESS_FLOOR = 2000        # min ess: ONE-SIDED floor on the FRESH run; never a committed-vs-fresh band

# --- the quadrature cells: tol(class) = min(cap, max(QUAD_TOL_SCATTER_FACTOR x measured scatter, floor)) ---
# relative distance; the scatter is the measured cross-platform difference of the same cells (audit_scatter)
QUAD_TOL_CAP = {"moment": 1e-5, "quantile": 5e-3, "quantile_L": 2e-2}
QUAD_TOL_FLOOR = 1e-6
QUAD_TOL_SCATTER_FACTOR = 10.0

# --- chain settings. FC-001's chain settings (mirrored by forecast.py's D6 guard: the two literals below) ----
MAIN_WARMUP, MAIN_SAMPLES = 1000, 4000
NUTS_WARMUP, NUTS_SAMPLES = 1000, 2000    # the NUTS cross-check: 4 sequential chains, seed=0

# --- quadrature resolution: the published cells, and the prior-sensitivity sweep ---------------------
RES_PUB = 1.0
SWEEP_RES = 0.5

# --- prior-sensitivity sweep grid (weak-prior mode only) ---------------------------------------------
_R_BOXES = [((0.10, 0.60), "legacy"), ((0.05, 0.70), "wide"), ((0.00, 0.80), "widest")]
_OS0_BOXES = [((0.60, 1.10), "legacy"), ((0.50, 1.20), "wide"), ((0.40, 1.30), "widest")]
_LC_BOXES = [((0.8e8, 2.0e8), "narrow"), ((0.6e8, 2.4e8), "wide")]

_WEAK = calibrate.WEAK_OMEGA_S0_PRIOR
_KAM = calibrate.KAMIMURA_OMEGA_S0_PRIOR
_DEFAULT = dict(R_box=calibrate.R_PRIOR_DEFAULT, lam_box=calibrate.LAMBDA_C_PRIOR_W, cov="L3")
#: every published posterior: (omega_s0 prior, quadrature keywords). The FC-001 pair takes FC-001's own priors
#: (openmucf.forecast) on the card's lambda_c support and on the widened one.
POSTERIORS = {
    "weak": (_WEAK, dict(_DEFAULT)),
    "kamimura": (_KAM, dict(_DEFAULT)),
    "fc001_card_support": (
        forecast.OMEGA_S0_PRIOR, dict(R_box=forecast.R_PRIOR, lam_box=calibrate.LAMBDA_C_PRIOR_DEFAULT, cov="L3")
    ),
    "fc001_widened_support": (
        forecast.OMEGA_S0_PRIOR, dict(R_box=forecast.R_PRIOR, lam_box=calibrate.LAMBDA_C_PRIOR_W, cov="L3")
    ),
    "l2_default": (_WEAK, dict(_DEFAULT, cov="L2")),
    "l1_default": (_WEAK, dict(_DEFAULT, cov="L1")),
}
CONVERGENCE_POSTERIORS = ("weak", "kamimura", "fc001_card_support", "fc001_widened_support")
INPUT_FILES = {
    "side_channel_inputs_csv_sha256": "openmucf/data/side_channel_inputs.csv",
    "uq_priors_csv_sha256": "openmucf/data/uq_priors.csv",
    "fc001_card_sha256": "forecasts/FC-001-mufuse.json",
}


def lc_label(box):
    return f"lc[{box[0]:.3g},{box[1]:.3g}]"


def _boxes_label(r_box, o_box, l_box):
    return f"R[{r_box[0]:.2f},{r_box[1]:.2f}] os0[{o_box[0]:.2f},{o_box[1]:.2f}] {lc_label(l_box)}"


# ============================================================================ computation
def _posterior(name, res=None, **kw):
    prior, opts = POSTERIORS[name]
    return calibrate.posterior_quadrature(prior, res=RES_PUB if res is None else res, **opts, **kw)


def _cells_of(post):
    out = {q: {st: float(post[q][st]) for st in calibrate.STATS} for q in calibrate.QUAD_QUANTITIES}
    out["corr_omega_s0_R"] = float(post["corr_omega_s0_R"])
    return out


def _rails(post, boxes):
    """True if any parameter's 95% interval endpoint sits within 1% of its box bound (railing)."""
    for name, (lo, hi) in boxes.items():
        width = hi - lo
        c = post[name]
        if width > 0 and (abs(c["q0.025"] - lo) / width < 0.01 or abs(c["q0.975"] - hi) / width < 0.01):
            return True
    return False


def _sweep_rows():
    rows = []
    for r_box, r_lbl in _R_BOXES:
        for o_box, o_lbl in _OS0_BOXES:
            for l_box, l_lbl in _LC_BOXES:
                post = calibrate.posterior_quadrature(
                    ("uniform", *o_box), R_box=r_box, lam_box=l_box, cov="L3", res=SWEEP_RES
                )
                boxes = {"omega_s0_pct": o_box, "R": r_box, "lambda_c": l_box}
                rows.append({
                    "config": f"R={r_lbl}/os0={o_lbl}/lc={l_lbl}",
                    "R_box": list(r_box), "os0_box": list(o_box), "lc_box": list(l_box),
                    "corr": float(post["corr_omega_s0_R"]),
                    "R_sd": float(post["R"]["sd"]),
                    "ose_mean": float(post["omega_s_eff_pct"]["mean"]),
                    "ose_sd": float(post["omega_s_eff_pct"]["sd"]),
                    "rails": bool(_rails(post, boxes)),
                })
    return rows


def _bisect(fun, p, lo, hi, rtol=1e-10):
    while fun(hi) < p:
        hi *= 2.0
    while fun(lo) > p:
        lo /= 2.0
    while hi - lo > rtol * 0.5 * (hi + lo):
        mid = 0.5 * (lo + hi)
        if fun(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def q_net_cells(weak, n=48):
    """Q_net = X_mu E_F eta_thermal eta_acc / (E_mu 1e3) with the weak posterior X_mu and the frozen uq boxes,
    each uniform: mean and sd in closed form, quantiles by n-node Gauss-Legendre over each box against the
    posterior X_mu CDF, solved by bisection."""
    box = {p.name: (p.low, p.high) for p in uq.PARAMS}
    eth, eacc, emu = box["eta_thermal"], box["eta_acc"], box["E_mu_GeV"]

    def u_moments(lo, hi):
        return 0.5 * (lo + hi), (lo * lo + lo * hi + hi * hi) / 3.0

    ex = weak["X_mu"]["mean"]
    ex2 = weak["X_mu"]["sd"] ** 2 + ex * ex
    m_th, m2_th = u_moments(*eth)
    m_ac, m2_ac = u_moments(*eacc)
    a, b = emu
    m_inv, m2_inv = np.log(b / a) / (b - a), (1.0 / a - 1.0 / b) / (b - a)
    mean = ex * E_F_MEV * m_th * m_ac * m_inv / 1e3
    e2 = ex2 * E_F_MEV**2 * m2_th * m2_ac * m2_inv / 1e6
    edges, cdf = weak["X_mu_cdf"]
    gx, gw = np.polynomial.legendre.leggauss(n)
    gw = gw / gw.sum()

    def nodes(lo, hi):
        return 0.5 * (lo + hi) + 0.5 * (hi - lo) * gx

    th, ac, em = nodes(*eth), nodes(*eacc), nodes(*emu)
    wts = gw[:, None, None] * gw[None, :, None] * gw[None, None, :]
    fac = (em[None, None, :] * 1.0e3) / (E_F_MEV * th[:, None, None] * ac[None, :, None])

    def f_q(q):
        return float((wts * np.interp(q * fac, edges, cdf)).sum())

    out = {"mean": float(mean), "sd": float(np.sqrt(e2 - mean * mean))}
    for p in (0.025, 0.5, 0.975):
        out[f"q{p}"] = float(_bisect(f_q, p, 1e-5, 2.0))
    return out


def legacy_tension():
    w = calibrate.OBS_W
    x = 1.0 / (w["omega_s_pct"] / 100.0 + LAMBDA_0 / w["phi_lambda_c"])
    return {
        "x_mu": x,
        "z_sigma_at_12": (x - w["xmu"]) / calibrate.OBS["xmu_sd"],
        "z_sigma_at_10": (x - w["xmu"]) / w["xmu_sd"],
    }


def _inputs():
    return {k: provenance.file_sha256(v) for k, v in INPUT_FILES.items()}


def _computed_on():
    return {"platform": f"{platform.system()} {platform.machine()}", "python": platform.python_version(),
            "numpy": np.__version__}


def compute_cells(posts=None):
    if posts is None:
        posts = {name: _posterior(name) for name in POSTERIORS}
    return {
        "schema": SCHEMA,
        "estimator": {
            "res": RES_PUB, "sweep_res": SWEEP_RES, "grid": dict(calibrate.QUAD_GRID),
            "bins": {k: v[2] for k, v in calibrate.QUAD_BINS.items()},
            "l_other_pct": list(sidechannels.ANCHOR_L_OTHER_PCT),
            "lambda_c_box": list(calibrate.LAMBDA_C_PRIOR_W), "covariance": "L3",
        },
        "posteriors": {name: _cells_of(p) for name, p in posts.items()},
        "sweep": _sweep_rows(),
        "q_net": q_net_cells(posts["weak"]),
        "legacy_tension": legacy_tension(),
        "inputs": _inputs(),
        "computed_on": _computed_on(),
    }


def nuts_cross_check(posteriors):
    """NUTS on ``calibrate.model_w`` for the weak and Kamimura priors: (record, weak draws)."""
    rec, weak_draws = {}, None
    for name, kwargs in (
        ("weak", dict(omega_s0_prior=calibrate.WEAK_OMEGA_S0_PRIOR)),
        ("kamimura", dict(omega_s0_prior=calibrate.KAMIMURA_OMEGA_S0_PRIOR)),
    ):
        mcmc, draws = calibrate.run_model_w(num_warmup=NUTS_WARMUP, num_samples=NUTS_SAMPLES, seed=0, **kwargs)
        diag = calibrate.summarize_w(mcmc)
        quad = posteriors[name]
        diag["max_abs_dev_over_mcse"] = max(
            abs(diag["mean"][q] - quad[q]["mean"]) / diag["mcse"][q] for q in calibrate.QUAD_QUANTITIES
        )
        rec[name] = diag
        if name == "weak":
            weak_draws = draws
    return rec, weak_draws


def dump(cells, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cells, sort_keys=True, indent=2, ensure_ascii=True) + "\n")


def load(path=CELLS_JSON):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ================================================================================= markdown rendering
def _g(v):
    return f"{v:.3g}"


def _param_rows(post, names):
    rows = []
    for label, q in names:
        c = post[q]
        rows.append(f"| {label} | {_g(c['mean'])} | {_g(c['sd'])} | {_g(c['q0.025'])} | {_g(c['q0.5'])} | "
                    f"{_g(c['q0.975'])} |")
    return "\n".join(rows)


_PARAM_HEAD = "| parameter | mean | sd | 2.5% | median | 97.5% |\n|---|---|---|---|---|---|\n"
_TITLE_WEAK = "Weak omega_s0 prior (experimental data only) -- exposes the degeneracy"
_TITLE_KAM = "Informative omega_s0 prior (Kamimura 0.857+-0.03 %) -- partially resolves it"
_TITLE_CONV_OLD = "Convergence diagnostics (4 chains, sequential)"
_TITLE_SWEEP_OLD = "Prior-sensitivity sweep (weak-prior mode; 4 chains x 1000)"
_TITLE_PUSH = "Posterior pushforward (state-of-knowledge X_mu and Q_net)"
_WEAK_ROWS = [("omega_s_eff_pct", "omega_s_eff_pct"), ("lambda_c", "lambda_c"), ("omega_s0_pct", "omega_s0_pct"),
              ("R", "R"), ("W_pct", "W_pct"), ("L_other_pct", "L_pct")]
_KAM_ROWS = [("omega_s0_pct", "omega_s0_pct"), ("R", "R"), ("omega_s_eff_pct", "omega_s_eff_pct")]
_QNET_LABEL = "Q_net (posterior X_mu x frozen uq economics boxes; deterministic)"
_QNET_LABEL_OLD = "Q_net (hybrid: posterior kinetics x ignorance-box economics)"
_RHO_ROWS = {"default box; rho_obs=+0.5": "default box; L2 likelihood",
             "default box; rho_obs=-0.5": "default box; L1 likelihood"}


def _lead():
    w = calibrate.OBS_W
    return (
        "# CALIBRATION.md -- Bayesian calibration to experiment (auto-generated)\n\n"
        "Calibrated (omega_s0, R, lambda_c, L_other) to the three summaries Breunlich et al. (1987) print for "
        f"liquid D/T at 23 K: the effective sticking omega_s = {w['omega_s_pct']:g}+-{w['omega_s_sd']:g} % (p.331), "
        f"the maximum cycle rate phi*lambda_c = {w['phi_lambda_c'] / 1e6:g}+-{w['phi_lambda_c_sd'] / 1e6:g} us^-1 "
        f"and the yield X_mu = {w['xmu']:g}+-{w['xmu_sd']:g} (p.332), with their common "
        f"+-{100 * w['calibration']:g} % absolute calibration (Fig. 4 caption) entering all three.\n"
        "The yield relation sees the total loss per cycle W = omega_s0*(1-R) + L_other, where L_other is the "
        "side-channel loss at the anchor fill (`openmucf.sidechannels`); 0.45 % measures the sticking alone, "
        "because the source subtracted those losses from the raw loss per cycle (p.331).\n"
        "Estimator: deterministic quadrature (`openmucf.calibrate.posterior_quadrature`), which `make audit` "
        "checks within stated tolerances; NUTS on the same model is a cross-check (section \"NUTS cross-check\").\n"
    )


def _sweep_now(cells):
    by = {r["config"]: r for r in cells["sweep"]}
    for old, new in _RHO_ROWS.items():
        key = "l2_default" if "L2" in new else "l1_default"
        p = cells["posteriors"][key]
        by[old] = {"config": new, "boxes": "default", "corr": p["corr_omega_s0_R"], "R_sd": p["R"]["sd"],
                   "ose_mean": p["omega_s_eff_pct"]["mean"], "ose_sd": p["omega_s_eff_pct"]["sd"], "rails": None}
    return by


def _now_value(cells, section, row, col):
    po = cells["posteriors"]
    if section in (_TITLE_WEAK, _TITLE_KAM):
        c = po["weak" if section == _TITLE_WEAK else "kamimura"][row]
        return {"mean": _g(c["mean"]), "sd": _g(c["sd"]), "mcse": None,
                "95% CI": f"[{_g(c['q0.025'])}, {_g(c['q0.975'])}]"}[col]
    if section == _TITLE_CONV_OLD:
        d = cells["nuts_cross_check"]["weak" if row == "weak" else "kamimura"]
        return {"max r_hat": f"{d['max_rhat']:.3f}", "min ess": f"{d['min_ess']:.0f}",
                "divergences": str(d["divergences"])}[col]
    if section == _TITLE_SWEEP_OLD:
        r = _sweep_now(cells)[row]
        if col == "boxes":
            return r["boxes"] if "boxes" in r else _boxes_label(r["R_box"], r["os0_box"], r["lc_box"])
        rails = "-" if r["rails"] is None else ("yes" if r["rails"] else "no")
        return {"corr": f"{r['corr']:.2f}", "R sd": _g(r["R_sd"]), "ose mean": _g(r["ose_mean"]),
                "ose sd": _g(r["ose_sd"]), "rails?": rails}[col]
    if section == _TITLE_PUSH:
        if row == _QNET_LABEL_OLD:
            c = cells["q_net"]
            sd = c["sd"]
        else:
            c = po["weak" if "weak" in row else "kamimura"]["X_mu"]
            sd = c["sd"]
        return {"mean": f"{c['mean']:.4g}", "sd": _g(sd), "95% CI": f"[{c['q0.025']:.4g}, {c['q0.975']:.4g}]"}[col]
    raise KeyError(section)


_SECTION_TAG = {_TITLE_WEAK: "weak", _TITLE_KAM: "Kamimura", _TITLE_CONV_OLD: "convergence",
                _TITLE_SWEEP_OLD: "sweep", _TITLE_PUSH: "pushforward"}


def _moved_rows(cells):
    rows = []
    for key in sorted(cells["shipped_v1_3_0"]):
        doc, section, row, col = (key.split("|") + [None])[:4]
        if doc != CALIBRATION_MD:
            continue
        before = cells["shipped_v1_3_0"][key]
        now = _now_value(cells, section, row, col)
        if now == before:
            continue
        label = f"{_SECTION_TAG[section]}: {row} {col}"
        rows.append(f"| {label} | {before} | {'(column removed)' if now is None else now} |")
    return rows


def _amendment(cells):
    lt = cells["legacy_tension"]
    rows = _moved_rows(cells)
    return (
        f"## AMENDMENT {AMENDMENT_DATE} -- the anchor basis\n\n"
        "Until this amendment the calibration fed the measured effective sticking into the yield relation as the "
        "total loss per cycle and left the measured cycle rate out of the likelihood; the yield's uncertainty was "
        "taken as +-12, where the source prints +-10.\n"
        f"With the paired values (0.45 %, 1.45e8 s^-1) that relation gives a yield of {lt['x_mu']:.1f}, "
        f"{lt['z_sigma_at_12']:.2f} sigma from the measured 113 at +-12 and {lt['z_sigma_at_10']:.2f} sigma at "
        "+-10: the measurements alone did not force this correction; the source's definition of the quantities "
        "did.\n"
        "The table compares each published cell before and after.\n\n"
        "| cell | as printed before | now |\n|---|---|---|\n" + "\n".join(rows) + "\n"
    )


_BOX_PARAGRAPH = (
    "## Prior boxes\n\n"
    "The default prior boxes were WIDENED 2026-07-12 (a disclosed statistical correction -- no target\n"
    "involved): the previous boxes provably RAILED -- the old weak-chain 95% CI had R hi 0.588 against the old\n"
    "R bound 0.60, and omega_s0 lo 0.608 against the old omega_s0 bound 0.60. New boxes: `R ~ Uniform(0.00,\n"
    "0.80)` (was 0.10-0.60) and weak `omega_s0_pct ~ Uniform(0.50, 1.20)` (was 0.60-1.10); the lambda_c box\n"
    "was then unchanged (0.8-1.6e8); with lambda_c in the likelihood since the amendment above it is\n"
    "(0.8-2.0e8), because the narrower box truncated the posterior.\n"
)


def _corr_range(cells):
    corrs = [r["corr"] for r in cells["sweep"]]
    return min(corrs), max(corrs)


def _posterior_sections(cells):
    w, k = cells["posteriors"]["weak"], cells["posteriors"]["kamimura"]
    lo, hi = _corr_range(cells)
    kp = calibrate.KAMIMURA_OMEGA_S0_PRIOR
    return (
        f"## {_TITLE_WEAK}\n" + _PARAM_HEAD + _param_rows(w, _WEAK_ROWS) + "\n\n"
        f"**omega_s0 - R correlation = {w['corr_omega_s0_R']:.2f}** (prior-conditional). The posterior concentrates\n"
        "on the CURVE omega_s0*(1-R) = omega_s_eff (the product is pinned by the data); the linear (Pearson)\n"
        "correlation is a descriptive statistic of that curved ridge and is prior-support-dependent -- it ranges\n"
        f"[{lo:.2f}, {hi:.2f}] across the prior-sensitivity sweep below. The effective sticking (product)\n"
        "and lambda_c are well constrained; omega_s0 and R are not separable. (Figure `figures/calibration.png`.)\n\n"
        f"## {_TITLE_KAM}\n" + _PARAM_HEAD + _param_rows(k, _KAM_ROWS) + "\n\n"
        f"The Kamimura *theory* input tightens omega_s0 (sd {w['omega_s0_pct']['sd']:.3g} -> "
        f"{k['omega_s0_pct']['sd']:.3g} %)\n"
        "and hence R -- but R still inherits that uncertainty.\n\n"
        "## AMENDMENT 2026-09-22 -- the Kamimura prior bounded to the weak-prior support\n\n"
        f"The Kamimura chain uses a Normal with location {kp[1]} and scale {kp[2]}, truncated to the weak prior's "
        f"support [{kp[3]}, {kp[4]}].\n"
        "Within that support the prior is unchanged at double precision; the bound removes the negative-omega_s0 "
        "region in which a chain could trap, so the convergence cells and their audit describe this sampler "
        "across seeds.\n"
        "The Kamimura cells are a new realization replacing those in this document as released in v1.2.0.\n"
        "The FC-001 forecast retains its registered untruncated prior.\n\n"
        "## Finding\n"
        "Experiment alone determines **effective sticking and the cycling rate**, not the microscopic\n"
        "sticking/reactivation split. Separating omega_s0 from R -- and predicting how R changes at high density "
        "--\n"
        "requires an independent microscopic calculation. **That is exactly the Phase-3 reactivation surrogate,**\n"
        "and this degeneracy is the quantitative reason it is needed.\n\n"
    )


def quad_tolerances(scatter):
    """tol(class) = min(cap, max(QUAD_TOL_SCATTER_FACTOR x scatter, QUAD_TOL_FLOOR))."""
    return {c: min(QUAD_TOL_CAP[c], max(QUAD_TOL_SCATTER_FACTOR * scatter[c], QUAD_TOL_FLOOR))
            for c in QUAD_TOL_CAP}


def _estimator_section(cells):
    est, conv = cells["estimator"], cells["convergence"]
    g = {k: int(round(v * est["res"])) for k, v in est["grid"].items()}
    head = (
        "## Estimator and convergence\n\n"
        f"The cells are computed by `openmucf.calibrate.posterior_quadrature` at res {est['res']:g}, on midpoint "
        f"grids of {g['ne']} x {g['nl']} x {g['nL']} cells in (omega_s_eff, lambda_c, L_other) and {g['ns']} x "
        f"{g['ns']} cells in (omega_s0, R), and are stored at full precision in `{CELLS_JSON}`.\n"
        "The table gives each posterior's largest relative change between that resolution and twice it, per "
        "class of cell, and the criterion of each class.\n\n"
        "| posterior | means and sds | quantiles (not L_other) | L_other quantiles |\n|---|---|---|---|\n"
    )
    body = "".join(
        f"| {name} | {c['moment']:.2g} | {c['quantile']:.2g} | {c['quantile_L']:.2g} |\n"
        for name, c in conv["posteriors"].items()
    )
    crit = conv["criteria"]
    body += f"| criterion | {crit['moment']:g} | {crit['quantile']:g} | {crit['quantile_L']:g} |\n\n"
    sc = cells.get("audit_scatter")
    if sc is None:
        return head + body
    tol = quad_tolerances(sc["scatter"])
    names = {"moment": "means, sds and correlations", "quantile": "quantiles (not L_other)",
             "quantile_L": "L_other quantiles"}
    plats = " and ".join(f"{p['platform']} (Python {p['python']}, numpy {p['numpy']})" for p in sc["platforms"])
    aud = (
        f"`make audit` recomputes every cell and accepts it within min(cap, max({QUAD_TOL_SCATTER_FACTOR:g} x "
        f"the measured cross-platform scatter, {QUAD_TOL_FLOOR:g})) relative; the scatter was measured between "
        f"{plats}.\n\n"
        "| class | cap | measured scatter | tolerance |\n|---|---|---|---|\n"
        + "".join(f"| {names[c]} | {QUAD_TOL_CAP[c]:g} | {sc['scatter'][c]:.2g} | {tol[c]:.2g} |\n"
                  for c in QUAD_TOL_CAP)
        + "\n"
    )
    return head + body + aud


def _nuts_section(cells):
    n = cells["nuts_cross_check"]
    rows = "".join(
        f"| {lbl} | {n[k]['max_rhat']:.3f} | {n[k]['min_ess']:.0f} | {n[k]['divergences']} | "
        f"{n[k]['max_abs_dev_over_mcse']:.2f} |\n"
        for lbl, k in (("weak", "weak"), ("Kamimura", "kamimura"))
    )
    return (
        "## NUTS cross-check (4 chains, sequential)\n"
        "| chain | max r_hat | min ess | divergences | max offset of a mean from quadrature, in MCSE |\n"
        "|---|---|---|---|---|\n" + rows + "\n"
        f"NUTS on the same priors and likelihood (`openmucf.calibrate.model_w`, {NUTS_WARMUP} warmup and "
        f"{NUTS_SAMPLES} draws per chain) is an independent check of the quadrature: it passes when max r_hat < "
        f"{calibrate.NUTS_MAX_RHAT:g}, min ess > {calibrate.NUTS_MIN_ESS}, no draw diverges and every mean lies "
        f"within {calibrate.NUTS_MAX_DEV_MCSE:g} MCSE of the quadrature mean.\n"
        f"`make audit` reruns it, accepting max r_hat within {AUDIT_RTOL_RHAT:.0%} of this record, a fresh min ess "
        f"of at least {AUDIT_ESS_FLOOR} (`min ess` above describes this run and is not a reproduction target), "
        "divergences exactly as recorded, and the same agreement with the quadrature.\n\n"
    )


def _sweep_section(cells):
    head = (f"## Prior-sensitivity sweep (weak-prior mode; quadrature at res {cells['estimator']['sweep_res']:g})\n"
            "| config | boxes | corr | R sd | ose mean | ose sd | rails? |\n|---|---|---|---|---|---|---|\n")
    rows = [
        f"| {r['config']} | {_boxes_label(r['R_box'], r['os0_box'], r['lc_box'])} | {r['corr']:.2f} | "
        f"{_g(r['R_sd'])} | {_g(r['ose_mean'])} | {_g(r['ose_sd'])} | {'yes' if r['rails'] else 'no'} |"
        for r in cells["sweep"]
    ]
    for key, lbl in (("l2_default", "default box; L2 likelihood"), ("l1_default", "default box; L1 likelihood")):
        p = cells["posteriors"][key]
        rows.append(f"| {lbl} | default | {p['corr_omega_s0_R']:.2f} | {_g(p['R']['sd'])} | "
                    f"{_g(p['omega_s_eff_pct']['mean'])} | {_g(p['omega_s_eff_pct']['sd'])} | - |")
    lo, hi = _corr_range(cells)
    ose = [r["ose_mean"] for r in cells["sweep"]]
    n_rail = sum(bool(r["rails"]) for r in cells["sweep"])
    rails = ("it is `no` for every config here" if n_rail == 0
             else f"it is `yes` for {n_rail} of the {len(cells['sweep'])} configs")
    close = (
        f"\n\nThe product omega_s0(1-R) (= ose mean) spans [{min(ose):.3g}, {max(ose):.3g}] across the sweep; corr "
        f"and R-width are support-dependent (corr range [{lo:.2f}, {hi:.2f}] across the sweep) -- the degeneracy "
        "is the finding, its two-decimal value is not. "
        f"`rails?` flags a 95% interval endpoint within 1% of its OWN box bound; {rails}. "
        "The last two rows refit the default box with the likelihood's covariance replaced: L2 keeps only the "
        "cycle-rate/yield correlation the yield formula implies, L1 treats the three summaries as independent.\n"
    )
    return head + "\n".join(rows) + close


def _pushforward_section(cells):
    po, qn = cells["posteriors"], cells["q_net"]
    rows = []
    for lbl, c in (("X_mu (weak-box posterior)", po["weak"]["X_mu"]),
                   ("X_mu (Kamimura posterior)", po["kamimura"]["X_mu"]), (_QNET_LABEL, qn)):
        rows.append(f"| {lbl} | {c['mean']:.4g} | {_g(c['sd'])} | [{c['q0.025']:.4g}, {c['q0.975']:.4g}] |")
    return (
        f"\n## {_TITLE_PUSH}\n| quantity | mean | sd | 95% CI |\n|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "X_mu is the weak-prior (and Kamimura-prior) posterior of `1/(W/100 + lambda_0/lambda_c)` -- the "
        "state-of-knowledge (posterior) interval, as opposed to the ignorance-box propagation in FINDINGS.md. "
        "Q_net multiplies the weak-box posterior X_mu by the FROZEN uq boxes for E_mu / eta_acc / eta_thermal, "
        "each uniform: mean and sd in closed form, quantiles by Gauss-Legendre quadrature over the three boxes "
        "against the posterior X_mu distribution (posterior kinetics x ignorance-box economics; the economics is "
        "an ignorance box, not a posterior).\n"
    )


def _side_channel_section():
    a = sidechannels.anchor_l_other()
    c = a["components_pct"]
    rows = [("tt (tmut)", c["tt"]), ("dd (dmud, 3He + n branch)", c["dd"]), ("protium (pmud + pmut)", c["p"]),
            ("impurities and 3He", c["imp"]), ("total", a["central_pct"]), ("sd", a["sigma_pct"])]
    return (
        "\n## Side-channel loss at the anchor fill\n"
        "| channel | loss per cycle (%) |\n|---|---|\n"
        + "".join(f"| {lbl} | {v:.4f} |\n" for lbl, v in rows)
        + "\nL_other is computed at the anchor fill's conditions (c_t, normalized cycle rate and protium fraction) "
        "and is not carried to other conditions; FINDINGS.md section 3 uses it only where it says so.\n"
    )


def build_md(cells):
    return (
        _lead() + "\n" + _amendment(cells) + "\n" + _BOX_PARAGRAPH + "\n" + _posterior_sections(cells)
        + _estimator_section(cells) + _nuts_section(cells) + _sweep_section(cells) + "\n"
        + _pushforward_section(cells) + _side_channel_section()
    )


# ================================================================================= the FC-001 note
def _card_scenario_a():
    card = json.loads(Path(forecast.CARD_PATH).read_text(encoding="utf-8"))
    sc = [s for s in card["payload"]["scenarios"] if s["name"] == "A"]
    assert len(sc) == 1
    return {p["target_id"]: p for p in sc[0]["predictions"]}


def vb_figures(cells):
    """(Vb_old, Vb_new, Vb_pct): the shipped print of V_breunlich_lambdac on the channels-on page (the cells'
    shipped record), the committed page's print, and 100*(new/old - 1) to 3 decimals."""
    old = cells["shipped_v1_3_0"]["VALIDATION_CHANNELS.md|V_breunlich_lambdac|predicted"]
    rows = [ln for ln in Path(VALIDATION_CHANNELS_MD).read_text(encoding="utf-8").splitlines()
            if ln.startswith("| V_breunlich_lambdac |")]
    if len(rows) != 1:
        raise ValueError(f"{VALIDATION_CHANNELS_MD}: {len(rows)} V_breunlich_lambdac rows")
    new = rows[0].split("|")[4].strip()
    return old, new, f"{100.0 * (float(new) / float(old) - 1.0):.3f}"


def build_note(cells):
    """The FC-001 dated note: the card's scenario-A medians at phi = 1.2 beside the corrected basis.
    Raises unless every corrected median lies inside the card's own 95 % interval."""
    preds = _card_scenario_a()
    po = cells["posteriors"]
    cc, cw = po["fc001_card_support"], po["fc001_widened_support"]
    rows = []
    for label, tid, q in (("lambda_c [s^-1]", "lambda_c@phi=1.2", "lambda_c"),
                          ("omega_s_eff [%]", "omega_s_eff@phi=1.2", "omega_s_eff_pct")):
        p = preds[tid]
        lo, hi = p["ci95"]
        for name, post in (("fc001_card_support", cc), ("fc001_widened_support", cw)):
            med = post[q]["q0.5"]
            if not lo <= med <= hi:
                raise ValueError(f"{name}.{q}.q0.5 = {med:.6g} lies outside the card's 95 % interval [{lo}, {hi}]")
        rows.append(
            f"| {label} | {p['median']:.4g} | {cc[q]['q0.5']:.4g} (x{cc[q]['q0.5'] / p['median']:.3f}) | "
            f"{cw[q]['q0.5']:.4g} (x{cw[q]['q0.5'] / p['median']:.3f}) | [{lo:.4g}, {hi:.4g}] |"
        )
    w = calibrate.OBS_W
    lt = cells["legacy_tension"]
    vb_old, vb_new, vb_pct = vb_figures(cells)
    card_box = POSTERIORS["fc001_card_support"][1]["lam_box"]
    wide_box = POSTERIORS["fc001_widened_support"][1]["lam_box"]
    return (
        f"# FC-001 dated note ({AMENDMENT_DATE}) -- the calibration basis the card was computed on\n\n"
        "This note sits beside `FC-001-mufuse.json` and changes nothing in it: the card is not edited or "
        "re-hashed, and it is\nscored as registered.\n\n"
        "The card's predictions are conditional on the calibration basis that stood when it was registered. "
        "That calibration\nused the measured effective sticking, "
        f"{w['omega_s_pct']:g} %, from which Breunlich et al. (1987, p.331) had already subtracted the\n"
        "dmud, tmut, pmud and pmut fusion losses, as the total loss per cycle in the yield relation, and it left "
        "the measured\ncycle rate, "
        f"{w['phi_lambda_c'] / 1e6:g} us^-1 (p.332), out of the likelihood. With those paired values the relation "
        f"gives a yield of\n{lt['x_mu']:.1f}, {lt['z_sigma_at_12']:.2f} sigma from the measured 113 at the card's "
        f"yield uncertainty of 12, and {lt['z_sigma_at_10']:.2f} sigma at the +-10 the\n"
        "source prints: the measurements alone did not force the correction; the source's definition of the "
        "quantities did.\n`CALIBRATION.md` now fits the three summaries with the side-channel loss at the anchor "
        "fill.\n\n"
        f"| scenario-A median at phi = 1.2 | card | corrected basis, the card's lambda_c support [{card_box[0]:.3g}, "
        f"{card_box[1]:.3g}] s^-1 | corrected basis, the refit's support [{wide_box[0]:.3g}, {wide_box[1]:.3g}] "
        "s^-1 | card 95 % interval |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "Every corrected median lies inside the card's own 95 % interval. The scenario-A cycle-rate medians at "
        "phi = 2.0 and\n2.4 scale by phi/1.2 on either basis.\n\n"
        "One figure outside the card moved against an expectation stated before it was computed. With the ttmu "
        "channel live,\nthe cycle rate the channels-on validation page infers at 300 K and liquid density "
        "(`V_breunlich_lambdac` in\n`VALIDATION_CHANNELS.md`) had been expected to stay at "
        f"{vb_old} s^-1; it is {vb_new} s^-1, {vb_pct} %, because ttmu\n"
        "formation takes atoms from the tmu pool and the muons that do not stick in the tt branch return split "
        "between the two\nhyperfine states (`openmucf/cycle.py`), which moves the mean d-t cycle time. The card "
        "does not depend on that page:\n`openmucf.forecast`, which computes it, does not load "
        "`openmucf.validate`.\n"
    )


# ================================================================================= figure
def _write_figure(weak):
    Path("figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    os0 = np.asarray(weak["omega_s0_pct"])
    R = np.asarray(weak["R"])
    corr = float(np.corrcoef(os0, R)[0, 1])
    ax[0].hexbin(os0, R, gridsize=40, cmap="viridis")
    ax[0].set_xlabel(r"initial sticking $\omega_s^0$ [%]")
    ax[0].set_ylabel(r"reactivation $R$")
    ax[0].set_title(f"Degeneracy ridge (corr={corr:.2f})")
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


# ================================================================================= driver
def _write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _render_texts(cells):
    """Both documents, built before either is written: a refusal (``build_note``) leaves every file as it was."""
    return {CALIBRATION_MD: build_md(cells), NOTE_PATH: build_note(cells)}


def render(cells):
    for path, text in _render_texts(cells).items():
        _write_text(path, text)


def regenerate():
    """Compute every cell, run and gate the NUTS cross-check, write the cells file, render the documents and
    the figure. The shipped record, the convergence record and the measured scatter are carried forward from
    the existing cells file (they are written by the steps that measure them)."""
    old = load()
    for key in ("shipped_v1_3_0", "convergence"):
        if key not in old:
            raise SystemExit(f"{CELLS_JSON} has no {key!r}: it is written before the cells are computed")
    posts = {name: _posterior(name) for name in POSTERIORS}
    cells = compute_cells(posts)
    for key in ("shipped_v1_3_0", "convergence", "audit_scatter"):
        if key in old:
            cells[key] = old[key]
    ncc, weak_draws = nuts_cross_check(cells["posteriors"])
    bad = {k: calibrate.nuts_gate_failures(v, cells["posteriors"][k]) for k, v in ncc.items()}
    if any(bad.values()):
        raise SystemExit(f"NUTS cross-check FAILED: {bad}")
    cells["nuts_cross_check"] = ncc
    cells = json.loads(json.dumps(cells))      # the values the cells file will hold
    texts = _render_texts(cells)                # every refusal comes before the first write
    dump(cells, CELLS_JSON)
    for path, text in texts.items():
        _write_text(path, text)
    _write_figure(weak_draws)
    print(f"wrote {CELLS_JSON}, {CALIBRATION_MD}, {NOTE_PATH} and figures/calibration.png")
    for name in ("weak", "kamimura"):
        p = cells["posteriors"][name]
        print(f"[{name}] omega_s_eff {p['omega_s_eff_pct']['mean']:.4f} lambda_c {p['lambda_c']['mean']:.5g} "
              f"R {p['R']['mean']:.4f} X_mu {p['X_mu']['mean']:.4f} | NUTS max r_hat {ncc[name]['max_rhat']:.4f} "
              f"min ess {ncc[name]['min_ess']:.0f} div {ncc[name]['divergences']} "
              f"max dev {ncc[name]['max_abs_dev_over_mcse']:.2f} MCSE")


# ============================================================================= audit
def _cell_specs(colname):
    """Sub-values to audit in a cell under ``colname``: list of (label, tol_kind); [] => skip the column."""
    c = colname.strip().lower()
    if "diverg" in c:
        return [("divergences", "div")]
    if "r_hat" in c or "rhat" in c:
        return [("r_hat", "rhat")]
    if "mcse" in c:
        return [("mcse", "mcse")]
    if "ess" in c:                                     # min ess -- and any future ESS column: floor, not band
        return [("ess", "ess_floor")]
    if "corr" in c:
        return [("corr", "corr")]
    if "mean" in c and "sd" in c:                      # a combined "mean +- sd" cell
        return [("mean", "mean"), ("sd", "sd")]
    if "sd" in c:                                      # "sd", "R sd", "ose sd"
        return [("sd", "sd")]
    if "mean" in c:                                    # "mean", "ose mean"
        return [("mean", "mean")]
    return []                                          # parameter / config / boxes / 95% CI / rails


# RELATIVE-tolerance classes only. "div" (exact equality) and "ess_floor" (one-sided) are deliberately
# absent: membership here is what makes a cell's COMMITTED value a reproduction target, and _rel_distance
# refuses any kind that is not a member.
_TOL = {
    "mean": AUDIT_RTOL_MEAN, "sd": AUDIT_RTOL_SD, "corr": AUDIT_RTOL_CORR,
    "mcse": AUDIT_RTOL_ESS_MCSE, "rhat": AUDIT_RTOL_RHAT,
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


def _rel_distance(committed, fresh, kind) -> float:
    """Committed-vs-fresh distance in the cell's own class units (relative, except exact-eq classes)."""
    if kind == "div":
        return 0.0 if committed == fresh else float("inf")
    if kind not in _TOL:
        # Must-not-return guard: a ONE-SIDED cell has no committed-vs-fresh distance, and computing one
        # for `min ess` is precisely the 2026-08-13 defect coming back (a band that reds on a BETTER
        # fresh realization). Structural, so it cannot be reintroduced by editing the class map alone.
        raise AssertionError(
            f"{kind!r} is not a relative-tolerance class: it has no committed-vs-fresh band (see "
            f"AUDIT_ESS_FLOOR -- min ess is gated one-sided and its committed value is description only)"
        )
    denom = max(abs(committed), abs(fresh))
    return 0.0 if denom == 0.0 else abs(committed - fresh) / denom


def _within(committed, fresh, kind):
    """Accept a cell in its own class. One-sided classes ignore ``committed`` BY CONSTRUCTION."""
    if kind == "div":                       # divergence counts are exact-equality, not a tolerance class
        return committed == fresh
    if kind == "ess_floor":                 # convergence diagnostic: floor the FRESH run, never band it
        return fresh >= AUDIT_ESS_FLOOR
    return _rel_distance(committed, fresh, kind) <= _TOL[kind]


def _quantile_class(q, st):
    if st in ("mean", "sd"):
        return "moment"
    return "quantile_L" if q == "L_pct" else "quantile"


def quad_cells(cells):
    out = []
    for name, post in cells["posteriors"].items():
        for q in calibrate.QUAD_QUANTITIES:
            for st in calibrate.STATS:
                out.append((f"posteriors.{name}.{q}.{st}", post[q][st], _quantile_class(q, st)))
        out.append((f"posteriors.{name}.corr_omega_s0_R", post["corr_omega_s0_R"], "moment"))
    for r in cells["sweep"]:
        for k in ("corr", "R_sd", "ose_mean", "ose_sd"):
            out.append((f"sweep[{r['config']}].{k}", r[k], "moment"))
    for st, v in cells["q_net"].items():
        out.append((f"q_net.{st}", v, _quantile_class("Q_net", st)))
    for k, v in cells["legacy_tension"].items():
        out.append((f"legacy_tension.{k}", v, "moment"))
    return out


def _exact_cells(cells):
    return {f"sweep[{r['config']}].{k}": r[k] for r in cells["sweep"] for k in ("rails", "R_box", "os0_box",
                                                                              "lc_box")}


def compare_cells(committed, fresh, tol):
    """(a): fresh quadrature cells against the committed ones, each within its class tolerance."""
    a = {p: (v, c) for p, v, c in quad_cells(committed)}
    b = {p: v for p, v, _ in quad_cells(fresh)}
    problems, worst = [], {}
    if set(a) != set(b):
        problems.append(f"cell sets differ: {sorted(set(a) ^ set(b))[:5]}")
    for p, (v, c) in a.items():
        if p not in b:
            continue
        d = calibrate._rel(v, b[p])
        if c not in worst or d > worst[c][0]:
            worst[c] = (d, p, v, b[p])
        if not d <= tol[c]:
            problems.append(f"{p}: committed {v!r} vs fresh {b[p]!r}, relative {d:.3g} > tol {tol[c]:.3g}")
    ea, eb = _exact_cells(committed), _exact_cells(fresh)
    problems += [f"{p}: committed {ea[p]!r} vs fresh {eb.get(p)!r}" for p in ea if ea[p] != eb.get(p)]
    return problems, worst


def committed_tolerances(cells):
    sc = cells.get("audit_scatter")
    if sc is None:
        return {c: QUAD_TOL_FLOOR for c in QUAD_TOL_CAP}
    tol = quad_tolerances(sc["scatter"])
    if sc.get("tolerance") != tol:
        raise SystemExit(f"audit_scatter.tolerance {sc.get('tolerance')} is not the rule's {tol}")
    return tol


def input_problems(cells):
    """(d): every input digest the cells record must be the live file's."""
    live = _inputs()
    rec = cells.get("inputs", {})
    probs = [f"inputs.{k}: recorded {rec.get(k)} vs live {v}" for k, v in live.items() if rec.get(k) != v]
    probs += [f"inputs.{k}: no such input" for k in rec if k not in live]
    return probs


def render_problems(cells):
    """(b): the committed CALIBRATION.md and FC-001 note must be the render of the committed cells."""
    probs = []
    for path, text in ((CALIBRATION_MD, build_md(cells)), (NOTE_PATH, build_note(cells))):
        if not Path(path).exists():
            probs.append(f"{path}: missing")
        elif Path(path).read_text(encoding="utf-8") != text:
            probs.append(f"{path}: differs from the render of the committed cells")
    return probs


def nuts_audit_problems(committed, fresh, quad):
    """(c): a fresh NUTS cross-check against the committed record -- max r_hat within its relative class,
    min ess against the one-sided floor, divergences exact -- and the gate against the committed quadrature."""
    probs = []
    for name, c in committed.items():
        f = fresh[name]
        for key, kind in (("max_rhat", "rhat"), ("min_ess", "ess_floor"), ("divergences", "div")):
            if not _within(c[key], f[key], kind):
                what = (f"BELOW the structural floor {AUDIT_ESS_FLOOR}" if kind == "ess_floor"
                        else f"committed {c[key]!r}")
                probs.append(f"nuts_cross_check.{name}.{key}: fresh {f[key]!r} ({what})")
        probs += [f"nuts_cross_check.{name}: {m}" for m in calibrate.nuts_gate_failures(f, quad[name])]
    return probs


def audit():
    committed = load()
    problems = input_problems(committed)
    problems += render_problems(committed)
    tol = committed_tolerances(committed)
    fresh = compute_cells()
    p_a, worst = compare_cells(committed, fresh, tol)
    problems += p_a
    ncc, _ = nuts_cross_check(committed["posteriors"])
    problems += nuts_audit_problems(committed["nuts_cross_check"], ncc, committed["posteriors"])
    report = [f"    [{c:<10}] tol={tol[c]:.3g}  worst={d:.3g} ({d / tol[c]:.1%} of tolerance)  committed {cv!r} "
              f"vs fresh {fv!r}  {p}" for c, (d, p, cv, fv) in sorted(worst.items())]
    for name, f in ncc.items():
        report.append(f"    [nuts {name}] max r_hat {f['max_rhat']:.4f}  fresh min ess {f['min_ess']:.0f} "
                      f"({f['min_ess'] / AUDIT_ESS_FLOOR:.2f}x the floor)  divergences {f['divergences']}  "
                      f"max mean offset {max(abs(f['mean'][q] - committed['posteriors'][name][q]['mean']) / f['mcse'][q] for q in calibrate.QUAD_QUANTITIES):.2f} MCSE")
    if problems:
        raise SystemExit("calibration audit FAILED:\n  " + "\n  ".join(problems) + "\n  per-class worst:\n"
                         + "\n".join(report))
    print(f"calibration audit OK: {len(quad_cells(committed))} quadrature cells within their class tolerance, "
          f"documents equal to their render, inputs current, NUTS cross-check within its classes")
    print("\n".join(report))


# ============================================================================= measurement steps
def converge(name, out):
    pub = _posterior(name, res=RES_PUB)
    fine = _posterior(name, res=2 * RES_PUB)
    rec = {"name": name, "res_pub": RES_PUB, "res_fine": 2 * RES_PUB, "pub": _cells_of(pub), "fine": _cells_of(fine)}
    rec["failures"] = calibrate.convergence_failures(rec["pub"], rec["fine"])
    dump(rec, out)
    for cell, d, tol in calibrate.convergence_changes(rec["pub"], rec["fine"]):
        print(f"{name:24s} {cell:28s} {d:.3e}  criterion {tol:g}  {'OK' if d <= tol else 'FAIL'}")
    print(f"{name}: {len(rec['failures'])} failure(s)")
    return 1 if rec["failures"] else 0


def _class_max(rec):
    out = {"moment": 0.0, "quantile": 0.0, "quantile_L": 0.0}
    for cell, d, _ in calibrate.convergence_changes(rec["pub"], rec["fine"]):
        q, st = cell.split(".", 1)
        c = _quantile_class(q, st)
        out[c] = max(out[c], d)
    return out


def record_convergence(paths):
    recs = {r["name"]: r for r in (load(p) for p in paths)}
    if sorted(recs) != sorted(CONVERGENCE_POSTERIORS):
        raise SystemExit(f"convergence runs {sorted(recs)} != {sorted(CONVERGENCE_POSTERIORS)}")
    bad = {n: calibrate.convergence_failures(r["pub"], r["fine"]) for n, r in recs.items()}
    if any(bad.values()) or any(r["res_pub"] != RES_PUB for r in recs.values()):
        raise SystemExit(f"convergence unmet at res {RES_PUB}: {bad}")
    cells = load()
    cells["convergence"] = {
        "res_pub": RES_PUB, "res_fine": 2 * RES_PUB,
        "criteria": {"moment": calibrate.CONV_RTOL_MOMENT, "quantile": calibrate.CONV_RTOL_Q,
                     "quantile_L": calibrate.CONV_RTOL_Q_L},
        "posteriors": {n: _class_max(r) for n, r in recs.items()},
    }
    dump(cells, CELLS_JSON)
    print(json.dumps(cells["convergence"], indent=2, sort_keys=True))


def record_scatter(path):
    cells, other = load(), load(path)
    a = {p: v for p, v, _ in quad_cells(cells)}
    scatter = {c: 0.0 for c in QUAD_TOL_CAP}
    for p, v, c in quad_cells(other):
        scatter[c] = max(scatter[c], calibrate._rel(a[p], v))
    ea, eb = _exact_cells(cells), _exact_cells(other)
    if ea != eb:
        raise SystemExit("exact cells differ between the platforms")
    cells["audit_scatter"] = {"platforms": [cells["computed_on"], other["computed_on"]], "scatter": scatter,
                              "tolerance": quad_tolerances(scatter)}
    dump(cells, CELLS_JSON)
    print(json.dumps(cells["audit_scatter"], indent=2, sort_keys=True))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--audit" in argv:
        audit()
    elif "--render" in argv:
        cells = load()
        probs = input_problems(cells)
        if probs:
            raise SystemExit("render refused: " + "; ".join(probs))
        render(cells)
        print(f"rendered {CALIBRATION_MD} and {NOTE_PATH} from {CELLS_JSON}")
    elif "--cells-out" in argv:
        dump(compute_cells(), argv[argv.index("--cells-out") + 1])
    elif "--converge" in argv:
        raise SystemExit(converge(argv[argv.index("--converge") + 1], argv[argv.index("--out") + 1]))
    elif "--record-convergence" in argv:
        record_convergence(argv[argv.index("--record-convergence") + 1:])
    elif "--record-scatter" in argv:
        record_scatter(argv[argv.index("--record-scatter") + 1])
    else:
        regenerate()


if __name__ == "__main__":
    main()
