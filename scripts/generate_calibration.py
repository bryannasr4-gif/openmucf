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
input digests the cells record are the live files'. Every reader that renders, audits or publishes from the cells
file takes it only through `admit_calibration_cells`; the regeneration, which rewrites the file, renders only from
the record it writes, once that record (the parts it carries from the old file included) passes the same checks.
"""

import contextlib
import copy
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import dataclass
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
#: The documents as released at RELEASED_TAG, whose printed cells the shipped record holds: RELEASED_DOCUMENTS carries
#: each text verbatim, and each must have the git blob id the tag gives it (`git rev-parse v1.3.0:<document>`).
RELEASED_TAG = "v1.3.0"
RELEASED_COMMIT = "66a87e869214713f5e1fbd3a8725a5f78442089d"
RELEASED_BLOBS = {
    "CALIBRATION.md": "a8db7f228ed6e670f07db2cf27163e6366f10c9e",
    "VALIDATION.md": "c5c175161994afe31ad76a6c606e3acbe8ded8a6",
    "VALIDATION_CHANNELS.md": "931dba6d5dd9bacc40d02ee2fbedadd99fb7543e",
}
RELEASED_DOCUMENTS = "data/calibration/released_v1.3.0.json"


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


def _sweep_config(r_lbl, o_lbl, l_lbl):
    return f"R={r_lbl}/os0={o_lbl}/lc={l_lbl}"


def _sweep_grid():
    """Config name -> its three boxes, in the order `_sweep_rows` computes them."""
    return {
        _sweep_config(r_lbl, o_lbl, l_lbl): {"R_box": list(r_box), "os0_box": list(o_box), "lc_box": list(l_box)}
        for r_box, r_lbl in _R_BOXES for o_box, o_lbl in _OS0_BOXES for l_box, l_lbl in _LC_BOXES
    }


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
                    "config": _sweep_config(r_lbl, o_lbl, l_lbl),
                    "R_box": list(r_box), "os0_box": list(o_box), "lc_box": list(l_box),
                    "corr": float(post["corr_omega_s0_R"]),
                    "R_sd": float(post["R"]["sd"]),
                    "ose_mean": float(post["omega_s_eff_pct"]["mean"]),
                    "ose_sd": float(post["omega_s_eff_pct"]["sd"]),
                    "rails": bool(_rails(post, boxes)),
                })
    return rows


SWEEP_SCALARS = ("corr", "R_sd", "ose_mean", "ose_sd")
SWEEP_FIELDS = ("config", "R_box", "os0_box", "lc_box", "rails", *SWEEP_SCALARS)


def sweep_problems(rows):
    """The sweep as `_sweep_rows` writes it, checked on the raw list before anything keys it by config: one row
    per grid config, in grid order, each with the fields SWEEP_FIELDS and no other, its config's own boxes and a
    Boolean railing flag. A repeated config is named, never collapsed into one row. (The four scalars of a row
    are checked with the other measured scalars, in `record_problems`.)"""
    if not isinstance(rows, list):
        return [f"sweep: {type(rows).__name__} is not a list of rows"]
    grid = _sweep_grid()
    probs, configs = [], []
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or set(r) != set(SWEEP_FIELDS):
            got = sorted(r) if isinstance(r, dict) else type(r).__name__
            probs.append(f"sweep row {i}: fields {got} are not {sorted(SWEEP_FIELDS)}")
            continue
        cfg = r["config"]
        if not isinstance(cfg, str):
            probs.append(f"sweep row {i}: config {cfg!r} is not a name")
            continue
        configs.append(cfg)
        for key, box in grid.get(cfg, {}).items():
            if r[key] != box:
                probs.append(f"sweep[{cfg}].{key}: {r[key]!r} is not its config's box {box!r}")
        if type(r["rails"]) is not bool:
            probs.append(f"sweep[{cfg}].rails: {r['rails']!r} is not a Boolean")
    counts = Counter(configs)
    probs += [f"sweep[{c}]: config repeated on {n} rows" for c, n in counts.items() if n > 1]
    probs += [f"sweep[{c}]: missing" for c in grid if c not in counts]
    probs += [f"sweep[{c}]: not a config of the grid" for c in counts if c not in grid]
    if not probs and configs != list(grid):
        probs.append(f"sweep: rows {configs} are not in the grid's order {list(grid)}")
    return probs


def checked_sweep(cells):
    """The sweep rows keyed by config, in grid order, for every reader that keys or renders them: raises
    ValueError naming each `sweep_problems` defect instead."""
    probs = sweep_problems(cells.get("sweep"))
    if probs:
        raise ValueError("sweep refused: " + "; ".join(probs))
    return {r["config"]: r for r in cells["sweep"]}


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


Q_NET_QUANTILES = (0.025, 0.5, 0.975)
Q_NET_STATS = ("mean", "sd") + tuple(f"q{p}" for p in Q_NET_QUANTILES)


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
    for p in Q_NET_QUANTILES:
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


def _estimator():
    """The estimator settings the cells are computed with, as the cells file records them."""
    return {
        "res": RES_PUB, "sweep_res": SWEEP_RES, "grid": dict(calibrate.QUAD_GRID),
        "bins": {k: v[2] for k, v in calibrate.QUAD_BINS.items()},
        "l_other_pct": list(sidechannels.ANCHOR_L_OTHER_PCT),
        "lambda_c_box": list(calibrate.LAMBDA_C_PRIOR_W), "covariance": "L3",
    }


def compute_cells(posts=None):
    if posts is None:
        posts = {name: _posterior(name) for name in POSTERIORS}
    return {
        "schema": SCHEMA,
        "estimator": _estimator(),
        "posteriors": {name: _cells_of(p) for name, p in posts.items()},
        "sweep": _sweep_rows(),
        "q_net": q_net_cells(posts["weak"]),
        "legacy_tension": legacy_tension(),
        "inputs": _inputs(),
        "computed_on": _computed_on(),
    }


#: the chains `nuts_cross_check` records
NUTS_CHAINS = ("weak", "kamimura")


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


def _cells_text(cells):
    """The bytes of a cells file, as text: canonical JSON (sorted keys, indent 2, ASCII) and a final newline."""
    return json.dumps(cells, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def dump(cells, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_cells_text(cells))


def load(path=CELLS_JSON):
    """A cells or run record, refused (ValueError naming the file and the key) if any object in it repeats a key
    (`openmucf.provenance.load_json`). Nothing renders or publishes what this returns: the documents are built only
    from what `admit_calibration_cells` admits, and the measurement steps that rewrite the cells file
    (`regenerate`, `record_convergence`, `record_scatter`) admit what they carry or bind."""
    return provenance.load_json(path)


# ================================================================================= the admission gate
class _FrozenDict(dict):
    """A dict that refuses every change in place; a deep copy of it is an ordinary dict."""

    def _refuse(self, *args, **kwargs):
        raise TypeError("the admitted calibration cells are read-only")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _refuse

    def __copy__(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return {k: copy.deepcopy(v, memo) for k, v in self.items()}

    def __reduce__(self):
        return dict, (dict(self),)


class _FrozenList(list):
    """A list that refuses every change in place; a deep copy of it is an ordinary list."""

    def _refuse(self, *args, **kwargs):
        raise TypeError("the admitted calibration cells are read-only")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse
    append = clear = extend = insert = pop = remove = reverse = sort = _refuse

    def __copy__(self):
        return list(self)

    def __deepcopy__(self, memo):
        return [copy.deepcopy(v, memo) for v in self]

    def __reduce__(self):
        return list, (list(self),)


def _freeze(node):
    if isinstance(node, dict):
        return _FrozenDict({k: _freeze(v) for k, v in node.items()})
    if isinstance(node, list):
        return _FrozenList(_freeze(v) for v in node)
    return node


@dataclass(frozen=True)
class AdmittedCells:
    """The cells file as `admit_calibration_cells` admitted it: ``data`` is the record, read-only; ``sha256`` is the
    digest of the admitted bytes by `openmucf.provenance.file_sha256`'s recipe; ``path`` is the file read."""

    path: str
    sha256: str
    data: dict


class CellsRefused(SystemExit):
    """The gate's refusal of a cells file: ``problems`` names each defect."""

    def __init__(self, path, problems):
        self.problems = list(problems)
        super().__init__(f"{path} refused: " + "; ".join(self.problems))


def _admit_bytes(raw, path, *, sampled=True):
    """The record ``raw`` (the bytes of a cells file) admitted by `admission_problems`, or CellsRefused."""
    try:
        text = raw.decode("utf-8")
        cells = provenance.loads_json(text, path)
    except ValueError as e:
        raise CellsRefused(path, [f"not a JSON record without repeated keys ({e})"]) from None
    probs = admission_problems(cells, sampled=sampled)
    if probs:
        raise CellsRefused(path, probs)
    return AdmittedCells(path=str(path), sha256=provenance.text_sha256(text), data=_freeze(cells))


def admit_calibration_cells(path=CELLS_JSON):
    """The one admission of the calibration cells file: every reader that renders, audits or publishes from it (the
    render and the audit here, scripts/generate_findings.py and scripts/generate_frontier.py) takes only what this
    returns. Reads the bytes at ``path`` once, refuses them (CellsRefused, a SystemExit naming each problem) unless
    `admission_problems` passes the record they hold, and returns that record read-only with the digest of those
    bytes (`AdmittedCells`)."""
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise CellsRefused(path, [f"unreadable ({type(e).__name__}: {e})"]) from None
    return _admit_bytes(raw, path)


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
    by = dict(checked_sweep(cells))
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
        d = cells["nuts_cross_check"][{"weak": "weak", "Kamimura": "kamimura"}[row]]
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
        c = {_QNET_LABEL_OLD: cells["q_net"], "X_mu (weak-box posterior)": po["weak"]["X_mu"],
             "X_mu (Kamimura posterior)": po["kamimura"]["X_mu"]}[row]
        return {"mean": f"{c['mean']:.4g}", "sd": _g(c["sd"]),
                "95% CI": f"[{c['q0.025']:.4g}, {c['q0.975']:.4g}]"}[col]
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
    corrs = [r["corr"] for r in checked_sweep(cells).values()]
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
    sweep = list(checked_sweep(cells).values())
    head = (f"## Prior-sensitivity sweep (weak-prior mode; quadrature at res {cells['estimator']['sweep_res']:g})\n"
            "| config | boxes | corr | R sd | ose mean | ose sd | rails? |\n|---|---|---|---|---|---|---|\n")
    rows = [
        f"| {r['config']} | {_boxes_label(r['R_box'], r['os0_box'], r['lc_box'])} | {r['corr']:.2f} | "
        f"{_g(r['R_sd'])} | {_g(r['ose_mean'])} | {_g(r['ose_sd'])} | {'yes' if r['rails'] else 'no'} |"
        for r in sweep
    ]
    for key, lbl in (("l2_default", "default box; L2 likelihood"), ("l1_default", "default box; L1 likelihood")):
        p = cells["posteriors"][key]
        rows.append(f"| {lbl} | default | {p['corr_omega_s0_R']:.2f} | {_g(p['R']['sd'])} | "
                    f"{_g(p['omega_s_eff_pct']['mean'])} | {_g(p['omega_s_eff_pct']['sd'])} | - |")
    lo, hi = min(r["corr"] for r in sweep), max(r["corr"] for r in sweep)      # the rows read above, checked once
    ose = [r["ose_mean"] for r in sweep]
    n_rail = sum(bool(r["rails"]) for r in sweep)
    rails = ("it is `no` for every config here" if n_rail == 0
             else f"it is `yes` for {n_rail} of the {len(sweep)} configs")
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
    """The card's scenario-A predictions keyed by target; refused (ValueError) unless the card has one scenario A
    and no target repeats in it."""
    card = load(forecast.CARD_PATH)
    sc = [s for s in card["payload"]["scenarios"] if s["name"] == "A"]
    if len(sc) != 1:
        raise ValueError(f"{forecast.CARD_PATH}: {len(sc)} scenarios named A")
    preds = sc[0]["predictions"]
    out = {p["target_id"]: p for p in preds}
    if len(out) != len(preds):
        raise ValueError(f"{forecast.CARD_PATH}: scenario A repeats a target_id")
    return out


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
    the existing cells file (they are written by the steps that measure them); its computed part is replaced.
    Before it samples, the new cells and every record carried with them pass the gate's checks, the NUTS record
    it is about to measure aside (`admission_problems` with ``sampled`` False); the record it writes is then
    admitted as the exact bytes it writes, and the documents are rendered from what was admitted."""
    old = load()
    for key in ("shipped_v1_3_0", "convergence"):
        if key not in old:
            raise SystemExit(f"{CELLS_JSON} has no {key!r}: it is written before the cells are computed")
    posts = {name: _posterior(name) for name in POSTERIORS}
    cells = compute_cells(posts)
    for key in ("shipped_v1_3_0", "convergence", "audit_scatter"):
        if key in old:
            cells[key] = old[key]
    cells = json.loads(json.dumps(cells))      # the values the cells file will hold
    probs = admission_problems(cells, sampled=False)
    if probs:
        raise SystemExit("regeneration refused: before sampling, the cells or a record carried with them:\n  "
                         + "\n  ".join(probs))
    ncc, weak_draws = nuts_cross_check(cells["posteriors"])
    bad = {k: calibrate.nuts_gate_failures(v, cells["posteriors"][k]) for k, v in ncc.items()}
    if any(bad.values()):
        raise SystemExit(f"NUTS cross-check FAILED: {bad}")
    cells["nuts_cross_check"] = ncc
    text = _cells_text(cells)
    try:
        admitted = _admit_bytes(text.encode("utf-8"), CELLS_JSON)     # what every consumer of the file will check
    except CellsRefused as e:
        raise SystemExit("regeneration refused: the record to be written:\n  " + "\n  ".join(e.problems)) from None
    texts = _render_texts(admitted.data)        # every refusal comes before the first write
    _write_text(CELLS_JSON, text)
    for path, doc in texts.items():
        _write_text(path, doc)
    _write_figure(weak_draws)
    print(f"wrote {CELLS_JSON}, {CALIBRATION_MD}, {NOTE_PATH} and figures/calibration.png")
    for name in ("weak", "kamimura"):
        p = admitted.data["posteriors"][name]
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
    """(path, value, class) of every compared scalar, read through `checked_sweep`: call on a record that
    `record_problems` admits."""
    out = []
    for name, post in cells["posteriors"].items():
        for q in calibrate.QUAD_QUANTITIES:
            for st in calibrate.STATS:
                out.append((f"posteriors.{name}.{q}.{st}", post[q][st], _quantile_class(q, st)))
        out.append((f"posteriors.{name}.corr_omega_s0_R", post["corr_omega_s0_R"], "moment"))
    for cfg, r in checked_sweep(cells).items():
        for k in SWEEP_SCALARS:
            out.append((f"sweep[{cfg}].{k}", r[k], "moment"))
    for st, v in cells["q_net"].items():
        out.append((f"q_net.{st}", v, _quantile_class("Q_net", st)))
    for k, v in cells["legacy_tension"].items():
        out.append((f"legacy_tension.{k}", v, "moment"))
    return out


def _exact_cells(cells):
    """The railing flags, compared exactly. (The boxes are their config's own by `sweep_problems`.)"""
    return {f"sweep[{cfg}].rails": r["rails"] for cfg, r in checked_sweep(cells).items()}


#: the computed part of a cells record, and the parts the measurement steps add to it
_COMPUTED_KEYS = ("schema", "estimator", "posteriors", "sweep", "q_net", "legacy_tension", "inputs", "computed_on")
_CARRIED_KEYS = ("shipped_v1_3_0", "convergence", "audit_scatter", "nuts_cross_check")
_PLATFORM_KEYS = ("platform", "python", "numpy")


def _raw_scalars(rec):
    """(key path, value) of every leaf of the measured groups, walked on the raw record: nested objects are
    descended, anything else is a leaf, and the sweep contributes its rows' SWEEP_SCALARS as ("sweep", config,
    field) (call on a sweep that `sweep_problems` admits). Paths are tuples of keys, so a key that contains a dot
    cannot stand in for two levels."""
    out = []

    def walk(path, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk((*path, k), v)
        else:
            out.append((path, node))

    for group in ("posteriors", "q_net", "legacy_tension"):
        walk((group,), rec[group])
    out += [(("sweep", r["config"], k), r[k]) for r in rec["sweep"] for k in SWEEP_SCALARS]
    return out


def _expected_leaves():
    """The key paths `_raw_scalars` must find, from the definitions (the same scalars as `expected_quad_paths`)."""
    out = set()
    for name in POSTERIORS:
        out |= {("posteriors", name, q, st) for q, st in _CONV_MEMBERS}
        out.add(("posteriors", name, "corr_omega_s0_R"))
    out |= {("sweep", cfg, k) for cfg in _sweep_grid() for k in SWEEP_SCALARS}
    out |= {("q_net", st) for st in Q_NET_STATS}
    out |= {("legacy_tension", k) for k in legacy_tension()}
    return out


def _path_text(path):
    """A key path as `quad_cells` names it."""
    if path[0] == "sweep" and len(path) == 3:
        return f"sweep[{path[1]}].{path[2]}"
    return ".".join(str(k) for k in path)


def _platform_problems(label, on):
    if not isinstance(on, dict) or set(on) != set(_PLATFORM_KEYS) or not all(isinstance(v, str) for v in on.values()):
        return [f"{label}: {on!r} is not a platform description {list(_PLATFORM_KEYS)}"]
    return []


def record_problems(rec, label):
    """A computed cells record, checked on its raw structure before any reader keys or projects it: it holds
    every computed part and nothing besides them and the parts the measurement steps add; its schema and
    estimator are the live ones; its sweep passes `sweep_problems`; its measured scalars are those of
    `_expected_leaves()`, none missing and none extra, each a finite int or float and never a Boolean; and
    computed_on describes a platform."""
    if not isinstance(rec, dict):
        return [f"{label}: {type(rec).__name__} is not a cells record"]
    probs = [f"{label}: no {k!r}" for k in _COMPUTED_KEYS if k not in rec]
    probs += [f"{label}: unexpected member {k!r}" for k in sorted(set(rec) - {*_COMPUTED_KEYS, *_CARRIED_KEYS})]
    if probs:
        return probs
    if rec["schema"] != SCHEMA:
        probs.append(f"{label}: schema {rec['schema']!r} is not {SCHEMA!r}")
    if rec["estimator"] != _estimator():
        probs.append(f"{label}: estimator {rec['estimator']!r} is not the live estimator {_estimator()!r}")
    probs += _platform_problems(f"{label}: computed_on", rec["computed_on"])
    sweep = sweep_problems(rec["sweep"])
    if sweep:
        return probs + [f"{label}: {m}" for m in sweep]
    got = _raw_scalars(rec)
    want = _expected_leaves()
    have = {p for p, _ in got}
    missing = sorted(_path_text(p) for p in want - have)
    extra = sorted(_path_text(p) for p in have - want)
    probs += [f"{label}: {len(missing)} scalar(s) missing, e.g. {missing[:3]}"] if missing else []
    probs += [f"{label}: {len(extra)} unexpected scalar(s), e.g. {extra[:3]}"] if extra else []
    probs += [f"{label}: {_path_text(p)} = {v!r} is not a finite number" for p, v in got if not _finite(v)]
    return probs


def compare_cells(committed, fresh, tol):
    """(a): fresh quadrature cells against the committed ones, each within its class tolerance. Both records are
    admitted by `record_problems` first, so the two sides hold the same expected cells, read from rows whose
    configs are unique."""
    problems = record_problems(committed, "committed") + record_problems(fresh, "fresh")
    if problems:
        return problems, {}
    a = {p: (v, c) for p, v, c in quad_cells(committed)}
    b = {p: v for p, v, _ in quad_cells(fresh)}
    worst = {}
    for p, (v, c) in a.items():
        d = calibrate._rel(v, b[p])
        if c not in worst or d > worst[c][0]:
            worst[c] = (d, p, v, b[p])
        if not d <= tol[c]:
            problems.append(f"{p}: committed {v!r} vs fresh {b[p]!r}, relative {d:.3g} > tol {tol[c]:.3g}")
    ea, eb = _exact_cells(committed), _exact_cells(fresh)
    problems += [f"{p}: committed {ea[p]!r} vs fresh {eb[p]!r}" for p in ea if ea[p] != eb[p]]
    return problems, worst


def scatter_record_problems(sc):
    """The measured scatter as its consumers read it: platforms, scatter and tolerance and no other member; two
    platform descriptions; one scatter per QUAD_TOL_CAP class, each a finite number at least 0."""
    if not isinstance(sc, dict) or set(sc) != {"platforms", "scatter", "tolerance"}:
        got = sorted(sc) if isinstance(sc, dict) else type(sc).__name__
        return [f"audit_scatter: members {got} are not ['platforms', 'scatter', 'tolerance']"]
    probs = []
    plats = sc["platforms"]
    if not isinstance(plats, list) or len(plats) != 2:
        probs.append(f"audit_scatter.platforms: {plats!r} is not two platform descriptions")
    else:
        for i, on in enumerate(plats):
            probs += _platform_problems(f"audit_scatter.platforms[{i}]", on)
    s = sc["scatter"]
    if not isinstance(s, dict) or set(s) != set(QUAD_TOL_CAP):
        return probs + [f"audit_scatter.scatter: {s!r} does not hold one value per class {sorted(QUAD_TOL_CAP)}"]
    probs += [f"audit_scatter.scatter.{c}: {v!r} is not a finite number at least 0" for c, v in s.items()
              if not (_finite(v) and v >= 0)]
    return probs


def committed_tolerances(cells):
    """The class tolerances the committed record sets: the floor before a scatter is measured, else the rule of
    the recorded scatter, which `scatter_record_problems` admits and whose recorded tolerance must be the rule's."""
    sc = cells.get("audit_scatter")
    if sc is None:
        return {c: QUAD_TOL_FLOOR for c in QUAD_TOL_CAP}
    probs = scatter_record_problems(sc)
    if probs:
        raise SystemExit("audit_scatter refused:\n  " + "\n  ".join(probs))
    tol = quad_tolerances(sc["scatter"])
    if sc.get("tolerance") != tol:
        raise SystemExit(f"audit_scatter.tolerance {sc.get('tolerance')} is not the rule's {tol}")
    return tol


def input_problems(cells):
    """(d): the cells record one digest per INPUT_FILES file and no other, each the live file's."""
    live = _inputs()
    rec = cells.get("inputs")
    if not isinstance(rec, dict):
        return [f"inputs: {rec!r} is not a record of input digests"]
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


_NUTS_FIELDS = ("mean", "mcse", "max_rhat", "min_ess", "divergences", "max_abs_dev_over_mcse")


def _nuts_offset(d, quad):
    return max(abs(d["mean"][q] - quad[q]["mean"]) / d["mcse"][q] for q in calibrate.QUAD_QUANTITIES)


def nuts_record_problems(cells):
    """The stored NUTS cross-check as CALIBRATION.md renders it, on a record `record_problems` admits: one record
    per NUTS_CHAINS chain with the fields _NUTS_FIELDS and no other; a finite mean and a positive finite MCSE per
    QUAD_QUANTITIES quantity; finite diagnostics and an integer divergence count. It must pass the gate it was
    written under against the cells' quadrature, and its rendered offset must be the one its means and MCSE give."""
    rec = cells.get("nuts_cross_check")
    if not isinstance(rec, dict) or set(rec) != set(NUTS_CHAINS):
        got = sorted(rec) if isinstance(rec, dict) else rec
        return [f"nuts_cross_check: chains {got!r} are not {sorted(NUTS_CHAINS)}"]
    probs = []
    for name in NUTS_CHAINS:
        d, lbl = rec[name], f"nuts_cross_check.{name}"
        if not isinstance(d, dict) or set(d) != set(_NUTS_FIELDS):
            got = sorted(d) if isinstance(d, dict) else type(d).__name__
            probs.append(f"{lbl}: fields {got} are not {sorted(_NUTS_FIELDS)}")
            continue
        bad = []
        for key in ("mean", "mcse"):
            if not isinstance(d[key], dict) or set(d[key]) != set(calibrate.QUAD_QUANTITIES):
                bad.append(f"{lbl}.{key}: {d[key]!r} does not hold one value per quantity")
                continue
            bad += [f"{lbl}.{key}.{q}: {v!r} is not a finite number" for q, v in d[key].items() if not _finite(v)]
            if key == "mcse":
                bad += [f"{lbl}.mcse.{q}: {v!r} is not positive" for q, v in d[key].items() if _finite(v) and v <= 0]
        bad += [f"{lbl}.{key}: {d[key]!r} is not a finite number" for key in ("max_rhat", "min_ess",
                                                                              "max_abs_dev_over_mcse")
                if not _finite(d[key])]
        if type(d["divergences"]) is not int:
            bad.append(f"{lbl}.divergences: {d['divergences']!r} is not a count")
        if bad:
            probs += bad
            continue
        quad = cells["posteriors"][name]
        probs += [f"{lbl}: {m}" for m in calibrate.nuts_gate_failures(d, quad)]
        if d["max_abs_dev_over_mcse"] != _nuts_offset(d, quad):
            probs.append(f"{lbl}.max_abs_dev_over_mcse: {d['max_abs_dev_over_mcse']!r} is not its means' offset "
                         f"{_nuts_offset(d, quad)!r}")
    return probs


#: the validation-page prints the shipped record carries besides CALIBRATION.md's cells, as (document, target); the
#: FC-001 note reads the second (`vb_figures`)
SHIPPED_VALIDATION = (("VALIDATION.md", "V_petitjean"), ("VALIDATION_CHANNELS.md", "V_breunlich_lambdac"))
SHIPPED_VALIDATION_KEYS = tuple(f"{doc}|{target}|predicted" for doc, target in SHIPPED_VALIDATION)


def git_blob_id(data: bytes) -> str:
    """The id git gives a file holding ``data``: SHA-1 over the header 'blob <size>' with a NUL, then the bytes."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _validation_print(md_text, target):
    """The `predicted` cell of ``target``'s row of a validation page's table."""
    lines = md_text.splitlines()
    head = [ln for ln in lines if ln.startswith("| target |")]
    rows = [ln for ln in lines if ln.startswith(f"| {target} |")]
    if len(head) != 1 or len(rows) != 1:
        raise ValueError(f"{len(head)} header row(s) and {len(rows)} {target} row(s)")
    cols = [c.strip() for c in head[0].strip().strip("|").split("|")]
    cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    return cells[cols.index("predicted")]


def released_prints(path=RELEASED_DOCUMENTS):
    """(prints, problems): the record the shipped record must equal, derived from the documents as released. The
    file at ``path`` must hold RELEASED_TAG, RELEASED_COMMIT and the RELEASED_BLOBS documents, each text with its
    released git blob id. prints maps "CALIBRATION.md|<section>|<row>|<column>" to every data cell of the released
    CALIBRATION.md's tables as `_parse_tables` reads them, and "<document>|<target>|predicted" to each
    SHIPPED_VALIDATION print. When the file fails, prints is None and problems names each failure."""
    try:
        rec = provenance.load_json(path)
    except (OSError, ValueError) as e:
        return None, [f"{path}: the released documents are unreadable ({type(e).__name__}: {e})"]
    want = ["commit", "documents", "tag"]
    if not isinstance(rec, dict) or sorted(rec) != want:
        got = sorted(rec) if isinstance(rec, dict) else type(rec).__name__
        return None, [f"{path}: members {got} are not {want}"]
    probs = []
    if rec["tag"] != RELEASED_TAG or rec["commit"] != RELEASED_COMMIT:
        probs.append(f"{path}: {rec['tag']!r} at {rec['commit']!r} is not {RELEASED_TAG} at {RELEASED_COMMIT}")
    docs = rec["documents"]
    if not isinstance(docs, dict) or sorted(docs) != sorted(RELEASED_BLOBS):
        got = sorted(docs) if isinstance(docs, dict) else type(docs).__name__
        return None, probs + [f"{path}: documents {got} are not {sorted(RELEASED_BLOBS)}"]
    for name, blob in RELEASED_BLOBS.items():
        got = git_blob_id(docs[name].encode("utf-8")) if isinstance(docs[name], str) else None
        if got != blob:
            probs.append(f"{path}: {name} is not the document released at {RELEASED_TAG} (git blob {got}, "
                         f"released {blob})")
    if probs:
        return None, probs
    prints = {}
    for title, header, rows in _parse_tables(docs[CALIBRATION_MD]):
        if header is None:
            continue
        for r in rows:
            for col, val in zip(header[1:], r[1:], strict=True):
                key = f"{CALIBRATION_MD}|{title}|{r[0]}|{col}"
                if key in prints:
                    return None, [f"{path}: the released {CALIBRATION_MD} prints {key!r} twice"]
                prints[key] = val
    for doc, target in SHIPPED_VALIDATION:
        try:
            prints[f"{doc}|{target}|predicted"] = _validation_print(docs[doc], target)
        except ValueError as e:
            return None, [f"{path}: the released {doc}: {e}"]
    return prints, []


def shipped_problems(cells, *, sampled=True):
    """The record of the documents as released (``shipped_v1_3_0``), whose values the amendment table prints as
    "as printed before" and the FC-001 note as the V_breunlich_lambdac print it had, on a record `record_problems`
    admits: it must be exactly `released_prints()` -- each key derived from the released documents present, no
    other key, every value the released print -- and each CALIBRATION.md key must name a cell this record renders.
    With ``sampled`` False (before the NUTS record exists) the released convergence-table keys, whose cells are that
    record's, are held to the released prints alone; they are resolved once it is stored."""
    sh = cells.get("shipped_v1_3_0")
    if not isinstance(sh, dict):
        return [f"shipped_v1_3_0: {type(sh).__name__} is not a record"]
    want, problems = released_prints()
    if want is None:
        return [f"shipped_v1_3_0: not checkable against the released documents: {p}" for p in problems]
    probs = [f"shipped_v1_3_0: no {k!r} (released {want[k]!r})" for k in sorted(want) if k not in sh]
    probs += [f"shipped_v1_3_0[{k!r}]: not a cell of the released documents" for k in sorted(sh) if k not in want]
    for key in sorted(set(sh) & set(want)):
        lbl = f"shipped_v1_3_0[{key!r}]"
        if sh[key] != want[key]:
            probs.append(f"{lbl}: {sh[key]!r} is not the released print {want[key]!r}")
        elif key.startswith(f"{CALIBRATION_MD}|") and (sampled or key.split("|")[1] != _TITLE_CONV_OLD):
            try:
                _now_value(cells, *key.split("|")[1:])
            except (KeyError, TypeError, ValueError) as e:
                probs.append(f"{lbl}: names no cell of this record ({type(e).__name__}: {e})")
    return probs


def scatter_problems(cells):
    """The measured scatter record and the tolerance it sets (`committed_tolerances`), as problems. A record without
    one is the record before the scatter is measured, at the floor tolerance."""
    sc = cells.get("audit_scatter")
    if sc is None:
        return []
    probs = scatter_record_problems(sc)
    if not probs and sc["tolerance"] != quad_tolerances(sc["scatter"]):
        probs.append(f"audit_scatter.tolerance {sc['tolerance']} is not the rule's {quad_tolerances(sc['scatter'])}")
    return probs


def admission_problems(cells, *, sampled=True):
    """Every check of `admit_calibration_cells`, on a record: the computed record (`record_problems`); then its input
    digests against the live files (`input_problems`), the shipped record against the released documents
    (`shipped_problems`), the scatter record and its tolerance (`scatter_problems`), the convergence record at that
    tolerance (`convergence_problems`) and the NUTS record (`nuts_record_problems`). With ``sampled`` False the NUTS
    record is not checked: that is the regeneration's check before it samples the record it will store."""
    probs = record_problems(cells, "cells")
    if probs:
        return probs
    probs = input_problems(cells) + shipped_problems(cells, sampled=sampled)
    scatter = scatter_problems(cells)
    if scatter:
        return probs + scatter
    probs += convergence_problems(cells, committed_tolerances(cells))
    if sampled:
        probs += nuts_record_problems(cells)
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
    try:                # a record the gate refuses is not rendered, recomputed or sampled against
        committed = admit_calibration_cells().data
    except CellsRefused as e:
        raise SystemExit("calibration audit FAILED:\n  " + "\n  ".join(e.problems)) from None
    problems = render_problems(committed)
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
                      f"max mean offset {_nuts_offset(f, committed['posteriors'][name]):.2f} MCSE")
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


def _finite(v):
    """A measured scalar: an int or float (never a bool, None or string) that is finite."""
    return type(v) in (int, float) and math.isfinite(v)


def _criteria():
    return {"moment": calibrate.CONV_RTOL_MOMENT, "quantile": calibrate.CONV_RTOL_Q,
            "quantile_L": calibrate.CONV_RTOL_Q_L}


_CONV_MEMBERS = [(q, st) for q in calibrate.QUAD_QUANTITIES for st in calibrate.STATS]
#: every stored member of a run's posterior: each quantity's stats and the omega_s0-R correlation
_RUN_MEMBERS = [*_CONV_MEMBERS, ("corr_omega_s0_R", None)]


def _member(post, q, st):
    v = post.get(q) if isinstance(post, dict) else None
    if st is None:
        return v
    return v.get(st) if isinstance(v, dict) else None


def _posterior_member_problems(label, post):
    """A stored posterior holds the members _RUN_MEMBERS and no other, each a finite number."""
    if not isinstance(post, dict):
        return [f"{label}: {post!r} is not a posterior"]
    want = {q for q, _ in _RUN_MEMBERS}
    probs = [f"{label}: unexpected member(s) {sorted(set(post) - want)}"] if set(post) - want else []
    for q in calibrate.QUAD_QUANTITIES:
        extra = sorted(set(post[q]) - set(calibrate.STATS)) if isinstance(post.get(q), dict) else []
        probs += [f"{label}.{q}: unexpected stat(s) {extra}"] if extra else []
    for q, st in _RUN_MEMBERS:
        v = _member(post, q, st)
        if not _finite(v):
            probs.append(f"{label}.{q}{'' if st is None else '.' + st}: {v!r} is not a finite number")
    return probs


def _run_problems(name, run):
    """A convergence run of ``name``: publication resolution RES_PUB, fine resolution twice it, both posteriors
    holding exactly their stored members, each a finite number (the correlation included), and every change
    within its criterion."""
    probs = []
    if run.get("res_pub") != RES_PUB:
        probs.append(f"{name}: res_pub {run.get('res_pub')!r} is not the publication resolution {RES_PUB:g}")
    if run.get("res_fine") != 2 * RES_PUB:
        probs.append(f"{name}: res_fine {run.get('res_fine')!r} is not twice the publication resolution")
    for side in ("pub", "fine"):
        probs += _posterior_member_problems(f"{name}.{side}", run.get(side))
    if not probs:
        probs += [f"{name}: {m}" for m in calibrate.convergence_failures(run["pub"], run["fine"])]
    return probs


def _binding_problems(name, pub, live, tol):
    """The run's publication-resolution posterior must be the live cells' ``name`` posterior: every member and the
    correlation within the audit tolerance of its class (the rule that accepts a recomputed cell as the same)."""
    if not isinstance(live, dict):
        return [f"{name}: the cells hold no {name!r} posterior for the run to certify"]
    probs = []
    for q, st in _RUN_MEMBERS:
        a, b = _member(pub, q, st), _member(live, q, st)
        cls = "moment" if st is None else _quantile_class(q, st)
        label = q if st is None else f"{q}.{st}"
        d = calibrate._rel(a, b) if _finite(a) and _finite(b) else math.inf
        if not d <= tol[cls]:
            probs.append(f"{name}.{label}: run {a!r} vs cells {b!r}, relative {d:.3g} > tol {tol[cls]:.3g}")
    return probs


_CONV_KEYS = ("res_pub", "res_fine", "criteria", "posteriors", "runs")


def convergence_problems(cells, tol):
    """The convergence record certifies these cells: it holds _CONV_KEYS and no other member, its resolutions are
    the estimator's and twice it, its criteria are the live ones, it holds the run (pub and fine, nothing else) of
    every CONVERGENCE_POSTERIORS posterior, each run meets _run_problems, the recorded maxima are that run's, and
    each run's publication posterior is the cells' own."""
    conv = cells.get("convergence")
    if not isinstance(conv, dict):
        return ["convergence: no record"]
    probs = []
    if set(conv) - set(_CONV_KEYS):
        probs.append(f"convergence: unexpected member(s) {sorted(set(conv) - set(_CONV_KEYS))}")
    res = cells.get("estimator", {}).get("res")
    if conv.get("res_pub") != RES_PUB or res != RES_PUB:
        probs.append(f"convergence.res_pub {conv.get('res_pub')!r} / estimator res {res!r} are not {RES_PUB:g}")
    if conv.get("res_fine") != 2 * RES_PUB:
        probs.append(f"convergence.res_fine {conv.get('res_fine')!r} is not twice the publication resolution")
    if conv.get("criteria") != _criteria():
        probs.append(f"convergence.criteria {conv.get('criteria')!r} are not the live criteria {_criteria()!r}")
    runs, maxima = conv.get("runs") or {}, conv.get("posteriors") or {}
    want = sorted(CONVERGENCE_POSTERIORS)
    if not isinstance(runs, dict) or not isinstance(maxima, dict) or sorted(runs) != want or sorted(maxima) != want:
        got = [sorted(x) if isinstance(x, dict) else type(x).__name__ for x in (runs, maxima)]
        return probs + [f"convergence: runs {got[0]} and maxima {got[1]} are not {want}"]
    for name in CONVERGENCE_POSTERIORS:
        if not isinstance(runs[name], dict) or set(runs[name]) != {"pub", "fine"}:
            got = sorted(runs[name]) if isinstance(runs[name], dict) else runs[name]
            probs.append(f"convergence: {name}: run members {got!r} are not ['fine', 'pub']")
            continue
        run = dict(runs[name], res_pub=conv.get("res_pub"), res_fine=conv.get("res_fine"))
        p = _run_problems(name, run)
        if not p:
            if _class_max(run) != maxima[name]:
                p.append(f"{name}: recorded maxima {maxima[name]} are not its run's {_class_max(run)}")
            p += _binding_problems(name, run["pub"], cells.get("posteriors", {}).get(name), tol)
        probs += [f"convergence: {m}" for m in p]
    return probs


def record_convergence(paths):
    """Write the convergence record from the ``--converge`` runs: refused unless there is one valid run per
    CONVERGENCE_POSTERIORS posterior. The cells file is read in one of two stated stages: before the first
    regeneration it holds none of the computed members (the runs are then recorded alone, and the regeneration
    binds them before it samples); once it holds any, it must hold the whole computed record, admitted with its
    live input digests (`record_problems`, `input_problems`), and each run's publication posterior must be the
    cells' own. The runs are stored with the record, so every consumer re-checks the binding."""
    runs = [load(p) for p in paths]
    names = [r.get("name") if isinstance(r, dict) else None for r in runs]
    if len(set(map(str, names))) != len(runs) or sorted(map(str, names)) != sorted(CONVERGENCE_POSTERIORS):
        raise SystemExit(f"convergence runs {names} != {sorted(CONVERGENCE_POSTERIORS)}")
    recs = {r["name"]: r for r in runs}
    probs = [m for n in CONVERGENCE_POSTERIORS for m in _run_problems(n, recs[n])]
    cells = load()
    computed = any(k in cells for k in _COMPUTED_KEYS)
    if not probs and computed:
        probs = record_problems(cells, "cells") or input_problems(cells)
    if not probs and computed:
        tol = committed_tolerances(cells)
        probs = [m for n in CONVERGENCE_POSTERIORS
                 for m in _binding_problems(n, recs[n]["pub"], cells["posteriors"].get(n), tol)]
    if probs:
        raise SystemExit("convergence record refused:\n  " + "\n  ".join(probs))
    cells["convergence"] = {
        "res_pub": RES_PUB, "res_fine": 2 * RES_PUB, "criteria": _criteria(),
        "posteriors": {n: _class_max(recs[n]) for n in CONVERGENCE_POSTERIORS},
        "runs": {n: {"pub": recs[n]["pub"], "fine": recs[n]["fine"]} for n in CONVERGENCE_POSTERIORS},
    }
    dump(cells, CELLS_JSON)
    print(json.dumps({k: v for k, v in cells["convergence"].items() if k != "runs"}, indent=2, sort_keys=True))


def expected_quad_paths():
    """(path, class) of every scalar `quad_cells` reads from a complete cells record, from the definitions: each
    POSTERIORS posterior, each sweep config, each Q_NET_STATS stat and each legacy_tension key."""
    out = set()
    for name in POSTERIORS:
        out |= {(f"posteriors.{name}.{q}.{st}", _quantile_class(q, st)) for q, st in _CONV_MEMBERS}
        out.add((f"posteriors.{name}.corr_omega_s0_R", "moment"))
    for cfg in _sweep_grid():
        out |= {(f"sweep[{cfg}].{k}", "moment") for k in SWEEP_SCALARS}
    out |= {(f"q_net.{st}", _quantile_class("Q_net", st)) for st in Q_NET_STATS}
    out |= {(f"legacy_tension.{k}", "moment") for k in legacy_tension()}
    return out


def record_scatter(path):
    """Write the measured cross-platform scatter: refused unless both records pass `record_problems`, were computed
    from the same inputs, those the live files have (`input_problems`), and carry the same railing flags."""
    cells, other = load(), load(path)
    probs = record_problems(cells, "this platform") + record_problems(other, "the other platform")
    if not probs:
        probs += input_problems(cells)
        if cells["inputs"] != other["inputs"]:
            probs.append(f"the platforms' inputs differ: {cells['inputs']} vs {other['inputs']}")
        ea, eb = _exact_cells(cells), _exact_cells(other)
        probs += [f"{p}: this platform {ea[p]!r}, the other platform {eb[p]!r}" for p in ea if ea[p] != eb[p]]
    if probs:
        raise SystemExit("scatter record refused:\n  " + "\n  ".join(probs))
    a = {p: v for p, v, _ in quad_cells(cells)}
    scatter = {c: 0.0 for c in QUAD_TOL_CAP}
    for p, v, c in quad_cells(other):
        scatter[c] = max(scatter[c], calibrate._rel(a[p], v))
    cells["audit_scatter"] = {"platforms": [cells["computed_on"], other["computed_on"]], "scatter": scatter,
                              "tolerance": quad_tolerances(scatter)}
    dump(cells, CELLS_JSON)
    print(json.dumps(cells["audit_scatter"], indent=2, sort_keys=True))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--audit" in argv:
        audit()
    elif "--render" in argv:
        try:
            admitted = admit_calibration_cells()
        except CellsRefused as e:
            raise SystemExit("render refused: " + "; ".join(e.problems)) from None
        render(admitted.data)
        print(f"rendered {CALIBRATION_MD} and {NOTE_PATH} from {CELLS_JSON}")
    elif "--cells-out" in argv:
        fresh = json.loads(json.dumps(compute_cells()))
        probs = record_problems(fresh, "computed cells")
        if probs:
            raise SystemExit("cells refused: " + "; ".join(probs))
        dump(fresh, argv[argv.index("--cells-out") + 1])
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
