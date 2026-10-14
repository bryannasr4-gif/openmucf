"""openmucf.sidechannels -- the side-channel loss per cycle at the liquid anchor fill of Breunlich et al.
(1987).

Breunlich et al. (1987) report the effective sticking after correcting the raw loss per cycle for the dmud,
tmut, pmud and pmut fusion losses (p.331), while their yield relation uses the raw loss (p.330 Eq. (2),
p.332). L_other is that difference at the anchor fill, the tt, dd and protium fusion losses plus the losses
to impurities and 3He:

    L_tt  = c_t * lambda_ttmu * omega_tt * T_t     T_t = 1/lambda_c(norm) - T_d   (tmu dwell, p.330 Eq. (3))
    L_dd  = c_d * q_1s * P_dd * br_he3 * omega_d   q_1s = 1/(1 + a*c_t), two-hyperfine-state dmu dwell
    L_p   = c_p * k_p / lambda_c(norm)
    L_imp

Every input, with its source and locator, is a row of ``data/side_channel_inputs.csv``.
"""

from __future__ import annotations

import copy
import csv
import functools
import itertools
import math
from pathlib import Path

import numpy as np

INPUTS_CSV = Path(__file__).resolve().parent / "data" / "side_channel_inputs.csv"

#: (central, sd) of L_other at the anchor fill in percent, as the calibration's likelihood uses them;
#: tests/test_sidechannels.py binds both to ``anchor_l_other()`` rounded to 4 decimals.
ANCHOR_L_OTHER_PCT = (0.1388, 0.0197)
ANCHOR_L_OTHER_FRACTION = ANCHOR_L_OTHER_PCT[0] / 100

_ANALYSES = ("i", "ii")
_STIELTJES_POINTS = 200_001


def _num(s):
    s = s.strip()
    return None if s in ("", "-") else float(s)


def load_inputs(path=INPUTS_CSV):
    """Rows of the inputs CSV keyed by symbol, numeric fields as floats (``None`` where the CSV has none)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out[r["symbol"]] = {
            "value": _num(r["value"]),
            "unc": _num(r["unc"]),
            "distribution": r["distribution"].strip(),
            "lo": _num(r["dist_lo"]),
            "hi": _num(r["dist_hi"]),
            "unit": r["unit"],
            "source_bibkey": r["source_bibkey"],
            "source_locator": r["source_locator"],
        }
    return out


def components(c_t, lambda_c_norm, lam_dt, a, ltt_norm, w_tt, lam32, w_d, k_p, c_p, imp, *, inputs=None):
    """Loss per cycle (fractions) of each side channel at the fill (c_t, lambda_c_norm) for the given inputs.

    Arrays broadcast. Raises ValueError for a < 0, where q_1s = 1/(1 + a*c_t) exceeds 1.
    """
    a = np.asarray(a, dtype=float)
    if np.any(a < 0):
        raise ValueError("a < 0 gives q_1s = 1/(1 + a*c_t) > 1")
    inp = load_inputs() if inputs is None else inputs
    lam_flip = inp["lambda_dmu_flip"]["value"]
    lam_dd_12 = inp["lambda_dd_12"]["value"]
    br_he3 = inp["br_he3"]["value"]
    c_d = 1.0 - c_t
    q = 1.0 / (1.0 + a * c_t)
    out32 = c_t * lam_dt + c_d * lam_flip + c_d * lam32
    t32 = (2.0 / 3.0) / out32
    p_flip = c_d * lam_flip / out32
    out12 = c_t * lam_dt + c_d * lam_dd_12
    t12 = (1.0 / 3.0 + (2.0 / 3.0) * p_flip) / out12
    p_dd = c_d * (lam32 * t32 + lam_dd_12 * t12)
    t_d = c_d * q * (t32 + t12)
    t_t = 1.0 / lambda_c_norm - t_d
    return {
        "tt": c_t * ltt_norm * w_tt * t_t,
        "dd": c_d * q * p_dd * br_he3 * w_d,
        "p": c_p * k_p / lambda_c_norm,
        "imp": imp,
        "T_t": t_t,
        "T_d": t_d,
        "q1s": q,
    }


def _stieltjes_gauss(x, w, n):
    """Order-n Gauss rule of the discrete measure sum_j w_j delta(x - x_j) (discretized Stieltjes)."""
    w = w / w.sum()
    alpha, beta = np.zeros(n), np.zeros(n)
    p_prev, p = np.zeros_like(x), np.ones_like(x)
    nrm_prev = 1.0
    for k in range(n):
        nrm = float((w * p * p).sum())
        alpha[k] = float((w * x * p * p).sum()) / nrm
        beta[k] = 1.0 if k == 0 else nrm / nrm_prev
        p_next = (x - alpha[k]) * p - (beta[k] if k > 0 else 0.0) * p_prev
        p_prev, p, nrm_prev = p, p_next, nrm
    jac = np.diag(alpha) + np.diag(np.sqrt(beta[1:]), 1) + np.diag(np.sqrt(beta[1:]), -1)
    nodes, vec = np.linalg.eigh(jac)
    return nodes, vec[0, :] ** 2


def truncated_normal_gauss(mu, sd, lo, order):
    """Order-``order`` Gauss rule of N(mu, sd) truncated to [lo, inf), on a midpoint discretization over
    [max(lo, mu - 12 sd), mu + 12 sd]; ``lo = -inf`` gives the untruncated rule (Gauss-Hermite)."""
    lo_ = max(lo, mu - 12.0 * sd)
    hi = mu + 12.0 * sd
    h = (hi - lo_) / _STIELTJES_POINTS
    x = lo_ + h * (np.arange(_STIELTJES_POINTS) + 0.5)
    return _stieltjes_gauss(x, np.exp(-0.5 * ((x - mu) / sd) ** 2), order)


def _normal_axis(mu, sd, order):
    x, w = np.polynomial.hermite_e.hermegauss(order)
    return mu + sd * x, w / w.sum()


def _uniform_axis(lo, hi, order):
    x, w = np.polynomial.legendre.leggauss(order)
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * x, w / w.sum()


def _anchor_setup(inp):
    ct = inp["ct_anchor"]["value"]
    lcn = inp["lambda_c_norm_anchor"]["value"]
    dens = inp["jinr_density_lhd"]["value"]
    ltt = inp["lambda_ttmu_table"]["value"] / dens
    s_ltt = inp["lambda_ttmu_table"]["unc"] / dens
    return ct, lcn, ltt, s_ltt


def _mid(row):
    return 0.5 * (row["lo"] + row["hi"])


def _analysis(inp, an, order):
    ct, lcn, ltt, s_ltt = _anchor_setup(inp)
    dt, aa = inp[f"lambda_dt_{an}"], inp[f"a_q1s_{an}"]
    cen = components(
        ct, lcn, dt["value"], aa["value"], ltt, inp["omega_tt"]["value"], _mid(inp["lambda_dd_32"]),
        inp["omega_d"]["value"], inp["k_p"]["value"], _mid(inp["c_p"]), _mid(inp["imp"]), inputs=inp,
    )
    cen = {k: float(cen[k]) for k in ("tt", "dd", "p", "imp")}
    a_nodes, a_w = truncated_normal_gauss(aa["value"], aa["unc"], 0.0, order)
    axes = [
        _normal_axis(dt["value"], dt["unc"], order),
        (a_nodes, a_w),
        _normal_axis(ltt, s_ltt, order),
        _normal_axis(inp["omega_tt"]["value"], inp["omega_tt"]["unc"], order),
        _uniform_axis(inp["lambda_dd_32"]["lo"], inp["lambda_dd_32"]["hi"], order),
        _normal_axis(inp["omega_d"]["value"], inp["omega_d"]["unc"], order),
        _normal_axis(inp["k_p"]["value"], inp["k_p"]["unc"], order),
        _uniform_axis(inp["c_p"]["lo"], inp["c_p"]["hi"], order),
        _uniform_axis(inp["imp"]["lo"], inp["imp"]["hi"], order),
    ]
    m1 = m2 = 0.0
    for (x0, w0), (x1, w1) in itertools.product(zip(*axes[0], strict=True), zip(*axes[1], strict=True)):
        grids = np.meshgrid(*[ax[0] for ax in axes[2:]], indexing="ij")
        wts = axes[2][1]
        for ax in axes[3:]:
            wts = np.multiply.outer(wts, ax[1])
        c = components(ct, lcn, x0, x1, *grids, inputs=inp)
        tot = c["tt"] + c["dd"] + c["p"] + c["imp"]
        w = w0 * w1 * wts
        m1 += float((w * tot).sum())
        m2 += float((w * tot * tot).sum())
    return {
        "central_pct": 100.0 * sum(cen.values()),
        "components_pct": {k: 100.0 * v for k, v in cen.items()},
        "mean_pct": 100.0 * m1,
        "sd_pct": 100.0 * math.sqrt(max(m2 - m1 * m1, 0.0)),
    }


@functools.lru_cache(maxsize=4)
def _anchor_l_other(order):
    inp = load_inputs()
    an = {k: _analysis(inp, k, order) for k in _ANALYSES}
    comp = {
        k: sum(an[a]["components_pct"][k] for a in _ANALYSES) / len(_ANALYSES)
        for k in ("tt", "dd", "p", "imp")
    }
    tots = [an[a]["central_pct"] for a in _ANALYSES]
    half_spread = 0.5 * (max(tots) - min(tots))
    var = sum(an[a]["sd_pct"] ** 2 for a in _ANALYSES) / len(_ANALYSES)
    intrinsic = comp["tt"] + comp["dd"]
    return {
        "central_pct": sum(tots) / len(tots),
        "sigma_pct": math.sqrt(var + half_spread**2),
        "components_pct": comp,
        "intrinsic_pct": intrinsic,
        "intrinsic_fraction": intrinsic / 100,
        "analyses": an,
    }


def anchor_l_other(order=5):
    """L_other at the anchor fill in percent: the mean over the two kinetic analyses of Table I at central
    inputs (uniform inputs at their midpoints), and its sd from order-``order`` tensor quadrature over the
    continuous inputs with a >= 0 enforced, combined over the analyses as sqrt(mean variance + (half their
    spread)^2)."""
    return copy.deepcopy(_anchor_l_other(order))
