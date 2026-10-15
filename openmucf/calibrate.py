"""openmucf.calibrate -- Bayesian calibration of the cycle parameters to experiment.

The published calibration, ``posterior_quadrature`` (deterministic quadrature) with ``model_w`` as its NUTS
cross-check, fits (omega_s0, R, lambda_c, L_other) to the three summaries Breunlich et al. (1987) print
(``OBS_W``) with the covariance of their common absolute calibration (``observation_covariance``):
    omega_s      = omega_s0 * (1 - R)                 (p.331: the sticking, side-channel losses subtracted)
    phi*lambda_c = lambda_c                           (p.332: the maximum cycle rate)
    X_mu         = 1/(W/100 + lambda_0/lambda_c)      (p.332), W = omega_s0 * (1 - R) + L_other
with L_other, the side-channel loss per cycle at the anchor fill, from ``openmucf.sidechannels``.

The legacy model (``model``, ``run_mcmc``, ``run_mcmc_full``, ``OBS``) observes the effective sticking and
the yield alone, as the measured sticking and the whole loss per cycle; the frozen FC-001 path and design.py
use it.

Finding (identifiability): the yield/sticking data constrain the PRODUCT omega_s0*(1-R) (= omega_s_eff)
and lambda_c, but NOT omega_s0 and R separately -- the posterior concentrates on the CURVE
omega_s0*(1-R) = omega_s_eff (a product pinned by the data). The linear (Pearson) correlation of that
curved ridge is a descriptive statistic and is prior-support-dependent; the constraint, not its
two-decimal correlation, is the finding. Breaking the degeneracy needs an independent microscopic
constraint on the split, i.e. the Phase-3 reactivation calculation. Run with an informative omega_s0
prior (Kamimura) to see it partially resolve.

Sampling API
------------
* ``run_mcmc`` / ``run_mcmc_full`` default to 4 sequential chains x 2000 draws (multi-chain diagnostics).
* ``summarize(samples, mcmc=...)`` optionally attaches split-R_hat / ESS / divergence diagnostics via
  ``numpyro.diagnostics`` (no new runtime dependency).
* The default prior boxes were WIDENED 2026-07-12 (a disclosed statistical correction -- no
  target involved) because the previous boxes provably railed: R ~ Uniform(0.00, 0.80) (was 0.10-0.60;
  the weak-chain 95% CI hi 0.588 sat against the old 0.60 bound) and weak omega_s0_pct ~ Uniform(0.50,
  1.20) (was 0.60-1.10; the weak-chain 95% CI lo 0.608 sat against the old 0.60 bound). The legacy
  model's lambda_c box is unchanged (0.8-1.6e8; a measured-band prior, not railing); the published model's
  is ``LAMBDA_C_PRIOR_W``. See CALIBRATION.md.
* The FC-001 registered realization is PINNED in ``openmucf.forecast.posterior_samples`` (single chain,
  the OLD R box, the registered seed/warmup/samples) -- never changed while a card is registered against
  it (the FC-001 registered-card freeze). FC-002+ use the new defaults.

Convergence caveat (Kamimura chain): the informative omega_s0 Normal is truncated to the weak prior's
support, removing the negative-omega_s0 artifact basin in which a chain could trap. The registered FC-001
realization retains its untruncated prior and the default init strategy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS

from . import _jaxcfg
from .constants import LAMBDA_0

# kept for the frozen FC-001 path and design.py; the source prints the yield as 113 +- 10
OBS = dict(omega_s_eff_obs=0.45, omega_s_eff_sd=0.05, xmu_obs=113.0, xmu_sd=12.0)

# WIDENED default prior boxes (2026-07-12; see the module docstring + CALIBRATION.md). The lambda_c box
# is unchanged. Parameters (R_prior, lambda_c_prior) are exposed so that (a) the FC-001 realization can be
# pinned to its registered OLD R box and (b) the prior-sensitivity sweep can vary each box.
WEAK_OMEGA_S0_PRIOR = ("uniform", 0.50, 1.20)
# The calibration chain bounds the Kamimura Normal by the weak prior's support.
KAMIMURA_OMEGA_S0_PRIOR = ("truncnormal", 0.857, 0.03, WEAK_OMEGA_S0_PRIOR[1], WEAK_OMEGA_S0_PRIOR[2])
R_PRIOR_DEFAULT = (0.00, 0.80)
LAMBDA_C_PRIOR_DEFAULT = (0.8e8, 1.6e8)

# multi-chain defaults (multi-chain diagnostics gated in CI). FC-001 is pinned to 1 chain in forecast.py.
NUM_CHAINS_DEFAULT = 4
CHAIN_METHOD_DEFAULT = "sequential"

# diagnostics are reported for these sites (the 3 sampled parameters + the 2 deterministics).
_DIAG_SITES = ("omega_s0_pct", "R", "lambda_c", "omega_s_eff_pct", "X_mu")


def model(
    omega_s_eff_obs=0.45,
    omega_s_eff_sd=0.05,
    xmu_obs=113.0,
    xmu_sd=12.0,
    omega_s0_prior=WEAK_OMEGA_S0_PRIOR,
    R_prior=R_PRIOR_DEFAULT,
    lambda_c_prior=LAMBDA_C_PRIOR_DEFAULT,
    obs_corr=0.0,
):
    """omega_s0_prior accepts uniform, normal, or truncated Normal tuples.

    ``R_prior`` / ``lambda_c_prior`` are (lo, hi) uniform boxes. ``obs_corr`` (default 0.0) treats the two
    observations as independent Gaussians -- published as separate summary statistics, primary covariance
    unobtainable pre-acquisition. A nonzero ``obs_corr`` switches to a MultivariateNormal
    likelihood with off-diagonal correlation rho_obs (the covariance-sensitivity cut, CALIBRATION.md); it
    is NEVER used by the default/registered chains, so the obs_corr=0.0 path is byte-identical to the
    original two-independent-Normal likelihood.
    """
    if omega_s0_prior[0] == "normal":
        omega_s0 = numpyro.sample("omega_s0_pct", dist.Normal(omega_s0_prior[1], omega_s0_prior[2]))
    elif omega_s0_prior[0] == "uniform":
        omega_s0 = numpyro.sample("omega_s0_pct", dist.Uniform(omega_s0_prior[1], omega_s0_prior[2]))
    elif omega_s0_prior[0] == "truncnormal":
        omega_s0 = numpyro.sample(
            "omega_s0_pct",
            dist.TruncatedNormal(
                omega_s0_prior[1], omega_s0_prior[2], low=omega_s0_prior[3], high=omega_s0_prior[4]
            ),
        )
    else:
        raise ValueError(f"unknown omega_s0 prior kind: {omega_s0_prior[0]}")
    R = numpyro.sample("R", dist.Uniform(R_prior[0], R_prior[1]))
    lambda_c = numpyro.sample("lambda_c", dist.Uniform(lambda_c_prior[0], lambda_c_prior[1]))

    ose_pct = omega_s0 * (1.0 - R)
    numpyro.deterministic("omega_s_eff_pct", ose_pct)
    xmu = 1.0 / (ose_pct / 100.0 + LAMBDA_0 / lambda_c)
    numpyro.deterministic("X_mu", xmu)

    if obs_corr == 0.0:
        numpyro.sample("obs_ose", dist.Normal(ose_pct, omega_s_eff_sd), obs=omega_s_eff_obs)
        numpyro.sample("obs_xmu", dist.Normal(xmu, xmu_sd), obs=xmu_obs)
    else:
        cov = obs_corr * omega_s_eff_sd * xmu_sd
        mean = jnp.stack([ose_pct, xmu])
        covariance = jnp.array(
            [[omega_s_eff_sd**2, cov], [cov, xmu_sd**2]]
        )
        numpyro.sample(
            "obs_joint",
            dist.MultivariateNormal(mean, covariance_matrix=covariance),
            obs=jnp.array([omega_s_eff_obs, xmu_obs]),
        )


def run_mcmc_full(
    num_warmup=1000,
    num_samples=2000,
    seed=0,
    omega_s0_prior=WEAK_OMEGA_S0_PRIOR,
    R_prior=R_PRIOR_DEFAULT,
    lambda_c_prior=LAMBDA_C_PRIOR_DEFAULT,
    obs_corr=0.0,
    num_chains=NUM_CHAINS_DEFAULT,
    chain_method=CHAIN_METHOD_DEFAULT,
    **obs,
):
    """Run NUTS and return ``(mcmc, samples)``. ``samples`` is the pooled (across-chain) draws dict.

    ``extra_fields=("diverging",)`` is collected for the divergence diagnostic; it does not alter the
    parameter draws (so the obs_corr=0.0 / single-chain / OLD-box path reproduces the FC-001 realization
    byte-for-byte -- verified by ``git diff --exit-code forecasts/`` and the FC-001 pin test).
    """
    _jaxcfg.require_x64("NUTS calibration")  # float32 draws would be silently, invisibly wrong
    data = {
        **OBS,
        **obs,
        "omega_s0_prior": omega_s0_prior,
        "R_prior": R_prior,
        "lambda_c_prior": lambda_c_prior,
        "obs_corr": obs_corr,
    }
    mcmc = MCMC(
        NUTS(model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",), **data)
    return mcmc, mcmc.get_samples()


def run_mcmc(*args, **kwargs):
    """Back-compatible wrapper: return only the pooled samples dict (see :func:`run_mcmc_full`)."""
    return run_mcmc_full(*args, **kwargs)[1]


def summarize(samples, mcmc=None):
    """Posterior summary. With ``mcmc`` given, also attach multi-chain convergence diagnostics.

    Adds ``"diagnostics"`` = {site: {"r_hat", "ess", "mcse"}} (split-Gelman-Rubin / effective sample size
    via ``numpyro.diagnostics``; mcse = sd / sqrt(ess)) and ``"n_divergences"`` (int). No new dependency.
    """

    def stats(a):
        a = np.asarray(a)
        return {
            "mean": float(a.mean()),
            "sd": float(a.std()),
            "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)),
        }

    out = {k: stats(samples[k]) for k in ("omega_s0_pct", "R", "lambda_c", "omega_s_eff_pct", "X_mu")}
    out["corr_omega_s0_R"] = float(
        np.corrcoef(np.asarray(samples["omega_s0_pct"]), np.asarray(samples["R"]))[0, 1]
    )
    if mcmc is not None:
        grouped = mcmc.get_samples(group_by_chain=True)
        diags = {}
        for site in _DIAG_SITES:
            x = np.asarray(grouped[site])  # shape (num_chains, num_samples)
            ess = float(effective_sample_size(x))
            sd = out[site]["sd"]
            diags[site] = {
                "r_hat": float(split_gelman_rubin(x)),
                "ess": ess,
                "mcse": (sd / np.sqrt(ess)) if ess > 0 else float("nan"),
            }
        out["diagnostics"] = diags
        out["n_divergences"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return out


# ================================================================ the published calibration (anchor basis)
# Breunlich et al., PRL 58, 329 (1987): effective sticking p.331; maximum cycle rate and yield p.332; the
# calibration errors of the cycle rates, Fig. 4 caption.
OBS_W = dict(
    omega_s_pct=0.45, omega_s_sd=0.05, phi_lambda_c=145e6, phi_lambda_c_sd=12e6, xmu=113.0, xmu_sd=10.0,
    calibration=0.08, lambda_stat=0.02,
)
LAMBDA_C_PRIOR_W = (0.8e8, 2.0e8)

QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)
QUAD_QUANTITIES = ("omega_s_eff_pct", "lambda_c", "omega_s0_pct", "R", "W_pct", "X_mu", "L_pct")
STATS = ("mean", "sd") + tuple(f"q{p}" for p in QUANTILES)
#: grid sizes at res = 1 (each scales with res); e runs over (0, E_HI) %, L_other over its mean +- L_SPAN sd
QUAD_GRID = dict(ne=8000, nl=1600, nL=120, ns=1600)
E_HI = 0.90
L_SPAN = 8.0
#: cells per block of the quadrature loop
_BLOCK_CELLS = 2**21
#: CDF bin edges (lo, hi, number of bins); None = the parameter box
QUAD_BINS = {
    "omega_s_eff_pct": (0.0, 0.9, 90000),
    "W_pct": (0.0, 1.3, 130000),
    "lambda_c": (None, None, 120000),
    "X_mu": (40.0, 260.0, 220000),
    "L_pct": (None, None, 16000),
}
#: convergence criterion (res vs 2 res), relative: non-L quantiles, L_other quantiles, means and sds
CONV_RTOL_Q, CONV_RTOL_Q_L, CONV_RTOL_MOMENT = 1e-3, 5e-3, 1e-6
#: the NUTS cross-check gate
NUTS_MAX_RHAT, NUTS_MIN_ESS, NUTS_MAX_DEV_MCSE = 1.01, 400, 3.0


def observation_covariance(kind="L3"):
    """Covariance of the residuals (omega_s [%], phi*lambda_c [s^-1], X_mu): ``L3`` the common calibration,
    ``L2`` the cycle-rate/yield correlation alone, ``L1`` independent."""
    w = OBS_W
    om, phl, x = w["omega_s_pct"], w["phi_lambda_c"], w["xmu"]
    so, sl, sx = w["omega_s_sd"], w["phi_lambda_c_sd"], w["xmu_sd"]
    k, el = w["calibration"], w["lambda_stat"]
    if kind == "L3":
        e_s = np.sqrt((sx / x) ** 2 - k**2 - el**2)
        e_o = np.sqrt(so**2 - (om * k) ** 2)
        a = np.array(
            [[-om * k, 0.0, 0.0, e_o], [phl * k, phl * el, 0.0, 0.0], [x * k, x * el, -x * e_s, 0.0]]
        )
        return a @ a.T
    if kind == "L2":
        rho = (sl / phl) / (sx / x)
        return np.array([[so**2, 0.0, 0.0], [0.0, sl**2, rho * sl * sx], [0.0, rho * sl * sx, sx**2]])
    if kind == "L1":
        return np.diag([so**2, sl**2, sx**2])
    raise ValueError(f"unknown covariance kind: {kind}")


def _npdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def _support(prior):
    kind = prior[0]
    if kind == "uniform":
        return "uniform", prior[1], prior[2], None, None
    if kind == "normal":
        return "truncnormal", prior[1] - 14.0 * prior[2], prior[1] + 14.0 * prior[2], prior[1], prior[2]
    if kind == "truncnormal":
        return "truncnormal", prior[3], prior[4], prior[1], prior[2]
    raise ValueError(f"unknown omega_s0 prior kind: {kind}")


def _g_of_e(e, prior, r_lo, r_hi):
    kind, a, b, mu, sd = _support(prior)
    if kind == "uniform":
        lo = np.maximum(a, e / (1 - r_lo))
        hi = np.minimum(b, e / (1 - r_hi))
        ok = hi > lo
        return np.where(ok, (np.log(np.where(ok, hi, 1.0)) - np.log(np.where(ok, lo, 1.0))) / (b - a), 0.0)
    out = np.zeros_like(e)
    for i, ev in enumerate(e):
        lo, hi = max(a, ev / (1 - r_lo)), min(b, ev / (1 - r_hi))
        if hi > lo:
            s = np.linspace(lo, hi, 801)
            out[i] = float(np.trapezoid(_npdf(s, mu, sd) / s, s))
    return out


class _RampCDF:

    def __init__(self, edges):
        self.edges = edges
        self.x0 = edges[0]
        self.dx = (edges[-1] - edges[0]) / (len(edges) - 1)
        self.c = np.zeros(len(edges) + 1)
        self.d = np.zeros(len(edges) + 1)

    def add(self, lo, hi, mass):
        lo = np.ravel(lo) - self.x0
        hi = np.ravel(hi) - self.x0
        c = np.ravel(mass) / (hi - lo)
        n = len(self.edges)
        for t, cc in ((lo, c), (hi, -c)):
            k = np.clip(np.floor(t / self.dx).astype(np.int64) + 1, 0, n)
            self.c += np.bincount(k, weights=cc, minlength=n + 1)
            self.d += np.bincount(k, weights=cc * t, minlength=n + 1)

    def cdf(self, total):
        x = self.edges - self.x0
        return (x * np.cumsum(self.c[:-1]) - np.cumsum(self.d[:-1])) / total


def _quantile(edges, cdf, p):
    j = int(np.clip(np.searchsorted(cdf, p, side="left"), 1, len(edges) - 1))
    f0, f1 = cdf[j - 1], cdf[j]
    if f1 <= f0:
        return float(edges[j])
    return float(edges[j - 1] + (p - f0) / (f1 - f0) * (edges[j] - edges[j - 1]))


def _summary(mean, sd, edges, cdf):
    return {"mean": float(mean), "sd": float(sd), **{f"q{p}": _quantile(edges, cdf, p) for p in QUANTILES}}


def posterior_quadrature(prior, *, R_box=(0.0, 0.8), lam_box=LAMBDA_C_PRIOR_W, cov="L3", res=1.0,
                         l_other=None, grid=None, bins=1.0):
    """The calibration posterior by deterministic quadrature on midpoint grids in (omega_s_eff, lambda_c,
    L_other) and (omega_s0, R), numpy only.

    Quantiles come from the cell-integrated CDF: each cell's mass is spread uniformly over the interval the
    cell spans in that quantity. Returns mean, sd and ``QUANTILES`` per ``QUAD_QUANTITIES`` entry,
    ``corr_omega_s0_R`` and ``X_mu_cdf`` = (edges, cdf).
    """
    from . import sidechannels

    l_mu, l_sd = sidechannels.ANCHOR_L_OTHER_PCT if l_other is None else l_other
    g = {k: int(round(v * res)) for k, v in QUAD_GRID.items()}
    g.update(grid or {})
    ne, nl, nL, ns = g["ne"], g["nl"], g["nL"], g["ns"]
    he = E_HI / ne
    e = he * (np.arange(ne) + 0.5)
    hl = (lam_box[1] - lam_box[0]) / nl
    lam = lam_box[0] + hl * (np.arange(nl) + 0.5)
    l_lo, l_hi = max(0.0, l_mu - L_SPAN * l_sd), l_mu + L_SPAN * l_sd
    hL = (l_hi - l_lo) / nL
    lg = l_lo + hL * (np.arange(nL) + 0.5)
    ci = np.linalg.inv(observation_covariance(cov))
    ge = _g_of_e(e, prior, *R_box)
    r1 = OBS_W["phi_lambda_c"] - lam
    dec = LAMBDA_0 / lam
    dec_lo = LAMBDA_0 / (lam - 0.5 * hl)
    dec_hi = LAMBDA_0 / (lam + 0.5 * hl)
    pdf_l = _npdf(lg, l_mu, l_sd)

    def bin_edges(q, lo, hi):
        blo, bhi, nb = QUAD_BINS[q]
        return np.linspace(lo if blo is None else blo, hi if bhi is None else bhi, int(round(nb * bins)) + 1)

    edges = {
        "omega_s_eff_pct": bin_edges("omega_s_eff_pct", None, None),
        "W_pct": bin_edges("W_pct", None, None),
        "lambda_c": bin_edges("lambda_c", *lam_box),
        "X_mu": bin_edges("X_mu", None, None),
        "L_pct": bin_edges("L_pct", l_lo, l_hi),
    }
    acc = {k: _RampCDF(v) for k, v in edges.items()}
    f = np.zeros(ne)
    m_e = np.zeros(ne)
    m_l = np.zeros(nl)
    m_L = np.zeros(nL)
    w1 = w2 = x1 = x2 = x_out = 0.0
    xe = edges["X_mu"]
    rows = max(1, _BLOCK_CELLS // nl)
    for b0 in range(0, ne, rows):
        eb = e[b0:b0 + rows]
        r0 = (OBS_W["omega_s_pct"] - eb)[:, None]
        base = ci[0, 0] * r0 * r0 + 2 * ci[0, 1] * r0 * r1[None, :] + ci[1, 1] * (r1 * r1)[None, :]
        lin = 2 * ci[0, 2] * r0 + 2 * ci[1, 2] * r1[None, :]
        gb = ge[b0:b0 + rows][:, None]
        for j, lv in enumerate(lg):
            x = 1.0 / ((eb[:, None] + lv) / 100.0 + dec[None, :])
            r2 = OBS_W["xmu"] - x
            lik = np.exp(-0.5 * (base + (lin + ci[2, 2] * r2) * r2))
            f[b0:b0 + rows] += lik.sum(axis=1) * (pdf_l[j] * hl * hL)
            w = lik * (gb * (pdf_l[j] * he * hl * hL))
            we = w.sum(axis=1)
            wsum = we.sum()
            m_e[b0:b0 + rows] += we
            m_l += w.sum(axis=0)
            m_L[j] += wsum
            wv = eb + lv
            w1 += (we * wv).sum()
            w2 += (we * wv * wv).sum()
            x1 += (w * x).sum()
            x2 += (w * x * x).sum()
            acc["W_pct"].add(wv - 0.5 * (he + hL), wv + 0.5 * (he + hL), we)
            x_lo = 1.0 / ((eb[:, None] + 0.5 * he + lv + 0.5 * hL) / 100.0 + dec_lo[None, :])
            x_hi = 1.0 / ((eb[:, None] - 0.5 * he + lv - 0.5 * hL) / 100.0 + dec_hi[None, :])
            acc["X_mu"].add(x_lo, x_hi, w)
            if x_lo.min() < xe[0] or x_hi.max() > xe[-1]:
                x_out += w[(x_lo < xe[0]) | (x_hi > xe[-1])].sum()
    z = m_L.sum()
    if x_out > 1e-9 * z:
        raise ValueError(f"X_mu mass outside its CDF edges: {x_out / z:.3g} of the total")
    acc["omega_s_eff_pct"].add(e - 0.5 * he, e + 0.5 * he, m_e)
    acc["lambda_c"].add(lam - 0.5 * hl, lam + 0.5 * hl, m_l)
    acc["L_pct"].add(lg - 0.5 * hL, lg + 0.5 * hL, m_L)
    mom = {
        "omega_s_eff_pct": ((m_e * e).sum(), (m_e * e * e).sum()),
        "W_pct": (w1, w2),
        "lambda_c": ((m_l * lam).sum(), (m_l * lam * lam).sum()),
        "X_mu": (x1, x2),
        "L_pct": ((m_L * lg).sum(), (m_L * lg * lg).sum()),
    }
    out = {}
    for k in edges:
        m = mom[k][0] / z
        sd = np.sqrt(max(mom[k][1] / z - m * m, 0.0))
        out[k] = _summary(m, sd, edges[k], acc[k].cdf(z))
    x_cdf = (edges["X_mu"], acc["X_mu"].cdf(z))
    kind, s_lo, s_hi, mu, sd0 = _support(prior)
    hs = (s_hi - s_lo) / ns
    s = s_lo + hs * (np.arange(ns) + 0.5)
    hr = (R_box[1] - R_box[0]) / ns
    rr = R_box[0] + hr * (np.arange(ns) + 0.5)
    ps = np.full_like(s, 1.0 / (s_hi - s_lo)) if kind == "uniform" else _npdf(s, mu, sd0)
    s2, rr2 = np.meshgrid(s, rr, indexing="ij")
    p = ps[:, None] * np.interp(s2 * (1 - rr2), e, f, left=0.0, right=0.0)
    pw = p / p.sum()
    for k, v, h0, nodes, axis in (("omega_s0_pct", s2, hs, s, 1), ("R", rr2, hr, rr, 0)):
        m = float((pw * v).sum())
        sd = float(np.sqrt((pw * (v - m) ** 2).sum()))
        cell_edges = np.concatenate([nodes - 0.5 * h0, [nodes[-1] + 0.5 * h0]])
        cdf = np.concatenate([[0.0], np.cumsum(pw.sum(axis=axis))])
        out[k] = _summary(m, sd, cell_edges, cdf)
    ms, mr = out["omega_s0_pct"]["mean"], out["R"]["mean"]
    out["corr_omega_s0_R"] = float(
        (pw * (s2 - ms) * (rr2 - mr)).sum() / (out["omega_s0_pct"]["sd"] * out["R"]["sd"])
    )
    out["X_mu_cdf"] = x_cdf
    return out


def _rel(a, b):
    d = max(abs(a), abs(b))
    return 0.0 if d == 0.0 else abs(a - b) / d


def convergence_changes(pub, fine):
    rows = []
    for q in QUAD_QUANTITIES:
        for st in STATS:
            quantile_tol = CONV_RTOL_Q_L if q == "L_pct" else CONV_RTOL_Q
            tol = CONV_RTOL_MOMENT if st in ("mean", "sd") else quantile_tol
            rows.append((f"{q}.{st}", _rel(pub[q][st], fine[q][st]), tol))
    return rows


def convergence_failures(pub, fine):
    """Every cell whose change between ``pub`` and ``fine`` (res and 2 res) exceeds its criterion."""
    return [
        f"{name}: relative change {d:.3g} > {tol:g}" for name, d, tol in convergence_changes(pub, fine)
        if not d <= tol
    ]


def nuts_gate_failures(diag, quad):
    """The NUTS cross-check gate: max r_hat < 1.01, min ess > 400, no divergence, and every mean of
    ``QUAD_QUANTITIES`` within 3 MCSE of the quadrature mean ``quad[q]["mean"]``."""
    bad = []
    if not diag["max_rhat"] < NUTS_MAX_RHAT:
        bad.append(f"max_rhat {diag['max_rhat']:.4f} >= {NUTS_MAX_RHAT}")
    if not diag["min_ess"] > NUTS_MIN_ESS:
        bad.append(f"min_ess {diag['min_ess']:.4g} <= {NUTS_MIN_ESS}")
    if diag["divergences"] != 0:
        bad.append(f"divergences {diag['divergences']} != 0")
    for q in QUAD_QUANTITIES:
        dev = abs(diag["mean"][q] - quad[q]["mean"]) / diag["mcse"][q]
        if not dev <= NUTS_MAX_DEV_MCSE:
            bad.append(f"{q} mean {dev:.2f} MCSE from the quadrature mean")
    return bad


def _sample_omega_s0(prior):
    kind = prior[0]
    if kind == "normal":
        return numpyro.sample("omega_s0_pct", dist.Normal(prior[1], prior[2]))
    if kind == "uniform":
        return numpyro.sample("omega_s0_pct", dist.Uniform(prior[1], prior[2]))
    if kind == "truncnormal":
        return numpyro.sample(
            "omega_s0_pct", dist.TruncatedNormal(prior[1], prior[2], low=prior[3], high=prior[4])
        )
    raise ValueError(f"unknown omega_s0 prior kind: {kind}")


def model_w(omega_s0_prior=WEAK_OMEGA_S0_PRIOR, R_prior=R_PRIOR_DEFAULT, lambda_c_prior=LAMBDA_C_PRIOR_W,
            l_other=None, cov="L3"):
    """The ``posterior_quadrature`` model for NUTS."""
    from . import sidechannels

    l_mu, l_sd = sidechannels.ANCHOR_L_OTHER_PCT if l_other is None else l_other
    omega_s0 = _sample_omega_s0(omega_s0_prior)
    R = numpyro.sample("R", dist.Uniform(R_prior[0], R_prior[1]))
    lambda_c = numpyro.sample("lambda_c", dist.Uniform(lambda_c_prior[0], lambda_c_prior[1]))
    l_pct = numpyro.sample("L_pct", dist.TruncatedNormal(l_mu, l_sd, low=0.0))
    e = omega_s0 * (1.0 - R)
    numpyro.deterministic("omega_s_eff_pct", e)
    w = e + l_pct
    numpyro.deterministic("W_pct", w)
    xmu = 1.0 / (w / 100.0 + LAMBDA_0 / lambda_c)
    numpyro.deterministic("X_mu", xmu)
    obs = jnp.array([OBS_W["omega_s_pct"], OBS_W["phi_lambda_c"], OBS_W["xmu"]])
    numpyro.sample(
        "obs_w",
        dist.MultivariateNormal(
            jnp.stack([e, lambda_c, xmu]), covariance_matrix=jnp.asarray(observation_covariance(cov))
        ),
        obs=obs,
    )


def run_model_w(omega_s0_prior=WEAK_OMEGA_S0_PRIOR, *, num_warmup=1000, num_samples=2000, seed=0,
                num_chains=NUM_CHAINS_DEFAULT, chain_method=CHAIN_METHOD_DEFAULT, **kw):
    """Run NUTS on ``model_w``; returns ``(mcmc, pooled samples)``."""
    _jaxcfg.require_x64("NUTS on the anchor-basis model")
    mcmc = MCMC(
        NUTS(model_w),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",), omega_s0_prior=omega_s0_prior, **kw)
    return mcmc, mcmc.get_samples()


def summarize_w(mcmc):
    """Means, MCSE and the gate's diagnostics of a ``run_model_w`` run."""
    grouped = mcmc.get_samples(group_by_chain=True)
    mean, mcse, rhat, ess = {}, {}, {}, {}
    for q in QUAD_QUANTITIES:
        x = np.asarray(grouped[q])
        ess[q] = float(effective_sample_size(x))
        rhat[q] = float(split_gelman_rubin(x))
        mean[q] = float(x.mean())
        mcse[q] = float(x.std()) / float(np.sqrt(ess[q]))
    return {
        "mean": mean,
        "mcse": mcse,
        "max_rhat": max(rhat.values()),
        "min_ess": min(ess.values()),
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
    }
