"""openmucf.design -- Bayesian experimental design (BOED) over the calibrated posterior.

Which NEXT muCF experiment would most sharpen the ``(omega_s^eff, R)`` estimand that
``openmucf.calibrate`` leaves partly degenerate? This module ranks a fixed registry of candidate
measurements by two metrics, BOTH computed over the EXISTING numpyro posterior (no new physics model,
no new likelihood on the calibration data -- a candidate is only an ADDED future observable):

* :func:`sd_contraction` -- **PRIMARY.** Preposterior median posterior-standard-deviation contraction
  for ``omega_s^eff`` (well-posed) and, class-conditionally, ``R``. For each of ``n_synth`` synthetic
  datasets drawn at the candidate's design point it refits the calibration model with the candidate's
  observable appended and measures how much the posterior sd shrinks.
* :func:`eig_nested_mc` -- **SECONDARY.** Nested Monte-Carlo Expected Information Gain (bits) of the
  candidate observable, taken over the existing posterior as the design prior.

Estimand discipline (the reason R is reported class-conditionally)
------------------------------------------------------------------
``omega_s^eff = omega_s0 * (1 - R)`` is directly observable from neutron yields, so its EIG /
contraction is well posed. ``R`` (the microscopic sticking/reactivation split) is NOT identified by
neutron-only observables without an ASSUMED structural form for how reactivation behaves away from the
calibration density. We therefore report R's contraction in BOTH structural classes and report the
class-flip as a finding:

* **constant-R** (scenario A): reactivation at the high-density design point is the SAME latent ``R``
  as at calibration, so a high-density neutron measurement maps straight back onto ``R``.
* **R(phi)-inflated** (scenario B): the design-point reactivation is a decoupled latent
  ``R_hi ~ Uniform(0.15, 0.60)`` (the ``forecast.py`` registered prior); a neutron-only measurement then
  informs only ``omega_s0`` and leaves the calibration ``R`` essentially untouched.

The X-ray/neutron-ratio candidate (C4) constrains ``omega_s0`` DIRECTLY (independent of the R(phi)
form), so its R-contraction is robust across both classes -- which is exactly why the ranking flips.

Kept OUT of ``openmucf.__all__`` (a library, like ``calibrate``/``forecast``); introduces no new runtime
dependency. All numbers are NUTS/Monte-Carlo derived: they reproduce to Monte-Carlo error, NOT
byte-identically -- see ``DESIGN.md`` and the 5%-relative-tolerance ``--audit`` in
``scripts/generate_design.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import calibrate
from .constants import LAMBDA_0

# --- pre-registered design settings (do not adjust after seeing outputs; I2) --------------------------
# Base knowledge = the WEAK-prior calibration chain: the omega_s0/R degeneracy this design exists to
# break (the +0.84 ridge of CALIBRATION.md; the same chain the X-ray feasibility study used).
BASE_PRIOR = ("uniform", 0.60, 1.10)
NUM_WARMUP = 1000        # mirrors scripts/generate_calibration.py + forecast.py chains
NUM_SAMPLES = 4000

PHI_ANCHOR = 1.2         # canonical liquid operating point (forecast.PHI_ANCHOR / validate _OP)

# R(phi)-inflated (scenario-B) registered prior on high-density reactivation (forecast.py D2).
R_INFLATED_LO, R_INFLATED_HI = 0.15, 0.60

# base-model priors (verbatim from calibrate.model), reused by every refit
_R_LO, _R_HI = 0.10, 0.60
_LAMBDA_C_LO, _LAMBDA_C_HI = 0.8e8, 1.6e8
_OBS = dict(omega_s_eff_obs=0.45, omega_s_eff_sd=0.05, xmu_obs=113.0, xmu_sd=12.0)

_LN2 = np.log(2.0)


# ============================================================================ candidate registry
@dataclass(frozen=True)
class Candidate:
    """One candidate future experiment: an observable added to the calibration likelihood.

    ``kind`` selects the observable ``mu(theta)``; ``sigma_rel`` is the stated measurement precision
    RELATIVE to the observable's magnitude (matching the X-ray feasibility convention, so precisions are
    comparable across candidates). ``class_sensitive`` is True iff the observable's mean depends on the
    reactivation AT the design point (so the constant-R vs R(phi)-inflated distinction changes its
    R-information);
    the X-ray-ratio and cycling-rate observables do not, so their R-contraction is class-independent.
    """

    id: str
    label: str
    design_point: str
    kind: str                 # "neutron_slope" | "cycling_rate" | "neutron_ose" | "xray_ratio" | "constant"
    class_sensitive: bool
    sigma_rel: float
    phi: float | None = None
    kappa_w: float = 0.0      # relative half-width of the kappa band (xray_ratio only)
    source: str = "scripts/generate_design.py"
    extra: dict = field(default_factory=dict)


def _mu(cand: Candidate, os0_pct, R_design, lambda_c, kappa=None):
    """Observable mean mu(theta) for ``cand``. ``R_design`` is the reactivation at the design point
    (= calibration R under constant-R, = R_hi under R(phi)-inflated). Vectorized over numpy arrays."""
    ose_frac = os0_pct * (1.0 - R_design) / 100.0
    if cand.kind == "neutron_slope":
        # muon "disappearance" slope at density phi: lambda_dis = lambda_0 + lambda_c(phi) * ose_frac,
        # with lambda_c(phi) = (phi / PHI_ANCHOR) * lambda_c (the analytic density scaling). [s^-1]
        return LAMBDA_0 + (cand.phi / PHI_ANCHOR) * lambda_c * ose_frac
    if cand.kind == "cycling_rate":
        return lambda_c                                   # direct cycling-rate readout [s^-1]
    if cand.kind == "neutron_ose":
        return os0_pct * (1.0 - R_design)                 # effective sticking at phi [percent]
    if cand.kind == "xray_ratio":
        return kappa * os0_pct / 100.0                    # K X-rays per fusion neutron ~ kappa * omega_s0
    if cand.kind == "constant":
        return np.full(np.shape(os0_pct), LAMBDA_0)       # exact replicate of an already-pinned constant
    raise ValueError(f"unknown candidate kind {cand.kind!r}")


# The four DECIDED candidates. Stated precisions are pre-registered design assumptions (a neutron
# disappearance/cycling fit to ~5%; the X-ray ratio at the X-ray feasibility study's pre-registered best
# cell w=0.10 / sigma_rel=0.02). They are numerical design inputs, NOT tuned to a target: the class-flip
# finding is structural (R(phi)-inflation decouples R regardless of precision) and the sigma-sweep sanity
# gate reports the monotone trend around them.
_C1 = Candidate("C1", "neutron disappearance slope", "phi=2.0 (MuFusE-like, liquid-scaled)",
                "neutron_slope", class_sensitive=True, sigma_rel=0.05, phi=2.0)
_C2 = Candidate("C2", "lambda_c(T) cycling rate", "T=800 K", "cycling_rate",
                class_sensitive=False, sigma_rel=0.05)
_C3 = Candidate("C3", "effective sticking omega_s^eff", "phi=2.4 (MuFusE point)", "neutron_ose",
                class_sensitive=True, sigma_rel=0.05, phi=2.4)
_C4 = Candidate("C4", "X-ray/neutron ratio", "kappa-band w=0.10, sigma_rel=0.02 (X-ray feasibility "
                "best cell)", "xray_ratio", class_sensitive=False, sigma_rel=0.02, kappa_w=0.10)


def registry(xray_verdict_pct: float, threshold_pct: float = 15.0) -> dict:
    """Return the candidate registry, applying the C4 conditional STRUCTURALLY.

    C4 (X-ray/neutron ratio) is included IFF the X-ray feasibility verdict ``xray_verdict_pct`` (the
    best-cell sd(R) contraction logged by the X-ray feasibility study) is >= ``threshold_pct``;
    otherwise the registry drops it. The verdict is passed in (never hard-coded) so the conditional is a
    pure function of the logged number -- exercised both ways by the tests.
    """
    reg = {c.id: c for c in (_C1, _C2, _C3)}
    if xray_verdict_pct >= threshold_pct:
        reg[_C4.id] = _C4
        dropped = None
    else:
        dropped = f"C4 dropped (X-ray verdict {xray_verdict_pct:.2f}% < {threshold_pct:.0f}%)"
    return {"candidates": reg, "c4_included": xray_verdict_pct >= threshold_pct, "dropped": dropped,
            "xray_verdict_pct": xray_verdict_pct, "threshold_pct": threshold_pct}


def _resolve(candidate) -> Candidate:
    if isinstance(candidate, Candidate):
        return candidate
    for c in (_C1, _C2, _C3, _C4):
        if c.id == candidate:
            return c
    raise KeyError(f"unknown candidate id {candidate!r} (known: C1..C4)")


def replicate_candidate() -> Candidate:
    """An exact-replicate/null measurement: it re-observes an already-pinned constant (lambda_0), so its
    likelihood is independent of theta and its EIG must be zero to Monte-Carlo noise (sanity gate 1)."""
    return Candidate("C0", "exact-replicate (already-pinned constant)", "n/a", "constant",
                     class_sensitive=False, sigma_rel=0.05)


# ================================================================================ posterior draws
def base_posterior(seed: int = 0) -> dict:
    """Draw the current knowledge = the WEAK-prior calibration posterior (the forecast.posterior_samples
    pattern, weak/degeneracy chain). Returns numpy arrays for omega_s0_pct, R, lambda_c, omega_s_eff_pct.
    """
    # R box pinned to design's own (_R_LO, _R_HI) = the box every refit uses (line ~60), so the
    # contraction sd_refit/sd_base stays coherent. calibrate WIDENED its default R box to (0.00, 0.80)
    # when the default R box was widened; without this pin the base and refit posteriors would use
    # different R priors. num_chains is left at calibrate's new default (4) -- the DESIGN.md shift is
    # absorbed by --audit's
    # 5%-rel / 3pp-abs tolerances.
    s = calibrate.run_mcmc(num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, seed=seed,
                           omega_s0_prior=BASE_PRIOR, R_prior=(_R_LO, _R_HI))
    return {
        "omega_s0_pct": np.asarray(s["omega_s0_pct"]),
        "R": np.asarray(s["R"]),
        "lambda_c": np.asarray(s["lambda_c"]),
        "omega_s_eff_pct": np.asarray(s["omega_s_eff_pct"]),
    }


def _draw_kappa(cand: Candidate, rng, n):
    if cand.kind != "xray_ratio":
        return None
    return rng.uniform(1.0 - cand.kappa_w, 1.0 + cand.kappa_w, n)


# ============================================================================ nested-MC EIG (bits)
def _eig_nats(y, mu_self, mu_pool, sigma):
    """Nested-MC EIG in NATS. ``y``/``mu_self`` are (n_outer,), ``mu_pool`` is (n_inner,).

    EIG = mean_n [ log N(y_n | mu_self_n, sigma) - log( mean_m N(y_n | mu_pool_m, sigma) ) ].
    The Gaussian normalisation cancels between the self term and every pooled term, so only the
    quadratic exponents remain (numerically stable via log-mean-exp).
    """
    inv2s2 = 0.5 / (sigma * sigma)
    self_term = -inv2s2 * (y - mu_self) ** 2                          # (n_outer,)
    d = y[:, None] - mu_pool[None, :]                                 # (n_outer, n_inner)
    q = -inv2s2 * d * d
    qmax = q.max(axis=1)
    log_marginal = qmax + np.log(np.mean(np.exp(q - qmax[:, None]), axis=1))
    return float(np.mean(self_term - log_marginal))


def eig_nested_mc(candidate, n_outer: int = 256, n_inner: int = 256, seed: int = 0,
                  cls: str = "constant", sigma_rel: float | None = None,
                  samples: dict | None = None) -> dict:
    """Nested-MC Expected Information Gain (bits) of ``candidate`` over the existing weak-prior posterior.

    The posterior is the design prior; the candidate adds ``y ~ Normal(mu(theta), sigma)`` with
    ``sigma = sigma_rel * |mean(mu)|``. Draws ``n_outer`` (theta, y) outer pairs and marginalises the
    predictive over ``n_inner`` independent posterior draws. ``cls`` selects the reactivation link for
    class-sensitive observables ('constant' or 'inflated'). The inner/outer settings are echoed in the
    output dict.

    NESTED-MC BIAS CAVEAT (mandatory): the log-mean-exp marginal log-likelihood is NEGATIVELY biased by
    Jensen (a mean-of-logs underestimates the log-of-a-mean), so the reported EIG -- which SUBTRACTS that
    marginal -- carries an O(1/n_inner) POSITIVE bias: a slight over-estimate that shrinks with n_inner.
    Rankings (the deliverable) are robust to it; absolute bits are indicative. ``n_inner`` is reported so
    the bias scale is visible.
    """
    cand = _resolve(candidate)
    if sigma_rel is None:
        sigma_rel = cand.sigma_rel
    if samples is None:
        samples = base_posterior(seed=seed)
    rng = np.random.default_rng(seed)
    n = samples["omega_s0_pct"].size
    oi = rng.integers(0, n, n_outer)
    ii = rng.integers(0, n, n_inner)

    def link(idx, k):
        os0 = samples["omega_s0_pct"][idx]
        lc = samples["lambda_c"][idx]
        if cand.class_sensitive and cls == "inflated":
            r_design = rng.uniform(R_INFLATED_LO, R_INFLATED_HI, idx.size)
        else:
            r_design = samples["R"][idx]
        return _mu(cand, os0, r_design, lc, kappa=k)

    ko = _draw_kappa(cand, rng, n_outer)
    ki = _draw_kappa(cand, rng, n_inner)
    mu_self = np.asarray(link(oi, ko), dtype=float)
    mu_pool = np.asarray(link(ii, ki), dtype=float)
    sigma = sigma_rel * abs(float(np.mean(mu_self)))
    if sigma == 0.0:
        sigma = 1.0  # degenerate scale guard (constant/null observable): magnitude is irrelevant to EIG
    y = mu_self + rng.normal(0.0, sigma, n_outer)
    eig_nats = _eig_nats(y, mu_self, mu_pool, sigma)
    return {
        "candidate": cand.id,
        "class": cls if cand.class_sensitive else "class-independent",
        "eig_bits": eig_nats / _LN2,
        "eig_nats": eig_nats,
        "n_outer": n_outer,
        "n_inner": n_inner,
        "sigma_rel": sigma_rel,
        "seed": seed,
        "bias_caveat": "nested-MC has O(1/n_inner) positive bias; bits indicative, rankings robust",
    }


# ============================================================ sd-contraction refit (PRIMARY metric)
def _refit_sd(cand: Candidate, cls: str, y_obs: float, sigma: float, seed: int):
    """Refit the calibration model with the candidate observable appended; return (sd_ose, sd_R)."""
    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    w = cand.kappa_w

    def model():
        os0 = numpyro.sample("omega_s0_pct", dist.Uniform(BASE_PRIOR[1], BASE_PRIOR[2]))
        R = numpyro.sample("R", dist.Uniform(_R_LO, _R_HI))
        lambda_c = numpyro.sample("lambda_c", dist.Uniform(_LAMBDA_C_LO, _LAMBDA_C_HI))
        ose_pct = os0 * (1.0 - R)
        numpyro.deterministic("omega_s_eff_pct", ose_pct)
        xmu = 1.0 / (ose_pct / 100.0 + LAMBDA_0 / lambda_c)
        numpyro.sample("obs_ose", dist.Normal(ose_pct, _OBS["omega_s_eff_sd"]), obs=_OBS["omega_s_eff_obs"])
        numpyro.sample("obs_xmu", dist.Normal(xmu, _OBS["xmu_sd"]), obs=_OBS["xmu_obs"])
        if cand.class_sensitive and cls == "inflated":
            r_design = numpyro.sample("R_hi", dist.Uniform(R_INFLATED_LO, R_INFLATED_HI))
        else:
            r_design = R
        if cand.kind == "xray_ratio":
            kappa = numpyro.sample("kappa", dist.Uniform(1.0 - w, 1.0 + w))
            mu = kappa * os0 / 100.0
        elif cand.kind == "neutron_slope":
            mu = LAMBDA_0 + (cand.phi / PHI_ANCHOR) * lambda_c * (os0 * (1.0 - r_design) / 100.0)
        elif cand.kind == "cycling_rate":
            mu = lambda_c
        elif cand.kind == "neutron_ose":
            mu = os0 * (1.0 - r_design)
        else:
            raise ValueError(f"kind {cand.kind!r} is not refittable")
        numpyro.sample("obs_design", dist.Normal(mu, sigma), obs=y_obs)

    mcmc = MCMC(NUTS(model), num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    s = mcmc.get_samples()
    return float(np.asarray(s["omega_s_eff_pct"]).std()), float(np.asarray(s["R"]).std())


def sd_contraction(candidate, n_synth: int = 8, seed: int = 0, samples: dict | None = None,
                   classes=("constant", "inflated")) -> dict:
    """PRIMARY metric: preposterior median posterior-sd contraction from adding ``candidate``.

    For each of ``n_synth`` synthetic datasets -- a ``theta*`` drawn from the current (weak-prior)
    posterior, an observation ``y* = mu(theta*) + noise`` placed at the candidate's design point -- refit
    the calibration model with the observable appended and record the contraction
    ``(sd_before - sd_after) / sd_before`` of the posterior sd of ``omega_s^eff`` and ``R``. Report the
    MEDIAN over the synthetic datasets.

    ``omega_s^eff`` contraction is well posed (reported under the constant-R / calibrated-model link).
    ``R`` contraction is reported for each class in ``classes``; for a class-INSENSITIVE candidate
    (X-ray ratio, cycling rate) both classes coincide by construction and only one refit is run.
    """
    cand = _resolve(candidate)
    if samples is None:
        samples = base_posterior(seed=seed)
    sd_before_ose = float(samples["omega_s_eff_pct"].std())
    sd_before_R = float(samples["R"].std())

    rng = np.random.default_rng(seed)
    n = samples["omega_s0_pct"].size
    idx = rng.integers(0, n, n_synth)
    run_classes = classes if cand.class_sensitive else ("constant",)

    per_class: dict[str, dict] = {}
    ose_contractions: list[float] = []  # from the constant-R (calibrated-model) refit -> well-posed headline
    for cls in run_classes:
        c_ose, c_R = [], []
        for j in range(n_synth):
            i = int(idx[j])
            os0, R, lc = samples["omega_s0_pct"][i], samples["R"][i], samples["lambda_c"][i]
            if cand.class_sensitive and cls == "inflated":
                r_design = rng.uniform(R_INFLATED_LO, R_INFLATED_HI)
            else:
                r_design = float(R)
            if cand.kind == "xray_ratio":
                kappa = np.array([float(rng.uniform(1.0 - cand.kappa_w, 1.0 + cand.kappa_w))])
            else:
                kappa = None
            mu_true = float(_mu(cand, np.array([os0]), np.array([r_design]), np.array([lc]), kappa=kappa)[0])
            sigma = cand.sigma_rel * abs(mu_true)
            y_star = mu_true + float(rng.normal(0.0, sigma)) if sigma > 0 else mu_true
            sd_ose, sd_R = _refit_sd(cand, cls, y_star, sigma if sigma > 0 else 1.0, seed=1000 * j + 7)
            c_ose.append((sd_before_ose - sd_ose) / sd_before_ose)
            c_R.append((sd_before_R - sd_R) / sd_before_R)
        per_class[cls] = {
            "R_contraction": float(np.median(c_R)),
            "ose_contraction": float(np.median(c_ose)),
        }
        if cls == "constant":
            ose_contractions = c_ose

    # R contraction is class-independent for insensitive candidates: mirror the single run into 'inflated'
    if not cand.class_sensitive and "inflated" in classes:
        per_class["inflated"] = dict(per_class["constant"])

    return {
        "candidate": cand.id,
        "class_sensitive": cand.class_sensitive,
        "n_synth": n_synth,
        "seed": seed,
        "sd_before": {"omega_s_eff_pct": sd_before_ose, "R": sd_before_R},
        # PRIMARY well-posed headline (constant-R / calibrated-model link):
        "ose_contraction": float(np.median(ose_contractions)),
        # class-conditional R:
        "R_contraction": {cls: per_class[cls]["R_contraction"] for cls in classes},
        "ose_contraction_by_class": {cls: per_class[cls]["ose_contraction"] for cls in per_class},
    }


# ============================================================ Sobol-consistency (sanity gate 3)
def sobol_consistency(sigma_rel: float = 0.02, n: int = 200_000, seed: int = 0) -> dict:
    """Small-noise limit: the parameter a tiny-sigma X_mu measurement informs MOST must equal the top
    Sobol driver of X_mu (over the same uq contested prior box). Uses ``openmucf.uq``'s box + observable.

    Returns per-parameter sd contraction (importance-weighted posterior over the uq box) and the top
    parameter; the test compares the top against :func:`openmucf.uq.sobol_indices`.
    """
    from . import uq

    params = [p for p in uq.PARAMS if p.name in ("omega_s0_pct", "R", "lambda_c")]
    rng = np.random.default_rng(seed)
    draws = {p.name: rng.uniform(p.low, p.high, n) for p in params}
    xmu = uq.xmu(draws["omega_s0_pct"], draws["R"], draws["lambda_c"])
    # synthetic tight observation at the box centre's X_mu; weight the prior draws by the Gaussian likelihood
    mu0 = {p.name: 0.5 * (p.low + p.high) for p in params}
    y = float(uq.xmu(mu0["omega_s0_pct"], mu0["R"], mu0["lambda_c"]))
    sigma = sigma_rel * abs(y)
    logw = -0.5 * ((xmu - y) / sigma) ** 2
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w * w))
    contraction = {}
    for p in params:
        x = draws[p.name]
        prior_sd = float(x.std())
        mean_w = float(np.sum(w * x))
        post_sd = float(np.sqrt(max(np.sum(w * (x - mean_w) ** 2), 0.0)))
        contraction[p.name] = (prior_sd - post_sd) / prior_sd
    top = max(contraction, key=contraction.get)
    return {"contraction": contraction, "top_param": top, "ess": ess, "sigma_rel": sigma_rel, "n": n}
