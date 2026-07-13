"""openmucf.uq -- uncertainty quantification, global sensitivity, breakeven falsification.

The headline-finding layer. It runs on the closed-form forward map (validated to < 1% against the
differentiable ODE in ``cycle.py``), so millions of evaluations are laptop-tractable; the ODE is
used only to *cross-check* the gradients (``cross_check_gradient``).

Deliverables
------------
* ``local_sensitivities`` -- autodiff elasticities dln(Y)/dln(theta) for X_mu and net-electrical Q.
* ``sobol_indices``       -- SALib first/total-order global sensitivity.
* ``forward_uq``          -- Monte-Carlo propagation -> credible intervals on X_mu, Q_sci, Q_net.
* ``breakeven_audit``     -- honest, uncertainty-propagated verdict on the 2026 N_mu>500 / Q>2 claims,
                             plus the "what-would-have-to-be-true" required (R, lambda_c).

Priors are UNIFORM over each input's contested range (maximally honest about ignorance). Ranges are
taken from the ledger's contested rows; every choice is documented in :data:`PARAMS`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import E_F_MEV, LAMBDA_0

# Registered-priors file (provenance type: registered prior; I3). The UQ box VALUES are frozen this
# wave; this file makes them machine-sourced instead of module literals (regression-locked by
# tests/test_uq_priors.py::test_params_from_ledger_matches_frozen_literals).
_PRIORS_CSV = Path(__file__).resolve().parent / "data" / "uq_priors.csv"


@dataclass(frozen=True)
class Param:
    name: str
    nominal: float
    low: float
    high: float
    unit: str
    note: str


def params_from_ledger(path=None) -> list[Param]:
    """Load the UQ priors from ``openmucf/data/uq_priors.csv`` (host-side read) into ``Param`` objects.

    ``Param.note`` is populated from the CSV ``rationale`` column. The file order defines the parameter
    order used by the Sobol/forward-UQ boxes, so it must match the historical literal order.
    """
    src = Path(path) if path is not None else _PRIORS_CSV
    params: list[Param] = []
    with open(src, newline="") as f:
        for row in csv.DictReader(f):
            params.append(
                Param(
                    name=row["name"].strip(),
                    nominal=float(row["nominal"]),
                    low=float(row["low"]),
                    high=float(row["high"]),
                    unit=row["unit"].strip(),
                    note=row["rationale"].strip(),
                )
            )
    return params


PARAMS = params_from_ledger()
NAMES = [p.name for p in PARAMS]
NOMINAL = {p.name: p.nominal for p in PARAMS}


# ----------------------------------------------------------------------------- forward maps (numpy)
def xmu(omega_s0_pct, R, lambda_c):
    """X_mu = 1 / (omega_s0*(1-R) + lambda_0/lambda_c). Vectorized over numpy arrays."""
    ose = (omega_s0_pct / 100.0) * (1.0 - R)
    return 1.0 / (ose + LAMBDA_0 / lambda_c)


def q_sci(omega_s0_pct, R, lambda_c, E_mu_GeV, **_):
    return xmu(omega_s0_pct, R, lambda_c) * E_F_MEV / (E_mu_GeV * 1.0e3)


def q_net(omega_s0_pct, R, lambda_c, E_mu_GeV, eta_acc, eta_thermal, blanket_M=1.0):
    x = xmu(omega_s0_pct, R, lambda_c)
    return x * E_F_MEV * eta_thermal * blanket_M * eta_acc / (E_mu_GeV * 1.0e3)


# ------------------------------------------------------------------------- local sensitivities (jax)
def local_sensitivities():
    """Autodiff elasticities dln(Y)/dln(theta) at the nominal point, for X_mu and Q_net."""
    import jax
    import jax.numpy as jnp

    t0 = jnp.array([NOMINAL[n] for n in NAMES])

    def _xmu(t):
        os0, R, lc = t[0], t[1], t[2]
        return 1.0 / ((os0 / 100.0) * (1.0 - R) + LAMBDA_0 / lc)

    def _qnet(t):
        os0, R, lc, Emu, eacc, eth = t
        x = 1.0 / ((os0 / 100.0) * (1.0 - R) + LAMBDA_0 / lc)
        return x * E_F_MEV * eth * eacc / (Emu * 1.0e3)

    def elasticities(f):
        y = f(t0)
        g = jax.grad(f)(t0)
        return {NAMES[i]: float(g[i] * t0[i] / y) for i in range(len(NAMES))}

    return {"X_mu": elasticities(_xmu), "Q_net": elasticities(_qnet)}


def cross_check_gradient(ose=0.005, lambda_c=1.30e8, tol=0.03):
    """Gradient of X_mu w.r.t. effective sticking: analytic vs autodiff-through-the-ODE. Returns dict."""
    import jax
    import jax.numpy as jnp

    from .cycle import fusions_per_muon_ode

    def xmu_ode(o):
        y0 = jnp.array([0.0, 0.75, 0.25, 0.0, 0.0, 0.0])
        return fusions_per_muon_ode(LAMBDA_0, 0.0, 1e9, lambda_c, lambda_c, o, y0=y0)

    def xmu_an(o):
        return 1.0 / (o + LAMBDA_0 / lambda_c)

    g_ode = float(jax.grad(xmu_ode)(ose))
    g_an = float(jax.grad(xmu_an)(ose))
    rel = abs(g_ode - g_an) / abs(g_an)
    return {"grad_ode": g_ode, "grad_analytic": g_an, "rel_diff": rel, "agree": rel < tol}


# ------------------------------------------------------------------------------ global Sobol (SALib)
def _salib_api():
    from SALib.analyze.sobol import analyze
    from SALib.sample.sobol import sample

    return sample, analyze


def sobol_indices(N=4096, output="X_mu", seed=0, bounds=None, num_resamples=200):
    """First (S1) and total (ST) order Sobol indices for 'X_mu' or 'Q_net', WITH bootstrap CIs.

    Seeded (sample seed AND bootstrap seed) for reproducibility, so both the point estimates and the
    ``S1_conf`` / ``ST_conf`` 95% bootstrap half-widths are byte-stable across environments. ``bounds``
    overrides the per-input sampling box (default: the contested-range box from ``PARAMS``); pass a custom
    box to probe how the ranking depends on prior width (see :func:`sobol_robustness`). Returns a dict with
    keys ``S1``, ``ST``, ``S1_conf``, ``ST_conf`` (each an input-name -> value mapping).
    """
    sample, analyze = _salib_api()
    if bounds is None:
        bounds = [[p.low, p.high] for p in PARAMS]
    problem = {"num_vars": len(NAMES), "names": NAMES, "bounds": bounds}
    X = sample(problem, N, calc_second_order=False, seed=seed)
    if output == "X_mu":
        Y = xmu(X[:, 0], X[:, 1], X[:, 2])
    else:
        Y = q_net(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5])
    Si = analyze(
        problem, Y, calc_second_order=False, print_to_console=False,
        num_resamples=num_resamples, seed=seed,
    )
    return {k: dict(zip(NAMES, Si[k], strict=False)) for k in ("S1", "ST", "S1_conf", "ST_conf")}


def sobol_robustness(N=8192, output="X_mu", rel=0.15, seed=0):
    """Total-order Sobol S_T for ``output`` under two prior boxes, to expose prior-WIDTH dependence:

    (a) the contested-range box (``PARAMS.low/high``) -- where a relatively wide *measured* range (e.g. R
        at +/-~36% of nominal vs omega_s0 at +/-~9%) can dominate the variance; and
    (b) an equal-relative box (each input nominal * (1 +/- ``rel``)) -- where the ranking instead follows
        the local elasticities. The ranking generally REORDERS between the two, so "R drives the variance"
        is a prior-conditional statement, not a bare physics fact. Returns
        ``{'rel', 'contested_box', 'equal_relative_box'}`` (each value an ST dict keyed by input name).
    """
    contested = sobol_indices(N=N, output=output, seed=seed)["ST"]
    eqrel_bounds = [[p.nominal * (1.0 - rel), p.nominal * (1.0 + rel)] for p in PARAMS]
    equal_relative = sobol_indices(N=N, output=output, seed=seed, bounds=eqrel_bounds)["ST"]
    return {"rel": rel, "contested_box": contested, "equal_relative_box": equal_relative}


# ------------------------------------------------------------------------------- forward UQ (numpy)
def _draw(n, seed):
    rng = np.random.default_rng(seed)
    return {p.name: rng.uniform(p.low, p.high, n) for p in PARAMS}


def _ci(a):
    return {
        "lo": float(np.percentile(a, 2.5)),
        "med": float(np.median(a)),
        "hi": float(np.percentile(a, 97.5)),
    }


def forward_uq(n=400_000, seed=0, blanket_M=1.0):
    d = _draw(n, seed)
    x = xmu(d["omega_s0_pct"], d["R"], d["lambda_c"])
    qs = q_sci(d["omega_s0_pct"], d["R"], d["lambda_c"], d["E_mu_GeV"])
    qn = q_net(
        d["omega_s0_pct"], d["R"], d["lambda_c"], d["E_mu_GeV"], d["eta_acc"], d["eta_thermal"], blanket_M
    )
    return {
        "X_mu": _ci(x),
        "Q_sci": _ci(qs),
        "Q_net": _ci(qn),
        "P_Qsci_gt1": float((qs > 1).mean()),
        "P_Qnet_gt1": float((qn > 1).mean()),
        "samples": {"X_mu": x, "Q_net": qn},
    }


# ------------------------------------------------------------------- breakeven falsification (numpy)
def breakeven_audit(n=400_000, seed=1):
    """Honest, uncertainty-propagated verdict on the 2026 N_mu>500 / Q>2 projections.

    Under the MEASURED uncertainty ranges, report P(X_mu>500), P(Q_sci>2), P(Q_net>1); then the
    'what-would-have-to-be-true' required (R, lambda_c) to reach X_mu=500, versus what is measured.
    """
    d = _draw(n, seed)
    x = xmu(d["omega_s0_pct"], d["R"], d["lambda_c"])
    qs = q_sci(d["omega_s0_pct"], d["R"], d["lambda_c"], d["E_mu_GeV"])
    qn = q_net(d["omega_s0_pct"], d["R"], d["lambda_c"], d["E_mu_GeV"], d["eta_acc"], d["eta_thermal"])

    # What-would-have-to-be-true for X_mu = 500 (target).
    target = 500.0
    os0 = NOMINAL["omega_s0_pct"]  # % -- single-sourced from the registered priors (= 0.857 nominal)
    # (a) absolute cap set by decay alone (omega_s_eff -> 0): X_mu_max(lambda_c) = lambda_c/lambda_0
    lc_needed_decay_only = LAMBDA_0 * target  # lambda_c s.t. lambda_c/lambda_0 = 500 (sticking=0)
    # (b) with best measured lambda_c = 1.45e8, the maximum achievable X_mu even at zero sticking:
    xmu_cap_at_measured_lc = 1.45e8 / LAMBDA_0
    # (c) required reactivation R at an optimistic lambda_c = 3e8 to hit 500:
    lc_opt = 3.0e8
    need_ose = 1.0 / target - LAMBDA_0 / lc_opt  # required omega_s_eff
    R_req = 1.0 - need_ose / (os0 / 100.0) if need_ose > 0 else float("nan")
    # (d) even at infinite lambda_c: omega_s_eff <= 1/target, i.e. the R floor no density can beat.
    #     Computed (not transcribed) so the shipped "R >= 0.77" can never silently drift from the
    #     omega_s0 nominal it derives from (cross-vendor review hardening, 2026-07-08).
    R_req_inf_lc = 1.0 - (1.0 / target) / (os0 / 100.0)
    # (e) uq-6: the same infinite-lambda_c requirement over the FROZEN omega_s0 box edges + nominal, so the
    #     "R >= 0.77" point value carries its box-band (higher initial sticking needs MORE reactivation).
    os0_lo = min(NOMINAL["omega_s0_pct"], PARAMS[0].low)
    os0_hi = max(NOMINAL["omega_s0_pct"], PARAMS[0].high)
    R_req_inf_band = {
        f"{o:.3f}": float(1.0 - (1.0 / target) / (o / 100.0))
        for o in (PARAMS[0].low, os0, PARAMS[0].high)
    }

    return {
        "measured_ranges": {p.name: [p.low, p.high] for p in PARAMS},
        "P_xmu_gt500": float((x > target).mean()),
        "P_qsci_gt2": float((qs > 2).mean()),
        "P_qnet_gt1": float((qn > 1).mean()),
        "xmu_cap_at_measured_lambda_c": float(xmu_cap_at_measured_lc),
        "lambda_c_needed_for_500_zero_sticking": float(lc_needed_decay_only),
        "R_required_at_lambda_c_3e8": float(R_req),
        "R_required_at_infinite_lambda_c": float(R_req_inf_lc),
        "R_required_at_infinite_lambda_c_band": R_req_inf_band,
        # R_required rises with initial sticking, so the band lo is at the omega_s0-box LOW edge:
        "R_required_band_lo": float(1.0 - (1.0 / target) / (os0_lo / 100.0)),
        "R_required_band_hi": float(1.0 - (1.0 / target) / (os0_hi / 100.0)),
        "note": (
            "Scope: liquid-scale density (phi <= ~1.45), where lambda_c <= 1.45e8 is the measured max. "
            "To reach X_mu=500 there you need BOTH lambda_c >~ 2.3-3e8 AND reactivation R >~ 0.9 "
            "(vs the model-derived collisional R ~0.35, Kou-Chen Eq.33 -- experiment constrains only the "
            "product omega_s_eff ~0.45%, not R itself; our Kamimura-prior posterior gives R = 0.46 +- 0.06). "
            "Density scaling (lambda_c = phi*lambda_c_tilde) could supply the lambda_c factor alone at "
            "the demonstrated DAC phi=2.4 -- but even at infinite lambda_c, X_mu=500 needs omega_s_eff "
            "<= 0.2%, i.e. R >= 0.77. The binding requirement is reactivation, at any density."
        ),
    }


def qnet_tier_panel(emu_lo, emu_hi, n=400_000, seed=0, blanket_M=1.0):
    """Forward-UQ Q_net under a REPLACED E_mu prior box -- the muon-cost tier panel (WS-E, deviation E1).

    Draws every uq :data:`PARAMS` input from its default box, then OVERRIDES ``E_mu_GeV`` with
    ``Uniform(emu_lo, emu_hi)``. This probes the Q_net distribution under a muon-cost *tier's* E_mu prior
    WITHOUT changing the default box: :func:`forward_uq`, :func:`sobol_indices`, :func:`breakeven_audit`
    and :data:`PARAMS` are all untouched (the flat [2, 10] GeV default stands; the tier panel is an
    additional FINDINGS section, not a replacement -- deviation E1). Seeded and deterministic; the seeded
    numpy-MC median is byte-stable cross-environment (same class as :func:`forward_uq`'s shipped medians).

    Returns ``{"P_gt1": P(Q_net > 1), "median": median Q_net}`` under the tier's E_mu prior.
    """
    rng = np.random.default_rng(seed)
    d = {p.name: rng.uniform(p.low, p.high, n) for p in PARAMS}
    d["E_mu_GeV"] = rng.uniform(emu_lo, emu_hi, n)
    qn = q_net(
        d["omega_s0_pct"], d["R"], d["lambda_c"], d["E_mu_GeV"], d["eta_acc"], d["eta_thermal"], blanket_M
    )
    return {"P_gt1": float((qn > 1).mean()), "median": float(np.median(qn))}
