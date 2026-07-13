"""openmucf.forecast -- pre-registered, hash-stamped probabilistic forecast cards (FC-001).

FC-001 is a pushforward of the EXISTING calibrated posterior (``openmucf.calibrate``, Kamimura-informative
chain) through the EXISTING analytic map (``openmucf.analytic``) -- NO new physics. It forecasts the
effective sticking ``omega_s_eff`` [percent] and the cycling rate ``lambda_c`` [s^-1] at stated high-density
conditions phi in {1.2, 2.0, 2.4}, under two pre-registered scenarios:

* **A (constant-R, calibrated-model forecast):** the joint posterior (omega_s0, R) is held fixed and
  lambda_c = phi * lambda_c_tilde with lambda_c_tilde = lambda_c / 1.2 (the phi=1.2 liquid anchor).
* **B (honest ignorance bound):** R is replaced by a registered prior Uniform(0.15, 0.60) (keeping the
  posterior omega_s0 marginal), and lambda_c above the phi<=1.45 model-validity edge becomes a STRUCTURAL
  BRACKET (floor = saturation at 1.45*lambda_c_tilde; ceiling = full phi-linearity).

The card has three sections: ``payload`` (hashed, immutable), ``generation`` (unhashed env metadata),
``registration`` (unhashed, mutable; ships in ``draft`` status). See ``forecasts/FORECAST_PROTOCOL.md``
for the pre-registration, basis-conversion rules, and scoring conventions.

This module is a library (kept OUT of ``openmucf.__all__``, like ``calibrate``/``validate``); it introduces
no new runtime dependency. Reactivation transport lineage: Stodden 1990 / Rafelski-Muller 1988-89.
"""

from __future__ import annotations

import hashlib
import json
import re
from importlib import metadata as _md
from pathlib import Path

import numpy as np

from . import analytic, calibrate
from .rates import RATES_CSV

# --- Pre-registered constants (do not adjust after seeing outputs; see FORECAST_PROTOCOL.md) ----------
# Chain settings mirror scripts/generate_calibration.py:19 (the Kamimura-informative chain).
# calibrate.run_mcmc DEFAULTS differ, so every argument is passed explicitly below.
NUM_WARMUP = 1000
NUM_SAMPLES = 4000
SEED = 0
OMEGA_S0_PRIOR = ("normal", 0.857, 0.03)  # Kamimura 0.857 +- 0.03 %
# PINNED to the FC-001 registered realization -- never change while a card is registered against it
# (the FC-001 registered-card freeze). calibrate widened its DEFAULT R box to (0.00, 0.80) and now
# defaults to 4 chains; the
# registered card was computed with a SINGLE chain and the OLD R box Uniform(0.10, 0.60), so both are
# pinned here explicitly. FC-002+ use the new calibrate defaults.
R_PRIOR = (0.10, 0.60)     # registered-realization R box (calibrate's OLD default)
NUM_CHAINS = 1             # registered-realization single chain

PHI_GRID = (1.2, 2.0, 2.4)      # D7 scoring grid
PHI_ANCHOR = 1.2                # D1 canonical liquid operating point (validate.py _OP)
VALIDITY_EDGE = 1.45            # D3 model-validity edge (rates.csv R_col validity "liquid; phi<=1.45")
R_LO, R_HI = 0.15, 0.60         # D2 registered prior (assumption)
SCENARIO_B_SEED = 0             # numpy RNG seed for the Scenario-B R draws (recorded in the card)
OBSERVABLES = ("omega_s_eff", "lambda_c")
SIGFIG = 4                      # predictions rounded to 4 significant figures before serialization

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "forecasts" / "forecast.schema.json"
CARD_PATH = Path(__file__).resolve().parent.parent / "forecasts" / "FC-001-mufuse.json"


# --------------------------------------------------------------------------- deterministic serialization
def canonical_json(obj) -> str:
    """Canonical JSON used for hashing: sorted keys, tight separators, ASCII."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_sha256(card: dict) -> str:
    """SHA-256 hex digest over ``canonical_json(card['payload'])`` (registration/generation edits cannot
    change it)."""
    return hashlib.sha256(canonical_json(card["payload"]).encode("utf-8")).hexdigest()


def ledger_sha256(path: Path | str | None = None) -> str:
    """SHA-256 over the LF-normalized UTF-8 text of the rate ledger (immune to CRLF checkouts)."""
    p = Path(path) if path is not None else RATES_CSV
    text = p.read_bytes().decode("utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _round_sig(x: float, sig: int = SIGFIG) -> float:
    """Round to ``sig`` significant figures (deterministic; non-finite/zero pass through)."""
    x = float(x)
    if x == 0.0 or not np.isfinite(x):
        return x
    from math import floor, log10

    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


# ----------------------------------------------------------------------------- posterior + pushforward
def posterior_samples() -> dict:
    """Draw the calibrated posterior (Kamimura-informative chain, D6 settings) and derive lambda_c_tilde.

    Returns numpy arrays for omega_s0_pct, R, lambda_c (actual liquid rate), omega_s_eff_pct, and
    lambda_tilde_c = lambda_c / PHI_ANCHOR.
    """
    s = calibrate.run_mcmc(
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        seed=SEED,
        omega_s0_prior=OMEGA_S0_PRIOR,
        R_prior=R_PRIOR,      # pinned OLD box (the FC-001 freeze; calibrate default now widened)
        num_chains=NUM_CHAINS,  # pinned single chain (the FC-001 freeze; calibrate default now 4)
    )
    lam_c = np.asarray(s["lambda_c"])
    return {
        "omega_s0_pct": np.asarray(s["omega_s0_pct"]),
        "R": np.asarray(s["R"]),
        "lambda_c": lam_c,
        "omega_s_eff_pct": np.asarray(s["omega_s_eff_pct"]),
        "lambda_tilde_c": lam_c / PHI_ANCHOR,
    }


def _predictive_arrays(samples: dict, phi: float, scenario: str) -> dict:
    """Raw predictive sample arrays for one (phi, scenario). lambda_c may be a bracket dict {floor, ceiling}.

    Single source of truth for both card summarization (``pushforward``) and scoring (``score_card``).
    """
    os0 = samples["omega_s0_pct"]
    ose = samples["omega_s_eff_pct"]
    lam_tilde = samples["lambda_tilde_c"]
    # lambda_c(phi) goes through the EXISTING analytic map (analytic.cycling_rate = phi * lambda_c_tilde).
    if scenario == "A":
        return {"omega_s_eff": ose, "lambda_c": analytic.cycling_rate(phi, lam_tilde)}
    if scenario == "B":
        rng = np.random.default_rng(SCENARIO_B_SEED)
        r_b = rng.uniform(R_LO, R_HI, size=os0.size)
        ose_b = os0 * (1.0 - r_b)  # keep the posterior omega_s0 marginal; replace R (breaks the ridge)
        if phi <= VALIDITY_EDGE:
            return {"omega_s_eff": ose_b, "lambda_c": analytic.cycling_rate(phi, lam_tilde)}
        return {
            "omega_s_eff": ose_b,
            "lambda_c": {
                "floor": analytic.cycling_rate(VALIDITY_EDGE, lam_tilde),  # saturation at the validity edge
                "ceiling": analytic.cycling_rate(phi, lam_tilde),  # full phi-linearity
            },
        }
    raise ValueError(f"unknown scenario {scenario!r} (expected 'A' or 'B')")


def _ensemble_stats(a) -> dict:
    a = np.asarray(a)
    return {
        "median": _round_sig(np.median(a)),
        "ci68": [_round_sig(np.percentile(a, 16)), _round_sig(np.percentile(a, 84))],
        "ci95": [_round_sig(np.percentile(a, 2.5)), _round_sig(np.percentile(a, 97.5))],
        "n_samples": int(a.size),
    }


def _ensemble_prediction(a, unit: str) -> dict:
    return {"prediction_type": "ensemble", "unit": unit, **_ensemble_stats(a)}


def _bracket_prediction(floor_a, ceil_a, unit: str) -> dict:
    floor = _ensemble_stats(floor_a)
    ceil = _ensemble_stats(ceil_a)
    return {
        "prediction_type": "bracket",
        "unit": unit,
        "median_range": [floor["median"], ceil["median"]],
        "ci68": [min(floor["ci68"][0], ceil["ci68"][0]), max(floor["ci68"][1], ceil["ci68"][1])],
        "ci95": [min(floor["ci95"][0], ceil["ci95"][0]), max(floor["ci95"][1], ceil["ci95"][1])],
        "n_samples_per_limb": floor["n_samples"],
        "limbs": {"floor": floor, "ceiling": ceil},
    }


def pushforward(samples: dict, phi: float, scenario: str) -> dict:
    """Summarize the predictive arrays for one (phi, scenario) into typed card predictions.

    Returns ``{'omega_s_eff': <prediction>, 'lambda_c': <prediction>}`` where each prediction is an
    ensemble ({median, ci68, ci95, n_samples}) or a bracket ({median_range, ci68, ci95 envelope unions,
    n_samples_per_limb, limbs}).
    """
    arr = _predictive_arrays(samples, phi, scenario)
    ose_pred = _ensemble_prediction(arr["omega_s_eff"], "percent")
    lc = arr["lambda_c"]
    if isinstance(lc, dict):
        lam_pred = _bracket_prediction(lc["floor"], lc["ceiling"], "s^-1")
    else:
        lam_pred = _ensemble_prediction(lc, "s^-1")
    return {"omega_s_eff": ose_pred, "lambda_c": lam_pred}


# ----------------------------------------------------------------------------------- card construction
def _target_id(observable: str, phi: float) -> str:
    return f"{observable}@phi={phi}"


_UNIT = {"omega_s_eff": "percent", "lambda_c": "s^-1"}

_TARGET_SOURCE = {
    "omega_s_eff": (
        "Measured liquid effective sticking omega_s_eff = (0.45 +- 0.05)% "
        "(Breunlich-Kammel-Cohen-Leon, Annu. Rev. Nucl. Part. Sci. 39, 311 (1989); "
        "validation_targets.csv V_petitjean_omega); high-density behaviour is the MuFusE unknown "
        "(arXiv:2606.05333). Scored against Acceleron's published effective sticking at the stated (phi,T)."
    ),
    "lambda_c": (
        "Measured maximum liquid cycling rate lambda_c up to 1.45e8 s^-1 (Breunlich 1989; "
        "validation_targets.csv V_breunlich_lambdac); density scaling lambda_c = phi * lambda_c_tilde "
        "(openmucf.analytic.cycling_rate). Scored against Acceleron's published/normalized cycling rate "
        "at the stated (phi,T)."
    ),
}

_STATED_CONDITIONS_NOTE = (
    "non-simultaneous envelope; forecast is conditional on the (phi,T) Acceleron actually reports "
    "(933 MPa @ 100 K ~ 2.4 phi; 385 MPa @ 300 K ~ 1.14 phi -- peak phi and T are not simultaneous). "
    "The v1 map has no omega_s_eff T-resolution and only weak lambda_c(T), so targets are scored on phi; "
    "see FORECAST_PROTOCOL.md for the T caveat."
)

_R_MOTIVATION = (
    "Registered prior (assumption), NOT a measured spread. [0.15, 0.60] is a slightly floor-trimmed "
    "version of the calibration model's OWN R prior Uniform(0.10, 0.60) (openmucf/calibrate.py); its "
    "interior covers the Kou-Chen model reactivation R = 0.35 +- 0.05 (rates.csv R_col; arXiv:2606.07077 "
    "Eq.33) and the documented liquid-density d-t reactivation spread ~0.25-0.35 (LITERATURE.md; "
    "Breunlich 1989; Cohen; Markushin; Stodden 1990). The upper edge deliberately covers the "
    "enhanced-reactivation direction implied by the reported high-density sticking anomaly (measured "
    "omega_s at high density typically 10-50% below standard theory; arXiv:2606.05333). The "
    "Kamimura-prior posterior places ~25% of its mass at R > 0.5, so truncating the interval at 0.5 "
    "would be dishonest."
)

_SCENARIO_META = {
    "A": {
        "label": "constant-R (calibrated-model forecast)",
        "kind": "calibrated-model forecast",
        "assumptions": [
            "R held constant beyond its stated liquid validity range (rates.csv: 'liquid; phi<=1.45').",
            "lambda_c assumed phi-linear beyond measured liquid support (phi ~ 1.2).",
            "Joint posterior (omega_s0, R) from the Kamimura-informative chain is held fixed.",
        ],
    },
    "B": {
        "label": "honest ignorance (R = Uniform(0.15, 0.60); lambda_c bracketed above phi=1.45)",
        "kind": "ignorance bound",
        "assumptions": [
            "R replaced by the registered prior Uniform(0.15, 0.60), keeping the posterior omega_s0 "
            "marginal -- this deliberately breaks the calibration's omega_s0-R correlation, so B is a "
            "sensitivity cut, not a coherent posterior.",
            "B's central value is a midpoint-of-ignorance artifact, not a central estimate; at phi=1.2 "
            "its omega_s_eff center sits ABOVE Scenario A's (uniform-R midpoint 0.375 < posterior R mean "
            "~0.46).",
            "lambda_c above the phi<=1.45 model-validity edge is a STRUCTURAL BRACKET (floor = saturation "
            "at 1.45*lambda_c_tilde; ceiling = full phi-linearity), not a distribution.",
            "B covers nothing above R = 0.60; outcomes there resolve outside both scenarios.",
        ],
    },
}


def _targets() -> list:
    out = []
    for phi in PHI_GRID:
        for obs in OBSERVABLES:
            out.append(
                {
                    "target_id": _target_id(obs, phi),
                    "observable": obs,
                    "unit": _UNIT[obs],
                    "phi": phi,
                    "stated_conditions": {
                        "phi": phi,
                        "T_K": [100, 150, 300],
                        "note": _STATED_CONDITIONS_NOTE,
                    },
                    "source": _TARGET_SOURCE[obs],
                    "resolution": (
                        f"Resolved by mapping Acceleron's published {obs} at the reported run-averaged phi "
                        "to this target per FORECAST_PROTOCOL.md basis-conversion rules; "
                        "'not scoreable under pre-fixed conversion rules' is a legitimate outcome."
                    ),
                }
            )
    return out


def _scenarios(samples: dict) -> list:
    out = []
    for name in ("A", "B"):
        preds = []
        for phi in PHI_GRID:
            pf = pushforward(samples, phi, name)
            for obs in OBSERVABLES:
                pred = {"target_id": _target_id(obs, phi), **pf[obs]}
                preds.append(pred)
        meta = _SCENARIO_META[name]
        out.append(
            {
                "name": name,
                "label": meta["label"],
                "kind": meta["kind"],
                "assumptions": meta["assumptions"],
                "predictions": preds,
            }
        )
    return out


def _env() -> dict:
    import platform as _platform
    import sys

    import jax

    def _ver(pkg: str) -> str:
        try:
            return _md.version(pkg)
        except _md.PackageNotFoundError:  # pragma: no cover - all present in the supported env
            return "unknown"

    return {
        "python": sys.version.split()[0],
        "jax": _ver("jax"),
        "jaxlib": _ver("jaxlib"),
        "numpyro": _ver("numpyro"),
        "numpy": _ver("numpy"),
        "platform": sys.platform,
        "machine": _platform.machine(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
    }


def _scoring_rules() -> dict:
    return {
        "interval_score": (
            "HEADLINE proper score (both prediction types): Winkler interval score "
            "IS_alpha = (hi - lo) + (2/alpha)*(lo - y)*[y<lo] + (2/alpha)*(y - hi)*[y>hi], reported at "
            "interval_score_68 (alpha=0.32) and interval_score_95 (alpha=0.05); lower is better. One "
            "number scores a bracket, so Scenario A (ensemble) and Scenario B (bracket) are directly "
            "comparable on a single strictly-proper score. See FORECAST_PROTOCOL.md sec.5. "
            "Applies at RESOLUTION-time scoring (scoring code); the frozen FC-001 card is untouched."
        ),
        "ensemble": {
            "metrics": ["interval_score", "CRPS", "interval_coverage"],
            "crps_formula": "CRPS = mean|X - y| - (1/(2 n^2)) sum_{i,j} |x_i - x_j|  (empirical-CDF form)",
            "intervals": ["ci68", "ci95"],
        },
        "bracket": {
            "metrics": ["interval_score", "interval_coverage", "per_limb_CRPS"],
            "interval_score_headline": (
                "the Winkler interval score on the envelope-union ci68/ci95 is the bracket HEADLINE metric"
            ),
            "per_limb_crps": (
                "reported as a [best, worst] pair over the two structural limbs (diagnostic); no headline"
            ),
            "note": (
                "coverage uses the envelope-union ci68/ci95 of the floor (saturation at the phi<=1.45 "
                "validity edge) and ceiling (full phi-linearity) limbs"
            ),
        },
        "estimator_conventions": {
            "median": "numpy.median",
            "ci68": "[16th, 84th] equal-tailed percentiles (numpy.percentile, 'linear')",
            "ci95": "[2.5th, 97.5th] equal-tailed percentiles (numpy.percentile, 'linear')",
        },
        "both_scenarios_reported": True,
        "no_post_hoc_selection": (
            "Both scenarios are always scored and reported together; A = calibrated-model forecast, "
            "B = ignorance bound. No post-hoc selection or reweighting."
        ),
        "coverage_caveat": (
            "Scoring targets share the same posterior draws, so coverage counts are descriptive, not "
            "independent trials."
        ),
    }


def _provenance() -> dict:
    return {
        "ledger_sha256": ledger_sha256(),
        "posterior_spec": {
            "model": "openmucf.calibrate.model",
            "omega_s0_prior": list(OMEGA_S0_PRIOR),
            "priors_note": (
                "As coded in openmucf/calibrate.py: R ~ Uniform(0.10, 0.60); "
                "lambda_c ~ Uniform(0.8e8, 1.6e8) (actual liquid cycling rate). Informative omega_s0 prior "
                "= Kamimura Normal(0.857, 0.03) %."
            ),
            "seed": SEED,
            "num_warmup": NUM_WARMUP,
            "num_samples": NUM_SAMPLES,
        },
        "scenario_b_sampling": {
            "distribution": "uniform",
            "bounds": [R_LO, R_HI],
            "generator": "numpy.random.default_rng",
            "seed": SCENARIO_B_SEED,
            "n": NUM_SAMPLES,
        },
        "r_registered_prior": {
            "value": [R_LO, R_HI],
            "kind": "registered prior (assumption)",
            "motivation": _R_MOTIVATION,
        },
        "code_paths": [
            "openmucf/forecast.py",
            "openmucf/calibrate.py",
            "openmucf/analytic.py",
            "openmucf/data/rates.csv",
        ],
        "phi_anchor": PHI_ANCHOR,
        "lambda_tilde_c_definition": (
            "lambda_c_tilde = lambda_c / 1.2, where lambda_c is the calibrated actual liquid-condition "
            "cycling rate anchored at the canonical liquid operating point phi = 1.2 (openmucf/validate.py "
            "_OP). Actual rate at density phi: lambda_c(phi) = phi * lambda_c_tilde "
            "(openmucf.analytic.cycling_rate)."
        ),
        "transport_lineage": (
            "No new physics. Every card number is a pushforward of the existing calibrated posterior "
            "(openmucf.calibrate) through the existing analytic map (openmucf.analytic), or a labeled "
            "registered prior. Reactivation transport lineage: Stodden 1990 / Rafelski-Muller 1988-89."
        ),
    }


def _resolution_criteria() -> dict:
    return {
        "procedure": (
            "Each target resolves when Acceleron/MuFusE publishes the corresponding quantity. Published "
            "model-dependent fitted values are converted to targets via the pre-fixed basis-conversion "
            "rules in forecasts/FORECAST_PROTOCOL.md; phi(t) drift uses the run-averaged phi as stated. "
            "Scoring uses only the pre-registered estimator conventions and metrics."
        ),
        "not_scoreable_outcome": (
            "If no pre-fixed conversion rule maps a published quantity to a target, the target resolves to "
            "'not scoreable under pre-fixed conversion rules' -- a declared, legitimate outcome, not spun."
        ),
        "excluded_points": [
            "Acceleron's publicly presented preliminary DT cycling datum near 2.2x liquid density at ~50 K "
            "is a POSTDICTION (publicly visible before this card) and is OMITTED from scoring entirely to "
            "avoid contaminating the registry."
        ],
    }


def build_card(samples: dict | None = None) -> dict:
    """Build the full FC-001 card (payload + generation + registration), with payload_sha256 populated.

    ``samples`` (from :func:`posterior_samples`) may be supplied to avoid re-running the MCMC (used by the
    test fixture); when ``None`` a fresh posterior is drawn.
    """
    if samples is None:
        samples = posterior_samples()
    payload = {
        "id": "FC-001",
        "title": (
            "FC-001 -- pre-registered MuFusE forecast: effective sticking omega_s_eff and cycling rate "
            "lambda_c at high density"
        ),
        "protocol": "forecasts/FORECAST_PROTOCOL.md",
        "targets": _targets(),
        "scenarios": _scenarios(samples),
        "resolution_criteria": _resolution_criteria(),
        "scoring_rules": _scoring_rules(),
        "provenance": _provenance(),
    }
    card = {
        "payload": payload,
        "generation": {"env": _env()},
        "registration": {
            "payload_sha256": "",
            "registered_utc": None,
            "code_version_tag": None,
            "zenodo_doi": None,
            "status": "draft",
        },
    }
    card["registration"]["payload_sha256"] = payload_sha256(card)
    return card


# ------------------------------------------------------------------------------------------- validation
def _hex64(s) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def validate_card(card: dict, schema_path: Path | str | None = None) -> None:
    """Hand-rolled structural + fence validation (rates.py style: raise ValueError listing every problem).

    No new runtime dependency. Reads the schema only for the top-level required-key list; the substantive
    checks (D3 discriminator, card fences, registration state) are coded here.
    """
    schema_path = Path(schema_path) if schema_path is not None else SCHEMA_PATH
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8")) if Path(schema_path).exists() else {}
    top_required = schema.get("required", ["payload", "generation", "registration"])

    errors: list[str] = []
    for key in top_required:
        if key not in card:
            errors.append(f"missing top-level section '{key}'")
    if errors:
        raise ValueError("forecast card validation failed:\n  " + "\n  ".join(errors))

    payload = card["payload"]
    for key in ("id", "title", "protocol", "targets", "scenarios", "resolution_criteria",
                "scoring_rules", "provenance"):
        if key not in payload:
            errors.append(f"payload missing '{key}'")

    expected_ids = [_target_id(obs, phi) for phi in PHI_GRID for obs in OBSERVABLES]

    targets = payload.get("targets", [])
    if len(targets) != 6:
        errors.append(f"expected 6 targets, found {len(targets)}")
    tids = [t.get("target_id") for t in targets]
    if sorted(tids) != sorted(expected_ids):
        errors.append(f"target_ids {sorted(tids)} != pre-registered grid {sorted(expected_ids)}")
    for t in targets:
        for key in ("observable", "unit", "phi", "stated_conditions", "source", "resolution"):
            if key not in t:
                errors.append(f"target {t.get('target_id')!r} missing '{key}'")
        sc = t.get("stated_conditions", {})
        if sc.get("T_K") != [100, 150, 300]:
            errors.append(f"target {t.get('target_id')!r} stated_conditions.T_K != [100,150,300]")

    scenarios = payload.get("scenarios", [])
    names = [s.get("name") for s in scenarios]
    if names != ["A", "B"]:
        errors.append(f"scenarios must be exactly ['A','B'] in order, found {names}")
    for s in scenarios:
        preds = s.get("predictions", [])
        if sorted(p.get("target_id") for p in preds) != sorted(expected_ids):
            errors.append(f"scenario {s.get('name')!r} predictions do not cover the target grid")
        for p in preds:
            errors.extend(_check_prediction(s.get("name"), p))

    _check_registration(card.get("registration", {}), errors)
    _check_scoring_and_provenance(payload, errors)

    if errors:
        raise ValueError("forecast card validation failed:\n  " + "\n  ".join(errors))


def _is_bracket_target(scenario_name: str, target_id: str) -> bool:
    """D3: only Scenario-B lambda_c at phi > 1.45 is a bracket; everything else is an ensemble."""
    if scenario_name != "B" or not target_id.startswith("lambda_c@"):
        return False
    return float(target_id.split("phi=")[1]) > VALIDITY_EDGE


def _check_prediction(scenario_name, p: dict) -> list:
    errs = []
    tid = p.get("target_id")
    ptype = p.get("prediction_type")
    want_bracket = _is_bracket_target(scenario_name, tid)
    if want_bracket and ptype != "bracket":
        errs.append(f"scenario {scenario_name} {tid}: expected bracket, got {ptype!r}")
    if (not want_bracket) and ptype != "ensemble":
        errs.append(f"scenario {scenario_name} {tid}: expected ensemble, got {ptype!r}")
    if ptype == "ensemble":
        for key in ("median", "ci68", "ci95", "n_samples"):
            if key not in p:
                errs.append(f"scenario {scenario_name} {tid}: ensemble missing '{key}'")
    elif ptype == "bracket":
        for key in ("median_range", "ci68", "ci95", "n_samples_per_limb", "limbs"):
            if key not in p:
                errs.append(f"scenario {scenario_name} {tid}: bracket missing '{key}'")
        # D3: brackets carry NO single median and NO headline CRPS.
        for forbidden in ("median", "crps", "n_samples"):
            if forbidden in p:
                errs.append(f"scenario {scenario_name} {tid}: bracket must not carry '{forbidden}'")
        limbs = p.get("limbs", {})
        if set(limbs) != {"floor", "ceiling"}:
            errs.append(f"scenario {scenario_name} {tid}: bracket limbs must be {{floor, ceiling}}")
    else:
        errs.append(f"scenario {scenario_name} {tid}: bad prediction_type {ptype!r}")
    return errs


def _check_registration(reg: dict, errors: list) -> None:
    status = reg.get("status")
    if status not in ("draft", "registered"):
        errors.append(f"registration.status must be 'draft' or 'registered', found {status!r}")
    fields = ("registered_utc", "code_version_tag", "zenodo_doi")
    if status == "draft":
        for key in fields:
            if reg.get(key) is not None:
                errors.append(f"registration.{key} must be null in a draft card, found {reg.get(key)!r}")
    elif status == "registered":
        for key in fields:
            if not reg.get(key):
                errors.append(f"registration.{key} must be set in a registered card")
        utc = reg.get("registered_utc")
        if utc is not None and not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", str(utc)):
            errors.append("registration.registered_utc must be ISO-8601 UTC (...Z)")
    if not _hex64(reg.get("payload_sha256")):
        errors.append("registration.payload_sha256 must be a 64-char hex digest")


def _check_scoring_and_provenance(payload: dict, errors: list) -> None:
    sr = payload.get("scoring_rules", {})
    if sr.get("both_scenarios_reported") is not True:
        errors.append("scoring_rules.both_scenarios_reported must be true")
    rc = payload.get("resolution_criteria", {})
    if not rc.get("excluded_points"):
        errors.append("resolution_criteria.excluded_points must list the excluded ARPA-E point")
    prov = payload.get("provenance", {})
    if not _hex64(prov.get("ledger_sha256")):
        errors.append("provenance.ledger_sha256 must be a 64-char hex digest")
    if prov.get("phi_anchor") != PHI_ANCHOR:
        errors.append(f"provenance.phi_anchor must be {PHI_ANCHOR}")
    rp = prov.get("r_registered_prior", {})
    if rp.get("value") != [R_LO, R_HI]:
        errors.append(f"provenance.r_registered_prior.value must be [{R_LO}, {R_HI}]")
    if rp.get("kind") != "registered prior (assumption)":
        errors.append("provenance.r_registered_prior.kind must be 'registered prior (assumption)'")
    sb = prov.get("scenario_b_sampling", {})
    if sb.get("bounds") != [R_LO, R_HI]:
        errors.append(f"provenance.scenario_b_sampling.bounds must be [{R_LO}, {R_HI}]")


# --------------------------------------------------------------------------------------------- scoring
def crps_empirical(samples, y: float) -> float:
    """CRPS of an empirical predictive distribution vs a scalar truth ``y`` (empirical-CDF / energy form)::

        CRPS = mean|X - y| - (1/(2 n^2)) sum_{i,j} |x_i - x_j|

    Computed via the sorted O(n log n) identity sum_{i,j}|x_i-x_j| = 2 * sum_i (2i - n - 1) x_(i).
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    term1 = float(np.mean(np.abs(x - y)))
    i = np.arange(1, n + 1)
    term2 = float(np.sum((2 * i - n - 1) * x) / (n * n))  # = (1/(2 n^2)) sum_{i,j} |x_i - x_j|
    return term1 - term2


def interval_score(lo: float, hi: float, y: float, alpha: float) -> float:
    """Winkler / (negatively-oriented) interval score of a central (1-alpha) interval [lo, hi] vs truth y::

        IS_alpha = (hi - lo) + (2/alpha)*(lo - y)*[y < lo] + (2/alpha)*(y - hi)*[y > hi]

    Strictly proper for interval forecasts: it rewards SHARPNESS (the width term) and penalizes a MISS
    (2/alpha times the shortfall). Lower is better. Unlike per-limb CRPS or a raw coverage count, one
    number scores a bracket, so Scenario A (ensemble) and Scenario B (bracket) are comparable on one
    proper score. alpha = 0.32 for a 68% interval, 0.05 for a 95% interval.
    """
    lo, hi, y = float(lo), float(hi), float(y)
    penalty = 0.0
    if y < lo:
        penalty += (2.0 / alpha) * (lo - y)
    if y > hi:
        penalty += (2.0 / alpha) * (y - hi)
    return (hi - lo) + penalty


def coverage(intervals, ys) -> dict:
    """Fraction of ``ys`` covered by matching ``intervals`` ([lo, hi] pairs). ``ys`` may be a scalar."""
    intervals = list(intervals)
    if np.isscalar(ys):
        ys = [ys] * len(intervals)
    covered = sum(1 for (lo, hi), y in zip(intervals, ys, strict=True) if lo <= y <= hi)
    n = len(intervals)
    return {"n": n, "covered": covered, "fraction": (covered / n if n else float("nan"))}


def score_card(card: dict, resolved: dict, samples: dict | None = None) -> dict:
    """Score a card against resolved truths ``{target_id: value}`` (or 'not_scoreable').

    Ensemble targets -> CRPS + interval coverage; bracket targets -> coverage + per-limb CRPS [best, worst].
    Regenerates the predictive draws from the recorded posterior/sampling spec (pass ``samples`` from a
    module-scoped fixture to avoid re-running the MCMC).
    """
    if samples is None:
        samples = posterior_samples()
    out: dict = {}
    for scen in card["payload"]["scenarios"]:
        name = scen["name"]
        sres: dict = {}
        for pred in scen["predictions"]:
            tid = pred["target_id"]
            y = resolved.get(tid)
            if y is None or y == "not_scoreable":
                sres[tid] = {"status": "not_scoreable"}
                continue
            obs, phi = tid.split("@")[0], float(tid.split("phi=")[1])
            arr = _predictive_arrays(samples, phi, name)[obs]
            cov = {"ci68": bool(pred["ci68"][0] <= y <= pred["ci68"][1]),
                   "ci95": bool(pred["ci95"][0] <= y <= pred["ci95"][1])}
            # HEADLINE proper score for BOTH types (comparable A vs B): Winkler interval score at 68/95%.
            is68 = interval_score(pred["ci68"][0], pred["ci68"][1], y, 0.32)
            is95 = interval_score(pred["ci95"][0], pred["ci95"][1], y, 0.05)
            if pred["prediction_type"] == "ensemble":
                sres[tid] = {"prediction_type": "ensemble",
                             "interval_score_68": is68, "interval_score_95": is95,
                             "crps": crps_empirical(arr, y), "coverage": cov}
            else:
                c_floor = crps_empirical(arr["floor"], y)
                c_ceil = crps_empirical(arr["ceiling"], y)
                sres[tid] = {"prediction_type": "bracket",
                             "interval_score_68": is68, "interval_score_95": is95,
                             "crps_per_limb": [min(c_floor, c_ceil), max(c_floor, c_ceil)],
                             "coverage": cov}
        out[name] = sres
    return out


# ------------------------------------------------------------------------------------------ card I/O
def _dump(card: dict) -> str:
    return json.dumps(card, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write_card(path: Path | str = CARD_PATH) -> dict:
    """Build a fresh card and write it to ``path`` (deterministic, sorted). Returns the card."""
    card = build_card()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_dump(card), encoding="utf-8")
    return card


def regenerate(path: Path | str = CARD_PATH) -> dict:
    """Rewrite the card payload and refresh registration.payload_sha256 from the NEW payload, preserving
    all OTHER registration fields (registered_utc, code_version_tag, zenodo_doi, status) from the on-disk
    card if present."""
    p = Path(path)
    old_reg = {}
    if p.exists():
        old_reg = json.loads(p.read_text(encoding="utf-8")).get("registration", {})
    card = build_card()
    for key in ("registered_utc", "code_version_tag", "zenodo_doi", "status"):
        if key in old_reg:
            card["registration"][key] = old_reg[key]
    card["registration"]["payload_sha256"] = payload_sha256(card)  # always from the fresh payload
    p.write_text(_dump(card), encoding="utf-8")
    return card


def render_forecasts_md(card_paths) -> str:
    """Deterministically render the FORECASTS.md registry table from on-disk cards (no MCMC)."""
    lines = [
        "# FORECASTS.md -- pre-registered forecast registry (auto-generated by `openmucf.forecast`)",
        "",
        "Hash-stamped, pre-registered probabilistic forecasts. Each card is a pushforward of the "
        "existing calibrated posterior through the analytic map (no new physics); see "
        "`forecasts/FORECAST_PROTOCOL.md` for the pre-registration, basis-conversion rules, and "
        "CRPS/coverage scoring. Registration "
        "(git tag / Zenodo DOI / timestamp) is added at the first tagged release; draft cards carry a "
        "mutable registration block.",
        "",
        "| id | title | status | targets | payload_sha256 (12) | protocol |",
        "|---|---|---|---|---|---|",
    ]
    for cp in card_paths:
        card = json.loads(Path(cp).read_text(encoding="utf-8"))
        pl, reg = card["payload"], card["registration"]
        obs = ", ".join(sorted({t["observable"] for t in pl["targets"]}))
        summary = f"{len(pl['targets'])} targets ({obs}); phi in {{1.2, 2.0, 2.4}}"
        digest = (reg.get("payload_sha256") or "")[:12]
        lines.append(
            f"| {pl['id']} | {pl['title']} | {reg.get('status')} | {summary} | `{digest}` | "
            f"[{pl['protocol']}]({pl['protocol']}) |"
        )
    lines += [
        "",
        "Determinism: bit-identical regeneration under the recorded environment (including platform); "
        "cross-platform regeneration reproduces to Monte-Carlo error. Regenerate with "
        "`python scripts/generate_forecast.py` (or `make forecast`).",
    ]
    return "\n".join(lines) + "\n"
