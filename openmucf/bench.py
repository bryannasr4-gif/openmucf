"""openmucf.bench -- muCF-Bench: one case registry over two kinds of case, one runner, one report.

Two sources of truth, exposed through a single interface (never merged into one):

* **Validation side** -- the pre-registered trust gate. These cases are the reproduction- and
  consistency-tier RESULT ids that ``validate.run()`` emits (``validate.py`` owns
  ``validation_targets.csv`` and ``VALIDATION.md``; this module does not re-read the CSV or re-decide any
  verdict, it just re-exposes each ``validate.Result``). The three registered independent-prediction
  targets, which FAIL by design, are executed in ``VALIDATION.md``'s class column and deliberately kept
  out of this *friendly-reproduction* registry -- a FAIL there is a verdict about the v1 model, not the
  reproduction of a published number.
* **Reproduction side** -- self-contained JSON cases under ``openmucf/data/benchmarks/*.json`` (shipped as
  package data). Each is a *friendly reproduction*: PASS means OpenMuCF reproduces a published number, not
  a verdict on anyone's claim. Cases blocked on an unacquired document render as PENDING (run nothing,
  fail nothing) with the blocking document named.

``report_markdown`` renders both into ``BENCHMARKS.md`` (regenerated + diffed by ``make audit``). The
committed ``VALIDATION.md`` scoreboard is untouched -- this is an additive, citable view over it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import analytic, cycle, validate
from .rates import DATA

CASES_DIR = DATA / "benchmarks"

# The 7 reproduction/consistency-tier RESULT ids that ``validate.run()`` emits (post-ledger), in
# emission order -- the "validation side" of the registry. NOT the raw ``validation_targets.csv``
# target_ids (the CSV also carries paired-observation and context-only rows that produce no result),
# and NOT the five registered independent-prediction FAIL findings (V_petitjean_omega, V_faifman_900K,
# V_faifman_lowT, and the sourced Yamashita-Kino comparators V_yamashita_ratio, V_yamashita_curve) --
# those are executed in VALIDATION.md but excluded here (pre-registered divergence findings, not
# friendly reproductions). The mapping/exclusions are rendered as footnotes in BENCHMARKS.md.
VALIDATION_IDS = (
    "V_kouchen_base",
    "V_kouchen_best",
    "V_petitjean",
    "V_yamashita_lcT",
    "V_breunlich_lambdac",
    "V_faifman_peak",
    "V_nagamine_trend",
)

_REQUIRED_KEYS = {"id", "type", "title", "engine", "inputs", "provenance", "status", "notes"}
_VALID_TYPES = {"reproduction"}
_VALID_STATUS = {"active", "blocked-acquisition"}
_VALID_ENGINES = {"analytic", "cycle"}
_PROV_KEYS = {"source_bibkey", "locator", "input_basis"}


@dataclass(frozen=True)
class BenchResult:
    """One row of the bench report. ``predicted``/``expected`` are pre-rendered strings so that
    single-value validation cases and multi-input reproduction cases render uniformly."""

    id: str
    type: str  # "validation" | "reproduction"
    verdict: str  # "PASS" | "FAIL" | "DEFERRED" | "PENDING"
    predicted: str
    expected: str
    tolerance: str
    source: str
    note: str


def _check_case(case: dict, name: str) -> None:
    """Structural schema check (stdlib only). Raises ``ValueError`` listing every problem."""
    errors: list[str] = []
    if not isinstance(case, dict):
        raise ValueError(f"bench case {name}: top-level JSON must be an object")
    missing = _REQUIRED_KEYS - set(case)
    if missing:
        errors.append(f"missing required keys {sorted(missing)}")
    if case.get("type") not in _VALID_TYPES:
        errors.append(f"bad type {case.get('type')!r} (allowed: {sorted(_VALID_TYPES)})")
    if case.get("status") not in _VALID_STATUS:
        errors.append(f"bad status {case.get('status')!r} (allowed: {sorted(_VALID_STATUS)})")
    if case.get("engine") not in _VALID_ENGINES:
        errors.append(f"bad engine {case.get('engine')!r} (allowed: {sorted(_VALID_ENGINES)})")
    prov = case.get("provenance")
    if not isinstance(prov, dict) or (_PROV_KEYS - set(prov)):
        errors.append(f"provenance must be an object with keys {sorted(_PROV_KEYS)}")
    inputs = case.get("inputs")
    status = case.get("status")
    engine = case.get("engine")
    if not isinstance(inputs, list):
        errors.append("inputs must be a list")
    elif status == "blocked-acquisition":
        if inputs:
            errors.append("blocked-acquisition case must have empty inputs (no guessed conditions)")
        if "published_value" not in case:
            errors.append("blocked-acquisition case must carry a top-level published_value")
    elif status == "active":
        if not inputs:
            errors.append("active case must have at least one input")
        for i, inp in enumerate(inputs):
            if not isinstance(inp, dict):
                errors.append(f"input {i} must be an object")
                continue
            for k in ("label", "omega_s_eff_pct", "published_value", "tolerance"):
                if k not in inp:
                    errors.append(f"input {i} missing {k!r}")
            if engine == "analytic" and "lambda_c" not in inp:
                errors.append(f"input {i} (analytic) missing 'lambda_c'")
            if engine == "cycle":
                cond = inp.get("conditions")
                if not isinstance(cond, dict) or {"T", "phi", "c_t"} - set(cond):
                    errors.append(f"input {i} (cycle) needs conditions with T, phi, c_t")
    if errors:
        raise ValueError(f"bad bench case {name}: " + "; ".join(errors))


def load_cases(cases_dir=None) -> dict[str, dict]:
    """Load + schema-check every ``*.json`` reproduction case, keyed by ``id`` (sorted by filename)."""
    root = Path(cases_dir) if cases_dir is not None else CASES_DIR
    cases: dict[str, dict] = {}
    for path in sorted(Path(root).glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        _check_case(case, path.name)
        cid = case["id"]
        if cid in cases:
            raise ValueError(f"duplicate bench case id {cid!r}")
        cases[cid] = case
    return cases


def case_ids(rates, cases_dir=None) -> list[str]:
    """All bench case ids: the 8 validation result ids + the JSON reproduction ids (10 total)."""
    return list(VALIDATION_IDS) + list(load_cases(cases_dir))


def _validation_bench_result(result: validate.Result) -> BenchResult:
    """Wrap a ``validate.Result`` as a BenchResult (verdict/predicted copied verbatim -- no re-decision)."""
    verdict = "DEFERRED" if result.passed is None else ("PASS" if result.passed else "FAIL")
    pred = "n/a" if result.predicted != result.predicted else f"{result.predicted:.4g}"
    return BenchResult(
        id=result.target_id,
        type="validation",
        verdict=verdict,
        predicted=pred,
        expected=result.observed,
        tolerance=result.tolerance,
        source="openmucf.validate (validation_targets.csv trust gate)",
        note=result.note,
    )


def _run_json_case(rates, case: dict) -> BenchResult:
    """Run one JSON reproduction case (or render it PENDING if blocked)."""
    prov = case["provenance"]
    source = f"{prov['source_bibkey']} ({prov['locator']})"
    if case["status"] == "blocked-acquisition":
        return BenchResult(
            id=case["id"],
            type="reproduction",
            verdict="PENDING",
            predicted="n/a",
            expected=str(case.get("published_value", "?")),
            tolerance="n/a",
            source=source,
            note=case["notes"],
        )
    engine = case["engine"]
    preds: list[str] = []
    exps: list[str] = []
    tols: list[str] = []
    all_pass = True
    for inp in case["inputs"]:
        ose = float(inp["omega_s_eff_pct"]) / 100.0
        if engine == "analytic":
            pred = float(analytic.fusions_per_muon(ose, float(inp["lambda_c"])))
        else:  # cycle
            c = inp["conditions"]
            pred = float(
                cycle.fusions_per_muon_from_conditions(
                    rates, float(c["T"]), float(c["phi"]), float(c["c_t"]), omega_s_eff=ose
                )
            )
        ok = validate._within(pred, float(inp["published_value"]), inp["tolerance"])
        all_pass = all_pass and ok
        preds.append(f"{pred:.4g}")
        exps.append(str(inp["published_value"]))
        tols.append(inp["tolerance"])
    tol = tols[0] if len(set(tols)) == 1 else "; ".join(tols)
    return BenchResult(
        id=case["id"],
        type="reproduction",
        verdict="PASS" if all_pass else "FAIL",
        predicted=" / ".join(preds),
        expected=" / ".join(exps),
        tolerance=tol,
        source=source,
        note=case["notes"],
    )


def run_case(rates, case_id, cases_dir=None) -> BenchResult:
    """Run one case by id: a validation id dispatches to ``validate.run()``; a JSON id to its engine."""
    if case_id in VALIDATION_IDS:
        results = {r.target_id: r for r in validate.run(rates)}
        return _validation_bench_result(results[case_id])
    cases = load_cases(cases_dir)
    if case_id not in cases:
        raise KeyError(f"unknown bench case id {case_id!r}")
    return _run_json_case(rates, cases[case_id])


def run_all(rates, cases_dir=None) -> list[BenchResult]:
    """Run every case: the 8 validation ids (one ``validate.run()`` call) then the JSON cases."""
    results = {r.target_id: r for r in validate.run(rates)}
    out = [_validation_bench_result(results[vid]) for vid in VALIDATION_IDS]
    out += [_run_json_case(rates, case) for case in load_cases(cases_dir).values()]
    return out


_VERDICT_MD = {"PASS": "PASS", "FAIL": "**FAIL**", "DEFERRED": "DEFERRED", "PENDING": "PENDING"}


def report_markdown(results) -> str:
    """Render BENCHMARKS.md: every case (id, type, source, verdict) + the mapping/exclusion footnotes."""
    lines = [
        "# BENCHMARKS.md -- muCF-Bench registry (auto-generated by `openmucf.bench`)",
        "",
        "One registry over two kinds of case. **Validation** cases are the pre-registered trust gate "
        "(re-exposed from `openmucf.validate`; the committed `VALIDATION.md` remains the single source of "
        "validation truth and is unchanged by this view). **Reproduction** cases are self-contained, "
        "citable friendly reproductions of published numbers -- PASS means OpenMuCF reproduces the "
        "published value within a pre-registered tolerance, not a verdict on anyone's claim.",
        "",
        "Reproduce one case from an installed package with `openmucf reproduce <case-id>` "
        "(or `openmucf reproduce --all`); regenerate this file with `make bench`.",
        "",
        "| case | type | source | predicted | expected | tolerance | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.id} | {r.type} | {r.source} | {r.predicted} | {r.expected} | "
            f"{r.tolerance} | {_VERDICT_MD.get(r.verdict, r.verdict)} |"
        )
    n_pass = sum(1 for r in results if r.verdict == "PASS")
    n_fail = sum(1 for r in results if r.verdict == "FAIL")
    n_defer = sum(1 for r in results if r.verdict == "DEFERRED")
    n_pend = sum(1 for r in results if r.verdict == "PENDING")
    lines += [
        "",
        f"**Summary: {len(results)} cases -- {n_pass} pass, {n_fail} fail, "
        f"{n_defer} deferred, {n_pend} pending.**",
        "",
        "Notes on the validation-to-registry mapping (the validation side re-exposes engine RESULTS, not "
        "raw CSV rows):",
        "",
        "- The five registered independent-prediction targets (`V_petitjean_omega`, `V_faifman_900K`, "
        "`V_faifman_lowT`, and the sourced Yamashita-Kino comparators `V_yamashita_ratio`, "
        "`V_yamashita_curve`) are executed in `VALIDATION.md` (they FAIL by design) and are deliberately "
        "NOT bench cases -- this registry reproduces published numbers, not pre-registered divergence "
        "findings. In particular `V_petitjean` here runs the CSV row `V_petitjean_Xmu`; `V_petitjean_omega` "
        "is its separate registered sticking prediction, shown in VALIDATION.md's class column.",
        "- The context-only rows `A_acceleron_density` and `A_acceleron_anomaly` (tolerance `context-only`) "
        "are regime anchors, not runnable reproductions, and are deliberately not bench cases.",
        "- Blocked reproduction cases render as PENDING with the blocking document named; they run nothing "
        "and fail nothing until the document is acquired.",
    ]
    return "\n".join(lines) + "\n"
