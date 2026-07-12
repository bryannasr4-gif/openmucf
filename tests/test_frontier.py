"""WS-Q: frontier.py -- inverse-design "what would have to be true".

These tests lock: the closed-form R-floor reproduces the FINDINGS.md sec.3 "R >= 0.77" headline
bit-for-bit (regression lock to openmucf.uq.breakeven_audit); the closed-form and optimistix-Newton
solver paths agree to < 1e-9 (relative) for every free variable, with q_net(inverse) == target to < 1e-12;
the (lambda_c, R) frontier is monotone decreasing in lambda_c; the Newton solver converges (no NaNs) from
the documented initialisations across the grid; the optimistix dependency is DECLARED (import + pyproject);
and FRONTIER.md + its manifest regenerate deterministically and verify. NO verdict-shaped surface is built
(sec.3.2 fence): the module exposes only inverse-design functions.
"""

from __future__ import annotations

import importlib.util
import math
import tomllib
from pathlib import Path

import pytest

import openmucf
from openmucf import frontier, provenance, uq
from openmucf.analytic import effective_sticking, fusions_per_muon
from openmucf.frontier import (
    FREE_VARS,
    LAMBDA_C_NOMINAL,
    OMEGA_S0_DEFAULT,
    R_NOMINAL,
    frontier_lambda_c_R,
    lambda_c_required,
    q_net_at,
    r_required,
    solve_inverse,
)
from openmucf.systems import SystemChain

REPO = Path(openmucf.__file__).resolve().parent.parent
_SCRIPT = REPO / "scripts" / "generate_frontier.py"

# Solver cross-check target: a modest q_net (above the nominal operating point) chosen so all four free
# variables have a finite, feasible required value -- matches the generator's SOLVER_TARGET_QNET.
_TARGET = 0.06


def _load_generator():
    """Import the generator by path (file I/O + MCMC figure are guarded behind main(), so import is inert)."""
    spec = importlib.util.spec_from_file_location("_gen_frontier", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _closed_form(free_var, target, chain, omega_s0, R, lambda_c):
    """Independent closed-form required value (the algebra the solver must reproduce)."""
    x_fixed = float(fusions_per_muon(effective_sticking(omega_s0, R), lambda_c))
    netf = chain.blanket_M * chain.eta_thermal * chain.eta_acc * (1.0 - chain.recirc_fraction)
    x_tgt = target * chain.E_mu_MeV / (chain.E_per_fusion_MeV * netf)
    if free_var == "E_mu_GeV":
        return x_fixed * chain.E_per_fusion_MeV * netf / target / 1.0e3
    if free_var == "eta_acc":
        denom = x_fixed * chain.E_per_fusion_MeV * chain.blanket_M * chain.eta_thermal
        denom *= 1.0 - chain.recirc_fraction
        return target * chain.E_mu_MeV / denom
    if free_var == "R":
        return r_required(x_tgt, lambda_c, omega_s0)
    if free_var == "lambda_c":
        return lambda_c_required(x_tgt, R, omega_s0)
    raise KeyError(free_var)


# --------------------------------------------------------------------------------------------------
# The regression lock: X_mu=500, lambda_c->inf reproduces the FINDINGS sec.3 "R >= 0.77" headline
# --------------------------------------------------------------------------------------------------
def test_r_floor_reproduces_findings_R_ge_077():
    """r_required(500, inf) equals uq.breakeven_audit's R-floor BIT-FOR-BIT and formats to the committed
    FINDINGS.md sec.3 headline 'R >= 0.77'. This is the regression lock to the existing finding."""
    r_inf = r_required(500.0, math.inf)  # omega_s0 defaults to the Kamimura 0.857% fraction
    be = uq.breakeven_audit(n=100, seed=1)  # R_required fields are deterministic (independent of n)
    assert r_inf == be["R_required_at_infinite_lambda_c"]  # bit-identical, not just close
    assert f"{r_inf:.2f}" == "0.77"
    # and it is the number FINDINGS.md actually ships
    findings = (REPO / "FINDINGS.md").read_bytes().decode("utf-8")
    assert "R >= 0.77" in findings.replace("\\", "")


def test_r_required_generalizes_findings_endpoints():
    """The generalised identity reproduces the FINDINGS sec.3 (lambda_c, R) endpoints exactly:
    (2.28e8 -> R->1) and (3.00e8 -> 0.94), matching uq.breakeven_audit at 3e8."""
    be = uq.breakeven_audit(n=100, seed=1)
    assert r_required(500.0, 3.0e8) == be["R_required_at_lambda_c_3e8"]  # bit-identical
    assert f"{r_required(500.0, 3.0e8):.2f}" == "0.94"
    assert r_required(500.0, 2.28e8) > 0.999  # R -> 1 at the density anchor


# --------------------------------------------------------------------------------------------------
# Closed-form vs optimistix-Newton solver: agree to < 1e-9 (relative); residual == target to < 1e-12
# --------------------------------------------------------------------------------------------------
def test_closed_form_vs_solver_agreement_all_free_vars():
    """For every free variable, the Newton root-find over the differentiable q_net graph reproduces the
    closed-form required value to < 1e-9 relative (absolute 1e-9 is meaningless for the ~3e8 lambda_c root;
    relative is the correct, and far-exceeded, gate)."""
    sc = SystemChain()
    target = _TARGET  # chosen so all four required values are finite/feasible
    for fv in FREE_VARS:
        cf = _closed_form(fv, target, sc, OMEGA_S0_DEFAULT, R_NOMINAL, LAMBDA_C_NOMINAL)
        sv = solve_inverse(target, fv, R=R_NOMINAL, lambda_c=LAMBDA_C_NOMINAL)
        assert math.isfinite(sv)
        assert math.isclose(sv, cf, rel_tol=1e-9), (fv, sv, cf)


def test_q_net_of_inverse_equals_target():
    """q_net evaluated at each inverse solution returns the target to < 1e-12 (the round-trip invariant)."""
    target = _TARGET
    for fv in FREE_VARS:
        sv = solve_inverse(target, fv, R=R_NOMINAL, lambda_c=LAMBDA_C_NOMINAL)
        got = q_net_at(fv, sv, R=R_NOMINAL, lambda_c=LAMBDA_C_NOMINAL)
        assert math.isclose(got, target, rel_tol=0.0, abs_tol=1e-12), (fv, got, target)


def test_solver_converges_across_grid_no_nans():
    """Newton converges (finite, no NaNs) from the documented initialisations across a target grid, for
    every free variable -- the off-grid-init robustness gate."""
    sc = SystemChain()
    for target in (0.02, 0.04, 0.06, 0.075):
        for fv in FREE_VARS:
            cf = _closed_form(fv, target, sc, OMEGA_S0_DEFAULT, R_NOMINAL, LAMBDA_C_NOMINAL)
            if not math.isfinite(cf):
                continue  # a closed-form 'inf' (unreachable) has no finite root -- not a solver case
            sv = solve_inverse(target, fv, R=R_NOMINAL, lambda_c=LAMBDA_C_NOMINAL)
            assert math.isfinite(sv) and not math.isnan(sv), (target, fv, sv)
            assert math.isclose(sv, cf, rel_tol=1e-9), (target, fv, sv, cf)


# --------------------------------------------------------------------------------------------------
# Frontier monotonicity + the honest 'unreachable' readouts
# --------------------------------------------------------------------------------------------------
def test_frontier_monotonic_decreasing_in_lambda_c():
    """d R_required / d lambda_c < 0: along every X_mu row, R falls strictly as lambda_c rises."""
    grid = [1.0e8, 1.3e8, 1.45e8, 2.0e8, 2.28e8, 3.0e8]
    for X in (150.0, 284.0, 500.0):
        rs = [r for _, r in frontier_lambda_c_R(X, grid)]
        assert all(a > b for a, b in zip(rs, rs[1:], strict=False)), (X, rs)
        # the decay-free limit is the strict floor (below every finite-lambda_c value on the row)
        assert r_required(X, math.inf) < rs[-1]


def test_lambda_c_required_inverts_forward_map_and_flags_unreachable():
    """lambda_c_required round-trips the forward yield map, and returns inf when the sticking floor alone
    caps X below the target (unreachable at that R)."""
    X, R = 150.0, 0.35
    lc = lambda_c_required(X, R)
    assert math.isfinite(lc)
    # forward-map round trip: X = 1/(omega_s_eff + lambda_0/lambda_c)
    x_back = float(fusions_per_muon(effective_sticking(OMEGA_S0_DEFAULT, R), lc))
    assert math.isclose(x_back, X, rel_tol=1e-9)
    # unreachable: X above the sticking-floor cap 1/omega_s_eff -> inf
    cap = 1.0 / effective_sticking(OMEGA_S0_DEFAULT, R)
    assert lambda_c_required(cap * 1.5, R) == math.inf


def test_r_required_infeasible_readout_not_clamped():
    """A required R > 1 is reported honestly (never clamped into [0,1]): X_mu=500 at the measured
    lambda_c=1.0e8 needs R = 1.30 (> 1, physically unreachable in reactivation alone)."""
    r = r_required(500.0, 1.0e8)
    assert r > 1.0
    assert f"{r:.4f}" == "1.2978"


# --------------------------------------------------------------------------------------------------
# Dependency declaration: optimistix is a DIRECT import + a declared [project] dependency
# --------------------------------------------------------------------------------------------------
def test_optimistix_import_declared():
    """frontier imports optimistix directly, and pyproject.toml declares it as an explicit [project]
    dependency (promoted from transitive-via-diffrax) -- the honest direct-import declaration."""
    assert frontier.optx.__name__ == "optimistix"
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    assert any(d.replace(" ", "").lower().startswith("optimistix") for d in deps), deps


def test_frontier_is_inverse_design_only_no_verdict_surface():
    """sec.3.2 fence: frontier exposes ONLY inverse-design helpers -- no scenario/verdict registry, enum,
    or table-audit surface. Guards against the fenced-out verdict mode creeping in."""
    public = {n for n in dir(frontier) if not n.startswith("_")}
    forbidden = {"verdict", "scenario", "registry", "audit_table", "ykc_table", "Verdict", "Scenario"}
    assert public.isdisjoint(forbidden), public & forbidden
    assert "frontier" not in getattr(openmucf, "__all__", [])  # submodule, out of the eager surface


def test_solve_inverse_rejects_unknown_free_var():
    with pytest.raises(ValueError):
        solve_inverse(0.06, "not_a_var", R=R_NOMINAL, lambda_c=LAMBDA_C_NOMINAL)


# --------------------------------------------------------------------------------------------------
# Doc / manifest determinism
# --------------------------------------------------------------------------------------------------
def test_frontier_md_regenerates_byte_identical():
    """FRONTIER.md rebuilt from the generator equals the committed file byte-for-byte (LF-normalised).
    Closed-form only -- no MCMC/solver in the byte-diffed path."""
    gen = _load_generator()
    rebuilt = gen.build_markdown(gen.build_headline())
    committed = (REPO / "FRONTIER.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert rebuilt == committed


def test_frontier_manifest_verifies():
    """The committed FRONTIER_MANIFEST.json verifies against the committed FRONTIER.md (no doc drift)."""
    failures = provenance.check_manifest(REPO / "FRONTIER_MANIFEST.json", repo_root=REPO)
    assert failures == [], failures


def test_frontier_manifest_solver_numbers_are_six_sig_figs():
    """WAVE2 A2: every solver-worked-example value shipped in FRONTIER.md is quantised to <= 6 significant
    figures (so the byte-diffed doc cannot depend on iterative-solver noise)."""
    gen = _load_generator()
    H = gen.build_headline()
    for key in ("solver_E_mu_GeV", "solver_eta_acc", "solver_R", "solver_lambda_c"):
        digits = H[key].lower().replace("e+", "e").split("e")[0].replace(".", "").replace("-", "").lstrip("0")
        assert len(digits) <= 6, (key, H[key], digits)
