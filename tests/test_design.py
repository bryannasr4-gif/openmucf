"""WS-D design.py -- Bayesian experimental design over the calibrate posterior.

The three sanity gates (zero-EIG replicate, EIG monotone in precision, Sobol-consistency in the
small-noise limit) are TESTS here, plus the C4 registry conditional (mocked X-ray verdict, both ways),
the nested-MC settings echo, and (slow) the sd-contraction refit behaviour. EIG runs over a single shared
posterior fixture; the Sobol gate is pure numpy; only the sd-contraction refit test carries NUTS refits
and is marked `slow`.
"""

import numpy as np
import pytest

from openmucf import design, uq


@pytest.fixture(scope="module")
def base():
    """One weak-prior calibration posterior, shared by the EIG sanity-gate tests (seeded)."""
    return design.base_posterior(seed=0)


# --------------------------------------------------------------------- sanity gate 1: zero-EIG replicate
def test_zero_eig_for_exact_replicate(base):
    """Re-observing an already-pinned constant is independent of theta -> EIG = 0 to MC noise."""
    out = design.eig_nested_mc(design.replicate_candidate(), n_outer=256, n_inner=256, seed=0, samples=base)
    assert abs(out["eig_bits"]) < 0.02, out["eig_bits"]


# ------------------------------------------------------------- sanity gate 2: EIG monotone in precision
def test_eig_monotone_in_stated_precision(base):
    """A tighter measurement never lowers EIG (3-point sigma sweep on a real candidate, C1)."""
    eigs = [design.eig_nested_mc("C1", sigma_rel=s, seed=0, samples=base)["eig_bits"]
            for s in (0.10, 0.05, 0.02)]
    assert eigs[0] <= eigs[1] + 1e-9 <= eigs[2] + 1e-9, eigs
    assert eigs[2] > eigs[0]  # strictly more information at 5x tighter precision


# ------------------------------------------------------- sanity gate 3: Sobol-consistency (small noise)
def test_sobol_consistency_small_noise_limit():
    """The parameter a tiny-sigma X_mu measurement informs most == the top Sobol driver of X_mu over the
    SAME uq contested prior box."""
    sc = design.sobol_consistency(sigma_rel=0.02, n=100_000, seed=0)
    si = uq.sobol_indices(N=4096, output="X_mu", seed=0)["ST"]
    top_sobol = max(("omega_s0_pct", "R", "lambda_c"), key=lambda k: si[k])
    assert sc["top_param"] == top_sobol == "R", (sc["top_param"], top_sobol)


# --------------------------------------------------------------- C4 registry conditional (mocked verdict)
def test_registry_c4_conditional_included_and_dropped():
    """C4 is a STRUCTURAL function of the passed-in X-ray verdict: included at >=15%, dropped below."""
    inc = design.registry(42.95, threshold_pct=15.0)  # the logged weak-prior-chain verdict
    assert inc["c4_included"] is True
    assert "C4" in inc["candidates"] and inc["dropped"] is None

    out = design.registry(10.0, threshold_pct=15.0)  # a counterfactual sub-threshold verdict
    assert out["c4_included"] is False
    assert "C4" not in out["candidates"]
    assert out["dropped"] is not None and "C4 dropped" in out["dropped"]

    # exactly at threshold is INCLUDED (>=)
    assert design.registry(15.0, threshold_pct=15.0)["c4_included"] is True


# ------------------------------------------------------------------ nested-MC settings echoed in output
def test_nested_mc_settings_echoed_in_output():
    """The inner/outer settings + the mandatory bias caveat are reported IN the output dict (no NUTS: a
    synthetic posterior exercises the estimator plumbing)."""
    rng = np.random.default_rng(0)
    synth = {
        "omega_s0_pct": rng.uniform(0.60, 1.10, 500),
        "R": rng.uniform(0.10, 0.60, 500),
        "lambda_c": rng.uniform(0.8e8, 1.6e8, 500),
    }
    synth["omega_s_eff_pct"] = synth["omega_s0_pct"] * (1.0 - synth["R"])
    out = design.eig_nested_mc("C2", n_outer=64, n_inner=128, seed=3, samples=synth)
    assert out["n_outer"] == 64 and out["n_inner"] == 128 and out["seed"] == 3
    assert "bias" in out["bias_caveat"].lower()
    assert out["candidate"] == "C2"


# --------------------------------------------------------------------- sd-contraction refit (NUTS; slow)
@pytest.mark.slow
def test_sd_contraction_refit_behaviour(base):
    """The PRIMARY metric refits with the observable appended, and reports its own Monte-Carlo error.

    Asserts STRUCTURAL properties only. An earlier version of this test asserted an OUTCOME at n_synth=3
    (``c1["R_contraction"]["inflated"] < 0.03`` and a strict constant > inflated ordering) -- a coin-flip
    on three noisy refits whose per-refit spread is 4-18 pp. It passed on the host it was written on and
    encoded the same over-claim the 2026-07-23 cross-architecture reproduction refuted. Whether the class
    contrast is RESOLVED is a measurement DESIGN.md now reports with its sigma; it is not a test fixture.
    """
    c2 = design.sd_contraction("C2", n_synth=3, seed=0, samples=base)
    assert c2["R_contraction"]["constant"] == c2["R_contraction"]["inflated"]  # class-independent
    assert c2["R_contraction_class_delta"]["value"] == 0.0                     # ... exactly, by construction
    assert set(c2["sd_before"]) == {"omega_s_eff_pct", "R"}
    assert c2["n_synth"] == 3

    c1 = design.sd_contraction("C1", n_synth=3, seed=0, samples=base)
    assert c1["class_sensitive"] is True
    # the inflated refit really decoupled R at the design point (different draws, not a mirrored copy)
    assert c1["raw"]["constant"]["R"] != c1["raw"]["inflated"]["R"]
    # common random numbers: both classes used the SAME theta* draws, so sd_before is shared
    assert c1["sd_before"] == c2["sd_before"]
    # every reported cell carries a finite, strictly positive Monte-Carlo standard error
    for se in (c1["ose_contraction_se"], c1["R_contraction_se"]["constant"],
               c1["R_contraction_se"]["inflated"], c1["R_contraction_class_delta"]["se"]):
        assert se == se and se > 0.0
    assert c1["R_contraction_class_delta"]["paired"] is True
