"""WS-T slow gate: interval calibration of the counts-level likelihood (200 seeded replicas).

Marked `slow`: deselected from the default `pytest -q` (and CI) by the `addopts = "-m 'not slow'"` in
pyproject.toml; run it explicitly with `pytest -m slow -q` (~9 min). It is the workstream's own gate
(G-T2), executed once and its observed coverage recorded in RESULTS -- NOT a CI job (the coverage run is
too long for CI, and its Poisson replicas are deterministic only on a fixed platform).

What it checks: the DISAPPEARANCE RATE lambda_n is the quantity a delta-pulse counts histogram actually
constrains (the decay slope); its 95% credible interval should cover the truth at ~the nominal rate.
(omega_s_eff and lambda_c individually are prior-limited -- see the identifiability note in likelihood.py
-- so THEIR intervals over-cover; lambda_n is the honestly data-constrained, calibration-testable one.)
"""

import numpy as np
import pytest

from openmucf import likelihood
from openmucf.constants import LAMBDA_0

# Canonical liquid OP truth (ledger-nominal cycling rate, central in the informative prior).
_T_EDGES = np.linspace(0.0, 30.0e-6, 65)
_OSE_PCT, _LAMBDA_C, _AMP, _BKG = 0.557, 1.30e8, 1.0e5, 3.0e7
_LAMBDA_N_TRUE = LAMBDA_0 + (_OSE_PCT / 100.0) * _LAMBDA_C
_N_REPLICAS = 200


@pytest.mark.slow
def test_interval_calibration_200_replicas():
    """95% credible interval for lambda_n covers truth in [88%, 99%] of 200 seeded synthetic replicas
    (observed 189/200 = 94.5% on the pinned x86 env; binomial slack for 200 draws)."""
    expected = np.asarray(
        likelihood.expected_counts_closed_form(_T_EDGES, _OSE_PCT / 100.0, _LAMBDA_C, _AMP, _BKG)
    )
    covered = 0
    for i in range(_N_REPLICAS):
        counts = np.random.default_rng(1000 + i).poisson(expected)
        samples = likelihood.fit_spectrum(_T_EDGES, counts, num_warmup=300, num_samples=800, seed=i)
        lo, hi = np.percentile(samples["lambda_n"], [2.5, 97.5])
        covered += int(lo <= _LAMBDA_N_TRUE <= hi)
    frac = covered / _N_REPLICAS
    assert 0.88 <= frac <= 0.99, f"lambda_n 95% CI coverage {covered}/{_N_REPLICAS} = {frac:.1%}"
