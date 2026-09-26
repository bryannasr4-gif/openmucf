"""The jax cache bound of the slow MCMC loops (`bound_jax_caches` in `tests/conftest.py`).

It clears after exactly the fits that end a block of `JAX_CACHE_CLEAR_EVERY`, and both slow loops call
it once per fit with their loop index. Clearing leaves the draws of `likelihood.fit_spectrum` and
`calibrate.run_mcmc` bit-identical at reduced sampler settings on every run, and, in the `slow` test
here, at the slow loops' sampler settings and seeds over their first `2 * JAX_CACHE_CLEAR_EVERY + 1`
fits, which it compares by a digest of each fit's draws and prints.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib

import conftest
import jax
import numpy as np
import pytest

from openmucf import calibrate, likelihood
from openmucf.constants import LAMBDA_0

TESTS = pathlib.Path(__file__).parent
K = conftest.JAX_CACHE_CLEAR_EVERY


def _name(func: ast.expr) -> str:
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def test_bound_clears_after_the_last_fit_of_each_block_only(monkeypatch):
    assert K >= 2  # at K == 1 an off-by-one in the block arithmetic would be invisible
    cleared = []
    monkeypatch.setattr(jax, "clear_caches", lambda: cleared.append(True))
    returned = [i for i in range(3 * K) if conftest.bound_jax_caches(i)]
    assert returned == [K - 1, 2 * K - 1, 3 * K - 1]
    assert len(cleared) == 3


def test_both_slow_loops_call_the_bound_once_per_fit():
    for name, fit in (("test_calibrate_sbc.py", "run_mcmc"), ("test_twin_coverage.py", "fit_spectrum")):
        tree = ast.parse((TESTS / name).read_text(encoding="utf-8"))
        loops = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and any(isinstance(c, ast.Call) and _name(c.func) == fit for c in ast.walk(node))
        ]
        assert len(loops) == 1, (name, len(loops))
        loop = loops[0]
        calls = [c for c in ast.walk(loop) if isinstance(c, ast.Call) and _name(c.func) == "bound_jax_caches"]
        top = [
            s.value for s in loop.body
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
            and _name(s.value.func) == "bound_jax_caches"
        ]
        assert len(calls) == 1 and len(top) == 1, name
        assert isinstance(loop.target, ast.Name), name
        assert [ast.unparse(a) for a in top[0].args] == [loop.target.id], name


def _draws(samples) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in samples.items()}


def _assert_bit_identical(plain, cleared) -> None:
    assert len(plain) == len(cleared)
    for a, b in zip(plain, cleared, strict=True):
        assert sorted(a) == sorted(b)
        for key in a:
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
            assert a[key].tobytes() == b[key].tobytes(), key


def _spectrum_fits(seeds, clear: bool):
    t_edges = np.linspace(0.0, 30.0e-6, 17)
    expected = np.asarray(likelihood.expected_counts_closed_form(t_edges, 0.00557, 1.30e8, 1.0e5, 3.0e7))
    out = []
    for seed in seeds:
        if clear:
            jax.clear_caches()
        counts = np.random.default_rng(seed).poisson(expected)
        samples = likelihood.fit_spectrum(t_edges, counts, num_warmup=50, num_samples=100, seed=seed)
        out.append(_draws(samples))
    return out


def _calibration_fits(seeds, clear: bool):
    out = []
    for seed in seeds:
        if clear:
            jax.clear_caches()
        out.append(_draws(calibrate.run_mcmc(num_warmup=50, num_samples=100, seed=seed, num_chains=2)))
    return out


def test_clearing_leaves_the_spectrum_draws_bit_identical():
    plain = _spectrum_fits((0, 1), clear=False)
    cleared = _spectrum_fits((0, 1), clear=True)
    _assert_bit_identical(plain, cleared)


def test_clearing_leaves_the_calibration_draws_bit_identical():
    plain = _calibration_fits((0, 1), clear=False)
    cleared = _calibration_fits((0, 1), clear=True)
    _assert_bit_identical(plain, cleared)


#: Fits per loop in the slow comparison: the fits after each of the first two clears are among them.
N_SLOW = 2 * K + 1


def _digest(samples) -> str:
    """SHA-256 over every draw array of one fit: site name, dtype, shape and raw bytes, sites sorted."""
    h = hashlib.sha256()
    for key in sorted(samples):
        a = np.ascontiguousarray(np.asarray(samples[key]))
        h.update(f"{key}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _slow_twin_fit(i):
    t_edges = np.linspace(0.0, 30.0e-6, 65)
    expected = likelihood.expected_counts_closed_form(t_edges, 0.557 / 100.0, 1.30e8, 1.0e5, 3.0e7)
    counts = np.random.default_rng(1000 + i).poisson(np.asarray(expected))
    return likelihood.fit_spectrum(t_edges, counts, num_warmup=300, num_samples=800, seed=i)


def _slow_sbc_inputs(n):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        os0 = rng.uniform(*calibrate.WEAK_OMEGA_S0_PRIOR[1:])
        r = rng.uniform(*calibrate.R_PRIOR_DEFAULT)
        lc = rng.uniform(*calibrate.LAMBDA_C_PRIOR_DEFAULT)
        ose = os0 * (1.0 - r)
        xmu = 1.0 / (ose / 100.0 + LAMBDA_0 / lc)
        out.append((float(rng.normal(ose, calibrate.OBS["omega_s_eff_sd"])),
                    float(rng.normal(xmu, calibrate.OBS["xmu_sd"]))))
    return out


def _slow_sbc_fit(i, y_ose, y_xmu):
    return calibrate.run_mcmc(
        num_warmup=500, num_samples=1000, seed=i, num_chains=2,
        omega_s_eff_obs=y_ose, xmu_obs=y_xmu,
    )


def _sampler_keywords(tree: ast.AST, fit: str) -> list[dict[str, str]]:
    return [
        {kw.arg: ast.unparse(kw.value) for kw in c.keywords}
        for c in ast.walk(tree) if isinstance(c, ast.Call) and _name(c.func) == fit
    ]


def test_the_slow_comparison_uses_the_slow_loops_sampler_settings_and_seeds():
    here = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
    mine = {f.name: f for f in here.body if isinstance(f, ast.FunctionDef)}
    for name, fit, helper in (("test_calibrate_sbc.py", "run_mcmc", "_slow_sbc_fit"),
                              ("test_twin_coverage.py", "fit_spectrum", "_slow_twin_fit")):
        loop = _sampler_keywords(ast.parse((TESTS / name).read_text(encoding="utf-8")), fit)
        ours = _sampler_keywords(mine[helper], fit)
        assert len(loop) == 1 and ours == loop, (name, loop, ours)


@pytest.mark.slow
def test_clearing_leaves_the_slow_loops_draws_bit_identical():
    """The first N_SLOW fits of each slow loop's sampler call, run once with no clear between fits and once
    through `bound_jax_caches`, give equal draw digests fit by fit. Prints every pair."""
    sbc = _slow_sbc_inputs(N_SLOW)
    for name, fit in (("sbc", lambda i: _slow_sbc_fit(i, *sbc[i])), ("twin", _slow_twin_fit)):
        jax.clear_caches()
        plain = [_digest(fit(i)) for i in range(N_SLOW)]
        jax.clear_caches()
        cleared = []
        for i in range(N_SLOW):
            cleared.append(_digest(fit(i)))
            conftest.bound_jax_caches(i)
        jax.clear_caches()
        for i, (p, c) in enumerate(zip(plain, cleared, strict=True)):
            print(f"cache-bound digest {name} fit {i:02d} plain={p} cleared={c}")
        assert plain == cleared, name
