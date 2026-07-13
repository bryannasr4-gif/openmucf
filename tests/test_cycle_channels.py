"""WS-N: two absorbing loss channels (ttmu side-branch + He-3 scavenging) + analytic v2 + reduction gate.

The engine default is channels-OFF; these tests exercise the explicit opt-in and the reduction to v1.
The FULL 50-point bit-exact reduction + the FINDINGS byte-lock are pinned by
``tests/test_cycle.py::test_wsn_norm_excludes_loss_accumulators_bit_exact`` (Fable amendment, spec §3.4).
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from openmucf import analytic as A
from openmucf import cycle, load_rates

_REF = json.loads((Path(__file__).parent / "cycle_v1_reference.json").read_text(encoding="utf-8"))


def test_reduction_gate_channels_zero():
    """G-N1: the channels-zeroed 8-state solve reproduces the recorded v1 trajectory to atol 1e-9, and a
    legacy 6-element y0 still solves (the padding contract)."""
    a6 = _REF["args6"]
    ts = _REF["ts"]
    yref = np.array(_REF["ys"])
    for i in range(0, len(ts), 10):  # representative subset (full 50-pt bit-exact lock is in test_cycle.py)
        y = np.asarray(cycle.solve_cycle(*a6, t1=float(ts[i])).ys[-1])[:6]
        assert np.max(np.abs(y - yref[i])) < 1e-9, (i, float(np.max(np.abs(y - yref[i]))))
    # padding contract: a length-6 y0 still solves and gives the same X_mu as the default 8-vector
    x6 = float(cycle.solve_cycle(*a6, y0=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])).ys[-1, 3])
    x8 = float(cycle.solve_cycle(*a6).ys[-1, 3])
    assert x6 == x8


def test_conservation_with_channels():
    """Muon conservation (incl. the two new accumulators y[6]+y[7]) holds to 1e-9 with channels ON."""
    rt = load_rates()
    p = cycle.params_from_conditions(rt, 300.0, 1.2, 0.5, c_he=0.2, include_loss_channels=True)
    p["lambda_tt"] = 5.0e6  # force a nonzero tt rate (the ledger tt row is blocked=0.0) to exercise loss_tt
    sol = cycle.solve_cycle(**p)
    assert abs(cycle.conservation_residual(sol)) < 1e-9


def test_tt_channel_reduces_xmu_monotonically():
    """X_mu strictly decreases as the tt loss (omega_tt * lambda_tt) grows."""
    rt = load_rates()
    base = cycle.params_from_conditions(rt, 300.0, 1.2, 0.5)
    xs = [float(cycle.fusions_per_muon_ode(**dict(base, lambda_tt=lt, omega_tt=0.14))) for lt in
          (0.0, 2.0e6, 5.0e6, 1.0e7)]
    assert all(b < a for a, b in zip(xs, xs[1:], strict=False)), xs


def test_he_channel_absorbs_from_dmu():
    """loss_he > 0 iff c_he > 0, and X_mu decreases as c_he grows (He-3 scavenging out of the dmu pool).

    Uses the real ledger ``lambda_dhe3`` (Fotev et al., arXiv:2001.09927), so this also checks the
    ledger value flows through params_from_conditions -> the He channel."""
    rt = load_rates()
    xs = []
    for c_he in (0.0, 0.1, 0.3):
        p = cycle.params_from_conditions(rt, 300.0, 1.2, 0.5, c_he=c_he, include_loss_channels=True)
        sol = cycle.solve_cycle(**p)
        xs.append(float(sol.ys[-1, 3]))
        assert (float(sol.ys[-1, 7]) > 0.0) == (c_he > 0.0)
    assert xs[0] > xs[1] > xs[2], xs


def test_channels_off_is_default():
    """params_from_conditions default (channels OFF) reproduces the recorded v1 quickstart to rel 1e-9."""
    rt = load_rates()
    x = float(cycle.fusions_per_muon_from_conditions(rt, 300.0, 1.2, 0.5))
    assert abs(x - 114.47527542334024) / 114.47527542334024 < 1e-9, x
    p = cycle.params_from_conditions(rt, 300.0, 1.2, 0.5)
    assert p["lambda_tt"] == 0.0 and p["omega_tt"] == 0.0 and p["lambda_he"] == 0.0


def test_analytic_v2_matches_ode_with_channels():
    """G-N2: closed-form v2 matches the single-pool ODE (tt channel ON) to <1% over a 3-point grid."""
    lam0, lc, ose = 4.552e5, 1.2e8, 0.0055
    for lam_tt, w_tt in ((1.0e6, 0.14), (5.0e6, 0.14), (2.0e7, 0.5)):
        v2 = A.fusions_per_muon_v2(ose, lc, lambda_0=lam0, tt_loss_rate=lam_tt, omega_tt=w_tt)
        y0 = jnp.array([0.0, 0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])  # single-pool limit, start in tmu
        x_ode = float(
            cycle.solve_cycle(lam0, 0.0, 1e9, lc, lc, ose, y0=y0, lambda_tt=lam_tt, omega_tt=w_tt).ys[-1, 3]
        )
        assert abs(v2 - x_ode) / x_ode < 0.01, (lam_tt, w_tt, v2, x_ode)


def test_analytic_v2_reduces_to_v1():
    """channels at 0 -> v2 is IDENTICAL to fusions_per_muon (tt_loss_rate=0 OR omega_tt=0)."""
    for ose, lc in ((0.0045, 1.0e8), (0.0057, 1.45e8)):
        assert A.fusions_per_muon_v2(ose, lc) == A.fusions_per_muon(ose, lc)
        assert A.fusions_per_muon_v2(ose, lc, tt_loss_rate=1.0e7, omega_tt=0.0) == A.fusions_per_muon(ose, lc)


def test_recapture_off_bit_exact():
    """d-recapture OFF (f_d=0.0) reproduces the committed v1 reference to atol 1e-9 (reduction gate
    G-N1): with f_d=0.0 the new routing terms are IEEE-exact identities (x + 0.0*r == x, 1.0*r == r),
    so the locked step sequence does NOT drift. Extends the reference gate with the new args signature
    (explicit f_d=0.0), and confirms the explicit-f_d=0.0 path equals the default path bit-for-bit."""
    a6 = _REF["args6"]
    ts = _REF["ts"]
    yref = np.array(_REF["ys"])
    worst = 0.0
    for i in range(0, len(ts), 10):
        y = np.asarray(cycle.solve_cycle(*a6, f_d=0.0, t1=float(ts[i])).ys[-1])[:6]
        worst = max(worst, float(np.max(np.abs(y - yref[i]))))
        assert np.max(np.abs(y - yref[i])) < 1e-9, (i, float(np.max(np.abs(y - yref[i]))))
    x_default = float(cycle.solve_cycle(*a6).ys[-1, 3])
    x_explicit0 = float(cycle.solve_cycle(*a6, f_d=0.0).ys[-1, 3])
    assert x_default == x_explicit0, (x_default, x_explicit0)
    print(f"\nrecapture-off reduction gate: worst |got - ref| = {worst:.3e} (atol 1e-9); "
          f"N_fus default == f_d=0.0 explicit: {x_default!r}")


def test_recapture_bracket_sign():
    """q_1s recapture lowers X_mu at the canonical point (the freed muon detours through the dmu pool
    and races decay one extra transfer). The drop magnitude is O(10%) -- reported, not asserted tightly
    (wide band [2%, 20%])."""
    rt = load_rates()
    x_off = float(cycle.fusions_per_muon_from_conditions(rt, 300.0, 1.2, 0.5))
    x_on = float(cycle.fusions_per_muon_from_conditions(rt, 300.0, 1.2, 0.5, q_1s=1.0))
    drop = (x_off - x_on) / x_off
    assert x_on < x_off, (x_off, x_on)
    assert 0.02 <= drop <= 0.20, drop
    print(f"\nq_1s=1.0 recapture bracket (canonical): X_off={x_off:.4f} X_on={x_on:.4f} "
          f"drop={drop:.4f} (band [0.02, 0.20])")
