"""openmucf.formation -- resonant dt-mu (Vesman) formation rate lambda_dtmu(T, phi, F).

A physically-grounded, data-anchored *resonance-averaged* model (v1):

  energy-resolved:   lambda_dtmu_energy(E, F) = sum of Vesman resonances (Gaussian peaks)
  thermal:           lambda_dtmu(T, phi, F)  = eta * phi * < lambda_dtmu_energy >_Maxwell(T)

Anchored to measured data (provenance in openmucf/data/references.bib):
  * the beam resonance peak lambda_dtmu = (7.1 +- 1.8)e9 s^-1 at E = 0.423 eV, tmu F=1 (Fujiwara 2000);
  * thermal-scale formation ~1e8 s^-1 near room temperature and rising with T to ~800 K
    (Yamashita-Kino 2022), which the Maxwellian average reproduces because kT << the 0.42 eV
    resonance over 20-800 K, so cooling/heating moves along the resonance flank;
  * eta = epithermal enhancement (ledger row eta_dtmu): 1 (bare theory) .. 5 (Yamashita-Kino fit).

NOT the full Faifman/Adamczak quantum grid nor the condensed-matter S(q,omega) rate -- those are the
documented upgrade path (Phase 3). This model is calibrated so the cycle reproduces the canonical
lambda_c(T) band; the F=0 amplitudes are set from the room-temperature anchor (see _CALIB below).
"""

from __future__ import annotations

import jax.numpy as jnp

K_B = 8.617333e-5  # eV / K

# Vesman resonances per hyperfine F: (E_res [eV], amplitude [s^-1], width [eV]).
# F=1 carries the measured 0.423 eV beam resonance (amplitude = the measured peak 7.1e9).
# F=0 (lower hyperfine) carries near-threshold resonances that drive low-temperature formation;
# its amplitudes are scaled by _CALIB["F0"] to hit the room-temperature thermal anchor.
_RESONANCES = {
    0: [(0.050, 5.0e8, 0.040), (0.250, 1.5e9, 0.070)],
    1: [(0.423, 7.1e9, 0.036), (0.080, 3.0e8, 0.045)],
}
# Thermal-scale calibration: set so lambda_dtmu(300 K, F=0, phi=1) ~ 1.3e8 s^-1 -- i.e. the ~1e8
# room-temperature anchor (Yamashita-Kino 2022) and the canonical lambda_c(300 K) ~ 1.4e8. This scales
# ONLY the thermal average; the energy-resolved peak (lambda_dtmu_energy) stays the measured 7.1e9.
#
# _CALIB re-anchor (2026-07-08): the quadrature grid changed from linear to geometric (_EGRID below;
# for low-T convergence). _CALIB is the pre-existing DISCLOSED anchor whose defining condition is
# "match the 300 K thermal scale" (above), so when the grid change altered the quadrature the two
# factors were RE-DERIVED to reproduce the 300 K anchor rates bit-exactly. This honors the anchor's
# own definition; it is NOT tuning to a target. Recorded 300 K anchors on the OLD linear grid (before
# the change), to full precision:
#     a0 = lambda_dtmu(300, 1, F=0) = 130059577.61950417
#     a1 = lambda_dtmu(300, 1, F=1) = 53094361.62224904
# Re-derivation: new _CALIB[F] = a_F / _maxwell_avg(300, F)|geomspace  (old _CALIB was 0.31/0.31 on the
# linear grid). Verified: lambda_dtmu(300, 1, F) reproduces a0/a1 to relative 0.0 (bit-exact), so every
# 300 K downstream number is preserved; T != 300 K values shift slightly (the intended better quadrature).
_CALIB = {"F0": 0.30922830500330367, "F1": 0.3094896596177261}

# Geometric (log-spaced) quadrature grid for the Maxwell average: a linear grid put only ~4 points below
# 0.01 eV, where the Maxwell weight concentrates for T <~ 50 K, so low-T formation was grid-unconverged
# (changed by >7% under a grid doubling); the geomspace grid is converged to <0.05% there.
_EGRID = jnp.geomspace(1.0e-4, 2.0, 800)  # eV


def lambda_dtmu_energy(E, F=1):
    """Energy-resolved resonant formation rate [s^-1] (sum of Vesman resonances). Autodiff-friendly."""
    total = 0.0
    for E_r, A, w in _RESONANCES[F]:
        total = total + A * jnp.exp(-0.5 * ((E - E_r) / w) ** 2)
    return total


def _maxwell_pdf(E, T):
    kT = K_B * T
    return 2.0 * jnp.sqrt(E / jnp.pi) * kT ** (-1.5) * jnp.exp(-E / kT)


def _trapz(y, x):
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))


def _maxwell_avg(T, F):
    lam = lambda_dtmu_energy(_EGRID, F)
    f = _maxwell_pdf(_EGRID, T)
    return _trapz(lam * f, _EGRID)


def lambda_dtmu(T, phi=1.0, F=0, eta=1.0):
    """Thermally-averaged resonant dt-mu formation rate [s^-1] at T [K], density phi, hyperfine F.

    ``eta`` is the epithermal enhancement (ledger row eta_dtmu; 1 = bare theory, ~5 = Yamashita-Kino fit).
    """
    return eta * phi * _CALIB["F0" if F == 0 else "F1"] * _maxwell_avg(T, F)
