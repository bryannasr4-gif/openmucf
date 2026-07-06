"""OpenMuCF quickstart -- your first five minutes.

Run it:

    python examples/quickstart.py

It walks the v1 spine end to end: load the FAIR rate ledger, compute fusions-per-muon two ways
(closed form + differentiable ODE), read the honest energy ladder (scientific vs net-electrical
breakeven), and export GEANT4-ready rate tables. Every number is produced by the engine -- nothing
is hard-coded here.
"""

from openmucf import analytic, cycle, interop, load_rates
from openmucf.energy import EnergyChain

# 1. Load the validated FAIR rate ledger (provenance-enforced single source of truth).
rates = load_rates()
print(f"[1] ledger loaded: {len(rates.symbols())} rates "
      f"({len(rates.contested())} contested, {len(rates.needs_verification())} need verification)")

# 2. Fusions per muon at liquid-D/T-ish conditions, from the differentiable cycle ODE.
T, phi, c_t = 300.0, 1.2, 0.5
x_mu_ode = float(cycle.fusions_per_muon_from_conditions(rates, T=T, phi=phi, c_t=c_t))
print(f"[2] X_mu(T={T} K, phi={phi}, c_t={c_t}) = {x_mu_ode:.1f}  (full diffrax ODE network)")

# 3. Closed-form cross-check (gate V1: the ODE must reproduce this to < 1%).
os0 = rates.value("omega_s0") / 100.0
ose = analytic.effective_sticking(os0, rates.value("R_col"))
lam_c = analytic.cycling_rate(phi, 1.2e8)  # density-normalized cycling rate ~ ledger band
x_mu_analytic = float(analytic.fusions_per_muon(ose, lam_c))
print(f"[3] closed-form X_mu = {x_mu_analytic:.1f}  (omega_s^eff = {ose*100:.3f}%)")

# 4. The honest energy ladder -- scientific vs net-electrical breakeven.
chain = EnergyChain()
print(f"[4] record X_mu ~150 | scientific breakeven {chain.breakeven_xmu_sci():.0f} "
      f"| net-electrical breakeven {chain.breakeven_xmu_net():.0f}")
print(f"    Q_sci(150) = {chain.Q_sci(150):.3f}   Q_net(150) = {chain.Q_net_electrical(150):.3f}")

# 5. Export GEANT4-ready rate tables (interop stub -- complement, never compete).
written = interop.export_all(rates, "quickstart_export", fmt="both")
print(f"[5] exported rate tables -> {sorted(written)} (CSV+JSON in ./quickstart_export/)")

print("\nNext: `make findings` reproduces the sensitivity/breakeven headline; see docs/ and FINDINGS.md.")
