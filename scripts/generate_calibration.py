"""Run the Bayesian calibration, write CALIBRATION.md + figure. Run from repo root:

python scripts/generate_calibration.py
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from openmucf import calibrate  # noqa: E402

os.makedirs("figures", exist_ok=True)

weak = calibrate.run_mcmc(num_warmup=1000, num_samples=4000, seed=0)
kam = calibrate.run_mcmc(num_warmup=1000, num_samples=4000, seed=0, omega_s0_prior=("normal", 0.857, 0.03))
sw = calibrate.summarize(weak)
sk = calibrate.summarize(kam)

# ---- figure: the identifiability ridge + marginals ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
os0 = np.asarray(weak["omega_s0_pct"])
R = np.asarray(weak["R"])
ax[0].hexbin(os0, R, gridsize=40, cmap="viridis")
ax[0].set_xlabel(r"initial sticking $\omega_s^0$ [%]")
ax[0].set_ylabel(r"reactivation $R$")
ax[0].set_title(f"Degeneracy ridge (corr={sw['corr_omega_s0_R']:.2f})")
ax[1].hist(
    np.asarray(weak["omega_s_eff_pct"]),
    bins=60,
    density=True,
    alpha=0.8,
    color="#3388aa",
    label=r"$\omega_s^{eff}$ (constrained)",
)
ax[1].hist(R, bins=60, density=True, alpha=0.5, color="#aa6633", label="R (poorly constrained)")
ax[1].axvline(0.45, color="k", ls="--", lw=1, label="measured $\\omega_s^{eff}$=0.45%")
ax[1].set_xlabel("value")
ax[1].set_title("marginals: product pinned, split is not")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/calibration.png", dpi=140)
plt.close(fig)


def _row(name, s):
    v = s[name]
    return f"| {name} | {v['mean']:.3g} | {v['sd']:.3g} | [{v['lo']:.3g}, {v['hi']:.3g}] |"


md = f"""# CALIBRATION.md -- Bayesian calibration to experiment (auto-generated)

Calibrated (omega_s0, R, lambda_c) to Petitjean/Breunlich: omega_s_eff = 0.45+-0.05 %, X_mu = 113+-12,
via numpyro NUTS. See `openmucf/calibrate.py`.

## Weak omega_s0 prior (experimental data only) -- exposes the degeneracy
| parameter | mean | sd | 95% CI |
|---|---|---|---|
{_row("omega_s_eff_pct", sw)}
{_row("lambda_c", sw)}
{_row("omega_s0_pct", sw)}
{_row("R", sw)}

**omega_s0 - R correlation = {sw["corr_omega_s0_R"]:.2f}.** The effective sticking (product) and lambda_c
are well constrained, but omega_s0 and R are strongly correlated (a positive ridge along fixed
omega_s0*(1-R)): the yield/sticking data pin the product, NOT the split. (Figure `figures/calibration.png`.)

## Informative omega_s0 prior (Kamimura 0.857+-0.03 %) -- partially resolves it
| parameter | mean | sd | 95% CI |
|---|---|---|---|
{_row("omega_s0_pct", sk)}
{_row("R", sk)}
{_row("omega_s_eff_pct", sk)}

The Kamimura *theory* input tightens omega_s0 (sd {sw["omega_s0_pct"]["sd"]:.3g} -> {sk["omega_s0_pct"]["sd"]:.3g} %)
and hence R -- but R still inherits that uncertainty.

## Finding
Experiment alone determines **effective sticking and the cycling rate**, not the microscopic
sticking/reactivation split. Separating omega_s0 from R -- and predicting how R changes at high density --
requires an independent microscopic calculation. **That is exactly the Phase-3 reactivation surrogate,**
and this degeneracy is the quantitative reason it is needed.
"""

with open("CALIBRATION.md", "w") as f:
    f.write(md)

print("wrote CALIBRATION.md and figures/calibration.png")
print(
    f"[weak prior]  omega_s_eff = {sw['omega_s_eff_pct']['mean']:.3f} +- {sw['omega_s_eff_pct']['sd']:.3f} %"
    f" | lambda_c = {sw['lambda_c']['mean']:.3g} | corr(omega_s0,R) = {sw['corr_omega_s0_R']:.2f}"
)
print(f"[weak prior]  omega_s0 sd = {sw['omega_s0_pct']['sd']:.3f}, R sd = {sw['R']['sd']:.3f}")
print(f"[Kamimura]    omega_s0 sd = {sk['omega_s0_pct']['sd']:.3f}, R sd = {sk['R']['sd']:.3f}")
