"""Generate FRONTIER.md + figures/frontier.png + FRONTIER_MANIFEST.json.

    python scripts/generate_frontier.py

Content (WAVE2_EXECUTION_SPEC sec.3, WS-Q): the inverse-design mode -- "what would have to be true".
The (lambda_c, R) reactivation frontier for X_mu in {150, 284, 500} (closed-form), the density-independent
R floor that reproduces the FINDINGS.md sec.3 "R >= 0.77" headline, and a solver-backed general inverse
over the differentiable systems.q_net graph.

Framing (I1/I8): INVERSE-DESIGN ONLY, never a verdict. It states requirements exactly as FINDINGS.md
sec.3 does; it encodes no scenario/verdict registry (fenced OUT, sec.3.2).

Byte-stability (WAVE2 sec.0-A A2): FRONTIER.md + FRONTIER_MANIFEST.json ship ONLY closed-form float64
numbers (byte-stable cross-arch, like SYSTEMS.md); the solver-backed inverses are cross-checked against
those closed forms to < 1e-9 (relative) in the tests, and the shipped worked-example digits are the
closed-form values quantised to 6 significant figures -- so nothing byte-diffed depends on iterative-solver
noise. The PNG (which draws the Kamimura-prior MCMC posterior cloud) is NEVER byte-diffed.

Audit wiring: regenerate all three; git-diff ONLY FRONTIER.md + FRONTIER_MANIFEST.json; append the manifest
to `provenance --check`. Computation lives in importable helpers (no import side effects); the MCMC figure
+ file I/O + printing are guarded behind main() so tests import and assert on the tables without regen.
"""

from __future__ import annotations

import math
from pathlib import Path

from openmucf import frontier
from openmucf.analytic import effective_sticking, fusions_per_muon
from openmucf.constants import LAMBDA_0
from openmucf.frontier import (
    LAMBDA_C_NOMINAL,
    OMEGA_S0_DEFAULT,
    R_NOMINAL,
    lambda_c_required,
    r_required,
)
from openmucf.provenance import ManifestEntry, file_sha256, write_manifest
from openmucf.rates import RATES_CSV
from openmucf.systems import SystemChain

# The frontier grid (WS-Q): X_mu targets and the cycling-rate columns. 1.45e8 = measured liquid max;
# 2.28e8 / 3.00e8 are the FINDINGS sec.3 optimistic-density anchors; "inf" is the decay-free R floor.
X_MU_TARGETS = [150.0, 284.0, 500.0]
LAMBDA_C_GRID = [1.00e8, 1.45e8, 2.28e8, 3.00e8, math.inf]

# Measured liquid-condition cycling band (uq_priors.csv lambda_c row); shaded in the figure.
LAMBDA_C_BAND = (1.00e8, 1.45e8)

# Reference reactivation markers (FINDINGS.md sec.3): the Kou-Chen Eq.33 collisional value and the
# Kamimura-prior calibration posterior MEAN. Literals here (byte-stable), matching FINDINGS.md; the figure
# draws the live posterior cloud whose mean sits at this R.
R_COL_REFERENCE = "0.35"
R_KAMIMURA_REFERENCE = "0.46"
R_KAMIMURA_SD = "0.06"

# Solver worked-example target: a modest q_net (above the nominal operating point) chosen so ALL FOUR free
# variables have a finite, feasible required value -- a clean cross-check of the four inverses at once.
SOLVER_TARGET_QNET = 0.06


def _sig6(x: float) -> str:
    """6 significant figures (the WAVE2 A2 solver-print precision). Applied to closed-form values here."""
    return f"{x:.6g}"


def _lc_label(lc: float) -> str:
    return "inf" if math.isinf(lc) else f"{lc:.2e}"


def build_headline() -> dict[str, str]:
    """Single source of truth: every formatted string shared by FRONTIER.md and the manifest.

    All values are closed-form (pure-Python float64) -> byte-stable. Nothing here runs a solver or MCMC.
    """
    H: dict[str, str] = {}
    sc = SystemChain()

    # --- the (lambda_c, R_required) frontier matrix (closed-form r_required) ---
    for X in X_MU_TARGETS:
        xi = f"{X:.0f}"
        for lc in LAMBDA_C_GRID:
            H[f"R_{xi}_{_lc_label(lc)}"] = f"{r_required(X, lc):.4f}"

    # --- the marquee density-independent R floor (reproduces FINDINGS sec.3 EXACTLY) ---
    H["R_floor_500"] = f"{r_required(500.0, math.inf):.2f}"  # 0.77
    H["R_col"] = R_COL_REFERENCE
    H["R_kamimura"] = R_KAMIMURA_REFERENCE
    H["R_kamimura_sd"] = R_KAMIMURA_SD

    # --- nominal operating point (closed-form forward map) ---
    x_nom = float(fusions_per_muon(effective_sticking(OMEGA_S0_DEFAULT, R_NOMINAL), LAMBDA_C_NOMINAL))
    q_nom = x_nom * sc.E_per_fusion_MeV * sc._net_efficiency_factor() / sc.E_mu_MeV
    H["x_mu_nominal"] = _sig6(x_nom)
    H["q_net_nominal"] = _sig6(q_nom)
    H["sticking_floor_xmu"] = _sig6(1.0 / effective_sticking(OMEGA_S0_DEFAULT, R_NOMINAL))

    # --- solver worked example: closed-form required value per free var at SOLVER_TARGET_QNET ---
    netf = sc._net_efficiency_factor()
    x_tgt = SOLVER_TARGET_QNET * sc.E_mu_MeV / (sc.E_per_fusion_MeV * netf)
    H["solver_target"] = f"{SOLVER_TARGET_QNET:.2f}"
    H["solver_x_target"] = _sig6(x_tgt)
    cf = {
        "E_mu_GeV": x_nom * sc.E_per_fusion_MeV * netf / SOLVER_TARGET_QNET / 1.0e3,
        "eta_acc": SOLVER_TARGET_QNET * sc.E_mu_MeV
        / (x_nom * sc.E_per_fusion_MeV * sc.blanket_M * sc.eta_thermal * (1.0 - sc.recirc_fraction)),
        "R": r_required(x_tgt, LAMBDA_C_NOMINAL),
        "lambda_c": lambda_c_required(x_tgt, R_NOMINAL),
    }
    for k, v in cf.items():
        H[f"solver_{k}"] = _sig6(v)
    return H


def _frontier_table_md(H: dict[str, str]) -> str:
    cols = " | ".join(_lc_label(lc) for lc in LAMBDA_C_GRID)
    head = (
        f"| X_mu (target) \\ lambda_c [s^-1] | {cols} |\n"
        "|" + "---|" * (len(LAMBDA_C_GRID) + 1) + "\n"
    )
    rows = []
    for X in X_MU_TARGETS:
        xi = f"{X:.0f}"
        cells = " | ".join(H[f"R_{xi}_{_lc_label(lc)}"] for lc in LAMBDA_C_GRID)
        rows.append(f"| {xi} | {cells} |")
    return head + "\n".join(rows)


def build_markdown(H: dict[str, str]) -> str:
    table = _frontier_table_md(H)
    return f"""# FRONTIER.md -- inverse-design frontiers: "what would have to be true" \
(auto-generated by `scripts/generate_frontier.py`)

> **INVERSE-DESIGN, NOT A VERDICT (I1/I8).** This is the requirement form of the same forward physics used
> everywhere else in OpenMuCF (`analytic` yield map + `systems` energy graph) -- no new physics. Given a
> target it reports the *required* value of one parameter, exactly as FINDINGS.md sec.3 states "R >= 0.77
> is required". It renders **no verdict** on any external projection and encodes **no** scenario/verdict
> registry (deliberately fenced out). A required value outside [0, 1] (for R or eta_acc) or at `inf` (for
> lambda_c) is the honest readout that the target is unreachable in that variable -- reported, not clamped.

## The (lambda_c, R) reactivation frontier (closed-form)
For each target yield X_mu, the reactivation `R` that would be required at a given cycling rate `lambda_c`,
from the identity `R = 1 - (1/X_mu - lambda_0/lambda_c) / omega_s0` (omega_s0 = {OMEGA_S0_DEFAULT} bare
fraction, the Kamimura nominal; lambda_0 = {LAMBDA_0:.0f} s^-1, the muon decay rate). The `inf` column is
the decay-free limit -- the R floor no density can beat.

{table}

`R` falls monotonically as `lambda_c` rises along every row (a faster cycle lets muon decay carry more of
the loss budget, so less reactivation is required): `d R_required / d lambda_c < 0`. Cells with `R > 1` are
the honest readout that the target is **unreachable in reactivation alone** at that cycling rate (no
physical R exceeds 1). Every cell is closed-form float64 algebra -- byte-stable, printed to 4 decimals.

## The density-independent R floor (the marquee requirement)
At X_mu = 500 and `lambda_c -> inf`, the required reactivation is **R >= {H["R_floor_500"]}** -- the
decay-free floor, reproduced here bit-for-bit from `openmucf.uq.breakeven_audit` and matching the FINDINGS.md
sec.3 headline. Density scaling (`lambda_c = phi * lambda_c_tilde`) can supply the cycling-rate factor, but
**sticking it cannot**: the floor is independent of `lambda_c`. For reference, the model-derived collisional
reactivation is R_col = **{H["R_col"]}** (Kou-Chen Eq.33), and the Kamimura-prior calibration posterior
gives R = **{H["R_kamimura"]}** +- {H["R_kamimura_sd"]} (`CALIBRATION.md`); both sit far below the
{H["R_floor_500"]} floor. Expressed as a requirement, an X_mu = 500 mechanism must push reactivation to
R ~ 0.9+ at any density -- a falsifiable, quantitative bet on the reactivation physics.

## Solver-backed general inverse (optimistix Newton over the q_net graph)
`openmucf.frontier.solve_inverse` inverts the full differentiable `systems.q_net` graph for ONE free
variable among {list(frontier.FREE_VARS)} at a `q_net` target, with the others fixed -- an
`optimistix.root_find` (Newton) capability that generalises beyond the closed forms. At the nominal
operating point (omega_s0 = {OMEGA_S0_DEFAULT}, R = {R_NOMINAL}, lambda_c = {LAMBDA_C_NOMINAL:.2e} s^-1 =>
X_mu = {H["x_mu_nominal"]}, q_net = {H["q_net_nominal"]}), lifting the net-electrical gain to
q_net = {H["solver_target"]} (X_mu-equivalent {H["solver_x_target"]}) requires, in each single lever:

| free variable | required value | direction from nominal |
|---|---|---|
| E_mu_GeV | {H["solver_E_mu_GeV"]} | cheaper muons (from 5.0 GeV) |
| eta_acc | {H["solver_eta_acc"]} | higher wall-plug (from 0.30) |
| R | {H["solver_R"]} | more reactivation (from {R_NOMINAL}) |
| lambda_c | {H["solver_lambda_c"]} | faster cycling (from {LAMBDA_C_NOMINAL:.2e}) |

Values are closed-form and printed at 6 significant figures; the Newton solver over `q_net` reproduces each
to < 1e-9 (relative), and `q_net(inverse solution) == target` to < 1e-12 -- the consistency gate, locked by
`tests/test_frontier.py`. Because this model is analytically invertible, the solver never carries a shipped
digit on its own; it is the general capability, gate-checked against the exact algebra.

## Figure
`figures/frontier.png` -- the (lambda_c, R) frontier curves for X_mu in {{150, 284, 500}} with the measured
lambda_c band shaded, the **Kamimura-prior calibration posterior cloud** (numpyro NUTS on the
Petitjean/Breunlich anchors with the Kamimura omega_s0 = 0.857 +- 0.03 % prior; its mean IS the R =
{H["R_kamimura"]} marker), and the R_col = {H["R_col"]} / R = {H["R_kamimura"]} reference markers overlaid --
the "what would have to be true" picture. (Regenerated every audit; never byte-diffed.)
"""


def build_manifest_entries(H: dict[str, str]) -> list[ManifestEntry]:
    import re

    def _entry(entry_id, pattern):
        return ManifestEntry(
            id=entry_id, value=H[entry_id], pattern=pattern,
            source_type="derivation", source="scripts/generate_frontier.py", doc="FRONTIER.md",
        )

    entries = [
        _entry("R_floor_500", rf"\*\*R >= {re.escape(H['R_floor_500'])}\*\*"),
        _entry("R_col", rf"R_col = \*\*{re.escape(H['R_col'])}\*\*"),
        _entry("R_kamimura", rf"posterior\s+gives R = \*\*{re.escape(H['R_kamimura'])}\*\*"),
        _entry("x_mu_nominal", rf"X_mu = {re.escape(H['x_mu_nominal'])},"),
        _entry("q_net_nominal", rf"q_net = {re.escape(H['q_net_nominal'])}\)"),
        _entry("solver_x_target", rf"X_mu-equivalent {re.escape(H['solver_x_target'])}\)"),
        _entry("solver_E_mu_GeV", rf"\| E_mu_GeV \| {re.escape(H['solver_E_mu_GeV'])} \|"),
        _entry("solver_eta_acc", rf"\| eta_acc \| {re.escape(H['solver_eta_acc'])} \|"),
        _entry("solver_R", rf"\| R \| {re.escape(H['solver_R'])} \|"),
        _entry("solver_lambda_c", rf"\| lambda_c \| {re.escape(H['solver_lambda_c'])} \|"),
    ]
    # anchor the FINDINGS sec.3 frontier cells (X=500 at 2.28e8 -> ~R->1, at 3.00e8 -> 0.94, at inf -> 0.77)
    for X, lc in ((500.0, 2.28e8), (500.0, 3.00e8), (500.0, math.inf), (150.0, 1.45e8)):
        xi, lcl = f"{X:.0f}", _lc_label(lc)
        key = f"R_{xi}_{lcl}"
        entries.append(
            ManifestEntry(
                id=key, value=H[key], pattern=rf"\| {xi} \|.*{re.escape(H[key])}",
                source_type="derivation", source="scripts/generate_frontier.py", doc="FRONTIER.md",
            )
        )
    return entries


def _write_figure(H: dict[str, str]) -> None:
    """Draw figures/frontier.png: closed-form frontier curves + the Kamimura-prior MCMC posterior cloud.

    Runs the numpyro NUTS Kamimura chain (never byte-diffed; MCMC is not cross-arch bit-stable). Imported
    lazily so importing this module (for the closed-form tables) stays free of matplotlib/MCMC.
    """
    import os

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from openmucf import calibrate

    kam = calibrate.run_mcmc(
        num_warmup=1000, num_samples=4000, seed=0, omega_s0_prior=("normal", 0.857, 0.03)
    )
    lc_post = np.asarray(kam["lambda_c"])
    R_post = np.asarray(kam["R"])

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    lc_curve = np.linspace(0.9e8, 3.2e8, 400)
    for X in X_MU_TARGETS:
        Rv = np.array([r_required(X, float(lc)) for lc in lc_curve])
        Rv[(Rv < 0.0) | (Rv > 1.05)] = np.nan  # only the physical band
        ax.plot(lc_curve, Rv, lw=2.2, label=rf"$X_\mu={X:.0f}$ frontier")

    ax.axvspan(*LAMBDA_C_BAND, color="0.85", alpha=0.6, label="measured $\\lambda_c$ band (liquid)")
    ax.scatter(lc_post, R_post, s=4, alpha=0.18, color="#3366aa", label="Kamimura-prior posterior")
    ax.axhline(float(R_COL_REFERENCE), color="#aa6633", ls="--", lw=1.4,
               label=f"$R_{{col}}={R_COL_REFERENCE}$ (Kou-Chen)")
    ax.axhline(float(R_KAMIMURA_REFERENCE), color="#227722", ls=":", lw=1.6,
               label=f"$R={R_KAMIMURA_REFERENCE}$ (Kamimura mean)")

    ax.set_xlabel(r"cycling rate $\lambda_c$ [s$^{-1}$]")
    ax.set_ylabel(r"required reactivation $R$")
    ax.set_ylim(0.20, 1.05)
    ax.set_xlim(0.9e8, 3.2e8)
    ax.set_title(r'Inverse-design: "what would have to be true" for $X_\mu\in\{150,284,500\}$')
    ax.legend(fontsize=8, loc="upper right")
    fig.text(
        0.01, 0.005,
        "Frontier curves are closed-form; the posterior cloud is the numpyro NUTS Kamimura-prior "
        "calibration chain (omega_s0=0.857+-0.03%). Requirement form, not a verdict.",
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig("figures/frontier.png", dpi=140)
    plt.close(fig)


def main() -> None:
    H = build_headline()
    Path("FRONTIER.md").write_text(build_markdown(H), encoding="utf-8")
    entries = build_manifest_entries(H)
    inputs = {
        "rates_csv_sha256": file_sha256(RATES_CSV),
        "omega_s0_default_fraction": OMEGA_S0_DEFAULT,
        "lambda_0_s^-1": float(LAMBDA_0),
        "R_nominal": R_NOMINAL,
        "lambda_c_nominal_s^-1": LAMBDA_C_NOMINAL,
        "x_mu_targets": X_MU_TARGETS,
        "lambda_c_grid_s^-1": [None if math.isinf(lc) else lc for lc in LAMBDA_C_GRID],
        "solver_target_q_net": SOLVER_TARGET_QNET,
    }
    write_manifest("FRONTIER_MANIFEST.json", entries, inputs, generated_by="scripts/generate_frontier.py")
    _write_figure(H)
    print(f"wrote FRONTIER.md + FRONTIER_MANIFEST.json ({len(entries)} entries) + figures/frontier.png")
    print(f"R floor (X_mu=500, lambda_c->inf): R >= {H['R_floor_500']} (reproduces FINDINGS sec.3)")
    print(f"nominal operating point: X_mu={H['x_mu_nominal']} q_net={H['q_net_nominal']}")
    print(f"solver worked example @ q_net={H['solver_target']}: "
          f"E_mu={H['solver_E_mu_GeV']} eta_acc={H['solver_eta_acc']} "
          f"R={H['solver_R']} lambda_c={H['solver_lambda_c']}")


if __name__ == "__main__":
    main()
