"""Reproduction of the Kou-Chen cycle-closure criterion (arXiv:2607.10989, bibkey KouChenLawson2026).

**Which paper.** `KouChenLawson2026` is the Lawson-inspired cycle-closure criterion. It is NOT
`KouChen2026` (arXiv:2606.07077, the external-field reactivation rate network), which the older
`V_kouchen_base` / `V_kouchen_best` targets reproduce. Both are Kou & Chen 2026 and the two keys are
deliberately distinct; :func:`test_the_registered_bibkey_is_the_cycle_closure_paper` holds them apart.

**What this proves, stated so it cannot be read as more.** Every row here is a REPRODUCTION of the
published algebra of another group's paper, fed with that paper's own adopted inputs. Nothing in this
file is an independent prediction, and a PASS says only that our closed form and theirs are the same
map -- which is the point: it makes "we reproduce their Table I" a machine-checked claim rather than a
sentence. Their inputs are historical accounting choices, not measurements (their own words for the
3/5/8 GeV cost scale), so a PASS carries no endorsement of those choices.

**No second copy of their algebra.** Each quantity is computed from a closed form openmucf already
ships:

  their Eq.(2)  N_fus,mu = L_mu/(1 + omega_eff*L_mu)   ==  ``analytic.fusions_per_muon``, whose
                X_mu = 1/(omega_s_eff + lambda_0/lambda_c) is the same map under L_mu = lambda_c/lambda_0
  their Eq.(3)  the inverse projection                 ==  the exact inverse of that closed form; taken
                by the same inversion ``openmucf.validate`` already uses for V_breunlich_lambdac, and
                round-tripped back through ``fusions_per_muon`` below
  their Eq.(9)  N_L = E_cost/(eta_sys*E_use)           ==  ``analytic.breakeven_xmu``
  their Eq.(8)  G_mu = (eta_sys*E_use/E_cost)*N_fus,mu ==  ``analytic.energy_gain``
  their Eq.(12) omega_crit = 1/(G_mu*N_L)              ==  the reciprocal of that demand, which is the
                asymptotic ceiling of ``fusions_per_muon`` itself (X_mu -> 1/omega_s_eff as
                lambda_c -> infinity); asserted as that ceiling rather than restated

**The one difference, disclosed not reconciled.** Eq.(9) and ``breakeven_xmu`` are the same function
of different adopted constants: they divide by a useful cycle energy E_use = 20.4 MeV that their paper
does not source, openmucf divides by the ledger's E_fusion = 17.6 MeV. Reproducing their table means
feeding their divisor; our own default breakeven is a different number and stays one
(:func:`test_their_useful_cycle_energy_is_not_the_ledger_fusion_energy`).

Targets, tolerances and the per-row rationale live in ``openmucf/data/validation_targets.csv`` under
the ``V_kouchenlawson_`` prefix; the verdict semantics are ``openmucf.validate``'s own, so this gate
and the trust gate cannot drift apart on what "within tolerance" means.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from openmucf import formation, validate
from openmucf.analytic import breakeven_xmu, energy_gain, fusions_per_muon
from openmucf.rates import TARGETS_CSV, load_rates

#: Row-id prefix of every target this file owns.
PREFIX = "V_kouchenlawson_"

#: Their adopted accounting inputs (Table I caption and Sec. IV). Not ours, not measured: carried as
#: literals so that feeding them is visible, and cross-checked against each row's declared
#: ``conditions`` by :func:`test_declared_conditions_match_the_inputs_fed`.
E_USE_MEV = 20.4  # "useful D-T cycle energy", unsourced in the paper
ETA_SYS = 1.0  # their nominal system factor
ETA_SYS_CONSERVATIVE = 0.4  # their conservative variant
E_COST_GEV_GAIN = 5.0  # the cost convention their Table I gain column fixes
G_MU_BOUNDARY = 1.0  # the target gain their Sec. IV boundaries are quoted at

#: Table I's four anchors: (adopted yield Y_f, adopted effective sticking as a fraction).
ANCHOR_INPUTS = {
    "sin": (124.0, 0.0057),
    "lampf": (150.0, 0.0045),
    "petitjean_lo": (100.0, 0.0050),
    "petitjean_hi": (150.0, 0.0050),
}

#: (E_cost [GeV], eta_sys) for the four Sec. IV accounting cases.
COST_CASES = {
    "3GeV": (3.0, ETA_SYS),
    "5GeV": (5.0, ETA_SYS),
    "8GeV": (8.0, ETA_SYS),
    "5GeV_eta040": (5.0, ETA_SYS_CONSERVATIVE),
}


def _cycle_strength_from_yield(y_f: float, omega: float, lambda_0: float) -> float:
    """L_mu implied by a yield, by INVERTING our closed form -- their Eq.(3), never re-derived.

    ``openmucf.validate`` already inverts ``fusions_per_muon`` this way for V_breunlich_lambdac:
    ``lambda_c = lambda_0/(1/X_mu - omega_s_eff)``. Dividing that cycling rate by ``lambda_0`` is the
    dimensionless cycle strength, so the projection is our own inversion read in their variables.
    """
    lambda_c = lambda_0 / (1.0 / y_f - omega)
    return lambda_c / lambda_0


def _predictions(rates) -> dict[str, float]:
    """Every registered row's predicted value, keyed by target id, from openmucf's closed forms."""
    lambda_0 = rates.value("lambda_mu_decay")
    out: dict[str, float] = {}

    # L_mu. The SIN/Crowe row is their Lambda_c*tau_mu projection, and both of its inputs are
    # rows of OUR ledger from the same primary they cite, so it is computed from the ledger
    # rather than from their table; the other three are the Eq.(3) inversion of our closed form.
    out[PREFIX + "Lmu_sin"] = rates.value("lambda_c_solid_12K") / lambda_0
    for anchor in ("lampf", "petitjean_lo", "petitjean_hi"):
        y_f, omega = ANCHOR_INPUTS[anchor]
        out[PREFIX + "Lmu_" + anchor] = _cycle_strength_from_yield(y_f, omega, lambda_0)

    # N_L (their Eq.(9)) and the no-go boundary (their Eq.(12)) for the four accounting cases.
    for case, (e_cost, eta_sys) in COST_CASES.items():
        n_l = float(breakeven_xmu(E_f_MeV=E_USE_MEV, E_mu_GeV=e_cost, eta_conv=eta_sys))
        out[PREFIX + "NL_" + case] = n_l
        out[PREFIX + "omegacrit_" + case] = 100.0 / (G_MU_BOUNDARY * n_l)

    # G_mu (their Eq.(8)) at the 5 GeV convention, from each anchor's yield through our closed form.
    # The SIN/Crowe sticking is taken from the ledger too, so that row's gain rests on the ledger's
    # condition-paired Crowe1987 rows end to end rather than on one ledger value and one literal;
    # test_sin_projection_is_computed_from_our_own_ledger pins that pair to Table I's declared 0.57%.
    for anchor in ANCHOR_INPUTS:
        omega = ANCHOR_INPUTS[anchor][1]
        if anchor == "sin":
            omega = rates.value("omega_s_eff_solid_12K") / 100.0
        lambda_c = out[PREFIX + "Lmu_" + anchor] * lambda_0
        n_fus = float(fusions_per_muon(omega, lambda_c, lambda_0))
        out[PREFIX + "Gmu_" + anchor] = float(
            energy_gain(n_fus, ETA_SYS, E_f_MeV=E_USE_MEV, E_mu_GeV=E_COST_GEV_GAIN)
        )
    return out


def _registered_rows() -> dict[str, dict[str, str]]:
    return {k: v for k, v in validate._load_targets().items() if k.startswith(PREFIX)}


def _printed_ulp(printed: str) -> float:
    """One unit in the last printed place of a value as the paper printed it ('4.24e2' -> 1.0)."""
    mantissa, _, exponent = printed.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return 10.0 ** (-decimals + (int(exponent) if exponent else 0))


# ---------------------------------------------------------------------------------------------
# The reproduction itself
# ---------------------------------------------------------------------------------------------


def test_every_registered_row_is_reproduced_within_its_tolerance():
    """The scoreboard: each registered row's published figure, against our closed form's value."""
    rows, pred = _registered_rows(), _predictions(load_rates())
    failures = []
    for tid, row in sorted(rows.items()):
        got, published, tol = pred[tid], float(row["value"]), row["tolerance"]
        if not validate._within(got, published, tol):
            failures.append(f"{tid}: ours {got!r} vs published {published!r}, tolerance {tol}")
    assert not failures, "registered Kou-Chen targets outside tolerance:\n  " + "\n  ".join(failures)


def test_every_registered_row_is_exercised():
    """No row may be registered in the CSV and then never run -- registration without a check is decor."""
    assert set(_registered_rows()) == set(_predictions(load_rates()))
    assert len(_registered_rows()) == 16  # 4 L_mu + 4 N_L + 4 omega_crit + 4 G_mu


def _mutated_targets(tmp_path, target: str, **fields: str) -> dict[str, dict[str, str]]:
    """The live CSV with one row's fields overwritten, loaded through the same loader."""
    with open(TARGETS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["target_id"].strip() == target:
            row.update(fields)
    path = tmp_path / "validation_targets_mutated.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return validate._load_targets(str(path))


def test_verdicts_are_csv_driven(tmp_path):
    """The gate reads the CSV, it does not restate it -- and BOTH columns are load-bearing.

    For an interval tolerance ``openmucf.validate._within`` tests membership of the band alone, so the
    band is what decides the verdict and the printed value is what the band must be built from. Moving
    either one alone is caught: the band here, the value by
    :func:`test_tolerance_bands_are_the_printed_precision`, which re-derives the band from it.
    """
    target = PREFIX + "NL_5GeV"
    got = _predictions(load_rates())[target]
    live = validate._load_targets()[target]
    assert validate._within(got, float(live["value"]), live["tolerance"]) is True

    # (i) move the band off our value -> FAIL.
    moved = _mutated_targets(tmp_path, target, tolerance="[900.45,901.55]")[target]
    assert validate._within(got, float(moved["value"]), moved["tolerance"]) is False

    # (ii) move the published value, keeping the band -> the band no longer follows the rule.
    bumped = _mutated_targets(tmp_path, target, value="999")[target]
    lo, hi = (float(x) for x in bumped["tolerance"].strip("[]").split(","))
    half = 0.55 * _printed_ulp(bumped["value"])
    assert (lo, hi) != pytest.approx((999.0 - half, 999.0 + half))


# ---------------------------------------------------------------------------------------------
# Why the numbers are theirs and the algebra is ours
# ---------------------------------------------------------------------------------------------


def test_inverse_projection_round_trips_through_our_closed_form():
    """Their Eq.(3) is the exact inverse of our ``fusions_per_muon``, not a second copy of it.

    Each anchor's projected cycle strength is pushed back through the FORWARD closed form; recovering
    the yield it was built from is what licenses calling the inversion ours rather than theirs.
    """
    lambda_0 = load_rates().value("lambda_mu_decay")
    for anchor in ("lampf", "petitjean_lo", "petitjean_hi"):
        y_f, omega = ANCHOR_INPUTS[anchor]
        cycle_strength = _cycle_strength_from_yield(y_f, omega, lambda_0)
        back = float(fusions_per_muon(omega, cycle_strength * lambda_0, lambda_0))
        assert back == pytest.approx(y_f, rel=1e-12), anchor


def test_sin_projection_is_computed_from_our_own_ledger():
    """The SIN/Crowe cycle strength comes from ledger rows, not from their table.

    ``lambda_c_solid_12K`` and ``omega_s_eff_solid_12K`` are the condition-paired Crowe1987 anchors
    openmucf already carries (see tests/test_anchor_pairs.py); their Table I quotes the same primary.
    So this row reproduces their projection out of our own data, which is the strongest claim in the
    block -- and the yield that pair implies is their quoted Y_f = 124 +- 10.
    """
    rates = load_rates()
    lambda_c = rates.value("lambda_c_solid_12K")
    omega = rates.value("omega_s_eff_solid_12K") / 100.0
    assert (lambda_c, omega * 100.0) == (1.93e8, pytest.approx(0.57))
    yield_implied = float(fusions_per_muon(omega, lambda_c, rates.value("lambda_mu_decay")))
    assert 114.0 <= yield_implied <= 134.0, yield_implied  # their quoted 124 +- 10

    # Their convention is Lambda_c*tau_mu; ours is lambda_c/lambda_0. The ledger's lambda_0 is a
    # 4-significant-figure PDG rounding, so the two differ far inside the printed precision.
    ours = _predictions(rates)[PREFIX + "Lmu_sin"]
    theirs = lambda_c * 2.1969811e-6  # tau_mu as the ledger's own conditions field records it
    assert abs(ours - theirs) / theirs < 1e-4


def test_omega_crit_is_the_asymptotic_ceiling_of_our_closed_form():
    """Their Eq.(12) boundary is where our own closed form's ceiling meets the demand.

    ``fusions_per_muon`` tends to 1/omega_s_eff as lambda_c grows, so a system whose sticking exceeds
    1/(G_mu*N_L) cannot reach the target at ANY finite cycle rate. Asserting that ceiling is what makes
    the registered omega_crit rows a property of our map rather than a restatement of their inequality.
    """
    rates = load_rates()
    lambda_0 = rates.value("lambda_mu_decay")
    pred = _predictions(rates)
    for case, (e_cost, eta_sys) in COST_CASES.items():
        demand = G_MU_BOUNDARY * float(
            breakeven_xmu(E_f_MeV=E_USE_MEV, E_mu_GeV=e_cost, eta_conv=eta_sys)
        )
        omega_crit = pred[PREFIX + "omegacrit_" + case] / 100.0
        ceiling = float(fusions_per_muon(omega_crit, 1.0e30, lambda_0))
        assert ceiling == pytest.approx(demand, rel=1e-9), case
        # And it is a boundary, not a target: just above it the ceiling falls short at any rate.
        starved = float(fusions_per_muon(omega_crit * 1.01, 1.0e30, lambda_0))
        assert starved < demand


def test_their_useful_cycle_energy_is_not_the_ledger_fusion_energy():
    """The disclosed difference: same closed form, different divisor, so different breakeven.

    Their Eq.(9) and ``breakeven_xmu`` are one function. Fed their unsourced E_use = 20.4 MeV it
    returns their 245; fed the ledger's E_fusion it returns openmucf's own ~284. Neither number is
    corrected into the other, here or anywhere else.
    """
    rates = load_rates()
    assert rates.value("E_fusion") == 17.6
    theirs = float(breakeven_xmu(E_f_MeV=E_USE_MEV, E_mu_GeV=5.0, eta_conv=ETA_SYS))
    ours = float(breakeven_xmu())  # ledger defaults: E_fusion 17.6 MeV, E_mu_cost 5.0 GeV
    assert theirs == pytest.approx(245.098, abs=5e-4)
    assert ours == pytest.approx(284.091, abs=5e-4)
    assert ours / theirs == pytest.approx(E_USE_MEV / rates.value("E_fusion"), rel=1e-12)


# ---------------------------------------------------------------------------------------------
# The registration itself: bands, bibkey, declared conditions
# ---------------------------------------------------------------------------------------------


def test_tolerance_bands_are_the_printed_precision():
    """Every band is the registered rule: the printed figure +- 0.55 units in its last printed place.

    0.5 is the strict rounding half-width their quotation implies; the extra 0.05 keeps a value that
    lands exactly on a rounding boundary off a floating-point knife edge (the 8 GeV omega_crit is
    exactly such a value). Checking the rule here is what stops a band being widened row by row.
    """
    for tid, row in sorted(_registered_rows().items()):
        published, half = float(row["value"]), 0.55 * _printed_ulp(row["value"])
        lo, hi = row["tolerance"].strip("[]").split(",")
        assert float(lo) == pytest.approx(published - half, rel=1e-12), tid
        assert float(hi) == pytest.approx(published + half, rel=1e-12), tid


def test_the_rounding_boundary_guard_is_not_decorative():
    """The 0.05-unit guard earns its place on a measured case, not on a hypothetical one.

    The 8 GeV boundary is exactly 0.255 %, which is the lower edge of the strict interval its printed
    0.26 % implies. Algebraically identical spellings of Eq.(12) land on both sides of that edge in
    float64, so under a strict band the verdict would turn on how the boundary was spelled. Under the
    registered band every spelling passes.
    """
    n_l = float(breakeven_xmu(E_f_MeV=E_USE_MEV, E_mu_GeV=8.0, eta_conv=ETA_SYS))
    spellings = (100.0 / n_l, (1.0 / n_l) * 100.0, (E_USE_MEV / 8000.0) * 100.0)
    strict = "[0.255,0.265]"
    verdicts = {validate._within(s, 0.26, strict) for s in spellings}
    assert verdicts == {True, False}, spellings  # the strict band really does straddle

    registered = _registered_rows()[PREFIX + "omegacrit_8GeV"]["tolerance"]
    assert all(validate._within(s, 0.26, registered) for s in spellings), spellings


def test_the_guard_rescues_no_row():
    """The anti-tuning check: widening the band did not turn a single FAIL into a PASS.

    The 0.05-unit guard exists to remove a dependence on how Eq.(12) is spelled, not to buy headroom.
    So every row must clear the STRICT +-0.5-unit rounding interval as well -- as this file spells the
    arithmetic, all sixteen do, and the 8 GeV boundary does it by sitting exactly on the edge, which is
    the whole reason the guard is there. If a future change makes the guard load-bearing for some row,
    that is a decision to take in the open, not a band to widen quietly.
    """
    pred = _predictions(load_rates())
    for tid, row in sorted(_registered_rows().items()):
        published, half = float(row["value"]), 0.5 * _printed_ulp(row["value"])
        assert published - half <= pred[tid] <= published + half, (
            f"{tid} needs the guard to pass: {pred[tid]!r} outside the strict band "
            f"[{published - half}, {published + half}]"
        )


def test_bands_exclude_the_neighbouring_printed_value():
    """Tight enough to fail a real disagreement: one unit in the last printed place is already out."""
    for tid, row in sorted(_registered_rows().items()):
        published, ulp, tol = float(row["value"]), _printed_ulp(row["value"]), row["tolerance"]
        assert not validate._within(published + ulp, published, tol), tid
        assert not validate._within(published - ulp, published, tol), tid


def test_registered_but_not_scored_by_the_engine_trust_gate():
    """These rows are registered in the CSV and deliberately absent from VALIDATION.md's scoreboard.

    Both facts are pinned together here. If a later change scores them in ``validate.run()``, this test
    fires and forces the disclosure paragraph to be revisited in the same edit, rather than leaving a
    document that says they are excluded next to a table that includes them.
    """
    scored = {r.target_id for r in validate.run(load_rates())}
    assert scored.isdisjoint(_registered_rows()), sorted(scored & set(_registered_rows()))

    repo = Path(__file__).resolve().parents[1]
    for doc in ("VALIDATION.md", "VALIDATION_CHANNELS.md"):
        text = (repo / doc).read_text(encoding="utf-8")
        assert "deliberately NOT scored above" in text, doc
        assert PREFIX in text and "tests/test_koucheng2026.py" in text, doc

    for tid, row in sorted(_registered_rows().items()):
        assert "NOT SCORED IN VALIDATION.md" in row["notes"], tid


def test_these_rows_cannot_detect_an_engine_defect():
    """The measured reason they are not scored above: they are blind to the engine, by construction.

    Halving the formation model's calibration scale is a gross engine defect. It moves most of the
    scored rows and none of these, because nothing here calls the ODE or the formation model -- only
    the closed forms and three ledger values. That is a property worth pinning rather than asserting:
    it is exactly why adding these to the trust gate would inflate a count of engine tests.
    """
    rates = load_rates()
    before_scored = {r.target_id: r.predicted for r in validate.run(rates)}
    before_ours = _predictions(rates)

    original = dict(formation._CALIB)
    try:
        formation._CALIB.update({k: v * 0.5 for k, v in original.items()})
        after_scored = {r.target_id: r.predicted for r in validate.run(rates)}
        after_ours = _predictions(rates)
    finally:
        formation._CALIB.clear()
        formation._CALIB.update(original)
    assert original == formation._CALIB

    moved = [
        tid
        for tid, was in before_scored.items()
        if was == was and was != after_scored[tid]  # was == was skips the DEFERRED nan row
    ]
    assert len(moved) == 8, sorted(moved)  # the count the disclosure paragraph states
    assert after_ours == before_ours, "an engine defect moved a row that does not use the engine"
    assert _predictions(load_rates()) == before_ours  # and the restore is real


def test_the_registered_bibkey_is_the_cycle_closure_paper():
    """All 16 rows cite the cycle-closure paper, and it is not the rate-network paper."""
    from openmucf.rates import REFS_BIB

    assert {row["source_bibkey"] for row in _registered_rows().values()} == {"KouChenLawson2026"}
    bib = REFS_BIB.read_text(encoding="utf-8")
    assert "@article{KouChenLawson2026," in bib and "2607.10989" in bib
    assert "@article{KouChen2026," in bib and "2606.07077" in bib


def test_declared_conditions_match_the_inputs_fed():
    """Each row's ``conditions`` states the inputs this file actually feeds -- no silent drift."""
    rows = _registered_rows()
    for anchor, (y_f, omega) in ANCHOR_INPUTS.items():
        conditions = rows[PREFIX + "Lmu_" + anchor]["conditions"]
        assert f"Y_f={y_f:g}" in conditions, (anchor, conditions)
        assert f"omega_s_eff={omega * 100.0:.2f}%" in conditions, (anchor, conditions)
    for case, (e_cost, eta_sys) in COST_CASES.items():
        for tid in (PREFIX + "NL_" + case, PREFIX + "omegacrit_" + case):
            conditions = rows[tid]["conditions"]
            assert f"E_cost={e_cost:g} GeV" in conditions, (tid, conditions)
            assert f"E_use={E_USE_MEV:g} MeV" in conditions, (tid, conditions)
            assert f"eta_sys={eta_sys:g}" in conditions, (tid, conditions)
    for anchor in ANCHOR_INPUTS:
        conditions = rows[PREFIX + "Gmu_" + anchor]["conditions"]
        assert f"E_cost={E_COST_GEV_GAIN:g} GeV" in conditions, (anchor, conditions)
        assert f"E_use={E_USE_MEV:g} MeV" in conditions, (anchor, conditions)
