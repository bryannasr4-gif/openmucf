"""Regression lock + consistency checks for the registered UQ-priors file (openmucf/data/uq_priors.csv)."""

import csv

import pytest

from openmucf import load_rates, uq


def test_params_from_ledger_matches_frozen_literals():
    """uq.PARAMS (now file-sourced) equals the six frozen pre-schema literals.

    Compares (name, nominal, low, high, unit) ONLY; ``note`` is deliberately excluded -- the rationale
    strings intentionally supersede the old note strings, so the NUMBERS are the regression lock, the
    prose is not.
    """
    frozen = [
        ("omega_s0_pct", 0.857, 0.80, 0.95, "%"),
        ("R", 0.35, 0.20, 0.45, "-"),
        ("lambda_c", 1.30e8, 1.00e8, 1.45e8, "s^-1"),
        ("E_mu_GeV", 5.0, 2.0, 10.0, "GeV"),
        ("eta_acc", 0.30, 0.10, 0.50, "-"),
        ("eta_thermal", 0.40, 0.35, 0.45, "-"),
    ]
    params = uq.params_from_ledger()
    assert len(params) == len(frozen)
    for p, (name, nominal, low, high, unit) in zip(params, frozen, strict=True):
        assert (p.name, p.nominal, p.low, p.high, p.unit) == (name, nominal, low, high, unit)


def test_lambda_c_prior_matches_ledger_bounds():
    """The lambda_c prior box == the ledger lambda_c_liquid dist bounds (single actual-rate basis)."""
    r = load_rates()
    lo, hi = r.dist_bounds("lambda_c_liquid")
    lam = next(p for p in uq.params_from_ledger() if p.name == "lambda_c")
    assert (lam.low, lam.high) == (lo, hi)


def test_ledger_symbols_resolve_and_values_in_box():
    """Every non-empty ledger_symbols entry resolves in the ledger and its value lies inside [low, high]."""
    r = load_rates()
    with open(uq._PRIORS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        syms = [s.strip() for s in row["ledger_symbols"].split(";") if s.strip()]
        lo, hi = float(row["low"]), float(row["high"])
        for s in syms:
            assert s in r, f"{row['name']}: ledger symbol {s!r} not in ledger"
            v = r.value(s)
            assert lo <= v <= hi, f"{row['name']}: linked {s} value {v} outside [{lo}, {hi}]"


def _priors_rows():
    with open(uq._PRIORS_CSV, newline="") as f:
        return f.read().splitlines()


def test_each_parameter_stated_twice_is_refused(tmp_path):
    """Each prior given a second row is refused naming it: the readers that key the parameters by name (the
    calibration's Q_net boxes, NOMINAL) would otherwise keep the last row alone."""
    lines = _priors_rows()
    names = [p.name for p in uq.params_from_ledger()]
    assert len(lines) == len(names) + 1
    admitted = []
    for i, name in enumerate(names):
        path = tmp_path / f"priors_{i}.csv"
        path.write_text("\n".join([*lines, lines[i + 1]]) + "\n", encoding="utf-8")
        try:
            uq.params_from_ledger(path)
        except ValueError as e:
            if f"parameter(s) [{name!r}] on more than one row" in str(e):
                continue
        admitted.append(name)
    assert admitted == []


def test_a_header_column_stated_twice_is_refused(tmp_path):
    lines = _priors_rows()
    path = tmp_path / "priors.csv"
    path.write_text("\n".join([lines[0] + ",low", *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"column\(s\) \['low'\] repeated in the header"):
        uq.params_from_ledger(path)
