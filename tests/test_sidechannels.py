"""openmucf.sidechannels: the side-channel loss per cycle at the liquid anchor fill and its inputs."""

from __future__ import annotations

import copy
import csv
import io
import math
import re
from pathlib import Path

import numpy as np
import pytest

from openmucf import sidechannels

REPO = Path(__file__).resolve().parents[1]

# The pre-registered values these tests bind: L_central and the components (mean over the two analyses) of the
# registered propagation, and sigma_L with a >= 0 enforced, all in percent.
L_CENTRAL_PCT = 0.1388
SIGMA_L_PCT = 0.019748
COMPONENTS_PCT = {"tt": 0.0641, "dd": 0.0154, "p": 0.0493, "imp": 0.0100}


def anchor_failures(a):
    """Every way ``a`` (an ``anchor_l_other()`` result) misses the bound values; empty when it meets them."""
    bad = []
    if abs(a["central_pct"] / L_CENTRAL_PCT - 1.0) > 1e-3:
        bad.append(f"central_pct {a['central_pct']}")
    if abs(a["sigma_pct"] / SIGMA_L_PCT - 1.0) > 1e-3:
        bad.append(f"sigma_pct {a['sigma_pct']}")
    for k, v in COMPONENTS_PCT.items():
        if abs(a["components_pct"][k] - v) > 5e-5:
            bad.append(f"components_pct[{k}] {a['components_pct'][k]}")
    if (round(a["central_pct"], 4), round(a["sigma_pct"], 4)) != sidechannels.ANCHOR_L_OTHER_PCT:
        bad.append("ANCHOR_L_OTHER_PCT")
    c = a["components_pct"]
    if not math.isclose(a["intrinsic_pct"], c["tt"] + c["dd"], rel_tol=1e-15):
        bad.append("intrinsic_pct")
    if not math.isclose(a["intrinsic_fraction"], a["intrinsic_pct"] / 100, rel_tol=1e-15):
        bad.append("intrinsic_fraction")
    return bad


def test_anchor_l_other_reproduces_the_registered_values():
    assert anchor_failures(sidechannels.anchor_l_other()) == []
    assert sidechannels.ANCHOR_L_OTHER_PCT[0] / 100 == sidechannels.ANCHOR_L_OTHER_FRACTION


@pytest.mark.parametrize(
    "key, factor, named",
    [("central_pct", 1.002, "central_pct"), ("sigma_pct", 1.002, "sigma_pct"), ("sigma_pct", 1.02, "ANCHOR")],
)
def test_anchor_failures_names_a_perturbed_value(key, factor, named):
    """Drill: a copy of the computed result with one value moved past its tolerance is named."""
    a = copy.deepcopy(sidechannels.anchor_l_other())
    a[key] *= factor
    assert any(f.startswith(named) for f in anchor_failures(a))


def test_anchor_failures_names_a_moved_component():
    a = copy.deepcopy(sidechannels.anchor_l_other())
    a["components_pct"]["tt"] += 6e-5
    assert any(f.startswith("components_pct[tt]") for f in anchor_failures(a))


def test_components_refuse_a_negative_a():
    inp = sidechannels.load_inputs()
    args = (0.3157, 121.711e6, 280e6, -0.5, 2.272e6, 0.139, 3.365e6, 0.122, 6.0e6, 0.01, 1.0e-4)
    with pytest.raises(ValueError, match="q_1s"):
        sidechannels.components(*args, inputs=inp)
    with pytest.raises(ValueError):
        sidechannels.components(*args[:3], np.array([1.0, -1e-9]), *args[4:], inputs=inp)
    ok = sidechannels.components(*args[:3], 0.0, *args[4:], inputs=inp)
    assert float(ok["q1s"]) == 1.0


@pytest.mark.parametrize("an", ["i", "ii"])
def test_truncated_rule_reduces_to_gauss_hermite_untruncated(an):
    """The a-axis rule, with the truncation removed, reproduces numpy's Gauss-Hermite nodes and weights."""
    row = sidechannels.load_inputs()[f"a_q1s_{an}"]
    mu, sd = row["value"], row["unc"]
    nodes, wts = sidechannels.truncated_normal_gauss(mu, sd, -math.inf, 5)
    x, w = np.polynomial.hermite_e.hermegauss(5)
    assert np.max(np.abs(np.sort(nodes) - (mu + sd * x))) < 1e-12
    assert np.max(np.abs(wts[np.argsort(nodes)] - w / w.sum())) < 1e-12
    tn, tw = sidechannels.truncated_normal_gauss(mu, sd, 0.0, 5)
    assert tn.min() > 0.0 and math.isclose(tw.sum(), 1.0, rel_tol=1e-12)


def test_inputs_csv_rows_are_complete_and_cited():
    inp = sidechannels.load_inputs()
    bib = (REPO / "openmucf" / "data" / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M))
    with open(sidechannels.INPUTS_CSV, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    assert [r["symbol"] for r in raw] == list(inp)
    for sym, r in inp.items():
        assert r["source_bibkey"] in keys, sym
        assert r["source_locator"].strip(), sym
        if r["distribution"] == "normal":
            assert r["unc"] is not None and r["unc"] > 0, sym
        elif r["distribution"] == "uniform":
            assert r["lo"] < r["hi"] and math.isclose(r["value"], 0.5 * (r["lo"] + r["hi"])), sym
        else:
            assert r["distribution"] == "fixed" and r["unc"] is None, sym


def _inputs_copy(tmp_path, edit):
    lines = sidechannels.INPUTS_CSV.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "inputs.csv"
    path.write_text("\n".join(edit(lines)) + "\n", encoding="utf-8")
    return path


def test_load_inputs_refuses_a_symbol_on_two_rows(tmp_path):
    """Each symbol given a second row (the copy changed): refused naming it, never reduced to the last row."""
    lines = sidechannels.INPUTS_CSV.read_text(encoding="utf-8").splitlines()
    assert sidechannels.load_inputs(_inputs_copy(tmp_path, lambda ls: ls)) == sidechannels.load_inputs()
    symbols = list(sidechannels.load_inputs())
    assert len(symbols) == len(lines) - 1
    for i, sym in enumerate(symbols, start=1):
        path = _inputs_copy(tmp_path, lambda ls, i=i: [*ls, ls[i]])
        with pytest.raises(ValueError, match=f"symbol '{sym}' is on more than one row"):
            sidechannels.load_inputs(path)


@pytest.mark.parametrize("column", ["value", "unc", "dist_lo", "dist_hi"])
@pytest.mark.parametrize("token", ["nan", "inf", "-inf", "NaN"])
def test_load_inputs_refuses_a_non_finite_number(tmp_path, column, token):
    """Each numeric field of each row set to a non-finite number is refused naming the symbol and the column,
    never carried into a loss."""
    lines = sidechannels.INPUTS_CSV.read_text(encoding="utf-8").splitlines()
    header = next(csv.reader([lines[0]]))
    col = header.index(column)
    for i in range(1, len(lines)):
        sym = next(csv.reader([lines[i]]))[header.index("symbol")]

        def edit(ls, i=i):
            row = next(csv.reader([ls[i]]))
            row[col] = token
            buf = io.StringIO()
            csv.writer(buf, lineterminator="").writerow(row)
            return [*ls[:i], buf.getvalue(), *ls[i + 1:]]
        with pytest.raises(ValueError, match=re.escape(f"{sym}.{column}: '{token}' is not a finite number")):
            sidechannels.load_inputs(_inputs_copy(tmp_path, edit))


def test_load_inputs_refuses_a_repeated_header_column(tmp_path):
    """Each column of the header stated twice (a copy of its values appended): refused naming it, never
    reduced to the last copy."""
    lines = sidechannels.INPUTS_CSV.read_text(encoding="utf-8").splitlines()
    header = next(csv.reader([lines[0]]))
    for j, name in enumerate(header):
        def edit(ls, j=j):
            out = []
            for ln in ls:
                row = next(csv.reader([ln]))
                buf = io.StringIO()
                csv.writer(buf, lineterminator="").writerow([*row, row[j]])
                out.append(buf.getvalue())
            return out
        with pytest.raises(ValueError, match=re.escape(f"column(s) ['{name}'] repeated in the header")):
            sidechannels.load_inputs(_inputs_copy(tmp_path, edit))


def ledger_tt_failures(rates, inp):
    """The ledger's lambda_ttmu must be the tabled rate over the JINR target density (value and unc), and its
    omega_tt the inputs' value and unc."""
    bad = []
    dens = inp["jinr_density_lhd"]["value"]
    tab = inp["lambda_ttmu_table"]
    if not math.isclose(rates["lambda_ttmu"].value, tab["value"] / dens, rel_tol=1e-12):
        bad.append("lambda_ttmu.value")
    if not math.isclose(rates["lambda_ttmu"].unc, tab["unc"] / dens, rel_tol=1e-12):
        bad.append("lambda_ttmu.unc")
    for f in ("value", "unc"):
        if getattr(rates["omega_tt"], f) != inp["omega_tt"][f]:
            bad.append(f"omega_tt.{f}")
    return bad


def test_ledger_tt_rows_are_the_side_channel_inputs():
    from openmucf import load_rates

    assert ledger_tt_failures(load_rates(), sidechannels.load_inputs()) == []


@pytest.mark.parametrize("sym, field", [("lambda_ttmu_table", "value"), ("lambda_ttmu_table", "unc"),
                                        ("omega_tt", "value"), ("omega_tt", "unc")])
def test_ledger_tt_binding_drill(sym, field):
    """Drill: an inputs copy with one of the bound values changed is named."""
    from openmucf import load_rates

    inp = copy.deepcopy(sidechannels.load_inputs())
    inp[sym][field] *= 1.01
    assert ledger_tt_failures(load_rates(), inp)
