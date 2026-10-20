"""Primitive evidence replay and adverse branches of the historical numerical rule."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from openmucf.g4 import d3_numerical as d

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence():
    return d.load_evidence(ROOT)


@pytest.fixture(scope="module")
def evaluated(evidence):
    return d.evaluate_all(ROOT, evidence)


def terms(**changes):
    base = {"dx": (0.0, False), "csteps": (0.0, True), "domain": (0.0, True),
            "tol": (0.0, False), "phase": (0.0, False)}
    base.update(changes)
    return [d.Term(k, *v) for k, v in base.items()]


def compare(**kw):
    args = dict(q=0.0, rounding=0.0, ref=0.0, bound=0.0, target=1.0,
                terms=terms(), identity_ok=True)
    args.update(kw)
    return d.compare(**args)


def test_complete_historical_replay(evaluated, evidence):
    with (ROOT / "data/g4/d3/numerical_check.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert set(evaluated) == set(evidence.expected)
    assert len(evaluated) == len(rows) > 0
    for row in rows:
        key = int(row["Z"]), int(row["A"]), row["uehling"] == "on", row["quantity"]
        assert evaluated[key].historical_summary() == row
        assert evaluated[key].verdict == "QUALIFIED"
        for est in (evaluated[key].prereg, evaluated[key].a3):
            assert est.gates == {"G1": True, "G2": True, "G3": True, "G4": True, "G5": None}


def test_real_evidence_signed_terms(evidence, evaluated):
    """Pin signed refinement and reference shifts at a nonzero historical witness."""
    key = (36, 84, True, "K-e2")
    case = evidence.cases[key[:2]]
    payload = evidence.payloads[key[:3]]

    def centroid_line(bindings):
        return bindings["K1"] - (2*bindings["L2"] + 4*bindings["L3"])/6

    def producer(role):
        quantities, bindings = d.producer_quantities(evidence.runs[*key[:3], role], case)
        value = centroid_line({s: float(v[0]) for s, v in bindings.items()})
        return value, quantities[key[3]][1]

    qmd = producer("dx3")[0]
    expected = {name: qmd-producer(role)[0] for name, role in (("N_u", "nu"), ("N_s", "ns"))}
    eh = payload["e_h_eV"]
    base = {s: -payload["A"][s]["prod"]["B_Ha"]*eh for s in ("K1", "L2", "L3")}
    for name, shift in (("csteps", "C:herm5000"), ("domain", "dom:1e-5")):
        moved = {s: value-payload["eps_level_Ha"][shift][s]*eh for s, value in base.items()}
        expected[name] = centroid_line(moved)-centroid_line(base)
    assert {name: f"{value:+.3e}" for name, value in expected.items()} == {
        "N_u": "-1.481e-07", "N_s": "+1.257e-08", "csteps": "+7.609e-07", "domain": "+9.722e-05",
    }
    ladder = [producer(f"dx{k}") for k in (1, 2, 3, 4)]
    row = evaluated[key]
    assert not row.floor
    for aware, estimator in ((False, row.prereg), (True, row.a3)):
        actual = {term.name: term for term in estimator.terms}
        for name, value in expected.items():
            assert actual[name] == d.Term(name, value, True)
        dx, signed, _ = d.richardson([q for q, _ in ladder],
                                     [rho for _, rho in ladder] if aware else None)
        prediction = sum([dx if signed else 0.0, *expected.values()])
        assert estimator.d_pred == prediction
        assert f"{prediction:.6f}" == "-0.059737"


def test_five_real_floor_witnesses(evidence, evaluated):
    original = d.evaluate_all(ROOT, evidence, floor_amendment=False)
    changed = {k for k in evaluated if evaluated[k].verdict != original[k].verdict}
    assert changed == {(6, 12, False, "level:L3"), (6, 12, False, "line:L3-M5"),
                       (5, 11, True, "line:Q12-R14"), (4, 9, False, "e2"),
                       (49, 115, False, "line:Q13-R15")}
    for key in changed:
        assert evaluated[key].floor
        assert original[key].verdict == "NUMERICAL_FAIL"


@pytest.mark.parametrize("field,value", [("nodes", 1), ("kappa", 1)])
def test_reference_identity(evidence, field, value):
    changed = copy.deepcopy(evidence)
    changed.payloads[6, 12, True]["A"]["K1"]["prod"][field] = value
    row = d.evaluate_all(ROOT, changed, members={(6, 12)})[6, 12, True, "K"]
    assert row.prereg.gates == {"G1": False, "G2": True, "G3": True, "G4": True, "G5": None}
    assert row.verdict == "NUMERICAL_FAIL"
    assert row.reason == "identity"


@pytest.mark.parametrize("field,value", [("n", "2"), ("l", "1"), ("s", "0"),
                                        ("nodes_P", "1"), ("binding_eV", "0")])
def test_producer_identity_and_headers(evidence, field, value):
    changed = copy.deepcopy(evidence)
    changed.runs[6, 12, True, "dx3"]["states"]["K1"][field] = value
    row = d.evaluate_all(ROOT, changed, members={(6, 12)})[6, 12, True, "K"]
    assert not row.prereg.gates["G1"]
    assert row.prereg.gates["G2"] and row.prereg.gates["G3"] and row.prereg.gates["G4"]
    assert row.verdict == "NUMERICAL_FAIL"


def test_g2_boundary_and_reference_refinement(evidence):
    assert compare(bound=0.2).gates["G2"]
    assert not compare(bound=math.nextafter(0.2, math.inf)).gates["G2"]
    changed = copy.deepcopy(evidence)
    p = changed.payloads[6, 12, True]
    p["A"]["K1"]["rtol13"]["B_Ha"] += 0.015/p["e_h_eV"]
    row = d.evaluate_all(ROOT, changed, members={(6, 12)})[6, 12, True, "K"]
    assert row.prereg.gates["G1"] and not row.prereg.gates["G2"] and row.prereg.gates["G3"]


def test_g3_discrepancy_and_ladder_branches(evaluated):
    row = evaluated[6, 12, True, "K"]
    adverse = d.compare(row.q_ev, 0.0, row.q_ref_ev+0.1, row.b_ref_ev, row.target_ev,
                        list(row.prereg.terms), True, uehling=True)
    assert adverse.gates["G1"] and adverse.gates["G2"] and not adverse.gates["G3"]
    assert compare(q=1.0).gates["G3"]
    assert not compare(q=math.nextafter(1.0, math.inf)).gates["G3"]
    ladder = compare(terms=terms(dx=(1.01, False)))
    assert ladder.gates["G1"] and ladder.gates["G2"] and not ladder.gates["G3"]
    assert compare(terms=terms(dx=(1.0, False))).gates["G3"]


def test_g4_signed_domain_and_sign_reversal():
    bad = compare(terms=terms(domain=(0.5, True)))
    assert bad.gates == {"G1": True, "G2": True, "G3": True, "G4": False, "G5": None}
    good = compare(q=0.3, terms=terms(domain=(0.3, True)))
    reverse = compare(q=0.3, terms=terms(domain=(-0.3, True)))
    assert good.gates["G4"] and not reverse.gates["G4"]
    assert good.b_num == reverse.b_num


def test_first_failure_order():
    assert compare(identity_ok=False, bound=0.3, q=2).reason == "identity"
    assert compare(bound=0.3, q=2).reason == "reference"
    assert compare(q=2).reason == "agreement"
    assert compare(terms=terms(domain=(0.5, True))).reason == "unattributed"


def test_record_requires_both_estimators(evaluated):
    row = evaluated[6, 12, True, "K"]
    bad = compare(q=2)
    for kwargs in ({"prereg": bad}, {"a3": bad}):
        mutated = replace(row, **kwargs)
        assert mutated.verdict == "NUMERICAL_FAIL"
        assert mutated.reason == "agreement"
        assert mutated.b_num_ev == max(mutated.prereg.b_num, mutated.a3.b_num)


def test_richardson_stable_and_unsigned_fallbacks():
    qs = [8.0, 2.0, 0.5, 0.125]
    for rho in (None, [1e-12]*4):
        value, signed, orders = d.richardson(qs, rho)
        assert signed and value == 0.5
        assert orders == [2.0, 2.0]
    assert d.richardson([3.0, 2.0, 3.0, 2.0])[:2] == (2.0, False)
    assert d.richardson(qs, [1.0]*4)[:2] == (4.75, False)
    assert math.isinf(d.richardson([3.0, 2.0, 1.0, 0.0])[0])
    assert math.isinf(d.richardson([1.0, 0.0])[0])


def test_richardson_fallback_preserves_accumulation_order():
    # Each half-ulp is rounded separately by the realized historical expression.
    rho = [0.0, 0.0, 2**-53, 2**-53]
    value, signed, _ = d.richardson([3.0, 2.0, 3.0, 2.0], rho)
    assert not signed
    assert value == 2.0


def test_historical_full_precision_fallback(evidence):
    row = d.evaluate_all(ROOT, evidence, members={(2, 3)})[2, 3, True, "level:L3"]
    assert row.a3.terms[0] == d.Term("dx", 4.561823756240961e-07, False)


@pytest.mark.parametrize("key,target", [((21, 45), 4.1), ((32, 74), 1.7),
                                       ((44, 102), 2.9), ((53, 127), 3.3),
                                       ((60, 142), 5.1), ((79, 197), 14.6)])
def test_measured_targets_preserve_registered_decimal_values(evidence, key, target):
    assert evidence.targets[key] == target
    assert d.target_for(evidence, key, "level:K1", 1.0) == target


def test_richardson_stability_boundary():
    def ladder(prior):
        d1, d2 = 4.0, 1.0
        d0 = d1*2**prior
        return [d0+d1+d2, d1+d2, d2, 0.0]
    assert d.richardson(ladder(2.3-1e-12))[1]
    assert not d.richardson(ladder(2.3+1e-12))[1]


def test_order_resolution_sign_size_and_width_boundaries():
    assert d.order_resolved(4, 1, 0.001, 0.001)
    assert not d.order_resolved(-4, 1, 0.001, 0.001)
    assert not d.order_resolved(4, 1, 4, 0)
    assert not d.order_resolved(4, 1, 0, 1)
    x = (2**0.05-1)/(2**0.05+1)
    assert d.order_resolved(1, 1, x*(1-1e-10), x*(1-1e-10))
    assert not d.order_resolved(1, 1, x*(1+1e-10), x*(1+1e-10))


def test_floor_every_step_inclusive_unit_and_strict_print_boundary():
    assert d.floor_ladder([6.0, 4.0, 2.0, 0.0], 1.0)
    assert not d.floor_ladder([7.0, 4.0, 2.0, 0.0], 1.0)
    assert not d.floor_ladder([], 1.0)
    assert not d.floor_ladder([1.0, 0.0], 1.0)
    # Synthetic positive-unit truth table: print floor is separately reachable here.
    assert d.floor_ladder([0.75, 0.5, 0.25, 0.0], 0.01, 0.3)
    assert not d.floor_ladder([0.75, 0.5, 0.25, 0.0], 0.01, 0.25)


@pytest.mark.parametrize("role", ["dx1", "tol", "phase2", "nu", "ns"])
def test_missing_required_run(evidence, role):
    changed = copy.deepcopy(evidence)
    del changed.runs[6, 12, True, role]
    with pytest.raises(d.EvidenceError, match="required run"):
        d.evaluate_all(ROOT, changed, members={(6, 12)})


@pytest.mark.parametrize("field", ["dom:1e-5", "C:herm5000"])
def test_missing_reference_shift(evidence, field):
    changed = copy.deepcopy(evidence)
    del changed.payloads[6, 12, True]["eps_level_Ha"][field]
    with pytest.raises(d.EvidenceError, match="primitive"):
        d.evaluate_all(ROOT, changed, members={(6, 12)})


@pytest.mark.parametrize("variant", list(d.VARIANTS))
def test_every_reference_variant_is_required(evidence, variant):
    changed = copy.deepcopy(evidence)
    del changed.payloads[6, 12, True]["A"]["K1"][variant]
    with pytest.raises(d.EvidenceError):
        d.evaluate_all(ROOT, changed, members={(6, 12)})


@pytest.mark.parametrize("kwargs", [{"q": math.nan}, {"ref": math.inf}, {"rounding": -1},
                                   {"bound": -1}, {"target": 0}, {"target": -1}])
def test_nonfinite_and_invalid_bounds(kwargs):
    with pytest.raises(d.EvidenceError):
        compare(**kwargs)


def test_missing_terms_are_adverse_and_g5_is_inapplicable():
    assert compare().gates["G5"] is None
    bad = compare(terms=terms(domain=(math.nan, True)))
    assert bad.verdict == "NUMERICAL_FAIL"
    assert math.isinf(bad.b_ladder) and math.isinf(bad.b_num)
    assert bad.terms[-1] == d.Term("missing", math.inf, False)
    with pytest.raises(d.EvidenceError, match="extrapolation"):
        compare(extrapolated=True)


def test_budget_arithmetic_includes_each_contribution():
    result = compare(q=0.3, rounding=0.04, bound=0.05,
                     terms=terms(dx=(0.02, True), domain=(0.06, True), csteps=(-0.01, True),
                                 tol=(-0.03, False), phase=(0.07, False)))
    assert result.b_ladder == pytest.approx(0.23)
    assert result.d_obs == pytest.approx(0.3)
    assert result.d_pred == pytest.approx(0.07)
    assert result.allowance == pytest.approx(0.29)
    assert result.b_num == pytest.approx(0.35)


@pytest.mark.parametrize("name", ["dx", "tol", "phase", "csteps", "domain", "N_u", "N_s"])
def test_required_term_coverage_and_signedness(name):
    all_terms = terms(N_u=(0, True), N_s=(0, True))
    with pytest.raises(d.EvidenceError, match="required budget term"):
        compare(terms=[t for t in all_terms if t.name != name], uehling=True)
    if name != "dx":
        with pytest.raises(d.EvidenceError, match="signedness"):
            compare(terms=[replace(t, signed=not t.signed) if t.name == name else t for t in all_terms],
                    uehling=True)


def test_duplicate_terms_refused():
    with pytest.raises(d.EvidenceError, match="required budget term"):
        compare(terms=terms()+[d.Term("dx", 0, False)])


def test_wrong_term_names_refused():
    changed = [replace(t, name="unknown") if t.name == "tol" else t for t in terms()]
    with pytest.raises(d.EvidenceError, match="required budget term"):
        compare(terms=changed)


def test_unique_keys_refuse_duplicates():
    with pytest.raises(d.EvidenceError, match="duplicate"):
        d._unique([{"key": 1}, {"key": 1}], lambda row: row["key"], "fixture")


@pytest.mark.parametrize("change", ["missing", "empty", "extra"])
def test_expected_coverage_is_independent(evidence, change):
    expected = evidence.expected
    if change == "missing":
        expected = expected[:-1]
    elif change == "empty":
        expected = ()
    else:
        expected += ((999, 999, True, "K"),)
    with pytest.raises(d.EvidenceError, match="coverage"):
        d.evaluate_all(ROOT, replace(evidence, expected=expected))


def test_empty_member_selection_cannot_pass(evidence):
    with pytest.raises(d.EvidenceError, match="coverage"):
        d.evaluate_all(ROOT, evidence, members=set())


def copy_bundle(tmp_path):
    shutil.copytree(ROOT / d.REFERENCE_RELPATH, tmp_path / d.REFERENCE_RELPATH)
    for name in ("validation_cells.csv", "mudirac_runs.csv", "mudirac_lines.csv", "mudirac_states.csv"):
        shutil.copyfile(ROOT / "data/g4/d3" / name, tmp_path / "data/g4/d3" / name)
    return tmp_path / d.REFERENCE_RELPATH


def revise(base, name, value):
    path = base / name
    path.write_text(json.dumps(value)+"\n", encoding="ascii")
    protocol = json.loads((base / "protocol.json").read_text())
    protocol["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (base / "protocol.json").write_text(json.dumps(protocol), encoding="ascii")


def test_bundle_digest_guard(tmp_path):
    base = copy_bundle(tmp_path)
    with (base / "cases.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(d.EvidenceError, match="digest"):
        d.load_evidence(tmp_path)


@pytest.mark.parametrize("change", ["empty", "duplicate", "missing", "wrong_quantity"])
def test_cross_product_inventory_guard(tmp_path, change):
    base = copy_bundle(tmp_path)
    keys = json.loads((base / "expected_keys.json").read_text())
    if change == "empty":
        keys = []
    elif change == "duplicate":
        keys.append(keys[0])
    elif change == "missing":
        keys.pop()
    else:
        keys[0][-1] = "wrong"
    revise(base, "expected_keys.json", keys)
    with pytest.raises(d.EvidenceError, match="cross product"):
        d.load_evidence(tmp_path)


@pytest.mark.parametrize("change", ["schema", "recipe"])
def test_protocol_recipe_is_fixed(tmp_path, change):
    base = copy_bundle(tmp_path)
    protocol = json.loads((base / "protocol.json").read_text())
    if change == "schema":
        protocol["schema"] = 2
    else:
        protocol["recipe"]["extrapolated"] = True
    (base / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(d.EvidenceError, match="unsupported"):
        d.load_evidence(tmp_path)


@pytest.mark.parametrize("change", ["run_missing", "payload_missing", "input", "settings", "mapping",
                                   "production", "physical", "radius", "run_key", "mapped_line",
                                   "mapped_state"])
def test_primitive_joins(evidence, change):
    changed = copy.deepcopy(evidence)
    run = changed.runs[6, 12, True, "dx3"]
    if change == "run_missing":
        del changed.runs[6, 12, True, "dx3"]
    elif change == "payload_missing":
        del changed.payloads[6, 12, True]
    elif change == "input":
        run = changed.runs[6, 12, True, "phase1"]
        assert run["public_run"] is None
        run["input_text"] += "\n"
    elif change == "settings":
        run["settings"]["k"] = 4
    elif change == "mapping":
        run = changed.runs[6, 12, True, "tol"]
        assert run["public_run"] is not None
        run["public_run"] = "unknown"
    elif change == "production":
        changed.cases[6, 12]["production"]["on"] = "unknown"
    elif change == "physical":
        changed.payloads[6, 12, True]["Z"] = 7
    elif change == "run_key":
        run["z"] = 7
    elif change == "mapped_line":
        run["lines"]["K1-L2"] = "0"
    elif change == "mapped_state":
        run["states"]["K1"]["binding_eV"] = "0"
    else:
        changed.payloads[6, 12, True]["radius_fm"] *= 2
    with pytest.raises(d.EvidenceError):
        d._validate_joins(ROOT, changed)


def test_target_rule_is_not_one_step_helper(evidence):
    assert d.target_for(evidence, (6, 12), "level:K1", 1.0) == evidence.targets[6, 12]
    assert d.target_for(evidence, (6, 12), "line:K1-L2", 1.0) == evidence.targets[6, 12]
    assert d.target_for(evidence, (6, 12), "K-e2", 1.0) == evidence.targets[6, 12]
    assert d.target_for(evidence, (6, 12), "K", 1.0) == 0.001
    assert d.target_for(evidence, (999, 999), "K", 2e6) == 2.0
    unlabelled = replace(evidence, targets={(999, 999): 0.05})
    assert d.target_for(unlabelled, (999, 999), "K-e2", 1.0) == 0.001


def test_reference_bound_is_quantity_maximum(evidence):
    payload = copy.deepcopy(evidence.payloads[6, 12, True])
    base = payload["A"]
    for s in d.CHAIN:
        for v in d.VARIANTS:
            base[s][v]["B_Ha"] = -1.0
        for values in payload["B"].values():
            values[s] = -1.0
    base["K1"]["rtol13"]["B_Ha"] = -2.0
    base["L2"]["rmin100"]["B_Ha"] = -2.0
    _, bound, _ = d.reference_quantities(payload)
    assert bound["line:K1-L2"] == payload["e_h_eV"]


@pytest.mark.parametrize("family,variant", [("A", v) for v in d.VARIANTS]
                         + [("B", v) for v in ("220", "300")])
def test_reference_bound_includes_each_variant(evidence, family, variant):
    payload = copy.deepcopy(evidence.payloads[6, 12, True])
    for label in d.CHAIN:
        for values in payload["A"][label].values():
            values["B_Ha"] = -1.0
        for values in payload["B"].values():
            values[label] = -1.0
    if family == "A":
        payload["A"]["K1"][variant]["B_Ha"] = -2.0
    else:
        payload["B"][variant]["K1"] = -2.0
    q, bound, _ = d.reference_quantities(payload)
    assert bound["K"] == payload["e_h_eV"]
    assert q["K"] == (2 if (family, variant) == ("A", "prod") else 1)*payload["e_h_eV"]


def test_nonfinite_header_is_refused():
    with pytest.raises(d.EvidenceError, match="nonfinite"):
        d.header_half_unit("nan")


@pytest.mark.parametrize("change", ["keyword", "number", "duplicate"])
def test_input_recipe_is_bound_to_physical_case(evidence, change):
    run = copy.deepcopy(evidence.runs[6, 12, True, "dx3"])
    if change == "keyword":
        run["input_text"] = run["input_text"].replace("element: C", "element: N")
    elif change == "number":
        run["input_text"] = run["input_text"].replace("radius: 3.", "radius: 4.")
    else:
        run["input_text"] += "element: C\n"
    with pytest.raises(d.EvidenceError):
        d._validate_input(run, evidence.cases[6, 12])


def test_join_checks_recipe_after_digest_is_updated(evidence):
    changed = copy.deepcopy(evidence)
    run = changed.runs[6, 12, True, "phase1"]
    assert run["public_run"] is None
    run["input_text"] = run["input_text"].replace("element: C", "element: N")
    run["input_sha256"] = hashlib.sha256(run["input_text"].encode("ascii")).hexdigest()
    with pytest.raises(d.EvidenceError, match="input recipe"):
        d._validate_joins(ROOT, changed)


@pytest.mark.parametrize("keyword,value", [("reduced_mass", "FALSE"),
                                          ("optimise_fermi_parameters", "TRUE"),
                                          ("ideal_atom_minshell", "R"), ("xr_lines", "K1-L2"),
                                          ("state_print_precision", "6"), ("output", "0")])
def test_all_fixed_recipe_settings_are_bound(evidence, keyword, value):
    run = copy.deepcopy(evidence.runs[6, 12, True, "dx1"])
    run["input_text"] = "\n".join(f"{keyword}: {value}" if line.startswith(keyword+":") else line
                                   for line in run["input_text"].splitlines())
    with pytest.raises(d.EvidenceError, match="input recipe"):
        d._validate_input(run, evidence.cases[6, 12])


@pytest.mark.parametrize("line", ["loggrid_center: 1.1", "unknown: TRUE"])
def test_extra_recipe_settings_are_refused(evidence, line):
    run = copy.deepcopy(evidence.runs[6, 12, True, "dx1"])
    run["input_text"] += line+"\n"
    with pytest.raises(d.EvidenceError, match="unsupported keywords"):
        d._validate_input(run, evidence.cases[6, 12])


def test_missing_producer_state_fails_identity(evidence):
    changed = copy.deepcopy(evidence)
    del changed.runs[6, 12, True, "dx3"]["states"]["K1"]
    result = d.evaluate_all(ROOT, changed, members={(6, 12)})[6, 12, True, "K"]
    assert result.prereg.gates == {"G1": False, "G2": True, "G3": True, "G4": True, "G5": None}
    assert result.verdict == "NUMERICAL_FAIL"


@pytest.mark.parametrize("where", ["A", "B", "domain", "first_order", "conversion"])
def test_reference_numerical_primitives_are_checked(evidence, where):
    changed = copy.deepcopy(evidence)
    payload = changed.payloads[6, 12, False]
    if where == "A":
        payload["A"]["K1"]["match05"]["B_Ha"] = math.nan
    elif where == "B":
        payload["B"]["300"]["K1"] = math.inf
    elif where == "domain":
        payload["eps_level_Ha"]["dom:1e-5"]["K1"] = math.nan
    elif where == "first_order":
        payload["first_order_level_Ha"]["VU_F1"]["L1"] = math.nan
    else:
        payload["e_h_eV"] *= 2
    with pytest.raises(d.EvidenceError):
        d.evaluate_all(ROOT, changed, members={(6, 12)})


@pytest.mark.parametrize("field", ["reduced_mass", "radius_fm", "rms_run", "t_fm"])
@pytest.mark.parametrize("value", [math.nan, math.inf, 0.0, -1.0])
def test_physical_inputs_are_finite_and_positive(evidence, field, value):
    changed = copy.deepcopy(evidence)
    changed.cases[6, 12][field] = value
    with pytest.raises(d.EvidenceError, match="physical case input"):
        d.evaluate_all(ROOT, changed, members={(6, 12)})
    with pytest.raises(d.EvidenceError, match="physical case input"):
        d._validate_joins(ROOT, changed)
    with pytest.raises(d.EvidenceError, match="physical case input"):
        d.with_payloads(changed, {(6, 12, True): changed.payloads[6, 12, True]})


@pytest.mark.parametrize("change", ["empty", "unknown", "mass", "radius", "key", "switch",
                                   "missing_A", "missing_B", "nonfinite"])
def test_fresh_payload_overrides_are_bound_and_complete(evidence, change):
    payload = copy.deepcopy(evidence.payloads[6, 12, True])
    overrides = {(6, 12, True): payload}
    if change == "empty":
        overrides = {}
    elif change == "unknown":
        overrides = {(999, 999, True): payload}
    elif change == "mass":
        payload["mu"] = str(float(payload["mu"])+1)
    elif change == "radius":
        payload["rms_run"] *= 2
    elif change == "key":
        payload["Z"] = 7
    elif change == "switch":
        payload["uehling"] = False
    elif change == "missing_A":
        del payload["A"]["L1"]["rmin100"]
    elif change == "missing_B":
        del payload["B"]["220"]
    else:
        payload["B"]["300"]["L1"] = math.nan
    if change in ("empty", "unknown"):
        match = "empty or unknown reference payload override"
    elif change in ("mass", "radius", "key", "switch"):
        match = "unusable reference payload override: reference physical key or physical input mismatch"
    elif change == "missing_A":
        match = "unusable reference payload override: 'rmin100'"
    elif change == "missing_B":
        match = "unusable reference payload override: '220'"
    else:
        match = "unusable reference payload override: nonfinite reference numerical primitive"
    with pytest.raises(d.EvidenceError, match=match):
        d.with_payloads(evidence, overrides)


def test_fresh_payload_overrides_preserve_other_evidence(evidence):
    payload = copy.deepcopy(evidence.payloads[6, 12, True])
    payload["A"]["K1"]["prod"]["B_Ha"] += 1e-12
    fresh = d.with_payloads(evidence, {(6, 12, True): payload})
    assert fresh.payloads[6, 12, True] is payload
    assert fresh.payloads[6, 12, False] is evidence.payloads[6, 12, False]
    assert fresh.cases is evidence.cases and fresh.runs is evidence.runs
    assert fresh.expected is evidence.expected and fresh.targets is evidence.targets
    assert fresh.protocol is evidence.protocol


@pytest.mark.parametrize("role", ["dx1", "dx2", "dx3", "dx4", "tol", "phase1", "phase2",
                                 "phase3", "nu", "ns"])
def test_unusable_required_run_is_adverse(evidence, role):
    changed = copy.deepcopy(evidence)
    changed.runs[6, 12, True, role]["output"]["GE"] = "FAILED"
    with pytest.raises(d.EvidenceError, match="unusable required run"):
        d.evaluate_all(ROOT, changed, members={(6, 12)})


@pytest.mark.parametrize("name,label", [
    ("cases.json", "case"), ("payloads.jsonl", "payload"), ("runs.jsonl", "run"),
])
def test_loader_rejects_duplicate_primitive_record_keys(tmp_path, name, label):
    base = copy_bundle(tmp_path)
    path = base / name
    if name == "cases.json":
        value = json.loads(path.read_text(encoding="ascii"))
        value["cases"].append(copy.deepcopy(value["cases"][0]))
        path.write_text(json.dumps(value) + "\n", encoding="ascii")
    else:
        original = path.read_bytes()
        path.write_bytes(original + original.splitlines(keepends=True)[0])
    # Pass the provenance check so only the duplicate-key validator can refuse this input.
    protocol = json.loads((base / "protocol.json").read_text(encoding="ascii"))
    protocol["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (base / "protocol.json").write_text(json.dumps(protocol), encoding="ascii")
    with pytest.raises(d.EvidenceError, match=f"duplicate {label} key"):
        d.load_evidence(tmp_path)
