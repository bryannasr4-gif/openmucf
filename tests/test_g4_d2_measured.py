"""The measured oxide capture vectors: the committed files are their recomputation, the decisions are the
shared rule's, and every input the generator reads refuses what it must.

`openmucf/g4/d2_measured.py` turns each Stanislaus et al. (1987) Table 1 row into a terminal vector, decides
precision and improvement through `openmucf/g4/science_decisions.py`, and writes the tables Geant4 reads.
These tests re-derive the numbers by other means (floating-point exponentials, exact rationals), and drill
each loader and check with hostile input.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
import pathlib
from fractions import Fraction

import pytest

from openmucf.g4 import d1_evaluated as ev
from openmucf.g4 import d2_measured as dm
from openmucf.g4 import provenance, spec
from openmucf.g4 import science_decisions as sd

REPO = pathlib.Path(__file__).resolve().parents[1]
DIRECTORY = REPO / "data" / "g4" / "d2" / "measured"
MEASUREMENTS = REPO / "data" / "g4" / "reference" / "d2" / "measurements.csv"
PRINTED = REPO / "data" / "g4" / "reference" / "d2" / "printed_rows.csv"
SHARED = REPO / "data" / "g4" / "d1" / "bound_decay_response_v1.json"
INPUTS = DIRECTORY / dm.INPUTS_NAME


def _generator():
    path = REPO / "scripts" / "generate_g4d2_measured.py"
    spec_ = importlib.util.spec_from_file_location("generate_g4d2_measured", path)
    assert spec_ and spec_.loader
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


def generate(measurements=None, printed=None, inputs=None, shared=None) -> dict[str, bytes]:
    return dm.generate(
        MEASUREMENTS.read_bytes() if measurements is None else measurements,
        PRINTED.read_bytes() if printed is None else printed,
        INPUTS.read_bytes() if inputs is None else inputs,
        SHARED.read_bytes() if shared is None else shared,
    )


@pytest.fixture(scope="module")
def evaluation() -> dict:
    return json.loads((DIRECTORY / dm.EVALUATION_NAME).read_bytes())


def inputs_obj() -> dict:
    return json.loads(INPUTS.read_bytes())


def dumped(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8")


# --------------------------------------------------------------------------------------------
# the committed directory is its recomputation
# --------------------------------------------------------------------------------------------


def test_the_committed_directory_and_page_are_their_recomputation():
    _generator().check(REPO)


def test_the_roster_is_every_printed_row_of_the_source():
    """The rows are exactly the `Present measurement` cells of Table 1 that the transcription carries."""
    rows = dm.load_source_rows(MEASUREMENTS.read_bytes(), PRINTED.read_bytes())
    printed = [
        line.split(",")
        for line in PRINTED.read_text(encoding="utf-8").splitlines()[1:]
        if line.startswith(dm.SOURCE_ID + ",") and ",data," in line and "Present measurement" in line
    ]
    assert len(rows) == len(printed) > 0
    assert {r.formula for r in rows} == {p[6] for p in printed}


def test_every_vector_is_the_exact_transform_of_its_row(evaluation):
    """p_M = m A / (n + m A) exactly; the written doubles are within 1e-12 relative of it; the stock value is
    m Z / (m Z + 0.56 * 8 n); the stock value lies above the entry on every row."""
    for r in evaluation["rows"]:
        a = Fraction(r["A"])
        p = r["m"] * a / (r["n"] + r["m"] * a)
        assert abs(Fraction(r["delivery"]["p_metal"]) - p) < Fraction(1, 10**19)
        assert abs(float(r["delivery"]["p_metal_binary64"]) - float(p)) <= 1e-12 * float(p)
        assert abs(float(r["delivery"]["p_oxygen_binary64"]) - float(1 - p)) <= 1e-12 * float(1 - p)
        s = Fraction(r["m"] * r["Z"]) / (r["m"] * r["Z"] + Fraction(448, 100) * r["n"])
        assert abs(Fraction(r["stock_p_metal"]) - s) < Fraction(1, 10**19)
        assert s > p


def test_the_precision_bound_is_recomputed_in_floating_point(evaluation):
    """The lower bound on U is the larger end of |p - q(A')| over A' in [(A - u) e^-0.04, (A + u) e^0.04];
    in binary64 with the platform's exp it agrees to 1e-12, and the committed decimal is not above it."""
    for r in evaluation["rows"]:
        a, u, m, n = float(r["A"]), float(r["u"]), r["m"], r["n"]

        def q(x: float, m=m, n=n) -> float:
            return m * x / (n + m * x)

        p = q(a)
        u_float = max(p - q((a - u) * math.exp(-0.04)), q((a + u) * math.exp(0.04)) - p)
        committed = float(r["precision"]["U_at_least"])
        assert committed <= u_float * (1 + 1e-12)
        assert math.isclose(committed, u_float, rel_tol=1e-11)


def test_precision_excludes_meets_on_every_row_and_decides_nothing(evaluation):
    """Each row's evidence is partial: L <= 0 at the source centre and U >= the nested bound. Every bound
    exceeds 0.01, so MEETS is excluded on every row; no row is MEETS or DOES_NOT_MEET."""
    rows = evaluation["rows"]
    for r in rows:
        again = sd.precision(sd.PartialLoss(0.0, float(r["precision"]["U_at_least"])), 0.01)
        assert r["precision"]["verdict"] == again["verdict"] == sd.NOT_ESTABLISHED
        assert r["precision"]["excluded"] == again["excluded"]
        assert sd.MEETS_EXCLUDED in r["precision"]["excluded"]
    lowest = min(rows, key=lambda r: float(r["precision"]["U_at_least"]))
    assert evaluation["minimum_U_at_least"] == {
        "formula": lowest["formula"],
        "U_at_least": lowest["precision"]["U_at_least"],
    }
    assert evaluation["counts"]["precision_meets_excluded"] == len(rows)


def _parts():
    inputs = dm.load_inputs(INPUTS.read_bytes())
    rows = dm.load_source_rows(MEASUREMENTS.read_bytes(), PRINTED.read_bytes())
    return inputs, rows, dm.bind_source_identity(SHARED.read_bytes(), inputs)


def unscored(inputs: dm.Inputs, **changes) -> dm.Inputs:
    """An altered recipe under an identity no registration carries: evaluated by the numerical API only."""
    return dataclasses.replace(inputs, decision_id=dm.UNSCORED_TEST_PREFIX + "drill", **changes)


def test_the_precision_gate_can_return_its_other_outcome():
    """With a budget above the nested bounds, MEETS is no longer excluded on the rows below it: the exclusion
    follows from the numbers, not from the code path. The count the record carries is the recomputed one."""
    inputs, rows, binding = _parts()
    ev2 = dm.evaluate(rows, unscored(inputs, budget=Fraction(3, 100)), binding)
    excluded = [sd.MEETS_EXCLUDED in r["precision"]["excluded"] for r in ev2["rows"]]
    assert any(excluded) and not all(excluded)
    assert ev2["counts"]["precision_meets_excluded"] == sum(excluded)
    assert all(r["precision"]["verdict"] == sd.NOT_ESTABLISHED for r in ev2["rows"])


@pytest.mark.parametrize(
    "edit",
    [
        lambda o: o["decision_record"].update(budget_tv="0.03"),
        lambda o: o["decision_record"].update(budget_tv="0.010"),  # the same number is accepted: see below
        lambda o: o["decision_record"]["statistics"].update(log_scale="0.002"),
        lambda o: o["decision_record"]["statistics"].update(coverage_k=1),
        lambda o: o["decision_record"].update(publication_resolution="0.0005"),
        lambda o: o["hygroscopic"].update(formulas=["CaO", "Na2O2", "P2O5"]),
        lambda o: o["hygroscopic"].update(formulas=["CaO", "CrO3", "MgO", "Na2O2", "P2O5"]),
        lambda o: o["decision_record"].update(id="d2-measured-regions-v9"),
    ],
)
def test_the_decision_identity_is_bound_to_its_frozen_recipe(edit):
    """Any other budget, statistical scale, coverage, resolution or water roster under the registered
    identity, and any unregistered identity, is refused by the loader with the registration refusal; the
    same numbers written differently are the same recipe."""
    obj = inputs_obj()
    edit(obj)
    if obj["decision_record"]["budget_tv"] == "0.010":
        assert dm.load_inputs(dumped(obj)).budget == Fraction(1, 100)
        return
    with pytest.raises(dm.MeasuredError, match=dm.RECIPE_REFUSAL):
        dm.load_inputs(dumped(obj))
    with pytest.raises(dm.MeasuredError, match=dm.RECIPE_REFUSAL):
        generate(inputs=dumped(obj))


def test_an_altered_recipe_cannot_be_evaluated_under_the_registered_identity():
    inputs, rows, binding = _parts()
    with pytest.raises(dm.MeasuredError, match=dm.RECIPE_REFUSAL):
        dm.evaluate(rows, dataclasses.replace(inputs, stat_log_scale=Fraction(2, 1000)), binding)


def test_the_page_states_only_what_the_record_carries(evaluation):
    """The generated precision sentence is the registered one at the frozen recipe; a record whose numbers or
    identity are not the registered recipe, or that does not exclude MEETS on every row, renders nothing."""
    page = _generator().render_document(evaluation).decode("utf-8")
    required = (
        "Under the registered Cartesian uncertainty construction, the assumed 2% statistical standard "
        "scale at k=2 together with the published carbon sensitivity excludes a TV<=0.01 precision "
        "certificate for every oxide in this table. The result does not establish that every central "
        "estimate is wrong by more than 0.01."
    )
    assert required in " ".join(page.split())
    inputs, rows, binding = _parts()
    altered = dm.evaluate(rows, unscored(inputs, stat_log_scale=Fraction(2, 1000)), binding)
    with pytest.raises(dm.MeasuredError, match=dm.RECIPE_REFUSAL):
        _generator().render_document(altered)
    relabelled = copy.deepcopy(altered)
    relabelled["decision_record"] = evaluation["decision_record"]
    with pytest.raises(dm.MeasuredError, match=dm.RECIPE_REFUSAL):
        _generator().render_document(relabelled)
    partial = copy.deepcopy(evaluation)
    partial["rows"][0]["precision"]["excluded"] = [sd.DOES_NOT_MEET_EXCLUDED]
    with pytest.raises(dm.MeasuredError, match="every oxide"):
        _generator().render_document(partial)


def test_the_generator_check_refuses_a_tampered_page(tmp_path):
    """`check` (not `generate`) compares the committed page with its rendering, through the command line."""
    import shutil

    for rel in ("data/g4/reference/d2", "data/g4/d2/measured"):
        shutil.copytree(REPO / rel, tmp_path / rel)
    (tmp_path / "data/g4/d1").mkdir(parents=True)
    shutil.copyfile(SHARED, tmp_path / "data/g4/d1" / SHARED.name)
    (tmp_path / "docs").mkdir()
    page = (REPO / "docs/measured-oxide-capture.md").read_bytes()
    (tmp_path / "docs/measured-oxide-capture.md").write_bytes(page)
    _generator().main(["check", "--root", str(tmp_path)])  # control
    (tmp_path / "docs/measured-oxide-capture.md").write_bytes(
        page.replace(b"NOT_ESTABLISHED", b"ESTABLISHED", 1)
    )
    with pytest.raises(SystemExit, match="not its rendering"):
        _generator().main(["check", "--root", str(tmp_path)])
    assert (tmp_path / "docs/measured-oxide-capture.md").read_bytes() != page  # check wrote nothing


def test_improvement_is_not_established_on_any_row_and_names_what_is_missing(evaluation):
    hygro = set(inputs_obj()["hygroscopic"]["formulas"])
    for r in evaluation["rows"]:
        assert r["improvement"]["verdict"] == sd.NOT_ESTABLISHED
        assert r["improvement"]["evidence"] == sd.INCOMPLETE
        want = ["E2 response not validated"] + (
            ["water uptake bound unsupported"] if r["formula"] in hygro else []
        )
        assert r["improvement"]["missing"] == want
        assert r["hygroscopic"] == (r["formula"] in hygro)
    assert evaluation["counts"]["improvement_established"] == 0
    assert evaluation["actual_region_drills"]["precision_power"].startswith("NOT_DEMONSTRATED")
    assert evaluation["actual_region_drills"]["real_positive_improvement_control"].startswith("NOT_AVAILABLE")


def test_the_conditional_sensitivity_is_never_a_decision(evaluation):
    """The one decision record comes first and owns every verdict; the conditional sensitivity names its rows
    and carries its disclosure, no verdict and no number, and changes no row."""
    attempts = evaluation["attempts"]
    assert attempts[0] == {
        "id": evaluation["decision_record"],
        "role": dm.ROLE_DECISION,
        "note": "the decision record: every verdict in rows[] is this attempt's",
    }
    sens = [a for a in attempts if a["role"] == dm.ROLE_SENSITIVITY]
    assert len(sens) == 1 and "verdict" not in json.dumps(sens) and sd.ESTABLISHED not in json.dumps(sens)
    assert set(sens[0]) == {"id", "role", "statement", "disclosure", "rows", "qualifications"}
    without = inputs_obj()
    without["conditional_sensitivities"] = []
    ev2 = json.loads(generate(inputs=dumped(without))[dm.EVALUATION_NAME])
    assert ev2["rows"] == evaluation["rows"]


@pytest.mark.parametrize("text", ["ESTABLISHED for Al2O3", "verdict: positive"])
def test_a_sensitivity_carrying_a_verdict_is_refused(text):
    obj = inputs_obj()
    obj["conditional_sensitivities"][0]["statement"] = text
    with pytest.raises(dm.MeasuredError, match="carries no verdict"):
        dm.load_inputs(dumped(obj))


def test_a_later_attempt_is_registered_unscored_and_changes_no_decision(evaluation):
    obj = inputs_obj()
    obj["later_attempts"] = [{"id": "d2-measured-v3", "status": "NOT_SCORED", "note": "pending"}]
    ev2 = json.loads(generate(inputs=dumped(obj))[dm.EVALUATION_NAME])
    assert ev2["rows"] == evaluation["rows"]
    assert ev2["attempts"][-1] == {
        "id": "d2-measured-v3",
        "role": dm.ROLE_LATER,
        "status": "NOT_SCORED",
        "note": "pending",
    }
    obj["later_attempts"][0]["status"] = "ESTABLISHED"
    with pytest.raises(dm.MeasuredError, match="NOT_SCORED"):
        dm.load_inputs(dumped(obj))


# --------------------------------------------------------------------------------------------
# the shared record, read in place
# --------------------------------------------------------------------------------------------


def shared_obj() -> dict:
    return json.loads(SHARED.read_bytes())


def resealed(obj: dict) -> bytes:
    obj = copy.deepcopy(obj)
    obj["payload_sha256"] = ev.payload_digest(obj["payload"])
    return dumped(obj)


def test_the_rows_bind_to_the_paper_identity_of_the_committed_record(evaluation):
    record = shared_obj()
    assert evaluation["shared_record"] == {
        "id": record["id"],
        "payload_sha256": record["payload_sha256"],
        "source_identity": dm.SOURCE_IDENTITY,
        "source_identity_status": record["payload"]["source_response"][dm.SOURCE_IDENTITY]["status"],
    }
    assert {r["source_identity"] for r in evaluation["rows"]} == {dm.SOURCE_IDENTITY}


def test_a_paper_identity_aliased_to_the_thesis_is_refused_even_when_resealed():
    obj = shared_obj()
    sr = obj["payload"]["source_response"]
    sr[dm.SOURCE_IDENTITY] = copy.deepcopy(sr[dm.THESIS_IDENTITY])
    with pytest.raises(dm.MeasuredError):
        generate(shared=resealed(obj))
    obj["payload_sha256"] = ev.payload_digest(obj["payload"])
    obj2 = inputs_obj()
    obj2["shared_record"]["payload_sha256"] = obj["payload_sha256"]
    with pytest.raises(dm.MeasuredError, match="thesis|status"):
        generate(inputs=dumped(obj2), shared=dumped(obj))


def _with_shared(obj: dict) -> tuple[bytes, bytes]:
    """The record resealed, and inputs naming its new digest: only the identity rules can refuse it."""
    obj["payload_sha256"] = ev.payload_digest(obj["payload"])
    inp = inputs_obj()
    inp["shared_record"]["payload_sha256"] = obj["payload_sha256"]
    return dumped(inp), dumped(obj)


def test_identical_paper_and_thesis_identities_are_refused():
    """Both identities the same object, each claiming nothing resolved: the paper is not distinct."""
    obj = shared_obj()
    sr = obj["payload"]["source_response"]
    sr[dm.THESIS_IDENTITY] = {"q_formula": "UNRESOLVED", "status": "UNRESOLVED"}
    sr[dm.SOURCE_IDENTITY] = {"q_formula": "UNRESOLVED", "status": "UNRESOLVED"}
    inp, shared = _with_shared(obj)
    with pytest.raises(dm.MeasuredError, match="differ"):
        generate(inputs=inp, shared=shared)


def test_an_evaluator_assumed_paper_identity_must_state_its_assumption():
    """A paper identity may be explicitly evaluator-assumed only with the assumption and the treatment of its
    discrepancy written down; with them it binds (positive control), without either it is refused."""
    obj = shared_obj()
    paper = obj["payload"]["source_response"][dm.SOURCE_IDENTITY]
    paper.update(
        status="EVALUATOR_ASSUMED",
        q_formula="1 - 0.5 (Z/137)^2",
        assumption="the thesis formula",
        discrepancy="not established for the paper",
    )
    inp, shared = _with_shared(copy.deepcopy(obj))
    ev2 = json.loads(generate(inputs=inp, shared=shared)[dm.EVALUATION_NAME])
    assert ev2["shared_record"]["source_identity_status"] == "EVALUATOR_ASSUMED"
    for key in ("assumption", "discrepancy"):
        bad = copy.deepcopy(obj)
        del bad["payload"]["source_response"][dm.SOURCE_IDENTITY][key]
        inp, shared = _with_shared(bad)
        with pytest.raises(dm.MeasuredError, match="assumption"):
            generate(inputs=inp, shared=shared)


@pytest.mark.parametrize(
    "edit",
    [
        lambda o: o["payload"]["definitions"].update(free_tau_ns="2197.0"),
        lambda o: o["payload"]["source_response"]["d2_response"].update(delivery_mode="EVALUATED"),
        lambda o: o["payload"]["source_response"].pop(dm.SOURCE_IDENTITY),
        lambda o: o.update(id="bound-decay-response-v2"),
    ],
)
def test_a_changed_shared_record_is_refused(edit):
    """One unit changing the record (or its digest) without the other: the payload digest no longer matches
    the one this evaluation consumed, or the paper identity, delivery mode or record identity is wrong."""
    obj = shared_obj()
    edit(obj)
    with pytest.raises(dm.MeasuredError):
        generate(shared=resealed(obj))
    with pytest.raises(dm.MeasuredError):
        generate(shared=dumped(obj))


def test_a_reinterpretation_of_a_reported_value_is_refused(evaluation):
    assert dm.delivered_vector(evaluation, "Al2O3") == {13: 47 / 122, 8: 1 - 47 / 122}
    with pytest.raises(dm.ConversionRefused) as exc:
        dm.delivered_vector(evaluation, "Al2O3", "EVALUATED_E2_RESPONSE")
    assert exc.value.code == dm.CONVERSION_REFUSAL == "CONVERSION_NOT_ESTABLISHED"


# --------------------------------------------------------------------------------------------
# the tables Geant4 reads
# --------------------------------------------------------------------------------------------


def test_every_table_parses_and_its_provenance_is_bound_to_it(evaluation):
    names = {r["profile"] for r in evaluation["rows"]}
    assert {p.name for p in DIRECTORY.glob("*.g4dat")} == {f"{n}.g4dat" for n in names}
    for r in evaluation["rows"]:
        raw = (DIRECTORY / f"{r['profile']}.prov.json").read_bytes()
        table = spec.parse((DIRECTORY / f"{r['profile']}.g4dat").read_text(encoding="ascii"))
        provenance.check_canonical_bytes(raw)
        provenance.check_source_digest(table, raw)
        provenance.check_against_table(table, provenance.from_json_obj(json.loads(raw)))
        by_z = {int(rec[0]): (rec[1], rec[2]) for rec in table.records}
        assert by_z == {
            r["Z"]: (float(r["delivery"]["p_metal_binary64"]), float(r["m"])),
            8: (float(r["delivery"]["p_oxygen_binary64"]), float(r["n"])),
        }
        assert spec.validity_assignments(table)["material"] == r["formula"]


def test_the_inventory_lists_every_file_with_its_digest():
    inventory = json.loads((DIRECTORY / dm.INVENTORY_NAME).read_bytes())
    files = {p.name for p in DIRECTORY.iterdir() if p.is_file()} - {dm.INVENTORY_NAME}
    assert set(inventory["files"]) == files
    import hashlib

    for name, rec in inventory["files"].items():
        raw = (DIRECTORY / name).read_bytes()
        assert rec == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


# --------------------------------------------------------------------------------------------
# hostile input
# --------------------------------------------------------------------------------------------


def _committed() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in DIRECTORY.iterdir() if p.is_file()}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(
            {"stanislaus1987-al2o3.g4dat": c["stanislaus1987-al2o3.g4dat"].replace(b"0.38", b"0.39")}
        ),
        lambda c: c.pop("stanislaus1987-pbo.prov.json"),
        lambda c: c.update({"extra.g4dat": b"#GRAMMAR 1.0\n"}),
        lambda c: c.update(
            {dm.EVALUATION_NAME: c[dm.EVALUATION_NAME].replace(b'"NOT_ESTABLISHED"', b'"ESTABLISHED"', 1)}
        ),
        lambda c: c.update({dm.EVALUATION_NAME: c[dm.EVALUATION_NAME].replace(b"\n", b"\r\n")}),
        lambda c: c.update({dm.INVENTORY_NAME: c[dm.INVENTORY_NAME] + b" "}),
    ],
)
def test_a_tampered_committed_directory_is_refused(mutate):
    committed = _committed()
    mutate(committed)
    with pytest.raises(dm.MeasuredError):
        dm.check_committed(committed, generate())


def test_a_resealed_lie_is_refused():
    """An evaluation whose verdict is changed and whose inventory digest is recomputed to match is still not
    its recomputation from the source rows."""
    import hashlib

    committed = _committed()
    obj = json.loads(committed[dm.EVALUATION_NAME])
    obj["rows"][2]["improvement"]["verdict"] = sd.ESTABLISHED
    committed[dm.EVALUATION_NAME] = dm.canonical_bytes(obj)
    inv = json.loads(committed[dm.INVENTORY_NAME])
    inv["files"][dm.EVALUATION_NAME] = {
        "bytes": len(committed[dm.EVALUATION_NAME]),
        "sha256": hashlib.sha256(committed[dm.EVALUATION_NAME]).hexdigest(),
    }
    committed[dm.INVENTORY_NAME] = dm.canonical_bytes(inv)
    with pytest.raises(dm.MeasuredError, match="recomputation"):
        dm.check_committed(committed, generate())


def _measurement_lines() -> list[str]:
    return MEASUREMENTS.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def _al_line(lines: list[str]) -> int:
    return next(i for i, line in enumerate(lines) if line.startswith("Stanislaus1987a-Table 1-17-5,"))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (",0.94,0.03,", ",nan,0.03,"),
        (",0.94,0.03,", ",inf,0.03,"),
        (",0.94,0.03,", ",-inf,0.03,"),
        (",0.94,0.03,", ",,0.03,"),
        (",0.94,0.03,", ",True,0.03,"),
        (",0.94,0.03,", ",0.940,0.03,"),
        (",0.94,0.03,", ",0.94,0.00,"),
        (",0.94,0.03,", ",0.94,-0.03,"),
        (",0.94,0.03,", ",0.95,0.03,"),
        ("A(Al/O)", "A(Si/O)"),
        (",atomic_capture_ratio,", ",capture_rate,"),
        (",Al2O3,solid,", ",Al2O3H,solid,"),
    ],
)
def test_a_hostile_measurement_row_is_refused(old, new):
    lines = _measurement_lines()
    i = _al_line(lines)
    assert old in lines[i]
    lines[i] = lines[i].replace(old, new, 1)
    with pytest.raises(dm.MeasuredError):
        generate(measurements="\n".join(lines).encode("utf-8"))


@pytest.mark.parametrize(
    "edit",
    [
        lambda lines: lines.insert(_al_line(lines) + 1, lines[_al_line(lines)]),  # duplicate row
        lambda lines: lines.pop(_al_line(lines)),  # missing row
        lambda lines: lines.__setitem__(0, lines[0] + ",extra"),  # extra header
        lambda lines: lines.__setitem__(
            0, lines[0].replace("reported_value", "reported_uncertainty")
        ),  # duplicate
        lambda lines: lines.__setitem__(0, ",".join(lines[0].split(",")[:-1])),  # dropped column
        lambda lines: [
            lines.pop(i)
            for i in sorted(
                (j for j, line in enumerate(lines) if line.startswith("Stanislaus1987a")), reverse=True
            )
        ],  # empty roster
    ],
)
def test_a_hostile_measurement_file_is_refused(edit):
    lines = _measurement_lines()
    edit(lines)
    with pytest.raises(dm.MeasuredError):
        generate(measurements="\n".join(lines).encode("utf-8"))


def test_a_byte_order_mark_is_refused():
    with pytest.raises(dm.MeasuredError):
        generate(measurements=b"\xef\xbb\xbf" + MEASUREMENTS.read_bytes())


@pytest.mark.parametrize(
    "edit",
    [
        lambda o: o["decision_record"].update(budget_tv="nan"),
        lambda o: o["decision_record"].update(budget_tv="-0.01"),
        lambda o: o["decision_record"].update(budget_tv="0"),
        lambda o: o["decision_record"].update(budget_tv=0.01),
        lambda o: o["decision_record"].update(budget_tv=""),
        lambda o: o["decision_record"]["statistics"].update(coverage_k=True),
        lambda o: o["decision_record"]["statistics"].update(coverage_k=2.0),
        lambda o: o["decision_record"]["statistics"].update(label="MEASURED"),
        lambda o: o["decision_record"]["statistics"].update(log_scale="inf"),
        lambda o: o.update(unexpected=1),
        lambda o: o.pop("hygroscopic"),
        lambda o: o["hygroscopic"].update(formulas=["P2O5", "CaO"]),
        lambda o: o["hygroscopic"].update(formulas=["CaO", "CaO"]),
        lambda o: o["hygroscopic"].update(formulas=["CaO", "H2O"]),
        lambda o: o["shared_record"].update(source_identity="stanislaus1983_thesis"),
        lambda o: o["shared_record"].update(payload_sha256="0" * 64),
        lambda o: o["conditional_sensitivities"][0].update(conditional_min_margin_tv={"Al2O3": "0.0552"}),
        lambda o: o["conditional_sensitivities"][0].update(rows=["SiO", "Al2O3"]),
        lambda o: o["conditional_sensitivities"][0].update(rows=["Al2O3", "Al2O3"]),
        lambda o: o["conditional_sensitivities"][0].update(rows=[]),
        lambda o: o["conditional_sensitivities"][0].update(rows=["Al2O3", "XyO"]),
        lambda o: o["conditional_sensitivities"][0]["qualifications"].update(MgO="x"),
        lambda o: o["conditional_sensitivities"].append(copy.deepcopy(o["conditional_sensitivities"][0])),
        lambda o: o.update(later_attempts={}),
    ],
)
def test_hostile_inputs_are_refused(edit):
    obj = inputs_obj()
    edit(obj)
    with pytest.raises(dm.MeasuredError):
        generate(inputs=dumped(obj))


def test_duplicate_json_keys_are_refused():
    raw = INPUTS.read_bytes().replace(
        b'"source_id": "Stanislaus1987a"', b'"source_id": "Stanislaus1987a", "source_id": "Stanislaus1987a"'
    )
    with pytest.raises(dm.MeasuredError):
        dm.load_inputs(raw)


def test_a_hygroscopic_formula_off_the_roster_is_refused():
    """The roster check at the numerical API (the loader would refuse the altered water roster first)."""
    inputs, rows, binding = _parts()
    with pytest.raises(dm.MeasuredError, match="roster"):
        dm.evaluate(rows, unscored(inputs, hygroscopic=(*inputs.hygroscopic, "Li2O")), binding)


# --------------------------------------------------------------------------------------------
# isolated controls, one guard each
# --------------------------------------------------------------------------------------------


def _edit_both(old: str, new: str) -> tuple[bytes, bytes]:
    """The Al2O3 cell changed consistently in the measurement row and the printed cell."""
    lines = _measurement_lines()
    i = _al_line(lines)
    assert old in lines[i]
    lines[i] = lines[i].replace(old, new, 1)
    printed = PRINTED.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    (j,) = [k for k, line in enumerate(printed) if line.startswith("Stanislaus1987a,") and ",Al2O3," in line]
    a, u = new.strip(",").split(",")
    f = printed[j].split(",")
    f[7], f[8] = a, u
    printed[j] = ",".join(f)
    return "\n".join(lines).encode("utf-8"), "\n".join(printed).encode("utf-8")


@pytest.mark.parametrize(
    ("new", "guard"),
    [
        (",0.94,0.00,", "u positive"),
        (",0.03,0.03,", "core lower end"),
    ],
)
def test_a_cell_changed_in_both_files_meets_its_own_guard(new, guard):
    measurements, printed = _edit_both(",0.94,0.03,", new)
    with pytest.raises(dm.MeasuredError):
        generate(measurements=measurements, printed=printed)


def _printed_lines() -> list[str]:
    return PRINTED.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def _al_printed(lines: list[str]) -> int:
    (j,) = [k for k, line in enumerate(lines) if line.startswith("Stanislaus1987a,") and ",Al2O3," in line]
    return j


@pytest.mark.parametrize(
    "extra",
    [
        lambda line: line.replace("Stanislaus1987a,", "Other1990,", 1),  # another source's cell
        lambda line: line.replace(",data,", ",note,", 1),  # not a data row
        lambda line: line.replace("Present measurement", "Egidy(81)", 1).replace(
            ",0.94,", ",0.96,"
        ),  # comparator
    ],
)
def test_cells_outside_the_present_measurement_column_are_ignored(extra):
    lines = _printed_lines()
    lines.insert(_al_printed(lines) + 1, extra(lines[_al_printed(lines)]))
    assert generate(printed="\n".join(lines).encode("utf-8")) == generate()


def test_a_repeated_printed_cell_is_refused():
    lines = _printed_lines()
    lines.insert(_al_printed(lines) + 1, lines[_al_printed(lines)])
    with pytest.raises(dm.MeasuredError, match="repeats"):
        generate(printed="\n".join(lines).encode("utf-8"))


def test_a_printed_cell_without_a_measurement_row_is_refused():
    lines = _printed_lines()
    lines.insert(_al_printed(lines) + 1, lines[_al_printed(lines)].replace(",Al2O3,", ",Li2O,"))
    with pytest.raises(dm.MeasuredError, match="different formulas"):
        generate(printed="\n".join(lines).encode("utf-8"))


def test_an_empty_roster_in_both_files_is_refused():
    m = [x for x in _measurement_lines() if not x.startswith("Stanislaus1987a")]
    pr = [x for x in _printed_lines() if not x.startswith("Stanislaus1987a")]
    with pytest.raises(dm.MeasuredError, match="empty"):
        dm.load_source_rows("\n".join(m).encode("utf-8"), "\n".join(pr).encode("utf-8"))
    inputs, rows, binding = _parts()
    with pytest.raises(dm.MeasuredError, match="empty"):
        dm.evaluate([], inputs, binding)


def test_a_repeated_record_id_and_a_repeated_formula_are_each_refused():
    lines = _measurement_lines()
    i = _al_line(lines)
    sio2 = next(k for k, x in enumerate(lines) if x.startswith("Stanislaus1987a-") and ",SiO2,solid," in x)
    same_id = list(lines)
    same_id[sio2] = lines[i].split(",", 1)[0] + "," + lines[sio2].split(",", 1)[1]
    with pytest.raises(dm.MeasuredError, match="duplicate"):
        generate(measurements="\n".join(same_id).encode("utf-8"))
    same_formula = list(lines)
    same_formula.insert(
        i + 1, lines[i].replace("Stanislaus1987a-Table 1-17-5,", "Stanislaus1987a-Table 1-17-99,", 1)
    )
    with pytest.raises(dm.MeasuredError, match="duplicate"):
        generate(measurements="\n".join(same_formula).encode("utf-8"))


def test_a_short_measurement_record_and_a_liquid_phase_are_refused():
    lines = _measurement_lines()
    i = _al_line(lines)
    short = list(lines)
    short[i] = short[i].rsplit(",", 1)[0]
    with pytest.raises(dm.MeasuredError, match="fields"):
        generate(measurements="\n".join(short).encode("utf-8"))
    liquid = list(lines)
    liquid[i] = liquid[i].replace(",Al2O3,solid,", ",Al2O3,liquid,", 1)
    with pytest.raises(dm.MeasuredError, match="solid"):
        generate(measurements="\n".join(liquid).encode("utf-8"))


@pytest.mark.parametrize(
    "edit",
    [
        lambda o: o.update(schema="openmucf.d2-measured-inputs/0"),
        lambda o: o.update(source_id="Other1990"),
        lambda o: o["decision_record"]["statistics"].update(locator=" leading space"),
        lambda o: o.update(conditional_sensitivities={}),
        lambda o: o["conditional_sensitivities"][0].update(qualifications=[]),
        lambda o: o["conditional_sensitivities"][0].update(rows=["Al2O3", "Li2O", "SiO2"]),
        lambda o: o["hygroscopic"].update(formulas=["Al2Si3", "CaO"]),
    ],
)
def test_more_hostile_inputs_are_refused(edit):
    obj = inputs_obj()
    edit(obj)
    with pytest.raises(dm.MeasuredError):
        generate(inputs=dumped(obj))


def test_inputs_with_cr_or_as_text_are_refused():
    with pytest.raises(dm.MeasuredError, match="CR"):
        dm.load_inputs(INPUTS.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(dm.MeasuredError, match="bytes"):
        dm.load_inputs(INPUTS.read_text(encoding="utf-8"))  # type: ignore[arg-type]
    with pytest.raises(dm.MeasuredError, match="bytes"):
        dm.load_source_rows(MEASUREMENTS.read_text(encoding="utf-8"), PRINTED.read_bytes())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "edit",
    [
        lambda o: o.update(schema="openmucf.bound-decay-response/0"),
        lambda o: o["payload"]["source_response"]["d2_response"].update(delivery_mode="EVALUATED"),
        lambda o: o["payload"]["source_response"]["d2_response"].update(conversion_refusal="CONVERTED"),
        lambda o: o["payload"]["source_response"][dm.SOURCE_IDENTITY].update(q_formula="1 - 0.5 (Z/137)^2"),
        lambda o: o["payload"]["source_response"][dm.SOURCE_IDENTITY].update(
            status="RESOLVED", assumption="a", discrepancy="b"
        ),
    ],
)
def test_a_shared_record_change_is_refused_with_its_digest_registered(edit):
    """The record resealed and the inputs naming its new digest, so only the rule under test can refuse it."""
    obj = shared_obj()
    edit(obj)
    inp, shared = _with_shared(obj)
    with pytest.raises(dm.MeasuredError):
        generate(inputs=inp, shared=shared)


def test_a_stale_payload_digest_field_is_refused():
    obj = shared_obj()
    obj["payload_sha256"] = "0" * 64
    with pytest.raises(dm.MeasuredError, match="digest"):
        generate(shared=dumped(obj))


def test_a_row_whose_entry_is_not_below_stock_is_refused():
    inputs, rows, binding = _parts()
    al = next(r for r in rows if r.formula == "Al2O3")
    high = dataclasses.replace(al, a_text="99.00", u_text="0.03")
    with pytest.raises(dm.MeasuredError, match="stock"):
        dm.evaluate([high], unscored(inputs, hygroscopic=(), sensitivities=()), binding)


def test_the_float_bound_is_the_largest_binary64_not_above_the_decimal():
    from decimal import Decimal

    for text in ("0.1", "0.0144081147435", "0.5"):
        f = dm._float_at_most(text)
        assert Decimal(f) <= Decimal(text) < Decimal(math.nextafter(f, math.inf))
