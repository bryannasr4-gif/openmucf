"""Measured terminal capture vectors for oxides, and the decisions taken against them.

Stanislaus et al., Nucl. Phys. A 475 (1987) 630, Table 1, report for each oxide ``M_mO_n`` the per-atom
atomic capture ratio ``A`` of the metal to oxygen, with an uncertainty the paper states is systematic, from
the sensitivity of ``A`` to the carbon background, and the note that statistical errors are not included
"as they are only about 2%" (p. 636). For each row this module delivers the probability that a stopped
negative muon ends on each element,

    p_M = m A / (n + m A),   p_O = 1 - p_M,

the source-reported value carried as reported (``SOURCE_REPORTED_FROZEN``), and decides, through
:mod:`openmucf.g4.science_decisions`:

* **precision** of that vector, total variation ``|p_M - q_M|`` against the budget 0.01: the source centre is
  an admissible reference point, so ``L = 0``; the statistics (2 per cent, an assumed one-sigma scale, at
  ``k = 2``) together with the printed carbon sensitivity form a sub-region of every completion of the
  registered reference region, and its supremum is the lower bound on ``U`` this module records. Where that
  bound exceeds the budget, ``MEETS`` is excluded; the verdict is ``NOT_ESTABLISHED``. This does not say that
  any central value is wrong by more than the budget.
* **robust improvement** over Geant4's stock selector for the same nominal material, whose metal probability
  is ``m Z / (m Z + 0.56 * 8 n)``: ``NOT_ESTABLISHED``, because the response of the source's decay-electron
  detection to the electron energy (its ``E2`` correction) is not validated, and for the rows the source flags
  as hygroscopic no bound on water uptake is supported. The other components of the region (the time origin,
  the run-to-run difference and the fixed-lifetime template response) are evaluated from inputs this
  repository does not redistribute; they are named, not distributed.

The vectors apply to the hydrogen-free nominal solid of each formula; a material with any other element is
refused by :mod:`openmucf.g4.d2_material`. Nuclear-capture branching and isotope capture preferences are
other observables. A requested reinterpretation of a reported value under another convention refuses with
``CONVERSION_NOT_ESTABLISHED``.

Every number here is computed from the committed source rows in exact rational or 50-digit decimal arithmetic;
the committed evaluation is accepted only if it is byte for byte its recomputation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from typing import Any

from openmucf.g4 import d1_evaluated, provenance, spec
from openmucf.g4 import d2 as selector
from openmucf.g4 import science_decisions as decisions

SOURCE_ID = "Stanislaus1987a"
INPUTS_SCHEMA = "openmucf.d2-measured-inputs/1"
EVALUATION_SCHEMA = "openmucf.d2-measured-evaluation/1"
INVENTORY_SCHEMA = "openmucf.d2-measured-inventory/1"
SEAM = "d2_atomic_capture"
TABLE = "material_vector"
COLUMNS = ("Z", "P", "n")
PROFILE_PREFIX = "stanislaus1987-"
DATASET = "G4MuonicData"
DATASET_VERSION = "0.1.0"
GENERATOR = "openmucf-g4 1.3.0"
VALIDITY = "form:vector basis:atom scope:hydrogen_free_nominal_solid status:SOURCE_REPORTED_PROVISIONAL"
DELIVERY_MODE = "SOURCE_REPORTED_FROZEN"
CONVERSION_REFUSAL = "CONVERSION_NOT_ESTABLISHED"
SOURCE_IDENTITY = "stanislaus1987_paper"
THESIS_IDENTITY = "stanislaus1983_thesis"
ROLE_DECISION, ROLE_SENSITIVITY, ROLE_LATER = "decision_record", "conditional_sensitivity", "later_attempt"
#: Relative tolerance between a written binary64 probability and its exact rational value.
DELIVERY_TOL = Fraction(1, 10**12)
#: Significant digits of the decimal strings the evaluation records.
DIGITS = 20
BOUND_DIGITS = 12
OXYGEN_Z = 8
_AMOUNT = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{2}$")
_DECIMAL = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")
_FORMULA = re.compile(r"([A-Z][a-z]?)([0-9]*)")
_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Recipe:
    """The frozen inputs a registered decision record was decided with."""

    budget: Fraction
    stat_log_scale: Fraction
    coverage_k: int
    resolution: Fraction
    hygroscopic: tuple[str, ...]


#: Every decision record this module may produce, bound to its frozen recipe. A record with another identity,
#: or this identity with other inputs, is not a registered decision: it is refused.
REGISTERED_RECIPES = {
    "d2-measured-regions-v2": Recipe(
        budget=Fraction(1, 100),
        stat_log_scale=Fraction(2, 100),
        coverage_k=2,
        resolution=Fraction(5, 1000),
        hygroscopic=("CaO", "CrO3", "Na2O2", "P2O5"),
    ),
}
RECIPE_REFUSAL = (
    "registered decision recipe differs from its frozen inputs; a new attempt requires registration"
)
#: An identity no registration can carry: the test suite's numerical drills evaluate altered recipes under it,
#: through :func:`evaluate` only; it never reaches the loader, the generator or the committed record.
UNSCORED_TEST_PREFIX = "unscored-test-"


class MeasuredError(ValueError):
    """An input, a record or a request this module refuses."""


class ConversionRefused(MeasuredError):
    """A reinterpretation of a source-reported value that no source or response input supports."""

    code = CONVERSION_REFUSAL


# --------------------------------------------------------------------------------------------
# typed inputs
# --------------------------------------------------------------------------------------------


def _strict_json(raw: bytes, where: str) -> Any:
    if not isinstance(raw, bytes):
        raise MeasuredError(f"{where}: bytes required")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise MeasuredError(f"{where}: a byte-order mark or CR")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=d1_evaluated._reject_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as exc:
        raise MeasuredError(f"{where}: {exc}") from exc


def _keys(obj: Any, keys: Sequence[str], where: str) -> Mapping[str, Any]:
    if not isinstance(obj, dict) or set(obj) != set(keys):
        got = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise MeasuredError(f"{where}: keys {got} differ from {sorted(keys)}")
    return obj


def _text(x: Any, where: str) -> str:
    if not isinstance(x, str) or not x or x != x.strip():
        raise MeasuredError(f"{where}: a non-empty trimmed string is required")
    return x


def _decimal_text(x: Any, where: str) -> Fraction:
    if not isinstance(x, str) or not _DECIMAL.fullmatch(x):
        raise MeasuredError(f"{where}: {x!r} is not a plain decimal string")
    value = Fraction(x)
    if not value > 0:
        raise MeasuredError(f"{where}: {x!r} is not positive")
    return value


def _int(x: Any, where: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 1:
        raise MeasuredError(f"{where}: {x!r} is not a positive integer")
    return x


def parse_oxide(formula: Any) -> tuple[int, int, int]:
    """``(Z_M, m, n)`` of a binary oxide ``M_mO_n``; anything else is refused."""
    text = _text(formula, "formula")
    parts = _FORMULA.findall(text)
    if "".join(a + b for a, b in parts) != text:
        raise MeasuredError(f"formula {text!r} does not parse")
    counts: dict[int, int] = {}
    for symbol, digits in parts:
        z = selector._SYMBOLS.get(symbol)
        if z is None or z in counts or digits.startswith("0"):
            raise MeasuredError(f"formula {text!r}: element {symbol!r}")
        counts[z] = int(digits) if digits else 1
    if set(counts) - {OXYGEN_Z} == set() or OXYGEN_Z not in counts or len(counts) != 2:
        raise MeasuredError(f"formula {text!r} is not a binary oxide")
    (metal,) = set(counts) - {OXYGEN_Z}
    return metal, counts[metal], counts[OXYGEN_Z]


@dataclass(frozen=True)
class SourceRow:
    record_id: str
    formula: str
    z: int
    m: int
    n: int
    a_text: str
    u_text: str
    locator: str

    @property
    def a(self) -> Fraction:
        return Fraction(self.a_text)

    @property
    def u(self) -> Fraction:
        return Fraction(self.u_text)

    @property
    def symbol(self) -> str:
        return selector._ELEMENT_SYMBOLS[self.z - 1]


def _csv_rows(raw: bytes, columns: Sequence[str], where: str) -> list[dict[str, str]]:
    if not isinstance(raw, bytes) or raw.startswith(b"\xef\xbb\xbf"):
        raise MeasuredError(f"{where}: bytes without a byte-order mark required")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    records = list(csv.reader(io.StringIO(text)))
    if not records or tuple(records[0]) != tuple(columns):
        raise MeasuredError(f"{where}: header differs from the schema")
    out = []
    for i, rec in enumerate(records[1:], 2):
        if len(rec) != len(columns):
            raise MeasuredError(f"{where}: record {i} has {len(rec)} fields")
        out.append(dict(zip(columns, rec, strict=True)))
    return out


PRINTED_COLUMNS = (
    "source_id",
    "table",
    "printed_page",
    "pdf_page",
    "row_ordinal",
    "row_kind",
    "target_raw",
    "value_raw",
    "unc_raw",
    "column_raw",
    "note",
)


def load_source_rows(measurements: bytes, printed: bytes) -> tuple[SourceRow, ...]:
    """The roster: every ``Stanislaus1987a`` row of ``measurements.csv``, each checked against the transcribed
    ``Present measurement`` cell of ``printed_rows.csv``; the two sets must be equal. Sorted by metal Z then
    formula."""
    rows = [
        r
        for r in _csv_rows(measurements, selector.MEASUREMENT_COLUMNS, "measurements")
        if r["source_id"] == SOURCE_ID
    ]
    printed_cells = {}
    for r in _csv_rows(printed, PRINTED_COLUMNS, "printed rows"):
        if (
            r["source_id"] == SOURCE_ID
            and r["row_kind"] == "data"
            and r["column_raw"] == "Present measurement"
        ):
            if r["target_raw"] in printed_cells:
                raise MeasuredError(f"printed rows: {r['target_raw']} repeats")
            printed_cells[r["target_raw"]] = (r["value_raw"], r["unc_raw"])
    out, ids, formulas = [], set(), set()
    for r in rows:
        z, m, n = parse_oxide(r["material_formula"])
        a, u = r["reported_value"], r["reported_uncertainty"]
        if not (_AMOUNT.fullmatch(a) and _AMOUNT.fullmatch(u)) or not Fraction(a) > 0 or not Fraction(u) > 0:
            raise MeasuredError(f"{r['record_id']}: A {a!r} and u {u!r} must be positive two-decimal amounts")
        if Fraction(a) - Fraction(u) - Fraction(5, 1000) <= 0:
            raise MeasuredError(f"{r['record_id']}: the core's lower end is not positive")
        if r["observable"] != "atomic_capture_ratio" or r["phase"] != "solid":
            raise MeasuredError(f"{r['record_id']}: not a solid-target atomic capture ratio")
        symbol = selector._ELEMENT_SYMBOLS[z - 1]
        if r["inferable_parameter"] != f"A({symbol}/O)":
            raise MeasuredError(f"{r['record_id']}: inferable parameter {r['inferable_parameter']!r}")
        if r["record_id"] in ids or r["material_formula"] in formulas:
            raise MeasuredError(f"{r['record_id']}: duplicate record or formula")
        ids.add(r["record_id"])
        formulas.add(r["material_formula"])
        if printed_cells.get(r["material_formula"]) != (a, u):
            raise MeasuredError(f"{r['record_id']}: differs from the printed cell")
        out.append(SourceRow(r["record_id"], r["material_formula"], z, m, n, a, u, r["locator"]))
    if not out:
        raise MeasuredError("the roster is empty")
    if set(printed_cells) != formulas:
        raise MeasuredError("the printed cells and the measurement rows name different formulas")
    return tuple(sorted(out, key=lambda r: (r.z, r.formula)))


@dataclass(frozen=True)
class Sensitivity:
    id: str
    statement: str
    disclosure: str
    rows: tuple[str, ...]
    margins: tuple[tuple[str, str], ...]
    qualifications: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Inputs:
    decision_id: str
    budget: Fraction
    stat_log_scale: Fraction
    coverage_k: int
    resolution: Fraction
    statistics_locator: str
    carbon_locator: str
    hygroscopic_locator: str
    hygroscopic: tuple[str, ...]
    shared_path: str
    shared_id: str
    shared_payload_sha256: str
    sensitivities: tuple[Sensitivity, ...]
    later_attempts: tuple[tuple[str, str], ...]


_INPUT_KEYS = (
    "schema",
    "source_id",
    "decision_record",
    "hygroscopic",
    "shared_record",
    "conditional_sensitivities",
    "later_attempts",
)
_DECISION_KEYS = ("id", "budget_tv", "statistics", "carbon_locator", "publication_resolution")
_STAT_KEYS = ("log_scale", "coverage_k", "label", "locator")
_HYGRO_KEYS = ("locator", "formulas")
_SHARED_KEYS = ("path", "id", "payload_sha256", "source_identity")
#: A conditional sensitivity names its rows and its conditional minimum margin in total variation on each.
_SENS_KEYS = ("id", "statement", "disclosure", "rows", "conditional_min_margin_tv", "qualifications")


def load_inputs(raw: bytes) -> Inputs:
    obj = _keys(_strict_json(raw, "inputs"), _INPUT_KEYS, "inputs")
    if obj["schema"] != INPUTS_SCHEMA or obj["source_id"] != SOURCE_ID:
        raise MeasuredError("inputs: schema or source")
    d = _keys(obj["decision_record"], _DECISION_KEYS, "decision_record")
    st = _keys(d["statistics"], _STAT_KEYS, "statistics")
    if st["label"] != "ASSUMED_K1":
        raise MeasuredError("statistics: the scale is an assumed one-sigma value")
    h = _keys(obj["hygroscopic"], _HYGRO_KEYS, "hygroscopic")
    if not isinstance(h["formulas"], list) or h["formulas"] != sorted(set(h["formulas"])):
        raise MeasuredError("hygroscopic: a sorted list of distinct formulas")
    for f in h["formulas"]:
        parse_oxide(f)
    sh = _keys(obj["shared_record"], _SHARED_KEYS, "shared_record")
    if not isinstance(sh["payload_sha256"], str) or not _SHA.fullmatch(sh["payload_sha256"]):
        raise MeasuredError("shared_record: payload digest")
    if sh["source_identity"] != SOURCE_IDENTITY:
        raise MeasuredError(f"shared_record: the rows bind to {SOURCE_IDENTITY}")
    sens = []
    if not isinstance(obj["conditional_sensitivities"], list):
        raise MeasuredError("conditional_sensitivities: a list")
    for i, s in enumerate(obj["conditional_sensitivities"]):
        s = _keys(s, _SENS_KEYS, f"conditional_sensitivities[{i}]")
        names = s["rows"]
        quals = s["qualifications"]
        if (
            not isinstance(names, list)
            or not names
            or names != sorted(set(names))
            or not isinstance(quals, dict)
        ):
            raise MeasuredError("a sensitivity lists its rows, sorted and distinct")
        for f in names:
            parse_oxide(f)
        margins = s["conditional_min_margin_tv"]
        if not isinstance(margins, dict) or set(margins) != set(names):
            raise MeasuredError("a sensitivity gives one margin for each row it lists")
        for f, v in margins.items():
            _decimal_text(v, f"sensitivity margin {f}")
        if not set(quals) <= set(names):
            raise MeasuredError("a qualification names a row the sensitivity lists")
        text = json.dumps(s)
        if decisions.ESTABLISHED in text or "verdict" in text:
            raise MeasuredError("a conditional sensitivity carries no verdict")
        sens.append(
            Sensitivity(
                _text(s["id"], "sensitivity id"),
                _text(s["statement"], "statement"),
                _text(s["disclosure"], "disclosure"),
                tuple(names),
                tuple(sorted((f, v) for f, v in margins.items())),
                tuple(sorted((f, _text(v, "qualification")) for f, v in quals.items())),
            )
        )
    if not isinstance(obj["later_attempts"], list):
        raise MeasuredError("later_attempts: a list")
    later = []
    for i, a in enumerate(obj["later_attempts"]):
        a = _keys(a, ("id", "status", "note"), f"later_attempts[{i}]")
        # A later attempt is registered here before it is scored; scoring one is new code with its own
        # evaluation, never a field of this record.
        if a["status"] != "NOT_SCORED":
            raise MeasuredError("a later attempt is carried only as NOT_SCORED")
        later.append((_text(a["id"], "later attempt id"), _text(a["note"], "later attempt note")))
    ids = [d["id"]] + [s.id for s in sens] + [i for i, _ in later]
    if len(set(ids)) != len(ids):
        raise MeasuredError("attempt identities repeat")
    inputs = Inputs(
        decision_id=_text(d["id"], "decision id"),
        budget=_decimal_text(d["budget_tv"], "budget"),
        stat_log_scale=_decimal_text(st["log_scale"], "log scale"),
        coverage_k=_int(st["coverage_k"], "coverage k"),
        resolution=_decimal_text(d["publication_resolution"], "publication resolution"),
        statistics_locator=_text(st["locator"], "statistics locator"),
        carbon_locator=_text(d["carbon_locator"], "carbon locator"),
        hygroscopic_locator=_text(h["locator"], "hygroscopic locator"),
        hygroscopic=tuple(h["formulas"]),
        shared_path=_text(sh["path"], "shared path"),
        shared_id=_text(sh["id"], "shared id"),
        shared_payload_sha256=sh["payload_sha256"],
        sensitivities=tuple(sens),
        later_attempts=tuple(later),
    )
    check_recipe(inputs)
    return inputs


def check_recipe(inputs: Inputs) -> None:
    """The decision identity must be registered, and its inputs must be its frozen recipe."""
    recipe = REGISTERED_RECIPES.get(inputs.decision_id)
    if recipe != Recipe(
        inputs.budget, inputs.stat_log_scale, inputs.coverage_k, inputs.resolution, inputs.hygroscopic
    ):
        raise MeasuredError(f"{inputs.decision_id}: {RECIPE_REFUSAL}")


@dataclass(frozen=True)
class SourceBinding:
    record_id: str
    payload_sha256: str
    identity: str
    status: str
    delivery_mode: str
    conversion_refusal: str


def bind_source_identity(raw: bytes, inputs: Inputs) -> SourceBinding:
    """The historical rows bind to the paper's identity in the shared record, read in place. The record's
    payload digest is recomputed and must be the one the inputs name; the paper and thesis identities must
    both exist and differ, and the paper's must not claim the thesis formula as resolved."""
    obj = _keys(
        _strict_json(raw, "shared record"),
        ("id", "payload", "payload_sha256", "schema", "status"),
        "shared record",
    )
    if obj["id"] != inputs.shared_id or obj["schema"] != d1_evaluated.CONVENTION_SCHEMA:
        raise MeasuredError("shared record: identity or schema")
    digest = d1_evaluated.payload_digest(obj["payload"])
    if digest != obj["payload_sha256"] or digest != inputs.shared_payload_sha256:
        raise MeasuredError("shared record: payload digest differs from its recomputation or the inputs")
    sr = obj["payload"]["source_response"]
    paper, thesis = sr.get(SOURCE_IDENTITY), sr.get(THESIS_IDENTITY)
    if not isinstance(paper, dict) or not isinstance(thesis, dict) or paper == thesis:
        raise MeasuredError("shared record: the paper and thesis identities must both exist and differ")
    status = paper.get("status")
    if status == "UNRESOLVED":
        if paper.get("q_formula") != "UNRESOLVED":
            raise MeasuredError("shared record: an unresolved paper identity carries no formula")
    elif status == "EVALUATOR_ASSUMED":
        if not (isinstance(paper.get("assumption"), str) and isinstance(paper.get("discrepancy"), str)):
            raise MeasuredError(
                "shared record: an assumed paper identity states its assumption and discrepancy"
            )
    else:
        raise MeasuredError(f"shared record: the paper identity has status {status!r}")
    resp = sr.get("d2_response")
    if (
        not isinstance(resp, dict)
        or resp.get("delivery_mode") != DELIVERY_MODE
        or resp.get("conversion_refusal") != CONVERSION_REFUSAL
    ):
        raise MeasuredError("shared record: the D2 delivery mode or conversion refusal")
    return SourceBinding(
        obj["id"], digest, SOURCE_IDENTITY, paper["status"], DELIVERY_MODE, CONVERSION_REFUSAL
    )


# --------------------------------------------------------------------------------------------
# the transform and the decisions
# --------------------------------------------------------------------------------------------


def entry_probability(m: int, n: int, a: Fraction) -> Fraction:
    """``p_M = m A / (n + m A)``, exact."""
    return m * a / (n + m * a)


def stock_probability(z: int, m: int, n: int) -> Fraction:
    """The stock selector's nominal metal probability, ``m Z / (m Z + 0.56 * 8 n)``, exact."""
    return Fraction(m * z) / (m * z + Fraction(56, 100) * OXYGEN_Z * n)


def _dec(x: Fraction | Decimal, digits: int = DIGITS, rounding: str = ROUND_HALF_EVEN) -> str:
    with localcontext() as ctx:
        ctx.prec = 60
        d = x if isinstance(x, Decimal) else Decimal(x.numerator) / Decimal(x.denominator)
        if d == 0:
            return "0"
        exp = d.adjusted() - digits + 1
        return format(d.quantize(Decimal(1).scaleb(exp), rounding=rounding), "f")


def _plain(x: Fraction) -> str:
    """A terminating decimal fraction written exactly, without trailing zeros."""
    with localcontext() as ctx:
        ctx.prec = 60
        d = (Decimal(x.numerator) / Decimal(x.denominator)).normalize()
    if Fraction(d) != x:
        raise MeasuredError(f"{x} is not a terminating decimal")
    return format(d, "f")


def _float_at_most(text: str) -> float:
    """The largest binary64 not above the decimal ``text``."""
    f = float(text)
    if Decimal(f) > Decimal(text):
        f = math.nextafter(f, -math.inf)
    return f


def nested_upper(row: SourceRow, inputs: Inputs, resolution: bool) -> Decimal:
    """The supremum of ``|p_M - q_M|`` over the statistics-and-carbon sub-region (``resolution`` adds the
    publication resolution of the two-decimal value): ``q(A)`` is increasing, so it is at an end."""
    with localcontext() as ctx:
        ctx.prec = 50
        k = Decimal(inputs.coverage_k) * (
            Decimal(inputs.stat_log_scale.numerator) / Decimal(inputs.stat_log_scale.denominator)
        )
        r = Decimal(inputs.resolution.numerator) / Decimal(inputs.resolution.denominator) if resolution else 0
        a, u = Decimal(row.a_text), Decimal(row.u_text)
        lo, hi = (a - u - r) * (-k).exp(), (a + u + r) * k.exp()
        p = Decimal(row.m) * a / (row.n + row.m * a)

        def q(x: Decimal) -> Decimal:
            return row.m * x / (row.n + row.m * x)

        return max(p - q(lo), q(hi) - p)


def evaluate(rows: Sequence[SourceRow], inputs: Inputs, binding: SourceBinding) -> dict[str, Any]:
    """The evaluation record, every field derived from the rows, the inputs and the shared-record binding.
    A registered identity is decided only with its frozen recipe; an identity with the unscored-test prefix
    lets a drill evaluate an altered recipe, and no such record can be loaded, generated or committed."""
    if not inputs.decision_id.startswith(UNSCORED_TEST_PREFIX):
        check_recipe(inputs)
    if not rows:
        raise MeasuredError("the roster is empty")
    formulas = {r.formula for r in rows}
    if not set(inputs.hygroscopic) <= formulas:
        raise MeasuredError("a hygroscopic formula is not on the roster")
    for sens in inputs.sensitivities:
        if not set(sens.rows) <= formulas:
            raise MeasuredError("a sensitivity names a formula not on the roster")
    budget = float(inputs.budget)
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        p = entry_probability(r.m, r.n, r.a)
        stock = stock_probability(r.z, r.m, r.n)
        if not stock > p:
            raise MeasuredError(f"{r.formula}: the stock value is not above the entry")
        p64 = float(p)
        o64 = float(1 - p)
        for exact, written in ((p, p64), (1 - p, o64)):
            if abs(Fraction(written) - exact) > DELIVERY_TOL * exact:
                raise MeasuredError(f"{r.formula}: binary64 delivery error")
        u_nested = _dec(nested_upper(r, inputs, False), BOUND_DIGITS, ROUND_DOWN)
        u_core = _dec(nested_upper(r, inputs, True), BOUND_DIGITS, ROUND_DOWN)
        prec = decisions.precision(decisions.PartialLoss(0.0, _float_at_most(u_nested)), budget)
        missing = ["E2 response not validated"]
        if r.formula in inputs.hygroscopic:
            missing.append("water uptake bound unsupported")
        imp = decisions.robust_improvement(decisions.IncompleteMargin(tuple(missing)))
        out_rows.append(
            {
                "formula": r.formula,
                "record_id": r.record_id,
                "Z": r.z,
                "m": r.m,
                "n": r.n,
                "A": r.a_text,
                "u": r.u_text,
                "hygroscopic": r.formula in inputs.hygroscopic,
                "source_identity": binding.identity,
                "delivery": {
                    "mode": binding.delivery_mode,
                    "p_metal": _dec(p),
                    "p_metal_binary64": spec.format_float(p64),
                    "p_oxygen_binary64": spec.format_float(o64),
                },
                "stock_p_metal": _dec(stock),
                "nominal_margin_tv": _dec(stock - p),
                "precision": {
                    "verdict": prec["verdict"],
                    "evidence": prec["evidence"],
                    "L_at_most": "0",
                    "U_at_least": u_nested,
                    "U_at_least_with_resolution": u_core,
                    "excluded": prec["excluded"],
                },
                "improvement": {
                    "verdict": imp["verdict"],
                    "evidence": imp["evidence"],
                    "missing": imp["missing"],
                },
                "profile": profile(r),
            }
        )
    attempts: list[dict[str, Any]] = [
        {
            "id": inputs.decision_id,
            "role": ROLE_DECISION,
            "note": "the decision record: every verdict in rows[] is this attempt's",
        }
    ]
    for sens in inputs.sensitivities:
        attempts.append(
            {
                "id": sens.id,
                "role": ROLE_SENSITIVITY,
                "statement": sens.statement,
                "disclosure": sens.disclosure,
                "rows": list(sens.rows),
                "conditional_min_margin_tv": dict(sens.margins),
                "qualifications": dict(sens.qualifications),
            }
        )
    for later_id, note in inputs.later_attempts:
        attempts.append({"id": later_id, "role": ROLE_LATER, "status": "NOT_SCORED", "note": note})
    return {
        "schema": EVALUATION_SCHEMA,
        "source_id": SOURCE_ID,
        "decision_record": inputs.decision_id,
        "budget_tv": _plain(inputs.budget),
        "statistics": {
            "log_scale": _plain(inputs.stat_log_scale),
            "coverage_k": inputs.coverage_k,
            "label": "ASSUMED_K1",
            "locator": inputs.statistics_locator,
        },
        "carbon_locator": inputs.carbon_locator,
        "publication_resolution": _plain(inputs.resolution),
        "hygroscopic_locator": inputs.hygroscopic_locator,
        "shared_record": {
            "id": binding.record_id,
            "payload_sha256": binding.payload_sha256,
            "source_identity": binding.identity,
            "source_identity_status": binding.status,
        },
        "components": COMPONENTS,
        "rows": out_rows,
        "counts": {
            "rows": len(out_rows),
            "precision_meets_excluded": sum(
                decisions.MEETS_EXCLUDED in r["precision"]["excluded"] for r in out_rows
            ),
            "improvement_established": sum(
                r["improvement"]["verdict"] == decisions.ESTABLISHED for r in out_rows
            ),
        },
        "minimum_U_at_least": _minimum(out_rows),
        "actual_region_drills": _actual_region_drills(out_rows),
        "attempts": attempts,
    }


def _actual_region_drills(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Whether the drills that need a real row with a complete region could run: a precision-power drill
    needs a complete region, a real positive improvement control needs an established row."""
    complete = [r["formula"] for r in rows if r["precision"]["evidence"] == "complete"]
    established = [r["formula"] for r in rows if r["improvement"]["verdict"] == decisions.ESTABLISHED]
    return {
        "precision_power": "DEMONSTRATED_ON " + ",".join(complete)
        if complete
        else "NOT_DEMONSTRATED: no row's reference region is complete",
        "real_positive_improvement_control": "AVAILABLE " + ",".join(established)
        if established
        else "NOT_AVAILABLE: no row establishes improvement; the drills that need one cannot run on it",
    }


def _minimum(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    best = min(rows, key=lambda r: Decimal(r["precision"]["U_at_least"]))
    return {"formula": best["formula"], "U_at_least": best["precision"]["U_at_least"]}


#: The components of the registered reference region and what this repository carries of each.
COMPONENTS = {
    "statistics_and_carbon": "SUPPORTED: the source's statement, carried here; the nested sub-region of the "
    "precision bound",
    "publication_resolution": "SUPPORTED: half a unit of the printed second decimal",
    "time_origin": "EVALUATED_NOT_DISTRIBUTED: a cross-method regression, conditional on the comparison "
    "method's own systematics",
    "run_to_run": "EVALUATED_NOT_DISTRIBUTED: from repeated measurements of the same formulas",
    "template_response": "EVALUATED_NOT_DISTRIBUTED: the response of the fitted amplitudes to the fixed "
    "lifetimes",
    "electron_detection_E2": "NOT_VALIDATED: the response of the decay-electron detection to the electron "
    "energy",
    "water_uptake": "UNSUPPORTED on the rows the source flags as hygroscopic; not applicable elsewhere",
}


def profile(row: SourceRow) -> str:
    return PROFILE_PREFIX + row.formula.lower()


# --------------------------------------------------------------------------------------------
# the tables Geant4 reads
# --------------------------------------------------------------------------------------------


def table_files(evaluation: Mapping[str, Any]) -> dict[str, bytes]:
    """The Layer-1 and Layer-2 file of each row, named after its profile."""
    out: dict[str, bytes] = {}
    for r in evaluation["rows"]:
        name = r["profile"]
        z = r["Z"]
        doc = provenance.ProvDocument(
            dataset=DATASET,
            version=DATASET_VERSION,
            profile=name,
            seam=SEAM,
            precedence=("openmucf",),
            rows={
                str(key): provenance.ProvRow(
                    source_bibkey=SOURCE_ID,
                    source_locator=f"Nucl. Phys. A 475 (1987) 630, Table 1, row {r['formula']} "
                    f"(A = {r['A']} +/- {r['u']})",
                    unc_type="exp",
                    conditions=f"nominal solid {r['formula']}, hydrogen-free; terminal element of a stopped "
                    "negative muon",
                    validity_range=f"a material whose elements are exactly {r['formula']}'s, in its atom "
                    "ratio",
                    evaluation_method="p_metal = m A / (n + m A), p_O = 1 - p_metal, from the source-"
                    "reported per-atom ratio A (SOURCE_REPORTED_FROZEN); decisions in evaluation.json",
                    single_source=True,
                    needs_verification=False,
                    recommendation="",
                    evaluation_id=evaluation["decision_record"],
                    source_library="openmucf",
                    isotope_resolved=False,
                )
                for key in sorted((OXYGEN_Z, z))
            },
        )
        prov = provenance.document_bytes(doc)
        records = tuple(
            sorted(
                (
                    (OXYGEN_Z, float(r["delivery"]["p_oxygen_binary64"]), float(r["n"])),
                    (z, float(r["delivery"]["p_metal_binary64"]), float(r["m"])),
                )
            )
        )
        table = spec.G4DatTable(
            directives={
                "GRAMMAR": "1.0",
                "DATASET": DATASET,
                "VERSION": DATASET_VERSION,
                "PROFILE": name,
                "SEAM": SEAM,
                "TABLE": TABLE,
                "GENERATOR": GENERATOR,
                "SOURCEDIGEST": provenance.source_digest(prov),
                "UNITS": "P=1 n=1",
                "COLUMNS": " ".join(COLUMNS),
                "VALIDITY": f"{VALIDITY} material:{r['formula']}",
            },
            records=records,
        )
        g4dat = spec.render(table).encode("ascii")
        provenance.check_against_table(spec.parse(g4dat.decode("ascii")), doc)
        out[f"{name}.g4dat"] = g4dat
        out[f"{name}.prov.json"] = prov
    return out


def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode("ascii")


def inventory(files: Mapping[str, bytes]) -> bytes:
    """Every generated file of the directory with its size and SHA-256."""
    return canonical_bytes(
        {
            "schema": INVENTORY_SCHEMA,
            "files": {
                name: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                for name, b in sorted(files.items())
            },
        }
    )


INPUTS_NAME = "inputs.json"
EVALUATION_NAME = "evaluation.json"
INVENTORY_NAME = "inventory.json"


def generate(measurements: bytes, printed: bytes, inputs_raw: bytes, shared: bytes) -> dict[str, bytes]:
    """Every generated file of ``data/g4/d2/measured/`` (the committed ``inputs.json`` excluded)."""
    inputs = load_inputs(inputs_raw)
    rows = load_source_rows(measurements, printed)
    binding = bind_source_identity(shared, inputs)
    evaluation = evaluate(rows, inputs, binding)
    files = table_files(evaluation)
    files[EVALUATION_NAME] = canonical_bytes(evaluation)
    files[INVENTORY_NAME] = inventory({**files, INPUTS_NAME: inputs_raw})
    return files


# --------------------------------------------------------------------------------------------
# reading the committed record
# --------------------------------------------------------------------------------------------

_EVALUATION_KEYS = (
    "schema",
    "source_id",
    "decision_record",
    "budget_tv",
    "statistics",
    "carbon_locator",
    "publication_resolution",
    "hygroscopic_locator",
    "shared_record",
    "components",
    "rows",
    "counts",
    "minimum_U_at_least",
    "actual_region_drills",
    "attempts",
)


def check_committed(committed: Mapping[str, bytes], recomputed: Mapping[str, bytes]) -> None:
    """The committed directory is exactly the recomputed files plus ``inputs.json``: no file missing, none
    extra, every one byte for byte its recomputation, and the evaluation's decision fields are its decision
    record's. Derived fields are never read from the committed files."""
    want = set(recomputed) | {INPUTS_NAME}
    if set(committed) != want:
        raise MeasuredError(f"files differ: {sorted(set(committed) ^ want)}")
    for name, b in recomputed.items():
        if committed[name] != b:
            raise MeasuredError(f"{name} is not its recomputation")
    ev = _keys(_strict_json(committed[EVALUATION_NAME], "evaluation"), _EVALUATION_KEYS, "evaluation")
    roles = [a["role"] for a in ev["attempts"]]
    if roles.count(ROLE_DECISION) != 1 or ev["attempts"][0]["id"] != ev["decision_record"]:
        raise MeasuredError("exactly one decision record, first")
    for a in ev["attempts"][1:]:
        if a["role"] not in (ROLE_SENSITIVITY, ROLE_LATER) or "verdict" in json.dumps(a):
            raise MeasuredError("an attempt other than the decision record carries a verdict")


def precision_declaration(evaluation: Mapping[str, Any]) -> dict[str, str]:
    """The values the generated page states about precision, read from the evaluated record and checked
    against the registered recipe of its decision identity; the page's "every oxide" holds only if every row
    excludes MEETS. Anything else refuses, so no page can state the frozen recipe for another one."""
    recipe = REGISTERED_RECIPES.get(evaluation["decision_record"])
    stats = evaluation["statistics"]
    stated = Recipe(
        Fraction(evaluation["budget_tv"]),
        Fraction(stats["log_scale"]),
        stats["coverage_k"],
        Fraction(evaluation["publication_resolution"]),
        tuple(sorted(r["formula"] for r in evaluation["rows"] if r["hygroscopic"])),
    )
    if recipe is None or stated != recipe:
        raise MeasuredError(f"{evaluation['decision_record']}: {RECIPE_REFUSAL}")
    rows = evaluation["rows"]
    excluded = [r["formula"] for r in rows if MEETS_EXCLUDED_TOKEN in r["precision"]["excluded"]]
    if len(excluded) != len(rows) or evaluation["counts"]["precision_meets_excluded"] != len(rows):
        raise MeasuredError("the page states an exclusion for every oxide, which this record does not carry")
    return {
        "percent": _plain(recipe.stat_log_scale * 100),
        "k": str(recipe.coverage_k),
        "budget": _plain(recipe.budget),
    }


MEETS_EXCLUDED_TOKEN = decisions.MEETS_EXCLUDED


def delivered_vector(
    evaluation: Mapping[str, Any], formula: str, convention: str = DELIVERY_MODE
) -> dict[int, float]:
    """The probability per Z of a row, under the source's own convention only; any other refuses."""
    if convention != DELIVERY_MODE:
        raise ConversionRefused(
            f"{CONVERSION_REFUSAL}: {formula} under {convention!r}: no source or response "
            "input supports a reinterpretation of the reported value"
        )
    for r in evaluation["rows"]:
        if r["formula"] == formula:
            return {
                r["Z"]: float(r["delivery"]["p_metal_binary64"]),
                OXYGEN_Z: float(r["delivery"]["p_oxygen_binary64"]),
            }
    raise MeasuredError(f"no row {formula!r}")
