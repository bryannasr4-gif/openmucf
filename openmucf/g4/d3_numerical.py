"""Recompute the historical numerical certificate from primitive archived evidence.

Only the standard library is used. Fresh reference solving is a separate operation.
All arithmetic preserves the historical float operation order and decimal telescoping.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

REFERENCE_RELPATH = "data/g4/d3/reference"
ALPHA = 7.2973525664e-3
C = 137.035999139
EH = 27.211386245981
SHELLS = "KLMNOPQRS"
VARIANTS = ("prod", "rtol13", "rmin100", "rmax50", "match05")
LABELLED_CENTROID = {(4, 9), (6, 12), (8, 16), (11, 23), (13, 27), (14, 28), (18, 40)}
Key = tuple[int, int, bool, str]
PayloadKey = tuple[int, int, bool]


class EvidenceError(ValueError):
    """Missing, inconsistent or unusable evidence cannot qualify a record."""


def circular(n: int) -> list[str]:
    return ["K1"] if n == 1 else [f"{SHELLS[n - 1]}{2*n-2}", f"{SHELLS[n - 1]}{2*n-1}"]


CHAIN = [s for n in range(1, 9) for s in circular(n)]


def state(label: str) -> tuple[int, int, int, int, int]:
    n, sub = SHELLS.index(label[0]) + 1, int(label[1:])
    l = 0 if sub == 1 else sub // 2
    two_j = 1 if sub == 1 else 2*l + (1 if sub % 2 else -1)
    kappa = -(l+1) if two_j == 2*l+1 else l
    return n, l, two_j, kappa, n-l-1


def chain_lines() -> list[str]:
    return [f"{lo}-{hi}" for n in range(2, 9) for lo in circular(n-1) for hi in circular(n)
            if abs(state(lo)[2] - state(hi)[2]) <= 2]


QUANTITIES = tuple([f"level:{s}" for s in CHAIN] + [f"line:{s}" for s in chain_lines()]
                   + ["K"] + [f"e{n}" for n in range(2, 9)] + ["K-e2"]
                   + [f"e{n}-e{n+1}" for n in range(2, 8)])


def quantities(bindings: dict[str, float]) -> dict[str, float]:
    """Consumer transform, applied to each reference variant before taking a maximum."""
    q = {f"level:{s}": bindings[s] for s in CHAIN}
    q.update({f"line:{ln}": bindings[ln.split("-")[0]] - bindings[ln.split("-")[1]]
              for ln in chain_lines()})
    q["K"] = bindings["K1"]
    for n in range(2, 9):
        lo, hi = circular(n)
        l = n-1
        q[f"e{n}"] = (2*l*bindings[lo] + (2*l+2)*bindings[hi]) / (4*l+2)
    q["K-e2"] = q["K"] - q["e2"]
    q.update({f"e{n}-e{n+1}": q[f"e{n}"] - q[f"e{n+1}"] for n in range(2, 8)})
    return q


def states_of(q: str) -> list[str]:
    if q.startswith("level:"):
        return [q[6:]]
    if q.startswith("line:"):
        return q[5:].split("-")
    if q == "K":
        return ["K1"]
    if q == "K-e2":
        return ["K1"] + circular(2)
    return [s for part in q.split("-") for s in circular(int(part[1:]))]


def _unique(records: list[Any], key: Any, label: str) -> dict[Any, Any]:
    result = {}
    for row in records:
        k = key(row)
        if k in result:
            raise EvidenceError(f"duplicate {label} key {k}")
        result[k] = row
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


@dataclass
class Evidence:
    cases: dict[tuple[int, int], dict[str, Any]]
    payloads: dict[PayloadKey, dict[str, Any]]
    runs: dict[tuple[int, int, bool, str], dict[str, Any]]
    expected: tuple[Key, ...]
    targets: dict[tuple[int, int], float]
    protocol: dict[str, Any]


def load_evidence(root: Path) -> Evidence:
    """Read only the supplied repository root and check provenance separately from arithmetic."""
    root = Path(root)
    base = root / REFERENCE_RELPATH
    protocol = json.loads((base / "protocol.json").read_text(encoding="ascii"))
    if protocol["schema"] != 1 or protocol["recipe"] != {
        "k": 3, "N_u": 60, "N_s": 12, "tol": 1e-7, "extrapolated": False,
    }:
        raise EvidenceError("unsupported schema or extrapolated recipe")
    for name in ("cases.json", "payloads.jsonl", "runs.jsonl", "expected_keys.json"):
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != protocol["files"].get(name):
            raise EvidenceError(f"evidence digest mismatch: {name}")
    cases = _unique(json.loads((base / "cases.json").read_text())["cases"],
                    lambda r: (r["z"], r["a"]), "case")
    payloads = _unique(_jsonl(base / "payloads.jsonl"),
                       lambda r: (r["z"], r["a"], r["uehling"]), "payload")
    payloads = {k: r["payload"] for k, r in payloads.items()}
    runs = _unique(_jsonl(base / "runs.jsonl"),
                  lambda r: (r["z"], r["a"], r["uehling"], r["role"]), "run")
    expected = tuple(tuple(row) for row in json.loads((base / "expected_keys.json").read_text()))
    cross = {(z, a, on, q) for z, a in cases for on in (False, True) for q in QUANTITIES}
    if not cross or len(expected) != len(set(expected)) or set(expected) != cross:
        raise EvidenceError("expected inventory is empty, duplicate or not the exact quantity cross product")
    measured: dict[tuple[int, int], Decimal] = {}
    with (root / "data/g4/d3/validation_cells.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["gated"] == "true" or row["reason"] == "centroid":
                key = int(row["Z"]), int(row["A"])
                # The registration stores decimal targets; convert units and scale
                # in decimal arithmetic before their single conversion to float.
                sigma = Decimal(row["unc_keV"]) * 1000
                measured[key] = min(measured.get(key, Decimal("Infinity")), sigma)
    targets = {key: float(val / 10) for key, val in measured.items()}
    ev = Evidence(cases, payloads, runs, expected, targets, protocol)
    _validate_joins(root, ev)
    return ev


def _roles(on: bool) -> set[str]:
    return {"dx1", "dx2", "dx3", "dx4", "tol", "phase1", "phase2", "phase3"} | (
        {"nu", "ns"} if on else set())


def _settings(role: str) -> dict[str, Any]:
    k = int(role[-1]) if role.startswith("dx") else 3
    d: dict[str, Any] = {"k": k, "N_u": 90 if role == "nu" else 60,
                         "N_s": 16 if role == "ns" else 12,
                         "energy_tol": 1e-8 if role == "tol" else 1e-7,
                         "loggrid_step": 0.005 / 2**k}
    if role.startswith("phase"):
        d["phase"] = int(role[-1])
        d["loggrid_center"] = math.exp(d["phase"] * d["loggrid_step"] / 4)
    return d


def _validate_joins(root: Path, evidence: Evidence) -> None:
    wanted = {(z, a, on, role) for z, a in evidence.cases for on in (False, True) for role in _roles(on)}
    if set(evidence.runs) != wanted or set(evidence.payloads) != {k[:3] for k in wanted}:
        raise EvidenceError("missing or extra primitive run or reference payload")
    for case in evidence.cases.values():
        _validate_case(case)
    with (root / "data/g4/d3/mudirac_runs.csv").open(newline="") as stream:
        public = {r["run"]: r for r in csv.DictReader(stream)}
    with (root / "data/g4/d3/mudirac_lines.csv").open(newline="") as stream:
        public_lines = {(r["run"], r["line"]): r["delta_e_eV"] for r in csv.DictReader(stream)}
    with (root / "data/g4/d3/mudirac_states.csv").open(newline="") as stream:
        public_states = {(r["run"], r["state"]): r for r in csv.DictReader(stream)}
    for (z, a, on, role), run in evidence.runs.items():
        digest = hashlib.sha256(run["input_text"].encode("ascii")).hexdigest()
        if digest != run["input_sha256"] or run["settings"] != _settings(role):
            raise EvidenceError("input bytes or numerical settings mismatch")
        if (run["z"], run["a"], run["uehling"], run["role"]) != (z, a, on, role):
            raise EvidenceError("run physical key mismatch")
        _validate_input(run, evidence.cases[z, a])
        original = public.get(run["public_run"]) if run["public_run"] else None
        if run["public_run"] and (original is None or (
            int(original["Z"]), int(original["A"]), original["uehling"] == "on", original["input_sha256"]
        ) != (z, a, on, digest)):
            raise EvidenceError("stale public run mapping")
        if role == "dx3" and evidence.cases[z, a]["production"]["on" if on else "off"] != run["public_run"]:
            raise EvidenceError("production mapping mismatch")
        if original is not None:
            if any(public_lines.get((run["public_run"], line)) != value
                   for line, value in run["lines"].items()):
                raise EvidenceError("mapped public line differs from the primitive record")
            for label, state_record in run["states"].items():
                exposed = public_states.get((run["public_run"], label))
                if exposed is None or any(exposed[k] != state_record[k]
                                          for k in ("n", "l", "s", "binding_eV")):
                    raise EvidenceError("mapped public state differs from the primitive record")
    for (z, a, on), payload in evidence.payloads.items():
        _validate_payload((z, a, on), payload, evidence.cases[z, a])


def _validate_case(case: dict[str, Any]) -> None:
    values = [case[k] for k in ("reduced_mass", "radius_fm", "rms_run", "t_fm")]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise EvidenceError("nonfinite or nonpositive physical case input")


def _validate_payload(key: PayloadKey, payload: dict[str, Any], case: dict[str, Any]) -> None:
    z, a, on = key
    if (payload["Z"], payload["A_mass"], payload["uehling"], float(payload["mu"])) != (
        z, a, on, case["reduced_mass"]
    ) or any(payload[k] != case[k] for k in ("radius_fm", "rms_run", "t_fm")):
        raise EvidenceError("reference physical key or physical input mismatch")
    _validate_reference(payload)


def _validate_input(run: dict[str, Any], case: dict[str, Any]) -> None:
    fields = {}
    for line in run["input_text"].splitlines():
        if line.strip() and not line.startswith("#"):
            key, value = line.split(":", 1)
            if key in fields:
                raise EvidenceError("duplicate input recipe keyword")
            fields[key] = value.strip()
    expected = {"element": case["symbol"], "isotope": str(case["a"]),
                "uehling_correction": "TRUE" if run["uehling"] else "FALSE",
                "nuclear_model": "FERMI2", "fermi2_rms_exact": "TRUE", "nuclear_mass": "TRUE",
                "ev_codata2022": "TRUE", "fermi2_hermite": "TRUE", "uehling_density": "fermi2",
                "reduced_mass": "TRUE", "optimise_fermi_parameters": "FALSE", "ideal_atom_minshell": "S"}
    numbers = {"radius": case["radius_fm"], "fermi_t": case["t_fm"], "fermi2_rms_fm": case["rms_run"],
               "loggrid_step": run["settings"]["loggrid_step"],
               "energy_tol": run["settings"]["energy_tol"], "uehling_steps": run["settings"]["N_u"],
               "uehling_shell_nodes": run["settings"]["N_s"],
               "xr_print_precision": int(run["output"]["xr_precision"]),
               "state_header_precision": int(run["output"]["header_precision"]),
               "state_print_precision": 12, "output": 2}
    if "loggrid_center" in run["settings"]:
        numbers["loggrid_center"] = run["settings"]["loggrid_center"]
    if set(fields) != set(expected) | set(numbers) | {"xr_lines"}:
        raise EvidenceError("input recipe has missing or unsupported keywords")
    line_recipe = ("K1-L2:L3,L2:L3-M4:M5,M4:M5-N6:N7,N6:N7-O8:O9,O8:O9-P10:P11,"
                   "P10:P11-Q12:Q13,Q12:Q13-R14:R15,")
    if fields["xr_lines"] not in {line_recipe+"R15-S17", line_recipe+"L3-N5,R15-S17"}:
        raise EvidenceError("input recipe printed-line vocabulary differs from the fixed recipe")
    if any(fields.get(k) != v for k, v in expected.items()) or any(
        k not in fields or float(fields[k]) != v for k, v in numbers.items()
    ):
        raise EvidenceError("input recipe physical key, settings or precision mismatch")


def with_payloads(evidence: Evidence, payloads: dict[PayloadKey, dict[str, Any]]) -> Evidence:
    """Use freshly computed payloads, preserving the frozen producer records and population."""
    if not payloads or not payloads.keys() <= evidence.payloads.keys():
        raise EvidenceError("empty or unknown reference payload override")
    try:
        for key, payload in payloads.items():
            case = evidence.cases[key[:2]]
            _validate_case(case)
            _validate_payload(key, payload, case)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise EvidenceError(f"unusable reference payload override: {exc}") from exc
    return replace(evidence, payloads={**evidence.payloads, **payloads})


def float_floor(mu: float) -> float:
    return 2.0 ** (math.floor(math.log2(mu*C*C)) - 52) / (1.0/EH)


def header_half_unit(text: str) -> float:
    value = Decimal(text)
    if not value.is_finite():
        raise EvidenceError("nonfinite producer header")
    exponent = value.as_tuple().exponent
    assert isinstance(exponent, int)
    return float(Decimal(5).scaleb(exponent-1))


def producer_quantities(run: dict[str, Any], case: dict[str, Any]) -> tuple[dict[str, tuple[float, float]],
                                                                          dict[str, tuple[Decimal, float]]]:
    """Reconstruct bindings from printed lines, using the binary's analytic S17 anchor."""
    mu, z = case["reduced_mass"], case["z"]
    floor = float_floor(mu)
    half = 0.5 * 10.0 ** -int(run["output"]["xr_precision"]) + floor
    gamma = math.sqrt(81-(z*ALPHA)**2)
    energy = mu*C**2 / math.sqrt(1+(z*ALPHA/gamma)**2) - mu*C**2
    anchor = -energy / (1.0/EH)
    lines = run["lines"]
    with localcontext() as ctx:
        ctx.prec = 50
        b = {"R15": (Decimal(repr(anchor))+Decimal(lines["R15-S17"]), half)}
        for n in range(7, 0, -1):
            lo, hi = circular(n)[-1], circular(n+1)[-1]
            b[lo] = b[hi][0]+Decimal(lines[f"{lo}-{hi}"]), b[hi][1]+half
        for n in range(1, 8):
            lo, hi = circular(n)[0], circular(n+1)[0]
            b[hi] = b[lo][0]-Decimal(lines[f"{lo}-{hi}"]), b[lo][1]+half
    q = quantities({k: float(v[0]) for k, v in b.items()})
    result = {}
    for name, val in q.items():
        if name.startswith("line:"):
            result[name] = float(Decimal(lines[name[5:]])), half
        else:
            result[name] = val, sum(b[s][1] for s in states_of(name))
    return result, b


def reference_quantities(payload: dict[str, Any]) -> tuple[dict[str, float], dict[str, float],
                                                         dict[str, dict[str, float]]]:
    _validate_reference(payload)
    eh = payload["e_h_eV"]
    sets = {f"A:{v}": quantities({s: -payload["A"][s][v]["B_Ha"]*eh for s in CHAIN})
            for v in VARIANTS}
    sets.update({f"B:{v}": quantities({s: -payload["B"][v][s]*eh for s in CHAIN})
                 for v in ("220", "300")})
    prod = sets["A:prod"]
    bound = {q: max(abs(values[q]-val) for values in sets.values()) for q, val in prod.items()}
    shifts = {}
    base = {s: -payload["A"][s]["prod"]["B_Ha"]*eh for s in CHAIN}
    for key in ("dom:1e-5", "C:herm5000"):
        moved = quantities({s: base[s]-payload["eps_level_Ha"][key][s]*eh for s in CHAIN})
        shifts[key] = {q: moved[q]-prod[q] for q in prod}
    return prod, bound, shifts


def _validate_reference(payload: dict[str, Any]) -> None:
    if payload["e_h_eV"] != EH:
        raise EvidenceError("reference energy conversion differs from the fixed Hamiltonian")
    values = [payload["A"][s][v]["B_Ha"] for s in CHAIN+["L1"] for v in VARIANTS]
    values += [payload["B"][v][s] for v in ("220", "300") for s in CHAIN+["L1"]]
    values += [payload["eps_level_Ha"][key][s] for key in ("dom:1e-5", "C:herm5000") for s in CHAIN]
    if not payload["uehling"]:
        values += [payload["first_order_level_Ha"]["VU_F1"][s] for s in CHAIN+["L1"]]
    if not all(math.isfinite(value) for value in values):
        raise EvidenceError("nonfinite reference numerical primitive")


def floor_ladder(seq: list[float], unit: float, print_floor: float = 1e-12) -> bool:
    return len(seq) == 4 and all(abs(a-b) <= 2*unit or abs(a-b) < print_floor
                                 for a, b in zip(seq[:-1], seq[1:], strict=True))


def order_resolved(d1: float, d2: float, e1: float, e2: float, ratio: float = 2.0) -> bool:
    a1, a2 = abs(d1), abs(d2)
    if d1*d2 <= 0 or a1 <= e1 or a2 <= e2:
        return False
    width = (math.log((a1+e1)/(a2-e2))-math.log((a1-e1)/(a2+e2))) / math.log(ratio)
    return width <= 0.1


def richardson(qs: list[float], rho: list[float] | None = None) -> tuple[float, bool, list[float]]:
    if len(qs) != 4 or (rho is not None and len(rho) != len(qs)):
        return math.inf, False, []
    ps = []
    for i in range(len(qs)-2):
        d1, d2 = qs[i]-qs[i+1], qs[i+1]-qs[i+2]
        ok = d2 != 0 and d1*d2 > 0
        if ok and rho is not None:
            ok = order_resolved(d1, d2, rho[i]+rho[i+1], rho[i+1]+rho[i+2])
        ps.append(math.log(d1/d2)/math.log(2) if ok else math.nan)
    dq = qs[-2]-qs[-1]
    if all(math.isfinite(p) for p in ps[-2:]) and abs(ps[-1]-ps[-2]) <= 0.3:
        power = 2**ps[-1]
        return (math.copysign(math.inf, dq) if power == 1 else dq*power/(power-1)), True, ps
    if rho is None:
        return 2*abs(dq), False, ps
    return 2*(abs(dq)+rho[-2]+rho[-1]), False, ps


@dataclass(frozen=True)
class Term:
    name: str
    value: float
    signed: bool


@dataclass(frozen=True)
class Estimator:
    b_ladder: float
    b_num: float
    d_obs: float
    d_pred: float
    allowance: float
    gates: dict[str, bool | None]
    verdict: str
    reason: str
    terms: tuple[Term, ...]


def compare(q: float, rounding: float, ref: float, bound: float, target: float,
            terms: list[Term], identity_ok: bool, *, extrapolated: bool = False,
            uehling: bool = False) -> Estimator:
    """Both budget branches and all four applicable scientific gates are evaluated."""
    if extrapolated:
        raise EvidenceError("extrapolation is not applicable to the fixed recipe")
    required = {"dx", "domain", "csteps", "tol", "phase"} | ({"N_u", "N_s"} if uehling else set())
    if {t.name for t in terms} != required or len(terms) != len(required):
        raise EvidenceError("missing, duplicate or extra required budget term")
    if any(t.signed != (t.name in {"domain", "csteps", "N_u", "N_s"}) for t in terms if t.name != "dx"):
        raise EvidenceError("budget term signedness differs from the fixed recipe")
    if not all(math.isfinite(v) for v in (q, rounding, ref, bound, target)) or (
        rounding < 0 or bound < 0 or target <= 0
    ):
        raise EvidenceError("nonfinite input, negative bound or nonpositive target")
    if any(not math.isfinite(t.value) for t in terms):
        terms = [t for t in terms if math.isfinite(t.value)] + [Term("missing", math.inf, False)]
    ladder = sum(abs(t.value) for t in terms)+rounding
    obs = q-ref
    pred = sum(t.value for t in terms if t.signed)
    allowance = sum(abs(t.value) for t in terms if not t.signed)+bound+0.1*target+rounding
    bnum = max(ladder, abs(obs)+bound)
    gates = {"G1": bool(identity_ok), "G2": bound <= 0.2*target, "G3": bnum <= target,
             "G4": abs(obs-pred) <= allowance, "G5": None}
    failures = [g for g, ok in gates.items() if ok is False]
    reason = {"G1": "identity", "G2": "reference", "G3": "agreement", "G4": "unattributed"}
    return Estimator(ladder, bnum, obs, pred, allowance, gates,
                     "NUMERICAL_FAIL" if failures else "QUALIFIED",
                     reason[failures[0]] if failures else "", tuple(terms))


@dataclass(frozen=True)
class Evaluation:
    key: Key
    q_ev: float
    q_ref_ev: float
    b_ref_ev: float
    target_ev: float
    bue_1_ev: float | None
    domain_ev: float
    csteps_ev: float
    prereg: Estimator
    a3: Estimator
    floor: bool
    case: dict[str, Any]
    run: dict[str, Any]

    @property
    def b_num_ev(self) -> float:
        return max(self.prereg.b_num, self.a3.b_num)

    @property
    def verdict(self) -> str:
        good = all(e.verdict == "QUALIFIED" for e in (self.prereg, self.a3))
        return "QUALIFIED" if good else "NUMERICAL_FAIL"

    @property
    def reason(self) -> str:
        return self.prereg.reason or self.a3.reason

    def historical_summary(self) -> dict[str, str]:
        z, a, on, q = self.key
        return {"set": self.case["cohort"], "Z": str(z), "A": str(a), "kind": "anchor",
                "uehling": "on" if on else "off", "quantity": q, "Q_eV": f"{self.q_ev:.6f}",
                "Q_ref_eV": f"{self.q_ref_ev:.6f}", "b_ref_eV": f"{self.b_ref_ev:.3e}",
                "b_num_eV": f"{self.b_num_ev:.3e}", "target_eV": f"{self.target_ev:.4g}",
                "verdict": self.verdict, "bue_1_eV": repr(self.bue_1_ev) if on else "",
                "run": self.run["public_run"], "input_sha256": self.run["input_sha256"],
                "probe_c_bohr": self.case["probe_c_bohr"], "probe_t_bohr": self.case["probe_t_bohr"]}


def target_for(evidence: Evidence, key: tuple[int, int], q: str, reference: float) -> float:
    nt = evidence.targets.get(key)
    eng = max(1e-3, 1e-6*abs(reference))
    if nt is None:
        return eng
    if q.startswith(("line:", "level:")) or (q == "K-e2" and key in LABELLED_CENTROID):
        return nt
    return min(nt, eng)


def _identity(payload: dict[str, Any], run: dict[str, Any], q: str,
              bindings: dict[str, tuple[Decimal, float]], case: dict[str, Any]) -> bool:
    for lab in states_of(q):
        n, l, two_j, kappa, nodes = state(lab)
        rec, producer = payload["A"][lab]["prod"], run["states"].get(lab)
        if rec["nodes"] != nodes or rec["kappa"] != kappa or producer is None:
            return False
        spin = 1 if l == 0 or two_j == 2*l+1 else 0
        if tuple(int(producer[k]) for k in ("n", "l", "s", "nodes_P")) != (n, l, spin, nodes):
            return False
        h = producer["binding_eV"]
        gap = abs(float(bindings[lab][0])+float(h))
        allowance = (header_half_unit(h)+float_floor(case["reduced_mass"])+bindings[lab][1]
                     +0.5*10**-int(run["output"]["xr_precision"])+float_floor(case["reduced_mass"]))
        if gap > allowance:
            return False
    return True


def evaluate_all(root: Path, evidence: Evidence | None = None, *, floor_amendment: bool = True,
                 members: set[tuple[int, int]] | None = None) -> dict[Key, Evaluation]:
    """Evaluate both estimators without reading or trusting stored CSV verdicts."""
    try:
        return _evaluate_all(root, evidence, floor_amendment=floor_amendment, members=members)
    except (KeyError, TypeError, OverflowError) as exc:
        raise EvidenceError(f"missing or unusable primitive evidence: {exc}") from exc


def _evaluate_all(root: Path, evidence: Evidence | None = None, *, floor_amendment: bool = True,
                  members: set[tuple[int, int]] | None = None) -> dict[Key, Evaluation]:
    ev = evidence if evidence is not None else load_evidence(root)
    out = {}
    for (z, a), case in ev.cases.items():
        if members is not None and (z, a) not in members:
            continue
        _validate_case(case)
        off = ev.payloads[z, a, False]
        bue = quantities({s: -off["first_order_level_Ha"]["VU_F1"][s]*off["e_h_eV"] for s in CHAIN})
        for on in (False, True):
            payload = ev.payloads[z, a, on]
            ref, bounds, shifts = reference_quantities(payload)
            rq = {}
            for role in _roles(on):
                run = ev.runs.get((z, a, on, role))
                if run is None or run["output"]["GE"] != "COMPLETE":
                    raise EvidenceError(f"missing or unusable required run: {(z, a, on, role)}")
                rq[role] = producer_quantities(run, case)
            run = ev.runs[z, a, on, "dx3"]
            for q in QUANTITIES:
                seq = [rq[f"dx{k}"][0][q][0] for k in (1, 2, 3, 4)]
                rho = [rq[f"dx{k}"][0][q][1] for k in (1, 2, 3, 4)]
                qmd, rounding = rq["dx3"][0][q]
                floor = floor_amendment and floor_ladder(seq, float_floor(case["reduced_mass"]))
                target = target_for(ev, (z, a), q, ref[q])
                identity = _identity(payload, run, q, rq["dx3"][1], case)
                estimators = []
                for aware in (False, True):
                    dx, signed = ((2*(abs(seq[-2]-seq[-1])+rho[-2]+rho[-1]), False) if floor else
                                  richardson(seq, rho if aware else None)[:2])
                    terms = [Term("dx", dx, signed)]
                    if on:
                        terms.extend(Term(name, qmd-rq[role][0][q][0], True)
                                     for name, role in (("N_u", "nu"), ("N_s", "ns")))
                    terms.extend([Term("csteps", shifts["C:herm5000"][q], True),
                                  Term("domain", shifts["dom:1e-5"][q], True),
                                  Term("tol", abs(qmd-rq["tol"][0][q][0]), False),
                                  Term("phase", max(abs(rq[f"phase{p}"][0][q][0]-qmd)
                                                    for p in (1, 2, 3)), False)])
                    estimators.append(compare(qmd, rounding, ref[q], bounds[q], target, terms, identity,
                                              uehling=on))
                key = z, a, on, q
                out[key] = Evaluation(key, qmd, ref[q], bounds[q], target, bue[q] if on else None,
                                      shifts["dom:1e-5"][q], shifts["C:herm5000"][q],
                                      estimators[0], estimators[1], floor, case, run)
    expected = {key for key in ev.expected if members is None or key[:2] in members}
    if not expected or set(out) != expected:
        raise EvidenceError("evaluated coverage differs from the frozen expected inventory")
    return out
