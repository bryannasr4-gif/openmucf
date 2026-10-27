"""Generate and check the evaluated bound-muon rate pair.

``generate`` writes ``data/g4/d1/bound_rate_evaluation.json`` (every derived quantity and decision of the
evaluated pair) and ``data/g4/d1/bound_decay_response_v1.json`` (the D1 roles of the shared D1/D2 convention
record) from the committed primitives ``data/g4/d1/bound_rate_inputs.json``, the decay-rate oracle and the
compendium rows. ``check`` recomputes both from the primitives and refuses a committed file whose fields differ
(numbers to 1e-12 relative: the reference ranges come from a numerical search, so they are not byte-pinned), and
checks that the dataset's pair table carries exactly the production tuples. It also renders ``docs/evaluated-bound-rates.md`` from the record and the primitives, so every number on that
page is the checked one. ``make audit`` runs ``check``. The
work is delegated to :mod:`openmucf.g4.d1_lifetime`, which owns the measurement records this pair extends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openmucf.g4 import d1_evaluated as evaluated
from openmucf.g4 import d1_lifetime as lifetime
from openmucf.g4 import provenance, spec

ROOT = Path(__file__).resolve().parents[1]


def render(obj: object) -> bytes:
    """Indented, sorted, LF-terminated JSON; floats by their shortest round-trip repr."""
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def generate(root: Path) -> None:
    record = lifetime.evaluation(root)
    convention = lifetime.convention(root, record)
    (root / lifetime.EVALUATION_RELPATH).write_bytes(render(record))
    (root / lifetime.CONVENTION_RELPATH).write_bytes(render(convention))
    (root / DOCUMENT_RELPATH).write_bytes(render_document(record, lifetime.evaluated_inputs(root)))
    decisions = record["decisions"]
    assert isinstance(decisions, dict)
    print(f"evaluation out={lifetime.EVALUATION_RELPATH} convention out={lifetime.CONVENTION_RELPATH} "
          f"payload_sha256={convention['payload_sha256']}")
    for name, value in decisions.items():
        print(f"decision {name} {value['verdict']} L={value['L']!r} U={value['U']!r}")


def check(root: Path) -> None:
    record = lifetime.evaluation(root)
    evaluated.check_evaluation((root / lifetime.EVALUATION_RELPATH).read_bytes(), record)
    convention = lifetime.convention(root, record)
    committed = evaluated.parse_json((root / lifetime.CONVENTION_RELPATH).read_bytes())
    if evaluated.payload_digest(committed.get("payload")) != committed.get("payload_sha256"):
        raise SystemExit("evaluated check FAILED: the convention record's payload digest does not match its payload")
    if evaluated.canonical_bytes(committed) != evaluated.canonical_bytes(convention):
        raise SystemExit("evaluated check FAILED: the convention record differs from its recomputation")
    table = spec.parse((root / lifetime.PAIR_LAYER1_RELPATH).read_bytes().decode("ascii"))
    raw = (root / lifetime.PAIR_LAYER2_RELPATH).read_bytes()
    provenance.check_canonical_bytes(raw)
    provenance.check_source_digest(table, raw)
    provenance.check_against_table(table, provenance.from_json_obj(evaluated.parse_json(raw)))
    expected = [(z, a, float(tau), float(q), float(free_tau)) for z, a, tau, q, free_tau in
                lifetime.evaluated_tuples(root)]
    declared = {key: table.directives.get(key) for key in ("PROFILE", "TABLE", "UNITS", "COLUMNS", "SEAM")}
    wanted = {"PROFILE": evaluated.PAIR_PROFILE, "TABLE": evaluated.PAIR_TABLE, "UNITS": evaluated.PAIR_UNITS,
              "COLUMNS": " ".join(evaluated.PAIR_COLUMNS), "SEAM": "d1_nuclear_capture"}
    if declared != wanted or not table.directives.get("VALIDITY", "").endswith(" A:listed"):
        raise SystemExit(f"evaluated check FAILED: the pair table declares {declared}, not {wanted} with A:listed")
    if [tuple(record) for record in table.records] != expected:
        raise SystemExit("evaluated check FAILED: the pair table does not carry exactly the production tuples")
    # a text checkout may carry CRLF: the page is compared line by line, its rendering is LF
    page = (root / DOCUMENT_RELPATH).read_bytes().replace(b"\r\n", b"\n")
    if page != render_document(record, lifetime.evaluated_inputs(root)):
        raise SystemExit(f"evaluated check FAILED: {DOCUMENT_RELPATH} is not the page the record renders")
    print(f"evaluated check OK: {lifetime.EVALUATION_RELPATH}, {lifetime.CONVENTION_RELPATH}, {DOCUMENT_RELPATH} and "
          f"{lifetime.PAIR_LAYER1_RELPATH} equal their recomputation")


DOCUMENT_RELPATH = "docs/evaluated-bound-rates.md"


def g(x: object, digits: int = 9) -> str:
    """A number for the page: `digits` significant digits."""
    return format(float(x), f".{digits}g")  # type: ignore[arg-type]


def render_document(record: dict, inputs: object) -> bytes:
    """The page: every number rendered from the checked evaluation record or the typed primitives."""
    from openmucf.g4 import d1_evaluated as ev

    raw = inputs.raw  # type: ignore[attr-defined]
    al = record["q"]["Al27"]
    t = record["tuple"]
    p = record["pair"]
    m = record["mapping"]
    box = record["box"]
    dec = record["decisions"]
    rng = record["ranges"]
    hf = raw["hyperfine"]
    weight = raw["constants"]["upper_state_weight"]
    us = 1e3  # ns^-1 to per microsecond
    rows_q = "\n".join(f"| `{k}` | {g(v)} |" for k, v in sorted(al["members"].items()))
    rows_ledger = "\n".join(f"| `{k}` | {g(v, 4)} |" for k, v in al["ledger"].items())
    names = {"late_lifetime": "late lifetime (ns)", "stopped_mean": "stopped mean (ns)",
             "capture_probability": "capture probability", "decay_rate": "decay rate (per µs)"}
    scale = {"late_lifetime": 1.0, "stopped_mean": 1.0, "capture_probability": 1.0, "decay_rate": us}
    rows_dec = "\n".join(
        f"| {names[k]} | {g(dec[k]['prediction'] * scale[k])} | {g(rng[k]['low'] * scale[k])} – "
        f"{g(rng[k]['high'] * scale[k])} | {dec[k]['norm']} {g(dec[k]['budget'])} | {g(dec[k]['L'], 6)} | "
        f"{g(dec[k]['U'], 6)} | **{dec[k]['verdict']}** |"
        for k, _ in ev.DECISIONS)
    imp_rows = []
    for route, s in record["stock"].items():
        for k in ("late_lifetime", "stopped_mean", "capture_probability"):
            v = s["improvement"][k]
            stock = s["lifetime_ns"] if k != "capture_probability" else s["capture_probability"]
            imp_rows.append(f"| {route} | {names[k]} | {g(stock)} | {g(v['min_advantage'], 6)} | **{v['verdict']}** |")
    pw = record["power"]
    mm = record["mean_matched"]
    tc = record["transition_crosscheck"]
    oq, cq = record["q"]["O16"], record["q"]["C12"]
    oc, cc = record["census"]["O16"], record["census"]["C12"]
    fin = record["stock_table_finding"]
    fin_rows = "\n".join(
        f"| {k} | {g(v['own'][0], 5)} ± {g(v['own'][1], 3)} | {g(v['comparison'][0], 5)} ± {g(v['comparison'][1], 3)} | "
        f"{g(v['unscaled_average'], 7)} | {g(v['stock'], 5)} | {g(v['difference_in_combined_printed_units'], 4)} |"
        for k, v in fin.items())
    lo_a = float(hf["amplitude_center"]) - float(hf["amplitude_coverage_k"]) * float(hf["amplitude_standard_uncertainty"])
    hi_a = float(hf["amplitude_center"]) + float(hf["amplitude_coverage_k"]) * float(hf["amplitude_standard_uncertainty"])
    text = f"""# The evaluated aluminium-27 bound-muon rate pair

This page is generated by `scripts/generate_g4evaluated_rates.py` from `data/g4/d1/bound_rate_evaluation.json`,
which `make audit` recomputes from the committed primitives `data/g4/d1/bound_rate_inputs.json`. Every number
below is rendered from that checked record.

**Summary.** The late lifetime and the decay rate meet provisional budgets (conditional, evaluator-assessed
regions); the stopped mean and the capture probability are not established: the hyperfine capture contrast is
unresolved. The late-lifetime decision has no validation power, because the delivered lifetime is the source's
own value. On Geant4's atomic route the pair robustly improves the late lifetime over the stock pair; no other
improvement is established. Oxygen-16 and carbon-12 are evaluated as decay factors only; no runtime pair is
emitted for them.

## What is delivered

One primitive tuple per admitted isotope, in the `bound_rate_primitives` table of the `evaluated-bound` profile
(`data/g4/d1/d1_bound_rate.evaluated-bound.g4dat`, inside the `G4MuonicData` dataset):

| Z | A | tau (ns) | q | free_tau (ns) |
|---|---|---|---|---|
| {t['Z']} | {t['A']} | {t['tau_ns']} | {t['q']} | {t['free_tau_ns']} |

A consumer forms both rates from the one tuple: decay `d = q / free_tau`, capture `c = 1/tau - d`, so
`c + d = 1/tau` exactly. For the row above, `c` = {g(p['capture_per_ns'] * us, 12)} per µs, `d` =
{g(p['decay_per_ns'] * us, 12)} per µs, and the capture probability `c/(c+d)` is {g(p['capture_probability'], 12)}.

- **tau** is the late-time electron lifetime of the self-supported aluminium plate of Mizuno et al.,
  Phys. Rev. C 112, 024307 (2025), Table I, read from [arXiv:2501.05897v2](https://arxiv.org/abs/2501.05897v2)
  (the journal version was not read), with standard uncertainty {raw['lifetime']['standard_uncertainty_ns']} ns.
  Its stated components are kept as metadata and not added to it. The source fits a single exponential from a
  time origin about {raw['lifetime']['time_origin_ns']} ns after the end of its second beam pulse.
- **q** is the bound-to-free decay factor: the finite-charge static value {raw['nuclei']['Al27']['static_q']} of
  Uesaka et al., Atomic Data and Nuclear Data Tables 171, 101809 (2026)
  ([arXiv:2602.07501v2](https://arxiv.org/abs/2602.07501v2)), plus the adopted linear recoil term
  `(Z alpha)^2 eta` ({g(al['recoil'], 6)}; [derivation](low-z-bound-muon-decay.md)) and the signed point-Uehling
  kinetic term ({g(al['uehling_kinetic'], 6)}; the potential of Szafron and Czarnecki, Phys. Rev. D 94, 051301 (2016),
  [arXiv:1608.05447](https://arxiv.org/abs/1608.05447)). The assembly is {g(al['assembly'], 12)}; the tuple
  carries it to {raw['q_serialization_decimals']} decimals (rounding displacement {g(t['rounding_displacement'], 4)}).
- **free_tau** is the empirical positive-muon lifetime, as printed in Uesaka et al., Eq. (3).

## Selecting it in Geant4

Both existing options are needed: the dataset option (`G4HadronicParameters::SetEnableMuonicData(true)`) and the
common-rate switch (`SetEnableMuonicBoundRates(true)`), in either order before the first stopped muon, and the
variable `G4MUONICDATA_D1_PAIR_PROFILE=evaluated-bound`. The pair profile is selected independently of the
capture profile (`G4MUONICDATA_D1_PROFILE`): the aluminium-27 key, which a normal aluminium target reaches, takes
the pair on both stopped-muon routes. Other keys retain the selected capture profile and the existing route-specific formula or fallback. With the dataset option on, the loader parses and validates every table. Both switches and the pair-profile selection are required to use an evaluated row. Natural-composition (`A = 0`) requests have no pair. Unset, the pair selector is `compiled` and nothing
changes. `G4MuonicBoundRateProvider::Describe(Z, A)` names, per key, the capture profile, the pair profile, the
answer (pair row or formula) and its decay-factor convention; within one run aluminium uses the evaluated factor
while other elements use the leading formula.

Refused, never silently replaced by another pair: the pair profile with the switch off (at the first stopped
muon, whatever the option order), the legacy `G4MUONICDATA_PROFILE` naming the pair profile, a pair profile with
no pair table, a pair table whose tuple or whose derived rates are malformed or that carries no row, and a pair
table whose rows are not the rows its digest-checked provenance file admits (a substituted, moved or removed row).
The consumer lifetime `1/(capture + decay)` must be finite and positive. With the fatal exception
suppressed, every later lookup throws.

## The decay-factor region

The frozen rule is `W_Q = max_k |Q_a - Q_k| + sum |ledger|`, labelled EVALUATOR_TYPE_B: an evaluator assessment
covering screening, bound daughters, polarization, charge modelling and static numerics, not a confidence
interval or a bound. Prescriptions compared:

| prescription | Q |
|---|---|
{rows_q}

The 2011 aluminium value of Czarnecki, Garcia i Tormo and Marciano (Phys. Rev. D 84, 013006 (2011)) sets the
electron mass to zero in both the bound and the free width, so it is kept as printed; the point-Coulomb values of
Czarnecki, Davydov and Kaygorodov (Phys. Rev. D 113, 036028 (2026)) are printed relative to a massless-electron
free width and are divided by `f(m_e/m_mu)` = {g(record['f_factor'], 12)}. Ledger (absolute values summed, not
combined as variances):

| item | value |
|---|---|
{rows_ledger}

Sum {g(al['ledger_sum'], 6)}; spread {g(al['spread'], 6)}; half-width {g(al['half_width'], 6)}; region
[{g(al['region'][0], 10)}, {g(al['region'][1], 10)}].

## The late-slope mapping and the hyperfine model

27Al has two hyperfine states; the upper one carries {weight} of the initial population, captures at
`c_plus = c_minus - Delta` and converts to the lower one at rate `h`. A single exponential fitted from the
source's time origin to the decay-electron spectrum estimates the late slope with a bias bounded, over the whole
positive closure at `h` = {g(m['worst_transition_per_ns'] * us, 4)} per µs, by the self-consistent
{g(m['self_consistent_eps'], 10)}, rounded outward to {m['eps_map']}. With the source's tau ± 2 standard
uncertainties the late-lifetime interval is [{g(box['tau_late'][0], 12)}, {g(box['tau_late'][1], 12)}] ns.

The contrast region is read from the neutron time-spectrum amplitude `a = n_plus(0) Delta / c_minus`:
`a` = {hf['amplitude_center']} ± {hf['amplitude_coverage_k']} × {hf['amplitude_standard_uncertainty']}
(Measday, Physics Reports 354, 243 (2001), Table 3.8, aluminium, with Measday's doubled error; T. J. Stocki,
UBC PhD thesis (1998), Table 5.2), so `a` in [{g(lo_a, 4)}, {g(hi_a, 4)}] and `Delta/c_minus` in
[{g(box['contrast'][0], 6)}, {g(box['contrast'][1], 6)}]. The transition rate is
{hf['transition_center_per_us']} ± {hf['transition_coverage_k']} × {hf['transition_standard_uncertainty_per_us']} per µs
(Suzuki, Measday and Roalsvig, Phys. Rev. C 35, 2212 (1987), Table I); the inverse-variance combination of
the cross-check values ({hf['transition_crosscheck_locator']}) is {g(tc['mean_per_us'], 6)} ±
{g(tc['u_per_us'], 6)} per µs, inside it. A registered unresolved discrepancy: {hf['registered_discrepancy']}.

## Decisions

Over the frozen product of regions (late lifetime, decay factor, free lifetime, contrast, transition rate), with
the stopped mean `M = tau_late (1 + f Delta / A)` and capture probability `1 - d M` of the exact two-state model,
`L` and `U` are the smallest and largest error of the delivered value: MEETS when `U` is within the budget,
DOES_NOT_MEET when `L` exceeds it, NOT_ESTABLISHED otherwise. The budgets are provisional.

| quantity | delivered | reference range | budget | L | U | decision |
|---|---|---|---|---|---|---|
{rows_dec}

The late-lifetime decision is source-precision conditional (no validation power). The stopped mean and the
capture probability depend on the hyperfine capture contrast; the input that would settle them is an inclusive
(total-capture) aluminium hyperfine capture contrast, or a calibrated early-time electron transient in aluminium.

Adverse power on the same region: a decay factor of {raw['power']['q_mutant']} at the same tau gives
DOES_NOT_MEET for the decay rate (`L` = {g(pw['q_mutant']['decay_rate']['L'], 6)}) and for the capture probability
(`L` = {g(pw['q_mutant']['capture_probability']['L'], 6)}); a stopped mean {raw['power']['mean_mutant_factor']}
times the delivered one gives DOES_NOT_MEET (`L` = {g(pw['mean_mutant']['L'], 6)}).

## Improvement over Geant4's stock pair

Both options off, from Geant4's compiled capture table and each route's own decay rate. Robust improvement
requires a strictly smaller error at every point of the reference range.

| route | quantity | stock | smallest advantage | decision |
|---|---|---|---|---|
{chr(10).join(imp_rows)}

A mean-matched scalar (`c' = 1/M - d`) at `Delta/c_minus` = {g(mm['contrast'], 4)} and `h` =
{g(mm['transition_per_ns'] * us, 4)} per µs would give stopped mean {g(mm['mean_ns'], 9)} ns and capture probability
{g(mm['capture_probability'], 9)} at a late-lifetime cost of {g(mm['late_lifetime_cost'], 4)}; it is a diagnostic,
not the production estimand.

## Deferred rows

| nucleus | Q assembly | half-width | region | runtime |
|---|---|---|---|---|
| 16O | {g(oq['assembly'], 10)} | {g(oq['half_width'], 6)} | [{g(oq['region'][0], 10)}, {g(oq['region'][1], 10)}] | {record['roster']['O16']} |
| 12C | {g(cq['assembly'], 10)} | {g(cq['half_width'], 6)} | [{g(cq['region'][0], 10)}, {g(cq['region'][1], 10)}] | {record['roster']['C12']} |

The carbon prescriptions use the decay factor recovered from Suzuki, Measday and Roalsvig's natural-carbon mean
life and capture rate, not their printed rounding. The compendium carries {oc['lifetime_records']} bare-oxygen
and {cc['lifetime_records']} natural-carbon lifetime records; their `tau ± 2u` hulls are descriptive only. A
runtime pair for either needs a between-experiment term frozen before scoring and a reconstruction of the source's
finite-window estimator.

## A finding about Geant4's compiled capture table

The compiled entries for Mg, Al, Si and P equal, at four decimals, the unscaled inverse-variance average of
Suzuki, Measday and Roalsvig's own measurement and the row of their reference 42 (10^6/s). For aluminium the two
differ by many times their combined printed errors; the average carries no scale factor. This is a
convention/consistency finding, not a calibrated physical discrepancy.

| element | own | reference 42 | unscaled average | compiled | difference (combined printed errors) |
|---|---|---|---|---|---|
{fin_rows}

## What this does not establish

The actual coverage of the evaluator regions; an independently measured inclusive aluminium capture branching;
the conversion between the neutron-selected amplitude and the total-capture contrast; arbitrary early-time
detector response; pure-isotope oxygen-16 or carbon-12 lifetimes. The shared D1/D2 convention record is
`data/g4/d1/bound_decay_response_v1.json`.
"""
    return text.encode("utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "check"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.root)
    else:
        check(args.root)


if __name__ == "__main__":
    main()
