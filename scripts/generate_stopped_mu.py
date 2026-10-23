"""Compare the committed stopped-muon cells with the measured yields under the pre-registration's rule, and
generate `BENCHMARK_STOPPED_MU.md`.

  python scripts/generate_stopped_mu.py compare [--prereg <file>]

reads `data/g4/stopped_mu/preregistration.json` and `cells.jsonl`, admits every cell, builds one comparison
row per row of the pre-registration and judges it in exact arithmetic, and writes `comparison.csv` and
`verdicts.csv` beside them; it compares the stock muonic cascade of `openmucf/g4/capture_cascade.py` with
the measured ratios of `cascade_reference.csv` under `cascade_rule.json` and tests the harvested cascades of
`cascade_harvest.csv` and `cascade_transitions.csv`, writing `cascade_comparison.csv` and `cascade_consumer.csv`; it
then renders the document from those files, `builds.csv`, `reproduced.csv`,
`data/g4/d1/geant4_natural_isotopes.csv` and the vendored `G4EmCaptureCascade.cc` of `third_party/geant4/`, and its
section on the other stages of a stopped muon from the committed files of the datasets that `STAGE_FILES` names, each
read and refused by `stage_inputs` before any file is written. No float decides a byte: every number the
document takes from those files fills a slot of `render`, whose spans the prose census pins; the generator itself
defines the display precision of the row table and the citations of the prior work.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

from openmucf.g4 import d1_lifetime as lifetime
from openmucf.g4 import d1_mixture as mixture
from openmucf.g4 import d2
from openmucf.g4 import d3_contract as d3c
from openmucf.g4 import spec as g4spec

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "g4" / "stopped_mu"
MODULE = ROOT / "openmucf" / "g4" / "stopped_mu.py"
spec = importlib.util.spec_from_file_location("g4_stopped_mu_reference", MODULE)
assert spec is not None and spec.loader is not None
sm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sm
spec.loader.exec_module(sm)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def comparison(prereg: Path = DATA / "preregistration.json", cells: Path = DATA / "cells.jsonl"):
    """(policy, cells, comparison rows, verdicts) from the committed files."""
    policy = sm.load_policy(prereg)
    admitted = sm.load_cells(cells, policy)
    rows = sm.comparison_rows(policy, admitted)
    return policy, admitted, rows, sm.judge(policy, rows)


def compare(prereg: Path = DATA / "preregistration.json") -> None:
    # the files of the datasets are read and refused before any file is written
    stages = stage_inputs(ROOT)
    _, _, rows, verdicts = comparison(prereg)
    write_csv(DATA / "comparison.csv", sm.COMPARISON_COLUMNS, rows)
    write_csv(
        DATA / "verdicts.csv",
        sm.VERDICT_COLUMNS,
        [{"row_id": v.row_id, "verdict": v.verdict, "kind": v.kind, "pull": v.pull} for v in verdicts],
    )
    inp = load_inputs(prereg, stages)
    tables = canonical_tables(inp)
    write_csv(DATA / "cascade_comparison.csv", cc.COMPARISON_COLUMNS, tables.comparison)
    write_csv(DATA / "cascade_consumer.csv", cc.CONSUMER_COLUMNS, tables.consumer)
    # the audit of the written files, before the document: each file's bytes must be the canonical serialization
    # of the canonical table of the inputs, and, read raw (its header and every record, before any mapping), its
    # exact columns must give every decision and verdict and equal the canonical table as a whole
    problems = written_table_problems(
        "cascade_consumer",
        read_bytes(DATA / "cascade_consumer.csv"),
        cc.CONSUMER_COLUMNS,
        tables.consumer,
        lambda rows: consumer_problems(rows, inp, tables.consumer),
    )
    problems += written_table_problems(
        "cascade_comparison",
        read_bytes(DATA / "cascade_comparison.csv"),
        cc.COMPARISON_COLUMNS,
        tables.comparison,
        lambda rows: comparison_problems(rows, inp, tables.comparison),
    )
    if problems:
        raise ValueError(f"the written cascade tables do not certify their decisions: {problems[:3]}")
    # the document is rendered from the canonical tables, which the written files have just been proved to be
    DOCUMENT.write_text(render(inp, tables).text, encoding="utf-8", newline="\n")
    tally: dict[str, int] = {}
    for v in verdicts:
        tally[v.verdict] = tally.get(v.verdict, 0) + 1
    print(
        f"stopped_mu compare rows={len(rows)} "
        + " ".join(f"{k}={tally[k]}" for k in sorted(tally))
        + f" document={DOCUMENT.name}"
    )


# ------------------------------------------------------------------------------------------------- the document
DOCUMENT = ROOT / "BENCHMARK_STOPPED_MU.md"
NATURAL = ROOT / "data" / "g4" / "d1" / "geant4_natural_isotopes.csv"
#: Display precision of the row table: significant digits of x_se and sigma_tot, decimals of the rescue bracket
#: (rounded outward). comparison.csv carries every value in full.
SE_DIGITS = 3
BRACKET_DECIMALS = 4

KRIKLER = (
    "Krikler compared the proton spectrum Geant4 10.1 gives for muon capture in aluminium with preliminary "
    "AlCap data (proceedings of NuFact15, pp. 408–417, <https://inspirehep.net/literature/1693740>)."
)
BLACKWELL = (
    "Blackwell compared the mean number of neutrons per capture that Geant4 9.6.p02 with the Shielding 2.0 "
    "physics list gives for negative muons stopped in aluminium, silicon, calcium, iron, silver, gold and "
    "lead, counting only the neutrons the capture produces directly, with the measured means that the "
    "neutron-mean rows here also use, and reported a systematic underestimate for these elements (doctoral "
    "thesis, University of Sheffield (2015), pp. 74–76, "
    "<https://etheses.whiterose.ac.uk/id/eprint/10432/1/Tim_Blackwell_PhD_Thesis.pdf>)."
)
GAPONENKO = (
    "Gaponenko et al. compared Geant4 10.2.p03 with their proton and deuteron spectra from aluminium, showing "
    "the prediction of the precompound model and stating that the default models of muon capture agree poorly "
    "with their data (Phys. Rev. C 101, 035502 (2020), arXiv:1908.06902)."
)
KNEISSL = (
    "Kneißl reported a difference between simulated and measured rates of neutron signals that follow muon "
    "captures in lead, with Geant4 10.3, and limited its interpretation by the response of the detectors "
    "(doctoral thesis, Technische Universität München (2019), "
    "<https://mediatum.ub.tum.de/doc/1475093/document.pdf>)."
)
LARIAT = (
    "The LArIAT collaboration compared the residual nuclei of muon capture at rest in argon with Geant4 10.3.p03 "
    "(Phys. Rev. Lett. 134, 131801 (2025), arXiv:2408.05133)."
)
PRIOR_WORK = " ".join((KRIKLER, BLACKWELL, GAPONENKO, KNEISSL, LARIAT))
#: The value the line-yield test |x - 0| <= k_pass x_se compares a net line yield with; the sentence that names it
#: is emitted only on the branch that test decides, so it fills a slot of that branch.
NULL_YIELD = "zero"
#: Words the document never uses (word boundaries, case-insensitive).
FORBIDDEN = (
    "first",
    "nobody noticed",
    "no published comparison",
    "still unfixed",
    "caused",
    "introduced",
    "regression",
    "absent",
    "every physics list",
    "all physics lists",
    "both routes fail",
)


@dataclass(frozen=True)
class Slot:
    text: str


@dataclass(frozen=True)
class Rendered:
    text: str
    #: (line number from 1, start column, end column) of every span the data filled.
    slots: tuple[tuple[int, int, int], ...]


class Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.slots: list[tuple[int, int, int]] = []

    def line(self, *parts: str | Slot) -> None:
        text = ""
        for part in parts:
            if isinstance(part, Slot):
                if part.text:
                    self.slots.append((len(self.lines) + 1, len(text), len(text) + len(part.text)))
                text += part.text
            else:
                text += part
        assert text == " ".join(text.split()) or not text.strip(), f"whitespace in a generated line: {text!r}"
        self.lines.append(text)

    def blank(self) -> None:
        self.lines.append("")

    def rendered(self) -> Rendered:
        return Rendered("\n".join(self.lines) + "\n", tuple(self.slots))


def S(value: object) -> Slot:
    return Slot(str(value))


def joined(items: list[object], last: str = "and") -> list[str | Slot]:
    """[a, ", ", b, " and ", c] with every item a slot."""
    out: list[str | Slot] = []
    for i, item in enumerate(items):
        if i:
            out.append(f" {last} " if i == len(items) - 1 else ", ")
        out.append(item if isinstance(item, Slot) else S(item))
    return out


def sig(value: str, digits: int) -> str:
    """``value`` to ``digits`` significant digits, half-even, as a plain decimal."""
    with localcontext() as ctx:
        ctx.prec = 60
        d = Decimal(value)
        if d == 0:
            return "0"
        q = Decimal(1).scaleb(d.adjusted() - (digits - 1))
        return format(d.quantize(q, rounding=ROUND_HALF_EVEN), "f")


def sigma_tot(row: dict[str, str]) -> str:
    with localcontext() as ctx:
        ctx.prec = 60
        s = (Decimal(row["ref_stat"]) ** 2 + Decimal(row["ref_syst"]) ** 2 + Decimal(row["x_se"]) ** 2).sqrt()
        return sig(str(s), SE_DIGITS)


def outward(lo: str, hi: str) -> tuple[str, str]:
    q = Decimal(1).scaleb(-BRACKET_DECIMALS)
    return (
        str(Decimal(lo).quantize(q, rounding=ROUND_FLOOR)),
        str(Decimal(hi).quantize(q, rounding=ROUND_CEILING)),
    )


@dataclass
class Inputs:
    policy: dict[str, Any]
    cells: dict[tuple[str, str, str], dict[str, Any]]
    rows: list[dict[str, str]]
    verdicts: list[Any]
    builds: list[dict[str, str]]
    reproduced: list[dict[str, str]]
    natural: list[dict[str, str]]
    cascade: Cascade
    stages: Stages


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_bytes(path: Path) -> bytes:
    """The bytes of a written file, as the audit of compare reads them."""
    return path.read_bytes()


def raw_rows(name: str, data: bytes, columns: tuple[str, ...]) -> tuple[list[dict[str, str]], list[str]]:
    """The rows of a written table read raw, before any mapping can merge or drop a cell: the bytes decoded as
    UTF-8 (strictly), the records split by csv.reader, the header equal to ``columns`` exactly (each name once, in
    order, no other) and every record exactly len(columns) fields. Returns (rows, problems); the rows are mapped
    to their columns only when there is no problem."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [], [f"{name}: the file is not UTF-8"]
    records = list(csv.reader(io.StringIO(text, newline="")))
    if not records:
        return [], [f"{name}: the file is empty"]
    header, body = records[0], records[1:]
    problems = []
    if header != list(columns):
        problems.append(f"{name}: the header is not the schema, each column once and in order")
    widths = [i for i, record in enumerate(body, 2) if len(record) != len(columns)]
    if widths:
        problems.append(f"{name}: records {widths[:3]} do not have exactly {len(columns)} fields")
    if problems:
        return [], problems
    return [dict(zip(columns, record, strict=True)) for record in body], []


def written_table_problems(
    name: str,
    data: bytes,
    columns: tuple[str, ...],
    canonical: list[dict[str, str]],
    row_problems: Callable[[list[dict[str, str]]], list[str]],
) -> list[str]:
    """Every problem of a written table, from its bytes: they must be the canonical serialization of the canonical
    table (cc.csv_text, UTF-8: the display columns included, since they follow from the exact values; no BOM,
    CRLF, reordered, duplicated or extra column, or changed final newline); and, independently, the table read raw
    (raw_rows) must pass ``row_problems``, the exact recompute and the whole-table equality of its exact columns."""
    out = []
    if data != cc.csv_text(columns, canonical).encode("utf-8"):
        out.append(f"{name}: the bytes are not the canonical serialization of the table the inputs give")
    rows, problems = raw_rows(name, data, columns)
    return out + (problems if problems else row_problems(rows))


def load_inputs(prereg: Path = DATA / "preregistration.json", stages: Stages | None = None) -> Inputs:
    policy, cells, rows, verdicts = comparison(prereg)
    return Inputs(
        policy,
        cells,
        rows,
        verdicts,
        read_csv(DATA / "builds.csv"),
        read_csv(DATA / "reproduced.csv"),
        read_csv(NATURAL),
        load_cascade(policy),
        stage_inputs(ROOT) if stages is None else stages,
    )


def alternative_verdicts(inp: Inputs) -> list[tuple[str, str, str]]:
    """(row_id, verdict, alternative verdict) of every row whose cell carries an alternative treatment, the rule
    re-run with the alternative x and x_se (and the rescue edges recomputed from them)."""
    rows = []
    touched = set()
    for r in inp.rows:
        cell = inp.cells[(r["version"], r["config"], r["target"])]
        alt = cell.get("alternative")
        if alt is None:
            rows.append(r)
            continue
        q = alt["quantities"][r["predicate"]]
        new = dict(r, x=q["x"], x_se=q["x_se"])
        if r["rescue_lo"]:
            new["rescue_lo"], new["rescue_hi"] = sm.edges(
                r["ref"], r["ref_stat"], r["ref_syst"], q["x_se"], inp.policy
            )
        touched.add(r["row_id"])
        rows.append(new)
    alt_verdicts = {v.row_id: v.verdict for v in sm.judge(inp.policy, rows)}
    return [(v.row_id, v.verdict, alt_verdicts[v.row_id]) for v in inp.verdicts if v.row_id in touched]


def revision_pairs(inp: Inputs) -> tuple[list[tuple[dict[str, str], dict[str, str]]], list[str]]:
    """The (first revision, second revision) row pairs of the same predicate and configuration, and the row ids of
    the first revision whose pair differs in x, x_se, a reference cell or the verdict."""
    v1, v2 = inp.policy["setup"]["versions"]
    verdict = {v.row_id: v for v in inp.verdicts}
    by = {(r["predicate"], r["config"], r["version"]): r for r in inp.rows}
    pairs, differ = [], []
    for r in inp.rows:
        if r["version"] != v1:
            continue
        o = by[(r["predicate"], r["config"], v2)]
        pairs.append((r, o))
        keys = ("x", "x_se", "ref", "ref_stat", "ref_syst", "rescue_lo", "rescue_hi")
        if any(r[k] != o[k] for k in keys) or verdict[r["row_id"]].verdict != verdict[o["row_id"]].verdict:
            differ.append(r["row_id"])
    return pairs, differ


def counts(verdicts: list[Any]) -> dict[str, int]:
    out = {"FAIL": 0, "PASS": 0, "UNQUALIFIED": 0}
    for v in verdicts:
        out[v.verdict] += 1
    return out


def body_pairs(inp: Inputs) -> tuple[int, int]:
    v1, v2 = inp.policy["setup"]["versions"]
    same = differ = 0
    for (v, c, t), cell in sorted(inp.cells.items()):
        if v != v1:
            continue
        for a, b in zip(cell["chunks"], inp.cells[(v2, c, t)]["chunks"], strict=True):
            if a["compact_body_sha256"] == b["compact_body_sha256"]:
                same += 1
            else:
                differ += 1
    return same, differ


def reproduction(inp: Inputs) -> tuple[list[tuple[str, str, str]], list[tuple[dict[str, str], str]]]:
    """Admit `reproduced.csv` before the document states anything about it: the reproduced cells in file order
    and, for every re-run chunk whose compact text or binary differs from the record, the row and ``compact``
    or ``binary``.

    Raises ``ValueError`` for a record the screen cannot judge: no row at all; a row of a cell the committed
    cells do not hold; a reproduced cell whose rows are not exactly its committed chunks, each once; a row whose
    offset or event count differs from its committed chunk's; or reproduced cells that do not cover every
    revision, every configuration, every target basis and, when a committed cell holds an event without an
    outcome, a cell that holds one."""
    if not inp.reproduced:
        raise ValueError("reproduced.csv lists no chunk: the reproduction screen would be vacuous")
    setup = inp.policy["setup"]
    tags = setup["tags"]
    builds = {b["tag"]: b for b in inp.builds}
    by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for r in inp.reproduced:
        key = (r["version"], r["config"], r["target"])
        if key not in inp.cells:
            raise ValueError(f"reproduced.csv names {'/'.join(key)}, which the committed cells do not hold")
        by_cell.setdefault(key, []).append(r)
    faults: list[tuple[dict[str, str], str]] = []
    for key, rows in by_cell.items():
        cell = inp.cells[key]
        listed = [r["chunk"] for r in rows]
        committed = [c["chunk"] for c in cell["chunks"]]
        if sorted(listed) != sorted(committed):  # as multisets: a missing, an extra and a repeated row differ
            raise ValueError(
                f"reproduced.csv lists the chunks {listed} of {'/'.join(key)}, whose committed chunks are {committed}"
            )
        for r in rows:
            [ch] = [c for c in cell["chunks"] if c["chunk"] == r["chunk"]]
            for field in ("offset", "events"):
                if r[field] != str(ch[field]):
                    raise ValueError(
                        f"reproduced.csv gives {'/'.join(key)} {r['chunk']} {field} {r[field]}, the committed "
                        f"chunk {ch[field]}"
                    )
            if r["compact_text_sha256"] != ch["compact_text_sha256"]:
                faults.append((r, "compact"))
            if r["binary_sha256"] != builds[tags[r["version"]]]["rebuilt_binary_sha256"]:
                faults.append((r, "binary"))
    basis = {t["target"]: t["basis"] for t in setup["targets"]}
    covered = {
        "revisions": ({k[0] for k in by_cell}, set(setup["versions"])),
        "configurations": ({k[1] for k in by_cell}, set(setup["configs"])),
        "target bases": ({basis[k[2]] for k in by_cell}, set(basis.values())),
        "events without an outcome": (
            {bool(inp.cells[k]["n_no_outcome"]) for k in by_cell} & {True},
            {bool(c["n_no_outcome"]) for c in inp.cells.values()} & {True},
        ),
    }
    short = [what for what, (got, want) in covered.items() if got != want]
    if short:
        raise ValueError(f"the reproduced cells do not cover the {', '.join(short)} of the committed cells")
    return list(by_cell), faults


# ------------------------------------------------------------------------------------- the muonic X-ray cascade
CASCADE_MODULE = ROOT / "openmucf" / "g4" / "capture_cascade.py"
_cspec = importlib.util.spec_from_file_location("g4_capture_cascade_reference", CASCADE_MODULE)
assert _cspec is not None and _cspec.loader is not None
cc = importlib.util.module_from_spec(_cspec)
sys.modules[_cspec.name] = cc
_cspec.loader.exec_module(cc)
VENDORED = ROOT / "third_party" / "geant4"
#: The width of the integer type ``G4int`` names (``using G4int = int`` in ``G4Types.hh``) on the platforms the
#: harvest ran on; ``cascade_domain`` checks that the module's domain is exactly where Z^4 fits it.
INT_BITS = 32
#: Display precision of the cascade's row table (decimals) and of the domain extremes (decimals of a percent).
CASCADE_DECIMALS = 6
DOMAIN_DECIMALS = 3
SOURCE_NAMES = {
    "orth1982": "Orth et al.",
    "ensdf_measday2007": "ENSDF (Measday et al.)",
    "kirch1999": "Kirch et al.",
}
CATALDO = (
    "Cataldo et al. compared a Geant4-based simulation in which a single de-excitation from a MuDirac database "
    "replaces the cascade with a silicon spectrum measured at ISIS, found the intensities of the simulated spectra "
    "not well reproduced, and wrote that implementing their database in Geant4 would require an alternative class "
    "to `G4EmCaptureCascade` (Condens. Matter 8, 101 (2023), <https://doi.org/10.3390/condmat8040101>)."
)
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


@dataclass
class Cascade:
    rule: dict[str, Any]
    reference: list[dict[str, str]]
    harvest: list[dict[str, str]]
    transitions: list[dict[str, str]]
    sources: dict[str, str]


def load_cascade(policy: dict[str, Any]) -> Cascade:
    tags = policy["setup"]["tags"]
    return Cascade(
        json.loads((DATA / "cascade_rule.json").read_text(encoding="utf-8")),
        read_csv(DATA / "cascade_reference.csv"),
        read_csv(DATA / "cascade_harvest.csv"),
        read_csv(DATA / "cascade_transitions.csv"),
        {
            v: (VENDORED / tags[v] / "G4EmCaptureCascade.cc").read_text(encoding="utf-8")
            for v in policy["setup"]["versions"]
        },
    )


def cascade_binding(inp: Inputs) -> Any:
    """The rule statements bound to the vendored source of the first revision, the second's equal to them."""
    v1, v2 = inp.policy["setup"]["versions"]
    return cc.bind_source(inp.cascade.sources[v1], inp.cascade.sources[v2])


def cell_of(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["version"], row["config"], row["target"])


def cascade_harvest(inp: Inputs) -> dict[tuple[str, str, str], dict[str, int]]:
    """Admit the harvested cascades before anything is stated about them, and return each cell's retained
    counts.

    Raises ``ValueError`` when the rule does not name the same cells, in the same order, on both revisions,
    or when the harvest is not the rule's: its cells not exactly the rule's cells in the rule's order; a cell's events, seed base, route, Z or A not the rule's and the pre-registration's; a binary
    that ``builds.csv`` does not record for its revision (the binary of the committed cells or the rebuild); a
    refused stop, or stops, reconstructed chains
    and events that differ; transition rows other than the module's transitions in its order; a count above the
    stops; entries into the K shell that do not sum to the stops; Auger or photon totals other than the sums of
    the transition counts."""
    c = inp.cascade
    h = c.rule["harvest"]
    setup = inp.policy["setup"]
    cells = [(x["version"], x["config"], x["target"]) for x in h["cells"]]
    v1, v2 = setup["versions"]
    by_version = {v: [(x["config"], x["target"]) for x in h["cells"] if x["version"] == v] for v in (v1, v2)}
    if by_version[v1] != by_version[v2] or len(h["cells"]) != 2 * len(by_version[v1]):
        raise ValueError("the harvest does not name the same cells, in the same order, on both revisions")
    if [cell_of(r) for r in c.harvest] != cells:
        raise ValueError("cascade_harvest.csv does not hold the rule's cells in the rule's order")
    builds = {b["tag"]: {b["reference_binary_sha256"], b["rebuilt_binary_sha256"]} for b in inp.builds}
    keys = cc.transition_keys()
    by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for t in c.transitions:
        by_cell.setdefault(cell_of(t), []).append(t)
    if set(by_cell) != set(cells):
        raise ValueError("cascade_transitions.csv does not hold the rule's cells")
    out = {}
    for r in c.harvest:
        cell = cell_of(r)
        name = "/".join(cell)
        args = next(t["arguments"] for t in setup["targets"] if t["target"] == r["target"])
        expected = {
            "events": str(h["events"]),
            "seed_base": str(h["seed_base"]),
            "route": setup["configurations"][r["config"]]["route"],
            "Z": args[1],
            "A": args[3],
            "refused": "0",
            "stops": r["events"],
            "chains": r["events"],
        }
        for k, v in expected.items():
            if r[k] != v:
                raise ValueError(f"{name}: {k} {r[k]} where {v} is required")
        if r["binary_sha256"] not in builds[setup["tags"][r["version"]]]:
            raise ValueError(f"{name}: binary_sha256 {r['binary_sha256']} is no binary builds.csv records")
        rows = by_cell[cell]
        if [t["transition"] for t in rows] != keys:
            raise ValueError(f"{name}: the transition rows are not the module's transitions in its order")
        counts = {t["transition"]: int(t["stops"]) for t in rows}
        stops = int(r["stops"])
        if any(not 0 <= v <= stops for v in counts.values()):
            raise ValueError(f"{name}: a transition count outside 0..stops")
        entries = sum(counts[f"g{n}->1"] for n in range(2, cc.LEVELS + 1)) + counts["e2->1"]
        if entries != stops:
            raise ValueError(f"{name}: {entries} entries into the K shell for {stops} stops")
        augers = sum(v for k, v in counts.items() if k.startswith("e"))
        photons = sum(v for k, v in counts.items() if k.startswith("g"))
        if (str(augers), str(photons)) != (r["augers"], r["photons"]):
            raise ValueError(f"{name}: Auger and photon totals are not the sums of the transition counts")
        out[cell] = cc.observed_counts(counts)
    return out


def cascade_consumer(inp: Inputs, rules: dict[str, int] | None = None) -> list[dict[str, str]]:
    """The rows of `cascade_consumer.csv`: every retained count of every harvested cell decided by the exact
    binomial test of the rule against the model of ``rules`` (default the stock rules)."""
    observed = cascade_harvest(inp)
    rows = []
    for r in inp.cascade.harvest:
        rows += cc.consumer_rows(
            r, observed[cell_of(r)], int(r["Z"]), int(r["stops"]), inp.cascade.rule["harvest"], rules
        )
    return rows


def read_back(columns: tuple[str, ...], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows as a reader of the CSV sees them: written by the module's csv_text and read back, so every check
    reads what the file holds."""
    return list(csv.DictReader(io.StringIO(cc.csv_text(columns, rows))))


#: The identity of a row of each cascade table: a table holds each identity once, in the canonical order.
CONSUMER_KEY = ("version", "config", "target", "count")
COMPARISON_KEY = ("source", "target", "observable", "k")


@dataclass(frozen=True)
class Tables:
    """The two cascade tables, as a reader of their CSV files sees them."""

    consumer: list[dict[str, str]]
    comparison: list[dict[str, str]]


def canonical_tables(inp: Inputs) -> Tables:
    """The canonical tables of the authoritative inputs (the rule, the harvest, the reference and the model),
    computed by the module and read back as a reader of the CSV files sees them."""
    return Tables(
        read_back(cc.CONSUMER_COLUMNS, cascade_consumer(inp)),
        read_back(cc.COMPARISON_COLUMNS, cascade_comparison(inp)),
    )


def table_problems(
    name: str,
    rows: list[dict[str, str]],
    canonical: list[dict[str, str]],
    columns: tuple[str, ...],
    display: tuple[str, ...],
    key: tuple[str, ...],
) -> list[str]:
    """Every way a table differs from the canonical one in its exact projection: every column but the
    documented display-only ones. The table must not be empty, must hold each canonical row identity exactly once
    and no other, in the canonical order, and every cell outside ``display`` must equal the canonical cell (the
    exact columns are in their canonical text, so equal text is equal value)."""
    if not canonical:
        return [f"{name}: the inputs give no rows"]
    if not rows:
        return [f"{name}: the table has no rows (an empty or header-only file)"]
    out = []
    ident = [tuple(r.get(k) for k in key) for r in rows]
    expected = [tuple(r[k] for k in key) for r in canonical]
    seen: dict[tuple[str | None, ...], int] = {}
    for k in ident:
        seen[k] = seen.get(k, 0) + 1
    out += [f"{name} {k}: the row appears {c} times" for k, c in seen.items() if c > 1]
    out += [f"{name} {k}: a row the inputs give is missing" for k in expected if k not in seen]
    out += [f"{name} {k}: a row the inputs do not give" for k in seen if k not in set(expected)]
    if not out and ident != expected:
        out.append(f"{name}: the rows are not in the canonical order")
    by_key = dict(zip(expected, canonical, strict=True))
    for k, r in zip(ident, rows, strict=True):
        want = by_key.get(k)
        if want is None:
            continue
        if tuple(r) != columns:
            out.append(f"{name} {k}: the columns are not the schema")
            continue
        out += [
            f"{name} {k}: {c} is not what the inputs give"
            for c in columns
            if c not in display and r[c] != want[c]
        ]
    return out


def consumer_problems(
    rows: list[dict[str, str]], inp: Inputs, canonical: list[dict[str, str]] | None = None
) -> list[str]:
    """Every problem of a consumer table read from text, as a whole: the module recomputes every decision from
    the exact columns of each row (cc.consumer_problems, with the rule's per-side budget), and the table's exact
    projection must be the canonical table of the inputs (table_problems: every row once, in order, every
    exact field and decision). The display columns are never read. ``canonical`` defaults to the table the
    inputs give."""
    if canonical is None:
        canonical = read_back(cc.CONSUMER_COLUMNS, cascade_consumer(inp))
    hv = inp.cascade.rule["harvest"]
    out = cc.consumer_problems(rows, cc.alpha_side(hv, len(cc.retained_counts(cc.transitions(1)))))
    return out + table_problems(
        "cascade_consumer", rows, canonical, cc.CONSUMER_COLUMNS, cc.CONSUMER_DISPLAY, CONSUMER_KEY
    )


def cascade_comparison(inp: Inputs) -> list[dict[str, str]]:
    return cc.comparison_rows(inp.cascade.reference, inp.cascade.rule)


def comparison_problems(
    rows: list[dict[str, str]], inp: Inputs, canonical: list[dict[str, str]] | None = None
) -> list[str]:
    """Every problem of a comparison table read from text, as a whole: the module recomputes every verdict from
    the exact columns of each row (cc.comparison_problems), and the table's exact projection must be the
    canonical table of the inputs (table_problems: every row once, in order, its metadata, model, measured, the
    four variances and the verdicts, with the sensitivity rows). The display columns are never read.
    ``canonical`` defaults to the table the inputs give."""
    if canonical is None:
        canonical = read_back(cc.COMPARISON_COLUMNS, cascade_comparison(inp))
    out = cc.comparison_problems(rows, inp.cascade.rule["ratios"])
    return out + table_problems(
        "cascade_comparison", rows, canonical, cc.COMPARISON_COLUMNS, cc.COMPARISON_DISPLAY, COMPARISON_KEY
    )


def observable_label(obs: str, k: str) -> list[str | Slot]:
    if obs == "kb_over_ka":
        return ["Kβ/Kα"]
    if obs == "ka_over_ktot":
        return ["Kα/ΣK"]
    return ["(n ≥ ", S(k), ")/Kα"]


def percent(x: Fraction, places: int, up: bool) -> str:
    return cc.outward(100 * x, places, up)


def q(value: str, places: int = CASCADE_DECIMALS) -> str:
    return str(Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN))


def render_cascade(w: Writer, inp: Inputs, tables: Tables) -> None:
    c = inp.cascade
    v1, v2 = inp.policy["setup"]["versions"]
    tags = inp.policy["setup"]["tags"]
    b = cascade_binding(inp)
    rules, ln = b.rules, b.lines
    top = rules["start_level"] + 1
    zmax = cc.Z_MAX
    if not zmax ** rules["power"] <= 2 ** (INT_BITS - 1) - 1 < (zmax + 1) ** rules["power"]:
        raise ValueError("the module's domain is not where the integer power fits the integer type")
    z_pow = "Z" + str(rules["power"]).translate(SUPERSCRIPT)
    w.line("## The muonic X-ray cascade")
    w.blank()
    # ---- a. what the code does
    w.line(
        "**What the code does.** In unpatched Geant4, `G4EmCaptureCascade::ApplyYourself` "
        "(`third_party/geant4/",
        S(tags[v1]),
        "/G4EmCaptureCascade.cc`, lines ",
        S(ln["n_elec"]),
        "–",
        S(ln["loop"]),
        ") places a stopped negative muon on the level of principal quantum number n = ",
        S(top),
        ", emits that level's binding energy as an electron (line ",
        S(ln["capture_electron"]),
        ") and then takes the muon down level by level until it reaches the K shell (line ",
        S(ln["loop"]),
        "). With Z the atomic number and U a uniform random number from the unit interval, an Auger electron "
        "takes the muon to the next level down when (",
        S(z_pow),
        " + ",
        S(rules["auger_weight"]),
        ") U < ",
        S(rules["auger_weight"]),
        " (line ",
        S(ln["auger_test"]),
        "), which is allowed only while fewer than Z electrons, the capture electron included, have been emitted "
        "(lines ",
        S(ln["n_elec"]),
        " and ",
        S(ln["auger_count0"]),
        "); otherwise a photon takes it to the next level down with ",
        S(rules["photon_weight"]),
        " times the probability of each lower level, every lower level being equally likely (lines ",
        S(ln["photon_draw"]),
        "–",
        S(ln["photon_clamp"]),
        "). The rule statements of `third_party/geant4/",
        S(tags[v2]),
        "/G4EmCaptureCascade.cc` are the same once whitespace is removed. The source comments name Mukhopadhyay "
        "(Phys. Rep. 30, 1 (1977)) for the probabilities and Wu and Wilets (Ann. Rev. Nucl. Sci. 19, 527 (1969)) "
        "for the photon cascade; this document does not compare those papers with the code. Which transition "
        "the cascade takes depends on Z alone: the mass number enters only the energies of the levels.",
    )
    w.blank()
    # ---- b. expected transitions
    keys = cc.transition_keys()
    n_photon = sum(1 for k in keys if k.startswith("g"))
    n_auger = len(keys) - n_photon
    w.line(
        "**Expected transitions.** Taking U as a continuous uniform draw, `openmucf/g4/capture_cascade.py` "
        "propagates the exact probability of every state (level, electrons emitted) from n = ",
        S(top),
        " to the K shell in rational arithmetic and returns the expected number per stopped muon of each of the ",
        S(len(keys)),
        " transitions the rules allow: a photon from each level to each lower level (",
        S(n_photon),
        ") and an Auger electron from each level to the next (",
        S(n_auger),
        "). These are the expectations of the coded rules, not of the random-number generator's discrete "
        "output. They are computed for every Z from ",
        S(1),
        " to ",
        S(zmax),
        ": the source forms ",
        S(z_pow),
        " as a product of `G4int`, a ",
        S(INT_BITS),
        "-bit `int`, which holds it up to Z = ",
        S(zmax),
        " and overflows from Z = ",
        S(zmax + 1),
        ".",
    )
    w.blank()
    # ---- c. the compiled cascade
    hv = c.rule["harvest"]
    rows = tables.consumer
    problems = consumer_problems(rows, inp, canonical=rows)
    if problems:
        raise ValueError(f"cascade_consumer rows break their schema: {problems[:3]}")
    rejected = [r for r in rows if r["decision"] != "pass"]
    if rejected:
        raise ValueError(
            f"the harvested cascade rejects {len(rejected)} counts: the module does not describe it"
        )
    per_cell = {cell_of(r): r for r in c.harvest}
    n_counts = len(cc.retained_counts(cc.transitions(1)))
    budget = Fraction(hv["budget_per_cell"])
    a_side = cc.alpha_side(hv, n_counts)
    bertini = [x["target"] for x in hv["cells"] if x["config"] == "bertini" and x["version"] == v1]
    others = [x for x in hv["cells"] if x["config"] != "bertini" and x["version"] == v1]
    w.line(
        "**The compiled cascade.** The `cascade` command of `cpp/stopped_muon/stopped_muon.py` ran ",
        S(hv["events"]),
        " events in each of ",
        S(len(hv["cells"])),
        " cells with the registered builds of both revisions: the single isotopes ",
        *joined(bertini),
        " in the `bertini` configuration (the default route) and ",
        *joined([x["target"] for x in others]),
        " also in `",
        S(others[0]["config"]),
        "` (the muonic-atom route), each event i seeded with ",
        S(hv["seed_base"]),
        " + i. For every stop the module rebuilt the cascade from the records energy for energy: the electron "
        "carrying the binding energy of the capture level, then each electron the energy difference of the "
        "level it leaves and the next level down, admitted only while the Auger count is below Z, and each photon "
        "the energy difference of the level it leaves and a single lower level, ending in the K shell. "
        "Every event was a single stop whose cascade was rebuilt; the table gives the counts "
        "(`data/g4/stopped_mu/cascade_harvest.csv`):",
    )
    w.blank()
    w.line(
        "| revision | configuration | target | Z | stops | cascades rebuilt | refused | Auger electrons | photons |"
    )
    w.line("|---|---|---|---|---|---|---|---|---|")
    for r in c.harvest:
        w.line(
            "| ",
            S(r["version"]),
            " | ",
            S(r["config"]),
            " | ",
            S(r["target"]),
            " | ",
            S(r["Z"]),
            " | ",
            S(r["stops"]),
            " | ",
            S(r["chains"]),
            " | ",
            S(r["refused"]),
            " | ",
            S(r["augers"]),
            " | ",
            S(r["photons"]),
            " |",
        )
    w.blank()
    w.line(
        "Each cell's ",
        S(n_counts),
        " retained counts, the number of stops taking each of the ",
        S(len(keys)),
        " transitions and the numbers entering the K shell by a photon from n ≥ ",
        *joined([S(k) for k in cc.UNIONS], "and"),
        ", are each binomial under the module's expectation, since a stop leaves each level at most once. "
        "Each count is tested exactly: the tail of its binomial distribution beyond the count, on the side "
        "away from the mode, is bounded in outward-rounded interval arithmetic, and the count is rejected when "
        "the upper bound is at most ",
        S(f"{a_side.numerator}/{a_side.denominator}"),
        ", the budget of ",
        S(hv["budget_per_cell"]),
        " per cell split over the ",
        S(n_counts),
        " counts and both sides. The probability of a false rejection is then at most ",
        S(hv["budget_per_cell"]),
        " per cell and ",
        S(f"{len(hv['cells']) * budget}"),
        " over the ",
        S(len(hv["cells"])),
        " cells, whatever the dependence between counts and cells; no asymptotic distribution enters. All ",
        S(len(rows)),
        " counts lie inside their acceptance regions (`data/g4/stopped_mu/cascade_consumer.csv`): in each "
        "of these counts the compiled cascades agree with the expected transitions, and the agreement claimed "
        "is exactly that set of counts.",
    )
    w.blank()
    weight = int(hv["drills"]["photon_weight_drill"])
    drill = cascade_consumer(inp, {**rules, "photon_weight": weight})
    per = [sum(1 for r in drill if cell_of(r) == cell_of(h) and r["decision"] == "reject") for h in c.harvest]
    if all(per):
        w.line(
            "The test can tell a neighbouring rule apart: against the expectation of the same rules with photon "
            "weight ",
            S(weight),
            " instead of ",
            S(rules["photon_weight"]),
            ", it rejects every cell (",
            S(min(per)),
            " to ",
            S(max(per)),
            " of the ",
            S(n_counts),
            " counts of a cell).",
        )
    else:
        w.line(
            "Against the expectation of the same rules with photon weight ",
            S(weight),
            " instead of ",
            S(rules["photon_weight"]),
            ", the test rejects ",
            S(sum(1 for x in per if x)),
            " of the ",
            S(len(per)),
            " cells.",
        )
    w.blank()
    rev_pairs = [
        (per_cell[(v1, x["config"], x["target"])], per_cell[(v2, x["config"], x["target"])])
        for x in hv["cells"]
        if x["version"] == v1
    ]
    rev_differ = [p for p in rev_pairs if p[0]["sequences_sha256"] != p[1]["sequences_sha256"]]
    route_pairs = []
    for v in (v1, v2):
        for x in others:
            route_pairs.append(
                (per_cell[(v, "bertini", x["target"])], per_cell[(v, x["config"], x["target"])])
            )
    route_differ = [p for p in route_pairs if p[0]["sequences_sha256"] != p[1]["sequences_sha256"]]
    distinct = len({r["sequences_sha256"] for r in c.harvest})
    if not rev_differ:
        w.line(
            "Both revisions rebuilt identical cascades, stop for stop, in each of the ",
            S(len(rev_pairs)),
            " pairs of cells that share a configuration and target (equal sha256 of the reconstructed "
            "transitions), so each pair is a single sample, not a replication.",
        )
    else:
        w.line(
            "These pairs of cells that share a configuration and target rebuilt different cascades on the "
            "revisions: ",
            *joined([S(f"{p[0]['config']} {p[0]['target']}") for p in rev_differ]),
            "; the other ",
            S(len(rev_pairs) - len(rev_differ)),
            " pairs rebuilt identical cascades and are a single sample each.",
        )
    if not route_differ:
        w.line(
            "Both routes rebuilt identical cascades on ",
            *joined(sorted({p[0]["target"] for p in route_pairs})),
            " on each revision. The ",
            S(len(c.harvest)),
            " cells therefore hold ",
            S(distinct),
            " distinct samples of cascades.",
        )
    else:
        w.line(
            "The routes rebuilt different cascades on ",
            *joined([S(f"{p[0]['version']} {p[0]['target']}") for p in route_differ]),
            ". The ",
            S(len(c.harvest)),
            " cells hold ",
            S(distinct),
            " distinct samples of cascades.",
        )
    w.blank()
    # ---- d. over the domain, then the prior work after this first sentence that states a stock value
    zs = range(1, zmax + 1)
    ratio = {z: cc.ka_over_ktot(z) for z in zs}
    zhi = max(zs, key=lambda z: ratio[z])
    zlo = min(zs, key=lambda z: ratio[z])
    free = range(
        cc.LEVELS, zmax + 1
    )  # Z - 1 >= the levels below the capture level: the Auger cap cannot bind
    kb = [cc.kb_over_ka(z) for z in free]
    hi8 = [cc.high_over_ka(z, 8) for z in free]
    w.line(
        "**Over the domain.** Over Z from ",
        S(1),
        " to ",
        S(zmax),
        ", Kα/ΣK, the photons from n = ",
        S(2),
        " into the K shell over all photons into it, is at most ",
        S(percent(ratio[zhi], DOMAIN_DECIMALS, True)),
        " % (at Z = ",
        S(zhi),
        ") and at least ",
        S(percent(ratio[zlo], DOMAIN_DECIMALS, False)),
        " % (at Z = ",
        S(zlo),
        "), rounded outward.",
    )
    w.blank()
    w.line("**Prior work.** ", CATALDO)
    w.blank()
    w.line(
        "For Z ≥ ",
        S(cc.LEVELS),
        ", where Z exceeds the ",
        S(cc.LEVELS - 1),
        " levels below the capture level and the Auger cap cannot bind, Kβ/Kα lies in [",
        S(percent(min(kb), DOMAIN_DECIMALS, False)),
        " %, ",
        S(percent(max(kb), DOMAIN_DECIMALS, True)),
        " %] and (n ≥ ",
        S(8),
        ")/Kα in [",
        S(percent(min(hi8), DOMAIN_DECIMALS, False)),
        " %, ",
        S(percent(max(hi8), DOMAIN_DECIMALS, True)),
        " %], rounded outward. Each value is computed from the enumeration, not from a formula.",
    )
    w.blank()
    # ---- e. against measurement
    comp = tables.comparison
    problems = comparison_problems(comp, inp, canonical=comp)
    if problems:
        raise ValueError(f"cascade_comparison rows break their schema: {problems[:3]}")
    w.line(
        "**Against measurement.** The ratios of the cascade are compared with the muonic K-series intensities "
        "of Orth et al. (Phys. Rev. A 25, 876 (1982), Table I: ratios to Kα, condensed targets), the ENSDF "
        "evaluations (Basunia and Hurst, Nucl. Data Sheets 134, 1 (2016); Basunia, Nucl. Data Sheets 114, 1189 "
        "(2013)) of the measurement of Measday et al. (Phys. Rev. C 76, 035504 (2007); solid aluminium and natural "
        "silicon) and Kirch et al. (Phys. Rev. A 59, 3375 (1999), Tables I–III: low-pressure gases), transcribed "
        "twice in `data/g4/stopped_mu/cascade_reference.csv`. A ratio of a pair of printed members carries a "
        "relative error equal to the root sum of squares of the members' relative errors, the errors "
        "independent; Kα/ΣK is Kα over the sum of the printed "
        "members, with σ = σ(Kα)/ΣK and the sum held fixed (the convention that decides), and (n ≥ k)/Kα sums "
        "the model over the same k as the measured grouped member. A row HOLDS when |m − d| ≤ ",
        S(c.rule["ratios"]["holds_within"]),
        " σ, is in TENSION when it lies beyond that and within ",
        S(c.rule["ratios"]["tension_within"]),
        " σ, and FAILS beyond; the decision is made on squares in exact arithmetic and the pull (m − d)/σ "
        "decides nothing. The model carries no uncertainty: it is the exact expectation of the code. The "
        "condensed-phase rows are the primary set and the gas rows a secondary set, counted apart:",
    )
    w.blank()
    w.line("| source | set | observable | rows | HOLDS | TENSION | FAILS |")
    w.line("|---|---|---|---|---|---|---|")
    groups: list[tuple[str, str, str]] = []
    for r in comp:
        g = (r["source"], r["set"], r["observable"])
        if g not in groups:
            groups.append(g)
    for src, st, obs in groups:
        sel = [r for r in comp if (r["source"], r["set"], r["observable"]) == (src, st, obs)]
        ks = sorted({r["k"] for r in sel})
        label: list[str | Slot] = (
            ["(n ≥ k)/Kα, k = ", *joined([S(k) for k in ks], "or")]
            if obs == "high_over_ka" and len(ks) > 1
            else observable_label(obs, ks[0])
        )
        w.line(
            "| ",
            SOURCE_NAMES[src],
            " | ",
            S(st),
            " | ",
            *label,
            " | ",
            S(len(sel)),
            *[x for v in cc.VERDICTS for x in (" | ", S(sum(1 for r in sel if r["verdict"] == v)))],
            " |",
        )
    w.blank()
    ensdf = [r for r in comp if r["source"] == "ensdf_measday2007" and r["observable"] == "ka_over_ktot"]
    above = all(cc.read_fraction(r, "measured") > ratio[zhi] for r in ensdf)
    w.line(
        "The measured Kα/ΣK of ",
        *joined([S(r["target"]) for r in ensdf]),
        " (ENSDF) is ",
        *joined([S(cc.fixed(100 * cc.read_fraction(r, "measured"), 2)) for r in ensdf]),
        " %, ",
        "above the largest value the code gives for any Z in its domain."
        if above
        else "within the code's range.",
    )
    w.blank()
    knorm = [r for r in comp if r["observable"] == "ka_over_ktot"]
    flip = [r for r in knorm if r["verdict_independent"] != r["verdict"]]
    w.line(
        "Under independent propagation of every member instead of the convention, the verdict changes for ",
        S(len(flip)) if flip else "none",
        " of the ",
        S(len(knorm)),
        " Kα/ΣK rows" + (":" if flip else "."),
        *(
            [
                " ",
                *[
                    x
                    for i, r in enumerate(flip)
                    for x in (
                        "; " if i else "",
                        S(r["target"]),
                        " goes from ",
                        S(r["verdict"]),
                        " (pull ",
                        S(r["pull"]),
                        ") to ",
                        S(r["verdict_independent"]),
                        " (pull ",
                        S(r["pull_independent"]),
                        ")",
                    )
                ],
                ".",
            ]
            if flip
            else []
        ),
    )
    scanned = [r for r in comp if r["verdict_rho_minus1"]]
    moved = [r for r in scanned if len({r["verdict"], r["verdict_rho_minus1"], r["verdict_rho_plus1"]}) > 1]
    w.line(
        "For the ",
        S(len(scanned)),
        " ratios formed from a pair of printed members, a correlation between the members' errors anywhere from −",
        S(1),
        " to +",
        S(1),
        " ",
        "changes no verdict." if not moved else "changes these verdicts: ",
        *(
            [
                x
                for i, r in enumerate(moved)
                for x in (
                    ", " if i else "",
                    S(r["target"]),
                    " ",
                    *observable_label(r["observable"], r["k"]),
                    " (",
                    S(r["verdict_rho_minus1"]),
                    " to ",
                    S(r["verdict_rho_plus1"]),
                    ")",
                )
            ]
            + ["."]
            if moved
            else []
        ),
    )
    w.blank()
    w.line(
        "**The rows.** The model, measured ratio and σ are shown to ",
        S(CASCADE_DECIMALS),
        " decimals; `data/g4/stopped_mu/cascade_comparison.csv` gives the model to ",
        S(9),
        " decimals with every sensitivity column, and every quantity a verdict is decided from as an exact "
        "numerator and denominator, from which the verdicts are recomputed. The silicon rows of ENSDF are a measurement on natural "
        "silicon filed under the ",
        S(28),
        "Si evaluation; the harvested ",
        S(next(x["target"] for x in c.rule["harvest"]["cells"] if x["target"].startswith("Si-"))),
        " cell above is a check of the code, not of that measurement.",
    )
    w.blank()
    w.line("| source | target | Z | phase | observable | model | measured | σ | pull | verdict |")
    w.line("|---|---|---|---|---|---|---|---|---|---|")
    for r in comp:
        w.line(
            "| ",
            SOURCE_NAMES[r["source"]],
            " | ",
            S(r["target"]),
            " | ",
            S(r["Z"]),
            " | ",
            S(r["phase"]),
            " | ",
            *observable_label(r["observable"], r["k"]),
            " | ",
            S(q(r["model"])),
            " | ",
            S(q(r["measured"])),
            " | ",
            S(q(r["sigma"])),
            " | ",
            S(r["pull"]),
            " | ",
            S(r["verdict"]),
            " |",
        )
    w.blank()
    # ---- f. the rule and when it was fixed
    w.line(
        "**The rule and when it was fixed** (`data/g4/stopped_mu/cascade_rule.json`). The table states when "
        "each part was fixed and what had already been seen:"
    )
    w.blank()
    w.line("| part | fixed (UTC) | relation to the data | contents |")
    w.line("|---|---|---|---|")
    for part in c.rule["fixed"]:
        bold = "**" if part["post_data"] else ""
        w.line(
            "| ",
            S(part["part"]),
            " | ",
            S(part["fixed_utc"]),
            " | ",
            bold,
            S(part["relation_to_data"]),
            bold,
            " | ",
            S(part["contents"]),
            " |",
        )
    w.blank()
    for seen in c.rule["seen_before_fixing"]:
        w.line("- ", S(seen["part"]), ": seen before it was fixed, ", S(seen["seen"]), ".")
    w.line("- the correlation scan: ", S(c.rule["ratios"]["correlation_scan"]), ".")
    w.blank()
    # ---- h. limits
    sums = {}
    for r in c.reference:
        if r["source"] == "ensdf_measday2007":
            sums[r["target"]] = sums.get(r["target"], Fraction(0)) + cc.read_exact(r["value"], "value")
    w.line(
        "**Limits.** No Kα per stopped muon is compared, because the quoted intensities assume that the K "
        "series accounts for every stop and so do not establish photons per stop independently: Measday et al. "
        "give their intensities per muon stop, normalised so that the series sums to all the stops (Phys. Rev. C "
        "76, 035504 (2007), Tables II and XII), and state in their text that this assumes the series accounts "
        "for every stop; the ENSDF evaluations repeat them "
        "as intensities per capture, whose members sum to ",
        *joined([S(cc.fixed(sums[t], 2)) for t in sums]),
        " %; and Kirch et al. take the K series to account for every muon to extract absolute yields. "
        "The model has no phase or chemistry input, so the liquid and gas rows also test effects it does not "
        "represent. The ENSDF rows read the evaluations, not the paper's own tables. U is taken as a continuous "
        "uniform draw. The comparison tests the branching rules together and does not establish which of them "
        "a disagreement comes from. The energies of these transitions are not compared here "
        "(`DATASET_D3.md`). Unpatched builds only.",
    )
    w.blank()


# ------------------------------------------------------------------------ the other stages of a stopped muon
#: The committed files of the datasets that the section on the other stages of a stopped muon restates, relative
#: to the repository root. The section reads these and no other file: it states no result of the transported
#: natural-target comparison, and the integrity of each file is the job of its own dataset's checks.
STAGE_FILES = {
    "mean_life_comparison": "data/g4/d1/mean_life_comparison.csv",
    "record_classes": "data/g4/d1/record_classes.csv",
    "selector_summary": "data/g4/d2/selector_summary.csv",
    "shell_projection": "data/g4/d3/shell_projection.csv",
    "incompatible_groups": "data/g4/d3/incompatible_groups.csv",
    "numerical_check_status": "data/g4/d3/numerical_check_status.json",
}
#: The columns the section's wording was written for. A table whose owning module names other columns is refused
#: until someone re-reads the wording against the new table.
STAGE_COLUMNS = {
    "mean_life_comparison": (
        "source",
        "locator",
        "label",
        "z",
        "basis",
        "measured_state",
        "key_a",
        "profile",
        "lookup",
        "population",
        "mean_life_ns",
        "mean_life_unc_ns",
        "measured_rate",
        "measured_rate_unc",
        "printed_capture_rate",
        "capture_rate",
        "capture_rate_unc",
        "decay_rate_bound",
        "decay_rate_helper",
        "rate_bound",
        "rate_helper",
        "within_bound",
        "within_helper",
    ),
    "record_classes": (
        "profile",
        "z",
        "a",
        "isotope_resolved",
        "needs_verification",
        "target_kind",
        "target_evidence",
        "value_lineage",
        "lineage_evidence",
        "answer_class",
    ),
    "selector_summary": (
        "scope",
        "rows",
        "to_oxygen",
        "hydrogenous",
        "selector_above",
        "ratio_min",
        "ratio_median",
        "ratio_max",
        "z_ratio_closer",
        "z_ratio_above",
        "selector_above_z_ratio",
    ),
    "shell_projection": (
        "source",
        "Z",
        "A",
        "transition",
        "quantity",
        "consumer_quantity",
        "initial_n",
        "final_n",
        "measured_keV",
        "unc_keV",
        "tol_keV",
        "solver_keV",
        "consumer_keV",
        "stock_keV",
        "solver_residual_keV",
        "consumer_residual_keV",
        "stock_residual_keV",
        "solver_within",
        "consumer_within",
        "stock_within",
        "consumer_closer_than_stock",
        "representation_error_keV",
        "solver_numeric_target_keV",
        "solver_numeric_shifts_keV",
        "solver_numeric_qualification",
        "consumer_numeric_target_keV",
        "consumer_numeric_shifts_keV",
        "consumer_numeric_qualification",
        "locator",
        "copy_read",
    ),
    "incompatible_groups": (
        "source",
        "Z",
        "A",
        "initial_n",
        "final_n",
        "lines",
        "quantities",
        "lower_keV",
        "upper_keV",
        "gap_keV",
        "empty",
        "basis",
    ),
}
#: The column tuple of each table's owning module, read when the table is read.
STAGE_OWNERS: dict[str, Callable[[], tuple[str, ...]]] = {
    "mean_life_comparison": lambda: lifetime.COLUMNS,
    "record_classes": lambda: mixture.CLASS_COLUMNS,
    "selector_summary": lambda: d2.SUMMARY_COLUMNS,
    "shell_projection": lambda: d3c.PROJECTION_COLUMNS,
    "incompatible_groups": lambda: d3c.GROUPS_COLUMNS,
}
#: The columns that identify a row of each table; no two rows may share them.
STAGE_KEYS = {
    "mean_life_comparison": ("profile", "source", "locator"),
    "record_classes": ("profile", "z", "a"),
    "selector_summary": ("scope",),
    "shell_projection": ("source", "Z", "A", "transition", "quantity", "locator"),
    "incompatible_groups": ("source", "Z", "A", "initial_n", "final_n"),
}
#: The integer, ratio and flag cells the section reads.
STAGE_INTEGERS = {
    "record_classes": ("z", "a"),
    "selector_summary": (
        "rows",
        "to_oxygen",
        "hydrogenous",
        "selector_above",
        "z_ratio_closer",
        "z_ratio_above",
        "selector_above_z_ratio",
    ),
    "shell_projection": ("Z", "A"),
    "incompatible_groups": ("Z", "A", "initial_n", "final_n"),
}
STAGE_RATIOS = {"selector_summary": ("ratio_min", "ratio_median", "ratio_max")}
STAGE_FLAGS = {
    "record_classes": ("isotope_resolved", "needs_verification"),
    "shell_projection": ("solver_within", "consumer_within", "stock_within", "consumer_closer_than_stock"),
    "incompatible_groups": ("empty",),
}
#: The counts of the method-gated row of the selector summary, each a number of the rows it describes.
SELECTOR_COUNTS = ("to_oxygen", "hydrogenous", "selector_above")
#: The sources of the measured lines, as `DATASET_D3.md` cites them.
D3_SOURCES = {"Fricke1995": "Fricke et al. (1995)", "Saito2025": "Saito et al. (2025)"}
FLAG_VALUES = ("true", "false")
INTEGER = re.compile(r"0|[1-9][0-9]*")
RATIO = re.compile(r"[0-9]+[.][0-9]+")


class StageError(ValueError):
    """A committed file of a dataset that the section on the other stages of a stopped muon does not admit."""


@dataclass(frozen=True)
class Stages:
    """Exactly the values the section on the other stages of a stopped muon states, read from STAGE_FILES."""

    #: the method-gated row of the selector summary
    selector_rows: int
    selector_to_oxygen: int
    selector_hydrogenous: int
    selector_above: int
    ratio_min: str
    ratio_median: str
    ratio_max: str
    #: the D3 profile, the sources of the measured lines, and the counts of the projection and the groups
    d3_profile: str
    d3_sources: tuple[str, ...]
    gated: int
    stock_within: int
    consumer_within: int
    solver_within: int
    consumer_closer: int
    groups: int
    groups_empty: int
    #: the record of the numerical check
    status: str
    failed_rows: int
    check_rows: int
    check_cases: int
    #: the records of the compiled-in capture table (the `parity` profile)
    records: int
    settled: int
    unsettled: int
    resolved: int
    unresolved: int
    classes: tuple[tuple[str, int], ...]
    mixed: tuple[tuple[int, int], ...]
    #: lifetime.summary of the mean-life comparison
    summary: tuple[tuple[str, int, int, int], ...]


def stage_table(root: Path, name: str) -> list[dict[str, str]]:
    """The rows of the table ``name`` of STAGE_FILES, read raw: its owning module's columns must be the ones the
    section was written for; the bytes strict UTF-8 with no CR; the header exactly those columns and every record
    exactly that wide (raw_rows); at least a row; no two rows with the same key; and every integer, ratio and flag
    cell the section reads in its vocabulary."""
    relpath, columns = STAGE_FILES[name], STAGE_COLUMNS[name]
    if tuple(STAGE_OWNERS[name]()) != columns:
        raise StageError(f"{relpath}: its module's columns are not the columns this section was written for")
    data = (root / relpath).read_bytes()
    if b"\r" in data:
        raise StageError(f"{relpath}: the file contains a CR byte")
    rows, problems = raw_rows(relpath, data, columns)
    if problems:
        raise StageError("; ".join(problems))
    if not rows:
        raise StageError(f"{relpath}: the file has a header and no rows")
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row[column] for column in STAGE_KEYS[name])
        if key in seen:
            raise StageError(f"{relpath}: the key {dict(zip(STAGE_KEYS[name], key, strict=True))} repeats")
        seen.add(key)
        for column in STAGE_INTEGERS.get(name, ()):
            if not INTEGER.fullmatch(row[column]):
                raise StageError(f"{relpath}: {column} {row[column]!r} is not a non-negative integer")
        for column in STAGE_RATIOS.get(name, ()):
            if not RATIO.fullmatch(row[column]):
                raise StageError(f"{relpath}: {column} {row[column]!r} is not a non-negative decimal ratio")
        for column in STAGE_FLAGS.get(name, ()):
            if row[column] not in FLAG_VALUES:
                raise StageError(f"{relpath}: {column} {row[column]!r} is not true or false")
    return rows


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise StageError(f"{STAGE_FILES['numerical_check_status']}: the key {key!r} repeats")
        out[key] = value
    return out


def _json_constant(name: str) -> Any:
    raise StageError(f"{STAGE_FILES['numerical_check_status']}: {name} is not a number")


def stage_status(root: Path) -> tuple[str, int, int, int]:
    """(status, failed rows, rows, cases) of the numerical check's status record: no repeated key and no NaN or
    Infinity; the status in d3c.NUMERICAL_CHECK_STATUSES and with a sentence of STATUS_TEXT; every count an int
    (not a bool); a row and a case at least, no more cases than rows, no more failed rows than rows, as many
    failed rows as recorded failures, and a failure at least for a status that reports failed rows."""
    relpath = STAGE_FILES["numerical_check_status"]
    try:
        record = json.loads(
            (root / relpath).read_bytes().decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_json_constant,
        )
        status = record["status"]
        fresh = record["fresh_solve"]
        failed, failures = fresh["failed_rows"], fresh["failures"]
        rows, cases = fresh["population"]["rows"], fresh["population"]["cases"]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageError(f"{relpath}: not a JSON record ({error})") from None
    except (KeyError, TypeError) as error:
        raise StageError(f"{relpath}: a field the section reads is missing ({error!r})") from None
    if status not in d3c.NUMERICAL_CHECK_STATUSES:
        raise StageError(f"{relpath}: the status {status!r} is not a status of the D3 contract")
    if set(STATUS_TEXT) != set(d3c.NUMERICAL_CHECK_STATUSES):
        raise StageError(
            f"{relpath}: the statuses of the D3 contract and the sentences of STATUS_TEXT differ"
        )
    counts = {"failed_rows": failed, "population.rows": rows, "population.cases": cases}
    for field, value in counts.items():
        if type(value) is not int:
            raise StageError(f"{relpath}: {field} {value!r} is not an integer")
    if not isinstance(failures, list):
        raise StageError(f"{relpath}: failures is not a list")
    if rows < 1 or not 1 <= cases <= rows:
        raise StageError(
            f"{relpath}: the population of {rows} rows and {cases} cases is empty or inconsistent"
        )
    if not 0 <= failed <= rows or failed != len(failures):
        raise StageError(
            f"{relpath}: failed_rows {failed} is not the {len(failures)} failures within {rows} rows"
        )
    if status in FAILING_STATUSES and not failed:
        raise StageError(f"{relpath}: the status {status} records no failed row")
    return status, failed, rows, cases


def stage_inputs(root: Path) -> Stages:
    """The values of the section on the other stages of a stopped muon, from the files of STAGE_FILES under
    ``root`` and no other file; StageError on any input the section does not admit."""
    # the capturing atom
    relpath = STAGE_FILES["selector_summary"]
    gated_rows = [row for row in stage_table(root, "selector_summary") if row["scope"] == "method_gated"]
    if not gated_rows:
        raise StageError(f"{relpath}: no row method_gated")
    [selector] = gated_rows
    selector_rows = int(selector["rows"])
    if selector_rows < 1:
        raise StageError(f"{relpath}: the row method_gated describes no row")
    for column in SELECTOR_COUNTS:
        if int(selector[column]) > selector_rows:
            raise StageError(f"{relpath}: {column} exceeds the rows of method_gated")
    # the X-ray energies
    relpath = STAGE_FILES["shell_projection"]
    projection = stage_table(root, "shell_projection")
    for row in projection:
        if row["source"] not in D3_SOURCES:
            raise StageError(f"{relpath}: the source {row['source']!r} has no citation")
    groups = stage_table(root, "incompatible_groups")
    status, failed, check_rows, check_cases = stage_status(root)
    # the capture record
    relpath = STAGE_FILES["record_classes"]
    classes = stage_table(root, "record_classes")
    vocabulary = (*mixture.CLASSES, mixture.EVALUATED_NATURAL)
    for row in classes:
        if row["answer_class"] not in vocabulary:
            raise StageError(f"{relpath}: the class {row['answer_class']!r} is not a class of the D1 module")
    parity = [row for row in classes if row["profile"] == g4spec.PARITY_PROFILE]
    if not parity:
        raise StageError(f"{relpath}: no record of the profile {g4spec.PARITY_PROFILE}")
    for row in parity:
        if row["answer_class"] not in mixture.CLASS_TEXT:
            raise StageError(f"{relpath}: the class {row['answer_class']!r} has no definition to print")
    settled = [row for row in parity if row["needs_verification"] == "false"]
    tally = {name: sum(row["answer_class"] == name for row in parity) for name in mixture.CLASS_TEXT}
    # the disappearance rate
    relpath = STAGE_FILES["mean_life_comparison"]
    lives = stage_table(root, "mean_life_comparison")
    for row in lives:
        if row["population"] not in lifetime.POPULATIONS:
            raise StageError(f"{relpath}: the population {row['population']!r} is not one of the D1 module's")
        for suffix, _ in lifetime.ROUTES:
            cell = row[f"within_{suffix}"]
            if cell not in FLAG_VALUES and (cell or row["population"] == lifetime.COMPARED):
                raise StageError(f"{relpath}: within_{suffix} {cell!r} on a {row['population']} row")
    summary = tuple(lifetime.summary(lives))
    for profile, compared, *_ in summary:
        if not compared:
            raise StageError(f"{relpath}: the profile {profile!r} has no compared row")
    return Stages(
        selector_rows=selector_rows,
        selector_to_oxygen=int(selector["to_oxygen"]),
        selector_hydrogenous=int(selector["hydrogenous"]),
        selector_above=int(selector["selector_above"]),
        ratio_min=selector["ratio_min"],
        ratio_median=selector["ratio_median"],
        ratio_max=selector["ratio_max"],
        d3_profile=d3c.md.PROFILE,
        d3_sources=tuple(D3_SOURCES[name] for name in dict.fromkeys(row["source"] for row in projection)),
        gated=len(projection),
        stock_within=sum(row["stock_within"] == "true" for row in projection),
        consumer_within=sum(row["consumer_within"] == "true" for row in projection),
        solver_within=sum(row["solver_within"] == "true" for row in projection),
        consumer_closer=sum(row["consumer_closer_than_stock"] == "true" for row in projection),
        groups=len(groups),
        groups_empty=sum(row["empty"] == "true" for row in groups),
        status=status,
        failed_rows=failed,
        check_rows=check_rows,
        check_cases=check_cases,
        records=len(parity),
        settled=len(settled),
        unsettled=len(parity) - len(settled),
        resolved=sum(row["isotope_resolved"] == "true" for row in settled),
        unresolved=sum(row["isotope_resolved"] == "false" for row in settled),
        classes=tuple((name, count) for name, count in tally.items() if count),
        mixed=tuple((int(row["z"]), int(row["a"])) for row in parity if row["answer_class"] == mixture.MIXED),
        summary=summary,
    )


def _status_reproduction_failed(s: Stages) -> list[str | Slot]:
    return [
        "The check that sets the values of the production runs beside those an independent radial Dirac solver "
        "computed for the same equations has the status `",
        S(s.status),
        "` (`data/g4/d3/numerical_check_status.json`): a fresh solve of all ",
        S(s.check_rows),
        " rows, over ",
        S(s.check_cases),
        " nuclides, changed the independent solver's estimate of its own error by more than the registered "
        "reproduction tolerance on ",
        S(s.failed_rows),
        " rows, so its error estimates and the check's verdicts are historical until a corrected computation "
        "replaces them; no table value comes from the independent solver.",
    ]


#: The sentence of each status of the numerical check, keyed as d3c.NUMERICAL_CHECK_STATUSES; a status with no
#: sentence is refused.
STATUS_TEXT: dict[str, Callable[[Stages], list[str | Slot]]] = {
    "REPRODUCTION_FAILED": _status_reproduction_failed,
}
#: The statuses whose sentence reports failed rows: each needs a failure at least.
FAILING_STATUSES = ("REPRODUCTION_FAILED",)


def route_clauses(summary: tuple[tuple[str, int, int, int], ...]) -> list[list[str | Slot]]:
    """For each set of routes that leave a compared row outside the band, the profiles under which exactly those
    routes do, grouped in the comparison's order, as a clause: each route, a single route, or no route."""
    names = [name for _, name in lifetime.ROUTES]
    patterns: dict[tuple[str, ...], list[str]] = {}
    for profile, compared, *within in summary:
        outside = tuple(name for name, count in zip(names, within, strict=True) if count < compared)
        patterns.setdefault(outside, []).append(profile)
    clauses: list[list[str | Slot]] = []
    for outside, profiles in patterns.items():
        under: list[str | Slot] = ["under ", *joined([S(f"`{p}`") for p in profiles])]
        if len(outside) == len(names):
            clauses.append([*under, " each route leaves a compared row outside the band"])
        elif outside:
            verb = " leaves a compared row" if len(outside) == 1 else " leave compared rows"
            clauses.append(
                [*under, " only ", *joined([f"`{n}`" for n in outside]), verb, " outside the band"]
            )
        else:
            clauses.append([*under, " no route leaves a compared row outside the band"])
    return clauses


def render_stages(w: Writer, s: Stages) -> None:
    """The section on the other stages of a stopped muon: the capturing atom, the X-ray energies, the capture
    record and the disappearance rate, each restated from its dataset's committed files with a pointer to the
    dataset's document, in the order a stopped muon meets them."""
    w.line("## Other stages of a stopped muon, from the dataset documents")
    w.blank()
    w.line(
        "A stopped negative muon is captured by an atom, cascades down to the atom's ground state emitting muonic "
        "X-rays, and disappears from it by nuclear capture or by decay. The sections above compare the capture "
        "products and the cascade's intensities with measurement; this section restates what the dataset "
        "documents state about the other stages, from the committed files their own checks read, and judges "
        "nothing of its own."
    )
    w.blank()
    w.line("### The capturing atom")
    w.blank()
    w.line(
        "In a material of several elements, `G4ElementSelector::SelectZandA` picks the element whose nucleus "
        "takes the muon, and it contains no step that moves the muon between elements (`DATASET_D2.md`). "
        "`DATASET_D2.md` compares the per-atom capture ratio the selector's weights give with measured per-atom "
        "ratios under a method-aware rule, adopted after the selector's ratio had already been compared with the "
        "rows it gates (`data/g4/d2/selector_summary.csv`). Of the ",
        S(s.selector_rows),
        " rows the rule gates, ",
        S(s.selector_to_oxygen),
        " gives" if s.selector_to_oxygen == 1 else " give",
        " the ratio of an element to oxygen; the selector's ratio lies above the measured ratio on ",
        S(s.selector_above),
        " of them, and over all of them it lies at between ",
        S(s.ratio_min),
        " and ",
        S(s.ratio_max),
        " times the measured ratio, and at ",
        S(s.ratio_median),
        " times it at the median.",
    )
    if s.selector_hydrogenous:
        w.blank()
        w.line(
            "Of those rows, ",
            S(s.selector_hydrogenous),
            " is" if s.selector_hydrogenous == 1 else " are",
            " of a material that holds hydrogen, in which a muon captured by hydrogen can pass to a heavier "
            "nucleus nearby; for such a row the difference may come from such transfer rather than from the "
            "weights, and this comparison does not tell them apart.",
        )
    w.blank()
    w.line("### The X-ray energies")
    w.blank()
    w.line(
        "The cascade emits the line between a pair of shells at a single energy, the difference of their shell "
        "energies, even where a source prints more than a single line for the pair (`DATASET_D3.md`). "
        "`DATASET_D3.md` holds that energy "
        "against the band of each measured line on the ",
        S(s.gated),
        " gated rows of the lines ",
        *joined([S(source) for source in s.d3_sources]),
        " measured (`data/g4/d3/shell_projection.csv`). The energy the unpatched cascade emits lies within the "
        "band on ",
        S(s.stock_within),
        " of them, and the shell difference a patched cascade emits from the energy tables of the `",
        S(s.d3_profile),
        "` profile on ",
        S(s.consumer_within),
        "; the patched shell difference is closer to the measured energy than the unpatched cascade's on ",
        S(s.consumer_closer),
        ". The line MuDirac prints for the orbit-resolved transition, which no cascade emits, lies within the "
        "band on ",
        S(s.solver_within),
        ".",
    )
    w.blank()
    if s.groups_empty:
        w.line(
            "Where a source measured more than a single distinct line between the same shells of a nuclide, the "
            "lines' bands have an empty intersection in ",
            S(s.groups_empty),
            " of the ",
            S(s.groups),
            " such groups (`data/g4/d3/incompatible_groups.csv`), and in each of those no single shell energy can "
            "lie within the band of every line it stands for.",
        )
    else:
        w.line(
            "Where a source measured more than a single distinct line between the same shells of a nuclide, the "
            "lines' bands intersect in each of the ",
            S(s.groups),
            " such groups (`data/g4/d3/incompatible_groups.csv`).",
        )
    w.blank()
    w.line(*STATUS_TEXT[s.status](s))
    w.blank()
    w.line("### The capture record")
    w.blank()
    w.line(
        "The nucleus the selector draws gets its capture rate from `GetMuonCaptureRate(Z, A)`. The compiled-in "
        "table answers only a nucleus whose Z and A both match a record, and the Goulard–Primakoff formula "
        "answers every other, so a target of natural composition reaches an element's record only on the stops "
        "that draw the record's own mass number (`DATASET_D1.md`)."
    )
    w.blank()
    w.line(
        "`DATASET_D1.md` checks each of the ",
        S(s.records),
        " records of the compiled-in table, the records of its `",
        S(g4spec.PARITY_PROFILE),
        "` profile, against the primary its value is attributed to (`data/g4/d1/record_classes.csv`): "
        "`needs_verification` is `false`, settled against the primary, on ",
        S(s.settled),
        " of them and `true`, still open, on ",
        S(s.unsettled),
        "; of the settled records, `isotope_resolved` is `true` on ",
        S(s.resolved),
        " and `false`, as an established finding, on ",
        S(s.unresolved),
        ". Each record falls in a class of what answers a stop:",
    )
    w.blank()
    w.line("| class | definition | records |")
    w.line("|---|---|---|")
    for name, count in s.classes:
        w.line("| `", S(name), "` | ", S(mixture.CLASS_TEXT[name]), " | ", S(count), " |")
    if s.mixed:
        w.blank()
        w.line(
            "The value of each `",
            S(mixture.MIXED),
            "` record, keyed (Z, A) = ",
            *joined([S(f"({z}, {a})") for z, a in s.mixed]),
            ", is such a combination: a reconstructed combination, not a demonstrated history.",
        )
    w.blank()
    w.line("### The disappearance rate")
    w.blank()
    w.line(
        "A bound muon disappears from the ground state of the muonic atom by nuclear capture or by decay, and "
        "Geant4 computes the sum of those rates on the routes of `G4MuonMinusBoundDecay` and "
        "`G4MuonicAtomHelper`, each with its own decay rate: `G4MuonMinusBoundDecay` draws the time of a stop in "
        "`G4MuonMinusCapture`, the process the reference physics lists QBBC and FTFP_BERT give the negative muon "
        "at rest, and `G4MuonicAtomHelper` gives a muonic atom the lifetime `G4MuonicAtomDecay` reads "
        "(`DATASET_D1.md`). `DATASET_D1.md` "
        "compares each route's rate, under each profile of the capture table, with the mean lives the sources "
        "printed for their own measurements, and counts a rate within the band it defines "
        "(`data/g4/d1/mean_life_comparison.csv`):",
    )
    w.blank()
    header = lifetime.SUMMARY_HEADER.splitlines()
    for line in header:
        w.line(line)
    for profile, compared, first_route, second_route in s.summary:
        w.line("| `", S(profile), "` | ", S(compared), " | ", S(first_route), " | ", S(second_route), " |")
    w.blank()
    clauses = route_clauses(s.summary)
    parts: list[str | Slot] = []
    for i, clause in enumerate(clauses):
        if i:
            parts.append("; ")
        parts.extend(clause)
    first_part = parts[0]
    assert isinstance(first_part, str) and first_part == "under "
    w.line("Under ", *parts[1:], ".")
    w.blank()
    w.line(
        "The comparison is of the rates the code returns for a single nucleus, not of a transported simulation of "
        "a target; `DATASET_D1.md` compares the transported mixture of each natural target with its measured mean "
        "life."
    )
    w.blank()


def stages_text(s: Stages) -> Rendered:
    """The section on the other stages of a stopped muon alone, as render writes it."""
    w = Writer()
    render_stages(w, s)
    return w.rendered()


def render(inp: Inputs, tables: Tables | None = None) -> Rendered:
    """The document. Its cascade section is rendered from ``tables``: compare passes the tables it has proved
    equal to the written files; otherwise the canonical tables of the inputs."""
    pol = inp.policy
    setup, rule = pol["setup"], pol["rule"]
    v1, v2 = setup["versions"]
    configs = setup["configs"]
    verdict = {v.row_id: v for v in inp.verdicts}
    pairs, differ = revision_pairs(inp)
    identical = not differ
    shown = [r for r in inp.rows if r["version"] == v1] if identical else list(inp.rows)
    w = Writer()

    # ---- title and scope
    w.line("# Stopped negative muons in Geant4: a benchmark against measurement")
    w.blank()
    w.line(
        "This document compares the secondaries that unpatched Geant4 ",
        S(v1),
        " and ",
        S(v2),
        " create when the nucleus of a single-element target captures a stopped negative muon with yields "
        "measured for the same elements, and the muonic X-ray cascade that comes before the capture with measured "
        "K-series intensities. It is generated by `scripts/generate_stopped_mu.py` from the files of "
        "`data/g4/stopped_mu/`, the natural compositions of `data/g4/d1/geant4_natural_isotopes.csv` and the "
        "vendored `G4EmCaptureCascade.cc` of `third_party/geant4/`; the generator defines display precision and "
        "source citations. Its section on capture products and its section on the muonic X-ray cascade hold the "
        "comparisons; the application that made the outputs is in `cpp/stopped_muon/`. A further section, on "
        "the other stages of a stopped muon, restates from the committed files of the datasets what "
        "`DATASET_D1.md`, `DATASET_D2.md` and `DATASET_D3.md` state about the atom that captures the muon, the "
        "energies of its X-rays, the capture-table record that answers a stop and the rate at which the bound "
        "muon disappears.",
    )
    w.blank()

    # ---- what is run
    geo = setup["geometry"]
    w.line("## What is run")
    w.blank()
    w.line(
        "Each event starts with ",
        S(setup["primary"]),
        ", a ",
        S(geo["shape"]),
        " of ",
        S(geo["half_side_m"]),
        " m half-side filled with a single element at a density of ",
        S(geo["density_g_per_cm3"]),
        " g cm⁻³, and Geant4 transports it with the ",
        S(setup["physics_list"]),
        " physics list on ",
        S(setup["threads"]),
        " thread" if setup["threads"] == 1 else " threads",
        ". The event with global number i is seeded with ",
        S(setup["seed_base"]),
        " + i. Each cell holds the earliest ",
        S(setup["captures_per_configuration"]),
        " nuclear captures of a target, configuration and revision in event order, and every event of a cell "
        "must be a single nuclear capture or a single decay in orbit, apart from the events without an outcome "
        "that the rule's post-data addition treats (below).",
    )
    w.blank()
    conf = setup["configurations"]
    pc_env = conf["bertini_pc"]["environment"]
    [(pc_var, pc_val)] = pc_env.items()
    w.line("The configurations are reached as follows.")
    w.blank()
    w.line(
        "- `bertini`: the `",
        S(conf["bertini"]["route"]),
        "` route, what an unmodified QBBC application "
        "receives: `G4MuonMinusCapture` hands the capture to the Bertini cascade (`G4CascadeInterface`).",
    )
    w.line(
        "- `bertini_pc`: the same route with `",
        S(f"{pc_var}={pc_val}"),
        "`, which switches on the precompound de-excitation of the Bertini cascade.",
    )
    w.line(
        "- `precompound`: the `",
        S(conf["precompound"]["route"]),
        "` route, `G4MuonicAtomDecay` handing the "
        "capture to `G4MuMinusCapturePrecompound`, which an application receives only by registering constructors: "
        "QBBC's stopping constructor replaced by a `G4StoppingPhysics` whose last argument switches the capture at "
        "rest of a negative muon off in that constructor, a constructor that adds `G4MuonMinusAtomicCapture` as "
        "the rest process of the negative muon, and `G4MuonicAtomDecayPhysics`. Its rows describe what an "
        "application receives after those registrations, not what QBBC gives by default.",
    )
    w.blank()
    opt = setup["harness_options"]
    if opt[opt.index("--opt-in") + 1] != "off":
        raise ValueError("a cell runs with the dataset opt-in on")
    w.line(
        "This opt-in by constructor registration is not the dataset opt-in. The harness option `--opt-in` switches "
        "`G4HadronicParameters::SetEnableMuonicData`, which exists only in a build that carries the patches of "
        "`cpp/patches/`; an unpatched build refuses `--opt-in on`, every cell runs with `--opt-in off`, and no row "
        "scores patched behaviour."
    )
    w.blank()
    targets = setup["targets"]
    natural = [t["target"] for t in targets if t["basis"] == "natural"]
    single = [t["target"] for t in targets if t["basis"] == "single_isotope"]
    scored = []
    for t in targets:
        if any(p["target"] == t["target"] for p in pol["predicates"]):
            scored.append(t)
    unscored = [t["target"] for t in targets if t not in scored]
    w.line(
        "The cells run ",
        S(len(targets)),
        " targets: ",
        S(len(natural)),
        " of natural composition, each the element `G4NistManager::FindOrBuildElement` builds (",
        *joined(natural),
        "), and ",
        S(len(single)),
        " single isotopes built from their Z and A (",
        *joined(single),
        "). The scored rows use ",
        S(len(scored)),
        " of them, as the table gives; the other ",
        S(len(unscored)),
        " (",
        *joined(unscored),
        ") appear only in the cell records.",
    )
    w.blank()
    w.line("| predicates | target | basis |")
    w.line("|---|---|---|")
    for t in scored:
        preds = [p["predicate"] for p in pol["predicates"] if p["target"] == t["target"]]
        basis = "natural composition" if t["basis"] == "natural" else "single isotope"
        w.line("| ", *joined(preds, "and"), " | ", S(t["target"]), " | ", basis, " |")
    w.blank()
    gamma_targets = sorted({p["target"] for p in pol["predicates"] if p["family"] == "gamma"})
    for t in scored:
        if t["basis"] == "single_isotope":
            z, a = t["arguments"][1], t["arguments"][3]
            isotopes = [r for r in inp.natural if r["z"] == z]
            if len(isotopes) == 1 and isotopes[0]["a"] == a:
                w.line(
                    "The single isotope ",
                    S(t["target"]),
                    " is also the element of natural composition: the natural element Geant4 builds for Z = ",
                    S(z),
                    " has the single isotope A = ",
                    S(isotopes[0]["a"]),
                    " (`data/g4/d1/geant4_natural_isotopes.csv`).",
                )
                w.blank()
    ds = setup["datasets"]
    w.line("Each revision reads its datasets from the directories the pre-registration binds by name:")
    w.blank()
    w.line("| variable | ", S(v1), " | ", S(v2), " |")
    w.line("|---|---|---|")
    for var in ds["variables"]:
        w.line("| `", S(var), "` | `", S(ds["binding"][v1][var]), "` | `", S(ds["binding"][v2][var]), "` |")
    w.blank()

    # ---- the rule and when it was fixed
    eb = rule["efficiency_bracket"]
    rescue_families = {r["family"] for r in pol["rows"] if r["rescue"]}
    if rescue_families != {"nmean"}:
        raise ValueError(f"the rescue rows are not the neutron-mean rows: {sorted(rescue_families)}")
    w.line("## The rule and when it was fixed")
    w.blank()
    w.line(
        "Each row compares the model's x, with its sampling standard error x_se, with a measured reference ref "
        "of statistical and systematic uncertainty ref_stat and ref_syst, through σ_tot, the square root of the "
        "sum of the squares of ref_stat, ref_syst and x_se. Where a source supplies an unseparated error, "
        "ref_stat stores it as a bookkeeping component; this does not assert a source-provided "
        "statistical/systematic split. A row is PASS when |x − ref| ≤ ",
        S(rule["k_pass"]),
        " σ_tot. It is FAIL when |x − ref| > ",
        S(rule["k_fail"]),
        " σ_tot and, for a neutron-mean row, x also lies outside its rescue bracket "
        "[L, U]. Every other row is UNQUALIFIED: in tension when |x − ref| ≤ ",
        S(rule["k_fail"]),
        " σ_tot, "
        "efficiency-limited when it lies beyond that but inside the bracket. Each band includes its edge, and "
        "every decision is made on exact rationals parsed from the decimal strings.",
    )
    w.blank()
    w.line(
        "The rescue bracket [L, U] of a neutron-mean row is a conservative envelope of the union of its ",
        S(rule["k_fail"]),
        " σ bands over r = ε₀/ε, with ε₀ = ",
        S(eb["eps0"]),
        " and a detection efficiency ε anywhere in [",
        S(eb["eps_lo"]),
        ", ",
        S(eb["eps_hi"]),
        "], a design range fixed with the rule: the endpoint ratios are enclosed by r_lo = ",
        S(eb["r_lo"]),
        " (rounded down) and r_hi = ",
        S(eb["r_hi"]),
        ", L is the lower edge of the band about r_lo ref, with ref_stat and "
        "ref_syst scaled by r_lo, and U the upper edge of the band about r_hi ref, scaled by r_hi, L rounded down "
        "and U up at ",
        S(rule["rescue_quantum"]),
        "; `rule.rescue_edges` of the pre-registration states them in "
        "full. A missing row, an extra row, a reference cell that differs from the pre-registration or a rescue edge "
        "that differs from its recomputation is refused, never judged.",
    )
    w.blank()
    w.line(
        "The pre-registration was fixed in parts, each before the data it governs unless it is marked as fixed "
        "after data:"
    )
    w.blank()
    w.line("| part | fixed (UTC) | relation to the data | contents |")
    w.line("|---|---|---|---|")
    for part in pol["fixed"]:
        bold = "**" if part["post_data"] else ""
        w.line(
            "| ",
            S(part["part"]),
            " | ",
            S(part["fixed_utc"]),
            " | ",
            bold,
            S(part["relation_to_data"]),
            bold,
            " | ",
            S(part["contents"]),
            " |",
        )
    w.blank()
    for part in pol["fixed"]:
        if part.get("erratum"):
            w.line("The addition fixed after data carries this correction: ", S(part["erratum"]), ".")
            w.blank()
    w.line("What had been seen before a part was fixed:")
    w.blank()
    for seen in pol["seen_before_fixing"]:
        w.line(
            "- ",
            S(seen["part"]),
            ": seen before it was fixed, ",
            S(seen["seen"]),
            "; ",
            S(seen["what_was_chosen_without_it"]),
            ".",
        )
    w.blank()

    # ---- capture products
    w.line("## Capture products")
    w.blank()
    all_counts = counts(inp.verdicts)
    w.line(
        "**The verdicts.** Of the ",
        S(len(inp.verdicts)),
        " rows, ",
        S(all_counts["FAIL"]),
        " are FAIL, ",
        S(all_counts["PASS"]),
        " PASS and ",
        S(all_counts["UNQUALIFIED"]),
        " UNQUALIFIED; the FAIL rows are "
        "documented failures of the model against measurement and the PASS rows documented agreement, both at "
        "the stated configuration and revision.",
    )
    w.blank()
    for v in (v1, v2):
        c = counts([x for x in inp.verdicts if x.row_id.split("|")[1] == v])
        w.line(
            "- Geant4 ",
            S(v),
            ": ",
            S(c["FAIL"]),
            " FAIL, ",
            S(c["PASS"]),
            " PASS, ",
            S(c["UNQUALIFIED"]),
            " UNQUALIFIED.",
        )
    w.blank()
    if identical:
        w.line(
            "Every row of Geant4 ",
            S(v2),
            " has the x, x_se, reference cells, rescue edges and verdict of the row of Geant4 ",
            S(v1),
            " with the same predicate and configuration, so the tables below list the ",
            S(len(shown)),
            " rows of Geant4 ",
            S(v1),
            " once.",
        )
    else:
        w.line(
            "These ",
            S(len(differ)),
            " row pairs differ between the revisions, so the tables below list every row:",
        )
        w.blank()
        for rid in differ:
            r, o = next(p for p in pairs if p[0]["row_id"] == rid)
            w.line(
                "- `",
                S(r["predicate"]),
                "`, `",
                S(r["config"]),
                "`: x ",
                S(r["x"]),
                " (",
                S(verdict[rid].verdict),
                ") on Geant4 ",
                S(v1),
                " and ",
                S(o["x"]),
                " (",
                S(verdict[o["row_id"]].verdict),
                ") on Geant4 ",
                S(v2),
                ".",
            )
    w.blank()
    shown_v = [verdict[r["row_id"]] for r in shown]
    families = []
    for p in pol["predicates"]:
        if p["family"] not in families:
            families.append(p["family"])
    w.line(
        "**By family and configuration**",
        " (Geant4 " if identical else "",
        S(v1) if identical else "",
        ")" if identical else "",
        ":",
    )
    w.blank()
    w.line("| family | configuration | FAIL | PASS | UNQUALIFIED |")
    w.line("|---|---|---|---|---|")
    for fam in families:
        for cfg in configs:
            c = counts(
                [v for v, r in zip(shown_v, shown, strict=True) if r["family"] == fam and r["config"] == cfg]
            )
            w.line(
                "| ",
                S(fam),
                " | ",
                S(cfg),
                " | ",
                S(c["FAIL"]),
                " | ",
                S(c["PASS"]),
                " | ",
                S(c["UNQUALIFIED"]),
                " |",
            )
    w.blank()
    w.line("**By predicate**, over the configurations:")
    w.blank()
    w.line("| predicate | target | FAIL | PASS | UNQUALIFIED |")
    w.line("|---|---|---|---|---|")
    for p in pol["predicates"]:
        c = counts([v for v, r in zip(shown_v, shown, strict=True) if r["predicate"] == p["predicate"]])
        w.line(
            "| `",
            S(p["predicate"]),
            "` | ",
            S(p["target"]),
            " | ",
            S(c["FAIL"]),
            " | ",
            S(c["PASS"]),
            " | ",
            S(c["UNQUALIFIED"]),
            " |",
        )
    w.blank()
    w.line(
        "**The rows.** x_se and σ_tot are shown to ",
        S(SE_DIGITS),
        " significant digits and the rescue bracket rounded outward to ",
        S(BRACKET_DECIMALS),
        " decimals; `data/g4/stopped_mu/comparison.csv` gives every "
        "value in full and `verdicts.csv` the verdict and pull = (x − ref)/σ_tot of every row.",
    )
    w.blank()
    head = (
        "| predicate | target | "
        + ("" if identical else "revision | ")
        + "configuration | x | x_se | ref | ref_stat | ref_syst | σ_tot | pull | rescue [L, U] | verdict |"
    )
    w.line(head)
    w.line("|" + "---|" * (head.count("|") - 1))
    for r in shown:
        v = verdict[r["row_id"]]
        bracket: list[str | Slot] = ["–"]
        if r["rescue_lo"]:
            lo, hi = outward(r["rescue_lo"], r["rescue_hi"])
            bracket = ["[", S(lo), ", ", S(hi), "]"]
        rev: list[str | Slot] = [] if identical else [S(r["version"]), " | "]
        w.line(
            "| `",
            S(r["predicate"]),
            "` | ",
            S(r["target"]),
            " | ",
            *rev,
            S(r["config"]),
            " | ",
            S(r["x"]),
            " | ",
            S(sig(r["x_se"], SE_DIGITS)),
            " | ",
            S(r["ref"]),
            " | ",
            S(r["ref_stat"]),
            " | ",
            S(r["ref_syst"]),
            " | ",
            S(sigma_tot(r)),
            " | ",
            S(v.pull),
            " | ",
            *bracket,
            " | ",
            S(v.verdict),
            " |",
        )
    w.blank()
    unq = [v for v in inp.verdicts if v.verdict == "UNQUALIFIED"]
    tension = [v for v in unq if v.kind == "tension"]
    limited = [v for v in unq if v.kind == "efficiency_limited"]
    w.line(
        "**UNQUALIFIED rows.** Of the ",
        S(len(unq)),
        " UNQUALIFIED rows, ",
        S(len(tension)),
        " are in tension: they lie more than ",
        S(rule["k_pass"]),
        " and at most ",
        S(rule["k_fail"]),
        " σ_tot from their reference, so the rule neither passes nor fails them. The other ",
        S(len(limited)),
        " are efficiency-limited: they lie beyond ",
        S(rule["k_fail"]),
        " σ_tot of the mean at the published "
        "efficiency but inside the conservative rescue envelope, so the rule does not exclude them over "
        "its design range of detection efficiencies.",
    )
    w.blank()
    gp = sm.gamma_params(pol)
    z, a = gp.residual
    anchors = {x["anchor_id"]: x for x in pol["anchors"]}
    w.line(
        "**The γ lines of ",
        S(a),
        "Mg.** Each γ row counts, per nuclear capture of ",
        *joined(gamma_targets),
        ", the capture-step photons whose energy in the rest frame of a ",
        S(a),
        "Mg residual lies within the tag "
        "window of the transition, minus the sidebands, and compares the net count with the yield per capture "
        "the reference gives; every such cell passed the per-line descendant guard and the tag-width gate the "
        "pre-registration sets:",
    )
    w.blank()
    for r in shown:
        if r["family"] != "gamma":
            continue
        e0 = anchors[
            pol["predicates"][[p["predicate"] for p in pol["predicates"]].index(r["predicate"])][
                "anchor_ids"
            ][0]
        ]["window_lo"]
        rev = [] if identical else [", Geant4 ", S(r["version"])]
        x, se, k = Fraction(r["x"]), Fraction(r["x_se"]), Fraction(rule["k_pass"])
        if abs(x) <= k * se:
            w.line(
                "- the ",
                S(e0),
                " keV line, `",
                S(r["config"]),
                "`",
                *rev,
                ": x = ",
                S(r["x"]),
                " with x_se = ",
                S(r["x_se"]),
                ", so |x| ≤ ",
                S(rule["k_pass"]),
                " x_se: the net line yield consistent with ",
                S(NULL_YIELD),
                ", against the reference ",
                S(r["ref"]),
                ".",
            )
        else:
            w.line(
                "- the ",
                S(e0),
                " keV line, `",
                S(r["config"]),
                "`",
                *rev,
                ": x = ",
                S(r["x"]),
                " ± ",
                S(r["x_se"]),
                " against the reference ",
                S(r["ref"]),
                ".",
            )
    w.blank()
    nmean_rows = [r for r in shown if r["family"] == "nmean"]
    fails = sum(1 for r in nmean_rows if verdict[r["row_id"]].verdict == "FAIL")
    w.line(
        "**Neutrons.** A neutron-mean row is FAIL only when its x lies beyond ",
        S(rule["k_fail"]),
        " σ_tot of the mean at the published efficiency and outside the whole rescue bracket; ",
        S(fails),
        " of the ",
        S(len(nmean_rows)),
        " neutron-mean rows of the table are. The f2 rows compare the second factorial moment "
        "of the multiplicity divided by the square of its mean, with the same statistic of the measured "
        "distributions. Independent binomial detection of each neutron at a constant probability is "
        "sufficient to leave f2 unchanged and is the working assumption here. The "
        "table gives each row's x, reference and bracket; no ratio of x to the reference is formed here.",
    )
    w.blank()
    no_out = [c for _, c in sorted(inp.cells.items()) if c["n_no_outcome"]]
    post = next(p for p in pol["fixed"] if p["post_data"])
    alt = alternative_verdicts(inp)
    changed = [x for x in alt if x[1] != x[2]]
    w.line(
        "**Events without an outcome.** These cells hold events whose capture step returned neither "
        "nuclear-capture products nor a decay:"
    )
    w.blank()
    for c in no_out:
        w.line(
            "- ",
            S(c["version"]),
            " `",
            S(c["config"]),
            "` ",
            S(c["target"]),
            ": ",
            S(c["n_no_outcome"]),
            " of ",
            S(c["stops_total"]),
            " stops (event ",
            *joined(c["no_outcome_events"]),
            ").",
        )
    if not no_out:
        w.line("- none.")
    w.blank()
    relation = post["relation_to_data"]
    lead = "fixed after data: "
    if relation.startswith(lead):
        relation = relation[len(lead) :]
    w.line(
        "The rule excludes such an event from both denominators and refuses a cell in which more than ",
        S(rule["no_outcome"]["rate_limit"]),
        " of the stops have no outcome. That treatment is the pre-registration's ",
        S(post["part"]),
        " (",
        S(post["fixed_utc"]),
        "), **fixed after data**: ",
        S(relation),
        ". Counted instead "
        "as a capture with no products, the alternative treatment, each such event enters the earliest captures "
        "and moves the last capture out; the rule re-run on the alternative values of the ",
        S(len(alt)),
        " rows of these cells changes ",
        S(len(changed)),
        " of their verdicts",
        ":" if changed else ".",
    )
    for rid, old, new in changed:
        w.line("- `", S(rid), "`: ", S(old), " under the rule, ", S(new), " under the alternative.")
    w.blank()
    same, diff = body_pairs(inp)
    if diff:
        w.line(
            "**Chunk bodies.** Of the ",
            S(same + diff),
            " pairs of chunks the revisions ran with the same seeds, ",
            S(same),
            " have identical compact bodies (the compact text without its header line) and ",
            S(diff),
            " differ",
            "; every row value is nevertheless identical between the revisions."
            if identical
            else "; the rows that differ are listed above.",
        )
    else:
        w.line(
            "**Chunk bodies.** All ",
            S(same),
            " pairs of chunks the revisions ran with the same seeds have "
            "identical compact bodies (the compact text without its header line).",
        )
    w.blank()
    w.line(
        "**Shared seeds.** Every cell seeds the event with global number i with ",
        S(setup["seed_base"]),
        " + i, "
        "so the cells of a target, across configurations and revisions, are not independent samples: they are "
        "never pooled, and rows that agree between the revisions are not replications.",
    )
    w.blank()
    w.line("**Prior work and what these comparisons add.** ", PRIOR_WORK)
    w.blank()
    w.line(
        "Beside that work, the rows here compare, on Geant4 ",
        S(v1),
        " and ",
        S(v2),
        " and in the configurations ",
        *joined([f"`{c}`" for c in configs]),
        ", the final AlCap yields with their uncertainties for protons, "
        "deuterons, tritons and alphas from aluminium, silicon and titanium; the TWIST integrals on every "
        "configuration; the γ lines of ",
        S(a),
        "Mg per nuclear capture against Measday et al.; the mean neutron "
        "number per capture against the same measured means, judged within the efficiency bracket; and the f2 "
        "statistic against MacDonald et al., under a rule fixed before the runs apart from the addition the "
        "section above marks as fixed after data.",
    )
    w.blank()
    nmax = max(
        int(aid.rsplit(".", 1)[1])
        for p in pol["predicates"]
        if p["predicate"] in ("f2.Al", "f2.Si")
        for aid in p["anchor_ids"]
    )
    w.line(
        "**Limits.** QBBC only; the capture step only (reactions of the secondaries and later decays are not "
        "counted); a single element of the stated density in a cube of the stated half-side, natural targets with "
        "the composition `G4NistManager` builds; the neutron means are MacDonald et al.'s as Singer tabulates them, "
        "read in Blackwell's thesis; the γ-line references are the yields Measday et al. give per capture, "
        "normalised through their capture probability, which the ENSDF evaluation repeats; the f2 references for "
        "aluminium and silicon come from an evaluated distribution that assumes no capture emits more than ",
        S(nmax),
        " neutrons, while the model's f2 keeps every multiplicity; the f2 references for lead and silver "
        "are those of the detected distributions at efficiency ",
        S(eb["eps0"]),
        "; the working assumption of independent binomial detection at a constant probability is sufficient "
        "for equality with the true f2. The predicate annotations use this sufficient condition as a "
        "working assumption; the efficiency bracket "
        "is a fixed design range, not an uncertainty a source states; this benchmark does not establish the "
        "mechanism behind any disagreement.",
    )
    w.blank()

    render_cascade(w, inp, canonical_tables(inp) if tables is None else tables)
    render_stages(w, inp.stages)

    # ---- reproducing
    w.line("## Reproducing")
    w.blank()
    w.line(
        "`cpp/stopped_muon/README.md` describes the application and its driver. A cell is rebuilt and re-run "
        "against an unpatched Geant4 install and its datasets with:"
    )
    w.blank()
    w.line("```sh")
    w.line("python3 cpp/stopped_muon/stopped_muon.py build --install <prefix> --build <dir>")
    w.line(
        "python3 cpp/stopped_muon/stopped_muon.py reproduce --cell <version>/<config>/<target> --binary <exe> "
        "--install <prefix> --data <dir> --work <dir> --out <file>"
    )
    w.line(
        "python3 cpp/stopped_muon/stopped_muon.py cascade --binary <exe> --install <prefix> --data <dir> "
        "--work <dir> --version <version> --config <config> --target <target> --out <file> --transitions <file>"
    )
    w.line("```")
    w.blank()
    w.line(
        "The `cascade` command runs a cell of the harvest of `data/g4/stopped_mu/cascade_rule.json` with the "
        "environment and dataset checks of a cell's run, rebuilds every cascade from the records, deletes them "
        "and appends the cell's row to the file `--out` names and its transition counts to the file "
        "`--transitions` names; "
        "`data/g4/stopped_mu/cascade_harvest.csv` records each cell's binary sha256, seed base and the sha256 of "
        "its rebuilt cascades, and each cell's binary is a binary of its revision in the table below."
    )
    w.blank()
    tags = setup["tags"]
    builds = {b["tag"]: b for b in inp.builds}
    rep_cells, bad = reproduction(inp)
    w.line(
        "The cells ",
        *joined([" ".join(k) for k in rep_cells]),
        " were re-run chunk by chunk with the "
        "application rebuilt from the committed sources (`data/g4/stopped_mu/reproduced.csv`):",
        "" if bad else " each of their ",
        "" if bad else S(len(inp.reproduced)),
        ""
        if bad
        else " chunks gave compact text "
        "with the sha256 the committed cell records, so the re-run cells reduce to the committed records.",
    )
    for r, kind in bad:
        chunk: list[str | Slot] = [
            "- ",
            S(r["version"]),
            " ",
            S(r["config"]),
            " ",
            S(r["target"]),
            " ",
            S(r["chunk"]),
        ]
        if kind == "compact":
            w.line(
                *chunk,
                " gave compact text with sha256 ",
                S(r["compact_text_sha256"]),
                ", which the committed cell does not record.",
            )
        else:
            w.line(
                *chunk,
                " was run with binary sha256 ",
                S(r["binary_sha256"]),
                ", which differs from the recorded rebuilt binary sha256 ",
                S(builds[tags[r["version"]]]["rebuilt_binary_sha256"]),
                ".",
            )
    w.blank()
    w.line("The builds (`data/g4/stopped_mu/builds.csv`):")
    w.blank()
    w.line("| revision | geant4-config | compiler | CMake | binary of the committed cells | rebuild |")
    w.line("|---|---|---|---|---|---|")
    for b in inp.builds:
        same_binary = b["reference_binary_sha256"] == b["rebuilt_binary_sha256"]
        w.line(
            "| ",
            S(b["tag"]),
            " | ",
            S(b["geant4_config_version"]),
            " | ",
            S(b["compiler"]),
            " | ",
            S(b["cmake"]),
            " | `",
            S(b["reference_binary_sha256"]),
            "` | ",
            "the same bytes" if same_binary else "`",
            "" if same_binary else S(b["rebuilt_binary_sha256"]),
            "" if same_binary else "`",
            " |",
        )
    w.blank()
    differing = [b["tag"] for b in inp.builds if b["reference_binary_sha256"] != b["rebuilt_binary_sha256"]]
    if differing:
        w.line(
            "The rebuild of ",
            *joined(differing),
            " differs from the binary that made the committed cells; its re-run chunks are compared above.",
        )
    else:
        w.line(
            "On both revisions the rebuild of the committed sources is byte-identical to the binary that made the "
            "committed cells."
        )
    w.blank()

    # ---- files and licensing
    w.line("## Files")
    w.blank()
    w.line(
        "- `data/g4/stopped_mu/preregistration.json`: the setup, the measured anchors, the predicates, the rows "
        "and the rule, with when each part was fixed."
    )
    w.line(
        "- `data/g4/stopped_mu/cells.jsonl`: the reduced record of every cell, with the sha256 of each chunk's "
        "compact text and body."
    )
    w.line("- `data/g4/stopped_mu/builds.csv` and `reproduced.csv`: the builds and the re-run chunks.")
    w.line(
        "- `data/g4/stopped_mu/comparison.csv` and `verdicts.csv`: the rows and their verdicts, which "
        "`scripts/generate_stopped_mu.py compare` writes with this document as a step of `make audit`."
    )
    w.line(
        "- `data/g4/stopped_mu/cascade_reference.csv`: the measured K-series intensities, a row per printed "
        "value, transcribed twice; `cascade_rule.json`: the comparison rule and the harvest's test, with when "
        "each part was fixed."
    )
    w.line(
        "- `data/g4/stopped_mu/cascade_harvest.csv` and `cascade_transitions.csv`: the harvested cascades, a "
        "row per cell and a row per cell and transition, written by the `cascade` command."
    )
    w.line(
        "- `data/g4/stopped_mu/cascade_comparison.csv` and `cascade_consumer.csv`: the comparison with "
        "measurement and the decision on every retained count, which `scripts/generate_stopped_mu.py compare` "
        "writes with this document. Each certified quantity (a probability, a budget, a bound, a model value, a "
        "measured ratio, a variance) is an exact rational in a `_num` and a `_den` column, and every decision "
        "is recomputed from those columns alone; the decimal columns are for display."
    )
    w.line(
        "- `cpp/stopped_muon/`: the application and its driver; `openmucf/g4/stopped_mu.py`: the classifier, the "
        "reducer and the rule; `openmucf/g4/capture_cascade.py`: the exact cascade, its binding to the source, "
        "the reconstruction and the tests of the cascade."
    )
    w.line(
        "- `data/g4/d1/mean_life_comparison.csv` and `record_classes.csv`, `data/g4/d2/selector_summary.csv`, and "
        "`data/g4/d3/shell_projection.csv`, `incompatible_groups.csv` and `numerical_check_status.json`: the files "
        "of the datasets the section on the other stages of a stopped muon restates, which `make audit` "
        "regenerates or checks with their own documents."
    )
    w.blank()
    w.line("## Licensing")
    w.blank()
    w.line(
        "The files of this benchmark carry the licences the License section of the README gives them by path."
    )
    return w.rendered()


def forbidden_hits(text: str) -> list[str]:
    return [p for p in FORBIDDEN if re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE)]


def document(prereg: Path = DATA / "preregistration.json") -> Rendered:
    return render(load_inputs(prereg))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--prereg", default=str(DATA / "preregistration.json"))
    args = parser.parse_args()
    if args.cmd == "compare":
        compare(Path(args.prereg))


if __name__ == "__main__":
    main()
