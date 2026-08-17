"""openmucf.mucost -- the open muon-cost ledger loader (curated compilation with provenance).

Loads ``openmucf/data/muon_cost.csv`` (one row per published or derived muon-production energy
cost), validates each row against ``openmucf/data/muon_cost.schema.json``, and cross-checks that
every ``source_bibkey`` resolves in ``openmucf/data/references.bib``. Mirrors ``openmucf.rates``.

This is a **compilation with provenance, not an evaluation**: ``normalized_GeV_per_mu`` is beam energy
per muon in GeV **on that row's own accounting basis** (wall-plug = this / eta_acc, kept separate);
every OURS-normalization step is recorded verbatim in ``derivation``; T3 facility rows are original
derivations ("implied, derived here, formula shown") from public beam-power/muon-rate numbers, since no
facility reports GeV-per-stopped-muon; and an accounting credit (e.g. Kelly's x2.5 recapture,
stated in his abstract) is
recorded in its own flagged column, never silently folded into the normalized value.

**Bases are heterogeneous and are NOT commensurable.** The column was previously named
``normalized_GeV_per_stopped_mu``, which wrongly implied a single per-stopped basis; the rows in fact
mix per-produced, per-collected, per-stopped-in-D-T and per-stopped-in-another-target figures, and one
row counts mu+ and mu- together. ``basis_class`` and ``charge_basis`` make that machine-readable, and
:meth:`MuonCostTable.is_basis_homogeneous` lets a caller check before aggregating. A per-produced or
per-collected figure is a LOWER BOUND on the per-stopped-in-D-T cost, because collection and stopping
fractions are both < 1.

**A cost is a point on a 2-D grid, not a scalar.** The two axes are :data:`MUCF_CHAIN` (``stage`` --
how far along produce -> capture -> transport -> moderate -> stop-useful-in-D-T the muon has got) and
``numeraire`` (the units the energy is counted in: beam kinetic, or electrical on either of two
facility denominators). **Wall-plug is a numeraire, not a stage:** dividing by an accelerator
efficiency changes the units and applies at *any* stage, so treating it as a sequential node would
make "electrical energy per transported muon" inexpressible. ``basis_class`` is the deprecated 1-D
predecessor of ``stage``, kept as an alias. Because the two axes are independent, every aggregation
here (:meth:`MuonCostTable.tier_median` and friends) is **restricted to a single numeraire**, defaulting
to :data:`BEAM_KINETIC` -- medianing beam-kinetic and electrical figures together would be a units error.

**Composition is basis-typed and refuses to lie.** :class:`ChainValue` carries a figure together with
its stage, numeraire and the evidence status of every factor composed into it. If any composed factor
is ``author_declared_arbitrary``/``assumption``/``absent``, or if the chain has not reached the terminal
stage, the result is a **BOUND, not a value** -- and :meth:`ChainValue.render_value` *raises* rather
than print it as one. Every omitted factor is <= 1, so the bound is one-sided (costs can only rise).
No row in the literature today has a fully-sourced chain, which is the finding, not a gap in the code.

Not part of the eager-import surface (like ``calibrate``/``validate``/``forecast``); reached as a
submodule. The rate ledger (``openmucf.rates``) remains the source of truth for microscopic physics;
this is the E_mu single accounting home (the ``E_mu_cost`` rate-ledger row points here).
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .rates import bibkeys

_PKG = Path(__file__).resolve().parent
DATA = _PKG / "data"
MUON_COST_CSV = DATA / "muon_cost.csv"
MUON_COST_SCHEMA = DATA / "muon_cost.schema.json"

VALID_TIER = {"T1-design-study", "T2-demonstrated-tech", "T3-operating-facility"}
TIER_ORDER = ("T1-design-study", "T2-demonstrated-tech", "T3-operating-facility")

# Accounting bases. Only ``stopped_in_dt`` is the quantity a muCF energy balance actually needs;
# ``produced``/``collected`` are LOWER BOUNDS on it, and ``stopped_other_target`` is stopped somewhere
# that is not D-T fuel. Values on different classes are not commensurable.
VALID_BASIS_CLASS = {"produced", "collected", "stopped_in_dt", "stopped_other_target", ""}
VALID_CHARGE_BASIS = {"mu_minus", "mixed", "mu_plus_only", ""}
# Classes whose figure understates the true per-stopped-in-D-T cost (collection/stopping fractions < 1).
LOWER_BOUND_CLASSES = frozenset({"produced", "collected"})

# ------------------------------------------------------------------------------------------------
# The two basis axes. `stage` says how far along the muon chain the cost is counted; `numeraire` says
# what kind of energy it is counted in. They are INDEPENDENT -- which is exactly why wall-plug is a
# numeraire and not a stage (see the module docstring).
# ------------------------------------------------------------------------------------------------
#: The muCF chain, in order. Only the last entry is the quantity a muCF energy balance actually needs.
MUCF_CHAIN: tuple[str, ...] = ("produced", "captured", "transported", "moderated", "stopped_useful_in_dt")
TERMINAL_STAGE = MUCF_CHAIN[-1]
#: Stopped somewhere that is not D-T fuel: a real cost, but NOT a point on the muCF chain at all.
OFF_CHAIN_STAGES = frozenset({"stopped_other_target"})
VALID_STAGE = set(MUCF_CHAIN) | OFF_CHAIN_STAGES | {""}

BEAM_KINETIC = "beam_kinetic"
VALID_NUMERAIRE = {BEAM_KINETIC, "electrical_minimal", "electrical_site", ""}

#: Statuses that carry real provenance. Anything else makes a composed figure a BOUND.
SOURCED_STATUSES = frozenset({"primary", "primary_cited", "derived_here"})
NON_SOURCED_STATUSES = frozenset({"author_declared_arbitrary", "assumption", "absent"})
VALID_EVIDENCE_STATUS = SOURCED_STATUSES | NON_SOURCED_STATUSES

#: The deprecated `basis_class` alias -> the `stage` it maps onto. Kept so the two cannot drift.
STAGE_FROM_BASIS_CLASS = {
    "produced": "produced",
    "collected": "transported",
    "stopped_in_dt": "stopped_useful_in_dt",
    "stopped_other_target": "stopped_other_target",
    "": "",
}


class BasisError(ValueError):
    """Raised when a composition or a rendering would misrepresent a figure's accounting basis."""


@dataclass(frozen=True)
class ChainValue:
    """A muon-cost figure that knows its own basis and whether it is a VALUE or only a BOUND.

    ``statuses`` accumulates the ``evidence_status`` of every factor composed in, so the verdict is
    derived from provenance rather than asserted. The figure is a **value** only when it has reached
    :data:`TERMINAL_STAGE` *and* every composed factor is sourced; otherwise it is a **bound**, and
    :meth:`render_value` refuses. Because every omitted or unsourced factor is <= 1, the bias is
    always one-sided: the true cost can only be higher.
    """

    value_GeV: float
    stage: str
    numeraire: str
    charge_basis: str
    statuses: tuple[str, ...]
    provenance: tuple[str, ...]

    @property
    def missing_stages(self) -> tuple[str, ...]:
        """Chain stages not yet reached (each contributes a factor <= 1 that would raise the cost)."""
        return MUCF_CHAIN[MUCF_CHAIN.index(self.stage) + 1 :]

    @property
    def unsourced_statuses(self) -> tuple[str, ...]:
        return tuple(s for s in self.statuses if s in NON_SOURCED_STATUSES)

    @property
    def is_bound(self) -> bool:
        return bool(self.unsourced_statuses) or bool(self.missing_stages)

    @property
    def bias_direction(self) -> str:
        """Which way the truth lies. Always ``'lower'`` for a bound -- never a symmetric interval."""
        return "lower" if self.is_bound else "none"

    def why_bound(self) -> str:
        """Human-readable reason, for printing next to the number. Empty iff this is a value."""
        why = []
        if self.missing_stages:
            why.append("stages not reached: " + ", ".join(self.missing_stages))
        if self.unsourced_statuses:
            why.append("non-sourced factors: " + ", ".join(self.unsourced_statuses))
        return "; ".join(why)

    def render(self, digits: int = 2) -> str:
        """The figure with an explicit bound marker when it is one. Always safe to print."""
        prefix = ">= " if self.is_bound else ""
        return f"{prefix}{self.value_GeV:.{digits}f} GeV"

    def render_value(self, digits: int = 2) -> str:
        """The figure as a plain value. **Raises** :class:`BasisError` if it is only a bound.

        This refusal is the point of the class: it makes the basis error unrepresentable rather than
        merely discouraged, so a caller cannot quote an incomplete or unsourced chain as a result.
        """
        if self.is_bound:
            raise BasisError(
                f"refusing to render a bound as a value ({self.value_GeV:.{digits}f} GeV at stage "
                f"'{self.stage}', numeraire '{self.numeraire}'): {self.why_bound()}. "
                f"Use render() -- the true cost is one-sided ({self.bias_direction})."
            )
        return f"{self.value_GeV:.{digits}f} GeV"

    def compose(self, factor: float, to_stage: str, status: str, label: str) -> ChainValue:
        """Advance along the chain by dividing by a sub-unity delivery ``factor``.

        The numeraire is unchanged (that is a separate conversion); the stage must strictly advance,
        and the factor's own ``status`` is carried forward, so composing an
        ``author_declared_arbitrary`` factor permanently marks the result as a bound.
        """
        if to_stage not in MUCF_CHAIN:
            raise BasisError(f"cannot compose to stage {to_stage!r}: not on the muCF chain {MUCF_CHAIN}")
        if MUCF_CHAIN.index(to_stage) <= MUCF_CHAIN.index(self.stage):
            raise BasisError(f"composition must advance the chain: {self.stage!r} -> {to_stage!r}")
        if status not in VALID_EVIDENCE_STATUS:
            raise BasisError(f"unknown evidence_status {status!r}")
        if not 0.0 < factor <= 1.0:
            raise BasisError(f"a delivery factor must lie in (0, 1]; got {factor!r}")
        return ChainValue(
            value_GeV=self.value_GeV / factor,
            stage=to_stage,
            numeraire=self.numeraire,
            charge_basis=self.charge_basis,
            statuses=self.statuses + (status,),
            provenance=self.provenance + (label,),
        )


@dataclass(frozen=True)
class MuonCost:
    source_id: str
    citation: str
    year: int
    tier: str
    basis_as_published: str
    projectile_target: str
    capture_scheme: str
    recapture_credit_applied: bool
    recapture_factor: float  # NaN if none quoted
    eta_acc_assumption: float  # NaN if the source states none
    eta_mu_assumption: float  # NaN if the source states none
    eta_mu_evidence_status: str  # "" iff no eta_mu is stated
    value_as_published: str
    unit_as_published: str
    normalized_GeV_per_mu: float  # NaN iff the digit is not pinned (needs_verification row)
    numeraire: str
    stage: str
    evidence_status: str
    useful_fraction_sourced: bool | None  # None where stage is not the terminal one
    basis_class: str
    charge_basis: str
    derivation: str
    source_bibkey: str
    source_locator: str
    needs_verification: bool
    notes: str

    @property
    def has_normalized(self) -> bool:
        """True iff a numeric normalized value is present (False for an unpinned nv row)."""
        return not math.isnan(self.normalized_GeV_per_mu)

    @property
    def understates_stopped_in_dt_cost(self) -> bool:
        """True iff this row's basis makes it a LOWER BOUND on cost per mu- stopped in D-T fuel.

        A per-produced or per-collected figure omits the collection and stopping fractions (both < 1),
        so the real per-stopped-in-D-T cost is higher. A ``mixed`` charge basis counts mu+ alongside
        mu-, which understates the mu--only cost the same way.
        """
        return self.basis_class in LOWER_BOUND_CLASSES or self.charge_basis == "mixed"

    @property
    def wallplug_lower_bound_GeV(self) -> float:
        """Wall-plug-equivalent GeV per muon = normalized / eta_acc, on THIS ROW'S basis.

        ``eta_acc`` is the electrical -> muon-beam (wall-plug) efficiency the source states, so this
        converts BEAM GeV to WALL-PLUG GeV; it is NOT a collection or stopping correction. Whether the
        result is a *lower bound* on the wall-plug cost per mu- stopped in D-T depends on the row's
        basis -- see :attr:`understates_stopped_in_dt_cost`. NaN if the source states no eta_acc.
        """
        if math.isnan(self.eta_acc_assumption) or not self.has_normalized:
            return float("nan")
        return self.normalized_GeV_per_mu / self.eta_acc_assumption

    @property
    def eta_mu_is_sourced(self) -> bool:
        """True iff this row states an eta_mu whose evidence status is a sourced one.

        Kelly, Hart & Rose state theirs is an "arbitrary but reasonable assumption" and that they do
        not know the real value, so theirs is False -- it may be displayed, never composed into a
        quotable result.
        """
        return self.eta_mu_evidence_status in SOURCED_STATUSES

    def chain_point(self) -> ChainValue:
        """This row as a typed point on the muCF chain. Raises for rows that are not on it.

        A ``stopped_other_target`` row is a real measurement but not a muCF cost at any stage, so it
        cannot become a :class:`ChainValue` at all. A terminal-stage row whose source never establishes
        the "useful" qualifier picks up an ``assumption`` status here, which keeps it a bound.
        """
        if not self.has_normalized:
            raise BasisError(f"{self.source_id}: no pinned value, so it has no chain point")
        if self.stage in OFF_CHAIN_STAGES:
            raise BasisError(
                f"{self.source_id}: stage {self.stage!r} is not on the muCF chain "
                f"(the muons are stopped outside D-T fuel) and can never enter a muCF cost"
            )
        if self.stage not in MUCF_CHAIN:
            raise BasisError(f"{self.source_id}: stage {self.stage!r} is not a chain stage")
        statuses: tuple[str, ...] = (self.evidence_status,)
        provenance: tuple[str, ...] = (self.source_id,)
        if self.stage == TERMINAL_STAGE and self.useful_fraction_sourced is not True:
            statuses += ("assumption",)
            provenance += (f"{self.source_id}:useful-qualifier-not-established",)
        return ChainValue(
            value_GeV=self.normalized_GeV_per_mu,
            stage=self.stage,
            numeraire=self.numeraire,
            charge_basis=self.charge_basis,
            statuses=statuses,
            provenance=provenance,
        )


def _to_bool(s: str) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes"}


def _to_float(s: str) -> float:
    s = str(s).strip()
    return float(s) if s not in {"", "-", "nan"} else float("nan")


def _to_int(s: str) -> int:
    return int(str(s).strip())


class MuonCostTable:
    """Validated, ordered collection of :class:`MuonCost` rows, keyed by ``source_id``."""

    def __init__(self, rows: list[MuonCost]):
        self._rows = list(rows)
        self._by_id = {r.source_id: r for r in rows}

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, source_id: str) -> MuonCost:
        return self._by_id[source_id]

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._by_id

    def ids(self):
        return [r.source_id for r in self._rows]

    def tier(self, t: str) -> list[MuonCost]:
        """All rows in tier ``t`` (in file order). Raises ``KeyError`` on an unknown tier."""
        if t not in VALID_TIER:
            raise KeyError(f"unknown tier {t!r}; expected one of {sorted(VALID_TIER)}")
        return [r for r in self._rows if r.tier == t]

    def needs_verification(self) -> list[MuonCost]:
        return [r for r in self._rows if r.needs_verification]

    def rows_in_numeraire(self, numeraire: str, tier: str | None = None) -> list[MuonCost]:
        """Pinned rows counted in ``numeraire`` (optionally within one tier)."""
        if numeraire not in VALID_NUMERAIRE:
            raise KeyError(f"unknown numeraire {numeraire!r}; expected {sorted(VALID_NUMERAIRE - {''})}")
        rows = self._rows if tier is None else self.tier(tier)
        return [r for r in rows if r.has_normalized and r.numeraire == numeraire]

    def normalized_values(self, tier: str | None = None, numeraire: str = BEAM_KINETIC) -> list[float]:
        """Pinned GeV-per-muon values in ONE numeraire (default beam-kinetic), optionally one tier.

        The numeraire restriction is not a convenience filter, it is a units guard: the ledger holds
        beam-kinetic *and* electrical figures, and mixing them in one aggregate would be a units error
        on top of the stage-basis error this table already documents.
        """
        return [r.normalized_GeV_per_mu for r in self.rows_in_numeraire(numeraire, tier)]

    def basis_classes(self, tier: str | None = None, numeraire: str = BEAM_KINETIC) -> set[str]:
        """The distinct ``basis_class`` values among pinned rows in one numeraire."""
        return {r.basis_class for r in self.rows_in_numeraire(numeraire, tier) if r.basis_class}

    def stages(self, tier: str | None = None, numeraire: str = BEAM_KINETIC) -> set[str]:
        """The distinct ``stage`` values among pinned rows in one numeraire."""
        return {r.stage for r in self.rows_in_numeraire(numeraire, tier) if r.stage}

    def numeraires(self) -> set[str]:
        """Every numeraire present among pinned rows."""
        return {r.numeraire for r in self._rows if r.has_normalized and r.numeraire}

    def is_basis_homogeneous(self, tier: str | None = None) -> bool:
        """True iff every pinned row shares one ``basis_class``, i.e. aggregating them is meaningful."""
        return len(self.basis_classes(tier)) <= 1

    def tier_median(self, tier: str, numeraire: str = BEAM_KINETIC) -> float:
        """Median GeV/muon for ``tier`` within ONE numeraire (default beam-kinetic, over pinned rows).

        WARNING: this medians whatever *stages* the tier happens to contain. No tier is currently
        stage-homogeneous, so a cross-tier ratio of these medians is NOT a same-basis comparison --
        check :meth:`is_basis_homogeneous` and disclose the composition before quoting one. The
        numeraire, by contrast, IS held fixed here, because medianing beam-kinetic against electrical
        figures would not even be dimensionally meaningful.
        (``statistics.median`` sorts internally, so the result is independent of row order.)
        """
        import statistics

        vals = self.normalized_values(tier, numeraire)
        if not vals:
            raise ValueError(f"tier {tier!r} has no pinned values in numeraire {numeraire!r}")
        return statistics.median(vals)


def load_muon_cost(
    csv_path: Path = MUON_COST_CSV,
    schema_path: Path = MUON_COST_SCHEMA,
    check_refs: bool = True,
) -> MuonCostTable:
    """Load + validate the muon-cost ledger. Raises ``ValueError`` listing every problem."""
    schema = json.loads(Path(schema_path).read_text()) if Path(schema_path).exists() else {}
    required = schema.get("required", [])
    known_keys = bibkeys() if check_refs else None

    rows: list[MuonCost] = []
    seen_ids: set[str] = set()
    errors: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            sid = (row.get("source_id") or "?").strip()
            for col in required:
                if not (row.get(col) or "").strip():
                    errors.append(f"row {i} ({sid}): missing required '{col}'")
            tier = (row.get("tier") or "").strip()
            if tier and tier not in VALID_TIER:
                errors.append(f"row {i} ({sid}): bad tier '{tier}' (expected {sorted(VALID_TIER)})")
            applied = _to_bool(row.get("recapture_credit_applied", ""))
            factor = _to_float(row.get("recapture_factor", ""))
            # Consistency: a credit cannot be APPLIED without a factor. (A factor may be RECORDED
            # without being applied -- Kelly's x2.5 is recorded, applied=false, never folded in.)
            if applied and math.isnan(factor):
                errors.append(f"row {i} ({sid}): recapture_credit_applied=true but no recapture_factor")
            nv = _to_bool(row.get("needs_verification", ""))
            norm = _to_float(row.get("normalized_GeV_per_mu", ""))
            has_norm = not math.isnan(norm)
            if has_norm and norm <= 0.0:
                errors.append(f"row {i} ({sid}): normalized_GeV_per_mu must be > 0 (got {norm})")
            if not has_norm and not nv:
                errors.append(
                    f"row {i} ({sid}): empty normalized_GeV_per_mu is allowed only when "
                    f"needs_verification=true"
                )
            basis_class = (row.get("basis_class") or "").strip()
            charge_basis = (row.get("charge_basis") or "").strip()
            numeraire = (row.get("numeraire") or "").strip()
            stage = (row.get("stage") or "").strip()
            evidence_status = (row.get("evidence_status") or "").strip()
            eta_mu = _to_float(row.get("eta_mu_assumption", ""))
            eta_mu_status = (row.get("eta_mu_evidence_status") or "").strip()
            useful_raw = (row.get("useful_fraction_sourced") or "").strip()
            useful = _to_bool(useful_raw) if useful_raw else None
            if numeraire not in VALID_NUMERAIRE:
                errors.append(
                    f"row {i} ({sid}): bad numeraire '{numeraire}' "
                    f"(expected {sorted(VALID_NUMERAIRE - {''})})"
                )
            if stage not in VALID_STAGE:
                errors.append(
                    f"row {i} ({sid}): bad stage '{stage}' (expected {sorted(VALID_STAGE - {''})})"
                )
            if evidence_status not in VALID_EVIDENCE_STATUS:
                errors.append(
                    f"row {i} ({sid}): bad evidence_status '{evidence_status}' "
                    f"(expected {sorted(VALID_EVIDENCE_STATUS)})"
                )
            if eta_mu_status and eta_mu_status not in VALID_EVIDENCE_STATUS:
                errors.append(f"row {i} ({sid}): bad eta_mu_evidence_status '{eta_mu_status}'")
            # An eta_mu digit without a status would be composable-looking but ungraded, and a status
            # without a digit grades nothing: the pair travels together or not at all.
            if math.isnan(eta_mu) != (eta_mu_status == ""):
                errors.append(
                    f"row {i} ({sid}): eta_mu_assumption and eta_mu_evidence_status must both be "
                    f"present or both empty (got {eta_mu!r} / '{eta_mu_status}')"
                )
            # The deprecated alias must keep agreeing with the axis that superseded it, or downstream
            # code reading either one would silently see two different accounting bases.
            expected_stage = STAGE_FROM_BASIS_CLASS.get(basis_class)
            if expected_stage is not None and stage != expected_stage:
                errors.append(
                    f"row {i} ({sid}): stage '{stage}' contradicts deprecated basis_class "
                    f"'{basis_class}' (which maps to '{expected_stage}')"
                )
            # A pinned row must say what units it is in and how far along the chain it sits.
            if has_norm and not (numeraire and stage):
                errors.append(
                    f"row {i} ({sid}): a pinned value requires both numeraire and stage "
                    f"(empty is allowed only on an unpinned needs_verification row)"
                )
            # 'useful' is only a question at the terminal stage; claiming it elsewhere is meaningless.
            if useful is not None and stage != TERMINAL_STAGE:
                errors.append(
                    f"row {i} ({sid}): useful_fraction_sourced is only meaningful at stage "
                    f"'{TERMINAL_STAGE}' (got stage '{stage}')"
                )
            if has_norm and stage == TERMINAL_STAGE and useful is None:
                errors.append(
                    f"row {i} ({sid}): a pinned '{TERMINAL_STAGE}' row must state "
                    f"useful_fraction_sourced (true/false), never leave it unsaid"
                )
            if basis_class not in VALID_BASIS_CLASS:
                errors.append(
                    f"row {i} ({sid}): bad basis_class '{basis_class}' "
                    f"(expected {sorted(VALID_BASIS_CLASS - {''})})"
                )
            if charge_basis not in VALID_CHARGE_BASIS:
                errors.append(
                    f"row {i} ({sid}): bad charge_basis '{charge_basis}' "
                    f"(expected {sorted(VALID_CHARGE_BASIS - {''})})"
                )
            # a pinned row must declare what its number actually counts, or it cannot be aggregated
            if has_norm and not (basis_class and charge_basis):
                errors.append(
                    f"row {i} ({sid}): a pinned normalized value requires both basis_class and "
                    f"charge_basis (empty is allowed only on an unpinned needs_verification row)"
                )
            if check_refs and known_keys is not None:
                for key in re.split(r"[;,]", row.get("source_bibkey") or ""):
                    key = key.strip()
                    if key and key not in known_keys:
                        errors.append(f"row {i} ({sid}): source_bibkey '{key}' not in references.bib")
            try:
                mc = MuonCost(
                    source_id=sid,
                    citation=(row.get("citation") or "").strip(),
                    year=_to_int(row.get("year", "0") or "0"),
                    tier=tier,
                    basis_as_published=(row.get("basis_as_published") or "").strip(),
                    projectile_target=(row.get("projectile_target") or "").strip(),
                    capture_scheme=(row.get("capture_scheme") or "").strip(),
                    recapture_credit_applied=applied,
                    recapture_factor=factor,
                    eta_acc_assumption=_to_float(row.get("eta_acc_assumption", "")),
                    eta_mu_assumption=eta_mu,
                    eta_mu_evidence_status=eta_mu_status,
                    value_as_published=(row.get("value_as_published") or "").strip(),
                    unit_as_published=(row.get("unit_as_published") or "").strip(),
                    normalized_GeV_per_mu=norm,
                    numeraire=numeraire,
                    stage=stage,
                    evidence_status=evidence_status,
                    useful_fraction_sourced=useful,
                    basis_class=basis_class,
                    charge_basis=charge_basis,
                    derivation=(row.get("derivation") or "").strip(),
                    source_bibkey=(row.get("source_bibkey") or "").strip(),
                    source_locator=(row.get("source_locator") or "").strip(),
                    needs_verification=nv,
                    notes=(row.get("notes") or "").strip(),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row {i} ({sid}): parse error {exc}")
                continue
            if mc.source_id in seen_ids:
                errors.append(f"duplicate source_id '{mc.source_id}'")
            seen_ids.add(mc.source_id)
            rows.append(mc)

    if errors:
        raise ValueError("muon-cost ledger validation failed:\n  " + "\n  ".join(errors))
    return MuonCostTable(rows)
