"""openmucf.mucost -- the open muon-cost ledger loader (curated compilation with provenance).

Loads ``openmucf/data/muon_cost.csv`` (one row per published or derived muon-production energy
cost), validates each row against ``openmucf/data/muon_cost.schema.json``, and cross-checks that
every ``source_bibkey`` resolves in ``openmucf/data/references.bib``. Mirrors ``openmucf.rates``.

This is a **compilation with provenance, not an evaluation**: ``normalized_GeV_per_mu`` is beam energy
per muon in GeV **on that row's own accounting basis** (wall-plug = this / eta_acc, kept separate);
every OURS-normalization step is recorded verbatim in ``derivation``; T3 facility rows are original
derivations ("implied, derived here, formula shown") from public beam-power/muon-rate numbers, since no
facility reports GeV-per-stopped-muon; and an accounting credit (e.g. Kelly's x2.5 recapture) is
recorded in its own flagged column, never silently folded into the normalized value.

**Bases are heterogeneous and are NOT commensurable.** The column was previously named
``normalized_GeV_per_stopped_mu``, which wrongly implied a single per-stopped basis; the rows in fact
mix per-produced, per-collected, per-stopped-in-D-T and per-stopped-in-another-target figures, and one
row counts mu+ and mu- together. ``basis_class`` and ``charge_basis`` make that machine-readable, and
:meth:`MuonCostTable.is_basis_homogeneous` lets a caller check before aggregating. A per-produced or
per-collected figure is a LOWER BOUND on the per-stopped-in-D-T cost, because collection and stopping
fractions are both < 1.

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
    value_as_published: str
    unit_as_published: str
    normalized_GeV_per_mu: float  # NaN iff the digit is not pinned (needs_verification row)
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

    def normalized_values(self, tier: str | None = None) -> list[float]:
        """Pinned normalized GeV-per-muon values (skips unpinned nv rows), optionally one tier."""
        rows = self._rows if tier is None else self.tier(tier)
        return [r.normalized_GeV_per_mu for r in rows if r.has_normalized]

    def basis_classes(self, tier: str | None = None) -> set[str]:
        """The distinct ``basis_class`` values among pinned rows (optionally within one tier)."""
        rows = self._rows if tier is None else self.tier(tier)
        return {r.basis_class for r in rows if r.has_normalized and r.basis_class}

    def is_basis_homogeneous(self, tier: str | None = None) -> bool:
        """True iff every pinned row shares one ``basis_class``, i.e. aggregating them is meaningful."""
        return len(self.basis_classes(tier)) <= 1

    def tier_median(self, tier: str) -> float:
        """Median normalized GeV/muon for ``tier`` (over pinned rows).

        WARNING: this medians whatever bases the tier happens to contain. No tier is currently
        basis-homogeneous, so a cross-tier ratio of these medians is NOT a same-basis comparison --
        check :meth:`is_basis_homogeneous` and disclose the composition before quoting one.
        (``statistics.median`` sorts internally, so the result is independent of row order.)
        """
        import statistics

        vals = self.normalized_values(tier)
        if not vals:
            raise ValueError(f"tier {tier!r} has no pinned normalized values")
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
                    value_as_published=(row.get("value_as_published") or "").strip(),
                    unit_as_published=(row.get("unit_as_published") or "").strip(),
                    normalized_GeV_per_mu=norm,
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
