"""openmucf.mucost -- the open muon-cost ledger loader (curated compilation with provenance).

Loads ``openmucf/data/muon_cost.csv`` (one row per published or derived muon-production energy
cost), validates each row against ``openmucf/data/muon_cost.schema.json``, and cross-checks that
every ``source_bibkey`` resolves in ``openmucf/data/references.bib``. Mirrors ``openmucf.rates``.

This is a **compilation with provenance, not an evaluation**: the single auditable basis is
``normalized_GeV_per_stopped_mu`` (beam energy per muon, in GeV; wall-plug = this / eta_acc, kept
separate); every OURS-normalization step is recorded verbatim in ``derivation``; T3 facility rows are
original derivations ("implied, derived here, formula shown") from public beam-power/muon-rate numbers,
since no facility reports GeV-per-stopped-muon; and an accounting credit (e.g. Kelly's x2.5 recapture)
is recorded in its own flagged column, never silently folded into the normalized value.

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
    normalized_GeV_per_stopped_mu: float  # NaN iff the digit is not pinned (needs_verification row)
    derivation: str
    source_bibkey: str
    source_locator: str
    needs_verification: bool
    notes: str

    @property
    def has_normalized(self) -> bool:
        """True iff a numeric normalized value is present (False for an unpinned nv row)."""
        return not math.isnan(self.normalized_GeV_per_stopped_mu)


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
        return [r.normalized_GeV_per_stopped_mu for r in rows if r.has_normalized]

    def tier_median(self, tier: str) -> float:
        """Median normalized GeV/muon for ``tier`` (over pinned rows). Proves the 10^3 gap (G-E2)."""
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
            norm = _to_float(row.get("normalized_GeV_per_stopped_mu", ""))
            has_norm = not math.isnan(norm)
            if has_norm and norm <= 0.0:
                errors.append(f"row {i} ({sid}): normalized_GeV_per_stopped_mu must be > 0 (got {norm})")
            if not has_norm and not nv:
                errors.append(
                    f"row {i} ({sid}): empty normalized_GeV_per_stopped_mu is allowed only when "
                    f"needs_verification=true"
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
                    normalized_GeV_per_stopped_mu=norm,
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
