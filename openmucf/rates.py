"""openmucf.rates -- the FAIR rate-ledger loader (single source of truth).

Loads ``openmucf/data/rates.csv`` (microscopic input constants) and
``openmucf/data/validation_targets.csv`` (observations to reproduce), validates each row
against ``openmucf/data/rates.schema.json``, and cross-checks that every ``source_bibkey``
resolves in ``openmucf/data/references.bib``. The data ships inside the package
(``[tool.setuptools.package-data]``) so non-editable installs work.

Design rules
------------
* Every value is traceable to a source key (provenance is enforced, not optional).
* The loader runs host-side; it hands the engine a plain nominal-value vector that
  downstream JAX code makes differentiable. No Python-side branching on traced values.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
DATA = _PKG / "data"
RATES_CSV = DATA / "rates.csv"
TARGETS_CSV = DATA / "validation_targets.csv"
SCHEMA_JSON = DATA / "rates.schema.json"
REFS_BIB = DATA / "references.bib"

VALID_STATUS = {"established", "contested"}
VALID_DISTRIBUTION = {"normal", "lognormal", "uniform", "asym_interval", ""}
VALID_RECOMMENDATION = {"recommended", "superseded", ""}
VALID_PHASE = {"gas", "liquid", "solid", "any", "intrinsic", ""}
VALID_TARGET = {"D2", "DT", "HD", "T2", "any", "n/a", ""}
# distributions that require explicit dist_lo/dist_hi bounds (an interval, not value+-unc):
DIST_NEEDS_BOUNDS = {"uniform", "asym_interval"}


@dataclass(frozen=True)
class Rate:
    symbol: str
    description: str
    value: float
    unit: str
    unc: float
    unc_type: str
    conditions: str
    source_bibkey: str
    source_locator: str
    status: str
    validity_range: str
    single_source: bool
    needs_verification: bool
    notes: str
    # WS-L typed condition/recommendation projections (backward-compatible defaults):
    distribution: str = ""
    dist_lo: float = float("nan")
    dist_hi: float = float("nan")
    recommendation: str = ""
    phase: str = ""
    target_molecule: str = ""


def _to_bool(s: str) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes"}


def _to_float(s: str) -> float:
    s = str(s).strip()
    return float(s) if s not in {"", "-", "nan"} else float("nan")


def bibkeys(bib_path: Path = REFS_BIB) -> set:
    """All citation keys defined in references.bib."""
    if not Path(bib_path).exists():
        return set()
    return set(re.findall(r"@\w+\{([^,]+),", Path(bib_path).read_text()))


class RatesTable:
    """Validated, dict-like collection of :class:`Rate` rows."""

    def __init__(self, rates: dict):
        self._rates = rates

    def __getitem__(self, sym: str) -> Rate:
        return self._rates[sym]

    def __contains__(self, sym: str) -> bool:
        return sym in self._rates

    def __len__(self) -> int:
        return len(self._rates)

    def symbols(self):
        return list(self._rates.keys())

    def value(self, sym: str) -> float:
        return self._rates[sym].value

    def get(self, sym: str):
        return self._rates.get(sym)

    def contested(self):
        return [r for r in self._rates.values() if r.status == "contested"]

    def needs_verification(self):
        return [r for r in self._rates.values() if r.needs_verification]

    def dist_bounds(self, sym: str) -> tuple[float, float]:
        """(dist_lo, dist_hi) for ``sym`` (NaNs if the row has no explicit interval)."""
        r = self._rates[sym]
        return (r.dist_lo, r.dist_hi)

    def nominal_vector(self, symbols: Sequence[str]):
        """jnp float64 vector of nominal values, for autodiff/UQ. Order = ``symbols``."""
        import jax.numpy as jnp

        return jnp.array([self._rates[s].value for s in symbols], dtype=jnp.float64)

    def uncertainty_vector(self, symbols: Sequence[str]):
        import jax.numpy as jnp

        return jnp.array([self._rates[s].unc for s in symbols], dtype=jnp.float64)


def load_rates(
    csv_path: Path = RATES_CSV,
    schema_path: Path = SCHEMA_JSON,
    check_refs: bool = True,
) -> RatesTable:
    """Load + validate the rate ledger. Raises ``ValueError`` listing every problem."""
    schema = json.loads(Path(schema_path).read_text()) if Path(schema_path).exists() else {}
    required = schema.get("required", [])
    known_keys = bibkeys() if check_refs else None

    rates: dict = {}
    errors: list = []
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            sym = (row.get("symbol") or "?").strip()
            for col in required:
                if not (row.get(col) or "").strip():
                    errors.append(f"row {i} ({sym}): missing required '{col}'")
            status = (row.get("status") or "").strip()
            if status and status not in VALID_STATUS:
                errors.append(f"row {i} ({sym}): bad status '{status}'")
            # WS-L typed columns: enum membership + interval-distribution bounds.
            dist = (row.get("distribution") or "").strip()
            if dist not in VALID_DISTRIBUTION:
                errors.append(f"row {i} ({sym}): bad distribution '{dist}'")
            if dist in DIST_NEEDS_BOUNDS:
                lo_s = (row.get("dist_lo") or "").strip()
                hi_s = (row.get("dist_hi") or "").strip()
                if not lo_s or not hi_s:
                    errors.append(f"row {i} ({sym}): distribution '{dist}' requires dist_lo and dist_hi")
                else:
                    try:
                        if float(lo_s) >= float(hi_s):
                            errors.append(f"row {i} ({sym}): dist_lo >= dist_hi ({lo_s} >= {hi_s})")
                    except ValueError:
                        errors.append(f"row {i} ({sym}): non-numeric dist bounds ('{lo_s}','{hi_s}')")
            rec = (row.get("recommendation") or "").strip()
            if rec not in VALID_RECOMMENDATION:
                errors.append(f"row {i} ({sym}): bad recommendation '{rec}'")
            phase = (row.get("phase") or "").strip()
            if phase not in VALID_PHASE:
                errors.append(f"row {i} ({sym}): bad phase '{phase}'")
            tgt = (row.get("target_molecule") or "").strip()
            if tgt not in VALID_TARGET:
                errors.append(f"row {i} ({sym}): bad target_molecule '{tgt}'")
            if check_refs and known_keys is not None:
                for key in re.split(r"[;,]", row.get("source_bibkey") or ""):
                    key = key.strip()
                    if key and key not in known_keys:
                        errors.append(f"row {i} ({sym}): source_bibkey '{key}' not in references.bib")
            try:
                r = Rate(
                    symbol=sym,
                    description=(row.get("description") or "").strip(),
                    value=_to_float(row.get("value", "")),
                    unit=(row.get("unit") or "").strip(),
                    unc=_to_float(row.get("unc", "")),
                    unc_type=(row.get("unc_type") or "").strip(),
                    conditions=(row.get("conditions") or "").strip(),
                    source_bibkey=(row.get("source_bibkey") or "").strip(),
                    source_locator=(row.get("source_locator") or "").strip(),
                    status=status,
                    validity_range=(row.get("validity_range") or "").strip(),
                    single_source=_to_bool(row.get("single_source", "")),
                    needs_verification=_to_bool(row.get("needs_verification", "")),
                    notes=(row.get("notes") or "").strip(),
                    distribution=dist,
                    dist_lo=_to_float(row.get("dist_lo", "")),
                    dist_hi=_to_float(row.get("dist_hi", "")),
                    recommendation=rec,
                    phase=phase,
                    target_molecule=tgt,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row {i} ({sym}): parse error {exc}")
                continue
            if r.symbol in rates:
                errors.append(f"duplicate symbol '{r.symbol}'")
            rates[r.symbol] = r

    # At most one 'recommended' per symbol prefix family (e.g. {omega_s0, omega_s0_legacy}).
    fam_recommended: dict = {}
    for r in rates.values():
        if r.recommendation == "recommended":
            fam = r.symbol.replace("_legacy", "")
            fam_recommended[fam] = fam_recommended.get(fam, 0) + 1
    for fam, count in fam_recommended.items():
        if count > 1:
            errors.append(f"family '{fam}': {count} rows marked 'recommended' (at most one allowed)")

    if errors:
        raise ValueError("rate ledger validation failed:\n  " + "\n  ".join(errors))
    return RatesTable(rates)


def omega_fraction(rate_or_value) -> float:
    """Convert a sticking value stored in percent to a bare fraction."""
    v = rate_or_value.value if isinstance(rate_or_value, Rate) else float(rate_or_value)
    return v / 100.0
