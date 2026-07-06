"""Regenerate (or audit) the FC-001 forecast card + FORECASTS.md registry.

    python scripts/generate_forecast.py            # full regenerate (runs the MCMC): card + FORECASTS.md
    python scripts/generate_forecast.py --audit    # no MCMC: hash-consistency check + render FORECASTS.md

Importable without side effects (all work is inside ``main()``). The card payload is MCMC-derived, so it is
NOT exact-diffed in CI (same precedent as CALIBRATION.md); FORECASTS.md is rendered deterministically from the
on-disk card and IS exact-diffed by ``make audit``. See forecasts/FORECAST_PROTOCOL.md.
"""

import json
import sys
from pathlib import Path

from openmucf import forecast

ROOT = Path(__file__).resolve().parent.parent
CARD = ROOT / "forecasts" / "FC-001-mufuse.json"
FORECASTS_MD = ROOT / "FORECASTS.md"


def regenerate() -> None:
    card = forecast.regenerate(CARD)
    forecast.validate_card(card)
    FORECASTS_MD.write_text(forecast.render_forecasts_md([CARD]), encoding="utf-8")
    reg = card["registration"]
    print(f"wrote {CARD.relative_to(ROOT)}  (payload_sha256={reg['payload_sha256'][:12]}..., "
          f"status={reg['status']})")
    print(f"wrote {FORECASTS_MD.relative_to(ROOT)}")


def audit() -> None:
    """Hash-consistency + structural check of the ON-DISK card, then render FORECASTS.md from it (no MCMC)."""
    card = json.loads(CARD.read_text(encoding="utf-8"))
    recomputed = forecast.payload_sha256(card)
    recorded = card["registration"]["payload_sha256"]
    if recomputed != recorded:
        raise SystemExit(f"FC-001 payload hash mismatch: recomputed {recomputed} != recorded {recorded}")
    forecast.validate_card(card)
    FORECASTS_MD.write_text(forecast.render_forecasts_md([CARD]), encoding="utf-8")
    print(f"forecast audit OK: FC-001 hash-consistent ({recorded[:12]}...); FORECASTS.md rendered from disk")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--audit" in argv:
        audit()
    else:
        regenerate()


if __name__ == "__main__":
    main()
