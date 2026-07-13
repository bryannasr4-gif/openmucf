"""Digitized Yamashita-Kino Fig. 3a lambda_c(T) table: schema, monotonicity, and the
determinism gate (re-running the committed digitizer reproduces the committed CSV byte-for-byte)."""

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "openmucf" / "data" / "yamashita_kino_lc_T.csv"
FIG = REPO / "openmucf" / "data" / "yamashita_kino_fig3a.png"
SCRIPT = REPO / "scripts" / "digitize_yamashita_fig3a.py"
EXPECTED_HEADER = ["T_K", "lambda_c_s^-1", "digitization_unc_rel", "source_bibkey", "source_locator"]


def _rows():
    with open(CSV, newline="") as f:
        return list(csv.DictReader(f))


def test_yamashita_csv_schema_and_load():
    """Header matches the digitizer contract; every row parses to a positive rate at a valid T."""
    with open(CSV, newline="") as f:
        header = next(csv.reader(f))
    assert header == EXPECTED_HEADER
    rows = _rows()
    assert len(rows) == 14
    for r in rows:
        assert float(r["lambda_c_s^-1"]) > 0.0
        assert 20 <= int(r["T_K"]) <= 800
        assert r["source_bibkey"] == "YamashitaKino2022"
        assert 0.0 < float(r["digitization_unc_rel"]) < 1.0


def test_yamashita_csv_monotonic_T():
    """Temperatures are strictly increasing and unique (a well-formed T-grid)."""
    ts = [int(r["T_K"]) for r in _rows()]
    assert ts == sorted(ts)
    assert len(set(ts)) == len(ts)


def test_digitizer_determinism(tmp_path):
    """Re-running the committed digitizer on the committed figure reproduces the committed CSV
    byte-for-byte (deterministic extraction; the anti-hallucination gate for the sourced comparator)."""
    out = tmp_path / "regen.csv"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(FIG), str(out)],
        check=True,
        capture_output=True,
    )
    assert out.read_bytes() == CSV.read_bytes(), "digitizer output drifted from the committed CSV"
