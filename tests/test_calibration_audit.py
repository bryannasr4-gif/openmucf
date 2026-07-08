"""WS-A (Fable amendment, 2026-07-07): the CALIBRATION --audit table parser is generic (WS-N-ready),
and its tolerances are pinned against a *silent* softening (WAVE1 spec 1.5 never-soften rule).
"""

import importlib.util
from pathlib import Path

import openmucf

_SCRIPT = Path(openmucf.__file__).resolve().parent.parent / "scripts" / "generate_calibration.py"


def _load_script():
    """Import the script module by path (no MCMC runs: its work is guarded behind main())."""
    spec = importlib.util.spec_from_file_location("_gen_calibration", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_tables_generic_three_sections():
    """_parse_tables handles an arbitrary number of chains/tables -- locks WS-N's third-table readiness."""
    mod = _load_script()
    md = (
        "# header\n\n"
        "## Chain one\n| parameter | mean | sd | 95% CI |\n|---|---|---|---|\n"
        "| a | 1.0 | 0.1 | [0.8, 1.2] |\n| b | 2.0 | 0.2 | [1.6, 2.4] |\n\n"
        "## Chain two\n| parameter | mean | sd | 95% CI |\n|---|---|---|---|\n"
        "| c | 3.0 | 0.3 | [2.4, 3.6] |\n\n"
        "## Chain three (channels-on)\n| parameter | mean | sd | 95% CI |\n|---|---|---|---|\n"
        "| d | 4.0 | 0.4 | [3.2, 4.8] |\n"
    )
    sections = mod._parse_tables(md)
    assert [t for t, _ in sections] == ["Chain one", "Chain two", "Chain three (channels-on)"]
    assert sections[0][1]["a"] == (1.0, 0.1)
    assert sections[0][1]["b"] == (2.0, 0.2)
    assert sections[1][1]["c"] == (3.0, 0.3)
    assert sections[2][1]["d"] == (4.0, 0.4)


def test_audit_tolerances_pinned():
    """Any *silent* softening of the audit tolerances trips this test (same literal-substring guard as
    test_forecast.py::test_d6). The tolerances were amended 2026-07-07 by the planning model to
    2% mean / 8% sd; changing them requires deliberately editing this pin."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "AUDIT_RTOL_MEAN = 0.02" in src
    assert "AUDIT_RTOL_SD = 0.08" in src
