"""The CALIBRATION --audit parser is generic (header-aware, WS-N-ready), and its per-quantity tolerance
CLASSES are pinned against a *silent* softening (WAVE1 spec 1.5 / WAVE3 never-soften rule).
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


def test_parse_tables_generic_header_aware():
    """_parse_tables returns (title, header, rows) for an arbitrary number of tables/columns -- so a chain
    or column added later is picked up without editing the parser."""
    mod = _load_script()
    md = (
        "# header\n\n"
        "## Chain one\n| parameter | mean | sd | mcse | 95% CI |\n|---|---|---|---|---|\n"
        "| a | 1.0 | 0.1 | 0.01 | [0.8, 1.2] |\n| b | 2.0 | 0.2 | 0.02 | [1.6, 2.4] |\n\n"
        "## Convergence\n| chain | max r_hat | min ess | divergences |\n|---|---|---|---|\n"
        "| weak | 1.001 | 2500 | 0 |\n"
    )
    sections = mod._parse_tables(md)
    assert [t for t, _, _ in sections] == ["Chain one", "Convergence"]
    assert sections[0][1] == ["parameter", "mean", "sd", "mcse", "95% CI"]
    assert sections[0][2][0] == ["a", "1.0", "0.1", "0.01", "[0.8, 1.2]"]
    assert sections[1][1] == ["chain", "max r_hat", "min ess", "divergences"]


def test_cell_specs_class_map():
    """Column-name -> tolerance-class routing is the classifier the audit relies on."""
    mod = _load_script()
    assert mod._cell_specs("mean") == [("mean", "mean")]
    assert mod._cell_specs("R sd") == [("sd", "sd")]
    assert mod._cell_specs("mcse") == [("mcse", "ess_mcse")]
    assert mod._cell_specs("min ess") == [("ess", "ess_mcse")]
    assert mod._cell_specs("max r_hat") == [("r_hat", "rhat")]
    assert mod._cell_specs("corr") == [("corr", "corr")]
    assert mod._cell_specs("divergences") == [("divergences", "div")]
    assert mod._cell_specs("95% CI") == []       # intervals are not audited
    assert mod._cell_specs("boxes") == []
    assert mod._cell_specs("rails?") == []


def test_audit_tolerances_pinned():
    """Any *silent* softening of the audit tolerance CLASSES trips this test (same literal-substring guard
    as test_forecast.py::test_d6). Changing them requires deliberately editing this pin + a dated RESULTS
    note (never-soften rule)."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "AUDIT_RTOL_MEAN = 0.02" in src        # mean cells 2%
    assert "AUDIT_RTOL_SD = 0.08" in src          # sd cells 8%
    assert "AUDIT_RTOL_CORR = 0.08" in src        # corr cells 8%
    assert "AUDIT_RTOL_ESS_MCSE = 0.20" in src    # ess + mcse cells 20%
    assert "AUDIT_RTOL_RHAT = 0.02" in src        # r_hat cells 2%
    assert "AUDIT_ATOL_DIVERGENCES = 0" in src    # divergences EXACT == 0
