"""The `openmucf` console entry point (openmucf.cli): subcommands + exit-code contract."""

import json

import pytest

from openmucf import bench, cli


def test_version_prints_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "openmucf 1.1.0" in capsys.readouterr().out


def test_no_command_prints_help_exits_zero(capsys):
    assert cli.main([]) == 0
    assert "reproduce" in capsys.readouterr().out


def test_reproduce_all_exits_zero(capsys):
    # No case FAILs, so `reproduce --all` exits 0 (PENDING/DEFERRED do not fail the run).
    assert cli.main(["reproduce", "--all"]) == 0
    out = capsys.readouterr().out
    assert "kou-chen-2026" in out and "jones-1986" in out


def test_reproduce_pass_case_exits_zero(capsys):
    assert cli.main(["reproduce", "kou-chen-2026"]) == 0
    assert "[PASS] kou-chen-2026" in capsys.readouterr().out


def test_reproduce_pending_case_exits_zero(capsys):
    # A blocked-acquisition (PENDING) case runs nothing and fails nothing -> exit 0.
    assert cli.main(["reproduce", "jones-1986"]) == 0
    assert "[PENDING] jones-1986" in capsys.readouterr().out


def test_reproduce_validation_id_exits_zero(capsys):
    assert cli.main(["reproduce", "V_kouchen_base"]) == 0
    assert "[PASS] V_kouchen_base" in capsys.readouterr().out


def test_validate_subcommand_exits_zero(capsys):
    assert cli.main(["validate"]) == 0
    assert "trust gate" in capsys.readouterr().out


def test_unknown_case_id_is_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["reproduce", "no-such-case"])
    assert exc.value.code == 2


def test_reproduce_fail_case_exits_one(tmp_path, monkeypatch, capsys):
    # A synthetic FAILing case must propagate exit code 1 through the CLI.
    case = {
        "id": "synthetic-fail",
        "type": "reproduction",
        "title": "synthetic",
        "engine": "analytic",
        "inputs": [
            {"label": "x", "omega_s_eff_pct": 0.45, "lambda_c": 1e8,
             "published_value": 1.0, "tolerance": "+-1%"}
        ],
        "provenance": {"source_bibkey": "none", "locator": "synthetic", "input_basis": "synthetic test"},
        "status": "active",
        "notes": "synthetic FAIL case",
    }
    (tmp_path / "synthetic-fail.json").write_text(json.dumps(case), encoding="utf-8")
    monkeypatch.setattr(bench, "CASES_DIR", tmp_path)
    assert cli.main(["reproduce", "synthetic-fail"]) == 1
    assert "[FAIL] synthetic-fail" in capsys.readouterr().out
