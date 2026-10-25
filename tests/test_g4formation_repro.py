"""The public formation checker has its observer sources and complete log-producing matrix."""

import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "cpp/clock/formation_validation"
prepare_spec = spec_from_file_location("formation_prepare", SUPPORT / "prepare.py")
prepare = module_from_spec(prepare_spec)
prepare_spec.loader.exec_module(prepare)
run_spec = spec_from_file_location("formation_run", SUPPORT / "run.py")
run = module_from_spec(run_spec)
with patch.dict("sys.modules", {"prepare": prepare}):
    run_spec.loader.exec_module(run)


def test_published_observer_bodies_match_the_tested_source_hashes():
    tested = json.loads((SUPPORT / "tested_sources.json").read_text())
    for name, digest in tested["files"].items():
        assert hashlib.sha256((SUPPORT / name).read_bytes()).hexdigest() == digest
    assert 'extern "C" int clock_audit_event()' in (SUPPORT / "formation_observer.cc").read_text()
    transport = (SUPPORT / "audit_transport.cc").read_text()
    assert "AUDIT_RNG" in transport and "getTheEngine()->put(state)" in transport
    assert 'extern "C" int clock_audit_event()' in transport


@pytest.mark.parametrize("revision", prepare.REVISIONS)
def test_trace_patch_adds_only_passive_observers(revision):
    patch_text = (SUPPORT / f"trace-g4-{revision}.patch").read_text()
    removed = [
        line for line in patch_text.splitlines() if line.startswith("-") and not line.startswith("---")
    ]
    assert not removed
    additions = "\n".join(
        line[1:] for line in patch_text.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    assert all(tag in additions for tag in ("AUDIT_FRAGMENT", "AUDIT_RP", "AUDIT_PICK"))
    assert "GetFormationTime()" in additions and "GetTOF()" in additions
    assert "G4UniformRand" not in additions and "SetTime(" not in additions and "fTime =" not in additions
    assert prepare.CORRECTED in patch_text


def test_comparator_statements_are_explicit_and_cannot_silently_miss_the_getter():
    text = "#include <algorithm>\n" + prepare.CORRECTED + "\n"
    variants = prepare.variants(text)
    assert variants["new"] == text
    assert variants["old"] == "fTime = time0 + rp->GetTOF();\n"
    assert variants["zero"] == "fTime = time0;\n"
    with pytest.raises(ValueError, match="exactly one"):
        prepare.variants(text + prepare.CORRECTED)


def test_patch_application_isolated_from_enclosing_or_caller_git_context(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_DIR", "wrong-repository/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "wrong-worktree")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", "wrong-ceiling")
    monkeypatch.setenv("D1D2_REPRO_MARKER", "preserved")
    with patch.object(prepare.subprocess, "run") as run_git:
        prepare.apply_patch_file(Path("trace.patch"), tmp_path / "kernel", tmp_path)
    assert run_git.call_count == 2
    for call in run_git.call_args_list:
        env = call.kwargs["env"]
        assert "GIT_DIR" not in env and "GIT_WORK_TREE" not in env
        assert env["GIT_CEILING_DIRECTORIES"] == str(tmp_path.resolve())
        assert env["D1D2_REPRO_MARKER"] == "preserved"


def test_public_run_plan_produces_every_checker_input_with_declared_counts():
    plan = run.plan_runs(Path("objects"), Path("logs"))
    assert len(plan) == 30 and len({path for _, path in plan}) == 30
    expected = {
        f"{v}-{mode}-{origin}.log"
        for v in ("old", "new", "zero")
        for mode in ("controlled", "default", "light")
        for origin in (0, 250, 1000)
    }
    expected |= {f"{v}-integrated-250.log" for v in ("old", "new", "zero")}
    assert {path.name for _, path in plan} == expected
    for command, path in plan:
        if "integrated" in path.name:
            assert command[1:] == ["atomic", "plain", "250", "120", "12345"]
        else:
            mode = command[1]
            assert command[3] == ("50" if mode == "default" else "4")
