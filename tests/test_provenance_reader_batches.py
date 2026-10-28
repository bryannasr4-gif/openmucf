"""The reader differential splits its documents into command lines that Windows accepts."""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _script():
    script = ROOT / "cpp/test/check_provenance_reader.py"
    spec = importlib.util.spec_from_file_location("check_provenance_reader", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _line(args: list[str]) -> int:
    return sum(len(arg) + 3 for arg in args)


def test_batches_keep_every_path_in_order_within_the_budget():
    mod = _script()
    head = ["C:/build/Release/g4muonicdata_bound_rate_check.exe", "provenance-rows"]
    corpus = pathlib.Path("D:/a/openmucf/build/provenance_reader_corpus")
    paths = [corpus / f"random-{k:04d}.json" for k in range(1500)]
    out = mod.batches(head, paths)
    assert [p for batch in out for p in batch] == [str(p) for p in paths]
    assert len(out) > 1
    assert all(_line(head + batch) <= mod.COMMAND_LINE_BUDGET for batch in out)
    assert mod.COMMAND_LINE_BUDGET < 32767


def test_a_path_longer_than_the_budget_runs_alone():
    mod = _script()
    head = ["reader", "provenance-rows"]
    paths = [pathlib.Path("short.json"), pathlib.Path("x" * 200), pathlib.Path("tail.json")]
    out = mod.batches(head, paths, budget=100)
    assert out == [["short.json"], ["x" * 200], ["tail.json"]]
    assert mod.batches(head, paths[1:], budget=100) == [["x" * 200], ["tail.json"]]


def test_no_paths_give_no_batches():
    assert _script().batches(["reader", "provenance-rows"], []) == []
