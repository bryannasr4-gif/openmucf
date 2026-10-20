"""An accurately represented adverse certificate cannot enter the generated publication."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmucf.g4 import d3_contract as contract
from openmucf.g4.sources import mudirac130 as md


@pytest.mark.parametrize("verdicts", [[], ["NUMERICAL_FAIL"], ["QUALIFIED", "NUMERICAL_FAIL"]])
def test_publication_rejects_empty_or_adverse(verdicts):
    checks = {(i, i, True, "K"): SimpleNamespace(verdict=value) for i, value in enumerate(verdicts)}
    with pytest.raises(md.CellError, match="publication refused"):
        contract.require_numerical_qualification(checks)


def test_publication_accepts_qualified():
    contract.require_numerical_qualification({(6, 12, True, "K"): SimpleNamespace(verdict="QUALIFIED")})


def test_generator_requires_qualification_before_build(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "d3_publication_generator", root / "scripts/generate_g4data.py")
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generator
    spec.loader.exec_module(generator)
    checks = {(6, 12, True, "K"): SimpleNamespace(verdict="NUMERICAL_FAIL")}
    monkeypatch.setattr(contract, "load_bundle", lambda root: SimpleNamespace(checks=checks))
    monkeypatch.setattr(generator.d1src, "load", lambda path: pytest.fail("publication guard bypassed"))
    with pytest.raises(md.CellError, match="publication refused"):
        generator.build_dataset_artifacts()
