"""WS-A: constants single-sourcing + the engine-number regression lock.

``openmucf.constants`` re-exports three physical constants from the ledger so multiple engine modules
stop carrying forked literals. These tests prove the rewire is numerically inert and that no fork remains.
"""

from pathlib import Path

import pytest

import openmucf
from openmucf import constants, cycle, load_rates
from openmucf.energy import EnergyChain

_PKG = Path(openmucf.__file__).resolve().parent


def test_constants_match_ledger():
    """constants.py re-exports the three ledger rows exactly (so the rewire changed zero numbers)."""
    r = load_rates(check_refs=False)
    assert r.value("lambda_mu_decay") == constants.LAMBDA_0
    assert r.value("E_fusion") == constants.E_F_MEV
    assert r.value("E_mu_cost") == constants.E_MU_GEV_DEFAULT


def _executable_lines(text: str) -> str:
    """Heuristic strip of full-line comments and triple-quoted blocks (WAVE1 spec 1.6 test 2).

    Deliberately simple -- NOT a Python parser. It drops lines whose stripped form starts with '#'
    and the contents of triple-quoted docstring blocks. Good enough to tell an executable constant
    literal from a comment/docstring mention; documented as heuristic per the spec.
    """
    out = []
    in_doc = False
    doc_quote = ""
    for line in text.splitlines():
        s = line.strip()
        if in_doc:
            if doc_quote in s:
                in_doc = False
            continue
        if s.startswith("#"):
            continue
        opened = False
        for q in ('"""', "'''"):
            if s.startswith(q):
                # a one-line docstring ("""...""") is fully non-executable; a bare opener starts a block
                if not (len(s) >= 2 * len(q) and s.endswith(q) and s.count(q) >= 2):
                    in_doc = True
                    doc_quote = q
                opened = True
                break
        if not opened:
            out.append(line)
    return "\n".join(out)


def test_no_duplicate_literals():
    """No engine module (except constants.py, and the ledger loader rates.py) still carries the
    physical-constant literals in executable code -- they now come from the ledger via constants.py."""
    skip = {"constants.py", "rates.py"}
    offenders = []
    for path in sorted(_PKG.glob("*.py")):
        if path.name in skip:
            continue
        code = _executable_lines(path.read_text(encoding="utf-8"))
        for literal in ("4.552e5", "= 17.6"):
            if literal in code:
                offenders.append(f"{path.name}: {literal!r}")
    assert offenders == [], f"forked constant literals in executable code: {offenders}"


def test_engine_numbers_unchanged():
    """The rewire is numerically inert: the canonical quickstart yield and the scientific breakeven are
    bit-for-bit what they were before WS-A (full-precision values recorded on clean main, 2026-07-07)."""
    r = load_rates()
    x = float(cycle.fusions_per_muon_from_conditions(r, T=300, phi=1.2, c_t=0.5))
    assert x == pytest.approx(114.47527542334024, rel=1e-9)
    assert EnergyChain().breakeven_xmu_sci() == pytest.approx(5.0e3 / 17.6, rel=1e-12)
