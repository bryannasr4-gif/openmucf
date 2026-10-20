"""DESIGN.md's Method and Basis paragraphs against the record they cite.

DESIGN.md ranks the next experiments over a calibration chain pinned in `openmucf.design`: the omega_s0, R
and lambda_c boxes calibrate used before they were widened, under `openmucf.calibrate.model`'s likelihood.
Its Method paragraph attributes that chain's omega_s0/R correlation to the prior-sensitivity sweep of
CALIBRATION.md as released in v1.3.0, and its Basis paragraph gives the correlation the corrected sweep
prints for the same omega_s0 and R boxes. Both figures are read here from the calibration record as the
calibration's gate admits it (the released print from ``shipped_v1_3_0``, the corrected one from the sweep
rows), independently of the generator's own reading, and every figure the two paragraphs print must be the
record's or the design's pins. No NUTS runs.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from openmucf import calibrate, design

REPO = Path(__file__).resolve().parents[1]
ROW = "R=legacy/os0=legacy/lc=narrow"
RELEASED = "CALIBRATION.md|Prior-sensitivity sweep (weak-prior mode; 4 chains x 1000)|" + ROW + "|"


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def _cells():
    """The calibration record as its gate admits it, read from the repository root."""
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        return _load("scripts/generate_calibration.py", "_gen_calibration_basis").admit_calibration_cells()
    finally:
        os.chdir(cwd)


def _thaw(node):
    """A mutable copy of the admitted (read-only) record, for the drills."""
    if isinstance(node, Mapping):
        return {k: _thaw(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_thaw(v) for v in node]
    return node


def _sweep_row(data):
    rows = [r for r in data["sweep"] if r["config"] == ROW]
    return rows[0] if len(rows) == 1 else None


def _paragraphs(design_md: str) -> str:
    """The Method and Basis paragraphs of DESIGN.md, whitespace-normalised."""
    text = design_md.replace("\r\n", "\n")
    start = text.index("**Method (no new physics).**")
    basis = text.index("**Basis: the likelihood before the anchor-basis correction.**")
    end = text.find("\n\n", basis)
    return " ".join(text[start:end].split())


def _box(lo: float, hi: float, fmt: str) -> str:
    return f"({lo:{fmt}}, {hi:{fmt}})"


def basis_failures(design_md: str, data) -> list[str]:
    """Each figure the Method and Basis paragraphs must print and do not, from the record ``data`` (the
    admitted calibration cells) and the design's pins; and each mismatch between the boxes the cited sweep
    rows ran and the design's pinned omega_s0 and R boxes, which is what makes them the rows for "these
    boxes"."""
    par = _paragraphs(design_md)
    released_corr = data["shipped_v1_3_0"][RELEASED + "corr"]
    released_boxes = data["shipped_v1_3_0"][RELEASED + "boxes"]
    row = _sweep_row(data)
    if row is None:
        return [f"the record does not hold exactly one sweep row {ROW}"]
    pins = {"os0": (design.BASE_PRIOR[1], design.BASE_PRIOR[2]), "R": (design._R_LO, design._R_HI)}
    bad = []
    # the cited rows ran the design's omega_s0 and R boxes, released and corrected alike
    for name, (lo, hi) in pins.items():
        if f"{name}[{lo:.2f},{hi:.2f}]" not in released_boxes:
            bad.append(f"the released row's boxes {released_boxes!r} are not the design's {name} box")
    if (list(row["os0_box"]), list(row["R_box"])) != (list(pins["os0"]), list(pins["R"])):
        bad.append(f"the corrected sweep row's boxes {row['os0_box']}, {row['R_box']} are not the design's")
    edge = re.search(r"lc\[[^,\]]+,([^\]]+)\]", released_boxes)
    obs = design._OBS
    need = {
        "the design's boxes": (
            f"Uniform omega_s0 {_box(*pins['os0'], '.2f')}, Uniform R {_box(*pins['R'], '.2f')} and Uniform "
            f"lambda_c {_box(design._LAMBDA_C_LO, design._LAMBDA_C_HI, '.2g')} s^-1"
        ),
        "the design's likelihood": (
            f"the effective sticking {obs['omega_s_eff_obs']:g} +- {obs['omega_s_eff_sd']:g} % taken as the "
            f"whole loss per cycle and the yield {obs['xmu_obs']:g} +- {obs['xmu_sd']:g}, with no cycle-rate "
            "measurement"
        ),
        "the released correlation": (
            f"as released in v1.3.0 printed as +{released_corr} for these boxes (its row {ROW},"
        ),
        "the released box edge": f"printed at one significant figure as {edge.group(1) if edge else '?'})",
        "the corrected likelihood's yield": f"the yield carries its printed +-{calibrate.OBS_W['xmu_sd']:g}.",
        "the corrected sweep's box": f"cycle-rate box {_box(row['lc_box'][0], row['lc_box'][1], '.2g')} s^-1",
        "the corrected sweep's correlation": f"an omega_s0/R correlation of {row['corr']:.2f}:",
    }
    bad += [f"{what}: {text!r} not in DESIGN.md" for what, text in need.items() if text not in par]
    return bad


def _design_md() -> str:
    return (REPO / "DESIGN.md").read_text(encoding="utf-8")


def test_design_prints_the_record_it_cites():
    assert basis_failures(_design_md(), _cells().data) == []


def test_the_cited_figures_differ():
    """The released print and the corrected sweep's are different figures, so the drills tell them apart."""
    data = _cells().data
    assert data["shipped_v1_3_0"][RELEASED + "corr"] != f"{_sweep_row(data)['corr']:.2f}"


@pytest.mark.parametrize("old, new", [
    ("printed as +0.84 for these boxes", "printed as +0.85 for these boxes"),
    ("an omega_s0/R correlation of 0.82:", "an omega_s0/R correlation of 0.85:"),
    ("cycle-rate box (8e+07, 2e+08) s^-1", "cycle-rate box (8e+07, 1.6e+08) s^-1"),
    ("its printed +-10.", "its printed +-12."),
    ("Uniform lambda_c (8e+07, 1.6e+08) s^-1", "Uniform lambda_c (8e+07, 2e+08) s^-1"),
])
def test_basis_drill_refuses_an_altered_figure(old, new):
    """Drill: one figure of the Method or Basis paragraph altered in a copy of DESIGN.md is refused."""
    doc = " ".join(_design_md().split())
    assert old in doc, old
    assert basis_failures(doc.replace(old, new), _cells().data)


def test_basis_drill_refuses_an_altered_record():
    """Drill: the record's released print or its corrected sweep row altered under an unchanged DESIGN.md is
    refused, and so is a sweep row that ran other boxes."""
    base = _thaw(_cells().data)
    for edit in (
        lambda d: d["shipped_v1_3_0"].__setitem__(RELEASED + "corr", "0.85"),
        lambda d: _sweep_row(d).__setitem__("corr", 0.85),
        lambda d: _sweep_row(d).__setitem__("os0_box", [0.5, 1.2]),
    ):
        data = _thaw(base)
        edit(data)
        assert data != base
        assert basis_failures(_design_md(), data)
