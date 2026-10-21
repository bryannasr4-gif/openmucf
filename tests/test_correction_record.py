"""The change log's entry for the joint loss-channel response, the reactivation floor by side-channel loss,
the design note's basis and the release the citation metadata describe: each figure it states is the value
the file that publishes it prints, and it states no other figure.

The homes are the manifests of FINDINGS.md, MATERIALITY.md and FRONTIER.md, the calibration record as its
gate admits it (the correlation release v1.3.0's sweep printed, and the corrected sweep's), and the
`#VERSION` of the D3 tables on the tree. Every other numeric token of the entry belongs to a declared class,
each with its reason.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HEADING = "### Fixed -- the joint response of the deferred loss channels"
ROW = "R=legacy/os0=legacy/lc=narrow"
RELEASED = "CALIBRATION.md|Prior-sensitivity sweep (weak-prior mode; 4 chains x 1000)|" + ROW + "|"

#: (pattern, reason): the numeric tokens of the entry that are not figures a home publishes
CLASSES = (
    (r"\b\d{4}-\d{2}-\d{2}\b", "a date: the entry's own, and that of the CALIBRATION.md amendment it names"),
    (r"\bv\d+\.\d+\.\d+\b", "a release version string"),
    (r"\b[Ss]ection \d+\b", "a section number of the document the sentence names"),
    (r"\b3He\b", "the nuclide helium-3"),
    (r"\bomega_s0\b", "the symbol of the initial sticking"),
    (r"\bD3\b", "the dataset's name"),
    (r"`?\bmudirac130\w*`?", "a D3 profile name"),
)


def _manifest(name: str) -> dict:
    return {e["id"]: e["value"] for e in json.loads((REPO / name).read_text(encoding="utf-8"))["entries"]}


@functools.lru_cache(maxsize=1)
def _cells():
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        spec = importlib.util.spec_from_file_location(
            "_gen_calibration_record", REPO / "scripts" / "generate_calibration.py"
        )
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        return gen.admit_calibration_cells().data
    finally:
        os.chdir(cwd)


def homes() -> dict:
    """The published values the entry's figures must equal."""
    fm, mm, rm = _manifest("FINDINGS_MANIFEST.json"), _manifest("MATERIALITY_MANIFEST.json"), _manifest(
        "FRONTIER_MANIFEST.json")
    cells = _cells()
    versions = {re.search(r"^#VERSION\s+(\S+)\s*$", p.read_text(encoding="utf-8"), re.M).group(1)
                for p in sorted((REPO / "data/g4/d3").glob("*.g4dat"))}
    assert len(versions) == 1, versions
    sweep = [r for r in cells["sweep"] if r["config"] == ROW]
    assert len(sweep) == 1
    return {
        "sum": fm["structural_band_pct"],
        "joint": [mm[f"{op}_joint_mag"] for op in ("OP-B", "OP-C", "OP-D")],
        "he_given": (mm["he_given_min"], mm["he_given_max"]),
        "floor": rm["R_floor_500"], "floor_anchor": rm["R_floor_500_anchor"],
        "rise": rm["R_floor_rise"], "step": rm["L_step_pct"],
        "v130_corr": cells["shipped_v1_3_0"][RELEASED + "corr"],
        "sweep_corr": f"{sweep[0]['corr']:.2f}",
        "dataset_version": versions.pop(),
    }


def figure_phrases(h: dict) -> list[str]:
    """Each figure of the entry in the words that state it, built from its home."""
    b, c, d = h["joint"]
    return [
        f"by up to {h['sum']} % combined",
        f"the model lowers the yield by {b}, {c} and {d} % at OP-B, OP-C and OP-D",
        f"with the other two channels on, {h['he_given'][0]} to {h['he_given'][1]} fusions per muon",
        f"the floor R >= {h['floor']} at any density when sticking and decay are the only losses",
        f"R >= {h['floor_anchor']} at that fill's side-channel loss",
        f"the side-channel loss per cycle, {h['rise']} for each {h['step']} %",
        f"attributed the +{h['v130_corr']} omega_s0/R correlation",
        f"prints {h['sweep_corr']} for the note's omega_s0 and R boxes",
        f"the `mudirac130p1` profile, at dataset version {h['dataset_version']}, is in no release yet",
    ]


def entry(changelog: str) -> str:
    """The entry: its heading line through the line before the next heading, whitespace-normalised."""
    text = changelog.replace("\r\n", "\n")
    hits = [m.start() for m in re.finditer(rf"^{re.escape(HEADING)}", text, re.M)]
    assert len(hits) == 1, len(hits)
    body = text.index("\n", hits[0])
    end = re.search(r"^#{2,3} ", text[body:], re.M)
    return " ".join(text[hits[0]: body + end.start() if end else len(text)].split())


def entry_failures(changelog: str, h: dict) -> list[str]:
    """Each figure phrase the entry lacks, and each numeric token in it that is neither a bound figure nor of
    a declared class."""
    text = entry(changelog)
    phrases = figure_phrases(h)
    bad = [f"not in the entry: {p!r}" for p in phrases if p not in text]
    rest = text
    for p in phrases:
        rest = rest.replace(p, " ")
    for pattern, _ in CLASSES:
        rest = re.sub(pattern, " ", rest)
    bad += [f"unbound figure {t!r}" for t in re.findall(r"\d[\d.]*", rest)]
    return bad


def _changelog() -> str:
    return (REPO / "CHANGELOG.md").read_text(encoding="utf-8")


def test_every_figure_of_the_entry_is_its_homes():
    assert entry_failures(_changelog(), homes()) == []


def test_the_entry_states_each_figure_once_per_phrase():
    """Each phrase is found exactly once, so a drill that alters it cannot be masked by a second copy."""
    text = entry(_changelog())
    for p in figure_phrases(homes()):
        assert text.count(p) == 1, p


def _nudge(figure: str) -> str:
    return figure[:-1] + str((int(figure[-1]) + 1) % 10)


@pytest.mark.parametrize("key", ["sum", "joint", "he_given", "floor", "floor_anchor", "rise", "step",
                                 "v130_corr", "sweep_corr", "dataset_version"])
def test_correction_record_drill_refuses_a_perturbed_figure(key):
    """Drill: each figure of the entry, altered in a copy of the change log, is refused."""
    h = homes()
    old = h[key][-1] if isinstance(h[key], (list, tuple)) else h[key]
    phrase = next(p for p in figure_phrases(h) if old in p)
    changelog = _changelog()
    assert changelog.count(phrase) == 1, phrase
    # the figure as a whole token of its phrase (0.1 is not the 0.1 of 0.12)
    nudged = re.sub(rf"(?<![\d.]){re.escape(old)}(?![\d])", _nudge(old), phrase, count=1)
    assert nudged != phrase, key
    drilled = changelog.replace(phrase, nudged)
    assert drilled != changelog, key
    assert any(f == f"not in the entry: {phrase!r}" for f in entry_failures(drilled, h)), key


def test_correction_record_drill_refuses_an_unbound_figure():
    """Drill: a figure no home publishes, appended to the entry, is refused."""
    changelog = _changelog()
    anchor = "are regenerated."
    assert anchor in entry(changelog)
    drilled = changelog.replace(anchor, anchor + " The joint response moves by 12.5 %.", 1)
    assert any("unbound figure '12.5'" in f for f in entry_failures(drilled, homes()))
