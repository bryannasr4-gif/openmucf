"""The change log's anchor-basis entry: the figures of its "Numbers that move" and "An expectation that was
wrong" bullets are the values their homes print."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
W = "CALIBRATION.md|Weak omega_s0 prior (experimental data only) -- exposes the degeneracy|"
K = "CALIBRATION.md|Informative omega_s0 prior (Kamimura 0.857+-0.03 %) -- partially resolves it|"
P = "CALIBRATION.md|Posterior pushforward (state-of-knowledge X_mu and Q_net)|"


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


def _bullet(changelog, head):
    hits = [ln for ln in changelog.splitlines() if ln.startswith(f"- **{head}**")]
    assert len(hits) == 1, (head, len(hits))
    return hits[0]


def _predicted(page, target):
    rows = [ln for ln in page.splitlines() if ln.startswith(f"| {target} |")]
    assert len(rows) == 1, target
    return rows[0].split("|")[4].strip()


def record_failures(changelog, cells, findings_manifest, validation, validation_channels):
    sh, po = cells["shipped_v1_3_0"], cells["posteriors"]
    fm = {e["id"]: e["value"] for e in findings_manifest["entries"]}
    vp = _predicted(validation, "V_petitjean")
    vb = _predicted(validation_channels, "V_breunlich_lambdac")
    moved = _bullet(changelog, "Numbers that move.")
    figures = {
        f"from {sh[W + 'lambda_c|mean']} to {po['weak']['lambda_c']['mean']:.3g} s^-1",
        f"from {sh[W + 'omega_s_eff_pct|mean']} to {po['weak']['omega_s_eff_pct']['mean']:.3g} %",
        f"R from {sh[K + 'R|mean']} to {po['kamimura']['R']['mean']:.3g}",
        f"yield from {sh[P + 'X_mu (weak-box posterior)|mean']} to {po['weak']['X_mu']['mean']:.4g}.",
        f"floor R >= {fm['R_required_anchor']} at",
        f"beside R >= {fm['R_required']},",
        f"reaches {fm['structural_band_pct']} %",
        f"total loss, {vp} where {sh['VALIDATION.md|V_petitjean|predicted']} was printed",
    }
    bad = [f"Numbers that move: {f!r} not in the bullet" for f in sorted(figures) if f not in moved]
    bound = {t for f in figures for t in re.findall(r"\d[\d.e+]*\d|\d", f)}
    loose = [t for t in re.findall(r"\d[\d.e+]*\d|\d", moved) if t not in bound and t not in ("2", "3")]
    bad += [f"Numbers that move: unbound figure {t!r}" for t in loose]
    old = sh["VALIDATION_CHANNELS.md|V_breunlich_lambdac|predicted"]
    pct = f"{100.0 * (float(vb) / float(old) - 1.0):.3f}"
    wrong = _bullet(changelog, "An expectation that was wrong.")
    for f in (f"stay at {old} s^-1.", f"it is {vb} s^-1, {pct} %:"):
        if f not in wrong:
            bad.append(f"An expectation that was wrong: {f!r} not in the bullet")
    return bad


def _homes():
    cells, manifest = json.loads(_read("CALIBRATION_CELLS.json")), json.loads(_read("FINDINGS_MANIFEST.json"))
    return _read("CHANGELOG.md"), cells, manifest, _read("VALIDATION.md"), _read("VALIDATION_CHANNELS.md")


def test_changelog_moved_figures_stand_in_their_homes():
    assert record_failures(*_homes()) == []


@pytest.mark.parametrize("old, new, named", [
    (" to 1.44e+08 s^-1", " to 1.45e+08 s^-1", "Numbers that move"),
    ("reaches 26.67 %", "reaches 26.68 %", "Numbers that move"),
    ("it is 143577225.6 s^-1", "it is 143577225.7 s^-1", "An expectation that was wrong"),
    ("-0.171 %:", "-0.172 %:", "An expectation that was wrong"),
])
def test_changelog_record_drill(old, new, named):
    """Drill: one figure of a bullet altered in a copy of the change log is named."""
    changelog, *rest = _homes()
    assert changelog.count(old) == 1, old
    assert [f for f in record_failures(changelog.replace(old, new), *rest) if f.startswith(named)]
