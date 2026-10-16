"""MATERIALITY.md structural sensitivity brackets (scripts/generate_materiality.py).

Brackets are one-sided channel toggles reported side-by-side with the parametric CI, never convolved in.
Each bracket toggles one channel: the ttmu side-branch (in-target and density-normalized readings, c_He = 0,
recapture off), 3He scavenging (tt off) or the d-recapture routing.
"""

import importlib.util
import json
from pathlib import Path

import openmucf
from openmucf import load_rates, provenance

REPO_ROOT = Path(openmucf.__file__).resolve().parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "generate_materiality.py"
MANIFEST = REPO_ROOT / "MATERIALITY_MANIFEST.json"
DOC = REPO_ROOT / "MATERIALITY.md"


def _load_script():
    """Import the generator module by path (its file I/O is guarded behind main(), so import is inert)."""
    spec = importlib.util.spec_from_file_location("_gen_materiality", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_operating_points_and_c_he_match_spec():
    """Operating points + conditions + c_He levels are exactly the set pinned in this test."""
    mod = _load_script()
    assert mod.OPERATING_POINTS == {
        "OP-A": (1.25, 300.0, 0.5),
        "OP-B": (1.2, 800.0, 0.5),
        "OP-C": (2.0, 150.0, 0.5),
        "OP-D": (2.4, 100.0, 0.5),
    }
    assert mod.C_HE_LEVELS == (1e-4, 1e-3)
    # OP-A is the anchor-adjacent, non-headline row; the headline set is exactly OP-B/C/D.
    assert mod.ANCHOR_ADJACENT == "OP-A"
    assert mod.HEADLINE_OPS == ("OP-B", "OP-C", "OP-D")
    assert "OP-A" not in mod.HEADLINE_OPS


def test_he_brackets_nonpositive_and_monotone():
    """Sign check: the live 3He loss channel gives one-sided brackets (<= 0); a muon can
    only be lost, and more helium loses more (bracket strictly more negative at higher c_He)."""
    mod = _load_script()
    br = mod.he_brackets(load_rates())
    for op in mod.OPERATING_POINTS:
        b_lo = br[op][1e-4]["bracket"]
        b_hi = br[op][1e-3]["bracket"]
        assert b_lo <= 0.0 and b_hi <= 0.0, f"{op}: loss-channel bracket must be <= 0"
        assert b_hi < b_lo < 0.0, f"{op}: bracket must deepen with c_He (10x helium)"
        # the "with" value equals the channels-OFF value plus the (negative) bracket, by construction
        assert br[op][1e-4]["with"] == br[op]["off"] + br[op][1e-4]["bracket"]


def test_tt_brackets_toggle_the_tt_channel_alone():
    """The live ttmu channel gives one-sided brackets; the density-normalized reading (the larger rate) loses
    more; and the combined band is the sum of the three brackets."""
    mod = _load_script()
    rates = load_rates()
    tt = mod.tt_brackets(rates)
    for op in mod.OPERATING_POINTS:
        b_in, b_dn = tt[op]["in_target"]["bracket"], tt[op]["density_normalized"]["bracket"]
        assert b_dn < b_in < 0.0, op
        assert tt[op]["in_target"]["with"] == tt[op]["off"] + b_in
    he, q1s = mod.he_brackets(rates), mod.q1s_brackets(rates)
    comb = mod.combined_band(he, q1s, tt)
    for op in mod.HEADLINE_OPS:
        parts = (tt[op]["in_target"]["bracket"], he[op][1e-3]["bracket"], q1s[op][1.0]["bracket"])
        assert comb[op]["combined"] == parts[0] + parts[1] + parts[2]
    assert mod.structural_band_pct(rates) == max(abs(comb[op]["pct"]) for op in mod.HEADLINE_OPS)


def test_he_brackets_toggle_the_he_channel_alone():
    """With the tt channel live in the ledger, the 3He brackets still toggle 3He alone (lambda_tt zeroed)."""
    from openmucf import cycle

    mod = _load_script()
    rates = load_rates()
    br = mod.he_brackets(rates)
    for op, (phi, T, c_t) in mod.OPERATING_POINTS.items():
        for c_he in mod.C_HE_LEVELS:
            p = cycle.params_from_conditions(rates, T, phi, c_t, include_loss_channels=True, c_he=c_he)
            p["lambda_tt"] = 0.0
            assert br[op][c_he]["with"] == float(cycle.fusions_per_muon_ode(**p)), (op, c_he)
            assert p["lambda_he"] > 0.0


def test_manifest_checks_clean():
    """Every committed MATERIALITY_MANIFEST value still appears (anchored) in MATERIALITY.md."""
    assert provenance.check_manifest(MANIFEST, repo_root=REPO_ROOT) == []


def test_manifest_mutation_detected(tmp_path):
    """Corrupt one bracket digit in a scratch MATERIALITY.md and repoint a copied manifest at it: the
    check must flag the now-missing value (in-suite version of the manifest mutation drill)."""
    doc = DOC.read_text(encoding="utf-8")
    corrupted = doc.replace("-0.180", "-0.181", 1)  # OP-D 3He bracket at c_He=1e-3
    assert corrupted != doc, "expected the OP-D c_He=1e-3 bracket -0.180 to be present"
    (tmp_path / "MATERIALITY.md").write_text(corrupted, encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    (tmp_path / "MATERIALITY_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    failures = provenance.check_manifest(tmp_path / "MATERIALITY_MANIFEST.json", repo_root=tmp_path)
    assert failures
    assert any("OP-D_bracket" in f for f in failures)


def test_manifest_inputs_shas_current():
    """The recorded input SHAs equal a fresh LF-normalized hash of the ledger + the FINDINGS manifest."""
    inputs = json.loads(MANIFEST.read_text(encoding="utf-8"))["inputs"]
    assert inputs["rates_csv_sha256"] == provenance.file_sha256(REPO_ROOT / "openmucf/data/rates.csv")
    assert inputs["findings_manifest_sha256"] == provenance.file_sha256(REPO_ROOT / "FINDINGS_MANIFEST.json")
