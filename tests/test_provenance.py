"""WS-A: the machine-checkable provenance manifest (openmucf.provenance + FINDINGS_MANIFEST.json)."""

import json
from pathlib import Path

import openmucf
from openmucf import provenance
from openmucf.rates import RATES_CSV, TARGETS_CSV

REPO_ROOT = Path(openmucf.__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "FINDINGS_MANIFEST.json"

# The 25 headline ids the FINDINGS manifest must always carry (WAVE1 spec 1.4 minimum set).
MIN_IDS = {
    "sobol_xmu_ST_R", "sobol_xmu_ST_lambda_c", "sobol_xmu_ST_omega_s0_pct",
    "sobol_qnet_ST_E_mu_GeV", "sobol_qnet_ST_eta_acc",
    "robustness_R_box_i", "robustness_R_box_ii",
    "robustness_lambda_c_box_i", "robustness_lambda_c_box_ii",
    "robustness_omega_s0_pct_box_i", "robustness_omega_s0_pct_box_ii",
    "xmu_ci_lo", "xmu_ci_med", "xmu_ci_hi",
    "qsci_ci_lo", "qsci_ci_med", "qsci_ci_hi",
    "qnet_ci_lo", "qnet_ci_med", "qnet_ci_hi",
    "P_qsci_gt1", "P_qnet_gt1", "P_xmu_gt500",
    "cap_zero_sticking", "R_required",
}


def test_manifest_checks_clean():
    """Every committed manifest value still appears (anchored) in its doc."""
    assert provenance.check_manifest(MANIFEST, repo_root=REPO_ROOT) == []


def test_manifest_mutation_detected(tmp_path):
    """Corrupt one digit of FINDINGS.md in a scratch copy and repoint a copied manifest at it: the check
    must flag the now-missing value. In-suite version of the G-A1 mutation drill (WAVE1 spec 1.6 test 5)."""
    findings = (REPO_ROOT / "FINDINGS.md").read_text(encoding="utf-8")
    corrupted = findings.replace("**X_mu = 319**", "**X_mu = 318**", 1)  # zero-sticking cap 319 -> 318
    assert corrupted != findings
    (tmp_path / "FINDINGS.md").write_text(corrupted, encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    (tmp_path / "FINDINGS_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    failures = provenance.check_manifest(tmp_path / "FINDINGS_MANIFEST.json", repo_root=tmp_path)
    assert failures  # non-empty
    assert any("cap_zero_sticking" in f for f in failures)


def test_manifest_entry_coverage():
    """Every one of the 25 minimum ids is present (WAVE1 spec 1.4: extend, don't shrink)."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = {e["id"] for e in manifest["entries"]}
    assert ids >= MIN_IDS, f"missing ids: {sorted(MIN_IDS - ids)}"


def test_manifest_inputs_shas_current():
    """The recorded input SHAs equal a fresh LF-normalized hash of the two ledger CSVs."""
    inputs = json.loads(MANIFEST.read_text(encoding="utf-8"))["inputs"]
    assert inputs["rates_csv_sha256"] == provenance.file_sha256(RATES_CSV)
    assert inputs["validation_targets_csv_sha256"] == provenance.file_sha256(TARGETS_CSV)


def test_file_sha256_lf_normalization(tmp_path):
    """CRLF and LF encodings of the same text hash identically (immune to autocrlf checkouts)."""
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"alpha\nbeta\ngamma\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    assert provenance.file_sha256(lf) == provenance.file_sha256(crlf)
