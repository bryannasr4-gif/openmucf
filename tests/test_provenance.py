"""The machine-checkable provenance manifest (openmucf.provenance + FINDINGS_MANIFEST.json)."""

import json
from pathlib import Path

import openmucf
from openmucf import provenance
from openmucf.rates import RATES_CSV, TARGETS_CSV

REPO_ROOT = Path(openmucf.__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "FINDINGS_MANIFEST.json"

# The 25 headline ids the FINDINGS manifest must always carry (the minimum set).
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
    must flag the now-missing value. In-suite version of the manifest mutation drill."""
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
    """Every one of the 25 minimum ids is present (the set may be extended, never shrunk)."""
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


def test_manifest_pins_the_first_order_indices_and_the_interaction_share():
    """The first-order Sobol index of each input whose total-order index is pinned, and the ST - S1
    share of the top X_mu driver, are manifest values of their own."""
    ids = {e["id"] for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["entries"]}
    for st_id in sorted(i for i in ids if i.startswith(("sobol_xmu_ST_", "sobol_qnet_ST_"))):
        assert st_id.replace("_ST_", "_S1_") in ids, st_id
    assert "sobol_xmu_interaction_R" in ids


def test_each_sobol_index_is_pinned_to_its_own_cell(tmp_path):
    """A first-order and a total-order value of one row, swapped between their entries, fail the check.

    Both indices sit in the same table row, so an entry anchored on the whole row would accept the other
    index's value; each entry's matched text must be its own cell.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in manifest["entries"]}
    st_ids = [i for i in sorted(by_id) if i.startswith(("sobol_xmu_ST_", "sobol_qnet_ST_"))]
    pairs = [(i, i.replace("_ST_", "_S1_")) for i in st_ids]
    assert pairs
    findings = (REPO_ROOT / "FINDINGS.md").read_text(encoding="utf-8")
    (tmp_path / "FINDINGS.md").write_text(findings, encoding="utf-8")
    for st_id, s1_id in pairs:
        assert by_id[st_id]["value"] != by_id[s1_id]["value"], (st_id, s1_id)
        for target, other in ((st_id, s1_id), (s1_id, st_id)):
            swapped = json.loads(json.dumps(manifest))
            next(e for e in swapped["entries"] if e["id"] == target)["value"] = by_id[other]["value"]
            (tmp_path / "FINDINGS_MANIFEST.json").write_text(json.dumps(swapped), encoding="utf-8")
            failures = provenance.check_manifest(tmp_path / "FINDINGS_MANIFEST.json", repo_root=tmp_path)
            assert [f for f in failures if f.startswith(f"{target}:")], (target, failures)
