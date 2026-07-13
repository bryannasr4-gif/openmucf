"""Validation tests for the FAIR rate ledger (Phase 1)."""

import csv
import math
import re

from openmucf import load_rates
from openmucf.rates import DATA, RATES_CSV, REFS_BIB, TARGETS_CSV, bibkeys

MUON_COST_CSV = DATA / "muon_cost.csv"
BIB_UNRESOLVED = DATA / "bib_unresolved.txt"


def test_ledger_loads_and_validates():
    rt = load_rates()  # raises ValueError on any provenance/schema problem
    assert len(rt) >= 10


def test_settled_values():
    rt = load_rates()
    assert math.isclose(rt.value("lambda_mu_decay"), 4.552e5, rel_tol=1e-3)
    assert math.isclose(rt.value("omega_s0"), 0.857, rel_tol=1e-9)
    assert math.isclose(rt.value("E_fusion"), 17.6, rel_tol=1e-9)
    assert rt["omega_s0"].source_bibkey == "Kamimura2023"


def test_every_rate_is_sourced_and_in_bib():
    rt = load_rates()
    known = bibkeys()
    for r in rt._rates.values():
        assert r.source_bibkey, f"{r.symbol} has no source"
        for key in r.source_bibkey.replace(";", ",").split(","):
            key = key.strip()
            if key:
                assert key in known, f"{r.symbol}: bibkey {key} missing from references.bib"


def test_contested_and_unverified_are_flagged_not_hidden():
    rt = load_rates()
    # The two single-quoted/placeholder formation points must be flagged for Phase-2 digitization.
    for sym in ("lambda_dtmu_lowT", "lambda_dt_transfer", "lambda_10_spinflip"):
        assert rt[sym].needs_verification is True


def test_nominal_vector_is_float64():
    rt = load_rates()
    v = rt.nominal_vector(["lambda_mu_decay", "omega_s0", "R_col", "lambda_f_dtmu"])
    assert str(v.dtype) == "float64"
    assert v.shape == (4,)
    # ordering preserved
    assert math.isclose(float(v[0]), 4.552e5, rel_tol=1e-3)


def test_validation_targets_present_and_well_formed():
    assert TARGETS_CSV.exists()
    with open(TARGETS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = {r["target_id"] for r in rows}
    # the two marquee Kou-Chen reproduction anchors and the Acceleron regime anchor
    assert {"V_kouchen_base", "V_kouchen_best", "A_acceleron_anomaly"} <= ids


def test_ledger_new_columns_validate(tmp_path):
    """The loader accepts the extended ledger; a bad `distribution` enum value raises."""
    from pathlib import Path

    import pytest

    from openmucf.rates import RATES_CSV

    r = load_rates()
    assert "lambda_c_liquid" in r
    assert r["eta_dtmu"].distribution == "asym_interval"
    assert r.dist_bounds("lambda_c_liquid") == (1.00e8, 1.45e8)

    text = Path(RATES_CSV).read_bytes().decode("utf-8").replace("\r\n", "\n")
    bad = text.replace(",asym_interval,1.0,5.0,,gas,D2", ",NOTADIST,1.0,5.0,,gas,D2")
    assert bad != text  # the target substring was actually present
    p = tmp_path / "bad_rates.csv"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        load_rates(csv_path=p)


def test_recommended_superseded_pair():
    """omega_s0 is `recommended`, omega_s0_legacy is `superseded`; exactly one `recommended` in the family."""
    r = load_rates()
    assert r["omega_s0"].recommendation == "recommended"
    assert r["omega_s0_legacy"].recommendation == "superseded"
    fam_recommended = [
        s
        for s in r.symbols()
        if r[s].recommendation == "recommended" and s.replace("_legacy", "") == "omega_s0"
    ]
    assert fam_recommended == ["omega_s0"]


def test_wsn_loss_channel_rows_load_and_flagged():
    """The three WS-N loss-channel rows load, pass the enum/interval loader, and are needs_verification.

    lambda_ttmu ships the I10 blocked fallback (0.0 + `blocked:` note); lambda_dhe3 ships a real value
    from a live open source (Fotev et al., arXiv:2001.09927 = bibkey Fotev2020)."""
    r = load_rates()
    for sym in ("lambda_ttmu", "omega_tt", "lambda_dhe3"):
        assert sym in r, sym
        assert r[sym].needs_verification is True, sym
        assert r[sym].phase and r[sym].target_molecule, sym  # non-empty typed projections (WS-L schema)
    assert r.value("lambda_ttmu") == 0.0  # blocked fallback -> refit keys on this (SKIP)
    assert r["lambda_ttmu"].notes.startswith("blocked:")
    assert r["lambda_ttmu"].source_bibkey == "BomTT2005"
    assert r.value("lambda_dhe3") > 0.0
    assert r["lambda_dhe3"].source_bibkey == "Fotev2020"


# --- FAIR provenance: DOI/URL backfill -------------------------------------------------------

def _bib_entries():
    """(bibkey, entry-body) for every entry in references.bib."""
    text = REFS_BIB.read_text(encoding="utf-8")
    blocks = [b for b in re.split(r"(?=^@)", text, flags=re.M) if b.lstrip().startswith("@")]
    return [(re.match(r"@\w+\{([^,]+),", b).group(1).strip(), b) for b in blocks]


def _bib_keys_with_identifier():
    """Bibkeys carrying a machine-resolvable `doi` or `url` field."""
    return {k for k, body in _bib_entries() if re.search(r"\n\s*(?:doi|url)\s*=\s*\{", body)}


def _unresolved_entries():
    """Parsed bib_unresolved.txt rows: list of (bibkey, why, route)."""
    rows = []
    for line in BIB_UNRESOLVED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            rows.append(parts)
    return rows


def _referenced_bibkeys():
    """Every bibkey cited by rates.csv / validation_targets.csv / muon_cost.csv (';'/',' split)."""
    keys = set()
    for path in (RATES_CSV, TARGETS_CSV, MUON_COST_CSV):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = (row.get("source_bibkey") or "").strip()
                for key in raw.replace(";", ",").split(","):
                    key = key.strip()
                    if key:
                        keys.add(key)
    return keys


def test_every_referenced_bibkey_resolves_or_is_listed_unresolved():
    """Every bibkey a shipped CSV cites has a live-verified doi/url in references.bib,
    or is recorded in bib_unresolved.txt with an acquisition route. No orphan, no fabricated id."""
    known = bibkeys()
    resolvable = _bib_keys_with_identifier()
    unresolved = {r[0] for r in _unresolved_entries()}
    for key in sorted(_referenced_bibkeys()):
        assert key in known, f"{key} cited by a data CSV but missing from references.bib"
        assert key in resolvable or key in unresolved, (
            f"{key}: no doi/url in references.bib and not listed in bib_unresolved.txt "
            "(add a live-verified identifier or record the acquisition route)"
        )


def test_bib_unresolved_entries_are_well_formed_and_known():
    """Each bib_unresolved.txt line is `bibkey | why | route`, all fields non-empty, key in the bib,
    and does NOT also carry a doi/url (an entry is resolved xor listed-unresolved, never both)."""
    resolvable = _bib_keys_with_identifier()
    rows = _unresolved_entries()
    assert rows, "bib_unresolved.txt has no entries to review"
    for parts in rows:
        assert len(parts) == 3, f"expected 'bibkey | why | route', got {parts!r}"
        key, why, route = parts
        assert key and why and route, f"empty field in bib_unresolved row: {parts!r}"
        assert key in bibkeys(), f"unresolved bibkey {key!r} not defined in references.bib"
        assert key not in resolvable, f"{key} is both DOI/URL-resolved and listed unresolved"


def test_backfilled_dois_are_wellformed():
    """Every doi field is a bare DOI (starts 10.), never a full URL; arXiv dois use the 10.48550 form."""
    for key, body in _bib_entries():
        for doi in re.findall(r"\n\s*doi\s*=\s*\{([^}]*)\}", body):
            assert doi.startswith("10."), f"{key}: doi {doi!r} is not a bare DOI"
            assert "doi.org" not in doi and "http" not in doi, f"{key}: doi {doi!r} must not be a URL"
