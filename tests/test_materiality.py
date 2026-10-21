"""MATERIALITY.md structural sensitivity brackets (scripts/generate_materiality.py).

Brackets are one-sided channel toggles reported side-by-side with the parametric CI, never convolved in.
Each bracket toggles one channel: the ttmu side-branch (in-target and density-normalized readings, c_He = 0,
recapture off), 3He scavenging (tt off) or the d-recapture routing. Section 5 also switches the three on
together; its published figures are checked against the exact linear-algebra oracle, and every per-row table
value of the manifest is anchored to its own section, table, row and column.
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


# ------------------------------------------------------------------ the three channels together (section 5)
# The published joint response is checked against the engine's exact linear-algebra oracle (matrix exponential
# and solve, openmucf.exact), a different algorithm from the ODE the generator integrates. The parameters are
# assembled here from params_from_conditions, not through the generator's helpers.
_T1 = 3.0e-5  # the ODE's end time (cycle.solve_cycle), at which the oracle is evaluated


def _oracle_x(rates, op, *, tt="in_target", c_he=1e-3, q_1s=1.0, channels=True):
    from openmucf import cycle, exact, sidechannels

    phi, T, c_t = _load_script().OPERATING_POINTS[op]
    p = cycle.params_from_conditions(rates, T, phi, c_t, include_loss_channels=channels, c_he=c_he, q_1s=q_1s)
    if tt is None:
        p["lambda_tt"] = 0.0
    elif tt == "density_normalized":
        p["lambda_tt"] = sidechannels.load_inputs()["lambda_ttmu_table"]["value"] * phi * c_t
    return exact.finite_totals(exact.pack_args(**p), _T1)["X_mu"]


def joint_oracle(rates) -> dict:
    """Every section-5 figure, by the exact oracle: ``{manifest id: value}`` (MATERIALITY) and, under the
    ``"findings"`` key, the extremes FINDINGS.md section 2 states."""
    mod = _load_script()
    out: dict = {}
    mags: dict = {"joint": [], "dn": [], "sum": []}
    for op in mod.HEADLINE_OPS:
        off = _oracle_x(rates, op, channels=False, c_he=0.0, q_1s=None)
        joint = _oracle_x(rates, op)
        dn = _oracle_x(rates, op, tt="density_normalized")
        alone = {
            "tt": _oracle_x(rates, op, c_he=0.0, q_1s=None) - off,
            "he": _oracle_x(rates, op, tt=None, q_1s=None) - off,
            "q1s": _oracle_x(rates, op, channels=False, c_he=0.0) - off,
        }
        given = {
            "tt": joint - _oracle_x(rates, op, tt=None),
            "he": joint - _oracle_x(rates, op, c_he=0.0),
            "q1s": joint - _oracle_x(rates, op, q_1s=None),
        }
        total = sum(alone.values())
        out.update({
            f"{op}_off": off,
            f"{op}_joint_with": joint,
            f"{op}_joint": joint - off,
            f"{op}_joint_pct": 100.0 * (joint - off) / off,
            f"{op}_joint_mag": abs(100.0 * (joint - off) / off),
            f"{op}_joint_dn": dn - off,
            f"{op}_joint_dn_pct": 100.0 * (dn - off) / off,
            f"{op}_comb_total": total,
            f"{op}_comb_pct": 100.0 * total / off,
        })
        for ch in ("tt", "he", "q1s"):
            out[f"{op}_alone_{ch}"] = alone[ch]
            out[f"{op}_given_{ch}"] = given[ch]
        mags["joint"].append(abs(out[f"{op}_joint_pct"]))
        mags["dn"].append(abs(out[f"{op}_joint_dn_pct"]))
        mags["sum"].append(abs(out[f"{op}_comb_pct"]))
    he_given = [abs(out[f"{op}_given_he"]) for op in mod.HEADLINE_OPS]
    out.update({
        "he_given_min": min(he_given),
        "he_given_max": max(he_given),
        "joint_pct_min": min(mags["joint"]),
        "joint_pct_max": max(mags["joint"]),
        "comb_worst_pct2": max(mags["sum"]),
    })
    out["findings"] = {
        "joint_pct_lo": min(mags["joint"]),
        "joint_pct_hi": max(mags["joint"]),
        "joint_dn_pct_lo": min(mags["dn"]),
        "joint_dn_pct_hi": max(mags["dn"]),
        "sum_pct_lo": min(mags["sum"]),
        "structural_band_pct": max(mags["sum"]),
    }
    return out


def _tolerance(printed: str, reference: float) -> float:
    """Half a unit in the last printed place, plus 1e-8 relative."""
    dp = len(printed.split(".")[1]) if "." in printed else 0
    return 0.5 * 10.0**-dp + 1e-8 * abs(reference)


def joint_failures(published: dict, oracle: dict) -> list[str]:
    """Each oracle figure that the published strings do not carry within the printed precision; a figure
    the oracle names and the published set lacks is a failure too."""
    bad = []
    for key, ref in oracle.items():
        if key not in published:
            bad.append(f"{key}: not published")
            continue
        if not abs(float(published[key]) - ref) <= _tolerance(published[key], ref):
            bad.append(f"{key}: published {published[key]} but the exact oracle gives {ref:.9g}")
    return bad


def _manifest_values(path) -> dict:
    return {e["id"]: e["value"] for e in json.loads(Path(path).read_text(encoding="utf-8"))["entries"]}


def test_published_joint_response_is_the_exact_oracle_s():
    """MATERIALITY_MANIFEST's section-5 figures (the joint response, its density-normalized reading, the sum
    of the brackets, each channel alone and with the other two on) and FINDINGS_MANIFEST's section-2 extremes
    are the exact oracle's to half a unit in the printed place (plus 1e-8 relative)."""
    oracle = joint_oracle(load_rates())
    findings = oracle.pop("findings")
    assert joint_failures(_manifest_values(MANIFEST), oracle) == []
    assert joint_failures(_manifest_values(REPO_ROOT / "FINDINGS_MANIFEST.json"), findings) == []
    # the settings each document prints are the ones the oracle ran (c_He = 1e-3, q_1s = 1.0)
    for path in (MANIFEST, REPO_ROOT / "FINDINGS_MANIFEST.json"):
        v = _manifest_values(path)
        assert (float(v["joint_c_he"]), float(v["joint_q1s"])) == (1e-3, 1.0), path


def test_joint_response_drill_refuses_the_sum_and_the_alone_bracket():
    """Drill: the sum's % printed as the joint %, or 3He's alone bracket printed as its value with the other
    two on, is refused, naming the figure."""
    oracle = joint_oracle(load_rates())
    oracle.pop("findings")
    published = _manifest_values(MANIFEST)
    for op in _load_script().HEADLINE_OPS:
        for key, src in ((f"{op}_joint_pct", f"{op}_comb_pct"), (f"{op}_given_he", f"{op}_alone_he")):
            drilled = dict(published)
            assert drilled[key] != drilled[src], (key, src)
            drilled[key] = drilled[src]
            failures = joint_failures(drilled, oracle)
            assert failures and all(f.startswith(f"{key}:") for f in failures), (key, failures)


def test_joint_is_not_the_sum_and_every_channel_lowers_it():
    """The interaction the joint response exists to show: at every headline operating point the joint
    response differs from the sum of the brackets by more than 1 X_mu, and each channel, with the other two
    on, lowers X_mu. The sum and the joint runs take one setting."""
    mod = _load_script()
    rates = load_rates()
    jr = mod.joint_response(rates)
    comb = mod.combined_band(mod.he_brackets(rates), mod.q1s_brackets(rates), mod.tt_brackets(rates))
    for op in mod.HEADLINE_OPS:
        assert abs(jr[op]["delta"] - comb[op]["combined"]) > 1.0, op
        assert all(v < 0.0 for v in jr[op]["given"].values()), (op, jr[op]["given"])
    assert (mod.JOINT_C_HE, mod.JOINT_Q1S) == (1e-3, 1.0)
    assert mod.JOINT_C_HE in mod.C_HE_LEVELS and mod.JOINT_Q1S in mod.Q1S_LEVELS


def conservation_failures(sols: dict) -> list[str]:
    """Every operating point whose solve does not conserve probability, two-sided:
    ``conservation_residual`` is signed, so a one-sided bound would admit any negative error."""
    from openmucf import cycle

    return [
        f"{op}: residual {cycle.conservation_residual(s):.3e}"
        for op, s in sols.items()
        if not abs(cycle.conservation_residual(s)) < 1e-12
    ]


def test_joint_solves_conserve_probability():
    jr = _load_script().joint_response(load_rates())
    assert conservation_failures({op: r["sol"] for op, r in jr.items()}) == []


def test_conservation_drill_refuses_an_omitted_loss_accumulator():
    """Drill: each populated loss accumulator (y[6] ttmu, y[7] 3He) zeroed in the final state is refused at
    every headline operating point."""
    import types

    jr = _load_script().joint_response(load_rates())
    for k in (6, 7):
        drilled = {}
        for op, r in jr.items():
            assert float(r["sol"].ys[-1, k]) > 1e-3, (op, k)  # the drill removes a real loss
            drilled[op] = types.SimpleNamespace(ys=r["sol"].ys.at[-1, k].set(0.0))
        assert len(conservation_failures(drilled)) == len(jr), k


def _in_section(doc: str, section: int, old: str, new: str) -> str:
    """Replace the first ``old`` inside section N only."""
    import re

    m = re.search(rf"\n## {section}\. [^\n]*\n(?:(?!\n## )[\s\S])*", doc)
    body = m.group(0)
    assert old in body, (section, old)
    return doc[: m.start()] + body.replace(old, new, 1) + doc[m.end() :]


def test_manifest_values_are_anchored_to_their_own_table(tmp_path):
    """Drills: a value corrupted inside its own section only is named, even where the same string stands in
    another table of the page: section 4's q_1s = 1.0 bracket and section 2's channels-OFF value (each
    repeated in a later table), and both 3He cells of section 5's second table."""
    doc = DOC.read_bytes().decode("utf-8").replace("\r\n", "\n")
    v = _manifest_values(MANIFEST)
    assert v["OP-D_q10_bracket"] == v["OP-D_alone_q1s"]  # section 5 repeats the section-4 bracket
    assert v["OP-D_bracket_c1em03"] == v["OP-D_alone_he"]  # and the section-2 one
    assert doc.count(f"| {v['OP-B_off']} |") >= 2  # section 2's table and section 5's first table
    manifest = MANIFEST.read_text(encoding="utf-8")
    drills = ((4, "OP-D_q10_bracket"), (2, "OP-B_off"), (5, "OP-D_alone_he"), (5, "OP-D_given_he"))
    for section, key in drills:
        value = v[key]
        nudged = value[:-1] + str((int(value[-1]) + 1) % 10)
        drilled = _in_section(doc, section, f" {value} |", f" {nudged} |")
        assert drilled != doc, key
        (tmp_path / "MATERIALITY.md").write_text(drilled, encoding="utf-8")
        (tmp_path / "MATERIALITY_MANIFEST.json").write_text(manifest, encoding="utf-8")
        failures = provenance.check_manifest(tmp_path / "MATERIALITY_MANIFEST.json", repo_root=tmp_path)
        assert any(f.startswith(f"{key}:") for f in failures), (key, failures)
