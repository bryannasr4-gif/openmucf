"""Generate MATERIALITY.md + MATERIALITY_MANIFEST.json (deterministic). Run from repo root:

    python scripts/generate_materiality.py

Content (WAVE1_EXECUTION_SPEC sec.6, WS-M): one-at-a-time absorbing-loss-channel toggles at fixed
operating points, reported as ONE-SIDED "structural sensitivity brackets" (X_mu^with - X_mu^without)
beside the section-2 forward-UQ CI width for scale. These are NOT error bars and are never combined
into any likelihood or CI (pre-registered combination rule: side-by-side only).

Two channels, per the WS-N network:
  * He-3 scavenging (dmu + 3He): LIVE (ledger `lambda_dhe3`=1.92e8 s^-1, Fotev et al. 2020). Toggled at
    two illustrative static helium fractions c_He in {1e-4, 1e-3}.
  * ttmu side-branch: BLOCKED. The ledger `lambda_ttmu` row shipped the WS-N sec.3.3 machine-representable
    fallback (value 0.0, notes beginning `blocked:`) because no open source pins its x-phi-x-c_t
    normalization. This generator DETECTS that marker and renders the ttmu rows "blocked -- pending
    acquisition of <the document named in the row notes>", NEVER a (misleading) zero bracket.

Everything here runs the v1 cycle model with the loss channel as the ONLY toggle, so "channels OFF" is
the byte-exact v1 engine and the bracket isolates the single channel's structural effect. Fully
deterministic (no MCMC/sampling), so `make audit` byte-diffs MATERIALITY.md + MATERIALITY_MANIFEST.json.

The computation lives in importable helpers (no side effects on import); file I/O + printing are guarded
behind ``main()`` so the test suite can import and assert on the operating-point/channel definitions
without regenerating the doc (mirrors scripts/generate_calibration.py).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from openmucf import cycle, provenance
from openmucf.rates import RATES_CSV, load_rates

# --------------------------------------------------------------------------- DECIDED spec parameters
# Operating points (WAVE1 sec.6.2, deviation D12). OP-A is anchor-adjacent: kept IN the table for
# completeness (accounting.md exists) but flagged and EXCLUDED from every headline/summary sentence.
# (phi, T [K], c_t)
OPERATING_POINTS: dict[str, tuple[float, float, float]] = {
    "OP-A": (1.25, 300.0, 0.5),  # Kou-Chen-like, anchor-adjacent (NON-HEADLINE)
    "OP-B": (1.2, 800.0, 0.5),  # high-T
    "OP-C": (2.0, 150.0, 0.5),  # MuFusE mid
    "OP-D": (2.4, 100.0, 0.5),  # MuFusE peak
}
ANCHOR_ADJACENT = "OP-A"  # excluded from headline sentences (D12)
HEADLINE_OPS = ("OP-B", "OP-C", "OP-D")
# Two illustrative tritium-decay-accumulation levels (static per-run helium fraction).
C_HE_LEVELS: tuple[float, ...] = (1e-4, 1e-3)
TRITIUM_HALFLIFE_YR = 12.32  # 3H beta decay -> 3He ingrowth driver (12.32 yr, standard)
# d-recapture / q_1s routing levels (contested cascade ground-state fraction). f_d = (1 - c_t) * q_1s.
Q1S_LEVELS: tuple[float, ...] = (0.4, 0.7, 1.0)

FINDINGS_MANIFEST = "FINDINGS_MANIFEST.json"


def _che_tag(c_he: float) -> str:
    """Filesystem/id-safe tag for a helium fraction, e.g. 1e-4 -> 'c1em4'."""
    return "c" + f"{c_he:.0e}".replace("-", "m").replace("+", "p").replace(".", "")


def _che_label(c_he: float) -> str:
    """Human label, e.g. 1e-4 -> '1e-4'."""
    return f"{c_he:.0e}"


def op_label(op: str) -> str:
    """ASCII table label for an operating point, e.g. 'OP-B (phi=1.2, T=800 K, c_t=0.5)'."""
    phi, T, c_t = OPERATING_POINTS[op]
    label = f"{op} (phi={phi:g}, T={T:g} K, c_t={c_t:g})"
    if op == ANCHOR_ADJACENT:
        label += " -- anchor-adjacent"
    return label


def he_brackets(rates) -> dict:
    """Compute the He-3 (dmu + 3He scavenging) structural brackets at every operating point.

    Returns ``{op: {"off": X_off, che_level: {"with": X_on, "bracket": X_on - X_off}, ...}}``.
    "off" is the channels-OFF v1 engine (byte-exact v1); "with" toggles ONLY the He-3 channel on at the
    given static c_He (the ttmu row is inert at 0.0, so include_loss_channels engages He-3 only). The
    bracket is one-sided and negative (an absorbing loss can only reduce X_mu).
    """
    out: dict = {}
    for op, (phi, T, c_t) in OPERATING_POINTS.items():
        x_off = float(cycle.fusions_per_muon_from_conditions(rates, T, phi, c_t))
        rec = {"off": x_off}
        for c_he in C_HE_LEVELS:
            x_on = float(
                cycle.fusions_per_muon_from_conditions(
                    rates, T, phi, c_t, include_loss_channels=True, c_he=c_he
                )
            )
            rec[c_he] = {"with": x_on, "bracket": x_on - x_off}
        out[op] = rec
    return out


def _q1s_tag(q: float) -> str:
    """Id-safe tag for a q_1s level, e.g. 0.4 -> 'q04', 1.0 -> 'q10'."""
    return f"q{int(round(q * 10)):02d}"


def q1s_brackets(rates) -> dict:
    """Compute the d-recapture / q_1s routing brackets at every operating point.

    Returns ``{op: {"off": X_off, q_1s: {"with": X_on, "bracket": X_on - X_off}, ...}}``. "off" is the
    recapture-OFF v1 engine (byte-exact v1); "with" routes a fraction ``f_d = (1 - c_t) * q_1s`` of the
    surviving-sticking flux through the dmu pool (one extra transfer that races decay), so the bracket is
    one-sided and negative. This is a re-routing, NOT an absorbing loss (the muon is not removed).
    """
    out: dict = {}
    for op, (phi, T, c_t) in OPERATING_POINTS.items():
        x_off = float(cycle.fusions_per_muon_from_conditions(rates, T, phi, c_t))
        rec = {"off": x_off}
        for q in Q1S_LEVELS:
            x_on = float(cycle.fusions_per_muon_from_conditions(rates, T, phi, c_t, q_1s=q))
            rec[q] = {"with": x_on, "bracket": x_on - x_off}
        out[op] = rec
    return out


def combined_band(he_br: dict, q1s_br: dict) -> dict:
    """Combined one-sided downward structural band at the HEADLINE operating points: the sum of the two
    live channels (3He scavenging at c_He=1e-3 + d-recapture at q_1s=1.0). The ttmu side-branch stays
    blocked (un-pinned), so the true band can only be MORE negative than this live sum.

    Returns ``{op: {"he": <3He bracket>, "q1s": <q_1s bracket>, "combined": <sum>, "pct": <% of X_off>}}``.
    """
    out: dict = {}
    for op in HEADLINE_OPS:
        x_off = he_br[op]["off"]
        he_b = he_br[op][1e-3]["bracket"]  # 3He at the larger illustrative helium level
        q_b = q1s_br[op][1.0]["bracket"]  # d-recapture at the maximal q_1s
        combined = he_b + q_b
        out[op] = {"he": he_b, "q1s": q_b, "combined": combined, "pct": 100.0 * combined / x_off}
    return out


def tt_blocked_status(rates) -> tuple[bool, str]:
    """Detect the WS-N sec.3.3 blocked-fallback marker on the ttmu formation-rate row.

    Returns ``(is_blocked, document_name)``. ``is_blocked`` is True when the ``lambda_ttmu`` ledger row
    ships value 0.0 with notes beginning ``blocked:`` (the machine-representable fallback). The document
    name is parsed from the row's own notes ("pending acquisition of <doc>"), so the rendered message
    names the blocking DOCUMENT (public-doc rule), never a private tracker or a zero bracket.
    """
    row = rates.get("lambda_ttmu")
    if row is None:
        return (False, "")
    is_blocked = row.notes.startswith("blocked:") and row.value == 0.0
    doc = "the primary ttmu tt-fusion tables"
    m = re.search(r"pending acquisition of (.+?\))", row.notes)
    if m:
        doc = m.group(1).strip()
    return (is_blocked, doc)


def read_forward_uq_ci(manifest_path=FINDINGS_MANIFEST) -> tuple[int, int]:
    """Read the section-2 forward-UQ 95% CI on X_mu (lo, hi) from the committed FINDINGS manifest.

    FINDINGS_MANIFEST.json is deterministic and byte-stable across environments, so this is an
    order-independent read (the CI is a fixed scale reference, not recomputed here).
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    vals = {e["id"]: e["value"] for e in manifest["entries"]}
    return (int(vals["xmu_ci_lo"]), int(vals["xmu_ci_hi"]))


def ingrowth_pct_per_month(halflife_yr: float = TRITIUM_HALFLIFE_YR) -> float:
    """Fractional 3He ingrowth per month from tritium decay: ln2/halflife, expressed as %/month."""
    return 100.0 * (math.log(2.0) / halflife_yr) / 12.0


def build_headline(rates) -> dict:
    """Assemble the single-source-of-truth formatted strings shared by the doc and the manifest."""
    br = he_brackets(rates)
    q1s_br = q1s_brackets(rates)
    comb = combined_band(br, q1s_br)
    ci_lo, ci_hi = read_forward_uq_ci()
    is_blocked, tt_doc = tt_blocked_status(rates)

    H: dict[str, str] = {}
    for op in OPERATING_POINTS:
        H[f"{op}_off"] = f"{br[op]['off']:.3f}"
        for c_he in C_HE_LEVELS:
            tag = _che_tag(c_he)
            H[f"{op}_with_{tag}"] = f"{br[op][c_he]['with']:.3f}"
            H[f"{op}_bracket_{tag}"] = f"{br[op][c_he]['bracket']:+.3f}"
        for q in Q1S_LEVELS:
            qt = _q1s_tag(q)
            H[f"{op}_{qt}_with"] = f"{q1s_br[op][q]['with']:.3f}"
            H[f"{op}_{qt}_bracket"] = f"{q1s_br[op][q]['bracket']:+.3f}"
    for op in HEADLINE_OPS:
        # 3He shown at 2 dp in the combined table (a summary rounding, distinct from the 3-dp section-2
        # figure) so the combined row never re-uses a section-2 bracket string verbatim; q_1s and the
        # combined total keep full 3-dp precision.
        H[f"{op}_comb_he2"] = f"{comb[op]['he']:+.2f}"
        H[f"{op}_comb_q1s"] = f"{comb[op]['q1s']:+.3f}"
        H[f"{op}_comb_total"] = f"{comb[op]['combined']:+.3f}"
        H[f"{op}_comb_pct"] = f"{comb[op]['pct']:+.2f}"
    # the largest combined downward band across the headline operating points (drives the summary claim).
    worst_op = min(HEADLINE_OPS, key=lambda op: comb[op]["pct"])
    H["comb_worst_pct"] = f"{abs(comb[worst_op]['pct']):.1f}"
    H["comb_worst_op"] = worst_op
    H["ci_lo"] = f"{ci_lo}"
    H["ci_hi"] = f"{ci_hi}"
    H["ci_width"] = f"{ci_hi - ci_lo}"
    H["ingrowth_pct_month"] = f"{ingrowth_pct_per_month():.2f}"
    H["tt_blocked"] = f"blocked -- pending acquisition of {tt_doc}" if is_blocked else "(ttmu channel LIVE)"
    H["_tt_is_blocked"] = "1" if is_blocked else "0"  # not a manifest entry; drives rendering only
    return H


# ------------------------------------------------------------------------------------- MATERIALITY.md
def _he_table(H: dict) -> str:
    head = (
        "| operating point | X_mu (channels OFF) | X_mu (3He, c_He=1e-4) | bracket "
        "| X_mu (3He, c_He=1e-3) | bracket |\n|---|---|---|---|---|---|\n"
    )
    lo, hi = _che_tag(1e-4), _che_tag(1e-3)
    rows = []
    for op in OPERATING_POINTS:
        rows.append(
            f"| {op_label(op)} | {H[f'{op}_off']} | {H[f'{op}_with_{lo}']} | {H[f'{op}_bracket_{lo}']} "
            f"| {H[f'{op}_with_{hi}']} | {H[f'{op}_bracket_{hi}']} |"
        )
    return head + "\n".join(rows)


def _tt_table() -> str:
    head = "| operating point | ttmu bracket |\n|---|---|\n"
    rows = [f"| {op_label(op)} | blocked -- pending acquisition |" for op in OPERATING_POINTS]
    return head + "\n".join(rows)


def _q1s_table(H: dict) -> str:
    head = (
        "| operating point | X_mu (recapture OFF) | X_mu (q_1s=0.4) | bracket | X_mu (q_1s=0.7) "
        "| bracket | X_mu (q_1s=1.0) | bracket |\n|---|---|---|---|---|---|---|---|\n"
    )
    q04, q07, q10 = _q1s_tag(0.4), _q1s_tag(0.7), _q1s_tag(1.0)
    rows = []
    for op in OPERATING_POINTS:
        rows.append(
            f"| {op_label(op)} | {H[f'{op}_off']} | {H[f'{op}_{q04}_with']} | {H[f'{op}_{q04}_bracket']} "
            f"| {H[f'{op}_{q07}_with']} | {H[f'{op}_{q07}_bracket']} "
            f"| {H[f'{op}_{q10}_with']} | {H[f'{op}_{q10}_bracket']} |"
        )
    return head + "\n".join(rows)


def _combined_table(H: dict) -> str:
    head = (
        "| operating point | d-recapture (q_1s=1.0) | 3He (c_He=1e-3) | ttmu | combined (live) "
        "| % of X_mu(OFF) |\n|---|---|---|---|---|---|\n"
    )
    rows = []
    for op in HEADLINE_OPS:
        rows.append(
            f"| {op_label(op)} | {H[f'{op}_comb_q1s']} | {H[f'{op}_comb_he2']} | blocked "
            f"| {H[f'{op}_comb_total']} | {H[f'{op}_comb_pct']}% |"
        )
    return head + "\n".join(rows)


def build_markdown(H: dict) -> str:
    return f"""# MATERIALITY.md -- structural sensitivity brackets (auto-generated by `scripts/generate_materiality.py`)

> Structural sensitivity brackets: one-sided, one-at-a-time channel toggles. These are NOT error bars \
and are never combined into the parametric CI; the anchor-condition rows are re-attribution-constrained \
(see docs/accounting.md).

## 1. What this is (and the combination rule)
Each bracket is `X_mu^with - X_mu^without` for ONE absorbing loss channel toggled on at a fixed operating
point, with the v1 cycle model otherwise unchanged ("without" == the byte-exact v1 engine). They are
**structural sensitivity brackets**, not a model-form error budget and not error bars: the pre-registered
combination rule is **side-by-side only** -- a bracket is NEVER convolved into any likelihood, posterior,
or the section-2 forward-UQ CI. A loss channel can only remove muons, so every live bracket is one-sided
(<= 0).

Operating points (fixed conditions; c_t=0.5 throughout):

- **OP-A** (phi=1.25, T=300 K) -- Kou-Chen-like, **anchor-adjacent**: kept in the tables for completeness
  but EXCLUDED from every headline sentence and interpreted only via docs/accounting.md (the measured
  effective parameters already contain deferred physics as it occurred at the anchors -- re-attribution
  applies here, so an anchor-adjacent bracket is not a clean structural addition).
- **OP-B** (phi=1.2, T=800 K) -- high-T; **OP-C** (phi=2.0, T=150 K) -- MuFusE mid;
  **OP-D** (phi=2.4, T=100 K) -- MuFusE peak.

Channels covered: the two absorbing loss channels in the v1.1 network (ttmu side-branch; 3He scavenging),
plus the per-cycle d-recapture / q_1s routing (section 4 -- a re-routing, not an absorbing loss: the freed
muon detours through the dmu pool and races decay one extra transfer). The ddmu / d-d branch and the
epithermal-formation (eta) enhancement remain OUT of scope here: the ddmu channel does not exist in the
engine yet (documented -5..-15% headroom, see docs/accounting.md), and eta is already reported as its own
structural bracket in FINDINGS.md section 1c (one-home rule I5).

## 2. 3He scavenging channel (LIVE): dmu + 3He
The dmu + 3He scavenging rate `lambda_dhe3` = 1.92e8 s^-1 (Fotev et al. 2020, arXiv:2001.09927; open) is
live in the ledger; it acts on the dmu pool and is engaged at a STATIC per-run helium fraction c_He
(never time-evolved inside the single-muon ODE). 3He grows in from tritium beta decay at
ln(2) / {TRITIUM_HALFLIFE_YR:g} yr = {100.0 * math.log(2.0) / TRITIUM_HALFLIFE_YR:.4f}%/yr, i.e.
**{H["ingrowth_pct_month"]}%/month** of the tritium inventory; the two levels below (c_He = 1e-4 and 1e-3,
~0.01% and ~0.1% helium) are illustrative low accumulations bracketing the first days-to-weeks of a sealed
DT fill -- both well under one month's ingrowth.

{_he_table(H)}

For scale: the FINDINGS.md section-2 forward-UQ 95% CI on X_mu (liquid box) spans
[{H["ci_lo"]}, {H["ci_hi"]}] -- a width of **{H["ci_width"]}** X_mu units. Every 3He bracket above is
<= ~0.18 in magnitude, i.e. under ~0.6% of that parametric CI width: at present helium levels the 3He
channel is a small, one-sided structural correction, reported beside the CI and never folded into it.

## 3. ttmu side-branch channel (BLOCKED)
**ttmu side-branch: {H["tt_blocked"]}.** The ttmu formation-rate row `lambda_ttmu` ships the
machine-representable fallback value 0.0 (channel inert) because no open source pins its density/c_t
normalization; the companion loss fraction `omega_tt`=0.14 is recorded for the re-attribution once the
formation rate is acquired. Toggling `include_loss_channels` therefore leaves X_mu unchanged for this
channel at every operating point, but a **zero bracket would be misleading** -- it would assert the ttmu
channel is negligible, which is exactly what is NOT yet established. The bracket is reported as blocked,
not zero:

{_tt_table()}

## 4. d-recapture / q_1s routing bracket (LIVE)
The freed muon that survives sticking is, in v1, recycled straight back to the tmu pool (3/4 F=1, 1/4 F=0).
The per-cycle **d-recapture** correction routes a fraction `f_d = (1 - c_t) * q_1s` of that surviving flux
through the **dmu** pool instead (the muon recaptures on deuterium and must transfer again, racing decay
one extra step); `q_1s` is the contested cascade ground-state fraction. This is a re-routing, not an
absorbing loss -- but it lowers X_mu, so the bracket is one-sided and negative. It is now explicit in
`cycle.py` (`params_from_conditions(q_1s=...)`); the `_CALIB` unfolding that would re-attribute this leg
against the 300 K anchor stays acquisition-gated (docs/accounting.md).

{_q1s_table(H)}

At the anchor-condition (300 K, 1.2 phi, c_t=0.5) MODEL_SPEC section-8 estimated this leg at -6% .. -14%
for q_1s=0.4..1.0; the explicit computation confirms it to rounding (-5.96% .. -13.68%; see the MODEL_SPEC
dated note). Above the compressed-gas onset the formation model is a placeholder (RED tier), so OP-C/OP-D
q_1s brackets share the off-anchor caveat of every off-anchor number here.

## 5. Combined structural band (headline operating points)
Summing the two LIVE channels -- 3He scavenging (c_He=1e-3, section 2) and d-recapture (q_1s=1.0,
section 4) -- with the ttmu side-branch still **blocked** (section 3, un-pinned pending acquisition), the
one-sided downward structural band across the headline operating points is (3He shown at 2 dp; the
combined total is the full-precision sum):

{_combined_table(H)}

The band is dominated by d-recapture at the maximal q_1s; 3He is a ~0.1% correction at these helium levels.
The largest combined live band is **~{H["comb_worst_pct"]}%** of X_mu (at {H["comb_worst_op"]}), consistent
with the ~10-15% one-sided structural headroom named in MODEL_SPEC section 8 -- and it is a LOWER bound,
because the ttmu side-branch (blocked) would add further downward. These brackets are reported beside the
parametric CI, never convolved into it.

## 6. Summary (headline operating points OP-B / OP-C / OP-D only)
Across the headline operating points OP-B (high-T), OP-C (MuFusE mid) and OP-D (MuFusE peak), the live
absorbing-loss channel (3He scavenging) contributes a one-sided structural bracket no larger than
about -0.18 X_mu units at 0.1% helium -- under ~0.6% of the section-2 forward-UQ CI width
({H["ci_width"]}), and smaller still at 0.01% helium. The per-cycle d-recapture routing (section 4) is the
dominant deferred correction: up to **~{H["comb_worst_pct"]}%** of X_mu combined with 3He at
{H["comb_worst_op"]} (section 5), one-sided downward. The ttmu side-branch bracket is **blocked -- pending
acquisition** and is deliberately NOT rendered as a zero. The anchor-adjacent OP-A row is listed for
completeness only and carries no headline claim (docs/accounting.md). No bracket on this page is combined
into any CI or likelihood.
"""


# ------------------------------------------------------------- machine-checkable provenance (MANIFEST)
def build_manifest_entries(H: dict) -> list:
    def _entry(entry_id, pattern, source_type="derivation", source="scripts/generate_materiality.py"):
        return provenance.ManifestEntry(
            id=entry_id,
            value=H[entry_id],
            pattern=pattern,
            source_type=source_type,
            source=source,
            doc="MATERIALITY.md",
        )

    entries = []
    lo, hi = _che_tag(1e-4), _che_tag(1e-3)
    for op in OPERATING_POINTS:
        # every value on an operating point's He-table row is anchored to that row via the "OP-x" tag.
        entries.append(_entry(f"{op}_off", rf"{op}[^\n]*{re.escape(H[f'{op}_off'])}"))
        for tag in (lo, hi):
            entries.append(_entry(f"{op}_with_{tag}", rf"{op}[^\n]*{re.escape(H[f'{op}_with_{tag}'])}"))
            entries.append(
                _entry(f"{op}_bracket_{tag}", rf"{op}[^\n]*{re.escape(H[f'{op}_bracket_{tag}'])}")
            )
        for q in Q1S_LEVELS:  # d-recapture / q_1s bracket (section 4) values
            qt = _q1s_tag(q)
            entries.append(_entry(f"{op}_{qt}_with", rf"{op}[^\n]*{re.escape(H[f'{op}_{qt}_with'])}"))
            entries.append(_entry(f"{op}_{qt}_bracket", rf"{op}[^\n]*{re.escape(H[f'{op}_{qt}_bracket'])}"))
    for op in HEADLINE_OPS:  # combined structural band (section 5)
        entries.append(_entry(f"{op}_comb_q1s", rf"{op}[^\n]*{re.escape(H[f'{op}_comb_q1s'])}"))
        entries.append(_entry(f"{op}_comb_total", rf"{op}[^\n]*{re.escape(H[f'{op}_comb_total'])}"))
        entries.append(_entry(f"{op}_comb_pct", rf"{op}[^\n]*{re.escape(H[f'{op}_comb_pct'])}%"))
    entries += [
        _entry("ci_lo", rf"spans\s*\[{re.escape(H['ci_lo'])}, {re.escape(H['ci_hi'])}\]"),
        _entry("ci_hi", rf"spans\s*\[{re.escape(H['ci_lo'])}, {re.escape(H['ci_hi'])}\]"),
        _entry("ci_width", rf"width of \*\*{re.escape(H['ci_width'])}\*\* X_mu units"),
        _entry("ingrowth_pct_month", rf"\*\*{re.escape(H['ingrowth_pct_month'])}%/month\*\*"),
        _entry(
            "tt_blocked",
            rf"ttmu side-branch: {re.escape(H['tt_blocked'])}",
            source_type="ledger_row",
            source="openmucf/data/rates.csv:lambda_ttmu",
        ),
    ]
    return entries


def main() -> None:
    rates = load_rates()
    H = build_headline(rates)
    Path("MATERIALITY.md").write_text(build_markdown(H), encoding="utf-8")
    entries = build_manifest_entries(H)
    inputs = {
        "rates_csv_sha256": provenance.file_sha256(RATES_CSV),
        "findings_manifest_sha256": provenance.file_sha256(FINDINGS_MANIFEST),
    }
    provenance.write_manifest(
        "MATERIALITY_MANIFEST.json", entries, inputs, generated_by="scripts/generate_materiality.py"
    )
    hi_tag = _che_tag(1e-3)
    br_summary = {op: H[f"{op}_bracket_{hi_tag}"] for op in OPERATING_POINTS}
    print(f"wrote MATERIALITY.md + MATERIALITY_MANIFEST.json ({len(entries)} entries)")
    print(f"3He brackets @ c_He=1e-3: {br_summary}")
    print(f"ttmu channel: {H['tt_blocked']}")
    print(f"forward-UQ CI width for scale = {H['ci_width']} (X_mu [{H['ci_lo']}, {H['ci_hi']}])")


if __name__ == "__main__":
    main()
