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
    ci_lo, ci_hi = read_forward_uq_ci()
    is_blocked, tt_doc = tt_blocked_status(rates)

    H: dict[str, str] = {}
    for op in OPERATING_POINTS:
        H[f"{op}_off"] = f"{br[op]['off']:.3f}"
        for c_he in C_HE_LEVELS:
            tag = _che_tag(c_he)
            H[f"{op}_with_{tag}"] = f"{br[op][c_he]['with']:.3f}"
            H[f"{op}_bracket_{tag}"] = f"{br[op][c_he]['bracket']:+.3f}"
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

Channels covered: the two absorbing loss channels in the v1.1 network (ttmu side-branch; 3He scavenging).
The ddmu / d-d branch and the epithermal-formation (eta) enhancement are OUT of scope here:
the ddmu channel does not exist in the engine yet (documented -5..-15% headroom, see docs/accounting.md),
and eta is already reported as its own structural bracket in FINDINGS.md section 1c (one-home rule I5).

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

## 4. Summary (headline operating points OP-B / OP-C / OP-D only)
Across the headline operating points OP-B (high-T), OP-C (MuFusE mid) and OP-D (MuFusE peak), the only
live absorbing-loss channel (3He scavenging) contributes a one-sided structural bracket no larger than
about -0.18 X_mu units at 0.1% helium -- under ~0.6% of the section-2 forward-UQ CI width
({H["ci_width"]}), and smaller still at 0.01% helium. The ttmu side-branch bracket is **blocked -- pending
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
