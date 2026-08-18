"""Two guards over the *prose* that surrounds the muon-cost ledger.

Neither guard reads the ledger. They bind the sentences written *about* it, which no manifest,
byte-diff or `provenance --check` can see: a manifest pins a value against the document that renders
it, so a claim typed into a docstring, a comment or a hand-written CHANGELOG line can be false while
every existing gate stays green.

- **G1 :func:`test_quantified_claims_registered`** enumerates the claims its two regexes match over
  the named paths and requires each to be registered as either exercised by a named test or
  explicitly deferred with a reason. It cannot tell truth from falsehood, and it is a net with
  stated holes rather than a proof of coverage -- see the holes listed on the test itself. What it
  does buy is that a universal matching that shape cannot enter those paths without a registry diff.
- **G2 :func:`test_prose_arithmetic_recomputes`** recomputes arithmetic that prose states in full.

Both keep their bookkeeping in TSV files beside this one, keyed by the SHA-1 of the
whitespace-normalized claim line -- never by line number, which churns on every commit.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------------------
# G1 -- quantified claims about the ledger
# --------------------------------------------------------------------------------------------------

#: The prose-bearing paths that carry claims about the muon-cost ledger. Everything the
#: cost-basis work writes about the ledger lands in one of these; nothing else is in scope, and a new
#: prose home for ledger claims must be added here deliberately.
CLAIM_PATHS = (
    "openmucf/mucost.py",
    "scripts/generate_mucost.py",
    "tests/test_mucost.py",
    "MUON_COST.md",
    "CHANGELOG.md",
    "README.md",
    "paper/paper.md",
    "ADOPTERS.md",
    "openmucf/data/references.bib",
    "openmucf/data/muon_cost.schema.json",
    # This file. A guard that exempts itself is not a guard: both of the false measured numbers this
    # module shipped in its first revision -- a file count and a claim count, each stale the moment
    # the commit that stated them landed -- lived here, in the one prose-bearing file the registry
    # did not cover.
    "tests/test_ledger_claims.py",
)

#: Genuine universals and uniqueness words only. Modals (`must`, `cannot`), ordinals and
#: `both`/`identical`/`unchanged` are deliberately OUT: they are not quantifiers, and including them
#: roughly doubles the surface without catching a known defect (measured when the guard was sized:
#: 528 raw hits against the 268 this pair found, and every target line was in the smaller set).
STRONG = re.compile(
    r"\b(every|all|each|none|never|always|only|sole|solely|exactly"
    r"|unique|uniquely|neither|any|entire|without\s+exception)\b",
    re.IGNORECASE,
)

#: Nouns that make a sentence a claim about the ledger rather than about anything else. `bound` and
#: `cost` are load-bearing: without them two known false universals escape the net entirely.
LEDGER = re.compile(
    r"\b(row|rows|tier|tiers|cell|cells|anchor|anchors|source|sources|basis|bases"
    r"|numeraire|numeraires|stage|stages|ledger|entry|entries|chain|chains|headline"
    r"|manifest|bibkey|bibkeys|evidence_status|charge_basis|bound|bounds|cost|costs"
    r"|value|values|figure|figures|quotient|denominator|problem|problems|contract"
    r"|contracts|claim|claims|number|numbers)\b",
    re.IGNORECASE,
)

#: How many enumerated claims may still be `UNREVIEWED`. **One-sided and monotone NON-INCREASING**,
#: on the exact precedent of ``AUDIT_ESS_FLOOR``: raising it is a visible diff to this line and must
#: be argued for, lowering it needs no argument. These are pre-existing claims that the cost-basis
#: work did not write; a round of review has already shown such claims can be false, so this defers
#: them rather than dropping them.
LEDGER_CLAIMS_UNREVIEWED_CEILING = 92

REGISTRY = Path(__file__).with_name("ledger_claims_registry.tsv")
VALID_PREFIXES = ("EXERCISED:", "REGISTERED:")


def _normalize(line: str) -> str:
    return " ".join(line.split())


def claim_sha1(line: str) -> str:
    return hashlib.sha1(_normalize(line).encode("utf-8")).hexdigest()


def enumerate_claims() -> list[tuple[str, int, str, str]]:
    """(path, lineno, sha1, normalized text) for every claim line, in file then line order."""
    out: list[tuple[str, int, str, str]] = []
    for rel in CLAIM_PATHS:
        text = (REPO / rel).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if STRONG.search(line) and LEDGER.search(line):
                out.append((rel, lineno, claim_sha1(line), _normalize(line)))
    return out


def _read_registry() -> dict[str, tuple[str, str, str]]:
    """{sha1: (path, status, note)} -- the note is free text and is never parsed."""
    rows: dict[str, tuple[str, str, str]] = {}
    for raw in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        assert len(parts) == 4, f"registry row must have 4 tab-separated fields: {raw!r}"
        sha, path, status, note = parts
        assert sha not in rows, f"duplicate registry row for {sha}"
        rows[sha] = (path, status, note)
    return rows


def test_quantified_claims_registered():
    """Claims the enumerator matches are registered; nothing unreviewed beyond the ceiling.

    **The enumerator, stated verbatim -- this is the whole definition of what gets caught.** Over
    the paths in :data:`CLAIM_PATHS`, read **per line** (not per docstring: an AST pass over function
    and class docstrings misses the module docstring and every comment, which is where a real defect
    was found hiding). A line is a **claim** iff it matches BOTH of these, case-insensitively:

    - STRONG ``\\b(every|all|each|none|never|always|only|sole|solely|exactly|unique|uniquely|neither
      |any|entire|without\\s+exception)\\b``
    - LEDGER ``\\b(row|rows|tier|tiers|cell|cells|anchor|anchors|source|sources|basis|bases|numeraire
      |numeraires|stage|stages|ledger|entry|entries|chain|chains|headline|manifest|bibkey|bibkeys
      |evidence_status|charge_basis|bound|bounds|cost|costs|value|values|figure|figures|quotient
      |denominator|problem|problems|contract|contracts|claim|claims|number|numbers)\\b``

    Each claim is keyed by the SHA-1 of its whitespace-normalized text. The test fails when a claim is
    not in the registry, when a registry row is no longer enumerated (deleting a claim cannot silently
    keep its credit), and when the ``UNREVIEWED`` count exceeds
    :data:`LEDGER_CLAIMS_UNREVIEWED_CEILING`.

    **Known holes, stated rather than left to be found.** This catches an unreviewed universal only
    within the shape below; it is not a proof that none exists.

    1. **The key is the text, not (path, text).** Copying an already-registered sentence into a
       different watched file changes no hash, so it enters with no registry diff. Re-keying on
       ``(path, sha1)`` is the fix and is a separate change: it turns every duplicate into its own
       row and so moves the ceiling, which must not happen as a side effect.
    2. **A universal split across two lines escapes**, because the scan is per line -- the mechanism
       is stated above, and this is its consequence.
    3. **``no`` and ``nothing`` are absent from STRONG** while ``none`` is present. They are the
       commonest universal negatives; adding them was measured to pull in 65 further lines, most of
       them not universals at all (``no pinned value``, ``no eta_acc``), so it is a reviewed
       expansion rather than a one-word edit.
    4. **:data:`CLAIM_PATHS` is hand-maintained and nothing tests it for completeness.** Prose about
       the ledger outside those paths is invisible here -- including the free-text
       ``basis_as_published`` / ``derivation`` / ``notes`` columns of ``muon_cost.csv`` itself, which
       carry per-row basis claims and ship in the data package.

    **What this does NOT do:** it does not decide whether a claim is true. ``REGISTERED:`` records a
    judgement made by a person and its substance is not machine-checked -- an empty or worthless
    reason passes, so the review layer is exactly as strong as the reviewer. ``EXERCISED:`` names a
    test that fails when the claim is negated, and that naming is checked below only for
    resolvability, not for strength.
    """
    claims = enumerate_claims()
    registry = _read_registry()
    enumerated = {sha for _, _, sha, _ in claims}

    missing = [(p, n, s, t) for p, n, s, t in claims if s not in registry]
    detail = "\n".join(f"  {s}\t{p}:{n}\t{t}" for p, n, s, t in missing)
    assert not missing, f"unregistered ledger claim(s) -- add a row to {REGISTRY.name}:\n{detail}"

    stale = sorted(s for s in registry if s not in enumerated)
    detail = "\n".join(f"  {s}\t{registry[s][0]}\t{registry[s][2]}" for s in stale)
    assert not stale, f"registry rows no longer enumerated -- delete from {REGISTRY.name}:\n{detail}"

    bad = {s: st for s, (_, st, _) in registry.items() if st != "UNREVIEWED"
           and not st.startswith(VALID_PREFIXES)}
    assert not bad, f"status must be UNREVIEWED or start with {VALID_PREFIXES}: {bad}"

    # An EXERCISED row must name a test that exists. This does not prove the test is strong; it
    # removes the failure mode where the named node id was never real in the first place.
    for sha, (_, status, _) in sorted(registry.items()):
        if not status.startswith("EXERCISED:"):
            continue
        node = status[len("EXERCISED:"):].strip()
        file_part, _, func = node.partition("::")
        target = REPO / file_part
        assert target.is_file(), f"{sha}: EXERCISED names a missing file: {node}"
        assert func, f"{sha}: EXERCISED must name a test function: {node}"
        assert f"def {func}(" in target.read_text(encoding="utf-8"), (
            f"{sha}: EXERCISED names a test that does not exist: {node}"
        )

    unreviewed = sorted(s for s, (_, st, _) in registry.items() if st == "UNREVIEWED")
    assert len(unreviewed) <= LEDGER_CLAIMS_UNREVIEWED_CEILING, (
        f"{len(unreviewed)} UNREVIEWED claims exceeds the ceiling "
        f"{LEDGER_CLAIMS_UNREVIEWED_CEILING}; the ceiling is monotone non-increasing -- review the "
        f"new claims instead of raising it"
    )


# --------------------------------------------------------------------------------------------------
# G2 -- arithmetic written out in prose
# --------------------------------------------------------------------------------------------------

#: A number: interior commas allowed but never a trailing one (without that restriction the `6` of
#: `6, voltage_V` parses as a term); optional decimals; optional exponent. The two lookaheads stop the
#: engine settling for a prefix -- without them the `1` of `1e-9` reads as the subtraction `1 - 9`.
_NUM = r"\d(?:[\d,]*\d)?(?:\.\d+)?(?:[eE][-+]?\d+)?(?![eE][-+]?\d)(?![\d,])"
#: Unit words after an operand, skipped so that `3.61 GeV / 0.77 muons-per-beam-particle` parses.
#: Each repetition MUST begin with whitespace: without that the outer and inner `*` overlap and the
#: match time is exponential in the length of the word.
_UNIT = r"(?:[ \t]+[A-Za-z][A-Za-z0-9_%\u00b5\u00b7/-]*)*"
_OP = r"[*x\u00d7/+\u2212-]"

STATEMENT = re.compile(
    rf"({_NUM}){_UNIT}"
    rf"((?:[ \t]*{_OP}[ \t]*{_NUM}{_UNIT})+)"
    rf"[ \t]*=[ \t]*({_NUM})[ \t]*(%?)(?!%)"
)
_TERM = re.compile(rf"({_OP})[ \t]*({_NUM})")

#: Directories never descended. Pruning these and collecting `*.md`, `*.py`, `*.bib` was measured to
#: reproduce `git ls-files` on this tree exactly -- identical sets, no count quoted here because a
#: count in a comment goes stale on the next file added, and this comment cannot count itself. The
#: walk is over the WORKING TREE, not the index: an untracked `*.md` sitting in a checkout is scanned.
_PRUNE = {".git", "__pycache__", "node_modules", "build", "dist",
          ".pytest_cache", ".mypy_cache", ".ruff_cache"}

EXCEPTIONS = Path(__file__).with_name("prose_arithmetic_exceptions.tsv")


def _scanned_files() -> list[str]:
    out: list[str] = []
    for root, dirs, fnames in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in _PRUNE and not d.startswith(".venv") and not d.endswith(".egg-info")]
        for f in fnames:
            if f.endswith((".md", ".py", ".bib")):
                out.append(str(Path(root, f).relative_to(REPO)).replace("\\", "/"))
    return sorted(out)


def _to_float(tok: str) -> float:
    return float(tok.replace(",", ""))


def _evaluate(first: str, rest: str) -> float:
    """Left to right, with `*` `x` `/` binding tighter than `+` `-`. No ``eval``."""
    vals = [_to_float(first)]
    ops: list[str] = []
    for m in _TERM.finditer(rest):
        ops.append(m.group(1))
        vals.append(_to_float(m.group(2)))
    i = 0
    while i < len(ops):
        if ops[i] in "*x\u00d7/":
            vals[i:i + 2] = [vals[i] / vals[i + 1] if ops[i] == "/" else vals[i] * vals[i + 1]]
            del ops[i]
        else:
            i += 1
    total = vals[0]
    for op, v in zip(ops, vals[1:], strict=True):
        total = total + v if op == "+" else total - v
    return total


def scan_arithmetic() -> tuple[list[tuple[str, int, str, str, float, bool]], dict[str, str]]:
    """(every statement checked as (path, line, sha1, text, computed, ok), {sha1: report line}).

    The second element holds only the statements that do NOT recompute, keyed the same way the
    exceptions file is: by the SHA-1 of the whitespace-normalized line, never by line number.
    """
    checked: list[tuple[str, int, str, str, float, bool]] = []
    bad: dict[str, str] = {}
    for rel in _scanned_files():
        try:
            lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, 1):
            for m in STATEMENT.finditer(line):
                first, rest, result, pct = m.groups()
                tail = line[m.end():]
                if re.match(r"[ \t]*[-\u2013\u2014][ \t]*\d", tail):
                    continue  # `1.4%-6.1%` is a range, not a result
                if re.match(rf"{_OP}\d", tail):
                    continue  # `= 5/3` -- the result is a fraction, outside the matched form
                try:
                    got = _evaluate(first, rest)
                except ZeroDivisionError:
                    continue
                if pct:
                    got *= 100.0
                places = len(result.split(".")[1]) if "." in result else 0
                ok = round(got, places) == round(_to_float(result), places)
                sha = claim_sha1(line)
                checked.append((rel, lineno, sha, m.group(0), got, ok))
                if not ok:
                    bad[sha] = f"  {rel}:{lineno}  {m.group(0)!r} -> computed {got!r}"
    return checked, bad


def _read_exceptions() -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for raw in EXCEPTIONS.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        assert len(parts) == 4, f"exception row must have 4 tab-separated fields: {raw!r}"
        sha, path, measured, ruling = parts
        rows[sha] = (path, ruling)
    return rows


def test_prose_arithmetic_recomputes():
    """Arithmetic a sentence writes out in full must recompute, at the precision it prints.

    Scope, exactly: the *expression-then-result* form ``<num>[unit] <op> <num>[unit] ... = <num>[%]``,
    matched **per line**, over every ``*.md`` / ``*.py`` / ``*.bib`` file in the working tree outside
    the virtual environments. **This guard is a net with known holes, and the holes are listed here
    rather than left to be discovered** -- an undisclosed boundary would turn an open question into a
    false all-clear, which is worse than no guard.

    Statements it does NOT match (a wrong one written this way passes):

    1. **The reversed form**, result before expression: ``E_mu_GeV: float = 4.70  # 3.61 GeV beam /
       0.77 muons per beam particle``.
    2. **A statement wrapped across two lines**, because the scan is per line.
    3. **A negative result** -- :data:`_NUM` carries no sign, so ``4.85 - 178 -> -170`` is unmatchable.
    4. **A non-ASCII operator**, e.g. the interpunct in ``20.4 · 150 -> 3210``.
    5. **A unit glued to its operand** without a space, e.g. ``3.61GeV / 0.77 -> 9.99``.
    6. **A word between ``=`` and the result**, e.g. ``-> roughly 5.69``.
    7. Anything the two skip rules below drop: a result followed by ``-<digit>`` (read as a range) or
       immediately by an operator and a digit (read as a fraction such as ``-> 5/3``).

    Statements it may match and mis-report (a CORRECT one written this way is flagged):

    8. **Mixed units in one expression** -- units are skipped, not converted, so ``1 GeV + 500 MeV ->
       1.5 GeV`` recomputes to 501.
    9. **A percentage whose expression already multiplies by 100** -- the ``%`` branch scales by 100
       again, so ``1.3/12.5*100 -> 10.4%`` recomputes to 1040.

    A flag is therefore evidence to weigh, never an instruction to add an exception row. It
    checks arithmetic, not physics: a correctly-computed quotient of two wrong inputs passes.

    A statement that does not recompute fails the test. The exceptions file exists for a defect that
    is real, already ruled, and owned by other work -- never to make a red run green.
    """
    checked, bad = scan_arithmetic()
    assert checked, "the scanner found no arithmetic at all -- the matcher is broken, not the prose"

    exceptions = _read_exceptions()
    unexpected = sorted(set(bad) - set(exceptions))
    assert not unexpected, "prose arithmetic that does not recompute:\n" + "\n".join(
        bad[s] for s in unexpected
    )

    unused = sorted(set(exceptions) - set(bad))
    assert not unused, (
        "exception row(s) whose statement now recomputes (or no longer exists) -- delete them:\n"
        + "\n".join(f"  {s}\t{exceptions[s][0]}" for s in unused)
    )


if __name__ == "__main__":  # regenerate the registry skeleton: python tests/test_ledger_claims.py
    for _p, _n, _s, _t in enumerate_claims():
        print(f"{_s}\t{_p}\tUNREVIEWED\t{_t[:120]}")
