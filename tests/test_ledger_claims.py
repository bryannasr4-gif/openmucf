"""Two guards over the *prose* that surrounds the muon-cost ledger.

Neither guard reads the ledger. They bind the sentences written *about* it, which no manifest,
byte-diff or `provenance --check` can see: a manifest pins a value against the document that renders
it, so a claim typed into a docstring, a comment or a hand-written CHANGELOG line can be false while
every existing gate stays green.

- **G1 :func:`test_quantified_claims_registered`** enumerates the lines its two regexes match over
  the named paths and requires each to carry a registry row. A row records one of three statuses:
  ``EXERCISED:`` names a test, ``REGISTERED:`` records a human judgement, and ``UNREVIEWED`` records
  that neither has happened yet, capped by :data:`LEDGER_CLAIMS_UNREVIEWED_CEILING`. G1 decides
  nothing about truth. What it buys is that no ``(path, text)`` pair can enter or leave those paths
  without a registry diff.
- **G2 :func:`test_prose_arithmetic_recomputes`** recomputes arithmetic that prose states in full.

Both keep their bookkeeping in TSV files beside this one, keyed on the path and the SHA-1 of the
whitespace-normalized line -- never on line number, which churns on every commit.
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

#: The prose-bearing paths this guard reads. A new prose home for ledger claims must be added here
#: deliberately; one that is not listed is not read.
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
    # This file. A guard that exempts itself is not a guard: this module's own prose is watched on
    # the same terms as every other path here.
    "tests/test_ledger_claims.py",
)

#: Genuine universals and uniqueness words only. Modals (`must`, `cannot`), ordinals and
#: `both`/`identical`/`unchanged` are deliberately OUT: they are not quantifiers.
STRONG = re.compile(
    r"\b(every|all|each|none|never|always|only|sole|solely|exactly"
    r"|unique|uniquely|neither|any|entire|without\s+exception)\b",
    re.IGNORECASE,
)

#: Nouns that make a sentence a claim about the ledger rather than about anything else. `bound` and
#: `cost` are deliberately in: the ledger's basis universals are commonly phrased with them.
LEDGER = re.compile(
    r"\b(row|rows|tier|tiers|cell|cells|anchor|anchors|source|sources|basis|bases"
    r"|numeraire|numeraires|stage|stages|ledger|entry|entries|chain|chains|headline"
    r"|manifest|bibkey|bibkeys|evidence_status|charge_basis|bound|bounds|cost|costs"
    r"|value|values|figure|figures|quotient|denominator|problem|problems|contract"
    r"|contracts|claim|claims|number|numbers)\b",
    re.IGNORECASE,
)

#: How many matched lines may still be `UNREVIEWED`. **One-sided and monotone NON-INCREASING**, on
#: the exact precedent of ``AUDIT_ESS_FLOOR``: raising it is a visible diff to this line and must be
#: argued for, lowering it needs no argument. Review defers rather than drops: an unreviewed line
#: stays enumerated and capped instead of leaving the registry.
LEDGER_CLAIMS_UNREVIEWED_CEILING = 94

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


def _read_registry() -> dict[tuple[str, str], str]:
    """{(path, sha1): status}. Three tab-separated fields per row: sha1, path, status.

    The registry carries no copy of the matched line. That line lives in the file the path names and
    the enumerator prints it on failure; copying it here made this file a second home for text
    written somewhere else.
    """
    rows: dict[tuple[str, str], str] = {}
    for raw in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        assert len(parts) == 3, f"registry row must have 3 tab-separated fields: {raw!r}"
        sha, path, status = parts
        assert (path, sha) not in rows, f"duplicate registry row for {(path, sha)}"
        rows[(path, sha)] = status
    return rows


def test_quantified_claims_registered():
    """Lines the enumerator matches carry a registry row; nothing unreviewed beyond the ceiling.

    **The enumerator, stated verbatim -- this is the whole definition of what gets caught.** Over
    the paths in :data:`CLAIM_PATHS`, read **per line** (not per docstring: an AST pass over function
    and class docstrings reads neither the module docstring nor any comment). A line is a **claim**
    iff it matches BOTH of these, case-insensitively:

    - STRONG ``\\b(every|all|each|none|never|always|only|sole|solely|exactly|unique|uniquely|neither
      |any|entire|without\\s+exception)\\b``
    - LEDGER ``\\b(row|rows|tier|tiers|cell|cells|anchor|anchors|source|sources|basis|bases|numeraire
      |numeraires|stage|stages|ledger|entry|entries|chain|chains|headline|manifest|bibkey|bibkeys
      |evidence_status|charge_basis|bound|bounds|cost|costs|value|values|figure|figures|quotient
      |denominator|problem|problems|contract|contracts|claim|claims|number|numbers)\\b``

    Each match is keyed by its path and the SHA-1 of its whitespace-normalized text. The test fails
    when a match carries no row, when a row is no longer matched (deleting a claim cannot silently
    keep its credit), and when the ``UNREVIEWED`` count exceeds
    :data:`LEDGER_CLAIMS_UNREVIEWED_CEILING`.

    This checks exactly what those two patterns match, line by line, over exactly those paths; a
    universal in any other form, split across lines, or written anywhere else is unchecked. It does
    not decide whether a matched claim is true: ``REGISTERED:`` records a judgement made by a person
    and its substance is not machine-checked, so that layer is exactly as strong as the reviewer, and
    ``EXERCISED:`` names a test, checked below for existence rather than for strength.
    """
    claims = enumerate_claims()
    registry = _read_registry()
    enumerated = {(p, s) for p, _, s, _ in claims}

    missing = [(p, n, s, t) for p, n, s, t in claims if (p, s) not in registry]
    detail = "\n".join(f"  {s} {p}:{n} {t}" for p, n, s, t in missing)
    assert not missing, f"unregistered ledger claim(s) -- add a row to {REGISTRY.name}:\n{detail}"

    stale = sorted(k for k in registry if k not in enumerated)
    detail = "\n".join(f"  {sha} {path}" for path, sha in stale)
    assert not stale, f"registry rows no longer matched -- delete from {REGISTRY.name}:\n{detail}"

    bad = {k: st for k, st in registry.items()
           if st != "UNREVIEWED" and not st.startswith(VALID_PREFIXES)}
    assert not bad, f"status must be UNREVIEWED or start with {VALID_PREFIXES}: {bad}"

    # An EXERCISED row must name a test that exists. This does not prove the test is strong; it
    # removes the failure mode where the named node id was never real in the first place.
    for key, status in sorted(registry.items()):
        if not status.startswith("EXERCISED:"):
            continue
        node = status[len("EXERCISED:"):].strip()
        file_part, _, func = node.partition("::")
        target = REPO / file_part
        assert target.is_file(), f"{key}: EXERCISED names a missing file: {node}"
        assert func, f"{key}: EXERCISED must name a test function: {node}"
        assert f"def {func}(" in target.read_text(encoding="utf-8"), (
            f"{key}: EXERCISED names a test that does not exist: {node}"
        )

    unreviewed = sorted(k for k, st in registry.items() if st == "UNREVIEWED")
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

#: Directories never descended. The walk is over the WORKING TREE, not the index: an untracked
#: `*.md` sitting in a checkout is scanned.
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

    Mechanism: the *expression-then-result* form ``<num>[unit] <op> <num>[unit] ... = <num>[%]``,
    matched **per line** by :data:`STATEMENT` over every ``*.md`` / ``*.py`` / ``*.bib`` file in the
    working tree outside the pruned directories, then evaluated left to right with ``*`` ``x`` ``/``
    binding tighter than ``+`` ``-``. It checks exactly what :data:`STATEMENT` matches; arithmetic
    written in any other form is unchecked.

    Units are skipped rather than converted and a ``%`` result is scaled by 100, so a statement that
    is correct can still be reported. A flag is evidence to weigh, never an instruction to add an
    exception row. It checks arithmetic, not physics: a correctly-computed quotient of two wrong
    inputs passes.

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
    for _p, _s in sorted({(_p, _s) for _p, _n, _s, _t in enumerate_claims()}):
        print(f"{_s}\t{_p}\tUNREVIEWED")
