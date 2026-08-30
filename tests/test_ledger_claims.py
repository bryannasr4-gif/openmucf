"""Three guards over the *prose* that surrounds the muon-cost ledger.

None of them reads the ledger. They bind the sentences written *about* it, which no manifest,
byte-diff or `provenance --check` can see: a manifest pins a value against the document that renders
it, so a claim typed into a docstring, a comment or a hand-written CHANGELOG line can be false while
every existing gate stays green.

- **G1 :func:`test_quantified_claims_registered`** enumerates the lines its two patterns match over
  the named paths and requires each to carry a registry row. A row records one of three statuses:
  ``EXERCISED:`` names a test, ``REGISTERED:`` records a human judgement, and ``UNREVIEWED`` records
  that neither has happened yet, capped by :data:`LEDGER_CLAIMS_UNREVIEWED_CEILING`. G1 decides
  nothing about truth. What it buys is that no ``(path, text)`` pair can enter or leave those paths
  without a registry diff.
- **G2 :func:`test_prose_arithmetic_recomputes`** recomputes arithmetic that prose states in full.
- **G3 :func:`test_figure_text_registered`** enumerates the text a shipped figure renders -- titles,
  axis labels, legend labels, annotations -- from the generators that save one, and requires each
  string to carry a registry row. A figure is outside the byte-diff and outside G1's line surface,
  which is how a false label survived three sweeps.

All three keep their bookkeeping in TSV files beside this one, keyed on the SHA-1 of the
whitespace-normalized text -- the registries additionally on the path -- never on line number, which
churns on every commit.

The two G1 patterns are BUILT from the form tables below, and :func:`test_guard_forms_are_exampled`
requires every form to be atomic, to own at least one example sentence in
``ledger_claims_examples.tsv``, and to be the only form on its side that matches that sentence; so a
form cannot be deleted, or shadowed by a neighbour, without a test going red. Before this, a form
that happened to match no line of the tree could be removed in silence -- thirteen of them could.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import sys
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
    # The prose homes read and registered in the 2026-08-24 claim sweep before they were added here:
    # the shipped data's own free text, the CC-BY data-package descriptor, and the two generated
    # documents that restate ledger claims, together with the generators that emit them.
    "openmucf/data/muon_cost.csv",
    # The edge table and its schema, added when they shipped: they are the same kind of prose home
    # as the node table and its schema above, and a guard that watched one and not the other would
    # be blindest exactly where the newest claims are.
    "openmucf/data/muon_cost_chain.csv",
    "openmucf/data/muon_cost_chain.schema.json",
    "datapackage.json",
    "FINDINGS.md",
    "NEUTRONOMICS.md",
    "scripts/generate_findings.py",
    "scripts/generate_neutronomics.py",
    # The shipped list of bibkeys with an unresolved identifier: a prose home the 2026-08-24 sweep
    # read, added here after a drill restored a retracted universal to it and the suite stayed
    # green.
    "openmucf/data/bib_unresolved.txt",
    # This file. A guard that exempts itself is not a guard: this module's own prose is watched on
    # the same terms as every other path here. The three TSV files beside it are not listed: they
    # are this guard's own fixtures and bookkeeping (form examples, registry rows), not prose.
    "tests/test_ledger_claims.py",
)

#: The universal and uniqueness forms. Each entry is one regex fragment, matched between ``\b``
#: anchors, case-insensitively; :func:`test_guard_forms_are_exampled` requires it to be atomic (no
#: top-level ``|``) and to own an example sentence that no other form here matches. Modals
#: (`must`, `cannot`), ordinals and `both`/`identical`/`unchanged` are deliberately OUT: they are
#: not quantifiers. A form is admitted only after a tracked line of this repository's own prose has
#: been shown to state a universal the pattern missed because of it, and only in a change that reads
#: and rules every line the form adds; `exact` entered that way (a shipped descriptor it hid was
#: true, and had never been enumerable). Negation -- `not`, `cannot`, `\w+n't` -- is the measured
#: residue: it states a universal ("does not depend on") that this table cannot see, and at
#: 2026-08-30 it adds 322 lines, so it lands as its own change, read in full, not here.
STRONG_FORMS = (
    "every", "all", "each", "none", "never", "always", "only", "sole", "solely", "exactly", "exact",
    "unique", "uniquely", "neither", "any", "entire", r"without\s+exception", "no", "nothing",
)

#: Words that make a sentence a claim about the ledger rather than about anything else. `bound` and
#: `cost` are deliberately in: the ledger's basis universals are commonly phrased with them.
#: The verb form of `headline` is included, after a drill wrote a sentence using it that matched
#: nothing here. The aggregate nouns -- `ratio`, `spread`, `aggregate`, `median`, `box`, `edge` and
#: their plurals -- entered after a drill landed "the published ratio never overstates the spread"
#: in a watched path and the suite stayed green; the plurals of `ledger`, `manifest`, `quotient` and
#: `denominator` entered with them. Same admission rule as :data:`STRONG_FORMS`.
LEDGER_FORMS = (
    "row", "rows", "tier", "tiers", "cell", "cells", "anchor", "anchors", "source", "sources",
    "basis", "bases", "numeraire", "numeraires", "stage", "stages", "ledger", "ledgers",
    "entry", "entries", "chain", "chains", "headline", "headlines", "manifest", "manifests",
    "bibkey", "bibkeys", "evidence_status", "charge_basis", "bound", "bounds", "cost", "costs",
    "value", "values", "figure", "figures", "quotient", "quotients", "denominator", "denominators",
    "problem", "problems", "contract", "contracts", "claim", "claims", "number", "numbers",
    "ratio", "ratios", "spread", "spreads", "aggregate", "aggregates", "median", "medians",
    "box", "boxes", "edge", "edges",
)


def build_pattern(forms: tuple[str, ...]) -> re.Pattern[str]:
    """``\\b(f1|f2|...)\\b``, case-insensitive -- the one place a form table becomes a pattern."""
    return re.compile(r"\b(" + "|".join(forms) + r")\b", re.IGNORECASE)


STRONG = build_pattern(STRONG_FORMS)
LEDGER = build_pattern(LEDGER_FORMS)

#: ``side <TAB> form <TAB> example``: for each form, a sentence it alone makes a claim of. Kept out
#: of this module so that the examples, which match the patterns by construction, are not themselves
#: enumerated as claims about the ledger.
EXAMPLES = Path(__file__).with_name("ledger_claims_examples.tsv")

#: How many matched lines may still be `UNREVIEWED`. **One-sided and monotone NON-INCREASING**, on
#: the exact precedent of ``AUDIT_ESS_FLOOR``: raising it is a visible diff to this line and must be
#: argued for, lowering it needs no argument. Review defers rather than drops: an unreviewed line
#: stays enumerated and capped instead of leaving the registry.
LEDGER_CLAIMS_UNREVIEWED_CEILING = 80

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


def _read_registry(path: Path = REGISTRY) -> dict[tuple[str, str], str]:
    """{(path, sha1): status}. Three tab-separated fields per row: sha1, path, status.

    The registry carries no copy of the matched text. That text lives in the file the path names and
    the enumerator prints it on failure; copying it here made this file a second home for text
    written somewhere else.
    """
    rows: dict[tuple[str, str], str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        assert len(parts) == 3, f"registry row must have 3 tab-separated fields: {raw!r}"
        sha, rel, status = parts
        assert (rel, sha) not in rows, f"duplicate registry row for {(rel, sha)}"
        rows[(rel, sha)] = status
    return rows


def _check_registry(
    registry: dict[tuple[str, str], str],
    enumerated: set[tuple[str, str]],
    ceiling: int,
    name: str,
) -> None:
    """The four checks every registry gets: no unregistered match, no stale row, statuses well formed
    (an EXERCISED row names a test that exists), and UNREVIEWED under its ceiling."""
    stale = sorted(k for k in registry if k not in enumerated)
    detail = "\n".join(f"  {sha} {rel}" for rel, sha in stale)
    assert not stale, f"registry rows no longer matched -- delete from {name}:\n{detail}"

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
    assert len(unreviewed) <= ceiling, (
        f"{len(unreviewed)} UNREVIEWED entries in {name} exceeds the ceiling {ceiling}; the ceiling "
        f"is monotone non-increasing -- review the new entries instead of raising it"
    )


def test_quantified_claims_registered():
    """Lines the enumerator matches carry a registry row; nothing unreviewed beyond the ceiling.

    **The enumerator, stated in full -- this is the whole definition of what gets caught.** Over
    the paths in :data:`CLAIM_PATHS`, read **per line** (not per docstring: an AST pass over function
    and class docstrings reads neither the module docstring nor any comment). A line is a **claim**
    iff one form of :data:`STRONG_FORMS` and one form of :data:`LEDGER_FORMS` both match it, each
    between ``\\b`` anchors, case-insensitively -- :func:`build_pattern` is the only construction, and
    ``python tests/test_ledger_claims.py --patterns`` prints the two compiled patterns.

    Each match is keyed by its path and the SHA-1 of its whitespace-normalized text. The test fails
    when a match carries no row, when a row is no longer matched (deleting a claim cannot silently
    keep its credit), and when the ``UNREVIEWED`` count exceeds
    :data:`LEDGER_CLAIMS_UNREVIEWED_CEILING`.

    This checks exactly what those two patterns match, line by line, over exactly those paths; a
    universal in any other form, split across lines, or written anywhere else is unchecked. Two such
    forms are measured and registered rather than unknown: a universal stated by negation (`does
    not depend on`), and a wrapped sentence whose quantifier and ledger word fall on different
    rendered lines -- whether or not one of its lines happens to match on its own, since an edit to
    the other line then re-keys nothing. It does not decide whether a matched claim is true:
    ``REGISTERED:`` records a judgement made by a person and its substance is not machine-checked,
    so that layer is exactly as strong as the reviewer, and ``EXERCISED:`` names a test, checked for
    existence rather than for strength.
    """
    claims = enumerate_claims()
    registry = _read_registry()
    enumerated = {(p, s) for p, _, s, _ in claims}

    missing = [(p, n, s, t) for p, n, s, t in claims if (p, s) not in registry]
    detail = "\n".join(f"  {s} {p}:{n} {t}" for p, n, s, t in missing)
    assert not missing, f"unregistered ledger claim(s) -- add a row to {REGISTRY.name}:\n{detail}"
    _check_registry(registry, enumerated, LEDGER_CLAIMS_UNREVIEWED_CEILING, REGISTRY.name)


def _read_examples() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for raw in EXAMPLES.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        assert len(parts) == 3, f"example row must have 3 tab-separated fields: {raw!r}"
        side, form, example = parts
        assert side in ("STRONG", "LEDGER"), f"side must be STRONG or LEDGER: {raw!r}"
        rows.append((side, form, example))
    return rows


_SIDES = {"STRONG": (STRONG_FORMS, LEDGER), "LEDGER": (LEDGER_FORMS, STRONG)}


def test_guard_forms_are_exampled():
    """Every form is atomic, owns an example, and is the ONLY form on its side matching it.

    The three invariants together are what make a deletion visible: a form removed from its table
    leaves an example row naming a form that does not exist; an example removed from the TSV leaves
    a form owning no example; a form whose spelling drifts stops matching its own example; and a
    form that swallows a neighbour's example (`sole` would, if it were written ``sole\\w*``) fails
    the uniqueness check on that neighbour. Each example must also be a claim -- match the OTHER
    side -- so the fixture exercises the enumerator as written, not a weaker version of it.
    """
    examples = _read_examples()
    owned: dict[tuple[str, str], int] = {}
    for side, form, example in examples:
        forms, other = _SIDES[side]
        assert form in forms, f"example names a form that is not in {side}_FORMS: {form!r}"
        assert "|" not in form, f"form is not atomic: {form!r}"
        assert build_pattern((form,)).search(example), f"{side} form {form!r} does not match {example!r}"
        also = [f for f in forms if f != form and build_pattern((f,)).search(example)]
        assert not also, f"{side} example {example!r} for {form!r} is also matched by {also}"
        assert other.search(example), f"example is not a claim (no match on the other side): {example!r}"
        owned[(side, form)] = owned.get((side, form), 0) + 1
    for side, (forms, _) in _SIDES.items():
        unexampled = [f for f in forms if (side, f) not in owned]
        assert not unexampled, f"{side} forms with no example row in {EXAMPLES.name}: {unexampled}"


def test_deleting_any_form_breaks_its_example():
    """The mutation drill, run in-suite: with any one form removed, its example stops being a claim.

    :func:`test_guard_forms_are_exampled` implies this; it is asserted directly because the property
    that matters -- no single-token edit to either table can pass in silence -- was measured by hand
    once and would regress silently if it only followed from the other test's structure.
    """
    examples = _read_examples()
    for side, (forms, other) in _SIDES.items():
        for form in forms:
            without = build_pattern(tuple(f for f in forms if f != form))
            own = [ex for s, f, ex in examples if s == side and f == form]
            assert own, f"{side} form {form!r} has no example to drill"
            for example in own:
                assert other.search(example), f"{example!r} is not a claim on the other side"
                assert not without.search(example), (
                    f"removing {side} form {form!r} leaves {example!r} still enumerated -- another "
                    f"form covers it, so its deletion would be silent"
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


def _walk(top: Path, suffixes: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for root, dirs, fnames in os.walk(top):
        dirs[:] = [d for d in dirs
                   if d not in _PRUNE and not d.startswith(".venv") and not d.endswith(".egg-info")]
        for f in fnames:
            if f.endswith(suffixes):
                out.append(str(Path(root, f).relative_to(REPO)).replace("\\", "/"))
    return sorted(out)


def _scanned_files() -> list[str]:
    return _walk(REPO, (".md", ".py", ".bib"))


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


# --------------------------------------------------------------------------------------------------
# G3 -- the text a shipped figure renders
# --------------------------------------------------------------------------------------------------

#: matplotlib calls whose positional string arguments are drawn into a figure.
FIGURE_TEXT_CALLS = frozenset({
    "set_title", "set_xlabel", "set_ylabel", "set_label", "suptitle", "title", "xlabel", "ylabel",
    "annotate", "text", "figtext", "set_xticklabels", "set_yticklabels", "legend",
})
#: keyword arguments, on any call, whose string value is drawn into a figure (`plot(label=...)`,
#: `axvline(label=...)`, `legend(title=...)`, and the text calls' own first parameters passed by
#: name: `annotate(text=)`, `text(s=)`, `suptitle(t=)`, `set_xticks(labels=)`). Any call is
#: admitted rather than a list of plotting methods: over-enumerating a string that turns out not
#: to reach a figure costs one registry row. Both tables above are still hand-kept lists, so each
#: member owns a snippet in :data:`FIGURE_TEXT_EXAMPLES` that the extractor must find through that
#: member alone -- the same discipline as the G1 form tables, enforced by
#: :func:`test_figure_text_calls_are_exampled`.
FIGURE_TEXT_KWARGS = frozenset({"label", "title", "xlabel", "ylabel", "text", "s", "t", "labels"})
#: ``(table, member, snippet)``: one snippet per member of the two tables above, each yielding
#: exactly the string ``"probe"`` through that member and nothing once the member is removed.
FIGURE_TEXT_EXAMPLES = (
    ("call", "set_title", 'ax.set_title("probe")'),
    ("call", "set_xlabel", 'ax.set_xlabel("probe")'),
    ("call", "set_ylabel", 'ax.set_ylabel("probe")'),
    ("call", "set_label", 'colorbar.set_label("probe")'),
    ("call", "suptitle", 'fig.suptitle("probe")'),
    ("call", "title", 'plt.title("probe")'),
    ("call", "xlabel", 'plt.xlabel("probe")'),
    ("call", "ylabel", 'plt.ylabel("probe")'),
    ("call", "annotate", 'ax.annotate("probe", (0, 0))'),
    ("call", "text", 'ax.text(0, 0, "probe")'),
    ("call", "figtext", 'fig.figtext(0, 0, "probe")'),
    ("call", "set_xticklabels", 'ax.set_xticklabels(["probe"])'),
    ("call", "set_yticklabels", 'ax.set_yticklabels(["probe"])'),
    ("call", "legend", 'ax.legend(["probe"])'),
    ("kwarg", "label", 'ax.plot(x, y, label="probe")'),
    ("kwarg", "title", 'ax.legend(title="probe")'),
    ("kwarg", "xlabel", 'ax.set(xlabel="probe")'),
    ("kwarg", "ylabel", 'ax.set(ylabel="probe")'),
    ("kwarg", "text", 'ax.annotate(text="probe", xy=(0, 0))'),
    ("kwarg", "s", 'ax.text(0, 0, s="probe")'),
    ("kwarg", "t", 'fig.suptitle(t="probe")'),
    ("kwarg", "labels", 'ax.set_xticks([0], labels=["probe"])'),
)
FIGURE_DIR = REPO / "figures"
FIGURE_TEXT_REGISTRY = Path(__file__).with_name("figure_text_registry.tsv")
#: A figure string is composed, not inherited: nothing predates this guard, so nothing is deferred.
FIGURE_TEXT_UNREVIEWED_CEILING = 0


def figure_generators() -> list[str]:
    """Every ``*.py`` under ``scripts/`` and ``openmucf/`` whose source calls ``savefig``.

    Derived from what ships -- a generator is one that saves a figure -- rather than from a list of
    file names, after a hand-kept inventory once counted a generator that emits no figure and missed
    three text-bearing calls in ones that do.
    """
    out = []
    for rel in _walk(REPO / "scripts", (".py",)) + _walk(REPO / "openmucf", (".py",)):
        src = (REPO / rel).read_text(encoding="utf-8")
        # the substring is a cheap pre-filter; the AST decides, so a comment that merely mentions
        # savefig( does not make a module a generator
        if "savefig(" in src and any(
            isinstance(n, ast.Call) and _call_name(n) == "savefig" for n in ast.walk(ast.parse(src))
        ):
            out.append(rel)
    return sorted(out)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _string_values(node: ast.AST, consts: dict[str, str]) -> list[str]:
    """The string(s) an argument node carries: a literal, an f-string with ``{}`` for each field, a
    list or tuple of those, or a module-level constant bound to one. Text assembled at run time from
    data (``LABELS[key]``, ``f"{value}"``) is not a literal and is outside this guard; it is stated
    here rather than approximated."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            parts.append(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}")
        return ["".join(parts)]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [s for e in node.elts for s in _string_values(e, consts)]
    if isinstance(node, ast.Name) and node.id in consts:
        return [consts[node.id]]
    return []


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    consts: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            vals = _string_values(node.value, {})
            if len(vals) == 1:
                consts[node.targets[0].id] = vals[0]
    return consts


def enumerate_figure_text() -> list[tuple[str, int, str, str]]:
    """(generator path, lineno, sha1, normalized text) for every figure-text string, in file then
    line order. A newline inside a title normalizes to a space, so re-wrapping a string for layout
    keeps its key."""
    out: list[tuple[str, int, str, str]] = []
    for rel in figure_generators():
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        for lineno, s in _figure_text_in(tree, _module_string_constants(tree)):
            out.append((rel, lineno, claim_sha1(s), _normalize(s)))
    return out


def _figure_text_in(
    tree: ast.AST,
    consts: dict[str, str],
    calls: frozenset[str] = FIGURE_TEXT_CALLS,
    kwargs: frozenset[str] = FIGURE_TEXT_KWARGS,
) -> list[tuple[int, str]]:
    """(lineno, raw string) for every figure-text string in one parsed module, sorted."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) in calls:
            for a in node.args:
                found.extend((node.lineno, s) for s in _string_values(a, consts))
        for kw in node.keywords:
            if kw.arg in kwargs:
                found.extend((node.lineno, s) for s in _string_values(kw.value, consts))
    return sorted(found)


def test_figure_text_calls_are_exampled():
    """Every member of the two G3 tables owns a snippet the extractor reads through it alone.

    The tables are hand-kept lists, which is the shape of hole the G1 form tables just closed: a
    member that no shipped generator happens to use could be deleted with every guard green. Each
    snippet must yield exactly ``"probe"`` with the tables as shipped and nothing with its member
    removed, and every member must own at least one snippet.
    """
    owned: set[tuple[str, str]] = set()
    for table, member, snippet in FIGURE_TEXT_EXAMPLES:
        pool = FIGURE_TEXT_CALLS if table == "call" else FIGURE_TEXT_KWARGS
        assert member in pool, f"{table} example names a member that is not in the table: {member!r}"
        tree = ast.parse(snippet)
        assert [s for _, s in _figure_text_in(tree, {})] == ["probe"], f"{snippet!r} did not yield probe"
        without = pool - {member}
        reduced = (
            _figure_text_in(tree, {}, calls=without) if table == "call"
            else _figure_text_in(tree, {}, kwargs=without)
        )
        assert reduced == [], f"{snippet!r} still yields text without {table} member {member!r}"
        owned.add((table, member))
    for table, pool in (("call", FIGURE_TEXT_CALLS), ("kwarg", FIGURE_TEXT_KWARGS)):
        unexampled = sorted(m for m in pool if (table, m) not in owned)
        assert not unexampled, f"{table} members with no snippet in FIGURE_TEXT_EXAMPLES: {unexampled}"


def test_figure_text_registered():
    """Every string a figure generator draws carries a registry row; none is unreviewed.

    Over :func:`figure_generators`, every string literal (or f-string template, or module constant)
    passed positionally to a call in :data:`FIGURE_TEXT_CALLS` or by a keyword in
    :data:`FIGURE_TEXT_KWARGS` is keyed by the generator's path and the SHA-1 of its
    whitespace-normalized text. It checks exactly those strings. Unchecked, and stated rather than
    approximated: text built from data at run time (``LABELS[key]`` -- the point labels of
    ``figures/muon_cost_gap.png`` are drawn that way), a string reached through an alias or
    ``getattr``, through ``*args``, through ``+`` concatenation or a local variable bound to a
    literal, a ``matplotlib.text.Text`` set through any other call, and a figure written by a module
    outside ``scripts/`` and ``openmucf/``. It does not decide whether a label is true; the
    ``REGISTERED:`` reason records what was checked against, and is as strong as its reviewer.
    """
    strings = enumerate_figure_text()
    assert strings, "no figure text found at all -- the extractor is broken, not the generators"
    for rel in figure_generators():
        assert any(p == rel for p, _, _, _ in strings), f"{rel} saves a figure but yielded no text"
    registry = _read_registry(FIGURE_TEXT_REGISTRY)
    enumerated = {(p, s) for p, _, s, _ in strings}
    missing = [(p, n, s, t) for p, n, s, t in strings if (p, s) not in registry]
    detail = "\n".join(f"  {s} {p}:{n} {t}" for p, n, s, t in missing)
    assert not missing, f"unregistered figure text -- add a row to {FIGURE_TEXT_REGISTRY.name}:\n{detail}"
    _check_registry(registry, enumerated, FIGURE_TEXT_UNREVIEWED_CEILING, FIGURE_TEXT_REGISTRY.name)


def test_every_shipped_figure_is_named_by_a_generator():
    """Each ``figures/*.png`` on disk is named, as a literal, by a generator that saves a figure --
    and each such literal names a file that exists. A figure with no generator is one whose text
    G3 cannot enumerate; a literal with no file is a generator writing somewhere the tree does not
    track."""
    shipped = sorted(p.name for p in FIGURE_DIR.glob("*.png"))
    assert shipped, "no shipped figures found -- the figure directory moved"
    sources = {rel: (REPO / rel).read_text(encoding="utf-8") for rel in figure_generators()}
    for name in shipped:
        assert any(f"figures/{name}" in src for src in sources.values()), (
            f"figures/{name} is shipped but no figure generator names it"
        )
    for rel, src in sources.items():
        for name in re.findall(r"figures/([A-Za-z0-9_.-]+\.png)", src):
            assert (FIGURE_DIR / name).is_file(), f"{rel} names figures/{name}, which does not exist"


if __name__ == "__main__":
    # python tests/test_ledger_claims.py            -> the G1 registry skeleton (all UNREVIEWED)
    # python tests/test_ledger_claims.py --figures  -> the G3 registry skeleton
    # python tests/test_ledger_claims.py --patterns -> the two compiled G1 patterns
    if "--patterns" in sys.argv:
        print(f"STRONG = {STRONG.pattern}\nLEDGER = {LEDGER.pattern}")
    elif "--figures" in sys.argv:
        for _p, _s in sorted({(_p, _s) for _p, _n, _s, _t in enumerate_figure_text()}):
            print(f"{_s}\t{_p}\tUNREVIEWED")
    else:
        for _p, _s in sorted({(_p, _s) for _p, _n, _s, _t in enumerate_claims()}):
            print(f"{_s}\t{_p}\tUNREVIEWED")
