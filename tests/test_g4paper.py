"""The draft paper on the muonic-atom data layer: its shape, its citations and its open sections.

`tests/test_g4prose.py` holds every number the draft states to a computed value (the draft is one
of its `PROSE_PATHS`). This file holds what that check cannot see: the sections the format asks
for, in its order; a length inside its bounds; every citation resolving to one bibliography entry
and every entry cited; the sections only the author may write left as their placeholder; no
sentence claiming the layer evaluates the data it carries; and the continuous-integration job that
builds the draft. Each guard has a drill that feeds it a planted defect and requires it to fire.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
PAPER = REPO / "paper" / "muonic-data" / "paper.md"
BIB = REPO / "paper" / "muonic-data" / "paper.bib"
CI = REPO / ".github" / "workflows" / "ci.yml"

#: The sections the journal's paper format lists, in its order.
SECTIONS = (
    "Summary", "Statement of need", "State of the field", "Software design",
    "Research impact statement", "AI usage disclosure", "Acknowledgements", "References",
)
#: The bounds the format states for the length of the paper, in words.
WORDS_MIN, WORDS_MAX = 750, 1750
#: The sections only the author writes; each holds exactly this placeholder until then.
AUTHOR_SECTIONS = ("AI usage disclosure", "Acknowledgements")
PLACEHOLDER = "<!-- TODO: the author completes this section before any submission. -->"
#: A sentence saying this work evaluates the data it carries.
EVALUATION_CLAIM = re.compile(
    r"\b(?:we|this work|this software|the authors?)\s+(?:have\s+)?evaluat\w*|\bour\s+evaluat\w*", re.I
)
CITATION = re.compile(r"(?<![\w.])@([A-Za-z][\w:-]*\w)")
BIB_KEY = re.compile(r"^@\w+\{([^,\s]+),", re.M)


def body(text: str) -> str:
    """The paper after its YAML metadata block."""
    assert text.startswith("---\n"), "the paper does not open with its metadata block"
    return text.split("\n---\n", 1)[1]


def headings(text: str) -> list[str]:
    return re.findall(r"(?m)^# (.+?)\s*$", body(text))


def word_count(text: str) -> int:
    """Whitespace-separated words of the body, HTML comments removed."""
    return len(re.sub(r"<!--.*?-->", "", body(text), flags=re.S).split())


def section(text: str, name: str) -> str:
    """The text between heading `name` and the next heading, stripped."""
    parts = re.split(r"(?m)^# (.+?)\s*$", body(text))
    found = [parts[i + 1] for i in range(1, len(parts) - 1, 2) if parts[i] == name]
    assert len(found) == 1, (name, len(found))
    return found[0].strip()


def citation_problems(text: str, bib: str) -> list[str]:
    cited = set(CITATION.findall(body(text)))
    keys = BIB_KEY.findall(bib)
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    problems = [f"duplicate bibliography key {k}" for k in repeated]
    problems += [f"cited, not in the bibliography: {k}" for k in sorted(cited - set(keys))]
    problems += [f"in the bibliography, never cited: {k}" for k in sorted(set(keys) - cited)]
    return problems


def test_the_sections_are_the_formats_in_its_order():
    assert headings(PAPER.read_text(encoding="utf-8")) == list(SECTIONS)


def test_the_length_is_inside_the_formats_bounds():
    assert WORDS_MIN <= word_count(PAPER.read_text(encoding="utf-8")) <= WORDS_MAX


def test_every_citation_resolves_and_every_entry_is_cited():
    problems = citation_problems(PAPER.read_text(encoding="utf-8"), BIB.read_text(encoding="utf-8"))
    assert not problems, problems


def test_the_author_sections_hold_only_their_placeholder():
    text = PAPER.read_text(encoding="utf-8")
    for name in AUTHOR_SECTIONS:
        assert section(text, name) == PLACEHOLDER, name


def test_no_sentence_claims_an_evaluation():
    hits = EVALUATION_CLAIM.findall(PAPER.read_text(encoding="utf-8"))
    assert not hits, hits


def test_continuous_integration_builds_the_draft():
    ci = CI.read_text(encoding="utf-8")
    assert "paper-path: paper/muonic-data/paper.md" in ci
    assert "path: paper/muonic-data/paper.pdf" in ci


# ---- drills: each guard above fires on a planted defect ----


def test_drill_a_missing_or_moved_section_is_seen():
    text = PAPER.read_text(encoding="utf-8")
    assert headings(text.replace("# State of the field\n", "")) != list(SECTIONS)
    swapped = text.replace("# Summary\n", "# @@A\n").replace("# Statement of need\n", "# Summary\n")
    assert headings(swapped.replace("# @@A\n", "# Statement of need\n")) != list(SECTIONS)


def test_drill_a_length_outside_the_bounds_is_seen():
    text = PAPER.read_text(encoding="utf-8")
    assert word_count(text + "\nword" * (WORDS_MAX + 1)) > WORDS_MAX
    assert word_count("---\ntitle: x\n---\n# Summary\n" + "word " * 10) < WORDS_MIN


def test_drill_an_unresolved_or_orphan_citation_is_seen():
    text, bib = PAPER.read_text(encoding="utf-8"), BIB.read_text(encoding="utf-8")
    planted = text + "\n[@Planted1999]\n"
    assert "cited, not in the bibliography: Planted1999" in citation_problems(planted, bib)
    orphan = bib + "\n@misc{Orphan2000,\n  title = {x}\n}\n"
    assert "in the bibliography, never cited: Orphan2000" in citation_problems(text, orphan)
    first = BIB_KEY.findall(bib)[0]
    twice = bib + f"\n@misc{{{first},\n  title = {{x}}\n}}\n"
    assert f"duplicate bibliography key {first}" in citation_problems(text, twice)


def test_drill_text_in_an_author_section_is_seen():
    text = PAPER.read_text(encoding="utf-8")
    for name in AUTHOR_SECTIONS:
        planted = text.replace(f"# {name}\n\n{PLACEHOLDER}", f"# {name}\n\n{PLACEHOLDER}\nplanted", 1)
        assert planted != text
        assert section(planted, name) != PLACEHOLDER


def test_drill_an_evaluation_claim_is_seen():
    for planted in ("We evaluated muon capture rates.", "Our evaluation of the rates", "this work evaluates"):
        assert EVALUATION_CLAIM.search(planted), planted
