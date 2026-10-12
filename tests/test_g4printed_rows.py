"""The printed-row ledgers quote a source's own text only as a table title or a single sentence.

`data/g4/d1/suzuki1987_printed_rows.csv` and `data/g4/reference/d2/printed_rows.csv` carry, in their `note`
column, the text of each caption, note and footnote line a table prints. Of a caption a ledger keeps the
table's title: its label and the first sentence after it. Of a note or footnote of more than a sentence it
keeps only its footnote marker, if any. In place of the text it leaves out it gives
`[text omitted; sha256 <64 hex digits>]`, the SHA-256 of the UTF-8 bytes of that text, so that a reader
holding the source can confirm what stood there. The `normalization` column of
`data/g4/reference/d2/measurements.csv` cites a footnote by its locator and keeps to one sentence as well, and
its `capture_branch_correction` column gives the words in which a source states that correction only as the
same mark.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGERS = ("data/g4/d1/suzuki1987_printed_rows.csv", "data/g4/reference/d2/printed_rows.csv")
MEASUREMENTS = "data/g4/reference/d2/measurements.csv"
#: The row kinds whose `note` cell holds the source's own text rather than a printed column label.
PROSE_KINDS = ("caption", "note", "footnote")
#: The mark that stands in for text a ledger leaves out, at the end of its cell.
OMITTED = re.compile(r"(?:^| )\[text omitted; sha256 [0-9a-f]{64}\]$")
#: A `capture_branch_correction` cell: the mark in place of the source's words, then `; `-separated segments
#: of this repository's own.
CAPTURE_BRANCH = re.compile(r"\[text omitted; sha256 [0-9a-f]{64}\](?:; [^;]+)+")
#: A table's label at the head of a caption: `TABLE III.`, `Table 1 :`, `TABLE IV. (Continued).`.
LABEL = re.compile(r"^(?:TABLE|Table) [0-9IVX]+\.?(?: ?\([Cc]ontinued\)\.?)?\s*:?\s*")
#: A footnote's marker at the head of its text: `^a`, `^b`.
MARKER = re.compile(r"^\^[a-z]")
#: Abbreviations whose full stop ends no sentence.
ABBREVIATIONS = ("Ref.", "Refs.", "Sec.", "Eq.", "Eqs.", "Fig.", "viz.", "e.g.", "i.e.")
#: A sentence end: a full stop, question or exclamation mark, a closing parenthesis after it or not, white
#: space, and a capital letter, an opening parenthesis or a footnote marker.
SENTENCE_END = re.compile(r"[.!?]\)?\s+(?=[A-Z(^])")


def _rows(relpath: str) -> list[dict[str, str]]:
    with (REPO / relpath).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sentences(text: str) -> int:
    """How many sentences `text` holds: one more than its sentence ends."""
    for abbreviation in ABBREVIATIONS:
        text = text.replace(abbreviation, abbreviation[:-1])
    return len(SENTENCE_END.findall(text.strip())) + 1


def quoted(cell: str, kind: str) -> str:
    """What `cell` quotes beyond what identifies it: the omission mark comes off, and so does a caption's
    table label or a footnote's marker."""
    text = OMITTED.sub("", cell).rstrip()
    return (LABEL if kind == "caption" else MARKER).sub("", text, count=1)


def test_printed_ledgers_quote_a_table_title_or_one_sentence() -> None:
    """Every caption, note and footnote cell of both ledgers quotes at most one sentence beyond its label or
    marker, and every `normalization` cell of the measurement rows is one sentence."""
    for relpath in LEDGERS:
        rows = [row for row in _rows(relpath) if row["row_kind"] in PROSE_KINDS]
        assert rows, relpath
        for row in rows:
            assert sentences(quoted(row["note"], row["row_kind"])) == 1, (relpath, row["note"])
    for row in _rows(MEASUREMENTS):
        assert sentences(row["normalization"]) == 1, (row["record_id"], row["normalization"])


def test_capture_branch_cells_give_a_sources_words_only_as_their_sha256() -> None:
    """Every `capture_branch_correction` cell of the measurement rows that is not empty opens with the
    omission mark, which stands in place of the words in which its source states the correction; the
    segments after it are this repository's own, and the last is the source's page, as
    `tests/test_g4d2.py` requires."""
    cells = [cell for row in _rows(MEASUREMENTS) if (cell := row["capture_branch_correction"])]
    assert cells
    for cell in cells:
        assert CAPTURE_BRANCH.fullmatch(cell), cell


def test_every_omission_mark_names_a_sha256() -> None:
    """A cell that leaves text out says so in the one form :data:`OMITTED` reads, with a full digest."""
    marks = [row["note"] for relpath in LEDGERS for row in _rows(relpath) if "[text omitted" in row["note"]]
    assert marks
    assert all(OMITTED.search(mark) for mark in marks), marks


def test_drill_a_second_sentence_or_a_short_digest_is_refused() -> None:
    """The guards above, shown to fire: a caption's title passes and the same title with a sentence after it
    does not; a one-sentence footnote passes and the same footnote with a second sentence does not; an
    abbreviation's full stop ends no sentence; an omission mark with a short digest is not read as one; and a
    capture-branch cell that quotes its source's words, with no mark or with a short digest, is refused."""
    title = "TABLE III. Compendium of total muon capture results for light nuclei."
    assert sentences(quoted(title, "caption")) == 1
    assert sentences(quoted(title + " When underlined it is an estimate.", "caption")) == 2
    footnote = "^aDenotes the results of this experiment."
    assert sentences(quoted(footnote, "footnote")) == 1
    assert sentences(quoted(footnote + " See Ref. 26 for details.", "footnote")) == 2
    assert sentences("(these F data show the lifetime for the lower hf state; see Sec. IV D).") == 1
    assert not OMITTED.search("TABLE III. [text omitted; sha256 0123abcd]")
    mark = "[text omitted; sha256 " + "0" * 64 + "]"
    assert CAPTURE_BRANCH.fullmatch(mark + "; Eqs. (3), (5), (8); a p.7-8")
    assert not CAPTURE_BRANCH.fullmatch("we must correct for those events; Eqs. (3), (5), (8); a p.7-8")
    assert not CAPTURE_BRANCH.fullmatch("[text omitted; sha256 0123abcd]; a p.4")
