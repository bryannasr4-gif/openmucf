"""T-64..T-66, T-71 -- the `.g4dat` conformance corpus: does every file still get its verdict?

`FORMAT_SPEC.md` section 4 defines sixteen codes and a three-phase reporting order. That section is
the contract two independent readers -- the Python reference and the C++ validator -- have to agree
on, and prose alone cannot make them agree: a file with three defects in it has one correct code
under the ordering rules and two plausible wrong ones. `tests/fixtures/g4dat_conformance/` turns the
section into artifacts, one per code, per ordering rule and per tie-break, with a recorded verdict.

What each test here is actually for:

* **T-64** -- the corpus is *generated*, so it can be rebuilt; the audit proves the committed bytes
  are what the generator produces today, in both directions (no orphaned file, no missing one).
* **T-65** -- every recorded verdict still recomputes. This is the test that would catch a change to
  `spec.py` that silently moved a code or a line number.
* **T-66** -- the three older `g4dat_bad` fixtures are in the corpus byte-for-byte and keep their
  verdicts, so folding them in did not quietly create a second, divergent copy.
* **T-71** -- the drill: corrupt a member and T-65's comparison must fail *naming the file*.

No code and no line number is written down anywhere in this file. Every expected value is read from
`expected.tsv`, and `expected.tsv` is produced by the reference implementation -- a fixture whose
expected code was typed by hand tests its author's belief about the format, not the format.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
CORPUS = REPO / "tests" / "fixtures" / "g4dat_conformance"
BAD_FIXTURES = REPO / "tests" / "fixtures" / "g4dat_bad"


def conformance_generator():
    """`scripts/generate_g4dat_conformance.py`, loaded by path -- `scripts/` is not a package."""
    path = REPO / "scripts" / "generate_g4dat_conformance.py"
    spec = importlib.util.spec_from_file_location("generate_g4dat_conformance", path)
    assert spec and spec.loader, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_rows(directory: pathlib.Path) -> dict[str, tuple[str, int]]:
    """`expected.tsv` as ``{file name: (code, line)}``, read as bytes and split by hand.

    Bytes rather than `read_text`: universal newlines would rewrite a CRLF in this file into an LF
    and hide exactly the kind of checkout damage `.gitattributes` exists to prevent.
    """
    raw = (directory / "expected.tsv").read_bytes().decode("ascii")
    rows: dict[str, tuple[str, int]] = {}
    for line in raw.splitlines():
        name, code, lineno = line.split("\t")
        rows[name] = (code, int(lineno))
    return rows


def test_t64_the_corpus_regenerates_to_the_committed_bytes(tmp_path):
    """`--audit` on the committed directory: every member is what the generator produces today.

    The drill in the second half is the more important one. `audit()` is only a guard if it fails
    when a byte moves, and the way to know that is to move one: a copy of the directory with a
    single flipped byte must be rejected, and the message must name the file, because "the corpus
    drifted" sends a maintainer to read fifty of them.
    """
    generator = conformance_generator()
    generator.audit()  # the committed directory; raises SystemExit on any drift

    scratch = tmp_path / "g4dat_conformance"
    shutil.copytree(CORPUS, scratch)
    victim = scratch / "e014_inf.g4dat"
    payload = bytearray(victim.read_bytes())
    payload[-2] ^= 0x01  # one bit of one byte -- the smallest corruption the audit must still see
    assert bytes(payload) != victim.read_bytes()
    victim.write_bytes(bytes(payload))

    with pytest.raises(SystemExit) as raised:
        generator.audit(directory=scratch)
    assert victim.name in str(raised.value), raised.value


def test_t65_every_recorded_verdict_still_recomputes():
    """Each `expected.tsv` row equals what the reference implementation says about that file.

    Three things are asserted together on purpose. The verdicts must match; the row count must equal
    the generator's case count (a `len()`, never a literal, so adding a case cannot leave the table
    half-updated); and the row set must equal the set of `.g4dat` files on disk, so a member nobody
    recorded a verdict for cannot sit in the directory being read by nothing.
    """
    generator = conformance_generator()
    rows = expected_rows(CORPUS)

    assert len(rows) == len(generator.CASES)
    on_disk = {path.name for path in CORPUS.glob("*.g4dat")}
    assert on_disk == set(rows)

    mismatched = []
    for name, recorded in sorted(rows.items()):
        actual = generator.reference_verdict(CORPUS / name)
        if actual != recorded:
            mismatched.append(f"{name}: recorded {recorded}, reference says {actual}")
    assert not mismatched, "; ".join(mismatched)


def test_t71_drill_a_corrupted_member_is_reported_by_name(tmp_path):
    """Drill for T-65: change one member's content and the row comparison must name that file.

    The corruption is chosen to move the *verdict*, not merely the bytes: an accept case is given a
    defect, so a comparison that only checked "some row differs" and a comparison that named the
    file are told apart here.
    """
    generator = conformance_generator()
    scratch = tmp_path / "g4dat_conformance"
    shutil.copytree(CORPUS, scratch)

    victim = scratch / "ok_leading_plus.g4dat"
    before = generator.reference_verdict(victim)
    victim.write_bytes(victim.read_bytes().replace(b"+1.5", b"+1.5x", 1))
    after = generator.reference_verdict(victim)
    assert before != after, "the drill did not change the file's verdict"

    rows = expected_rows(scratch)
    mismatched = [
        f"{name}: recorded {rows[name]}, reference says {generator.reference_verdict(scratch / name)}"
        for name in sorted(rows)
        if generator.reference_verdict(scratch / name) != rows[name]
    ]
    assert mismatched, "a corrupted member went unnoticed"
    assert all(victim.name in problem for problem in mismatched), mismatched


def test_t66_the_older_fixtures_are_in_the_corpus_byte_for_byte():
    """The three `g4dat_bad` fixtures are corpus members under their own names, unchanged.

    They predate this corpus and three older tests assert against their exact bytes. Copying rather
    than regenerating them is deliberate; this test is what stops the copy from drifting into a
    second source of truth. Their verdicts are recomputed from the ORIGINALS, so the row is checked
    against the fixture rather than against the copy of itself.
    """
    generator = conformance_generator()
    rows = expected_rows(CORPUS)

    originals = sorted(BAD_FIXTURES.glob("*.g4dat"))
    assert originals, BAD_FIXTURES
    for original in originals:
        member = CORPUS / original.name
        assert member.exists(), f"{original.name} is not a corpus member"
        assert member.read_bytes() == original.read_bytes(), original.name
        assert rows[original.name] == generator.reference_verdict(original), original.name
