"""Documentation integrity: a document this repository names must be a document it contains.

A citation is a promise that the reader can go and look. `MODEL_SPEC.md sec.4` is checkable; the
same sentence naming a document that is not here is not, and the reader has no way to tell the two
apart -- they are identical in shape. Manifests, byte-diffs and `provenance --check`
cannot see this class at all: they bind numbers to the documents that render them, and say nothing
about a document that was never here.

**Scope, exactly, so this docstring cannot be read as promising more than it delivers.** The guard
matches filenames that carry the ``.md`` suffix, and only those. A reference written as an
identifier without a suffix -- a section id, a project-internal shorthand, the name of a document
given without its extension -- is INVISIBLE to it, and no amount of tightening the regex would
change that: those forms are not distinguishable from ordinary prose. Reference hygiene for
suffix-free forms is not this test's job and is not claimed by it.

What it does buy, mechanically and for every tracked text file: no `*.md` citation can enter this
repository pointing at a document the repository does not ship, and none can be orphaned later by
a rename or a deletion -- the same edit that removes the file fails this test. "Tracked" is meant
literally: the file list comes from `git ls-files`, not from a walk of the working tree, because an
untracked scratch file would otherwise satisfy a citation that no clone can resolve.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Directories that are not part of the tracked tree: environments, caches and build output.
_SKIP_DIRS = {
    ".git", ".venv", ".venv-win", "__pycache__", ".pytest_cache", ".ruff_cache",
    "build", "dist", "node_modules",
}

#: A Markdown filename. The trailing lookahead is load-bearing: without it the ``md5`` attribute
#: access in ``openmucf/g4/emit.py`` reads as a citation of a document named for the module
#: (measured -- it was this guard's only false positive on its first run over the tree).
_CITATION = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*\.md(?![A-Za-z0-9])")

#: Deliberate exceptions. An entry is a (filename, reason) pair, and the reason is the point: a
#: citation that cannot resolve has to be defended in words, not merely tolerated. Empty today,
#: and a change that needs to add one should be read as a finding first.
ALLOWED_UNRESOLVED: dict[str, str] = {}


def _tracked_text_files() -> list[Path]:
    """The files git actually publishes, when git can say so.

    The distinction is not pedantic: an UNTRACKED file satisfies a walk of the working tree while
    shipping nothing, so a citation could be "resolved" by a scratch file that no clone will ever
    have -- the guard would be green about a document the repository does not contain. `git ls-files`
    is therefore the primary source. The walk survives only as the fallback for a source tree
    unpacked without its history (an sdist), where every present file is by definition a shipped one,
    and :func:`_listing_is_from_git` reports which of the two ran.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files"],
            capture_output=True, text=True, check=True, timeout=60,
        )
        listed = [REPO / line for line in out.stdout.splitlines() if line]
        if listed:
            return listed
    except (OSError, subprocess.SubprocessError):
        pass
    files = []
    for root, dirs, fnames in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info")]
        files.extend(Path(root) / f for f in fnames)
    return files


def _listing_is_from_git() -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files"],
            capture_output=True, text=True, check=True, timeout=60,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def test_every_cited_markdown_document_exists() -> None:
    """Every ``*.md`` filename written anywhere in the tree resolves to a file in the tree."""
    files = _tracked_text_files()
    shipped = {p.name for p in files if p.suffix == ".md"}
    assert shipped, "no Markdown documents found -- the walk is broken, not the tree"

    unresolved: dict[str, set[str]] = {}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: nothing to cite from
        # BibTeX escapes the underscore, so FORMAT\_SPEC.md is one citation, not a miss.
        for name in _CITATION.findall(text.replace("\\_", "_")):
            if name not in shipped and name not in ALLOWED_UNRESOLVED:
                unresolved.setdefault(name, set()).add(str(path.relative_to(REPO)))

    assert not unresolved, (
        "these documents are cited but not shipped: "
        + "; ".join(f"{n} (in {', '.join(sorted(w))})" for n, w in sorted(unresolved.items()))
    )


def test_the_guard_would_notice_a_broken_citation() -> None:
    """The check above is only worth its runtime if it can fail; prove that on a synthetic input.

    A test that has never been seen to go red is an assertion about the tree AND an unverified
    assertion about itself. This one exercises the second half.
    """
    absent = "NOT_A_SHIPPED_DOCUMENT" + ".md"  # assembled, so this file cites nothing unresolvable
    shipped = {"README.md"}
    text = f"see README.md and {absent}, but not the md5 attribute of a hashing module"
    found = {n for n in _CITATION.findall(text) if n not in shipped}
    assert found == {absent}, found
    assert not [n for n in _CITATION.findall("hashlib.md5(b'x')") if n], "md5 must not read as a citation"


def test_the_file_list_comes_from_git_not_from_a_directory_walk() -> None:
    """An untracked file must not be able to satisfy a citation.

    This is the property the module docstring claims, and it is the one a directory walk quietly
    fails: an untracked scratch document dropped into the tree makes a walk-based guard report the
    repository as shipping it. Asserted here rather than described, because the walk is still
    present as a fallback and nothing else would notice if it became the primary path again.
    """
    if not _listing_is_from_git():
        import pytest

        pytest.skip("no git history here (sdist); the walk fallback is the documented behaviour")
    listed = {p.resolve() for p in _tracked_text_files()}
    # Assembled, so this file does not itself cite a document the tree does not contain.
    stub = REPO / ("UNTRACKED_STUB_FOR_THIS_TEST" + ".md")
    stub.write_text("scratch", encoding="utf-8")
    try:
        assert stub.resolve() not in {p.resolve() for p in _tracked_text_files()}
        assert stub.resolve() not in listed
    finally:
        stub.unlink()


def test_every_allowed_exception_carries_a_reason() -> None:
    """An exception without a written reason is a silent hole; there is no way to add one."""
    for name, reason in ALLOWED_UNRESOLVED.items():
        assert reason.strip(), f"{name} is exempted with no reason given"
