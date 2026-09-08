"""T-72, T-73 -- the Geant4 overlay patches: do they still say what the repository says?

`cpp/patches/g4-v11.4.2-muonicdata.patch` is where the reader meets Geant4. It adds the reader to
Geant4's own tree and inserts a lookup into the two compiled-in copies of the muon-capture tables.
Two things about it can rot silently: the reader it carries can drift from `cpp/include` + `cpp/src`
(the repository's copy is the one every other test exercises), and its context lines can drift from
the vendored upstream files it was cut against (then it no longer applies where it claims to).
Neither needs Geant4 to check, so both are checked here, on every platform, in ordinary CI.

What each test here is actually for:

* **T-72** -- the patch parses as a unified diff, touches exactly the declared set of files, applies
  (with a zero-fuzz applier written here, not `git apply`, so the check is the same on every
  runner) to the vendored copies of the two seam files without deleting a line of them, and the
  two reader files it adds equal the repository's byte for byte. The registration patch touches
  only the dataset-definitions file and adds exactly the committed snippet's entry.
* **T-73** -- the drill: alter one context line and the applier must refuse, naming the hunk; and
  the patch README states the sweep digest by reference, never as a literal, and carries no
  digit-bearing token the patches themselves do not.

Every path set and every expected line here is read from the repository -- the patch files, the
vendored sources, the snippet -- never typed as a number.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re

import pytest
import test_g4parity as parity

REPO = pathlib.Path(__file__).resolve().parents[1]
PATCHES = REPO / "cpp" / "patches"
BEHAVIOUR = PATCHES / "g4-v11.4.2-muonicdata.patch"
REGISTRATION = PATCHES / "g4-v11.4.2-register-dataset.patch"
README = PATCHES / "README.md"
VENDORED = REPO / "third_party" / "geant4" / "v11.4.2"
SNIPPET = REPO / "data" / "g4" / "d1" / "geant4_add_dataset.snippet"

#: The files the behaviour patch is declared to touch, as a literal set: two reader files and one
#: glue file added to Geant4's global-management module, that module's source list, and the two
#: seam files. A count would let a dropped file and an added file cancel out.
BEHAVIOUR_PATHS = frozenset(
    {
        "source/global/management/include/G4MuonicDataOverlay.hh",
        "source/global/management/include/G4MuonicDataTable.hh",
        "source/global/management/sources.cmake",
        "source/global/management/src/G4MuonicDataOverlay.cc",
        "source/global/management/src/G4MuonicDataTable.cc",
        "source/particles/management/src/G4MuonicAtomHelper.cc",
        "source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc",
    }
)
#: The two seam files, and the vendored copy each one's hunks must apply to.
SEAMS = {
    "source/particles/management/src/G4MuonicAtomHelper.cc": VENDORED / "G4MuonicAtomHelper.cc",
    "source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc": VENDORED / "G4MuonMinusBoundDecay.cc",
}
#: The two reader files the patch adds, and the repository file each must equal.
READER = {
    "source/global/management/include/G4MuonicDataTable.hh": REPO / "cpp/include/G4MuonicDataTable.hh",
    "source/global/management/src/G4MuonicDataTable.cc": REPO / "cpp/src/G4MuonicDataTable.cc",
}
REGISTRATION_PATH = "cmake/Modules/G4DatasetDefinitions.cmake"


# A unified-diff applier -- zero fuzz, bytes in, bytes out
# ----------------------------------------------------------


class PatchError(Exception):
    """The patch does not parse, or a hunk does not match the file it is applied to."""


@dataclasses.dataclass
class Hunk:
    header: bytes
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    #: (marker, content, has_newline); marker is b" ", b"-" or b"+".
    lines: list[tuple[bytes, bytes, bool]]


@dataclasses.dataclass
class FilePatch:
    old_path: bytes  # the `---` operand: b"a/<path>" or b"/dev/null"
    new_path: bytes  # the `+++` operand: b"b/<path>"
    hunks: list[Hunk]
    #: The `index <old>..<new>` operands: git's abbreviated blob ids of the file before and after
    #: the hunks, as the patch itself declares them; an added file's `old` is all zeros.
    index_old: bytes = b""
    index_new: bytes = b""


_HUNK = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def split_lines(raw: bytes) -> list[tuple[bytes, bool]]:
    """``[(content, has_newline)]``, splitting on LF only -- a CR is content, never a line end."""
    parts = raw.split(b"\n")
    lines = [(part, True) for part in parts[:-1]]
    if parts[-1]:
        lines.append((parts[-1], False))
    return lines


def parse_patch(raw: bytes) -> list[FilePatch]:
    """Parse a `git diff` unified patch. Header lines between `diff --git` and `---` are skipped."""
    files: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None
    remaining_old = remaining_new = 0
    for number, (line, _) in enumerate(split_lines(raw), 1):
        if line.startswith(b"diff --git "):
            current = FilePatch(b"", b"", [])
            files.append(current)
            hunk = None
            continue
        if current is None:
            raise PatchError(f"line {number}: content before the first 'diff --git' header")
        if hunk is not None and (remaining_old or remaining_new):
            if line.startswith(b"\\"):
                # "\ No newline at end of file" qualifies the line before it.
                marker, content, _ = hunk.lines[-1]
                hunk.lines[-1] = (marker, content, False)
                continue
            marker, content = line[:1], line[1:]
            if marker not in (b" ", b"-", b"+"):
                where = f"{current.new_path.decode()}: hunk {hunk.header.decode()}: line {number}"
                raise PatchError(f"{where}: bad marker {line!r}")
            hunk.lines.append((marker, content, True))
            if marker != b"+":
                remaining_old -= 1
            if marker != b"-":
                remaining_new -= 1
            continue
        if line.startswith(b"\\") and hunk is not None:
            marker, content, _ = hunk.lines[-1]
            hunk.lines[-1] = (marker, content, False)
            continue
        if line.startswith(b"--- "):
            current.old_path = line[4:]
            continue
        if line.startswith(b"+++ "):
            current.new_path = line[4:]
            continue
        if line.startswith(b"index "):
            current.index_old, _, current.index_new = line[6:].split(b" ")[0].partition(b"..")
            continue
        match = _HUNK.match(line)
        if match:
            old_start, old_count, new_start, new_count = (
                int(match.group(1)),
                int(match.group(2) or 1),
                int(match.group(3)),
                int(match.group(4) or 1),
            )
            hunk = Hunk(line, old_start, old_count, new_start, new_count, [])
            current.hunks.append(hunk)
            remaining_old, remaining_new = old_count, new_count
            continue
        # `new file mode ...` and any other header line: ignored.
    for file in files:
        for h in file.hunks:
            olds = sum(1 for marker, _, _ in h.lines if marker != b"+")
            news = sum(1 for marker, _, _ in h.lines if marker != b"-")
            if (olds, news) != (h.old_count, h.new_count):
                where = f"{file.new_path.decode()}: hunk {h.header.decode()}"
                raise PatchError(f"{where}: declares -{h.old_count} +{h.new_count}, carries -{olds} +{news}")
    return files


def apply_file_patch(file: FilePatch, original: bytes) -> bytes:
    """Apply every hunk at its stated old start, zero fuzz, tracking the running line offset.

    Any context or `-` line that is not exactly the file's line at that position raises
    :class:`PatchError` naming the file and the hunk header -- the drill in T-73 relies on that.
    """
    lines = split_lines(original)
    offset = 0
    name = file.new_path.decode()
    for hunk in file.hunks:
        position = max(hunk.old_start - 1, 0) + offset
        for marker, content, has_newline in hunk.lines:
            if marker == b"+":
                lines.insert(position, (content, has_newline))
                position += 1
                continue
            where = f"{name}: hunk {hunk.header.decode()}"
            if position >= len(lines):
                raise PatchError(f"{where}: expected {content!r} past the end of the file")
            if lines[position] != (content, has_newline):
                raise PatchError(f"{where}: expected {content!r}, found {lines[position][0]!r}")
            if marker == b"-":
                del lines[position]
            else:
                position += 1
        offset += hunk.new_count - hunk.old_count
    return b"".join(content + (b"\n" if has_newline else b"") for content, has_newline in lines)


def by_new_path(files: list[FilePatch]) -> dict[str, FilePatch]:
    """``{path: FilePatch}`` keyed by the `+++` operand with its `b/` prefix removed."""
    out: dict[str, FilePatch] = {}
    for file in files:
        assert file.new_path.startswith(b"b/"), file.new_path
        out[file.new_path[2:].decode()] = file
    return out


# The checks, as functions of paths so the drills can point them at a corrupted copy
# --------------------------------------------------------------------------------------


def check_no_carriage_return(path: pathlib.Path) -> None:
    """(d) a patch file carries no CR byte: its added-file hunks are LF text compared byte for byte."""
    raw = path.read_bytes()
    offset = raw.find(b"\r")
    assert offset < 0, f"{path.name} carries a CR byte at offset {offset}"


def readme_tokens_with_digits(readme: pathlib.Path) -> list[str]:
    """Every whitespace-delimited token of the README that contains a digit, trimmed of backticks,
    a leading parenthesis and trailing punctuation."""
    tokens = []
    for raw in readme.read_text("utf-8").split():
        token = raw.strip("`").lstrip("(").rstrip(".,;:!?)").strip("`")
        if any(ch.isdigit() for ch in token):
            tokens.append(token)
    return tokens


def check_readme_numbers(readme: pathlib.Path, patches: list[pathlib.Path]) -> None:
    """(b) no 64-hex literal; (c) every digit-bearing token occurs in a patch's text or file name."""
    text = readme.read_text("utf-8")
    assert re.search(r"[0-9a-f]{64}", text) is None, "the README states a digest as a literal"
    corpus = "\n".join(p.read_bytes().decode("latin-1") for p in patches)
    names = " ".join(p.name for p in patches)
    foreign = [t for t in readme_tokens_with_digits(readme) if t not in corpus and t not in names]
    assert not foreign, f"README tokens the patches do not carry: {foreign}"


# T-72 -- the behaviour patch and the registration patch say what the repository says
# ---------------------------------------------------------------------------------------


def test_t72_the_behaviour_patch_touches_exactly_the_declared_files():
    files = by_new_path(parse_patch(BEHAVIOUR.read_bytes()))
    assert set(files) == BEHAVIOUR_PATHS


def test_t72_the_seam_hunks_apply_to_the_vendored_files_and_delete_nothing():
    files = by_new_path(parse_patch(BEHAVIOUR.read_bytes()))
    for path, vendored in SEAMS.items():
        file = files[path]
        assert file.old_path == b"a/" + path.encode(), file.old_path
        removed = [content for hunk in file.hunks for marker, content, _ in hunk.lines if marker == b"-"]
        assert not removed, f"{path}: the patch deletes {removed}"
        original = vendored.read_bytes()
        patched = apply_file_patch(file, original)
        added = sum(1 for hunk in file.hunks for marker, _, _ in hunk.lines if marker == b"+")
        assert added > 0, f"{path}: the patch adds nothing"
        assert len(split_lines(patched)) == len(split_lines(original)) + added


def test_t72_the_added_reader_files_equal_the_repository_copies_byte_for_byte():
    files = by_new_path(parse_patch(BEHAVIOUR.read_bytes()))
    for path, repo_file in READER.items():
        file = files[path]
        assert file.old_path == b"/dev/null", f"{path} is not added as a new file"
        added = apply_file_patch(file, b"")
        # The repository copies are `text` files, so an autocrlf checkout holds them as CRLF; that,
        # and nothing else, is normalised before the comparison.
        expected = repo_file.read_bytes().replace(b"\r\n", b"\n")
        assert added == expected, f"{path} differs from {repo_file.name}"


def test_t72_neither_patch_carries_a_carriage_return():
    check_no_carriage_return(BEHAVIOUR)
    check_no_carriage_return(REGISTRATION)


def test_t72_the_registration_patch_adds_exactly_the_snippets_entry():
    files = by_new_path(parse_patch(REGISTRATION.read_bytes()))
    assert set(files) == {REGISTRATION_PATH}
    file = files[REGISTRATION_PATH]
    markers = {marker for hunk in file.hunks for marker, _, _ in hunk.lines}
    assert markers <= {b" ", b"+"}, markers
    added = [content for hunk in file.hunks for marker, content, _ in hunk.lines if marker == b"+"]
    expected = [line for line, _ in split_lines(SNIPPET.read_bytes()) if not line.startswith(b"#")]
    assert added == expected


def test_t72_every_file_a_patch_touches_rebuilds_to_the_blob_its_index_line_declares():
    """Each `index <old>..<new>` line is git's own name for the file's bytes before and after the
    hunks. The pin is on the whole post-image, so a hunk that dropped an added line together with
    its count -- which the hunk arithmetic cannot see -- moves the blob id and fails here.

    An added file rebuilds from nothing to `new`; a seam file's vendored copy is `old` and its
    patched copy is `new`. The upstream files this repository does not vendor (`sources.cmake`,
    `G4DatasetDefinitions.cmake`) have no `old` bytes to rebuild from, so for them only the shape of
    the declaration is held: a non-zero `old`, and a `new` that differs from it.
    """
    not_rebuilt: set[str] = set()
    for patch in (BEHAVIOUR, REGISTRATION):
        for path, file in sorted(by_new_path(parse_patch(patch.read_bytes())).items()):
            assert re.fullmatch(rb"[0-9a-f]{7,40}", file.index_old), (path, file.index_old)
            assert re.fullmatch(rb"[0-9a-f]{7,40}", file.index_new), (path, file.index_new)
            new = file.index_new.decode()
            if file.old_path == b"/dev/null":
                assert file.index_old.strip(b"0") == b"", (path, file.index_old)
                assert parity.git_blob_id(apply_file_patch(file, b"")).startswith(new), path
            elif path in SEAMS:
                vendored = SEAMS[path].read_bytes()
                assert parity.git_blob_id(vendored).startswith(file.index_old.decode()), path
                assert parity.git_blob_id(apply_file_patch(file, vendored)).startswith(new), path
            else:
                assert file.index_old.strip(b"0") != b"", (path, file.index_old)
                assert file.index_new != file.index_old, path
                not_rebuilt.add(path)
    assert not_rebuilt == {"source/global/management/sources.cmake", REGISTRATION_PATH}


def test_t72_the_vendored_readme_names_the_seam_paths_the_behaviour_patch_touches():
    """The vendored README's `upstream path` cells are the seam paths, read from its table by the
    row labels: the BoundDecay cell is the reference module's `UPSTREAM_PATH`, the helper cell is
    the `SEAMS` key whose vendored copy is the helper, and together they are the paths the
    behaviour patch's seam hunks touch.
    """
    text = parity.VENDORED_README.read_text("utf-8")
    bound_decay = re.findall(r"^\| upstream path \| `([^`]+)` \|$", text, re.M)
    helper = re.findall(r"^\| `G4MuonicAtomHelper\.cc` upstream path \| `([^`]+)` \|$", text, re.M)
    assert len(bound_decay) == 1 and len(helper) == 1, (bound_decay, helper)
    assert bound_decay[0] == parity.d1.UPSTREAM_PATH
    helper_key = next(path for path, vendored in SEAMS.items() if vendored == parity.HELPER)
    assert helper[0] == helper_key
    assert {bound_decay[0], helper[0]} == set(SEAMS)


# T-73 -- the drill, and the README's numbers
# -------------------------------------------


def test_t73_drill_an_altered_context_line_is_refused_by_name():
    """Alter one context line of a seam hunk in memory: the applier must raise, naming that hunk.

    This is what separates an applier from a path matcher: a checker that only compared the `+++`
    set would pass a patch cut against some other revision of the same file names.
    """
    files = by_new_path(parse_patch(BEHAVIOUR.read_bytes()))
    path, vendored = next(iter(sorted(SEAMS.items())))
    file = files[path]
    hunk = file.hunks[-1]
    index = next(i for i, (marker, _, _) in enumerate(hunk.lines) if marker == b" ")
    marker, content, has_newline = hunk.lines[index]
    hunk.lines[index] = (marker, content + b"x", has_newline)
    with pytest.raises(PatchError) as raised:
        apply_file_patch(file, vendored.read_bytes())
    assert hunk.header.decode() in str(raised.value), raised.value
    assert path in str(raised.value), raised.value


def test_t73_drill_a_dropped_added_line_with_its_count_is_refused_by_the_blob_pin():
    """Drop the last added line of the reader's `.cc` hunk and lower the hunk's count to match: the
    hunk arithmetic still balances and the applier still succeeds, so nothing before the blob pin
    notices -- and the rebuilt file no longer names the blob the `index` line declares.
    """
    files = by_new_path(parse_patch(BEHAVIOUR.read_bytes()))
    path = next(p for p in READER if p.endswith(".cc"))
    file = files[path]
    declared = file.index_new.decode()
    assert parity.git_blob_id(apply_file_patch(file, b"")).startswith(declared)
    hunk = file.hunks[-1]
    index = max(i for i, (marker, _, _) in enumerate(hunk.lines) if marker == b"+")
    del hunk.lines[index]
    hunk.new_count -= 1
    rebuilt = apply_file_patch(file, b"")
    assert rebuilt, "the drill emptied the file"
    assert not parity.git_blob_id(rebuilt).startswith(declared)


def test_t73_the_readme_states_no_digest_literal_and_no_foreign_number():
    check_readme_numbers(README, [BEHAVIOUR, REGISTRATION])


def test_t73_drill_a_carriage_return_and_a_planted_number_are_caught(tmp_path):
    """The two README/patch guards, each shown to fire on a corrupted temporary copy."""
    patch_copy = tmp_path / BEHAVIOUR.name
    patch_copy.write_bytes(BEHAVIOUR.read_bytes().replace(b"\n", b"\r\n", 1))
    with pytest.raises(AssertionError, match="CR byte"):
        check_no_carriage_return(patch_copy)

    readme_copy = tmp_path / README.name
    planted = "36000"
    assert planted not in BEHAVIOUR.read_bytes().decode("latin-1")
    assert planted not in REGISTRATION.read_bytes().decode("latin-1")
    readme_copy.write_text(README.read_text("utf-8") + f"\nThe sweep has {planted} points.\n", "utf-8")
    with pytest.raises(AssertionError, match=planted):
        check_readme_numbers(readme_copy, [BEHAVIOUR, REGISTRATION])
