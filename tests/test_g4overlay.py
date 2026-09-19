"""T-72, T-73 -- the Geant4 overlay patches: do they still say what the repository says?

`cpp/patches/g4-v11.4.2-muonicdata.patch` and `cpp/patches/g4-v11.5.0.beta-muonicdata.patch` are
where the reader meets Geant4, one patch family per revision the overlay targets. Each adds the
reader to Geant4's own tree, adds the opt-in boolean to `G4HadronicParameters`, and inserts a
lookup into the two compiled-in copies of the muon-capture tables. Two things about a family can
rot silently: the reader it carries can drift from `cpp/include` + `cpp/src` (the repository's
copy is the one every other test exercises), and its context lines can drift from the vendored
upstream files it was cut against (then it no longer applies where it claims to). Neither needs
Geant4 to check, so both are checked here, for both families, on every platform, in ordinary CI.

What each test here is actually for:

* **T-72** -- the patch parses as a unified diff, touches exactly the declared set of files, applies
  (with a zero-fuzz applier written here, not `git apply`, so the check is the same on every
  runner) to the vendored copies of the two seam files without deleting a line of any file that
  existed before it, and the two reader files it adds equal the repository's byte for byte. The
  registration patch touches only the dataset-definitions file and adds exactly the committed
  snippet's entry.
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
README = PATCHES / "README.md"
SNIPPET = REPO / "data" / "g4" / "d1" / "geant4_add_dataset.snippet"

#: The two patch families, keyed by the upstream tag each was cut against: the behaviour patch,
#: the registration patch, and the directory holding that tag's vendored seam files.
FAMILIES: dict[str, tuple[pathlib.Path, pathlib.Path, pathlib.Path]] = {
    tag: (
        PATCHES / f"g4-{tag}-muonicdata.patch",
        PATCHES / f"g4-{tag}-register-dataset.patch",
        REPO / "third_party" / "geant4" / tag,
    )
    for tag in (parity.d1.UPSTREAM_TAG, parity.BETA_TAG)
}
#: Every patch of every family, in tag order -- what the README's digit-bearing tokens are held to.
ALL_PATCHES = [patch for tag in sorted(FAMILIES) for patch in FAMILIES[tag][:2]]

#: The files the behaviour patch is declared to touch, as a literal set: two reader files and one
#: glue file added to Geant4's particle-management module, that module's source list, the two
#: seam files, and the two files of `G4HadronicParameters` that gain the opt-in. A count would let
#: a dropped file and an added file cancel out.
BEHAVIOUR_PATHS = frozenset(
    {
        "source/particles/management/include/G4MuonicDataOverlay.hh",
        "source/particles/management/include/G4MuonicDataTable.hh",
        "source/particles/management/sources.cmake",
        "source/particles/management/src/G4MuonicAtomHelper.cc",
        "source/particles/management/src/G4MuonicDataOverlay.cc",
        "source/particles/management/src/G4MuonicDataTable.cc",
        "source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc",
        "source/processes/hadronic/util/include/G4HadronicParameters.hh",
        "source/processes/hadronic/util/src/G4HadronicParameters.cc",
    }
)
#: The one source list among them, derived from the declared set rather than re-typed; unpacking
#: a one-element tuple asserts there is exactly one.
(SOURCES_CMAKE,) = tuple(p for p in BEHAVIOUR_PATHS if p.endswith("/sources.cmake"))
#: The two `G4HadronicParameters` files, likewise derived: this repository vendors neither, so
#: only the shape of their `index` declaration is held (see the blob test).
HADRONIC_PARAMETERS = frozenset(p for p in BEHAVIOUR_PATHS if "/G4HadronicParameters." in p)
assert len(HADRONIC_PARAMETERS) == 2, HADRONIC_PARAMETERS
#: The two glue files, likewise derived: the same bytes in both families, and the one place the
#: profile variable is read.
GLUE = frozenset(p for p in BEHAVIOUR_PATHS if "/G4MuonicDataOverlay." in p)
assert len(GLUE) == 2, GLUE
(GLUE_CC,) = tuple(p for p in GLUE if p.endswith(".cc"))
#: Per family: the two seam files, and the vendored copy of that tag each one's hunks must apply to.
SEAMS: dict[str, dict[str, pathlib.Path]] = {
    tag: {
        "source/particles/management/src/G4MuonicAtomHelper.cc": vendored / "G4MuonicAtomHelper.cc",
        "source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc": (
            vendored / "G4MuonMinusBoundDecay.cc"
        ),
    }
    for tag, (_, _, vendored) in FAMILIES.items()
}
#: Per family: the reader and glue files the patch adds, and the repository file each must equal --
#: the same files for both families, since neither the reader nor the glue depends on the revision.
READER: dict[str, dict[str, pathlib.Path]] = {
    tag: {
        "source/particles/management/include/G4MuonicDataTable.hh": REPO / "cpp/include/G4MuonicDataTable.hh",
        "source/particles/management/src/G4MuonicDataTable.cc": REPO / "cpp/src/G4MuonicDataTable.cc",
        "source/particles/management/include/G4MuonicDataOverlay.hh": (
            REPO / "cpp/include/G4MuonicDataOverlay.hh"
        ),
        "source/particles/management/src/G4MuonicDataOverlay.cc": REPO / "cpp/src/G4MuonicDataOverlay.cc",
    }
    for tag in FAMILIES
}
REGISTRATION_PATH = "cmake/Modules/G4DatasetDefinitions.cmake"
#: Per tag: the upstream files this repository does not vendor, and git's own object name for each
#: at that tag's commit (`git rev-parse HEAD:<path>` on the pristine tree the patches were cut
#: against). These are pins, copied from that command's output: the `index` old id every patch
#: declares for such a file must be a prefix of one of them, or the patch was cut against other
#: bytes than the tag its name carries. What stays unheld is the post-image of such a file -- with
#: no `old` bytes to rebuild from, its `new` id is only required to differ.
PRISTINE_INDEX_OLD: dict[str, dict[str, str]] = {
    parity.d1.UPSTREAM_TAG: {
        "source/particles/management/sources.cmake": "e292ef656be716180f409eb8f0501246b42e1722",
        "source/processes/hadronic/util/include/G4HadronicParameters.hh": (
            "09d22d476dc9f2a66d6af7e29e41b25fb2000533"
        ),
        "source/processes/hadronic/util/src/G4HadronicParameters.cc": (
            "9a16ede106332b4c1bcb1e82c66aa04a6b729b2a"
        ),
        "cmake/Modules/G4DatasetDefinitions.cmake": "64feb989558c65a71ff33d6cdeb16cbc06944205",
    },
    parity.BETA_TAG: {
        "source/particles/management/sources.cmake": "e292ef656be716180f409eb8f0501246b42e1722",
        "source/processes/hadronic/util/include/G4HadronicParameters.hh": (
            "2d051de924af8ef3da06fb6445a957ca4fa50e52"
        ),
        "source/processes/hadronic/util/src/G4HadronicParameters.cc": (
            "5046e3461bf69940b23ea59a524b17efef31b0fa"
        ),
        "cmake/Modules/G4DatasetDefinitions.cmake": "fa46fc1956104cd857b3ab3e4bb0f30cb2474ecc",
    },
}
assert set(PRISTINE_INDEX_OLD) == set(FAMILIES), sorted(PRISTINE_INDEX_OLD)

family = pytest.mark.parametrize("tag", sorted(FAMILIES))


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


#: The one added line of `G4HadronicParameters.hh` that declares the opt-in member and its default.
MEMBER_FALSE = b"G4bool fEnableMuonicData = false;"


def added_lines(file: FilePatch) -> list[tuple[Hunk, bytes]]:
    """Every `+` line of every hunk of one file, with the hunk it belongs to."""
    return [(hunk, content) for hunk in file.hunks for marker, content, _ in hunk.lines if marker == b"+"]


def check_opt_in_shape(tag: str, files: dict[str, FilePatch]) -> None:
    """The opt-in as the patch README states it, over the `+` lines of the pre-existing files:
    `G4HadronicParameters.hh` gains exactly one line whose content is the member declared `false`;
    `G4MuonicDataTable::Enable()` is called from exactly one added line across every pre-existing
    file, in `G4HadronicParameters.cc`, in a hunk that adds a `SetEnableMuonicData(` line above it.
    Every message names the family, so a drill on one family says which."""
    (header,) = tuple(p for p in HADRONIC_PARAMETERS if p.endswith(".hh"))
    (source,) = tuple(p for p in HADRONIC_PARAMETERS if p.endswith(".cc"))
    members = [content.strip() for _, content in added_lines(files[header])]
    assert members.count(MEMBER_FALSE) == 1, (
        f"{tag}: {header} adds the member `false` {members.count(MEMBER_FALSE)} times"
    )
    callers = [
        (path, hunk)
        for path, file in sorted(files.items())
        if file.old_path != b"/dev/null"
        for hunk, content in added_lines(file)
        if b"G4MuonicDataTable::Enable()" in content
    ]
    assert len(callers) == 1, (
        f"{tag}: G4MuonicDataTable::Enable() is called from {len(callers)} added line(s): "
        f"{[(path, hunk.header.decode()) for path, hunk in callers]}"
    )
    ((path, hunk),) = callers
    assert path == source, f"{tag}: the only caller of Enable() is in {path}, not {source}"
    added = [content for _, content in added_lines(FilePatch(b"", b"", [hunk]))]
    index = next(i for i, content in enumerate(added) if b"G4MuonicDataTable::Enable()" in content)
    assert any(b"SetEnableMuonicData(" in content for content in added[:index]), (
        f"{tag}: no added `SetEnableMuonicData(` line precedes the Enable() call in hunk "
        f"{hunk.header.decode()}"
    )


def check_pristine_pin(tag: str, path: str, file: FilePatch, pins: dict[str, dict[str, str]]) -> None:
    """The `index` old id of a file this repository does not vendor is a prefix of the pinned
    object name of that file on the pristine tree of `tag`; the message names the path."""
    pristine = pins[tag][path]
    assert pristine.startswith(file.index_old.decode()), (
        f"{tag}: {path} declares index old {file.index_old.decode()}, the pristine tree holds {pristine}"
    )


# T-72 -- the behaviour patch and the registration patch say what the repository says
# ---------------------------------------------------------------------------------------


@family
def test_t72_the_behaviour_patch_touches_exactly_the_declared_files(tag: str):
    behaviour, _, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(behaviour.read_bytes()))
    assert set(files) == BEHAVIOUR_PATHS


@family
def test_t72_the_seam_hunks_apply_to_the_vendored_files_and_delete_nothing(tag: str):
    behaviour, _, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(behaviour.read_bytes()))
    # Every file that existed before the patch -- the seams, the source list, the two
    # `G4HadronicParameters` files -- is only ever added to: not one `-` line anywhere.
    for path, file in sorted(files.items()):
        if file.old_path == b"/dev/null":
            continue
        assert file.old_path == b"a/" + path.encode(), file.old_path
        removed = [content for hunk in file.hunks for marker, content, _ in hunk.lines if marker == b"-"]
        assert not removed, f"{path}: the patch deletes {removed}"
    for path, vendored in SEAMS[tag].items():
        file = files[path]
        original = vendored.read_bytes()
        patched = apply_file_patch(file, original)
        added = sum(1 for hunk in file.hunks for marker, _, _ in hunk.lines if marker == b"+")
        assert added > 0, f"{path}: the patch adds nothing"
        assert len(split_lines(patched)) == len(split_lines(original)) + added


@family
def test_t72_the_added_reader_files_equal_the_repository_copies_byte_for_byte(tag: str):
    behaviour, _, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(behaviour.read_bytes()))
    for path, repo_file in READER[tag].items():
        file = files[path]
        assert file.old_path == b"/dev/null", f"{path} is not added as a new file"
        added = apply_file_patch(file, b"")
        # The repository copies are `text` files, so an autocrlf checkout holds them as CRLF; that,
        # and nothing else, is normalised before the comparison.
        expected = repo_file.read_bytes().replace(b"\r\n", b"\n")
        assert added == expected, f"{path} differs from {repo_file.name}"


@family
def test_t72_neither_patch_carries_a_carriage_return(tag: str):
    behaviour, registration, _ = FAMILIES[tag]
    check_no_carriage_return(behaviour)
    check_no_carriage_return(registration)


@family
def test_t72_the_registration_patch_adds_exactly_the_snippets_entry(tag: str):
    _, registration, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(registration.read_bytes()))
    assert set(files) == {REGISTRATION_PATH}
    file = files[REGISTRATION_PATH]
    markers = {marker for hunk in file.hunks for marker, _, _ in hunk.lines}
    assert markers <= {b" ", b"+"}, markers
    added = [content for hunk in file.hunks for marker, content, _ in hunk.lines if marker == b"+"]
    expected = [line for line, _ in split_lines(SNIPPET.read_bytes()) if not line.startswith(b"#")]
    assert added == expected


@family
def test_t72_every_file_a_patch_touches_rebuilds_to_the_blob_its_index_line_declares(tag: str):
    """Each `index <old>..<new>` line is git's own name for the file's bytes before and after the
    hunks. The pin is on the whole post-image, so a hunk that dropped an added line together with
    its count -- which the hunk arithmetic cannot see -- moves the blob id and fails here.

    An added file rebuilds from nothing to `new`; a seam file's vendored copy is `old` and its
    patched copy is `new`. The upstream files this repository does not vendor (`sources.cmake`,
    `G4DatasetDefinitions.cmake`, the two `G4HadronicParameters` files) have no `old` bytes to
    rebuild from: for them the declared `old` is held to the object name `PRISTINE_INDEX_OLD` pins
    for that file on the tag's pristine tree, and a `new` that differs from it is required. What
    stays unheld is the post-image of such a file -- nothing here rebuilds it.
    """
    behaviour, registration, _ = FAMILIES[tag]
    seams = SEAMS[tag]
    not_rebuilt: set[str] = set()
    for patch in (behaviour, registration):
        for path, file in sorted(by_new_path(parse_patch(patch.read_bytes())).items()):
            assert re.fullmatch(rb"[0-9a-f]{7,40}", file.index_old), (path, file.index_old)
            assert re.fullmatch(rb"[0-9a-f]{7,40}", file.index_new), (path, file.index_new)
            new = file.index_new.decode()
            if file.old_path == b"/dev/null":
                assert file.index_old.strip(b"0") == b"", (path, file.index_old)
                assert parity.git_blob_id(apply_file_patch(file, b"")).startswith(new), path
            elif path in seams:
                vendored = seams[path].read_bytes()
                assert parity.git_blob_id(vendored).startswith(file.index_old.decode()), path
                assert parity.git_blob_id(apply_file_patch(file, vendored)).startswith(new), path
            else:
                check_pristine_pin(tag, path, file, PRISTINE_INDEX_OLD)
                assert file.index_new != file.index_old, path
                not_rebuilt.add(path)
    assert not_rebuilt == {SOURCES_CMAKE, REGISTRATION_PATH} | HADRONIC_PARAMETERS
    assert not_rebuilt == set(PRISTINE_INDEX_OLD[tag]), sorted(PRISTINE_INDEX_OLD[tag])


def test_t72_the_glue_files_are_identical_across_families_and_read_the_profile_variable():
    """The glue `.hh` and `.cc`, rebuilt from each family's added-file hunks, are byte-identical
    across the two families -- the glue does not depend on the revision -- and the `.cc` reads
    `G4MUONICDATA_PROFILE` at exactly one place and raises the profile error code at exactly one.
    """
    rebuilt: dict[str, dict[str, bytes]] = {}
    for tag in sorted(FAMILIES):
        behaviour, _, _ = FAMILIES[tag]
        files = by_new_path(parse_patch(behaviour.read_bytes()))
        rebuilt[tag] = {}
        for path in sorted(GLUE):
            assert files[path].old_path == b"/dev/null", (tag, path)
            rebuilt[tag][path] = apply_file_patch(files[path], b"")
    first, second = sorted(rebuilt)
    for path in sorted(GLUE):
        assert rebuilt[first][path] == rebuilt[second][path], f"{path} differs between {first} and {second}"
    glue_cc = rebuilt[first][GLUE_CC]
    assert glue_cc.count(b'std::getenv("G4MUONICDATA_PROFILE")') == 1, glue_cc.count(b"G4MUONICDATA_PROFILE")
    assert glue_cc.count(b'"G4MuonicData004"') == 1, glue_cc.count(b"G4MuonicData004")


@family
def test_t72_the_opt_in_starts_false_and_its_setter_is_the_only_caller_of_enable(tag: str):
    """The sentence of the patch README that the opt-in is off by default and its setter is the
    only caller of `G4MuonicDataTable::Enable()`, held to the behaviour patch's own `+` lines."""
    behaviour, _, _ = FAMILIES[tag]
    check_opt_in_shape(tag, by_new_path(parse_patch(behaviour.read_bytes())))


def test_t72_the_vendored_readme_names_the_seam_paths_the_behaviour_patch_touches():
    """The vendored README's `upstream path` cells are the seam paths, read from its table by the
    row labels: the BoundDecay cell is the reference module's `UPSTREAM_PATH`, the helper cell is
    the `SEAMS` key whose vendored copy is the helper, and together they are the paths the
    behaviour patch's seam hunks touch. The rows read are the `v11.4.2` table's; the beta family
    touches the same two paths, which the path-set test above holds for both.
    """
    seams = SEAMS[parity.d1.UPSTREAM_TAG]
    text = parity.VENDORED_README.read_text("utf-8")
    bound_decay = re.findall(r"^\| upstream path \| `([^`]+)` \|$", text, re.M)
    helper = re.findall(r"^\| `G4MuonicAtomHelper\.cc` upstream path \| `([^`]+)` \|$", text, re.M)
    assert len(bound_decay) == 1 and len(helper) == 1, (bound_decay, helper)
    assert bound_decay[0] == parity.d1.UPSTREAM_PATH
    helper_key = next(path for path, vendored in seams.items() if vendored == parity.HELPER)
    assert helper[0] == helper_key
    assert {bound_decay[0], helper[0]} == set(seams)


# T-73 -- the drill, and the README's numbers
# -------------------------------------------


@family
def test_t73_drill_an_altered_context_line_is_refused_by_name(tag: str):
    """Alter one context line of a seam hunk in memory: the applier must raise, naming that hunk.

    This is what separates an applier from a path matcher: a checker that only compared the `+++`
    set would pass a patch cut against some other revision of the same file names.
    """
    behaviour, _, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(behaviour.read_bytes()))
    path, vendored = next(iter(sorted(SEAMS[tag].items())))
    file = files[path]
    hunk = file.hunks[-1]
    index = next(i for i, (marker, _, _) in enumerate(hunk.lines) if marker == b" ")
    marker, content, has_newline = hunk.lines[index]
    hunk.lines[index] = (marker, content + b"x", has_newline)
    with pytest.raises(PatchError) as raised:
        apply_file_patch(file, vendored.read_bytes())
    assert hunk.header.decode() in str(raised.value), raised.value
    assert path in str(raised.value), raised.value


@family
def test_t73_drill_a_dropped_added_line_with_its_count_is_refused_by_the_blob_pin(tag: str):
    """Drop the last added line of the reader's `.cc` hunk and lower the hunk's count to match: the
    hunk arithmetic still balances and the applier still succeeds, so nothing before the blob pin
    notices -- and the rebuilt file no longer names the blob the `index` line declares.
    """
    behaviour, _, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(behaviour.read_bytes()))
    path = next(p for p in READER[tag] if p.endswith(".cc"))
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
    check_readme_numbers(README, ALL_PATCHES)


@family
def test_t73_drill_a_dropped_member_default_and_a_second_enable_caller_are_refused_by_family(tag: str):
    """Two in-memory corruptions of the behaviour patch, each refused with the family named: the
    `+` line declaring the member `false` removed (with the hunk's count lowered so the
    arithmetic still balances), and a second `+` line calling `G4MuonicDataTable::Enable()`
    planted in a seam hunk."""
    behaviour, _, _ = FAMILIES[tag]
    (header,) = tuple(p for p in HADRONIC_PARAMETERS if p.endswith(".hh"))

    files = by_new_path(parse_patch(behaviour.read_bytes()))
    check_opt_in_shape(tag, files)
    hunk = next(
        hunk
        for hunk in files[header].hunks
        for marker, content, _ in hunk.lines
        if marker == b"+" and content.strip() == MEMBER_FALSE
    )
    index = next(
        i for i, (_, content, _) in enumerate(hunk.lines) if content.strip() == MEMBER_FALSE
    )
    del hunk.lines[index]
    hunk.new_count -= 1
    with pytest.raises(AssertionError, match=re.escape(tag)) as raised:
        check_opt_in_shape(tag, files)
    assert "0 times" in str(raised.value), raised.value

    files = by_new_path(parse_patch(behaviour.read_bytes()))
    path, _ = next(iter(sorted(SEAMS[tag].items())))
    seam_hunk = files[path].hunks[-1]
    seam_hunk.lines.append((b"+", b"  G4MuonicDataTable::Enable();", True))
    seam_hunk.new_count += 1
    with pytest.raises(AssertionError, match=re.escape(tag)) as raised:
        check_opt_in_shape(tag, files)
    assert "2 added line(s)" in str(raised.value), raised.value


@family
def test_t73_drill_an_altered_pristine_pin_is_refused_by_path(tag: str):
    """One pinned pristine object name altered in its first hex digit -- inside the abbreviated
    prefix the patch declares -- so the pin check fails and names that path, for every file the
    pin covers."""
    behaviour, registration, _ = FAMILIES[tag]
    files: dict[str, FilePatch] = {}
    for patch in (behaviour, registration):
        files.update(by_new_path(parse_patch(patch.read_bytes())))
    for path, pinned in sorted(PRISTINE_INDEX_OLD[tag].items()):
        check_pristine_pin(tag, path, files[path], PRISTINE_INDEX_OLD)
        altered = ("0" if pinned[0] != "0" else "1") + pinned[1:]
        pins = {tag: {**PRISTINE_INDEX_OLD[tag], path: altered}}
        with pytest.raises(AssertionError, match=re.escape(path)):
            check_pristine_pin(tag, path, files[path], pins)


@family
def test_t73_drill_a_carriage_return_and_a_planted_number_are_caught(tag: str, tmp_path):
    """The two README/patch guards, each shown to fire on a corrupted temporary copy."""
    behaviour, registration, _ = FAMILIES[tag]
    patch_copy = tmp_path / behaviour.name
    patch_copy.write_bytes(behaviour.read_bytes().replace(b"\n", b"\r\n", 1))
    with pytest.raises(AssertionError, match="CR byte"):
        check_no_carriage_return(patch_copy)

    readme_copy = tmp_path / README.name
    planted = "36000"
    for patch in ALL_PATCHES:
        assert planted not in patch.read_bytes().decode("latin-1"), patch.name
    readme_copy.write_text(README.read_text("utf-8") + f"\nThe sweep has {planted} points.\n", "utf-8")
    with pytest.raises(AssertionError, match=planted):
        check_readme_numbers(readme_copy, [behaviour, registration])
