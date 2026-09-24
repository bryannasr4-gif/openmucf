"""T-72, T-73 -- the Geant4 overlay patches: do they still say what the repository says?

`cpp/patches/g4-v11.4.2-muonicdata.patch` and `cpp/patches/g4-v11.5.0.beta-muonicdata.patch` are
where the reader meets Geant4, one patch family per revision the overlay targets. Each adds the
reader, its semantic layer and its glue to Geant4's own tree, adds the opt-in boolean to
`G4HadronicParameters`, and
inserts lookups into the two compiled-in copies of the muon-capture tables, the muonic cascade and
the helper's K energy. Two things about a family can
rot silently: the reader it carries can drift from `cpp/include` + `cpp/src` (the repository's
copy is the one every other test exercises), and its context lines can drift from the vendored
upstream files it was cut against (then it no longer applies where it claims to). Neither needs
Geant4 to check, so both are checked here, for both families, on every platform, in ordinary CI.

What each test here is actually for:

* **T-72** -- the patch parses as a unified diff, touches exactly the declared set of files, applies
  (with a zero-fuzz applier written here, not `git apply`, so the check is the same on every
  runner) to the vendored copies of the seam files without deleting a line of any file that
  existed before it but the lines it declares it changes, and the reader, semantic-layer and glue
  files it adds equal the repository's byte for byte. The
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
import math
import pathlib
import re
import struct

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

#: The files the behaviour patch is declared to touch, as a literal set: two reader files and two
#: glue files added to Geant4's particle-management module, that module's source list, the seam
#: files and the helper's header, and the two files of `G4HadronicParameters` that gain the opt-in.
#: A count would let a dropped file and an added file cancel out.
BEHAVIOUR_PATHS = frozenset(
    {
        "source/particles/management/include/G4MuonicAtomHelper.hh",
        "source/particles/management/include/G4MuonicDataOverlay.hh",
        "source/particles/management/include/G4MuonicDataSemantics.hh",
        "source/particles/management/include/G4MuonicDataTable.hh",
        "source/particles/management/sources.cmake",
        "source/particles/management/src/G4MuonicAtomHelper.cc",
        "source/particles/management/src/G4MuonicDataOverlay.cc",
        "source/particles/management/src/G4MuonicDataSemantics.cc",
        "source/particles/management/src/G4MuonicDataTable.cc",
        "source/processes/hadronic/stopping/src/G4EmCaptureCascade.cc",
        "source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc",
        "source/processes/hadronic/stopping/src/G4MuonicAtomDecay.cc",
        "source/processes/hadronic/util/include/G4HadronicParameters.hh",
        "source/processes/hadronic/util/src/G4HadronicParameters.cc",
    }
)
#: The helper's header, which gains the two-argument K-energy declaration; not vendored, so its
#: `index` old id is held to a pin like the other upstream files this repository does not carry.
(HELPER_HH,) = tuple(p for p in BEHAVIOUR_PATHS if p.endswith("/G4MuonicAtomHelper.hh"))
#: The cascade, the one file the D3 lookups are inserted into besides the helper.
(CASCADE,) = tuple(p for p in BEHAVIOUR_PATHS if p.endswith("/G4EmCaptureCascade.cc"))
#: The helper's source, which gains the two public K-energy forms and the private compiled-in one.
(HELPER_CC,) = tuple(p for p in BEHAVIOUR_PATHS if p.endswith("/G4MuonicAtomHelper.cc"))
#: The two existing lines the patch changes: each call of the one-argument K energy that has the
#: muonic atom's base ion in scope, which now passes its mass number. Old call, new call, by path.
CALL_SITES: dict[str, tuple[bytes, bytes]] = {
    "source/particles/management/src/G4MuonicAtomHelper.cc": (
        b"GetKShellEnergy(G4double(Z))", b"GetKShellEnergy(G4double(Z), A)"
    ),
    "source/processes/hadronic/stopping/src/G4MuonicAtomDecay.cc": (
        b"GetKShellEnergy(Zd)", b"GetKShellEnergy(Zd, baseion->GetAtomicMass())"
    ),
}
#: The one definition the patch renames: the compiled-in K energy keeps its body, and its
#: arithmetic order, under a private name, so each public form can query its own key before
#: falling to it. Old name, new name, by path -- held exactly like a call site.
RENAMES: dict[str, tuple[bytes, bytes]] = {
    HELPER_CC: (
        b"G4double G4MuonicAtomHelper::GetKShellEnergy(G4double Z)",
        b"G4double G4MuonicAtomHelper::GetCompiledKShellEnergy(G4double Z)",
    ),
}
#: The cascade's branching block, which no added line may touch.
BRANCHING_TOKENS = (b"G4UniformRand", b"nLevel", b"pGamma", b"AddNewParticle")
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
#: Per family: the seam files, and the vendored copy of that tag each one's hunks must apply to --
#: the two compiled-in copies of the capture tables, the cascade and the decay process.
SEAMS: dict[str, dict[str, pathlib.Path]] = {
    tag: {
        "source/particles/management/src/G4MuonicAtomHelper.cc": vendored / "G4MuonicAtomHelper.cc",
        "source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc": (
            vendored / "G4MuonMinusBoundDecay.cc"
        ),
        **{
            parity.D3_SEAM_UPSTREAM_PATHS[name]: vendored / name
            for name in (parity.CASCADE_NAME, parity.DECAY_NAME)
        },
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
        "source/particles/management/include/G4MuonicDataSemantics.hh": (
            REPO / "cpp/include/G4MuonicDataSemantics.hh"
        ),
        "source/particles/management/src/G4MuonicDataSemantics.cc": (
            REPO / "cpp/src/G4MuonicDataSemantics.cc"
        ),
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
        "source/particles/management/include/G4MuonicAtomHelper.hh": (
            "e2973354ebca8d77c7b5f79437f9ecae966ea67b"
        ),
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
        "source/particles/management/include/G4MuonicAtomHelper.hh": (
            "fa725ef8f191973e39555eefd8062afaaab9d285"
        ),
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


#: Every line of a pre-existing file the patch is allowed to change, by path: the call sites and
#: the renamed definition. A file absent from here is only ever added to.
CHANGES: dict[str, list[tuple[bytes, bytes]]] = {}
for _path, _pair in list(CALL_SITES.items()) + list(RENAMES.items()):
    CHANGES.setdefault(_path, []).append(_pair)


def check_deletions(files: dict[str, FilePatch]) -> None:
    """Every file that existed before the patch -- the seams, the helper's header, the source list,
    the two `G4HadronicParameters` files -- is only ever added to, except where `CHANGES` declares a
    line for it: it deletes exactly as many lines as it declares changes, each deleted line holds
    its old text exactly once, and the hunk that removes it adds exactly that line with the old text
    replaced by the new. Every message names the path."""
    for path, file in sorted(files.items()):
        if file.old_path == b"/dev/null":
            continue
        assert file.old_path == b"a/" + path.encode(), file.old_path
        removed = [
            (hunk, content) for hunk in file.hunks for marker, content, _ in hunk.lines if marker == b"-"
        ]
        declared = list(CHANGES.get(path, []))
        if not declared:
            assert not removed, f"{path}: the patch deletes {[content for _, content in removed]}"
            continue
        assert len(removed) == len(declared), (
            f"{path}: the patch deletes {len(removed)} line(s), not the {len(declared)} it declares"
        )
        for hunk, content in removed:
            matching = [pair for pair in declared if content.count(pair[0]) == 1]
            assert matching, f"{path}: the deleted line holds no declared text once: {content!r}"
            old, new = matching[0]
            declared.remove((old, new))
            added = [line for marker, line, _ in hunk.lines if marker == b"+"]
            assert added == [content.replace(old, new)], (
                f"{path}: hunk {hunk.header.decode()} adds {added}, not the line with {new!r}"
            )


@family
def test_t72_the_seam_hunks_apply_to_the_vendored_files_and_delete_only_the_call_lines(tag: str):
    behaviour, _, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(behaviour.read_bytes()))
    check_deletions(files)
    for path, vendored in SEAMS[tag].items():
        file = files[path]
        original = vendored.read_bytes()
        patched = apply_file_patch(file, original)
        added = sum(1 for hunk in file.hunks for marker, _, _ in hunk.lines if marker == b"+")
        removed = sum(1 for hunk in file.hunks for marker, _, _ in hunk.lines if marker == b"-")
        assert added > 0, f"{path}: the patch adds nothing"
        assert len(split_lines(patched)) == len(split_lines(original)) + added - removed


@family
def test_t72_the_helper_header_adds_the_mass_number_form_and_the_private_compiled_form(tag: str):
    """The header gains exactly two declarations: the public form that also takes a mass number,
    and, under `private:`, the compiled-in form the two public ones fall to."""
    behaviour, _, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(behaviour.read_bytes()))
    added = [content for _, content in added_lines(files[HELPER_HH]) if content.strip()]
    assert added == [
        b"    static G4double GetKShellEnergy(G4double Z, G4int A);",
        b"  private:",
        b"    static G4double GetCompiledKShellEnergy(G4double Z);",
    ], added


@family
def test_t72_each_public_k_energy_form_queries_its_own_key_and_falls_to_the_compiled_one(tag: str):
    """The split: the compiled-in body is reached only under its private name, each public form
    is defined once and queries exactly its own key, and the form that takes a mass number does
    not reach the natural-composition row by calling the one that does not."""
    behaviour, _, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(behaviour.read_bytes()))
    added = [content for _, content in added_lines(files[HELPER_CC])]
    for signature in (
        b"G4double G4MuonicAtomHelper::GetKShellEnergy(G4double Z)",
        b"G4double G4MuonicAtomHelper::GetKShellEnergy(G4double Z, G4int A)",
        b"      if (const G4double* v = G4MuonicDataOverlay::KShell(iz, 0)) return *v * keV;",
        b"      if (const G4double* v = G4MuonicDataOverlay::KShell(iz, A)) return *v * keV;",
    ):
        assert added.count(signature) == 1, (tag, signature, added.count(signature))
    assert added.count(b"  return GetCompiledKShellEnergy(Z);") == 2, tag
    assert b"  return GetKShellEnergy(Z);" not in added, (
        f"{tag}: {HELPER_CC} reaches the one-argument form from the two-argument one"
    )


def check_cascade_lookups(tag: str, files: dict[str, FilePatch]) -> None:
    """The cascade's `+` lines carry the k-shell and the level lookup once each, and no added line of
    any pre-existing file carries a token of the cascade's branching block."""
    added = [content for _, content in added_lines(files[CASCADE])]
    for call in (b"G4MuonicDataOverlay::KShell(Z, A)", b"G4MuonicDataOverlay::Levels(Z, A,"):
        hits = sum(content.count(call) for content in added)
        assert hits == 1, f"{tag}: {CASCADE} adds {call!r} {hits} times"
    for path, file in sorted(files.items()):
        if file.old_path == b"/dev/null":
            continue
        for _, content in added_lines(file):
            tokens = [token for token in BRANCHING_TOKENS if token in content]
            assert not tokens, f"{tag}: {path} adds a line carrying {tokens}: {content!r}"


@family
def test_t72_the_cascade_adds_both_lookups_once_and_no_branching_token(tag: str):
    behaviour, _, _ = FAMILIES[tag]
    check_cascade_lookups(tag, by_new_path(parse_patch(behaviour.read_bytes())))


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
    `G4DatasetDefinitions.cmake`, the helper's header, the two `G4HadronicParameters` files) have no
    `old` bytes to
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
    assert not_rebuilt == {SOURCES_CMAKE, REGISTRATION_PATH, HELPER_HH} | HADRONIC_PARAMETERS
    assert not_rebuilt == set(PRISTINE_INDEX_OLD[tag]), sorted(PRISTINE_INDEX_OLD[tag])


def test_t72_the_glue_files_are_identical_across_families_and_read_the_profile_variables():
    """The glue `.hh` and `.cc`, rebuilt from each family's added-file hunks, are byte-identical
    across the two families -- the glue does not depend on the revision -- and the `.cc` reads the
    environment at exactly one place, names exactly the discovery variable and the three profile
    variables, and raises exactly the five error codes.
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
    assert glue_cc.count(b"std::getenv(") == 1, glue_cc.count(b"std::getenv(")
    named = sorted(set(re.findall(rb'[A-Za-z]\w*\("(G4MUONICDATA[A-Z0-9_]*)"', glue_cc)))
    assert named == [
        b"G4MUONICDATA",
        b"G4MUONICDATA_D1_PROFILE",
        b"G4MUONICDATA_D3_PROFILE",
        b"G4MUONICDATA_PROFILE",
    ], named
    codes = sorted(set(re.findall(rb'"(G4MuonicData[0-9]{3})"', glue_cc)))
    assert codes == [
        b"G4MuonicData001",
        b"G4MuonicData002",
        b"G4MuonicData003",
        b"G4MuonicData004",
        b"G4MuonicData005",
    ], codes


@family
def test_t72_the_opt_in_starts_false_and_its_setter_is_the_only_caller_of_enable(tag: str):
    """The sentence of the patch README that the opt-in is off by default and its setter is the
    only caller of `G4MuonicDataTable::Enable()`, held to the behaviour patch's own `+` lines."""
    behaviour, _, _ = FAMILIES[tag]
    check_opt_in_shape(tag, by_new_path(parse_patch(behaviour.read_bytes())))


def test_t72_the_vendored_readme_names_the_seam_paths_the_behaviour_patch_touches():
    """The vendored README's `upstream path` cells are the seam paths, read from its table by the
    row labels: the BoundDecay cell is the reference module's `UPSTREAM_PATH`, the helper, cascade
    and decay cells are the `SEAMS` keys whose vendored copies carry those names, and together they
    are the paths the behaviour patch's seam hunks touch. The rows read are the `v11.4.2` table's;
    the beta family touches the same paths, which the path-set test above holds for both.
    """
    seams = SEAMS[parity.d1.UPSTREAM_TAG]
    text = parity.VENDORED_README.read_text("utf-8")
    bound_decay = re.findall(r"^\| upstream path \| `([^`]+)` \|$", text, re.M)
    helper = re.findall(r"^\| `G4MuonicAtomHelper\.cc` upstream path \| `([^`]+)` \|$", text, re.M)
    assert len(bound_decay) == 1 and len(helper) == 1, (bound_decay, helper)
    assert bound_decay[0] == parity.d1.UPSTREAM_PATH
    helper_key = next(path for path, vendored in seams.items() if vendored == parity.HELPER)
    assert helper[0] == helper_key
    others = []
    for name in (parity.CASCADE_NAME, parity.DECAY_NAME):
        cells = re.findall(rf"^\| `{re.escape(name)}` upstream path \| `([^`]+)` \|$", text, re.M)
        assert len(cells) == 1, (name, cells)
        key = next(path for path, vendored in seams.items() if vendored.name == name)
        assert cells[0] == key, (name, cells[0], key)
        others.append(cells[0])
    assert {bound_decay[0], helper[0], *others} == set(seams)


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
def test_t73_drill_a_planted_deletion_and_a_changed_call_are_refused_by_path(tag: str):
    """Two in-memory corruptions of the behaviour patch, each refused with the path named: a context
    line of the cascade's hunk turned into a third `-` line (the hunk's count lowered so its
    arithmetic still balances), and a call site whose added line passes other arguments."""
    behaviour, _, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(behaviour.read_bytes()))
    check_deletions(files)
    hunk = files[CASCADE].hunks[-1]
    index = next(i for i, (marker, _, _) in enumerate(hunk.lines) if marker == b" ")
    _, content, has_newline = hunk.lines[index]
    hunk.lines[index] = (b"-", content, has_newline)
    hunk.new_count -= 1
    with pytest.raises(AssertionError, match=re.escape(f"{CASCADE}: the patch deletes")):
        check_deletions(files)

    files = by_new_path(parse_patch(behaviour.read_bytes()))
    path = next(iter(sorted(CALL_SITES)))
    old, new = CALL_SITES[path]
    call_hunk = next(h for h in files[path].hunks for marker, _, _ in h.lines if marker == b"-")
    at = next(i for i, (marker, _, _) in enumerate(call_hunk.lines) if marker == b"+")
    marker, content, has_newline = call_hunk.lines[at]
    call_hunk.lines[at] = (marker, content.replace(new, new.replace(b", A)", b", Z)")), has_newline)
    with pytest.raises(AssertionError, match=re.escape(f"{path}: hunk")):
        check_deletions(files)


@family
def test_t73_drill_an_added_branching_token_and_a_second_lookup_are_refused(tag: str):
    """The cascade's lookups and the branching block, each corrupted in memory: an added line that
    draws `G4UniformRand`, and the k-shell lookup added a second time -- each refused with the family
    named."""
    behaviour, _, _ = FAMILIES[tag]
    files = by_new_path(parse_patch(behaviour.read_bytes()))
    check_cascade_lookups(tag, files)
    hunk = next(h for h in files[CASCADE].hunks for marker, _, _ in h.lines if marker == b"+")
    hunk.lines.append((b"+", b"  G4double draw = G4UniformRand();", True))
    hunk.new_count += 1
    with pytest.raises(AssertionError, match=re.escape(f"{tag}: {CASCADE} adds a line carrying")):
        check_cascade_lookups(tag, files)

    files = by_new_path(parse_patch(behaviour.read_bytes()))
    hunk = next(h for h in files[CASCADE].hunks for marker, _, _ in h.lines if marker == b"+")
    hunk.lines.append((b"+", b"  G4MuonicDataOverlay::KShell(Z, A);", True))
    hunk.new_count += 1
    with pytest.raises(AssertionError, match=re.escape(f"{tag}: {CASCADE} adds")):
        check_cascade_lookups(tag, files)


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


# T-109 -- the D3 harvests: what a patched cascade and helper must emit, derived from the tables
# ------------------------------------------------------------------------------------------------
#
# `cpp/tools/harvest_d3.cc` prints the helper's K energy per Z and one capture cascade per (Z, A);
# `harvest_d3_overlay.cc` prints the same with the opt-in on, then the two-argument K energy. These
# functions are what the evidence runs check those harvests with: with the profile unset, the
# patched harvest equals the unpatched one.

#: Geant4's keV in its internal energy unit (MeV), the double the glue multiplies a table value by.
KEV = 1.0e-3
#: The levels the cascade carries, `fLevelEnergy[0]` .. `fLevelEnergy[13]`.
CASCADE_LEVELS = 14
_HEX = r"-?0x[0-9a-f]+(?:\.[0-9a-f]+)?p[+-][0-9]+"
_K_LINE = re.compile(rf"K ([1-9][0-9]*) ({_HEX})")
_KA_LINE = re.compile(rf"KA ([1-9][0-9]*) ([1-9][0-9]*) ({_HEX})")
_SECONDARY = re.compile(rf"([eg]):({_HEX})|([A-Za-z][A-Za-z0-9_+-]*)")


class HarvestError(Exception):
    """A harvest line outside the grammar the D3 drivers print."""


@dataclasses.dataclass(frozen=True)
class Cascade:
    levels: tuple[float, ...]
    #: ``(kind, kinetic energy)``: kind `e` or `g`, or another particle's name with energy None.
    secondaries: tuple[tuple[str, float | None], ...]
    edep: float


@dataclasses.dataclass(frozen=True)
class D3Harvest:
    k: dict[int, float]
    cascades: dict[tuple[int, int], Cascade]
    ka: dict[tuple[int, int], float]


def _bits(value: float) -> bytes:
    return struct.pack(">d", value)


def parse_d3_harvest(text: str) -> D3Harvest:
    """Every line of a D3 harvest, strictly: `K`, `C` and `KA` lines only, each key once."""
    k: dict[int, float] = {}
    cascades: dict[tuple[int, int], Cascade] = {}
    ka: dict[tuple[int, int], float] = {}
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for number, line in enumerate(lines, 1):
        match = _K_LINE.fullmatch(line)
        if match:
            z = int(match.group(1))
            if z in k:
                raise HarvestError(f"line {number}: K {z} repeated")
            k[z] = float.fromhex(match.group(2))
            continue
        match = _KA_LINE.fullmatch(line)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            if key in ka:
                raise HarvestError(f"line {number}: KA {key} repeated")
            ka[key] = float.fromhex(match.group(3))
            continue
        tokens = line.split(" ")
        if tokens[0] != "C" or len(tokens) < 3 + CASCADE_LEVELS + 3:
            raise HarvestError(f"line {number}: not a harvest line: {line[:80]!r}")
        try:
            key = (int(tokens[1]), int(tokens[2]))
            head = tokens[3 : 3 + CASCADE_LEVELS]
            if not all(re.fullmatch(_HEX, token) for token in head):
                raise ValueError("a level is not a hexadecimal float")
            count = int(tokens[3 + CASCADE_LEVELS])
            start = 4 + CASCADE_LEVELS
            if len(tokens) != start + count + 2 or tokens[start + count] != "edep":
                raise ValueError(f"{count} secondaries then `edep <value>` expected")
            if not re.fullmatch(_HEX, tokens[-1]):
                raise ValueError("edep is not a hexadecimal float")
            secondaries = []
            for token in tokens[start : start + count]:
                piece = _SECONDARY.fullmatch(token)
                if not piece:
                    raise ValueError(f"secondary {token!r}")
                if piece.group(1):
                    secondaries.append((piece.group(1), float.fromhex(piece.group(2))))
                else:
                    secondaries.append((piece.group(3), None))
        except ValueError as error:
            raise HarvestError(f"line {number}: {error}: {line[:80]!r}") from None
        if key in cascades:
            raise HarvestError(f"line {number}: C {key} repeated")
        cascades[key] = Cascade(
            tuple(float.fromhex(token) for token in head), tuple(secondaries), float.fromhex(tokens[-1])
        )
    return D3Harvest(k, cascades, ka)


def level_path(key: tuple[int, int], cascade: Cascade) -> list[tuple[str, int, int]]:
    """``[(kind, from, to)]``: the capture electron at level 13, each Auger electron one level down,
    and each photon to the unique level below the current one whose difference equals its energy --
    every electron's energy checked against its levels on the way."""
    levels = cascade.levels
    assert all(a > b for a, b in zip(levels, levels[1:], strict=False)), (
        f"{key}: levels not strictly decreasing"
    )
    top = CASCADE_LEVELS - 1
    first = cascade.secondaries[0] if cascade.secondaries else None
    assert first is not None and first[0] == "e" and _bits(first[1]) == _bits(levels[top]), (
        f"{key}: the first secondary is not the capture electron at level {top}"
    )
    path = [("e", top, top)]
    current = top
    for index, (kind, energy) in enumerate(cascade.secondaries[1:], 1):
        assert energy is not None, f"{key}: secondary {index} is a {kind}"
        if kind == "e":
            below = current - 1
            assert below >= 0 and _bits(energy) == _bits(levels[below] - levels[current]), (
                f"{key}: electron {index} is not the Auger energy of level {current}"
            )
        else:
            hits = [i for i in range(current) if _bits(levels[i] - levels[current]) == _bits(energy)]
            assert len(hits) == 1, f"{key}: photon {index} matches levels {hits} below {current}"
            below = hits[0]
        path.append((kind, current, below))
        current = below
    return path


def _resolve(table: dict[tuple[int, int], object], z: int, a: int):
    """The record `LookupNatural` finds: the exact key, else (Z, 0), else None."""
    return table.get((z, a), table.get((z, 0)))


def check_unset(ref: D3Harvest, pat: D3Harvest) -> dict[str, int]:
    """With the profile unset: every K and C value of the patched harvest equals the unpatched one
    bitwise, and every two-argument K energy equals its Z's one-argument K energy."""
    assert set(pat.k) == set(ref.k), sorted(set(pat.k) ^ set(ref.k))
    for z in ref.k:
        assert _bits(pat.k[z]) == _bits(ref.k[z]), f"K {z}: {pat.k[z]!r} vs {ref.k[z]!r}"
    assert set(pat.cascades) == set(ref.cascades), "the cascade key sets differ"
    for key, cascade in ref.cascades.items():
        other = pat.cascades[key]
        assert [_bits(v) for v in other.levels] == [_bits(v) for v in cascade.levels], f"C {key}: levels"
        assert [(kind, None if e is None else _bits(e)) for kind, e in other.secondaries] == [
            (kind, None if e is None else _bits(e)) for kind, e in cascade.secondaries
        ], f"C {key}: secondaries"
        assert _bits(other.edep) == _bits(cascade.edep), f"C {key}: edep"
    assert not ref.ka, "the unpatched harvest carries KA lines"
    assert set(pat.ka) == set(pat.cascades), "the KA keys are not the cascade keys"
    for (z, a), value in pat.ka.items():
        assert _bits(value) == _bits(pat.k[z]), f"KA {(z, a)}: {value!r} vs K {z} {pat.k[z]!r}"
    return {"K": len(ref.k), "C": len(ref.cascades), "KA": len(pat.ka)}


def check_profile(
    ref: D3Harvest,
    pat: D3Harvest,
    kshell: dict[tuple[int, int], float],
    levels: dict[tuple[int, int], tuple[float, ...]],
) -> dict[str, object]:
    """Under a profile carrying the D3 tables, per (Z, A): each table resolved as `LookupNatural`
    resolves it; level 0 the k-shell value times keV, levels 1 .. count the level values times keV,
    and every other level the unpatched one; the same particle types and the same level path as
    the unpatched cascade, every patched energy its own levels' difference; each Z's K energy the
    (Z, 0) value times keV, else the unpatched one; each two-argument K energy the resolved value
    times keV, else the unpatched K energy of its Z; and wherever a k-shell record resolves, the
    cascade's level 0 and the two-argument K energy are one double. Returns the counts, the Z with
    no row and the Z with rows but no (Z, 0) row, and the measures of the seam."""
    counts = {len(values) for values in levels.values()}
    assert len(counts) == 1, counts
    (count,) = counts
    assert set(pat.k) == set(ref.k) and set(pat.cascades) == set(ref.cascades), "key sets differ"
    for z, value in ref.k.items():
        natural = kshell.get((z, 0))
        expected = natural * KEV if natural is not None else value
        assert _bits(pat.k[z]) == _bits(expected), f"K {z}: {pat.k[z]!r}, expected {expected!r}"
    resolved_k = resolved_levels = equal_k = 0
    worst_ref = worst_pat = (0, None)
    photon = level0 = (0.0, None)
    for key, before in ref.cascades.items():
        z, a = key
        after = pat.cascades[key]
        k_value, level_values = _resolve(kshell, z, a), _resolve(levels, z, a)
        expected = list(before.levels)
        if k_value is not None:
            expected[0] = k_value * KEV
            resolved_k += 1
        if level_values is not None:
            for i in range(1, min(count, CASCADE_LEVELS - 1) + 1):
                expected[i] = level_values[i - 1] * KEV
            resolved_levels += 1
        for i, (got, want) in enumerate(zip(after.levels, expected, strict=True)):
            assert _bits(got) == _bits(want), f"C {key}: level {i} is {got!r}, the tables give {want!r}"
        kinds = [kind for kind, _ in after.secondaries]
        assert kinds == [kind for kind, _ in before.secondaries], f"C {key}: particle types differ"
        path_before, path_after = level_path(key, before), level_path(key, after)
        assert path_after == path_before, f"C {key}: level path differs"
        want_ka = k_value * KEV if k_value is not None else ref.k[z]
        assert _bits(pat.ka[key]) == _bits(want_ka), f"KA {key}: {pat.ka[key]!r}, expected {want_ka!r}"
        if k_value is not None:
            assert _bits(after.levels[0]) == _bits(pat.ka[key]), f"{key}: level 0 and KA differ"
            equal_k += 1
        for name, cascade in (("ref", before), ("pat", after)):
            ulps = parity._ulp_distance(cascade.edep, cascade.levels[0])
            if name == "ref" and ulps > worst_ref[0]:
                worst_ref = (ulps, key)
            if name == "pat" and ulps > worst_pat[0]:
                worst_pat = (ulps, key)
        seam_before = before.levels[7] - before.levels[8]
        seam_after = after.levels[7] - after.levels[8]
        change = abs(seam_after - seam_before) / seam_before
        if change > photon[0]:
            photon = (change, key)
        change = abs(after.levels[0] - before.levels[0]) / before.levels[0]
        if change > level0[0]:
            level0 = (change, key)
    assert set(pat.ka) == set(pat.cascades), "the KA keys are not the cascade keys"
    zs = {z for z, _ in kshell}
    return {
        "keys": len(ref.cascades),
        "resolved_k": resolved_k,
        "resolved_levels": resolved_levels,
        "level0_equals_ka": equal_k,
        "no_row": sorted(set(range(1, max(zs) + 1)) - zs),
        "rows_but_no_natural_row": sorted(z for z in zs if (z, 0) not in kshell),
        "edep_vs_level0_max_ulps_ref": worst_ref,
        "edep_vs_level0_max_ulps_pat": worst_pat,
        "photon_9_to_8_max_relative_change": photon,
        "level0_max_relative_change": level0,
    }


def load_d3_tables(kshell_path: pathlib.Path, levels_path: pathlib.Path):
    """The two committed D3 tables as the checker wants them: ``{(Z, A): value}`` and
    ``{(Z, A): (e2 .. eN)}``, the level columns read by name from `#COLUMNS`."""
    from openmucf.g4 import spec

    kshell_table = spec.parse(kshell_path.read_bytes().decode("ascii"))
    level_table = spec.parse(levels_path.read_bytes().decode("ascii"))
    k_columns = kshell_table.directives["COLUMNS"].split()
    l_columns = level_table.directives["COLUMNS"].split()
    value = k_columns.index("value")
    e_columns = [i for i, name in enumerate(l_columns) if re.fullmatch(r"e[0-9]+", name)]
    kshell = {(r[0], r[1]): r[value] for r in kshell_table.records}
    levels = {(r[0], r[1]): tuple(r[i] for i in e_columns) for r in level_table.records}
    return kshell, levels


def _synthetic_harvests() -> tuple[str, str, dict, dict]:
    """An unpatched and a patched harvest of one key, Z = 3 and A = 7, built here from levels and a
    fixed level path, with the tables that turn the one into the other."""
    z, a = 3, 7
    e = 1.0e-2
    before = [2.0e-2] + [e / float((i + 1) * (i + 1)) for i in range(1, CASCADE_LEVELS)]
    kshell = {(z, 0): 21.5, (z, a): 21.5}
    level_values = tuple(1.01 * e / float(n * n) / KEV for n in range(2, 9))
    levels = {(z, 0): level_values, (z, a): level_values}
    after = list(before)
    after[0] = kshell[(z, a)] * KEV
    for i in range(1, 8):
        after[i] = level_values[i - 1] * KEV
    steps = [("e", 13, 12), ("g", 12, 8), ("g", 8, 7), ("g", 7, 1), ("g", 1, 0)]

    def render(values: list[float], ka: float | None) -> str:
        secondaries = [f"e:{values[13].hex()}"]
        edep = values[13]
        for kind, top, below in steps:
            energy = values[below] - values[top]
            secondaries.append(f"{kind}:{energy.hex()}")
            edep += energy
        line = " ".join(["C", str(z), str(a), *(v.hex() for v in values), str(len(secondaries)),
                         *secondaries, "edep", edep.hex()])
        k = kshell[(z, 0)] * KEV if ka is not None else before[0]
        text = f"K {z} {k.hex()}\n{line}\n"
        if ka is not None:
            text += f"KA {z} {a} {ka.hex()}\n"
        return text

    return render(before, None), render(after, kshell[(z, a)] * KEV), kshell, levels


def test_t109_the_checker_passes_a_synthetic_patched_harvest_and_its_unset_counterpart():
    ref_text, pat_text, kshell, levels = _synthetic_harvests()
    ref, pat = parse_d3_harvest(ref_text), parse_d3_harvest(pat_text)
    result = check_profile(ref, pat, kshell, levels)
    assert (result["keys"], result["resolved_k"], result["level0_equals_ka"]) == (1, 1, 1)
    unset = ref_text + "".join(f"KA {z} {a} {ref.k[z].hex()}\n" for z, a in ref.cascades)
    assert check_unset(ref, parse_d3_harvest(unset)) == {"K": 1, "C": 1, "KA": 1}


def _replace_token(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def test_t109_drill_a_photon_one_ulp_off_a_changed_type_a_moved_level_and_a_wrong_ka_are_named():
    ref_text, pat_text, kshell, levels = _synthetic_harvests()
    ref = parse_d3_harvest(ref_text)
    cascade = parse_d3_harvest(pat_text).cascades[(3, 7)]
    photon = cascade.secondaries[2][1]
    off = math.nextafter(photon, math.inf)
    drills = [
        ("photon", _replace_token(pat_text, f"g:{photon.hex()}", f"g:{off.hex()}"),
         "photon 2 matches levels []"),
        ("type", _replace_token(pat_text, f"g:{photon.hex()}", f"e:{photon.hex()}"), "particle types differ"),
        ("level 8",
         _replace_token(pat_text, f" {cascade.levels[8].hex()} ", f" {(cascade.levels[8] * 2).hex()} "),
         "level 8 is"),
        ("KA", pat_text.replace("KA 3 7 ", "KA 3 7 -"), "KA (3, 7)"),
    ]
    for name, mutated, message in drills:
        assert mutated != pat_text, name
        with pytest.raises(AssertionError, match=re.escape(message)):
            check_profile(ref, parse_d3_harvest(mutated), kshell, levels)


def test_t109_drill_a_line_outside_the_grammar_is_refused_by_number():
    ref_text, _, _, _ = _synthetic_harvests()
    with pytest.raises(HarvestError, match="line 3"):
        parse_d3_harvest(ref_text + "Geant4 says hello\n")


# T-111 -- `G4NucleiProperties` reads six particle masses from the particle table; without them a
# Z == A key outside its mass tables gets a nuclear mass of zero. `HarvestD3()` constructs them first.
HARVEST_D3 = REPO / "cpp" / "tools" / "harvest_d3.cc"
HARVEST_PARTICLES = ("G4Proton::Proton()", "G4Neutron::Neutron()", "G4Deuteron::Deuteron()",
                     "G4Triton::Triton()", "G4Alpha::Alpha()", "G4He3::He3()")


def particles_before_harvest(source: str) -> list[str]:
    """Every constructor call missing from `HarvestD3()`'s body or placed after its first harvest."""
    # A call inside a block or line comment is not a call: both are removed before the body is read.
    source = re.sub("/[*].*?[*]/", "", source, flags=re.DOTALL)
    source = re.sub("//.*", "", source)
    head = "void HarvestD3() {"
    assert source.count(head) == 1, head
    start, depth, end = source.index(head) + len(head), 1, None
    for end in range(start, len(source)):
        depth += {"{": 1, "}": -1}.get(source[end], 0)
        if depth == 0:
            break
    body = source[start:end]
    marks = {mark: body.find(mark) for mark in ("new G4EmCaptureCascade", "GetKShellEnergy")}
    assert depth == 0 and min(marks.values()) >= 0, (depth, marks)
    problems = [f"{call} is not in HarvestD3()" for call in HARVEST_PARTICLES if call not in body]
    return problems + [f"{call} comes after {mark}" for call in HARVEST_PARTICLES if call in body
                       for mark, first in marks.items() if body.find(call) > first]


def test_t111_the_d3_harvest_constructs_the_six_particles_before_it_harvests():
    assert particles_before_harvest(HARVEST_D3.read_text(encoding="utf-8")) == []


def test_t111_drill_deleting_any_one_constructor_call_is_refused_by_name():
    source = HARVEST_D3.read_text(encoding="utf-8")
    for call in HARVEST_PARTICLES:
        assert source.count(call) == 1, call
        assert particles_before_harvest(source.replace(call, "")) == [f"{call} is not in HarvestD3()"]


def test_t111_drill_a_constructor_call_inside_a_comment_is_refused_by_name():
    source = HARVEST_D3.read_text(encoding="utf-8")
    for call in HARVEST_PARTICLES:
        assert source.count(f"{call};") == 1, call
        for mutated in (source.replace(call, f"/* {call} */"), source.replace(f"{call};", f"// {call};\n")):
            assert particles_before_harvest(mutated) == [f"{call} is not in HarvestD3()"]


def test_t111_drill_particle_table_check_precedes_the_first_print():
    source = HARVEST_D3.read_text(encoding="utf-8")
    expected = ("neutron", "deuteron", "triton", "alpha", "He3", "proton")
    assert source.count("G4ParticleTable::GetParticleTable()->FindParticle(name)") == 1
    assert all(f'"{name}"' in source for name in expected)
    assert source.index("FindParticle(name)") < source.index('std::printf("K %d')
    assert 'std::fprintf(stderr, "harvest_d3: the particle table lacks %s\\n", name);' in source
    assert "std::exit(2);" in source
