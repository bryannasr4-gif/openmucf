"""openmucf.g4.emit -- the shipping artifacts: a deterministic archive and its registration snippet.

A dataset is shipped as one gzipped tar archive whose members -- the Layer-1 ``.g4dat`` files, the
Layer-2 ``*.prov.json`` files they were generated from, and a generated ``README`` and ``History`` --
all sit under one top-level directory named as Geant4's dataset machinery expects after unpacking.
This module builds that archive, checksums it, and writes the ``geant4_add_dataset(...)`` block a
build system needs in order to register it.

**Determinism is the whole point.** A tar entry carries an mtime, a uid/gid, a user/group name and a
permission bits field; a gzip container carries an mtime and, if you let it, the source filename.
Every one of those is a channel through which the machine that built the archive leaks into the
archive's bytes -- and an artifact whose bytes depend on who built it cannot be byte-diff audited,
cannot be checksummed once and shipped, and cannot be reproduced by a reader checking our work. All
of them are pinned here; ``FORMAT_SPEC.md`` **section 8** is the normative statement of the same
table, so an outside reader can reconstruct the archive's container from the public document alone.
Reproducing its exact bytes -- and therefore its ``MD5SUM`` -- additionally needs a compatible zlib,
because the compressed stream is not something either document can pin; section 8 says so.
Entries are written in sorted name order, and the archive is a pure function of ``{name: bytes}``.

One honest limit, stated rather than implied (and disclosed in section 8, not only here): the
*container* metadata this module writes is fixed by construction, but the DEFLATE stream inside it
comes from zlib, and two zlib builds are not guaranteed to emit byte-identical compressed output for
the same input. Container determinism is therefore asserted directly (:func:`gzip_header`), and
whole-archive determinism is proven per platform by test and by CI rather than assumed from this
docstring.

Standard library only, and no import of the kinetics modules (enforced by test, not by comment).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from collections.abc import Mapping, Sequence

__all__ = [
    "ARCHIVE_EXTENSION",
    "add_dataset_snippet",
    "archive_name",
    "build_tarball",
    "dataset_directory",
    "gzip_header",
    "history_member",
    "readme_member",
    "tarball_md5",
]

#: What Geant4's dataset machinery expects to download and unpack.
ARCHIVE_EXTENSION = "tar.gz"

#: Pinned member metadata. Read-write for the owner, readable by everyone, owned by nobody in
#: particular, and stamped with the epoch rather than with the moment the build happened.
_MEMBER_MODE = 0o644
_EPOCH = 0
#: The ustar header's name field. Longer names force a GNU/PAX extension header, whose exact bytes
#: vary between Python versions -- a determinism hazard that a length check turns into a loud error.
_MAX_MEMBER_NAME = 100


def dataset_directory(name: str, version: str) -> str:
    """The directory a Geant4 dataset unpacks to: ``DIRECTORY = NAME + VERSION``, per the rule at
    ``cmake/Modules/G4InstallData.cmake`` lines 234-235 of Geant4 v11.4.2."""
    return f"{name}{version}"


def archive_name(filename: str, version: str) -> str:
    """The packed dataset's file name: ``FILE = FILENAME.VERSION.EXTENSION``, per the rule at
    ``cmake/Modules/G4InstallData.cmake`` lines 229-230 of Geant4 v11.4.2."""
    return f"{filename}.{version}.{ARCHIVE_EXTENSION}"


def build_tarball(members: Mapping[str, bytes], *, directory: str) -> bytes:
    """Build a deterministic ``.tar.gz`` from ``{file name: exact bytes}``, every member stored as
    ``directory/file name``.

    The same mapping always produces the same bytes on the same zlib: entries are written in sorted
    stored-name order, every metadata field is pinned, and the gzip container carries neither a
    timestamp nor a source filename.
    """
    # Exactly one directory component, and it is this one (``FORMAT_SPEC.md`` section 8): the
    # archive unpacks to ``directory`` and nothing else, and no directory-entry member is written
    # -- an unpacker creates the directory from the members' paths. A separator in either the
    # directory or a file name would be a different layout, not a longer name. The ASCII rule is
    # stated for its message, honestly: the length check below already rejects a non-ASCII name,
    # but as a ``UnicodeEncodeError`` from ``str.encode`` rather than as a statement about archive
    # names. No determinism channel was ever open here -- that check has always run first -- and
    # saying otherwise would be claiming a fix for a hole that did not exist.
    if (
        not directory
        or "/" in directory
        or "\\" in directory
        or directory in (".", "..")
        or not directory.isascii()
    ):
        raise ValueError(
            f"archive directory {directory!r} must be a non-empty US-ASCII name, with no path "
            "separator and not a dot entry"
        )
    stored: dict[str, bytes] = {}
    for name in sorted(members):
        if not name or "/" in name or "\\" in name:
            raise ValueError(
                f"archive member name {name!r} must be a plain file name, with no path separator"
            )
        if not name.isascii():
            raise ValueError(
                f"archive member name {name!r} must be US-ASCII; a non-ASCII name is encoded with the "
                "builder's filesystem encoding and would make the archive machine-dependent"
            )
        path = f"{directory}/{name}"
        if len(path.encode("ascii")) > _MAX_MEMBER_NAME:
            raise ValueError(
                f"archive member name {path!r} exceeds the {_MAX_MEMBER_NAME}-byte ustar limit; a "
                "longer name needs an extension header whose bytes are not version-stable"
            )
        stored[path] = members[name]

    raw = io.BytesIO()
    # USTAR explicitly: the default format has changed across Python versions, and the archive's
    # bytes must not depend on which interpreter built it. Every stored name fits the ustar name
    # field, so the header's ``prefix`` field stays empty.
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(stored):
            payload = stored[name]
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = _EPOCH
            info.mode = _MEMBER_MODE
            info.type = tarfile.REGTYPE
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))

    compressed = io.BytesIO()
    with gzip.GzipFile(
        fileobj=compressed, mode="wb", compresslevel=9, mtime=_EPOCH, filename=""
    ) as stream:
        stream.write(raw.getvalue())
    return compressed.getvalue()


def gzip_header(archive: bytes) -> dict[str, int]:
    """The gzip container fields that must not vary: magic, method, flags, mtime, XFL and OS.

    Split out so that a determinism failure can be attributed. If these are stable and the archive
    bytes are not, the difference is in the DEFLATE stream (a zlib build difference); if these move,
    it is this module.
    """
    if len(archive) < 10 or archive[:3] != b"\x1f\x8b\x08":
        raise ValueError("not a gzip stream")
    return {
        "magic": int.from_bytes(archive[:2], "big"),
        "method": archive[2],
        "flags": archive[3],
        "mtime": int.from_bytes(archive[4:8], "little"),
        "xfl": archive[8],
        "os": archive[9],
    }


def tarball_md5(archive: bytes) -> str:
    """MD5 of the archive bytes, lowercase hex.

    MD5 because that is what ``geant4_add_dataset``'s ``MD5SUM`` field is: this is a download
    integrity check against corruption, not a security boundary, and the algorithm is fixed by the
    consumer rather than chosen here. ``usedforsecurity=False`` says so to the runtime and keeps
    this working on a FIPS-restricted build.
    """
    return hashlib.md5(archive, usedforsecurity=False).hexdigest()


#: The attribution notice of ``FORMAT_SPEC.md`` section 9, line for line without the quote prefix.
_NOTICE_LINES = (
    "This product includes software developed by Members of the Geant4 Collaboration",
    "( http://cern.ch/geant4 ).",
)

#: One archive table: ``(layer1_name, layer2_name, table, profile, seam, rows)`` -- the Layer-1
#: member's file name, its Layer-2 sibling's, the ``#TABLE``, ``#PROFILE`` and ``#SEAM`` values
#: and the record count. Everything ``README`` and ``History`` say comes from these and the
#: dataset's name and version; nothing about the machine or the moment that built the archive.
TableEntry = tuple[str, str, str, str, str, int]


def _ascii_member(lines: Sequence[str]) -> bytes:
    """LF-joined, trailing newline, US-ASCII -- a member's bytes are a function of its lines."""
    return ("\n".join(lines) + "\n").encode("ascii")


def readme_member(*, name: str, version: str, files: Sequence[TableEntry]) -> bytes:
    """The ``README`` member, in the shape of the ``README`` a Geant4 dataset directory carries:
    the directory's name, what reads the files, the file list, and the attribution notice."""
    lines = [
        dataset_directory(name, version),
        "",
        "The data in this directory are read by G4MuonicDataTable: each .g4dat file is one table "
        "in the G4MuonicData format, and its .prov.json sibling is the per-row provenance whose "
        "SHA-256 the table's #SOURCEDIGEST names.",
        "FORMAT_SPEC.md in the openmucf repository states the format.",
        "",
        "The following files can be found here:",
    ]
    for layer1, layer2, table, profile, seam, rows in files:
        lines.append(f"  - {layer1}: #TABLE {table}, #PROFILE {profile}, #SEAM {seam}, {rows} records")
        lines.append(f"  - {layer2}: Layer 2 for {layer1}")
    lines.append("  - README: this file")
    lines.append("  - History: the version this directory carries and its tables")
    lines.append("")
    lines.extend(_NOTICE_LINES)
    return _ascii_member(lines)


def history_member(*, name: str, version: str, files: Sequence[TableEntry]) -> bytes:
    """The ``History`` member: one entry, the version this archive carries, and its tables.

    One entry only, because the generator has no source for what earlier versions carried; the
    repository's ``CHANGELOG.md`` is the history across versions.
    """
    title = f"History for {name} files:"
    lines = [title, "-" * len(title), "", version]
    for layer1, _layer2, table, profile, _seam, rows in files:
        lines.append(f"  {layer1}: #TABLE {table}, #PROFILE {profile}, {rows} records")
    return _ascii_member(lines)


def add_dataset_snippet(
    *,
    name: str,
    version: str,
    filename: str,
    envvar: str,
    md5: str,
    extension: str = ARCHIVE_EXTENSION,
) -> str:
    """The ``geant4_add_dataset(...)`` block that registers this dataset at configure time.

    Registration is the "mode 1" discovery path of ``FORMAT_SPEC.md`` section 5, and it requires an
    upstream change; until that happens every user is on mode 2 and sets the environment variable by
    hand. The block is generated rather than hand-maintained so that its ``MD5SUM`` cannot drift
    from the archive it describes.

    The header comment is part of the artifact on purpose: a snippet found on its own must say that
    its names are provisional.
    """
    lines = [
        "# Generated by scripts/generate_g4data.py -- do not edit by hand.",
        "# Paste into cmake/Modules/G4DatasetDefinitions.cmake to register the dataset.",
        "#",
        "# Provisional: the dataset name and the environment variable are placeholders (see",
        "# FORMAT_SPEC.md).",
        "geant4_add_dataset(",
        f"  NAME      {name}",
        f"  VERSION   {version}",
        f"  FILENAME  {filename}",
        f"  EXTENSION {extension}",
        f"  ENVVAR    {envvar}",
        f"  MD5SUM    {md5}",
        "  )",
    ]
    return "\n".join(lines) + "\n"
