"""openmucf.g4.emit -- the shipping artifacts: a deterministic archive and its registration snippet.

A dataset is shipped as one gzipped tar archive holding the Layer-1 ``.g4dat`` files and the Layer-2
``*.prov.json`` files they were generated from. This module builds that archive, checksums it, and
writes the ``geant4_add_dataset(...)`` block a build system needs in order to register it.

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
from collections.abc import Mapping

__all__ = [
    "ARCHIVE_EXTENSION",
    "add_dataset_snippet",
    "build_tarball",
    "gzip_header",
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


def build_tarball(members: Mapping[str, bytes]) -> bytes:
    """Build a deterministic ``.tar.gz`` from ``{member name: exact bytes}``.

    The same mapping always produces the same bytes on the same zlib: entries are written in sorted
    name order, every metadata field is pinned, and the gzip container carries neither a timestamp
    nor a source filename.
    """
    for name in sorted(members):
        # The archive is FLAT (``FORMAT_SPEC.md`` section 8), so a separator of either kind is a
        # different layout, not a longer name. The ASCII rule is stated for its message, honestly:
        # the length check below already rejects a non-ASCII name, but as a ``UnicodeEncodeError``
        # from ``str.encode`` rather than as a statement about archive names. No determinism channel
        # was ever open here -- that check has always run first -- and saying otherwise would be
        # claiming a fix for a hole that did not exist.
        if not name or "/" in name or "\\" in name:
            raise ValueError(
                f"archive member name {name!r} must be a plain flat name, with no path separator"
            )
        if not name.isascii():
            raise ValueError(
                f"archive member name {name!r} must be US-ASCII; a non-ASCII name is encoded with the "
                "builder's filesystem encoding and would make the archive machine-dependent"
            )
        if len(name.encode("ascii")) > _MAX_MEMBER_NAME:
            raise ValueError(
                f"archive member name {name!r} exceeds the {_MAX_MEMBER_NAME}-byte ustar limit; a "
                "longer name needs an extension header whose bytes are not version-stable"
            )

    raw = io.BytesIO()
    # USTAR explicitly: the default format has changed across Python versions, and the archive's
    # bytes must not depend on which interpreter built it.
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            payload = members[name]
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
    its names are provisional and that the archive it checksums is flat.
    """
    lines = [
        "# Generated by scripts/generate_g4data.py -- do not edit by hand.",
        "# Paste into cmake/Modules/G4DatasetDefinitions.cmake to register the dataset.",
        "#",
        "# Provisional: the dataset name and the environment variable are placeholders (see",
        "# FORMAT_SPEC.md), and the archive is FLAT -- its members sit at the archive root rather",
        "# than under a <FILENAME><VERSION> directory. Fixing the installed layout is part of",
        "# registration, not of the format, and is deliberately not decided here.",
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
