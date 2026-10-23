"""Build ``harvest_decay.cc`` against an unpatched Geant4 installation, run it, and write the raw directory of
one build that ``scripts/generate_g4lifetime.py harvest`` converts into ``data/g4/d1/d1_decay_rate.oracle``.

  python3 cpp/tools/decay_harvest.py --install <prefix> --work <dir> --out <raw>/<build>

``--out`` must not exist, and ``--install`` must hold an executable ``bin/geant4-config`` and a
``bin/geant4.sh``. The tool writes, in ``--out``:

* ``version.txt``, what ``geant4-config --version`` printed;
* ``gcc.txt``, the first line of ``g++ --version``;
* ``sha256.txt``, the SHA-256 of the driver it compiles;
* ``flags.txt``, the compile flags, written from :data:`FLAGS`, the tuple the compile command is built from;
* ``decay.txt`` and ``stderr.txt``, what the driver printed on each stream.

Every command (``geant4-config``, ``g++`` and the driver) runs in a ``bash`` that has sourced the
installation's ``bin/geant4.sh``, so the compiler sees the installation's flags and the driver loads its
libraries. The conversion reads each build's flags from its ``flags.txt``, never from a constant. The tool
does not check that the installation is unpatched; the build name the caller gives in ``--out`` states it.
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "harvest_decay.cc"
#: The flags the driver is compiled with; ``flags.txt`` records them for the conversion.
FLAGS = ("-std=c++17", "-O2", "-ffp-contract=off")
#: Runs ``"$@"`` after sourcing the script passed as ``$0``, discarding what the script prints.
_SOURCED = 'source "$0" >/dev/null 2>&1 && exec "$@"'


class HarvestError(RuntimeError):
    """An installation, a directory or a command the harvest refuses."""


def compile_argv(source: Path | str, binary: Path | str, cflags_libs: list[str]) -> list[str]:
    """The compile command: ``g++``, :data:`FLAGS`, the source, the binary, then the installation's
    ``geant4-config --cflags --libs``."""
    return ["g++", *FLAGS, str(source), "-o", str(binary), *cflags_libs]


def flags_text() -> str:
    """The text of ``flags.txt``: :data:`FLAGS` on one line."""
    return " ".join(FLAGS) + "\n"


def in_installation(install: Path, argv: list[str]) -> list[str]:
    """``argv`` run in a ``bash`` that has sourced ``install``'s ``bin/geant4.sh``."""
    return ["bash", "-c", _SOURCED, str(install / "bin" / "geant4.sh"), *argv]


def _write(path: Path, text: str) -> None:
    """``text`` as ASCII with LF line ends, on any platform."""
    path.write_bytes(text.encode("ascii"))


def _output(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise HarvestError(f"{argv!r} failed ({result.returncode}): {result.stderr[-2000:]}")
    return result.stdout


def check_paths(install: Path, work: Path, out: Path) -> Path:
    """The installation's ``geant4-config``, after refusing an existing ``out``, an installation without an
    executable ``bin/geant4-config`` or without ``bin/geant4.sh``, and a non-empty ``work``."""
    if out.exists():
        raise HarvestError(f"{out} exists: a harvest never writes into an existing directory")
    config = install / "bin" / "geant4-config"
    if not (config.is_file() and os.access(config, os.X_OK)):
        raise HarvestError(f"{install} has no executable bin/geant4-config")
    if not (install / "bin" / "geant4.sh").is_file():
        raise HarvestError(f"{install} has no bin/geant4.sh")
    if work.exists() and any(work.iterdir()):
        raise HarvestError(f"{work} is not empty")
    return config


def harvest(install: Path, work: Path, out: Path) -> None:
    """Compile the driver against ``install`` in ``work``, run it and write ``out``."""
    config = check_paths(install, work, out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True)
    _write(out / "version.txt", _output(in_installation(install, [str(config), "--version"])))
    gcc = _output(in_installation(install, ["g++", "--version"])).splitlines()[0]
    _write(out / "gcc.txt", gcc + "\n")
    _write(out / "sha256.txt", hashlib.sha256(SOURCE.read_bytes()).hexdigest() + "\n")
    _write(out / "flags.txt", flags_text())
    binary = work / "harvest_decay"
    cflags_libs = _output(in_installation(install, [str(config), "--cflags", "--libs"])).split()
    _output(in_installation(install, compile_argv(SOURCE, binary, cflags_libs)))
    with (out / "decay.txt").open("wb") as stdout, (out / "stderr.txt").open("wb") as stderr:
        rc = subprocess.call(in_installation(install, [str(binary)]), stdout=stdout, stderr=stderr)
    if rc != 0:
        raise HarvestError(f"the driver exited with {rc}; see {out / 'stderr.txt'}")
    print(
        f"HARVEST {out} version={(out / 'version.txt').read_text('ascii').strip()} gcc={gcc!r} "
        f"flags={flags_text().strip()!r}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    harvest(args.install, args.work, args.out)


if __name__ == "__main__":
    main()
