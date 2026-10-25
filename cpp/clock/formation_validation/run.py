"""Build and run the finite formation-time comparator matrix for one Geant4 revision."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from prepare import PROCESS_DIR, PROCESS_NAMES, REVISIONS, prepare


def plan_runs(objects: Path, logs: Path) -> list[tuple[list[str], Path]]:
    runs = []
    for variant in ("old", "new", "zero"):
        for origin in (0, 250, 1000):
            for mode, count in (("controlled", 4), ("default", 50), ("light", 4)):
                command = [str(objects / f"wrapper_{variant}"), mode, str(origin), str(count)]
                runs.append((command, logs / f"{variant}-{mode}-{origin}.log"))
        command = [str(objects / f"transport_{variant}"), "atomic", "plain", "250", "120", "12345"]
        runs.append((command, logs / f"{variant}-integrated-250.log"))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", choices=REVISIONS, required=True)
    parser.add_argument("--geant4-source", type=Path, required=True)
    parser.add_argument("--geant4-install", type=Path, required=True)
    parser.add_argument(
        "--geant4-dir", type=Path, help="Override the installation's lib/cmake/Geant4 directory"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, installation = args.output_root.resolve(), args.geant4_install.resolve()
    output = prepare(args.revision, args.geant4_source, root / args.revision)
    objects, logs = output / "objects", root / "logs" / args.revision
    logs.mkdir(parents=True, exist_ok=False)
    config = args.geant4_dir.resolve() if args.geant4_dir else installation / "lib/cmake/Geant4"
    if not (config / "Geant4Config.cmake").is_file():
        raise FileNotFoundError(f"Geant4Config.cmake not found in {config}; supply --geant4-dir")
    sources = ";".join(str(output / "kernel" / PROCESS_DIR / name) for name in PROCESS_NAMES[:3])
    with (output / "configure.log").open("w", encoding="utf-8") as stream:
        subprocess.run(
            [
                "cmake",
                "-S",
                str(output / "app"),
                "-B",
                str(objects),
                f"-DGeant4_DIR={config}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCLOCK_KERNEL_SOURCES={sources}",
            ],
            check=True,
            timeout=120,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    with (output / "compile.log").open("w", encoding="utf-8") as stream:
        subprocess.run(
            ["cmake", "--build", str(objects), "--parallel", "2"],
            check=True,
            timeout=360,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    environment = os.environ.copy()
    libraries = [str(installation / "lib"), str(installation / "lib64")]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    # Dataset variables and other scientific settings are inherited from the caller.
    for command, path in plan_runs(objects, logs):
        with path.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command, check=True, timeout=180, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
    print(f"Completed {args.revision}; logs: {logs}")


if __name__ == "__main__":
    main()
