"""V-13 -- compare the F-3 producer's figures with the ones ``DATASET_D1.md`` states.

The document's figures are read from its F-3 block by pattern, never typed here. With ``--exact``
(the compiler family the document's own measurement names) all five must agree, and the maximum
relative difference must agree to the two figures the document prints; without it the check passes
on the premise the finding rests on -- that a contracted build moves the expression at all -- and
prints the figures it saw. A producer that skipped for want of FMA is a skip that is
printed, and ``--require-fma`` turns it into a failure on the platforms that must not skip.
``cpp/tools/README.md`` restates three of the figures; with ``--readme`` they must equal the
document's, and so, under ``--exact``, the producer's.

Standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCUMENT_FIGURES = re.compile(
    r"(\d+) points \| (\d+) differ \| (\d+) differ by more than 1 ulp \| max (\d+) ulp at "
    r"\(Z=(\d+), A=(\d+)\)"
)
PRODUCER_LINE = re.compile(
    r"^F3 cxx=(\S+) points=(\d+) differ=(\d+) over1ulp=(\d+) max_ulp=(\d+) at=\((-?\d+),(-?\d+)\) "
    r"max_rel=(\S+)$"
)
README_FIGURES = re.compile(r"up to \*\*(\d+) ulp\*\*, with (\d+) of the (\d+) swept points")
DOCUMENT_MAX_REL = re.compile(r"maximum relative difference (\d(?:\.\d+)?e-\d+)")
SKIPPED_PREFIX = "F3 SKIPPED"


def document_figures(document: Path) -> tuple[int, int, int, int, int, int]:
    """(points, differ, over 1 ulp, max ulp, Z, A) from the document's F-3 block -- exactly one match."""
    matches = DOCUMENT_FIGURES.findall(document.read_text("utf-8"))
    if len(matches) != 1:
        raise SystemExit(
            f"V-13 FAIL {document} carries {len(matches)} F-3 figure lines, expected exactly one"
        )
    points, differ, over_one, max_ulp, z, a = matches[0]
    return int(points), int(differ), int(over_one), int(max_ulp), int(z), int(a)


def document_max_rel(document: Path) -> str:
    """The maximum relative difference the document's F-3 block states, as printed -- exactly one match."""
    matches = DOCUMENT_MAX_REL.findall(document.read_text("utf-8"))
    if len(matches) != 1:
        raise SystemExit(
            f"V-13 FAIL {document} carries {len(matches)} maximum-relative-difference lines, "
            "expected exactly one"
        )
    return matches[0]


def readme_figures(readme: Path) -> tuple[int, int, int]:
    """(max ulp, differ, points) from the harvest tooling's restatement -- exactly one match."""
    matches = README_FIGURES.findall(readme.read_text("utf-8"))
    if len(matches) != 1:
        raise SystemExit(
            f"V-13 FAIL {readme} restates the F-3 figures {len(matches)} times, expected exactly once"
        )
    max_ulp, differ, points = matches[0]
    return int(max_ulp), int(differ), int(points)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--producer-output", type=Path, required=True, help="the line g4muonicdata_f3 wrote")
    parser.add_argument("--document", type=Path, required=True, help="DATASET_D1.md")
    parser.add_argument(
        "--exact", action="store_true", help="require all five figures to equal the document's"
    )
    parser.add_argument("--require-fma", action="store_true", help="a producer skip is a failure")
    parser.add_argument(
        "--readme", type=Path, help="cpp/tools/README.md, whose restated figures must equal the document's"
    )
    args = parser.parse_args(argv)

    stated = document_figures(args.document)
    stated_rel = document_max_rel(args.document)
    restated = ""
    if args.readme is not None:
        readme = readme_figures(args.readme)
        if readme != (stated[3], stated[1], stated[0]):
            print(
                f"V-13 FAIL {args.readme} restates max_ulp={readme[0]} differ={readme[1]} "
                f"points={readme[2]}; the document states max_ulp={stated[3]} differ={stated[1]} "
                f"points={stated[0]}"
            )
            return 1
        restated = f" readme max_ulp={readme[0]} differ={readme[1]} points={readme[2]}"

    line = args.producer_output.read_text("utf-8").strip()
    if line.startswith(SKIPPED_PREFIX):
        if args.require_fma:
            print(f"V-13 FAIL {line} (this platform must not skip)")
            return 1
        print(f"V-13 SKIPPED {line}")
        return 0
    match = PRODUCER_LINE.match(line)
    if match is None:
        print(f"V-13 FAIL unrecognised producer output: {line!r}")
        return 1
    cxx = match.group(1)
    seen = tuple(int(match.group(i)) for i in range(2, 8))
    figures = (
        f"cxx={cxx} points={seen[0]} differ={seen[1]} over1ulp={seen[2]} max_ulp={seen[3]} "
        f"at=(Z={seen[4]}, A={seen[5]}) max_rel={match.group(8)}{restated}"
    )
    document_rel = f" document max_rel={stated_rel}"
    if args.exact:
        if seen != stated:
            print(
                f"V-13 FAIL exact {figures}; the document states points={stated[0]} differ={stated[1]} "
                f"over1ulp={stated[2]} max_ulp={stated[3]} at=(Z={stated[4]}, A={stated[5]})"
            )
            return 1
        # To the two figures the document prints: the producer writes more, and a different build
        # of the same compiler family moves the trailing ones.
        if f"{float(match.group(8)):.1e}" != f"{float(stated_rel):.1e}":
            print(f"V-13 FAIL exact {figures}; the document states max_rel={stated_rel}")
            return 1
        print(f"V-13 PASS exact {figures}{document_rel}")
        return 0
    if seen[1] == 0:
        print(f"V-13 FAIL premise {figures}; no point moved, so the contracted build did not differ")
        return 1
    print(f"V-13 PASS premise {figures}{document_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
