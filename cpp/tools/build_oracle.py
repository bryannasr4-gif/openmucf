"""Turn a raw harvest into ``data/g4/d1/d1_gp_sweep.oracle``.

    python cpp/tools/build_oracle.py --sweep sweep.txt --degenerate degenerate.txt \
        --build "<line 1>" --build "<line 2>" --build "<line 3>" -o data/g4/d1/d1_gp_sweep.oracle

The two drivers in this directory print raw ``%a`` lines; this is the step between that and the
committed file, and it is here for the same reason the drivers are: a committed harvested artifact
whose producing code is not committed is exactly the reproducibility hole vendoring the upstream
source exists to close. Leaving this step out was a real gap -- the digest rule and the subset rule
were documented well enough to re-derive by hand, but "documented" and "committed" are not the same
standard, and this file is the one the rest of ``cpp/tools/README.md`` demands.

**The values are the harvest's, never this script's.** Rates and effective charges are passed
through as the exact ``%a`` strings the Geant4-linked binary printed; nothing here recomputes a
number. What the script does compute is the sha256 over the harvested doubles and *which* harvested
rows get spelled out in the diagnostic subset. The table-hit keys come from the vendored upstream
source, because "table hit" is a fact about upstream's array and not about our arithmetic -- and
selecting which measured rows to echo cannot make a wrong measurement look right.

**The build description is an argument, not a constant.** The oracle header records the build that
produced the values, so the person running the harvest supplies it: a header claiming a build that
was not used would be worse than no header. ``cpp/tools/README.md`` carries the exact invocation
that reproduces the committed file.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from openmucf.g4.sources import d1_nuclear_capture as d1  # noqa: E402

BANNER = (
    "G4MuonicData D1 -- compiled-Geant4 oracle for the nuclear-capture seam",
    "",
    "HARVESTED, not generated. These values came out of a Geant4-linked binary; nothing in this",
    "repository can produce them, which is exactly why comparing the Python reference",
    "implementation against them is evidence and not a tautology. `make g4data` does not rewrite",
    'this file and `make audit` does not byte-diff it -- "regenerate and compare" is undefined for',
    "a harvested input. It is protected instead by re-derivation: tests/test_g4parity.py recomputes",
    "every value here in Python and compares, which is a stronger check than re-reading bytes.",
    "",
)
BUILD_PREAMBLE = (
    "The build is part of the measurement, not a footnote: the identical expression compiled with",
    "floating-point contraction enabled returns results up to thousands of ulp away, so a parity",
    "claim that does not name its build says nothing. See cpp/tools/README.md and DATASET_D1.md.",
)
DIGEST_RULE = (
    "sha256 over the concatenated big-endian IEEE-754 binary64 bytes of",
    "rate(Z, A), in that order. Bytes rather than text: unambiguous in both",
    "languages, with no formatting degrees of freedom.",
)
SUBSET_PREAMBLE = (
    "The subset below exists so that a digest mismatch is DIAGNOSABLE -- a bare hash tells a",
    "maintainer nothing about what moved. It is chosen by rule, not by taste: every table hit,",
    "every Z's first negative A (which is the negative-rate finding's own evidence, so the fixture",
    "and the finding are the same data), and the corners of the sweep box.",
)
COLUMNS = "Z A rate_hexfloat   (compare on PARSED values, never on these strings)"
DEGENERATE_RULE = (
    "excluded from the sweep and from the digest: these return non-finite",
    "values, and a NaN has no single bit pattern to hash. Compared by",
    "CLASSIFICATION rather than by value. Registered as a finding, not fixed.",
)


def comment(text: str = "") -> str:
    """A header comment line: ``#`` alone when empty, ``# `` plus the text otherwise."""
    return f"# {text}".rstrip()


def field(name: str, value: str) -> str:
    """A ``# name  value`` header field, on the column the whole header is aligned to."""
    return f"# {name:<18}{value}"


def continuation(value: str) -> str:
    """A wrapped header value, indented onto the same column as the field it continues."""
    return f"#{' ' * 19}{value}"


def read_sweep(path: Path, zeff_size: int) -> tuple[dict[tuple[int, int], str], dict[int, str]]:
    """Parse ``harvest_d1``'s output into ``{(Z, A): hexfloat}`` and ``{Z: hexfloat}``.

    Both sections are validated against what the vendored source says they must contain, and for the
    same reason: the driver prints the sweep first and the effective charges last, so an interrupted
    run leaves exactly the shape that looks complete -- a full sweep box and a short ``ZEFF`` tail.
    The header this file goes on to write says the ``ZEFF`` rows pin the clamp "at both ends", which
    a truncated tail would make false while every value in it stayed correct.
    """
    rates: dict[tuple[int, int], str] = {}
    zeff: dict[int, str] = {}
    for number, line in enumerate(path.read_text("ascii").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        # Name the bad line. A bare IndexError out of a three-line parser sends a maintainer looking
        # for a bug in this script when what they have is a truncated or half-written harvest.
        if len(fields) != 3:
            raise SystemExit(
                f"{path}:{number}: expected 3 fields, got {len(fields)}: {line!r}. A harvest line is "
                f"'Z A hexfloat' or 'ZEFF Z hexfloat'; this file is truncated or not a harvest."
            )
        if fields[0] == "ZEFF":
            zeff[int(fields[1])] = fields[2]
        else:
            rates[int(fields[0]), int(fields[1])] = fields[2]
    expected = {
        (z, a)
        for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1)
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1)
    }
    if set(rates) != expected:
        raise SystemExit(
            f"{path} does not cover the sweep box: {len(rates)} rows against {len(expected)} "
            f"expected. The oracle's digest is defined over the whole box, so a partial harvest "
            f"cannot produce one."
        )
    # Z 0..zeff_size inclusive: every entry of the table, plus one probe past its last index, which
    # is what makes the clamp observable at the top end. Derived from the vendored source, never
    # written down here -- if upstream's table changes length, this moves with it.
    expected_zeff = set(range(zeff_size + 1))
    if set(zeff) != expected_zeff:
        missing = sorted(expected_zeff - set(zeff))
        extra = sorted(set(zeff) - expected_zeff)
        raise SystemExit(
            f"{path}: the ZEFF section must cover Z 0..{zeff_size} ({len(expected_zeff)} rows) to "
            f"pin the clamp at both ends; got {len(zeff)} rows, missing {missing}, unexpected "
            f"{extra}. The driver prints these last, so a short tail here means the harvest was "
            f"interrupted."
        )
    return rates, zeff


def select_subset(
    rates: dict[tuple[int, int], str], source: d1.D1Extraction
) -> tuple[list[tuple[int, int]], int, int, int]:
    """The diagnostic subset, by the rule the oracle header states, with its three tallies.

    Every table hit, every Z's first negative A, and the corners of the sweep box -- deduplicated.
    Rule (ii) is not padding: those rows are the negative-rate finding's own evidence, so the
    fixture and the finding are the same data.
    """
    hits = {(z, a) for z, a, _, _ in source.capture_records}
    first_negative = set()
    for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1):
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1):
            if float.fromhex(rates[z, a]) < 0.0:
                first_negative.add((z, a))
                break
    corners = {
        (z, a)
        for z in (d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX)
        for a in (d1.SWEEP_A_MIN, d1.SWEEP_A_MAX)
    }
    missing = sorted(hits - set(rates))
    if missing:
        raise SystemExit(f"the harvest does not cover table hits {missing}; it is not this dataset")
    return sorted(hits | first_negative | corners), len(hits), len(first_negative), len(corners)


RATE_PROBE_DECL = re.compile(r"probes\[\]\[2\]\s*=\s*\{(.*?)\}\s*;", re.DOTALL)
RATE_PROBE_PAIR = re.compile(r"\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}")
CLAMP_PROBE_DECL = re.compile(r"for\s*\(\s*int\s+Z\s*:\s*\{([^}]*)\}\s*\)")


def check_degenerate(path: Path, driver: Path) -> list[str]:
    """The degenerate harvest, checked against the probe set its own driver declares.

    The sweep's two sections are validated against the vendored table; this one has no such
    reference -- the probes exist nowhere but in the driver -- so the driver is what it is checked
    against. Without this the block is spliced in verbatim, and since the driver prints the clamp
    probes last, an interrupted run drops exactly the rows that pin the clamp's far end.
    """
    if not driver.exists():
        raise SystemExit(f"{driver} is missing; the degenerate probe set cannot be verified")
    text = driver.read_text("ascii")
    rate_block, clamp_block = RATE_PROBE_DECL.search(text), CLAMP_PROBE_DECL.search(text)
    if not rate_block or not clamp_block:
        raise SystemExit(f"{driver} no longer declares its probes as literal lists")
    declared_rates = [(int(z), int(a)) for z, a in RATE_PROBE_PAIR.findall(rate_block.group(1))]
    declared_clamps = [int(z) for z in clamp_block.group(1).split(",")]

    lines = path.read_text("ascii").splitlines()
    harvested_rates = [
        (int(f[1]), int(f[2])) for f in (line.split() for line in lines) if f and f[0] == "RATE"
    ]
    harvested_clamps = [
        int(f[1]) for f in (line.split() for line in lines) if f and f[0] == "ZEFFCLAMP"
    ]
    if harvested_rates != declared_rates or harvested_clamps != declared_clamps:
        raise SystemExit(
            f"{path} does not carry the probes {driver.name} harvests. Rate probes: got "
            f"{harvested_rates}, expected {declared_rates}. Clamp probes: got {harvested_clamps}, "
            f"expected {declared_clamps}. The driver prints the clamp probes last, so a short tail "
            f"here means the harvest was interrupted."
        )
    return lines


def sweep_digest(rates: dict[tuple[int, int], str]) -> str:
    """The pre-registered digest rule, over the HARVESTED doubles: row-major, Z outermost."""
    running = hashlib.sha256()
    for z in range(d1.SWEEP_Z_MIN, d1.SWEEP_Z_MAX + 1):
        for a in range(d1.SWEEP_A_MIN, d1.SWEEP_A_MAX + 1):
            running.update(struct.pack(">d", float.fromhex(rates[z, a])))
    return running.hexdigest()


def render(
    sweep: Path, degenerate: Path, build: list[str], source_path: Path, driver: Path
) -> str:
    """The whole oracle, as text, from the two harvest files and the recorded build."""
    source = d1.extract(source_path.read_text("ascii"))
    rates, zeff = read_sweep(sweep, len(source.zeff))
    degenerate_lines = check_degenerate(degenerate, driver)
    subset, hits, negatives, corners = select_subset(rates, source)
    swept = len(rates)

    lines = [comment(text) for text in BANNER]
    lines += [
        field("upstream_commit", d1.UPSTREAM_COMMIT),
        field("upstream_path", d1.UPSTREAM_PATH),
        field("upstream_blob", d1.UPSTREAM_BLOB_ID),
        field("driver", "cpp/tools/harvest_d1.cc"),
        field("driver_degenerate", "cpp/tools/harvest_d1_degenerate.cc"),
        comment(),
    ]
    lines += [comment(text) for text in BUILD_PREAMBLE]
    lines += [field("build", build[0])] + [continuation(text) for text in build[1:]]
    lines += [
        comment(),
        field(
            "sweep",
            f"Z {d1.SWEEP_Z_MIN}..{d1.SWEEP_Z_MAX} x A {d1.SWEEP_A_MIN}..{d1.SWEEP_A_MAX} = "
            f"{swept} points, row-major, Z ascending outermost",
        ),
        field("digest_rule", DIGEST_RULE[0]),
    ]
    lines += [continuation(text) for text in DIGEST_RULE[1:]]
    lines += [field("fullsweep_sha256", sweep_digest(rates)), comment()]
    lines += [comment(text) for text in SUBSET_PREAMBLE]
    lines += [
        field(
            "subset",
            f"{len(subset)} points = {hits} table hits + {negatives} first-negative + "
            f"{corners} corners, deduplicated",
        ),
        field("columns", COLUMNS),
    ]
    lines += [f"{z} {a} {rates[z, a]}" for z, a in subset]
    lines += [
        comment(),
        field("zeff", f"Z {min(zeff)}..{max(zeff)}, pinning the clamp at both ends"),
    ]
    lines += [f"ZEFF {z} {zeff[z]}" for z in sorted(zeff)]
    lines += [comment(), field("degenerate", DEGENERATE_RULE[0])]
    lines += [continuation(text) for text in DEGENERATE_RULE[1:]]
    lines += degenerate_lines
    lines += ["#END"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", type=Path, required=True, help="harvest_d1 output")
    parser.add_argument("--degenerate", type=Path, required=True, help="harvest_d1_degenerate output")
    parser.add_argument(
        "--build",
        action="append",
        required=True,
        metavar="LINE",
        help="one line of the build description, repeatable; recorded in the header verbatim",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO / d1.VENDORED_RELPATH,
        help="the vendored upstream source the table hits are read from",
    )
    parser.add_argument(
        "--driver-degenerate",
        type=Path,
        default=Path(__file__).resolve().parent / "harvest_d1_degenerate.cc",
        help="the driver whose declared probe set the degenerate harvest must match",
    )
    parser.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    args = parser.parse_args(argv)

    text = render(args.sweep, args.degenerate, args.build, args.source, args.driver_degenerate)
    if args.output:
        args.output.write_text(text, encoding="ascii", newline="\n")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
