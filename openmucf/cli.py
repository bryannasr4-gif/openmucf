"""openmucf.cli -- the command-line front end (stdlib argparse only, no new dependency).

Installed as the ``openmucf`` console script (``[project.scripts]``). Subcommands:

* ``openmucf reproduce <case-id>`` -- run one bench case and print its result.
* ``openmucf reproduce --all``      -- run every case.
* ``openmucf validate``             -- print the trust-gate table (same content as VALIDATION.md).
* ``openmucf --version``            -- print the package version.

Exit code: 0 on PASS / DEFERRED / PENDING, 1 if any run case FAILs, 2 on a usage error.
"""

from __future__ import annotations

import argparse

from . import __version__, bench, load_rates, validate


def _format_result(r: bench.BenchResult) -> str:
    return (
        f"[{r.verdict}] {r.id} ({r.type})\n"
        f"  predicted: {r.predicted}\n"
        f"  expected:  {r.expected}\n"
        f"  tolerance: {r.tolerance}\n"
        f"  source:    {r.source}\n"
        f"  note:      {r.note}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="openmucf",
        description="OpenMuCF -- reproduce the muCF-Bench cases and the validation trust gate.",
    )
    parser.add_argument("--version", action="version", version=f"openmucf {__version__}")
    sub = parser.add_subparsers(dest="command")

    rep = sub.add_parser("reproduce", help="run one bench case (or --all) and print the result")
    rep.add_argument("case_id", nargs="?", help="a validation id or a JSON reproduction case id")
    rep.add_argument("--all", action="store_true", help="run every registered case")

    sub.add_parser("validate", help="print the validation trust-gate table")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    rates = load_rates()

    if args.command == "validate":
        print(validate.report_markdown(validate.run(rates)))
        return 0

    # args.command == "reproduce"
    if args.all:
        results = bench.run_all(rates)
        print("\n\n".join(_format_result(r) for r in results))
        return 1 if any(r.verdict == "FAIL" for r in results) else 0
    if not args.case_id:
        parser.error("reproduce needs a case id or --all")
    try:
        result = bench.run_case(rates, args.case_id)
    except KeyError:
        ids = ", ".join(bench.case_ids(rates))
        parser.error(f"unknown case id {args.case_id!r}; known ids: {ids}")
    print(_format_result(result))
    return 1 if result.verdict == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
