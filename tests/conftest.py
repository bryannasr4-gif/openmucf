"""Session-level resource report for the test suite. Reports only -- asserts nothing, gates nothing.

WHY THIS EXISTS. The weekly `slow` job runs three long MCMC gates in ONE process: the SBC rank
uniformity run (200 rounds x 2 chains), the sd-contraction refit, and the twin interval-calibration
run (200 replicas). On 2026-08-24 the macos-15 leg of that job was lost 45 minutes into the suite;
GitHub attributed it to "the hosted runner lost communication with the server. Anything in your
workflow that terminates the runner process, starves it for CPU/Memory, or blocks its network
access can cause this error." Nothing in the suite reported how much memory it had used, so the
one hypothesis that annotation raises -- memory starvation -- could not be checked afterwards, on
that run or on any of the green ones. The hosted macOS runner has roughly half the memory of the
Linux one, which is exactly the asymmetry that would make such a failure single-platform.

So the numbers are printed on every run, pass or fail: a peak-memory figure that is never recorded
cannot be compared against the run that died. This hook is deliberately assertion-free -- a
resource budget asserted without a measured basis is a gate nobody can defend, and the basis is
what this is here to collect.
"""

from __future__ import annotations

import os
import sys


def _peak_rss_bytes() -> int | None:
    """Peak resident set size of this process in BYTES, or None where it cannot be read.

    `ru_maxrss` is reported in BYTES on macOS (Darwin) and in KIBIBYTES on Linux -- Darwin is the
    outlier here, and the *BSDs report kibibytes like Linux, which is why the branch tests for
    darwin rather than for "BSD". The units genuinely differ between the two platforms this job
    runs on, so reading it as one unit on both is a silent factor-1024 error -- in the direction
    that would make a macOS run look 1024x smaller than it is, i.e. it would hide precisely the
    thing this file exists to surface.
    """
    try:
        import resource  # POSIX only; absent on Windows
    except ImportError:
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def _physical_memory_bytes() -> int | None:
    """Total physical memory in bytes, or None if the platform will not say."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def pytest_terminal_summary(terminalreporter) -> None:
    """Print peak RSS (and the share of physical memory it used) after the run summary."""
    peak = _peak_rss_bytes()
    if peak is None:
        return  # Windows: no `resource` module. Silence beats a wrong number.
    gib = float(1 << 30)
    total = _physical_memory_bytes()
    line = f"peak RSS {peak / gib:.2f} GiB"
    if total:
        line += f" of {total / gib:.2f} GiB physical ({100.0 * peak / total:.1f}%)"
    terminalreporter.write_line(line)
