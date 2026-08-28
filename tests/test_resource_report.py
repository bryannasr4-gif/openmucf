"""Drill the session resource report in `conftest.py` on BOTH platform branches, from any platform.

The report itself asserts nothing, so nothing else would ever exercise it -- and its one piece of
real logic is a unit conversion that is only wrong on the platform you are not running on:
`ru_maxrss` is bytes on macOS and kibibytes on Linux. A factor-1024 error there would under-report
a macOS run by three orders of magnitude, which is exactly the direction that would hide a memory
problem rather than surface one. So both branches are exercised here with a stub, on every platform
including Windows, where the reporter itself is inert.
"""

from __future__ import annotations

import sys
import types

import conftest


class _Reporter:
    """Stands in for pytest's terminal reporter; records what would have been printed."""

    def __init__(self):
        self.lines: list[str] = []

    def write_line(self, line):
        self.lines.append(line)


def _stub_resource(monkeypatch, ru_maxrss):
    """Install a fake `resource` module whose getrusage reports `ru_maxrss`."""
    mod = types.SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _who: types.SimpleNamespace(ru_maxrss=ru_maxrss),
    )
    monkeypatch.setitem(sys.modules, "resource", mod)


def test_macos_reports_ru_maxrss_as_bytes(monkeypatch):
    _stub_resource(monkeypatch, 3 * (1 << 30))  # 3 GiB, already in bytes
    monkeypatch.setattr(conftest.sys, "platform", "darwin")
    assert conftest._peak_rss_bytes() == 3 * (1 << 30)


def test_linux_reports_ru_maxrss_as_kibibytes(monkeypatch):
    _stub_resource(monkeypatch, 3 * (1 << 20))  # 3 GiB expressed in KiB
    monkeypatch.setattr(conftest.sys, "platform", "linux")
    assert conftest._peak_rss_bytes() == 3 * (1 << 30)


def test_the_two_platforms_do_not_agree_on_the_raw_number(monkeypatch):
    """The whole point of the branch: the SAME raw value means different things on the two OSes."""
    raw = 5 * (1 << 20)
    _stub_resource(monkeypatch, raw)
    monkeypatch.setattr(conftest.sys, "platform", "darwin")
    as_mac = conftest._peak_rss_bytes()
    _stub_resource(monkeypatch, raw)
    monkeypatch.setattr(conftest.sys, "platform", "linux")
    assert conftest._peak_rss_bytes() == as_mac * 1024


def test_missing_resource_module_reports_nothing_rather_than_a_wrong_number(monkeypatch):
    """Windows has no `resource`; the reporter must stay silent, not print a fabricated figure."""

    def _no_resource(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("no resource module on this platform")
        return _real_import(name, *args, **kwargs)

    _real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    monkeypatch.setattr("builtins.__import__", _no_resource)
    assert conftest._peak_rss_bytes() is None

    rep = _Reporter()
    conftest.pytest_terminal_summary(rep)
    assert rep.lines == []


def test_summary_line_states_the_share_of_physical_memory(monkeypatch):
    monkeypatch.setattr(conftest, "_peak_rss_bytes", lambda: 3 * (1 << 30))
    monkeypatch.setattr(conftest, "_physical_memory_bytes", lambda: 12 * (1 << 30))
    rep = _Reporter()
    conftest.pytest_terminal_summary(rep)
    assert rep.lines == ["peak RSS 3.00 GiB of 12.00 GiB physical (25.0%)"]


def test_summary_line_omits_the_share_when_physical_memory_is_unknown(monkeypatch):
    monkeypatch.setattr(conftest, "_peak_rss_bytes", lambda: 1 << 30)
    monkeypatch.setattr(conftest, "_physical_memory_bytes", lambda: None)
    rep = _Reporter()
    conftest.pytest_terminal_summary(rep)
    assert rep.lines == ["peak RSS 1.00 GiB"]
