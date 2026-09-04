"""Do not buy a long option over an earnings print.

The pipeline already fetched the earnings date and computed whether it fell inside the
contract's life -- then used the answer only as a label on the alert. So on 2026-09-03 at
15:45 it produced a LULU call breakout at $121.31, fifteen minutes before that print.
LULU opened the next morning near $99. Nothing about earnings stopped it: the Velocity
Gate rejected it on an unrelated volatility threshold, and the same setup had been saved
by the Exhaustion Gate six days earlier. Two near-misses, both by luck.

The structure is what makes it a rule rather than a preference. The gap direction is
unpredictable, and IV is at its peak on the day you pay for the contract, so the crush
works against a long option even when the direction is right.

These tests pin the window arithmetic. The gate itself is a two-line branch on the value
they compute; what can silently go wrong is the boundary and the timezone.
"""
import pytest

from src.data.earnings_calendar import EarningsCalendar
from src.execution.signal_pipeline import EARNINGS_GATE

# The shipped rule, not a copy of it. These tests used to assert against a local
# reimplementation of the comparison, so a boundary change in the pipeline left them
# green.
in_window = EarningsCalendar.earnings_in_window


class TestTheWindow:
    def test_the_lulu_case(self):
        """Signalled 2026-09-03, 7-14 DTE contract, print that evening."""
        assert in_window("2026-09-03", "2026-09-18", "2026-09-03") is True

    def test_a_print_after_expiry_is_not_our_problem(self):
        assert in_window("2026-09-25", "2026-09-18", "2026-09-04") is False

    def test_a_print_already_behind_us_is_not_either(self):
        assert in_window("2026-09-03", "2026-09-18", "2026-09-04") is False

    def test_earnings_on_expiry_day_still_counts(self):
        """The contract is alive that morning, so the gap still lands on it."""
        assert in_window("2026-09-18", "2026-09-18", "2026-09-04") is True

    def test_earnings_today_counts(self):
        assert in_window("2026-09-04", "2026-09-18", "2026-09-04") is True


class TestTheSwitch:
    def test_it_is_on_by_default(self):
        """Unlike every other switch here. A structural loss is not a preference."""
        assert EARNINGS_GATE is True

    @pytest.mark.parametrize("value,expected", [
        ("off", False), ("0", False), ("false", False), ("no", False),
        ("on", True), ("", True), ("anything-else", True),
    ])
    def test_it_honours_the_env(self, value, expected, monkeypatch):
        import importlib

        import src.execution.signal_pipeline as sp
        monkeypatch.setenv("EARNINGS_GATE", value)
        importlib.reload(sp)
        try:
            assert sp.EARNINGS_GATE is expected
        finally:
            monkeypatch.delenv("EARNINGS_GATE", raising=False)
            importlib.reload(sp)


class TestItFailsOpen:
    """The calendar does not know every ticker -- AAPL returns no date at all. Blocking
    every unknown would silence most alerts, so an unknown date must not block."""

    @pytest.mark.parametrize("earnings", [None, "", "not-a-date", "2026-13-45"])
    def test_an_unusable_earnings_date_does_not_block(self, earnings):
        assert in_window(earnings, "2026-09-18", "2026-09-04") is False

    @pytest.mark.parametrize("expiration", [None, "", "not-a-date"])
    def test_an_unusable_expiration_does_not_block(self, expiration):
        assert in_window("2026-09-10", expiration, "2026-09-04") is False

    def test_it_defaults_to_market_time_when_today_is_omitted(self):
        """Host time on a UTC box rolls over five hours early and can decide a print is
        already behind us. Passing no `today` must not fall back to date.today()."""
        import inspect

        src = inspect.getsource(EarningsCalendar.earnings_in_window)
        assert "market_today_date()" in src
        assert "date.today()" not in src


class TestTheLifecycleRule:
    def test_the_gate_removes_the_speculative_position(self):
        """Every gate that returns early must call remove_position first, or the alert
        is suppressed while the position lingers and is tracked as if it were held."""
        import inspect

        from src.execution import signal_pipeline

        src = inspect.getsource(signal_pipeline)
        gate = src.split("BLOCKED by Earnings Gate")[1].split("return")[0]
        assert "remove_position" in gate
