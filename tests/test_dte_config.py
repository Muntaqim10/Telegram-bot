"""The configurable option DTE window.

The bot used to hardcode 14-21 days. That is this account's worst bucket (57 contracts,
42.1% win, -$2,757) while 0-1 DTE is its best, so the window has to be a setting -- and
everything that echoes it, including alert headers, has to follow the setting rather than
a stale string.

These tests reload modules under a modified environment, so the autouse fixture restores
the default state afterwards or later tests inherit a 0-2 DTE world.
"""
import importlib
import inspect

import pytest


def reload_pricer(monkeypatch, **env):
    for key in ("OPTION_MIN_DTE", "OPTION_MAX_DTE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    import src.execution.options_pricer as op
    return importlib.reload(op)


@pytest.fixture(autouse=True)
def restore_modules():
    """Reloading options_pricer rebinds DTE_LABEL, which the formatter imported by
    value. Both must go back to their defaults after every test in this module."""
    yield
    import src.execution.options_pricer as op
    import src.utils.telegram_formatter as fmt
    importlib.reload(op)
    importlib.reload(fmt)


class TestDefaults:
    def test_preserve_the_existing_behaviour(self, monkeypatch):
        op = reload_pricer(monkeypatch)
        assert (op.TARGET_MIN_DTE, op.TARGET_MAX_DTE) == (14, 21)
        assert op.TARGET_IDEAL_DTE == 17, "ideal is the midpoint, not a fixed 18"
        assert op.DTE_LABEL == "14-21 DTE"


class TestShortWindow:
    def test_is_honoured(self, monkeypatch):
        op = reload_pricer(monkeypatch, OPTION_MIN_DTE=0, OPTION_MAX_DTE=2)
        assert (op.TARGET_MIN_DTE, op.TARGET_MAX_DTE) == (0, 2)
        assert op.TARGET_IDEAL_DTE == 1
        assert op.FALLBACK_FLOOR_DTE == 0, "the floor must not go negative"
        assert op.DTE_LABEL == "0-2 DTE"

    def test_alert_headers_follow_it(self, monkeypatch):
        reload_pricer(monkeypatch, OPTION_MIN_DTE=0, OPTION_MAX_DTE=2)
        import src.utils.telegram_formatter as fmt
        importlib.reload(fmt)
        from src.models.signal import TradeSignal

        msg = fmt.format_telegram_alert(TradeSignal(
            ticker="PLTR", price=150.0, signal_direction="Long",
            strategy_type="Volume Breakout", timestamp="2026-09-03 10:00:00",
            conviction="⚪ UNSCORED", win_probability=0.36, stop_loss=145.0,
            take_profit=160.0, catalyst="t", historical_edge="n/a", context_score="x"))

        assert "0-2 DTE" in msg
        assert "14-21 DTE" not in msg, "a stale literal is still in the header"


class TestMalformedConfiguration:
    def test_inverted_bounds_are_swapped_not_obeyed(self, monkeypatch):
        op = reload_pricer(monkeypatch, OPTION_MIN_DTE=30, OPTION_MAX_DTE=7)
        assert (op.TARGET_MIN_DTE, op.TARGET_MAX_DTE) == (7, 30)

    def test_non_numeric_and_negative_fall_back(self, monkeypatch):
        op = reload_pricer(monkeypatch, OPTION_MIN_DTE="soon", OPTION_MAX_DTE=-5)
        assert (op.TARGET_MIN_DTE, op.TARGET_MAX_DTE) == (14, 21)


def test_the_pipeline_no_longer_hardcodes_the_window():
    import src.execution.signal_pipeline as sp
    source = inspect.getsource(sp.SignalPipeline.process_signal)
    assert "min_dte=14" not in source
    assert "get_target_expiration(ticker, session=" in source
