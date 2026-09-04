"""Alerts must not be lost to a missing dict key.

signal_pipeline read the alert time as a bare `signal["timestamp"]` subscript while every
other optional field on the same call used `.get()`. momentum_movers does not emit that
key, so the subscript raised KeyError -- and it raised *after* add_position() had already
registered the speculative position and *before* the alert was dispatched.

main.py wraps process_signal in `except Exception` and logs a single line, so the failure
was silent: no alert reached Telegram, no row reached trade_log, and the position stayed
in active_positions forever. Over one session that cost 49 tickers' alerts and left 24
positions behind that had never been proposed to anyone.

These tests pin both halves: the producer emits the key, and the consumer survives a
signal that lacks it.
"""
from datetime import datetime

import pytest

from src.strategy.momentum_movers import MomentumMoversStrategy


def momentum_signal(**overrides):
    """A signal shaped exactly as momentum_movers emits one."""
    signal = MomentumMoversStrategy()._build_signal(
        ticker="NVDA",
        direction="Long",
        entry_price=230.28,
        catalyst_type="HOD Velocity Breakout",
        invalidation_level=228.0,
        z_vol=2.4,
        rsi=63.1,
        vwap=229.4,
    )
    signal.update(overrides)
    return signal


def format_timestamp(signal):
    """The coercion signal_pipeline applies when building the TradeSignal."""
    value = signal.get("timestamp") or datetime.now()
    return value.strftime("%Y-%m-%d %H:%M:%S") if hasattr(value, "strftime") else str(value)


class TestTheProducer:
    def test_momentum_movers_emits_a_timestamp(self):
        """The strategy that broke the pipeline now carries the key the others do."""
        assert "timestamp" in momentum_signal()

    def test_it_is_a_real_datetime_not_a_string(self):
        """donchian_daily emits datetime.now(); the pipeline formats via strftime."""
        assert isinstance(momentum_signal()["timestamp"], datetime)

    def test_every_strategy_agrees_on_the_key(self):
        """A new strategy omitting this is the exact bug that lost 49 tickers."""
        import inspect

        from src.strategy import donchian_daily, extended_hours_scanner, orb_intraday

        for module in (momentum_movers_module(), donchian_daily,
                       extended_hours_scanner, orb_intraday):
            source = inspect.getsource(module)
            assert '"timestamp"' in source, f"{module.__name__} emits no timestamp"


def momentum_movers_module():
    from src.strategy import momentum_movers
    return momentum_movers


class TestTheConsumer:
    def test_a_signal_without_the_key_does_not_raise(self):
        """The regression. A bare subscript raised KeyError and lost the alert."""
        signal = momentum_signal()
        del signal["timestamp"]
        assert format_timestamp(signal)  # must not raise

    def test_the_fallback_is_a_formatted_string(self):
        signal = momentum_signal()
        del signal["timestamp"]
        # "%Y-%m-%d %H:%M:%S" -- what add_trade writes into trade_log.timestamp
        datetime.strptime(format_timestamp(signal), "%Y-%m-%d %H:%M:%S")

    def test_a_supplied_datetime_is_preserved_not_replaced(self):
        """The fallback must not overwrite a real signal time."""
        when = datetime(2026, 9, 3, 15, 38, 0)
        assert format_timestamp(momentum_signal(timestamp=when)) == "2026-09-03 15:38:00"

    def test_a_string_timestamp_passes_through(self):
        """extended_hours_scanner supplies one already formatted."""
        assert format_timestamp(momentum_signal(timestamp="2026-09-03 15:38:00")) == \
            "2026-09-03 15:38:00"

    @pytest.mark.parametrize("empty", [None, ""])
    def test_an_empty_timestamp_falls_back_rather_than_writing_blank(self, empty):
        """`or` rather than a `is None` check: a blank string is not an alert time."""
        datetime.strptime(format_timestamp(momentum_signal(timestamp=empty)),
                          "%Y-%m-%d %H:%M:%S")


class TestTheSubscriptIsGone:
    def test_the_pipeline_no_longer_subscripts_timestamp(self):
        """Guards the specific line. `.get()` is the convention everywhere else here."""
        import inspect

        from src.execution import signal_pipeline

        # Comments quote the old line deliberately, so judge the code alone.
        code = [line for line in inspect.getsource(signal_pipeline).splitlines()
                if not line.lstrip().startswith("#")]
        assert 'signal["timestamp"]' not in "\n".join(code), \
            "a bare subscript here loses the alert and strands the position"
