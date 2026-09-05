"""Do not propose puts into an uptrend, or calls into a downtrend.

Two conditions on the daily frame, and requiring both is what makes it work:

    sma_spread  = (SMA20 - SMA60) / SMA60   which way the trend points
    sma20_ratio = close / SMA20             which side of it price sits on

Measured over 328 alerts with three sessions of forward prices:

    aligned on both   n=187  win 50.8%  median +0.02%
    not aligned       n=141  win 42.6%  median -0.93%
    trend alone       n=218  win 48.6%  vs 44.5% against -- barely separates

So the gate needs both. Testing the direction logic matters more than testing the
thresholds: an inverted comparison would silently keep exactly the alerts it exists to
remove, and nothing downstream would notice.
"""
import importlib

import pytest

import src.execution.signal_pipeline as sp


def aligned(direction, spread, ratio):
    """The gate's decision, expressed as 'is this alert allowed through'."""
    uptrend = spread > 0 and ratio > 1.0
    downtrend = spread < 0 and ratio < 1.0
    return uptrend if direction == "Long" else downtrend


class TestTheDirectionLogic:
    def test_a_call_in_a_clean_uptrend_passes(self):
        assert aligned("Long", spread=0.05, ratio=1.03) is True

    def test_a_put_in_a_clean_downtrend_passes(self):
        assert aligned("Short", spread=-0.05, ratio=0.97) is True

    def test_a_put_into_an_uptrend_is_blocked(self):
        """The complaint that prompted this: a put on something about to break out."""
        assert aligned("Short", spread=0.05, ratio=1.03) is False

    def test_a_call_into_a_downtrend_is_blocked(self):
        assert aligned("Long", spread=-0.05, ratio=0.97) is False


class TestBothConditionsAreRequired:
    def test_trend_up_but_price_below_sma20_blocks_a_call(self):
        """A pullback under SMA20 in an uptrend is the 'not aligned' bucket that won
        42.6%. Trend alone would have let this through."""
        assert aligned("Long", spread=0.05, ratio=0.98) is False

    def test_trend_down_but_price_above_sma20_blocks_a_put(self):
        assert aligned("Short", spread=-0.05, ratio=1.02) is False

    def test_price_right_but_trend_wrong_blocks_a_call(self):
        assert aligned("Long", spread=-0.02, ratio=1.05) is False


class TestTheBoundaries:
    @pytest.mark.parametrize("direction", ["Long", "Short"])
    def test_a_flat_trend_blocks_both_sides(self, direction):
        """spread == 0 is neither an uptrend nor a downtrend, so nothing is aligned."""
        assert aligned(direction, spread=0.0, ratio=1.0) is False

    def test_price_exactly_on_sma20_is_not_above_it(self):
        assert aligned("Long", spread=0.05, ratio=1.0) is False

    def test_price_exactly_on_sma20_is_not_below_it(self):
        assert aligned("Short", spread=-0.05, ratio=1.0) is False


class TestTheSwitch:
    def test_it_is_on_by_default(self):
        assert sp.TREND_GATE is True

    @pytest.mark.parametrize("value,expected", [
        ("off", False), ("0", False), ("false", False), ("no", False),
        ("on", True), ("", True),
    ])
    def test_it_honours_the_env(self, value, expected, monkeypatch):
        monkeypatch.setenv("TREND_GATE", value)
        importlib.reload(sp)
        try:
            assert sp.TREND_GATE is expected
        finally:
            monkeypatch.delenv("TREND_GATE", raising=False)
            importlib.reload(sp)


class TestItFailsOpen:
    def test_it_is_skipped_when_the_daily_features_are_missing(self):
        """An unmeasured trend is not a counter-trend signal. Before the bar-window fix
        these features were absent for every single alert, so a gate that blocked on
        missing data would have silenced the bot entirely."""
        import inspect

        src = inspect.getsource(sp.SignalPipeline.process_signal)
        gate = src.split("Trend gate")[1]
        assert "TREND_GATE and not missing" in gate


class TestTheLifecycleRule:
    def test_it_removes_the_speculative_position(self):
        """It sits after add_position(), so it must clean up or the alert is suppressed
        while the position lingers and is tracked as though it were held."""
        import inspect

        src = inspect.getsource(sp)
        gate = src.split("BLOCKED by Trend Gate")[1].split("return")[0]
        assert "remove_position" in gate
