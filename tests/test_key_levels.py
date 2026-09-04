"""The D/W/M level strategy: one idea, one alert type.

The ten headline categories it replaces produced 438 alerts of which 8 are gradable
against a real fill. These tests pin the properties that keep this from re-growing into
the same sprawl: one signal per evaluation, level/timeframe/trigger carried as fields,
and the slowest frame winning when several fire at once.
"""
from datetime import datetime

import pandas as pd
import pytest

from src.strategy.key_levels import BREAK_BUFFER_ATR, KeyLevelStrategy

ATR = 4.0
BREAK = ATR * BREAK_BUFFER_ATR  # 1.0
# A price just clear of the level: past the noise buffer, inside the extension cap.
FRESH_BREAK_PX = 100 + BREAK + 0.1


def frame(rows):
    """Daily OHLC indexed by date, oldest first."""
    idx = pd.date_range("2026-06-01", periods=len(rows), freq="D", name="date")
    return pd.DataFrame(
        [{"open": r[0], "high": r[1], "low": r[2], "close": r[3], "volume": 1e6} for r in rows],
        index=idx,
    )


def flat_history(days=120, high=100.0, low=90.0):
    """A long, featureless history so weekly/monthly periods exist to resample."""
    return frame([(95, high, low, 95)] * days)


@pytest.fixture
def strat():
    return KeyLevelStrategy()


class TestLevels:
    def test_uses_the_prior_day_not_today(self, strat):
        """Today's high is still moving; a level price sets itself can never break."""
        df = frame([(95, 100, 90, 95), (95, 500, 5, 95)])  # today is the wild bar
        assert strat.compute_levels(df)["day"] == {"high": 100.0, "low": 90.0}

    def test_weekly_and_monthly_come_from_completed_periods(self, strat):
        levels = strat.compute_levels(flat_history(120))
        assert {"day", "week", "month"} <= set(levels)

    def test_too_little_history_yields_nothing(self, strat):
        assert strat.compute_levels(frame([(95, 100, 90, 95)])) == {}
        assert strat.compute_levels(pd.DataFrame()) == {}


class TestBreak:
    def test_above_the_level_is_a_long(self, strat):
        out = strat.evaluate("NVDA", 100 + BREAK + 0.1, flat_history(), ATR)
        assert out[0]["direction"] == "Long"
        assert out[0]["level_trigger"] == "BREAK"

    def test_below_the_level_is_a_short(self, strat):
        out = strat.evaluate("NVDA", 90 - BREAK - 0.1, flat_history(), ATR)
        assert out[0]["direction"] == "Short"

    def test_a_close_sitting_on_the_level_does_not_fire(self, strat):
        """Without the buffer, a level mid-consolidation fires on every wobble."""
        assert strat.evaluate("NVDA", 100.05, flat_history(), ATR) == []

    def test_the_buffer_scales_with_volatility(self, strat):
        """The same 0.6 move is noise on a wide-range name and a break on a quiet one.

        Both bounds move with ATR, so the test price has to sit inside the quiet name's
        band (100.25 < px <= 101.0 at ATR 1.0) while staying under the wide name's
        0.25-ATR noise buffer of 5.0.
        """
        assert strat.evaluate("NVDA", 100.6, flat_history(), atr=20.0) == []
        assert strat.evaluate("NVDA", 100.6, flat_history(), atr=1.0) != []

    def test_already_extended_is_not_a_fresh_break(self, strat):
        """A live scan fired SNPS 29 points below a level it broke days earlier. Once
        price is more than an ATR past the level the move has happened; the stop would
        be miles away and what is left is chasing."""
        assert strat.evaluate("NVDA", 100 + ATR * 1.5, flat_history(), ATR) == []
        assert strat.evaluate("NVDA", 90 - ATR * 1.5, flat_history(), ATR) == []

    def test_just_past_the_level_still_fires(self, strat):
        """The cap must not swallow the setup it exists to isolate."""
        assert strat.evaluate("NVDA", 100 + ATR * 0.5, flat_history(), ATR) != []


class TestReject:
    def test_tagging_the_high_then_closing_back_is_a_short(self, strat):
        out = strat.evaluate("NVDA", 98.0, flat_history(), ATR, session_high=100.5, session_low=97.0)
        assert out[0]["direction"] == "Short"
        assert out[0]["level_trigger"] == "REJECT"

    def test_tagging_the_low_then_closing_back_is_a_long(self, strat):
        out = strat.evaluate("NVDA", 92.0, flat_history(), ATR, session_high=95.0, session_low=89.5)
        assert out[0]["direction"] == "Long"

    def test_never_reaching_the_level_is_not_a_rejection(self, strat):
        assert strat.evaluate("NVDA", 98.0, flat_history(), ATR,
                              session_high=99.0, session_low=97.0) == []


class TestOneAlertNotTen:
    def test_only_one_signal_per_evaluation(self, strat):
        """Breaking day, week and month at once must not send three alerts."""
        assert len(strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR)) == 1

    def test_the_slowest_frame_wins(self, strat):
        out = strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR)
        assert out[0]["level_timeframe"] == "month"

    def test_the_headline_is_one_of_two_strings(self, strat):
        out = strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR)
        assert out[0]["catalyst_type"] in {"LEVEL BREAK", "LEVEL REJECT"}

    @pytest.mark.parametrize("field", ["level_timeframe", "level_name", "level_price",
                                       "level_trigger"])
    def test_the_detail_is_a_field_not_a_category(self, strat, field):
        assert field in strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR)[0]


class TestPipelineContract:
    """Keys process_signal reads. A missing one is silently swallowed by main.py."""

    @pytest.mark.parametrize("key", ["ticker", "direction", "entry_price", "catalyst_type",
                                     "timestamp", "invalidation_level", "hod", "lod"])
    def test_emits(self, strat, key):
        assert key in strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR)[0]

    def test_timestamp_is_a_datetime(self, strat):
        """momentum_movers omitted this and every one of its alerts was lost."""
        assert isinstance(strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR)[0]["timestamp"],
                          datetime)

    def test_the_level_is_the_stop(self, strat):
        out = strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR)[0]
        assert out["invalidation_level"] == out["level_price"]

    def test_model_features_ride_along_so_alerts_arrive_scored(self, strat):
        feats = {"sma_spread": 0.05, "sma20_ratio": 1.06, "rsi_14": 55.4}
        out = strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR, features=feats)[0]
        assert all(out[k] == v for k, v in feats.items())

    def test_missing_features_are_not_faked(self, strat):
        """Scoring constants is what made every intraday alert land on two numbers."""
        out = strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR, features=None)[0]
        assert "rsi_14" not in out


class TestNoRepeatAlerts:
    def test_the_same_level_fires_once_a_day(self, strat):
        sig = strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR)[0]
        assert strat.already_fired("NVDA", sig) is False
        assert strat.already_fired("NVDA", sig) is True

    def test_a_different_ticker_is_unaffected(self, strat):
        sig = strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR)[0]
        strat.already_fired("NVDA", sig)
        assert strat.already_fired("AAPL", sig) is False

    def test_the_daily_reset_clears_it(self, strat):
        sig = strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), ATR)[0]
        strat.already_fired("NVDA", sig)
        strat.reset_daily_state()
        assert strat.already_fired("NVDA", sig) is False


class TestDegradesQuietly:
    @pytest.mark.parametrize("atr", [0, None, -1])
    def test_no_atr_means_no_signal_rather_than_a_guess(self, strat, atr):
        assert strat.evaluate("NVDA", FRESH_BREAK_PX, flat_history(), atr) == []

    def test_no_history_means_no_signal(self, strat):
        assert strat.evaluate("NVDA", FRESH_BREAK_PX, pd.DataFrame(), ATR) == []
