"""The daily bar window has to clear the indicator warm-up, or nothing gets scored.

compute_indicators() returns an empty frame below 60 bars, because SMA_60 needs 60. The
fetch window is expressed in CALENDAR days, and 80 calendar days is only ~57 trading
days -- so the guard tripped for every ticker on every scan, forever.

Both caches are populated after that call:

    df = self.compute_indicators(df)
    if df.empty:
        return []                       # <-- always taken
    self.atr_cache[ticker] = ...
    self.feature_cache[ticker] = ...

so atr_cache and feature_cache stayed empty for the life of the process. Every alert was
marked UNSCORED (620 in one session, 0 scored), the model never saw a live signal, and
every trailing stop fell back to a percentage of price instead of a measured ATR.

The bug was a single default. These tests pin the arithmetic that made it wrong.
"""
import numpy as np
import pandas as pd
import pytest

from src.strategy.donchian_daily import DonchianSwingStrategy

# Trading days per calendar day, ignoring holidays: 5 weekdays in 7.
TRADING_DAY_RATIO = 5 / 7
# compute_indicators' hard floor -- the longest rolling window it computes.
WARMUP_BARS = 60


def bars(n, start=100.0):
    """A synthetic daily frame shaped like Tradier's, with no NaNs of its own."""
    idx = pd.date_range("2026-01-01", periods=n, freq="D", name="date")
    close = start + np.arange(n, dtype=float)
    return pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(n, 1_000_000, dtype=float),
    }, index=idx)


class TestTheWarmupGuard:
    def test_below_sixty_bars_yields_nothing(self):
        """The behaviour that silently emptied both caches."""
        assert DonchianSwingStrategy.compute_indicators(bars(WARMUP_BARS - 1)).empty

    def test_at_sixty_bars_it_computes(self):
        assert not DonchianSwingStrategy.compute_indicators(bars(WARMUP_BARS)).empty

    def test_the_old_window_was_short_by_design(self):
        """80 calendar days is ~57 trading days: below the floor, every single scan."""
        assert 80 * TRADING_DAY_RATIO < WARMUP_BARS


class TestTheConfiguredWindow:
    def test_clears_the_warmup(self):
        lookback = DonchianSwingStrategy.DEFAULT_LOOKBACK_DAYS
        assert lookback * TRADING_DAY_RATIO >= WARMUP_BARS, (
            f"{lookback} calendar days is ~{lookback * TRADING_DAY_RATIO:.0f} trading "
            f"days, under the {WARMUP_BARS} compute_indicators needs"
        )

    def test_leaves_room_for_holidays(self):
        """~9 market holidays a year, so clearing the floor exactly is not enough."""
        expected = DonchianSwingStrategy.DEFAULT_LOOKBACK_DAYS * TRADING_DAY_RATIO
        assert expected - WARMUP_BARS >= 10, (
            f"only {expected - WARMUP_BARS:.0f} bars of margin; a holiday-heavy stretch "
            f"would empty the caches again"
        )

    def test_is_the_default_the_fetch_actually_uses(self):
        """A constant nothing reads would leave the bug in place."""
        import inspect

        sig = inspect.signature(DonchianSwingStrategy.fetch_daily_bars)
        assert sig.parameters["lookback_days"].default == \
            DonchianSwingStrategy.DEFAULT_LOOKBACK_DAYS


class TestTheCachesTheWindowFeeds:
    """What the lookback fix actually unblocked.

    The extractor's own contract -- which features it returns, and that it refuses an
    unusable frame -- is covered by TestDailyFeatures in test_model_honesty.py. This
    only asserts the link that was broken: a frame built from the configured window
    survives the warm-up guard and yields usable values for both caches.
    """

    def test_a_frame_from_the_configured_window_populates_both_caches(self):
        bars_in_window = int(DonchianSwingStrategy.DEFAULT_LOOKBACK_DAYS * TRADING_DAY_RATIO)
        df = DonchianSwingStrategy.compute_indicators(bars(bars_in_window))
        assert not df.empty, "the window no longer clears the warm-up guard"

        atr = df["ATR_14"].iloc[-1]          # what atr_cache stores
        feats = DonchianSwingStrategy().extract_model_features(df)  # what feature_cache stores
        assert not pd.isna(atr)
        assert feats is not None
        for name, value in feats.items():
            assert value is not None and not pd.isna(value), f"{name} is unusable"
