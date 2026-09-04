"""Trade the daily, weekly and monthly highs and lows. One idea, one alert type.

This replaces the ORB / CRB / fakeout / momentum setups, which between them produced ten
headline categories over 438 alerts -- of which only 8 can be graded against a real fill.
Ten categories over eight gradable outcomes means no category can ever earn its keep, so
the fix is fewer things measured better rather than more things measured not at all.

The levels are the ones every desk watches, in the order that matters:

    prior month high / low   the slowest frame -- direction
    prior week  high / low   the middle frame -- trend
    prior day   high / low   the fast frame   -- today's setup

Two triggers at any level:

    BREAK    price closes through the level                -> go with it
    REJECT   price tags the level and closes back inside   -> fade it

Both are emitted, tagged, so which one earns its keep is settled by money rather than by
opinion. Level, timeframe and trigger are FIELDS on one alert type, not separate
headlines -- that is what keeps this from re-growing into ten categories, and it means a
question like "do breaks work" is answered across every timeframe at once instead of
being split four ways.

Bars come from DonchianSwingStrategy's cache, so this costs no extra API calls, and the
daily model features ride along on the same frame -- these alerts arrive scored rather
than UNSCORED.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

# How far through a level price must close before it counts as a break, as a share of
# daily ATR. Without it, a level sitting mid-consolidation fires on every tick that
# wobbles across it. A quarter of an ATR is roughly a normal bar's noise.
BREAK_BUFFER_ATR = 0.25

# ...and how far past it price may be and still count. A break is only tradeable near the
# level: once price is an ATR beyond it the move has happened, the stop is miles away, and
# what is left is chasing. Measured on a live 120-ticker scan, without this cap 77 of 120
# names triggered -- SNPS fired 29 points below a level it had broken days earlier. The
# cap is what turns "is price on the far side of this level" into "did it just break".
MAX_EXTENSION_ATR = 1.0

# A reject needs price to have actually reached the level intraday and then closed back
# through it by this much ATR -- otherwise every close near a level reads as a rejection.
REJECT_BUFFER_ATR = 0.15

# Slowest first. Only the strongest triggered level fires, so a day that breaks the prior
# day, week and month highs at once produces one monthly alert, not three.
LEVEL_PRECEDENCE = ("month", "week", "day")


class KeyLevelStrategy:
    """Detects breaks of, and rejections at, the prior D/W/M highs and lows."""

    def __init__(self, donchian_strategy=None):
        # Reuses the Donchian bar cache rather than fetching its own. That strategy is
        # kept for exactly this reason: it is the daily-data feed the rest of the engine
        # depends on for ATR and the model features.
        self.donchian = donchian_strategy
        self._fired_today: Dict[str, set] = {}

    # ---------------------------------------------------------------- levels

    @staticmethod
    def compute_levels(daily: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Prior completed day, week and month high/low from a daily OHLC frame.

        'Prior completed' matters: the current period's high is still moving, so using it
        would compare price against a level it sets itself, which can never break.
        """
        if daily is None or daily.empty or len(daily) < 2:
            return {}

        out: Dict[str, Dict[str, float]] = {}

        prev_day = daily.iloc[-2]
        out["day"] = {"high": float(prev_day["high"]), "low": float(prev_day["low"])}

        for name, rule in (("week", "W"), ("month", "ME")):
            periods = daily.resample(rule).agg({"high": "max", "low": "min"}).dropna()
            if len(periods) >= 2:
                prior = periods.iloc[-2]
                out[name] = {"high": float(prior["high"]), "low": float(prior["low"])}

        return out

    # ---------------------------------------------------------------- triggers

    def evaluate(
        self,
        ticker: str,
        price: float,
        daily: pd.DataFrame,
        atr: float,
        session_high: Optional[float] = None,
        session_low: Optional[float] = None,
        features: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Return at most one signal: the strongest level currently triggered.

        session_high/low are today's extremes so far. A reject needs them -- it asks
        whether price reached the level and came back, which a single last price cannot
        answer on its own.
        """
        levels = self.compute_levels(daily)
        if not levels or not atr or atr <= 0:
            return []

        session_high = session_high if session_high is not None else price
        session_low = session_low if session_low is not None else price
        break_buf = atr * BREAK_BUFFER_ATR
        max_ext = atr * MAX_EXTENSION_ATR
        reject_buf = atr * REJECT_BUFFER_ATR

        for timeframe in LEVEL_PRECEDENCE:
            if timeframe not in levels:
                continue
            high = levels[timeframe]["high"]
            low = levels[timeframe]["low"]

            # BREAK: closed clear of the level, but not so far past it that the move has
            # already happened. Both bounds matter -- see MAX_EXTENSION_ATR.
            if high + break_buf < price <= high + max_ext:
                return [self._signal(ticker, "Long", price, timeframe, "BREAK",
                                     "high", high, session_high, session_low, features)]
            if low - max_ext <= price < low - break_buf:
                return [self._signal(ticker, "Short", price, timeframe, "BREAK",
                                     "low", low, session_high, session_low, features)]

            # REJECT: reached the level today, then closed back inside it.
            if session_high >= high and price < high - reject_buf:
                return [self._signal(ticker, "Short", price, timeframe, "REJECT",
                                     "high", high, session_high, session_low, features)]
            if session_low <= low and price > low + reject_buf:
                return [self._signal(ticker, "Long", price, timeframe, "REJECT",
                                     "low", low, session_high, session_low, features)]

        return []

    # ---------------------------------------------------------------- signal

    @staticmethod
    def _signal(ticker, direction, price, timeframe, trigger, side, level,
                session_high, session_low, features) -> Dict[str, Any]:
        """One alert type. Timeframe, level and trigger are fields, not headlines."""
        label = {"day": "Prior Day", "week": "Prior Week", "month": "Prior Month"}[timeframe]
        arrow = "↑" if direction == "Long" else "↓"

        signal = {
            "ticker": ticker,
            "direction": direction,
            "entry_price": float(price),
            # The one headline. Everything that used to be a separate category is below.
            "catalyst_type": f"LEVEL {trigger}",
            "level_timeframe": timeframe,
            "level_name": f"{label} {side.title()}",
            "level_price": float(level),
            "level_trigger": trigger,
            "strategy": "KEY_LEVELS",
            # The level is the natural stop: if price goes back through it, the reason
            # for the trade is gone. No invented percentage.
            "invalidation_level": float(level),
            # The Exhaustion Gate measures how much of the daily range is already spent.
            "hod": float(session_high),
            "lod": float(session_low),
            "timestamp": datetime.now(),
            "tf_confluence": f"{label} {side} {arrow} {trigger}",
        }
        # Daily model features ride along on the frame we already have, so these alerts
        # arrive scored instead of UNSCORED.
        if features:
            signal.update({k: v for k, v in features.items() if v is not None})
        return signal

    # ---------------------------------------------------------------- state

    def already_fired(self, ticker: str, signal: Dict[str, Any]) -> bool:
        """One alert per ticker per level per day. A level that breaks at 10:00 is not a
        fresh setup at 10:05, and re-firing it is how an alerter teaches you to mute it."""
        key = f"{signal['level_timeframe']}:{signal['level_trigger']}:{signal['direction']}"
        fired = self._fired_today.setdefault(ticker, set())
        if key in fired:
            return True
        fired.add(key)
        return False

    def reset_daily_state(self) -> None:
        self._fired_today.clear()
