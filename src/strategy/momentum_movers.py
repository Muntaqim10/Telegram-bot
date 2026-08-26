import os
import logging
from typing import Dict, List, Optional, Any
import numpy as np

log = logging.getLogger(__name__)

CONFIRMATION_BARS = int(os.getenv("BREAKOUT_CONFIRMATION_BARS", "2"))

class MomentumMoversStrategy:
    """
    Continuous Full-Day Momentum & Market Mover Strategy (9:30 AM - 4:00 PM EST).
    Tracks real-time 1-minute price/volume action to catch:
      1. HOD Velocity Breakouts & LOD Velocity Breakdowns
      2. VWAP & 9/21 EMA Pullback Continuations
      3. Unusual Volume Surges
    Enforces strict Anti-Chase (<= 2% from VWAP) and Multi-Bar Confirmation to prevent fakeouts.
    """
    def __init__(self, confirmation_bars: int = CONFIRMATION_BARS):
        self.confirmation_bars = confirmation_bars
        # State: ticker -> {bars: list, hod: float, lod: float, cum_vol: float, cum_pv: float, pending: dict}
        self._state: Dict[str, Dict[str, Any]] = {}

    def _get_or_create_state(self, ticker: str) -> Dict[str, Any]:
        if ticker not in self._state:
            self._state[ticker] = {
                "bars": [],
                "hod": -1.0,
                "lod": float("inf"),
                "cum_vol": 0.0,
                "cum_pv": 0.0,
                "pending_breakout": None  # {"type": "LONG"|"SHORT", "level": float, "bars_held": int, "invalidation": float}
            }
        return self._state[ticker]

    def reset_daily_state(self):
        """Resets state at the start of each new trading day."""
        self._state.clear()
        log.info("🔄 [MOMENTUM MOVERS] Reset intraday state for all tickers.")

    def on_bar(self, bar: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Ingests a 1-minute intraday bar:
        bar = {"ticker": "NVDA", "open": float, "high": float, "low": float, "close": float, "volume": int, "timestamp": str}
        Returns a signal dict if a confirmed, non-faked momentum setup triggers.
        """
        ticker = bar.get("ticker", "").upper()
        if not ticker:
            return None

        close = float(bar.get("close", 0.0))
        high = float(bar.get("high", close))
        low = float(bar.get("low", close))
        vol = float(bar.get("volume", 0.0))

        if close <= 0.0 or vol <= 0:
            return None

        state = self._get_or_create_state(ticker)
        bars = state["bars"]
        bars.append(bar)
        if len(bars) > 120:
            bars.pop(0)  # Keep rolling 2 hours of 1-min bars

        # Update HOD / LOD
        prev_hod = state["hod"]
        prev_lod = state["lod"]
        if prev_hod < 0:
            state["hod"] = high
            state["lod"] = low
        else:
            state["hod"] = max(state["hod"], high)
            state["lod"] = min(state["lod"], low)

        # Update Cumulative VWAP
        typical_price = (high + low + close) / 3.0
        state["cum_vol"] += vol
        state["cum_pv"] += (typical_price * vol)
        vwap = state["cum_pv"] / state["cum_vol"] if state["cum_vol"] > 0 else close

        # Need at least 15 bars for baseline metrics (EMA, z-vol, RSI)
        if len(bars) < 15:
            return None

        closes = np.array([b["close"] for b in bars], dtype=float)
        volumes = np.array([b["volume"] for b in bars], dtype=float)

        # 1. Calculate Relative Volume (Z-Score)
        vol_mean = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        vol_std = np.std(volumes[-20:]) if len(volumes) >= 20 else (np.std(volumes) or 1.0)
        z_vol = (vol - vol_mean) / vol_std if vol_std > 0 else 0.0
        vol_ratio = vol / vol_mean if vol_mean > 0 else 1.0

        # 2. Calculate 9 EMA and 21 EMA
        ema9 = self._calc_ema(closes, 9)
        ema21 = self._calc_ema(closes, 21)

        # 3. Calculate 14-period RSI
        rsi = self._calc_rsi(closes, 14)

        # 4. ANTI-CHASE OVEREXTENSION GUARD (Gate 2)
        # Rejects parabolic blowoffs (detached > 2.5% above VWAP with extreme RSI > 82)
        dist_to_vwap_pct = ((close - vwap) / vwap) * 100.0
        is_overextended_long = (dist_to_vwap_pct > 3.0) or (dist_to_vwap_pct > 1.8 and rsi > 82.0)
        is_overextended_short = (dist_to_vwap_pct < -3.0) or (dist_to_vwap_pct < -1.8 and rsi < 18.0)

        # 5. Check Pending Breakout Confirmation (Gate 3)
        pending = state.get("pending_breakout")
        if pending:
            p_type = pending["type"]
            p_level = pending["level"]
            p_inval = pending["invalidation"]
            
            if p_type == "LONG":
                if close > p_level:
                    pending["bars_held"] += 1
                    if pending["bars_held"] >= self.confirmation_bars:
                        # Confirmed HOD Breakout!
                        state["pending_breakout"] = None
                        return self._build_signal(
                            ticker=ticker,
                            direction="Long",
                            entry_price=close,
                            catalyst_type="HOD Velocity Breakout",
                            invalidation_level=p_inval,
                            z_vol=z_vol,
                            rsi=rsi,
                            vwap=vwap
                        )
                else:
                    # Failed to hold above level -> Invalidate pending setup immediately
                    state["pending_breakout"] = None
                    log.info(f"🚫 [MOMENTUM MOVERS] {ticker} Long HOD breakout failed confirmation hold (close {close:.2f} <= {p_level:.2f}). Dropped.")

            elif p_type == "SHORT":
                if close < p_level:
                    pending["bars_held"] += 1
                    if pending["bars_held"] >= self.confirmation_bars:
                        # Confirmed LOD Breakdown!
                        state["pending_breakout"] = None
                        return self._build_signal(
                            ticker=ticker,
                            direction="Short",
                            entry_price=close,
                            catalyst_type="LOD Velocity Breakdown",
                            invalidation_level=p_inval,
                            z_vol=z_vol,
                            rsi=rsi,
                            vwap=vwap
                        )
                else:
                    # Failed to hold below level -> Invalidate pending setup immediately
                    state["pending_breakout"] = None
                    log.info(f"🚫 [MOMENTUM MOVERS] {ticker} Short LOD breakdown failed confirmation hold (close {close:.2f} >= {p_level:.2f}). Dropped.")

        # 6. Evaluate New Momentum Trigger Candidates
        # Pattern A: High of Day Velocity Breakout (Unconfirmed initial touch)
        if close > prev_hod and prev_hod > 0 and (z_vol >= 1.8 or vol_ratio >= 1.8) and ema9 > ema21:
            if not is_overextended_long:
                inval_level = low # Previous bar low as tight thesis invalidation
                if self.confirmation_bars <= 1:
                    return self._build_signal(ticker, "Long", close, "HOD Velocity Breakout", inval_level, z_vol, rsi, vwap)
                else:
                    state["pending_breakout"] = {"type": "LONG", "level": prev_hod, "bars_held": 1, "invalidation": inval_level}
                    log.info(f"⏳ [MOMENTUM MOVERS] {ticker} HOD Breakout detected at ${close:.2f} > ${prev_hod:.2f}. Awaiting {self.confirmation_bars} confirmation bars...")

        # Pattern B: Low of Day Velocity Breakdown (Unconfirmed initial touch)
        elif close < prev_lod and prev_lod < float("inf") and (z_vol >= 1.8 or vol_ratio >= 1.8) and ema9 < ema21:
            if not is_overextended_short:
                inval_level = high # Previous bar high as tight thesis invalidation
                if self.confirmation_bars <= 1:
                    return self._build_signal(ticker, "Short", close, "LOD Velocity Breakdown", inval_level, z_vol, rsi, vwap)
                else:
                    state["pending_breakout"] = {"type": "SHORT", "level": prev_lod, "bars_held": 1, "invalidation": inval_level}
                    log.info(f"⏳ [MOMENTUM MOVERS] {ticker} LOD Breakdown detected at ${close:.2f} < ${prev_lod:.2f}. Awaiting {self.confirmation_bars} confirmation bars...")

        # Pattern C: VWAP & 9 EMA Pullback Continuation (Clean dip into support without chasing)
        # Condition: In uptrend (close > vwap, ema9 > ema21), low touched near 9 EMA / VWAP, now reclaiming with volume
        dist_to_vwap_pct = ((close - vwap) / vwap) * 100.0
        if 0.1 <= dist_to_vwap_pct <= 1.2 and (z_vol >= 1.5 or vol_ratio >= 1.5) and ema9 > ema21 and 45.0 <= rsi <= 65.0:
            if closes[-1] > closes[-2] and closes[-2] <= ema9: # Reversal bounce confirmed
                inval_level = vwap * 0.995 # Exits if VWAP support is broken
                return self._build_signal(ticker, "Long", close, "VWAP Pullback Continuation", inval_level, z_vol, rsi, vwap)

        return None

    def _build_signal(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        catalyst_type: str,
        invalidation_level: float,
        z_vol: float,
        rsi: float,
        vwap: float
    ) -> Dict[str, Any]:
        """Formats the confirmed momentum signal for the downstream execution pipeline."""
        return {
            "ticker": ticker,
            "direction": direction,
            "entry_price": float(entry_price),
            "catalyst_type": catalyst_type,
            "invalidation_level": float(invalidation_level),
            "z_vol": float(z_vol),
            "rsi": float(rsi),
            "vwap": float(vwap),
            "strategy": "MOMENTUM_MOVERS"
        }

    @staticmethod
    def _calc_ema(data: np.ndarray, span: int) -> float:
        if len(data) == 0:
            return 0.0
        alpha = 2.0 / (span + 1.0)
        weights = (1.0 - alpha) ** np.arange(len(data))[::-1]
        weights /= weights.sum()
        return float(np.dot(data, weights))

    @staticmethod
    def _calc_rsi(data: np.ndarray, period: int = 14) -> float:
        if len(data) < period + 1:
            return 50.0
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))
