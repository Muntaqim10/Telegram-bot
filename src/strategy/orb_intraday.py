import os
import logging
import pandas as pd
from typing import Dict, Optional

log = logging.getLogger(__name__)

class ORBStrategy:
    """
    Advanced Intraday Strategy tracking ORBs, VWAP, and EMAs.
    Features:
    1. Morning ORB Breakouts
    2. Mid-Day Trend Continuation (10:30-11:30)
    3. Dynamic VWAP Breakdowns (All day)
    """
    def __init__(self, orb_minutes: int = 15):
        self.orb_minutes = orb_minutes
        self.quiet_seconds = int(os.getenv("QUIET_MODE_SECONDS", "900"))
        self.opening_ranges = {} # Dict[ticker, Dict[str, float]]
        self.active_signals = {}
        self.intraday_state = {} # Dict[ticker, dict]
        
    def _get_market_session(self, timestamp: pd.Timestamp) -> str:
        """Determines market session: PREMARKET, RTH (Regular Trading Hours), or POSTMARKET."""
        hour = timestamp.hour
        minute = timestamp.minute
        if (hour < 9) or (hour == 9 and minute < 30):
            return "PREMARKET"
        elif (hour == 9 and minute >= 30) or (10 <= hour < 16):
            return "RTH"
        else:
            return "POSTMARKET"

    def process_tick(self, ticker: str, price: float, volume: float, timestamp: pd.Timestamp) -> Optional[Dict]:
        """
        Processes a real-time price tick.
        Maintains Pre-Market, Regular Trading Hours (RTH), and Post-Market states.
        """
        try:
            price = float(price)
            volume = float(volume) if volume is not None else 0.0
        except (ValueError, TypeError):
            return None

        session = self._get_market_session(timestamp)
        
        # Maintain overall state structure (4-Timeframe Confluence Engine)
        state = self.intraday_state.setdefault(ticker, {
            "cum_pv": 0.0, "cum_vol": 0.0, "vwap": price,
            "ema9": price, "ema21": price, 
            "ema9_5m": price, "ema21_5m": price, "last_5m_bar": timestamp.minute // 5,
            "last_minute": timestamp.minute,
            "last_price": price,
            "pm_high": price, "pm_low": price, "pm_vol": 0,
            "post_high": price, "post_low": price
        })
        state["last_price"] = price
        
        # 1. Pre-Market Tick Handling (4:00 AM - 9:30 AM EST) [Timeframe 4: Premarket / Daily Level]
        if session == "PREMARKET":
            state["pm_high"] = max(state["pm_high"], price)
            state["pm_low"] = min(state["pm_low"], price)
            state["pm_vol"] += volume
            return None # Do not dispatch options alerts during pre-market
            
        # 2. Post-Market Tick Handling (4:00 PM - 8:00 PM EST)
        elif session == "POSTMARKET":
            state["post_high"] = max(state["post_high"], price)
            state["post_low"] = min(state["post_low"], price)
            return None # Do not dispatch options alerts post-market

        # 3. Regular Trading Hours (RTH: 9:30 AM - 4:00 PM EST)
        # Update VWAP
        state["cum_vol"] += volume
        state["cum_pv"] += price * volume
        if state["cum_vol"] > 0:
            state["vwap"] = state["cum_pv"] / state["cum_vol"]
            
        # Timeframe 1 (1-Minute Execution): Update 1m EMAs
        if timestamp.minute != state["last_minute"]:
            alpha9 = 2 / (9 + 1)
            alpha21 = 2 / (21 + 1)
            state["ema9"] = (price * alpha9) + (state["ema9"] * (1 - alpha9))
            state["ema21"] = (price * alpha21) + (state["ema21"] * (1 - alpha21))
            state["last_minute"] = timestamp.minute

        # Timeframe 2 (5-Minute Structure): Update 5m EMAs every 5 minutes
        current_5m_bar = timestamp.minute // 5
        if current_5m_bar != state.get("last_5m_bar"):
            alpha9_5m = 2 / (9 + 1)
            alpha21_5m = 2 / (21 + 1)
            state["ema9_5m"] = (price * alpha9_5m) + (state["ema9_5m"] * (1 - alpha9_5m))
            state["ema21_5m"] = (price * alpha21_5m) + (state["ema21_5m"] * (1 - alpha21_5m))
            state["last_5m_bar"] = current_5m_bar
            
        # Maintain Session High (HOD), Low (LOD), and Closing Range Tracking
        state["hod"] = max(state.get("hod", price), price)
        state["lod"] = min(state.get("lod", price), price)

        # Track Closing Range (3:45 PM - 4:00 PM EST Previous Day & Afternoon Session)
        is_closing_range_window = (timestamp.hour == 15) and (timestamp.minute >= 45)
        state["crb_high"] = max(state.get("crb_high", price), price)
        state["crb_low"] = min(state.get("crb_low", price), price)

        # Common indicator export across timeframes
        vwap_ratio = price / state["vwap"] if state["vwap"] > 0 else 1.0
        ema9_ratio = price / state["ema9"] if state["ema9"] > 0 else 1.0
        ema_trend_1m_bullish = 1 if state["ema9"] > state["ema21"] else 0
        ema_trend_5m_bullish = 1 if state["ema9_5m"] > state["ema21_5m"] else 0
        hod_ratio = price / state["hod"] if state["hod"] > 0 else 1.0
        lod_ratio = price / state["lod"] if state["lod"] > 0 else 1.0

        # Opening Range vs. Closing Range Intraday Breakout (CRB) Engine
        is_in_orb_window = (timestamp.hour == 9) and (timestamp.minute < (30 + self.orb_minutes))
        
        if ticker not in self.opening_ranges:
            self.opening_ranges[ticker] = {"high": price, "low": price, "orb_established": False}
            
        orb = self.opening_ranges[ticker]
        
        if is_in_orb_window:
            if price > orb["high"]: orb["high"] = price
            if price < orb["low"]: orb["low"] = price
            return None
            
        else:
            if not orb.get("orb_established"):
                orb["orb_established"] = True
                
            # =======================
            # CLOSING RANGE & INTRADAY BREAKOUT EVALUATION (CRB)
            # =======================
            last_sig_info = self.active_signals.get(ticker)
            if last_sig_info and isinstance(last_sig_info, dict):
                last_time = last_sig_info.get("time")
                if last_time and (timestamp - last_time).total_seconds() < self.quiet_seconds:
                    return None # Per-ticker quiet period cool-off window
                
            signal_data = {
                "ticker": ticker,
                "entry_price": price,
                "timestamp": timestamp,
                "vwap_ratio": vwap_ratio,
                "ema9_ratio": ema9_ratio,
                "ema_trend_bullish": ema_trend_1m_bullish,
                "ema_trend_5m_bullish": ema_trend_5m_bullish,
                "hod": state["hod"],
                "lod": state["lod"],
                "pm_high": state.get("pm_high", price),
                "pm_low": state.get("pm_low", price),
                "crb_high": state.get("crb_high", price),
                "crb_low": state.get("crb_low", price),
                "hod_ratio": hod_ratio,
                "lod_ratio": lod_ratio,
                "volume": volume,
                "tf_confluence": "1m+5m+CRB+Daily"
            }

            # VWAP Overextension Check (+/- 2.5% from VWAP)
            is_overextended_upside = vwap_ratio > 1.025
            is_overextended_downside = vwap_ratio < 0.975

            # 1. Call Setups: Dual ORB + CRB Breakouts & Breakdown Fakeout Reversals
            if not is_overextended_upside:
                # Highest Conviction: Dual ORB + CRB Breakout (Price > ORB High AND Price > CRB High)
                if price > orb["high"] and price >= state.get("crb_high", price) and price >= state["vwap"] and ema_trend_5m_bullish == 1:
                    self.active_signals[ticker] = {"direction": "Long", "time": timestamp}
                    signal_data.update({"direction": "Long", "catalyst_type": "🔥 DUAL ORB+CRB CALL BREAKOUT", "tf_confluence": "1m+5m+ORB+CRB+Daily"})
                    return signal_data

                # Reversal Entry: Failed Breakdown Reclaim (Price was below ORB Low, now reclaims VWAP & 5m 9-EMA)
                if price <= orb["low"] * 1.005 and price >= state["vwap"] and ema_trend_5m_bullish == 1:
                    self.active_signals[ticker] = {"direction": "Long", "time": timestamp}
                    signal_data.update({"direction": "Long", "catalyst_type": "🔄 FAKEOUT BREAKDOWN CALL REVERSAL", "tf_confluence": "Reversal+5m+VWAP"})
                    return signal_data

                if price > orb["high"] and price >= state["vwap"] and ema_trend_5m_bullish == 1:
                    self.active_signals[ticker] = {"direction": "Long", "time": timestamp}
                    signal_data.update({"direction": "Long", "catalyst_type": "Opening Range Call Breakout (ORB)"})
                    return signal_data

                if price >= state.get("crb_high", price) and price >= state["vwap"] and ema_trend_5m_bullish == 1:
                    self.active_signals[ticker] = {"direction": "Long", "time": timestamp}
                    signal_data.update({"direction": "Long", "catalyst_type": "Closing Range Call Breakout (CRB)"})
                    return signal_data

                if vwap_ratio >= 1.002 and ema9_ratio >= 1.001 and ema_trend_1m_bullish == 1 and ema_trend_5m_bullish == 1:
                    self.active_signals[ticker] = {"direction": "Long", "time": timestamp}
                    signal_data.update({"direction": "Long", "catalyst_type": "Bullish 5m+1m Trend Continuation"})
                    return signal_data

            # 2. Put Setups: Dual ORB + CRB Breakdowns & Breakout Fakeout Reversals
            if not is_overextended_downside:
                # Highest Conviction: Dual ORB + CRB Breakdown (Price < ORB Low AND Price < CRB Low)
                if price < orb["low"] and price <= state.get("crb_low", price) and price <= state["vwap"] and ema_trend_5m_bullish == 0:
                    self.active_signals[ticker] = {"direction": "Short", "time": timestamp}
                    signal_data.update({"direction": "Short", "catalyst_type": "🔥 DUAL ORB+CRB PUT BREAKDOWN", "tf_confluence": "1m+5m+ORB+CRB+Daily"})
                    return signal_data

                # Reversal Entry: Failed Breakout Reversal (Price pushed above ORB High, lost VWAP & 5m 9-EMA)
                if price >= orb["high"] * 0.995 and price <= state["vwap"] and ema_trend_5m_bullish == 0:
                    self.active_signals[ticker] = {"direction": "Short", "time": timestamp}
                    signal_data.update({"direction": "Short", "catalyst_type": "🔄 FAKEOUT BREAKOUT PUT REVERSAL", "tf_confluence": "Reversal+5m+VWAP"})
                    return signal_data

                if price < orb["low"] and price <= state["vwap"] and ema_trend_5m_bullish == 0:
                    self.active_signals[ticker] = {"direction": "Short", "time": timestamp}
                    signal_data.update({"direction": "Short", "catalyst_type": "Opening Range Put Breakdown (ORB)"})
                    return signal_data

                if price <= state.get("crb_low", price) and price <= state["vwap"] and ema_trend_5m_bullish == 0:
                    self.active_signals[ticker] = {"direction": "Short", "time": timestamp}
                    signal_data.update({"direction": "Short", "catalyst_type": "Closing Range Put Breakdown (CRB)"})
                    return signal_data

                if vwap_ratio <= 0.998 and ema9_ratio <= 0.999 and ema_trend_1m_bullish == 0 and ema_trend_5m_bullish == 0:
                    self.active_signals[ticker] = {"direction": "Short", "time": timestamp}
                    signal_data.update({"direction": "Short", "catalyst_type": "Bearish 5m+1m Trend Breakdown"})
                    return signal_data
                
        return None
        
    def reset_daily(self):
        """Clears the intraday memory for the next trading day."""
        self.opening_ranges.clear()
        self.active_signals.clear()
        self.intraday_state.clear()
        log.debug("ORB & VWAP strategy reset for the new day.")
