import os
import logging
import pandas as pd
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

class ExtendedHoursScanner:
    """
    Detects significant price/volume moves during premarket and postmarket sessions.
    Fires informational alerts, and queues priority RTH evaluation on the next open.
    """
    def __init__(self):
        self.move_pct_threshold = float(os.getenv("EXTENDED_HOURS_MOVE_PCT", "5.0"))
        self.min_vol_threshold = float(os.getenv("EXTENDED_HOURS_MIN_VOL", "100000"))
        
        # ticker -> { "prev_close": float, "cum_vol": float, "last_price": float, "alert_sent_today": bool }
        self.ticker_state = {}
        # Date string -> set of tickers that fired an extended hours alert
        self.triggered_tickers = {}
        
    def _get_market_session(self, timestamp: pd.Timestamp) -> str:
        hour = timestamp.hour
        minute = timestamp.minute
        if (hour < 9) or (hour == 9 and minute < 30):
            return "PREMARKET"
        elif (hour == 9 and minute >= 30) or (10 <= hour < 16):
            return "RTH"
        else:
            return "POSTMARKET"

    def process_event(self, event: Dict[str, Any], current_timestamp: pd.Timestamp):
        """
        Synchronously processes a raw tick or quote event to track extended hours movers.
        """
        ticker = event.get("symbol")
        raw_price = event.get("price") or event.get("last") or event.get("bid")
        raw_vol = event.get("size", 0)
        
        # Tradier summary events have prevClose
        if event.get("type") == "summary":
            if "prevClose" in event and ticker:
                if ticker not in self.ticker_state:
                    self.ticker_state[ticker] = {
                        "prev_close": float(event["prevClose"]), 
                        "cum_vol": 0.0, 
                        "last_price": float(event["prevClose"]), 
                        "alert_sent_today": False
                    }
                else:
                    self.ticker_state[ticker]["prev_close"] = float(event["prevClose"])
            return

        if not ticker or not raw_price:
            return
            
        try:
            price = float(raw_price)
            volume = float(raw_vol)
        except (ValueError, TypeError):
            return

        session = self._get_market_session(current_timestamp)

        state = self.ticker_state.setdefault(ticker, {
            "prev_close": price,
            "cum_vol": 0.0,
            "last_price": price,
            "alert_sent_today": False
        })
        
        if session == "RTH":
            # During RTH, we update prev_close so at 4:00 PM it natively carries over as the close
            state["prev_close"] = price
            state["cum_vol"] = 0.0
            state["last_price"] = price
            state["alert_sent_today"] = False
            return
            
        # Extended Hours Accumulation
        state["cum_vol"] += volume
        state["last_price"] = price

    def evaluate_movers(self, current_timestamp: pd.Timestamp) -> list:
        """
        Periodically polled by the async loop to yield alerts.
        """
        session = self._get_market_session(current_timestamp)
        if session == "RTH":
            return [] # We only alert during extended hours

        today_str = current_timestamp.strftime("%Y-%m-%d")
        alerts = []

        for ticker, state in self.ticker_state.items():
            if state["prev_close"] <= 0:
                continue
                
            pct_change = ((state["last_price"] - state["prev_close"]) / state["prev_close"]) * 100.0
            
            if abs(pct_change) >= self.move_pct_threshold and state["cum_vol"] >= self.min_vol_threshold:
                if not state["alert_sent_today"]:
                    state["alert_sent_today"] = True
                    
                    if today_str not in self.triggered_tickers:
                        self.triggered_tickers[today_str] = set()
                    self.triggered_tickers[today_str].add(ticker)
                    
                    direction = "Long" if pct_change > 0 else "Short"
                    log.info(f"🌙 EXTENDED HOURS MOVER: {ticker} moved {pct_change:.2f}% (Vol: {state['cum_vol']})")
                    
                    alerts.append({
                        "ticker": ticker,
                        "direction": direction,
                        "pct_change": pct_change,
                        "price": state["last_price"],
                        "volume": state["cum_vol"],
                        "session": session,
                        "prev_close": state["prev_close"]
                    })
        return alerts
        
    def consume_priority_signal(self, ticker: str, current_timestamp: pd.Timestamp) -> Optional[Dict]:
        """
        Called on a ticker's first RTH tick to construct a synthetic priority signal if it fired overnight.
        """
        session = self._get_market_session(current_timestamp)
        if session != "RTH":
            return None
            
        today_str = current_timestamp.strftime("%Y-%m-%d")
        if today_str in self.triggered_tickers and ticker in self.triggered_tickers[today_str]:
            self.triggered_tickers[today_str].remove(ticker)
            
            state = self.ticker_state.get(ticker, {})
            price = state.get("last_price", 0.0)
            prev_close = state.get("prev_close", price)
            
            # For the synthetic signal, we assume direction based on the gap
            direction = "Long" if price >= prev_close else "Short"
            
            return {
                "ticker": ticker,
                "direction": direction,
                "price": price,
                "timestamp": current_timestamp,
                "volume": state.get("cum_vol", 0),
                "strategy": "EXTENDED_GAP"
            }
        return None
