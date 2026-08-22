import logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

class RiskManager:
    """
    Handles intraday risk management, specifically focusing on trailing stops
    that give >5% momentum runners room to breathe during normal intraday pullbacks.
    """
    def __init__(self, atr_multiplier: float = 2.5, min_profit_pct: float = 0.10):
        self.atr_multiplier = atr_multiplier
        self.min_profit_pct = min_profit_pct
        self.active_positions = {}

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculates the Average True Range (ATR).
        Requires columns: 'high', 'low', 'close'.
        Ideally executed on a 5-minute timeframe for trend runners.
        """
        if len(df) < period + 1:
            return 0.0
            
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        atr = true_range.rolling(period).mean().iloc[-1]
        return atr

    def update_trailing_stop(self, ticker: str, current_price: float, current_atr: float, direction: str) -> dict:
        """
        Updates the trailing stop for an active position.
        """
        if ticker not in self.active_positions:
            return {"status": "NO_POSITION"}
            
        pos = self.active_positions[ticker]
        entry_price = pos["entry_price"]
        current_stop = pos["trailing_stop"]
        
        atr_distance = current_atr * self.atr_multiplier
        
        if direction == "Long":
            # If price moves up, raise the stop
            new_stop = current_price - atr_distance
            if new_stop > current_stop:
                pos["trailing_stop"] = new_stop
                
            # Check if stopped out
            if current_price <= pos["trailing_stop"]:
                profit_pct = (current_price - entry_price) / entry_price
                return {"status": "STOPPED_OUT", "exit_price": current_price, "pnl": profit_pct}
                
        elif direction == "Short":
            # If price moves down, lower the stop
            new_stop = current_price + atr_distance
            if new_stop < current_stop or current_stop == 0:
                pos["trailing_stop"] = new_stop
                
            # Check if stopped out
            if current_price >= pos["trailing_stop"]:
                profit_pct = (entry_price - current_price) / entry_price
                return {"status": "STOPPED_OUT", "exit_price": current_price, "pnl": profit_pct}
                
        return {"status": "ACTIVE", "current_stop": pos["trailing_stop"]}

    def add_position(self, ticker: str, entry_price: float, initial_atr: float, direction: str):
        """
        Registers a new intraday runner position.
        """
        initial_stop = entry_price - (initial_atr * self.atr_multiplier) if direction == "Long" else entry_price + (initial_atr * self.atr_multiplier)
        
        self.active_positions[ticker] = {
            "entry_price": entry_price,
            "trailing_stop": initial_stop,
            "direction": direction
        }
        log.info(f"Position added for {ticker} ({direction}) at {entry_price}. Initial Stop: {initial_stop}")
