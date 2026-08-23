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
        Updates the trailing stop for an active position and checks for Take-Profit hits.
        """
        if ticker not in self.active_positions:
            return {"status": "NO_POSITION"}
            
        pos = self.active_positions[ticker]
        entry_price = pos["entry_price"]
        current_stop = pos["trailing_stop"]
        take_profit = pos["take_profit"]
        
        atr_distance = current_atr * self.atr_multiplier if current_atr > 0 else 0
        
        if direction == "Long":
            # Check if Take Profit hit
            if current_price >= take_profit:
                profit_pct = (current_price - entry_price) / entry_price
                return {"status": "TP_HIT", "exit_price": current_price, "pnl": profit_pct}

            # If price moves up, raise the stop
            if atr_distance > 0:
                new_stop = current_price - atr_distance
                if new_stop > current_stop:
                    pos["trailing_stop"] = new_stop
                
            # Check if stopped out
            if current_price <= pos["trailing_stop"]:
                profit_pct = (current_price - entry_price) / entry_price
                return {"status": "STOPPED_OUT", "exit_price": current_price, "pnl": profit_pct}
                
        elif direction == "Short":
            # Check if Take Profit hit
            if current_price <= take_profit:
                profit_pct = (entry_price - current_price) / entry_price
                return {"status": "TP_HIT", "exit_price": current_price, "pnl": profit_pct}

            # If price moves down, lower the stop
            if atr_distance > 0:
                new_stop = current_price + atr_distance
                if new_stop < current_stop or current_stop == 0:
                    pos["trailing_stop"] = new_stop
                
            # Check if stopped out
            if current_price >= pos["trailing_stop"]:
                profit_pct = (entry_price - current_price) / entry_price
                return {"status": "STOPPED_OUT", "exit_price": current_price, "pnl": profit_pct}
                
        return {"status": "ACTIVE", "current_stop": pos["trailing_stop"]}

    def close_trade(self, ticker: str, outcome_status: str, exit_price: float, pnl_pct: float):
        """
        Removes the active position and logs the final outcome to CSV for ML training.
        """
        if ticker not in self.active_positions:
            return

        pos = self.active_positions.pop(ticker)
        
        # Log to file
        import os
        import csv
        from datetime import datetime
        
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/trade_outcomes.csv"))
        file_exists = os.path.exists(file_path)
        
        try:
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "ticker", "direction", "entry_price", "exit_price", "outcome", "pnl_pct"])
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ticker,
                    pos["direction"],
                    f"{pos['entry_price']:.2f}",
                    f"{exit_price:.2f}",
                    outcome_status,
                    f"{pnl_pct:.4f}"
                ])
            log.info(f"[{ticker}] TRADE CLOSED: {outcome_status}. PnL: {pnl_pct*100:.2f}% (Exit: ${exit_price:.2f})")
        except Exception as e:
            log.error(f"Failed to write trade outcome for {ticker}: {e}")

    def add_position(self, ticker: str, entry_price: float, initial_atr: float, direction: str, stop_loss: float = None, take_profit: float = None) -> dict:
        """
        Registers a new intraday runner position.
        Returns the computed (or provided) stop_loss and take_profit levels.
        """
        if stop_loss is not None and take_profit is not None:
            initial_stop = stop_loss
            tp = take_profit
        else:
            atr_distance = initial_atr * self.atr_multiplier
            if direction == "Long":
                initial_stop = entry_price - atr_distance
                tp = entry_price + (atr_distance * 2.0)  # 2:1 reward-to-risk
            else:
                initial_stop = entry_price + atr_distance
                tp = entry_price - (atr_distance * 2.0)
        
        self.active_positions[ticker] = {
            "entry_price": entry_price,
            "trailing_stop": initial_stop,
            "take_profit": tp,
            "direction": direction
        }
        log.info(f"Position added for {ticker} ({direction}) at {entry_price}. Stop: {initial_stop:.2f}, TP: {tp:.2f}")
        return {"stop_loss": initial_stop, "take_profit": tp}
