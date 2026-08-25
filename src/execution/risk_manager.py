import logging
import pandas as pd
import numpy as np
import asyncio
from src.database import close_trade as db_close_trade

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
        self.pending_evaluations = set()
        
    def mark_pending(self, ticker: str) -> bool:
        """Atomically mark a ticker as currently evaluating to prevent double-entry race conditions."""
        if ticker in self.active_positions or ticker in self.pending_evaluations:
            return False
        self.pending_evaluations.add(ticker)
        return True
        
    def clear_pending(self, ticker: str):
        """Release the evaluation lock for a ticker."""
        self.pending_evaluations.discard(ticker)

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
        true_range = ranges.max(axis=1)
        
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
        outcome = None
        
        if direction == "Long":
            # Check if Take Profit hit
            if current_price >= take_profit:
                profit_pct = (current_price - entry_price) / entry_price
                outcome = {"status": "TP_HIT", "exit_price": current_price, "pnl": profit_pct}

            # If price moves up, raise the stop
            elif current_price <= pos["trailing_stop"]:
                profit_pct = (current_price - entry_price) / entry_price
                outcome = {"status": "STOPPED_OUT", "exit_price": current_price, "pnl": profit_pct}
            else:
                current_profit_pct = (current_price - entry_price) / entry_price
                if atr_distance > 0 and current_profit_pct >= self.min_profit_pct:
                    new_stop = current_price - atr_distance
                    if new_stop > current_stop:
                        pos["trailing_stop"] = new_stop
                
        elif direction == "Short":
            # Check if Take Profit hit
            if current_price <= take_profit:
                profit_pct = (entry_price - current_price) / entry_price
                outcome = {"status": "TP_HIT", "exit_price": current_price, "pnl": profit_pct}

            # Check if stopped out
            elif current_price >= pos["trailing_stop"]:
                profit_pct = (entry_price - current_price) / entry_price
                outcome = {"status": "STOPPED_OUT", "exit_price": current_price, "pnl": profit_pct}
            else:
                # If price moves down, lower the stop
                current_profit_pct = (entry_price - current_price) / entry_price
                if atr_distance > 0 and current_profit_pct >= self.min_profit_pct:
                    new_stop = current_price + atr_distance
                    if new_stop < current_stop or current_stop == 0:
                        pos["trailing_stop"] = new_stop

        if outcome is not None:
            # Linear delta approximation (ESTIMATE ONLY - delta-only, no theta/gamma adjustment, no live option quote)
            # This estimates the option's return using entry delta as a linear approximation since we don't have a continuous live option tick feed.
            opt_entry = pos.get("option_entry_price")
            opt_delta = pos.get("option_entry_delta")
            if opt_entry and opt_entry > 0 and opt_delta is not None and opt_delta != 0:
                stock_move = current_price - entry_price
                estimated_opt_move = stock_move * abs(opt_delta)
                if direction == "Short":
                    estimated_opt_move = -stock_move * abs(opt_delta)
                estimated_opt_price = max(0.01, opt_entry + estimated_opt_move)
                outcome["estimated_option_pnl_pct"] = (estimated_opt_price - opt_entry) / opt_entry
            else:
                outcome["estimated_option_pnl_pct"] = None  # no option data available, don't fabricate a number
            return outcome
                
        return {"status": "ACTIVE", "current_stop": pos["trailing_stop"]}

    def attach_option_pricing(self, ticker: str, option_entry_price: float, option_entry_delta: float, option_entry_theta: float) -> None:
        """Attaches real option entry data to an already-registered position."""
        if ticker in self.active_positions:
            self.active_positions[ticker]["option_entry_price"] = option_entry_price
            self.active_positions[ticker]["option_entry_delta"] = option_entry_delta
            self.active_positions[ticker]["option_entry_theta"] = option_entry_theta

    def close_trade(self, ticker: str, outcome_status: str, exit_price: float, pnl_pct: float, estimated_option_pnl_pct: float = None):
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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_exists = os.path.exists(file_path)
        
        opt_pnl_str = f"{estimated_option_pnl_pct:.4f}" if estimated_option_pnl_pct is not None else ""
        
        try:
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "ticker", "direction", "entry_price", "exit_price", "outcome", "pnl_pct", "estimated_option_pnl_pct"])
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ticker,
                    pos["direction"],
                    f"{pos['entry_price']:.2f}",
                    f"{exit_price:.2f}",
                    outcome_status,
                    f"{pnl_pct:.4f}",
                    opt_pnl_str
                ])
            opt_log = f", Option PnL Est: {estimated_option_pnl_pct*100:.1f}%" if estimated_option_pnl_pct is not None else ""
            log.info(f"[{ticker}] TRADE CLOSED: {outcome_status}. Stock PnL: {pnl_pct*100:.2f}%{opt_log} (Exit: ${exit_price:.2f})")
        except Exception as e:
            log.error(f"Failed to write trade outcome for {ticker}: {e}")

        # Archive to SQLite database
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(db_close_trade(ticker, exit_price, outcome_status))
        except RuntimeError:
            log.warning(f"[{ticker}] Not running in an async event loop, database archive skipped.")
        except Exception as e:
            log.warning(f"Failed to schedule database archive for {ticker}: {e}")

    def add_position(
        self, 
        ticker: str, 
        entry_price: float, 
        initial_atr: float, 
        direction: str, 
        stop_loss: float = None, 
        take_profit: float = None,
        option_entry_price: float = None,
        option_entry_delta: float = None,
        option_entry_theta: float = None
    ) -> dict:
        """
        Registers a new intraday runner position.
        Returns the computed (or provided) stop_loss and take_profit levels.
        """
        atr_distance = initial_atr * self.atr_multiplier
        
        if stop_loss is not None:
            initial_stop = stop_loss
        else:
            initial_stop = entry_price - atr_distance if direction == "Long" else entry_price + atr_distance
            
        if take_profit is not None:
            tp = take_profit
        else:
            from src.utils.math_utils import calculate_take_profit
            tp = calculate_take_profit(entry_price, initial_atr, direction)
        
        self.active_positions[ticker] = {
            "entry_price": entry_price,          # keep as-is (stock price, still useful context)
            "trailing_stop": initial_stop,
            "take_profit": tp,
            "direction": direction,
            "option_entry_price": option_entry_price,
            "option_entry_delta": option_entry_delta,
            "option_entry_theta": option_entry_theta,
        }
        log.info(f"Position added for {ticker} ({direction}) at {entry_price}. Stop: {initial_stop:.2f}, TP: {tp:.2f}")
        return {"stop_loss": initial_stop, "take_profit": tp}
        
    def reset_daily(self):
        """Clears active positions to prevent stale runners from leaking into the next day."""
        self.active_positions.clear()

